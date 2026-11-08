//! Speculation in the Nemotron backend: the drafters behind `Backend::propose_batch_scored` (the MTP head, the copy drafter, nothing),
//! the arbiter that picks among them per round, the per-slot record of what each round offered, and the telemetry (per source and per
//! draft depth) the measurements read. The GLM backend's `families/glm/spec.rs` is the model of this file and is not edited.
//!
//! The contract is the engine's: a drafter only PROPOSES. The verify window decides every committed token (`accept` = the keyed sample
//! equals the draft), so whatever this file proposes, drafted == serial, byte for byte. What it can change is speed, and the one thing
//! it must not do is disturb the trunk's state: a draft call touches only the head's own cache and scratch
//! ([`NemotronRuntime::mtp_draft`]).
//!
//! A round, as the backend sees it: `propose` (called by the engine just before the window, for the streams that may draft) returns each
//! stream's drafts and remembers what it offered for the slot; `on_step` learns how many of them the engine put in the window;
//! `on_commit` learns how many were accepted, updates the telemetry and, in `auto`, the arbiter's estimates. A slot's record is consumed
//! by its commit; a sequence that ends never commits and its slot is cleared by `end`.
//!
//! **Who decides.** `--spec mtp` drafts at a fixed depth and `--spec copy` offers what the copy drafter finds; both leave the engine's
//! own row budget in charge (`engine::assemble`: the step table trims an offer whose estimated acceptance does not pay for its rows).
//! `--spec auto` gives the decision to [`gb_spec::arbiter::Arbiter`]: per stream, none, copy or the head at a depth, by committed tokens
//! per model-millisecond with this model's costs ([`NemotronCost`]), and the engine's budget is made flat so that it does not
//! second-guess it. `--spec none` never reaches this file (the engine does not draft).
//!
//! **One head call per round.** Every stream the arbiter wants head drafts for is asked in ONE `mtp_draft` call (the head reads its
//! weights once for all of them). A stream whose store of pending rows would overflow after another window is asked with depth 0: the
//! call absorbs its rows and drafts nothing for it ("absorb-only"), so a slot that the arbiter keeps on the copy drafter or on serial
//! rows never loses its head to a full store.
//!
//! **A head that fails costs nothing but the head.** A recoverable error from `mtp_draft` switches drafting off for the slots of that
//! call (`mtp_disable_slot`, and the slot's own flag so it is not asked again), logs one `event=mtp_error` line, and the round goes on
//! with the copy offer or none: the trunk was not touched. A fatal one (a sticky device error, lost state) is returned to the backend,
//! which latches it like every other fatal error.

use super::backend::{NemotronRuntime, WINDOW_ROWS};
use super::mtp::{DraftAsk, MTP_MAX_DEPTH};
use crate::error::ModelError;
use gb_engine::{Proposal, ProposeRequest};
use gb_sampling::Sampling;
use gb_sched::CostModel;
use gb_spec::arbiter::{Arbiter, ArbiterConfig, ArbiterCost, Ask, Choice};
use gb_spec::{CopyDrafter, DraftPick, SpecMode};
use std::borrow::Cow;
use std::time::Instant;

/// The depth `--spec mtp` drafts at, and the deepest chain `--spec auto` may offer (`REXL3_MTP_DEPTH` overrides it). Three drafts per
/// round: the GB10 evidence in `docs/NEMOTRON_MTP_RESEARCH.md` is that depth 3 beat 7 (about 108 against 80-93 tokens a second) and
/// that NVIDIA's Spark recipes use 3; it is a cap, so a stream whose acceptance falls is held lower by the arbiter.
pub const MTP_DEFAULT_DEPTH: usize = 3;
/// Most drafts the copy drafter offers in a round (TensorFold's windows: enter on 8 matching tokens, offer up to 7).
pub const COPY_MAX: usize = 7;
const COPY_MIN_MATCH: usize = 8;
/// Depths the per-depth telemetry keeps.
const DEPTHS: usize = 8;
/// Rows of one slot's store of pending tapped rows (`[32][4096]` f32): the runtime's `TapBook` switches a slot's drafting off, SILENTLY,
/// when a commit would take it past this (`n + keep > 32`, so a single commit of more than 32 rows kills a slot whatever it held). The
/// round's commit keeps at most the base row and the drafts the engine was asked for (`1 + ProposeRequest::max`), so a stream whose store
/// has less room than that must absorb BEFORE its window runs (`must_absorb`, an absorb-only ask of depth 0). The memory model prices the
/// store with this same number.
pub const MTP_TAP_ROWS: usize = super::mtp::MTP_TAP_CAP;

/// Where a round's drafts came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// No drafts: one serial row.
    None,
    Copy,
    Mtp,
}

impl Source {
    fn idx(self) -> usize {
        match self {
            Source::None => 0,
            Source::Copy => 1,
            Source::Mtp => 2,
        }
    }
    fn of(c: Choice) -> Self {
        match c {
            Choice::None => Source::None,
            Choice::Copy(_) => Source::Copy,
            Choice::Mtp(_) => Source::Mtp,
        }
    }
}

/// What a backend speculates with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecConfig {
    pub mode: SpecMode,
    /// Drafts per round of `--spec mtp`, and the deepest chain of `--spec auto`: `1..=MTP_MAX_DEPTH`.
    pub mtp_depth: usize,
    /// The head drafts over a restricted vocabulary (a sub-head table of the `lm_head`, built at load: the default 49,152-id list unless
    /// `REXL3_MTP_VOCAB` says `off` or names another file): the arbiter prices a draft call with the restricted table of its
    /// [`CostTables`] and starts from the restricted head's acceptance priors (`engine::spec_settings` sets it from the sub-table the
    /// plan priced, and makes the table from the list's id count with [`MtpCost::restricted_to`] unless it is the measured size). The
    /// runtime seam does not carry it: this selects the cost table and priors only, and the model uses the restricted head whenever it
    /// built one.
    pub restricted_head: bool,
    /// How the head proposes for a SAMPLED request: its argmax, or the target's own keyed draw over its candidates at each draft's
    /// position (keyed-noise drafting). A greedy request is the argmax either way and is never handed a sampler.
    /// `REXL3_NEMO_DRAFT_NOISE=0|1`.
    pub noise: DraftPick,
    /// Rows of the model's decode window (all streams): the arbiter never plans a round past it. [`WINDOW_ROWS`] unless the model is wide
    /// (`NemotronBackend::with_window_rows` sets it).
    pub window_rows: usize,
    /// `--spec auto` in a WIDE round ([`WIDE_ROUND_STREAMS`] streams or more): every stream whose head is ready drafts this deep and the
    /// arbiter is not asked (under [`ArbTable::Single`] its step table is a one-sequence table, which misprices a step of many sequences:
    /// it puts the base rows far under their cost and the draft rows above theirs relative to them, [`VERIFY_BATCHED`]). 0: the arbiter
    /// decides, wide or not. `REXL3_NEMO_WIDE_DEPTH`.
    pub wide_depth: usize,
    /// The verify table `--spec auto`'s arbiter prices a round with: [`VERIFY_MS`] by total rows ([`ArbTable::Single`], the default and
    /// the v0.1.0 behaviour) or [`VERIFY_BATCHED`] by sequences and rows ([`ArbTable::Batched`]). `REXL3_NEMO_ARB_TABLE=single|batched`.
    pub arb_table: ArbTable,
}

/// Keyed-noise drafting unless `REXL3_NEMO_DRAFT_NOISE=0` (the same default as the GLM drafter, whose live A/B is in `docs/GLM53.md`).
pub const DRAFT_NOISE_DEFAULT: DraftPick = DraftPick::Keyed;

/// Which verify table the arbiter of `--spec auto` prices a round with (`REXL3_NEMO_ARB_TABLE`). It changes which drafts are proposed and
/// never a committed token: the verify window decides every token whatever was proposed (drafted == serial).
///
/// It is read by rank 0's scheduler only. A tensor-parallel follower builds no speculation runtime (`tp_boot::follow`): the drafts and
/// their depths reach it as the arguments of rank 0's calls, so the ranks need not agree on it and the Hello's facts (`tp_runtime::NemoFacts`)
/// do not carry it, like `REXL3_MTP_DEPTH` and `REXL3_NEMO_DRAFT_NOISE` (`docs/NEMOTRON_TP.md` 5.4.2). A two-rank group has no measured
/// batched table: `batched` there prices with the group's single table ([`CostTables::tp_two_ranks`], logged at boot).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArbTable {
    /// [`VERIFY_MS`] by the round's total rows, whatever the sequences they belong to.
    Single,
    /// [`VERIFY_BATCHED`]: by the round's sequences and its rows.
    Batched,
}

impl ArbTable {
    /// `single` or `batched`: the `arb_table=` of the `draft_config` line.
    pub fn label(self) -> &'static str {
        match self {
            ArbTable::Single => "single",
            ArbTable::Batched => "batched",
        }
    }
}

/// The table unless `REXL3_NEMO_ARB_TABLE` says otherwise: the one-sequence table of v0.1.0 until the owner's A/B decides.
pub const ARB_TABLE_DEFAULT: ArbTable = ArbTable::Single;

/// The environment variable names of the settings.
pub const DEPTH_VAR: &str = "REXL3_MTP_DEPTH";
pub const NOISE_VAR: &str = "REXL3_NEMO_DRAFT_NOISE";
pub const WIDE_DEPTH_VAR: &str = "REXL3_NEMO_WIDE_DEPTH";
pub const ARB_TABLE_VAR: &str = "REXL3_NEMO_ARB_TABLE";

impl SpecConfig {
    /// Both `--spec mtp` and the cap of `--spec auto` are [`MTP_DEFAULT_DEPTH`]: with the shipped tables, raising auto's cap LOSES tok/s on
    /// the eight bench prompts (cap 4 -44.5, cap 5 -44.4, cap 6 -44.1 against 45.3 at cap 3, `docs/NEMOTRON_OVERNIGHT_2026-10-11.md`),
    /// because the tables price a chain step and a verify row below what the served rounds pay. `REXL3_MTP_DEPTH=4..7` raises it.
    pub fn new(mode: SpecMode) -> Self {
        Self {
            mode,
            mtp_depth: MTP_DEFAULT_DEPTH,
            restricted_head: false,
            noise: DRAFT_NOISE_DEFAULT,
            window_rows: WINDOW_ROWS,
            wide_depth: 0,
            arb_table: ARB_TABLE_DEFAULT,
        }
    }

    /// `new`, with the settings given as the raw text of the three variables (unset is `None`): `REXL3_MTP_DEPTH`,
    /// `REXL3_NEMO_DRAFT_NOISE` and `REXL3_NEMO_ARB_TABLE`. A malformed value is an error: a measurement must not silently run on the
    /// default.
    pub fn from_vars(mode: SpecMode, depth: Option<&str>, noise: Option<&str>, table: Option<&str>) -> Result<Self, ModelError> {
        let mut c = Self::new(mode);
        if let Some(v) = depth {
            c.mtp_depth = parse_depth(v).map_err(ModelError::Config)?;
        }
        c.noise = parse_noise(noise).map_err(ModelError::Config)?;
        c.arb_table = parse_arb_table(table).map_err(ModelError::Config)?;
        Ok(c)
    }

    /// `from_vars` over the process environment, with the effective settings logged once.
    pub fn from_env(mode: SpecMode) -> Result<Self, ModelError> {
        let depth = std::env::var(DEPTH_VAR).ok();
        let noise = std::env::var(NOISE_VAR).ok();
        let table = std::env::var(ARB_TABLE_VAR).ok();
        let mut c = Self::from_vars(mode, depth.as_deref(), noise.as_deref(), table.as_deref())?;
        if let Ok(v) = std::env::var(WIDE_DEPTH_VAR) {
            let d: usize = v.trim().parse().map_err(|_| ModelError::Config(format!("{WIDE_DEPTH_VAR}={v} is not a number")))?;
            if d > MTP_MAX_DEPTH {
                return Err(ModelError::Config(format!("{WIDE_DEPTH_VAR} must be 0..={MTP_MAX_DEPTH}, got {d}")));
            }
            c.wide_depth = d;
        }
        eprintln!(
            "rexl3 event=draft_config mode={} mtp_depth={} pick={} wide_depth={} arb_table={} ({DEPTH_VAR}=1..={MTP_MAX_DEPTH}, {NOISE_VAR}=0|1, {WIDE_DEPTH_VAR}=0..={MTP_MAX_DEPTH}, {ARB_TABLE_VAR}=single|batched)",
            c.mode,
            c.mtp_depth,
            if c.noise.is_keyed() { "keyed" } else { "argmax" },
            c.wide_depth,
            c.arb_table.label()
        );
        Ok(c)
    }
}

/// `REXL3_NEMO_ARB_TABLE`: `single` or `batched` (surrounding blanks ignored, any case); unset is [`ARB_TABLE_DEFAULT`]; anything else,
/// the empty value included, is an error.
pub fn parse_arb_table(v: Option<&str>) -> Result<ArbTable, String> {
    let Some(v) = v else { return Ok(ARB_TABLE_DEFAULT) };
    match v.trim().to_ascii_lowercase().as_str() {
        "single" => Ok(ArbTable::Single),
        "batched" => Ok(ArbTable::Batched),
        _ => Err(format!("{ARB_TABLE_VAR}={v:?}: expected single or batched")),
    }
}

/// `REXL3_NEMO_DRAFT_NOISE`: `0` (the head's argmax) or `1` (the target's keyed draw over the head's candidates; the words of
/// `DraftPick::from_setting` are accepted too); unset is [`DRAFT_NOISE_DEFAULT`]; anything else is an error.
pub fn parse_noise(v: Option<&str>) -> Result<DraftPick, String> {
    DraftPick::from_setting(v, NOISE_VAR, DRAFT_NOISE_DEFAULT)
}

/// `REXL3_MTP_DEPTH`: a number in `1..=MTP_MAX_DEPTH`.
pub fn parse_depth(v: &str) -> Result<usize, String> {
    let d: usize = v.trim().parse().map_err(|_| format!("{DEPTH_VAR}={v} is not a number"))?;
    if d == 0 || d > MTP_MAX_DEPTH {
        return Err(format!("{DEPTH_VAR} must be 1..={MTP_MAX_DEPTH}, got {d}"));
    }
    Ok(d)
}

/// The sampler each ask of a draft call carries: the request's own when keyed drafting is on AND the request is sampled (temperature
/// above 0), none otherwise (the head's argmax, which is also what a greedy request is: it has no noise to key).
pub fn keyed_samplers(noise: DraftPick, samplings: &[&Sampling]) -> Vec<Option<Sampling>> {
    samplings.iter().map(|s| (noise.is_keyed() && s.temperature > 0.0).then_some(**s)).collect()
}

// ------------------------------------------------------------------------------------------------------------------------ costs

/// The verify window of this model by total rows, in milliseconds, with the commit that follows it: the step table the arbiter prices a
/// round with under `REXL3_NEMO_ARB_TABLE=single` (the default; `batched` prices with [`VERIFY_BATCHED`]), and the engine's row-budget curve for `--spec copy` (`engine::STEP_MS` is this table). **MEASURED** on the real 88-layer
/// model with the fast MoE kernels and the decode-window graphs on, 2026-10-11, node 1, the final kernels. One and two rows, and sixteen
/// and more, are `tests/nemotron_omni_costs.rs` (one sequence at a 256-token context, `rows` rows of real text committed in full and
/// flushed, the median of 7, two runs averaged: 38 / 46 / 79 / 110 / 202 ms). Three to eight rows are SERVED rounds (`--spec mtp` at depth
/// 2 to 7 over the eight bench prompts, `tools/nemotron_step_bench.sh`: the verify phase 49.7 / 55.6 / 61.8 / 65.7 / 70.2 / 74.9 ms plus
/// the commit's 1.5): the test's single oracle text shares more experts among its rows than drafts of free text do, so it prices a row past
/// the fourth 2 to 6 ms below what a round pays, and the depth decision lives on exactly those rows. Concave (the experts the rows share
/// are read once). The earlier tables were 37 / 46 / 52 / 57 / 65 / 69 / 80 / 111 / 196 ms (2026-10-10), 269 ms for one row and about 100
/// ms for each row after it (the I3 harness run, before the MoE work) and 40 / 74 / 201 ms (the fast MoE kernels without the window
/// graphs). The step will fall again as the perf workers land: regenerate it, hand it to [`SpecRuntime::set_costs`] or replace the
/// constant, no other change; the arbiter's online scale (`NemotronCost::observe_window`, clamped to 0.7..1.6) follows the table meanwhile.
pub const VERIFY_MS: &[(usize, f64)] = &[
    (1, 38.0),
    (2, 46.0),
    (3, 51.0),
    (4, 57.0),
    (5, 63.0),
    (6, 67.0),
    (7, 71.0),
    (8, 75.0),
    (16, 84.0),
    (32, 112.0),
    (64, 202.0),
];

/// The routed experts one row picks in every MoE layer, as a fraction of them: `num_experts_per_tok / n_routed_experts` of the pack's
/// `config.json`, 22 of 512. The architecture's, not fitted ([`VERIFY_BATCHED`]).
pub const P_BASE_ROUTED: f64 = 22.0 / 512.0;

/// The round of `n` sequences over `R` rows in all as `REXL3_NEMO_ARB_TABLE=batched` prices it, in milliseconds: the verify step by its
/// sequences and its rows, the commits that follow it, and what the head's draft call costs beyond [`MtpCost`] when it serves several
/// streams. [`VERIFY_MS`] is a table of ONE sequence's rows.
///
///   `window(n, R) = floor + experts * (1 - (1 - p_base)^n * (1 - p_draft)^(R - n)) + per_draft_row * (R - n)`
///   `ms(n, R) = window(n, R) + commit_per_stream * n`,   draft call of `s >= 2` streams `d` deep: `+ s * (head_per_stream +
///   head_per_stream_step * (d - 1))`
///
/// The window is the expert union of the step and what a draft row costs beyond it: each sequence's base row picks a fraction `p_base` of
/// the routed experts of every MoE layer (22 of 512), a draft row picks new ones with the smaller `p_draft` (it continues its own sequence's
/// text and shares experts with its base row), `experts` is the time to stream every routed expert once, `per_draft_row` is a draft row's
/// own work (its rows of the dense projections, the attention and the per-row state records) and `floor` the rest of the step. A draft row
/// is priced at its MARGINAL cost in a step of `n` sequences: the k-th adds `experts * p_draft * (1 - p_base)^n * (1 - p_draft)^(k - 1) +
/// per_draft_row`, which falls as the sequences have read more of the experts. `commit_per_stream` is the commit of every sequence, which
/// replays its kept rows into every Mamba layer's state: enqueued at `commit` and waited for by the next sync, it is paid by every round
/// whatever is drafted, so it is in the base of every pick. The head terms are what a draft call of several streams costs beyond the draft
/// model fitted at 1 to 4 asks (its absorb and its chained steps over the bf16 MoE of the head, a union of its own).
///
/// **A round of fewer than `min_streams` sequences (one) is priced by the single table**, as `single` prices it, to the bit: the one-sequence
/// table was measured on exactly that round, and the union fit, whose weight is in the wide rounds, prices a serial one-stream round 3.4 ms
/// over its measured 36.8 ms, which makes a deeper draft look cheaper than it is (on the filler prompts of the R-TABLE A/B, acceptance 0.81
/// / 0.72 / 0.66, the batched table of the first A/B offered a third draft in 34 % of the one-stream rounds against 24 % under single, and
/// lost 2.0 to 3.4 % of one-stream ITL at 128 to 8,192 prompt tokens). From two sequences on, the batched table.
///
/// **MEASURED** (2026-10-16, the shipped v0.1.0 binary `cluster-bin-rc2final-7cc3bf3c` and the A/B binary 5bd049fb (the same kernels), the
/// qualified profile at 16 slots; `tools/nemotron_arb_table.py`, provenance in `docs/NEMOTRON_MTP_PLAN.md` "The batched verify table"):
/// 665 points of 65 phase dumps on nodes 1 and 4 (`/metrics` differenced around each phase per round width; 90,452 rounds): the perf
/// client's one-stream latency phase (prompts of 128 to 32,768 tokens), its concurrency ladder (1 to 32 streams of short prompts) and its
/// 16-stream bench; `--spec auto` at caps 2 to 5 under both tables, greedy and sampled, `--spec mtp` at depths 2 and 3 and `--spec none`.
/// The window: verify-phase rms 1.8 ms over the 665 points; pooled by width over the drafting rounds, rms 1.0 ms and none off by more than
/// 2.1 ms, from 52 ms (one stream, 3.4 rows) to 213 ms (16 streams, 35 rows). At 16 streams the whole round (verify, draft, commit) of the
/// model against the fixed-depth runs: serial 177.8 / 177.7 ms measured, one deep 247.7 / 246.7, two 299.1 / 297.7, three 342.9 / 342.6.
/// `per_draft_row` 0.51 ms and `p_draft` 0.0206 are what pins the 64-row rounds that the first table (no per-row term) priced 13 ms under.
/// `commit_per_stream` 1.02 ms is identified by the `--spec none` rounds, whose verify step waits for the commits (no draft call does):
/// the first table's 1.77 ms held the head's own excess too. The head terms: the pinned rounds (every stream the same depth: 16 streams
/// one deep, 2 to 16 two deep, 8 to 16 three deep; 24,000 rounds) less the draft model and the commits. `p_base` is the architecture's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchedVerify {
    /// Milliseconds of the step that do not depend on the experts read, nor on the draft rows.
    pub floor: f64,
    /// Milliseconds to read every routed expert once.
    pub experts: f64,
    /// The fraction of the routed experts a sequence's base row picks, and the fraction of new ones a draft row picks.
    pub p_base: f64,
    pub p_draft: f64,
    /// Milliseconds of a draft row's own work beyond the experts it reads.
    pub per_draft_row: f64,
    /// Milliseconds of one sequence's commit (its kept rows replayed into its recurrent state).
    pub commit_per_stream: f64,
    /// Milliseconds a draft call of several streams costs beyond [`MtpCost::ms`]: per stream, and per stream and chained step.
    pub head_per_stream: f64,
    pub head_per_stream_step: f64,
    /// Rounds of fewer sequences, and draft calls of fewer streams, are priced as under the single table.
    pub min_streams: usize,
}

/// The batched table of this model (see [`BatchedVerify`]), fitted by `tools/nemotron_arb_table.py`.
pub const VERIFY_BATCHED: BatchedVerify = BatchedVerify {
    floor: 27.76,
    experts: 264.83,
    p_base: P_BASE_ROUTED,
    p_draft: 0.0206,
    per_draft_row: 0.509,
    commit_per_stream: 1.020,
    head_per_stream: 0.738,
    head_per_stream_step: 0.280,
    min_streams: 2,
};

impl BatchedVerify {
    /// The window of `streams` sequences over `rows` rows (each sequence holds at least its base row: a count of streams above the rows is
    /// read as the rows).
    pub fn window_ms(&self, streams: usize, rows: usize) -> f64 {
        let n = streams.min(rows);
        let untouched = (1.0 - self.p_base).powi(n as i32) * (1.0 - self.p_draft).powi((rows - n) as i32);
        self.floor + self.experts * (1.0 - untouched) + self.per_draft_row * (rows - n) as f64
    }

    /// The commits of `streams` sequences.
    pub fn commit_ms(&self, streams: usize) -> f64 {
        self.commit_per_stream * streams as f64
    }

    /// The round's verify step and the commits that follow it.
    pub fn ms(&self, streams: usize, rows: usize) -> f64 {
        self.window_ms(streams, rows) + self.commit_ms(streams.min(rows))
    }

    /// What a draft call for `streams` streams chaining `depth` drafts (0: absorb only) costs beyond [`MtpCost::ms`].
    pub fn head_extra_ms(&self, streams: usize, depth: usize) -> f64 {
        let s = streams as f64;
        s * self.head_per_stream + s * self.head_per_stream_step * depth.saturating_sub(1) as f64
    }

    /// Whether this table prices a round of `streams` sequences (or a draft call of that many streams).
    pub fn prices(&self, streams: usize) -> bool {
        streams >= self.min_streams
    }

    /// Every number finite and non-negative, the fractions in `(0, 1)`, and a round of two sequences or more priced by it.
    pub fn validate(&self, what: &str) -> Result<(), String> {
        let times = [
            self.floor,
            self.experts,
            self.per_draft_row,
            self.commit_per_stream,
            self.head_per_stream,
            self.head_per_stream_step,
        ];
        if times.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(format!("{what}: a time is negative or not finite"));
        }
        if [self.p_base, self.p_draft].iter().any(|p| !p.is_finite() || *p <= 0.0 || *p >= 1.0) {
            return Err(format!("{what}: an expert fraction is outside (0, 1)"));
        }
        if self.min_streams == 0 {
            return Err(format!("{what}: min_streams must be at least 1"));
        }
        Ok(())
    }
}

/// The draft call's time as a function of (streams, absorbed rows, depth). The terms, which the fit below puts numbers on:
/// * **absorb**: a launch-and-sync `base` and `per_stream` a stream, plus the head's two blocks (`eh_proj`, attention, the latent MoE) over
///   all the pending rows of all the streams, `rows`: a table by total rows, because the routed experts the rows pick are read once
///   whichever row picked them;
/// * **head**: the `lm_head` and the pick over one row a stream, once for the first draft: `lm_head` (it reads the head's 1.07 GB once
///   however many streams ask, so it is the one term a restricted vocabulary shrinks);
/// * **chain**: each further draft is one head row a stream through the two blocks and a head again: `lm_head + step + step_per_stream`
///   a stream.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpCost {
    pub base: f64,
    pub per_stream: f64,
    /// (pending rows in all, milliseconds of the two blocks over them); zero rows cost nothing; past the table the last slope goes on.
    pub rows: Cow<'static, [(usize, f64)]>,
    /// The `lm_head` GEMV and the pick of one step, whatever the streams.
    pub lm_head: f64,
    /// The two blocks over one chained row, and what each further stream's row adds (its own routed experts).
    pub step: f64,
    pub step_per_stream: f64,
}

/// Rows of the trunk's `lm_head`, the full draft vocabulary.
pub const LM_HEAD_ROWS: usize = 131_072;
/// Rows of the restricted head the measured restricted table is for: the default draft vocabulary (`mtp_vocab_48k.txt`, 49,152 ids).
pub const LM_HEAD_RESTRICTED_ROWS: usize = 49_152;

impl MtpCost {
    fn rows_ms(&self, rows: usize) -> f64 {
        let t: &[(usize, f64)] = &self.rows;
        if rows == 0 || t.is_empty() {
            return 0.0;
        }
        let (mut r0, mut t0) = (0usize, 0.0f64);
        for &(r1, t1) in t {
            if rows <= r1 {
                return t0 + (t1 - t0) * (rows - r0) as f64 / (r1 - r0) as f64;
            }
            (r0, t0) = (r1, t1);
        }
        // past the table: the last segment's slope
        let n = t.len();
        let ((ra, ta), (rb, tb)) = if n >= 2 { (t[n - 2], t[n - 1]) } else { ((0, 0.0), t[n - 1]) };
        tb + (tb - ta) / (rb - ra) as f64 * (rows - rb) as f64
    }

    /// Milliseconds of a call for `streams` streams absorbing `rows` pending rows in all and drafting `depth` deep (`depth == 0`: only
    /// absorbing).
    pub fn ms(&self, streams: usize, rows: usize, depth: usize) -> f64 {
        let s = streams as f64;
        let mut t = self.base + self.per_stream * s + self.rows_ms(rows);
        if depth >= 1 {
            t += self.lm_head;
        }
        if depth >= 2 {
            t += (depth - 1) as f64 * (self.lm_head + self.step + self.step_per_stream * s);
        }
        t
    }

    /// This model with the head over a restricted vocabulary of `rows` token ids: only the `lm_head` term moves. Linear between the two
    /// measured points (the full 131,072 rows and the 49,152-row default head: [`MTP_COST`] and [`MTP_COST_RESTRICTED`]); the other
    /// sizes are interpolated, not measured separately (the 32,768-row head, measured once, falls on the line to 0.05 ms).
    pub fn restricted_to(&self, rows: usize) -> MtpCost {
        let rows = rows.clamp(1, LM_HEAD_ROWS);
        let (full, small) = (MTP_COST.lm_head, MTP_COST_RESTRICTED.lm_head);
        let frac = (rows as f64 - LM_HEAD_RESTRICTED_ROWS as f64) / (LM_HEAD_ROWS - LM_HEAD_RESTRICTED_ROWS) as f64;
        // the ratio to the full head keeps a caller's own measured full-head `lm_head` as the scale
        let lm = (small + frac * (full - small)).max(0.0) / full * self.lm_head;
        MtpCost { lm_head: lm, ..self.clone() }
    }

    /// [`MtpCost::restricted_to`] for a head whose restricted sub-table is bf16 rows whatever the trunk's `lm_head` is (the MixedK form
    /// decodes its EXL3 rows to bf16 at load, D2), from THIS restricted model (measured at the default 49,152 ids): the `lm_head` term
    /// moves along the same bf16 line, scaled by this model's own default-size term, so an EXL3 full `lm_head` never prices a bf16
    /// sub-table. At [`LM_HEAD_RESTRICTED_ROWS`] it is this model, bit for bit.
    pub fn restricted_sub_table_to(&self, rows: usize) -> MtpCost {
        let rows = rows.clamp(1, LM_HEAD_ROWS);
        if rows == LM_HEAD_RESTRICTED_ROWS {
            return self.clone();
        }
        let (full, small) = (MTP_COST.lm_head, MTP_COST_RESTRICTED.lm_head);
        let frac = (rows as f64 - LM_HEAD_RESTRICTED_ROWS as f64) / (LM_HEAD_ROWS - LM_HEAD_RESTRICTED_ROWS) as f64;
        let lm = (small + frac * (full - small)).max(0.0) / small * self.lm_head;
        MtpCost { lm_head: lm, ..self.clone() }
    }

    /// This model with every term scaled by the ratio of the same term of `num` to `den` (a term whose `den` is zero is kept): each
    /// scalar, and each point of the absorb table by `num`'s over `den`'s absorb at that many rows. What the two-rank tables of a 4-bit
    /// head are made of ([`CostTables::tp_two_ranks_for`]): the measured bf16 group tables carried over by the single-node 4-bit / bf16
    /// ratios. An EXTRAPOLATION, never a measurement. `num == den` returns this model unchanged, bit for bit.
    pub fn scaled_per_term(&self, num: &MtpCost, den: &MtpCost) -> MtpCost {
        let r = |x: f64, a: f64, b: f64| if b > 0.0 && a != b { x * (a / b) } else { x };
        let rows: Vec<(usize, f64)> = self.rows.iter().map(|&(n, t)| (n, r(t, num.rows_ms(n), den.rows_ms(n)))).collect();
        MtpCost {
            base: r(self.base, num.base, den.base),
            per_stream: r(self.per_stream, num.per_stream, den.per_stream),
            rows: Cow::Owned(rows),
            lm_head: r(self.lm_head, num.lm_head, den.lm_head),
            step: r(self.step, num.step, den.step),
            step_per_stream: r(self.step_per_stream, num.step_per_stream, den.step_per_stream),
        }
    }

    /// The table must be ascending in rows from at least one row, and every number finite and non-negative.
    pub fn validate(&self, what: &str) -> Result<(), String> {
        let scalars = [self.base, self.per_stream, self.lm_head, self.step, self.step_per_stream];
        if scalars.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(format!("{what}: a cost is negative or not finite"));
        }
        if self.rows.is_empty() {
            return Err(format!("{what}: the absorb table has no points"));
        }
        if self.rows[0].0 == 0 || self.rows.windows(2).any(|w| w[0].0 >= w[1].0) {
            return Err(format!("{what}: the absorb table must be strictly ascending in rows from 1"));
        }
        if self.rows.iter().any(|p| !p.1.is_finite() || p.1 < 0.0) {
            return Err(format!("{what}: an absorb time is negative or not finite"));
        }
        Ok(())
    }
}

// MEASURED (`tests/nemotron_omni_costs.rs`, node 1, 2026-10-11, the final kernels): the wall time of `mtp_draft` (absorb + chain + one
// sync), median of 7 after 2 warm-ups, a 256-token context, real taps of the oracle's text. Asks x depth, pending rows PER ASK 1 / 8:
//   full head         1 ask  7.2 / 7.1   14.1 / 14.3  21.8 / 20.9  27.7 / 29.1  34.6 / 34.7  (depth 1 .. 5)
//                     4 asks 8.8 / 9.0   18.1 / 17.5  26.9 / 25.4  34.9 / 34.0  43.5 / 42.5
//   48k restricted    1 ask  4.2 / 4.1    8.1 / 8.2   12.2 / 12.3  16.2 / 16.3  20.3 / 20.3
//                     4 asks 5.8 / 6.0   12.0 / 11.0  18.1 / 16.3  23.3 / 21.8  29.2 / 28.2
// (the 48k head is the default one: `mtp_vocab_48k.txt`, 49,152 ids.) The stage profile of 1 ask x 1 row x depth 1: lm_head 4.63 (48k
// restricted 1.74), MoE block 1.58, attention 0.38, eh_proj 0.35, rest 0.05.
// The draft call changed at 7db8258: an absorb builds every pending row's keys and values (eh_proj and the key and value projections read
// their weights once whatever the rows) and runs the attention, the MoE block and the final norm over each sequence's LAST row only. A
// call no longer costs by the pending rows (8 rows read the same as 1: the cells above) but by its sequences. The tables before it
// (2026-10-10) were 7.1 / 10.9 (full) and 4.2 / 8.0 (restricted) at 1 ask x depth 1 and 1 / 8 rows.
// The head is bf16 and runs the original bf16 MoE path: the step's fast 4-bit kernels do not touch it.
// The fit (`fit_flat` in the test: the chained step from the increments between depths, a line through the asks; the absorb by least
// squares in the streams and the square root of the pending rows, which the data says is flat within noise: `rows` keeps a 0.01 ms
// term so the table stays ascending and concave as the arbiter's shape tests require; `lm_head` the stage-measured time, base 0.10):
//   T(s streams, R pending rows in all, d depth) = base + per_stream s + rows(R) + lm_head + (d - 1)(lm_head + step + step_per_stream s)
// Error against the 40 cells of each table: full head at most 6.6%, mean 1.9%; restricted at most 7.0%, mean 2.9%.
// `tests::the_fitted_tables_reproduce_the_measured_draft_calls` holds both. Beyond 32 total pending rows the last slope goes on.
// To regenerate after the step changes again: `tests/nemotron_omni_costs.rs` prints both in this format (REXL3_MTP_VOCAB=off for the full).

/// The draft-call model with the trunk's full `lm_head` (131,072 rows): fitted to the measurements above.
pub const MTP_COST: MtpCost = MtpCost {
    base: 0.10,
    per_stream: 0.60,
    rows: Cow::Borrowed(&[(1, 2.12), (2, 2.13), (3, 2.13), (4, 2.13), (8, 2.14), (16, 2.15), (24, 2.16), (32, 2.17)]),
    lm_head: 4.63,
    step: 1.83,
    step_per_stream: 0.52,
};

/// The same with the head over the default restricted vocabulary of 49,152 ids (its own measurement and fit: the `lm_head` 1.74 ms
/// instead of 4.63, the rest within noise of the full head's). [`MtpCost::restricted_to`] gives the other sizes.
pub const MTP_COST_RESTRICTED: MtpCost = MtpCost {
    base: 0.10,
    per_stream: 0.57,
    rows: Cow::Borrowed(&[(1, 1.82), (2, 1.83), (3, 1.83), (4, 1.83), (8, 1.84), (16, 1.85), (24, 1.86), (32, 1.87)]),
    lm_head: 1.74,
    step: 1.73,
    step_per_stream: 0.54,
};

// ---------------------------------------------------------------------------------------------------------------- the 4-bit head (R-HEAD4)

// The draft call of a pack whose head projections are EXL3 (the 4-bit head: the MoE block's 1,028 projections stored as 4-bit trellis,
// `bridge::take_mtp`; `eh_proj` kept bf16 by the flagged-matrix policy, the attention, the norms, the router and the restricted `lm_head`
// sub-table bf16). The cost tables are keyed by the LOADED head's form ([`HeadForm`], `NemotronRuntime::head_form`): one binary serves
// both packs, and a bf16 head gets [`MTP_COST`] / [`MTP_COST_RESTRICTED`] exactly.
//
// MEASURED (`tests/nemotron_omni_costs.rs` `print_the_cost_tables_in_the_format_of_the_constants`, N4 anonymized host-d, 2026-10-16 16:05 to 16:08
// CDT, test binary @1914de49, ref-text-1k, the 4-bit pack `hf-nemotron-main-h4`, index eb0f1ada..., back to back with the bf16 pack in
// the same session): the wall time of `mtp_draft`, median of 7 after 2 warm-ups, asks 1..4 x depth 1..5 x pending rows per ask 1 / 8 (40
// cells per table, `tests::MEASURED_H4` / `MEASURED_RESTRICTED_H4`), fitted by the test's own `fit_flat` (base 0.10, `lm_head` the
// stage-measured time), printed and pasted verbatim:
//   restricted (REXL3_MTP_VOCAB default, 49,152 ids): worst 5.1%, mean 1.7% (5.5% / 1.7% with the constants' two decimals)
//   full lm_head (REXL3_MTP_VOCAB=off):               worst 3.8%, mean 1.3% (3.8% / 1.4%)
// Stage profile of 1 ask x 1 row x depth 1: moe block 0.42 ms (bf16 head, same session: 1.57 restricted / 1.66 full), lm_head 1.75
// restricted / 4.67 full (bf16: 1.75 / 4.59: unchanged, the sub-table is bf16), eh_proj 0.37 / 0.35 (bf16 here too), attention 0.40.
// Against the same session's bf16 fits (per_stream / rows(1) / lm_head / step / step_per_stream):
//   restricted: bf16 0.62 / 1.79 / 1.75 / 1.71 / 0.56  ->  4-bit 0.25 / 0.96 / 1.75 / 0.93 / 0.25
//   full:       bf16 0.69 / 1.73 / 4.59 / 1.93 / 0.55  ->  4-bit 0.31 / 0.92 / 4.67 / 0.85 / 0.31
// The same-session bf16 fits sit within ~9% of the shipped [`MTP_COST`] / [`MTP_COST_RESTRICTED`] (2026-10-11, node 1), which stay as
// they are; the 4-bit head's chained step and per-stream terms are about half of them, far outside the fit's ~7% band.

/// The draft-call model of a 4-bit head with the trunk's full `lm_head` (131,072 rows): the measurement above, `REXL3_MTP_VOCAB=off`.
pub const MTP_COST_H4: MtpCost = MtpCost {
    base: 0.10,
    per_stream: 0.31,
    rows: Cow::Borrowed(&[(1, 0.92), (2, 0.93), (3, 0.93), (4, 0.94), (8, 0.95), (16, 0.96), (24, 0.98), (32, 0.99)]),
    lm_head: 4.67,
    step: 0.85,
    step_per_stream: 0.31,
};

/// The draft-call model of a 4-bit head over the default restricted vocabulary (49,152 ids): the measurement above.
/// [`MtpCost::restricted_to`] gives the other sizes.
pub const MTP_COST_RESTRICTED_H4: MtpCost = MtpCost {
    base: 0.10,
    per_stream: 0.25,
    rows: Cow::Borrowed(&[(1, 0.96), (2, 0.97), (3, 0.98), (4, 0.98), (8, 1.00), (16, 1.02), (24, 1.04), (32, 1.05)]),
    lm_head: 1.75,
    step: 0.93,
    step_per_stream: 0.25,
};

// Two ranks with the 4-bit head split (`CostTables::tp_two_ranks_for`): NOT MEASURED. The group's bf16 tables ([`MTP_COST_TP`],
// [`MTP_COST_RESTRICTED_TP`]) scaled per term by the single-node 4-bit / bf16 ratios ([`MtpCost::scaled_per_term`]), EXTRAPOLATED:
// restricted per_stream 0.57 -> 0.25, lm_head 0.93 -> 0.94, step 1.62 -> 0.87, step_per_stream 0.32 -> 0.15, base 2.22 kept, the
// absorb x0.53; full per_stream 0.60 -> 0.31, lm_head 2.40 -> 2.42, step 1.62 -> 0.75, step_per_stream 0.32 -> 0.19. The boot log says so
// (`event=tp_costs ... head=exl3`), and the arbiter's online scale (clamped 0.7..1.6) follows the real rounds.

// ------------------------------------------------------------------------------------------------ the MixedK form (v0.1.2): MEASURED

// The MixedK pack (vcruz305/Nemotron-Super-VL-EXL3-3.85bpw @39638152, exllamav3 1.5.1.post1): routed experts at 2 to 6 bits, the
// other trunk linears at 4 to 8, the trunk's `lm_head` EXL3 K8, the head fully EXL3 K4 (its attention and `eh_proj` included), the
// router gate F16 (held as stored, widened in register, D1), and the restricted draft head's rows decoded to bf16 at load (D2). Its
// verify step runs another trunk (other widths, on the any-width decode kernels `nm2_gemmk`), its full `lm_head` is an EXL3 GEMV and
// its head reads EXL3 attention: none of the tables above was measured on that form ([`HeadForm::MixedK`]), so it has its own.
//
// MEASURED (`tests/nemotron_omni_costs.rs` `print_the_cost_tables_in_the_format_of_the_constants`, N4 anonymized host-d, 2026-10-17 15:33 to
// 15:34 CDT, test binary @3fca49f3 (rexl3-serve bbbf6e4d of the same build), ref-text-1k, his pack (index 75291cea...), `REXL3_COSTS_WIDE=1`,
// once with `REXL3_MTP_VOCAB=default` (the restricted head) and once unset (the full one), back to back with our 4-bit head pack
// `hf-nemotron-main-h4` in the same session): printed and pasted verbatim, the cells in `tests::MEASURED_MIXEDK` /
// `MEASURED_RESTRICTED_MIXEDK`.
//   draft call, restricted (49,152 ids): worst 5.0%, mean 1.5% (the fit against its 40 cells); full `lm_head`: worst 2.1%, mean 0.9%
//   stage profile of 1 ask x 1 row x depth 1 (ours in the same session): attention block 0.18 ms (0.35), eh_proj 0.16 (0.35), moe block
//   0.43 (0.41), lm_head restricted 1.64 (1.72), lm_head full (EXL3 K8) 2.36 (bf16: 4.67 in the H4 measurement)
//   so against the H4 tables the call's chained step is 0.62 / 0.66 ms (0.93 / 0.85) and the absorb 0.51 / 0.57 (0.96 / 0.92)
//   verify window (one sequence at ~256 tokens, median of 7, the two runs averaged; ours in the same session in brackets):
//   36 (37), 43 (45), 50 (52), 56 (58), 60 (61), 62 (63), 65 (67), 66 (69) ms at 1 to 8 rows, 78 (79), 113 (109), 198 (196) at 16 / 32 / 64
// The verify table is the test's at every row. [`VERIFY_MS`]'s 3 to 8 rows are SERVED rounds, which pay 2 to 6 ms more a row past the
// fourth than the test's oracle text (its doc; this session's ours: 61 / 63 / 67 / 69 against the shipped 63 / 67 / 71 / 75), so the
// MixedK table prices those rows lower than ours does on the same footing; the arbiter's online scale (`NemotronCost::observe_window`,
// clamped 0.7..1.6, `SCALE_RANGE`) follows the real rounds either way, and a cost only decides which drafts are offered, never a committed
// token. No batched table: [`VERIFY_BATCHED`]'s union fit is our trunk's, so `REXL3_NEMO_ARB_TABLE=batched` prices this form with the
// single table. Two ranks: no group of this form was measured; [`CostTables::tp_two_ranks_for`] scales the group's tables by these
// single-node ones (EXTRAPOLATED, said so in the boot log).

/// Whether the MixedK tables were measured on that form (the boot log says `measured on the MixedK form`, or `UNMEASURED` while they are
/// placeholders: the K = 4 trunk's [`VERIFY_MS`] and the 4-bit head's [`MTP_COST_H4`] / [`MTP_COST_RESTRICTED_H4`]).
pub const MIXEDK_MEASURED: bool = true;

/// The verify window of the MixedK form, by total rows: the measurement above.
pub const VERIFY_MS_MIXEDK: &[(usize, f64)] = &[
    (1, 36.0),
    (2, 43.0),
    (3, 50.0),
    (4, 56.0),
    (5, 60.0),
    (6, 62.0),
    (7, 65.0),
    (8, 66.0),
    (16, 78.0),
    (32, 113.0),
    (64, 198.0),
];

/// The MixedK form's draft call with the trunk's full (EXL3 K8) `lm_head`: the measurement above, `REXL3_MTP_VOCAB` unset.
pub const MTP_COST_MIXEDK: MtpCost = MtpCost {
    base: 0.10,
    per_stream: 0.12,
    rows: Cow::Borrowed(&[(1, 0.57), (2, 0.57), (3, 0.57), (4, 0.58), (8, 0.58), (16, 0.60), (24, 0.60), (32, 0.61)]),
    lm_head: 2.36,
    step: 0.66,
    step_per_stream: 0.12,
};

/// The MixedK form's draft call over the default restricted vocabulary (49,152 bf16 rows decoded at load, D2): the measurement above,
/// `REXL3_MTP_VOCAB=default`. Other sizes: [`MtpCost::restricted_sub_table_to`].
pub const MTP_COST_RESTRICTED_MIXEDK: MtpCost = MtpCost {
    base: 0.10,
    per_stream: 0.18,
    rows: Cow::Borrowed(&[(1, 0.51), (2, 0.51), (3, 0.52), (4, 0.52), (8, 0.53), (16, 0.54), (24, 0.55), (32, 0.55)]),
    lm_head: 1.64,
    step: 0.62,
    step_per_stream: 0.14,
};

/// The form of the loaded pack's draft head and `lm_head`, which picks the cost tables ([`CostTables::shipped_for`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeadForm {
    /// Every projection bf16 (the published pack).
    #[default]
    Bf16,
    /// Some projection EXL3 (the 4-bit head pack), the head's attention and the trunk's `lm_head` bf16.
    Exl3,
    /// The trunk's `lm_head` EXL3, or the head's attention EXL3: the MixedK pack's form (v0.1.2). Its tables are its own:
    /// [`VERIFY_MS_MIXEDK`], [`MTP_COST_MIXEDK`], [`MTP_COST_RESTRICTED_MIXEDK`] (UNMEASURED placeholders, [`MIXEDK_MEASURED`]).
    MixedK,
}

/// The constant names of a form's cost tables in this file: what `tests/nemotron_omni_costs.rs` prints its measurement as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostTableNames {
    pub verify: &'static str,
    pub mtp: &'static str,
    pub mtp_restricted: &'static str,
}

impl HeadForm {
    /// `bf16`, `exl3` or `mixedk`: the `head=` of the boot log line.
    pub fn label(&self) -> &'static str {
        match self {
            HeadForm::Bf16 => "bf16",
            HeadForm::Exl3 => "exl3",
            HeadForm::MixedK => "mixedk",
        }
    }

    /// The form of a pack from what it holds: an EXL3 trunk `lm_head` or EXL3 head attention is [`HeadForm::MixedK`]; otherwise any
    /// EXL3 head projection (`eh_proj`, the MoE block's) is [`HeadForm::Exl3`]; otherwise [`HeadForm::Bf16`].
    pub fn of(head_has_exl3: bool, head_attention_exl3: bool, lm_head_exl3: bool) -> HeadForm {
        if lm_head_exl3 || head_attention_exl3 {
            HeadForm::MixedK
        } else if head_has_exl3 {
            HeadForm::Exl3
        } else {
            HeadForm::Bf16
        }
    }

    /// The single-node tables of this form in this file, by constant name (the verify table of the bf16 and the 4-bit head forms is
    /// the same trunk's, [`VERIFY_MS`]).
    pub fn table_names(&self) -> CostTableNames {
        match self {
            HeadForm::Bf16 => CostTableNames { verify: "VERIFY_MS", mtp: "MTP_COST", mtp_restricted: "MTP_COST_RESTRICTED" },
            HeadForm::Exl3 => {
                CostTableNames { verify: "VERIFY_MS", mtp: "MTP_COST_H4", mtp_restricted: "MTP_COST_RESTRICTED_H4" }
            }
            HeadForm::MixedK => CostTableNames {
                verify: "VERIFY_MS_MIXEDK",
                mtp: "MTP_COST_MIXEDK",
                mtp_restricted: "MTP_COST_RESTRICTED_MIXEDK",
            },
        }
    }

    /// Whether this form's tables were measured on it (the MixedK ones: [`MIXEDK_MEASURED`]).
    pub fn measured(&self) -> bool {
        match self {
            HeadForm::Bf16 | HeadForm::Exl3 => true,
            HeadForm::MixedK => MIXEDK_MEASURED,
        }
    }
}

/// `table` with each point scaled by the ratio of `num` to `den` at its rows (a point where they are equal is kept bit for bit): the
/// two-rank verify table of a form measured on one node only ([`CostTables::tp_two_ranks_for`]). An EXTRAPOLATION.
fn scaled_verify(table: &[(usize, f64)], num: &[(usize, f64)], den: &[(usize, f64)]) -> Vec<(usize, f64)> {
    let (Ok(n), Ok(d)) = (CostModel::try_from_table(num.to_vec()), CostModel::try_from_table(den.to_vec())) else {
        return table.to_vec();
    };
    table
        .iter()
        .map(|&(r, t)| {
            let (a, b) = (n.ms(r), d.ms(r));
            (r, if b > 0.0 && a != b { t * (a / b) } else { t })
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------------------- two-rank tensor parallelism

/// The verify window under a two-rank tensor-parallel group (`docs/NEMOTRON_TP.md`), by total rows, in milliseconds. **MEASURED** on the
/// live two-node group (node 1 rank 0, node 4 rank 1, the original pack, graphs on, the head split, `NCCL_MAX_NCHANNELS=4`, 2026-10-11
/// 04:07 to 04:16 CDT, the maintainers' head-table session): served rounds of `--spec mtp` at fixed depth 2 to 6 over the eight bench
/// prompts (192 tokens), the verify phase of the round at the mean window rows 2.99 / 3.96 / 4.95 / 5.90 / 6.88: 41.5 / 46.0 / 51.1 /
/// 53.9 / 58.1 ms; read at whole rows (interpolated: 3 to 7 rows). One row is the serial step of the bring-up (`--spec none`, 31.0 ms)
/// and two rows the line between it and three. EXTRAPOLATED, not measured: 8 rows (the slope of 4 to 7), and 16 to 64 rows (0.71 of the
/// single-node table plus the collectives' tax of `docs/NEMOTRON_TP.md` 10.5). Against the single-node table of [`VERIFY_MS`] the
/// measured points are 0.80 to 0.82 of it.
pub const VERIFY_MS_TP: &[(usize, f64)] = &[
    (1, 31.0),
    (2, 36.5),
    (3, 41.5),
    (4, 46.2),
    (5, 51.2),
    (6, 54.3),
    (7, 58.6),
    (8, 62.5),
    (16, 71.0),
    (32, 96.0),
    (64, 168.0),
];

// MEASURED on the same runs: the engine's wall time of `mtp_draft` (1 stream, 1 to 3 pending rows) at depth 2 / 3 / 4 / 5 / 6 =
// 7.89 / 10.83 / 14.00 / 16.45 / 19.44 ms with the head split (restricted 48k head): a line 2.23 + 2.87 d (a chain step 2.87 ms against
// 4.0 on one node; the 2.2 ms is the fixed cost of a call: the stream sync, the index uploads, the downloads and rank 1 entering the
// call, which on one node was 0.1). The terms are set so the model reproduces that line; the split of a step's 2.87 ms into `lm_head`
// (half of the single node's 1.74 and one allreduce of 192 KiB, 0.93), the blocks (1.62) and a further stream (the experts' share of
// the single node's 0.54, halved: 0.32) is NOT measured separately (one stream was run): the multi-stream terms are extrapolated.

/// The draft-call model under a two-rank group with the trunk's full `lm_head` split in halves (`REXL3_MTP_VOCAB=off`): the restricted
/// table's blocks with the full head's `lm_head` (half of 4.63 ms and one allreduce of 512 KiB).
pub const MTP_COST_TP: MtpCost = MtpCost {
    base: 2.22,
    per_stream: 0.60,
    rows: Cow::Borrowed(&[(1, 1.68), (2, 1.69), (3, 1.69), (4, 1.69), (8, 1.70), (16, 1.71), (24, 1.72), (32, 1.73)]),
    lm_head: 2.40,
    step: 1.62,
    step_per_stream: 0.32,
};

/// The same with the default restricted head (49,152 ids): the measured line above. `MtpCost::restricted_to` gives the other sizes.
pub const MTP_COST_RESTRICTED_TP: MtpCost = MtpCost {
    base: 2.22,
    per_stream: 0.57,
    rows: Cow::Borrowed(&[(1, 1.38), (2, 1.39), (3, 1.39), (4, 1.39), (8, 1.40), (16, 1.41), (24, 1.42), (32, 1.43)]),
    lm_head: 0.93,
    step: 1.62,
    step_per_stream: 0.32,
};

/// Where the arbiter's cost tables of a boot come from, for the boot log and the tests ([`CostTables::for_group`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostBasis {
    /// One node, or a group that runs whole models in lockstep: the shipped single-node tables, measured.
    SingleNode,
    /// Two ranks: [`VERIFY_MS_TP`] and the split-head draft tables, measured on the live two-node group.
    MeasuredTwoRanks,
    /// Three ranks or more: no such group was measured; the single-node tables with the online scale clamp.
    Unmeasured { ranks: usize },
}

impl CostBasis {
    /// The basis of a group of `ranks`: one node, the measured two-rank group, or a larger group nobody measured.
    pub fn of_ranks(ranks: usize) -> Self {
        match ranks {
            0 | 1 => CostBasis::SingleNode,
            2 => CostBasis::MeasuredTwoRanks,
            r => CostBasis::Unmeasured { ranks: r },
        }
    }

    /// `single`, `measured_2_ranks` or `unmeasured`: the `basis=` of the boot log line.
    pub fn label(&self) -> &'static str {
        match self {
            CostBasis::SingleNode => "single_node",
            CostBasis::MeasuredTwoRanks => "measured_two_ranks",
            CostBasis::Unmeasured { .. } => "unmeasured",
        }
    }
}

/// The cost tables a backend's arbiter prices a round with. The seam where re-measured tables replace the shipped ones: build one, hand
/// it to [`SpecRuntime::set_costs`] (or `engine::assemble_with`).
#[derive(Clone, Debug, PartialEq)]
pub struct CostTables {
    /// (rows, milliseconds) of a verify window: [`VERIFY_MS`] or the measured table.
    pub verify: Vec<(usize, f64)>,
    pub mtp: MtpCost,
    pub mtp_restricted: MtpCost,
    /// The verify step by sequences and rows that `REXL3_NEMO_ARB_TABLE=batched` prices with ([`VERIFY_BATCHED`]); `None` where none was
    /// measured (a two-rank group), and `batched` then prices with `verify` as `single` does.
    pub batched: Option<BatchedVerify>,
}

impl CostTables {
    /// The tables the engine ships with: [`VERIFY_MS`] and [`MTP_COST`] / [`MTP_COST_RESTRICTED`], all fitted to or measured on the real
    /// model (the step will get faster: regenerate them with `tests/nemotron_omni_costs.rs`), and [`VERIFY_BATCHED`] (regenerate it from
    /// served `/metrics` dumps with `tools/nemotron_arb_table.py`).
    pub fn shipped() -> Self {
        Self { verify: VERIFY_MS.to_vec(), mtp: MTP_COST, mtp_restricted: MTP_COST_RESTRICTED, batched: Some(VERIFY_BATCHED) }
    }

    /// The tables of a single node with a head of `head`'s form: [`CostTables::shipped`] for a bf16 head (bit for bit), and for a 4-bit
    /// head the same with [`MTP_COST_H4`] / [`MTP_COST_RESTRICTED_H4`] as the draft-call tables (the verify step is the trunk's).
    pub fn shipped_for(head: HeadForm) -> Self {
        match head {
            HeadForm::Bf16 => Self::shipped(),
            HeadForm::Exl3 => Self { mtp: MTP_COST_H4, mtp_restricted: MTP_COST_RESTRICTED_H4, ..Self::shipped() },
            // its own trunk and head: the MixedK tables (UNMEASURED placeholders until `MIXEDK_MEASURED`), no batched table
            HeadForm::MixedK => Self {
                verify: VERIFY_MS_MIXEDK.to_vec(),
                mtp: MTP_COST_MIXEDK,
                mtp_restricted: MTP_COST_RESTRICTED_MIXEDK,
                batched: None,
            },
        }
    }

    /// [`CostTables::tp_two_ranks`] for a head of `head`'s form. A bf16 head: those tables, bit for bit. A 4-bit head: no two-rank draft
    /// call of it was measured, so with the head split ([`MTP_COST_TP`] / [`MTP_COST_RESTRICTED_TP`] in use) each term of the group's
    /// bf16 tables is scaled by the single-node ratio of the 4-bit head's term to the bf16 head's ([`MtpCost::scaled_per_term`]):
    /// UNMEASURED, extrapolated, said so in the boot log (`engine::log_cost_basis_for`); with the head replicated the single-node call
    /// runs, whose 4-bit tables stay. The arbiter's online scale (clamped 0.7..1.6) follows the real rounds either way.
    pub fn tp_two_ranks_for(head_split: bool, head: HeadForm) -> Self {
        let mut t = Self::tp_two_ranks(head_split);
        if head == HeadForm::Exl3 {
            if head_split {
                t.mtp = MTP_COST_TP.scaled_per_term(&MTP_COST_H4, &MTP_COST);
                t.mtp_restricted = MTP_COST_RESTRICTED_TP.scaled_per_term(&MTP_COST_RESTRICTED_H4, &MTP_COST_RESTRICTED);
            } else {
                t.mtp = MTP_COST_H4;
                t.mtp_restricted = MTP_COST_RESTRICTED_H4;
            }
        }
        if head == HeadForm::MixedK {
            // no group of this form was measured: the group's verify table scaled per row by the form's single-node table over ours,
            // and the draft tables as the 4-bit head's are carried over (every one an extrapolation of UNMEASURED placeholders until
            // `MIXEDK_MEASURED`; with the placeholders as they are, exactly the 4-bit head's group tables)
            t.verify = scaled_verify(VERIFY_MS_TP, VERIFY_MS_MIXEDK, VERIFY_MS);
            if head_split {
                t.mtp = MTP_COST_TP.scaled_per_term(&MTP_COST_MIXEDK, &MTP_COST);
                t.mtp_restricted = MTP_COST_RESTRICTED_TP.scaled_per_term(&MTP_COST_RESTRICTED_MIXEDK, &MTP_COST_RESTRICTED);
            } else {
                t.mtp = MTP_COST_MIXEDK;
                t.mtp_restricted = MTP_COST_RESTRICTED_MIXEDK;
            }
        }
        t
    }

    /// [`CostTables::for_group`] for a head of `head`'s form: one rank (or three and more) [`CostTables::shipped_for`], two ranks
    /// [`CostTables::tp_two_ranks_for`]. A bf16 head gets exactly what `for_group` gives.
    pub fn for_group_for(ranks: usize, head_split: bool, head: HeadForm) -> (Self, CostBasis) {
        let basis = CostBasis::of_ranks(ranks);
        match basis {
            CostBasis::MeasuredTwoRanks => (Self::tp_two_ranks_for(head_split, head), basis),
            _ => (Self::shipped_for(head), basis),
        }
    }

    /// The tables of a rank-0 engine of a two-rank tensor-parallel group: [`VERIFY_MS_TP`] and, when the draft head is split across the
    /// ranks (`head_split`: both its experts and its restricted `lm_head`, the default), [`MTP_COST_TP`] and [`MTP_COST_RESTRICTED_TP`].
    /// A head replicated on both ranks (or split in one part only) runs the single-node call, whose tables stay: the draft call of
    /// the replicated layout was measured at 12.2 ms at depth 3 against 12.4 on one node. No batched table: none was measured on the
    /// group, and the single node's would price the group's step (0.80 of the single node's at 3 to 7 rows) by another machine.
    pub fn tp_two_ranks(head_split: bool) -> Self {
        let mut t = Self::shipped();
        t.verify = VERIFY_MS_TP.to_vec();
        t.batched = None;
        if head_split {
            t.mtp = MTP_COST_TP;
            t.mtp_restricted = MTP_COST_RESTRICTED_TP;
        }
        t
    }

    /// The tables of a rank-0 engine of a tensor-parallel group of `ranks`, and where they come from ([`CostBasis`]): one rank the
    /// shipped single-node tables, TWO ranks the measured group tables ([`CostTables::tp_two_ranks`]), three and more the SINGLE-NODE
    /// tables, labelled unmeasured. No group of three or more was ever run on a fabric (only two nodes are usable), and the two-rank
    /// tables carry a collective tax of that group's size (81 allreduces of two-rank rings), so extending them would be a guess; the
    /// arbiter's online scale (`NemotronCost::observe_window` / `observe_draft`, clamped to 0.7..1.6) follows the real rounds meanwhile.
    /// What this changes is which drafts are offered, never a committed token.
    pub fn for_group(ranks: usize, head_split: bool) -> (Self, CostBasis) {
        let basis = CostBasis::of_ranks(ranks);
        match basis {
            CostBasis::MeasuredTwoRanks => (Self::tp_two_ranks(head_split), basis),
            _ => (Self::shipped(), basis),
        }
    }

    /// These tables with another verify table (rows, milliseconds).
    pub fn with_verify(mut self, verify: &[(usize, f64)]) -> Self {
        self.verify = verify.to_vec();
        self
    }

    /// Whether every table is one the arbiter can use (the verify table is a valid `CostModel` table, the draft tables ascend).
    pub fn validate(&self) -> Result<(), String> {
        CostModel::try_from_table(self.verify.clone()).map_err(|e| format!("verify table: {e}"))?;
        self.mtp.validate("draft cost")?;
        self.mtp_restricted.validate("restricted draft cost")?;
        self.batched.map_or(Ok(()), |b| b.validate("batched verify table"))
    }
}

/// What the arbiter prices a round with: the step table and the draft model, each scaled by how the last rounds really ran against them
/// (clamped: a single odd round must not move a decision far).
pub struct NemotronCost {
    verify: CostModel,
    pub full: MtpCost,
    pub restricted: MtpCost,
    /// Price a draft call with `restricted` (`SpecConfig::restricted_head`).
    pub use_restricted: bool,
    /// The batched table, when the tables have one ([`CostTables::batched`]), and which table prices a round (`SpecConfig::arb_table`):
    /// `batched` without one prices as `single` does.
    pub batched: Option<BatchedVerify>,
    pub table: ArbTable,
    /// Measured over predicted, smoothed: of the rounds and draft calls the single table and the draft model price, and apart, of those
    /// the batched table prices (each scale learns only against the model that priced what it measures).
    pub verify_scale: f64,
    pub mtp_scale: f64,
    pub batched_verify_scale: f64,
    pub batched_mtp_scale: f64,
}

const SCALE_DECAY: f64 = 0.9;
const SCALE_RANGE: (f64, f64) = (0.7, 1.6);

impl NemotronCost {
    /// The tables `t`, priced by [`ARB_TABLE_DEFAULT`] until `table` is set.
    pub fn new(t: &CostTables, use_restricted: bool) -> Result<Self, String> {
        t.validate()?;
        Ok(Self {
            verify: CostModel::try_from_table(t.verify.clone())?,
            full: t.mtp.clone(),
            restricted: t.mtp_restricted.clone(),
            use_restricted,
            batched: t.batched,
            table: ARB_TABLE_DEFAULT,
            verify_scale: 1.0,
            mtp_scale: 1.0,
            batched_verify_scale: 1.0,
            batched_mtp_scale: 1.0,
        })
    }
    /// The draft model in use.
    pub fn mtp(&self) -> &MtpCost {
        if self.use_restricted {
            &self.restricted
        } else {
            &self.full
        }
    }
    /// The batched table when it prices the rounds: `table` is [`ArbTable::Batched`] and the tables have one.
    pub fn batched_in_use(&self) -> Option<&BatchedVerify> {
        self.batched.as_ref().filter(|_| self.table == ArbTable::Batched)
    }
    /// The batched table when it prices a round of `streams` sequences, or a draft call of `streams` streams: in use, and
    /// [`BatchedVerify::prices`] that many (a lone sequence is priced by the single table under either setting).
    pub fn batched_for(&self, streams: usize) -> Option<&BatchedVerify> {
        self.batched_in_use().filter(|b| b.prices(streams))
    }
    /// A window of `rows` rows took `ms` (priced as one sequence's: [`NemotronCost::observe_window_of`] with the window's sequences).
    pub fn observe_window(&mut self, rows: usize, ms: f64) {
        let predicted = self.verify.ms(rows);
        self.verify_scale = Self::blend(self.verify_scale, ms / predicted);
    }
    /// A window of `streams` sequences over `rows` rows took `ms` (the forward and its top-k: the commits that follow it are not in it).
    /// A window the single table prices is [`NemotronCost::observe_window`], to the bit; one the batched table prices moves its own scale
    /// against [`BatchedVerify::window_ms`], so that each scale measures its table and not the sequences the other leaves out.
    pub fn observe_window_of(&mut self, streams: usize, rows: usize, ms: f64) {
        match self.batched_for(streams).copied() {
            Some(b) => self.batched_verify_scale = Self::blend(self.batched_verify_scale, ms / b.window_ms(streams, rows)),
            None => self.observe_window(rows, ms),
        }
    }
    /// A draft call for `streams` streams took `ms`. Its wait for the device includes the commits the round before it enqueued
    /// ([`BatchedVerify`]): under the batched table, which prices them in the verify step and prices the call with its head terms, the
    /// prediction carries both (one commit per stream of the call, the streams that committed just before), so that nothing is counted
    /// twice; a call the single table's model prices is that model alone, as it always was.
    pub fn observe_draft(&mut self, streams: usize, rows: usize, depth: usize, ms: f64) {
        let predicted = self.mtp().ms(streams, rows, depth);
        match self.batched_for(streams).copied() {
            Some(b) => {
                let predicted = predicted + b.head_extra_ms(streams, depth) + b.commit_ms(streams);
                self.batched_mtp_scale = Self::blend(self.batched_mtp_scale, ms / predicted);
            }
            None => self.mtp_scale = Self::blend(self.mtp_scale, ms / predicted),
        }
    }
    fn blend(scale: f64, observed_ratio: f64) -> f64 {
        if !observed_ratio.is_finite() || observed_ratio <= 0.0 {
            return scale;
        }
        (scale * SCALE_DECAY + observed_ratio * (1.0 - SCALE_DECAY)).clamp(SCALE_RANGE.0, SCALE_RANGE.1)
    }
}

impl ArbiterCost for NemotronCost {
    fn verify_ms(&self, rows: usize) -> f64 {
        self.verify.ms(rows) * self.verify_scale
    }
    /// A round the single table prices (every round under [`ArbTable::Single`], a lone sequence under either): the trait's own default,
    /// `verify_ms(rows)`, the v0.1.0 pricing, to the bit. A round the batched table prices: its window times its online scale (what
    /// `observe_window_of` measures) and the commits of the round's sequences.
    fn verify_ms_of(&self, streams: usize, rows: usize) -> f64 {
        match self.batched_for(streams) {
            Some(b) => b.window_ms(streams, rows) * self.batched_verify_scale + b.commit_ms(streams.min(rows)),
            None => self.verify_ms(rows),
        }
    }
    /// A call the batched table prices (several streams, under [`ArbTable::Batched`]) carries its head terms; any other is the draft model,
    /// to the bit.
    fn mtp_ms(&self, streams: usize, absorb_rows: usize, depth: usize) -> f64 {
        match self.batched_for(streams) {
            Some(b) => (self.mtp().ms(streams, absorb_rows, depth) + b.head_extra_ms(streams, depth)) * self.batched_mtp_scale,
            None => self.mtp().ms(streams, absorb_rows, depth) * self.mtp_scale,
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------- telemetry

/// Counters per source (`Source::idx`) and per draft depth; the measurements read them.
#[derive(Clone, Debug, Default)]
pub struct SpecStats {
    /// Rounds a stream's window was fed with drafts from the source (`Source::None`: rounds with no drafts at all).
    pub rounds: [u64; 3],
    /// Drafts the engine verified, drafts accepted, tokens committed (accepted + 1 per round), by source.
    pub verified: [u64; 3],
    pub accepted: [u64; 3],
    pub committed: [u64; 3],
    /// Per source and depth `j` (index `j - 1`): rounds that verified a depth-`j` draft, and of those the rounds where it was accepted.
    pub reached: [[u64; DEPTHS]; 3],
    pub hits: [[u64; DEPTHS]; 3],
    /// Rounds by how many drafts rode in the window, per source (index = drafts, 0..=DEPTHS): what the arbiter or the fixed depth
    /// really asked for.
    pub by_depth: [[u64; DEPTHS + 1]; 3],
    /// Wall milliseconds and calls of the head's draft call (absorb + chain), and the pending rows its absorbs held.
    pub mtp_ms: f64,
    pub mtp_calls: u64,
    pub mtp_rows: u64,
    /// Calls that were absorb-only for at least one stream (a store made room without drafting).
    pub absorb_only_calls: u64,
    /// Streams that gave up drafting for their sequence because a wide round would have had the head absorb their full store.
    pub wide_disabled: u64,
    /// Draft calls that failed (the slots of the call were switched off), and draft lists the runtime answered that the backend dropped
    /// as malformed (a token outside the vocabulary).
    pub mtp_errors: u64,
    pub mtp_bad_drafts: u64,
    /// Wall milliseconds, calls and rows of the verify windows.
    pub window_ms: f64,
    pub windows: u64,
    pub window_rows: u64,
}

impl SpecStats {
    /// Mean tokens committed per round of `s`.
    pub fn tokens_per_round(&self, s: Source) -> f64 {
        let i = s.idx();
        if self.rounds[i] == 0 {
            0.0
        } else {
            self.committed[i] as f64 / self.rounds[i] as f64
        }
    }
    /// Conditional acceptance of a depth-`j` draft (1-based) of `s`: accepted over reached.
    pub fn depth_acceptance(&self, s: Source, j: usize) -> f64 {
        let (r, h) = (self.reached[s.idx()][j - 1], self.hits[s.idx()][j - 1]);
        if r == 0 {
            0.0
        } else {
            h as f64 / r as f64
        }
    }

    fn record(&mut self, source: Source, verified: usize, accepted: usize) {
        let i = source.idx();
        self.rounds[i] += 1;
        self.verified[i] += verified as u64;
        self.accepted[i] += accepted as u64;
        self.committed[i] += accepted as u64 + 1;
        self.by_depth[i][verified.min(DEPTHS)] += 1;
        for j in 1..=verified.min(DEPTHS) {
            // depth j is reached when the j - 1 drafts before it were all accepted
            if accepted + 1 >= j {
                self.reached[i][j - 1] += 1;
                if accepted >= j {
                    self.hits[i][j - 1] += 1;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------------- runtime

/// What a slot's last `propose` offered, until its commit.
struct Round {
    choice: Choice,
    drafts: Vec<u32>,
    /// How many of `drafts` were in the window (the engine may trim the offer); 0 until `on_step`.
    verified: usize,
}

struct SlotSpec {
    copy: CopyDrafter,
    round: Option<Round>,
    /// The head failed for this sequence: it is not asked again (the runtime was told with `mtp_disable_slot`).
    mtp_off: bool,
    /// Totals of this sequence, for the line `end` logs.
    rounds: u64,
    committed: u64,
    by_source: [u64; 3],
}

impl SlotSpec {
    fn new() -> Self {
        Self {
            copy: CopyDrafter::new(COPY_MIN_MATCH, COPY_MAX),
            round: None,
            mtp_off: false,
            rounds: 0,
            committed: 0,
            by_source: [0; 3],
        }
    }
}

/// The speculation state of one backend.
pub struct SpecRuntime {
    pub cfg: SpecConfig,
    slots: Vec<SlotSpec>,
    pub stats: SpecStats,
    pub arbiter: Arbiter,
    pub cost: NemotronCost,
}

fn arbiter_config(cfg: &SpecConfig) -> ArbiterConfig {
    let depth = cfg.mtp_depth.clamp(1, MTP_MAX_DEPTH);
    ArbiterConfig {
        allow_mtp: cfg.mode.uses_mtp(),
        allow_copy: cfg.mode.uses_copy(),
        // `--spec auto` never drafts deeper than the depth setting
        max_mtp_depth: depth,
        warmup_depth: ArbiterConfig::default().warmup_depth.min(depth),
        max_rows: cfg.window_rows,
        // P(draft k right | drafts 0..k-1 right) measured by MT-A on 8 natural prompts (992 positions, full vocabulary, node 1):
        // 0.936 0.917 0.897 0.866 for k = 0..3: the first prior, and each further depth keeps 0.975 of the one before it (the mean of
        // 0.980 0.978 0.965). The 48k restricted head measured 0.928 0.912 0.893 0.860 on the same prompts (0.008 / 0.005 / 0.004 / 0.006
        // less per depth: the prior is lowered by the first of them). The arbiter learns the stream's own rates from there.
        mtp_prior: vec![if cfg.restricted_head { MTP_ACCEPT_PRIOR_RESTRICTED } else { MTP_ACCEPT_PRIOR }],
        mtp_chain: tuning(CHAIN_VAR).unwrap_or(MTP_ACCEPT_CHAIN),
        decay: tuning(DECAY_VAR).unwrap_or(ArbiterConfig::default().decay),
        ..ArbiterConfig::default()
    }
}

/// Measurement knobs of the arbiter (a number in `(0, 1]`; unset or malformed is the shipped value): the factor a deeper draft's prior keeps
/// of the one before it, and the decay of an acceptance estimate per observation (the window of an estimate is about `1 / (1 - decay)`
/// rounds). They exist to measure how the depth choice trades noise against tracking (`docs/NEMOTRON_OVERNIGHT_2026-10-11.md`).
pub const CHAIN_VAR: &str = "REXL3_NEMO_ARB_CHAIN";
pub const DECAY_VAR: &str = "REXL3_NEMO_ARB_DECAY";

fn tuning(var: &str) -> Option<f64> {
    std::env::var(var).ok()?.trim().parse::<f64>().ok().filter(|v| *v > 0.0 && *v <= 1.0)
}

/// Streams in one round from which `--spec auto` stops absorbing a full tap store for a stream (see `SpecRuntime::propose`): a round this
/// wide is a throughput round, in which the head's absorb costs a row of the head for every row of the step.
pub const WIDE_ROUND_STREAMS: usize = 32;

/// The measured acceptance of the first draft, and the factor each further depth keeps (see `arbiter_config`).
pub const MTP_ACCEPT_PRIOR: f64 = 0.936;
/// The same for the head over the 48k restricted vocabulary (English and code prompts: other scripts do worse, and the online estimates
/// carry the difference).
pub const MTP_ACCEPT_PRIOR_RESTRICTED: f64 = 0.928;
pub const MTP_ACCEPT_CHAIN: f64 = 0.975;

impl SpecRuntime {
    /// The shipped cost tables ([`CostTables::shipped`]); [`SpecRuntime::set_costs`] replaces them.
    pub fn new(cfg: SpecConfig, slots: usize) -> Self {
        let mut cost = NemotronCost::new(&CostTables::shipped(), cfg.restricted_head).expect("the built-in tables are valid");
        cost.table = cfg.arb_table;
        Self {
            cfg,
            slots: (0..slots).map(|_| SlotSpec::new()).collect(),
            stats: SpecStats::default(),
            arbiter: Arbiter::new(arbiter_config(&cfg)),
            cost,
        }
    }

    /// Price rounds with `t` from now on (measured tables); the online scales start over. A malformed table is refused and the old
    /// one stays.
    pub fn set_costs(&mut self, t: &CostTables) -> Result<(), String> {
        self.cost = NemotronCost::new(t, self.cfg.restricted_head)?;
        self.cost.table = self.cfg.arb_table;
        Ok(())
    }

    /// `cfg` is public and a measurement moves it between runs: the arbiter and the cost table in use follow it.
    fn follow_cfg(&mut self) {
        let a = arbiter_config(&self.cfg);
        self.arbiter.cfg.allow_mtp = a.allow_mtp;
        self.arbiter.cfg.allow_copy = a.allow_copy;
        self.arbiter.cfg.max_mtp_depth = a.max_mtp_depth;
        self.arbiter.cfg.warmup_depth = a.warmup_depth;
        self.arbiter.cfg.max_rows = a.max_rows;
        self.arbiter.cfg.mtp_prior = a.mtp_prior;
        self.cost.use_restricted = self.cfg.restricted_head;
        self.cost.table = self.cfg.arb_table;
    }

    /// A new sequence starts in `slot`.
    pub fn begin(&mut self, slot: usize) {
        if let Some(s) = self.slots.get_mut(slot) {
            *s = SlotSpec::new();
            self.arbiter.begin(slot);
        }
    }

    /// The sequence in `slot` ended (or failed): log its totals and forget it.
    pub fn end(&mut self, slot: usize) {
        let Some(s) = self.slots.get_mut(slot) else { return };
        let s = std::mem::replace(s, SlotSpec::new());
        if s.rounds > 0 {
            eprintln!(
                "rexl3 event=spec_seq slot={slot} rounds={} tokens={} tokens_per_round={:.3} rounds_none={} rounds_copy={} rounds_mtp={}",
                s.rounds,
                s.committed,
                s.committed as f64 / s.rounds as f64,
                s.by_source[0],
                s.by_source[1],
                s.by_source[2]
            );
        }
    }

    /// Whether the head is off for `slot` (it failed in this sequence).
    pub fn mtp_off(&self, slot: usize) -> bool {
        self.slots.get(slot).is_some_and(|s| s.mtp_off)
    }

    /// The drafts for the streams that asked this round, one entry per request (`Some(Proposal)`, with no tokens for none: the engine's
    /// own drafter is not consulted for an entry). The mode decides the source. `allowed(slot, first_position, rows)` is the backend's
    /// `spec_allowed_for_window`: whether the pending rows the head would absorb are text.
    ///
    /// An `Err` is the runtime's fatal error (the backend latches it); a recoverable one never leaves this function.
    pub fn propose<M: NemotronRuntime>(
        &mut self,
        m: &mut M,
        reqs: &[ProposeRequest],
        allowed: &dyn Fn(usize, usize, usize) -> bool,
    ) -> Result<Vec<Option<Proposal>>, ModelError> {
        if let Some(r) = reqs.iter().find(|r| r.slot >= self.slots.len()) {
            return Err(ModelError::Request(format!(
                "drafting for slot {}, but the model has {} slots",
                r.slot,
                self.slots.len()
            )));
        }
        self.follow_cfg();
        let mode = self.cfg.mode;
        let head = m.has_mtp() && mode.uses_mtp();
        let vocab = m.vocab_size();
        // the copy drafter's offer is free: look at it for every stream that may use it
        let copy: Vec<Vec<u32>> = reqs
            .iter()
            .map(|r| if mode.uses_copy() { self.slots[r.slot].copy.propose(r.tokens, r.max) } else { Vec::new() })
            .collect();
        // what the head could do for each stream: it must not have failed in this sequence, rows must be pending, and none of them may
        // be an image's (`spec_allowed_for_window`: an absorb from the tap store has only the token's embedding to offer)
        let mut ready: Vec<Option<usize>> = reqs
            .iter()
            .map(|r| {
                if !head || r.max == 0 || self.slots[r.slot].mtp_off {
                    return None;
                }
                let n = m.mtp_pending_rows(r.slot).filter(|&n| n > 0)?;
                // the pending rows are the taps of the last `n` committed positions: the sequence ends with one token not yet fed
                let first = r.tokens.len().saturating_sub(n + 1);
                allowed(r.slot, first, n).then_some(n)
            })
            .collect();
        // the store takes at most the base row and the `max` drafts of this round's window at its commit: absorb first when that would not fit
        let mut must_absorb: Vec<bool> =
            reqs.iter().zip(&ready).map(|(r, n)| n.is_some_and(|n| n + 1 + r.max > MTP_TAP_ROWS)).collect();
        // A WIDE round under `--spec auto` (many streams in one step): a stream whose store is full would have the head absorb 24 or more
        // rows through its bf16 MoE, a cost of about a row of the head for every row of the step, for a draft the arbiter does not pick in a
        // step this wide. It gives up drafting for the rest of its sequence instead (`mtp_disable_slot`: never an error, never a byte of
        // output; the slot's next sequence drafts again). A lone stream or a few are never wide, and keep the head.
        let wide = mode == SpecMode::Auto && reqs.len() >= WIDE_ROUND_STREAMS;
        // `wide_depth`: the head drafts for every stream that has it ready, so nothing is given up and nothing is absorbed for nothing
        let wide_drafts = wide && self.cfg.wide_depth > 0;
        if wide && !wide_drafts {
            for i in 0..reqs.len() {
                if must_absorb[i] {
                    m.mtp_disable_slot(reqs[i].slot);
                    self.slots[reqs[i].slot].mtp_off = true;
                    self.stats.wide_disabled += 1;
                    ready[i] = None;
                    must_absorb[i] = false;
                }
            }
        }
        let depth_cap = self.cfg.mtp_depth.clamp(1, MTP_MAX_DEPTH);
        let picks: Vec<Choice> = match mode {
            SpecMode::Auto if wide_drafts => {
                let depth = self.cfg.wide_depth.min(MTP_MAX_DEPTH);
                reqs.iter()
                    .enumerate()
                    .map(|(i, r)| if ready[i].is_some() { Choice::Mtp(depth.min(r.max)) } else { Choice::None })
                    .collect()
            }
            SpecMode::Auto => {
                let asks: Vec<Ask> = reqs
                    .iter()
                    .enumerate()
                    .map(|(i, r)| Ask {
                        stream: r.slot,
                        max: r.max,
                        copy_len: copy[i].len().min(r.max),
                        mtp_ok: ready[i].is_some(),
                        absorb_rows: ready[i].unwrap_or(0),
                        must_absorb: must_absorb[i],
                    })
                    .collect();
                self.arbiter.choose(&self.cost, &asks)
            }
            SpecMode::Mtp => reqs
                .iter()
                .enumerate()
                .map(|(i, r)| if ready[i].is_some() { Choice::Mtp(depth_cap.min(r.max)) } else { Choice::None })
                .collect(),
            SpecMode::Copy => {
                (0..reqs.len()).map(|i| if copy[i].is_empty() { Choice::None } else { Choice::Copy(copy[i].len()) }).collect()
            }
            SpecMode::None => vec![Choice::None; reqs.len()],
        };
        // run the head for the streams that draft with it, and, to make room in a store that would overflow, absorb-only for those that
        // do not: all of them in ONE call
        let mut asked: Vec<usize> = Vec::new();
        let mut list: Vec<DraftAsk> = Vec::new();
        for (i, (r, c)) in reqs.iter().zip(&picks).enumerate() {
            match *c {
                Choice::Mtp(d) => {
                    asked.push(i);
                    list.push(DraftAsk { slot: r.slot, tokens: r.tokens, depth: d.min(MTP_MAX_DEPTH) });
                }
                _ if must_absorb[i] => {
                    asked.push(i);
                    list.push(DraftAsk { slot: r.slot, tokens: r.tokens, depth: 0 });
                }
                _ => {}
            }
        }
        let mut out: Vec<Option<Proposal>> = vec![Some(Proposal::default()); reqs.len()];
        let mut choice = picks.clone();
        if !list.is_empty() {
            let keyed = keyed_samplers(self.cfg.noise, &asked.iter().map(|&i| reqs[i].sampling).collect::<Vec<_>>());
            let (streams, rows, depth) = (
                list.len(),
                asked.iter().map(|&i| ready[i].unwrap_or(0)).sum::<usize>(),
                list.iter().map(|a| a.depth).max().unwrap_or(0),
            );
            let t = Instant::now();
            let answered = m.mtp_draft(&list, &keyed);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let failure = match answered {
                Ok(d) if d.len() == list.len() => {
                    self.stats.mtp_ms += ms;
                    self.stats.mtp_calls += 1;
                    self.stats.mtp_rows += rows as u64;
                    self.stats.absorb_only_calls += u64::from(list.iter().any(|a| a.depth == 0));
                    self.cost.observe_draft(streams, rows, depth, ms);
                    for ((&i, a), d) in asked.iter().zip(&list).zip(d) {
                        if a.depth == 0 {
                            continue;
                        }
                        // the offer is what the head chained, at most what was asked for and what the engine can take, and only real tokens
                        match d {
                            Some(d) if !d.ids.is_empty() => {
                                let n = d.ids.len().min(a.depth).min(reqs[i].max);
                                let mut ids = d.ids;
                                ids.truncate(n);
                                if ids.iter().any(|&t| t as usize >= vocab) {
                                    self.stats.mtp_bad_drafts += 1;
                                    choice[i] = Choice::None;
                                    continue;
                                }
                                // the confidences ride along only when there is one per token
                                let conf = if d.vals.len() >= n { d.vals[..n].to_vec() } else { Vec::new() };
                                out[i] = Some(Proposal { tokens: ids, conf, ..Default::default() });
                            }
                            // the slot could not draft after all: the copy offer, if the mode has one, is all there is
                            _ => choice[i] = Choice::None,
                        }
                    }
                    None
                }
                Ok(d) => Some(ModelError::Request(format!("the runtime answered {} drafts for {} asks", d.len(), list.len()))),
                Err(e) => Some(e),
            };
            if let Some(e) = failure {
                // the head is off for every slot of the call (the error does not say which one failed): the trunk was not touched, so
                // the round and the requests go on with what the copy drafter offers, or serially
                self.stats.mtp_errors += 1;
                let slots: Vec<usize> = list.iter().map(|a| a.slot).collect();
                for &s in &slots {
                    m.mtp_disable_slot(s);
                    self.slots[s].mtp_off = true;
                }
                eprintln!(
                    "rexl3 level=warn event=mtp_error slots={slots:?} error=\"{e}\" (drafting from the MTP head is off for these sequences; \
                     their rounds go on without head drafts)"
                );
                if e.is_fatal() {
                    return Err(e);
                }
                for &i in &asked {
                    if matches!(choice[i], Choice::Mtp(_)) {
                        choice[i] = Choice::None;
                    }
                }
            }
        }
        for i in 0..reqs.len() {
            match picks[i] {
                Choice::Copy(n) => {
                    out[i] = Some(Proposal { tokens: copy[i][..n.min(copy[i].len())].to_vec(), ..Default::default() });
                }
                Choice::Mtp(_) if choice[i] == Choice::None && !copy[i].is_empty() && mode != SpecMode::Mtp => {
                    out[i] = Some(Proposal { tokens: copy[i].clone(), ..Default::default() });
                    choice[i] = Choice::Copy(copy[i].len());
                }
                _ => {}
            }
        }
        for (i, r) in reqs.iter().enumerate() {
            let drafts = out[i].as_ref().map(|p| p.tokens.clone()).unwrap_or_default();
            // the choice as it ended up: a pick whose offer came out empty was a serial round
            let c = match choice[i] {
                Choice::Copy(_) if drafts.is_empty() => Choice::None,
                Choice::Copy(_) => Choice::Copy(drafts.len()),
                Choice::Mtp(_) if drafts.is_empty() => Choice::None,
                Choice::Mtp(_) => Choice::Mtp(drafts.len()),
                Choice::None => Choice::None,
            };
            self.slots[r.slot].round = Some(Round { choice: c, drafts, verified: 0 });
        }
        Ok(out)
    }

    /// A verify window of `streams` sequences and `rows` rows in all took `ms` of wall time (the forward and its top-k).
    pub fn on_window(&mut self, streams: usize, rows: usize, ms: f64) {
        self.stats.window_ms += ms;
        self.stats.windows += 1;
        self.stats.window_rows += rows as u64;
        self.cost.observe_window_of(streams, rows, ms);
    }

    /// The engine put `rows` rows (the base row and the drafts it kept) in `slot`'s window.
    pub fn on_step(&mut self, slot: usize, rows: usize) {
        if let Some(r) = self.slots.get_mut(slot).and_then(|s| s.round.as_mut()) {
            r.verified = rows.saturating_sub(1).min(r.drafts.len());
        }
    }

    /// `keep` rows of `slot`'s window were accepted: `keep - 1` drafts, and one sample past them.
    pub fn on_commit(&mut self, slot: usize, keep: usize) {
        let accepted = keep.saturating_sub(1);
        let Some(s) = self.slots.get_mut(slot) else { return };
        let (choice, verified) = match s.round.take() {
            Some(r) => (if r.verified == 0 { Choice::None } else { r.choice }, r.verified),
            None => (Choice::None, 0),
        };
        let source = Source::of(choice);
        s.rounds += 1;
        s.committed += accepted as u64 + 1;
        s.by_source[source.idx()] += 1;
        let accepted = accepted.min(verified);
        self.stats.record(source, verified, accepted);
        if self.cfg.mode == SpecMode::Auto {
            self.arbiter.observe(slot, choice, verified, accepted);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stats_count_a_round_by_depth_and_source() {
        let mut s = SpecStats::default();
        // an MTP round of three drafts, the first two accepted: depth 1 and 2 hit, depth 3 was reached and missed
        s.record(Source::Mtp, 3, 2);
        assert_eq!((s.rounds[2], s.verified[2], s.accepted[2], s.committed[2]), (1, 3, 2, 3));
        assert_eq!(s.reached[2][..3], [1, 1, 1]);
        assert_eq!(s.hits[2][..3], [1, 1, 0]);
        // a round whose first draft was rejected never reaches depth 2
        s.record(Source::Mtp, 3, 0);
        assert_eq!(s.reached[2][..3], [2, 1, 1]);
        assert_eq!(s.hits[2][..3], [1, 1, 0]);
        assert!((s.depth_acceptance(Source::Mtp, 1) - 0.5).abs() < 1e-12);
        assert!((s.depth_acceptance(Source::Mtp, 2) - 1.0).abs() < 1e-12);
        assert!((s.tokens_per_round(Source::Mtp) - 2.0).abs() < 1e-12);
        // all accepted: every depth reached and hit
        s.record(Source::Copy, 2, 2);
        assert_eq!(s.reached[1][..2], [1, 1]);
        assert_eq!(s.hits[1][..2], [1, 1]);
        assert_eq!(s.committed[1], 3);
    }

    #[test]
    fn a_round_with_no_drafts_in_the_window_counts_as_none() {
        let mut r = SpecRuntime::new(SpecConfig::new(SpecMode::Mtp), 2);
        // the engine asked, the drafter offered two, the engine's budget trimmed them to nothing: the round is a serial one
        r.slots[1].round = Some(Round { choice: Choice::Mtp(2), drafts: vec![5, 6], verified: 0 });
        r.on_step(1, 1);
        r.on_commit(1, 1);
        assert_eq!((r.stats.rounds[0], r.stats.rounds[2]), (1, 0));
        // a trimmed offer: two drafts offered, one in the window and accepted
        r.slots[0].round = Some(Round { choice: Choice::Mtp(2), drafts: vec![5, 6], verified: 0 });
        r.on_step(0, 2);
        r.on_commit(0, 2);
        assert_eq!((r.stats.rounds[2], r.stats.verified[2], r.stats.accepted[2], r.stats.committed[2]), (1, 1, 1, 2));
        // a commit with no recorded round (the stream was never asked) is a serial round too, and the record is consumed
        r.on_commit(0, 1);
        assert_eq!(r.stats.rounds[0], 2);
        assert!(r.slots[0].round.is_none());
    }

    #[test]
    fn auto_feeds_what_happened_back_to_the_arbiter_and_the_fixed_modes_do_not() {
        let mut auto = SpecRuntime::new(SpecConfig::new(SpecMode::Auto), 1);
        let before = auto.arbiter.mtp_alpha(0, 1);
        for _ in 0..30 {
            auto.slots[0].round = Some(Round { choice: Choice::Mtp(2), drafts: vec![1, 2], verified: 0 });
            auto.on_step(0, 3);
            auto.on_commit(0, 1); // both drafts rejected
        }
        assert!(auto.arbiter.mtp_alpha(0, 1) < before - 0.3, "{} -> {}", before, auto.arbiter.mtp_alpha(0, 1));
        let mut fixed = SpecRuntime::new(SpecConfig::new(SpecMode::Mtp), 1);
        for _ in 0..30 {
            fixed.slots[0].round = Some(Round { choice: Choice::Mtp(2), drafts: vec![1, 2], verified: 0 });
            fixed.on_step(0, 3);
            fixed.on_commit(0, 1);
        }
        assert!((fixed.arbiter.mtp_alpha(0, 1) - before).abs() < 1e-12, "a fixed depth never learns: it is what was asked for");
    }

    #[test]
    fn a_commit_counts_the_sample_past_the_drafts_as_a_committed_token_and_not_as_an_accepted_draft() {
        let mut r = SpecRuntime::new(SpecConfig::new(SpecMode::Mtp), 1);
        // three drafts verified, all rejected: the commit keeps the one row of the sample
        r.slots[0].round = Some(Round { choice: Choice::Mtp(3), drafts: vec![1, 2, 3], verified: 0 });
        r.on_step(0, 4);
        r.on_commit(0, 1);
        assert_eq!((r.stats.verified[2], r.stats.accepted[2], r.stats.committed[2]), (3, 0, 1));
        // two accepted: keep 3 rows, two drafts and the sample
        r.slots[0].round = Some(Round { choice: Choice::Mtp(3), drafts: vec![1, 2, 3], verified: 0 });
        r.on_step(0, 4);
        r.on_commit(0, 3);
        assert_eq!((r.stats.verified[2], r.stats.accepted[2], r.stats.committed[2]), (6, 2, 4));
    }

    #[test]
    fn the_noise_setting_is_validated_and_defaults_to_the_documented_pick() {
        assert_eq!(SpecConfig::new(SpecMode::Mtp).noise, DRAFT_NOISE_DEFAULT);
        assert_eq!(parse_noise(None), Ok(DRAFT_NOISE_DEFAULT));
        assert_eq!(parse_noise(Some("0")), Ok(DraftPick::Argmax));
        assert_eq!(parse_noise(Some("1")), Ok(DraftPick::Keyed));
        assert_eq!(parse_noise(Some(" 1 ")), Ok(DraftPick::Keyed));
        let e = parse_noise(Some("2")).unwrap_err();
        assert!(e.contains("REXL3_NEMO_DRAFT_NOISE=2"), "{e}");
        assert!(parse_noise(Some("")).is_err(), "an empty value is not a choice");
        assert!(parse_noise(Some("maybe")).is_err());
    }

    #[test]
    fn the_noise_setting_decides_which_sampled_asks_carry_their_requests_sampler() {
        let (a, b) = (
            Sampling { seed: 1, temperature: 0.7, ..Default::default() },
            Sampling { seed: 2, temperature: 0.0, ..Default::default() },
        );
        // keyed: the sampled request carries its sampler, the greedy one has no noise to key
        assert_eq!(keyed_samplers(DraftPick::Keyed, &[&a, &b]), vec![Some(a), None]);
        // argmax: nobody does
        assert_eq!(keyed_samplers(DraftPick::Argmax, &[&a, &b]), vec![None, None]);
        assert!(keyed_samplers(DraftPick::Keyed, &[]).is_empty());
    }

    #[test]
    fn the_depth_setting_is_validated() {
        assert_eq!(SpecConfig::new(SpecMode::Mtp).mtp_depth, MTP_DEFAULT_DEPTH);
        assert_eq!(MTP_DEFAULT_DEPTH, 3);
        assert_eq!(parse_depth("3"), Ok(3));
        assert_eq!(parse_depth(" 4 "), Ok(4));
        assert_eq!(parse_depth("7"), Ok(7));
        assert!(parse_depth("0").unwrap_err().contains(&format!("1..={MTP_MAX_DEPTH}")));
        assert!(parse_depth("8").unwrap_err().contains("got 8"));
        assert!(parse_depth("two").unwrap_err().contains("not a number"));
        assert!(parse_depth("REXL3").unwrap_err().contains("REXL3_MTP_DEPTH"));
    }

    #[test]
    fn the_config_takes_both_variables_and_a_malformed_one_is_an_error_not_the_default() {
        let c = SpecConfig::from_vars(SpecMode::Auto, Some("2"), Some("0"), Some("batched")).unwrap();
        assert_eq!(
            (c.mode, c.mtp_depth, c.noise, c.restricted_head, c.arb_table),
            (SpecMode::Auto, 2, DraftPick::Argmax, false, ArbTable::Batched)
        );
        let d = SpecConfig::from_vars(SpecMode::Mtp, None, None, None).unwrap();
        assert_eq!((d.mtp_depth, d.noise, d.arb_table), (MTP_DEFAULT_DEPTH, DRAFT_NOISE_DEFAULT, ARB_TABLE_DEFAULT));
        assert!(
            matches!(SpecConfig::from_vars(SpecMode::Mtp, Some("9"), None, None), Err(ModelError::Config(m)) if m.contains(&format!("1..={MTP_MAX_DEPTH}")))
        );
        assert!(matches!(SpecConfig::from_vars(SpecMode::Mtp, None, Some("maybe"), None), Err(ModelError::Config(_))));
        assert!(
            matches!(SpecConfig::from_vars(SpecMode::Auto, None, None, Some("wide")), Err(ModelError::Config(m)) if m.contains(ARB_TABLE_VAR))
        );
    }

    #[test]
    fn auto_never_drafts_deeper_than_the_depth_setting() {
        for depth in 1..=MTP_MAX_DEPTH {
            let r = SpecRuntime::new(SpecConfig { mtp_depth: depth, ..SpecConfig::new(SpecMode::Auto) }, 1);
            assert_eq!(r.arbiter.cfg.max_mtp_depth, depth);
            assert!(r.arbiter.cfg.warmup_depth <= depth);
        }
    }

    // ---- the cost tables

    /// The measurements the constants were fitted to (`tests/nemotron_omni_costs.rs`, node 1, 2026-10-11): (asks, depth) -> milliseconds at
    /// 1 / 8 pending rows per ask, the full head and the 48k restricted head (the default list).
    type Grid = [((usize, usize), [f64; 2]); 20];
    const MEASURED: Grid = [
        ((1, 1), [7.2, 7.1]),
        ((1, 2), [14.1, 14.3]),
        ((1, 3), [21.8, 20.9]),
        ((1, 4), [27.7, 29.1]),
        ((1, 5), [34.6, 34.7]),
        ((2, 1), [7.7, 8.0]),
        ((2, 2), [15.6, 15.5]),
        ((2, 3), [23.5, 23.9]),
        ((2, 4), [30.7, 31.0]),
        ((2, 5), [38.1, 39.2]),
        ((3, 1), [9.0, 9.2]),
        ((3, 2), [16.8, 16.5]),
        ((3, 3), [25.1, 24.7]),
        ((3, 4), [33.1, 35.1]),
        ((3, 5), [40.7, 41.4]),
        ((4, 1), [8.8, 9.0]),
        ((4, 2), [18.1, 17.5]),
        ((4, 3), [26.9, 25.4]),
        ((4, 4), [34.9, 34.0]),
        ((4, 5), [43.5, 42.5]),
    ];
    const MEASURED_RESTRICTED: Grid = [
        ((1, 1), [4.2, 4.1]),
        ((1, 2), [8.1, 8.2]),
        ((1, 3), [12.2, 12.3]),
        ((1, 4), [16.2, 16.3]),
        ((1, 5), [20.3, 20.3]),
        ((2, 1), [4.7, 5.0]),
        ((2, 2), [9.4, 9.3]),
        ((2, 3), [14.4, 13.7]),
        ((2, 4), [19.0, 17.8]),
        ((2, 5), [24.0, 22.3]),
        ((3, 1), [5.7, 5.7]),
        ((3, 2), [11.0, 10.3]),
        ((3, 3), [16.4, 14.8]),
        ((3, 4), [21.6, 19.3]),
        ((3, 5), [26.5, 24.4]),
        ((4, 1), [5.8, 6.0]),
        ((4, 2), [12.0, 11.0]),
        ((4, 3), [18.1, 16.3]),
        ((4, 4), [23.3, 21.8]),
        ((4, 5), [29.2, 28.2]),
    ];
    /// The 4-bit head pack's measurements the H4 constants were fitted to (`tests/nemotron_omni_costs.rs` on `hf-nemotron-main-h4`, N4,
    /// 2026-10-16, ref-text-1k): the full `lm_head` (`REXL3_MTP_VOCAB=off`) and the 48k restricted head (the default list).
    const MEASURED_H4: Grid = [
        ((1, 1), [6.0, 6.1]),
        ((1, 2), [11.6, 11.7]),
        ((1, 3), [17.4, 17.6]),
        ((1, 4), [23.2, 23.4]),
        ((1, 5), [29.0, 29.2]),
        ((2, 1), [6.2, 6.3]),
        ((2, 2), [12.4, 12.5]),
        ((2, 3), [18.6, 18.8]),
        ((2, 4), [24.7, 24.9]),
        ((2, 5), [30.9, 31.1]),
        ((3, 1), [6.7, 6.9]),
        ((3, 2), [13.6, 13.5]),
        ((3, 3), [20.3, 20.0]),
        ((3, 4), [26.5, 26.5]),
        ((3, 5), [33.0, 33.1]),
        ((4, 1), [6.8, 6.9]),
        ((4, 2), [13.6, 13.4]),
        ((4, 3), [20.3, 20.0]),
        ((4, 4), [27.0, 26.8]),
        ((4, 5), [33.5, 33.5]),
    ];
    const MEASURED_RESTRICTED_H4: Grid = [
        ((1, 1), [2.9, 3.1]),
        ((1, 2), [5.8, 6.0]),
        ((1, 3), [9.0, 8.9]),
        ((1, 4), [11.6, 11.8]),
        ((1, 5), [14.6, 14.7]),
        ((2, 1), [3.2, 3.4]),
        ((2, 2), [6.5, 6.6]),
        ((2, 3), [9.6, 9.6]),
        ((2, 4), [12.9, 12.7]),
        ((2, 5), [16.0, 15.9]),
        ((3, 1), [3.6, 3.8]),
        ((3, 2), [7.2, 7.2]),
        ((3, 3), [10.7, 10.7]),
        ((3, 4), [14.3, 14.4]),
        ((3, 5), [17.7, 18.0]),
        ((4, 1), [3.7, 3.9]),
        ((4, 2), [7.5, 7.3]),
        ((4, 3), [11.1, 11.0]),
        ((4, 4), [14.7, 14.6]),
        ((4, 5), [18.2, 18.4]),
    ];
    /// His MixedK pack's measurements the MixedK constants were fitted to (`tests/nemotron_omni_costs.rs` on
    /// `vcruz-nemotron-mixedk-385-39638152`, N4, 2026-10-17, ref-text-1k): the full EXL3 K8 `lm_head` (`REXL3_MTP_VOCAB` unset) and the
    /// 48k restricted head (the default list).
    const MEASURED_MIXEDK: Grid = [
        ((1, 1), [3.1, 3.1]),
        ((1, 2), [6.3, 6.2]),
        ((1, 3), [9.3, 9.3]),
        ((1, 4), [12.4, 12.4]),
        ((1, 5), [15.5, 15.6]),
        ((2, 1), [3.3, 3.4]),
        ((2, 2), [6.6, 6.6]),
        ((2, 3), [9.9, 9.9]),
        ((2, 4), [13.2, 13.2]),
        ((2, 5), [16.3, 16.6]),
        ((3, 1), [3.4, 3.5]),
        ((3, 2), [6.8, 6.8]),
        ((3, 3), [10.3, 10.2]),
        ((3, 4), [13.6, 13.5]),
        ((3, 5), [17.0, 17.0]),
        ((4, 1), [3.5, 3.6]),
        ((4, 2), [7.0, 6.9]),
        ((4, 3), [10.6, 10.3]),
        ((4, 4), [14.0, 13.8]),
        ((4, 5), [17.6, 17.2]),
    ];
    const MEASURED_RESTRICTED_MIXEDK: Grid = [
        ((1, 1), [2.5, 2.4]),
        ((1, 2), [4.8, 4.8]),
        ((1, 3), [7.2, 7.2]),
        ((1, 4), [9.6, 9.6]),
        ((1, 5), [12.0, 12.0]),
        ((2, 1), [2.6, 2.7]),
        ((2, 2), [5.2, 5.1]),
        ((2, 3), [7.8, 7.7]),
        ((2, 4), [10.3, 10.1]),
        ((2, 5), [13.0, 12.6]),
        ((3, 1), [2.8, 2.9]),
        ((3, 2), [5.6, 5.5]),
        ((3, 3), [8.5, 8.1]),
        ((3, 4), [11.2, 10.8]),
        ((3, 5), [14.0, 13.5]),
        ((4, 1), [2.9, 3.0]),
        ((4, 2), [5.8, 5.7]),
        ((4, 3), [8.9, 8.4]),
        ((4, 4), [11.7, 11.0]),
        ((4, 5), [14.6, 13.6]),
    ];
    const ROWS_PER_ASK: [usize; 2] = [1, 8];

    /// The draft-call tables follow the loaded head's form: a bf16 head gets today's tables bit for bit (one node, the two-rank group with
    /// the head split or replicated, three ranks), a 4-bit head the H4 tables, and under two ranks with the head split the group's bf16
    /// tables carried over term by term by the single-node 4-bit / bf16 ratios.
    #[test]
    fn the_selector_picks_the_bf16_tables_for_a_bf16_head_and_the_h4_tables_for_an_exl3_head() {
        assert_eq!(HeadForm::default(), HeadForm::Bf16);
        assert_eq!((HeadForm::Bf16.label(), HeadForm::Exl3.label()), ("bf16", "exl3"));
        // bf16: exactly the shipped tables, whatever the group
        assert_eq!(CostTables::shipped_for(HeadForm::Bf16), CostTables::shipped());
        for split in [true, false] {
            assert_eq!(CostTables::tp_two_ranks_for(split, HeadForm::Bf16), CostTables::tp_two_ranks(split));
            for ranks in [1, 2, 3, 4] {
                assert_eq!(
                    CostTables::for_group_for(ranks, split, HeadForm::Bf16),
                    CostTables::for_group(ranks, split),
                    "{ranks} ranks"
                );
            }
        }
        // a 4-bit head: the H4 draft tables, the trunk's verify tables
        let h4 = CostTables::shipped_for(HeadForm::Exl3);
        assert_eq!((h4.mtp.clone(), h4.mtp_restricted.clone()), (MTP_COST_H4, MTP_COST_RESTRICTED_H4));
        assert_eq!((h4.verify.clone(), h4.batched), (CostTables::shipped().verify, CostTables::shipped().batched));
        assert_eq!(CostTables::for_group_for(1, true, HeadForm::Exl3).0, h4);
        assert_eq!(CostTables::for_group_for(3, true, HeadForm::Exl3), (h4.clone(), CostBasis::Unmeasured { ranks: 3 }));
        // two ranks, head split: the group's tables scaled per term (unmeasured); replicated: the single-node 4-bit call
        let (tp, basis) = CostTables::for_group_for(2, true, HeadForm::Exl3);
        assert_eq!(basis, CostBasis::MeasuredTwoRanks);
        assert_eq!(tp.verify, VERIFY_MS_TP.to_vec());
        assert_eq!(tp.mtp, MTP_COST_TP.scaled_per_term(&MTP_COST_H4, &MTP_COST));
        assert_eq!(tp.mtp_restricted, MTP_COST_RESTRICTED_TP.scaled_per_term(&MTP_COST_RESTRICTED_H4, &MTP_COST_RESTRICTED));
        // the extrapolated group tables of the measured 4-bit head: the chained step and per-stream terms about halve, the lm_head (a bf16
        // sub-table either way) and the call's fixed cost stay
        let r = &tp.mtp_restricted;
        assert!((r.step - 0.871).abs() < 0.01 && (r.step_per_stream - 0.148).abs() < 0.01, "{r:?}");
        assert!((r.per_stream - 0.25).abs() < 0.01 && (r.lm_head - MTP_COST_RESTRICTED_TP.lm_head).abs() < 0.01, "{r:?}");
        assert_eq!(r.base, MTP_COST_RESTRICTED_TP.base);
        assert!((tp.mtp.step - 0.752).abs() < 0.01 && (tp.mtp.lm_head - 2.42).abs() < 0.01, "{:?}", tp.mtp);
        for d in 1..=5 {
            let (h4, bf) = (r.ms(1, 1, d), MTP_COST_RESTRICTED_TP.ms(1, 1, d));
            assert!(h4 < bf && h4 > 0.5 * bf, "depth {d}: {h4:.2} against the bf16 group's {bf:.2}");
            assert!(
                h4 < MTP_COST_RESTRICTED_H4.ms(1, 1, d) + 2.2,
                "depth {d}: the group's fixed cost is the only thing above one node"
            );
        }
        let rep = CostTables::tp_two_ranks_for(false, HeadForm::Exl3);
        assert_eq!((rep.mtp, rep.mtp_restricted, rep.verify), (MTP_COST_H4, MTP_COST_RESTRICTED_H4, VERIFY_MS_TP.to_vec()));
        tp.validate().unwrap();
        // the per-term scaling: equal tables leave a model as it is, a halved step halves the step and nothing else
        assert_eq!(MTP_COST_TP.scaled_per_term(&MTP_COST, &MTP_COST), MTP_COST_TP);
        let faster = MtpCost {
            step: MTP_COST.step / 2.0,
            rows: Cow::Owned(MTP_COST.rows.iter().map(|&(r, t)| (r, t * 0.8)).collect()),
            ..MTP_COST
        };
        let s = MTP_COST_TP.scaled_per_term(&faster, &MTP_COST);
        assert!((s.step - MTP_COST_TP.step / 2.0).abs() < 1e-12, "{}", s.step);
        assert_eq!(
            (s.base, s.per_stream, s.lm_head, s.step_per_stream),
            (MTP_COST_TP.base, MTP_COST_TP.per_stream, MTP_COST_TP.lm_head, MTP_COST_TP.step_per_stream)
        );
        for (&(r, a), &(r2, b)) in s.rows.iter().zip(MTP_COST_TP.rows.iter()) {
            assert!(r == r2 && (a - 0.8 * b).abs() < 1e-9, "{r}: {a} against {b}");
        }
        s.validate("scaled").unwrap();
    }

    /// v0.1.2 P2-9: the third form. Our pack's form (a 4-bit head: `HeadForm::Exl3`) and the bf16 head keep their tables bit for bit
    /// (the test above, unchanged); the MixedK form gets its own tables, UNMEASURED placeholders of the nearest measured ones.
    #[test]
    fn the_mixedk_form_has_its_own_tables_which_are_labelled_placeholders_until_measured() {
        // the selector: an EXL3 lm_head or EXL3 head attention is the MixedK form, whatever else; our two packs are what they were
        assert_eq!(HeadForm::of(false, false, false), HeadForm::Bf16);
        assert_eq!(HeadForm::of(true, false, false), HeadForm::Exl3, "our pack: the MoE block EXL3, attention and lm_head bf16");
        for (head, attn, lm) in
            [(true, true, true), (true, false, true), (true, true, false), (false, false, true), (false, true, false)]
        {
            assert_eq!(HeadForm::of(head, attn, lm), HeadForm::MixedK, "{head} {attn} {lm}");
        }
        assert_eq!(HeadForm::MixedK.label(), "mixedk");
        assert!(HeadForm::Bf16.measured() && HeadForm::Exl3.measured());
        assert_eq!(HeadForm::MixedK.measured(), MIXEDK_MEASURED);
        // the names the cost test prints a measurement under are this file's constants
        assert_eq!(HeadForm::Exl3.table_names().mtp, "MTP_COST_H4");
        assert_eq!(HeadForm::Bf16.table_names().mtp_restricted, "MTP_COST_RESTRICTED");
        let n = HeadForm::MixedK.table_names();
        assert_eq!((n.verify, n.mtp, n.mtp_restricted), ("VERIFY_MS_MIXEDK", "MTP_COST_MIXEDK", "MTP_COST_RESTRICTED_MIXEDK"));
        // the tables
        let mk = CostTables::shipped_for(HeadForm::MixedK);
        mk.validate().unwrap();
        assert_eq!(
            (mk.verify.clone(), mk.mtp.clone(), mk.mtp_restricted.clone(), mk.batched),
            (VERIFY_MS_MIXEDK.to_vec(), MTP_COST_MIXEDK, MTP_COST_RESTRICTED_MIXEDK, None)
        );
        assert_eq!(CostTables::for_group_for(1, true, HeadForm::MixedK), (mk.clone(), CostBasis::SingleNode));
        assert_eq!(CostTables::for_group_for(3, true, HeadForm::MixedK), (mk.clone(), CostBasis::Unmeasured { ranks: 3 }));
        let (tp, basis) = CostTables::for_group_for(2, true, HeadForm::MixedK);
        assert_eq!(basis, CostBasis::MeasuredTwoRanks);
        tp.validate().unwrap();
        assert_eq!(tp.verify, scaled_verify(VERIFY_MS_TP, VERIFY_MS_MIXEDK, VERIFY_MS));
        assert_eq!(tp.mtp, MTP_COST_TP.scaled_per_term(&MTP_COST_MIXEDK, &MTP_COST));
        assert_eq!(tp.batched, None);
        let rep = CostTables::tp_two_ranks_for(false, HeadForm::MixedK);
        assert_eq!((rep.mtp, rep.mtp_restricted), (MTP_COST_MIXEDK, MTP_COST_RESTRICTED_MIXEDK));
        // until measured, the placeholders are their sources exactly (pasting a measurement without setting the flag fails here), and
        // the form's group tables are then the 4-bit head's
        if !MIXEDK_MEASURED {
            assert_eq!(VERIFY_MS_MIXEDK, VERIFY_MS);
            assert_eq!((MTP_COST_MIXEDK, MTP_COST_RESTRICTED_MIXEDK), (MTP_COST_H4, MTP_COST_RESTRICTED_H4));
            assert_eq!(tp, CostTables { batched: None, ..CostTables::tp_two_ranks_for(true, HeadForm::Exl3) });
            assert_eq!(tp.verify, VERIFY_MS_TP.to_vec(), "a ratio of one keeps every point bit for bit");
        } else {
            // measured: its own tables (not the placeholders), the group's verify table scaled per row by the form's measured ratio
            assert_ne!(VERIFY_MS_MIXEDK, VERIFY_MS);
            assert_ne!((MTP_COST_MIXEDK, MTP_COST_RESTRICTED_MIXEDK), (MTP_COST_H4, MTP_COST_RESTRICTED_H4));
            let single = NemotronCost::new(&mk, true).unwrap();
            let ours = NemotronCost::new(&CostTables::shipped_for(HeadForm::Exl3), true).unwrap();
            for (&(r, a), &(r2, b)) in tp.verify.iter().zip(VERIFY_MS_TP.iter()) {
                let want = b * single.verify_ms(r) / ours.verify_ms(r);
                assert!(r == r2 && (a - want).abs() < 1e-9, "{r} rows: {a} against {want}");
            }
        }
        // the scaled verify table follows a measured single-node table per row
        let slower: Vec<(usize, f64)> = VERIFY_MS.iter().map(|&(r, t)| (r, t * 1.25)).collect();
        for (&(r, a), &(r2, b)) in scaled_verify(VERIFY_MS_TP, &slower, VERIFY_MS).iter().zip(VERIFY_MS_TP.iter()) {
            assert!(r == r2 && (a - 1.25 * b).abs() < 1e-9, "{r}: {a} {b}");
        }
        // a sub-table of bf16 rows is priced along the bf16 line from the restricted model, never from an EXL3 full lm_head
        let r = &MTP_COST_RESTRICTED_MIXEDK;
        assert_eq!(r.restricted_sub_table_to(LM_HEAD_RESTRICTED_ROWS), *r);
        let full_rows = r.restricted_sub_table_to(LM_HEAD_ROWS);
        assert!((full_rows.lm_head / r.lm_head - MTP_COST.lm_head / MTP_COST_RESTRICTED.lm_head).abs() < 1e-12);
        assert!(r.restricted_sub_table_to(32_768).lm_head < r.lm_head);
    }

    /// The online scales stay inside `SCALE_RANGE` whatever the MixedK tables are off by (they were placeholders until measured): a
    /// verify step three times the table (the reference kernels of checkpoint 1 were) clamps at 1.6, and a draft call at 0.4 at 0.7.
    #[test]
    fn the_online_scales_clamp_a_placeholder_that_is_far_off() {
        let mut c = NemotronCost::new(&CostTables::shipped_for(HeadForm::MixedK), true).unwrap();
        for _ in 0..200 {
            c.observe_window(4, 3.0 * VERIFY_MS_MIXEDK.iter().find(|p| p.0 == 4).unwrap().1);
            c.observe_draft(1, 1, 3, 0.4 * MTP_COST_RESTRICTED_MIXEDK.ms(1, 1, 3));
        }
        assert_eq!((c.verify_scale, c.mtp_scale), (SCALE_RANGE.1, SCALE_RANGE.0));
        let table4 = VERIFY_MS_MIXEDK.iter().find(|p| p.0 == 4).unwrap().1;
        assert!((c.verify_ms(4) - SCALE_RANGE.1 * table4).abs() < 1e-9);
    }

    #[test]
    fn the_fitted_tables_reproduce_the_measured_draft_calls() {
        for (cost, grid, what, worst_ok, mean_ok) in [
            (&MTP_COST, MEASURED, "full head", 0.075, 0.025),
            (&MTP_COST_RESTRICTED, MEASURED_RESTRICTED, "48k restricted head", 0.08, 0.035),
            (&MTP_COST_H4, MEASURED_H4, "4-bit head, full lm_head", 0.045, 0.02),
            (&MTP_COST_RESTRICTED_H4, MEASURED_RESTRICTED_H4, "4-bit head, 48k restricted head", 0.06, 0.025),
            (&MTP_COST_MIXEDK, MEASURED_MIXEDK, "MixedK head, full EXL3 lm_head", 0.045, 0.02),
            (&MTP_COST_RESTRICTED_MIXEDK, MEASURED_RESTRICTED_MIXEDK, "MixedK head, 48k restricted head", 0.075, 0.025),
        ] {
            let (mut worst, mut sum, mut n) = (0.0f64, 0.0f64, 0);
            for ((asks, depth), times) in grid {
                for (rows, measured) in ROWS_PER_ASK.into_iter().zip(times) {
                    let model = cost.ms(asks, asks * rows, depth);
                    let err = (model - measured).abs() / measured;
                    assert!(
                        err < worst_ok,
                        "{what}: {asks} asks x {rows} rows x depth {depth}: {model:.2} ms against {measured} measured"
                    );
                    (worst, sum, n) = (worst.max(err), sum + err, n + 1);
                }
            }
            assert_eq!(n, 40, "{what}");
            let mean = sum / n as f64;
            assert!(mean < mean_ok, "{what}: mean error {mean:.3}, worst {worst:.3}");
        }
        // the stage profile of one ask, one row, depth 1: the lm_head is 4.63 ms of 7.2 (1.74 of 4.2 restricted)
        assert_eq!((MTP_COST.lm_head, MTP_COST_RESTRICTED.lm_head), (4.63, 1.74));
        assert!((MTP_COST.ms(1, 1, 1) - 7.2).abs() < 0.5 && (MTP_COST_RESTRICTED.ms(1, 1, 1) - 4.2).abs() < 0.3);
        // the restricted head saves its lm_head in every step: 1 ask x depth 3 went 21.8 -> 12.2 ms
        assert!((MTP_COST.ms(1, 1, 3) - MTP_COST_RESTRICTED.ms(1, 1, 3) - 3.0 * (4.63 - 1.74)).abs() < 1.0);
        // the layers do not depend on the vocabulary: the two fits agree on the blocks to within the noise of a fit
        assert!(
            (MTP_COST.step - MTP_COST_RESTRICTED.step).abs() < 0.4
                && (MTP_COST.step_per_stream - MTP_COST_RESTRICTED.step_per_stream).abs() < 0.15
        );
        for (&(r, a), &(r2, b)) in MTP_COST.rows.iter().zip(MTP_COST_RESTRICTED.rows.iter()) {
            assert!(r == r2 && (a - b).abs() < 0.7, "R {r}: {a} against {b}");
        }
        // a restricted head of another size moves only that term, linearly between the two measured sizes
        let r64 = MTP_COST.restricted_to(65_536);
        assert!(r64.lm_head > MTP_COST_RESTRICTED.lm_head && r64.lm_head < MTP_COST.lm_head, "{}", r64.lm_head);
        assert!((MTP_COST.restricted_to(LM_HEAD_RESTRICTED_ROWS).lm_head - MTP_COST_RESTRICTED.lm_head).abs() < 1e-9);
        // the line through the full and the 48k head predicts the 32k head, measured once at 1.22 ms (stage profile)
        assert!((MTP_COST.restricted_to(32_768).lm_head - 1.22).abs() < 0.1, "{}", MTP_COST.restricted_to(32_768).lm_head);
        assert!((MTP_COST.restricted_to(LM_HEAD_ROWS).lm_head - MTP_COST.lm_head).abs() < 1e-9);
        assert_eq!((r64.step, r64.rows.clone()), (MTP_COST.step, MTP_COST.rows.clone()));
    }

    #[test]
    fn the_group_tables_reproduce_the_served_fixed_depth_rounds_and_sit_under_the_single_node_ones() {
        // served rounds of the two-node group, the head split, 1 stream: the draft call at depth 2 to 6 (restricted 48k head)
        for (d, ms) in [(2usize, 7.89f64), (3, 10.83), (4, 14.00), (5, 16.45), (6, 19.44)] {
            let model = MTP_COST_RESTRICTED_TP.ms(1, 1, d);
            assert!((model - ms).abs() / ms < 0.03, "depth {d}: {model:.2} ms against {ms} measured");
            assert!(model < MTP_COST_RESTRICTED.ms(1, 1, d), "depth {d}: the group's call is cheaper than the single node's");
        }
        // the verify phase at the window rows those rounds ran (mean rows 2.99 / 3.96 / 4.95 / 5.90 / 6.88)
        let tp = CostTables::tp_two_ranks(true);
        tp.validate().unwrap();
        let c = NemotronCost::new(&tp, true).unwrap();
        for (rows, ms) in [(3usize, 41.48f64), (4, 45.96), (5, 51.08), (6, 53.90), (7, 58.09)] {
            let model = c.verify_ms(rows);
            assert!((model - ms).abs() / ms < 0.02, "{rows} rows: {model:.2} ms against {ms} measured");
        }
        // ascending, under the single node's at every row (the ranks each read half), and the collectives' tax grows with the rows
        assert!(VERIFY_MS_TP.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 < w[1].1));
        let single = NemotronCost::new(&CostTables::shipped(), true).unwrap();
        for &(rows, ms) in VERIFY_MS_TP {
            assert!(ms < single.verify_ms(rows), "{rows} rows");
        }
        // a head that is not split on both parts keeps the single node's draft tables and takes the group's verify table
        let rep = CostTables::tp_two_ranks(false);
        assert_eq!((rep.mtp.clone(), rep.mtp_restricted.clone()), (MTP_COST, MTP_COST_RESTRICTED));
        assert_eq!(rep.verify, VERIFY_MS_TP.to_vec());
        // the restricted head of another size moves the lm_head term of the group's table only, by the same line
        let r = MTP_COST_TP.restricted_to(LM_HEAD_RESTRICTED_ROWS);
        assert!((r.lm_head - MTP_COST_TP.lm_head * MTP_COST_RESTRICTED.lm_head / MTP_COST.lm_head).abs() < 1e-9);
    }

    #[test]
    fn the_arbiters_acceptance_priors_are_the_measured_ones() {
        let mut a = Arbiter::new(arbiter_config(&SpecConfig::new(SpecMode::Auto)));
        for (j, want) in [(1usize, 0.936f64), (2, 0.917), (3, 0.897), (4, 0.866)] {
            let got = a.mtp_alpha(0, j);
            assert!((got - want).abs() < 0.02, "depth {j}: prior {got:.3} against {want} measured");
        }
        // the 48k restricted head (MT-A, the same prompts): 0.928 0.912 0.893 0.860, a little under the full head's at every depth
        let cfg = SpecConfig { restricted_head: true, ..SpecConfig::new(SpecMode::Auto) };
        let mut r = Arbiter::new(arbiter_config(&cfg));
        for (j, want) in [(1usize, 0.928f64), (2, 0.912), (3, 0.893), (4, 0.860)] {
            let got = r.mtp_alpha(0, j);
            assert!((got - want).abs() < 0.02, "restricted depth {j}: prior {got:.3} against {want} measured");
            assert!(got < a.mtp_alpha(0, j), "a restricted head's prior is below the full head's at depth {j}");
        }
        // a runtime whose config moves to the restricted head follows it for the sequences that start after
        let mut rt = SpecRuntime::new(SpecConfig::new(SpecMode::Auto), 1);
        rt.cfg.restricted_head = true;
        rt.follow_cfg();
        rt.begin(0);
        assert!((rt.arbiter.mtp_alpha(0, 1) - 0.928).abs() < 1e-9);
    }

    #[test]
    fn the_tables_have_the_shape_the_arbiter_relies_on() {
        for (m, what) in [(MTP_COST, "full head"), (MTP_COST_RESTRICTED, "restricted head")] {
            m.validate(what).unwrap();
            // a draft call costs more for more streams, more pending rows and more depth
            assert!(m.ms(2, 4, 2) > m.ms(1, 4, 2) && m.ms(1, 8, 2) > m.ms(1, 4, 2) && m.ms(1, 4, 3) > m.ms(1, 4, 2), "{what}");
            assert!(m.ms(1, 4, 0) < m.ms(1, 4, 1), "{what}: an absorb alone runs no lm_head");
            // concave in the pending rows, like the window itself, and past the table the last slope goes on
            assert!(m.ms(1, 2, 0) - m.ms(1, 1, 0) > m.ms(1, 64, 0) - m.ms(1, 63, 0), "{what}");
            assert!(m.ms(1, 80, 1) > m.ms(1, 64, 1), "{what}");
            assert!(m.rows_ms(0) == 0.0, "zero rows cost nothing");
        }
        // the restricted head saves its share of every head the call runs: one in the first draft, one in each chained draft
        assert!(MTP_COST_RESTRICTED.ms(1, 3, 2) < MTP_COST.ms(1, 3, 2) - 3.0);
        assert!(
            (MTP_COST_RESTRICTED.ms(1, 3, 0) - MTP_COST.ms(1, 3, 0)).abs() < 0.5,
            "an absorb alone runs no head: the two fits differ in the blocks only"
        );
        // the verify table is the measured one, ascending from one row, read as it is
        let c = NemotronCost::new(&CostTables::shipped(), false).unwrap();
        for (rows, ms) in [
            (1usize, 38.0),
            (2, 46.0),
            (3, 51.0),
            (4, 57.0),
            (5, 63.0),
            (6, 67.0),
            (7, 71.0),
            (8, 75.0),
            (16, 84.0),
            (32, 112.0),
            (64, 202.0),
        ] {
            assert!((c.verify_ms(rows) - ms).abs() < 1e-9, "{rows} rows");
        }
        assert!(VERIFY_MS.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 < w[1].1));
        // a window of four rows costs 1.5 serial steps, not four: the arbiter's trade is real, and the experts the rows share make it
        // concave (the table before the MoE work was nearly linear: the arbiter reads whichever table it is given)
        assert!(c.verify_ms(4) < 2.0 * c.verify_ms(1) && c.verify_ms(4) > c.verify_ms(1));
        assert!(c.verify_ms(2) - c.verify_ms(1) > c.verify_ms(9) - c.verify_ms(8));
        // between the points it interpolates
        assert!((c.verify_ms(12) - (75.0 + 84.0) / 2.0).abs() < 1e-9);
        // a draft round of three at the prior pays against serial on these numbers: 3 drafts cost less than the tokens they buy
        assert!(c.mtp_ms(1, 1, 3) < 3.0 * c.verify_ms(1));
    }

    #[test]
    fn the_cost_model_scales_with_what_was_observed_within_bounds() {
        let mut c = NemotronCost::new(&CostTables::shipped(), false).unwrap();
        // a window slower than the table moves the scale up, by a smoothed step, to a limit
        for _ in 0..200 {
            c.observe_window(3, 52.0 * 10.0);
        }
        assert!((c.verify_scale - SCALE_RANGE.1).abs() < 1e-9, "{}", c.verify_scale);
        for _ in 0..200 {
            c.observe_window(3, 1.0);
        }
        assert!((c.verify_scale - SCALE_RANGE.0).abs() < 1e-9);
        // nonsense is ignored
        let s = c.verify_scale;
        c.observe_window(3, f64::NAN);
        c.observe_window(3, -4.0);
        assert_eq!(c.verify_scale, s);
        // the draft model's scale moves on its own
        let m = c.mtp_scale;
        for _ in 0..200 {
            c.observe_draft(2, 4, 2, 1e6);
        }
        assert!(c.mtp_scale > m && (c.mtp_scale - SCALE_RANGE.1).abs() < 1e-9);
        assert_eq!(c.verify_scale, s);
        // what the arbiter reads is the table times its scale, for both
        assert!((c.mtp_ms(2, 4, 2) - MTP_COST.ms(2, 4, 2) * c.mtp_scale).abs() < 1e-9);
        assert!((c.verify_ms(4) - 57.0 * c.verify_scale).abs() < 1e-9);
    }

    #[test]
    fn measured_tables_replace_the_shipped_ones_and_a_malformed_table_is_refused() {
        let mut r = SpecRuntime::new(SpecConfig::new(SpecMode::Auto), 1);
        let before = r.cost.verify_ms(8);
        let measured = CostTables::shipped().with_verify(&[(1, 20.0), (8, 30.0), (64, 100.0)]);
        r.set_costs(&measured).unwrap();
        assert!((r.cost.verify_ms(1) - 20.0).abs() < 1e-9 && r.cost.verify_ms(8) < before);
        // the draft tables are replaced too, and the restricted flag picks the table in use
        let mut cheap = MTP_COST;
        cheap.lm_head = 0.5;
        let t = CostTables { mtp: cheap, ..CostTables::shipped() };
        r.set_costs(&t).unwrap();
        assert!((r.cost.mtp_ms(1, 1, 1) - (0.10 + 0.60 + 2.12 + 0.5)).abs() < 1e-9);
        r.cfg.restricted_head = true;
        r.follow_cfg();
        assert!((r.cost.mtp_ms(1, 1, 1) - (0.10 + 0.57 + 1.82 + 1.74)).abs() < 1e-9);
        // malformed tables are refused and the old ones stay
        let keep = r.cost.verify_ms(8);
        for bad in [
            CostTables::shipped().with_verify(&[(2, 20.0)]),
            CostTables::shipped().with_verify(&[(1, 20.0), (1, 21.0)]),
            CostTables::shipped().with_verify(&[(1, f64::NAN)]),
            CostTables { mtp: MtpCost { rows: Cow::Owned(vec![(4, 1.0), (2, 2.0)]), ..MTP_COST }, ..CostTables::shipped() },
            CostTables { mtp_restricted: MtpCost { lm_head: -1.0, ..MTP_COST_RESTRICTED }, ..CostTables::shipped() },
            CostTables { mtp: MtpCost { rows: Cow::Owned(vec![]), ..MTP_COST }, ..CostTables::shipped() },
        ] {
            assert!(r.set_costs(&bad).is_err(), "{bad:?}");
            assert_eq!(r.cost.verify_ms(8), keep);
        }
    }

    // ---- the batched verify table (`REXL3_NEMO_ARB_TABLE=batched`)

    /// The drafting rounds [`VERIFY_BATCHED`] was fitted to (`tools/nemotron_arb_table.py`; nodes 1 and 4, 2026-10-16, the v0.1.0 binary
    /// and the A/B binary, 16 slots: `--spec auto` at caps 2 to 5 under both tables, `--spec mtp` at depths 2 and 3, the perf client's
    /// one-stream latency phase, ladder and 16-stream bench, greedy and sampled): (streams, rounds, rows a round, verify ms a round), pooled
    /// by the width of the round.
    const SERVED: &[(usize, f64, f64, f64)] = &[
        (1, 24531.0, 3.392, 52.49),
        (2, 9267.0, 7.458, 79.66),
        (3, 2253.0, 10.664, 100.67),
        (4, 7009.0, 15.483, 123.77),
        (5, 930.0, 16.891, 133.71),
        (6, 745.0, 19.817, 148.58),
        (7, 1071.0, 24.303, 166.54),
        (8, 7929.0, 29.362, 182.99),
        (9, 605.0, 29.572, 185.41),
        (10, 582.0, 31.289, 192.89),
        (11, 722.0, 34.474, 202.92),
        (12, 6638.0, 36.491, 209.15),
        (13, 423.0, 36.657, 212.32),
        (14, 572.0, 35.719, 212.42),
        (15, 1060.0, 34.592, 211.53),
        (16, 20604.0, 35.116, 213.16),
    ];

    /// The serial rounds of `--spec none` (node 4, ladder and bench), whose verify step holds the commits: (streams, rounds, verify ms).
    const SERVED_SERIAL: &[(usize, f64, f64)] =
        &[(1, 801.0, 36.78), (2, 792.0, 49.62), (4, 780.0, 73.31), (8, 762.0, 114.55), (12, 744.0, 151.49), (16, 1440.0, 177.61)];

    /// The window of [`BatchedVerify`] at a mean of rows (the served points are means over rounds).
    fn window_at(b: &BatchedVerify, streams: usize, rows: f64) -> f64 {
        let n = streams as f64;
        b.floor
            + b.experts * (1.0 - (1.0 - b.p_base).powi(streams as i32) * (1.0 - b.p_draft).powf(rows - n))
            + b.per_draft_row * (rows - n)
    }

    #[test]
    fn the_batched_table_reproduces_the_served_rounds() {
        let b = VERIFY_BATCHED;
        b.validate("shipped").unwrap();
        // the fraction a base row picks is the architecture's: 22 of the 512 routed experts
        assert_eq!(b.p_base, 22.0 / 512.0);
        assert!(b.p_draft < b.p_base, "a draft row shares experts with its own sequence's base row");
        let (mut sse, mut weight, mut worst) = (0.0f64, 0.0f64, 0.0f64);
        for &(n, rounds, rows, ms) in SERVED {
            let err = window_at(&b, n, rows) - ms;
            assert!(err.abs() < 2.5, "{n} streams, {rows:.2} rows a round: {:.2} ms against {ms} served", window_at(&b, n, rows));
            (sse, weight, worst) = (sse + rounds * err * err, weight + rounds, worst.max(err.abs()));
        }
        let rms = (sse / weight).sqrt();
        assert!(rms < 1.2, "round-weighted rms {rms:.2} ms (worst {worst:.2})");
        // the serial rounds of two streams and more: the window and the commits (a lone sequence is the single table's: see
        // `a_lone_sequence_is_priced_by_the_single_table_under_either_setting`)
        for &(n, _, ms) in SERVED_SERIAL.iter().filter(|p| p.0 >= 2) {
            assert!((b.ms(n, n) - ms).abs() / ms < 0.05, "{n} serial streams: {:.2} ms against {ms} served", b.ms(n, n));
        }
        // the whole rows are the same formula
        for (n, rows) in [(1usize, 1usize), (4, 13), (16, 32), (16, 64)] {
            assert!((b.window_ms(n, rows) - window_at(&b, n, rows as f64)).abs() < 1e-9);
        }
        // the 16-stream round at 32 rows: the single table prices it at 112 ms, near half of what it took
        assert!(VERIFY_MS.contains(&(32, 112.0)) && 112.0 < 0.6 * b.window_ms(16, 32));
        // the whole 16-stream round (verify, draft call, commits) of the fixed-depth runs (node 4): serial, one, two and three deep
        let round16 = |d: usize| {
            b.ms(16, 16 * (1 + d)) + if d == 0 { 0.0 } else { MTP_COST_RESTRICTED.ms(16, 32, d) + b.head_extra_ms(16, d) }
        };
        for (d, served) in [(0usize, 177.7f64), (1, 246.7), (2, 297.7), (3, 342.6)] {
            assert!((round16(d) - served).abs() / served < 0.01, "16 streams, depth {d}: {:.1} ms against {served}", round16(d));
        }
    }

    #[test]
    fn the_batched_table_prices_a_draft_row_at_its_marginal_cost_in_a_step_of_many_sequences() {
        let b = VERIFY_BATCHED;
        for n in [1usize, 2, 4, 8, 16] {
            for rows in n..64 {
                // dearer for every row, and the same rows over one more sequence dearer still (a base row picks more new experts than a
                // draft row, and its sequence commits)
                assert!(b.ms(n, rows + 1) > b.ms(n, rows), "{n} streams, {rows} rows");
                assert!(b.ms(n + 1, rows + 1) > b.ms(n, rows + 1), "{n} streams, {rows} rows");
                // concave in the draft rows: the experts the step has read are not read again
                if rows > n {
                    assert!(b.ms(n, rows + 1) - b.ms(n, rows) < b.ms(n, rows) - b.ms(n, rows - 1), "{n} streams, {rows} rows");
                }
            }
        }
        // a draft row's marginal cost falls with the sequences in the step, by the experts they have read: its expert part at 16 streams is
        // (1 - p_base)^15 = 0.52 of a lone stream's; its own work is the same
        let first = |n: usize| b.ms(n, n + 1) - b.ms(n, n) - b.per_draft_row;
        assert!((first(16) / first(1) - (1.0 - b.p_base).powi(15)).abs() < 1e-9, "{:.2} against {:.2}", first(16), first(1));
        // and as a share of the round it is far smaller than the single table says: 16 streams drafting one each (16 -> 32 rows) adds a
        // third to the single table's step, a quarter to the batched one
        let mut single = NemotronCost::new(&CostTables::shipped(), true).unwrap();
        single.table = ArbTable::Single;
        let (s0, s1) = (single.verify_ms_of(16, 16), single.verify_ms_of(16, 32));
        let (b0, b1) = (b.ms(16, 16), b.ms(16, 32));
        assert!((s1 / s0 - 4.0 / 3.0).abs() < 1e-9, "{s0} {s1}");
        assert!(b1 / b0 < 1.3, "{b0:.1} -> {b1:.1}");
        // a count of sequences above the rows is read as the rows (each sequence holds its base row); no rows is the floor
        assert_eq!(b.ms(20, 16), b.ms(16, 16));
        assert_eq!(b.window_ms(0, 0), b.floor);
        // the head terms: per stream of the call, and per stream and chained step
        assert_eq!(b.head_extra_ms(16, 0), 16.0 * b.head_per_stream);
        assert_eq!(b.head_extra_ms(16, 1), 16.0 * b.head_per_stream);
        assert!((b.head_extra_ms(16, 3) - 16.0 * (b.head_per_stream + 2.0 * b.head_per_stream_step)).abs() < 1e-12);
    }

    #[test]
    fn single_prices_every_round_as_v010_did_and_batched_only_when_asked_and_measured() {
        let mut c = NemotronCost::new(&CostTables::shipped(), true).unwrap();
        assert_eq!((c.table, c.batched), (ArbTable::Single, Some(VERIFY_BATCHED)));
        (c.verify_scale, c.mtp_scale) = (1.3, 1.2);
        for n in 1..=16usize {
            for rows in n..=64 {
                assert_eq!(c.verify_ms_of(n, rows).to_bits(), c.verify_ms(rows).to_bits(), "{n} streams, {rows} rows");
            }
            for d in 0..=3 {
                assert_eq!(c.mtp_ms(n, 2 * n, d).to_bits(), (c.mtp().ms(n, 2 * n, d) * 1.2).to_bits(), "{n} streams, depth {d}");
            }
        }
        // batched, two sequences and more: the window times its own online scale and the commits; the draft call with its head terms
        c.table = ArbTable::Batched;
        c.batched_verify_scale = 1.1;
        c.batched_mtp_scale = 0.9;
        let b = VERIFY_BATCHED;
        assert!((c.verify_ms_of(16, 40) - (b.window_ms(16, 40) * 1.1 + b.commit_ms(16))).abs() < 1e-9);
        let m = MTP_COST_RESTRICTED.ms(16, 32, 3);
        assert!((c.mtp_ms(16, 32, 3) - (m + b.head_extra_ms(16, 3)) * 0.9).abs() < 1e-9);
        // a two-rank group has no batched table: `batched` prices with the group's table by rows
        let tp = CostTables::tp_two_ranks(true);
        assert!(tp.batched.is_none() && CostTables::shipped().batched == Some(VERIFY_BATCHED));
        let mut g = NemotronCost::new(&tp, true).unwrap();
        g.table = ArbTable::Batched;
        assert!(g.batched_in_use().is_none() && g.batched_for(16).is_none());
        assert_eq!(g.verify_ms_of(16, 32).to_bits(), g.verify_ms(32).to_bits());
        // a malformed batched table is refused like the others
        for bad in [
            BatchedVerify { p_draft: 1.5, ..VERIFY_BATCHED },
            BatchedVerify { p_base: 0.0, ..VERIFY_BATCHED },
            BatchedVerify { floor: -1.0, ..VERIFY_BATCHED },
            BatchedVerify { commit_per_stream: f64::NAN, ..VERIFY_BATCHED },
            BatchedVerify { per_draft_row: -0.1, ..VERIFY_BATCHED },
            BatchedVerify { head_per_stream_step: f64::INFINITY, ..VERIFY_BATCHED },
            BatchedVerify { min_streams: 0, ..VERIFY_BATCHED },
        ] {
            let t = CostTables { batched: Some(bad), ..CostTables::shipped() };
            assert!(t.validate().is_err() && NemotronCost::new(&t, true).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_lone_sequence_is_priced_by_the_single_table_under_either_setting() {
        // the one-sequence table was measured on exactly this round; the union fit prices a serial one-stream round over what it took
        let lone = SERVED_SERIAL.iter().find(|p| p.0 == 1).unwrap();
        assert!(VERIFY_BATCHED.ms(1, 1) > lone.2 + 2.0 && (VERIFY_MS[0].1 - lone.2).abs() < 2.0, "{lone:?}");
        let mut x = NemotronCost::new(&CostTables::shipped(), true).unwrap();
        let mut y = NemotronCost::new(&CostTables::shipped(), true).unwrap();
        (x.table, y.table) = (ArbTable::Single, ArbTable::Batched);
        assert!(VERIFY_BATCHED.min_streams == 2 && y.batched_for(1).is_none() && y.batched_for(2).is_some());
        // pricing: the same bits for every row count and depth, whatever the scales say
        for (cost, v, m) in [(&mut x, 1.3, 0.8), (&mut y, 1.3, 0.8)] {
            (cost.verify_scale, cost.mtp_scale, cost.batched_verify_scale, cost.batched_mtp_scale) = (v, m, 1.5, 1.4);
        }
        for rows in 1..=8 {
            assert_eq!(x.verify_ms_of(1, rows).to_bits(), y.verify_ms_of(1, rows).to_bits(), "{rows} rows");
        }
        for d in 0..=3 {
            assert_eq!(x.mtp_ms(1, 2, d).to_bits(), y.mtp_ms(1, 2, d).to_bits(), "depth {d}");
        }
        // learning: a lone sequence's window and draft call move the single scales, the batched ones are left to the wide rounds
        let (vb, mb) = (y.batched_verify_scale, y.batched_mtp_scale);
        for cost in [&mut x, &mut y] {
            cost.observe_window_of(1, 3, 60.0);
            cost.observe_draft(1, 2, 2, 12.0);
        }
        assert_eq!((x.verify_scale.to_bits(), x.mtp_scale.to_bits()), (y.verify_scale.to_bits(), y.mtp_scale.to_bits()));
        assert_eq!((y.batched_verify_scale, y.batched_mtp_scale), (vb, mb));
        // and so the arbiter picks for one stream exactly what it picks under single, at every acceptance
        for (first, chain) in [(0.95, 0.98), (0.87, 1.0), (0.81, 0.89), (0.7, 1.0), (0.65, 1.0), (0.6, 1.0), (0.4, 1.0)] {
            assert_eq!(
                auto_picks(1, ArbTable::Batched, first, chain),
                auto_picks(1, ArbTable::Single, first, chain),
                "acceptance {first}, chain {chain}"
            );
        }
    }

    #[test]
    fn the_online_scales_measure_the_table_in_use() {
        let b = VERIFY_BATCHED;
        // single: the window's sequences do not matter, the scale moves exactly as `observe_window` moves it
        let (mut x, mut y) =
            (NemotronCost::new(&CostTables::shipped(), true).unwrap(), NemotronCost::new(&CostTables::shipped(), true).unwrap());
        for (n, rows, ms) in [(16usize, 32usize, 207.0f64), (1, 4, 55.0), (8, 31, 187.0)] {
            x.observe_window_of(n, rows, ms);
            y.observe_window(rows, ms);
            x.observe_draft(n, n * 2, 1, 40.0);
            y.observe_draft(n, n * 2, 1, 40.0);
        }
        assert_eq!((x.verify_scale.to_bits(), x.mtp_scale.to_bits()), (y.verify_scale.to_bits(), y.mtp_scale.to_bits()));
        assert_eq!((x.batched_verify_scale, x.batched_mtp_scale), (1.0, 1.0), "single never moves the batched scales");
        // a 16-stream round as the batched table predicts it leaves its scales alone (the single one would be pushed to its clamp)
        let mut c = NemotronCost::new(&CostTables::shipped(), true).unwrap();
        c.table = ArbTable::Batched;
        for _ in 0..50 {
            c.observe_window_of(16, 32, b.window_ms(16, 32));
            // the draft call waits for the round's commits: a call that took the draft model, its head terms and the commits is as predicted
            c.observe_draft(16, 30, 1, MTP_COST_RESTRICTED.ms(16, 30, 1) + b.head_extra_ms(16, 1) + b.commit_ms(16));
        }
        assert!((c.batched_verify_scale - 1.0).abs() < 1e-9 && (c.batched_mtp_scale - 1.0).abs() < 1e-9);
        assert_eq!((c.verify_scale, c.mtp_scale), (1.0, 1.0), "the single scales are the lone sequences'");
        for _ in 0..50 {
            x.observe_window_of(16, 32, b.window_ms(16, 32));
        }
        assert!((x.verify_scale - SCALE_RANGE.1).abs() < 1e-9, "{}", x.verify_scale);
    }

    #[test]
    fn the_table_setting_is_validated_and_the_runtime_follows_it() {
        assert_eq!(SpecConfig::new(SpecMode::Auto).arb_table, ArbTable::Single);
        assert_eq!(ARB_TABLE_DEFAULT, ArbTable::Single);
        assert_eq!(parse_arb_table(None), Ok(ArbTable::Single));
        assert_eq!(parse_arb_table(Some("single")), Ok(ArbTable::Single));
        assert_eq!(parse_arb_table(Some(" batched ")), Ok(ArbTable::Batched));
        assert_eq!(parse_arb_table(Some("BATCHED")), Ok(ArbTable::Batched));
        for bad in ["", "1", "batch", "multi", "single,batched"] {
            let e = parse_arb_table(Some(bad)).unwrap_err();
            assert!(e.contains("REXL3_NEMO_ARB_TABLE") && e.contains("single or batched"), "{bad:?}: {e}");
        }
        assert_eq!((ArbTable::Single.label(), ArbTable::Batched.label()), ("single", "batched"));
        // the runtime prices with the configured table, follows a change of it, and keeps it when measured tables replace the shipped ones
        let mut rt = SpecRuntime::new(SpecConfig { arb_table: ArbTable::Batched, ..SpecConfig::new(SpecMode::Auto) }, 2);
        assert_eq!(rt.cost.table, ArbTable::Batched);
        rt.set_costs(&CostTables::shipped().with_verify(&[(1, 20.0), (64, 100.0)])).unwrap();
        assert_eq!(rt.cost.table, ArbTable::Batched);
        rt.cfg.arb_table = ArbTable::Single;
        rt.follow_cfg();
        assert!(rt.cost.batched_in_use().is_none());
        // the window's sequences reach the cost model
        rt.cfg.arb_table = ArbTable::Batched;
        rt.follow_cfg();
        rt.on_window(16, 32, VERIFY_BATCHED.window_ms(16, 32) * 1.2);
        assert!(
            rt.cost.batched_verify_scale > 1.0
                && rt.cost.verify_scale == 1.0
                && rt.stats.windows == 1
                && rt.stats.window_rows == 32
        );
    }

    /// The arbiter of `--spec auto` as a boot builds it (the restricted head's draft costs, the cap of 3) over `n` warm streams whose
    /// acceptance is pinned: `first` for the first draft and each further depth `chain` of the one before (0.867 and 0.95: 0.867 / 0.824 /
    /// 0.783, the measured rates of rounds of 2+ streams were 0.867 / 0.848 / 0.783; 0.81 and 0.89: the one-stream filler prompts of the
    /// R-TABLE A/B, 0.81 / 0.72 / 0.66 measured).
    fn auto_picks(n: usize, table: ArbTable, first: f64, chain: f64) -> Vec<Choice> {
        let cfg = SpecConfig { restricted_head: true, arb_table: table, ..SpecConfig::new(SpecMode::Auto) };
        let mut rt = SpecRuntime::new(cfg, n);
        rt.arbiter.cfg.mtp_prior = vec![first];
        rt.arbiter.cfg.mtp_chain = chain;
        rt.arbiter.cfg.prior_weight = 1e9;
        rt.arbiter.cfg.warmup_rounds = 0;
        let asks: Vec<Ask> = (0..n)
            .map(|s| {
                rt.arbiter.begin(s);
                Ask { stream: s, max: 7, copy_len: 0, mtp_ok: true, absorb_rows: 2, must_absorb: false }
            })
            .collect();
        rt.arbiter.choose(&rt.cost, &asks)
    }

    fn drafts(p: &[Choice]) -> usize {
        p.iter().map(|c| c.drafts()).sum()
    }

    #[test]
    fn the_batched_table_drafts_deeper_at_eight_to_sixteen_streams_and_single_drafts_as_it_served() {
        for (first, chain) in [(0.867, 0.95), (0.818, 0.9)] {
            let (s4, b4) = (auto_picks(4, ArbTable::Single, first, chain), auto_picks(4, ArbTable::Batched, first, chain));
            let (s16, b16) = (auto_picks(16, ArbTable::Single, first, chain), auto_picks(16, ArbTable::Batched, first, chain));
            // four streams: every stream drafts under both
            assert!(s4.iter().chain(&b4).all(|c| matches!(c, Choice::Mtp(d) if *d >= 2)), "{first}: {s4:?} {b4:?}");
            // sixteen: the single table holds every stream at one draft, the 32 rows of its kink (what the served rounds did: 8,665 of
            // them, rows 31.97 a round); the batched one prices the 16 base rows at what they cost and drafts deeper
            assert!(s16.iter().all(|c| *c == Choice::Mtp(1)), "{first}: single at 16 streams {s16:?}");
            assert!(drafts(&b16) > drafts(&s16), "{first}: batched {} drafts against single {}", drafts(&b16), drafts(&s16));
            assert!(b16.iter().all(|c| c.drafts() >= 1) && drafts(&b16) + 16 <= WINDOW_ROWS, "{b16:?}");
        }
        // at the measured rates of the wide rounds, every stream of 8 and 16 drafts the whole cap (the fixed-depth runs: depth 3 was the
        // fastest at every width)
        for n in [8usize, 16] {
            assert!(auto_picks(n, ArbTable::Batched, 0.867, 0.95).iter().all(|c| *c == Choice::Mtp(3)), "{n} streams");
        }
    }

    /// Mean drafts per stream the arbiter picks (`auto_picks`) at flat acceptances: the report's table, and its limits.
    #[test]
    fn the_pick_table() {
        let mean = |n: usize, t: ArbTable, a: f64, chain: f64| drafts(&auto_picks(n, t, a, chain)) as f64 / n as f64;
        eprintln!("mean drafts per stream, single | batched, at 1 / 4 / 8 / 16 streams");
        for (a, chain) in [(0.4, 1.0), (0.6, 1.0), (0.7, 1.0), (0.87, 1.0), (0.81, 0.89), (0.867, 0.95)] {
            let row = |t| [1usize, 4, 8, 16].map(|n| format!("{:.2}", mean(n, t, a, chain))).join(" ");
            eprintln!("  acceptance {a} chain {chain}: {} | {}", row(ArbTable::Single), row(ArbTable::Batched));
            // never deeper than single at one stream; never deeper than single at 4 to 16 streams when a draft is a coin flip or worse
            assert_eq!(mean(1, ArbTable::Batched, a, chain), mean(1, ArbTable::Single, a, chain));
            if a <= 0.4 {
                for n in [4usize, 8, 16] {
                    assert!(mean(n, ArbTable::Batched, a, chain) <= mean(n, ArbTable::Single, a, chain), "{a}: {n} streams");
                }
            }
        }
    }

    /// The arbiter's view through a model that implements only the trait's two required methods: the pricing every model had before
    /// `ArbiterCost::verify_ms_of`.
    struct RowsOnly<'a>(&'a NemotronCost);
    impl ArbiterCost for RowsOnly<'_> {
        fn verify_ms(&self, rows: usize) -> f64 {
            self.0.verify_ms(rows)
        }
        fn mtp_ms(&self, streams: usize, absorb_rows: usize, depth: usize) -> f64 {
            self.0.mtp_ms(streams, absorb_rows, depth)
        }
    }

    #[test]
    fn single_picks_exactly_what_the_rows_only_pricing_picks() {
        for n in [1usize, 2, 3, 4, 8, 12, 16] {
            for (first, chain) in [(0.95, 0.98), (0.867, 0.95), (0.818, 0.9), (0.5, 0.9), (0.2, 0.8)] {
                let cfg = SpecConfig { restricted_head: true, ..SpecConfig::new(SpecMode::Auto) };
                let (mut x, mut y) = (SpecRuntime::new(cfg, n), SpecRuntime::new(cfg, n));
                for rt in [&mut x, &mut y] {
                    rt.arbiter.cfg.mtp_prior = vec![first];
                    rt.arbiter.cfg.mtp_chain = chain;
                    rt.arbiter.cfg.warmup_rounds = 0;
                }
                let asks: Vec<Ask> = (0..n)
                    .map(|s| Ask { stream: s, max: 7, copy_len: 0, mtp_ok: true, absorb_rows: 2, must_absorb: false })
                    .collect();
                let a = x.arbiter.choose(&x.cost, &asks);
                let b = y.arbiter.choose(&RowsOnly(&y.cost), &asks);
                assert_eq!(a, b, "{n} streams, acceptance {first} chain {chain}");
            }
        }
    }
}
