//! The DSpark draft inside the MiniCPM-V-4.7 engine (E4b of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`): everything that joins E4a's
//! stand-alone [`DraftModel`] to E3's text engine, and nothing of the target's arithmetic.
//!
//! # What the target gives the draft, and when
//!
//! * **Taps.** After layer `i` of a forward whose index is in the draft config's `target_layer_ids` (a runtime list: 3, 11, 19, 27, 35),
//!   the residual stream `x` (bf16 `[rows, 2048]`, which E3 already keeps in bf16: nothing is rounded) is copied into one tap buffer
//!   `[rows][taps][2048]`, for EVERY row the target processes: the rows of each prefill chunk and the rows of each verify window. The
//!   copy reads `x` and writes a buffer of its own; no bit of the target's output depends on it ([`SpecRig::export`]).
//! * **Prompt rows** are absorbed into the draft's pages at the end of each prefill chunk ([`SpecRig::after_prefill`]): every position of the
//!   prompt reaches the draft's context, whatever chunk grid or prefill path carried it.
//! * **Accepted rows** are not absorbed at once. `Backend::commit(slot, keep)` is lazy in this family (the recurrent state is replayed by
//!   the next forward), and so is this: when the commit is applied the `keep` accepted rows of the window's tap buffer are copied into the
//!   slot's small ring ([`SpecRig::commit_window`]), and the ring is absorbed (one `ingest_taps` call per slot) only when a round needs the
//!   draft ([`SpecRig::propose`]) or the ring is full. A stream that is not drafting therefore costs the draft one device copy a round, not
//!   a 0.3 ms ingestion: the speculation budget can send a stream back to plain decoding and be exactly as fast as a build with no
//!   draft. Rows past the accepted ones are never absorbed, so a rejected draft costs the draft's context nothing and the draft's
//!   committed length is never rolled back on this path.
//!
//! # The context is the target's committed length, or the slot does not draft
//!
//! Every slot carries the host-side book [`SlotBook`]: rows absorbed into the draft's pages, rows waiting in the ring. A slot whose book
//! equals the target's committed length may draft. Any row that does not extend it (a prompt resumed at a position the draft never saw: a
//! sequence restored from a prefix-cache snapshot that holds no valid context of this draft; a failed absorb; an injected fault) turns
//! the slot to `Lost` for the rest of the sequence: it gets no proposal (budget 0, plain decoding), it costs the draft nothing, and no
//! other slot is touched. A restore from a snapshot that does hold one declares the context valid with [`SpecRig::adopt_context`]:
//! the draft's pages `0..len` ARE the snapshot's pages, shared by reference like the target's, and the partly filled one is copied
//! with the target's (`prefix.rs`).
//!
//! # Pages
//!
//! The draft's page `i` is the target's page `i` (same ids, one block table per slot, E4a). A verify round's block writes K/V at `P .. P + 14`
//! whether or not the target's window needs those pages, and the target's pool is sized to what admission admits (a page of slack a slot):
//! a draft that took pages from it could starve another request. So the draft owns ONE private overflow page per slot (id `kv_pages +
//! slot`, priced as `kv_draft_overflow` in `memory::allocations`) appended to the slot's list: block rows past the target's last page
//! land there, and context rows never do (committed positions always have a target page). The draft never allocates from the target's pool.
//!
//! # Failure
//!
//! A recoverable error of the draft (a refused request, an allocation, a kernel error that is not sticky) makes that slot `Lost` and
//! gives the engine `None` for it: the engine's own drafter (none) leaves the stream at plain decoding. It is never recorded as a backend
//! error, because a recorded error fails every sequence of the call it surfaces in. Only a fatal (sticky device) error is returned.

use super::config::DSparkConfig;
use super::dspark::{DraftModel, DraftOptions, Head, HeadKind, IngestSeg, ProposeReq, BLOCK, HIDDEN};
use crate::error::ModelError;
use crate::kvtables::KvTables;
use crate::rows::RowBatch;
use gb_cuda::{glue, DevBuf, Stream};
use gb_sampling::Sampling;
use std::path::Path;
use std::time::Instant;

/// Committed rows a slot keeps in its ring before they are absorbed (a verify window commits at most 16).
pub const RING_ROWS: usize = 48;
/// Streams of one draft round (8 x 15 rows are the 128 rows of the decode kernels); more streams run as further rounds.
pub const MAX_STREAMS: usize = 8;
/// Tap rows absorbed per `ingest_taps` chunk.
pub const MAX_CTX_ROWS: usize = 512;

// ------------------------------------------------------------------------------------------------------------- the book

/// Why a slot's draft context is not the target's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// Rows arrived at a position past what the draft holds: a sequence restored from a snapshot that carries no draft pages, or a
    /// prompt resumed in the middle.
    Restored,
    /// Rows arrived at a position the draft already holds, or out of order: its context cannot be extended.
    Gap,
    /// A draft call for the slot failed (or a test injected a failure).
    Fault,
    /// The slot's pages could not be given to the draft.
    Pages,
}

impl Why {
    pub fn name(self) -> &'static str {
        match self {
            Why::Restored => "restored without draft context",
            Why::Gap => "gap in the draft context",
            Why::Fault => "draft failure",
            Why::Pages => "no draft pages",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotState {
    /// The draft holds rows `0..absorbed`; `pending` more are in the ring: the target's committed length is their sum.
    Live { absorbed: usize, pending: usize },
    /// The draft's context is not the target's: no proposals for this sequence.
    Lost(Why),
}

/// What a prefill chunk does to a slot's draft context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkStep {
    /// Absorb the chunk's rows at its first position.
    Absorb,
    /// Nothing (the slot is lost, or just became so).
    Skip,
}

/// What committing `keep` rows does to a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingStep {
    /// Copy the rows to the ring at `at`.
    Copy {
        at: usize,
    },
    /// The ring cannot take them: absorb what it holds, then copy at 0.
    AbsorbThenCopy,
    Skip,
}

/// The host-side bookkeeping of the draft's context per slot. Pure: the device work it licenses is [`SpecRig`]'s.
#[derive(Clone, Debug)]
pub struct SlotBook {
    slots: Vec<SlotState>,
}

impl SlotBook {
    pub fn new(slots: usize) -> Self {
        Self { slots: vec![SlotState::Live { absorbed: 0, pending: 0 }; slots] }
    }

    pub fn state(&self, slot: usize) -> SlotState {
        self.slots[slot]
    }

    /// Slots of the book.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// A new sequence: an empty, live context.
    pub fn begin(&mut self, slot: usize) {
        self.slots[slot] = SlotState::Live { absorbed: 0, pending: 0 };
    }

    pub fn lose(&mut self, slot: usize, why: Why) {
        self.slots[slot] = SlotState::Lost(why);
    }

    /// The target's committed length when the slot may draft.
    pub fn committed(&self, slot: usize) -> Option<usize> {
        match self.slots[slot] {
            SlotState::Live { absorbed, pending } => Some(absorbed + pending),
            SlotState::Lost(_) => None,
        }
    }

    pub fn pending(&self, slot: usize) -> usize {
        match self.slots[slot] {
            SlotState::Live { pending, .. } => pending,
            SlotState::Lost(_) => 0,
        }
    }

    /// A prefill chunk of `rows` rows starting at `pos0` was run by the target. `Absorb`: it extends the context exactly. Anything
    /// else loses the slot (and `Skip` is returned for it and for every later chunk).
    pub fn prefill_chunk(&mut self, slot: usize, pos0: usize) -> ChunkStep {
        match self.slots[slot] {
            SlotState::Lost(_) => ChunkStep::Skip,
            SlotState::Live { absorbed, pending } => {
                if pending == 0 && pos0 == absorbed {
                    ChunkStep::Absorb
                } else {
                    // a position past the context: the rows before it were never seen (a restore); anything else is a gap in the order
                    let why = if pending == 0 && pos0 > absorbed { Why::Restored } else { Why::Gap };
                    self.slots[slot] = SlotState::Lost(why);
                    ChunkStep::Skip
                }
            }
        }
    }

    /// `rows` rows were absorbed into the pages.
    pub fn absorbed(&mut self, slot: usize, rows: usize) {
        if let SlotState::Live { absorbed, pending } = &mut self.slots[slot] {
            *absorbed += rows;
            debug_assert_eq!(*pending, 0);
        }
    }

    /// The window of the slot kept `keep` rows.
    pub fn commit(&mut self, slot: usize, keep: usize) -> RingStep {
        match self.slots[slot] {
            SlotState::Lost(_) => RingStep::Skip,
            SlotState::Live { .. } if keep == 0 => RingStep::Skip,
            SlotState::Live { pending, .. } => {
                if pending + keep <= RING_ROWS {
                    RingStep::Copy { at: pending }
                } else {
                    RingStep::AbsorbThenCopy
                }
            }
        }
    }

    pub fn copied(&mut self, slot: usize, keep: usize) {
        if let SlotState::Live { pending, .. } = &mut self.slots[slot] {
            *pending += keep;
        }
    }

    /// The ring's `pending` rows were absorbed.
    pub fn flushed(&mut self, slot: usize) {
        if let SlotState::Live { absorbed, pending } = &mut self.slots[slot] {
            *absorbed += *pending;
            *pending = 0;
        }
    }

    /// The seam of the prefix cache: the draft's pages `0..len` of the slot are valid (a snapshot's draft pages were shared by
    /// reference), so the context is `len` rows and the slot may draft.
    pub fn adopt(&mut self, slot: usize, len: usize) {
        self.slots[slot] = SlotState::Live { absorbed: len, pending: 0 };
    }
}

// --------------------------------------------------------------------------------------------------------- the interface

/// One stream's ask of a draft round.
#[derive(Clone, Copy, Debug)]
pub struct SpecAsk<'a> {
    pub slot: usize,
    /// The sequence so far, ending with the anchor (the last sampled token, not yet fed to the target).
    pub tokens: &'a [u32],
    pub sampling: &'a Sampling,
    /// Drafts the budget wants; the draft always runs its whole block and the answer is cut to this.
    pub max: usize,
}

/// One stream's answer: the drafted tokens and the head's calibrated probability that each is accepted given the earlier ones were.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecProposal {
    pub tokens: Vec<u32>,
    pub accept: Vec<f32>,
}

/// Counters of what the draft did, for the benchmark tables and the tests.
#[derive(Clone, Debug)]
pub struct SpecCounters {
    /// Draft rounds (calls of the draft model), streams in them, and drafted tokens returned.
    pub rounds: u64,
    pub streams: u64,
    pub proposed: u64,
    /// Ingestion calls and tap rows absorbed (prompt rows and accepted rows), with the host time of both.
    pub absorb_calls: u64,
    pub absorb_rows: u64,
    pub absorb_ns: u64,
    pub propose_ns: u64,
    /// Asks answered with no proposal, by reason.
    pub refused_lost: u64,
    pub refused_mismatch: u64,
    pub refused_context: u64,
    pub lost_slots: u64,
    /// A drafted token at or past the vocabulary (must stay 0: the pad is masked in the draft's top-K).
    pub pad_leaks: u64,
    /// Rounds by the number of drafts committed (index = accepted drafts, 0..=15), over the windows that carried this draft's
    /// proposals: with a full 15-draft window, index k is "the first rejection was at depth k".
    pub first_reject: [u64; BLOCK + 1],
    /// By draft position (0-based) and confidence decile: drafts tested (every earlier one accepted) and accepted.
    pub reached: [[u64; 10]; BLOCK],
    pub accepted: [[u64; 10]; BLOCK],
}

impl Default for SpecCounters {
    fn default() -> Self {
        Self {
            rounds: 0,
            streams: 0,
            proposed: 0,
            absorb_calls: 0,
            absorb_rows: 0,
            absorb_ns: 0,
            propose_ns: 0,
            refused_lost: 0,
            refused_mismatch: 0,
            refused_context: 0,
            lost_slots: 0,
            pad_leaks: 0,
            first_reject: [0; BLOCK + 1],
            reached: [[0; 10]; BLOCK],
            accepted: [[0; 10]; BLOCK],
        }
    }
}

/// Device bytes the draft holds beyond the target's engine: weights and pool (priced in the plan's weights and pages), and the scratch,
/// tap buffer and rings (inside the engine reserve).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecBytes {
    pub weights: u64,
    /// K and V pages, the overflow pages included.
    pub pool: u64,
    pub scratch: u64,
    pub taps: u64,
    pub ring: u64,
}

impl SpecBytes {
    /// What the engine reserve must hold for the draft.
    pub fn engine_reserve(&self) -> u64 {
        self.scratch + self.taps + self.ring
    }
}

/// Sizes the rig is built for (the target's, from the boot).
#[derive(Clone, Copy, Debug)]
pub struct SpecSizes {
    pub slots: usize,
    /// The target's KV pages (the draft's own overflow pages come after them).
    pub kv_pages: usize,
    pub max_context: usize,
    /// The target's blocks per slot.
    pub max_blocks: usize,
    /// Rows the tap buffer holds: the widest of a prefill chunk and a window.
    pub tap_rows: usize,
    pub vocab: usize,
    pub head_stride: usize,
    pub head_bits: usize,
    pub target_layers: usize,
}

/// Test hooks: failures and corruptions injected into the draft's calls, to prove isolation and exactness.
#[derive(Default)]
pub struct Inject {
    /// The next absorb or propose touching this slot fails with a recoverable error.
    pub fail_slot: Option<usize>,
    /// Called on every full proposal of the draft (slot, round number of the rig, tokens): may corrupt it before it is cut to the
    /// budget's ask.
    #[allow(clippy::type_complexity)]
    pub corrupt: Option<Box<dyn FnMut(usize, u64, &mut Vec<u32>)>>,
    /// Cost attribution of the attachment (a measurement, never set in serving): the tap export of the forwards recorded from now
    /// on does not run (the recorded window graphs must be dropped for it to take effect) / the accepted rows are not copied to
    /// the rings (the draft's context is then wrong: nothing may draft).
    pub skip_export: bool,
    pub skip_ring: bool,
}

/// A proposal kept until its window's commit says how far it got.
#[derive(Clone, Debug)]
struct Outstanding {
    tokens: Vec<u32>,
    conf: Vec<f32>,
}

/// The draft, its tap buffer and rings, and the book of every slot.
pub struct SpecRig {
    pub draft: DraftModel,
    pub cfg: DSparkConfig,
    /// Tap index of each target layer (`None`: not a tap).
    tap_of_layer: Vec<Option<usize>>,
    n_taps: usize,
    /// bf16 `[tap_rows][taps][2048]`: the rows of the forward that ran last.
    taps: DevBuf,
    /// bf16 `[slots][RING_ROWS][taps][2048]`: accepted rows waiting to be absorbed.
    ring: DevBuf,
    /// The draft's own blocking stream: its uploads and kernels are ordered among themselves, and it only reads what the target's
    /// (non-blocking) stream has finished (the target synchronises at the end of every forward and commit).
    stream: Stream,
    pub book: SlotBook,
    /// Candidates a row's Markov walk takes (1..=64).
    pub k_top: usize,
    kv_pages: usize,
    max_blocks: usize,
    /// The block list each slot's draft table holds (the target's blocks and the overflow page).
    blocks: Vec<Vec<u32>>,
    outstanding: Vec<Option<Outstanding>>,
    pub counters: SpecCounters,
    pub inject: Inject,
    bytes: SpecBytes,
    vocab: usize,
    ring_slot_bytes: usize,
    row_bytes: usize,
    /// Whose draft this is, for the prefix cache: the draft pack's fingerprint, the tap layers and the page layout it adds.
    identity: String,
}

fn lose_why(e: &ModelError) -> Why {
    if matches!(e, ModelError::Request(_)) {
        Why::Pages
    } else {
        Why::Fault
    }
}

impl SpecRig {
    /// Load the draft at `dir` for a target of `z`'s sizes and allocate the tap buffer and the rings. Refuses (before anything is read) a
    /// draft that does not belong to the target; an allocation error is the boot's.
    pub fn new(cfg: &DSparkConfig, dir: &Path, z: SpecSizes) -> Result<Self, ModelError> {
        if z.slots == 0 || z.tap_rows == 0 || z.kv_pages == 0 {
            return Err(ModelError::Config(format!(
                "draft rig of {} slots, {} tap rows, {} pages",
                z.slots, z.tap_rows, z.kv_pages
            )));
        }
        let n_taps = cfg.target_layer_ids.len();
        let mut tap_of_layer = vec![None; z.target_layers];
        for (ti, &l) in cfg.target_layer_ids.iter().enumerate() {
            match tap_of_layer.get_mut(l) {
                Some(slot) => *slot = Some(ti),
                None => {
                    return Err(ModelError::Config(format!(
                        "draft tap layer {l} is not a layer of the target ({})",
                        z.target_layers
                    )))
                }
            }
        }
        let opts = DraftOptions {
            slots: z.slots,
            pages: z.kv_pages + z.slots,
            max_streams: z.slots.clamp(1, MAX_STREAMS),
            max_ctx_rows: MAX_CTX_ROWS,
            max_context: z.max_context,
            max_blocks: z.max_blocks + 1,
            vocab: z.vocab,
            target_layers: z.target_layers,
            head_stride: z.head_stride,
            head: HeadKind::Exl3 { bits: z.head_bits },
        };
        let (draft, _total) = DraftModel::new(cfg, dir, opts)?;
        let fp = crate::fingerprint::fingerprint(&crate::pack::Pack::open(dir)?.ck);
        let identity =
            format!("dspark|pack={fp}|taps={:?}|kv=bf16|page={}", cfg.target_layer_ids, super::memory::draft_page_bytes(cfg));
        let row_bytes = n_taps * HIDDEN * 2;
        let taps = DevBuf::zeroed(z.tap_rows * row_bytes)?;
        let ring = DevBuf::zeroed(z.slots * RING_ROWS * row_bytes)?;
        let bytes = SpecBytes {
            weights: draft.weight_bytes() as u64,
            pool: draft.kv().bytes(),
            scratch: draft.scratch_bytes() as u64,
            taps: taps.bytes() as u64,
            ring: ring.bytes() as u64,
        };
        let k_top = match std::env::var("REXL3_MINICPM_DRAFT_K") {
            Ok(v) => v
                .parse::<usize>()
                .ok()
                .filter(|k| (1..=64).contains(k))
                .ok_or_else(|| ModelError::Config(format!("REXL3_MINICPM_DRAFT_K={v}: expected 1..=64")))?,
            Err(_) => 64,
        };
        Ok(Self {
            draft,
            cfg: cfg.clone(),
            tap_of_layer,
            n_taps,
            taps,
            ring,
            stream: Stream::new()?,
            book: SlotBook::new(z.slots),
            k_top,
            kv_pages: z.kv_pages,
            max_blocks: z.max_blocks + 1,
            blocks: vec![Vec::new(); z.slots],
            outstanding: vec![None; z.slots],
            counters: SpecCounters::default(),
            inject: Inject::default(),
            bytes,
            vocab: z.vocab,
            ring_slot_bytes: RING_ROWS * row_bytes,
            row_bytes,
            identity,
        })
    }

    pub fn bytes(&self) -> SpecBytes {
        self.bytes
    }

    /// Whose draft this is (pack fingerprint, tap layers, page layout): what a prefix snapshot's draft context is valid for.
    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn taps(&self) -> usize {
        self.n_taps
    }

    /// Rows the tap buffer holds.
    pub fn tap_rows(&self) -> usize {
        self.taps.bytes() / self.row_bytes
    }

    /// The tap buffer (bf16 `[rows][taps][2048]`, the last forward's rows): tests compare it with the target's own residual stream.
    pub fn tap_buffer(&self) -> &DevBuf {
        &self.taps
    }

    fn overflow_page(&self, slot: usize) -> u32 {
        (self.kv_pages + slot) as u32
    }

    // ---------------------------------------------------------------------------------------------------- the target's side

    /// After layer `layer` of a forward of `rows` rows: copy `x` (bf16 `[rows, 2048]`) into this layer's tap column, if it is a tap.
    pub fn export(&self, layer: usize, x: &DevBuf, rows: usize, st: &Stream) -> Result<(), ModelError> {
        let Some(Some(ti)) = self.tap_of_layer.get(layer) else { return Ok(()) };
        if self.inject.skip_export {
            return Ok(());
        }
        if rows > self.tap_rows() {
            return Err(ModelError::Request(format!("a forward of {rows} rows past the tap buffer of {}", self.tap_rows())));
        }
        // SAFETY: bf16 rows of 2,048 elements: `x` holds `rows` of them (the caller's buffer is sized for the widest forward), and the
        // tap buffer `rows x taps` of them; the destination column `ti` is inside each row.
        unsafe {
            glue::copy_rows(
                x.ptr().cast_const(),
                HIDDEN,
                self.taps.ptr_at(ti * HIDDEN * 2),
                self.n_taps * HIDDEN,
                rows,
                HIDDEN,
                st,
            )?;
        }
        Ok(())
    }

    /// A new sequence in `slot` (the target's `begin_slot`).
    pub fn begin_slot(&mut self, slot: usize) {
        self.book.begin(slot);
        self.outstanding[slot] = None;
        // the draft's committed length and page list start over; its table is brought up to date before the first use
        self.blocks[slot].clear();
        if let Err(e) = self.draft.begin_slot(slot, &[]) {
            self.lose(slot, lose_why(&e), &e);
        }
    }

    /// The sequence of `slot` ended (the target's `end_slot`).
    pub fn end_slot(&mut self, slot: usize) {
        self.book.begin(slot);
        self.outstanding[slot] = None;
    }

    fn lose(&mut self, slot: usize, why: Why, e: &ModelError) {
        if !matches!(self.book.state(slot), SlotState::Lost(_)) {
            self.counters.lost_slots += 1;
            eprintln!("rexl3 level=warn event=minicpm_draft_lost slot={slot} why=\"{}\" error=\"{e}\"", why.name());
        }
        self.book.lose(slot, why);
        self.outstanding[slot] = None;
    }

    /// The prefill chunk of `rows` rows at `pos0` of `slot` has run (the tap buffer holds its rows): absorb them.
    pub fn after_prefill(&mut self, slot: usize, pos0: usize, rows: usize, kv: &KvTables) -> Result<(), ModelError> {
        let was_live = matches!(self.book.state(slot), SlotState::Live { .. });
        if self.book.prefill_chunk(slot, pos0) == ChunkStep::Skip {
            // a slot turns lost once (a later chunk of a lost slot is skipped silently)
            if let (true, SlotState::Lost(why)) = (was_live, self.book.state(slot)) {
                self.counters.lost_slots += 1;
                eprintln!("rexl3 level=warn event=minicpm_draft_lost slot={slot} why=\"{}\" position={pos0}", why.name());
            }
            return Ok(());
        }
        let t = Instant::now();
        let r = self.absorb_rows(slot, pos0, rows, TapSource::Window, kv);
        self.counters.absorb_ns += t.elapsed().as_nanos() as u64;
        match r {
            Ok(()) => {
                self.book.absorbed(slot, rows);
                self.counters.absorb_calls += 1;
                self.counters.absorb_rows += rows as u64;
                Ok(())
            }
            Err(e) if e.is_fatal() => Err(e),
            Err(e) => {
                self.lose(slot, lose_why(&e), &e);
                Ok(())
            }
        }
    }

    /// The commit of a window was applied: `keeps[i]` rows of sequence `i` of `win` are committed. Copy their taps to the slots' rings
    /// (on `st`, the target's stream, whose next synchronisation orders them before anything the draft's stream reads).
    pub fn commit_window(&mut self, win: &RowBatch, keeps: &[i32], kv: &KvTables, st: &Stream) -> Result<(), ModelError> {
        for (i, &keep) in keeps.iter().enumerate() {
            let slot = win.seq_slot[i] as usize;
            let rows = win.seq_rows(i);
            let keep = (keep.max(0) as usize).min(rows.len());
            self.note_outcome(slot, win, i, keep);
            let mut step = self.book.commit(slot, keep);
            if step == RingStep::AbsorbThenCopy {
                // the ring is full of rows nobody asked for: absorb them (the target's stream has finished the earlier copies)
                st.sync()?;
                match self.absorb_ring(slot, kv) {
                    Ok(()) => step = RingStep::Copy { at: 0 },
                    Err(e) if e.is_fatal() => return Err(e),
                    Err(e) => {
                        self.lose(slot, lose_why(&e), &e);
                        step = RingStep::Skip;
                    }
                }
            }
            let RingStep::Copy { at } = step else { continue };
            let src_row = rows.start;
            // SAFETY: `keep` contiguous rows of `row_bytes` in the tap buffer (rows of this window) and in this slot's ring (`at +
            // keep <= RING_ROWS`, checked by the book).
            if !self.inject.skip_ring {
                unsafe {
                    glue::copy_rows(
                        self.taps.ptr_at(src_row * self.row_bytes).cast_const(),
                        self.row_bytes / 2,
                        self.ring.ptr_at(slot * self.ring_slot_bytes + at * self.row_bytes),
                        self.row_bytes / 2,
                        keep,
                        self.row_bytes / 2,
                        st,
                    )?;
                }
            }
            self.book.copied(slot, keep);
        }
        Ok(())
    }

    /// What the committed rows say about the draft's last proposal of the slot (counters only).
    fn note_outcome(&mut self, slot: usize, win: &RowBatch, seq: usize, keep: usize) {
        let Some(out) = self.outstanding[slot].take() else { return };
        let rows = win.seq_rows(seq);
        let verified = rows.len().saturating_sub(1);
        if verified == 0 || keep == 0 {
            return;
        }
        // only a window that carried this proposal's tokens counts (a stale proposal or another drafter's rows are skipped)
        let carried = &win.ids[rows.start + 1..rows.end];
        if carried.len() > out.tokens.len() || carried.iter().zip(&out.tokens).any(|(&a, &b)| a as u32 != b) {
            return;
        }
        let accepted = keep - 1;
        let reached = verified.min(accepted + 1);
        for j in 0..reached.min(BLOCK) {
            let bin = ((out.conf[j].clamp(0.0, 1.0) * 10.0) as usize).min(9);
            self.counters.reached[j][bin] += 1;
            if j < accepted {
                self.counters.accepted[j][bin] += 1;
            }
        }
        self.counters.first_reject[accepted.min(BLOCK)] += 1;
    }

    // ------------------------------------------------------------------------------------------------------ the draft's side

    /// Make the draft's page table of `slot` the target's blocks plus the slot's overflow page.
    fn sync_blocks(&mut self, slot: usize, kv: &KvTables) -> Result<(), ModelError> {
        let t = kv.blocks_of(slot);
        let mut want: Vec<u32> = Vec::with_capacity(t.len() + 1);
        want.extend_from_slice(t);
        want.push(self.overflow_page(slot));
        if want.len() > self.max_blocks {
            return Err(ModelError::Request(format!(
                "slot {slot}: {} blocks past the draft's table of {}",
                want.len(),
                self.max_blocks
            )));
        }
        if self.blocks[slot] != want {
            self.draft.set_blocks(slot, &want)?;
            self.blocks[slot] = want;
        }
        Ok(())
    }

    fn absorb_rows(&mut self, slot: usize, pos0: usize, rows: usize, src: TapSource, kv: &KvTables) -> Result<(), ModelError> {
        if self.inject.fail_slot == Some(slot) {
            self.inject.fail_slot = None;
            return Err(ModelError::Request("injected draft failure".into()));
        }
        self.sync_blocks(slot, kv)?;
        let (buf, row0) = match src {
            TapSource::Window => (&self.taps, 0),
            TapSource::Ring => (&self.ring, slot * RING_ROWS),
        };
        self.draft.ingest_taps(slot, pos0, rows, buf, row0, &self.stream)?;
        self.stream.sync()?;
        Ok(())
    }

    /// Absorb the rows waiting in `slot`'s ring.
    fn absorb_ring(&mut self, slot: usize, kv: &KvTables) -> Result<(), ModelError> {
        let SlotState::Live { absorbed, pending } = self.book.state(slot) else { return Ok(()) };
        if pending == 0 {
            return Ok(());
        }
        let t = Instant::now();
        self.absorb_rows(slot, absorbed, pending, TapSource::Ring, kv)?;
        self.counters.absorb_ns += t.elapsed().as_nanos() as u64;
        self.counters.absorb_calls += 1;
        self.counters.absorb_rows += pending as u64;
        self.book.flushed(slot);
        Ok(())
    }

    /// Drafts for the asks of a round. `None` for a stream that cannot draft now (its context is not the target's, a failure, no
    /// pages): the engine leaves it at plain decoding. `embed` and `head` are the target's own tables. The caller has applied every
    /// pending commit (`MiniCpmModel::flush`) so that the rings hold every committed row.
    pub fn propose(
        &mut self,
        embed: &DevBuf,
        head: &Head<'_>,
        kv: &KvTables,
        asks: &[SpecAsk<'_>],
    ) -> Result<Vec<Option<SpecProposal>>, ModelError> {
        let mut out: Vec<Option<SpecProposal>> = vec![None; asks.len()];
        let mut ready: Vec<usize> = Vec::new();
        for (i, a) in asks.iter().enumerate() {
            let slot = a.slot;
            if slot >= self.blocks.len() || a.tokens.is_empty() || a.max == 0 {
                continue;
            }
            if asks[..i].iter().any(|b| b.slot == slot) {
                self.counters.refused_mismatch += 1;
                continue;
            }
            let p = a.tokens.len() - 1;
            match self.book.committed(slot) {
                None => self.counters.refused_lost += 1,
                Some(c) if c != p => {
                    // the engine and the draft disagree about the committed length: never guess, lose the slot
                    self.counters.refused_mismatch += 1;
                    let e = ModelError::Request(format!("slot {slot}: anchor position {p}, draft context {c}"));
                    self.lose(slot, Why::Gap, &e);
                }
                Some(_) => {
                    // the block's 15 rows must fit the target's pages and the slot's own overflow page, and the context limit: a
                    // stream that does not is answered with nothing here, so that the draft call below never refuses a group for it
                    let cap = (kv.blocks_of(slot).len() + 1) * crate::kvtables::BLOCK_TOKENS;
                    if p + BLOCK > cap || p > self.draft.options().max_context {
                        self.counters.refused_context += 1;
                    } else {
                        ready.push(i);
                    }
                }
            }
        }
        for group in ready.chunks(self.draft_streams()) {
            // the pages first, then what each stream committed since its last round absorbed in ONE pass (a slot that fails here is
            // lost alone: the preconditions are checked per slot before the batch)
            let mut live: Vec<usize> = Vec::with_capacity(group.len());
            for &i in group {
                let slot = asks[i].slot;
                match self.sync_blocks(slot, kv) {
                    Ok(()) => live.push(i),
                    Err(e) if e.is_fatal() => return Err(e),
                    Err(e) => self.lose(slot, lose_why(&e), &e),
                }
            }
            let segs: Vec<IngestSeg> = live
                .iter()
                .filter_map(|&i| match self.book.state(asks[i].slot) {
                    SlotState::Live { absorbed, pending } if pending > 0 => {
                        Some(IngestSeg { slot: asks[i].slot, pos0: absorbed, rows: pending, taps_row0: asks[i].slot * RING_ROWS })
                    }
                    _ => None,
                })
                .collect();
            if !segs.is_empty() {
                let t = Instant::now();
                let r = self.draft.ingest_taps_batch(&segs, &self.ring, &self.stream).and_then(|_| Ok(self.stream.sync()?));
                self.counters.absorb_ns += t.elapsed().as_nanos() as u64;
                match r {
                    Ok(()) => {
                        self.counters.absorb_calls += 1;
                        for g in &segs {
                            self.counters.absorb_rows += g.rows as u64;
                            self.book.flushed(g.slot);
                        }
                    }
                    Err(e) if e.is_fatal() => return Err(e),
                    Err(e) => {
                        // a batch cannot be attributed to one slot: none of its slots is trusted with this context again
                        for g in &segs {
                            self.lose(g.slot, Why::Fault, &e);
                        }
                        live.retain(|&i| !segs.iter().any(|g| g.slot == asks[i].slot));
                    }
                }
            }
            if live.is_empty() {
                continue;
            }
            if let Some(f) = self.inject.fail_slot {
                if let Some(&i) = live.iter().find(|&&i| asks[i].slot == f) {
                    self.inject.fail_slot = None;
                    let e = ModelError::Request("injected draft failure".into());
                    self.lose(asks[i].slot, Why::Fault, &e);
                    live.retain(|&j| j != i);
                    if live.is_empty() {
                        continue;
                    }
                }
            }
            let reqs: Vec<ProposeReq<'_>> = live
                .iter()
                .map(|&i| {
                    let a = &asks[i];
                    ProposeReq { slot: a.slot, anchor: *a.tokens.last().unwrap(), p: a.tokens.len() - 1, sampling: a.sampling }
                })
                .collect();
            let t = Instant::now();
            let res = self.draft.propose(embed, head, &reqs, self.k_top, &self.stream);
            self.counters.propose_ns += t.elapsed().as_nanos() as u64;
            match res {
                Ok(props) => {
                    self.counters.rounds += 1;
                    self.counters.streams += live.len() as u64;
                    for (&i, mut p) in live.iter().zip(props) {
                        let slot = asks[i].slot;
                        if let Some(f) = self.inject.corrupt.as_mut() {
                            f(slot, self.counters.rounds, &mut p.tokens);
                        }
                        if p.tokens.iter().any(|&t| t as usize >= self.vocab) {
                            self.counters.pad_leaks += 1;
                            continue;
                        }
                        self.outstanding[slot] = Some(Outstanding { tokens: p.tokens.clone(), conf: p.confidence.clone() });
                        let n = asks[i].max.min(p.tokens.len());
                        self.counters.proposed += n as u64;
                        p.tokens.truncate(n);
                        p.confidence.truncate(n);
                        out[i] = Some(SpecProposal { tokens: p.tokens, accept: p.confidence });
                    }
                }
                Err(e) if e.is_fatal() => return Err(e),
                Err(e) => {
                    // the call failed for the group: none of its streams gets a draft now, and none is trusted with this context again
                    for &i in &live {
                        self.lose(asks[i].slot, Why::Fault, &e);
                    }
                }
            }
        }
        Ok(out)
    }

    fn draft_streams(&self) -> usize {
        self.draft.options().max_streams
    }

    /// The seam E6b fills when a restore carries the draft's pages: the draft's pages `0..len` of `slot` are the snapshot's, shared by
    /// reference like the target's, so the draft's context is `len` rows (call after the restore has begun the slot and installed its
    /// page list; the next prefill slice resumes at `len`). Without this call a restored slot never drafts.
    pub fn adopt_context(&mut self, slot: usize, len: usize, kv: &KvTables) -> Result<(), ModelError> {
        self.sync_blocks(slot, kv)?;
        // the draft's committed length is the restore's: `ingest_taps` refuses a hole, and the rows below `len` already exist
        self.draft.adopt_len(slot, len)?;
        self.book.adopt(slot, len);
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum TapSource {
    Window,
    Ring,
}

// ---------------------------------------------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prompt_extends_the_context_chunk_by_chunk_and_a_resume_at_zero_is_the_start() {
        let mut b = SlotBook::new(2);
        assert_eq!(b.prefill_chunk(0, 0), ChunkStep::Absorb);
        b.absorbed(0, 512);
        assert_eq!(b.prefill_chunk(0, 512), ChunkStep::Absorb);
        b.absorbed(0, 100);
        assert_eq!(b.committed(0), Some(612));
        // another slot is not touched
        assert_eq!(b.state(1), SlotState::Live { absorbed: 0, pending: 0 });
    }

    #[test]
    fn a_restored_slot_loses_its_draft_context_and_every_later_chunk_is_skipped() {
        let mut b = SlotBook::new(1);
        // a prefix-cache restore: the slot resumes at 1,024 with no draft rows below it
        assert_eq!(b.prefill_chunk(0, 1024), ChunkStep::Skip);
        assert_eq!(b.state(0), SlotState::Lost(Why::Restored));
        assert_eq!(b.prefill_chunk(0, 1536), ChunkStep::Skip);
        assert_eq!(b.committed(0), None, "a lost slot never drafts");
        assert_eq!(b.commit(0, 5), RingStep::Skip);
        // a new sequence in the slot starts live again
        b.begin(0);
        assert_eq!(b.prefill_chunk(0, 0), ChunkStep::Absorb);
    }

    #[test]
    fn the_adopt_seam_re_enables_a_restored_slot() {
        let mut b = SlotBook::new(1);
        b.begin(0);
        b.adopt(0, 1024);
        assert_eq!(b.prefill_chunk(0, 1024), ChunkStep::Absorb);
        b.absorbed(0, 10);
        assert_eq!(b.committed(0), Some(1034));
    }

    #[test]
    fn a_position_below_the_context_is_a_gap_not_a_restore() {
        let mut b = SlotBook::new(1);
        assert_eq!(b.prefill_chunk(0, 0), ChunkStep::Absorb);
        b.absorbed(0, 300);
        assert_eq!(b.prefill_chunk(0, 100), ChunkStep::Skip);
        assert_eq!(b.state(0), SlotState::Lost(Why::Gap));
    }

    #[test]
    fn accepted_rows_wait_in_the_ring_until_it_is_full_or_a_round_needs_them() {
        let mut b = SlotBook::new(1);
        assert_eq!(b.prefill_chunk(0, 0), ChunkStep::Absorb);
        b.absorbed(0, 40);
        assert_eq!(b.commit(0, 16), RingStep::Copy { at: 0 });
        b.copied(0, 16);
        assert_eq!(b.commit(0, 16), RingStep::Copy { at: 16 });
        b.copied(0, 16);
        assert_eq!(b.committed(0), Some(72), "the target's committed length is absorbed + pending");
        assert_eq!(b.pending(0), 32);
        // 32 + 16 = 48 fits; one more row does not
        assert_eq!(b.commit(0, 16), RingStep::Copy { at: 32 });
        b.copied(0, 16);
        assert_eq!(b.commit(0, 1), RingStep::AbsorbThenCopy);
        b.flushed(0);
        assert_eq!(b.state(0), SlotState::Live { absorbed: 88, pending: 0 });
        assert_eq!(b.commit(0, 1), RingStep::Copy { at: 0 });
        // a window that kept nothing leaves the book alone
        assert_eq!(b.commit(0, 0), RingStep::Skip);
    }

    #[test]
    fn a_prefill_chunk_with_rows_pending_is_a_gap() {
        // rows committed by a window are still in the ring when a prefill chunk arrives (a resumed conversation on a live slot):
        // the chunk's position is the committed length, but the book does not extend over pending rows
        let mut b = SlotBook::new(1);
        b.prefill_chunk(0, 0);
        b.absorbed(0, 10);
        b.commit(0, 2);
        b.copied(0, 2);
        assert_eq!(b.prefill_chunk(0, 12), ChunkStep::Skip);
        assert_eq!(b.state(0), SlotState::Lost(Why::Gap));
    }

    #[test]
    fn the_why_names_every_reason() {
        for w in [Why::Restored, Why::Gap, Why::Fault, Why::Pages] {
            assert!(!w.name().is_empty());
        }
    }
}
