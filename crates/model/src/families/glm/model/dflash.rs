//! The GLM model's side of the DFlash2 draft (`families/glm/dflash.rs` has the design): the tap buffer the forward fills
//! (`Op::Tap`), the absorb of committed rows into each slot's draft context (prompt chunks after their forward, windows at their commit),
//! the draft window a prefix snapshot keeps, and the draft call.
//!
//! Nothing here touches the trunk's state: the draft has its own ring and scratch, the taps are a copy of the streams, and every launch is
//! on the model's stream after the forward that wrote what it reads. So a draft can change how many tokens a round commits, never which.

use super::snapshot::InChunkCut;
use super::GlmModel;
use crate::draft::{Absorb, Draft, DraftReq, TAPS};
use crate::error::ModelError;
use crate::families::glm::alloc::TOKENIZER_VOCAB;
use crate::families::glm::dflash::{CtxBook, TAP_LAYERS, WINDOW};
use crate::rows::RowBatch;
use gb_cuda::dflash2::DraftCtxLayout;
use gb_cuda::DevBuf;
use gb_sampling::Sampling;

/// The DFlash2 draft and what the model keeps for it.
pub struct DflashState {
    pub draft: Draft,
    /// bf16 `[rows, TAPS * hidden]`: the taps of every row of the last forward (a prompt chunk or a window), until the next forward.
    pub(super) taps: DevBuf,
    /// bf16 `[rows, hidden]`: one tap layer's mean of the streams, before it is placed in its column of `taps`.
    pub(super) tmp: DevBuf,
    pub(super) books: Vec<CtxBook>,
    /// Per slot, the first prompt position the draft must absorb (`dflash::ctx_from`; 0 unless the backend says so).
    pub(super) from: Vec<usize>,
    /// Bytes of the draft's weights (its cost model reads them).
    pub weight_bytes: u64,
    /// Rows absorbed and rounds drafted (telemetry).
    pub absorbed: u64,
    pub rounds: u64,
}

/// One sequence asking the draft for drafts.
pub struct DflashAsk<'a> {
    pub slot: usize,
    /// The sequence so far, ending with the one token not yet fed to the trunk (the anchor).
    pub tokens: &'a [u32],
    /// Drafts wanted, 1..=7.
    pub depth: usize,
    /// The request's sampler when the walk draws as the target does (`None`: the selector's argmax).
    pub sampling: Option<Sampling>,
}

/// The largest absorb one call makes: below the ring's capacity, so a write never overwrites a position it also needs.
const ABSORB_PIECE: usize = WINDOW;

impl DflashState {
    /// The model's draft state over a loaded `draft` for a model whose forwards carry at most `rows` rows of `hidden` and `slots` slots.
    pub fn new(draft: Draft, rows: usize, hidden: usize, slots: usize, weight_bytes: u64) -> Result<Self, ModelError> {
        if draft.geom.hidden != hidden {
            return Err(ModelError::Config(format!(
                "the draft reads taps of {} but the target's hidden size is {hidden}",
                draft.geom.hidden
            )));
        }
        if draft.slots < slots {
            return Err(ModelError::Config(format!("the draft has {} context slots for a model of {slots}", draft.slots)));
        }
        Ok(Self {
            draft,
            taps: DevBuf::zeroed(rows * TAPS * hidden * 2)?,
            tmp: DevBuf::zeroed(rows * hidden * 2)?,
            books: vec![CtxBook::new(); slots],
            from: vec![0; slots],
            weight_bytes,
            absorbed: 0,
            rounds: 0,
        })
    }

    /// Bytes this state allocated besides the draft itself (the tap buffers).
    pub fn tap_bytes(&self) -> u64 {
        (self.taps.bytes() + self.tmp.bytes()) as u64
    }

    /// Which tap column the output of trunk layer `layer` goes to, if any.
    pub fn tap_of(layer: usize) -> Option<usize> {
        TAP_LAYERS.iter().position(|&l| l == layer)
    }
}

impl GlmModel {
    /// Give the model a DFlash2 draft. Must come before the first forward: the forwards' schedule (and with it every captured window graph)
    /// carries the taps from then on.
    pub fn enable_dflash(&mut self, draft: Draft, weight_bytes: u64) -> Result<(), ModelError> {
        if self.mtp.is_some() {
            return Err(ModelError::Config("a model drafts with its NextN layer or with a DFlash2 draft, not both".into()));
        }
        if TAP_LAYERS.iter().any(|&l| l >= self.layers) {
            return Err(ModelError::Config(format!("the draft taps layers {TAP_LAYERS:?} of a model of {} layers", self.layers)));
        }
        // the draft's head is this model's whole lm_head (a rank of a group holds it whole and only computes its strips: `model/tp.rs`)
        if (self.dev.lm_head.k, self.dev.lm_head.n) != (self.cfg.hidden, self.cfg.vocab) {
            return Err(ModelError::Config(format!(
                "the DFlash2 draft's head is the target's lm_head, and this model holds {}x{} of {}x{}",
                self.dev.lm_head.k, self.dev.lm_head.n, self.cfg.hidden, self.cfg.vocab
            )));
        }
        let rows = self.s.rows.max(self.window_rows);
        self.dflash = Some(DflashState::new(draft, rows, self.cfg.hidden, self.slots, weight_bytes)?);
        Ok(())
    }

    pub fn has_dflash(&self) -> bool {
        self.dflash.is_some()
    }

    pub fn dflash(&self) -> Option<&DflashState> {
        self.dflash.as_ref()
    }

    /// A new sequence in `slot` (`begin_slot`).
    pub(super) fn dflash_begin(&mut self, slot: usize) {
        if let Some(d) = self.dflash.as_mut() {
            d.books[slot] = CtxBook::new();
            d.from[slot] = 0;
        }
    }

    /// The first prompt position of `slot`'s prompt the draft must absorb (`dflash::ctx_from`): the backend says so before each slice.
    pub fn set_dflash_from(&mut self, slot: usize, from: usize) {
        if let Some(d) = self.dflash.as_mut() {
            d.from[slot] = from;
        }
    }

    /// Debug (the tap test): the tap buffer's first `rows` rows, bf16 bits `[rows][TAPS * hidden]`. Synchronises.
    pub fn dflash_tap_rows(&self, rows: usize) -> Result<Vec<u16>, ModelError> {
        let d = self.dflash.as_ref().ok_or_else(|| ModelError::Config("the model has no DFlash2 draft".into()))?;
        self.stream.sync()?;
        let mut v = vec![0u16; rows * TAPS * self.cfg.hidden];
        d.taps.download_range(0, &mut v)?;
        Ok(v)
    }

    /// The slot's draft context bookkeeping.
    pub fn dflash_book(&self, slot: usize) -> Option<CtxBook> {
        self.dflash.as_ref().map(|d| d.books[slot])
    }

    /// Absorb the rows `a..b` (absolute positions; tap rows from `row0`, the row of position `a`) of `slot` in pieces the ring can take.
    fn dflash_absorb(
        d: &mut DflashState,
        slot: usize,
        row0: usize,
        a: usize,
        b: usize,
        st: &gb_cuda::Stream,
    ) -> Result<(), ModelError> {
        let piece = ABSORB_PIECE.min(d.draft.max_ctx_rows());
        let mut at = a;
        while at < b {
            let n = (b - at).min(piece);
            d.draft.absorb(&d.taps, row0 + (at - a), n, slot, at, st)?;
            d.absorbed += n as u64;
            at += n;
        }
        Ok(())
    }

    /// A prompt chunk of `rows` rows of `slot` from `start` has run (the tap buffer holds its rows): absorb the rows the draft will read, in
    /// pieces split at the in-chunk snapshot cuts, each cut's draft window copied into its snapshot right after the rows before it are absorbed.
    pub(super) fn dflash_prefill_chunk(&mut self, slot: usize, start: usize, rows: usize) -> Result<(), ModelError> {
        let GlmModel { dflash, snaps, snap_cuts, stream, .. } = self;
        let Some(d) = dflash.as_mut() else { return Ok(()) };
        let from = d.from[slot];
        let cuts: Vec<InChunkCut> = snap_cuts.iter().copied().filter(|c| c.slot == slot).collect();
        let Some(range) = d.books[slot].prefill(start, rows, from) else {
            // the context is not sound: the snapshots of this chunk keep no window (a restore of one drafts nothing)
            if let Some(sp) = snaps.as_mut() {
                for c in &cuts {
                    sp.save_draft_window(c.snap, &d.draft, slot, None, stream)?;
                }
            }
            return Ok(());
        };
        let mut at = range.start;
        for c in &cuts {
            if c.at > at {
                Self::dflash_absorb(d, slot, at - start, at, c.at, stream)?;
                at = c.at;
            }
            if let Some(sp) = snaps.as_mut() {
                let w = d.books[slot].window_of(c.at);
                sp.save_draft_window(c.snap, &d.draft, slot, Some(w), stream)?;
            }
        }
        if range.end > at {
            Self::dflash_absorb(d, slot, at - start, at, range.end, stream)?;
        }
        Ok(())
    }

    /// The kept rows of the last window join their slots' draft contexts (called by the commit, before the next forward overwrites the taps).
    pub(super) fn dflash_commit(&mut self, win: &RowBatch, keep: &[i32]) -> Result<(), ModelError> {
        let GlmModel { dflash, stream, .. } = self;
        let Some(d) = dflash.as_mut() else { return Ok(()) };
        let mut items: Vec<Absorb> = Vec::new();
        for (i, &slot) in win.seq_slot.iter().enumerate() {
            let (slot, k) = (slot as usize, keep[i].max(0) as usize);
            if k == 0 {
                continue;
            }
            let rows = win.seq_rows(i);
            let pos0 = win.positions[rows.start] as usize;
            if d.books[slot].window(pos0, k.min(rows.len())) {
                items.push(Absorb { row0: rows.start, n: k.min(rows.len()), slot, pos0 });
            }
        }
        if !items.is_empty() {
            d.absorbed += items.iter().map(|a| a.n as u64).sum::<u64>();
            d.draft.absorb_batch(&d.taps, &items, stream)?;
        }
        Ok(())
    }

    /// Whether `slot` may draft for a sequence of `tokens` tokens (the last one not yet fed): its context ends right before the anchor and
    /// covers its window.
    pub fn dflash_ready(&self, slot: usize, tokens: usize) -> bool {
        tokens > 0 && self.dflash.as_ref().is_some_and(|d| d.books[slot].ready(tokens - 1))
    }

    /// One draft round for the asking sequences (every one must be [`GlmModel::dflash_ready`]): their drafts, each cut at the first id
    /// outside the tokenizer's vocabulary (the head's padding columns are never a token). Applies the pending commits first.
    pub fn dflash_draft(&mut self, asks: &[DflashAsk<'_>]) -> Result<Vec<Vec<u32>>, ModelError> {
        self.flush()?;
        let greedy = Sampling { temperature: 0.0, ..Sampling::default() };
        for a in asks {
            if !self.dflash_ready(a.slot, a.tokens.len()) {
                return Err(ModelError::Request(format!(
                    "slot {} cannot draft at position {}",
                    a.slot,
                    a.tokens.len().saturating_sub(1)
                )));
            }
        }
        let samplings: Vec<Sampling> = asks.iter().map(|a| a.sampling.unwrap_or(greedy)).collect();
        let reqs: Vec<DraftReq> = asks
            .iter()
            .zip(&samplings)
            .map(|(a, s)| DraftReq {
                anchor: *a.tokens.last().expect("checked above"),
                slot: a.slot,
                p: a.tokens.len() - 1,
                sampling: s,
                want: a.depth.clamp(1, crate::draft::DRAFTS),
                coverage: 1.0,
            })
            .collect();
        let GlmModel { dflash, dev, stream, cfg, .. } = self;
        let d = dflash.as_mut().expect("checked by dflash_ready");
        let mut out = Vec::with_capacity(reqs.len());
        for chunk in reqs.chunks(d.draft.max_seqs) {
            for mut t in d.draft.propose_batch(&dev.embed, &dev.lm_head, cfg.vocab, chunk, stream)? {
                if let Some(cut) = t.iter().position(|&x| x as usize >= TOKENIZER_VOCAB) {
                    t.truncate(cut);
                }
                out.push(t);
            }
        }
        d.rounds += 1;
        Ok(out)
    }

    /// Put a snapshot's draft window back into `slot`'s ring (a prefix restore of `len` tokens): the slot's context is then the snapshot's.
    pub(super) fn dflash_restore(&mut self, slot: usize, snap: u32, len: usize) -> Result<(), ModelError> {
        let GlmModel { dflash, snaps, stream, .. } = self;
        let Some(d) = dflash.as_mut() else { return Ok(()) };
        let Some(sp) = snaps.as_ref() else { return Ok(()) };
        d.books[slot] = match sp.load_draft_window(snap, &mut d.draft, slot, len, stream)? {
            Some(lo) => CtxBook { ok: true, lo, end: len },
            None => CtxBook { ok: false, lo: len, end: len },
        };
        Ok(())
    }

    /// The draft window of `slot` after `len` prompt tokens into snapshot `snap` (a snapshot between chunks).
    pub(super) fn dflash_snapshot(&mut self, slot: usize, snap: u32, len: usize) -> Result<(), ModelError> {
        let GlmModel { dflash, snaps, stream, .. } = self;
        let (Some(d), Some(sp)) = (dflash.as_ref(), snaps.as_mut()) else { return Ok(()) };
        let b = d.books[slot];
        let w = (b.ok && b.end == len).then(|| b.window_of(len));
        sp.save_draft_window(snap, &d.draft, slot, w, stream)
    }
}

const _: () = assert!(DraftCtxLayout::DEFAULT_CAP > ABSORB_PIECE, "one absorb must fit the ring");
