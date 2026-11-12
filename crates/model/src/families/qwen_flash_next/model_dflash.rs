//! The Flash-Next model's side of the DFlash2 draft (`dflash.rs` has the design): the tap buffer the forward fills, the absorb of
//! committed rows into each slot's draft context (prompt chunks after their forward, windows once their keep counts are known), the
//! draft window a prefix snapshot keeps, and the draft call.
//!
//! Nothing here touches the trunk's state: the draft has its own ring and scratch, the taps are a copy of the streams, and every launch is
//! on the model's stream after the forward that wrote what it reads. So a draft can change how many tokens a round commits, never which.
//!
//! **The commit.** Flash-Next defers a window's recurrent-state commit into the next window's graph (`gdn::window_owed`). The draft's
//! absorb of the kept rows needs only the taps and the keep counts, not the state, so it never forces that commit: it runs once per window
//! and slot, at the first of a draft round, a flush, or the next window's preparation, always before the next forward overwrites the taps.

use super::FlashNextModel;
use crate::draft::{Absorb, Draft, DraftReq, TAPS};
use crate::error::ModelError;
use crate::families::qwen_flash_next::dflash::{tap_of, CtxBook, WINDOW};
use gb_cuda::dflash2::DraftCtxLayout;
use gb_cuda::{fn_mhc, DevBuf};
use gb_sampling::Sampling;
use std::collections::BTreeSet;

/// The DFlash2 draft and what the model keeps for it.
pub struct DflashState {
    pub draft: Draft,
    /// bf16 `[rows, TAPS * hidden]`: the taps of every row of the last forward (a prompt chunk or a window), until the next forward.
    taps: DevBuf,
    books: Vec<CtxBook>,
    /// Per slot, the first prompt position the draft must absorb (`dflash::ctx_from`; 0 unless the backend says so).
    from: Vec<usize>,
    /// The slots of the last window whose kept rows were absorbed already.
    absorbed_window: BTreeSet<usize>,
    /// Bytes of the draft's weights.
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
const _: () = assert!(DraftCtxLayout::DEFAULT_CAP > ABSORB_PIECE, "one absorb must fit the ring");

impl DflashState {
    /// Bytes this state allocated besides the draft itself (the tap buffer).
    pub fn tap_bytes(&self) -> u64 {
        self.taps.bytes() as u64
    }
}

/// The tap buffer's bytes for a model whose forwards carry at most `rows` rows of `hidden`.
pub fn tap_buffer_bytes(rows: usize, hidden: usize) -> u64 {
    (rows * TAPS * hidden * 2) as u64
}

impl FlashNextModel {
    /// Give the model a DFlash2 draft. Must come before the first forward: every forward (and every captured window graph) carries the
    /// tap launches from then on. A model drafts with its MTP head or with a draft, not both.
    pub fn enable_dflash(&mut self, draft: Draft, weight_bytes: u64) -> Result<(), ModelError> {
        if self.w.mtp.is_some() {
            return Err(ModelError::Config(
                "a Flash-Next model drafts with its MTP head or with a DFlash2 draft, not both".into(),
            ));
        }
        if !self.graphs.is_empty() {
            return Err(ModelError::Config("the DFlash2 draft joins the model before its first window".into()));
        }
        let (h, layers) = (self.cfg.hidden, self.w.layers.len());
        if draft.geom.hidden != h {
            return Err(ModelError::Config(format!(
                "the draft reads taps of {} but the target's hidden size is {h}",
                draft.geom.hidden
            )));
        }
        if super::super::dflash::TAP_LAYERS.iter().any(|&l| l >= layers) {
            return Err(ModelError::Config(format!(
                "the draft taps layers {:?} of a model of {layers} layers",
                super::super::dflash::TAP_LAYERS
            )));
        }
        if draft.slots < self.opts.slots {
            return Err(ModelError::Config(format!(
                "the draft has {} context slots for a model of {}",
                draft.slots, self.opts.slots
            )));
        }
        // the draft's head is the target's whole lm_head
        if (self.w.lm_head.k, self.w.lm_head.n) != (h, self.cfg.vocab) {
            return Err(ModelError::Config(format!(
                "the DFlash2 draft's head is the target's lm_head, and this model holds {}x{} of {h}x{}",
                self.w.lm_head.k, self.w.lm_head.n, self.cfg.vocab
            )));
        }
        let rows = self.rows;
        self.dflash = Some(DflashState {
            draft,
            taps: DevBuf::zeroed(tap_buffer_bytes(rows, h) as usize)?,
            books: vec![CtxBook::new(); self.opts.slots],
            from: vec![0; self.opts.slots],
            absorbed_window: BTreeSet::new(),
            weight_bytes,
            absorbed: 0,
            rounds: 0,
        });
        Ok(())
    }

    pub fn has_dflash(&self) -> bool {
        self.dflash.is_some()
    }

    pub fn dflash(&self) -> Option<&DflashState> {
        self.dflash.as_ref()
    }

    /// The slot's draft context bookkeeping.
    pub fn dflash_book(&self, slot: usize) -> Option<CtxBook> {
        self.dflash.as_ref().map(|d| d.books[slot])
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

    /// The forward's tap after trunk layer `layer` (a device launch on the model's stream; inside a window graph when one is captured).
    pub(super) fn dflash_tap(&mut self, layer: usize, rows: usize) -> Result<(), ModelError> {
        let Some(t) = tap_of(layer) else { return Ok(()) };
        let Some(d) = self.dflash.as_mut() else { return Ok(()) };
        let (h, hc) = (self.cfg.hidden, self.cfg.hc_streams);
        fn_mhc::hc_tap(&self.x_streams, &mut d.taps, h, hc, TAPS * h, t * h, rows, &self.stream)?;
        Ok(())
    }

    /// A new window replaces the last one: none of its slots has been absorbed yet.
    pub(super) fn dflash_new_window(&mut self) {
        if let Some(d) = self.dflash.as_mut() {
            d.absorbed_window.clear();
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

    /// A prompt chunk of `rows` rows of `slot` from `start` has run (the tap buffer holds its rows): absorb the rows the draft will read.
    /// Flash-Next takes its prefix snapshots between slices, so a chunk has no cut inside it.
    pub(super) fn dflash_prefill_chunk(&mut self, slot: usize, start: usize, rows: usize) -> Result<(), ModelError> {
        let FlashNextModel { dflash, stream, .. } = self;
        let Some(d) = dflash.as_mut() else { return Ok(()) };
        let from = d.from[slot];
        let Some(range) = d.books[slot].prefill(start, rows, from) else { return Ok(()) };
        let piece = ABSORB_PIECE.min(d.draft.max_ctx_rows());
        let mut at = range.start;
        while at < range.end {
            let n = (range.end - at).min(piece);
            d.draft.absorb(&d.taps, at - start, n, slot, at, stream)?;
            d.absorbed += n as u64;
            at += n;
        }
        Ok(())
    }

    /// The kept rows of the last window join their slots' draft contexts, for every slot with a pending keep count not absorbed yet.
    /// Called before anything can overwrite the taps (a flush, the next window's preparation, a draft round); never applies the
    /// recurrent-state commit itself.
    pub(super) fn dflash_absorb_pending(&mut self) -> Result<(), ModelError> {
        let FlashNextModel { dflash, stream, last_window, pending, .. } = self;
        let (Some(d), Some(win)) = (dflash.as_mut(), last_window.as_ref()) else { return Ok(()) };
        let mut items: Vec<Absorb> = Vec::new();
        for (i, &slot) in win.seq_slot.iter().enumerate() {
            let slot = slot as usize;
            let Some(&k) = pending.get(&slot) else { continue };
            if k == 0 || !d.absorbed_window.insert(slot) {
                continue;
            }
            let rows = win.seq_rows(i);
            let n = k.min(rows.len());
            let pos0 = win.positions[rows.start] as usize;
            if d.books[slot].window(pos0, n) {
                items.push(Absorb { row0: rows.start, n, slot, pos0 });
            }
        }
        if !items.is_empty() {
            d.absorbed += items.iter().map(|a| a.n as u64).sum::<u64>();
            d.draft.absorb_batch(&d.taps, &items, stream)?;
        }
        Ok(())
    }

    /// Whether `slot` may draft for a sequence of `tokens` tokens (the last one not yet fed): its context ends right before the anchor and
    /// covers its window. Counts the last window's pending keep, which the round absorbs first.
    pub fn dflash_ready(&self, slot: usize, tokens: usize) -> bool {
        let Some(d) = self.dflash.as_ref() else { return false };
        if tokens == 0 || slot >= d.books.len() {
            return false;
        }
        let mut b = d.books[slot];
        if !d.absorbed_window.contains(&slot) {
            if let (Some(win), Some(&k)) = (self.last_window.as_ref(), self.pending.get(&slot)) {
                if let Some(i) = win.seq_slot.iter().position(|&s| s as usize == slot) {
                    let rows = win.seq_rows(i);
                    if k > 0 && !b.window(win.positions[rows.start] as usize, k.min(rows.len())) {
                        return false;
                    }
                }
            }
        }
        b.ready(tokens - 1)
    }

    /// One draft round for the asking sequences (every one must be [`FlashNextModel::dflash_ready`]): their drafts, in ask order.
    /// Absorbs the last window's kept rows first; leaves the deferred recurrent-state commit to the next window.
    pub fn dflash_draft(&mut self, asks: &[DflashAsk<'_>]) -> Result<Vec<Vec<u32>>, ModelError> {
        for a in asks {
            if !self.dflash_ready(a.slot, a.tokens.len()) {
                return Err(ModelError::Request(format!(
                    "slot {} cannot draft at position {}",
                    a.slot,
                    a.tokens.len().saturating_sub(1)
                )));
            }
        }
        self.dflash_absorb_pending()?;
        let greedy = Sampling { temperature: 0.0, ..Sampling::default() };
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
        let FlashNextModel { dflash, w, stream, cfg, .. } = self;
        let d = dflash.as_mut().expect("checked by dflash_ready");
        let mut out = Vec::with_capacity(reqs.len());
        for chunk in reqs.chunks(d.draft.max_seqs) {
            out.extend(d.draft.propose_batch(&w.embed, &w.lm_head, cfg.vocab, chunk, stream)?);
        }
        d.rounds += 1;
        Ok(out)
    }

    /// The draft window of `slot` after `len` prompt tokens, as a snapshot keeps it: the positions it holds, if its context is sound there.
    pub(crate) fn dflash_window_of(&self, slot: usize, len: usize) -> Option<std::ops::Range<usize>> {
        let b = self.dflash.as_ref()?.books[slot];
        (b.ok && b.end == len).then(|| b.window_of(len))
    }

    /// Copy the draft ring's positions `w` of `slot` into snapshot `snap` of the per-layer buffers `dk` / `dv`.
    pub(crate) fn dflash_save_window(
        &self,
        slot: usize,
        w: std::ops::Range<usize>,
        dk: &mut [DevBuf],
        dv: &mut [DevBuf],
        snap: usize,
    ) -> Result<(), ModelError> {
        let d = self.dflash.as_ref().ok_or_else(|| ModelError::Config("the model has no DFlash2 draft".into()))?;
        d.draft.save_window(slot, w.start, w.len(), dk, dv, snap, &self.stream)
    }

    /// Put a snapshot's draft window (`lo..len`, or none) back into `slot`'s ring (a prefix restore of `len` tokens): the slot's context is
    /// then the snapshot's.
    pub(crate) fn dflash_load_window(
        &mut self,
        slot: usize,
        win: Option<usize>,
        len: usize,
        dk: &[DevBuf],
        dv: &[DevBuf],
        snap: usize,
    ) -> Result<(), ModelError> {
        let FlashNextModel { dflash, stream, .. } = self;
        let Some(d) = dflash.as_mut() else { return Ok(()) };
        d.books[slot] = match win {
            Some(lo) => {
                d.draft.load_window(slot, lo, len - lo, dk, dv, snap, stream)?;
                CtxBook { ok: true, lo, end: len }
            }
            None => CtxBook { ok: false, lo: len, end: len },
        };
        Ok(())
    }
}
