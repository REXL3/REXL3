//! The per-slot bookkeeping of the head's pending tapped rows (milestone M2, task MT-A): which committed rows still wait for an absorb,
//! gaps and overflows that switch a slot's drafting off (never an error), and the carry across prefill chunks. Pure host logic, tested
//! on the CPU. It is the GLM drafter's `TapBook` (`families/glm/model/mtp.rs`) with the same indexing, so the same invariants and tests
//! hold; the one Nemotron difference is below the book, in how an entry index becomes a cache position.
//!
//! **Entries.** Entry `m` pairs the tap `h_{m-1}` (the trunk's post-`norm_f` row of the position that PREDICTED `t_m`) with the token
//! `t_m`. The reference head (`mtp_ref`, vLLM's `NemotronHMTP`) has no entry for `m = 0` (nothing predicted `t_0`; GLM's drafter builds one
//! from a zero row, this head does not) and stores entry `m` at the head's cache position `m - 1`, the position of its tap: row `i` of
//! the reference pairs the hidden state at position `i` with the token at `i + 1`. The book does not care: `first` is an entry index.

/// Why a slot's draft state cannot serve a call. Any of these switches the slot's MTP off for the rest of its sequence (drafts are
/// proposals: losing them costs speed, never a byte); the slot's next `begin_slot` switches it back on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BookError {
    /// The slot's MTP state was switched off (a gap, an overflow, a mismatch, or rows pending that cannot be absorbed).
    Off,
    /// A prefill chunk or a window did not continue where the slot's tapped rows end.
    Gap,
    /// More rows are waiting for an absorb than the slot's tap store holds.
    Overflow,
    /// The tokens the engine reports and the rows the store holds disagree.
    Mismatch,
    /// Nothing is pending: there is no new hidden row to draft from.
    Nothing,
}

/// The bookkeeping of one slot's pending tapped rows. The store holds `n` rows: the taps of the committed positions `first - 1 ..
/// first - 1 + n`, i.e. the hidden inputs of the head's entries `first .. first + n` that are not built yet (entry `m` pairs tap
/// `h_{m-1}` with token `t_m`); every entry below `first` is built. A fresh slot is `first = 0, n = 0`; after a prompt of `T` tokens it
/// is `first = T, n = 1` (the carry `h_{T-1}`); after an absorb of a sequence of `len` tokens, `first = len, n = 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TapBook {
    pub ok: bool,
    pub first: usize,
    pub n: usize,
}

impl Default for TapBook {
    fn default() -> Self {
        Self::new()
    }
}

impl TapBook {
    pub const fn new() -> Self {
        Self { ok: true, first: 0, n: 0 }
    }

    /// A new sequence in the slot.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// No drafts from this slot until it is reset (an image in the window, or any [`BookError`]).
    pub fn disable(&mut self) {
        self.ok = false;
    }

    /// A prompt chunk of `rows` rows from position `start` was run through the trunk: the entries `start .. start + rows` are built in the
    /// same call, from the carry (the tap of position `start - 1`, held in the store's row 0) and this chunk's taps, and the chunk's last
    /// tap becomes the new carry. Returns whether the chunk's first row reads the carry.
    pub fn prefill_chunk(&mut self, start: usize, rows: usize) -> Result<bool, BookError> {
        if !self.ok {
            return Err(BookError::Off);
        }
        let carry = start > 0;
        let continues = if carry { self.n == 1 && self.first == start } else { self.n == 0 && self.first == 0 };
        if !continues || rows == 0 {
            self.ok = false;
            return Err(BookError::Gap);
        }
        self.first = start + rows;
        self.n = 1;
        Ok(carry)
    }

    /// A trunk window whose first row sat at position `row0_pos` was committed with `keep` rows: their taps join the store (only the kept
    /// rows: the draft state advances only with accepted rows). Returns the store row the first of them goes to.
    pub fn window_commit(&mut self, row0_pos: usize, keep: usize, cap: usize) -> Result<usize, BookError> {
        if !self.ok {
            return Err(BookError::Off);
        }
        if keep == 0 {
            return Ok(self.n);
        }
        // Where the pending rows would continue: right after the built entries (`first + n`). A fresh slot (`first = 0`, nothing built) may
        // only start at position 0: a window at a later position on a slot whose prompt never went through the head has no entries below
        // it (a restored prefix, say), and drafting from that state would read an empty cache. GLM's book accepts any position there.
        let fresh = self.n == 0 && self.first == 0;
        if row0_pos + 1 != self.first + self.n && !(fresh && row0_pos == 0) {
            self.ok = false;
            return Err(BookError::Gap);
        }
        if fresh {
            self.first = row0_pos + 1;
        }
        if self.n + keep > cap {
            self.ok = false;
            return Err(BookError::Overflow);
        }
        let dst = self.n;
        self.n += keep;
        Ok(dst)
    }

    /// May a draft call for a sequence of `tokens` tokens (the last one not yet fed to the trunk) absorb the store? Its entries `first ..
    /// tokens` need exactly the `n` rows held.
    pub fn absorb_check(&mut self, tokens: usize) -> Result<(), BookError> {
        if !self.ok {
            return Err(BookError::Off);
        }
        if self.n == 0 {
            return Err(BookError::Nothing);
        }
        if tokens != self.first + self.n {
            self.ok = false;
            return Err(BookError::Mismatch);
        }
        Ok(())
    }

    /// [`TapBook::absorb_check`] without switching the slot off on a mismatch: could a draft call absorb now?
    pub fn can_absorb(&self, tokens: usize) -> bool {
        self.ok && self.n > 0 && tokens == self.first + self.n
    }

    /// The store was absorbed for a sequence of `tokens` tokens: entries below `tokens` are built.
    pub fn absorbed(&mut self, tokens: usize) {
        self.first = tokens;
        self.n = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::super::mtp::MTP_TAP_CAP;
    use super::*;

    const CAP: usize = MTP_TAP_CAP;

    #[test]
    fn a_prompt_leaves_the_last_tap_as_the_carry() {
        let mut b = TapBook::new();
        // 300 tokens in chunks of 128: the first chunk starts from nothing, the next two from the carry
        assert_eq!(b.prefill_chunk(0, 128), Ok(false));
        assert_eq!((b.first, b.n), (128, 1));
        assert_eq!(b.prefill_chunk(128, 128), Ok(true));
        assert_eq!(b.prefill_chunk(256, 44), Ok(true));
        assert_eq!((b.first, b.n), (300, 1));
        // the first sampled token is the 301st: the entry for token 300 is the one pending
        assert!(b.can_absorb(301) && !b.can_absorb(300) && !b.can_absorb(302), "exactly the sequence length the store matches");
        assert_eq!(b.absorb_check(301), Ok(()));
        b.absorbed(301);
        assert_eq!((b.first, b.n), (301, 0));
    }

    #[test]
    fn only_the_kept_rows_of_a_window_join_the_store_and_the_next_absorb_needs_exactly_them() {
        let mut b = TapBook::new();
        b.prefill_chunk(0, 20).unwrap();
        b.absorb_check(21).unwrap();
        b.absorbed(21);
        // a window of 4 rows (the pending token at position 20 and three drafts) with 2 kept: positions 20 and 21 were committed, and with
        // them two new tokens (an accepted draft and a fresh sample) make the sequence 23 tokens long
        assert_eq!(b.window_commit(20, 2, CAP), Ok(0));
        assert_eq!((b.first, b.n), (21, 2));
        assert_eq!(b.absorb_check(23), Ok(()), "entries 21 and 22 need the taps of positions 20 and 21");
        // a store with the rejected rows in it no longer matches the tokens: the slot switches itself off instead of drafting from the
        // wrong hidden rows
        let mut c = TapBook::new();
        c.prefill_chunk(0, 20).unwrap();
        c.absorbed(21);
        c.window_commit(20, 4, CAP).unwrap();
        assert_eq!(c.absorb_check(23), Err(BookError::Mismatch));
        assert!(!c.ok);
        assert_eq!(c.absorb_check(23), Err(BookError::Off));
    }

    #[test]
    fn rows_accumulate_until_a_draft_call_absorbs_them_and_a_gap_or_overflow_switches_the_slot_off() {
        let mut b = TapBook::new();
        b.prefill_chunk(0, 10).unwrap();
        // no draft call between three windows: the rows pile up in order behind the carry
        assert_eq!(b.window_commit(10, 1, CAP), Ok(1));
        assert_eq!(b.window_commit(11, 3, CAP), Ok(2));
        assert_eq!(b.window_commit(14, 2, CAP), Ok(5));
        assert_eq!((b.first, b.n), (10, 7));
        assert_eq!(b.absorb_check(17), Ok(()));
        // a window that does not continue where the rows end is a gap
        let mut g = b.clone();
        assert_eq!(g.window_commit(17, 1, CAP), Err(BookError::Gap), "the next row is at position 16");
        assert!(!g.ok);
        // more rows than the store holds
        let mut o = TapBook::new();
        o.prefill_chunk(0, 10).unwrap();
        assert_eq!(o.window_commit(10, CAP, CAP), Err(BookError::Overflow));
        assert!(!o.ok);
        // a switched-off slot stays off, for windows and prefill chunks, until it is reset
        assert_eq!(o.window_commit(10, 1, CAP), Err(BookError::Off));
        assert_eq!(o.prefill_chunk(10, 5), Err(BookError::Off));
        o.reset();
        assert_eq!(o.prefill_chunk(0, 5), Ok(false));
    }

    #[test]
    fn a_prefill_that_does_not_continue_is_refused_and_nothing_pending_means_no_draft() {
        let mut b = TapBook::new();
        assert_eq!(b.prefill_chunk(5, 5), Err(BookError::Gap), "a chunk that is not the first, on a fresh slot");
        let mut b = TapBook::new();
        b.prefill_chunk(0, 8).unwrap();
        assert_eq!(b.prefill_chunk(9, 4), Err(BookError::Gap), "a chunk that skips a row");
        let mut b = TapBook::new();
        b.prefill_chunk(0, 8).unwrap();
        b.absorbed(9);
        assert_eq!(b.absorb_check(9), Err(BookError::Nothing));
        // a slot that is switched off before or during its prefill drafts nothing
        let mut b = TapBook::new();
        b.disable();
        assert_eq!(b.prefill_chunk(0, 8), Err(BookError::Off));
        assert_eq!(b.absorb_check(9), Err(BookError::Off));
    }

    #[test]
    fn a_zero_row_chunk_and_a_one_row_prompt_are_handled() {
        // an empty chunk is a gap, not a silent no-op
        let mut b = TapBook::new();
        assert_eq!(b.prefill_chunk(0, 0), Err(BookError::Gap));
        // a one-token prompt: its only tap is the carry; the first sampled token makes the sequence 2 tokens long
        let mut b = TapBook::new();
        assert_eq!(b.prefill_chunk(0, 1), Ok(false));
        assert_eq!((b.first, b.n), (1, 1));
        assert!(b.can_absorb(2));
    }

    #[test]
    fn a_window_on_a_slot_the_head_never_prefilled_is_a_gap_unless_it_starts_at_position_zero() {
        // a window at position 37 on a fresh slot: no entries exist below it
        let mut b = TapBook::new();
        assert_eq!(b.window_commit(37, 1, CAP), Err(BookError::Gap));
        assert!(!b.ok);
        // a window at position 0 is the start of a sequence decoded without a prompt: its taps are entries 1..
        let mut b = TapBook::new();
        assert_eq!(b.window_commit(0, 2, CAP), Ok(0));
        assert_eq!((b.first, b.n), (1, 2));
        assert!(b.can_absorb(3));
    }

    #[test]
    fn a_window_after_an_absorb_starts_at_the_position_of_the_pending_token() {
        // the window of the round that follows an absorb of a `len`-token sequence starts at position `len - 1`
        let mut b = TapBook::new();
        b.prefill_chunk(0, 50).unwrap();
        b.absorbed(51);
        assert_eq!(b.window_commit(50, 1, CAP), Ok(0));
        assert_eq!((b.first, b.n), (51, 1));
        assert!(b.can_absorb(52));
        // and one that starts anywhere else after rows are pending is a gap
        assert_eq!(b.window_commit(52, 1, CAP), Err(BookError::Gap));
    }
}
