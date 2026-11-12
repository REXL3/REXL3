//! A fake [`NemotronRuntime`] for the backend and engine tests: no GPU, no pack.
//!
//! The model is a hash machine with the two kinds of state the real one has. A slot keeps its committed tokens (the "KV": the history a
//! row's logits read, with the copy behaviour real models have, so the copy drafter gets tokens accepted) and a recurrent state folded
//! over them (the "Mamba state": a thirty-second of positions have their argmax decided by it, so a wrong commit changes tokens).
//!
//! Row invariance and rollback hold BY CONSTRUCTION here, so a test failure is the backend's bookkeeping and never the fake's logic:
//! * row `j` of a window is a pure function of the slot's committed tokens and state and of the window's rows `0..=j`; no other row,
//!   sequence, batch width or call order enters it;
//! * a window touches nothing; `commit(slot, keep)` folds the first `keep` window rows into the slot and drops the window, so rows past
//!   `keep` leave no trace;
//! * a prefill chunk folds its rows the same way, so chunked equals whole.
//!
//! The fake also states the runtime's contract strictly: whatever the backend does that the contract forbids (an unbegun slot, a
//! position gap, a window past 64 rows, a commit past its window, a prefill slice past `prefill_rows`) is returned as an error and
//! recorded in [`FakeRuntime::violations`], which every test that expects a clean run asserts empty.
//!
//! **The prefix cache's fake** (`with_prefix_snapshots`, milestone M5; wrapped as [`SnapFake`], the type that says it can snapshot). It
//! makes what a restore must bring back observable, the way the GLM mock does (`gb_engine::mock::MockBackend::with_glm_shape`):
//! * the context lives IN refcounted 256-token pages of a pool and a row's logits are read from them, so a shared page that someone
//!   clobbered, or one freed while still referenced, changes a reply; pages are never cleared (device memory: a new holder sees what
//!   the previous one wrote);
//! * each page carries a stripe of the head's entries (the 9th stripe of the real pool): entry for position `p` is a function of the tap
//!   of position `p` and the token at `p + 1`, built one position behind the trunk exactly as `mtp::chunk_entries` and the absorb do,
//!   so the entry at position `len - 1` is written AFTER a cut of `len` tokens;
//! * the Mamba state is `(s, conv)`: the conv history is the last three tokens and steers the logits for three rows after a cut;
//! * the head's store of pending taps (row 0 is the carry) is per slot and is not cleared by a begin (a device buffer), its `TapBook` is
//!   the runtime's own, and a draft call's accept rolls depend on a digest of the head's entries of the whole sequence, so a stale or
//!   unrestored entry shows in the speculation counters and never in a reply (a draft only proposes);
//! * snapshots are arena slots (`prefix::SnapshotBook`, the book the real model uses) that hold the state and a reference to the pages;
//!   `SnapFault` makes a restore or a release wrong in one named way.
#![allow(dead_code)]

use gb_cuda::rt::Block;
use gb_cuda::DevBuf;
use gb_engine::mock::mock_logits;
use gb_engine::PrefixHandle;
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::{ModelOptions, NemotronRuntime};
use gb_model::families::nemotron_h_omni::mtp::{DraftAsk, Drafts, MTP_MAX_DEPTH, MTP_TAP_CAP};
use gb_model::families::nemotron_h_omni::prefix::{restore_pages, SnapMeta, SnapshotBook};
use gb_model::families::nemotron_h_omni::tapbook::{BookError, TapBook};
use gb_model::families::nemotron_h_omni::vision::{NemotronImageFeatures, SpliceOp};
use gb_model::forward::Candidates;
use gb_model::rows::RowBatch;
use gb_sampling::{top_candidates, try_choose, Sampling};
use std::sync::Arc;

pub const VOCAB: usize = 512;
pub const SLOTS: usize = 8;
/// What a runtime reads: the widest window and the most candidates, as the real one's buffers are sized.
const WINDOW_ROWS: usize = 64;
const MAX_CANDIDATES: usize = 64;
/// Tokens a KV page holds.
pub const PAGE: usize = 256;

/// One call the backend made on the runtime, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Call {
    Begin(usize),
    End(usize),
    Prefill {
        slot: usize,
        first: usize,
        rows: usize,
        want: Option<usize>,
    },
    Window {
        slots: Vec<usize>,
        rows: usize,
        count: usize,
    },
    Commit {
        slot: usize,
        keep: usize,
    },
    Flush,
    /// One `mtp_draft` call: per ask the slot, the length of the sequence it carried and the depth asked, and per ask the seed of the
    /// sampler it carried (`None`: the head's argmax). One entry per ask, in ask order.
    MtpDraft {
        slots: Vec<usize>,
        lens: Vec<usize>,
        depths: Vec<usize>,
        keyed: Vec<Option<u64>>,
    },
    /// `mtp_disable_slot(slot)`.
    MtpDisable(usize),
    /// `snapshot_slot(slot, len)` and whether it took one.
    Snapshot {
        slot: usize,
        len: usize,
        taken: bool,
    },
    /// `restore_snapshot(slot, handle, len)`.
    Restore {
        slot: usize,
        snap: u32,
        len: usize,
    },
    /// `release_snapshot(handle)`.
    Release {
        snap: u32,
    },
}

/// What the fake's MTP head does when it is asked to draft (`FakeRuntime::with_head`).
#[derive(Clone, Debug)]
pub enum HeadScript {
    /// Each draft is the fake's true next token at its position (what the engine samples there: the keyed draw over the candidates when
    /// the ask carries a sampler, the argmax otherwise), right with probability `accept[depth - 1]` (a hash of the state and the
    /// position, so a run repeats) and a different token otherwise. The chain feeds its own drafts back, as the real head does.
    Truth { accept: Vec<f64> },
    /// Every draft is a token the target does not pick (the true token plus one): acceptance zero, numerically clean.
    Garbage,
    /// The first draft of every chain is a token id outside the vocabulary (a head that mapped its columns wrongly): the backend must
    /// not hand it to the trunk.
    BadIds,
    /// `mtp_draft` fails for `count` calls from its `from_call`-th on (1-based; a fatal error is a lost-state error, else a recoverable
    /// request error). The slots of a failed call are not touched by the fake itself: only `mtp_disable_slot` switches one off.
    Errors { from_call: usize, count: usize, fatal: bool },
}

impl HeadScript {
    /// A head that is always right.
    pub fn perfect() -> Self {
        HeadScript::Truth { accept: vec![1.0; MTP_MAX_DEPTH] }
    }
}

/// The capacity of a slot's store of pending tapped rows: the real runtime's (`mtp::MTP_TAP_CAP`), and the fake's per-slot bookkeeping of
/// the head IS the runtime's own `TapBook`, so the limit and every rule of when a slot switches itself off are the real ones by
/// construction: a gap, an overflow (`n + keep > 32`: also any single commit of more than 32 rows), a sequence that does not match the rows.
pub const TAP_ROWS: usize = MTP_TAP_CAP;

/// Where an injected failure strikes (its first matching call).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    Begin,
    Prefill,
    Window,
    Flush,
    /// `snapshot_slot` fails (nothing may stay allocated).
    Snapshot,
    /// `restore_snapshot` fails.
    Restore,
    /// `enable_snapshots` fails: the device refused the arena.
    Arena,
}

/// What a restore or a release of the fake does wrong, in one named way (the prefix cache's mutants; each is killed by a named test of
/// `nemotron_omni_backend_prefix.rs`). `None` is the honest runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SnapFault {
    #[default]
    None,
    /// The recurrent state S is left as `begin` made it (zero).
    StateNotRestored,
    /// The conv history is left as `begin` made it (zero): wrong for the three rows after the cut.
    ConvNotRestored,
    /// The head's store row 0 (the carry) is left as the previous owner of the slot had it.
    CarryNotRestored,
    /// The head's `TapBook` is left as `begin` reset it: the first resumed chunk is a gap and the slot drafts nothing.
    BookNotRestored,
    /// The page that holds position `len - 1` is shared by reference, never copied (no copy-on-write).
    LastPageShared,
    /// The page that holds position `len - 1` is shared when it is FULL (`len % 256 == 0`) and copied only when it is partial: the rule
    /// of the families without a head one position behind the trunk.
    FullLastPageShared,
    /// The copied page's tokens are copied, the head's stripe of it is not.
    HeadStripeNotCopied,
    /// The state of the next live snapshot slot is restored instead of the handle's.
    WrongSnapshot,
    /// A release keeps the arena slot (it never comes back).
    ReleaseLeaksSlot,
    /// A release keeps the page references (the pages never come back).
    ReleaseKeepsPages,
}

/// The recurrent state: S folded over every token, and the conv history (the last three tokens).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Mamba {
    s: u64,
    conv: [u32; 3],
}

impl Mamba {
    fn push(self, t: u32) -> Self {
        Self { s: fold(self.s, t), conv: [self.conv[1], self.conv[2], t] }
    }

    /// What the logits read: S alone for a plain fake, S and the conv history for the cache's.
    fn eff(self, cache: bool) -> u64 {
        if cache {
            h(self.s ^ 0xC04F_C04F_0001, h(self.conv[0] as u64, h(self.conv[1] as u64, self.conv[2] as u64)))
        } else {
            self.s
        }
    }
}

#[derive(Clone, Default)]
struct SlotState {
    /// The slot a failed `begin_slot` left behind: not usable, and the only thing `end_slot` may be called on without a begin.
    half_begun: bool,
    committed: Vec<u32>,
    mamba: Mamba,
    window: Vec<u32>,
}

/// The pool of the fake's cache mode: pages of tokens and head entries with reference counts, and a block table per slot.
struct Pages {
    /// `token + 1` per position (0: never written), per page.
    tok: Vec<Vec<u32>>,
    /// The head's entry per position, per page (the head's stripe of the page).
    head: Vec<Vec<u64>>,
    refs: Vec<u32>,
    free: Vec<u32>,
    tables: Vec<Vec<u32>>,
}

impl Pages {
    fn new(pages: usize, slots: usize) -> Self {
        Self {
            tok: vec![vec![0; PAGE]; pages],
            head: vec![vec![0; PAGE]; pages],
            refs: vec![0; pages],
            free: (0..pages as u32).rev().collect(),
            tables: vec![Vec::new(); slots],
        }
    }
    fn alloc(&mut self) -> u32 {
        let p = self.free.pop().expect("the fake's page pool is exhausted (a test sized it too small)");
        self.refs[p as usize] = 1;
        p
    }
    fn retain(&mut self, p: u32) {
        assert!(self.refs[p as usize] > 0, "retain of a free page {p}");
        self.refs[p as usize] += 1;
    }
    fn release(&mut self, p: u32) {
        let r = &mut self.refs[p as usize];
        assert!(*r > 0, "double release of page {p}");
        *r -= 1;
        if *r == 0 {
            self.free.push(p);
        }
    }
    fn ensure(&mut self, slot: usize, tokens: usize) {
        while self.tables[slot].len() * PAGE < tokens {
            let p = self.alloc();
            self.tables[slot].push(p);
        }
    }
    fn write_tok(&mut self, slot: usize, pos: usize, t: u32) {
        self.ensure(slot, pos + 1);
        let p = self.tables[slot][pos / PAGE] as usize;
        self.tok[p][pos % PAGE] = t.wrapping_add(1);
    }
    fn read(&self, slot: usize, n: usize) -> Vec<u32> {
        (0..n)
            .map(|pos| match self.tables[slot].get(pos / PAGE) {
                Some(&p) => self.tok[p as usize][pos % PAGE].wrapping_sub(1),
                None => 0,
            })
            .collect()
    }
    fn write_head(&mut self, slot: usize, pos: usize, e: u64) {
        self.ensure(slot, pos + 1);
        let p = self.tables[slot][pos / PAGE] as usize;
        self.head[p][pos % PAGE] = e;
    }
    fn read_head(&self, slot: usize, pos: usize) -> u64 {
        self.tables[slot].get(pos / PAGE).map_or(0, |&p| self.head[p as usize][pos % PAGE])
    }
    fn release_slot(&mut self, slot: usize) {
        for p in std::mem::take(&mut self.tables[slot]) {
            self.release(p);
        }
    }
    fn in_use(&self) -> usize {
        self.refs.iter().filter(|&&r| r > 0).count()
    }
}

/// What an arena slot of the fake holds: the state of the snapshot and (through the handle) a reference to its pages.
#[derive(Clone)]
struct SnapState {
    mamba: Mamba,
    carry: u64,
    /// The tokens the engine committed (the bookkeeping the contract is stated in; the pages hold what the model computed over).
    committed: Vec<u32>,
}

struct CacheModel {
    pages: Pages,
    book: SnapshotBook,
    snaps: Vec<Option<SnapState>>,
    /// The head's store of pending tapped rows per slot (row 0 is the carry across prefill chunks). Device memory: a begin does not clear it.
    stores: Vec<Vec<u64>>,
    fault: SnapFault,
    /// Prefill slices may carry image rows (`prefill_chunk_mm` accepts a splice).
    images: bool,
}

/// One slot as the cache's tests compare it at a cut: what a restored slot must have in common with a cold donor that stopped there. Read
/// from the pages themselves (the physical copy, not the runtime's bookkeeping).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotImage {
    pub tokens: Vec<u32>,
    pub s: u64,
    pub conv: [u32; 3],
    /// The head's entries of positions `0..len - 1` (the entry of position `len - 1` is not part of a snapshot).
    pub head: Vec<u64>,
    /// The head's store row 0.
    pub carry: Option<u64>,
    /// `TapBook { ok, first, n }` of the slot, `None` without a head.
    pub book: Option<(bool, usize, usize)>,
}

pub struct FakeRuntime {
    pub opts: ModelOptions,
    slots: Vec<Option<SlotState>>,
    pub calls: Vec<Call>,
    /// Contract violations by the caller (see the module documentation). Empty after a correct run.
    pub violations: Vec<String>,
    /// An error to return at the first call of a kind; taken when it fires.
    pub fail: Option<(Site, ModelError)>,
    /// Candidates per row a prefill slice returns (the runtime decides it: `want` carries no count).
    pub prefill_count: usize,
    pub mtp: bool,
    /// What the head does when asked to draft (read only when `mtp` is set).
    pub head: HeadScript,
    head_slots: Vec<TapBook>,
    /// `mtp_draft` calls so far (successful or not).
    pub mtp_draft_calls: usize,
    /// Slots whose drafting the fake's own bookkeeping switched off, with why: a store that would have overflowed, or a sequence that
    /// did not match the stored rows. A correct backend never causes either.
    pub head_lost: Vec<(usize, &'static str)>,
    /// The prefix cache's fake (see the module documentation); `None`: the plain fake.
    cache: Option<CacheModel>,
    /// The rows of the slice being run that a splice replaced, `(row, the token the model computed over)`; set by `prefill_chunk_mm`.
    seen_rows: Vec<(usize, u32)>,
}

fn h(a: u64, b: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_add(0xD1B5_4A32_D192_ED03);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 29)
}

fn fold(state: u64, token: u32) -> u64 {
    h(state ^ 0x5851_F42D_4C95_7F2D, token as u64 + 1)
}

/// The tap of a position: a function of everything before it, as a hidden state is.
fn tap_of(eff: u64) -> u64 {
    h(eff, 0xCA77_0000_0001)
}

/// The head's entry that pairs `tap` (of position `p`) with the token at `p + 1`.
fn entry_of(tap: u64, token: u32) -> u64 {
    h(tap, token as u64 + 0x1000_0000)
}

impl FakeRuntime {
    pub fn new(slots: usize) -> Self {
        Self {
            opts: ModelOptions { slots, kv_pages: 64, max_context: 16384, prefill_rows: 64, layers: None, mtp: false },
            slots: vec![None; slots],
            calls: Vec::new(),
            violations: Vec::new(),
            fail: None,
            prefill_count: MAX_CANDIDATES,
            mtp: false,
            head: HeadScript::perfect(),
            head_slots: vec![TapBook::new(); slots],
            mtp_draft_calls: 0,
            head_lost: Vec::new(),
            cache: None,
            seen_rows: Vec::new(),
        }
    }

    /// This runtime has an MTP head that behaves as `script` says.
    pub fn with_head(mut self, script: HeadScript) -> Self {
        self.mtp = true;
        self.head = script;
        self
    }

    /// This runtime keeps its context in refcounted pages and can take, restore and release snapshots (`snapshots` arena slots, `pages` pages
    /// in the pool, `fault` injected). The trait's snapshot methods are [`SnapFake`]'s: this type keeps the defaults, so the plain tests do not
    /// see a cache.
    pub fn with_prefix_snapshots(mut self, snapshots: usize, pages: usize, fault: SnapFault) -> Self {
        let slots = self.slots.len();
        self.opts.kv_pages = pages;
        self.cache = Some(CacheModel {
            pages: Pages::new(pages, slots),
            book: SnapshotBook::new(snapshots),
            snaps: vec![None; snapshots],
            stores: vec![Vec::new(); slots],
            fault,
            images: false,
        });
        self
    }

    /// Prefill slices may carry image rows (the cache's fake only): a row spliced from an image's features is a token that is a function of the
    /// features and the row, so a reply depends on the image's content and not on the placeholder.
    pub fn with_images(mut self) -> Self {
        if let Some(c) = self.cache.as_mut() {
            c.images = true;
        }
        self
    }

    /// The draft calls so far, in order: (slots, sequence lengths, depths, sampler seeds).
    #[allow(clippy::type_complexity)]
    pub fn mtp_calls(&self) -> Vec<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<Option<u64>>)> {
        self.calls
            .iter()
            .filter_map(|c| match c {
                Call::MtpDraft { slots, lens, depths, keyed } => {
                    Some((slots.clone(), lens.clone(), depths.clone(), keyed.clone()))
                }
                _ => None,
            })
            .collect()
    }

    /// Whether the head of `slot` is off, as the fake's bookkeeping sees it (a disabled slot, an overflow or a mismatch).
    pub fn head_off(&self, slot: usize) -> bool {
        self.head_slots.get(slot).is_some_and(|h| !h.ok)
    }

    /// A different token from `t`, in the vocabulary.
    fn other_than(t: u32) -> u32 {
        (t + 1) % VOCAB as u32
    }

    /// The fake's true next tokens after `tokens` (the sequence, ending with the token not yet fed), `depth` of them, each as the chain
    /// of the head would produce it under `script`. With a sampler the pick is the keyed draw over the candidates, as the engine's.
    /// `digest`: the cache's fake mixes a digest of the head's entries of the whole sequence into every accept roll (`None`: the plain fake).
    fn chain(
        committed: &[u32],
        mamba: Mamba,
        tokens: &[u32],
        depth: usize,
        keyed: Option<&Sampling>,
        script: &HeadScript,
        digest: Option<u64>,
    ) -> Drafts {
        let cache = digest.is_some();
        let last = *tokens.last().expect("a sequence has a last token");
        let mut ctx = committed.to_vec();
        ctx.push(last);
        let mut m = mamba.push(last);
        let mut out = Drafts::default();
        for k in 0..depth {
            let position = (tokens.len() + k) as u64;
            let count = keyed.map_or(1, |s| s.candidate_count(VOCAB));
            let (ids, vals) = Self::candidates_after(&ctx, m.eff(cache), count, cache);
            let truth = match keyed {
                Some(s) => try_choose(&vals, &ids, position, s).expect("the fake's candidates are finite"),
                None => ids[0],
            };
            let right = match script {
                HeadScript::Truth { accept } => {
                    let p = accept.get(k).or(accept.last()).copied().unwrap_or(1.0);
                    let base = h(m.eff(cache), position);
                    let base = match digest {
                        Some(d) => h(base, d),
                        None => base,
                    };
                    let roll = h(base, 0xA5A5 + k as u64) as f64 / u64::MAX as f64;
                    roll < p
                }
                _ => false,
            };
            let pick = match script {
                HeadScript::BadIds if k == 0 => VOCAB as u32 + 5,
                _ if right => truth,
                _ => Self::other_than(truth),
            };
            out.vals.push(ids.iter().position(|&i| i == pick).map_or(0.0, |j| vals[j]));
            out.ids.push(pick);
            ctx.push(pick);
            m = m.push(pick);
        }
        out
    }

    pub fn with_prefill_rows(mut self, rows: usize) -> Self {
        self.opts.prefill_rows = rows;
        self
    }

    /// Slots begun and not ended, as the runtime sees them.
    pub fn begun(&self) -> Vec<usize> {
        self.slots.iter().enumerate().filter_map(|(i, s)| s.as_ref().filter(|s| !s.half_begun).map(|_| i)).collect()
    }

    /// Tokens committed in `slot`.
    pub fn committed(&self, slot: usize) -> Option<&[u32]> {
        self.slots.get(slot)?.as_ref().filter(|s| !s.half_begun).map(|s| s.committed.as_slice())
    }

    pub fn count(&self, f: impl Fn(&Call) -> bool) -> usize {
        self.calls.iter().filter(|c| f(c)).count()
    }

    pub fn ends(&self, slot: usize) -> usize {
        self.count(|c| *c == Call::End(slot))
    }

    fn violate(&mut self, what: String) -> ModelError {
        self.violations.push(what.clone());
        ModelError::Request(format!("runtime contract: {what}"))
    }

    fn injected(&mut self, site: Site) -> Option<ModelError> {
        if self.fail.as_ref().is_some_and(|(s, _)| *s == site) {
            return self.fail.take().map(|(_, e)| e);
        }
        None
    }

    /// The candidates after `ctx` with recurrent state `state`. The cache's fake lets the state decide a quarter of the positions (a plain
    /// fake one in thirty-two), so that a restore that gets the state wrong is seen within a few rows.
    fn candidates_after(ctx: &[u32], state: u64, count: usize, cache: bool) -> (Vec<u32>, Vec<f32>) {
        let mut l = mock_logits(ctx, VOCAB);
        let every = if cache { 4 } else { 32 };
        if h(state, 99) % every == 0 {
            l[(h(state, 7) % VOCAB as u64) as usize] = 7.0;
        }
        top_candidates(&l, count)
    }

    /// Tokens committed in `slot`, or a violation when it is not begun.
    fn committed_len(&mut self, slot: usize) -> Result<usize, ModelError> {
        match self.slots.get(slot) {
            Some(Some(s)) if !s.half_begun => Ok(s.committed.len()),
            _ => Err(self.violate(format!("slot {slot} is not begun"))),
        }
    }

    // ---- the one place a token enters a slot, and the one place its context is read (the cache's pages sit behind both)

    fn is_cache(&self) -> bool {
        self.cache.is_some()
    }

    /// The context a slot's logits read: the pages' tokens in the cache's fake (a clobbered shared page changes a reply), else the committed
    /// tokens.
    fn context(&self, slot: usize) -> Vec<u32> {
        let s = self.slots[slot].as_ref().expect("a begun slot");
        match &self.cache {
            Some(c) => c.pages.read(slot, s.committed.len()),
            None => s.committed.clone(),
        }
    }

    /// Commit one token to a begun slot (pages, bookkeeping, state); returns the state the logits read after it.
    fn append(&mut self, slot: usize, t: u32) -> u64 {
        self.append_as(slot, t, t)
    }

    /// [`FakeRuntime::append`] for a row whose context is not its token: `t` is what the engine committed (the bookkeeping the contract is
    /// stated in), `seen` what the model computed over (an image row: a function of the features), which is what the pages hold and the
    /// state folds.
    fn append_as(&mut self, slot: usize, t: u32, seen: u32) -> u64 {
        let pos = self.slots[slot].as_ref().expect("a begun slot").committed.len();
        if let Some(c) = self.cache.as_mut() {
            c.pages.write_tok(slot, pos, seen);
        }
        let cache = self.is_cache();
        let s = self.slots[slot].as_mut().expect("a begun slot");
        s.committed.push(t);
        s.mamba = s.mamba.push(seen);
        s.mamba.eff(cache)
    }

    /// The head's digest over the entries of positions `0..upto` of `slot`'s sequence.
    fn head_digest(&self, slot: usize, upto: usize) -> u64 {
        let c = self.cache.as_ref().expect("the cache's fake");
        let mut d = 0x9E37u64;
        for p in 0..upto {
            d = h(d ^ c.pages.read_head(slot, p), p as u64 + 1);
        }
        d | 1
    }

    // ---- the cache's fake: what the trait's snapshot methods of `SnapFake` do

    /// Pages in use (a holder: a slot or a snapshot), for the books.
    pub fn pages_in_use(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.pages.in_use())
    }

    /// Arena slots holding a snapshot.
    pub fn snapshots_in_use(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.book.live_count())
    }

    /// Pages the slot's table holds.
    pub fn slot_pages(&self, slot: usize) -> Vec<u32> {
        self.cache.as_ref().map(|c| c.pages.tables[slot].clone()).unwrap_or_default()
    }

    /// Holders of a page.
    pub fn page_refs(&self, page: u32) -> u32 {
        self.cache.as_ref().map_or(0, |c| c.pages.refs[page as usize])
    }

    /// Pages of the pool.
    pub fn pool_pages(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.pages.refs.len())
    }

    /// `slot` as a cut compares it (see [`SlotImage`]); `None` when it is not begun.
    pub fn slot_image(&self, slot: usize) -> Option<SlotImage> {
        let s = self.slots.get(slot)?.as_ref().filter(|s| !s.half_begun)?;
        let c = self.cache.as_ref()?;
        let len = s.committed.len();
        Some(SlotImage {
            tokens: c.pages.read(slot, len),
            s: s.mamba.s,
            conv: s.mamba.conv,
            head: (0..len.saturating_sub(1)).map(|p| c.pages.read_head(slot, p)).collect(),
            carry: c.stores[slot].first().copied(),
            book: self.mtp.then(|| {
                let b = &self.head_slots[slot];
                (b.ok, b.first, b.n)
            }),
        })
    }

    fn snap_enabled(&mut self) -> Result<(), ModelError> {
        if self.cache.is_none() {
            return Err(self.violate("a snapshot call on a runtime built without with_prefix_snapshots".into()));
        }
        Ok(())
    }

    fn snap_take(&mut self, slot: usize, len: usize) -> Result<Option<PrefixHandle>, ModelError> {
        self.snap_enabled()?;
        let have = match self.slots.get(slot) {
            Some(Some(s)) if !s.half_begun => Some((s.committed.len(), s.window.len(), s.mamba, s.committed.clone())),
            _ => None,
        };
        let Some((committed, window, mamba, tokens)) = have else {
            self.calls.push(Call::Snapshot { slot, len, taken: false });
            return Err(self.violate(format!("a snapshot of slot {slot}, which is not begun")));
        };
        if committed != len || len == 0 {
            self.calls.push(Call::Snapshot { slot, len, taken: false });
            return Err(self.violate(format!("a snapshot of {len} tokens of slot {slot}, which has {committed} committed")));
        }
        if window != 0 {
            self.calls.push(Call::Snapshot { slot, len, taken: false });
            return Err(self.violate(format!("a snapshot of slot {slot} with a window of {window} rows open")));
        }
        if let Some(e) = self.injected(Site::Snapshot) {
            self.calls.push(Call::Snapshot { slot, len, taken: false });
            return Err(e);
        }
        // the flag is "the head's book was sound": a head that is off drafts nothing after a restore, as it would not have uncached
        let mtp_ok = if self.mtp { self.head_slots[slot].ok } else { true };
        let c = self.cache.as_mut().expect("checked");
        let Some(id) = c.book.take(SnapMeta { len, mtp_ok }) else {
            self.calls.push(Call::Snapshot { slot, len, taken: false });
            return Ok(None);
        };
        let total = len.div_ceil(PAGE);
        let pages: Vec<u32> = c.pages.tables[slot][..total].to_vec();
        for &p in &pages {
            c.pages.retain(p);
        }
        let carry = c.stores[slot].first().copied().unwrap_or(0);
        c.snaps[id as usize] = Some(SnapState { mamba, carry, committed: tokens });
        self.calls.push(Call::Snapshot { slot, len, taken: true });
        Ok(Some(PrefixHandle { snap: id, pages }))
    }

    fn snap_restore(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), ModelError> {
        self.snap_enabled()?;
        self.calls.push(Call::Restore { slot, snap: handle.snap, len });
        let begun_empty = match self.slots.get(slot) {
            Some(Some(s)) if !s.half_begun => Some(s.committed.is_empty() && s.window.is_empty()),
            _ => None,
        };
        match begun_empty {
            None => return Err(self.violate(format!("a restore into slot {slot}, which is not begun"))),
            Some(false) => return Err(self.violate(format!("a restore into slot {slot}, which is not empty"))),
            Some(true) => {}
        }
        if let Some(e) = self.injected(Site::Restore) {
            return Err(e);
        }
        let mtp = self.mtp;
        let c = self.cache.as_mut().expect("checked");
        // the checks the real runtime makes (a refused restore changes nothing)
        let meta = c.book.meta(handle.snap).filter(|m| m.len == len && len > 0 && handle.pages.len() == len.div_ceil(PAGE));
        let Some(meta) = meta else {
            return Err(ModelError::Request(format!(
                "restore of snapshot {} at {len} tokens refused: not live, another length, or another page count",
                handle.snap
            )));
        };
        // WrongSnapshot: the state is read from the next live arena slot, the checks above were made on the handle's own
        let src = if c.fault == SnapFault::WrongSnapshot {
            let n = c.snaps.len() as u32;
            (1..n).map(|d| (handle.snap + d) % n).find(|&i| c.snaps[i as usize].is_some()).unwrap_or(handle.snap)
        } else {
            handle.snap
        };
        let snap = c.snaps[src as usize].clone().expect("a live snapshot holds its state");
        // the engine's tokens are the handle's own: only the model's state is read from `src`
        let committed = c.snaps[handle.snap as usize].as_ref().expect("live").committed.clone();
        // pages `0..last` by reference, the page of position len - 1 copied (the rule of `prefix::restore_pages`) unless the fault says otherwise
        let rp = restore_pages(len);
        for &p in &handle.pages[..rp.shared] {
            c.pages.retain(p);
            c.pages.tables[slot].push(p);
        }
        let last = handle.pages[rp.shared];
        let aligned = len % PAGE == 0;
        let share = match c.fault {
            SnapFault::LastPageShared => true,
            SnapFault::FullLastPageShared => aligned,
            _ => false,
        };
        if share {
            c.pages.retain(last);
            c.pages.tables[slot].push(last);
        } else {
            let new = c.pages.alloc();
            c.pages.tok[new as usize] = c.pages.tok[last as usize].clone();
            if c.fault != SnapFault::HeadStripeNotCopied {
                c.pages.head[new as usize] = c.pages.head[last as usize].clone();
            }
            c.pages.tables[slot].push(new);
        }
        let fault = c.fault;
        let s = self.slots[slot].as_mut().expect("begun");
        s.committed = committed;
        if fault != SnapFault::StateNotRestored {
            s.mamba.s = snap.mamba.s;
        }
        if fault != SnapFault::ConvNotRestored {
            s.mamba.conv = snap.mamba.conv;
        }
        if mtp {
            if fault != SnapFault::CarryNotRestored {
                self.cache.as_mut().expect("checked").stores[slot] = vec![snap.carry];
            }
            if fault != SnapFault::BookNotRestored {
                self.head_slots[slot] = TapBook { ok: meta.mtp_ok, first: len, n: 1 };
            }
        }
        Ok(())
    }

    fn snap_release(&mut self, handle: PrefixHandle) -> Result<(), ModelError> {
        self.snap_enabled()?;
        self.calls.push(Call::Release { snap: handle.snap });
        let c = self.cache.as_mut().expect("checked");
        let fault = c.fault;
        if fault == SnapFault::ReleaseLeaksSlot {
            // the arena slot is never given back: not even its state is dropped
            if c.book.meta(handle.snap).is_none() {
                return Err(ModelError::Request(format!("snapshot {} is not live", handle.snap)));
            }
        } else {
            c.book.release(handle.snap).map_err(ModelError::Request)?;
            c.snaps[handle.snap as usize] = None;
        }
        if fault != SnapFault::ReleaseKeepsPages {
            for &p in &handle.pages {
                c.pages.release(p);
            }
        }
        Ok(())
    }
}

impl NemotronRuntime for FakeRuntime {
    fn vocab_size(&self) -> usize {
        VOCAB
    }

    fn has_mtp(&self) -> bool {
        self.mtp
    }

    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.calls.push(Call::Begin(slot));
        if let Some(e) = self.injected(Site::Begin) {
            // a begin that failed may have left the slot half-made: the one slot the caller may end without having begun it
            if let Some(s) = self.slots.get_mut(slot) {
                *s = Some(SlotState { half_begun: true, ..SlotState::default() });
            }
            return Err(e);
        }
        // a runtime refuses to begin a slot that is already begun: the backend ends a live slot first
        if matches!(self.slots.get(slot), Some(Some(s)) if !s.half_begun) {
            return Err(self.violate(format!("begin_slot of slot {slot}, which is already begun (AlreadyBegun)")));
        }
        match self.slots.get_mut(slot) {
            Some(s) => {
                *s = Some(SlotState::default());
                if let Some(h) = self.head_slots.get_mut(slot) {
                    h.reset();
                }
                if let Some(c) = self.cache.as_mut() {
                    c.pages.release_slot(slot);
                }
                Ok(())
            }
            None => Err(self.violate(format!("slot {slot} is past the pool"))),
        }
    }

    fn end_slot(&mut self, slot: usize) {
        self.calls.push(Call::End(slot));
        // the runtime's `end_slot` is idempotent, but the backend is held to ending a slot once and only a slot it made
        if !matches!(self.slots.get(slot), Some(Some(_))) {
            self.violations
                .push(format!("end_slot of slot {slot}, which is not begun (a second end, or one that was never begun)"));
        }
        if let Some(s) = self.slots.get_mut(slot) {
            *s = None;
        }
        if let Some(h) = self.head_slots.get_mut(slot) {
            h.reset();
        }
        if let Some(c) = self.cache.as_mut() {
            c.pages.release_slot(slot);
        }
    }

    fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        let slot = b.seq_slot.first().map_or(usize::MAX, |&s| s as usize);
        self.calls.push(Call::Prefill { slot, first: b.positions.first().map_or(0, |&p| p as usize), rows: b.rows, want });
        if let Some(e) = self.injected(Site::Prefill) {
            return Err(e);
        }
        if b.n_seq() != 1 {
            return Err(self.violate(format!("a prefill slice is one sequence, got {}", b.n_seq())));
        }
        if b.rows == 0 || b.rows > self.opts.prefill_rows {
            return Err(self.violate(format!("a prefill slice takes 1..={} rows, got {}", self.opts.prefill_rows, b.rows)));
        }
        if let Some(r) = want {
            if r >= b.rows {
                return Err(self.violate(format!("want row {r} of a {}-row slice", b.rows)));
            }
        }
        let prefill_count = self.prefill_count;
        let at = b.positions[0] as usize;
        let have = self.committed_len(slot)?;
        if at != have {
            return Err(self.violate(format!("a prefill slice at position {at} continues a context of {have} tokens")));
        }
        let cache = self.is_cache();
        let mut out = None;
        let mut taps = Vec::with_capacity(b.rows);
        // the rows a splice replaced: the model computed over a token that is a function of the image's features, the engine committed the
        // placeholder (see `prefill_chunk_mm`)
        let seen_rows = std::mem::take(&mut self.seen_rows);
        let seen_of = |r: usize, t: u32| seen_rows.iter().find(|(row, _)| *row == r).map_or(t, |&(_, s)| s);
        for (r, &t) in b.ids.iter().enumerate() {
            let t = t as u32;
            let eff = self.append_as(slot, t, seen_of(r, t));
            if cache {
                taps.push(tap_of(eff));
            }
            if want == Some(r) {
                let ctx = self.context(slot);
                let (ids, vals) = Self::candidates_after(&ctx, eff, prefill_count, cache);
                out = Some(Candidates { rows: 1, count: ids.len(), ids: ids.into_iter().map(|i| i as i32).collect(), vals });
            }
        }
        // the head's store: a prompt chunk continues the carry (or starts the sequence) and leaves its last tap as the new carry
        if self.mtp {
            match self.head_slots[slot].prefill_chunk(at, b.rows) {
                Ok(_) => {
                    if cache {
                        // the entries of the chunk's positions: the token of row j pairs with the tap before it (the carry for row 0), at the
                        // position of its tap, one before its own: so the entry of the chunk's last position waits for the next chunk
                        let c = self.cache.as_mut().expect("cache");
                        let carry = c.stores[slot].first().copied().unwrap_or(0);
                        for (j, &t) in b.ids.iter().enumerate() {
                            let p = at + j;
                            if p == 0 {
                                continue;
                            }
                            let before = if j == 0 { carry } else { taps[j - 1] };
                            // an image row's entry takes the feature row as its embedding half, as the real head's hook does
                            c.pages.write_head(slot, p - 1, entry_of(before, seen_of(j, t as u32)));
                        }
                        c.stores[slot] = vec![taps[b.rows - 1]];
                    }
                }
                Err(BookError::Off) => {}
                Err(_) => self.head_lost.push((slot, "a prefill chunk did not continue the store")),
            }
        }
        Ok(out)
    }

    /// With the cache's fake and `with_images`, the rows a splice names are tokens derived from the image's features and the feature row; the
    /// default of the trait (an empty splice is a plain slice, any other is refused by name) otherwise.
    fn prefill_chunk_mm(
        &mut self,
        b: &RowBatch,
        want: Option<usize>,
        splice: &[SpliceOp],
    ) -> Result<Option<Candidates>, ModelError> {
        if splice.is_empty() {
            return self.prefill_chunk(b, want);
        }
        if !self.cache.as_ref().is_some_and(|c| c.images) {
            return Err(ModelError::NotBuilt(
                "this Nemotron runtime cannot splice image features (nemotron_radio_vision is the model's, not this runtime's)"
                    .into(),
            ));
        }
        // what the model computes over at an image row: a token no text id can be, a function of the image's content (its buffer's identity
        // here) and of the feature row. The engine committed the placeholder; the pages, the state and the head's entry take this one.
        let mut seen = Vec::new();
        for op in splice {
            let id = op.feats.buf.ptr() as u64;
            for k in 0..op.n {
                seen.push((op.dst_row + k, 0x4000_0000 | (h(id, (op.src_row + k) as u64) as u32 & 0x3FFF_FFFF)));
            }
        }
        self.seen_rows = seen;
        let r = self.prefill_chunk(b, want);
        self.seen_rows.clear();
        r
    }

    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        self.calls.push(Call::Window { slots: b.seq_slot.iter().map(|&s| s as usize).collect(), rows: b.rows, count });
        if let Some(e) = self.injected(Site::Window) {
            return Err(e);
        }
        if b.rows == 0 || b.rows > WINDOW_ROWS {
            return Err(self.violate(format!("a window takes 1..={WINDOW_ROWS} rows, got {}", b.rows)));
        }
        if count == 0 || count > MAX_CANDIDATES {
            return Err(self.violate(format!("candidate count must be 1..={MAX_CANDIDATES}, got {count}")));
        }
        let cache = self.is_cache();
        let mut ids = Vec::with_capacity(b.rows * count);
        let mut vals = Vec::with_capacity(b.rows * count);
        for i in 0..b.n_seq() {
            let slot = b.seq_slot[i] as usize;
            let rows = b.seq_rows(i);
            let at = b.positions[rows.start] as usize;
            let tokens: Vec<u32> = b.ids[rows.clone()].iter().map(|&t| t as u32).collect();
            let have = self.committed_len(slot)?;
            if at != have {
                return Err(self.violate(format!("a window at position {at} continues a context of {have} tokens")));
            }
            let mut ctx = self.context(slot);
            let mut m = self.slots[slot].as_ref().unwrap().mamba;
            self.slots[slot].as_mut().unwrap().window = tokens.clone();
            for &t in &tokens {
                ctx.push(t);
                m = m.push(t);
                let (i_, v_) = Self::candidates_after(&ctx, m.eff(cache), count, cache);
                ids.extend(i_.into_iter().map(|x| x as i32));
                vals.extend(v_);
            }
        }
        Ok(Candidates { rows: b.rows, count, ids, vals })
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        self.calls.push(Call::Commit { slot, keep });
        let Some(Some(s)) = self.slots.get_mut(slot).filter(|s| s.as_ref().is_some_and(|s| !s.half_begun)) else {
            self.violations.push(format!("commit of slot {slot}, which is not begun"));
            return;
        };
        if keep == 0 || keep > WINDOW_ROWS || keep > s.window.len() {
            let have = s.window.len();
            self.violations.push(format!("commit of {keep} rows of slot {slot}'s window of {have}"));
            return;
        }
        let window = std::mem::take(&mut s.window);
        let row0_pos = s.committed.len();
        let cache = self.is_cache();
        let mut taps = Vec::with_capacity(keep);
        for &t in &window[..keep] {
            let eff = self.append(slot, t);
            if cache {
                taps.push(tap_of(eff));
            }
        }
        // the head's store takes the kept rows (the runtime's `TapBook`): a window that does not continue the stored rows, or one that
        // would overflow the store, switches the slot's drafting off, silently, as the real one does
        if self.mtp {
            match self.head_slots[slot].window_commit(row0_pos, keep, TAP_ROWS) {
                Ok(dst) => {
                    if let Some(c) = self.cache.as_mut() {
                        let st = &mut c.stores[slot];
                        st.truncate(dst);
                        st.extend(taps);
                    }
                }
                Err(BookError::Off) => {}
                Err(BookError::Overflow) => self.head_lost.push((slot, "the store of pending rows overflowed")),
                Err(_) => self.head_lost.push((slot, "a window did not continue the stored rows")),
            }
        }
    }

    fn flush(&mut self) -> Result<(), ModelError> {
        self.calls.push(Call::Flush);
        match self.injected(Site::Flush) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn mtp_draft(&mut self, asks: &[DraftAsk<'_>], keyed: &[Option<Sampling>]) -> Result<Vec<Option<Drafts>>, ModelError> {
        self.calls.push(Call::MtpDraft {
            slots: asks.iter().map(|a| a.slot).collect(),
            lens: asks.iter().map(|a| a.tokens.len()).collect(),
            depths: asks.iter().map(|a| a.depth).collect(),
            keyed: keyed.iter().map(|k| k.as_ref().map(|s| s.seed)).collect(),
        });
        self.mtp_draft_calls += 1;
        if !self.mtp {
            return Err(self.violate("mtp_draft on a runtime that has no head".into()));
        }
        if keyed.len() != asks.len() {
            return Err(self.violate(format!("{} samplers for {} asks", keyed.len(), asks.len())));
        }
        if asks.is_empty() {
            return Err(self.violate("an empty draft call".into()));
        }
        if let HeadScript::Errors { from_call, count, fatal } = self.head {
            if self.mtp_draft_calls >= from_call && self.mtp_draft_calls - from_call < count {
                return Err(if fatal {
                    ModelError::StateLost("the head's cache was lost".into())
                } else {
                    ModelError::Request("scripted head failure".into())
                });
            }
        }
        let cache = self.is_cache();
        let mut seen = std::collections::BTreeSet::new();
        let mut out = Vec::with_capacity(asks.len());
        for (a, k) in asks.iter().zip(keyed) {
            if a.depth > MTP_MAX_DEPTH {
                return Err(self.violate(format!("a draft depth of {} (at most {MTP_MAX_DEPTH})", a.depth)));
            }
            if !seen.insert(a.slot) {
                return Err(self.violate(format!("slot {} appears twice in one draft call", a.slot)));
            }
            let slot = match self.slots.get(a.slot) {
                Some(Some(s)) if !s.half_begun => s.clone(),
                _ => return Err(self.violate(format!("a draft for slot {}, which is not begun", a.slot))),
            };
            if a.tokens.len() != slot.committed.len() + 1 || a.tokens[..a.tokens.len() - 1] != slot.committed[..] {
                return Err(self.violate(format!(
                    "slot {}'s ask carries {} tokens that are not its {} committed tokens and one more",
                    a.slot,
                    a.tokens.len(),
                    slot.committed.len()
                )));
            }
            let (first, pending) = (self.head_slots[a.slot].first, self.head_slots[a.slot].n);
            match self.head_slots[a.slot].absorb_check(a.tokens.len()) {
                Ok(()) => {}
                Err(BookError::Off | BookError::Nothing) => {
                    out.push(None);
                    continue;
                }
                Err(_) => {
                    self.head_lost.push((a.slot, "the sequence did not match the stored rows"));
                    out.push(None);
                    continue;
                }
            }
            // the absorb: every pending row is built, the store is empty. The cache's fake builds the entries: entry `m` pairs the tap of
            // position `m - 1` (store row `m - first`) with the token at `m`, and sits at position `m - 1`
            if cache {
                let c = self.cache.as_mut().expect("cache");
                for i in 0..pending {
                    let m = first + i;
                    if m == 0 {
                        continue;
                    }
                    let tap = c.stores[a.slot].get(i).copied().unwrap_or(0);
                    c.pages.write_head(a.slot, m - 1, entry_of(tap, a.tokens[m]));
                }
                c.stores[a.slot].clear();
            }
            self.head_slots[a.slot].absorbed(a.tokens.len());
            if a.depth == 0 {
                out.push(Some(Drafts::default()));
                continue;
            }
            let ctx = self.context(a.slot);
            let digest = cache.then(|| self.head_digest(a.slot, a.tokens.len() - 1));
            out.push(Some(Self::chain(&ctx, slot.mamba, a.tokens, a.depth, k.as_ref(), &self.head, digest)));
        }
        Ok(out)
    }

    fn mtp_disable_slot(&mut self, slot: usize) {
        self.calls.push(Call::MtpDisable(slot));
        if let Some(h) = self.head_slots.get_mut(slot) {
            h.disable();
        }
    }

    fn mtp_pending_rows(&self, slot: usize) -> Option<usize> {
        if !self.mtp {
            return None;
        }
        let b = self.head_slots.get(slot)?;
        b.ok.then_some(b.n)
    }
}

// ------------------------------------------------------------------------------------------------------------------ the snapshotting type

/// [`FakeRuntime`] as a runtime TYPE that says it can snapshot: the plan carves a cache only for such a type
/// (`NemotronRuntime::snapshots_supported`), so the plain [`FakeRuntime`] of the existing tests is planned exactly as before. Build it with
/// [`SnapFake::new`]; everything of the fake is reachable through `Deref`.
pub struct SnapFake(pub FakeRuntime);

impl SnapFake {
    /// `slots` slots, `snapshots` arena slots, a pool of `pages` pages, `fault` injected. The head is on (`HeadScript::perfect`) unless the
    /// caller replaces it.
    pub fn new(slots: usize, snapshots: usize, pages: usize, fault: SnapFault) -> Self {
        Self(FakeRuntime::new(slots).with_head(HeadScript::perfect()).with_prefix_snapshots(snapshots, pages, fault))
    }
}

impl std::ops::Deref for SnapFake {
    type Target = FakeRuntime;
    fn deref(&self) -> &FakeRuntime {
        &self.0
    }
}

impl std::ops::DerefMut for SnapFake {
    fn deref_mut(&mut self) -> &mut FakeRuntime {
        &mut self.0
    }
}

/// An image's features for the fake: a host-backed buffer whose address is the image's identity (two images are two buffers, one image is one
/// `Arc`), `tokens` rows in a `merged` grid.
pub fn fake_features(tokens: usize, merged: (usize, usize)) -> Arc<NemotronImageFeatures> {
    let block = Arc::new(Block::alloc_pageable_host(4096).expect("pageable host memory needs no device"));
    Arc::new(NemotronImageFeatures { buf: DevBuf::view(&block, 0, 4096), tokens, merged, permit: None })
}

impl NemotronRuntime for SnapFake {
    fn snapshots_supported() -> bool {
        true
    }
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
    fn has_mtp(&self) -> bool {
        self.0.has_mtp()
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.0.begin_slot(slot)
    }
    fn end_slot(&mut self, slot: usize) {
        self.0.end_slot(slot)
    }
    fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        self.0.prefill_chunk(b, want)
    }
    fn prefill_chunk_mm(
        &mut self,
        b: &RowBatch,
        want: Option<usize>,
        splice: &[SpliceOp],
    ) -> Result<Option<Candidates>, ModelError> {
        self.0.prefill_chunk_mm(b, want, splice)
    }
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        self.0.window_candidates(b, count)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.0.commit(slot, keep)
    }
    fn flush(&mut self) -> Result<(), ModelError> {
        self.0.flush()
    }
    fn mtp_draft(&mut self, asks: &[DraftAsk<'_>], keyed: &[Option<Sampling>]) -> Result<Vec<Option<Drafts>>, ModelError> {
        self.0.mtp_draft(asks, keyed)
    }
    fn mtp_disable_slot(&mut self, slot: usize) {
        self.0.mtp_disable_slot(slot)
    }
    fn mtp_pending_rows(&self, slot: usize) -> Option<usize> {
        self.0.mtp_pending_rows(slot)
    }

    fn enable_snapshots(&mut self, slots: usize) -> Result<(), ModelError> {
        if let Some(e) = self.0.injected(Site::Arena) {
            return Err(e);
        }
        // the arena exists from construction (`with_prefix_snapshots`); the boot's request must match what was built
        match self.0.cache.as_ref() {
            Some(c) if c.book.capacity() == slots => Ok(()),
            Some(c) => Err(ModelError::Config(format!(
                "the fake was built with {} snapshot slots, the plan carved {slots}",
                c.book.capacity()
            ))),
            None => Err(ModelError::Config("the fake was built without snapshots".into())),
        }
    }
    fn snapshot_arena_bytes(&self) -> u64 {
        // what the real arena would hold: `slots x snapshot_bytes(state, mtp)`, with the real figures (the plan counts them the same way)
        let slots = self.0.cache.as_ref().map_or(0, |c| c.book.capacity() as u64);
        slots * gb_model::families::nemotron_h_omni::prefix::snapshot_bytes(170_229_760, self.0.mtp)
    }
    fn free_snapshots(&self) -> usize {
        self.0.cache.as_ref().map_or(0, |c| c.book.free_count())
    }
    fn snapshot_slot(&mut self, slot: usize, len: usize) -> Result<Option<PrefixHandle>, ModelError> {
        self.0.snap_take(slot, len)
    }
    fn restore_snapshot(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), ModelError> {
        self.0.snap_restore(slot, handle, len)
    }
    fn release_snapshot(&mut self, handle: PrefixHandle) -> Result<(), ModelError> {
        self.0.snap_release(handle)
    }
}
