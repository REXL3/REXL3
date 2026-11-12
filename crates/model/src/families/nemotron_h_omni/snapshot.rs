//! The prefix cache's side of the Nemotron model (milestone M5, GPU side): a pool of snapshot slots allocated once at boot, the snapshot of a
//! slot after a prefill slice, and the restore that puts another (begun, empty) slot in that state. Design and numbers:
//! `docs/NEMOTRON_LONG_CONTEXT.md`; the arithmetic (snapshot bytes, which page is copied, the arena's free list) is [`super::prefix`], the same
//! code the CPU tests run against the fake runtime.
//!
//! A snapshot is taken between two prefill slices, when the slot's state is quiet (a slice commits every row it ran, in the layer, at
//! `keep = rows`), so that every copy is a plain device-to-device copy of what the slice left, enqueued on the model's own stream:
//!
//! * the 40 Mamba2 layers' S (f32 `[128][128][64]`, 4 MiB) and conv history (bf16 `[3][10,240]`, 60 KiB) of the slot, into the arena slot;
//! * the head's carry (the post-`norm_f` tap of position `len - 1`, f32 `[4096]`, row 0 of the slot's tap store) when the head's
//!   bookkeeping is sound for exactly `len` (`{ ok, first: len, n: 1 }`); the snapshot records the flag and nothing else of the book;
//! * a reference to the first `ceil(len / 256)` KV pages (`KvTables::retain_prefix`): the 8 trunk layers' stripes and the head's are the
//!   same pages. Nothing is copied at snapshot time, so a snapshot allocates no page and cannot fail for want of one.
//!
//! A restore starts from a slot `begin_slot` has just emptied. It shares pages `0..last` by reference and **copies page
//! `last = (len - 1) / 256` into a page of the slot's own, in every stripe, the head's included, always** (`prefix::restore_pages`): the
//! head writes its entry for position `len - 1` AFTER the cut, so even a full last page (`len % 256 == 0`) is not safe to share the way the
//! other families share it. Then S, conv, the carry go back, the head's book is `{ ok, first: len, n: 1 }` (without it the first chunk after
//! the restore finds a gap and the slot silently stops drafting) and the slot's length is `len`.
//!
//! **Why the bits are those of a cold prefill.** Every row a slice writes (the KV rows of the 8 attention layers, the head's entries) is a
//! function of its token and the rows before it whatever the slice grid (`docs/NEMOTRON_ATTN_LONG.md`, `docs/NEMOTRON_MOE_PREFILL.md`); S and
//! conv are the committed state after those rows; the only other thing the next row reads is the head's carry. All three are copied, and the
//! pages below the cut are never written again by anyone (the donor continues at `len`, a restorer's writes land in its own copy).
//!
//! **Handles.** `PrefixHandle { snap, pages }` is the engine's and carries no more. `snap` is the arena slot in its low 16 bits and a tag in
//! the high 16 (a hash of the model's process-wide id, the slot, how many snapshots this arena has taken, the length and the pages), and
//! the arena keeps the snapshot's own pages and checksum per slot: a handle of another model, a stale one (released, the slot taken again), a
//! page list that is not the snapshot's, a length that is not its length, and a double release are each refused by name and change nothing.

use super::model::{Core, KvGeom, NemotronModel, StatePools, CONV_SLOT_BYTES, S_SLOT_BYTES};
use super::mtp::MtpState;
use super::prefix::{self, SnapMeta, SnapshotBook};
use crate::error::ModelError;
use crate::kvtables::BLOCK_TOKENS;
use gb_cuda::nemotron_attn::KV_TOKEN_ELEMS;
use gb_cuda::DevBuf;
use gb_engine::PrefixHandle;
use std::sync::atomic::{AtomicU64, Ordering};

/// Bytes of one page in one stripe, K (or V): 256 tokens x 2 kv heads x 128 x fp16 = 128 KiB.
pub const PAGE_STRIPE_BYTES: usize = BLOCK_TOKENS * KV_TOKEN_ELEMS * 2;
// (an int8 pool's page is `KvGeom::page_bytes`, 66,048 bytes: the copies and digests below use the pool's own page size)

/// The most snapshot slots an arena can hold: the slot index is the low 16 bits of a handle's `snap`.
pub const MAX_SNAPSHOT_SLOTS: usize = 0xFFFF;

/// A deliberate bug in the restore, so the gates can show that they notice it (`mutants_are_caught` in `tests/nemotron_omni_prefix_gpu.rs`).
/// Production sets none.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapMutant {
    /// S is left as `begin_slot` made it (zeros).
    StateNotRestored,
    /// The conv history is left as `begin_slot` made it.
    ConvNotRestored,
    /// The head's carry is not copied: the store's row 0 is whatever the slot's previous owner left.
    CarryNotRestored,
    /// The head's book is left as `begin_slot` reset it (`first: 0, n: 0`): the first chunk after the restore finds a gap.
    BookNotRestored,
    /// The page of position `len - 1` is shared with the entry, never copied.
    LastPageShared,
    /// That page is shared when it is FULL (`len % 256 == 0`), the other families' rule (they copy only a partial page).
    FullLastPageShared,
    /// The page is copied in the trunk's stripes and not in the head's.
    HeadStripeNotCopied,
}

/// Process-wide ids of arenas, so a handle of another model's arena is told from this one's.
static ARENAS: AtomicU64 = AtomicU64::new(1);

fn fnv(h: u64, x: u64) -> u64 {
    (h ^ x).wrapping_mul(0x100_0000_01b3)
}

/// What the arena keeps about one live snapshot besides its meta: the pages it holds a reference to and the checksum of its bookkeeping.
#[derive(Clone, Debug)]
struct SnapRec {
    /// The high 16 bits of the handle's `snap`: the low 16 bits of `check`.
    tag: u16,
    /// Which snapshot of this arena it was (1, 2, ...): a slot taken again is another take.
    take: u64,
    pages: Vec<u32>,
    check: u64,
}

/// The snapshot slots of the model: one allocation of `slots x stride` bytes, per slot `[S of every Mamba layer | conv of every layer |
/// carry]`, and the books ([`prefix::SnapshotBook`] for the free list and `{ len, mtp_ok }`, [`SnapRec`] for pages, tag and checksum).
pub struct SnapArena {
    buf: DevBuf,
    stride: usize,
    mamba_layers: usize,
    mtp: bool,
    book: SnapshotBook,
    rec: Vec<Option<SnapRec>>,
    uid: u64,
    takes: u64,
    mutant: Option<SnapMutant>,
}

impl SnapArena {
    /// Bytes of one slot of an arena over `mamba_layers` Mamba layers, with the head's carry when `mtp`: [`prefix::snapshot_bytes`] of the
    /// state of one slot, which is what the memory plan counts.
    pub fn slot_bytes(mamba_layers: usize, mtp: bool) -> usize {
        prefix::snapshot_bytes(StatePools { slots: 1, layers: mamba_layers }.slot_bytes() as u64, mtp) as usize
    }

    /// Allocates `slots` snapshot slots (not zeroed: a slot is written by a snapshot before a restore reads it, and the book says which hold
    /// one). The allocation is one `cudaMalloc`; a device that refuses it is the caller's named boot error.
    pub fn new(mamba_layers: usize, slots: usize, mtp: bool) -> Result<Self, ModelError> {
        if slots == 0 || slots > MAX_SNAPSHOT_SLOTS {
            return Err(ModelError::Config(format!("a snapshot arena holds 1..={MAX_SNAPSHOT_SLOTS} slots, got {slots}")));
        }
        if mamba_layers == 0 {
            return Err(ModelError::Config("a snapshot arena needs at least one Mamba layer (the model has none)".into()));
        }
        let stride = Self::slot_bytes(mamba_layers, mtp);
        let buf = DevBuf::alloc(slots * stride)?;
        Ok(Self {
            buf,
            stride,
            mamba_layers,
            mtp,
            book: SnapshotBook::new(slots),
            rec: vec![None; slots],
            uid: ARENAS.fetch_add(1, Ordering::Relaxed),
            takes: 0,
            mutant: None,
        })
    }

    /// Device bytes of the arena.
    pub fn bytes(&self) -> u64 {
        self.buf.bytes() as u64
    }

    pub fn slots(&self) -> usize {
        self.book.capacity()
    }

    pub fn free_count(&self) -> usize {
        self.book.free_count()
    }

    fn s_off(&self, id: u32, layer: usize) -> usize {
        id as usize * self.stride + layer * S_SLOT_BYTES
    }

    fn conv_off(&self, id: u32, layer: usize) -> usize {
        id as usize * self.stride + self.mamba_layers * S_SLOT_BYTES + layer * CONV_SLOT_BYTES
    }

    fn carry_off(&self, id: u32) -> usize {
        id as usize * self.stride + self.mamba_layers * (S_SLOT_BYTES + CONV_SLOT_BYTES)
    }

    /// The checksum of one snapshot's bookkeeping: this arena, its slot, the take it was, its length, the head flag and every page.
    fn check_of(&self, id: u32, take: u64, meta: SnapMeta, pages: &[u32]) -> u64 {
        let mut h = fnv(0xcbf2_9ce4_8422_2325, self.uid);
        h = fnv(h, id as u64);
        h = fnv(h, take);
        h = fnv(h, meta.len as u64);
        h = fnv(h, meta.mtp_ok as u64);
        for &p in pages {
            h = fnv(h, p as u64);
        }
        h
    }

    /// The live snapshot `handle` names, or the refusal. Nothing changes. `len` is the length the caller means to use it for, when it knows.
    fn lookup(&self, handle: &PrefixHandle, len: Option<usize>) -> Result<(u32, SnapMeta), ModelError> {
        let (id, tag) = (handle.snap & 0xFFFF, (handle.snap >> 16) as u16);
        let bad = |m: String| Err(ModelError::Request(m));
        if id as usize >= self.slots() {
            return bad(format!(
                "snapshot handle {:#x} does not belong to this arena: slot {id} is out of range, the arena has {} slots (another model's handle?)",
                handle.snap,
                self.slots()
            ));
        }
        let Some(meta) = self.book.meta(id) else {
            return bad(format!("snapshot {id} is not live (released already, or never taken)"));
        };
        let rec = self.rec[id as usize].as_ref().expect("a live slot has its record");
        if rec.tag != tag {
            return bad(format!(
                "snapshot handle {:#x} does not belong to this arena: slot {id} holds another snapshot (a stale handle, or another model's)",
                handle.snap
            ));
        }
        if rec.pages != handle.pages {
            return bad(format!(
                "the {} pages of the handle are not the snapshot {id}'s ({} pages): the handle was not given by this arena",
                handle.pages.len(),
                rec.pages.len()
            ));
        }
        // the bookkeeping the snapshot was taken with is unchanged (the checksum was made from the same fields at the take)
        if self.check_of(id, rec.take, meta, &rec.pages) != rec.check {
            return bad(format!("snapshot {id}: its bookkeeping no longer matches its checksum (corrupted)"));
        }
        if let Some(len) = len {
            if meta.len != len {
                return bad(format!(
                    "snapshot {id} holds a prefix of {} tokens, a restore of {len} was asked for: length mismatch",
                    meta.len
                ));
            }
        }
        Ok((id, meta))
    }
}

/// Everything a prompt leaves in a slot that a cached run must reproduce, read back to the host (the gates compare two of them bit for
/// bit): every Mamba layer's S and conv, the K and V rows `0..len` of the trunk's attention layers, the head's rows `0..len - 1`, the
/// carry and the head's book.
#[derive(Debug, PartialEq, Eq, Default)]
pub struct PrefixDigest {
    /// Per Mamba layer: S as f32 bits and the conv history.
    pub mamba: Vec<(Vec<u32>, Vec<u16>)>,
    /// Per trunk attention layer: K and V rows `0..len`, fp16 bits `[len][256]`.
    pub kv: Vec<(Vec<u16>, Vec<u16>)>,
    /// The head's layer: K and V rows `0..len - 1` (its entry for position `len - 1` is built by the next slice, after the cut).
    pub head: Option<(Vec<u16>, Vec<u16>)>,
    /// The slot's carry (store row 0, f32 bits), `None` without the head or when its bookkeeping is off (the row is not meaningful then).
    pub carry: Option<Vec<u32>>,
    /// The head's book `(ok, first, n)`.
    pub book: Option<(bool, usize, usize)>,
}

impl PrefixDigest {
    /// What differs from `other`, by name (empty when nothing does).
    pub fn describe_diff(&self, other: &PrefixDigest) -> String {
        let mut d = Vec::new();
        if let Some(l) = self.mamba.iter().zip(&other.mamba).position(|(a, b)| a.0 != b.0) {
            d.push(format!("Mamba S of layer #{l}"));
        }
        if let Some(l) = self.mamba.iter().zip(&other.mamba).position(|(a, b)| a.1 != b.1) {
            d.push(format!("Mamba conv of layer #{l}"));
        }
        if self.mamba.len() != other.mamba.len() {
            d.push("the number of Mamba layers".into());
        }
        for (l, (a, b)) in self.kv.iter().zip(&other.kv).enumerate() {
            if a != b {
                let row = KV_TOKEN_ELEMS;
                let first = (0..a.0.len() / row).find(|&p| {
                    a.0[p * row..(p + 1) * row] != b.0[p * row..(p + 1) * row]
                        || a.1[p * row..(p + 1) * row] != b.1[p * row..(p + 1) * row]
                });
                d.push(format!("K/V of attention layer #{l} (first row {first:?})"));
            }
        }
        if self.head != other.head {
            d.push("the head's K/V rows".into());
        }
        if self.carry != other.carry {
            d.push("the head's carry".into());
        }
        if self.book != other.book {
            d.push(format!("the head's book {:?} vs {:?}", self.book, other.book));
        }
        d.join("; ")
    }
}

impl NemotronModel {
    /// Allocates the snapshot arena: `slots` snapshot slots of [`SnapArena::slot_bytes`] bytes (the head's carry included when the model has the
    /// head), once, after the model and before any prompt. The plan counted exactly these bytes (`prefix::snapshot_bytes`).
    pub fn enable_snapshots(&mut self, slots: usize) -> Result<(), ModelError> {
        self.check_alive()?;
        if self.snaps.is_some() {
            return Err(ModelError::Config("the snapshot arena is already enabled".into()));
        }
        self.snaps = Some(SnapArena::new(self.core.mamba.len(), slots, self.core.mtp.is_some())?);
        Ok(())
    }

    /// Debug: makes the restore deliberately wrong (tests only; `None` is the production behaviour).
    #[doc(hidden)]
    pub fn set_snapshot_mutant(&mut self, mutant: Option<SnapMutant>) {
        if let Some(a) = self.snaps.as_mut() {
            a.mutant = mutant;
        }
    }

    /// Device bytes of the arena (0 before `enable_snapshots`).
    pub fn snapshot_arena_bytes(&self) -> u64 {
        self.snaps.as_ref().map_or(0, SnapArena::bytes)
    }

    /// Snapshot slots no snapshot holds right now.
    pub fn free_snapshots(&self) -> usize {
        self.snaps.as_ref().map_or(0, SnapArena::free_count)
    }

    /// Snapshot slots the arena has.
    pub fn snapshot_slots(&self) -> usize {
        self.snaps.as_ref().map_or(0, SnapArena::slots)
    }

    /// Pages of the KV pool no sequence and no snapshot holds.
    pub fn free_pages(&self) -> usize {
        self.kv_tab.free_blocks()
    }

    /// `slot`'s pages in position order, each with the references held on it (slots and snapshots): the gates read the sharing off it.
    #[doc(hidden)]
    pub fn slot_pages(&self, slot: usize) -> Vec<(u32, u32)> {
        self.kv_tab.blocks_of(slot).iter().map(|&b| (b, self.kv_tab.refcount(b))).collect()
    }

    /// Snapshots `slot`'s state after exactly `len` tokens (a prefill slice has just ended at `len`: the slot holds `len` committed tokens and no
    /// open window). `Ok(None)`: every arena slot is taken (the cut is skipped, not an error). On `Err` nothing stays allocated and no page
    /// reference is held. Everything is enqueued on the model's stream, ordered after the slice that left the state.
    pub fn snapshot_slot(&mut self, slot: usize, len: usize) -> Result<Option<PrefixHandle>, ModelError> {
        self.check_alive()?;
        let bad = |m: String| Err(ModelError::Request(m));
        if len == 0 {
            return bad("a snapshot of no tokens".into());
        }
        if self.snaps.is_none() {
            return bad("the snapshot arena is not enabled (enable_snapshots)".into());
        }
        let have = self.slots.len(slot)?;
        if have != len {
            return bad(format!("slot {slot} holds {have} committed tokens, a snapshot of {len} was asked for: length mismatch"));
        }
        if let Some(rows) = self.window.pending(slot) {
            return bad(format!(
                "slot {slot} has an uncommitted window of {rows} rows: a snapshot is taken between prefill slices"
            ));
        }
        let pages = len.div_ceil(BLOCK_TOKENS);
        if self.kv_tab.blocks_of(slot).len() < pages {
            return bad(format!(
                "slot {slot} holds {} pages, {pages} are needed for {len} tokens",
                self.kv_tab.blocks_of(slot).len()
            ));
        }
        if self.free_snapshots() == 0 {
            return Ok(None);
        }
        // a lagged slot's pool is the state as of its last fold: the snapshot takes the true one
        self.fold_lagged(&[slot])?;
        // the previous slice's launches are done and a refused commit surfaces here, not inside a snapshot
        self.flush()?;
        let mtp_ok = match self.core.mtp.as_ref() {
            Some(m) => {
                let b = m.book(slot);
                b.ok && b.first == len && b.n == 1
            }
            None => true,
        };
        let meta = SnapMeta { len, mtp_ok };
        let NemotronModel { core, snaps, kv_tab, .. } = self;
        let Core { mamba, mtp, stream, .. } = core;
        let a = snaps.as_mut().expect("checked above");
        let id = a.book.take(meta).expect("a free slot, checked above");
        let copied = (|| -> Result<(), ModelError> {
            for (o, m) in mamba.iter().enumerate() {
                let (so, co) = (a.s_off(id, o), a.conv_off(id, o));
                a.buf.copy_range_from(so, &m.s, StatePools::s_offset(slot), S_SLOT_BYTES, stream)?;
                a.buf.copy_range_from(co, &m.conv, StatePools::conv_offset(slot), CONV_SLOT_BYTES, stream)?;
            }
            if a.mtp && mtp_ok {
                if let Some(h) = mtp.as_ref() {
                    let co = a.carry_off(id);
                    a.buf.copy_range_from(
                        co,
                        h.store_buf(),
                        MtpState::carry_offset(slot),
                        prefix::CARRY_BYTES as usize,
                        stream,
                    )?;
                }
            }
            Ok(())
        })();
        if let Err(e) = copied {
            let _ = a.book.release(id);
            self.note(&e);
            return Err(e);
        }
        // the pages last: the donor's later writes are ordered after the copies above on the same stream
        let held = kv_tab.retain_prefix(slot, pages);
        a.takes += 1;
        let take = a.takes;
        let check = a.check_of(id, take, meta, &held);
        let tag = (check & 0xFFFF) as u16;
        a.rec[id as usize] = Some(SnapRec { tag, take, pages: held.clone(), check });
        Ok(Some(PrefixHandle { snap: (tag as u32) << 16 | id, pages: held }))
    }

    /// Puts `slot`, which `begin_slot` has just emptied, in the state `handle` holds for `len` tokens: pages `0..last` by reference, page
    /// `last = (len - 1) / 256` copied into one of the slot's own in every stripe (the head's included), S, conv and the head's carry copied
    /// back, the head's book `{ ok, first: len, n: 1 }`, the slot's length `len`. A handle that is not live, a stale or foreign one, a `len`
    /// that is not the snapshot's, a page list that is not its list and a slot that is not empty are each refused by name before anything is
    /// touched. On a failure after the first change the slot is the caller's to end (the backend gives it up).
    pub fn restore_snapshot(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), ModelError> {
        self.check_alive()?;
        let bad = |m: String| Err(ModelError::Request(m));
        let Some(a) = self.snaps.as_ref() else {
            return bad("the snapshot arena is not enabled (enable_snapshots)".into());
        };
        let (id, meta) = a.lookup(handle, Some(len))?;
        let rp = prefix::restore_pages(len);
        if rp.total != handle.pages.len() {
            return bad(format!("a prefix of {len} tokens is {} pages, the handle holds {}", rp.total, handle.pages.len()));
        }
        let kv_pages = self.core.opts.kv_pages;
        if let Some(&p) = handle.pages.iter().find(|&&p| p as usize >= kv_pages || self.kv_tab.refcount(p) == 0) {
            return bad(format!("page {p} of the handle is not a held page of this model's pool of {kv_pages}"));
        }
        let have = self.slots.len(slot)?;
        if have != 0 || !self.kv_tab.blocks_of(slot).is_empty() {
            return bad(format!(
                "slot {slot} holds {have} tokens and {} pages: a restore needs a begun, empty slot",
                self.kv_tab.blocks_of(slot).len()
            ));
        }
        if self.kv_tab.free_blocks() == 0 {
            return bad(format!(
                "KV pages of slot {slot}: no free page for the copy of the prefix's last page ({} pages in the pool)",
                kv_pages
            ));
        }
        let (shared, copied) = (rp.shared, rp.copied().expect("len > 0"));
        let mutant = self.snaps.as_ref().and_then(|a| a.mutant);
        let share_last = mutant == Some(SnapMutant::LastPageShared)
            || (mutant == Some(SnapMutant::FullLastPageShared) && len % BLOCK_TOKENS == 0);
        // from here the slot changes
        self.window.forget(slot);
        let dst = if share_last {
            self.kv_tab.adopt_shared(slot, &handle.pages);
            None
        } else {
            self.kv_tab.adopt_shared(slot, &handle.pages[..shared]);
            match self.kv_tab.alloc_block(slot) {
                Ok(b) => Some(b as usize),
                Err(e) => {
                    self.kv_tab.release(slot);
                    return bad(format!("KV pages of slot {slot}: {e:?}"));
                }
            }
        };
        self.table_dirty = true;
        let src = handle.pages[copied] as usize;
        let g = self.core.kv_geom;
        // the stripes a page copy covers: every attention layer, the head's (the last) included
        let stripes = if mutant == Some(SnapMutant::HeadStripeNotCopied) {
            g.layers - usize::from(self.core.mtp.is_some())
        } else {
            g.layers
        };
        let r = (|| -> Result<(), ModelError> {
            let NemotronModel { core, snaps, .. } = self;
            let Core { kv, mamba, mtp, stream, .. } = core;
            // the page: every stripe of K and V (the 8 trunk layers' and the head's), one copy each
            if let Some(dst) = dst {
                // a page of the pool's dtype: an int8 page carries its exponents, so one copy moves values and scales together
                let pb = g.page_bytes();
                for layer in 0..stripes {
                    let base = g.layer_offset(layer);
                    kv.k.copy_within(base + dst * pb, base + src * pb, pb, stream)?;
                    kv.v.copy_within(base + dst * pb, base + src * pb, pb, stream)?;
                }
            }
            let a = snaps.as_ref().expect("checked above");
            for (o, m) in mamba.iter_mut().enumerate() {
                if mutant != Some(SnapMutant::StateNotRestored) {
                    m.s.copy_range_from(StatePools::s_offset(slot), &a.buf, a.s_off(id, o), S_SLOT_BYTES, stream)?;
                }
                if mutant != Some(SnapMutant::ConvNotRestored) {
                    m.conv.copy_range_from(StatePools::conv_offset(slot), &a.buf, a.conv_off(id, o), CONV_SLOT_BYTES, stream)?;
                }
            }
            if let Some(h) = mtp.as_mut() {
                if a.mtp && meta.mtp_ok && mutant != Some(SnapMutant::CarryNotRestored) {
                    h.store_buf_mut().copy_range_from(
                        MtpState::carry_offset(slot),
                        &a.buf,
                        a.carry_off(id),
                        prefix::CARRY_BYTES as usize,
                        stream,
                    )?;
                }
                if mutant != Some(SnapMutant::BookNotRestored) {
                    h.restore_book(slot, len, meta.mtp_ok);
                }
            }
            Ok(())
        })();
        if let Err(e) = r {
            self.slots.poison(slot);
            self.note(&e);
            return Err(e);
        }
        self.slots.advance(slot, len)?;
        Ok(())
    }

    /// Gives a snapshot back: its arena slot and its page references. Exactly once per handle: a second release, a stale or foreign handle and
    /// a page list that is not the snapshot's are refused by name and change nothing (a bogus page list would otherwise release pages
    /// someone else holds).
    pub fn release_snapshot(&mut self, handle: PrefixHandle) -> Result<(), ModelError> {
        let Some(a) = self.snaps.as_mut() else {
            return Err(ModelError::Request("the snapshot arena is not enabled (enable_snapshots)".into()));
        };
        let (id, _) = a.lookup(&handle, None)?;
        a.book.release(id).map_err(ModelError::Request)?;
        a.rec[id as usize] = None;
        self.kv_tab.release_pages(&handle.pages);
        Ok(())
    }

    /// Everything the first `len` tokens of `slot` left behind (see [`PrefixDigest`]). Synchronises; for the gates.
    #[doc(hidden)]
    pub fn prefix_digest(&mut self, slot: usize, len: usize) -> Result<PrefixDigest, ModelError> {
        self.flush()?;
        let have = self.slots.len(slot)?;
        if have != len {
            return Err(ModelError::Request(format!("slot {slot} holds {have} tokens, a digest of {len} was asked for")));
        }
        let mut d = PrefixDigest::default();
        for ord in 0..self.core.mamba.len() {
            let (s, c) = self.debug_mamba_state(ord, slot)?;
            d.mamba.push((s.iter().map(|x| x.to_bits()).collect(), c));
        }
        let g: KvGeom = self.core.kv_geom;
        let blocks = self.kv_tab.blocks_of(slot).to_vec();
        // the rows `0..n` of one stripe of one pool, page by page (one read of a page per page and stripe, not the whole pool). An int8
        // page's rows are its value bytes and its exponent bytes, both compared raw (as u16 pairs: a row is 256 + 2 bytes)
        let (kbuf, vbuf) = (&self.core.kv.k, &self.core.kv.v);
        let pb = g.page_bytes();
        let q8 = g.dtype.is_8bit();
        let rows = |layer: usize, n: usize, pool: &DevBuf| -> Result<Vec<u16>, ModelError> {
            let mut out = Vec::with_capacity(n * KV_TOKEN_ELEMS);
            let mut page = vec![0u16; pb / 2];
            for (j, &blk) in blocks.iter().enumerate().take(n.div_ceil(BLOCK_TOKENS)) {
                pool.download_range(g.layer_offset(layer) + blk as usize * pb, &mut page)?;
                let take = (n - j * BLOCK_TOKENS).min(BLOCK_TOKENS);
                if q8 {
                    let ex = gb_cuda::nemotron_attn::kv8::EXP_OFF / 2;
                    out.extend_from_slice(&page[..take * KV_TOKEN_ELEMS / 2]);
                    out.extend_from_slice(&page[ex..ex + take]);
                } else {
                    out.extend_from_slice(&page[..take * KV_TOKEN_ELEMS]);
                }
            }
            Ok(out)
        };
        let trunk = g.layers - usize::from(self.core.mtp.is_some());
        for layer in 0..trunk {
            d.kv.push((rows(layer, len, kbuf)?, rows(layer, len, vbuf)?));
        }
        if self.core.mtp.is_some() {
            let layer = g.layers - 1;
            d.head = Some((rows(layer, len - 1, kbuf)?, rows(layer, len - 1, vbuf)?));
            let b = self.mtp_book(slot).cloned();
            d.book = b.as_ref().map(|b| (b.ok, b.first, b.n));
            if b.as_ref().is_some_and(|b| b.ok) {
                d.carry = Some(self.mtp_store_rows(slot, 1)?.iter().map(|x| x.to_bits()).collect());
            }
        }
        Ok(d)
    }
}
