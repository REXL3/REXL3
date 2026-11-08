//! MiniCPM-V-4.7's engine-owned prefix snapshots (E6b). The engine owns indexing, pinning, cut planning and eviction
//! (`gb_sched::prefix`, `gb_engine::prefix`); this module owns the device snapshots and the page references. A copy-adapt of
//! `qwen_flash_next::prefix` (design decision 5.2a: no shared layer), cut to this backbone: there is no PLE ring, no sparse-attention
//! indexer pool and no MTP head, so a snapshot is the recurrent state of the 30 gated delta net layers and nothing else.
//!
//! **What a prompt leaves behind** (all of it comes back on a hit, nothing else may differ):
//!
//! * the 10 attention layers' K and V rows of positions `0..len`, in pages of 256 tokens that the entry holds by reference
//!   (`KvTables::retain_prefix`); the page that is only partly filled (`len % 256 != 0`) is copied into a page the restored slot owns
//!   (copy-on-write at attach), the full pages are shared and never written again by anyone;
//! * per gated delta net layer the delta-rule state `S` (fp32 `[32][128][128]`) and the convolution history (bf16 `[3][8192]`),
//!   2,146,304 B a layer, **64,389,120 B a snapshot** (61.4 MiB), the same bytes a sequence slot holds;
//! * the slot's rope delta (`MiniCpmModel::rope_delta`: 0 for text and before the first picture, the canvas offset after one; E5b),
//!   kept in the snapshot's host-side [`Meta`];
//! * nothing else: logits are recomputed by the suffix, windows are never cached (decode rounds and prefill round differently, and a
//!   state taken after generated tokens is not the state an uncached request has).
//!
//! **Bits.** A snapshot is interchangeable only inside the arithmetic that produced it: the prefill arm is the prompt TOTAL's
//! (`PrefillPath`), so only prompts of more than 128 tokens (the pinned GEMM for every chunk) are cached, and the prefill MoE is the
//! model's (E6a: the grouped `fmoe::prefill`, or E3's CoopMK under `REXL3_MINICPM_PREFILL_MOE=coopmk`, whose bits differ). Both are in
//! the cache path id ([`path`]: 1 for the grouped arm, 2 for CoopMK), which is part of every index key and of the snapshot's
//! [`Meta`], and the arm is in the namespace; a restore refuses a snapshot of the other arm and the backend refuses to continue a
//! restored slot into a prompt that would take the other arithmetic. Chunk invariance
//! (`a_tiled_prefill_leaves_the_same_bits_in_any_chunk_grid`, `prefill_moe_leaves_the_same_bits_in_any_chunk_grid`) is what lets a cut
//! sit anywhere and the suffix use any grid.
//!
//! **The draft (E4b).** The draft's context K and V ride in the SAME page ids (`dspark.rs`, "the draft's page `i` is the page with the
//! target's id `i`"), so a shared page carries the draft's rows for free and the partial page's copy-on-write ([`copy_page`]) copies
//! the draft's page with the target's. What a snapshot adds is one fact: whether the donor's draft context was the target's committed
//! length at the cut (`spec.rs`, `SlotBook`), recorded with the identity of the draft that made it ([`SnapshotArena::draft_ctx`]). A
//! restore resumes the draft's context only when the model's draft is that very draft (pack fingerprint, tap layers and page layout:
//! [`MiniCpmModel::draft_identity`]); otherwise the slot has the target's state and no draft context, and never drafts (budget 0, plain
//! decoding: the draft's rows for a prefix cannot be rebuilt without running the target over it, which is what the cache saves). The
//! namespace carries the same identity, so a snapshot made with one draft is not even found by an engine with another or with none.
//! The draft's positions are token indices, so nothing about a picture (the rope delta, the canvas) is in the draft's context: the
//! restored slot has the committed length and the same page ids.
//!
//! **Images and video (E5b).** The namespace carries the image and video preprocessing id ([`caps_with`]; [`NO_IMAGES`] for an
//! engine without a tower), the engine's key stream carries each picture's content hash, a cut never lands inside a picture or a
//! visual group (`backend.rs`: `cached_slice_mm`, the canvas rule `Geometry::cut_outside_frames` on top of the engine's own
//! `outside_images`), and the snapshot's [`Meta`] carries the offset of the text after the cut (the slot's rope delta, set before the
//! capture and restored with it).
//!
//! Backend integration (`backend.rs`): `prefix_caps` is [`caps`], `prefix_path(total)` is [`path`] when the arena exists,
//! `prefix_restore` calls [`restore`] (on error the slot ends), `prefix_release` calls [`release`] exactly once per handle,
//! `prefill_slice_cuts` calls [`slice_with_cuts`] and maps its [`SliceResult`] to the engine's `CutStep`; the same routine with no
//! cuts is the uncached prefill path. A cut inside the natural chunk ENDS that slice at the cut: one more pinned W_q pass instead of a
//! recurrent-state callback inside the chunk (as Flash-Next).

use super::config::MiniCpmConfig;
use super::memory;
use super::model::MiniCpmModel;
use super::state::GDN;
use crate::error::ModelError;
use crate::forward::{Candidates, PrefillPath};
use crate::kvtables::BLOCK_TOKENS;
use crate::rows::RowBatch;
use gb_cuda::DevBuf;
use gb_engine::prefix::{Captured, PrefixCaps, PrefixHandle};
use gb_engine::RowRequest;
use gb_sched::prefix::CacheBudget;

/// The cacheable prefill arithmetic of the default arm: the pinned GEMM for every chunk of a prompt of more than 128 tokens and the
/// grouped prefill MoE (`fmoe::prefill`).
pub const PATH_ID: u32 = 1;
/// The same pinned-GEMM prefill with E3's CoopMK prefill MoE (`REXL3_MINICPM_PREFILL_MOE=coopmk`): other bits, another path id, so a
/// snapshot of one arm is never found, and never restored, by the other.
pub const PATH_ID_COOPMK: u32 = 2;

/// The cache path id of the model's prefill arm (`grouped`: [`MiniCpmModel::prefill_moe_grouped`]).
pub fn arm_path(grouped: bool) -> u32 {
    if grouped {
        PATH_ID
    } else {
        PATH_ID_COOPMK
    }
}

/// The name of the prefill MoE arm a cache path id stands for (the namespace's, and the refusals').
pub fn arm_name(path: u32) -> &'static str {
    if path == PATH_ID_COOPMK {
        "coopmk"
    } else {
        "grouped"
    }
}

/// The cache path id of a prompt of `total` tokens under the prefill MoE arm `grouped`, `None` when its state may not be cached or
/// restored (the decode-GEMV arm).
pub fn path(total: usize, grouped: bool) -> Option<u32> {
    (PrefillPath::for_prompt(total) == PrefillPath::Tiled).then_some(arm_path(grouped))
}

/// The namespace slot of a text-only engine (E5b: the image preprocessing id, `MiniCpmPreprocessing::id()`).
pub const NO_IMAGES: &str = "none";
/// The namespace slot of an engine with no draft; with one it is the draft's identity (`SpecRig::identity`: the draft pack's
/// fingerprint, its tap layers and the page layout it adds).
pub const NO_DRAFT: &str = "none";

/// The hash a snapshot's record keeps of the draft identity that made its draft context (FNV-1a 64, as the namespace's).
pub fn draft_key(identity: &str) -> u64 {
    identity.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

/// The pack fingerprint's shape: `sha256:` and 64 lowercase hex digits (`fingerprint::fingerprint`). A cache keyed by anything else
/// would let two packs share entries.
pub fn check_fingerprint(fingerprint: &str) -> Result<(), ModelError> {
    let hex = fingerprint.strip_prefix("sha256:").unwrap_or("");
    if hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        Ok(())
    } else {
        Err(ModelError::Config(format!(
            "the prefix cache is keyed by the pack's fingerprint (`sha256:` and 64 hex digits), got `{fingerprint}`"
        )))
    }
}

/// The namespace folds in everything that makes snapshots non-interchangeable: the pack (its fingerprint), the family, the arithmetic
/// of the prompt (the path ids; also part of every index key), the arithmetic choices (kernels, the pinned GEMM algorithm, the
/// prefill MoE arm `moe`: `"grouped"` or `"coopmk"`), the page size and the KV and state formats, the images' preprocessing and the
/// draft. No snapshot persists across boots, and an entry of one namespace is invisible to another.
pub fn namespace_of(fingerprint: &str, pinned_algorithm: &str, moe: &str, images: &str, draft: &str) -> u64 {
    let key = format!(
        "rexl3-minicpmv4_7-prefix-v2|family=minicpmv4_7|pack={fingerprint}|arith=pinned-tiled-per-row-router-gdn-prefill-hv32-attn-prefill-q16kv2|moe={moe}|pin={pinned_algorithm}|page={BLOCK_TOKENS}|kv=bf16x2|gdn=f32+bf16|images={images}|draft={draft}"
    );
    key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

/// Full pages are shared, and the last partial page gets a private full-page copy: `(shared pages, has a partial page)`.
pub fn page_plan(len: usize) -> (usize, bool) {
    (len / BLOCK_TOKENS, len % BLOCK_TOKENS != 0)
}

/// End of the slice that starts at `done`: the next absolute chunk boundary, never past a planned cut (a cut on the boundary does not
/// shorten the chunk).
pub fn slice_end(done: usize, total: usize, chunk: usize, cuts: &[usize]) -> usize {
    let chunk = chunk.max(1);
    let end = done.saturating_add(chunk - done % chunk).min(total);
    cuts.iter().copied().find(|&c| c > done && c < end).unwrap_or(end)
}

/// Layout of one snapshot slot in the arena (bytes): every GDN layer's `S`, then every layer's convolution history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub gdn_layers: usize,
    /// `S` of one layer of one sequence (fp32 `[32][128][128]`).
    pub s: usize,
    /// The convolution history of one layer of one sequence (bf16 `[3][8192]`).
    pub conv: usize,
}

impl Layout {
    pub fn from_config(cfg: &MiniCpmConfig) -> Self {
        Self {
            gdn_layers: cfg.linear_layers(),
            s: cfg.lin_value_heads * cfg.lin_value_dim * cfg.lin_key_dim * 4,
            conv: cfg.lin_conv_kernel.saturating_sub(1) * cfg.conv_channels() * 2,
        }
    }
    /// The layout the kernels' pools use (`GdnShape::Hv32`): what the arena's copies are sized by.
    pub fn of_pools() -> Self {
        Self { gdn_layers: 0, s: GDN.s_slot_elems() * 4, conv: GDN.conv_slot_elems() * 2 }
    }
    fn s_at(self, i: usize) -> usize {
        i * self.s
    }
    fn conv_at(self, i: usize) -> usize {
        self.gdn_layers * self.s + i * self.conv
    }
    pub fn bytes(self) -> usize {
        self.gdn_layers * (self.s + self.conv)
    }
}

/// Bytes of one snapshot slot (the memory model's `prefix_snapshot` item).
pub fn snapshot_bytes(cfg: &MiniCpmConfig) -> u64 {
    Layout::from_config(cfg).bytes() as u64
}

/// Bytes of one KV page across the 10 attention layers (the memory model's `kv_target` item).
pub fn page_bytes(cfg: &MiniCpmConfig) -> u64 {
    memory::PAGE_TOKENS * memory::target_kv_bytes_per_token(cfg)
}

/// The cache's physical size for `bytes` of memory: whole entry units of one snapshot slot and the assumed pages
/// (`CacheBudget::fit`; the planner's `plan_memory_cached` does the same when given `--prefix-cache-mib` or `auto`).
pub fn budget_for_bytes(cfg: &MiniCpmConfig, bytes: u64) -> CacheBudget {
    CacheBudget::fit(bytes, snapshot_bytes(cfg), page_bytes(cfg))
}

/// What the host keeps of a snapshot: the cut it was taken at, the slot's rope delta (0 for text) and the path of the prompt that
/// made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Meta {
    pub len: usize,
    pub rope_delta: i32,
    pub path: u32,
}

/// Pure slot bookkeeping: a handle remains live until the engine releases it exactly once.
struct SlotBook {
    free: Vec<u32>,
    meta: Vec<Option<Meta>>,
}

impl SlotBook {
    fn new(n: usize) -> Self {
        Self { free: (0..n as u32).rev().collect(), meta: vec![None; n] }
    }
    fn alloc(&mut self) -> Option<u32> {
        self.free.pop()
    }
    fn free(&mut self, id: u32) -> Result<(), ModelError> {
        let m = self.meta.get_mut(id as usize).ok_or_else(|| ModelError::Request("snapshot id out of range".into()))?;
        if m.take().is_none() {
            return Err(ModelError::Request("snapshot already released or incomplete".into()));
        }
        self.free.push(id);
        Ok(())
    }
    fn abandon(&mut self, id: u32) {
        self.meta[id as usize] = None;
        self.free.push(id);
    }
}

/// The preallocated snapshot slots (one device buffer the boot allocated, `ModelState::snapshots`) and their host ownership table.
pub struct SnapshotArena {
    /// `None` for a bookkeeping-only arena (the host tests of the backend): capture and restore refuse.
    data: Option<DevBuf>,
    layout: Layout,
    slots: SlotBook,
    budget: CacheBudget,
    /// Bytes the draft adds to a page of the pool (`memory::draft_page_bytes`; 0 without a draft): the budget's page is the target's
    /// and the draft's together.
    draft_page_bytes: u64,
    /// Per snapshot id: the identity hash ([`draft_key`]) of the draft whose context was the target's committed length at the cut,
    /// `None` when there was no draft or its context was not valid.
    draft_ctx: Vec<Option<u64>>,
}

impl SnapshotArena {
    /// The arena over `data`, the buffer the boot allocated for `budget.snapshot_slots` snapshots of the model's layout. The KV
    /// allocation must already include `budget.page_blocks` above the admission share (`BootPlan::kv_pages`).
    pub fn new(cfg: &MiniCpmConfig, data: DevBuf, budget: CacheBudget) -> Result<Self, ModelError> {
        Self::new_with_draft(cfg, data, budget, 0)
    }

    /// [`SnapshotArena::new`] for an engine with a draft: the budget's page is the target's page and the draft's
    /// (`draft_page_bytes`, `memory::draft_page_bytes`).
    pub fn new_with_draft(
        cfg: &MiniCpmConfig,
        data: DevBuf,
        budget: CacheBudget,
        draft_page_bytes: u64,
    ) -> Result<Self, ModelError> {
        let layout = Layout::from_config(cfg);
        Self::check(&layout, cfg, &budget, draft_page_bytes)?;
        let want = budget
            .snapshot_slots
            .checked_mul(layout.bytes())
            .ok_or_else(|| ModelError::Config("prefix arena size overflow".into()))?;
        if data.bytes() != want {
            return Err(ModelError::Config(format!(
                "the prefix arena holds {} B, the plan prices {} snapshots of {} B = {want} B",
                data.bytes(),
                budget.snapshot_slots,
                layout.bytes()
            )));
        }
        Ok(Self {
            data: Some(data),
            layout,
            slots: SlotBook::new(budget.snapshot_slots),
            budget,
            draft_page_bytes,
            draft_ctx: vec![None; budget.snapshot_slots],
        })
    }

    /// An arena that books slots and holds no device memory: the host contract of the backend runs against it with a fake runtime.
    pub fn bookkeeping_only(cfg: &MiniCpmConfig, budget: CacheBudget) -> Result<Self, ModelError> {
        let layout = Layout::from_config(cfg);
        Self::check(&layout, cfg, &budget, 0)?;
        Ok(Self {
            data: None,
            layout,
            slots: SlotBook::new(budget.snapshot_slots),
            budget,
            draft_page_bytes: 0,
            draft_ctx: vec![None; budget.snapshot_slots],
        })
    }

    fn check(layout: &Layout, cfg: &MiniCpmConfig, budget: &CacheBudget, draft_page_bytes: u64) -> Result<(), ModelError> {
        let gdn = cfg.gdn_shape();
        let pools_s = gdn.s_slot_elems() * 4;
        let pools_conv = gdn.conv_slot_elems() * 2;
        if layout.s != pools_s || layout.conv != pools_conv {
            return Err(ModelError::Config(format!(
                "the snapshot layout ({} + {} B a layer) is not the delta-net pools' ({} + {} B)",
                layout.s, layout.conv, pools_s, pools_conv
            )));
        }
        if !budget.enabled()
            || budget.snapshot_bytes != layout.bytes() as u64
            || budget.page_bytes != page_bytes(cfg) + draft_page_bytes
        {
            return Err(ModelError::Config("the prefix budget does not match the model's snapshot and page sizes".into()));
        }
        Ok(())
    }

    pub fn free_count(&self) -> usize {
        self.slots.free.len()
    }
    /// Snapshot slots a handle holds right now.
    pub fn held_count(&self) -> usize {
        self.slots.meta.iter().filter(|m| m.is_some()).count()
    }
    pub fn allocated_bytes(&self) -> u64 {
        self.data.as_ref().map_or(0, |d| d.bytes() as u64)
    }
    pub fn budget(&self) -> CacheBudget {
        self.budget
    }
    pub fn layout(&self) -> Layout {
        self.layout
    }
    /// The host record of a live snapshot.
    pub fn meta(&self, id: u32) -> Option<Meta> {
        self.slots.meta.get(id as usize).copied().flatten()
    }

    /// The identity hash of the draft whose context was valid at snapshot `id`'s cut (`None`: no draft, or no valid context).
    pub fn draft_ctx(&self, id: u32) -> Option<u64> {
        self.draft_ctx.get(id as usize).copied().flatten()
    }

    /// Bytes the draft adds to a page (0 without a draft).
    pub fn draft_page_bytes(&self) -> u64 {
        self.draft_page_bytes
    }

    /// Test seam: record that snapshot `id` holds a valid draft context of the draft with identity hash `key` (what a capture does).
    pub fn set_draft_ctx(&mut self, id: u32, key: Option<u64>) {
        if let Some(c) = self.draft_ctx.get_mut(id as usize) {
            *c = key;
        }
    }

    fn buf(&self) -> Result<&DevBuf, ModelError> {
        self.data.as_ref().ok_or_else(|| ModelError::Config("prefix snapshots need the device arena".into()))
    }

    /// Snapshot slot `id`'s bytes as the device holds them: `S` of every layer, then every layer's convolution history (diagnostics
    /// and tests). A synchronous copy: the model's stream must be synchronised first.
    pub fn read_slot(&self, id: u32) -> Result<Vec<u8>, ModelError> {
        let data = self.buf()?;
        let n = self.layout.bytes();
        if id as usize >= self.slots.meta.len() {
            return Err(ModelError::Request(format!("snapshot id {id} of {}", self.slots.meta.len())));
        }
        let mut v = vec![0u8; n];
        data.download_range(id as usize * n, &mut v)?;
        Ok(v)
    }

    /// Book a snapshot slot without device work (the host tests; `capture` is the real path).
    pub fn book(&mut self, len: usize, rope_delta: i32) -> Option<u32> {
        self.book_with_path(len, rope_delta, PATH_ID)
    }

    /// [`SnapshotArena::book`] under the cache path `path` ([`PATH_ID`] or [`PATH_ID_COOPMK`]).
    pub fn book_with_path(&mut self, len: usize, rope_delta: i32, path: u32) -> Option<u32> {
        let id = self.slots.alloc()?;
        self.slots.meta[id as usize] = Some(Meta { len, rope_delta, path });
        Some(id)
    }

    /// Test seam: record snapshot `id` as made under the cache path `path` (what a snapshot of the other prefill MoE arm looks like to
    /// a restore). Returns the path it had.
    pub fn retag(&mut self, id: u32, path: u32) -> Option<u32> {
        let m = self.slots.meta.get_mut(id as usize)?.as_mut()?;
        Some(std::mem::replace(&mut m.path, path))
    }

    /// Give back a booked slot (the host tests; `release` is the real path).
    pub fn unbook(&mut self, id: u32) -> Result<(), ModelError> {
        self.slots.free(id)?;
        self.set_draft_ctx(id, None);
        Ok(())
    }
}

/// What the engine needs of the arena: the namespace of its keys (under the model's prefill MoE arm `grouped`) and the physical budget.
pub fn caps(arena: &SnapshotArena, fingerprint: &str, pinned_algorithm: &str, grouped: bool) -> PrefixCaps {
    caps_with_draft(arena, fingerprint, pinned_algorithm, grouped, NO_IMAGES, NO_DRAFT)
}

/// [`caps`] for an engine that serves images (E5b): `images` is the identity of the image and video preprocessing
/// (`MiniCpmPreprocessing::id()`: backend, scale, patch, slices, mode, normalisation), which decides what a pixel hash stands for.
/// Two engines that preprocess differently share no entry, and no entry is shared with a text-only engine.
pub fn caps_with(arena: &SnapshotArena, fingerprint: &str, pinned_algorithm: &str, grouped: bool, images: &str) -> PrefixCaps {
    caps_with_draft(arena, fingerprint, pinned_algorithm, grouped, images, NO_DRAFT)
}

/// [`caps_with`] for an engine whose model runs the draft of identity `draft` (`MiniCpmModel::draft_identity`, [`NO_DRAFT`] without
/// one): a snapshot made with one draft is in another namespace than an engine with another draft or none.
pub fn caps_with_draft(
    arena: &SnapshotArena,
    fingerprint: &str,
    pinned_algorithm: &str,
    grouped: bool,
    images: &str,
    draft: &str,
) -> PrefixCaps {
    let moe = arm_name(arm_path(grouped));
    PrefixCaps { namespace: namespace_of(fingerprint, pinned_algorithm, moe, images, draft), budget: arena.budget }
}

/// Copy the slot's recurrent state out into snapshot slot `id`.
fn copy_state_out(m: &MiniCpmModel, data: &mut DevBuf, lay: Layout, id: usize, slot: usize) -> Result<(), ModelError> {
    let base = id * lay.bytes();
    for i in 0..lay.gdn_layers {
        data.copy_range_from(base + lay.s_at(i), &m.pools.s[i], slot * lay.s, lay.s, &m.stream)?;
        data.copy_range_from(base + lay.conv_at(i), &m.pools.conv[i], slot * lay.conv, lay.conv, &m.stream)?;
    }
    Ok(())
}

/// Copy snapshot slot `id` into the slot's recurrent state.
fn copy_state_in(m: &mut MiniCpmModel, data: &DevBuf, lay: Layout, id: usize, slot: usize) -> Result<(), ModelError> {
    let base = id * lay.bytes();
    for i in 0..lay.gdn_layers {
        m.pools.s[i].copy_range_from(slot * lay.s, data, base + lay.s_at(i), lay.s, &m.stream)?;
        m.pools.conv[i].copy_range_from(slot * lay.conv, data, base + lay.conv_at(i), lay.conv, &m.stream)?;
    }
    Ok(())
}

/// Capture after `len` prefill rows of `slot`: the recurrent state is copied out (stream-ordered before any later write of the
/// donor), the KV pages of the prefix are retained by reference, and the host record is made. `Ok(None)` when no snapshot slot is
/// free (the engine skips the cut). Only a prefill's state is captured: never call this after a decode window.
pub fn capture(m: &mut MiniCpmModel, a: &mut SnapshotArena, slot: usize, len: usize) -> Result<Option<Captured>, ModelError> {
    if len == 0 || slot >= m.opts.slots || len > m.opts.max_context || m.kv_tab.blocks_of(slot).len() < len.div_ceil(BLOCK_TOKENS)
    {
        return Err(ModelError::Request("prefix capture is not at a valid committed token cut".into()));
    }
    a.buf()?;
    let Some(id) = a.slots.alloc() else {
        return Ok(None);
    };
    let lay = a.layout;
    let copied = m.flush().and_then(|()| copy_state_out(m, a.data.as_mut().expect("checked above"), lay, id as usize, slot));
    if let Err(e) = copied {
        a.slots.abandon(id);
        return Err(e);
    }
    // all device copies are enqueued before the references are taken; a later donor write is ordered after them on the same stream
    let pages = m.kv_tab.retain_prefix(slot, len.div_ceil(BLOCK_TOKENS));
    a.slots.meta[id as usize] = Some(Meta { len, rope_delta: m.rope_delta[slot], path: arm_path(m.prefill_moe_grouped()) });
    // the draft's context rows of the prefix are in the retained pages (same page ids); whether they are valid is the book's word at
    // this very cut, and whose they are is the draft's identity
    a.draft_ctx[id as usize] = if m.draft_context_is(slot, len) { m.draft_identity().map(draft_key) } else { None };
    Ok(Some(Captured { len, handle: PrefixHandle { snap: id, pages } }))
}

/// Restore: begin `slot` over again from `handle`: the full pages by reference, the partial page copied to a page the slot owns, the
/// recurrent state copied in, the rope delta set. The next prefill slice is called with `done == len`. On error the caller ends the
/// slot (this function already has).
pub fn restore(
    m: &mut MiniCpmModel,
    a: &SnapshotArena,
    slot: usize,
    handle: &PrefixHandle,
    len: usize,
) -> Result<(), ModelError> {
    let meta = a
        .meta(handle.snap)
        .filter(|x| x.len == len && len > 0 && handle.pages.len() == len.div_ceil(BLOCK_TOKENS))
        .ok_or_else(|| ModelError::Request("prefix handle or token length is invalid".into()))?;
    if slot >= m.opts.slots {
        return Err(ModelError::Request("prefix slot out of range".into()));
    }
    // the prefill MoE arm is part of the arithmetic: a snapshot of the other arm holds other bits and is refused, not adapted
    let arm = arm_path(m.prefill_moe_grouped());
    if meta.path != arm {
        return Err(ModelError::Request(format!(
            "a snapshot made under the {} prefill MoE cannot be restored by a model that runs the {} one (their bits differ)",
            arm_name(meta.path),
            arm_name(arm)
        )));
    }
    let data = a.buf()?;
    // the draft's context comes back only from a snapshot that holds one of THIS model's draft (same pack, taps and layout)
    let adopt = a.draft_ctx(handle.snap).is_some() && a.draft_ctx(handle.snap) == m.draft_identity().map(draft_key);
    m.begin_slot(slot)?;
    let (full, partial) = page_plan(len);
    m.kv_tab.adopt_shared(slot, &handle.pages[..full]);
    let copy = (|| -> Result<(), ModelError> {
        if partial {
            let src = handle.pages[full] as usize;
            let dst = m.kv_tab.alloc_block(slot).map_err(|e| ModelError::Request(format!("prefix copy-on-write page: {e:?}")))?
                as usize;
            copy_page(m, src, dst)?;
        }
        m.upload_table()?;
        copy_state_in(m, data, a.layout, handle.snap as usize, slot)?;
        m.rope_delta[slot] = meta.rope_delta;
        if adopt {
            // the shared pages and the copied one hold the draft's rows `0..len`: the slot's draft context is `len` rows
            m.adopt_draft_context(slot, len)?;
        }
        // the copies above (the copy-on-write page of K/V and the draft's page, and the recurrent state) are enqueued on the target's
        // NON-blocking stream. A legacy-stream readback (`download_range`, a plain cudaMemcpy) and the draft's blocking stream do not
        // order with it, so the restore completes only once the copies have landed.
        m.stream.sync()?;
        Ok(())
    })();
    if copy.is_err() {
        m.end_slot(slot);
    }
    copy
}

/// The copy-on-write of the partial page: K and V of every attention layer, whole pages (a page is 256 KiB a layer), and the draft's
/// page of the same id (its context pages are indexed by the target's page ids: 2.5 MiB, E4b).
fn copy_page(m: &mut MiniCpmModel, src: usize, dst: usize) -> Result<(), ModelError> {
    for kv in m.kv.iter_mut() {
        let bytes = kv.block_bytes();
        kv.k.copy_within(dst * bytes, src * bytes, bytes, &m.stream)?;
        kv.v.copy_within(dst * bytes, src * bytes, bytes, &m.stream)?;
    }
    m.copy_draft_page(src, dst)?;
    Ok(())
}

/// Every handle releases one arena slot and its page references exactly once; index eviction is engine-owned. A second release of
/// the same handle is an error and releases nothing.
pub fn release(m: &mut MiniCpmModel, a: &mut SnapshotArena, handle: PrefixHandle) -> Result<(), ModelError> {
    a.slots.free(handle.snap)?;
    a.set_draft_ctx(handle.snap, None);
    m.kv_tab.release_pages(&handle.pages);
    Ok(())
}

pub struct SliceResult {
    pub end: usize,
    pub row: Option<Candidates>,
    pub captured: Vec<Captured>,
}

/// One backend slice's arguments; `prompt` is the entire prompt, not only the chunk.
pub struct SliceRequest<'a> {
    pub slot: usize,
    pub prompt: &'a [u32],
    pub done: usize,
    pub candidates: usize,
    pub cuts: &'a [usize],
    pub chunk: usize,
}

/// A slice with planned cuts; the caller translates [`SliceResult`] to the engine's `CutStep`. A cut at the prompt end is not useful:
/// a strict prefix leaves a row to recompute. The same routine with empty cuts is the uncached prefill path (it begins the slot when
/// `done == 0`).
pub fn slice_with_cuts(m: &mut MiniCpmModel, a: &mut SnapshotArena, req: SliceRequest<'_>) -> Result<SliceResult, ModelError> {
    let SliceRequest { slot, prompt, done, candidates, cuts, chunk } = req;
    if prompt.is_empty() || done >= prompt.len() || slot >= m.opts.slots || cuts.windows(2).any(|w| w[0] >= w[1]) {
        return Err(ModelError::Request("invalid MiniCPM-V-4.7 prefill slice or unordered cuts".into()));
    }
    if done == 0 {
        m.begin_slot(slot)?;
    }
    m.flush()?;
    let prefill_path = PrefillPath::for_prompt(prompt.len());
    let cuts = if path(prompt.len(), m.prefill_moe_grouped()).is_some() { cuts } else { &[] };
    let end = slice_end(done, prompt.len(), chunk.min(m.opts.prefill_rows), cuts);
    let b = RowBatch::from_requests(&[RowRequest { slot, first_position: done as u64, tokens: prompt[done..end].to_vec() }]);
    let row = m.prefill_chunk(&b, prefill_path, (end == prompt.len()).then_some(candidates))?;
    let captured =
        if end < prompt.len() && cuts.contains(&end) { capture(m, a, slot, end)?.into_iter().collect() } else { Vec::new() };
    Ok(SliceResult { end, row, captured })
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::{draft, real};
    use super::super::memory::{allocations, memory_profile, sum_of, AllocKind};
    use super::*;
    use crate::memory::MemoryOptions;

    const FP_A: &str = "sha256:9bc579ceb1ae00bd016922d354e340bb395e9da52438cc226d1c898d5e63d03f";
    const FP_B: &str = "sha256:9bc579ceb1ae00bd016922d354e340bb395e9da52438cc226d1c898d5e63d03e";

    #[test]
    fn arbitrary_cuts_plan_a_private_partial_page_and_a_cut_ends_its_slice() {
        assert_eq!(page_plan(255), (0, true));
        assert_eq!(page_plan(256), (1, false));
        assert_eq!(page_plan(257), (1, true));
        assert_eq!(page_plan(513), (2, true));
        assert_eq!(slice_end(0, 1000, 512, &[300, 900]), 300);
        assert_eq!(slice_end(300, 1000, 512, &[300, 900]), 512);
        assert_eq!(slice_end(512, 1000, 512, &[300, 900]), 900);
        assert_eq!(slice_end(900, 1000, 512, &[300, 900]), 1000);
        // a cut on the chunk boundary does not shorten the chunk
        assert_eq!(slice_end(0, 2000, 512, &[512]), 512);
        assert_eq!(slice_end(0, 2000, 512, &[1024]), 512);
    }

    #[test]
    fn the_snapshot_is_the_sequence_slots_state_and_the_page_is_the_pools() {
        let c = real();
        let lay = Layout::from_config(&c);
        assert_eq!(lay, Layout { gdn_layers: 30, s: 2_097_152, conv: 49_152 });
        assert_eq!(lay.bytes(), 64_389_120);
        assert_eq!(snapshot_bytes(&c), 64_389_120);
        // the layout equals what the kernels' pools use, and what the memory model prices (a snapshot and a slot hold the same state)
        assert_eq!((lay.s, lay.conv), (Layout::of_pools().s, Layout::of_pools().conv));
        let items = allocations(&c, None);
        assert_eq!(sum_of(&items, AllocKind::PerSnapshot), snapshot_bytes(&c));
        assert_eq!(sum_of(&items, AllocKind::PerSnapshot), sum_of(&items, AllocKind::PerSlot));
        assert_eq!(memory::snapshot_bytes(&c), snapshot_bytes(&c));
        let p = memory_profile(&c, None, &MemoryOptions::serving_defaults()).unwrap();
        assert_eq!(snapshot_bytes(&c), p.state_bytes_per_slot());
        assert_eq!(page_bytes(&c), p.kv_page_bytes());
        assert_eq!(page_bytes(&c), 256 * 20_480);
        // two entry units: a slot and the assumed 8 pages each (one MiB short is no cache at all)
        let unit = snapshot_bytes(&c) + 8 * page_bytes(&c);
        assert_eq!(budget_for_bytes(&c, 2 * unit - 1).snapshot_slots, 0);
        let b = budget_for_bytes(&c, 2 * unit);
        assert_eq!((b.snapshot_slots, b.page_blocks), (2, 16));
        assert_eq!(b.bytes(), 2 * unit);
        // a 261,120-token prompt is 1,020 pages: only a budget of 128 units (about 12.7 GiB) can hold its entry
        assert!(budget_for_bytes(&c, 8 << 30).page_blocks < 1020);
        assert!(budget_for_bytes(&c, 13 << 30).page_blocks >= 1020);
    }

    #[test]
    fn only_the_pinned_gemm_arm_is_cached_and_each_prefill_moe_arm_has_its_own_path() {
        for grouped in [true, false] {
            assert_eq!((path(1, grouped), path(128, grouped)), (None, None));
        }
        assert_eq!((path(129, true), path(262_144, true)), (Some(PATH_ID), Some(PATH_ID)));
        assert_eq!((path(129, false), path(262_144, false)), (Some(PATH_ID_COOPMK), Some(PATH_ID_COOPMK)));
        assert_ne!(PATH_ID, PATH_ID_COOPMK);
        assert_eq!((arm_name(PATH_ID), arm_name(PATH_ID_COOPMK)), ("grouped", "coopmk"));
    }

    #[test]
    fn namespaces_of_unlike_state_are_isolated() {
        let n = namespace_of(FP_A, "pin-1", "grouped", NO_IMAGES, NO_DRAFT);
        assert_eq!(n, namespace_of(FP_A, "pin-1", "grouped", NO_IMAGES, NO_DRAFT), "deterministic");
        for (what, other) in [
            ("pack fingerprint", namespace_of(FP_B, "pin-1", "grouped", NO_IMAGES, NO_DRAFT)),
            ("pinned algorithm", namespace_of(FP_A, "pin-2", "grouped", NO_IMAGES, NO_DRAFT)),
            ("prefill MoE arm", namespace_of(FP_A, "pin-1", "coopmk", NO_IMAGES, NO_DRAFT)),
            ("image preprocessing", namespace_of(FP_A, "pin-1", "grouped", "minicpmv-bicubic-torch-9-16x", NO_DRAFT)),
            ("draft", namespace_of(FP_A, "pin-1", "grouped", NO_IMAGES, "sha256:draft")),
        ] {
            assert_ne!(n, other, "{what} must separate namespaces");
        }
    }

    #[test]
    fn a_snapshot_names_the_draft_that_made_its_context_and_the_budget_page_holds_the_drafts_planes() {
        let (c, d) = (real(), draft());
        // the draft's identity is in the namespace: another draft pack, other tap layers, another page layout, no draft
        let a = "dspark|pack=sha256:aaaa|taps=[3, 11, 19, 27, 35]|kv=bf16|page=2621440";
        let b = a.replace("aaaa", "bbbb");
        let taps = a.replace("[3, 11,", "[2, 11,");
        let layout = a.replace("2621440", "1310720");
        let ns = |draft: &str| namespace_of(FP_A, "pin-1", "grouped", NO_IMAGES, draft);
        let all = [ns(NO_DRAFT), ns(a), ns(&b), ns(&taps), ns(&layout)];
        for i in 0..all.len() {
            for j in 0..i {
                assert_ne!(all[i], all[j], "namespaces {i} and {j} must differ");
            }
        }
        assert_ne!(draft_key(a), draft_key(&b));
        // the pool's page holds the target's planes and the draft's: the budget the plan hands the arena says so, and the arena that
        // is not told refuses it (and the other way round)
        let dp = memory::draft_page_bytes(&d);
        assert_eq!(dp, 2_621_440);
        let lay = Layout::from_config(&c);
        let with = CacheBudget::fit(3 << 30, snapshot_bytes(&c), page_bytes(&c) + dp);
        assert!(SnapshotArena::check(&lay, &c, &with, dp).is_ok());
        assert!(SnapshotArena::check(&lay, &c, &with, 0).is_err(), "a draft's page offered to an arena built without one");
        assert!(SnapshotArena::check(&lay, &c, &budget_for_bytes(&c, 3 << 30), dp).is_err(), "no draft page in the budget");
        // the per-snapshot record of whose draft context it holds follows the snapshot slot: set at capture, gone at release
        let mut arena = SnapshotArena::bookkeeping_only(&c, budget_for_bytes(&c, 3 << 30)).unwrap();
        let id = arena.book(300, 0).unwrap();
        assert_eq!(arena.draft_ctx(id), None);
        arena.set_draft_ctx(id, Some(draft_key(a)));
        assert_eq!(arena.draft_ctx(id), Some(draft_key(a)));
        arena.unbook(id).unwrap();
        let again = arena.book(300, 0).unwrap();
        assert_eq!(again, id, "the slot is recycled");
        assert_eq!(arena.draft_ctx(again), None, "a recycled snapshot slot holds no draft context of its previous entry");
    }

    #[test]
    fn a_cache_is_keyed_by_a_real_pack_fingerprint() {
        assert!(check_fingerprint(FP_A).is_ok());
        for bad in
            ["", "sha256:", "sha256:xyz", "md5:9bc579ceb1ae00bd016922d354e340bb395e9da52438cc226d1c898d5e63d03f", &FP_A[..70]]
        {
            let e = check_fingerprint(bad).unwrap_err().to_string();
            assert!(e.contains("fingerprint"), "{bad}: {e}");
        }
        let upper = FP_A.to_uppercase().replace("SHA256", "sha256");
        assert!(check_fingerprint(&upper).is_err(), "the fingerprint is lowercase hex");
    }

    #[test]
    fn an_arena_must_match_the_model_it_is_built_for() {
        let c = real();
        let good = budget_for_bytes(&c, 3 << 30);
        assert!(SnapshotArena::bookkeeping_only(&c, good).is_ok());
        for bad in [
            CacheBudget { snapshot_bytes: good.snapshot_bytes + 1, ..good },
            CacheBudget { page_bytes: good.page_bytes / 2, ..good },
            CacheBudget::NONE,
        ] {
            assert!(SnapshotArena::bookkeeping_only(&c, bad).is_err());
        }
    }

    #[test]
    fn the_book_releases_every_handle_exactly_once() {
        let mut b = SlotBook::new(2);
        let x = b.alloc().unwrap();
        let y = b.alloc().unwrap();
        assert!(b.alloc().is_none());
        b.meta[x as usize] = Some(Meta { len: 257, rope_delta: 0, path: PATH_ID });
        b.free(x).unwrap();
        assert!(b.free(x).is_err(), "a second release of one handle is refused");
        assert!(b.free(9).is_err());
        assert_eq!(b.alloc(), Some(x));
        b.abandon(x);
        b.abandon(y);
        assert_eq!(b.free.len(), 2);
        // through the arena: held and free always sum to the slots
        let c = real();
        let mut a = SnapshotArena::bookkeeping_only(&c, budget_for_bytes(&c, 3 << 30)).unwrap();
        let slots = a.budget().snapshot_slots;
        let id = a.book(300, 7).unwrap();
        assert_eq!(a.meta(id), Some(Meta { len: 300, rope_delta: 7, path: PATH_ID }));
        assert_eq!((a.held_count(), a.free_count()), (1, slots - 1));
        a.unbook(id).unwrap();
        assert!(a.unbook(id).is_err());
        assert_eq!((a.held_count(), a.free_count()), (0, slots));
    }

    #[test]
    fn index_bookkeeping_eviction_and_page_refcounts_are_engine_owned() {
        use gb_sched::prefix::{Limits, PrefixIndex};
        use std::sync::Arc;
        let mut i = PrefixIndex::new(13, Limits { snapshot_slots: 2, page_blocks: 3 });
        let (pa, pb) = (Arc::new(vec![1u32; 600]), Arc::new(vec![2u32; 600]));
        let make = |len, snap, pages| Captured { len, handle: PrefixHandle { snap, pages } };
        assert!(i.insert(PATH_ID, pa.clone(), make(300, 0, vec![7, 8])).stored());
        assert!(i.insert(PATH_ID, pb.clone(), make(500, 1, vec![7, 9])).stored());
        assert_eq!(i.pages_held(), 3);
        assert_eq!(i.page_holders(7), 2);
        // both entries are in use (looked up, not yet unpinned) and the older one is the least recently used: nothing may be evicted
        let ha = i.lookup(PATH_ID, &pa).unwrap();
        let hb = i.lookup(PATH_ID, &pb).unwrap();
        assert_eq!((ha.len, hb.len), (300, 500));
        assert!(i.evict_for_snapshots(1, 0).is_empty(), "a pinned entry (a handle in use) is never the victim");
        assert_eq!(i.pages_held(), 3);
        // the older one is released from use: it is the victim now, and its pages go with it unless another entry holds them
        i.unpin(ha.key);
        let released = i.evict_for_snapshots(1, 0);
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].snap, 0);
        assert_eq!(i.page_holders(7), 1);
        assert_eq!(i.page_holders(8), 0);
        assert_eq!(i.page_holders(9), 1);
        i.unpin(hb.key);
        i.check().unwrap();
    }
}
