//! The Nemotron prefix cache's arithmetic: the pure functions the backend (`backend.rs`), the plan (`engine.rs`) and the GPU model's snapshot
//! hooks are built from, and that the CPU tests pin. Design, numbers and the evidence plan: `docs/NEMOTRON_LONG_CONTEXT.md`; the engine's side of the
//! cache (index, cut policy, budget, opt-out): `docs/PREFIX_CACHE.md`.
//!
//! **What a snapshot of `len` tokens is** (everything a prompt leaves behind that its pages do not carry):
//! * the recurrent state of the 40 Mamba2 layers: S (f32 `[128][128][64]`) and the conv history (bf16 `[3][10,240]`), 170,229,760 bytes;
//! * the MTP head's carry: the post-`norm_f` tap row of position `len - 1` (f32 `[4096]`, 16,384 bytes), which the first entry the resumed prompt
//!   builds reads, and the soundness flag of the head's `TapBook` (`{ ok, first: len, n: 1 }` is rebuilt from `len` and the flag);
//! * the KV pages by reference (8 trunk layers and the head's stripe in the same pages, 256 tokens a page): pages `0..last` shared, the page
//!   `last = (len - 1) / 256` **copied**, always: the head writes its entry for position `len - 1` AFTER the cut, so a full last page is not safe to
//!   share the way the other families share it (`restore_pages`).
//!
//! **The path rule is one sentence about positions.** Every attention row, whichever entry point carries it (decode, verify window, prefill
//! slice), is attended by the exact tree below absolute position `FLASH_FROM_POS` (8,192) and by the flash tensor-core kernel at and above
//! it (`gb_cuda::nemotron_attn_prefill`, `docs/NEMOTRON_ATTN_LONG.md`), so a row's bits are independent of the slice grid and of the window it
//! would have been in at every position (gated: chunked == whole, windows == prefill, window == serial, drafted == serial, at 12k tokens),
//! and a cut is valid at any token count. What a snapshot holds depends on the arithmetic of the rows that made it, so a prompt of at most
//! 8,192 tokens (every row below the threshold: the tree's arithmetic alone) has [`PATH_ID_TREE`] and a longer one [`PATH_ID_FLASH`]
//! ([`path_of`]); a cut at or below 8,192 tokens is bit-valid under both, and is stored under the id of the prompt that made it.

use super::backend::next_chunk_end;
use super::memory::PAGE_TOKENS;
use gb_sched::prefix::{CacheBudget, ASSUMED_PAGES_PER_ENTRY};
use sha2::{Digest, Sha256};
use std::path::Path;

/// The key's path id of a prompt of at most [`FLASH_FROM_POS`] tokens: every row of it is below the threshold, so the tree computed all of
/// them (the arithmetic of every prompt before the flash path existed).
pub const PATH_ID_TREE: u32 = 1;

/// The key's path id of a prompt longer than [`FLASH_FROM_POS`] tokens: its rows at and above the threshold were computed by the flash kernel.
pub const PATH_ID_FLASH: u32 = 2;

/// The first absolute position the flash kernel serves (`gb_cuda::nemotron_attn_prefill::FLASH_FROM_POS`).
pub const FLASH_FROM_POS: usize = gb_cuda::nemotron_attn_prefill::FLASH_FROM_POS;

/// The key's path id of a prompt of `total` tokens: the arithmetic its prefill rows (positions `0..total`) run on.
pub const fn path_of(total: usize) -> u32 {
    if total <= FLASH_FROM_POS {
        PATH_ID_TREE
    } else {
        PATH_ID_FLASH
    }
}

/// The arithmetic every attention row is computed in, as a tag of the namespace: the position rule and its parameters
/// (`gb_cuda::nemotron_attn_prefill::FLASH_PATH_ID`). The large-row MoE prefill (package P1), if its kernel does not reproduce the decode
/// kernel's bits, is another arithmetic: its prompts get another path id from `Backend::prefix_path` and this tag changes with the format.
pub const PREFILL_ARITH: &str = "attn=tree<8192|flash>=8192(v1)+decode-kernels-rows<=64";

/// The cache's format version: part of the namespace, so a binary that stores snapshots differently cannot see another's entries.
pub const FORMAT: &str = "rexl3-nemotron-prefix-v1";

/// Bytes of the head's carry: one f32 tap row of the trunk's hidden width.
pub const CARRY_BYTES: u64 = (gb_cuda::nemotron_glue::HIDDEN * 4) as u64;

/// Bytes of one snapshot slot: the recurrent state of a slot (`state_bytes_per_seq`: every Mamba layer's S and conv) and, when the head is
/// loaded, its carry row. The memory plan and the arena both use this figure.
pub fn snapshot_bytes(state_bytes_per_seq: u64, mtp: bool) -> u64 {
    state_bytes_per_seq.saturating_add(if mtp { CARRY_BYTES } else { 0 })
}

/// The key namespace of this pack's snapshots: everything their bytes depend on besides the tokens. Another pack, another cache format, the head
/// present or not (its stripe is in the pages and its carry in the snapshot) or another arithmetic is another namespace, so entries are never
/// shared across them (the index is empty at every boot anyway). The prefill slice is NOT in it: rows are slice-grid invariant.
pub fn namespace_of(identity: &str, mtp: bool) -> u64 {
    namespace_with(identity, mtp, gb_cuda::nemotron_attn::configured_kv_dtype().unwrap_or_default())
}

/// [`namespace_of`] for a KV cache of `kv`: an fp16 entry is never read as int8 or the other way. The fp16 namespace's text is the one
/// before the switch existed (`kv=f16`), so nothing about an fp16 boot changes.
pub fn namespace_with(identity: &str, mtp: bool, kv: gb_cuda::nemotron_attn::KvDtype) -> u64 {
    let kv = match kv {
        gb_cuda::nemotron_attn::KvDtype::F16 => "f16",
        gb_cuda::nemotron_attn::KvDtype::Int8 => "int8-pow2-token-head(v1)",
    };
    let text = format!(
        "{FORMAT}|{identity}|mtp={}|page={PAGE_TOKENS}|kv={kv}|state=f32+bf16|carry=f32|arith={PREFILL_ARITH}{}{}",
        if mtp { "head" } else { "none" },
        // P1 adds nothing (its namespace is the one it always was); P2 is another arithmetic and another namespace
        super::contract::current().arith_tag(),
        if super::contract::current() == super::contract::PrefillContract::P2 && super::contract::router_mma() {
            "+router=mma"
        } else {
            ""
        },
    );
    text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// A short identity of the pack in `dir`: the digest of its `config.json` and of its safetensors index (every tensor, its shard and its place).
/// Header-level on purpose: nothing persists across boots, so the identity only has to tell packs apart inside one process's life and keep the
/// namespace honest; a payload edit that leaves both files unchanged is not detected (nor could the cache outlive the process to be fooled by it).
pub fn pack_identity(dir: &Path) -> Result<String, String> {
    let mut h = Sha256::new();
    h.update(b"rexl3-nemotron-pack-v1\0");
    for name in ["config.json", "model.safetensors.index.json"] {
        let path = dir.join(name);
        let bytes = std::fs::read(&path).map_err(|e| format!("pack identity: {}: {e}", path.display()))?;
        h.update(name.as_bytes());
        h.update([0]);
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(&bytes);
    }
    let d = h.finalize();
    Ok(d.iter().take(16).map(|b| format!("{b:02x}")).collect())
}

/// How a prefix of `len` tokens maps onto pages at restore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RestorePages {
    /// Pages the prefix occupies (`ceil(len / 256)`): what an entry holds a reference to.
    pub total: usize,
    /// Pages `0..shared` are shared by reference.
    pub shared: usize,
}

impl RestorePages {
    /// The page that is copied into a page the restored slot owns: the one that holds position `len - 1`, `None` for an empty prefix.
    pub fn copied(&self) -> Option<usize> {
        (self.total > 0).then_some(self.shared)
    }
}

/// Which pages of a prefix of `len` tokens are shared and which one is copied. **The page of position `len - 1` is always copied, a full one
/// included**: the head is one position behind the trunk, so its entry for position `len - 1` is written by the resumed prompt's first slice, and
/// when `len` is a multiple of 256 that position is the last row of a full page that the donor and every other sharer still read (the other
/// families copy only a partial page, because nothing writes into a full one again).
pub fn restore_pages(len: usize) -> RestorePages {
    if len == 0 {
        return RestorePages { total: 0, shared: 0 };
    }
    let last = (len - 1) / PAGE_TOKENS as usize;
    RestorePages { total: last + 1, shared: last }
}

/// Where the slice that starts at `done` ends: the next multiple of `chunk` (the absolute grid, so a restore at any `done` resumes on it), cut
/// short at the first of `cuts` inside it. A cut at the end of the slice is taken after the slice (no shortening). The bits of every row are the
/// same whatever the grid, so ending a slice early changes nothing but how many slices the prompt takes.
pub fn slice_end(done: usize, total: usize, chunk: usize, cuts: &[usize]) -> usize {
    let natural = next_chunk_end(done, total, chunk.max(1));
    cuts.iter().copied().filter(|&c| c > done && c < natural).min().unwrap_or(natural)
}

/// Whether taking a snapshot at `cut` costs an extra forward: it does unless `cut` is on the slice grid (every shared cut `256 x M` is, for a
/// chunk of 64; an end cut `P - tail` is for one prompt length in 64).
pub fn cut_is_free(cut: usize, chunk: usize) -> bool {
    cut % chunk.max(1) == 0
}

// ---------------------------------------------------------------------------------------------------------------------------- the budget

/// An entry is assumed to hold this fraction of the context limit (`max_context / 8`) when a byte budget is split into snapshot slots and
/// pages: eight such entries are one whole context, so a budget of eight units always holds one entry of the longest admissible prompt.
pub const ENTRY_CONTEXT_FRACTION: u64 = 8;

/// The fewest pages an entry is assumed to hold: the shared planner's own assumption (8 pages, a 2,048-token prefix).
pub const MIN_ENTRY_PAGES: u64 = ASSUMED_PAGES_PER_ENTRY;

/// How many units `Auto` is raised to at least, so one entry of the longest admissible prompt fits ([`ENTRY_CONTEXT_FRACTION`] units).
pub const AUTO_MIN_UNITS: u64 = ENTRY_CONTEXT_FRACTION;

/// Pages one entry is assumed to hold: `max(8, ceil(pages_of_the_context / 8))` (128 at the 262,144-token default, 512 at 1M).
pub fn entry_pages(max_context: usize) -> u64 {
    let pages = (max_context as u64).div_ceil(PAGE_TOKENS);
    pages.div_ceil(ENTRY_CONTEXT_FRACTION).max(MIN_ENTRY_PAGES)
}

/// Bytes of one budget unit: a snapshot slot and `entry_pages` pages.
pub fn unit_bytes(snapshot_bytes: u64, page_bytes: u64, entry_pages: u64) -> u64 {
    snapshot_bytes.saturating_add(entry_pages.saturating_mul(page_bytes))
}

/// The budget that fits `bytes`: whole units of one snapshot slot plus `entry_pages` pages, disabled (like `CacheBudget::fit`) below two units, since
/// a one-entry cache evicts itself on every insert.
pub fn split_budget(bytes: u64, snapshot_bytes: u64, page_bytes: u64, entry_pages: u64) -> CacheBudget {
    let unit = unit_bytes(snapshot_bytes, page_bytes, entry_pages);
    let units = if unit == 0 { 0 } else { bytes / unit };
    if units < 2 {
        return CacheBudget { snapshot_bytes, page_bytes, ..CacheBudget::NONE };
    }
    CacheBudget {
        snapshot_slots: units as usize,
        page_blocks: units.saturating_mul(entry_pages) as usize,
        snapshot_bytes,
        page_bytes,
    }
}

/// The most `Auto` spends: the shared 8 GiB cap, raised so that [`AUTO_MIN_UNITS`] units fit (one maximal entry), and in the plan still at most
/// half of what the KV pool left over.
pub fn auto_cap_bytes(snapshot_bytes: u64, page_bytes: u64, entry_pages: u64, shared_cap: u64) -> u64 {
    shared_cap.max(AUTO_MIN_UNITS.saturating_mul(unit_bytes(snapshot_bytes, page_bytes, entry_pages)))
}

// ------------------------------------------------------------------------------------------------------------------------------ the arena

/// What an arena slot says about the snapshot it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapMeta {
    /// The token count of the prefix.
    pub len: usize,
    /// The head's `TapBook` was sound when the snapshot was taken (a restored slot drafts only then).
    pub mtp_ok: bool,
}

/// The arena's bookkeeping, pure: a free list and the meta of each live slot. A handle is live from `take` until `release`, exactly once: the
/// runtime owns the bytes, this owns who may use them. (The CPU fake and the GPU model share it, so the books are the same code.)
#[derive(Clone, Debug)]
pub struct SnapshotBook {
    free: Vec<u32>,
    meta: Vec<Option<SnapMeta>>,
}

impl SnapshotBook {
    pub fn new(slots: usize) -> Self {
        Self { free: (0..slots as u32).rev().collect(), meta: vec![None; slots] }
    }

    /// Snapshot slots the arena has.
    pub fn capacity(&self) -> usize {
        self.meta.len()
    }

    /// Slots no snapshot holds.
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Slots a snapshot holds.
    pub fn live_count(&self) -> usize {
        self.meta.len() - self.free.len()
    }

    /// A free slot, now holding `meta`; `None` when every slot is taken.
    pub fn take(&mut self, meta: SnapMeta) -> Option<u32> {
        let id = self.free.pop()?;
        self.meta[id as usize] = Some(meta);
        Some(id)
    }

    /// What slot `id` holds, `None` when it is free or out of range.
    pub fn meta(&self, id: u32) -> Option<SnapMeta> {
        self.meta.get(id as usize).copied().flatten()
    }

    /// Give slot `id` back. A slot that is out of range, free already or released twice is an error by name and changes nothing.
    pub fn release(&mut self, id: u32) -> Result<SnapMeta, String> {
        let slots = self.meta.len();
        let m = self
            .meta
            .get_mut(id as usize)
            .ok_or_else(|| format!("snapshot {id} is out of range: the arena has {slots} slots"))?;
        let meta = m.take().ok_or_else(|| format!("snapshot {id} is not live (released already, or never taken)"))?;
        self.free.push(id);
        Ok(meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PACK: &str = include_str!("../../../../../testdata/nemotron_omni/config.json");

    #[test]
    fn a_snapshot_is_the_mamba_state_and_the_heads_carry_row() {
        // 40 layers x (S f32 128 x 128 x 64 + conv bf16 3 x 10,240)
        let state = 40 * (4_194_304 + 61_440);
        assert_eq!(state, 170_229_760);
        assert_eq!(CARRY_BYTES, 16_384);
        assert_eq!(snapshot_bytes(state, false), 170_229_760);
        assert_eq!(snapshot_bytes(state, true), 170_246_144);
        assert_eq!(170_246_144.0 / (1u64 << 20) as f64, 162.359375, "162.4 MiB");
        // as many tokens of KV (head included, 9,216 B a token) as a snapshot costs
        assert_eq!(170_246_144 / 9216, 18_472);
    }

    #[test]
    fn the_carry_is_one_f32_row_of_the_models_hidden_width() {
        let cfg = super::super::config::OmniConfig::from_json(&serde_json::from_str(PACK).unwrap()).unwrap();
        assert_eq!(CARRY_BYTES, cfg.hidden as u64 * 4);
    }

    #[test]
    fn the_page_of_the_last_position_is_always_the_one_that_is_copied() {
        for (len, total, shared) in [
            (1usize, 1usize, 0usize),
            (2, 1, 0),
            (255, 1, 0),
            (256, 1, 0), // aligned: the last page is FULL and is still the copied one
            (257, 2, 1),
            (511, 2, 1),
            (512, 2, 1),
            (513, 3, 2),
            (1_048_560, 4096, 4095),
            (1_048_576, 4096, 4095),
        ] {
            let p = restore_pages(len);
            assert_eq!((p.total, p.shared), (total, shared), "len {len}");
            assert_eq!(p.copied(), Some(shared));
            assert_eq!(p.total, len.div_ceil(256), "an entry holds a reference to every page of the prefix");
            assert!(p.shared + 1 == p.total, "exactly one page is copied");
        }
        assert_eq!(restore_pages(0), RestorePages { total: 0, shared: 0 });
        assert_eq!(restore_pages(0).copied(), None);
    }

    #[test]
    fn a_slice_ends_at_the_first_cut_inside_it_and_the_grid_survives() {
        // no cuts: the next multiple of the chunk, or the prompt's end
        assert_eq!(slice_end(0, 1000, 64, &[]), 64);
        assert_eq!(slice_end(64, 1000, 64, &[]), 128);
        assert_eq!(slice_end(960, 1000, 64, &[]), 1000);
        // a cut inside ends it there
        assert_eq!(slice_end(0, 1000, 64, &[30, 900]), 30);
        // and the next slice goes to the next multiple (the grid is the one the prompt would have had), not to cut + 64
        assert_eq!(slice_end(30, 1000, 64, &[30, 900]), 64);
        assert_eq!(slice_end(64, 1000, 64, &[30, 900]), 128);
        // a cut at the slice's end, at `done` or past it changes nothing
        assert_eq!(slice_end(0, 1000, 64, &[64]), 64);
        assert_eq!(slice_end(64, 1000, 64, &[64]), 128);
        assert_eq!(slice_end(0, 1000, 64, &[65, 900]), 64);
        // unordered cuts take the first one inside, not the first listed
        assert_eq!(slice_end(0, 1000, 64, &[50, 20]), 20);
        // a chunk of 1 row is every row; a smaller deadline limit puts the cuts on a finer grid
        assert_eq!(slice_end(5, 1000, 1, &[]), 6);
        assert_eq!(slice_end(0, 1000, 48, &[40]), 40);
        // the slice always advances and never passes the prompt or the chunk
        for done in 0..200usize {
            for chunk in [1usize, 7, 48, 64] {
                let end = slice_end(done, 200, chunk, &[3, 63, 64, 65, 100, 199]);
                assert!(end > done && end <= 200 && end - done <= chunk, "{done} {chunk} -> {end}");
            }
        }
    }

    #[test]
    fn a_cut_on_the_grid_is_free_and_any_other_costs_a_forward() {
        assert!(cut_is_free(256, 64) && cut_is_free(0, 64) && cut_is_free(1_048_576, 64));
        assert!(!cut_is_free(255, 64) && !cut_is_free(1_048_560 - 1, 64));
        // an end cut P - 16 is on the grid for one prompt length in 64
        assert_eq!((1000..1064).filter(|&p| cut_is_free(p - 16, 64)).count(), 1);
    }

    #[test]
    fn the_path_id_follows_the_prompts_rows_and_the_position_rule_is_the_one_the_kernels_run() {
        // a prompt's rows are positions 0..total: all below the threshold up to 8,192 tokens, one at or above it from 8,193
        assert_eq!(FLASH_FROM_POS, gb_cuda::nemotron_attn_prefill::FLASH_FROM_POS);
        assert_eq!(FLASH_FROM_POS, 8192);
        for total in [1usize, 2, 63, 64, 1000, 8191, 8192] {
            assert_eq!(path_of(total), PATH_ID_TREE, "{total}");
        }
        for total in [8193usize, 8194, 12_333, 100_000, 1_048_576] {
            assert_eq!(path_of(total), PATH_ID_FLASH, "{total}");
        }
        assert_ne!(PATH_ID_TREE, PATH_ID_FLASH);
        // the last row of a prompt of 8,192 tokens is position 8,191, the first row the flash kernel serves is position 8,192
        assert!(!gb_cuda::nemotron_attn_prefill::uses_flash(8192 - 1));
        assert!(gb_cuda::nemotron_attn_prefill::uses_flash(8192));
        // the namespace carries the rule: a cache made before the rule existed cannot be read as this one's
        assert!(PREFILL_ARITH.contains("tree<8192") && PREFILL_ARITH.contains("flash>=8192"));
    }

    #[test]
    fn the_namespace_separates_what_makes_snapshots_non_interchangeable() {
        let a = namespace_of("pack-a", true);
        assert_eq!(a, namespace_of("pack-a", true), "a function of its inputs");
        assert_ne!(a, namespace_of("pack-b", true), "another pack");
        assert_ne!(a, namespace_of("pack-a", false), "a snapshot with the head's carry and stripe is not one without it");
        use gb_cuda::nemotron_attn::KvDtype;
        assert_ne!(
            namespace_with("pack-a", true, KvDtype::F16),
            namespace_with("pack-a", true, KvDtype::Int8),
            "int8 pages are not fp16 pages"
        );
    }

    #[test]
    fn the_pack_identity_follows_the_config_and_the_index() {
        let dir = std::env::temp_dir().join(format!("rexl3-nemo-ident-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(pack_identity(&dir).unwrap_err().contains("config.json"), "a missing file is named");
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        std::fs::write(dir.join("model.safetensors.index.json"), "{\"weight_map\":{}}").unwrap();
        let a = pack_identity(&dir).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(a, pack_identity(&dir).unwrap());
        std::fs::write(dir.join("model.safetensors.index.json"), "{\"weight_map\":{\"a\":\"s\"}}").unwrap();
        assert_ne!(a, pack_identity(&dir).unwrap(), "another index is another pack");
        std::fs::write(dir.join("config.json"), "{ }").unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_entry_is_an_eighth_of_the_context_and_never_below_the_shared_planners_eight_pages() {
        assert_eq!(entry_pages(8_192), 8, "the shared assumption is the floor");
        assert_eq!(entry_pages(16_384), 8);
        assert_eq!(entry_pages(32_768), 16);
        assert_eq!(entry_pages(262_144), 128);
        assert_eq!(entry_pages(1 << 20), 512);
        assert_eq!(entry_pages(1_000_000), 489, "ceil(3,907 / 8)");
        // eight entries hold at least the whole context's pages: one maximal entry always fits a budget of eight units
        for ctx in [1usize, 255, 256, 257, 32_768, 100_000, 262_144, 1_000_000, 1 << 20] {
            assert!(8 * entry_pages(ctx) >= (ctx as u64).div_ceil(256), "{ctx}");
        }
    }

    #[test]
    fn a_budget_is_whole_units_and_two_at_least() {
        let (snap, page) = (170_246_144u64, 2_359_296u64);
        let unit = unit_bytes(snap, page, 128);
        assert_eq!(unit, 472_236_032, "450.4 MiB");
        assert!(!split_budget(2 * unit - 1, snap, page, 128).enabled(), "one unit evicts itself on every insert");
        let b = split_budget(2 * unit, snap, page, 128);
        assert_eq!((b.snapshot_slots, b.page_blocks, b.snapshot_bytes, b.page_bytes), (2, 256, snap, page));
        assert_eq!(b.bytes(), 2 * unit);
        // 8 GiB at the default context: 18 entries and 2,304 pages (589,824 tokens, 2.25 contexts)
        let b = split_budget(8 << 30, snap, page, entry_pages(262_144));
        assert_eq!((b.snapshot_slots, b.page_blocks), (18, 2304));
        assert!(b.bytes() <= 8 << 30);
        // the shared planner's split of the same bytes could not hold a prompt of 100,000 tokens (391 pages): 45 entries, 360 pages
        let shared = CacheBudget::fit(8 << 30, snap, page);
        assert_eq!((shared.snapshot_slots, shared.page_blocks), (45, 360));
        assert!(shared.page_blocks < 100_000usize.div_ceil(256));
        assert!(b.page_blocks >= 100_000usize.div_ceil(256));
    }

    #[test]
    fn nemotron_keeps_its_own_rule_and_plans_a_full_window_entry_at_1m() {
        // Nemotron does not use the shared window rule (`plan_memory_cached_windowed`): its `Auto` is raised until eight units fit, and eight
        // units of `max(8, ceil(context_pages / 8))` pages are one whole 1,048,576-token window
        let (snap, page) = (170_246_144u64, 2_359_296u64);
        let window = 1u64 << 20;
        let ppe = entry_pages(window as usize);
        assert_eq!(ppe, 512);
        let cap = auto_cap_bytes(snap, page, ppe, 8 << 30);
        let b = split_budget(cap, snap, page, ppe);
        assert_eq!((b.snapshot_slots, b.page_blocks), (8, 4096), "a full-window entry: 1,048,576 / 256 pages");
        assert_eq!(b.reach_tokens(), window);
        assert_eq!(b.bytes(), cap);
    }

    #[test]
    fn auto_is_raised_until_one_entry_of_the_longest_prompt_fits() {
        let (snap, page) = (170_246_144u64, 2_359_296u64);
        let shared_cap = 8u64 << 30;
        // at the default context 8 units are 3.5 GiB: the shared cap governs
        assert_eq!(auto_cap_bytes(snap, page, entry_pages(262_144), shared_cap), shared_cap);
        // at 1M they are 10.27 GiB and the cap rises
        let ppe = entry_pages(1 << 20);
        let cap = auto_cap_bytes(snap, page, ppe, shared_cap);
        assert_eq!(cap, 8 * (snap + 512 * page));
        assert!(cap > shared_cap);
        let b = split_budget(cap, snap, page, ppe);
        assert_eq!((b.snapshot_slots, b.page_blocks), (8, 4096), "one maximal prompt: 4,096 pages");
        assert!(b.page_blocks >= (1usize << 20).div_ceil(256));
        // and below the cap a smaller box gets what it has
        assert!(split_budget(shared_cap, snap, page, ppe).snapshot_slots < 8);
    }

    #[test]
    fn the_arena_book_hands_a_slot_out_once_and_takes_it_back_once() {
        let mut b = SnapshotBook::new(2);
        assert_eq!((b.capacity(), b.free_count(), b.live_count()), (2, 2, 0));
        let m = SnapMeta { len: 300, mtp_ok: true };
        let x = b.take(m).unwrap();
        let y = b.take(SnapMeta { len: 700, mtp_ok: false }).unwrap();
        assert_ne!(x, y);
        assert!(b.take(m).is_none(), "full");
        assert_eq!((b.free_count(), b.live_count()), (0, 2));
        assert_eq!(b.meta(x), Some(m));
        assert_eq!(b.release(x), Ok(m));
        assert!(b.release(x).unwrap_err().contains("not live"), "a second release is an error by name");
        assert!(b.release(9).unwrap_err().contains("out of range"));
        assert_eq!(b.meta(x), None);
        assert_eq!((b.free_count(), b.live_count()), (1, 1));
        assert_eq!(b.take(m), Some(x), "the slot is reusable");
    }
}
