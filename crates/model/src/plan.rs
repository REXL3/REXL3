//! The startup memory plan on a unified-memory box.
//!
//! On the GB10 an out-of-memory can hang the whole host, so nothing is allocated on hope. The allocatable bytes are what is free
//! once the weights are resident (`min(cudaMemGetInfo free, MemAvailable)`) less three holdbacks, each subtracted exactly once,
//! here: the family's engine reserve (activation scratch, window records, workspaces: `MemoryProfile::engine_reserve_bytes`), the
//! vision reserve the caller decided to count, and a floor kept for the OS and driver. The per-slot state comes out first, the
//! rest becomes KV pages, and admission is given exactly those numbers, so a request it admits can always get its pages.
//!
//! Everything family-specific arrives as a validated [`MemoryProfile`], whose KV page is at least one byte by construction: the
//! division below cannot be by zero.

use crate::kvtables::BLOCK_TOKENS;
use gb_sched::prefix::{cache_margin_bytes, CacheBudget, WindowRule};
use gb_sched::{MemoryError, MemoryProfile};

/// How much memory the prefix cache may take (`--prefix-cache-mib`; design: `docs/PREFIX_CACHE.md`, "Admission and memory").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheRequest {
    /// No cache.
    Off,
    /// What the plan has spare after the KV pool and the slot state, at most `cap_bytes` and at most half of it: the cache never takes
    /// memory the KV pool would have had (a box whose free memory shrank, for instance after a buffered boot, simply gets no cache).
    Auto { cap_bytes: u64 },
    /// Exactly this many bytes, taken before the KV pool is sized: a request the box cannot meet is an error, not a smaller cache.
    Fixed(u64),
}

/// The default cap of `CacheRequest::Auto`: 8 GiB, about 25 entries of a 2k-token prefix.
pub const AUTO_CACHE_CAP_BYTES: u64 = 8 << 30;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryPlan {
    /// Pages in the KV pool (each holds `BLOCK_TOKENS` tokens of every layer that caches context).
    pub kv_blocks: usize,
    /// Bytes of one page across all layers that cache context.
    pub block_bytes: u64,
    /// Bytes reserved per sequence slot for recurrent state and the drafter's context.
    pub state_bytes_per_slot: u64,
    /// The most tokens that can be in flight across all sequences.
    pub kv_tokens: u64,
    /// Bytes admission may hand out (KV plus slot state): what `Admission::new` is given as its capacity.
    pub admit_bytes: u64,
    /// Bytes held back for the engine's own allocations after the plan (already out of `admit_bytes`).
    pub engine_reserve_bytes: u64,
    /// Bytes held back for the vision tower's encoder scratch and image features (already out of `admit_bytes`).
    pub vision_reserve_bytes: u64,
    /// The prefix cache: snapshot slots and KV pages carved out for it, on top of `kv_blocks` (the pool holds `kv_blocks + cache.page_blocks`
    /// pages; admission is told only about `admit_bytes`). `CacheBudget::NONE` without one.
    pub cache: CacheBudget,
    /// What was left after the KV pool and the slot state (the bytes `CacheRequest::Auto` takes at most half of).
    pub spare_bytes: u64,
    /// The resident `W_q` budget of the prefill tail-floor lever (`REXL3_WQ_RESIDENT_GIB`; 0 = off, the default). It
    /// comes out of the allocatable bytes before the KV pool is sized and is part of `planned_bytes`, so a boot the
    /// box cannot hold is refused, never quietly oversubscribed. The slots fill lazily as prompts run.
    pub wq_budget_bytes: u64,
}

impl MemoryPlan {
    /// The admission this plan's engine runs: the byte capacity the plan hands out (`admit_bytes`; the planner already took the weights, the
    /// reserves and the floor out of it), the longest context the server accepts, and the KV pool in tokens (`kv_tokens`).
    ///
    /// The pool is declared because the byte capacity alone cannot say it: it prices the slot state of every sequence, but the state of ALL
    /// slots is allocated up front, so with fewer sequences running the idle slots' state read as room for KV the pool does not have. A prompt
    /// past the pool was admitted, prefilled for minutes and failed `OutOfPages` (`docs/EVAL_RESULTS.md` "Serve hardening"). Every engine
    /// builds its admission here (a source guard in `serve::tests` keeps it so), and the limit it states is
    /// [`gb_sched::Admission::request_limit`]: `min(max_model_len, kv_tokens)`.
    pub fn admission(
        &self,
        model: gb_sched::ModelMemory,
        slots: usize,
        max_model_len: usize,
    ) -> Result<gb_sched::Admission, MemoryError> {
        Ok(gb_sched::Admission::new(self.admit_bytes, 0, model, slots)?
            .with_max_context(max_model_len as u64)
            .with_kv_tokens(self.kv_tokens))
    }

    /// Bytes the engine allocates beyond the weights under this plan for `slots` slots: the engine and vision reserves, the slots' state, the KV
    /// pool's pages, the prefix cache (its snapshot slots and its pages) and the resident `W_q` budget. Always at most `free - floor` of the call
    /// that made the plan (the test `a_plan_never_asks_for_more_than_it_was_given`); the boot error of an allocation that fails anyway quotes it.
    pub fn planned_bytes(&self, slots: usize) -> u64 {
        self.engine_reserve_bytes
            + self.vision_reserve_bytes
            + self.state_bytes_per_slot * slots as u64
            + self.kv_blocks as u64 * self.block_bytes
            + self.cache.bytes()
            + self.wq_budget_bytes
    }
}

/// The least a shrunk slot count must leave each slot, in tokens of the KV pool: a count whose pool holds less is not a boot but a refusal that
/// looks like one (the pool is shared, so four slots with 17,664 tokens between them hold a conversation of 4,400 tokens each).
pub const MIN_TOKENS_PER_SLOT: u64 = 4096;

/// A slot count and the plan that goes with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotChoice {
    /// What the operator asked for (or the default).
    pub requested: usize,
    /// What the plan holds.
    pub slots: usize,
    pub plan: MemoryPlan,
}

impl SlotChoice {
    pub fn shrunk(&self) -> bool {
        self.slots < self.requested
    }
}

/// The most slots, at most `requested`, whose plan fits and leaves a pool of at least `min(kv_tokens_cap, slots x MIN_TOKENS_PER_SLOT)` tokens.
/// `plan_at(slots)` is the family's plan for that count (the vision reserve moves with it). `None`: not even one slot is viable.
pub fn largest_viable_slots<F>(requested: usize, kv_tokens_cap: u64, plan_at: F) -> Option<SlotChoice>
where
    F: Fn(usize) -> Result<MemoryPlan, PlanError>,
{
    (1..=requested).rev().find_map(|slots| {
        let plan = plan_at(slots).ok()?;
        (plan.kv_tokens >= (slots as u64 * MIN_TOKENS_PER_SLOT).min(kv_tokens_cap)).then_some(SlotChoice {
            requested,
            slots,
            plan,
        })
    })
}

/// The slot count a boot plans with. `auto` (the count was not asked for): the most that fit, see [`largest_viable_slots`]; where none does, the
/// request's own outcome, a refusal in the planner's words or a thin pool as it always was. Not `auto` (`--slots N`): exactly `requested` or the
/// planner's refusal, never a different count.
pub fn choose_slots<F>(requested: usize, auto: bool, kv_tokens_cap: u64, plan_at: F) -> Result<SlotChoice, PlanError>
where
    F: Fn(usize) -> Result<MemoryPlan, PlanError>,
{
    if auto {
        if let Some(choice) = largest_viable_slots(requested, kv_tokens_cap, &plan_at) {
            return Ok(choice);
        }
    }
    plan_at(requested).map(|plan| SlotChoice { requested, slots: requested, plan })
}

#[derive(Debug, PartialEq, Eq)]
pub enum PlanError {
    /// Nothing is left for the KV cache after the reserves, the floor and the slot state.
    NoRoom { allocatable: u64, needed: u64 },
    /// The profile was built for a different page size than the KV pool uses.
    PageMismatch { profile: u64, pool: u64 },
    /// At least one sequence slot is needed.
    NoSlots,
    /// The byte arithmetic does not fit in 64 bits.
    Overflow(&'static str),
    /// The memory model was refused.
    Memory(MemoryError),
    /// An explicit prefix-cache size that cannot hold two entries (a snapshot slot and its share of pages each).
    CacheTooSmall { bytes: u64, entry_bytes: u64 },
    /// An explicit prefix-cache size cut by pages first ([`plan_memory_cached_windowed`]) that is under the least that is a cache: the
    /// minimum snapshot slots and the pages of [`gb_sched::prefix::MIN_CACHE_REACH_TOKENS`] tokens.
    CacheBelowMinimumReach { bytes: u64, minimum_bytes: u64 },
}

impl From<MemoryError> for PlanError {
    fn from(e: MemoryError) -> Self {
        PlanError::Memory(e)
    }
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::NoRoom { allocatable, needed } => write!(
                f,
                "not enough memory: {:.1} GiB allocatable after the floor, {:.1} GiB needed for slot state and at least one page",
                *allocatable as f64 / (1u64 << 30) as f64,
                *needed as f64 / (1u64 << 30) as f64
            ),
            PlanError::PageMismatch { profile, pool } => {
                write!(f, "the memory profile is for {profile}-token pages but the KV pool uses {pool}-token pages")
            }
            PlanError::NoSlots => write!(f, "at least one sequence slot is needed"),
            PlanError::Overflow(what) => write!(f, "byte arithmetic overflows in {what}"),
            PlanError::Memory(e) => write!(f, "{e}"),
            PlanError::CacheTooSmall { bytes, entry_bytes } => write!(
                f,
                "--prefix-cache-mib {} MiB cannot hold two entries of {} MiB each (a snapshot slot and 8 pages); give it at least {} MiB, or 0 to turn the cache off",
                bytes >> 20,
                entry_bytes >> 20,
                (2 * entry_bytes).div_ceil(1 << 20)
            ),
            PlanError::CacheBelowMinimumReach { bytes, minimum_bytes } => write!(
                f,
                "--prefix-cache-mib {} MiB is under the least that is a cache ({} snapshot slots and {} tokens of KV pages); give it at least {} MiB, or 0 to turn the cache off",
                bytes >> 20,
                gb_sched::prefix::MIN_CACHE_SNAPSHOT_SLOTS,
                gb_sched::prefix::MIN_CACHE_REACH_TOKENS,
                minimum_bytes.div_ceil(1 << 20)
            ),
        }
    }
}

impl std::error::Error for PlanError {}

/// `free_bytes`: min(cudaMemGetInfo free, MemAvailable) measured after the weights are resident (the engine reserve and the
/// vision reserve are taken out here, not by the caller). `floor_bytes`: kept free for the OS, page cache and driver.
/// `vision_reserve_bytes`: what the caller counts for the vision tower (zero when the tower is already resident and its features
/// are deliberately uncounted, as for Qwen; `MemoryProfile::vision_reserve_bytes` otherwise). `kv_tokens_cap`: do not plan
/// more KV than this many tokens (the longest context the server will accept times a small concurrency, no more).
pub fn plan_memory(
    profile: &MemoryProfile,
    free_bytes: u64,
    floor_bytes: u64,
    slots: usize,
    kv_tokens_cap: u64,
    vision_reserve_bytes: u64,
) -> Result<MemoryPlan, PlanError> {
    plan_memory_cached(profile, free_bytes, floor_bytes, slots, kv_tokens_cap, vision_reserve_bytes, CacheRequest::Off, 0, 0)
}

/// [`plan_memory`] with a prefix cache: `cache` says how much it may take and `snapshot_bytes` is what one snapshot slot costs (the backend's
/// own figure, `SnapPool::bytes_per_slot`). An explicit size is carved out first (the KV pool gets what remains and the plan fails closed if
/// less than the slot state and one page is left); `Auto` only spends what the KV pool left over, so a plan that fit without a cache is exactly
/// the plan with it, and `kv_blocks` is the same either way. `wq_budget_bytes` is the resident `W_q` of the prefill tail-floor lever
/// (`REXL3_WQ_RESIDENT_GIB`), which comes out of the allocatable bytes before the KV pool is sized: a boot the box cannot hold is refused.
#[allow(clippy::too_many_arguments)]
pub fn plan_memory_cached(
    profile: &MemoryProfile,
    free_bytes: u64,
    floor_bytes: u64,
    slots: usize,
    kv_tokens_cap: u64,
    vision_reserve_bytes: u64,
    cache: CacheRequest,
    snapshot_bytes: u64,
    wq_budget_bytes: u64,
) -> Result<MemoryPlan, PlanError> {
    plan_memory_cached_with(
        profile,
        free_bytes,
        floor_bytes,
        slots,
        kv_tokens_cap,
        vision_reserve_bytes,
        cache,
        snapshot_bytes,
        wq_budget_bytes,
        CacheSizing::Legacy,
    )
}

/// [`plan_memory_cached`] for the families that size the prefix cache from the window (Qwen3.8, GLM and Flash-Next;
/// `docs/PREFIX_CACHE_SIZING.md` section 5): the cache gets the pages of one full `window_tokens` entry before anything else
/// ([`gb_sched::prefix::CacheBudget::fit_pages_first`]) for BOTH an explicit size and `Auto`, and the plan reports the reach it ended with.
///
/// - `Fixed(bytes)`: the rule is cut from exactly those bytes, taken before the KV pool is sized (the rest of the pin, past the target, is spent
///   too: the operator asked for it).
/// - `Auto`: the rule is cut from what the KV pool left, less [`gb_sched::prefix::cache_margin_bytes`] (the margin that keeps a boot-to-boot
///   whole-GiB swing of the free-memory reading from sizing the cache at the edge of what fits). The 8 GiB cap of the legacy `Auto` does not apply.
/// - `Off`: no cache, as everywhere.
///
/// A family opts in by calling this instead of [`plan_memory_cached`]; every other family keeps calling the old one, which is unchanged.
#[allow(clippy::too_many_arguments)]
pub fn plan_memory_cached_windowed(
    profile: &MemoryProfile,
    free_bytes: u64,
    floor_bytes: u64,
    slots: usize,
    kv_tokens_cap: u64,
    vision_reserve_bytes: u64,
    cache: CacheRequest,
    snapshot_bytes: u64,
    wq_budget_bytes: u64,
    window_tokens: u64,
) -> Result<MemoryPlan, PlanError> {
    plan_memory_cached_with(
        profile,
        free_bytes,
        floor_bytes,
        slots,
        kv_tokens_cap,
        vision_reserve_bytes,
        cache,
        snapshot_bytes,
        wq_budget_bytes,
        CacheSizing::Window { window_tokens },
    )
}

/// How the prefix cache is cut into snapshot slots and pages.
#[derive(Clone, Copy)]
enum CacheSizing {
    /// An explicit size in whole units of one snapshot slot and 8 pages, `Auto` at most half the spare memory and 8 GiB
    /// ([`CacheBudget::fit`]): every family that has not opted in.
    Legacy,
    /// The window rule ([`CacheBudget::fit_pages_first`]).
    Window { window_tokens: u64 },
}

#[allow(clippy::too_many_arguments)]
fn plan_memory_cached_with(
    profile: &MemoryProfile,
    free_bytes: u64,
    floor_bytes: u64,
    slots: usize,
    kv_tokens_cap: u64,
    vision_reserve_bytes: u64,
    cache: CacheRequest,
    snapshot_bytes: u64,
    wq_budget_bytes: u64,
    sizing: CacheSizing,
) -> Result<MemoryPlan, PlanError> {
    if slots == 0 {
        return Err(PlanError::NoSlots);
    }
    let pool = BLOCK_TOKENS as u64;
    if profile.shape().page_tokens != pool {
        return Err(PlanError::PageMismatch { profile: profile.shape().page_tokens, pool });
    }
    let block_bytes = profile.kv_page_bytes();
    let state_bytes_per_slot = profile.state_bytes_per_slot();
    let slot_state = state_bytes_per_slot.checked_mul(slots as u64).ok_or(PlanError::Overflow("slot state"))?;
    let engine_reserve_bytes = profile.engine_reserve_bytes();
    // an explicit cache size comes out of the allocatable bytes first; the KV pool is sized from what is left
    let fixed = match cache {
        CacheRequest::Fixed(bytes) => match sizing {
            CacheSizing::Legacy => {
                let b = CacheBudget::fit(bytes, snapshot_bytes, block_bytes);
                if !b.enabled() {
                    return Err(PlanError::CacheTooSmall {
                        bytes,
                        entry_bytes: snapshot_bytes + gb_sched::prefix::ASSUMED_PAGES_PER_ENTRY * block_bytes,
                    });
                }
                Some(b)
            }
            CacheSizing::Window { window_tokens } => {
                let rule = WindowRule { window_tokens, live_slots: slots };
                let b = CacheBudget::fit_pages_first(bytes, snapshot_bytes, block_bytes, rule, true);
                if !b.enabled() {
                    return Err(PlanError::CacheBelowMinimumReach {
                        bytes,
                        minimum_bytes: CacheBudget::pages_first_minimum_bytes(snapshot_bytes, block_bytes),
                    });
                }
                Some(b)
            }
        },
        _ => None,
    };
    let allocatable = free_bytes
        .saturating_sub(engine_reserve_bytes)
        .saturating_sub(vision_reserve_bytes)
        .saturating_sub(floor_bytes)
        .saturating_sub(fixed.map_or(0, |b| b.bytes()))
        .saturating_sub(wq_budget_bytes);
    let needed = slot_state.checked_add(block_bytes).ok_or(PlanError::Overflow("slot state plus one page"))?;
    if allocatable < needed {
        return Err(PlanError::NoRoom { allocatable, needed });
    }
    let kv_bytes = allocatable - slot_state;
    // every sequence may waste part of its last page, so keep `slots` pages beyond what the token budget implies
    let mut kv_blocks = kv_bytes.checked_div(block_bytes).ok_or(PlanError::Overflow("page size"))? as usize;
    let cap_blocks = (kv_tokens_cap as usize).div_ceil(BLOCK_TOKENS) + slots;
    kv_blocks = kv_blocks.min(cap_blocks).max(1);
    let admit_kv_blocks = kv_blocks.saturating_sub(slots).max(1);
    let kv_tokens = (admit_kv_blocks * BLOCK_TOKENS) as u64;
    let admit_bytes = kv_tokens
        .checked_mul(profile.kv_bytes_per_token())
        .and_then(|kv| kv.checked_add(slot_state))
        .ok_or(PlanError::Overflow("admission capacity"))?;
    // what the KV pool and the slot state left: `Auto` takes at most half of it
    let spare_bytes = allocatable.saturating_sub(slot_state).saturating_sub(kv_blocks as u64 * block_bytes);
    let cache = match cache {
        CacheRequest::Off => CacheBudget::NONE,
        CacheRequest::Fixed(_) => fixed.expect("computed above"),
        CacheRequest::Auto { cap_bytes } => match sizing {
            CacheSizing::Legacy => CacheBudget::fit(cap_bytes.min(spare_bytes / 2), snapshot_bytes, block_bytes),
            CacheSizing::Window { window_tokens } => {
                let budget = spare_bytes.saturating_sub(cache_margin_bytes(spare_bytes));
                let rule = WindowRule { window_tokens, live_slots: slots };
                CacheBudget::fit_pages_first(budget, snapshot_bytes, block_bytes, rule, false)
            }
        },
    };
    Ok(MemoryPlan {
        kv_blocks,
        block_bytes,
        state_bytes_per_slot,
        kv_tokens,
        admit_bytes,
        engine_reserve_bytes,
        vision_reserve_bytes,
        cache,
        spare_bytes,
        wq_budget_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QwenConfig;
    use crate::families::glm::{memory as glm_memory, GlmConfig};
    use crate::families::qwen;
    use crate::memory::{MemoryOptions, VisionOptions, ENCODERS_IN_FLIGHT};
    use gb_sched::{Admission, KvLayerClass, KvLayout, MemoryModel, ModelMemory, StepShape, VisionMemory};

    const QWEN_CONFIG: &str = include_str!("../../../testdata/qwen38/config.json");
    const GLM_CONFIG: &str = include_str!("../../../testdata/glm53/config.json");
    const GIB: u64 = 1 << 30;

    fn real_qwen() -> QwenConfig {
        QwenConfig::from_json(&serde_json::from_str(QWEN_CONFIG).unwrap()).unwrap()
    }

    fn qwen_profile() -> MemoryProfile {
        qwen::memory_profile(&real_qwen(), &MemoryOptions::serving_defaults()).unwrap()
    }

    fn glm_profile(mtp: bool) -> MemoryProfile {
        let c = GlmConfig::from_json(&serde_json::from_str(GLM_CONFIG).unwrap()).unwrap();
        glm_memory::memory_profile(&c, &MemoryOptions { mtp, ..MemoryOptions::serving_defaults() }).unwrap()
    }

    // ---- planner arithmetic

    #[test]
    fn a_page_is_sixteen_mib() {
        let p = plan_memory(&qwen_profile(), 80 * GIB, 6 * GIB, 8, u64::MAX / 2, 0).unwrap();
        assert_eq!(p.block_bytes, 16 << 20);
    }

    #[test]
    fn slot_state_comes_out_before_kv() {
        let profile = qwen_profile();
        let p = plan_memory(&profile, 60 * GIB, 6 * GIB, 8, u64::MAX / 2, 0).unwrap();
        let slot = profile.state_bytes_per_slot() * 8;
        // admission is never promised more than was planned, and the pool holds the promised tokens plus a page per slot
        assert!(p.admit_bytes + 6 * GIB + p.engine_reserve_bytes <= 60 * GIB);
        assert!(p.kv_blocks as u64 * p.block_bytes + slot + 6 * GIB + p.engine_reserve_bytes <= 60 * GIB);
        assert!(p.kv_blocks * BLOCK_TOKENS >= p.kv_tokens as usize + 8 * BLOCK_TOKENS);
        // a request admitted at the planned token budget can always get its pages: sum(ceil(T/256)) <= tokens/256 + slots
        assert!(p.kv_tokens / BLOCK_TOKENS as u64 + 8 <= p.kv_blocks as u64);
    }

    #[test]
    fn the_token_cap_limits_the_pool() {
        let p = plan_memory(&qwen_profile(), 100 * GIB, 6 * GIB, 4, 262144, 0).unwrap();
        assert_eq!(p.kv_blocks, 262144 / 256 + 4);
        assert_eq!(p.kv_tokens, 262144);
    }

    #[test]
    fn no_room_is_an_error_not_a_zero_pool() {
        let profile = qwen_profile();
        assert!(matches!(plan_memory(&profile, 6 * GIB, 6 * GIB, 4, 1 << 20, 0), Err(PlanError::NoRoom { .. })));
        assert!(plan_memory(&profile, 12 * GIB, 6 * GIB, 8, 1 << 20, 0).is_err(), "8 slots of state do not fit in 1 GiB");
    }

    #[test]
    fn the_planner_refuses_what_it_cannot_plan() {
        let profile = qwen_profile();
        assert_eq!(plan_memory(&profile, 100 * GIB, 6 * GIB, 0, 1 << 20, 0), Err(PlanError::NoSlots));
        // a profile built for another page size than the pool's
        struct OtherPage;
        impl MemoryModel for OtherPage {
            fn family(&self) -> &str {
                "other-page"
            }
            fn kv_layout(&self) -> KvLayout {
                KvLayout::new().with(KvLayerClass::gqa(16, 4, 256, 2))
            }
            fn recurrent_layers(&self) -> u64 {
                0
            }
            fn state_bytes_per_seq(&self) -> u64 {
                0
            }
            fn window_record_bytes_per_row(&self) -> u64 {
                0
            }
            fn scratch_bytes(&self, rows: u64) -> u64 {
                rows
            }
            fn activation_reserve(&self) -> u64 {
                1
            }
            fn drafter_state_bytes_per_slot(&self) -> u64 {
                0
            }
            fn vision(&self) -> Option<VisionMemory> {
                None
            }
        }
        let odd = MemoryProfile::new(&OtherPage, StepShape { page_tokens: 128, window_rows: 8, scratch_rows: 8 }).unwrap();
        assert_eq!(
            plan_memory(&odd, 100 * GIB, 6 * GIB, 4, 1 << 20, 0),
            Err(PlanError::PageMismatch { profile: 128, pool: 256 })
        );
    }

    #[test]
    fn each_holdback_comes_out_of_the_allocatable_bytes_exactly_once() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let profile = glm_profile(false);
        let (floor, vision) = (8 * GIB, 1_000_000_000);
        let free = 40 * GIB;
        let p = plan_memory(&profile, free, floor, 8, 1 << 30, vision).unwrap();
        // the engine reserve (5 GiB blanket) and the vision reserve are recorded, and with the floor and the slot state they
        // leave exactly the pool the page count came from
        assert_eq!((p.engine_reserve_bytes, p.vision_reserve_bytes), (5 * GIB, vision));
        let allocatable = free - 5 * GIB - vision - floor;
        let slot_state = 8 * profile.state_bytes_per_slot();
        assert_eq!(p.kv_blocks as u64, (allocatable - slot_state) / p.block_bytes);
        // one more byte of vision reserve is one fewer byte for pages, never a double count
        let tighter = plan_memory(&profile, free, floor, 8, 1 << 30, vision + p.block_bytes).unwrap();
        assert_eq!(tighter.kv_blocks, p.kv_blocks - 1);
    }

    // ---- the Qwen plan is pinned to the numbers the engine shipped with ------------------------------------------------------
    //
    // These literals were produced by the code as it stood at c655bea (before the family seam), for the real Qwen3.8-27B
    // `config.json` (`testdata/qwen38/config.json`). They must never be edited to make a refactor pass: a Qwen pack has to plan
    // byte-for-byte the memory it always did.

    /// The arithmetic `serve::build_engine` applies to a Qwen pack: the planner takes the 5 GiB engine reserve, the floor and the
    /// slot state (the 16 MiB of draft context per slot included) out of `free`, and `Admission` is handed the plan's `admit_bytes`
    /// with no floor, no weights and no reserve of its own. Returns the plan, the `ModelMemory` admission is built with, and the
    /// capacity admission ends up with.
    fn qwen_engine_plan(
        c: &QwenConfig,
        free: u64,
        floor: u64,
        slots: usize,
        cap: u64,
    ) -> Result<(MemoryPlan, ModelMemory, u64), PlanError> {
        let profile = qwen::memory_profile(c, &MemoryOptions::serving_defaults())?;
        // a pack served without images: no tower, so no features in flight and nothing counted (what every Qwen plan was before Q-ADM)
        let p = plan_memory(&profile, free, floor, slots, cap, 0)?;
        let mm = ModelMemory::from_profile(&profile, 0, 0);
        let capacity = Admission::new(p.admit_bytes, 0, mm.clone(), slots)?.capacity();
        Ok((p, mm, capacity))
    }

    #[test]
    fn the_qwen_plan_is_pinned_to_what_the_engine_shipped_with() {
        let c = real_qwen();
        // (free GiB after the weights, slots, kv token cap) -> (kv_blocks, kv_tokens, admit_bytes, admission capacity)
        type Row = ((u64, usize, u64), (usize, u64, u64, u64));
        let table: [Row; 12] = [
            ((20, 1, 32768), (129, 32768, 2318204928, 2318204928)),
            ((20, 8, 262144), (366, 91648, 7372013568, 7372013568)),
            ((20, 4, 1 << 20), (407, 103168, 7444103168, 7444103168)),
            ((40, 1, 262144), (1025, 262144, 17350590464, 17350590464)),
            ((40, 8, 32768), (136, 32768, 3513253888, 3513253888)),
            ((40, 4, 1 << 20), (1687, 430848, 28918939648, 28918939648)),
            ((80, 1, 262144), (1025, 262144, 17350590464, 17350590464)),
            ((80, 8, 262144), (1032, 262144, 18545639424, 18545639424)),
            ((80, 1, 1 << 20), (4097, 1048576, 68890198016, 68890198016)),
            ((100, 4, 262144), (1028, 262144, 17862754304, 17862754304)),
            ((100, 8, 1 << 20), (4104, 1048576, 70085246976, 70085246976)),
            ((100, 4, 32768), (132, 32768, 2830368768, 2830368768)),
        ];
        for ((free_gib, slots, cap), (blocks, tokens, admit, capacity)) in table {
            let (p, mm, got_capacity) = qwen_engine_plan(&c, free_gib << 30, 8 << 30, slots, cap).unwrap();
            let at = format!("free {free_gib} GiB, {slots} slots, cap {cap}");
            assert_eq!(p.kv_blocks, blocks, "{at}");
            assert_eq!(p.block_bytes, 16 << 20, "{at}");
            assert_eq!(p.state_bytes_per_slot, 170_721_280, "{at}: 48 x (48 x 128 x 128 x 4 + 3 x 10240 x 2) + 16 MiB");
            assert_eq!(p.kv_tokens, tokens, "{at}");
            assert_eq!(p.admit_bytes, admit, "{at}");
            assert_eq!(got_capacity, capacity, "{at}");
            // what admission is built with
            assert_eq!(mm.kv_bytes_per_token, 65536, "{at}");
            assert_eq!(mm.state_bytes_per_slot, 170_721_280, "{at}");
            assert_eq!((mm.weights_bytes, mm.activation_reserve), (0, 0), "{at}");
            // the one reserve a Qwen plan takes is the 5 GiB blanket, and it counts no vision memory
            assert_eq!((p.engine_reserve_bytes, p.vision_reserve_bytes), (5 << 30, 0), "{at}");
        }
    }

    #[test]
    fn the_qwen_plan_has_no_room_exactly_where_it_always_had_none() {
        let c = real_qwen();
        // 13 GiB free, minus the 5 GiB scratch reserve, minus the 8 GiB floor: nothing allocatable
        let e = qwen_engine_plan(&c, 13 << 30, 8 << 30, 4, 1 << 20).unwrap_err();
        assert_eq!(e, PlanError::NoRoom { allocatable: 0, needed: 4 * 170_721_280 + (16 << 20) });
        // one byte more than slot state plus one page is enough, one byte less is not
        let needed = 4 * 170_721_280 + (16 << 20);
        let free = (13 << 30) + needed;
        assert!(qwen_engine_plan(&c, free, 8 << 30, 4, 1 << 20).is_ok());
        assert!(qwen_engine_plan(&c, free - 1, 8 << 30, 4, 1 << 20).is_err());
    }

    // ---- GLM: the plan on the box it targets, and the hazard end to end

    /// 121.7 GiB visible less the 91.67 GiB pack leaves about 30 GiB (docs/GLM53.md, the bring-up plan: "headroom after weights is
    /// only about 17 GiB" once the 8 GiB floor and the 5 GiB reserve are out).
    #[test]
    fn the_glm_plan_by_hand_on_the_target_box() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let free = 30 * GIB;
        let vision = glm_profile(false)
            .vision_reserve_bytes(ENCODERS_IN_FLIGHT, VisionOptions::SERVER_DEFAULT.concurrent_images(8))
            .unwrap();
        // the options leave vision on by default: one encoder's scratch (589,824,068) + 32 image buffers of 64 MiB (4096 tokens of f32 features)
        assert_eq!(vision, 589_824_068 + 32 * 67_108_864);
        let p = plan_memory(&glm_profile(false), free, 8 * GIB, 8, 262_144, vision).unwrap();
        assert_eq!(p.block_bytes, 11 * 278_528);
        assert_eq!(p.state_bytes_per_slot, 291_672_200);
        // allocatable = 30 GiB - 5 GiB - 2,737,307,716 - 8 GiB = 15,516,303,292; less 8 slots x 291,672,200 of state it would hold 4,302 pages,
        // so the 262,144-token cap (1,024 pages + 8 slots) governs
        assert_eq!(p.kv_blocks, 1024 + 8);
        assert_eq!(p.kv_tokens, 262_144);
        assert_eq!(p.admit_bytes, 8 * 291_672_200 + 262_144 * 11_968);
        assert_eq!(p.admit_bytes, 5_470_716_992);
        // the MTP head's cache (a 12th layer) is charged for every token: 13,056 B instead of 11,968; and its own per-slot state, the 12th
        // DSA ring (128 KiB) and the store of pending tapped hidden rows (32 fp16 rows of 4,096 = 256 KiB), for every slot
        let m = plan_memory(&glm_profile(true), free, 8 * GIB, 8, 262_144, vision).unwrap();
        assert_eq!(m.block_bytes, 12 * 278_528);
        assert_eq!(m.state_bytes_per_slot, 291_672_200 + 131_072 + 262_144);
        assert_eq!(m.admit_bytes, 8 * (291_672_200 + 131_072 + 262_144) + 262_144 * 13_056);
        assert_eq!(m.kv_tokens, 262_144);
    }

    #[test]
    fn a_glm_box_without_room_fails_closed() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        // only 12 GiB free after the weights: 5 GiB reserve and an 8 GiB floor leave nothing
        let e = plan_memory(&glm_profile(true), 12 * GIB, 8 * GIB, 8, 262_144, 0).unwrap_err();
        assert!(matches!(e, PlanError::NoRoom { allocatable: 0, .. }), "{e}");
        // and eight slots of KDA state alone (8 x 278 MiB = 2.2 GiB) exceed what is left on a 17 GiB box once vision is counted
        let vision = 15 * GIB;
        assert!(matches!(plan_memory(&glm_profile(true), 30 * GIB, 8 * GIB, 8, 262_144, vision), Err(PlanError::NoRoom { .. })));
    }

    #[test]
    fn head_dim_zero_cannot_reach_the_planner_or_admission_for_either_family() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        // the real GLM config has head_dim 0 and plans 17,600 B per token and a non-zero admission
        let v: serde_json::Value = serde_json::from_str(GLM_CONFIG).unwrap();
        assert_eq!(v["text_config"]["head_dim"], 0);
        let profile = glm_profile(false);
        let plan = plan_memory(&profile, 40 * GIB, 8 * GIB, 4, 262_144, 0).unwrap();
        let mut a = Admission::new(plan.admit_bytes, 0, ModelMemory::from_profile(&profile, 0, 0), 4).unwrap();
        assert!(a.cost(1) >= 11_968 + profile.state_bytes_per_slot());
        assert!(a.admit(262_144).is_ok());
        // every token is charged: the planned pool holds exactly one full context, so a second one is refused, not admitted free
        assert!(matches!(a.admit(262_144), Err(gb_sched::AdmitError::NoMemory { .. })));
        // a Qwen config with head_dim 0 never gets a profile, so there is nothing to hand the planner
        let mut q: serde_json::Value = serde_json::from_str(QWEN_CONFIG).unwrap();
        q["text_config"]["head_dim"] = serde_json::json!(0);
        let zero = QwenConfig::from_json(&q).unwrap();
        assert!(matches!(qwen::memory_profile(&zero, &MemoryOptions::serving_defaults()), Err(gb_sched::MemoryError::Zero(_))));
        // and admission refuses a literal zero whatever built it
        let literal = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 0, state_bytes_per_slot: 0, activation_reserve: 0 };
        assert!(Admission::new(plan.admit_bytes, 0, literal, 4).is_err());
    }

    #[test]
    fn plan_error_text_names_the_problem() {
        let e = plan_memory(&qwen_profile(), GIB, 0, 1, 1 << 20, 0).unwrap_err();
        assert!(e.to_string().contains("not enough memory"), "{e}");
        assert!(PlanError::NoSlots.to_string().contains("slot"));
    }

    // ---- the prefix cache's share of the plan

    /// What one snapshot slot costs on the shipped pair (a draft is loaded): the backend's own formula.
    fn snap_bytes() -> u64 {
        crate::snapshot::SnapPool::bytes_per_slot(&real_qwen(), true)
    }

    fn cached(free_gib: u64, slots: usize, cap: u64, req: CacheRequest) -> Result<MemoryPlan, PlanError> {
        plan_memory_cached(&qwen_profile(), free_gib * GIB, 8 * GIB, slots, cap, 0, req, snap_bytes(), 0)
    }

    #[test]
    fn the_cache_off_is_the_plan_the_engine_shipped_with() {
        for (free, slots, cap) in [(20u64, 8usize, 262_144u64), (40, 4, 1 << 20), (80, 8, 262_144), (100, 4, 32_768)] {
            let off = cached(free, slots, cap, CacheRequest::Off).unwrap();
            let old = plan_memory(&qwen_profile(), free * GIB, 8 * GIB, slots, cap, 0).unwrap();
            assert_eq!(off, old);
            assert_eq!(off.cache, CacheBudget::NONE);
        }
    }

    #[test]
    fn auto_never_changes_the_kv_plan_and_spends_at_most_half_of_what_is_spare() {
        let mut enabled = 0;
        for free in [14u64, 16, 20, 24, 30, 40, 60, 80, 100] {
            for (slots, cap) in [(1usize, 32_768u64), (4, 262_144), (8, 262_144), (8, 1 << 20)] {
                let Ok(off) = cached(free, slots, cap, CacheRequest::Off) else { continue };
                let auto = cached(free, slots, cap, CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES }).unwrap();
                let at = format!("free {free} GiB, {slots} slots, cap {cap}");
                // the KV pool, admission and the reserves are exactly the plan without a cache
                assert_eq!(
                    (auto.kv_blocks, auto.kv_tokens, auto.admit_bytes),
                    (off.kv_blocks, off.kv_tokens, off.admit_bytes),
                    "{at}"
                );
                assert_eq!(auto.spare_bytes, off.spare_bytes, "{at}");
                assert!(
                    auto.cache.bytes() <= auto.spare_bytes / 2 && auto.cache.bytes() <= AUTO_CACHE_CAP_BYTES,
                    "{at}: {:?}",
                    auto.cache
                );
                // and the whole plan still fits in what was free
                let total = auto.kv_blocks as u64 * auto.block_bytes
                    + slots as u64 * auto.state_bytes_per_slot
                    + auto.cache.bytes()
                    + auto.engine_reserve_bytes
                    + 8 * GIB;
                assert!(total <= free * GIB, "{at}: {total} > {}", free * GIB);
                enabled += auto.cache.enabled() as usize;
            }
        }
        assert!(enabled > 10, "the grid has boxes that get a cache ({enabled})");
    }

    #[test]
    fn a_box_whose_free_memory_shrank_gets_no_cache_and_keeps_its_kv() {
        // 20 GiB free, 8 slots, the full 262,144-token context asked for: the pool takes everything it can, nothing is spare, no cache
        let p = cached(20, 8, 262_144, CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES }).unwrap();
        assert!(!p.cache.enabled() && p.cache.bytes() == 0, "{:?}", p.cache);
        assert_eq!(p.kv_blocks, cached(20, 8, 262_144, CacheRequest::Off).unwrap().kv_blocks);
    }

    #[test]
    fn a_fixed_cache_is_carved_out_before_the_kv_pool_is_sized() {
        let (free, slots, cap) = (40u64, 4usize, 1u64 << 20);
        let off = cached(free, slots, cap, CacheRequest::Off).unwrap();
        let fixed = cached(free, slots, cap, CacheRequest::Fixed(4 * GIB)).unwrap();
        assert!(fixed.cache.enabled() && fixed.cache.bytes() <= 4 * GIB);
        // the pool is smaller by what the cache took, in whole pages (give or take the page the division rounds)
        let taken = fixed.cache.bytes();
        let dropped = (off.kv_blocks - fixed.kv_blocks) as u64;
        assert!(
            dropped >= taken / off.block_bytes && dropped <= taken / off.block_bytes + 1,
            "{dropped} pages dropped for {taken} bytes"
        );
        let total = fixed.kv_blocks as u64 * fixed.block_bytes
            + slots as u64 * fixed.state_bytes_per_slot
            + taken
            + fixed.engine_reserve_bytes
            + 8 * GIB;
        assert!(total <= free * GIB);
        // a request the box cannot meet fails closed
        assert!(matches!(cached(14, 4, cap, CacheRequest::Fixed(8 * GIB)), Err(PlanError::NoRoom { .. })));
        // so does one that cannot hold two entries
        let e = cached(40, 4, cap, CacheRequest::Fixed(300 << 20)).unwrap_err();
        assert!(matches!(e, PlanError::CacheTooSmall { .. }) && e.to_string().contains("--prefix-cache-mib"), "{e}");
    }

    // ---- the cache sized from the window (Qwen3.8, GLM and Flash-Next opt in; `docs/PREFIX_CACHE_SIZING.md` section 5)

    const GLM_SNAP: u64 = 147_646_464;

    #[test]
    fn the_page_and_snapshot_sizes_behind_the_pins_are_pinned() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        // the arithmetic of `CacheBudget::fit_pages_first` in `gb_sched::prefix::tests` uses these exact figures: when one changes, redo the pins
        let gc = GlmConfig::from_json(&serde_json::from_str(GLM_CONFIG).unwrap()).unwrap();
        assert_eq!(glm_profile(true).kv_page_bytes(), 3_342_336);
        assert_eq!(crate::families::glm::model::snapshot::SnapPool::bytes_per_slot(&gc, true).unwrap(), GLM_SNAP);
        assert_eq!(qwen_profile().kv_page_bytes(), 16_777_216);
        assert_eq!(snap_bytes(), 195_887_104);
    }

    fn windowed(free_gib: u64, slots: usize, cap: u64, req: CacheRequest, window: u64) -> Result<MemoryPlan, PlanError> {
        plan_memory_cached_windowed(&qwen_profile(), free_gib * GIB, 8 * GIB, slots, cap, 0, req, snap_bytes(), 0, window)
    }

    fn glm_windowed(free_gib: u64, cap: u64, req: CacheRequest, window: u64) -> Result<MemoryPlan, PlanError> {
        plan_memory_cached_windowed(&glm_profile(true), free_gib * GIB, 4 * GIB, 1, cap, 0, req, GLM_SNAP, 0, window)
    }

    const AUTO: CacheRequest = CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES };

    #[test]
    fn off_is_planned_identically_by_both_entry_points() {
        for free in [20u64, 40, 80, 110] {
            for slots in [1usize, 4, 8] {
                for cap in [65_536u64, 262_144] {
                    assert_eq!(
                        cached(free, slots, cap, CacheRequest::Off),
                        windowed(free, slots, cap, CacheRequest::Off, 262_144)
                    );
                }
            }
        }
    }

    #[test]
    fn the_legacy_entry_point_cuts_the_cache_as_before() {
        // an explicit 40 GiB in 8-page units, `Auto` capped at 8 GiB and half the spare memory: what MiniCPM and Nemotron still get
        let old = cached(100, 6, 262_144, CacheRequest::Fixed(40 * GIB)).unwrap();
        assert_eq!((old.cache.snapshot_slots, old.cache.page_blocks), (130, 1040));
        let auto = cached(80, 8, 262_144, AUTO).unwrap();
        assert_eq!((auto.cache.snapshot_slots, auto.cache.page_blocks), (26, 208));
        // the window rule is another cut of the same pin
        let new = windowed(100, 6, 262_144, CacheRequest::Fixed(40 * GIB), 262_144).unwrap();
        assert_eq!((new.cache.snapshot_slots, new.cache.page_blocks), (48, 1999));
    }

    #[test]
    fn an_auto_window_cache_is_the_target_where_it_fits_and_never_takes_what_the_kv_pool_had() {
        let (slots, cap, w) = (6usize, 262_144u64, 262_144u64);
        let mut at_target = 0;
        for free in [40u64, 60, 80, 100, 110, 120] {
            let off = windowed(free, slots, cap, CacheRequest::Off, w).unwrap();
            let p = windowed(free, slots, cap, AUTO, w).unwrap();
            // the KV pool, admission and the reserves are exactly the plan without a cache
            assert_eq!(
                (p.kv_blocks, p.kv_tokens, p.admit_bytes, p.spare_bytes),
                (off.kv_blocks, off.kv_tokens, off.admit_bytes, off.spare_bytes)
            );
            let margin = gb_sched::prefix::cache_margin_bytes(p.spare_bytes);
            assert!(p.cache.bytes() <= p.spare_bytes - margin, "free {free}: {:?} spare {}", p.cache, p.spare_bytes);
            if p.spare_bytes - margin >= 34_766_061_568 {
                assert_eq!((p.cache.snapshot_slots, p.cache.page_blocks), (24, 1792), "free {free}: the target");
                at_target += 1;
            }
            if p.cache.enabled() {
                assert_eq!(p.cache.reach_tokens(), p.cache.page_blocks as u64 * 256);
            }
        }
        assert!(at_target >= 2, "the grid has boxes that get the whole target ({at_target})");
        // a box with no spare memory still gets no cache, and its KV pool is untouched
        let none = windowed(20, 8, 262_144, AUTO, w).unwrap();
        assert!(!none.cache.enabled());
        assert_eq!(none.kv_blocks, windowed(20, 8, 262_144, CacheRequest::Off, w).unwrap().kv_blocks);
    }

    #[test]
    fn a_whole_gib_swing_of_the_free_memory_reading_does_not_turn_the_cache_off_or_shrink_the_target() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        // `MemAvailable` crossing a whole-GiB boundary between two boots moves the spare memory by 1 GiB (GLM and RED-SNOW, 16 to 15 GiB): the
        // margin keeps the cache from being sized at the edge of what fits
        let mut identical = 0;
        let mut checked = 0;
        for (qwen, lo, hi, w) in [(true, 40u64, 130u64, 262_144u64), (false, 8u64, 40u64, 262_144u64)] {
            for free in lo..=hi {
                let plan = |f: u64| {
                    if qwen {
                        windowed(f, 6, 262_144, AUTO, w)
                    } else {
                        glm_windowed(f, 262_144, AUTO, w)
                    }
                };
                let (Ok(high), Ok(low)) = (plan(free), plan(free - 1)) else { continue };
                let (fam_page, fam_snap, live) =
                    if qwen { (16_777_216u64, 195_887_104u64, 6usize) } else { (3_342_336, GLM_SNAP, 1) };
                let rule = gb_sched::prefix::WindowRule { window_tokens: w, live_slots: live };
                let target = rule.target_pages() * fam_page + rule.wanted_slots() * fam_snap;
                let floor = rule.full_entry_pages() * fam_page + 2 * fam_snap;
                let budget_low = low.spare_bytes - gb_sched::prefix::cache_margin_bytes(low.spare_bytes);
                if budget_low >= target {
                    assert_eq!(high.cache, low.cache, "qwen {qwen} free {free}: both readings get the target");
                    identical += 1;
                }
                if budget_low >= floor {
                    assert!(high.cache.enabled() && low.cache.enabled(), "qwen {qwen} free {free}: never off at boot");
                    assert!(
                        high.cache.reach_tokens() >= w && low.cache.reach_tokens() >= w,
                        "qwen {qwen} free {free}: reach under the window"
                    );
                }
                checked += 1;
            }
        }
        assert!(identical >= 10 && checked >= 20, "the grid exercises the swing ({identical} identical of {checked})");
    }

    #[test]
    fn an_explicit_pin_is_the_same_cache_at_either_free_memory_reading() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        // the 1,024 MiB RED-SNOW pin: the cache is carved out first, so the reading moves the KV pool and never the cache
        let a = glm_windowed(20, 98_304, CacheRequest::Fixed(1024 << 20), 262_144).unwrap();
        let b = glm_windowed(19, 98_304, CacheRequest::Fixed(1024 << 20), 262_144).unwrap();
        assert_eq!(a.cache, b.cache);
        assert_eq!((a.cache.snapshot_slots, a.cache.page_blocks), (2, 232));
    }

    #[test]
    fn a_window_sized_plan_fits_the_pin_and_equals_what_the_backend_allocates() {
        for pin_gib in [2u64, 8, 17, 24, 40] {
            let (free, slots, cap) = (110u64, 6usize, 262_144u64);
            let p = windowed(free, slots, cap, CacheRequest::Fixed(pin_gib * GIB), 262_144).unwrap();
            let c = p.cache;
            assert!(c.enabled() && c.bytes() <= pin_gib * GIB, "{pin_gib} GiB: {c:?}");
            // the backend allocates `snapshot_slots x snapshot slot` and adds `page_blocks` pages to the pool: the plan's bytes are that sum
            assert_eq!(c.bytes(), c.snapshot_slots as u64 * 195_887_104 + c.page_blocks as u64 * (16 << 20));
            // the cache comes out first: the whole plan still fits in what was free
            let total = p.kv_blocks as u64 * p.block_bytes
                + slots as u64 * p.state_bytes_per_slot
                + c.bytes()
                + p.engine_reserve_bytes
                + 8 * GIB;
            assert!(total <= free * GIB, "{pin_gib} GiB: {total} > {}", free * GIB);
            assert_eq!(c.reach_tokens(), c.page_blocks as u64 * 256);
        }
    }

    #[test]
    fn a_window_sized_pin_under_the_minimum_names_the_size_to_give() {
        let e = windowed(60, 4, 1 << 20, CacheRequest::Fixed(300 << 20), 262_144).unwrap_err();
        let min = gb_sched::prefix::CacheBudget::pages_first_minimum_bytes(195_887_104, 16 << 20);
        assert!(matches!(e, PlanError::CacheBelowMinimumReach { minimum_bytes, .. } if minimum_bytes == min), "{e:?}");
        let text = e.to_string();
        assert!(
            text.contains("--prefix-cache-mib") && text.contains(&format!("at least {} MiB", min.div_ceil(1 << 20))),
            "{text}"
        );
        // the size it names is enough
        assert!(windowed(60, 4, 1 << 20, CacheRequest::Fixed(min), 262_144).unwrap().cache.enabled());
    }

    #[test]
    fn only_the_release_families_size_the_cache_from_the_window() {
        // the opt-in boundary: Qwen3.8, GLM (one node and each rank of a TP2 group, `docs/GLM_TP.md` 4.2), Flash-Next and Ling-3.1 (bailing_moe_v3,
        // whose cache pages are the host budget of its snapshots' MLA copies) call `plan_memory_cached_windowed`; MiniCPM and Nemotron keep their own
        // cut, and DeepSeek has no code for it. A family that picks the new rule up by accident fails here by name.
        fn code(s: &str) -> &str {
            s.split("#[cfg(test)]").next().unwrap()
        }
        for (name, src, opted_in) in [
            ("serve.rs (Qwen3.8)", include_str!("serve.rs"), true),
            ("glm/engine.rs", include_str!("families/glm/engine.rs"), true),
            ("qwen_flash_next/engine.rs", include_str!("families/qwen_flash_next/engine.rs"), true),
            ("glm/tp_boot.rs", include_str!("families/glm/tp_boot.rs"), true),
            ("bailing_moe_v3/engine.rs", include_str!("families/bailing_moe_v3/engine.rs"), true),
            ("minicpmv4_7/boot.rs", include_str!("families/minicpmv4_7/boot.rs"), false),
            ("nemotron_h_omni/engine.rs", include_str!("families/nemotron_h_omni/engine.rs"), false),
            ("nemotron_h_omni/tp_boot.rs", include_str!("families/nemotron_h_omni/tp_boot.rs"), false),
            ("deepseek_v41/engine.rs", include_str!("families/deepseek_v41/engine.rs"), false),
            ("deepseek_v41/memory.rs", include_str!("families/deepseek_v41/memory.rs"), false),
        ] {
            assert_eq!(code(src).contains("plan_memory_cached_windowed"), opted_in, "{name}");
        }
        // and the legacy entry point is what the others call
        assert!(include_str!("families/minicpmv4_7/boot.rs").contains("plan_memory_cached("));
    }

    #[test]
    fn the_cache_plan_equals_what_the_backend_allocates() {
        // the plan's bytes are the formula `snapshot_slots x snapshot slot + page_blocks x page`; `Model::enable_prefix_cache` allocates the
        // snapshot slots from the same `SnapPool::bytes_per_slot` and `build_engine` adds `page_blocks` to the pool's pages (the GPU test
        // `the_snapshot_pool_is_what_the_plan_says` checks the device's side of it)
        let p = cached(80, 8, 262_144, CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES }).unwrap();
        assert!(p.cache.enabled());
        assert_eq!(p.cache.snapshot_bytes, 195_887_104);
        assert_eq!(p.cache.page_bytes, 16 << 20);
        assert_eq!(p.cache.bytes(), p.cache.snapshot_slots as u64 * 195_887_104 + p.cache.page_blocks as u64 * (16 << 20));
        assert_eq!(p.cache.page_blocks, 8 * p.cache.snapshot_slots);
        // at 80 GiB free the cap (8 GiB) governs: 8 GiB / 330,104,832 bytes an entry is 26 entries
        assert_eq!(p.cache.snapshot_slots, 26);
        // without a draft a slot is the recurrent state alone
        let q = plan_memory_cached(
            &qwen_profile(),
            80 * GIB,
            8 * GIB,
            8,
            262_144,
            0,
            CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES },
            crate::snapshot::SnapPool::bytes_per_slot(&real_qwen(), false),
            0,
        )
        .unwrap();
        assert_eq!(q.cache.snapshot_bytes, 153_944_064);
        assert!(q.cache.snapshot_slots > p.cache.snapshot_slots);
    }

    // ---- nothing is planned that the box was not given, and the boot history is not an input

    #[test]
    fn the_admission_a_plan_builds_declares_the_pool_it_planned() {
        // `MemoryPlan::admission` must carry the planned KV pool in tokens (`.with_kv_tokens`): a plan built
        // admission without it reports only the context window, so a prompt past the pool is admitted and dies
        // late with `OutOfPages` (the hole the serve-hardening work closed at the source). Qwen's
        // planner uses pure CPU sizes here. The GLM variant is a separate explicit native-oracle
        // test: its large-row workspace sizing is unavailable in the forced no-CUDA profile.
        let floor = 8 * GIB;
        // Qwen at 20 GiB free: the pool sits far below the window (71-91k tokens measured on the box).
        let p = plan_memory(&qwen_profile(), 20 * GIB, floor, 8, 262_144, 0).unwrap();
        assert!(p.kv_tokens < 262_144, "qwen: pool {} not below the window", p.kv_tokens);
        let mm = ModelMemory::from_profile(&qwen_profile(), 0, 0);
        let a = p.admission(mm, 8, 262_144).unwrap();
        assert_eq!(a.kv_tokens(), Some(p.kv_tokens), "qwen: admission does not declare the planned pool");
        assert_eq!(a.request_limit(), Some(p.kv_tokens), "qwen: limit is not min(window, pool)");
    }

    #[test]
    #[ignore = "native CUDA workspace-size oracle required; pure planner check, no GLM weight load"]
    fn the_glm_admission_plan_declares_its_native_workspace_pool() {
        assert!(gb_cuda::rt::kernels_built(), "native workspace-size oracle must be compiled, not a CPU stub");
        let floor = 8 * GIB;
        // GLM at 16 GiB free: the pool is below the window (the pinned `--slots 4` boot of the handoff measured
        // 35,768 tokens against a 262,144 window).
        let g = plan_memory(&glm_profile(true), 16 * GIB, floor, 4, 262_144, 1_000_000_000).unwrap();
        assert!(g.kv_tokens < 262_144, "glm: pool {} not below the window", g.kv_tokens);
        let gmm = ModelMemory::from_profile(&glm_profile(true), 0, 0);
        let ga = g.admission(gmm, 4, 262_144).unwrap();
        assert_eq!(ga.kv_tokens(), Some(g.kv_tokens), "glm: admission does not declare the planned pool");
        assert_eq!(ga.request_limit(), Some(g.kv_tokens), "glm: limit is not min(window, pool)");
    }

    #[test]
    fn a_plan_never_asks_for_more_than_it_was_given() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        // what the engine allocates beyond the weights (the reserves, the slots' state, the pool's pages, the cache) is at most `free - floor`, for both
        // families, every free size, slot count, context cap and cache request: the plan cannot over-commit the pool it was sized from
        let floor = 8 * GIB;
        let mut plans = 0;
        for free_gib in [14u64, 17, 20, 24, 30, 40, 60, 80, 113] {
            let free = free_gib * GIB;
            for slots in [1usize, 2, 4, 8, 32] {
                for cap in [32_768u64, 262_144, 1 << 20] {
                    for (profile, vision) in [(qwen_profile(), 0u64), (glm_profile(true), 1_000_000_000)] {
                        for req in [
                            CacheRequest::Off,
                            CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES },
                            CacheRequest::Fixed(2 * GIB),
                        ] {
                            let Ok(p) = plan_memory_cached(&profile, free, floor, slots, cap, vision, req, snap_bytes(), 0)
                            else {
                                continue;
                            };
                            let at = format!("{} at free {free_gib} GiB, {slots} slots, cap {cap}, {req:?}", profile.family());
                            assert!(
                                p.planned_bytes(slots) + floor <= free,
                                "{at}: {} + {floor} > {free}",
                                p.planned_bytes(slots)
                            );
                            // the sum is the six terms and no other
                            assert_eq!(
                                p.planned_bytes(slots),
                                p.engine_reserve_bytes
                                    + p.vision_reserve_bytes
                                    + slots as u64 * p.state_bytes_per_slot
                                    + p.kv_blocks as u64 * p.block_bytes
                                    + p.cache.bytes()
                                    + p.wq_budget_bytes,
                                "{at}"
                            );
                            plans += 1;
                        }
                    }
                }
            }
        }
        assert!(plans > 200, "the grid plans on most boxes ({plans})");
    }

    /// A slot count is viable when its plan holds `min(cap, slots x MIN_TOKENS_PER_SLOT)` tokens.
    fn viable(p: &Result<MemoryPlan, PlanError>, slots: usize, cap: u64) -> bool {
        p.as_ref().is_ok_and(|p| p.kv_tokens >= (slots as u64 * MIN_TOKENS_PER_SLOT).min(cap))
    }

    #[test]
    fn the_default_slot_count_is_the_most_that_fit_and_an_explicit_one_is_never_changed() {
        let profile = qwen_profile();
        let cap = 262_144;
        let (mut shrunk, mut kept) = (0, 0);
        for free_mib in (13_000u64..=40_000).step_by(1_250) {
            let at = |s: usize| plan_memory(&profile, free_mib * (1 << 20), 8 * GIB, s, cap, 0);
            let requested = 8;
            let pinned = choose_slots(requested, false, cap, at);
            // an explicit count is exactly its own plan or its own refusal
            match &pinned {
                Ok(c) => assert_eq!((c.slots, &c.plan), (requested, &at(requested).unwrap()), "free {free_mib} MiB"),
                Err(e) => assert_eq!(e, &at(requested).unwrap_err(), "free {free_mib} MiB"),
            }
            let Ok(auto) = choose_slots(requested, true, cap, at) else {
                // nothing viable and the request has no plan either: the refusal is the request's
                assert!(pinned.is_err() && (1..=requested).all(|s| !viable(&at(s), s, cap)), "free {free_mib} MiB");
                continue;
            };
            assert_eq!(auto.requested, requested);
            if viable(&at(requested), requested, cap) {
                // the box holds the request: nothing changes
                assert_eq!(auto, pinned.unwrap(), "free {free_mib} MiB");
                kept += 1;
            } else if auto.shrunk() {
                // the most that fit, and viable
                assert!(viable(&at(auto.slots), auto.slots, cap), "free {free_mib} MiB: {auto:?}");
                assert!(
                    (auto.slots + 1..=requested).all(|s| !viable(&at(s), s, cap)),
                    "free {free_mib} MiB: a larger count is viable"
                );
                assert_eq!(auto.plan, at(auto.slots).unwrap());
                shrunk += 1;
            } else {
                // nothing viable but the request plans (a thin pool, as it always was): it is kept
                assert_eq!(auto.slots, requested);
                assert!((1..=requested).all(|s| !viable(&at(s), s, cap)), "free {free_mib} MiB");
            }
        }
        assert!(shrunk >= 3 && kept >= 3, "the grid has boxes on both sides of the frontier (shrunk {shrunk}, kept {kept})");
    }

    #[test]
    fn capacity_is_the_same_whatever_the_node_did_before_the_boot() {
        use crate::hostmem::{planning_free, MemInfo};
        // one box in two histories: after a cold boot MemFree is what is available; after 80 GiB of reads the cache holds 80 GiB of it
        let after_weights = 94 * GIB + GIB / 3;
        let cold = MemInfo { total: 121 * GIB, free: after_weights, available: after_weights + GIB / 5, cached: GIB };
        let dirty = MemInfo { total: 121 * GIB, free: 12 * GIB, available: after_weights, cached: 82 * GIB };
        let plan = |m: &MemInfo, req| {
            plan_memory_cached(&qwen_profile(), planning_free(m.free, Some(m)), 8 * GIB, 8, 262_144, 0, req, snap_bytes(), 0)
                .unwrap()
        };
        for req in [CacheRequest::Off, CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES }] {
            let (a, b) = (plan(&cold, req), plan(&dirty, req));
            assert_eq!(a, b, "{req:?}: the KV pool and the cache budget do not depend on the history");
            assert_eq!(
                (a.kv_tokens, a.cache.bytes() > 7 * GIB),
                (262_144, req != CacheRequest::Off),
                "the full context and the 8 GiB cache (cap {:?})",
                a.cache
            );
        }
        // what the old rule (the smaller of the device's free and MemAvailable) made of the same dirty box
        let old = plan_memory_cached(
            &qwen_profile(),
            dirty.free.min(dirty.available),
            8 * GIB,
            8,
            262_144,
            0,
            CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES },
            snap_bytes(),
            0,
        );
        assert!(old.as_ref().map_or(true, |p| p.kv_tokens < 262_144 && !p.cache.enabled()), "{old:?}");
    }

    // ---- the W_q residency budget: priced, fail-closed --------------------------------------------------------------------------------

    /// The 400 prefill matrices of the target pack are 45.31 GiB of fp16: the full residency the ceiling page prices.
    #[test]
    fn the_wq_budget_is_priced_and_fails_closed() {
        let p_off =
            plan_memory_cached(&qwen_profile(), 80 * GIB, 8 * GIB, 8, 262_144, 0, CacheRequest::Off, snap_bytes(), 0).unwrap();
        assert_eq!(p_off.wq_budget_bytes, 0, "off by default: no budget, no bytes");
        let p_on = plan_memory_cached(
            &qwen_profile(),
            80 * GIB,
            8 * GIB,
            8,
            262_144,
            0,
            CacheRequest::Off,
            snap_bytes(),
            45_312 * GIB / 1000,
        )
        .unwrap();
        assert_eq!(p_on.wq_budget_bytes, 45_312 * GIB / 1000);
        // the budget comes out of the allocatable bytes: here the pool sits at the token cap, so the spare absorbs
        // it, exactly
        assert_eq!(
            p_off.spare_bytes - p_on.spare_bytes,
            p_on.wq_budget_bytes,
            "the spare lost {} for a budget of {}",
            p_off.spare_bytes - p_on.spare_bytes,
            p_on.wq_budget_bytes
        );
        // and it is part of what the plan says it allocates
        assert_eq!(p_on.planned_bytes(8) - p_off.planned_bytes(8), p_on.wq_budget_bytes);
        // with no cap the pool itself shrinks by the budget (rounded to whole pages, the remainder spare)
        let u = plan_memory_cached(&qwen_profile(), 80 * GIB, 8 * GIB, 8, u64::MAX / 2, 0, CacheRequest::Off, snap_bytes(), 0)
            .unwrap();
        let v = plan_memory_cached(
            &qwen_profile(),
            80 * GIB,
            8 * GIB,
            8,
            u64::MAX / 2,
            0,
            CacheRequest::Off,
            snap_bytes(),
            45_312 * GIB / 1000,
        )
        .unwrap();
        let lost = u.kv_blocks - v.kv_blocks;
        assert!(
            lost as u64 * u.block_bytes >= v.wq_budget_bytes && lost as u64 * u.block_bytes < v.wq_budget_bytes + u.block_bytes,
            "the pool lost {lost} pages ({}) for a budget of {}",
            lost as u64 * u.block_bytes,
            v.wq_budget_bytes
        );
        // fail-closed: a box that cannot hold the budget AND the slot state and one page is refused, not shrunk
        let e = plan_memory_cached(
            &qwen_profile(),
            20 * GIB,
            8 * GIB,
            8,
            262_144,
            0,
            CacheRequest::Off,
            snap_bytes(),
            45_312 * GIB / 1000,
        )
        .unwrap_err();
        assert!(matches!(e, PlanError::NoRoom { .. }), "{e}");
        // a full-pool budget on the box that has it (the 121 GiB GB10 after the weights leaves ~93 GiB free)
        let full = plan_memory_cached(
            &qwen_profile(),
            93 * GIB,
            8 * GIB,
            8,
            262_144,
            0,
            CacheRequest::Off,
            snap_bytes(),
            45_312 * GIB / 1000,
        )
        .unwrap();
        assert!(full.kv_tokens > 0, "a budget the box holds is granted with a pool beside it");
    }

    /// The knob's environment parse: `REXL3_WQ_RESIDENT_GIB` is a number of GiB or a boot error, never a silent default.
    #[test]
    fn the_wq_knob_parses_gib_or_refuses() {
        use crate::wq::WQ_RESIDENT_ENV;
        let was = std::env::var(WQ_RESIDENT_ENV).ok();
        std::env::set_var(WQ_RESIDENT_ENV, "45.31");
        assert_eq!(crate::wq::budget_bytes(None).unwrap(), Some(48_651_242_045), "45.31 GiB is the pack's 400 matrices");
        std::env::set_var(WQ_RESIDENT_ENV, "0");
        assert_eq!(crate::wq::budget_bytes(None).unwrap(), None, "zero is off");
        std::env::remove_var(WQ_RESIDENT_ENV);
        assert_eq!(crate::wq::budget_bytes(None).unwrap(), None, "unset is off");
        std::env::set_var(WQ_RESIDENT_ENV, "junk");
        assert!(crate::wq::budget_bytes(None).is_err());
        match was {
            Some(v) => std::env::set_var(WQ_RESIDENT_ENV, v),
            None => std::env::remove_var(WQ_RESIDENT_ENV),
        }
    }

    // ---- Q-ADM: the Qwen plan counts the features of the images in flight ---------------------------------------------------------

    #[test]
    fn the_qwen_plan_counts_the_features_of_the_images_in_flight_and_moves_a_memory_bound_pool_by_exactly_that() {
        let profile = qwen_profile();
        let vision =
            |slots: usize| qwen::counted_vision_reserve(&profile, MemoryOptions::serving_defaults().vision, true, slots).unwrap();
        // 4 images a slot of 40 MiB each: 10 pages a slot, whole pages
        assert_eq!(
            (vision(1), vision(4), vision(6), vision(8)),
            (10 * (16 << 20), 40 * (16 << 20), 60 * (16 << 20), 80 * (16 << 20))
        );
        let (mut bound, mut capped) = (0, 0);
        for free_gib in [20u64, 24, 30, 40, 60, 80, 100] {
            for slots in [1usize, 4, 6, 8] {
                let Ok(before) = plan_memory(&profile, free_gib * GIB, 8 * GIB, slots, 262_144, 0) else { continue };
                let at = format!("free {free_gib} GiB, {slots} slots");
                let Ok(after) = plan_memory(&profile, free_gib * GIB, 8 * GIB, slots, 262_144, vision(slots)) else {
                    continue; // a box that just fit before: refused now, in the planner's words, not at the allocation
                };
                assert_eq!(after.vision_reserve_bytes, vision(slots), "{at}");
                // the reserve is part of what the plan allocates, so the sum of the plan stays inside what the box was given
                assert!(after.planned_bytes(slots) + 8 * GIB <= free_gib * GIB, "{at}");
                let pages = (before.kv_blocks - after.kv_blocks) as u64;
                if before.kv_tokens < 262_144 {
                    // memory binds the pool: it gives up exactly the pages the features take
                    assert_eq!(pages * after.block_bytes, vision(slots), "{at}");
                    assert_eq!(before.kv_tokens - after.kv_tokens, pages * 256, "{at}");
                    bound += 1;
                } else {
                    // the 262,144-token cap binds: the pool is the cap while the box has room for the features beside it
                    assert!(pages * after.block_bytes <= vision(slots), "{at}");
                    capped += (after.kv_tokens == before.kv_tokens) as usize;
                }
            }
        }
        assert!(bound >= 4 && capped >= 12, "the grid has both kinds of box (memory-bound {bound}, cap-bound {capped})");
        // the gate's server (6 slots, 93 GiB after the weights): nothing moves, the pool, the 8 GiB cache and its entries are the same
        let cache = CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES };
        let (a, b) = (
            plan_memory_cached(&profile, 93 * GIB, 8 * GIB, 6, 262_144, 0, cache, snap_bytes(), 0).unwrap(),
            plan_memory_cached(&profile, 93 * GIB, 8 * GIB, 6, 262_144, vision(6), cache, snap_bytes(), 0).unwrap(),
        );
        assert_eq!((a.kv_blocks, a.kv_tokens, a.admit_bytes, a.cache), (b.kv_blocks, b.kv_tokens, b.admit_bytes, b.cache));
        assert_eq!(b.cache.snapshot_slots, 26);
    }

    // ---- admission follows the pool (serve hardening, finding 1) ---------------------------------------------------------------
    //
    // A 197,048-token prompt into a 194,048-token GLM pool was admitted and failed `OutOfPages` after 298.7 s of prefill. The plan's `admit_bytes`
    // is the pool's KV bytes plus the state of EVERY slot, and admission charges each sequence its own slot's state, so with fewer sequences
    // running the idle slots' state read as room for KV the pool does not have. These tests drive the plans of both families through the
    // admission the engines build (`MemoryPlan::admission`) and the real page tables (`KvTables`, what both backends allocate pages from).

    use crate::kvtables::KvTables;
    use gb_sched::AdmitError;

    const WINDOW: usize = 262_144;

    /// (name, profile, slots, plan) over a grid of boxes: some where the cap binds the pool at the window, some where memory binds it.
    fn pool_cases() -> Vec<(String, MemoryProfile, usize, MemoryPlan)> {
        let mut v = Vec::new();
        for (name, profile) in [("qwen", qwen_profile()), ("glm", glm_profile(true))] {
            // (17 to 20 GiB is where a GLM pack's box lands after the weights: its pool is memory-bound there, as the reported 194,048 was)
            for free_gib in [17u64, 18, 20, 24, 30, 40, 100] {
                for slots in [1usize, 2, 4, 8] {
                    let cache = CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES };
                    if let Ok(plan) =
                        plan_memory_cached(&profile, free_gib * GIB, 8 * GIB, slots, WINDOW as u64, 0, cache, snap_bytes(), 0)
                    {
                        v.push((format!("{name}, {free_gib} GiB free, {slots} slots"), profile.clone(), slots, plan));
                    }
                }
            }
        }
        v
    }

    /// What `serve::build_engine` and the GLM engine built before this change: the byte capacity and the window, no pool.
    fn old_admission(profile: &MemoryProfile, plan: &MemoryPlan, slots: usize) -> Admission {
        Admission::new(plan.admit_bytes, 0, ModelMemory::from_profile(profile, 0, 0), slots)
            .unwrap()
            .with_max_context(WINDOW as u64)
    }

    /// A small deterministic generator (the test needs a workload, not randomness worth a dependency).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, n: u64) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) % n
        }
    }

    /// Drive `admission` with a mixed workload against the pool's real page tables: submit a request of a random size (some past the pool, some a
    /// large part of it, some small), and when it is admitted take every page it can ever use (`ensure(slot, tokens)`, the worst case of a
    /// sequence that runs to its limit), release a random running one now and then. `cache_pages` pages are held by a stand-in for the prefix
    /// cache (a table of its own) for the whole run. Returns (requests admitted, requests whose pages could not be had, requests refused as
    /// too long).
    fn drive(mut admission: Admission, plan: &MemoryPlan, slots: usize, cache_pages: usize, seed: u64) -> (usize, usize, usize) {
        let max_blocks = WINDOW.div_ceil(256);
        // the pool the backend allocates: the plan's pages plus the cache's, the cache's holder being one extra table
        let mut kv = KvTables::new(plan.kv_blocks + cache_pages, slots + 1, max_blocks);
        if cache_pages > 0 {
            kv.ensure(slots, cache_pages * 256).expect("the cache's own pages are in the pool");
        }
        let mut rng = Lcg(seed);
        let mut running: Vec<usize> = Vec::new();
        let (mut admitted, mut out_of_pages, mut too_long) = (0, 0, 0);
        for _ in 0..400 {
            let pool = plan.kv_tokens;
            let tokens = match rng.next(4) {
                0 => 1 + rng.next(300),
                1 => 1 + rng.next(pool / 4 + 1),
                2 => 1 + rng.next(pool),
                _ => pool / 2 + rng.next(pool / 2 + 4096), // up to a little past the pool
            };
            match admission.admit(tokens) {
                Ok(slot) => {
                    admitted += 1;
                    if kv.ensure(slot, tokens as usize).is_err() {
                        out_of_pages += 1;
                    }
                    running.push(slot);
                }
                Err(AdmitError::ContextTooLong { .. }) => too_long += 1,
                Err(_) => {} // NoSlot or NoMemory: the request waits
            }
            // finish a running request (always when every slot is taken, else now and then)
            if !running.is_empty() && (rng.next(3) == 0 || running.len() >= slots) {
                let at = rng.next(running.len() as u64) as usize;
                let slot = running.swap_remove(at);
                kv.release(slot);
                admission.release(slot);
            }
        }
        (admitted, out_of_pages, too_long)
    }

    #[test]
    fn the_admission_a_plan_builds_limits_a_request_to_the_pool_or_the_window_whichever_is_smaller() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        for (at, profile, slots, plan) in pool_cases() {
            let mm = ModelMemory::from_profile(&profile, 0, 0);
            for window in [4096usize, plan.kv_tokens as usize, WINDOW, 1 << 20] {
                let a = plan.admission(mm.clone(), slots, window).unwrap();
                assert_eq!(a.request_limit(), Some((window as u64).min(plan.kv_tokens)), "{at}, window {window}");
                assert_eq!(a.kv_tokens(), Some(plan.kv_tokens), "{at}");
                assert_eq!(a.capacity(), plan.admit_bytes, "{at}: the byte capacity is the plan's, as before");
            }
        }
    }

    #[test]
    fn a_prompt_past_the_pool_is_refused_at_once_for_both_families_where_the_old_rule_admitted_it() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let mut shown = [0usize; 2];
        for (at, profile, slots, plan) in pool_cases() {
            // the pool is below the window: memory binds it (at the cap there is no room above the pool to ask for)
            if plan.kv_tokens as usize >= WINDOW || slots < 2 {
                continue;
            }
            let mm = ModelMemory::from_profile(&profile, 0, 0);
            let new = plan.admission(mm, slots, WINDOW).unwrap();
            let old = old_admission(&profile, &plan, slots);
            // the slack the old rule left: the state of the idle slots, in tokens (a sequence is charged one slot's state, the plan paid for all)
            let slack = (slots as u64 - 1) * plan.state_bytes_per_slot / profile.kv_bytes_per_token();
            assert!(slack > 0, "{at}");
            // (the window still refuses what is past it, so the pool-and-window gap bounds what the old rule could admit)
            let gap = WINDOW as u64 - plan.kv_tokens;
            for over in [1, 3_000.min(slack).min(gap), slack.min(gap)] {
                let tokens = plan.kv_tokens + over;
                assert_eq!(old.check(tokens), Ok(()), "{at}: the old rule admits {over} tokens past the pool (the bug)");
                assert_eq!(
                    new.check(tokens),
                    Err(AdmitError::ContextTooLong { tokens, max: plan.kv_tokens }),
                    "{at}: {over} tokens past the pool"
                );
            }
            assert_eq!(new.check(plan.kv_tokens), Ok(()), "{at}: the pool's own size is admitted");
            shown[at.starts_with("glm") as usize] += 1;
        }
        assert!(shown[0] >= 3 && shown[1] >= 3, "the grid has boxes where memory binds the pool for each family ({shown:?})");
    }

    #[test]
    fn requests_that_fit_the_pool_never_run_out_of_pages_and_the_old_rule_did() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let (mut old_failures, mut new_admitted, mut new_too_long, mut runs) = (0, 0, 0, 0);
        for (at, profile, slots, plan) in pool_cases() {
            let mm = ModelMemory::from_profile(&profile, 0, 0);
            for seed in 0..6 {
                // the prefix cache holds its whole quota of pages for the entire run: its pages are on top of the pool, not part of it
                for cache_pages in [0, plan.cache.page_blocks] {
                    let new = plan.admission(mm.clone(), slots, WINDOW).unwrap();
                    let (admitted, out_of_pages, too_long) = drive(new, &plan, slots, cache_pages, seed);
                    assert_eq!(
                        out_of_pages, 0,
                        "{at}, seed {seed}, cache pages {cache_pages}: a request that was admitted ran out of pages"
                    );
                    new_admitted += admitted;
                    new_too_long += too_long;
                    let (_, failures, _) = drive(old_admission(&profile, &plan, slots), &plan, slots, cache_pages, seed);
                    old_failures += failures;
                    runs += 1;
                }
            }
        }
        // the driver is not vacuous: the same workload through the old construction runs out of pages, and the new one serves a lot of it
        assert!(old_failures > 0, "the old admission never ran out of pages in {runs} runs: the workload proves nothing");
        assert!(new_admitted > runs * 20, "admitted {new_admitted} in {runs} runs");
        assert!(new_too_long > 0, "the workload asks for more than the pool sometimes");
    }

    #[test]
    fn a_solo_request_of_the_pools_own_size_runs_whatever_the_slot_count() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        // the plan's `kv_blocks` has a page per slot beyond the pool for the rounding of each sequence's last page: a request of exactly
        // `kv_tokens` finds its pages with every other slot idle, and the cache's pages are not touched
        for (at, profile, slots, plan) in pool_cases() {
            let mm = ModelMemory::from_profile(&profile, 0, 0);
            let mut a = plan.admission(mm, slots, WINDOW).unwrap();
            let mut kv = KvTables::new(plan.kv_blocks + plan.cache.page_blocks, slots + 1, WINDOW.div_ceil(256));
            if plan.cache.page_blocks > 0 {
                kv.ensure(slots, plan.cache.page_blocks * 256).unwrap();
            }
            let want = plan.kv_tokens.min(WINDOW as u64);
            let slot = a.admit(want).unwrap_or_else(|e| panic!("{at}: {e:?}"));
            assert_eq!(kv.ensure(slot, want as usize), Ok(()), "{at}");
        }
    }
}

/// The planner tests as nemotron/main has them, on the Nemotron 3.5 Super VL profile (the Nemotron-only tree retargeted them from the Qwen
/// and GLM profiles, and lane G3 pinned the engine reserve by part under P1 and P2); `tests` above holds the same rules on the Qwen and GLM
/// profiles, as main has them. Both run.
#[cfg(test)]
mod nemotron_tests {
    use super::*;
    use crate::families::nemotron_h_omni::config::OmniConfig;
    use crate::families::nemotron_h_omni::memory::NemotronMemory;
    use crate::family::MemorySource;
    use crate::memory::MemoryOptions;
    use gb_sched::{Admission, KvLayerClass, KvLayout, MemoryModel, ModelMemory, StepShape, VisionMemory};

    // The planner's own arithmetic, driven by the Nemotron profile since the Qwen and GLM families left the tree (their pinned
    // plans left with them).
    const NEMO_CONFIG: &str = include_str!("../../../testdata/nemotron_omni/config.json");
    const GIB: u64 = 1 << 30;

    fn nemo_profile(mtp: bool) -> MemoryProfile {
        let c = OmniConfig::from_json(&serde_json::from_str(NEMO_CONFIG).unwrap()).unwrap();
        NemotronMemory::new(&c, 0).profile(&MemoryOptions { mtp, ..MemoryOptions::serving_defaults() }).unwrap()
    }

    // ---- planner arithmetic

    #[test]
    fn slot_state_comes_out_before_kv() {
        let profile = nemo_profile(false);
        let p = plan_memory(&profile, 60 * GIB, 6 * GIB, 8, u64::MAX / 2, 0).unwrap();
        let slot = profile.state_bytes_per_slot() * 8;
        // admission is never promised more than was planned, and the pool holds the promised tokens plus a page per slot
        assert!(p.admit_bytes + 6 * GIB + p.engine_reserve_bytes <= 60 * GIB);
        assert!(p.kv_blocks as u64 * p.block_bytes + slot + 6 * GIB + p.engine_reserve_bytes <= 60 * GIB);
        assert!(p.kv_blocks * BLOCK_TOKENS >= p.kv_tokens as usize + 8 * BLOCK_TOKENS);
        // a request admitted at the planned token budget can always get its pages: sum(ceil(T/256)) <= tokens/256 + slots
        assert!(p.kv_tokens / BLOCK_TOKENS as u64 + 8 <= p.kv_blocks as u64);
    }

    #[test]
    fn the_token_cap_limits_the_pool() {
        let p = plan_memory(&nemo_profile(false), 100 * GIB, 6 * GIB, 4, 262144, 0).unwrap();
        assert_eq!(p.kv_blocks, 262144 / 256 + 4);
        assert_eq!(p.kv_tokens, 262144);
    }

    #[test]
    fn no_room_is_an_error_not_a_zero_pool() {
        let profile = nemo_profile(false);
        assert!(matches!(plan_memory(&profile, 6 * GIB, 6 * GIB, 4, 1 << 20, 0), Err(PlanError::NoRoom { .. })));
        assert!(plan_memory(&profile, 12 * GIB, 6 * GIB, 8, 1 << 20, 0).is_err(), "8 slots of state do not fit in 1 GiB");
    }

    #[test]
    fn the_planner_refuses_what_it_cannot_plan() {
        let profile = nemo_profile(false);
        assert_eq!(plan_memory(&profile, 100 * GIB, 6 * GIB, 0, 1 << 20, 0), Err(PlanError::NoSlots));
        // a profile built for another page size than the pool's
        struct OtherPage;
        impl MemoryModel for OtherPage {
            fn family(&self) -> &str {
                "other-page"
            }
            fn kv_layout(&self) -> KvLayout {
                KvLayout::new().with(KvLayerClass::gqa(16, 4, 256, 2))
            }
            fn recurrent_layers(&self) -> u64 {
                0
            }
            fn state_bytes_per_seq(&self) -> u64 {
                0
            }
            fn window_record_bytes_per_row(&self) -> u64 {
                0
            }
            fn scratch_bytes(&self, rows: u64) -> u64 {
                rows
            }
            fn activation_reserve(&self) -> u64 {
                1
            }
            fn drafter_state_bytes_per_slot(&self) -> u64 {
                0
            }
            fn vision(&self) -> Option<VisionMemory> {
                None
            }
        }
        let odd = MemoryProfile::new(&OtherPage, StepShape { page_tokens: 128, window_rows: 8, scratch_rows: 8 }).unwrap();
        assert_eq!(
            plan_memory(&odd, 100 * GIB, 6 * GIB, 4, 1 << 20, 0),
            Err(PlanError::PageMismatch { profile: 128, pool: 256 })
        );
    }

    #[test]
    fn each_holdback_comes_out_of_the_allocatable_bytes_exactly_once() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let profile = nemo_profile(false);
        let (floor, vision) = (8 * GIB, 1_000_000_000);
        let free = 40 * GIB;
        let p = plan_memory(&profile, free, floor, 8, 1 << 30, vision).unwrap();
        // the engine reserve and the vision reserve are recorded, and with the floor and the slot state they leave exactly the pool
        // the page count came from. The reserve is max(the 5 GiB blanket, scratch + window records) (memory.rs). At
        // MemoryOptions::serving_defaults (4,096 prefill rows, 262,144 positions) the modelled bytes are above the blanket, so it is
        // the modelled bytes, by part (computed by the profile, lane G3 2026-10-15): activations 3,235,905,536 (4,096 rows), top-k
        // workspace 268,435,456, decode MoE scratch 106,724,528, large-row MoE prefill scratch 1,678,665,744, attention partials
        // 51,118,600, EXL3 workspace 33,554,432 (P1; under P2 67,108,864: the tensor-core linear's fp16 input of the 4,096-row slice,
        // `contract::workspace_bytes`), window records 160,563,200, and under P2 the tensor-core MoE block's scratch 792,403,984
        // (`MmaScratch::bytes(4096)`): P1 5,534,967,496, P2 6,360,925,912 (+ 792,403,984 + 33,554,432)
        let reserve = match crate::families::nemotron_h_omni::contract::current() {
            crate::families::nemotron_h_omni::contract::PrefillContract::P1 => 5_534_967_496,
            crate::families::nemotron_h_omni::contract::PrefillContract::P2 => 6_360_925_912,
        };
        assert_eq!(profile.scratch_bytes() + profile.window_record_bytes(), reserve, "the modelled reserve, by part as above");
        assert!(reserve > 5 * GIB, "above the blanket at 4,096 rows");
        assert_eq!((p.engine_reserve_bytes, p.vision_reserve_bytes), (reserve, vision));
        let allocatable = free - reserve - vision - floor;
        let slot_state = 8 * profile.state_bytes_per_slot();
        assert_eq!(p.kv_blocks as u64, (allocatable - slot_state) / p.block_bytes);
        // one more byte of vision reserve is one fewer byte for pages, never a double count
        let tighter = plan_memory(&profile, free, floor, 8, 1 << 30, vision + p.block_bytes).unwrap();
        assert_eq!(tighter.kv_blocks, p.kv_blocks - 1);
    }

    // ---- the Qwen plan is pinned to the numbers the engine shipped with ------------------------------------------------------
    //
    // These literals were produced by the code as it stood at c655bea (before the family seam), for the real Qwen3.8-27B
    // `config.json` (`testdata/qwen38/config.json`). They must never be edited to make a refactor pass: a Qwen pack has to plan
    // byte-for-byte the memory it always did.

    // ---- GLM: the plan on the box it targets, and the hazard end to end

    #[test]
    fn plan_error_text_names_the_problem() {
        let e = plan_memory(&nemo_profile(false), GIB, 0, 1, 1 << 20, 0).unwrap_err();
        assert!(e.to_string().contains("not enough memory"), "{e}");
        assert!(PlanError::NoSlots.to_string().contains("slot"));
    }

    // ---- the prefix cache's share of the plan

    /// What one snapshot slot costs: a fixed input to the planner (the figure the Qwen3.8 pair's backend formula gave, 195,887,104
    /// bytes, kept as a literal when that family left the tree).
    fn snap_bytes() -> u64 {
        195_887_104
    }

    fn cached(free_gib: u64, slots: usize, cap: u64, req: CacheRequest) -> Result<MemoryPlan, PlanError> {
        plan_memory_cached(&nemo_profile(false), free_gib * GIB, 8 * GIB, slots, cap, 0, req, snap_bytes(), 0)
    }

    #[test]
    fn the_cache_off_is_the_plan_the_engine_shipped_with() {
        for (free, slots, cap) in [(20u64, 8usize, 262_144u64), (40, 4, 1 << 20), (80, 8, 262_144), (100, 4, 32_768)] {
            let off = cached(free, slots, cap, CacheRequest::Off).unwrap();
            let old = plan_memory(&nemo_profile(false), free * GIB, 8 * GIB, slots, cap, 0).unwrap();
            assert_eq!(off, old);
            assert_eq!(off.cache, CacheBudget::NONE);
        }
    }

    #[test]
    fn auto_never_changes_the_kv_plan_and_spends_at_most_half_of_what_is_spare() {
        let mut enabled = 0;
        for free in [14u64, 16, 20, 24, 30, 40, 60, 80, 100] {
            for (slots, cap) in [(1usize, 32_768u64), (4, 262_144), (8, 262_144), (8, 1 << 20)] {
                let Ok(off) = cached(free, slots, cap, CacheRequest::Off) else { continue };
                let auto = cached(free, slots, cap, CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES }).unwrap();
                let at = format!("free {free} GiB, {slots} slots, cap {cap}");
                // the KV pool, admission and the reserves are exactly the plan without a cache
                assert_eq!(
                    (auto.kv_blocks, auto.kv_tokens, auto.admit_bytes),
                    (off.kv_blocks, off.kv_tokens, off.admit_bytes),
                    "{at}"
                );
                assert_eq!(auto.spare_bytes, off.spare_bytes, "{at}");
                assert!(
                    auto.cache.bytes() <= auto.spare_bytes / 2 && auto.cache.bytes() <= AUTO_CACHE_CAP_BYTES,
                    "{at}: {:?}",
                    auto.cache
                );
                // and the whole plan still fits in what was free
                let total = auto.kv_blocks as u64 * auto.block_bytes
                    + slots as u64 * auto.state_bytes_per_slot
                    + auto.cache.bytes()
                    + auto.engine_reserve_bytes
                    + 8 * GIB;
                assert!(total <= free * GIB, "{at}: {total} > {}", free * GIB);
                enabled += auto.cache.enabled() as usize;
            }
        }
        assert!(enabled > 10, "the grid has boxes that get a cache ({enabled})");
    }

    // ---- nothing is planned that the box was not given, and the boot history is not an input

    #[test]
    fn a_plan_never_asks_for_more_than_it_was_given() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        // what the engine allocates beyond the weights (the reserves, the slots' state, the pool's pages, the cache) is at most `free - floor`, for both
        // families, every free size, slot count, context cap and cache request: the plan cannot over-commit the pool it was sized from
        let floor = 8 * GIB;
        let mut plans = 0;
        for free_gib in [14u64, 17, 20, 24, 30, 40, 60, 80, 113] {
            let free = free_gib * GIB;
            for slots in [1usize, 2, 4, 8, 32] {
                for cap in [32_768u64, 262_144, 1 << 20] {
                    for (profile, vision) in [(nemo_profile(false), 0u64), (nemo_profile(true), 1_000_000_000)] {
                        for req in [
                            CacheRequest::Off,
                            CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES },
                            CacheRequest::Fixed(2 * GIB),
                        ] {
                            let Ok(p) = plan_memory_cached(&profile, free, floor, slots, cap, vision, req, snap_bytes(), 0)
                            else {
                                continue;
                            };
                            let at = format!("{} at free {free_gib} GiB, {slots} slots, cap {cap}, {req:?}", profile.family());
                            assert!(
                                p.planned_bytes(slots) + floor <= free,
                                "{at}: {} + {floor} > {free}",
                                p.planned_bytes(slots)
                            );
                            // the sum is the six terms and no other
                            assert_eq!(
                                p.planned_bytes(slots),
                                p.engine_reserve_bytes
                                    + p.vision_reserve_bytes
                                    + slots as u64 * p.state_bytes_per_slot
                                    + p.kv_blocks as u64 * p.block_bytes
                                    + p.cache.bytes()
                                    + p.wq_budget_bytes,
                                "{at}"
                            );
                            plans += 1;
                        }
                    }
                }
            }
        }
        assert!(plans > 200, "the grid plans on most boxes ({plans})");
    }

    #[test]
    fn capacity_is_the_same_whatever_the_node_did_before_the_boot() {
        use crate::hostmem::{planning_free, MemInfo};
        // one box in two histories: after a cold boot MemFree is what is available; after 80 GiB of reads the cache holds 80 GiB of it
        let after_weights = 94 * GIB + GIB / 3;
        let cold = MemInfo { total: 121 * GIB, free: after_weights, available: after_weights + GIB / 5, cached: GIB };
        let dirty = MemInfo { total: 121 * GIB, free: 12 * GIB, available: after_weights, cached: 82 * GIB };
        let plan = |m: &MemInfo, req| {
            plan_memory_cached(&nemo_profile(false), planning_free(m.free, Some(m)), 8 * GIB, 8, 262_144, 0, req, snap_bytes(), 0)
                .unwrap()
        };
        for req in [CacheRequest::Off, CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES }] {
            let (a, b) = (plan(&cold, req), plan(&dirty, req));
            assert_eq!(a, b, "{req:?}: the KV pool and the cache budget do not depend on the history");
            assert_eq!(
                (a.kv_tokens, a.cache.bytes() > 7 * GIB),
                (262_144, req != CacheRequest::Off),
                "the full context and the 8 GiB cache (cap {:?})",
                a.cache
            );
        }
        // what the old rule (the smaller of the device's free and MemAvailable) made of the same dirty box
        let old = plan_memory_cached(
            &nemo_profile(false),
            dirty.free.min(dirty.available),
            8 * GIB,
            8,
            262_144,
            0,
            CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES },
            snap_bytes(),
            0,
        );
        assert!(old.as_ref().map_or(true, |p| p.kv_tokens < 262_144 && !p.cache.enabled()), "{old:?}");
    }

    // ---- the W_q residency budget: priced, fail-closed --------------------------------------------------------------------------------

    // ---- Q-ADM: the Qwen plan counts the features of the images in flight ---------------------------------------------------------

    // ---- admission follows the pool (serve hardening, finding 1) ---------------------------------------------------------------
    //
    // A 197,048-token prompt into a 194,048-token GLM pool was admitted and failed `OutOfPages` after 298.7 s of prefill. The plan's `admit_bytes`
    // is the pool's KV bytes plus the state of EVERY slot, and admission charges each sequence its own slot's state, so with fewer sequences
    // running the idle slots' state read as room for KV the pool does not have. These tests drive the plans of both families through the
    // admission the engines build (`MemoryPlan::admission`) and the real page tables (`KvTables`, what both backends allocate pages from).

    use crate::kvtables::KvTables;
    use gb_sched::AdmitError;

    const WINDOW: usize = 262_144;

    /// (name, profile, slots, plan) over a grid of boxes: some where the cap binds the pool at the window, some where memory binds it.
    fn pool_cases() -> Vec<(String, MemoryProfile, usize, MemoryPlan)> {
        let mut v = Vec::new();
        for (name, profile) in [("qwen", nemo_profile(false)), ("glm", nemo_profile(true))] {
            // (17 to 20 GiB is where a GLM pack's box landed after the weights: its pool was memory-bound there, as the reported 194,048
            // was; the Nemotron profile's pool is memory-bound below about 17 GiB at these slot counts (8 GiB floor, the reserve of
            // `each_holdback_...`), so 15 and 16 GiB keep at least three such boxes for each profile under P1 and P2)
            for free_gib in [15u64, 16, 17, 18, 20, 24, 30, 40, 100] {
                for slots in [1usize, 2, 4, 8] {
                    let cache = CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES };
                    if let Ok(plan) =
                        plan_memory_cached(&profile, free_gib * GIB, 8 * GIB, slots, WINDOW as u64, 0, cache, snap_bytes(), 0)
                    {
                        v.push((format!("{name}, {free_gib} GiB free, {slots} slots"), profile.clone(), slots, plan));
                    }
                }
            }
        }
        v
    }

    /// What `serve::build_engine` and the GLM engine built before this change: the byte capacity and the window, no pool.
    fn old_admission(profile: &MemoryProfile, plan: &MemoryPlan, slots: usize) -> Admission {
        Admission::new(plan.admit_bytes, 0, ModelMemory::from_profile(profile, 0, 0), slots)
            .unwrap()
            .with_max_context(WINDOW as u64)
    }

    /// A small deterministic generator (the test needs a workload, not randomness worth a dependency).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, n: u64) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) % n
        }
    }

    /// Drive `admission` with a mixed workload against the pool's real page tables: submit a request of a random size (some past the pool, some a
    /// large part of it, some small), and when it is admitted take every page it can ever use (`ensure(slot, tokens)`, the worst case of a
    /// sequence that runs to its limit), release a random running one now and then. `cache_pages` pages are held by a stand-in for the prefix
    /// cache (a table of its own) for the whole run. Returns (requests admitted, requests whose pages could not be had, requests refused as
    /// too long).
    fn drive(mut admission: Admission, plan: &MemoryPlan, slots: usize, cache_pages: usize, seed: u64) -> (usize, usize, usize) {
        let max_blocks = WINDOW.div_ceil(256);
        // the pool the backend allocates: the plan's pages plus the cache's, the cache's holder being one extra table
        let mut kv = KvTables::new(plan.kv_blocks + cache_pages, slots + 1, max_blocks);
        if cache_pages > 0 {
            kv.ensure(slots, cache_pages * 256).expect("the cache's own pages are in the pool");
        }
        let mut rng = Lcg(seed);
        let mut running: Vec<usize> = Vec::new();
        let (mut admitted, mut out_of_pages, mut too_long) = (0, 0, 0);
        for _ in 0..400 {
            let pool = plan.kv_tokens;
            let tokens = match rng.next(4) {
                0 => 1 + rng.next(300),
                1 => 1 + rng.next(pool / 4 + 1),
                2 => 1 + rng.next(pool),
                _ => pool / 2 + rng.next(pool / 2 + 4096), // up to a little past the pool
            };
            match admission.admit(tokens) {
                Ok(slot) => {
                    admitted += 1;
                    if kv.ensure(slot, tokens as usize).is_err() {
                        out_of_pages += 1;
                    }
                    running.push(slot);
                }
                Err(AdmitError::ContextTooLong { .. }) => too_long += 1,
                Err(_) => {} // NoSlot or NoMemory: the request waits
            }
            // finish a running request (always when every slot is taken, else now and then)
            if !running.is_empty() && (rng.next(3) == 0 || running.len() >= slots) {
                let at = rng.next(running.len() as u64) as usize;
                let slot = running.swap_remove(at);
                kv.release(slot);
                admission.release(slot);
            }
        }
        (admitted, out_of_pages, too_long)
    }

    #[test]
    fn the_admission_a_plan_builds_limits_a_request_to_the_pool_or_the_window_whichever_is_smaller() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        for (at, profile, slots, plan) in pool_cases() {
            let mm = ModelMemory::from_profile(&profile, 0, 0);
            for window in [4096usize, plan.kv_tokens as usize, WINDOW, 1 << 20] {
                let a = plan.admission(mm.clone(), slots, window).unwrap();
                assert_eq!(a.request_limit(), Some((window as u64).min(plan.kv_tokens)), "{at}, window {window}");
                assert_eq!(a.kv_tokens(), Some(plan.kv_tokens), "{at}");
                assert_eq!(a.capacity(), plan.admit_bytes, "{at}: the byte capacity is the plan's, as before");
            }
        }
    }

    #[test]
    fn a_prompt_past_the_pool_is_refused_at_once_for_both_families_where_the_old_rule_admitted_it() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let mut shown = [0usize; 2];
        for (at, profile, slots, plan) in pool_cases() {
            // the pool is below the window: memory binds it (at the cap there is no room above the pool to ask for)
            if plan.kv_tokens as usize >= WINDOW || slots < 2 {
                continue;
            }
            let mm = ModelMemory::from_profile(&profile, 0, 0);
            let new = plan.admission(mm, slots, WINDOW).unwrap();
            let old = old_admission(&profile, &plan, slots);
            // the slack the old rule left: the state of the idle slots, in tokens (a sequence is charged one slot's state, the plan paid for all)
            let slack = (slots as u64 - 1) * plan.state_bytes_per_slot / profile.kv_bytes_per_token();
            assert!(slack > 0, "{at}");
            // (the window still refuses what is past it, so the pool-and-window gap bounds what the old rule could admit)
            let gap = WINDOW as u64 - plan.kv_tokens;
            for over in [1, 3_000.min(slack).min(gap), slack.min(gap)] {
                let tokens = plan.kv_tokens + over;
                assert_eq!(old.check(tokens), Ok(()), "{at}: the old rule admits {over} tokens past the pool (the bug)");
                assert_eq!(
                    new.check(tokens),
                    Err(AdmitError::ContextTooLong { tokens, max: plan.kv_tokens }),
                    "{at}: {over} tokens past the pool"
                );
            }
            assert_eq!(new.check(plan.kv_tokens), Ok(()), "{at}: the pool's own size is admitted");
            shown[at.starts_with("glm") as usize] += 1;
        }
        assert!(shown[0] >= 3 && shown[1] >= 3, "the grid has boxes where memory binds the pool for each family ({shown:?})");
    }

    #[test]
    fn requests_that_fit_the_pool_never_run_out_of_pages_and_the_old_rule_did() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let (mut old_failures, mut new_admitted, mut new_too_long, mut runs) = (0, 0, 0, 0);
        for (at, profile, slots, plan) in pool_cases() {
            let mm = ModelMemory::from_profile(&profile, 0, 0);
            for seed in 0..6 {
                // the prefix cache holds its whole quota of pages for the entire run: its pages are on top of the pool, not part of it
                for cache_pages in [0, plan.cache.page_blocks] {
                    let new = plan.admission(mm.clone(), slots, WINDOW).unwrap();
                    let (admitted, out_of_pages, too_long) = drive(new, &plan, slots, cache_pages, seed);
                    assert_eq!(
                        out_of_pages, 0,
                        "{at}, seed {seed}, cache pages {cache_pages}: a request that was admitted ran out of pages"
                    );
                    new_admitted += admitted;
                    new_too_long += too_long;
                    let (_, failures, _) = drive(old_admission(&profile, &plan, slots), &plan, slots, cache_pages, seed);
                    old_failures += failures;
                    runs += 1;
                }
            }
        }
        // the driver is not vacuous: the same workload through the old construction runs out of pages, and the new one serves a lot of it
        assert!(old_failures > 0, "the old admission never ran out of pages in {runs} runs: the workload proves nothing");
        assert!(new_admitted > runs * 20, "admitted {new_admitted} in {runs} runs");
        assert!(new_too_long > 0, "the workload asks for more than the pool sometimes");
    }

    #[test]
    fn a_solo_request_of_the_pools_own_size_runs_whatever_the_slot_count() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        // the plan's `kv_blocks` has a page per slot beyond the pool for the rounding of each sequence's last page: a request of exactly
        // `kv_tokens` finds its pages with every other slot idle, and the cache's pages are not touched
        for (at, profile, slots, plan) in pool_cases() {
            let mm = ModelMemory::from_profile(&profile, 0, 0);
            let mut a = plan.admission(mm, slots, WINDOW).unwrap();
            let mut kv = KvTables::new(plan.kv_blocks + plan.cache.page_blocks, slots + 1, WINDOW.div_ceil(256));
            if plan.cache.page_blocks > 0 {
                kv.ensure(slots, plan.cache.page_blocks * 256).unwrap();
            }
            let want = plan.kv_tokens.min(WINDOW as u64);
            let slot = a.admit(want).unwrap_or_else(|e| panic!("{at}: {e:?}"));
            assert_eq!(kv.ensure(slot, want as usize), Ok(()), "{at}");
        }
    }
}
