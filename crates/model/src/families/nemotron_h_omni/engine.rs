//! The Nemotron serve engine: builds the scheduler-facing engine from a loaded model and the family plan, on
//! [`super::backend::NemotronBackend`]. It mirrors `qwen_flash_next::engine` minus the n-gram table, and
//! holds no device code: the model is constructed by the caller, from the options this module plans.
//!
//! **The prefix cache** (M5, `docs/NEMOTRON_LONG_CONTEXT.md`): [`plan_serving_for`] carves it for a runtime type that can snapshot
//! (`NemotronRuntime::snapshots_supported`), [`assemble`] gives the runtime its arena and the backend its budget and namespace, and the engine
//! is told to use it. The split of the budget is [`super::prefix::split_budget`]: a unit is a snapshot slot and an eighth of the context in pages.
//!
//! The order of a boot, which the pieces below make explicit so the family's `build` is a straight line:
//!
//! 1. [`check_options`]: the options the engine cannot honour are refused by name, before anything is read.
//! 2. [`plan_serving`]: the memory plan (slots, KV pages) from the free memory after the weights, priced by the family's
//!    [`crate::family::MemorySource`], with the MTP head's cache and buffers when the pack offers the head and `--spec` drafts with
//!    it. The model's [`ModelOptions`] come out of it.
//! 3. the caller loads the model with those options (device code);
//! 4. [`assemble`]: the backend (with its speculation runtime: the head, the copy drafter and the arbiter), the admission (from the
//!    plan, never a constant) and the engine.
//!
//! [`build`] runs 1 to 4 with the load as a closure; [`build_serve`] erases the backend type for `ModelFamily::build`.
//!
//! **`--spec`.** `auto` (the default) lets the arbiter choose per stream and round among none, the copy drafter and the head at a depth
//! (`spec.rs`); `mtp` drafts with the head at `REXL3_MTP_DEPTH` (default 3); `copy` offers the copy drafter's matches only; `none`
//! never drafts. `auto` and `mtp` make the engine's own row budget flat (the arbiter, or the fixed depth, decided how many rows ride);
//! `copy` keeps the step table's trim.

use super::backend::{prefill_rows_max, ModelOptions, NemotronBackend, NemotronRuntime, WINDOW_ROWS};
use super::mtp::{restricted_vocab_ids_for, VocabSetting};
use super::prefix;
use super::spec::{ArbTable, CostBasis, CostTables, HeadForm, MtpCost, SpecConfig, LM_HEAD_RESTRICTED_ROWS};
use crate::error::ModelError;
use crate::family::{BuildCtx, FamilyPlan};
use crate::hostmem::{self, MemInfo};
use crate::memory::{MemoryOptions, VisionOptions};
use crate::plan::{choose_slots, plan_memory, CacheRequest, MemoryPlan, PlanError, MIN_TOKENS_PER_SLOT};
use crate::serve::{allocation_error, gib, ServeOptions};
use gb_engine::{boot, Drafter, Engine, EngineConfig, NoDrafts, PrefillPolicy, ServeEngine};
use gb_sched::{CostModel, MemoryProfile, ModelMemory};
use gb_spec::SpecMode;

/// The drafts one stream may offer per round (the copy drafter's block).
const MAX_DRAFTS: usize = 7;

/// Milliseconds a step takes by rows, for the engine's draft budgeting (`CostModel`) under `--spec copy`, and the arbiter's verify
/// table under `auto`: [`super::spec::VERIFY_MS`], where the numbers and their derivation live. It shapes how many drafts are verified
/// and never which token is committed, so a wrong curve costs speed and not exactness. MEASURED on the real model with the fast MoE
/// kernels and the window graphs (37 ms for one row, 69 for eight, 196 for sixty-four; `tests/nemotron_omni_costs.rs` regenerates it);
/// the step will fall again as the perf workers land, through [`assemble_with_cost`] or [`assemble_with`] and no code change.
pub const STEP_MS: &[(usize, f64)] = super::spec::VERIFY_MS;

/// The rows of one prefill slice for these options: `--prefill-chunk`, at most [`prefill_rows_max`] (512 by default, 256 under the opt-out
/// `REXL3_NEMO_PREFILL_ROWS_MAX=256`: above the 64 rows of a window the MoE layers run the large-row prefill path, which reads each expert's
/// weights once per slice; `model::MAX_PREFILL_ROWS` and `model::WIDE_PREFILL_ROWS` have the reasons). Slicing never changes a result (chunked prefill equals whole by the exactness gate), so an operator's larger value, and
/// the shipped default of 4,096, are clamped with a log line and not refused.
pub fn prefill_rows(o: &ServeOptions) -> usize {
    o.prefill_chunk.min(prefill_rows_max())
}

/// The environment variable that asks a window wider than `--window-rows` takes (the shared option stops at 128 rows, the decode kernels'
/// limit for the other families): the value replaces `--window-rows`, `WINDOW_ROWS..=MAX_WINDOW_ROWS`.
pub const WINDOW_ROWS_VAR: &str = "REXL3_NEMO_WINDOW_ROWS";

/// The rows of the decode window this boot runs: `--window-rows` (64 unless given), or `REXL3_NEMO_WINDOW_ROWS` when it is set. One
/// window of that many rows (all sequences, base rows and drafts) is one forward; above 64 the MoE layers take the large-row path, which
/// reads each expert's weights once for the whole window. A single-row stream's step is unchanged by it.
pub fn window_rows(o: &ServeOptions) -> Result<usize, ModelError> {
    window_rows_with(o, std::env::var(WINDOW_ROWS_VAR).ok().as_deref())
}

/// [`window_rows`] with the value of [`WINDOW_ROWS_VAR`] given (`None`: unset), for a caller that must not read the environment.
pub fn window_rows_with(o: &ServeOptions, var: Option<&str>) -> Result<usize, ModelError> {
    let rows = match var {
        Some(v) => {
            v.trim().parse::<usize>().map_err(|_| ModelError::Config(format!("{WINDOW_ROWS_VAR}={v:?}: not a number of rows")))?
        }
        None => o.window_rows,
    };
    super::model::check_window_rows(rows).map_err(|e| ModelError::Config(format!("Nemotron 3.5 Super --window-rows: {e}")))?;
    Ok(rows)
}

/// The environment variable that forces the fused decode step on (`1`, `on`) or off (`0`, `off`); unset, it is on exactly for a wide
/// window (more than 64 rows), where many sequences each pay the commit's read and write of their state (`NemotronBackend::with_fused_decode`).
pub const FUSED_DECODE_VAR: &str = "REXL3_NEMO_FUSED_DECODE";

/// Whether a boot with a decode window of `window` rows runs fused decode steps ([`FUSED_DECODE_VAR`]).
pub fn fused_decode(window: usize) -> Result<bool, ModelError> {
    fused_decode_with(window, std::env::var(FUSED_DECODE_VAR).ok().as_deref())
}

/// [`fused_decode`] with the value of [`FUSED_DECODE_VAR`] given (`None`: unset).
pub fn fused_decode_with(window: usize, var: Option<&str>) -> Result<bool, ModelError> {
    match var.map(str::trim) {
        None => Ok(window > WINDOW_ROWS),
        Some("1" | "on" | "true" | "yes") => Ok(true),
        Some("0" | "off" | "false" | "no") => Ok(false),
        Some(v) => Err(ModelError::Config(format!("{FUSED_DECODE_VAR}={v:?}: expected 0, 1, on or off"))),
    }
}

/// A tensor-parallel group (`--tp-ranks 2`) is built and verified for the 64-row decode window, the plain decode step and the serial
/// prefill: a wide window, the fused or the lagged decode and the prefill burst are refused by name rather than run unverified. Reads the
/// same settings the boot does ([`window_rows`], [`fused_decode`], [`lag_period_for`], [`prefill_burst`]).
pub fn check_tp_plain(o: &ServeOptions) -> Result<(), ModelError> {
    let window = window_rows(o)?;
    check_tp_plain_with(window, lag_period_for(window)?, fused_decode(window)?, prefill_burst(window)?)
}

/// [`check_tp_plain`] with the resolved settings given: the window's rows, the lagged decode's fold period (`0`: off), whether the fused
/// decode step is on and the prefill burst's prompt count (`None`: off).
pub fn check_tp_plain_with(window: usize, lag: usize, fused: bool, burst: Option<usize>) -> Result<(), ModelError> {
    let what = if window > WINDOW_ROWS {
        Some(format!("a decode window of {window} rows ({WINDOW_ROWS_VAR} / --window-rows)"))
    } else if lag >= 2 {
        Some(format!("the lagged decode ({}={lag})", super::model::LAG_ENV))
    } else if fused {
        Some(format!("the fused decode step ({FUSED_DECODE_VAR})"))
    } else if burst.is_some() {
        Some(format!("the prefill burst ({PREFILL_BURST_VAR})"))
    } else {
        None
    };
    match what {
        None => Ok(()),
        Some(w) => Err(ModelError::Config(format!(
            "{w} is not built for tensor parallelism (--tp-ranks 2 runs the {WINDOW_ROWS}-row window with the plain decode and the serial \
             prefill); boot a single rank for it, or unset the setting"
        ))),
    }
}

/// The fold period of the LAGGED decode a boot with a decode window of `window` rows runs (`REXL3_NEMO_LAG`, [`super::model::LAG_ENV`]): the
/// value of the variable (`0` or `1`: off, `2..=16`), or, unset, [`DEFAULT_WIDE_LAG`] for a wide window (more than 64 rows) and off for a
/// narrow one. A single-row sequence's Mamba state is then stored once every K steps; the rows between wait in a per-slot ring and are
/// re-applied by the next step (the same sequential advance: the same bits).
pub fn lag_period_for(window: usize) -> Result<usize, ModelError> {
    lag_period_with(window, std::env::var(super::model::LAG_ENV).ok().as_deref())
}

/// The fold period a wide boot runs when [`super::model::LAG_ENV`] is unset (measured, 256 streams on a node: no lag 291 tok/s, K = 4
/// 351, K = 8 362).
pub const DEFAULT_WIDE_LAG: usize = 8;

/// [`lag_period_for`] with the value of the variable given (`None`: unset).
pub fn lag_period_with(window: usize, var: Option<&str>) -> Result<usize, ModelError> {
    match var.map(str::trim) {
        None => Ok(if window > WINDOW_ROWS { DEFAULT_WIDE_LAG } else { 0 }),
        Some(v) => match v.parse::<usize>() {
            Ok(0 | 1) => Ok(0),
            Ok(k) if k <= 16 => Ok(k),
            _ => Err(ModelError::Config(format!(
                "{}={v:?}: expected 0 (off) or a fold period of 2 to 16 steps",
                super::model::LAG_ENV
            ))),
        },
    }
}

/// Under the P2 prefill contract (the default) a boot whose prefill rows cap is 64 or less is refused by name (v0.1.1,
/// `contract::check_rows_cap`): the model would build no tensor-core MoE scratch and its MoE layers would run P1's fp32 kernels inside a
/// P2 prefill, a hybrid no fingerprint names. With the 64-row window that is `--prefill-chunk` 64 or less; P1 takes any chunk. The
/// family's boot calls it on every rank before anything is bound, planned or read (a follower says so on its connection to rank 0); the
/// option check below does not, so the engine's tests keep their fake runtime of 64-row slices.
pub fn check_prefill_contract(o: &ServeOptions) -> Result<(), ModelError> {
    super::contract::check_rows_cap(super::contract::current(), prefill_rows(o), window_rows(o)?).map_err(ModelError::Config)
}

/// The options this engine cannot honour, refused by name before anything is loaded: what the shared check refuses (`serve::check_options`)
/// and the caps that follow the family. The decode window is 64 rows unless `--window-rows` (or `REXL3_NEMO_WINDOW_ROWS`) asks for a
/// wider one, up to `model::MAX_WINDOW_ROWS`. `--spec mtp` is not refused here (this check sees the options and
/// not the pack): [`plan_serving`] refuses it by name when the pack offers no MTP head, and [`assemble`] when the loaded model has none.
/// The P2 rows cap is the boot's own check ([`check_prefill_contract`]).
pub fn check_options(o: &ServeOptions) -> Result<(), ModelError> {
    crate::serve::check_options(o)?;
    window_rows(o)?;
    if o.max_model_len == 0 {
        return Err(ModelError::Config("--max-model-len must be at least 1".into()));
    }
    if o.draft_dir.is_some() {
        return Err(ModelError::Config(
            "--draft-dir: Nemotron 3.5 Super has no external draft model; its drafters are the copy drafter and its own MTP head"
                .into(),
        ));
    }
    Ok(())
}

/// Whether the plan prices the MTP head, which is whether the model loads it (`ModelOptions::mtp`, which `NemotronModel::load` reads together
/// with the same two conditions below): the pack offers it (`FamilyPlan::drafters`, which lists `mtp` once the pack has the head and
/// `nemotron_mtp` is built), `--spec` drafts with it, and `REXL3_MTP` is not `off`. At plan time the model is not loaded, so the pack's
/// offer is what "the runtime has a head" can mean; [`assemble`] checks the loaded model against the plan.
pub fn head_priced(fam: &FamilyPlan, o: &ServeOptions) -> bool {
    head_priced_if(fam, o, super::mtp::head_wanted(None))
}

/// [`head_priced`] with the answer of `REXL3_MTP` given (`mtp::head_wanted(None)`), for a test that does not touch the environment.
pub fn head_priced_if(fam: &FamilyPlan, o: &ServeOptions, wanted: bool) -> bool {
    wanted && o.drafting && o.spec.uses_mtp() && fam.drafters.contains(&"mtp")
}

/// What [`PlanInputs::weight_bytes`] must be: the resident weights. `LayerPlan::bytes` lists the head's tensors whatever the options, but
/// they reach the device only when the model loads the head ([`head_priced`]): without it the plan would hold back 5.5 GiB nobody uses.
/// `head_listed` is the bytes of the plan's `UnitId::Mtp` tensors.
pub fn resident_weight_bytes(listed: u64, head_listed: u64, fam: &FamilyPlan, o: &ServeOptions) -> u64 {
    if head_priced(fam, o) {
        listed
    } else {
        listed.saturating_sub(head_listed)
    }
}

/// [`resident_weight_bytes`] for one rank of a TP group: only what the rank holds is charged (`tp_shard::ShardPlan`: its experts, its
/// columns of the dense linears, its `lm_head` rows, and every replicated tensor in full), so `plan_serving` sees the free memory the
/// rank really has and can admit the slots and KV pages the other half of the weights used to take. `ShardPlan::single()` gives
/// [`resident_weight_bytes`] of `LayerPlan::bytes`. `experts` is `OmniConfig::experts`.
pub fn resident_weight_bytes_tp(
    plan: &super::weights::LayerPlan,
    experts: usize,
    shard: &super::tp_shard::ShardPlan,
    fam: &FamilyPlan,
    o: &ServeOptions,
) -> u64 {
    let listed = shard.rank_bytes(plan, experts, |_| true);
    let head = shard.rank_bytes(plan, experts, |u| matches!(u, super::weights::UnitId::Mtp(_)));
    resident_weight_bytes(listed, head, fam, o)
}

/// What a group plans from, over ALL its ranks: the largest of [`resident_weight_bytes_tp`] over `shards` (one plan per rank, rank
/// `0..R`), so that every rank plans the same slots and KV pages from the same free memory and the boot gate compares like with like.
/// A rank holds a little more or less than another (the `in_proj` tile and the experts of an uneven cut, the vocabulary rows): the
/// plan is for the fullest. Two ranks: rank 1 holds the odd `in_proj` tile (9,344 against 9,216 columns) and the larger half of the
/// vocabulary's units. An empty group is zero.
pub fn resident_weight_bytes_group(
    plan: &super::weights::LayerPlan,
    experts: usize,
    shards: &[super::tp_shard::ShardPlan],
    fam: &FamilyPlan,
    o: &ServeOptions,
) -> u64 {
    shards.iter().map(|s| resident_weight_bytes_tp(plan, experts, s, fam, o)).max().unwrap_or(0)
}

/// Rows of the restricted draft head this process builds, or `None` for the full head: the ids of the list `REXL3_MTP_VOCAB` names
/// (unset: the default 49,152-id list; `off`: none; a file: its ids), sorted and de-duplicated, from the same function the model builds
/// its sub-table with (`mtp::restricted_vocab_ids`). The model makes the sub-table at load, AFTER the weights and outside the plan's
/// profile; the plan prices it from this count ([`sub_head_bytes`]) and the arbiter prices its draft call with it.
pub fn restricted_head_rows() -> Result<Option<usize>, ModelError> {
    restricted_head_rows_for(&VocabSetting::from_env())
}

/// [`restricted_head_rows`] for a setting given (the seam tests use instead of the process environment).
pub fn restricted_head_rows_for(setting: &VocabSetting) -> Result<Option<usize>, ModelError> {
    Ok(restricted_vocab_ids_for(setting)?.map(|ids| ids.len()))
}

/// The distinct token ids of a `mtp_vocab.py` file.
pub fn restricted_rows_of_file(path: &std::path::Path) -> Result<usize, ModelError> {
    let ids = restricted_vocab_ids_for(&VocabSetting::File(path.to_path_buf()))?;
    Ok(ids.map_or(0, |v| v.len()))
}

/// Device bytes one id of a restricted head's sub-table takes: its `lm_head` row (bf16 `[4096]`) and its id map entry (i32).
pub const SUB_HEAD_ROW_BYTES: u64 = 4096 * 2 + 4;

/// Device bytes of a restricted head's sub-table over `rows` token ids: their `lm_head` rows (bf16 `[rows][4096]`) and the id map (i32).
/// `mtp::SubHead::bytes` of the table the model builds is exactly this (a GPU test holds it).
pub fn sub_head_bytes(rows: usize) -> u64 {
    rows as u64 * SUB_HEAD_ROW_BYTES
}

/// Whether this boot loads the RADIO tower: the server asked for images (`ServeOptions::vision`, which `--no-vision` leaves unset) and the
/// pack has a vision component to serve them with. The plan prices the tower (the features of the images in flight, and the tower's own
/// bytes in `FamilyPlan::memory_profile`) exactly when this is true, and the family's `build` loads it exactly when this is true: ONE
/// predicate, as `head_priced` is for the head, so the plan and the allocation cannot disagree about it.
pub fn vision_wanted(fam: &FamilyPlan, o: &ServeOptions) -> bool {
    o.vision.is_some() && fam.vision.is_some()
}

/// The images this boot serves, as the memory model prices them: the per-image token limit the tower holds (`vision::image_token_cap`) and
/// the server's own count of images in one request.
pub fn vision_options(fam: &FamilyPlan, o: &ServeOptions) -> Option<VisionOptions> {
    vision_wanted(fam, o).then(|| VisionOptions {
        max_image_tokens: super::vision::image_token_cap(o.max_image_tokens),
        max_images_per_request: VisionOptions::SERVER_DEFAULT.max_images_per_request,
    })
}

/// What the plan needs from the box, measured by the caller (the device calls are theirs).
#[derive(Clone, Copy, Debug)]
pub struct PlanInputs<'a> {
    /// `cudaMemGetInfo` free, logged and the fallback when the host reports no `MemAvailable`.
    pub cuda_free: u64,
    /// `/proc/meminfo`, when it has `MemAvailable` (`MemInfo::read`).
    pub host: Option<&'a MemInfo>,
    /// The resident weights: `LayerPlan::bytes(&inventory)`, taken out of the free memory before anything is priced.
    pub weight_bytes: u64,
    /// The model's own context limit (`OmniConfig::max_context`), which `--max-model-len` is clamped to.
    pub model_max_context: usize,
}

/// A serving plan: the settings the model is built with and everything the engine is assembled from.
#[derive(Clone, Debug)]
pub struct ServePlan {
    /// What `--slots` asked for.
    pub requested_slots: usize,
    /// What the plan holds (fewer than requested only when the slots were not asked for: `ServeOptions::slots_auto`).
    pub slots: usize,
    /// `min(--max-model-len, the model's limit)`.
    pub max_context: usize,
    /// The options to load the model with.
    pub model: ModelOptions,
    pub profile: MemoryProfile,
    pub plan: MemoryPlan,
    /// What the plan was sized from: the free memory after the weights, rounded to the planning quantum.
    pub free_after_weights: u64,
    /// The plan prices the MTP head (its cache per token, its tap stores, its buffers): [`head_priced`]. The loader must allocate the
    /// head's cache layer exactly when this is true, or the pages it holds are not the pages the profile counted.
    pub mtp: bool,
    /// Bytes of the restricted draft head's sub-table, taken out of the free memory with the weights (the model builds it at load).
    pub sub_head_bytes: u64,
    /// The images the plan prices (`vision_options`): the per-image token limit the tower is built for, `None` for a text-only boot. The
    /// features reserve is `plan.vision_reserve_bytes`; the tower's weights and workspace are `vision::weight_bytes` and
    /// `vision::scratch_bytes(tokens)`, which the tower checks against its allocations at load and which the loader takes out of the free
    /// memory before the plan measures it.
    pub vision: Option<VisionOptions>,
    /// The token the placeholders of an image are (`VisionSpec::image_token_id`), when images are served: the backend refuses a span over
    /// other tokens.
    pub image_token: Option<u32>,
    /// The pack's identity (`prefix::pack_identity`) when the plan carved a prefix cache (`plan.cache.enabled()`), the namespace's input;
    /// `None` without one. The cache's snapshot slots and pages are `plan.cache`; the pool the model is built with already holds its pages
    /// (`model.kv_pages == plan.kv_blocks + plan.cache.page_blocks`) and admission was never told about them.
    pub pack_identity: Option<String>,
}

/// The memory plan of a boot: the family's profile at these options, the slot count the free memory holds, and the model options that
/// follow. Pure arithmetic over `inp` and the family's memory model: nothing is allocated and no device is asked. The one thing it reads
/// is the vocabulary file `REXL3_MTP_VOCAB` names (a count of ids), when the head is priced: [`plan_serving_with`] takes the bytes.
pub fn plan_serving(fam: &FamilyPlan, o: &ServeOptions, inp: &PlanInputs<'_>) -> Result<ServePlan, ModelError> {
    plan_serving_vocab(fam, o, inp, &VocabSetting::from_env())
}

/// [`plan_serving`] with the draft vocabulary setting given (what `REXL3_MTP_VOCAB` is, for a caller that must not depend on the
/// environment): the sub-head is priced when the head is, from the ids of that setting.
pub fn plan_serving_vocab(
    fam: &FamilyPlan,
    o: &ServeOptions,
    inp: &PlanInputs<'_>,
    vocab: &VocabSetting,
) -> Result<ServePlan, ModelError> {
    let sub = if head_priced(fam, o) { restricted_head_rows_for(vocab)?.map_or(0, sub_head_bytes) } else { 0 };
    plan_serving_with(fam, o, inp, sub)
}

/// [`plan_serving`] with the bytes of the restricted head's sub-table given (zero for the full head). Plans no prefix cache.
pub fn plan_serving_with(
    fam: &FamilyPlan,
    o: &ServeOptions,
    inp: &PlanInputs<'_>,
    sub_head_bytes: u64,
) -> Result<ServePlan, ModelError> {
    plan_serving_with_cache(fam, o, inp, sub_head_bytes, false)
}

/// [`plan_serving`] for the runtime type `M`: when `M` can snapshot (`NemotronRuntime::snapshots_supported`), the plan also carves the prefix
/// cache `o.prefix_cache` asks for, out of the memory the KV pool leaves (`Auto`, at most half of it) or before the pool is sized (an explicit
/// `--prefix-cache-mib`); a runtime that cannot is planned exactly as [`plan_serving`] does. This is what [`build_with`] plans with.
pub fn plan_serving_for<M: NemotronRuntime>(
    fam: &FamilyPlan,
    o: &ServeOptions,
    inp: &PlanInputs<'_>,
) -> Result<ServePlan, ModelError> {
    let sub = if head_priced(fam, o) { restricted_head_rows()?.map_or(0, sub_head_bytes) } else { 0 };
    plan_serving_with_cache(fam, o, inp, sub, M::snapshots_supported())
}

/// The plan's cache for one slot count: [`prefix::split_budget`] over `request`'s bytes. An explicit size is taken first (the KV pool gets what
/// remains, and the plan fails closed if the slot state and one page do not fit); `Auto` only spends what the pool left over, so it never makes
/// the KV plan smaller.
#[allow(clippy::too_many_arguments)]
fn plan_with_cache(
    profile: &MemoryProfile,
    free: u64,
    floor: u64,
    slots: usize,
    kv_tokens_cap: u64,
    features: u64,
    request: CacheRequest,
    snapshot_bytes: u64,
    entry_pages: u64,
) -> Result<MemoryPlan, PlanError> {
    let page_bytes = profile.kv_page_bytes();
    match request {
        CacheRequest::Off => plan_memory(profile, free, floor, slots, kv_tokens_cap, features),
        CacheRequest::Fixed(bytes) => {
            let budget = prefix::split_budget(bytes, snapshot_bytes, page_bytes, entry_pages);
            let mut plan = plan_memory(profile, free.saturating_sub(budget.bytes()), floor, slots, kv_tokens_cap, features)?;
            plan.cache = budget;
            Ok(plan)
        }
        CacheRequest::Auto { cap_bytes } => {
            let mut plan = plan_memory(profile, free, floor, slots, kv_tokens_cap, features)?;
            let cap = prefix::auto_cap_bytes(snapshot_bytes, page_bytes, entry_pages, cap_bytes);
            plan.cache = prefix::split_budget(cap.min(plan.spare_bytes / 2), snapshot_bytes, page_bytes, entry_pages);
            Ok(plan)
        }
    }
}

/// The co-tenancy headroom (`--kv-headroom-tokens`) is a demand, not a cap: the plan must hold one context (`base_cap`) plus `headroom` tokens of
/// pool, or the boot is refused by name (`plan_memory` would otherwise plan what fits, silently less). And a headroom must not cost the `Auto`
/// prefix cache its one maximal entry (`max_entry_pages` pages): when the plan WITHOUT the headroom (`without`) holds one and the plan with it does
/// not, the boot is refused too, naming both ways out.
fn check_headroom(
    plan: &MemoryPlan,
    base_cap: u64,
    headroom: u64,
    without: impl FnOnce() -> Result<MemoryPlan, PlanError>,
    max_entry_pages: u64,
) -> Result<(), ModelError> {
    let want = base_cap + headroom;
    if plan.kv_tokens < want {
        return Err(ModelError::Config(format!(
            "Nemotron memory plan: --kv-headroom-tokens {headroom} asks for a KV pool of {want} tokens ({base_cap} for one context plus the \
             headroom), the box holds {} tokens; lower --kv-headroom-tokens or free memory",
            plan.kv_tokens
        )));
    }
    let holds_one = |p: &MemoryPlan| p.cache.page_blocks as u64 >= max_entry_pages;
    if !holds_one(plan) {
        if let Ok(base) = without() {
            if holds_one(&base) {
                return Err(ModelError::Config(format!(
                    "Nemotron memory plan: --kv-headroom-tokens {headroom} leaves the prefix cache {} pages, fewer than one maximal entry \
                     ({max_entry_pages} pages) that it holds without the headroom; lower the headroom or size the cache with \
                     --prefix-cache-mib",
                    plan.cache.page_blocks
                )));
            }
        }
    }
    Ok(())
}

/// [`plan_serving_with`] that carves the prefix cache when `snapshots` (the runtime can snapshot) and `o.prefix_cache` asks for one.
pub fn plan_serving_with_cache(
    fam: &FamilyPlan,
    o: &ServeOptions,
    inp: &PlanInputs<'_>,
    sub_head_bytes: u64,
    snapshots: bool,
) -> Result<ServePlan, ModelError> {
    check_options(o)?;
    // `--spec mtp` asks for a drafter the pack has to offer: refused by name before anything is priced or read
    if o.spec == SpecMode::Mtp && !fam.drafters.contains(&"mtp") {
        return Err(if fam.mtp_layers.is_empty() {
            ModelError::Config(
                "--spec mtp: this pack has no MTP head (`num_nextn_predict_layers` is 0); use --spec auto, copy or none".into(),
            )
        } else {
            ModelError::NotBuilt(
                "--spec mtp: the Nemotron MTP head (component `nemotron_mtp`) is not built in this release; use --spec auto, copy or none"
                    .into(),
            )
        });
    }
    let max_context = o.max_model_len.min(inp.model_max_context);
    // the head's cache (1 KiB a token), its tap stores and its buffers are counted exactly when the head will be allocated
    let mtp = head_priced(fam, o);
    // the images in flight: the tower is loaded before the free memory is measured (it is out of `free` below), so the features are the part
    // of vision the pool must be planned around, and they grow with the slots (every slot may hold a request of several images)
    let vision = vision_options(fam, o);
    let profile = fam
        .memory_profile(&MemoryOptions {
            window_rows: window_rows(o)?,
            prefill_rows: prefill_rows(o),
            max_context,
            mtp,
            vision,
            trees: false,
        })
        .map_err(|e| ModelError::Config(e.to_string()))?;
    // sized from the host's MemAvailable and never the smaller of it and the device's free (see `hostmem`); the weights come out once
    let cuda_free = inp.cuda_free;
    let free = hostmem::planning_free(cuda_free, inp.host).saturating_sub(inp.weight_bytes).saturating_sub(sub_head_bytes);
    // the pool: one context (or less, `--kv-tokens`), plus the co-tenancy headroom asked for (`--kv-headroom-tokens`, 0 by default). A single
    // request stays limited to the context (`Admission::request_limit` is the smaller of the two), so the headroom only ever seats OTHER requests
    let base_cap = o.kv_tokens.unwrap_or(max_context).min(max_context) as u64;
    let headroom = o.kv_headroom_tokens as u64;
    let kv_tokens_cap = base_cap
        .checked_add(headroom)
        .ok_or_else(|| ModelError::Config(format!("--kv-headroom-tokens {headroom} overflows the pool")))?;
    // the prefix cache: a snapshot slot is the recurrent state of a slot (and the head's carry), a page is the profile's; an entry is assumed to hold
    // an eighth of the context in pages (`prefix::entry_pages`), not the shared planner's 8, so that a long prompt's end cut can be stored
    let request = if snapshots { o.prefix_cache.request() } else { CacheRequest::Off };
    let snapshot_bytes = prefix::snapshot_bytes(profile.state_bytes_per_seq(), mtp);
    let entry_pages = prefix::entry_pages(max_context);
    if let CacheRequest::Fixed(bytes) = request {
        let unit = prefix::unit_bytes(snapshot_bytes, profile.kv_page_bytes(), entry_pages);
        if !prefix::split_budget(bytes, snapshot_bytes, profile.kv_page_bytes(), entry_pages).enabled() {
            return Err(ModelError::Config(format!(
                "--prefix-cache-mib {} cannot hold two entries of {} MiB each (a snapshot slot and {entry_pages} pages, an eighth of the \
                 {max_context}-token context); give it at least {} MiB, or 0 to turn the cache off",
                bytes >> 20,
                unit >> 20,
                (2 * unit).div_ceil(1 << 20)
            )));
        }
    }
    let plan_with_cap = |cap: u64, slots: usize| {
        let features = match vision {
            Some(v) => profile.vision_reserve_bytes(0, v.concurrent_images(slots)).map_err(PlanError::Memory)?,
            None => 0,
        };
        plan_with_cache(&profile, free, o.floor_bytes, slots, cap, features, request, snapshot_bytes, entry_pages)
    };
    let plan_at = |slots: usize| plan_with_cap(kv_tokens_cap, slots);
    let choice = choose_slots(o.slots, o.slots_auto, kv_tokens_cap, plan_at).map_err(|e| {
        ModelError::Config(format!("Nemotron memory plan: {e}; MemAvailable after the weights {:.1} GiB", gib(free)))
    })?;
    if choice.plan.kv_tokens < (choice.slots as u64 * MIN_TOKENS_PER_SLOT).min(kv_tokens_cap) {
        return Err(ModelError::Config(format!(
            "Nemotron KV pool too small for {} slots: {} tokens planned; use fewer --slots or a smaller --floor",
            choice.slots, choice.plan.kv_tokens
        )));
    }
    if headroom > 0 {
        let max_entry_pages = (max_context as u64).div_ceil(256);
        check_headroom(&choice.plan, base_cap, headroom, || plan_with_cap(base_cap, choice.slots), max_entry_pages)?;
    }
    // the pool holds the pages the cache may keep on top of the pages admission hands out (admission is never told about them)
    let model = ModelOptions {
        slots: choice.slots,
        kv_pages: choice.plan.kv_blocks + choice.plan.cache.page_blocks,
        max_context,
        prefill_rows: prefill_rows(o),
        layers: None,
        mtp,
    };
    let pack_identity =
        if choice.plan.cache.enabled() { Some(prefix::pack_identity(&fam.dir).map_err(ModelError::Config)?) } else { None };
    Ok(ServePlan {
        requested_slots: choice.requested,
        slots: choice.slots,
        max_context,
        model,
        profile,
        plan: choice.plan,
        free_after_weights: free,
        mtp,
        sub_head_bytes,
        vision,
        image_token: fam.vision.as_ref().filter(|_| vision.is_some()).map(|v| v.image_token_id),
        pack_identity,
    })
}

/// The engine from a loaded runtime and the plan it was loaded for: the backend with its speculation runtime (the head when the model
/// has one and `--spec` uses it, the copy drafter, the arbiter priced by [`CostTables::shipped`]), the admission (from the
/// plan's pool, so a prompt past the KV pool is refused at submit and never prefilled for minutes), and the engine's row budget.
/// The speculation settings are read from the environment (`REXL3_MTP_DEPTH`, `REXL3_NEMO_DRAFT_NOISE`, `REXL3_NEMO_ARB_TABLE`; malformed
/// values are errors).
pub fn assemble<M: NemotronRuntime>(
    model: M,
    o: &ServeOptions,
    sp: &ServePlan,
) -> Result<Engine<NemotronBackend<M>>, ModelError> {
    // the draft-call tables follow the LOADED head's form (a bf16 head: today's tables exactly; a 4-bit head: the H4 tables)
    let head = model.head_form();
    let (cfg, costs) = spec_settings_for(o, sp, None, head)?;
    log_cost_basis_for(o, head);
    assemble_with(model, o, sp, cfg, &costs)
}

/// [`log_cost_basis`] with the loaded head's form named: the tensor-parallel group's line gains `head=bf16|exl3` (and, for a 4-bit head
/// under the measured two-rank tables with the head split, says its draft tables are the bf16 group's scaled by the single-node ratios:
/// unmeasured), and a single node with the MTP head drafting logs `event=mtp_costs head=... tables=...`.
pub fn log_cost_basis_for(o: &ServeOptions, head: HeadForm) {
    if let Some(t) = o.tp.as_ref().filter(|t| t.is_group()) {
        let basis = tp_cost_basis(o);
        let split = {
            let h = super::tp_arm::head_split();
            h.experts && h.lm_head
        };
        let tables = match (basis, head) {
            (CostBasis::SingleNode, HeadForm::Bf16) => "single-node (shipped)".to_string(),
            (CostBasis::SingleNode, HeadForm::Exl3) => "single-node (shipped verify, MTP_COST_H4 draft)".to_string(),
            (CostBasis::MeasuredTwoRanks, HeadForm::Bf16) => "two-rank group (measured VERIFY_MS_TP, MTP_COST_TP)".to_string(),
            (CostBasis::MeasuredTwoRanks, HeadForm::Exl3) if split => "two-rank group (measured VERIFY_MS_TP; draft MTP_COST_TP scaled per term \
                by the single-node MTP_COST_H4 / MTP_COST ratios: unmeasured, extrapolated)"
                .to_string(),
            (CostBasis::MeasuredTwoRanks, HeadForm::Exl3) => {
                "two-rank group (measured VERIFY_MS_TP; replicated head: single-node MTP_COST_H4 draft)".to_string()
            }
            (CostBasis::SingleNode | CostBasis::MeasuredTwoRanks, HeadForm::MixedK) => mixedk_tables_label(basis, split),
            (CostBasis::Unmeasured { ranks }, HeadForm::MixedK) => format!(
                "{}; no group of {ranks} ranks was ever measured on a fabric, the online verify scale (clamped 0.7..1.6) follows the \
                 real rounds",
                mixedk_tables_label(CostBasis::SingleNode, split)
            ),
            (CostBasis::Unmeasured { ranks }, _) => format!(
                "single-node (shipped{}); no group of {ranks} ranks was ever measured on a fabric, the online verify scale (clamped \
                 0.7..1.6) follows the real rounds",
                if head == HeadForm::Exl3 { ", MTP_COST_H4 draft" } else { "" }
            ),
        };
        eprintln!("rexl3 event=tp_costs ranks={} basis={} head={} tables=\"{}\"", t.ranks, basis.label(), head.label(), tables);
        return;
    }
    if o.spec.uses_mtp() {
        eprintln!(
            "rexl3 event=mtp_costs head={} basis=single_node tables=\"{}\"",
            head.label(),
            match head {
                HeadForm::Bf16 => "MTP_COST / MTP_COST_RESTRICTED (measured, bf16 head)".to_string(),
                HeadForm::Exl3 => "MTP_COST_H4 / MTP_COST_RESTRICTED_H4 (4-bit head)".to_string(),
                HeadForm::MixedK => mixedk_tables_label(CostBasis::SingleNode, true),
            }
        );
    }
}

/// The `tables=` of a MixedK pack's cost line (`spec::HeadForm::MixedK`: an EXL3 `lm_head` or EXL3 head attention): its own tables,
/// named `UNMEASURED` while they are placeholders (`spec::MIXEDK_MEASURED`), and for a two-rank group the extrapolation they get.
pub fn mixedk_tables_label(basis: CostBasis, split: bool) -> String {
    let state = if HeadForm::MixedK.measured() {
        "measured on the MixedK form"
    } else {
        "UNMEASURED placeholders: the K=4 trunk's VERIFY_MS and the 4-bit head's MTP_COST_H4 until tests/nemotron_omni_costs.rs \
         measures this form; the online scales (clamped 0.7..1.6) follow the real rounds"
    };
    match basis {
        CostBasis::MeasuredTwoRanks if split => format!(
            "two-rank group (VERIFY_MS_TP scaled per row by VERIFY_MS_MIXEDK / VERIFY_MS; draft MTP_COST_TP scaled per term by the \
             single-node MTP_COST_MIXEDK / MTP_COST ratios: extrapolated; {state})"
        ),
        CostBasis::MeasuredTwoRanks => format!(
            "two-rank group (VERIFY_MS_TP scaled per row by VERIFY_MS_MIXEDK / VERIFY_MS; replicated head: single-node MTP_COST_MIXEDK \
             draft; {state})"
        ),
        _ => format!("VERIFY_MS_MIXEDK, MTP_COST_MIXEDK / MTP_COST_RESTRICTED_MIXEDK ({state})"),
    }
}

/// The boot's one line on where the arbiter's cost tables come from, for a tensor-parallel group (nothing for one node): `basis=
/// measured_two_ranks` for the group the tables were measured on, `basis=unmeasured` for three ranks and more, which are priced with
/// the single node's tables and followed by the arbiter's online scale (clamped to 0.7..1.6).
pub fn log_cost_basis(o: &ServeOptions) {
    let Some(t) = o.tp.as_ref().filter(|t| t.is_group()) else { return };
    let basis = tp_cost_basis(o);
    eprintln!(
        "rexl3 event=tp_costs ranks={} basis={} tables=\"{}\"",
        t.ranks,
        basis.label(),
        match basis {
            CostBasis::SingleNode => "single-node (shipped)".to_string(),
            CostBasis::MeasuredTwoRanks => "two-rank group (measured VERIFY_MS_TP, MTP_COST_TP)".to_string(),
            CostBasis::Unmeasured { ranks } => format!(
                "single-node (shipped); no group of {ranks} ranks was ever measured on a fabric, the online verify scale (clamped \
                 0.7..1.6) follows the real rounds"
            ),
        }
    );
}

/// The speculation settings and cost tables a boot uses: the environment's (`REXL3_MTP_DEPTH`, `REXL3_NEMO_DRAFT_NOISE`,
/// `REXL3_NEMO_ARB_TABLE`), the shipped tables (with `verify` for the verify table when given), and, when the plan priced the head with a
/// restricted sub-table (the default 49,152-id list unless `REXL3_MTP_VOCAB` says otherwise), the restricted head: the arbiter prices a
/// draft call with the lm_head of the rows the plan priced (`ServePlan::sub_head_bytes`, so what is priced, what is allocated and what
/// the arbiter assumes are one count).
pub fn spec_settings(
    o: &ServeOptions,
    sp: &ServePlan,
    verify: Option<&[(usize, f64)]>,
) -> Result<(SpecConfig, CostTables), ModelError> {
    spec_settings_for(o, sp, verify, HeadForm::Bf16)
}

/// [`spec_settings`] for a loaded head of `head`'s form: the draft-call tables are the H4 ones for a 4-bit head
/// ([`CostTables::shipped_for`], [`CostTables::tp_two_ranks_for`]); a bf16 head gets exactly [`spec_settings`]'s.
pub fn spec_settings_for(
    o: &ServeOptions,
    sp: &ServePlan,
    verify: Option<&[(usize, f64)]>,
    head: HeadForm,
) -> Result<(SpecConfig, CostTables), ModelError> {
    let mut cfg = SpecConfig::from_env(o.spec)?;
    // the measured two-rank tables only for the group they were measured on; three ranks and more, a lockstep group of whole models and
    // `REXL3_NEMO_COSTS=single` price with the single node's (`tp_cost_basis`)
    let mut costs = if tp_costs_wanted(o) {
        let h = super::tp_arm::head_split();
        CostTables::tp_two_ranks_for(h.experts && h.lm_head, head)
    } else {
        CostTables::shipped_for(head)
    };
    if let Some(v) = verify {
        costs = costs.with_verify(v);
    }
    if sp.mtp && sp.sub_head_bytes > 0 {
        let rows = (sp.sub_head_bytes / SUB_HEAD_ROW_BYTES) as usize;
        cfg.restricted_head = true;
        costs.mtp_restricted = restricted_costs_for(&costs, rows, head);
    }
    // `REXL3_NEMO_ARB_TABLE=batched` where no batched table was measured (a two-rank group): said once, and the group's own table prices
    if cfg.mode == SpecMode::Auto && cfg.arb_table == ArbTable::Batched && costs.batched.is_none() {
        eprintln!(
            "rexl3 event=arb_table requested=batched used=single reason=\"no batched verify table was measured for this group; the arbiter \
             prices rounds with the group's verify table by rows\""
        );
    }
    Ok((cfg, costs))
}

/// The environment variable that picks the arbiter's cost tables of a two-rank group: unset or `tp` the group's measured tables
/// ([`CostTables::tp_two_ranks`]), `single` the single-node tables (the A/B of `docs/NEMOTRON_TP.md` 9.2). It changes which drafts are
/// offered and never a committed token.
pub const COSTS_ENV: &str = "REXL3_NEMO_COSTS";

/// Whether this boot prices rounds with the two-rank group's MEASURED tables: a group of exactly two ranks (the only size ever measured
/// on a fabric), whose ranks hold shards (not `REXL3_TP_COMM=none`, which runs whole models in lockstep at the single node's speed),
/// and [`COSTS_ENV`] is not `single`. A group of three ranks or more is priced with the single node's tables, labelled `unmeasured`
/// in the boot log ([`tp_cost_basis`]); the arbiter's online scale clamp follows the real rounds.
pub fn tp_costs_wanted(o: &ServeOptions) -> bool {
    matches!(tp_cost_basis(o), CostBasis::MeasuredTwoRanks)
}

/// Where this boot's cost tables come from ([`CostBasis`]), from the options, `REXL3_TP_COMM` and [`COSTS_ENV`]: the measured
/// two-rank tables only for a two-rank sharded group, the single node's for everything else, with the rank count named when the group
/// is larger than anything measured.
pub fn tp_cost_basis(o: &ServeOptions) -> CostBasis {
    let Some(t) = o.tp.as_ref().filter(|t| t.is_group()) else { return CostBasis::SingleNode };
    if !super::tp_boot::sharded() || matches!(std::env::var(COSTS_ENV).as_deref(), Ok("single")) {
        return CostBasis::SingleNode;
    }
    CostBasis::of_ranks(t.ranks)
}

/// The draft-call table for a restricted head of `rows` token ids: the measured table when the head is the default list's size (49,152),
/// otherwise the full head's with the `lm_head` term of that many rows (linear between the two measured sizes: [`MtpCost::restricted_to`]).
pub fn restricted_costs(costs: &CostTables, rows: usize) -> MtpCost {
    if rows == LM_HEAD_RESTRICTED_ROWS {
        costs.mtp_restricted.clone()
    } else {
        costs.mtp.restricted_to(rows)
    }
}

/// [`restricted_costs`] for a head of `head`'s form: a bf16 or 4-bit head exactly that; the MixedK form, whose full `lm_head` is EXL3
/// and whose sub-table is bf16 rows (D2), from its restricted model along the bf16 line (`MtpCost::restricted_sub_table_to`).
pub fn restricted_costs_for(costs: &CostTables, rows: usize, head: HeadForm) -> MtpCost {
    match head {
        HeadForm::Bf16 | HeadForm::Exl3 => restricted_costs(costs, rows),
        HeadForm::MixedK => costs.mtp_restricted.restricted_sub_table_to(rows),
    }
}

/// [`assemble`] with a step cost table of the caller's (rows, milliseconds): the seam where measured points replace [`STEP_MS`], and
/// what a test uses to make drafting worth asking for. The table is the engine's row budget under `--spec copy` and the arbiter's verify
/// table under `auto`; the draft-call tables stay the shipped ones (use [`assemble_with`] to replace them too).
pub fn assemble_with_cost<M: NemotronRuntime>(
    model: M,
    o: &ServeOptions,
    sp: &ServePlan,
    step_ms: &[(usize, f64)],
) -> Result<Engine<NemotronBackend<M>>, ModelError> {
    let head = model.head_form();
    let (cfg, costs) = spec_settings_for(o, sp, Some(step_ms), head)?;
    assemble_with(model, o, sp, cfg, &costs)
}

/// [`assemble`] with the speculation settings and every cost table the caller's: the seam where the measured tables replace the
/// shipped ones without a code change. `cfg.mode` must be `o.spec`.
pub fn assemble_with<M: NemotronRuntime>(
    model: M,
    o: &ServeOptions,
    sp: &ServePlan,
    cfg: SpecConfig,
    costs: &CostTables,
) -> Result<Engine<NemotronBackend<M>>, ModelError> {
    if cfg.mode != o.spec {
        return Err(ModelError::Config(format!(
            "the speculation settings are for --spec {}, the options say --spec {}",
            cfg.mode, o.spec
        )));
    }
    // the loaded model is the final word on whether `--spec mtp` can be honoured (the plan could only ask the pack)
    if o.spec == SpecMode::Mtp && !model.has_mtp() {
        return Err(ModelError::Config(
            "--spec mtp: the loaded model has no MTP head (the pack lacks one, or it was not loaded); use --spec auto, copy or none"
                .into(),
        ));
    }
    // the plan priced the head's cache stripe, tap stores and buffers (`ModelOptions::mtp`) exactly when it was to be loaded: a model that
    // carries a head the plan did not price, and would draft with it, holds pages and buffers the plan never counted
    if o.drafting && o.spec.uses_mtp() && model.has_mtp() && !sp.mtp {
        return Err(ModelError::Config(
            "the model carries an MTP head the memory plan did not price (its KV stripe, tap stores and buffers): load it with the \
             plan's ModelOptions (`mtp`), or start with --spec copy or none"
                .into(),
        ));
    }
    let window = window_rows(o)?;
    let mut backend = NemotronBackend::with_runtime(model, prefill_rows(o), &sp.model)?
        .with_window_rows(window)?
        .with_fused_decode(fused_decode(window)?);
    if let Some(id) = sp.image_token {
        backend = backend.with_image_token(id);
    }
    // the prefix cache the plan carved: the runtime allocates its arena (what the plan counted, to the byte), the backend gets the budget and the
    // namespace, and the engine below is told to use it. A plan with no cache (`plan_serving`, or a runtime that cannot snapshot) skips all of it.
    let cache = sp.plan.cache;
    if cache.enabled() {
        if !M::snapshots_supported() {
            return Err(ModelError::Config(
                "the plan carved a prefix cache, but this runtime cannot snapshot a slot: plan with `plan_serving_for::<M>`"
                    .into(),
            ));
        }
        // an allocation the device refuses although the plan fits is the boot's named refusal, as `build_with` makes it for the load
        backend.m.enable_snapshots(cache.snapshot_slots).map_err(|e| {
            allocation_error("the prefix cache's snapshot arena", e, sp.plan.planned_bytes(sp.slots), o.floor_bytes, None)
        })?;
        let (want, got) = (cache.snapshot_slots as u64 * cache.snapshot_bytes, backend.m.snapshot_arena_bytes());
        if want != got {
            return Err(ModelError::Config(format!(
                "the memory model and the allocation disagree: the plan counted {want} bytes of snapshot arena ({} slots of {} B), the runtime holds {got}",
                cache.snapshot_slots, cache.snapshot_bytes
            )));
        }
        let identity = sp.pack_identity.as_deref().ok_or_else(|| {
            ModelError::Config(
                "the plan carved a prefix cache but names no pack identity (plan with `plan_serving_for::<M>`)".into(),
            )
        })?;
        let ns = prefix::namespace_of(identity, backend.m.has_mtp());
        backend = backend.with_prefix_cache(ns, cache)?;
    }
    let drafting = o.drafting && o.spec.drafts();
    if drafting {
        backend = backend.with_speculation(cfg, costs)?;
    }
    let mm = ModelMemory::from_profile(&sp.profile, 0, 0);
    let plan = &sp.plan;
    let admission = plan.admission(mm, sp.slots, sp.max_context).map_err(|e| ModelError::Config(e.to_string()))?;
    // the engine's own per-sequence drafter is the fallback for a round the backend declines to answer (a failed backend call)
    let make: Box<dyn Fn() -> Box<dyn Drafter>> = if drafting && o.spec.uses_copy() {
        Box::new(|| Box::new(gb_spec::CopyDrafter::new(8, MAX_DRAFTS)))
    } else {
        Box::new(|| Box::new(NoDrafts))
    };
    let cost = row_budget_for(o.spec, drafting, &costs.verify, window);
    boot::mark("kv_planned", Some(gib(plan.kv_tokens * sp.profile.kv_bytes_per_token())));
    let engine = Engine::new(backend, admission, EngineConfig { max_rows: window, max_drafts: MAX_DRAFTS, cost, drafting }, make);
    // a wide boot serves a crowd: the prompts that arrive together are prefilled together (the backend puts their slices through shared
    // forwards, `NemotronBackend::prefill_slices`), instead of one slice a decode round, which would take minutes to seat 128 prompts
    let engine = match prefill_burst(window)? {
        Some(cap) => engine.with_prefill_policy(PrefillPolicy::Burst { div: 1, cap }),
        None => engine,
    };
    Ok(if cache.enabled() { engine.with_prefix_cache(nemotron_prefix_config(&|name| std::env::var(name).ok())?) } else { engine })
}

/// The Nemotron prefix config: the interactive cut policy (cuts that serve a second question about a long document and a short tool
/// loop; the default policy serves neither), then the environment's overrides as in [`crate::serve::prefix_config_from_env`].
fn nemotron_prefix_config(var: &dyn Fn(&str) -> Option<String>) -> Result<gb_engine::PrefixConfig, ModelError> {
    crate::serve::interactive_prefix_config_from(var)
}

/// The environment variable that sets how many waiting prompts one decode round prefills (`0` or `off`: one slice a round, the engine's
/// `Serial` policy); unset, a wide window (more than 64 rows) prefills up to 32 and a narrow one keeps `Serial`.
pub const PREFILL_BURST_VAR: &str = "REXL3_NEMO_PREFILL_BURST";

/// The prompts a round's prefill takes at most, or `None` for one slice a round ([`PREFILL_BURST_VAR`]).
pub fn prefill_burst(window: usize) -> Result<Option<usize>, ModelError> {
    prefill_burst_with(window, std::env::var(PREFILL_BURST_VAR).ok().as_deref())
}

/// [`prefill_burst`] with the value of [`PREFILL_BURST_VAR`] given (`None`: unset).
pub fn prefill_burst_with(window: usize, var: Option<&str>) -> Result<Option<usize>, ModelError> {
    match var.map(str::trim) {
        None => Ok((window > WINDOW_ROWS).then_some(32)),
        Some("0" | "off") => Ok(None),
        Some(v) => match v.parse::<usize>() {
            Ok(n) if n >= 1 => Ok(Some(n)),
            _ => Err(ModelError::Config(format!("{PREFILL_BURST_VAR}={v:?}: expected a number of prompts, or 0 or off"))),
        },
    }
}

/// The engine's own row budget (`EngineConfig::cost`, which trims an offer whose estimated acceptance does not pay for its rows).
/// `--spec auto` hands the decision to the arbiter and `--spec mtp` drafts at the depth it was asked for: the engine's budget must not
/// second-guess either (it knows one acceptance rate per stream, whatever the drafter, and no draft cost), so it is flat: more rows
/// never cost more, and every draft rides. `--spec copy` keeps the step table's trim (`verify`), and so does a run that does not draft.
pub fn row_budget(spec: SpecMode, drafting: bool, verify: &[(usize, f64)]) -> CostModel {
    row_budget_for(spec, drafting, verify, WINDOW_ROWS)
}

/// [`row_budget`] for a decode window of `window_rows` rows (a wide model).
pub fn row_budget_for(spec: SpecMode, drafting: bool, verify: &[(usize, f64)], window_rows: usize) -> CostModel {
    if drafting && matches!(spec, SpecMode::Auto | SpecMode::Mtp) {
        CostModel::from_table(vec![(1, 1.0), (window_rows.max(WINDOW_ROWS), 1.0)])
    } else {
        CostModel::from_table(verify.to_vec())
    }
}

/// A boot: check the options, plan the memory, load the model with the planned options (`load`: the caller's device code, handed the
/// [`ModelOptions`]), assemble the engine. A load that the device refuses for memory is reported with what the plan asked for and what
/// the host had (`serve::allocation_error`); any other error passes through.
pub fn build<M, L>(
    fam: &FamilyPlan,
    ctx: &BuildCtx<'_>,
    inp: &PlanInputs<'_>,
    load: L,
) -> Result<Engine<NemotronBackend<M>>, ModelError>
where
    M: NemotronRuntime,
    L: FnOnce(&ModelOptions) -> Result<M, ModelError>,
{
    build_with(fam, ctx, inp, load, |_| Ok(()))
}

/// [`build`] that hands the plan to `ready` once the engine is assembled (and not before: a boot that failed has published nothing). The
/// family's `build` uses it to put the image encoder, whose features budget is the plan's `vision_reserve_bytes`, behind the server's
/// vision slot only when the engine it serves exists.
pub fn build_with<M, L, R>(
    fam: &FamilyPlan,
    ctx: &BuildCtx<'_>,
    inp: &PlanInputs<'_>,
    load: L,
    ready: R,
) -> Result<Engine<NemotronBackend<M>>, ModelError>
where
    M: NemotronRuntime,
    L: FnOnce(&ModelOptions) -> Result<M, ModelError>,
    R: FnOnce(&ServePlan) -> Result<(), ModelError>,
{
    build_with_plan(fam, ctx, inp, |mo, _| load(mo), ready)
}

/// [`build_with`] whose `load` is also given the whole plan: a tensor-parallel leader needs the prefix cache's budget (`plan.cache`) to
/// build its boot gate, because the ranks must agree on it before any weight is read (`docs/NEMOTRON_TP.md` 12.5). Everything else is
/// [`build_with`]; a single process has no use for it.
pub fn build_with_plan<M, L, R>(
    fam: &FamilyPlan,
    ctx: &BuildCtx<'_>,
    inp: &PlanInputs<'_>,
    load: L,
    ready: R,
) -> Result<Engine<NemotronBackend<M>>, ModelError>
where
    M: NemotronRuntime,
    L: FnOnce(&ModelOptions, &ServePlan) -> Result<M, ModelError>,
    R: FnOnce(&ServePlan) -> Result<(), ModelError>,
{
    let o = ctx.options;
    let sp = plan_serving_for::<M>(fam, o, inp)?;
    if prefill_rows(o) < o.prefill_chunk {
        eprintln!(
            "rexl3 level=info event=prefill_chunk_clamped requested={} granted={} (a prefill slice is at most {} rows)",
            o.prefill_chunk,
            prefill_rows(o),
            prefill_rows_max()
        );
    }
    if sp.slots < sp.requested_slots {
        eprintln!("rexl3 event=slots_shrunk requested={} granted={}", sp.requested_slots, sp.slots);
    }
    eprintln!("{}", hostmem::memory_view_line(inp.cuda_free, inp.host, o.floor_bytes));
    eprintln!(
        "rexl3 event=nemotron_plan weights_gib={:.2} free_after_gib={:.2} slots={} kv_pages={} kv_tokens={} max_context={} spec={} mtp_head_priced={} sub_head_mib={:.0} window_rows={} fused_decode={} lag={} kv_dtype={}",
        gib(inp.weight_bytes),
        gib(sp.free_after_weights),
        sp.slots,
        sp.plan.kv_blocks,
        sp.plan.kv_tokens,
        sp.max_context,
        o.spec,
        sp.mtp,
        sp.sub_head_bytes as f64 / 1048576.0,
        window_rows(o)?,
        fused_decode(window_rows(o)?)?,
        lag_period_for(window_rows(o)?)?,
        gb_cuda::nemotron_attn::configured_kv_dtype().unwrap_or_default().name()
    );
    boot::fact("kv_dtype", gb_cuda::nemotron_attn::configured_kv_dtype().unwrap_or_default().name().to_string());
    boot::fact("plan_free_gib", format!("{:.1}", gib(sp.free_after_weights)));
    boot::fact("slots", sp.slots.to_string());
    boot::fact("kv_pages", sp.plan.kv_blocks.to_string());
    boot::fact("kv_tokens", sp.plan.kv_tokens.to_string());
    boot::fact("spec", o.spec.to_string());
    boot::fact("mtp_head_priced", sp.mtp.to_string());
    if sp.plan.cache.enabled() {
        let c = &sp.plan.cache;
        eprintln!(
            "rexl3 event=prefix_cache_plan entries={} pages={} gib={:.2} spare_gib={:.1} entry_pages={} snapshot_mib={:.1}",
            c.snapshot_slots,
            c.page_blocks,
            gib(c.bytes()),
            gib(sp.plan.spare_bytes),
            prefix::entry_pages(sp.max_context),
            c.snapshot_bytes as f64 / 1048576.0
        );
        boot::fact("prefix_cache", format!("{} entries, {} pages, {:.2} GiB", c.snapshot_slots, c.page_blocks, gib(c.bytes())));
    } else if M::snapshots_supported() {
        // the runtime can snapshot and no cache was planned: the request was `Off`, or `Auto` on a box with no spare for two entries
        let why = match o.prefix_cache.request() {
            CacheRequest::Off => "off",
            _ => "no spare memory for two entries after the KV pool (pass --prefix-cache-mib to carve one out of it)",
        };
        eprintln!("rexl3 event=prefix_cache_plan entries=0 reason=\"{why}\" spare_gib={:.1}", gib(sp.plan.spare_bytes));
        boot::fact("prefix_cache", why);
    }
    let model = load(&sp.model, &sp).map_err(|e| {
        allocation_error(
            "Nemotron weights and engine buffers",
            e,
            inp.weight_bytes.saturating_add(sp.plan.planned_bytes(sp.slots)),
            o.floor_bytes,
            inp.host,
        )
    })?;
    let engine = assemble(model, o, &sp)?;
    ready(&sp)?;
    Ok(engine)
}

/// [`build`] with the backend type erased, which is what `ModelFamily::build` returns.
pub fn build_serve<M, L>(
    fam: &FamilyPlan,
    ctx: &BuildCtx<'_>,
    inp: &PlanInputs<'_>,
    load: L,
) -> Result<Box<dyn ServeEngine>, ModelError>
where
    M: NemotronRuntime + 'static,
    L: FnOnce(&ModelOptions) -> Result<M, ModelError>,
{
    Ok(Box::new(build(fam, ctx, inp, load)?))
}

/// [`build_with_plan`] with the backend type erased.
pub fn build_serve_with_plan<M, L, R>(
    fam: &FamilyPlan,
    ctx: &BuildCtx<'_>,
    inp: &PlanInputs<'_>,
    load: L,
    ready: R,
) -> Result<Box<dyn ServeEngine>, ModelError>
where
    M: NemotronRuntime + 'static,
    L: FnOnce(&ModelOptions, &ServePlan) -> Result<M, ModelError>,
    R: FnOnce(&ServePlan) -> Result<(), ModelError>,
{
    Ok(Box::new(build_with_plan(fam, ctx, inp, load, ready)?))
}

/// [`build_with`] with the backend type erased.
pub fn build_serve_with<M, L, R>(
    fam: &FamilyPlan,
    ctx: &BuildCtx<'_>,
    inp: &PlanInputs<'_>,
    load: L,
    ready: R,
) -> Result<Box<dyn ServeEngine>, ModelError>
where
    M: NemotronRuntime + 'static,
    L: FnOnce(&ModelOptions) -> Result<M, ModelError>,
    R: FnOnce(&ServePlan) -> Result<(), ModelError>,
{
    Ok(Box::new(build_with(fam, ctx, inp, load, ready)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// v0.1.2 P2-9: the restricted draft table of our two forms is what it was for every vocabulary size; the MixedK form's follows its
    /// own restricted model (a bf16 sub-table under an EXL3 lm_head), and its cost line names its tables as unmeasured placeholders.
    #[test]
    fn the_restricted_table_of_a_bf16_or_4_bit_head_is_unchanged_and_the_mixedk_one_is_its_own() {
        for head in [HeadForm::Bf16, HeadForm::Exl3] {
            for split in [true, false] {
                for costs in [CostTables::shipped_for(head), CostTables::tp_two_ranks_for(split, head)] {
                    for rows in [1usize, 5_000, 32_768, LM_HEAD_RESTRICTED_ROWS, 62_017, 131_072] {
                        assert_eq!(restricted_costs_for(&costs, rows, head), restricted_costs(&costs, rows), "{head:?} {rows}");
                    }
                }
            }
        }
        let mk = CostTables::shipped_for(HeadForm::MixedK);
        assert_eq!(restricted_costs_for(&mk, LM_HEAD_RESTRICTED_ROWS, HeadForm::MixedK), mk.mtp_restricted);
        assert_eq!(restricted_costs_for(&mk, 32_768, HeadForm::MixedK), mk.mtp_restricted.restricted_sub_table_to(32_768));
        for (basis, split) in
            [(CostBasis::SingleNode, true), (CostBasis::MeasuredTwoRanks, true), (CostBasis::MeasuredTwoRanks, false)]
        {
            let l = mixedk_tables_label(basis, split);
            assert!(l.contains("MIXEDK"), "{l}");
            assert_eq!(l.contains("UNMEASURED"), !super::super::spec::MIXEDK_MEASURED, "{l}");
            assert!(!l.contains('"'), "{l}");
        }
    }

    /// Nemotron cuts with the interactive policy (the default one serves neither a second question about a document nor a short tool
    /// loop); the environment still overrides each value, and the default config path is unchanged.
    #[test]
    fn nemotron_cuts_for_interactive_use_and_the_environment_still_decides() {
        use gb_sched::prefix::CutPolicy;
        let from = |vars: &[(&str, &str)]| {
            let vars: Vec<(String, String)> = vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            nemotron_prefix_config(&move |n| vars.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone()))
        };
        assert_eq!(from(&[]).unwrap().policy, CutPolicy::interactive());
        assert_ne!(CutPolicy::interactive(), CutPolicy::default());
        let c = from(&[("REXL3_PREFIX_TAIL", "16"), ("REXL3_PREFIX_MIN_CUT", "512")]).unwrap();
        assert_eq!((c.policy.tail, c.policy.min_cut), (16, 512));
        assert_eq!((c.policy.min_gap, c.policy.ghost_min_gain), (128, 256), "what is not overridden stays interactive");
        assert!(from(&[("REXL3_PREFIX_TAIL", "long")]).is_err());
        assert_eq!(crate::serve::prefix_config_from(&|_| None).unwrap(), gb_engine::PrefixConfig::default());
    }
}
