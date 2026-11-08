//! The boot of a MiniCPM-V-4.7 pack with no forward (E2 of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`): scan, bind, validate, plan,
//! load, allocate, verify; nothing runs a kernel.
//!
//! [`boot`] is what the engine's `build` calls (E3): it returns a [`LoadedModel`] (the typed weights on the device, the engine's
//! persistent state allocated at the planned sizes, and a [`BootReport`]) and [`LoadedModel::into_engine`] wraps the model, a
//! `Backend` and the `Engine` around it (`engine::engine_from`). `MiniCpmFamily::build` boots only when every component the text
//! engine needs is built (the server's preflight calls `build` only to learn what is missing, and a 28 GB load is not a preflight).
//!
//! The order is the fail-closed one every family has, so that a pack or a box that cannot serve is refused before it is read:
//!
//! 1. the options and the config's shapes (a draft, images and a window other than 64 rows are refused by the component that is not
//!    built; a config the kernels were not built for is refused by key and value; a truncated layer count only on request);
//! 2. one scan of the shard headers and codebook markers, and the binding of every tensor against the recipe: a missing, misshapen,
//!    misplaced or unknown tensor, a `.mcg` marker, an unpadded head or an `mtp.*` tensor is named, and nothing is read;
//! 3. the pack's fingerprint against the validated list (empty for this model until the full gates have run on a pack): an unknown
//!    pack is refused unless `--allow-unvalidated-pack`;
//! 4. the kernel gates over the shapes the pack will run: the decode plan and the pinned prefill GEMM of every distinct
//!    (K, N), and the MoE launch parameters ([`KC`], hidden 2,048 / intermediate 512);
//! 5. admission: the planner is given `MemAvailable` (rounded down to a GiB) less the weights, which are known to the byte from
//!    the headers, and refuses by name when the weights and the floor do not fit or the pool cannot hold a context per slot;
//! 6. the weights through the pipelined loader, then the KV pool, the delta-net pools and the window buffers at the planned sizes;
//! 7. the check that makes the plan real: the arena carved exactly the planned weight bytes, the claimed buffers sum to the
//!    plan's payload, and every state item holds exactly what `memory::allocations` prices.

use super::config::{DSparkConfig, MiniCpmConfig};
use super::device::{LoadStats, MiniCpmWeights};
use super::inventory;
use super::memory;
use super::prefix;
use super::state::{compare_with_plan_cached, ModelState, StateBytes};
use super::weights::WeightMap;
use crate::error::ModelError;
use crate::hostmem::{self, MemInfo};
use crate::memory::MemoryOptions;
use crate::pack::Pack;
use crate::plan::{choose_slots, plan_memory_cached, MemoryPlan, MIN_TOKENS_PER_SLOT};
use crate::serve::{gib, ServeOptions};
use gb_cuda::fn_moe::COOPMK_SHAPES;
use gb_engine::{boot, ServeEngine};
use gb_sched::MemoryProfile;
use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;

/// Rows of the widest verify window (four streams of 16); the family's window is fixed.
pub const WINDOW_ROWS: usize = memory::WINDOW_ROWS as usize;

/// K chunks of the FrostMoE gate/up pass at hidden 2,048 (`fmoe::DEFAULT_KC` is 10 and invalid here; the sweep that chose 4 is
/// `tests/minicpm_moe.rs` of the kernels).
pub const KC: usize = gb_cuda::fmoe::MINICPM_KC;
const _: () = assert!(gb_cuda::fmoe::kc_valid(2048, KC), "MINICPM_KC must be a valid chunk count at hidden 2048");

/// The component that is the serial forward, prefill, windows and the engine (E3): `model`, `backend`, `engine`.
pub const FORWARD_COMPONENT: &str = "forward_minicpmv4_7";

/// Settings of a boot that are not the server's.
#[derive(Clone, Copy, Debug, Default)]
pub struct BootExtras {
    /// Accept a config with fewer than 40 layers (the converter's smoke packs, whose `layer_types` are cut to N). Never set by the
    /// serving path: the registry's probe refuses such a pack by name.
    pub allow_truncated_layers: bool,
}

// -------------------------------------------------------------------------------------------------------------- the plan

/// A DSpark draft a boot loads beside the target (E4b): its config, parsed and checked against the target before anything is read,
/// the directory it is loaded from (an EXL3 draft pack, or the bf16 checkpoint for the dense reference path: the loader takes whichever
/// the directory holds) and the bytes its weights will take (the shard files, a few MB over what is uploaded).
#[derive(Clone, Debug)]
pub struct DraftPlan {
    pub cfg: DSparkConfig,
    pub dir: std::path::PathBuf,
    pub weight_bytes: u64,
}

/// The admission of a boot, from numbers alone (no device, no file): what the weights take, what is left, and the slot count and
/// pool the planner granted.
#[derive(Clone, Debug)]
pub struct BootPlan {
    pub slots: usize,
    /// What `--max-model-len` and the model's own limit allow.
    pub max_context: usize,
    /// Bytes of the target's weights on the device (the sum of the load plan's regions).
    pub weight_bytes: u64,
    /// `MemAvailable` (or the device's free where the host reports none), rounded down to a GiB.
    pub planning_free: u64,
    /// What the planner was given: `planning_free` less the weights (the draft's included) and the tower's.
    pub free_after_weights: u64,
    /// Bytes of the tower's weights (E5b: loaded after the plan, so taken out of what the planner is given; 0 without images).
    pub vision_weight_bytes: u64,
    pub memory: MemoryPlan,
    /// The draft this boot loads, whose weights `free_after_weights` already pays for and whose pages and overflow pages the
    /// plan's page and slot sizes include.
    pub draft: Option<DraftPlan>,
}

impl BootPlan {
    /// KV pages the pool holds: the admitted share and the prefix cache's extra pages (`--prefix-cache-mib`, `auto`; none without a
    /// cache). Admission is told only about the share (`MemoryPlan::admit_bytes`), so sequences keep exactly the capacity they had.
    pub fn kv_pages(&self) -> usize {
        self.memory.kv_blocks + self.memory.cache.page_blocks
    }

    /// Snapshot slots of the prefix cache (0: no cache), allocated at boot as one arena.
    pub fn snapshot_slots(&self) -> usize {
        self.memory.cache.snapshot_slots
    }
}

/// The options this boot has no component for, refused by naming it.
pub fn check_boot_options(o: &ServeOptions) -> Result<(), ModelError> {
    crate::serve::check_options(o)?;
    if o.window_rows != WINDOW_ROWS {
        return Err(ModelError::Config(format!("MiniCPM-V-4.7 --window-rows must be {WINDOW_ROWS}, got {}", o.window_rows)));
    }
    if o.max_model_len == 0 {
        return Err(ModelError::Config("--max-model-len must be at least 1".into()));
    }
    if o.vision.is_some() {
        // the tower is loaded by `engine::engine_from` (E5b); its launch size is an option of its own, refused by name here, before a
        // file is read, because the memory plan prices it
        super::vision::check_max_patches()?;
    }
    Ok(())
}

/// The DSpark draft this run loads, if any: `--draft DIR` with drafting on and a `--spec` that allows a model drafter (`--spec copy`
/// leaves the pack unloaded, `--spec none` and `--no-drafts` run no drafter). The draft's config is read and refused by name, before a
/// weight is read, when it does not belong to this target (hidden size, vocabulary, layer count, tap layers that are not full-attention
/// layers of the target, mask id) or names something the kernels do not implement.
pub fn draft_for(o: &ServeOptions, cfg: &MiniCpmConfig) -> Result<Option<DraftPlan>, ModelError> {
    if !o.drafting || !o.spec.drafts() {
        return Ok(None);
    }
    let Some(dir) = crate::serve::draft_dir_for(o.spec, o.draft_dir.clone()) else { return Ok(None) };
    let dc = DSparkConfig::from_dir(&dir).map_err(|e| ModelError::Config(format!("--draft {}: {e}", dir.display())))?;
    dc.check_pairing(cfg).map_err(|e| ModelError::Config(format!("--draft {}: {e}", dir.display())))?;
    let problems = super::dspark::config_problems(&dc, cfg.vocab, cfg.layers);
    if !problems.is_empty() {
        return Err(ModelError::Config(format!("--draft {}: draft config: {}", dir.display(), problems.join("; "))));
    }
    let pack = Pack::open(&dir).map_err(|e| ModelError::Config(format!("--draft {}: {e}", dir.display())))?;
    // every tensor of the recipe is there, in its shape (the headers only): a draft with a missing or misshapen tensor is refused here,
    // not after the target's 25 GiB have been read
    let mut tensors = inventory::Tensors::new();
    for (name, e) in &pack.ck.plain {
        tensors.insert(name.clone(), (e.dtype.clone(), e.shape.clone()));
    }
    for (prefix, g) in &pack.ck.groups {
        for suffix in ["trellis", g.in_scales, g.out_scales, g.codebook.name()] {
            let name = format!("{prefix}.{suffix}");
            if let Some(e) = pack.ck.part(&name) {
                tensors.insert(name, (e.dtype.clone(), e.shape.clone()));
            }
        }
    }
    let errs = inventory::check_draft(&dc, &tensors);
    if !errs.is_empty() {
        let head: Vec<&str> = errs.iter().take(6).map(String::as_str).collect();
        return Err(ModelError::Config(format!(
            "--draft {}: {} tensors of the DSpark recipe are wrong or missing: {}{}",
            dir.display(),
            errs.len(),
            head.join("; "),
            if errs.len() > 6 { format!(" (and {} more)", errs.len() - 6) } else { String::new() }
        )));
    }
    let weight_bytes = pack
        .shard_files()
        .iter()
        .map(|f| {
            let path = dir.join(f);
            std::fs::metadata(&path).map(|m| m.len()).map_err(|e| ModelError::Config(format!("{}: {e}", path.display())))
        })
        .sum::<Result<u64, _>>()?;
    Ok(Some(DraftPlan { cfg: dc, dir, weight_bytes }))
}

/// The family's memory model at the serving settings of `o`, validated.
pub fn boot_profile(cfg: &MiniCpmConfig, o: &ServeOptions) -> Result<(MemoryProfile, usize), ModelError> {
    let draft = draft_for(o, cfg)?;
    boot_profile_with(cfg, o, draft.as_ref())
}

/// [`boot_profile`] for a draft already read by [`draft_for`] (the plan reads it once).
pub fn boot_profile_with(
    cfg: &MiniCpmConfig,
    o: &ServeOptions,
    draft: Option<&DraftPlan>,
) -> Result<(MemoryProfile, usize), ModelError> {
    let max_context = o.max_model_len.min(cfg.max_position_embeddings);
    let profile = memory::memory_profile(
        cfg,
        draft.map(|d| &d.cfg),
        &MemoryOptions {
            window_rows: WINDOW_ROWS,
            prefill_rows: o.prefill_chunk,
            max_context,
            mtp: false,
            vision: crate::serve::memory_options(o, false).vision,
            trees: false,
        },
    )
    .map_err(|e| ModelError::Config(e.to_string()))?;
    Ok((profile, max_context))
}

/// What the plan prices for the slots' state, the KV pool's pages and the prefix cache: `planned_bytes` less the engine's reserve and the
/// vision tower's (E5b: the tower's workspace and feature rows, `vision_reserve_bytes`, are allocated by `engine::publish_vision`).
pub(crate) fn priced_state_bytes(m: &MemoryPlan, slots: usize) -> u64 {
    m.planned_bytes(slots) - m.engine_reserve_bytes - m.vision_reserve_bytes
}

/// Admission on a unified-memory box, fail-closed: `cuda_free` and `host` are what the box says now, `weight_bytes` what the headers
/// say the weights take. Refuses by name (a) weights that with the floor do not fit `MemAvailable`, (b) a plan with no room for
/// the slots' state and a page, (c) a pool that cannot hold [`MIN_TOKENS_PER_SLOT`] tokens a slot.
pub fn plan_boot(
    cfg: &MiniCpmConfig,
    o: &ServeOptions,
    weight_bytes: u64,
    cuda_free: u64,
    host: Option<&MemInfo>,
) -> Result<BootPlan, ModelError> {
    let draft = draft_for(o, cfg)?;
    plan_boot_with(cfg, o, draft, weight_bytes, cuda_free, host)
}

/// [`plan_boot`] with the draft already read by [`draft_for`]: what the host tests of the plan with the draft, the tower and the cache
/// together call (the draft's pack is not needed to price it).
pub fn plan_boot_with(
    cfg: &MiniCpmConfig,
    o: &ServeOptions,
    draft: Option<DraftPlan>,
    weight_bytes: u64,
    cuda_free: u64,
    host: Option<&MemInfo>,
) -> Result<BootPlan, ModelError> {
    let (profile, max_context) = boot_profile_with(cfg, o, draft.as_ref())?;
    // the draft's weights are paid for like the target's; its pages are inside the plan's page size and its overflow pages in the
    // slot's state (the profile above was built with its config)
    let draft_bytes = draft.as_ref().map_or(0, |d| d.weight_bytes);
    let planning_free = hostmem::planning_free(cuda_free, host);
    // the tower (E5b) is loaded after the plan is made: its weights come out of what the planner is given, and its workspace and the
    // feature buffers of the images in flight are the plan's vision reserve (they move with the slot count); the draft's weights are
    // paid for like the target's
    let vision_weight_bytes = profile.vision().and_then(|v| v.weight_bytes).unwrap_or(0);
    let resident = weight_bytes.saturating_add(draft_bytes).saturating_add(vision_weight_bytes);
    if planning_free < resident.saturating_add(o.floor_bytes) {
        return Err(ModelError::Config(format!(
            "MiniCPM-V-4.7 weights do not fit: {:.2} GiB of weights{}{} and a {:.1} GiB floor against {:.1} GiB available \
             (MemAvailable, rounded down to a GiB); stop what holds memory or lower --floor-gib{}",
            gib(weight_bytes),
            if draft_bytes > 0 { format!(" plus {:.2} GiB of draft", gib(draft_bytes)) } else { String::new() },
            if vision_weight_bytes > 0 {
                format!(" plus {:.2} GiB of vision tower", gib(vision_weight_bytes))
            } else {
                String::new()
            },
            gib(o.floor_bytes),
            gib(planning_free),
            if vision_weight_bytes > 0 { " (--no-vision serves text without the tower)" } else { "" }
        )));
    }
    let free = planning_free - resident;
    let cap = o.kv_tokens.unwrap_or(max_context).min(max_context) as u64;
    // the tower's reserve (E5b): one workspace and the features of the images a slot may hold, which move with the slot count
    let images = crate::serve::memory_options(o, false).vision;
    // the prefix cache (E6b): an explicit size is carved out before the KV pool is sized, `auto` takes at most half of what the pool
    // leaves (and 8 GiB); the family answers `prefix_cache() == true`, so `serve::start` has already turned it off when it is not wanted
    let snapshot_bytes = prefix::snapshot_bytes(cfg);
    let plan_at = |slots: usize| {
        let reserve = match images {
            Some(v) => profile
                .vision_reserve_bytes(1, v.concurrent_images(slots))
                .map_err(|_| crate::plan::PlanError::Overflow("vision reserve"))?,
            None => 0,
        };
        plan_memory_cached(&profile, free, o.floor_bytes, slots, cap, reserve, o.prefix_cache.request(), snapshot_bytes, 0)
    };
    let choice = choose_slots(o.slots, o.slots_auto, cap, plan_at).map_err(|e| {
        ModelError::Config(format!(
            "MiniCPM-V-4.7 memory plan: {e}; MemAvailable after the {:.2} GiB of weights {:.1} GiB, floor {:.1} GiB",
            gib(resident),
            gib(free),
            gib(o.floor_bytes)
        ))
    })?;
    if choice.plan.kv_tokens < (choice.slots as u64 * MIN_TOKENS_PER_SLOT).min(cap) {
        return Err(ModelError::Config(format!(
            "MiniCPM-V-4.7 KV pool too small for {} slots: {} tokens planned, {} needed ({} a slot)",
            choice.slots,
            choice.plan.kv_tokens,
            (choice.slots as u64 * MIN_TOKENS_PER_SLOT).min(cap),
            MIN_TOKENS_PER_SLOT
        )));
    }
    Ok(BootPlan {
        slots: choice.slots,
        max_context,
        weight_bytes,
        planning_free,
        free_after_weights: free,
        vision_weight_bytes,
        memory: choice.plan,
        draft,
    })
}

// -------------------------------------------------------------------------------------------------------- kernel gates

/// What the kernels do for one EXL3 shape of this model: the decode plan and the pinned prefill tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapePlan {
    pub k: usize,
    pub n: usize,
    /// The bit widths the pack runs this shape at.
    pub bits: BTreeSet<usize>,
    /// `(K splits, warps a block)` of the decode path.
    pub decode: (usize, usize),
    /// `(tile, stages, custom option)` of the pinned prefill GEMM (cublasLt enum values).
    pub prefill: (u32, u32, u32),
}

/// Row counts the pinned prefill GEMM is checked at: every decode window row count, and the prefill chunk.
fn pin_rows(prefill_rows: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (1..=WINDOW_ROWS).collect();
    v.push(prefill_rows.max(1));
    v.sort_unstable();
    v.dedup();
    v
}

/// Bytes of FrostMoE's workspace for a full window at this model's MoE shape and [`KC`] (0 for a shape the kernel is not built for):
/// what the forward allocates once, and the number the boot reports so the engine reserve can be read against it.
pub fn fmoe_workspace_bytes(cfg: &MiniCpmConfig) -> usize {
    gb_cuda::fmoe::workspace_bytes(WINDOW_ROWS, cfg.experts_per_tok, cfg.hidden, cfg.moe_intermediate, cfg.kc())
}

/// The kernel gates over the pack's shapes: every (K, N) has a decode plan and a pinned prefill GEMM the cuBLAS here runs at every
/// row count the engine will use, and the MoE shape (hidden 2,048, intermediate 512, [`KC`]) is one the kernels are built for. A
/// shape the kernels refuse stops the boot before the weights are read.
pub fn kernel_gates(
    cfg: &MiniCpmConfig,
    shapes: &BTreeSet<(usize, usize, usize)>,
    prefill_rows: usize,
) -> Result<Vec<ShapePlan>, ModelError> {
    if !COOPMK_SHAPES.contains(&(cfg.hidden, cfg.moe_intermediate)) {
        return Err(ModelError::Config(format!(
            "MoE hidden {} / intermediate {}: the grouped-expert kernels are built for {:?}",
            cfg.hidden, cfg.moe_intermediate, COOPMK_SHAPES
        )));
    }
    let kc = cfg.kc();
    if !gb_cuda::fmoe::kc_valid(cfg.hidden, kc) {
        return Err(ModelError::Config(format!("K chunks {kc} are not valid at hidden {}", cfg.hidden)));
    }
    if fmoe_workspace_bytes(cfg) == 0 {
        return Err(ModelError::Config(format!(
            "FrostMoE has no workspace for a {WINDOW_ROWS}-row window at top-{}, hidden {}, intermediate {} and {kc} K chunks",
            cfg.experts_per_tok, cfg.hidden, cfg.moe_intermediate
        )));
    }
    gb_cuda::exl3_linear::prefill_init()?;
    let rows = pin_rows(prefill_rows);
    let mut by_shape: std::collections::BTreeMap<(usize, usize), BTreeSet<usize>> = Default::default();
    for &(k, n, bits) in shapes {
        by_shape.entry((k, n)).or_default().insert(bits);
    }
    let mut out = Vec::new();
    for ((k, n), bits) in by_shape {
        let decode = gb_cuda::exl3_linear::decode_plan(k, n).map_err(|e| {
            ModelError::Config(format!("EXL3 linear {k} x {n}: the decode kernel has no plan for this shape ({e})"))
        })?;
        let prefill = gb_cuda::exl3_linear::prefill_pin_for(k, n)
            .map_err(|e| ModelError::Config(format!("EXL3 linear {k} x {n}: no pinned prefill GEMM is defined for it ({e})")))?;
        for &m in &rows {
            gb_cuda::exl3_linear::prefill_pin_check(m, k, n).map_err(|e| {
                ModelError::Config(format!(
                    "the pinned prefill GEMM ({}) does not run a {m} x {k} x {n} problem here ({e}); row-invariant prefill cannot be \
                     guaranteed on this cuBLAS",
                    gb_cuda::exl3_linear::prefill_pin_name()
                ))
            })?;
        }
        out.push(ShapePlan { k, n, bits, decode, prefill });
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------------------------- reports

/// Seconds spent in each phase of the boot.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseSeconds {
    pub scan: f64,
    pub bind: f64,
    pub validate: f64,
    pub cuda_init: f64,
    pub gates: f64,
    pub plan: f64,
    pub weights: f64,
    pub state: f64,
    pub total: f64,
}

/// What a boot held and how long it took.
#[derive(Clone, Debug)]
pub struct BootReport {
    pub pack: String,
    pub layers: usize,
    pub tensors: usize,
    pub groups: usize,
    pub bit_histogram: std::collections::BTreeMap<String, usize>,
    pub fingerprint: String,
    pub validated: Option<&'static str>,
    pub seconds: PhaseSeconds,
    /// Weight bytes: planned (`WeightMap::upload_bytes`), the loader's arena as carved, and the buffers' payload.
    pub weights_planned: u64,
    pub weights_arena_used: u64,
    pub weights_arena_reserved: u64,
    pub weights_payload: u64,
    pub pointer_tables: u64,
    /// The tower and the projector the pack carries (recorded, not loaded).
    pub tower_tensors: usize,
    pub tower_bytes: u64,
    /// The state, as the device holds it, and the plan's reserve it is carved from.
    pub state: StateBytes,
    pub engine_reserve: u64,
    pub slots: usize,
    /// The whole pool: the admitted share and the prefix cache's extra pages.
    pub kv_pages: usize,
    pub kv_tokens: u64,
    /// The prefix cache's snapshot slots and extra pages (0, 0: no cache).
    pub snapshot_slots: usize,
    pub cache_pages: usize,
    /// `MemAvailable` before the load and after the state, and the device's free at the same two points.
    pub host_before: Option<MemInfo>,
    pub host_after: Option<MemInfo>,
    pub cuda_free_before: u64,
    pub cuda_free_after: u64,
    pub shapes: Vec<ShapePlan>,
    /// FrostMoE's workspace at a full window and [`KC`]: allocated by the forward, inside the engine reserve.
    pub fmoe_workspace_bytes: usize,
}

impl BootReport {
    /// Bytes the device holds for this boot beyond the CUDA context: the weights' arena blocks, the pointer tables and the state.
    pub fn resident_bytes(&self) -> u64 {
        self.weights_arena_reserved + self.pointer_tables + self.state.total()
    }

    /// What `MemAvailable` fell by across the load and the state allocation (`None` where the host reports none).
    pub fn host_drop(&self) -> Option<i64> {
        let (b, a) = (self.host_before.as_ref()?, self.host_after.as_ref()?);
        Some(b.available as i64 - a.available as i64)
    }

    /// The `rexl3 event=minicpm_boot` line.
    pub fn line(&self) -> String {
        let s = &self.seconds;
        format!(
            "rexl3 event=minicpm_boot layers={} tensors={} groups={} slots={} kv_pages={} kv_tokens={} prefix_entries={} weights_gib={:.3} \
             arena_gib={:.3} state_gib={:.3} resident_gib={:.3} memavail_drop_gib={} scan_s={:.2} bind_s={:.2} validate_s={:.2} cuda_s={:.2} \
             gates_s={:.2} plan_s={:.2} weights_s={:.2} state_s={:.2} total_s={:.2} fmoe_workspace_mib={:.1} validated={}",
            self.layers,
            self.tensors,
            self.groups,
            self.slots,
            self.kv_pages,
            self.kv_tokens,
            self.snapshot_slots,
            gib(self.weights_planned),
            gib(self.weights_arena_reserved),
            gib(self.state.total()),
            gib(self.resident_bytes()),
            self.host_drop().map_or("n/a".to_string(), |d| format!("{:.3}", d as f64 / (1u64 << 30) as f64)),
            s.scan,
            s.bind,
            s.validate,
            s.cuda_init,
            s.gates,
            s.plan,
            s.weights,
            s.state,
            s.total,
            self.fmoe_workspace_bytes as f64 / (1 << 20) as f64,
            self.validated.unwrap_or("no"),
        )
    }
}

// ------------------------------------------------------------------------------------------------------------- the boot

/// A booted model: the weights on the device, the engine's persistent state at the planned sizes, and the report. No forward.
pub struct LoadedModel {
    pub cfg: MiniCpmConfig,
    pub weights: MiniCpmWeights,
    pub state: ModelState,
    pub plan: BootPlan,
    pub report: BootReport,
}

impl LoadedModel {
    /// Valid ids of the head's output; the columns from here to [`LoadedModel::logits_width`] are padding every consumer masks.
    pub fn valid_vocab(&self) -> usize {
        self.weights.vocab
    }

    /// Width of one row of logits (the padded vocabulary).
    pub fn logits_width(&self) -> usize {
        self.weights.lm_head_width
    }

    /// The capability line of this build: what is built and gated, and what is not.
    pub fn capability_line(&self) -> String {
        format!(
            "minicpmv4_7: weights loader and text forward built and booted ({} tensors bound, {:.2} GiB weights, {} slots, {} KV \
             pages); images and video built (vision_minicpmv, loaded with the server's vision); prefix cache built ({} snapshot slots); \
             built: DSpark draft (--draft DIR, E4b)",
            self.report.tensors,
            gib(self.report.weights_planned),
            self.report.slots,
            self.report.kv_pages,
            self.report.snapshot_slots
        )
    }

    /// The engine around this model (E3): the model on these weights and this state, the backend, the admission and the engine
    /// (`engine::engine_from`).
    pub fn into_engine(self, o: &ServeOptions) -> Result<Box<dyn ServeEngine>, ModelError> {
        Ok(Box::new(super::engine::engine_from(self, o)?))
    }
}

/// Boot the pack named by `o.model_dir` (see the module's order). The serving path's settings: a 40-layer pack only.
pub fn boot(o: &ServeOptions) -> Result<LoadedModel, ModelError> {
    boot_with(o, BootExtras::default())
}

/// [`boot`] with [`BootExtras`].
pub fn boot_with(o: &ServeOptions, extras: BootExtras) -> Result<LoadedModel, ModelError> {
    let t_total = Instant::now();
    let mut sec = PhaseSeconds::default();
    check_boot_options(o)?;
    let cfg = MiniCpmConfig::from_dir(&o.model_dir).map_err(ModelError::Config)?;
    cfg.check_shapes_for(extras.allow_truncated_layers).map_err(ModelError::Config)?;
    if cfg.quant_method != "exl3" {
        return Err(ModelError::Config(format!("quantization_config.quant_method {}: not an EXL3 pack", cfg.quant_method)));
    }

    // one scan of the shard headers and the codebook markers; the bind names every problem it finds, a refused group included
    let t = Instant::now();
    let ck = gb_exl3::checkpoint::scan(&o.model_dir, true)
        .map_err(|e| ModelError::Config(format!("{}: {e}", o.model_dir.display())))?;
    sec.scan = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let map = WeightMap::from_checkpoint(&ck, &cfg).map_err(|errs| refusal(&errs))?;
    sec.bind = t.elapsed().as_secs_f64();
    let pack = Pack { ck };
    boot::mark("packs_scanned", None);

    // the validated list; the smoke packs and every other conversion boot only on the operator's word
    let t = Instant::now();
    let fingerprint = crate::fingerprint::fingerprint(&pack.ck);
    // with a draft the PAIR is what the validated list names (target and draft), as for every family
    let draft_pack = match draft_for(o, &cfg)? {
        Some(d) => Some(Pack::open(&d.dir).map_err(|e| ModelError::Config(format!("--draft {}: {e}", d.dir.display())))?),
        None => None,
    };
    let draft_fingerprint = draft_pack.as_ref().map(|p| crate::fingerprint::fingerprint(&p.ck));
    crate::serve::check_packs(&pack, draft_pack.as_ref(), o.allow_unvalidated_pack)?;
    let validated = crate::fingerprint::validated_pack(&fingerprint, draft_fingerprint.as_deref());
    sec.validate = t.elapsed().as_secs_f64();
    boot::mark("packs_checked", None);

    let t = Instant::now();
    crate::shapes::check_driver(&gb_cuda::rt::versions()).map_err(ModelError::Config)?;
    gb_cuda::rt::init_context()?;
    sec.cuda_init = t.elapsed().as_secs_f64();
    boot::mark("cuda_ready", None);
    crate::serve::boot_checkpoint()?;

    // the kernel gates over the shapes this pack runs, before a weight is read
    let t = Instant::now();
    let shapes: BTreeSet<(usize, usize, usize)> =
        map.groups.values().map(|g| (g.k, g.n, (g.bits.twice() / 2) as usize)).collect();
    let shape_plans = kernel_gates(&cfg, &shapes, o.prefill_chunk)?;
    sec.gates = t.elapsed().as_secs_f64();

    // admission from what the box says now and what the headers say the weights take
    let t = Instant::now();
    let cuda_free = gb_cuda::rt::mem_info()?.0 as u64;
    let host_before = MemInfo::read();
    eprintln!("{}", hostmem::memory_view_line(cuda_free, host_before.as_ref(), o.floor_bytes));
    let plan = plan_boot(&cfg, o, map.upload_bytes(), cuda_free, host_before.as_ref())?;
    sec.plan = t.elapsed().as_secs_f64();
    eprintln!(
        "rexl3 event=minicpm_plan weights_gib={:.2} free_after_gib={:.2} slots={} kv_pages={} kv_tokens={} reserve_gib={:.2}",
        gib(plan.weight_bytes),
        gib(plan.free_after_weights),
        plan.slots,
        plan.kv_pages(),
        plan.memory.kv_tokens,
        gib(plan.memory.engine_reserve_bytes)
    );
    boot::fact("plan_free_gib", format!("{:.1}", gib(plan.free_after_weights)));
    boot::fact("slots", plan.slots.to_string());
    boot::fact("kv_pages", plan.kv_pages().to_string());
    boot::fact("kv_tokens", plan.memory.kv_tokens.to_string());
    if plan.memory.cache.enabled() {
        eprintln!(
            "rexl3 event=minicpm_prefix_plan entries={} pages={} gib={:.2} snapshot_mib={:.1}",
            plan.memory.cache.snapshot_slots,
            plan.memory.cache.page_blocks,
            gib(plan.memory.cache.bytes()),
            plan.memory.cache.snapshot_bytes as f64 / (1 << 20) as f64
        );
        boot::fact(
            "prefix_cache",
            format!("{} entries, {:.2} GiB", plan.memory.cache.snapshot_slots, gib(plan.memory.cache.bytes())),
        );
    } else {
        let reason = match o.prefix_cache {
            crate::serve::PrefixCacheOption::Off | crate::serve::PrefixCacheOption::Mib(0) => "off",
            _ => "insufficient spare memory for two entries",
        };
        eprintln!("rexl3 event=minicpm_prefix_plan entries=0 reason=\"{reason}\"");
    }
    boot::mark("kv_planned", Some(gib(plan.memory.kv_tokens * plan.memory.block_bytes / crate::kvtables::BLOCK_TOKENS as u64)));
    let planned_beyond_weights = plan.memory.planned_bytes(plan.slots);

    // the weights, through the pipelined loader
    let t = Instant::now();
    let (weights, stats) = MiniCpmWeights::load_from(&pack, &map, &cfg, |_, _| {}).map_err(|e| {
        crate::serve::allocation_error(
            "MiniCPM-V-4.7 weights",
            e,
            plan.weight_bytes.saturating_add(planned_beyond_weights),
            o.floor_bytes,
            host_before.as_ref(),
        )
    })?;
    sec.weights = t.elapsed().as_secs_f64();
    boot::mark("weights_loaded", Some(gib(plan.weight_bytes)));
    crate::serve::boot_checkpoint()?;
    verify_weights(&map, &weights, &stats)?;

    // the state at the planned sizes
    let t = Instant::now();
    let state = ModelState::alloc_cached(&cfg, plan.slots, plan.kv_pages(), WINDOW_ROWS, plan.snapshot_slots()).map_err(|e| {
        crate::serve::allocation_error(
            "MiniCPM-V-4.7 KV pool and recurrent state",
            e,
            plan.weight_bytes.saturating_add(planned_beyond_weights),
            o.floor_bytes,
            host_before.as_ref(),
        )
    })?;
    sec.state = t.elapsed().as_secs_f64();
    let held = state.bytes();
    let diffs = compare_with_plan_cached(
        &memory::allocations(&cfg, None),
        &held,
        plan.slots,
        plan.kv_pages(),
        WINDOW_ROWS,
        plan.snapshot_slots(),
    );
    if !diffs.is_empty() {
        return Err(ModelError::Config(format!("MiniCPM-V-4.7 plan and allocation disagree: {}", diffs.join("; "))));
    }
    // the plan's own figures for what it carved out of the reserve: the state it prices is the state the device holds, the prefix
    // cache's snapshots and pages included (`MemoryPlan::planned_bytes` less the engine reserve and the tower's: the tower is loaded
    // by `engine_from`, after this check, and priced against its own allocations there)
    let priced = priced_state_bytes(&plan.memory, plan.slots);
    // with a draft the plan also prices its pool and overflow pages (the page size and the slot's state), which the engine builder
    // allocates and compares item by item
    let draft_pool = plan.draft.as_ref().map_or(0, |d| memory::draft_pool_bytes(&cfg, &d.cfg, plan.kv_pages(), plan.slots));
    let slot_and_page_bytes = held.gdn_state + held.gdn_conv_history + held.kv_target + held.prefix_snapshots + draft_pool;
    if priced != slot_and_page_bytes {
        return Err(ModelError::Config(format!(
            "MiniCPM-V-4.7 memory plan prices {priced} B of slot state and KV pages, the device holds {slot_and_page_bytes} B"
        )));
    }
    if held.gdn_records + held.gdn_raw_qkv > plan.memory.engine_reserve_bytes {
        return Err(ModelError::Config(format!(
            "MiniCPM-V-4.7 window records {} B exceed the plan's engine reserve {} B",
            held.gdn_records + held.gdn_raw_qkv,
            plan.memory.engine_reserve_bytes
        )));
    }
    sec.total = t_total.elapsed().as_secs_f64();

    let report = BootReport {
        pack: o.model_dir.display().to_string(),
        layers: cfg.layers,
        tensors: map.slots.len(),
        groups: map.groups.len(),
        bit_histogram: map.bit_histogram(),
        fingerprint,
        validated,
        seconds: sec,
        weights_planned: map.upload_bytes(),
        weights_arena_used: stats.arena_used,
        weights_arena_reserved: stats.arena_reserved,
        weights_payload: map.payload_bytes(),
        pointer_tables: stats.table_bytes,
        tower_tensors: weights.tower.tensors,
        tower_bytes: weights.tower.bytes,
        state: held,
        engine_reserve: plan.memory.engine_reserve_bytes,
        slots: plan.slots,
        kv_pages: plan.kv_pages(),
        kv_tokens: plan.memory.kv_tokens,
        snapshot_slots: plan.snapshot_slots(),
        cache_pages: plan.memory.cache.page_blocks,
        host_before,
        host_after: MemInfo::read(),
        cuda_free_before: cuda_free,
        cuda_free_after: gb_cuda::rt::mem_info().map(|m| m.0 as u64).unwrap_or(0),
        shapes: shape_plans,
        fmoe_workspace_bytes: fmoe_workspace_bytes(&cfg),
    };
    eprintln!("{}", report.line());
    boot::mark("ready", None);
    Ok(LoadedModel { cfg, weights, state, plan, report })
}

/// The check that makes the weights plan real: the arena carved the planned bytes, and the claimed buffers sum to the plan's payload
/// (and the pointer tables). The serial reference loader has no arena; its buffers still have to sum.
fn verify_weights(map: &WeightMap, w: &MiniCpmWeights, stats: &LoadStats) -> Result<(), ModelError> {
    if stats.report.is_some() && stats.arena_used != map.upload_bytes() {
        return Err(ModelError::Config(format!(
            "MiniCPM-V-4.7 weights plan and allocation disagree: the arena carved {} B, the load plan has {} B",
            stats.arena_used,
            map.upload_bytes()
        )));
    }
    let planned_payload = map.payload_bytes() + stats.table_bytes;
    if w.bytes as u64 != planned_payload {
        return Err(ModelError::Config(format!(
            "MiniCPM-V-4.7 weights plan and allocation disagree: the claimed buffers hold {} B, the load plan's payload and pointer \
             tables are {planned_payload} B",
            w.bytes
        )));
    }
    Ok(())
}

/// The refusal of a pack: how many problems, and the first of them by name.
fn refusal(errs: &[String]) -> ModelError {
    let head: Vec<&str> = errs.iter().take(8).map(String::as_str).collect();
    let more = if errs.len() > 8 { format!(" (and {} more)", errs.len() - 8) } else { String::new() };
    ModelError::Config(format!("MiniCPM-V-4.7 pack: {} problems: {}{more}", errs.len(), head.join("; ")))
}

/// The header-only verdict on a pack directory, for the capability report of `build`: `None` for a directory with no weight shard
/// (a config alone), else what the bind says. Reads headers only; never a weight, and not the 30,971 codebook markers either (the
/// scattered 4-byte reads that make the boot's scan cost seconds on a cold disk): which marker a group carries is its name, and its
/// value is the boot's to check. `serve::preflight` and `start` both reach `build`, so this must stay cheap.
pub fn pack_verdict(dir: &Path) -> Option<String> {
    let has_shards =
        std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).any(|e| e.path().extension().is_some_and(|x| x == "safetensors"));
    if !has_shards {
        return None;
    }
    let cfg = match MiniCpmConfig::from_dir(dir) {
        Ok(c) => c,
        Err(e) => return Some(format!("pack: config unreadable ({e})")),
    };
    let ck = match gb_exl3::checkpoint::scan(dir, false) {
        Ok(ck) => ck,
        Err(e) => return Some(format!("pack: shards unreadable ({e})")),
    };
    Some(match WeightMap::from_checkpoint(&ck, &cfg) {
        Ok(map) => format!(
            "pack: {} tensors bind to the recipe, {:.2} GiB of weights, the {} tower tensors ({:.2} GiB) recorded",
            map.slots.len(),
            gib(map.upload_bytes()),
            map.load_plan().recorded.len(),
            gib(map.load_plan().recorded_bytes())
        ),
        Err(errs) => {
            let head: Vec<&str> = errs.iter().take(3).map(String::as_str).collect();
            format!("pack: {} problems: {}", errs.len(), head.join("; "))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::real;
    use super::super::inventory;
    use super::*;

    const TABLE: &str = include_str!("../../../../../testdata/minicpmv4_7/tensor_inventory.tsv");
    const GIB: u64 = 1 << 30;

    fn weights() -> u64 {
        WeightMap::bind(&real(), &inventory::expand_table(TABLE).unwrap()).unwrap().upload_bytes()
    }

    fn host(avail_gib: u64) -> MemInfo {
        MemInfo { total: 121 * GIB, free: avail_gib * GIB / 4, available: avail_gib * GIB, cached: 0 }
    }

    fn opts(slots: usize) -> ServeOptions {
        let mut o = ServeOptions::new("/nonexistent".into());
        o.slots = slots;
        o.prefill_chunk = 512;
        // the plan tests of the pool alone; the cache's plan is `the_prefix_cache_is_carved_out_...`
        o.prefix_cache = crate::serve::PrefixCacheOption::Off;
        o
    }

    #[test]
    fn the_nominal_pack_is_about_seventeen_and_a_half_gib_of_weights() {
        // experts at 4 bits (30,720 matrices of 524,288 B of trellis and 5,120 B of scales), the rest at 6, the embedding bf16
        let w = weights();
        let experts = 30_720 * (524_288 + 4096 + 1024);
        assert!(w > experts);
        assert!((17 * GIB..18 * GIB).contains(&w), "{:.2} GiB", gib(w));
        // the rest is the embedding (1.02 GB), the head (0.38 GB) and 40 layers of trunk at 6 bits
        assert!(w - experts > (1_016_397_824 + 381_190_144) as u64);
    }

    #[test]
    fn admission_grants_the_pool_of_one_context_at_one_to_eight_slots() {
        let cfg = real();
        let w = weights();
        for slots in 1..=8usize {
            let p = plan_boot(&cfg, &opts(slots), w, 90 * GIB, Some(&host(113))).unwrap();
            assert_eq!(p.slots, slots);
            assert_eq!(p.planning_free, 113 * GIB);
            assert_eq!(p.free_after_weights, 113 * GIB - w);
            // the pool holds the whole context of one stream and a page of slack a slot; admission sees exactly it
            assert_eq!(p.memory.kv_blocks, 262_144 / 256 + slots);
            assert_eq!(p.kv_pages(), 262_144 / 256 + slots);
            assert_eq!(p.memory.kv_tokens, 262_144);
            assert_eq!((p.memory.cache.page_blocks, p.snapshot_slots()), (0, 0), "the cache is off in these options");
            assert_eq!(p.memory.state_bytes_per_slot, 64_389_120);
            assert_eq!(p.memory.block_bytes, 256 * 20_480);
            // and what it plans to allocate stays inside what it was given
            assert!(p.memory.planned_bytes(slots) <= p.free_after_weights - opts(slots).floor_bytes);
        }
    }

    /// E6b: the prefix cache's bytes are planned and allocated with the pool. An explicit size is carved out first (the pool keeps
    /// the plan it had without a cache, shrunk only by what the size takes), `auto` only spends what the pool left, a size too small
    /// for two entries is refused by name, and at every slot count the plan stays inside what it was given and prices the arena.
    #[test]
    fn the_prefix_cache_is_carved_out_priced_and_never_makes_the_admitted_pool_smaller_under_auto() {
        use crate::serve::PrefixCacheOption::{Auto, Mib, Off};
        let cfg = real();
        let w = weights();
        let unit = prefix::snapshot_bytes(&cfg) + 8 * prefix::page_bytes(&cfg);
        for slots in 1..=8usize {
            let base = plan_boot(&cfg, &opts(slots), w, 90 * GIB, Some(&host(113))).unwrap();
            // auto: at most half of what the pool and the slots left, and at most 8 GiB; the admitted pool is exactly what it was
            let o = ServeOptions { prefix_cache: Auto, ..opts(slots) };
            let p = plan_boot(&cfg, &o, w, 90 * GIB, Some(&host(113))).unwrap();
            assert_eq!(p.memory.kv_blocks, base.memory.kv_blocks, "{slots} slots: the cache never makes the KV plan smaller");
            assert_eq!(p.memory.kv_tokens, base.memory.kv_tokens);
            assert!(p.memory.cache.enabled() && p.memory.cache.bytes() <= 8 * GIB, "{slots} slots");
            assert_eq!(p.kv_pages(), p.memory.kv_blocks + p.memory.cache.page_blocks);
            assert_eq!(p.snapshot_slots(), p.memory.cache.snapshot_slots);
            assert_eq!(p.memory.cache.snapshot_bytes, 64_389_120);
            assert_eq!(p.memory.cache.page_bytes, 256 * 20_480);
            assert!(p.memory.planned_bytes(slots) <= p.free_after_weights - o.floor_bytes, "{slots} slots");
            // an explicit size: whole units of a snapshot and eight pages, taken out before the pool is sized
            let o = ServeOptions { prefix_cache: Mib(5_000), ..opts(slots) };
            let p = plan_boot(&cfg, &o, w, 90 * GIB, Some(&host(113))).unwrap();
            let units = (5_000u64 << 20) / unit;
            assert_eq!((p.snapshot_slots() as u64, p.memory.cache.page_blocks as u64), (units, units * 8), "{slots} slots");
            assert_eq!(p.memory.cache.bytes(), units * unit);
            assert_eq!(
                p.memory.planned_bytes(slots),
                base.memory.engine_reserve_bytes
                    + slots as u64 * 64_389_120
                    + p.memory.kv_blocks as u64 * p.memory.block_bytes
                    + units * unit
            );
            assert!(p.memory.planned_bytes(slots) <= p.free_after_weights - o.floor_bytes);
            // off, and a size of nothing, are no cache
            for off in [Off, Mib(0)] {
                let p =
                    plan_boot(&cfg, &ServeOptions { prefix_cache: off, ..opts(slots) }, w, 90 * GIB, Some(&host(113))).unwrap();
                assert_eq!((p.snapshot_slots(), p.memory.cache.page_blocks, p.kv_pages()), (0, 0, p.memory.kv_blocks));
            }
        }
        // one MiB under two units is refused by name (a one-entry cache evicts itself), and the refusal says what to give
        let o = ServeOptions { prefix_cache: Mib(2 * unit / (1 << 20)), ..opts(4) };
        let e = plan_boot(&cfg, &o, w, 90 * GIB, Some(&host(113))).unwrap_err().to_string();
        assert!(e.contains("--prefix-cache-mib") && e.contains("cannot hold two entries"), "{e}");
    }

    #[test]
    fn weights_that_do_not_fit_are_refused_by_name_before_anything_is_planned() {
        let cfg = real();
        let w = weights();
        // MemAvailable below the weights plus the floor: the refusal quotes both figures
        let e = plan_boot(&cfg, &opts(4), w, 90 * GIB, Some(&host(24))).unwrap_err().to_string();
        assert!(e.contains("MiniCPM-V-4.7 weights do not fit") && e.contains("24.0 GiB available"), "{e}");
        // the device's own figure is used where the host reports none
        let e = plan_boot(&cfg, &opts(4), w, 20 * GIB, None).unwrap_err().to_string();
        assert!(e.contains("20.0 GiB available"), "{e}");
        // MemAvailable is rounded down to a GiB before it is compared: one byte short of a GiB boundary is the GiB below
        let need = (w.div_ceil(GIB) + 8) * GIB;
        let just_short = MemInfo { available: need - 1, ..host(0) };
        let o = ServeOptions { floor_bytes: 8 * GIB, ..opts(1) };
        let e = plan_boot(&cfg, &o, w, 90 * GIB, Some(&just_short)).unwrap_err().to_string();
        assert!(e.contains("weights do not fit"), "{e}");
    }

    #[test]
    fn a_pool_that_cannot_hold_a_context_a_slot_is_refused_and_a_default_slot_count_shrinks() {
        let cfg = real();
        let w = weights();
        let first_fit = |slots: usize| -> (u64, BootPlan) {
            let o = opts(slots);
            let start = w.div_ceil(GIB) + 8;
            for avail in start..start + 40 {
                match plan_boot(&cfg, &o, w, 0, Some(&host(avail))) {
                    Ok(p) => return (avail, p),
                    Err(e) => assert!(e.to_string().contains("MiniCPM-V-4.7"), "{e}"),
                }
            }
            panic!("no plan up to {} GiB", start + 40);
        };
        // eight slots need the floor (8 GiB), the reserve (5), their state (8 x 61.4 MiB) and a pool of 8 x 4,096 tokens
        // (128 pages of 5 MiB): the edge is within the next GiB after about 14.1 GiB over the weights
        let (avail, p) = first_fit(8);
        let over = avail * GIB - w;
        assert!((14 * GIB..17 * GIB).contains(&over), "{:.2} GiB over the weights", gib(over));
        assert_eq!(p.slots, 8);
        assert!(p.memory.kv_tokens >= 8 * MIN_TOKENS_PER_SLOT);
        // a slot count left at its default shrinks to the count that fits, and the pool it keeps holds a context a slot
        let o = ServeOptions { slots_auto: true, ..opts(8) };
        let p = plan_boot(&cfg, &o, w, 0, Some(&host(avail - 1))).unwrap();
        assert!((1..8).contains(&p.slots), "{}", p.slots);
        assert!(p.memory.kv_tokens >= (p.slots as u64 * MIN_TOKENS_PER_SLOT).min(262_144));
        // one slot fits at a smaller box than eight
        let (a1, p1) = first_fit(1);
        assert!(a1 < avail && p1.slots == 1);
        // nothing fits at the weights, the floor and nothing else
        let e = plan_boot(&cfg, &opts(1), w, 0, Some(&host(w.div_ceil(GIB) + 8))).unwrap_err().to_string();
        assert!(e.contains("MiniCPM-V-4.7"), "{e}");
    }

    #[test]
    fn options_the_boot_has_no_component_for_are_refused_by_name() {
        let mut o = opts(1);
        assert!(check_boot_options(&o).is_ok());
        o.window_rows = 128;
        let e = check_boot_options(&o).unwrap_err().to_string();
        assert!(e.contains("--window-rows must be 64"), "{e}");
        // a draft is built (E4b): the option check passes, and the draft's own config is what refuses a directory that holds none
        let mut o = opts(1);
        o.draft_dir = Some("/draft-that-does-not-exist".into());
        assert!(check_boot_options(&o).is_ok());
        let cfg = crate::families::minicpmv4_7::config::tests::real();
        let e = draft_for(&o, &cfg).unwrap_err().to_string();
        assert!(e.contains("--draft /draft-that-does-not-exist"), "{e}");
        // no drafting, or a spec mode without a model drafter, leaves the pack unloaded and unread
        o.drafting = false;
        assert!(draft_for(&o, &cfg).unwrap().is_none());
        o.drafting = true;
        o.spec = gb_spec::SpecMode::Copy;
        assert!(draft_for(&o, &cfg).unwrap().is_none());
        o.spec = gb_spec::SpecMode::None;
        assert!(draft_for(&o, &cfg).unwrap().is_none());
        // images are a component now (E5b): the options check lets them through, the memory plan prices the tower
        let mut o = opts(1);
        o.vision = Some(Default::default());
        assert!(check_boot_options(&o).is_ok());
        let mut o = opts(1);
        o.prefill_chunk = 100_000;
        assert!(check_boot_options(&o).is_err());
        o = opts(0);
        assert!(check_boot_options(&o).is_err());
    }

    /// A directory holding only a draft `config.json` (edited): `draft_for` reads the config and refuses before it opens a pack.
    fn draft_dir_with(tag: &str, edit: impl FnOnce(&mut serde_json::Value)) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("rexl3-draft-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(crate::families::minicpmv4_7::config::tests::DSPARK_CONFIG).unwrap();
        edit(&mut v);
        std::fs::write(dir.join("config.json"), v.to_string()).unwrap();
        dir
    }

    #[test]
    fn a_draft_that_does_not_belong_to_the_target_is_refused_by_name_before_anything_is_read() {
        let cfg = crate::families::minicpmv4_7::config::tests::real();
        type Edit = Box<dyn Fn(&mut serde_json::Value)>;
        let cases: Vec<(&str, Edit, &str)> = vec![
            ("hidden", Box::new(|v| v["hidden_size"] = serde_json::json!(4096)), "hidden_size 4096 against the target's 2048"),
            (
                "vocab",
                Box::new(|v| v["vocab_size"] = serde_json::json!(248_320)),
                "vocab_size 248320 against the target's 248144",
            ),
            (
                "taps-not-full-attention",
                Box::new(|v| v["dflash_config"]["target_layer_ids"] = serde_json::json!([3, 11, 19, 27, 36])),
                "tap layer 36 is not a full-attention layer of the target",
            ),
            (
                "taps-past-the-target",
                Box::new(|v| v["dflash_config"]["target_layer_ids"] = serde_json::json!([3, 11, 19, 27, 43])),
                "tap layer 43 is not a layer of the target (40 layers)",
            ),
            (
                "layers",
                Box::new(|v| v["num_target_layers"] = serde_json::json!(48)),
                "num_target_layers 48 against the target's 40",
            ),
            ("mask", Box::new(|v| v["dflash_config"]["mask_token_id"] = serde_json::json!(248_144)), "mask_token_id 248144"),
            ("block", Box::new(|v| v["block_size"] = serde_json::json!(14)), "block_size is 14"),
        ];
        for (tag, edit, needle) in cases {
            let dir = draft_dir_with(tag, |v| edit(v));
            let mut o = opts(1);
            o.draft_dir = Some(dir.clone());
            let e = draft_for(&o, &cfg).err().unwrap_or_else(|| panic!("{tag}: the draft was accepted")).to_string();
            assert!(e.contains(needle) || e.contains(&needle.replace(" is ", " ")), "{tag}: {e} does not say {needle:?}");
            assert!(e.contains("--draft"), "{tag}: {e}");
            // plan_boot reaches the same refusal (the planner reads the draft before it plans a byte)
            let e2 = plan_boot(&cfg, &o, weights(), 0, Some(&host(1000))).expect_err("refused at the plan").to_string();
            assert!(e2.contains("--draft"), "{tag}: {e2}");
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn a_good_draft_config_without_a_pack_is_refused_at_the_pack() {
        // the config is the real one; the directory holds no weights: the refusal names the directory, not a component
        let cfg = crate::families::minicpmv4_7::config::tests::real();
        let dir = draft_dir_with("nopack", |_| {});
        let mut o = opts(1);
        o.draft_dir = Some(dir.clone());
        let e = draft_for(&o, &cfg).expect_err("no weights").to_string();
        assert!(e.contains("--draft") && e.contains(&dir.display().to_string()), "{e}");
        assert!(e.contains("tensors of the DSpark recipe are wrong or missing"), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_tower_is_priced_in_the_plan_at_one_to_eight_slots() {
        use super::super::vision;
        let cfg = real();
        let w = weights();
        for slots in 1..=8usize {
            let mut o = opts(slots);
            o.max_image_tokens = 4096;
            let text = plan_boot(&cfg, &o, w, 90 * GIB, Some(&host(113))).unwrap();
            assert_eq!((text.vision_weight_bytes, text.memory.vision_reserve_bytes), (0, 0), "no tower, nothing priced");
            o.vision = Some(Default::default());
            let p = plan_boot(&cfg, &o, w, 90 * GIB, Some(&host(113))).unwrap();
            // the weights the headers give plus the tower's, as the loader will hold them
            assert_eq!(p.vision_weight_bytes, vision::weight_bytes());
            assert_eq!(p.free_after_weights, 113 * GIB - w - vision::weight_bytes());
            // the reserve: one workspace and the features of every image a slot may hold (4 images of 4,096 tokens x 4,096 B)
            let want = vision::workspace_bytes(vision::DEFAULT_MAX_PATCHES) + slots as u64 * 4 * vision::feature_bytes(4096);
            assert_eq!(p.memory.vision_reserve_bytes, want, "{slots} slots");
            // what the plan allocates beyond the weights stays inside what it was given, with the tower's bytes counted
            assert!(p.memory.planned_bytes(slots) <= p.free_after_weights - o.floor_bytes);
            // and what the boot compares with the device's state is the slots', the pool's and the cache's, never the tower's (it is
            // loaded by `engine_from`): an equality that counted the reserve would refuse every boot that has a tower
            assert_eq!(
                priced_state_bytes(&p.memory, slots),
                slots as u64 * p.memory.state_bytes_per_slot
                    + p.memory.kv_blocks as u64 * p.memory.block_bytes
                    + p.memory.cache.bytes(),
                "{slots} slots"
            );
            assert_eq!(p.slots, slots);
            // the pool is the text plan's less what the vision reserve and the tower's weights took
            assert!(p.memory.kv_blocks <= text.memory.kv_blocks);
        }
        // a box that holds the weights but not the tower is refused by name, and the refusal says how to serve text
        let mut o = opts(1);
        o.vision = Some(Default::default());
        let tight = (w + o.floor_bytes).div_ceil(GIB);
        assert!(tight * GIB >= w + o.floor_bytes && tight * GIB < w + o.floor_bytes + vision::weight_bytes());
        let e = plan_boot(&cfg, &o, w, 0, Some(&host(tight))).unwrap_err().to_string();
        assert!(e.contains("weights do not fit") && e.contains("vision tower") && e.contains("--no-vision"), "{e}");
        o.vision = None;
        let e = plan_boot(&cfg, &o, w, 0, Some(&host(tight))).unwrap_err().to_string();
        assert!(!e.contains("weights do not fit"), "without the tower the weights fit and the plan is what refuses: {e}");
    }

    /// E4b with E5b and E6b: the draft's weights, pages and overflow pages, the tower's weights and reserve and the cache's snapshots
    /// and pages in one plan at one to eight slots. The boot's equality (`priced_state_bytes` against what the device holds for the
    /// slots, the pool and the arena, the draft's pool added by the engine's own count) must hold with all three; the draft's planes are in
    /// the page and in the slot's state, the weights of the draft and the tower come off what the planner is given, and the tower's
    /// reserve is the plan's, never the state's.
    #[test]
    fn the_draft_the_tower_and_the_cache_are_priced_together_at_one_to_eight_slots() {
        use super::super::vision;
        let cfg = real();
        let dcfg = crate::families::minicpmv4_7::config::tests::draft();
        let w = weights();
        let dw = 451_186_749u64; // the 6.0 bpw draft's shard files
        let (page, draft_page, overflow) = (256 * (20_480 + 10_240), 256 * 10_240u64, 2_621_440u64);
        for slots in 1..=8usize {
            let mut o = ServeOptions { prefix_cache: crate::serve::PrefixCacheOption::Auto, ..opts(slots) };
            o.max_image_tokens = 4096;
            o.vision = Some(Default::default());
            let d = DraftPlan { cfg: dcfg.clone(), dir: "/draft".into(), weight_bytes: dw };
            let p = plan_boot_with(&cfg, &o, Some(d), w, 90 * GIB, Some(&host(113))).unwrap();
            assert_eq!(p.slots, slots);
            // the target's weights, the draft's and the tower's all come off what the planner is given
            assert_eq!(p.free_after_weights, 113 * GIB - w - dw - vision::weight_bytes(), "{slots} slots");
            // the page and the slot's state carry the draft's planes
            assert_eq!((p.memory.block_bytes, p.memory.cache.page_bytes), (page, page), "{slots} slots");
            assert_eq!(p.memory.state_bytes_per_slot, 64_389_120 + overflow, "{slots} slots");
            assert!(p.memory.cache.enabled(), "{slots} slots: the cache takes what the pool left");
            // the tower's reserve as without the draft
            let want = vision::workspace_bytes(vision::DEFAULT_MAX_PATCHES) + slots as u64 * 4 * vision::feature_bytes(4096);
            assert_eq!(p.memory.vision_reserve_bytes, want, "{slots} slots");
            // the boot's equality: the slots' state with the overflow page, the pool's pages with the draft's and the arena; the tower's
            // reserve and the engine's are the plan's own and not in it
            assert_eq!(
                priced_state_bytes(&p.memory, slots),
                slots as u64 * (64_389_120 + overflow) + p.memory.kv_blocks as u64 * page + p.memory.cache.bytes(),
                "{slots} slots"
            );
            // the draft's pool as the engine builder allocates it is the pool's pages and one overflow page a slot: the same bytes the
            // plan priced in the page and in the state
            assert_eq!(
                memory::draft_pool_bytes(&cfg, &dcfg, p.kv_pages(), slots),
                p.kv_pages() as u64 * draft_page + slots as u64 * overflow,
                "{slots} slots"
            );
            assert!(p.memory.planned_bytes(slots) <= p.free_after_weights - o.floor_bytes, "{slots} slots");
        }
    }

    #[test]
    fn the_refusal_lists_the_first_problems_and_counts_the_rest() {
        let errs: Vec<String> = (0..11).map(|i| format!("tensor.{i}: missing")).collect();
        let e = refusal(&errs).to_string();
        assert!(e.contains("11 problems") && e.contains("tensor.0: missing") && e.contains("tensor.7: missing"), "{e}");
        assert!(!e.contains("tensor.8") && e.contains("(and 3 more)"), "{e}");
    }

    #[test]
    fn the_moe_chunk_count_is_the_minicpm_one_and_flash_nexts_default_is_invalid_at_this_width() {
        // hidden 2,048 has 16 blocks of 128: 4, 8 and 16 K chunks divide it within the 48 KiB of staging, Flash-Next's 10 does not
        assert_eq!(KC, gb_cuda::fmoe::MINICPM_KC);
        assert_eq!(KC, 4);
        assert!(gb_cuda::fmoe::kc_valid(real().hidden, KC));
        assert!(!gb_cuda::fmoe::kc_valid(real().hidden, gb_cuda::fmoe::DEFAULT_KC));
        // and the instantiated grouped-expert shape is this model's (hidden, intermediate)
        assert!(COOPMK_SHAPES.contains(&(real().hidden, real().moe_intermediate)));
    }

    #[test]
    fn the_pin_rows_cover_every_window_row_count_and_the_chunk() {
        let v = pin_rows(4096);
        assert_eq!(v.len(), 65);
        assert_eq!((v[0], v[63], v[64]), (1, 64, 4096));
        assert_eq!(pin_rows(32).len(), 64);
    }

    #[test]
    fn a_config_alone_has_no_pack_verdict() {
        let dir = std::env::temp_dir().join(format!("rexl3-minicpm-verdict-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(pack_verdict(&dir).is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
