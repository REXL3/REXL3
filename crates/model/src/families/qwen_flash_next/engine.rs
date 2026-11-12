//! Build a text-only Flash-Next engine. Weight regions and the warm n-gram table are
//! budgeted before loading, because the model contract allocates KV and weights together.
use super::backend::FlashNextBackend;
use super::config::FlashNextConfig;
use super::model::{FlashNextModel, ModelOptions};
use super::ngram::NgramTable;
use super::prefix;
use super::weights::WeightMap;
use crate::error::ModelError;
use crate::hostmem::{self, MemInfo};
use crate::memory::MemoryOptions;
use crate::pack::Pack;
use crate::plan::{choose_slots, plan_memory_cached_windowed, MIN_TOKENS_PER_SLOT};
use crate::serve::{gib, PrefixCacheOption, ServeOptions};
use gb_engine::{boot, Drafter, Engine, EngineConfig, NoDrafts};
use gb_exl3::checkpoint::{read_header_checked, scan};
use gb_sched::{CostModel, ModelMemory};
use sha2::{Digest, Sha256};
use std::path::Path;

const WINDOW_ROWS: usize = super::memory::WINDOW_ROWS;
pub const STEP_MS: &[(usize, f64)] =
    &[(1, 29.0), (2, 34.1), (3, 38.7), (4, 43.6), (5, 47.9), (6, 52.0), (8, 59.2), (16, 85.0), (32, 130.0), (64, 220.0)];

/// Budget from reclaimable MemAvailable, not the smaller CUDA MemFree. The model
/// has not been loaded yet, so its weight regions must be deducted before the KV plan
/// is sized. The warmed n-gram table is NOT deducted: it lives in reclaimable page cache,
/// which `MemAvailable` already counts as recoverable, so subtracting it again would
/// double-count it as an anonymous/device reservation (and shrink the KV pool below what
/// a CF 131072 context needs). The planner takes its own reserve and floor out separately
/// (once).
///
/// The bytes the exact MTP draft-head copy adds to the weight plan: the first `MTP_HEAD_N`
/// columns of the packed LM head, derived from `cfg.vocab` (the same domain `mtpk::pick`
/// reads, validated again at model load). A pack whose checkpoint LM head is missing or does
/// not cover the candidate domain is refused here, before any memory plan is sized.
fn draft_head_plan_bytes(
    groups: &std::collections::BTreeMap<String, super::weights::Exl3Group>,
    vocab: usize,
) -> Result<u64, ModelError> {
    let h = groups.get("lm_head").ok_or_else(|| ModelError::Config("missing packed LM head".into()))?;
    let nv = super::model::draft_head_nv(vocab);
    super::model::check_draft_head_shapes(nv, h.n)?;
    let bits = h.bits.twice() as usize / 2;
    Ok(super::model::draft_head_prefix_bytes(h.k, nv, bits) as u64)
}

/// Device memory left for the KV plan after the loaded weights are deducted from the
/// planning figure (used by the 262K plan before the pool is sized).
fn free_after_load(cuda_free: u64, host: Option<&MemInfo>, weights: u64) -> u64 {
    hostmem::planning_free(cuda_free, host).saturating_sub(weights)
}

/// The per-sequence drafter factory. Any drafting run installs a copy drafter per stream, so a round the backend
/// serves no native drafts for — every multi-request decode batch (the Flash-Next MTP head proposes for one sequence
/// at a time) and any pack without an MTP head — still speculates per stream. Native MTP drafts override it whenever
/// the backend proposes them (single-request rounds). Drafts never change what is emitted, only how many rows ride in
/// the verify window, so the fallback keeps the exactness contract.
fn make_drafter(drafting: bool) -> Box<dyn Fn() -> Box<dyn Drafter>> {
    if drafting {
        Box::new(|| Box::new(gb_spec::CopyDrafter::new(8, 7)))
    } else {
        Box::new(|| Box::new(NoDrafts))
    }
}

/// `REXL3_NGRAM_WARM`: unset or `auto` decides from headroom, `0` turns warming off, any
/// other value forces it on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarmMode {
    Off,
    Auto,
    On,
}

pub fn warm_mode_from_env(value: Option<&str>) -> WarmMode {
    match value {
        None | Some("auto") => WarmMode::Auto,
        Some("0") => WarmMode::Off,
        Some(_) => WarmMode::On,
    }
}

/// Whether the n-gram warm should run. `Auto` runs it only when the reclaimable headroom left
/// after the weights still holds the whole table plus the floor; on a tight box the kernel would
/// reclaim the warm pages under pressure anyway, so skipping avoids the churn. `On` ignores
/// headroom, `Off` never warms. With no host figure (an old kernel) the device's numbers plan
/// the pool, and auto warms.
pub fn wants_warm(mode: WarmMode, host: Option<&MemInfo>, weights: u64, table: u64, floor: u64) -> bool {
    match mode {
        WarmMode::Off => false,
        WarmMode::On => true,
        WarmMode::Auto => host.map_or(true, |h| h.available.saturating_sub(weights) >= table.saturating_add(floor)),
    }
}

fn warm_word(want: bool, mode: WarmMode) -> &'static str {
    if want {
        "on"
    } else if mode == WarmMode::Auto {
        "auto-skip"
    } else {
        "off"
    }
}

fn check_flash_options(o: &ServeOptions) -> Result<(), ModelError> {
    crate::serve::check_options(o)?;
    if o.window_rows != WINDOW_ROWS {
        return Err(ModelError::Config(format!("Flash-Next --window-rows must be {WINDOW_ROWS}")));
    }
    if o.max_model_len == 0 {
        return Err(ModelError::Config("--max-model-len must be at least 1".into()));
    }
    if o.vision.is_some() {
        return Err(ModelError::NotBuilt("Flash-Next images need vision_flash_next; start with --no-vision".into()));
    }
    Ok(())
}

/// The upstream canonical pack fingerprint covers indexed EXL3 groups and plain tensors.
/// Sidecars are outside the shard index, so fold their checked tensor descriptions
/// (including payload offsets) and file lengths without reading their payloads.
/// As with upstream's fingerprint, a payload edit with unchanged metadata is not detected.
fn pack_header_fingerprint(dir: &Path) -> Result<String, ModelError> {
    let ck = scan(dir, true).map_err(|e| ModelError::Config(format!("pack fingerprint: {e}")))?;
    let issues: Vec<_> = ck.issues.iter().filter(|issue| !issue.contains(" as the index says")).collect();
    if !issues.is_empty() {
        return Err(ModelError::Config(format!(
            "pack fingerprint: {}",
            issues.into_iter().map(String::as_str).collect::<Vec<_>>().join("; ")
        )));
    }
    let mut h = Sha256::new();
    h.update(b"rexl3-flash-next-pack-headers-v1\0");
    h.update(crate::fingerprint::fingerprint(&ck).as_bytes());
    for file in [super::inventory::NGRAM_FILE, super::inventory::MTP_MIXER_FILE] {
        let path = dir.join(file);
        let (_, _, entries) = read_header_checked(&path).map_err(|e| ModelError::Config(format!("pack fingerprint: {e}")))?;
        let size = path.metadata().map_err(|e| ModelError::Config(format!("{}: {e}", path.display())))?.len();
        h.update(file.as_bytes());
        h.update([0]);
        h.update(size.to_le_bytes());
        let mut header_digest = Sha256::new();
        header_digest.update((entries.len() as u64).to_le_bytes());
        for (name, entry) in entries {
            let dtype =
                entry.get("dtype").and_then(|v| v.as_str()).ok_or_else(|| ModelError::Config(format!("{file}:{name}: dtype")))?;
            let shape = entry
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| ModelError::Config(format!("{file}:{name}: shape")))?;
            let offsets = entry
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| ModelError::Config(format!("{file}:{name}: offsets")))?;
            header_digest.update(name.as_bytes());
            header_digest.update([0]);
            header_digest.update(dtype.as_bytes());
            header_digest.update([0]);
            for values in [shape, offsets] {
                header_digest.update((values.len() as u64).to_le_bytes());
                for value in values {
                    let n =
                        value.as_u64().ok_or_else(|| ModelError::Config(format!("{file}:{name}: noninteger shape or offset")))?;
                    header_digest.update(n.to_le_bytes());
                }
            }
        }
        h.update(header_digest.finalize());
    }
    Ok(format!("sha256:{:x}", h.finalize()))
}

/// The pack gate before anything is loaded: the target against the validated list and, with `--draft`, the draft beside it (the shared
/// `serve::check_packs`, which records `target_fingerprint`, `draft_fingerprint`, `pack_validated` (the pair's name for a listed pair)
/// and `draft_validated` as boot facts, the identity `/metrics` and `/v1/models` report). Returns `load`'s value and the target's
/// fingerprint.
fn validate_pack_before_load<T>(
    dir: &Path,
    draft_dir: Option<&Path>,
    allow_unvalidated: bool,
    load: impl FnOnce() -> Result<T, ModelError>,
) -> Result<(T, String), ModelError> {
    let pack = Pack::open(dir)?;
    let draft = draft_dir.map(Pack::open).transpose()?;
    let target = crate::serve::check_packs(&pack, draft.as_ref(), allow_unvalidated)?;
    boot::mark("packs_checked", None);
    Ok((load()?, target))
}

pub fn build_engine(o: &ServeOptions) -> Result<Engine<FlashNextBackend>, ModelError> {
    check_flash_options(o)?;
    let cfg = FlashNextConfig::from_dir(&o.model_dir).map_err(ModelError::Config)?;
    cfg.check_shapes().map_err(ModelError::Config)?;
    // The declared prefill contract: the compensated-fp16 tensor-core hyper-connection pre-mix (hcg) is the
    // always-on default for tiled prefill on every qualifying shape (H % 4 == 0, hc >= 1, rank >= 1 — true of
    // this family); decode and GEMV prefill keep the fp32 `hcx::pre` reference. There is no switch (the env
    // reader was removed); the contract is unconditional. The marker is asserted by the gate recipe's expect_log.
    let prefill_contract = if cfg.hidden % 4 == 0 && cfg.hc_streams >= 1 && cfg.hc_lowrank >= 1 {
        "hcg-compensated-fp16"
    } else {
        "hcx-fp32-reference"
    };
    let max_context = o.max_model_len.min(cfg.max_position_embeddings);
    // a DFlash2 draft given with `--draft` drafts instead of the MTP head, which is then left on disk (its KV planes are not planned)
    let draft_dir = draft_dir_of(o)?;
    let with_mtp = cfg.mtp_layers > 0 && draft_dir.is_none();
    let profile = super::memory::memory_profile(
        &cfg,
        &MemoryOptions {
            window_rows: WINDOW_ROWS,
            prefill_rows: o.prefill_chunk,
            max_context,
            mtp: with_mtp,
            vision: None,
            trees: false,
        },
    )
    .map_err(|e| ModelError::Config(e.to_string()))?;

    // WeightMap::from_dir opens and validates the entire text pack's headers,
    // including both sidecars, without reading its large tensor payloads.
    let weights = WeightMap::from_dir(&o.model_dir, &cfg)
        .map_err(|errors| ModelError::Config(format!("Flash-Next pack: {}", errors.join("; "))))?;
    boot::mark("packs_scanned", None);
    let mode = warm_mode_from_env(std::env::var("REXL3_NGRAM_WARM").ok().as_deref());
    let (loaded, target_fp) = validate_pack_before_load(&o.model_dir, draft_dir, o.allow_unvalidated_pack, || {
        let table_bytes = if mode != WarmMode::Off && !cfg.ple_layer_ids.is_empty() {
            let table = NgramTable::open(&o.model_dir, &cfg).map_err(ModelError::Config)?;
            (table.rows as u64)
                .checked_mul(table.words as u64)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| ModelError::Config("n-gram table size overflows".into()))?
        } else {
            0
        };
        gb_cuda::rt::init_context()?;
        Ok((table_bytes, gb_cuda::rt::mem_info()?.0 as u64))
    })?;
    let (table_bytes, cuda_free) = loaded;
    boot::mark("cuda_ready", None);
    let mut weight_bytes = weights
        .regions
        .iter()
        .try_fold(0u64, |sum, r| sum.checked_add(r.bytes))
        .ok_or_else(|| ModelError::Config("weight region sizes overflow".into()))?;
    // The exact MTP draft-head prefix: the model owns a device copy of the first MTP_HEAD_N columns of the LM head
    // (the drafter's whole candidate domain), so the plan accounts for its bytes before `free_after_load` sizes the KV
    // plan. Same domain and bytes as the model's `FlashNextModel::new` (both derive from `cfg.vocab` and validate the
    // checkpoint head covers it), so the plan stays truthful for the 262K contract.
    if with_mtp {
        let extra = draft_head_plan_bytes(&weights.groups, cfg.vocab)?;
        weight_bytes =
            weight_bytes.checked_add(extra).ok_or_else(|| ModelError::Config("prefix-head accounting overflow".into()))?;
        eprintln!("rexl3 event=mtp_exact_prefix_plan extra_bytes={extra}");
    }
    // The DFlash2 draft is loaded before the room that is left is measured (MemAvailable then no longer counts it); its tap buffer, which
    // the model allocates later, is planned with the weights.
    let draft = match draft_dir {
        Some(dir) => {
            let d = load_draft(dir, &cfg, &weights, o.slots, o.prefill_chunk, o.floor_bytes)?;
            let tap = super::model::tap_buffer_bytes(o.prefill_chunk.max(WINDOW_ROWS), cfg.hidden);
            weight_bytes =
                weight_bytes.checked_add(tap).ok_or_else(|| ModelError::Config("draft tap buffer accounting overflow".into()))?;
            eprintln!(
                "rexl3 event=model_drafter drafter=dflash2 draft={} form={} draft_gib={:.2} tap_gib={:.3} validated=\"{}\" mtp=not_loaded",
                d.check.fingerprint,
                d.form,
                gib(d.total_bytes),
                gib(tap),
                d.check.how(&target_fp)
            );
            boot::fact("model_drafter", "dflash2 (--draft); MTP head not loaded");
            Some(d)
        }
        None => None,
    };
    let host = MemInfo::read();
    let want_warm = mode != WarmMode::Off
        && !cfg.ple_layer_ids.is_empty()
        && wants_warm(mode, host.as_ref(), weight_bytes, table_bytes, o.floor_bytes);
    let free = free_after_load(cuda_free, host.as_ref(), weight_bytes);
    eprintln!("{}", hostmem::memory_view_line(cuda_free, host.as_ref(), o.floor_bytes));
    eprintln!(
        "rexl3 event=flash_next_plan_weights weights_gib={:.2} warm_table_gib={:.2} ngram_warm={} free_after_gib={:.2} prefill_contract={}",
        gib(weight_bytes),
        gib(table_bytes),
        warm_word(want_warm, mode),
        gib(free),
        prefill_contract
    );
    boot::fact("plan_free_gib", format!("{:.1}", gib(free)));
    boot::fact("ngram_warm", warm_word(want_warm, mode));
    boot::fact("prefill_contract", prefill_contract);
    let kv_tokens_cap = o.kv_tokens.unwrap_or(max_context).min(max_context) as u64;
    let snapshot_bytes = prefix::snapshot_bytes_with(&cfg, with_mtp, draft.is_some());
    // the cache is sized from the window, for `auto` and an explicit `--prefix-cache-mib` alike (`docs/PREFIX_CACHE_SIZING.md` section 5)
    let plan_at = |slots: usize| {
        plan_memory_cached_windowed(
            &profile,
            free,
            o.floor_bytes,
            slots,
            kv_tokens_cap,
            0,
            o.prefix_cache.request(),
            snapshot_bytes,
            0,
            max_context as u64,
        )
    };
    let choice = choose_slots(o.slots, o.slots_auto, kv_tokens_cap, plan_at).map_err(|e| {
        ModelError::Config(format!("Flash-Next memory plan: {e}; MemAvailable after weights and warm table {:.1} GiB", gib(free)))
    })?;
    if choice.plan.kv_tokens < (choice.slots as u64 * MIN_TOKENS_PER_SLOT).min(kv_tokens_cap) {
        return Err(ModelError::Config("Flash-Next KV pool too small for the requested slots".into()));
    }
    let slots = choice.slots;
    if choice.shrunk() {
        eprintln!("rexl3 event=slots_shrunk requested={} granted={slots}", choice.requested);
    }
    let plan = choice.plan;
    eprintln!(
        "rexl3 event=flash_next_kv_plan kv_tokens={} slots={} kv_pages={} free_after_gib={:.2}",
        plan.kv_tokens,
        slots,
        plan.kv_blocks,
        gib(free)
    );
    if plan.cache.enabled() {
        eprintln!(
            "rexl3 event=flash_next_prefix_plan entries={} pages={} gib={:.2} reach_tokens={} window={max_context}",
            plan.cache.snapshot_slots,
            plan.cache.page_blocks,
            gib(plan.cache.bytes()),
            plan.cache.reach_tokens()
        );
        boot::fact("prefix_cache_reach_tokens", plan.cache.reach_tokens().to_string());
        if plan.cache.reach_tokens() < max_context as u64 {
            eprintln!(
                "rexl3 level=warn event=prefix_cache_reach reach_tokens={} window={max_context} note=\"a prompt longer than the reach is never served from the cache; raise --prefix-cache-mib\"",
                plan.cache.reach_tokens()
            );
        }
    } else {
        let reason = if o.prefix_cache == PrefixCacheOption::Off { "off" } else { "insufficient spare memory for two entries" };
        eprintln!("rexl3 event=flash_next_prefix_plan entries=0 reason={reason}");
    }
    // the cache namespace covers the draft too: a snapshot keeps the draft's window
    let fingerprint = if plan.cache.enabled() {
        let target = pack_header_fingerprint(&o.model_dir)?;
        Some(match &draft {
            Some(d) => format!("{target}|dflash2={}", d.check.fingerprint),
            None => target,
        })
    } else {
        None
    };
    let pool_pages = plan
        .kv_blocks
        .checked_add(plan.cache.page_blocks)
        .ok_or_else(|| ModelError::Config("Flash-Next KV page count overflows".into()))?;
    boot::fact("slots", slots.to_string());
    boot::fact("kv_pages", plan.kv_blocks.to_string());
    boot::fact("kv_tokens", plan.kv_tokens.to_string());
    let mut model = FlashNextModel::load_with(
        &o.model_dir,
        ModelOptions { slots, kv_pages: pool_pages, max_context, prefill_rows: o.prefill_chunk, warm: want_warm },
        with_mtp,
    )
    .map_err(|e| {
        crate::serve::allocation_error(
            "Flash-Next weights and engine buffers",
            e,
            weight_bytes.saturating_add(plan.planned_bytes(slots)),
            o.floor_bytes,
            host.as_ref(),
        )
    })?;
    // the draft joins the model before its snapshot arena (whose slots then keep the draft's window) and before any forward (the taps)
    if let Some(d) = draft {
        model.enable_dflash(d.draft, d.weight_bytes)?;
    }
    let mtp_k = if o.drafting && o.spec.uses_mtp() && model.has_mtp() {
        std::env::var("REXL3_MTP_K").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(6).min(WINDOW_ROWS - 1)
    } else {
        0
    };
    let mut backend = FlashNextBackend::new(model, o.prefill_chunk, mtp_k);
    if let Some(fingerprint) = fingerprint {
        backend = backend.with_prefix_cache(plan.cache, fingerprint).map_err(|e| {
            crate::serve::allocation_error(
                "Flash-Next prefix snapshot arena",
                e,
                weight_bytes.saturating_add(plan.planned_bytes(slots)),
                o.floor_bytes,
                host.as_ref(),
            )
        })?;
    }
    let mm = ModelMemory::from_profile(&profile, 0, 0);
    let admission = plan.admission(mm, slots, max_context).map_err(|e| ModelError::Config(e.to_string()))?;
    let drafting = o.drafting && o.spec.drafts();
    let make = make_drafter(drafting);
    boot::mark("kv_planned", Some(gib(plan.kv_tokens * profile.kv_bytes_per_token())));
    let engine = Engine::new(
        backend,
        admission,
        EngineConfig { max_rows: WINDOW_ROWS, max_drafts: 7, cost: CostModel::from_table(STEP_MS.to_vec()), drafting },
        make,
    );
    Ok(engine.with_prefix_cache(flash_next_prefix_config(&|name| std::env::var(name).ok())?))
}

/// The DFlash2 draft directory this boot drafts with: `--draft` when drafting is on. `--spec mtp` names the pack's MTP head, so it is
/// refused with a draft rather than served with another drafter (`--spec copy` never reaches here with one: `serve::draft_dir_for`).
pub fn draft_dir_of(o: &ServeOptions) -> Result<Option<&Path>, ModelError> {
    let Some(dir) = o.draft_dir.as_deref().filter(|_| o.drafting && o.spec.drafts()) else { return Ok(None) };
    if o.spec == gb_spec::SpecMode::Mtp {
        return Err(ModelError::Config(format!(
            "--spec mtp drafts with the pack's MTP head, and --draft {} gives a DFlash2 draft instead: pass --spec auto to draft with it, \
             or leave --draft out to draft with the MTP head",
            dir.display()
        )));
    }
    Ok(Some(dir))
}

/// A loaded DFlash2 draft, before it joins the model.
pub struct LoadedDraft {
    pub draft: crate::draft::Draft,
    pub check: crate::draft::DraftCheck,
    /// Bytes of its weights (the context rings excluded) and of everything it allocated.
    pub weight_bytes: u64,
    pub total_bytes: u64,
    pub form: &'static str,
}

/// Load the DFlash2 draft at `dir` for a target of `cfg` (whose `lm_head` the pack's `weights` describe) serving `slots` slots in prompt
/// chunks of `chunk` rows: the structural check first (refused by name), then its weights and context rings.
pub fn load_draft(
    dir: &Path,
    cfg: &FlashNextConfig,
    weights: &WeightMap,
    slots: usize,
    chunk: usize,
    floor: u64,
) -> Result<LoadedDraft, ModelError> {
    let check = super::dflash::check_dir(dir, cfg)?;
    let pack = Pack::open(dir).map_err(|e| ModelError::Config(format!("{}: {e}", dir.display())))?;
    let head = weights.groups.get("lm_head").ok_or_else(|| ModelError::Config("missing packed LM head".into()))?;
    let head_bits = head.bits.twice() as usize / 2;
    let form = if pack.has_group("fc") { "exl3" } else { "bf16" };
    let before = MemInfo::read().map(|m| m.available);
    // one arena block sized to the pack, not the default 1 GiB blocks (their unused tail would be most of a 0.5 GiB draft's footprint)
    let mut ctx = crate::loadctx::LoadCtx::from_env_with_block(crate::draft::arena_block_bytes(&pack));
    // the absorbs come in pieces of at most one window (`model_dflash.rs`), so the draft's context scratch is sized for one piece
    let ctx_rows = chunk.min(super::dflash::WINDOW);
    let (draft, bytes) = crate::draft::Draft::declare_geom(
        &mut ctx,
        &pack,
        crate::draft::DraftGeom::FLASH_NEXT,
        head_bits,
        cfg.vocab,
        slots,
        ctx_rows,
        crate::draft::BLOCK,
    )?;
    ctx.finish()?;
    let ring = gb_cuda::dflash2::DraftCtxLayout::qwen38(slots).total_bytes() as u64;
    let after = MemInfo::read().map(|m| m.available);
    let total_bytes = match (before, after) {
        (Some(b), Some(a)) => b.saturating_sub(a).max(bytes as u64),
        _ => bytes as u64,
    };
    if after.is_some_and(|a| a < floor) {
        return Err(ModelError::Config(format!(
            "the DFlash2 draft ({:.2} GiB) leaves MemAvailable below the {:.2} GiB floor: start without --draft",
            gib(total_bytes),
            gib(floor)
        )));
    }
    Ok(LoadedDraft { draft, check, weight_bytes: bytes as u64 - ring, total_bytes, form })
}

/// The Flash-Next prefix config: the interactive cut policy (cuts that serve a second question about a long document and a short tool
/// loop; the default policy serves neither), then the environment's overrides as in [`crate::serve::prefix_config_from_env`]. The restore
/// path is exact at any cut: `restore` copies the partial page and scrubs the donor's continuation (K/V rows and QSA pooled-key groups at
/// or beyond the cut) before the resumed sequence rewrites those positions, so no `cut_align` rounding is needed and short shared prefixes
/// (under one KV page) hit again.
fn flash_next_prefix_config(var: &dyn Fn(&str) -> Option<String>) -> Result<gb_engine::PrefixConfig, ModelError> {
    let c = crate::serve::interactive_prefix_config_from(var)?;
    Ok(c)
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
