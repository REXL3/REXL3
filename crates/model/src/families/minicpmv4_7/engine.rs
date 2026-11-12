//! Build a text-only MiniCPM-V-4.7 engine around a booted model (E3): `boot::boot` (scan, bind, validate, gate, plan, load, allocate,
//! verify: nothing here re-plans) gives the weights and the state at the planned sizes; this adds the working set the forward needs
//! (checked against the plan's engine reserve), the backend, the admission and the engine.
//!
//! A draft (E4b): `--draft DIR` loads the DSpark draft after the target (`spec::SpecRig`: the weights are paid for in the plan's
//! weights, its pages and overflow pages in the plan's page and slot sizes, its scratch, tap buffer and rings in the engine reserve),
//! attaches it to the model (`MiniCpmModel::attach_spec`) and runs the engine with 15 drafts a stream, the measured cost grid below
//! and the `Scored` budget policy reading the draft's calibrated confidences (`Proposal::accept`). Without `--draft` the engine is
//! E3's: the copy drafter of the engine core and no tap export.
//!
//! Images (E5b): when the server asks for vision (`ServeOptions::vision`) the tower is loaded here, after the plan has priced it
//! (`memory::MiniCpmMemory::vision`), checked against the plan's figures, and published as a `VisionService` for the HTTP side; the
//! engine's thread never runs it (the tower owns a stream, a workspace and its weights; it shares nothing with the text forward).
//!
//! The prefix cache (E6b): the boot allocated the snapshot arena at the planned size (`--prefix-cache-mib`, `auto`; the pool already
//! holds the cache's extra pages), it moves from the state to the backend here, and the engine is told to use it. With vision on the
//! cache's namespace carries the image preprocessing's identity (E5b: `MiniCpmPreprocessing::id()`, the video frames' too), so a
//! snapshot made for one image or one setting is never found for another or for text. With a draft the snapshot also names the
//! draft's context (`prefix.rs`, `spec.rs`): the draft's pages are the pool's pages, so they are shared by reference with the
//! snapshot's, and a restore resumes the draft's context too.

use super::backend::MiniCpmBackend;
use super::boot::{boot, boot_profile, BootPlan, LoadedModel, WINDOW_ROWS};
use super::memory;
use super::model::{MiniCpmModel, ModelOptions};
use super::prefix::SnapshotArena;
use super::spec::{SpecRig, MAX_STREAMS};
use super::vision::{self, MiniCpmEncoder, MiniCpmTower};
use crate::error::ModelError;
use crate::hostmem::MemInfo;
use crate::serve::{gib, FeatureBudget, ServeOptions, VisionService, VisionSlot};
use gb_engine::{BudgetPolicy, Drafter, Engine, EngineConfig, NoDrafts};
use gb_sched::{CostModel, DraftCost, MemoryProfile, ModelMemory, RoundCost, StepCost};
use std::sync::Arc;

/// Milliseconds of one forward of N rows (all streams together, window graph replayed, candidates read back, accepted rows committed),
/// measured on the GB10 (anonymized host-a, exclusive lock, 6.0 bpw pack, context 1,024, 2026-10-12; `bench_decode_windows_prefill_and_stages` of
/// `tests/minicpmv4_7_forward.rs`): one stream of 1, 2, 4, 8 and 16 rows, and 4 streams of 8 and of 16 rows for 32 and 64. The engine's
/// budget policy decides from this table how many drafts ride in a window; the expert union is concave in rows (a 16-row window costs
/// 2.2x a 1-row step, not 16x). A pack at another bit width moves the table, not its shape.
pub const STEP_MS: &[(usize, f64)] = &[(1, 15.0), (2, 17.7), (4, 20.8), (8, 25.3), (16, 33.5), (32, 53.1), (64, 77.8)];

/// The most drafts one stream offers a round with no draft model (the copy drafter's block; a 16-row verify window is the family's
/// ceiling, which the DSpark draft fills: `dspark::BLOCK` = 15 drafts).
pub const MAX_DRAFTS: usize = 7;

/// The verify step by (streams, rows of all streams together), with its commits: the engine's budget reads it, so the number of drafts
/// that ride in a window follows what a window costs and not a guess. Replaces [`STEP_MS`] when a draft is attached (rows alone cannot
/// tell four streams of one row from one stream of four). Measured on this build, 6.0 bpw pack, anonymized host-a under the exclusive lock, context
/// 512, the median of three sweeps of 12 steps each (`bench_spec_costs` of `tests/minicpmv4_7_spec.rs`, 2026-10-13; the one-stream
/// column reproduces E3's STEP_MS to 3%): see `docs/MINICPMV4_7_INTEGRATION_PLAN.md` section 15.
pub const STEP_GRID: &[(usize, usize, f64)] = &[
    (1, 1, 15.33),
    (1, 2, 17.67),
    (1, 3, 19.32),
    (1, 4, 20.59),
    (1, 5, 21.90),
    (1, 6, 22.82),
    (1, 8, 24.99),
    (1, 10, 28.24),
    (1, 12, 30.22),
    (1, 14, 32.05),
    (1, 16, 33.51),
    (2, 2, 18.83),
    (2, 4, 21.71),
    (2, 6, 24.29),
    (2, 8, 26.46),
    (2, 10, 28.77),
    (2, 12, 30.56),
    (2, 16, 34.69),
    (2, 20, 39.63),
    (2, 24, 43.09),
    (2, 28, 46.70),
    (2, 32, 49.30),
    (3, 3, 21.69),
    (3, 6, 25.32),
    (3, 9, 28.88),
    (3, 12, 32.02),
    (3, 15, 35.19),
    (3, 18, 40.03),
    (3, 24, 45.94),
    (3, 30, 51.98),
    (3, 36, 59.14),
    (3, 42, 62.96),
    (3, 48, 67.73),
    (4, 4, 25.62),
    (4, 8, 29.35),
    (4, 12, 34.28),
    (4, 16, 38.05),
    (4, 20, 42.63),
    (4, 24, 45.78),
    (4, 32, 53.68),
    (4, 40, 59.75),
    (4, 48, 66.33),
    (4, 56, 74.57),
    (4, 64, 79.93),
    (6, 6, 31.11),
    (6, 12, 37.11),
    (6, 18, 41.97),
    (6, 24, 48.12),
    (6, 30, 53.43),
    (6, 36, 57.57),
    (6, 48, 65.96),
    (6, 60, 76.43),
    (8, 8, 33.04),
    (8, 16, 40.70),
    (8, 24, 47.72),
    (8, 32, 53.76),
    (8, 40, 61.23),
    (8, 48, 65.77),
    (8, 64, 79.25),
];

/// One draft round by streams: the batched absorb of the rows each stream committed since its last round (one pass: 0.37 ms at one
/// stream, 0.48 at eight), the block's five layers, the head over the block's rows, the Markov walk and the confidence head, measured
/// through the engine path on this build (the commit's replay applied first; context 512, the median of 12 rounds; `bench_spec_costs`,
/// 2026-10-13: E4a's stand-alone 3.45 / 3.78 / 4.60 ms at 1 / 2 / 4 streams do not include the absorb). The draft always runs its whole
/// 15-row block, so the head's table is flat.
pub const DRAFT_ROUND_MS: &[(usize, f64)] = &[(1, 3.84), (2, 4.21), (3, 4.65), (4, 5.05), (6, 6.64), (8, 7.63)];

/// Extra milliseconds of a round per context token the streams hold: the target's attention reads 20 KiB of K and V a token once a
/// round (one row: 15.15 ms at 1,024 tokens, 16.04 at 8,192: 0.00012 ms a token; a 16-row window reads it twice: 0.00025). The draft's
/// own block adds 0.0001 ms a token in rounds that draft (3.47 ms at 1,024, 4.16 at 8,192), which the model has no term for: the
/// budget is slightly optimistic about drafting at tens of thousands of tokens of context.
pub const KV_MS_PER_TOKEN: f64 = 0.00012;

/// The cost of a round with the DSpark draft, as the budget policies read it.
pub fn round_cost() -> RoundCost {
    RoundCost {
        verify: StepCost::from_grid(STEP_GRID.to_vec()),
        tree_verify: None,
        draft: Some(DraftCost::new(DRAFT_ROUND_MS.to_vec(), vec![(1, 0.001), (128, 0.001)], 0.0).with_group(MAX_STREAMS)),
        kv_ms_per_token: KV_MS_PER_TOKEN,
        kv_pass_ms_per_token: 0.0,
        fixed_ms: 0.0,
    }
}

/// The serving engine of the pack at `o.model_dir`: the serving boot (a 40-layer pack only), then [`engine_from`].
pub fn build_engine(o: &ServeOptions) -> Result<Engine<MiniCpmBackend>, ModelError> {
    engine_from(boot(o)?, o)
}

/// The engine around a booted model. `loaded.state` becomes the model's state, so what the boot's plan priced is what runs.
pub fn engine_from(loaded: LoadedModel, o: &ServeOptions) -> Result<Engine<MiniCpmBackend>, ModelError> {
    let LoadedModel { cfg, weights, mut state, plan, report } = loaded;
    // the snapshot arena belongs to the backend; the rest of the state is the model's
    let arena_buf = state.snapshots.take();
    let (profile, _) = boot_profile(&cfg, o)?;
    let model_cfg = cfg.clone();
    let (slots, kv_pages, max_context) = (plan.slots, plan.kv_pages(), plan.max_context);
    let draft_pool = plan.draft.as_ref().map(|d| memory::draft_pool_bytes(&cfg, &d.cfg, kv_pages, slots));
    let mut model =
        MiniCpmModel::new(cfg, weights, state, ModelOptions { slots, kv_pages, max_context, prefill_rows: o.prefill_chunk })
            .map_err(|e| {
                crate::serve::allocation_error(
                    "MiniCPM-V-4.7 working set",
                    e,
                    plan.weight_bytes.saturating_add(plan.memory.planned_bytes(slots)),
                    o.floor_bytes,
                    report.host_before.as_ref(),
                )
            })?;
    // the DSpark draft (E4b): loaded after the target, its pool compared with what the plan priced, then attached to the model
    if let (Some(dp), Some(want_pool)) = (&plan.draft, draft_pool) {
        let rig = SpecRig::new(&dp.cfg, &dp.dir, model.spec_sizes()).map_err(|e| {
            crate::serve::allocation_error(
                "MiniCPM-V-4.7 DSpark draft",
                e,
                plan.weight_bytes.saturating_add(dp.weight_bytes).saturating_add(plan.memory.planned_bytes(slots)),
                o.floor_bytes,
                report.host_before.as_ref(),
            )
        })?;
        let b = rig.bytes();
        if b.pool != want_pool {
            return Err(ModelError::Config(format!(
                "MiniCPM-V-4.7 plan and allocation disagree: the draft's pool holds {} B, the plan prices {want_pool} B ({kv_pages} \
                 pages and {slots} overflow pages)",
                b.pool
            )));
        }
        // the shard files are a few MB over what is uploaded (headers); more than that is a plan that under-priced the weights
        if b.weights > dp.weight_bytes + (4 << 20) {
            return Err(ModelError::Config(format!(
                "MiniCPM-V-4.7 plan and allocation disagree: the draft's weights hold {} B, its files are {} B",
                b.weights, dp.weight_bytes
            )));
        }
        eprintln!(
            "rexl3 event=minicpm_draft kind={:?} weights_mib={:.1} pool_mib={:.1} scratch_mib={:.1} taps_mib={:.1} ring_mib={:.1} \
             k_top={} taps={:?}",
            rig.draft.weight_kind(),
            b.weights as f64 / (1 << 20) as f64,
            b.pool as f64 / (1 << 20) as f64,
            b.scratch as f64 / (1 << 20) as f64,
            b.taps as f64 / (1 << 20) as f64,
            b.ring as f64 / (1 << 20) as f64,
            rig.k_top,
            dp.cfg.target_layer_ids
        );
        model.attach_spec(rig)?;
    }
    // the plan == allocations of the working set: what the forward allocated beyond the slots' state and the pages (and the draft's
    // scratch, tap buffer and rings) is inside the engine reserve the planner held back (the state itself was compared item by item
    // at boot)
    let bytes = model.bytes();
    let spec_reserve = model.spec().map_or(0, |s| s.bytes().engine_reserve());
    if bytes.engine_reserve() + spec_reserve > plan.memory.engine_reserve_bytes {
        return Err(ModelError::Config(format!(
            "MiniCPM-V-4.7 working set {:.3} GiB (scratch {:.3}, workspaces {:.3}, window records {:.3}, draft {:.3}) exceeds the \
             plan's engine reserve of {:.3} GiB; lower --prefill-chunk or --max-model-len",
            gib(bytes.engine_reserve() + spec_reserve),
            gib(bytes.scratch),
            gib(bytes.workspaces),
            gib(bytes.window_records),
            gib(spec_reserve),
            gib(plan.memory.engine_reserve_bytes)
        )));
    }
    eprintln!(
        "rexl3 event=minicpm_working_set scratch_mib={:.1} workspaces_mib={:.1} window_records_mib={:.1} draft_mib={:.1} \
         reserve_gib={:.2}",
        bytes.scratch as f64 / (1 << 20) as f64,
        bytes.workspaces as f64 / (1 << 20) as f64,
        bytes.window_records as f64 / (1 << 20) as f64,
        spec_reserve as f64 / (1 << 20) as f64,
        gib(plan.memory.engine_reserve_bytes)
    );
    let mut backend = MiniCpmBackend::new(model, o.prefill_chunk);
    // plan == allocations of the cache: the arena the boot allocated is the plan's snapshot slots, no more and no fewer
    let cache = plan.memory.cache;
    let wanted = cache.snapshot_slots as u64 * cache.snapshot_bytes;
    let held = arena_buf.as_ref().map_or(0, |b| b.bytes() as u64);
    if held != wanted {
        return Err(ModelError::Config(format!(
            "MiniCPM-V-4.7 prefix arena holds {held} B, the plan prices {} snapshots of {} B = {wanted} B",
            cache.snapshot_slots, cache.snapshot_bytes
        )));
    }
    // the namespace of the cache carries what makes two images' rows differ at equal pixels: the preprocessing of the images and of
    // a video's frames (profile, slices, mode, normalisation)
    if o.vision.is_some() {
        let image = vision::image_preprocessing(&o.model_dir)?;
        let video = vision::video_preprocessing(&image)?;
        backend = backend.with_image_preprocessing(format!("{}|video:{}", image.id(), video.id()));
    }
    if let Some(buf) = arena_buf {
        // the budget's page is the target's and the draft's: the pool's pages carry both
        let draft_page = plan.draft.as_ref().map_or(0, |d| memory::draft_page_bytes(&d.cfg));
        let arena = SnapshotArena::new_with_draft(&model_cfg, buf, cache, draft_page)?;
        backend = backend.with_prefix_arena(arena, report.fingerprint.clone())?;
        eprintln!(
            "rexl3 event=minicpm_prefix_cache entries={} pages={} arena_mib={:.1} pool_pages={}",
            cache.snapshot_slots,
            cache.page_blocks,
            held as f64 / (1 << 20) as f64,
            plan.kv_pages()
        );
    }
    let mm = ModelMemory::from_profile(&profile, 0, 0);
    let admission = plan.memory.admission(mm, slots, max_context).map_err(|e| ModelError::Config(e.to_string()))?;
    if let Some(slot) = &o.vision {
        publish_vision(o, slot, &profile, &plan, report.host_before.as_ref())?;
    }
    let drafting = o.drafting && o.spec.drafts();
    let native = backend.m.has_draft();
    // with the DSpark draft the engine's drafter of last resort is none: a stream the draft cannot serve decodes plainly (the
    // budget's cost grid is the draft's, and a copy drafter's rounds would be priced as the draft's)
    let make: Box<dyn Fn() -> Box<dyn Drafter>> = if drafting && !native && o.spec.uses_copy() {
        Box::new(|| Box::new(gb_spec::CopyDrafter::new(8, 7)))
    } else {
        Box::new(|| Box::new(NoDrafts))
    };
    let max_drafts = if native { super::dspark::BLOCK } else { MAX_DRAFTS };
    let engine = Engine::new(
        backend,
        admission,
        EngineConfig { max_rows: WINDOW_ROWS, max_drafts, cost: CostModel::from_table(STEP_MS.to_vec()), drafting },
        make,
    );
    // the engine runs the cache only when its backend offers one (`Backend::prefix_caps`), so this is a no-op without an arena
    let engine = engine.with_prefix_cache(crate::serve::prefix_config_from_env()?);
    if !native {
        return Ok(engine);
    }
    // the draft's calibrated confidences (`Proposal::accept`) trim each round's window, and the draft is asked for its whole block
    // whenever the stream is worth a draft round at all: the block runs whole, so a shorter ask saves nothing and only holds back
    // picks the confidences would have kept (Leg D, section 15 of the plan). REXL3_SPEC_BUDGET picks another policy.
    let policy = crate::costs::budget_policy_from_env(BudgetPolicy::Scored)?;
    let learn = crate::costs::spec_learn_from_env()?;
    eprintln!("rexl3 event=spec_budget policy={} learn={learn} family=minicpmv4_7", format!("{policy:?}").to_lowercase());
    Ok(engine.with_budget(policy, round_cost()).with_learning(learn))
}

/// Load the tower and publish it as the server's vision service (E5b). The plan priced it: its weights and workspace are compared with
/// what the memory model said (a difference is a refusal, not a surprise), and the plan's vision reserve less the workspace is the byte
/// budget of the feature buffers of the images in flight.
pub fn publish_vision(
    o: &ServeOptions,
    slot: &VisionSlot,
    profile: &MemoryProfile,
    plan: &BootPlan,
    host_before: Option<&MemInfo>,
) -> Result<(), ModelError> {
    let priced = profile
        .vision()
        .ok_or_else(|| ModelError::Config("MiniCPM-V-4.7: images were asked for but the memory plan prices no tower".into()))?;
    let max_patches = vision::check_max_patches()?;
    let pre = vision::image_preprocessing(&o.model_dir)?;
    let t = std::time::Instant::now();
    let tower = MiniCpmTower::load(&o.model_dir, max_patches).map_err(|e| {
        crate::serve::allocation_error(
            "MiniCPM-V-4.7 vision tower",
            e,
            plan.vision_weight_bytes.saturating_add(plan.memory.vision_reserve_bytes),
            o.floor_bytes,
            host_before,
        )
    })?;
    let (weights, workspace) = tower.bytes();
    if priced.weight_bytes != Some(weights) || priced.encoder_scratch_bytes != workspace {
        return Err(ModelError::Config(format!(
            "MiniCPM-V-4.7 vision plan and allocation disagree: the tower holds {weights} B of weights and {workspace} B of workspace, \
             the memory plan priced {:?} and {}",
            priced.weight_bytes, priced.encoder_scratch_bytes
        )));
    }
    let features = plan.memory.vision_reserve_bytes.checked_sub(workspace).ok_or_else(|| {
        ModelError::Config(format!(
            "MiniCPM-V-4.7 vision reserve {} B is smaller than the tower's workspace {workspace} B",
            plan.memory.vision_reserve_bytes
        ))
    })?;
    eprintln!(
        "rexl3 event=minicpm_vision weights_mib={:.1} workspace_mib={:.1} features_mib={:.1} max_patches={max_patches} downsample={} \
         max_slice_nums={} resize={} load_s={:.2}",
        weights as f64 / (1 << 20) as f64,
        workspace as f64 / (1 << 20) as f64,
        features as f64 / (1 << 20) as f64,
        pre.downsample.name(),
        pre.max_slice_nums,
        pre.profile.name(),
        t.elapsed().as_secs_f64()
    );
    let video = vision::video_preprocessing(&pre)?;
    let encoder = MiniCpmEncoder::new(tower, pre, FeatureBudget::new(features)).with_video_preprocessing(video);
    let _ = slot.set(Arc::new(VisionService::new(Box::new(encoder), o.max_image_tokens)));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cost_table_is_a_valid_monotone_model_that_reaches_the_window() {
        CostModel::try_from_table(STEP_MS.to_vec()).unwrap();
        StepCost::try_from_grid(STEP_GRID.to_vec()).unwrap();
        assert_eq!(STEP_MS[0].0, 1);
        assert_eq!(STEP_MS.last().unwrap().0, WINDOW_ROWS);
        assert!(STEP_MS.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 <= w[1].1));
        // a stream's 16-row verify window is inside the table, and the widest step is four of them
        assert!(STEP_MS.iter().any(|&(r, _)| r == 16));
        const _: () = assert!(MAX_DRAFTS < 16, "a stream's drafts and its pending row fit its 16-row window");
        const _: () = assert!(super::super::dspark::BLOCK < 16, "the draft's 15 picks and the pending row fit a 16-row window");
    }

    #[test]
    fn the_draft_round_cost_is_a_valid_model_that_prices_the_draft_by_streams() {
        let c = round_cost();
        let d = c.draft.as_ref().unwrap();
        // one stream's round is the measured one; more streams cost more; no drafting streams cost nothing
        assert!((d.ms(1, 15) - 3.84).abs() < 0.01, "{}", d.ms(1, 15));
        assert!(d.ms(4, 60) > d.ms(2, 30) && d.ms(2, 30) > d.ms(1, 15));
        assert_eq!(d.ms(0, 0), 0.0);
        // a plan of 16 rows for one stream costs the 16-row window and the draft
        let one = gb_sched::RoundShape { streams: 1, rows: 16, drafting: 1, head_rows: 15, kv_tokens: 0 };
        assert!((c.ms(&one, false) - (33.51 + 3.84 + 0.0)).abs() < 0.02, "{}", c.ms(&one, false));
    }
}
