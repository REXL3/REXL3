//! The Nemotron engine builder (`engine.rs`) over a fake runtime and the family's real memory model: option checks that refuse by name,
//! the memory plan and the model options it hands the loader, admission taken from the plan, and an engine that serves. No GPU, no
//! pack: the plan is built from the published `config.json`.

mod nemotron_omni_backend_support;

use gb_engine::harness::drafted_equals_serial;
use gb_engine::{Backend, Request, ServeEngine};
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::{ModelOptions, NemotronBackend};
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::engine::{
    assemble, assemble_with, assemble_with_cost, build, build_serve, check_options, check_prefill_contract, head_priced_if,
    plan_serving, plan_serving_with, prefill_rows, resident_weight_bytes, restricted_rows_of_file, row_budget, sub_head_bytes,
    PlanInputs, ServePlan, STEP_MS,
};
use gb_model::families::nemotron_h_omni::memory::NemotronMemory;
use gb_model::families::nemotron_h_omni::spec::{CostTables, Source, SpecConfig, LM_HEAD_ROWS};
use gb_model::family::{BuildCtx, FamilyPlan, VisionSpec};
use gb_model::hostmem::MemInfo;
use gb_model::memory::VisionOptions;
use gb_model::serve::{PrefixCacheOption, ServeOptions, SpecMode};
use gb_sampling::Sampling;
use nemotron_omni_backend_support::{Call, FakeRuntime, HeadScript, VOCAB};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

const GIB: u64 = 1 << 30;
const PACK: &str = include_str!("../../../testdata/nemotron_omni/config.json");

fn cfg() -> OmniConfig {
    OmniConfig::from_json(&serde_json::from_str(PACK).unwrap()).unwrap()
}

/// The plan `probe` will hand the engine once the memory model is wired in.
fn family_plan() -> FamilyPlan {
    let c = cfg();
    FamilyPlan {
        family: "nemotron_h_omni",
        dir: PathBuf::from("/nonexistent"),
        layers: c.layer_specs(),
        mtp_layers: c.mtp_specs(),
        memory: Box::new(NemotronMemory::new(&c, 0)),
        vision: None,
        chat_format: "nemotron_omni",
        drafters: vec!["copy"],
        required_components: vec![],
    }
}

/// Serving options the engine's option check and plan accept: 64-row slices, which the fake runtime takes. Under the P2 prefill contract
/// (the default) the family's boot refuses them since v0.1.1 (`check_prefill_contract`: a rows cap of 64 or less); P1 boots them.
fn options() -> ServeOptions {
    let mut o = ServeOptions::new(PathBuf::from("/nonexistent"));
    o.prefill_chunk = 64;
    o.slots = 2;
    o.floor_bytes = 8 * GIB;
    o
}

/// A box with `available` GiB for plans and the weights resident.
struct Host {
    host: MemInfo,
    weights: u64,
}

impl Host {
    fn new(available_gib: u64, weights_gib: u64) -> Self {
        let a = available_gib * GIB;
        Self { host: MemInfo { total: 121 * GIB, free: a, available: a, cached: 0 }, weights: weights_gib * GIB }
    }
    fn inputs(&self) -> PlanInputs<'_> {
        PlanInputs {
            cuda_free: self.host.free,
            host: Some(&self.host),
            weight_bytes: self.weights,
            model_max_context: cfg().max_context,
        }
    }
}

fn refusal(o: &ServeOptions) -> String {
    check_options(o).expect_err("the options should be refused").to_string()
}

// ---- check_options

#[test]
fn the_shipped_default_prefill_chunk_is_clamped_to_the_slice_cap_and_zero_is_still_refused() {
    // `ServeOptions::new` says 4,096 rows a slice. Under the default prefill contract P2 (since rc2d) the cap is
    // `contract::P2_MAX_PREFILL_ROWS`, 4,096: the shipped slice is 4,096 rows, unclamped (the slice Q2's verdict and S5's measurements
    // ran). Under P1 it is the large-row MoE prefill's (512 rows by default since 2026-10-15, 256 under `REXL3_NEMO_PREFILL_ROWS_MAX=256`;
    // `backend::prefill_rows_max`): the engine slices at the cap and says so
    use gb_model::families::nemotron_h_omni::contract;
    let cap = gb_model::families::nemotron_h_omni::backend::prefill_rows_max();
    match contract::current() {
        contract::PrefillContract::P2 => assert_eq!(cap, contract::P2_MAX_PREFILL_ROWS),
        contract::PrefillContract::P1 => assert!(cap == 256 || cap == 512, "{cap}"),
    }
    let o = ServeOptions::new(PathBuf::from("/nonexistent"));
    assert_eq!(o.prefill_chunk, 4096);
    assert!(check_options(&o).is_ok());
    assert_eq!(prefill_rows(&o), cap);
    let mut o = options();
    // the engine's option check takes every chunk; under P2 the family's boot refuses 64 or less (`check_prefill_contract`, below)
    for chunk in [1, 33, 64, 65, 256, 257, 512, 513, 4096] {
        o.prefill_chunk = chunk;
        assert!(check_options(&o).is_ok(), "{chunk}");
        assert_eq!(prefill_rows(&o), chunk.min(cap), "{chunk}");
    }
    o.prefill_chunk = 0;
    assert!(refusal(&o).contains("--prefill-chunk"), "the shared check still applies");
}

#[test]
fn under_p2_the_boot_refuses_a_prefill_rows_cap_of_64_or_less_by_name_and_p1_takes_any_chunk() {
    // v0.1.1 (`docs/NEMOTRON_PREFILL_SPEED.md` 6.17): at a rows cap of 64 or less the model builds no tensor-core MoE scratch, so a P2
    // prefill would run P1's MoE kernels. The process's contract decides (`contract::check_rows_cap` has both contracts' table).
    use gb_model::families::nemotron_h_omni::contract;
    let p2 = contract::current() == contract::PrefillContract::P2;
    let mut o = options();
    for (chunk, p2_refused) in [(16, true), (64, true), (65, false), (4096, false)] {
        o.prefill_chunk = chunk;
        match check_prefill_contract(&o) {
            Ok(()) => assert!(!(p2 && p2_refused), "{chunk}: accepted under P2"),
            Err(e) => {
                assert!(p2 && p2_refused, "{chunk}: {e}");
                let e = e.to_string();
                assert!(e.contains(&format!("--prefill-chunk {chunk}")) && e.contains("more than 64 rows"), "{e}");
                assert!(e.contains(contract::CONTRACT_VAR), "the way to keep the chunk is named: {e}");
            }
        }
    }
    // a decode window wider than 64 rows sizes the MoE scratches itself: a narrow slice then runs P2's MoE block and boots
    o.prefill_chunk = 16;
    o.window_rows = 128;
    assert!(check_prefill_contract(&o).is_ok());
}

#[test]
fn the_window_width_the_vision_flag_the_drafter_flags_and_a_zero_context_are_refused_by_name() {
    let mut o = options();
    o.window_rows = 32;
    assert!(refusal(&o).contains("--window-rows"));
    // a wide decode window is the aggregate-throughput lever: 64 up to the model's cap (the shared option stops at 128 rows itself)
    for rows in [64, 65, 96, 128] {
        o.window_rows = rows;
        assert!(check_options(&o).is_ok(), "{rows}");
    }
    o.window_rows = 129;
    assert!(refusal(&o).contains("--window-rows"), "the shared check's bound still applies");
    let mut o = options();
    o.max_model_len = 0;
    assert!(refusal(&o).contains("--max-model-len"));
    // images are served now: the option check lets the vision slot through (the plan prices it and the family's build loads the tower:
    // `images_are_priced_exactly_when_the_tower_is_loaded`; a build whose component list lacks the tower is refused by `serve::preflight`)
    let mut o = options();
    o.vision = Some(Arc::new(OnceLock::new()));
    assert!(check_options(&o).is_ok());
    // `--spec mtp` is no longer refused here: the option check does not see the pack, and the plan and the assembly ask it by name
    // (`spec_mtp_is_not_refused_by_the_option_check_and_the_pack_and_the_model_are_asked`)
    for ok in [SpecMode::Auto, SpecMode::Mtp, SpecMode::Copy, SpecMode::None] {
        let mut o = options();
        o.spec = ok;
        assert!(check_options(&o).is_ok(), "{ok}");
    }
    let mut o = options();
    o.draft_dir = Some(PathBuf::from("/some/draft"));
    assert!(refusal(&o).contains("--draft-dir"));
}

// ---- the memory plan

#[test]
fn the_plan_hands_the_loader_the_model_options_it_priced() {
    // 100 GiB available, 60 GiB of weights: 40 GiB, less the 5 GiB reserve and the 8 GiB floor, holds a 262,144-token pool
    let b = Host::new(100, 60);
    let sp = plan_serving(&family_plan(), &options(), &b.inputs()).unwrap();
    assert_eq!((sp.requested_slots, sp.slots, sp.max_context), (2, 2, 262_144));
    assert_eq!(sp.free_after_weights, 40 * GIB);
    // the pool is the context (one full sequence's worth) and a page a slot of slack
    assert_eq!(
        sp.model,
        ModelOptions { slots: 2, kv_pages: 1024 + 2, max_context: 262_144, prefill_rows: 64, layers: None, mtp: false }
    );
    assert_eq!(sp.plan.kv_tokens, 262_144);
    assert_eq!(sp.plan.state_bytes_per_slot, 170_229_760);
    assert_eq!(sp.plan.engine_reserve_bytes, 5 * GIB);
    assert_eq!(sp.profile.kv_bytes_per_token(), 8192);
    // what the plan lays out fits the box with the floor to spare
    assert!(sp.plan.planned_bytes(2) + 60 * GIB + 8 * GIB <= 100 * GIB);
}

#[test]
fn the_context_is_clamped_to_the_models_and_the_pool_to_the_context() {
    let b = Host::new(100, 60);
    let mut o = options();
    o.max_model_len = 4 << 20;
    o.kv_tokens = Some(8 << 20);
    let sp = plan_serving(&family_plan(), &o, &b.inputs()).unwrap();
    assert_eq!(sp.max_context, 1 << 20, "the model's 1,048,576 tokens");
    assert_eq!(sp.plan.kv_tokens, 1 << 20, "a pool past one context is not planned");
    assert_eq!(sp.model.kv_pages, 4096 + 2);
    // the 1M-token pool is 8 GiB of KV on top of the weights
    assert_eq!(sp.plan.kv_blocks as u64 * sp.plan.block_bytes, (4096 + 2) * (2 << 20));
}

#[test]
fn a_box_that_cannot_hold_the_plan_is_refused_with_what_it_had() {
    // 70 GiB available less 60 GiB of weights is 10 GiB; the 5 GiB reserve and the 8 GiB floor do not fit
    let b = Host::new(70, 60);
    let e = plan_serving(&family_plan(), &options(), &b.inputs()).unwrap_err().to_string();
    assert!(e.contains("Nemotron memory plan") && e.contains("not enough memory") && e.contains("10.0 GiB"), "{e}");
    // room for the slot state and a few pages, not for six slots' worth of pool
    let b = Host::new(74, 60);
    let mut o = options();
    o.slots = 6;
    let e = plan_serving(&family_plan(), &o, &b.inputs()).unwrap_err().to_string();
    assert!(e.contains("KV pool too small for 6 slots"), "{e}");
    // asked for by default (not by flag) the same box boots with the slots that fit
    o.slots_auto = true;
    let sp = plan_serving(&family_plan(), &o, &b.inputs()).unwrap();
    assert!(sp.slots < 6 && sp.requested_slots == 6, "{} of {}", sp.slots, sp.requested_slots);
    assert_eq!(sp.model.slots, sp.slots);
}

#[test]
fn a_memory_model_that_refuses_the_shape_stops_the_boot_before_any_load() {
    // a family plan whose model prices a token at zero
    let mut c = cfg();
    c.head_dim = 0;
    let mut fam = family_plan();
    fam.memory = Box::new(NemotronMemory::new(&c, 0));
    let b = Host::new(100, 60);
    let o = options();
    let mut loaded = false;
    let e = build(&fam, &BuildCtx { options: &o }, &b.inputs(), |_| {
        loaded = true;
        Ok(FakeRuntime::new(2))
    })
    .err()
    .unwrap();
    assert!(!loaded, "the model was loaded for a plan that was never priced");
    assert!(e.to_string().contains("nemotron_h_omni") && e.to_string().contains("plane `k`"), "{e}");
}

// ---- images are priced exactly when the tower is loaded

fn vision_family_plan() -> FamilyPlan {
    let mut fam = family_plan();
    fam.vision = Some(VisionSpec {
        tower: "nemotron_radio_vision",
        patch_size: 16,
        merge_size: 2,
        temporal_patch_size: 1,
        out_hidden: 4096,
        image_token_id: 18,
        max_image_tokens: 3328,
    });
    fam
}

#[test]
fn images_are_priced_exactly_when_the_tower_is_loaded() {
    use gb_model::families::nemotron_h_omni::engine::vision_wanted;
    // one image of the largest size is 3,328 rows of bf16 [4096]; every slot may hold a request of 4 images
    let per_image = 3328 * 4096 * 2;
    let mut o = options();
    o.vision = Some(Arc::new(OnceLock::new()));
    // the library default of 4,096 tokens is above the tower's 3,328: the plan prices what the tower can produce
    assert_eq!(o.max_image_tokens, 4096);
    let tight = Host::new(74, 60);
    let with = plan_serving(&vision_family_plan(), &o, &tight.inputs()).unwrap();
    assert_eq!(with.vision, Some(VisionOptions { max_image_tokens: 3328, max_images_per_request: 4 }));
    assert_eq!(with.plan.vision_reserve_bytes, 2 * 4 * per_image);
    assert_eq!(with.profile.vision().unwrap().feature_bytes_per_image, per_image);
    // text only (`--no-vision` leaves the slot unset): the plan of before, to the byte
    let mut text_o = o.clone();
    text_o.vision = None;
    let text = plan_serving(&vision_family_plan(), &text_o, &tight.inputs()).unwrap();
    assert_eq!((text.vision, text.plan.vision_reserve_bytes), (None, 0));
    assert!(text.profile.vision().is_none());
    // the reserve comes out of the pool: the same box plans fewer KV pages, by the reserve to within a page
    assert!(
        with.plan.kv_blocks < text.plan.kv_blocks,
        "{} pages with images, {} without",
        with.plan.kv_blocks,
        text.plan.kv_blocks
    );
    let lost = (text.plan.kv_blocks - with.plan.kv_blocks) as u64 * with.plan.block_bytes;
    let reserve = with.plan.vision_reserve_bytes;
    assert!(
        lost + with.plan.block_bytes > reserve && lost <= reserve + with.plan.block_bytes,
        "{lost} B of pool for {reserve} B"
    );
    assert_eq!(
        with.model,
        ModelOptions { kv_pages: with.plan.kv_blocks, ..text.model.clone() },
        "the loader's options differ in the pool only"
    );
    // the reserve grows with the slots (each may hold its own request of images)
    let mut o4 = o.clone();
    o4.slots = 4;
    let four = plan_serving(&vision_family_plan(), &o4, &Host::new(100, 60).inputs()).unwrap();
    assert_eq!(four.plan.vision_reserve_bytes, 4 * 4 * per_image);
    // a lower --max-image-tokens buys memory
    o.max_image_tokens = 1024;
    let small = plan_serving(&vision_family_plan(), &o, &tight.inputs()).unwrap();
    assert_eq!(small.plan.vision_reserve_bytes, 2 * 4 * 1024 * 4096 * 2);
    // a pack without a vision component prices nothing, whatever the server asks (and the family's build loads no tower)
    let none = plan_serving(&family_plan(), &o, &tight.inputs()).unwrap();
    assert_eq!((none.vision, none.plan.vision_reserve_bytes), (None, 0));
    assert!(vision_wanted(&vision_family_plan(), &o));
    assert!(!vision_wanted(&family_plan(), &o));
    assert!(!vision_wanted(&vision_family_plan(), &text_o));
}

// ---- admission comes from the plan

fn engine_for(o: &ServeOptions, sp: &ServePlan) -> gb_engine::Engine<NemotronBackend<FakeRuntime>> {
    assemble(FakeRuntime::new(sp.slots), o, sp).unwrap()
}

#[test]
fn admission_takes_its_pool_and_limit_from_the_plan_not_a_constant() {
    let o = options();
    // a roomy box: the context limit governs
    let roomy = plan_serving(&family_plan(), &o, &Host::new(100, 60).inputs()).unwrap();
    let e = engine_for(&o, &roomy);
    assert_eq!(e.admission().request_limit(), Some(262_144));
    assert_eq!(e.admission().capacity(), roomy.plan.admit_bytes);
    assert_eq!(e.admission().kv_tokens(), Some(roomy.plan.kv_tokens));
    // a tight box: 14 GiB after the weights leaves 1 GiB past the reserve and the floor, so the pool (not the context) is the limit
    let tight = plan_serving(&family_plan(), &o, &Host::new(74, 60).inputs()).unwrap();
    assert!(tight.plan.kv_tokens < 262_144, "{}", tight.plan.kv_tokens);
    let e = engine_for(&o, &tight);
    assert_eq!(e.admission().request_limit(), Some(tight.plan.kv_tokens));
    // a prompt past the pool is refused at submit, with nothing prefilled
    let mut e = e;
    let long = Request {
        sampling: Sampling::default(),
        prompt: vec![5; tight.plan.kv_tokens as usize],
        max_new: 8,
        stop: vec![],
        draft: true,
        images: vec![],
        cache: true,
    };
    assert!(e.submit_deferred(long).is_err());
    assert_eq!(e.backend.m.calls.len(), 0, "the runtime was not touched");
}

#[test]
fn an_engine_built_from_a_plan_serves_and_drafted_equals_serial() {
    // four sequences at once need four slots: the harness submits them together and the plan's pool is what admits them
    let mut o = options();
    o.slots = 4;
    let sp = plan_serving(&family_plan(), &o, &Host::new(100, 60).inputs()).unwrap();
    assert_eq!(sp.model.slots, 4);
    // the first-principles step curve makes a window of 8 rows three times a serial row, which the engine's budget does not draft against
    // at its starting acceptance (see `STEP_MS`); a flat curve is what makes this test ask for drafts. Exactness does not depend on it.
    let flat = [(1, 70.0), (16, 74.0), (32, 110.0), (64, 190.0)];
    let make_engine = |drafting: bool| {
        let mut o = o.clone();
        o.drafting = drafting;
        assemble_with_cost(FakeRuntime::new(sp.slots), &o, &sp, &flat).unwrap()
    };
    let reqs: Vec<Request> = (0..4)
        .map(|i| {
            let unit: Vec<u32> = (0..12).map(|j| ((i * 13 + j * 5) % (VOCAB - 8)) as u32 + 4).collect();
            Request {
                prompt: unit.repeat(4 + i * 2),
                sampling: Sampling { seed: 7 + i as u64, temperature: 0.0, ..Default::default() },
                max_new: 96,
                stop: vec![],
                draft: true,
                images: vec![],
                cache: true,
            }
        })
        .collect();
    let r = drafted_equals_serial(&make_engine, &reqs).unwrap();
    assert!(r.accepted > 0, "no copy draft was accepted, so the drafted path was not exercised");
    // and the engine as built with its own curve gives the same tokens, drafting or not
    let production = |drafting: bool| {
        let mut o = o.clone();
        o.drafting = drafting;
        engine_for(&o, &sp)
    };
    drafted_equals_serial(&production, &reqs).unwrap();
    // `--spec none` runs every round as one row
    let mut o = options();
    o.spec = SpecMode::None;
    let mut e = engine_for(&o, &sp);
    for r in &reqs {
        e.submit(r.clone()).unwrap();
    }
    e.run().unwrap();
    let out = e.take_outputs();
    assert!(out.iter().all(|x| x.drafted == 0 && x.tokens.len() == 96));
    assert!(e.backend.m.violations.is_empty(), "{:?}", e.backend.m.violations);
}

// ---- the build

#[test]
fn build_loads_with_the_planned_options_once_and_assembles_the_engine() {
    let o = options();
    let b = Host::new(100, 60);
    let mut seen: Vec<ModelOptions> = Vec::new();
    let engine = build(&family_plan(), &BuildCtx { options: &o }, &b.inputs(), |opts| {
        seen.push(opts.clone());
        Ok(FakeRuntime::new(opts.slots))
    })
    .unwrap();
    assert_eq!(
        seen,
        vec![ModelOptions { slots: 2, kv_pages: 1026, max_context: 262_144, prefill_rows: 64, layers: None, mtp: false }]
    );
    assert_eq!(engine.backend.prefill_rows(), 64);
    assert!(engine.backend.m.calls.is_empty(), "assembling the engine does not run the model");
    assert_eq!(engine.admission().request_limit(), Some(262_144));
}

#[test]
fn build_serve_erases_the_backend_for_the_family() {
    let mut o = options();
    o.prefix_cache = PrefixCacheOption::Off;
    let b = Host::new(100, 60);
    let engine =
        build_serve(&family_plan(), &BuildCtx { options: &o }, &b.inputs(), |opts| Ok(FakeRuntime::new(opts.slots))).unwrap();
    let mut engine: Box<dyn ServeEngine> = engine;
    assert_eq!(engine.request_limit(), Some(262_144));
    assert!(engine.admission_stats().is_some());
    let id = engine
        .submit_deferred(Request {
            sampling: Sampling { temperature: 0.0, ..Default::default() },
            prompt: vec![9, 10, 11],
            max_new: 5,
            stop: vec![],
            draft: false,
            images: vec![],
            cache: false,
        })
        .unwrap();
    while engine.step().unwrap() {}
    let out = engine.take_outputs();
    assert_eq!((out.len(), out[0].id, out[0].tokens.len()), (1, id, 5));
}

#[test]
fn a_load_the_device_refuses_for_memory_names_the_plan_and_other_errors_pass_through() {
    let o = options();
    let b = Host::new(100, 60);
    let oom = build(&family_plan(), &BuildCtx { options: &o }, &b.inputs(), |_| -> Result<FakeRuntime, _> {
        Err(ModelError::Cuda(gb_cuda::CudaError(2)))
    })
    .err()
    .unwrap();
    let text = oom.to_string();
    assert!(matches!(oom, ModelError::Config(_)));
    assert!(text.contains("Nemotron weights and engine buffers") && text.contains("cudaErrorMemoryAllocation"), "{text}");
    assert!(text.contains("--slots"), "{text}");
    let other = build(&family_plan(), &BuildCtx { options: &o }, &b.inputs(), |_| -> Result<FakeRuntime, _> {
        Err(ModelError::Config("tensor missing".into()))
    })
    .err()
    .unwrap();
    assert!(matches!(&other, ModelError::Config(m) if m == "tensor missing"), "{other}");
}

#[test]
fn options_the_engine_cannot_honour_stop_the_boot_before_the_plan_and_the_load() {
    let b = Host::new(100, 60);
    let mut o = options();
    o.window_rows = 32;
    let mut loaded = false;
    let e = build(&family_plan(), &BuildCtx { options: &o }, &b.inputs(), |_| {
        loaded = true;
        Ok(FakeRuntime::new(8))
    })
    .err()
    .unwrap();
    assert!(!loaded);
    assert!(e.to_string().contains("--window-rows"), "{e}");
}

#[test]
fn the_cost_curve_is_ascending_and_a_valid_table() {
    assert!(STEP_MS.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 < w[1].1));
    assert_eq!((STEP_MS[0].0, STEP_MS.last().unwrap().0), (1, 64));
    // the measured table of the real 88-layer model with the fast MoE kernels and the window graphs (`nemotron_omni_costs`, node 1, one
    // sequence at 256 tokens, median of 7, 2026-10-11; rows 3 to 8 from served rounds): 38 ms for one row (26 tokens a second), 202 ms for sixty-four
    assert_eq!(STEP_MS[0].1, 38.0);
    assert_eq!(STEP_MS.last().unwrap().1, 202.0);
    assert!(
        STEP_MS.iter().all(|&(rows, ms)| rows == 1 || ms < 38.0 * rows as f64),
        "a window is cheaper than its rows served alone"
    );
}

// ---- the MTP head through the engine (the fake runtime's scripted head)

/// The family plan of a pack that offers the head (`nemotron_mtp` built): `drafters` lists `mtp`.
fn mtp_family() -> FamilyPlan {
    let mut f = family_plan();
    f.drafters = vec!["mtp", "copy"];
    f
}

/// Serve options and the plan for `slots` slots under `--spec spec`, for a pack that offers the head.
fn head_plan(slots: usize, spec: SpecMode) -> (ServeOptions, ServePlan) {
    let mut o = options();
    o.slots = slots;
    o.spec = spec;
    let sp = plan_serving(&mtp_family(), &o, &Host::new(100, 60).inputs()).unwrap();
    (o, sp)
}

type HeadEngine = gb_engine::Engine<NemotronBackend<FakeRuntime>>;

fn head_engine(o: &ServeOptions, sp: &ServePlan, script: HeadScript, cfg: SpecConfig, drafting: bool) -> HeadEngine {
    let mut o = o.clone();
    o.drafting = drafting;
    assemble_with(FakeRuntime::new(sp.slots).with_head(script), &o, sp, cfg, &CostTables::shipped()).unwrap()
}

/// A request whose prompt has no long repeat, so the copy drafter has little to say and the head is the drafter that matters.
fn plain_request(i: usize, sampled: bool, max_new: usize) -> Request {
    let prompt: Vec<u32> = (0..24 + 3 * i).map(|j| ((j * 37 + i * 101 + 11) % (VOCAB - 8)) as u32 + 4).collect();
    let sampling = if sampled {
        Sampling { seed: 40 + i as u64, temperature: 0.7, top_k: 20, top_p: 0.95, min_p: 0.0 }
    } else {
        Sampling { seed: i as u64, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 }
    };
    Request { prompt, sampling, max_new, stop: vec![], draft: true, images: vec![], cache: true }
}

/// Two greedy and two sampled requests.
fn mixed_requests(max_new: usize) -> Vec<Request> {
    (0..4).map(|i| plain_request(i, i % 2 == 1, max_new)).collect()
}

/// A request whose prompt is a repeating unit: the copy drafter has matches from the first round.
fn repeating_request(i: usize, max_new: usize) -> Request {
    let unit: Vec<u32> = (0..12).map(|j| ((i * 13 + j * 5) % (VOCAB - 8)) as u32 + 4).collect();
    Request {
        prompt: unit.repeat(5),
        sampling: Sampling { seed: 7 + i as u64, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 },
        max_new,
        stop: vec![],
        draft: true,
        images: vec![],
        cache: true,
    }
}

fn run_to_end(e: &mut HeadEngine, reqs: &[Request]) -> Vec<gb_engine::Output> {
    for r in reqs {
        e.submit(r.clone()).unwrap();
    }
    e.run().unwrap();
    assert!(e.take_failed().is_empty(), "no request may fail");
    e.take_outputs()
}

fn mode_cfg(mode: SpecMode) -> SpecConfig {
    SpecConfig::new(mode)
}

#[test]
fn spec_mtp_is_not_refused_by_the_option_check_and_the_pack_and_the_model_are_asked() {
    let mut o = options();
    o.spec = SpecMode::Mtp;
    assert!(check_options(&o).is_ok(), "the option check cannot see the pack: the plan and the assembly ask it");
    // a pack that does not offer the head (the component is not built): refused by name before anything is priced
    let e = plan_serving(&family_plan(), &o, &Host::new(100, 60).inputs()).unwrap_err();
    assert!(matches!(&e, ModelError::NotBuilt(m) if m.contains("--spec mtp") && m.contains("nemotron_mtp")), "{e}");
    // a pack with no head layers at all
    let mut f = family_plan();
    f.mtp_layers = vec![];
    let e = plan_serving(&f, &o, &Host::new(100, 60).inputs()).unwrap_err();
    assert!(matches!(&e, ModelError::Config(m) if m.contains("--spec mtp") && m.contains("no MTP head")), "{e}");
    // the pack offers it: planned
    assert!(plan_serving(&mtp_family(), &o, &Host::new(100, 60).inputs()).is_ok());
    // every other mode is allowed whatever the pack offers
    for ok in [SpecMode::Auto, SpecMode::Copy, SpecMode::None] {
        let mut o = options();
        o.spec = ok;
        assert!(plan_serving(&family_plan(), &o, &Host::new(100, 60).inputs()).is_ok(), "{ok}");
    }
    // the loaded model is the last word: a model with no head cannot be asked for `--spec mtp`
    let (o, sp) = head_plan(2, SpecMode::Mtp);
    let e = assemble(FakeRuntime::new(sp.slots), &o, &sp).err().expect("a model without a head must be refused");
    assert!(matches!(&e, ModelError::Config(m) if m.contains("--spec mtp") && m.contains("no MTP head")), "{e}");
    assert!(assemble(FakeRuntime::new(sp.slots).with_head(HeadScript::perfect()), &o, &sp).is_ok());
    // settings that are for another mode are refused
    let e = assemble_with(FakeRuntime::new(2), &o, &sp, SpecConfig::new(SpecMode::Auto), &CostTables::shipped()).err().unwrap();
    assert!(e.to_string().contains("--spec auto"), "{e}");
}

#[test]
fn the_plan_prices_the_head_exactly_when_the_pack_offers_it_and_spec_drafts_with_it() {
    let base = {
        let mut o = options();
        o.spec = SpecMode::Copy;
        plan_serving(&mtp_family(), &o, &Host::new(100, 60).inputs()).unwrap()
    };
    assert!(!base.mtp);
    assert_eq!((base.profile.kv_bytes_per_token(), base.plan.state_bytes_per_slot), (8192, 170_229_760));
    for spec in [SpecMode::Auto, SpecMode::Mtp] {
        let (_, sp) = head_plan(2, spec);
        assert!(sp.mtp, "{spec}");
        // the head's cache: 1 KiB a token; its tap store: 32 rows of 4,096 f32 a slot
        assert_eq!(sp.profile.kv_bytes_per_token(), 8192 + 1024, "{spec}");
        assert_eq!(sp.profile.drafter_state_bytes_per_slot(), 32 * 4096 * 4);
        assert_eq!(sp.plan.state_bytes_per_slot, 170_229_760 + 524_288, "{spec}");
        assert!(sp.profile.scratch_bytes() > base.profile.scratch_bytes());
        // the pool is smaller for the same free memory: the page is dearer
        assert!(sp.plan.block_bytes > base.plan.block_bytes);
    }
    // not priced: a pack that does not offer the head, and the modes and flags that never draft with it
    let mut o = options();
    o.spec = SpecMode::Auto;
    assert!(!plan_serving(&family_plan(), &o, &Host::new(100, 60).inputs()).unwrap().mtp);
    for (spec, drafting) in [(SpecMode::Copy, true), (SpecMode::None, true), (SpecMode::Auto, false), (SpecMode::Mtp, false)] {
        let mut o = options();
        o.spec = spec;
        o.drafting = drafting;
        let sp = plan_serving(&mtp_family(), &o, &Host::new(100, 60).inputs()).unwrap();
        assert!(!sp.mtp, "{spec} drafting {drafting}");
        assert_eq!(sp.profile.kv_bytes_per_token(), 8192);
    }
    // admission charges the tap store with the slot and the head's cache with the token
    let (o, sp) = head_plan(2, SpecMode::Auto);
    let e = assemble_with(
        FakeRuntime::new(2).with_head(HeadScript::perfect()),
        &o,
        &sp,
        mode_cfg(SpecMode::Auto),
        &CostTables::shipped(),
    )
    .unwrap();
    assert_eq!(e.admission().request_limit(), Some(262_144));
    assert_eq!(e.admission().capacity(), sp.plan.admit_bytes);
}

#[test]
fn the_engines_row_budget_is_flat_under_auto_and_mtp_and_the_step_table_under_copy_and_none() {
    let table = [(1usize, 269.0f64), (4, 575.0), (64, 2618.0)];
    for (spec, drafting, flat) in [
        (SpecMode::Auto, true, true),
        (SpecMode::Mtp, true, true),
        (SpecMode::Copy, true, false),
        (SpecMode::None, true, false),
        (SpecMode::Auto, false, false),
    ] {
        let c = row_budget(spec, drafting, &table);
        let (one, many) = (c.ms(1), c.ms(64));
        assert_eq!(one == many, flat, "{spec} drafting {drafting}: {one} ms at one row, {many} at 64");
        if !flat {
            assert_eq!((one, many), (269.0, 2618.0), "{spec}: the step table as it is");
        }
    }
}

#[test]
fn a_perfect_head_drafts_deep_every_draft_is_accepted_and_the_text_is_the_serial_text() {
    let (o, sp) = head_plan(4, SpecMode::Mtp);
    let reqs = mixed_requests(80);
    let make = |drafting: bool| head_engine(&o, &sp, HeadScript::perfect(), mode_cfg(SpecMode::Mtp), drafting);
    let r = drafted_equals_serial(&make, &reqs).unwrap();
    // the head path ran: drafts were verified, and a perfect head (greedy argmax, and the keyed draw for the sampled requests) is
    // right every time
    assert!(r.drafted > 0 && r.accepted > 0, "{r:?}");
    assert_eq!(r.accepted, r.drafted, "a perfect head's drafts are the serial samples: {r:?}");
    assert!(r.tokens_per_round() > 3.0, "depth 3 buys about four tokens a round: {:.2}", r.tokens_per_round());
    // and through the backend's counters
    let mut e = make(true);
    let outs = run_to_end(&mut e, &reqs);
    let spec = e.backend.spec.as_ref().unwrap();
    let st = &spec.stats;
    assert!(st.rounds[Source::Mtp as usize] > 0 && st.verified[Source::Mtp as usize] > 0);
    assert_eq!(st.accepted[Source::Mtp as usize], st.verified[Source::Mtp as usize]);
    assert_eq!(st.verified[Source::Copy as usize], 0, "--spec mtp offers no copy drafts");
    assert!(e.backend.m.violations.is_empty() && e.backend.m.head_lost.is_empty(), "{:?}", e.backend.m.violations);
    assert!(outs.iter().all(|x| x.tokens.len() == 80));
    // a perfect head is held to the depth setting and every depth was reached
    let deepest = e.backend.m.mtp_calls().iter().flat_map(|c| c.2.clone()).max().unwrap();
    assert_eq!(deepest, 3, "the default depth");
}

#[test]
fn the_depth_setting_caps_the_chain_under_mtp_and_under_auto() {
    for (mode, depth) in
        [(SpecMode::Mtp, 1usize), (SpecMode::Mtp, 2), (SpecMode::Mtp, 4), (SpecMode::Auto, 1), (SpecMode::Auto, 2)]
    {
        let (o, sp) = head_plan(2, mode);
        let cfg = SpecConfig { mtp_depth: depth, ..mode_cfg(mode) };
        let mut e = head_engine(&o, &sp, HeadScript::perfect(), cfg, true);
        run_to_end(&mut e, &[plain_request(0, false, 60), plain_request(1, true, 60)]);
        let depths: Vec<usize> = e.backend.m.mtp_calls().iter().flat_map(|c| c.2.clone()).collect();
        assert!(!depths.is_empty(), "{mode} depth {depth}: the head never drafted");
        assert!(depths.iter().all(|&d| d <= depth), "{mode} depth {depth}: asked for {:?}", depths.iter().max());
        if mode == SpecMode::Mtp {
            assert!(
                depths.iter().filter(|&&d| d > 0).any(|&d| d == depth),
                "{mode} depth {depth}: the fixed depth was never asked"
            );
        }
        assert!(e.backend.m.violations.is_empty(), "{:?}", e.backend.m.violations);
    }
}

#[test]
fn auto_prefers_the_head_where_its_acceptance_is_high() {
    let (o, sp) = head_plan(4, SpecMode::Auto);
    let reqs = mixed_requests(80);
    let make = |drafting: bool| head_engine(&o, &sp, HeadScript::perfect(), mode_cfg(SpecMode::Auto), drafting);
    drafted_equals_serial(&make, &reqs).unwrap();
    let mut e = make(true);
    run_to_end(&mut e, &reqs);
    let st = &e.backend.spec.as_ref().unwrap().stats;
    let total: u64 = st.rounds.iter().sum();
    let mtp = st.rounds[Source::Mtp as usize];
    assert!(mtp * 10 >= total * 8, "the arbiter drafted with the head in {mtp} of {total} rounds");
    assert!(st.tokens_per_round(Source::Mtp) > 3.0, "{:.2}", st.tokens_per_round(Source::Mtp));
    assert!(e.backend.spec.as_ref().unwrap().arbiter.stats.picks[2] > 0);
}

#[test]
fn auto_falls_back_to_serial_and_to_copy_when_the_head_is_garbage_and_fixed_mtp_pays_for_it() {
    // a head that is always wrong, on text the copy drafter can predict: the arbiter must learn it and stop paying for the head
    let (o, sp) = head_plan(4, SpecMode::Auto);
    let reqs: Vec<Request> = (0..4).map(|i| repeating_request(i, 120)).collect();
    let make = |drafting: bool| head_engine(&o, &sp, HeadScript::Garbage, mode_cfg(SpecMode::Auto), drafting);
    let r = drafted_equals_serial(&make, &reqs).unwrap();
    assert!(r.accepted > 0, "the copy drafter's drafts were accepted: {r:?}");
    let mut e = make(true);
    run_to_end(&mut e, &reqs);
    let st = &e.backend.spec.as_ref().unwrap().stats;
    let total: u64 = st.rounds.iter().sum();
    let mtp = st.rounds[Source::Mtp as usize];
    assert_eq!(st.accepted[Source::Mtp as usize], 0, "a garbage head is never right");
    assert!(mtp * 2 < total, "the arbiter kept drafting with a head that is always wrong: {mtp} of {total} rounds");
    assert!(
        st.rounds[Source::Copy as usize] > 0 && st.accepted[Source::Copy as usize] > 0,
        "and it fell back to the copy drafter"
    );
    assert!(st.verified[Source::Mtp as usize] < st.verified[Source::Copy as usize]);
    // on text with no matches it falls back to serial rows
    let reqs: Vec<Request> = (0..2).map(|i| plain_request(i, false, 100)).collect();
    let mut e = make(true);
    run_to_end(&mut e, &reqs);
    let st = &e.backend.spec.as_ref().unwrap().stats;
    let total: u64 = st.rounds.iter().sum();
    assert!(st.rounds[Source::Mtp as usize] * 4 < total, "{:?} of {total}", st.rounds);
    assert!(e.backend.spec.as_ref().unwrap().arbiter.stats.probes > 0, "a source judged poor is still looked at now and then");
    // a fixed depth keeps paying for a garbage head: every round drafts with it, and the text is still the serial text
    let (om, spm) = head_plan(2, SpecMode::Mtp);
    let reqs: Vec<Request> = (0..2).map(|i| plain_request(i, false, 60)).collect();
    let make = |drafting: bool| head_engine(&om, &spm, HeadScript::Garbage, mode_cfg(SpecMode::Mtp), drafting);
    let r = drafted_equals_serial(&make, &reqs).unwrap();
    assert_eq!(r.accepted, 0);
    assert!(r.drafted > 0);
    let mut e = make(true);
    run_to_end(&mut e, &reqs);
    let st = &e.backend.spec.as_ref().unwrap().stats;
    assert_eq!(st.rounds[Source::Mtp as usize], st.rounds.iter().sum::<u64>(), "every round drafts with the head");
}

#[test]
fn a_partly_right_head_gives_consistent_per_source_and_per_depth_telemetry() {
    let (o, sp) = head_plan(4, SpecMode::Mtp);
    let reqs = mixed_requests(160);
    let script = HeadScript::Truth { accept: vec![0.9, 0.5, 0.2] };
    let make = |drafting: bool| head_engine(&o, &sp, script.clone(), mode_cfg(SpecMode::Mtp), drafting);
    let r = drafted_equals_serial(&make, &reqs).unwrap();
    assert!(r.accepted > 0 && r.accepted < r.drafted, "{r:?}");
    let mut e = make(true);
    let outs = run_to_end(&mut e, &reqs);
    let st = &e.backend.spec.as_ref().unwrap().stats;
    let m = Source::Mtp as usize;
    // the conditional acceptance of each depth is what the script says (a draft is right with that probability whatever came before)
    for (j, want) in [(1usize, 0.9f64), (2, 0.5), (3, 0.2)] {
        let got = st.depth_acceptance(Source::Mtp, j);
        assert!((got - want).abs() < 0.12, "depth {j}: {got:.3} against {want} over {} rounds", st.reached[m][j - 1]);
    }
    for src in 0..3 {
        // every source: the hits of the depths sum to the accepted drafts, committed tokens are the accepted drafts and one a round,
        // and the rounds by depth sum to the rounds
        assert_eq!(st.hits[src].iter().sum::<u64>(), st.accepted[src]);
        assert_eq!(st.committed[src], st.accepted[src] + st.rounds[src]);
        assert_eq!(st.by_depth[src].iter().sum::<u64>(), st.rounds[src]);
        let drafts: u64 = st.by_depth[src].iter().enumerate().map(|(d, n)| d as u64 * n).sum();
        assert_eq!(drafts, st.verified[src]);
        for j in 0..7 {
            // a depth is reached only after the one before it hit, and a hit was reached
            assert!(st.hits[src][j] <= st.reached[src][j]);
            assert!(st.reached[src][j + 1] <= st.hits[src][j]);
        }
    }
    // the backend sees every committed round; the engine also counts the last round of each sequence, which ends without a commit
    let (drafted, accepted): (u64, u64) = outs.iter().fold((0, 0), |a, x| (a.0 + x.drafted, a.1 + x.accepted));
    let (v, a) = (st.verified.iter().sum::<u64>(), st.accepted.iter().sum::<u64>());
    assert!(v <= drafted && a <= accepted);
    assert!(drafted - v <= 3 * outs.len() as u64, "{drafted} drafted, {v} counted by the backend");
    // the arbiter's own picks are only for `auto`: the fixed depth never learned
    assert_eq!(e.backend.spec.as_ref().unwrap().arbiter.stats.picks, [0, 0, 0]);
}

#[test]
fn a_head_that_errors_never_fails_a_request_and_is_not_asked_again_for_the_sequence() {
    let (o, sp) = head_plan(4, SpecMode::Mtp);
    let reqs = mixed_requests(70);
    // from the third draft call on, every call fails
    let script = HeadScript::Errors { from_call: 3, count: usize::MAX, fatal: false };
    let make = |drafting: bool| head_engine(&o, &sp, script.clone(), mode_cfg(SpecMode::Mtp), drafting);
    let r = drafted_equals_serial(&make, &reqs).unwrap();
    assert_eq!(r.sequences, 4);
    let mut e = make(true);
    let outs = run_to_end(&mut e, &reqs);
    assert!(outs.iter().all(|x| x.tokens.len() == 70), "every request completed");
    assert!(e.backend.take_error().is_none());
    // the failing call switched off the slots it served; none of them was asked again, so the head failed once per call that had a slot
    // that was not yet off, and the calls before it drafted
    let st = &e.backend.spec.as_ref().unwrap().stats;
    assert!(st.mtp_errors >= 1);
    assert!(st.verified[Source::Mtp as usize] > 0, "the first two calls drafted");
    let disabled: Vec<usize> =
        e.backend.m.calls.iter().filter_map(|c| if let Call::MtpDisable(s) = c { Some(*s) } else { None }).collect();
    assert!(!disabled.is_empty());
    assert!(e.backend.m.mtp_draft_calls <= 2 + 4, "{} head calls: a disabled slot was asked again", e.backend.m.mtp_draft_calls);
    assert!(e.backend.m.violations.is_empty(), "{:?}", e.backend.m.violations);
    // under `auto` the failed head costs the arbiter nothing but the head: the round goes on, serially or with a copy offer. The one
    // failing call served both streams (both are active at the first round), so both are off for their sequences and it is not asked again
    let (oa, spa) = head_plan(2, SpecMode::Auto);
    let reqs: Vec<Request> = (0..2).map(|i| plain_request(i, i == 1, 90)).collect();
    let script = HeadScript::Errors { from_call: 1, count: 1, fatal: false };
    let make = |drafting: bool| head_engine(&oa, &spa, script.clone(), mode_cfg(SpecMode::Auto), drafting);
    drafted_equals_serial(&make, &reqs).unwrap();
    let mut e = make(true);
    run_to_end(&mut e, &reqs);
    let st = &e.backend.spec.as_ref().unwrap().stats;
    assert_eq!(st.mtp_errors, 1);
    assert_eq!(e.backend.m.mtp_draft_calls, 1, "a switched-off slot is not asked again");
    assert_eq!(st.rounds[Source::Mtp as usize], 0);
    assert!(st.rounds[Source::None as usize] + st.rounds[Source::Copy as usize] > 0);
}

#[test]
fn a_fatal_head_error_stops_the_engine_like_any_other_fatal_error() {
    let (o, sp) = head_plan(2, SpecMode::Mtp);
    let mut e =
        head_engine(&o, &sp, HeadScript::Errors { from_call: 1, count: usize::MAX, fatal: true }, mode_cfg(SpecMode::Mtp), true);
    e.submit(plain_request(0, false, 40)).unwrap();
    let err = e.run().unwrap_err();
    assert!(err.is_fatal(), "{err:?}");
}

#[test]
fn the_noise_flag_reaches_the_runtime_only_for_sampled_requests_and_only_when_it_is_on() {
    // keyed on (the default): the sampled request's seed rides with its asks, the greedy request's asks carry none
    for (noise, want_seed) in [(gb_spec::DraftPick::Keyed, Some(41u64)), (gb_spec::DraftPick::Argmax, None)] {
        let (o, sp) = head_plan(1, SpecMode::Mtp);
        let cfg = SpecConfig { noise, ..mode_cfg(SpecMode::Mtp) };
        let mut e = head_engine(&o, &sp, HeadScript::perfect(), cfg, true);
        run_to_end(&mut e, &[plain_request(1, true, 40)]);
        let keyed: Vec<Option<u64>> = e.backend.m.mtp_calls().iter().flat_map(|c| c.3.clone()).collect();
        assert!(!keyed.is_empty());
        assert!(keyed.iter().all(|k| *k == want_seed), "{noise:?}: {keyed:?}");
        // and a greedy request never carries one
        let mut e = head_engine(&o, &sp, HeadScript::perfect(), cfg, true);
        run_to_end(&mut e, &[plain_request(0, false, 40)]);
        assert!(e.backend.m.mtp_calls().iter().flat_map(|c| c.3.clone()).all(|k| k.is_none()));
    }
    // the keyed pick is worth having: a head that draws the target's own keyed sample is right where its argmax is not
    let (o, sp) = head_plan(2, SpecMode::Mtp);
    let sampled: Vec<Request> = (0..2)
        .map(|i| Request {
            sampling: Sampling { seed: 50 + i as u64, temperature: 1.0, top_k: 20, top_p: 1.0, min_p: 0.0 },
            ..plain_request(i, false, 120)
        })
        .collect();
    let acc = |noise| {
        let cfg = SpecConfig { noise, ..mode_cfg(SpecMode::Mtp) };
        let mut e = head_engine(&o, &sp, HeadScript::perfect(), cfg, true);
        run_to_end(&mut e, &sampled);
        let st = e.backend.spec.as_ref().unwrap().stats.clone();
        st.accepted[Source::Mtp as usize] as f64 / st.verified[Source::Mtp as usize].max(1) as f64
    };
    let (keyed, argmax) = (acc(gb_spec::DraftPick::Keyed), acc(gb_spec::DraftPick::Argmax));
    assert!(keyed > 0.999, "{keyed}");
    assert!(argmax < keyed - 0.05, "argmax {argmax} against keyed {keyed}: the sampled requests must have noise to key");
}

#[test]
fn two_streams_share_one_head_call_a_round() {
    let (o, sp) = head_plan(2, SpecMode::Mtp);
    let mut e = head_engine(&o, &sp, HeadScript::perfect(), mode_cfg(SpecMode::Mtp), true);
    run_to_end(&mut e, &[plain_request(0, false, 60), plain_request(1, true, 60)]);
    let calls = e.backend.m.mtp_calls();
    let asks: usize = calls.iter().map(|c| c.0.len()).sum();
    assert!(calls.iter().any(|c| c.0.len() == 2), "no call served both streams");
    assert!(calls.len() < asks, "{} calls for {asks} asks: one call a stream", calls.len());
    // and a slot is never asked twice in one call
    for c in &calls {
        let mut s = c.0.clone();
        s.sort_unstable();
        s.dedup();
        assert_eq!(s.len(), c.0.len(), "{c:?}");
    }
    assert!(e.backend.m.violations.is_empty());
}

#[test]
fn a_slot_that_the_arbiter_keeps_off_the_head_still_never_loses_it_to_a_full_store() {
    // a garbage head is judged poor within a few rounds: from then on the stream is serial or copy, and its store of pending rows fills
    // one row a round. The forced absorb keeps the slot's head alive and its store from overflowing (the fake switches a slot off when it
    // would, as the real store does). With a cheap draft call the arbiter may ride the absorb it has to make anyway with a one-deep
    // draft (the absorb is paid either way): the store is then emptied by a head round, not an absorb-only call; both are checked.
    let (o, sp) = head_plan(1, SpecMode::Auto);
    let mut e = head_engine(&o, &sp, HeadScript::Garbage, mode_cfg(SpecMode::Auto), true);
    // no probes of the head: its own tries absorb too, and this test is about the store the arbiter keeps off the head
    e.backend.spec.as_mut().unwrap().arbiter.cfg.probe_every = u32::MAX;
    let outs = run_to_end(&mut e, &[plain_request(0, false, 200)]);
    assert_eq!(outs[0].tokens.len(), 200);
    assert!(e.backend.m.head_lost.is_empty(), "the store overflowed or fell out of step: {:?}", e.backend.m.head_lost);
    let rounds: u64 = e.backend.spec.as_ref().unwrap().stats.rounds.iter().sum();
    // a store of 32 rows takes a window of 8 after 24: at least one head call every 25 rounds, whoever asked for it
    assert!(
        e.backend.m.mtp_draft_calls as u64 >= rounds / 26,
        "{} head calls in {rounds} rounds: a store was left to fill",
        e.backend.m.mtp_draft_calls
    );
    assert!(
        e.backend.spec.as_ref().unwrap().arbiter.stats.forced_absorbs
            + e.backend.spec.as_ref().unwrap().stats.rounds[Source::Mtp as usize]
            > 0
    );
    assert!(e.backend.m.violations.is_empty());

    // a head so dear that the arbiter never drafts with it: every store is made room for by an absorb-only call of depth 0
    let mut costs = CostTables::shipped();
    costs.mtp.lm_head = 1.0e6;
    costs.mtp.step = 1.0e6;
    let mut e2 =
        assemble_with(FakeRuntime::new(sp.slots).with_head(HeadScript::Garbage), &o, &sp, mode_cfg(SpecMode::Auto), &costs)
            .unwrap();
    e2.backend.spec.as_mut().unwrap().arbiter.cfg.probe_every = u32::MAX;
    run_to_end(&mut e2, &[plain_request(0, false, 200)]);
    assert!(e2.backend.m.head_lost.is_empty(), "{:?}", e2.backend.m.head_lost);
    let st = &e2.backend.spec.as_ref().unwrap().stats;
    assert_eq!(st.rounds[Source::Mtp as usize], 0, "the arbiter never picks a head this dear");
    assert!(st.absorb_only_calls > 0, "no absorb-only call in {} rounds", st.rounds.iter().sum::<u64>());
    let zero_depth = e2.backend.m.mtp_calls().iter().flat_map(|c| c.2.clone()).filter(|&d| d == 0).count();
    assert_eq!(zero_depth as u64, st.absorb_only_calls);
    assert!(e2.backend.spec.as_ref().unwrap().arbiter.stats.forced_absorbs > 0);
    assert!(e2.backend.m.violations.is_empty());
}

#[test]
fn a_wide_round_under_auto_does_not_absorb_a_full_store_it_gives_the_stream_up_and_a_narrow_one_keeps_it() {
    use gb_model::families::nemotron_h_omni::spec::WIDE_ROUND_STREAMS;
    // 40 streams of 120 tokens: the stores fill one row a round, and a round of 40 streams is wide. The head is judged poor, so the
    // arbiter would draft nothing, and the absorb that would make room is the cost this round shape must not pay.
    let n = WIDE_ROUND_STREAMS + 8;
    let (o, sp) = head_plan(n, SpecMode::Auto);
    let mut e = head_engine(&o, &sp, HeadScript::Garbage, mode_cfg(SpecMode::Auto), true);
    e.backend.spec.as_mut().unwrap().arbiter.cfg.probe_every = u32::MAX;
    let reqs: Vec<Request> = (0..n).map(|i| plain_request(i, false, 120)).collect();
    let outs = run_to_end(&mut e, &reqs);
    assert_eq!(outs.len(), n);
    assert!(outs.iter().all(|o| o.tokens.len() == 120));
    let st = &e.backend.spec.as_ref().unwrap().stats;
    assert!(st.wide_disabled > 0, "no stream gave the head up in a round of {n} streams");
    assert_eq!(st.absorb_only_calls, 0, "a wide round made the head absorb a full store");
    assert!(e.backend.m.violations.is_empty());
    // the same load in narrow rounds (a few streams at a time) keeps absorbing: a stream is never switched off there
    let (o, sp) = head_plan(2, SpecMode::Auto);
    let mut e = head_engine(&o, &sp, HeadScript::Garbage, mode_cfg(SpecMode::Auto), true);
    e.backend.spec.as_mut().unwrap().arbiter.cfg.probe_every = u32::MAX;
    run_to_end(&mut e, &[plain_request(0, false, 120), plain_request(1, false, 120)]);
    assert_eq!(e.backend.spec.as_ref().unwrap().stats.wide_disabled, 0);
}

#[test]
fn a_wide_depth_makes_every_ready_stream_of_a_wide_round_draft_and_the_text_is_the_serial_text() {
    use gb_model::families::nemotron_h_omni::spec::WIDE_ROUND_STREAMS;
    let n = WIDE_ROUND_STREAMS + 4;
    let (o, sp) = head_plan(n, SpecMode::Auto);
    let reqs: Vec<Request> = (0..n).map(|i| plain_request(i, false, 60)).collect();
    let cfg = SpecConfig { wide_depth: 3, ..mode_cfg(SpecMode::Auto) };
    let make = |drafting: bool| head_engine(&o, &sp, HeadScript::perfect(), cfg, drafting);
    // drafted == serial: what the head offers changes how many rounds, never a token
    let r = drafted_equals_serial(&make, &reqs).unwrap();
    assert!(r.accepted > 0);
    let mut e = make(true);
    run_to_end(&mut e, &reqs);
    let st = &e.backend.spec.as_ref().unwrap().stats;
    assert!(st.rounds[Source::Mtp as usize] > 0, "the head drafted in wide rounds");
    assert_eq!(st.wide_disabled, 0, "nothing was given up: the head was used");
    assert!(e.backend.m.violations.is_empty());
}

#[test]
fn spec_copy_and_none_never_ask_the_head_and_copy_still_drafts() {
    let reqs: Vec<Request> = (0..3).map(|i| repeating_request(i, 100)).collect();
    for mode in [SpecMode::Copy, SpecMode::None] {
        let (o, sp) = head_plan(3, mode);
        let make = |drafting: bool| head_engine(&o, &sp, HeadScript::perfect(), mode_cfg(mode), drafting);
        let r = drafted_equals_serial(&make, &reqs).unwrap();
        let mut e = make(true);
        run_to_end(&mut e, &reqs);
        assert_eq!(e.backend.m.mtp_draft_calls, 0, "{mode}");
        assert_eq!(e.backend.m.count(|c| matches!(c, Call::MtpDraft { .. })), 0);
        if mode == SpecMode::Copy {
            assert!(r.accepted > 0, "the copy drafter ran");
        } else {
            assert_eq!(r.drafted, 0, "--spec none never drafts");
        }
    }
}

#[test]
fn a_runtime_without_a_head_serves_auto_with_the_copy_drafter_alone() {
    let (o, sp) = head_plan(2, SpecMode::Auto);
    let reqs: Vec<Request> = (0..2).map(|i| repeating_request(i, 100)).collect();
    let make = |drafting: bool| {
        let mut o = o.clone();
        o.drafting = drafting;
        // the plan priced a head the loaded model does not have: over-reserved, never a refusal under `auto`
        assemble_with(FakeRuntime::new(sp.slots), &o, &sp, mode_cfg(SpecMode::Auto), &CostTables::shipped()).unwrap()
    };
    let r = drafted_equals_serial(&make, &reqs).unwrap();
    assert!(r.accepted > 0);
    let mut e = make(true);
    run_to_end(&mut e, &reqs);
    assert_eq!(e.backend.m.mtp_draft_calls, 0);
    assert!(e.backend.m.violations.is_empty());
}

#[test]
fn a_cost_table_of_the_callers_reaches_the_arbiter_and_a_malformed_one_is_refused_at_assembly() {
    let (o, sp) = head_plan(1, SpecMode::Auto);
    let e =
        assemble_with_cost(FakeRuntime::new(1).with_head(HeadScript::perfect()), &o, &sp, &[(1, 20.0), (8, 25.0), (64, 90.0)])
            .unwrap();
    let spec = e.backend.spec.as_ref().unwrap();
    use gb_spec::arbiter::ArbiterCost;
    assert!((spec.cost.verify_ms(1) - 20.0).abs() < 1e-9 && (spec.cost.verify_ms(8) - 25.0).abs() < 1e-9);
    // draft tables replaced too
    let mut costs = CostTables::shipped();
    costs.mtp.base = 7.0;
    let e =
        assemble_with(FakeRuntime::new(1).with_head(HeadScript::perfect()), &o, &sp, mode_cfg(SpecMode::Auto), &costs).unwrap();
    assert!((e.backend.spec.as_ref().unwrap().cost.mtp().base - 7.0).abs() < 1e-12);
    let bad = CostTables::shipped().with_verify(&[(2, 10.0)]);
    let e = assemble_with(FakeRuntime::new(1), &o, &sp, mode_cfg(SpecMode::Auto), &bad).err().unwrap();
    assert!(matches!(&e, ModelError::Config(m) if m.contains("verify table")), "{e}");
    // the shipped verify table is the engine's step table, and it is the measured one
    assert_eq!(CostTables::shipped().verify, STEP_MS.to_vec());
}

#[test]
fn a_reused_engine_serves_every_mode_in_turn_through_one_backend() {
    // a measurement moves `spec.cfg` between runs on one engine (`glm_spec_engine`): the arbiter and the depth follow it
    let (o, sp) = head_plan(2, SpecMode::Auto);
    let mut e = head_engine(&o, &sp, HeadScript::perfect(), mode_cfg(SpecMode::Auto), true);
    let reqs = [plain_request(0, false, 50), plain_request(1, true, 50)];
    let serial = {
        let mut s = head_engine(&o, &sp, HeadScript::perfect(), mode_cfg(SpecMode::Auto), false);
        run_to_end(&mut s, &reqs).into_iter().map(|x| x.tokens).collect::<Vec<_>>()
    };
    for (mode, depth) in [(SpecMode::Mtp, 2usize), (SpecMode::Mtp, 4), (SpecMode::Auto, 3), (SpecMode::Copy, 3)] {
        let spec = e.backend.spec.as_mut().unwrap();
        spec.cfg.mode = mode;
        spec.cfg.mtp_depth = depth;
        let before = e.backend.m.calls.len();
        let outs = run_to_end(&mut e, &reqs);
        assert_eq!(outs.into_iter().map(|x| x.tokens).collect::<Vec<_>>(), serial, "{mode} depth {depth}");
        let depths: Vec<usize> = e.backend.m.calls[before..]
            .iter()
            .filter_map(|c| if let Call::MtpDraft { depths, .. } = c { Some(depths.clone()) } else { None })
            .flatten()
            .collect();
        assert!(depths.iter().all(|&d| d <= depth), "{mode} depth {depth}: {depths:?}");
        if mode == SpecMode::Copy {
            assert!(depths.is_empty(), "the copy mode asked the head");
        }
    }
    assert!(e.backend.m.violations.is_empty(), "{:?}", e.backend.m.violations);
}

#[test]
fn copy_rounds_that_commit_eight_rows_a_round_never_overflow_a_store_the_head_is_kept_off() {
    // a head that is always wrong is judged poor within a few rounds, and these streams are served by the copy drafter: windows of eight
    // rows, whose commits fill a store four times as fast as serial rows. The forced absorb must keep every slot's head alive (the fake
    // models the runtime's store with the runtime's own book: `head_lost` lists every slot it switched off)
    let (o, sp) = head_plan(3, SpecMode::Auto);
    let reqs: Vec<Request> = (0..3).map(|i| repeating_request(i, 240)).collect();
    let make = |drafting: bool| head_engine(&o, &sp, HeadScript::Garbage, mode_cfg(SpecMode::Auto), drafting);
    drafted_equals_serial(&make, &reqs).unwrap();
    let mut e = make(true);
    run_to_end(&mut e, &reqs);
    assert!(e.backend.m.head_lost.is_empty(), "a slot lost its head: {:?}", e.backend.m.head_lost);
    let st = &e.backend.spec.as_ref().unwrap().stats;
    assert!(st.accepted[Source::Copy as usize] > 0, "the copy drafter served these streams");
    assert!(st.absorb_only_calls > 0, "a store near its limit was absorbed without a draft");
    let widest = e
        .backend
        .m
        .calls
        .iter()
        .filter_map(|c| if let Call::Commit { keep, .. } = c { Some(*keep) } else { None })
        .max()
        .unwrap();
    assert!(widest <= 8, "a round commits the base row and at most seven drafts: {widest}");
    assert!(e.backend.m.violations.is_empty());
}

#[test]
fn the_plan_and_the_model_agree_on_the_pool_the_head_adds_a_stripe_to() {
    use gb_model::families::nemotron_h_omni::model::KvGeom;
    for spec in [SpecMode::Copy, SpecMode::Auto] {
        let (_, sp) = head_plan(2, spec);
        // the model's pool for `sp.model.kv_pages` pages: 8 attention layers of the trunk, and the head's stripe when the options carry it
        let layers = 8 + usize::from(sp.model.mtp);
        // the dtype the process is configured for (REXL3_KV_DTYPE): the plan prices the same
        let dtype = gb_cuda::nemotron_attn::configured_kv_dtype().unwrap();
        let g = KvGeom { pages: sp.model.kv_pages, layers, dtype };
        assert_eq!(sp.model.mtp, sp.mtp);
        assert_eq!(
            g.token_bytes() as u64,
            sp.profile.kv_bytes_per_token(),
            "{spec}: a token costs the same in the plan and in the pool"
        );
        assert_eq!(
            g.pool_bytes() as u64,
            sp.plan.kv_blocks as u64 * sp.plan.block_bytes,
            "{spec}: the pool is the pages the plan counted"
        );
        // and the plan's page is the model's page
        assert_eq!(KvGeom { pages: 1, layers, dtype }.pool_bytes() as u64, sp.plan.block_bytes);
    }
}

// ---- ModelOptions.mtp: the plan prices the head exactly when the model allocates it

#[test]
fn the_plans_model_options_carry_the_head_flag_the_plan_priced() {
    for (spec, drafting, drafters, want) in [
        (SpecMode::Auto, true, vec!["mtp", "copy"], true),
        (SpecMode::Mtp, true, vec!["mtp", "copy"], true),
        (SpecMode::Copy, true, vec!["mtp", "copy"], false),
        (SpecMode::None, true, vec!["mtp", "copy"], false),
        (SpecMode::Auto, false, vec!["mtp", "copy"], false),
        (SpecMode::Auto, true, vec!["copy"], false),
    ] {
        let mut fam = family_plan();
        fam.drafters = drafters.clone();
        let mut o = options();
        o.spec = spec;
        o.drafting = drafting;
        let sp = plan_serving(&fam, &o, &Host::new(100, 60).inputs()).unwrap();
        assert_eq!((sp.mtp, sp.model.mtp), (want, want), "{spec} drafting {drafting} {drafters:?}");
        assert_eq!(head_priced_if(&fam, &o, true), want);
        // `REXL3_MTP=off` (the model does not load the head): not priced
        assert!(!head_priced_if(&fam, &o, false));
    }
}

// ---- the wide decode window (AGG): priced exactly, and what it leaves for slots

#[test]
fn a_wide_window_is_priced_in_its_records_and_scratch_and_the_slot_table_follows() {
    use gb_model::families::nemotron_h_omni::memory::{NemotronMemory, ACTIVATION_RESERVE};
    use gb_model::family::MemorySource;
    use gb_model::memory::MemoryOptions;
    use gb_model::plan::{plan_memory, MIN_TOKENS_PER_SLOT};
    let m = NemotronMemory::new(&cfg(), 0);
    let head_listed = 5_884_651_520u64;
    // the weights the repaired pack lists (head included), and without the head: what a boot of --spec none holds
    let (w_head, w_nohead) = ((63.60 * GIB as f64) as u64, (63.60 * GIB as f64) as u64 - head_listed);
    let mopts = |window: usize, prefill: usize, mtp: bool| MemoryOptions {
        window_rows: window,
        prefill_rows: prefill,
        max_context: 40_960,
        mtp,
        vision: None,
        trees: false,
    };
    // the price of the window: its records are 2.5 MB a row over the 40 Mamba layers, the step buffers and the large-row MoE scratch follow
    // the widest step, and the head's window taps the window (4 KiB x 4 B a row)
    for (window, prefill) in [(64usize, 64usize), (64, 256), (128, 64), (128, 256), (192, 256), (256, 256)] {
        for mtp in [false, true] {
            let o = mopts(window, prefill, mtp);
            let p = match m.profile(&o) {
                Ok(p) => p,
                Err(e) => panic!("{window}/{prefill} mtp {mtp}: {e}"),
            };
            assert_eq!(p.window_record_bytes(), window as u64 * 2_508_800, "{window} rows of records");
            let rows = window.max(prefill) as u64;
            assert_eq!(
                p.scratch_bytes(),
                m.scratch_for(rows, 40_960, mtp, window as u64).total(),
                "{window}/{prefill} mtp {mtp}"
            );
            if mtp {
                let head_taps = m.mtp_head_for_window(rows, window as u64).tap_window;
                assert_eq!(head_taps, window.max(64) as u64 * 4096 * 4);
            }
        }
    }
    // a window of 64 rows is the profile it always was, at any prefill width
    assert_eq!(
        m.profile(&mopts(64, 256, false)).unwrap().scratch_bytes(),
        m.scratch(256, 40_960).total(),
        "the narrow window's scratch is unchanged"
    );
    // the lag rings (a fold period of up to 16 steps, 40 layers, 256 slots: 21,248 B a row) are not in the profile: they sit inside the
    // same blanket with the records and the scratch, and the three together stay below it
    let ring_bytes = |slots: u64, k: u64| 40 * slots * (k - 1) * 21_248;
    let wide_all = m.profile(&mopts(256, 256, true)).unwrap();
    assert!(
        wide_all.window_record_bytes() + wide_all.scratch_bytes() + ring_bytes(256, 16) < ACTIVATION_RESERVE,
        "{}",
        wide_all.window_record_bytes() + wide_all.scratch_bytes() + ring_bytes(256, 16)
    );
    // the records and the scratch of a 256-row window are well inside the blanket the planner holds back, so a wide window costs no slots
    let wide = m.profile(&mopts(256, 256, true)).unwrap();
    assert!(
        wide.window_record_bytes() + wide.scratch_bytes() < ACTIVATION_RESERVE,
        "{}",
        wide.window_record_bytes() + wide.scratch_bytes()
    );
    assert_eq!(wide.engine_reserve_bytes(), m.profile(&mopts(64, 64, true)).unwrap().engine_reserve_bytes());

    // the slots a box holds: MemAvailable after the weights, the 8 GiB floor and the planner's reserve; a slot is 170,229,760 B of state
    // (plus the head's 512 KiB tap store) and the KV pool is one `--max-model-len` of tokens
    let floor = 8 * GIB;
    let max_slots = |avail_gib: u64, weights: u64, window: usize, mtp: bool, max_len: usize| -> usize {
        let p = m.profile(&MemoryOptions { max_context: max_len, ..mopts(window, 256, mtp) }).unwrap();
        let free = (avail_gib * GIB).saturating_sub(weights);
        let cap = max_len as u64;
        (1..=1024usize)
            .take_while(|&s| match plan_memory(&p, free, floor, s, cap, 0) {
                Ok(plan) => plan.kv_tokens >= (s as u64 * MIN_TOKENS_PER_SLOT).min(cap),
                Err(_) => false,
            })
            .last()
            .unwrap_or(0)
    };
    eprintln!("max slots (floor 8 GiB, window rows 256, prefill 256; MemAvailable of node 1 / 100 / node 4):");
    eprintln!(
        "  max-model-len | head on ({:.1} GiB weights): 100 / 110 / 116 GiB | head off ({:.1} GiB): 100 / 110 / 116 GiB",
        w_head as f64 / GIB as f64,
        w_nohead as f64 / GIB as f64
    );
    for len in [1024usize, 2048, 4096, 40_960] {
        eprintln!(
            "  {len:>13} | {:>8} {:>6} {:>6} | {:>8} {:>6} {:>6}",
            max_slots(100, w_head, 256, true, len),
            max_slots(110, w_head, 256, true, len),
            max_slots(116, w_head, 256, true, len),
            max_slots(100, w_nohead, 256, false, len),
            max_slots(110, w_nohead, 256, false, len),
            max_slots(116, w_nohead, 256, false, len)
        );
    }
    // the table's own checks: the head's weights and tap stores cost slots, a wider window costs none that matter, and 128 slots fit
    // without the head
    let (on, off) = (max_slots(110, w_head, 128, true, 40_960), max_slots(110, w_nohead, 128, false, 40_960));
    assert!(off > on, "the head's 5.5 GiB and its per-slot tap store: {off} vs {on}");
    assert!(off >= 128, "128 slots fit without the head: {off}");
    assert_eq!(
        max_slots(110, w_nohead, 64, false, 40_960),
        off,
        "the window's records and scratch live inside the planner's blanket"
    );
}

#[test]
fn a_tensor_parallel_group_refuses_the_wide_window_the_lagged_and_fused_decode_and_the_burst_by_name() {
    use gb_model::families::nemotron_h_omni::engine::check_tp_plain_with;
    // the 64-row window with the plain decode and the serial prefill is what a group is built for
    check_tp_plain_with(64, 0, false, None).unwrap();
    for (window, lag, fused, burst, name) in [
        (128usize, 0usize, false, None, "REXL3_NEMO_WINDOW_ROWS"),
        (256, 8, true, Some(32), "REXL3_NEMO_WINDOW_ROWS"),
        (64, 8, false, None, "REXL3_NEMO_LAG"),
        (64, 0, true, None, "REXL3_NEMO_FUSED_DECODE"),
        (64, 0, false, Some(32), "REXL3_NEMO_PREFILL_BURST"),
    ] {
        let e = check_tp_plain_with(window, lag, fused, burst).expect_err(name).to_string();
        assert!(e.contains(name) && e.contains("tensor parallelism"), "{name}: {e}");
    }
}

#[test]
fn the_wide_window_settings_are_parsed_and_refused_by_name() {
    use gb_model::families::nemotron_h_omni::engine::{fused_decode_with, prefill_burst_with, window_rows_with};
    let o = options();
    assert_eq!(window_rows_with(&o, None).unwrap(), 64);
    let mut o2 = options();
    o2.window_rows = 128;
    assert_eq!(window_rows_with(&o2, None).unwrap(), 128);
    // the variable replaces the option (it reaches past the shared option's 128 rows)
    assert_eq!(window_rows_with(&o, Some("192")).unwrap(), 192);
    assert_eq!(window_rows_with(&o, Some(" 256 ")).unwrap(), 256);
    for bad in ["257", "63", "0", "wide", ""] {
        let e = window_rows_with(&o, Some(bad)).expect_err(bad).to_string();
        assert!(e.contains("REXL3_NEMO_WINDOW_ROWS") || e.contains("--window-rows"), "{bad}: {e}");
    }
    // the fused decode step is on for a wide window and off for a narrow one, unless asked
    assert!(!fused_decode_with(64, None).unwrap());
    assert!(fused_decode_with(65, None).unwrap());
    assert!(fused_decode_with(64, Some("1")).unwrap());
    assert!(!fused_decode_with(128, Some("off")).unwrap());
    assert!(fused_decode_with(128, Some("maybe")).is_err());
    // the lagged decode: off for a narrow window, the default fold period for a wide one, the variable's value otherwise
    use gb_model::families::nemotron_h_omni::engine::{lag_period_with, DEFAULT_WIDE_LAG};
    assert_eq!(lag_period_with(64, None).unwrap(), 0);
    assert_eq!(lag_period_with(128, None).unwrap(), DEFAULT_WIDE_LAG);
    assert_eq!(lag_period_with(128, Some("4")).unwrap(), 4);
    assert_eq!(lag_period_with(128, Some("0")).unwrap(), 0);
    assert_eq!(lag_period_with(128, Some("1")).unwrap(), 0);
    assert_eq!(lag_period_with(64, Some("8")).unwrap(), 8);
    assert!(lag_period_with(128, Some("17")).is_err() && lag_period_with(128, Some("many")).is_err());
    // so is the prefill burst
    assert_eq!(prefill_burst_with(64, None).unwrap(), None);
    assert_eq!(prefill_burst_with(128, None).unwrap(), Some(32));
    assert_eq!(prefill_burst_with(64, Some("8")).unwrap(), Some(8));
    assert_eq!(prefill_burst_with(128, Some("off")).unwrap(), None);
    assert!(prefill_burst_with(128, Some("many")).is_err());
}

#[test]
fn the_resident_weights_count_the_heads_tensors_only_when_it_loads() {
    let head = 5_884_651_520u64;
    let listed = 60 * GIB + head;
    let mut o = options();
    o.spec = SpecMode::Auto;
    assert_eq!(resident_weight_bytes(listed, head, &mtp_family(), &o), listed, "the head loads: its tensors are resident");
    o.spec = SpecMode::Copy;
    assert_eq!(resident_weight_bytes(listed, head, &mtp_family(), &o), 60 * GIB, "no head, no 5.5 GiB held back for it");
    // a pack that does not offer the head: its tensors are listed and never uploaded
    o.spec = SpecMode::Auto;
    assert_eq!(resident_weight_bytes(listed, head, &family_plan(), &o), 60 * GIB);
    assert_eq!(resident_weight_bytes(5, head, &family_plan(), &o), 0, "saturating");
}

#[test]
fn the_restricted_heads_table_comes_out_of_the_free_memory_with_the_weights() {
    // MT-A: 256 / 384 / 512 MiB of table at 32k / 48k / 64k ids (plus 4 bytes an id for the map)
    assert_eq!(sub_head_bytes(32_768), 268_566_528);
    assert_eq!(sub_head_bytes(65_536) >> 20, 512);
    let (o, _) = head_plan(2, SpecMode::Auto);
    let b = Host::new(100, 60);
    let full = plan_serving_with(&mtp_family(), &o, &b.inputs(), 0).unwrap();
    let restricted = plan_serving_with(&mtp_family(), &o, &b.inputs(), sub_head_bytes(32_768)).unwrap();
    assert_eq!((full.sub_head_bytes, restricted.sub_head_bytes), (0, 268_566_528));
    // the free memory is the planning quantum of what is left: smaller by the table (to the quantum)
    assert!(restricted.free_after_weights < full.free_after_weights);
    assert!(full.free_after_weights - restricted.free_after_weights <= 1 << 30);
    // on a box where the pool is what the memory decides, the table shrinks the pool by its bytes, to a page
    let tight = Host::new(75, 60);
    let mut o2 = options();
    o2.spec = SpecMode::Auto;
    o2.slots = 1;
    let a = plan_serving_with(&mtp_family(), &o2, &tight.inputs(), 0).unwrap();
    let c = plan_serving_with(&mtp_family(), &o2, &tight.inputs(), 1 << 30).unwrap();
    assert!(c.plan.kv_tokens < a.plan.kv_tokens, "{} against {}", c.plan.kv_tokens, a.plan.kv_tokens);
    // the plan is the same shape otherwise: the head's pages, tap stores and buffers are in both
    assert_eq!(a.profile, c.profile);
    // not priced without the head
    let none = plan_serving_with(&family_plan(), &o2, &tight.inputs(), 0).unwrap();
    assert!(!none.mtp && none.sub_head_bytes == 0);
}

#[test]
fn a_vocabulary_file_is_counted_as_the_model_counts_it() {
    let dir = std::env::temp_dir().join(format!("rexl3-vocab-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let write = |name: &str, text: &str| {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        p
    };
    // ids, comments, repeats and any order: sorted and de-duplicated, as `mtp_restrict_vocab` does
    let p = write("ok", "# a vocabulary\n5 3 9\n3 # again\n\n9 11\n");
    assert_eq!(restricted_rows_of_file(&p).unwrap(), 4);
    for (name, text, needle) in [("bad", "1 2 x\n", "not a token id"), ("empty", "# nothing\n\n", "empty")] {
        let e = restricted_rows_of_file(&write(name, text)).unwrap_err();
        assert!(matches!(&e, ModelError::Config(m) if m.contains("REXL3_MTP_VOCAB") && m.contains(needle)), "{e}");
    }
    let e = restricted_rows_of_file(&dir.join("missing")).unwrap_err();
    assert!(matches!(&e, ModelError::Config(m) if m.contains("REXL3_MTP_VOCAB") && m.contains("missing")), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_model_with_a_head_the_plan_did_not_price_is_refused_when_it_would_draft_with_it() {
    // the pack's offer is missing from the plan (`drafters` has no `mtp`): the plan did not price the head's stripe, tap stores or buffers
    let mut o = options();
    o.spec = SpecMode::Auto;
    let sp = plan_serving(&family_plan(), &o, &Host::new(100, 60).inputs()).unwrap();
    assert!(!sp.mtp);
    let e = assemble(FakeRuntime::new(sp.slots).with_head(HeadScript::perfect()), &o, &sp)
        .err()
        .expect("an unpriced head must be refused");
    assert!(matches!(&e, ModelError::Config(m) if m.contains("did not price") && m.contains("ModelOptions")), "{e}");
    // `--spec copy` and `none` never draft with it: whatever the model carries is not the plan's business (it is never asked)
    for ok in [SpecMode::Copy, SpecMode::None] {
        let mut o = options();
        o.spec = ok;
        let sp = plan_serving(&family_plan(), &o, &Host::new(100, 60).inputs()).unwrap();
        assert!(assemble(FakeRuntime::new(sp.slots).with_head(HeadScript::perfect()), &o, &sp).is_ok(), "{ok}");
    }
    // a model without the head under a plan that priced it is only over-reserved: served, with the copy drafter
    let (o, sp) = head_plan(2, SpecMode::Auto);
    assert!(sp.mtp && assemble(FakeRuntime::new(sp.slots), &o, &sp).is_ok());
}

#[test]
fn a_restricted_head_of_the_measured_size_is_priced_with_the_measured_table_and_other_sizes_by_their_lm_head() {
    use gb_model::families::nemotron_h_omni::engine::restricted_costs;
    use gb_model::families::nemotron_h_omni::spec::{LM_HEAD_RESTRICTED_ROWS, MTP_COST, MTP_COST_RESTRICTED};
    let t = CostTables::shipped();
    assert_eq!(LM_HEAD_RESTRICTED_ROWS, 49_152, "the measured table is the default list's");
    assert_eq!(
        restricted_costs(&t, 49_152),
        MTP_COST_RESTRICTED,
        "the 48k head is the measured one, not the full table with a smaller lm_head"
    );
    let r64 = restricted_costs(&t, 65_536);
    let r32 = restricted_costs(&t, 32_768);
    assert!(
        r32.lm_head < MTP_COST_RESTRICTED.lm_head && MTP_COST_RESTRICTED.lm_head < r64.lm_head && r64.lm_head < MTP_COST.lm_head
    );
    // the layers are the full head's: only the lm_head moved
    assert_eq!(
        (r64.step, r64.step_per_stream, r64.rows.clone()),
        (MTP_COST.step, MTP_COST.step_per_stream, MTP_COST.rows.clone())
    );
    // a caller's own tables are what is scaled
    let mut mine = CostTables::shipped();
    mine.mtp.lm_head = 8.42;
    assert!((restricted_costs(&mine, LM_HEAD_ROWS).lm_head - 8.42).abs() < 1e-9);
}

// ---- the restricted draft vocabulary is the default (49,152 ids, `mtp_vocab_48k.txt`)

/// The default list: what the plan prices, what the model's `load` builds its sub-table from and what the arbiter assumes are ONE list.
#[test]
fn the_default_draft_vocabulary_is_priced_whenever_the_head_is_and_the_arbiter_prices_its_draft_calls_with_it() {
    use gb_model::families::nemotron_h_omni::engine::{
        plan_serving_vocab, restricted_head_rows_for, spec_settings, SUB_HEAD_ROW_BYTES,
    };
    use gb_model::families::nemotron_h_omni::mtp::{restricted_vocab_ids_for, VocabSetting};
    use gb_model::families::nemotron_h_omni::spec::{LM_HEAD_RESTRICTED_ROWS, MTP_COST, MTP_COST_RESTRICTED};
    let list = restricted_vocab_ids_for(&VocabSetting::Default).unwrap().unwrap();
    assert_eq!(list.len(), 49_152);
    assert_eq!(restricted_head_rows_for(&VocabSetting::Default).unwrap(), Some(49_152));
    assert_eq!(restricted_head_rows_for(&VocabSetting::Off).unwrap(), None);
    assert_eq!(LM_HEAD_RESTRICTED_ROWS, list.len(), "the measured restricted table is for the default list's size");
    // 384 MiB of rows and 192 KiB of id map
    assert_eq!(sub_head_bytes(list.len()), 402_849_792);
    assert_eq!(sub_head_bytes(list.len()), list.len() as u64 * SUB_HEAD_ROW_BYTES);
    assert_eq!(sub_head_bytes(list.len()) >> 20, 384);
    let b = Host::new(100, 60);
    for spec in [SpecMode::Auto, SpecMode::Mtp] {
        let mut o = options();
        o.spec = spec;
        let on = plan_serving_vocab(&mtp_family(), &o, &b.inputs(), &VocabSetting::Default).unwrap();
        assert!(on.mtp, "{spec}: the head is priced");
        assert_eq!(on.sub_head_bytes, sub_head_bytes(49_152), "{spec}: and with it the sub-head the default list builds");
        let off = plan_serving_vocab(&mtp_family(), &o, &b.inputs(), &VocabSetting::Off).unwrap();
        assert!(off.mtp && off.sub_head_bytes == 0, "{spec}: REXL3_MTP_VOCAB=off: the full head, no sub-table priced");
        // the memory the sub-table takes is out of the free memory, to the planning quantum
        assert!(on.free_after_weights < off.free_after_weights);
        assert!(off.free_after_weights - on.free_after_weights <= 1 << 30);
        // the arbiter: restricted costs and priors exactly when the plan priced the sub-table
        let (cfg, costs) = spec_settings(&o, &on, None).unwrap();
        assert!(cfg.restricted_head, "{spec}: the draft calls are priced with the restricted head");
        assert_eq!(costs.mtp_restricted, MTP_COST_RESTRICTED);
        assert_eq!(costs.mtp, MTP_COST, "the full head's table is untouched");
        let (cfg, _) = spec_settings(&o, &off, None).unwrap();
        assert!(!cfg.restricted_head, "{spec}: off: the full head's table");
    }
    // a file replaces the list: its ids are what is priced (and the table is the interpolation for that size)
    let dir = std::env::temp_dir().join(format!("rexl3-default-vocab-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("v");
    let ids: Vec<String> = (0..20_000u32).map(|i| (i * 3).to_string()).collect();
    std::fs::write(&p, format!("# 20,000 ids\n{}\n", ids.join(" "))).unwrap();
    let mut o = options();
    o.spec = SpecMode::Auto;
    let file = plan_serving_vocab(&mtp_family(), &o, &b.inputs(), &VocabSetting::File(p.clone())).unwrap();
    assert_eq!(file.sub_head_bytes, sub_head_bytes(20_000));
    let (cfg, costs) = spec_settings(&o, &file, None).unwrap();
    assert!(cfg.restricted_head);
    assert!(costs.mtp_restricted.lm_head < MTP_COST_RESTRICTED.lm_head, "{}", costs.mtp_restricted.lm_head);
    let _ = std::fs::remove_dir_all(&dir);
    // a missing file is an error that names the variable, not a silent fall back to the full head
    let e = plan_serving_vocab(&mtp_family(), &o, &b.inputs(), &VocabSetting::File("/nonexistent/vocab".into())).unwrap_err();
    assert!(matches!(&e, ModelError::Config(m) if m.contains("REXL3_MTP_VOCAB=/nonexistent/vocab")), "{e}");
    // no head priced, no sub-table: `--spec copy` / `none`, and a pack that does not offer the head
    for spec in [SpecMode::Copy, SpecMode::None] {
        let mut o = options();
        o.spec = spec;
        let sp = plan_serving_vocab(&mtp_family(), &o, &b.inputs(), &VocabSetting::Default).unwrap();
        assert!(!sp.mtp && sp.sub_head_bytes == 0, "{spec}");
        assert!(!spec_settings(&o, &sp, None).unwrap().0.restricted_head, "{spec}");
    }
    let mut o = options();
    o.spec = SpecMode::Auto;
    let none = plan_serving_vocab(&family_plan(), &o, &b.inputs(), &VocabSetting::Default).unwrap();
    assert!(!none.mtp && none.sub_head_bytes == 0, "a pack without the head has no sub-table to price");
}

/// A two-rank group (rank 0's engine) prices rounds with the group's measured tables, a single node with the single node's; the restricted
/// head's lm_head term follows the sub-table's size in both. (The environment is not touched: `REXL3_NEMO_COSTS`, `REXL3_TP_COMM` and
/// `REXL3_TP_HEAD` are read as the process has them, and none is set by the suite.)
#[test]
fn a_two_rank_group_is_priced_with_the_groups_tables_and_a_single_node_with_its_own() {
    use gb_model::families::nemotron_h_omni::engine::{plan_serving_vocab, spec_settings};
    use gb_model::families::nemotron_h_omni::mtp::VocabSetting;
    use gb_model::families::nemotron_h_omni::spec::{
        MTP_COST, MTP_COST_RESTRICTED, MTP_COST_RESTRICTED_TP, MTP_COST_TP, VERIFY_MS, VERIFY_MS_TP,
    };
    use gb_model::serve::TpOptions;
    if ["REXL3_NEMO_COSTS", "REXL3_TP_COMM", "REXL3_TP_HEAD"].iter().any(|k| std::env::var_os(k).is_some()) {
        eprintln!("not run: the environment names a TP table or layout switch");
        return;
    }
    let b = Host::new(100, 60);
    let mut o = options();
    o.spec = SpecMode::Auto;
    let sp = plan_serving_vocab(&mtp_family(), &o, &b.inputs(), &VocabSetting::Default).unwrap();
    // one node: the shipped tables
    let (_, single) = spec_settings(&o, &sp, None).unwrap();
    assert_eq!(
        (single.verify.clone(), single.mtp.clone(), single.mtp_restricted.clone()),
        (VERIFY_MS.to_vec(), MTP_COST, MTP_COST_RESTRICTED)
    );
    // a single-rank `--tp-ranks 1` is not a group
    o.tp = Some(TpOptions::new(1, 0, "127.0.0.1:29600"));
    assert_eq!(spec_settings(&o, &sp, None).unwrap().1, single);
    // two ranks: the group's tables, restricted to the default list's size as measured
    o.tp = Some(TpOptions::new(2, 0, "127.0.0.1:29600"));
    let (cfg, group) = spec_settings(&o, &sp, None).unwrap();
    assert!(cfg.restricted_head);
    assert_eq!(group.verify, VERIFY_MS_TP.to_vec());
    assert_eq!((group.mtp.clone(), group.mtp_restricted.clone()), (MTP_COST_TP, MTP_COST_RESTRICTED_TP));
    // a caller's verify table still wins, and the draft tables stay the group's
    let (_, mine) = spec_settings(&o, &sp, Some(&[(1, 20.0), (8, 30.0)])).unwrap();
    assert_eq!((mine.verify, mine.mtp_restricted), (vec![(1, 20.0), (8, 30.0)], MTP_COST_RESTRICTED_TP));
}

/// Three ranks or more: no group of that size was ever measured, so the arbiter prices with the single node's tables (the online
/// scale follows the rounds) and the boot labels the basis `unmeasured`; the measured two-rank tables stay for exactly two ranks.
#[test]
fn a_group_of_three_ranks_or_more_is_priced_with_the_single_node_tables_and_labelled_unmeasured() {
    use gb_model::families::nemotron_h_omni::engine::{plan_serving_vocab, spec_settings, tp_cost_basis, tp_costs_wanted};
    use gb_model::families::nemotron_h_omni::mtp::VocabSetting;
    use gb_model::families::nemotron_h_omni::spec::{
        CostBasis, MTP_COST, MTP_COST_RESTRICTED, MTP_COST_RESTRICTED_TP, MTP_COST_TP, VERIFY_MS, VERIFY_MS_TP,
    };
    use gb_model::serve::TpOptions;
    if ["REXL3_NEMO_COSTS", "REXL3_TP_COMM", "REXL3_TP_HEAD"].iter().any(|k| std::env::var_os(k).is_some()) {
        eprintln!("not run: the environment names a TP table or layout switch");
        return;
    }
    // the mapping itself, without options
    assert_eq!(CostTables::for_group(1, true).1, CostBasis::SingleNode);
    assert_eq!(CostTables::for_group(2, true).1, CostBasis::MeasuredTwoRanks);
    assert_eq!(CostTables::for_group(2, true).0.verify, VERIFY_MS_TP.to_vec());
    assert_eq!(CostTables::for_group(2, false).0.mtp, MTP_COST, "a head that is not split keeps the single node's draft table");
    for r in [3usize, 4, 8] {
        let (t, basis) = CostTables::for_group(r, true);
        assert_eq!(basis, CostBasis::Unmeasured { ranks: r });
        assert_eq!(basis.label(), "unmeasured");
        assert_eq!((t.verify, t.mtp, t.mtp_restricted), (VERIFY_MS.to_vec(), MTP_COST, MTP_COST_RESTRICTED));
    }
    // through the boot's own entry point
    let b = Host::new(100, 60);
    let mut o = options();
    o.spec = SpecMode::Auto;
    let sp = plan_serving_vocab(&mtp_family(), &o, &b.inputs(), &VocabSetting::Default).unwrap();
    o.tp = Some(TpOptions::new(2, 0, "127.0.0.1:29600"));
    assert_eq!(tp_cost_basis(&o), CostBasis::MeasuredTwoRanks);
    assert!(tp_costs_wanted(&o));
    let (_, two) = spec_settings(&o, &sp, None).unwrap();
    assert_eq!((two.verify, two.mtp, two.mtp_restricted), (VERIFY_MS_TP.to_vec(), MTP_COST_TP, MTP_COST_RESTRICTED_TP));
    for r in [3usize, 4, 8] {
        o.tp = Some(TpOptions::new(r, 0, "127.0.0.1:29600"));
        assert_eq!(tp_cost_basis(&o), CostBasis::Unmeasured { ranks: r });
        assert!(!tp_costs_wanted(&o), "the measured two-rank tables are for two ranks");
        let (cfg, t) = spec_settings(&o, &sp, None).unwrap();
        assert!(cfg.restricted_head, "the restricted head's lm_head term still follows the sub-table");
        assert_eq!((t.verify, t.mtp), (VERIFY_MS.to_vec(), MTP_COST), "{r} ranks: the single node's verify and draft tables");
        assert_eq!(t.mtp_restricted, MTP_COST_RESTRICTED, "{r} ranks: the default list's restricted table is the measured size");
    }
}

/// What is priced is what is allocated: the sub-table the model builds for the default list is `sub_head_bytes` of its ids, and the
/// memory model's own count (`NemotronMemory::with_restricted_head`) agrees with the plan's.
#[test]
fn what_the_plan_prices_for_the_sub_head_is_what_the_memory_model_and_the_table_count() {
    use gb_model::families::nemotron_h_omni::mtp::{restricted_vocab_ids_for, VocabSetting};
    let ids = restricted_vocab_ids_for(&VocabSetting::Default).unwrap().unwrap();
    let mem = NemotronMemory::new(&cfg(), 0).with_restricted_head(ids.len());
    let head = mem.mtp_head();
    assert_eq!(head.sub_head, sub_head_bytes(ids.len()), "the memory model and the plan count the same bytes");
    // bf16 [rows][4096] and an i32 map: the layout `SubHead` allocates
    assert_eq!(head.sub_head, ids.len() as u64 * 4096 * 2 + ids.len() as u64 * 4);
    // the list is what `mtp_restrict_vocab` takes: ascending, distinct, inside the vocabulary
    assert!(ids.windows(2).all(|w| w[0] < w[1]));
    assert!((*ids.last().unwrap() as usize) < cfg().vocab);
}

/// The engine under the DEFAULT restricted head (the plan priced the 49,152-id sub-table: the restricted cost table and the restricted
/// prior), not the unrestricted config the other engine tests start from: a head that is always right is used, and one that is never right
/// (text in a script the list leaves out) is dropped after a few rounds of each sequence.
#[test]
fn under_the_default_restricted_head_a_right_head_is_used_and_a_head_that_is_never_right_is_dropped_within_a_few_rounds() {
    use gb_model::families::nemotron_h_omni::engine::plan_serving_vocab;
    use gb_model::families::nemotron_h_omni::mtp::VocabSetting;
    let mut o = options();
    o.spec = SpecMode::Auto;
    let sp = plan_serving_vocab(&mtp_family(), &o, &Host::new(100, 60).inputs(), &VocabSetting::Default).unwrap();
    assert!(sp.mtp && sp.sub_head_bytes == sub_head_bytes(49_152));
    let reqs: Vec<Request> = (0..2).map(|i| plain_request(i, false, 120)).collect();
    // never right
    let mut e = assemble(FakeRuntime::new(sp.slots).with_head(HeadScript::Garbage), &o, &sp).unwrap();
    run_to_end(&mut e, &reqs);
    let spec = e.backend.spec.as_ref().unwrap();
    assert!(spec.cfg.restricted_head, "the arbiter is told the head is restricted");
    let (total, mtp) = (spec.stats.rounds.iter().sum::<u64>(), spec.stats.rounds[Source::Mtp as usize]);
    assert_eq!(spec.stats.accepted[Source::Mtp as usize], 0);
    // the prior counts as three observations and each wrong draft is one more, then the head is only probed now and then: about ten
    // head rounds in a sequence of 120 tokens, under one round in eight, not a round in two
    assert!(mtp <= 12 * reqs.len() as u64 && mtp * 8 < total, "{mtp} head rounds of {total} for {} sequences", reqs.len());
    eprintln!(
        "a never-right head under the restricted prior: {mtp} head rounds of {total} for {} sequences of 120 tokens",
        reqs.len()
    );
    // always right
    let mut e = assemble(FakeRuntime::new(sp.slots).with_head(HeadScript::perfect()), &o, &sp).unwrap();
    run_to_end(&mut e, &reqs);
    let spec = e.backend.spec.as_ref().unwrap();
    let (total, mtp) = (spec.stats.rounds.iter().sum::<u64>(), spec.stats.rounds[Source::Mtp as usize]);
    assert!(mtp * 10 >= total * 9, "a head that is always right drafts nearly every round: {mtp} of {total}");
}
