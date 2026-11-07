//! Images and video through the MiniCPM-V-4.7 engine on a real pack (E5b of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`): the tower in the
//! engine's process, the format's prompts, the splice, the engine's invariants with an image stream among text streams, and the Leg C
//! oracle (engine path against the Hugging Face bf16 reference). GPU; every test is `#[ignore]`, prints `not run:` without its inputs.
//!
//! ```text
//! # a full-pack boot is 27 GiB resident: exclusive lock for the whole run, a small KV pool, one test at a time
//! flock -x -w 3600 /tmp/rexl3-gpu.lock env REXL3_MINICPM_DIR=~/models/minicpm-v-4.7-35b-a3b/packs/q3-6.00bpw \
//!     REXL3_MINICPM_KV_TOKENS=16384 REXL3_MINICPM_MM_DIR=~/projects/minicpm-mm \
//!     cargo test --release -p gb-server --test minicpmv4_7_mm -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! The Leg C pair: `leg_c_continuations` (the engine's greedy 64 tokens a case, written to `REXL3_MINICPM_MM_OUT/cont.json` with the
//! engine's prompt ids), then `tools/minicpm_v47/mm_ref_logits.py ref --cont cont.json` (the reference teacher-forced on prompt +
//! continuation), then `leg_c_kl` (the engine teacher-forced on the same sequence, KL(reference || engine) on the continuation rows).

use gb_engine::harness::serial_batched_equals_solo;
use gb_engine::{
    Backend, CutPolicy, Drafter, Engine, EngineConfig, ImageSpan, NoDrafts, Output, PrefixConfig, Request, RowOutput, RowRequest,
};
use gb_model::families::minicpmv4_7::backend::MiniCpmBackend;
use gb_model::families::minicpmv4_7::boot::{boot_profile, boot_with, BootExtras, BootPlan, LoadedModel};
use gb_model::families::minicpmv4_7::config::{DSparkConfig, MiniCpmConfig};
use gb_model::families::minicpmv4_7::engine::{engine_from, publish_vision, STEP_MS};
use gb_model::families::minicpmv4_7::memory::draft_page_bytes;
use gb_model::families::minicpmv4_7::model::{MiniCpmModel, ModelOptions, WINDOW_ROWS};
use gb_model::families::minicpmv4_7::prefix;
use gb_model::families::minicpmv4_7::vision::{
    self, image_preprocessing, video_preprocessing, MiniCpmEncoder, MiniCpmFeatures, MiniCpmTower,
};
use gb_model::serve::{feature_ref, FeatureBudget, ImageEncoder, PrefixCacheOption, ServeOptions, VisionService, VisionSlot};
use gb_sampling::Sampling;
use gb_sched::{CostModel, MemoryProfile, ModelMemory};
use gb_server::chat_format::{ChatFormat, VideoOptions};
use gb_server::formats::minicpmv::MiniCpmFormat;
use gb_server::formats::nemotron_omni::prep::data_uri;
use gb_server::tokenizer::Tok;
use gb_server::vision::{ImageRef, Limits};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

fn pack_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_MINICPM_DIR") {
        Ok(d) if Path::new(&d).join("config.json").exists() => {
            if !gb_cuda::device_available() {
                eprintln!("not run: no CUDA device");
                return None;
            }
            Some(PathBuf::from(d))
        }
        _ => {
            eprintln!("not run: REXL3_MINICPM_DIR is not set to a pack");
            None
        }
    }
}

fn mm_dir() -> Option<PathBuf> {
    let d = std::env::var("REXL3_MINICPM_MM_DIR").map(PathBuf::from).unwrap_or_else(|_| home().join("projects/minicpm-mm"));
    if d.join("cases/cases.json").exists() {
        Some(d)
    } else {
        eprintln!("not run: {} has no cases (tools/minicpm_v47/mm_cases.py)", d.display());
        None
    }
}

/// One engine run of a text prompt: the prompt, its greedy tokens, the head's logits of its last rows and the candidates of its
/// last row (ids and value bits).
type TextRun = (Vec<u32>, Vec<u32>, Vec<u16>, (Vec<u32>, Vec<u32>));
/// One continuation row of Leg C: KL(reference || engine), cosine, whether the argmaxes agree, the reference's top-1 margin.
type Stat = (f64, f64, bool, f64);

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn b64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(b.len() * 4 / 3 + 4);
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        s.push(T[(n >> 18) as usize & 63] as char);
        s.push(T[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    s
}

fn data_url(dir: &Path, name: &str) -> String {
    format!("data:image/png;base64,{}", b64(&std::fs::read(dir.join(name)).unwrap()))
}

fn greedy() -> Sampling {
    Sampling { seed: 0, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 }
}

fn request(prompt: Vec<u32>, images: Vec<ImageSpan>, max_new: usize) -> Request {
    Request { prompt, sampling: greedy(), max_new, stop: vec![], draft: false, images, cache: false }
}

// ------------------------------------------------------------------------------------------------------------------ the rig

#[derive(Clone)]
struct Shared(Rc<RefCell<MiniCpmBackend>>);

impl Backend for Shared {
    fn vocab_size(&self) -> usize {
        self.0.borrow().vocab_size()
    }
    fn take_error(&mut self) -> Option<gb_engine::BackendError> {
        self.0.borrow_mut().take_error()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.0.borrow_mut().prefill(slot, start, tokens, candidates)
    }
    fn prefill_piece(&mut self, slot: usize, start: u64, tokens: &[u32], total: usize, candidates: usize) -> RowOutput {
        self.0.borrow_mut().prefill_piece(slot, start, tokens, total, candidates)
    }
    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> gb_engine::PrefillStep {
        self.0.borrow_mut().prefill_slice(slot, prompt, images, done, candidates)
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        self.0.borrow_mut().step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.0.borrow_mut().commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        self.0.borrow_mut().release(slot)
    }
}

struct Rig {
    backend: Rc<RefCell<MiniCpmBackend>>,
    plan: BootPlan,
    profile: MemoryProfile,
    cfg: MiniCpmConfig,
    slots: usize,
    max_ctx: usize,
    dir: PathBuf,
    svc: Option<Arc<VisionService>>,
    format: MiniCpmFormat,
    tok: Tok,
    limits: Limits,
}

impl Rig {
    fn load(slots: usize, max_ctx: usize, prefill_rows: usize, tower: bool) -> Option<Rig> {
        let dir = pack_dir()?;
        let cfg = MiniCpmConfig::from_dir(&dir).unwrap();
        let mut o = ServeOptions::new(dir.clone());
        o.slots = slots;
        o.slots_auto = false;
        o.allow_unvalidated_pack = true;
        o.max_model_len = max_ctx;
        o.prefill_chunk = prefill_rows;
        if let Some(n) = std::env::var("REXL3_MINICPM_KV_TOKENS").ok().and_then(|v| v.parse::<usize>().ok()) {
            o.kv_tokens = Some(n);
        }
        // this rig builds the model from the booted state itself: no snapshot arena (the cache tests use `CacheRig`)
        o.prefix_cache = PrefixCacheOption::Off;
        let slot: VisionSlot = Default::default();
        if tower {
            o.vision = Some(slot.clone());
        }
        let t = std::time::Instant::now();
        let loaded =
            boot_with(&o, BootExtras { allow_truncated_layers: cfg.layers < 40 }).unwrap_or_else(|e| panic!("boot: {e}"));
        let (profile, _) = boot_profile(&loaded.cfg, &o).unwrap();
        let LoadedModel { cfg, weights, state, plan, report } = loaded;
        let opts = ModelOptions { slots: plan.slots, kv_pages: plan.kv_pages(), max_context: plan.max_context, prefill_rows };
        let model = MiniCpmModel::new(cfg.clone(), weights, state, opts).unwrap_or_else(|e| panic!("model: {e}"));
        if tower {
            publish_vision(&o, &slot, &profile, &plan, report.host_before.as_ref()).unwrap_or_else(|e| panic!("tower: {e}"));
        }
        eprintln!(
            "loaded {} layers, {} slots, {} pages, context {}, prefill rows {prefill_rows}, tower {tower} in {:.1} s",
            cfg.layers,
            plan.slots,
            plan.kv_pages(),
            plan.max_context,
            t.elapsed().as_secs_f64()
        );
        let backend = MiniCpmBackend::new(model, prefill_rows);
        let template = std::fs::read_to_string(dir.join("chat_template.jinja")).unwrap();
        let image = image_preprocessing(&dir).unwrap();
        let video = video_preprocessing(&image).unwrap();
        let mut format = MiniCpmFormat::new(&template).unwrap().with_preprocessing(image, video);
        format
            .enable_video(&VideoOptions { max_frames: 128, max_tokens: 16384, pruning_rate: 0.0, require_square_grid: false })
            .unwrap();
        Some(Rig {
            backend: Rc::new(RefCell::new(backend)),
            max_ctx: plan.max_context,
            slots: plan.slots,
            plan,
            profile,
            cfg,
            tok: Tok::from_dir(&dir).unwrap(),
            dir,
            svc: tower.then(|| slot.get().cloned().expect("the tower was published")),
            format,
            limits: Limits::default(),
        })
    }

    fn shared(&self) -> Shared {
        let mut b = self.backend.borrow_mut();
        for s in 0..self.slots {
            b.release(s);
        }
        let _ = b.take_error();
        drop(b);
        Shared(self.backend.clone())
    }

    fn engine(&self) -> Engine<Shared> {
        let mm = ModelMemory::from_profile(&self.profile, 0, 0);
        let admission = self.plan.memory.admission(mm, self.slots, self.max_ctx).unwrap();
        let make: Box<dyn Fn() -> Box<dyn Drafter>> = Box::new(|| Box::new(NoDrafts));
        Engine::new(
            self.shared(),
            admission,
            EngineConfig { max_rows: WINDOW_ROWS, max_drafts: 7, cost: CostModel::from_table(STEP_MS.to_vec()), drafting: false },
            make,
        )
    }

    fn with_model<R>(&self, f: impl FnOnce(&mut MiniCpmModel) -> R) -> R {
        f(&mut self.backend.borrow_mut().m)
    }

    /// The prompt of one case as the server builds it: ids and the spans (images and video frames go through the encoder).
    fn prepare(&self, c: &Value, dir: &Path) -> (Vec<u32>, Vec<ImageSpan>) {
        let mut content = Vec::new();
        match c["kind"].as_str().unwrap() {
            "image" => {
                for f in c["images"].as_array().unwrap() {
                    content.push(json!({"type": "image_url", "image_url": {"url": data_url(dir, f.as_str().unwrap())}}));
                }
            }
            "video" => {
                let frames: Vec<String> =
                    c["frames"].as_array().unwrap().iter().map(|f| data_url(dir, f.as_str().unwrap())).collect();
                content.push(json!({"type": "video_url", "video_url": {"url": frames}}));
            }
            _ => {}
        }
        content.push(json!({"type": "text", "text": c["question"]}));
        let messages = json!([{"role": "user", "content": content}]);
        self.prepare_messages(&messages)
    }

    fn prepare_messages(&self, messages: &Value) -> (Vec<u32>, Vec<ImageSpan>) {
        ingest_messages(&self.format, &self.tok, self.limits, self.svc.as_ref(), messages)
    }

    fn generate(&self, ids: &[u32], spans: &[ImageSpan], n: usize) -> Vec<u32> {
        let mut e = self.engine();
        e.submit(request(ids.to_vec(), spans.to_vec(), n)).unwrap();
        e.run().unwrap();
        let out = e.take_outputs().remove(0);
        assert!(self.backend.borrow_mut().take_error().is_none());
        out.tokens
    }
}

/// The ids and spans of a chat the way the server builds them: media through the encoder (`svc`), text through the template.
fn ingest_messages(
    format: &MiniCpmFormat,
    tok: &Tok,
    limits: Limits,
    svc: Option<&Arc<VisionService>>,
    messages: &Value,
) -> (Vec<u32>, Vec<ImageSpan>) {
    let kwargs = serde_json::Map::new();
    if let Some(svc) = svc {
        let has_media = messages.to_string().contains("image_url") || messages.to_string().contains("video_url");
        if has_media {
            let vf = format.video().expect("video is on");
            let mut fetch = |im: &ImageRef| data_uri(im, &limits);
            let ing = vf.ingest_frames(tok, messages, &kwargs, &limits, 8 << 20, svc, None, &mut fetch).unwrap();
            return (ing.ids, ing.images);
        }
    }
    let r = format.render_bounded(messages, &kwargs, 8 << 20).unwrap();
    (tok.encode_bounded(&r.text, 8 << 20).unwrap(), Vec::new())
}

fn cases_of(dir: &Path) -> Vec<Value> {
    let text = std::fs::read_to_string(dir.join("cases/cases.json")).unwrap();
    serde_json::from_str::<Value>(&text).unwrap()["cases"].as_array().unwrap().clone()
}

/// The cases a Leg C run covers: all of them, or the comma-separated names of `REXL3_MINICPM_MM_CASES`.
fn selected_cases(dir: &Path) -> Vec<Value> {
    let all = cases_of(dir);
    match std::env::var("REXL3_MINICPM_MM_CASES") {
        Ok(list) if !list.trim().is_empty() => {
            let want: Vec<&str> = list.split(',').map(str::trim).collect();
            all.into_iter().filter(|c| want.contains(&c["name"].as_str().unwrap())).collect()
        }
        _ => all,
    }
}

fn case_named(dir: &Path, name: &str) -> Value {
    cases_of(dir).into_iter().find(|c| c["name"] == name).unwrap_or_else(|| panic!("case {name}"))
}

fn rows_bytes(f: &MiniCpmFeatures) -> Vec<u16> {
    let mut v = vec![0u16; f.tokens * 2048];
    f.buf.download_range(0, &mut v).unwrap();
    v
}

// --------------------------------------------------------------------------------------------------------------- invariants

/// The text path with the tower in the process: the same prompts give the same candidates, the same logits and the same greedy
/// tokens as a boot without a tower, bit for bit, while the tower is encoding on its own stream.
#[test]
#[ignore]
fn text_is_bit_identical_with_the_tower_loaded() {
    let Some(mm) = mm_dir() else { return };
    // the same three text prompts of the three prefill arms (GEMV up to 128 tokens, the pinned GEMM above)
    let run = |tower: bool| -> Option<Vec<TextRun>> {
        let rig = Rig::load(2, 4096, 1024, tower)?;
        let v = rig.cfg.vocab;
        let prompts: Vec<Vec<u32>> = [60usize, 700, 2100]
            .iter()
            .enumerate()
            .map(|(k, &n)| {
                (0..n).map(|i| ((i as u64 * 2_654_435_761 + k as u64 * 40_503) % (v as u64 - 16) + 8) as u32).collect()
            })
            .collect();
        // with the tower loaded, keep it busy on its own stream through every text run
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker = rig.svc.clone().map(|svc| {
            let stop = stop.clone();
            let case = case_named(&mm, "big");
            let url = data_url(&mm.join("cases"), case["images"][0].as_str().unwrap());
            let pp = image_preprocessing(&rig.dir).unwrap();
            std::thread::spawn(move || {
                let r = ImageRef { message: 0, part: 0, url };
                let limits = Limits::default();
                let bytes = data_uri(&r, &limits).unwrap();
                let mut n = 0;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    // a different image each time, so that every encode is a tower launch and not a cache hit
                    let mut p = gb_server::vision::prepare_bytes(&r, &bytes, &limits, &pp, None).unwrap();
                    p.pixel_hash ^= n;
                    let _ = svc.encode(&p);
                    n += 1;
                }
                n
            })
        });
        let mut out = Vec::new();
        if rig.svc.is_some() {
            // an image request first, in the slot the text prompts use: nothing of it (its pages, its state, its rope delta) may show in
            // a later text request
            let (ids, spans) = rig.prepare(&case_named(&mm, "doc"), &mm.join("cases"));
            let _ = rig.generate(&ids, &spans, 4);
        }
        for p in &prompts {
            // the text path itself, first (what an engine runs for a text prompt): the top-16 candidates of the last row, bit for bit
            let cand = {
                let mut b = rig.backend.borrow_mut();
                let mut done = 0;
                let row = loop {
                    let step = b.prefill_slice(0, p, &[], done, 16);
                    done = step.done;
                    if let Some(r) = step.row {
                        break r;
                    }
                };
                b.release(0);
                (row.ids.clone(), row.values.iter().map(|x| x.to_bits()).collect::<Vec<u32>>())
            };
            let tokens = rig.generate(p, &[], 12);
            let mut logits = Vec::new();
            rig.backend
                .borrow_mut()
                .prefill_logits(0, p, &[], p.len() - 4, p.len(), |_, _, l| logits.extend_from_slice(l))
                .unwrap();
            out.push((p.clone(), tokens, logits, cand));
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(w) = worker {
            eprintln!("the tower made {} encodes beside the text runs", w.join().unwrap());
        }
        Some(out)
    };
    let Some(without) = run(false) else { return };
    let Some(with) = run(true) else { return };
    for (k, (a, b)) in without.iter().zip(&with).enumerate() {
        assert_eq!(a.3, b.3, "prompt {k}: the candidates of the prompt's last row differ with the tower loaded");
        assert_eq!(a.1, b.1, "prompt {k}: greedy tokens differ with the tower loaded");
        assert_eq!(a.2, b.2, "prompt {k}: the head's logits of the last rows differ with the tower loaded");
    }
    eprintln!("text bit-identical with the tower loaded: {} prompts, tokens and last-row logits equal", with.len());
}

/// An image stream, a video stream and text streams in one engine: each equals its solo run (the batched window and the slices
/// interleaved with the image prefill change no bit of any stream).
#[test]
#[ignore]
fn an_image_slot_beside_text_slots_changes_no_stream() {
    let Some(mm) = mm_dir() else { return };
    let Some(rig) = Rig::load(4, 4096, 512, true) else { return };
    let v = rig.cfg.vocab;
    let img = rig.prepare(&case_named(&mm, "photo"), &mm.join("cases"));
    let vid = rig.prepare(&case_named(&mm, "video"), &mm.join("cases"));
    let text = |n: usize, seed: u64| -> Vec<u32> {
        (0..n).map(|i| ((i as u64 * 2_654_435_761 + seed * 40_503) % (v as u64 - 16) + 8) as u32).collect()
    };
    let reqs = vec![
        request(img.0.clone(), img.1.clone(), 20),
        request(text(150, 1), vec![], 24),
        request(vid.0.clone(), vid.1.clone(), 16),
        request(text(40, 2), vec![], 28),
    ];
    let make_engine = |_drafting: bool| rig.engine();
    let r = serial_batched_equals_solo(&make_engine, &reqs).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "image + video + 2 text streams: {} sequences, {} tokens, {} rounds, each equal to solo",
        r.sequences, r.tokens, r.rounds
    );
    assert!(rig.backend.borrow_mut().take_error().is_none());
}

/// An image prompt prefilled on any chunk grid (and a video prompt: one visual group of several frames) ends in the same candidates,
/// bit for bit: the triples come from the whole prompt, so where the cut falls changes nothing.
#[test]
#[ignore]
fn an_image_prompt_in_any_chunk_grid_ends_in_the_same_candidates() {
    let Some(mm) = mm_dir() else { return };
    let Some(rig) = Rig::load(1, 4096, 1024, true) else { return };
    for name in ["big", "two", "video", "small"] {
        let (ids, spans) = rig.prepare(&case_named(&mm, name), &mm.join("cases"));
        let mut first: Option<(Vec<u32>, Vec<u32>)> = None;
        for chunk in [37usize, 64, 100, 256, 512, 1024] {
            rig.shared();
            let mut b = rig.backend.borrow_mut();
            b.chunk = chunk;
            let mut done = 0;
            let row = loop {
                let step = b.prefill_slice(0, &ids, &spans, done, 16);
                assert!(step.done > done);
                done = step.done;
                if let Some(r) = step.row {
                    break r;
                }
            };
            b.release(0);
            assert!(b.take_error().is_none());
            let got = (row.ids.clone(), row.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            match &first {
                None => first = Some(got),
                Some(f) => assert_eq!(&got, f, "{name}: chunk {chunk} ends in other candidates"),
            }
        }
        eprintln!("{name}: {} tokens, six chunk grids, one set of candidates", ids.len());
    }
}

/// A request whose image cannot be spliced fails alone: its neighbours finish with their solo tokens.
#[test]
#[ignore]
fn a_bad_image_request_fails_alone_through_the_engine() {
    let Some(mm) = mm_dir() else { return };
    let Some(rig) = Rig::load(3, 4096, 512, true) else { return };
    let v = rig.cfg.vocab;
    let (ids, spans) = rig.prepare(&case_named(&mm, "doc"), &mm.join("cases"));
    let good_text: Vec<u32> = (0..90).map(|i| ((i as u64 * 2_654_435_761) % (v as u64 - 16) + 8) as u32).collect();
    let want_text = rig.generate(&good_text, &[], 10);
    let want_image = rig.generate(&ids, &spans, 10);
    // the same prompt with the span cut short (the engine's own geometry check passes: `merged` follows): its pads no longer
    // number its feature rows, which only the backend knows
    let mut bad = spans.clone();
    bad[0].tokens = 100;
    bad[0].merged = (1, 100);
    let mut e = rig.engine();
    let a = e.submit_deferred(request(good_text.clone(), vec![], 10)).unwrap();
    let b = e.submit_deferred(request(ids.clone(), bad, 10)).unwrap();
    let c = e.submit_deferred(request(ids.clone(), spans.clone(), 10)).unwrap();
    e.run().unwrap();
    let failed = e.take_failed();
    assert_eq!(failed.len(), 1, "{failed:?}");
    assert_eq!(failed[0].0, b, "the bad image request is the one that fails");
    assert!(failed[0].1.contains("pad tokens") || failed[0].1.contains("MiniCPM"), "{}", failed[0].1);
    let out = e.take_outputs();
    let tokens = |id: u64| out.iter().find(|o| o.id == id).map(|o| o.tokens.clone()).unwrap_or_default();
    assert_eq!(tokens(a), want_text, "the text stream beside a failed image request changed");
    assert_eq!(tokens(c), want_image, "the good image stream beside a failed image request changed");
    // the backend is clean afterwards
    let again = rig.generate(&ids, &spans, 10);
    assert_eq!(again, want_image);
    eprintln!("a bad image request failed alone ({}), neighbours bit-equal", failed[0].1);
}

/// The encoder's cache: a hit is the miss's rows byte for byte (and the rows of an encoder that keeps no cache), a change of the
/// downsample mode is a different key and never a stale hit.
#[test]
#[ignore]
fn the_feature_cache_hit_is_the_miss_byte_for_byte_and_the_mode_is_in_the_key() {
    let Some(mm) = mm_dir() else { return };
    let Some(dir) = pack_dir() else { return };
    let case = case_named(&mm, "photo");
    let url = data_url(&mm.join("cases"), case["images"][0].as_str().unwrap());
    let r = ImageRef { message: 0, part: 0, url };
    let limits = Limits::default();
    let bytes = data_uri(&r, &limits).unwrap();
    let pre16 = image_preprocessing(&dir).unwrap();
    let pre4 = pre16.clone().with_downsample(gb_vision::minicpmv::Downsample::X4);
    let p16 = gb_server::vision::prepare_bytes(&r, &bytes, &limits, &pre16, None).unwrap();
    let make = |cache: u64| {
        let tower = MiniCpmTower::load(&dir, vision::DEFAULT_MAX_PATCHES).unwrap();
        MiniCpmEncoder::new(tower, pre16.clone(), FeatureBudget::new(256 << 20)).with_cache_bytes(cache)
    };
    let mut cached = make(64 << 20);
    let first = cached.encode(&p16).unwrap();
    let second = cached.encode(&p16).unwrap();
    assert_eq!((cached.cache().hits, cached.cache().misses), (1, 1));
    assert!(Arc::ptr_eq(&first.features, &second.features), "a hit shares the miss's rows");
    let a = rows_bytes(feature_ref::<MiniCpmFeatures>(&*first.features).unwrap());
    // a second tower with no cache encodes the same pixels from scratch: the hit's bytes are the fresh bytes
    let mut fresh = make(0);
    let f = fresh.encode(&p16).unwrap();
    assert_eq!(fresh.cache().len(), 0);
    let b = rows_bytes(feature_ref::<MiniCpmFeatures>(&*f.features).unwrap());
    assert_eq!(a.len(), b.len());
    assert!(a == b, "the cached rows differ from a fresh encode");
    // another downsample mode: the same pixel bytes and a different key; the rows are 4x as many, not the cached ones
    cached.set_preprocessing(pre4.clone());
    let p4 = gb_server::vision::prepare_bytes(&r, &bytes, &limits, &pre4, None).unwrap();
    let x4 = cached.encode(&p4).unwrap();
    assert_eq!(cached.cache().hits, 1, "a mode change must not hit");
    assert_eq!(x4.tokens, first.tokens * 4);
    // and back: the 16x entry is still there
    cached.set_preprocessing(pre16.clone());
    let again = cached.encode(&p16).unwrap();
    assert_eq!(cached.cache().hits, 2);
    assert!(Arc::ptr_eq(&first.features, &again.features));
    // different pixels (the hash differs), same size: a miss
    let mut other = p16.clone();
    other.pixel_hash ^= 1;
    let o = cached.encode(&other).unwrap();
    assert_eq!(cached.cache().hits, 2);
    assert!(!Arc::ptr_eq(&first.features, &o.features));
    eprintln!("feature cache: hit == miss == fresh encode ({} bytes), 4x and other pixels miss", a.len() * 2);
}

/// The tower's allocations against the memory model's figures at three launch sizes.
#[test]
#[ignore]
fn the_towers_allocations_are_what_the_memory_model_prices() {
    let Some(dir) = pack_dir() else { return };
    for max_patches in [1024usize, 4096, vision::DEFAULT_MAX_PATCHES] {
        let t = MiniCpmTower::load(&dir, max_patches).unwrap();
        assert_eq!(t.bytes(), (vision::weight_bytes(), vision::workspace_bytes(max_patches)), "max_patches {max_patches}");
        assert_eq!(t.convert_report().overflow, 0, "no weight overflows fp16");
    }
}

// ------------------------------------------------------------------------------------------------------------ Leg C oracle

fn out_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_MINICPM_MM_OUT") {
        Ok(d) => {
            std::fs::create_dir_all(&d).unwrap();
            Some(PathBuf::from(d))
        }
        Err(_) => {
            eprintln!("not run: REXL3_MINICPM_MM_OUT is not set (where cont.json and the table go)");
            None
        }
    }
}

/// The engine's greedy continuation of every case (and its prompt ids, compared with the processor's when its output is there).
#[test]
#[ignore]
fn leg_c_continuations() {
    let (Some(mm), Some(out)) = (mm_dir(), out_dir()) else { return };
    let n = env_usize("REXL3_MINICPM_MM_TOKENS", 64);
    let Some(rig) = Rig::load(2, 8192, env_usize("REXL3_MINICPM_PREFILL_ROWS", 4096), true) else { return };
    let hf: Option<Value> =
        std::fs::read_to_string(mm.join("proc/hf_prompts.json")).ok().and_then(|t| serde_json::from_str(&t).ok());
    let (mut cont, mut prompts) = (serde_json::Map::new(), serde_json::Map::new());
    for c in selected_cases(&mm) {
        let name = c["name"].as_str().unwrap().to_string();
        let t0 = std::time::Instant::now();
        let (ids, spans) = rig.prepare(&c, &mm.join("cases"));
        let t_prep = t0.elapsed().as_secs_f64();
        if let Some(h) = &hf {
            let want: Vec<u64> = h[&name]["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            let got: Vec<u64> = ids.iter().map(|&t| t as u64).collect();
            assert_eq!(got, want, "{name}: the engine's prompt is not the processor's");
        }
        let t1 = std::time::Instant::now();
        let tokens = rig.generate(&ids, &spans, n);
        eprintln!(
            "{name}: prompt {} tokens, {} spans, prepare {:.2}s, {n} tokens in {:.2}s: {:?}",
            ids.len(),
            spans.len(),
            t_prep,
            t1.elapsed().as_secs_f64(),
            rig.tok.decode(&tokens, false).chars().take(120).collect::<String>()
        );
        cont.insert(name.clone(), json!(tokens));
        prompts.insert(name, json!(ids));
    }
    std::fs::write(out.join("cont.json"), Value::Object(cont).to_string()).unwrap();
    std::fs::write(out.join("engine_prompts.json"), Value::Object(prompts).to_string()).unwrap();
}

fn read_npy_f32(path: &Path) -> (usize, usize, Vec<f32>) {
    let raw = std::fs::read(path).unwrap();
    assert_eq!(&raw[..6], b"\x93NUMPY");
    let hl = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    let header = std::str::from_utf8(&raw[10..10 + hl]).unwrap();
    assert!(header.contains("'<f4'") && !header.contains("True"), "{header}");
    let shape = header.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
    let dims: Vec<usize> = shape.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse().unwrap()).collect();
    let data = &raw[10 + hl..];
    let v: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(v.len(), dims[0] * dims[1]);
    (dims[0], dims[1], v)
}

fn f16(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 if frac == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        e => sign * (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    }
}

/// One position: KL(reference || engine) over the vocabulary, the cosine of the raw logits, both argmaxes and the reference's top-1 margin.
fn compare(engine: &[u16], reference: &[f32], head_f16: bool) -> (f64, f64, usize, usize, f64) {
    let n = reference.len();
    let e: Vec<f64> =
        engine[..n].iter().map(|&b| if head_f16 { f16(b) as f64 } else { f32::from_bits((b as u32) << 16) as f64 }).collect();
    let r: Vec<f64> = reference.iter().map(|&x| x as f64).collect();
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    let (mut emax, mut rmax) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let (mut ae, mut ar) = (0usize, 0usize);
    let (mut r1, mut r2) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for i in 0..n {
        ab += e[i] * r[i];
        aa += e[i] * e[i];
        bb += r[i] * r[i];
        if e[i] > emax {
            emax = e[i];
            ae = i;
        }
        if r[i] > rmax {
            rmax = r[i];
            ar = i;
        }
        if r[i] > r1 {
            r2 = r1;
            r1 = r[i];
        } else if r[i] > r2 {
            r2 = r[i];
        }
    }
    let (mut ze, mut zr) = (0f64, 0f64);
    for i in 0..n {
        ze += (e[i] - emax).exp();
        zr += (r[i] - rmax).exp();
    }
    let (lze, lzr) = (ze.ln() + emax, zr.ln() + rmax);
    let mut kl = 0f64;
    for i in 0..n {
        let lp = r[i] - lzr;
        kl += lp.exp() * (lp - (e[i] - lze));
    }
    (kl, ab / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE), ae, ar, r1 - r2)
}

/// The engine teacher-forced on prompt + its own continuation, against the reference's logits of the same sequence on the
/// continuation rows: KL(reference || engine), top-1 agreement, cosine, per case and the first 8 rows apart.
#[test]
#[ignore]
fn leg_c_kl() {
    let (Some(mm), Some(out)) = (mm_dir(), out_dir()) else { return };
    let ref_dir = std::env::var("REXL3_MINICPM_MM_REF").map(PathBuf::from).unwrap_or_else(|_| mm.join("ref"));
    let cont: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("cont.json")).expect("cont.json (leg_c_continuations)")).unwrap();
    let Some(rig) = Rig::load(1, 8192, env_usize("REXL3_MINICPM_PREFILL_ROWS", 4096), true) else { return };
    let head_f16 = rig.with_model(|m| m.logits_are_f16());
    let v = rig.cfg.vocab;
    let mut table = Vec::new();
    let mut stale: Vec<String> = Vec::new();
    for c in selected_cases(&mm) {
        let name = c["name"].as_str().unwrap().to_string();
        let npy = ref_dir.join(format!("logits_{name}.npy"));
        if !npy.exists() {
            eprintln!("{name}: no reference logits ({})", npy.display());
            continue;
        }
        let (rows, width, reference) = read_npy_f32(&npy);
        assert_eq!(width, v, "{name}: reference width");
        let (mut ids, spans) = rig.prepare(&c, &mm.join("cases"));
        let n = ids.len();
        let tokens: Vec<u32> = cont[&name].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
        // the reference was teacher-forced on ONE continuation: if the engine's has changed since (another arithmetic, another pack),
        // the logits below belong to other tokens and the numbers would mean nothing
        let meta: Value =
            serde_json::from_str(&std::fs::read_to_string(ref_dir.join(format!("meta_{name}.json"))).unwrap()).unwrap();
        let ref_tail: Vec<u32> = meta["ids"].as_array().unwrap()[n..].iter().map(|t| t.as_u64().unwrap() as u32).collect();
        if ref_tail != tokens {
            eprintln!("LEGC-STALE {name}: the reference was run on another continuation than cont.json holds (rerun mm_ref_logits.py ref)");
            stale.push(name);
            continue;
        }
        assert_eq!(rows, tokens.len(), "{name}: the reference holds {rows} rows for {} continuation tokens", tokens.len());
        ids.extend(&tokens);
        // the row n-1 predicts the first continuation token; the last continuation token's own row is not compared
        let (lo, hi) = (n - 1, n - 1 + rows);
        let mut stats: Vec<Stat> = Vec::new();
        let mut spans_full = spans.clone();
        // spans keep their place; the continuation is plain text after them
        for s in &mut spans_full {
            s.merged = (1, s.tokens);
        }
        rig.backend
            .borrow_mut()
            .prefill_logits(0, &ids, &spans_full, lo, hi, |first, k, logits| {
                for j in 0..k {
                    let r = first + j - lo;
                    let (kl, cos, ae, ar, margin) =
                        compare(&logits[j * v..(j + 1) * v], &reference[r * v..(r + 1) * v], head_f16);
                    stats.push((kl, cos, ae == ar, margin));
                }
            })
            .unwrap();
        assert_eq!(stats.len(), rows, "{name}");
        let mean = |f: &dyn Fn(&Stat) -> f64, s: &[Stat]| s.iter().map(f).sum::<f64>() / s.len() as f64;
        let first8 = &stats[..stats.len().min(8)];
        let decided: Vec<_> = stats.iter().filter(|s| s.3 >= 0.25).collect();
        let row = json!({
            "case": name, "prompt_tokens": n, "rows": rows, "spans": spans.len(),
            "kl_mean": mean(&|s| s.0, &stats), "kl_first8": mean(&|s| s.0, first8),
            "kl_max": stats.iter().map(|s| s.0).fold(0.0, f64::max),
            "top1": mean(&|s| s.2 as u8 as f64, &stats), "top1_first8": mean(&|s| s.2 as u8 as f64, first8),
            "top1_decided": decided.iter().filter(|s| s.2).count() as f64 / decided.len().max(1) as f64,
            "decided_rows": decided.len(),
            "cos_mean": mean(&|s| s.1, &stats),
        });
        eprintln!("LEGC {row}");
        table.push(row);
    }
    std::fs::write(out.join("leg_c.json"), Value::Array(table).to_string()).unwrap();
    assert!(stale.is_empty(), "references of other continuations: {stale:?}");
}

// ----------------------------------------------------------------------------------------------------------------- latency

/// TTFT-shaped numbers of one image at 16x with 4 and with 10 crops, uncached and cached (the encoder's feature cache): the CPU's
/// decode and preprocessing, the tower (a miss and a hit), the engine's prefill and first token. Alone on the box (exclusive lock).
#[test]
#[ignore]
fn bench_image_latency() {
    let Some(mm) = mm_dir() else { return };
    let Some(rig) = Rig::load(1, 8192, 4096, true) else { return };
    let svc = rig.svc.clone().unwrap();
    let pp = image_preprocessing(&rig.dir).unwrap();
    let limits = Limits::default();
    for (name, file) in [("4 crops", "bench_4crops.png"), ("10 crops", "bench_10crops.png")] {
        let url = data_url(&mm.join("cases"), file);
        let r = ImageRef { message: 0, part: 0, url: url.clone() };
        let bytes = data_uri(&r, &limits).unwrap();
        let messages = json!([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "Describe this image."}]}]);
        // the stages alone: the CPU's decode + preprocessing, the tower on a key it has never seen (a miss: the pixel hash is varied),
        // the tower on a key it has (a hit)
        for pass in 0..3u64 {
            let t0 = std::time::Instant::now();
            let mut pre = gb_server::vision::prepare_bytes(&r, &bytes, &limits, &pp, None).unwrap();
            let t_cpu = t0.elapsed().as_secs_f64() * 1e3;
            pre.pixel_hash ^= 0x5eed_0000 + pass;
            let t1 = std::time::Instant::now();
            let enc = svc.encode(&pre).unwrap();
            let t_miss = t1.elapsed().as_secs_f64() * 1e3;
            let t2 = std::time::Instant::now();
            let _ = svc.encode(&pre).unwrap();
            let t_hit = t2.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "LATENCY {name} stages {pass}: decode+preprocess {t_cpu:.1} ms, tower miss {t_miss:.1} ms ({} tokens), tower hit {t_hit:.2} ms",
                enc.tokens
            );
        }
        // the request: the whole ingest (decode, preprocess, tower or its cache) then the engine's prefill and first token. Request 0
        // is the first time the service sees this image (a cold key); the next ones hit
        let mut e = rig.engine();
        for pass in 0..3 {
            let t2 = std::time::Instant::now();
            let (ids, spans) = rig.prepare_messages(&messages);
            let t_ingest = t2.elapsed().as_secs_f64() * 1e3;
            let t3 = std::time::Instant::now();
            e.submit_deferred(request(ids.clone(), spans.clone(), 1)).unwrap();
            e.run().unwrap();
            let t_gen = t3.elapsed().as_secs_f64() * 1e3;
            let _ = e.take_outputs();
            eprintln!(
                "LATENCY {name} request {pass}: ingest {t_ingest:.1} ms, engine prefill {} tokens + first token {t_gen:.0} ms, TTFT {:.0} ms",
                ids.len(),
                t_ingest + t_gen
            );
        }
    }
}

// ------------------------------------------------------------------------------------------------ images under the prefix cache

/// The engine as `serve` builds it with a prefix cache of `units` entry units (a unit is one snapshot slot and the planner's 8
/// pages) and the tower: images and video go through the encoder, their prompts through the cache.
struct CacheRig {
    eng: gb_engine::Engine<MiniCpmBackend>,
    plan: BootPlan,
    kv_pages: usize,
    svc: Arc<VisionService>,
    format: MiniCpmFormat,
    tok: Tok,
    limits: Limits,
    cases: PathBuf,
}

impl CacheRig {
    fn load(slots: usize, max_ctx: usize, chunk: usize, units: u64) -> Option<CacheRig> {
        // the policy the engine tests use: cuts from 128 tokens (the shipped minimum is 512), so that short prompts exercise it
        let policy = PrefixConfig {
            policy: CutPolicy { min_cut: 128, tail: 16, ghost_min_gain: 512, min_gap: 64 },
            ghost_blocks: 1 << 12,
            ..Default::default()
        };
        Self::load_with(slots, max_ctx, chunk, units, Some(policy))
    }

    /// `policy` replaces the cut policy `engine_from` configured (the shipped one); `None` keeps it.
    fn load_with(slots: usize, max_ctx: usize, chunk: usize, units: u64, policy: Option<PrefixConfig>) -> Option<CacheRig> {
        Self::load_full(slots, max_ctx, chunk, units, policy, None)
    }

    /// [`CacheRig::load_with`] with the DSpark draft at `draft` attached (E4b): the pool's page then holds the draft's rows too, so an
    /// entry unit is a snapshot slot and 8 pages of the target's and the draft's.
    fn load_full(
        slots: usize,
        max_ctx: usize,
        chunk: usize,
        units: u64,
        policy: Option<PrefixConfig>,
        draft: Option<PathBuf>,
    ) -> Option<CacheRig> {
        let dir = pack_dir()?;
        let cases = mm_dir()?.join("cases");
        let cfg = MiniCpmConfig::from_dir(&dir).unwrap();
        let draft_page = match &draft {
            Some(d) => draft_page_bytes(&DSparkConfig::from_dir(d).unwrap_or_else(|e| panic!("{}: {e}", d.display()))),
            None => 0,
        };
        let unit = prefix::snapshot_bytes(&cfg) + 8 * (prefix::page_bytes(&cfg) + draft_page);
        let mut o = ServeOptions::new(dir.clone());
        o.draft_dir = draft;
        o.slots = slots;
        o.slots_auto = false;
        o.allow_unvalidated_pack = true;
        o.max_model_len = max_ctx;
        o.prefill_chunk = chunk;
        o.prefix_cache = PrefixCacheOption::Mib((units * unit).div_ceil(1 << 20));
        let slot: VisionSlot = Default::default();
        o.vision = Some(slot.clone());
        let loaded =
            boot_with(&o, BootExtras { allow_truncated_layers: cfg.layers < 40 }).unwrap_or_else(|e| panic!("boot: {e}"));
        let (plan, report) = (loaded.plan.clone(), loaded.report.clone());
        let eng = engine_from(loaded, &o).unwrap_or_else(|e| panic!("engine: {e}"));
        assert!(eng.prefix_stats().enabled, "the engine from `engine_from` runs the cache the plan carved out");
        assert_eq!(plan.memory.cache.snapshot_slots as u64, units);
        // the production path turned the cache on by itself; a test policy replaces the configuration, not the decision
        let eng = match policy {
            Some(p) => eng.with_prefix_cache(p),
            None => eng,
        };
        let template = std::fs::read_to_string(dir.join("chat_template.jinja")).unwrap();
        let image = image_preprocessing(&dir).unwrap();
        let video = video_preprocessing(&image).unwrap();
        let mut format = MiniCpmFormat::new(&template).unwrap().with_preprocessing(image, video);
        format
            .enable_video(&VideoOptions { max_frames: 128, max_tokens: 16384, pruning_rate: 0.0, require_square_grid: false })
            .unwrap();
        Some(CacheRig {
            eng,
            plan,
            kv_pages: report.kv_pages,
            svc: slot.get().cloned().expect("the tower was published"),
            format,
            tok: Tok::from_dir(&dir).unwrap(),
            limits: Limits::default(),
            cases,
        })
    }

    /// A chat with an optional system message, the pictures of `files` (or one video of them as frames), then `question`.
    fn chat(&self, system: Option<&str>, files: &[&str], video: bool, question: &str) -> Value {
        let url = |f: &str| data_url(&self.cases, f);
        let mut content = Vec::new();
        if video {
            let frames: Vec<String> = files.iter().map(|f| url(f)).collect();
            content.push(json!({"type": "video_url", "video_url": {"url": frames}}));
        } else {
            for f in files {
                content.push(json!({"type": "image_url", "image_url": {"url": url(f)}}));
            }
        }
        content.push(json!({"type": "text", "text": question}));
        let mut messages = Vec::new();
        if let Some(sys) = system {
            messages.push(json!({"role": "system", "content": sys}));
        }
        messages.push(json!({"role": "user", "content": content}));
        Value::Array(messages)
    }

    fn prepare(&self, messages: &Value) -> (Vec<u32>, Vec<ImageSpan>) {
        ingest_messages(&self.format, &self.tok, self.limits, Some(&self.svc), messages)
    }

    /// One request alone through the engine, to the end.
    fn one(&mut self, ids: &[u32], spans: &[ImageSpan], sampling: Sampling, n: usize, cache: bool) -> Output {
        self.one_with(ids, spans, sampling, n, cache, false)
    }

    /// Several requests at once through the engine (one slot each), to the end: `(ids, spans, sampling, cache, draft)`; the outputs in
    /// request order.
    #[allow(clippy::type_complexity)]
    fn burst(&mut self, reqs: &[(&[u32], &[ImageSpan], Sampling, bool, bool)], n: usize) -> Vec<Output> {
        let ids: Vec<u64> = reqs
            .iter()
            .map(|(p, s, sampling, cache, draft)| {
                let r = Request {
                    prompt: p.to_vec(),
                    sampling: *sampling,
                    max_new: n,
                    stop: vec![],
                    draft: *draft,
                    images: s.to_vec(),
                    cache: *cache,
                };
                self.eng.submit(r).unwrap()
            })
            .collect();
        self.eng.run().unwrap();
        let failed = self.eng.take_failed();
        assert!(failed.is_empty(), "no request may fail: {failed:?}");
        let mut outs = self.eng.take_outputs();
        ids.iter()
            .map(|id| {
                let at = outs.iter().position(|o| o.id == *id).expect("every request finished");
                outs.remove(at)
            })
            .collect()
    }

    /// [`CacheRig::one`] with the request's own choice of the draft (`draft`: the engine may draft it; the rig needs a draft attached).
    fn one_with(&mut self, ids: &[u32], spans: &[ImageSpan], sampling: Sampling, n: usize, cache: bool, draft: bool) -> Output {
        let r = Request { prompt: ids.to_vec(), sampling, max_new: n, stop: vec![], draft, images: spans.to_vec(), cache };
        let id = self.eng.submit(r).unwrap();
        self.eng.run().unwrap();
        let failed = self.eng.take_failed();
        assert!(failed.is_empty(), "no request may fail: {failed:?}");
        let mut outs = self.eng.take_outputs();
        outs.retain(|o| o.id == id);
        assert_eq!(outs.len(), 1);
        outs.remove(0)
    }
}

fn sampled(seed: u64) -> Sampling {
    Sampling { seed, temperature: 0.8, top_k: 20, top_p: 0.95, min_p: 0.0 }
}

fn same_reply(what: &str, got: &Output, want: &Output) {
    let at = got.tokens.iter().zip(&want.tokens).position(|(x, y)| x != y);
    assert!(
        got.tokens == want.tokens && (got.finish, got.stopped) == (want.finish, want.stopped),
        "{what}: the replies differ (first difference at {at:?}; {} against {} tokens)",
        got.tokens.len(),
        want.tokens.len()
    );
}

/// A question long enough that the last 16 tokens of the prompt (where the engine's end cut falls) are text after the picture.
const LONG_QUESTION: &str =
    "Please look at the whole picture carefully before you answer. Describe in plain words everything that you can \
see or read in it, mention the colours and the layout, and say which part of it you think is the most important one.";

/// Through the real engine: an image or video prompt served from the prefix cache equals its uncached reply token for token, greedy
/// and sampled, whatever the turn that continues it; a prompt of the same ids and another picture never hits; a cut that would fall
/// inside a picture is never taken; and a long text prefix shared by prompts of different pictures is cached and hit.
#[test]
#[ignore]
fn an_image_prompt_served_from_the_prefix_cache_equals_the_uncached_one_greedy_and_sampled() {
    let Some(mut rig) = CacheRig::load(2, 4096, 512, 24) else { return };
    let follow = rig.tok.encode_bounded(" Then answer again in one short sentence.", 1 << 20).unwrap();
    let (n_new, sampling) = (24usize, sampled(11));
    let hits = |r: &CacheRig| r.eng.prefix_stats().hits;

    // 1. a picture, two pictures and a clip, each: the uncached replies, the prompt's own entry, its continuation by a second turn
    let scenarios: [(&str, Vec<&str>, bool); 3] = [
        ("one picture", vec!["doc_0.png"], false),
        ("two pictures", vec!["two_0.png", "two_1.png"], false),
        (
            "a clip of six frames",
            vec!["video_0.png", "video_1.png", "video_2.png", "video_3.png", "video_4.png", "video_5.png"],
            true,
        ),
    ];
    for (name, files, video) in &scenarios {
        let (a_ids, a_spans) = rig.prepare(&rig.chat(None, files, *video, LONG_QUESTION));
        let last_end = a_spans.iter().map(|s| s.start + s.tokens).max().unwrap();
        assert!(a_ids.len() - 16 >= last_end, "{name}: the end cut is after the last picture");
        assert!(a_spans.iter().all(|s| s.key != 0), "{name}: every picture has a cache identity");
        let b_ids: Vec<u32> = a_ids.iter().chain(&follow).copied().collect();
        let end_cut_a = a_ids.len() - 16;
        let (ref_a_g, ref_a_s) =
            (rig.one(&a_ids, &a_spans, greedy(), n_new, false), rig.one(&a_ids, &a_spans, sampling, n_new, false));
        let (ref_b_g, ref_b_s) =
            (rig.one(&b_ids, &a_spans, greedy(), n_new, false), rig.one(&b_ids, &a_spans, sampling, n_new, false));
        assert_eq!(ref_a_g.cached_tokens + ref_b_g.cached_tokens, 0, "{name}: the reference path looks nothing up");
        let inserts0 = rig.eng.prefix_stats().inserts;

        let t = std::time::Instant::now();
        let got_a = rig.one(&a_ids, &a_spans, greedy(), n_new, true);
        let cold_ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(got_a.cached_tokens, 0, "{name}: the first sighting hits nothing");
        same_reply(&format!("{name}: donor"), &got_a, &ref_a_g);
        assert!(rig.eng.prefix_stats().inserts > inserts0, "{name}: the donor left an entry");

        // the second turn hits the donor's end cut: after the pictures, so the restored slot carries the canvas's rope delta
        let h0 = hits(&rig);
        let t = std::time::Instant::now();
        let got_b = rig.one(&b_ids, &a_spans, greedy(), n_new, true);
        let hit_ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(got_b.cached_tokens, end_cut_a, "{name}: the hit is the donor's end cut");
        assert_eq!(hits(&rig), h0 + 1);
        same_reply(&format!("{name}: second turn, greedy"), &got_b, &ref_b_g);
        let got_bs = rig.one(&b_ids, &a_spans, sampling, n_new, true);
        assert!(got_bs.cached_tokens >= end_cut_a, "{name}: sampled second turn hits too ({})", got_bs.cached_tokens);
        same_reply(&format!("{name}: second turn, sampled"), &got_bs, &ref_b_s);
        // the donor's prompt again: its own end cut (greedy and sampled)
        let again = rig.one(&a_ids, &a_spans, greedy(), n_new, true);
        assert_eq!(again.cached_tokens, end_cut_a, "{name}: the identical prompt hits its end cut");
        same_reply(&format!("{name}: identical prompt"), &again, &ref_a_g);
        let again_s = rig.one(&a_ids, &a_spans, sampling, n_new, true);
        assert_eq!(again_s.cached_tokens, end_cut_a);
        same_reply(&format!("{name}: identical prompt, sampled"), &again_s, &ref_a_s);
        eprintln!(
            "CACHE {name}: prompt {} tokens, hit {end_cut_a}; whole request {cold_ms:.0} ms first sighting, {hit_ms:.0} ms with the hit \
             (24 tokens)",
            a_ids.len()
        );
    }

    // 2. the same ids, another picture: no hit, and its own reply (the pictures' identity is part of the key)
    let (p_ids, p_spans) = rig.prepare(&rig.chat(None, &["photo_0.png"], false, LONG_QUESTION));
    let (c_ids, c_spans) = rig.prepare(&rig.chat(None, &["chart_0.png"], false, LONG_QUESTION));
    assert_eq!(p_ids, c_ids, "the two 800x600 pictures give one prompt layout");
    assert_ne!(p_spans[0].key, c_spans[0].key, "and two cache identities");
    let ref_p = rig.one(&p_ids, &p_spans, greedy(), n_new, false);
    let ref_c = rig.one(&c_ids, &c_spans, greedy(), n_new, false);
    assert_ne!(ref_p.tokens, ref_c.tokens, "the replies depend on the picture (the control has power)");
    let got_p = rig.one(&p_ids, &p_spans, greedy(), n_new, true);
    same_reply("photo, first sighting", &got_p, &ref_p);
    let got_c = rig.one(&c_ids, &c_spans, greedy(), n_new, true);
    assert_eq!(got_c.cached_tokens, 0, "identical ids with another picture hit nothing");
    same_reply("chart after photo", &got_c, &ref_c);
    let got_p2 = rig.one(&p_ids, &p_spans, greedy(), n_new, true);
    assert_eq!(got_p2.cached_tokens, p_ids.len() - 16, "the photo's own entry is still its own");
    same_reply("photo again", &got_p2, &ref_p);

    // 3. a cut that would fall inside the picture (a short question: the end cut is inside the frame) is dropped, not taken
    let (s_ids, s_spans) = rig.prepare(&rig.chat(None, &["small_0.png"], false, "What is shown here?"));
    let frame = (s_spans[0].start, s_spans[0].start + s_spans[0].tokens);
    let end_cut = s_ids.len() - 16;
    let inside = end_cut > frame.0 && end_cut < frame.1;
    eprintln!("CACHE short question: prompt {} tokens, frame {frame:?}, end cut {end_cut} inside: {inside}", s_ids.len());
    let ref_s = rig.one(&s_ids, &s_spans, greedy(), n_new, false);
    let inserts0 = rig.eng.prefix_stats().inserts;
    let got_s = rig.one(&s_ids, &s_spans, greedy(), n_new, true);
    assert_eq!(got_s.cached_tokens, 0);
    if inside {
        assert_eq!(rig.eng.prefix_stats().inserts, inserts0, "a prompt whose only cut is inside its picture leaves no entry");
    }
    same_reply("short question", &got_s, &ref_s);
    let got_s2 = rig.one(&s_ids, &s_spans, greedy(), n_new, true);
    same_reply("short question again", &got_s2, &ref_s);
    if inside {
        assert_eq!(got_s2.cached_tokens, 0, "nothing to hit");
    }

    // 4. a long system prompt shared by chats of different pictures: the second leaves the shared text prefix as an entry (the
    // cut is before the picture, so its rope delta is 0), the third hits it with its own picture's prefill after it
    let system = "You are a careful assistant that answers questions about pictures. Always answer in complete sentences, never \
                  invent text that is not visible in the picture, and say so when you cannot tell. "
        .repeat(24);
    let chats: Vec<(Vec<u32>, Vec<ImageSpan>)> = ["photo_0.png", "chart_0.png", "doc_0.png"]
        .iter()
        .map(|f| rig.prepare(&rig.chat(Some(&system), &[f], false, LONG_QUESTION)))
        .collect();
    let first_picture = chats.iter().map(|(_, s)| s[0].start).min().unwrap();
    assert!(first_picture >= 600, "the system prompt is {first_picture} tokens");
    let refs: Vec<Output> = chats.iter().map(|(i, s)| rig.one(i, s, greedy(), n_new, false)).collect();
    let mut cached = Vec::new();
    for (k, (i, s)) in chats.iter().enumerate() {
        let got = rig.one(i, s, greedy(), n_new, true);
        same_reply(&format!("system prompt, chat {k}"), &got, &refs[k]);
        cached.push(got.cached_tokens);
    }
    eprintln!("CACHE system prompt of {first_picture} tokens, three pictures: cached tokens {cached:?}");
    assert_eq!(cached[0], 0);
    assert!(cached[2] > 0 && cached[2] <= first_picture, "the third chat hit the shared text prefix: {cached:?}");
    let again = rig.one(&chats[2].0, &chats[2].1, sampled(5), n_new, true);
    let want = rig.one(&chats[2].0, &chats[2].1, sampled(5), n_new, false);
    same_reply("system prompt, sampled", &again, &want);

    // the books balance with nothing running
    let st = rig.eng.prefix_stats();
    assert_eq!(st.restore_failures, 0);
    let arena = rig.eng.backend.prefix_arena().unwrap();
    assert_eq!(arena.held_count() as u64, st.entries);
    assert_eq!(arena.held_count() + arena.free_count(), rig.plan.memory.cache.snapshot_slots);
    assert_eq!(rig.eng.backend.m.kv_tab.free_blocks() as u64 + st.pages_held, rig.kv_pages as u64, "pages: free + held = pool");
    eprintln!("prefix stats at the end: {st:?}");
}

/// The DSpark draft (E4b) with images and the prefix cache together, through the engine `engine_from` builds (tower, cache and draft all
/// on): a drafted request on a picture, two pictures or a clip equals the serial, uncached, undrafted reply token for token, greedy and
/// sampled, whether it is served from the prefix cache or not; a second turn hits the donor's end cut and DRAFTS from the donor's draft
/// context (the snapshot's draft rows are in the pages it shares and the partial page it copies: restoring them is what makes
/// `drafted > 0` on a hit); a burst of hits shares the pool; and a long system prompt shared by chats of different pictures is hit by
/// a drafted request that prefills its own picture after it.
#[test]
#[ignore]
fn a_drafted_image_prompt_served_from_the_prefix_cache_equals_the_uncached_serial_one_greedy_and_sampled() {
    let draft = std::env::var_os("REXL3_DSPARK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join("models/minicpm-v-4.7-35b-a3b/draft-packs/dspark-6.00bpw"));
    if !draft.join("config.json").exists() {
        eprintln!("not run: no draft at {}", draft.display());
        return;
    }
    // the cut policy of the engine tests (the shipped minimum is 512 tokens): short prompts exercise the cache
    let policy = PrefixConfig {
        policy: CutPolicy { min_cut: 128, tail: 16, ghost_min_gain: 512, min_gap: 64 },
        ghost_blocks: 1 << 12,
        ..Default::default()
    };
    let Some(mut rig) = CacheRig::load_full(2, 4096, 512, 24, Some(policy), Some(draft)) else { return };
    assert!(rig.eng.backend.m.has_draft(), "the engine runs with the draft attached");
    assert!(rig.eng.prefix_stats().enabled);
    let follow = rig.tok.encode_bounded(" Then answer again in one short sentence.", 1 << 20).unwrap();
    let (n_new, sampling) = (40usize, sampled(11));
    let hits = |r: &CacheRig| r.eng.prefix_stats().hits;
    // acceptance length of a reply: tokens a verify round commits
    let al = |o: &Output| if o.rounds == 0 { 0.0 } else { 1.0 + o.accepted as f64 / o.rounds as f64 };

    let scenarios: [(&str, Vec<&str>, bool); 3] = [
        ("one picture", vec!["doc_0.png"], false),
        ("two pictures", vec!["two_0.png", "two_1.png"], false),
        (
            "a clip of six frames",
            vec!["video_0.png", "video_1.png", "video_2.png", "video_3.png", "video_4.png", "video_5.png"],
            true,
        ),
    ];
    for (name, files, video) in &scenarios {
        let (a_ids, a_spans) = rig.prepare(&rig.chat(None, files, *video, LONG_QUESTION));
        let b_ids: Vec<u32> = a_ids.iter().chain(&follow).copied().collect();
        let end_cut_a = a_ids.len() - 16;
        // the serial references: no draft, no cache
        let ref_a_g = rig.one_with(&a_ids, &a_spans, greedy(), n_new, false, false);
        let ref_a_s = rig.one_with(&a_ids, &a_spans, sampling, n_new, false, false);
        let ref_b_g = rig.one_with(&b_ids, &a_spans, greedy(), n_new, false, false);
        let ref_b_s = rig.one_with(&b_ids, &a_spans, sampling, n_new, false, false);
        assert_eq!(ref_a_g.drafted + ref_b_g.drafted, 0, "{name}: the reference is not drafted");

        // 1. uncached, drafted: the draft's context is the picture prompt's rows (taps exported for every row, image rows included)
        let d_a = rig.one_with(&a_ids, &a_spans, greedy(), n_new, false, true);
        same_reply(&format!("{name}: drafted, uncached, greedy"), &d_a, &ref_a_g);
        let d_as = rig.one_with(&a_ids, &a_spans, sampling, n_new, false, true);
        same_reply(&format!("{name}: drafted, uncached, sampled"), &d_as, &ref_a_s);
        assert!(
            d_a.drafted > 0 && d_a.accepted > 0,
            "{name}: the draft drafts after a picture ({} drafted, {} accepted)",
            d_a.drafted,
            d_a.accepted
        );
        eprintln!(
            "DRAFT {name}: prompt {} tokens; greedy {} rounds, {} drafted, {} accepted (AL {:.2}); sampled AL {:.2}",
            a_ids.len(),
            d_a.rounds,
            d_a.drafted,
            d_a.accepted,
            al(&d_a),
            al(&d_as)
        );

        // 2. the donor under the cache, drafted: nothing to hit, an entry left behind
        let inserts0 = rig.eng.prefix_stats().inserts;
        let got_a = rig.one_with(&a_ids, &a_spans, greedy(), n_new, true, true);
        assert_eq!(got_a.cached_tokens, 0, "{name}: the first sighting hits nothing");
        same_reply(&format!("{name}: donor, drafted"), &got_a, &ref_a_g);
        assert!(rig.eng.prefix_stats().inserts > inserts0, "{name}: the donor left an entry");
        assert!(got_a.drafted > 0, "{name}: the donor drafts");

        // 3. the second turn hits the donor's end cut (after the pictures: the restored slot carries the canvas's rope delta) and
        // drafts from the draft context the snapshot holds
        let h0 = hits(&rig);
        let got_b = rig.one_with(&b_ids, &a_spans, greedy(), n_new, true, true);
        assert_eq!(got_b.cached_tokens, end_cut_a, "{name}: the hit is the donor's end cut");
        assert_eq!(hits(&rig), h0 + 1);
        same_reply(&format!("{name}: second turn, greedy, drafted"), &got_b, &ref_b_g);
        assert!(
            got_b.drafted > 0 && got_b.accepted > 0,
            "{name}: a restored slot drafts from the donor's draft context ({} drafted, {} accepted)",
            got_b.drafted,
            got_b.accepted
        );
        let got_bs = rig.one_with(&b_ids, &a_spans, sampling, n_new, true, true);
        assert!(got_bs.cached_tokens >= end_cut_a, "{name}: sampled second turn hits too ({})", got_bs.cached_tokens);
        same_reply(&format!("{name}: second turn, sampled, drafted"), &got_bs, &ref_b_s);
        assert!(got_bs.drafted > 0);
        // the donor's prompt again: its own end cut
        let again = rig.one_with(&a_ids, &a_spans, greedy(), n_new, true, true);
        assert_eq!(again.cached_tokens, end_cut_a, "{name}: the identical prompt hits its end cut");
        same_reply(&format!("{name}: identical prompt, drafted"), &again, &ref_a_g);
        assert!(again.drafted > 0);

        // 4. a burst of hits: two slots at once, one greedy and one sampled, both drafted, both from the cache
        let both = rig.burst(&[(&b_ids, &a_spans, greedy(), true, true), (&a_ids, &a_spans, sampling, true, true)], n_new);
        same_reply(&format!("{name}: burst, second turn"), &both[0], &ref_b_g);
        same_reply(&format!("{name}: burst, identical prompt, sampled"), &both[1], &ref_a_s);
        assert!(both.iter().all(|o| o.cached_tokens > 0 && o.drafted > 0), "{name}: the burst's streams hit the cache and draft");
        eprintln!(
            "DRAFT {name} (cache hit {end_cut_a}): second turn {} rounds, {} drafted, {} accepted (AL {:.2})",
            got_b.rounds,
            got_b.drafted,
            got_b.accepted,
            al(&got_b)
        );
    }

    // a long system prompt shared by chats of different pictures: the second chat leaves the shared text prefix as an entry (before the
    // picture: delta 0), the third hits it with a drafted request that prefills its own picture after the restored prefix
    let system = "You are a careful assistant that answers questions about pictures. Always answer in complete sentences, never \
                  invent text that is not visible in the picture, and say so when you cannot tell. "
        .repeat(24);
    let chats: Vec<(Vec<u32>, Vec<ImageSpan>)> = ["photo_0.png", "chart_0.png", "doc_0.png"]
        .iter()
        .map(|f| rig.prepare(&rig.chat(Some(&system), &[f], false, LONG_QUESTION)))
        .collect();
    let refs: Vec<Output> = chats.iter().map(|(i, s)| rig.one_with(i, s, greedy(), n_new, false, false)).collect();
    let mut cached = Vec::new();
    for (k, (i, s)) in chats.iter().enumerate() {
        let got = rig.one_with(i, s, greedy(), n_new, true, true);
        same_reply(&format!("system prompt, chat {k}, drafted"), &got, &refs[k]);
        assert!(got.drafted > 0, "chat {k} drafts");
        cached.push(got.cached_tokens);
    }
    eprintln!("DRAFT system prompt, three pictures: cached tokens {cached:?}");
    assert_eq!(cached[0], 0);
    assert!(cached[2] > 0, "the third chat hit the shared text prefix: {cached:?}");

    // the books balance with nothing running (the draft's overflow pages are not the pool's: they never move)
    let st = rig.eng.prefix_stats();
    assert_eq!(st.restore_failures, 0);
    let arena = rig.eng.backend.prefix_arena().unwrap();
    assert_eq!(arena.held_count() as u64, st.entries);
    assert_eq!(arena.held_count() + arena.free_count(), rig.plan.memory.cache.snapshot_slots);
    assert_eq!(rig.eng.backend.m.kv_tab.free_blocks() as u64 + st.pages_held, rig.kv_pages as u64, "pages: free + held = pool");
    eprintln!("prefix stats at the end: {st:?}");
}

// ---- the backend-level exactness of an image snapshot

fn hash_bytes(h: &mut std::collections::hash_map::DefaultHasher, buf: &gb_cuda::DevBuf, at: usize, n: usize) {
    use std::hash::Hasher;
    let mut data = vec![0u8; n];
    buf.download_range(at, &mut data).unwrap();
    h.write(&data);
}

/// Everything a prompt of `len` tokens leaves in `slot`: the GDN state, the convolution history, the KV rows (through the slot's
/// own page table) and the rope delta.
fn slot_digest(m: &mut MiniCpmModel, slot: usize, len: usize) -> u64 {
    use std::hash::Hasher;
    m.flush().unwrap();
    m.stream.sync().unwrap();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let slots = m.opts.slots;
    for s in &m.pools.s {
        hash_bytes(&mut h, s, slot * (s.bytes() / slots), s.bytes() / slots);
    }
    for c in &m.pools.conv {
        hash_bytes(&mut h, c, slot * (c.bytes() / slots), c.bytes() / slots);
    }
    let blocks = m.kv_tab.blocks_of(slot).to_vec();
    assert!(blocks.len() >= len.div_ceil(256), "slot {slot} holds {} pages for {len} tokens", blocks.len());
    for cache in &m.kv {
        let token = cache.kv_heads.token_elems() * 2;
        for p0 in (0..len).step_by(256) {
            let rows = (len - p0).min(256);
            let at = blocks[p0 / 256] as usize * cache.block_bytes();
            hash_bytes(&mut h, &cache.k, at, rows * token);
            hash_bytes(&mut h, &cache.v, at, rows * token);
        }
    }
    h.write(&m.rope_delta[slot].to_le_bytes());
    h.finish()
}

/// A prompt with pictures prefilled in `slot` from `done` with snapshots at `cuts`: the captures (each with the slot's digest read
/// the moment its slice ended there) and the last row.
fn drive(
    b: &mut MiniCpmBackend,
    slot: usize,
    ids: &[u32],
    spans: &[ImageSpan],
    mut done: usize,
    cuts: &[usize],
) -> (Vec<(gb_engine::prefix::Captured, u64)>, RowOutput) {
    let mut taken = Vec::new();
    loop {
        let step = b.prefill_slice_cuts(slot, ids, spans, done, 16, cuts);
        assert!(step.step.done > done, "no progress at {done}");
        done = step.step.done;
        for c in step.captured {
            assert_eq!(c.len, done, "a cut is taken exactly where it was asked");
            let at_cut = slot_digest(&mut b.m, slot, c.len);
            taken.push((c, at_cut));
        }
        if let Some(row) = step.step.row {
            assert!(b.take_error().is_none(), "the prefill failed");
            return (taken, row);
        }
    }
}

/// An image prompt's snapshot is the donor's slot at the cut: state, convolution history, KV rows and the rope delta, byte for byte;
/// the continuation after the restore, on another chunk grid, ends at the uncached bits and the uncached state; and a cut asked for
/// inside the picture is not taken.
#[test]
#[ignore]
fn an_image_snapshot_restores_the_exact_state_and_the_rope_delta() {
    let Some(mut rig) = CacheRig::load(3, 4096, 512, 6) else { return };
    let (ids, spans) = rig.prepare(&rig.chat(None, &["doc_0.png"], false, LONG_QUESTION));
    let total = ids.len();
    let (frame_start, frame_end) = (spans[0].start, spans[0].start + spans[0].tokens);
    let end_cut = total - 16;
    assert!(frame_end <= end_cut && frame_start > 0);
    let inside = frame_start + spans[0].tokens / 2;
    let b = &mut rig.eng.backend;
    b.chunk = 512;
    // the uncached run on slot 2: no cuts
    let (none, reference) = drive(b, 2, &ids, &spans, 0, &[]);
    assert!(none.is_empty());
    let ref_state = slot_digest(&mut b.m, 2, total);
    let ref_delta = b.m.rope_delta[2];
    assert_ne!(ref_delta, 0, "the canvas of a picture moves the positions of the text after it (the control has power)");
    // the donor on slot 0: cuts before the picture, inside it, at the end of the picture and at the end cut
    let asked = [frame_start, inside, frame_end, end_cut];
    let (taken, row) = drive(b, 0, &ids, &spans, 0, &asked);
    let lens: Vec<usize> = taken.iter().map(|(c, _)| c.len).collect();
    assert_eq!(lens, vec![frame_start, frame_end, end_cut], "the cut inside the picture is not taken");
    assert!(row.bits_eq(&reference), "taking snapshots does not change the prompt's bits");
    assert_eq!(slot_digest(&mut b.m, 0, total), ref_state, "nor its state");
    for (cap, at_cut) in &taken {
        let cut = cap.len;
        let meta = b.prefix_arena().unwrap().meta(cap.handle.snap).unwrap();
        let want_delta = if cut <= frame_start { 0 } else { ref_delta };
        assert_eq!(meta.rope_delta, want_delta, "the snapshot at {cut} names the offset of the text after it");
        // restore into slot 1, compare with the donor as it was at the cut, then finish on another chunk grid
        b.chunk = 257;
        b.prefix_restore(1, &cap.handle, cut).unwrap_or_else(|e| panic!("restore at {cut}: {e}"));
        assert_eq!(slot_digest(&mut b.m, 1, cut), *at_cut, "state, KV rows and rope delta after the restore at {cut}");
        let (more, row) = drive(b, 1, &ids, &spans, cut, &[]);
        assert!(more.is_empty());
        assert!(row.bits_eq(&reference), "candidates after a hit at {cut}: {:?} against {:?}", row.ids, reference.ids);
        assert_eq!(slot_digest(&mut b.m, 1, total), ref_state, "state after a hit at {cut}");
        assert_eq!(b.m.rope_delta[1], ref_delta, "decode after a hit at {cut} runs at the uncached offset");
        b.release(1);
    }
    // every handle goes back exactly once, and the pool is whole
    let handles: Vec<gb_engine::PrefixHandle> = taken.into_iter().map(|(c, _)| c.handle).collect();
    b.release(0);
    b.release(2);
    for h in handles {
        b.prefix_release(h);
    }
    assert!(b.take_error().is_none());
    assert_eq!(rig.eng.backend.m.kv_tab.free_blocks(), rig.kv_pages, "every page is back in the pool");
}

/// Time to the first token of one image at 16x with the prefix cache on (the engine as `serve` builds it): the first request of a
/// prompt (the ingest and the whole prefill) and the same prompt again (the ingest from the feature cache, the restore at the end cut,
/// the 16-token tail). Under the SHIPPED cut policy (nothing below 512 tokens is cut: the 4-crop prompt of 340 tokens is not cached) and,
/// for the 4-crop prompt, under the test policy (cuts from 128). Alone on the box (exclusive lock).
#[test]
#[ignore]
fn bench_image_latency_with_the_prefix_cache() {
    for shipped in [true, false] {
        let policy = (!shipped).then_some(PrefixConfig {
            policy: CutPolicy { min_cut: 128, tail: 16, ghost_min_gain: 512, min_gap: 64 },
            ghost_blocks: 1 << 12,
            ..Default::default()
        });
        let Some(mut rig) = CacheRig::load_with(1, 8192, 4096, 6, policy) else { return };
        let which = if shipped { "shipped policy" } else { "min_cut 128" };
        for (name, file) in [("4 crops", "bench_4crops.png"), ("10 crops", "bench_10crops.png")] {
            if !shipped && name != "4 crops" {
                continue;
            }
            let messages = rig.chat(None, &[file], false, LONG_QUESTION);
            let mut line = Vec::new();
            for pass in 0..4 {
                let t = std::time::Instant::now();
                let (ids, spans) = rig.prepare(&messages);
                let ingest = t.elapsed().as_secs_f64() * 1e3;
                let t = std::time::Instant::now();
                let out = rig.one(&ids, &spans, greedy(), 1, true);
                let gen = t.elapsed().as_secs_f64() * 1e3;
                eprintln!(
                    "LATENCY cache ({which}) {name} request {pass}: prompt {} tokens, hit {}, ingest {ingest:.1} ms, prefill + first token {gen:.0} ms, TTFT {:.0} ms",
                    ids.len(),
                    out.cached_tokens,
                    ingest + gen
                );
                line.push((ids.len(), out.cached_tokens, ingest + gen));
            }
            assert_eq!(line[0].1, 0, "{name}: the first sighting hits nothing");
            let cacheable = !shipped || line[0].0 - 16 >= 512;
            if cacheable {
                assert!(line[1].1 > 0 && line[2].1 > 0 && line[3].1 > 0, "{name}: the repeats hit: {line:?}");
            } else {
                assert!(line.iter().all(|l| l.1 == 0), "{name}: below the shipped minimum cut nothing is cached: {line:?}");
            }
        }
    }
}
