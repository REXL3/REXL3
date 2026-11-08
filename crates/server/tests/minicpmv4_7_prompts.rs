//! The MiniCPM-V-4.7 prompt of an image or a clip, token for token against the Hugging Face processor (E5b of
//! `docs/MINICPMV4_7_INTEGRATION_PLAN.md`): the chat template, the tokenizer, the crop plan of the real preprocessing and the layout of
//! `formats/minicpmv.rs`, with a host-only encoder standing in for the tower (the rows are not the question here, the ids and the
//! spans are). CPU only.
//!
//! Needs the pack folder's `tokenizer.json`, `chat_template.jinja`, `config.json` and `preprocessor_config.json`
//! (`REXL3_MINICPM_DIR`, default `~/models/minicpm-v-4.7-35b-a3b/packs/q3-6.00bpw`) and the cases of `tools/minicpm_v47/mm_cases.py`
//! with the processor's output of `mm_ref_logits.py process` (`REXL3_MINICPM_MM_DIR`, default `~/projects/minicpm-mm`: `cases/`,
//! `proc/hf_prompts.json`); prints `not run:` otherwise.

use gb_model::error::ModelError;
use gb_model::families::minicpmv4_7::splice::{layout_of, FrameLayout, HostFeatures, IMAGE_PAD, VIDEO_PAD};
use gb_model::families::minicpmv4_7::vision::{image_preprocessing, video_preprocessing};
use gb_model::mrope_canvas::Downsample;
use gb_model::serve::{EncodedImage, ImageEncoder, VisionService};
use gb_server::chat_format::{ChatFormat, EncodedRef, VideoOptions};
use gb_server::formats::minicpmv::MiniCpmFormat;
use gb_server::formats::nemotron_omni::prep::data_uri;
use gb_server::tokenizer::Tok;
use gb_server::vision::{ImageRef, Limits};
use gb_vision::minicpmv::{Downsample as VisionDownsample, MiniCpmPreprocessing};
use gb_vision::Preprocessed;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

struct Env {
    pack: PathBuf,
    cases: PathBuf,
    hf: Value,
}

fn env() -> Option<Env> {
    let pack = std::env::var("REXL3_MINICPM_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home().join("models/minicpm-v-4.7-35b-a3b/packs/q3-6.00bpw"));
    let work = std::env::var("REXL3_MINICPM_MM_DIR").map(PathBuf::from).unwrap_or_else(|_| home().join("projects/minicpm-mm"));
    for f in ["tokenizer.json", "chat_template.jinja", "config.json", "preprocessor_config.json"] {
        if !pack.join(f).exists() {
            eprintln!("not run: {} has no {f}", pack.display());
            return None;
        }
    }
    let hf = work.join("proc/hf_prompts.json");
    let Ok(text) = std::fs::read_to_string(&hf) else {
        eprintln!("not run: {} is missing (tools/minicpm_v47/mm_ref_logits.py process)", hf.display());
        return None;
    };
    Some(Env { pack, cases: work.join("cases"), hf: serde_json::from_str(&text).unwrap() })
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

/// The tower's place: the crop plan of the preprocessing for the image's size, as host features.
struct HostEncoder {
    image: MiniCpmPreprocessing,
    video: MiniCpmPreprocessing,
}

impl ImageEncoder for HostEncoder {
    fn encode(&mut self, p: &Preprocessed) -> Result<EncodedImage, ModelError> {
        for pp in [&self.image, &self.video] {
            let Ok(plan) = pp.plan(p.size.0, p.size.1) else { continue };
            if p.grid == (1, plan.grid.0, plan.grid.1) && p.patches.len() == plan.total_patches * p.patch_dim {
                let layout = FrameLayout {
                    rows: plan.total_tokens,
                    crops: plan.target_sizes(),
                    downsample: match pp.downsample {
                        VisionDownsample::X16 => Downsample::X16,
                        VisionDownsample::X4 => Downsample::X4,
                    },
                    slice_grid: plan.grid,
                };
                return Ok(EncodedImage {
                    tokens: plan.total_tokens,
                    merged: (1, plan.total_tokens),
                    features: Arc::new(HostFeatures(layout)),
                });
            }
        }
        Err(ModelError::Request("no preprocessing of the host encoder makes this grid".into()))
    }
}

struct Rig {
    format: MiniCpmFormat,
    tok: Tok,
    svc: VisionService,
    limits: Limits,
}

fn rig(e: &Env, image: MiniCpmPreprocessing, video: MiniCpmPreprocessing) -> Rig {
    let template = std::fs::read_to_string(e.pack.join("chat_template.jinja")).unwrap();
    let mut format = MiniCpmFormat::new(&template).unwrap().with_preprocessing(image.clone(), video.clone());
    format
        .enable_video(&VideoOptions { max_frames: 128, max_tokens: 16384, pruning_rate: 0.0, require_square_grid: false })
        .unwrap();
    Rig {
        format,
        tok: Tok::from_dir(&e.pack).unwrap(),
        svc: VisionService::new(Box::new(HostEncoder { image, video }), 4096),
        limits: Limits::default(),
    }
}

fn messages(c: &Value, dir: &Path) -> Value {
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
    json!([{"role": "user", "content": content}])
}

fn ingest(r: &Rig, messages: &Value) -> gb_server::formats::nemotron_omni::prep::Ingested {
    let kwargs = serde_json::Map::new();
    let vf = r.format.video().expect("video is on");
    let limits = r.limits;
    let mut fetch = |im: &ImageRef| data_uri(im, &limits);
    vf.ingest_frames(&r.tok, messages, &kwargs, &limits, 8 << 20, &r.svc, None, &mut fetch).unwrap()
}

fn hf_ids(e: &Env, name: &str) -> Vec<u32> {
    e.hf[name]["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
}

fn cases(e: &Env) -> Vec<Value> {
    let text = std::fs::read_to_string(e.cases.join("cases.json")).unwrap();
    serde_json::from_str::<Value>(&text).unwrap()["cases"].as_array().unwrap().clone()
}

#[test]
fn the_prompt_of_every_case_is_the_processors_token_for_token() {
    let Some(e) = env() else { return };
    let image = image_preprocessing(&e.pack).unwrap();
    let video = video_preprocessing(&image).unwrap();
    let r = rig(&e, image, video);
    let mut compared = 0usize;
    for c in cases(&e) {
        let name = c["name"].as_str().unwrap();
        if c["kind"] == "text" {
            continue;
        }
        let msgs = messages(&c, &e.cases);
        let ing = ingest(&r, &msgs);
        let want = hf_ids(&e, name);
        if let Some(i) = (0..want.len().min(ing.ids.len())).find(|&i| want[i] != ing.ids[i]) {
            panic!(
                "{name}: token {i} is {} but the processor's is {} (lengths {} / {})",
                ing.ids[i],
                want[i],
                ing.ids.len(),
                want.len()
            );
        }
        assert_eq!(ing.ids.len(), want.len(), "{name}: prompt length");
        assert!(ing.starts_in_reasoning, "{name}: thinking is on by default");
        // one span a frame, inside the prompt, never overlapping, each holding exactly its feature rows' worth of pads
        let video = c["kind"] == "video";
        let frames = if video { c["frames"].as_array().unwrap().len() } else { c["images"].as_array().unwrap().len() };
        assert!(ing.images.len() >= frames, "{name}: {} spans", ing.images.len());
        let pad = if video { VIDEO_PAD } else { IMAGE_PAD };
        let mut at = 0;
        for s in &ing.images {
            assert!(s.start >= at && s.start + s.tokens <= ing.ids.len() && s.merged == (1, s.tokens), "{name}: span {s:?}");
            at = s.start + s.tokens;
            let layout = layout_of(&*s.features).unwrap();
            let pads = ing.ids[s.start..s.start + s.tokens].iter().filter(|&&t| t == pad).count();
            assert_eq!(pads, layout.rows, "{name}: pads of a span against its feature rows");
            assert_eq!(ing.ids[s.start], 248_078, "{name}: a span starts at its <image> marker");
            assert_ne!(s.key, 0);
        }
        // every pad token of the prompt is inside a span
        let total_pads = ing.ids.iter().filter(|&&t| t == pad).count();
        let in_spans: usize =
            ing.images.iter().map(|s| ing.ids[s.start..s.start + s.tokens].iter().filter(|&&t| t == pad).count()).sum();
        assert_eq!(total_pads, in_spans, "{name}");
        compared += want.len();
        eprintln!("{name}: {} tokens equal, {} spans", want.len(), ing.images.len());
    }
    assert!(compared > 1500, "{compared}");
}

#[test]
fn the_image_path_places_the_same_prompt_as_the_whole_ingest() {
    // `chat_prompt_mm` of http.rs renders, tokenizes and then calls `place_images` with what the encoder returned
    let Some(e) = env() else { return };
    let image = image_preprocessing(&e.pack).unwrap();
    let video = video_preprocessing(&image).unwrap();
    let r = rig(&e, image.clone(), video);
    for c in cases(&e) {
        if c["kind"] != "image" {
            continue;
        }
        let name = c["name"].as_str().unwrap();
        let msgs = messages(&c, &e.cases);
        let rendered = r.format.render_bounded(&msgs, &serde_json::Map::new(), 8 << 20).unwrap();
        let ids = r.tok.encode_bounded(&rendered.text, 8 << 20).unwrap();
        // encode each image the way the server does (decode, preprocess, the host encoder)
        let mut feats: Vec<(EncodedImage, usize)> = Vec::new();
        for f in c["images"].as_array().unwrap() {
            let url = data_url(&e.cases, f.as_str().unwrap());
            let im = ImageRef { message: 0, part: 0, url };
            let bytes = data_uri(&im, &r.limits).unwrap();
            let pre = gb_server::vision::prepare_bytes(&im, &bytes, &r.limits, &image, None).unwrap();
            feats.push((r.svc.encode(&pre).unwrap(), 0));
        }
        let refs: Vec<EncodedRef<'_>> =
            feats.iter().map(|(en, _)| EncodedRef { tokens: en.tokens, merged: en.merged, features: &*en.features }).collect();
        let (out, placed) = r.format.place_images(&r.tok, &ids, &refs).unwrap();
        assert_eq!(out, hf_ids(&e, name), "{name}: place_images differs from the processor");
        let whole = ingest(&r, &msgs);
        assert_eq!(whole.ids, out, "{name}: the two paths place different prompts");
        for (p, s) in placed.iter().zip(&whole.images) {
            assert_eq!((p.start, p.tokens, p.merged), (s.start, s.tokens, s.merged), "{name}");
        }
    }
}

#[test]
fn a_thumbnail_only_image_and_four_x_mode_change_the_layout_as_the_processor_does() {
    let Some(e) = env() else { return };
    // 4x: 256 tokens a crop; the host plan is the preprocessing's, the ids are the layout's
    let image = image_preprocessing(&e.pack).unwrap().with_downsample(VisionDownsample::X4);
    let video = video_preprocessing(&image).unwrap();
    let r = rig(&e, image, video);
    let c = cases(&e).into_iter().find(|c| c["name"] == "small").unwrap();
    let ing = ingest(&r, &messages(&c, &e.cases));
    assert_eq!(ing.images.len(), 1);
    let pads = ing.ids.iter().filter(|&&t| t == IMAGE_PAD).count();
    assert_eq!(pads, 256, "a 336 x 336 image at 4x: one crop of 24 x 24 patches... made 16 x 16 tokens");
    // no slice markers for a thumbnail alone
    assert!(!ing.ids.contains(&248_088));
}

#[test]
fn the_positions_of_every_case_are_get_rope_index_through_the_whole_request_path() {
    // the prompt the server builds, its spans, and `Geometry` (what the backend feeds the forward) against the reference's own
    // `get_rope_index` on the processor's tensors
    let Some(e) = env() else { return };
    let image = image_preprocessing(&e.pack).unwrap();
    let video = video_preprocessing(&image).unwrap();
    let r = rig(&e, image, video);
    let mut tokens = 0;
    for c in cases(&e) {
        let name = c["name"].as_str().unwrap();
        if c["kind"] == "text" {
            continue;
        }
        let ing = ingest(&r, &messages(&c, &e.cases));
        let geo = gb_model::families::minicpmv4_7::splice::Geometry::new(&ing.ids, &ing.images).unwrap();
        let want = e.hf[name]["positions"].as_array().expect("hf_prompts.json holds positions (process stage)");
        assert_eq!(want.len(), ing.ids.len(), "{name}");
        for (i, w) in want.iter().enumerate() {
            let w: Vec<i64> = w.as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
            let got = geo.canvas.positions[i];
            assert_eq!(
                [got[0] as i64, got[1] as i64, got[2] as i64],
                [w[0], w[1], w[2]],
                "{name}: token {i} (id {})",
                ing.ids[i]
            );
        }
        assert_eq!(geo.rope_delta() as i64, e.hf[name]["rope_delta"].as_i64().unwrap(), "{name}: rope delta");
        tokens += want.len();
    }
    assert!(tokens > 2000);
}
