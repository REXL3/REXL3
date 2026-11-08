//! The image ingress of the Nemotron format: the processor's expanded texts byte for byte (`testdata/nemotron_omni/vision_pre/texts.json`),
//! the ids, spans and content keys of a request, the media refusals, and the same flow over HTTP on the mock engine with a mock encoder
//! (the slot the vision tower fills). No model, no GPU.
//!
//! `REXL3_NEMO_OMNI_DIR` (the pack folder) adds the check against the real tokenizer: the processor's `input_ids` for every prompt.
mod common;

use base64::Engine;
use common::{start, Opts};
use gb_model::{
    error::ModelError,
    serve::{EncodedImage, ImageEncoder, VisionService},
};
use gb_server::api::ApiError;
use gb_server::chat_format::{ChatFormat, FormatRegistry};
use gb_server::formats::nemotron_omni::prep::{data_uri, IngestOptions, NemotronEncoder, NemotronImagePrep, PreparedImage};
use gb_server::formats::nemotron_omni::{expand_text, OmniFormat};
use gb_server::http::VisionHandle;
use gb_server::preparation::PreparationGate;
use gb_server::tokenizer::Tok;
use gb_server::vision::{ImageRef, Limits};
use gb_server::vision_ingress::{HttpsIngress, IngressPolicy};
use gb_vision::nemotron::NemotronPreprocessing;
use gb_vision::preprocessor::ImagePreprocessor;
use serde_json::{json, Map, Value};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

fn testdata() -> PathBuf {
    repo_root().join("testdata")
}

fn fixture(name: &str) -> Value {
    let p = testdata().join("nemotron_omni/vision_pre").join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}

/// A model folder of the Nemotron format with a small synthetic tokenizer (whitespace words, the chat and image tokens as special added tokens,
/// numbered after the vocabulary as the library does) and the pack's own template, config and processor parameters.
fn model_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("rexl3-nemotron-omni-vision-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let added = |id: u32, content: &str| {
            json!({"id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true})
        };
        let specials = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<image>", "<img>", "</img>"];
        let tj = json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": specials.iter().enumerate().map(|(i, s)| added(3 + i as u32, s)).collect::<Vec<_>>(),
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "a": 1, "b": 2}, "unk_token": "[UNK]"}
        });
        std::fs::write(dir.join("tokenizer.json"), tj.to_string()).unwrap();
        // an end-of-turn id the mock can never produce: a request runs to its max_tokens
        std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": [100000], "temperature": 1.0}"#).unwrap();
        for (from, to) in [
            ("nemotron_omni/chat_template.jinja", "chat_template.jinja"),
            ("nemotron_omni/config.json", "config.json"),
            ("nemotron_omni/vision_pre/preprocessor_config.json", "preprocessor_config.json"),
        ] {
            std::fs::copy(testdata().join(from), dir.join(to)).unwrap();
        }
        dir
    })
    .clone()
}

fn open_format() -> Arc<dyn ChatFormat> {
    FormatRegistry::builtin().open(&model_dir(), None).unwrap().into()
}

fn omni() -> OmniFormat {
    OmniFormat::new(&std::fs::read_to_string(testdata().join("nemotron_omni/chat_template.jinja")).unwrap())
        .unwrap()
        .with_preprocessing(NemotronPreprocessing::default())
}

fn png_uri(name: &str) -> String {
    let bytes = std::fs::read(testdata().join("vision").join(name)).unwrap();
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes))
}

fn image_part(name: &str) -> Value {
    json!({"type": "image_url", "image_url": {"url": png_uri(name)}})
}

fn tok() -> Tok {
    Tok::from_dir(&model_dir()).unwrap()
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

fn prep(limits: Limits) -> NemotronImagePrep {
    NemotronImagePrep::new(Arc::new(NemotronPreprocessing::default()), limits)
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// the processor's texts

#[test]
fn the_expanded_texts_are_the_processors_byte_for_byte() {
    let fx = fixture("texts.json");
    let cases = fx["cases"].as_array().unwrap();
    assert!(cases.len() >= 6);
    let f = omni();
    let cfg = NemotronPreprocessing::default();
    let tok = tok();
    let mut with_images = 0;
    for c in cases {
        let name = c["name"].as_str().unwrap();
        let kwargs: Map<String, Value> = c["chat_template_kwargs"].as_object().unwrap().clone();
        // the template, through our renderer, against the processor's `apply_chat_template`
        let rendered = f.render(&c["messages"], &kwargs).unwrap().text;
        assert_eq!(rendered, c["rendered"].as_str().unwrap(), "{name}: the rendered prompt");
        // the tiles of the images in order, from the sizes alone
        let sizes: Vec<(usize, usize)> = c["images"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| (s[0].as_u64().unwrap() as usize, s[1].as_u64().unwrap() as usize))
            .collect();
        let tiles: Vec<_> = sizes.iter().map(|&(w, h)| cfg.plan_for(w, h, sizes.len()).unwrap()).collect();
        let counts: Vec<usize> = tiles.iter().map(|t| t.tokens).collect();
        let want_counts: Vec<usize> =
            c["image_tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert_eq!(counts, want_counts, "{name}: tokens per image");
        for (t, cv) in tiles.iter().zip(c["canvases"].as_array().unwrap()) {
            assert_eq!(
                [t.canvas().0 as u64, t.canvas().1 as u64],
                [cv[0].as_u64().unwrap(), cv[1].as_u64().unwrap()],
                "{name}: canvas"
            );
        }
        // the text the processor hands the tokenizer
        let expanded = if sizes.is_empty() { rendered.clone() } else { expand_text(&rendered, &counts).unwrap() };
        assert_eq!(expanded, c["expanded"].as_str().unwrap(), "{name}: the expanded text");
        // the ids of the expanded text are the ids of the expansion done on the tokenized prompt
        let by_text = tok.encode(&expanded).unwrap();
        let (by_ids, spans) = f.expand_images(&tok, &tok.encode(&rendered).unwrap(), &counts).unwrap();
        assert_eq!(by_ids, by_text, "{name}: ids of the text and the id-level expansion");
        assert_eq!(spans.iter().map(|s| s.tokens).collect::<Vec<_>>(), counts);
        with_images += !sizes.is_empty() as usize;
    }
    assert!(with_images >= 5);
}

#[test]
fn the_real_tokenizer_gives_the_processors_input_ids() {
    let Some(dir) = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from).filter(|d| d.join("tokenizer.json").is_file())
    else {
        eprintln!("REXL3_NEMO_OMNI_DIR is not set: the real-tokenizer check is skipped");
        return;
    };
    let tok = Tok::from_dir(&dir).unwrap();
    let f = omni();
    let cfg = NemotronPreprocessing::default();
    let fx = fixture("texts.json");
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let rendered = c["rendered"].as_str().unwrap();
        let sizes: Vec<(usize, usize)> = c["images"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| (s[0].as_u64().unwrap() as usize, s[1].as_u64().unwrap() as usize))
            .collect();
        let counts: Vec<usize> = sizes.iter().map(|&(w, h)| cfg.plan_for(w, h, sizes.len()).unwrap().tokens).collect();
        let crc = |ids: &[u32]| crc32fast::hash(&ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>());
        let want_n = c["n_input_ids"].as_u64().unwrap() as usize;
        let want_crc = c["input_ids_crc32"].as_u64().unwrap() as u32;
        // the expanded text, and the expansion of the tokenized prompt: both are the processor's input_ids
        let from_text = tok.encode(c["expanded"].as_str().unwrap()).unwrap();
        let (from_ids, spans) =
            f.expand_images(&tok, &tok.encode(rendered).unwrap(), &counts).unwrap_or_else(|e| panic!("{name}: {}", e.message));
        assert_eq!((from_text.len(), crc(&from_text)), (want_n, want_crc), "{name}: ids of the expanded text");
        assert_eq!(from_ids, from_text, "{name}: id-level expansion");
        // the runs of the image id are the spans
        let runs: Vec<(usize, usize)> = c["image_runs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| (r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as usize))
            .collect();
        assert_eq!(spans.iter().map(|s| (s.start, s.tokens)).collect::<Vec<_>>(), runs, "{name}: spans");
    }
    // the format's own ids are the pack's: <image> 18, <img> 19, </img> 20 (config.json img_context_token_id)
    assert_eq!((tok.token_to_id("<image>"), tok.token_to_id("<img>"), tok.token_to_id("</img>")), (Some(18), Some(19), Some(20)));
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// media

#[test]
fn text_requests_render_as_before_and_media_is_gated() {
    let f = omni();
    // a text conversation renders exactly the processor's text (the fixture holds two of them), through the registry's own format as well
    let fx = fixture("texts.json");
    let reg = open_format();
    for c in fx["cases"].as_array().unwrap().iter().filter(|c| c["images"].as_array().unwrap().is_empty()) {
        let kwargs: Map<String, Value> = c["chat_template_kwargs"].as_object().unwrap().clone();
        let want = c["rendered"].as_str().unwrap();
        assert_eq!(f.render(&c["messages"], &kwargs).unwrap().text, want);
        assert_eq!(reg.render(&c["messages"], &kwargs).unwrap().text, want);
        assert!(reg.check_media(&c["messages"], false).is_ok(), "text goes through with vision off");
    }
    let img = json!([{"role": "user", "content": [image_part("check_512x512.png")]}]);
    assert!(reg.check_media(&img, true).is_ok());
    let e = reg.check_media(&img, false).unwrap_err();
    assert_eq!(e.status, 400);
    assert!(e.message.contains("messages[0].content[0]") && e.message.contains("image support"), "{}", e.message);
    let video = json!([{"role": "user", "content": [image_part("check_512x512.png"), {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}}]}]);
    let e = reg.check_media(&video, true).unwrap_err();
    assert!(e.message.contains("messages[0].content[1]") && e.message.contains("video_url"), "{}", e.message);
    let audio = json!([{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AA", "format": "wav"}}]}]);
    let e = reg.check_media(&audio, true).unwrap_err();
    assert!(e.message.contains("input_audio") && e.message.contains("no audio"), "{}", e.message);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// a request with images

fn two_image_messages() -> Value {
    json!([{"role": "user", "content": [
        {"type": "text", "text": "Compare"},
        image_part("check_512x512.png"),
        {"type": "text", "text": "with"},
        image_part("rgba_200x150.png"),
    ]}])
}

#[test]
fn a_request_with_two_images_has_the_expected_ids_spans_and_keys() {
    let (f, tok) = (omni(), tok());
    let p = prep(Limits::default());
    let messages = two_image_messages();
    let kwargs = Map::new();
    let calls = Mutex::new(Vec::new());
    let ing = p
        .ingest(&f, &tok, &messages, IngestOptions::default(), |r: &ImageRef| {
            calls.lock().unwrap().push((r.message, r.part));
            data_uri(r, p.limits())
        })
        .unwrap();
    assert_eq!(*calls.lock().unwrap(), [(0, 1), (0, 3)], "the images in the template's order");

    // the tiles: 512x512 stays 32x32 patches (256 tokens), 200x150 -> 38x30 (285 tokens)
    let tokens: Vec<usize> = ing.prepared.iter().map(|i| i.tokens).collect();
    assert_eq!(tokens, [256, 285]);
    assert_eq!(ing.prepared[0].canvas, (512, 512));
    assert_eq!(ing.prepared[1].canvas, (608, 480));
    assert_eq!(ing.prepared[1].source, (200, 150));
    assert_eq!(ing.prepared[0].pixels.len(), 3 * 512 * 512);

    // the prompt: the rendered text with each `<image>` expanded, as the processor would tokenize it
    let rendered = f.render(&messages, &kwargs).unwrap();
    assert_eq!(rendered.text.matches("<image>").count(), 2);
    let want_ids = tok.encode(&expand_text(&rendered.text, &tokens).unwrap()).unwrap();
    assert_eq!(ing.ids, want_ids);
    let (pad, open, close) =
        (tok.token_to_id("<image>").unwrap(), tok.token_to_id("<img>").unwrap(), tok.token_to_id("</img>").unwrap());
    assert_eq!(ing.ids.iter().filter(|&&t| t == pad).count(), 256 + 285);
    assert_eq!(ing.starts_in_reasoning, rendered.starts_in_reasoning);

    // the spans: where each run of `<image>` sits, its size and merged grid
    assert_eq!(ing.images.len(), 2);
    for (sp, want) in ing.images.iter().zip([(256usize, (16usize, 16usize)), (285, (15, 19))]) {
        assert_eq!((sp.tokens, sp.merged), want);
        assert_eq!(sp.tokens, sp.merged.0 * sp.merged.1);
        assert_eq!(ing.ids[sp.start - 1], open);
        assert!(ing.ids[sp.start..sp.start + sp.tokens].iter().all(|&t| t == pad));
        assert_eq!(ing.ids[sp.start + sp.tokens], close);
    }
    assert!(ing.images[0].start + 256 < ing.images[1].start, "in prompt order, not overlapping");
    assert_eq!(ing.ids.len(), tok.encode(&rendered.text).unwrap().len() + 256 + 285 + 2 * 2 - 2);

    // the keys: image_content_hash(canvas hash, canvas size, the preprocessing's id), the value http.rs computes for the other families
    let pp_id = fnv(&NemotronPreprocessing::default().identity());
    for (sp, prepared) in ing.images.iter().zip(&ing.prepared) {
        let want =
            gb_sched::image_content_hash(prepared.pixel_hash, prepared.canvas.0 as u32, prepared.canvas.1 as u32, pp_id).max(1);
        assert_eq!(sp.key, want);
        assert_eq!(prepared.key, want);
        assert_ne!(sp.key, 0);
        // the generic path (ImagePreprocessor + http.rs) reaches the same hash from the same bytes
        let rgb = NemotronPreprocessing::default()
            .decode(&std::fs::read(testdata().join("vision/check_512x512.png")).unwrap())
            .unwrap();
        if prepared.source == (512, 512) {
            assert_eq!(NemotronPreprocessing::default().preprocess(&rgb).unwrap().pixel_hash, prepared.pixel_hash);
        }
    }
    assert_ne!(ing.images[0].key, ing.images[1].key);
    // with no encoder the span carries the prepared image: a backend can run the tower on it
    let held = ing.images[0].features.downcast_ref::<PreparedImage>().expect("deferred features are the prepared image");
    assert_eq!((held.tokens, held.key), (256, ing.images[0].key));
}

#[test]
fn the_same_image_twice_has_one_key_and_a_changed_preprocessing_another() {
    let (f, tok) = (omni(), tok());
    let msgs = json!([{"role": "user", "content": [image_part("grad_64x64.png"), image_part("grad_64x64.png")]}]);
    let p = prep(Limits::default());
    let a = p.ingest(&f, &tok, &msgs, IngestOptions::default(), |r| data_uri(r, p.limits())).unwrap();
    assert_eq!(a.images[0].key, a.images[1].key, "equal pixels, size and preprocessing: one identity");
    assert_eq!(a.prepared[0].pixels, a.prepared[1].pixels);
    // another normalisation (or window) is another key for the same file
    let other = NemotronPreprocessing { norm_mean: [0.5; 3], ..NemotronPreprocessing::default() };
    let p2 = NemotronImagePrep::new(Arc::new(other), Limits::default());
    let b = p2.ingest(&f, &tok, &msgs, IngestOptions::default(), |r| data_uri(r, p2.limits())).unwrap();
    assert_ne!(a.images[0].key, b.images[0].key);
    assert_ne!(p.id(), p2.id());
    // and the same pixels in another shape are another key, by the size in the hash
    assert_ne!(p.key(a.prepared[0].pixel_hash, (512, 512)), p.key(a.prepared[0].pixel_hash, (256, 1024)));
}

#[test]
fn a_typed_image_tag_pairs_with_the_image_part_and_an_unpaired_one_is_refused() {
    let (f, tok) = (omni(), tok());
    let p = prep(Limits::default());
    let ok =
        json!([{"role": "user", "content": [{"type": "text", "text": "First <image> then look"}, image_part("grad_64x64.png")]}]);
    let ing = p.ingest(&f, &tok, &ok, IngestOptions::default(), |r| data_uri(r, p.limits())).unwrap();
    assert_eq!(ing.images.len(), 1);
    assert_eq!(ing.images[0].tokens, 256);
    // two typed tags and one image: the processor would raise; here it is a 400 naming the counts
    let bad =
        json!([{"role": "user", "content": [{"type": "text", "text": "<image> and <image>"}, image_part("grad_64x64.png")]}]);
    let e = p.ingest(&f, &tok, &bad, IngestOptions::default(), |r| data_uri(r, p.limits())).err().unwrap();
    assert_eq!(e.status, 400);
    assert!(e.message.contains("2 image placeholders but 1 images"), "{}", e.message);
    // a typed tag and no image: untouched text for the engine, but ingest is for image requests: no image parts, no spans
    let none = json!([{"role": "user", "content": "plain <image> text"}]);
    let e = p.ingest(&f, &tok, &none, IngestOptions::default(), |r| data_uri(r, p.limits())).err().unwrap();
    assert!(e.message.contains("1 image placeholders but 0 images"), "{}", e.message);
}

struct Counting {
    seen: Mutex<Vec<(usize, usize, usize)>>,
    fail: Option<fn() -> ModelError>,
}

impl NemotronEncoder for Counting {
    fn encode(&self, image: &PreparedImage) -> Result<Arc<dyn std::any::Any + Send + Sync>, ModelError> {
        self.seen.lock().unwrap().push((image.canvas.0, image.canvas.1, image.tokens));
        match self.fail {
            Some(f) => Err(f()),
            None => Ok(Arc::new(image.tokens)),
        }
    }
}

#[test]
fn the_encoder_is_called_once_per_image_in_order_and_its_failures_keep_their_status() {
    let (f, tok) = (omni(), tok());
    let p = prep(Limits::default());
    let messages = two_image_messages();
    let enc = Counting { seen: Mutex::new(Vec::new()), fail: None };
    let ing = p
        .ingest(&f, &tok, &messages, IngestOptions { encoder: Some(&enc), ..Default::default() }, |r| data_uri(r, p.limits()))
        .unwrap();
    assert_eq!(*enc.seen.lock().unwrap(), [(512, 512, 256), (608, 480, 285)]);
    assert_eq!(*ing.images[1].features.downcast_ref::<usize>().unwrap(), 285, "the span carries what the encoder returned");
    for (fail, status) in [
        (Some((|| ModelError::Busy("the feature memory is in use".into())) as fn() -> ModelError), 503),
        (Some(|| ModelError::Request("not an image the tower takes".into())), 400),
        (Some(|| ModelError::StateLost("the encoder is gone".into())), 500),
    ] {
        let enc = Counting { seen: Mutex::new(Vec::new()), fail };
        let e = p
            .ingest(&f, &tok, &messages, IngestOptions { encoder: Some(&enc), ..Default::default() }, |r| data_uri(r, p.limits()))
            .err()
            .unwrap();
        assert_eq!(e.status, status, "{}", e.message);
        assert!(e.message.contains("messages[0].content[1]"), "{}", e.message);
        assert_eq!(enc.seen.lock().unwrap().len(), 1, "the first failure ends the request");
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// bounds

#[test]
fn an_image_larger_than_the_tower_is_bounded_by_its_token_limit() {
    // the processor's own rule caps any size at 13,312 patches = 3,328 tokens (the tower's max_image_tokens)
    let p = prep(Limits::default());
    for (w, h) in [(8000usize, 8000usize), (7680, 4320), (6000, 4000), (8000, 64), (64, 8000)] {
        let t = p.plan(w, h, 1).unwrap();
        assert!(t.tokens <= 3328, "{w}x{h}: {} tokens", t.tokens);
    }
    // a real 6000 x 4000 image is resized to 2240 x 1504 = 3,290 tokens and goes through
    let (f, tok) = (omni(), tok());
    let msgs = json!([{"role": "user", "content": [image_part("grad_6000x4000_cap.png")]}]);
    let ing = p.ingest(&f, &tok, &msgs, IngestOptions::default(), |r| data_uri(r, p.limits())).unwrap();
    assert_eq!((ing.prepared[0].canvas, ing.prepared[0].tokens, ing.images[0].tokens), ((2240, 1504), 3290, 3290));
    assert_eq!(ing.images[0].merged, (47, 70));
    // a stricter limit (`--max-image-tokens`) refuses it before the resize, naming the numbers
    let tight = prep(Limits { max_tokens_per_image: 3000, ..Limits::default() });
    let e = tight.ingest(&f, &tok, &msgs, IngestOptions::default(), |r| data_uri(r, p.limits())).err().unwrap();
    assert_eq!(e.status, 400);
    assert!(
        e.message.contains("messages[0].content[0]")
            && e.message.contains("3290 tokens")
            && e.message.contains("2240x1504")
            && e.message.contains("limit is 3000"),
        "{}",
        e.message
    );
}

#[test]
fn the_byte_limits_and_bad_data_are_400s_and_413s() {
    let (f, tok) = (omni(), tok());
    let msgs = two_image_messages();
    let p = prep(Limits { max_bytes: 100, ..Limits::default() });
    let png = std::fs::read(testdata().join("vision/check_512x512.png")).unwrap();
    let e = p.ingest(&f, &tok, &msgs, IngestOptions::default(), |_: &ImageRef| Ok(png.clone())).err().unwrap();
    assert_eq!(e.status, 413, "{}", e.message);
    let p = prep(Limits::default());
    let bad = json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]);
    let e = p.ingest(&f, &tok, &bad, IngestOptions::default(), |r| data_uri(r, p.limits())).err().unwrap();
    assert_eq!(e.status, 400);
    assert!(e.message.contains("messages[0].content[0]"), "{}", e.message);
    let remote = json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]);
    let e: ApiError = p.ingest(&f, &tok, &remote, IngestOptions::default(), |r| data_uri(r, p.limits())).err().unwrap();
    assert!(e.message.contains("data: URIs"), "{}", e.message);
    // too many images
    let many = json!([{"role": "user", "content": (0..5).map(|_| image_part("grad_64x64.png")).collect::<Vec<_>>()}]);
    let e = p.ingest(&f, &tok, &many, IngestOptions::default(), |r| data_uri(r, p.limits())).err().unwrap();
    assert!(e.message.contains("at most 4 images"), "{}", e.message);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// over HTTP: the same flow through the server's generic image path, with a mock encoder where the tower goes

/// What the mock tower saw of one image: the canvas, the grid `(t, h, w)` and the number of bf16 values.
type Seen = Arc<Mutex<Vec<((usize, usize), (usize, usize, usize), usize)>>>;

struct MockTower {
    seen: Seen,
}

impl ImageEncoder for MockTower {
    fn encode(&mut self, pre: &gb_vision::Preprocessed) -> Result<EncodedImage, ModelError> {
        self.seen.lock().unwrap().push((pre.size, pre.grid, pre.patches.len()));
        Ok(EncodedImage { tokens: pre.n_patches() / 4, merged: (pre.grid.1 / 2, pre.grid.2 / 2), features: Arc::new(()) })
    }
}

fn http_server(with_vision: bool) -> (common::Server, Seen) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let vision = with_vision.then(|| {
        let slot = Arc::new(OnceLock::new());
        slot.set(Arc::new(VisionService::new(Box::new(MockTower { seen: seen.clone() }), 3328))).unwrap();
        VisionHandle {
            slot,
            limits: Limits { max_tokens_per_image: 3328, ..Limits::default() },
            pp: Arc::new(NemotronPreprocessing::default()),
            gate: PreparationGate::new(Default::default()).unwrap(),
            ingress: HttpsIngress::new(IngressPolicy::default()).unwrap(),
        }
    });
    let s = start(Opts { dir: Some(model_dir()), vision, ..Default::default() });
    (s, seen)
}

fn post(port: u16, body: &Value) -> common::Reply {
    common::http_timeout(port, "POST", "/v1/chat/completions", Some(body), Duration::from_secs(60))
}

#[test]
fn over_http_two_images_reach_the_encoder_and_the_prompt_has_the_expanded_length() {
    let (s, seen) = http_server(true);
    assert_eq!(s.state.format.id(), "nemotron_omni");
    let messages = two_image_messages();
    let body = json!({"messages": messages, "max_tokens": 2, "chat_template_kwargs": {"enable_thinking": false}});
    let r = post(s.port, &body);
    assert_eq!(r.status, 200, "{}", r.body);
    // the encoder saw the bf16 canvases of the two tiles, in order
    assert_eq!(*seen.lock().unwrap(), [((512, 512), (1, 32, 32), 3 * 512 * 512), ((608, 480), (1, 30, 38), 3 * 608 * 480)]);
    // the prompt the engine got is the rendered text with both runs expanded
    let mut kwargs = Map::new();
    kwargs.insert("enable_thinking".into(), json!(false));
    let rendered = s.state.format.render(&messages, &kwargs).unwrap().text;
    let want = s.state.tok.encode(&expand_text(&rendered, &[256, 285]).unwrap()).unwrap().len();
    assert_eq!(r.json()["usage"]["prompt_tokens"], want);

    // a text-only request through the same server is the plain rendered text
    let plain = json!({"messages": [{"role": "user", "content": "a b a"}], "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": false}});
    let r = post(s.port, &plain);
    assert_eq!(r.status, 200, "{}", r.body);
    let want = s.state.tok.encode(&s.state.format.render(&plain["messages"], &kwargs).unwrap().text).unwrap().len();
    assert_eq!(r.json()["usage"]["prompt_tokens"], want);
    assert_eq!(seen.lock().unwrap().len(), 2, "no encode for a text request");
}

#[test]
fn over_http_without_vision_images_are_refused_and_video_is_named() {
    let (s, seen) = http_server(false);
    let body = json!({"messages": two_image_messages(), "max_tokens": 1});
    let r = post(s.port, &body);
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(r.body.contains("messages[0].content[1]") && r.body.contains("image support"), "{}", r.body);
    let (s2, _) = http_server(true);
    let video = json!({"messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}]}], "max_tokens": 1});
    let r = post(s2.port, &video);
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(r.body.contains("video_url") && r.body.contains("not supported"), "{}", r.body);
    assert!(seen.lock().unwrap().is_empty());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// WebP

fn webp_bytes(name: &str) -> Vec<u8> {
    std::fs::read(testdata().join("vision/webp").join(name)).unwrap()
}

fn uri(mime: &str, bytes: &[u8]) -> String {
    format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes))
}

fn one_image(url: String) -> Value {
    json!([{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": url}}]}])
}

/// A PNG (8-bit RGB) of these pixels.
fn png_of(img: &gb_vision::Rgb) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut e = png::Encoder::new(&mut out, img.width as u32, img.height as u32);
        e.set_color(png::ColorType::Rgb);
        e.set_depth(png::BitDepth::Eight);
        e.write_header().unwrap().write_image_data(&img.data).unwrap();
    }
    out
}

#[test]
fn a_webp_image_is_ingested_like_a_png_of_its_pixels_whatever_the_declared_type() {
    let (f, tok) = (omni(), tok());
    let p = prep(Limits::default());
    for name in
        ["lossy_q75_97x61.webp", "lossy_alpha_q80_63x65.webp", "lossless_alpha_exact_63x65.webp", "lossless_exif6_37x21.webp"]
    {
        let webp = webp_bytes(name);
        let png = png_of(&NemotronPreprocessing::default().decode(&webp).unwrap());
        let ingest = |url: String| {
            let msgs = one_image(url);
            p.ingest(&f, &tok, &msgs, IngestOptions::default(), |r| data_uri(r, p.limits()))
                .unwrap_or_else(|e| panic!("{name}: {e:?}"))
        };
        let want = ingest(uri("image/png", &png));
        for mime in ["image/webp", "image/png", "application/octet-stream"] {
            let got = ingest(uri(mime, &webp));
            assert_eq!(got.ids, want.ids, "{name} as {mime}");
            let (a, b) = (&got.prepared[0], &want.prepared[0]);
            assert_eq!(
                (a.source, a.canvas, a.tokens, a.pixel_hash, a.key),
                (b.source, b.canvas, b.tokens, b.pixel_hash, b.key),
                "{name}"
            );
            assert_eq!(a.pixels, b.pixels, "{name}: the tower's input");
        }
    }
}

#[test]
fn other_formats_and_animated_webp_are_400s_that_name_the_formats() {
    let (f, tok) = (omni(), tok());
    let p = prep(Limits::default());
    let gif = b"GIF89a\x01\0\x01\0\0\0\0;";
    let e = p
        .ingest(&f, &tok, &one_image(uri("image/gif", gif)), IngestOptions::default(), |r| data_uri(r, p.limits()))
        .err()
        .unwrap();
    assert_eq!(e.status, 400);
    assert!(
        e.message.contains("messages[0].content[1]") && e.message.contains("only PNG, JPEG and WebP images are supported"),
        "{}",
        e.message
    );
    let anim = webp_bytes("animated_2f_8x8.webp");
    let e = p
        .ingest(&f, &tok, &one_image(uri("image/webp", &anim)), IngestOptions::default(), |r| data_uri(r, p.limits()))
        .err()
        .unwrap();
    assert_eq!(e.status, 400);
    assert!(e.message.contains("animated images are not supported"), "{}", e.message);
}

#[test]
fn over_http_a_webp_image_reaches_the_encoder() {
    let (s, seen) = http_server(true);
    let body = json!({"messages": one_image(uri("image/webp", &webp_bytes("lossy_q75_97x61.webp"))), "max_tokens": 1,
        "chat_template_kwargs": {"enable_thinking": false}});
    let r = post(s.port, &body);
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(seen.lock().unwrap().len(), 1);
    let gif = json!({"messages": one_image(uri("image/gif", b"GIF89a\x01\0\x01\0\0\0\0;")), "max_tokens": 1});
    let r = post(s.port, &gif);
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(r.body.contains("PNG, JPEG and WebP"), "{}", r.body);
}

/// The repository root: `REXL3_REPO_ROOT` when set (a prebuilt test binary run from an installed bin directory that carries the
/// tree's `testdata/`, e.g. on a node without this checkout), else the checkout this binary was built from.
fn repo_root() -> std::path::PathBuf {
    std::env::var_os("REXL3_REPO_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}
