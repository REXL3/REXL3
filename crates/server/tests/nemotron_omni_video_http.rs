//! The video path of the Nemotron format over HTTP on the mock engine with a mock encoder where the tower goes (milestone M4): a request with a
//! video part is routed by `http.rs` to the format's video ingress, the frames reach the encoder as tubelets, the pruning mask is computed on
//! the host from the encoder's features, the prompt the engine gets is the processor's text with each tubelet's kept rows, and every refusal
//! names the part. No model, no GPU.
mod common;

use base64::Engine;
use common::{start, Opts};
use gb_model::error::ModelError;
use gb_model::serve::{EncodedImage, ImageEncoder, KeepRows, VideoInput, VisionService};
use gb_server::chat_format::{ChatFormat, VideoOptions};
use gb_server::formats::nemotron_omni::video::expand_video_text;
use gb_server::formats::nemotron_omni::OmniFormat;
use gb_server::http::VisionHandle;
use gb_server::preparation::PreparationGate;
use gb_server::vision::Limits;
use gb_server::vision_ingress::{HttpsIngress, IngressPolicy};
use gb_vision::nemotron::NemotronPreprocessing;
use gb_vision::Rgb;
use serde_json::{json, Map, Value};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

fn testdata() -> PathBuf {
    repo_root().join("testdata")
}

/// A model folder of the Nemotron format with a small synthetic tokenizer (whitespace words, the chat and image tokens as special added tokens).
fn model_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("rexl3-nemotron-omni-video-http-{}", std::process::id()));
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

fn png_uri(rgb: &Rgb) -> String {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, rgb.width as u32, rgb.height as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().unwrap();
        w.write_image_data(&rgb.data).unwrap();
    }
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(out))
}

fn flat_frames(n: usize, w: usize, h: usize) -> Vec<String> {
    (0..n)
        .map(|i| png_uri(&Rgb { width: w, height: h, data: (0..w * h * 3).map(|k| ((k * 7 + i * 31) % 251) as u8).collect() }))
        .collect()
}

fn video_request(frames: &[String], question: &str) -> Value {
    json!({
        "messages": [{"role": "user", "content": [{"type": "text", "text": question}, {"type": "video_url", "video_url": {"url": frames}}]}],
        "max_tokens": 2,
        "chat_template_kwargs": {"enable_thinking": false}
    })
}

fn text_request(text: &str) -> Value {
    json!({"messages": [{"role": "user", "content": text}], "max_tokens": 2, "chat_template_kwargs": {"enable_thinking": false}})
}

/// What the mock tower saw of one video.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Seen {
    frames: usize,
    canvas: (usize, usize),
    pruning: bool,
    kept: Vec<usize>,
}

enum Mode {
    /// Every row kept (or the host mask applied to identical features: the first tubelet whole, then the tie rule).
    Plain,
    /// The encoder reports the memory for features in use.
    Busy,
}

struct MockTower {
    seen: Arc<Mutex<Vec<Seen>>>,
    mode: Mode,
}

impl ImageEncoder for MockTower {
    fn encode(&mut self, _pre: &gb_vision::Preprocessed) -> Result<EncodedImage, ModelError> {
        Err(ModelError::NotBuilt("the video tests encode no image".into()))
    }

    fn encode_video(&mut self, video: &VideoInput<'_>, keep: Option<KeepRows<'_>>) -> Result<Vec<EncodedImage>, ModelError> {
        if matches!(self.mode, Mode::Busy) {
            return Err(ModelError::Busy("the memory for image features is in use (mock)".into()));
        }
        let (w, h) = video.canvas;
        let tokens = (w / 16) * (h / 16) / 4;
        let merged = (h / 32, w / 32);
        let tubelets = video.frames.len().div_ceil(2);
        let mut kept = vec![tokens; tubelets];
        if let Some(keep) = keep {
            // identical features in every tubelet: every score is 0 but the first tubelet's (255), so the host function keeps the first
            // tubelet whole and fills the rest by the lowest flat index
            let features = vec![0x3f80u16; tubelets * tokens * 4096];
            let mask = keep(&features).map_err(ModelError::Request)?;
            assert_eq!(mask.len(), tubelets * tokens);
            kept = mask.chunks(tokens).map(|c| c.iter().filter(|&&k| k).count()).collect();
        }
        self.seen.lock().unwrap().push(Seen {
            frames: video.frames.len(),
            canvas: video.canvas,
            pruning: keep.is_some(),
            kept: kept.clone(),
        });
        Ok(kept
            .into_iter()
            .map(|k| EncodedImage { tokens: k, merged: if k == tokens { merged } else { (1, k) }, features: Arc::new(()) })
            .collect())
    }
}

fn omni(options: Option<VideoOptions>) -> OmniFormat {
    let t = std::fs::read_to_string(testdata().join("nemotron_omni/chat_template.jinja")).unwrap();
    let mut f = OmniFormat::new(&t).unwrap().with_preprocessing(NemotronPreprocessing::default());
    if let Some(o) = options {
        f.enable_video(&o).unwrap();
    }
    f
}

fn server(format: OmniFormat, vision: bool, mode: Mode) -> (common::Server, Arc<Mutex<Vec<Seen>>>) {
    server_cap(format, vision, mode, 3328)
}

/// [`server`] with `--max-image-tokens` of `cap`.
fn server_cap(format: OmniFormat, vision: bool, mode: Mode, cap: usize) -> (common::Server, Arc<Mutex<Vec<Seen>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let vision = vision.then(|| {
        let slot = Arc::new(OnceLock::new());
        slot.set(Arc::new(VisionService::new(Box::new(MockTower { seen: seen.clone(), mode }), 3328))).unwrap();
        VisionHandle {
            slot,
            limits: Limits { max_tokens_per_image: cap, ..Limits::default() },
            pp: Arc::new(NemotronPreprocessing::default()),
            gate: PreparationGate::new(Default::default()).unwrap(),
            ingress: HttpsIngress::new(IngressPolicy::default()).unwrap(),
        }
    });
    let f: Arc<dyn ChatFormat> = Arc::new(format);
    let s = start(Opts { dir: Some(model_dir()), vision, format: Some(f), ..Default::default() });
    (s, seen)
}

fn post(port: u16, body: &Value) -> common::Reply {
    common::http_timeout(port, "POST", "/v1/chat/completions", Some(body), Duration::from_secs(60))
}

fn prompt_tokens(r: &common::Reply) -> u64 {
    r.json()["usage"]["prompt_tokens"].as_u64().unwrap()
}

/// The tokens of the prompt `format` makes of `request`'s messages with `counts` rows in each tubelet (the processor's text).
fn want_tokens(s: &common::Server, request: &Value, tubelets: usize, counts: &[usize]) -> u64 {
    let mut kwargs = Map::new();
    kwargs.insert("enable_thinking".into(), json!(false));
    let rendered = s.state.format.render(&request["messages"], &kwargs).unwrap().text;
    // the labels: "Frame 1 and frame 2: ", "Frame 3 and frame 4: ", ...
    let labels: Vec<String> = (0..tubelets).map(|g| format!("Frame {} and frame {}: ", 2 * g + 1, 2 * g + 2)).collect();
    s.state.tok.encode(&expand_video_text(&rendered, &labels, counts).unwrap()).unwrap().len() as u64
}

#[test]
fn over_http_a_video_reaches_the_encoder_as_tubelets_and_the_prompt_is_the_processors() {
    let (s, seen) = server(omni(Some(VideoOptions::default())), true, Mode::Plain);
    assert!(s.state.format.video().is_some());
    // 4 frames of 512 x 512: two tubelets of 256 rows
    let req = video_request(&flat_frames(4, 512, 512), "What happens in this clip?");
    let r = post(s.port, &req);
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(*seen.lock().unwrap(), [Seen { frames: 4, canvas: (512, 512), pruning: false, kept: vec![256, 256] }]);
    assert_eq!(prompt_tokens(&r), want_tokens(&s, &req, 2, &[256, 256]));
    // an odd count: the last tubelet is the last frame twice, and the third frame's label stands alone
    let req = video_request(&flat_frames(3, 512, 512), "And now?");
    let r = post(s.port, &req);
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(seen.lock().unwrap()[1].frames, 3);
    let mut kwargs = Map::new();
    kwargs.insert("enable_thinking".into(), json!(false));
    let rendered = s.state.format.render(&req["messages"], &kwargs).unwrap().text;
    let labels = vec!["Frame 1 and frame 2: ".to_string(), "Frame 3: ".to_string()];
    let want = s.state.tok.encode(&expand_video_text(&rendered, &labels, &[256, 256]).unwrap()).unwrap().len() as u64;
    assert_eq!(prompt_tokens(&r), want);
}

#[test]
fn over_http_a_text_prompt_before_and_after_a_video_is_the_same_prompt() {
    let (s, seen) = server(omni(Some(VideoOptions::default())), true, Mode::Plain);
    let t = text_request("a b a b");
    let before = post(s.port, &t);
    assert_eq!(before.status, 200, "{}", before.body);
    let v = post(s.port, &video_request(&flat_frames(2, 512, 512), "a"));
    assert_eq!(v.status, 200, "{}", v.body);
    let after = post(s.port, &t);
    assert_eq!(after.status, 200, "{}", after.body);
    assert_eq!(prompt_tokens(&before), prompt_tokens(&after));
    assert_eq!(before.json()["choices"][0]["message"], after.json()["choices"][0]["message"]);
    assert_eq!(seen.lock().unwrap().len(), 1, "the text requests did not touch the encoder");
}

#[test]
fn over_http_pruning_is_decided_on_the_host_and_a_tubelet_that_keeps_nothing_has_no_run() {
    let opts = VideoOptions { pruning_rate: 0.5, ..VideoOptions::default() };
    let (s, seen) = server(omni(Some(opts)), true, Mode::Plain);
    // two tubelets of 256: keep = max(256, int(2 * 256 * 0.5)) = 256, all of them the first tubelet's
    let req = video_request(&flat_frames(4, 512, 512), "What moves?");
    let r = post(s.port, &req);
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(*seen.lock().unwrap(), [Seen { frames: 4, canvas: (512, 512), pruning: true, kept: vec![256, 0] }]);
    assert_eq!(prompt_tokens(&r), want_tokens(&s, &req, 2, &[256, 0]));
    // three tubelets: keep = int(3 * 256 * 0.5) = 384: the first whole, 128 of the identical rest by the lowest index (tubelet 1)
    let req = video_request(&flat_frames(6, 512, 512), "What moves?");
    let r = post(s.port, &req);
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(seen.lock().unwrap()[1].kept, vec![256, 128, 0]);
    assert_eq!(prompt_tokens(&r), want_tokens(&s, &req, 3, &[256, 128, 0]));
    // one tubelet is never pruned (the model's own rule)
    let req = video_request(&flat_frames(2, 512, 512), "What moves?");
    let r = post(s.port, &req);
    assert_eq!(r.status, 200, "{}", r.body);
    let last = seen.lock().unwrap().last().cloned().unwrap();
    assert_eq!((last.pruning, last.kept), (false, vec![256]));
}

#[test]
fn over_http_strict_square_refuses_a_tile_the_reference_cannot_prune() {
    let opts = VideoOptions { pruning_rate: 0.7, require_square_grid: true, ..VideoOptions::default() };
    let (s, seen) = server(omni(Some(opts)), true, Mode::Plain);
    // a 16:9 frame is a 42 x 24 patch tile: 252 tokens a tubelet, not a perfect square
    let r = post(s.port, &video_request(&flat_frames(4, 640, 360), "x"));
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(r.body.contains("pruning is on") && r.body.contains("252 tokens"), "{}", r.body);
    // a square frame is served
    let r = post(s.port, &video_request(&flat_frames(4, 512, 512), "x"));
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[test]
fn over_http_a_non_square_tile_is_pruned_on_its_true_grid_by_default() {
    // the reference raises for a tile whose tokens are not a perfect square (252 here); the default policy computes the mask on the tile's own
    // 12 x 21 grid, which has no reference run: three tubelets at 0.5 keep int(3 * 252 * 0.5) = 378: the first whole and 126 of the rest
    let opts = VideoOptions { pruning_rate: 0.5, ..VideoOptions::default() };
    let (s, seen) = server(omni(Some(opts)), true, Mode::Plain);
    let req = video_request(&flat_frames(6, 640, 360), "x");
    let r = post(s.port, &req);
    assert_eq!(r.status, 200, "{}", r.body);
    // the mock's identical features: tubelet 0 whole, then the lowest flat indices
    assert_eq!(*seen.lock().unwrap(), [Seen { frames: 6, canvas: (672, 384), pruning: true, kept: vec![252, 126, 0] }]);
    assert_eq!(prompt_tokens(&r), want_tokens(&s, &req, 3, &[252, 126, 0]));
}

#[test]
fn over_http_the_refusals_name_the_part_and_the_flag() {
    // vision off: a video part names the vision tower
    let (s, seen) = server(omni(Some(VideoOptions::default())), false, Mode::Plain);
    let r = post(s.port, &video_request(&flat_frames(2, 64, 64), "x"));
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(r.body.contains("messages[0].content[1]") && r.body.contains("vision tower"), "{}", r.body);
    assert!(seen.lock().unwrap().is_empty());
    // video off in the format (a server that did not enable it): refused as before, by name
    let (s, _) = server(omni(None), true, Mode::Plain);
    let r = post(s.port, &video_request(&flat_frames(2, 64, 64), "x"));
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(r.body.contains("video_url") && r.body.contains("not supported"), "{}", r.body);
    // the frame and token budgets
    let opts = VideoOptions { max_frames: 4, max_tokens: 500, ..VideoOptions::default() };
    let (s, seen) = server(omni(Some(opts)), true, Mode::Plain);
    let r = post(s.port, &video_request(&flat_frames(5, 512, 512), "x"));
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(r.body.contains("5 frames") && r.body.contains("--max-video-frames"), "{}", r.body);
    let r = post(s.port, &video_request(&flat_frames(4, 512, 512), "x"));
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(r.body.contains("512 tokens") && r.body.contains("--max-video-tokens"), "{}", r.body);
    // an image beside the video
    let both = json!({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": flat_frames(1, 64, 64)[0]}},
        {"type": "video_url", "video_url": {"url": flat_frames(2, 512, 512)}}]}], "max_tokens": 1});
    let r = post(s.port, &both);
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(r.body.contains("an image and a video"), "{}", r.body);
    assert!(seen.lock().unwrap().is_empty(), "no refusal reached the encoder");
    // a container
    let mp4 = json!({"messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}]}], "max_tokens": 1});
    let r = post(s.port, &mp4);
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(r.body.contains("mp4") && r.body.contains("messages[0].content[0]"), "{}", r.body);
    // a tile the image workspace cannot stage twice: a 256-token tile needs --max-image-tokens 512 (the tubelet's two frames' patches)
    let (s, seen) = server_cap(omni(Some(VideoOptions::default())), true, Mode::Plain, 400);
    let r = post(s.port, &video_request(&flat_frames(2, 512, 512), "x"));
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(r.body.contains("--max-image-tokens must be at least 512") && r.body.contains("it is 400"), "{}", r.body);
    assert!(seen.lock().unwrap().is_empty());
    let (s, seen) = server_cap(omni(Some(VideoOptions::default())), true, Mode::Plain, 512);
    let r = post(s.port, &video_request(&flat_frames(2, 512, 512), "x"));
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[test]
fn over_http_a_busy_encoder_is_a_503_with_retry_after() {
    let (s, _) = server(omni(Some(VideoOptions::default())), true, Mode::Busy);
    let r = post(s.port, &video_request(&flat_frames(2, 512, 512), "x"));
    assert_eq!(r.status, 503, "{}", r.body);
    assert!(r.header("retry-after").is_some(), "{:?}", r.headers);
    assert!(r.body.contains("messages[0].content[1]"), "{}", r.body);
}

/// The repository root: `REXL3_REPO_ROOT` when set (a prebuilt test binary run from an installed bin directory that carries the
/// tree's `testdata/`, e.g. on a node without this checkout), else the checkout this binary was built from.
fn repo_root() -> std::path::PathBuf {
    std::env::var_os("REXL3_REPO_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}
