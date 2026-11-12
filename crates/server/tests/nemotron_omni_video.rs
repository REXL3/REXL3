//! The video ingress of the Nemotron format without a model: the processor's expanded texts byte for byte
//! (`testdata/nemotron_omni/video_pre/videos.json`), the ids, spans and keys of a request, what the pruning leaves of the prompt (the model's own
//! drop of the placeholder rows), the budgets and every refusal by name. No GPU: the tower is a mock behind `NemotronVideoEncoder`.
//!
//! `REXL3_NEMO_OMNI_DIR` (the pack folder) adds the check against the real tokenizer: the processor's `input_ids` of every prompt, pruned too.
use base64::Engine;
use gb_model::error::ModelError;
use gb_server::api::ApiError;
use gb_server::chat_format::{ChatFormat, FormatRegistry};
use gb_server::formats::nemotron_omni::prep::{
    data_uri, DeferredTubelet, EncodedTubelet, EncodedVideo, IngestOptions, KeepRows, NemotronImagePrep, NemotronVideoEncoder,
    PreparedVideo,
};
use gb_server::formats::nemotron_omni::video::{expand_video_text, VideoLimits, VideoSettings};
use gb_server::formats::nemotron_omni::{expand_text, OmniFormat};
use gb_server::tokenizer::Tok;
use gb_server::vision::{ImageRef, Limits};
use gb_vision::nemotron::{f32_to_bf16_bits, NemotronPreprocessing};
use gb_vision::nemotron_video::{frame_labels, kept_per_tubelet, VideoConfig, VideoMetadata};
use gb_vision::rgb_decode::{decode_rgb, AlphaPolicy};
use gb_vision::Rgb;
use serde_json::{json, Map, Value};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

fn testdata() -> PathBuf {
    repo_root().join("testdata")
}

fn fixture(name: &str) -> Value {
    let p = testdata().join("nemotron_omni/video_pre").join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}

fn u(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

/// A model folder of the Nemotron format with a small synthetic tokenizer (whitespace words, the chat and image tokens as special added tokens).
fn model_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("rexl3-nemotron-omni-video-{}", std::process::id()));
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

fn tok() -> Tok {
    Tok::from_dir(&model_dir()).unwrap()
}

fn omni() -> OmniFormat {
    OmniFormat::new(&std::fs::read_to_string(testdata().join("nemotron_omni/chat_template.jinja")).unwrap())
        .unwrap()
        .with_preprocessing(NemotronPreprocessing::default())
}

fn omni_video(settings: VideoSettings) -> OmniFormat {
    omni().with_video(settings).unwrap()
}

fn prep_for(f: &OmniFormat, limits: Limits) -> NemotronImagePrep {
    NemotronImagePrep::from_format(f, limits).unwrap()
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// frames

fn crop(img: &Rgb, x: usize, y: usize, w: usize, h: usize) -> Rgb {
    let mut data = Vec::with_capacity(w * h * 3);
    for row in y..y + h {
        data.extend_from_slice(&img.data[(row * img.width + x) * 3..(row * img.width + x + w) * 3]);
    }
    Rgb { width: w, height: h, data }
}

fn source(file: &str) -> Rgb {
    decode_rgb(&std::fs::read(testdata().join("vision").join(file)).unwrap(), AlphaPolicy::Drop).unwrap()
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

/// The frames of a fixture video (`file`, the crops), as data URIs.
fn fixture_frames(v: &Value) -> Vec<String> {
    let img = source(v["file"].as_str().unwrap());
    v["frame_records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            let c = &f["crop"];
            png_uri(&crop(&img, u(&c[0]), u(&c[1]), u(&c[2]), u(&c[3])))
        })
        .collect()
}

/// `n` flat frames of one size and colour, as data URIs (cheap: tiny PNGs).
fn flat_frames(n: usize, w: usize, h: usize) -> Vec<String> {
    (0..n)
        .map(|i| png_uri(&Rgb { width: w, height: h, data: (0..w * h * 3).map(|k| ((k * 7 + i * 31) % 251) as u8).collect() }))
        .collect()
}

fn video_part(frames: &[String]) -> Value {
    json!({"type": "video_url", "video_url": {"url": frames}})
}

fn user(parts: Vec<Value>) -> Value {
    json!([{"role": "user", "content": parts}])
}

fn question_and_video(frames: &[String]) -> Value {
    user(vec![json!({"type": "text", "text": "What happens in this clip?"}), video_part(frames)])
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// the seeded recipe of `tools/nemotron_omni/gen_video_fixture.py` (as `crates/vision/tests/nemotron_video.rs` mirrors it)

fn splitmix(x: u64) -> u64 {
    let x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn stream(seed: u64, tag: u64, idx: u64) -> u64 {
    splitmix(seed.wrapping_mul(0xD1B5_4A32_D192_ED03).wrapping_add(tag.wrapping_mul(0x8CB9_2BA7_2F3D_8DD7)).wrapping_add(idx))
}

fn gaussish(seed: u64, tag: u64, idx: u64) -> f32 {
    let mut v = 0i64;
    for k in 0..3 {
        v += (stream(seed, 3 * tag + k, idx) >> 42) as i64;
    }
    (v - 3 * (1 << 21)) as f32 * (1.0 / 2_097_152.0)
}

fn unit(seed: u64, tag: u64, idx: u64) -> f32 {
    (stream(seed, tag, idx) >> 40) as f32 * (1.0 / 16_777_216.0)
}

fn recipe_bits(r: &Value) -> Vec<u16> {
    let (seed, t, n, hidden) = (r["seed"].as_u64().unwrap(), u(&r["t"]), u(&r["n"]), u(&r["hidden"]));
    let kind = r["kind"].as_str().unwrap();
    let (amp, p) = (r["amp"].as_f64().unwrap() as f32, r["p"].as_f64().unwrap() as f32);
    let mut cur: Vec<f32> = (0..n * hidden).map(|k| gaussish(seed, 0, k as u64)).collect();
    let mut x = cur.clone();
    for g in 1..t {
        if kind == "indep" {
            cur = (0..n * hidden).map(|k| gaussish(seed, g as u64, k as u64)).collect();
        } else {
            let mut nxt = vec![0f32; n * hidden];
            for i in 0..n {
                let scale = unit(seed, 1000 + g as u64, i as u64) * amp;
                let copy = kind == "dup" && unit(seed, 2000 + g as u64, i as u64) < p;
                for k in 0..hidden {
                    let noise = gaussish(seed, g as u64, (i * hidden + k) as u64);
                    nxt[i * hidden + k] = if copy { cur[i * hidden + k] } else { cur[i * hidden + k] + scale * noise };
                }
            }
            cur = nxt;
        }
        x.extend_from_slice(&cur);
    }
    if r.get("mean").and_then(Value::as_bool).unwrap_or(false) {
        let c: Vec<f32> = (0..hidden).map(|k| 2.0 * gaussish(seed, 500, k as u64)).collect();
        for row in x.chunks_exact_mut(hidden) {
            for (v, c) in row.iter_mut().zip(&c) {
                *v += c;
            }
        }
    }
    x.iter().map(|&v| f32_to_bf16_bits(v)).collect()
}

fn mask_from_hex(hex: &str, n: usize) -> Vec<bool> {
    let bytes: Vec<u8> = (0..hex.len() / 2).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect();
    (0..n).map(|i| bytes[i / 8] >> (i % 8) & 1 == 1).collect()
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// the mock tower

struct MockTower {
    /// Host features of the whole video (tubelet-major bf16 rows of 4096); generated from the key when `None`.
    features: Option<Vec<u16>>,
    calls: Mutex<Vec<(usize, bool)>>,
    fail: Option<fn() -> ModelError>,
}

impl MockTower {
    fn new() -> Self {
        Self { features: None, calls: Mutex::new(Vec::new()), fail: None }
    }
}

impl NemotronVideoEncoder for MockTower {
    fn encode_video(&self, video: &PreparedVideo, keep: Option<KeepRows<'_>>) -> Result<EncodedVideo, ModelError> {
        self.calls.lock().unwrap().push((video.tubelets(), keep.is_some()));
        if let Some(f) = self.fail {
            return Err(f());
        }
        let (g, n) = (video.tubelets(), video.plan.tokens_per_tubelet);
        let feats = self
            .features
            .clone()
            .unwrap_or_else(|| (0..g * n * 4096).map(|i| f32_to_bf16_bits(gaussish(video.key, 7, i as u64))).collect());
        assert_eq!(feats.len(), g * n * 4096);
        let mask = match keep {
            Some(k) => Some(k(&feats).map_err(ModelError::Request)?),
            None => None,
        };
        let tubelets = (0..g)
            .map(|t| {
                let rows: Vec<u16> = (0..n)
                    .filter(|&i| mask.as_ref().is_none_or(|m| m[t * n + i]))
                    .flat_map(|i| feats[(t * n + i) * 4096..(t * n + i + 1) * 4096].iter().copied())
                    .collect();
                EncodedTubelet { tokens: rows.len() / 4096, features: Arc::new(rows) }
            })
            .collect();
        Ok(EncodedVideo { tubelets })
    }
}

fn ingest(
    p: &NemotronImagePrep,
    f: &OmniFormat,
    messages: &Value,
    tower: Option<&MockTower>,
) -> Result<gb_server::formats::nemotron_omni::prep::Ingested, ApiError> {
    let t = tok();
    p.ingest(
        f,
        &t,
        messages,
        IngestOptions { video_encoder: tower.map(|t| t as &dyn NemotronVideoEncoder), ..Default::default() },
        |r: &ImageRef| data_uri(r, p.limits()),
    )
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// the processor's texts

#[test]
fn the_expanded_texts_are_the_processors_byte_for_byte_for_every_metadata_variant() {
    let fx = fixture("videos.json");
    let f = omni();
    let tok = tok();
    let mut checked = 0;
    for v in fx["videos"].as_array().unwrap() {
        let (frames, tokens, tubelets) = (u(&v["frames"]), u(&v["tokens_per_tubelet"]), u(&v["tubelets"]));
        for t in v["texts"].as_array().unwrap() {
            let name = format!("{} / {}", v["name"], t["name"]);
            // the template, through our renderer, against the processor's `apply_chat_template`
            let rendered = f.render(&t["messages"], &Map::new()).unwrap().text;
            assert_eq!(rendered, t["rendered"].as_str().unwrap(), "{name}: the rendered prompt");
            let meta = match t["metadata"].as_object() {
                None => VideoMetadata::default(),
                Some(m) => VideoMetadata {
                    fps: m["fps"].as_f64(),
                    frame_indices: m["frame_indices"].as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap()).collect()),
                },
            };
            let labels = frame_labels(frames, &meta).unwrap();
            let expanded = expand_video_text(&rendered, &labels, &vec![tokens; tubelets]).unwrap();
            assert_eq!(expanded, t["expanded"].as_str().unwrap(), "{name}: the text the tokenizer is handed");
            // the one-placeholder rewrite and the id expansion give the ids of that text
            let single =
                rendered.replacen("<video>", &gb_server::formats::nemotron_omni::video::one_placeholder_per_tubelet(&labels), 1);
            let (ids, spans) = f.expand_images(&tok, &tok.encode(&single).unwrap(), &vec![tokens; tubelets]).unwrap();
            assert_eq!(ids, tok.encode(&expanded).unwrap(), "{name}: ids of the text and the id-level expansion");
            let runs: Vec<(usize, usize)> = t["image_runs"].as_array().unwrap().iter().map(|r| (u(&r[0]), u(&r[1]))).collect();
            // the mini tokenizer has other ids for the words, so the run LENGTHS are the fixture's and the starts are the run starts of its own ids
            assert_eq!(
                spans.iter().map(|s| s.tokens).collect::<Vec<_>>(),
                runs.iter().map(|r| r.1).collect::<Vec<_>>(),
                "{name}"
            );
            checked += 1;
        }
    }
    assert!(checked >= 8);
}

#[test]
fn the_second_video_tag_stays_literal_in_the_processor_and_is_refused_here() {
    let fx = fixture("videos.json");
    let two = &fx["second_tag_stays"];
    let f = omni();
    let rendered = f.render(&two["messages"], &Map::new()).unwrap().text;
    assert_eq!(rendered.matches("<video>").count(), 2);
    // the processor replaces the first tag only
    let labels = frame_labels(2, &VideoMetadata::default()).unwrap();
    let want = two["expanded"].as_str().unwrap();
    let wide = fx["videos"].as_array().unwrap().iter().find(|v| v["name"] == "wide_2frames").unwrap();
    let got = expand_video_text(&rendered, &labels, &[u(&wide["tokens_per_tubelet"])]).unwrap();
    assert_eq!(got, want);
    assert_eq!(got.matches("<video>").count(), 1);
    // and a request that would have two tags is refused by name
    let f = omni_video(VideoSettings::default());
    let p = prep_for(&f, Limits::default());
    let msgs = user(vec![json!({"type": "text", "text": "a <video> b <video> c"}), video_part(&flat_frames(2, 512, 512))]);
    let e = ingest(&p, &f, &msgs, None).err().unwrap();
    assert_eq!(e.status, 400);
    assert!(e.message.contains("2 <video> placeholders but one video"), "{}", e.message);
}

fn real_tokenizer() -> Option<Tok> {
    std::env::var_os("REXL3_NEMO_OMNI_DIR")
        .map(PathBuf::from)
        .filter(|d| d.join("tokenizer.json").is_file())
        .map(|d| Tok::from_dir(&d).unwrap())
}

#[test]
fn the_real_tokenizer_gives_the_processors_input_ids_pruned_and_not() {
    let Some(tok) = real_tokenizer() else {
        eprintln!("REXL3_NEMO_OMNI_DIR is not set: the real-tokenizer check is skipped");
        return;
    };
    let fx = fixture("videos.json");
    let f = omni();
    let crc = |ids: &[u32]| crc32fast::hash(&ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>());
    for v in fx["videos"].as_array().unwrap() {
        let (frames, tokens, tubelets) = (u(&v["frames"]), u(&v["tokens_per_tubelet"]), u(&v["tubelets"]));
        for t in v["texts"].as_array().unwrap() {
            let name = format!("{} / {}", v["name"], t["name"]);
            let rendered = t["rendered"].as_str().unwrap();
            let meta = match t["metadata"].as_object() {
                None => VideoMetadata::default(),
                Some(m) => VideoMetadata {
                    fps: m["fps"].as_f64(),
                    frame_indices: m["frame_indices"].as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap()).collect()),
                },
            };
            let labels = frame_labels(frames, &meta).unwrap();
            let single =
                rendered.replacen("<video>", &gb_server::formats::nemotron_omni::video::one_placeholder_per_tubelet(&labels), 1);
            let (ids, spans) = f.expand_images(&tok, &tok.encode(&single).unwrap(), &vec![tokens; tubelets]).unwrap();
            assert_eq!(
                (ids.len(), crc(&ids)),
                (u(&t["n_input_ids"]), t["input_ids_crc32"].as_u64().unwrap() as u32),
                "{name}: input_ids"
            );
            let runs: Vec<(usize, usize)> = t["image_runs"].as_array().unwrap().iter().map(|r| (u(&r[0]), u(&r[1]))).collect();
            assert_eq!(spans.iter().map(|s| (s.start, s.tokens)).collect::<Vec<_>>(), runs, "{name}: spans");
            assert_eq!(ids, tok.encode(t["expanded"].as_str().unwrap()).unwrap(), "{name}");
        }
    }
    assert_eq!((tok.token_to_id("<image>"), tok.token_to_id("<img>"), tok.token_to_id("</img>")), (Some(18), Some(19), Some(20)));
    // the model's drop of the placeholder rows: the ids of the text with each tubelet's kept count are the ids the processor's input_ids keep
    let sq = fx["videos"].as_array().unwrap().iter().find(|v| v["name"] == "square_5frames").unwrap();
    let plain = sq["texts"].as_array().unwrap().iter().find(|t| t["name"] == "plain").unwrap();
    for p in fx["pruned_square"].as_array().unwrap() {
        let kept: Vec<usize> = p["kept_per_tubelet"].as_array().unwrap().iter().map(u).collect();
        let labels = frame_labels(5, &VideoMetadata::default()).unwrap();
        let text = expand_video_text(plain["rendered"].as_str().unwrap(), &labels, &kept).unwrap();
        assert_eq!(text, p["pruned_text"].as_str().unwrap());
        let ids = tok.encode(&text).unwrap();
        assert_eq!(
            (ids.len(), crc(&ids)),
            (u(&p["n_pruned_ids"]), p["pruned_ids_crc32"].as_u64().unwrap() as u32),
            "rate {}",
            p["rate"]
        );
        // and by the ids: the full expansion with the placeholders of each run cut to the kept count
        let single = plain["rendered"].as_str().unwrap().replacen(
            "<video>",
            &gb_server::formats::nemotron_omni::video::one_placeholder_per_tubelet(&labels),
            1,
        );
        let (by_ids, _) = f.expand_images(&tok, &tok.encode(&single).unwrap(), &kept).unwrap();
        assert_eq!(by_ids, ids);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// a request with a video, no pruning

fn square_video() -> (Value, Vec<String>) {
    let fx = fixture("videos.json");
    let v = fx["videos"].as_array().unwrap().iter().find(|v| v["name"] == "square_5frames").unwrap().clone();
    let frames = fixture_frames(&v);
    (v, frames)
}

#[test]
fn a_five_frame_video_has_three_tubelets_their_spans_ids_and_keys() {
    let (v, frames) = square_video();
    let f = omni_video(VideoSettings::default());
    let p = prep_for(&f, Limits::default());
    let t = tok();
    let messages = question_and_video(&frames);
    let ing = ingest(&p, &f, &messages, None).unwrap();
    let prepared = ing.video.as_ref().expect("a video request");
    assert!(ing.prepared.is_empty());
    assert_eq!(
        (prepared.plan.frames, prepared.tubelets(), prepared.plan.tokens_per_tubelet, prepared.plan.tokens),
        (5, 3, 256, 768)
    );
    assert_eq!(prepared.pre.canvas(), (512, 512));
    assert_eq!(prepared.source, (320, 320));
    assert_eq!(prepared.labels, ["Frame 1 and frame 2: ", "Frame 3 and frame 4: ", "Frame 5: "]);

    // the prompt is the processor's expanded text, tokenized
    let rendered = f.render(&messages, &Map::new()).unwrap();
    let want_text = expand_video_text(&rendered.text, &prepared.labels, &[256, 256, 256]).unwrap();
    assert_eq!(ing.ids, t.encode(&want_text).unwrap());
    assert_eq!(ing.starts_in_reasoning, rendered.starts_in_reasoning);
    // the same text the fixture's processor made for the same messages shape (its prompt has a system turn, ours the template's own)
    let plain = v["texts"].as_array().unwrap().iter().find(|x| x["name"] == "plain").unwrap();
    assert_eq!(plain["image_runs"].as_array().unwrap().len(), 3);

    // three spans of 256 rows, their merged grid the tile's, each inside its `<img>` ... `</img>`
    let (pad, open, close) =
        (t.token_to_id("<image>").unwrap(), t.token_to_id("<img>").unwrap(), t.token_to_id("</img>").unwrap());
    assert_eq!(ing.images.len(), 3);
    for sp in &ing.images {
        assert_eq!((sp.tokens, sp.merged), (256, (16, 16)));
        assert_eq!(ing.ids[sp.start - 1], open);
        assert!(ing.ids[sp.start..sp.start + 256].iter().all(|&x| x == pad));
        assert_eq!(ing.ids[sp.start + 256], close);
    }
    assert!(ing.images.windows(2).all(|w| w[0].start + 256 < w[1].start));
    assert_eq!(ing.ids.iter().filter(|&&x| x == pad).count(), 768);
    // with no encoder the spans carry the video and the tubelet index
    for (g, sp) in ing.images.iter().enumerate() {
        let d = sp.features.downcast_ref::<DeferredTubelet>().expect("deferred");
        assert_eq!(d.index, g);
        assert_eq!(d.video.key, prepared.key);
    }
    // keys: distinct per tubelet, never 0, and not a still image's
    let keys: Vec<u64> = ing.images.iter().map(|s| s.key).collect();
    assert!(keys.iter().all(|&k| k != 0));
    assert_eq!(keys.iter().collect::<std::collections::BTreeSet<_>>().len(), 3);
    let still = NemotronPreprocessing::default().preprocess(&crop(&source(v["file"].as_str().unwrap()), 0, 0, 320, 320)).unwrap();
    for k in &keys {
        assert_ne!(*k, p.key(still.pixel_hash, (512, 512)), "a tubelet must not share a still image's cache key");
    }
    // the tubelets' planes: frame 2g then frame 2g + 1, the last frame twice for the padded tubelet
    let t2 = prepared.tubelet_bf16(2);
    let per = 3 * 512 * 512;
    assert_eq!(t2.len(), 2 * per);
    assert_eq!(t2[..per], t2[per..]);
}

#[test]
fn tubelets_that_share_a_pair_of_frames_share_a_key_and_a_changed_frame_does_not() {
    let (_, frames) = square_video();
    let f = omni_video(VideoSettings::default());
    let p = prep_for(&f, Limits::default());
    let long = ingest(&p, &f, &question_and_video(&frames), None).unwrap();
    let short = ingest(&p, &f, &question_and_video(&frames[..4]), None).unwrap();
    assert_eq!(short.images.len(), 2);
    // the first two tubelets are the same pairs of frames: the same features, so the same keys
    assert_eq!(short.images[0].key, long.images[0].key);
    assert_eq!(short.images[1].key, long.images[1].key);
    // frame 5 alone is a tubelet of [f5, f5]; the long video's third
    let mut changed = frames.clone();
    changed[1] = frames[4].clone();
    let other = ingest(&p, &f, &question_and_video(&changed), None).unwrap();
    assert_ne!(other.images[0].key, long.images[0].key, "a frame of the pair changed");
    assert_eq!(other.images[1].key, long.images[1].key);
    // the video's own key follows every frame
    assert_ne!(other.video.as_ref().unwrap().key, long.video.as_ref().unwrap().key);
    // a different preprocessing identity or pruning setting is another key for the same frames
    let pruned_cfg = VideoSettings { config: VideoConfig { pruning_rate: 0.5, ..Default::default() }, ..Default::default() };
    let f2 = omni_video(pruned_cfg);
    let p2 = prep_for(&f2, Limits::default());
    let tower = MockTower::new();
    let c = ingest(&p2, &f2, &question_and_video(&frames), Some(&tower)).unwrap();
    assert_ne!(c.video.as_ref().unwrap().key, long.video.as_ref().unwrap().key);
}

#[test]
fn an_encoder_without_pruning_is_called_once_with_no_keep_callback_and_its_rows_ride_the_spans() {
    let (_, frames) = square_video();
    let f = omni_video(VideoSettings::default());
    let p = prep_for(&f, Limits::default());
    let tower = MockTower::new();
    let ing = ingest(&p, &f, &question_and_video(&frames), Some(&tower)).unwrap();
    assert_eq!(*tower.calls.lock().unwrap(), [(3, false)]);
    assert_eq!(ing.images.len(), 3);
    for sp in &ing.images {
        let rows = sp.features.downcast_ref::<Vec<u16>>().expect("the encoder's handle");
        assert_eq!(rows.len(), 256 * 4096);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// EVS

#[test]
fn pruning_is_the_models_drop_of_the_placeholder_rows() {
    let fx = fixture("videos.json");
    let (v, frames) = square_video();
    let t = tok();
    for pv in fx["pruned_square"].as_array().unwrap() {
        let rate = pv["rate"].as_f64().unwrap();
        let settings = VideoSettings { config: VideoConfig { pruning_rate: rate, ..Default::default() }, ..Default::default() };
        let f = omni_video(settings);
        let p = prep_for(&f, Limits::default());
        let tower = MockTower { features: Some(recipe_bits(&pv["recipe"])), calls: Mutex::new(Vec::new()), fail: None };
        let messages = question_and_video(&frames);
        let ing = ingest(&p, &f, &messages, Some(&tower)).unwrap();
        assert_eq!(
            *tower.calls.lock().unwrap(),
            [(3, true)],
            "one call, with the callback: pruning is on and there are three tubelets"
        );
        let kept: Vec<usize> = pv["kept_per_tubelet"].as_array().unwrap().iter().map(u).collect();
        let mask = mask_from_hex(pv["mask_hex"].as_str().unwrap(), 3 * 256);
        assert_eq!(kept_per_tubelet(&mask, 3, 256), kept, "rate {rate}");
        // the prompt: the labels and tags as the processor makes them, each tubelet's run cut to its kept count
        let rendered = f.render(&messages, &Map::new()).unwrap().text;
        let labels = frame_labels(5, &VideoMetadata::default()).unwrap();
        let text = expand_video_text(&rendered, &labels, &kept).unwrap();
        assert_eq!(ing.ids, t.encode(&text).unwrap(), "rate {rate}: ids");
        // the fixture's own pruned text (the model's drop applied to the processor's expansion), minus the template's differences
        let want = pv["pruned_text"].as_str().unwrap();
        let chunk = |s: &str| s[s.find("Frame 1").unwrap()..s.rfind("</img>").unwrap() + 6].to_string();
        assert_eq!(chunk(&text), chunk(want), "rate {rate}: the video's text");
        // spans: only tubelets that kept rows, in order, with the kept count; a tubelet with none has none
        let spans: Vec<usize> = ing.images.iter().map(|s| s.tokens).collect();
        assert_eq!(spans, kept.iter().copied().filter(|&k| k > 0).collect::<Vec<_>>());
        for sp in &ing.images {
            assert_eq!(sp.tokens, sp.merged.0 * sp.merged.1);
        }
        // the rows the encoder kept are the rows the mask flags
        let all: Vec<u16> = recipe_bits(&pv["recipe"]);
        let mut at = 0usize;
        for (g, &k) in kept.iter().enumerate() {
            if k == 0 {
                continue;
            }
            let rows = ing.images[at].features.downcast_ref::<Vec<u16>>().unwrap();
            let want: Vec<u16> = (0..256)
                .filter(|&i| mask[g * 256 + i])
                .flat_map(|i| all[(g * 256 + i) * 4096..(g * 256 + i + 1) * 4096].iter().copied())
                .collect();
            assert_eq!(rows, &want, "tubelet {g}");
            at += 1;
        }
        // the number of kept rows is data-independent: the plan says it before the tower runs
        assert_eq!(ing.video.as_ref().unwrap().plan.kept_tokens, u(&pv["keep"]));
        assert_eq!(ing.ids.iter().filter(|&&x| x == t.token_to_id("<image>").unwrap()).count(), u(&pv["keep"]));
        let _ = &v;
    }
}

#[test]
fn pruning_needs_an_encoder_and_a_square_tile_can_be_required() {
    let (_, frames) = square_video();
    let on = VideoSettings { config: VideoConfig { pruning_rate: 0.7, require_square_grid: false }, ..Default::default() };
    let f = omni_video(on);
    let p = prep_for(&f, Limits::default());
    let e = ingest(&p, &f, &question_and_video(&frames), None).err().unwrap();
    assert_eq!(e.status, 500, "{}", e.message);
    assert!(e.message.contains("no video encoder"), "{}", e.message);
    // one tubelet: pruning cannot drop anything, so no encoder is asked for a mask and none is needed
    let one = ingest(&p, &f, &question_and_video(&frames[..2]), None).unwrap();
    assert_eq!(one.images.len(), 1);
    // a 16:9 video has 252 tokens a tubelet: the model's own EVS raises, strict refuses it by name; the default serves it on its true grid
    let wide = flat_frames(4, 640, 360);
    let strict = VideoSettings { config: VideoConfig { pruning_rate: 0.7, require_square_grid: true }, ..Default::default() };
    let fs = omni_video(strict);
    let ps = prep_for(&fs, Limits::default());
    let tower = MockTower::new();
    let e = ingest(&ps, &fs, &question_and_video(&wide), Some(&tower)).err().unwrap();
    assert!(
        e.message.contains("252") && e.message.contains("perfect square") && e.message.contains("messages[0].content[1]"),
        "{}",
        e.message
    );
    assert!(tower.calls.lock().unwrap().is_empty(), "refused before the tower");
    let ing = ingest(&p, &f, &question_and_video(&wide), Some(&tower)).unwrap();
    assert_eq!(*tower.calls.lock().unwrap(), [(2, true)]);
    assert_eq!(ing.video.as_ref().unwrap().plan.tokens_per_tubelet, 252);
    // 2 tubelets of 252: keep max(252, int(504 * 0.3)) = 252, the first tubelet whole
    assert_eq!(ing.images.iter().map(|s| s.tokens).collect::<Vec<_>>(), [252]);
}

#[test]
fn the_encoders_failures_keep_their_status_and_a_wrong_answer_is_a_500() {
    let (_, frames) = square_video();
    let f = omni_video(VideoSettings::default());
    let p = prep_for(&f, Limits::default());
    let msgs = question_and_video(&frames);
    for (fail, status) in [
        (Some((|| ModelError::Busy("the feature memory is in use".into())) as fn() -> ModelError), 503),
        (Some(|| ModelError::Request("not frames the tower takes".into())), 400),
        (Some(|| ModelError::StateLost("the encoder is gone".into())), 500),
    ] {
        let tower = MockTower { features: None, calls: Mutex::new(Vec::new()), fail };
        let e = ingest(&p, &f, &msgs, Some(&tower)).err().unwrap();
        assert_eq!(e.status, status, "{}", e.message);
        assert!(e.message.contains("messages[0].content[1]"), "{}", e.message);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// refusals and budgets

fn refused(f: &OmniFormat, messages: &Value, allow_images: bool, want: &[&str]) {
    let e = f.check_media(messages, allow_images).unwrap_err();
    assert_eq!(e.status, 400, "{}", e.message);
    for w in want {
        assert!(e.message.contains(w), "{w:?} not in {:?}", e.message);
    }
}

#[test]
fn video_is_off_by_default_and_on_it_needs_the_vision_tower() {
    let frames = flat_frames(2, 64, 64);
    let msgs = question_and_video(&frames);
    // as before: refused by name, whatever the vision flag
    let off = omni();
    assert!(!off.serves_video());
    for allow in [true, false] {
        refused(&off, &msgs, allow, &["messages[0].content[1]", "video_url", "not available yet"]);
    }
    // the registry's format has it off too (the server flags are not wired yet)
    let reg = FormatRegistry::builtin().open(&model_dir(), None).unwrap();
    assert!(reg.check_media(&msgs, true).is_err());
    // on: accepted with vision, refused without it naming the part
    let on = omni_video(VideoSettings::default());
    assert!(on.serves_video());
    assert!(on.check_media(&msgs, true).is_ok());
    refused(&on, &msgs, false, &["messages[0].content[1]", "video_url", "image support", "vision tower"]);
    // text still passes, audio is still refused by name
    assert!(on.check_media(&json!([{"role": "user", "content": "hi"}]), false).is_ok());
    let audio = user(vec![json!({"type": "input_audio", "input_audio": {"data": "AA", "format": "wav"}})]);
    refused(&on, &audio, true, &["input_audio", "no audio"]);
    // the settings are checked when they are given
    assert!(omni()
        .with_video(VideoSettings { config: VideoConfig { pruning_rate: 1.5, ..Default::default() }, ..Default::default() })
        .is_err());
    assert!(omni()
        .with_video(VideoSettings { limits: VideoLimits { max_frames: 0, max_tokens: 10 }, ..Default::default() })
        .is_err());
}

#[test]
fn a_video_part_is_checked_by_name_before_any_pixel() {
    let on = omni_video(VideoSettings::default());
    let f = flat_frames(2, 64, 64);
    let text = json!({"type": "text", "text": "q"});
    // one video, no image beside it
    let two = user(vec![video_part(&f), json!({"type": "video", "video": f})]);
    refused(&on, &two, true, &["messages[0].content[1]", "at most one video", "messages[0].content[0]"]);
    let img = json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}});
    refused(
        &on,
        &user(vec![img.clone(), video_part(&f)]),
        true,
        &["messages[0].content[0]", "an image and a video", "messages[0].content[1]"],
    );
    refused(&on, &user(vec![video_part(&f), img]), true, &["an image and a video"]);
    // frames
    refused(&on, &user(vec![text.clone(), video_part(&[])]), true, &["messages[0].content[1]", "at least one frame"]);
    let many = vec!["data:image/png;base64,AA==".to_string(); 33];
    refused(&on, &user(vec![video_part(&many)]), true, &["33 frames", "the limit is 32", "--max-video-frames"]);
    let tight = omni_video(VideoSettings { limits: VideoLimits { max_frames: 4, max_tokens: 4096 }, ..Default::default() });
    assert!(tight.check_media(&user(vec![video_part(&vec!["data:image/png;base64,AA==".to_string(); 4])]), true).is_ok());
    refused(
        &tight,
        &user(vec![video_part(&vec!["data:image/png;base64,AA==".to_string(); 5])]),
        true,
        &["5 frames", "the limit is 4"],
    );
    refused(&on, &user(vec![json!({"type": "video_url", "video_url": {"url": [3, 4]}})]), true, &["frame 1 is not a URL"]);
    refused(&on, &user(vec![json!({"type": "video_url", "video_url": {"url": ["x", ""]}})]), true, &["frame 2 is empty"]);
    refused(&on, &user(vec![json!({"type": "video_url", "video_url": {"nope": 1}})]), true, &["no `url`"]);
    refused(&on, &user(vec![json!({"type": "video_url"})]), true, &["needs its frames"]);
    // containers and bare URLs
    refused(
        &on,
        &user(vec![json!({"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}})]),
        true,
        &["container", "mp4"],
    );
    refused(
        &on,
        &user(vec![json!({"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}})]),
        true,
        &["list of frames"],
    );
    refused(&on, &user(vec![json!({"type": "video_url", "video_url": 7})]), true, &["array of image data URLs"]);
    // the frame-sequence string: JPEG frames split on commas, a container subtype refused
    let seq = user(vec![json!({"type": "video_url", "video_url": {"url": "data:video/jpeg;base64,AAAA,BBBB,CCCC"}})]);
    assert!(on.check_media(&seq, true).is_ok());
    let parsed = gb_server::formats::nemotron_omni::video::the_video(&seq, &VideoLimits::default()).unwrap().unwrap();
    assert_eq!(parsed.frames, ["data:image/jpeg;base64,AAAA", "data:image/jpeg;base64,BBBB", "data:image/jpeg;base64,CCCC"]);
    // WebP frames: the same split (each frame is then a still image through the image decoders); `webm` stays a refused container
    let seq = user(vec![json!({"type": "video_url", "video_url": {"url": "data:video/webp;base64,AAAA,BBBB"}})]);
    let parsed = gb_server::formats::nemotron_omni::video::the_video(&seq, &VideoLimits::default()).unwrap().unwrap();
    assert_eq!(parsed.frames, ["data:image/webp;base64,AAAA", "data:image/webp;base64,BBBB"]);
    refused(
        &on,
        &user(vec![json!({"type": "video_url", "video_url": {"url": "data:video/webm;base64,AAAA"}})]),
        true,
        &["container", "webm"],
    );
    // metadata
    let with = |extra: Value| {
        let mut p = json!({"type": "video_url", "video_url": {"url": f}});
        for (k, v) in extra.as_object().unwrap() {
            p["video_url"][k] = v.clone();
        }
        user(vec![p])
    };
    assert!(on.check_media(&with(json!({"fps": 2.0, "frame_indices": [0, 15]})), true).is_ok());
    refused(&on, &with(json!({"frame_indices": [0, 1]})), true, &["needs `fps`"]);
    refused(&on, &with(json!({"fps": 0})), true, &["`fps` must be a number above 0"]);
    refused(&on, &with(json!({"fps": -3.0})), true, &["`fps` must be a number above 0"]);
    refused(&on, &with(json!({"fps": "fast"})), true, &["`fps`"]);
    refused(&on, &with(json!({"fps": 2.0, "frame_indices": [0]})), true, &["1 entries for 2 frames"]);
    refused(&on, &with(json!({"fps": 2.0, "frame_indices": [0, -1]})), true, &["non-negative"]);
    refused(&on, &with(json!({"fps": 1e-300})), true, &["whole frame duration"]);
    // a payload under no type the template renders would be dropped without a trace
    refused(&on, &user(vec![json!({"video": f.clone()})]), true, &["dropped silently", "`video`"]);
    // a media key on a text part is still refused (the shared rule, with the frames not copied)
    refused(
        &on,
        &user(vec![json!({"type": "text", "text": "x", "input_audio": {"data": "AA"}})]),
        true,
        &["messages[0].content[0]", "not supported"],
    );
}

#[test]
fn the_budgets_are_enforced_from_the_sizes_before_the_resize() {
    // tokens of the video: 5 square frames are 3 x 256 = 768
    let f = omni_video(VideoSettings { limits: VideoLimits { max_frames: 32, max_tokens: 500 }, ..Default::default() });
    let p = prep_for(&f, Limits::default());
    let e = ingest(&p, &f, &question_and_video(&flat_frames(5, 512, 512)), None).err().unwrap();
    assert_eq!(e.status, 400);
    assert!(
        e.message.contains("messages[0].content[1]")
            && e.message.contains("768 tokens")
            && e.message.contains("3 tubelets of 256")
            && e.message.contains("5 frames")
            && e.message.contains("limit is 500"),
        "{}",
        e.message
    );
    // exactly at the limit is served (4 frames: 512 tokens)
    let f = omni_video(VideoSettings { limits: VideoLimits { max_frames: 32, max_tokens: 512 }, ..Default::default() });
    let p = prep_for(&f, Limits::default());
    assert_eq!(ingest(&p, &f, &question_and_video(&flat_frames(4, 512, 512)), None).unwrap().images.len(), 2);
    // the per-frame tile against `--max-image-tokens`: a 333 x 517 frame is 260 tokens
    let f = omni_video(VideoSettings::default());
    let p = prep_for(&f, Limits { max_tokens_per_image: 256, ..Limits::default() });
    let e = ingest(&p, &f, &question_and_video(&flat_frames(2, 333, 517)), None).err().unwrap();
    assert!(
        e.message.contains("260 tokens") && e.message.contains("416x640") && e.message.contains("limit is 256"),
        "{}",
        e.message
    );
    // a strip the tower cannot run (100,000 x 1 resolves to a tile of 20,236 patches, above the 13,312 of the tower) is the preprocessing's own
    // refusal, named by the part
    let p = prep_for(&f, Limits::default());
    let e = ingest(&p, &f, &question_and_video(&flat_frames(2, 100_000, 1)), None).err().unwrap();
    assert_eq!(e.status, 400);
    assert!(e.message.contains("messages[0].content[1]"), "{}", e.message);
    // one tile for every frame: a second frame of another aspect is refused by number
    let mixed = vec![flat_frames(1, 640, 480).remove(0), flat_frames(1, 480, 640).remove(0)];
    let e = ingest(&p, &f, &question_and_video(&mixed), None).err().unwrap();
    assert!(e.message.contains("frame 2") && e.message.contains("one tile"), "{}", e.message);
    // frames of different source sizes that resolve to the same tile are one video
    let same = vec![flat_frames(1, 640, 480).remove(0), flat_frames(1, 641, 480).remove(0)];
    assert!(ingest(&p, &f, &question_and_video(&same), None).is_ok());
}

#[test]
fn the_byte_limits_are_413s_naming_the_frame() {
    let f = omni_video(VideoSettings::default());
    let frames = flat_frames(3, 128, 128);
    let one = std::fs::metadata(testdata().join("vision/check_512x512.png")).unwrap().len() as usize;
    let _ = one;
    let p = prep_for(&f, Limits { max_bytes: 100, ..Limits::default() });
    let e = ingest(&p, &f, &question_and_video(&frames), None).err().unwrap();
    assert_eq!(e.status, 413, "{}", e.message);
    assert!(e.message.contains("messages[0].content[1]") && e.message.contains("frame 1"), "{}", e.message);
    let bytes_of = |uri: &str| gb_vision::decode::data_uri_bytes(uri).unwrap().len();
    let each = bytes_of(&frames[0]);
    let p = prep_for(&f, Limits { max_total_bytes: 2 * each + each / 2, ..Limits::default() });
    let e = ingest(&p, &f, &question_and_video(&frames), None).err().unwrap();
    assert_eq!(e.status, 413, "{}", e.message);
    assert!(e.message.contains("frame 3") && e.message.contains("in all"), "{}", e.message);
    // bad data names the frame
    let mut bad = frames.clone();
    bad[1] = "data:image/png;base64,AAAA".into();
    let p = prep_for(&f, Limits::default());
    let e = ingest(&p, &f, &question_and_video(&bad), None).err().unwrap();
    assert_eq!(e.status, 400);
    assert!(e.message.contains("messages[0].content[1]") && e.message.contains("frame 2"), "{}", e.message);
    // a remote URL is the fetcher's business: the data-URI fetcher refuses it by name, with the frame
    let mut remote = frames.clone();
    remote[0] = "https://example.com/a.png".into();
    let e = ingest(&p, &f, &question_and_video(&remote), None).err().unwrap();
    assert!(e.message.contains("frame 1") && e.message.contains("data: URIs"), "{}", e.message);
}

#[test]
fn literal_placeholders_beside_a_video_are_refused() {
    let f = omni_video(VideoSettings::default());
    let p = prep_for(&f, Limits::default());
    let frames = flat_frames(2, 512, 512);
    for (text, want) in [("see <image> here", "literal <image>"), ("scratch <|placeholder|> marker", "<|placeholder|>")] {
        let msgs = user(vec![json!({"type": "text", "text": text}), video_part(&frames)]);
        let e = ingest(&p, &f, &msgs, None).err().unwrap();
        assert_eq!(e.status, 400);
        assert!(e.message.contains(want), "{}", e.message);
    }
    // a typed <video> takes the place of the part's
    let msgs = user(vec![json!({"type": "text", "text": "Look at <video> closely"}), video_part(&frames)]);
    let ing = ingest(&p, &f, &msgs, None).unwrap();
    assert_eq!(ing.images.len(), 1);
}

#[test]
fn images_and_text_are_untouched_by_a_video_enabled_ingress() {
    // the same two-image request gives the same ids, spans and keys with and without the video path on
    let png = |name: &str| {
        let bytes = std::fs::read(testdata().join("vision").join(name)).unwrap();
        format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes))
    };
    let msgs = user(vec![
        json!({"type": "text", "text": "Compare"}),
        json!({"type": "image_url", "image_url": {"url": png("check_512x512.png")}}),
        json!({"type": "text", "text": "with"}),
        json!({"type": "image_url", "image_url": {"url": png("rgba_200x150.png")}}),
    ]);
    let (plain, with) = (omni(), omni_video(VideoSettings::default()));
    let (pp, pw) = (prep_for(&plain, Limits::default()), prep_for(&with, Limits::default()));
    let a = ingest(&pp, &plain, &msgs, None).unwrap();
    let b = ingest(&pw, &with, &msgs, None).unwrap();
    assert_eq!(a.ids, b.ids);
    assert_eq!(
        a.images.iter().map(|s| (s.start, s.tokens, s.merged, s.key)).collect::<Vec<_>>(),
        b.images.iter().map(|s| (s.start, s.tokens, s.merged, s.key)).collect::<Vec<_>>()
    );
    assert!(a.video.is_none() && b.video.is_none());
    // text with a video-enabled format renders as before
    let t = json!([{"role": "user", "content": "a b a"}]);
    assert_eq!(plain.render(&t, &Map::new()).unwrap().text, with.render(&t, &Map::new()).unwrap().text);
    // the processor's own expansion of a single image through `expand_text` is untouched
    assert_eq!(expand_text("a<image>b", &[3]).unwrap(), "a<img><image><image><image></img>b");
}

/// The repository root: `REXL3_REPO_ROOT` when set (a prebuilt test binary run from an installed bin directory that carries the
/// tree's `testdata/`, e.g. on a node without this checkout), else the checkout this binary was built from.
fn repo_root() -> std::path::PathBuf {
    std::env::var_os("REXL3_REPO_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}
