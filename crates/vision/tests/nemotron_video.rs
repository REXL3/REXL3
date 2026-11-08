//! The Nemotron 3.5 Super VL video path against the pack's own HF code (`testdata/nemotron_omni/video_pre`, made by
//! `tools/nemotron_omni/gen_video_fixture.py` with transformers 5.12.1 and torch 2.14.0 on a GB10, CPU): `compute_retention_mask` bit for bit, the
//! tubelet stack and patch layout of `forward_video`, the token arithmetic, three videos' pixel tensors and the processor's labels.
//!
//! Debug builds run the seeded EVS cases at hidden 4096 of up to 4 tubelets only (an unoptimised 16 x 256 x 4096 case takes tens of seconds);
//! `cargo test --release -p gb-vision --test nemotron_video` runs them all.

use gb_vision::decode::base64_decode;
use gb_vision::nemotron::{f32_to_bf16_bits, num_tokens, target_patches_video, NemotronPreprocessing};
use gb_vision::nemotron_video::{
    bf16_bits_to_f32, dissimilarity_bf16, evs_keep_count, frame_labels, kept_per_tubelet, patchify_tubelet, plan_video,
    preprocess_video, retention_mask_bits, square_grid, tubelet_count, PreprocessedVideo, SumOrder, VideoConfig, VideoMetadata,
    VIDEO_PATCH_DIM,
};
use gb_vision::rgb_decode::{decode_rgb, AlphaPolicy};
use gb_vision::{Rgb, VisionError};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn fixture(name: &str) -> Value {
    let p = root().join("nemotron_omni/video_pre").join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}

fn sha_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn b64_u16(v: &Value) -> Vec<u16> {
    let bytes = base64_decode(v.as_str().unwrap()).unwrap();
    bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect()
}

fn b64_f32(v: &Value) -> Vec<f32> {
    let bytes = base64_decode(v.as_str().unwrap()).unwrap();
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn u(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// The seeded recipe: every line mirrors `recipe_bits` of the generator, integer for integer.

fn splitmix(x: u64) -> u64 {
    let x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn stream(seed: u64, tag: u64, idx: u64) -> u64 {
    let base = seed.wrapping_mul(0xD1B5_4A32_D192_ED03).wrapping_add(tag.wrapping_mul(0x8CB9_2BA7_2F3D_8DD7));
    splitmix(base.wrapping_add(idx))
}

fn gaussish(seed: u64, tag: u64, idx: u64) -> f32 {
    let mut v = 0i64;
    for k in 0..3 {
        v += (stream(seed, 3 * tag + k, idx) >> 42) as i64;
    }
    v -= 3 * (1 << 21);
    v as f32 * (1.0 / 2_097_152.0)
}

fn unit(seed: u64, tag: u64, idx: u64) -> f32 {
    (stream(seed, tag, idx) >> 40) as f32 * (1.0 / 16_777_216.0)
}

fn recipe_bits(r: &Value) -> Vec<u16> {
    let (seed, t, n, hidden) = (r["seed"].as_u64().unwrap(), u(&r["t"]), u(&r["n"]), u(&r["hidden"]));
    let kind = r["kind"].as_str().unwrap();
    let (amp, p) = (r["amp"].as_f64().unwrap() as f32, r["p"].as_f64().unwrap() as f32);
    let mut cur: Vec<f32> = (0..n * hidden).map(|k| gaussish(seed, 0, k as u64)).collect();
    let mut x: Vec<f32> = Vec::with_capacity(t * n * hidden);
    x.extend_from_slice(&cur);
    for g in 1..t {
        if kind == "indep" {
            cur = (0..n * hidden).map(|k| gaussish(seed, g as u64, k as u64)).collect();
        } else {
            let scale: Vec<f32> = (0..n).map(|i| unit(seed, 1000 + g as u64, i as u64) * amp).collect();
            let copy: Vec<bool> = (0..n).map(|i| kind == "dup" && unit(seed, 2000 + g as u64, i as u64) < p).collect();
            let mut nxt = vec![0f32; n * hidden];
            for i in 0..n {
                for k in 0..hidden {
                    let noise = gaussish(seed, g as u64, (i * hidden + k) as u64);
                    nxt[i * hidden + k] = if copy[i] { cur[i * hidden + k] } else { cur[i * hidden + k] + scale[i] * noise };
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
    if r.get("dimscale").and_then(Value::as_bool).unwrap_or(false) {
        let s: Vec<f32> = (0..hidden)
            .map(|k| {
                let e = (unit(seed, 600, k as u64) * 12.0f32) as i32 - 6;
                f32::from_bits(((127 + e) as u32) << 23)
            })
            .collect();
        for row in x.chunks_exact_mut(hidden) {
            for (v, s) in row.iter_mut().zip(&s) {
                *v *= s;
            }
        }
    }
    x.iter().map(|&v| f32_to_bf16_bits(v)).collect()
}

fn mask_from_hex(hex: &str, n: usize) -> Vec<bool> {
    let bytes: Vec<u8> = (0..hex.len() / 2).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect();
    (0..n).map(|i| bytes[i / 8] >> (i % 8) & 1 == 1).collect()
}

fn case_bits(c: &Value) -> Vec<u16> {
    if let Some(r) = c.get("recipe") {
        let bits = recipe_bits(r);
        let sha = sha_hex(&bits.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>());
        assert_eq!(
            sha,
            c["bits_sha256"].as_str().unwrap(),
            "{}: the recipe does not regenerate the generator's embeddings",
            c["name"]
        );
        bits
    } else {
        b64_u16(&c["bits_b64"])
    }
}

/// Whether a case is cheap enough for an unoptimised build.
fn affordable(c: &Value) -> bool {
    !cfg!(debug_assertions) || u(&c["t"]) * u(&c["h"]) * u(&c["w"]) * u(&c["hidden"]) <= 4 * 256 * 4096
}

struct Agreement {
    cases: usize,
    scores: usize,
    scores_equal: usize,
    masks: usize,
    masks_equal: usize,
}

fn evs_against_fixture(order: SumOrder) -> Agreement {
    let fx = fixture("evs.json");
    let mut a = Agreement { cases: 0, scores: 0, scores_equal: 0, masks: 0, masks_equal: 0 };
    for c in fx["cases"].as_array().unwrap() {
        if !affordable(c) {
            continue;
        }
        let name = c["name"].as_str().unwrap();
        let (t, h, w, hidden) = (u(&c["t"]), u(&c["h"]), u(&c["w"]), u(&c["hidden"]));
        let bits = case_bits(c);
        assert_eq!(bits.len(), t * h * w * hidden, "{name}");
        let want = b64_u16(&c["dissim_b64"]);
        let started = std::time::Instant::now();
        let got = dissimilarity_bf16(&bits, t, h * w, hidden, order);
        if t * h * w * hidden >= 10_000_000 {
            eprintln!(
                "{name}: {t} x {} tokens x {hidden}: dissimilarity in {:.0} ms ({order:?})",
                h * w,
                started.elapsed().as_secs_f64() * 1e3
            );
        }
        a.cases += 1;
        a.scores += want.len();
        let equal = got.iter().zip(&want).filter(|(g, w)| g == w).count();
        a.scores_equal += equal;
        if order == SumOrder::TorchCpu {
            assert_eq!(
                equal,
                want.len(),
                "{name}: {} of {} scores differ from torch's bf16 composite",
                want.len() - equal,
                want.len()
            );
        }
        for (q, m) in c["masks"].as_object().unwrap() {
            let rate: f64 = q.parse().unwrap();
            let want_mask = mask_from_hex(m["mask_hex"].as_str().unwrap(), t * h * w);
            let mask = retention_mask_bits(&bits, t, h, w, hidden, rate, order).unwrap();
            assert_eq!(mask.iter().filter(|&&k| k).count(), u(&m["keep"]), "{name} q={q}: kept rows");
            a.masks += 1;
            let same = mask == want_mask;
            a.masks_equal += same as usize;
            if order == SumOrder::TorchCpu {
                assert!(same, "{name} q={q}: the retention mask differs from the model's function");
            }
        }
    }
    a
}

#[test]
fn the_dissimilarity_and_every_mask_are_the_models_function_bit_for_bit() {
    let a = evs_against_fixture(SumOrder::TorchCpu);
    eprintln!(
        "torch order: {} cases, {} of {} scores identical, {} of {} masks identical",
        a.cases, a.scores_equal, a.scores, a.masks_equal, a.masks
    );
    assert!(a.cases >= 15);
    assert_eq!(a.scores_equal, a.scores);
    assert_eq!(a.masks_equal, a.masks);
}

#[test]
fn a_plain_left_to_right_order_is_close_and_this_is_how_close() {
    // not a gate on equality: what a naive port would get. The scores can differ by one bf16 step where a rounding sits on a boundary.
    let a = evs_against_fixture(SumOrder::Sequential);
    eprintln!(
        "sequential order: {} cases, {} of {} scores identical ({:.4}%), {} of {} masks identical",
        a.cases,
        a.scores_equal,
        a.scores,
        100.0 * a.scores_equal as f64 / a.scores as f64,
        a.masks_equal,
        a.masks
    );
    assert!(a.scores_equal as f64 >= 0.99 * a.scores as f64);
    assert!(a.masks_equal as f64 >= 0.9 * a.masks as f64);
}

#[test]
fn the_seeded_cases_have_the_structure_that_makes_ties_decisive() {
    let fx = fixture("evs.json");
    for c in fx["cases"].as_array().unwrap() {
        if let Some(d) = c.get("distinct_scores") {
            let tokens = u(&c["t"]) * u(&c["h"]) * u(&c["w"]);
            // bf16 scores of 1000s of tokens fall on a few dozen values: the cut lands inside a tie and the index decides it
            assert!(u(d) < 64 && tokens >= 192, "{}: {} distinct scores of {tokens}", c["name"], d);
        }
    }
}

#[test]
fn the_pruned_square_video_mask_and_counts_are_the_models() {
    let vf = fixture("videos.json");
    let (g, n) = (3usize, 256usize);
    for p in vf["pruned_square"].as_array().unwrap() {
        let bits = recipe_bits(&p["recipe"]);
        assert_eq!(sha_hex(&bits.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>()), p["bits_sha256"].as_str().unwrap());
        let rate = p["rate"].as_f64().unwrap();
        let mask = retention_mask_bits(&bits, g, 16, 16, 4096, rate, SumOrder::TorchCpu).unwrap();
        assert_eq!(mask, mask_from_hex(p["mask_hex"].as_str().unwrap(), g * n), "rate {rate}");
        let kept: Vec<usize> = p["kept_per_tubelet"].as_array().unwrap().iter().map(u).collect();
        assert_eq!(kept_per_tubelet(&mask, g, n), kept);
        assert_eq!(kept.iter().sum::<usize>(), u(&p["keep"]));
        assert_eq!(evs_keep_count(g, n, rate), u(&p["keep"]));
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Tubelets, tokens.

#[test]
fn the_stack_and_the_patches_are_forward_videos_and_image_to_patches() {
    let fx = fixture("tubelets.json");
    for c in fx["stack_cases"].as_array().unwrap() {
        let (n, h, w) = (u(&c["frames"]), u(&c["h"]), u(&c["w"]));
        let frames_bits = b64_u16(&c["frames_b64"]);
        let per = 3 * h * w;
        assert_eq!(frames_bits.len(), n * per);
        let v = PreprocessedVideo {
            tile: gb_vision::nemotron::TilePlan { w_patches: w / 16, h_patches: h / 16, tokens: (w / 16) * (h / 16) / 4 },
            frames: frames_bits.chunks(per).map(|f| f.to_vec()).collect(),
            frame_hashes: vec![0; n],
            pixel_hash: 0,
        };
        let want_stack = b64_u16(&c["stack_b64"]);
        assert_eq!(v.tubelets(), tubelet_count(n));
        assert_eq!(v.stack(), want_stack, "{n} frames: the channel stack");
        assert_eq!(c["stack_shape"], serde_json::json!([tubelet_count(n), 6, h, w]));
        let want_patches = b64_u16(&c["patches_b64"]);
        let per_tubelet = (h / 16) * (w / 16) * VIDEO_PATCH_DIM;
        for g in 0..v.tubelets() {
            let p = patchify_tubelet(&v.tubelet(g), h, w);
            assert_eq!(p, want_patches[g * per_tubelet..(g + 1) * per_tubelet], "{n} frames, tubelet {g}: patches");
        }
    }
}

#[test]
fn the_token_table_is_the_processors_tile_and_the_functions_keep_count() {
    let fx = fixture("tubelets.json");
    let cfg = NemotronPreprocessing::default();
    let rows = fx["token_table"].as_array().unwrap();
    assert!(rows.len() >= 100);
    let mut squares = 0;
    for r in rows {
        let (w, h, frames) = (u(&r["w"]), u(&r["h"]), u(&r["frames"]));
        let (wp, hp) = target_patches_video(w, h, &cfg).unwrap();
        assert_eq!((wp, hp, num_tokens(wp, hp, &cfg)), (u(&r["w_patches"]), u(&r["h_patches"]), u(&r["tokens"])), "{w}x{h}");
        let tokens = u(&r["tokens"]);
        assert_eq!(tubelet_count(frames), u(&r["tubelets"]));
        for (q, keep) in r["keep"].as_object().unwrap() {
            let rate: f64 = q.parse().unwrap();
            assert_eq!(evs_keep_count(tubelet_count(frames), tokens, rate), u(keep), "{w}x{h} x {frames} frames, q {q}");
        }
        assert_eq!(square_grid(tokens).is_some(), r["square"].as_u64().unwrap() == 1, "{w}x{h}: tokens {tokens}");
        squares += (r["square"].as_u64().unwrap() == 1) as usize;
        // the plan, with and without pruning
        let on = VideoConfig { pruning_rate: 0.7, require_square_grid: false };
        let p = plan_video(&cfg, &on, w, h, frames).unwrap();
        assert_eq!((p.tubelets, p.tokens_per_tubelet, p.kept_tokens), (u(&r["tubelets"]), tokens, u(&r["keep"]["0.7"])));
    }
    assert!(squares > 0 && squares < rows.len());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Videos.

fn crop(img: &Rgb, x: usize, y: usize, w: usize, h: usize) -> Rgb {
    assert!(x + w <= img.width && y + h <= img.height);
    let mut data = Vec::with_capacity(w * h * 3);
    for row in y..y + h {
        data.extend_from_slice(&img.data[(row * img.width + x) * 3..(row * img.width + x + w) * 3]);
    }
    Rgb { width: w, height: h, data }
}

fn axis(n: usize, step: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).step_by(step).collect();
    if *v.last().unwrap() != n - 1 {
        v.push(n - 1);
    }
    v
}

fn load_video(v: &Value) -> Vec<Rgb> {
    let bytes = std::fs::read(root().join("vision").join(v["file"].as_str().unwrap())).unwrap();
    assert_eq!(sha_hex(&bytes), v["source_sha256"].as_str().unwrap(), "not the file the fixture was made from");
    let img = decode_rgb(&bytes, AlphaPolicy::Drop).unwrap();
    v["frame_records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            let c = &f["crop"];
            crop(&img, u(&c[0]), u(&c[1]), u(&c[2]), u(&c[3]))
        })
        .collect()
}

#[test]
fn three_videos_are_the_processors_pixels_and_the_models_stack() {
    let fx = fixture("videos.json");
    let cfg = NemotronPreprocessing::default();
    let mut worst = 0f64;
    let mut unequal = 0usize;
    for v in fx["videos"].as_array().unwrap() {
        let name = v["name"].as_str().unwrap();
        let frames = load_video(v);
        assert_eq!(frames.len(), u(&v["frames"]), "{name}");
        // the plan from the size alone
        let plan = plan_video(&cfg, &VideoConfig::default(), frames[0].width, frames[0].height, frames.len()).unwrap();
        assert_eq!((plan.tile.w_patches, plan.tile.h_patches), (u(&v["w_patches"]), u(&v["h_patches"])), "{name}");
        assert_eq!((plan.tokens_per_tubelet, plan.tubelets), (u(&v["tokens_per_tubelet"]), u(&v["tubelets"])), "{name}");
        let (cw, ch) = plan.tile.canvas();
        assert_eq!([cw, ch], [u(&v["canvas"][0]), u(&v["canvas"][1])], "{name}");
        // every frame's float32 tensor
        for (fi, (img, rec)) in frames.iter().zip(v["frame_records"].as_array().unwrap()).enumerate() {
            let p = cfg.preprocess_video_frame(img).unwrap();
            assert_eq!(p.pixels.len(), 3 * cw * ch);
            let l = &rec["lattice"];
            let (ys, xs) = (axis(ch, u(&l["step_y"])), axis(cw, u(&l["step_x"])));
            assert_eq!((ys.len(), xs.len()), (u(&l["ny"]), u(&l["nx"])), "{name} frame {fi}");
            let want = b64_f32(&l["f32_le_b64"]);
            for c in 0..3 {
                for (iy, &y) in ys.iter().enumerate() {
                    for (ix, &x) in xs.iter().enumerate() {
                        let g = p.pixels[c * cw * ch + y * cw + x];
                        let w = want[(c * ys.len() + iy) * xs.len() + ix];
                        worst = worst.max((g as f64 - w as f64).abs());
                        unequal += (g.to_bits() != w.to_bits()) as usize;
                    }
                }
            }
            let sha = sha_hex(&p.pixels.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            assert_eq!(sha, rec["tensor_sha256"].as_str().unwrap(), "{name} frame {fi}: the float32 tensor");
        }
        // the video: the stack of bf16 planes
        let pre = preprocess_video(&cfg, &frames).unwrap();
        assert_eq!(pre.tile, plan.tile);
        assert_eq!(pre.tubelets(), plan.tubelets);
        let stack = pre.stack();
        let g = pre.tubelets();
        assert_eq!(v["stack"]["shape"], serde_json::json!([g, 6, ch, cw]));
        let sha = sha_hex(&stack.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>());
        assert_eq!(sha, v["stack"]["sha256"].as_str().unwrap(), "{name}: the bf16 stack");
        let l = &v["stack"]["lattice"];
        let (ys, xs) = (axis(ch, u(&l["step_y"])), axis(cw, u(&l["step_x"])));
        let want = b64_u16(&l["u16_le_b64"]);
        assert_eq!(want.len(), g * 6 * ys.len() * xs.len());
        for gi in 0..g {
            for plane in 0..6 {
                for (iy, &y) in ys.iter().enumerate() {
                    for (ix, &x) in xs.iter().enumerate() {
                        let got = stack[((gi * 6 + plane) * ch + y) * cw + x];
                        assert_eq!(
                            got,
                            want[((gi * 6 + plane) * ys.len() + iy) * xs.len() + ix],
                            "{name}: tubelet {gi} plane {plane} ({x}, {y})"
                        );
                    }
                }
            }
        }
        // an odd count repeats its last frame in the last tubelet, and the hashes follow the frames
        if frames.len() % 2 == 1 {
            let last = pre.tubelet(g - 1);
            assert_eq!(last[..3 * cw * ch], last[3 * cw * ch..], "{name}: the padded tubelet");
        }
        assert_eq!(pre.frame_hashes.len(), frames.len());
        eprintln!(
            "{name}: {} frames of {}x{} -> {cw}x{ch}, {} tubelets of {} tokens, stack sha256 identical",
            frames.len(),
            frames[0].width,
            frames[0].height,
            g,
            plan.tokens_per_tubelet
        );
    }
    eprintln!("float32 lattices: worst abs difference {worst:.3e}, {unequal} values differing in a bit");
    assert!(worst <= 1e-5);
}

#[test]
fn the_labels_are_the_processors_for_every_metadata_variant() {
    let fx = fixture("videos.json");
    let mut checked = 0;
    for v in fx["videos"].as_array().unwrap() {
        let n = u(&v["frames"]);
        for t in v["texts"].as_array().unwrap() {
            let meta = match t["metadata"].as_object() {
                None => VideoMetadata::default(),
                Some(m) => VideoMetadata {
                    fps: m["fps"].as_f64(),
                    frame_indices: m["frame_indices"].as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap()).collect()),
                },
            };
            let want: Vec<String> = t["labels"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
            assert_eq!(frame_labels(n, &meta).unwrap(), want, "{} / {}", v["name"], t["name"]);
            checked += 1;
        }
    }
    assert!(checked >= 8);
}

#[test]
fn what_python_raises_is_an_error_here() {
    let cfg = NemotronPreprocessing::default();
    let strict = VideoConfig { pruning_rate: 0.7, require_square_grid: true };
    let e: VisionError = plan_video(&cfg, &strict, 1920, 1080, 8).unwrap_err();
    assert!(e.to_string().contains("252"), "{e}");
    assert!(plan_video(&cfg, &strict, 512, 512, 8).is_ok());
    // a frame whose tile differs from the first's
    let img = |w: usize, h: usize| Rgb { width: w, height: h, data: vec![9; w * h * 3] };
    let e = preprocess_video(&cfg, &[img(640, 480), img(480, 640)]).unwrap_err().to_string();
    assert!(e.contains("frame 2") && e.contains("one tile"), "{e}");
    // 8 bf16 values as f32: the f32 entry point rounds like the model's cast
    let f: Vec<f32> = (0..2 * 4 * 8).map(|i| bf16_bits_to_f32(f32_to_bf16_bits(i as f32 * 0.37))).collect();
    let m = gb_vision::nemotron_video::retention_mask(&f, 2, 2, 2, 8, 0.0);
    assert!(m.iter().all(|&k| k), "rate 0 keeps everything");
}
