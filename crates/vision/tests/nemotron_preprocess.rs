//! The Nemotron 3.5 Super VL image preprocessing against the pack's own HF processor (`testdata/nemotron_omni/vision_pre`, made by
//! `tools/nemotron_omni/gen_vision_pre_fixture.py` with transformers 5.12.1 and torch 2.14.0 on a GB10): the tile and token count of every
//! `(w, h)` of the table, the pixel tensors of real images, the normalisation table, and properties of the tile rule.

use gb_vision::nemotron::{
    num_tokens, per_image_budget, target_patches, target_patches_video, NemotronPreprocessing, MAX_IMAGE_TOKENS,
    TOWER_MAX_PATCHES,
};
use gb_vision::preprocessor::ImagePreprocessor;
use gb_vision::rgb_decode::{decode_rgb, AlphaPolicy};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn fixture(name: &str) -> Value {
    let p = root().join("nemotron_omni/vision_pre").join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}

fn b64_f32(v: &Value) -> Vec<f32> {
    let bytes = gb_vision::decode::base64_decode(v.as_str().unwrap()).unwrap();
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn as_tile(v: &Value) -> Option<[usize; 3]> {
    v.as_array().map(|a| [a[0].as_u64().unwrap() as usize, a[1].as_u64().unwrap() as usize, a[2].as_u64().unwrap() as usize])
}

fn config_named(table: &Value, name: &str) -> (NemotronPreprocessing, usize, usize) {
    let c = &table["configs"][name];
    let cfg = NemotronPreprocessing {
        max_model_len: c["max_model_len"].as_u64().unwrap() as usize,
        ..NemotronPreprocessing::default()
    };
    (cfg, c["n_images"].as_u64().unwrap() as usize, c["tokens_available"].as_u64().unwrap() as usize)
}

#[test]
fn the_pack_config_is_read_to_the_defaults() {
    let text = std::fs::read_to_string(root().join("nemotron_omni/vision_pre/preprocessor_config.json")).unwrap();
    assert_eq!(NemotronPreprocessing::from_json_str(&text).unwrap(), NemotronPreprocessing::default());
    let dir = root().join("nemotron_omni/vision_pre");
    assert_eq!(NemotronPreprocessing::from_dir(&dir).unwrap(), NemotronPreprocessing::default());
    // the numbers the table's configs were made with
    let meta = fixture("meta.json");
    assert_eq!(meta["processor_config"]["max_num_patches"], 13_312);
    assert_eq!(MAX_IMAGE_TOKENS, 3_328);
}

#[test]
fn every_tile_and_token_count_of_the_table_is_the_processors() {
    let table = fixture("table.json");
    let columns: Vec<String> = table["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect();
    let rows = table["rows"].as_array().unwrap();
    assert!(rows.len() >= 3000, "{} pairs", rows.len());
    let video_cfg = NemotronPreprocessing::default();
    let (mut compared, mut errors, mut mismatches) = (0usize, 0usize, Vec::new());
    let (mut min_side, mut max_side) = (usize::MAX, 0usize);
    for row in rows {
        let (w, h) = (row[0].as_u64().unwrap() as usize, row[1].as_u64().unwrap() as usize);
        min_side = min_side.min(w.min(h));
        if w <= 8000 && h <= 8000 {
            max_side = max_side.max(w.max(h));
        }
        for (i, col) in columns.iter().enumerate() {
            let want = as_tile(&row[2 + i]);
            let got = if col == "video" {
                // the rule itself (the frame path then refuses a tile the tower cannot run; the processor does not)
                target_patches_video(w, h, &video_cfg).ok().map(|(wp, hp)| [wp, hp, num_tokens(wp, hp, &video_cfg)])
            } else {
                let (cfg, n, available) = config_named(&table, col);
                // the budget rule and the tile rule, apart and together
                assert_eq!(per_image_budget(&cfg, n), available, "{col}: budget of {n} images at {}", cfg.max_model_len);
                let direct = target_patches(w, h, available, &cfg).ok();
                let planned = cfg.plan_for(w, h, n).ok().map(|t| [t.w_patches, t.h_patches, t.tokens]);
                assert_eq!(direct, planned.map(|p| (p[0], p[1])), "{w}x{h} {col}: target_patches vs plan_for");
                planned
            };
            compared += 1;
            errors += want.is_none() as usize;
            if got != want {
                mismatches.push(format!("{w}x{h} {col}: processor {want:?}, here {got:?}"));
            }
        }
    }
    eprintln!(
        "{} (w, h) pairs, sides {min_side}..={max_side} (plus a few past 8000), {} configurations each: {compared} tiles compared, {errors} of them \
         processor exceptions; {} mismatches",
        rows.len(),
        columns.len(),
        mismatches.len()
    );
    assert!(mismatches.is_empty(), "{} mismatches, first: {:?}", mismatches.len(), &mismatches[..mismatches.len().min(8)]);
    assert!(errors >= 10 && min_side == 1 && max_side == 8000);
}

#[test]
fn the_normalisation_table_is_exact() {
    let fx = fixture("images.json");
    let want = b64_f32(&fx["norm_lut_f32_le_b64"]);
    assert_eq!(want.len(), 3 * 256);
    let cfg = NemotronPreprocessing::default();
    let mut bad = 0;
    for c in 0..3 {
        for v in 0..256usize {
            let got = (v as f32 / 255.0 - cfg.norm_mean[c]) / cfg.norm_std[c];
            bad += (got.to_bits() != want[c * 256 + v].to_bits()) as usize;
        }
    }
    assert_eq!(bad, 0, "(v / 255 - mean) / std in float32 differs from torch's in {bad} of 768 entries");
}

/// Each fixture image through decode (alpha dropped) and the preprocessing; the processor's tensor sampled on a lattice (computed once).
fn image_results() -> &'static Vec<(String, f64, usize, bool)> {
    static RESULTS: std::sync::OnceLock<Vec<(String, f64, usize, bool)>> = std::sync::OnceLock::new();
    RESULTS.get_or_init(compute_image_results)
}

fn compute_image_results() -> Vec<(String, f64, usize, bool)> {
    let fx = fixture("images.json");
    let table = fixture("table.json");
    let mut out = Vec::new();
    for im in fx["images"].as_array().unwrap() {
        let name = im["file"].as_str().unwrap().to_string();
        let bytes = std::fs::read(root().join("vision").join(&name)).unwrap();
        let src_hash: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(src_hash, im["source_sha256"].as_str().unwrap(), "{name}: not the file the fixture was made from");
        let (cfg, _, _) = config_named(&table, im["config"].as_str().unwrap());
        let rgb = decode_rgb(&bytes, AlphaPolicy::Drop).unwrap();
        assert_eq!([rgb.width, rgb.height], [im["size"][0].as_u64().unwrap() as usize, im["size"][1].as_u64().unwrap() as usize]);
        let p = cfg.preprocess(&rgb).unwrap();
        assert_eq!(
            (p.w_patches, p.h_patches, p.tokens),
            (
                im["w_patches"].as_u64().unwrap() as usize,
                im["h_patches"].as_u64().unwrap() as usize,
                im["tokens"].as_u64().unwrap() as usize
            ),
            "{name}: tile"
        );
        let (cw, ch) = p.canvas();
        assert_eq!([cw, ch], [im["canvas"][0].as_u64().unwrap() as usize, im["canvas"][1].as_u64().unwrap() as usize]);
        assert_eq!(p.pixels.len(), 3 * cw * ch);
        // the lattice
        let l = &im["lattice"];
        let (sy, sx, ny, nx) = (
            l["step_y"].as_u64().unwrap() as usize,
            l["step_x"].as_u64().unwrap() as usize,
            l["ny"].as_u64().unwrap() as usize,
            l["nx"].as_u64().unwrap() as usize,
        );
        let axis = |n: usize, step: usize| -> Vec<usize> {
            let mut v: Vec<usize> = (0..n).step_by(step).collect();
            if *v.last().unwrap() != n - 1 {
                v.push(n - 1);
            }
            v
        };
        let (ys, xs) = (axis(ch, sy), axis(cw, sx));
        assert_eq!((ys.len(), xs.len()), (ny, nx), "{name}: lattice shape");
        let want = b64_f32(&l["f32_le_b64"]);
        assert_eq!(want.len(), 3 * ny * nx);
        let mut max_abs = 0f64;
        let mut unequal = 0usize;
        for c in 0..3 {
            for (iy, &y) in ys.iter().enumerate() {
                for (ix, &x) in xs.iter().enumerate() {
                    let g = p.pixels[c * cw * ch + y * cw + x];
                    let w = want[(c * ny + iy) * nx + ix];
                    max_abs = max_abs.max((g as f64 - w as f64).abs());
                    unequal += (g.to_bits() != w.to_bits()) as usize;
                }
            }
        }
        // the whole tensor: its per-channel sums (float64, numpy's pairwise order against ours: a relative 1e-9) and its SHA-256
        for c in 0..3 {
            let s: f64 = p.pixels[c * cw * ch..(c + 1) * cw * ch].iter().map(|&v| v as f64).sum();
            let w = im["sum_f64"][c].as_f64().unwrap();
            assert!(
                (s - w).abs() <= 1e-9 * w.abs().max(1.0) * (cw * ch) as f64 / 1e4 + 1e-6,
                "{name} channel {c}: sum {s} vs {w}"
            );
        }
        let mut h = Sha256::new();
        for v in &p.pixels {
            h.update(v.to_le_bytes());
        }
        let sha: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        let exact = sha == im["tensor_sha256"].as_str().unwrap();
        // the extremes
        for c in 0..3 {
            let (mn, mx) =
                p.pixels[c * cw * ch..(c + 1) * cw * ch].iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            assert!((mn as f64 - im["min"][c].as_f64().unwrap()).abs() <= 1e-5, "{name}: min of channel {c}");
            assert!((mx as f64 - im["max"][c].as_f64().unwrap()).abs() <= 1e-5, "{name}: max of channel {c}");
        }
        eprintln!(
            "{name:26} {}x{} -> {cw}x{ch} ({} tokens) config {:8} lattice {}x{}x3: max abs {max_abs:.3e}, {unequal} values differ in a bit, whole tensor sha256 {}",
            rgb.width,
            rgb.height,
            p.tokens,
            im["config"].as_str().unwrap(),
            ny,
            nx,
            if exact { "identical" } else { "differs" }
        );
        out.push((name, max_abs, unequal, exact));
    }
    out
}

#[test]
fn pixel_tensors_are_the_processors_within_1e_5() {
    let res = image_results();
    assert!(res.len() >= 4);
    let worst = res.iter().map(|r| r.1).fold(0.0, f64::max);
    eprintln!("{} images, worst max abs over the lattices: {worst:.3e}", res.len());
    for (name, max_abs, _, _) in res {
        assert!(*max_abs <= 1e-5, "{name}: max abs {max_abs:e} against the processor's pixel_values");
    }
}

#[test]
fn the_resize_is_torchs_bit_for_bit_on_the_fixture_images() {
    // the committed gate is 1e-5 (the test above); this one pins what was measured: every sampled value and every whole tensor is identical,
    // because the kernel here follows torch 2.14.0's float32 operation order (see the module documentation of `nemotron`)
    for (name, _, unequal, exact) in image_results().iter() {
        assert_eq!(*unequal, 0, "{name}: sampled values that differ in a bit");
        assert!(exact, "{name}: the tensor's SHA-256 differs from the processor's");
    }
}

#[test]
fn the_tile_is_even_in_bounds_pure_and_idempotent() {
    // a deterministic stream of (w, h): log-uniform sides over 1..=8000, a third with a tiny side
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let side =
        |next: &mut dyn FnMut() -> u64| -> usize { ((next() % 1_000_000) as f64 / 1e6 * 8000f64.ln()).exp().max(1.0) as usize };
    let mut cases: Vec<(usize, usize)> = Vec::new();
    for i in 0..6000 {
        let (w, h) = (side(&mut next), side(&mut next));
        cases.push(match i % 3 {
            0 => (w, h),
            1 => (w, 1 + (next() % 24) as usize),
            _ => (1 + (next() % 24) as usize, h),
        });
    }
    let cfg = NemotronPreprocessing::default();
    for &(w, h) in &cases {
        let t = cfg.plan_for(w, h, 1).unwrap_or_else(|e| panic!("{w}x{h}: {e}"));
        assert_eq!(t, cfg.plan_for(w, h, 1).unwrap(), "{w}x{h}: not a pure function");
        assert!(t.w_patches % 2 == 0 && t.h_patches % 2 == 0, "{w}x{h}: {t:?}");
        let patches = t.w_patches * t.h_patches;
        assert!((1024..=TOWER_MAX_PATCHES).contains(&patches), "{w}x{h}: {patches} patches");
        assert_eq!(t.tokens * 4, patches);
        assert!(t.tokens <= MAX_IMAGE_TOKENS);
        // an image that already has its tile's size keeps the tile
        let (cw, ch) = t.canvas();
        assert_eq!(cfg.plan_for(cw, ch, 1).unwrap(), t, "{w}x{h}: tile {cw}x{ch} is not a fixed point");
    }
    // a small window: the count is bounded by the budget unless the short side is already the 2-patch minimum, and a tile past the pack's minimum
    // only comes from the scale-up, which runs only above 1024
    for mml in [260usize, 1028, 2048, 8192] {
        let small = NemotronPreprocessing { max_model_len: mml, ..cfg.clone() };
        let budget = per_image_budget(&small, 1);
        for &(w, h) in &cases {
            let t = small.plan_for(w, h, 1).unwrap_or_else(|e| panic!("{w}x{h} at {mml}: {e}"));
            let patches = t.w_patches * t.h_patches;
            assert!(t.w_patches % 2 == 0 && t.h_patches % 2 == 0);
            assert!(
                patches <= budget || t.w_patches.min(t.h_patches) == 2,
                "{w}x{h} at {mml}: {patches} patches over a budget of {budget}"
            );
            if budget > 1024 {
                assert!(patches >= 1024.min(budget), "{w}x{h} at {mml}: {patches} patches under the minimum");
            }
            let (cw, ch) = t.canvas();
            assert_eq!(small.plan_for(cw, ch, 1).unwrap(), t, "{w}x{h} at {mml}: not a fixed point");
        }
    }
    // video frames: an even tile close to 1024 patches for a frame that is not extreme
    for (w, h) in [(1920usize, 1080usize), (1080, 1920), (640, 480), (224, 224), (3840, 2160), (1000, 700)] {
        let t = cfg.plan_video(w, h).unwrap();
        let patches = t.w_patches * t.h_patches;
        assert!(t.w_patches % 2 == 0 && t.h_patches % 2 == 0);
        assert!((870..=1090).contains(&patches), "{w}x{h}: {patches} patches");
    }
}

#[test]
fn a_larger_image_is_bounded_by_the_towers_token_limit() {
    let cfg = NemotronPreprocessing::default();
    for (w, h) in [(8000usize, 8000usize), (7680, 4320), (8000, 100), (100, 8000), (30_000, 30_000)] {
        let t = cfg.plan_for(w, h, 1).unwrap();
        assert!(t.tokens <= MAX_IMAGE_TOKENS, "{w}x{h}: {} tokens", t.tokens);
    }
    // no image takes more than the limit; the largest ones take nearly all of it
    assert!(cfg.plan_for(8000, 8000, 1).unwrap().tokens > 3_000);
}

#[test]
fn the_output_does_not_depend_on_the_thread_count() {
    let rgb = decode_rgb(&std::fs::read(root().join("vision/rings_333x517.png")).unwrap(), AlphaPolicy::Drop).unwrap();
    let base = NemotronPreprocessing { threads: 1, ..NemotronPreprocessing::default() }.preprocess(&rgb).unwrap();
    for t in [2usize, 3, 8, 64] {
        let p = NemotronPreprocessing { threads: t, ..NemotronPreprocessing::default() }.preprocess(&rgb).unwrap();
        assert_eq!(p, base, "{t} threads");
    }
    let p = NemotronPreprocessing::default().preprocess(&rgb).unwrap();
    assert_eq!(p, base);
}

#[test]
fn the_trait_agrees_with_the_inherent_api_and_names_what_it_carries() {
    let cfg = NemotronPreprocessing::default();
    let pp: &dyn ImagePreprocessor = &cfg;
    assert_eq!(pp.family(), "nemotron-h-omni");
    assert_eq!(pp.id(), cfg.identity());
    let bytes = std::fs::read(root().join("vision/check_512x512.png")).unwrap();
    let rgb = pp.decode(&bytes).unwrap();
    let plan = pp.plan(rgb.width, rgb.height).unwrap();
    assert_eq!((plan.canvas, plan.grid, plan.tokens), ((512, 512), (1, 32, 32), 256));
    let pre = pp.preprocess(&rgb).unwrap();
    let f = cfg.preprocess(&rgb).unwrap();
    assert_eq!(pre.patches, f.bf16_bits());
    assert_eq!(pre.pixel_hash, f.pixel_hash);
    assert_eq!(pre.size, plan.canvas);
}
