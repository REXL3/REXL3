//! GLM-5.3-Flash preprocessing against the real HuggingFace `Glm5NextProcessor` and the ExLlamaV3 fork (`tools/gen_glm_vision_fixtures.py`,
//! `testdata/vision/glm`). Both references are recorded for every image. The Rust profile `Fork` must equal the fork byte for byte, in the resized
//! canvas, the fp16 tensor, the grid and the placeholder count. The profile `Hf` must equal the HF processor (torchvision backend, torch's own uint8
//! resize) byte for byte, in the canvas, the float32 `pixel_values`, their fp16 cast, the grid and the placeholder count, for every image resized or not.

use gb_vision::glm::{still_patches, GlmPreprocessing, Profile};
use gb_vision::preprocessor::ImagePreprocessor;
use gb_vision::{resample, torch_resample, Rgb, VisionError};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision/glm")
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sha(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

fn json(name: &str) -> Value {
    let t = std::fs::read_to_string(dir().join(name))
        .unwrap_or_else(|e| panic!("{name}: {e} (run tools/gen_glm_vision_fixtures.py)"));
    serde_json::from_str(&t).unwrap()
}

fn cases() -> Vec<Value> {
    json("fixtures.json")["cases"].as_array().unwrap().clone()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
}

/// The deterministic hash pattern of `pattern()` in the generator (wrapping u32 arithmetic): kind 0 noise, kind 1 hard 3-pixel checker.
fn pattern(w: usize, h: usize, kind: u64, seed: u64) -> Vec<u8> {
    let salt = ((seed * 2_654_435_761) % (1 << 32)) as u32;
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h as u32 {
        for x in 0..w as u32 {
            for c in 0..3u32 {
                out.push(if kind == 0 {
                    let mut v = x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663) ^ c.wrapping_mul(83_492_791) ^ salt;
                    v ^= v >> 13;
                    v = v.wrapping_mul(0x5BD1_E995);
                    v ^= v >> 15;
                    ((v >> 8) & 255) as u8
                } else if ((x / 3 + y / 3 + c + seed as u32) & 1) == 1 {
                    255
                } else {
                    0
                });
            }
        }
    }
    out
}

/// The generator's `procedural()` sources.
fn procedural(kind: &str, w: usize, h: usize, seed: u64) -> Rgb {
    let data = match kind {
        "gradient" => (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| [x * 255 / (w - 1).max(1), y * 255 / (h - 1).max(1), (x + y) * 255 / (w + h - 2).max(1)])
            })
            .map(|v| v as u8)
            .collect(),
        "checker" => (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    let c = (((x / 8) + (y / 8)) % 2) * 255;
                    [c, 255 - c, (x * 7 + y * 3) % 256]
                })
            })
            .map(|v| v as u8)
            .collect(),
        "hash" => pattern(w, h, 0, seed),
        "hashchecker" => pattern(w, h, 1, seed),
        k => panic!("unknown procedural kind {k}"),
    };
    Rgb { width: w, height: h, data }
}

/// The source pixels of a case as the Rust side obtains them, and whether they come from a JPEG decode we do not claim to reproduce.
fn source(p: &GlmPreprocessing, c: &Value) -> Rgb {
    if let Some(g) = c["gen"].as_object() {
        return procedural(
            g["kind"].as_str().unwrap(),
            g["w"].as_u64().unwrap() as usize,
            g["h"].as_u64().unwrap() as usize,
            g["seed"].as_u64().unwrap(),
        );
    }
    // a JPEG is checked from Pillow's own decode, so the pipeline after the decoder is exact; the decoder has its own test
    let file = c["pil_decoded_file"].as_str().or(c["file"].as_str()).unwrap();
    let bytes = std::fs::read(dir().join(file)).unwrap();
    p.decode(&bytes).unwrap_or_else(|e| panic!("{}: {e}", c["name"]))
}

/// Every check of one fixture image against the HF processor through the `Hf` profile; the messages of what differs (none when it is exact).
fn check_hf(c: &Value) -> Vec<String> {
    let name = c["name"].as_str().unwrap();
    let p = GlmPreprocessing::glm53_flash(Profile::Hf);
    let hf = &c["hf"];
    let mut bad = vec![];
    let img = source(&p, c);
    if sha(&img.data) != hf["rgb_sha256"] {
        bad.push(format!("{name}: RGB conversion differs from image.convert(\"RGB\")"));
    }
    let g = p.geometry(img.width, img.height).unwrap();
    let want = |k: &str| usizes(&hf[k]);
    if [g.canvas.0, g.canvas.1] != want("canvas_size")[..] {
        bad.push(format!("{name}: canvas {:?}, HF {:?}", g.canvas, want("canvas_size")));
    }
    if [g.content.0, g.content.1] != want("content_size")[..] {
        bad.push(format!("{name}: content {:?}, HF {:?}", g.content, want("content_size")));
    }
    if [g.grid.0, g.grid.1, g.grid.2] != want("grid")[..] || g.tokens != hf["n_tokens"].as_u64().unwrap() as usize {
        bad.push(format!("{name}: grid/tokens {:?}/{}, HF {:?}/{}", g.grid, g.tokens, want("grid"), hf["n_tokens"]));
    }
    let (canvas, _) = p.canvas_rgb(&img).unwrap();
    let (f32s, _) = p.preprocess_f32(&img).unwrap();
    let pre = p.preprocess(&img).unwrap();
    let resized = g.content != (img.width, img.height);
    if hf["resized"].as_bool() != Some(resized) {
        bad.push(format!("{name}: resized is {resized} here, the fixture says {}", hf["resized"]));
    }
    // every byte of every tensor is the HF processor's (its torchvision backend, torch's uint8 resize), resized or not
    if sha(&canvas.data) != hf["canvas_sha256"] {
        bad.push(format!("{name}: resized + padded canvas differs from HF's"));
    }
    if sha(&f32_bytes(&f32s)) != hf["f32_sha256"] {
        bad.push(format!("{name}: float32 pixel_values differ from HF's"));
    }
    if sha(&f16_bytes(&pre.patches)) != hf["f16_sha256"] {
        bad.push(format!("{name}: fp16 patches differ from HF pixel_values.half()"));
    }
    // HF's own second backend (`Glm5NextImageProcessorPil`, what a CPU-only HF install runs: Pillow resize, HF's normalisation): this profile's fp16
    // tensor equals it exactly where HF's two backends agree with each other, and differs from it where they do not
    let pil_backend = &c["hf_pil_backend"];
    let backends_agree = pil_backend["same_f16_as_hf"].as_bool().unwrap();
    if (sha(&f16_bytes(&pre.patches)) == pil_backend["f16_sha256"]) != backends_agree {
        bad.push(format!(
            "{name}: equality with HF's PIL backend is {}, HF's two backends agree: {backends_agree}",
            !backends_agree
        ));
    }
    // the Pillow variant of HF's pixels (`pil_*`): the canvas equals it exactly where the recorded deviation of Pillow's resize from torch's is
    // zero, so the equalities above are the torch kernel's doing wherever the two kernels differ. Nothing resized: no deviation at all
    let dev = &hf["pil_vs_hf"];
    if hf["pil_f16_sha256"] != pil_backend["f16_sha256"] {
        bad.push(format!("{name}: the Pillow variant of HF's pixels is not what HF's own PIL backend produces (the fixture)"));
    }
    if (sha(&canvas.data) == hf["pil_canvas_sha256"]) != (dev["n_diff"].as_u64() == Some(0)) {
        bad.push(format!("{name}: the canvas and the Pillow variant disagree with the recorded deviation {dev}"));
    }
    if !resized && (dev["n_diff"].as_u64() != Some(0) || dev["f16_n_diff"].as_u64() != Some(0) || !backends_agree) {
        bad.push(format!("{name}: the fixture records a deviation for an image that is not resized: {dev}"));
    }
    // the fp16 tensor is the cast of the float32 one, and a still image repeats its frame across the temporal patch
    let cast: Vec<u16> = f32s.iter().map(|&v| gb_vision::f32_to_f16_bits(v)).collect();
    if cast != pre.patches {
        bad.push(format!("{name}: fp16 patches are not the cast of the float32 tensor"));
    }
    let half = still_patches(&pre, 2);
    if half.len() * 2 != pre.patches.len() {
        bad.push(format!("{name}: still_patches has the wrong length"));
    }
    if pre.size != g.canvas || pre.grid != g.grid || pre.patch_dim != 1176 || p.n_tokens(&pre) != g.tokens {
        bad.push(format!("{name}: Preprocessed disagrees with the geometry"));
    }
    bad
}

fn check_fork(c: &Value) -> Vec<String> {
    check_fork_with(&GlmPreprocessing::glm53_flash(Profile::Fork), c)
}

fn check_fork_with(p: &GlmPreprocessing, c: &Value) -> Vec<String> {
    let name = c["name"].as_str().unwrap();
    let fork = &c["fork"];
    let mut bad = vec![];
    let img = source(p, c);
    if sha(&img.data) != fork["rgb_sha256"] {
        bad.push(format!("{name}: RGB conversion differs from the fork's convert_to_rgb"));
    }
    let (canvas, g) = p.canvas_rgb(&img).unwrap();
    if sha(&canvas.data) != fork["canvas_sha256"] {
        bad.push(format!("{name}: canvas differs from the fork's"));
    }
    if [g.grid.0, g.grid.1, g.grid.2] != usizes(&c["hf"]["grid"])[..] {
        bad.push(format!("{name}: the fork and HF disagree on the grid (the generator asserts they do not)"));
    }
    let pre = p.preprocess(&img).unwrap();
    if sha(&f16_bytes(&pre.patches)) != fork["f16_sha256"] {
        bad.push(format!("{name}: fp16 patches differ from the fork's .half() tensor"));
    }
    bad
}

#[test]
fn hf_profile_matches_the_hf_processor_byte_for_byte() {
    let all = cases();
    assert!(all.len() >= 50, "{} cases", all.len());
    let bad: Vec<String> = all.iter().flat_map(check_hf).collect();
    assert!(bad.is_empty(), "{} mismatches:\n{}", bad.len(), bad.join("\n"));
    // the fixtures exercise both branches, with and without transparency, and torch's resize does differ from Pillow's in most resized images: the
    // equality above is not one that Pillow's kernel would also give
    let resized: Vec<&Value> = all.iter().filter(|c| c["hf"]["resized"].as_bool().unwrap()).collect();
    let not_resized = all.len() - resized.len();
    let alpha = resized.iter().filter(|c| c["agree"]["rgb"].as_bool() == Some(false)).count();
    let pillow_differs = resized.iter().filter(|c| c["hf"]["pil_vs_hf"]["n_diff"].as_u64().unwrap() > 0).count();
    eprintln!(
        "{} of {} images exact against HF (canvas, float32, fp16): {not_resized} not resized, {} resized ({alpha} with alpha dropped), {pillow_differs} of \
         the resized ones differ from the Pillow resize of the same pixels",
        all.len(),
        all.len(),
        resized.len()
    );
    assert!(
        not_resized >= 25 && resized.len() >= 30 && alpha >= 6 && pillow_differs >= 25,
        "{not_resized} not resized, {} resized, {alpha} alpha, {pillow_differs} where Pillow differs",
        resized.len()
    );
}

#[test]
fn fork_profile_matches_the_fork_byte_for_byte() {
    let mut n = 0;
    let mut bad = vec![];
    for c in cases() {
        if c["fork"].is_null() {
            eprintln!("{}: the fork cannot convert this image ({})", c["name"], c["fork_error"]);
            continue;
        }
        n += 1;
        bad.extend(check_fork(&c));
    }
    assert!(n >= 50, "only {n} cases have a fork reference");
    assert!(bad.is_empty(), "{} mismatches:\n{}", bad.len(), bad.join("\n"));
}

#[test]
fn the_default_preprocessing_is_the_fork_byte_for_byte() {
    // `GlmPreprocessing::default()` (no profile named) must be the fork's preprocess on every image, alpha and resized ones included
    let p = GlmPreprocessing::default();
    assert_eq!(p.profile, Profile::Fork);
    let (mut n, mut bad) = (0, vec![]);
    for c in cases() {
        if c["fork"].is_null() {
            continue;
        }
        n += 1;
        bad.extend(check_fork_with(&p, &c));
    }
    assert!(n >= 50, "only {n} cases have a fork reference");
    assert!(bad.is_empty(), "{} mismatches:\n{}", bad.len(), bad.join("\n"));
}

#[test]
fn the_two_profiles_differ_exactly_where_the_references_do() {
    // `agree` in the fixtures says whether the references coincide on the RGB conversion, the canvas and the fp16 tensor; the Rust profiles must
    // coincide on exactly those cases (a profile switch that does nothing would pass both tests above on images where the references agree)
    let (hf, fork) = (GlmPreprocessing::glm53_flash(Profile::Hf), GlmPreprocessing::glm53_flash(Profile::Fork));
    let (mut rgb_differs, mut canvas_differs, mut f16_differs) = (0, 0, 0);
    for c in cases() {
        if c["fork"].is_null() {
            continue;
        }
        let name = c["name"].as_str().unwrap();
        let ag = &c["agree"];
        let img_hf = source(&hf, &c);
        let img_fork = source(&fork, &c);
        assert_eq!(img_hf.data == img_fork.data, ag["rgb"].as_bool().unwrap(), "{name}: RGB agreement");
        let (a, _) = hf.canvas_rgb(&img_hf).unwrap();
        let (b, _) = fork.canvas_rgb(&img_fork).unwrap();
        assert_eq!(a.data == b.data, ag["canvas"].as_bool().unwrap(), "{name}: canvas agreement");
        // the fork resizes with Pillow: its canvas is the Pillow variant of HF's pixels exactly where the RGB conversions agree
        assert_eq!(
            sha(&b.data) == c["hf"]["pil_canvas_sha256"],
            ag["pil_canvas"].as_bool().unwrap(),
            "{name}: Fork canvas against the Pillow variant"
        );
        let (pa, pb) = (hf.preprocess(&img_hf).unwrap(), fork.preprocess(&img_fork).unwrap());
        let n16 = pa.patches.iter().zip(&pb.patches).filter(|(x, y)| x != y).count();
        assert_eq!(
            n16 as u64,
            ag["f16_n_diff"].as_u64().unwrap(),
            "{name}: number of fp16 elements that differ between the profiles"
        );
        rgb_differs += (img_hf.data != img_fork.data) as usize;
        canvas_differs += (a.data != b.data) as usize;
        f16_differs += (n16 > 0) as usize;
    }
    eprintln!("profiles differ in RGB for {rgb_differs}, in the canvas for {canvas_differs}, in fp16 for {f16_differs} images");
    assert!(rgb_differs >= 8 && canvas_differs >= 20 && f16_differs >= 50);
}

#[test]
fn normalisation_tables_equal_both_references_for_every_byte_value() {
    let t = json("norm_luts.json");
    let bits = |k: &str| -> Vec<u64> { t[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect() };
    for (profile, k32, k16) in [(Profile::Hf, "hf_f32_bits", "hf_f16_bits"), (Profile::Fork, "fork_f32_bits", "fork_f16_bits")] {
        let p = GlmPreprocessing::glm53_flash(profile);
        let (l32, l16) = (p.lut_f32(), p.lut_f16());
        let flat32: Vec<u64> = l32.iter().flatten().map(|v| v.to_bits() as u64).collect();
        let flat16: Vec<u64> = l16.iter().flatten().map(|&v| v as u64).collect();
        assert_eq!(flat32, bits(k32), "{profile:?}: float32 table");
        assert_eq!(flat16, bits(k16), "{profile:?}: fp16 table");
    }
    let p = GlmPreprocessing::glm53_flash(Profile::Hf);
    let m255: Vec<u64> = p.mean.iter().map(|m| (m * 255.0f32).to_bits() as u64).collect();
    let s255: Vec<u64> = p.std.iter().map(|s| (s * 255.0f32).to_bits() as u64).collect();
    assert_eq!(m255, bits("mean_x255_bits"));
    assert_eq!(s255, bits("std_x255_bits"));
    assert_eq!(t["n_f16_diff"].as_u64(), Some(3));
}

#[test]
fn torch_resize_matches_tvf_resize() {
    // `torchvision.transforms.v2.functional.resize` (bicubic, antialias) on a channels-last uint8 tensor, the layout the HF processors pass: the
    // generator records the sha256 of its result and of PIL's `Image.resize(BICUBIC)` for the same pixels
    let t = json("resize_cases.json");
    let rows = t["cases"].as_array().unwrap();
    assert!(rows.len() >= 180);
    let (mut bad, mut differ) = (vec![], 0);
    for r in rows {
        let (src, dst) = (usizes(&r["src"]), usizes(&r["dst"]));
        let img = pattern(src[0], src[1], r["kind"].as_u64().unwrap(), r["seed"].as_u64().unwrap());
        let torch = torch_resample::bicubic_rgb_torch(&img, src[0], src[1], dst[0], dst[1]);
        let pil = resample::bicubic_rgb_pil(&img, src[0], src[1], dst[0], dst[1]);
        if sha(&torch) != r["torch_sha256"] {
            bad.push(format!(
                "{src:?} -> {dst:?}: torch kernel differs from tvF.resize ({} samples differ between tvF.resize and PIL)",
                r["n_diff"]
            ));
        }
        if sha(&pil) != r["pil_sha256"] {
            bad.push(format!("{src:?} -> {dst:?}: Pillow kernel differs from PIL"));
        }
        // the two kernels differ exactly where the two references do
        assert_eq!(
            torch != pil,
            r["n_diff"].as_u64().unwrap() > 0,
            "{src:?} -> {dst:?}: the kernels must differ where tvF.resize and PIL do"
        );
        differ += (torch != pil) as usize;
    }
    eprintln!(
        "{} resize shapes: the torch kernel equals tvF.resize in all but {}, and differs from the Pillow kernel in {differ}",
        rows.len(),
        bad.len()
    );
    assert!(bad.is_empty(), "{} mismatches:\n{}", bad.len(), bad.join("\n"));
    assert!(differ > 100, "the fixtures must exercise the difference between the kernels");
}

#[test]
fn pillow_resize_matches_pil() {
    let t = json("resize_cases.json");
    let rows = t["cases"].as_array().unwrap();
    assert!(rows.len() >= 180);
    let mut bad = vec![];
    for r in rows {
        let (src, dst) = (usizes(&r["src"]), usizes(&r["dst"]));
        let img = pattern(src[0], src[1], r["kind"].as_u64().unwrap(), r["seed"].as_u64().unwrap());
        let pil = resample::bicubic_rgb_pil(&img, src[0], src[1], dst[0], dst[1]);
        if sha(&pil) != r["pil_sha256"] {
            bad.push(format!("{src:?} -> {dst:?}: Pillow kernel differs from PIL"));
        }
    }
    assert!(bad.is_empty(), "{} mismatches:\n{}", bad.len(), bad.join("\n"));
    // how far torch's uint8 resize is from Pillow's on these shapes (measured by the generator; not a property of this code): the figures
    // `docs/VISION.md` quotes
    let differing = rows.iter().filter(|r| r["n_diff"].as_u64().unwrap() > 0).count();
    let worst = rows.iter().map(|r| r["max_abs_diff"].as_u64().unwrap()).max().unwrap();
    assert_eq!((rows.len(), differing, worst), (190, 167, 3), "the documentation states these: update it");
}

/// All 190 resize cases through the torch kernel on exact thread counts (the default policy would run most of these inline): tvF.resize's bytes
/// however the rows of both passes are split.
#[test]
fn torch_resize_matches_tvf_resize_on_every_thread_count() {
    let t = json("resize_cases.json");
    let rows = t["cases"].as_array().unwrap();
    let mut bad = vec![];
    for r in rows {
        let (src, dst) = (usizes(&r["src"]), usizes(&r["dst"]));
        let img = pattern(src[0], src[1], r["kind"].as_u64().unwrap(), r["seed"].as_u64().unwrap());
        for threads in [1usize, 2, 3, 5, 8, 16, 64] {
            let out = torch_resample::bicubic_rgb_torch_threads(&img, src[0], src[1], dst[0], dst[1], threads);
            if sha(&out) != r["torch_sha256"] {
                bad.push(format!("{src:?} -> {dst:?} on {threads} threads differs from tvF.resize"));
            }
        }
        for cap in [0usize, 3] {
            let out = torch_resample::bicubic_rgb_torch_with(&img, src[0], src[1], dst[0], dst[1], cap);
            if sha(&out) != r["torch_sha256"] {
                bad.push(format!("{src:?} -> {dst:?} with a cap of {cap} differs from tvF.resize"));
            }
        }
    }
    assert!(rows.len() >= 180 && bad.is_empty(), "{} cases, {} mismatches:\n{}", rows.len(), bad.len(), bad.join("\n"));
}

/// All 190 resize cases of `resize_cases.json` (1 to 20,000 pixels a side, the two-step strips included) on exact thread counts: PIL's bytes however
/// the rows of both passes are split (the default policy would run most of these inline).
#[test]
fn pillow_resize_matches_pil_on_every_thread_count() {
    let t = json("resize_cases.json");
    let rows = t["cases"].as_array().unwrap();
    let mut bad = vec![];
    for r in rows {
        let (src, dst) = (usizes(&r["src"]), usizes(&r["dst"]));
        let img = pattern(src[0], src[1], r["kind"].as_u64().unwrap(), r["seed"].as_u64().unwrap());
        for threads in [1usize, 2, 3, 5, 8, 16, 64] {
            let out = resample::bicubic_rgb_pil_threads(&img, src[0], src[1], dst[0], dst[1], threads);
            if sha(&out) != r["pil_sha256"] {
                bad.push(format!("{src:?} -> {dst:?} on {threads} threads differs from PIL"));
            }
        }
        for cap in [0usize, 3] {
            let out = resample::bicubic_rgb_pil_with(&img, src[0], src[1], dst[0], dst[1], cap);
            if sha(&out) != r["pil_sha256"] {
                bad.push(format!("{src:?} -> {dst:?} with a cap of {cap} differs from PIL"));
            }
        }
    }
    assert!(rows.len() >= 180 && bad.is_empty(), "{} cases, {} mismatches:\n{}", rows.len(), bad.len(), bad.join("\n"));
}

fn budget(min: usize, max: usize) -> GlmPreprocessing {
    GlmPreprocessing {
        min_image_tokens: min,
        max_image_tokens: max,
        // the sweep compares the canvas arithmetic with HF, which has no size limit
        max_pixels: 1 << 40,
        max_side: 1 << 20,
        ..GlmPreprocessing::default()
    }
}

#[test]
fn canvas_content_and_grid_match_hf_over_the_size_sweep() {
    let t = json("canvas_sweep.json");
    let mut total = 0;
    let mut bad = vec![];
    for s in t["sweeps"].as_array().unwrap() {
        let (min, max) = (s["min_image_tokens"].as_u64().unwrap() as usize, s["max_image_tokens"].as_u64().unwrap() as usize);
        let p = budget(min, max);
        let rows = usizes(&s["rows"]);
        assert_eq!(rows.len(), s["n"].as_u64().unwrap() as usize * 6);
        for r in rows.chunks_exact(6) {
            let (h, w) = (r[0], r[1]);
            total += 1;
            match p.geometry(w, h) {
                Ok(g) => {
                    if (g.grid.1, g.grid.2, g.content.1, g.content.0) != (r[2], r[3], r[4], r[5]) || g.tokens != r[2] * r[3] / 4 {
                        bad.push(format!(
                            "budget {min}..{max}, {h}x{w}: got grid {:?} content {:?}, HF grid ({}, {}) content ({}, {})",
                            g.grid, g.content, r[2], r[3], r[4], r[5]
                        ));
                    }
                }
                Err(e) => bad.push(format!("budget {min}..{max}, {h}x{w}: {e}")),
            }
        }
    }
    eprintln!("{total} sizes compared with HF's canvas and content");
    assert!(total > 15_000);
    assert!(
        bad.is_empty(),
        "{} of {total} differ, e.g.\n{}",
        bad.len(),
        bad.iter().take(10).cloned().collect::<Vec<_>>().join("\n")
    );
}

#[test]
fn token_counts_match_the_slice_2_sweep() {
    // glm53_image_count_cases.json from branch glm53-slice2-chatformat (HF `get_number_of_image_patches` and the processor's grid)
    let t = json("slice2_image_count_cases.json");
    let p = GlmPreprocessing::default();
    let cfg = &t["processor"];
    assert_eq!((cfg["patch_size"].as_u64(), cfg["max_image_tokens"].as_u64()), (Some(14), Some(8000)));
    let mut n = 0;
    for r in t["cases"].as_array().unwrap() {
        let (h, w) = (r["height"].as_u64().unwrap() as usize, r["width"].as_u64().unwrap() as usize);
        let g = p.geometry(w, h).unwrap();
        assert_eq!(g.grid.1 * g.grid.2, r["patches"].as_u64().unwrap() as usize, "{h}x{w}: patches");
        assert_eq!(g.tokens, r["tokens"].as_u64().unwrap() as usize, "{h}x{w}: tokens");
        if let Some(grid) = r["grid"].as_array() {
            assert_eq!(vec![g.grid.0, g.grid.1, g.grid.2], usizes(&Value::Array(grid.clone())), "{h}x{w}: grid");
        }
        n += 1;
    }
    assert!(n >= 40);
}

#[test]
fn the_shipped_processor_config_parses_to_the_defaults() {
    let cfg = json("processor_config.json");
    for profile in [Profile::Hf, Profile::Fork] {
        assert_eq!(GlmPreprocessing::from_processor_config(&cfg, profile).unwrap(), GlmPreprocessing::glm53_flash(profile));
    }
    assert_eq!(GlmPreprocessing::from_processor_config(&cfg, Profile::default()).unwrap(), GlmPreprocessing::default());
}

#[test]
fn a_lower_token_budget_follows_the_hf_override() {
    // the server may cap an image below the model's 8000 tokens; HF accepts `max_image_tokens` as a keyword and the sweep covers 4096 / 1024 / 2048
    let t = json("canvas_sweep.json");
    let caps: Vec<u64> = t["sweeps"].as_array().unwrap().iter().map(|s| s["max_image_tokens"].as_u64().unwrap()).collect();
    assert_eq!(caps, [8000, 4096, 1024, 2048]);
    let p = GlmPreprocessing::default().with_max_image_tokens(1024).unwrap();
    assert!(p.tokens(4032, 3024).unwrap() <= 1024);
    assert!(GlmPreprocessing::default().with_max_image_tokens(8).is_err(), "a cap below the 16-token minimum is a config error");
}

#[test]
fn jpeg_decode_is_close_to_pillow_and_orientation_is_applied() {
    // zune-jpeg is not libjpeg-turbo: the pipeline is exact from the same pixels (above); the decode itself differs by a few levels
    let p = GlmPreprocessing::default();
    let mut worst = 0u8;
    let mut n = 0;
    for c in cases() {
        if !c["jpeg"].as_bool().unwrap() {
            continue;
        }
        let name = c["name"].as_str().unwrap();
        let ours =
            p.decode(&std::fs::read(dir().join(c["file"].as_str().unwrap())).unwrap()).unwrap_or_else(|e| panic!("{name}: {e}"));
        let pil = p.decode(&std::fs::read(dir().join(c["pil_decoded_file"].as_str().unwrap())).unwrap()).unwrap();
        assert_eq!((ours.width, ours.height), (pil.width, pil.height), "{name}: decoded size (EXIF orientation)");
        let d = ours.data.iter().zip(&pil.data).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        let n_diff = ours.data.iter().zip(&pil.data).filter(|(a, b)| a != b).count();
        eprintln!("{name}: {n_diff} of {} samples differ from Pillow's decode, by at most {d}/255", ours.data.len());
        worst = worst.max(d);
        n += 1;
    }
    assert!(n >= 8);
    // measured: 1 to 4 on the photographic-style and gradient cases, 10 on a synthetic 4:2:0 image of concentric colour rings (chroma upsampling at
    // hard chroma edges); a bound that catches a wrong colour matrix or a broken IDCT, not a claim of bit-exactness
    assert!(worst <= 12, "JPEG decode differs from libjpeg-turbo by {worst}");
}

#[test]
fn trait_objects_serve_both_families_and_the_qwen_path_is_unchanged() {
    // Qwen through the trait is the existing free functions
    let q: Box<dyn ImagePreprocessor> = Box::new(gb_vision::Preprocessing::qwen38());
    let qdir = dir().join("..");
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(qdir.join("fixtures.json")).unwrap()).unwrap();
    let mut n = 0;
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        if name.starts_with("jpeg") {
            continue;
        }
        let img = q.decode(&std::fs::read(qdir.join(c["file"].as_str().unwrap())).unwrap()).unwrap();
        let plan = q.plan(img.width, img.height).unwrap();
        let pre = q.preprocess(&img).unwrap();
        assert_eq!(plan.grid, pre.grid, "{name}");
        assert_eq!(plan.tokens, c["n_patches"].as_u64().unwrap() as usize / 4, "{name}");
        assert_eq!(plan.canvas, pre.size, "{name}");
        assert_eq!(
            sha(&f16_bytes(&pre.patches)),
            c["patches_sha256"].as_str().unwrap(),
            "{name}: the Qwen tensor through the trait"
        );
        n += 1;
    }
    assert!(n >= 10);
    // the trait object of the default preprocessing is the fork's
    let g: Box<dyn ImagePreprocessor> = Box::new(GlmPreprocessing::default());
    let plan = g.plan(4032, 3024).unwrap();
    assert_eq!((plan.canvas, plan.grid, plan.tokens), ((2884, 2156), (1, 154, 206), 7931));
    assert!(g.id().contains(":fork:"), "{}", g.id());
    let ids = [
        q.id(),
        g.id(),
        GlmPreprocessing::glm53_flash(Profile::Hf).id(),
        GlmPreprocessing::default().with_max_image_tokens(4096).unwrap().id(),
    ];
    for i in 0..ids.len() {
        for j in 0..i {
            assert_ne!(ids[i], ids[j], "preprocessing ids must separate every input-changing parameter");
        }
    }
    // decoding through the trait object follows the default profile's alpha rule (white), not HF's (dropped)
    let rgba = std::fs::read(dir().join("rgba_ramp_64x48.png")).unwrap();
    assert_eq!(g.decode(&rgba).unwrap(), GlmPreprocessing::glm53_flash(Profile::Fork).decode(&rgba).unwrap());
    assert_ne!(g.decode(&rgba).unwrap(), GlmPreprocessing::glm53_flash(Profile::Hf).decode(&rgba).unwrap());
    assert_eq!((q.family(), g.family()), ("qwen3.8-vl", "glm5-next"));
}

fn png_chunk(out: &mut Vec<u8>, t: &[u8; 4], d: &[u8]) {
    out.extend((d.len() as u32).to_be_bytes());
    out.extend(t);
    out.extend(d);
    let mut crc = 0xffff_ffffu32;
    for &b in t.iter().chain(d) {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 == 1 { (crc >> 1) ^ 0xedb8_8320 } else { crc >> 1 };
        }
    }
    out.extend((!crc).to_be_bytes());
}

/// A PNG with the given IHDR and one IDAT chunk of `idat` bytes (not a decodable image: a header for the limit checks).
fn header_only_png(w: u32, h: u32, color_type: u8, bit_depth: u8, idat: &[u8]) -> Vec<u8> {
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr = Vec::new();
    ihdr.extend(w.to_be_bytes());
    ihdr.extend(h.to_be_bytes());
    ihdr.extend([bit_depth, color_type, 0, 0, 0]);
    png_chunk(&mut out, b"IHDR", &ihdr);
    png_chunk(&mut out, b"IDAT", idat);
    png_chunk(&mut out, b"IEND", &[]);
    out
}

#[test]
fn oversized_empty_and_broken_files_are_refused_before_any_allocation() {
    let p = GlmPreprocessing::default();
    let msg = |bytes: &[u8]| match p.decode(bytes) {
        Err(e) => e.to_string(),
        Ok(i) => panic!("accepted a {}x{} image", i.width, i.height),
    };
    // a 100,000 x 100,000 header (10 Gpixel) must not reach the allocation of its pixel buffer
    assert!(msg(&header_only_png(100_000, 100_000, 2, 8, &[0x78, 0x9c, 0x03, 0x00])).contains("decoded size limit"));
    // zero dimensions are invalid PNG
    assert!(msg(&header_only_png(0, 10, 2, 8, &[0x78, 0x9c, 0x03, 0x00])).contains("png"));
    // 10 x 10 claiming data that is not there
    assert!(msg(&header_only_png(10, 10, 2, 8, &[0x78, 0x9c, 0x03, 0x00])).contains("png"));
    // (since E4, 617eb3e4, the decode reads WebP: a RIFF/WEBP header is a WebP whose sizes are checked, and the refusal names the third format)
    assert!(msg(b"RIFF....WEBPVP8 ").contains("webp: truncated file"), "{}", msg(b"RIFF....WEBPVP8 "));
    assert!(msg(b"").contains("only PNG, JPEG and WebP"));
    assert!(msg(&[0xff, 0xd8, 0xff, 0xe0]).contains("jpeg"));
    // a real PNG cut in half
    let real = std::fs::read(dir().join("rgba_ramp_64x48.png")).unwrap();
    assert!(msg(&real[..real.len() / 2]).contains("png"));
    // animated PNG (acTL and an fcTL before the default image's IDAT) is refused with its own message
    let mut head = real[..33].to_vec();
    png_chunk(&mut head, b"acTL", &[0, 0, 0, 2, 0, 0, 0, 0]);
    png_chunk(&mut head, b"fcTL", &[0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0]);
    head.extend_from_slice(&real[33..]);
    assert!(msg(&head).contains("animated"), "{}", msg(&head));
}

#[test]
fn errors_name_the_problem() {
    let p = GlmPreprocessing::default();
    let e = p.preprocess(&Rgb { width: 1 << 17, height: 1, data: vec![] }).unwrap_err();
    assert!(matches!(e, VisionError::Unsupported(ref m) if m.contains("side exceeds")), "{e}");
}

#[test]
fn preprocessing_is_deterministic_and_the_hash_covers_the_canvas() {
    let p = GlmPreprocessing::default();
    let img = procedural("hash", 150, 90, 3);
    let (a, b) = (p.preprocess(&img).unwrap(), p.preprocess(&img).unwrap());
    assert_eq!(a, b);
    let mut other = img.clone();
    other.data[1000] ^= 1;
    assert_ne!(a.pixel_hash, p.preprocess(&other).unwrap().pixel_hash);
    // the same pixels give the same canvas under both profiles when nothing is resized (the hash covers the canvas, so it is equal) and, when a
    // resize happens (60x40 is upscaled to the 112-pixel minimum canvas), different canvases: torch's kernel against Pillow's. The tensors always
    // differ, by the normalisation arithmetic at least
    let hf = GlmPreprocessing::glm53_flash(Profile::Hf);
    let small = procedural("hash", 60, 40, 4);
    assert_ne!(p.geometry(60, 40).unwrap().content, (60, 40));
    for (img, resized) in [(&img, false), (&small, true)] {
        let (f, h) = (p.preprocess(img).unwrap(), hf.preprocess(img).unwrap());
        assert_eq!(f.pixel_hash != h.pixel_hash, resized, "{}x{}", img.width, img.height);
        assert_ne!(f.patches, h.patches, "normalisation arithmetic differs between the profiles");
    }
}

/// One image of the pixel sweep through the `Hf` profile against the HF processor, byte for byte in the canvas, the float32 tensor and the fp16 tensor
/// (the fixture holds the three hashes of every image), and the grid. Returns (1 when the image is resized, 1 when it is resized and Pillow's resize
/// of the same pixels would differ from HF's).
fn check_hf_sweep_case(bad: &mut Vec<String>, what: &str, pre: &GlmPreprocessing, img: &Rgb, r: &Value) -> (usize, usize) {
    let (v, g) = pre.preprocess_f32(img).unwrap();
    let resized = g.content != (img.width, img.height);
    if r["resized"].as_bool() != Some(resized) {
        bad.push(format!("{what}: resized is {resized} here, the fixture says {}", r["resized"]));
    }
    if sha(&pre.canvas_rgb(img).unwrap().0.data) != r["hf_canvas_sha256"] {
        bad.push(format!("{what}: Hf canvas differs from the HF processor's"));
    }
    if sha(&f32_bytes(&v)) != r["hf_f32_sha256"] {
        bad.push(format!("{what}: Hf float32 tensor differs from the HF processor's"));
    }
    let f16 = pre.preprocess(img).unwrap().patches;
    if sha(&f16_bytes(&f16)) != r["hf_f16_sha256"] {
        bad.push(format!("{what}: Hf fp16 tensor differs from the HF processor's"));
    }
    let cast: Vec<u16> = v.iter().map(|&x| gb_vision::f32_to_f16_bits(x)).collect();
    if cast != f16 {
        bad.push(format!("{what}: the fp16 tensor is not the cast of the float32 one"));
    }
    if [g.grid.0, g.grid.1, g.grid.2] != usizes(&r["grid"])[..] {
        bad.push(format!("{what}: grid {:?}, HF {}", g.grid, r["grid"]));
    }
    let pillow_differs = r["pil_vs_hf"]["n_diff"].as_u64().unwrap() > 0;
    if !resized && pillow_differs {
        bad.push(format!("{what}: the fixture records a deviation for an image that is not resized"));
    }
    // the Pillow variant of HF's pixels (made by running `PIL.Image.resize`) is what `Hf` equals only where Pillow's resize happens to equal torch's
    let equals_pillow_variant = sha(&f32_bytes(&v)) == r["hf_pil_f32_sha256"];
    if equals_pillow_variant == pillow_differs {
        bad.push(format!("{what}: equality with the Pillow variant disagrees with the recorded deviation"));
    }
    (resized as usize, (resized && pillow_differs) as usize)
}

#[test]
fn many_odd_images_and_lowered_caps_follow_both_references() {
    let t = json("pixel_sweep.json");
    let (hf, fork) = (GlmPreprocessing::glm53_flash(Profile::Hf), GlmPreprocessing::glm53_flash(Profile::Fork));
    let mut bad = vec![];
    let rows = t["sweep"].as_array().unwrap();
    assert!(rows.len() >= 200);
    let (mut resized, mut pillow_differs) = (0, 0);
    for r in rows {
        let (w, h) = (r["w"].as_u64().unwrap() as usize, r["h"].as_u64().unwrap() as usize);
        let img = Rgb { width: w, height: h, data: pattern(w, h, r["kind"].as_u64().unwrap(), r["seed"].as_u64().unwrap()) };
        let (a, b) = check_hf_sweep_case(&mut bad, &format!("{w}x{h} (kind {})", r["kind"]), &hf, &img, r);
        (resized, pillow_differs) = (resized + a, pillow_differs + b);
        if sha(&f16_bytes(&fork.preprocess(&img).unwrap().patches)) != r["fork_f16_sha256"] {
            bad.push(format!("{w}x{h} (kind {}): Fork profile differs from the fork", r["kind"]));
        }
    }
    eprintln!("{} random images, {resized} of them resized, {pillow_differs} of those differ from the Pillow resize", rows.len());
    assert!(resized >= 80, "the sweep must exercise the resizers ({resized})");
    assert!(pillow_differs >= 60, "the sweep must exercise the difference between the kernels ({pillow_differs})");
    // a lowered token cap (the server's --max-image-tokens) is HF's `max_image_tokens` override, end to end
    let (mut tall, mut capped_resized, mut capped_differs) = (0, 0, 0);
    for r in t["caps"].as_array().unwrap() {
        let (w, h, cap) =
            (r["w"].as_u64().unwrap() as usize, r["h"].as_u64().unwrap() as usize, r["cap"].as_u64().unwrap() as usize);
        let name = r["name"].as_str().unwrap();
        let img = Rgb { width: w, height: h, data: pattern(w, h, r["kind"].as_u64().unwrap(), r["seed"].as_u64().unwrap()) };
        let capped = hf.clone().with_max_image_tokens(cap).unwrap();
        if name.starts_with("tall") {
            // taller than 100x its width and shrinking: Pillow 12.2's two-step, which the Fork profile reproduces through `bicubic_rgb_pil` and
            // torch's kernel, which `Hf` runs, does not have
            let g = capped.geometry(w, h).unwrap();
            assert!(h > 100 * w && g.content.1 < h, "{name}: {g:?} does not exercise the two-step rule");
            tall += 1;
        }
        let (a, b) = check_hf_sweep_case(&mut bad, &format!("{name} with cap {cap}"), &capped, &img, r);
        (capped_resized, capped_differs) = (capped_resized + a, capped_differs + b);
        assert!(capped.tokens(w, h).unwrap() <= cap, "{w}x{h}: over the cap {cap}");
        let f = fork.clone().with_max_image_tokens(cap).unwrap().preprocess(&img).unwrap();
        if sha(&f16_bytes(&f.patches)) != r["fork_f16_sha256"] {
            bad.push(format!("{name} with cap {cap}: Fork profile differs from the fork"));
        }
    }
    eprintln!(
        "{} lowered-cap cases, {capped_resized} resized, {capped_differs} of those differ from the Pillow resize",
        t["caps"].as_array().unwrap().len()
    );
    assert!(
        tall >= 3 && capped_resized >= 8 && capped_differs >= 5,
        "{tall} tall strips, {capped_resized} resized capped images, {capped_differs} differing"
    );
    assert!(bad.is_empty(), "{} mismatches:\n{}", bad.len(), bad.join("\n"));
}

#[test]
fn the_thread_count_never_changes_the_output() {
    // 4032x3024 is resized (both passes threaded by the field: Pillow's under Fork, torch's under Hf) and patchified, 1800x1200 only patchified
    // (above the inline threshold), 700x400 only patchified and under it
    for (w, h, seed) in [(4032usize, 3024usize, 5u64), (1800, 1200, 6), (700, 400, 7)] {
        let img = Rgb { width: w, height: h, data: pattern(w, h, 0, seed) };
        for profile in [Profile::Hf, Profile::Fork] {
            let mut p = GlmPreprocessing::glm53_flash(profile);
            p.threads = 1;
            let serial = p.preprocess(&img).unwrap();
            for threads in [2, 3, 7, 0] {
                p.threads = threads;
                assert_eq!(p.preprocess(&img).unwrap(), serial, "{profile:?} {w}x{h} with {threads} threads");
            }
        }
    }
}

#[test]
fn rope_angle_tables_match_the_forks_for_the_glm_and_qwen_head_dimensions() {
    use gb_vision::rope2d;
    let t = json("rope_cases.json");
    let (theta, merge) = (t["theta"].as_f64().unwrap() as f32, t["merge"].as_u64().unwrap() as usize);
    let mut n = 0;
    for c in t["cases"].as_array().unwrap() {
        let (hd, grid) = (c["head_dim"].as_u64().unwrap() as usize, usizes(&c["grid"]));
        let name = format!("head_dim {hd} grid {grid:?}");
        // inv_freq: torch's float32 pow against Rust's powf may differ in the last place
        let want: Vec<u32> = c["inv_freq_bits"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let inv = rope2d::inv_freq(hd, theta);
        assert_eq!(inv.len(), want.len(), "{name}");
        for (a, b) in inv.iter().zip(&want) {
            assert!(a.to_bits().abs_diff(*b) <= 1, "{name}: inv_freq {a} vs {}", f32::from_bits(*b));
        }
        // the ordering and the products are exact given the same inv_freq
        let pyinv: Vec<f32> = want.iter().map(|b| f32::from_bits(*b)).collect();
        let a = rope2d::angles_with(&pyinv, grid[0], grid[1], merge);
        let bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(sha(&bytes), c["angles_sha256"].as_str().unwrap(), "{name}: angle table");
        if let Some(bits) = c["angles_bits"].as_array() {
            let bits: Vec<u32> = bits.iter().map(|x| x.as_u64().unwrap() as u32).collect();
            assert!(a.iter().zip(&bits).all(|(x, y)| x.to_bits() == *y), "{name}: small table");
        }
        // and the table Rust builds on its own is within a float32 ulp or two of the reference
        let own = rope2d::angles(hd, theta, grid[0], grid[1], merge);
        let worst = own.iter().zip(&a).map(|(x, y)| x.to_bits().abs_diff(y.to_bits())).max().unwrap();
        assert!(worst <= 4, "{name}: own table differs by {worst} ulp");
        n += 1;
    }
    assert!(n >= 14);
}
