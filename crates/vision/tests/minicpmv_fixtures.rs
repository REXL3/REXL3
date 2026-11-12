//! MiniCPM-V-4.7 preprocessing against the real transformers 5.18 image processors (`tools/gen_minicpmv_vision_fixtures.py`,
//! `testdata/vision/minicpmv`): the torchvision backend (`MiniCPMV4_6ImageProcessor`) and the PIL backend (`MiniCPMV4_6ImageProcessorPil`).
//! Every image is compared by the sha256 of HF's float32 `pixel_values`, of its bf16 cast and of each crop's bf16 columns, by the grid, the
//! `target_sizes` and the token counts at 16x and 4x; the slicing arithmetic is swept over ~3,600 sizes; the normalisation tables are
//! compared for all 3 x 256 byte values.

use gb_vision::minicpmv::{
    find_best_resize, get_refine_size, get_sliced_grid, Downsample, MiniCpmPreprocessing, ResizeProfile, MAX_CROP_SIDE_PATCHES,
    MAX_TOTAL_PATCHES, PATCH_DIM,
};
use gb_vision::preprocessor::ImagePreprocessor;
use gb_vision::Rgb;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision")
}

fn sha(b: &[u8]) -> String {
    Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

fn json(name: &str) -> Value {
    let t = std::fs::read_to_string(dir().join("minicpmv").join(name))
        .unwrap_or_else(|e| panic!("{name}: {e} (run tools/gen_minicpmv_vision_fixtures.py)"));
    serde_json::from_str(&t).unwrap()
}

fn cases() -> Vec<Value> {
    json("fixtures.json")["cases"].as_array().unwrap().clone()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
}

/// The generator's `pattern()` (wrapping u32 arithmetic): kind 0 noise, kind 1 hard 3-pixel checker.
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

fn source(p: &MiniCpmPreprocessing, c: &Value) -> Rgb {
    if let Some(g) = c.get("gen").filter(|g| g.is_object()) {
        return procedural(
            g["kind"].as_str().unwrap(),
            g["w"].as_u64().unwrap() as usize,
            g["h"].as_u64().unwrap() as usize,
            g["seed"].as_u64().unwrap(),
        );
    }
    let bytes = std::fs::read(dir().join(c["file"].as_str().unwrap())).unwrap();
    p.decode(&bytes).unwrap_or_else(|e| panic!("{}: {e}", c["name"]))
}

fn profile_of(name: &str) -> ResizeProfile {
    if name == "torch" {
        ResizeProfile::Torch
    } else {
        ResizeProfile::Pil
    }
}

/// Every check of one recorded result (`r`) of one backend against the pipeline `p` on `img`; the messages of what differs.
fn check_record(name: &str, label: &str, p: &MiniCpmPreprocessing, img: &Rgb, r: &Value) -> Vec<String> {
    let mut bad = vec![];
    let (plan, f32s) = p.preprocess_crops_f32(img).unwrap_or_else(|e| panic!("{name} {label}: {e}"));
    let input = p.preprocess_crops(img).unwrap();
    let want_ts: Vec<[u32; 2]> = r["target_sizes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| [t[0].as_u64().unwrap() as u32, t[1].as_u64().unwrap() as u32])
        .collect();
    if input.target_sizes != want_ts || plan.target_sizes() != want_ts {
        bad.push(format!("{name} {label}: target_sizes {:?}, HF {want_ts:?}", input.target_sizes));
    }
    let g = usizes(&r["grid"]);
    if [plan.grid.0, plan.grid.1] != g[..] {
        bad.push(format!("{name} {label}: grid {:?}, HF {g:?}", plan.grid));
    }
    if plan.total_patches != r["total_patches"].as_u64().unwrap() as usize
        || plan.crops.len() != r["n_crops"].as_u64().unwrap() as usize
    {
        bad.push(format!(
            "{name} {label}: {} patches / {} crops, HF {}/{}",
            plan.total_patches,
            plan.crops.len(),
            r["total_patches"],
            r["n_crops"]
        ));
    }
    for (key, d) in [("x16", Downsample::X16), ("x4", Downsample::X4)] {
        let q = p.clone().with_downsample(d);
        let qp = q.plan(img.width, img.height).unwrap();
        let toks: Vec<usize> = qp.crops.iter().map(|c| c.tokens).collect();
        if toks != usizes(&r["tokens"][key]) {
            bad.push(format!("{name} {label} {key}: tokens {toks:?}, HF-derived {}", r["tokens"][key]));
        }
        if qp.total_tokens != toks.iter().sum::<usize>() {
            bad.push(format!("{name} {label} {key}: total_tokens"));
        }
    }
    if sha(&f32_bytes(&f32s)) != r["f32_sha256"] {
        bad.push(format!("{name} {label}: float32 pixel_values differ from HF's"));
    }
    if sha(&u16_bytes(&input.pixels)) != r["bf16_sha256"] {
        bad.push(format!("{name} {label}: bf16 pixels differ from HF pixel_values.to(bfloat16)"));
    }
    // the bf16 tensor is the RNE cast of the float32 one, element for element
    let cast: Vec<u16> = f32s.iter().map(|&v| gb_vision::nemotron::f32_to_bf16_bits(v)).collect();
    if cast != input.pixels {
        bad.push(format!("{name} {label}: bf16 pixels are not the cast of the float32 tensor"));
    }
    // per crop: columns [off * 14, (off + n) * 14) of every (channel, row)
    let total = plan.total_patches;
    let want = r["crop_bf16_sha256"].as_array().unwrap();
    let mut off = 0;
    for (k, c) in plan.crops.iter().enumerate() {
        let mut bytes = Vec::with_capacity(3 * 14 * c.patches() * 14 * 2);
        for row in 0..3 * 14 {
            bytes.extend(u16_bytes(&input.pixels[row * total * 14 + off * 14..row * total * 14 + (off + c.patches()) * 14]));
        }
        if sha(&bytes) != want[k] {
            bad.push(format!("{name} {label}: crop {k} differs"));
        }
        off += c.patches();
    }
    if input.pixels.len() != 3 * 14 * total * 14 || input.pixels.len() != plan.total_patches * PATCH_DIM {
        bad.push(format!("{name} {label}: tensor length"));
    }
    bad
}

fn check_backend(backend: &str) -> (usize, usize, usize) {
    let all = cases();
    assert!(all.len() >= 60, "{} cases", all.len());
    let base = MiniCpmPreprocessing::minicpmv47().with_profile(profile_of(backend));
    let (mut bad, mut images, mut elements, mut variants) = (vec![], 0usize, 0usize, 0usize);
    for c in &all {
        let name = c["name"].as_str().unwrap();
        let img = source(&base, c);
        if sha(&img.data) != c["rgb_sha256"] {
            bad.push(format!("{name}: RGB conversion differs from Pillow's image.convert(\"RGB\")"));
            continue;
        }
        if usizes(&c["size"]) != [img.width, img.height] {
            bad.push(format!("{name}: size"));
        }
        let r = &c[backend];
        bad.extend(check_record(name, "default", &base, &img, r));
        images += 1;
        elements += r["total_patches"].as_u64().unwrap() as usize * PATCH_DIM;
        if let Some(v) = r.get("variants").filter(|v| v.is_object()) {
            for (key, rv) in v.as_object().unwrap() {
                let ms: usize = key.strip_prefix("max_slice_nums_").unwrap().parse().unwrap();
                let q = base.clone().with_max_slice_nums(ms).unwrap();
                bad.extend(check_record(name, key, &q, &img, rv));
                variants += 1;
                elements += rv["total_patches"].as_u64().unwrap() as usize * PATCH_DIM;
            }
        }
    }
    assert!(bad.is_empty(), "{backend}: {} mismatches:\n{}", bad.len(), bad.join("\n"));
    eprintln!(
        "{backend}: {images} images + {variants} variants exact against HF ({elements} float32 elements compared by sha256)"
    );
    (images, variants, elements)
}

#[test]
fn torch_profile_matches_the_torchvision_backend_byte_for_byte() {
    let (images, variants, _) = check_backend("torch");
    assert!(images >= 60 && variants >= 20);
}

#[test]
fn pil_profile_matches_the_pil_backend_byte_for_byte() {
    let (images, variants, _) = check_backend("pil");
    assert!(images >= 60 && variants >= 20);
}

#[test]
fn the_placeholder_runs_equal_the_real_processors_prompt() {
    // `MiniCPMV4_7Processor.replace_image_token` over `target_sizes.prod(-1) // divisor`: the runs of image pad ids it puts in the prompt
    let base = MiniCpmPreprocessing::minicpmv47();
    let mut n = 0;
    for c in cases() {
        let Some(runs) = c.get("processor_pad_runs").filter(|r| r.is_object()) else { continue };
        let name = c["name"].as_str().unwrap();
        let (w, h) = (c["size"][0].as_u64().unwrap() as usize, c["size"][1].as_u64().unwrap() as usize);
        for (key, p) in [
            ("x16", base.clone()),
            ("x4", base.clone().with_downsample(Downsample::X4)),
            ("max_slice_nums_1", base.clone().with_max_slice_nums(1).unwrap()),
            ("max_slice_nums_36", base.clone().with_max_slice_nums(36).unwrap()),
        ] {
            let plan = p.plan(w, h).unwrap();
            let got: Vec<usize> = plan.crops.iter().map(|c| c.tokens).collect();
            assert_eq!(got, usizes(&runs[key]), "{name} {key}");
            assert_eq!(
                ImagePreprocessor::plan(&p, w, h).unwrap().tokens,
                got.iter().sum::<usize>(),
                "{name} {key}: ImagePlan.tokens"
            );
            n += 1;
        }
    }
    assert!(n >= 50, "{n} runs compared");
}

#[test]
fn the_fixture_set_covers_the_shapes_it_claims() {
    let all = cases();
    let grid = |c: &Value| usizes(&c["torch"]["grid"]);
    let sliced = all.iter().filter(|c| grid(c)[0] > 0).count();
    let unsliced = all.len() - sliced;
    let modes: std::collections::BTreeSet<String> = all.iter().map(|c| c["mode"].as_str().unwrap().to_string()).collect();
    let max_crops = all.iter().map(|c| c["torch"]["n_crops"].as_u64().unwrap()).max().unwrap();
    let rows: std::collections::BTreeSet<usize> = all.iter().map(|c| grid(c)[0]).collect();
    eprintln!(
        "{} images: {sliced} sliced, {unsliced} not, modes {modes:?}, grid rows {rows:?}, up to {max_crops} crops",
        all.len()
    );
    assert!(sliced >= 30 && unsliced >= 20, "{sliced} sliced, {unsliced} not");
    for m in ["RGB", "RGBA", "L", "LA", "P"] {
        assert!(modes.contains(m), "no {m} image in the fixtures: {modes:?}");
    }
    assert!(modes.iter().any(|m| m.starts_with("I;16") || m == "I"), "no 16-bit gray image: {modes:?}");
    assert!(max_crops >= 10, "no 9-slice image");
    // both 1-wide and 1-tall, and a 1x1
    let sizes: Vec<Vec<usize>> = all.iter().map(|c| usizes(&c["size"])).collect();
    assert!(
        sizes.contains(&vec![1, 1])
            && sizes.iter().any(|s| s[0] == 1 && s[1] > 100)
            && sizes.iter().any(|s| s[1] == 1 && s[0] > 100)
    );
    // the two-step Pillow rule (taller than 100x its width, height shrinks) is among them
    assert!(sizes.iter().any(|s| s[1] > 100 * s[0] && s[0] > 8));
}

#[test]
fn the_two_profiles_differ_exactly_where_the_hf_backends_do() {
    let (t, l) = (
        MiniCpmPreprocessing::minicpmv47().with_profile(ResizeProfile::Torch),
        MiniCpmPreprocessing::minicpmv47().with_profile(ResizeProfile::Pil),
    );
    let (mut differ, mut same) = (0, 0);
    for c in cases() {
        let name = c["name"].as_str().unwrap();
        let img = source(&t, &c);
        let a = t.preprocess_crops(&img).unwrap();
        let b = l.preprocess_crops(&img).unwrap();
        let hf_same = c["torch"]["bf16_sha256"] == c["pil"]["bf16_sha256"];
        assert_eq!(
            a.pixels == b.pixels,
            hf_same,
            "{name}: the profiles agree here, HF's backends {}",
            if hf_same { "do" } else { "do not" }
        );
        let n_diff = a.pixels.iter().zip(&b.pixels).filter(|(x, y)| x != y).count();
        if hf_same {
            same += 1;
        } else {
            differ += 1;
            assert_eq!(n_diff as u64, c["profiles_agree"]["n_bf16_diff"].as_u64().unwrap(), "{name}: bf16 elements that differ");
        }
        // the profiles never differ in the plan
        assert_eq!(a.plan, b.plan, "{name}");
    }
    eprintln!("torch and pil bf16 tensors: {same} images identical, {differ} differ (as HF's two backends do)");
    // a profile switch that did nothing would pass the byte tests above on the images where HF's backends agree, and fail here on the rest
    assert!(differ >= 50 && same >= 3, "{differ} differ, {same} same");
}

#[test]
fn normalisation_tables_equal_both_backends_for_every_byte_value() {
    let t = json("norm_luts.json");
    let bits = |k: &str| -> Vec<u64> { t[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect() };
    for profile in [ResizeProfile::Torch, ResizeProfile::Pil] {
        let p = MiniCpmPreprocessing::minicpmv47().with_profile(profile);
        let name = profile.name();
        let flat32: Vec<u64> = p.lut_f32().iter().flatten().map(|v| v.to_bits() as u64).collect();
        let flat16: Vec<u64> = p.lut_bf16().iter().flatten().map(|&v| v as u64).collect();
        assert_eq!(flat32, bits(&format!("{name}_f32_bits")), "{name}: float32 table");
        assert_eq!(flat16, bits(&format!("{name}_bf16_bits")), "{name}: bf16 table");
    }
    // the two backends' float32 tables differ (half of the 768 values, in the last bit) and their bf16 tables do not
    assert_eq!(t["n_f32_diff"].as_u64(), Some(384));
    assert_eq!(t["n_bf16_diff"].as_u64(), Some(0));
    let t32 = MiniCpmPreprocessing::minicpmv47().lut_f32();
    let p32 = MiniCpmPreprocessing::minicpmv47().with_profile(ResizeProfile::Pil).lut_f32();
    let n = t32.iter().flatten().zip(p32.iter().flatten()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(n, 384);
    let mean255: Vec<u64> = vec![(0.5f32 * (1.0f64 / (1.0 / 255.0f64)) as f32).to_bits() as u64; 3];
    assert_eq!(mean255, bits("torch_mean_x255_bits"));
}

#[test]
fn slicing_arithmetic_matches_the_processor_over_the_whole_sweep() {
    let sweep = json("slicing_sweep.json");
    let (scale, patch) = (448usize, 14usize);
    assert_eq!((sweep["scale_resolution"].as_u64(), sweep["patch_size"].as_u64()), (Some(448), Some(14)));
    let mut n = 0usize;
    let (mut grids, mut refused, mut ties) = (0usize, 0usize, 0usize);
    for ms in ["9", "36", "1", "4"] {
        let max_slices: usize = ms.parse().unwrap();
        let p = MiniCpmPreprocessing::minicpmv47().with_max_slice_nums(max_slices).unwrap();
        let rows = sweep[ms].as_array().unwrap();
        assert!(rows.len() >= 500, "{ms}: {} rows", rows.len());
        let mut bad = vec![];
        for r in rows {
            let v = usizes(r);
            let (h, w, gr, gc, sh, sw, rh, rw) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            let grid = get_sliced_grid((h, w), max_slices, scale);
            if grid != (gr > 0).then_some((gr, gc)) {
                bad.push(format!("{ms} {h}x{w}: grid {grid:?}, HF [{gr}, {gc}]"));
                continue;
            }
            let src = find_best_resize((h, w), scale, patch, grid.is_none());
            if src != (sh, sw) {
                bad.push(format!("{ms} {h}x{w}: source size {src:?}, HF ({sh}, {sw})"));
            }
            if let Some(g) = grid {
                let refine = get_refine_size((h, w), g, scale, patch, true);
                if refine != (rh, rw) {
                    bad.push(format!("{ms} {h}x{w}: refine size {refine:?}, HF ({rh}, {rw})"));
                }
                grids += 1;
            }
            // the plan: refused exactly when a crop side or the total is over the limits (computed from HF's sizes)
            let slice_side = if gr > 0 { (rh / gr / patch).max(rw / gc / patch) } else { 0 };
            let side = (sh / patch).max(sw / patch).max(slice_side);
            let total = (sh * sw + rh * rw) / (patch * patch);
            let refuse = side > MAX_CROP_SIDE_PATCHES || total > MAX_TOTAL_PATCHES;
            match (p.plan(w, h), refuse) {
                (Err(_), true) => refused += 1,
                (Ok(plan), false) => {
                    let ok = plan.grid == (gr, gc)
                        && (plan.crops[0].px_h, plan.crops[0].px_w) == (sh, sw)
                        && plan.refine_size().unwrap_or((0, 0)) == (rw, rh)
                        && plan.total_patches == total;
                    if !ok {
                        bad.push(format!("{ms} {h}x{w}: plan {plan:?}"));
                    }
                    // the placeholder count at 16x is HF's `target_sizes.prod(-1) // 16` summed
                    if plan.total_tokens != plan.crops.iter().map(|c| c.gh * c.gw / 16).sum::<usize>() {
                        bad.push(format!("{ms} {h}x{w}: tokens"));
                    }
                }
                (got, _) => bad.push(format!(
                    "{ms} {h}x{w}: plan {:?}, refuse = {refuse} (side {side}, total {total})",
                    got.map(|p| p.grid)
                )),
            }
            // an exact log-ratio tie: the ratio of two candidate grids' ratios is a square of W/H
            if grid.is_some() && h == w {
                ties += 1;
            }
            n += 1;
        }
        assert!(bad.is_empty(), "{} mismatches:\n{}", bad.len(), bad.iter().take(30).cloned().collect::<Vec<_>>().join("\n"));
    }
    eprintln!("slicing sweep: {n} (size, max_slice_nums) rows exact; {grids} with a grid, {refused} refused, {ties} square images with a grid");
    assert!(n >= 6000 && grids >= 3000 && refused >= 10 && ties >= 10, "{n} {grids} {refused} {ties}");
}

#[test]
fn the_output_does_not_depend_on_the_thread_count() {
    for name in ["m_1344x1792", "m_4000x300", "grad_1080x1920", "m_100x20000_twostep", "rgba_200x150", "m_1x1"] {
        let c = cases().into_iter().find(|c| c["name"] == name).unwrap();
        for profile in [ResizeProfile::Torch, ResizeProfile::Pil] {
            let base = MiniCpmPreprocessing::minicpmv47().with_profile(profile);
            let img = source(&base, &c);
            let one = base.clone().with_threads(1).preprocess_crops(&img).unwrap();
            for t in [2usize, 3, 5, 8, 0] {
                let many = base.clone().with_threads(t).preprocess_crops(&img).unwrap();
                assert_eq!(many, one, "{name} {profile:?} on {t} threads");
            }
            let (_, f1) = base.clone().with_threads(1).preprocess_crops_f32(&img).unwrap();
            let (_, f8) = base.clone().with_threads(8).preprocess_crops_f32(&img).unwrap();
            assert_eq!(f1, f8, "{name} {profile:?} float32");
        }
    }
}

#[test]
fn the_seam_carries_the_packed_tensor_and_the_original_size() {
    let p = MiniCpmPreprocessing::minicpmv47();
    let c = cases().into_iter().find(|c| c["name"] == "m_1344x1792").unwrap();
    let img = source(&p, &c);
    let input = p.preprocess_crops(&img).unwrap();
    let pre = ImagePreprocessor::preprocess(&p, &img).unwrap();
    assert_eq!(pre.size, (img.width, img.height), "size is the ORIGINAL size for this family");
    assert_eq!(pre.grid, (1, input.plan.grid.0, input.plan.grid.1));
    assert_eq!(pre.patch_dim, 588);
    assert_eq!(pre.patches, input.pixels);
    assert_eq!(pre.pixel_hash, input.pixel_hash);
    let plan = ImagePreprocessor::plan(&p, img.width, img.height).unwrap();
    assert_eq!(plan.tokens, input.plan.total_tokens);
    assert_eq!(plan.canvas, (input.plan.crops[0].px_w, input.plan.crops[0].px_h));
    assert_eq!(plan.grid, (1, input.plan.grid.0, input.plan.grid.1));
    assert_eq!(ImagePreprocessor::family(&p), "minicpmv4.7");
    assert_eq!(ImagePreprocessor::id(&p), p.id());
    // the hash moves with a pixel, and with the profile through the resize
    let mut img2 = img.clone();
    img2.data[100] ^= 1;
    assert_ne!(p.preprocess_crops(&img2).unwrap().pixel_hash, input.pixel_hash);
}

#[test]
fn decode_drops_alpha_and_refuses_what_it_cannot_read() {
    let p = MiniCpmPreprocessing::minicpmv47();
    // the committed RGBA PNG: the colour under a transparent pixel stays (convert("RGB")), as the fixtures' rgb_sha256 asserts for every mode
    let c = cases().into_iter().find(|c| c["name"] == "rgba_200x150").unwrap();
    let img = source(&p, &c);
    assert_eq!((img.width, img.height, img.data.len()), (200, 150, 200 * 150 * 3));
    assert_eq!(sha(&img.data), c["rgb_sha256"], "alpha dropped: Pillow's convert(\"RGB\")");
    assert!(p.decode(b"GIF89a....").is_err());
    assert!(p.decode(&[]).is_err());
}

/// `cargo test --release -p gb-vision --test minicpmv_fixtures -- --ignored --nocapture`: what the preprocessing of large images costs, per
/// profile (both resizes, normalise, bf16 rounding, packing, hash), on the default thread count.
#[test]
#[ignore]
fn preprocessing_cost_of_large_images() {
    for (w, h) in [(1344usize, 1792usize), (6000, 4000), (4032, 3024)] {
        let img = procedural("hash", w, h, 1);
        for profile in [ResizeProfile::Torch, ResizeProfile::Pil] {
            let p = MiniCpmPreprocessing::minicpmv47().with_profile(profile);
            let _ = p.preprocess_crops(&img).unwrap(); // warm up
            let t = std::time::Instant::now();
            let n = 5;
            let mut crops = 0;
            for _ in 0..n {
                crops = p.preprocess_crops(&img).unwrap().plan.crops.len();
            }
            eprintln!("{w}x{h} {profile:?}: {crops} crops, {:.1} ms", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
        }
    }
}

#[test]
fn hostile_inputs_are_refused_or_handled_without_a_panic() {
    let p = MiniCpmPreprocessing::minicpmv47();
    // a buffer of the wrong length, an empty image, a pixel count over the limit, an aspect no crop can take
    assert!(p.preprocess_crops(&Rgb { width: 4, height: 4, data: vec![0; 10] }).is_err());
    assert!(p.preprocess_crops(&Rgb { width: 0, height: 4, data: vec![] }).is_err());
    assert!(p.plan(usize::MAX, 2).is_err());
    assert!(p.plan(1 << 20, 1 << 20).is_err());
    assert!(p.plan(30_000, 1).is_err());
    assert!(p.plan(1, 30_000).is_err());
    // the extremes that are allowed run to the end
    for (w, h) in [(1usize, 1usize), (1, 2), (2, 1), (3, 3000), (3000, 3), (13, 13), (1, 4000)] {
        let img = Rgb { width: w, height: h, data: vec![200; w * h * 3] };
        for profile in [ResizeProfile::Torch, ResizeProfile::Pil] {
            let r = p.clone().with_profile(profile).preprocess_crops(&img);
            if let Ok(r) = r {
                assert_eq!(r.pixels.len(), r.plan.total_patches * PATCH_DIM);
                assert_eq!(r.target_sizes.len(), r.plan.crops.len());
            }
        }
    }
}
