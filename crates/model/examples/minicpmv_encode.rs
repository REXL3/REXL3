//! Stand-alone MiniCPM-V-4.7 image encoder (E5a of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`): an image file in, the projector's embedding
//! rows out, through the family's own preprocessing, tower, window merger and projector (no language model, no server).
//!
//!     minicpmv_encode CHECKPOINT_DIR IMAGE [--mode 16x|4x] [--profile torch|pil] [--max-slice-nums N] [--max-patches N]
//!                     [--repeat N] [--out-f32 FILE] [--out-bf16 FILE]
//!
//! CHECKPOINT_DIR is the bf16 checkpoint folder or an EXL3 pack (the 459 tower tensors are the same in both). PNG and JPEG are read
//! through the preprocessing's own decoder (alpha dropped as `convert("RGB")` does). It prints the crop plan, the preprocessing and
//! encode times, and statistics of the features (the projector's fp32 output and the bf16 rows the language model would receive).
//! `--out-f32` / `--out-bf16` write the rows raw (little endian, `[tokens, 2048]`). GPU runs belong under `flock /tmp/rexl3-gpu.lock`.

use gb_model::families::minicpmv4_7::vision::{mode_of, MiniCpmTower, DEFAULT_MAX_PATCHES, OUT_HIDDEN};
use gb_vision::minicpmv::{Downsample, MiniCpmPreprocessing, ResizeProfile};
use gb_vision::preprocessor::ImagePreprocessor;
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "usage: minicpmv_encode CHECKPOINT_DIR IMAGE [--mode 16x|4x] [--profile torch|pil] [--max-slice-nums N] [--max-patches N] \
         [--repeat N] [--out-f32 FILE] [--out-bf16 FILE]"
    );
    std::process::exit(2);
}

fn f32_to_bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(dir), Some(image)) = (args.next(), args.next()) else { usage() };
    let (mut mode, mut profile, mut max_slices, mut max_patches, mut repeat) =
        ("16x".to_string(), "torch".to_string(), None, DEFAULT_MAX_PATCHES, 3usize);
    let (mut out_f32, mut out_bf16) = (None, None);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--mode" => mode = val(),
            "--profile" => profile = val(),
            "--max-slice-nums" => max_slices = Some(val().parse::<usize>().unwrap_or_else(|_| usage())),
            "--max-patches" => max_patches = val().parse().unwrap_or_else(|_| usage()),
            "--repeat" => repeat = val().parse().unwrap_or_else(|_| usage()),
            "--out-f32" => out_f32 = Some(val()),
            "--out-bf16" => out_bf16 = Some(val()),
            _ => usage(),
        }
    }
    let dir = std::path::PathBuf::from(dir);
    let mut pre = MiniCpmPreprocessing::from_dir(&dir).unwrap_or_else(|_| MiniCpmPreprocessing::minicpmv47());
    pre = pre.with_downsample(Downsample::parse(&mode).unwrap_or_else(|| usage()));
    pre = pre.with_profile(match profile.as_str() {
        "torch" => ResizeProfile::Torch,
        "pil" => ResizeProfile::Pil,
        _ => usage(),
    });
    if let Some(n) = max_slices {
        pre = pre.with_max_slice_nums(n).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        });
    }
    let bytes = std::fs::read(&image).unwrap_or_else(|e| {
        eprintln!("{image}: {e}");
        std::process::exit(1)
    });
    let rgb = ImagePreprocessor::decode(&pre, &bytes).unwrap_or_else(|e| {
        eprintln!("{image}: {e}");
        std::process::exit(1)
    });
    let t = Instant::now();
    let plan = pre.plan(rgb.width, rgb.height).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    let input = pre.preprocess_crops(&rgb).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    let pre_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "image {}x{} ({} profile, {mode}): {} crops (thumbnail {}x{} px{}), {} patches, {} tokens; preprocess {pre_ms:.1} ms; pixel hash {:016x}",
        rgb.width,
        rgb.height,
        profile,
        plan.crops.len(),
        plan.crops[0].px_w,
        plan.crops[0].px_h,
        if plan.grid.0 > 0 {
            format!(", {} x {} slices of {}x{} px", plan.grid.0, plan.grid.1, plan.crops[1].px_w, plan.crops[1].px_h)
        } else {
            String::new()
        },
        plan.total_patches,
        plan.total_tokens,
        input.pixel_hash
    );
    if !gb_cuda::device_available() {
        eprintln!("no CUDA device: the plan above is all this build can do");
        return;
    }
    let t = Instant::now();
    let mut tower = MiniCpmTower::load(&dir, max_patches).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    let (wb, sb) = tower.bytes();
    println!(
        "tower loaded in {:.1} s: weights {:.0} MiB, workspace {:.0} MiB; {:?}",
        t.elapsed().as_secs_f64(),
        wb as f64 / 1048576.0,
        sb as f64 / 1048576.0,
        tower.convert_report()
    );
    let mode = mode_of(pre.downsample);
    let mut feats: Vec<f32> = Vec::new();
    for k in 0..repeat.max(1) {
        let t = Instant::now();
        feats = tower.encode_packed_f32(&input.pixels, &input.target_sizes, mode).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1)
        });
        println!("encode #{k}: {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    assert_eq!(feats.len(), plan.total_tokens * OUT_HIDDEN);
    let n = feats.len() as f64;
    let (mut lo, mut hi, mut sum, mut sq, mut bad) = (f32::INFINITY, f32::NEG_INFINITY, 0f64, 0f64, 0usize);
    for &v in &feats {
        if !v.is_finite() {
            bad += 1;
            continue;
        }
        lo = lo.min(v);
        hi = hi.max(v);
        sum += v as f64;
        sq += v as f64 * v as f64;
    }
    let bits: Vec<u16> = feats.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    let h = bits.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        let h = (h ^ (b & 0xff) as u64).wrapping_mul(0x0100_0000_01b3);
        (h ^ (b >> 8) as u64).wrapping_mul(0x0100_0000_01b3)
    });
    println!(
        "features [{} tokens, {OUT_HIDDEN}]: min {lo:.4} max {hi:.4} mean {:.5} rms {:.4} non-finite {bad}; bf16 digest {h:016x}",
        plan.total_tokens,
        sum / n,
        (sq / n).sqrt()
    );
    if let Some(p) = out_f32 {
        let raw: Vec<u8> = feats.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&p, raw).unwrap();
        println!("wrote {p}");
    }
    if let Some(p) = out_bf16 {
        let raw: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&p, raw).unwrap();
        println!("wrote {p}");
    }
}
