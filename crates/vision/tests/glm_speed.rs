//! CPU cost of the image preprocessing at serving sizes, for both entries (Qwen's `Preprocessing` and the GLM `GlmPreprocessing`): the whole call
//! (`preprocessing_speed`), every stage of it (`stage_costs`: resize with the pad, the table-driven layout, the hash, with the bytes the layout writes
//! and the rate it achieves) and the layout and the hash for each thread count (`layout_scaling`).
//! `cargo test --release -p gb-vision --test glm_speed -- --ignored --nocapture`.
//!
//! Not an assertion about speed (machines differ); it prints the median of 7 timed runs (after one warm-up run, min and max beside it) so a regression
//! is visible and the numbers in `docs/VISION.md` can be redone. Noise content, so no shortcut can hide in the data.

use gb_vision::glm::{GlmPreprocessing, Profile};
use gb_vision::{layout, pixel_hash, resample, resize, smart_resize, torch_resample, Preprocessing, Rgb};
use std::time::Instant;

const RUNS: usize = 7;

fn noise(w: usize, h: usize) -> Rgb {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let data = (0..w * h * 3)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect();
    Rgb { width: w, height: h, data }
}

/// Median, min and max in milliseconds of `RUNS` timed runs of `f` after one untimed run.
fn timed<T>(mut f: impl FnMut() -> T) -> (f64, f64, f64) {
    drop(f());
    let mut ms: Vec<f64> = (0..RUNS)
        .map(|_| {
            let t = Instant::now();
            let out = f();
            let e = t.elapsed().as_secs_f64() * 1e3;
            drop(out);
            e
        })
        .collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (ms[RUNS / 2], ms[0], ms[RUNS - 1])
}

fn show((med, lo, hi): (f64, f64, f64)) -> String {
    format!("{med:7.1} ms (min {lo:.1}, max {hi:.1})")
}

#[test]
#[ignore]
fn preprocessing_speed() {
    let qwen = Preprocessing::qwen38();
    let glm = GlmPreprocessing::default();
    eprintln!("median of {RUNS} runs after a warm-up, noise content; the resize alone, then the whole call");
    for (w, h) in [(640, 480), (1920, 1080), (4032, 3024), (6000, 4000), (3, 5000)] {
        let img = noise(w, h);
        // GLM entry, default profile (both profiles share the resize)
        let g = glm.geometry(w, h).unwrap();
        let (cw, ch) = g.content;
        if (cw, ch) != (w, h) {
            let t = timed(|| resample::bicubic_rgb_pil(&img.data, w, h, cw, ch));
            eprintln!("{w}x{h}: GLM resize to {cw}x{ch} {}", show(t));
        } else {
            eprintln!("{w}x{h}: GLM not resized (canvas {}x{}, patchify only)", g.canvas.0, g.canvas.1);
        }
        eprintln!("{w}x{h}: GLM preprocess ({} tokens) {}", g.tokens, show(timed(|| glm.preprocess(&img).unwrap())));
        // Qwen entry
        match smart_resize(w, h, &qwen) {
            Ok((nw, nh)) => {
                if (nw, nh) != (w, h) {
                    let t = timed(|| resize::bicubic_rgb(&img.data, w, h, nw, nh));
                    eprintln!("{w}x{h}: Qwen resize to {nw}x{nh} {}", show(t));
                }
                eprintln!("{w}x{h}: Qwen preprocess {}", show(timed(|| gb_vision::preprocess(&img, &qwen).unwrap())));
            }
            Err(e) => eprintln!("{w}x{h}: Qwen refuses ({e})"),
        }
    }
    // a strip taller than 100x its width that shrinks is resized in Pillow 12.2's two steps; the canvas rule leaves such an image alone unless the token
    // cap bites, so lower the cap (as the fixtures do) to take that path
    let capped = GlmPreprocessing { max_image_tokens: 16, ..GlmPreprocessing::default() };
    for (w, h) in [(3, 5000), (40, 20000)] {
        let img = noise(w, h);
        let g = capped.geometry(w, h).unwrap();
        let (cw, ch) = g.content;
        let t = timed(|| resample::bicubic_rgb_pil(&img.data, w, h, cw, ch));
        eprintln!("{w}x{h} under a 16-token cap: GLM two-step resize to {cw}x{ch} {}", show(t));
        eprintln!("{w}x{h} under a 16-token cap: GLM preprocess {}", show(timed(|| capped.preprocess(&img).unwrap())));
    }
    // the profiles and the small and middle sizes of the earlier record (`Hf` resizes with the torch kernel, `Fork` with Pillow's)
    for profile in [Profile::Hf, Profile::Fork] {
        let p = GlmPreprocessing::glm53_flash(profile);
        for (w, h) in [(224, 224), (1024, 768), (2800, 2240)] {
            let img = noise(w, h);
            let t = timed(|| p.preprocess(&img).unwrap());
            eprintln!("{profile:?} {w}x{h}: GLM preprocess {}", show(t));
        }
    }
}

/// The resize alone for each thread count (`resize::bicubic_rgb_threads`: exactly that many threads, no inline rule), with the work in multiply-accumulates
/// the policy divides into threads; the number `MIN_WORK_PER_WORKER` and the scaling table of `docs/VISION.md` come from here.
/// `cargo test --release -p gb-vision --test glm_speed resize_scaling -- --ignored --nocapture`.
#[test]
#[ignore]
fn resize_scaling() {
    // SPEED_SHAPES=0,2 (indices into the list below) and SPEED_COUNTS=1,8 narrow a run while tuning
    let pick = |var: &str| -> Option<Vec<usize>> {
        std::env::var(var).ok().map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
    };
    let counts = pick("SPEED_COUNTS").unwrap_or_else(|| vec![1usize, 2, 3, 4, 6, 8, 12, 16, 20]);
    eprintln!("resize alone, median ms of {RUNS} after a warm-up; default workers {}", resize::default_workers());
    // big: what a serving request sees; small: where the inline rule has to decide
    let shapes: [(usize, usize, usize, usize); 14] = [
        (4032, 3024, 2874, 2156),
        (6000, 4000, 3052, 2034),
        (4032, 3024, 4032, 3008),
        (1920, 1080, 1920, 1088),
        (3000, 2000, 112, 75),
        (640, 480, 320, 240),
        (1024, 768, 512, 384),
        (448, 448, 336, 336),
        (256, 256, 224, 224),
        (160, 120, 112, 84),
        (100, 100, 112, 112),
        (64, 64, 112, 112),
        (60, 40, 112, 80),
        (32, 32, 112, 112),
    ];
    // what a worker costs before it computes anything: a scope that spawns and joins `n` threads that do nothing (the inline rule weighs this against the work)
    for n in [1usize, 3, 7] {
        let (med, _, _) = timed(|| std::thread::scope(|s| (0..n).for_each(|_| drop(s.spawn(|| std::hint::black_box(1))))));
        eprintln!("spawn and join of {n} empty scoped threads: {:.0} us", med * 1e3);
    }
    let only = pick("SPEED_SHAPES");
    for (i, (sw, sh, dw, dh)) in shapes.into_iter().enumerate() {
        if only.as_ref().is_some_and(|o| !o.contains(&i)) {
            continue;
        }
        let img = noise(sw, sh);
        let work = resize::resize_work(sw, sh, dw, dh);
        let policy = resize::resize_workers(resize::default_workers(), work);
        let mut line = format!("{sw}x{sh} -> {dw}x{dh} ({:.2} M MACs, policy {policy} workers):", work as f64 / 1e6);
        for &t in &counts {
            let (med, _, _) = timed(|| resize::bicubic_rgb_threads(&img.data, sw, sh, dw, dh, t));
            line += &format!(" {t}:{med:.3}");
        }
        let (med, _, _) = timed(|| resize::bicubic_rgb(&img.data, sw, sh, dw, dh));
        eprintln!("{line} | policy {med:.3} ms");
    }
}

/// The first three fields of `/proc/loadavg` (the node is shared; the figures of `docs/VISION.md` are quoted with them).
fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        .unwrap_or_default()
}

/// The torch kernel against the Pillow kernel on the resizes the GLM canvas rule asks for (4032x3024 and 6000x4000 at the default budget, 1920x1080
/// under a server cap of 1024 image tokens, which does resize it), serial and with the default policy, and the whole `preprocess` of the two
/// profiles: `Fork` resizes with Pillow's kernel, `Hf` with torch's. The load average is printed before and after.
/// `cargo test --release -p gb-vision --test glm_speed torch_vs_pillow_resize -- --ignored --nocapture`.
#[test]
#[ignore]
fn torch_vs_pillow_resize() {
    eprintln!(
        "load average before: {}; default workers {}; median of {RUNS} after a warm-up (min, max)",
        loadavg(),
        resize::default_workers()
    );
    for (w, h, cap) in [(4032usize, 3024usize, 8000usize), (6000, 4000, 8000), (1920, 1080, 1024)] {
        let (fork, hf) = (
            GlmPreprocessing::glm53_flash(Profile::Fork).with_max_image_tokens(cap).unwrap(),
            GlmPreprocessing::glm53_flash(Profile::Hf).with_max_image_tokens(cap).unwrap(),
        );
        let g = fork.geometry(w, h).unwrap();
        let (cw, ch) = g.content;
        assert_eq!(g.content, hf.geometry(w, h).unwrap().content);
        assert!((cw, ch) != (w, h), "{w}x{h} is not resized at {cap} tokens");
        let img = noise(w, h);
        let macs = resize::resize_work(w, h, cw, ch);
        assert_eq!(macs, torch_resample::torch_resize_work(w, h, cw, ch));
        eprintln!(
            "{w}x{h} -> {cw}x{ch} ({cap} tokens, {:.0} M MACs, {} workers by the policy):",
            macs as f64 / 1e6,
            resize::resize_workers(resize::default_workers(), macs)
        );
        let p1 = timed(|| resample::bicubic_rgb_pil_threads(&img.data, w, h, cw, ch, 1));
        let t1 = timed(|| torch_resample::bicubic_rgb_torch_threads(&img.data, w, h, cw, ch, 1));
        let pn = timed(|| resample::bicubic_rgb_pil(&img.data, w, h, cw, ch));
        let tn = timed(|| torch_resample::bicubic_rgb_torch(&img.data, w, h, cw, ch));
        eprintln!("    resize, 1 thread:  Pillow {}   torch {}   torch / Pillow {:.2}", show(p1), show(t1), t1.0 / p1.0);
        eprintln!("    resize, policy:    Pillow {}   torch {}   torch / Pillow {:.2}", show(pn), show(tn), tn.0 / pn.0);
        let (pf, ph) = (timed(|| fork.preprocess(&img).unwrap()), timed(|| hf.preprocess(&img).unwrap()));
        eprintln!("    preprocess:        Fork   {}   Hf    {}   Hf / Fork {:.2}", show(pf), show(ph), ph.0 / pf.0);
    }
    // the scaling of the torch kernel on the photo, to check that the policy's `MIN_WORK_PER_WORKER` (measured for Pillow's kernel) fits it
    let img = noise(4032, 3024);
    let mut line = "torch resize 4032x3024 -> 2874x2156 by thread count (ms):".to_string();
    for t in [1usize, 2, 4, 8, 16] {
        line +=
            &format!(" {t}:{:.1}", timed(|| torch_resample::bicubic_rgb_torch_threads(&img.data, 4032, 3024, 2874, 2156, t)).0);
    }
    eprintln!("{line}");
    let small = noise(256, 256);
    let mut line = "torch resize 256x256 -> 224x224 (ms):".to_string();
    for t in [1usize, 2, 4, 8] {
        line += &format!(" {t}:{:.3}", timed(|| torch_resample::bicubic_rgb_torch_threads(&small.data, 256, 256, 224, 224, t)).0);
    }
    eprintln!("{line}");
    eprintln!("load average after: {}", loadavg());
}

/// GB/s of `bytes` moved in `ms` milliseconds.
fn gbps(bytes: usize, ms: f64) -> f64 {
    bytes as f64 / (ms * 1e-3) / 1e9
}

/// Every stage of both entries at serving sizes (decode excluded): the resize (alone, then with the pad for GLM), the table-driven layout with the
/// bytes it writes and the rate that is (read of the pixels plus write of the tensor), the hash of the canvas, and the whole call, which overlaps
/// the hash with the layout (so it is not the sum). `SPEED_SIZES=2,3` (indices into the list below) narrows a run.
/// `cargo test --release -p gb-vision --test glm_speed stage_costs -- --ignored --nocapture`.
#[test]
#[ignore]
fn stage_costs() {
    let only: Option<Vec<usize>> =
        std::env::var("SPEED_SIZES").ok().map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect());
    let workers = resize::default_workers();
    eprintln!("median of {RUNS} after a warm-up (min, max), {workers} default workers; layout = normalise + patch layout, hash = FNV-1a of the canvas");
    let qwen = Preprocessing::qwen38();
    let (t, _, _) = timed(|| qwen.lut_f16());
    eprintln!("Qwen value table (768 entries): {:.1} us", t * 1e3);
    for (i, (w, h)) in [(640usize, 480usize), (1920, 1080), (4032, 3024), (6000, 4000)].into_iter().enumerate() {
        if only.as_ref().is_some_and(|o| !o.contains(&i)) {
            continue;
        }
        let img = noise(w, h);
        // Qwen
        let (nw, nh) = smart_resize(w, h, &qwen).unwrap();
        let resized = (nw, nh) != (w, h);
        let data = if resized { resize::bicubic_rgb(&img.data, w, h, nw, nh) } else { img.data.clone() };
        let lut = qwen.lut_f16();
        let n = layout::layout_workers(workers, nw * nh);
        let out_bytes = nw * nh * 12;
        let lay = timed(|| layout::patchify(&data, nw, nh, &lut, qwen.shape(), n));
        let hash = timed(|| pixel_hash(nw, nh, &data));
        let whole = timed(|| gb_vision::preprocess(&img, &qwen).unwrap());
        let rs = if resized { show(timed(|| resize::bicubic_rgb(&img.data, w, h, nw, nh))) } else { "not resized".to_string() };
        eprintln!("{w}x{h} Qwen -> {nw}x{nh}:");
        eprintln!("    resize   {rs}");
        eprintln!(
            "    layout   {} on {n} workers: {:.1} MB out, {:.1} GB/s (read + write)",
            show(lay),
            out_bytes as f64 / 1e6,
            gbps(out_bytes + nw * nh * 3, lay.0)
        );
        eprintln!("    hash     {} ({:.2} ns/byte)", show(hash), hash.0 * 1e6 / (nw * nh * 3) as f64);
        eprintln!("    whole    {}", show(whole));
    }
    for profile in [Profile::Fork, Profile::Hf] {
        let p = GlmPreprocessing::glm53_flash(profile);
        for (i, (w, h)) in [(640usize, 480usize), (1920, 1080), (4032, 3024), (6000, 4000)].into_iter().enumerate() {
            if only.as_ref().is_some_and(|o| !o.contains(&i)) {
                continue;
            }
            let img = noise(w, h);
            let g = p.geometry(w, h).unwrap();
            let (cw, ch) = g.content;
            let (tw, th) = g.canvas;
            eprintln!("{w}x{h} GLM {profile:?}: content {cw}x{ch} in canvas {tw}x{th} ({} tokens)", g.tokens);
            if (cw, ch) != (w, h) {
                let t = match profile {
                    Profile::Hf => timed(|| torch_resample::bicubic_rgb_torch(&img.data, w, h, cw, ch)),
                    Profile::Fork => timed(|| resample::bicubic_rgb_pil(&img.data, w, h, cw, ch)),
                };
                eprintln!("    resize   {}", show(t));
            }
            eprintln!(
                "    canvas   {} (resize and pad; a view of the input when neither)",
                show(timed(|| p.canvas_rgb(&img).unwrap()))
            );
            let (canvas, _) = p.canvas_rgb(&img).unwrap();
            let lut = p.lut_f16();
            let n = layout::layout_workers(workers, tw * th);
            let out_bytes = tw * th * 12;
            let lay = timed(|| layout::patchify(&canvas.data, tw, th, &lut, p.shape(), n));
            eprintln!(
                "    layout   {} on {n} workers: {:.1} MB out, {:.1} GB/s (read + write)",
                show(lay),
                out_bytes as f64 / 1e6,
                gbps(out_bytes + tw * th * 3, lay.0)
            );
            let hash = timed(|| pixel_hash(tw, th, &canvas.data));
            eprintln!("    hash     {} ({:.2} ns/byte)", show(hash), hash.0 * 1e6 / (tw * th * 3) as f64);
            eprintln!("    whole    {}", show(timed(|| p.preprocess(&img).unwrap())));
            let pre = p.preprocess(&img).unwrap();
            eprintln!(
                "    still_patches (the half-width tensor for a folded tower) {}",
                show(timed(|| gb_vision::glm::still_patches(&pre, 2)))
            );
        }
    }
}

/// The layout kernel into a buffer that has been written before (`layout::patchify_into`: no allocation, no page faults) against a fresh zeroed
/// allocation each time (`layout::patchify`), for 1 and 8 workers: what the loops cost, and what the first touch of a 145 MB tensor adds.
/// `cargo test --release -p gb-vision --test glm_speed layout_kernel -- --ignored --nocapture`.
#[test]
#[ignore]
fn layout_kernel() {
    let qwen = Preprocessing::qwen38();
    let lut = qwen.lut_f16();
    eprintln!("layout kernel on a warm buffer vs a fresh allocation, median ms of {RUNS}; patch 16, fp16");
    for (w, h) in [(4032usize, 3008usize), (1920, 1088), (640, 480), (256, 256)] {
        let img = noise(w, h);
        let len = layout::tensor_len(w, h, qwen.shape());
        let mut out = vec![0u16; len];
        let mut line = format!("{w}x{h} ({:.1} MB out):", len as f64 * 2.0 / 1e6);
        for workers in [1usize, 8, 16] {
            let (warm, _, _) = timed(|| layout::patchify_into(&mut out, &img.data, w, h, &lut, qwen.shape(), workers));
            let (fresh, _, _) = timed(|| layout::patchify(&img.data, w, h, &lut, qwen.shape(), workers));
            line += &format!(
                " | {workers} workers: warm {warm:.2} ms ({:.2} ns/px, {:.1} GB/s), fresh {fresh:.2} ms",
                warm * 1e6 / (w * h) as f64,
                gbps(len * 2 + w * h * 3, warm)
            );
        }
        eprintln!("{line}");
    }
    // the loops with everything in the L1 and L2 caches: one thread, the same small image 200 times over (a source of 12 to 196 KB, a tensor of 49 KB to
    // 786 KB), so what is left is the instruction stream, without the misses of a large source and the faults of a large tensor
    for (w, h) in [(64usize, 64usize), (128, 128), (256, 256)] {
        let img = noise(w, h);
        let mut out = vec![0u16; layout::tensor_len(w, h, qwen.shape())];
        let (ms, _, _) = timed(|| {
            for _ in 0..200 {
                layout::patchify_into(&mut out, &img.data, w, h, &lut, qwen.shape(), 1);
            }
        });
        eprintln!("{w}x{h} cache-resident, 1 thread: {:.2} ns/px", ms * 1e6 / (200 * w * h) as f64);
    }
}

/// The layout alone for each thread count (`layout::patchify`: exactly that many workers, no inline rule) and the hash of the same bytes, from a
/// thumbnail to a 12 MP photo; the serial cost per pixel and the point where a worker stops paying for its start (`MIN_PIXELS_PER_WORKER`) come from here.
/// `cargo test --release -p gb-vision --test glm_speed layout_scaling -- --ignored --nocapture`.
#[test]
#[ignore]
fn layout_scaling() {
    let counts = [1usize, 2, 3, 4, 6, 8, 12, 16, 20];
    let qwen = Preprocessing::qwen38();
    let lut = qwen.lut_f16();
    eprintln!(
        "layout alone (Qwen tables, patch 16), median ms of {RUNS} after a warm-up; default workers {}",
        resize::default_workers()
    );
    for (w, h) in [
        (4032usize, 3008usize),
        (1920, 1088),
        (1280, 960),
        (640, 480),
        (512, 512),
        (384, 352),
        (256, 256),
        (192, 160),
        (128, 128),
        (64, 64),
    ] {
        let img = noise(w, h);
        let mut line =
            format!("{w}x{h} ({} px, policy {} workers):", w * h, layout::layout_workers(resize::default_workers(), w * h));
        let mut serial = 0.0;
        for &t in &counts {
            let (med, _, _) = timed(|| layout::patchify(&img.data, w, h, &lut, qwen.shape(), t));
            if t == 1 {
                serial = med;
            }
            line += &format!(" {t}:{med:.3}");
        }
        let (hash, _, _) = timed(|| pixel_hash(w, h, &img.data));
        eprintln!("{line} | serial {:.2} ns/px | hash {hash:.3} ms", serial * 1e6 / (w * h) as f64);
    }
    // what a worker costs before it computes anything, as in `resize_scaling`
    for n in [1usize, 3, 7] {
        let (med, _, _) = timed(|| std::thread::scope(|s| (0..n).for_each(|_| drop(s.spawn(|| std::hint::black_box(1))))));
        eprintln!("spawn and join of {n} empty scoped threads: {:.0} us", med * 1e3);
    }
}
