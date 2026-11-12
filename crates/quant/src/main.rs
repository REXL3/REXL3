//! `rexl3-quant COMMAND [OPTIONS]`
//!
//! cuBLAS runs in pedantic math (no TF32, the prescribed fp32 precision in every phase); `--default-math` selects
//! CUBLAS_DEFAULT_MATH, which PyTorch uses with TF32 off (the same bits on the measured sets).
//!
//! Commands:
//! - `info`: the CUDA device and the cuBLAS / cuSOLVER versions this build loads.
//! - `parity --dump DIR [--energy-order] [--refit-guard] [--stages] [--limit N] [--workers N] [--default-math] [--out FILE]`: quantize
//!   every tensor of an oracle dump and compare the stored outputs bit for bit (exit 1 on any difference).
//! - `bench --dump DIR [same options]`: the same run, reporting the stage times against the dump's.
//! - `tensor --model PATH --key NAME --hessian FILE --count N --out FILE [--bits K] [--seed S] [--parity]
//!   [--refit-guard] [--default-math]`: quantize one linear weight (`[out, in]`, F32 / F16 / BF16) of a safetensors file
//!   or folder with its input Hessian sum (`H`, F32 `[in, in]`, summed over N rows); writes `NAME.suh`, `.svh`,
//!   `.trellis` and `.mul1` (NAME without a trailing `.weight`) to FILE and prints a JSON summary.
//! - `capture --model DIR --rows FILE --out DIR [--layers N] [--cal-rows N] [--cal-cols N] [--batch-rows N] [--bits K]
//!   [--capture-gemm wide|batched] [--oracle DIR] [--default-math]`: the teacher's capture of the first N decoder
//!   layers, each against the quantized prefix (layers before it quantized at K bits); writes
//!   `m<module>/hessians.safetensors` and `capture.json` per layer and, with an oracle directory, compares H, row and
//!   routing counts and the stream entering each layer bit for bit (exit 1 on any difference).

use std::process::ExitCode;

const USAGE: &str = "usage: rexl3-quant COMMAND [OPTIONS]\n\ncommands:\n  info    print the device and library versions\n  parity  --dump DIR [--energy-order] [--refit-guard] [--stages] [--limit N] [--workers N] [--default-math] [--out FILE]\n  bench   --dump DIR [same options]\n  tensor  --model PATH --key NAME --hessian FILE --count N --out FILE [--bits K] [--seed S] [--parity] [--refit-guard] [--default-math]\n  capture --model DIR --rows FILE --out DIR [--layers N] [--cal-rows N] [--cal-cols N] [--batch-rows N] [--bits K] [--capture-gemm wide|batched] [--oracle DIR] [--default-math]\n";

fn usage(msg: &str) -> ExitCode {
    eprintln!("error: {msg}\n{USAGE}");
    ExitCode::from(2)
}

fn info() -> ExitCode {
    let v = gb_cuda::rt::versions();
    println!("pack suffix: {}", gb_quant::PACK_SUFFIX);
    let s = |o: &Option<String>| o.clone().unwrap_or_else(|| "unavailable".into());
    println!("cuda runtime {} driver {} kernels {} {}", s(&v.cudart), s(&v.driver), v.arch, v.kernel_hash);
    match gb_quant::gpu::versions() {
        Ok(l) => {
            println!("cublas {}", l.cublas);
            println!("cusolver {}", l.cusolver);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_dump(bench: bool, args: &mut dyn Iterator<Item = String>) -> ExitCode {
    let mut dump = None;
    let mut out = None;
    let mut cfg = gb_quant::bench::RunConfig {
        energy_order: false,
        refit_guard: false,
        limit: None,
        stages: false,
        workers: 1,
        mode: gb_quant::gpu::cublas::MathMode::Pedantic,
    };
    let mut mode = gb_quant::gpu::cublas::MathMode::Pedantic;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dump" => dump = args.next(),
            "--out" => out = args.next(),
            "--energy-order" => cfg.energy_order = true,
            "--refit-guard" => cfg.refit_guard = true,
            "--stages" => cfg.stages = true,
            "--default-math" => mode = gb_quant::gpu::cublas::MathMode::Default,
            "--workers" => match args.next().and_then(|v| v.parse::<usize>().ok()).filter(|v| *v >= 1) {
                Some(v) => cfg.workers = v,
                None => return usage("--workers needs a positive number"),
            },
            "--limit" => match args.next().and_then(|v| v.parse().ok()) {
                Some(v) => cfg.limit = Some(v),
                None => return usage("--limit needs a number"),
            },
            _ => return usage(&format!("unknown option {a}")),
        }
    }
    let Some(dump) = dump else { return usage("--dump DIR is required") };
    cfg.mode = mode;
    let mut g = match gb_quant::core::dev::Gpu::new(mode) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let report = match gb_quant::bench::run(&mut g, std::path::Path::new(&dump), &cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let text = serde_json::to_string_pretty(&report).unwrap_or_default();
    if let Some(p) = out {
        if let Err(e) = std::fs::write(&p, text + "\n") {
            eprintln!("error: {p}: {e}");
            return ExitCode::FAILURE;
        }
    }
    let s = &report["seconds"];
    println!(
        "tensors {} pass {} rust {:.2} s python {:.2} s",
        report["tensors"],
        report["pass"],
        s["rust"].as_f64().unwrap_or(0.0),
        s["python"].as_f64().unwrap_or(0.0)
    );
    if bench {
        for st in report["stages"].as_array().into_iter().flatten() {
            println!("  {:16} {:8.3} s", st[0].as_str().unwrap_or(""), st[1].as_f64().unwrap_or(0.0));
        }
        return ExitCode::SUCCESS;
    }
    if report["pass"].as_bool() == Some(true) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn tensor(args: &mut dyn Iterator<Item = String>) -> ExitCode {
    let (mut model, mut key, mut hess, mut count, mut out) = (None, None, None, None, None);
    let (mut bits, mut seed, mut parity, mut guard) = (2.0f64, 0u64, false, false);
    let mut mode = gb_quant::gpu::cublas::MathMode::Pedantic;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next(),
            "--key" => key = args.next(),
            "--hessian" => hess = args.next(),
            "--out" => out = args.next(),
            "--count" => match args.next().and_then(|v| v.parse::<u64>().ok()) {
                Some(v) => count = Some(v),
                None => return usage("--count needs a row count"),
            },
            "--bits" => match args.next().and_then(|v| v.parse::<f64>().ok()).filter(|v| gb_quant::core::bits_of(*v).is_some()) {
                Some(v) => bits = v,
                None => return usage("--bits needs a width the format stores (1-8, 1.5, 2.5, 3.5)"),
            },
            "--seed" => match args.next().and_then(|v| v.parse::<u64>().ok()) {
                Some(v) => seed = v,
                None => return usage("--seed needs a number"),
            },
            "--parity" => parity = true,
            "--refit-guard" => guard = true,
            "--default-math" => mode = gb_quant::gpu::cublas::MathMode::Default,
            _ => return usage(&format!("unknown option {a}")),
        }
    }
    let (Some(model), Some(key), Some(hess), Some(count), Some(out)) = (model, key, hess, count, out) else {
        return usage("tensor needs --model, --key, --hessian, --count and --out");
    };
    match run_tensor(&model, &key, &hess, count, &out, bits, seed, parity, guard, mode) {
        Ok(v) => {
            println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn pad_to(v: &[f32], rows: usize, cols: usize, rp: usize, cp: usize) -> Vec<f32> {
    let mut out = vec![0f32; rp * cp];
    for i in 0..rows {
        out[i * cp..i * cp + cols].copy_from_slice(&v[i * cols..i * cols + cols]);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_tensor(
    model: &str,
    key: &str,
    hess: &str,
    count: u64,
    out: &str,
    bits: f64,
    seed: u64,
    parity: bool,
    guard: bool,
    mode: gb_quant::gpu::cublas::MathMode,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    use gb_quant::core;
    let src = gb_quant::source::find(std::path::Path::new(model), key)?;
    let w = gb_quant::source::read(&src, key)?;
    if w.shape.len() != 2 {
        return Err(format!("{key}: shape {:?} is not a matrix", w.shape).into());
    }
    let (n0, k0) = (w.shape[0], w.shape[1]);
    let h = gb_quant::source::read(std::path::Path::new(hess), "H")?;
    if h.shape != [k0, k0] {
        return Err(format!("H is {:?}, the weight has {k0} inputs", h.shape).into());
    }
    let (k, n) = (k0.div_ceil(128) * 128, n0.div_ceil(128) * 128);
    // quantizer orientation: W = weight^T (k = in, n = out), zero-padded to multiples of 128
    let mut wt = vec![0f32; k0 * n0];
    for o in 0..n0 {
        for i in 0..k0 {
            wt[i * n0 + o] = w.data[o * k0 + i];
        }
    }
    let wt = pad_to(&wt, k0, n0, k, n);
    let hp = pad_to(&h.data, k0, k0, k, k);
    let mut opts = if parity { core::Options::parity(bits) } else { core::Options::new(bits) };
    opts.refit_guard = guard;
    let mut g = core::dev::Gpu::new(mode)?;
    let t0 = std::time::Instant::now();
    let prep = core::prepare(&g, Some(&hp), k, count, &core::signs(seed, 0, k), &opts)?;
    let r = core::quantize(
        &mut g,
        &prep,
        &core::Tensor { weight: &wt, k, n, out_sensitivity: None },
        &core::signs(seed, 1, n),
        &opts,
    )?;
    let secs = t0.elapsed().as_secs_f64();
    let b = opts.bits()?;
    // the stored parts are named after the linear (a trailing ".weight" dropped)
    let key = key.strip_suffix(".weight").unwrap_or(key);
    let parts = vec![
        gb_quant::pack::Part::f16(format!("{key}.suh"), vec![k], &r.suh),
        gb_quant::pack::Part::f16(format!("{key}.svh"), vec![n], &r.svh),
        gb_quant::pack::Part::i16(format!("{key}.trellis"), vec![k / 16, n / 16, b.tile_words()], &r.trellis),
        gb_quant::pack::Part::i32_scalar(format!("{key}.mul1"), core::MUL1_MARKER),
    ];
    gb_quant::pack::write_safetensors(std::path::Path::new(out), &parts, None)?;
    Ok(serde_json::json!({
        "key": key, "in": k0, "out": n0, "padded": [k, n], "bits": bits, "seed": seed, "energy_order": opts.energy_order,
        "refit_guard": guard, "refit_kept": r.refit_kept, "g": r.g, "proxy_err": r.proxy_err, "proxy_err_rounding": r.proxy_err_rounding,
        "out_scales": r.apply_out_scales, "seconds": secs,
        "stages": r.times.stages.iter().map(|(n, v)| serde_json::json!([n, v])).collect::<Vec<_>>(),
    }))
}

fn capture(args: &mut dyn Iterator<Item = String>) -> ExitCode {
    use gb_quant::teacher::{run, CaptureGemm, Options};
    let (mut model, mut rows, mut out, mut oracle) = (None, None, None, None);
    let mut cfg_layers = 1usize;
    let (mut cal_rows, mut cal_cols) = (None, None);
    let mut batch_rows = 4usize;
    let mut bits = 2.0f64;
    let mut opts = Options::default();
    let mut mode = gb_quant::gpu::cublas::MathMode::Pedantic;
    while let Some(a) = args.next() {
        let num = |v: Option<String>| v.and_then(|x| x.parse::<usize>().ok()).filter(|x| *x >= 1);
        match a.as_str() {
            "--model" => model = args.next(),
            "--rows" => rows = args.next(),
            "--out" => out = args.next(),
            "--oracle" => oracle = args.next(),
            "--layers" => match num(args.next()) {
                Some(v) => cfg_layers = v,
                None => return usage("--layers needs a positive number"),
            },
            "--cal-rows" => match num(args.next()) {
                Some(v) => cal_rows = Some(v),
                None => return usage("--cal-rows needs a positive number"),
            },
            "--cal-cols" => match num(args.next()) {
                Some(v) => cal_cols = Some(v),
                None => return usage("--cal-cols needs a positive number"),
            },
            "--batch-rows" => match num(args.next()) {
                Some(v) => batch_rows = v,
                None => return usage("--batch-rows needs a positive number"),
            },
            "--bits" => match args.next().and_then(|v| v.parse::<f64>().ok()).filter(|k| gb_quant::core::bits_of(*k).is_some()) {
                Some(v) => bits = v,
                None => return usage("--bits needs a supported rate"),
            },
            "--capture-gemm" => match args.next().as_deref() {
                Some("wide") => opts.capture_gemm = CaptureGemm::Wide,
                Some("batched") => opts.capture_gemm = CaptureGemm::Batched,
                _ => return usage("--capture-gemm is wide or batched"),
            },
            "--default-math" => mode = gb_quant::gpu::cublas::MathMode::Default,
            _ => return usage(&format!("unknown option {a}")),
        }
    }
    let (Some(model), Some(rows), Some(out)) = (model, rows, out) else {
        return usage("capture needs --model, --rows and --out");
    };
    let cfg = run::RunConfig {
        model: model.into(),
        rows: rows.into(),
        cal_rows,
        cal_cols,
        batch_rows,
        layers: cfg_layers,
        bits,
        out: out.into(),
        oracle: oracle.map(Into::into),
        mode,
        opts,
    };
    match run::run(&cfg) {
        Ok(v) => {
            println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
            let failed = v["modules"].as_array().is_some_and(|ms| {
                ms.iter().any(|m| {
                    let o = &m["oracle"];
                    let s = &m["stream_in"];
                    (!o.is_null() && (o["identical"] != o["qmaps"] || o["routing_identical"] != true))
                        || (!s.is_null() && s["differing"] != 0)
                })
            });
            if failed {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("info") => match args.next() {
            None => info(),
            Some(a) => usage(&format!("info takes no arguments, got {a}")),
        },
        Some("parity") => run_dump(false, &mut args),
        Some("bench") => run_dump(true, &mut args),
        Some("tensor") => tensor(&mut args),
        Some("capture") => capture(&mut args),
        Some("-h" | "--help") => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some(c) => usage(&format!("unknown command {c}")),
        None => usage("missing command"),
    }
}
