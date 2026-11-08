//! `exl3-inspect MODEL_DIR [--bandwidth-gbs 240] [--rows 1,2,4,6,8]`
//!
//! Gate 0 tool: reads safetensors headers only (no weight data) and prints the per-category bit-width
//! mix, bytes on disk, groups that do not parse, anything odd in the headers, and the bytes read per decoded
//! token, which bounds autoregressive speed at a given DRAM bandwidth. 240 GB/s is TensorFold's measured GB10
//! figure, not a spec value; pass your own measurement.
//!
//! For a mixture-of-experts model the traffic counts only the routed experts a step touches (config.json
//! `n_routed_experts` and `num_experts_per_tok`), not every expert. Exit status is 1 when any group is refused.
#![forbid(unsafe_code)]

use gb_exl3::budget::{categorize, moe_shape, num_layers, Bytes, Category};
use gb_exl3::checkpoint::{config_fields, scan};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

fn gb(b: u64) -> f64 {
    b as f64 / 1e9
}

fn usage(msg: &str) -> ExitCode {
    eprintln!("error: {msg}\nusage: exl3-inspect MODEL_DIR [--bandwidth-gbs 240] [--rows 1,2,4,6,8]");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut dir: Option<PathBuf> = None;
    let mut bw = 240.0f64;
    let mut rows: Vec<u64> = vec![1, 2, 4, 6, 8];
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bandwidth-gbs" => match args.next().and_then(|v| v.parse::<f64>().ok()).filter(|v| v.is_finite() && *v > 0.0) {
                Some(v) => bw = v,
                None => return usage("--bandwidth-gbs needs a positive number"),
            },
            "--rows" => match args.next().map(|v| v.split(',').map(|r| r.trim().parse::<u64>()).collect::<Result<Vec<_>, _>>()) {
                Some(Ok(v)) if !v.is_empty() && v.iter().all(|&r| r > 0) => rows = v,
                _ => return usage("--rows needs a comma-separated list of positive integers"),
            },
            _ if a.starts_with("--") => return usage(&format!("unknown option {a}")),
            _ => dir = Some(PathBuf::from(a)),
        }
    }
    let Some(dir) = dir else { return usage("missing MODEL_DIR") };
    let ck = match scan(&dir, true) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: cannot scan {}: {e}", dir.display());
            return ExitCode::from(2);
        }
    };
    println!("model: {}", dir.display());
    match config_fields(&ck.config) {
        Some(f) => println!("config exl3 fields: {}", serde_json::to_string(f).unwrap_or_default()),
        None => println!("config: no exl3 quantization block found"),
    }
    let layers = num_layers(&ck.config);
    let moe = moe_shape(&ck.config);
    // (category, bits, codebook) -> (groups, bytes)
    let mut hist: BTreeMap<(Category, String, &str), (usize, u64)> = BTreeMap::new();
    let mut bytes = Bytes::default();
    for g in ck.groups.values() {
        let c = categorize(&g.prefix, layers);
        let e = hist.entry((c, format!("{:>4}", g.bits.as_f64()), g.codebook.name())).or_default();
        e.0 += 1;
        e.1 += g.bytes;
        bytes.add(c, g.bytes);
    }
    let mut plain_cat: BTreeMap<Category, (usize, u64)> = BTreeMap::new();
    for (name, e) in &ck.plain {
        let c = categorize(name, layers);
        let p = plain_cat.entry(c).or_default();
        p.0 += 1;
        p.1 += e.nbytes();
        bytes.add(c, e.nbytes());
    }
    println!("\nEXL3 groups: {}   plain tensors: {}   unparsed groups: {}", ck.groups.len(), ck.plain.len(), ck.bad.len());
    println!("\n{:<15} {:>5} {:<6} {:>7} {:>10}", "category", "bits", "cb", "groups", "GB");
    for ((c, b, cb), (n, by)) in &hist {
        println!("{:<15} {:>5} {:<6} {:>7} {:>10.3}", c.label(), b, cb, n, gb(*by));
    }
    println!("\n{:<15} {:>7} {:>10}", "plain", "tensors", "GB");
    for (c, (n, by)) in &plain_cat {
        println!("{:<15} {:>7} {:>10.3}", c.label(), n, gb(*by));
    }
    if !ck.bad.is_empty() {
        println!("\nunparsed groups (this engine refuses them):");
        for (p, why) in &ck.bad {
            println!("  {p}: {why}");
        }
    }
    if !ck.issues.is_empty() {
        println!("\nheader issues ({}):", ck.issues.len());
        for i in &ck.issues {
            println!("  {i}");
        }
    }
    println!(
        "\non disk (GB): routed-experts={:.3} decoder={:.3} head={:.3} other={:.3} mtp={:.3} embedding={:.3} vision={:.3}",
        gb(bytes.routed),
        gb(bytes.decoder),
        gb(bytes.head),
        gb(bytes.other),
        gb(bytes.mtp),
        gb(bytes.embedding),
        gb(bytes.vision)
    );
    let all = bytes.all_experts_bound();
    println!("weights if every expert were read per token: {:.3} GB -> ceiling at {bw} GB/s ~ {:.1} tok/s (a dense-model reading; wrong for MoE)", gb(all), if all > 0 { bw / gb(all) } else { 0.0 });
    match moe {
        Some(m) if bytes.routed > 0 => {
            println!(
                "\nmixture of experts: {} routed, {} per token; always-read weights {:.3} GB, routed experts {:.3} GB",
                m.n_routed,
                m.top_k,
                gb(bytes.always_read()),
                gb(bytes.routed)
            );
            println!("(uniform-routing estimate: real routing is correlated across tokens, so wide windows touch fewer experts than this)");
            println!("{:>6} {:>12} {:>12} {:>14}", "rows", "experts hit", "GB/step", "steps/s ceiling");
            for &r in &rows {
                let by = bytes.step_bytes(Some(m), r);
                println!("{:>6} {:>11.1}% {:>12.3} {:>14.1}", r, 100.0 * m.touched_fraction(r), by / 1e9, bw / (by / 1e9));
            }
            let one = bytes.step_bytes(Some(m), 1);
            println!(
                "autoregressive ceiling at {bw} GB/s ~ {:.1} tok/s (weights only; KV and recurrent-state reads come on top)",
                bw / (one / 1e9)
            );
        }
        _ => {
            if all > 0 {
                println!(
                    "autoregressive ceiling at {bw} GB/s ~ {:.1} tok/s (weights only; KV and recurrent-state reads come on top)",
                    bw / gb(all)
                );
            }
        }
    }
    if ck.bad.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
