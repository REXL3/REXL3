//! DX-1 (nemo-only/v012/PASS_RULES_GLM_ADDENDUM_AMEND1.md D-A1): the GLM-5.3-Flash sc_4.10bpw vision tower's EXL3 matrices through REXL3's host
//! decode (`gb_exl3::decode_weight`), CPU only. Both tests are `#[ignore]` and env-gated ("not run" when unset).
//!
//! * `dx1_the_towers_matrices_decode_like_exllamav3` is DX-1's precondition, by the method of R-MK-format (lane E's `mkfmt`): for every matrix
//!   of the reference directory `REXL3_DX1_REF` (written by `tools/glm_dx1_reconstruct_ref.py` with exllamav3 1.6.2 as shipped: `list.json`,
//!   `<i>.wq_f16` = `exllamav3_ext.reconstruct`'s W_q bits `[K_in, N_out]`, `<i>.wdec_f64` = its weight-tensor chain (had_l, times suh,
//!   had_r, times svh) carried out in float64 with its own Hadamard matrix), W_q from `gb_exl3::unpack` must be bit for bit theirs and the
//!   dense decode within rel = ||ours - theirs|| / ||theirs|| <= 1e-6. Informational, when the reference holds it: `<i>.wfused_f16`, the
//!   fp16 original-basis weight of exllamav3's fused `reconstruct_had_slice` (the path its tower takes at 1,024 rows or more), against our
//!   float64 decode. Writes one JSON line per matrix to `REXL3_DX1_REPORT` when set.
//! * `dx1_the_towers_matrices_are_dumped`: every EXL3 group under `model.visual.` decoded to `REXL3_DX1_OUT/<prefix>.f64` (raw little-endian
//!   float64, row-major `[k_in, n_out]`: `y = x @ W (+ bias)`) and `list.json` (`{"matrices": [{"prefix", "k", "n", "bits", "bias", ...}]}`):
//!   the dequantized weights the fp64 host forward (`tools/glm_vision_fp64.py`) reads.
//!
//! `REXL3_DX1_PACK` is the pack directory (the sc_4.10bpw pack on N2/N3: `~/projects/glm53-flash-exl3-sc410-c6b8ac3f`).

use gb_exl3::checkpoint::{scan, Checkpoint};
use gb_exl3::{codebook, decode_weight, fnv1a, unpack};
use serde_json::{json, Value};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn pack() -> Option<(PathBuf, Checkpoint)> {
    let Ok(p) = std::env::var("REXL3_DX1_PACK") else {
        eprintln!("REXL3_DX1_PACK is not set: not run");
        return None;
    };
    let t = Instant::now();
    let ck = scan(Path::new(&p), true).unwrap_or_else(|e| panic!("scan {p}: {e}"));
    let visual = ck.groups.keys().filter(|k| k.starts_with("model.visual.")).count();
    eprintln!(
        "scan {p}: {} EXL3 groups ({visual} under model.visual.), {} plain tensors, {} refused groups, {:.1} s",
        ck.groups.len(),
        ck.plain.len(),
        ck.bad.len(),
        t.elapsed().as_secs_f64()
    );
    Some((PathBuf::from(p), ck))
}

/// Reads `n` little-endian values of `size` bytes from `path` in large blocks, handing each to `f` with its index.
fn stream(path: &Path, n: usize, size: usize, mut f: impl FnMut(usize, &[u8])) {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(file.metadata().unwrap().len() as usize, n * size, "{}: size", path.display());
    let mut r = BufReader::with_capacity(1 << 24, file);
    let mut buf = vec![0u8; (1 << 23) / size * size];
    let mut idx = 0usize;
    while idx < n {
        let take = ((n - idx) * size).min(buf.len());
        r.read_exact(&mut buf[..take]).unwrap();
        for c in buf[..take].chunks_exact(size) {
            f(idx, c);
            idx += 1;
        }
    }
}

#[test]
#[ignore]
fn dx1_the_towers_matrices_decode_like_exllamav3() {
    let Ok(refdir) = std::env::var("REXL3_DX1_REF") else {
        eprintln!("REXL3_DX1_REF is not set: not run");
        return;
    };
    let Some((_, ck)) = pack() else { return };
    let refdir = PathBuf::from(refdir);
    let list: Value = serde_json::from_str(&std::fs::read_to_string(refdir.join("list.json")).unwrap()).unwrap();
    let mut report = std::env::var("REXL3_DX1_REPORT").ok().map(|p| std::fs::File::create(p).unwrap());
    let mut all_pass = true;
    let ms = list["matrices"].as_array().unwrap();
    assert!(!ms.is_empty(), "the reference lists no matrix");
    for m in ms {
        let t = Instant::now();
        let i = m["i"].as_u64().unwrap() as usize;
        let prefix = m["prefix"].as_str().unwrap();
        assert!(!ck.bad.contains_key(prefix), "{prefix}: refused by the scan: {:?}", ck.bad.get(prefix));
        let g = ck.groups.get(prefix).unwrap_or_else(|| panic!("{prefix}: not an EXL3 group of the scan"));
        let lg = ck.load_group(prefix).unwrap_or_else(|e| panic!("{prefix}: {e}"));
        let (k, n) = (lg.kt * 16, lg.nt * 16);
        let shape_ok = (k as u64, n as u64) == (m["k"].as_u64().unwrap(), m["n"].as_u64().unwrap())
            && (g.bits.as_f64() - m["bits"].as_f64().unwrap()).abs() < 1e-9;
        let table = codebook(g.codebook);
        // W_q bit for bit
        let wq = unpack(&lg.trellis, lg.kt, lg.nt, g.bits, &table);
        let mut wq_mismatch = 0usize;
        stream(&refdir.join(format!("{i}.wq_f16")), k * n, 2, |j, c| {
            if wq[j] != u16::from_le_bytes([c[0], c[1]]) {
                wq_mismatch += 1;
            }
        });
        let fnv_ours = fnv1a(wq.iter().flat_map(|x| x.to_le_bytes()));
        drop(wq);
        // the dense host decode against exllamav3's float64 chain
        let dense = decode_weight(&lg.trellis, lg.kt, lg.nt, g.bits, &table, &lg.suh, &lg.svh);
        let (mut d2, mut t2, mut maxd) = (0f64, 0f64, 0f64);
        stream(&refdir.join(format!("{i}.wdec_f64")), k * n, 8, |j, c| {
            let b = f64::from_le_bytes(c.try_into().unwrap());
            let d = dense[j] - b;
            d2 += d * d;
            t2 += b * b;
            maxd = maxd.max(d.abs());
        });
        let rel = (d2 / t2).sqrt();
        // informational: exllamav3's fused fp16 original-basis weight (its tower's path at 1,024+ rows) against our float64 decode
        let fused = refdir.join(format!("{i}.wfused_f16"));
        let fused_rel = fused.exists().then(|| {
            let (mut fd2, mut ft2) = (0f64, 0f64);
            stream(&fused, k * n, 2, |j, c| {
                let b = gb_exl3::f16::f16_to_f64(u16::from_le_bytes([c[0], c[1]]));
                fd2 += (b - dense[j]) * (b - dense[j]);
                ft2 += dense[j] * dense[j];
            });
            (fd2 / ft2).sqrt()
        });
        let pass = shape_ok && wq_mismatch == 0 && rel <= 1e-6;
        all_pass &= pass;
        let rec = json!({"i": i, "prefix": prefix, "bits": g.bits.as_f64(), "codebook": format!("{:?}", g.codebook), "k": k, "n": n,
            "bias": g.bias, "shape_bits_match_reference": shape_ok, "wq_mismatches": wq_mismatch, "wq_fnv_rexl3": format!("{fnv_ours:#018x}"),
            "wq_sha256_exllamav3": m["wq_sha256"], "decode_rel_diff": rel, "decode_max_abs_diff": maxd, "norm": t2.sqrt(),
            "fused_fp16_rel_vs_rexl3_f64": fused_rel, "pass": pass, "s": (t.elapsed().as_secs_f64() * 10.0).round() / 10.0});
        eprintln!("{rec}");
        if let Some(f) = report.as_mut() {
            writeln!(f, "{rec}").unwrap();
        }
    }
    eprintln!("DX-1 precondition: {} matrices, all pass: {all_pass}", ms.len());
    assert!(all_pass, "REXL3's host decode differs from exllamav3 1.6.2's reconstruct: DX-1 stops here");
}

#[test]
#[ignore]
fn dx1_the_towers_matrices_are_dumped() {
    let Ok(out) = std::env::var("REXL3_DX1_OUT") else {
        eprintln!("REXL3_DX1_OUT is not set: not run");
        return;
    };
    let out = PathBuf::from(out);
    assert!(!out.exists(), "REXL3_DX1_OUT={} exists: a new directory every run", out.display());
    let Some((p, ck)) = pack() else { return };
    std::fs::create_dir_all(&out).unwrap();
    let mut prefixes: Vec<&String> = ck.groups.keys().filter(|k| k.starts_with("model.visual.")).collect();
    prefixes.sort();
    assert!(!prefixes.is_empty(), "the pack has no EXL3 group under model.visual.");
    let t0 = Instant::now();
    let mut rows = Vec::new();
    for prefix in prefixes {
        let t = Instant::now();
        let g = &ck.groups[prefix.as_str()];
        let lg = ck.load_group(prefix).unwrap_or_else(|e| panic!("{prefix}: {e}"));
        let (k, n) = (lg.kt * 16, lg.nt * 16);
        let dense = decode_weight(&lg.trellis, lg.kt, lg.nt, g.bits, &codebook(g.codebook), &lg.suh, &lg.svh);
        let mut w = BufWriter::with_capacity(1 << 24, std::fs::File::create(out.join(format!("{prefix}.f64"))).unwrap());
        for v in &dense {
            w.write_all(&v.to_le_bytes()).unwrap();
        }
        w.flush().unwrap();
        let norm = dense.iter().map(|v| v * v).sum::<f64>().sqrt();
        rows.push(json!({"prefix": prefix, "k": k, "n": n, "bits": g.bits.as_f64(), "codebook": format!("{:?}", g.codebook), "bias": g.bias,
            "in_scales": g.in_scales, "out_scales": g.out_scales, "norm": norm, "fnv_f64": format!("{:#018x}", fnv1a(dense.iter().flat_map(|v| v.to_le_bytes())))}));
        eprintln!(
            "{prefix}: [{k}, {n}] K{} norm {norm:.6e} decoded and written in {:.2} s",
            g.bits.as_f64(),
            t.elapsed().as_secs_f64()
        );
    }
    let doc = json!({"what": "DX-1: REXL3 host decode (gb_exl3::decode_weight) of the vision tower's EXL3 groups, float64 [k_in, n_out] row-major",
        "pack": p, "matrices": rows});
    std::fs::write(out.join("list.json"), serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    eprintln!(
        "{} matrices written to {} in {:.1} s",
        doc["matrices"].as_array().unwrap().len(),
        out.display(),
        t0.elapsed().as_secs_f64()
    );
}
