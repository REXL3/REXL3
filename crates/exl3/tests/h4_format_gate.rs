//! R-HEAD4 (a), the format gate of the 4-bit MTP head, engine side.
//!
//! `tools/nemotron_omni/h4_format_gate.py` quantizes one head routed expert with the ExLlamaV3 quantizer at the trunk's settings, writes
//! its `.trellis`, `.suh`, `.svh` and `.mul1` under the trunk's names and dtypes (with an index), and records the quantizer's own decode
//! of those packed parts (`wdec_f64.bin`, `[K, N]` float64) and W_q's FNV-1a (`gate.json`). This loads the group through the engine's
//! checkpoint scan (the index places it), decodes it with the host EXL3 path (`gb_exl3::unpack` for W_q, `gb_exl3::forward` for the
//! layer: what `mtp_ref` runs on a quantized head) and requires W_q bit for bit and the layer within 1e-6 of the matrix's norm.
//!
//! Off by default: `REXL3_H4_GATE_DIR=<dir> cargo test --release -p gb-exl3 --test h4_format_gate -- --ignored --nocapture`

use gb_exl3::checkpoint::scan_indexed;
use gb_exl3::{codebook, decode_weight, fnv1a, forward, unpack, Codebook, MARKER_MUL1};
use serde_json::Value;
use std::path::Path;

#[test]
#[ignore]
fn a_quantized_head_matrix_decodes_in_the_engine_as_the_quantizer_decodes_it() {
    let dir = std::env::var("REXL3_H4_GATE_DIR").expect("set REXL3_H4_GATE_DIR to the format gate's output folder");
    let dir = Path::new(&dir);
    let gate: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("gate.json")).unwrap()).unwrap();
    let prefix = gate["prefix"].as_str().unwrap();
    let (k, n) = (gate["k"].as_u64().unwrap() as usize, gate["n"].as_u64().unwrap() as usize);

    let ck = scan_indexed(dir, true).unwrap();
    assert!(ck.bad.is_empty(), "refused groups: {:?}", ck.bad);
    assert!(ck.issues.is_empty(), "scan issues: {:?}", ck.issues);
    let g = ck.groups.get(prefix).unwrap_or_else(|| panic!("{prefix} is not an EXL3 group of the scan: {:?}", ck.groups.keys()));
    assert_eq!(g.codebook, Codebook::Mul1, "codebook");
    assert_eq!(g.bits.twice(), 8, "4 bits per weight");
    assert_eq!((g.k, g.n), (k, n), "shape");
    assert_eq!((g.in_scales, g.out_scales, g.bias), ("suh", "svh", false), "fp16 scales, no bias");
    assert_eq!(ck.markers.get(&format!("{prefix}.mul1")).copied(), Some(MARKER_MUL1), "the mul1 marker");

    let lg = ck.load_group(prefix).unwrap();
    let wq = unpack(&lg.trellis, lg.kt, lg.nt, lg.group.bits, &codebook(Codebook::Mul1));
    let want8: Vec<u16> = gate["first8_fp16_bits"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u16).collect();
    assert_eq!(&wq[..8], &want8[..], "W_q's first values");
    let fnv = fnv1a(wq.iter().flat_map(|x| x.to_le_bytes()));
    assert_eq!(fnv, gate["wq_fnv"].as_u64().unwrap(), "W_q differs from exllamav3_ext.reconstruct's");

    // the layer applied to the identity: row i is e_i through `forward`, i.e. the decoded weight's row i, [K, N]
    let mut eye = vec![0f64; k * k];
    for i in 0..k {
        eye[i * k + i] = 1.0;
    }
    let ours = forward(&eye, k, &lg.trellis, lg.kt, lg.nt, lg.group.bits, &codebook(Codebook::Mul1), &lg.suh, &lg.svh, None);
    let bytes = std::fs::read(dir.join("wdec_f64.bin")).unwrap();
    assert_eq!(bytes.len(), k * n * 8, "wdec_f64.bin is not [{k}, {n}] float64");
    let theirs: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let (mut d2, mut t2, mut maxd) = (0f64, 0f64, 0f64);
    for (a, b) in ours.iter().zip(&theirs) {
        d2 += (a - b) * (a - b);
        t2 += b * b;
        maxd = maxd.max((a - b).abs());
    }
    let rel = (d2 / t2).sqrt();
    eprintln!(
        "{prefix}: K={k} N={n} mul1 4 bits; W_q fnv {fnv:#018x} equal; engine decode vs quantizer decode rel {rel:.3e} (max abs {maxd:.3e}, \
         norm {:.6e})",
        t2.sqrt()
    );
    assert!(rel <= 1e-6, "rel diff {rel:e} > 1e-6 of the matrix norm");

    // the dense host decode (`gb_exl3::decode_weight`, what the reference loads a quantized head with) is the same weight
    let dense = decode_weight(&lg.trellis, lg.kt, lg.nt, lg.group.bits, &codebook(Codebook::Mul1), &lg.suh, &lg.svh);
    let (mut d2, mut t2) = (0f64, 0f64);
    for (a, b) in dense.iter().zip(&theirs) {
        d2 += (a - b) * (a - b);
        t2 += b * b;
    }
    let rel_dense = (d2 / t2).sqrt();
    eprintln!("{prefix}: dense host decode vs quantizer decode rel {rel_dense:.3e}");
    assert!(rel_dense <= 1e-6, "dense decode rel diff {rel_dense:e} > 1e-6 of the matrix norm");
}
