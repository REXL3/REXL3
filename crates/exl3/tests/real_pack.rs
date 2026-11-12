//! Real tensors against ExLlamaV3's own kernel.
//!
//! `testdata/fixtures/real_pack.json` holds, for real groups of a published EXL3 pack, the FNV-1a hash of W_q (fp16
//! bits) as reconstructed by `exllamav3_ext.reconstruct` (see `tools/real_pack_crosscheck.py`). This decodes the same
//! groups with `gb_exl3::unpack` and requires every hash to match. Every other fixture comes from TensorFold's
//! Python, so this is the only check against a second, independent implementation on real data.
//!
//! Off by default because it needs the pack: `REXL3_PACK=/path/to/pack cargo test --release -- --ignored real_pack`

use gb_exl3::checkpoint::scan;
use gb_exl3::{codebook, fnv1a, unpack, Codebook};
use serde_json::Value;
use std::path::Path;

#[test]
#[ignore]
fn real_pack_decodes_like_exllamav3() {
    let root = std::env::var("REXL3_PACK").expect("set REXL3_PACK to the model folder the fixture was made from");
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/fixtures/real_pack.json");
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(p).expect("run tools/real_pack_crosscheck.py")).unwrap();
    let ck = scan(Path::new(&root), true).unwrap();
    assert!(ck.bad.is_empty(), "{:?}", ck.bad);
    let groups = fx["groups"].as_array().unwrap();
    assert!(groups.len() >= 8, "the fixture should cover several widths");
    let mut widths = std::collections::BTreeSet::new();
    for g in groups {
        let prefix = g["prefix"].as_str().unwrap();
        let lg = ck.load_group(prefix).unwrap_or_else(|e| panic!("{prefix}: {e}"));
        let want_cb = Codebook::from_name(g["codebook"].as_str().unwrap()).unwrap();
        assert_eq!(lg.group.codebook, want_cb, "{prefix}");
        assert_eq!(lg.group.bits.twice() as u64, 2 * g["bits"].as_u64().unwrap(), "{prefix}");
        assert_eq!((lg.group.k as u64, lg.group.n as u64), (g["k"].as_u64().unwrap(), g["n"].as_u64().unwrap()), "{prefix}");
        let w = unpack(&lg.trellis, lg.kt, lg.nt, lg.group.bits, &codebook(want_cb));
        let first8: Vec<u64> = w[..8].iter().map(|&x| x as u64).collect();
        let want8: Vec<u64> = g["first8_fp16_bits"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert_eq!(first8, want8, "{prefix}: first values differ");
        let got = fnv1a(w.iter().flat_map(|x| x.to_le_bytes()));
        assert_eq!(got, g["wq_fnv"].as_u64().unwrap(), "{prefix}: W_q differs from ExLlamaV3's reconstruction");
        widths.insert(lg.group.bits.twice());
        eprintln!("{prefix}: {} bits {} matches ExLlamaV3", lg.group.bits.as_f64(), want_cb.name());
    }
    assert!(widths.len() >= 4, "widths covered: {widths:?}");
}
