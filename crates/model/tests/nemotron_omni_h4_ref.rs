//! The 4-bit MTP head's calibration, CPU side: `mtp_ref` on rows of a tap dump (`REXL3_NEMO_TAP_DUMP`), so the calibration's torch port
//! of the head (`tools/nemotron_omni/h4_head_calib.py --verify`) can be checked against the reference before its Hessians are trusted.
//! No GPU: the reference on the pack's head weights (bf16, or a 4-bit head decoded on the host), about 12 GB of host memory.
//!
//! `REXL3_NEMO_OMNI_DIR=<pack> REXL3_H4_DUMP_FILE=<taps-*.bin> REXL3_H4_REF_OUT=<dir> [REXL3_H4_REF_ROWS=16]
//!  cargo test --release -p gb-model --test nemotron_omni_h4_ref -- --ignored --nocapture`
//!
//! Writes `<dir>/{fused,after_attn,after_moe,out}.f32` (`[rows, 4096]` f32 little-endian) and `<dir>/ref.json` (the record used, the
//! rows, the token ids the head was fed): row `i` pairs the tap of position `first + i` with the token at `first + i + 1`.

mod nemotron_omni_mtp_support;

use gb_model::families::nemotron_h_omni::mtp_ref::{forward_taps, MtpKv};
use gb_model::families::nemotron_h_omni::tapdump;
use nemotron_omni_mtp_support::hostref::Pack;
use nemotron_omni_mtp_support::{config, pack_dir};
use std::path::PathBuf;

#[test]
#[ignore]
fn mtp_ref_on_the_rows_of_a_tap_dump() {
    let (Some(dir), Some(dump), Some(out)) =
        (pack_dir(), std::env::var_os("REXL3_H4_DUMP_FILE"), std::env::var_os("REXL3_H4_REF_OUT").map(PathBuf::from))
    else {
        eprintln!("not run: set REXL3_NEMO_OMNI_DIR, REXL3_H4_DUMP_FILE and REXL3_H4_REF_OUT");
        return;
    };
    let rows: usize = std::env::var("REXL3_H4_REF_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let bytes = std::fs::read(&dump).unwrap();
    let records = tapdump::decode(&bytes).unwrap_or_else(|at| panic!("malformed record at byte {at}"));
    // the first prefill record that starts a sequence and holds one row more than asked (the last row's token closes the pairs)
    let r = records
        .iter()
        .find(|r| r.kind == 0 && r.first == 0 && r.ids.len() > rows)
        .unwrap_or_else(|| panic!("no prefill record from position 0 with more than {rows} rows"));
    let h = r.width;
    assert_eq!(h, 4096);
    let cfg = config(&dir);
    let pack = Pack::open(&dir);
    let t = std::time::Instant::now();
    let w = pack.head_weights(&cfg);
    eprintln!("host head weights loaded in {:.1} s", t.elapsed().as_secs_f64());
    let next: Vec<u32> = r.ids[1..=rows].iter().map(|&i| i as u32).collect();
    let embed = pack.bf16_rows("language_model.backbone.embeddings.weight", &next, h);
    let positions: Vec<usize> = (0..rows).collect();
    let mut kv = MtpKv::new(&w.geom);
    let taps = forward_taps(&w, &embed, &r.taps[..rows * h], &positions, &mut kv);
    std::fs::create_dir_all(&out).unwrap();
    for (name, v) in
        [("fused", &taps.fused), ("after_attn", &taps.after_attn), ("after_moe", &taps.after_moe), ("out", &taps.out)]
    {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(out.join(format!("{name}.f32")), b).unwrap();
    }
    let meta = serde_json::json!({
        "dump": dump.to_string_lossy(), "slot": r.slot, "first": r.first, "rows": rows,
        "ids": r.ids[..=rows].to_vec(), "pack": dir.to_string_lossy(),
    });
    std::fs::write(out.join("ref.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    eprintln!("mtp_ref over {rows} rows of slot {} written to {}", r.slot, out.display());
}
