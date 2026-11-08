//! The engine's side of the vision SDPA check (P3, `docs/NEMOTRON_VISION_SDPA.md`): the CURRENT RADIO tower, vision_final_layernorm,
//! pixel shuffle and mlp1 run on the pixels of a cases directory, and every stage's output is written next to the pixels as raw
//! little-endian f32 for `tools/nemotron_omni/vision_sdpa_check.py` to compare with the torch tower in its four reference modes.
//!
//! Env-gated (the test prints "not run" and passes without them; GPU only, run it under the device lock):
//!
//! * `REXL3_NEMO_OMNI_DIR` (or `REXL3_NEMOTRON_PACK`): the pack directory (the vision tensors are read through its index).
//! * `REXL3_NEMO_SDPA_DIR`: the cases directory written by `vision_sdpa_check.py prepare`: `manifest.tsv` (`name <TAB> height <TAB>
//!   width` per line) and `<name>.pixels.bf16` (the HF processor's `pixel_values`, bf16 bits as little-endian u16, `[3, H, W]`).
//!
//! Per case it writes `<name>.engine.tower.f32` (`[patches, 1280]`, the features before vision_final_layernorm), `<name>.engine.final_ln.f32`
//! (`[patches, 1280]`, the fp16 output widened), `<name>.engine.mlp1.f32` (`[patches / 4, 4096]`, mlp1's fp32 output before the encoder's
//! bf16 rounding) and `<name>.engine.blocks.f32` (`[33, 3, 1280]`: the embeddings and the 32 block outputs at the cls token, the first
//! patch and the last token, as the existing tower test's trace).
//!
//! No gate lives here and nothing is asserted about the numbers: the comparison is the script's. The pixels reach the tower exactly as
//! the encoder gives them (bf16 widened to fp16 by round to nearest even; the count of values that fp16 cannot hold exactly, which can only
//! be magnitudes below 6.1e-5 where fp16 turns subnormal, is printed).
#![cfg(not(gb_no_cuda))]

use gb_cuda::nemotron_vit as nv;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use std::path::{Path, PathBuf};

/// `not run: REXL3_NEMO_...` is the form scripts/gpu_gate.sh does not count as a skipped test (these run in scripts/gpu_gate_nemotron.sh).
fn env_dir(names: &[&str]) -> Option<PathBuf> {
    let v = names.iter().find_map(std::env::var_os).map(PathBuf::from);
    if v.is_none() {
        eprintln!("not run: {} is not set", names[0]);
    }
    v
}

fn write_f32(path: &Path, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

#[test]
fn engine_stages_for_the_sdpa_check() {
    if !device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    let (Some(pack), Some(dir)) = (env_dir(&["REXL3_NEMO_OMNI_DIR", "REXL3_NEMOTRON_PACK"]), env_dir(&["REXL3_NEMO_SDPA_DIR"]))
    else {
        return;
    };
    let manifest = std::fs::read_to_string(dir.join("manifest.tsv")).unwrap_or_else(|e| panic!("manifest.tsv: {e}"));
    let cases: Vec<(String, usize, usize)> = manifest
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 3, "manifest line {l:?}: name, height, width");
            (f[0].to_string(), f[1].parse().unwrap(), f[2].parse().unwrap())
        })
        .collect();
    assert!(!cases.is_empty(), "empty manifest");
    let max_patches = cases
        .iter()
        .map(|(n, h, w)| {
            let (gh, gw) = nv::patch_grid(*h, *w).unwrap_or_else(|| panic!("{n}: {h}x{w} is not a runnable size"));
            gh * gw
        })
        .max()
        .unwrap();

    let st = Stream::new().unwrap();
    let w = nv::Weights::load(&pack).unwrap();
    let mut ws = nv::Workspace::new(max_patches).unwrap();
    for (name, h, wd) in &cases {
        let (gh, gw) = nv::patch_grid(*h, *wd).unwrap();
        let n = gh * gw;
        let np = n / 4;
        let raw = std::fs::read(dir.join(format!("{name}.pixels.bf16"))).unwrap_or_else(|e| panic!("{name}: pixels: {e}"));
        assert_eq!(raw.len(), 3 * h * wd * 2, "{name}: pixel bytes");
        // bf16 bits widened to fp16 as `glm_vision::bf16_to_f16` does (round to nearest even); a value fp16 cannot hold exactly is counted
        let mut inexact = 0usize;
        let px16: Vec<u16> = raw
            .chunks_exact(2)
            .map(|c| {
                let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                let b = f64_to_f16(v as f64);
                if f16_to_f64(b) != v as f64 {
                    inexact += 1;
                }
                b
            })
            .collect();
        let pix = DevBuf::from_slice(&px16).unwrap();

        let mut tr = nv::Trace::default();
        nv::tower_forward(&w, &mut ws, &pix, *h, *wd, Some(&mut tr), &st).unwrap();
        st.sync().unwrap();

        let mut feat = vec![0f32; n * nv::HIDDEN];
        ws.h().download_range(nv::FEATURE_OFFSET, &mut feat).unwrap();
        let mut lnf16 = vec![0u16; n * nv::HIDDEN];
        ws.final_ln().download_range(0, &mut lnf16).unwrap();
        let lnf: Vec<f32> = lnf16.iter().map(|&b| f16_to_f64(b) as f32).collect();
        let mut mlp1 = vec![0f32; np * nv::LLM_HIDDEN];
        ws.mlp1().download_range(0, &mut mlp1).unwrap();
        assert!(feat.iter().chain(&lnf).chain(&mlp1).all(|v| v.is_finite()), "{name}: non-finite values");
        assert_eq!(tr.blocks.len(), nv::BLOCKS + 1);
        let blocks: Vec<f32> = tr.blocks.iter().flatten().copied().collect();
        assert_eq!(blocks.len(), (nv::BLOCKS + 1) * 3 * nv::HIDDEN);

        write_f32(&dir.join(format!("{name}.engine.tower.f32")), &feat);
        write_f32(&dir.join(format!("{name}.engine.final_ln.f32")), &lnf);
        write_f32(&dir.join(format!("{name}.engine.mlp1.f32")), &mlp1);
        write_f32(&dir.join(format!("{name}.engine.blocks.f32")), &blocks);
        println!("{name}: {h}x{wd}, {n} patches, {np} tokens; pixels fp16-inexact {inexact} of {}", px16.len());
    }
}
