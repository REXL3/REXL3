//! Before / after digests of the published pack's real MoE layers (v0.1.2 lane K, R-MK-noreg (b)): the decode path (`forward_in`), the
//! P1 prefill path (`forward_prefill_in`) and the P2 tensor-core block (`forward_prefill_mma`) on fixed inputs, at the decode and
//! verify windows and at prefill widths, for layers with 4-bit experts only and with bf16 experts among them. Each line is printed as
//! `DIGEST <layer> <path> <rows> <out> <logits> <picks>` (FNV-1a 64 of the f32 / i32 bits); the same file built on the tree before the
//! MixedK work (with that tree's `Weights`, which has no gate tag) prints the lines the candidate must reproduce exactly.
//!
//!     REXL3_NEMO_OMNI_DIR=<our pack> cargo test -p gb-cuda --release --test nemotron_moe_k4_digest -- --nocapture --test-threads=1
#![cfg(not(gb_no_cuda))]
mod common;

use common::Rng;
use gb_cuda::nemotron_moe::{MoeScratch, Proj, Weights, EXPERTS, HIDDEN, TOPK};
use gb_cuda::nemotron_moe_prefill::mma::{self, MmaScratch};
use gb_cuda::nemotron_moe_prefill::{PrefillMoe, PrefillScratch};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::{scan_indexed, Checkpoint};
use gb_exl3::fnv1a;
use std::path::PathBuf;

const SCALE: f32 = 5.0;

fn digest_f32(v: &[f32]) -> u64 {
    fnv1a(v.iter().flat_map(|x| x.to_bits().to_le_bytes()))
}

fn digest_i32(v: &[i32]) -> u64 {
    fnv1a(v.iter().flat_map(|x| x.to_le_bytes()))
}

fn plain_u16(ck: &Checkpoint, name: &str) -> Vec<u16> {
    let e = ck.plain.get(name).unwrap_or_else(|| panic!("the pack has no plain tensor {name}"));
    assert_eq!(e.dtype, "BF16", "{name}");
    ck.read_u16(e).unwrap()
}

fn plain_f32(ck: &Checkpoint, name: &str) -> Vec<f32> {
    let e = ck.plain.get(name).unwrap_or_else(|| panic!("the pack has no plain tensor {name}"));
    ck.read_floats(e).unwrap().into_iter().map(|v| v as f32).collect()
}

/// An EXL3 projection, or the bf16 `{prefix}.weight` `[n, k]` when the pack stores it so.
fn proj(ck: &Checkpoint, prefix: &str, k: usize, n: usize) -> Proj {
    match ck.load_group(prefix) {
        Ok(g) => Proj::exl3_from_group(&g).unwrap(),
        Err(_) => Proj::bf16(k, n, &plain_u16(ck, &format!("{prefix}.weight"))).unwrap(),
    }
}

fn layer(ck: &Checkpoint, i: usize) -> Weights {
    use gb_cuda::nemotron_moe::{EXPERT_INTER, LATENT, SHARED_INTER};
    let m = format!("language_model.backbone.layers.{i}.mixer");
    let up = (0..EXPERTS).map(|e| proj(ck, &format!("{m}.experts.{e}.up_proj"), LATENT, EXPERT_INTER)).collect();
    let down = (0..EXPERTS).map(|e| proj(ck, &format!("{m}.experts.{e}.down_proj"), EXPERT_INTER, LATENT)).collect();
    Weights {
        gate: DevBuf::from_slice(&plain_u16(ck, &format!("{m}.gate.weight"))).unwrap(),
        gate_dtype: gb_cuda::nemotron_moe::GateDtype::Bf16,
        router_bias: DevBuf::from_slice(&plain_f32(ck, &format!("{m}.gate.e_score_correction_bias"))).unwrap(),
        scale: SCALE,
        fc1: proj(ck, &format!("{m}.fc1_latent_proj"), HIDDEN, LATENT),
        fc2: proj(ck, &format!("{m}.fc2_latent_proj"), LATENT, HIDDEN),
        shared_up: proj(ck, &format!("{m}.shared_experts.up_proj"), HIDDEN, SHARED_INTER),
        shared_down: proj(ck, &format!("{m}.shared_experts.down_proj"), SHARED_INTER, HIDDEN),
        up,
        down,
    }
}

#[test]
fn published_pack_layers_digest_on_every_path() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let Some(dir) = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set");
        return;
    };
    let ck = scan_indexed(&dir, true).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let st = Stream::new().unwrap();
    let max = 1024;
    let h = Rng(0xD16E57).vec(max * HIDDEN, -1.0, 1.0);
    let mut ms = MoeScratch::new(64).unwrap();
    let mut ps = PrefillScratch::new(max).unwrap();
    let mut mm = MmaScratch::new(max).unwrap();
    // layer 1 (4-bit experts), 26 (bf16 experts 111 among them), 81 (bf16 expert 461), 87 (the last E layer)
    for li in [1usize, 26, 81, 87] {
        let moe = PrefillMoe::tables(layer(&ck, li)).unwrap();
        let line = |path: &str, rows: usize, out: &DevBuf, logits: Vec<f32>, picks: Vec<i32>| {
            st.sync().unwrap();
            let o = out.to_vec::<f32>().unwrap();
            println!(
                "DIGEST {li:2} {path:6} {rows:4} {:016x} {:016x} {:016x}",
                digest_f32(&o[..rows * HIDDEN]),
                digest_f32(&logits),
                digest_i32(&picks)
            );
        };
        for rows in [1usize, 2, 7, 64] {
            let hd = DevBuf::from_slice(&h[..rows * HIDDEN]).unwrap();
            let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
            ms.poison(&st).unwrap();
            moe.forward_in(&mut ms, &hd, rows, &mut out, &st).unwrap();
            let (pick, _) = ms.download_route(rows, &st).unwrap();
            line("decode", rows, &out, ms.download_logits(rows, &st).unwrap(), pick);
        }
        for rows in [65usize, 256, 512] {
            let hd = DevBuf::from_slice(&h[..rows * HIDDEN]).unwrap();
            let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
            ps.poison(&st).unwrap();
            moe.forward_prefill_in(&mut ps, &hd, rows, &mut out, &st).unwrap();
            let (pick, _) = ps.download_route(rows, &st).unwrap();
            line("p1", rows, &out, ps.download_logits(rows, &st).unwrap(), pick);
        }
        mma::set_tile(4);
        for rows in [256usize, 1024] {
            let hd = DevBuf::from_slice(&h[..rows * HIDDEN]).unwrap();
            let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
            ps.poison(&st).unwrap();
            mm.poison(&st).unwrap();
            moe.forward_prefill_mma(&mut ps, &mut mm, &hd, rows, &mut out, &st).unwrap();
            let (pick, _) = ps.download_route(rows, &st).unwrap();
            line("p2", rows, &out, ps.download_logits(rows, &st).unwrap(), pick);
        }
        let _ = TOPK;
    }
}
