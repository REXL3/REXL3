//! Packed EXL3 fixture shared by MoE kernel tests.
use super::Rng;
use gb_exl3::checkpoint::{Group, LoadedGroup};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::{Bits, Codebook};

fn scale_vec(r: &mut Rng, n: usize, mag: f64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            f16_to_f64(f64_to_f16(
                (if r.next_u64() & 1 == 1 { -1.0 } else { 1.0 }) * mag * (0.5 + (r.next_u64() >> 40) as f64 / 16777216.0),
            ))
        })
        .collect()
}
/// Deterministic random packed trellis and f16-exact scales, in the checkpoint's tile layout.
pub fn synth_group(k: usize, n: usize, bits: usize, cb: Codebook, seed: u64) -> LoadedGroup {
    let mut r = Rng(seed);
    let words = (k / 16) * (n / 16) * 16 * bits;
    let trellis: Vec<i16> = (0..words).map(|_| r.next_u64() as i16).collect();
    let group = Group {
        prefix: "synthetic".into(),
        bits: Bits::from_twice(2 * bits as u8).unwrap(),
        codebook: cb,
        k,
        n,
        in_scales: "suh",
        out_scales: "svh",
        bias: false,
        files: vec![],
        bytes: 0,
    };
    let svh = scale_vec(&mut r, n, 1.0 / (k as f64).sqrt());
    LoadedGroup { group, trellis, kt: k / 16, nt: n / 16, suh: scale_vec(&mut r, k, 1.0), svh, bias: None }
}
