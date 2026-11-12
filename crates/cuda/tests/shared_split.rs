//! The shared-expert split (lane E, GLM Phase 2 E2.6): two ranks each run their half of a shared expert up to the down projection's
//! partials ([`SharedExpert::forward_split_partials`]: gate / up strips with their own plans, the clamped SwiGLU, the down K-split
//! partials of the half), write `-0.0` in the peer's partial slots, the two `[sk][rows][hidden]` buffers are summed as the TP all-reduce
//! sums them (in rank order; `x + -0.0 == x`), and [`SharedExpert::finish_split`] gives [`SharedExpert::forward`]'s output bit for bit, at
//! every decode-window row count, at the sc_4.10bpw widths (8 bits) and the 2.25bpw pack's (3 / 2 bits).
//!
//!     cargo test --release -p gb-cuda --test shared_split -- --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]
mod common;

use common::exl3_synth::synth_group;
use common::Rng;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::moe::{SharedExpert, SharedScratch};
use gb_cuda::nemotron_tp::scatter_cols_f32;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::f64_to_f16;
use gb_exl3::Codebook;

fn layer(k: usize, n: usize, bits: usize, seed: u64) -> Exl3Weights {
    Exl3Weights::from_loaded(&synth_group(k, n, bits, Codebook::Mcg, seed)).unwrap()
}

#[test]
fn two_ranks_halves_of_the_shared_expert_finish_to_its_forward_bit_for_bit() {
    if !device_available() {
        eprintln!("no CUDA device: not run");
        return;
    }
    let st = Stream::new().unwrap();
    let (hidden, inter) = (4096usize, 2048usize);
    let mut cases = 0;
    // (gate / up bits, down bits): sc_4.10bpw's shared expert, and the 2.25bpw pack's
    for (bgu, bd) in [(8usize, 8usize), (3, 2)] {
        let sh = SharedExpert::new(
            layer(hidden, inter, bgu, 0x5A1 + bgu as u64),
            layer(hidden, inter, bgu, 0x5A2 + bgu as u64),
            layer(inter, hidden, bd, 0x5A3 + bd as u64),
            10.0,
        )
        .unwrap();
        let (sk, _) = sh.down_plan().unwrap();
        assert!(sk % 2 == 0, "the down projection's plan has {sk} K splits");
        let halves = [(0, sk / 2), (sk / 2, sk)];
        for (r, &(s0, s1)) in halves.iter().enumerate() {
            assert_eq!(sh.down_split_k_range(s0, s1).unwrap(), (r * inter / 2, (r + 1) * inter / 2), "rank {r}'s half");
        }
        for rows in [1usize, 2, 3, 4, 5, 8, 17, 64] {
            let mut rng = Rng(0xC0FFEE + rows as u64 + bgu as u64);
            let x: Vec<u16> = (0..rows * hidden).map(|_| f64_to_f16(rng.uniform(-2.0, 2.0) as f64)).collect();
            let xd = DevBuf::from_slice(&x).unwrap();
            let mut sc = SharedScratch::new(&sh, 64).unwrap();
            let mut full = DevBuf::zeroed(rows * hidden * 4).unwrap();
            sh.forward(&xd, Dtype::F16, rows, &mut full, &mut sc, &st).unwrap();
            st.sync().unwrap();
            let want: Vec<u32> = full.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect();
            // each rank: its partials in its slots, -0.0 in the peer's (the pair buffer's tail on a GLM rank)
            let slot = rows * hidden;
            let mut sum = vec![0f32; sk * slot];
            for (r, &(s0, s1)) in halves.iter().enumerate() {
                let mut z = DevBuf::zeroed_view(sk * slot * 4).unwrap();
                z.memset_async(0x3F, &st).unwrap(); // poison: every byte the rank leaves unwritten would show
                let (p0, p1) = halves[1 - r];
                let mut peer = z.subview(p0 * slot * 4, (p1 - p0) * slot * 4);
                let other = DevBuf::zeroed(16).unwrap();
                scatter_cols_f32(&other, &mut peer, (p1 - p0) * rows, 0, hidden, 0, &st).unwrap();
                let mut ws = DevBuf::zeroed(sh.split_workspace_bytes(s0, s1, rows).unwrap()).unwrap();
                let mut sc_r = SharedScratch::new(&sh, 64).unwrap();
                sh.forward_split_partials(&xd, Dtype::F16, rows, (s0, s1), &mut z, s0 * slot * 4, &mut sc_r, &mut ws, &st)
                    .unwrap();
                st.sync().unwrap();
                let zr: Vec<f32> = z.to_vec().unwrap();
                if r == 0 {
                    sum = zr;
                } else {
                    // the all-reduce: rank 0's buffer plus rank 1's, element by element (IEEE add: x + -0.0 == x for every x)
                    for (a, b) in sum.iter_mut().zip(&zr) {
                        *a += *b;
                    }
                }
            }
            let zd = DevBuf::from_slice(&sum).unwrap();
            let mut got = DevBuf::zeroed(rows * hidden * 4).unwrap();
            sh.finish_split(&zd, rows, &mut got, &st).unwrap();
            st.sync().unwrap();
            let got: Vec<u32> = got.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect();
            if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
                panic!(
                    "bits {bgu}/{bd}, {rows} rows: element {i} (row {}) differs: {:#010x} vs {:#010x}",
                    i / hidden,
                    got[i],
                    want[i]
                );
            }
            cases += 1;
        }
        eprintln!(
            "shared expert {hidden} -> {inter} -> {hidden}, {bgu}/{bd} bits, down plan sk {sk}: the two halves finish to forward"
        );
    }
    eprintln!("{cases} cases: the shared-expert split (strips, SwiGLU, down partials, -0.0 in the peer's slots, summed, finished) == forward bit for bit");
}
