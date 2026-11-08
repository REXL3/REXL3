//! `nemotron_ssm::commit_all` (the replay of every Mamba layer in two launches) against the single-layer calls it replaces:
//! `commit_state` and `commit_conv` per layer. Several layers with their own records, retained rows and pools; sequences of unequal
//! lengths in non-identity slots; keeps from 0 (state untouched) through the whole sequence. Every byte of every pool must match.

#![cfg(not(gb_no_cuda))]

use gb_cuda::gdn::BatchBuf;
use gb_cuda::nemotron_ssm::*;
use gb_cuda::num::f32_to_bf16;
use gb_cuda::{device_available, DevBuf, Stream};

fn splitmix64(z: u64) -> u64 {
    let z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Rng(u64);

impl Rng {
    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let v = splitmix64(self.0);
                self.0 = self.0.wrapping_add(1);
                lo + (hi - lo) * ((v >> 40) as f32 / 16_777_216.0)
            })
            .collect()
    }
}

fn bf16(v: &[f32]) -> DevBuf {
    let u: Vec<u16> = v.iter().map(|&x| f32_to_bf16(x)).collect();
    DevBuf::from_slice(&u).unwrap()
}

/// One layer's inputs and its two pools.
struct Layer {
    rec: Record,
    xin: DevBuf,
    s: Vec<f32>,
    conv: Vec<u16>,
}

#[test]
fn commit_all_is_the_single_layer_commits_of_every_layer() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    let mut rng = Rng(0xc0ffee);
    let lens = [7usize, 1, 20, 3];
    let slots = [2usize, 0, 4, 1];
    let n_slots = 6;
    let rows: usize = lens.iter().sum();
    let n_layers = 4;

    // a shared set of weights; each layer gets its own record by running `pre` on its own inputs
    let conv_w = bf16(&rng.vec(CONV_DIM * CONV_TAPS, -0.5, 0.5));
    let conv_b = bf16(&rng.vec(CONV_DIM, -0.5, 0.5));
    let a_log = DevBuf::from_slice(&rng.vec(HEADS, -1.0, 1.0)).unwrap();
    let dt_bias = DevBuf::from_slice(&rng.vec(HEADS, -1.0, 1.0)).unwrap();
    let w = Weights { conv_w: &conv_w, conv_b: &conv_b, a_log: &a_log, dt_bias: &dt_bias, d: &a_log };
    let bb = BatchBuf::with_pool_slots(&lens, &slots, n_slots).unwrap();
    let batch = bb.view();

    let mut layers: Vec<Layer> = Vec::new();
    for _ in 0..n_layers {
        let xin = DevBuf::from_slice(&rng.vec(rows * CONV_DIM, -2.0, 2.0)).unwrap();
        let dt_raw = DevBuf::from_slice(&rng.vec(rows * HEADS, -2.0, 2.0)).unwrap();
        let s = rng.vec(n_slots * S_SLOT_ELEMS, -1.0, 1.0);
        let conv: Vec<u16> = rng.vec(n_slots * CONV_SLOT_ELEMS, -2.0, 2.0).iter().map(|&x| f32_to_bf16(x)).collect();
        let mut rec = Record::alloc(rows).unwrap();
        let conv_dev = DevBuf::from_slice(&conv).unwrap();
        pre(&st, &xin, &dt_raw, &w, &conv_dev, &batch, &mut rec).unwrap();
        layers.push(Layer { rec, xin, s, conv });
    }
    st.sync().unwrap();

    for keep in [vec![0i32, 0, 0, 0], vec![1, 1, 1, 1], vec![3, 1, 20, 0], vec![7, 1, 20, 3]] {
        let keep_dev = DevBuf::from_slice(&keep).unwrap();
        // the single-layer calls
        let mut want: Vec<(Vec<u32>, Vec<u16>)> = Vec::new();
        for l in &layers {
            let mut s = DevBuf::from_slice(&l.s).unwrap();
            let mut conv = DevBuf::from_slice(&l.conv).unwrap();
            commit_state(&st, &l.rec, &mut s, &batch, &keep_dev).unwrap();
            commit_conv(&st, &l.xin, &mut conv, &batch, &keep_dev).unwrap();
            st.sync().unwrap();
            want.push((s.to_vec::<u32>().unwrap(), conv.to_vec::<u16>().unwrap()));
        }
        // the one-launch replay over the same layers
        let mut s_bufs: Vec<DevBuf> = layers.iter().map(|l| DevBuf::from_slice(&l.s).unwrap()).collect();
        let mut conv_bufs: Vec<DevBuf> = layers.iter().map(|l| DevBuf::from_slice(&l.conv).unwrap()).collect();
        let table = {
            let tl: Vec<CommitLayer<'_>> = layers
                .iter()
                .zip(s_bufs.iter().zip(conv_bufs.iter()))
                .map(|(l, (s, conv))| CommitLayer { rec: &l.rec, xin: &l.xin, s, conv })
                .collect();
            CommitTable::new(&tl).unwrap()
        };
        commit_all(&st, &table, &batch, &keep_dev).unwrap();
        st.sync().unwrap();
        for (i, (ws, wc)) in want.iter().enumerate() {
            let gs = s_bufs[i].to_vec::<u32>().unwrap();
            let gc = conv_bufs[i].to_vec::<u16>().unwrap();
            assert!(gs == *ws, "keep {keep:?}, layer {i}: S pools differ at {:?}", gs.iter().zip(ws).position(|(a, b)| a != b));
            assert!(
                gc == *wc,
                "keep {keep:?}, layer {i}: conv pools differ at {:?}",
                gc.iter().zip(wc).position(|(a, b)| a != b)
            );
        }
        // a layer's pools did move for a non-zero keep (the comparison is not vacuous)
        if keep.iter().any(|&k| k > 0) {
            let init: Vec<u32> = layers[0].s.iter().map(|x| x.to_bits()).collect();
            assert!(want[0].0 != init, "keep {keep:?}: the commit left S unchanged");
        }
        s_bufs.clear();
        conv_bufs.clear();
    }
}
