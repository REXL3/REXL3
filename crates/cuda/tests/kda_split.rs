//! The KDA head split (lane E, GLM Phase 2 item 5): a rank that runs heads `[h0, h1)` of a 64-head layer as a layer of its own
//! (`Dims { heads: 32 }`, the weights of `LayerWeights::heads_of`, the heads' qkv channels, a 32-head pool) leaves exactly the 64-head
//! layer's outputs and state for those heads: windows of several sequences with partial and full commits (the replay path), chained
//! rounds, and prefill (pieces of 128 rows). Bit for bit, both halves.
//!
//!     cargo test --release -p gb-cuda --test kda_split -- --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]

mod common;

use common::Rng;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::kda::*;
use gb_cuda::num::f32_to_bf16;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::f64_to_f16;

const D: usize = HEAD_DIM;
const HID: usize = GLM_HIDDEN;

fn f16s(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f64_to_f16(x as f64)).collect()
}
fn bf16s(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f32_to_bf16(x)).collect()
}

struct HostW {
    b: Vec<u16>,
    f_a: Vec<u16>,
    g_a: Vec<u16>,
    f_b: Vec<u16>,
    g_b: Vec<u16>,
    conv_w: Vec<u16>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    o_norm: Vec<u16>,
}

impl HostW {
    fn random(rng: &mut Rng, h: usize) -> Self {
        let (vd, c) = (h * D, 3 * h * D);
        Self {
            b: f16s(&rng.vec(h * HID, -0.03, 0.03)),
            f_a: f16s(&rng.vec(RANK * HID, -0.03, 0.03)),
            g_a: f16s(&rng.vec(RANK * HID, -0.03, 0.03)),
            f_b: f16s(&rng.vec(vd * RANK, -0.1, 0.1)),
            g_b: f16s(&rng.vec(vd * RANK, -0.1, 0.1)),
            conv_w: bf16s(&rng.vec(c * 4, -0.6, 0.6)),
            a_log: rng.vec(h, 0.0, 2.5),
            dt_bias: rng.vec(vd, -4.0, 0.5),
            o_norm: bf16s(&rng.vec(D, 0.5, 1.5)),
        }
    }
}

struct DevW {
    b: DevBuf,
    f_a: DevBuf,
    g_a: DevBuf,
    f_b: DevBuf,
    g_b: DevBuf,
    conv_w: DevBuf,
    a_log: DevBuf,
    dt_bias: DevBuf,
    o_norm: DevBuf,
}

impl DevW {
    fn new(w: &HostW) -> Self {
        let u = |v: &Vec<u16>| DevBuf::from_slice(v).unwrap();
        Self {
            b: u(&w.b),
            f_a: u(&w.f_a),
            g_a: u(&w.g_a),
            f_b: u(&w.f_b),
            g_b: u(&w.g_b),
            conv_w: u(&w.conv_w),
            a_log: DevBuf::from_slice(&w.a_log).unwrap(),
            dt_bias: DevBuf::from_slice(&w.dt_bias).unwrap(),
            o_norm: u(&w.o_norm),
        }
    }
    fn view(&self) -> Weights<'_> {
        Weights {
            b: &self.b,
            f_a: &self.f_a,
            g_a: &self.g_a,
            f_b: &self.f_b,
            g_b: &self.g_b,
            conv_w: &self.conv_w,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            o_norm: &self.o_norm,
        }
    }
}

/// The channels `[q | k | v]` of heads `[h0, h1)` of rows of `3 * h * 128` values.
fn qkv_heads<T: Copy>(v: &[T], h: usize, h0: usize, h1: usize) -> Vec<T> {
    let c = h * D;
    let rows = v.len() / (3 * c);
    let mut out = Vec::with_capacity(rows * 3 * (h1 - h0) * D);
    for r in 0..rows {
        for third in 0..3 {
            let base = r * 3 * c + third * c;
            out.extend_from_slice(&v[base + h0 * D..base + h1 * D]);
        }
    }
    out
}

/// Columns of heads `[h0, h1)` of rows of `h * 128` values.
fn head_cols<T: Copy>(v: &[T], h: usize, h0: usize, h1: usize) -> Vec<T> {
    let w = h * D;
    v.chunks(w).flat_map(|row| row[h0 * D..h1 * D].to_vec()).collect()
}

fn same<T: PartialEq + Copy + std::fmt::Debug>(what: &str, got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!("{what}: element {i} differs ({:?} vs {:?})", got[i], want[i]);
    }
}

fn same_f32(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits()) {
        panic!("{what}: element {i} differs ({:e} vs {:e})", got[i], want[i]);
    }
}

/// One layer of `d` and its pool; the window / commit / prefill calls of the tests.
struct Layer<'a> {
    st: &'a Stream,
    d: Dims,
    w: Weights<'a>,
    p: Params,
    pool: Pool,
}

impl<'a> Layer<'a> {
    fn window(
        &mut self,
        x: &[u16],
        qkv: &[f32],
        lens: &[usize],
        slots: &[usize],
        keep: Option<&[i32]>,
        write_final: bool,
    ) -> Vec<u16> {
        let batch = BatchBuf::new(lens, slots).unwrap();
        let rows = batch.view().rows;
        let xd = DevBuf::from_slice(x).unwrap();
        let qd = DevBuf::from_slice(qkv).unwrap();
        let mut rec = Record::alloc(self.d, rows).unwrap();
        let mut ws = Workspace::alloc(self.d, rows).unwrap();
        let mut out = DevBuf::zeroed(rows * self.d.v_dim() * 2).unwrap();
        forward_window(
            self.st,
            self.d,
            &self.w,
            &self.p,
            &xd,
            Dtype::F16,
            &qd,
            Dtype::F32,
            &mut self.pool,
            &batch.view(),
            &mut rec,
            &mut ws,
            &mut out,
            write_final,
        )
        .unwrap();
        let k = keep.map(|k| DevBuf::from_slice(k).unwrap());
        commit(self.st, self.d, &self.w, &self.p, &mut self.pool, &batch.view(), &rec, &mut ws, k.as_ref(), write_final).unwrap();
        self.st.sync().unwrap();
        out.to_vec::<u16>().unwrap()[..rows * self.d.v_dim()].to_vec()
    }

    fn prefill(&mut self, x: &[u16], qkv: &[f32], slot: usize) -> Vec<u16> {
        let rows = x.len() / HID;
        let xd = DevBuf::from_slice(x).unwrap();
        let qd = DevBuf::from_slice(qkv).unwrap();
        let mut rec = Record::alloc(self.d, PREFILL_PIECE_ROWS).unwrap();
        let mut ws = Workspace::alloc(self.d, PREFILL_PIECE_ROWS).unwrap();
        let mut out = DevBuf::zeroed(rows * self.d.v_dim() * 2).unwrap();
        prefill(
            self.st,
            self.d,
            &self.w,
            &self.p,
            &xd,
            Dtype::F16,
            &qd,
            Dtype::F32,
            &mut self.pool,
            slot,
            rows,
            &mut rec,
            &mut ws,
            &mut out,
        )
        .unwrap();
        self.st.sync().unwrap();
        out.to_vec::<u16>().unwrap()[..rows * self.d.v_dim()].to_vec()
    }

    fn state(&self, slot: usize) -> (Vec<f32>, Vec<u16>) {
        (self.pool.read_state(slot).unwrap(), self.pool.read_conv(slot).unwrap())
    }
}

#[test]
fn a_rank_of_32_heads_leaves_the_64_head_layers_bits_for_its_heads() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    let full = Dims::glm();
    let (h, half) = (full.heads, full.heads / 2);
    let mut rng = Rng(0x5D17);
    let hw = HostW::random(&mut rng, h);
    let dw = DevW::new(&hw);
    let n_slots = 4;
    // starting states: random, the same heads in the half pools
    let init: Vec<(Vec<f32>, Vec<u16>)> =
        (0..n_slots).map(|_| (rng.vec(h * D * D, -0.3, 0.3), bf16s(&rng.vec(3 * 3 * h * D, -2.5, 2.5)))).collect();
    // the windows: (lens, slots, keep) of chained rounds, multi-sequence, partial and full accepts, with and without the final state
    type Round = (Vec<usize>, Vec<usize>, Option<Vec<i32>>, bool);
    let rounds: Vec<Round> = vec![
        (vec![1], vec![0], None, true),
        (vec![3, 1, 2], vec![1, 0, 2], Some(vec![2, 1, 0]), true),
        (vec![4, 4], vec![0, 3], Some(vec![4, 1]), false),
        (vec![17, 64, 5], vec![2, 1, 3], Some(vec![9, 64, 5]), true),
        (vec![128], vec![0], Some(vec![77]), true),
        (vec![1, 1, 1, 1], vec![0, 1, 2, 3], None, false),
    ];
    let mut total_rows = 0;
    for rank in 0..2 {
        let (h0, h1) = (rank * half, (rank + 1) * half);
        let dr = Dims { heads: half, hidden: full.hidden };
        let lw = LayerWeights::heads_of(&dw.view(), full, h0, h1, &st).unwrap();
        // the slice is the host's slice, byte for byte
        // conv_w is [channels][4 taps]: one "row" of 3 h 128 channels of 4 taps each
        let cw_part: Vec<u16> = qkv_heads(&hw.conv_w.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect::<Vec<_>>(), h, h0, h1)
            .into_iter()
            .flatten()
            .collect();
        let v = lw.view();
        same("b", &v.b.to_vec::<u16>().unwrap(), &hw.b[h0 * HID..h1 * HID]);
        same("f_b", &v.f_b.to_vec::<u16>().unwrap(), &hw.f_b[h0 * D * RANK..h1 * D * RANK]);
        same("conv_w", &v.conv_w.to_vec::<u16>().unwrap(), &cw_part);
        same_f32("A_log", &v.a_log.to_vec::<f32>().unwrap(), &hw.a_log[h0..h1]);
        same_f32("dt_bias", &v.dt_bias.to_vec::<f32>().unwrap(), &hw.dt_bias[h0 * D..h1 * D]);

        let mut a = Layer { st: &st, d: full, w: dw.view(), p: Params::glm(), pool: Pool::new(full, n_slots).unwrap() };
        let mut b = Layer { st: &st, d: dr, w: lw.view(), p: Params::glm(), pool: Pool::new(dr, n_slots).unwrap() };
        for (slot, (s, c)) in init.iter().enumerate() {
            a.pool.write_slot(slot, s, c).unwrap();
            b.pool.write_slot(slot, &s[h0 * D * D..h1 * D * D], &qkv_heads(c, h, h0, h1)).unwrap();
        }
        for (ri, (lens, slots, keep, wf)) in rounds.iter().enumerate() {
            let rows: usize = lens.iter().sum();
            let x = f16s(&rng.vec(rows * HID, -2.0, 2.0));
            let qkv = rng.vec(rows * 3 * h * D, -2.5, 2.5);
            let oa = a.window(&x, &qkv, lens, slots, keep.as_deref(), *wf);
            let ob = b.window(&x, &qkv_heads(&qkv, h, h0, h1), lens, slots, keep.as_deref(), *wf);
            same(&format!("rank {rank} round {ri} out"), &ob, &head_cols(&oa, h, h0, h1));
            for slot in 0..n_slots {
                let ((sa, ca), (sb, cb)) = (a.state(slot), b.state(slot));
                same_f32(&format!("rank {rank} round {ri} slot {slot} state"), &sb, &sa[h0 * D * D..h1 * D * D]);
                same(&format!("rank {rank} round {ri} slot {slot} conv"), &cb, &qkv_heads(&ca, h, h0, h1));
            }
            total_rows += rows;
        }
        // prefill on slot 3 (pieces of 128 rows), after the windows
        for rows in [1usize, 127, 128, 129, 300] {
            let x = f16s(&rng.vec(rows * HID, -2.0, 2.0));
            let qkv = rng.vec(rows * 3 * h * D, -2.5, 2.5);
            let oa = a.prefill(&x, &qkv, 3);
            let ob = b.prefill(&x, &qkv_heads(&qkv, h, h0, h1), 3);
            same(&format!("rank {rank} prefill {rows} out"), &ob, &head_cols(&oa, h, h0, h1));
            let ((sa, ca), (sb, cb)) = (a.state(3), b.state(3));
            same_f32(&format!("rank {rank} prefill {rows} state"), &sb, &sa[h0 * D * D..h1 * D * D]);
            same(&format!("rank {rank} prefill {rows} conv"), &cb, &qkv_heads(&ca, h, h0, h1));
            total_rows += rows;
        }
        eprintln!("rank {rank} (heads {h0}..{h1}): every window, commit and prefill output and state bit-identical to the 64-head layer's");
    }
    eprintln!("{total_rows} rows: the 32-head ranks equal the 64-head layer bit for bit (outputs, state, conv history)");
}
