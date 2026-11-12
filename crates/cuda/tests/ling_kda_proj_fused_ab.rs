//! A/B bit-equality of the fused KDA projections (`ling_kda_proj_fused`: f and g in one launch sharing the input
//! rotation; the qkv+f+g triple in one launch; the MLA q_a+kv_a and dense gate+up pairs at bf16 out) against the
//! separate `gb_exl3_linear_k2` calls on identical random inputs, across the decode row counts that exercise NT = 1..4
//! and the H = 2 sweeps, including heterogeneous n0 != n1 pairs and the split-K narrow-layer plan.
#![cfg(not(gb_no_cuda))]
mod common;

// The dense gate/up pair shape (K2 = 6, bf16 out, SK = 1 / WK = 8 at this K).
const K_DENSE: usize = 2048;
const N_DENSE: usize = 10240;
// The MLA q_a/kv_a pair shape (K2 = 10, heterogeneous n, bf16 out, the narrow layers split K: SK = 4 here).
const K_MLA: usize = 2048;
const N0_MLA: usize = 1536;
const N1_MLA: usize = 640;
// The KDA triple shape (projection 0 at K2 = 7 with f32 out, projections 1/2 at K2 = 6 with f16 out, all SK = 1).
const K_TRIPLE: usize = 2048;
const N0_TRIPLE: usize = 24576;
const N12_TRIPLE: usize = 8192;
const CB: i32 = 2;

use common::Rng;
use gb_cuda::attention::{BlockTable, RowIndex};
use gb_cuda::exl3_linear::{linear_workspace_bytes, Dtype, Exl3Weights};
use gb_cuda::glue;
use gb_cuda::kda::{self, Dims as KdaDims, Params, Pool, Record, Weights, Workspace};
use gb_cuda::ling_fmoe;
use gb_cuda::ling_kda_proj_fused::{
    bf16_linear_f32, fused_decode_pair, fused_decode_pair_ext, fused_decode_triple, linear_add_rmsnorm, linear_prerot,
    mla_qkv_qb_fused, norm_rot_in_pair, norm_rot_in_single, norm_rot_in_triple, norm_rot_in_triple_ext, pair_workspace_bytes,
    rmsnorm_linear_bf16, silu_mul_linear_add_rmsnorm, silu_mul_linear_add_rmsnorm_ext, topk_f16, topk_workspace_bytes,
    workspace_bytes, Prerot, Proj,
};
use gb_cuda::ling_mla_attn as tc;
use gb_cuda::mla_gated::{self, MlaGatedCache};
use gb_cuda::rt::{DevBuf, Stream};

fn gpu() -> Option<Stream> {
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    gb_cuda::rt::init_context().ok()?;
    Stream::new().ok()
}

fn bf(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn f16(v: f64) -> u16 {
    gb_exl3::f16::f64_to_f16(v)
}

fn dl<T: Copy + Default>(v: &DevBuf, st: &Stream) -> Vec<T> {
    st.sync().unwrap();
    v.to_vec().unwrap()
}

fn check_eq<T: PartialEq + std::fmt::Debug + Copy>(name: &str, a: &[T], b: &[T]) -> bool {
    assert_eq!(a.len(), b.len(), "{name}: len");
    for i in 0..a.len() {
        if a[i] != b[i] {
            eprintln!("{name}: first mismatch at {i}: A={:?} B={:?}", a[i], b[i]);
            return false;
        }
    }
    true
}

fn diff_count(a: &[u16], b: &[u16]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn diff_count_f32(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn nw_for(k: usize, n: usize, k2: usize) -> usize {
    (k / 16) * (n / 16) * (4 * k2)
}

// One side of the fused MLA pair/q_b in the A/B tests: a `Proj` over random device tensors.
fn proj_mla<'a>(y: &'a mut DevBuf, t: &'a DevBuf, s: &'a DevBuf, v: &'a DevBuf, k: usize, n: usize, k2: usize) -> Proj<'a> {
    Proj { trellis: t, suh: s, svh: v, y, k, n, k2, codebook: CB }
}

// A minimal Proj for the separate `norm_rot_in_{triple,single}` reference (only suh is read by the rotation).
fn proj_norm_rot<'a>(y: &'a mut DevBuf, t: &'a DevBuf, s: &'a DevBuf, v: &'a DevBuf, k: usize, n: usize, k2: usize) -> Proj<'a> {
    Proj { trellis: t, suh: s, svh: v, y, k, n, k2, codebook: CB }
}

fn fused_pair_matches_two_launches_for(k2: usize, cb: i32) {
    let Some(st) = gpu() else { return };
    let mut r = Rng(0x91a9_5b84 ^ k2 as u64);
    let k = 512usize;
    let n = 1024usize;
    let nw = nw_for(k, n, k2);

    for &m in &[1usize, 8, 17, 33, 65] {
        for trial in 0..3 {
            let x: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let t0: Vec<u32> = (0..nw).map(|_| r.next_u64() as u32).collect();
            let s0: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v0: Vec<u16> = (0..n).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let t1: Vec<u32> = (0..nw).map(|_| r.next_u64() as u32).collect();
            let s1: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v1: Vec<u16> = (0..n).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();

            let x_d = DevBuf::from_slice(&x).unwrap();
            let t0_d = DevBuf::from_slice(&t0).unwrap();
            let s0_d = DevBuf::from_slice(&s0).unwrap();
            let v0_d = DevBuf::from_slice(&v0).unwrap();
            let t1_d = DevBuf::from_slice(&t1).unwrap();
            let s1_d = DevBuf::from_slice(&s1).unwrap();
            let v1_d = DevBuf::from_slice(&v1).unwrap();

            // reference: two separate k2_forward launches
            let mut yf_ref = vec![0u16; m * n];
            let mut yg_ref = vec![0u16; m * n];
            let ws_ref_bytes = linear_workspace_bytes(m, k, n, k2 / 2);
            let mut ws_ref = DevBuf::zeroed(ws_ref_bytes).unwrap();
            let mut yf_ref_d = DevBuf::from_slice(&yf_ref).unwrap();
            let mut yg_ref_d = DevBuf::from_slice(&yg_ref).unwrap();
            Exl3Weights::k2_forward(
                &x_d,
                &mut yf_ref_d,
                m,
                Dtype::BF16,
                Dtype::F16,
                k,
                n,
                k2,
                gb_exl3::Codebook::Mul1,
                &t0_d,
                &s0_d,
                &v0_d,
                None,
                &mut ws_ref,
                &st,
            )
            .unwrap();
            Exl3Weights::k2_forward(
                &x_d,
                &mut yg_ref_d,
                m,
                Dtype::BF16,
                Dtype::F16,
                k,
                n,
                k2,
                gb_exl3::Codebook::Mul1,
                &t1_d,
                &s1_d,
                &v1_d,
                None,
                &mut ws_ref,
                &st,
            )
            .unwrap();

            // fused: one rotation + one grid
            let mut yf = vec![0u16; m * n];
            let mut yg = vec![0u16; m * n];
            let mut yf_d = DevBuf::from_slice(&yf).unwrap();
            let mut yg_d = DevBuf::from_slice(&yg).unwrap();
            let ws_f_bytes = pair_workspace_bytes(m, k, n, n);
            let mut ws_f = DevBuf::zeroed(ws_f_bytes).unwrap();
            fused_decode_pair(
                &x_d,
                Proj { trellis: &t0_d, suh: &s0_d, svh: &v0_d, y: &mut yf_d, k, n, k2, codebook: cb },
                Proj { trellis: &t1_d, suh: &s1_d, svh: &v1_d, y: &mut yg_d, k, n, k2, codebook: cb },
                m,
                Dtype::BF16,
                Dtype::F16,
                &mut ws_f,
                None,
                None,
                &st,
            )
            .unwrap();

            st.sync().unwrap();
            yf = yf_d.to_vec().unwrap();
            yg = yg_d.to_vec().unwrap();
            yf_ref = yf_ref_d.to_vec().unwrap();
            yg_ref = yg_ref_d.to_vec().unwrap();

            assert!(
                check_eq(&format!("f m={m} trial={trial}"), &yf, &yf_ref),
                "fused f differs from the separate launch at m={m} trial={trial}"
            );
            assert!(
                check_eq(&format!("g m={m} trial={trial}"), &yg, &yg_ref),
                "fused g differs from the separate launch at m={m} trial={trial}"
            );
        }
    }
}

#[test]
#[ignore]
fn fused_pair_k2_7_matches_two_launches() {
    fused_pair_matches_two_launches_for(7, 2);
}

#[test]
#[ignore]
fn fused_pair_k2_6_matches_two_launches() {
    fused_pair_matches_two_launches_for(6, 2);
}

// The dense gate/up pair and the MLA q_a/kv_a pair at bf16 out (K2 = 6, n0 = n1 = 10240 at SK = 1; K2 = 10,
// n0 = 1536, n1 = 640 at SK = 4, heterogeneous n).
fn fused_pair_bf16_matches_two_launches_for(k2: usize, n0: usize, n1: usize, k: usize) {
    let Some(st) = gpu() else { return };
    let mut r = Rng(0x6e3a_9b84 ^ k2 as u64 ^ (n0 as u64) << 20);
    let nw0 = nw_for(k, n0, k2);
    let nw1 = nw_for(k, n1, k2);

    for &m in &[1usize, 8, 17, 65] {
        for trial in 0..2 {
            let x: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let t0: Vec<u32> = (0..nw0).map(|_| r.next_u64() as u32).collect();
            let s0: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v0: Vec<u16> = (0..n0).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let t1: Vec<u32> = (0..nw1).map(|_| r.next_u64() as u32).collect();
            let s1: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v1: Vec<u16> = (0..n1).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();

            let x_d = DevBuf::from_slice(&x).unwrap();
            let t0_d = DevBuf::from_slice(&t0).unwrap();
            let s0_d = DevBuf::from_slice(&s0).unwrap();
            let v0_d = DevBuf::from_slice(&v0).unwrap();
            let t1_d = DevBuf::from_slice(&t1).unwrap();
            let s1_d = DevBuf::from_slice(&s1).unwrap();
            let v1_d = DevBuf::from_slice(&v1).unwrap();

            let mut yf_ref = vec![0u16; m * n0];
            let mut yg_ref = vec![0u16; m * n1];
            let ws0_ref = linear_workspace_bytes(m, k, n0, k2 / 2);
            let ws1_ref = linear_workspace_bytes(m, k, n1, k2 / 2);
            let mut ws_ref = DevBuf::zeroed(ws0_ref.max(ws1_ref)).unwrap();
            let mut yf_ref_d = DevBuf::from_slice(&yf_ref).unwrap();
            let mut yg_ref_d = DevBuf::from_slice(&yg_ref).unwrap();
            Exl3Weights::k2_forward(
                &x_d,
                &mut yf_ref_d,
                m,
                Dtype::BF16,
                Dtype::BF16,
                k,
                n0,
                k2,
                gb_exl3::Codebook::Mul1,
                &t0_d,
                &s0_d,
                &v0_d,
                None,
                &mut ws_ref,
                &st,
            )
            .unwrap();
            Exl3Weights::k2_forward(
                &x_d,
                &mut yg_ref_d,
                m,
                Dtype::BF16,
                Dtype::BF16,
                k,
                n1,
                k2,
                gb_exl3::Codebook::Mul1,
                &t1_d,
                &s1_d,
                &v1_d,
                None,
                &mut ws_ref,
                &st,
            )
            .unwrap();

            let mut yf = vec![0u16; m * n0];
            let mut yg = vec![0u16; m * n1];
            let mut yf_d = DevBuf::from_slice(&yf).unwrap();
            let mut yg_d = DevBuf::from_slice(&yg).unwrap();
            let ws_f_bytes = pair_workspace_bytes(m, k, n0, n1);
            let mut ws_f = DevBuf::zeroed(ws_f_bytes).unwrap();
            fused_decode_pair(
                &x_d,
                Proj { trellis: &t0_d, suh: &s0_d, svh: &v0_d, y: &mut yf_d, k, n: n0, k2, codebook: CB },
                Proj { trellis: &t1_d, suh: &s1_d, svh: &v1_d, y: &mut yg_d, k, n: n1, k2, codebook: CB },
                m,
                Dtype::BF16,
                Dtype::BF16,
                &mut ws_f,
                None,
                None,
                &st,
            )
            .unwrap();

            st.sync().unwrap();
            yf = yf_d.to_vec().unwrap();
            yg = yg_d.to_vec().unwrap();
            yf_ref = yf_ref_d.to_vec().unwrap();
            yg_ref = yg_ref_d.to_vec().unwrap();

            assert!(
                check_eq(&format!("bf16 f k2={k2} n0={n0} n1={n1} m={m} trial={trial}"), &yf, &yf_ref),
                "fused bf16 f differs from the separate launch at k2={k2} n0={n0} n1={n1} m={m} trial={trial}"
            );
            assert!(
                check_eq(&format!("bf16 g k2={k2} n0={n0} n1={n1} m={m} trial={trial}"), &yg, &yg_ref),
                "fused bf16 g differs from the separate launch at k2={k2} n0={n0} n1={n1} m={m} trial={trial}"
            );
        }
    }
}

#[test]
#[ignore]
fn fused_pair_dense_bf16_matches_two_launches() {
    fused_pair_bf16_matches_two_launches_for(6, N_DENSE, N_DENSE, K_DENSE);
}

#[test]
#[ignore]
fn fused_pair_mla_bf16_hetero_matches_two_launches() {
    fused_pair_bf16_matches_two_launches_for(10, N0_MLA, N1_MLA, K_MLA);
}

// The KDA qkv+f+g triple: projection 0 at K2 = 7 with f32 out, projections 1/2 at K2 = 6 with f16 out.
#[test]
#[ignore]
fn fused_triple_matches_three_launches() {
    let Some(st) = gpu() else { return };
    let mut r = Rng(0x11e5_5b84);
    let (k, k2_0, k2_12) = (K_TRIPLE, 7usize, 6usize);
    let nw0 = nw_for(k, N0_TRIPLE, k2_0);
    let nw1 = nw_for(k, N12_TRIPLE, k2_12);
    assert_eq!(workspace_bytes(1, k), 3 * 16 * k * 2);

    for &m in &[1usize, 8, 17, 65] {
        for trial in 0..2 {
            let x: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let t0: Vec<u32> = (0..nw0).map(|_| r.next_u64() as u32).collect();
            let s0: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v0: Vec<u16> = (0..N0_TRIPLE).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let t1: Vec<u32> = (0..nw1).map(|_| r.next_u64() as u32).collect();
            let s1: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v1: Vec<u16> = (0..N12_TRIPLE).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let t2: Vec<u32> = (0..nw1).map(|_| r.next_u64() as u32).collect();
            let s2: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v2: Vec<u16> = (0..N12_TRIPLE).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();

            let x_d = DevBuf::from_slice(&x).unwrap();
            let t0_d = DevBuf::from_slice(&t0).unwrap();
            let s0_d = DevBuf::from_slice(&s0).unwrap();
            let v0_d = DevBuf::from_slice(&v0).unwrap();
            let t1_d = DevBuf::from_slice(&t1).unwrap();
            let s1_d = DevBuf::from_slice(&s1).unwrap();
            let v1_d = DevBuf::from_slice(&v1).unwrap();
            let t2_d = DevBuf::from_slice(&t2).unwrap();
            let s2_d = DevBuf::from_slice(&s2).unwrap();
            let v2_d = DevBuf::from_slice(&v2).unwrap();

            let mut y0_ref = vec![0u32; m * N0_TRIPLE];
            let mut y1_ref = vec![0u16; m * N12_TRIPLE];
            let mut y2_ref = vec![0u16; m * N12_TRIPLE];
            let ws_ref_bytes =
                linear_workspace_bytes(m, k, N0_TRIPLE, k2_0 / 2).max(linear_workspace_bytes(m, k, N12_TRIPLE, k2_12 / 2));
            let mut ws_ref = DevBuf::zeroed(ws_ref_bytes).unwrap();
            let mut y0_ref_d = DevBuf::from_slice(&y0_ref).unwrap();
            let mut y1_ref_d = DevBuf::from_slice(&y1_ref).unwrap();
            let mut y2_ref_d = DevBuf::from_slice(&y2_ref).unwrap();
            Exl3Weights::k2_forward(
                &x_d,
                &mut y0_ref_d,
                m,
                Dtype::BF16,
                Dtype::F32,
                k,
                N0_TRIPLE,
                k2_0,
                gb_exl3::Codebook::Mul1,
                &t0_d,
                &s0_d,
                &v0_d,
                None,
                &mut ws_ref,
                &st,
            )
            .unwrap();
            Exl3Weights::k2_forward(
                &x_d,
                &mut y1_ref_d,
                m,
                Dtype::BF16,
                Dtype::F16,
                k,
                N12_TRIPLE,
                k2_12,
                gb_exl3::Codebook::Mul1,
                &t1_d,
                &s1_d,
                &v1_d,
                None,
                &mut ws_ref,
                &st,
            )
            .unwrap();
            Exl3Weights::k2_forward(
                &x_d,
                &mut y2_ref_d,
                m,
                Dtype::BF16,
                Dtype::F16,
                k,
                N12_TRIPLE,
                k2_12,
                gb_exl3::Codebook::Mul1,
                &t2_d,
                &s2_d,
                &v2_d,
                None,
                &mut ws_ref,
                &st,
            )
            .unwrap();

            let mut y0 = vec![0u32; m * N0_TRIPLE];
            let mut y1 = vec![0u16; m * N12_TRIPLE];
            let mut y2 = vec![0u16; m * N12_TRIPLE];
            let mut y0_d = DevBuf::from_slice(&y0).unwrap();
            let mut y1_d = DevBuf::from_slice(&y1).unwrap();
            let mut y2_d = DevBuf::from_slice(&y2).unwrap();
            let ws_f_bytes = workspace_bytes(m, k);
            let mut ws_f = DevBuf::zeroed(ws_f_bytes).unwrap();
            fused_decode_triple(
                &x_d,
                Proj { trellis: &t0_d, suh: &s0_d, svh: &v0_d, y: &mut y0_d, k, n: N0_TRIPLE, k2: k2_0, codebook: CB },
                Proj { trellis: &t1_d, suh: &s1_d, svh: &v1_d, y: &mut y1_d, k, n: N12_TRIPLE, k2: k2_12, codebook: CB },
                Proj { trellis: &t2_d, suh: &s2_d, svh: &v2_d, y: &mut y2_d, k, n: N12_TRIPLE, k2: k2_12, codebook: CB },
                m,
                Dtype::BF16,
                Dtype::F32,
                Dtype::F16,
                &mut ws_f,
                None,
                None,
                None,
                &st,
            )
            .unwrap();

            st.sync().unwrap();
            y0 = y0_d.to_vec().unwrap();
            y1 = y1_d.to_vec().unwrap();
            y2 = y2_d.to_vec().unwrap();
            y0_ref = y0_ref_d.to_vec().unwrap();
            y1_ref = y1_ref_d.to_vec().unwrap();
            y2_ref = y2_ref_d.to_vec().unwrap();

            assert!(
                check_eq(&format!("triple qkv m={m} trial={trial}"), &y0, &y0_ref),
                "fused triple qkv differs from the separate launch at m={m} trial={trial}"
            );
            assert!(
                check_eq(&format!("triple f m={m} trial={trial}"), &y1, &y1_ref),
                "fused triple f differs from the separate launch at m={m} trial={trial}"
            );
            assert!(
                check_eq(&format!("triple g m={m} trial={trial}"), &y2, &y2_ref),
                "fused triple g differs from the separate launch at m={m} trial={trial}"
            );
        }
    }
}

#[test]
#[ignore]
fn debug_shape_sweep() {
    let Some(st) = gpu() else { return };
    for &(n0, n1) in &[(128usize, 128usize), (256, 256), (512, 512), (1024, 1024), (1536, 1536)] {
        let mut r = Rng(0xabc ^ n0 as u64 ^ (n1 as u64) << 12);
        let k = 512usize;
        let k2 = 7usize;
        let nw = nw_for(k, n0, k2);
        let m = 1usize;
        let x: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
        let t0: Vec<u32> = (0..nw).map(|_| r.next_u64() as u32).collect();
        let s0: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
        let v0: Vec<u16> = (0..n0).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
        let t1 = t0.clone();
        let s1 = s0.clone();
        let v1 = v0.clone();
        let v0a = v0[..n0].to_vec();
        let v1a = v1[..n1].to_vec();

        let x_d = DevBuf::from_slice(&x).unwrap();
        let t0_d = DevBuf::from_slice(&t0).unwrap();
        let s0_d = DevBuf::from_slice(&s0).unwrap();
        let v0_d = DevBuf::from_slice(&v0a).unwrap();
        let t1_d = DevBuf::from_slice(&t1).unwrap();
        let s1_d = DevBuf::from_slice(&s1).unwrap();
        let v1_d = DevBuf::from_slice(&v1a).unwrap();

        // reference
        let mut yf_ref = vec![0u16; m * n0];
        let mut yg_ref = vec![0u16; m * n1];
        let ws_ref_bytes = linear_workspace_bytes(m, k, 1024, k2 / 2);
        let mut ws_ref = DevBuf::zeroed(ws_ref_bytes).unwrap();
        let mut yf_ref_d = DevBuf::from_slice(&yf_ref).unwrap();
        let mut yg_ref_d = DevBuf::from_slice(&yg_ref).unwrap();
        Exl3Weights::k2_forward(
            &x_d,
            &mut yf_ref_d,
            m,
            Dtype::BF16,
            Dtype::F16,
            k,
            n0,
            k2,
            gb_exl3::Codebook::Mul1,
            &t0_d,
            &s0_d,
            &v0_d,
            None,
            &mut ws_ref,
            &st,
        )
        .unwrap();
        Exl3Weights::k2_forward(
            &x_d,
            &mut yg_ref_d,
            m,
            Dtype::BF16,
            Dtype::F16,
            k,
            n1,
            k2,
            gb_exl3::Codebook::Mul1,
            &t1_d,
            &s1_d,
            &v1_d,
            None,
            &mut ws_ref,
            &st,
        )
        .unwrap();

        // fused
        let mut yf = vec![0u16; m * n0];
        let mut yg = vec![0u16; m * n1];
        let mut yf_d = DevBuf::from_slice(&yf).unwrap();
        let mut yg_d = DevBuf::from_slice(&yg).unwrap();
        let ws_f_bytes = pair_workspace_bytes(m, k, n0, n1);
        let mut ws_f = DevBuf::zeroed(ws_f_bytes).unwrap();
        fused_decode_pair(
            &x_d,
            Proj { trellis: &t0_d, suh: &s0_d, svh: &v0_d, y: &mut yf_d, k, n: n0, k2, codebook: CB },
            Proj { trellis: &t1_d, suh: &s1_d, svh: &v1_d, y: &mut yg_d, k, n: n1, k2, codebook: CB },
            m,
            Dtype::BF16,
            Dtype::F16,
            &mut ws_f,
            None,
            None,
            &st,
        )
        .unwrap();
        st.sync().unwrap();
        yf = yf_d.to_vec().unwrap();
        yg = yg_d.to_vec().unwrap();
        yf_ref = yf_ref_d.to_vec().unwrap();
        yg_ref = yg_ref_d.to_vec().unwrap();

        eprintln!("DBG shape n0={n0} n1={n1}");
        eprintln!("DBG sym: fused f-vs-g diffs={}", diff_count(&yf, &yg));
        eprintln!("DBG sym: ref f-vs-g diffs={}", diff_count(&yf_ref, &yg_ref));
        eprintln!("DBG sym: fused f-vs-ref f diffs={}", diff_count(&yf, &yf_ref));
        eprintln!("DBG sym: fused g-vs-ref g diffs={}", diff_count(&yg, &yg_ref));
        for b in 0..(n0 / 128) {
            let lo = b * 128;
            let hi = lo + 128;
            eprintln!("DBG f block {b}: fused-ref diffs={}", diff_count(&yf[lo..hi], &yf_ref[lo..hi]));
        }
        let dump = |name: &str, v: &[u16]| {
            let mut t = String::new();
            for (i, x) in v.iter().enumerate() {
                if i % 16 == 0 && i > 0 {
                    t.push('\n');
                }
                t.push_str(&format!("{:04x} ", x));
            }
            std::fs::write(format!("/tmp/{name}_{n0}_{n1}.txt"), t).unwrap();
        };
        dump("yf", &yf);
        dump("yf_ref", &yf_ref);
        dump("yg", &yg);
        dump("yg_ref", &yg_ref);
        if n0 >= 256 {
            eprintln!("DBG cross: fused f1-vs-fused g1 eq={}", diff_count(&yf[128..256], &yg[128..256]));
            eprintln!("DBG cross: fused f1-vs-ref g1 eq={}", diff_count(&yf[128..256], &yg_ref[128..256]));
        }
        assert!(yf == yf_ref, "fused f differs from reference");
        eprintln!("DBG (forced-single) f-vs-ref diffs={}", diff_count(&yf, &yf_ref));
        for b in 0..(n0 / 128) {
            let lo = b * 128;
            let hi = lo + 128;
            eprintln!("DBG f block {b}: fused-ref diffs={}", diff_count(&yf[lo..hi], &yf_ref[lo..hi]));
        }
    }
}

// ---------------------------------------------------------------------------
// gb_ling_linear_add_rmsnorm vs gb_exl3_linear_k2 + glue::add_rmsnorm: the fused
// epilogue must produce the same residual (hidden_out) and norm (y) bytes as the
// separate launches, across the decode/verify row counts (m = 1..8), the KDA
// o_proj shape (K2 = 10, k = 8192, f16 x, n = dim = 4096) and the dense FFN
// down_proj shape (K2 = 7, k = 10240, bf16 x).
fn fused_linear_add_rmsnorm_matches_separate_for(k2: usize, k: usize, x_dtype: Dtype) {
    let Some(st) = gpu() else { return };
    let lin_cnt = DevBuf::zeroed(1024).unwrap();

    let n = 4096usize;
    let dim = 4096usize;
    let eps = 1e-6f32;
    let nw = nw_for(k, n, k2);
    let mut r = Rng(0x5b84_2a7e ^ k2 as u64 ^ (k as u64) << 16);
    for &m in &[1usize, 2, 4, 8] {
        for trial in 0..3 {
            let xs: Vec<u16> = match x_dtype {
                Dtype::F16 => (0..m * k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect(),
                _ => (0..m * k).map(|_| bf(r.uniform(-0.5, 0.5))).collect(),
            };
            let t: Vec<u32> = (0..nw).map(|_| r.next_u64() as u32).collect();
            let s: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v: Vec<u16> = (0..n).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let hidden: Vec<u16> = (0..m * dim).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let w: Vec<f32> = (0..dim).map(|_| r.uniform(0.5, 1.5)).collect();

            let x_d = DevBuf::from_slice(&xs).unwrap();
            let t_d = DevBuf::from_slice(&t).unwrap();
            let s_d = DevBuf::from_slice(&s).unwrap();
            let v_d = DevBuf::from_slice(&v).unwrap();
            let hidden_d = DevBuf::from_slice(&hidden).unwrap();
            let w_d = DevBuf::from_slice(&w).unwrap();

            let ws_bytes = linear_workspace_bytes(m, k, n, k2 / 2);
            let mut ws_a = DevBuf::zeroed(ws_bytes).unwrap();
            let mut ws_b = DevBuf::zeroed(ws_bytes).unwrap();

            // Route A: the separate launches (k2_forward, then glue::add_rmsnorm).
            let mut res = vec![0u16; m * n];
            let mut res_d = DevBuf::from_slice(&res).unwrap();
            Exl3Weights::k2_forward(
                &x_d,
                &mut res_d,
                m,
                x_dtype,
                Dtype::BF16,
                k,
                n,
                k2,
                gb_exl3::Codebook::Mul1,
                &t_d,
                &s_d,
                &v_d,
                None,
                &mut ws_a,
                &st,
            )
            .unwrap();
            let mut exp_hidden = hidden.clone();
            let mut exp_norm = vec![0u16; m * dim];
            let eh_d = DevBuf::from_slice(&exp_hidden).unwrap();
            let en_d = DevBuf::from_slice(&exp_norm).unwrap();
            unsafe {
                gb_cuda::glue::add_rmsnorm(
                    hidden_d.ptr().cast_const(),
                    res_d.ptr().cast_const(),
                    w_d.ptr().cast_const(),
                    eh_d.ptr(),
                    en_d.ptr(),
                    m,
                    dim,
                    eps,
                    &st,
                )
                .unwrap();
            }

            // Route B: the fused launch.
            let got_hidden = hidden.clone();
            let got_norm = vec![0u16; m * dim];
            let mut gh_d = DevBuf::from_slice(&got_hidden).unwrap();
            let mut gn_d = DevBuf::from_slice(&got_norm).unwrap();
            gb_cuda::ling_kda_proj_fused::linear_add_rmsnorm(
                &x_d,
                x_dtype,
                &t_d,
                &s_d,
                &v_d,
                &hidden_d,
                &w_d,
                &mut gh_d,
                &mut gn_d,
                m,
                k,
                n,
                k2,
                CB,
                dim,
                eps,
                &mut ws_b,
                false,
                &Prerot::default(),
                Some(&lin_cnt),
                &st,
            )
            .unwrap();

            st.sync().unwrap();
            let eh: Vec<u16> = eh_d.to_vec().unwrap();
            let en: Vec<u16> = en_d.to_vec().unwrap();
            let gh: Vec<u16> = gh_d.to_vec().unwrap();
            let gn: Vec<u16> = gn_d.to_vec().unwrap();
            assert!(
                check_eq(&format!("hidden k2={k2} k={k} x={:?} m={m} trial={trial}", x_dtype), &gh, &eh),
                "fused linear_add_rmsnorm hidden differs from the separate launches at k2={k2} k={k} m={m} trial={trial}"
            );
            assert!(
                check_eq(&format!("norm k2={k2} k={k} x={:?} m={m} trial={trial}", x_dtype), &gn, &en),
                "fused linear_add_rmsnorm norm differs from the separate launches at k2={k2} k={k} m={m} trial={trial}"
            );
        }
    }
}

#[test]
#[ignore]
fn fused_linear_add_rmsnorm_k2_10_matches_separate() {
    // the KDA o_proj / MLA dense shape (5-bit, f16 input)
    fused_linear_add_rmsnorm_matches_separate_for(10, 8192, Dtype::F16);
}

#[test]
#[ignore]
fn fused_linear_add_rmsnorm_k2_7_matches_separate() {
    // the dense FFN down_proj shape (3.5-bit, bf16 input)
    fused_linear_add_rmsnorm_matches_separate_for(7, 10240, Dtype::BF16);
}

// gb_ling_bf16_linear_f32 vs mla_gated::bf16_linear_f32: the single-wave multi-row
// router GEMV must produce the same fp32 logit bits as the scalar kernel at every
// row count (1..8, plus 13 to exercise the chunk-of-8 path).
#[test]
#[ignore]
fn fused_bf16_linear_f32_matches_mla_gated() {
    let Some(st) = gpu() else { return };
    let (k, n) = (4096usize, 512usize);
    let mut r = Rng(0x6e3a_9b84 ^ 0x5b84);
    for &rows in &[1usize, 2, 4, 8, 13] {
        for trial in 0..3 {
            let x: Vec<u16> = (0..rows * k).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let w: Vec<u16> = (0..n * k).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let x_d = DevBuf::from_slice(&x).unwrap();
            let w_d = DevBuf::from_slice(&w).unwrap();
            let mut yr = vec![0f32; rows * n];
            let mut yr_d = DevBuf::from_slice(&yr).unwrap();
            gb_cuda::mla_gated::bf16_linear_f32(&x_d, &w_d, &mut yr_d, rows, k, n, &st).unwrap();
            let mut yf = vec![0f32; rows * n];
            let mut yf_d = DevBuf::from_slice(&yf).unwrap();
            gb_cuda::ling_kda_proj_fused::bf16_linear_f32(&x_d, &w_d, &mut yf_d, rows, k, n, &st).unwrap();
            st.sync().unwrap();
            yr = yr_d.to_vec().unwrap();
            yf = yf_d.to_vec().unwrap();
            let mut bad = None;
            for i in 0..yr.len() {
                if yr[i].to_bits() != yf[i].to_bits() {
                    bad = Some(i);
                    break;
                }
            }
            assert!(
                bad.is_none(),
                "fused bf16_linear_f32 differs from mla_gated at rows={rows} trial={trial}: first mismatch at {} (ref {} vs fused {})",
                bad.unwrap_or(0),
                yr[bad.unwrap_or(0)],
                yf[bad.unwrap_or(0)]
            );
        }
    }
}

// gb_ling_fmoe_forward vs fmoe::forward: the Ling-owned decode MoE (register-cached group router + overlapped-smem
// gate/up K reduction) must be bit-identical to the stock path at every decode row count on the real Ling shape
// (H=4096, KC=8 — where the stock smem layout does not fit 100 KiB and falls back to global partials).
#[test]
#[ignore]
fn ling_fmoe_forward_matches_fmoe_forward() {
    let Some(st) = gpu() else { return };
    let (h, i, ne, top_k, kc, max_tokens) = (4096usize, 1024usize, 512usize, 8usize, 8usize, 64usize);
    const K2: usize = 7;
    let words = (h / 16) * (i / 16) * 4 * K2;
    let mut r = Rng(0x51b8_5b84 ^ 0x5b84 ^ K2 as u64);
    // one shared trellis and sign-vector buffer per matrix; every expert entry points at it (reads stay in bounds)
    let trel: Vec<u32> = (0..words).map(|_| r.next_u64() as u32).collect();
    let suh: Vec<u16> = (0..h.max(i)).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let svh: Vec<u16> = (0..h.max(i)).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let g_t_d = DevBuf::from_slice(&trel).unwrap();
    let u_t_d = DevBuf::from_slice(&trel).unwrap();
    let d_t_d = DevBuf::from_slice(&trel).unwrap();
    let g_su_d = DevBuf::from_slice(&suh).unwrap();
    let u_su_d = DevBuf::from_slice(&suh).unwrap();
    let d_su_d = DevBuf::from_slice(&suh).unwrap();
    let g_sv_d = DevBuf::from_slice(&svh).unwrap();
    let u_sv_d = DevBuf::from_slice(&svh).unwrap();
    let d_sv_d = DevBuf::from_slice(&svh).unwrap();
    let n_experts = ne + 1;
    let mk_tbl = |b: &DevBuf| -> DevBuf {
        let v = vec![b.ptr() as u64; n_experts];
        DevBuf::from_slice(&v).unwrap()
    };
    let g_t = mk_tbl(&g_t_d);
    let u_t = mk_tbl(&u_t_d);
    let d_t = mk_tbl(&d_t_d);
    let g_su = mk_tbl(&g_su_d);
    let u_su = mk_tbl(&u_su_d);
    let d_su = mk_tbl(&d_su_d);
    let g_sv = mk_tbl(&g_sv_d);
    let u_sv = mk_tbl(&u_sv_d);
    let d_sv = mk_tbl(&d_sv_d);
    let ktab: Vec<i32> = vec![K2 as i32; n_experts * 3];
    let ktab_d = DevBuf::from_slice(&ktab).unwrap();
    let tb = gb_cuda::fn_moe::ExpertTables {
        g_t: &g_t,
        g_su: &g_su,
        g_sv: &g_sv,
        u_t: &u_t,
        u_su: &u_su,
        u_sv: &u_sv,
        d_t: &d_t,
        d_su: &d_su,
        d_sv: &d_sv,
        ktab: &ktab_d,
    };
    let ws_bytes = gb_cuda::fmoe::workspace_bytes(max_tokens, top_k, h, i, kc);
    for &t in &[1usize, 2, 4, 8] {
        for trial in 0..3 {
            let x: Vec<u16> = (0..t * h).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let lg: Vec<f32> = (0..t * ne).map(|_| r.uniform(-2.0, 2.0)).collect();
            let sg: Vec<u16> = (0..t).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
            let eb: Vec<f32> = (0..ne).map(|_| r.uniform(-0.25, 0.25)).collect();
            let x_d = DevBuf::from_slice(&x).unwrap();
            let lg_d = DevBuf::from_slice(&lg).unwrap();
            let sg_d = DevBuf::from_slice(&sg).unwrap();
            let eb_d = DevBuf::from_slice(&eb).unwrap();
            let shape = gb_cuda::fmoe::Shape { t, ne, top_k, h, i, kc, max_tokens };
            let gr = Some(gb_cuda::fmoe::GroupRoute { expert_bias: Some(&eb_d), n_group: 8, topk_group: 2, scale: 1.0 });
            let limits = gb_cuda::fmoe::SwigluLimit { routed: 5.0, shared: 5.0 };
            let run_stock = |ws: &mut DevBuf, out: &mut DevBuf| -> Vec<u16> {
                gb_cuda::fmoe::forward(&x_d, &lg_d, gb_cuda::fmoe::LogitsDtype::F32, &sg_d, &tb, out, shape, gr, limits, ws, &st)
                    .unwrap();
                st.sync().unwrap();
                out.to_vec().unwrap()
            };
            let run_ling = |ws: &mut DevBuf, out: &mut DevBuf| -> Vec<u16> {
                ling_fmoe::forward(&x_d, &lg_d, gb_cuda::fmoe::LogitsDtype::F32, &sg_d, &tb, out, shape, gr, limits, ws, &st)
                    .unwrap();
                st.sync().unwrap();
                out.to_vec().unwrap()
            };
            let mut ws_a = DevBuf::zeroed(ws_bytes).unwrap();
            let mut ws_b = DevBuf::zeroed(ws_bytes).unwrap();
            let mut out_a = DevBuf::zeroed(t * h * 2).unwrap();
            let mut out_b = DevBuf::zeroed(t * h * 2).unwrap();
            let a = run_stock(&mut ws_a, &mut out_a);
            let b = run_ling(&mut ws_b, &mut out_b);
            let diff = diff_count(&a, &b);
            assert_eq!(diff, 0, "ling_fmoe_forward differs from fmoe::forward at t={t} trial={trial}: {diff} bit diffs");
            // also against the forced-legacy stock path (the stable contract even if the stock smem budget changes)
            let prev = gb_cuda::fmoe::set_legacy(true);
            let a2 = run_stock(&mut ws_a, &mut out_a);
            gb_cuda::fmoe::set_legacy(prev);
            let diff2 = diff_count(&a2, &b);
            assert_eq!(diff2, 0, "ling_fmoe_forward differs from legacy fmoe::forward at t={t} trial={trial}: {diff2} bit diffs");
            // the fused-router variant: gb_ling_fmoe_forward_with_router computes the router logits inside its single
            // fused launch, so both the written logits (must equal bf16_linear_f32's bits) and the MoE output (must
            // equal the stock path on those logits) are checked
            let w: Vec<u16> = (0..ne * h).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let w_d = DevBuf::from_slice(&w).unwrap();
            let mut lg_d = DevBuf::zeroed(t * ne * 4).unwrap();
            bf16_linear_f32(&x_d, &w_d, &mut lg_d, t, h, ne, &st).unwrap();
            st.sync().unwrap();
            let lg_ref = lg_d.to_vec().unwrap();
            let mut out_ref = DevBuf::zeroed(t * h * 2).unwrap();
            let mut ws_ref = DevBuf::zeroed(ws_bytes).unwrap();
            gb_cuda::fmoe::forward(
                &x_d,
                &lg_d,
                gb_cuda::fmoe::LogitsDtype::F32,
                &sg_d,
                &tb,
                &mut out_ref,
                shape,
                gr,
                limits,
                &mut ws_ref,
                &st,
            )
            .unwrap();
            st.sync().unwrap();
            let refv = out_ref.to_vec().unwrap();
            let mut ws_c = DevBuf::zeroed(ws_bytes).unwrap();
            let mut out_c = DevBuf::zeroed(t * h * 2).unwrap();
            ling_fmoe::forward_with_router(
                &x_d,
                &w_d,
                &mut lg_d,
                gb_cuda::fmoe::LogitsDtype::F32,
                &sg_d,
                &tb,
                &mut out_c,
                shape,
                gr,
                limits,
                &mut ws_c,
                &st,
                None,
            )
            .unwrap();
            st.sync().unwrap();
            let c = out_c.to_vec().unwrap();
            let diff3 = diff_count(&refv, &c);
            assert_eq!(
                diff3, 0,
                "ling_fmoe forward_with_router differs from fmoe::forward at t={t} trial={trial}: {diff3} bit diffs"
            );
            let lg_after = lg_d.to_vec().unwrap();
            let diff4 = diff_count_f32(&lg_ref, &lg_after);
            assert_eq!(diff4, 0, "fused router logits differ from bf16_linear_f32 at t={t} trial={trial}: {diff4} bit diffs");
        }
    }
}

// The post-MoE epilogue fused into the down kernel's last finishing warp (gb_ling_fmoe_forward_with_router's
// epilogue params) vs the same forward + the separate `norm_rot_in_triple`/`norm_rot_in_single` launches: the
// hidden stream, the next input norm and the prerot buffers must all be 0-bit-diff.
#[test]
#[ignore]
fn ling_fmoe_down_epilogue_matches_separate() {
    let Some(st) = gpu() else { return };
    let (h, i, ne, top_k, kc, max_tokens) = (4096usize, 1024usize, 512usize, 8usize, 8usize, 64usize);
    const K2: usize = 7;
    let words = (h / 16) * (i / 16) * 4 * K2;
    let mut r = Rng(0x51b8_5b84 ^ 0x5b84 ^ K2 as u64 ^ 0xdead);
    let trel: Vec<u32> = (0..words).map(|_| r.next_u64() as u32).collect();
    let suh: Vec<u16> = (0..h.max(i)).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let svh: Vec<u16> = (0..h.max(i)).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let g_t_d = DevBuf::from_slice(&trel).unwrap();
    let u_t_d = DevBuf::from_slice(&trel).unwrap();
    let d_t_d = DevBuf::from_slice(&trel).unwrap();
    let g_su_d = DevBuf::from_slice(&suh).unwrap();
    let u_su_d = DevBuf::from_slice(&suh).unwrap();
    let d_su_d = DevBuf::from_slice(&suh).unwrap();
    let g_sv_d = DevBuf::from_slice(&svh).unwrap();
    let u_sv_d = DevBuf::from_slice(&svh).unwrap();
    let d_sv_d = DevBuf::from_slice(&svh).unwrap();
    let n_experts = ne + 1;
    let mk_tbl = |b: &DevBuf| -> DevBuf {
        let v = vec![b.ptr() as u64; n_experts];
        DevBuf::from_slice(&v).unwrap()
    };
    let g_t = mk_tbl(&g_t_d);
    let u_t = mk_tbl(&u_t_d);
    let d_t = mk_tbl(&d_t_d);
    let g_su = mk_tbl(&g_su_d);
    let u_su = mk_tbl(&u_su_d);
    let d_su = mk_tbl(&d_su_d);
    let g_sv = mk_tbl(&g_sv_d);
    let u_sv = mk_tbl(&u_sv_d);
    let d_sv = mk_tbl(&d_sv_d);
    let ktab: Vec<i32> = vec![K2 as i32; n_experts * 3];
    let ktab_d = DevBuf::from_slice(&ktab).unwrap();
    let tb = gb_cuda::fn_moe::ExpertTables {
        g_t: &g_t,
        g_su: &g_su,
        g_sv: &g_sv,
        u_t: &u_t,
        u_su: &u_su,
        u_sv: &u_sv,
        d_t: &d_t,
        d_su: &d_su,
        d_sv: &d_sv,
        ktab: &ktab_d,
    };
    let ws_bytes = gb_cuda::fmoe::workspace_bytes(max_tokens, top_k, h, i, kc);
    let eps = 1e-6f32;
    // the next layer's input norm weights and its rotations' suh buffers (the epilogue's prerot)
    let nw: Vec<f32> = (0..h).map(|_| r.uniform(0.5, 1.5)).collect();
    let r_suh: Vec<u16> = (0..h).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    // the reference `norm_rot_in_triple` also runs the triple linear: real projection buffers (n = 8192 gives the
    // SK = 1 plan at k = 4096), their outputs are scratch only
    let n0 = 8192usize;
    let words_n = (h / 16) * (n0 / 16) * 4 * K2;
    let ntrel: Vec<u32> = (0..words_n).map(|_| r.next_u64() as u32).collect();
    let nsvh: Vec<u16> = (0..n0).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let ntrel_d = DevBuf::from_slice(&ntrel).unwrap();
    let nsvh_d = DevBuf::from_slice(&nsvh).unwrap();
    let nw_d = DevBuf::from_slice(&nw).unwrap();
    let r_suh_d = DevBuf::from_slice(&r_suh).unwrap();
    for &t in &[1usize, 8] {
        for trial in 0..2 {
            let x: Vec<u16> = (0..t * h).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let hidden: Vec<u16> = (0..t * h).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let sg: Vec<u16> = (0..t).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
            let eb: Vec<f32> = (0..ne).map(|_| r.uniform(-0.25, 0.25)).collect();
            let w: Vec<u16> = (0..ne * h).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let x_d = DevBuf::from_slice(&x).unwrap();
            let mut hidden_a_d = DevBuf::from_slice(&hidden).unwrap();
            let hidden_b_d = DevBuf::from_slice(&hidden).unwrap();
            let sg_d = DevBuf::from_slice(&sg).unwrap();
            let eb_d = DevBuf::from_slice(&eb).unwrap();
            let w_d = DevBuf::from_slice(&w).unwrap();
            let shape = gb_cuda::fmoe::Shape { t, ne, top_k, h, i, kc, max_tokens };
            let gr = Some(gb_cuda::fmoe::GroupRoute { expert_bias: Some(&eb_d), n_group: 8, topk_group: 2, scale: 1.0 });
            let limits = gb_cuda::fmoe::SwigluLimit { routed: 5.0, shared: 5.0 };
            let mut lg_d = DevBuf::zeroed(t * ne * 4).unwrap();

            // Route A: the forward (no epilogue) + the separate norm/rot_in launches.
            let mut out_a = DevBuf::zeroed(t * h * 2).unwrap();
            let mut ws_a = DevBuf::zeroed(ws_bytes).unwrap();
            ling_fmoe::forward_with_router(
                &x_d,
                &w_d,
                &mut lg_d,
                gb_cuda::fmoe::LogitsDtype::F32,
                &sg_d,
                &tb,
                &mut out_a,
                shape,
                gr,
                limits,
                &mut ws_a,
                &st,
                None,
            )
            .unwrap();
            let mut norm_a = DevBuf::zeroed(t * h * 2).unwrap();
            let mut xh_a0 = DevBuf::zeroed(t * h * 2).unwrap();
            let mut xh_a1 = DevBuf::zeroed(t * h * 2).unwrap();
            let mut xh_a2 = DevBuf::zeroed(t * h * 2).unwrap();
            let mut ws_n = DevBuf::zeroed(workspace_bytes(t, h)).unwrap();
            let hid_a_x = hidden_a_d.subview(0, t * h * 2);
            let mut y_ref0 = DevBuf::zeroed(t * n0 * 2).unwrap();
            let mut y_ref1 = DevBuf::zeroed(t * n0 * 2).unwrap();
            let mut y_ref2 = DevBuf::zeroed(t * n0 * 2).unwrap();
            norm_rot_in_triple(
                &hid_a_x,
                Some(&out_a),
                &nw_d,
                Some(&mut hidden_a_d),
                &mut norm_a,
                proj_norm_rot(&mut y_ref0, &ntrel_d, &r_suh_d, &nsvh_d, h, n0, K2),
                proj_norm_rot(&mut y_ref1, &ntrel_d, &r_suh_d, &nsvh_d, h, n0, K2),
                proj_norm_rot(&mut y_ref2, &ntrel_d, &r_suh_d, &nsvh_d, h, n0, K2),
                t,
                h,
                Dtype::F16,
                Dtype::F16,
                K2,
                K2,
                CB,
                eps,
                &mut ws_n,
                &st,
            )
            .unwrap();
            // reference rotations of the norm output (the prerot buffers), one single rotation per suh
            unsafe extern "C" {
                fn gb_exl3_rotate_in(
                    x: *const core::ffi::c_void,
                    x_dtype: i32,
                    suh: *const core::ffi::c_void,
                    xh: *mut core::ffi::c_void,
                    m: i32,
                    k: i32,
                    stream: *mut core::ffi::c_void,
                ) -> i32;
            }
            unsafe {
                assert_eq!(gb_exl3_rotate_in(norm_a.ptr(), 1, r_suh_d.ptr(), xh_a0.ptr(), t as i32, h as i32, st.raw()), 0);
                assert_eq!(gb_exl3_rotate_in(norm_a.ptr(), 1, r_suh_d.ptr(), xh_a1.ptr(), t as i32, h as i32, st.raw()), 0);
                assert_eq!(gb_exl3_rotate_in(norm_a.ptr(), 1, r_suh_d.ptr(), xh_a2.ptr(), t as i32, h as i32, st.raw()), 0);
            }

            // Route B: the same forward with the down-kernel epilogue (triple prerot).
            let mut out_b = DevBuf::zeroed(t * h * 2).unwrap();
            let mut ws_b = DevBuf::zeroed(ws_bytes).unwrap();
            let mut norm_b = DevBuf::zeroed(t * h * 2).unwrap();
            let mut xh_b0 = DevBuf::zeroed(t * h * 2).unwrap();
            let mut xh_b1 = DevBuf::zeroed(t * h * 2).unwrap();
            let mut xh_b2 = DevBuf::zeroed(t * h * 2).unwrap();
            let prerot = Prerot::triple(&r_suh_d, &r_suh_d, &r_suh_d, &mut xh_b0, &mut xh_b1, &mut xh_b2);
            let epilogue = ling_fmoe::FmoeEpilogue { hidden: &hidden_b_d, norm_w: &nw_d, norm_out: &mut norm_b, eps, prerot };
            ling_fmoe::forward_with_router(
                &x_d,
                &w_d,
                &mut lg_d,
                gb_cuda::fmoe::LogitsDtype::F32,
                &sg_d,
                &tb,
                &mut out_b,
                shape,
                gr,
                limits,
                &mut ws_b,
                &st,
                Some(epilogue),
            )
            .unwrap();
            st.sync().unwrap();
            let tag = format!("triple t={t} trial={trial}");
            let ok = check_eq(&format!("{tag} out"), &dl::<u16>(&out_a, &st), &dl::<u16>(&out_b, &st))
                & check_eq(&format!("{tag} hidden"), &dl::<u16>(&hidden_a_d, &st), &dl::<u16>(&hidden_b_d, &st))
                & check_eq(&format!("{tag} norm"), &dl::<u16>(&norm_a, &st), &dl::<u16>(&norm_b, &st))
                & check_eq(&format!("{tag} xh0"), &dl::<u16>(&xh_a0, &st), &dl::<u16>(&xh_b0, &st))
                & check_eq(&format!("{tag} xh1"), &dl::<u16>(&xh_a1, &st), &dl::<u16>(&xh_b1, &st))
                & check_eq(&format!("{tag} xh2"), &dl::<u16>(&xh_a2, &st), &dl::<u16>(&xh_b2, &st));
            assert!(ok, "{tag}: down-kernel post-MoE epilogue diverged (triple)");
            eprintln!("{tag}: identical");

            // Single-prerot variant (the last-layer head site).
            let mut hidden_c_d = DevBuf::from_slice(&hidden).unwrap();
            let mut out_c = DevBuf::zeroed(t * h * 2).unwrap();
            let mut ws_c = DevBuf::zeroed(ws_bytes).unwrap();
            ling_fmoe::forward_with_router(
                &x_d,
                &w_d,
                &mut lg_d,
                gb_cuda::fmoe::LogitsDtype::F32,
                &sg_d,
                &tb,
                &mut out_c,
                shape,
                gr,
                limits,
                &mut ws_c,
                &st,
                None,
            )
            .unwrap();
            let mut norm_c = DevBuf::zeroed(t * h * 2).unwrap();
            let mut xh_c0 = DevBuf::zeroed(t * h * 2).unwrap();
            let mut ws_n2 = DevBuf::zeroed(workspace_bytes(t, h)).unwrap();
            let hid_c_x = hidden_c_d.subview(0, t * h * 2);
            norm_rot_in_single(
                &hid_c_x,
                Some(&out_c),
                &nw_d,
                Some(&mut hidden_c_d),
                &mut norm_c,
                &r_suh_d,
                &mut xh_c0,
                t,
                h,
                eps,
                &st,
            )
            .unwrap();
            let mut hidden_d_d = DevBuf::from_slice(&hidden).unwrap();
            let mut out_d = DevBuf::zeroed(t * h * 2).unwrap();
            let mut ws_d = DevBuf::zeroed(ws_bytes).unwrap();
            let mut norm_d = DevBuf::zeroed(t * h * 2).unwrap();
            let mut xh_d0 = DevBuf::zeroed(t * h * 2).unwrap();
            let prerot_s = Prerot::single(&r_suh_d, &mut xh_d0);
            let epilogue_s =
                ling_fmoe::FmoeEpilogue { hidden: &hidden_d_d, norm_w: &nw_d, norm_out: &mut norm_d, eps, prerot: prerot_s };
            ling_fmoe::forward_with_router(
                &x_d,
                &w_d,
                &mut lg_d,
                gb_cuda::fmoe::LogitsDtype::F32,
                &sg_d,
                &tb,
                &mut out_d,
                shape,
                gr,
                limits,
                &mut ws_d,
                &st,
                Some(epilogue_s),
            )
            .unwrap();
            st.sync().unwrap();
            let tag = format!("single t={t} trial={trial}");
            let ok = check_eq(&format!("{tag} out"), &dl::<u16>(&out_c, &st), &dl::<u16>(&out_d, &st))
                & check_eq(&format!("{tag} hidden"), &dl::<u16>(&hidden_c_d, &st), &dl::<u16>(&hidden_d_d, &st))
                & check_eq(&format!("{tag} norm"), &dl::<u16>(&norm_c, &st), &dl::<u16>(&norm_d, &st))
                & check_eq(&format!("{tag} xh0"), &dl::<u16>(&xh_c0, &st), &dl::<u16>(&xh_d0, &st));
            assert!(ok, "{tag}: down-kernel post-MoE epilogue diverged (single)");
            eprintln!("{tag}: identical");
        }
    }
}

/// Fused `window` + `norm` (`gb_ling_kda_window_and_norm` / `ling_kda_fused::window_and_norm`) vs the two separate
/// launches (`gb_kda_window` + `gb_kda_norm`) on identical random decode batches: `rows in {1, 2, 4, 8}` for
/// `n_seq in {1, 2}`, plus a 64-head sweep and an R = 4 sweep; `y`, the full `S` pool and the fp16 norm output must
/// be 0-bit-diff. `head_cnt` is allocated zeroed once and reused across every iteration (the kernel self-resets it).
#[test]
fn ling_kda_window_and_norm_matches_separate() {
    let Some(st) = gpu() else { return };
    let mut r = Rng(0x5b84_2611);
    const D: usize = 128;

    unsafe extern "C" {
        fn gb_kda_window(
            q: *const f32,
            k: *const f32,
            v: *const core::ffi::c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            parity: *const i32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            heads: i32,
            rz: i32,
            write_final: i32,
            r_step: i32,
            y: *mut core::ffi::c_void,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        fn gb_kda_norm(
            y: *const core::ffi::c_void,
            z: *const f32,
            w: *const core::ffi::c_void,
            eps: f32,
            rows: i32,
            heads: i32,
            out: *mut core::ffi::c_void,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        fn gb_ling_kda_window_and_norm(
            q: *const f32,
            k: *const f32,
            v: *const core::ffi::c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            parity: *const i32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            heads: i32,
            rz: i32,
            r_step: i32,
            y: *mut core::ffi::c_void,
            head_cnt: *mut i32,
            eps: f32,
            z: *const f32,
            norm_w: *const core::ffi::c_void,
            out: *mut core::ffi::c_void,
            o_suh: *const core::ffi::c_void,
            o_xh: *mut core::ffi::c_void,
            stream: *mut core::ffi::c_void,
        ) -> i32;
    }

    let eps = 1e-5f32;
    let mut combos: Vec<(usize, usize, usize, usize)> = Vec::new();
    for &heads in &[8usize, 64] {
        for &n_seq in &[1usize, 2] {
            for &rows in &[1usize, 2, 4, 8] {
                combos.push((heads, 8, n_seq, rows));
            }
        }
    }
    combos.push((8, 4, 2, 8)); // the R = 4 profile
    let combos_len = combos.len();

    let mut head_cnt: Option<DevBuf> = None;
    let mut done = 0usize;
    for (heads0, r_step, n_seq0, rows0) in combos {
        let (heads, n_seq, rows) = (heads0 as i32, n_seq0 as i32, rows0 as i32);
        // sequence lengths: first sequences take rows / n_seq (>= 1), the last the remainder (may be empty)
        let lens: Vec<i32> = {
            let mut l = Vec::with_capacity(n_seq as usize);
            let mut rem = rows;
            for i in 0..n_seq as usize {
                let take = if i + 1 == n_seq as usize { rem } else { (rows / n_seq).max(1) };
                l.push(take);
                rem -= take;
            }
            l
        };
        let mut seq_start: Vec<i32> = vec![0];
        for &l in &lens {
            let last = *seq_start.last().unwrap();
            seq_start.push(last + l);
        }
        let slot: Vec<i32> = (0..n_seq0).map(|i| i as i32).collect();
        let seq_len = n_seq0 * 2 * heads0 * D * D;
        let row_len = rows0 * heads0 * D;

        let s_host: Vec<f32> = (0..seq_len).map(|_| r.uniform(-0.5, 0.5)).collect();
        let parity_host: Vec<i32> = (0..n_seq0).map(|i| (i % 2) as i32).collect();
        let q_host: Vec<f32> = (0..row_len).map(|_| r.uniform(-0.5, 0.5)).collect();
        let k_host: Vec<f32> = (0..row_len).map(|_| r.uniform(-0.5, 0.5)).collect();
        let v_host: Vec<u16> = (0..row_len).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
        let g_host: Vec<f32> = (0..row_len).map(|_| r.uniform(-0.5, 0.5)).collect();
        let beta_host: Vec<f32> = (0..rows0 * heads0).map(|_| r.uniform(-0.5, 0.5)).collect();
        let z_host: Vec<f32> = (0..row_len).map(|_| r.uniform(-0.5, 0.5)).collect();
        let norm_w_host: Vec<u16> = (0..D).map(|_| bf(r.uniform(-0.5, 0.5))).collect();

        let mk = || {
            (
                DevBuf::from_slice(&s_host).unwrap(),
                DevBuf::from_slice(&parity_host).unwrap(),
                DevBuf::from_slice(&q_host).unwrap(),
                DevBuf::from_slice(&k_host).unwrap(),
                DevBuf::from_slice(&v_host).unwrap(),
                DevBuf::from_slice(&g_host).unwrap(),
                DevBuf::from_slice(&beta_host).unwrap(),
                DevBuf::from_slice(&z_host).unwrap(),
                DevBuf::from_slice(&norm_w_host).unwrap(),
                DevBuf::from_slice(&seq_start).unwrap(),
                DevBuf::from_slice(&slot).unwrap(),
            )
        };
        let (s_a, p_a, q_a, k_a, v_a, g_a, b_a, z_a, w_a, ss_a, sl_a) = mk();
        let (s_b, p_b, q_b, k_b, v_b, g_b, b_b, z_b, w_b, ss_b, sl_b) = mk();

        let y_bytes = rows0 * heads0 * D * 2;
        let mut y_a = DevBuf::zeroed(y_bytes).unwrap();
        let mut y_b = DevBuf::zeroed(y_bytes).unwrap();
        let mut out_a = DevBuf::zeroed(y_bytes).unwrap();
        let mut out_b = DevBuf::zeroed(y_bytes).unwrap();

        if head_cnt.is_none() {
            head_cnt = Some(DevBuf::zeroed(n_seq0 * heads0 * 4).unwrap());
        }

        // reference: two separate launches
        let rc = unsafe {
            gb_kda_window(
                q_a.ptr() as *const f32,
                k_a.ptr() as *const f32,
                v_a.ptr(),
                g_a.ptr() as *const f32,
                b_a.ptr() as *const f32,
                s_a.ptr() as *mut f32,
                p_a.ptr() as *const i32,
                ss_a.ptr() as *const i32,
                sl_a.ptr() as *const i32,
                n_seq,
                heads,
                0, // rz
                1, // write_final
                r_step as i32,
                y_a.ptr(),
                st.raw(),
            )
        };
        assert_eq!(rc, 0, "separate window rc={rc} (r_step={r_step} heads={heads} n_seq={n_seq} rows={rows})");
        let rc = unsafe { gb_kda_norm(y_a.ptr(), z_a.ptr() as *const f32, w_a.ptr(), eps, rows, heads, out_a.ptr(), st.raw()) };
        assert_eq!(rc, 0, "separate norm rc={rc}");

        // fused: one launch
        let hc = head_cnt.as_ref().unwrap();
        let rc = unsafe {
            gb_ling_kda_window_and_norm(
                q_b.ptr() as *const f32,
                k_b.ptr() as *const f32,
                v_b.ptr(),
                g_b.ptr() as *const f32,
                b_b.ptr() as *const f32,
                s_b.ptr() as *mut f32,
                p_b.ptr() as *const i32,
                ss_b.ptr() as *const i32,
                sl_b.ptr() as *const i32,
                n_seq,
                heads,
                0,
                r_step as i32,
                y_b.ptr(),
                hc.ptr() as *mut i32,
                eps,
                z_b.ptr() as *const f32,
                w_b.ptr(),
                out_b.ptr(),
                core::ptr::null(),
                core::ptr::null_mut(),
                st.raw(),
            )
        };
        assert_eq!(rc, 0, "fused window+norm rc={rc} (r_step={r_step} heads={heads} n_seq={n_seq} rows={rows})");

        st.sync().unwrap();
        let mut ya: Vec<u16> = vec![0; y_bytes / 2];
        let mut yb: Vec<u16> = vec![0; y_bytes / 2];
        y_a.download(&mut ya).unwrap();
        y_b.download(&mut yb).unwrap();
        let mut sa: Vec<f32> = vec![0.0; seq_len];
        let mut sb: Vec<f32> = vec![0.0; seq_len];
        s_a.download(&mut sa).unwrap();
        s_b.download(&mut sb).unwrap();
        let mut oa: Vec<u16> = vec![0; y_bytes / 2];
        let mut ob: Vec<u16> = vec![0; y_bytes / 2];
        out_a.download(&mut oa).unwrap();
        out_b.download(&mut ob).unwrap();

        let tag = format!("r_step={r_step} heads={heads} n_seq={n_seq} rows={rows}");
        assert_eq!(diff_count(&ya, &yb), 0, "y diff ({tag})");
        assert_eq!(diff_count_f32(&sa, &sb), 0, "S diff ({tag})");
        assert_eq!(diff_count(&oa, &ob), 0, "norm out diff ({tag})");
        done += 1;
    }
    assert_eq!(done, combos_len);
}

// prep/absorb/rope must produce bit-identical cache, q_lat and q_rot at every row count (1..24, crossing the
// 16-row absorb tile boundary). All buffers are fresh per arm; only the inputs are shared.
#[test]
#[ignore]
fn fused_mla_prep_absorb_rope_matches_separate() {
    let Some(st) = gpu() else { return };
    const NH: usize = 64;
    const QK: usize = 192;
    let mut r = Rng(0x7a11_1e5a ^ 0x5b84);
    for &rows in &[1usize, 2, 4, 8, 24] {
        for trial in 0..3 {
            let q: Vec<u16> = (0..rows * NH * QK).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let kv: Vec<u16> = (0..rows * 640).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let norm_w: Vec<u16> = (0..512).map(|_| bf(r.uniform(0.5, 1.5))).collect();
            let cos: Vec<f32> = (0..32).map(|_| r.uniform(-1.0, 1.0)).collect();
            let sin: Vec<f32> = (0..32).map(|_| r.uniform(-1.0, 1.0)).collect();
            let slots: Vec<i32> = vec![0; rows];
            let pos: Vec<i32> = (0..rows).map(|i| ((i * 137 + trial) % 2048) as i32).collect();
            let uk: Vec<u16> = (0..64 * 128 * 512).map(|_| bf(r.uniform(-0.25, 0.25))).collect();
            let uvt: Vec<u16> = (0..64 * 512 * 128).map(|_| bf(r.uniform(-0.25, 0.25))).collect();
            let g: Vec<u16> = (0..64 * 4096).map(|_| bf(r.uniform(-0.25, 0.25))).collect();

            let q_d = DevBuf::from_slice(&q).unwrap();
            let kv_d = DevBuf::from_slice(&kv).unwrap();
            let nw_d = DevBuf::from_slice(&norm_w).unwrap();
            let cos_d = DevBuf::from_slice(&cos).unwrap();
            let sin_d = DevBuf::from_slice(&sin).unwrap();
            let uk_d = DevBuf::from_slice(&uk).unwrap();
            let uvt_d = DevBuf::from_slice(&uvt).unwrap();
            let g_d = DevBuf::from_slice(&g).unwrap();
            let w = gb_cuda::mla_gated::MlaGatedWeights {
                uk: uk_d.subview(0, uk_d.bytes()),
                uvt: uvt_d.subview(0, uvt_d.bytes()),
                g: g_d.subview(0, g_d.bytes()),
                dtype: Dtype::BF16,
            };
            let table = BlockTable::from_host(&[vec![0, 1, 2, 3, 4, 5, 6, 7]], 8).unwrap();
            let ri = RowIndex::from_host(&slots, &pos).unwrap();

            // Route A: the three separate launches.
            let mut cache_a = MlaGatedCache::new(8, Dtype::BF16).unwrap();
            let mut q_lat_a = DevBuf::zeroed(rows * 64 * 512 * 2).unwrap();
            let mut q_rot_a = DevBuf::zeroed(rows * 64 * 64 * 2).unwrap();
            mla_gated::kv_prep(&kv_d, &nw_d, &cos_d, &sin_d, &ri, &table, &mut cache_a, rows, &st).unwrap();
            tc::absorb(&q_d, &w, &mut q_lat_a, rows, &st).unwrap();
            mla_gated::q_rope_only(&q_d, &cos_d, &sin_d, &ri, &mut q_rot_a, rows, &st).unwrap();

            // Route B: the fused launch.
            let mut cache_b = MlaGatedCache::new(8, Dtype::BF16).unwrap();
            let mut q_lat_b = DevBuf::zeroed(rows * 64 * 512 * 2).unwrap();
            let mut q_rot_b = DevBuf::zeroed(rows * 64 * 64 * 2).unwrap();
            tc::mla_prep_absorb_rope(
                &q_d,
                &kv_d,
                &nw_d,
                &cos_d,
                &sin_d,
                &ri,
                &table,
                &mut cache_b,
                &w,
                &mut q_lat_b,
                &mut q_rot_b,
                rows,
                &st,
            )
            .unwrap();

            st.sync().unwrap();
            let ca: Vec<u16> = cache_a.buf.to_vec().unwrap();
            let cb: Vec<u16> = cache_b.buf.to_vec().unwrap();
            let la: Vec<u16> = q_lat_a.to_vec().unwrap();
            let lb: Vec<u16> = q_lat_b.to_vec().unwrap();
            let ra: Vec<u16> = q_rot_a.to_vec().unwrap();
            let rb: Vec<u16> = q_rot_b.to_vec().unwrap();
            let tag = format!("rows={rows} trial={trial}");
            assert_eq!(diff_count(&ca, &cb), 0, "cache diff ({tag})");
            assert_eq!(diff_count(&la, &lb), 0, "q_lat diff ({tag})");
            assert_eq!(diff_count(&ra, &rb), 0, "q_rot diff ({tag})");
        }
    }
}

// gb_ling_rmsnorm_linear_bf16 vs glue::rmsnorm + Exl3Weights::k2_forward (the MLA q_b shape: k = 1536,
// n = 12288, k2 = 7, codebook 2, bf16 in/out): the fused norm + input rotation + linear must be bit-identical
// on both x (the in-place norm) and y at every decode row count.
#[test]
#[ignore]
fn fused_rmsnorm_linear_bf16_matches_separate() {
    let Some(st) = gpu() else { return };
    let (k, n, k2) = (1536usize, 12288usize, 7usize);
    let eps = 1e-6f32;
    let nw = nw_for(k, n, k2);
    let mut r = Rng(0x11e1_5a7c ^ k2 as u64 ^ (k as u64) << 16);
    for &m in &[1usize, 2, 4, 8] {
        for trial in 0..3 {
            let xs: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let t: Vec<u32> = (0..nw).map(|_| r.next_u64() as u32).collect();
            let s: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v: Vec<u16> = (0..n).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let w: Vec<f32> = (0..k).map(|_| r.uniform(0.5, 1.5)).collect();

            let t_d = DevBuf::from_slice(&t).unwrap();
            let s_d = DevBuf::from_slice(&s).unwrap();
            let v_d = DevBuf::from_slice(&v).unwrap();
            let w_d = DevBuf::from_slice(&w).unwrap();

            // Route A: separate launches (glue::rmsnorm, then k2_forward).
            let mut xa_d = DevBuf::from_slice(&xs).unwrap();
            let mut ya_d = DevBuf::zeroed(m * n * 2).unwrap();
            unsafe {
                gb_cuda::glue::rmsnorm(xa_d.ptr().cast_const(), w_d.ptr().cast_const(), xa_d.ptr(), m, k, eps, &st).unwrap();
            }
            let mut ws_a = DevBuf::zeroed(linear_workspace_bytes(m, k, n, k2 / 2)).unwrap();
            Exl3Weights::k2_forward(
                &xa_d,
                &mut ya_d,
                m,
                Dtype::BF16,
                Dtype::BF16,
                k,
                n,
                k2,
                gb_exl3::Codebook::Mul1,
                &t_d,
                &s_d,
                &v_d,
                None,
                &mut ws_a,
                &st,
            )
            .unwrap();

            // Route B: the fused launch.
            let mut xb_d = DevBuf::from_slice(&xs).unwrap();
            let mut yb_d = DevBuf::zeroed(m * n * 2).unwrap();
            let mut ws_b = DevBuf::zeroed(workspace_bytes(m, k)).unwrap();
            gb_cuda::ling_kda_proj_fused::rmsnorm_linear_bf16(
                &mut xb_d, &w_d, &t_d, &s_d, &v_d, &mut yb_d, m, k, n, k2, CB, eps, &mut ws_b, &st,
            )
            .unwrap();

            st.sync().unwrap();
            let xa: Vec<u16> = xa_d.to_vec().unwrap();
            let ya: Vec<u16> = ya_d.to_vec().unwrap();
            let xb: Vec<u16> = xb_d.to_vec().unwrap();
            let yb: Vec<u16> = yb_d.to_vec().unwrap();
            let tag = format!("m={m} trial={trial}");
            assert!(check_eq(&format!("norm {tag}"), &xb, &xa), "fused rmsnorm differs ({tag})");
            assert!(check_eq(&format!("y {tag}"), &yb, &ya), "fused q_b differs ({tag})");
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion 1: the batched commit tail (`kda::commit_tail_all`) against the per-layer `kda::commit` (the
// conv shift + parity flip of every KDA layer of a flush in one launch instead of n_kda). Both routes start
// from identical records/pools and must leave every pool byte (state, conv, parity) identical.
// ---------------------------------------------------------------------------
#[test]
#[ignore]
fn commit_tail_all_matches_per_layer_commit() {
    let Some(st) = gpu() else { return };
    let dims = KdaDims { heads: 8, hidden: 512 };
    let params = Params { lower: -5.0, eps: 1e-5, round: gb_cuda::kda::Round::NearestEven, r_step: 8 };
    let mut r = Rng(0xc0ffee);
    let n_layers = 3usize;
    let lens = [3usize, 5];
    let total = lens.iter().sum::<usize>();
    let conv_dim = dims.conv_dim();
    let vd = dims.v_dim();
    let conv_slot = dims.conv_slot_elems();

    // identical per-layer weights (prep/replay read conv_w only), records and pools for both routes
    let conv_w: Vec<u16> = (0..conv_dim * 4).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
    let conv_w_d = DevBuf::from_slice(&conv_w).unwrap();
    let w = Weights {
        b: &DevBuf::from_slice(&Vec::<u16>::new()).unwrap(),
        f_a: &DevBuf::from_slice(&Vec::<u16>::new()).unwrap(),
        g_a: &DevBuf::from_slice(&Vec::<u16>::new()).unwrap(),
        f_b: &DevBuf::from_slice(&Vec::<u16>::new()).unwrap(),
        g_b: &DevBuf::from_slice(&Vec::<u16>::new()).unwrap(),
        conv_w: &conv_w_d,
        a_log: &DevBuf::from_slice(&Vec::<f32>::new()).unwrap(),
        dt_bias: &DevBuf::from_slice(&Vec::<f32>::new()).unwrap(),
        o_norm: &DevBuf::from_slice(&Vec::<u16>::new()).unwrap(),
    };
    let mut recs_a = vec![];
    let mut recs_b = vec![];
    for _ in 0..n_layers {
        let mut ra = Record::alloc(dims, total).unwrap();
        let mut rb = Record::alloc(dims, total).unwrap();
        let qkv: Vec<u16> = (0..total * conv_dim).map(|_| bf(r.uniform(-2.0, 2.0))).collect();
        let g: Vec<f32> = (0..total * vd).map(|_| r.uniform(0.0, 1.0)).collect();
        let beta: Vec<f32> = (0..total * dims.heads).map(|_| r.uniform(0.0, 1.0)).collect();
        ra.qkv.upload(&qkv).unwrap();
        rb.qkv.upload(&qkv).unwrap();
        ra.g.upload(&g).unwrap();
        rb.g.upload(&g).unwrap();
        ra.beta.upload(&beta).unwrap();
        rb.beta.upload(&beta).unwrap();
        recs_a.push(ra);
        recs_b.push(rb);
    }
    let mut pools_a = vec![];
    let mut pools_b = vec![];
    for _ in 0..n_layers {
        let mut pa = Pool::new(dims, 2).unwrap();
        let mut pb = Pool::new(dims, 2).unwrap();
        for slot in 0..2usize {
            let state = vec![0.0f32; dims.s_buf_elems()];
            let conv: Vec<u16> = (0..conv_slot).map(|_| bf(r.uniform(-2.0, 2.0))).collect();
            pa.write_slot(slot, &state, &conv).unwrap();
            pb.write_slot(slot, &state, &conv).unwrap();
        }
        let par: Vec<i32> = (0..2).map(|_| (r.next_u64() & 1) as i32).collect();
        pa.parity.upload(&par).unwrap();
        pb.parity.upload(&par).unwrap();
        pools_a.push(pa);
        pools_b.push(pb);
    }
    let ss_d = DevBuf::from_slice(&[0i32, 3, 8]).unwrap();
    let slot_d = DevBuf::from_slice(&[0i32, 1]).unwrap();
    let keep_d = DevBuf::from_slice(&[2i32, 5]).unwrap();
    let batch = gb_cuda::kda::Batch { seq_start: &ss_d, slot: &slot_d, n_seq: 2, rows: total };

    // the batched table (route B)
    let mut bytes = Vec::with_capacity(n_layers * 24);
    for i in 0..n_layers {
        bytes.extend_from_slice(&(recs_b[i].qkv.ptr() as usize as u64).to_le_bytes());
        bytes.extend_from_slice(&(pools_b[i].conv.ptr() as usize as u64).to_le_bytes());
        bytes.extend_from_slice(&(pools_b[i].parity.ptr() as usize as u64).to_le_bytes());
    }
    let layers_d = DevBuf::from_slice(&bytes).unwrap();
    let mut ws_a = Workspace::alloc(dims, total).unwrap();
    let mut ws_b = Workspace::alloc(dims, total).unwrap();

    for &keep_all in &[true, false] {
        // route A: per-layer commit
        for i in 0..n_layers {
            let keep = if keep_all { None } else { Some(&keep_d) };
            kda::commit(&st, dims, &w, &params, &mut pools_a[i], &batch, &recs_a[i], &mut ws_a, keep, true).unwrap();
        }
        // route B: per-layer replay (partial only) + one batched tail
        for i in 0..n_layers {
            if !keep_all {
                gb_cuda::ling_kda_fused::replay(
                    &st,
                    dims,
                    &w,
                    &params,
                    &pools_b[i],
                    &batch,
                    &recs_b[i],
                    &mut ws_b,
                    Some(&keep_d),
                    true,
                )
                .unwrap();
            }
        }
        gb_cuda::ling_kda_fused::commit_tail_all(
            &st,
            dims,
            &layers_d,
            n_layers,
            &batch,
            if keep_all { None } else { Some(&keep_d) },
            true,
        )
        .unwrap();
        st.sync().unwrap();
        for i in 0..n_layers {
            let ok = check_eq(
                &format!("keep_all={keep_all} L{i} conv"),
                &dl::<u16>(&pools_a[i].conv, &st),
                &dl::<u16>(&pools_b[i].conv, &st),
            ) & check_eq(
                &format!("keep_all={keep_all} L{i} parity"),
                &dl::<i32>(&pools_a[i].parity, &st),
                &dl::<i32>(&pools_b[i].parity, &st),
            ) & check_eq(
                &format!("keep_all={keep_all} L{i} s"),
                &dl::<f32>(&pools_a[i].s, &st),
                &dl::<f32>(&pools_b[i].s, &st),
            );
            assert!(ok, "keep_all={keep_all} L{i}: batched tail diverged from per-layer commit");
        }
        eprintln!("keep_all={keep_all}: identical");
    }
}

// ---------------------------------------------------------------------------
// Fusion 2: `silu_mul_linear_add_rmsnorm` (swiglu fused into the input rotation) against
// `glue::swiglu` + `linear_add_rmsnorm` on identical random inputs.
// ---------------------------------------------------------------------------
#[test]
#[ignore]
fn silu_mul_linear_add_rmsnorm_matches_separate() {
    let Some(st) = gpu() else { return };
    let lin_cnt = DevBuf::zeroed(1024).unwrap();

    let mut r = Rng(0x51a);
    let k = 2048usize;
    let n = 2048usize;
    let k2 = 7usize;
    let cb = 2i32;
    let eps = 1e-6f32;
    let nw = nw_for(k, n, k2);
    for &m in &[1usize, 8] {
        for trial in 0..3 {
            let g: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
            let u: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
            let t: Vec<u32> = (0..nw).map(|_| r.next_u64() as u32).collect();
            let s: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v: Vec<u16> = (0..n).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let hidden: Vec<u16> = (0..m * n).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
            let w_norm: Vec<f32> = (0..n).map(|_| r.uniform(0.5, 1.5)).collect();

            let g_d = DevBuf::from_slice(&g).unwrap();
            let u_d = DevBuf::from_slice(&u).unwrap();
            let t_d = DevBuf::from_slice(&t).unwrap();
            let s_d = DevBuf::from_slice(&s).unwrap();
            let v_d = DevBuf::from_slice(&v).unwrap();
            let w_d = DevBuf::from_slice(&w_norm).unwrap();
            let hidden_a_d = DevBuf::from_slice(&hidden).unwrap();
            let hidden_b_d = DevBuf::from_slice(&hidden).unwrap();
            let mut a_d = DevBuf::zeroed(m * k * 2).unwrap();
            let mut out_hidden_a_d = DevBuf::from_slice(&hidden).unwrap();
            let mut out_hidden_b_d = DevBuf::from_slice(&hidden).unwrap();
            let mut y_a_d = DevBuf::zeroed(m * n * 2).unwrap();
            let mut y_b_d = DevBuf::zeroed(m * n * 2).unwrap();
            let ws_bytes = linear_workspace_bytes(m, k, n, k2 / 2);
            let mut ws_a = DevBuf::zeroed(ws_bytes).unwrap();
            let mut ws_b = DevBuf::zeroed(ws_bytes).unwrap();

            // route A: swiglu + linear_add_rmsnorm
            unsafe { glue::swiglu(g_d.ptr(), u_d.ptr(), a_d.ptr(), m * k, &st) }.unwrap();
            linear_add_rmsnorm(
                &a_d,
                Dtype::BF16,
                &t_d,
                &s_d,
                &v_d,
                &hidden_a_d,
                &w_d,
                &mut out_hidden_a_d,
                &mut y_a_d,
                m,
                k,
                n,
                k2,
                cb,
                n,
                eps,
                &mut ws_a,
                false,
                &Prerot::default(),
                Some(&lin_cnt),
                &st,
            )
            .unwrap();
            // route B: fused
            silu_mul_linear_add_rmsnorm(
                &g_d,
                &u_d,
                &t_d,
                &s_d,
                &v_d,
                &hidden_b_d,
                &w_d,
                &mut out_hidden_b_d,
                &mut y_b_d,
                m,
                k,
                n,
                k2,
                cb,
                n,
                eps,
                &mut ws_b,
                &Prerot::default(),
                Some(&lin_cnt),
                &st,
            )
            .unwrap();
            let ok = check_eq(
                &format!("m={m} t{trial} hidden_out"),
                &dl::<u16>(&out_hidden_a_d, &st),
                &dl::<u16>(&out_hidden_b_d, &st),
            ) & check_eq(&format!("m={m} t{trial} y"), &dl::<u16>(&y_a_d, &st), &dl::<u16>(&y_b_d, &st));
            assert!(ok, "m={m} trial {trial}: fused swiglu+linear diverged");
            eprintln!("m={m} trial {trial}: identical");
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion 3: `norm_rot_in_triple` / `norm_rot_in_pair` (rmsnorm / add_rmsnorm fused into the rotation)
// against `glue::rmsnorm`/`glue::add_rmsnorm` + `fused_decode_triple`/`fused_decode_pair`.
// ---------------------------------------------------------------------------
fn norm_triple_case(dim: usize, res: bool, m: usize, trial: usize, r: &mut Rng, st: &Stream) {
    let k = dim;
    let n0 = 8192usize;
    let n12 = 8192usize;
    let k2_0 = 7usize;
    let k2_12 = 6usize;
    let cb = 2i32;
    let eps = 1e-6f32;
    let nw0 = nw_for(k, n0, k2_0);
    let nw12 = nw_for(k, n12, k2_12);
    let x: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
    let resv: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
    let w_norm: Vec<f32> = (0..k).map(|_| r.uniform(0.5, 1.5)).collect();
    let t0: Vec<u32> = (0..nw0).map(|_| r.next_u64() as u32).collect();
    let s0: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let v0: Vec<u16> = (0..n0).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let t1: Vec<u32> = (0..nw12).map(|_| r.next_u64() as u32).collect();
    let s1: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let v1: Vec<u16> = (0..n12).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let t2: Vec<u32> = (0..nw12).map(|_| r.next_u64() as u32).collect();
    let s2: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let v2: Vec<u16> = (0..n12).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let x_d = DevBuf::from_slice(&x).unwrap();
    let res_d = DevBuf::from_slice(&resv).unwrap();
    let w_d = DevBuf::from_slice(&w_norm).unwrap();
    let t0_d = DevBuf::from_slice(&t0).unwrap();
    let s0_d = DevBuf::from_slice(&s0).unwrap();
    let v0_d = DevBuf::from_slice(&v0).unwrap();
    let t1_d = DevBuf::from_slice(&t1).unwrap();
    let s1_d = DevBuf::from_slice(&s1).unwrap();
    let v1_d = DevBuf::from_slice(&v1).unwrap();
    let t2_d = DevBuf::from_slice(&t2).unwrap();
    let s2_d = DevBuf::from_slice(&s2).unwrap();
    let v2_d = DevBuf::from_slice(&v2).unwrap();

    // route A: glue norm + triple
    let mut norm_a_d = DevBuf::zeroed(m * k * 2).unwrap();
    let mut hidden_a_d = DevBuf::from_slice(&x).unwrap();
    if res {
        unsafe { glue::add_rmsnorm(x_d.ptr(), res_d.ptr(), w_d.ptr(), hidden_a_d.ptr(), norm_a_d.ptr(), m, k, eps, &st) }
            .unwrap();
    } else {
        unsafe { glue::rmsnorm(x_d.ptr(), w_d.ptr(), norm_a_d.ptr(), m, k, eps, &st) }.unwrap();
    }
    let mut qkv_a_d = DevBuf::zeroed(m * n0 * 4).unwrap();
    let mut f_a_d = DevBuf::zeroed(m * n12 * 2).unwrap();
    let mut g_a_d = DevBuf::zeroed(m * n12 * 2).unwrap();
    let mut ws_a = DevBuf::zeroed(workspace_bytes(m, k)).unwrap();
    fused_decode_triple(
        &norm_a_d,
        Proj { trellis: &t0_d, suh: &s0_d, svh: &v0_d, y: &mut qkv_a_d, k, n: n0, k2: k2_0, codebook: cb },
        Proj { trellis: &t1_d, suh: &s1_d, svh: &v1_d, y: &mut f_a_d, k, n: n12, k2: k2_12, codebook: cb },
        Proj { trellis: &t2_d, suh: &s2_d, svh: &v2_d, y: &mut g_a_d, k, n: n12, k2: k2_12, codebook: cb },
        m,
        Dtype::BF16,
        Dtype::F32,
        Dtype::F16,
        &mut ws_a,
        None,
        None,
        None,
        st,
    )
    .unwrap();

    // route B: fused
    let mut norm_b_d = DevBuf::zeroed(m * k * 2).unwrap();
    let mut hidden_b_d = DevBuf::from_slice(&x).unwrap();
    let mut qkv_b_d = DevBuf::zeroed(m * n0 * 4).unwrap();
    let mut f_b_d = DevBuf::zeroed(m * n12 * 2).unwrap();
    let mut g_b_d = DevBuf::zeroed(m * n12 * 2).unwrap();
    let mut ws_b = DevBuf::zeroed(workspace_bytes(m, k)).unwrap();
    let (res_h, ho) = if res { (Some(&res_d), Some(&mut hidden_b_d)) } else { (None, None) };
    norm_rot_in_triple(
        &x_d,
        res_h,
        &w_d,
        ho,
        &mut norm_b_d,
        Proj { trellis: &t0_d, suh: &s0_d, svh: &v0_d, y: &mut qkv_b_d, k, n: n0, k2: k2_0, codebook: cb },
        Proj { trellis: &t1_d, suh: &s1_d, svh: &v1_d, y: &mut f_b_d, k, n: n12, k2: k2_12, codebook: cb },
        Proj { trellis: &t2_d, suh: &s2_d, svh: &v2_d, y: &mut g_b_d, k, n: n12, k2: k2_12, codebook: cb },
        m,
        k,
        Dtype::F32,
        Dtype::F16,
        k2_0,
        k2_12,
        cb,
        eps,
        &mut ws_b,
        st,
    )
    .unwrap();

    let tag = format!("res={res} m={m} t{trial}");
    let ok = check_eq(&format!("{tag} norm"), &dl::<u16>(&norm_a_d, &st), &dl::<u16>(&norm_b_d, &st))
        & check_eq(&format!("{tag} qkv"), &dl::<f32>(&qkv_a_d, &st), &dl::<f32>(&qkv_b_d, &st))
        & check_eq(&format!("{tag} f"), &dl::<u16>(&f_a_d, &st), &dl::<u16>(&f_b_d, &st))
        & check_eq(&format!("{tag} g"), &dl::<u16>(&g_a_d, &st), &dl::<u16>(&g_b_d, &st));
    if res {
        let ok = ok & check_eq(&format!("{tag} hidden"), &dl::<u16>(&hidden_a_d, &st), &dl::<u16>(&hidden_b_d, &st));
        assert!(ok, "{tag}: fused norm+triple diverged");
    } else {
        assert!(ok, "{tag}: fused norm+triple diverged");
    }
    eprintln!("{tag}: identical");
}

#[test]
#[ignore]
fn norm_rot_in_triple_matches_separate() {
    let Some(st) = gpu() else { return };
    let mut r = Rng(0x7137);
    for &dim in &[2048usize, 4096] {
        for &res in &[false, true] {
            for &m in &[1usize, 8] {
                for trial in 0..2 {
                    norm_triple_case(dim, res, m, trial, &mut r, &st);
                }
            }
        }
    }
}

#[test]
#[ignore]
fn norm_rot_in_pair_matches_separate() {
    let Some(st) = gpu() else { return };
    let mut r = Rng(0x7a11);
    let k = 2048usize;
    let n = 2048usize;
    let k2 = 7usize;
    let cb = 2i32;
    let eps = 1e-6f32;
    let nw = nw_for(k, n, k2);
    for &res in &[false, true] {
        for &m in &[1usize, 8] {
            for trial in 0..2 {
                let x: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
                let resv: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
                let w_norm: Vec<f32> = (0..k).map(|_| r.uniform(0.5, 1.5)).collect();
                let t0: Vec<u32> = (0..nw).map(|_| r.next_u64() as u32).collect();
                let s0: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
                let v0: Vec<u16> = (0..n).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
                let t1: Vec<u32> = (0..nw).map(|_| r.next_u64() as u32).collect();
                let s1: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
                let v1: Vec<u16> = (0..n).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
                let x_d = DevBuf::from_slice(&x).unwrap();
                let res_d = DevBuf::from_slice(&resv).unwrap();
                let w_d = DevBuf::from_slice(&w_norm).unwrap();
                let t0_d = DevBuf::from_slice(&t0).unwrap();
                let s0_d = DevBuf::from_slice(&s0).unwrap();
                let v0_d = DevBuf::from_slice(&v0).unwrap();
                let t1_d = DevBuf::from_slice(&t1).unwrap();
                let s1_d = DevBuf::from_slice(&s1).unwrap();
                let v1_d = DevBuf::from_slice(&v1).unwrap();

                // route A
                let mut norm_a_d = DevBuf::zeroed(m * k * 2).unwrap();
                let mut hidden_a_d = DevBuf::from_slice(&x).unwrap();
                if res {
                    unsafe {
                        glue::add_rmsnorm(x_d.ptr(), res_d.ptr(), w_d.ptr(), hidden_a_d.ptr(), norm_a_d.ptr(), m, k, eps, &st)
                    }
                    .unwrap();
                } else {
                    unsafe { glue::rmsnorm(x_d.ptr(), w_d.ptr(), norm_a_d.ptr(), m, k, eps, &st) }.unwrap();
                }
                let mut ya_d = DevBuf::zeroed(m * n * 2).unwrap();
                let mut yb_d = DevBuf::zeroed(m * n * 2).unwrap();
                let mut ws_a = DevBuf::zeroed(pair_workspace_bytes(m, k, n, n)).unwrap();
                fused_decode_pair(
                    &norm_a_d,
                    Proj { trellis: &t0_d, suh: &s0_d, svh: &v0_d, y: &mut ya_d, k, n, k2, codebook: cb },
                    Proj { trellis: &t1_d, suh: &s1_d, svh: &v1_d, y: &mut yb_d, k, n, k2, codebook: cb },
                    m,
                    Dtype::BF16,
                    Dtype::BF16,
                    &mut ws_a,
                    None,
                    None,
                    &st,
                )
                .unwrap();

                // route B
                let mut norm_b_d = DevBuf::zeroed(m * k * 2).unwrap();
                let mut hidden_b_d = DevBuf::from_slice(&x).unwrap();
                let mut ya_f_d = DevBuf::zeroed(m * n * 2).unwrap();
                let mut yb_f_d = DevBuf::zeroed(m * n * 2).unwrap();
                let mut ws_b = DevBuf::zeroed(pair_workspace_bytes(m, k, n, n)).unwrap();
                let (res_h, ho) = if res { (Some(&res_d), Some(&mut hidden_b_d)) } else { (None, None) };
                norm_rot_in_pair(
                    &x_d,
                    res_h,
                    &w_d,
                    ho,
                    &mut norm_b_d,
                    Proj { trellis: &t0_d, suh: &s0_d, svh: &v0_d, y: &mut ya_f_d, k, n, k2, codebook: cb },
                    Proj { trellis: &t1_d, suh: &s1_d, svh: &v1_d, y: &mut yb_f_d, k, n, k2, codebook: cb },
                    m,
                    k,
                    Dtype::BF16,
                    k2,
                    cb,
                    eps,
                    &mut ws_b,
                    &st,
                )
                .unwrap();

                let tag = format!("res={res} m={m} t{trial}");
                let ok = check_eq(&format!("{tag} norm"), &dl::<u16>(&norm_a_d, &st), &dl::<u16>(&norm_b_d, &st))
                    & check_eq(&format!("{tag} y0"), &dl::<u16>(&ya_d, &st), &dl::<u16>(&ya_f_d, &st))
                    & check_eq(&format!("{tag} y1"), &dl::<u16>(&yb_d, &st), &dl::<u16>(&yb_f_d, &st));
                let ok = if res {
                    ok & check_eq(&format!("{tag} hidden"), &dl::<u16>(&hidden_a_d, &st), &dl::<u16>(&hidden_b_d, &st))
                } else {
                    ok
                };
                assert!(ok, "{tag}: fused norm+pair diverged");
                eprintln!("{tag}: identical");
            }
        }
    }
}

// The MLA qkv + q_b fusion (`mla_qkv_qb_fused`) vs the separate sequence it replaces: `fused_decode_pair`
// (or `norm_rot_in_pair`, or the xh_pre variant) + `rmsnorm_linear_bf16`. The fused call must leave q_a (the
// normalised bf16), kv and q bit-identical to the separate launches at every decode row count, in all three
// producer modes (plain, prerot xh_pre, and the fused input norm with/without a residual).
#[test]
#[ignore]
fn mla_qkv_qb_fused_matches_separate() {
    let Some(st) = gpu() else { return };
    let (k, k2, qb_k2) = (K_MLA, 10usize, 7usize);
    let (n0, n1, nq) = (N0_MLA, N1_MLA, 12288usize);
    let eps = 1e-6f32;
    let nw0 = nw_for(k, n0, k2);
    let nw1 = nw_for(k, n1, k2);
    let nwq = nw_for(n0, nq, qb_k2);
    let mut r = Rng(0x8b44_9b84 ^ k as u64 ^ (n0 as u64) << 20);
    for &m in &[1usize, 8] {
        for trial in 0..2 {
            let x: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let resv: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let w_norm: Vec<f32> = (0..k).map(|_| r.uniform(0.5, 1.5)).collect();
            let t0: Vec<u32> = (0..nw0).map(|_| r.next_u64() as u32).collect();
            let s0: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v0: Vec<u16> = (0..n0).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let t1: Vec<u32> = (0..nw1).map(|_| r.next_u64() as u32).collect();
            let s1: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v1: Vec<u16> = (0..n1).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let tq: Vec<u32> = (0..nwq).map(|_| r.next_u64() as u32).collect();
            let sq: Vec<u16> = (0..n0).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let vq: Vec<u16> = (0..nq).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let qa_w: Vec<f32> = (0..n0).map(|_| r.uniform(0.5, 1.5)).collect();

            let x_d = DevBuf::from_slice(&x).unwrap();
            let res_d = DevBuf::from_slice(&resv).unwrap();
            let w_d = DevBuf::from_slice(&w_norm).unwrap();
            let t0_d = DevBuf::from_slice(&t0).unwrap();
            let s0_d = DevBuf::from_slice(&s0).unwrap();
            let v0_d = DevBuf::from_slice(&v0).unwrap();
            let t1_d = DevBuf::from_slice(&t1).unwrap();
            let s1_d = DevBuf::from_slice(&s1).unwrap();
            let v1_d = DevBuf::from_slice(&v1).unwrap();
            let tq_d = DevBuf::from_slice(&tq).unwrap();
            let sq_d = DevBuf::from_slice(&sq).unwrap();
            let vq_d = DevBuf::from_slice(&vq).unwrap();
            let qa_w_d = DevBuf::from_slice(&qa_w).unwrap();

            // Route A: the separate sequence (fused_decode_pair + rmsnorm_linear_bf16).
            let mut qa_a_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            let mut kv_a_d = DevBuf::zeroed(m * n1 * 2).unwrap();
            let mut ws_a = DevBuf::zeroed(pair_workspace_bytes(m, k, n0, n1)).unwrap();
            fused_decode_pair(
                &x_d,
                proj_mla(&mut qa_a_d, &t0_d, &s0_d, &v0_d, k, n0, k2),
                proj_mla(&mut kv_a_d, &t1_d, &s1_d, &v1_d, k, n1, k2),
                m,
                Dtype::BF16,
                Dtype::BF16,
                &mut ws_a,
                None,
                None,
                &st,
            )
            .unwrap();
            let mut q_a_d = DevBuf::zeroed(m * nq * 2).unwrap();
            let mut ws_q = DevBuf::zeroed(workspace_bytes(m, n0)).unwrap();
            rmsnorm_linear_bf16(&mut qa_a_d, &qa_w_d, &tq_d, &sq_d, &vq_d, &mut q_a_d, m, n0, nq, qb_k2, CB, eps, &mut ws_q, &st)
                .unwrap();

            // Route B: the fused call.
            let mut qa_b_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            let mut kv_b_d = DevBuf::zeroed(m * n1 * 2).unwrap();
            let mut q_b_d = DevBuf::zeroed(m * nq * 2).unwrap();
            let mut ws_b = DevBuf::zeroed(pair_workspace_bytes(m, k, n0, n1)).unwrap();
            let mut qb_xh_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            mla_qkv_qb_fused(
                &x_d,
                Dtype::BF16,
                None,
                None,
                None,
                None,
                proj_mla(&mut qa_b_d, &t0_d, &s0_d, &v0_d, k, n0, k2),
                proj_mla(&mut kv_b_d, &t1_d, &s1_d, &v1_d, k, n1, k2),
                &qa_w_d,
                proj_mla(&mut q_b_d, &tq_d, &sq_d, &vq_d, n0, nq, qb_k2),
                m,
                eps,
                &mut ws_b,
                None,
                None,
                &mut qb_xh_d,
                None,
                &st,
            )
            .unwrap();

            let tag = format!("plain m={m} trial={trial}");
            let ok = check_eq(&format!("{tag} q_a"), &dl::<u16>(&qa_a_d, &st), &dl::<u16>(&qa_b_d, &st))
                & check_eq(&format!("{tag} kv"), &dl::<u16>(&kv_a_d, &st), &dl::<u16>(&kv_b_d, &st))
                & check_eq(&format!("{tag} q"), &dl::<u16>(&q_a_d, &st), &dl::<u16>(&q_b_d, &st));
            assert!(ok, "{tag}: fused MLA qkv+qb diverged (plain)");
            eprintln!("{tag}: identical");

            // Route A2/B2: prerot (xh_pre) mode — pre-rotate the input with each projection's suh.
            let mut xh0_d = DevBuf::zeroed(m * k * 2).unwrap();
            let mut xh1_d = DevBuf::zeroed(m * k * 2).unwrap();
            unsafe extern "C" {
                fn gb_exl3_rotate_in(
                    x: *const core::ffi::c_void,
                    x_dtype: i32,
                    suh: *const core::ffi::c_void,
                    xh: *mut core::ffi::c_void,
                    m: i32,
                    k: i32,
                    stream: *mut core::ffi::c_void,
                ) -> i32;
            }
            unsafe {
                assert_eq!(gb_exl3_rotate_in(x_d.ptr(), 1, s0_d.ptr(), xh0_d.ptr(), m as i32, k as i32, st.raw()), 0);
                assert_eq!(gb_exl3_rotate_in(x_d.ptr(), 1, s1_d.ptr(), xh1_d.ptr(), m as i32, k as i32, st.raw()), 0);
            }
            let mut qa_a2_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            let mut kv_a2_d = DevBuf::zeroed(m * n1 * 2).unwrap();
            let mut ws_a2 = DevBuf::zeroed(pair_workspace_bytes(m, k, n0, n1)).unwrap();
            fused_decode_pair(
                &x_d,
                proj_mla(&mut qa_a2_d, &t0_d, &s0_d, &v0_d, k, n0, k2),
                proj_mla(&mut kv_a2_d, &t1_d, &s1_d, &v1_d, k, n1, k2),
                m,
                Dtype::BF16,
                Dtype::BF16,
                &mut ws_a2,
                Some(&xh0_d),
                Some(&xh1_d),
                &st,
            )
            .unwrap();
            let mut q_a2_d = DevBuf::zeroed(m * nq * 2).unwrap();
            let mut ws_q2 = DevBuf::zeroed(workspace_bytes(m, n0)).unwrap();
            rmsnorm_linear_bf16(
                &mut qa_a2_d,
                &qa_w_d,
                &tq_d,
                &sq_d,
                &vq_d,
                &mut q_a2_d,
                m,
                n0,
                nq,
                qb_k2,
                CB,
                eps,
                &mut ws_q2,
                &st,
            )
            .unwrap();
            let mut qa_b2_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            let mut kv_b2_d = DevBuf::zeroed(m * n1 * 2).unwrap();
            let mut q_b2_d = DevBuf::zeroed(m * nq * 2).unwrap();
            let mut ws_b2 = DevBuf::zeroed(pair_workspace_bytes(m, k, n0, n1)).unwrap();
            let mut qb_xh2_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            mla_qkv_qb_fused(
                &x_d,
                Dtype::BF16,
                None,
                None,
                None,
                None,
                proj_mla(&mut qa_b2_d, &t0_d, &s0_d, &v0_d, k, n0, k2),
                proj_mla(&mut kv_b2_d, &t1_d, &s1_d, &v1_d, k, n1, k2),
                &qa_w_d,
                proj_mla(&mut q_b2_d, &tq_d, &sq_d, &vq_d, n0, nq, qb_k2),
                m,
                eps,
                &mut ws_b2,
                Some(&xh0_d),
                Some(&xh1_d),
                &mut qb_xh2_d,
                None,
                &st,
            )
            .unwrap();
            st.sync().unwrap();
            let tag = format!("prerot m={m} trial={trial}");
            let ok = check_eq(&format!("{tag} q_a"), &dl::<u16>(&qa_a2_d, &st), &dl::<u16>(&qa_b2_d, &st))
                & check_eq(&format!("{tag} kv"), &dl::<u16>(&kv_a2_d, &st), &dl::<u16>(&kv_b2_d, &st))
                & check_eq(&format!("{tag} q"), &dl::<u16>(&q_a2_d, &st), &dl::<u16>(&q_b2_d, &st));
            assert!(ok, "{tag}: fused MLA qkv+qb diverged (prerot)");
            eprintln!("{tag}: identical");

            // Route A3/B3: the fused input norm mode (rmsnorm, no residual).
            let mut norm_a_d = DevBuf::zeroed(m * k * 2).unwrap();
            unsafe { glue::rmsnorm(x_d.ptr(), w_d.ptr(), norm_a_d.ptr(), m, k, eps, &st) }.unwrap();
            let mut qa_a3_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            let mut kv_a3_d = DevBuf::zeroed(m * n1 * 2).unwrap();
            let mut ws_a3 = DevBuf::zeroed(pair_workspace_bytes(m, k, n0, n1)).unwrap();
            fused_decode_pair(
                &norm_a_d,
                proj_mla(&mut qa_a3_d, &t0_d, &s0_d, &v0_d, k, n0, k2),
                proj_mla(&mut kv_a3_d, &t1_d, &s1_d, &v1_d, k, n1, k2),
                m,
                Dtype::BF16,
                Dtype::BF16,
                &mut ws_a3,
                None,
                None,
                &st,
            )
            .unwrap();
            let mut q_a3_d = DevBuf::zeroed(m * nq * 2).unwrap();
            let mut ws_q3 = DevBuf::zeroed(workspace_bytes(m, n0)).unwrap();
            rmsnorm_linear_bf16(
                &mut qa_a3_d,
                &qa_w_d,
                &tq_d,
                &sq_d,
                &vq_d,
                &mut q_a3_d,
                m,
                n0,
                nq,
                qb_k2,
                CB,
                eps,
                &mut ws_q3,
                &st,
            )
            .unwrap();
            let mut norm_b_d = DevBuf::zeroed(m * k * 2).unwrap();
            let mut qa_b3_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            let mut kv_b3_d = DevBuf::zeroed(m * n1 * 2).unwrap();
            let mut q_b3_d = DevBuf::zeroed(m * nq * 2).unwrap();
            let mut ws_b3 = DevBuf::zeroed(pair_workspace_bytes(m, k, n0, n1)).unwrap();
            let mut qb_xh3_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            mla_qkv_qb_fused(
                &x_d,
                Dtype::BF16,
                None,
                Some(&w_d),
                None,
                Some(&mut norm_b_d),
                proj_mla(&mut qa_b3_d, &t0_d, &s0_d, &v0_d, k, n0, k2),
                proj_mla(&mut kv_b3_d, &t1_d, &s1_d, &v1_d, k, n1, k2),
                &qa_w_d,
                proj_mla(&mut q_b3_d, &tq_d, &sq_d, &vq_d, n0, nq, qb_k2),
                m,
                eps,
                &mut ws_b3,
                None,
                None,
                &mut qb_xh3_d,
                None,
                &st,
            )
            .unwrap();
            st.sync().unwrap();
            let tag = format!("norm m={m} trial={trial}");
            let ok = check_eq(&format!("{tag} norm"), &dl::<u16>(&norm_a_d, &st), &dl::<u16>(&norm_b_d, &st))
                & check_eq(&format!("{tag} q_a"), &dl::<u16>(&qa_a3_d, &st), &dl::<u16>(&qa_b3_d, &st))
                & check_eq(&format!("{tag} kv"), &dl::<u16>(&kv_a3_d, &st), &dl::<u16>(&kv_b3_d, &st))
                & check_eq(&format!("{tag} q"), &dl::<u16>(&q_a3_d, &st), &dl::<u16>(&q_b3_d, &st));
            assert!(ok, "{tag}: fused MLA qkv+qb diverged (norm)");
            eprintln!("{tag}: identical");
        }
    }
}

// Temporary diagnostic: synthetic prefill A/B — fmoe::prefill + glue::add_rmsnorm vs
// ling_fmoe::prefill_with_epilogue (fused combine). Must be bit-identical.
// Temporary diagnostic: synthetic prefill A/B — fmoe::prefill + glue::add_rmsnorm vs
// ling_fmoe::prefill_with_epilogue (fused combine). Must be bit-identical.
// Temporary diagnostic: synthetic prefill A/B — fmoe::prefill + glue::add_rmsnorm vs
// ling_fmoe::prefill_with_epilogue (fused combine). Must be bit-identical.
// The fused MLA epilogue inside the pair kernel's last CTA (pair_cnt) vs the same fused call with the separate
// `ling_mla_rot_out_qnorm_qb_rot_in_kernel` launch: every output (q_a, kv_a, q_b) must be 0-bit-diff, in the
// plain, prerot and input-norm modes.
#[test]
#[ignore]
fn mla_qkv_qb_fused_pair_cnt_matches_separate() {
    let Some(st) = gpu() else { return };
    let (k, k2, qb_k2) = (K_MLA, 10usize, 7usize);
    let (n0, n1, nq) = (N0_MLA, N1_MLA, 12288usize);
    let eps = 1e-6f32;
    let nw0 = nw_for(k, n0, k2);
    let nw1 = nw_for(k, n1, k2);
    let nwq = nw_for(n0, nq, qb_k2);
    let mut r = Rng(0x8b44_9b84 ^ k as u64 ^ (n0 as u64) << 20 ^ 0x5b84);
    for &m in &[1usize, 8] {
        for trial in 0..2 {
            let x: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let resv: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let w_norm: Vec<f32> = (0..k).map(|_| r.uniform(0.5, 1.5)).collect();
            let t0: Vec<u32> = (0..nw0).map(|_| r.next_u64() as u32).collect();
            let s0: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v0: Vec<u16> = (0..n0).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let t1: Vec<u32> = (0..nw1).map(|_| r.next_u64() as u32).collect();
            let s1: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v1: Vec<u16> = (0..n1).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let tq: Vec<u32> = (0..nwq).map(|_| r.next_u64() as u32).collect();
            let sq: Vec<u16> = (0..n0).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let vq: Vec<u16> = (0..nq).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let qa_w: Vec<f32> = (0..n0).map(|_| r.uniform(0.5, 1.5)).collect();

            let x_d = DevBuf::from_slice(&x).unwrap();
            let res_d = DevBuf::from_slice(&resv).unwrap();
            let w_d = DevBuf::from_slice(&w_norm).unwrap();
            let t0_d = DevBuf::from_slice(&t0).unwrap();
            let s0_d = DevBuf::from_slice(&s0).unwrap();
            let v0_d = DevBuf::from_slice(&v0).unwrap();
            let t1_d = DevBuf::from_slice(&t1).unwrap();
            let s1_d = DevBuf::from_slice(&s1).unwrap();
            let v1_d = DevBuf::from_slice(&v1).unwrap();
            let tq_d = DevBuf::from_slice(&tq).unwrap();
            let sq_d = DevBuf::from_slice(&sq).unwrap();
            let vq_d = DevBuf::from_slice(&vq).unwrap();
            let qa_w_d = DevBuf::from_slice(&qa_w).unwrap();
            let pair_cnt_d = DevBuf::zeroed(132).unwrap();
            let pair_cnt_b = DevBuf::zeroed(132).unwrap();

            // Route A: the fused call with the separate epilogue launch (pair_cnt = None).
            let mut qa_a_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            let mut kv_a_d = DevBuf::zeroed(m * n1 * 2).unwrap();
            let mut q_a_d = DevBuf::zeroed(m * nq * 2).unwrap();
            let mut ws_a = DevBuf::zeroed(pair_workspace_bytes(m, k, n0, n1)).unwrap();
            let mut qb_xh_a_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            mla_qkv_qb_fused(
                &x_d,
                Dtype::BF16,
                None,
                None,
                None,
                None,
                proj_mla(&mut qa_a_d, &t0_d, &s0_d, &v0_d, k, n0, k2),
                proj_mla(&mut kv_a_d, &t1_d, &s1_d, &v1_d, k, n1, k2),
                &qa_w_d,
                proj_mla(&mut q_a_d, &tq_d, &sq_d, &vq_d, n0, nq, qb_k2),
                m,
                eps,
                &mut ws_a,
                None,
                None,
                &mut qb_xh_a_d,
                None,
                &st,
            )
            .unwrap();

            // Route B: the same call with the pair-kernel fused epilogue (pair_cnt).
            let mut qa_b_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            let mut kv_b_d = DevBuf::zeroed(m * n1 * 2).unwrap();
            let mut q_b_d = DevBuf::zeroed(m * nq * 2).unwrap();
            let mut ws_b = DevBuf::zeroed(pair_workspace_bytes(m, k, n0, n1)).unwrap();
            let mut qb_xh_b_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            mla_qkv_qb_fused(
                &x_d,
                Dtype::BF16,
                None,
                None,
                None,
                None,
                proj_mla(&mut qa_b_d, &t0_d, &s0_d, &v0_d, k, n0, k2),
                proj_mla(&mut kv_b_d, &t1_d, &s1_d, &v1_d, k, n1, k2),
                &qa_w_d,
                proj_mla(&mut q_b_d, &tq_d, &sq_d, &vq_d, n0, nq, qb_k2),
                m,
                eps,
                &mut ws_b,
                None,
                None,
                &mut qb_xh_b_d,
                Some(&pair_cnt_b),
                &st,
            )
            .unwrap();
            st.sync().unwrap();
            let tag = format!("pair_cnt m={m} trial={trial}");
            let ok = check_eq(&format!("{tag} q_a"), &dl::<u16>(&qa_a_d, &st), &dl::<u16>(&qa_b_d, &st))
                & check_eq(&format!("{tag} kv"), &dl::<u16>(&kv_a_d, &st), &dl::<u16>(&kv_b_d, &st))
                & check_eq(&format!("{tag} q"), &dl::<u16>(&q_a_d, &st), &dl::<u16>(&q_b_d, &st));
            assert!(ok, "{tag}: pair-kernel MLA epilogue diverged");
            eprintln!("{tag}: identical");

            // prerot mode
            let mut xh0_d = DevBuf::zeroed(m * k * 2).unwrap();
            let mut xh1_d = DevBuf::zeroed(m * k * 2).unwrap();
            unsafe extern "C" {
                fn gb_exl3_rotate_in(
                    x: *const core::ffi::c_void,
                    x_dtype: i32,
                    suh: *const core::ffi::c_void,
                    xh: *mut core::ffi::c_void,
                    m: i32,
                    k: i32,
                    stream: *mut core::ffi::c_void,
                ) -> i32;
            }
            unsafe {
                assert_eq!(gb_exl3_rotate_in(x_d.ptr(), 1, s0_d.ptr(), xh0_d.ptr(), m as i32, k as i32, st.raw()), 0);
                assert_eq!(gb_exl3_rotate_in(x_d.ptr(), 1, s1_d.ptr(), xh1_d.ptr(), m as i32, k as i32, st.raw()), 0);
            }
            let mut qa_a2_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            let mut kv_a2_d = DevBuf::zeroed(m * n1 * 2).unwrap();
            let mut q_a2_d = DevBuf::zeroed(m * nq * 2).unwrap();
            let mut ws_a2 = DevBuf::zeroed(pair_workspace_bytes(m, k, n0, n1)).unwrap();
            let mut qb_xh_a2_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            mla_qkv_qb_fused(
                &x_d,
                Dtype::BF16,
                None,
                None,
                None,
                None,
                proj_mla(&mut qa_a2_d, &t0_d, &s0_d, &v0_d, k, n0, k2),
                proj_mla(&mut kv_a2_d, &t1_d, &s1_d, &v1_d, k, n1, k2),
                &qa_w_d,
                proj_mla(&mut q_a2_d, &tq_d, &sq_d, &vq_d, n0, nq, qb_k2),
                m,
                eps,
                &mut ws_a2,
                Some(&xh0_d),
                Some(&xh1_d),
                &mut qb_xh_a2_d,
                None,
                &st,
            )
            .unwrap();
            let mut qa_b2_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            let mut kv_b2_d = DevBuf::zeroed(m * n1 * 2).unwrap();
            let mut q_b2_d = DevBuf::zeroed(m * nq * 2).unwrap();
            let mut ws_b2 = DevBuf::zeroed(pair_workspace_bytes(m, k, n0, n1)).unwrap();
            let mut qb_xh_b2_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            mla_qkv_qb_fused(
                &x_d,
                Dtype::BF16,
                None,
                None,
                None,
                None,
                proj_mla(&mut qa_b2_d, &t0_d, &s0_d, &v0_d, k, n0, k2),
                proj_mla(&mut kv_b2_d, &t1_d, &s1_d, &v1_d, k, n1, k2),
                &qa_w_d,
                proj_mla(&mut q_b2_d, &tq_d, &sq_d, &vq_d, n0, nq, qb_k2),
                m,
                eps,
                &mut ws_b2,
                Some(&xh0_d),
                Some(&xh1_d),
                &mut qb_xh_b2_d,
                Some(&pair_cnt_b),
                &st,
            )
            .unwrap();
            st.sync().unwrap();
            let tag = format!("pair_cnt prerot m={m} trial={trial}");
            let ok = check_eq(&format!("{tag} q_a"), &dl::<u16>(&qa_a2_d, &st), &dl::<u16>(&qa_b2_d, &st))
                & check_eq(&format!("{tag} kv"), &dl::<u16>(&kv_a2_d, &st), &dl::<u16>(&kv_b2_d, &st))
                & check_eq(&format!("{tag} q"), &dl::<u16>(&q_a2_d, &st), &dl::<u16>(&q_b2_d, &st));
            assert!(ok, "{tag}: pair-kernel MLA epilogue diverged (prerot)");
            eprintln!("{tag}: identical");

            // input-norm mode
            let mut norm_a_d = DevBuf::zeroed(m * k * 2).unwrap();
            unsafe { glue::rmsnorm(x_d.ptr(), w_d.ptr(), norm_a_d.ptr(), m, k, eps, &st) }.unwrap();
            let mut qa_a3_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            let mut kv_a3_d = DevBuf::zeroed(m * n1 * 2).unwrap();
            let mut q_a3_d = DevBuf::zeroed(m * nq * 2).unwrap();
            let mut ws_a3 = DevBuf::zeroed(pair_workspace_bytes(m, k, n0, n1)).unwrap();
            let mut qb_xh_a3_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            mla_qkv_qb_fused(
                &x_d,
                Dtype::BF16,
                None,
                Some(&w_d),
                None,
                Some(&mut norm_a_d),
                proj_mla(&mut qa_a3_d, &t0_d, &s0_d, &v0_d, k, n0, k2),
                proj_mla(&mut kv_a3_d, &t1_d, &s1_d, &v1_d, k, n1, k2),
                &qa_w_d,
                proj_mla(&mut q_a3_d, &tq_d, &sq_d, &vq_d, n0, nq, qb_k2),
                m,
                eps,
                &mut ws_a3,
                None,
                None,
                &mut qb_xh_a3_d,
                None,
                &st,
            )
            .unwrap();
            let mut norm_b_d = DevBuf::zeroed(m * k * 2).unwrap();
            let mut qa_b3_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            let mut kv_b3_d = DevBuf::zeroed(m * n1 * 2).unwrap();
            let mut q_b3_d = DevBuf::zeroed(m * nq * 2).unwrap();
            let mut ws_b3 = DevBuf::zeroed(pair_workspace_bytes(m, k, n0, n1)).unwrap();
            let mut qb_xh_b3_d = DevBuf::zeroed(m * n0 * 2).unwrap();
            mla_qkv_qb_fused(
                &x_d,
                Dtype::BF16,
                None,
                Some(&w_d),
                None,
                Some(&mut norm_b_d),
                proj_mla(&mut qa_b3_d, &t0_d, &s0_d, &v0_d, k, n0, k2),
                proj_mla(&mut kv_b3_d, &t1_d, &s1_d, &v1_d, k, n1, k2),
                &qa_w_d,
                proj_mla(&mut q_b3_d, &tq_d, &sq_d, &vq_d, n0, nq, qb_k2),
                m,
                eps,
                &mut ws_b3,
                None,
                None,
                &mut qb_xh_b3_d,
                Some(&pair_cnt_b),
                &st,
            )
            .unwrap();
            st.sync().unwrap();
            let tag = format!("pair_cnt norm m={m} trial={trial}");
            let ok = check_eq(&format!("{tag} norm"), &dl::<u16>(&norm_a_d, &st), &dl::<u16>(&norm_b_d, &st))
                & check_eq(&format!("{tag} q_a"), &dl::<u16>(&qa_a3_d, &st), &dl::<u16>(&qa_b3_d, &st))
                & check_eq(&format!("{tag} kv"), &dl::<u16>(&kv_a3_d, &st), &dl::<u16>(&kv_b3_d, &st))
                & check_eq(&format!("{tag} q"), &dl::<u16>(&q_a3_d, &st), &dl::<u16>(&q_b3_d, &st));
            assert!(ok, "{tag}: pair-kernel MLA epilogue diverged (norm)");
            eprintln!("{tag}: identical");
        }
    }
}

// The dense projection's input rotation fused into the expand+gate epilogue (`gb_ling_mla_expand_gate_tc`
// with dense_suh/dense_xh) vs the separate `gb_exl3_rotate_in` on the gated output: gated bytes AND the
// rotated dense input must be bit-identical at every decode row count.
#[test]
#[ignore]
fn dense_rot_in_fused_expand_gate_matches_separate() {
    let Some(st) = gpu() else { return };
    const NH: usize = 64;
    const DC: usize = 512;
    const DN: usize = 128;
    const HIDDEN: usize = 4096;
    const DENSE_K: usize = NH * DN;
    let mut r = Rng(0x39c4_5b84);
    for &rows in &[1usize, 4, 8] {
        for trial in 0..2 {
            let o_lat: Vec<u16> = (0..rows * NH * DC).map(|_| bf(r.uniform(-0.25, 0.25))).collect();
            let x: Vec<u16> = (0..rows * HIDDEN).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let uvt: Vec<u16> = (0..NH * DC * DN).map(|_| bf(r.uniform(-0.25, 0.25))).collect();
            let g: Vec<u16> = (0..NH * HIDDEN).map(|_| bf(r.uniform(-0.25, 0.25))).collect();
            let suh: Vec<u16> = (0..DENSE_K).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();

            let o_lat_d = DevBuf::from_slice(&o_lat).unwrap();
            let x_d = DevBuf::from_slice(&x).unwrap();
            let uvt_d = DevBuf::from_slice(&uvt).unwrap();
            let g_d = DevBuf::from_slice(&g).unwrap();
            let suh_d = DevBuf::from_slice(&suh).unwrap();
            let w = gb_cuda::mla_gated::MlaGatedWeights {
                uk: uvt_d.subview(0, 64 * 128 * 512 * 2),
                uvt: uvt_d.subview(0, uvt_d.bytes()),
                g: g_d.subview(0, g_d.bytes()),
                dtype: Dtype::BF16,
            };

            // Route A: expand_gated, then the separate rotate_in of the gated output.
            let mut gated_a_d = DevBuf::zeroed(rows * DENSE_K * 2).unwrap();
            tc::expand_gated(&o_lat_d, &x_d, &w, &mut gated_a_d, rows, None, None, &st).unwrap();
            let mut dense_xh_a_d = DevBuf::zeroed(rows * DENSE_K * 2).unwrap();
            unsafe extern "C" {
                fn gb_exl3_rotate_in(
                    x: *const core::ffi::c_void,
                    x_dtype: i32,
                    suh: *const core::ffi::c_void,
                    xh: *mut core::ffi::c_void,
                    m: i32,
                    k: i32,
                    stream: *mut core::ffi::c_void,
                ) -> i32;
            }
            unsafe {
                assert_eq!(
                    gb_exl3_rotate_in(gated_a_d.ptr(), 1, suh_d.ptr(), dense_xh_a_d.ptr(), rows as i32, DENSE_K as i32, st.raw(),),
                    0
                );
            }

            // Route B: the fused expand+gate with the dense rotation inside.
            let mut gated_b_d = DevBuf::zeroed(rows * DENSE_K * 2).unwrap();
            let mut dense_xh_b_d = DevBuf::zeroed(rows * DENSE_K * 2).unwrap();
            tc::expand_gated(&o_lat_d, &x_d, &w, &mut gated_b_d, rows, Some(&suh_d), Some(&mut dense_xh_b_d), &st).unwrap();

            st.sync().unwrap();
            let tag = format!("rows={rows} trial={trial}");
            let ok = check_eq(&format!("{tag} gated"), &dl::<u16>(&gated_a_d, &st), &dl::<u16>(&gated_b_d, &st))
                & check_eq(&format!("{tag} dense_xh"), &dl::<u16>(&dense_xh_a_d, &st), &dl::<u16>(&dense_xh_b_d, &st));
            assert!(ok, "{tag}: fused dense rot_in diverged");
            eprintln!("{tag}: identical");
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion 1: `window_and_norm` with the KDA o_proj's input rotation fused into the norm epilogue
// (`gb_ling_kda_window_and_norm` with `o_suh`/`o_xh`) vs `gb_kda_window` + `gb_kda_norm` + the separate
// `gb_exl3_rotate_in` of the gated output. The o_proj's rotated input bytes must equal the separate launch's.
// ---------------------------------------------------------------------------
#[test]
fn ling_kda_window_and_norm_prerot_matches_separate() {
    let Some(st) = gpu() else { return };
    let mut r = Rng(0x5b84_2612);
    const D: usize = 128;

    unsafe extern "C" {
        fn gb_kda_window(
            q: *const f32,
            k: *const f32,
            v: *const core::ffi::c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            parity: *const i32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            heads: i32,
            rz: i32,
            write_final: i32,
            r_step: i32,
            y: *mut core::ffi::c_void,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        fn gb_kda_norm(
            y: *const core::ffi::c_void,
            z: *const f32,
            w: *const core::ffi::c_void,
            eps: f32,
            rows: i32,
            heads: i32,
            out: *mut core::ffi::c_void,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        fn gb_ling_kda_window_and_norm(
            q: *const f32,
            k: *const f32,
            v: *const core::ffi::c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            parity: *const i32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            heads: i32,
            rz: i32,
            r_step: i32,
            y: *mut core::ffi::c_void,
            head_cnt: *mut i32,
            eps: f32,
            z: *const f32,
            norm_w: *const core::ffi::c_void,
            out: *mut core::ffi::c_void,
            o_suh: *const core::ffi::c_void,
            o_xh: *mut core::ffi::c_void,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        fn gb_exl3_rotate_in(
            x: *const core::ffi::c_void,
            x_dtype: i32,
            suh: *const core::ffi::c_void,
            xh: *mut core::ffi::c_void,
            m: i32,
            k: i32,
            stream: *mut core::ffi::c_void,
        ) -> i32;
    }

    let eps = 1e-5f32;
    let combos: Vec<(usize, usize, usize, usize)> = vec![(8, 8, 1, 8), (64, 8, 1, 4), (64, 8, 2, 8), (8, 4, 2, 8)];
    let mut head_cnt: Option<DevBuf> = None;
    for (heads0, r_step, n_seq0, rows0) in combos {
        let (heads, n_seq, rows) = (heads0 as i32, n_seq0 as i32, rows0 as i32);
        let lens: Vec<i32> = {
            let mut l = Vec::with_capacity(n_seq as usize);
            let mut rem = rows;
            for i in 0..n_seq as usize {
                let take = if i + 1 == n_seq as usize { rem } else { (rows / n_seq).max(1) };
                l.push(take);
                rem -= take;
            }
            l
        };
        let mut seq_start: Vec<i32> = vec![0];
        for &l in &lens {
            let last = *seq_start.last().unwrap();
            seq_start.push(last + l);
        }
        let slot: Vec<i32> = (0..n_seq0).map(|i| i as i32).collect();
        let seq_len = n_seq0 * 2 * heads0 * D * D;
        let row_len = rows0 * heads0 * D;

        let s_host: Vec<f32> = (0..seq_len).map(|_| r.uniform(-0.5, 0.5)).collect();
        let parity_host: Vec<i32> = (0..n_seq0).map(|i| (i % 2) as i32).collect();
        let q_host: Vec<f32> = (0..row_len).map(|_| r.uniform(-0.5, 0.5)).collect();
        let k_host: Vec<f32> = (0..row_len).map(|_| r.uniform(-0.5, 0.5)).collect();
        let v_host: Vec<u16> = (0..row_len).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
        let g_host: Vec<f32> = (0..row_len).map(|_| r.uniform(-0.5, 0.5)).collect();
        let beta_host: Vec<f32> = (0..rows0 * heads0).map(|_| r.uniform(-0.5, 0.5)).collect();
        let z_host: Vec<f32> = (0..row_len).map(|_| r.uniform(-0.5, 0.5)).collect();
        let norm_w_host: Vec<u16> = (0..D).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
        // the o_proj's suh (one per column of the K = heads * D row)
        let suh_host: Vec<u16> = (0..heads0 * D).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();

        let mk = || {
            (
                DevBuf::from_slice(&s_host).unwrap(),
                DevBuf::from_slice(&parity_host).unwrap(),
                DevBuf::from_slice(&q_host).unwrap(),
                DevBuf::from_slice(&k_host).unwrap(),
                DevBuf::from_slice(&v_host).unwrap(),
                DevBuf::from_slice(&g_host).unwrap(),
                DevBuf::from_slice(&beta_host).unwrap(),
                DevBuf::from_slice(&z_host).unwrap(),
                DevBuf::from_slice(&norm_w_host).unwrap(),
                DevBuf::from_slice(&seq_start).unwrap(),
                DevBuf::from_slice(&slot).unwrap(),
            )
        };
        let (s_a, p_a, q_a, k_a, v_a, g_a, b_a, z_a, w_a, ss_a, sl_a) = mk();
        let (s_b, p_b, q_b, k_b, v_b, g_b, b_b, z_b, w_b, ss_b, sl_b) = mk();
        let suh_d = DevBuf::from_slice(&suh_host).unwrap();

        let y_bytes = rows0 * heads0 * D * 2;
        let mut y_a = DevBuf::zeroed(y_bytes).unwrap();
        let mut y_b = DevBuf::zeroed(y_bytes).unwrap();
        let mut out_a = DevBuf::zeroed(y_bytes).unwrap();
        let mut out_b = DevBuf::zeroed(y_bytes).unwrap();
        let mut xh_ref = DevBuf::zeroed(y_bytes).unwrap();
        let mut xh_fused = DevBuf::zeroed(y_bytes).unwrap();

        if head_cnt.is_none() {
            head_cnt = Some(DevBuf::zeroed(n_seq0 * heads0 * 4).unwrap());
        }

        // reference: window + norm, then the separate o_proj input rotation of the gated output
        let rc = unsafe {
            gb_kda_window(
                q_a.ptr() as *const f32,
                k_a.ptr() as *const f32,
                v_a.ptr(),
                g_a.ptr() as *const f32,
                b_a.ptr() as *const f32,
                s_a.ptr() as *mut f32,
                p_a.ptr() as *const i32,
                ss_a.ptr() as *const i32,
                sl_a.ptr() as *const i32,
                n_seq,
                heads,
                0,
                1,
                r_step as i32,
                y_a.ptr(),
                st.raw(),
            )
        };
        assert_eq!(rc, 0, "separate window rc={rc} (r_step={r_step} heads={heads} n_seq={n_seq} rows={rows})");
        let rc = unsafe { gb_kda_norm(y_a.ptr(), z_a.ptr() as *const f32, w_a.ptr(), eps, rows, heads, out_a.ptr(), st.raw()) };
        assert_eq!(rc, 0, "separate norm rc={rc}");
        let rc = unsafe {
            gb_exl3_rotate_in(
                out_a.ptr(),
                0, // F16
                suh_d.ptr(),
                xh_ref.ptr(),
                rows,
                (heads * D as i32),
                st.raw(),
            )
        };
        assert_eq!(rc, 0, "separate rotate_in rc={rc}");

        // fused: window + norm + the o_proj rotation in one launch
        let hc = head_cnt.as_ref().unwrap();
        let rc = unsafe {
            gb_ling_kda_window_and_norm(
                q_b.ptr() as *const f32,
                k_b.ptr() as *const f32,
                v_b.ptr(),
                g_b.ptr() as *const f32,
                b_b.ptr() as *const f32,
                s_b.ptr() as *mut f32,
                p_b.ptr() as *const i32,
                ss_b.ptr() as *const i32,
                sl_b.ptr() as *const i32,
                n_seq,
                heads,
                0,
                r_step as i32,
                y_b.ptr(),
                hc.ptr() as *mut i32,
                eps,
                z_b.ptr() as *const f32,
                w_b.ptr(),
                out_b.ptr(),
                suh_d.ptr(),
                xh_fused.ptr(),
                st.raw(),
            )
        };
        assert_eq!(rc, 0, "fused window+norm+prerot rc={rc}");

        st.sync().unwrap();
        let tag = format!("r_step={r_step} heads={heads} n_seq={n_seq} rows={rows}");
        let mut ya: Vec<u16> = vec![0; y_bytes / 2];
        let mut yb: Vec<u16> = vec![0; y_bytes / 2];
        y_a.download(&mut ya).unwrap();
        y_b.download(&mut yb).unwrap();
        let mut sa: Vec<f32> = vec![0.0; seq_len];
        let mut sb: Vec<f32> = vec![0.0; seq_len];
        s_a.download(&mut sa).unwrap();
        s_b.download(&mut sb).unwrap();
        let mut oa: Vec<u16> = vec![0; y_bytes / 2];
        let mut ob: Vec<u16> = vec![0; y_bytes / 2];
        out_a.download(&mut oa).unwrap();
        out_b.download(&mut ob).unwrap();
        let mut xra: Vec<u16> = vec![0; y_bytes / 2];
        let mut xrb: Vec<u16> = vec![0; y_bytes / 2];
        xh_ref.download(&mut xra).unwrap();
        xh_fused.download(&mut xrb).unwrap();
        assert_eq!(diff_count(&ya, &yb), 0, "y diff ({tag})");
        assert_eq!(diff_count_f32(&sa, &sb), 0, "S diff ({tag})");
        assert_eq!(diff_count(&oa, &ob), 0, "norm out diff ({tag})");
        assert_eq!(diff_count(&xra, &xrb), 0, "prerot xh diff ({tag})");
    }
}

// ---------------------------------------------------------------------------
// Fusion 2 (pair): `linear_add_rmsnorm` with the next sub-layer's rot_in_pair fused into its epilogue
// (`Prerot::pair`), then the pair run on the pre-rotated inputs vs the separate composition: linear_add_rmsnorm
// (no prerot) + fused_decode_pair rotating the norm itself. The k.o-shaped linear (k = 8192, f16 in) followed
// by a dense gate/up-shaped pair (k = 4096, bf16 in).
// ---------------------------------------------------------------------------
#[test]
#[ignore]
fn linear_add_rmsnorm_prerot_pair_matches_separate() {
    let Some(st) = gpu() else { return };
    let lin_cnt = DevBuf::zeroed(1024).unwrap();

    let k = 8192usize; // k.o width (f16 gated)
    let n = 4096usize; // dim
    let k2 = 10usize; // k.o k2 (5-bit)
    let pk = 4096usize; // gate/up input width (the norm)
    let pn = 10240usize; // gate/up output width
    let pk2 = 6usize; // gate/up k2 (3-bit)
    let cb = 2i32;
    let eps = 1e-6f32;
    let nw = nw_for(k, n, k2);
    let pnw = nw_for(pk, pn, pk2);
    let mut r = Rng(0x5b84_2621);
    for &m in &[1usize, 2, 4, 8] {
        for trial in 0..3 {
            let xs: Vec<u16> = (0..m * k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let t: Vec<u32> = (0..nw).map(|_| r.next_u64() as u32).collect();
            let s: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v: Vec<u16> = (0..n).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let hidden: Vec<u16> = (0..m * n).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let w: Vec<f32> = (0..n).map(|_| r.uniform(0.5, 1.5)).collect();
            let gt0: Vec<u32> = (0..pnw).map(|_| r.next_u64() as u32).collect();
            let gs0: Vec<u16> = (0..pk).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let gv0: Vec<u16> = (0..pn).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let gt1: Vec<u32> = (0..pnw).map(|_| r.next_u64() as u32).collect();
            let gs1: Vec<u16> = (0..pk).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let gv1: Vec<u16> = (0..pn).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();

            let x_d = DevBuf::from_slice(&xs).unwrap();
            let t_d = DevBuf::from_slice(&t).unwrap();
            let s_d = DevBuf::from_slice(&s).unwrap();
            let v_d = DevBuf::from_slice(&v).unwrap();
            let hidden_d = DevBuf::from_slice(&hidden).unwrap();
            let w_d = DevBuf::from_slice(&w).unwrap();
            let gt0_d = DevBuf::from_slice(&gt0).unwrap();
            let gs0_d = DevBuf::from_slice(&gs0).unwrap();
            let gv0_d = DevBuf::from_slice(&gv0).unwrap();
            let gt1_d = DevBuf::from_slice(&gt1).unwrap();
            let gs1_d = DevBuf::from_slice(&gs1).unwrap();
            let gv1_d = DevBuf::from_slice(&gv1).unwrap();

            let ws_bytes = linear_workspace_bytes(m, k, n, k2 / 2);
            let ws_pair_bytes = pair_workspace_bytes(m, pk, pn, pn);

            // ---- route A: separate composition (reference) ----
            let mut ws_a = DevBuf::zeroed(ws_bytes).unwrap();
            let mut res_d = DevBuf::zeroed(m * n * 2).unwrap();
            Exl3Weights::k2_forward(
                &x_d,
                &mut res_d,
                m,
                Dtype::F16,
                Dtype::BF16,
                k,
                n,
                k2,
                gb_exl3::Codebook::Mul1,
                &t_d,
                &s_d,
                &v_d,
                None,
                &mut ws_a,
                &st,
            )
            .unwrap();
            let mut eh_d = DevBuf::from_slice(&hidden).unwrap();
            let mut en_d = DevBuf::zeroed(m * n * 2).unwrap();
            unsafe {
                gb_cuda::glue::add_rmsnorm(
                    hidden_d.ptr().cast_const(),
                    res_d.ptr().cast_const(),
                    w_d.ptr().cast_const(),
                    eh_d.ptr(),
                    en_d.ptr(),
                    m,
                    n,
                    eps,
                    &st,
                )
                .unwrap();
            }
            let mut ws_pair_a = DevBuf::zeroed(ws_pair_bytes).unwrap();
            let mut gag_d = DevBuf::zeroed(m * pn * 2).unwrap();
            let mut gbg_d = DevBuf::zeroed(m * pn * 2).unwrap();
            fused_decode_pair(
                &en_d,
                Proj { trellis: &gt0_d, suh: &gs0_d, svh: &gv0_d, y: &mut gag_d, k: pk, n: pn, k2: pk2, codebook: cb },
                Proj { trellis: &gt1_d, suh: &gs1_d, svh: &gv1_d, y: &mut gbg_d, k: pk, n: pn, k2: pk2, codebook: cb },
                m,
                Dtype::BF16,
                Dtype::BF16,
                &mut ws_pair_a,
                None,
                None,
                &st,
            )
            .unwrap();

            // ---- route B: fused linear epilogue writes the pair's rotated inputs; the pair skips its rotation ----
            let mut ws_b = DevBuf::zeroed(ws_bytes).unwrap();
            let mut gh_d = DevBuf::from_slice(&hidden).unwrap();
            let mut gn_d = DevBuf::zeroed(m * n * 2).unwrap();
            let rh = m * pk * 2;
            let mut prv = DevBuf::zeroed_view(2 * rh).unwrap();
            let mut pr_xh0 = prv.subview(0, rh);
            let mut pr_xh1 = prv.subview(rh, rh);
            let pr = Prerot::pair(&gs0_d, &gs1_d, &mut pr_xh0, &mut pr_xh1);
            linear_add_rmsnorm(
                &x_d,
                Dtype::F16,
                &t_d,
                &s_d,
                &v_d,
                &hidden_d,
                &w_d,
                &mut gh_d,
                &mut gn_d,
                m,
                k,
                n,
                k2,
                cb,
                n,
                eps,
                &mut ws_b,
                false,
                &pr,
                Some(&lin_cnt),
                &st,
            )
            .unwrap();
            let mut ws_pair_b = DevBuf::zeroed(ws_pair_bytes).unwrap();
            let mut gcg_d = DevBuf::zeroed(m * pn * 2).unwrap();
            let mut gdg_d = DevBuf::zeroed(m * pn * 2).unwrap();
            let xh0 = prv.subview(0, rh);
            let xh1 = prv.subview(rh, rh);
            fused_decode_pair(
                &gn_d,
                Proj { trellis: &gt0_d, suh: &gs0_d, svh: &gv0_d, y: &mut gcg_d, k: pk, n: pn, k2: pk2, codebook: cb },
                Proj { trellis: &gt1_d, suh: &gs1_d, svh: &gv1_d, y: &mut gdg_d, k: pk, n: pn, k2: pk2, codebook: cb },
                m,
                Dtype::BF16,
                Dtype::BF16,
                &mut ws_pair_b,
                Some(&xh0),
                Some(&xh1),
                &st,
            )
            .unwrap();

            st.sync().unwrap();
            let tag = format!("m={m} trial={trial}");
            assert!(
                check_eq(&format!("{tag} hidden"), &dl::<u16>(&gh_d, &st), &dl::<u16>(&eh_d, &st)),
                "prerot pair: hidden differs at {tag}"
            );
            assert!(
                check_eq(&format!("{tag} norm"), &dl::<u16>(&gn_d, &st), &dl::<u16>(&en_d, &st)),
                "prerot pair: norm differs at {tag}"
            );
            assert!(
                check_eq(&format!("{tag} gate"), &dl::<u16>(&gcg_d, &st), &dl::<u16>(&gag_d, &st)),
                "prerot pair: gate differs at {tag}"
            );
            assert!(
                check_eq(&format!("{tag} up"), &dl::<u16>(&gdg_d, &st), &dl::<u16>(&gbg_d, &st)),
                "prerot pair: up differs at {tag}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion 2 (triple): `silu_mul_linear_add_rmsnorm` with the next layer's rot_in_triple fused into its epilogue
// (`Prerot::triple`), then the triple run on the pre-rotated inputs vs the separate composition:
// swiglu + silu_mul_linear_add_rmsnorm (no prerot) + fused_decode_triple rotating the norm itself. The dense
// down_proj shape (k = 10240, bf16 in) followed by a KDA qkv/f/g-shaped triple (k = 4096).
// ---------------------------------------------------------------------------
#[test]
#[ignore]
fn silu_mul_linear_add_rmsnorm_prerot_triple_matches_separate() {
    let Some(st) = gpu() else { return };
    let lin_cnt = DevBuf::zeroed(1024).unwrap();

    let k = 10240usize; // down_proj width (bf16 g/u)
    let n = 4096usize; // dim
    let k2 = 7usize; // down_proj k2 (3.5-bit)
    let cb = 2i32;
    let tk = 4096usize; // triple input width (the norm)
    let tn0 = 12288usize; // qkv out (SK == 1 at k = 4096)
    let tn12 = 8192usize; // f/g out
    let tk2_0 = 7usize;
    let tk2_12 = 6usize;
    let eps = 1e-6f32;
    let nw = nw_for(k, n, k2);
    let tnw = nw_for(tk, tn0, tk2_0);
    let mut r = Rng(0x5b84_2622);
    for &m in &[1usize, 2, 4, 8] {
        for trial in 0..3 {
            let g: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let u: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let t: Vec<u32> = (0..nw).map(|_| r.next_u64() as u32).collect();
            let s: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v: Vec<u16> = (0..n).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let hidden: Vec<u16> = (0..m * n).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let w: Vec<f32> = (0..n).map(|_| r.uniform(0.5, 1.5)).collect();
            let tt0: Vec<u32> = (0..tnw).map(|_| r.next_u64() as u32).collect();
            let ts0: Vec<u16> = (0..tk).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let tv0: Vec<u16> = (0..tn0).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let tt1: Vec<u32> = (0..tnw).map(|_| r.next_u64() as u32).collect();
            let ts1: Vec<u16> = (0..tk).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let tv1: Vec<u16> = (0..tn12).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let tt2: Vec<u32> = (0..tnw).map(|_| r.next_u64() as u32).collect();
            let ts2: Vec<u16> = (0..tk).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let tv2: Vec<u16> = (0..tn12).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();

            let g_d = DevBuf::from_slice(&g).unwrap();
            let u_d = DevBuf::from_slice(&u).unwrap();
            let t_d = DevBuf::from_slice(&t).unwrap();
            let s_d = DevBuf::from_slice(&s).unwrap();
            let v_d = DevBuf::from_slice(&v).unwrap();
            let hidden_d = DevBuf::from_slice(&hidden).unwrap();
            let w_d = DevBuf::from_slice(&w).unwrap();
            let tt0_d = DevBuf::from_slice(&tt0).unwrap();
            let ts0_d = DevBuf::from_slice(&ts0).unwrap();
            let tv0_d = DevBuf::from_slice(&tv0).unwrap();
            let tt1_d = DevBuf::from_slice(&tt1).unwrap();
            let ts1_d = DevBuf::from_slice(&ts1).unwrap();
            let tv1_d = DevBuf::from_slice(&tv1).unwrap();
            let tt2_d = DevBuf::from_slice(&tt2).unwrap();
            let ts2_d = DevBuf::from_slice(&ts2).unwrap();
            let tv2_d = DevBuf::from_slice(&tv2).unwrap();

            let ws_bytes = linear_workspace_bytes(m, k, n, k2 / 2);
            let ws_t_bytes = workspace_bytes(m, tk);

            // ---- route A: separate composition (reference) ----
            let mut ws_a = DevBuf::zeroed(ws_bytes).unwrap();
            let mut eh_d = DevBuf::from_slice(&hidden).unwrap();
            let mut en_d = DevBuf::zeroed(m * n * 2).unwrap();
            silu_mul_linear_add_rmsnorm(
                &g_d,
                &u_d,
                &t_d,
                &s_d,
                &v_d,
                &hidden_d,
                &w_d,
                &mut eh_d,
                &mut en_d,
                m,
                k,
                n,
                k2,
                cb,
                n,
                eps,
                &mut ws_a,
                &Prerot::default(),
                Some(&lin_cnt),
                &st,
            )
            .unwrap();
            let mut ws_t_a = DevBuf::zeroed(ws_t_bytes).unwrap();
            let mut qkv_a_d = DevBuf::zeroed(m * tn0 * 4).unwrap();
            let mut f_a_d = DevBuf::zeroed(m * tn12 * 2).unwrap();
            let mut g_a_d = DevBuf::zeroed(m * tn12 * 2).unwrap();
            fused_decode_triple(
                &en_d,
                Proj { trellis: &tt0_d, suh: &ts0_d, svh: &tv0_d, y: &mut qkv_a_d, k: tk, n: tn0, k2: tk2_0, codebook: cb },
                Proj { trellis: &tt1_d, suh: &ts1_d, svh: &tv1_d, y: &mut f_a_d, k: tk, n: tn12, k2: tk2_12, codebook: cb },
                Proj { trellis: &tt2_d, suh: &ts2_d, svh: &tv2_d, y: &mut g_a_d, k: tk, n: tn12, k2: tk2_12, codebook: cb },
                m,
                Dtype::BF16,
                Dtype::F32,
                Dtype::F16,
                &mut ws_t_a,
                None,
                None,
                None,
                &st,
            )
            .unwrap();

            // ---- route B: fused down epilogue writes the triple's rotated inputs; the triple skips its rotation ----
            let mut ws_b = DevBuf::zeroed(ws_bytes).unwrap();
            let mut gh_d = DevBuf::from_slice(&hidden).unwrap();
            let mut gn_d = DevBuf::zeroed(m * n * 2).unwrap();
            let rh = m * tk * 2;
            let mut prv = DevBuf::zeroed_view(3 * rh).unwrap();
            let mut pr_xh0 = prv.subview(0, rh);
            let mut pr_xh1 = prv.subview(rh, rh);
            let mut pr_xh2 = prv.subview(2 * rh, rh);
            let pr = Prerot::triple(&ts0_d, &ts1_d, &ts2_d, &mut pr_xh0, &mut pr_xh1, &mut pr_xh2);
            silu_mul_linear_add_rmsnorm(
                &g_d,
                &u_d,
                &t_d,
                &s_d,
                &v_d,
                &hidden_d,
                &w_d,
                &mut gh_d,
                &mut gn_d,
                m,
                k,
                n,
                k2,
                cb,
                n,
                eps,
                &mut ws_b,
                &pr,
                Some(&lin_cnt),
                &st,
            )
            .unwrap();
            let mut ws_t_b = DevBuf::zeroed(ws_t_bytes).unwrap();
            let mut qkv_b_d = DevBuf::zeroed(m * tn0 * 4).unwrap();
            let mut f_b_d = DevBuf::zeroed(m * tn12 * 2).unwrap();
            let mut g_b_d = DevBuf::zeroed(m * tn12 * 2).unwrap();
            let xh0 = prv.subview(0, rh);
            let xh1 = prv.subview(rh, rh);
            let xh2 = prv.subview(2 * rh, rh);
            fused_decode_triple(
                &gn_d,
                Proj { trellis: &tt0_d, suh: &ts0_d, svh: &tv0_d, y: &mut qkv_b_d, k: tk, n: tn0, k2: tk2_0, codebook: cb },
                Proj { trellis: &tt1_d, suh: &ts1_d, svh: &tv1_d, y: &mut f_b_d, k: tk, n: tn12, k2: tk2_12, codebook: cb },
                Proj { trellis: &tt2_d, suh: &ts2_d, svh: &tv2_d, y: &mut g_b_d, k: tk, n: tn12, k2: tk2_12, codebook: cb },
                m,
                Dtype::BF16,
                Dtype::F32,
                Dtype::F16,
                &mut ws_t_b,
                Some(&xh0),
                Some(&xh1),
                Some(&xh2),
                &st,
            )
            .unwrap();

            st.sync().unwrap();
            let tag = format!("m={m} trial={trial}");
            assert!(
                check_eq(&format!("{tag} hidden"), &dl::<u16>(&gh_d, &st), &dl::<u16>(&eh_d, &st)),
                "prerot triple: hidden differs at {tag}"
            );
            assert!(
                check_eq(&format!("{tag} norm"), &dl::<u16>(&gn_d, &st), &dl::<u16>(&en_d, &st)),
                "prerot triple: norm differs at {tag}"
            );
            assert!(
                check_eq(&format!("{tag} qkv"), &dl::<f32>(&qkv_b_d, &st), &dl::<f32>(&qkv_a_d, &st)),
                "prerot triple: qkv differs at {tag}"
            );
            assert!(
                check_eq(&format!("{tag} f"), &dl::<u16>(&f_b_d, &st), &dl::<u16>(&f_a_d, &st)),
                "prerot triple: f differs at {tag}"
            );
            assert!(
                check_eq(&format!("{tag} g"), &dl::<u16>(&g_b_d, &st), &dl::<u16>(&g_a_d, &st)),
                "prerot triple: g differs at {tag}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion 1: `attn_combine_kernel` folded into `attn_split_kernel` (the fused `tc::attend_split` against the
// pre-fusion two-launch reference `tc::attend_split_ref`, which still runs split-writes-parts + the separate
// combine kernel). Covers single-chunk rows (pos < 1024, direct inline combine), multi-chunk rows (pos >= 1024,
// the last-CTA fold) and padding rows (pos < 0, chunk-0 zeroing), mixed in one call.
// ---------------------------------------------------------------------------
const SPLIT_TEST_PAGES: usize = 12; // 12 x 256 = 3072 keys = 3 chunks
const SPLIT_TEST_CW: usize = 576;
const SPLIT_TEST_KEYS: usize = SPLIT_TEST_PAGES * 256;

fn attend_split_fused_combine_matches_separate_for(seed: u64) {
    let Some(st) = gpu() else { return };
    let mut rng = Rng(seed);
    let mut host_cache = vec![0u16; SPLIT_TEST_KEYS * SPLIT_TEST_CW];
    for k in 0..SPLIT_TEST_KEYS {
        let sc = 0.2 + 2.4 * rng.uniform(0.0, 1.0);
        let row: Vec<u16> = (0..SPLIT_TEST_CW).map(|_| bf(rng.uniform(-1.0, 1.0) * sc)).collect();
        host_cache[k * SPLIT_TEST_CW..(k + 1) * SPLIT_TEST_CW].copy_from_slice(&row);
    }
    let mut cache = MlaGatedCache::new(SPLIT_TEST_PAGES, Dtype::BF16).unwrap();
    cache.buf.upload(&host_cache).unwrap();
    let n = 8usize;
    let pos: Vec<i32> = vec![50, 2500, -1, 1024, 2047, 0, 1234, -1];
    let slots: Vec<i32> = vec![0; n];
    let ql: Vec<u16> = (0..n * mla_gated::N_HEADS * mla_gated::LATENT_DIM).map(|_| bf(rng.uniform(-4.0, 4.0))).collect();
    let qr: Vec<u16> = (0..n * mla_gated::N_HEADS * mla_gated::ROPE_DIM).map(|_| bf(rng.uniform(-4.0, 4.0))).collect();
    let q_lat = DevBuf::from_slice(&ql).unwrap();
    let q_rot = DevBuf::from_slice(&qr).unwrap();
    let fwd: Vec<u32> = (0..SPLIT_TEST_PAGES as u32).collect();
    let table = BlockTable::from_host(&vec![fwd], SPLIT_TEST_PAGES).unwrap();
    let ri = RowIndex::from_host(&slots, &pos).unwrap();
    let mut ws = DevBuf::zeroed(tc::SPLIT_WS_BYTES).unwrap();
    // run the fused path first, then the reference on the same workspace (the self-resetting counters must be
    // reusable across calls, so the reference run after a fused run also checks the reset)
    let mut o_a = DevBuf::alloc(n * mla_gated::N_HEADS * mla_gated::LATENT_DIM * 2).unwrap();
    let mut o_b = DevBuf::alloc(n * mla_gated::N_HEADS * mla_gated::LATENT_DIM * 2).unwrap();
    tc::attend_split(&q_lat, &q_rot, &cache, &table, &ri, &mut o_a, &mut ws, n, &st).unwrap();
    tc::attend_split_ref(&q_lat, &q_rot, &cache, &table, &ri, &mut o_b, &mut ws, n, &st).unwrap();
    tc::attend_split(&q_lat, &q_rot, &cache, &table, &ri, &mut o_a, &mut ws, n, &st).unwrap();
    st.sync().unwrap();
    let a = o_a.to_vec().unwrap();
    let b = o_b.to_vec().unwrap();
    assert!(
        check_eq("attend_split fused vs ref", &a, &b),
        "fused attn_split (inline + last-CTA combine) diverged from the two-launch reference"
    );
    let mut diffs = 0usize;
    for i in 0..n {
        let row_diffs = diff_count(&a[i * 32768..(i + 1) * 32768], &b[i * 32768..(i + 1) * 32768]);
        eprintln!("attend_split fused vs ref: row {i} pos {} bits {row_diffs}", pos[i]);
        diffs += row_diffs;
    }
    assert_eq!(diffs, 0, "total differing bits across rows");
}

#[test]
#[ignore]
fn attend_split_fused_combine_matches_separate() {
    attend_split_fused_combine_matches_separate_for(0x51a7_5b84);
}

// ---------------------------------------------------------------------------
// Fusion 2: `norm_rot_in_single` + `linear_prerot` (the last layer's final add_rmsnorm fused with the lm_head's
// input rotation, then the head linear on the pre-rotated input) against `glue::add_rmsnorm` +
// `Exl3Weights::k2_forward` (the head shape: k = dim = 4096, k2 = 12, codebook 2, f16 out).
// ---------------------------------------------------------------------------
fn norm_rot_in_single_plus_linear_prerot_case(m: usize, trial: usize, r: &mut Rng, st: &Stream) {
    let k = 4096usize;
    let n = 8192usize;
    let k2 = 12usize;
    let cb = 2i32;
    let eps = 1e-6f32;
    let nw = nw_for(k, n, k2);
    let x: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
    let resv: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
    let w_norm: Vec<f32> = (0..k).map(|_| r.uniform(0.5, 1.5)).collect();
    let t: Vec<u32> = (0..nw).map(|_| r.next_u64() as u32).collect();
    let s: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let v: Vec<u16> = (0..n).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let x_d = DevBuf::from_slice(&x).unwrap();
    let res_d = DevBuf::from_slice(&resv).unwrap();
    let w_d = DevBuf::from_slice(&w_norm).unwrap();
    let t_d = DevBuf::from_slice(&t).unwrap();
    let s_d = DevBuf::from_slice(&s).unwrap();
    let v_d = DevBuf::from_slice(&v).unwrap();

    // route A: glue::add_rmsnorm + k2_forward (the reference)
    let mut hidden_a_d = DevBuf::from_slice(&x).unwrap();
    let mut norm_a_d = DevBuf::zeroed(m * k * 2).unwrap();
    unsafe { glue::add_rmsnorm(x_d.ptr(), res_d.ptr(), w_d.ptr(), hidden_a_d.ptr(), norm_a_d.ptr(), m, k, eps, st) }.unwrap();
    let mut y_a_d = DevBuf::zeroed(m * n * 2).unwrap();
    let mut ws_a = DevBuf::zeroed(linear_workspace_bytes(m, k, n, k2 / 2)).unwrap();
    Exl3Weights::k2_forward(
        &norm_a_d,
        &mut y_a_d,
        m,
        Dtype::BF16,
        Dtype::F16,
        k,
        n,
        k2,
        gb_exl3::Codebook::Mul1,
        &t_d,
        &s_d,
        &v_d,
        None,
        &mut ws_a,
        st,
    )
    .unwrap();

    // route B: fused norm+rot single + prerot linear
    let mut hidden_b_d = DevBuf::from_slice(&x).unwrap();
    let mut norm_b_d = DevBuf::zeroed(m * k * 2).unwrap();
    let mut xh_d = DevBuf::zeroed(m * k * 2).unwrap();
    let mut y_b_d = DevBuf::zeroed(m * n * 2).unwrap();
    let mut ws_b = DevBuf::zeroed(workspace_bytes(m, k)).unwrap();
    norm_rot_in_single(&x_d, Some(&res_d), &w_d, Some(&mut hidden_b_d), &mut norm_b_d, &s_d, &mut xh_d, m, k, eps, st).unwrap();
    linear_prerot(&t_d, &v_d, None, &mut y_b_d, Dtype::F16, m, k, n, k2, cb, &xh_d, &mut ws_b, st).unwrap();

    let tag = format!("m={m} t{trial}");
    let ok = check_eq(&format!("{tag} hidden"), &dl::<u16>(&hidden_a_d, st), &dl::<u16>(&hidden_b_d, st))
        & check_eq(&format!("{tag} norm"), &dl::<u16>(&norm_a_d, st), &dl::<u16>(&norm_b_d, st))
        & check_eq(&format!("{tag} logits"), &dl::<u16>(&y_a_d, st), &dl::<u16>(&y_b_d, st));
    assert!(ok, "{tag}: fused head norm+rot+linear diverged");
    eprintln!("{tag}: identical");
}

#[test]
#[ignore]
fn norm_rot_in_single_plus_linear_prerot_matches_separate() {
    let Some(st) = gpu() else { return };
    let mut r = Rng(0x3c91_5b84);
    for &m in &[1usize, 8] {
        for trial in 0..2 {
            norm_rot_in_single_plus_linear_prerot_case(m, trial, &mut r, &st);
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion 3: `gb_ling_topk_f16` (stage 2 fused into the last stage-1 CTA) against `glm_glue::topk_f16`.
// ---------------------------------------------------------------------------
fn ling_topk_f16_matches_glm_for(rows: usize, stride: usize, valid: usize, count: usize, seed: u64) {
    let Some(st) = gpu() else { return };
    let mut rng = Rng(seed ^ (rows as u64) << 20 ^ (valid as u64) << 8);
    let raw: Vec<f32> = (0..rows * stride).map(|_| rng.uniform(-8.0, 8.0)).collect();
    let logits: Vec<u16> = raw.iter().map(|&v| f16(v as f64)).collect();
    let logits_d = DevBuf::from_slice(&logits).unwrap();
    let mut ids_a = DevBuf::zeroed(rows * count * 4).unwrap();
    let mut vals_a = DevBuf::zeroed(rows * count * 4).unwrap();
    let mut ids_b = DevBuf::zeroed(rows * count * 4).unwrap();
    let mut vals_b = DevBuf::zeroed(rows * count * 4).unwrap();
    let mut ws_a = DevBuf::zeroed(gb_cuda::glm_glue::topk_workspace_bytes(rows, valid, count)).unwrap();
    let mut ws_b = DevBuf::zeroed(gb_cuda::ling_kda_proj_fused::topk_workspace_bytes(rows, valid, count)).unwrap();
    gb_cuda::glm_glue::topk_f16(&logits_d, rows, stride, valid, count, &mut ids_a, &mut vals_a, &mut ws_a, &st).unwrap();
    gb_cuda::ling_kda_proj_fused::topk_f16(&logits_d, rows, stride, valid, count, &mut ids_b, &mut vals_b, &mut ws_b, &st)
        .unwrap();
    gb_cuda::ling_kda_proj_fused::topk_f16(&logits_d, rows, stride, valid, count, &mut ids_b, &mut vals_b, &mut ws_b, &st)
        .unwrap();
    st.sync().unwrap();
    let ia: Vec<i32> = ids_a.to_vec().unwrap();
    let ib: Vec<i32> = ids_b.to_vec().unwrap();
    let va: Vec<f32> = vals_a.to_vec().unwrap();
    let vb: Vec<f32> = vals_b.to_vec().unwrap();
    let tag = format!("rows={rows} valid={valid} count={count}");
    assert!(check_eq(&format!("{tag} ids"), &ia, &ib), "{tag}: ling topk ids diverged");
    assert!(check_eq(&format!("{tag} vals"), &va, &vb), "{tag}: ling topk vals diverged");
    eprintln!("{tag}: identical");
}

#[test]
#[ignore]
fn ling_topk_f16_matches_glm_topk_f16() {
    ling_topk_f16_matches_glm_for(1, 6000, 5000, 16, 0xbeef);
    ling_topk_f16_matches_glm_for(3, 6000, 5000, 64, 0xbeef);
    ling_topk_f16_matches_glm_for(8, 22000, 20000, 16, 0xf00d);
    ling_topk_f16_matches_glm_for(8, 22000, 20000, 64, 0xf00d);
    ling_topk_f16_matches_glm_for(1, 22000, 20000, 32, 0xcafe);
    // the model's exact shapes: vocab-wide logits (157184 columns = 10 chunks), count 8 (P = 16)
    ling_topk_f16_matches_glm_for(3, 157184, 157184, 8, 0x41a7);
    ling_topk_f16_matches_glm_for(8, 157184, 157184, 8, 0x41a7);
    ling_topk_f16_matches_glm_for(64, 157184, 157184, 8, 0x41a7);
    ling_topk_f16_matches_glm_for(1, 157184, 157184, 2, 0x51a7);
    ling_topk_f16_matches_glm_for(8, 157184, 157184, 2, 0x51a7);
}

// The workspace is reused across calls with VARYING rows/count on one buffer: the stage-1 candidate pools must never
// land in the counter region (fixed 256 bytes at offset 0), so a later call (e.g. rows = 1 / count = 2 after a
// rows = 3 / count = 8 verify window) still sees zeroed counters. The buffer is zeroed once at allocation; each call
// self-resets whatever counters it touched.
#[test]
#[ignore]
fn ling_topk_f16_reuse_across_varying_rows_counts() {
    let Some(st) = gpu() else { return };
    let mut rng = Rng(0x51a7_5b84);
    let stride = 157184usize;
    let valid = 157184usize;
    let logits: Vec<u16> = (0..8 * stride).map(|_| f16(rng.uniform(-8.0, 8.0) as f64)).collect();
    let logits_d = DevBuf::from_slice(&logits).unwrap();
    // one buffer, sized for the largest call, zeroed ONCE
    let ws_bytes = topk_workspace_bytes(8, valid, 8)
        .max(topk_workspace_bytes(3, valid, 8))
        .max(topk_workspace_bytes(1, valid, 8))
        .max(topk_workspace_bytes(1, valid, 2));
    let mut ws = DevBuf::zeroed(ws_bytes).unwrap();
    for &(rows, count) in &[(3usize, 8usize), (1, 8), (1, 2), (8, 8), (3, 2)] {
        let mut ids_a = DevBuf::zeroed(rows * count * 4).unwrap();
        let mut vals_a = DevBuf::zeroed(rows * count * 4).unwrap();
        let mut ids_b = DevBuf::zeroed(rows * count * 4).unwrap();
        let mut vals_b = DevBuf::zeroed(rows * count * 4).unwrap();
        let mut ws_g = DevBuf::zeroed(gb_cuda::glm_glue::topk_workspace_bytes(rows, valid, count)).unwrap();
        gb_cuda::glm_glue::topk_f16(&logits_d, rows, stride, valid, count, &mut ids_a, &mut vals_a, &mut ws_g, &st).unwrap();
        gb_cuda::ling_kda_proj_fused::topk_f16(&logits_d, rows, stride, valid, count, &mut ids_b, &mut vals_b, &mut ws, &st)
            .unwrap();
        st.sync().unwrap();
        let ia: Vec<i32> = ids_a.to_vec().unwrap();
        let ib: Vec<i32> = ids_b.to_vec().unwrap();
        let va: Vec<f32> = vals_a.to_vec().unwrap();
        let vb: Vec<f32> = vals_b.to_vec().unwrap();
        let tag = format!("reuse rows={rows} count={count}");
        assert!(check_eq(&format!("{tag} ids"), &ia, &ib), "{tag}: ling topk ids diverged");
        assert!(check_eq(&format!("{tag} vals"), &va, &vb), "{tag}: ling topk vals diverged");
        eprintln!("{tag}: identical");
    }
}

// ---------------------------------------------------------------------------
// Fusion 1: the dense gate/up pair's epilogue fuses the following down_proj's input rotation (swiglu(g, u) with
// d.down.suh into d.down's rotated input), replacing the standalone `ling_silu_mul_rot_in_kernel` launch. Route A
// is the separate sequence (gate/up pair + glue::swiglu + linear_add_rmsnorm with its own rotation); route B is the
// fused pair (`fused_decode_pair_ext` with down_suh/down_xh/col_cnt) + `silu_mul_linear_add_rmsnorm_ext` with
// skip_rot_in = true. Every output byte (g, u, hidden_out, y) must be bit-identical.
// ---------------------------------------------------------------------------
#[test]
#[ignore]
fn dense_pair_silu_rot_fused_matches_separate() {
    let Some(st) = gpu() else { return };
    let lin_cnt = DevBuf::zeroed(1024).unwrap();

    let mut r = Rng(0x51a_5b84);
    let k = 2048usize; // gate/up input width (and the down linear's k is n)
    let n = 8192usize; // gate/up width == down input width (SK = 1: n / 128 = 64 blocks)
    let k2 = 6usize;
    let cb = 2i32;
    let dn = 2048usize; // down output width == dim (<= 4096)
    let k2_dn = 7usize;
    let eps = 1e-6f32;
    let nw = nw_for(k, n, k2);
    let nw_dn = nw_for(n, dn, k2_dn);
    for &m in &[1usize, 8] {
        for trial in 0..2 {
            let x: Vec<u16> = (0..m * k).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
            let t0: Vec<u32> = (0..nw).map(|_| r.next_u64() as u32).collect();
            let s0: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v0: Vec<u16> = (0..n).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let t1: Vec<u32> = (0..nw).map(|_| r.next_u64() as u32).collect();
            let s1: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let v1: Vec<u16> = (0..n).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let td: Vec<u32> = (0..nw_dn).map(|_| r.next_u64() as u32).collect();
            let sd: Vec<u16> = (0..n).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let vd: Vec<u16> = (0..dn).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
            let hidden: Vec<u16> = (0..m * dn).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
            let w_norm: Vec<f32> = (0..dn).map(|_| r.uniform(0.5, 1.5)).collect();

            let x_d = DevBuf::from_slice(&x).unwrap();
            let t0_d = DevBuf::from_slice(&t0).unwrap();
            let s0_d = DevBuf::from_slice(&s0).unwrap();
            let v0_d = DevBuf::from_slice(&v0).unwrap();
            let t1_d = DevBuf::from_slice(&t1).unwrap();
            let s1_d = DevBuf::from_slice(&s1).unwrap();
            let v1_d = DevBuf::from_slice(&v1).unwrap();
            let td_d = DevBuf::from_slice(&td).unwrap();
            let sd_d = DevBuf::from_slice(&sd).unwrap();
            let vd_d = DevBuf::from_slice(&vd).unwrap();
            let w_d = DevBuf::from_slice(&w_norm).unwrap();
            let hidden_a_d = DevBuf::from_slice(&hidden).unwrap();
            let hidden_b_d = DevBuf::from_slice(&hidden).unwrap();

            // route A: gate/up pair + swiglu + linear_add_rmsnorm (its own rotation)
            let mut g_a_d = DevBuf::zeroed(m * n * 2).unwrap();
            let mut u_a_d = DevBuf::zeroed(m * n * 2).unwrap();
            let ws_a = linear_workspace_bytes(m, k, n, k2 / 2).max(linear_workspace_bytes(m, n, dn, k2_dn / 2));
            let mut ws_a_d = DevBuf::zeroed(ws_a).unwrap();
            fused_decode_pair(
                &x_d,
                Proj { trellis: &t0_d, suh: &s0_d, svh: &v0_d, y: &mut g_a_d, k, n, k2, codebook: cb },
                Proj { trellis: &t1_d, suh: &s1_d, svh: &v1_d, y: &mut u_a_d, k, n, k2, codebook: cb },
                m,
                Dtype::BF16,
                Dtype::BF16,
                &mut ws_a_d,
                None,
                None,
                &st,
            )
            .unwrap();
            let mut a_d = DevBuf::zeroed(m * n * 2).unwrap();
            unsafe { glue::swiglu(g_a_d.ptr(), u_a_d.ptr(), a_d.ptr(), m * n, &st) }.unwrap();
            let mut out_hidden_a_d = DevBuf::from_slice(&hidden).unwrap();
            let mut y_a_d = DevBuf::zeroed(m * dn * 2).unwrap();
            linear_add_rmsnorm(
                &a_d,
                Dtype::BF16,
                &td_d,
                &sd_d,
                &vd_d,
                &hidden_a_d,
                &w_d,
                &mut out_hidden_a_d,
                &mut y_a_d,
                m,
                n,
                dn,
                k2_dn,
                cb,
                dn,
                eps,
                &mut ws_a_d,
                false,
                &Prerot::default(),
                Some(&lin_cnt),
                &st,
            )
            .unwrap();

            // route B: the pair's epilogue writes the rotated down input; the down linear skips its rotation
            let mut g_b_d = DevBuf::zeroed(m * n * 2).unwrap();
            let mut u_b_d = DevBuf::zeroed(m * n * 2).unwrap();
            let col_cnt = DevBuf::zeroed(n / 128 * 4).unwrap();
            let ws_pair = pair_workspace_bytes(m, k, n, n);
            let mut ws_pair_d = DevBuf::zeroed(ws_pair).unwrap();
            // the epilogue writes the rotated down input at offset 0 of the down linear's own workspace, exactly
            // like the model (down_xh = ws.subview(0, ...); the silu ext call reads it back from ws + 0)
            let ws_dn = linear_workspace_bytes(m, n, dn, k2_dn / 2);
            let mut ws_dn_d = DevBuf::zeroed_view(ws_dn).unwrap();
            let mut xh_d = ws_dn_d.subview(0, m * n * 2);
            fused_decode_pair_ext(
                &x_d,
                Proj { trellis: &t0_d, suh: &s0_d, svh: &v0_d, y: &mut g_b_d, k, n, k2, codebook: cb },
                Proj { trellis: &t1_d, suh: &s1_d, svh: &v1_d, y: &mut u_b_d, k, n, k2, codebook: cb },
                m,
                Dtype::BF16,
                Dtype::BF16,
                &mut ws_pair_d,
                None,
                None,
                Some(&sd_d),
                Some(&mut xh_d),
                Some(&col_cnt),
                &st,
            )
            .unwrap();
            let mut out_hidden_b_d = DevBuf::from_slice(&hidden).unwrap();
            let mut y_b_d = DevBuf::zeroed(m * dn * 2).unwrap();
            silu_mul_linear_add_rmsnorm_ext(
                &g_b_d,
                &u_b_d,
                &td_d,
                &sd_d,
                &vd_d,
                &hidden_b_d,
                &w_d,
                &mut out_hidden_b_d,
                &mut y_b_d,
                m,
                n,
                dn,
                k2_dn,
                cb,
                dn,
                eps,
                &mut ws_dn_d,
                &Prerot::default(),
                true,
                Some(&lin_cnt),
                &st,
            )
            .unwrap();

            st.sync().unwrap();
            let tag = format!("m={m} t{trial}");
            let ok = check_eq(&format!("{tag} g"), &dl::<u16>(&g_a_d, &st), &dl::<u16>(&g_b_d, &st))
                & check_eq(&format!("{tag} u"), &dl::<u16>(&u_a_d, &st), &dl::<u16>(&u_b_d, &st))
                & check_eq(&format!("{tag} hidden_out"), &dl::<u16>(&out_hidden_a_d, &st), &dl::<u16>(&out_hidden_b_d, &st))
                & check_eq(&format!("{tag} y"), &dl::<u16>(&y_a_d, &st), &dl::<u16>(&y_b_d, &st));
            assert!(ok, "{tag}: dense pair+silu-rot fusion diverged");
            eprintln!("{tag}: identical");
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion 2: the layer-0 fused norm+triple kernel replaces the standalone embed launch on decode/verify windows:
// it reads the embed rows straight off the table (with the exact `embed_vec_k` out-of-range zeroing), writes them to
// the hidden stream and normalises them. Route A is `glue::embed` + `glue::rmsnorm` + `fused_decode_triple`;
// route B is `norm_rot_in_triple_ext` with token_ids/embed_table/vocab. Every output byte (hidden, norm, qkv, f,
// g) must be bit-identical.
// ---------------------------------------------------------------------------
fn norm_triple_embed_case(dim: usize, m: usize, trial: usize, r: &mut Rng, st: &Stream) {
    let k = dim;
    let vocab = 1024usize;
    let n0 = 8192usize;
    let n12 = 8192usize;
    let k2_0 = 7usize;
    let k2_12 = 6usize;
    let cb = 2i32;
    let eps = 1e-6f32;
    let nw0 = nw_for(k, n0, k2_0);
    let nw12 = nw_for(k, n12, k2_12);
    let table: Vec<u16> = (0..vocab * dim).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
    // some ids out of range (negative / >= vocab) to exercise the zero-row path
    let ids: Vec<i32> = (0..m).map(|_| r.next_u64() as i32 % (vocab as i32 + 7) - 3).collect();
    let w_norm: Vec<f32> = (0..k).map(|_| r.uniform(0.5, 1.5)).collect();
    let t0: Vec<u32> = (0..nw0).map(|_| r.next_u64() as u32).collect();
    let s0: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let v0: Vec<u16> = (0..n0).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let t1: Vec<u32> = (0..nw12).map(|_| r.next_u64() as u32).collect();
    let s1: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let v1: Vec<u16> = (0..n12).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let t2: Vec<u32> = (0..nw12).map(|_| r.next_u64() as u32).collect();
    let s2: Vec<u16> = (0..k).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let v2: Vec<u16> = (0..n12).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();

    let table_d = DevBuf::from_slice(&table).unwrap();
    let ids_d = DevBuf::from_slice(&ids).unwrap();
    let w_d = DevBuf::from_slice(&w_norm).unwrap();
    let t0_d = DevBuf::from_slice(&t0).unwrap();
    let s0_d = DevBuf::from_slice(&s0).unwrap();
    let v0_d = DevBuf::from_slice(&v0).unwrap();
    let t1_d = DevBuf::from_slice(&t1).unwrap();
    let s1_d = DevBuf::from_slice(&s1).unwrap();
    let v1_d = DevBuf::from_slice(&v1).unwrap();
    let t2_d = DevBuf::from_slice(&t2).unwrap();
    let s2_d = DevBuf::from_slice(&s2).unwrap();
    let v2_d = DevBuf::from_slice(&v2).unwrap();

    // route A: embed + rmsnorm + triple
    let mut hidden_a_d = DevBuf::zeroed(m * dim * 2).unwrap();
    unsafe { glue::embed(table_d.ptr(), ids_d.ptr(), hidden_a_d.ptr(), m, dim, vocab, st) }.unwrap();
    let mut norm_a_d = DevBuf::zeroed(m * dim * 2).unwrap();
    unsafe { glue::rmsnorm(hidden_a_d.ptr(), w_d.ptr(), norm_a_d.ptr(), m, dim, eps, st) }.unwrap();
    let mut qkv_a_d = DevBuf::zeroed(m * n0 * 4).unwrap();
    let mut f_a_d = DevBuf::zeroed(m * n12 * 2).unwrap();
    let mut g_a_d = DevBuf::zeroed(m * n12 * 2).unwrap();
    let mut ws_a = DevBuf::zeroed(workspace_bytes(m, k)).unwrap();
    fused_decode_triple(
        &norm_a_d,
        Proj { trellis: &t0_d, suh: &s0_d, svh: &v0_d, y: &mut qkv_a_d, k, n: n0, k2: k2_0, codebook: cb },
        Proj { trellis: &t1_d, suh: &s1_d, svh: &v1_d, y: &mut f_a_d, k, n: n12, k2: k2_12, codebook: cb },
        Proj { trellis: &t2_d, suh: &s2_d, svh: &v2_d, y: &mut g_a_d, k, n: n12, k2: k2_12, codebook: cb },
        m,
        Dtype::BF16,
        Dtype::F32,
        Dtype::F16,
        &mut ws_a,
        None,
        None,
        None,
        st,
    )
    .unwrap();

    // route B: fused embed + norm + triple (x is not read; h_out = hidden)
    let mut hidden_b_d = DevBuf::zeroed(m * dim * 2).unwrap();
    let x_dummy = DevBuf::zeroed(m * dim * 2).unwrap();
    let mut norm_b_d = DevBuf::zeroed(m * dim * 2).unwrap();
    let mut qkv_b_d = DevBuf::zeroed(m * n0 * 4).unwrap();
    let mut f_b_d = DevBuf::zeroed(m * n12 * 2).unwrap();
    let mut g_b_d = DevBuf::zeroed(m * n12 * 2).unwrap();
    let mut ws_b = DevBuf::zeroed(workspace_bytes(m, k)).unwrap();
    norm_rot_in_triple_ext(
        &x_dummy,
        None,
        &w_d,
        Some(&mut hidden_b_d),
        &mut norm_b_d,
        Proj { trellis: &t0_d, suh: &s0_d, svh: &v0_d, y: &mut qkv_b_d, k, n: n0, k2: k2_0, codebook: cb },
        Proj { trellis: &t1_d, suh: &s1_d, svh: &v1_d, y: &mut f_b_d, k, n: n12, k2: k2_12, codebook: cb },
        Proj { trellis: &t2_d, suh: &s2_d, svh: &v2_d, y: &mut g_b_d, k, n: n12, k2: k2_12, codebook: cb },
        m,
        k,
        Dtype::F32,
        Dtype::F16,
        k2_0,
        k2_12,
        cb,
        eps,
        &mut ws_b,
        Some(&ids_d),
        Some(&table_d),
        vocab,
        st,
    )
    .unwrap();

    let tag = format!("m={m} t{trial}");
    let ok = check_eq(&format!("{tag} hidden"), &dl::<u16>(&hidden_a_d, st), &dl::<u16>(&hidden_b_d, st))
        & check_eq(&format!("{tag} norm"), &dl::<u16>(&norm_a_d, st), &dl::<u16>(&norm_b_d, st))
        & check_eq(&format!("{tag} qkv"), &dl::<f32>(&qkv_a_d, st), &dl::<f32>(&qkv_b_d, st))
        & check_eq(&format!("{tag} f"), &dl::<u16>(&f_a_d, st), &dl::<u16>(&f_b_d, st))
        & check_eq(&format!("{tag} g"), &dl::<u16>(&g_a_d, st), &dl::<u16>(&g_b_d, st));
    assert!(ok, "{tag}: fused embed+norm+triple diverged");
    eprintln!("{tag}: identical");
}

#[test]
#[ignore]
fn norm_rot_in_triple_embed_matches_separate() {
    let Some(st) = gpu() else { return };
    let mut r = Rng(0x7137_5b84);
    for &dim in &[2048usize, 4096] {
        for &m in &[1usize, 8] {
            for trial in 0..2 {
                norm_triple_embed_case(dim, m, trial, &mut r, &st);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion 3: the P2 prefill prep + beta + full_fg gate fused launch (`gb_ling_kda_prefill_prep_gate_fgg`) against
// the separate `gb_kda_gate_from_full_fg` + `gb_kda_prep` (in_f32 = 0) launches. Every output byte (g_decay, z,
// beta, q, k, v) must be bit-identical.
// ---------------------------------------------------------------------------
fn prefill_prep_gate_fused_matches_separate_for(
    heads: usize,
    rows: usize,
    hidden: usize,
    trial: usize,
    r: &mut Rng,
    st: &Stream,
) {
    const D: usize = 128;
    unsafe extern "C" {
        fn gb_kda_gate_from_full_fg(
            f_out: *const core::ffi::c_void,
            g_out: *const core::ffi::c_void,
            x: *const core::ffi::c_void,
            b_weight: *const core::ffi::c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            lower: f32,
            g_decay_out: *mut f32,
            z_gate_out: *mut f32,
            beta_out: *mut f32,
            rows: i32,
            hidden: i32,
            heads: i32,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        fn gb_kda_prep(
            qkv: *const core::ffi::c_void,
            in_f32: i32,
            conv_w: *const core::ffi::c_void,
            conv_pool: *const core::ffi::c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            rows: i32,
            heads: i32,
            emit_q: i32,
            skip_full: i32,
            raw_out: *mut core::ffi::c_void,
            q: *mut f32,
            k: *mut f32,
            v: *mut core::ffi::c_void,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        fn gb_ling_kda_prefill_prep_gate_fgg(
            qkv: *const core::ffi::c_void,
            x: *const core::ffi::c_void,
            b_weight: *const core::ffi::c_void,
            conv_w: *const core::ffi::c_void,
            conv_pool: *const core::ffi::c_void,
            seq_start: *const i32,
            slot: *const i32,
            f_out: *const core::ffi::c_void,
            g_out: *const core::ffi::c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            lower: f32,
            n_seq: i32,
            rows: i32,
            heads: i32,
            hidden: i32,
            raw_out: *mut core::ffi::c_void,
            q: *mut f32,
            k: *mut f32,
            v: *mut core::ffi::c_void,
            g_decay: *mut f32,
            z_gate: *mut f32,
            beta: *mut f32,
            stream: *mut core::ffi::c_void,
        ) -> i32;
    }
    let conv_dim = 3 * heads * D;
    let c = heads * D;
    let qkv: Vec<u16> = (0..rows * conv_dim).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
    let conv_w: Vec<u16> = (0..conv_dim * 4).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
    let conv_pool: Vec<u16> = (0..3 * conv_dim).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
    let x: Vec<u16> = (0..rows * hidden).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
    let b_weight: Vec<u16> = (0..heads * hidden).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
    let f_out: Vec<u16> = (0..rows * c).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let g_out: Vec<u16> = (0..rows * c).map(|_| f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let a_log: Vec<f32> = (0..heads).map(|_| r.uniform(-2.0, 0.0)).collect();
    let dt_bias: Vec<f32> = (0..c).map(|_| r.uniform(-1.0, 1.0)).collect();
    let seq_start: Vec<i32> = vec![0, rows as i32];
    let slot: Vec<i32> = vec![0];

    let qkv_d = DevBuf::from_slice(&qkv).unwrap();
    let conv_w_d = DevBuf::from_slice(&conv_w).unwrap();
    let conv_pool_d = DevBuf::from_slice(&conv_pool).unwrap();
    let x_d = DevBuf::from_slice(&x).unwrap();
    let b_d = DevBuf::from_slice(&b_weight).unwrap();
    let f_d = DevBuf::from_slice(&f_out).unwrap();
    let g_d = DevBuf::from_slice(&g_out).unwrap();
    let a_d = DevBuf::from_slice(&a_log).unwrap();
    let dt_d = DevBuf::from_slice(&dt_bias).unwrap();
    let ss_d = DevBuf::from_slice(&seq_start).unwrap();
    let sl_d = DevBuf::from_slice(&slot).unwrap();

    let mk = || {
        (
            DevBuf::zeroed(rows * c * 4).unwrap(),         // g_decay
            DevBuf::zeroed(rows * c * 4).unwrap(),         // z
            DevBuf::zeroed(rows * heads * 4).unwrap(),     // beta
            DevBuf::zeroed(rows * heads * D * 4).unwrap(), // q
            DevBuf::zeroed(rows * heads * D * 4).unwrap(), // k
            DevBuf::zeroed(rows * heads * D * 2).unwrap(), // v
        )
    };
    let (gd_a, z_a, be_a, q_a, k_a, v_a) = mk();
    let (gd_b, z_b, be_b, q_b, k_b, v_b) = mk();
    let lower = -5.0f32;

    let rc = unsafe {
        gb_kda_gate_from_full_fg(
            f_d.ptr(),
            g_d.ptr(),
            x_d.ptr(),
            b_d.ptr(),
            a_d.ptr() as *const f32,
            dt_d.ptr() as *const f32,
            lower,
            gd_a.ptr() as *mut f32,
            z_a.ptr() as *mut f32,
            be_a.ptr() as *mut f32,
            rows as i32,
            hidden as i32,
            heads as i32,
            st.raw(),
        )
    };
    assert_eq!(rc, 0, "separate gate rc={rc}");
    let rc = unsafe {
        gb_kda_prep(
            qkv_d.ptr(),
            0,
            conv_w_d.ptr(),
            conv_pool_d.ptr(),
            ss_d.ptr() as *const i32,
            sl_d.ptr() as *const i32,
            core::ptr::null(),
            1,
            rows as i32,
            heads as i32,
            1,
            0,
            core::ptr::null_mut(),
            q_a.ptr() as *mut f32,
            k_a.ptr() as *mut f32,
            v_a.ptr(),
            st.raw(),
        )
    };
    assert_eq!(rc, 0, "separate prep rc={rc}");

    let rc = unsafe {
        gb_ling_kda_prefill_prep_gate_fgg(
            qkv_d.ptr(),
            x_d.ptr(),
            b_d.ptr(),
            conv_w_d.ptr(),
            conv_pool_d.ptr(),
            ss_d.ptr() as *const i32,
            sl_d.ptr() as *const i32,
            f_d.ptr(),
            g_d.ptr(),
            a_d.ptr() as *const f32,
            dt_d.ptr() as *const f32,
            lower,
            1,
            rows as i32,
            heads as i32,
            hidden as i32,
            core::ptr::null_mut(),
            q_b.ptr() as *mut f32,
            k_b.ptr() as *mut f32,
            v_b.ptr(),
            gd_b.ptr() as *mut f32,
            z_b.ptr() as *mut f32,
            be_b.ptr() as *mut f32,
            st.raw(),
        )
    };
    assert_eq!(rc, 0, "fused prep+gate rc={rc}");

    st.sync().unwrap();
    let tag = format!("heads={heads} rows={rows} t{trial}");
    let ok = check_eq(&format!("{tag} g_decay"), &dl::<f32>(&gd_a, st), &dl::<f32>(&gd_b, st))
        & check_eq(&format!("{tag} z"), &dl::<f32>(&z_a, st), &dl::<f32>(&z_b, st))
        & check_eq(&format!("{tag} beta"), &dl::<f32>(&be_a, st), &dl::<f32>(&be_b, st))
        & check_eq(&format!("{tag} q"), &dl::<f32>(&q_a, st), &dl::<f32>(&q_b, st))
        & check_eq(&format!("{tag} k"), &dl::<f32>(&k_a, st), &dl::<f32>(&k_b, st))
        & check_eq(&format!("{tag} v"), &dl::<u16>(&v_a, st), &dl::<u16>(&v_b, st));
    assert!(ok, "{tag}: fused prep+beta+gate diverged");
    eprintln!("{tag}: identical");
}

#[test]
#[ignore]
fn prefill_prep_gate_fused_matches_separate() {
    let Some(st) = gpu() else { return };
    let mut r = Rng(0x8a3_5b84);
    for &heads in &[8usize, 64] {
        for &rows in &[80usize, 256, 512, 1296] {
            for trial in 0..2 {
                prefill_prep_gate_fused_matches_separate_for(heads, rows, 2048, trial, &mut r, &st);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion 4: the P2 prefill chunk + gated norm + commit tail fused launch
// (`gb_ling_kda_prefill_chunk_norm_commit`) against the separate `gb_glm_p2_kda_chunk` (vs = 64) +
// `gb_kda_norm` + `gb_kda_commit_tail` launches. Every output byte (y, S, out, conv_pool, parity) must be
// bit-identical.
// ---------------------------------------------------------------------------
fn prefill_chunk_norm_commit_fused_matches_separate_for(heads: usize, rows: usize, trial: usize, r: &mut Rng, st: &Stream) {
    const D: usize = 128;
    unsafe extern "C" {
        fn gb_glm_p2_kda_chunk(
            q: *const core::ffi::c_void,
            k: *const core::ffi::c_void,
            v: *const core::ffi::c_void,
            g: *const core::ffi::c_void,
            beta: *const core::ffi::c_void,
            s: *mut core::ffi::c_void,
            parity: *const core::ffi::c_void,
            slot: i32,
            rows: i32,
            heads: i32,
            vs: i32,
            rz: i32,
            y: *mut core::ffi::c_void,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        fn gb_kda_norm(
            y: *const core::ffi::c_void,
            z: *const f32,
            w: *const core::ffi::c_void,
            eps: f32,
            rows: i32,
            heads: i32,
            out: *mut core::ffi::c_void,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        fn gb_kda_commit_tail(
            raw: *const core::ffi::c_void,
            conv_pool: *mut core::ffi::c_void,
            parity: *mut i32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            heads: i32,
            flip: i32,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        fn gb_ling_kda_prefill_chunk_norm_commit(
            q: *const f32,
            k: *const f32,
            v: *const core::ffi::c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            parity: *mut i32,
            slot: i32,
            rows: i32,
            heads: i32,
            rz: i32,
            y: *mut core::ffi::c_void,
            z: *const f32,
            norm_w: *const core::ffi::c_void,
            eps: f32,
            out: *mut core::ffi::c_void,
            raw: *const core::ffi::c_void,
            conv_pool: *mut core::ffi::c_void,
            head_cnt: *mut i32,
            stream: *mut core::ffi::c_void,
            ko_suh: *const core::ffi::c_void,
            ko_xh: *mut core::ffi::c_void,
        ) -> i32;
        fn gb_glm_p2_rotate(
            x: *const core::ffi::c_void,
            x_dtype: i32,
            ldx: i64,
            suh: *const core::ffi::c_void,
            suh_f32: i32,
            a: *mut core::ffi::c_void,
            m: i32,
            k: i32,
            stream: *mut core::ffi::c_void,
        ) -> i32;
    }
    let conv_dim = 3 * heads * D;
    let row_len = rows * heads * D;
    let seq_len = 2 * heads * D * D;
    let eps = 1e-5f32;

    let s_host: Vec<f32> = (0..seq_len).map(|_| r.uniform(-0.5, 0.5)).collect();
    let parity_host: Vec<i32> = vec![0];
    let q_host: Vec<f32> = (0..row_len).map(|_| r.uniform(-0.5, 0.5)).collect();
    let k_host: Vec<f32> = (0..row_len).map(|_| r.uniform(-0.5, 0.5)).collect();
    let v_host: Vec<u16> = (0..row_len).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
    let g_host: Vec<f32> = (0..row_len).map(|_| r.uniform(-0.5, 0.5)).collect();
    let beta_host: Vec<f32> = (0..rows * heads).map(|_| r.uniform(-0.5, 0.5)).collect();
    let z_host: Vec<f32> = (0..row_len).map(|_| r.uniform(-0.5, 0.5)).collect();
    let norm_w_host: Vec<u16> = (0..D).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
    let raw_host: Vec<u16> = (0..rows * conv_dim).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
    let pool_host: Vec<u16> = (0..3 * conv_dim).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
    let seq_start: Vec<i32> = vec![0, rows as i32];
    let slot: Vec<i32> = vec![0];

    let mk = || {
        (
            DevBuf::from_slice(&s_host).unwrap(),
            DevBuf::from_slice(&parity_host).unwrap(),
            DevBuf::from_slice(&q_host).unwrap(),
            DevBuf::from_slice(&k_host).unwrap(),
            DevBuf::from_slice(&v_host).unwrap(),
            DevBuf::from_slice(&g_host).unwrap(),
            DevBuf::from_slice(&beta_host).unwrap(),
            DevBuf::from_slice(&z_host).unwrap(),
            DevBuf::from_slice(&norm_w_host).unwrap(),
            DevBuf::from_slice(&raw_host).unwrap(),
            DevBuf::from_slice(&pool_host).unwrap(),
            DevBuf::from_slice(&seq_start).unwrap(),
            DevBuf::from_slice(&slot).unwrap(),
            DevBuf::zeroed(rows * heads * D * 2).unwrap(), // y
            DevBuf::zeroed(rows * heads * D * 2).unwrap(), // out
        )
    };
    let (s_a, p_a, q_a, k_a, v_a, g_a, b_a, z_a, w_a, raw_a, pool_a, ss_a, sl_a, y_a, o_a) = mk();
    let (s_b, p_b, q_b, k_b, v_b, g_b, b_b, z_b, w_b, raw_b, pool_b, ss_b, sl_b, y_b, o_b) = mk();

    let rc = unsafe {
        gb_glm_p2_kda_chunk(
            q_a.ptr(),
            k_a.ptr(),
            v_a.ptr(),
            g_a.ptr(),
            b_a.ptr(),
            s_a.ptr(),
            p_a.ptr(),
            0,
            rows as i32,
            heads as i32,
            64,
            0,
            y_a.ptr(),
            st.raw(),
        )
    };
    assert_eq!(rc, 0, "separate chunk rc={rc}");
    let rc = unsafe {
        gb_kda_norm(y_a.ptr(), z_a.ptr() as *const f32, w_a.ptr(), eps, rows as i32, heads as i32, o_a.ptr(), st.raw())
    };
    assert_eq!(rc, 0, "separate norm rc={rc}");
    let rc = unsafe {
        gb_kda_commit_tail(
            raw_a.ptr(),
            pool_a.ptr(),
            p_a.ptr() as *mut i32,
            ss_a.ptr() as *const i32,
            sl_a.ptr() as *const i32,
            core::ptr::null(),
            1,
            heads as i32,
            1,
            st.raw(),
        )
    };
    assert_eq!(rc, 0, "separate commit rc={rc}");

    let head_cnt = DevBuf::zeroed(heads * 4).unwrap();
    // the fused k.o input rotation (ko_suh/ko_xh): the separate path rotates the gated output with the same suh
    let suh_host: Vec<u16> = (0..heads * D).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
    let suh = DevBuf::from_slice(&suh_host).unwrap();
    let mut ko_a = DevBuf::zeroed(rows * heads * D * 2).unwrap();
    let mut ko_xh = DevBuf::zeroed(rows * heads * D * 2).unwrap();
    let rc = unsafe {
        gb_glm_p2_rotate(o_a.ptr(), 0, (heads * D) as i64, suh.ptr(), 0, ko_a.ptr(), rows as i32, (heads * D) as i32, st.raw())
    };
    assert_eq!(rc, 0, "separate rotate rc={rc}");
    let rc = unsafe {
        gb_ling_kda_prefill_chunk_norm_commit(
            q_b.ptr() as *const f32,
            k_b.ptr() as *const f32,
            v_b.ptr(),
            g_b.ptr() as *const f32,
            b_b.ptr() as *const f32,
            s_b.ptr() as *mut f32,
            p_b.ptr() as *mut i32,
            0,
            rows as i32,
            heads as i32,
            0,
            y_b.ptr(),
            z_b.ptr() as *const f32,
            w_b.ptr(),
            eps,
            o_b.ptr(),
            raw_b.ptr(),
            pool_b.ptr(),
            head_cnt.ptr() as *mut i32,
            st.raw(),
            suh.ptr(),
            ko_xh.ptr(),
        )
    };
    assert_eq!(rc, 0, "fused chunk+norm+commit rc={rc}");

    st.sync().unwrap();
    let tag = format!("heads={heads} rows={rows} t{trial}");
    let (ya, yb) = (dl::<u16>(&y_a, st), dl::<u16>(&y_b, st));
    let (sa, sb) = (dl::<f32>(&s_a, st), dl::<f32>(&s_b, st));
    let (oa, ob) = (dl::<u16>(&o_a, st), dl::<u16>(&o_b, st));
    let (pa, pb) = (dl::<u16>(&pool_a, st), dl::<u16>(&pool_b, st));
    let (ra, rb) = (dl::<i32>(&p_a, st), dl::<i32>(&p_b, st));
    let (ka, kb) = (dl::<u16>(&ko_a, st), dl::<u16>(&ko_xh, st));
    eprintln!(
        "{tag} diff counts: y={} S={} out={} pool={} parity={} ko={}",
        diff_count(&ya, &yb),
        diff_count_f32(&sa, &sb),
        diff_count(&oa, &ob),
        diff_count(&pa, &pb),
        ra.iter().zip(&rb).filter(|(x, y)| x != y).count(),
        diff_count(&ka, &kb),
    );
    // S is fp32 and may legitimately contain NaN payloads: compare bitwise (f32 `!=` treats NaN as unequal).
    let s_ok = sa.len() == sb.len() && sa.iter().zip(&sb).all(|(x, y)| x.to_bits() == y.to_bits());
    if !s_ok {
        let i = sa.iter().zip(&sb).position(|(x, y)| x.to_bits() != y.to_bits()).unwrap();
        eprintln!("{tag} S first bit diff at {i}: A={:08x} B={:08x}", sa[i].to_bits(), sb[i].to_bits());
    }
    let ok = check_eq(&format!("{tag} y"), &ya, &yb)
        & s_ok
        & check_eq(&format!("{tag} out"), &oa, &ob)
        & check_eq(&format!("{tag} pool"), &pa, &pb)
        & check_eq(&format!("{tag} parity"), &ra, &rb)
        & check_eq(&format!("{tag} ko"), &ka, &kb);
    assert!(ok, "{tag}: fused chunk+norm+commit diverged");
    eprintln!("{tag}: identical");
}

#[test]
#[ignore]
fn prefill_chunk_norm_commit_fused_matches_separate() {
    let Some(st) = gpu() else { return };
    let mut r = Rng(0x9c1_5b84);
    for &heads in &[8usize, 64] {
        for &rows in &[128usize, 204, 512, 1488] {
            for trial in 0..2 {
                prefill_chunk_norm_commit_fused_matches_separate_for(heads, rows, trial, &mut r, &st);
            }
        }
    }
}
