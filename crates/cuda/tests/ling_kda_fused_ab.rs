//! A/B bit-equality of the fused Ling KDA decode stage (`ling_kda_fused`) against the unfused pair
//! (`kda::gate_from_full_fg` + `kda::prep`) on identical random inputs, rows = 1 (the serial decode profile).
#![cfg(not(gb_no_cuda))]
mod common;

const HEADS: usize = 64;
const HIDDEN: usize = 4096;
const D: usize = 128;
const C: usize = HEADS * 128;
const CONV_DIM: usize = 3 * C;

use common::Rng;
use gb_cuda::kda::{self, Batch, Dims, Pool, Record, Weights, Workspace};
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

#[test]
#[ignore]
fn fused_matches_unfused_decode() {
    let Some(st) = gpu() else { return };
    let dims = Dims { heads: HEADS, hidden: HIDDEN };
    let mut r = Rng(0x91a7);
    let rows = 1usize;

    for trial in 0..5 {
        // inputs
        let x: Vec<u16> = (0..rows * HIDDEN).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
        let qkv: Vec<f32> = (0..rows * CONV_DIM).map(|_| r.uniform(-2.5, 2.5)).collect();
        let conv_w: Vec<u16> = (0..CONV_DIM * 4).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
        let conv_pool: Vec<u16> = (0..3 * CONV_DIM).map(|_| bf(r.uniform(-2.5, 2.5))).collect();
        let f_out: Vec<u16> = (0..rows * C).map(|_| f16(r.uniform(-1.0, 1.0) as f64)).collect();
        let g_out: Vec<u16> = (0..rows * C).map(|_| f16(r.uniform(-1.0, 1.0) as f64)).collect();
        let b_weight: Vec<u16> = (0..HEADS * HIDDEN).map(|_| bf(r.uniform(-0.05, 0.05))).collect();
        let a_log: Vec<f32> = (0..HEADS).map(|_| (1.0 + 15.0 * r.uniform(0.0, 1.0)).ln()).collect();
        let dt_bias: Vec<f32> = (0..C).map(|_| r.uniform(-1.0, 1.0)).collect();
        let seq_start: Vec<i32> = vec![0, rows as i32];
        let slot: Vec<i32> = vec![0];
        let lower = -5.0f32;

        let x_d = DevBuf::from_slice(&x).unwrap();
        let qkv_d = DevBuf::from_slice(&qkv).unwrap();
        let conv_w_d = DevBuf::from_slice(&conv_w).unwrap();
        let conv_pool_d = DevBuf::from_slice(&conv_pool).unwrap();
        let f_d = DevBuf::from_slice(&f_out).unwrap();
        let g_d = DevBuf::from_slice(&g_out).unwrap();
        let b_d = DevBuf::from_slice(&b_weight).unwrap();
        let a_d = DevBuf::from_slice(&a_log).unwrap();
        let dt_d = DevBuf::from_slice(&dt_bias).unwrap();
        let ss_d = DevBuf::from_slice(&seq_start).unwrap();
        let slot_d = DevBuf::from_slice(&slot).unwrap();
        let w = Weights {
            b: &b_d,
            f_a: &DevBuf::from_slice(&Vec::<u16>::new()).unwrap(),
            g_a: &DevBuf::from_slice(&Vec::<u16>::new()).unwrap(),
            f_b: &DevBuf::from_slice(&Vec::<u16>::new()).unwrap(),
            g_b: &DevBuf::from_slice(&Vec::<u16>::new()).unwrap(),
            conv_w: &conv_w_d,
            a_log: &a_d,
            dt_bias: &dt_d,
            o_norm: &DevBuf::from_slice(&Vec::<u16>::new()).unwrap(),
        };
        let mut pool = Pool::new(dims, 1).unwrap();
        pool.write_slot(0, &vec![0.0f32; HEADS * D * D], &conv_pool).unwrap();
        let batch = Batch { seq_start: &ss_d, slot: &slot_d, n_seq: 1, rows };

        // Route A: unfused pair.
        let mut rec_a = Record::alloc(dims, rows).unwrap();
        let mut ws_a = Workspace::alloc(dims, rows).unwrap();
        kda::gate_from_full_fg(
            &st,
            &f_d,
            &g_d,
            &x_d,
            &b_d,
            &a_d,
            &dt_d,
            lower,
            rows,
            HIDDEN,
            HEADS,
            &mut rec_a.g,
            ws_a.z_mut(),
            &mut rec_a.beta,
        )
        .unwrap();
        kda::prep(&st, dims, &w, &qkv_d, gb_cuda::exl3_linear::Dtype::F32, &mut pool, &batch, &mut rec_a, &mut ws_a).unwrap();

        // Route B: fused.
        let mut rec_b = Record::alloc(dims, rows).unwrap();
        let mut ws_b = Workspace::alloc(dims, rows).unwrap();
        gb_cuda::ling_kda_fused::prep_gate_fused(
            &st, dims, &w, &qkv_d, &pool, &batch, &mut rec_b, &mut ws_b, &f_d, &g_d, &x_d, &b_d, &a_d, &dt_d, lower,
        )
        .unwrap();

        let ok = check_eq("beta", &dl::<f32>(&rec_a.beta, &st), &dl::<f32>(&rec_b.beta, &st))
            & check_eq("g_decay", &dl::<f32>(&rec_a.g, &st), &dl::<f32>(&rec_b.g, &st))
            & check_eq("z", &dl::<f32>(ws_a.z(), &st), &dl::<f32>(ws_b.z(), &st))
            & check_eq("raw", &dl::<u16>(&rec_a.qkv, &st), &dl::<u16>(&rec_b.qkv, &st))
            & check_eq("q", &dl::<f32>(ws_a.q(), &st), &dl::<f32>(ws_b.q(), &st))
            & check_eq("k", &dl::<f32>(ws_a.k(), &st), &dl::<f32>(ws_b.k(), &st))
            & check_eq("v", &dl::<u16>(ws_a.v(), &st), &dl::<u16>(ws_b.v(), &st));
        assert!(ok, "trial {trial}: fused diverged from the unfused pair");
        eprintln!("trial {trial}: identical");
    }
}

/// A/B bit-equality of the full decode/verify fusion (`prep_beta_fgg_window_norm_fused`: prep + beta + fgg +
/// window + norm in one 128-thread-block launch) against the sequential pair `prep_gate_fused` + `window_and_norm`
/// on identical random batches: `rows in {1, 2, 4, 8}` for `n_seq in {1, 2}`, heads 8 and 64, plus the R = 4
/// profile (`r_step = 4`, rows = 8). Every output buffer — raw (qkv), q/k/v, g_decay, z, beta, y, the full S pool,
/// the fp16 norm output and the rotated o_proj input — must be 0-bit-diff.
#[test]
#[ignore]
fn prep_beta_fgg_window_norm_matches_separate() {
    let Some(st) = gpu() else { return };
    let mut r = Rng(0xda_ba_5b84);
    let mut head_cnt: Option<DevBuf> = None;
    let mut done = 0usize;
    let mut combos: Vec<(usize, usize, usize, usize)> = Vec::new(); // (heads, n_seq, rows, r_step)
    for &heads in &[8usize, HEADS] {
        for &n_seq in &[1usize, 2] {
            for &rows in &[1usize, 2, 4, 8] {
                combos.push((heads, n_seq, rows, 8));
            }
        }
    }
    combos.push((HEADS, 1, 8, 4)); // the R = 4 profile
    let combos_len = combos.len();

    for (heads0, n_seq0, rows0, r_step0) in combos {
        for trial in 0..3 {
            let p = kda::Params { lower: -5.0, eps: 1e-5, round: kda::Round::NearestEven, r_step: r_step0 };
            let dims = Dims { heads: heads0, hidden: HIDDEN };
            let (heads, n_seq, rows) = (heads0, n_seq0, rows0);
            // sequence lengths: first sequences take rows / n_seq (>= 1), the last the remainder (may be empty)
            let lens: Vec<usize> = {
                let mut l = Vec::with_capacity(n_seq);
                let mut rem = rows;
                for i in 0..n_seq {
                    let take = if i + 1 == n_seq { rem } else { (rows / n_seq).max(1) };
                    l.push(take);
                    rem -= take;
                }
                l
            };
            let mut seq_start: Vec<i32> = vec![0];
            for &l in &lens {
                let last = *seq_start.last().unwrap();
                seq_start.push(last + l as i32);
            }
            let slot: Vec<i32> = (0..n_seq).map(|i| i as i32).collect();
            let conv_dim = 3 * heads * 128;
            let row_len = rows * heads * 128;

            let qkv: Vec<f32> = (0..rows * conv_dim).map(|_| r.uniform(-2.5, 2.5)).collect();
            let conv_w: Vec<u16> = (0..conv_dim * 4).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let conv_pool: Vec<u16> = (0..3 * conv_dim).map(|_| bf(r.uniform(-2.5, 2.5))).collect();
            let f_out: Vec<u16> = (0..row_len).map(|_| f16(r.uniform(-1.0, 1.0) as f64)).collect();
            let g_out: Vec<u16> = (0..row_len).map(|_| f16(r.uniform(-1.0, 1.0) as f64)).collect();
            let x: Vec<u16> = (0..rows * HIDDEN).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let b_weight: Vec<u16> = (0..heads * HIDDEN).map(|_| bf(r.uniform(-0.05, 0.05))).collect();
            let a_log: Vec<f32> = (0..heads).map(|_| (1.0 + 15.0 * r.uniform(0.0, 1.0)).ln()).collect();
            let dt_bias: Vec<f32> = (0..heads * 128).map(|_| r.uniform(-1.0, 1.0)).collect();
            let norm_w: Vec<u16> = (0..128).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let o_suh: Vec<u16> = (0..heads * 128).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
            let s_host: Vec<f32> = (0..heads * 128 * 128).map(|_| r.uniform(-0.5, 0.5)).collect();
            let lower = -5.0f32;

            let qkv_d = DevBuf::from_slice(&qkv).unwrap();
            let conv_w_d = DevBuf::from_slice(&conv_w).unwrap();
            let conv_pool_d = DevBuf::from_slice(&conv_pool).unwrap();
            let f_d = DevBuf::from_slice(&f_out).unwrap();
            let g_d = DevBuf::from_slice(&g_out).unwrap();
            let x_d = DevBuf::from_slice(&x).unwrap();
            let b_d = DevBuf::from_slice(&b_weight).unwrap();
            let a_d = DevBuf::from_slice(&a_log).unwrap();
            let dt_d = DevBuf::from_slice(&dt_bias).unwrap();
            let norm_w_d = DevBuf::from_slice(&norm_w).unwrap();
            let o_suh_d = DevBuf::from_slice(&o_suh).unwrap();
            let ss_d = DevBuf::from_slice(&seq_start).unwrap();
            let slot_d = DevBuf::from_slice(&slot).unwrap();
            let empty = DevBuf::from_slice(&Vec::<u16>::new()).unwrap();
            let w = Weights {
                b: &b_d,
                f_a: &empty,
                g_a: &empty,
                f_b: &empty,
                g_b: &empty,
                conv_w: &conv_w_d,
                a_log: &a_d,
                dt_bias: &dt_d,
                o_norm: &norm_w_d,
            };
            let batch = Batch { seq_start: &ss_d, slot: &slot_d, n_seq, rows };

            let mut pool_a = Pool::new(dims, n_seq).unwrap();
            let mut pool_b = Pool::new(dims, n_seq).unwrap();
            for s in 0..n_seq {
                pool_a.write_slot(s, &s_host, &conv_pool).unwrap();
                pool_b.write_slot(s, &s_host, &conv_pool).unwrap();
            }
            if head_cnt.is_none() {
                head_cnt = Some(DevBuf::zeroed(2 * HEADS * 4).unwrap());
            }
            let hc = head_cnt.as_ref().unwrap();
            let y_bytes = rows * heads * 128 * 2;

            // Route A: separate prep_gate_fused + window_and_norm launches.
            let mut rec_a = Record::alloc(dims, rows).unwrap();
            let mut ws_a = Workspace::alloc(dims, rows).unwrap();
            gb_cuda::ling_kda_fused::prep_gate_fused(
                &st, dims, &w, &qkv_d, &pool_a, &batch, &mut rec_a, &mut ws_a, &f_d, &g_d, &x_d, &b_d, &a_d, &dt_d, lower,
            )
            .unwrap();
            let mut out_a = DevBuf::zeroed(y_bytes).unwrap();
            let mut oxh_a = DevBuf::zeroed(y_bytes).unwrap();
            gb_cuda::ling_kda_fused::window_and_norm(
                &st,
                dims,
                &p,
                &pool_a,
                &batch,
                &rec_a,
                &mut ws_a,
                hc,
                &w,
                Some(&o_suh_d),
                Some(&mut oxh_a),
                &mut out_a,
            )
            .unwrap();

            // Route B: the fully fused launch.
            let mut rec_b = Record::alloc(dims, rows).unwrap();
            let mut ws_b = Workspace::alloc(dims, rows).unwrap();
            let mut out_b = DevBuf::zeroed(y_bytes).unwrap();
            let mut oxh_b = DevBuf::zeroed(y_bytes).unwrap();
            gb_cuda::ling_kda_fused::prep_beta_fgg_window_norm_fused(
                &st,
                dims,
                &p,
                &w,
                &qkv_d,
                &pool_b,
                &batch,
                &mut rec_b,
                &mut ws_b,
                hc,
                &f_d,
                &g_d,
                &x_d,
                &b_d,
                &a_d,
                &dt_d,
                lower,
                Some(&o_suh_d),
                Some(&mut oxh_b),
                &mut out_b,
            )
            .unwrap();

            st.sync().unwrap();
            let tag = format!("heads={heads} n_seq={n_seq} rows={rows} r_step={r_step0} trial={trial}");
            let ok = check_eq("raw", &dl::<u16>(&rec_a.qkv, &st), &dl::<u16>(&rec_b.qkv, &st))
                & check_eq("q", &dl::<f32>(ws_a.q(), &st), &dl::<f32>(ws_b.q(), &st))
                & check_eq("k", &dl::<f32>(ws_a.k(), &st), &dl::<f32>(ws_b.k(), &st))
                & check_eq("v", &dl::<u16>(ws_a.v(), &st), &dl::<u16>(ws_b.v(), &st))
                & check_eq("g_decay", &dl::<f32>(&rec_a.g, &st), &dl::<f32>(&rec_b.g, &st))
                & check_eq("z", &dl::<f32>(ws_a.z(), &st), &dl::<f32>(ws_b.z(), &st))
                & check_eq("beta", &dl::<f32>(&rec_a.beta, &st), &dl::<f32>(&rec_b.beta, &st))
                & check_eq("y", &dl::<u16>(ws_a.y(), &st), &dl::<u16>(ws_b.y(), &st))
                & check_eq("S", &dl::<f32>(&pool_a.s, &st), &dl::<f32>(&pool_b.s, &st))
                & check_eq("out", &dl::<u16>(&out_a, &st), &dl::<u16>(&out_b, &st))
                & check_eq("o_xh", &dl::<u16>(&oxh_a, &st), &dl::<u16>(&oxh_b, &st));
            assert!(ok, "fused full diverged from the separate pair ({tag})");
            eprintln!("{tag}: identical");
            done += 1;
        }
    }
    assert_eq!(done, combos_len * 3);
}
