//! Bit-exact and launch-geometry coverage for the FP16 DSV4.1 index-score serving kernel.
#![cfg(not(gb_no_cuda))]

use gb_cuda::{DevBuf, Stream};
use std::ffi::c_void;

#[allow(clippy::too_many_arguments)]
unsafe extern "C" {
    fn gb_dsv4_index_score(
        q: *const c_void,
        index_k: *const c_void,
        w: *const f32,
        mask: *const u8,
        mask_stride: i32,
        score: *mut f32,
        rows: i32,
        n_pos: i32,
        n_heads: i32,
        dim: i32,
        scale: f32,
        start_pos: i32,
        ratio: i32,
        dtype: i32,
        force_scalar: i32,
        stream: *mut c_void,
    ) -> i32;
}

#[allow(clippy::too_many_arguments)]
fn launch(
    q: &DevBuf,
    k: &DevBuf,
    w: &DevBuf,
    mask: *const u8,
    mask_stride: usize,
    score: &DevBuf,
    rows: usize,
    n_pos: usize,
    scale: f32,
    start_pos: usize,
    ratio: usize,
    scalar: bool,
    st: &Stream,
) {
    let rc = unsafe {
        gb_dsv4_index_score(
            q.ptr(),
            k.ptr(),
            w.ptr() as *const f32,
            mask,
            mask_stride as i32,
            score.ptr() as *mut f32,
            rows as i32,
            n_pos as i32,
            32,
            128,
            scale,
            start_pos as i32,
            ratio as i32,
            0,
            scalar as i32,
            st.raw(),
        )
    };
    assert_eq!(rc, 0, "CUDA launcher rejected the call");
    st.sync().unwrap();
}

fn values(n: usize, seed: u64) -> Vec<u16> {
    let mut x = seed;
    (0..n)
        .map(|i| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let v = ((x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 3.0;
            let v = v + ((i % 7) as f32 - 3.0) * 0.03125;
            gb_exl3::f16::f64_to_f16(v as f64)
        })
        .collect()
}

#[test]
fn fp16_index_score_is_bit_identical_and_geometry_invariant() {
    let st = Stream::new().unwrap();
    for &(rows, n_pos, start_pos, ratio) in &[
        (1usize, 1usize, 0usize, 1usize),
        (3, 3, 0, 2),
        (7, 31, 13, 1),
        (7, 32, 13, 2),
        (9, 33, 17, 1),
        (5, 127, 29, 2),
        (3, 257, 31, 1),
    ] {
        let q_host = values(rows * 32 * 128, 0x1234_5678_9abc_def0);
        let k_host = values(n_pos * 128, 0xfeed_face_cafe_beef);
        let w_host: Vec<f32> = (0..rows * 32).map(|i| ((i * 13 % 29) as f32 - 14.0) / 17.0).collect();
        let mask_host: Vec<u8> = (0..rows * n_pos).map(|i| ((i * 7 + i / n_pos) % 5 != 0) as u8).collect();
        let q = DevBuf::from_slice(&q_host).unwrap();
        let k = DevBuf::from_slice(&k_host).unwrap();
        let w = DevBuf::from_slice(&w_host).unwrap();
        let mask = DevBuf::from_slice(&mask_host).unwrap();
        let scale = 1.0f32 / 128.0f32.sqrt();
        let sentinel = 12_345.0f32;
        let guard = 64usize;
        let initial = vec![sentinel; rows * n_pos + guard];
        let optimized = DevBuf::from_slice(&initial).unwrap();
        let reference = DevBuf::from_slice(&initial).unwrap();
        launch(&q, &k, &w, mask.ptr() as *const u8, n_pos, &optimized, rows, n_pos, scale, start_pos, ratio, false, &st);
        launch(&q, &k, &w, mask.ptr() as *const u8, n_pos, &reference, rows, n_pos, scale, start_pos, ratio, true, &st);
        let got = optimized.to_vec::<f32>().unwrap();
        let want = reference.to_vec::<f32>().unwrap();
        for i in 0..rows * n_pos {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "shape rows={rows} n_pos={n_pos} start={start_pos} ratio={ratio}, element {i}"
            );
        }
        assert_eq!(&got[rows * n_pos..], &vec![sentinel; guard], "optimized kernel overwrote its output tail");
        assert_eq!(&want[rows * n_pos..], &vec![sentinel; guard], "scalar kernel overwrote its output tail");

        // Every row must be independent of the number and position of its neighbours in the launch grid.
        for r in 0..rows {
            let qr = q.subview(r * 32 * 128 * 2, 32 * 128 * 2);
            let wr = w.subview(r * 32 * 4, 32 * 4);
            let mr = mask.subview(r * n_pos, n_pos);
            let one = DevBuf::from_slice(&initial[..n_pos + guard]).unwrap();
            launch(&qr, &k, &wr, mr.ptr() as *const u8, n_pos, &one, 1, n_pos, scale, start_pos + r, ratio, false, &st);
            let row = one.to_vec::<f32>().unwrap();
            for p in 0..n_pos {
                assert_eq!(
                    row[p].to_bits(),
                    got[r * n_pos + p].to_bits(),
                    "row-invariance rows={rows} n_pos={n_pos} row={r} pos={p}"
                );
            }
            assert_eq!(&row[n_pos..], &vec![sentinel; guard]);
        }

        // A different position-grid height must not change the common scores or touch the finite guard.
        if n_pos > 33 {
            let short_pos = n_pos - 1;
            let short = DevBuf::from_slice(&vec![sentinel; rows * short_pos + guard]).unwrap();
            launch(&q, &k, &w, mask.ptr() as *const u8, n_pos, &short, rows, short_pos, scale, start_pos, ratio, false, &st);
            let shorter = short.to_vec::<f32>().unwrap();
            for r in 0..rows {
                for p in 0..short_pos {
                    assert_eq!(
                        shorter[r * short_pos + p].to_bits(),
                        got[r * n_pos + p].to_bits(),
                        "chunk-grid invariance row={r} pos={p}"
                    );
                }
            }
            assert_eq!(&shorter[rows * short_pos..], &vec![sentinel; guard]);
        }
    }
}
