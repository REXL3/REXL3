//! The host-side index rules of the MiniCPM-V-4.7 tower against the Hugging Face functions (`tools/gen_minicpmv_pos_fixtures.py`):
//! the nearest-bucket position table (torch `bucketize`, float32 boundaries) exhaustively for every side n in 1..=2048, and the
//! window merger's window order. CPU only: no device needed.

use gb_cuda::mcv_vision as mv;

const DIR: &str = "../../testdata/minicpmv4_7/vision";

fn fnv(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

/// Bucket of every coordinate i / n for one axis (the engine's table of an n x 1 grid divided by the row stride).
fn engine_buckets(n: usize) -> Vec<u8> {
    mv::nearest_position_ids(n, 1)
        .iter()
        .map(|&v| {
            assert_eq!(v % mv::POS_SIDE as i32, 0, "a one-column grid has bucket 0 on the column axis");
            (v / mv::POS_SIDE as i32) as u8
        })
        .collect()
}

#[test]
fn the_position_buckets_equal_torch_bucketize_for_every_side_to_2048() {
    let text = std::fs::read_to_string(format!("{DIR}/pos_buckets.txt")).unwrap();
    let (mut explicit, mut hashed) = (0, 0);
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let n: usize = it.next().unwrap().parse().unwrap();
        let got = engine_buckets(n);
        assert_eq!(got.len(), n);
        let first = it.next().unwrap();
        if let Some(h) = first.strip_prefix('#') {
            assert_eq!(format!("{:016x}", fnv(&got)), h, "n = {n}: bucket hash differs from torch");
            hashed += 1;
        } else {
            let want: Vec<u8> = std::iter::once(first).chain(it).map(|v| v.parse().unwrap()).collect();
            assert_eq!(got, want, "n = {n}");
            explicit += 1;
        }
    }
    assert_eq!((explicit, hashed), (128, 2048 - 128), "the fixture covers n = 1..=2048");
}

#[test]
fn a_2d_table_is_the_outer_combination_of_the_two_axes() {
    for (gh, gw) in [(32usize, 32usize), (16, 24), (40, 12), (70, 70), (1, 1), (56, 8), (128, 5)] {
        let (bh, bw) = (engine_buckets(gh), engine_buckets(gw));
        let t = mv::nearest_position_ids(gh, gw);
        for i in 0..gh {
            for j in 0..gw {
                assert_eq!(t[i * gw + j], bh[i] as i32 * 70 + bw[j] as i32, "{gh}x{gw} at ({i}, {j})");
            }
        }
    }
}

#[test]
fn the_window_order_equals_get_vision_window_index() {
    let text = std::fs::read_to_string(format!("{DIR}/window_index.txt")).unwrap();
    let mut grids = 0;
    for line in text.lines() {
        let v: Vec<usize> = line.split_whitespace().map(|x| x.parse().unwrap()).collect();
        let (gh, gw) = (v[0], v[1]);
        let want: Vec<u32> = v[2..].iter().map(|&x| x as u32).collect();
        assert_eq!(mv::window_source(gh, gw), want, "grid {gh} x {gw}");
        grids += 1;
    }
    assert_eq!(grids, 13);
}
