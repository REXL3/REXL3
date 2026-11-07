//! The DSpark helper kernels (`csrc/dspark.cu`): the confidence head against its bit-exact host mirror (fixed accumulation and
//! butterfly order) at the real widths and at odd ones, row independence (a row's bits do not depend on its neighbours or its
//! position in the launch), an out-of-range previous token, and the bf16 <-> fp16 copies with their overflow counter.
//! GPU tests skip when there is no device.
#![cfg(not(gb_no_cuda))]

use gb_cuda::dspark::{self, bf16_to_f32, confidence_host, f16_to_f32};
use gb_cuda::{device_available, DevBuf, Stream};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 20_001) as f32 / 10_000.0 - 1.0
    }
}

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn f16_bits(v: f32) -> u16 {
    // round to nearest even for the magnitudes this test uses (|v| < 2, normal range)
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    let mut m = (man >> 13) as u16;
    let rest = man & 0x1fff;
    let mut e = exp as u16;
    if rest > 0x1000 || (rest == 0x1000 && (m & 1) == 1) {
        m += 1;
        if m == 0x400 {
            m = 0;
            e += 1;
        }
    }
    sign | (e << 10) | m
}

fn setup() -> Option<Stream> {
    if !device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    Some(Stream::new().unwrap())
}

#[test]
fn confidence_is_bit_exact_against_the_host_mirror() {
    let Some(st) = setup() else { return };
    let (hidden, rank, vocab) = (2048usize, 256usize, 5000usize);
    let mut rng = Lcg(7);
    let (n_seq, rows) = (3usize, 15usize);
    let r = n_seq * rows;
    let h: Vec<u16> = (0..r * hidden).map(|_| bf16_bits(rng.unit() * 3.0)).collect();
    let w1: Vec<u16> = (0..vocab * rank).map(|_| f16_bits(rng.unit() * 0.5)).collect();
    let w: Vec<f32> = (0..hidden + rank).map(|_| rng.unit() * 0.1).collect();
    let b = [0.37f32];
    let anchors: Vec<i32> = (0..n_seq).map(|_| (rng.next() % vocab as u64) as i32).collect();
    let tokens: Vec<i32> = (0..r).map(|_| (rng.next() % vocab as u64) as i32).collect();
    let (hd, td, ad) =
        (DevBuf::from_slice(&h).unwrap(), DevBuf::from_slice(&tokens).unwrap(), DevBuf::from_slice(&anchors).unwrap());
    let (w1d, wd, bd) = (DevBuf::from_slice(&w1).unwrap(), DevBuf::from_slice(&w).unwrap(), DevBuf::from_slice(&b).unwrap());
    let mut out = DevBuf::zeroed(r * 4).unwrap();
    dspark::confidence(&hd, &td, &ad, &w1d, &wd, &bd, &mut out, n_seq, rows, hidden, rank, vocab, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<f32> = out.to_vec().unwrap();
    for (s, &anchor) in anchors.iter().enumerate() {
        for j in 0..rows {
            let row = s * rows + j;
            let prev = if j == 0 { anchor } else { tokens[row - 1] } as usize;
            let m = &w1[prev * rank..(prev + 1) * rank];
            let want = confidence_host(&h[row * hidden..(row + 1) * hidden], m, &w, b[0]);
            // the dot and the bias are the same bits; the sigmoid's expf may differ in the last bit or two (device expf vs libm)
            assert!((got[row] - want).abs() <= want * 5e-7, "row {row} (seq {s}, j {j}): {} vs {want}", got[row]);
            assert!((0.0..=1.0).contains(&got[row]));
        }
    }
    // row independence: the same row alone (n_seq 1, rows 1 would change `prev`'s source, so launch the same layout on a sub-batch)
    let mut out1 = DevBuf::zeroed(rows * 4).unwrap();
    let (h1, t1, a1) = (
        DevBuf::from_slice(&h[rows * hidden..2 * rows * hidden]).unwrap(),
        DevBuf::from_slice(&tokens[rows..2 * rows]).unwrap(),
        DevBuf::from_slice(&anchors[1..2]).unwrap(),
    );
    dspark::confidence(&h1, &t1, &a1, &w1d, &wd, &bd, &mut out1, 1, rows, hidden, rank, vocab, &st).unwrap();
    st.sync().unwrap();
    let alone: Vec<f32> = out1.to_vec().unwrap();
    for j in 0..rows {
        assert_eq!(alone[j].to_bits(), got[rows + j].to_bits(), "row {j} alone vs in a batch of {n_seq}");
    }
}

#[test]
fn an_out_of_range_previous_token_is_a_zero_markov_embedding() {
    let Some(st) = setup() else { return };
    let (hidden, rank, vocab) = (2048usize, 256usize, 100usize);
    let mut rng = Lcg(11);
    let h: Vec<u16> = (0..hidden).map(|_| bf16_bits(rng.unit())).collect();
    let w1: Vec<u16> = (0..vocab * rank).map(|_| f16_bits(rng.unit())).collect();
    let w: Vec<f32> = (0..hidden + rank).map(|_| rng.unit() * 0.1).collect();
    let (hd, w1d, wd, bd) = (
        DevBuf::from_slice(&h).unwrap(),
        DevBuf::from_slice(&w1).unwrap(),
        DevBuf::from_slice(&w).unwrap(),
        DevBuf::from_slice(&[0.0f32]).unwrap(),
    );
    for bad in [-1i32, vocab as i32, i32::MAX] {
        let (td, ad) = (DevBuf::from_slice(&[0i32]).unwrap(), DevBuf::from_slice(&[bad]).unwrap());
        let mut out = DevBuf::zeroed(4).unwrap();
        dspark::confidence(&hd, &td, &ad, &w1d, &wd, &bd, &mut out, 1, 1, hidden, rank, vocab, &st).unwrap();
        st.sync().unwrap();
        let got: Vec<f32> = out.to_vec().unwrap();
        let want = confidence_host(&h, &vec![0u16; rank], &w, 0.0);
        assert!((got[0] - want).abs() <= want * 5e-7, "token {bad}: {} vs {want}", got[0]);
    }
}

#[test]
fn the_activation_copies_round_trip_and_count_overflow() {
    let Some(st) = setup() else { return };
    let n = 10_007usize;
    let mut rng = Lcg(3);
    let mut src: Vec<u16> = (0..n).map(|_| bf16_bits(rng.unit() * 100.0)).collect();
    // three values past fp16's range (65,504): 7e4, -1e6 and an infinity
    src[5] = bf16_bits(7.0e4);
    src[9000] = bf16_bits(-1.0e6);
    src[10_006] = 0x7f80;
    let sd = DevBuf::from_slice(&src).unwrap();
    let mut mid = DevBuf::zeroed(n * 2).unwrap();
    let mut bad = DevBuf::zeroed(4).unwrap();
    dspark::bf16_to_f16_copy(&sd, &mut mid, n, &mut bad, &st).unwrap();
    st.sync().unwrap();
    let c: Vec<i32> = bad.to_vec().unwrap();
    assert_eq!(c[0], 3, "three values do not fit fp16");
    let half: Vec<u16> = mid.to_vec().unwrap();
    for i in 0..n {
        if [5, 9000, 10_006].contains(&i) {
            continue;
        }
        // the fp16 value is the bf16 value rounded to 11 bits: within one fp16 ulp (2^-10 relative)
        let (a, b) = (bf16_to_f32(src[i]), f16_to_f32(half[i]));
        assert!((a - b).abs() <= a.abs() * 1e-3 + 1e-7, "element {i}: {a} vs {b}");
    }
    // back to bf16 and in place
    let mut back = DevBuf::zeroed(n * 2).unwrap();
    dspark::f16_to_bf16_copy(&mid, &mut back, n, &st).unwrap();
    let mut inplace = DevBuf::from_slice(&src).unwrap();
    let mut bad2 = DevBuf::zeroed(4).unwrap();
    dspark::bf16_to_f16_in_place(&mut inplace, n, &mut bad2, &st).unwrap();
    st.sync().unwrap();
    let (a, b): (Vec<u16>, Vec<u16>) = (mid.to_vec().unwrap(), inplace.to_vec().unwrap());
    assert_eq!(a, b, "in place == out of place");
    let bb: Vec<u16> = back.to_vec().unwrap();
    for i in (0..n).filter(|i| ![5, 9000, 10_006].contains(i)) {
        let (x, y) = (bf16_to_f32(src[i]), bf16_to_f32(bb[i]));
        assert!((x - y).abs() <= x.abs() * 8e-3 + 1e-7, "round trip {i}: {x} vs {y}");
    }
}
