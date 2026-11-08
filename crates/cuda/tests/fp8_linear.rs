//! FP8 E4M3 block-scaled linear (`csrc/fp8_linear.cu`): a host reference for the same arithmetic, a hand-computed
//! 2x2-tile dequantisation, a 128x256 GEMM against the reference, a case where dropping the scale changes the answer
//! by 16x, and the load/forward validation errors by name.
//!
//! The host tests (the reference itself, the hand-computed 2x2 case, the validation shape checks, the E8M0 decode)
//! run everywhere. The device tests print `not run: no device` and return when there is no GPU, so this binary is
//! green on a CPU-only box.
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::fp8_linear::*;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};

// ---------------------------------------------------------------------------------------------------- helpers

/// A device to run on, or `None` (with the reason printed) on a machine without one.
fn gpu() -> Option<Stream> {
    if !device_available() {
        eprintln!("not run: no device");
        return None;
    }
    Some(Stream::new().unwrap())
}

/// xorshift64*, so the test data is the same everywhere and needs no dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// A uniform f32 in `[lo, hi)`.
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * u
    }
    /// A finite E4M3 bit pattern (its 0x7F / 0xFF NaN encodings excluded).
    fn e4m3(&mut self) -> u8 {
        let b = (self.next_u64() & 0x7F) as u8;
        if b == 0x7F {
            0x3C
        } else {
            b
        }
    }
    /// An E8M0 byte around 1.0: `2^-6 .. 2^6`.
    fn e8m0(&mut self) -> u8 {
        (0x79 + (self.next_u64() % 13) as u8) & 0x7F
    }
}

/// The host reference in f64 over the packed bytes: `y[m][n] = sum_k x[m][k] * e4m3(W[n][k]) * 2^(b - 127)`.
/// `x` is already the value the device sees (fp16-rounded).
fn reference(x: &[f32], w: &[u8], s: &[u8], m: usize, n: usize, k: usize) -> Vec<f64> {
    let kt = k / TILE;
    let mut y = vec![0.0f64; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                let sc = e8m0_to_f32(s[(c / TILE) * kt + kk / TILE]) as f64;
                acc += x[r * k + kk] as f64 * e4m3_to_f32(w[c * k + kk]) as f64 * sc;
            }
            y[r * n + c] = acc;
        }
    }
    y
}

/// `max |got - want| / max |want|`.
fn max_rel(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        num = num.max((g - w).abs());
        den = den.max(w.abs());
    }
    num / den.max(1e-30)
}

// ---------------------------------------------------------------------------------------------------- E8M0

#[test]
fn e8m0_decodes_as_an_exponent_with_bias_127() {
    assert_eq!(e8m0_to_f32(127), 1.0);
    assert_eq!(e8m0_to_f32(128), 2.0);
    assert_eq!(e8m0_to_f32(126), 0.5);
    assert_eq!(e8m0_to_f32(131), 16.0);
    assert_eq!(e8m0_to_f32(123), 1.0 / 16.0);
    // b = 0 is 2^-127, a normal f32 subnormal: an exponent-field insert would give 0.0 here, so this is the case that
    // catches a `from_bits(b << 23)` decode.
    assert_eq!(e8m0_to_f32(0), 2.0f32.powi(-127));
    assert_ne!(e8m0_to_f32(0), 0.0);
    assert!(e8m0_to_f32(0xFF).is_nan());
}

#[test]
fn e4m3_byte_decodes_exactly() {
    assert_eq!(e4m3_to_f32(0x00), 0.0);
    assert_eq!(e4m3_to_f32(0x38), 1.0);
    assert_eq!(e4m3_to_f32(0x3C), 1.5);
    assert_eq!(e4m3_to_f32(0x40), 2.0);
    assert_eq!(e4m3_to_f32(0xB8), -1.0);
    assert_eq!(e4m3_to_f32(0x7E), 448.0); // the format's max
    assert_eq!(e4m3_to_f32(0x04), 2.0f32.powi(-7)); // subnormal: (4/8) * 2^-6
    assert!(e4m3_to_f32(0x7F).is_nan());
}

// ---------------------------------------------------------------------------------------------------- dequant, 2x2 tiles

/// The hand case: `N = K = 64`, so the weight is 2x2 scale tiles, every byte of a tile the same E4M3 value and every
/// tile its own power of two. `0x3C` is 1.5 (exp 7, mantissa 4/8) and `0x7F`/`0x80`/`0x7E`/`0x81` are 1, 2, 0.5 and
/// 4, so the four tiles dequantise to 1.5, 3.0, 0.75 and 6.0 — written out by hand, not computed.
fn hand_case() -> (usize, usize, Vec<u8>, Vec<u8>, [[f32; 2]; 2]) {
    let (n, k) = (64, 64);
    let w = vec![0x3Cu8; n * k];
    let s = vec![0x7F, 0x80, 0x7E, 0x81]; // [n/32][k/32] row major: (0,0) (0,1) (1,0) (1,1)
    (n, k, w, s, [[1.5, 3.0], [0.75, 6.0]])
}

#[test]
fn dequant_2x2_is_exact_on_the_host() {
    let (n, k, w, s, want) = hand_case();
    let kt = k / TILE;
    for ni in 0..n {
        for ki in 0..k {
            let v = e4m3_to_f32(w[ni * k + ki]) * e8m0_to_f32(s[(ni / TILE) * kt + ki / TILE]);
            let expect = want[ni / TILE][ki / TILE];
            assert_eq!(v.to_bits(), expect.to_bits(), "({ni}, {ki}) = {v} != {expect}");
        }
    }
}

#[test]
fn dequant_kernel_matches_the_hand_case() {
    let Some(st) = gpu() else { return };
    let (n, k, w, s, want) = hand_case();
    let wt = Fp8Weights::new(n, k, &w, &s).unwrap();
    let mut out = DevBuf::zeroed(n * k * 2).unwrap();
    wt.dequantise(&mut out, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<u16> = out.to_vec().unwrap();
    for ni in 0..n {
        for ki in 0..k {
            let v = f16_to_f64(got[ni * k + ki]) as f32;
            let expect = want[ni / TILE][ki / TILE];
            assert_eq!(v.to_bits(), expect.to_bits(), "({ni}, {ki}) = {v} != {expect}");
        }
    }
}

// ---------------------------------------------------------------------------------------------------- GEMM

fn build_case(rng: &mut Rng, m: usize, n: usize, k: usize) -> (Vec<u16>, Vec<u8>, Vec<u8>) {
    let x: Vec<u16> = (0..m * k).map(|_| f64_to_f16(rng.uniform(-1.0, 1.0) as f64)).collect();
    let w: Vec<u8> = (0..n * k).map(|_| rng.e4m3()).collect();
    let s: Vec<u8> = (0..(n / TILE) * (k / TILE)).map(|_| rng.e8m0()).collect();
    (x, w, s)
}

#[test]
fn gemm_128x256_matches_the_host_reference() {
    let Some(st) = gpu() else { return };
    let (m, n, k) = (128, 256, 256);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (x16, w, s) = build_case(&mut rng, m, n, k);
    let xf: Vec<f32> = x16.iter().map(|&h| f16_to_f64(h) as f32).collect();
    let want = reference(&xf, &w, &s, m, n, k);

    let wt = Fp8Weights::new(n, k, &w, &s).unwrap();
    let x = DevBuf::from_slice(&x16).unwrap();
    let mut y = DevBuf::zeroed(m * n * 2).unwrap();
    wt.forward(&x, &mut y, m, Dtype::F16, &st).unwrap();
    st.sync().unwrap();
    let got16: Vec<u16> = y.to_vec().unwrap();
    let got: Vec<f64> = got16.iter().map(|&h| f16_to_f64(h)).collect();

    let rel = max_rel(&got, &want);
    assert!(rel < 1e-2, "max relative error {rel} over 128x256");
    // The fp16 output's own rounding is the floor for this metric; print it so a regression is visible in the log.
    eprintln!("fp8 128x256: max |y - ref| / max |ref| = {rel:.3e}");
}

#[test]
fn gemm_bf16_input_takes_the_same_path() {
    let Some(st) = gpu() else { return };
    let (m, n, k) = (3, 64, 64);
    let mut rng = Rng(0x0BAD_C0DE_1234_5678);
    let (x16, w, s) = build_case(&mut rng, m, n, k);
    let xb: Vec<u16> = x16.iter().map(|&h| ((f16_to_f64(h) as f32).to_bits() >> 16) as u16).collect();
    let xf: Vec<f32> = xb.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
    let want = reference(&xf, &w, &s, m, n, k);

    let wt = Fp8Weights::new(n, k, &w, &s).unwrap();
    let x = DevBuf::from_slice(&xb).unwrap();
    let mut y = DevBuf::zeroed(m * n * 2).unwrap();
    wt.forward(&x, &mut y, m, Dtype::BF16, &st).unwrap();
    st.sync().unwrap();
    let got16: Vec<u16> = y.to_vec().unwrap();
    let got: Vec<f64> = got16.iter().map(|&h| f16_to_f64(h)).collect();
    // bf16's 8-bit mantissa is the error floor here, not the kernel.
    assert!(max_rel(&got, &want) < 1e-1);
}

#[test]
fn decode_and_prefill_paths_agree_bit_for_bit() {
    let Some(st) = gpu() else { return };
    let (n, k) = (256, 256);
    let mut rng = Rng(0xDEAD_BEEF_CAFE_0001);
    let (x16, w, s) = build_case(&mut rng, 64, n, k);
    let wt = Fp8Weights::new(n, k, &w, &s).unwrap();

    // 8 rows: the decode shape (grid.y = 1, 8 rows per block).
    let x8 = DevBuf::from_slice(&x16[..8 * k]).unwrap();
    let mut y8 = DevBuf::zeroed(8 * n * 2).unwrap();
    wt.forward(&x8, &mut y8, 8, Dtype::F16, &st).unwrap();
    // 64 rows: the prefill shape (32 rows per block). Its first 8 rows must be the same bits.
    let x64 = DevBuf::from_slice(&x16).unwrap();
    let mut y64 = DevBuf::zeroed(64 * n * 2).unwrap();
    wt.forward(&x64, &mut y64, 64, Dtype::F16, &st).unwrap();
    st.sync().unwrap();

    let a: Vec<u16> = y8.to_vec().unwrap();
    let b: Vec<u16> = y64.to_vec().unwrap();
    assert_eq!(&a[..], &b[..8 * n], "decode and prefill rows differ");
}

#[test]
fn small_m_pipelined_path_agrees_with_the_classic_path_bit_for_bit() {
    let Some(st) = gpu() else { return };
    // Decode shapes from the real model: shared expert w1/w3/w2, MLA wq_a/wkv/wo_b, plus an odd-K case so
    // the partial-chunk gating and the last-stage wait are exercised.
    let shapes: [(usize, usize); 6] = [(512, 5120), (2304, 5120), (5120, 2304), (1280, 5120), (5120, 8192), (1024, 1024)];
    let mut rng = Rng(0x5EED_5EED_5EED_5EED);
    for (n, k) in shapes {
        let (x16, w, s) = build_case(&mut rng, 8, n, k);
        let wt = Fp8Weights::new(n, k, &w, &s).unwrap();
        let x = DevBuf::from_slice(&x16).unwrap();
        for m in [1usize, 2, 3, 4, 6, 8] {
            let mut classic = DevBuf::zeroed(m * n * 2).unwrap();
            let mut fast = DevBuf::zeroed(m * n * 2).unwrap();
            wt.forward_classic(&x, &mut classic, m, Dtype::F16, &st).unwrap();
            wt.forward(&x, &mut fast, m, Dtype::F16, &st).unwrap();
            st.sync().unwrap();
            let a: Vec<u16> = classic.to_vec().unwrap();
            let b: Vec<u16> = fast.to_vec().unwrap();
            if a != b {
                let i = a.iter().zip(&b).position(|(x, y)| x != y).unwrap();
                eprintln!(
                    "m={m} n={n} k={k}: first mismatch at {i} (row {} col {}): classic={} fast={}",
                    i / n,
                    i % n,
                    a[i],
                    b[i]
                );
                let ndiff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
                eprintln!("  {ndiff} of {} elements differ", a.len());
            }
            assert_eq!(a, b, "pipelined path differs on [{m}, {n}, {k}]");
        }
    }
}

#[test]
fn small_m_pair_pipelined_path_agrees_with_the_classic_path_bit_for_bit() {
    let Some(st) = gpu() else { return };
    // wq_a+wkv and shared w1+w3 pair shapes, plus an odd-K tail.
    let pairs: [(usize, usize, usize); 3] = [(1280, 512, 5120), (2304, 2304, 5120), (640, 352, 1024)];
    let mut rng = Rng(0xABBA_BABA_ABBA_BABA);
    for (n1, n2, k) in pairs {
        let (x16, w1, s1) = build_case(&mut rng, 8, n1, k);
        let (_, w2, s2) = build_case(&mut rng, 8, n2, k);
        let wa = Fp8Weights::new(n1, k, &w1, &s1).unwrap();
        let wb = Fp8Weights::new(n2, k, &w2, &s2).unwrap();
        let x = DevBuf::from_slice(&x16).unwrap();
        for m in [1usize, 2, 4, 6, 8] {
            let mut y1c = DevBuf::zeroed(m * n1 * 2).unwrap();
            let mut y2c = DevBuf::zeroed(m * n2 * 2).unwrap();
            let mut y1n = DevBuf::zeroed(m * n1 * 2).unwrap();
            let mut y2n = DevBuf::zeroed(m * n2 * 2).unwrap();
            Fp8Weights::forward_pair_classic(&wa, &mut y1c, &wb, &mut y2c, &x, m, Dtype::F16, &st).unwrap();
            Fp8Weights::forward_pair(&wa, &mut y1n, &wb, &mut y2n, &x, m, Dtype::F16, &st).unwrap();
            st.sync().unwrap();
            let a1: Vec<u16> = y1c.to_vec().unwrap();
            let b1: Vec<u16> = y1n.to_vec().unwrap();
            let a2: Vec<u16> = y2c.to_vec().unwrap();
            let b2: Vec<u16> = y2n.to_vec().unwrap();
            assert_eq!(a1, b1, "pair 1 differs on [{m}, {n1}, {k}]");
            assert_eq!(a2, b2, "pair 2 differs on [{m}, {n2}, {k}]");
        }
    }
}

/// The scale has to be applied: two column tiles, one at 2^0 and one at 2^-4, unit weights and unit activations. The
/// unscaled answer is 32 for both tiles; the true one is 32 and 2. Dropping the scale is off by 16x on the second
/// tile, which any "did we multiply?" bug fails and no rounding could hide.
#[test]
fn scale_is_applied() {
    let Some(st) = gpu() else { return };
    let (m, n, k) = (4, 64, 32);
    let w = vec![0x38u8; n * k]; // 1.0
    let s = vec![0x7F, 0x7B]; // 2^0, 2^-4
    let x16: Vec<u16> = vec![f64_to_f16(1.0); m * k];
    let want = reference(&vec![1.0f32; m * k], &w, &s, m, n, k);

    let wt = Fp8Weights::new(n, k, &w, &s).unwrap();
    let x = DevBuf::from_slice(&x16).unwrap();
    let mut y = DevBuf::zeroed(m * n * 2).unwrap();
    wt.forward(&x, &mut y, m, Dtype::F16, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<f64> = y.to_vec::<u16>().unwrap().iter().map(|&h| f16_to_f64(h)).collect();

    for r in 0..m {
        assert_eq!(got[r * n], 32.0, "tile 0 row {r}");
        assert_eq!(got[r * n + 32], 2.0, "tile 1 row {r}");
    }
    // The same case with the scale dropped (every tile 2^0): the second tile is 16x too large.
    let unscaled = reference(&vec![1.0f32; m * k], &w, &[0x7F, 0x7F], m, n, k);
    let off = unscaled[32].abs() / want[32].abs();
    assert!(off > 10.0, "ignoring the scale must move the answer by >10x, got {off}x");
    assert!((off - 16.0).abs() < 1e-6, "expected exactly 16x, got {off}x");
}

// ---------------------------------------------------------------------------------------------------- validation

#[test]
fn load_rejects_a_short_weight() {
    let (n, k) = (64, 64);
    let e = Fp8Weights::new(n, k, &[0x38u8; 64 * 64 - 1], &[0x7F; 4]).err().unwrap();
    assert_eq!(e, Fp8Error::ShapeMismatch { what: "w [n, k]", expected: 4096, got: 4095 });
    assert!(e.to_string().contains("w [n, k]"));
}

#[test]
fn load_rejects_a_short_scale() {
    let (n, k) = (64, 64);
    let e = Fp8Weights::new(n, k, &[0x38u8; 64 * 64], &[0x7F; 3]).err().unwrap();
    assert_eq!(e, Fp8Error::ShapeMismatch { what: "scale [n/32, k/32]", expected: 4, got: 3 });
}

#[test]
fn load_rejects_a_ragged_shape() {
    let e = Fp8Weights::new(48, 64, &[0x38u8; 48 * 64], &[0x7F; 3]).err().unwrap();
    assert_eq!(e, Fp8Error::NotTiled { what: "n", value: 48, tile: 32 });
    let e = Fp8Weights::new(64, 40, &[0x38u8; 64 * 40], &[0x7F; 2]).err().unwrap();
    assert_eq!(e, Fp8Error::NotTiled { what: "k", value: 40, tile: 32 });
    let e = Fp8Weights::new(0, 0, &[], &[]).err().unwrap();
    assert_eq!(e, Fp8Error::NotTiled { what: "n", value: 0, tile: 32 });
}

#[test]
fn load_rejects_a_nan_scale_by_tile() {
    let (n, k) = (64, 64);
    let mut s = vec![0x7Fu8; 4];
    s[3] = 0xFF;
    let e = Fp8Weights::new(n, k, &[0x38u8; 64 * 64], &s).err().unwrap();
    assert_eq!(e, Fp8Error::ScaleNotFinite { n_tile: 1, k_tile: 1, bits: 0xFF });
    assert!(e.to_string().contains("0xff"));
}

#[test]
fn forward_rejects_zero_rows_and_short_buffers() {
    let Some(st) = gpu() else { return };
    let (n, k) = (64, 64);
    let wt = Fp8Weights::new(n, k, &[0x38u8; 64 * 64], &[0x7Fu8; 4]).unwrap();
    let mut y = DevBuf::zeroed(8 * n * 2).unwrap();

    let x = DevBuf::from_slice(&vec![0u16; 8 * k]).unwrap();
    assert_eq!(wt.forward(&x, &mut y, 0, Dtype::F16, &st).err().unwrap(), Fp8Error::BadRows { m: 0 });

    let x_short = DevBuf::from_slice(&vec![0u16; 2 * k]).unwrap();
    assert_eq!(
        wt.forward(&x_short, &mut y, 8, Dtype::F16, &st).err().unwrap(),
        Fp8Error::BufferTooSmall { what: "x", need: 8 * k * 2, have: 2 * k * 2 }
    );

    let mut y_short = DevBuf::zeroed(2 * n * 2).unwrap();
    assert_eq!(
        wt.forward(&x, &mut y_short, 8, Dtype::F16, &st).err().unwrap(),
        Fp8Error::BufferTooSmall { what: "y", need: 8 * n * 2, have: 2 * n * 2 }
    );

    let mut out_short = DevBuf::zeroed(16).unwrap();
    assert_eq!(
        wt.dequantise(&mut out_short, &st).err().unwrap(),
        Fp8Error::BufferTooSmall { what: "out", need: n * k * 2, have: 16 }
    );
}

// ---------------------------------------------------------------------------------------------------- the bf16 packer

/// Nearest E4M3 code by exhaustive search — the definition [`e4m3_from_f32`] has to reproduce in bit arithmetic.
fn nearest_e4m3(v: f32) -> u8 {
    let mut best = 0u8;
    let mut bd = f64::INFINITY;
    for b in 0u16..256 {
        let b = b as u8;
        if b == 0x7F || b == 0xFF {
            continue;
        }
        let d = (e4m3_to_f32(b) as f64 - v as f64).abs();
        if d < bd {
            bd = d;
            best = b;
        }
    }
    best
}

#[test]
fn e4m3_encoder_is_the_nearest_code_everywhere() {
    // every code point, its midpoints, and a dense sweep of magnitudes: the encoder must pick the nearest representable
    // value, which is what makes the packed compressor weights a rounding of the checkpoint rather than a re-scaling
    for b in 0u16..256 {
        let b = b as u8;
        if b == 0x7F || b == 0xFF {
            continue;
        }
        let v = e4m3_to_f32(b);
        assert_eq!(e4m3_from_f32(v), b, "a code point must encode to itself: {v}");
    }
    let mut checked = 0;
    for i in -40_000i32..=40_000 {
        let v = i as f32 * 0.011;
        let got = e4m3_from_f32(v);
        let want = nearest_e4m3(v);
        // ties may fall either way (the encoder rounds to even); the values must agree
        assert!(
            (e4m3_to_f32(got) - e4m3_to_f32(want)).abs() <= 1e-9 * e4m3_to_f32(want).abs().max(1.0),
            "v = {v}: encoded 0x{got:02x} ({}), nearest is 0x{want:02x} ({})",
            e4m3_to_f32(got),
            e4m3_to_f32(want)
        );
        checked += 1;
    }
    assert!(checked > 80_000);
    // saturation, zero, sign of zero
    assert_eq!(e4m3_from_f32(1e9), 0x7E);
    assert_eq!(e4m3_from_f32(-1e9), 0xFE);
    assert_eq!(e4m3_from_f32(0.0), 0x00);
    assert_eq!(e4m3_from_f32(-0.0), 0x80);
}

#[test]
fn pack_bf16_keeps_every_weight_within_the_tile_rounding() {
    // a 64x64 bf16 matrix with a 5-decade magnitude spread, so the per-tile scales differ
    let (n, k) = (64usize, 64usize);
    let mut r = Rng(0xfeed_face_1234_5678);
    let vals: Vec<f32> = (0..n * k).map(|i| r.uniform(-1.0, 1.0) * 10f32.powi((i / 512) as i32 - 2)).collect();
    let raw: Vec<u8> = vals.iter().flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
    let (w, s) = pack_bf16(n, k, &raw).unwrap();
    assert_eq!((w.len(), s.len()), (n * k, (n / TILE) * (k / TILE)));
    let kt = k / TILE;
    let mut worst = 0f64;
    let mut at = (0usize, 0f32, 0f32);
    for i in 0..n * k {
        let (row, col) = (i / k, i % k);
        let want = f32::from_bits((u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]) as u32) << 16);
        let scale = e8m0_to_f32(s[(row / TILE) * kt + col / TILE]);
        let got = e4m3_to_f32(w[i]) * scale;
        // E4M3 is a float: a tile's step is a sixteenth of the element's own magnitude, floored by the subnormal grid.
        // `|v|/16` is between half and one grid step, so a nearest rounding lands within 1.0 of it (ties included)
        let ulp = (want.abs() * 0.0625).max(scale * 0.001_953_125);
        let e = ((got - want).abs() / ulp) as f64;
        if e > worst {
            worst = e;
            at = (i, want, got);
        }
    }
    assert!(worst <= 1.0 + 1e-4, "the packer must round to the nearest code, worst was {worst} ulps at {at:?}");
    // a zero tile stays zero with a finite scale
    let zeros = vec![0u8; 32 * 32 * 2];
    let (w, s) = pack_bf16(32, 32, &zeros).unwrap();
    assert!(w.iter().all(|&b| b == 0) && s[0] != 0xFF);
}
