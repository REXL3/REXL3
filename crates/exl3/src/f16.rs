//! IEEE binary16 conversions without dependencies. `f64_to_f16` rounds once, to nearest even,
//! directly from float64 (never through f32, which would round twice). Checked against numpy's
//! `astype(np.float16)` by the fixtures in `tests/xcheck.rs`.

/// Exact fp16 -> f64.
pub fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f64;
    match e {
        0 => sign * m * (2.0f64).powi(-24),
        31 => {
            if m == 0.0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => sign * (1024.0 + m) * (2.0f64).powi(e - 25),
    }
}

/// f64 -> fp16 bits, round to nearest, ties to even; overflow to infinity.
pub fn f64_to_f16(x: f64) -> u16 {
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    if x.is_nan() {
        return sign | 0x7E00;
    }
    let a = x.abs();
    if a >= 65520.0 {
        return sign | 0x7C00; // 65520 is the tie between 65504 (odd) and 65536: rounds to infinity
    }
    let e = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023; // a is 0 or a normal f64 here
    if a == 0.0 || e < -14 {
        // subnormal range: units of 2^-24
        let m = (a * (2.0f64).powi(24)).round_ties_even() as u16; // exact scaling, one rounding
        return sign | m; // m == 1024 is exactly the smallest normal's bit pattern
    }
    let mut m = (a * (2.0f64).powi(10 - e)).round_ties_even() as u32; // in [1024, 2048]
    let mut e = e;
    if m == 2048 {
        m = 1024;
        e += 1;
    }
    if e > 15 {
        return sign | 0x7C00;
    }
    sign | (((e + 15) as u16) << 10) | (m - 1024) as u16
}

/// `table[h] == f64_to_f16(f16_to_f64(h))` for every fp16 bit pattern: what the checkpoint's fp16 scale vectors become on
/// the way through `f64` (every pattern but a NaN is its own image; every NaN turns into the canonical positive quiet NaN 0x7E00, since `f16_to_f64` returns `f64::NAN`).
/// The loader maps raw fp16 bytes through this table, which is exactly that round trip, 65,536 conversions once instead of one
/// per element.
pub fn canonical_f16_table() -> &'static [u16; 65536] {
    static TABLE: std::sync::OnceLock<Box<[u16; 65536]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([0u16; 65536]);
        for (h, slot) in t.iter_mut().enumerate() {
            *slot = f64_to_f16(f16_to_f64(h as u16));
        }
        t
    })
}

/// bf16 bits of `x`, rounded once to nearest, ties to even (no double rounding through f32: a value of exactly half a bf16 step between
/// two bf16 values rounds to the even one, whatever its float64 bits below the f32 grid are). NaN keeps its sign (quiet), infinities and
/// overflow past the largest bf16 are infinite, and the bf16 subnormals (down to 2^-133) are honoured.
pub fn f64_to_bf16(x: f64) -> u16 {
    let sign = ((x.to_bits() >> 48) & 0x8000) as u16;
    if x.is_nan() {
        return sign | 0x7fc0;
    }
    if x.is_infinite() {
        return sign | 0x7f80;
    }
    let a = x.abs();
    let e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    if e < -134 {
        return sign; // below half the smallest bf16 subnormal (f64 subnormals included): zero
    }
    // the bf16 step at this magnitude: 2^(e - 7) for a normal, 2^-133 below 2^-126; `a / q` is exact (a power-of-two scale)
    let qe = (e - 7).max(-133);
    let q = f64::from_bits(((qe + 1023) as u64) << 52);
    let v = (a / q).round_ties_even() * q;
    // `v` has at most 8 significant bits on the bf16 grid: exact in f32 (or past f32's range: infinite), its low 16 bits zero
    sign | ((v as f32).to_bits() >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_table_is_identity_except_nan_payloads() {
        let t = canonical_f16_table();
        for h in 0u16..=0xFFFF {
            let is_nan = (h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0;
            if is_nan {
                assert_eq!(t[h as usize], 0x7E00, "NaN {h:#06x}");
            } else {
                assert_eq!(t[h as usize], h, "h={h:#06x}");
            }
        }
    }

    #[test]
    fn roundtrip_all_finite() {
        for h in 0u16..=0xFFFF {
            let v = f16_to_f64(h);
            if v.is_nan() {
                continue;
            }
            let back = f64_to_f16(v);
            // +0/-0 keep their sign
            assert_eq!(back, h, "h={h:#06x} v={v}");
        }
    }

    #[test]
    fn ties_to_even() {
        // 1 + 2^-11 is halfway between 1.0 (even) and 1 + 2^-10: rounds down
        assert_eq!(f64_to_f16(1.0 + (2.0f64).powi(-11)), 0x3C00);
        // 1 + 3*2^-11 is halfway between 1+2^-10 (odd) and 1+2^-9: rounds up
        assert_eq!(f64_to_f16(1.0 + 3.0 * (2.0f64).powi(-11)), 0x3C02);
        assert_eq!(f64_to_f16(65504.0), 0x7BFF);
        assert_eq!(f64_to_f16(65519.99), 0x7BFF);
        assert_eq!(f64_to_f16(65520.0), 0x7C00);
        assert_eq!(f64_to_f16((2.0f64).powi(-25)), 0x0000); // tie with 2^-24: to even (0)
        assert_eq!(f64_to_f16(1.5 * (2.0f64).powi(-24)), 0x0002); // tie 1 vs 2 units: to even
    }

    #[test]
    fn f64_to_bf16_rounds_once_to_nearest_even() {
        let bf = |b: u16| f32::from_bits((b as u32) << 16) as f64;
        // exact bf16 values, signs, zeros, infinities, NaN, overflow and the subnormals
        for b in [0x0000u16, 0x8000, 0x3f80, 0xbf80, 0x4049, 0x7f7f, 0x0001, 0x8001, 0x007f, 0x0080] {
            assert_eq!(f64_to_bf16(bf(b)), b, "{b:#06x}");
        }
        assert_eq!(f64_to_bf16(f64::INFINITY), 0x7f80);
        assert_eq!(f64_to_bf16(f64::NEG_INFINITY), 0xff80);
        assert_eq!(f64_to_bf16(f64::NAN) & 0x7fc0, 0x7fc0);
        assert_eq!(f64_to_bf16(3.5e38), 0x7f80, "past the largest bf16");
        assert_eq!(f64_to_bf16(2f64.powi(-134)), 0x0000, "half the smallest subnormal ties to even (zero)");
        assert_eq!(f64_to_bf16(1.5 * 2f64.powi(-133)), 0x0002, "1.5 subnormal steps ties to even (2)");
        // a tie in bf16 rounds to even; just above the tie (below the f32 grid: f32 would round it to the tie) rounds up
        let one = 1.0f64;
        let half_step = 2f64.powi(-8);
        assert_eq!(f64_to_bf16(one + half_step), 0x3f80, "1 + half a step: tie, to even (1.0)");
        assert_eq!(f64_to_bf16(one + half_step + 2f64.powi(-40)), 0x3f81, "just past the tie: up, no double rounding");
        assert_eq!(f64_to_bf16(one + 3.0 * half_step), 0x3f82, "1 + 1.5 steps: tie, to even (1 + 2 steps)");
        // against a brute-force nearest-even search over the bf16 neighbours of random values
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..20_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let x = f64::from_bits((s & 0x800f_ffff_ffff_ffff) | ((1023 - 40 + (s >> 52) % 80) << 52));
            let b = f64_to_bf16(x);
            let (lo, hi) = (bf(b.wrapping_sub(1)), bf(b.wrapping_add(1)));
            let (d, dlo, dhi) = ((bf(b) - x).abs(), (lo - x).abs(), (hi - x).abs());
            assert!(d <= dlo && d <= dhi, "{x:e}: {b:#06x}");
            if d == dlo || d == dhi {
                assert_eq!(b & 1, 0, "{x:e}: a tie goes to the even neighbour");
            }
        }
    }
}
