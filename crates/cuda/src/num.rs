//! Host-side numeric helpers shared by every kernel wrapper and test: bf16 conversion with the same rounding the
//! kernels use (round to nearest even), and fp16 via `gb_exl3::f16`.

/// bf16 bits of an f32, round to nearest even (NaN stays NaN).
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x0040) as u16;
    }
    let lsb = (b >> 16) & 1;
    (b.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

pub fn bf16_vec(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f32_to_bf16(x)).collect()
}

pub fn bf16_to_f32_vec(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| bf16_to_f32(b)).collect()
}

/// Round an f32 to bf16 precision and back (what a bf16 store does to a value).
pub fn round_bf16(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        // halfway between two bf16 values rounds to the even mantissa
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8001)), 0x3F81);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
        assert_eq!(f32_to_bf16(-0.0), 0x8000);
        assert_eq!(round_bf16(std::f32::consts::PI), 3.140625);
    }
}
