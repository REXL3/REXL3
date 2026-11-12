//! MXFP4 (E2M1 x E8M0-per-32) linear, for the DeepSeek-V4.1 DSpark routed experts.
//!
//! # The layout
//!
//! `mtp.{0,1,2}.ffn.experts.e.w1.weight` is `I8 [2304, 2560]`: 5120 values a row, two to a byte. Its scale is
//! `F8_E8M0 [2304, 160]`, i.e. **one scale a row per 32 inputs** -- a finer grid than the fp8 trunk's 32x32 tile, so
//! [`Mxfp4Weights`] indexes scales `[n, k / 32]` and the k loop of the kernel is 32 wide.
//!
//! A byte holds element `2i` in the low nibble and `2i+1` in the high one (`torch.float4_e2m1fn_x2`), the convention
//! [`crate::dsv4_kvq4`]'s q4 plane already decodes. E2M1 is `{0, .5, 1, 1.5, 2, 3, 4, 6}` plus a sign bit; E8M0 is
//! `2^(b - 127)` with `0xFF` = NaN.
//!
//! # What this is for
//!
//! The DSpark MoE is 128 routed experts top-3 per token over `block_size` draft rows. At 5120 x 2304 a stage's routed
//! block is 2.26 GiB as MXFP4, which is why it is stored this way rather than as fp8; the plan of record is to
//! re-quantise it to EXL3 ~2.0 bpw and drop that to ~1.1 GiB, which is why the drafter's expert interface
//! ([`crate::dsv4_mtp_gpu::MtpExperts`]) is a trait rather than a concrete struct.
//!
//! # Arithmetic
//!
//! fp16 activations, fp32 accumulate, fp16 out -- the same contract as [`crate::fp8_linear`], so a segment of one
//! expert can be handed to either. `--fmad=false` in the .cu keeps the non-fma products as two roundings.

use crate::exl3_linear::Dtype;
use crate::rt::{CudaError, DevBuf, Stream};

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_fp4_linear(x: *const c_void, x_dtype: i32, w: *const u8, s: *const u8, y: *mut c_void,
                             m: i32, k: i32, n: i32, stream: *mut c_void) -> i32;
        pub fn gb_fp4_dequant(w: *const u8, s: *const u8, out: *mut c_void, n: i32, k: i32, stream: *mut c_void) -> i32;
    }
}

/// The scale tile: 32 inputs per scale byte.
pub const BLOCK: usize = 32;

/// One E2M1 nibble as f32.
pub fn e2m1_to_f32(nib: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[(nib & 7) as usize];
    if nib & 8 != 0 {
        -v
    } else {
        v
    }
}

/// The E8M0 byte as f32: `2^(b - 127)`.
pub fn e8m0_to_f32(b: u8) -> f32 {
    crate::fp8_linear::e8m0_to_f32(b)
}

/// `[n, k]` f32 -> the pack's MXFP4 bytes and E8M0 scales, the way the checkpoint was produced: `amax` a 32-input
/// block, `s = 2^ceil(log2(max(amax, 1e-4) / 6))`, `q = round_e2m1(x / s)`. A fixture builder, not a serving path.
pub fn pack_f32(n: usize, k: usize, x: &[f32]) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(x.len(), n * k, "x is [n, k]");
    assert_eq!(k % BLOCK, 0, "k must be a multiple of {BLOCK}");
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let kt = k / BLOCK;
    let mut w = vec![0u8; n * k / 2];
    let mut s = vec![0u8; n * kt];
    for r in 0..n {
        for b in 0..kt {
            let blk = &x[r * k + b * BLOCK..][..BLOCK];
            let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-4);
            let e = (amax / 6.0).log2().ceil();
            let sc = e.exp2();
            s[r * kt + b] = ((e + 127.0) as i32).clamp(1, 254) as u8;
            for (i, &v) in blk.iter().enumerate() {
                let q = (v / sc).clamp(-6.0, 6.0).abs();
                let mut best = 0usize;
                let mut bd = f32::INFINITY;
                for (c, &m) in MAG.iter().enumerate() {
                    let d = (q - m).abs();
                    if d < bd {
                        bd = d;
                        best = c;
                    }
                }
                let nib = (best as u8) | if v < 0.0 { 8 } else { 0 };
                let byte = r * (k / 2) + b * (BLOCK / 2) + i / 2;
                w[byte] |= if i % 2 == 0 { nib } else { nib << 4 };
            }
        }
    }
    (w, s)
}

/// The host view of the layout: `[n, k]` f32 from the packed bytes and scales. This is the reference the GPU GEMM is
/// tested against, written from the format above rather than from the kernel.
pub fn dequant_host(n: usize, k: usize, w: &[u8], s: &[u8]) -> Vec<f32> {
    assert_eq!(w.len(), n * k / 2, "w is [n, k/2]");
    let kt = k / BLOCK;
    assert_eq!(s.len(), n * kt, "s is [n, k/32]");
    let mut out = vec![0f32; n * k];
    for r in 0..n {
        for c in 0..k {
            let byte = w[r * (k / 2) + c / 2];
            let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out[r * k + c] = e2m1_to_f32(nib) * e8m0_to_f32(s[r * kt + c / BLOCK]);
        }
    }
    out
}

/// An `[n, k]` MXFP4 weight and its `[n, k/32]` E8M0 scale, owned on the device.
#[derive(Debug)]
pub struct Mxfp4Weights {
    w: DevBuf,
    s: DevBuf,
    n: usize,
    k: usize,
}

impl Mxfp4Weights {
    /// Upload `w` (`[n, k/2]` bytes, two E2M1 a byte) and `s` (`[n, k/32]` E8M0), validating both.
    pub fn new(n: usize, k: usize, w: &[u8], s: &[u8]) -> Result<Self, CudaError> {
        if n == 0 || k < BLOCK || k % BLOCK != 0 || n % BLOCK != 0 {
            return Err(CudaError(1));
        }
        if w.len() != n * k / 2 || s.len() != n * (k / BLOCK) {
            return Err(CudaError(1));
        }
        Ok(Self { w: DevBuf::from_slice(w)?, s: DevBuf::from_slice(s)?, n, k })
    }

    /// `[n, k]`.
    pub fn shape(&self) -> (usize, usize) {
        (self.n, self.k)
    }

    /// Bytes of the device weight, scales included -- what admission charges for it.
    pub fn weight_bytes(&self) -> usize {
        self.n * self.k / 2 + self.n * (self.k / BLOCK)
    }

    /// The raw device pointers, for a grouped kernel that indexes experts itself.
    pub fn parts(&self) -> (*const u8, *const u8) {
        (self.w.ptr() as *const u8, self.s.ptr() as *const u8)
    }

    /// `y [m, n] = x [m, k] @ W^T`, fp16 out, asynchronous on `st`. Decode shape for `m <= 8`, row tiles of 32 above.
    pub fn forward(&self, x: &DevBuf, y: &mut DevBuf, m: usize, x_dtype: Dtype, st: &Stream) -> Result<(), CudaError> {
        if m == 0 || m > 128 {
            return Err(CudaError(1));
        }
        if x.bytes() < m * self.k * x_dtype.size() || y.bytes() < m * self.n * 2 {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_fp4_linear(
                x.ptr(),
                x_dtype as i32,
                self.w.ptr() as *const u8,
                self.s.ptr() as *const u8,
                y.ptr(),
                m as i32,
                self.k as i32,
                self.n as i32,
                st.raw(),
            ))
        }
    }

    /// The dequantised weight `[n, k]` in fp16, for tests and for a debug dump.
    pub fn dequantise(&self, out: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
        if out.bytes() < self.n * self.k * 2 {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_fp4_dequant(
                self.w.ptr() as *const u8,
                self.s.ptr() as *const u8,
                out.ptr(),
                self.n as i32,
                self.k as i32,
                st.raw(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two decoders agree on every code point, so a bit-pattern question is never a kernel question.
    #[test]
    fn every_nibble_decodes_to_its_table_value() {
        const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for nib in 0u8..16 {
            let want = MAG[(nib & 7) as usize] * if nib & 8 != 0 { -1.0 } else { 1.0 };
            assert_eq!(e2m1_to_f32(nib), want, "nibble {nib:#x}");
        }
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(126), 0.5);
        assert_eq!(e8m0_to_f32(0), 2f32.powi(-127));
    }

    /// A packed weight round-trips to within one E2M1 step of its block scale, and the layout is row-major with the
    /// even element in the low nibble.
    #[test]
    fn packing_then_decoding_is_within_half_a_step() {
        let (n, k) = (32, 64);
        let mut x = vec![0f32; n * k];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i as f32 * 0.37).sin()) * 3.0;
        }
        let (w, s) = pack_f32(n, k, &x);
        assert_eq!(w.len(), n * k / 2);
        assert_eq!(s.len(), n * (k / BLOCK));
        let back = dequant_host(n, k, &w, &s);
        let kt = k / BLOCK;
        for r in 0..n {
            for b in 0..kt {
                let sc = e8m0_to_f32(s[r * kt + b]);
                for c in 0..BLOCK {
                    let got = back[r * k + b * BLOCK + c];
                    let want = x[r * k + b * BLOCK + c];
                    // E2M1's widest gap is 2 codes (4 -> 6), so one rounding is at most one scale out
                    assert!((got - want).abs() <= sc + 1e-6, "r{r} c{c}: {got} vs {want} (s {sc})");
                }
            }
        }
        // the scales are powers of two in [1, 254] (byte 0 is a subnormal, 0xFF is NaN)
        for &b in &s {
            assert!((1..=254).contains(&b), "scale byte {b}");
        }
    }

    /// The nibble order is fixed by a hand-built block: `[1, -2, 0.5, .., 6]` has `amax` 6, so scale 1, and packs as
    /// the codes 2, 12, 1 -- the even element in the low nibble.
    #[test]
    fn the_even_element_is_the_low_nibble() {
        let (n, k) = (32, 32);
        let mut x = vec![0f32; n * k];
        x[0] = 1.0;
        x[1] = -2.0;
        x[2] = 0.5;
        x[3] = 6.0;
        let (w, s) = pack_f32(n, k, &x);
        assert_eq!(s[0], 127, "amax 6 is scale 1.0");
        assert_eq!(w[0] & 0x0F, 0x2, "1.0 is code 2, in the low nibble");
        assert_eq!(w[0] >> 4, 0xC, "-2.0 is magnitude code 4 | sign, in the high nibble");
        assert_eq!(w[1] & 0x0F, 0x1, "0.5 is code 1");
        assert_eq!(w[1] >> 4, 0x7, "6.0 is code 7");
    }
}
