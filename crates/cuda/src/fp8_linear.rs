//! An FP8 E4M3 block-scaled linear layer: `y = x @ W^T` with a per-32x32-tile E8M0 scale (`csrc/fp8_linear.cu`).
//!
//! # The format
//!
//! `W` is `[N, K]` F8_E4M3, one byte per element, and `scale` is `[N/32, K/32]` F8_E8M0, so `scale[i][j]` covers rows
//! `32 i .. 32 i + 32` and columns `32 j .. 32 j + 32` of `W`. E8M0 is **exponent only**: the byte `b` is an unsigned
//! IEEE-754 exponent field with no sign and no mantissa, so
//!
//! ```text
//! dequant(w, b) = e4m3(w) * 2^(b - 127),   b = 0xFF is NaN (OCP MX)
//! ```
//!
//! `b = 0` is `2^-127`, a normal `f32` subnormal — an exponent-field insert (`from_bits(b << 23)`) would return 0.0
//! for it, so [`e8m0_to_f32`] goes through `powi` and is exact for every `b`.
//!
//! # Two paths, one kernel
//!
//! * [`Fp8Weights::forward`] with `m <= ` [`MAX_DECODE_ROWS`] (8): the decode shape, one block column per 32 output
//!   columns and no k split, so a row's sum is formed in one fixed order.
//! * [`Fp8Weights::forward`] with more rows: the same kernel at 32 rows per block.
//!
//! Both accumulate fp32 and write fp16. Neither splits k (no atomics, no workspace, no reduction order that depends on
//! `m`), so an output element is bit-identical whatever the row count of the call is: a decode row recomputed inside
//! a prefill batch gives the same bits.
//!
//! # Loading
//!
//! [`Fp8Weights::new`] takes the host bytes of both tensors exactly as they come out of the safetensors file (no
//! transposition, no repacking: the kernel indexes `W[n][k]` row major), validates the shapes and refuses a `0xFF`
//! (NaN) scale byte, so a bad checkpoint fails at load, named, and never as a silent NaN in a model's activations.

use crate::exl3_linear::Dtype;
use crate::rt::{CudaError, DevBuf, Stream};

/// Rows above which [`Fp8Weights::forward`] takes the prefill shape (32 rows per block).
pub const MAX_DECODE_ROWS: usize = 16;

/// The scale tile: `scale[i][j]` covers `32 i .. 32 i + 32` rows and `32 j .. 32 j + 32` columns of the weight.
pub const TILE: usize = 32;

/// `cudaErrorInvalidValue`, the code the kernels return for a shape they do not take.
#[allow(dead_code)]
const INVALID: CudaError = CudaError(1);

/// One E8M0 byte as an `f32` scale: `2^(b - 127)`, `0xFF` is NaN. Exact for every `b` (including `b = 0`, which is
/// the subnormal `2^-127` and not zero).
pub fn e8m0_to_f32(bits: u8) -> f32 {
    if bits == 0xFF {
        f32::NAN
    } else {
        2.0f32.powi(bits as i32 - 127)
    }
}

/// One E4M3 byte (no scale) as an `f32`: sign, 4-bit exponent biased by 7, 3-bit mantissa, `0x7F`/`0xFF` NaN.
pub fn e4m3_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 3) & 0x0F) as i32;
    let man = (bits & 0x07) as f32;
    let v = if exp == 0 {
        man / 8.0 * 2.0f32.powi(-6)
    } else if exp == 0x0F && man == 7.0 {
        return f32::NAN;
    } else {
        (1.0 + man / 8.0) * 2.0f32.powi(exp - 7)
    };
    sign * v
}

/// Why a [`Fp8Weights`] could not be built, or an [`Fp8Weights::forward`] refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fp8Error {
    /// A tensor's element count is not the one its declared shape needs.
    ShapeMismatch { what: &'static str, expected: usize, got: usize },
    /// `n` or `k` is zero or not a multiple of the 32-wide scale tile.
    NotTiled { what: &'static str, value: usize, tile: usize },
    /// A scale byte is `0xFF` (E8M0 NaN): the checkpoint is damaged, not merely imprecise.
    ScaleNotFinite { n_tile: usize, k_tile: usize, bits: u8 },
    /// `m` is zero, or a device buffer is too small for `m` rows.
    BadRows { m: usize },
    /// A device buffer is smaller than the `m` rows need.
    BufferTooSmall { what: &'static str, need: usize, have: usize },
    /// The device call failed.
    Cuda(CudaError),
}

impl std::fmt::Display for Fp8Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fp8Error::ShapeMismatch { what, expected, got } => {
                write!(f, "{what}: expected {expected} elements, got {got}")
            }
            Fp8Error::NotTiled { what, value, tile } => {
                write!(f, "{what} = {value} is not a positive multiple of {tile}")
            }
            Fp8Error::ScaleNotFinite { n_tile, k_tile, bits } => {
                write!(f, "scale[{n_tile}][{k_tile}] = 0x{bits:02x} is E8M0 NaN")
            }
            Fp8Error::BadRows { m } => write!(f, "row count {m} is not >= 1"),
            Fp8Error::BufferTooSmall { what, need, have } => {
                write!(f, "{what} holds {have} bytes, the call needs {need}")
            }
            Fp8Error::Cuda(e) => write!(f, "cuda: {e}"),
        }
    }
}

impl std::error::Error for Fp8Error {}

impl From<CudaError> for Fp8Error {
    fn from(e: CudaError) -> Self {
        Fp8Error::Cuda(e)
    }
}

#[cfg(not(gb_no_cuda))]
mod ffi {
    use core::ffi::c_void;
    extern "C" {
        pub fn gb_fp8_linear(
            x: *const c_void,
            x_dtype: i32,
            w: *const u8,
            s: *const u8,
            y: *mut c_void,
            m: i32,
            k: i32,
            n: i32,
            force_mma: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_fp8_linear_pair(
            x: *const c_void,
            x_dtype: i32,
            w1: *const u8,
            s1: *const u8,
            y1: *mut c_void,
            n1: i32,
            w2: *const u8,
            s2: *const u8,
            y2: *mut c_void,
            n2: i32,
            m: i32,
            k: i32,
            force_mma: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_fp8_dequant(w: *const u8, s: *const u8, out: *mut c_void, n: i32, k: i32, stream: *mut c_void) -> i32;
        /// Test-only classic-path entry points (the production [`Self::forward`] path is the pipelined
        /// small-M kernel; these exist so the A/B tests can compare new vs classic bit-for-bit without any
        /// env switch).
        pub fn gb_fp8_linear_classic(
            x: *const c_void,
            x_dtype: i32,
            w: *const u8,
            s: *const u8,
            y: *mut c_void,
            m: i32,
            k: i32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_fp8_linear_pair_classic(
            x: *const c_void,
            x_dtype: i32,
            w1: *const u8,
            s1: *const u8,
            y1: *mut c_void,
            n1: i32,
            w2: *const u8,
            s2: *const u8,
            y2: *mut c_void,
            n2: i32,
            m: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// An `[N, K]` E4M3 weight and its `[N/32, K/32]` E8M0 scale, owned on the device (or an EXL3 `[K, N]` group).
#[derive(Debug)]
pub struct Fp8Weights {
    w: DevBuf,
    s: DevBuf,
    n: usize,
    k: usize,
    exl3: Option<(crate::exl3_linear::Exl3Weights, std::sync::Mutex<DevBuf>)>,
}

impl Fp8Weights {
    /// Wrap an uploaded [`crate::exl3_linear::Exl3Weights`] (`k` inputs, `n` outputs) behind the `Fp8Weights` interface.
    pub fn from_exl3(exl: crate::exl3_linear::Exl3Weights) -> Result<Self, Fp8Error> {
        let (k, n) = (exl.k, exl.n);
        let ws_b = exl.workspace_bytes(crate::exl3_linear::MAX_DECODE_ROWS).max(256);
        let ws = DevBuf::alloc(ws_b)?;
        Ok(Self { w: DevBuf::alloc(16)?, s: DevBuf::alloc(16)?, n, k, exl3: Some((exl, std::sync::Mutex::new(ws))) })
    }

    /// Upload `w` (`[n, k]`, one byte per element) and `s` (`[n/32, k/32]` E8M0 bytes), validating both.
    pub fn new(n: usize, k: usize, w: &[u8], s: &[u8]) -> Result<Self, Fp8Error> {
        if n == 0 || k == 0 || n % TILE != 0 {
            return Err(Fp8Error::NotTiled { what: "n", value: n, tile: TILE });
        }
        if k % TILE != 0 {
            return Err(Fp8Error::NotTiled { what: "k", value: k, tile: TILE });
        }
        if w.len() != n * k {
            return Err(Fp8Error::ShapeMismatch { what: "w [n, k]", expected: n * k, got: w.len() });
        }
        let (nt, kt) = (n / TILE, k / TILE);
        if s.len() != nt * kt {
            return Err(Fp8Error::ShapeMismatch { what: "scale [n/32, k/32]", expected: nt * kt, got: s.len() });
        }
        if let Some(i) = s.iter().position(|&b| b == 0xFF) {
            return Err(Fp8Error::ScaleNotFinite { n_tile: i / kt, k_tile: i % kt, bits: 0xFF });
        }
        Ok(Self { w: DevBuf::from_slice(w)?, s: DevBuf::from_slice(s)?, n, k, exl3: None })
    }

    /// `[N, K]`.
    pub fn shape(&self) -> (usize, usize) {
        (self.n, self.k)
    }

    /// Bytes of the device weight.
    pub fn weight_bytes(&self) -> usize {
        if let Some((exl, _)) = &self.exl3 {
            exl.weight_bytes()
        } else {
            self.n * self.k
        }
    }

    fn check_io(&self, x: &DevBuf, y: &DevBuf, m: usize, x_dtype: Dtype) -> Result<(), Fp8Error> {
        if m == 0 {
            return Err(Fp8Error::BadRows { m });
        }
        let x_need = m * self.k * x_dtype.size();
        if x.bytes() < x_need {
            return Err(Fp8Error::BufferTooSmall { what: "x", need: x_need, have: x.bytes() });
        }
        let y_need = m * self.n * 2; // fp16 output
        if y.bytes() < y_need {
            return Err(Fp8Error::BufferTooSmall { what: "y", need: y_need, have: y.bytes() });
        }
        Ok(())
    }

    /// `y [m, n] = layer(x [m, k])`, fp16 out, asynchronous on `st`. Decode shape for `m <= MAX_DECODE_ROWS`, prefill
    /// shape above it; both are the same arithmetic in the same order, so an element's bits do not depend on `m`.
    pub fn forward(&self, x: &DevBuf, y: &mut DevBuf, m: usize, x_dtype: Dtype, st: &Stream) -> Result<(), Fp8Error> {
        self.forward_impl(x, y, m, x_dtype, false, st)
    }

    /// Prefill-slice variant: forces the tensor-core MMA path even for `m <= 16`, so a row's bits do
    /// not depend on the prefill chunk grid (decode keeps the pipelined small-M kernels).
    pub fn forward_force(&self, x: &DevBuf, y: &mut DevBuf, m: usize, x_dtype: Dtype, st: &Stream) -> Result<(), Fp8Error> {
        self.forward_impl(x, y, m, x_dtype, true, st)
    }

    fn forward_impl(
        &self,
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        x_dtype: Dtype,
        force_mma: bool,
        st: &Stream,
    ) -> Result<(), Fp8Error> {
        self.check_io(x, y, m, x_dtype)?;
        if let Some((exl, ws_mu)) = &self.exl3 {
            let mut ws = ws_mu.lock().unwrap();
            exl.forward_chunked(x, y, m, x_dtype, Dtype::F16, &mut ws, st)?;
            return Ok(());
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_fp8_linear(
                x.ptr(),
                x_dtype as i32,
                self.w.ptr() as *const u8,
                self.s.ptr() as *const u8,
                y.ptr(),
                m as i32,
                self.k as i32,
                self.n as i32,
                i32::from(force_mma),
                st.raw(),
            ))?;
            Ok(())
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (x, y, m, x_dtype, force_mma, st);
            Err(Fp8Error::Cuda(INVALID))
        }
    }

    /// Fused `y1 [m, n1] = w1(x [m, k])` and `y2 [m, n2] = w2(x [m, k])` in a single kernel launch at decode shapes.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_pair(
        w1: &Self,
        y1: &mut DevBuf,
        w2: &Self,
        y2: &mut DevBuf,
        x: &DevBuf,
        m: usize,
        x_dtype: Dtype,
        st: &Stream,
    ) -> Result<(), Fp8Error> {
        Self::forward_pair_impl(w1, y1, w2, y2, x, m, x_dtype, false, st)
    }

    /// Prefill-slice variant of [`Self::forward_pair`]: forces the MMA path even for `m <= 16`.
    pub fn forward_pair_force(
        w1: &Self,
        y1: &mut DevBuf,
        w2: &Self,
        y2: &mut DevBuf,
        x: &DevBuf,
        m: usize,
        x_dtype: Dtype,
        st: &Stream,
    ) -> Result<(), Fp8Error> {
        Self::forward_pair_impl(w1, y1, w2, y2, x, m, x_dtype, true, st)
    }

    fn forward_pair_impl(
        w1: &Self,
        y1: &mut DevBuf,
        w2: &Self,
        y2: &mut DevBuf,
        x: &DevBuf,
        m: usize,
        x_dtype: Dtype,
        force_mma: bool,
        st: &Stream,
    ) -> Result<(), Fp8Error> {
        if w1.k != w2.k {
            return Err(Fp8Error::ShapeMismatch { what: "k (pair)", expected: w1.k, got: w2.k });
        }
        if w1.exl3.is_some() || w2.exl3.is_some() {
            w1.forward(x, y1, m, x_dtype, st)?;
            return w2.forward(x, y2, m, x_dtype, st);
        }
        w1.check_io(x, y1, m, x_dtype)?;
        w2.check_io(x, y2, m, x_dtype)?;
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_fp8_linear_pair(
                x.ptr(),
                x_dtype as i32,
                w1.w.ptr() as *const u8,
                w1.s.ptr() as *const u8,
                y1.ptr(),
                w1.n as i32,
                w2.w.ptr() as *const u8,
                w2.s.ptr() as *const u8,
                y2.ptr(),
                w2.n as i32,
                m as i32,
                w1.k as i32,
                i32::from(force_mma),
                st.raw(),
            ))?;
            Ok(())
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (w1, y1, w2, y2, x, m, x_dtype, force_mma, st);
            Err(Fp8Error::Cuda(INVALID))
        }
    }

    /// Test-only reference path: the classic (non-pipelined) decode kernel, bit-for-bit the arithmetic the
    /// production [`Self::forward`] is required to reproduce. The GPU tests use this instead of env switches.
    pub fn forward_classic(&self, x: &DevBuf, y: &mut DevBuf, m: usize, x_dtype: Dtype, st: &Stream) -> Result<(), Fp8Error> {
        self.check_io(x, y, m, x_dtype)?;
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_fp8_linear_classic(
                x.ptr(),
                x_dtype as i32,
                self.w.ptr() as *const u8,
                self.s.ptr() as *const u8,
                y.ptr(),
                m as i32,
                self.k as i32,
                self.n as i32,
                st.raw(),
            ))?;
            Ok(())
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (x, y, m, x_dtype, st);
            Err(Fp8Error::Cuda(INVALID))
        }
    }

    /// Test-only reference path for [`Self::forward_pair`].
    #[allow(clippy::too_many_arguments)]
    pub fn forward_pair_classic(
        w1: &Self,
        y1: &mut DevBuf,
        w2: &Self,
        y2: &mut DevBuf,
        x: &DevBuf,
        m: usize,
        x_dtype: Dtype,
        st: &Stream,
    ) -> Result<(), Fp8Error> {
        if w1.k != w2.k {
            return Err(Fp8Error::ShapeMismatch { what: "k (pair)", expected: w1.k, got: w2.k });
        }
        w1.check_io(x, y1, m, x_dtype)?;
        w2.check_io(x, y2, m, x_dtype)?;
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_fp8_linear_pair_classic(
                x.ptr(),
                x_dtype as i32,
                w1.w.ptr() as *const u8,
                w1.s.ptr() as *const u8,
                y1.ptr(),
                w1.n as i32,
                w2.w.ptr() as *const u8,
                w2.s.ptr() as *const u8,
                y2.ptr(),
                w2.n as i32,
                m as i32,
                w1.k as i32,
                st.raw(),
            ))?;
            Ok(())
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (w1, y1, w2, y2, x, m, x_dtype, st);
            Err(Fp8Error::Cuda(INVALID))
        }
    }

    /// The dequantised weight `[n, k]` in fp16 (the tests' and a debug path's view of the layout), asynchronous on
    /// `st`.
    pub fn dequantise(&self, out: &mut DevBuf, st: &Stream) -> Result<(), Fp8Error> {
        let need = self.n * self.k * 2;
        if out.bytes() < need {
            return Err(Fp8Error::BufferTooSmall { what: "out", need, have: out.bytes() });
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_fp8_dequant(
                self.w.ptr() as *const u8,
                self.s.ptr() as *const u8,
                out.ptr(),
                self.n as i32,
                self.k as i32,
                st.raw(),
            ))?;
            Ok(())
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (out, st);
            Err(Fp8Error::Cuda(INVALID))
        }
    }
}

/// The E4M3 code point nearest `x` (round to nearest, ties away), saturating at ±448.
///
/// Bit arithmetic rather than a search over the 256 codes: the pack's auxiliary matrices (the compressor's `wkv`/`wgate`,
/// the indexer's `wk`/`weights_proj`) arrive as BF16 and are quantised here at load, which is 2 x 512 x 5120 elements a
/// layer — a code search would be 10^9 comparisons at boot.
pub fn e4m3_from_f32(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    if x.is_nan() {
        return sign | 0x7F;
    }
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    if a.is_infinite() {
        return sign | 0x7E;
    }
    let bits = a.to_bits();
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    if e < -6 {
        // subnormal range: steps of 2^-9, and 8 * 2^-9 is the smallest normal (0x08), so the top of this range encodes
        // as that code rather than saturating
        let q = (a * 512.0).round() as u8;
        return sign | q;
    }
    let mut man = ((bits >> 20) & 0x7) as i32;
    let rem = bits & 0x000F_FFFF;
    // nearest, ties to even: half of the dropped 20 bits is 0x8_0000
    if rem > 0x8_0000 || (rem == 0x8_0000 && man & 1 == 1) {
        man += 1;
    }
    let mut exp = e + 7;
    if man == 8 {
        man = 0;
        exp += 1;
    }
    if exp > 15 || (exp == 15 && man > 6) {
        // 448 is the largest finite code (0x7E); 0x7F is NaN
        return sign | 0x7E;
    }
    sign | ((exp as u8) << 3) | man as u8
}

/// Quantise an `[n, k]` BF16 matrix (little-endian bytes, row-major) to the pack's FP8 block layout: E4M3 elements with
/// one E8M0 scale per 32x32 tile, the scale being `2^ceil(log2(amax / 448))` so the tile's largest magnitude lands just
/// inside the code range.
///
/// The compressor and the indexer keep their weights in BF16 in the checkpoint (`model-*.safetensors` carries no scales
/// for them); the device path runs them through [`Fp8Weights`] like every other projection, so this is where that
/// conversion happens. It is the only lossy step of the bind — everything downstream is exact.
pub fn pack_bf16(n: usize, k: usize, raw: &[u8]) -> Result<(Vec<u8>, Vec<u8>), Fp8Error> {
    if n == 0 || k == 0 || n % TILE != 0 || k % TILE != 0 {
        return Err(Fp8Error::NotTiled { what: "bf16 [n, k]", value: n.min(k), tile: TILE });
    }
    if raw.len() != n * k * 2 {
        return Err(Fp8Error::ShapeMismatch { what: "bf16 [n, k]", expected: n * k * 2, got: raw.len() });
    }
    let bf16 = |i: usize| f32::from_bits((u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]) as u32) << 16);
    let (nt, kt) = (n / TILE, k / TILE);
    let mut bytes = vec![0u8; n * k];
    let mut scales = vec![0u8; nt * kt];
    for ni in 0..nt {
        for ki in 0..kt {
            let mut amax = 0f32;
            for r in 0..TILE {
                for c in 0..TILE {
                    amax = amax.max(bf16((ni * TILE + r) * k + ki * TILE + c).abs());
                }
            }
            // an all-zero tile takes the smallest exponent; its elements are all zero and any scale gives zero
            let e = if amax > 0.0 { (amax / 448.0).log2().ceil() } else { -127.0 };
            let s = 2.0f32.powi(e as i32);
            scales[ni * kt + ki] = (e as i32 + 127).clamp(0, 254) as u8;
            for r in 0..TILE {
                for c in 0..TILE {
                    let i = (ni * TILE + r) * k + ki * TILE + c;
                    bytes[i] = e4m3_from_f32(bf16(i) / s);
                }
            }
        }
    }
    Ok((bytes, scales))
}

/// The host reference for one output element, over the packed bytes: `sum_k e4m3(w[n][k]) * 2^(b - 127) * x[m][k]` in
/// f64, with `x` taken as f32. Used by the tests to check the kernels and by nothing on a hot path.
pub fn reference_row(x: &[f32], w: &[u8], s: &[u8], k: usize, n_row: usize) -> f64 {
    let kt = k / TILE;
    let mut acc = 0.0f64;
    for kk in 0..k {
        let sc = e8m0_to_f32(s[(n_row / TILE) * kt + kk / TILE]) as f64;
        acc += x[kk] as f64 * e4m3_to_f32(w[n_row * k + kk]) as f64 * sc;
    }
    acc
}
