//! A q4 KV cache for DeepSeek-V4.1-Flash attention: the compressed-plane entries and the DSA indexer keys at 4 bits a
//! value, the sliding-window ring at fp16, and the read kernels that dequantise in place.
//!
//! =====================================================================================================================
//! DESIGN
//! =====================================================================================================================
//!
//! ## What this cache is
//!
//! The reference engine keeps one 512-wide latent per token (`kv_heads == 1`, MLA), and the REXL3 device cache keeps
//! three planes derived from it: a 128-row sliding-window ring, a compressed plane of one entry per `compress_ratio`
//! tokens, and the DSA indexer's 128-wide keys (one per compressed entry). [`super::dsv4_attn::Dsv4Planes`] holds all
//! three at fp16 today; this module makes the compressed entries and the index keys q4.
//!
//! ## Row layout — 368 bytes for a 512-wide entry, every row 16-byte aligned
//!
//! ```text
//!   0 .. 224   the 448 `nope` dims, 2 per byte (low nibble = even dim)
//! 224 .. 238   one E8M0 scale per group of 32 (`ceil(log2(amax / 6)) + 127`)
//! 238 .. 240   pad, zero
//! 240 .. 368   the 64 RoPE dims, fp16, 8 x 16 B
//! ```
//!
//! `224 + 14 = 238` pads to 240, and `240 + 128 = 368 = 23 x 16`, so a row starts on a 16-byte boundary and every
//! row is exactly 23 `uint4` loads. fp16 rows are 1,024 B, so this is 2.78x smaller, not 4x: the RoPE tail is 35% of
//! a q4 row.
//!
//! The 64 RoPE dims stay fp16 for two reasons. They are rotated by a table of unit-norm cos/sin, so they carry no
//! scale for a group quantiser to lock onto — a 32-wide group in the tail straddles the rotation, and the rotation is
//! not block-diagonal in the way q4's groups are. And 64 dims x 2 B is 128 B, which keeps the row a whole number of
//! `uint4`s; a q4 tail would save 64 B a row and cost a third error term in the logit.
//!
//! ## Scale format: E8M0 (one byte a group), not fp16
//!
//! An fp16 scale costs 14 B a row (368 -> 382, which pads to 384: +4.3% of the plane) to buy one extra mantissa bit
//! on a scale that already adapts per 32 values, and it puts a rounding into every dequantise where `2^k` is exact in
//! the multiply. E8M0 is also the pack's own convention (`pow2_scale` in [`super::dsv4_attn::quant_dequant`], and
//! every FP8 weight scale is E8M0), so the store kernel is the same arithmetic the fp16 plane's writer already ran.
//!
//! ## Rotation: a 64-wide orthonormal Hadamard on each of the 7 blocks of the nope part
//!
//! 448 = 7 x 64, and [`hadamard64`] (a fast Walsh-Hadamard transform scaled by 1/8) is its own inverse, so the
//! transform costs 64 x 6 adds a block and needs no table. It is exact in f32 (all adds of like-signed values are
//! representable at these magnitudes) and it is *valid* here for a reason specific to absorbed MLA: the latent is both
//! K and V, so `q . k` becomes `(R q) . (R k)` and `sum_j p_j v_j` becomes `R (sum_j p_j v_j)` — one rotation of the
//! query on the read side and one inverse rotation of the output accumulator, not of every key. Measured effect on
//! attention-output error: on 640 isotropic 512-wide latents the rotation is a wash (cosine 0.99329 against 0.99328,
//! mean relative error 4.64e-3 against 4.63e-3); on the same rows with eight heavy dims it is not — cosine 0.99435
//! against 0.98579, mean 4.30e-3 against 6.78e-3 (1.6x), max 2.46e-2 against 5.09e-2 (2.1x), and the fp16 plane's own
//! error on that data is 3.91e-3, so the rotation takes q4 to within 10% of fp16. RMS normalisation fixes the row norm,
//! not the per-dimension spread, and a group quantiser only sees the spread. Kept.
//!
//! ## Windows: fp16
//!
//! 128 rows x 1,024 B x 40 layers is 5.2 MB, 0.5% of the cache at 32K, and the prefill path reads the same buffer the
//! compressor writes ([`super::dsv4_attn::Dsv4Scratch::kv`]), so a q4 ring would need a second encoded copy per chunk
//! and an unpack in the innermost loop. `gpu_attn_microbench` measures a q4 window as well (the kernel takes a format
//! a plane) and it is a pure cost: 128 window rows + a 512-entry selection x 64 heads takes 398 us a layer with the
//! window at fp16 and 425 us with it q4 (1 row), 465 against 491 (8 rows). So the ring stays fp16, which is also what
//! the prefill path reads straight out of the chunk scratch.
//!
//! ## The read's speed, measured
//!
//! Same microbenchmark, one layer, 640 rows a head: 265 us fp16 against 425 us q4 at one row, 323 against 491 at eight.
//! q4 is **1.6x slower in the attention read** and 2.78x smaller, and the two are not in tension the way they look: at
//! 640 rows a head the read is 0.2-0.7 MB, which is L2-resident either way, so the kernel is instruction- and
//! latency-bound (64 warps, 1.3 a scheduler) and a 4-bit unpack is 16 extra ALU ops a value where f16 has a conversion.
//! What q4 buys is admission: 1.22 GB -> 431 MB of cache at 32K and 4.87 GB -> 1.71 GB at 128K, which is what lets
//! the slot count or the context grow. fp16 stays the default; this is the opt-in.
//!
//! ## Indexer keys: q4, group 32, E8M0 — and *exact*
//!
//! The index keys are already quantise-dequantised to E2M1 with a power-of-two scale per 32 before they reach the
//! plane (`quant_dev(.., 32, 4, true)` then `plane_write_dev`). Re-encoding `code * 2^k` at the same group width with
//! the same `2^ceil(log2(amax/6))` recovers the same code and the same scale, so the q4 plane holds the fp16 plane's
//! values bit for bit and the top-512 selection cannot move: **512/512 overlap** over 32,768 keys, measured, and the
//! test asserts bit equality of the two planes. An E4M3 byte a value with one scale a row — 144 B an entry, 1.8x the
//! bytes — gets **473/512** (92.4%). q4 wins on both axes.
//!
//! ## The `ratio_hint` bug this fixes
//!
//! [`super::dsv4_attn::Dsv4Config::ratio_hint`] returns the **largest** ratio in the table (2), and every plane is
//! sized `max_seq / ratio_hint + 1`. But the compressed plane of a ratio-1 layer (layers 20-39) holds one entry per
//! *token*: 32,769 entries at 32K where the buffer has 16,385. A ratio-1 source writing at 32K walks off its plane by
//! 16,384 x 1,024 B = 16.8 MB a layer, into whatever the allocator put next, and the same 2x under-size applied to
//! the indexer's key plane, the level-1 candidate mask (`cand_stride`) and the indexer's score row. It is a real
//! overflow, not a rounding: [`crate::dsv4_attn::compress_slots`] is the per-layer count and every one of those
//! buffers is now sized from it. `a_ratio_one_layer_gets_its_own_plane` pins it.

use crate::dsv4_attn::Dsv4Config;

/// Width of a q4 scale group, in values.
pub const KV_GROUP: usize = 32;
/// The trailing dims of a latent that stay fp16 (the RoPE block).
pub const ROPE_DIMS: usize = 64;
/// Rows are a whole number of 16-byte vectors.
pub const ROW_ALIGN: usize = 16;
/// The largest magnitude an E2M1 code carries (`6 * scale` is the group's `amax`, up to the ceil).
pub const E2M1_MAX: f32 = 6.0;

/// `n` rounded up to a 16-byte multiple.
#[inline]
pub const fn align16(n: usize) -> usize {
    (n + ROW_ALIGN - 1) & !(ROW_ALIGN - 1)
}

/// Bytes of the q4 part of a row of `dims` values: `dims / 2` nibble bytes plus one E8M0 byte a group, 16-byte aligned.
#[inline]
pub const fn q4_packed_bytes(dims: usize) -> usize {
    align16(dims / 2 + dims.div_ceil(KV_GROUP))
}

/// Bytes of one 512-wide compressed entry at q4: 368.
#[inline]
pub const fn kvq4_row_bytes(head_dim: usize) -> usize {
    q4_packed_bytes(head_dim - ROPE_DIMS) + ROPE_DIMS * 2
}

/// Bytes of one 128-wide index key at q4: 80.
#[inline]
pub const fn kvq4_index_row_bytes(index_dim: usize) -> usize {
    q4_packed_bytes(index_dim)
}

/// The element type of a plane in the cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KvFmt {
    /// fp16, `2 * dims` bytes a row. The default: every golden was made with it.
    #[default]
    F16,
    /// 4 bits a value over the nope dims, E8M0 a group of 32, the RoPE tail fp16.
    Q4,
}

impl KvFmt {
    /// Bytes of one row of `dims` values, `rope_dims` of them kept fp16 at q4 (ignored at fp16).
    pub const fn row_bytes(self, dims: usize, rope_dims: usize) -> usize {
        match self {
            KvFmt::F16 => dims * 2,
            KvFmt::Q4 => q4_packed_bytes(dims - rope_dims) + rope_dims * 2,
        }
    }

    /// The flag's spelling.
    pub const fn name(self) -> &'static str {
        match self {
            KvFmt::F16 => "f16",
            KvFmt::Q4 => "q4",
        }
    }

    /// Parses `q4` (also `4`, `q4_0`, `nvfp4` no) or `f16`/`fp16`/`half`. A typo is an error, never a silent fp16.
    pub fn parse(v: &str) -> Result<Self, String> {
        match v.trim().to_ascii_lowercase().as_str() {
            "q4" | "4" | "kv4" | "q4kv" => Ok(KvFmt::Q4),
            "f16" | "fp16" | "half" => Ok(KvFmt::F16),
            other => Err(format!("{other:?}: expected q4 or f16")),
        }
    }
}

/// The environment variable that selects the cache's format for this family (`q4` or `f16`); the server's
/// `--kv-dtype q4` sets it and it overrides the flag (so a shell can pin a format for a test run).
pub const KV_FMT_ENV: &str = "REXL3_DSV41_KV";

/// The format this process is configured for: [`KV_FMT_ENV`] when set, else `fallback`. An unrecognised value is an
/// error, so a typo never silently serves the other dtype.
pub fn configured_fmt(fallback: KvFmt) -> Result<KvFmt, String> {
    match std::env::var(KV_FMT_ENV) {
        Err(_) => Ok(fallback),
        Ok(v) if v.trim().is_empty() => Ok(fallback),
        Ok(v) => KvFmt::parse(&v).map_err(|e| format!("{KV_FMT_ENV}={v:?}: {e}")),
    }
}

// -------------------------------------------------------------------------------------------------- the E2M1 grid

/// The eight magnitudes an E2M1 code carries, `code * scale` (index = the 3-bit magnitude).
pub const E2M1_LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Nearest E2M1 magnitude code to `q` (0..=7), ties to the lower code — the device's `round_e2m1` LUT walk.
#[inline]
pub fn e2m1_code(q: f32) -> u8 {
    let a = q.abs().min(E2M1_MAX);
    let mut best = 0u8;
    let mut bd = f32::INFINITY;
    for (i, &c) in E2M1_LUT.iter().enumerate() {
        let d = (a - c).abs();
        if d < bd {
            bd = d;
            best = i as u8;
        }
    }
    best
}

/// The E2M1 nibble for `q`: the 3-bit magnitude code with the sign in bit 3.
#[inline]
pub fn e2m1_nibble(q: f32) -> u8 {
    e2m1_code(q) | if q < 0.0 { 8 } else { 0 }
}

/// The value a nibble holds at `scale` (sign in bit 3, magnitude in bits 0..3).
#[inline]
pub fn nibble_value(n: u8, scale: f32) -> f32 {
    let v = E2M1_LUT[(n & 7) as usize] * scale;
    if n & 8 != 0 {
        -v
    } else {
        v
    }
}

/// The E8M0 byte for `amax` (scale `2^(b - 127) = 2^ceil(log2(max(amax, 1e-4) / 6))`), clamped to `[1, 254]`.
///
/// The low clamp is 1 and not 0 because byte 0 is `2^-127`, a subnormal in f32: `b << 23` is zero there, so a group of
/// zeros would dequantise as an exact zero either way but a `2^-127` scale would not survive the shift.
#[inline]
pub fn e8m0_byte(amax: f32) -> u8 {
    let a = amax.max(1e-4);
    let e = (a / E2M1_MAX).log2().ceil();
    (e + 127.0).clamp(1.0, 254.0) as u8
}

/// `2^(b - 127)`.
#[inline]
pub fn e8m0_value(b: u8) -> f32 {
    if b == 255 {
        f32::NAN
    } else {
        f32::from_bits((b as u32) << 23)
    }
}

// -------------------------------------------------------------------------------------------------- the rotation

/// The orthonormal 64-point Hadamard transform of `x` in place over each of its 64-wide blocks: `x <- x H / 8`, with
/// `H` the 64x64 Walsh-Hadamard matrix. `H H^T = 64 I`, so `H / 8` is orthogonal and its own inverse.
pub fn hadamard64(x: &mut [f32]) {
    debug_assert_eq!(x.len() % 64, 0, "the rotation is defined on 64-wide blocks");
    for b in x.chunks_mut(64) {
        let mut h = 1;
        while h < 64 {
            let mut i = 0;
            while i < 64 {
                for j in i..i + h {
                    let a = b[j];
                    let c = b[j + h];
                    b[j] = a + c;
                    b[j + h] = a - c;
                }
                i += 2 * h;
            }
            h *= 2;
        }
        for v in b.iter_mut() {
            *v *= 0.125;
        }
    }
}

// -------------------------------------------------------------------------------------------------- pack / unpack

/// Quantises one row (`nope` then `rope`) into `out` (`kvq4_row_bytes(nope.len() + ROPE_DIMS)` bytes), rotating the
/// nope part when `rotate`. Returns the number of bytes written.
pub fn pack_q4_row(nope: &[f32], rope: &[f32], rotate: bool, out: &mut [u8]) -> usize {
    let nb = q4_packed_bytes(nope.len());
    let total = nb + rope.len() * 2;
    assert!(out.len() >= total, "row buffer too small");
    let mut buf = nope.to_vec();
    if rotate {
        hadamard64(&mut buf);
    }
    let groups = nope.len().div_ceil(KV_GROUP);
    for g in 0..groups {
        let lo = g * KV_GROUP;
        let hi = (lo + KV_GROUP).min(nope.len());
        let amax = buf[lo..hi].iter().fold(0f32, |a, &v| a.max(v.abs()));
        let b = e8m0_byte(amax);
        let scale = e8m0_value(b);
        let inv = 1.0 / scale;
        out[nope.len() / 2 + g] = b;
        for i in lo..hi {
            let code = e2m1_nibble(buf[i] * inv);
            let byte = &mut out[i / 2];
            if i % 2 == 0 {
                *byte = (*byte & 0xF0) | code;
            } else {
                *byte = (*byte & 0x0F) | (code << 4);
            }
        }
    }
    for b in out[nope.len() / 2 + groups..nb].iter_mut() {
        *b = 0;
    }
    for (i, &v) in rope.iter().enumerate() {
        let h = half_bits(v);
        out[nb + 2 * i] = h as u8;
        out[nb + 2 * i + 1] = (h >> 8) as u8;
    }
    total
}

/// The f16 bit pattern of `v` (round to nearest even), as the device's `__float2half_rn` produces it.
#[inline]
pub fn half_bits(v: f32) -> u16 {
    let f = v.to_bits();
    let sign = ((f >> 16) & 0x8000) as u16;
    let exp = ((f >> 23) & 0xFF) as i32;
    let man = f & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | (((man != 0) as u16) * 0x200);
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut h = (sign as u32) | ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    h as u16
}

/// The f32 a f16 bit pattern holds.
#[inline]
pub fn half_value(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            let mut m = man;
            let mut e = 127 - 15 + 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFF << 23) | (man << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Dequantises one q4 row into `out` (`nope.len() + ROPE_DIMS` values), undoing the rotation when `rotate`.
pub fn unpack_q4_row(row: &[u8], nope_dims: usize, rotate: bool, out: &mut [f32]) {
    let nb = q4_packed_bytes(nope_dims);
    let groups = nope_dims.div_ceil(KV_GROUP);
    for g in 0..groups {
        let scale = e8m0_value(row[nope_dims / 2 + g]);
        let lo = g * KV_GROUP;
        let hi = (lo + KV_GROUP).min(nope_dims);
        for i in lo..hi {
            let code = if i % 2 == 0 { row[i / 2] & 0x0F } else { row[i / 2] >> 4 };
            out[i] = nibble_value(code, scale);
        }
    }
    let rope = out.len() - nope_dims;
    for i in 0..rope {
        let h = u16::from(row[nb + 2 * i]) | (u16::from(row[nb + 2 * i + 1]) << 8);
        out[nope_dims + i] = half_value(h);
    }
    if rotate {
        hadamard64(&mut out[..nope_dims]);
    }
}

// -------------------------------------------------------------------------------------------------- host twins

/// Softmax over `keys` (`[n][hd]`) with a sink logit, f64 accumulation: the host twin of the read kernel. `rotate`
/// applies the q4 layout's rotation to the query's nope dims and its inverse to the output accumulator.
pub fn attn_host(q: &[f32], heads: usize, hd: usize, keys: &[f32], sink: &[f32], scale: f32, rotate: bool) -> Vec<f32> {
    let n = keys.len() / hd;
    let nope = hd - ROPE_DIMS;
    let mut qr = q.to_vec();
    if rotate {
        for h in 0..heads {
            hadamard64(&mut qr[h * hd..h * hd + nope]);
        }
    }
    let mut out = vec![0f32; heads * hd];
    for h in 0..heads {
        let qh = &qr[h * hd..][..hd];
        let mut mx = f32::NEG_INFINITY;
        let mut logits = vec![0f32; n];
        for j in 0..n {
            let mut s = 0f64;
            for i in 0..hd {
                s += qh[i] as f64 * keys[j * hd + i] as f64;
            }
            logits[j] = (s as f32) * scale;
            mx = mx.max(logits[j]);
        }
        let mut sum = 0f64;
        let mut acc = vec![0f64; hd];
        for j in 0..n {
            let e = ((logits[j] - mx) as f64).exp();
            sum += e;
            for i in 0..hd {
                acc[i] += e * keys[j * hd + i] as f64;
            }
        }
        sum += ((sink[h] - mx) as f64).exp();
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for i in 0..hd {
            out[h * hd + i] = (acc[i] * inv) as f32;
        }
        if rotate {
            let o = &mut out[h * hd..][..hd];
            let (nope_part, rope_part) = o.split_at_mut(nope);
            hadamard64(nope_part);
            let _ = rope_part;
        }
    }
    out
}

/// Quantise-dequantise `x` in place over `block`-wide groups, E2M1, scale `amax / 6` (or `2^ceil(log2(amax / 6))`):
/// the host twin of the plane's *existing* pre-quantisation ([`super::dsv4_attn::quant_dequant`]).
pub fn qdq_e2m1(x: &mut [f32], block: usize, pow2: bool) {
    for c in x.chunks_mut(block) {
        let amax = c.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-4);
        let scale = if pow2 { (amax / E2M1_MAX).log2().ceil().exp2() } else { amax / E2M1_MAX };
        for v in c.iter_mut() {
            *v = nibble_value(e2m1_nibble(*v / scale), scale);
        }
    }
}

/// Relative error of `got` against `want`, max and mean over the vector, and the cosine similarity.
pub fn err_stats(got: &[f32], want: &[f32]) -> (f32, f32, f32) {
    let mut mx = 0f32;
    let mut sum = 0f64;
    let mut dot = 0f64;
    let mut ng = 0f64;
    let mut nw = 0f64;
    for (&g, &w) in got.iter().zip(want) {
        let d = (g - w).abs();
        mx = mx.max(d);
        sum += d as f64;
        dot += g as f64 * w as f64;
        ng += (g as f64) * (g as f64);
        nw += (w as f64) * (w as f64);
    }
    let mean = (sum / got.len() as f64) as f32;
    let cos = if ng > 0.0 && nw > 0.0 { (dot / (ng.sqrt() * nw.sqrt())) as f32 } else { 0.0 };
    (mx, mean, cos)
}

/// The q4 plane's byte count for a config's compressed plane on `layer`, at `tokens` of context: `nc` entries of
/// [`kvq4_row_bytes`], `nc` sized by *that layer's* ratio.
pub fn compress_plane_bytes(cfg: &Dsv4Config, layer: usize, tokens: usize, fmt: KvFmt) -> u64 {
    let r = cfg.ratio(layer) as usize;
    if r == 0 {
        return 0;
    }
    let nc = tokens.div_ceil(r) + 1;
    (nc * fmt.row_bytes(cfg.head_dim, ROPE_DIMS)) as u64
}

/// The q4 indexer-key plane's bytes on `layer`.
pub fn index_plane_bytes(cfg: &Dsv4Config, layer: usize, tokens: usize, fmt: KvFmt) -> u64 {
    let r = cfg.ratio(layer) as usize;
    if r == 0 || !cfg.is_index_source(layer) {
        return 0;
    }
    let nc = tokens.div_ceil(r) + 1;
    (nc * fmt.row_bytes(cfg.index_dim, 0)) as u64
}
// -------------------------------------------------------------------------------------------------- device launchers

use crate::{CudaError, DevBuf, Stream};

/// The ABI code of a format (0 = fp16, 1 = q4) — the kernel's `FMT_*` template argument.
pub const fn fmt_code(f: KvFmt) -> i32 {
    match f {
        KvFmt::F16 => 0,
        KvFmt::Q4 => 1,
    }
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        /// Pack `rows` f16 rows of `dims` (`rope_dims` of them a fp16 tail) into q4 rows.
        #[allow(clippy::too_many_arguments)]
        pub fn gb_dsv4_kvq4_store(
            src: *const c_void,
            src_row0: i32,
            src_stride: i32,
            dst: *mut c_void,
            rows: i32,
            row0: i32,
            dims: i32,
            rope_dims: i32,
            ring_slots: i32,
            dtype: i32,
            stream: *mut c_void,
        ) -> i32;
        /// The orthonormal 64-point Walsh-Hadamard transform in place over `sub` blocks of `dim` values a row.
        #[allow(clippy::too_many_arguments)]
        pub fn gb_dsv4_kvq4_rot(
            x: *mut c_void,
            dtype: i32,
            rows: i32,
            stride: i32,
            dim: i32,
            block: i32,
            sub: i32,
            stream: *mut c_void,
        ) -> i32;
        /// Sink attention over the gathered rows, dequantising a q4 plane on read; `fmt_*` 0 = fp16, 1 = q4.
        #[allow(clippy::too_many_arguments)]
        pub fn gb_dsv4_kvq4_attn(
            q: *const c_void,
            kv_win: *const c_void,
            kv_cmp: *const c_void,
            split: i32,
            sink: *const f32,
            idx: *const i32,
            idx_stride: i32,
            idx_len: i32,
            out: *mut c_void,
            rows: i32,
            n_heads: i32,
            head_dim: i32,
            fmt_win: i32,
            fmt_cmp: i32,
            win_row: i32,
            cmp_row: i32,
            scale: f32,
            dtype: i32,
            stream: *mut c_void,
        ) -> i32;
        /// Index scores over q4 or fp16 index keys.
        #[allow(clippy::too_many_arguments)]
        pub fn gb_dsv4_kvq4_index_score(
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
            fmt: i32,
            key_row: i32,
            dtype: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// Packs `rows` rows of `dims` from `src` (`[src_row0 + r]`, f16) into q4 rows of `dst` at `row0 + r`, or at
/// `(row0 + r) % ring_slots` when `ring_slots > 0`.
#[allow(clippy::too_many_arguments)]
pub fn store_dev(
    src: &DevBuf,
    src_row0: usize,
    dst: &DevBuf,
    rows: usize,
    row0: usize,
    dims: usize,
    rope_dims: usize,
    ring_slots: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    let rb = KvFmt::Q4.row_bytes(dims, rope_dims);
    assert!(src.bytes() >= (src_row0 + rows) * dims * 2, "store source too small");
    let need = if ring_slots > 0 { ring_slots } else { row0 + rows };
    assert!(dst.bytes() >= need * rb, "store destination too small");
    let rc = unsafe {
        ffi::gb_dsv4_kvq4_store(
            src.ptr_at(src_row0 * dims * 2),
            0,
            (dims * 2) as i32,
            dst.ptr(),
            rows as i32,
            row0 as i32,
            dims as i32,
            rope_dims as i32,
            ring_slots as i32,
            0,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// The orthonormal 64-point Walsh-Hadamard transform of `x` in place: `sub` blocks of `dim` values a row, `stride`
/// values a row. `dim` must be a multiple of 64.
#[allow(clippy::too_many_arguments)]
pub fn rot_dev(
    x: &DevBuf,
    rows: usize,
    stride: usize,
    dim: usize,
    block: usize,
    sub: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    assert!(x.bytes() >= rows * stride * 2, "rotation source too small");
    let rc =
        unsafe { ffi::gb_dsv4_kvq4_rot(x.ptr(), 0, rows as i32, stride as i32, dim as i32, block as i32, sub as i32, st.raw()) };
    CudaError::check(rc)
}

/// Sink attention over the gathered rows with a q4 (or fp16) compressed plane and window ring.
#[allow(clippy::too_many_arguments)]
pub fn attn_dev(
    q: &DevBuf,
    kv_win: &DevBuf,
    kv_cmp: &DevBuf,
    split: usize,
    sink: &DevBuf,
    idx: &DevBuf,
    idx_stride: usize,
    idx_len: usize,
    out: &DevBuf,
    rows: usize,
    n_heads: usize,
    head_dim: usize,
    win_fmt: KvFmt,
    cmp_fmt: KvFmt,
    scale: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    attn_raw_dev(
        q,
        kv_win,
        kv_cmp,
        split,
        sink,
        idx.ptr() as *const i32,
        idx_stride,
        idx_len,
        out,
        rows,
        n_heads,
        head_dim,
        win_fmt,
        cmp_fmt,
        scale,
        st,
    )
}

// the raw pointer crosses straight into the extern kernel; marking the wrapper `unsafe` would push the
// obligation onto every caller for no gain (the pointer is the caller's device index array)
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[allow(clippy::too_many_arguments)]
pub fn attn_raw_dev(
    q: &DevBuf,
    kv_win: &DevBuf,
    kv_cmp: &DevBuf,
    split: usize,
    sink: &DevBuf,
    idx_ptr: *const i32,
    idx_stride: usize,
    idx_len: usize,
    out: &DevBuf,
    rows: usize,
    n_heads: usize,
    head_dim: usize,
    win_fmt: KvFmt,
    cmp_fmt: KvFmt,
    scale: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    let rc = unsafe {
        ffi::gb_dsv4_kvq4_attn(
            q.ptr(),
            kv_win.ptr(),
            kv_cmp.ptr(),
            split as i32,
            sink.ptr() as *const f32,
            idx_ptr,
            idx_stride as i32,
            idx_len as i32,
            out.ptr(),
            rows as i32,
            n_heads as i32,
            head_dim as i32,
            fmt_code(win_fmt),
            fmt_code(cmp_fmt),
            win_fmt.row_bytes(head_dim, ROPE_DIMS) as i32,
            cmp_fmt.row_bytes(head_dim, ROPE_DIMS) as i32,
            scale,
            0,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

#[allow(clippy::too_many_arguments)]
pub fn index_score_dev(
    q: &DevBuf,
    index_k: &DevBuf,
    w: &DevBuf,
    mask: Option<&DevBuf>,
    mask_stride: usize,
    score: &DevBuf,
    rows: usize,
    n_pos: usize,
    n_heads: usize,
    dim: usize,
    scale: f32,
    start_pos: usize,
    ratio: usize,
    fmt: KvFmt,
    st: &Stream,
) -> Result<(), CudaError> {
    index_score_raw_dev(
        q,
        index_k,
        w,
        mask.map(|m| m.ptr() as *const u8).unwrap_or(core::ptr::null()),
        mask_stride,
        score,
        rows,
        n_pos,
        n_heads,
        dim,
        scale,
        start_pos,
        ratio,
        fmt,
        st,
    )
}

// the raw pointer crosses straight into the extern kernel; marking the wrapper `unsafe` would push the
// obligation onto every caller for no gain (the pointer is the caller's mask row)
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[allow(clippy::too_many_arguments)]
pub fn index_score_raw_dev(
    q: &DevBuf,
    index_k: &DevBuf,
    w: &DevBuf,
    mask_ptr: *const u8,
    mask_stride: usize,
    score: &DevBuf,
    rows: usize,
    n_pos: usize,
    n_heads: usize,
    dim: usize,
    scale: f32,
    start_pos: usize,
    ratio: usize,
    fmt: KvFmt,
    st: &Stream,
) -> Result<(), CudaError> {
    let rc = unsafe {
        ffi::gb_dsv4_kvq4_index_score(
            q.ptr(),
            index_k.ptr(),
            w.ptr() as *const f32,
            mask_ptr,
            mask_stride as i32,
            score.ptr() as *mut f32,
            rows as i32,
            n_pos as i32,
            n_heads as i32,
            dim as i32,
            scale,
            start_pos as i32,
            ratio as i32,
            fmt_code(fmt),
            fmt.row_bytes(dim, 0) as i32,
            0,
            st.raw(),
        )
    };
    CudaError::check(rc)
}
