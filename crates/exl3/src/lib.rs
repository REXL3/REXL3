//! EXL3, ExLlamaV3's trellis quantization (MIT, Copyright (c) 2025 Turboderp).
//!
//! This crate is a Rust port of TensorFold's `cuda/exl3/format.py` (MIT) and follows the bit
//! arithmetic written out in TensorFold's `docs/recipes/exl3.md`. It is the CPU reference every
//! GPU kernel in this engine must reproduce: `unpack` gives W_q exactly (fp16 bits), and `forward`
//! gives the layer's output in float64.
//!
//! Layer: `y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh + bias`, H = 128x128 Sylvester Hadamard / sqrt(128).
//! Tile: 256 values in a circular bitstream of 256 * bits bits; each value is the 16-bit window
//! ending at E(p), mapped through a codebook (`3inst`, `mcg`, `mul1`) to fp16.
#![forbid(unsafe_code)]

pub mod budget;
pub mod checkpoint;
pub mod encode;
pub mod f16;
pub mod loadstats;
pub mod tp;

use f16::{f16_to_f64, f64_to_f16};

pub const HAD: usize = 128;
const INST3_MUL: u64 = 89226354;
const INST3_ADD: u64 = 64248484;
const MCG_MUL: u64 = 0xCBAC1FED;
const MUL1_MUL: u64 = 0x83DCD12D;
const MASK: u64 = 0x8FFF8FFF;
const FLIP: u64 = 0x3B603B60;
const MUL1_SCALE: u16 = 0x1EEE;
const MUL1_BIAS: u16 = 0xC931;

/// Marker tensor values (int32 bit patterns) naming the codebook.
pub const MARKER_MCG: u32 = 0xCBAC1FED;
pub const MARKER_MUL1: u32 = 0x83DCD12D;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codebook {
    Inst3,
    Mcg,
    Mul1,
}

impl Codebook {
    pub fn name(&self) -> &'static str {
        match self {
            Codebook::Inst3 => "3inst",
            Codebook::Mcg => "mcg",
            Codebook::Mul1 => "mul1",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "3inst" => Some(Codebook::Inst3),
            "mcg" => Some(Codebook::Mcg),
            "mul1" => Some(Codebook::Mul1),
            _ => None,
        }
    }
    pub const ALL: [Codebook; 3] = [Codebook::Inst3, Codebook::Mcg, Codebook::Mul1];
}

/// Bits per weight, stored doubled so half widths (1.5, 2.5, 3.5) are exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Bits(u8);

impl Bits {
    /// Every width ExLlamaV3 writes, doubled: 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8.
    pub const ALL_TWICE: [u8; 11] = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16];

    pub fn from_twice(t: u8) -> Option<Bits> {
        Self::ALL_TWICE.contains(&t).then_some(Bits(t))
    }
    /// From a trellis' last dimension (16 * bits int16 words a tile).
    pub fn from_trellis_last(last: usize) -> Option<Bits> {
        if last % 8 != 0 {
            return None;
        }
        u8::try_from(last / 8).ok().and_then(Self::from_twice)
    }
    pub fn twice(&self) -> u8 {
        self.0
    }
    pub fn as_f64(&self) -> f64 {
        self.0 as f64 / 2.0
    }
    pub fn is_integer(&self) -> bool {
        self.0 % 2 == 0
    }
    /// int16 words per 16x16 tile: 16 * bits.
    pub fn tile_words(&self) -> usize {
        8 * self.0 as usize
    }
    /// E(p) for p = 0..255: where value p's 16-bit window ends in the tile's bitstream (exclusive).
    pub fn stream_ends(&self) -> [u32; 256] {
        let mut e = [0u32; 256];
        for (i, v) in e.iter_mut().enumerate() {
            let p1 = i as u32 + 1;
            *v = if self.is_integer() { p1 * (self.0 as u32 / 2) } else { (p1 * self.0 as u32 - (p1 % 2)) / 2 };
        }
        e
    }
}

/// fp16 bit pattern of every 16-bit state, for one codebook.
pub fn codebook(cb: Codebook) -> Vec<u16> {
    (0u64..65536)
        .map(|s| match cb {
            Codebook::Mul1 => {
                let x = (s * MUL1_MUL) & 0xFFFF_FFFF;
                let h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255);
                // exact in float64, so exactly one rounding to fp16
                f64_to_f16(h as f64 * f16_to_f64(MUL1_SCALE) + f16_to_f64(MUL1_BIAS))
            }
            Codebook::Mcg | Codebook::Inst3 => {
                let x = if cb == Codebook::Mcg { (s * MCG_MUL) & 0xFFFF_FFFF } else { (s * INST3_MUL + INST3_ADD) & 0xFFFF_FFFF };
                let x = (x & MASK) ^ FLIP;
                f64_to_f16(f16_to_f64((x & 0xFFFF) as u16) + f16_to_f64((x >> 16) as u16))
            }
        })
        .collect()
}

/// (row, column) in its 16x16 tile of each of a tile's 256 values, in stream order: lane l of a warp
/// holds values 8l..8l+7, its B fragments of the tile's two mma.m16n8k16.
pub fn tile_positions() -> ([u8; 256], [u8; 256]) {
    let mut r = [0u8; 256];
    let mut c = [0u8; 256];
    for p in 0..256usize {
        let (lane, j) = (p / 8, p % 8);
        r[p] = (2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)) as u8;
        c[p] = (lane / 4 + 8 * (j >> 2)) as u8;
    }
    (r, c)
}

/// The 16-bit state of each of a tile's 256 values, from its 16 * bits int16 words.
pub fn states(tile: &[i16], bits: Bits, ends: &[u32; 256]) -> [u16; 256] {
    let nw16 = bits.tile_words();
    assert_eq!(tile.len(), nw16, "tile must hold 16 * bits words");
    let nw = nw16 / 2;
    let words: Vec<u64> = (0..nw).map(|i| (tile[2 * i] as u16 as u64) | ((tile[2 * i + 1] as u16 as u64) << 16)).collect();
    let ring = 32 * nw as u64;
    let mut out = [0u16; 256];
    for p in 0..256 {
        let first = ends[p] as u64 + ring - 16;
        let i0 = ((first / 32) % nw as u64) as usize;
        let off = first % 32;
        let i1 = (i0 + 1) % nw;
        let pair = (words[i0] << 32) | words[i1];
        out[p] = ((pair >> (48 - off)) & 0xFFFF) as u16;
    }
    out
}

/// W_q [K, N] as fp16 bits, row-major, from trellis int16 [K/16, N/16, 16 * bits] (row-major).
pub fn unpack(trellis: &[i16], kt: usize, nt: usize, bits: Bits, table: &[u16]) -> Vec<u16> {
    let nw = bits.tile_words();
    assert_eq!(trellis.len(), kt * nt * nw, "trellis size does not match [K/16, N/16, 16*bits]");
    assert_eq!(table.len(), 65536);
    let (rows, cols) = tile_positions();
    let ends = bits.stream_ends();
    let n = nt * 16;
    let mut w = vec![0u16; kt * 16 * n];
    for kk in 0..kt {
        for nn in 0..nt {
            let base = (kk * nt + nn) * nw;
            let st = states(&trellis[base..base + nw], bits, &ends);
            for p in 0..256 {
                let r = kk * 16 + rows[p] as usize;
                let c = nn * 16 + cols[p] as usize;
                w[r * n + c] = table[st[p] as usize];
            }
        }
    }
    w
}

/// su / sv packed signs int16 [n/16] -> +1.0 / -1.0: bit b of word w set means element 16w + b is -1.
pub fn unpack_signs(packed: &[i16]) -> Vec<f64> {
    let mut out = Vec::with_capacity(packed.len() * 16);
    for &w in packed {
        let w = w as u16;
        for b in 0..16 {
            out.push(if (w >> b) & 1 == 1 { -1.0 } else { 1.0 });
        }
    }
    out
}

/// H / sqrt(128) applied in place to every block of 128 along a row of length `len`.
///
/// The loops keep their index form: `i & j` picks the Hadamard sign, and the summation order is part of the
/// float64 reference the fixtures were checked against.
#[allow(clippy::needless_range_loop)]
fn rotate_row(x: &mut [f64]) {
    assert!(x.len() % HAD == 0, "rotated dimension must be a multiple of 128");
    let scale = 1.0 / (HAD as f64).sqrt();
    let mut tmp = [0.0f64; HAD];
    for block in x.chunks_mut(HAD) {
        for j in 0..HAD {
            let mut acc = 0.0;
            for i in 0..HAD {
                let sign = if ((i & j).count_ones() & 1) == 1 { -1.0 } else { 1.0 };
                acc += block[i] * (sign * scale);
            }
            tmp[j] = acc;
        }
        block.copy_from_slice(&tmp);
    }
}

/// Reference layer in float64, ordered as the kernels order it: rotate input, W_q, rotate output.
/// `x` is [rows, K] row-major; returns [rows, N].
#[allow(clippy::too_many_arguments)]
pub fn forward(
    x: &[f64],
    rows: usize,
    trellis: &[i16],
    kt: usize,
    nt: usize,
    bits: Bits,
    table: &[u16],
    suh: &[f64],
    svh: &[f64],
    bias: Option<&[f64]>,
) -> Vec<f64> {
    let (k, n) = (kt * 16, nt * 16);
    assert_eq!(x.len(), rows * k);
    assert_eq!(suh.len(), k);
    assert_eq!(svh.len(), n);
    let wq: Vec<f64> = unpack(trellis, kt, nt, bits, table).into_iter().map(f16_to_f64).collect();
    let mut y = vec![0.0f64; rows * n];
    for r in 0..rows {
        let mut xh: Vec<f64> = (0..k).map(|i| x[r * k + i] * suh[i]).collect();
        rotate_row(&mut xh);
        let yr = &mut y[r * n..(r + 1) * n];
        for i in 0..k {
            let xi = xh[i];
            let wrow = &wq[i * n..(i + 1) * n];
            for j in 0..n {
                yr[j] += xi * wrow[j];
            }
        }
        rotate_row(yr);
        for j in 0..n {
            yr[j] *= svh[j];
            if let Some(b) = bias {
                yr[j] += b[j];
            }
        }
    }
    y
}

/// `x[0..128]` = H x / sqrt(128) in place, H the 128 x 128 Sylvester Hadamard matrix (`(-1)^popcount(i & j)`, the matrix
/// [`forward`]'s rotations use), by the fast Walsh-Hadamard butterflies. Equal to `rotate_row` up to float64 summation order.
fn fwht_128(x: &mut [f64]) {
    debug_assert_eq!(x.len(), HAD);
    let mut h = 1;
    while h < HAD {
        for i in (0..HAD).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (HAD as f64).sqrt();
    x.iter_mut().for_each(|v| *v *= scale);
}

/// The layer's effective weight in float64, `[K, N]` row-major: `W[i][j] = suh[i] * (H_K W_q H_N)[i][j] * svh[j]`, so that
/// `forward(x) = x @ W (+ bias)` for every `x`. W_q is [`unpack`]'s; the rotations are the fast Walsh-Hadamard transform of the matrix
/// [`forward`] uses (the two agree to float64 rounding). This is the host decode a reference that wants a dense weight (the MTP head's
/// `mtp_ref` on a 4-bit head) runs once per matrix, where [`forward`] would decode on every call.
pub fn decode_weight(trellis: &[i16], kt: usize, nt: usize, bits: Bits, table: &[u16], suh: &[f64], svh: &[f64]) -> Vec<f64> {
    let (k, n) = (kt * 16, nt * 16);
    assert!(k % HAD == 0 && n % HAD == 0, "K={k} and N={n} must be multiples of {HAD}");
    assert_eq!(suh.len(), k);
    assert_eq!(svh.len(), n);
    let mut w: Vec<f64> = unpack(trellis, kt, nt, bits, table).into_iter().map(f16_to_f64).collect();
    // H_K W_q: each column's blocks of 128 rows
    let mut col = [0.0f64; HAD];
    for kb in (0..k).step_by(HAD) {
        for j in 0..n {
            for (i, c) in col.iter_mut().enumerate() {
                *c = w[(kb + i) * n + j];
            }
            fwht_128(&mut col);
            for (i, c) in col.iter().enumerate() {
                w[(kb + i) * n + j] = *c;
            }
        }
    }
    // the row scales, then (.) H_N on each row's blocks of 128 columns, then the column scales
    for (i, row) in w.chunks_exact_mut(n).enumerate() {
        row.iter_mut().for_each(|v| *v *= suh[i]);
        for block in row.chunks_exact_mut(HAD) {
            fwht_128(block);
        }
        for (v, s) in row.iter_mut().zip(svh) {
            *v *= s;
        }
    }
    w
}

/// The effective weight of ONE 128-column strip of the layer, `[K, 128]` row-major: columns `128 strip .. 128 strip + 127` of
/// [`decode_weight`]'s `[K, N]`, bit for bit (every operation on a strip's elements is the same, in the same order: the K-side transform
/// is per column, the N-side per 128-column block, and the scales are per element). `trellis`, `suh` and `svh` are the whole layer's.
/// Decoding only the strips a caller needs is how a restricted vocabulary decodes some rows of an EXL3 `lm_head` (column `v` of the
/// layer is the head's row of token `v`) without the 8 GiB float64 matrix of the whole head.
#[allow(clippy::too_many_arguments)]
pub fn decode_strip(
    trellis: &[i16],
    kt: usize,
    nt: usize,
    bits: Bits,
    table: &[u16],
    suh: &[f64],
    svh: &[f64],
    strip: usize,
) -> Vec<f64> {
    let (k, n) = (kt * 16, nt * 16);
    assert!(k % HAD == 0 && n % HAD == 0, "K={k} and N={n} must be multiples of {HAD}");
    assert!(strip < n / HAD, "strip {strip} of a {n}-column layer");
    assert_eq!(suh.len(), k);
    assert_eq!(svh.len(), n);
    let nw = bits.tile_words();
    assert_eq!(trellis.len(), kt * nt * nw, "trellis size does not match [K/16, N/16, 16*bits]");
    assert_eq!(table.len(), 65536);
    let (rows, cols) = tile_positions();
    let ends = bits.stream_ends();
    // W_q of the strip's 8 tile columns, as `unpack` decodes them
    let mut w = vec![0.0f64; k * HAD];
    for kk in 0..kt {
        for t in 0..HAD / 16 {
            let nn = strip * (HAD / 16) + t;
            let base = (kk * nt + nn) * nw;
            let st = states(&trellis[base..base + nw], bits, &ends);
            for p in 0..256 {
                let r = kk * 16 + rows[p] as usize;
                let c = t * 16 + cols[p] as usize;
                w[r * HAD + c] = f16_to_f64(table[st[p] as usize]);
            }
        }
    }
    // H_K W_q on each column's blocks of 128 rows, then the row scales, H_N on the strip's row block, and the column scales
    let mut col = [0.0f64; HAD];
    for kb in (0..k).step_by(HAD) {
        for j in 0..HAD {
            for (i, c) in col.iter_mut().enumerate() {
                *c = w[(kb + i) * HAD + j];
            }
            fwht_128(&mut col);
            for (i, c) in col.iter().enumerate() {
                w[(kb + i) * HAD + j] = *c;
            }
        }
    }
    let sv = &svh[strip * HAD..(strip + 1) * HAD];
    for (i, row) in w.chunks_exact_mut(HAD).enumerate() {
        row.iter_mut().for_each(|v| *v *= suh[i]);
        fwht_128(row);
        for (v, s) in row.iter_mut().zip(sv) {
            *v *= s;
        }
    }
    w
}

/// FNV-1a 64 over little-endian bytes: used by fixtures to compare large outputs compactly.
pub fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths() {
        assert_eq!(Bits::from_trellis_last(32).unwrap().as_f64(), 2.0);
        assert_eq!(Bits::from_trellis_last(40).unwrap().as_f64(), 2.5);
        assert!(Bits::from_trellis_last(36).is_none());
        assert!(Bits::from_trellis_last(144).is_none()); // 9 bits
    }

    #[test]
    fn half_width_ends_alternate() {
        let e = Bits::from_twice(5).unwrap().stream_ends(); // 2.5 bits
        assert_eq!(&e[..4], &[2, 5, 7, 10]);
        assert_eq!(e[255], 640); // 256 * 2.5
    }

    #[test]
    fn the_dense_decode_is_the_layer_applied_to_the_identity() {
        // a random 4-bit mul1 group, K = 256, N = 384: decode_weight's rows are forward's outputs for the unit inputs
        let (kt, nt, bits) = (16, 24, Bits::from_twice(8).unwrap());
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let trellis: Vec<i16> = (0..kt * nt * bits.tile_words()).map(|_| next() as i16).collect();
        let suh: Vec<f64> = (0..kt * 16).map(|_| ((next() % 2000) as f64 - 1000.0) / 3000.0).collect();
        let svh: Vec<f64> = (0..nt * 16).map(|_| ((next() % 2000) as f64 - 1000.0) / 7000.0).collect();
        let table = codebook(Codebook::Mul1);
        let (k, n) = (kt * 16, nt * 16);
        let w = decode_weight(&trellis, kt, nt, bits, &table, &suh, &svh);
        let mut eye = vec![0.0; k * k];
        (0..k).for_each(|i| eye[i * k + i] = 1.0);
        let f = forward(&eye, k, &trellis, kt, nt, bits, &table, &suh, &svh, None);
        let (mut d2, mut t2) = (0.0f64, 0.0f64);
        for (a, b) in w.iter().zip(&f) {
            d2 += (a - b) * (a - b);
            t2 += b * b;
        }
        assert!(t2 > 0.0 && (d2 / t2).sqrt() < 1e-13, "rel {}", (d2 / t2).sqrt());
        // and a row through the dense weight is forward's row
        let x: Vec<f64> = (0..k).map(|i| ((i * 37 % 101) as f64 - 50.0) / 50.0).collect();
        let y = forward(&x, 1, &trellis, kt, nt, bits, &table, &suh, &svh, None);
        for j in 0..n {
            let d: f64 = (0..k).map(|i| x[i] * w[i * n + j]).sum();
            assert!((d - y[j]).abs() <= 1e-12 * (1.0 + y[j].abs()), "column {j}: {d} vs {}", y[j]);
        }
    }

    #[test]
    fn a_strip_decode_is_the_dense_decodes_columns_bit_for_bit_at_every_width() {
        let (kt, nt) = (16, 24);
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for twice in [2u8, 4, 6, 8, 10, 12, 14, 16] {
            let bits = Bits::from_twice(twice).unwrap();
            let trellis: Vec<i16> = (0..kt * nt * bits.tile_words()).map(|_| next() as i16).collect();
            let suh: Vec<f64> = (0..kt * 16).map(|_| ((next() % 2000) as f64 - 1000.0) / 3000.0).collect();
            let svh: Vec<f64> = (0..nt * 16).map(|_| ((next() % 2000) as f64 - 1000.0) / 7000.0).collect();
            let table = codebook(Codebook::Mul1);
            let (k, n) = (kt * 16, nt * 16);
            let w = decode_weight(&trellis, kt, nt, bits, &table, &suh, &svh);
            for strip in 0..n / HAD {
                let d = decode_strip(&trellis, kt, nt, bits, &table, &suh, &svh, strip);
                for i in 0..k {
                    for j in 0..HAD {
                        assert_eq!(
                            d[i * HAD + j].to_bits(),
                            w[i * n + strip * HAD + j].to_bits(),
                            "{twice} strip {strip} [{i}, {j}]"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn positions_are_a_permutation() {
        let (r, c) = tile_positions();
        let mut seen = [false; 256];
        for p in 0..256 {
            let i = r[p] as usize * 16 + c[p] as usize;
            assert!(!seen[i]);
            seen[i] = true;
        }
    }
}
