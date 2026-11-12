//! DSpark draft helpers (MiniCPM-V-4.7's draft; kernels in `csrc/dspark.cu`).
//!
//! * [`confidence`]: the draft's per-position acceptance estimate, `sigmoid(proj . [h_k ; markov_w1[x_{k-1}]] + b)` in fp32, one
//!   block per block row. `x_{k-1}` is the anchor for row 0 and the walk's pick of row `k - 1` otherwise.
//! * [`bf16_to_f16_copy`] / [`f16_to_bf16_copy`]: the activation conversions around the dense (cuBLAS) reference path, with a device
//!   counter of the values the fp16 range cannot hold (the plan's `bf16_to_f16` overflow flag).
//!
//! [`confidence_host`] is the host mirror (bit-exact: same accumulation and butterfly order).
#![allow(clippy::too_many_arguments)]

use crate::rt::{CudaError, DevBuf, Stream};

/// The threads a confidence block uses; the host mirror reproduces its reduction order from this.
pub const CONF_THREADS: usize = 256;

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_dspark_confidence(
            h: *const c_void,
            tokens: *const c_void,
            anchor: *const c_void,
            w1: *const c_void,
            w: *const c_void,
            b: *const c_void,
            out: *mut c_void,
            n_seq: i32,
            rows: i32,
            hidden: i32,
            rank: i32,
            vocab: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_dspark_bf16_to_f16(src: *const c_void, dst: *mut c_void, n: i64, bad: *mut c_void, stream: *mut c_void) -> i32;
        pub fn gb_dspark_f16_to_bf16(src: *const c_void, dst: *mut c_void, n: i64, stream: *mut c_void) -> i32;
    }
}

fn need(b: &DevBuf, bytes: usize, what: &str) -> Result<(), CudaError> {
    if b.bytes() < bytes {
        eprintln!("dspark {what}: buffer of {} bytes, the launch touches {bytes}", b.bytes());
        return Err(CudaError(1));
    }
    Ok(())
}

fn i(v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| CudaError(1))
}

/// Confidence of every block row of `n_seq` sequences (`rows` block rows each).
///
/// `h` bf16 `[n_seq * rows, hidden]` (final-normed block rows), `tokens` i32 `[n_seq * rows]` (the walk's picks), `anchor` i32
/// `[n_seq]`, `w1` f16 `[vocab, rank]` (`markov_w1`), `w` f32 `[hidden + rank]` (`proj.weight`), `b` f32 `[1]`, `out` f32
/// `[n_seq * rows]`. A previous token outside `0..vocab` contributes a zero Markov embedding.
pub fn confidence(
    h: &DevBuf,
    tokens: &DevBuf,
    anchor: &DevBuf,
    w1: &DevBuf,
    w: &DevBuf,
    b: &DevBuf,
    out: &mut DevBuf,
    n_seq: usize,
    rows: usize,
    hidden: usize,
    rank: usize,
    vocab: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    let r = n_seq * rows;
    need(h, r * hidden * 2, "confidence h")?;
    need(tokens, r * 4, "confidence tokens")?;
    need(anchor, n_seq * 4, "confidence anchor")?;
    need(w1, vocab * rank * 2, "confidence w1")?;
    need(w, (hidden + rank) * 4, "confidence w")?;
    need(b, 4, "confidence b")?;
    need(out, r * 4, "confidence out")?;
    CudaError::check(unsafe {
        ffi::gb_dspark_confidence(
            h.ptr(),
            tokens.ptr(),
            anchor.ptr(),
            w1.ptr(),
            w.ptr(),
            b.ptr(),
            out.ptr(),
            i(n_seq)?,
            i(rows)?,
            i(hidden)?,
            i(rank)?,
            i(vocab)?,
            st.raw(),
        )
    })
}

/// `dst` (fp16) `= src` (bf16) over `n` elements. `bad` (one device i32, zeroed by the caller) is incremented for every element
/// whose fp16 value is not finite (an input past 65,504, an infinity or a NaN).
pub fn bf16_to_f16_copy(src: &DevBuf, dst: &mut DevBuf, n: usize, bad: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
    need(src, n * 2, "bf16_to_f16 src")?;
    need(dst, n * 2, "bf16_to_f16 dst")?;
    need(bad, 4, "bf16_to_f16 bad")?;
    CudaError::check(unsafe { ffi::gb_dspark_bf16_to_f16(src.ptr(), dst.ptr(), n as i64, bad.ptr(), st.raw()) })
}

/// [`bf16_to_f16_copy`] over one buffer: element `i` is read and then written by the same thread, so the conversion is exact in
/// place (the loader's bf16 weights become fp16 where they sit).
pub fn bf16_to_f16_in_place(buf: &mut DevBuf, n: usize, bad: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
    need(buf, n * 2, "bf16_to_f16 in place")?;
    need(bad, 4, "bf16_to_f16 bad")?;
    CudaError::check(unsafe { ffi::gb_dspark_bf16_to_f16(buf.ptr(), buf.ptr(), n as i64, bad.ptr(), st.raw()) })
}

/// `dst` (bf16) `= src` (fp16) over `n` elements.
pub fn f16_to_bf16_copy(src: &DevBuf, dst: &mut DevBuf, n: usize, st: &Stream) -> Result<(), CudaError> {
    need(src, n * 2, "f16_to_bf16 src")?;
    need(dst, n * 2, "f16_to_bf16 dst")?;
    CudaError::check(unsafe { ffi::gb_dspark_f16_to_bf16(src.ptr(), dst.ptr(), n as i64, st.raw()) })
}

/// bf16 bits to f32.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// fp16 bits to f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 31) as u32, (h & 1023) as u32);
    let v = match (e, m) {
        (0, 0) => 0.0f32,
        (0, m) => m as f32 * (2.0f32).powi(-24),
        (31, 0) => f32::INFINITY,
        (31, _) => f32::NAN,
        (e, m) => (1.0 + m as f32 / 1024.0) * (2.0f32).powi(e as i32 - 15),
    };
    if s == 1 {
        -v
    } else {
        v
    }
}

/// The kernel's arithmetic on the host: confidence of one block row. `h` is the row's `hidden` bf16 bits, `m` its `rank` fp16 bits
/// (`markov_w1[x_{k-1}]`, or all zero bits for an out-of-range token), `w` `[hidden + rank]`.
pub fn confidence_host(h: &[u16], m: &[u16], w: &[f32], b: f32) -> f32 {
    let (hidden, rank) = (h.len(), m.len());
    assert_eq!(w.len(), hidden + rank);
    let mut part = vec![0f32; CONF_THREADS];
    for (t, p) in part.iter_mut().enumerate() {
        let mut acc = 0f32;
        let mut k = t;
        while k < hidden {
            acc = bf16_to_f32(h[k]).mul_add(w[k], acc);
            k += CONF_THREADS;
        }
        let mut k = t;
        while k < rank {
            acc = f16_to_f32(m[k]).mul_add(w[hidden + k], acc);
            k += CONF_THREADS;
        }
        *p = acc;
    }
    // warp butterfly 16, 8, 4, 2, 1 (every lane ends with the same bits: lane 0's value), then the warps added in order
    let mut warp_sums = Vec::new();
    for wp in 0..CONF_THREADS / 32 {
        let mut v: Vec<f32> = part[wp * 32..wp * 32 + 32].to_vec();
        let mut off = 16;
        while off >= 1 {
            let o: Vec<f32> = (0..32).map(|l| v[l] + v[l ^ off]).collect();
            v = o;
            off >>= 1;
        }
        warp_sums.push(v[0]);
    }
    let mut s = warp_sums[0];
    for w in &warp_sums[1..] {
        s += *w;
    }
    1.0 / (1.0 + (-(s + b)).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_decode_matches_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert!(f16_to_f32(0x7c00).is_infinite() && f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn the_host_mirror_is_a_sigmoid_of_the_dot() {
        // h = 1.0 (bf16 0x3f80) over 2048 elements, m = 0, w = 0.001: dot = 2.048
        let h = vec![0x3f80u16; 2048];
        let m = vec![0u16; 256];
        let w = vec![0.001f32; 2304];
        let c = confidence_host(&h, &m, &w, 0.0);
        let want = 1.0 / (1.0 + (-2.048f64).exp());
        assert!((c as f64 - want).abs() < 1e-5, "{c} vs {want}");
    }
}
