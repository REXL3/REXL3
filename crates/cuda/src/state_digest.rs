//! A content digest of device memory (`csrc/state_digest.cu`): the state a rank snapshotted for the prefix cache, compared with what it
//! holds after a restore. A two-node group restores a prefix on every rank; a rank that restored stale or no state but answered Ok
//! would feed wrong numbers into the all-reduce, and only this digest, taken over the restored state itself, sees that per request.
//!
//! **The digest.** Over a list of byte ranges (each 4-byte aligned), word `i` (4 bytes, little endian) of range `k` contributes
//! `mix64(mix64(seed_k + i) ^ word)`, all contributions summed modulo 2^64; a range's trailing `n % 4` bytes are one more word tagged with
//! their count, and the ranges' lengths are folded in at the end ([`finish`]). The sum makes it deterministic on the device (no order),
//! `mix64` (the SplitMix64 finalizer, a bijection) makes every word at a position count, and the position key makes a moved word
//! count. It is a 64-bit checksum against accidents (a stale entry, a copy that did not happen, a partial copy, the wrong slot), not a
//! cryptographic hash. [`digest_host`] is its CPU twin: the device digest of ranges equals it on the same bytes, so host-side parts of a
//! state can be folded in with [`sum_host`].
//!
//! **Cost.** One memory-bound pass over the bytes (the GB10's memory, or pageable host memory the GPU reads in place).
use crate::rt::{CudaError, DevBuf, Stream};
use std::ffi::c_void;

crate::launchers! {
    pub fn gb_state_digest_accumulate(p: *const c_void, bytes: u64, seed: u64, out: *mut c_void, s: *mut c_void) -> i32;
}

/// The SplitMix64 finalizer (a bijection of `u64`).
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The key of range `k` (its words are keyed `seed_of(k) + i`).
pub fn seed_of(k: usize) -> u64 {
    mix64((k as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x5354_4154_4544_4947)
}

/// The sum of range `k`'s contributions over `bytes`, on the CPU (what the device adds for the same range).
pub fn sum_host(k: usize, bytes: &[u8]) -> u64 {
    let seed = seed_of(k);
    let words = bytes.len() / 4;
    let mut acc = 0u64;
    for (i, c) in bytes.chunks_exact(4).enumerate() {
        let w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64;
        acc = acc.wrapping_add(mix64(mix64(seed.wrapping_add(i as u64)) ^ w));
    }
    let tail = &bytes[words * 4..];
    if !tail.is_empty() {
        let mut t = 0u64;
        for (j, b) in tail.iter().enumerate() {
            t |= (*b as u64) << (8 * j);
        }
        acc = acc.wrapping_add(mix64(mix64(seed.wrapping_add(words as u64)) ^ t ^ ((tail.len() as u64) << 32)));
    }
    acc
}

/// The digest of a sum over ranges of these lengths (range `k`'s length is folded in at position `k`).
pub fn finish(sum: u64, lens: impl IntoIterator<Item = usize>) -> u64 {
    let mut h = mix64(sum ^ 0x7265_786c_3364_6967);
    for (k, n) in lens.into_iter().enumerate() {
        h = mix64(h ^ (n as u64) ^ ((k as u64) << 48));
    }
    h
}

/// The digest of host byte ranges (range `k` is `ranges[k]`): the CPU twin of [`digest_device`].
pub fn digest_host(ranges: &[&[u8]]) -> u64 {
    let sum = ranges.iter().enumerate().fold(0u64, |s, (k, r)| s.wrapping_add(sum_host(k, r)));
    finish(sum, ranges.iter().map(|r| r.len()))
}

/// The sum of the contributions of device ranges (range `k` is `ranges[k]`: a device pointer, or pageable host memory the GPU reads in
/// place, 4-byte aligned, and its bytes), run on `stream` after whatever it has queued (a restore's copies), then synchronized.
pub fn sum_device(ranges: &[(*const c_void, usize)], stream: &Stream) -> Result<u64, CudaError> {
    let out = DevBuf::zeroed(8)?;
    for (k, &(p, n)) in ranges.iter().enumerate() {
        if n == 0 {
            continue;
        }
        // SAFETY: the caller hands device-readable ranges of `n` bytes; the kernel reads them and adds to `out`'s eight bytes
        CudaError::check(unsafe { gb_state_digest_accumulate(p, n as u64, seed_of(k), out.ptr(), stream.raw()) })?;
    }
    stream.sync()?;
    let mut sum = [0u64; 1];
    out.download(&mut sum)?;
    Ok(sum[0])
}

/// The digest of device ranges ([`sum_device`] then [`finish`]); equals [`digest_host`] over the same bytes.
pub fn digest_device(ranges: &[(*const c_void, usize)], stream: &Stream) -> Result<u64, CudaError> {
    Ok(finish(sum_device(ranges, stream)?, ranges.iter().map(|r| r.1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(n: usize, salt: u64) -> Vec<u8> {
        (0..n).map(|i| (mix64(i as u64 ^ salt) & 0xff) as u8).collect()
    }

    #[test]
    fn the_host_digest_sees_one_flipped_bit_a_moved_word_a_zeroed_range_and_a_swapped_range() {
        let a = bytes(4096 + 3, 1);
        let b = bytes(1000, 2);
        let base = digest_host(&[&a, &b]);
        assert_eq!(base, digest_host(&[&a, &b]), "deterministic");
        let mut flip = a.clone();
        flip[2049] ^= 0x10;
        assert_ne!(base, digest_host(&[&flip, &b]));
        let mut tail = a.clone();
        *tail.last_mut().unwrap() ^= 1;
        assert_ne!(base, digest_host(&[&tail, &b]), "the trailing bytes count");
        let mut moved = a.clone();
        moved.swap(0, 4);
        moved.swap(1, 5);
        moved.swap(2, 6);
        moved.swap(3, 7);
        assert_ne!(base, digest_host(&[&moved, &b]), "a word moved to another position counts");
        assert_ne!(base, digest_host(&[&vec![0u8; a.len()], &b]), "an empty (zero) restore differs");
        assert_ne!(base, digest_host(&[&b, &a]), "ranges are keyed by position");
        assert_ne!(base, digest_host(&[&a[..a.len() - 1], &b]), "a shorter range differs");
        assert_ne!(digest_host(&[&[], &b]), digest_host(&[&b, &[]]), "an empty range still has a place");
    }

    #[test]
    fn the_device_digest_equals_the_host_twin_and_sees_a_stale_copy() {
        if !crate::device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
        let st = Stream::new().unwrap();
        // sizes around the vector path's edges, an odd tail, a 16-byte-misaligned (4-aligned) start
        let a = bytes((1 << 20) + 12 + 2, 3);
        let b = bytes(37, 4);
        let da = DevBuf::from_slice(&a).unwrap();
        let db = DevBuf::from_slice(&b).unwrap();
        let ranges = [(da.ptr() as *const c_void, a.len()), (db.ptr() as *const c_void, b.len())];
        assert_eq!(digest_device(&ranges, &st).unwrap(), digest_host(&[&a, &b]));
        let off = [(da.ptr_at(4) as *const c_void, a.len() - 4)];
        assert_eq!(digest_device(&off, &st).unwrap(), digest_host(&[&a[4..]]), "a start that is 4- but not 16-aligned");
        // a range the restore did not write (it still holds the previous state) differs
        let stale = bytes(a.len(), 5);
        let ds = DevBuf::from_slice(&stale).unwrap();
        assert_ne!(digest_device(&[(ds.ptr() as *const c_void, a.len()), ranges[1]], &st).unwrap(), digest_host(&[&a, &b]));
        assert!(digest_device(&[(da.ptr_at(2) as *const c_void, 8)], &st).is_err(), "a misaligned range is refused");
    }

    /// The cost per GiB on this device (report only): `cargo test -p gb-cuda --release state_digest -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing: run alone on an idle GB10"]
    fn the_device_digest_cost_per_gib() {
        if !crate::device_available() {
            return;
        }
        let st = Stream::new().unwrap();
        let n = 1usize << 30;
        // device memory, and the pageable host memory a 1M-token DeepSeek slot's planes live in (read in place by the GPU)
        for (what, d) in [("device", DevBuf::zeroed(n).unwrap()), ("pageable host", DevBuf::zeroed_pageable_view(n).unwrap())] {
            let r = [(d.ptr() as *const c_void, n)];
            digest_device(&r, &st).unwrap();
            let t = std::time::Instant::now();
            let reps = 10;
            for _ in 0..reps {
                digest_device(&r, &st).unwrap();
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            eprintln!("state_digest {what} 1 GiB: {ms:.2} ms ({:.1} GB/s)", n as f64 / ms / 1e6);
        }
    }
}
