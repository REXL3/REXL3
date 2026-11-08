//! The CPU reference entropy encoder / packer for EXL3 trellis tiles: the exact inverse of
//! [`crate::states`]. It turns the 256 16-bit window states (`[u16; 256]`) of one 16x16 tile
//! into the tile's `16 * bits` `i16` words, so that
//!
//! ```text
//! states(&pack_tile(&idx, bits), bits, &bits.stream_ends()) == idx
//! ```
//!
//! # Bit layout (the inverse of `states`)
//!
//! A tile's bitstream is a circular ring of `256 * bits` bits, kept as `nw = 8 * bits` 32-bit
//! words (two `i16` words each, little-endian, low word first — exactly the layout `states` reads).
//! Bit `b` of the ring is bit `b % 32` of ring word `b / 32`, with the ring wrapping modulo
//! `32 * nw`. Within a word the bit order is reversed from the natural one: ring bit `32*i + j`
//! is bit `31 - j` of word `i`, so the 16-bit window ending at `E(p)` (the ring position where the
//! `16` bits of position `p` end, exclusively) is read by `states` as a big-endian window: ring
//! bit `E(p) - 16 + k` is state bit `15 - k`.
//!
//! Consecutive windows slide by `D(p) = E(p+1) - E(p)` bits (the integer width for integer `Bits`,
//! alternating `ceil`/`floor` for half-integer ones), so windows overlap: the `16 - D(p)` high
//! bits of state `p+1` are the low bits of state `p`. Only index vectors that respect this sharing
//! (a valid trellis path — exactly the vectors `states` can produce from *some* ring) can be
//! encoded losslessly; they are in one-to-one correspondence with the ring's `256 * bits` free
//! bits (for a width of `b` integer bits the constraint makes every state the previous one shifted
//! left `b` with `b` new low bits, so `idx[p] == idx[p + 16 / b]`). `pack_tile` writes the full
//! 16-bit window of every position (each ring bit is written by every window that contains it; the
//! windows of a valid path agree on every shared bit, so the writes converge on the ring the states
//! came from). Inputs that are not on a path have no lossless encoding (the packed words are still
//! deterministic, with the last writer of each shared bit winning), and no check is attempted — an
//! impossibility check is the roundtrip itself.

use crate::{states, Bits};

/// The 256 16-bit window states of one 16x16 tile, in stream order, as its `16 * bits` `i16` words.
///
/// Exactly the inverse of [`crate::states`]: for every index vector `idx` that lies on a valid
/// trellis path,
///
/// ```text
/// states(&pack_tile(&idx, bits), bits, &bits.stream_ends()) == idx
/// ```
///
/// and the packed words are the exact ring `states` read to produce `idx`.
pub fn pack_tile(indices: &[u16; 256], bits: Bits) -> Vec<i16> {
    let nw16 = bits.tile_words();
    let nw = nw16 / 2;
    let ends = bits.stream_ends();
    let ring = 32 * nw as u64;
    let mut words = vec![0u64; nw];
    for (p, &v) in indices.iter().enumerate() {
        // The same arithmetic `states` uses: the window's first bit is E(p) - 16 (mod ring).
        let first = ends[p] as u64 + ring - 16;
        let i0 = ((first / 32) % nw as u64) as usize;
        let off = first % 32;
        let i1 = (i0 + 1) % nw;
        // `states` reads out[p] bit k from pair bit (48 - off + k): word i0 bit (16 - off + k)
        // when k >= off - 16, word i1 bit (48 - off + k) otherwise. Write v's bits there.
        for k in 0..16u64 {
            let b = (v >> k) & 1;
            let (wi, wb) = if 16 + k >= off { (i0, 16 + k - off) } else { (i1, 48 + k - off) };
            if b == 1 {
                words[wi] |= 1u64 << wb;
            }
        }
    }
    let mut out = vec![0i16; nw16];
    for (i, &w) in words.iter().enumerate() {
        out[2 * i] = (w & 0xFFFF) as u16 as i16;
        out[2 * i + 1] = ((w >> 16) & 0xFFFF) as u16 as i16;
    }
    out
}

/// Pack a trellis of `[kt, nt, 256]` tile indices (row-major, tiles in stream order) into its
/// `[kt, nt, 16 * bits]` `i16` words: the concatenation of [`pack_tile`] over the tiles.
pub fn pack_trellis(indices: &[u16], kt: usize, nt: usize, bits: Bits) -> Vec<i16> {
    assert_eq!(indices.len(), kt * nt * 256, "indices must hold kt * nt * 256 states");
    let mut out = Vec::with_capacity(kt * nt * bits.tile_words());
    for chunk in indices.chunks_exact(256) {
        let mut tile = [0u16; 256];
        tile.copy_from_slice(chunk);
        out.extend_from_slice(&pack_tile(&tile, bits));
    }
    out
}

/// The `[kt, nt, 256]` 16-bit window states of a packed trellis `[kt, nt, 16 * bits]` (row-major),
/// exactly [`crate::unpack`]'s first step per tile.
pub fn unpack_trellis_states(trellis: &[i16], kt: usize, nt: usize, bits: Bits) -> Vec<u16> {
    let nw = bits.tile_words();
    assert_eq!(trellis.len(), kt * nt * nw, "trellis size does not match [K/16, N/16, 16*bits]");
    let ends = bits.stream_ends();
    let mut out = Vec::with_capacity(kt * nt * 256);
    for chunk in trellis.chunks_exact(nw) {
        out.extend_from_slice(&states(chunk, bits, &ends));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bits;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// A random ring: `nw` random 32-bit words as the tile's `i16` words. Its states are by
    /// construction on a valid trellis path, so packing them must reproduce the ring word for word.
    fn random_ring(r: &mut Rng, bits: Bits) -> Vec<i16> {
        let nw = bits.tile_words() / 2;
        let mut tile = vec![0i16; bits.tile_words()];
        for i in 0..nw {
            let w = r.next() as u32;
            tile[2 * i] = (w & 0xffff) as u16 as i16;
            tile[2 * i + 1] = ((w >> 16) & 0xffff) as u16 as i16;
        }
        tile
    }

    #[test]
    fn pack_tile_roundtrips_all_widths() {
        for &twice in &Bits::ALL_TWICE {
            let bits = Bits::from_twice(twice).unwrap();
            let mut r = Rng(0x9E37_79B9_7F4A_7C15 ^ twice as u64);
            for trial in 0..64 {
                let ring = random_ring(&mut r, bits);
                let ends = bits.stream_ends();
                let idx: [u16; 256] = states(&ring, bits, &ends);
                let packed = pack_tile(&idx, bits);
                // every 16-bit window state read back identically
                assert_eq!(states(&packed, bits, &ends), idx, "state roundtrip at {} bits, trial {trial}", bits.as_f64());
                // and the packed words are the exact ring the states came from
                assert_eq!(packed, ring, "word-exact ring at {} bits, trial {trial}", bits.as_f64());
            }
        }
    }

    #[test]
    fn pack_tile_words_fit_the_states_layout() {
        for &twice in &Bits::ALL_TWICE {
            let bits = Bits::from_twice(twice).unwrap();
            let t = pack_tile(&[0u16; 256], bits);
            assert_eq!(t.len(), bits.tile_words(), "{} bits", bits.as_f64());
            // zeros pack to zeros, and every word's two i16 halves are the u32 ring words
            let nw = bits.tile_words() / 2;
            let words: Vec<u64> = (0..nw).map(|i| (t[2 * i] as u16 as u64) | ((t[2 * i + 1] as u16 as u64) << 16)).collect();
            assert!(words.iter().all(|&w| w == 0));
        }
    }

    #[test]
    fn trellis_pack_unpack_roundtrips_all_widths() {
        for &twice in &Bits::ALL_TWICE {
            let bits = Bits::from_twice(twice).unwrap();
            let (kt, nt) = (3usize, 5usize);
            let mut r = Rng(0x2545_F491_4F6C_DD1D ^ twice as u64);
            let ends = bits.stream_ends();
            let rings: Vec<Vec<i16>> = (0..kt * nt).map(|_| random_ring(&mut r, bits)).collect();
            let idx: Vec<u16> = rings.iter().flat_map(|ring| states(ring, bits, &ends)).collect();
            let trellis = pack_trellis(&idx, kt, nt, bits);
            assert_eq!(trellis.len(), kt * nt * bits.tile_words());
            let got = unpack_trellis_states(&trellis, kt, nt, bits);
            assert_eq!(got, idx, "trellis state roundtrip at {} bits", bits.as_f64());
            let want: Vec<i16> = rings.into_iter().flatten().collect();
            assert_eq!(trellis, want, "trellis words at {} bits", bits.as_f64());
        }
    }
}
