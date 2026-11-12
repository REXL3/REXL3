//! Tensor-parallel width splitting of EXL3 groups (`docs/CLUSTER_TP2.md` M0).
//!
//! MoE-width TP: every expert's gate/up (N = intermediate width) and down (K = the same width, input side)
//! are split across R ranks at 128-column granularity — the trellis tile/Hadamard block size, so a slice
//! is whole tiles and needs no re-encoding. Three split modes:
//!
//! * **Uneven 128-aligned:** rank r takes a contiguous run of 128-tiles; total = every tile
//!   exactly once, zero padding. Ranks may differ in width (`⌈B/R⌉` vs `⌊B/R⌋` tiles), so kernel plans
//!   differ per rank; the combine barrier absorbs the skew. The extra tiles go to the LOW ranks.
//! * **Floor 128-aligned (the engine's rule):** the same, with rank r holding tiles
//!   `[⌊B r/R⌋, ⌊B (r+1)/R⌋)` ([`tile_cut`]). This is what the Nemotron tensor-parallel engine runs
//!   (`gb-model` `tp_shard::column_range`, a test there pins the two against each other); at two ranks
//!   and an odd `B` the HIGH rank holds the odd tile, the mirror image of Uneven. Take this one for a
//!   loader or a memory plan that must agree with an engine.
//! * **Uniform padded (fallback):** every rank runs `W' = 128·⌈B/R⌉` with dummy zero tiles appended —
//!   zeroed trellis rows AND zeroed output scales, so the dequantized columns are bit-zero and contribute
//!   exactly nothing to the allreduce (the combine adds partials; a zero partial is identity). Use when
//!   identical shapes across ranks matter (graph capture).
//!
//! Everything here is CPU: pure split functions over [`Group`] geometry plus
//! [`load_group_slice`], which reads only the slice's bytes from disk. The M1 ring allreduce consumes the
//! partials; nothing in this module touches a GPU or a collective.
//!
//! Trellis layout (`parse_group`, `LoadedGroup`): `[K/16, N/16, 16·bits]` i16. One N-tile = 8 `N/16`
//! rows; for K-row kk and N-tile b: bytes `[kk·nt + b·8 .. kk·nt + b·8 + 8) · tw` i16 each, `tw = 16·bits`.
//! Gate/up split on N (output side): trellis N-rows, `svh` (length N) and `bias` (length N) sliced with it;
//! `suh` (length K) stays whole. Down-projection: K = the same width, so its trellis K-rows are sliced
//! the same way with `suh` (length K) and `bias` absent — `svh` (length D) stays whole. This mirrors the
//! proven test-only `slice_blocks` in `crates/cuda/src/exl3_linear.rs` (the output rotation is per block,
//! so other columns never enter a block's arithmetic).

use crate::checkpoint::{Checkpoint, LoadedGroup};
use crate::Bits;
use std::io;

/// The 128-column tile the trellis and Hadamard rotation are built from (`HAD`).
pub const TILE: usize = 128;

/// How a rank's share of the width is decided.
///
/// **Which rule an engine uses.** The Nemotron tensor-parallel engine (`gb-model`, `nemotron_h_omni::tp_shard::column_range`, the rule
/// of every verified two-rank run: 9,216 | 9,344 columns of the 145-tile `in_proj`) cuts at `floor(tiles * rank / ranks)`: that is
/// [`SplitMode::Floor`], the one rule a memory plan, a loader and a kernel proof must share. [`SplitMode::Uneven`] is the earlier
/// "extra tiles to the LOW ranks" rule of the GLM CPU reference; at two ranks it is the mirror image of the engine's (73 | 72 tiles of
/// `in_proj`), so mixing the two in one deployment would misplace a slice. It is kept bit for bit as it was (the proofs built on it
/// do not depend on which rank holds the extra tile); new code takes [`SplitMode::Floor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitMode {
    /// Contiguous whole tiles, every tile on exactly one rank (zero padding; ranks may differ in width). The extra tiles of an uneven
    /// width go to the LOW ranks (`ranks < rem` get `base + 1`).
    Uneven,
    /// Equal width everywhere, dummy zero tiles appended (identical shapes; bit-zero contribution).
    Uniform,
    /// Contiguous whole tiles, every tile on exactly one rank, rank `r` holding `[floor(B r / R), floor(B (r + 1) / R))` of the `B`
    /// tiles ([`tile_cut`]): the engine's rule. The extra tiles of an uneven width are spread by that floor (the HIGH rank of two
    /// holds the odd tile), skew at most one tile between any two ranks.
    Floor,
}

/// The tile range `[lo, hi)` rank `rank` of `ranks` holds of `tiles` 128-column tiles under [`SplitMode::Floor`]: `floor(tiles * rank
/// / ranks)` to `floor(tiles * (rank + 1) / ranks)`. Every tile on exactly one rank, contiguous, ascending, no rank more than one tile
/// wider than another; a rank holds NO tile when `tiles < ranks` (a caller must replicate such a matrix: there is no zero-width EXL3
/// layer). For two ranks and the 145 tiles of the Nemotron `in_proj`: `[0, 72)` and `[72, 145)`.
pub fn tile_cut(tiles: usize, rank: usize, ranks: usize) -> (usize, usize) {
    assert!(ranks >= 1, "no ranks");
    assert!(rank < ranks, "rank {rank} out of {ranks}");
    (tiles * rank / ranks, tiles * (rank + 1) / ranks)
}

/// One rank's width assignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WidthSplit {
    /// Tiles assigned to this rank, contiguous ascending (dummy tiles counted in `width`).
    pub tiles: Vec<usize>,
    /// Width this rank runs (tiles · [`TILE`]).
    pub width: usize,
    /// Trailing dummy tiles (uniform mode only; always 0 in uneven).
    pub dummy: usize,
}

impl WidthSplit {
    /// This rank's real column range `[lo, hi)` in the full tensor (dummy tiles excluded).
    pub fn cols(&self) -> (usize, usize) {
        let real = self.tiles.len() - self.dummy;
        let lo = self.tiles.first().map_or(0, |&t| t * TILE);
        (lo, lo + real * TILE)
    }
}

/// Split `n` columns (a multiple of [`TILE`]) into `ranks` assignments; rank `r`'s share.
///
/// Uneven: B = n/TILE tiles, base = B/ranks, rem = B%ranks — ranks `< rem` get base+1 tiles, the rest
/// base; rank r's tiles are the contiguous run starting at `r·base + min(r, rem)`. Every tile exactly
/// once, cover complete, skew ≤ 1 tile between adjacent ranks.
///
/// Floor (the Nemotron engine's rule, [`tile_cut`]): rank r owns tiles `[⌊B r/ranks⌋, ⌊B (r+1)/ranks⌋)`; every tile exactly once,
/// the extra tiles of `B % ranks` spread by the floor (for two ranks and an odd `B` the HIGH rank holds the odd tile: the mirror image
/// of Uneven), skew ≤ 1 tile.
///
/// Uniform: W' = 128·⌈B/ranks⌉; rank r owns tiles `[r·W'/128, (r+1)·W'/128)`, the run past B is dummy.
pub fn split_width(n: usize, r: usize, ranks: usize, mode: SplitMode) -> WidthSplit {
    assert!(ranks >= 2, "TP needs at least 2 ranks");
    assert!(r < ranks, "rank {r} out of {ranks}");
    assert!(n % TILE == 0, "width {n} is not a multiple of the {TILE} tile");
    let b = n / TILE;
    match mode {
        SplitMode::Uneven => {
            let (base, rem) = (b / ranks, b % ranks);
            let start = r * base + r.min(rem);
            let len = base + usize::from(r < rem);
            WidthSplit { tiles: (start..start + len).collect(), width: len * TILE, dummy: 0 }
        }
        SplitMode::Floor => {
            let (start, end) = tile_cut(b, r, ranks);
            WidthSplit { tiles: (start..end).collect(), width: (end - start) * TILE, dummy: 0 }
        }
        SplitMode::Uniform => {
            let tiles_per = b.div_ceil(ranks);
            let (lo, hi) = (r * tiles_per, (r + 1) * tiles_per);
            WidthSplit { tiles: (lo..hi).collect(), width: tiles_per * TILE, dummy: hi.saturating_sub(b).min(tiles_per) }
        }
    }
}

/// Sides of a projection that a width split touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// N (output) side: trellis N-rows, `svh`, `bias` sliced; `suh` whole (gate/up projections).
    Output,
    /// K (input) side: trellis K-rows, `suh` sliced; `svh` whole (down projections).
    Input,
}

/// Which side a projection's split is on, from its role in the FFN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proj {
    GateUp,
    Down,
}

impl Proj {
    pub fn side(&self) -> Side {
        match self {
            Proj::GateUp => Side::Output,
            Proj::Down => Side::Input,
        }
    }
}

/// i16 lanes of one trellis row: `16 · bits` (the last trellis axis).
fn tw_of(bits: Bits) -> usize {
    16 * bits.twice() as usize / 2
}

/// Read exactly `count` i16 lanes starting at lane `start` of a 16-bit tensor entry (preadv-free serial
/// read; the slice path is per-rank bandwidth-bound, and one Open + seek + read is the same shape
/// `read_bytes` uses).
fn read_i16_run(ck: &Checkpoint, name: &str, lane_start: usize, count: usize) -> io::Result<Vec<i16>> {
    let e = ck
        .part(name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("{name} is not part of a scanned group")))?;
    let off = ck.file_offset(e)?;
    let bytes = e.nbytes();
    if (lane_start + count) as u64 * 2 > bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{name}: lanes [{lane_start}, {}) past the tensor's {} bytes", lane_start + count, bytes),
        ));
    }
    use std::io::Read;
    let mut f = std::fs::File::open(ck.root.join(&e.file))?;
    use std::io::Seek;
    f.seek(std::io::SeekFrom::Start(off + lane_start as u64 * 2))?;
    let mut raw = vec![0u8; count * 2];
    f.read_exact(&mut raw)?;
    Ok(raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect())
}

/// Read the f16 scale lanes `[lane_start, lane_start + count)` of a scale tensor.
fn read_scale_run(ck: &Checkpoint, name: &str, lane_start: usize, count: usize) -> io::Result<Vec<f64>> {
    let e = ck
        .part(name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("{name} is not part of a scanned group")))?;
    let off = ck.file_offset(e)?;
    if (lane_start + count) as u64 * 2 > e.nbytes() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{name}: scale lanes past the tensor ({lane_start}+{count} of {})", e.nbytes() / 2),
        ));
    }
    use std::io::{Read, Seek};
    let mut f = std::fs::File::open(ck.root.join(&e.file))?;
    f.seek(std::io::SeekFrom::Start(off + lane_start as u64 * 2))?;
    let mut raw = vec![0u8; count * 2];
    f.read_exact(&mut raw)?;
    Ok(raw.chunks_exact(2).map(|c| crate::f16::f16_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect())
}

/// The rank-r slice of one group under a width split. `proj` picks the side (gate/up → Output, down →
/// Input); `mode` uneven or uniform. For `Uniform`, dummy tiles are materialized as zeroed trellis lanes
/// AND zeroed scales, so their columns are bit-zero through dequantization.
///
/// The returned [`LoadedGroup`] is a valid group in its own right: parse-compatible geometry (whole
/// tiles), correct scale vectors, and — for Output-side slices with a bias — the bias sliced to the
/// rank's columns. Bias parts across ranks are disjoint column ranges, so the allreduce's sum is the
/// full bias: every rank keeps its own columns' slice, none zeroed.
pub fn load_group_slice(
    ck: &Checkpoint,
    prefix: &str,
    r: usize,
    ranks: usize,
    mode: SplitMode,
    proj: Proj,
) -> io::Result<LoadedGroup> {
    let g0 =
        ck.groups.get(prefix).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("no EXL3 group {prefix}")))?;
    // The width being split is the projection's "width" side: N for gate/up, K for down.
    let width = match proj.side() {
        Side::Output => g0.n,
        Side::Input => g0.k,
    };
    let ws = split_width(width, r, ranks, mode);
    let tw = tw_of(g0.bits);

    let mut group = g0.clone();
    let (trellis, sliced_scale, whole_scale, bias) = match proj.side() {
        Side::Output => {
            // trellis [K/16, N/16, tw]: for K-row kk, N-tile b: lanes (kk*nt + b*8)*tw .. +8*tw
            let nt = g0.n / 16;
            let real_tiles: Vec<usize> = ws.tiles.iter().copied().take(ws.tiles.len() - ws.dummy).collect();
            let mut trellis = Vec::with_capacity(g0.k / 16 * ws.tiles.len() * 8 * tw);
            for kk in 0..g0.k / 16 {
                for &b in &ws.tiles {
                    let tile = if b < nt / 8 {
                        read_i16_run(ck, &format!("{prefix}.trellis"), (kk * nt + b * 8) * tw, 8 * tw)?
                    } else {
                        vec![0i16; 8 * tw]
                    };
                    trellis.extend_from_slice(&tile);
                }
            }
            let svh = {
                let mut v = Vec::with_capacity(ws.width);
                for &b in &real_tiles {
                    let mut s = read_scale_run(ck, &format!("{prefix}.{}", g0.out_scales), b * TILE, TILE)?;
                    v.append(&mut s);
                }
                v.extend(std::iter::repeat_n(0.0, ws.dummy * TILE));
                v
            };
            let bias = if g0.bias {
                let e = ck
                    .part(&format!("{prefix}.bias"))
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("{prefix}.bias missing")))?;
                let full = ck.read_floats(e)?;
                let mut v = Vec::with_capacity(ws.width);
                for &b in &real_tiles {
                    v.extend_from_slice(&full[b * TILE..(b + 1) * TILE]);
                }
                v.extend(std::iter::repeat_n(0.0, ws.dummy * TILE));
                Some(v)
            } else {
                None
            };
            group.n = ws.width;
            (trellis, None::<Vec<f64>>, svh, bias)
        }
        Side::Input => {
            // trellis [K/16, D/16, tw]: K-tiles are rows; for K-tile b, take rows [b*8, b*8+8)
            let kt = g0.k / 16;
            let real_tiles: Vec<usize> = ws.tiles.iter().copied().take(ws.tiles.len() - ws.dummy).collect();
            let mut trellis = Vec::with_capacity(ws.tiles.len() * 8 * (g0.n / 16) * tw);
            for &b in &ws.tiles {
                if b < kt / 8 {
                    let start = b * 8 * g0.n / 16 * tw;
                    let run = read_i16_run(ck, &format!("{prefix}.trellis"), start, 8 * (g0.n / 16) * tw)?;
                    trellis.extend_from_slice(&run);
                } else {
                    // dummy K-tile: whole rows of zero lanes (bit-zero through dequantization)
                    trellis.extend(std::iter::repeat_n(0i16, 8 * (g0.n / 16) * tw));
                }
            }
            let suh = {
                let mut v = Vec::with_capacity(ws.width);
                for &b in &real_tiles {
                    let mut s = read_scale_run(ck, &format!("{prefix}.{}", g0.in_scales), b * TILE, TILE)?;
                    v.append(&mut s);
                }
                v.extend(std::iter::repeat_n(0.0, ws.dummy * TILE));
                v
            };
            group.k = ws.width;
            (trellis, Some(suh), Vec::new(), None)
        }
    };
    let _ = sliced_scale;
    // whole-side scales: read in full for the side not split
    let (suh, svh) = match proj.side() {
        Side::Output => {
            let e = ck
                .part(&format!("{prefix}.{}", g0.in_scales))
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("{prefix}.{} missing", g0.in_scales)))?;
            (ck.read_scale(e)?, whole_scale)
        }
        Side::Input => {
            let e = ck
                .part(&format!("{prefix}.{}", g0.out_scales))
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("{prefix}.{} missing", g0.out_scales)))?;
            (sliced_scale.unwrap_or_default(), ck.read_scale(e)?)
        }
    };
    Ok(LoadedGroup {
        group,
        trellis,
        kt: match proj.side() {
            Side::Output => g0.k / 16,
            Side::Input => ws.tiles.len() * 8,
        },
        nt: match proj.side() {
            Side::Output => ws.tiles.len() * 8,
            Side::Input => g0.n / 16,
        },
        suh,
        svh,
        bias,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uneven_splits_cover_every_tile_exactly_once() {
        for (n, ranks) in [(2048usize, 3usize), (2304, 5), (2048, 5), (512, 3), (640, 4), (128, 2), (2048, 2), (2048, 8)] {
            let mut all = std::collections::BTreeSet::new();
            let mut widths = vec![];
            for r in 0..ranks {
                let ws = split_width(n, r, ranks, SplitMode::Uneven);
                assert_eq!(ws.dummy, 0);
                assert_eq!(ws.width, ws.tiles.len() * TILE);
                // contiguity
                assert!(ws.tiles.windows(2).all(|w| w[1] == w[0] + 1), "{n}/{ranks} r{r}: {:?}", ws.tiles);
                all.extend(ws.tiles.iter().copied());
                widths.push(ws.width);
            }
            assert_eq!(all.len(), n / TILE, "{n}/{ranks}: cover");
            assert_eq!(*all.iter().next().unwrap(), 0);
            assert_eq!(*all.iter().last().unwrap(), n / TILE - 1);
            let skew = widths.iter().max().unwrap() - widths.iter().min().unwrap();
            assert!(skew <= TILE, "{n}/{ranks}: skew {skew} > one tile");
        }
    }

    #[test]
    fn floor_splits_cover_every_tile_once_in_ascending_runs_with_at_most_one_tile_of_skew() {
        for ranks in [2usize, 3, 4, 5, 8] {
            for n in [128usize, 256, 2048, 2304, 4096, 18560, 131072] {
                let b = n / TILE;
                let mut at = 0;
                let (mut lo_w, mut hi_w) = (usize::MAX, 0usize);
                for r in 0..ranks {
                    let ws = split_width(n, r, ranks, SplitMode::Floor);
                    assert_eq!(ws.dummy, 0);
                    assert_eq!(ws.width, ws.tiles.len() * TILE);
                    assert_eq!(ws.tiles.first().copied().unwrap_or(at), at, "{n}/{ranks} r{r}: ascending, no gap");
                    at += ws.tiles.len();
                    lo_w = lo_w.min(ws.tiles.len());
                    hi_w = hi_w.max(ws.tiles.len());
                    // the same numbers as the rule's own function
                    let (s, e) = tile_cut(b, r, ranks);
                    assert_eq!(ws.cols(), if e > s { (s * TILE, e * TILE) } else { (0, 0) }, "{n}/{ranks} r{r}");
                }
                assert_eq!(at, b, "{n}/{ranks}: every tile once");
                assert!(hi_w - lo_w <= 1, "{n}/{ranks}: skew {} tiles", hi_w - lo_w);
            }
        }
    }

    #[test]
    fn the_engines_rule_gives_the_odd_tile_to_the_high_rank_and_uneven_to_the_low_one() {
        // the Nemotron in_proj: 145 tiles. Floor is the verified engine's 9,216 | 9,344, Uneven its mirror image
        assert_eq!(split_width(18560, 0, 2, SplitMode::Floor).cols(), (0, 9216));
        assert_eq!(split_width(18560, 1, 2, SplitMode::Floor).cols(), (9216, 18560));
        assert_eq!(split_width(18560, 0, 2, SplitMode::Uneven).cols(), (0, 9344));
        assert_eq!(split_width(18560, 1, 2, SplitMode::Uneven).cols(), (9344, 18560));
        // a hidden-wide matrix (32 tiles) at three ranks: 10 | 11 | 11 tiles (Floor), 11 | 11 | 10 (Uneven)
        let w = |mode| (0..3).map(|r| split_width(4096, r, 3, mode).tiles.len()).collect::<Vec<_>>();
        assert_eq!(w(SplitMode::Floor), vec![10, 11, 11]);
        assert_eq!(w(SplitMode::Uneven), vec![11, 11, 10]);
        // an even split is the same under both rules
        for r in 0..2 {
            assert_eq!(split_width(4096, r, 2, SplitMode::Floor), split_width(4096, r, 2, SplitMode::Uneven));
        }
        // fewer tiles than ranks: Floor leaves ranks with no tile (the caller replicates the matrix)
        let held: Vec<usize> = (0..3).map(|r| split_width(256, r, 3, SplitMode::Floor).tiles.len()).collect();
        assert_eq!(held, vec![0, 1, 1]);
    }

    #[test]
    fn uniform_splits_are_equal_with_trailing_dummies() {
        for (n, ranks) in [(2048usize, 3usize), (2304, 4), (2048, 5), (2304, 7)] {
            let w = 128 * (n / 128).div_ceil(ranks);
            let mut total_dummy = 0;
            for r in 0..ranks {
                let ws = split_width(n, r, ranks, SplitMode::Uniform);
                assert_eq!(ws.width, w, "{n}/{ranks} r{r}");
                assert_eq!(ws.tiles.len() * TILE, w);
                total_dummy += ws.dummy;
            }
            assert_eq!(ranks * w - total_dummy * TILE, n, "{n}/{ranks}: exact total");
        }
    }

    #[test]
    fn cols_reports_the_real_range() {
        // 2048 / 5 uniform: 4 tiles (512) a rank, tiles 16..20 are dummy -> rank 4 holds no real columns
        let ws = split_width(2048, 4, 5, SplitMode::Uniform);
        assert_eq!(ws.cols(), (2048, 2048));
        let ws = split_width(2048, 2, 5, SplitMode::Uniform);
        assert_eq!(ws.cols(), (1024, 1536));
        let ws = split_width(2048, 0, 3, SplitMode::Uneven);
        assert_eq!(ws.cols(), (0, 768));
        // uneven rank 2 of 3 on 2048: B=16, base=5, rem=1 -> r0=6 tiles, r1/r2=5; r2 = [11,16) -> (1408, 2048)
        let ws = split_width(2048, 2, 3, SplitMode::Uneven);
        assert_eq!(ws.cols(), (1408, 2048));
    }

    #[test]
    fn dummy_tiles_only_past_the_real_width() {
        for (n, ranks) in [(2048usize, 3usize), (2304, 4), (2048, 5), (2304, 7)] {
            let b = n / TILE;
            for r in 0..ranks {
                let ws = split_width(n, r, ranks, SplitMode::Uniform);
                let real = ws.tiles.len() - ws.dummy;
                assert!(ws.tiles[..real].iter().all(|&t| t < b), "{n}/{ranks} r{r}");
                assert!(ws.tiles[real..].iter().all(|&t| t >= b), "{n}/{ranks} r{r}");
            }
        }
    }

    /// The numerical proof, on a real pack (REXL3_TP_PACK): the whole group's CPU reference forward
    /// equals the ranks' slice forwards. Output side (gate/up): bit-exact by CONCATENATION — each
    /// output column's arithmetic (input Hadamard over full K, per-column accumulation over i=0..K,
    /// per-128-block output Hadamard on tile-aligned blocks) is identical in the slice and the whole.
    /// Input side (down): the SUM of rank partials equals the whole within float64 regrouping —
    /// the whole accumulates i=0..K sequentially; rank partials accumulate their sub-ranges, so the
    /// sum over ranks (in rank order) differs only by addition regrouping at rank boundaries. That
    /// regrouping is the combine's fixed-order contract (M1/M2): recorded here as the tolerance case.
    #[test]
    fn real_pack_slice_forwards_match_the_whole_reference() {
        let Some(dir) = std::env::var("REXL3_TP_PACK").ok().filter(|p| !p.is_empty()) else {
            return; // env-gated: `not run` per repo convention when the pack is absent
        };
        let ck = crate::checkpoint::scan(std::path::Path::new(&dir), true).unwrap();
        let prefixes: Vec<String> = ck.groups.keys().cloned().collect();
        assert!(prefixes.len() >= 2);
        let mut x = 0x9e37_79b9u32;
        let mut rand = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x >> 8) as f64 / 256.0 - 0.5
        };
        let bits_of = |p: &str| ck.groups[p].bits;
        let cb_of = |p: &str| ck.groups[p].codebook;
        for (prefix, proj) in [(&prefixes[0], Proj::GateUp), (&prefixes[prefixes.len() - 1], Proj::Down)] {
            let whole = ck.load_group(prefix).unwrap();
            let table = crate::codebook(cb_of(prefix));
            let width = match proj.side() {
                Side::Output => whole.group.n,
                Side::Input => whole.group.k,
            };
            for ranks in [2usize, 3, 5] {
                if width / TILE < ranks {
                    continue;
                }
                for mode in [SplitMode::Uneven, SplitMode::Floor, SplitMode::Uniform] {
                    let rows = 4usize;
                    match proj.side() {
                        Side::Output => {
                            let x = (0..rows * whole.group.k).map(|_| rand()).collect::<Vec<_>>();
                            let want = crate::forward(
                                &x,
                                rows,
                                &whole.trellis,
                                whole.kt,
                                whole.nt,
                                bits_of(prefix),
                                &table,
                                &whole.suh,
                                &whole.svh,
                                whole.bias.as_deref(),
                            );
                            // concatenate the ranks' outputs at their real columns
                            let mut got = vec![0.0f64; rows * width];
                            for r in 0..ranks {
                                let sl = load_group_slice(&ck, prefix, r, ranks, mode, proj).unwrap();
                                let y = crate::forward(
                                    &x,
                                    rows,
                                    &sl.trellis,
                                    sl.kt,
                                    sl.nt,
                                    bits_of(prefix),
                                    &table,
                                    &sl.suh,
                                    &sl.svh,
                                    sl.bias.as_deref(),
                                );
                                let (lo, hi) = split_width(width, r, ranks, mode).cols();
                                for row in 0..rows {
                                    got[row * width + lo..row * width + hi]
                                        .copy_from_slice(&y[row * sl.group.n..row * sl.group.n + (hi - lo)]);
                                }
                            }
                            assert_eq!(got, want, "{prefix} Output {ranks} {mode:?}: bit-exact concatenation");
                        }
                        Side::Input => {
                            let x = (0..rows * whole.group.k).map(|_| rand()).collect::<Vec<_>>();
                            let want = crate::forward(
                                &x,
                                rows,
                                &whole.trellis,
                                whole.kt,
                                whole.nt,
                                bits_of(prefix),
                                &table,
                                &whole.suh,
                                &whole.svh,
                                whole.bias.as_deref(),
                            );
                            let mut got = vec![0.0f64; rows * whole.group.n];
                            // the regrouping bound needs the absolute-term scale: accumulate Σ|partial|
                            let mut abs_terms = vec![0.0f64; rows * whole.group.n];
                            for r in 0..ranks {
                                let ws = split_width(width, r, ranks, mode);
                                let sl = load_group_slice(&ck, prefix, r, ranks, mode, proj).unwrap();
                                // rank r's input: its real columns of x, zeros in dummy positions
                                let mut xr = vec![0.0f64; rows * sl.group.k];
                                let (lo, hi) = ws.cols();
                                for row in 0..rows {
                                    for (j, c) in (lo..hi).enumerate() {
                                        xr[row * sl.group.k + j] = x[row * whole.group.k + c];
                                    }
                                }
                                let y = crate::forward(
                                    &xr,
                                    rows,
                                    &sl.trellis,
                                    sl.kt,
                                    sl.nt,
                                    bits_of(prefix),
                                    &table,
                                    &sl.suh,
                                    &sl.svh,
                                    sl.bias.as_deref(),
                                );
                                for ((g, a), w) in got.iter_mut().zip(abs_terms.iter_mut()).zip(y.iter()) {
                                    *g += *w;
                                    *a += w.abs();
                                }
                            }
                            // Scale-aware regrouping bound: summing R f64 partial accumulations (each up
                            // to K terms) against the whole's single sequential pass differs by at most
                            // O(R·eps·Σ|terms|). A real slice bug (wrong rows, dropped tiles, misread
                            // scales) is orders of magnitude larger than this bound.
                            let eps = 1e-12;
                            let worst = got
                                .iter()
                                .zip(want.iter())
                                .zip(abs_terms.iter())
                                .map(|((&g, &w), &a)| (g - w).abs() / (a + eps))
                                .fold(0.0f64, f64::max);
                            assert!(
                                worst < eps,
                                "{prefix} Input {ranks} {mode:?}: sum-of-ranks vs whole, worst {worst:e} vs scale"
                            );
                            // record (not assert) whether this split happened to be bit-exact
                            let bit_exact = got == want;
                            eprintln!("{prefix} Input {ranks} {mode:?}: worst {worst:.2e} (scale-aware), bit_exact={bit_exact}");
                        }
                    }
                }
            }
        }
    }
}
