//! The Engram memory layers: a read-through view of the two FP8 token-lookup tables on trunk layers 1 and 14, hashed
//! exactly as the reference implementation does.
//!
//! An Engram is a memory: for every n-gram the model has seen, a row of learned vectors that the layer adds back into the
//! residual stream, gated by how well it matches that stream. The pack holds two of them, and they are the largest tensors
//! in it — layer 1 is `[384_006_168, 256]` `F8_E4M3` = 91.5 GiB, layer 14 is `[384_016_682, 256]` = 91.5 GiB, plus
//! 2.86 GiB of `F8_E8M0` block scales each. That is 203 GB against a 121 GiB box, so the one rule this module is built
//! around is that **the tables are never materialised**: [`MmapTable`] is a pair of read-only `mmap`s, every read is a
//! slice of the mapping, and no allocation in this file is sized by a row count. A row is 264 bytes on disk (256 fp8 +
//! 8 scale) and a token's gather is 24 of them.
//!
//! # The hash is not a choice
//!
//! The rows a token reads are a *function of the trained table*, so the hash has to be DeepSeek's, bit for bit; anything
//! else silently reads 24 unrelated rows out of 91.5 GiB and produces garbage. The three constant tables below are that
//! function, and every one of them is pinned by a test:
//!
//! * [`PRIMES`] — `2 layers x 3 n-gram orders x 8 heads = 48` distinct primes, drawn in order from the first prime above
//!   `engram_vocab_size - 1 = 15_999_999` with no reuse. Each `(order, head)` owns a prime-sized bucket range inside its
//!   layer's table, and the 24 ranges **tile it exactly**: `384_006_168` rows for layer 1 and `384_016_682` for layer 14.
//! * [`OFFSETS`] — where each of those 24 ranges starts, i.e. the running cumulative sum of the primes. Not a second
//!   thing to keep in sync: `the_primes_and_offsets_tile_both_tables_exactly` proves it.
//! * [`MULTIPLIERS`] — four odd per-(layer, look-back) constants, `2 * rng.integers(0, bound) + 1` from
//!   `numpy.random.default_rng(10007 * layer_id)`, with `bound = (i64::MAX / 129_280) / 2 = 35_672_076_256_399`. Odd and
//!   bounded so that `compressed_id * multiplier` cannot overflow `i64`.
//!
//! The hash itself ([`NgramHash`]) is a rolling XOR of the multiplied compressed ids over look-backs `0..=i`, reduced
//! modulo the `(order, head)` prime and shifted by its offset:
//!
//! ```text
//! c_s      = compressed[token_{p-s}]                 (compressed id, or pad_id once blocked)
//! rolling  = c_0 * MULT[0]  ^  c_1 * MULT[1]  ^  c_2 * MULT[2]  ^  c_3 * MULT[3]
//! row(i,j) = OFFSET[(i-1)*8 + j] + rolling % PRIME[i-1][j]      i = 1..=3, j = 0..8
//! ```
//!
//! Look-back is clamped at the start of the sequence and stops at any [`DEAD`] token (an image span): the n-gram is
//! filled with `pad_id` for the missing slots and stays filled for every longer order, so no n-gram ever spans a dead
//! token.
//!
//! # The compressed token map
//!
//! N-grams are hashed over *compressed* ids — one id per normalisation class of token text, so `" The"`, `"the"` and
//! `"THE"` hash alike. That map is 129,280 entries and its codomain is exactly 99,092 ids; both numbers matter, because
//! [`MULTIPLIER_BOUND`] was derived from the second one. It is built from the pack's `tokenizer.json` by a separate
//! script (see [`TokenMap`] for the exact file format) and loaded from [`MAP_ENV`] or an explicit path here; it is never
//! rebuilt at run time, because a tokenizer version skew would rehash the whole table silently.
//!
//! # What the pack pins, and what this module does not
//!
//! `engram_layer_ids = [1, 14]`, `engram_num_embeddings = [384_006_168, 384_016_682]`,
//! `engram_max_ngram_size = 4`, `engram_n_heads = 8`, `engram_head_dim = 256`,
//! `engram_compressed_vocab_size = 99_092`, `engram_pad_token_id = 2`. One row is 256 fp8 bytes + 8 scale bytes, and
//! `scale[j]` is the exponent of `2^(e-127)` covering columns `j*32 .. j*32+32`. `q_weight`/`k_weight` are bf16
//! `[4, 5120]`; `wkv` is FP8 `[5*5120, 3*8*256] = [25600, 6144]` with `[800, 192]` 32x32 block scales.
//!
//! Nothing here is a device path. [`MmapTable::gather_dequant`] is the host half of one — it dequantises a gather straight
//! out of the mapping into a caller-owned `f32` buffer — and the per-token cost is stated in
//! [`MmapTable::GATHER_BYTES_PER_TOKEN`].

use super::config::EngramConfig;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Columns covered by one scale byte: `scale[j]` scales `row[j*32 .. j*32+32]`.
pub const FP8_TILE: usize = 32;

/// The token ring's depth. Four is what [`MAX_NGRAM`] needs; 128 is what a decoding slot keeps anyway, for free.
pub const RING: usize = 128;

/// The n-gram window: `engram_max_ngram_size`. Orders 2..=4 are hashed, i.e. look-backs 0..=3.
pub const MAX_NGRAM: usize = 4;

/// Heads per order, and therefore hash columns per order: `engram_n_heads`.
pub const N_HEADS: usize = 8;

/// Orders hashed per position: `MAX_NGRAM - 1` — the 2-, 3- and 4-gram.
pub const ORDERS: usize = MAX_NGRAM - 1;

/// Hash ids one position reads: `ORDERS * N_HEADS = 24`, laid out order-major.
pub const HASH_COLS: usize = ORDERS * N_HEADS;

/// Entries in the compressed token map: one per tokenizer id.
pub const TOKEN_MAP_LEN: usize = 129_280;

/// Size of the compressed vocabulary — the codomain of [`TokenMap`], and the number every multiplier derives from.
pub const COMPRESSED_VOCAB: usize = 99_092;

/// `(i64::MAX / 129_280) / 2`, the exclusive bound the multipliers are drawn below before being doubled and made odd.
pub const MULTIPLIER_BOUND: i64 = 35_672_076_256_399;

/// The row count of each engram table, parallel to `engram_layer_ids = [1, 14]`.
pub const TABLE_ROWS: [usize; 2] = [384_006_168, 384_016_682];

/// A token that takes no part in an n-gram (an image span). It blocks every n-gram that spans it.
pub const DEAD: i64 = -1;

/// [`DEAD`] as the id array spells it (`i32`/`u32` casts of -1): an image-span position.
pub const DEAD_AS_U32: u32 = u32::MAX;

/// Where [`TokenMap::from_env`] looks for the compressed token map.
pub const MAP_ENV: &str = "REXL3_DSV41_ENGRAM_TOKEN_MAP";

/// Where the engram tables' shards live when the pack itself does not carry them: quantized packs
/// omit the two `[rows, 256]` F8 tables (they stay unquantized in the base model), so a boot points
/// this at the deepseek-ai base model dir — or an engram-only dir whose `model.safetensors.index.json`
/// lists the four table tensors — and [`super::engine`] resolves them from that scan.
pub const SHARDS_ENV: &str = "REXL3_DSV41_ENGRAM_SHARDS";

// ------------------------------------------------------------------------------------------------ the constant tables

/// Per-(layer, look-back) odd multipliers, `[layer][shift]`.
///
/// `numpy.random.default_rng(10007 * layer_id).integers(0, MULTIPLIER_BOUND, 4, int64) * 2 + 1`. Layer 1 is the first
/// row, layer 14 the second — indexed by position in `engram_layer_ids`, not by layer id.
pub const MULTIPLIERS: [[i64; MAX_NGRAM]; 2] = [
    [58_737_837_727_525, 3_709_723_095_905, 27_562_777_300_965, 56_710_653_182_447],
    [51_904_348_776_103, 39_482_587_155_911, 23_700_944_747_927, 63_326_921_340_579],
];

/// Bucket modulus per `(layer, order, head)`, `[layer][order - 2][head]`: 48 distinct primes above 15_999_999, drawn in
/// order and never reused, so the 24 ranges of a layer are disjoint.
pub const PRIMES: [[[i64; N_HEADS]; ORDERS]; 2] = [
    [
        [16_000_057, 16_000_079, 16_000_081, 16_000_097, 16_000_121, 16_000_129, 16_000_133, 16_000_183],
        [16_000_189, 16_000_207, 16_000_211, 16_000_253, 16_000_277, 16_000_289, 16_000_307, 16_000_321],
        [16_000_339, 16_000_381, 16_000_393, 16_000_399, 16_000_403, 16_000_409, 16_000_447, 16_000_463],
    ],
    [
        [16_000_477, 16_000_487, 16_000_499, 16_000_507, 16_000_511, 16_000_573, 16_000_609, 16_000_627],
        [16_000_667, 16_000_669, 16_000_693, 16_000_697, 16_000_711, 16_000_729, 16_000_759, 16_000_769],
        [16_000_781, 16_000_799, 16_000_813, 16_000_819, 16_000_841, 16_000_877, 16_000_879, 16_000_889],
    ],
];

/// Where each `(order, head)` bucket range starts, `[layer][(order - 2) * N_HEADS + head]`: the running cumulative sum
/// of [`PRIMES`] over the 24 ranges of the layer. The last range ends exactly at [`TABLE_ROWS`].
pub const OFFSETS: [[i64; HASH_COLS]; 2] = [
    [
        0,
        16_000_057,
        32_000_136,
        48_000_217,
        64_000_314,
        80_000_435,
        96_000_564,
        112_000_697,
        128_000_880,
        144_001_069,
        160_001_276,
        176_001_487,
        192_001_740,
        208_002_017,
        224_002_306,
        240_002_613,
        256_002_934,
        272_003_273,
        288_003_654,
        304_004_047,
        320_004_446,
        336_004_849,
        352_005_258,
        368_005_705,
    ],
    [
        0,
        16_000_477,
        32_000_964,
        48_001_463,
        64_001_970,
        80_002_481,
        96_003_054,
        112_003_663,
        128_004_290,
        144_004_957,
        160_005_626,
        176_006_319,
        192_007_016,
        208_007_727,
        224_008_456,
        240_009_215,
        256_009_984,
        272_010_765,
        288_011_564,
        304_012_377,
        320_013_196,
        336_014_037,
        352_014_914,
        368_015_793,
    ],
];

// ---------------------------------------------------------------------------------------------------- fp8 / e8m0 / bf16

/// `F8_E4M3FN` (the safetensors `F8_E4M3`): bias 7, no infinities, `0x7F`/`0xFF` are NaN, largest finite 448.
pub fn e4m3_to_f32(b: u8) -> f32 {
    gb_cuda::fp8_linear::e4m3_to_f32(b)
}

/// `F8_E8M0`: an unsigned exponent, `2^(e-127)`; `0xFF` is NaN and `0` is `2^-127` (an f32 subnormal, exact).
pub fn e8m0_to_f32(b: u8) -> f32 {
    match b {
        0xFF => f32::NAN,
        // 2^-127 is below f32's smallest normal (2^-126): the subnormal encoding has one bit at 2^22.
        0 => f32::from_bits(1 << 22),
        // For e in 1..=254 the exponent field of 2^(e-127) is exactly e.
        _ => f32::from_bits(u32::from(b) << 23),
    }
}

/// bf16 -> f32 is a shift: bf16 keeps f32's exponent and top seven mantissa bits.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// f32 -> bf16 bits, round to nearest even.
///
/// The reference dequantises in f32 and then casts the row to bf16 (`values.flatten(-2).to(torch.bfloat16)`), so every
/// gathered value is rounded once before `wkv` sees it; reproducing that rounding is part of reproducing the layer.
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    if x.is_nan() {
        // any quiet NaN payload; keep the sign and the top mantissa bits so the round trip is stable
        return ((x.to_bits() >> 16) as u16) | 0x0040;
    }
    let bits = x.to_bits();
    // Round the 16 discarded mantissa bits to nearest, ties to even: adding 0x7FFF + the LSB does it in one step.
    let round = 0x0000_7FFF + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(round);
    (rounded >> 16) as u16
}

/// The same value, held as an f32: the exact arithmetic the reference performs at the cast.
pub fn round_bf16(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16_bits(x))
}

const fn compute_e4m3_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    let mut b = 0usize;
    while b < 256 {
        let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = ((b >> 3) & 0x0F) as i32;
        let man = (b & 0x07) as f32;
        let v = if exp == 0 {
            (man / 8.0f32) * f32::from_bits((127 - 6) << 23)
        } else if exp == 0x0F && (b & 0x07) == 7 {
            f32::NAN
        } else {
            (1.0f32 + man / 8.0f32) * f32::from_bits(((exp - 7 + 127) as u32) << 23)
        };
        lut[b] = if v.is_nan() { f32::NAN } else { sign * v };
        b += 1;
    }
    lut
}

const E4M3_LUT: [f32; 256] = compute_e4m3_lut();

#[inline]
fn dequant_row_256_to_bf16(e: &[u8], k: &[u8], dst: &mut [u16; 256]) {
    for tile in 0..8 {
        let scale = e8m0_to_f32(k[tile]);
        let eb = &e[tile * 32..tile * 32 + 32];
        let db = &mut dst[tile * 32..tile * 32 + 32];
        for j in 0..32 {
            db[j] = f32_to_bf16_bits(E4M3_LUT[eb[j] as usize] * scale);
        }
    }
}

// ---------------------------------------------------------------------------------------------------- mapping

/// A read-only shared mapping of one regular file, or bytes this module quantised itself.
struct Mapping {
    ptr: *const u8,
    len: usize,
    /// `Some` when the bytes are owned (the boot-time EXL3->fp8 conversion), in which case `drop`
    /// frees the `Vec` and never unmaps.
    owned: Option<Vec<u8>>,
}

// SAFETY: the mapping is read-only for its whole life and is released exactly once, in `drop`.
unsafe impl Send for Mapping {}
// SAFETY: shared read-only bytes are safe to read from any thread.
unsafe impl Sync for Mapping {}

impl Mapping {
    /// Map `len` bytes of `path` starting at absolute file offset `offset`. The offset is page-aligned down and the
    /// mapping shifted back up, so a tensor that lives inside a shard at an arbitrary offset maps the same way a
    /// standalone file does. The whole range must lie inside the file — a short shard is an error, never a truncated map.
    fn open(path: &Path, offset: u64, len: usize) -> Result<Self, String> {
        use std::os::fd::AsRawFd;
        const PAGE: u64 = 4096;
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file_len = file.metadata().map_err(|e| format!("{}: {e}", path.display()))?.len();
        if len == 0 {
            return Err(format!("{}: a zero-length mapping", path.display()));
        }
        if offset.saturating_add(len as u64) > file_len {
            return Err(format!("{}: {len} bytes at {offset}, but the file is {file_len} long", path.display()));
        }
        let delta = (offset % PAGE) as usize;
        let map_off = (offset - delta as u64) as libc::off_t;
        let map_len = len + delta;
        // SAFETY: a read-only shared mapping of an open regular file, of a length inside its stated size; MAP_FAILED is checked below.
        let ptr =
            unsafe { libc::mmap(std::ptr::null_mut(), map_len, libc::PROT_READ, libc::MAP_SHARED, file.as_raw_fd(), map_off) };
        if ptr == libc::MAP_FAILED {
            return Err(format!("{}: mmap of {map_len} bytes at {map_off} failed", path.display()));
        }
        // The access pattern is one 264-byte row out of 384 million, at random: read-ahead is pure waste here.
        // SAFETY: ptr is a valid map_len-byte mapping.
        unsafe {
            libc::madvise(ptr, map_len, libc::MADV_RANDOM);
            libc::madvise(ptr, map_len, libc::MADV_DONTNEED);
        }
        // SAFETY: the mapping covers [map_off, map_off + map_len) and the tensor starts delta bytes in.
        let base = ptr.cast::<u8>();
        Ok(Self { ptr: unsafe { base.add(delta) }, len, owned: None })
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: ptr and len describe this live mapping, which outlives the borrow.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Own the bytes outright: the boot-time EXL3->fp8 conversion's result, never a file mapping.
    fn owned_bytes(bytes: Vec<u8>) -> Self {
        let ptr = bytes.as_ptr();
        let len = bytes.len();
        Self { ptr, len, owned: Some(bytes) }
    }

    /// The mapping's base address — what an H2D copy would be handed, never a copy of the table.
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        if self.owned.is_some() {
            return; // the Vec frees itself; there was never a mapping
        }
        // SAFETY: ptr and len came from one mmap and are unmapped exactly once, here.
        unsafe {
            libc::madvise(self.ptr.cast_mut().cast(), self.len, libc::MADV_DONTNEED);
            libc::munmap(self.ptr.cast_mut().cast(), self.len);
        }
    }
}

// ---------------------------------------------------------------------------------------------------- the table

/// One Engram embedding table, read through two `mmap`s: the FP8 weights `[rows, head_dim]` and their `F8_E8M0` block
/// scales `[rows, head_dim / 32]`.
///
/// `open` validates both byte lengths against the row count the *config* states (`engram_num_embeddings[layer]`) rather
/// than trusting the file, and sets `MADV_RANDOM` on both: a gather touches 24 rows spread uniformly over 91.5 GiB.
///
/// There is deliberately no `Vec<u8>` of the table, no `read_to_end`, and no `Vec` sized by `rows` anywhere in this type.
pub struct MmapTable {
    embed: Mapping,
    scale: Mapping,
    rows: usize,
    cols: usize,
}

impl MmapTable {
    /// Bytes one gathered row occupies on disk: 256 fp8 + 8 scale.
    pub const ROW_BYTES: usize = 256 + 8;
    /// Rows one token gathers: `ORDERS * N_HEADS = 24`.
    pub const ROWS_PER_TOKEN: usize = HASH_COLS;
    /// Bytes one token's gather touches (scales included), before dequantisation.
    pub const GATHER_BYTES_PER_TOKEN: usize = Self::ROWS_PER_TOKEN * Self::ROW_BYTES;

    /// Open the two shards of one table and validate them against the descriptor `bind` produced.
    ///
    /// `desc` is [`super::weights::MmapTable`]: its `rows` is the config's `engram_num_embeddings` entry and its two
    /// `Tensor`s carry the names and dtypes the header must have. The byte lengths are checked against `rows * head_dim`
    /// and `rows * head_dim / 32` — a short or long shard is an error naming both numbers, never a silently truncated map.
    pub fn open(embed: &Loc, scale: &Loc, desc: &super::weights::MmapTable) -> Result<Self, String> {
        use super::weights::Dtype;
        let cols = match desc.embed.shape.as_slice() {
            [_, c] => *c,
            s => return Err(format!("{}: expected a 2-D table, got {s:?}", desc.embed.name)),
        };
        if desc.rows == 0 || cols == 0 {
            return Err(format!("{}: a table with no rows or no columns", desc.embed.name));
        }
        if cols % FP8_TILE != 0 {
            return Err(format!("{}: {cols} columns is not a whole number of {FP8_TILE}-column scales", desc.embed.name));
        }
        if desc.embed.dtype != Dtype::Fp8 || desc.scale.dtype != Dtype::E8m0 {
            return Err(format!(
                "{}: expected F8_E4M3 weights and F8_E8M0 scales, got {:?} and {:?}",
                desc.embed.name, desc.embed.dtype, desc.scale.dtype
            ));
        }
        let want_scale = [desc.rows, cols / FP8_TILE];
        if desc.scale.shape != want_scale {
            return Err(format!("{}: scales are {:?}, expected {want_scale:?}", desc.scale.name, desc.scale.shape));
        }
        let want_embed = desc.rows.checked_mul(cols).ok_or("table size overflow")?;
        let want_scale = desc.rows.checked_mul(cols / FP8_TILE).ok_or("table size overflow")?;
        let embed_map = Mapping::open(&embed.path, embed.offset, want_embed)?;
        let scale_map = Mapping::open(&scale.path, scale.offset, want_scale)?;
        Ok(Self { embed: embed_map, scale: scale_map, rows: desc.rows, cols })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The mapping's base address, for a caller that wants to hand the table to a device without a copy.
    pub fn base_ptr(&self) -> (*const u8, *const u8) {
        (self.embed.as_ptr(), self.scale.as_ptr())
    }

    /// The raw bytes of one row: `(fp8 weights, e8m0 scales)`. No copy, no allocation, no dequantisation.
    pub fn row(&self, idx: usize) -> (&[u8], &[u8]) {
        let s = self.cols / FP8_TILE;
        let e = &self.embed.bytes()[idx * self.cols..(idx + 1) * self.cols];
        let k = &self.scale.bytes()[idx * s..(idx + 1) * s];
        (e, k)
    }

    /// One row dequantised into `out` as the reference produces it: `bf16(scale[j/32] * e4m3(weight[j]))`.
    ///
    /// The multiply is f32 and the rounding to bf16 happens once, after it — that is `values.float() *
    /// scales.float().unsqueeze(-1)` followed by `.to(torch.bfloat16)`.
    pub fn dequantised_row_f32(&self, idx: usize, out: &mut [f32]) -> Result<(), String> {
        if out.len() != self.cols {
            return Err(format!("row buffer is {} wide, the table is {}", out.len(), self.cols));
        }
        if idx >= self.rows {
            return Err(format!("row {idx} is outside a table of {}", self.rows));
        }
        let (e, k) = self.row(idx);
        for (j, (o, &b)) in out.iter_mut().zip(e).enumerate() {
            *o = round_bf16(e4m3_to_f32(b) * e8m0_to_f32(k[j / FP8_TILE]));
        }
        Ok(())
    }

    /// Dequantise a whole gather, row after row, into `out` — `ids.len() * cols` f32, in gather order.
    ///
    /// This is the exact gather `Engram.forward` performs: 24 rows of 256, each dequantised and rounded to bf16, flattened
    /// to `[24 * 256]` and fed to `wkv`.
    pub fn gather_dequant(&self, ids: &[usize], out: &mut [f32]) -> Result<(), String> {
        if out.len() != ids.len() * self.cols {
            return Err(format!(
                "gather buffer is {} wide, a {} row gather of {} columns is {}",
                out.len(),
                ids.len(),
                self.cols,
                ids.len() * self.cols
            ));
        }
        for (i, &id) in ids.iter().enumerate() {
            let at = i * self.cols;
            self.dequantised_row_f32(id, &mut out[at..at + self.cols])?;
        }
        Ok(())
    }

    /// Copy `ids.len()` rows straight out of the mapping into a caller-owned pair of buffers — the fp8 bytes and the scale
    /// bytes, in gather order. This is the only gather a device path needs: the destination is what gets uploaded.
    pub fn gather_raw(&self, ids: &[usize], weights: &mut [u8], scales: &mut [u8]) -> Result<(), String> {
        let s = self.cols / FP8_TILE;
        if weights.len() != ids.len() * self.cols || scales.len() != ids.len() * s {
            return Err(format!(
                "staging buffers are {} and {} bytes, a {} row gather is {} and {}",
                weights.len(),
                scales.len(),
                ids.len(),
                ids.len() * self.cols,
                ids.len() * s
            ));
        }
        for (i, &id) in ids.iter().enumerate() {
            if id >= self.rows {
                return Err(format!("row {id} is outside a table of {}", self.rows));
            }
            let (e, k) = self.row(id);
            weights[i * self.cols..(i + 1) * self.cols].copy_from_slice(e);
            scales[i * s..(i + 1) * s].copy_from_slice(k);
        }
        Ok(())
    }
}

impl std::fmt::Debug for MmapTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // the mappings themselves are addresses; what a reader wants to see is the geometry
        f.debug_struct("MmapTable").field("rows", &self.rows).field("cols", &self.cols).finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------------------------------- the token map

/// Why a [`TokenMap`] was rejected. Every variant names the number that was wrong.
#[derive(Debug)]
pub enum TokenMapError {
    /// The file could not be read.
    Io { path: PathBuf, source: std::io::Error },
    /// [`MAP_ENV`] is unset and no path was given.
    NoPath,
    /// The file is not exactly [`TOKEN_MAP_LEN`] little-endian `u32`.
    Length { path: PathBuf, got: usize, want: usize },
    /// The table does not cover exactly [`COMPRESSED_VOCAB`] distinct ids.
    Vocab { got: usize, want: usize },
    /// An id outside the compressed vocabulary: `compressed[id]` would index past the multipliers' safe range.
    IdOutOfRange { token: usize, id: u32, vocab: usize },
    /// A token id outside the map.
    TokenOutOfRange { token: u32, len: usize },
}

impl std::fmt::Display for TokenMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "engram token map {}: {source}", path.display()),
            Self::NoPath => write!(f, "engram token map: {MAP_ENV} is unset and no path was given"),
            Self::Length { path, got, want } => {
                write!(f, "engram token map {}: {got} bytes, expected {want} ({TOKEN_MAP_LEN} u32 little-endian)", path.display())
            }
            Self::Vocab { got, want } => {
                write!(f, "engram token map: {got} distinct ids, expected exactly {want}")
            }
            Self::IdOutOfRange { token, id, vocab } => {
                write!(f, "engram token map: entry {token} is {id}, outside the {vocab}-id compressed vocabulary")
            }
            Self::TokenOutOfRange { token, len } => {
                write!(f, "engram token map: token {token} is outside its {len} entries")
            }
        }
    }
}

impl std::error::Error for TokenMapError {}

/// The compressed token map: `token_id -> compressed_id`, `129_280` entries over `99_092` ids.
///
/// # File format
///
/// A flat array of [`TOKEN_MAP_LEN`] little-endian `u32`, in tokenizer-id order — entry `i` is the compressed id of token
/// `i`. 517,120 bytes, no header, no padding. It is the `lookup` list of the reference's `build_compressed_token_map`:
/// every token is decoded, normalised (`NFKC -> NFD -> StripAccents -> Lowercase`, whitespace runs collapsed to one space,
/// `Strip` with a `U+E000` sentinel so a lone space survives), and tokens that normalise alike share one id; a token whose
/// normalised text is empty keys by its raw form, and a partial-UTF-8 token (any `U+FFFD`) keys by its raw form too. Ids
/// are handed out in first-appearance order, so the table is not sorted and the mapping is not derivable from the config.
///
/// Both numbers are checked on load, because both are load-bearing: [`TOKEN_MAP_LEN`] is the tokenizer's vocabulary and
/// [`COMPRESSED_VOCAB`] is what [`MULTIPLIER_BOUND`] was computed from, so a tokenizer version skew would rehash every row
/// of 91.5 GiB of table with no other symptom.
pub struct TokenMap {
    ids: Vec<u32>,
}

impl TokenMap {
    /// Read and validate a map file.
    pub fn load(path: &Path) -> Result<Self, TokenMapError> {
        let bytes = std::fs::read(path).map_err(|source| TokenMapError::Io { path: path.to_path_buf(), source })?;
        Self::from_bytes_at(&bytes, path)
    }

    /// Validate a raw little-endian u32 byte slice.
    pub fn from_bytes_at(bytes: &[u8], path: &Path) -> Result<Self, TokenMapError> {
        let want = TOKEN_MAP_LEN * 4;
        if bytes.len() != want {
            return Err(TokenMapError::Length { path: path.to_path_buf(), got: bytes.len(), want });
        }
        let ids = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Self::from_ids(ids)
    }

    /// Read the map from [`MAP_ENV`].
    pub fn from_env() -> Result<Self, TokenMapError> {
        let path = std::env::var_os(MAP_ENV).ok_or(TokenMapError::NoPath)?;
        Self::load(Path::new(&path))
    }

    /// [`Self::from_env`], falling back to `<model_dir>/engram_token_map.u32` and then the in-binary bundled
    /// `testdata/deepseek_v41/engram_token_map.u32` when `REXL3_DSV41_ENGRAM_TOKEN_MAP` is unset.
    pub fn from_env_or_dir(model_dir: &Path) -> Result<Self, TokenMapError> {
        if std::env::var_os(MAP_ENV).filter(|s| !s.is_empty()).is_some() {
            return Self::from_env();
        }
        let fallback = model_dir.join("engram_token_map.u32");
        if fallback.exists() {
            return Self::load(&fallback);
        }
        const BUNDLED_BYTES: &[u8] = include_bytes!("../../../../../testdata/deepseek_v41/engram_token_map.u32");
        Self::from_bytes_at(BUNDLED_BYTES, Path::new("<bundled:testdata/deepseek_v41/engram_token_map.u32>"))
    }

    /// Validate a map already in memory. The two invariants: [`TOKEN_MAP_LEN`] entries, [`COMPRESSED_VOCAB`] distinct ids.
    pub fn from_ids(ids: Vec<u32>) -> Result<Self, TokenMapError> {
        if ids.len() != TOKEN_MAP_LEN {
            return Err(TokenMapError::Length { path: PathBuf::from("<memory>"), got: ids.len() * 4, want: TOKEN_MAP_LEN * 4 });
        }
        // 99,092 ids fit a 2^17-bit set; a bitmap keeps this allocation independent of the table sizes.
        let mut seen = vec![0u64; COMPRESSED_VOCAB.div_ceil(64)];
        let mut distinct = 0usize;
        for (token, &id) in ids.iter().enumerate() {
            let at = id as usize;
            if at >= COMPRESSED_VOCAB {
                return Err(TokenMapError::IdOutOfRange { token, id, vocab: COMPRESSED_VOCAB });
            }
            let bit = 1u64 << (at % 64);
            if seen[at / 64] & bit == 0 {
                seen[at / 64] |= bit;
                distinct += 1;
            }
        }
        if distinct != COMPRESSED_VOCAB {
            return Err(TokenMapError::Vocab { got: distinct, want: COMPRESSED_VOCAB });
        }
        Ok(Self { ids })
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The compressed id of a token.
    pub fn get(&self, token: u32) -> Result<u32, TokenMapError> {
        self.ids.get(token as usize).copied().ok_or(TokenMapError::TokenOutOfRange { token, len: self.ids.len() })
    }

    /// The `pad_id` the hash fills a blocked n-gram slot with: the *compressed* form of `engram_pad_token_id`.
    pub fn pad_id(&self, pad_token_id: u32) -> Result<i64, TokenMapError> {
        self.get(pad_token_id).map(i64::from)
    }
}

impl std::fmt::Debug for TokenMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenMap").field("len", &self.ids.len()).field("vocab", &COMPRESSED_VOCAB).finish()
    }
}

// ---------------------------------------------------------------------------------------------------- the hash

/// The n-gram hash: for each of the 24 `(order, head)` columns, which row of the layer's table a position reads.
///
/// # The function
///
/// ```text
/// c_s      = compressed[token_{p-s}]                  or pad_id, once blocked
/// rolling  = c_0 * MULT[0] ^ c_1 * MULT[1] ^ c_2 * MULT[2] ^ c_3 * MULT[3]     (i64, wrapping)
/// row(i,j) = OFFSET[(i-1)*8 + j] + rolling rem_euclid PRIME[i-1][j]            i = 1..=3, j = 0..8
/// ```
///
/// `rem_euclid` and not `%`: the reference's `rolling % primes` is torch's floored remainder, which is non-negative for a
/// negative `rolling`, while Rust's `%` would return a negative residue and index *before* the bucket range. Both sides
/// are `i64`, and the XOR is on the two's-complement bit pattern, which is the same operation either way.
///
/// # Blocking
///
/// A look-back that runs off the start of the sequence, or that lands on [`DEAD`], fills that slot with `pad_id` and
/// **every longer order** with it too — `blocked` is cumulative over increasing shift, so an n-gram never reaches past a
/// dead token and never straddles one. The window passed in is the last up-to-[`MAX_NGRAM`] compressed ids, oldest first;
/// a window shorter than four *is* the start of the sequence.
#[derive(Clone, Debug)]
pub struct NgramHash {
    layer_ids: Vec<usize>,
    pad: i64,
}

impl NgramHash {
    /// Build the hash for the layers the config names.
    ///
    /// The tables are fixed, so this checks the config *against* them rather than deriving anything: the layer list must be
    /// a prefix-ordered subsequence of `[1, 14]`, and the three shape fields must be the ones the constants were drawn for.
    /// A config that disagrees is a config for a different model, and silently hashing it would read garbage rows.
    pub fn new(cfg: &EngramConfig, map: &TokenMap) -> Result<Self, String> {
        if cfg.layer_ids.is_empty() {
            return Err("an engram hash needs at least one layer".into());
        }
        if cfg.layer_ids.len() > TABLE_ROWS.len() {
            return Err(format!("engram_layer_ids has {} entries, the tables cover {}", cfg.layer_ids.len(), TABLE_ROWS.len()));
        }
        for (i, &layer) in cfg.layer_ids.iter().enumerate() {
            let want = [1, 14][i];
            if layer != want {
                return Err(format!("engram_layer_ids is {:?}, expected the pack's {want} at index {i}", cfg.layer_ids));
            }
            if cfg.num_embeddings[i] != TABLE_ROWS[i] {
                return Err(format!(
                    "layer {layer} has {} rows, but the prime buckets tile exactly {}",
                    cfg.num_embeddings[i], TABLE_ROWS[i]
                ));
            }
        }
        if cfg.max_ngram_size != MAX_NGRAM {
            return Err(format!("engram_max_ngram_size is {}, the tables hash {MAX_NGRAM}-grams", cfg.max_ngram_size));
        }
        if cfg.n_heads != N_HEADS {
            return Err(format!("engram_n_heads is {}, the tables hash {N_HEADS} heads", cfg.n_heads));
        }
        if cfg.compressed_vocab_size != COMPRESSED_VOCAB || map.len() != TOKEN_MAP_LEN {
            return Err(format!(
                "engram_compressed_vocab_size is {} and the map has {} entries, expected {COMPRESSED_VOCAB} over {TOKEN_MAP_LEN}",
                cfg.compressed_vocab_size,
                map.len()
            ));
        }
        let pad = map.pad_id(cfg.pad_token_id).map_err(|e| e.to_string())?;
        Ok(Self { layer_ids: cfg.layer_ids.clone(), pad })
    }

    /// Which constant table a layer id uses.
    pub fn layer_slot(&self, layer: usize) -> Option<usize> {
        self.layer_ids.iter().position(|&l| l == layer)
    }

    pub fn columns(&self) -> usize {
        HASH_COLS
    }

    pub fn pad(&self) -> i64 {
        self.pad
    }

    /// The rolling hash before the per-head reduction — `window` oldest first, at most [`MAX_NGRAM`] long.
    ///
    /// Public so a test can pin the intermediate: the row ids are `offset + rolling % prime`, and a mismatch in the fold is
    /// much easier to see here than in 24 numbers.
    pub fn rolling(&self, layer_slot: usize, window: &[i64]) -> Result<i64, String> {
        let toks = self.window(layer_slot, window)?;
        let m = &MULTIPLIERS[layer_slot];
        let mut acc = toks[0].wrapping_mul(m[0]);
        for s in 1..MAX_NGRAM {
            acc ^= toks[s].wrapping_mul(m[s]);
        }
        Ok(acc)
    }

    /// The `(order, head)`-padded look-back window: `toks[s]` is the token `s` positions back, or `pad_id` once blocked.
    fn window(&self, layer_slot: usize, window: &[i64]) -> Result<[i64; MAX_NGRAM], String> {
        if layer_slot >= TABLE_ROWS.len() {
            return Err(format!("no engram table at index {layer_slot}"));
        }
        if window.is_empty() || window.len() > MAX_NGRAM {
            return Err(format!("an n-gram window is 1..={MAX_NGRAM} tokens, got {}", window.len()));
        }
        let mut toks = [self.pad; MAX_NGRAM];
        let mut blocked = false;
        for (s, slot) in toks.iter_mut().enumerate() {
            match window.len().checked_sub(1 + s) {
                // a shift that is inside the sequence: dead tokens block, and keep everything longer blocked
                Some(at) => {
                    let v = window[at];
                    if v == DEAD {
                        blocked = true;
                    }
                    if !blocked {
                        *slot = v;
                    }
                }
                // the shift runs off the start of the sequence
                None => blocked = true,
            }
        }
        Ok(toks)
    }

    /// The 24 row ids one position reads, order-major: columns `0..8` the 2-gram, `8..16` the 3-gram, `16..24` the 4-gram.
    pub fn rows_for(&self, layer_slot: usize, window: &[i64]) -> Result<[usize; HASH_COLS], String> {
        let toks = self.window(layer_slot, window)?;
        let m = &MULTIPLIERS[layer_slot];
        let mut rolling = toks[0].wrapping_mul(m[0]);
        let mut out = [0usize; HASH_COLS];
        for i in 1..MAX_NGRAM {
            rolling ^= toks[i].wrapping_mul(m[i]);
            for j in 0..N_HEADS {
                let col = (i - 1) * N_HEADS + j;
                let bucket = rolling.rem_euclid(PRIMES[layer_slot][i - 1][j]);
                out[col] = (OFFSETS[layer_slot][col] + bucket) as usize;
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------------- wkv

/// The engram's fused key/value projection: FP8 `[out, in]` with 32x32 block scales `[out/32, in/32]`.
///
/// `wkv` is 157 MB, which a serving box can hold — but it is read through an `mmap` for the same reason the tables are:
/// nothing in this family materialises a matrix it does not have to.
pub struct Fp8Matrix {
    weight: Mapping,
    scale: Mapping,
    out: usize,
    inp: usize,
}

impl Fp8Matrix {
    pub fn open(weight: &Loc, scale: &Loc, out: usize, inp: usize) -> Result<Self, String> {
        if out % FP8_TILE != 0 || inp % FP8_TILE != 0 {
            return Err(format!("a {out}x{inp} matrix is not a whole number of {FP8_TILE}x{FP8_TILE} tiles"));
        }
        let (want_w, want_s) = (out * inp, (out / FP8_TILE) * (inp / FP8_TILE));
        let w = Mapping::open(&weight.path, weight.offset, want_w)?;
        let s = Mapping::open(&scale.path, scale.offset, want_s)?;
        Ok(Self { weight: w, scale: s, out, inp })
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.out, self.inp)
    }

    /// `y = W x`, dequantising on the fly, rounded to bf16 once at the end (the reference's `nn.Linear` returns bf16).
    ///
    /// A host reference, not a serving path: the real `wkv` is 157 million multiply-adds per token, which is a kernel's
    /// job.
    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        if x.len() != self.inp {
            return Err(format!("wkv takes {} inputs, got {}", self.inp, x.len()));
        }
        let w = self.weight.bytes();
        let s = self.scale.bytes();
        let tiles = self.inp / FP8_TILE;
        let mut y = vec![0f32; self.out];
        for (o, yo) in y.iter_mut().enumerate() {
            let row = &w[o * self.inp..(o + 1) * self.inp];
            let srow = &s[(o / FP8_TILE) * tiles..(o / FP8_TILE + 1) * tiles];
            let mut acc = 0f32;
            for (t, &st) in srow.iter().enumerate() {
                let scale = e8m0_to_f32(st);
                let mut tile = 0f32;
                for i in 0..FP8_TILE {
                    let at = t * FP8_TILE + i;
                    tile += e4m3_to_f32(row[at]) * x[at];
                }
                acc += tile * scale;
            }
            *yo = round_bf16(acc);
        }
        Ok(y)
    }

    /// A matrix the pack stores as an EXL3 group (the quantized `wkv`): decoded once at boot through
    /// `gb_exl3::decode_weight` — the crate's own host decoder, the same path the MTP reference uses —
    /// transposed to the `[out, inp]` plain layout and quantised by [`quantize_e4m3_tiles`]. The same
    /// group yields the same bytes every boot (golden-pinned by test); the conversion's error is the
    /// e4m3 tile quantisation, bounded by test against the dense decode. The device path keeps
    /// reading the group natively; this is the host half's representation.
    pub fn from_group(g: &gb_exl3::checkpoint::LoadedGroup, out: usize, inp: usize) -> Result<Self, String> {
        let (gk, gn) = (g.group.k, g.group.n);
        if gk != inp || gn != out {
            return Err(format!("the EXL3 group is [k={gk}, n={gn}] but the engram wkv is [out={out}, inp={inp}]"));
        }
        if out % FP8_TILE != 0 || inp % FP8_TILE != 0 {
            return Err(format!("a {out}x{inp} matrix is not a whole number of {FP8_TILE}x{FP8_TILE} tiles"));
        }
        let dense =
            gb_exl3::decode_weight(&g.trellis, g.kt, g.nt, g.group.bits, &gb_exl3::codebook(g.group.codebook), &g.suh, &g.svh);
        // decode_weight returns W[i][j] with forward(x) = x @ W, i.e. [inp, out] row-major; the plain
        // layout matvec reads is [out, inp].
        let mut plain = vec![0f64; out * inp];
        for (i, row) in dense.chunks_exact(out).enumerate() {
            for (o, &v) in row.iter().enumerate() {
                plain[o * inp + i] = v;
            }
        }
        let (w, s) = quantize_e4m3_tiles(&plain, out, inp);
        let nw = Mapping::owned_bytes(w);
        let ns = Mapping::owned_bytes(s);
        Ok(Self { weight: nw, scale: ns, out, inp })
    }
}

/// Quantise a dense `[out, inp]` row-major f64 matrix into the fp8-e4m3 / e8m0 32x32-tile layout a
/// plain pack ships, deterministically: a tile's scale byte is `127 + exp` with
/// `exp = ceil(log2(amax / 448))` clamped to `[-127, 127]` (an all-zero tile takes `127`), each
/// value divides by `2^exp` and rounds to the nearest e4m3 code (`gb_cuda::fp8_linear::e4m3_from_f32`,
/// round-to-nearest-even, the encoder its own exhaustive test pins). The layout is exactly
/// [`Fp8Matrix::matvec`]'s: `weight[o * inp + i]` dequantised by `scale[(o/32) * (inp/32) + i/32]`.
fn quantize_e4m3_tiles(dense: &[f64], out: usize, inp: usize) -> (Vec<u8>, Vec<u8>) {
    let mut w = vec![0u8; out * inp];
    let mut s = vec![127u8; (out / FP8_TILE) * (inp / FP8_TILE)];
    let tiles_in = inp / FP8_TILE;
    for tob in 0..out / FP8_TILE {
        for tib in 0..tiles_in {
            let (o0, i0) = (tob * FP8_TILE, tib * FP8_TILE);
            let mut amax = 0f64;
            for o in 0..FP8_TILE {
                for i in 0..FP8_TILE {
                    let v = dense[(o0 + o) * inp + i0 + i].abs();
                    if v > amax {
                        amax = v;
                    }
                }
            }
            let exp = if amax == 0.0 { 0i32 } else { (amax / 448.0).log2().ceil().clamp(-127.0, 127.0) as i32 };
            s[tob * tiles_in + tib] = (127 + exp) as u8;
            let scale = 2f64.powi(exp);
            for o in 0..FP8_TILE {
                for i in 0..FP8_TILE {
                    let v = dense[(o0 + o) * inp + i0 + i] / scale;
                    w[(o0 + o) * inp + i0 + i] = gb_cuda::fp8_linear::e4m3_from_f32(v as f32);
                }
            }
        }
    }
    (w, s)
}

// ---------------------------------------------------------------------------------------------------- the gate

/// The gate's signed square root, then the sigmoid — `sigmoid(copysign(sqrt(|dot|.clamp_min(1e-6)), dot))`.
///
/// Split out because it is the one non-obvious piece of arithmetic in the layer: the magnitude is square-rooted and the
/// *sign* of the dot product is copied onto it, so a negative dot gives a gate in `(0, 0.5)` and a positive one in
/// `(0.5, 1)`. `copysign` rather than `signum()` so that `-0.0` lands on the same side the reference puts it.
pub fn signed_sqrt_gate(dot: f32) -> f32 {
    const CLAMP: f32 = 1e-6;
    let mag = dot.abs().max(CLAMP).sqrt();
    1.0 / (1.0 + (-mag.copysign(dot)).exp())
}

/// `x + gate * value` for every hyper-connection copy, plus the shared value row: the layer's whole contribution.
///
/// `x`, `key`, `q` and `k` are `[hc, dim]`; `value` is `[dim]`; `out` is `[(hc + 1), dim]` — the `kv` layout `wkv`
/// produces, `key` first and then the one shared `value`.
///
/// The normalisation is per `(token, hc copy)` over `dim`, **not** joint over the copies: `rstd` is the product of the two
/// independent reciprocal standard deviations, which is why each copy's gate is its own scalar.
// the eight slices are the eight tensors the layer's arithmetic reads; a struct here would only rename them
#[allow(clippy::too_many_arguments)]
pub fn gate_and_add(
    x: &[f32],
    key: &[f32],
    value: &[f32],
    q: &[f32],
    k: &[f32],
    hc: usize,
    dim: usize,
    eps: f32,
    out: &mut [f32],
) -> Result<(), String> {
    if x.len() != hc * dim || key.len() != hc * dim || q.len() != hc * dim || k.len() != hc * dim || value.len() != dim {
        return Err(format!(
            "gate shapes: x {}, key {}, q {}, k {}, value {} for {hc} copies of {dim}",
            x.len(),
            key.len(),
            q.len(),
            k.len(),
            value.len()
        ));
    }
    if out.len() != (hc + 1) * dim {
        return Err(format!("gate output is {} wide, expected {}", out.len(), (hc + 1) * dim));
    }
    let inv_sqrt_dim = (dim as f32).powf(-0.5);
    for c in 0..hc {
        let xr = &x[c * dim..(c + 1) * dim];
        let kr = &key[c * dim..(c + 1) * dim];
        let x_var = xr.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let k_var = kr.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let rstd = (x_var + eps).sqrt().recip() * (k_var + eps).sqrt().recip();
        let mut dot = 0f32;
        for d in 0..dim {
            // `q_weight * k_weight` is only ever used as a product, so it is formed here and never stored
            let w = q[c * dim + d] * k[c * dim + d];
            dot += xr[d] * w * kr[d];
        }
        let gate = signed_sqrt_gate(dot * rstd * inv_sqrt_dim);
        for d in 0..dim {
            out[c * dim + d] = xr[d] + gate * value[d];
        }
    }
    out[hc * dim..].copy_from_slice(value);
    Ok(())
}

// ---------------------------------------------------------------------------------------------------- state

/// One decoding slot's last [`RING`] *compressed* tokens, oldest-first, left-padded with `pad_id`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    pub ring: [i64; RING],
    pub filled: usize,
}

/// One slot's n-gram ring contents — what a tree round snapshots before its path windows run and replays
/// the accepted path over ([`EngramState::snapshot_slot`]/[`EngramState::restore_slot`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingSnap {
    ring: [i64; RING],
    filled: usize,
}

impl Slot {
    fn new(pad: i64) -> Self {
        Self { ring: [pad; RING], filled: 0 }
    }

    fn reset(&mut self, pad: i64) {
        self.ring = [pad; RING];
        self.filled = 0;
    }

    fn rewind(&mut self, drop_count: usize) {
        self.filled = self.filled.saturating_sub(drop_count);
    }

    fn push(&mut self, compressed: i64) {
        let at = self.filled % RING;
        self.ring[at] = compressed;
        self.filled += 1;
    }

    /// The last `n` compressed ids, oldest first. Before the sequence has `n` tokens the front is simply absent, which is
    /// exactly the "look-back clamps at the start" rule the hash implements.
    fn window(&self, n: usize) -> Vec<i64> {
        let have = self.filled.min(RING).min(n);
        (0..have).map(|i| self.ring[(self.filled + RING - have + i) % RING]).collect()
    }
}

/// Where one engram tensor's bytes live inside a pack: a shard file and the absolute byte offset of its data in it.
/// The scan resolves this; the lengths are the config's, never the file's.
#[derive(Clone, Debug)]
pub struct Loc {
    pub path: PathBuf,
    pub offset: u64,
}

/// Where the six tensors of one engram layer live inside a pack.
#[derive(Clone, Debug)]
pub struct EngramFiles {
    pub embed: Loc,
    pub scale: Loc,
    pub q_weight: Loc,
    /// the pack stores the n-gram projections bf16 (base layout) or f32 (the quantized packs)
    pub q_f32: bool,
    pub k_weight: Loc,
    pub k_f32: bool,
    /// the fused key/value projection: plain FP8 in the pack, or the EXL3 group the engine
    /// decodes at boot (whose `LoadedGroup` is passed to [`EngramState::open`] alongside)
    pub wkv: WkvFiles,
}

/// The wkv's two pack forms.
#[derive(Clone, Debug)]
pub enum WkvFiles {
    Plain { weight: Loc, scale: Loc },
    Group,
}

const CACHE_SLOTS: usize = 65_536;
const CACHE_WAYS: usize = 8;

#[inline]
fn cache_get(cache: &[(usize, [u16; 256])], idx: usize) -> Option<&[u16; 256]> {
    if cache.is_empty() {
        return None;
    }
    let mask = cache.len() - 1;
    let base = (idx.wrapping_mul(0x9E37_79B9_7F4A_7C15)) & mask;
    for w in 0..CACHE_WAYS {
        let slot = (base + w) & mask;
        let k = cache[slot].0;
        if k == idx {
            return Some(&cache[slot].1);
        }
        if k == usize::MAX {
            return None;
        }
    }
    None
}

#[inline]
fn cache_put(cache: &mut [(usize, [u16; 256])], idx: usize, val: [u16; 256]) {
    if cache.is_empty() {
        return;
    }
    let mask = cache.len() - 1;
    let base = (idx.wrapping_mul(0x9E37_79B9_7F4A_7C15)) & mask;
    for w in 0..CACHE_WAYS {
        let slot = (base + w) & mask;
        if cache[slot].0 == idx || cache[slot].0 == usize::MAX {
            cache[slot] = (idx, val);
            return;
        }
    }
    for w in (1..CACHE_WAYS).rev() {
        let dst = (base + w) & mask;
        let src = (base + w - 1) & mask;
        cache[dst] = cache[src];
    }
    cache[base & mask] = (idx, val);
}

/// The Engram branch of one layer: the table, the hash, the three projections, and the per-slot token rings.
///
/// Holds no `Vec` sized by the table: the table is two mappings, `q_weight`/`k_weight` are 4x5120 bf16 = 40 KiB each, and
/// the scratch a gather needs is `HASH_COLS * head_dim` `f32` — 24 KiB, allocated per call.
pub struct EngramState {
    table: MmapTable,
    hash: NgramHash,
    map: Arc<TokenMap>,
    wkv: Fp8Matrix,
    q_weight: Vec<f32>,
    k_weight: Vec<f32>,
    slots: Vec<Slot>,
    cache: std::sync::Mutex<Vec<(usize, [u16; 256])>>,
    hidden: usize,
    head_dim: usize,
    layer_slot: usize,
}

#[cfg(target_os = "linux")]
fn should_evict_engram_pages() -> bool {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static LOW_MEM: AtomicBool = AtomicBool::new(false);
    let c = CALLS.fetch_add(1, Ordering::Relaxed);
    if c.is_multiple_of(8) {
        let low = crate::hostmem::MemInfo::read().is_none_or(|m| m.available < (8u64 << 30));
        LOW_MEM.store(low, Ordering::Relaxed);
        return low;
    }
    LOW_MEM.load(Ordering::Relaxed)
}

#[cfg(target_os = "linux")]
#[inline]
fn madvise_row(e_base: usize, k_base: usize, idx: usize, advice: libc::c_int) {
    const PAGE: usize = 4096;
    let e_start = e_base + idx * 256;
    let e_page = e_start & !(PAGE - 1);
    let e_len = (((e_start + 256) + (PAGE - 1)) & !(PAGE - 1)) - e_page;
    let k_start = k_base + idx * 8;
    let k_page = k_start & !(PAGE - 1);
    let k_len = (((k_start + 8) + (PAGE - 1)) & !(PAGE - 1)) - k_page;
    unsafe {
        libc::madvise(e_page as *mut libc::c_void, e_len, advice);
        libc::madvise(k_page as *mut libc::c_void, k_len, advice);
    }
}

#[cfg(target_os = "linux")]
#[inline]
fn target_cache_slots() -> usize {
    if should_evict_engram_pages() {
        1 << 14
    } else {
        CACHE_SLOTS
    }
}

#[cfg(not(target_os = "linux"))]
#[inline]
fn target_cache_slots() -> usize {
    CACHE_SLOTS
}

impl EngramState {
    /// Open one engram layer: layer `layer` must be in `engram_layer_ids`, and its row count comes from the config.
    ///
    /// `hidden` is the trunk's residual width, which `EngramConfig` does not carry (it is `hidden_size` on the text config,
    /// and the engram's `q_weight`/`k_weight` are `[max_ngram_size, hidden]`). `map` is the compressed token map — see
    /// [`TokenMap`] for its file format.
    pub fn open(
        cfg: &EngramConfig,
        layer: usize,
        hidden: usize,
        files: &EngramFiles,
        slots: usize,
        wkv_group: Option<&gb_exl3::checkpoint::LoadedGroup>,
        map: Arc<TokenMap>,
    ) -> Result<Self, String> {
        let rows = cfg.table(layer).ok_or_else(|| format!("layer {layer} carries no engram table"))?;
        let d = cfg.head_dim;
        let desc = super::weights::MmapTable {
            layer,
            rows,
            embed: super::weights::Tensor::new(
                format!("layers.{layer}.engram.embed.weight"),
                super::weights::Dtype::Fp8,
                vec![rows, d],
            ),
            scale: super::weights::Tensor::new(
                format!("layers.{layer}.engram.embed.scale"),
                super::weights::Dtype::E8m0,
                vec![rows, d / FP8_TILE],
            ),
        };
        let table = MmapTable::open(&files.embed, &files.scale, &desc)?;
        let hash = NgramHash::new(cfg, &map)?;
        let layer_slot = hash.layer_slot(layer).ok_or_else(|| format!("layer {layer} is not in {:?}", cfg.layer_ids))?;
        let wkv = match (&files.wkv, wkv_group) {
            (WkvFiles::Plain { weight, scale }, _) => Fp8Matrix::open(
                weight,
                scale,
                super::weights::ENGRAM_WKV_ROW_FACTOR * hidden,
                super::weights::ENGRAM_WKV_COL_FACTOR * cfg.n_heads * d,
            )?,
            (WkvFiles::Group, Some(g)) => Fp8Matrix::from_group(
                g,
                super::weights::ENGRAM_WKV_ROW_FACTOR * hidden,
                super::weights::ENGRAM_WKV_COL_FACTOR * cfg.n_heads * d,
            )?,
            (WkvFiles::Group, None) => return Err("the wkv is the EXL3 group form but no loaded group was given".into()),
        };
        let q_weight = read_qk(&files.q_weight, cfg.max_ngram_size * hidden, files.q_f32)?;
        let k_weight = read_qk(&files.k_weight, cfg.max_ngram_size * hidden, files.k_f32)?;
        let pad = hash.pad();
        let _ = (d, rows);
        let init_cache = Vec::new();
        Ok(Self {
            table,
            hash,
            map,
            wkv,
            q_weight,
            k_weight,
            slots: (0..slots).map(|_| Slot::new(pad)).collect(),
            cache: std::sync::Mutex::new(init_cache),
            hidden,
            head_dim: d,
            layer_slot,
        })
    }

    pub fn table(&self) -> &MmapTable {
        &self.table
    }

    pub fn hash(&self) -> &NgramHash {
        &self.hash
    }

    /// Reset `slot`'s n-gram ring back to empty (`pad_id`).
    pub fn reset_slot(&mut self, slot: usize) {
        let pad = self.hash.pad();
        if let Some(s) = self.slots.get_mut(slot) {
            s.reset(pad);
        }
    }

    /// `slot`'s ring contents (the tree round's save/restore unit: the gather APPENDS per row, so sibling
    /// verify paths contaminate each other's n-gram context without the restore; T2.5a lane 2).
    pub fn snapshot_slot(&self, slot: usize) -> Option<RingSnap> {
        self.slots.get(slot).map(|s| RingSnap { ring: s.ring, filled: s.filled })
    }

    /// [`Self::snapshot_slot`]'s inverse.
    pub fn restore_slot(&mut self, slot: usize, snap: &RingSnap) {
        if let Some(s) = self.slots.get_mut(slot) {
            s.ring = snap.ring;
            s.filled = snap.filled;
        }
    }

    /// Rewind `slot`'s n-gram ring by `drop_count` rejected draft tokens.
    pub fn rewind(&mut self, slot: usize, drop_count: usize) {
        if let Some(s) = self.slots.get_mut(slot) {
            s.rewind(drop_count);
        }
    }

    /// Snapshot `slot`'s n-gram ring state.
    pub fn slot_snapshot(&self, slot: usize) -> Option<Slot> {
        self.slots.get(slot).cloned()
    }

    /// Restore `slot`'s n-gram ring state from `snap`.
    pub fn slot_restore(&mut self, slot: usize, snap: &Slot) {
        if let Some(s) = self.slots.get_mut(slot) {
            *s = snap.clone();
        }
    }

    /// The 24 rows one slot's current position reads.
    pub fn position_rows(&self, slot: usize) -> Result<[usize; HASH_COLS], String> {
        let s = self.slots.get(slot).ok_or_else(|| format!("slot {slot} is outside {}", self.slots.len()))?;
        self.hash.rows_for(self.layer_slot, &s.window(MAX_NGRAM))
    }

    /// Append tokens to a slot's ring, compressing them on the way in. Rejected draft positions are overwritten by the next
    /// store at the same indices.
    pub fn store(&mut self, slot: usize, tokens: &[u32]) -> Result<(), String> {
        self.store_masked(slot, tokens, None)
    }

    /// Issue `MADV_WILLNEED` for uncached table rows of each token in `tokens` without mutating `slot`'s ring.
    pub fn prefetch_batch(&self, slot: usize, tokens: &[i32]) {
        #[cfg(target_os = "linux")]
        {
            if tokens.len() > 16 || should_evict_engram_pages() {
                return;
            }
            let Some(mut sim) = self.slots.get(slot).cloned() else { return };
            const PAGE: usize = 4096;
            let s_cols = self.table.cols() / FP8_TILE;
            let e_base = self.table.base_ptr().0 as usize;
            let k_base = self.table.base_ptr().1 as usize;
            let max_rows = self.table.rows();
            let cache = self.cache.lock().ok();
            for &t in tokens {
                let Ok(cid) = self.map.get(t as u32) else { continue };
                sim.push(i64::from(cid));
                let Ok(rows) = self.hash.rows_for(self.layer_slot, &sim.window(MAX_NGRAM)) else { continue };
                for &idx in &rows {
                    if idx < max_rows {
                        if let Some(ref c) = cache {
                            if cache_get(c, idx).is_some() {
                                continue;
                            }
                        }
                        let e_page = (e_base + idx * self.table.cols()) & !(PAGE - 1);
                        let k_page = (k_base + idx * s_cols) & !(PAGE - 1);
                        unsafe {
                            libc::madvise(e_page as *mut libc::c_void, PAGE, libc::MADV_WILLNEED);
                            libc::madvise(k_page as *mut libc::c_void, PAGE, libc::MADV_WILLNEED);
                        }
                    }
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = (slot, tokens);
    }

    /// Single-threaded madvise + read + dequantise of all uncached rows for `tokens` into `self.cache` without mutating `slot`'s ring.
    pub fn warm_batch(&self, slot: usize, tokens: &[i32]) {
        if self.table.cols() != 256 {
            return;
        }
        let Some(mut sim) = self.slots.get(slot).cloned() else { return };
        let max_rows = self.table.rows();
        let mut misses = Vec::with_capacity(tokens.len() * HASH_COLS);
        {
            let Ok(mut cache) = self.cache.lock() else { return };
            if cache.is_empty() && max_rows >= CACHE_SLOTS {
                *cache = vec![(usize::MAX, [0u16; 256]); target_cache_slots()];
            }
            if cache.is_empty() {
                return;
            }
            for &t in tokens {
                let Ok(cid) = self.map.get(t as u32) else { continue };
                sim.push(i64::from(cid));
                let Ok(rows) = self.hash.rows_for(self.layer_slot, &sim.window(MAX_NGRAM)) else { continue };
                for &idx in &rows {
                    if idx < max_rows && cache_get(&cache, idx).is_none() {
                        misses.push(idx);
                    }
                }
            }
        }
        if misses.is_empty() {
            return;
        }
        misses.sort_unstable();
        misses.dedup();
        let table = &self.table;
        #[cfg(target_os = "linux")]
        {
            let e_base = table.base_ptr().0 as usize;
            let k_base = table.base_ptr().1 as usize;
            for &idx in &misses {
                madvise_row(e_base, k_base, idx, libc::MADV_WILLNEED);
            }
        }
        let mut decoded = vec![[0u16; 256]; misses.len()];
        for (&idx, dst) in misses.iter().zip(decoded.iter_mut()) {
            let (e, k) = table.row(idx);
            dequant_row_256_to_bf16(e, k, dst);
        }
        #[cfg(target_os = "linux")]
        if should_evict_engram_pages() {
            let e_base = table.base_ptr().0 as usize;
            let k_base = table.base_ptr().1 as usize;
            for &idx in &misses {
                madvise_row(e_base, k_base, idx, libc::MADV_DONTNEED);
            }
        }
        if let Ok(mut cache) = self.cache.lock() {
            if !cache.is_empty() {
                for (idx, buf) in misses.into_iter().zip(decoded.into_iter()) {
                    cache_put(&mut cache, idx, buf);
                }
            }
        }
    }

    /// Append `tokens` to `slot`'s ring and gather + dequantise the 24 rows of the current position as BF16 bits (`6,144` `u16`s).
    pub fn gather_row_bf16(&mut self, slot: usize, tokens: &[u32], out: &mut Vec<u16>) -> Result<(), String> {
        self.store_masked(slot, tokens, None)?;
        let rows = self.position_rows(slot)?;
        let max_rows = self.table.rows();
        let cols = self.table.cols();
        if cols == 256 {
            #[cfg(target_os = "linux")]
            let evict_pages = should_evict_engram_pages();
            let mut cache = self.cache.lock().unwrap();
            if cache.is_empty() && max_rows >= CACHE_SLOTS {
                *cache = vec![(usize::MAX, [0u16; 256]); target_cache_slots()];
            }
            let use_cache = !cache.is_empty();
            for &idx in &rows {
                if idx >= max_rows {
                    return Err(format!("row {idx} is outside a table of {max_rows}"));
                }
                if use_cache {
                    if let Some(hit) = cache_get(&cache, idx) {
                        out.extend_from_slice(hit);
                        continue;
                    }
                    let (e, k) = self.table.row(idx);
                    let mut buf = [0u16; 256];
                    dequant_row_256_to_bf16(e, k, &mut buf);
                    #[cfg(target_os = "linux")]
                    if evict_pages {
                        madvise_row(self.table.base_ptr().0 as usize, self.table.base_ptr().1 as usize, idx, libc::MADV_DONTNEED);
                    }
                    cache_put(&mut cache, idx, buf);
                    out.extend_from_slice(&buf);
                } else {
                    let (e, k) = self.table.row(idx);
                    let mut buf = [0u16; 256];
                    dequant_row_256_to_bf16(e, k, &mut buf);
                    out.extend_from_slice(&buf);
                }
            }
            return Ok(());
        }
        for &idx in &rows {
            if idx >= max_rows {
                return Err(format!("row {idx} is outside a table of {max_rows}"));
            }
            let (e, k) = self.table.row(idx);
            for (j, &b) in e.iter().enumerate() {
                out.push(f32_to_bf16_bits(E4M3_LUT[b as usize] * e8m0_to_f32(k[j / FP8_TILE])));
            }
        }
        Ok(())
    }

    /// Batch gather + dequantise for `tokens.len()` consecutive tokens.
    pub fn gather_batch_bf16(&mut self, slot: usize, tokens: &[i32], out: &mut Vec<u16>) -> Result<(), String> {
        let n_tok = tokens.len();
        let cols = self.table.cols();
        if cols != 256 {
            for &t in tokens {
                let tok = [t as u32];
                self.gather_row_bf16(slot, &tok, out)?;
            }
            return Ok(());
        }
        let max_rows = self.table.rows();
        let mut all_rows = Vec::with_capacity(n_tok * HASH_COLS);
        for &t in tokens {
            let tok = [t as u32];
            self.store_masked(slot, &tok, None)?;
            let r = self.position_rows(slot)?;
            for &idx in &r {
                if idx >= max_rows {
                    return Err(format!("row {idx} is outside a table of {max_rows}"));
                }
                all_rows.push(idx);
            }
        }
        let total_rows = all_rows.len();
        let base_off = out.len();
        out.resize(base_off + total_rows * 256, 0u16);
        let mut misses = Vec::new();
        let use_cache;
        {
            let mut cache = self.cache.lock().unwrap();
            if cache.is_empty() && cols == 256 && max_rows >= CACHE_SLOTS {
                *cache = vec![(usize::MAX, [0u16; 256]); target_cache_slots()];
            }
            use_cache = !cache.is_empty();
            for (pos, &idx) in all_rows.iter().enumerate() {
                if use_cache {
                    if let Some(hit) = cache_get(&cache, idx) {
                        out[base_off + pos * 256..base_off + (pos + 1) * 256].copy_from_slice(hit);
                        continue;
                    }
                }
                misses.push((pos, idx));
            }
        }
        if misses.is_empty() {
            return Ok(());
        }
        let table = &self.table;
        if misses.len() >= 8 {
            let mut decoded = vec![[0u16; 256]; misses.len()];
            let n_workers = 8usize;
            let chunk = misses.len().div_ceil(n_workers).max(1);
            std::thread::scope(|s| {
                for (m_chunk, d_chunk) in misses.chunks(chunk).zip(decoded.chunks_mut(chunk)) {
                    s.spawn(move || {
                        for (sub_m, sub_d) in m_chunk.chunks(64).zip(d_chunk.chunks_mut(64)) {
                            #[cfg(target_os = "linux")]
                            {
                                let e_base = table.base_ptr().0 as usize;
                                let k_base = table.base_ptr().1 as usize;
                                for &(_, idx) in sub_m {
                                    madvise_row(e_base, k_base, idx, libc::MADV_WILLNEED);
                                }
                            }
                            for (&(_, idx), dst) in sub_m.iter().zip(sub_d.iter_mut()) {
                                let (e, k) = table.row(idx);
                                dequant_row_256_to_bf16(e, k, dst);
                            }
                            #[cfg(target_os = "linux")]
                            if should_evict_engram_pages() {
                                let e_base = table.base_ptr().0 as usize;
                                let k_base = table.base_ptr().1 as usize;
                                for &(_, idx) in sub_m {
                                    madvise_row(e_base, k_base, idx, libc::MADV_DONTNEED);
                                }
                            }
                        }
                    });
                }
            });
            let mut cache = self.cache.lock().unwrap();
            for (&(pos, idx), buf) in misses.iter().zip(decoded.iter()) {
                out[base_off + pos * 256..base_off + (pos + 1) * 256].copy_from_slice(buf);
                if use_cache {
                    cache_put(&mut cache, idx, *buf);
                }
            }
            return Ok(());
        }
        let mut cache = self.cache.lock().unwrap();
        #[cfg(target_os = "linux")]
        let evict_pages = should_evict_engram_pages();
        #[cfg(target_os = "linux")]
        {
            let e_base = table.base_ptr().0 as usize;
            let k_base = table.base_ptr().1 as usize;
            for &(_, idx) in &misses {
                madvise_row(e_base, k_base, idx, libc::MADV_WILLNEED);
            }
        }
        for &(pos, idx) in &misses {
            if use_cache {
                if let Some(hit) = cache_get(&cache, idx) {
                    out[base_off + pos * 256..base_off + (pos + 1) * 256].copy_from_slice(hit);
                    continue;
                }
            }
            let (e, k) = table.row(idx);
            let mut buf = [0u16; 256];
            dequant_row_256_to_bf16(e, k, &mut buf);
            #[cfg(target_os = "linux")]
            if evict_pages {
                madvise_row(table.base_ptr().0 as usize, table.base_ptr().1 as usize, idx, libc::MADV_DONTNEED);
            }
            out[base_off + pos * 256..base_off + (pos + 1) * 256].copy_from_slice(&buf);
            if use_cache {
                cache_put(&mut cache, idx, buf);
            }
        }
        Ok(())
    }

    /// The same, with a `[tokens.len()]` mask: `false` stores [`DEAD`] (an image span), which blocks every n-gram it is
    /// part of.
    pub fn store_masked(&mut self, slot: usize, tokens: &[u32], alive: Option<&[bool]>) -> Result<(), String> {
        if let Some(m) = alive {
            if m.len() != tokens.len() {
                return Err(format!("token mask is {} long, {} tokens", m.len(), tokens.len()));
            }
        }
        let n = self.slots.len();
        let map = Arc::clone(&self.map);
        let s = self.slots.get_mut(slot).ok_or_else(|| format!("slot {slot} is outside {n}"))?;
        for (i, &t) in tokens.iter().enumerate() {
            // the image splice writes [`DEAD`] into the id array itself (model.py keeps image positions out of the
            // n-gram stream entirely), so a DEAD sentinel is stored as DEAD regardless of the mask
            let live = alive.is_none_or(|m| m[i]) && t != DEAD_AS_U32;
            s.push(if live { i64::from(map.get(t).map_err(|e| e.to_string())?) } else { DEAD });
        }
        Ok(())
    }

    /// The layer's additive contribution, `[(hc + 1) * hidden]` — the layout `wkv` produces.
    ///
    /// `x` is this position's residual stream, `[hc * hidden]` (the reference's `x[B, L, hc, dim]` with `B = L = 1`).
    /// The recipe, in the reference's own terms:
    ///
    /// ```text
    /// rows  = 24 hash ids for this position
    /// kv    = wkv(bf16(dequant(rows)).flatten())            [hc*dim, dim]
    /// key, value = kv.split([hc*dim, dim])
    /// rstd  = rsqrt(mean(x^2, -1) + eps) * rsqrt(mean(key^2, -1) + eps)      per (token, copy)
    /// dot   = sum(x * q * k * key, -1) * rstd * dim^-0.5
    /// gate  = sigmoid(copysign(sqrt(|dot|.clamp_min(1e-6)), dot))
    /// out   = x + gate * value
    /// ```
    pub fn forward(&mut self, slot: usize, tokens: &[u32], x: &[f32]) -> Result<Vec<f32>, String> {
        self.forward_masked(slot, tokens, None, x)
    }

    /// The same, with the reference's `token_mask`: a position whose mask is `false` has its gate shut, so it passes
    /// through the layer untouched (its `value` row is still written, ungated — `value` is not gated in the reference).
    pub fn forward_masked(&mut self, slot: usize, tokens: &[u32], alive: Option<&[bool]>, x: &[f32]) -> Result<Vec<f32>, String> {
        let hc = self.q_weight.len() / self.hidden;
        if x.len() != hc * self.hidden {
            return Err(format!("the residual stream is {} wide, expected {}", x.len(), hc * self.hidden));
        }
        self.store_masked(slot, tokens, alive)?;
        let rows = self.position_rows(slot)?;

        // 24 rows of head_dim, dequantised straight out of the mapping
        let mut gathered = vec![0f32; HASH_COLS * self.head_dim];
        self.table.gather_dequant(&rows, &mut gathered)?;
        let kv = self.wkv.matvec(&gathered)?;

        let d = self.hidden;
        let (key, value) = kv.split_at(hc * d);
        let mut out = vec![0f32; (hc + 1) * d];
        gate_and_add(x, key, value, &self.q_weight, &self.k_weight, hc, d, 1e-20, &mut out)?;
        // `gate.masked_fill(~token_mask, 0)`: a masked position keeps its residual stream and only writes the value row
        if let Some(false) = alive.and_then(|m| m.last()) {
            out[..hc * d].copy_from_slice(x);
        }
        Ok(out)
    }
}

/// Read a bf16 tensor of `n` elements as `f32`. 2 bytes per element, so 40 KiB for the pack's `q_weight`.
/// Read `n` elements at `loc` as bf16 (the base layout) or f32 (the quantized packs) into f32.
fn read_qk(loc: &Loc, n: usize, f32_bytes: bool) -> Result<Vec<f32>, String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(&loc.path).map_err(|e| format!("{}: {e}", loc.path.display()))?;
    f.seek(SeekFrom::Start(loc.offset)).map_err(|e| format!("{}: {e}", loc.path.display()))?;
    if f32_bytes {
        let mut bytes = vec![0u8; n * 4];
        f.read_exact(&mut bytes).map_err(|e| format!("{}: {e}", loc.path.display()))?;
        return Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect());
    }
    let mut bytes = vec![0u8; n * 2];
    f.read_exact(&mut bytes).map_err(|e| format!("{}: {e}", loc.path.display()))?;
    Ok(bytes.chunks_exact(2).map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect())
}

// ---------------------------------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A map that must validate: `129_280` entries, `99_092` distinct ids, and small token ids pass through unchanged.
    fn synthetic_map() -> TokenMap {
        let ids = (0..TOKEN_MAP_LEN).map(|i| (i % COMPRESSED_VOCAB) as u32).collect();
        TokenMap::from_ids(ids).unwrap()
    }

    fn small_cfg() -> EngramConfig {
        EngramConfig {
            layer_ids: vec![1, 14],
            num_embeddings: vec![384_006_168, 384_016_682],
            max_ngram_size: 4,
            vocab_size: 16_000_000,
            n_heads: 8,
            head_dim: 256,
            pad_token_id: 2,
            compressed_vocab_size: COMPRESSED_VOCAB,
        }
    }

    fn is_prime(n: i64) -> bool {
        if n < 2 {
            return false;
        }
        if n % 2 == 0 {
            return n == 2;
        }
        let mut d = 3;
        while d * d <= n {
            if n % d == 0 {
                return false;
            }
            d += 2;
        }
        true
    }

    // ---- the constant tables --------------------------------------------------------------------------------------

    #[test]
    fn the_primes_and_offsets_tile_both_tables_exactly() {
        for (slot, &rows) in TABLE_ROWS.iter().enumerate() {
            let mut end = 0i64;
            for col in 0..HASH_COLS {
                let p = PRIMES[slot][col / N_HEADS][col % N_HEADS];
                assert_eq!(OFFSETS[slot][col], end, "layer slot {slot} column {col} must start where column {col} - 1 ends");
                end += p;
            }
            assert_eq!(end as usize, rows, "layer slot {slot}: the 24 bucket ranges end at {end}, the table has {rows} rows");
        }
        // the two tables tile different totals, which is the point of reading the row count per layer
        assert_eq!(TABLE_ROWS, [384_006_168, 384_016_682]);

        // 48 distinct primes, all above the vocabulary and all actually prime
        let mut flat: Vec<i64> = PRIMES.iter().flatten().flatten().copied().collect();
        assert_eq!(flat.len(), 48);
        assert!(flat.iter().all(|&p| p > 15_999_999 && is_prime(p)), "{flat:?}");
        flat.sort_unstable();
        flat.dedup();
        assert_eq!(flat.len(), 48, "no prime is reused across the 48 buckets");
        // and they are the first 48 primes above 15_999_999, drawn in order — the first three are pinned
        assert_eq!(PRIMES[0][0][..3], [16_000_057, 16_000_079, 16_000_081]);
        assert_eq!(PRIMES[1][2][N_HEADS - 1], 16_000_889, "the last prime handed out");
        // the offsets the reference pinned
        assert_eq!(OFFSETS[0][..4], [0, 16_000_057, 32_000_136, 48_000_217]);
        assert_eq!(OFFSETS[1][HASH_COLS - 4..], [320_013_196, 336_014_037, 352_014_914, 368_015_793]);
    }

    #[test]
    fn the_multipliers_are_pinned() {
        assert_eq!(MULTIPLIERS[0], [58_737_837_727_525, 3_709_723_095_905, 27_562_777_300_965, 56_710_653_182_447]);
        assert_eq!(MULTIPLIERS[1], [51_904_348_776_103, 39_482_587_155_911, 23_700_944_747_927, 63_326_921_340_579]);
        // odd, and small enough that compressed_id * multiplier cannot overflow i64
        for m in MULTIPLIERS.iter().flatten() {
            assert_eq!(m & 1, 1, "{m} must be odd");
            assert!(*m <= 2 * MULTIPLIER_BOUND + 1, "{m} exceeds the drawn bound");
            assert!((COMPRESSED_VOCAB as i64).checked_mul(*m).is_some(), "{m} overflows with {COMPRESSED_VOCAB} ids");
        }
        // the bound itself: (i64::MAX / 129_280) / 2
        assert_eq!(MULTIPLIER_BOUND, (i64::MAX / TOKEN_MAP_LEN as i64) / 2);
    }

    // ---- the dtypes ----------------------------------------------------------------------------------------------

    #[test]
    fn e4m3_and_e8m0_decode_to_their_hand_computed_values() {
        // E4M3FN: bias 7, 3 mantissa bits, no infinity
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9), "subnormal unit");
        assert_eq!(e4m3_to_f32(0x08), 2f32.powi(-6), "smallest normal");
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        assert_eq!(e4m3_to_f32(0x3C), 1.5);
        assert_eq!(e4m3_to_f32(0x7E), 448.0, "largest finite");
        assert!(e4m3_to_f32(0x7F).is_nan());
        // E8M0: 2^(e-127)
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(0).to_bits(), 1 << 22, "2^-127, exact as an f32 subnormal");
        assert_eq!(e8m0_to_f32(1), f32::MIN_POSITIVE);
        assert_eq!(e8m0_to_f32(254), 2f32.powi(127));
        assert!(e8m0_to_f32(0xFF).is_nan());
        // bf16: a shift, and the round trip of a value bf16 holds exactly
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC000), -2.0);
        assert_eq!(round_bf16(1.5), 1.5);
        assert_eq!(round_bf16(2f32.powi(-127)), 2f32.powi(-127), "a bf16 subnormal, exact");
        // round to nearest even, at the seventh mantissa bit: a tie at 1 + 2^-8 goes to the even neighbour
        assert_eq!(round_bf16(f32::from_bits(0x3F80_8000)), 1.0, "tie, mantissa even, rounds down");
        assert_eq!(round_bf16(f32::from_bits(0x3F81_8000)), f32::from_bits(0x3F82_0000), "tie, mantissa odd, rounds up");
        assert_eq!(round_bf16(f32::from_bits(0x3F80_1000)), 1.0, "just above 1.0 still rounds to 1.0");
        assert_eq!(round_bf16(f32::from_bits(0x3F80_FFFF)), f32::from_bits(0x3F81_0000), "just below the next bf16");
    }

    // ---- the token map -------------------------------------------------------------------------------------------

    #[test]
    fn the_token_map_checks_its_length_and_its_vocabulary_by_name() {
        let dir = std::env::temp_dir().join(format!("rexl3-engram-map-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("engram_token_map.u32");

        // a good map round-trips
        let ids: Vec<u32> = (0..TOKEN_MAP_LEN).map(|i| (i % COMPRESSED_VOCAB) as u32).collect();
        let mut bytes = Vec::with_capacity(TOKEN_MAP_LEN * 4);
        for id in &ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        std::fs::write(&path, &bytes).unwrap();
        let map = TokenMap::load(&path).unwrap();
        assert_eq!((map.len(), map.get(100).unwrap(), map.get(2).unwrap()), (TOKEN_MAP_LEN, 100, 2));
        assert_eq!(map.pad_id(2).unwrap(), 2);

        // one entry short: a named Length error carrying both numbers
        std::fs::write(&path, &bytes[..bytes.len() - 4]).unwrap();
        match TokenMap::load(&path) {
            Err(TokenMapError::Length { got, want, .. }) => {
                assert_eq!((got, want), (TOKEN_MAP_LEN * 4 - 4, TOKEN_MAP_LEN * 4));
            }
            other => panic!("expected a Length error, got {other:?}"),
        }
        let e = TokenMap::load(&path).unwrap_err().to_string();
        assert!(e.contains("517120") && e.contains("517116"), "{e}");

        // a table with too few distinct ids is refused, and says how many it found
        let mut thin = ids.clone();
        for v in thin.iter_mut().skip(COMPRESSED_VOCAB) {
            *v = 5;
        }
        thin[0] = 1; // id 0 is now unused, and id 1 is used twice
        match TokenMap::from_ids(thin) {
            Err(TokenMapError::Vocab { got, want }) => assert_eq!((got, want), (COMPRESSED_VOCAB - 1, COMPRESSED_VOCAB)),
            other => panic!("expected a Vocab error, got {other:?}"),
        }

        // an id past the compressed vocabulary would index outside the safe multiplier range
        let mut wide = ids.clone();
        wide[7] = COMPRESSED_VOCAB as u32;
        match TokenMap::from_ids(wide) {
            Err(TokenMapError::IdOutOfRange { token, id, .. }) => {
                assert_eq!((token, id), (7, COMPRESSED_VOCAB as u32));
            }
            other => panic!("expected an IdOutOfRange error, got {other:?}"),
        }

        // and a token id past the map is an error, not a panic
        assert!(matches!(map.get(TOKEN_MAP_LEN as u32), Err(TokenMapError::TokenOutOfRange { .. })));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- the hash ------------------------------------------------------------------------------------------------

    #[test]
    fn hash_ids_are_pinned_for_a_five_token_sequence_at_both_layers() {
        let cfg = small_cfg();
        let map = synthetic_map();
        let h = NgramHash::new(&cfg, &map).unwrap();
        assert_eq!((h.columns(), h.pad(), h.layer_slot(1), h.layer_slot(14)), (24, 2, Some(0), Some(1)));
        assert_eq!(h.layer_slot(2), None);

        let seq = [100i64, 200, 300, 400, 500];
        // the fold, by hand, for the last position. Shift 0 is the *current* token, so the window read newest-first is
        // 500*M0 ^ 400*M1 ^ 300*M2 ^ 200*M3
        let want = 500i64.wrapping_mul(MULTIPLIERS[0][0])
            ^ 400i64.wrapping_mul(MULTIPLIERS[0][1])
            ^ 300i64.wrapping_mul(MULTIPLIERS[0][2])
            ^ 200i64.wrapping_mul(MULTIPLIERS[0][3]);
        assert_eq!(h.rolling(0, &[200, 300, 400, 500]).unwrap(), want);
        assert_eq!(want, 24_832_027_600_204_080, "the fold for (200, 300, 400, 500) at layer 1");

        // position 4 of the sequence, both layers, all 24 columns
        assert_eq!(
            h.rows_for(0, &seq[1..]).unwrap(),
            [
                5_110_997,
                29_982_261,
                32_249_043,
                50_417_843,
                77_786_219,
                86_939_723,
                107_522_364,
                117_127_165,
                136_417_549,
                145_186_027,
                168_479_071,
                189_292_017,
                201_380_993,
                223_479_182,
                232_692_812,
                250_581_238,
                266_769_576,
                285_630_745,
                302_511_362,
                318_962_349,
                335_267_024,
                351_729_661,
                354_819_289,
                372_204_440,
            ]
        );
        assert_eq!(
            h.rows_for(1, &seq[1..]).unwrap(),
            [
                7_325_096,
                20_060_760,
                44_972_937,
                61_598_687,
                69_916_850,
                95_295_733,
                106_807_374,
                120_670_545,
                139_709_995,
                159_438_000,
                172_250_758,
                179_735_226,
                205_965_626,
                223_769_167,
                232_306_000,
                251_207_530,
                257_208_580,
                281_848_490,
                302_830_939,
                316_405_848,
                328_900_534,
                346_624_448,
                356_505_304,
                373_918_596,
            ]
        );

        // every column lands inside its own bucket range, and the eight heads of an order are distinct rows
        for slot in 0..2 {
            let rows = h.rows_for(slot, &seq[1..]).unwrap();
            for (col, &row) in rows.iter().enumerate() {
                let lo = OFFSETS[slot][col] as usize;
                let hi = lo + PRIMES[slot][col / N_HEADS][col % N_HEADS] as usize;
                assert!((lo..hi).contains(&row), "layer {slot} column {col}: {row} outside {lo}..{hi}");
                assert!(row < TABLE_ROWS[slot]);
            }
            for o in 0..ORDERS {
                let mut head_rows = rows[o * N_HEADS..(o + 1) * N_HEADS].to_vec();
                head_rows.sort_unstable();
                head_rows.dedup();
                assert_eq!(head_rows.len(), N_HEADS, "layer {slot} order {o} repeated a row");
            }
            // a different order is a different read
            assert_ne!(&rows[0..N_HEADS], &rows[N_HEADS..2 * N_HEADS]);
        }
    }

    #[test]
    fn look_back_clamps_at_the_sequence_start() {
        let cfg = small_cfg();
        let map = synthetic_map();
        let h = NgramHash::new(&cfg, &map).unwrap();
        let pad = h.pad();

        // a one-token window has no look-back at all: every order is filled from the pad id
        assert_eq!(h.rows_for(0, &[100]).unwrap(), h.rows_for(0, &[pad, pad, pad, 100]).unwrap());
        // a two-token window is (pad, pad, 100, 200) in n-gram terms
        assert_eq!(h.rows_for(0, &[100, 200]).unwrap(), h.rows_for(0, &[pad, pad, 100, 200]).unwrap());
        assert_eq!(h.rows_for(0, &[100, 200, 300]).unwrap(), h.rows_for(0, &[pad, 100, 200, 300]).unwrap());
        // a full window is not padded, so it cannot equal the same three tokens behind a pad
        assert_ne!(h.rows_for(0, &[100, 200, 300, 400]).unwrap(), h.rows_for(0, &[pad, 200, 300, 400]).unwrap());

        // pinned: position 0 of [100, 200, 300, 400, 500] is the all-pad 4-gram
        assert_eq!(
            h.rows_for(1, &[100]).unwrap(),
            [
                9_301_312,
                20_767_889,
                47_333_475,
                59_713_582,
                73_904_853,
                93_945_192,
                109_781_402,
                117_718_983,
                128_438_946,
                147_273_693,
                165_294_416,
                186_967_581,
                206_826_259,
                216_369_618,
                226_970_320,
                241_179_030,
                268_097_076,
                287_109_912,
                291_239_560,
                313_584_319,
                331_524_344,
                337_649_134,
                361_103_320,
                382_371_426,
            ]
        );
        // the clamp is not the same as hashing a short window: (100) and (200, 100) differ
        assert_ne!(h.rows_for(0, &[100]).unwrap(), h.rows_for(0, &[200, 100]).unwrap());
        // a longer window than MAX_NGRAM is refused by name
        assert!(h.rows_for(0, &[1, 2, 3, 4, 5]).unwrap_err().contains("1..=4"));
        assert!(h.rows_for(2, &[1]).unwrap_err().contains("index 2"));
    }

    #[test]
    fn a_dead_token_blocks_every_ngram_that_spans_it() {
        let cfg = small_cfg();
        let map = synthetic_map();
        let h = NgramHash::new(&cfg, &map).unwrap();
        let pad = h.pad();
        assert_eq!(pad, 2, "the pad id is the compressed form of engram_pad_token_id");

        // [100, DEAD, 300]: the position of 300 sees 300, then DEAD, so shifts 1 and 2 are the pad id
        assert_eq!(h.rows_for(0, &[100, DEAD, 300]).unwrap(), h.rows_for(0, &[pad, pad, 300]).unwrap());
        // the dead token itself is all pad
        assert_eq!(h.rows_for(0, &[100, DEAD]).unwrap(), h.rows_for(0, &[pad, pad, pad, pad]).unwrap());
        // and a dead token further back blocks nothing at the positions in front of it
        assert_eq!(h.rows_for(0, &[DEAD, 100]).unwrap(), h.rows_for(0, &[pad, pad, pad, 100]).unwrap());

        // pinned, layer 1, position 2 of [100, DEAD, 300]
        assert_eq!(
            h.rows_for(1, &[100, DEAD, 300]).unwrap(),
            [
                14_842_979,
                20_720_781,
                40_590_529,
                48_513_224,
                76_478_244,
                94_170_706,
                106_205_887,
                120_282_660,
                141_509_674,
                149_902_194,
                170_647_009,
                187_445_197,
                198_252_080,
                209_895_820,
                224_055_957,
                250_134_658,
                258_705_505,
                283_722_957,
                296_096_767,
                315_122_113,
                320_915_905,
                339_251_359,
                361_609_122,
                377_399_973,
            ]
        );
        // pinned, layer 0, the same position
        assert_eq!(
            h.rows_for(0, &[100, DEAD, 300]).unwrap(),
            [
                1_335_116,
                25_612_190,
                46_731_689,
                55_706_711,
                69_235_340,
                89_762_636,
                100_029_592,
                116_551_511,
                128_471_958,
                152_723_934,
                174_119_541,
                182_903_748,
                199_461_223,
                215_769_883,
                224_269_808,
                243_356_428,
                267_223_736,
                278_441_890,
                288_549_119,
                317_610_787,
                331_654_534,
                344_724_099,
                358_278_349,
                382_571_899,
            ]
        );

        // a slot stores the mask as DEAD, so the same block comes out of the ring
        let mut s = Slot::new(pad);
        s.push(100);
        s.push(DEAD);
        s.push(300);
        assert_eq!(s.window(MAX_NGRAM), vec![100, DEAD, 300]);
        assert_eq!(h.rows_for(0, &s.window(MAX_NGRAM)).unwrap(), h.rows_for(0, &[100, DEAD, 300]).unwrap());
    }

    // ---- the table -----------------------------------------------------------------------------------------------

    struct Fixture {
        dir: PathBuf,
        rows: usize,
        cols: usize,
    }

    impl Fixture {
        /// A table of `rows x cols` fp8 with an e8m0 scale per 32 columns. `scale_of(row, tile)` picks the byte.
        fn new(
            tag: &str,
            rows: usize,
            cols: usize,
            fp8: impl Fn(usize, usize) -> u8,
            scale_of: impl Fn(usize, usize) -> u8,
        ) -> Self {
            let dir = std::env::temp_dir().join(format!("rexl3-engram-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let mut w = std::io::BufWriter::new(std::fs::File::create(dir.join("embed.weight")).unwrap());
            let mut s = std::io::BufWriter::new(std::fs::File::create(dir.join("embed.scale")).unwrap());
            let tiles = cols / FP8_TILE;
            for r in 0..rows {
                for c in 0..cols {
                    w.write_all(&[fp8(r, c)]).unwrap();
                }
                for t in 0..tiles {
                    s.write_all(&[scale_of(r, t)]).unwrap();
                }
            }
            w.flush().unwrap();
            s.flush().unwrap();
            Self { dir, rows, cols }
        }

        fn desc(&self) -> super::super::weights::MmapTable {
            super::super::weights::MmapTable {
                layer: 1,
                rows: self.rows,
                embed: super::super::weights::Tensor::new(
                    "layers.1.engram.embed.weight",
                    super::super::weights::Dtype::Fp8,
                    vec![self.rows, self.cols],
                ),
                scale: super::super::weights::Tensor::new(
                    "layers.1.engram.embed.scale",
                    super::super::weights::Dtype::E8m0,
                    vec![self.rows, self.cols / FP8_TILE],
                ),
            }
        }

        fn open(&self) -> MmapTable {
            // a standalone fixture file is the offset-0 case of a tensor inside a shard
            let at = |p: PathBuf| Loc { path: p, offset: 0 };
            MmapTable::open(&at(self.dir.join("embed.weight")), &at(self.dir.join("embed.scale")), &self.desc()).unwrap()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn a_synthetic_table_reads_back_exactly() {
        // row r, column c: fp8 = the byte the pattern names, scale = 0x7F (1.0) on even tiles, 0x7E (0.5) on odd
        let f = Fixture::new("read", 64, 256, |r, c| ((r * 7 + c * 3) % 0x7F) as u8, |_, t| if t % 2 == 0 { 0x7F } else { 0x7E });
        let t = f.open();
        assert_eq!((t.rows(), t.cols()), (64, 256));

        for &idx in &[0usize, 1, 63, 32] {
            let (w, s) = t.row(idx);
            assert_eq!((w.len(), s.len()), (256, 8));
            assert_eq!(s, &[0x7F, 0x7E, 0x7F, 0x7E, 0x7F, 0x7E, 0x7F, 0x7E]);
            // row() is a slice of the mapping, not a copy: the bytes are the pattern
            for (c, &b) in w.iter().enumerate() {
                assert_eq!(b, ((idx * 7 + c * 3) % 0x7F) as u8);
            }
            // dequantised_row_f32 applies the tile's scale, then rounds to bf16 exactly once
            let mut out = vec![0f32; 256];
            t.dequantised_row_f32(idx, &mut out).unwrap();
            for c in 0..256 {
                let v = e4m3_to_f32(w[c]) * if c / 32 % 2 == 0 { 1.0 } else { 0.5 };
                assert_eq!(out[c], round_bf16(v), "row {idx} column {c}");
            }
        }

        // a gather is the rows in order, and a short buffer is refused before anything is copied
        let ids = [0usize, 63, 17];
        let mut g = vec![0f32; ids.len() * 256];
        t.gather_dequant(&ids, &mut g).unwrap();
        for (i, &id) in ids.iter().enumerate() {
            let mut one = vec![0f32; 256];
            t.dequantised_row_f32(id, &mut one).unwrap();
            assert_eq!(&g[i * 256..(i + 1) * 256], &one[..]);
        }
        assert!(t.gather_dequant(&ids, &mut g[..255]).is_err());
        let mut bad = vec![0f32; 256];
        assert!(t.dequantised_row_f32(64, &mut bad).unwrap_err().contains("outside a table of 64"));
    }

    #[test]
    fn a_scale_of_two_to_the_minus_127_dequantises_exactly_in_f32() {
        // every weight byte is 1.0 (0x38) and every scale byte is 0x00 = 2^-127
        let f = Fixture::new("e8m0", 16, 256, |_, _| 0x38, |_, _| 0x00);
        let t = f.open();

        let mut wide = vec![0f32; 256];
        t.dequantised_row_f32(3, &mut wide).unwrap();
        assert!(wide.iter().all(|&v| v == 2f32.powi(-127)), "1.0 * 2^-127");
        assert_eq!(wide[0].to_bits(), 1 << 22, "2^-127 is an f32 subnormal with one bit set");
        assert!(wide[0] > 0.0 && wide[0] < f32::MIN_POSITIVE, "subnormal, not flushed");

        // 2.0 * 2^-127 is the smallest *normal* f32, and 0.5 * 2^-127 is exact too
        let f = Fixture::new("e8m0b", 16, 256, |_, c| if c % 2 == 0 { 0x40 } else { 0x30 }, |_, _| 0x00);
        let t = f.open();
        t.dequantised_row_f32(0, &mut wide).unwrap();
        assert_eq!(wide[0], 2f32.powi(-126));
        assert_eq!(wide[0], f32::MIN_POSITIVE);
        assert_eq!(wide[1].to_bits(), 1 << 21, "0.5 * 2^-127 = 2^-128");

        // and the round to bf16 keeps it: bf16's subnormals reach 2^-133, so 2^-127 survives the cast
        assert_eq!(round_bf16(2f32.powi(-127)), 2f32.powi(-127));
    }

    // ---- the gate ------------------------------------------------------------------------------------------------\

    #[test]
    fn the_gate_matches_a_hand_computation() {
        // two copies of dim 4: x is the all-ones stream, key is the all-twos stream, value the all-threes
        let (hc, dim) = (2usize, 4usize);
        let x = vec![1f32; hc * dim];
        let key = vec![2f32; hc * dim];
        let value = vec![3f32; dim];
        let q = vec![1f32; hc * dim];
        let k = vec![1f32; hc * dim];
        let mut out = vec![0f32; (hc + 1) * dim];
        gate_and_add(&x, &key, &value, &q, &k, hc, dim, 1e-6, &mut out).unwrap();

        // rstd = 1/sqrt(1 + eps) * 1/sqrt(4 + eps); dot = 4 * 1 * 1 * 2 * rstd * 4^-0.5
        let rstd = (1.0f32 + 1e-6).sqrt().recip() * (4.0f32 + 1e-6).sqrt().recip();
        let dot = 8.0f32 * rstd * (4f32).powf(-0.5);
        let want_gate = 1.0 / (1.0 + (-dot.abs().sqrt().copysign(dot)).exp());
        for c in 0..hc {
            for d in 0..dim {
                assert_eq!(out[c * dim + d], 1.0 + want_gate * 3.0, "copy {c} column {d}");
            }
        }
        // the shared value row is passed through untouched
        assert_eq!(&out[hc * dim..], &value[..]);

        // a negative dot gives a gate below one half, a positive one above; the sign survives the square root
        assert!(signed_sqrt_gate(-4.0) < 0.5 && signed_sqrt_gate(4.0) > 0.5);
        assert!((signed_sqrt_gate(4.0) + signed_sqrt_gate(-4.0) - 1.0).abs() < 1e-6, "an odd function, so the pair sums to 1");
        // the clamp keeps a zero dot off a zero gradient: gate(0) = sigmoid(sqrt(1e-6))
        assert_eq!(signed_sqrt_gate(0.0), 1.0 / (1.0 + (-(1e-6f32).sqrt()).exp()));
        // `copysign` follows the *sign bit*, so a negative zero lands on the low side — torch's `copysign` does the same
        assert_eq!(signed_sqrt_gate(-0.0), 1.0 / (1.0 + ((1e-6f32).sqrt()).exp()));
        assert!(signed_sqrt_gate(-0.0) < 0.5 && signed_sqrt_gate(0.0) > 0.5);
        // the normalisation is per copy, not joint: a stream whose copies differ gives different gates
        let x2 = [1.0f32, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0];
        let mut out2 = vec![0f32; (hc + 1) * dim];
        gate_and_add(&x2, &key, &value, &q, &k, hc, dim, 1e-6, &mut out2).unwrap();
        assert_ne!(out2[0], out2[dim], "each copy normalises over dim alone");
        assert!(gate_and_add(&x, &key, &value, &q, &k, hc, dim, 1e-6, &mut out[..3]).is_err());
    }

    // ---- the state -----------------------------------------------------------------------------------------------

    /// A whole tiny engram layer: 64 rows, 8 heads of 32 dims, hidden 32, 3 orders.
    struct Layer {
        dir: PathBuf,
        cfg: EngramConfig,
    }

    impl Layer {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("rexl3-engram-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            // A sparse table at the pack's real row count: the file is 12.3 GB of holes and a test writes only the 24 rows
            // it reads, so the geometry under test is the production one at no disk cost.
            let (h, d) = (32usize, 32usize);
            std::fs::File::create(dir.join("embed.weight")).unwrap().set_len((TABLE_ROWS[0] * d) as u64).unwrap();
            std::fs::File::create(dir.join("embed.scale")).unwrap().set_len((TABLE_ROWS[0] * d / FP8_TILE) as u64).unwrap();
            // q/k: bf16 [4, h], 1.0 on the diagonal
            let mut bf = Vec::new();
            for k in 0..4 {
                for i in 0..h {
                    let v: f32 = if i == k % h { 1.0 } else { 0.0 };
                    bf.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
                }
            }
            std::fs::write(dir.join("q_weight"), &bf).unwrap();
            std::fs::write(dir.join("k_weight"), &bf).unwrap();
            // wkv: [5*h, 24*d], fp8 1.0 on the diagonal, 1.0 over every 32x32 tile
            let (out, inp) = (5 * h, HASH_COLS * d);
            let mut ww = vec![0u8; out * inp];
            for o in 0..out {
                for i in 0..inp {
                    ww[o * inp + i] = if o % inp == i { 0x38 } else { 0x00 };
                }
            }
            let tiles = (out / FP8_TILE) * (inp / FP8_TILE);
            std::fs::write(dir.join("wkv.weight"), &ww).unwrap();
            std::fs::write(dir.join("wkv.scale"), vec![0x7Fu8; tiles]).unwrap();
            Self {
                dir,
                cfg: EngramConfig {
                    layer_ids: vec![1],
                    num_embeddings: vec![384_006_168],
                    max_ngram_size: 4,
                    vocab_size: 16_000_000,
                    n_heads: N_HEADS,
                    head_dim: d,
                    pad_token_id: 2,
                    compressed_vocab_size: COMPRESSED_VOCAB,
                },
            }
        }

        /// Write `rows` as all-ones fp8 with a 1.0 scale — the only rows these tests' gathers touch.
        fn write_rows(&self, rows: &[usize]) {
            use std::io::{Seek, SeekFrom};
            let d = self.cfg.head_dim;
            let mut w = std::fs::OpenOptions::new().write(true).open(self.dir.join("embed.weight")).unwrap();
            let mut s = std::fs::OpenOptions::new().write(true).open(self.dir.join("embed.scale")).unwrap();
            for &r in rows {
                w.seek(SeekFrom::Start((r * d) as u64)).unwrap();
                w.write_all(&[0x38u8; 32]).unwrap();
                s.seek(SeekFrom::Start(r as u64)).unwrap();
                s.write_all(&[0x7Fu8]).unwrap();
            }
            w.flush().unwrap();
            s.flush().unwrap();
        }

        fn files(&self) -> EngramFiles {
            // a standalone fixture file is the offset-0 case of a tensor inside a shard
            let at = |p: PathBuf| Loc { path: p, offset: 0 };
            EngramFiles {
                embed: at(self.dir.join("embed.weight")),
                scale: at(self.dir.join("embed.scale")),
                q_weight: at(self.dir.join("q_weight")),
                k_weight: at(self.dir.join("k_weight")),
                q_f32: false,
                k_f32: false,
                wkv: WkvFiles::Plain { weight: at(self.dir.join("wkv.weight")), scale: at(self.dir.join("wkv.scale")) },
            }
        }
    }

    impl Drop for Layer {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn a_short_prompt_is_left_padded_with_the_pad_id() {
        // the fixture table is 64 rows but the hash's rows are 384 million, so this test only exercises the hash
        let cfg = small_cfg();
        let map = synthetic_map();
        let h = NgramHash::new(&cfg, &map).unwrap();
        let pad = h.pad();
        // one token: the 2-gram is (pad, 7), the 3-gram (pad, pad, 7), the 4-gram (pad, pad, pad, 7)
        let mut s = Slot::new(pad);
        s.push(7);
        assert_eq!(s.window(MAX_NGRAM), vec![7]);
        assert_eq!(h.rows_for(0, &s.window(MAX_NGRAM)).unwrap(), h.rows_for(0, &[7]).unwrap());
        assert_eq!(h.rows_for(0, &[7]).unwrap()[..8], h.rows_for(0, &[pad, 7]).unwrap()[..8]);
        // and a full ring keeps only the last four tokens
        for t in 0..RING as i64 {
            s.push(t);
        }
        assert_eq!(s.window(MAX_NGRAM), vec![(RING - 4) as i64, (RING - 3) as i64, (RING - 2) as i64, (RING - 1) as i64]);
    }

    #[test]
    fn store_masked_treats_the_dead_sentinel_id_as_dead() {
        let l = Layer::new("store");
        let map = Arc::new(synthetic_map());
        let mut s = EngramState::open(&l.cfg, 1, 32, &l.files(), 2, None, Arc::clone(&map)).unwrap();
        // the image splice writes DEAD into the id array itself; storing it must be DEAD, not a token-map error
        s.store_masked(0, &[11u32, DEAD_AS_U32, 13u32], None).expect("the sentinel stores as DEAD");
        let x = vec![0.5f32; 4 * 32];
        let y = s
            .forward_masked(0, &[11u32, DEAD_AS_U32], Some(&[true, false]), &x)
            .expect("a gather over a DEAD position is masked, not an error");
        assert!(y.iter().all(|v| v.is_finite()));
    }

    /// A tree round's ring save/restore/replay (T2.5a lane 2): the gather APPENDS per row, so a sibling
    /// verify path's tokens sit in the next path's n-gram context without the restore — and the commit's
    /// replay (base + the accepted window's tokens) IS the serial ring.
    #[test]
    fn the_tree_rounds_ring_snapshot_restore_and_replay_match_a_serial_ring() {
        let l = Layer::new("tree-ring");
        let map = Arc::new(synthetic_map());
        let mut s = EngramState::open(&l.cfg, 1, 32, &l.files(), 2, None, Arc::clone(&map)).unwrap();
        s.store(0, &[11, 12, 13]).expect("the committed prefix stores");
        let snap = s.snapshot_slot(0).expect("slot 0 has a ring");
        // a sibling path's window appends its own tokens...
        s.store(0, &[21, 22]).expect("the sibling path stores");
        assert_ne!(s.snapshot_slot(0).unwrap(), snap, "the sibling's appends moved the ring");
        // ...and the restore puts it back exactly
        s.restore_slot(0, &snap);
        assert_eq!(s.snapshot_slot(0).unwrap(), snap, "the restore is exact");
        // the commit's replay over the restored base is the serial ring of the accepted path
        let accepted = [14u32, 15];
        s.store(0, &accepted).expect("the accepted window replays");
        let round = s.snapshot_slot(0).unwrap();
        let mut serial = EngramState::open(&l.cfg, 1, 32, &l.files(), 2, None, Arc::clone(&map)).unwrap();
        serial.store(0, &[11, 12, 13]).expect("the prefix stores");
        serial.store(0, &accepted).expect("the accepted window stores");
        assert_eq!(round, serial.snapshot_slot(0).unwrap(), "the replayed ring is the serial one");
    }

    #[test]
    fn forward_gathers_the_hash_rows_through_wkv_and_gates_them() {
        let l = Layer::new("forward");
        let map = Arc::new(synthetic_map());
        // the 24 rows the window [11, 12] reads, written as all-ones so the gather has a known value
        let probe = NgramHash::new(&l.cfg, &map).unwrap();
        let rows = probe.rows_for(0, &[11, 12]).unwrap();
        l.write_rows(&rows);

        let mut s = EngramState::open(&l.cfg, 1, 32, &l.files(), 2, None, Arc::clone(&map)).unwrap();
        let x = vec![0.5f32; 4 * 32];
        let y = s.forward(0, &[11, 12], &x).unwrap();
        assert_eq!(y.len(), 5 * 32, "wkv's 5 x hidden rows");
        assert_eq!(s.position_rows(0).unwrap(), rows, "forward reads the hash's rows, not others");
        assert!(y.iter().all(|v| v.is_finite()));

        // wkv is the identity on its first 5 * 32 rows, so key and value are all ones. q and k are the identity on the
        // first four columns of each copy, so the dot product sums one term per copy: rstd = 1/sqrt(0.25 + eps) *
        // 1/sqrt(1 + eps), dot = 0.5 * rstd * 32^-0.5
        let eps = 1e-6f32;
        let rstd = (0.25f32 + eps).sqrt().recip() * (1.0f32 + eps).sqrt().recip();
        let dot = 0.5f32 * rstd * (32f32).powf(-0.5);
        let gate = 1.0 / (1.0 + (-dot.abs().sqrt().copysign(dot)).exp());
        assert!((gate - 0.603_586_5).abs() < 1e-5, "gate = {gate}");
        for c in 0..4 {
            for d in 0..32 {
                assert!((y[c * 32 + d] - (0.5 + gate)).abs() < 1e-6, "copy {c} column {d}: {}", y[c * 32 + d]);
            }
        }
        assert!((y[4 * 32] - 1.0).abs() < 1e-6, "the shared value row is passed through ungated");

        // the same window reads the same rows; a different token reads different ones
        let mut s2 = EngramState::open(&l.cfg, 1, 32, &l.files(), 1, None, map).unwrap();
        assert_eq!(y, s2.forward(0, &[11, 12], &x).unwrap());
        let before = s2.position_rows(0).unwrap();
        s2.forward(0, &[13], &x).unwrap();
        assert_ne!(before, s2.position_rows(0).unwrap());
        // a masked position shuts the gate: the residual stream passes through, and only the value row is written
        // the mask stores DEAD, so the window this position reads is [11, 12, DEAD]
        l.write_rows(&probe.rows_for(0, &[11, 12, DEAD]).unwrap());
        let masked = s.forward_masked(0, &[21], Some(&[false]), &x).unwrap();
        assert_eq!(&masked[..4 * 32], &x[..], "a dead position is an identity on the copies");
        assert!((masked[4 * 32] - 1.0).abs() < 1e-6, "the value row is not gated");
        // a wrong-width residual stream is refused by name, and so is a slot that does not exist
        assert!(s2.forward(0, &[1], &[0.0; 4]).unwrap_err().contains("residual stream"));
        assert!(s2.forward(9, &[1], &x).unwrap_err().contains("slot 9"));
    }

    #[test]
    fn a_config_for_a_different_model_is_refused() {
        let map = synthetic_map();
        let bad = [
            EngramConfig { layer_ids: vec![2], ..small_cfg() },
            EngramConfig { num_embeddings: vec![4096, 384_016_682], ..small_cfg() },
            EngramConfig { max_ngram_size: 3, ..small_cfg() },
            EngramConfig { n_heads: 4, ..small_cfg() },
            EngramConfig { compressed_vocab_size: 100_000, ..small_cfg() },
        ];
        for cfg in bad {
            assert!(NgramHash::new(&cfg, &map).is_err(), "{cfg:?} must be refused");
        }
    }

    #[test]
    fn slot_snapshot_and_restore_preserves_ngram_ring_across_slots() {
        let mut s0 = Slot::new(2);
        for t in [10i64, 20, 30, 40, 50] {
            s0.push(t);
        }
        let snap = s0.clone();
        let mut s1 = snap;
        assert_eq!(s0.window(4), s1.window(4));
        s0.push(60);
        s1.push(60);
        assert_eq!(s0.window(4), s1.window(4));
    }
}

#[cfg(test)]
mod wkv_group_tests {
    use super::*;
    use gb_exl3::checkpoint::{Group, LoadedGroup};

    /// A synthetic 4-bit group of `[k, n]` with unit scales and a deterministic trellis pattern.
    fn synth_group(k: usize, n: usize) -> LoadedGroup {
        let bits = gb_exl3::Bits::from_twice(8).unwrap();
        let (kt, nt) = (k / 16, n / 16);
        let nw = bits.tile_words();
        let trellis: Vec<i16> = (0..kt * nt * nw).map(|i| ((i * 31 + 7) & 0xFFFF) as i16).collect();
        LoadedGroup {
            group: Group {
                prefix: "layers.1.engram.wkv".into(),
                bits,
                codebook: gb_exl3::Codebook::Mul1,
                k,
                n,
                in_scales: "suh",
                out_scales: "svh",
                bias: false,
                files: vec![],
                bytes: 0,
            },
            trellis,
            kt,
            nt,
            suh: vec![1.0; k],
            svh: vec![1.0; n],
            bias: None,
        }
    }

    const OUT: usize = 128;
    const INP: usize = 128;

    /// The golden: the same group yields byte-identical weight and scale every conversion.
    #[test]
    fn from_group_is_deterministic() {
        let g = synth_group(INP, OUT);
        let a = Fp8Matrix::from_group(&g, OUT, INP).unwrap();
        let b = Fp8Matrix::from_group(&g, OUT, INP).unwrap();
        assert_eq!(a.weight.bytes(), b.weight.bytes(), "weight bytes are the golden");
        assert_eq!(a.scale.bytes(), b.scale.bytes(), "scale bytes are the golden");
    }

    /// The conversion's error is the e4m3 tile quantisation and nothing more: every element is
    /// within 2^-4 relative (plus a scale-sized f32 cushion) of the dense decode.
    #[test]
    fn from_group_error_is_within_the_e4m3_tile_bound() {
        let g = synth_group(INP, OUT);
        let m = Fp8Matrix::from_group(&g, OUT, INP).unwrap();
        let dense =
            gb_exl3::decode_weight(&g.trellis, g.kt, g.nt, g.group.bits, &gb_exl3::codebook(g.group.codebook), &g.suh, &g.svh);
        let tiles = INP / FP8_TILE;
        for o in 0..OUT {
            for i in 0..INP {
                let want = dense[i * OUT + o] as f32;
                let scale = e8m0_to_f32(m.scale.bytes()[(o / FP8_TILE) * tiles + i / FP8_TILE]);
                let got = gb_cuda::fp8_linear::e4m3_to_f32(m.weight.bytes()[o * INP + i]) * scale;
                let bound = want.abs() * 0.0625 + scale * 1e-3;
                assert!((got - want).abs() <= bound, "[{o},{i}] {got} vs {want} (scale {scale})");
            }
        }
    }

    /// Sensitivity: a changed trellis word must change the converted bytes.
    #[test]
    fn from_group_is_sensitive_to_the_trellis() {
        let mut g = synth_group(INP, OUT);
        let a = Fp8Matrix::from_group(&g, OUT, INP).unwrap();
        g.trellis[3] = g.trellis[3].wrapping_add(0x00FF);
        let b = Fp8Matrix::from_group(&g, OUT, INP).unwrap();
        assert_ne!(a.weight.bytes(), b.weight.bytes());
    }

    /// A group whose [k, n] does not match the config's wkv extent is refused by name and number.
    #[test]
    fn from_group_refuses_a_mismatched_shape() {
        let g = synth_group(INP, OUT);
        let err = match Fp8Matrix::from_group(&g, OUT, 256) {
            Err(e) => e,
            Ok(_) => panic!("a mismatched shape must be refused"),
        };
        assert!(err.contains("256") && err.contains("128"), "names the numbers: {err}");
    }
}
