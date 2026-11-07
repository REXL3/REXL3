//! Attention of the Nemotron 3.5 Super VL NoPE GQA layers (`csrc/nemotron_attn.cu`): 32 query heads of 128 over 2 kv heads, no RoPE,
//! causal, scale 128^-0.5, a fp16 KV cache in 256-token pages behind a [`BlockTable`]. The reduction tree is the one of
//! `tools/attn_gpu_design.md`: 512-key chunks at absolute multiples of 512, 64-key tiles inside a chunk, P rounded to fp16 once per
//! tile, and the chunk states merged in a fixed balanced tree over the chunk index.
//!
//! * [`kv_write`] rounds the fp32 K and V projections to fp16 once and scatters them into their pages. It runs before attention.
//! * [`attn_decode`] serves one row or a verify window of up to [`MAX_DECODE_ROWS`] rows.
//! * [`attn_prefill`] serves any number of rows of any sequences.
//!
//! Both attention entry points launch the same two kernels (a partial per row, kv head and chunk, then a merge per row and query
//! head), so one code path produces the bits for both. A row's output depends only on its query, its own slot's keys up to its
//! position, and the tree; not on the other rows, their number or order, or where the blocks sit. The inputs and outputs are fp32
//! `[rows][32][128]` (q and out), and the caches are fp16.
//!
//! A second, prefill-only arithmetic for deep positions (a flash-style tensor-core kernel with its own path id) lives in
//! [`crate::nemotron_attn_prefill`]; it reads the same cache and block table and reuses this module's [`Scratch`] and [`RowIndex`].
//! Nothing in this module's kernels changes with it: decode, verify windows and prefill below the threshold run the tree.

use crate::attention::BlockTable;
use crate::{CudaError, DevBuf, Stream};

pub const N_Q_HEADS: usize = 32;
pub const N_KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 128;
pub const BLOCK_TOKENS: usize = 256;
/// Keys per chunk (absolute: chunk `c` covers positions `[512 c, 512 c + 512)`).
pub const CHUNK_KEYS: usize = 512;
/// Keys per tile inside a chunk.
pub const TILE_KEYS: usize = 64;
/// The most rows one [`attn_decode`] call takes (a verify window).
pub const MAX_DECODE_ROWS: usize = 8;
/// fp16 elements of one token's K (or V) row for both kv heads.
pub const KV_TOKEN_ELEMS: usize = N_KV_HEADS * HEAD_DIM;
/// fp32 elements of one query or output row (all 32 query heads).
pub const Q_ROW_ELEMS: usize = N_Q_HEADS * HEAD_DIM;
/// Chunks a scratch may hold (2^14 chunks is 8M keys). The merge stack is `log2(chunks) + 1` entries deep and holds 16.
pub const MAX_CHUNKS: usize = 1 << 14;
/// The first absolute position (0-based) the flash path serves, whichever entry point carries the row: rows with more than 8,192 keys
/// ([`crate::nemotron_attn_prefill`]). Below it a row is the tree's.
pub const FLASH_FROM_POS: usize = 8192;
/// Keys of one flash split: the partition of a flash row's keys into partials, at absolute multiples of it.
pub const FLASH_CHUNK_KEYS: usize = 8192;
/// The tree's chunks of a row below [`FLASH_FROM_POS`]: no row the tree serves needs more, whatever the context.
pub const TREE_CHUNKS_BELOW: usize = FLASH_FROM_POS / CHUNK_KEYS;

/// The flash splits of a launch built for `max_chunks` tree chunks (the deepest row's, or a graph bucket's): one per 8,192 keys, at
/// least one. A power-of-two bucket of at least 32 chunks gives `bucket / 16`, a power of two itself.
pub const fn flash_splits(max_chunks: usize) -> usize {
    let s = max_chunks.div_ceil(TREE_CHUNKS_BELOW);
    if s == 0 {
        1
    } else {
        s
    }
}

/// The KV cache's element type (`--kv-dtype`): fp16 (the default, the arithmetic every golden was made with) or int8 with a
/// power-of-two scale per (token, kv head) (`csrc/nemotron_kv8.cuh`, `docs/NEMOTRON_KV_INT8.md`). Both keep 256-token pages behind the
/// same block table; only a page's bytes differ. The int8 kernels dequantise exactly into the fp16 kernels' tiles, so on any cache the
/// int8 path's output is the fp16 path's output on the dequantised values, bit for bit. (An E4M3 fp8 format was evaluated and rejected:
/// [`KvDtype::parse`] refuses `fp8` by name.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KvDtype {
    #[default]
    F16,
    /// int8 values `q` in [-127, 127] and one power-of-two exponent per token and kv head: `q * 2^e`.
    Int8,
}

/// Why `fp8` is refused (the flag's and the variable's error).
pub const FP8_REJECTED: &str = "fp8 KV was evaluated and rejected, see docs/NEMOTRON_KV_INT8.md (use int8)";

impl KvDtype {
    /// Bytes of one layer's K (or V) page of [`BLOCK_TOKENS`] tokens: fp16 131,072; int8 65,536 of values and 512 exponent bytes.
    pub const fn page_bytes(self) -> usize {
        match self {
            KvDtype::F16 => BLOCK_TOKENS * KV_TOKEN_ELEMS * 2,
            KvDtype::Int8 => kv8::PAGE_BYTES,
        }
    }

    /// Bytes one token costs in one attention layer, K and V together: 1,024 (fp16) or 516 (int8).
    pub const fn token_bytes(self) -> usize {
        2 * self.page_bytes() / BLOCK_TOKENS
    }

    /// One byte a value: the 66,048-byte page of [`kv8`].
    pub const fn is_8bit(self) -> bool {
        !matches!(self, KvDtype::F16)
    }

    /// The flag's spelling (`fp16`, `int8`).
    pub const fn name(self) -> &'static str {
        match self {
            KvDtype::F16 => "fp16",
            KvDtype::Int8 => "int8",
        }
    }

    /// Parses the flag's value: `fp16` (also `f16`, `half`) or `int8` (also `i8`). `fp8` and its spellings are a named refusal
    /// ([`FP8_REJECTED`]); anything else is an error naming the choices.
    pub fn parse(v: &str) -> Result<Self, String> {
        match v.trim().to_ascii_lowercase().as_str() {
            "fp16" | "f16" | "half" => Ok(KvDtype::F16),
            "int8" | "i8" => Ok(KvDtype::Int8),
            "fp8" | "e4m3" | "fp8_e4m3" | "fp8e4m3" => Err(FP8_REJECTED.to_string()),
            other => Err(format!("{other:?}: expected fp16 or int8")),
        }
    }
}

/// The environment variable that selects the KV cache's element type for the process (`fp16`, the default, or `int8`); the server's
/// `--kv-dtype` flag sets it. Every TP rank must run with the same value (the ranks compare it at the rendezvous).
pub const KV_DTYPE_ENV: &str = "REXL3_KV_DTYPE";

/// The KV element type this process is configured for ([`KV_DTYPE_ENV`]; unset or empty: fp16). An unrecognised value, and `fp8`, are
/// errors, so a typo never silently serves another dtype.
pub fn configured_kv_dtype() -> Result<KvDtype, String> {
    match std::env::var(KV_DTYPE_ENV) {
        Err(_) => Ok(KvDtype::F16),
        Ok(v) if v.trim().is_empty() => Ok(KvDtype::F16),
        Ok(v) => KvDtype::parse(&v).map_err(|e| format!("{KV_DTYPE_ENV}={v:?}: {e}")),
    }
}

/// [`configured_kv_dtype`], panicking on an unrecognised value (the server validates the flag before anything is built).
pub fn kv_dtype() -> KvDtype {
    configured_kv_dtype().unwrap_or_else(|e| panic!("{e}"))
}

/// The int8 page (`csrc/nemotron_kv8.cuh`) and a host model of its quantiser: the device's bytes are pinned to these functions bit for
/// bit (`tests/nemotron_kv_int8.rs`).
pub mod kv8 {
    use super::{BLOCK_TOKENS, HEAD_DIM, KV_TOKEN_ELEMS, N_KV_HEADS};
    /// The values of a page, `[256 tokens][2 kv heads][128]` bytes.
    pub const DATA_BYTES: usize = BLOCK_TOKENS * KV_TOKEN_ELEMS;
    /// Where the exponents start: `[256 tokens][2 kv heads]` signed bytes.
    pub const EXP_OFF: usize = DATA_BYTES;
    /// One layer's K (or V) page: 66,048 bytes.
    pub const PAGE_BYTES: usize = DATA_BYTES + BLOCK_TOKENS * N_KV_HEADS;
    /// The exponent range: `q * 2^e` stays an exact fp16 value.
    pub const I8_EMIN: i32 = -24;
    pub const I8_EMAX: i32 = 8;

    /// The exponent of a row whose absolute maximum is `amax`: the smallest `e` with `amax / 2^e <= 127`, clamped.
    pub fn exponent_i8(amax: f32) -> i32 {
        let b = amax.to_bits();
        let big_e = ((b >> 23) & 0xff) as i32 - 127;
        let e = if (b & 0x7f_ffff) <= 0x7e_0000 { big_e - 6 } else { big_e - 5 };
        e.clamp(I8_EMIN, I8_EMAX)
    }

    /// One (token, kv head) row of 128 values: its exponent and the bytes (two's complement `q`).
    pub fn quantise_row_i8(x: &[f32]) -> (i8, Vec<u8>) {
        assert_eq!(x.len(), HEAD_DIM);
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let e = exponent_i8(amax);
        let inv = 2f32.powi(-e);
        (e as i8, x.iter().map(|&v| ((v * inv).round_ties_even().clamp(-127.0, 127.0) as i8) as u8).collect())
    }

    /// The value a stored byte holds at exponent `e` (fp32, exact).
    pub fn value(b: u8, e: i8) -> f32 {
        (b as i8) as f32 * 2f32.powi(e as i32)
    }

    /// The fp16 bits a stored byte dequantises to at exponent `e` (exact).
    pub fn dequant_i8_f16_bits(b: u8, e: i8) -> u16 {
        half_bits(value(b, e))
    }

    /// fp32 -> fp16 bits for values that are exactly representable (the dequantised ones are).
    fn half_bits(v: f32) -> u16 {
        let sign = if v.is_sign_negative() { 0x8000u16 } else { 0 };
        let a = v.abs();
        if a == 0.0 {
            return sign;
        }
        let e = ((a.to_bits() >> 23) & 0xff) as i32 - 127;
        if e < -14 {
            let m = a / 2f32.powi(-24);
            assert!(m.fract() == 0.0, "{v} is not an fp16 value");
            return sign | m as u16;
        }
        let m = (a / 2f32.powi(e) - 1.0) * 1024.0;
        assert!(m.fract() == 0.0 && e <= 15, "{v} is not an fp16 value");
        sign | (((e + 15) as u16) << 10) | m as u16
    }
}

/// Bytes of one (row, chunk or split) group of partials: 32 query heads of `(m, l)` (8 bytes) and `o[128]` (fp32).
pub const GROUP_BYTES: usize = N_Q_HEADS * (8 + HEAD_DIM * 4);

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_nemo_kv_write(
            k_in: *const c_void,
            v_in: *const c_void,
            slot: *const c_void,
            pos: *const c_void,
            table: *const c_void,
            max_blocks: i32,
            kc: *mut c_void,
            vc: *mut c_void,
            rows: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_nemo_attn_partial(
            q: *const c_void,
            kc: *const c_void,
            vc: *const c_void,
            table: *const c_void,
            max_blocks: i32,
            slot: *const c_void,
            pos: *const c_void,
            rows: i32,
            chunks: i32,
            limit: i32,
            stride: i32,
            part_ml: *mut c_void,
            part_o: *mut c_void,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_nemo_kv_write_i8(
            k_in: *const c_void,
            v_in: *const c_void,
            slot: *const c_void,
            pos: *const c_void,
            table: *const c_void,
            max_blocks: i32,
            kc: *mut c_void,
            vc: *mut c_void,
            rows: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_nemo_attn_partial_i8(
            q: *const c_void,
            kc: *const c_void,
            vc: *const c_void,
            table: *const c_void,
            max_blocks: i32,
            slot: *const c_void,
            pos: *const c_void,
            rows: i32,
            chunks: i32,
            limit: i32,
            stride: i32,
            part_ml: *mut c_void,
            part_o: *mut c_void,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_nemo_attn_merge(
            pos: *const c_void,
            part_ml: *const c_void,
            part_o: *const c_void,
            rows: i32,
            stride: i32,
            limit: i32,
            out: *mut c_void,
            s: *mut c_void,
        ) -> i32;
    }
}

/// One layer's paged K and V caches, each `n_blocks` pages of [`KvDtype::page_bytes`]: fp16 `[256 tokens][2 kv heads][128]`, or the
/// int8 page of [`kv8`].
#[derive(Debug)]
pub struct NemoKv {
    pub k: DevBuf,
    pub v: DevBuf,
    pub n_blocks: usize,
    pub dtype: KvDtype,
}

impl NemoKv {
    /// Allocates (uninitialised) fp16 caches of `n_blocks` pages each.
    pub fn new(n_blocks: usize) -> Result<Self, CudaError> {
        Self::new_typed(n_blocks, KvDtype::F16)
    }

    /// Allocates (uninitialised) caches of `n_blocks` pages each of `dtype`.
    pub fn new_typed(n_blocks: usize, dtype: KvDtype) -> Result<Self, CudaError> {
        let bytes = n_blocks * dtype.page_bytes();
        Ok(Self { k: DevBuf::alloc(bytes)?, v: DevBuf::alloc(bytes)?, n_blocks, dtype })
    }

    /// Token capacity.
    pub fn capacity_tokens(&self) -> usize {
        self.n_blocks * BLOCK_TOKENS
    }
}

/// Per-row `(slot, absolute position)` on the device, `i32` each. Row `r` writes its K and V at position `pos[r]` of its slot, and
/// attends to keys `0..=pos[r]`.
#[derive(Debug)]
pub struct RowIndex {
    pub slot: DevBuf,
    pub pos: DevBuf,
    pub rows: usize,
    max_pos: usize,
    /// A floor for [`RowIndex::max_chunks`] (0: none): the chunk count a captured graph was built for.
    min_chunks: usize,
    /// Host copies of the two arrays (the rows of the last upload): a prefill call's split at the position threshold and the flash
    /// path's row blocks are planned on the host.
    host_slot: Vec<i32>,
    host_pos: Vec<i32>,
}

impl RowIndex {
    /// Builds the index from host arrays of equal, non-zero length.
    pub fn from_host(slots: &[i32], pos: &[i32]) -> Result<Self, CudaError> {
        assert!(!pos.is_empty() && slots.len() == pos.len(), "RowIndex needs one slot and one position per row");
        assert!(pos.iter().all(|&p| p >= 0), "positions are absolute and not negative");
        let max_pos = pos.iter().copied().max().unwrap_or(0) as usize;
        Ok(Self {
            slot: DevBuf::from_slice(slots)?,
            pos: DevBuf::from_slice(pos)?,
            rows: pos.len(),
            max_pos,
            min_chunks: 0,
            host_slot: slots.to_vec(),
            host_pos: pos.to_vec(),
        })
    }

    /// An empty index with device buffers for up to `capacity` rows, to be filled by [`RowIndex::upload`] and reused for every step: no
    /// allocation per step, and the launches of a captured graph keep reading the same two buffers.
    pub fn with_capacity(capacity: usize) -> Result<Self, CudaError> {
        assert!(capacity > 0, "RowIndex needs room for at least one row");
        Ok(Self {
            slot: DevBuf::zeroed(capacity * 4)?,
            pos: DevBuf::zeroed(capacity * 4)?,
            rows: 0,
            max_pos: 0,
            min_chunks: 0,
            host_slot: Vec::with_capacity(capacity),
            host_pos: Vec::with_capacity(capacity),
        })
    }

    /// Rows this index has room for.
    pub fn capacity(&self) -> usize {
        self.slot.bytes() / 4
    }

    /// Rewrites the index in place (same device buffers; `rows` and the host-side deepest position follow it). The previous step must
    /// have finished reading the buffers: the copy is a synchronous one and does not wait for the stream.
    pub fn upload(&mut self, slots: &[i32], pos: &[i32]) -> Result<(), CudaError> {
        assert!(!pos.is_empty() && slots.len() == pos.len(), "RowIndex needs one slot and one position per row");
        assert!(pos.len() <= self.capacity(), "RowIndex of capacity {} given {} rows", self.capacity(), pos.len());
        assert!(pos.iter().all(|&p| p >= 0), "positions are absolute and not negative");
        self.slot.upload(slots)?;
        self.pos.upload(pos)?;
        self.rows = pos.len();
        self.max_pos = pos.iter().copied().max().unwrap_or(0) as usize;
        self.min_chunks = 0;
        self.keep_host(slots, pos);
        Ok(())
    }

    /// [`RowIndex::upload`] in stream order: the two copies are queued on `st`, so a kernel launched on `st` afterwards reads the new rows
    /// (a plain `upload` is a copy on the legacy default stream, which a non-blocking stream does not wait for). The arrays are pageable
    /// host memory and are staged before the call returns.
    pub fn upload_on(&mut self, slots: &[i32], pos: &[i32], st: &Stream) -> Result<(), CudaError> {
        assert!(!pos.is_empty() && slots.len() == pos.len(), "RowIndex needs one slot and one position per row");
        assert!(pos.len() <= self.capacity(), "RowIndex of capacity {} given {} rows", self.capacity(), pos.len());
        assert!(pos.iter().all(|&p| p >= 0), "positions are absolute and not negative");
        // SAFETY: both buffers are device allocations of at least `pos.len() * 4` bytes and the sources are that many readable bytes
        unsafe {
            st.memcpy_h2d_async(self.slot.ptr(), slots.as_ptr() as *const u8, std::mem::size_of_val(slots))?;
            st.memcpy_h2d_async(self.pos.ptr(), pos.as_ptr() as *const u8, std::mem::size_of_val(pos))?;
        }
        self.rows = pos.len();
        self.max_pos = pos.iter().copied().max().unwrap_or(0) as usize;
        self.min_chunks = 0;
        self.keep_host(slots, pos);
        Ok(())
    }

    fn keep_host(&mut self, slots: &[i32], pos: &[i32]) {
        self.host_slot.clear();
        self.host_slot.extend_from_slice(slots);
        self.host_pos.clear();
        self.host_pos.extend_from_slice(pos);
    }

    /// The rows' slots and positions as last uploaded (host copies, one per row).
    pub fn host_rows(&self) -> (&[i32], &[i32]) {
        (&self.host_slot, &self.host_pos)
    }

    /// The deepest row's position in this index.
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }

    /// The chunks the deepest row in this index reads, or the floor set by [`RowIndex::set_min_chunks`] if that is more. The attention launch
    /// is `(rows, kv heads, max_chunks)` blocks and a row with fewer chunks of its own exits early without changing a bit, so a launch
    /// made for more chunks than the deepest row has gives the same output (it is what lets one captured graph serve a range of contexts).
    pub fn max_chunks(&self) -> usize {
        ((self.max_pos + 1).div_ceil(CHUNK_KEYS)).max(self.min_chunks)
    }

    /// Makes [`RowIndex::max_chunks`] at least `chunks` until the next upload. The scratch must hold that many chunks.
    pub fn set_min_chunks(&mut self, chunks: usize) {
        self.min_chunks = chunks;
    }

    /// The chunks the deepest row reads, without any floor.
    pub fn real_chunks(&self) -> usize {
        (self.max_pos + 1).div_ceil(CHUNK_KEYS)
    }
}

/// The scratch of both arithmetics. The tree's per-(row, chunk, query head) partials, `(m, l)` and the unnormalized `o[128]`, fp32, for
/// `rows` rows of up to `max_chunks` chunks each; and the flash path's own, per (row, 8,192-key split, query head), with its row plan.
/// The tree never serves a row at or above [`FLASH_FROM_POS`], so a serving scratch ([`Scratch::new_serving`]) holds the tree's for
/// [`TREE_CHUNKS_BELOW`] chunks a row whatever the context, and what scales with the context is the flash path's, one group per 8,192
/// keys: 128 per row at 1M tokens.
#[derive(Debug)]
pub struct Scratch {
    part_ml: DevBuf,
    part_o: DevBuf,
    rows: usize,
    max_chunks: usize,
    flash_ml: DevBuf,
    flash_o: DevBuf,
    /// The flash path's row plan: a count and one `(first row, rows)` pair a row block (`1 + rows` pairs of two i32).
    flash_plan: DevBuf,
    flash_splits: usize,
}

impl Scratch {
    /// A scratch for DIRECT use of the tree's entry points ([`attn_decode`], [`attn_prefill`]) at any chunk count up to `max_chunks`:
    /// the tree's partials for `rows x max_chunks`, and the flash path's for the splits of that many chunks.
    pub fn new(rows: usize, max_chunks: usize) -> Result<Self, CudaError> {
        Self::build(rows, max_chunks, max_chunks)
    }

    /// The model's scratch for a context of `max_chunks` tree chunks (`ceil(max_context / 512)`): the tree's partials for at most
    /// [`TREE_CHUNKS_BELOW`] chunks a row, the flash path's for `flash_splits(max_chunks)` splits a row. `rows` is the widest pass:
    /// [`crate::nemotron_attn_prefill::attn_long`] runs a call of more rows in passes of that many.
    pub fn new_serving(rows: usize, max_chunks: usize) -> Result<Self, CudaError> {
        Self::build(rows, max_chunks, max_chunks.min(TREE_CHUNKS_BELOW))
    }

    fn build(rows: usize, context_chunks: usize, tree_chunks: usize) -> Result<Self, CudaError> {
        assert!(
            rows > 0 && context_chunks > 0 && context_chunks <= MAX_CHUNKS && tree_chunks > 0,
            "scratch of {rows} rows x {context_chunks} chunks"
        );
        let f_splits = flash_splits(context_chunks);
        let tree = rows * tree_chunks * N_Q_HEADS;
        let flash = rows * f_splits * N_Q_HEADS;
        Ok(Self {
            part_ml: DevBuf::alloc(tree * 8)?,
            part_o: DevBuf::alloc(tree * HEAD_DIM * 4)?,
            rows,
            max_chunks: tree_chunks,
            flash_ml: DevBuf::alloc(flash * 8)?,
            flash_o: DevBuf::alloc(flash * HEAD_DIM * 4)?,
            flash_plan: DevBuf::zeroed(Self::plan_bytes(rows))?,
            flash_splits: f_splits,
        })
    }

    /// Row capacity this scratch was built for (the widest pass).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The tree's chunk capacity per row: the stride of its partials.
    pub fn max_chunks(&self) -> usize {
        self.max_chunks
    }

    /// The flash path's split capacity per row.
    pub fn flash_splits(&self) -> usize {
        self.flash_splits
    }

    /// Bytes of the flash path's row plan for `rows` rows: a count and a pair per row block (at most one per row), 8 bytes each.
    pub const fn plan_bytes(rows: usize) -> usize {
        (rows + 1) * 8
    }

    /// Bytes of the tree's partial buffers for `rows` rows of up to `chunks` chunks.
    pub const fn tree_bytes(rows: usize, chunks: usize) -> usize {
        rows * chunks * GROUP_BYTES
    }

    /// Bytes of the flash path's buffers for `rows` rows of up to `splits` splits: the partials and the row plan.
    pub const fn flash_bytes(rows: usize, splits: usize) -> usize {
        rows * splits * GROUP_BYTES + Self::plan_bytes(rows)
    }

    /// Bytes of a scratch built by [`Scratch::new`]: the arithmetic of the allocation.
    pub const fn bytes_for(rows: usize, max_chunks: usize) -> usize {
        Self::tree_bytes(rows, max_chunks) + Self::flash_bytes(rows, flash_splits(max_chunks))
    }

    /// Bytes of a scratch built by [`Scratch::new_serving`], which the memory model (`nemotron_h_omni::memory`) prices.
    pub const fn serving_bytes(rows: usize, max_chunks: usize) -> usize {
        let tree = if max_chunks < TREE_CHUNKS_BELOW { max_chunks } else { TREE_CHUNKS_BELOW };
        Self::tree_bytes(rows, tree) + Self::flash_bytes(rows, flash_splits(max_chunks))
    }

    /// Bytes the scratch holds on the device: the sum of its five buffers (what the allocation is, as against
    /// [`Scratch::bytes_for`] and [`Scratch::serving_bytes`], the arithmetic the plan prices).
    pub fn allocated_bytes(&self) -> usize {
        [&self.part_ml, &self.part_o, &self.flash_ml, &self.flash_o, &self.flash_plan].iter().map(|b| b.bytes()).sum()
    }

    /// The flash path's `(m, l)`, `o` and row-plan buffers.
    pub(crate) fn flash_buffers(&mut self) -> (&mut DevBuf, &mut DevBuf, &mut DevBuf) {
        (&mut self.flash_ml, &mut self.flash_o, &mut self.flash_plan)
    }

    /// Fills the partials (both paths') with a NaN bit pattern, queued on `st`. For tests: a merge that reads a chunk the partial launch
    /// did not write cannot then pass by finding an earlier step's values.
    pub fn poison(&mut self, st: &Stream) -> Result<(), CudaError> {
        self.part_ml.memset_async(0xFF, st)?;
        self.part_o.memset_async(0xFF, st)?;
        self.flash_ml.memset_async(0xFF, st)?;
        self.flash_o.memset_async(0xFF, st)
    }
}

fn need(b: &DevBuf, bytes: usize, what: &str) {
    assert!(b.bytes() >= bytes, "nemotron_attn {what}: buffer of {} bytes, need {bytes}", b.bytes());
    assert!(b.ptr() as usize % 16 == 0, "nemotron_attn {what}: buffer is not 16-byte aligned");
}

fn i(v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| CudaError(1))
}

/// Scatters the rows' K and V (`k` and `v`: fp32 `[rows][256]`, kv heads concatenated) into their pages, rounding each once to fp16
/// (or, for an int8 cache, to int8 at the (token, kv head)'s power-of-two scale: [`kv8`]).
/// The block for each row's position must already be in the table.
pub fn kv_write(
    kv: &mut NemoKv,
    table: &BlockTable,
    idx: &RowIndex,
    k: &DevBuf,
    v: &DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    need(k, idx.rows * KV_TOKEN_ELEMS * 4, "k");
    need(v, idx.rows * KV_TOKEN_ELEMS * 4, "v");
    assert!(kv.k.bytes() >= kv.n_blocks * kv.dtype.page_bytes(), "nemotron_attn kv_write: a cache smaller than its pages");
    let launch = match kv.dtype {
        KvDtype::F16 => ffi::gb_nemo_kv_write,
        KvDtype::Int8 => ffi::gb_nemo_kv_write_i8,
    };
    CudaError::check(unsafe {
        launch(
            k.ptr(),
            v.ptr(),
            idx.slot.ptr(),
            idx.pos.ptr(),
            table.buf.ptr(),
            i(table.max_blocks)?,
            kv.k.ptr(),
            kv.v.ptr(),
            i(idx.rows)?,
            st.raw(),
        )
    })
}

fn attn(
    q: &DevBuf,
    kv: &NemoKv,
    table: &BlockTable,
    idx: &RowIndex,
    scratch: &mut Scratch,
    out: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    attn_range(q, kv, table, idx, 0, idx.rows, idx.max_chunks(), i32::MAX, scratch, out, st)
}

/// The tree over rows `first..first + rows` of `idx` (their queries and outputs are the same rows of `q` and `out`), the launch made
/// for `max_chunks` chunks, skipping the rows whose position is `limit` or more (another arithmetic's: `i32::MAX` skips none).
/// `first = 0, rows = idx.rows, limit = i32::MAX` is exactly the launch [`attn_decode`] and [`attn_prefill`] make. A row's bits do not
/// depend on which rows it is launched with, so a call may be cut into ranges.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_range(
    q: &DevBuf,
    kv: &NemoKv,
    table: &BlockTable,
    idx: &RowIndex,
    first: usize,
    rows: usize,
    max_chunks: usize,
    limit: i32,
    scratch: &mut Scratch,
    out: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    assert!(first + rows <= idx.rows, "nemotron_attn rows {first}..{} of an index of {}", first + rows, idx.rows);
    assert!(
        scratch.rows >= rows && scratch.max_chunks >= max_chunks,
        "nemotron_attn scratch too small for {rows} rows x {max_chunks}"
    );
    assert!(table.slots > 0 && kv.n_blocks > 0, "nemotron_attn needs a non-empty cache and block table");
    need(q, (first + rows) * Q_ROW_ELEMS * 4, "q");
    need(out, (first + rows) * Q_ROW_ELEMS * 4, "out");
    let (qp, op) = (q.ptr_at(first * Q_ROW_ELEMS * 4), out.ptr_at(first * Q_ROW_ELEMS * 4));
    let (sp, pp) = (idx.slot.ptr_at(first * 4), idx.pos.ptr_at(first * 4));
    let partial = match kv.dtype {
        KvDtype::F16 => ffi::gb_nemo_attn_partial,
        KvDtype::Int8 => ffi::gb_nemo_attn_partial_i8,
    };
    CudaError::check(unsafe {
        partial(
            qp,
            kv.k.ptr(),
            kv.v.ptr(),
            table.buf.ptr(),
            i(table.max_blocks)?,
            sp,
            pp,
            i(rows)?,
            i(max_chunks)?,
            limit,
            i(scratch.max_chunks)?,
            scratch.part_ml.ptr(),
            scratch.part_o.ptr(),
            st.raw(),
        )
    })?;
    CudaError::check(unsafe {
        ffi::gb_nemo_attn_merge(
            pp,
            scratch.part_ml.ptr(),
            scratch.part_o.ptr(),
            i(rows)?,
            i(scratch.max_chunks)?,
            limit,
            op,
            st.raw(),
        )
    })
}

/// Decode, or a verify window of up to [`MAX_DECODE_ROWS`] rows. `q` and `out` are fp32 `[rows][32][128]`. The rows' K and V must
/// already be in the cache ([`kv_write`]). Launches on `st` and returns without synchronising.
pub fn attn_decode(
    q: &DevBuf,
    kv: &NemoKv,
    table: &BlockTable,
    idx: &RowIndex,
    scratch: &mut Scratch,
    out: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    assert!(idx.rows <= MAX_DECODE_ROWS, "attn_decode takes at most {MAX_DECODE_ROWS} rows; use attn_prefill");
    attn(q, kv, table, idx, scratch, out, st)
}

/// Prefill: any number of rows, the same kernels as [`attn_decode`]. Launches on `st` and returns without synchronising.
pub fn attn_prefill(
    q: &DevBuf,
    kv: &NemoKv,
    table: &BlockTable,
    idx: &RowIndex,
    scratch: &mut Scratch,
    out: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    attn(q, kv, table, idx, scratch, out, st)
}
