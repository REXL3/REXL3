//! The DSA (DeepSeek sparse attention) indexer of GLM-5.3-Flash: index-key LayerNorm and raw-row ring, pooled-key plane, fp16 scoring and
//! the exact top-512 pool selection with the raw tail, for the 11 MLA layers and the MTP layer. Kernels: `csrc/dsa_index.cu`,
//! `csrc/dsa_score.cu`, `csrc/dsa_select.cu` (geometry and the regime rule in `csrc/dsa.cuh`). Oracle: the ExLlamaV3 fork's `k-pool`
//! path (`tools/dsa_oracle.py`). Facts and semantics are in the header of `dsa.cuh`; the contract is in the section below.
//!
//! # The contract
//!
//! Every call takes per-row `(slot, absolute position)` ([`RowIndex`]); the rows of one window are consecutive positions of one slot in
//! order (windows of different slots may share a call), at most [`MAX_ROWS`] rows per call (longer prefill chunks are fed in slabs; the
//! kernels are row-invariant so slabbing changes no bit).
//!
//! * A row's regime depends only on its own context: [`is_sparse`]`(pos) = pos + 1 > 2048`. A dense row needs no selection (the dense
//!   attention reads every key). The regime is NOT decided per window: in a verify window straddling 2,048 the first rows are dense
//!   and the later ones sparse, each exactly as if it were decoded alone.
//! * [`append`] runs for every row of every layer in every regime (the planes must be complete before the context first passes 2,048):
//!   `k_raw` (`wk` output, fp16 `[rows][128]`) is LayerNorm'ed, the row is stored with its `gate` in the ring, and a row at position
//!   `3 mod 4` completes pool `pos / 4` in the pool plane (recomputed from its four members whether or not it was written before, which
//!   makes rollback of rejected draft rows and pools straddling calls come out right without any bookkeeping).
//! * [`select`] scores the pooled keys (`q` fp16 `[rows][32][128]`, no rope in GLM-5.3: `qk_rope_head_dim` is 0; `w` fp16 `[rows][32]`
//!   the raw `weights_proj` output) and selects. Its result is a [`Selection`]: for row `r`, `meta[2r] == 1` (sparse) means
//!   `indices[r][0 .. meta[2r+1])` are the attended token positions, ascending, absolute (the 512 selected pools' 4 tokens each, then the
//!   0..3 raw tokens of the incomplete last pool; the count is 2048 + `(pos + 1) % 4`), and `-1` after them; `meta[2r] == 0` means the row
//!   is dense and `indices[r]` is not written. The sparse attention gather reads exactly this. A padding row (position -1, as a
//!   CUDA-graph bucket pads a call) is inert in every stage and reads as dense.
//! * One [`Scratch`] serves every layer of a model (the layers run in sequence on one stream and the gather reads the selection before the
//!   next layer's [`select`] overwrites it): 18 MB at 262,144 tokens. `n_pools_max` is an upper bound of `row_pools(pos)` over the call's
//!   rows (it sizes the grid; a graph-captured caller passes the capacity and the kernels skip what a row does not need).
//! * Cost at 262,144 tokens, per layer, one row: scoring reads the 16.8 MB pool plane (about 80 us, a streaming read of the same bytes takes
//!   61-68 us) and the selection takes about 45 us; windows of K+1 rows add about 4-6 us of tensor work per row. See the benchmark
//!   `bench_dsa_indexer` in `tests/dsa.rs`.
//!
//! # Planes
//!
//! * **Pool plane** `f16 [n_blocks][64][128]` ([`Planes::pool`]): pool `p` of table row `s` lives at block `table[s][p / 64]`, row
//!   `p % 64`. A block is one 256-token page: the same block ids as the latent cache's block table. 128 f16 per 4 tokens = 64 bytes per
//!   token per layer: the bytes scoring must read (16.8 MB per layer at 262,144 tokens).
//! * **Raw-row ring** `f16 [slots][ring_len][256]` ([`Planes::ring`]), `[LayerNorm(k) | gate]` of the last `ring_len` tokens of each
//!   slot, token `t` at row `t % ring_len`, `ring_len >= `[`RING_MIN`]` (a full window plus a pool), a multiple of 4. This replaces the
//!   oracle's per-token plane (256 f16 per token, 128 MiB per layer at 262,144 tokens, 1.4 GiB for 11 layers): the raw rows are read once,
//!   when their pool completes, and never again, so only the rows of the incomplete pool and of the open window need to live anywhere.
//!   Setting `ring_len` to the context capacity makes it a full plane of every token's row per slot (what the oracle keeps, but indexed by
//!   slot, not by block; the tests run both). The ring belongs to the slot, not to the blocks: a sequence that adopts a cached prefix
//!   whose length is not a multiple of 4, or moves to another slot, needs the ring rows of its incomplete pool copied (`len % 4` rows of each indexer layer,
//!   positions `len - len % 4 .. len`: the GLM prefix cache does it, `families/glm/model/snapshot.rs`; a prefix whose length is a multiple of 4, a
//!   page-aligned one included, needs nothing). The memory model is therefore a fixed `ring_len x 512` bytes per slot per layer instead of 512
//!   bytes per token per layer.

use crate::attention::{BlockTable, RowIndex};
use crate::rt::{CudaError, DevBuf, Stream};

/// `index_head_dim`.
pub const D: usize = 128;
/// `index_n_heads`.
pub const H: usize = 32;
/// `index_kpool`: tokens per pool.
pub const P: usize = 4;
/// `index_topk`: tokens attended per sparse row.
pub const TOPK: usize = 2048;
/// Pools selected per sparse row.
pub const SEL_POOLS: usize = TOPK / P;
/// Tokens per cache page.
pub const PAGE_TOKENS: usize = 256;
/// Pools per cache page.
pub const POOLS_PER_PAGE: usize = PAGE_TOKENS / P;
/// Largest selection: 512 pools of 4 tokens plus 3 tail tokens.
pub const K_OUT: usize = TOPK + P - 1;
/// Row stride of [`Selection::indices`] in elements.
pub const K_STRIDE: usize = 2056;
/// Rows per launch.
pub const MAX_ROWS: usize = 128;
/// Smallest ring length.
pub const RING_MIN: usize = MAX_ROWS + P;
const _: () = assert!(K_STRIDE >= K_OUT && K_STRIDE % 4 == 0);
/// The indexer-key LayerNorm epsilon of GLM-5.3-Flash (`MLAttention`'s `index_norm_eps` default, not overridden by the architecture).
pub const LN_EPS: f32 = 1e-6;

/// Whether a row at absolute position `pos` selects (its own context `pos + 1` exceeds [`TOPK`]); otherwise it is dense.
pub fn is_sparse(pos: usize) -> bool {
    pos + 1 > TOPK
}
/// Complete pools visible to a row at `pos`.
pub fn row_pools(pos: usize) -> usize {
    (pos + 1) / P
}
/// Raw tail tokens of a row at `pos` (the incomplete last pool).
pub fn row_tail(pos: usize) -> usize {
    (pos + 1) % P
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_dsa_norm_ring(
            k_raw: *const c_void,
            gate: *const c_void,
            ln_w: *const c_void,
            ln_b: *const c_void,
            eps: f32,
            row_slot: *const i32,
            row_pos: *const i32,
            rows: i32,
            kg: *mut c_void,
            ring: *mut c_void,
            ring_len: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_dsa_pool(
            kg: *const c_void,
            ring: *const c_void,
            ring_len: i32,
            ape: *const f32,
            row_slot: *const i32,
            row_pos: *const i32,
            rows: i32,
            pool: *mut c_void,
            block_table: *const i32,
            max_blocks: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_dsa_score(
            q: *const c_void,
            w: *const c_void,
            w_stride: i32,
            pool: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            rows: i32,
            n_pools_max: i32,
            s_stride: i32,
            scores: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_dsa_score_pools_per_block() -> i32;
        pub fn gb_dsa_select(
            scores: *const c_void,
            s_stride: i32,
            row_pos: *const i32,
            rows: i32,
            out: *mut i32,
            meta: *mut i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// One indexer layer's small weights: the key LayerNorm and the in-pool position embedding.
#[derive(Debug)]
pub struct IndexerWeights {
    /// `indexer.k_norm.weight`, fp16 `[128]`.
    pub ln_w: DevBuf,
    /// `indexer.k_norm.bias`, fp16 `[128]`.
    pub ln_b: DevBuf,
    /// `indexer.index_kpool_compress_ape`, fp32 `[4][128]`.
    pub ape: DevBuf,
    pub eps: f32,
}

impl IndexerWeights {
    pub fn new(ln_w_f16: &[u16], ln_b_f16: &[u16], ape: &[f32], eps: f32) -> Result<Self, CudaError> {
        assert!(ln_w_f16.len() == D && ln_b_f16.len() == D && ape.len() == P * D);
        Ok(Self { ln_w: DevBuf::from_slice(ln_w_f16)?, ln_b: DevBuf::from_slice(ln_b_f16)?, ape: DevBuf::from_slice(ape)?, eps })
    }
}

/// One indexer layer's planes (see the module docs).
#[derive(Debug)]
pub struct Planes {
    pub pool: DevBuf,
    pub ring: DevBuf,
    pub n_blocks: usize,
    pub slots: usize,
    pub ring_len: usize,
}

impl Planes {
    /// Bytes of the pool plane for `n_blocks` pages.
    pub fn pool_bytes(n_blocks: usize) -> usize {
        n_blocks * POOLS_PER_PAGE * D * 2
    }
    /// Bytes of the ring for `slots` slots.
    pub fn ring_bytes(slots: usize, ring_len: usize) -> usize {
        slots * ring_len * 2 * D * 2
    }
    pub fn new(n_blocks: usize, slots: usize, ring_len: usize) -> Result<Self, CudaError> {
        assert!(ring_len >= RING_MIN && ring_len % P == 0, "ring_len must be >= {RING_MIN} and a multiple of {P}");
        Ok(Self {
            pool: DevBuf::zeroed(Self::pool_bytes(n_blocks))?,
            ring: DevBuf::zeroed(Self::ring_bytes(slots, ring_len))?,
            n_blocks,
            slots,
            ring_len,
        })
    }
}

/// Per-call buffers: the staging rows, the score matrix and the selection. Sized for contexts up to `max_context` tokens.
#[derive(Debug)]
pub struct Scratch {
    /// `[MAX_ROWS][256]` f16: `[LayerNorm(k) | gate]` of the call's rows.
    pub kg: DevBuf,
    /// `[MAX_ROWS][s_stride]` f16.
    pub scores: DevBuf,
    /// `[MAX_ROWS][K_STRIDE]` i32.
    pub indices: DevBuf,
    /// `[MAX_ROWS][2]` i32: `(sparse, count)`.
    pub meta: DevBuf,
    pub s_stride: usize,
    pub max_context: usize,
}

/// A row-major view of a finished selection (see the module docs for the contract). The MLA module's gather input (`mla::SparseSel`:
/// `indices`, `meta`, `stride`) has the same three fields and reads exactly this.
#[derive(Debug)]
pub struct Selection<'a> {
    pub indices: &'a DevBuf,
    pub meta: &'a DevBuf,
    /// Elements between rows of `indices`.
    pub stride: usize,
}

/// Score columns a launch needs for contexts up to `max_context` tokens (pools, rounded up to the scoring block).
pub fn score_stride(max_context: usize) -> usize {
    let blk = unsafe { ffi::gb_dsa_score_pools_per_block() }.max(128) as usize;
    max_context.div_ceil(P).div_ceil(blk) * blk
}

impl Scratch {
    /// Bytes of GPU memory [`Scratch::new`] holds for contexts up to `max_context` tokens, computed without allocating.
    pub fn bytes_for(max_context: usize) -> usize {
        MAX_ROWS * 2 * D * 2 + MAX_ROWS * score_stride(max_context) * 2 + MAX_ROWS * K_STRIDE * 4 + MAX_ROWS * 2 * 4
    }

    pub fn new(max_context: usize) -> Result<Self, CudaError> {
        let s_stride = score_stride(max_context);
        Ok(Self {
            kg: DevBuf::zeroed(MAX_ROWS * 2 * D * 2)?,
            scores: DevBuf::zeroed(MAX_ROWS * s_stride * 2)?,
            indices: DevBuf::zeroed(MAX_ROWS * K_STRIDE * 4)?,
            meta: DevBuf::zeroed(MAX_ROWS * 2 * 4)?,
            s_stride,
            max_context,
        })
    }
    /// Bytes of GPU memory this scratch occupies (about 18 MB at 262,144 tokens).
    pub fn bytes(&self) -> usize {
        self.kg.bytes() + self.scores.bytes() + self.indices.bytes() + self.meta.bytes()
    }
    pub fn selection(&self) -> Selection<'_> {
        Selection { indices: &self.indices, meta: &self.meta, stride: K_STRIDE }
    }
}

fn check_rows(n_rows: usize, rows: &RowIndex) {
    assert!((1..=MAX_ROWS).contains(&n_rows) && n_rows <= rows.rows, "1..={MAX_ROWS} rows per call, got {n_rows}");
}

/// The key-side update for `n_rows` rows: LayerNorm of `k_raw` (fp16 `[n_rows][128]`) with `gate` (fp16 `[n_rows][128]`) into the ring and
/// the staging rows ([`norm_ring`]), then every pool the rows complete ([`pool_update`]). Same stream, in this order.
#[allow(clippy::too_many_arguments)]
pub fn append(
    w: &IndexerWeights,
    planes: &mut Planes,
    scratch: &mut Scratch,
    k_raw: &DevBuf,
    gate: &DevBuf,
    rows: &RowIndex,
    n_rows: usize,
    table: &BlockTable,
    stream: &Stream,
) -> Result<(), CudaError> {
    norm_ring(w, planes, scratch, k_raw, gate, rows, n_rows, stream)?;
    pool_update(w, planes, scratch, rows, n_rows, table, stream)
}

/// Stage 1 of [`append`]: LayerNorm + ring write. The staging rows are `scratch.kg`.
#[allow(clippy::too_many_arguments)]
pub fn norm_ring(
    w: &IndexerWeights,
    planes: &mut Planes,
    scratch: &mut Scratch,
    k_raw: &DevBuf,
    gate: &DevBuf,
    rows: &RowIndex,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    check_rows(n_rows, rows);
    assert!(k_raw.bytes() >= n_rows * D * 2 && gate.bytes() >= n_rows * D * 2);
    assert!(planes.slots >= 1 && planes.ring.bytes() == Planes::ring_bytes(planes.slots, planes.ring_len));
    let rc = unsafe {
        ffi::gb_dsa_norm_ring(
            k_raw.ptr(),
            gate.ptr(),
            w.ln_w.ptr(),
            w.ln_b.ptr(),
            w.eps,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            n_rows as i32,
            scratch.kg.ptr(),
            planes.ring.ptr(),
            planes.ring_len as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Stage 2 of [`append`]: complete the pools whose last member is one of the rows. Must follow [`norm_ring`] of the same rows.
pub fn pool_update(
    w: &IndexerWeights,
    planes: &mut Planes,
    scratch: &Scratch,
    rows: &RowIndex,
    n_rows: usize,
    table: &BlockTable,
    stream: &Stream,
) -> Result<(), CudaError> {
    check_rows(n_rows, rows);
    let rc = unsafe {
        ffi::gb_dsa_pool(
            scratch.kg.ptr(),
            planes.ring.ptr(),
            planes.ring_len as i32,
            w.ape.ptr() as *const f32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            n_rows as i32,
            planes.pool.ptr(),
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Scores for `n_rows` rows into `scratch.scores` (fp16 `[n_rows][s_stride]`; only columns `< row_pools(pos)` of sparse rows are defined).
/// `n_pools_max` bounds `row_pools(pos)` over the rows (it sizes the grid; a graph-captured caller passes the capacity).
#[allow(clippy::too_many_arguments)]
pub fn score(
    planes: &Planes,
    scratch: &mut Scratch,
    q: &DevBuf,
    w: &DevBuf,
    rows: &RowIndex,
    n_rows: usize,
    n_pools_max: usize,
    table: &BlockTable,
    stream: &Stream,
) -> Result<(), CudaError> {
    check_rows(n_rows, rows);
    assert!(q.bytes() >= n_rows * H * D * 2 && w.bytes() >= n_rows * H * 2);
    assert!(n_pools_max <= scratch.s_stride, "n_pools_max {n_pools_max} exceeds the scratch ({} pools)", scratch.s_stride);
    let rc = unsafe {
        ffi::gb_dsa_score(
            q.ptr(),
            w.ptr(),
            H as i32,
            planes.pool.ptr(),
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            n_rows as i32,
            n_pools_max as i32,
            scratch.s_stride as i32,
            scratch.scores.ptr(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Exact top-512 selection with the tail from `scores` (`[n_rows][s_stride]` f16, laid out as [`score`] writes it) into
/// `indices` (`[n_rows][K_STRIDE]` i32) and `meta` (`[n_rows][2]` i32). Exposed separately so the selection can be tested on arbitrary
/// score matrices.
#[allow(clippy::too_many_arguments)]
pub fn select_scores(
    scores: &DevBuf,
    s_stride: usize,
    rows: &RowIndex,
    n_rows: usize,
    indices: &mut DevBuf,
    meta: &mut DevBuf,
    stream: &Stream,
) -> Result<(), CudaError> {
    check_rows(n_rows, rows);
    assert!(s_stride % 8 == 0 && scores.bytes() >= n_rows * s_stride * 2);
    assert!(indices.bytes() >= n_rows * K_STRIDE * 4 && meta.bytes() >= n_rows * 8);
    let rc = unsafe {
        ffi::gb_dsa_select(
            scores.ptr(),
            s_stride as i32,
            rows.pos.ptr() as *const i32,
            n_rows as i32,
            indices.ptr() as *mut i32,
            meta.ptr() as *mut i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Scoring and selection for `n_rows` rows (see the module docs). The result is `scratch.selection()`.
#[allow(clippy::too_many_arguments)]
pub fn select(
    planes: &Planes,
    scratch: &mut Scratch,
    q: &DevBuf,
    w: &DevBuf,
    rows: &RowIndex,
    n_rows: usize,
    n_pools_max: usize,
    table: &BlockTable,
    stream: &Stream,
) -> Result<(), CudaError> {
    score(planes, scratch, q, w, rows, n_rows, n_pools_max, table, stream)?;
    let s_stride = scratch.s_stride;
    // split borrows: the scores are read, the selection written
    let Scratch { scores, indices, meta, .. } = scratch;
    select_scores(scores, s_stride, rows, n_rows, indices, meta, stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regime_rule() {
        // a row sees pos + 1 keys: 2048 keys (pos 2047) is still dense, the 2049th key (pos 2048) makes a row sparse
        assert!(!is_sparse(0) && !is_sparse(2046) && !is_sparse(2047) && is_sparse(2048));
        assert_eq!(row_pools(2047), 512);
        assert_eq!(row_tail(2047), 0);
        assert_eq!(row_tail(2048), 1);
        assert_eq!(K_OUT, SEL_POOLS * P + P - 1);
    }
}
