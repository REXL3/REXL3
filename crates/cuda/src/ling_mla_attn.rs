//! Ling-3.1-flash-1019 Gated-MLA under the LP2 tensor-core arithmetic contract (`csrc/ling_mla_attn.cu`):
//! the absorb (`q_nope @ W_UK`), the dense causal absorbed attention over the 576-wide cache rows, and the
//! expand (`o_lat @ W_UV`, optionally fused with the head-wise sigmoid `g_proj` output gate) as BF16
//! `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` kernels, in place of [`crate::mla_gated`]'s scalar
//! per-`(row, head)` `attend_kernel` and skinny absorb / expand.
//!
//! # Geometry (Ling-3.1, unchanged from [`crate::mla_gated`])
//!
//! 64 heads, `qk_nope_head_dim = v_head_dim = 128`, `qk_rope_head_dim = 64`, `kv_lora_rank = 512`, cache row
//! 576 bf16 (512 normalised latent + 64 rotated rope), softmax scale `192^-0.5`, full causal
//! (`key_pos <= query_pos`; `row_pos < 0` is a padding row). The paged cache, the block table, the packed
//! `kv_b_proj` (`uk`, `uvt`) and `g_proj` are [`crate::mla_gated`]'s — the same weights serve both contracts.
//!
//! # The LP2 arithmetic contract
//!
//! * (a) A row's output bits are a function of that row's operands, its slot's keys `0..=pos`, its own
//!   position and the packed weights only — not of the row count, the chunk grid, other slots, or the block
//!   placement. Absorb/expand tile rows in 16-row MMA groups but the M dimension of an MMA is independent
//!   rows; the attention CTA walks one row's keys in ascending tile order with one online-softmax state.
//! * (b) Every contraction is in a fixed order: absorb d = 0..127 (one fold block), expand c = 0..511 in
//!   128-c fold blocks (GLM P2's fold128), attention per 32-key tile with the latent d-quarters summed in
//!   the fixed order 0..3, then the rope part, then online softmax in key order.
//! * (c) BF16 operands, FP32 accumulation everywhere (`mma` accumulators, softmax state, the gate dot).
//! * (d) Deterministic run to run: no atomics. The prefill / unpaired kernels (`attn_pair_kernel`, `attn_kernel`) have no split-K and
//!   no workspace; the decode / verify / draft path ([`attend_split`]) splits a row's keys into ABSOLUTE 1,024-key chunks and merges
//!   them in ascending chunk order from a fixed-size workspace ([`SPLIT_WS_BYTES`]), see "Split-key decode attention" below.
//!
//! P2 is **not** bit-equal to P1 ([`crate::mla_gated`]): the contraction orders differ by design. Qualification
//! is f64-oracle bounded error + row/chunk/determinism tests (`crates/cuda/tests/ling_mla_attn.rs`).
//!
//! # Split-key decode attention ([`ARITH_ATTN`] v2)
//!
//! A decode step is one row (a verify window two to four) over up to the whole context: the single-CTA-per-head-group walk keeps 2 of
//! the GB10's 48 SMs busy and reads the cache at ~19 GB/s. [`attend_split`] cuts each row's keys `0..=pos` into chunks of
//! [`SPLIT_KEYS`] keys at multiples of [`SPLIT_KEYS`] from key 0, one CTA per (row, 32-head group, chunk). A chunk is the same per-tile
//! arithmetic as the walk, started from the zero softmax state; the chunks' `(m, l, o)` partials are merged in ascending chunk order
//! (`L = l_0 w_0`, then `fma(l_c, w_c, L)`; the same for `o`; `w_c = 2^(m_c - M)`, `M` the row's maximum) and rounded once to bf16.
//! What that buys, and what the tests pin (`tests/ling_mla_split.rs`):
//!
//! * a row's bits are a function of its q, its slot's keys `0..=pos` and `pos` alone: the same bits as a 1-row call when the row is one of
//!   2-64, whatever the other rows' positions or slots, and whatever the grid (the chunk grid spans the block table, a chunk past the
//!   row's last key exits) or the SM count: chunk boundaries are absolute and the merge order is fixed;
//! * a row with `pos < SPLIT_KEYS` has one chunk and equals the walk kernel (`attend`) bit for bit;
//! * the prefill kernels are unchanged, so a prefill row and a decode row of the same position agree to the bf16 rounding of the
//!   re-association, not bit for bit (the engine never needs them to: prefix snapshots are cut at prompt-time prefill boundaries).
//!
//! The selection is per call site (the model's `PrefillContract`), never an environment variable: P1 stays
//! [`crate::mla_gated::mla_gated_rows`]; P2 calls [`mla_gated_rows_tc`]. `kv_prep` and `q_rot` (RoPE) are the
//! shared scalar kernels of `mla_gated.cu` under both contracts — the rope rotation and the RMSNorm are not
//! contractions, so there is nothing to gain from tensor cores there.

use crate::attention::{BlockTable, RowIndex};
use crate::exl3_linear::Dtype;
use crate::mla_gated::{self, MlaGatedCache, MlaGatedWeights};
use crate::rt::{CudaError, DevBuf, Stream};
use core::ffi::c_void;

/// The arithmetic id of the LP2 absorb / expand (the prefix namespace's and fingerprint's `mla-ae=` under P2).
/// Bump with any change that moves a bit of `gb_ling_mla_absorb_tc` / `gb_ling_mla_expand_tc` /
/// `gb_ling_mla_expand_gate_tc` (tiling, fold order, gate tree).
pub const ARITH_AE: &str = "ling-mla-ae-fold128-bf16-v1";

/// The arithmetic id of the LP2 absorbed attention (`gb_ling_mla_attend_tc`: 32-key tiles, d-quarter order,
/// rope-last score chain, log2-domain online softmax). Part of the P2 prefix namespace and fingerprint.
///
/// `v1`: one online-softmax walk per (row, head group) for every call. `v2` (this id): the prefill kernels are `v1`'s walk unchanged; the
/// decode / verify / draft kernel is the split-key one ([`attend_split`]: absolute 1,024-key chunks, ascending-order merge). A change to the
/// chunk length, the merge formula or the fold of a chunk's tiles is a new id.
pub const ARITH_ATTN: &str = "ling-mla-attn-tc32-bf16-v2-decode-split1024";

/// Keys per chunk of the split-key decode attention: chunk `c` holds keys `c * SPLIT_KEYS .. (c + 1) * SPLIT_KEYS` of a row (named by
/// [`ARITH_ATTN`]; `csrc/ling_mla_attn.cu`'s `SPLIT_KEYS`, checked on a GPU build by `split_constants_match_the_kernel`).
pub const SPLIT_KEYS: usize = 1024;

/// (row, chunk) partial slots of the split path's workspace: one slot is `64 * (512 * 4 + 8)` bytes (the fp32 unnormalised `o` of 64 heads and
/// their `(m, l)`). A call is launched in row batches of `SPLIT_WS_SLOTS / chunks` rows, so the size does not grow with the rows or the
/// context; a block table wider than `SPLIT_WS_SLOTS * SPLIT_KEYS` tokens (327,680) is refused.
pub const SPLIT_WS_SLOTS: usize = 320;

/// Bytes of one (row, chunk) partial slot.
pub const SPLIT_SLOT_BYTES: usize = mla_gated::N_HEADS * (mla_gated::LATENT_DIM * 4 + 8);

/// The fused combine's per-(row, 32-head group) counters, after the parts (2 ints per slot row; the per-pass row
/// count never exceeds `SPLIT_WS_SLOTS / chunks`).
pub const SPLIT_CNT_BYTES: usize = 2 * SPLIT_WS_SLOTS * 4;

/// The workspace [`attend_split`] needs (allocate once, reuse for every layer of a step; the memory plan counts it).
pub const SPLIT_WS_BYTES: usize = SPLIT_WS_SLOTS * SPLIT_SLOT_BYTES + SPLIT_CNT_BYTES;

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_ling_mla_absorb_tc(
            q: *const c_void,
            q_row_stride: i64,
            w_uk: *const c_void,
            q_lat: *mut c_void,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_ling_mla_prep_absorb_rope_tc(
            q: *const c_void,
            q_row_stride: i64,
            kv: *const c_void,
            kv_stride: i32,
            norm_w: *const c_void,
            cos: *const c_void,
            sin: *const c_void,
            table_stride: i32,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            w_uk: *const c_void,
            q_lat: *mut c_void,
            cache: *mut c_void,
            q_rot: *mut c_void,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_ling_mla_attend_tc(
            q_lat: *const c_void,
            q_rot: *const c_void,
            cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            out: *mut c_void,
            rows: i32,
            pair: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_ling_mla_attend_split_tc(
            q_lat: *const c_void,
            q_rot: *const c_void,
            cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            out: *mut c_void,
            rows: i32,
            ws: *mut c_void,
            ws_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        /// Test-only: the pre-fusion two-launch split+combine sequence (the A/B bit test pins the fused
        /// `gb_ling_mla_attend_split_tc` against it).
        pub fn gb_ling_mla_attend_split_ref_tc(
            q_lat: *const c_void,
            q_rot: *const c_void,
            cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            out: *mut c_void,
            rows: i32,
            ws: *mut c_void,
            ws_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_ling_mla_expand_tc(
            o_lat: *const c_void,
            w_uvt: *const c_void,
            out: *mut c_void,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_ling_mla_expand_gate_tc(
            o_lat: *const c_void,
            w_uvt: *const c_void,
            x: *const c_void,
            gproj: *const c_void,
            out: *mut c_void,
            rows: i32,
            dense_suh: *const c_void,
            dense_xh: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
    }
}

fn check_bf16(w: &MlaGatedWeights) {
    assert!(
        w.dtype == Dtype::BF16,
        "the LP2 tensor-core MLA ({ARITH_AE} / {ARITH_ATTN}) is bf16 only: Ling-3.1 binds kv_b and the MLA \
         activations as bf16 (got {:?}); the fp16 instantiation of the scalar path is mla_gated's",
        w.dtype
    );
}

/// Tensor-core absorb: `q_lat[r][h][c] = sum_d q[r][h*192 + d] * W_UK[h][d][c]` for `n_rows` rows of all 64
/// heads, same layouts as [`crate::mla_gated::q_prep`]'s absorb half. bf16 only.
pub fn absorb(q: &DevBuf, w: &MlaGatedWeights, q_lat: &mut DevBuf, n_rows: usize, stream: &Stream) -> Result<(), CudaError> {
    check_bf16(w);
    let qs = mla_gated::N_HEADS * mla_gated::QK_HEAD_DIM;
    assert!(q.bytes() >= n_rows * qs * 2, "q rows must hold 64 x 192 bf16");
    assert!(q_lat.bytes() >= n_rows * mla_gated::N_HEADS * mla_gated::LATENT_DIM * 2);
    let rc = unsafe { ffi::gb_ling_mla_absorb_tc(q.ptr(), qs as i64, w.uk.ptr(), q_lat.ptr(), n_rows as i32, stream.raw()) };
    CudaError::check(rc)
}

/// The three pre-attention steps of a Gated-MLA layer under LP2 in ONE launch: [`crate::mla_gated::kv_prep`]
/// (RMSNorm + rope of the latent into the cache), [`absorb`] (q_nope @ W_UK on tensor cores) and
/// [`crate::mla_gated::q_rope_only`] (the rope half of q_prep). Bit-identical per output to the three
/// separate launches (`gb_ling_mla_prep_absorb_rope_tc` runs the exact unfused kernel bodies on one grid).
#[allow(clippy::too_many_arguments)]
pub fn mla_prep_absorb_rope(
    q: &DevBuf,
    kv: &DevBuf,
    norm_w: &DevBuf,
    cos: &DevBuf,
    sin: &DevBuf,
    rows: &RowIndex,
    table: &BlockTable,
    cache: &mut MlaGatedCache,
    w: &MlaGatedWeights,
    q_lat: &mut DevBuf,
    q_rot: &mut DevBuf,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    check_bf16(w);
    assert_eq!(cache.dtype, Dtype::BF16, "the fused prep/absorb/rope is bf16 only");
    assert!(n_rows <= rows.rows);
    let qs = mla_gated::N_HEADS * mla_gated::QK_HEAD_DIM;
    assert!(q.bytes() >= n_rows * qs * 2, "q rows must hold 64 x 192 bf16");
    assert!(kv.bytes() >= n_rows * mla_gated::KV_A_STRIDE * 2, "kv rows must hold the padded 640 stride");
    assert!(norm_w.bytes() >= mla_gated::LATENT_DIM * 2);
    assert!(cos.bytes() >= 32 * 4 && sin.bytes() >= 32 * 4);
    assert!(q_lat.bytes() >= n_rows * mla_gated::N_HEADS * mla_gated::LATENT_DIM * 2);
    assert!(q_rot.bytes() >= n_rows * mla_gated::N_HEADS * mla_gated::ROPE_DIM * 2);
    let rc = unsafe {
        ffi::gb_ling_mla_prep_absorb_rope_tc(
            q.ptr(),
            qs as i64,
            kv.ptr(),
            mla_gated::KV_A_STRIDE as i32,
            norm_w.ptr(),
            cos.ptr() as *const c_void,
            sin.ptr() as *const c_void,
            32,
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            w.uk.ptr(),
            q_lat.ptr(),
            cache.buf.ptr(),
            q_rot.ptr(),
            n_rows as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Tensor-core dense causal absorbed attention for `n_rows` rows: `q_lat`, `o_lat` bf16 `[rows][64][512]`,
/// `q_rot` bf16 `[rows][64][64]`. Row `r` attends keys `0..=row_pos[r]` of slot `row_slot[r]`; a negative
/// position writes zeros. Same visible contract as [`crate::mla_gated::attend`], LP2 arithmetic.
/// `pair`: run two consecutive query rows per CTA (16 heads), sharing each staged 32-key tile — the prefill
/// path, where all rows share one sequence slot. Validation (rows of any slot) passes `false`.
#[allow(clippy::too_many_arguments)]
pub fn attend(
    q_lat: &DevBuf,
    q_rot: &DevBuf,
    cache: &MlaGatedCache,
    table: &BlockTable,
    rows: &RowIndex,
    o_lat: &mut DevBuf,
    n_rows: usize,
    pair: bool,
    stream: &Stream,
) -> Result<(), CudaError> {
    attend_from(q_lat, q_rot, cache, table, 0, rows, o_lat, n_rows, pair, stream)
}

/// [`attend`] over a window of each slot's pages (see [`mla_gated::attend_from`]): logical page `j` is the slot's page
/// `first_block + j`, and `row_pos` counts from that page's first position. `first_block == 0` is [`attend`].
#[allow(clippy::too_many_arguments)]
pub fn attend_from(
    q_lat: &DevBuf,
    q_rot: &DevBuf,
    cache: &MlaGatedCache,
    table: &BlockTable,
    first_block: usize,
    rows: &RowIndex,
    o_lat: &mut DevBuf,
    n_rows: usize,
    pair: bool,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert!(first_block < table.max_blocks.max(1), "window starts at block {first_block} of {}", table.max_blocks);
    assert_eq!(cache.dtype, Dtype::BF16, "the LP2 attention is bf16 only");
    assert!(n_rows <= rows.rows);
    let row_bytes = mla_gated::N_HEADS * mla_gated::LATENT_DIM * 2;
    assert!(q_lat.bytes() >= n_rows * row_bytes && o_lat.bytes() >= n_rows * row_bytes);
    assert!(q_rot.bytes() >= n_rows * mla_gated::N_HEADS * mla_gated::ROPE_DIM * 2);
    let rc = unsafe {
        ffi::gb_ling_mla_attend_tc(
            q_lat.ptr(),
            q_rot.ptr(),
            cache.buf.ptr(),
            (table.buf.ptr() as *const i32).add(first_block),
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            o_lat.ptr(),
            n_rows as i32,
            pair as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// The split-key decode attention ([`ARITH_ATTN`] v2): the visible contract of [`attend`] for `n_rows` rows (row `r` attends keys
/// `0..=row_pos[r]` of slot `row_slot[r]`, a negative position writes zeros), computed as absolute [`SPLIT_KEYS`]-key chunks on separate
/// CTAs and merged in ascending chunk order. `ws` is a [`SPLIT_WS_BYTES`] scratch buffer (the partials). The launch depends on the block
/// table's width and the row count only, never on a position, so a CUDA graph captured around it replays at any context. Rows are
/// independent: any number of them, in any order, in any call, gives each row the same bits (`tests/ling_mla_split.rs`).
#[allow(clippy::too_many_arguments)]
pub fn attend_split(
    q_lat: &DevBuf,
    q_rot: &DevBuf,
    cache: &MlaGatedCache,
    table: &BlockTable,
    rows: &RowIndex,
    o_lat: &mut DevBuf,
    ws: &mut DevBuf,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    attend_split_from(q_lat, q_rot, cache, table, 0, rows, o_lat, ws, n_rows, stream)
}

/// [`attend_split`] over a window of each slot's pages: logical page `j` is the slot's page `first_block + j` and a row's key range
/// `0..=row_pos[r]` is counted from that page's first position (the chunks too: chunk `c` starts `c * SPLIT_KEYS` keys into the window).
/// `first_block == 0` is [`attend_split`].
#[allow(clippy::too_many_arguments)]
pub fn attend_split_from(
    q_lat: &DevBuf,
    q_rot: &DevBuf,
    cache: &MlaGatedCache,
    table: &BlockTable,
    first_block: usize,
    rows: &RowIndex,
    o_lat: &mut DevBuf,
    ws: &mut DevBuf,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert!(first_block < table.max_blocks.max(1), "window starts at block {first_block} of {}", table.max_blocks);
    assert_eq!(cache.dtype, Dtype::BF16, "the LP2 attention is bf16 only");
    assert!(n_rows <= rows.rows);
    let row_bytes = mla_gated::N_HEADS * mla_gated::LATENT_DIM * 2;
    assert!(q_lat.bytes() >= n_rows * row_bytes && o_lat.bytes() >= n_rows * row_bytes);
    assert!(q_rot.bytes() >= n_rows * mla_gated::N_HEADS * mla_gated::ROPE_DIM * 2);
    assert!(ws.bytes() >= SPLIT_WS_BYTES, "the split attention workspace is {} bytes, needs {SPLIT_WS_BYTES}", ws.bytes());
    assert!(
        table.max_blocks.div_ceil(SPLIT_KEYS / mla_gated::BLOCK_TOKENS) <= SPLIT_WS_SLOTS,
        "a block table of {} pages spans more than {SPLIT_WS_SLOTS} chunks of {SPLIT_KEYS} keys",
        table.max_blocks
    );
    let rc = unsafe {
        ffi::gb_ling_mla_attend_split_tc(
            q_lat.ptr(),
            q_rot.ptr(),
            cache.buf.ptr(),
            (table.buf.ptr() as *const i32).add(first_block),
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            o_lat.ptr(),
            n_rows as i32,
            ws.ptr(),
            ws.bytes(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Test-only reference: the same call as [`attend_split_from`] through the pre-fusion two-launch split+combine
/// sequence (`gb_ling_mla_attend_split_ref_tc`). The A/B bit test in `tests/ling_kda_proj_fused_ab.rs` pins the
/// fused [`attend_split_from`] against it bit for bit. Not used by the model.
#[allow(clippy::too_many_arguments)]
pub fn attend_split_ref_from(
    q_lat: &DevBuf,
    q_rot: &DevBuf,
    cache: &MlaGatedCache,
    table: &BlockTable,
    first_block: usize,
    rows: &RowIndex,
    o_lat: &mut DevBuf,
    ws: &mut DevBuf,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert!(first_block < table.max_blocks.max(1), "window starts at block {first_block} of {}", table.max_blocks);
    assert_eq!(cache.dtype, Dtype::BF16, "the LP2 attention is bf16 only");
    assert!(n_rows <= rows.rows);
    let row_bytes = mla_gated::N_HEADS * mla_gated::LATENT_DIM * 2;
    assert!(q_lat.bytes() >= n_rows * row_bytes && o_lat.bytes() >= n_rows * row_bytes);
    assert!(q_rot.bytes() >= n_rows * mla_gated::N_HEADS * mla_gated::ROPE_DIM * 2);
    assert!(ws.bytes() >= SPLIT_WS_BYTES, "the split attention workspace is {} bytes, needs {SPLIT_WS_BYTES}", ws.bytes());
    assert!(
        table.max_blocks.div_ceil(SPLIT_KEYS / mla_gated::BLOCK_TOKENS) <= SPLIT_WS_SLOTS,
        "a block table of {} pages spans more than {SPLIT_WS_SLOTS} chunks of {SPLIT_KEYS} keys",
        table.max_blocks
    );
    let rc = unsafe {
        ffi::gb_ling_mla_attend_split_ref_tc(
            q_lat.ptr(),
            q_rot.ptr(),
            cache.buf.ptr(),
            (table.buf.ptr() as *const i32).add(first_block),
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            o_lat.ptr(),
            n_rows as i32,
            ws.ptr(),
            ws.bytes(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Test-only reference: [`attend_split_ref_from`] on the full table (`first_block == 0`).
#[allow(clippy::too_many_arguments)]
pub fn attend_split_ref(
    q_lat: &DevBuf,
    q_rot: &DevBuf,
    cache: &MlaGatedCache,
    table: &BlockTable,
    rows: &RowIndex,
    o_lat: &mut DevBuf,
    ws: &mut DevBuf,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    attend_split_ref_from(q_lat, q_rot, cache, table, 0, rows, o_lat, ws, n_rows, stream)
}

/// Tensor-core expand: `out[r][h*128 + n] = sum_c o_lat[r][h][c] * W_UV[h][c][n]`. bf16 only.
pub fn expand(o_lat: &DevBuf, w: &MlaGatedWeights, out: &mut DevBuf, n_rows: usize, stream: &Stream) -> Result<(), CudaError> {
    check_bf16(w);
    assert!(o_lat.bytes() >= n_rows * mla_gated::N_HEADS * mla_gated::LATENT_DIM * 2);
    assert!(out.bytes() >= n_rows * mla_gated::N_HEADS * mla_gated::NOPE_DIM * 2);
    let rc = unsafe { ffi::gb_ling_mla_expand_tc(o_lat.ptr(), w.uvt.ptr(), out.ptr(), n_rows as i32, stream.raw()) };
    CudaError::check(rc)
}

/// The expand fused with the head-wise sigmoid output gate:
/// `gated[r][h*128+n] = (sum_c o_lat[r][h][c] * W_UV[h][c][n]) * sigmoid(x[r] . g_proj[h])`, one launch.
/// `x` is the layer input bf16 `[rows][4096]` (the reference's `g_proj` operand). Replaces
/// [`expand`] + [`crate::mla_gated::gate_apply`]. `dense_suh` (fp16 `[8192]`) and `dense_xh`
/// (fp16 `[rows][8192]`), when both given (decode, rows <= 8), additionally fuse the dense projection's input
/// rotation into the epilogue (the exact `ling_rot_in_kernel` math, bit-identical to a separate
/// `gb_exl3_rotate_in` on the gated output), so the dense linear can skip its `rot_in` launch.
#[allow(clippy::too_many_arguments)]
pub fn expand_gated(
    o_lat: &DevBuf,
    x: &DevBuf,
    w: &MlaGatedWeights,
    gated: &mut DevBuf,
    n_rows: usize,
    dense_suh: Option<&DevBuf>,
    dense_xh: Option<&mut DevBuf>,
    stream: &Stream,
) -> Result<(), CudaError> {
    check_bf16(w);
    assert!(x.bytes() >= n_rows * mla_gated::HIDDEN * 2, "x must be bf16 [rows][4096]");
    assert!(o_lat.bytes() >= n_rows * mla_gated::N_HEADS * mla_gated::LATENT_DIM * 2);
    assert!(gated.bytes() >= n_rows * mla_gated::N_HEADS * mla_gated::NOPE_DIM * 2);
    let rc = unsafe {
        ffi::gb_ling_mla_expand_gate_tc(
            o_lat.ptr(),
            w.uvt.ptr(),
            x.ptr(),
            w.g.ptr(),
            gated.ptr(),
            n_rows as i32,
            dense_suh.map_or(core::ptr::null(), |d| d.ptr()),
            dense_xh.as_ref().map_or(core::ptr::null_mut(), |d| d.ptr()),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// The whole attention math of one Gated-MLA layer under LP2 for `n_rows` rows, after the host has produced
/// `q` (nope+rope) and the padded `kv`: the shared scalar `kv_prep` (RMSNorm + rope into the cache), the
/// tensor-core absorb, the shared scalar `q_rope` rotation, the tensor-core absorbed attention, and the
/// tensor-core expand fused with the sigmoid gate. `x` is the layer input; `gated` is the
/// `attention.dense` input. Same buffers and cache as [`crate::mla_gated::mla_gated_rows`]; `out` is unused
/// scratch kept for signature symmetry (the fused expand+gate writes `gated` directly).
#[allow(clippy::too_many_arguments)]
pub fn mla_gated_rows_tc(
    x: &DevBuf,
    q: &DevBuf,
    kv: &DevBuf,
    norm_w: &DevBuf,
    cos: &DevBuf,
    sin: &DevBuf,
    w: &MlaGatedWeights,
    cache: &mut MlaGatedCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    q_lat: &mut DevBuf,
    q_rot: &mut DevBuf,
    o_lat: &mut DevBuf,
    gated: &mut DevBuf,
    pair: bool,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert_eq!(w.dtype, cache.dtype, "weights and cache must share one dtype");
    check_bf16(w);
    // RMSNorm + rope of the key into the cache, the tensor-core absorb and the rope half of q_prep in one
    // launch (the three steps are data-independent; the fused kernel runs each exact unfused body).
    mla_prep_absorb_rope(q, kv, norm_w, cos, sin, rows, table, cache, w, q_lat, q_rot, n_rows, stream)?;
    attend(q_lat, q_rot, cache, table, rows, o_lat, n_rows, pair, stream)?;
    expand_gated(o_lat, x, w, gated, n_rows, None, None, stream)
}

/// [`mla_gated_rows_tc`] with the dense projection's input rotation fused into the expand+gate epilogue
/// (`dense_suh`/`dense_xh`, when both given): the same steps, `expand_gated` runs with the rotation enabled, so
/// the dense linear can skip its `rot_in` launch (P2 prefill chunks and decode windows).
#[allow(clippy::too_many_arguments)]
pub fn mla_gated_rows_tc_prerot(
    x: &DevBuf,
    q: &DevBuf,
    kv: &DevBuf,
    norm_w: &DevBuf,
    cos: &DevBuf,
    sin: &DevBuf,
    w: &MlaGatedWeights,
    cache: &mut MlaGatedCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    q_lat: &mut DevBuf,
    q_rot: &mut DevBuf,
    o_lat: &mut DevBuf,
    gated: &mut DevBuf,
    pair: bool,
    dense_suh: Option<&DevBuf>,
    dense_xh: Option<&mut DevBuf>,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert_eq!(w.dtype, cache.dtype, "weights and cache must share one dtype");
    check_bf16(w);
    mla_prep_absorb_rope(q, kv, norm_w, cos, sin, rows, table, cache, w, q_lat, q_rot, n_rows, stream)?;
    attend(q_lat, q_rot, cache, table, rows, o_lat, n_rows, pair, stream)?;
    expand_gated(o_lat, x, w, gated, n_rows, dense_suh, dense_xh, stream)
}

/// [`mla_gated_rows_tc`] for a decode / verify / draft call: the same steps, with the attention run by [`attend_split`] (`ws` is its
/// [`SPLIT_WS_BYTES`] workspace) instead of the single-walk kernel.
#[allow(clippy::too_many_arguments)]
pub fn mla_gated_rows_tc_split(
    x: &DevBuf,
    q: &DevBuf,
    kv: &DevBuf,
    norm_w: &DevBuf,
    cos: &DevBuf,
    sin: &DevBuf,
    w: &MlaGatedWeights,
    cache: &mut MlaGatedCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    q_lat: &mut DevBuf,
    q_rot: &mut DevBuf,
    o_lat: &mut DevBuf,
    gated: &mut DevBuf,
    ws: &mut DevBuf,
    dense_suh: Option<&DevBuf>,
    dense_xh: Option<&mut DevBuf>,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert_eq!(w.dtype, cache.dtype, "weights and cache must share one dtype");
    check_bf16(w);
    mla_prep_absorb_rope(q, kv, norm_w, cos, sin, rows, table, cache, w, q_lat, q_rot, n_rows, stream)?;
    attend_split(q_lat, q_rot, cache, table, rows, o_lat, ws, n_rows, stream)?;
    expand_gated(o_lat, x, w, gated, n_rows, dense_suh, dense_xh, stream)
}

mod eh_ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_ling_eh_proj_cat_bf16(
            e_norm: *const c_void,
            h_norm: *const c_void,
            w: *const c_void,
            y: *mut c_void,
            rows: i32,
            h: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_ling_eh_proj_cat_bf16_tile(
            e_norm: *const c_void,
            h_norm: *const c_void,
            w: *const c_void,
            y: *mut c_void,
            rows: i32,
            h: i32,
            j: i32,
            r: i32,
            rows_per_cta: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// The MTP layer's `eh_proj` over `rows` rows, row-batched (the end of `csrc/ling_mla_attn.cu`): `y[r][j] = sum_{c<h} e_norm[r][c] *
/// w[j][c] + sum_{c<h} h_norm[r][c] * w[j][h+c]`, `e_norm`, `h_norm` bf16 `[rows, h]`, `w` bf16 `[h, 2h]`, `y` bf16 `[rows, h]`,
/// `h % 256 == 0`.
///
/// Bit-identical, row for row, to [`crate::mla_gated::eh_proj_cat_bf16`] (the same per-lane fp32 fma chain in the same order, the
/// same warp tree, the same final rounding), so the result of a row does not depend on the number of rows in the call, its index,
/// or the tile. Only the cost differs: the shared kernel re-streams the whole weight (64 MiB at `h = 4096`) once per row (0.28 ms
/// a row, the one-time stall of the first MTP round after a cold prompt), this one once per slab of 1,024 rows. Not an arithmetic
/// change: [`ARITH_AE`] and [`ARITH_ATTN`] are untouched, and no prefix namespace, fingerprint or golden moves.
pub fn eh_proj_cat_bf16(
    e_norm: &DevBuf,
    h_norm: &DevBuf,
    w: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    h: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    check_eh_proj(e_norm, h_norm, w, y, rows, h);
    let rc = unsafe {
        eh_ffi::gb_ling_eh_proj_cat_bf16(e_norm.ptr(), h_norm.ptr(), w.ptr(), y.ptr(), rows as i32, h as i32, stream.raw())
    };
    CudaError::check(rc)
}

/// [`eh_proj_cat_bf16`] with the tile (`j` columns a warp x `r` rows a weight read) and the rows a CTA walks as arguments:
/// the bit tests and the tile bench only. An uninstantiated `(j, r)` returns `cudaErrorInvalidValue`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn eh_proj_cat_bf16_tile(
    e_norm: &DevBuf,
    h_norm: &DevBuf,
    w: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    h: usize,
    j: usize,
    r: usize,
    rows_per_cta: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    check_eh_proj(e_norm, h_norm, w, y, rows, h);
    let rc = unsafe {
        eh_ffi::gb_ling_eh_proj_cat_bf16_tile(
            e_norm.ptr(),
            h_norm.ptr(),
            w.ptr(),
            y.ptr(),
            rows as i32,
            h as i32,
            j as i32,
            r as i32,
            rows_per_cta as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

fn check_eh_proj(e_norm: &DevBuf, h_norm: &DevBuf, w: &DevBuf, y: &DevBuf, rows: usize, h: usize) {
    assert!(h > 0 && h % 256 == 0, "eh_proj width must be a positive multiple of 256");
    assert!(e_norm.bytes() >= rows * h * 2, "e_norm must be bf16 [rows][h]");
    assert!(h_norm.bytes() >= rows * h * 2, "h_norm must be bf16 [rows][h]");
    assert!(w.bytes() >= h * (2 * h) * 2, "w must be bf16 [h][2h]");
    assert!(y.bytes() >= rows * h * 2, "y must be bf16 [rows][h]");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arith_ids_are_stable_and_distinct() {
        // These strings go into the P2 prefix namespace and fingerprint; an accidental edit is a silent
        // cache-poisoning hazard, so pin them.
        assert_eq!(ARITH_AE, "ling-mla-ae-fold128-bf16-v1");
        assert_eq!(ARITH_ATTN, "ling-mla-attn-tc32-bf16-v2-decode-split1024");
        assert_ne!(ARITH_AE, ARITH_ATTN);
        // the id names the chunk length: a different SPLIT_KEYS is a different id
        assert!(ARITH_ATTN.ends_with(&format!("split{SPLIT_KEYS}")));
    }

    #[test]
    fn the_split_workspace_is_fixed_and_whole_chunks_whole_pages() {
        assert_eq!(SPLIT_KEYS % mla_gated::BLOCK_TOKENS, 0);
        assert_eq!(SPLIT_KEYS % 32, 0);
        assert_eq!(SPLIT_SLOT_BYTES, 131_584);
        assert_eq!(SPLIT_WS_BYTES, 42_109_440);
        // the window the planner prices (262,144 tokens = 1,024 pages) is 256 chunks, inside the workspace's 320
        assert!(262_144usize.div_ceil(SPLIT_KEYS) <= SPLIT_WS_SLOTS);
    }

    #[test]
    fn geometry_matches_the_scalar_contract() {
        // The tensor-core path must not reinterpret the cache or the packed weights.
        assert_eq!(mla_gated::N_HEADS, 64);
        assert_eq!(mla_gated::LATENT_DIM, 512);
        assert_eq!(mla_gated::NOPE_DIM, 128);
        assert_eq!(mla_gated::ROPE_DIM, 64);
        assert_eq!(mla_gated::CACHE_TOKEN_ELEMS, 576);
        assert_eq!(mla_gated::QK_HEAD_DIM, 192);
        assert!((mla_gated::SM_SCALE - 0.07216878364870322).abs() < 1e-12);
    }
}
