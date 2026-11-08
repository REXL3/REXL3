//! Ling-3.1-flash-1019 (`BailingMoeV3MultiLatentAttention`) Gated MLA: 64 heads, `qk_nope_head_dim = 128`,
//! `qk_rope_head_dim = 64`, `v_head_dim = 128`, `kv_lora_rank = 512`, full causal attention (no DSA sparse
//! indexer), interleaved (adjacent-pair) RoPE with `theta = 6e6`, and a head-wise sigmoid output gate `g_proj`
//! (`[64, 4096]` bf16) applied before `attention.dense`. The kernels live in `csrc/mla_gated.cu`; this module is
//! their Rust side (dims, packed weights, rope tables, cache, launch wrappers).
//!
//! # The per-token cache layout
//!
//! One MLA layer's cache is a [`MlaGatedCache`]: `T [n_blocks][256 tokens][576]` — the **normalised** 512-wide
//! latent `c_kv` (cols 0..511) followed by the 64 **rotated** rope key (cols 512..575). A device block table
//! `i32 [slots][max_blocks]` ([`BlockTable`], shared with the GQA cache) maps `(slot, position / 256)` to a block
//! id, and [`RowIndex`] gives each row of a call its `(slot, absolute position)`.
//!
//! # The pipeline for one MLA layer
//!
//! 1. Host: `q_a_proj` → `q_a_layernorm` → `q_b_proj` → `q` `T [rows][64*192]` (head `h` at `h*192`: cols
//!    `[h*192, h*192+128)` nope, `[h*192+128, h*192+192)` rope); `kv_a_proj_with_mqa` (EXL3, padded to
//!    `(4096, 640)`) → `kv` `T [rows][640]` (cols `[0,512)` latent, `[512,576)` rope, `[576,640)` padding).
//! 2. [`kv_prep`] → `(slot, pos)` of the cache: RMSNorm of the latent (`kv_a_layernorm`) + rope rotation of the key.
//! 3. [`q_prep`] → `q_lat` `T [rows][64][512]` (nope absorbed through `W_UK`) and `q_rot` `T [rows][64][64]`.
//! 4. [`attend`] → `o_lat` `T [rows][64][512]`, dense causal attention over the cache.
//! 5. [`expand`] → `out` `T [rows][64*128]` through `W_UV`.
//! 6. [`gate_apply`] → `gated` `T [rows][64*128]`: `sigmoid(x . g_proj[h])` times head `h`'s 128 columns, the
//!    `attention.dense` input.
//!
//! # Row invariance (the contract this file exists to keep)
//!
//! Row `r`'s output bits depend only on its own query row, its slot's keys `0..=pos[r]` and its own position, for any
//! row count, any position, any block placement, any `max_context` and any other rows in the call — so prefill
//! chunkings decode-style appends give identical bits (see the header of `csrc/mla_gated.cu`).

use crate::attention::{BlockTable, RowIndex};
use crate::exl3_linear::Dtype;
use crate::rt::{CudaError, DevBuf, Stream};

/// Latent width (`kv_lora_rank`).
pub const LATENT_DIM: usize = 512;
/// Query heads.
pub const N_HEADS: usize = 64;
/// `qk_nope_head_dim` == `v_head_dim`.
pub const NOPE_DIM: usize = 128;
/// `qk_rope_head_dim`.
pub const ROPE_DIM: usize = 64;
/// `qk_head_dim`.
pub const QK_HEAD_DIM: usize = NOPE_DIM + ROPE_DIM;
/// Elements of one token's cache row (normalised latent + rotated rope).
pub const CACHE_TOKEN_ELEMS: usize = LATENT_DIM + ROPE_DIM;
/// Rot pe `theta` of `Ling-3.1-flash-1019` (`rope_theta = 6000000.0`).
pub const ROPE_THETA: f32 = 6000000.0;
/// `qk_head_dim ^ -0.5`.
pub const SM_SCALE: f32 = 0.07216878364870322;
/// Tokens per cache block (the page size of every paged cache in the engine).
pub const BLOCK_TOKENS: usize = 256;
/// The padded out-dimension of `kv_a_proj_with_mqa` (EXL3's 128-multiple rounding: 576 → 640).
pub const KV_A_STRIDE: usize = 640;
/// Hidden size (`model.layers.{l}.attention.g_proj.weight` rows).
pub const HIDDEN: usize = 4096;
/// `kv_b_proj` rows per head: `qk_nope_head_dim + v_head_dim`.
pub const KV_B_ROWS_PER_HEAD: usize = 2 * NOPE_DIM;

fn check_dtype(d: Dtype) -> i32 {
    assert!(matches!(d, Dtype::F16 | Dtype::BF16), "MLA kernels take fp16 or bf16 activations, not {d:?}");
    d as i32
}

/// One layer's paged latent+rope cache `T [n_blocks][256][576]`.
#[derive(Debug)]
pub struct MlaGatedCache {
    pub buf: DevBuf,
    pub n_blocks: usize,
    pub dtype: Dtype,
}

impl MlaGatedCache {
    /// Bytes of one block of latents.
    pub fn block_bytes(dtype: Dtype) -> usize {
        BLOCK_TOKENS * CACHE_TOKEN_ELEMS * dtype.size()
    }
    /// Allocates an (uninitialised) cache of `n_blocks` pages.
    pub fn new(n_blocks: usize, dtype: Dtype) -> Result<Self, CudaError> {
        check_dtype(dtype);
        Ok(Self { buf: DevBuf::alloc(n_blocks * Self::block_bytes(dtype))?, n_blocks, dtype })
    }
    pub fn capacity_tokens(&self) -> usize {
        self.n_blocks * BLOCK_TOKENS
    }
}

/// `kv_b_proj` and `g_proj` of one MLA layer split and packed for the kernels: `uk` `T [64][128 d][512 c]` (head
/// `h`'s rows `h*256 + d`, `d < 128`, contiguous in `c`), `uvt` `T [64][512 c][128 n]` (rows `h*256 + 128 + n`,
/// transposed to `[c][n]`), and `g` `bf16 [64][4096]`.
#[derive(Debug)]
pub struct MlaGatedWeights {
    pub uk: DevBuf,
    pub uvt: DevBuf,
    /// `g_proj.weight` (`[64, 4096]`, bf16).
    pub g: DevBuf,
    pub dtype: Dtype,
}

fn convert(v: u16, src: Dtype, dst: Dtype) -> u16 {
    if src == dst {
        return v;
    }
    let f = match src {
        Dtype::BF16 => crate::num::bf16_to_f32(v) as f64,
        Dtype::F16 => gb_exl3::f16::f16_to_f64(v),
        Dtype::F32 => panic!("kv_b_proj is 16-bit"),
    };
    match dst {
        Dtype::BF16 => crate::num::f32_to_bf16(f as f32),
        Dtype::F16 => gb_exl3::f16::f64_to_f16(f),
        Dtype::F32 => panic!("kv_b_proj is 16-bit"),
    }
}

/// Splits and packs the checkpoint's `kv_b_proj.weight` (`[64 * 256][512]`, 16-bit bits, row `h*256 + j`:
/// `j < 128` is `W_UK`, `128 <= j < 256` is `W_UV`) into `(uk, uvt)` host vectors in the kernels' layouts,
/// converting `src` -> `dst` through f32 (identity when equal).
pub fn pack_kv_b(rows: &[u16], src: Dtype, dst: Dtype) -> (Vec<u16>, Vec<u16>) {
    assert_eq!(rows.len(), N_HEADS * KV_B_ROWS_PER_HEAD * LATENT_DIM, "kv_b_proj must be [64*256][512]");
    let mut uk = vec![0u16; N_HEADS * NOPE_DIM * LATENT_DIM];
    let mut uvt = vec![0u16; N_HEADS * LATENT_DIM * NOPE_DIM];
    for h in 0..N_HEADS {
        for d in 0..NOPE_DIM {
            let k_row = &rows[(h * KV_B_ROWS_PER_HEAD + d) * LATENT_DIM..][..LATENT_DIM];
            for (c, &v) in k_row.iter().enumerate() {
                uk[(h * NOPE_DIM + d) * LATENT_DIM + c] = convert(v, src, dst);
            }
            let v_row = &rows[(h * KV_B_ROWS_PER_HEAD + NOPE_DIM + d) * LATENT_DIM..][..LATENT_DIM];
            for (c, &v) in v_row.iter().enumerate() {
                uvt[(h * LATENT_DIM + c) * NOPE_DIM + d] = convert(v, src, dst);
            }
        }
    }
    (uk, uvt)
}

impl MlaGatedWeights {
    /// Uploads `kv_b_proj.weight` (see [`pack_kv_b`]) and `g_proj.weight` (`[64, 4096]`, bf16 bits).
    pub fn from_checkpoint(kv_b: &[u16], kv_b_src: Dtype, g_proj: &[u16], dst: Dtype) -> Result<Self, CudaError> {
        check_dtype(dst);
        assert_eq!(g_proj.len(), N_HEADS * HIDDEN, "g_proj must be [64][4096]");
        let (uk, uvt) = pack_kv_b(kv_b, kv_b_src, dst);
        Ok(Self { uk: DevBuf::from_slice(&uk)?, uvt: DevBuf::from_slice(&uvt)?, g: DevBuf::from_slice(g_proj)?, dtype: dst })
    }
    /// Device bytes of all three halves.
    pub fn bytes(&self) -> usize {
        self.uk.bytes() + self.uvt.bytes() + self.g.bytes()
    }
}

/// The host-built RoPE trig tables: `cos`/`sin` `f32 [max_pos][32]` with
/// `inv_freq[i] = 1 / theta^(2i/64)` and `cos[pos*32 + i] = cos(pos * inv_freq[i])` (and likewise for sin). The
/// reference builds 64-wide `cos`/`sin` by concatenating `freqs` with itself; only the first 32 entries per position
/// are needed by the kernels (the rotation formula uses frequency `i` at positions `i` and `i + 32`).
pub fn rope_tables(max_pos: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0.0f32; max_pos * (ROPE_DIM / 2)];
    let mut sin = vec![0.0f32; max_pos * (ROPE_DIM / 2)];
    for i in 0..ROPE_DIM / 2 {
        let inv = 1.0 / theta.powf((2 * i) as f32 / ROPE_DIM as f32);
        for p in 0..max_pos {
            let a = p as f32 * inv;
            cos[p * (ROPE_DIM / 2) + i] = a.cos();
            sin[p * (ROPE_DIM / 2) + i] = a.sin();
        }
    }
    (cos, sin)
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        #[allow(clippy::too_many_arguments)]
        pub fn gb_mlag_kv_prep(
            kv: *const c_void,
            kv_stride: i32,
            norm_w: *const c_void,
            cos: *const f32,
            sin: *const f32,
            table_stride: i32,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            cache: *mut c_void,
            rows: i32,
            dtype: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_mlag_q_prep(
            q: *const c_void,
            q_row_stride: i64,
            w_uk: *const c_void,
            cos: *const f32,
            sin: *const f32,
            table_stride: i32,
            row_pos: *const i32,
            q_lat: *mut c_void,
            q_rot: *mut c_void,
            rows: i32,
            dtype: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_mlag_attend(
            q_lat: *const c_void,
            q_rot: *const c_void,
            cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            out: *mut c_void,
            rows: i32,
            dtype: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_mlag_expand(
            o_lat: *const c_void,
            w_uvt: *const c_void,
            out: *mut c_void,
            rows: i32,
            dtype: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_mlag_gate_apply(
            x: *const c_void,
            x_stride: i64,
            g: *const c_void,
            out: *const c_void,
            out_gated: *mut c_void,
            rows: i32,
            dtype: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_mlag_bf16_linear(
            x: *const c_void,
            w: *const c_void,
            y: *mut c_void,
            rows: i32,
            k: i32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_mlag_eh_proj_cat_bf16(
            e_norm: *const c_void,
            h_norm: *const c_void,
            w: *const c_void,
            y: *mut c_void,
            rows: i32,
            h: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// [`kv_prep`] for `n_rows` rows: RMSNorm of the latent (`kv_a_layernorm`, eps `1e-6`) and rope rotation of the
/// key, written into the cache at `(row_slot[r], row_pos[r])` (cols `512..576` of `kv`; cols `576..640` are EXL3
/// padding and are never read). A negative position skips the row.
#[allow(clippy::too_many_arguments)]
pub fn kv_prep(
    kv: &DevBuf,
    norm_w: &DevBuf,
    cos: &DevBuf,
    sin: &DevBuf,
    rows: &RowIndex,
    table: &BlockTable,
    cache: &mut MlaGatedCache,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    let dt = check_dtype(cache.dtype);
    assert!(n_rows <= rows.rows);
    assert!(kv.bytes() >= n_rows * KV_A_STRIDE * cache.dtype.size(), "kv rows must hold the padded 640 stride");
    assert!(norm_w.bytes() >= LATENT_DIM * cache.dtype.size());
    assert!(cos.bytes() >= 32 * 4 && sin.bytes() >= 32 * 4);
    let rc = unsafe {
        ffi::gb_mlag_kv_prep(
            kv.ptr(),
            KV_A_STRIDE as i32,
            norm_w.ptr(),
            cos.ptr() as *const f32,
            sin.ptr() as *const f32,
            32,
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            cache.buf.ptr(),
            n_rows as i32,
            dt,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// [`q_prep`] for `n_rows` rows: absorb `q_lat[r][h][c] = sum_d q[r][h][d] * W_UK[h][d][c]` (`d < 128`, one
/// fixed-order fp32 chain per element, rounded once) and rotate `q_rot` (rope cols `128..192` of each head).
/// `q`: `T [rows][64*192]` (head `h` at `h*192`); `q_lat`: `T [rows][64][512]`; `q_rot`: `T [rows][64][64]`.
#[allow(clippy::too_many_arguments)]
pub fn q_prep(
    q: &DevBuf,
    w: &MlaGatedWeights,
    cos: &DevBuf,
    sin: &DevBuf,
    rows: &RowIndex,
    q_lat: &mut DevBuf,
    q_rot: &mut DevBuf,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    let dt = check_dtype(w.dtype);
    let qs = N_HEADS * QK_HEAD_DIM;
    assert!(q.bytes() >= n_rows * qs * w.dtype.size(), "q rows must hold 64 x 192");
    assert!(q_lat.bytes() >= n_rows * N_HEADS * LATENT_DIM * w.dtype.size());
    assert!(q_rot.bytes() >= n_rows * N_HEADS * ROPE_DIM * w.dtype.size());
    let rc = unsafe {
        ffi::gb_mlag_q_prep(
            q.ptr(),
            qs as i64,
            w.uk.ptr(),
            cos.ptr() as *const f32,
            sin.ptr() as *const f32,
            32,
            rows.pos.ptr() as *const i32,
            q_lat.ptr(),
            q_rot.ptr(),
            n_rows as i32,
            dt,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Dense causal latent attention for `n_rows` rows: `q_lat`, `q_rot`, `o_lat` `T [rows][64][512]` (and
/// `T [rows][64][64]` for `q_rot`). Row `r` attends to keys `0..=row_pos[r]` of slot `row_slot[r]` (already written
/// by [`kv_prep`]); a negative position is a padding row (its `o_lat` is zeros). Each row's bits depend only on its
/// own query, its slot's keys and its position.
#[allow(clippy::too_many_arguments)]
pub fn attend(
    q_lat: &DevBuf,
    q_rot: &DevBuf,
    cache: &MlaGatedCache,
    table: &BlockTable,
    rows: &RowIndex,
    o_lat: &mut DevBuf,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    let dt = check_dtype(cache.dtype);
    assert!(n_rows <= rows.rows);
    let row_bytes = N_HEADS * LATENT_DIM * cache.dtype.size();
    assert!(q_lat.bytes() >= n_rows * row_bytes && o_lat.bytes() >= n_rows * row_bytes);
    assert!(q_rot.bytes() >= n_rows * N_HEADS * ROPE_DIM * cache.dtype.size());
    let rc = unsafe {
        ffi::gb_mlag_attend(
            q_lat.ptr(),
            q_rot.ptr(),
            cache.buf.ptr(),
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            o_lat.ptr(),
            n_rows as i32,
            dt,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Expand: `out[r][h*128 + n] = sum_c o_lat[r][h][c] * W_UV[h][c][n]`. `o_lat`: `T [rows][64][512]`, `out`:
/// `T [rows][64*128]`.
pub fn expand(o_lat: &DevBuf, w: &MlaGatedWeights, out: &mut DevBuf, n_rows: usize, stream: &Stream) -> Result<(), CudaError> {
    let dt = check_dtype(w.dtype);
    assert!(o_lat.bytes() >= n_rows * N_HEADS * LATENT_DIM * w.dtype.size());
    assert!(out.bytes() >= n_rows * N_HEADS * NOPE_DIM * w.dtype.size());
    let rc = unsafe { ffi::gb_mlag_expand(o_lat.ptr(), w.uvt.ptr(), out.ptr(), n_rows as i32, dt, stream.raw()) };
    CudaError::check(rc)
}

/// Apply the head-wise sigmoid gate: `gated[r][h*128 + n] = sigmoid(x[r] . g_proj[h]) * out[r][h*128 + n]`.
/// `x`: the layer input `T [rows][4096]` (the same `hidden_states` the reference's `g_proj` reads); `g_proj` is
/// bf16 `[64][4096]`; `out`, `gated`: `T [rows][64*128]`.
#[allow(clippy::too_many_arguments)]
pub fn gate_apply(
    x: &DevBuf,
    w: &MlaGatedWeights,
    out: &DevBuf,
    gated: &mut DevBuf,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    let dt = check_dtype(w.dtype);
    assert!(x.bytes() >= n_rows * HIDDEN * w.dtype.size());
    assert!(out.bytes() >= n_rows * N_HEADS * NOPE_DIM * w.dtype.size());
    assert!(gated.bytes() >= n_rows * N_HEADS * NOPE_DIM * w.dtype.size());
    let rc = unsafe {
        ffi::gb_mlag_gate_apply(x.ptr(), HIDDEN as i64, w.g.ptr(), out.ptr(), gated.ptr(), n_rows as i32, dt, stream.raw())
    };
    CudaError::check(rc)
}

/// Direct bf16 linear `y[r][j] = sum_c x[r][c] * w[j][c]` in one kernel launch (no dtype round-trips).
/// `x`: bf16 `[rows, k]`; `w`: bf16 `[n, k]`; `y`: bf16 `[rows, n]` (`k % 256 == 0`, `n >= 1`). One
/// warp computes one `(row, output column)` pair, so each row's bits depend only on its own input row.
pub fn bf16_linear(
    x: &DevBuf,
    w: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    k: usize,
    n: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert!(x.bytes() >= rows * k * 2, "x must be bf16 [rows][k]");
    assert!(w.bytes() >= n * k * 2, "w must be bf16 [n][k]");
    assert!(y.bytes() >= rows * n * 2, "y must be bf16 [rows][n]");
    let rc = unsafe { ffi::gb_mlag_bf16_linear(x.ptr(), w.ptr(), y.ptr(), rows as i32, k as i32, n as i32, stream.raw()) };
    CudaError::check(rc)
}

/// Fused `eh_proj` cat: `y[r][j] = sum_{c<h} e_norm[r][c] * w[j][c] + sum_{c<h} h_norm[r][c] * w[j][h+c]`
/// in one kernel launch. `e_norm`, `h_norm`: bf16 `[rows, h]`; `w`: bf16 `[h, 2h]`; `y`: bf16
/// `[rows, h]` (`h % 256 == 0`). Same warp-per-output scheme as [`bf16_linear`].
pub fn eh_proj_cat_bf16(
    e_norm: &DevBuf,
    h_norm: &DevBuf,
    w: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    h: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert!(e_norm.bytes() >= rows * h * 2, "e_norm must be bf16 [rows][h]");
    assert!(h_norm.bytes() >= rows * h * 2, "h_norm must be bf16 [rows][h]");
    assert!(w.bytes() >= h * (2 * h) * 2, "w must be bf16 [h][2h]");
    assert!(y.bytes() >= rows * h * 2, "y must be bf16 [rows][h]");
    let rc = unsafe {
        ffi::gb_mlag_eh_proj_cat_bf16(e_norm.ptr(), h_norm.ptr(), w.ptr(), y.ptr(), rows as i32, h as i32, stream.raw())
    };
    CudaError::check(rc)
}

/// The whole attention math of one Gated MLA layer for `n_rows` rows, after the host has produced `q` and `kv`
/// (nope+rope q, padded kv): kv_prep, q_prep, attend, expand, gate_apply. `x` is the layer input (the reference's
/// `g_proj` operand); `gated` is the `attention.dense` input.
#[allow(clippy::too_many_arguments)]
pub fn mla_gated_rows(
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
    out: &mut DevBuf,
    gated: &mut DevBuf,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert_eq!(w.dtype, cache.dtype, "weights and cache must share one dtype");
    kv_prep(kv, norm_w, cos, sin, rows, table, cache, n_rows, stream)?;
    q_prep(q, w, cos, sin, rows, q_lat, q_rot, n_rows, stream)?;
    attend(q_lat, q_rot, cache, table, rows, o_lat, n_rows, stream)?;
    expand(o_lat, w, out, n_rows, stream)?;
    gate_apply(x, w, out, gated, n_rows, stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_tables_match_the_interleave_reference() {
        // Ling-3.1's `apply_rotary_pos_emb_interleave` rotates pair (2i, 2i+1) with frequency i and stores
        // the results half-split (the form the kernels consume; see the header of csrc/mla_gated.cu):
        // y[i] = x[2i] c_i - x[2i+1] s_i, y[i+32] = x[2i] s_i + x[2i+1] c_i.
        let (cos, sin) = rope_tables(8, ROPE_THETA);
        let pos = 5usize;
        let x: Vec<f32> = (0..64).map(|i| 0.1 * i as f32).collect();
        let mut y = vec![0.0f32; 64];
        let mut expected = vec![0.0f32; 64];
        for i in 0..32 {
            // Independent cos/sin, recomputed from the frequency definition.
            let inv = 1.0 / ROPE_THETA.powf((2 * i) as f32 / ROPE_DIM as f32);
            let (c, s) = ((pos as f32 * inv).cos(), (pos as f32 * inv).sin());
            assert!((cos[pos * 32 + i] - c).abs() < 1e-6, "cos table entry {i} at pos {pos}");
            assert!((sin[pos * 32 + i] - s).abs() < 1e-6, "sin table entry {i} at pos {pos}");
            y[i] = x[2 * i] * c - x[2 * i + 1] * s;
            y[i + 32] = x[2 * i] * s + x[2 * i + 1] * c;
            expected[i] = x[2 * i] * c - x[2 * i + 1] * s;
            expected[i + 32] = x[2 * i] * s + x[2 * i + 1] * c;
        }
        for i in 0..64 {
            assert!((y[i] - expected[i]).abs() < 1e-6, "rope layout mismatch at {i}: {} vs {expected:?}", y[i]);
        }
    }

    #[test]
    fn pack_kv_b_splits_k_and_v_per_head_and_transposes_v() {
        let rows: Vec<u16> = (0..N_HEADS * KV_B_ROWS_PER_HEAD * LATENT_DIM)
            .map(|i| (((i / LATENT_DIM) * 7 + (i % LATENT_DIM) * 13) & 0x7fff) as u16)
            .collect();
        let (uk, uvt) = pack_kv_b(&rows, Dtype::BF16, Dtype::BF16);
        let at = |r: usize, c: usize| rows[r * LATENT_DIM + c];
        for (h, d, c) in [(0usize, 0usize, 0usize), (0, 127, 511), (1, 0, 3), (63, 127, 7), (17, 100, 400)] {
            assert_eq!(uk[(h * NOPE_DIM + d) * LATENT_DIM + c], at(h * 256 + d, c), "uk h{h} d{d} c{c}");
            assert_eq!(uvt[(h * LATENT_DIM + c) * NOPE_DIM + d], at(h * 256 + 128 + d, c), "uvt h{h} n{d} c{c}");
        }
    }

    #[test]
    fn dims_are_the_ling31_contract() {
        assert_eq!(N_HEADS, 64);
        assert_eq!(LATENT_DIM, 512);
        assert_eq!(NOPE_DIM, 128);
        assert_eq!(ROPE_DIM, 64);
        assert_eq!(QK_HEAD_DIM, 192);
        assert_eq!(CACHE_TOKEN_ELEMS, 576);
        assert_eq!(KV_A_STRIDE, 640);
    }
}
