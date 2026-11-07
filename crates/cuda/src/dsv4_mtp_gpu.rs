//! The GPU DSpark / MTP drafter: `Transformer.forward_spec` (model.py:1277) as three chained `DSparkBlock`s over
//! MXFP4 routed experts, producing a [`DsparkDraft`] in one call.
//!
//! # What this is
//!
//! The host reference in [`crate::dsv4_mtp`] is the semantics; this is the device form of it, shaped for a serve:
//!
//! ```text
//!   taps [1, 3*dim]  ->  main_x = main_norm(main_proj(taps))            (stage 0 only)
//!   h = embed([anchor, NOISE x (block_size - 1)]) x hc_mult             (stage 0 only)
//!   for stage in 0..3:  h = hc_post(h, moe(norm(hc_pre(h, ...))))       one MLA + one top-3 MoE each
//!   out = hc_pre(h) -> norm -> head -> markov walk -> confidence
//! ```
//!
//! # The contract the serve needs
//!
//! * **One call, no host reads.** [`DsparkGpu::draft`] issues launches and returns; the token ids come from a device
//!   argmax, so nothing is read back inside the round. [`DsparkGpu::read`] is the one small copy (ids + confidences,
//!   `4 * (block_size + 1) + 4 * block_size` bytes), and it is the caller's to make.
//! * **No allocation per call.** Every buffer is a [`DevBuf::view`] into one arena sized by [`DsparkGpu::bytes`];
//!   [`MtpScratch::bytes`] and [`MtpStage::weight_bytes`] are what admission prices.
//! * **CUDA-graph capturable.** Fixed shapes, a fixed launch count, no host-to-device copy on the stream (the anchor
//!   token is written by a kernel). The one runtime scalar is the slot count
//!   `n_kv = min(window, start_pos + 1) + block_size`, which stops changing at `start_pos + 1 >= window`; capture from
//!   there, or one graph a step for the first `window` steps.
//!
//! # Per-stage window KV
//!
//! Each stage owns its own ring of `window` raw-KV latents, written at `start_pos % window` from the trunk tap and read
//! back as `window ++ block`. The block's own rows are **not** in the ring: they are transient, and the next step would
//! overwrite the same slot. Seeded once by [`DsparkGpu::prefill_seed`].
//!
//! # Experts
//!
//! [`MtpExperts`] is the seam for the re-quantisation to EXL3 (see its doc for exactly what an implementation owes).

use crate::dsv4_attn::{quant_dev, rmsnorm_dev, rope_rows_dev, RopeSpec, RopeTables};
use crate::dsv4_moe::{moe_experts, MoeGroups, MoeScratch};
use crate::dsv4_mtp::{accept_prefix, DsparkDraft, MtpConfig};
use crate::exl3_linear::Dtype;
use crate::exl3_linear::Exl3Weights;
use crate::fp4_linear::Mxfp4Weights;
use crate::graph::{CaptureError, Graph};
use crate::rt::{Block, CudaError, DevBuf, Event, Stream};
use std::collections::HashMap;

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_fp8_linear(x: *const c_void, x_dtype: i32, w: *const u8, s: *const u8, y: *mut c_void,
                             m: i32, k: i32, n: i32, force_mma: i32, stream: *mut c_void) -> i32;
        pub fn gb_fp8_linear_pair(x: *const c_void, x_dtype: i32, w1: *const u8, s1: *const u8, y1: *mut c_void,
                                  n1: i32, w2: *const u8, s2: *const u8, y2: *mut c_void, n2: i32,
                                  m: i32, k: i32, force_mma: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_markov_walk_fast_bf16(logits: *mut c_void, ids: *mut i32, embed: *const c_void,
                                                 head: *const c_void, cand_ids_ws: *mut c_void,
                                                 cand_scores_ws: *mut c_void, walk_rows: i32, vocab: i32,
                                                 rank: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_markov_walk_keyed_bf16(logits: *mut c_void, ids: *mut i32, conf: *mut f32,
                                                  embed: *const c_void, head: *const c_void,
                                                  cand_ids_ws: *mut c_void, cand_scores_ws: *mut c_void,
                                                  walk_rows: i32, vocab: i32, rank: i32,
                                                  cand_k: i32, seed: u64, temperature: f64,
                                                  top_p: f64, min_p: f64, base_pos: u64,
                                                  stream: *mut c_void) -> i32;
        pub fn gb_dsv4_woa_grouped(o: *const c_void, w: *const u8, s: *const u8, out: *mut c_void, rows: i32,
                                   groups: i32, o_lora: i32, o_in: i32, dtype: i32, use_classic: i32,
                                   force_mma: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_swiglu_f16(gate: *const c_void, up: *const c_void, w: *const c_void, out: *mut c_void,
                                  n: i32, inter: i32, limit: f32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_moe_group(picks: *const i32, wts: *const c_void, counts: *mut i32, offsets: *mut i32,
                                 perm: *mut i32, pinv: *mut i32, wpack: *mut c_void, slots: i32, n_exp: i32,
                                 stream: *mut c_void) -> i32;
        pub fn gb_dsv4_moe_gather(x: *const c_void, perm: *const i32, xp: *mut c_void, slots: i32, topk: i32,
                                  h: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_moe_scatter(shared: *const c_void, yp: *const c_void, pinv: *const i32, wpack: *const c_void,
                                   out_f32: *mut c_void, out_f16: *mut c_void, rows: i32, topk: i32, h: i32,
                                   stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_topk_idxs(out: *mut c_void, bsz: i32, block_size: i32, reach: i32, window: i32,
                                     stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_topk_idxs_params(out: *mut c_void, bsz: i32, block_size: i32, params: *const c_void,
                                            stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_attn_params(q: *const c_void, kv: *const c_void, sink: *const f32, idxs: *const i32,
                                        o: *mut c_void, block: i32, n_heads: i32, hd: i32, scale: f32,
                                        params: *const c_void, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_fill_ids_params(ids: *mut i32, pre_mix: *mut f32, params: *const c_void,
                                           noise: i32, block: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_markov_walk_keyed_bf16_params(logits: *mut c_void, ids: *mut i32, conf: *mut f32,
                                                         embed: *const c_void, head: *const c_void,
                                                         cand_ids_ws: *mut c_void, cand_scores_ws: *mut c_void,
                                                         walk_rows: i32, vocab: i32, rank: i32,
                                                         params: *const c_void,
                                                         stream: *mut c_void) -> i32;
        pub fn gb_dsv4_ring_write_params(src: *const c_void, ring: *mut c_void, rows: i32, width: i32,
                                         params: *const c_void, dtype: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_rope_params(x: *mut c_void, dtype: i32, rows: i32, stride: i32, dim: i32, block: i32,
                                   sub: i32, cos: *const f32, sin: *const f32, table_stride: i32,
                                   pos_off: i32, pos_step: i32, inverse: i32, params: *const c_void,
                                   stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_markov_bias(logits: *mut c_void, tokens: *const c_void, embed: *const c_void,
                                       head: *const c_void, rows: i32, vocab: i32, rank: i32,
                                       stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_confidence(h: *const c_void, m: *const c_void, w: *const c_void, b: *const c_void,
                                      out: *mut c_void, rows: i32, dim: i32, rank: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_accept(conf: *const c_void, out: *mut c_void, n_seq: i32, block_size: i32,
                                  threshold: f32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_embed(h: *mut c_void, ids: *const i32, table: *const c_void, rows: i32, nc: i32,
                                 dim: i32, vocab: i32, dtype: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_hc_mixes(x: *const f32, fnw: *const f32, scale: *const f32, base: *const f32,
                                    pre: *mut f32, post: *mut f32, comb: *mut f32, rows: i32, dim: i32,
                                    rms_eps: f32, hc_eps: f32, iters: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_hc_pre(x: *const f32, pre: *const f32, y: *mut c_void, rows: i32, dim: i32,
                                  stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_hc_post(x: *mut f32, y: *const c_void, y_dtype: i32, post: *const f32,
                                   comb: *const f32, rows: i32, dim: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_attn(q: *const c_void, kv: *const c_void, sink: *const f32, idxs: *const i32,
                                o: *mut c_void, block: i32, n_heads: i32, hd: i32, n_kv: i32, scale: f32,
                                stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_argmax(logits: *const f32, out: *mut i32, rows: i32, vocab: i32,
                                  stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_logits(x: *const c_void, w: *const u8, s: *const u8, y: *mut f32, m: i32, k: i32,
                                  n: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_logits_typed(x: *const c_void, x_dtype: i32, w: *const u8, s: *const u8, y: *mut f32,
                                        m: i32, k: i32, n: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_markov_bias_bf16(logits: *mut c_void, tokens: *const c_void, embed: *const c_void,
                                            head: *const c_void, rows: i32, vocab: i32, rank: i32,
                                            stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_gather_rows_bf16(dst: *mut f32, ids: *const i32, table: *const c_void, rows: i32,
                                            width: i32, vocab: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_confidence_bf16(h: *const c_void, m: *const c_void, w: *const c_void, b: *const c_void,
                                           out: *mut c_void, rows: i32, dim: i32, rank: i32,
                                           stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_confidence_keyed_bf16(h: *const c_void, m: *const c_void, w: *const c_void, b: *const c_void,
                                                 out: *mut c_void, rows: i32, dim: i32, rank: i32, walk_rows: i32,
                                                 stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_tap_mean_bf16(streams: *const f32, taps_bf16: *mut c_void, rows: i32, dim: i32,
                                        tap_idx: i32, n_taps: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_schedule_verify_len(
            conf: *const f32,
            conf_len: i32,
            min_conf: f32,
            max_drafts: i32,
            cycle_ms: *const f32,
            out: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_hc_collapse_f32(x: *const f32, pre: *const f32, y: *mut f32, rows: i32, dim: i32,
                                       stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_router(x: *const c_void, gate: *const u16, logits: *mut f32, rows: i32, dim: i32,
                                  experts: i32, temp: f32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_select(scores: *const f32, bias: *const f32, picks: *mut i32, wts: *mut c_void,
                                  rows: i32, experts: i32, topk: i32, scale: f32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_fp4_gateup(x: *const c_void, w1: *const u8, s1: *const u8, w3: *const u8,
                                      s3: *const u8, offsets: *const i32, wpack: *const c_void, act: *mut c_void,
                                      experts: i32, dim: i32, inter: i32, limit: f32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_fp4_down(act: *const c_void, w2: *const u8, s2: *const u8, offsets: *const i32,
                                    y: *mut c_void, experts: i32, dim: i32, inter: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_widen(src: *const c_void, dst: *mut f32, n: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_gather_rows(dst: *mut f32, ids: *const i32, table: *const f32, rows: i32, width: i32,
                                       vocab: i32, stream: *mut c_void) -> i32;
        pub fn gb_dsv4_mtp_fill_ids(ids: *mut i32, pre_mix: *mut f32, anchor: i32, noise: i32, block: i32,
                                    stream: *mut c_void) -> i32;
    }
}

fn i32_of(v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| CudaError(1))
}

/// `rows` ids from `table` into the drafter's `h` (one row per id, `nc` channels of `dim`), T_BF16 (`dtype` = 1).
///
/// The safe seam for the call `gb_model` makes for `Op::Embed`: in a stub build (`cfg(gb_no_cuda)`, no nvcc) the kernel is
/// not compiled, so this returns `CudaError(-1)` instead of leaving an undefined symbol for the linker to find.
///
/// # Safety
/// Every pointer must be a live device buffer on `stream` with the shapes above; `ids` is host or device per the kernel's contract.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mtp_embed(
    h: *mut core::ffi::c_void,
    ids: *const i32,
    table: *const core::ffi::c_void,
    rows: i32,
    nc: i32,
    dim: i32,
    vocab: i32,
    dtype: i32,
    stream: *mut core::ffi::c_void,
) -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    {
        CudaError::check(ffi::gb_dsv4_mtp_embed(h, ids, table, rows, nc, dim, vocab, dtype, stream))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (h, ids, table, rows, nc, dim, vocab, dtype, stream);
        Err(CudaError(-1))
    }
}

// ================================================================================================================================
// 1. a dense fp8 [n, k] weight
// ================================================================================================================================

/// An `[n, k]` E4M3 weight with an `[n/32, k/32]` E8M0 scale -- the trunk's fp8 layout ([`crate::fp8_linear`]) -- with
/// the raw pointers kept so the grouped output projection can be launched against it.
#[derive(Debug)]
pub struct MtpDense {
    w: DevBuf,
    s: DevBuf,
    n: usize,
    k: usize,
}

impl MtpDense {
    pub fn new(n: usize, k: usize, w: &[u8], s: &[u8]) -> Result<Self, CudaError> {
        if n == 0 || k == 0 || n % 32 != 0 || k % 32 != 0 || w.len() != n * k || s.len() != (n / 32) * (k / 32) {
            return Err(CudaError(1));
        }
        Ok(Self { w: DevBuf::from_slice(w)?, s: DevBuf::from_slice(s)?, n, k })
    }

    pub fn from_parts(n: usize, k: usize, w: DevBuf, s: DevBuf) -> Result<Self, CudaError> {
        if n == 0 || k == 0 || n % 32 != 0 || k % 32 != 0 || w.bytes() != n * k || s.bytes() != (n / 32) * (k / 32) {
            return Err(CudaError(1));
        }
        Ok(Self { w, s, n, k })
    }

    pub fn logits_typed(&self, x: &DevBuf, x_dtype: Dtype, y: &mut DevBuf, m: usize, st: &Stream) -> Result<(), CudaError> {
        if m == 0 || x.bytes() < m * self.k * x_dtype.size() || y.bytes() < m * self.n * 4 {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_dsv4_mtp_logits_typed(
                x.ptr(),
                x_dtype as i32,
                self.w.ptr() as *const u8,
                self.s.ptr() as *const u8,
                y.ptr() as *mut f32,
                i32_of(m)?,
                i32_of(self.k)?,
                i32_of(self.n)?,
                st.raw(),
            ))
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.n, self.k)
    }

    pub fn bytes(&self) -> u64 {
        (self.w.bytes() + self.s.bytes()) as u64
    }

    /// `y [m, n] = x [m, k] @ W^T`, fp16 out.
    pub fn forward(&self, x: &DevBuf, y: &mut DevBuf, m: usize, st: &Stream) -> Result<(), CudaError> {
        if m == 0 || x.bytes() < m * self.k * 2 || y.bytes() < m * self.n * 2 {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_fp8_linear(
                x.ptr(),
                Dtype::F16 as i32,
                self.w.ptr() as *const u8,
                self.s.ptr() as *const u8,
                y.ptr(),
                i32_of(m)?,
                i32_of(self.k)?,
                i32_of(self.n)?,
                0,
                st.raw(),
            ))
        }
    }

    /// Fused `y1 [m, n1] = w1(x [m, k])` and `y2 [m, n2] = w2(x [m, k])`, fp16 out.
    pub fn forward_pair(
        w1: &Self,
        y1: &mut DevBuf,
        w2: &Self,
        y2: &mut DevBuf,
        x: &DevBuf,
        m: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if m == 0 || w1.k != w2.k || x.bytes() < m * w1.k * 2 || y1.bytes() < m * w1.n * 2 || y2.bytes() < m * w2.n * 2 {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_fp8_linear_pair(
                x.ptr(),
                Dtype::F16 as i32,
                w1.w.ptr() as *const u8,
                w1.s.ptr() as *const u8,
                y1.ptr(),
                i32_of(w1.n)?,
                w2.w.ptr() as *const u8,
                w2.s.ptr() as *const u8,
                y2.ptr(),
                i32_of(w2.n)?,
                i32_of(m)?,
                i32_of(w1.k)?,
                0,
                st.raw(),
            ))
        }
    }

    /// `y [m, n]` **fp32** = `x @ W^T`. For the head: 129,280 logits that a bias is then added to, so an fp16 rounding
    /// at the GEMM output would be a rounding on the sampled token.
    pub fn logits(&self, x: &DevBuf, y: &mut DevBuf, m: usize, st: &Stream) -> Result<(), CudaError> {
        if m == 0 || m > 8 || x.bytes() < m * self.k * 2 || y.bytes() < m * self.n * 4 {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_dsv4_mtp_logits(
                x.ptr(),
                self.w.ptr() as *const u8,
                self.s.ptr() as *const u8,
                y.ptr() as *mut f32,
                i32_of(m)?,
                i32_of(self.k)?,
                i32_of(self.n)?,
                st.raw(),
            ))
        }
    }

    /// The grouped output projection: `out[m][g][o_lora] = o[m][g][o_in] @ W[g*o_lora ..][..]^T` over `groups` blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn woa_grouped(
        &self,
        o: &DevBuf,
        out: &mut DevBuf,
        m: usize,
        groups: usize,
        o_lora: usize,
        o_in: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if o.bytes() < m * groups * o_in * 2 || out.bytes() < m * groups * o_lora * 2 {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_dsv4_woa_grouped(
                o.ptr(),
                self.w.ptr() as *const u8,
                self.s.ptr() as *const u8,
                out.ptr(),
                i32_of(m)?,
                i32_of(groups)?,
                i32_of(o_lora)?,
                i32_of(o_in)?,
                Dtype::F16 as i32,
                0,
                0,
                st.raw(),
            ))
        }
    }
}

// ================================================================================================================================
// 2. the expert seam
// ================================================================================================================================

/// The pack's MXFP4 experts: `w1` / `w3` `[inter, dim]`, `w2` `[dim, inter]`, E2M1 with an E8M0 scale a row per 32.
pub struct Mxfp4Experts {
    w1: Mxfp4Weights,
    w3: Mxfp4Weights,
    w2: Mxfp4Weights,
    n: usize,
    dim: usize,
    inter: usize,
}

impl Mxfp4Experts {
    /// `w1` / `w3` are `[inter, dim]`, `w2` is `[dim, inter]`; each is `(bytes, scales)`.
    pub fn new(
        n: usize,
        dim: usize,
        inter: usize,
        w1: (&[u8], &[u8]),
        w3: (&[u8], &[u8]),
        w2: (&[u8], &[u8]),
    ) -> Result<Self, CudaError> {
        Ok(Self {
            // `[experts][inter][dim]` and `[experts][dim][inter]`
            w1: Mxfp4Weights::new(n * inter, dim, w1.0, w1.1)?,
            w3: Mxfp4Weights::new(n * inter, dim, w3.0, w3.1)?,
            w2: Mxfp4Weights::new(n * dim, inter, w2.0, w2.1)?,
            n,
            dim,
            inter,
        })
    }
}

impl Mxfp4Experts {
    pub fn n_experts(&self) -> usize {
        self.n
    }
    pub fn inter_dim(&self) -> usize {
        self.inter
    }
    pub fn weight_bytes(&self) -> u64 {
        (self.w1.weight_bytes() + self.w2.weight_bytes() + self.w3.weight_bytes()) as u64
    }
    pub fn workspace_bytes(&self, _slots: usize) -> u64 {
        0
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gate_up(
        &self,
        x: &DevBuf,
        act: &mut DevBuf,
        offsets: &DevBuf,
        wpack: &DevBuf,
        slots: usize,
        limit: f32,
        _ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if x.bytes() < slots * self.dim * 2 || act.bytes() < slots * self.inter * 2 || offsets.bytes() < (self.n + 1) * 4 {
            return Err(CudaError(1));
        }
        let (w1, s1) = self.w1.parts();
        let (w3, s3) = self.w3.parts();
        unsafe {
            CudaError::check(ffi::gb_dsv4_mtp_fp4_gateup(
                x.ptr(),
                w1,
                s1,
                w3,
                s3,
                offsets.ptr() as *const i32,
                wpack.ptr(),
                act.ptr(),
                i32_of(self.n)?,
                i32_of(self.dim)?,
                i32_of(self.inter)?,
                limit,
                st.raw(),
            ))
        }
    }
    pub fn down(
        &self,
        act: &DevBuf,
        y: &mut DevBuf,
        offsets: &DevBuf,
        slots: usize,
        _ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if act.bytes() < slots * self.inter * 2 || y.bytes() < slots * self.dim * 2 {
            return Err(CudaError(1));
        }
        let (w2, s2) = self.w2.parts();
        unsafe {
            CudaError::check(ffi::gb_dsv4_mtp_fp4_down(
                act.ptr(),
                w2,
                s2,
                offsets.ptr() as *const i32,
                y.ptr(),
                i32_of(self.n)?,
                i32_of(self.dim)?,
                i32_of(self.inter)?,
                st.raw(),
            ))
        }
    }
}

/// One stage's 128 routed experts, in whichever bit width the pack ships.
///
/// # The pack changed under this file
///
/// The first pack put the MTP experts in MXFP4 (`w1.weight` I8 + `w1.scale` E8M0 per 32 inputs). The re-uploaded
/// `-SAGE-EXL3-1.62bpw` pack re-quantised them to **EXL3 2.00 bpw (K2)**: per expert `w1.trellis` I16
/// `[320, 144, 32]` + `suh` F16 `[5120]` + `svh` F16 `[2304]` + `mul1` (and `w2` transposed), with no `.weight` / `.scale`
/// left. So the routed block runs through the trunk's own grouped path
/// ([`crate::dsv4_moe::moe_experts`] over a [`MoeScratch`]): one `MoeScratch`, one `offsets` copy a stage, three
/// `Exl3Weights` an active expert. The MXFP4 variant is kept because it is written and measured, not because a pack
/// currently needs it.
pub enum RoutedExperts {
    /// EXL3, the v2 pack: `w1` `[inter, dim]`, `w3` `[inter, dim]`, `w2` `[dim, inter]` an expert.
    Exl3(Vec<[Exl3Weights; 3]>),
    /// MXFP4: `w1` / `w3` `[inter, dim]`, `w2` `[dim, inter]`, E2M1 with an E8M0 scale a row per 32.
    Mxfp4(Mxfp4Experts),
}

impl RoutedExperts {
    pub fn n_experts(&self) -> usize {
        match self {
            RoutedExperts::Exl3(g) => g.len(),
            RoutedExperts::Mxfp4(m) => m.n_experts(),
        }
    }

    pub fn weight_bytes(&self) -> u64 {
        match self {
            RoutedExperts::Exl3(g) => g.iter().flat_map(|e| e.iter()).map(|w| w.weight_bytes() as u64).sum(),
            RoutedExperts::Mxfp4(m) => m.weight_bytes(),
        }
    }

    /// The extra gemv workspace the variant needs at `slots` rows. EXL3 has none of its own: its arena is the
    /// [`MoeScratch`], which `DsparkGpu` allocates and prices separately.
    pub fn workspace_bytes(&self, slots: usize) -> u64 {
        match self {
            RoutedExperts::Exl3(_) => 0,
            RoutedExperts::Mxfp4(m) => m.workspace_bytes(slots),
        }
    }
}

// ================================================================================================================================
// 3. one stage's weights
// ================================================================================================================================

/// The hyper-connection mixer of one sublayer: `hc_{attn,ffn}_fn` `[24, 4*dim]` f32, `_base` `[24]` f32, `_scale` `[3]`.
#[derive(Debug)]
pub struct HcPack {
    pub(crate) f: DevBuf,
    pub(crate) b: DevBuf,
    pub(crate) s: DevBuf,
}

impl HcPack {
    pub fn new(dim: usize, f: &[f32], b: &[f32], s: &[f32; 3]) -> Result<Self, CudaError> {
        if f.len() != 24 * 4 * dim || b.len() != 24 {
            return Err(CudaError(1));
        }
        Ok(Self { f: DevBuf::from_slice(f)?, b: DevBuf::from_slice(b)?, s: DevBuf::from_slice(s)? })
    }

    pub fn from_parts(dim: usize, f: DevBuf, b: DevBuf, s: DevBuf) -> Result<Self, CudaError> {
        if f.bytes() != 24 * 4 * dim * 4 || b.bytes() != 24 * 4 || s.bytes() != 3 * 4 {
            return Err(CudaError(1));
        }
        Ok(Self { f, b, s })
    }
    pub fn bytes(&self) -> u64 {
        (self.f.bytes() + self.b.bytes() + self.s.bytes()) as u64
    }
}

/// `Block.hc_mixes`: `pre` / `post` / `comb` of `rows` rows of `h [rows][4][dim]` fp32.
#[allow(clippy::too_many_arguments)]
pub fn hc_mixes_dev(
    h: &DevBuf,
    site: &HcPack,
    rows: usize,
    dim: usize,
    cfg: crate::dsv4_moe::HcCfg,
    pre: &mut DevBuf,
    post: &mut DevBuf,
    comb: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    unsafe {
        CudaError::check(ffi::gb_dsv4_mtp_hc_mixes(
            h.ptr() as *const f32,
            site.f.ptr() as *const f32,
            site.s.ptr() as *const f32,
            site.b.ptr() as *const f32,
            pre.ptr() as *mut f32,
            post.ptr() as *mut f32,
            comb.ptr() as *mut f32,
            i32_of(rows)?,
            i32_of(dim)?,
            cfg.rms_eps,
            cfg.hc_eps,
            cfg.sinkhorn_iters,
            st.raw(),
        ))
    }
}

/// `Block.hc_pre`: collapse the four streams with `pre_mix [rows][4]` into `y [rows, dim]` fp16.
pub fn hc_pre_dev(h: &DevBuf, pre_mix: &DevBuf, y: &mut DevBuf, rows: usize, dim: usize, st: &Stream) -> Result<(), CudaError> {
    unsafe {
        CudaError::check(ffi::gb_dsv4_mtp_hc_pre(
            h.ptr() as *const f32,
            pre_mix.ptr() as *const f32,
            y.ptr(),
            i32_of(rows)?,
            i32_of(dim)?,
            st.raw(),
        ))
    }
}

/// `Block.hc_post`: `h[i][d] <- post[i] y[d] + sum_g comb[g][i] h[g][d]`, in place.
pub fn hc_post_dev(
    h: &DevBuf,
    y: &DevBuf,
    post: &DevBuf,
    comb: &DevBuf,
    rows: usize,
    dim: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    unsafe {
        CudaError::check(ffi::gb_dsv4_mtp_hc_post(
            h.ptr() as *mut f32,
            y.ptr(),
            Dtype::F16 as i32,
            post.ptr() as *const f32,
            comb.ptr() as *const f32,
            i32_of(rows)?,
            i32_of(dim)?,
            st.raw(),
        ))
    }
}

/// Everything one stage needs at bind time.
pub struct MtpStagePack<'a> {
    pub stage: usize,
    /// Longest absolute position the RoPE table has to cover: `>= max_seq + block_size + 1` (the block sits at
    /// `start_pos + 1 ..`, so the table is indexed past the last trunk position).
    pub seq: usize,
    pub wq_a: (usize, usize, &'a [u8], &'a [u8]),
    pub wq_b: (usize, usize, &'a [u8], &'a [u8]),
    pub wkv: (usize, usize, &'a [u8], &'a [u8]),
    pub wo_a: (usize, usize, &'a [u8], &'a [u8]),
    pub wo_b: (usize, usize, &'a [u8], &'a [u8]),
    pub q_norm: &'a [f32],
    pub kv_norm: &'a [f32],
    pub sink: &'a [f32],
    /// `ffn.gate.weight` as bf16 bits `[experts, dim]`.
    pub gate_bf16: &'a [u16],
    pub gate_bias: &'a [f32],
    pub routed: RoutedExperts,
    pub shared_w1: (usize, usize, &'a [u8], &'a [u8]),
    pub shared_w3: (usize, usize, &'a [u8], &'a [u8]),
    pub shared_w2: (usize, usize, &'a [u8], &'a [u8]),
    pub hc_attn: (&'a [f32], &'a [f32], &'a [f32; 3]),
    pub hc_ffn: (&'a [f32], &'a [f32], &'a [f32; 3]),
    pub attn_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    /// Stage 0: `main_proj` `[dim, taps * dim]`.
    pub main_proj: Option<(usize, usize, &'a [u8], &'a [u8])>,
    /// Stage 0: `main_norm` `[dim]`.
    pub main_norm: Option<&'a [f32]>,
    /// Last stage: the final norm `[dim]`.
    pub norm: Option<&'a [f32]>,
    /// Last stage: the markov token embedding `[vocab, rank]`, fp32.
    pub markov_embed: Option<&'a [f32]>,
    /// Last stage: the markov head `[vocab, rank]`, fp32.
    pub markov_head: Option<&'a [f32]>,
    /// Last stage: the markov token embedding `[vocab, rank]` pre-packed bf16 (the shipped pack's dtype).
    /// When present it wins over `markov_embed`, which is what lets a device draft round ride the keyed
    /// Gumbel walk (the fast BF16 path) instead of the argmax fallback loop.
    pub markov_embed_bf16: Option<&'a [u16]>,
    /// Last stage: the markov head `[vocab, rank]` pre-packed bf16 (see `markov_embed_bf16`).
    pub markov_head_bf16: Option<&'a [u16]>,
    /// Last stage: the confidence projection `[dim + rank]`, fp32. Picked up as bf16 by the fast BF16 path
    /// (see `MtpStage::new`); read as f32 by the slow path's `gb_dsv4_mtp_confidence`.
    pub conf_proj: Option<&'a [f32]>,
    /// Last stage: `mtp[-1].head`, the trunk's `ParallelHead` `[vocab, dim]` as fp8.
    pub lm_head: Option<(usize, usize, &'a [u8], &'a [u8])>,
}

fn dense(t: (usize, usize, &[u8], &[u8])) -> Result<MtpDense, CudaError> {
    MtpDense::new(t.0, t.1, t.2, t.3)
}

/// One `DSparkBlock` on the device: an MLA layer plus a top-3 MoE over `n_routed_experts` experts, both between
/// hyper-connections, with its own window ring. Stage 0 additionally owns `main_proj` / `main_norm`; the last stage owns
/// `norm`, the markov head and the confidence head -- exactly `if stage_id == 0` / `if stage_id == n_mtp_layers - 1`.
pub struct MtpStage {
    pub stage: usize,
    // attention
    pub wq_a: MtpDense,
    pub wq_b: MtpDense,
    pub wkv: MtpDense,
    pub wo_a: MtpDense,
    pub wo_b: MtpDense,
    pub q_norm: DevBuf,
    pub kv_norm: DevBuf,
    pub sink: DevBuf,
    pub rope: RopeTables,
    pub window: DevBuf,
    // feed-forward
    pub gate: DevBuf,
    pub gate_bias: DevBuf,
    pub routed: RoutedExperts,
    pub shared_w1: MtpDense,
    pub shared_w3: MtpDense,
    pub shared_w2: MtpDense,
    // hyper-connections + the two sublayer norms
    pub hc_attn: HcPack,
    pub hc_ffn: HcPack,
    pub attn_norm: DevBuf,
    pub ffn_norm: DevBuf,
    // stage 0
    pub main_proj: Option<MtpDense>,
    pub main_norm: Option<DevBuf>,
    // last stage
    pub norm: Option<DevBuf>,
    pub markov_embed: Option<DevBuf>,
    pub markov_head: Option<DevBuf>,
    pub conf_proj: Option<DevBuf>,
    pub conf_bias: Option<DevBuf>,
    /// `mtp[-1].head`: the trunk's `ParallelHead` `[vocab, dim]`, fp8.
    pub lm_head: Option<MtpDense>,
    /// `mtp[-1].head` when stored as EXL3 (`K=8` in the v2 pack).
    pub lm_head_exl3: Option<Exl3Weights>,
}

impl MtpStage {
    pub fn new(cfg: &MtpConfig, p: MtpStagePack<'_>) -> Result<Self, CudaError> {
        let last = p.stage + 1 == cfg.n_stages;
        Ok(Self {
            stage: p.stage,
            wq_a: dense(p.wq_a)?,
            wq_b: dense(p.wq_b)?,
            wkv: dense(p.wkv)?,
            wo_a: dense(p.wo_a)?,
            wo_b: dense(p.wo_b)?,
            q_norm: DevBuf::from_slice(p.q_norm)?,
            kv_norm: DevBuf::from_slice(p.kv_norm)?,
            sink: DevBuf::from_slice(p.sink)?,
            rope: RopeTables::new(RopeSpec {
                dim: cfg.rope_dim,
                seq: p.seq,
                original_seq_len: 0,
                base: cfg.rope_theta,
                factor: 1.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
            })?,
            window: DevBuf::zeroed(cfg.window * cfg.head_dim * 2)?,
            gate: DevBuf::from_slice(p.gate_bf16)?,
            gate_bias: DevBuf::from_slice(p.gate_bias)?,
            routed: p.routed,
            shared_w1: dense(p.shared_w1)?,
            shared_w3: dense(p.shared_w3)?,
            shared_w2: dense(p.shared_w2)?,
            hc_attn: HcPack::new(cfg.dim, p.hc_attn.0, p.hc_attn.1, p.hc_attn.2)?,
            hc_ffn: HcPack::new(cfg.dim, p.hc_ffn.0, p.hc_ffn.1, p.hc_ffn.2)?,
            attn_norm: DevBuf::from_slice(p.attn_norm)?,
            ffn_norm: DevBuf::from_slice(p.ffn_norm)?,
            main_proj: p.main_proj.map(dense).transpose()?,
            main_norm: p.main_norm.map(DevBuf::from_slice).transpose()?,
            norm: p.norm.map(DevBuf::from_slice).transpose()?,
            markov_embed: p
                .markov_embed_bf16
                .map(DevBuf::from_slice)
                .or_else(|| p.markov_embed.map(DevBuf::from_slice))
                .transpose()?,
            markov_head: p
                .markov_head_bf16
                .map(DevBuf::from_slice)
                .or_else(|| p.markov_head.map(DevBuf::from_slice))
                .transpose()?,
            conf_proj: match (&p.conf_proj, &p.markov_embed_bf16) {
                // fast BF16 path: the confidence kernel reads `proj` as bf16 (`u16`), so pack the f32 slice the
                // way the shipped model's `confidence_head.proj.weight` already is
                (Some(cp), Some(_)) => {
                    let packed: Vec<u16> = cp.iter().map(|&x| (x.to_bits() >> 16) as u16).collect();
                    Some(DevBuf::from_slice(&packed)?)
                }
                // slow fp32 path: `gb_dsv4_mtp_confidence` reads `proj` as f32
                (Some(cp), None) => Some(DevBuf::from_slice(cp)?),
                _ => None,
            },
            conf_bias: if last { Some(DevBuf::from_slice(&[0f32])?) } else { None },
            lm_head: p.lm_head.map(dense).transpose()?,
            lm_head_exl3: None,
        })
    }

    pub fn weight_bytes(&self) -> u64 {
        let mut t = 0u64;
        for d in [&self.wq_a, &self.wq_b, &self.wkv, &self.wo_a, &self.wo_b, &self.shared_w1, &self.shared_w3, &self.shared_w2] {
            t += d.bytes();
        }
        if let Some(d) = &self.main_proj {
            t += d.bytes();
        }
        if let Some(d) = &self.lm_head {
            t += d.bytes();
        }
        for d in [&self.markov_embed, &self.markov_head].into_iter().flatten() {
            t += d.bytes() as u64;
        }
        t += self.routed.weight_bytes();
        t += (self.q_norm.bytes() + self.kv_norm.bytes() + self.sink.bytes() + self.gate.bytes() + self.gate_bias.bytes()) as u64;
        t += (self.attn_norm.bytes() + self.ffn_norm.bytes() + self.window.bytes()) as u64;
        t += self.hc_attn.bytes() + self.hc_ffn.bytes();
        for d in [&self.main_norm, &self.norm, &self.conf_proj, &self.conf_bias].into_iter().flatten() {
            t += d.bytes() as u64;
        }
        t
    }
}

// ================================================================================================================================
// 4. scratch
// ================================================================================================================================

const SEG: usize = 512;
const NSEG: usize = 34;

struct Layout {
    o: [usize; NSEG],
    total: usize,
}

impl Layout {
    fn of(cfg: &MtpConfig) -> Self {
        let (d, hd, nh, blk, win) = (cfg.dim, cfg.head_dim, cfg.n_heads, cfg.block_size, cfg.window);
        let (e, topk, inter, vocab, rank) = (cfg.n_routed_experts, cfg.topk, cfg.inter_dim, cfg.vocab, cfg.markov_rank);
        let slots = blk * topk;
        let n_kv = win + blk;
        let sizes = [
            blk * 4 * d * 4,                                         // 0  h        fp32 residual streams
            blk * 4 * 4,                                             // 1  pre_mix  (threaded through the stages)
            blk * 4 * 4,                                             // 2  pre      (this block's attention mix)
            blk * 4 * 4,                                             // 3  post
            blk * 16 * 4,                                            // 4  comb
            blk * d * 2,                                             // 5  xn / xf  fp16 sublayer input
            blk * d * 2,                                             // 6  y        fp16 sublayer output
            blk * d * 2,                                             // 7  hc16     collapsed stream
            blk * d * 4,                                             // 8  hc32     widened, for the confidence head
            blk * d * 2,                                             // 9  main_x (one row a prefill chunk)
            blk * cfg.q_lora.max(cfg.o_lora).max(cfg.inter_dim) * 2, // 10 qr / shared gate
            blk * nh * hd * 2,                                       // 11 q
            blk * hd * 2,                                            // 12 kv (block rows)
            n_kv * hd * 2,                                           // 13 combined
            blk * nh * hd * 2,                                       // 14 o
            blk * cfg.o_groups * cfg.o_lora * 2,                     // 15 olora
            blk * e * 4,                                             // 16 router logits
            e * 4,                                                   // 17 counts
            (e + 1) * 4,                                             // 18 offsets
            slots * 4,                                               // 19 picks
            slots * 2,                                               // 20 wts
            slots * 4,                                               // 21 perm
            slots * 4,                                               // 22 pinv
            slots * 2,                                               // 23 wpack
            slots * d * 2,                                           // 24 xs
            slots * inter * 2,                                       // 25 act
            slots * d * 2,                                           // 26 yp
            blk * d * 2,                                             // 27 shared
            blk * d * 4,                                             // 28 moe f32
            blk * vocab * 4,                                         // 29 logits
            (blk + 1) * 4 + blk * rank * 4 + blk * 4 + 4,            // 30 ids, markov embeds, conf, acc
            blk * n_kv * 4,                                          // 31 topk idxs
            blk * 4,                                                 // 32 draft ids (anchor + noise)
            blk * 4,                                                 // 33 padding for a non-32-aligned rank buffer
        ];
        let mut o = [0usize; NSEG];
        let mut at = 0usize;
        for (i, s) in sizes.iter().enumerate() {
            o[i] = at;
            at += s.div_ceil(SEG) * SEG;
        }
        Self { o, total: at }
    }
}

/// Every buffer one draft round touches, in one arena: nothing allocates during a request and every segment is a
/// [`DevBuf::view`], so a segment is a pointer offset.
pub struct MtpScratch {
    rows: usize,
    arena: std::sync::Arc<Block>,
    // Every segment below is public: a test (and a debug dump) reads one piece at a time to say *where* a chain
    // diverged, which a single end-to-end comparison cannot. The serve touches `ids` / `conf` through `read`.
    pub h: DevBuf,
    pub pre_mix: DevBuf,
    pub pre: DevBuf,
    pub post: DevBuf,
    pub comb: DevBuf,
    pub xn: DevBuf,
    pub y: DevBuf,
    pub hc16: DevBuf,
    pub hc32: DevBuf,
    pub main_x: DevBuf,
    pub qr: DevBuf,
    pub q: DevBuf,
    pub kv: DevBuf,
    pub combined: DevBuf,
    pub o: DevBuf,
    pub olora: DevBuf,
    pub router: DevBuf,
    pub counts: DevBuf,
    pub offsets: DevBuf,
    pub picks: DevBuf,
    pub wts: DevBuf,
    pub perm: DevBuf,
    pub pinv: DevBuf,
    pub wpack: DevBuf,
    pub xs: DevBuf,
    pub act: DevBuf,
    pub yp: DevBuf,
    pub shared: DevBuf,
    pub moe32: DevBuf,
    pub logits: DevBuf,
    pub ids: DevBuf,
    pub mem: DevBuf,
    pub conf: DevBuf,
    pub acc: DevBuf,
    pub idxs: DevBuf,
    pub draft_ids: DevBuf,
}

impl MtpScratch {
    pub fn bytes(cfg: &MtpConfig) -> u64 {
        Layout::of(cfg).total as u64
    }

    pub fn new(cfg: &MtpConfig) -> Result<Self, CudaError> {
        let l = Layout::of(cfg);
        let arena = std::sync::Arc::new(Block::alloc(l.total)?);
        let (d, hd, nh, blk, win) = (cfg.dim, cfg.head_dim, cfg.n_heads, cfg.block_size, cfg.window);
        let (e, topk, inter, vocab, rank) = (cfg.n_routed_experts, cfg.topk, cfg.inter_dim, cfg.vocab, cfg.markov_rank);
        let slots = blk * topk;
        let n_kv = win + blk;
        let o = l.o;
        let v = |off: usize, bytes: usize| DevBuf::view(&arena, off, bytes);
        let (ids, conf, acc) = if let Ok(mb) = Block::alloc_mapped_host(((blk + 1) * 4 + blk * 4 + 4).div_ceil(256) * 256) {
            let mb = std::sync::Arc::new(mb);
            (
                DevBuf::view(&mb, 0, (blk + 1) * 4),
                DevBuf::view(&mb, (blk + 1) * 4, blk * 4),
                DevBuf::view(&mb, (blk + 1) * 4 + blk * 4, 4),
            )
        } else {
            (
                v(o[30], (blk + 1) * 4),
                v(o[30] + (blk + 1) * 4 + blk * rank * 4, blk * 4),
                v(o[30] + (blk + 1) * 4 + blk * rank * 4 + blk * 4, 4),
            )
        };
        let mut s = Self {
            rows: blk,
            h: v(o[0], blk * 4 * d * 4),
            pre_mix: v(o[1], blk * 4 * 4),
            pre: v(o[2], blk * 4 * 4),
            post: v(o[3], blk * 4 * 4),
            comb: v(o[4], blk * 16 * 4),
            xn: v(o[5], blk * d * 2),
            y: v(o[6], blk * d * 2),
            hc16: v(o[7], blk * d * 2),
            hc32: v(o[8], blk * d * 4),
            main_x: v(o[9], blk * d * 2),
            qr: v(o[10], blk * cfg.q_lora.max(cfg.o_lora).max(cfg.inter_dim) * 2),
            q: v(o[11], blk * nh * hd * 2),
            kv: v(o[12], blk * hd * 2),
            combined: v(o[13], n_kv * hd * 2),
            o: v(o[14], blk * nh * hd * 2),
            olora: v(o[15], blk * cfg.o_groups * cfg.o_lora * 2),
            router: v(o[16], blk * e * 4),
            counts: v(o[17], e * 4),
            offsets: v(o[18], (e + 1) * 4),
            picks: v(o[19], slots * 4),
            wts: v(o[20], slots * 2),
            perm: v(o[21], slots * 4),
            pinv: v(o[22], slots * 4),
            wpack: v(o[23], slots * 2),
            xs: v(o[24], slots * d * 2),
            act: v(o[25], slots * inter * 2),
            yp: v(o[26], slots * d * 2),
            shared: v(o[27], blk * d * 2),
            moe32: v(o[28], blk * d * 4),
            logits: v(o[29], blk * vocab * 4),
            ids,
            mem: v(o[30] + (blk + 1) * 4, blk * rank * 4),
            conf,
            acc,
            idxs: v(o[31], blk * n_kv * 4),
            draft_ids: v(o[32], blk * 4),
            arena,
        };
        // `make_identity_pre_mix`: stream 0 hot. A constant of the architecture, written once at bind.
        let mut pm = vec![0f32; blk * 4];
        for r in 0..blk {
            pm[r * 4] = 1.0;
        }
        s.pre_mix.upload(&pm)?;
        Ok(s)
    }

    pub fn bytes_held(&self) -> u64 {
        self.arena.bytes() as u64
    }
}

// ================================================================================================================================
// 5. the chain
// ================================================================================================================================

/// Maximum number of concurrent slots with independent MTP window rings.
pub const MAX_SLOTS: usize = 16;

/// The three chained stages plus the head: `Transformer.forward_spec`.
///
/// The pinned host staging of one round's read-back ([`DsparkGpu::read_begin`]): two mapped-host blocks (ids and
/// confidences) plus the event that marks the async D2H copies complete. Allocated lazily on first use.
struct DraftReadbackStage {
    ids: Block,
    conf: Block,
    sched: Block,
    ev: Event,
}

impl DraftReadbackStage {
    fn new(block_size: usize) -> Result<Self, CudaError> {
        Ok(Self {
            ids: Block::alloc_mapped_host((block_size + 1) * 4)?,
            conf: Block::alloc_mapped_host(block_size * 4)?,
            sched: Block::alloc_mapped_host(4)?,
            ev: Event::new()?,
        })
    }
}

pub struct DsparkGpu {
    pub cfg: MtpConfig,
    pub stages: Vec<MtpStage>,
    /// Per-stage, per-slot window rings `[n_stages][MAX_SLOTS]` (`slot 0` aliases `stages[s].window`).
    pub slot_windows: Vec<Vec<DevBuf>>,
    pub scratch: MtpScratch,
    /// The routed experts' gemv workspace (zero bytes for MXFP4, which needs none).
    ws: DevBuf,
    /// The trunk's grouped routed block, for an EXL3 pack: group / gather / per-expert gemv / scatter, one arena.
    ms: Option<MoeScratch>,
    /// Fast resident descriptor table per stage when all experts of the stage are EXL3 Mul1 1..=4 bits.
    ms_descs: Vec<Option<DevBuf>>,
    /// `mtp[i].embed` is the trunk's `Transformer.embed` (`[vocab, dim]` f16 or bf16 bits), borrowed, not owned.
    trunk_embed: DevBuf,
    embed_dtype: Dtype,
    taps_dtype: Dtype,
    /// Active slot (`0..MAX_SLOTS`) for `stage_forward` / `attn` / `prefill_seed` / `draft` calls.
    pub active_slot: usize,
    /// Last position already seeded into every stage's window ring for each slot `0..MAX_SLOTS`.
    pub seeded_pos: [Option<usize>; MAX_SLOTS],
    /// Pinned host staging + completion event for the round's async D2H read-back ([`Self::read_begin`]).
    stage: Option<DraftReadbackStage>,
    /// Device copy of [`Self::VERIFY_CYCLE_MS`] and the one-int schedule result (lazily allocated).
    sched_cycle: Option<DevBuf>,
    sched_out: Option<DevBuf>,
    /// Device-resident round params ([`Dsv4MtpRoundParams`]) for the draft-round graphs, and the pinned host
    /// staging the round's async H2D copy uploads from (lazily allocated on first graph-eligible round).
    round_params_dev: Option<DevBuf>,
    round_params_host: Option<Block>,
    /// The captured draft-round graphs, per `(owner, slot, seeded, keyed, walk_rows)`.
    round_graphs: Option<HashMap<DraftRoundKey, Graph>>,
    /// Consecutive capture/launch refusals; past [`REFUSAL_DISABLE_AT`] the feature is disabled for the process.
    graph_refusals: u32,
}

impl DsparkGpu {
    /// `trunk_embed` is the trunk's embedding table as f16 bits (`[vocab, dim]`): every stage is assigned it, and stage
    /// 0's `forward_embed` reads it.
    pub fn new(cfg: MtpConfig, stages: Vec<MtpStage>, trunk_embed: &[u16]) -> Result<Self, CudaError> {
        Self::from_parts(cfg, stages, DevBuf::from_slice(trunk_embed)?, Dtype::F16, Dtype::F16)
    }

    pub fn from_parts(
        cfg: MtpConfig,
        mut stages: Vec<MtpStage>,
        trunk_embed: DevBuf,
        embed_dtype: Dtype,
        taps_dtype: Dtype,
    ) -> Result<Self, CudaError> {
        let last = cfg.n_stages.saturating_sub(1);
        if stages.len() != cfg.n_stages
            || stages[0].main_proj.is_none()
            || stages[last].markov_embed.is_none()
            || (stages[last].lm_head.is_none() && stages[last].lm_head_exl3.is_none())
        {
            return Err(CudaError(1));
        }
        let win_bytes = cfg.window * cfg.head_dim * 2;
        let mut slot_windows = Vec::with_capacity(stages.len());
        for stg in &mut stages {
            let arena = DevBuf::zeroed_view(MAX_SLOTS * win_bytes)?;
            stg.window = arena.subview(0, win_bytes);
            let per_slot: Vec<DevBuf> = (0..MAX_SLOTS).map(|slot| arena.subview(slot * win_bytes, win_bytes)).collect();
            slot_windows.push(per_slot);
        }
        let slots = cfg.block_size * cfg.topk;
        let mut ws = stages.iter().map(|s| s.routed.workspace_bytes(slots)).max().unwrap_or(0);
        let exl3 = stages.iter().any(|s| matches!(s.routed, RoutedExperts::Exl3(_)));
        let mut ms = if exl3 {
            Some(MoeScratch::new(cfg.block_size, cfg.topk, cfg.dim, cfg.inter_dim, cfg.n_routed_experts)?)
        } else {
            None
        };
        let mut ms_descs = Vec::with_capacity(stages.len());
        for stg in &stages {
            match &stg.routed {
                RoutedExperts::Exl3(groups) => {
                    let refs: Vec<&[Exl3Weights]> = groups.iter().map(|g| g.as_slice()).collect();
                    ms_descs.push(crate::dsv4_moe::build_expert_descs(&refs, cfg.dim, cfg.inter_dim)?);
                }
                RoutedExperts::Mxfp4(_) => ms_descs.push(None),
            }
        }
        let scratch = MtpScratch::new(&cfg)?;
        // Pre-warm the EXL3 head linear's workspace to the largest a draft round can request (walk_rows ==
        // block_size), so a graph capture of the round never hits an allocation (which would refuse the capture).
        {
            let mut ms_mut = ms.as_mut();
            if let Some(last) = stages.last() {
                if let Some(hw) = &last.lm_head_exl3 {
                    let need = crate::exl3_linear::linear_workspace_bytes(cfg.block_size, cfg.dim, cfg.vocab, hw.bits);
                    if let Some(m) = ms_mut.as_mut() {
                        m.ensure_workspace(need)?;
                    } else if (ws as usize) < need {
                        ws = need as u64;
                    }
                }
            }
        }
        // Pre-warm the resident routed-experts workspace (the `MoeScratch` tail is used when it suffices; when it
        // does not, `ensure_workspace` grows `ws` with a `cudaMalloc`, which is illegal inside a stream capture. The
        // draft's port levels are the largest before round ever sees — `rows <= block_size`, `slots = rows*topk` —
        // so warming to that size now makes every capture allocation-free.
        {
            let align256 = |x: usize| (x + 255) & !255;
            let slots = cfg.block_size * cfg.topk;
            let z_slots = if cfg.block_size <= 16 { slots } else { 0 };
            let need =
                2 * align256(slots * 4) + 256 + align256(2 * slots * cfg.dim * 2) + align256(4 * z_slots * cfg.inter_dim * 4);
            if let Some(m) = ms.as_mut() {
                m.ensure_workspace(need)?;
            }
        }
        Ok(Self {
            cfg,
            stages,
            slot_windows,
            scratch,
            ws: DevBuf::zeroed(ws.max(1) as usize)?,
            ms,
            ms_descs,
            trunk_embed,
            embed_dtype,
            taps_dtype,
            active_slot: 0,
            seeded_pos: [None; MAX_SLOTS],
            stage: None,
            sched_cycle: None,
            sched_out: None,
            round_params_dev: None,
            round_params_host: None,
            round_graphs: None,
            graph_refusals: 0,
        })
    }

    /// Device bytes one draft round's scratch needs.
    pub fn bytes(cfg: &MtpConfig) -> u64 {
        MtpScratch::bytes(cfg)
    }

    pub fn weight_bytes(&self) -> u64 {
        let extra_slot_win = (MAX_SLOTS.saturating_sub(1) * self.stages.len() * self.cfg.window * self.cfg.head_dim * 2) as u64;
        self.stages.iter().map(|s| s.weight_bytes()).sum::<u64>()
            + extra_slot_win
            + (self.ws.bytes() + self.trunk_embed.bytes()) as u64
            + self.ms.as_ref().map_or(0, |m| m.bytes())
    }

    /// `forward_embed` (stage 0): `main_x = main_norm(main_proj(taps))`, then the block embedding.
    pub fn forward_embed(&mut self, taps: &DevBuf, anchor: i32, st: &Stream) -> Result<(), CudaError> {
        self.forward_embed_opt(Some(taps), anchor, None, st)
    }

    fn forward_embed_opt(
        &mut self,
        taps: Option<&DevBuf>,
        anchor: i32,
        params: Option<&DevBuf>,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let cfg = self.cfg.clone();
        let s0 = &self.stages[0];
        let sc = &mut self.scratch;
        if let Some(taps) = taps {
            if self.taps_dtype == Dtype::F16 {
                s0.main_proj.as_ref().unwrap().forward(taps, &mut sc.main_x, 1, st)?;
                rmsnorm_dev(&sc.main_x, s0.main_norm.as_ref().unwrap(), 1, cfg.dim, cfg.eps, st)?;
            } else {
                s0.main_proj.as_ref().unwrap().logits_typed(taps, self.taps_dtype, &mut sc.hc32, 1, st)?;
                crate::dsv4_attn::rmsnorm_f32_to_f16_dev(
                    &sc.hc32,
                    s0.main_norm.as_ref().unwrap(),
                    &mut sc.main_x,
                    1,
                    cfg.dim,
                    cfg.eps,
                    st,
                )?;
            }
        }
        // `[anchor, NOISE x (block_size - 1)]`, built on the device so a captured round carries no H2D copy.
        unsafe {
            match params {
                Some(p) => CudaError::check(ffi::gb_dsv4_mtp_fill_ids_params(
                    sc.ids.ptr() as *mut i32,
                    sc.pre_mix.ptr() as *mut f32,
                    p.ptr(),
                    cfg.noise_token_id,
                    i32_of(cfg.block_size)?,
                    st.raw(),
                ))?,
                None => CudaError::check(ffi::gb_dsv4_mtp_fill_ids(
                    sc.ids.ptr() as *mut i32,
                    sc.pre_mix.ptr() as *mut f32,
                    anchor,
                    cfg.noise_token_id,
                    i32_of(cfg.block_size)?,
                    st.raw(),
                ))?,
            }
            CudaError::check(ffi::gb_dsv4_mtp_embed(
                sc.h.ptr(),
                sc.ids.ptr() as *const i32,
                self.trunk_embed.ptr(),
                i32_of(cfg.block_size)?,
                4,
                i32_of(cfg.dim)?,
                i32_of(cfg.vocab)?,
                self.embed_dtype as i32,
                st.raw(),
            ))
        }
    }

    /// One draft round: `Transformer.forward_spec` for `start_pos > 0`. Launches only -- no host read, no allocation.
    pub fn draft(&mut self, taps: &DevBuf, anchor: i32, start_pos: usize, st: &Stream) -> Result<(), CudaError> {
        let bs = self.cfg.block_size;
        self.draft_n(taps, anchor, start_pos, bs, st)
    }

    /// One draft round with the sequential Markov walk capped at `max_drafts` (`1..=block_size`).
    pub fn draft_n(
        &mut self,
        taps: &DevBuf,
        anchor: i32,
        start_pos: usize,
        max_drafts: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.draft_n_sampled(taps, anchor, start_pos, max_drafts, None, st)
    }

    /// One draft round with optional keyed Gumbel sampling (`sampling.temperature > 0.0`).
    pub fn draft_n_sampled(
        &mut self,
        taps: &DevBuf,
        anchor: i32,
        start_pos: usize,
        max_drafts: usize,
        sampling: Option<MtpWalkSampling>,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let slot = self.active_slot;
        self.draft_n_keyed_slot(slot, taps, anchor, start_pos, max_drafts, sampling, st)
    }

    /// One draft round for `slot` (`0..MAX_SLOTS`) with the sequential Markov walk capped at `max_drafts` (`1..=block_size`).
    pub fn draft_n_slot(
        &mut self,
        slot: usize,
        taps: &DevBuf,
        anchor: i32,
        start_pos: usize,
        max_drafts: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.draft_n_keyed_slot(slot, taps, anchor, start_pos, max_drafts, None, st)
    }

    /// [`Self::draft_n`] that also captures the round's candidate lattice (T2.5a, the SPEC_TREE playbook): the tree
    /// search of a draft-tree round is host-side, so between the head's projection and the Markov walk — the only
    /// point where the per-level logits are still the PARENT-INDEPENDENT unary scores, before the walk biases them in
    /// place — this reads the walked rows back and assembles [`DsparkLattice`].
    pub fn draft_n_lattice(
        &mut self,
        taps: &DevBuf,
        anchor: i32,
        start_pos: usize,
        max_drafts: usize,
        k: usize,
        st: &Stream,
    ) -> Result<Option<DsparkLattice>, CudaError> {
        let slot = self.active_slot;
        self.draft_n_keyed_slot_capture(slot, taps, anchor, start_pos, max_drafts, None, Some(k), st)
    }

    /// One draft round for `slot` (`0..MAX_SLOTS`) with optional keyed Gumbel sampling (`sampling.temperature > 0.0`).
    #[allow(clippy::too_many_arguments)]
    pub fn draft_n_keyed_slot(
        &mut self,
        slot: usize,
        taps: &DevBuf,
        anchor: i32,
        start_pos: usize,
        max_drafts: usize,
        sampling: Option<MtpWalkSampling>,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.draft_n_keyed_slot_capture(slot, taps, anchor, start_pos, max_drafts, sampling, None, st).map(|_| ())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draft_n_keyed_slot_capture(
        &mut self,
        slot: usize,
        taps: &DevBuf,
        anchor: i32,
        start_pos: usize,
        max_drafts: usize,
        sampling: Option<MtpWalkSampling>,
        capture_k: Option<usize>,
        st: &Stream,
    ) -> Result<Option<DsparkLattice>, CudaError> {
        if start_pos == 0 || slot >= MAX_SLOTS {
            return Err(CudaError(1)); // prefill seeds the window; see `prefill_seed_slot`
        }
        self.active_slot = slot;
        let already_seeded = self.seeded_pos[slot] == Some(start_pos);
        let reach = self.cfg.window.min(start_pos + 1);
        let n_kv = reach + self.cfg.block_size;
        let walk_rows = max_drafts.clamp(1, self.cfg.block_size);
        let keyed = sampling.is_some_and(|s| s.temperature > 0.0);

        // Prepare the device round-params block (async H2D from pinned staging, ordered before any launch on `st`).
        self.prepare_round_params(anchor, start_pos, reach, n_kv, sampling, st)?;
        // An owned view of the params block, so the body can be called with `&mut self` (no borrow of `self` held).
        let params_view = {
            let blk = self.round_params_dev.as_ref().unwrap().block().expect("round params has a block").clone();
            DevBuf::view(&blk, 0, Dsv4MtpRoundParams::SIZE)
        };
        let params = Some(&params_view);

        // Graph path eligibility: no lattice read-back (`capture_k` none), the fast resident BF16-Markov walk, and
        // every stage's experts resident (no streamed/grouped fallback whose launch set could vary per round).
        let fast_markov = {
            let cfg = &self.cfg;
            let last = &self.stages[cfg.n_stages - 1];
            let embed = last.markov_embed.as_ref().unwrap();
            let bf16_markov = embed.bytes() == cfg.vocab * cfg.markov_rank * 2;
            let sc = &self.scratch;
            bf16_markov
                && cfg.markov_rank == 256
                && cfg.vocab >= 4096
                && sc.q.bytes() >= walk_rows * 1024 * 4
                && sc.o.bytes() >= 1024 * 4
                && crate::dsv4_moe::runtime_env("REXL3_DSV41_MTP_FULL_MARKOV").map(|v| v.trim() != "1").unwrap_or(true)
        };
        let resident = self.ms_descs.iter().all(Option::is_some);
        let graph_eligible =
            capture_k.is_none() && fast_markov && resident && round_graphs_enabled() && self.graph_refusals < REFUSAL_DISABLE_AT;

        if graph_eligible {
            let key = DraftRoundKey {
                owner: self.scratch.arena.ptr_at(0) as u64,
                slot: slot as u16,
                seeded: already_seeded,
                keyed,
                walk_rows: walk_rows as u16,
            };
            // Replay when a graph for this key exists. Pop it out so no borrow of `self.round_graphs` is held while
            // the launch/body runs; re-insert it after a successful launch.
            if let Some(g) = self.round_graphs.as_mut().and_then(|m| m.remove(&key)) {
                let r = g.launch(st);
                let map = self.round_graphs.get_or_insert_with(HashMap::new);
                match r {
                    Ok(()) => {
                        map.insert(key, g);
                        self.seeded_pos[slot] = Some(start_pos);
                        return Ok(None);
                    }
                    Err(e) if crate::graph::is_capture_code(e.0) => {
                        map.insert(key, g);
                        self.graph_refusals += 1;
                        return self.draft_round_body(slot, taps, anchor, start_pos, max_drafts, sampling, capture_k, params, st);
                    }
                    Err(e) => return Err(e),
                }
            }
            // Capture on first use; run the SAME body, then launch immediately (a capture alone runs nothing).
            match Graph::begin(st) {
                Ok(cap) => {
                    let ran =
                        self.draft_round_body(slot, taps, anchor, start_pos, max_drafts, sampling, None, params, st).map(|_| ());
                    match cap.finish(ran) {
                        Ok(g) => {
                            match g.launch(st) {
                                Ok(()) => {
                                    self.graph_refusals = 0;
                                    self.round_graphs.get_or_insert_with(HashMap::new).insert(key, g);
                                    return Ok(None);
                                }
                                Err(e) if crate::graph::is_capture_code(e.0) => {
                                    self.graph_refusals += 1;
                                    // the launch was refused: run the eager body (nothing ran)
                                    return self.draft_round_body(
                                        slot, taps, anchor, start_pos, max_drafts, sampling, capture_k, params, st,
                                    );
                                }
                                Err(e) => return Err(e),
                            }
                        }
                        Err(CaptureError::Closure(e)) => return Err(e),
                        Err(CaptureError::Refused(e)) => {
                            if e.is_fatal() {
                                return Err(e);
                            }
                            self.graph_refusals += 1;
                            // capture refused: fall back to the eager body
                        }
                    }
                }
                Err(e) => {
                    if e.is_fatal() {
                        return Err(e);
                    }
                    self.graph_refusals += 1;
                }
            }
        }

        // Eager (or after a refused capture): the same body, same params block.
        self.draft_round_body(slot, taps, anchor, start_pos, max_drafts, sampling, capture_k, params, st)
    }

    /// The launch body of one draft round, shared by the eager path and the CUDA-graph capture. When `params` is
    /// `Some` (the graph path), the per-round scalar kernels read `anchor` / `start_pos` / `reach` / `n_kv` / walk
    /// sampling from the device block; the eager path uses the same block so both are bit-identical.
    #[allow(clippy::too_many_arguments)]
    fn draft_round_body(
        &mut self,
        slot: usize,
        taps: &DevBuf,
        anchor: i32,
        start_pos: usize,
        max_drafts: usize,
        sampling: Option<MtpWalkSampling>,
        capture_k: Option<usize>,
        params: Option<&DevBuf>,
        st: &Stream,
    ) -> Result<Option<DsparkLattice>, CudaError> {
        let already_seeded = self.seeded_pos[slot] == Some(start_pos);
        let reach = self.cfg.window.min(start_pos + 1);
        let n_kv = reach + self.cfg.block_size;
        if let Err(e) = self.forward_embed_opt(if already_seeded { None } else { Some(taps) }, anchor, params, st) {
            return Err(e);
        }
        for s in 0..self.cfg.n_stages {
            if let Err(e) = self.stage_forward_params(s, start_pos, n_kv, params, st) {
                return Err(e);
            }
        }
        self.seeded_pos[slot] = Some(start_pos);
        self.head_n_sampled_capture(max_drafts, sampling, capture_k, params, st)
    }

    /// Write the round's `Dsv4MtpRoundParams` into pinned host staging and copy it async (H2D) onto the device
    /// block, ordered on `st` before the graph capture/replay. Idempotent; the device block is lazily allocated.
    fn prepare_round_params(
        &mut self,
        anchor: i32,
        start_pos: usize,
        reach: usize,
        n_kv: usize,
        sampling: Option<MtpWalkSampling>,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let s = sampling.unwrap_or(MtpWalkSampling { cand_k: 0, seed: 0, temperature: 0.0, top_p: 0.0, min_p: 0.0, base_pos: 0 });
        let params = Dsv4MtpRoundParams {
            anchor,
            start_pos: start_pos as i32,
            reach: reach as i32,
            n_kv: n_kv as i32,
            window: self.cfg.window as i32,
            block_size: self.cfg.block_size as i32,
            cand_k: s.cand_k.clamp(1, 64) as i32,
            _pad: 0,
            seed: s.seed,
            base_pos: s.base_pos,
            temperature: s.temperature,
            top_p: s.top_p,
            min_p: s.min_p,
        };
        if self.round_params_dev.is_none() {
            // No memset: the stream-ordered copy below writes all SIZE bytes. A `zeroed_view` memset runs on the legacy
            // stream, which a non-blocking `st` does not wait for, so it could land after that copy and zero the params.
            let block = std::sync::Arc::new(Block::alloc(Dsv4MtpRoundParams::SIZE)?);
            self.round_params_dev = Some(DevBuf::view(&block, 0, Dsv4MtpRoundParams::SIZE));
            self.round_params_host = Some(Block::alloc_mapped_host(Dsv4MtpRoundParams::SIZE)?);
        }
        let dev = self.round_params_dev.as_ref().unwrap();
        let host = self.round_params_host.as_ref().unwrap();
        // SAFETY: `host` is a mapped pinned block of SIZE bytes; `params` is a plain-old-data mirror of the C struct.
        unsafe {
            core::ptr::copy_nonoverlapping(
                &params as *const Dsv4MtpRoundParams as *const u8,
                host.host_ptr_at(0),
                Dsv4MtpRoundParams::SIZE,
            );
            st.memcpy_h2d_async(dev.ptr(), host.host_ptr_at(0), Dsv4MtpRoundParams::SIZE)?;
        }
        Ok(())
    }

    /// `Block.forward` for one stage, in place on the residual streams.
    pub fn stage_forward(&mut self, idx: usize, start_pos: usize, n_kv: usize, st: &Stream) -> Result<(), CudaError> {
        self.stage_forward_params(idx, start_pos, n_kv, None, st)
    }

    /// [`Self::stage_forward`] with an optional round-params block: when present, the per-round scalar kernels
    /// (ring writes, RoPE positions, top-k idxs, block attention n_kv) read their values from the device block so
    /// the same launch sequence replays under a CUDA graph as `start_pos` advances.
    pub fn stage_forward_params(
        &mut self,
        idx: usize,
        start_pos: usize,
        n_kv: usize,
        params: Option<&DevBuf>,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let cfg = self.cfg.clone();
        let (d, rows) = (cfg.dim, cfg.block_size);
        let hcp = cfg.hc;

        // --- attention sub-block
        {
            let stage = &self.stages[idx];
            let sc = &mut self.scratch;
            unsafe {
                CudaError::check(ffi::gb_dsv4_mtp_hc_mixes(
                    sc.h.ptr() as *const f32,
                    stage.hc_attn.f.ptr() as *const f32,
                    stage.hc_attn.s.ptr() as *const f32,
                    stage.hc_attn.b.ptr() as *const f32,
                    sc.pre.ptr() as *mut f32,
                    sc.post.ptr() as *mut f32,
                    sc.comb.ptr() as *mut f32,
                    i32_of(rows)?,
                    i32_of(d)?,
                    hcp.rms_eps,
                    hcp.hc_eps,
                    hcp.sinkhorn_iters,
                    st.raw(),
                ))?;
                // `x = self.hc_pre(x, pre_mix)`: the mix the *previous* block's FFN produced
                if self.taps_dtype == Dtype::BF16 {
                    CudaError::check(ffi::gb_dsv4_hc_collapse_f32(
                        sc.h.ptr() as *const f32,
                        sc.pre_mix.ptr() as *const f32,
                        sc.hc32.ptr() as *mut f32,
                        i32_of(rows)?,
                        i32_of(d)?,
                        st.raw(),
                    ))?;
                } else {
                    CudaError::check(ffi::gb_dsv4_mtp_hc_pre(
                        sc.h.ptr() as *const f32,
                        sc.pre_mix.ptr() as *const f32,
                        sc.xn.ptr(),
                        i32_of(rows)?,
                        i32_of(d)?,
                        st.raw(),
                    ))?;
                }
            }
            if self.taps_dtype == Dtype::BF16 {
                crate::dsv4_attn::rmsnorm_f32_to_f16_dev(&sc.hc32, &stage.attn_norm, &mut sc.xn, rows, d, cfg.eps, st)?;
            } else {
                rmsnorm_dev(&sc.xn, &stage.attn_norm, rows, d, cfg.eps, st)?;
            }
        }
        self.attn_params(idx, start_pos, n_kv, params, st)?;
        {
            let sc = &mut self.scratch;
            unsafe {
                CudaError::check(ffi::gb_dsv4_mtp_hc_post(
                    sc.h.ptr() as *mut f32,
                    sc.y.ptr(),
                    Dtype::F16 as i32,
                    sc.post.ptr() as *const f32,
                    sc.comb.ptr() as *const f32,
                    i32_of(rows)?,
                    i32_of(d)?,
                    st.raw(),
                ))?;
            }
        }

        // --- feed-forward sub-block. `hc_pre(x, attn_pre)` -- the mix attention produced, not the one the FFN just
        // computed -- so the attention's `pre` must survive this block's `hc_mixes`, which writes `pre_mix`.
        {
            let stage = &self.stages[idx];
            let sc = &mut self.scratch;
            unsafe {
                CudaError::check(ffi::gb_dsv4_mtp_hc_mixes(
                    sc.h.ptr() as *const f32,
                    stage.hc_ffn.f.ptr() as *const f32,
                    stage.hc_ffn.s.ptr() as *const f32,
                    stage.hc_ffn.b.ptr() as *const f32,
                    sc.pre_mix.ptr() as *mut f32,
                    sc.post.ptr() as *mut f32,
                    sc.comb.ptr() as *mut f32,
                    i32_of(rows)?,
                    i32_of(d)?,
                    hcp.rms_eps,
                    hcp.hc_eps,
                    hcp.sinkhorn_iters,
                    st.raw(),
                ))?;
                if self.taps_dtype == Dtype::BF16 {
                    CudaError::check(ffi::gb_dsv4_hc_collapse_f32(
                        sc.h.ptr() as *const f32,
                        sc.pre.ptr() as *const f32,
                        sc.hc32.ptr() as *mut f32,
                        i32_of(rows)?,
                        i32_of(d)?,
                        st.raw(),
                    ))?;
                } else {
                    CudaError::check(ffi::gb_dsv4_mtp_hc_pre(
                        sc.h.ptr() as *const f32,
                        sc.pre.ptr() as *const f32,
                        sc.xn.ptr(),
                        i32_of(rows)?,
                        i32_of(d)?,
                        st.raw(),
                    ))?;
                }
            }
            if self.taps_dtype == Dtype::BF16 {
                crate::dsv4_attn::rmsnorm_f32_to_f16_dev(&sc.hc32, &stage.ffn_norm, &mut sc.xn, rows, d, cfg.eps, st)?;
            } else {
                rmsnorm_dev(&sc.xn, &stage.ffn_norm, rows, d, cfg.eps, st)?;
            }
        }
        self.moe(idx, rows, st)?;
        let sc = &mut self.scratch;
        unsafe {
            CudaError::check(ffi::gb_dsv4_mtp_hc_post(
                sc.h.ptr() as *mut f32,
                sc.y.ptr(),
                Dtype::F16 as i32,
                sc.post.ptr() as *const f32,
                sc.comb.ptr() as *const f32,
                i32_of(rows)?,
                i32_of(d)?,
                st.raw(),
            ))
        }
    }

    /// `DSparkAttention.forward` for `start_pos > 0`.
    pub fn attn(&mut self, idx: usize, start_pos: usize, n_kv: usize, st: &Stream) -> Result<(), CudaError> {
        self.attn_params(idx, start_pos, n_kv, None, st)
    }

    /// [`Self::attn`] with an optional round-params block (see [`Self::stage_forward_params`]).
    pub fn attn_params(
        &mut self,
        idx: usize,
        start_pos: usize,
        n_kv: usize,
        params: Option<&DevBuf>,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let cfg = self.cfg.clone();
        let (hd, nh, n) = (cfg.head_dim, cfg.n_heads, cfg.block_size);
        let slot = self.active_slot.min(MAX_SLOTS - 1);
        let stage = &self.stages[idx];
        let win_buf = &self.slot_windows[idx][slot];
        let sc = &mut self.scratch;

        // the trunk tap becomes the newest window row (unless already seeded by `prefill_seed`)
        if self.seeded_pos[slot] != Some(start_pos) {
            stage.wkv.forward(&sc.main_x, &mut sc.kv, 1, st)?;
            rmsnorm_dev(&sc.kv, &stage.kv_norm, 1, hd, cfg.eps, st)?;
            match params {
                Some(p) => crate::dsv4_attn::rope_rows_dev_params(
                    &sc.kv,
                    1,
                    hd,
                    cfg.rope_dim,
                    hd,
                    1,
                    stage.rope.pair(),
                    0,
                    1,
                    false,
                    p,
                    st,
                )?,
                None => rope_rows_dev(&sc.kv, 1, hd, cfg.rope_dim, hd, 1, stage.rope.pair(), start_pos, 1, false, st)?,
            }
            quant_dev(&sc.kv, 1, hd, 32, 8, true, st)?;
            match params {
                Some(p) => crate::dsv4_attn::ring_write_dev_params(&sc.kv, 0, win_buf, 1, hd, p, st)?,
                None => crate::dsv4_attn::ring_write_dev(&sc.kv, 0, win_buf, 1, hd, start_pos % cfg.window, cfg.window, st)?,
            }
        }

        // the block's own Q and KV. Positions `start_pos + 1 ..`: the tap is the token AT `start_pos`, so the block's
        // first row is the next one (`freqs_cis[start_pos + seqlen : ...]`, model.py:1057).
        MtpDense::forward_pair(&stage.wq_a, &mut sc.qr, &stage.wkv, &mut sc.kv, &sc.xn, n, st)?;
        rmsnorm_dev(&sc.qr, &stage.q_norm, n, cfg.q_lora, cfg.eps, st)?;
        stage.wq_b.forward(&sc.qr, &mut sc.q, n, st)?;
        match params {
            Some(p) => crate::dsv4_attn::rope_rows_dev_params(
                &sc.q,
                n,
                nh * hd,
                cfg.rope_dim,
                hd,
                nh,
                stage.rope.pair(),
                1,
                1,
                false,
                p,
                st,
            )?,
            None => rope_rows_dev(&sc.q, n, nh * hd, cfg.rope_dim, hd, nh, stage.rope.pair(), start_pos + 1, 1, false, st)?,
        }
        rmsnorm_dev(&sc.kv, &stage.kv_norm, n, hd, cfg.eps, st)?;
        match params {
            Some(p) => {
                crate::dsv4_attn::rope_rows_dev_params(&sc.kv, n, hd, cfg.rope_dim, hd, 1, stage.rope.pair(), 1, 1, false, p, st)?
            }
            None => rope_rows_dev(&sc.kv, n, hd, cfg.rope_dim, hd, 1, stage.rope.pair(), start_pos + 1, 1, false, st)?,
        }
        quant_dev(&sc.kv, n, hd, 32, 8, true, st)?;

        // the combined plane: the ring, then the block
        crate::dsv4_attn::plane_write_dev(win_buf, 0, &sc.combined, cfg.window, hd, 0, st)?;
        crate::dsv4_attn::plane_write_dev(&sc.kv, 0, &sc.combined, n, hd, cfg.window * hd, st)?;

        let reach = cfg.window.min(start_pos + 1);
        if idx == 0 {
            unsafe {
                match params {
                    Some(p) => {
                        CudaError::check(ffi::gb_dsv4_mtp_topk_idxs_params(sc.idxs.ptr(), 1, i32_of(n)?, p.ptr(), st.raw()))?
                    }
                    None => CudaError::check(ffi::gb_dsv4_mtp_topk_idxs(
                        sc.idxs.ptr(),
                        1,
                        i32_of(n)?,
                        i32_of(reach)?,
                        i32_of(cfg.window)?,
                        st.raw(),
                    ))?,
                }
            }
        }

        let scale = 1.0f32 / (hd as f32).sqrt();
        unsafe {
            match params {
                Some(p) => CudaError::check(ffi::gb_dsv4_mtp_attn_params(
                    sc.q.ptr(),
                    sc.combined.ptr(),
                    stage.sink.ptr() as *const f32,
                    sc.idxs.ptr() as *const i32,
                    sc.o.ptr(),
                    i32_of(n)?,
                    i32_of(nh)?,
                    i32_of(hd)?,
                    scale,
                    p.ptr(),
                    st.raw(),
                ))?,
                None => CudaError::check(ffi::gb_dsv4_mtp_attn(
                    sc.q.ptr(),
                    sc.combined.ptr(),
                    stage.sink.ptr() as *const f32,
                    sc.idxs.ptr() as *const i32,
                    sc.o.ptr(),
                    i32_of(n)?,
                    i32_of(nh)?,
                    i32_of(hd)?,
                    i32_of(n_kv)?,
                    scale,
                    st.raw(),
                ))?,
            }
        }
        // undo the query rotation, then the grouped output projection
        match params {
            Some(p) => crate::dsv4_attn::rope_rows_dev_params(
                &sc.o,
                n,
                nh * hd,
                cfg.rope_dim,
                hd,
                nh,
                stage.rope.pair(),
                1,
                1,
                true,
                p,
                st,
            )?,
            None => rope_rows_dev(&sc.o, n, nh * hd, cfg.rope_dim, hd, nh, stage.rope.pair(), start_pos + 1, 1, true, st)?,
        }
        quant_dev(&sc.o, n, nh * hd, 32, 8, true, st)?;
        stage.wo_a.woa_grouped(&sc.o, &mut sc.olora, n, cfg.o_groups, cfg.o_lora, cfg.o_in_per_group(), st)?;
        stage.wo_b.forward(&sc.olora, &mut sc.y, n, st)
    }

    /// `MoE.forward`: noaux_tc router, top-`k` routed experts, one shared expert.
    pub fn moe(&mut self, idx: usize, rows: usize, st: &Stream) -> Result<crate::dsv4_moe::MoeStats, CudaError> {
        let cfg = self.cfg.clone();
        let (d, e, topk) = (cfg.dim, cfg.n_routed_experts, cfg.topk);
        {
            let stage = &self.stages[idx];
            let sc = &mut self.scratch;
            unsafe {
                CudaError::check(ffi::gb_dsv4_mtp_router(
                    sc.xn.ptr(),
                    stage.gate.ptr() as *const u16,
                    sc.router.ptr() as *mut f32,
                    i32_of(rows)?,
                    i32_of(d)?,
                    i32_of(e)?,
                    cfg.gate_temp,
                    st.raw(),
                ))?;
                CudaError::check(ffi::gb_dsv4_mtp_select(
                    sc.router.ptr() as *const f32,
                    stage.gate_bias.ptr() as *const f32,
                    sc.picks.ptr() as *mut i32,
                    sc.wts.ptr(),
                    i32_of(rows)?,
                    i32_of(e)?,
                    i32_of(topk)?,
                    cfg.route_scale,
                    st.raw(),
                ))?;
            }
        }
        // The shared expert is dense and unweighted. `act` carries its swiglu output ([rows, inter], and the routed
        // gate_up overwrites it afterwards) and `y` its [rows, dim] result, which the routed sum is added to.
        {
            let stage = &self.stages[idx];
            let sc = &mut self.scratch;
            MtpDense::forward_pair(&stage.shared_w1, &mut sc.qr, &stage.shared_w3, &mut sc.act, &sc.xn, rows, st)?;
            unsafe {
                CudaError::check(ffi::gb_dsv4_swiglu_f16(
                    sc.qr.ptr(),
                    sc.act.ptr(),
                    core::ptr::null(),
                    sc.act.ptr(),
                    i32_of(rows * cfg.inter_dim)?,
                    i32_of(cfg.inter_dim)?,
                    cfg.swiglu_limit,
                    st.raw(),
                ))?;
            }
            stage.shared_w2.forward(&sc.act, &mut sc.y, rows, st)?;
        }
        // --- routed experts
        let Self { cfg, stages, scratch: sc, ws, ms, ms_descs, .. } = self;
        match &stages[idx].routed {
            RoutedExperts::Exl3(groups) => {
                let m = ms.as_mut().ok_or(CudaError(1))?;
                if let Some(desc) = ms_descs.get(idx).and_then(|d| d.as_ref()) {
                    crate::dsv4_moe::moe_experts_resident(
                        m,
                        desc,
                        rows,
                        &sc.xn,
                        &mut sc.y,
                        &sc.picks,
                        &sc.wts,
                        &mut sc.moe32,
                        cfg.swiglu_limit,
                        st,
                    )
                } else {
                    moe_experts(
                        m,
                        |e| Ok(MoeGroups::Resident(&groups[e])),
                        rows,
                        &sc.xn,
                        &mut sc.y,
                        &sc.picks,
                        &sc.wts,
                        &mut sc.moe32,
                        cfg.swiglu_limit,
                        st,
                    )
                }
            }
            RoutedExperts::Mxfp4(mx) => {
                let slots = rows * cfg.topk;
                if sc.xs.bytes() < slots * cfg.dim * 2 {
                    return Err(CudaError(1));
                }
                unsafe {
                    CudaError::check(ffi::gb_dsv4_moe_group(
                        sc.picks.ptr() as *const i32,
                        sc.wts.ptr(),
                        sc.counts.ptr() as *mut i32,
                        sc.offsets.ptr() as *mut i32,
                        sc.perm.ptr() as *mut i32,
                        sc.pinv.ptr() as *mut i32,
                        sc.wpack.ptr(),
                        i32_of(slots)?,
                        i32_of(cfg.n_routed_experts)?,
                        st.raw(),
                    ))?;
                    CudaError::check(ffi::gb_dsv4_moe_gather(
                        sc.xn.ptr(),
                        sc.perm.ptr() as *const i32,
                        sc.xs.ptr(),
                        i32_of(slots)?,
                        i32_of(cfg.topk)?,
                        i32_of(cfg.dim)?,
                        st.raw(),
                    ))?;
                }
                mx.gate_up(&sc.xs, &mut sc.act, &sc.offsets, &sc.wpack, slots, cfg.swiglu_limit, ws, st)?;
                mx.down(&sc.act, &mut sc.yp, &sc.offsets, slots, ws, st)?;
                unsafe {
                    CudaError::check(ffi::gb_dsv4_moe_scatter(
                        sc.y.ptr(),
                        sc.yp.ptr(),
                        sc.pinv.ptr() as *const i32,
                        sc.wpack.ptr(),
                        sc.moe32.ptr(),
                        sc.y.ptr(),
                        i32_of(rows)?,
                        i32_of(cfg.topk)?,
                        i32_of(cfg.dim)?,
                        st.raw(),
                    ))
                }?;
                Ok(crate::dsv4_moe::MoeStats::default())
            }
        }
    }

    /// `DSparkBlock.forward_head`: collapse, norm, the markov walk over the block, then the confidence of every row.
    pub fn head(&mut self, st: &Stream) -> Result<(), CudaError> {
        let bs = self.cfg.block_size;
        self.head_n(bs, st)
    }

    /// `DSparkBlock.forward_head` with the sequential Markov walk capped at `max_drafts` (`1..=block_size`).
    pub fn head_n(&mut self, max_drafts: usize, st: &Stream) -> Result<(), CudaError> {
        self.head_n_sampled(max_drafts, None, st)
    }

    /// [`Self::head_n`] with the round's lattice captured between the head's projection and the walk (see
    /// [`Self::draft_n_lattice`]); the walk and the confidence read the same buffers either way.
    pub fn head_n_capture(
        &mut self,
        max_drafts: usize,
        capture_k: Option<usize>,
        st: &Stream,
    ) -> Result<Option<DsparkLattice>, CudaError> {
        self.head_n_sampled_capture(max_drafts, None, capture_k, None, st)
    }

    /// `DSparkBlock.forward_head` with optional keyed Gumbel sampling over the Markov walk.
    pub fn head_n_sampled(&mut self, max_drafts: usize, sampling: Option<MtpWalkSampling>, st: &Stream) -> Result<(), CudaError> {
        self.head_n_sampled_capture(max_drafts, sampling, None, None, st).map(|_| ())
    }

    pub fn head_n_sampled_capture(
        &mut self,
        max_drafts: usize,
        sampling: Option<MtpWalkSampling>,
        capture_k: Option<usize>,
        params: Option<&DevBuf>,
        st: &Stream,
    ) -> Result<Option<DsparkLattice>, CudaError> {
        let cfg = self.cfg.clone();
        let (rows, d, vocab, rank) = (cfg.block_size, cfg.dim, cfg.vocab, cfg.markov_rank);
        let walk_rows = max_drafts.clamp(1, rows);
        let Self { stages, scratch: sc, ms, ws, .. } = self;
        let last = &stages[cfg.n_stages - 1];
        unsafe {
            if last.lm_head_exl3.is_some() {
                CudaError::check(ffi::gb_dsv4_hc_collapse_f32(
                    sc.h.ptr() as *const f32,
                    sc.pre_mix.ptr() as *const f32,
                    sc.hc32.ptr() as *mut f32,
                    i32_of(rows)?,
                    i32_of(d)?,
                    st.raw(),
                ))?;
            } else {
                CudaError::check(ffi::gb_dsv4_mtp_hc_pre(
                    sc.h.ptr() as *const f32,
                    sc.pre_mix.ptr() as *const f32,
                    sc.hc16.ptr(),
                    i32_of(rows)?,
                    i32_of(d)?,
                    st.raw(),
                ))?;
                // the confidence head reads the collapsed stream *before* the final norm
                CudaError::check(ffi::gb_dsv4_mtp_widen(sc.hc16.ptr(), sc.hc32.ptr() as *mut f32, i32_of(rows * d)?, st.raw()))?;
            }
        }
        if last.lm_head_exl3.is_some() {
            crate::dsv4_attn::rmsnorm_f32_to_f16_dev(&sc.hc32, last.norm.as_ref().unwrap(), &mut sc.hc16, rows, d, cfg.eps, st)?;
        } else {
            rmsnorm_dev(&sc.hc16, last.norm.as_ref().unwrap(), rows, d, cfg.eps, st)?;
        }
        if let Some(hw) = &last.lm_head_exl3 {
            let need = crate::exl3_linear::linear_workspace_bytes(walk_rows, d, vocab, hw.bits);
            let work = match ms.as_mut() {
                Some(m) => m.ensure_workspace(need)?,
                None => {
                    if ws.bytes() < need {
                        // the EXL3 decode workspace needs no initialization; a legacy-stream memset could land after the
                        // forward below on a non-blocking `st` and zero its live scratch
                        *ws = DevBuf::alloc(need)?;
                    }
                    ws
                }
            };
            hw.forward(&sc.hc16, &mut sc.logits, walk_rows, Dtype::F16, Dtype::F32, work, st)?;
        } else {
            last.lm_head.as_ref().unwrap().logits(&sc.hc16, &mut sc.logits, walk_rows, st)?;
        }

        // the capture point: the walked rows of `sc.logits` still hold the parent-INDEPENDENT unary scores here --
        // the walk below biases them in place -- so a tree round's lattice reads them now; the candidate sets are
        // assembled after the walk, when its picks exist to ride each level's head
        let unary = match capture_k {
            Some(k) => {
                st.sync()?;
                let mut logits = vec![0f32; walk_rows * vocab];
                sc.logits.download_range(0, &mut logits)?;
                Some((k, logits))
            }
            None => None,
        };

        // the walk: `logits_k += head @ embed(x_{k-1})`, then `x_k = argmax(logits_k)` (or keyed Gumbel draw) -- sequential in k.
        let (embed, mhead) = (last.markov_embed.as_ref().unwrap(), last.markov_head.as_ref().unwrap());
        let (mh_vocab, mh_rank) = (cfg.vocab, cfg.markov_rank);
        let bf16_markov = embed.bytes() == mh_vocab * mh_rank * 2;
        let use_fast_markov =
            crate::dsv4_moe::runtime_env("REXL3_DSV41_MTP_FULL_MARKOV").map(|v| v.trim() != "1").unwrap_or(true);
        let keyed = sampling.filter(|s| s.temperature > 0.0);
        if bf16_markov
            && mh_rank == 256
            && mh_vocab >= 4096
            && sc.q.bytes() >= walk_rows * 1024 * 4
            && sc.o.bytes() >= 1024 * 4
            && use_fast_markov
        {
            unsafe {
                if let Some(s) = keyed {
                    match params {
                        Some(p) => CudaError::check(ffi::gb_dsv4_mtp_markov_walk_keyed_bf16_params(
                            sc.logits.ptr(),
                            sc.ids.ptr() as *mut i32,
                            sc.conf.ptr() as *mut f32,
                            embed.ptr(),
                            mhead.ptr(),
                            sc.q.ptr(),
                            sc.o.ptr(),
                            i32_of(walk_rows)?,
                            i32_of(mh_vocab)?,
                            i32_of(mh_rank)?,
                            p.ptr(),
                            st.raw(),
                        ))?,
                        None => CudaError::check(ffi::gb_dsv4_mtp_markov_walk_keyed_bf16(
                            sc.logits.ptr(),
                            sc.ids.ptr() as *mut i32,
                            sc.conf.ptr() as *mut f32,
                            embed.ptr(),
                            mhead.ptr(),
                            sc.q.ptr(),
                            sc.o.ptr(),
                            i32_of(walk_rows)?,
                            i32_of(mh_vocab)?,
                            i32_of(mh_rank)?,
                            i32_of(s.cand_k.clamp(1, 64))?,
                            s.seed,
                            s.temperature,
                            s.top_p,
                            s.min_p,
                            s.base_pos,
                            st.raw(),
                        ))?,
                    }
                } else {
                    CudaError::check(ffi::gb_dsv4_mtp_markov_walk_fast_bf16(
                        sc.logits.ptr(),
                        sc.ids.ptr() as *mut i32,
                        embed.ptr(),
                        mhead.ptr(),
                        sc.q.ptr(),
                        sc.o.ptr(),
                        i32_of(walk_rows)?,
                        i32_of(mh_vocab)?,
                        i32_of(mh_rank)?,
                        st.raw(),
                    ))?;
                }
            }
        } else {
            for k in 0..walk_rows {
                unsafe {
                    if bf16_markov {
                        CudaError::check(ffi::gb_dsv4_mtp_markov_bias_bf16(
                            sc.logits.ptr_at(k * vocab * 4),
                            sc.ids.ptr_at(k * 4),
                            embed.ptr(),
                            mhead.ptr(),
                            1,
                            i32_of(mh_vocab)?,
                            i32_of(mh_rank)?,
                            st.raw(),
                        ))?;
                    } else {
                        CudaError::check(ffi::gb_dsv4_mtp_markov_bias(
                            sc.logits.ptr_at(k * vocab * 4),
                            sc.ids.ptr_at(k * 4),
                            embed.ptr(),
                            mhead.ptr(),
                            1,
                            i32_of(mh_vocab)?,
                            i32_of(mh_rank)?,
                            st.raw(),
                        ))?;
                    }
                    CudaError::check(ffi::gb_dsv4_mtp_argmax(
                        sc.logits.ptr_at(k * vocab * 4) as *const f32,
                        sc.ids.ptr_at((k + 1) * 4) as *mut i32,
                        1,
                        i32_of(vocab)?,
                        st.raw(),
                    ))?;
                }
            }
        }
        unsafe {
            if bf16_markov {
                CudaError::check(ffi::gb_dsv4_mtp_gather_rows_bf16(
                    sc.mem.ptr() as *mut f32,
                    sc.ids.ptr() as *const i32,
                    embed.ptr(),
                    i32_of(rows)?,
                    i32_of(rank)?,
                    i32_of(vocab)?,
                    st.raw(),
                ))?;
                if keyed.is_some() {
                    CudaError::check(ffi::gb_dsv4_mtp_confidence_keyed_bf16(
                        sc.hc32.ptr(),
                        sc.mem.ptr(),
                        last.conf_proj.as_ref().unwrap().ptr(),
                        last.conf_bias.as_ref().unwrap().ptr(),
                        sc.conf.ptr(),
                        i32_of(rows)?,
                        i32_of(d)?,
                        i32_of(rank)?,
                        i32_of(walk_rows)?,
                        st.raw(),
                    ))?;
                } else {
                    CudaError::check(ffi::gb_dsv4_mtp_confidence_bf16(
                        sc.hc32.ptr(),
                        sc.mem.ptr(),
                        last.conf_proj.as_ref().unwrap().ptr(),
                        last.conf_bias.as_ref().unwrap().ptr(),
                        sc.conf.ptr(),
                        i32_of(rows)?,
                        i32_of(d)?,
                        i32_of(rank)?,
                        st.raw(),
                    ))?;
                }
            } else {
                CudaError::check(ffi::gb_dsv4_mtp_gather_rows(
                    sc.mem.ptr() as *mut f32,
                    sc.ids.ptr() as *const i32,
                    embed.ptr() as *const f32,
                    i32_of(rows)?,
                    i32_of(rank)?,
                    i32_of(vocab)?,
                    st.raw(),
                ))?;
                CudaError::check(ffi::gb_dsv4_mtp_confidence(
                    sc.hc32.ptr(),
                    sc.mem.ptr(),
                    last.conf_proj.as_ref().unwrap().ptr(),
                    last.conf_bias.as_ref().unwrap().ptr(),
                    sc.conf.ptr(),
                    i32_of(rows)?,
                    i32_of(d)?,
                    i32_of(rank)?,
                    st.raw(),
                ))?;
            }
        };
        let lattice = match unary {
            Some((k, logits)) => capture_lattice(last, sc, &cfg, walk_rows, k, &logits, st)?,
            None => None,
        };
        Ok(lattice)
    }

    /// The one host copy of a round: the `block_size + 1` ids and the `block_size` confidences.
    pub fn read(&self) -> Result<(Vec<i32>, Vec<f32>), CudaError> {
        Ok((self.scratch.ids.to_vec()?, self.scratch.conf.to_vec()?))
    }

    /// Enqueue the round's device-to-host copies (ids, confidences, and the device schedule result when one is in
    /// flight) into pinned staging on `st` and record the event that marks them complete. This returns without any
    /// host wait: the caller runs any work that does not need the values, then [`Self::read_end`] waits on the
    /// recorded event — replacing a full stream sync plus per-buffer synchronous downloads with one narrow wait that
    /// covers only these copies.
    pub fn read_begin(&mut self, st: &Stream) -> Result<(), CudaError> {
        let bs = self.cfg.block_size;
        if self.stage.is_none() {
            self.stage = Some(DraftReadbackStage::new(bs)?);
        }
        let stg = self.stage.as_ref().unwrap();
        // SAFETY: `stg`'s blocks are pinned (`alloc_mapped_host`) and `bytes` bytes of device scratch lie behind
        // each `src`; the copies are queued on `st`, ordered after the kernels that wrote the scratch.
        unsafe {
            st.memcpy_d2h_async(stg.ids.host_ptr_at(0), self.scratch.ids.ptr(), (bs + 1) * 4)?;
            st.memcpy_d2h_async(stg.conf.host_ptr_at(0), self.scratch.conf.ptr(), bs * 4)?;
            if let Some(out) = self.sched_out.as_ref() {
                st.memcpy_d2h_async(stg.sched.host_ptr_at(0), out.ptr(), 4)?;
            }
        }
        stg.ev.record(st)?;
        Ok(())
    }

    /// Wait for [`Self::read_begin`]'s copies, then return the staged ids and confidences, plus the device schedule
    /// result when [`Self::schedule_verify_len_launch`] ran this round. Nothing else on the stream is waited on.
    pub fn read_end(&self) -> Result<(Vec<i32>, Vec<f32>, Option<usize>), CudaError> {
        let stg = self.stage.as_ref().ok_or(CudaError(1))?;
        stg.ev.sync()?;
        let bs = self.cfg.block_size;
        let mut ids = vec![0i32; bs + 1];
        let mut conf = vec![0f32; bs];
        // SAFETY: the event guarantees the pinned staging holds `bs + 1` ids and `bs` confidences by now.
        unsafe {
            core::ptr::copy_nonoverlapping(stg.ids.host_ptr_at(0) as *const i32, ids.as_mut_ptr(), bs + 1);
            core::ptr::copy_nonoverlapping(stg.conf.host_ptr_at(0) as *const f32, conf.as_mut_ptr(), bs);
        }
        let sched = if self.sched_out.is_some() {
            let mut v = [0i32; 1];
            unsafe { core::ptr::copy_nonoverlapping(stg.sched.host_ptr_at(0) as *const i32, v.as_mut_ptr(), 1) }
            Some(v[0].max(0) as usize)
        } else {
            None
        };
        Ok((ids, conf, sched))
    }

    /// [`Self::read`] plus the biased logits `[block_size, vocab]` -- a debug and test path (2.6 MB at this vocab).
    pub fn read_full(&self) -> Result<DsparkDraft, CudaError> {
        let (ids, confidence) = self.read()?;
        Ok(DsparkDraft { ids, logits: self.scratch.logits.to_vec()?, confidence })
    }

    /// How many drafts `threshold` keeps, on the device, one int per sequence.
    pub fn accept(&self, threshold: f32, st: &Stream) -> Result<usize, CudaError> {
        unsafe {
            CudaError::check(ffi::gb_dsv4_mtp_accept(
                self.scratch.conf.ptr(),
                self.scratch.acc.ptr(),
                1,
                i32_of(self.cfg.block_size)?,
                threshold,
                st.raw(),
            ))?;
        }
        st.sync()?;
        Ok(self.scratch.acc.to_vec::<i32>()?[0].max(0) as usize)
    }

    /// `start_pos == 0`: seed every stage's window ring from the trunk taps, one chunk of `rows <= block_size`
    /// positions at a time, at position `pos0`. No draft is produced (`forward_spec` returns `None`), and the ring ends
    /// up with `slot = pos % window` for the last `window` positions -- what the reference's rotated split produces.
    pub fn prefill_seed(&mut self, taps: &DevBuf, pos0: usize, rows: usize, st: &Stream) -> Result<(), CudaError> {
        let slot = self.active_slot;
        self.prefill_seed_slot(slot, taps, pos0, rows, st)
    }

    /// Seed every stage's window ring for `slot` (`0..MAX_SLOTS`) from the trunk taps starting at `pos0`.
    pub fn prefill_seed_slot(
        &mut self,
        slot: usize,
        taps: &DevBuf,
        mut pos0: usize,
        mut rows: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let cfg = self.cfg.clone();
        let tapw = cfg.tap_width();
        if slot >= MAX_SLOTS || rows == 0 || taps.bytes() < rows * tapw * 2 {
            return Err(CudaError(1));
        }
        self.active_slot = slot;
        let taps = if rows > cfg.window {
            let skip = rows - cfg.window;
            pos0 += skip;
            rows = cfg.window;
            taps.subview(skip * tapw * 2, rows * tapw * 2)
        } else {
            taps.subview(0, rows * tapw * 2)
        };
        // `sc.logits` (~2.6 MB) can hold all `cfg.window = 64` rows of `hc32` + `main_x` + `kv` during `prefill_seed`;
        // for smaller toy scratch arenas, fall back to `sc.xs` / `sc.h` / `sc.act`.
        let max_logits_rows = self.scratch.logits.bytes() / (cfg.dim * 6 + cfg.head_dim * 2);
        let max_main_x_rows = (self.scratch.xs.bytes() / (cfg.dim * 2)).max(self.scratch.rows);
        let max_hc32_rows = (self.scratch.h.bytes() / (cfg.dim * 4)).max(self.scratch.rows);
        let max_kv_rows = (self.scratch.act.bytes() / (cfg.head_dim * 2)).max(self.scratch.rows);
        let fallback_chunk = self.scratch.rows.max(8.min(max_main_x_rows).min(max_hc32_rows).min(max_kv_rows));
        let chunk = fallback_chunk.max(max_logits_rows.min(cfg.window));
        let mut at = 0usize;
        while at < rows {
            let n = chunk.min(rows - at);
            let tv = taps.subview(at * tapw * 2, n * tapw * 2);
            self.prefill_chunk(&tv, pos0 + at, n, st)?;
            at += n;
        }
        self.seeded_pos[slot] = Some(pos0 + rows - 1);
        Ok(())
    }

    fn prefill_chunk(&mut self, taps: &DevBuf, pos0: usize, rows: usize, st: &Stream) -> Result<(), CudaError> {
        let cfg = self.cfg.clone();
        let (d, hd, win) = (cfg.dim, cfg.head_dim, cfg.window);
        let slot = self.active_slot.min(MAX_SLOTS - 1);
        let s0 = &self.stages[0];
        let sc = &mut self.scratch;
        let use_logits_arena = rows > sc.rows
            && (rows * d * 2 > sc.xs.bytes() || rows * d * 4 > sc.h.bytes() || rows * hd * 2 > sc.act.bytes())
            && sc.logits.bytes() >= rows * (d * 6 + hd * 2);
        let use_wide = rows > sc.rows && !use_logits_arena;
        let (main_x_v, mut kv_v) = if use_logits_arena {
            let mut hc32_v = sc.logits.subview(0, rows * d * 4);
            let mut mx_v = sc.logits.subview(rows * d * 4, rows * d * 2);
            let kv_sub = sc.logits.subview(rows * d * 6, rows * hd * 2);
            if self.taps_dtype == Dtype::F16 {
                s0.main_proj.as_ref().unwrap().forward(taps, &mut mx_v, rows, st)?;
                rmsnorm_dev(&mx_v, s0.main_norm.as_ref().unwrap(), rows, d, cfg.eps, st)?;
            } else {
                s0.main_proj.as_ref().unwrap().logits_typed(taps, self.taps_dtype, &mut hc32_v, rows, st)?;
                crate::dsv4_attn::rmsnorm_f32_to_f16_dev(
                    &hc32_v,
                    s0.main_norm.as_ref().unwrap(),
                    &mut mx_v,
                    rows,
                    d,
                    cfg.eps,
                    st,
                )?;
            }
            (mx_v, kv_sub)
        } else {
            let main_x_buf: &mut DevBuf = if use_wide { &mut sc.xs } else { &mut sc.main_x };
            if self.taps_dtype == Dtype::F16 {
                s0.main_proj.as_ref().unwrap().forward(taps, main_x_buf, rows, st)?;
                rmsnorm_dev(main_x_buf, s0.main_norm.as_ref().unwrap(), rows, d, cfg.eps, st)?;
            } else {
                let hc32_buf: &mut DevBuf = if use_wide { &mut sc.h } else { &mut sc.hc32 };
                s0.main_proj.as_ref().unwrap().logits_typed(taps, self.taps_dtype, hc32_buf, rows, st)?;
                let main_x_buf: &mut DevBuf = if use_wide { &mut sc.xs } else { &mut sc.main_x };
                let hc32_ref: &DevBuf = if use_wide { &sc.h } else { &sc.hc32 };
                let hc32_v = hc32_ref.subview(0, rows * d * 4);
                let mut main_x_v = main_x_buf.subview(0, rows * d * 2);
                crate::dsv4_attn::rmsnorm_f32_to_f16_dev(
                    &hc32_v,
                    s0.main_norm.as_ref().unwrap(),
                    &mut main_x_v,
                    rows,
                    d,
                    cfg.eps,
                    st,
                )?;
            }
            let mx_v = if use_wide { sc.xs.subview(0, rows * d * 2) } else { sc.main_x.subview(0, rows * d * 2) };
            let kv_sub = if use_wide { sc.act.subview(0, rows * hd * 2) } else { sc.kv.subview(0, rows * hd * 2) };
            (mx_v, kv_sub)
        };
        for s in 0..cfg.n_stages {
            let stage = &self.stages[s];
            let win_buf = &self.slot_windows[s][slot];
            stage.wkv.forward(&main_x_v, &mut kv_v, rows, st)?;
            rmsnorm_dev(&kv_v, &stage.kv_norm, rows, hd, cfg.eps, st)?;
            rope_rows_dev(&kv_v, rows, hd, cfg.rope_dim, hd, 1, stage.rope.pair(), pos0, 1, false, st)?;
            quant_dev(&kv_v, rows, hd, 32, 8, true, st)?;
            // One launch a non-wrapping piece: two rows of a chunk that straddles the end of the ring would otherwise
            // both target slot 0 and the winner would be undefined.
            let mut done = 0usize;
            while done < rows {
                let pos = pos0 + done;
                let n = (rows - done).min(win - pos % win);
                crate::dsv4_attn::ring_write_dev(&kv_v, done, win_buf, n, hd, pos % win, win, st)?;
                done += n;
            }
        }
        Ok(())
    }

    /// The host reference's acceptance rule, over a confidence slice: `model.py` has no rejection loop, so the rule is
    /// the arbiter's. Kept here so a test can compare the device count with it.
    pub fn accept_host(conf: &[f32], threshold: f32) -> usize {
        accept_prefix(conf, threshold)
    }

    /// Measured full-cycle step-time curve `T(R)` in milliseconds for `R = 1 + l` verify rows (`l = 0..=7`) on GB10.
    /// Reflects the register-resident `ROWS <= 4` fast window (`R = 1..=4`, `l = 0..=3`) vs `ROWS > 4` (`R >= 5`, `l >= 4`)
    /// plus the per-row routed-expert activation cost across 40 MoE layers with `fast_step_shuffled<2,4,6,8>()`.
    pub const VERIFY_CYCLE_MS: [f32; 8] = [46.0, 52.0, 54.0, 56.0, 61.5, 65.5, 69.5, 73.5];

    /// Choose the speculative verify length `l in 1..=max_drafts` that maximizes expected throughput
    /// `(1 + sum_{j=1..l} prod_{i=1..j} c_i) / T(1 + l)` subject to `c_j >= min_conf` for `j > 1`.
    pub fn schedule_verify_len(conf: &[f32], min_conf: f32, max_drafts: usize) -> usize {
        let cap = conf.len().min(max_drafts);
        if cap == 0 {
            return 0;
        }
        let mut cum_prod = 1.0f32;
        let mut expected_tokens = 1.0f32;
        let mut best_l = 1usize;
        let mut best_rate = 0.0f32;
        for l in 1..=cap {
            let c = conf[l - 1].clamp(0.0, 1.0);
            if l > 1 && c < min_conf {
                break;
            }
            cum_prod *= c;
            expected_tokens += cum_prod;
            let t_ms = Self::VERIFY_CYCLE_MS[l.min(Self::VERIFY_CYCLE_MS.len() - 1)];
            let rate = expected_tokens / t_ms;
            if l == 1 || rate >= best_rate {
                best_rate = rate;
                best_l = l;
            }
        }
        best_l
    }

    /// Enqueue the device mirror of [`Self::schedule_verify_len`] over the round's GPU-resident confidences
    /// (`scratch.conf`), one pass on `st` — no host wait; the result is copied by [`Self::read_begin`] and read
    /// by [`Self::read_end`] on the same event as the ids/confidences. Same clamp, same floor break, same
    /// accumulation order, same cycle table, same `>=` tie-break — the result is bit-for-bit the host function's
    /// for the same input vector.
    pub fn schedule_verify_len_launch(&mut self, min_conf: f32, max_drafts: usize, st: &Stream) -> Result<(), CudaError> {
        if self.sched_cycle.is_none() {
            self.sched_cycle = Some(DevBuf::from_slice(&Self::VERIFY_CYCLE_MS)?);
        }
        if self.sched_out.is_none() {
            self.sched_out = Some(DevBuf::zeroed(4)?);
        }
        let cycle = self.sched_cycle.as_ref().unwrap();
        let out = self.sched_out.as_ref().unwrap();
        let conf_len = self.cfg.block_size.min(max_drafts);
        schedule_verify_len_dev(&self.scratch.conf, conf_len, min_conf, max_drafts, cycle, out, st)
    }
}

pub fn tap_mean_bf16_dev(
    streams: &DevBuf,
    taps_bf16: &mut DevBuf,
    rows: usize,
    dim: usize,
    tap_idx: usize,
    n_taps: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0 {
        return Ok(());
    }
    unsafe {
        CudaError::check(ffi::gb_dsv4_mtp_tap_mean_bf16(
            streams.ptr() as *const f32,
            taps_bf16.ptr(),
            i32_of(rows)?,
            i32_of(dim)?,
            i32_of(tap_idx)?,
            i32_of(n_taps)?,
            st.raw(),
        ))
    }
}

/// The device mirror of [`DsparkGpu::schedule_verify_len`] over a confidence buffer: the host schedule's loop as a
/// single kernel (same clamp, same floor break, same accumulation order, same cycle table, same `>=` tie-break),
/// so its result is bit-for-bit the host's for the same vector. `cycle` must hold
/// [`DsparkGpu::VERIFY_CYCLE_MS`] and `out` one int; no host wait (the caller orders the result's copy/read).
pub fn schedule_verify_len_dev(
    conf: &DevBuf,
    conf_len: usize,
    min_conf: f32,
    max_drafts: usize,
    cycle: &DevBuf,
    out: &DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    unsafe {
        CudaError::check(ffi::gb_dsv4_mtp_schedule_verify_len(
            conf.ptr() as *const f32,
            i32_of(conf_len)?,
            min_conf,
            i32_of(max_drafts)?,
            cycle.ptr() as *const f32,
            out.ptr() as *mut i32,
            st.raw(),
        ))?
    }
    Ok(())
}

// -------------------------------------------------------------------------------------------------------- the lattice

/// The captured candidate lattice of one DSpark round ([`DsparkGpu::draft_n_lattice`], T2.5a): the pieces a tree
/// round's search runs on, in `gb_spec::tree::Lattice`'s own layout — `ids`/`unary` are `[rows][k]` flattened,
/// `edge` is `[rows][k parents][k candidates]` flattened, level 0's parent row 0 being the anchor (`ids`' slot 0).
///
/// Every level's candidate set carries the markov walk's own pick first (the full-vocabulary bias can rank the pick
/// outside the unary top-k, so the unary ranking alone could lose it), followed by the best unary candidates; a
/// short set is padded with [`u32::MAX`] at `-inf` unary, which the search's non-finite guard skips. The edge term
/// between a parent candidate and a candidate of the next level is the same `markov_head . markov_embed` product
/// the walk adds before its argmax, so `unary + edge` at the picked indices is the walk's own score.
#[derive(Clone, Debug, PartialEq)]
pub struct DsparkLattice {
    pub rows: usize,
    pub k: usize,
    pub ids: Vec<u32>,
    pub unary: Vec<f32>,
    pub edge: Vec<f32>,
    /// The walk's picks, one per level: the chain this round would draft without trees.
    pub picks: Vec<u32>,
}

/// One row of a `[vocab, rank]` markov table — f32 as loaded, or bf16 widened exactly (the bit shift is the
/// reference's own bf16 -> f32 conversion).
fn markov_row(table: &DevBuf, tok: usize, rank: usize, vocab: usize) -> Result<Vec<f32>, CudaError> {
    if tok >= vocab {
        return Ok(vec![0.0; rank]);
    }
    let bf16 = table.bytes() == vocab * rank * 2;
    let off = tok * if bf16 { rank * 2 } else { rank * 4 };
    if bf16 {
        let mut raw = vec![0u16; rank];
        table.download_range(off, &mut raw)?;
        Ok(raw.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect())
    } else {
        let mut v = vec![0f32; rank];
        table.download_range(off, &mut v)?;
        Ok(v)
    }
}

/// Assemble the round's lattice after the walk: the unary logits were read before the walk biased them; the picks
/// (`ids`' slots `1..`) and the anchor (slot 0) are read now, the edge terms come from the markov tables' rows.
/// `None` when the stack carries no markov tables (a toy configuration).
fn capture_lattice(
    last: &MtpStage,
    sc: &MtpScratch,
    cfg: &MtpConfig,
    walk_rows: usize,
    k: usize,
    unary_logits: &[f32],
    st: &Stream,
) -> Result<Option<DsparkLattice>, CudaError> {
    let (Some(embed), Some(head)) = (&last.markov_embed, &last.markov_head) else {
        return Ok(None);
    };
    let (rows, vocab, rank) = (walk_rows, cfg.vocab, cfg.markov_rank);
    let k = k.clamp(2, 8); // the search never branches wider than the engine's tuned maximum
    st.sync()?;
    let mut ids = vec![0i32; rows + 1];
    sc.ids.download_range(0, &mut ids)?;
    let mut cands: Vec<Vec<u32>> = Vec::with_capacity(rows);
    let mut unary: Vec<f32> = Vec::with_capacity(rows * k);
    let mut picks = Vec::with_capacity(rows);
    for d in 0..rows {
        let row = &unary_logits[d * vocab..(d + 1) * vocab];
        let pick = ids[d + 1] as u32;
        picks.push(pick);
        // the best `k - 1` unary candidates besides the pick (value descending, ties by lowest id), kept sorted
        let mut best: Vec<(f32, u32)> = Vec::with_capacity(k);
        for (t, &v) in row.iter().enumerate() {
            let t = t as u32;
            if t == pick {
                continue;
            }
            let at = best.partition_point(|&(bv, bt)| bv > v || (bv == v && bt < t));
            if at >= k - 1 {
                continue; // not among the k - 1 best
            }
            best.insert(at, (v, t));
            best.truncate(k - 1);
        }
        let mut level: Vec<u32> = Vec::with_capacity(k);
        level.push(pick);
        level.extend(best.iter().map(|&(_, t)| t));
        for &t in level.iter() {
            unary.push(if t == u32::MAX { f32::NEG_INFINITY } else { row[t as usize] });
        }
        for _ in level.len()..k {
            level.push(u32::MAX);
            unary.push(f32::NEG_INFINITY);
        }
        cands.push(level);
    }
    // the edge terms: level d's candidates scored against every parent candidate of level d - 1 (level 0's single
    // parent is the anchor)
    let mut edge = vec![0f32; rows * k * k];
    let mut embeds: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
    let mut heads: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
    for d in 0..rows {
        let anchor_parent = [ids[0] as u32];
        let parents: &[u32] = if d == 0 { &anchor_parent } else { &cands[d - 1] };
        for (p, &pt) in parents.iter().enumerate().take(k) {
            if pt == u32::MAX {
                continue;
            }
            let er = match embeds.get(&pt) {
                Some(v) => v,
                None => embeds.entry(pt).or_insert(markov_row(embed, pt as usize, rank, vocab)?),
            };
            for (c, &ct) in cands[d].iter().enumerate() {
                if ct == u32::MAX {
                    continue;
                }
                let hr = match heads.get(&ct) {
                    Some(v) => v,
                    None => heads.entry(ct).or_insert(markov_row(head, ct as usize, rank, vocab)?),
                };
                let e = er.iter().zip(hr.iter()).map(|(a, b)| a * b).sum::<f32>();
                edge[(d * k + p) * k + c] = e;
            }
        }
    }
    let mut ids_flat = Vec::with_capacity(rows * k);
    for level in &cands {
        ids_flat.extend_from_slice(level);
    }
    Ok(Some(DsparkLattice { rows, k, ids: ids_flat, unary, edge, picks }))
}

/// Keyed Gumbel sampling parameters for the device Markov walk when `temperature > 0.0`.
#[derive(Clone, Copy, Debug)]
pub struct MtpWalkSampling {
    pub cand_k: usize,
    pub seed: u64,
    pub temperature: f64,
    pub top_p: f64,
    pub min_p: f64,
    pub base_pos: u64,
}

/// Device-resident per-round parameters for the DFlash/MTP draft round graphs (must mirror
/// `csrc/dsv4_round_params.cuh`'s `Dsv4MtpRoundParams` exactly — field order, widths, padding).
///
/// Every value that varies round-to-round — the anchor token, the advancing `start_pos`, the derived
/// `reach` / `n_kv` and the keyed-walk sampling parameters — lives here on the device, rewritten by an
/// async H2D copy from pinned staging before each capture/replay, so a graph captured for a `(slot,
/// seeded, keyed, walk_rows)` shape replays unchanged as positions advance.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Dsv4MtpRoundParams {
    pub anchor: i32,
    pub start_pos: i32,
    pub reach: i32,
    pub n_kv: i32,
    pub window: i32,
    pub block_size: i32,
    pub cand_k: i32,
    pub _pad: i32,
    pub seed: u64,
    pub base_pos: u64,
    pub temperature: f64,
    pub top_p: f64,
    pub min_p: f64,
}

impl Dsv4MtpRoundParams {
    pub const SIZE: usize = 72;
}

/// The cache key of one draft-round graph. `owner` fingerprints the model instance (the scratch arena's
/// device address), `slot` the active window ring, `seeded` the pre-seeded/unseeded round shape,
/// `keyed` whether the keyed Gumbel walk runs (vs the fast unkeyed walk), `walk_rows` the walk length
/// (which changes launch counts).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DraftRoundKey {
    pub owner: u64,
    pub slot: u16,
    pub seeded: bool,
    pub keyed: bool,
    pub walk_rows: u16,
}

/// Consecutive capture/launch refusals after which the draft-round graph feature disables itself for this instance
/// (a broken capture path must degrade to eager silently, not retry every round).
const REFUSAL_DISABLE_AT: u32 = 4;

/// Process-level switch for the draft-round CUDA graphs (default on). It is an API knob, not an environment
/// switch: tests hold it off to run the eager reference, on to exercise the graph path. A capture/launch refusal
/// never turns it off silently — each refusal falls back to the eager body for that round.
static ROUND_GRAPHS_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Disable (`false`) or enable (`true`) the draft-round CUDA graphs for the process; returns the previous value.
pub fn set_round_graphs_enabled(v: bool) -> bool {
    ROUND_GRAPHS_ENABLED.swap(v, std::sync::atomic::Ordering::Relaxed)
}

fn round_graphs_enabled() -> bool {
    ROUND_GRAPHS_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}
