//! Ling-owned fused KDA decode stage: `csrc/ling_kda_fused.cu` runs the three independent per-layer stages that follow
//! the qkv/f/g projections (prep, the b_proj beta gate, the full-rank f/g gate) as ONE launch instead of three sequential
//! launches. Every element computes the same expressions in the same order as the separate kernels (`kda.cu`'s
//! `prep_kernel<true>`, `beta_from_b_kernel<X_BF16,1>`, `full_fg_gate_kernel<1>`), so the outputs are bit-identical; only
//! the launch count and the inter-kernel serialization change. Used by `bailing_moe_v3` for decode/verify windows of up to
//! 8 rows (the serial decode step and the verification the harness drives); every other shape keeps the unfused path.

use crate::kda::{self, Batch, Pool, Record, Weights, Workspace};
use crate::rt::{CudaError, DevBuf, Stream};

mod ffi {
    crate::launchers! {
        pub fn gb_ling_kda_prep_beta_fgg(
            qkv: *const core::ffi::c_void,
            x: *const core::ffi::c_void,
            b_weight: *const core::ffi::c_void,
            conv_w: *const core::ffi::c_void,
            conv_pool: *const core::ffi::c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            f_out: *const core::ffi::c_void,
            g_out: *const core::ffi::c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            lower: f32,
            n_seq: i32,
            rows: i32,
            heads: i32,
            hidden: i32,
            emit_q: i32,
            skip_full: i32,
            raw_out: *mut core::ffi::c_void,
            q: *mut f32,
            k: *mut f32,
            v: *mut core::ffi::c_void,
            g_decay: *mut f32,
            z_gate: *mut f32,
            beta: *mut f32,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_kda_window_and_norm(
            q: *const f32,
            k: *const f32,
            v: *const core::ffi::c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            parity: *const i32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            heads: i32,
            rz: i32,
            r_step: i32,
            y: *mut core::ffi::c_void,
            head_cnt: *mut i32,
            eps: f32,
            z: *const f32,
            norm_w: *const core::ffi::c_void,
            out: *mut core::ffi::c_void,
            o_suh: *const core::ffi::c_void,
            o_xh: *mut core::ffi::c_void,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_kda_prep_beta_fgg_window_norm(
            qkv: *const core::ffi::c_void,
            x: *const core::ffi::c_void,
            b_weight: *const core::ffi::c_void,
            conv_w: *const core::ffi::c_void,
            conv_pool: *const core::ffi::c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            f_out: *const core::ffi::c_void,
            g_out: *const core::ffi::c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            lower: f32,
            n_seq: i32,
            rows: i32,
            heads: i32,
            hidden: i32,
            emit_q: i32,
            skip_full: i32,
            raw_out: *mut core::ffi::c_void,
            q: *mut f32,
            k: *mut f32,
            v: *mut core::ffi::c_void,
            g_decay: *mut f32,
            z_gate: *mut f32,
            beta: *mut f32,
            s: *mut f32,
            parity: *const i32,
            rz: i32,
            r_step: i32,
            y: *mut core::ffi::c_void,
            head_cnt: *mut i32,
            eps: f32,
            z: *mut f32,
            norm_w: *const core::ffi::c_void,
            out: *mut core::ffi::c_void,
            o_suh: *const core::ffi::c_void,
            o_xh: *mut core::ffi::c_void,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_kda_prep(
            qkv: *const core::ffi::c_void,
            in_f32: i32,
            conv_w: *const core::ffi::c_void,
            conv_pool: *const core::ffi::c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            rows: i32,
            heads: i32,
            emit_q: i32,
            skip_full: i32,
            raw_out: *mut core::ffi::c_void,
            q: *mut f32,
            k: *mut f32,
            v: *mut core::ffi::c_void,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_kda_replay(
            k: *const f32,
            v: *const core::ffi::c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            parity: *const i32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            heads: i32,
            skip_full: i32,
            r_step: i32,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_kda_commit_tail_all(
            layers: *const core::ffi::c_void,
            n_layers: i32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            conv_dim: i32,
            flip: i32,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_kda_prefill_prep_gate_fgg(
            qkv: *const core::ffi::c_void,
            x: *const core::ffi::c_void,
            b_weight: *const core::ffi::c_void,
            conv_w: *const core::ffi::c_void,
            conv_pool: *const core::ffi::c_void,
            seq_start: *const i32,
            slot: *const i32,
            f_out: *const core::ffi::c_void,
            g_out: *const core::ffi::c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            lower: f32,
            n_seq: i32,
            rows: i32,
            heads: i32,
            hidden: i32,
            raw_out: *mut core::ffi::c_void,
            q: *mut f32,
            k: *mut f32,
            v: *mut core::ffi::c_void,
            g_decay: *mut f32,
            z_gate: *mut f32,
            beta: *mut f32,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_kda_prefill_chunk_norm_commit(
            q: *const f32,
            k: *const f32,
            v: *const core::ffi::c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            parity: *mut i32,
            slot: i32,
            rows: i32,
            heads: i32,
            rz: i32,
            y: *mut core::ffi::c_void,
            z: *const f32,
            norm_w: *const core::ffi::c_void,
            eps: f32,
            out: *mut core::ffi::c_void,
            raw: *const core::ffi::c_void,
            conv_pool: *mut core::ffi::c_void,
            head_cnt: *mut i32,
            stream: *mut core::ffi::c_void,
            ko_suh: *const core::ffi::c_void,
            ko_xh: *mut core::ffi::c_void,
            ) -> i32;
    }
}

/// One launch, `rows <= 8` (checked), replacing `kda::gate_from_full_fg` + `kda::prep` for the decode profile
/// (fp32 qkv, emit q, no keep, raw rows to the record).
#[allow(clippy::too_many_arguments)]
pub fn prep_gate_fused(
    st: &Stream,
    d: kda::Dims,
    w: &Weights<'_>,
    qkv: &DevBuf,
    pool: &Pool,
    batch: &Batch<'_>,
    rec: &mut Record,
    ws: &mut Workspace,
    f_out: &DevBuf,
    g_out: &DevBuf,
    x: &DevBuf,
    b_weight: &DevBuf,
    a_log: &DevBuf,
    dt_bias: &DevBuf,
    lower: f32,
) -> Result<(), CudaError> {
    debug_assert!(batch.rows <= 8 && batch.n_seq >= 1);
    let rc = unsafe {
        ffi::gb_ling_kda_prep_beta_fgg(
            qkv.ptr(),
            x.ptr(),
            b_weight.ptr(),
            w.conv_w.ptr(),
            pool.conv.ptr(),
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            core::ptr::null(),
            f_out.ptr(),
            g_out.ptr(),
            a_log.ptr() as *const f32,
            dt_bias.ptr() as *const f32,
            lower,
            batch.n_seq as i32,
            batch.rows as i32,
            d.heads as i32,
            d.hidden as i32,
            1, // emit_q
            0, // skip_full
            rec.qkv.ptr(),
            ws.q().ptr() as *mut f32,
            ws.k().ptr() as *mut f32,
            ws.v().ptr(),
            rec.g.ptr() as *mut f32,
            ws.z_mut().ptr() as *mut f32,
            rec.beta.ptr() as *mut f32,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// One launch, `rows <= 8` (checked), replacing `kda::window` (write_final, no keep) + `kda::norm` on the decode
/// profile: the `state_kernel<R, true, RZ>` body, then the `norm_kernel<1>` math of the last of the `G_X` blocks per
/// `(seq, head)` (a per-`(seq, head)` atomic counter in `head_cnt`, self-reset every call, decides it). Bit-identical to
/// the two separate launches. `head_cnt` must hold `n_seq * heads` zeroed i32s; `out` the fp16 `[rows, heads * 128]`
/// norm output (`s.gated`).
#[allow(clippy::too_many_arguments)]
pub fn window_and_norm(
    st: &Stream,
    d: kda::Dims,
    p: &kda::Params,
    pool: &kda::Pool,
    batch: &kda::Batch<'_>,
    rec: &kda::Record,
    ws: &mut kda::Workspace,
    head_cnt: &DevBuf,
    w: &kda::Weights<'_>,
    o_suh: Option<&DevBuf>,
    o_xh: Option<&mut DevBuf>,
    out: &mut DevBuf,
) -> Result<(), CudaError> {
    debug_assert!(batch.rows <= 8 && batch.n_seq >= 1);
    debug_assert!(batch.n_seq * d.heads <= head_cnt.bytes() / 4);
    let rc = unsafe {
        ffi::gb_ling_kda_window_and_norm(
            ws.q().ptr() as *const f32,
            ws.k().ptr() as *const f32,
            ws.v().ptr(),
            rec.g.ptr() as *const f32,
            rec.beta.ptr() as *const f32,
            pool.s.ptr() as *mut f32,
            pool.parity.ptr() as *const i32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            batch.n_seq as i32,
            d.heads as i32,
            (p.round == kda::Round::TowardZero) as i32,
            p.r_step as i32,
            ws.y().ptr() as *mut core::ffi::c_void,
            head_cnt.ptr() as *mut i32,
            p.eps,
            ws.z().ptr() as *const f32,
            w.o_norm.ptr() as *const core::ffi::c_void,
            out.ptr() as *mut core::ffi::c_void,
            o_suh.map_or(core::ptr::null(), |s| s.ptr()),
            o_xh.map_or(core::ptr::null_mut(), |x| x.ptr()),
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// One launch replacing `prep_gate_fused` + `window_and_norm` on the decode/verify profile
/// (`rows <= 8`, `n_seq` within `head_cnt`): prep + beta + fgg + window + norm in a single
/// `dim3(gx, heads, n_seq)` launch of 128-thread blocks. Bit-identical to the sequential pair
/// (every q/k/v/raw_out/g_decay/z/beta/y/S/out/o_xh element computes the same expressions in
/// the same order as the separate launches).
#[allow(clippy::too_many_arguments)]
pub fn prep_beta_fgg_window_norm_fused(
    st: &Stream,
    d: kda::Dims,
    p: &kda::Params,
    w: &Weights<'_>,
    qkv: &DevBuf,
    pool: &Pool,
    batch: &Batch<'_>,
    rec: &mut Record,
    ws: &mut Workspace,
    head_cnt: &DevBuf,
    f_out: &DevBuf,
    g_out: &DevBuf,
    x: &DevBuf,
    b_weight: &DevBuf,
    a_log: &DevBuf,
    dt_bias: &DevBuf,
    lower: f32,
    o_suh: Option<&DevBuf>,
    o_xh: Option<&mut DevBuf>,
    out: &mut DevBuf,
) -> Result<(), CudaError> {
    debug_assert!(batch.rows <= 8 && batch.n_seq >= 1);
    debug_assert!(batch.n_seq * d.heads <= head_cnt.bytes() / 4);
    let rc = unsafe {
        ffi::gb_ling_kda_prep_beta_fgg_window_norm(
            qkv.ptr(),
            x.ptr(),
            b_weight.ptr(),
            w.conv_w.ptr(),
            pool.conv.ptr(),
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            core::ptr::null(),
            f_out.ptr(),
            g_out.ptr(),
            a_log.ptr() as *const f32,
            dt_bias.ptr() as *const f32,
            lower,
            batch.n_seq as i32,
            batch.rows as i32,
            d.heads as i32,
            d.hidden as i32,
            1, // emit_q
            0, // skip_full
            rec.qkv.ptr(),
            ws.q().ptr() as *mut f32,
            ws.k().ptr() as *mut f32,
            ws.v().ptr(),
            rec.g.ptr() as *mut f32,
            ws.z_mut().ptr() as *mut f32,
            rec.beta.ptr() as *mut f32,
            pool.s.ptr() as *mut f32,
            pool.parity.ptr() as *const i32,
            (p.round == kda::Round::TowardZero) as i32,
            p.r_step as i32,
            ws.y().ptr() as *mut core::ffi::c_void,
            head_cnt.ptr() as *mut i32,
            p.eps,
            ws.z_mut().ptr() as *mut f32,
            w.o_norm.ptr() as *const core::ffi::c_void,
            out.ptr() as *mut core::ffi::c_void,
            o_suh.map_or(core::ptr::null(), |s| s.ptr()),
            o_xh.map_or(core::ptr::null_mut(), |x| x.ptr()),
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// Replay the accepted prefix of every sequence from the OLD committed state without running `commit_tail_kernel`.
#[allow(clippy::too_many_arguments)]
pub fn replay(
    st: &Stream,
    d: kda::Dims,
    w: &Weights<'_>,
    p: &kda::Params,
    pool: &Pool,
    batch: &Batch<'_>,
    rec: &Record,
    ws: &mut Workspace,
    keep: Option<&DevBuf>,
    window_wrote_final: bool,
) -> Result<(), CudaError> {
    let keep_ptr = keep.map_or(core::ptr::null(), |k| k.ptr() as *const i32);
    if !keep_ptr.is_null() || !window_wrote_final {
        let rc = unsafe {
            ffi::gb_kda_prep(
                rec.qkv.ptr(),
                0,
                w.conv_w.ptr(),
                pool.conv.ptr(),
                batch.seq_start.ptr() as *const i32,
                batch.slot.ptr() as *const i32,
                keep_ptr,
                batch.n_seq as i32,
                batch.rows as i32,
                d.heads as i32,
                0,
                window_wrote_final as i32,
                core::ptr::null_mut(),
                ws.q().ptr() as *mut f32,
                ws.k().ptr() as *mut f32,
                ws.v().ptr(),
                st.raw(),
            )
        };
        CudaError::check(rc)?;
        let rc = unsafe {
            ffi::gb_kda_replay(
                ws.k().ptr() as *const f32,
                ws.v().ptr(),
                rec.g.ptr() as *const f32,
                rec.beta.ptr() as *const f32,
                pool.s.ptr() as *mut f32,
                pool.parity.ptr() as *const i32,
                batch.seq_start.ptr() as *const i32,
                batch.slot.ptr() as *const i32,
                keep_ptr,
                batch.n_seq as i32,
                d.heads as i32,
                window_wrote_final as i32,
                p.r_step as i32,
                st.raw(),
            )
        };
        CudaError::check(rc)?;
    }
    Ok(())
}

/// Commit the tail (conv shift + parity flip) of every KDA layer in `layers_d` in ONE batched launch.
pub fn commit_tail_all(
    st: &Stream,
    d: kda::Dims,
    layers_d: &DevBuf,
    n_layers: usize,
    batch: &Batch<'_>,
    keep: Option<&DevBuf>,
    flip: bool,
) -> Result<(), CudaError> {
    assert!(n_layers >= 1 && layers_d.bytes() >= n_layers * 24, "kda: commit_tail_all layer table too small");
    let keep_ptr = keep.map_or(core::ptr::null(), |k| k.ptr() as *const i32);
    let rc = unsafe {
        ffi::gb_ling_kda_commit_tail_all(
            layers_d.ptr(),
            n_layers as i32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            keep_ptr,
            batch.n_seq as i32,
            d.conv_dim() as i32,
            flip as i32,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// One launch replacing `kda::gate_from_full_fg` + `kda::prep_prefill_bf16` on the P2 prefill path
/// (`prefill && !p1_dense`): the `beta_from_b_kernel<X_BF16, 4>` + `full_fg_gate_kernel<8>` bodies of the gate and
/// the `prep_kernel<false>` body (tile_rows = 64) dispatched across `blockIdx.y` of one 256-thread kernel.
/// Bit-identical per element to the three separate launches; every output byte is exactly the source kernels'.
/// `Rows` is the prefill chunk's (any >= 1); `n_seq == 1` (a prefill chunk is one sequence).
#[allow(clippy::too_many_arguments)]
pub fn prep_gate_fused_prefill(
    st: &Stream,
    d: kda::Dims,
    w: &Weights<'_>,
    qkv: &DevBuf,
    pool: &Pool,
    batch: &Batch<'_>,
    rec: &mut Record,
    ws: &mut Workspace,
    f_out: &DevBuf,
    g_out: &DevBuf,
    x: &DevBuf,
    b_weight: &DevBuf,
    a_log: &DevBuf,
    dt_bias: &DevBuf,
    lower: f32,
) -> Result<(), CudaError> {
    debug_assert!(batch.n_seq == 1);
    let rc = unsafe {
        ffi::gb_ling_kda_prefill_prep_gate_fgg(
            qkv.ptr(),
            x.ptr(),
            b_weight.ptr(),
            w.conv_w.ptr(),
            pool.conv.ptr(),
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            f_out.ptr(),
            g_out.ptr(),
            a_log.ptr() as *const f32,
            dt_bias.ptr() as *const f32,
            lower,
            batch.n_seq as i32,
            batch.rows as i32,
            d.heads as i32,
            d.hidden as i32,
            core::ptr::null_mut(), // raw_out: bf16 qkv already holds the rounded rows
            ws.q().ptr() as *mut f32,
            ws.k().ptr() as *mut f32,
            ws.v().ptr(),
            rec.g.ptr() as *mut f32,
            ws.z_mut().ptr() as *mut f32,
            rec.beta.ptr() as *mut f32,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// One launch replacing `kda::window_chunked` + `kda::norm` + `kda::commit_from` on the P2 prefill path
/// (`prefill && !p1_dense`): the `chunk_kernel<64>` body (glm_p2_kda.cu, verbatim), then the LAST CTA per head
/// (gridDim.x = 2, a per-head self-resetting atomic counter in `head_cnt`, `heads` zeroed i32s) runs the exact
/// `norm_kernel<4>` math for that head over every row and the exact `commit_tail_kernel` body for the head's
/// channels (whole-chunk prefill: keep = null, seq 0, conv shift off `raw` + parity flip). Bit-identical per
/// element to the three separate launches; every output byte is exactly the source kernels'.
#[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
pub fn chunk_norm_commit_fused(
    st: &Stream,
    d: kda::Dims,
    p: &kda::Params,
    pool: &mut kda::Pool,
    slot: usize,
    rows: usize,
    rec: &kda::Record,
    ws: &mut kda::Workspace,
    head_cnt: &DevBuf,
    w: &kda::Weights<'_>,
    raw: *const core::ffi::c_void,
    out: &mut DevBuf,
    ko_suh: Option<&DevBuf>,
    ko_xh: Option<&mut DevBuf>,
) -> Result<(), CudaError> {
    debug_assert!(rows >= 1 && slot < pool.n_slots && d.heads == pool.dims.heads);
    debug_assert!(d.heads <= head_cnt.bytes() / 4);
    let rc = unsafe {
        ffi::gb_ling_kda_prefill_chunk_norm_commit(
            ws.q().ptr() as *const f32,
            ws.k().ptr() as *const f32,
            ws.v().ptr(),
            rec.g.ptr() as *const f32,
            rec.beta.ptr() as *const f32,
            pool.s.ptr() as *mut f32,
            pool.parity.ptr() as *mut i32,
            slot as i32,
            rows as i32,
            d.heads as i32,
            (p.round == kda::Round::TowardZero) as i32,
            ws.y().ptr() as *mut core::ffi::c_void,
            ws.z().ptr() as *const f32,
            w.o_norm.ptr() as *const core::ffi::c_void,
            p.eps,
            out.ptr() as *mut core::ffi::c_void,
            raw,
            pool.conv.ptr() as *mut core::ffi::c_void,
            head_cnt.ptr() as *mut i32,
            st.raw(),
            ko_suh.map_or(core::ptr::null(), |d| d.ptr()),
            ko_xh.as_ref().map_or(core::ptr::null_mut(), |d| d.ptr()),
        )
    };
    CudaError::check(rc)
}
