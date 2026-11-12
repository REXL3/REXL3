//! The layer arm for the latent MoE mixer: the decode block (`NemotronMoe::forward_in`) for up to `WINDOW_ROWS` rows and the large-row
//! prefill block (`PrefillMoe::forward_prefill_in`) above that, each on the model's shared scratch.
//!
//! The block reads f32 rows and writes f32 rows, and a row's bits do not depend on how many rows share the call or on which path ran it
//! (`tests/nemotron_moe_prefill.rs` gates the two paths bit for bit), so a window and a prefill slice of any size agree row by row. The
//! scratches are shared by all 40 layers: the layers run one after another on one stream. A window and a slice of up to 64 rows take the
//! decode path (a graph captures it); a wider slice takes the prefill path, which reads each expert's weights once for the whole slice
//! instead of once per 64 rows. A layer the prefill path does not run (an expert in another width or codebook: none in the pack) takes
//! the decode path in calls of 64 rows.

use super::device::{LayerKind, Mixer};
use super::model::{Core, StepCtx};
use super::tp_arm::{moe_decode, prefill_moe_runs};
use super::tp_balance;
use crate::error::ModelError;
use gb_cuda::nemotron_moe::HIDDEN;

/// Whether layer `li` will run P2's tensor-core MoE block on a single rank for this step (lane S5): the layer loop may then make the
/// block's fc1 / shared-up inputs in the norm (`PrefillMoe::norm_rotate_prefill`), unless `REXL3_NEMO_P2_NORM_ROT=0` (read once).
pub fn mma_block_single(c: &Core, li: usize, x: &StepCtx<'_>) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("REXL3_NEMO_P2_NORM_ROT").map_or(true, |v| v.trim() != "0")) {
        return false;
    }
    let Some(Mixer::Moe(moe)) = c.w.layer(li).map(|l| &l.mixer) else {
        return false;
    };
    x.kind == super::model::Kind::Prefill
        && c.tp.topology().is_single()
        && gb_cuda::nemotron_dense_mma::active(x.rows)
        && c.moe_prefill.as_ref().is_some_and(|p| x.rows <= p.max_rows())
        && c.moe_mma.is_some()
        && prefill_moe_runs(&*c.tp)
        && moe.supports_prefill()
}

/// Reads `c.bufs.xn` for `x.rows` rows and writes `c.bufs.mix`, or (`Ok(true)`: a P2 prefill slice with no tap, on one rank (lane S5)
/// or on a rank of two after the `y` exchange (6.11)) adds the block's output into `c.bufs.x` itself. Layer `li` must be a moe layer.
pub fn run(c: &mut Core, li: usize, x: &StepCtx<'_>) -> Result<bool, ModelError> {
    let resid_ok = c.residual_fused(x);
    let resid_tp_ok = c.residual_fused_moe_tp(x);
    if c.map.kind(li) != Some(LayerKind::Moe) {
        return Err(ModelError::Request(format!("layer {li} is not a MoE layer on the device")));
    }
    let Some(Mixer::Moe(moe)) = c.w.layer(li).map(|l| &l.mixer) else {
        return Err(ModelError::Request(format!("layer {li} is not a MoE layer on the device")));
    };
    let rows = x.rows;
    let decode_rows = c.moe.max_rows();
    // The scratches are sized for the widest step and a step is checked against that before it runs, but the wrappers assert rather
    // than error, so a row count past them must never get there.
    let wide = c.moe_prefill.as_ref().map_or(0, |p| p.max_rows());
    if rows == 0 || rows > decode_rows.max(wide) {
        return Err(ModelError::Request(format!("{rows} rows in the MoE scratches of {decode_rows} and {wide}")));
    }
    // Tensor parallelism: a rank holds its half of the routed experts (the other half are absent: their pairs are exact zeros) and the
    // `[rows, 22, 1024]` routed down projection `y` is allreduced between the projection and the combine (`tp_arm::moe_decode`). On a
    // single rank both calls are the plain `forward_in` and `forward_prefill_in`.
    let comm = &*c.tp;
    // contract P2: a prefill slice of any width takes the prefill block (its dense projections are P2's inside the scope): a short tail
    // slice through the decode kernels would keep P1's bits and chunked != whole. A window (no scope) is P1 at every width.
    let p2_prefill = gb_cuda::nemotron_dense_mma::active(rows) && c.moe_prefill.is_some();
    let mut fused = false;
    if rows <= decode_rows && !p2_prefill {
        if let Some(m) = c.moe_mma.as_mut() {
            m.clear_pre();
        }
        moe_decode(comm, moe, &mut c.moe, &c.bufs.xn, rows, &mut c.bufs.mix, &c.stream)?;
        // the routing trace tap (tooling: `tp_balance`, off by default): the picks of this layer's route. A stream synchronisation, so
        // never with CUDA graphs on (the tracing test runs with them off).
        if tp_balance::tap_enabled() {
            let (picks, _) = c.moe.download_route(rows, &c.stream)?;
            tp_balance::tap_push(li, rows, picks);
        }
    } else if let (true, true, Some(p)) = (prefill_moe_runs(comm), moe.supports_prefill(), c.moe_prefill.as_mut()) {
        // under P2 a prefill slice takes the tensor-core block (lane S2); a window, or P1, the fp32 block
        let mma = if p2_prefill { c.moe_mma.as_mut() } else { None };
        match mma {
            // a single rank's P2 block with the residual add in fc2's epilogue (lane S5): x += routed + shared, bit for bit x + mix.
            // (At two ranks the same fusion is the `resid` of `moe_prefill_with` below: the `y` exchange runs before the combine and
            // fc2 is replicated, so the `+ x` is on the complete output, never on a rank's partial.)
            Some(m) if resid_ok && comm.topology().is_single() && moe.supports_prefill_mma_resid() => {
                moe.forward_prefill_mma_resid(p, m, &c.bufs.xn, rows, &mut c.bufs.x, &c.stream)?;
                fused = true;
            }
            mma => {
                // P2 at two ranks: the residual add fused into fc2's epilogue after the exchange (6.11), writing `x` in place of `mix`
                let resid = resid_tp_ok && mma.is_some() && !comm.topology().is_single() && moe.supports_prefill_mma_resid();
                let ypack = if p2_prefill { c.bufs.tp_ypack.as_mut() } else { None };
                // P2 at two ranks, a wide slice: the two row halves' blocks on two streams, so the first half's `y` exchange is on the wire
                // while the second half computes (rows are independent under P2 (a): the same bits as one call). REXL3_NEMO_P2TP_YOVERLAP=0
                // opts out.
                let split = p2_prefill
                    && rows >= 1024
                    && ypack.is_some()
                    && c.moe_half.as_ref().is_some_and(|h| h.prefill.max_rows() >= rows / 2)
                    && super::contract::p2tp_on("REXL3_NEMO_P2TP_YOVERLAP");
                if split {
                    let half = c.moe_half.as_mut().expect("checked above");
                    super::tp_arm::count_yoverlap();
                    let hrows = rows / 2;
                    let (a, b) = (rows - hrows, hrows);
                    let st = &c.stream;
                    // the second stream may start as soon as the input rows are ready
                    gb_cuda::graph::fork(st, &half.stream, &half.ev[0])?;
                    let row = HIDDEN * 4;
                    let (xa, xb) = (c.bufs.xn.subview(0, a * row), c.bufs.xn.subview(a * row, b * row));
                    // the halves write disjoint rows of `x` (fused) or of `mix`
                    let dst = if resid { &c.bufs.x } else { &c.bufs.mix };
                    let (mut oa, mut ob) = (dst.subview(0, a * row), dst.subview(a * row, b * row));
                    super::tp_arm::moe_prefill_with(comm, moe, p, mma, ypack, Some(&half.ev[2]), &xa, a, &mut oa, st, resid)?;
                    super::tp_arm::moe_prefill_with(
                        comm,
                        moe,
                        &mut half.prefill,
                        Some(&mut half.mma),
                        Some(&mut half.ypack),
                        Some(&half.ev[3]),
                        &xb,
                        b,
                        &mut ob,
                        &half.stream,
                        resid,
                    )?;
                    gb_cuda::graph::join(st, &half.stream, &half.ev[1])?;
                } else {
                    let ev = c.moe_half.as_ref().map(|h| &h.ev[2]);
                    let dst = if resid { &mut c.bufs.x } else { &mut c.bufs.mix };
                    super::tp_arm::moe_prefill_with(comm, moe, p, mma, ypack, ev, &c.bufs.xn, rows, dst, &c.stream, resid)?;
                }
                fused = resid;
            }
        }
        // the routing trace tap (tooling, off by default; a stream synchronisation): the large-row path's picks too, so the routing of a
        // prefill under P1 and under P2 can be compared (lane S, `p2route`)
        if tp_balance::tap_enabled() {
            let (picks, _) = p.download_route(rows, &c.stream)?;
            tp_balance::tap_push(li, rows, picks);
        }
    } else {
        if let Some(m) = c.moe_mma.as_mut() {
            m.clear_pre();
        }
        // the decode path in calls of its own width: the same bits, one read of the experts' weights per call
        let mut start = 0;
        while start < rows {
            let n = (rows - start).min(decode_rows);
            let h = c.bufs.xn.subview(start * HIDDEN * 4, n * HIDDEN * 4);
            let mut o = c.bufs.mix.subview(start * HIDDEN * 4, n * HIDDEN * 4);
            moe_decode(comm, moe, &mut c.moe, &h, n, &mut o, &c.stream)?;
            start += n;
        }
    }
    Ok(fused)
}
