//! The layer arm for the attention mixer: q, k, v, KV write, then attention by the one position rule, then `o`.
//!
//! No RoPE, no bias, no q or k norm. The rows' K and V are rounded to fp16 once, at the cache write (`kv_write`, at each row's absolute
//! position through the per-step row index), and attention reads the cache only. With `--kv-dtype int8` the same write rounds them to
//! int8 at a power-of-two scale per (token, kv head) and the kernels dequantise exactly into their fp16 tiles (`docs/NEMOTRON_KV_INT8.md`):
//! one write path for prefill, windows and decode, so every invariance below holds under int8 too.
//!
//! **The rule** (`gb_cuda::nemotron_attn_prefill::attn_long`): a row at absolute position below `FLASH_FROM_POS` (8,192) is attended
//! by the exact reduction tree (`tools/attn_gpu_design.md`), a row at or above it by the flash tensor-core kernel, **whichever entry
//! point carries it: a decode row, a verify-window row or a prefill row**. The choice is the row's, from its position alone, so a
//! row's bits do not depend on the kind of call, the rows around it or how a prompt was sliced, at every position: decode == verify ==
//! prefill, window == serial, chunked == whole, cached == uncached, with one path id for the model (`prefix::PATH_ID_FLASH`). A
//! call that cannot hold a row at or above the threshold (its launch is built for at most 16 chunks) is the tree's call, launch for
//! launch, as before. `docs/NEMOTRON_ATTN_LONG.md`.

use super::device::{LayerKind, Mixer};
use super::model::{Core, StepCtx};
use super::tp::splits;
use super::tp_arm::{linear_tp, qkv_tp};
use crate::error::ModelError;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::nemotron_attn::kv_write;
use gb_cuda::nemotron_attn_prefill::attn_long;

/// Reads `c.bufs.xn` for `x.rows` rows and writes `c.bufs.mix`. Layer `li` must be a attn layer.
pub fn run(c: &mut Core, li: usize, x: &StepCtx<'_>) -> Result<bool, ModelError> {
    let rows = x.rows;
    let Some(ord) = c.map.ordinal_of(li, LayerKind::Attn) else {
        return Err(ModelError::Request(format!("layer {li} is not an attention layer on the device")));
    };
    let Some(Mixer::Attn(a)) = c.w.layer(li).map(|l| &l.mixer) else {
        return Err(ModelError::Request(format!("layer {li} is not an attention layer on the device")));
    };
    let resid_ok = c.residual_fused(x);
    let st = &c.stream;

    // q, k and v are N-split across the TP ranks and land block by block in one merged buffer under ONE allreduce (the views `merged`
    // holds); on a single rank they are the three plain `linear_rows` calls into `bufs.q`, `bufs.k` and `bufs.v`.
    let merged = {
        let (b, tp) = (&mut c.bufs, &*c.tp);
        qkv_tp(tp, a, &b.xn, (&mut b.q, &mut b.k, &mut b.v), b.tp_qkv.as_mut(), b.tp_tmp.as_mut(), rows, &mut b.ws, st)?
    };
    let (q, k, v) = match &merged {
        Some((q, k, v)) => (q, k, v),
        None => (&c.bufs.q, &c.bufs.k, &c.bufs.v),
    };

    // the KV cache and the attention kernels are replicated and unchanged
    let mut kv = c.kv_layer(ord);
    kv_write(&mut kv, &c.table, x.row_index, k, v, st)?;
    attn_long(q, &kv, &c.table, x.row_index, x.batch.n_seq, &mut c.attn, &mut c.bufs.attn_o, st)?;

    let (b, tp) = (&mut c.bufs, &*c.tp);
    // a P2 prefill slice of a single rank adds o's output into the residual stream in the GEMM epilogue (`Core::residual_fused`)
    if resid_ok && gb_cuda::nemotron_dense_mma::Layer::of_weights(&a.o).supported() {
        gb_cuda::nemotron_dense_mma::linear_residual(&a.o, &b.attn_o, Dtype::F32, &mut b.x, rows, &mut b.ws, st)?;
        return Ok(true);
    }
    linear_tp(tp, &a.o, &b.attn_o, Dtype::F32, &mut b.mix, b.tp_tmp.as_mut(), rows, splits::O_COLS, &mut b.ws, st)?;
    Ok(false)
}
