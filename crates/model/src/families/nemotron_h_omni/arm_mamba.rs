//! The layer arm for the Mamba2 mixer: `in_proj`, split, conv and SSD scan (`nemotron_ssm::{pre, window}`), gated norm, `out_proj`.
//!
//! One layer, in stream order:
//!
//! 1. `in_proj.forward` (f32 in, f32 out) gives `[rows, 18560]`; `split_inproj` cuts it into z (rounded to the bf16 grid), xBC and dt.
//! 2. `pre` runs the causal conv over the conv history and the rows, and fills a [`Record`] with the post-conv rows, dt, g and decay.
//!    It reads the conv pool and never writes it.
//! 3. `window` runs the recurrence over the rows from the committed S and writes the bf16 core output. It reads S and never writes it.
//! 4. `gated_norm` (bf16 out) and `out_proj.forward` (bf16 in, f32 out) give `bufs.mix`.
//! 5. A prefill slice commits at once (`window_commit` stores S in the pass that reads it, then `commit_conv`; keep = every row of each
//!    sequence). A window leaves the state alone: its record and its f32 xBC rows stay in the layer's own buffers until
//!    [`NemotronModel::commit`] replays the kept rows.
//!
//! A FUSED decode window (`StepCtx::apply`, `NemotronModel::window_candidates_fused`) is a window whose single-row sequences are also
//! committed inside the layer: `window_commit_sel` stores their state in the pass that reads it (bit for bit `window` + `commit_state`)
//! and `commit_conv` moves their conv history, so their `commit` only moves the head's taps and the slot's length. The records stay per
//! layer (the sequences of several rows are still the plain window's).
//!
//! A prefill slice commits inside the layer, so one shared record serves every layer. A window needs a record per layer (the next
//! layers must not overwrite what a later commit replays), and its xBC rows are written by `split_inproj` straight into the layer's
//! retained buffer, so no copy is made.
//!
//! [`NemotronModel::commit`]: super::model::NemotronModel

use super::device::{LayerKind, Mixer};
use super::model::{Core, Kind, StepCtx};
use super::tp::splits;
use super::tp_arm::linear_tp;
use crate::error::ModelError;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::nemotron_glue::{split_inproj, XBC_COLS, Z_COLS};
use gb_cuda::nemotron_ssm::{self, Weights};

/// Whether a P2 prefill slice's tensor-core `in_proj` writes the split itself (lane S5; the same bytes either way): on unless
/// `REXL3_NEMO_P2_FUSE_SPLIT=0` (read once; for A/B timing).
fn fused_split() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REXL3_NEMO_P2_FUSE_SPLIT").map_or(true, |v| v.trim() != "0"))
}

/// Whether layer `m`'s in_proj takes the fused split (P2 prefill slice, single rank, the P2 linear active and supported, not opted out).
fn fused_in_ok(c: &Core, in_proj: &gb_cuda::exl3_linear::Exl3Weights, x: &StepCtx<'_>) -> bool {
    x.kind == Kind::Prefill
        && c.tp.topology().is_single()
        && fused_split()
        && gb_cuda::nemotron_dense_mma::active(x.rows)
        && gb_cuda::nemotron_dense_mma::Layer::of_weights(in_proj).supported()
}

/// The layer loop asks before layer `li`'s norm: true when the layer is a Mamba layer whose in_proj takes the fused split, so
/// the norm may write in_proj's rotated input into `bufs.ws` itself (`nemotron_dense_mma::norm_rot`) instead of `bufs.xn`; unless
/// `REXL3_NEMO_P2_NORM_ROT=0` (read once). Lane S5.
pub fn prerot_ok(c: &Core, li: usize, x: &StepCtx<'_>) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("REXL3_NEMO_P2_NORM_ROT").map_or(true, |v| v.trim() != "0")) {
        return false;
    }
    let Some(Mixer::Mamba(m)) = c.w.layer(li).map(|l| &l.mixer) else {
        return false;
    };
    fused_in_ok(c, &m.in_proj, x) && m.in_proj.k == 4096
}

/// Whether the fused in_proj split stores z and xBC as bf16 (lane S5; the same bits): on unless `REXL3_NEMO_P2_ZX_BF16=0` (read once).
fn zx_bf16_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REXL3_NEMO_P2_ZX_BF16").map_or(true, |v| v.trim() != "0"))
}

/// Whether a P2 prefill slice's gated norm writes out_proj's rotated input itself (lane S5; the same bits): on unless
/// `REXL3_NEMO_P2_GATED_ROT=0` (read once).
fn gated_rot_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REXL3_NEMO_P2_GATED_ROT").map_or(true, |v| v.trim() != "0"))
}

/// Bytes a slot needs in each direction of the head-group state re-replication: the S of 64 heads (64 x 128 x 64 f32) and the conv
/// history of the rank's channels (3 rows x (4,096 + 512 + 512) bf16).
pub const HG_STATE_BYTES_PER_SLOT: usize = 64 * 128 * 64 * 4 + 3 * (4096 + 512 + 512) * 2;

/// Reads `c.bufs.xn` for `x.rows` rows and writes `c.bufs.mix`. Layer `li` must be a mamba layer.
pub fn run(c: &mut Core, li: usize, x: &StepCtx<'_>) -> Result<bool, ModelError> {
    let rows = x.rows;
    let Some(ord) = c.map.ordinal_of(li, LayerKind::Mamba) else {
        return Err(ModelError::Request(format!("layer {li} is not a Mamba layer on the device")));
    };
    let Some(Mixer::Mamba(m)) = c.w.layer(li).map(|l| &l.mixer) else {
        return Err(ModelError::Request(format!("layer {li} is not a Mamba layer on the device")));
    };
    let resid_ok = c.residual_fused(x);
    let fused_in = fused_in_ok(c, &m.in_proj, x);
    // the layer loop's norm may have written in_proj's rotated input already (`prerot_ok`); take the flag either way
    let prerotated = std::mem::take(&mut c.prerot);
    assert!(!prerotated || fused_in, "the norm rotated in_proj's input for a layer that does not take the fused in_proj");
    // contract P2 at two ranks, a prefill slice: the head-group split (`run_hg`); every other step below, unchanged
    if x.kind == Kind::Prefill
        && m.in_proj_hg.is_some()
        && c.tp.topology().ranks == 2
        && c.tp.supports_swap()
        && gb_cuda::nemotron_dense_mma::active(rows)
        && c.bufs.tp_state.is_some()
        && c.bufs.tp_pack.is_some()
        && super::contract::p2tp_on(super::contract::MAMBA_HG_VAR)
    {
        // the head-group path writes `bufs.mix` (out_proj, replicated) and never fuses the residual: the layer loop adds it
        super::tp_arm::count_head_split(rows);
        return run_hg(c, li, ord, x).map(|()| false);
    }
    let st = &c.stream;

    // `in_proj` is N-split across the TP ranks when the plan slices it (two ranks: rank 0 columns [0, 9216), rank 1 [9216, 18560); at
    // three and more the plan replicates it, `shard` says whole and nothing below is issued): scatter and one allreduce fill the whole
    // `[rows, 18560]` row, and everything after it is replicated. On a single rank this is the plain `linear_rows` call.
    let xbc = match x.kind {
        Kind::Prefill => &mut c.bufs.xbc,
        Kind::Window => &mut c.win_xbc[ord],
    };
    // contract P2, a single rank's prefill slice: the tensor-core in_proj writes z, xBC and dt itself (lane S5: the bytes of the f32
    // output followed by `split_inproj`, without the f32 `[rows, 18560]` pass); otherwise the linear, then the split
    // and with it z and xBC stored as bf16 (the values gated_norm and the conv read them as: `nemotron_ssm::bf16_in`)
    let zx_bf16 = fused_in && zx_bf16_on();
    if fused_in {
        gb_cuda::nemotron_dense_mma::linear_split(
            &m.in_proj,
            &c.bufs.xn,
            Dtype::F32,
            rows,
            &mut c.bufs.z,
            xbc,
            &mut c.bufs.dt,
            Z_COLS,
            XBC_COLS,
            zx_bf16,
            prerotated,
            &mut c.bufs.ws,
            st,
        )?;
    } else {
        // (contract P2 at two ranks: the packed exchange, `tp_arm::linear_tp_p2`; otherwise exactly `linear_tp`; lane S. The split's
        // destinations for its fused path are a prefill slice's z, xBC and dt; rc2b merge: xBC through `xbc`, which is `bufs.xbc` then)
        let split_done = super::tp_arm::linear_tp_p2(
            &*c.tp,
            &m.in_proj,
            &c.bufs.xn,
            Dtype::F32,
            &mut c.bufs.proj,
            c.bufs.tp_tmp.as_mut(),
            c.bufs.tp_pack.as_mut(),
            match x.kind {
                Kind::Prefill => Some((&mut c.bufs.z, &mut *xbc, &mut c.bufs.dt)),
                Kind::Window => None,
            },
            c.bufs.tp_aux.as_ref(),
            rows,
            splits::IN_PROJ_COLS,
            &mut c.bufs.ws,
            st,
        )?;
        if !split_done {
            split_inproj(&c.bufs.proj, rows, &mut c.bufs.z, xbc, &mut c.bufs.dt, st)?;
        }
    }

    let rec = match x.kind {
        Kind::Prefill => &mut c.pre_rec,
        Kind::Window => &mut c.win_rec[ord],
    };
    let wts = Weights { conv_w: &m.conv_w, conv_b: &m.conv_b, a_log: &m.a_log, dt_bias: &m.dt_bias, d: &m.d };
    if zx_bf16 {
        nemotron_ssm::bf16_in::pre(st, xbc, &c.bufs.dt, &wts, &c.mamba[ord].conv, &x.batch, rec)?;
    } else {
        nemotron_ssm::pre(st, xbc, &c.bufs.dt, &wts, &c.mamba[ord].conv, &x.batch, rec)?;
    }
    match x.kind {
        // a slice commits every row of every sequence: the pass that reads S also stores the state after the last row
        Kind::Prefill => nemotron_ssm::window_commit(st, rec, &mut c.mamba[ord].s, &m.d, &x.batch, &mut c.bufs.ssm_y)?,
        Kind::Window => match (x.apply, x.lag) {
            // a LAGGED fused window: the state of an applied sequence is stored once every K steps; the rows between are pending in the
            // layer's ring and re-applied here (the same advance over the same bits)
            (Some(apply), Some(lag)) => nemotron_ssm::window_lag(
                st,
                rec,
                &mut c.mamba[ord].s,
                &m.d,
                &x.batch,
                apply,
                lag,
                &mut c.lag_rings[ord],
                &mut c.bufs.ssm_y,
            )?,
            // a fused decode window: the sequences whose one row is kept at once store their state in the pass that reads it
            // (bit for bit `window` + `commit_state`), the others are `window`'s: nothing is stored
            (Some(apply), None) => {
                nemotron_ssm::window_commit_sel(st, rec, &mut c.mamba[ord].s, &m.d, &x.batch, apply, &mut c.bufs.ssm_y)?
            }
            (None, _) => nemotron_ssm::window(st, rec, &c.mamba[ord].s, &m.d, &x.batch, &mut c.bufs.ssm_y)?,
        },
    }
    // a P2 prefill slice of a single rank adds out_proj's output into the residual stream in the GEMM epilogue (`Core::residual_fused`)
    let out_ok = gb_cuda::nemotron_dense_mma::Layer::of_weights(&m.out_proj).supported();
    let fused = resid_ok && out_ok;
    // and (lane S5) the gated norm writes out_proj's rotated input itself: the bits of gated_norm + the linear's input rotation
    let gated_rot = x.kind == Kind::Prefill
        && c.tp.topology().is_single()
        && gated_rot_on()
        && gb_cuda::nemotron_dense_mma::active(rows)
        && out_ok;
    if gated_rot {
        gb_cuda::nemotron_dense_mma::gated_rot(
            &c.bufs.ssm_y,
            &c.bufs.z,
            zx_bf16,
            &m.gate_norm,
            nemotron_ssm::NORM_EPS,
            &m.out_proj,
            rows,
            &mut c.bufs.ws,
            st,
        )?;
        let y = if fused { &mut c.bufs.x } else { &mut c.bufs.mix };
        gb_cuda::nemotron_dense_mma::linear_prerotated(&m.out_proj, &c.bufs.ws, y, fused, rows, st)?;
    } else {
        if zx_bf16 {
            nemotron_ssm::bf16_in::gated_norm(st, &c.bufs.ssm_y, &c.bufs.z, &m.gate_norm, rows, &mut c.bufs.gated)?;
        } else {
            nemotron_ssm::gated_norm(st, &c.bufs.ssm_y, &c.bufs.z, &m.gate_norm, rows, &mut c.bufs.gated)?;
        }
        if fused {
            gb_cuda::nemotron_dense_mma::linear_residual(
                &m.out_proj,
                &c.bufs.gated,
                Dtype::BF16,
                &mut c.bufs.x,
                rows,
                &mut c.bufs.ws,
                st,
            )?;
        } else {
            linear_tp(
                &*c.tp,
                &m.out_proj,
                &c.bufs.gated,
                Dtype::BF16,
                &mut c.bufs.mix,
                c.bufs.tp_tmp.as_mut(),
                rows,
                splits::HIDDEN,
                &mut c.bufs.ws,
                st,
            )?;
        }
    }

    if x.kind == Kind::Prefill && zx_bf16 {
        nemotron_ssm::bf16_in::commit_conv(st, xbc, &mut c.mamba[ord].conv, &x.batch, &c.keep)?;
    } else if x.kind == Kind::Prefill {
        nemotron_ssm::commit_conv(st, xbc, &mut c.mamba[ord].conv, &x.batch, &c.keep)?;
    } else if let Some(apply) = x.apply {
        // the conv history of the applied sequences (`apply` is their keep: 1 for a one-row sequence, 0 leaves a history alone)
        nemotron_ssm::commit_conv(st, xbc, &mut c.mamba[ord].conv, &x.batch, apply)?;
    }
    Ok(fused)
}

/// Contract P2 at two TP ranks, a prefill slice (lane S): the Mamba layer split by HEAD GROUPS instead of in_proj columns. Rank r owns
/// B/C groups [4 r, 4 r + 4) = heads [64 r, 64 r + 64). It computes the in_proj columns of its heads only (`in_proj_hg`: z and x of its
/// heads, B and C of its groups, dt of every head), the conv of its channels, the scan of its heads, the gated norm of its groups (each a
/// per-channel, per-head or per-group computation: `docs/NEMOTRON_PREFILL_SPEED.md` 6.10), then exchanges its half of the gated output
/// (bf16: the bytes out_proj reads) and the state it committed (S of its heads, conv history of its channels) so that both ranks hold the
/// whole layer as the single node does; out_proj is replicated, as before. No in_proj exchange.
fn run_hg(c: &mut Core, li: usize, ord: usize, x: &StepCtx<'_>) -> Result<(), ModelError> {
    use gb_cuda::nemotron_ssm::{HeadRange, CONV_DIM, CONV_STATE_ROWS, HEAD_DIM, INNER, STATE};
    let rows = x.rows;
    let Some(Mixer::Mamba(m)) = c.w.layer(li).map(|l| &l.mixer) else {
        return Err(ModelError::Request(format!("layer {li} is not a Mamba layer on the device")));
    };
    let w_hg = m.in_proj_hg.as_ref().expect("checked by run");
    let rank = c.tp.topology().rank;
    let (hr, peer) = (HeadRange::of_rank(rank), HeadRange::of_rank(1 - rank));
    let st = &c.stream;
    let tmp = c
        .bufs
        .tp_tmp
        .as_mut()
        .ok_or_else(|| ModelError::Config("tensor parallelism: no temp for the head-group in_proj".into()))?;
    // 1. the rank's in_proj columns, then where split_inproj writes them
    super::model::linear_rows(w_hg, &c.bufs.xn, tmp, rows, Dtype::F32, Dtype::F32, &mut c.bufs.ws, st)?;
    let (zo, _) = hr.inner_range();
    gb_cuda::nemotron_tp::split_hg(
        tmp,
        rows,
        4096,
        4096,
        512,
        zo,
        zo,
        hr.grp_lo * STATE,
        &mut c.bufs.z,
        &mut c.bufs.xbc,
        &mut c.bufs.dt,
        st,
    )?;
    // 2. conv (the rank's channels; dt of every head), scan (its heads, committing their S), gated norm (its groups)
    let wts = Weights { conv_w: &m.conv_w, conv_b: &m.conv_b, a_log: &m.a_log, dt_bias: &m.dt_bias, d: &m.d };
    nemotron_ssm::pre_range(st, &c.bufs.xbc, &c.bufs.dt, &wts, &c.mamba[ord].conv, &x.batch, &mut c.pre_rec, hr)?;
    nemotron_ssm::window_commit_range(st, &c.pre_rec, &mut c.mamba[ord].s, &m.d, &x.batch, &mut c.bufs.ssm_y, hr)?;
    nemotron_ssm::gated_norm_range(st, &c.bufs.ssm_y, &c.bufs.z, &m.gate_norm, rows, &mut c.bufs.gated, hr)?;
    // 3. the gated halves: rows x 4,096 bf16 each way (a byte copy of the operand out_proj reads)
    let (send, recv) = {
        let p = c.bufs.tp_pack.as_mut().expect("checked by run");
        (&mut p.0, &mut p.1)
    };
    let (row_b, half_b) = (INNER * 2, hr.n_heads() * HEAD_DIM * 2);
    send.copy_2d_from(0, half_b, &c.bufs.gated, hr.inner_range().0 * 2, row_b, half_b, rows, st)?;
    c.tp.swap_bytes(send, 0, rows * half_b, recv, 0, rows * half_b, st)?;
    c.bufs.gated.copy_2d_from(peer.inner_range().0 * 2, row_b, recv, 0, half_b, half_b, rows, st)?;
    // 4. out_proj, replicated: the single node's call on the whole gated row
    linear_tp(
        &*c.tp,
        &m.out_proj,
        &c.bufs.gated,
        Dtype::BF16,
        &mut c.bufs.mix,
        c.bufs.tp_tmp.as_mut(),
        rows,
        splits::HIDDEN,
        &mut c.bufs.ws,
        st,
    )?;
    // 5. the conv history of the rank's channels, then the committed state re-replicated: per slot of the slice, S of the rank's heads
    // and the conv rows of its channels one way, the peer's the other (byte copies: both ranks then hold the single node's pools)
    nemotron_ssm::commit_conv_range(st, &c.bufs.xbc, &mut c.mamba[ord].conv, &x.batch, &c.keep, hr)?;
    let mut slots: Vec<usize> = Vec::new();
    for &s in x.row_index.host_rows().0.iter().take(rows) {
        if !slots.contains(&(s as usize)) {
            slots.push(s as usize);
        }
    }
    let (ss, sr) = {
        let p = c.bufs.tp_state.as_mut().expect("checked by run");
        (&mut p.0, &mut p.1)
    };
    let s_slot = nemotron_ssm::S_SLOT_ELEMS * 4;
    let s_head = STATE * HEAD_DIM * 4;
    let s_half = hr.n_heads() * s_head;
    let conv_row = CONV_DIM * 2;
    let conv_slot = CONV_STATE_ROWS * conv_row;
    let conv_half: usize = hr.conv_ranges().iter().map(|(_, n)| n * 2 * CONV_STATE_ROWS).sum();
    let per = s_half + conv_half;
    if slots.len() * per > ss.bytes() {
        return Err(ModelError::Request(format!(
            "head-group state exchange: {} slots, buffer for {}",
            slots.len(),
            ss.bytes() / per
        )));
    }
    let pack = |buf: &mut gb_cuda::DevBuf,
                s_pool: &gb_cuda::DevBuf,
                conv: &gb_cuda::DevBuf,
                h: HeadRange|
     -> Result<(), gb_cuda::CudaError> {
        for (i, &slot) in slots.iter().enumerate() {
            let at = i * per;
            buf.copy_range_from(at, s_pool, slot * s_slot + h.head_lo() * s_head, s_half, st)?;
            let mut o = at + s_half;
            for (base, n) in h.conv_ranges() {
                buf.copy_2d_from(o, n * 2, conv, slot * conv_slot + base * 2, conv_row, n * 2, CONV_STATE_ROWS, st)?;
                o += n * 2 * CONV_STATE_ROWS;
            }
        }
        Ok(())
    };
    pack(ss, &c.mamba[ord].s, &c.mamba[ord].conv, hr)?;
    c.tp.swap_bytes(ss, 0, slots.len() * per, sr, 0, slots.len() * per, st)?;
    for (i, &slot) in slots.iter().enumerate() {
        let at = i * per;
        c.mamba[ord].s.copy_range_from(slot * s_slot + peer.head_lo() * s_head, sr, at, s_half, st)?;
        let mut o = at + s_half;
        for (base, n) in peer.conv_ranges() {
            c.mamba[ord].conv.copy_2d_from(slot * conv_slot + base * 2, conv_row, sr, o, n * 2, n * 2, CONV_STATE_ROWS, st)?;
            o += n * 2 * CONV_STATE_ROWS;
        }
    }
    Ok(())
}
