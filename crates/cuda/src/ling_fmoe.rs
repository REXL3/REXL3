//! Ling-owned FrostMoE decode (`csrc/ling_fmoe.cu`): bit-identical to [`fmoe::forward`] on every input, with the
//! group-limited router caching each lane's 16 sigmoid/biased scores in registers (1x `expf` per expert instead of the
//! legacy 9x reload), the router GEMV fused into the route launch on the Ling-3.1 shape ([`forward_with_router`]:
//! one 32-CTA launch writes the fp32 logits and routes, then gate/up and down), and the 4-CTA/SM
//! `ling_gate_up_kernel` whose atomic counters (like `ling_down_kernel`'s) self-reset, so no zero-fill launches are
//! needed. Used by `bailing_moe_v3` for decode/verify windows (trunk and the L72 MTP drafter).
#![allow(clippy::too_many_arguments)]

use crate::fmoe::{GroupRoute, LogitsDtype, Shape, SwigluLimit};
use crate::fn_moe::ExpertTables;
use crate::rt::{CudaError, DevBuf, Stream};

/// Refuses a buffer smaller than what a launch touches, before anything is queued (`CudaError(1)`, as an invalid value).
fn need(b: &DevBuf, bytes: usize, what: &str) -> Result<(), CudaError> {
    if b.bytes() < bytes {
        eprintln!("{what}: buffer of {} bytes, the launch touches {bytes}", b.bytes());
        return Err(CudaError(1));
    }
    Ok(())
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_ling_fmoe_workspace_bytes(max_tokens: i32, top_k: i32, h: i32, i: i32, kc: i32) -> usize;
        pub fn gb_ling_fmoe_forward(
            x: *const c_void, logits: *const c_void, logits_f32: i32, shared_gate: *const c_void,
            g_t: *const u64, g_su: *const u64, g_sv: *const u64,
            u_t: *const u64, u_su: *const u64, u_sv: *const u64,
            d_t: *const u64, d_su: *const u64, d_sv: *const u64,
            ktab: *const i32, out: *mut c_void,
            t: i32, ne: i32, top_k: i32, h: i32, i: i32, kc: i32, max_tokens: i32,
            ws: *mut c_void, stream: *mut c_void,
            expert_bias: *const f32, n_group: i32, topk_group: i32,
            routed_limit: f32, shared_limit: f32, routed_scaling: f32,
        ) -> i32;
        pub fn gb_ling_fmoe_forward_with_router(
            x: *const c_void, router_w: *const c_void, logits: *mut c_void, shared_gate: *const c_void,
            g_t: *const u64, g_su: *const u64, g_sv: *const u64,
            u_t: *const u64, u_su: *const u64, u_sv: *const u64,
            d_t: *const u64, d_su: *const u64, d_sv: *const u64,
            ktab: *const i32, out: *mut c_void,
            t: i32, ne: i32, top_k: i32, h: i32, i: i32, kc: i32, max_tokens: i32,
            ws: *mut c_void, stream: *mut c_void,
            expert_bias: *const f32, n_group: i32, topk_group: i32,
            routed_limit: f32, shared_limit: f32, routed_scaling: f32,
            hidden: *const c_void, norm_w: *const f32, norm_out: *mut c_void, eps: f32,
            suh_p0: *const c_void, xh_p0: *mut c_void,
            suh_p1: *const c_void, xh_p1: *mut c_void,
            suh_p2: *const c_void, xh_p2: *mut c_void,
        ) -> i32;
        pub fn gb_ling_fmoep_workspace_bytes(max_tokens: i32, top_k: i32, ne: i32, h: i32, i: i32) -> usize;
        pub fn gb_ling_fmoep_forward_with_epilogue(
            x: *const c_void, logits: *const c_void, logits_f32: i32, shared_gate: *const c_void,
            g_t: *const u64, g_su: *const u64, g_sv: *const u64,
            u_t: *const u64, u_su: *const u64, u_sv: *const u64,
            d_t: *const u64, d_su: *const u64, d_sv: *const u64,
            ktab: *const i32, out: *mut c_void,
            t: i32, ne: i32, top_k: i32, h: i32, i: i32, max_tokens: i32,
            ws: *mut c_void, stream: *mut c_void,
            expert_bias: *const f32, n_group: i32, topk_group: i32,
            routed_limit: f32, shared_limit: f32, routed_scaling: f32,
            hidden: *const c_void, norm_w: *const f32, norm_out: *mut c_void, eps: f32,
            next_suh: *const c_void, next_xh: *mut c_void,
        ) -> i32;
    }
}

/// Workspace bytes for up to `max_tokens` rows a call (the same layout as [`fmoe::workspace_bytes`]).
pub fn workspace_bytes(max_tokens: usize, top_k: usize, h: usize, i: usize, kc: usize) -> usize {
    unsafe { ffi::gb_ling_fmoe_workspace_bytes(max_tokens as i32, top_k as i32, h as i32, i as i32, kc as i32) }
}

/// The C ABI `logits_f32` flag (mirrors `fmoe::LogitsDtype::code`, which is private to `fmoe`).
fn logits_ffi_code(ld: LogitsDtype) -> i32 {
    match ld {
        LogitsDtype::Bf16 => 0,
        LogitsDtype::F32 => 1,
    }
}

/// One MoE layer for `shape.t` rows: route (top-k softmax of `logits` `[t, ne]` of `ld` — bf16, or fp32 for
/// Ling-3.1's `router_dtype = "fp32"`; shared expert weighted by sigmoid(`shared_gate` bf16 `[t]`)), every expert
/// run once for all rows that picked it, `out` bf16 `[t, h]`. Bit-identical to [`fmoe::forward`] with the same
/// arguments. `gr` (`None`: legacy softmax) and `limits` select the group-limited noaux_tc router and the SwiGLU
/// clamps.
pub fn forward(
    x: &DevBuf,
    logits: &DevBuf,
    ld: LogitsDtype,
    shared_gate: &DevBuf,
    tb: &ExpertTables<'_>,
    out: &mut DevBuf,
    shape: Shape,
    gr: Option<GroupRoute<'_>>,
    limits: SwigluLimit,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    let Shape { t, ne, top_k, h, i, kc, max_tokens } = shape;
    if t == 0 || t > max_tokens {
        eprintln!("rexl3 level=error launch contract: ling fmoe rows {t} outside 1..={max_tokens}");
        return Err(CudaError(1));
    }
    need(x, t * h * 2, "ling fmoe x")?;
    need(logits, t * ne * ld.size(), "ling fmoe logits")?;
    need(shared_gate, t * 2, "ling fmoe shared gate")?;
    tb.check(ne + 1, "ling fmoe")?;
    need(out, t * h * 2, "ling fmoe out")?;
    let op = out.ptr();
    need(ws, workspace_bytes(max_tokens, top_k, h, i, kc), "ling fmoe workspace")?;
    let wp = ws.ptr();
    let [gt, gsu, gsv, ut, usu, usv, dt, dsu, dsv] = tb.ptrs();
    let (bias_ptr, n_group, topk_group) = match gr {
        Some(g) => (g.expert_bias.map_or(core::ptr::null(), |b| b.ptr() as *const f32), g.n_group as i32, g.topk_group as i32),
        None => (core::ptr::null(), 0, 0),
    };
    let rc = unsafe {
        ffi::gb_ling_fmoe_forward(
            x.ptr(),
            logits.ptr(),
            logits_ffi_code(ld),
            shared_gate.ptr(),
            gt,
            gsu,
            gsv,
            ut,
            usu,
            usv,
            dt,
            dsu,
            dsv,
            tb.ktab.ptr() as *const i32,
            op,
            t as i32,
            ne as i32,
            top_k as i32,
            h as i32,
            i as i32,
            kc as i32,
            max_tokens as i32,
            wp,
            st.raw(),
            bias_ptr,
            n_group,
            topk_group,
            limits.routed,
            limits.shared,
            gr.map_or(0.0, |g| g.scale),
        )
    };
    CudaError::check(rc)
}

/// The post-MoE fused epilogue of [`forward_with_router`]: the down kernel's last finishing warp runs the residual
/// add (`hidden + out`) + the next layer's input norm (`norm_w` -> `norm_out`) + the next consumer's rotation
/// (`prerot`: the next KDA triple, MLA pair, or the lm_head single), replacing the separate
/// `norm_rot_in_{triple,pair,single}` launch. Only valid on the fused-router Ling-3.1 decode shape (see
/// [`forward_with_router`]).
pub struct FmoeEpilogue<'a> {
    pub hidden: &'a DevBuf,
    pub norm_w: &'a DevBuf,
    pub norm_out: &'a mut DevBuf,
    pub eps: f32,
    pub prerot: crate::ling_kda_proj_fused::Prerot<'a>,
}

/// Decode MoE with the router GEMV fused into the route launch (Ling-3.1 shapes only): when group routing, fp32
/// logits, `t <= 8`, `h == 4096` and `ne == 512`, the router logits are computed AND routed in one launch
/// (`gb_ling_fmoe_forward_with_router`), then gate/up and down — 3 launches instead of 4. The `logits` buffer is an
/// output on that path (same bits as [`ling_kda_proj_fused::bf16_linear_f32`]); on any other shape it must already
/// hold the logits (the caller computed them), and the two-launch path runs. Bit-identical to [`forward`] on every
/// input. `epilogue` (when given) additionally fuses the post-MoE residual add + next input norm + next consumer's
/// rotation into the down kernel; it requires the fused-router shape (an error otherwise).
pub fn forward_with_router(
    x: &DevBuf,
    router: &DevBuf,
    logits: &mut DevBuf,
    ld: LogitsDtype,
    shared_gate: &DevBuf,
    tb: &ExpertTables<'_>,
    out: &mut DevBuf,
    shape: Shape,
    gr: Option<GroupRoute<'_>>,
    limits: SwigluLimit,
    ws: &mut DevBuf,
    st: &Stream,
    epilogue: Option<FmoeEpilogue<'_>>,
) -> Result<(), CudaError> {
    let Shape { t, ne, top_k, h, i, kc, max_tokens } = shape;
    if t == 0 || t > max_tokens {
        eprintln!("rexl3 level=error launch contract: ling fmoe rows {t} outside 1..={max_tokens}");
        return Err(CudaError(1));
    }
    need(x, t * h * 2, "ling fmoe x")?;
    need(router, ne * h * 2, "ling fmoe router")?;
    need(logits, t * ne * ld.size(), "ling fmoe logits")?;
    need(shared_gate, t * 2, "ling fmoe shared gate")?;
    tb.check(ne + 1, "ling fmoe")?;
    need(out, t * h * 2, "ling fmoe out")?;
    need(ws, workspace_bytes(max_tokens, top_k, h, i, kc), "ling fmoe workspace")?;
    if let Some(e) = &epilogue {
        need(e.hidden, t * h * 2, "ling fmoe epilogue hidden")?;
        need(e.norm_w, h * 4, "ling fmoe epilogue norm_w")?;
        need(e.norm_out, t * h * 2, "ling fmoe epilogue norm_out")?;
        if let Some(x0) = &e.prerot.xh0 {
            need(x0, t * h * 2, "ling fmoe epilogue xh0")?;
        }
        if let Some(x1) = &e.prerot.xh1 {
            need(x1, t * h * 2, "ling fmoe epilogue xh1")?;
        }
        if let Some(x2) = &e.prerot.xh2 {
            need(x2, t * h * 2, "ling fmoe epilogue xh2")?;
        }
    }
    let fused = ld == LogitsDtype::F32
        && t <= 8
        && h == 4096
        && ne == 512
        && gr.as_ref().map_or(false, |g| g.expert_bias.is_some() && g.n_group >= 2 && g.topk_group >= 1);
    if epilogue.is_some() && !fused {
        return Err(CudaError(1));
    }
    let (bias_ptr, n_group, topk_group, scale) = match gr {
        Some(g) => {
            (g.expert_bias.map_or(core::ptr::null(), |b| b.ptr() as *const f32), g.n_group as i32, g.topk_group as i32, g.scale)
        }
        None => (core::ptr::null(), 0, 0, 0.0),
    };
    let [gt, gsu, gsv, ut, usu, usv, dt, dsu, dsv] = tb.ptrs();
    if !fused {
        crate::ling_kda_proj_fused::bf16_linear_f32(x, router, logits, t, h, ne, st)?;
    }
    let op = out.ptr();
    let wp = ws.ptr();
    let (hid, nw, no, eps_v) = match &epilogue {
        Some(e) => (e.hidden.ptr(), e.norm_w.ptr() as *const f32, e.norm_out.ptr(), e.eps),
        None => (core::ptr::null_mut(), core::ptr::null(), core::ptr::null_mut(), 0.0),
    };
    let (s0, x0) = match &epilogue {
        Some(e) => (
            e.prerot.suh0.map_or(core::ptr::null(), |d| d.ptr()),
            e.prerot.xh0.as_ref().map_or(core::ptr::null_mut(), |d| d.ptr()),
        ),
        None => (core::ptr::null(), core::ptr::null_mut()),
    };
    let (s1, x1) = match &epilogue {
        Some(e) => (
            e.prerot.suh1.map_or(core::ptr::null(), |d| d.ptr()),
            e.prerot.xh1.as_ref().map_or(core::ptr::null_mut(), |d| d.ptr()),
        ),
        None => (core::ptr::null(), core::ptr::null_mut()),
    };
    let (s2, x2) = match &epilogue {
        Some(e) => (
            e.prerot.suh2.map_or(core::ptr::null(), |d| d.ptr()),
            e.prerot.xh2.as_ref().map_or(core::ptr::null_mut(), |d| d.ptr()),
        ),
        None => (core::ptr::null(), core::ptr::null_mut()),
    };
    let rc = if fused {
        unsafe {
            ffi::gb_ling_fmoe_forward_with_router(
                x.ptr(),
                router.ptr(),
                logits.ptr() as *mut core::ffi::c_void,
                shared_gate.ptr(),
                gt,
                gsu,
                gsv,
                ut,
                usu,
                usv,
                dt,
                dsu,
                dsv,
                tb.ktab.ptr() as *const i32,
                op,
                t as i32,
                ne as i32,
                top_k as i32,
                h as i32,
                i as i32,
                kc as i32,
                max_tokens as i32,
                wp,
                st.raw(),
                bias_ptr,
                n_group,
                topk_group,
                limits.routed,
                limits.shared,
                scale,
                hid,
                nw,
                no,
                eps_v,
                s0,
                x0,
                s1,
                x1,
                s2,
                x2,
            )
        }
    } else {
        unsafe {
            ffi::gb_ling_fmoe_forward(
                x.ptr(),
                logits.ptr(),
                logits_ffi_code(ld),
                shared_gate.ptr(),
                gt,
                gsu,
                gsv,
                ut,
                usu,
                usv,
                dt,
                dsu,
                dsv,
                tb.ktab.ptr() as *const i32,
                op,
                t as i32,
                ne as i32,
                top_k as i32,
                h as i32,
                i as i32,
                kc as i32,
                max_tokens as i32,
                wp,
                st.raw(),
                bias_ptr,
                n_group,
                topk_group,
                limits.routed,
                limits.shared,
                scale,
            )
        }
    };
    CudaError::check(rc)
}

/// Workspace bytes for up to `max_tokens` rows of a prefill call (the same layout as [`fmoe::prefill_workspace_bytes`]).
pub fn prefill_workspace_bytes(max_tokens: usize, top_k: usize, ne: usize, h: usize, i: usize) -> usize {
    unsafe { ffi::gb_ling_fmoep_workspace_bytes(max_tokens as i32, top_k as i32, ne as i32, h as i32, i as i32) }
}

/// The post-MoE fused epilogue of [`prefill_with_epilogue`]: the combine launch's last pass runs the residual add
/// (`hidden + out`) + the next layer's input norm (`norm_w` -> `norm_out`) with the exact `glue::add_rmsnorm` math,
/// replacing the separate launch. `hidden_out` aliases `hidden` (the add is in place, exactly like
/// `glue::add_rmsnorm`'s `h_out`). `next_suh`/`next_xh`, when both given, additionally fuse the next layer's first
/// projection input rotation into the combine epilogue (the exact `gb_glm_p2_rotate` math on the normed row into
/// `next_xh` fp16 `[t, h]`), so that projection's P2 GEMM skips its rotate launch.
pub struct PrefillEpilogue<'a> {
    pub hidden: &'a DevBuf,
    pub norm_w: &'a DevBuf,
    pub norm_out: &'a mut DevBuf,
    pub eps: f32,
    pub next_suh: Option<&'a DevBuf>,
    pub next_xh: Option<&'a mut DevBuf>,
}

/// Prefill MoE with the scan+scatter and combine+add_rmsnorm fusions (`gb_ling_fmoep_forward_with_epilogue`):
/// bit-identical to [`fmoe::prefill`] + `glue::add_rmsnorm` with the same arguments, two launches fewer per call.
/// `epilogue` (when given) additionally fuses the post-MoE residual add + next input norm into the combine launch;
/// on any other shape the standalone combine launch runs (still with the scan+scatter fusion). Falls back through
/// the caller when the fused shape is unavailable (an error is returned).
pub fn prefill_with_epilogue(
    x: &DevBuf,
    logits: &DevBuf,
    ld: LogitsDtype,
    shared_gate: &DevBuf,
    tb: &ExpertTables<'_>,
    out: &mut DevBuf,
    shape: Shape,
    gr: Option<GroupRoute<'_>>,
    limits: SwigluLimit,
    ws: &mut DevBuf,
    st: &Stream,
    epilogue: Option<PrefillEpilogue<'_>>,
) -> Result<(), CudaError> {
    let Shape { t, ne, top_k, h, i, max_tokens, .. } = shape;
    if t == 0 || t > max_tokens {
        eprintln!("rexl3 level=error launch contract: ling fmoe prefill rows {t} outside 1..={max_tokens}");
        return Err(CudaError(1));
    }
    need(x, t * h * 2, "ling fmoe prefill x")?;
    need(logits, t * ne * ld.size(), "ling fmoe prefill logits")?;
    need(shared_gate, t * 2, "ling fmoe prefill shared gate")?;
    tb.check(ne + 1, "ling fmoe prefill")?;
    need(out, t * h * 2, "ling fmoe prefill out")?;
    let op = out.ptr();
    need(ws, prefill_workspace_bytes(max_tokens, top_k, ne, h, i), "ling fmoe prefill workspace")?;
    let wp = ws.ptr();
    if let Some(e) = &epilogue {
        need(e.hidden, t * h * 2, "ling fmoe prefill epilogue hidden")?;
        need(e.norm_w, h * 4, "ling fmoe prefill epilogue norm_w")?;
        need(e.norm_out, t * h * 2, "ling fmoe prefill epilogue norm_out")?;
        if let Some(s) = &e.next_suh {
            need(s, h * 2, "ling fmoe prefill epilogue next_suh")?;
        }
        if let Some(x) = &e.next_xh {
            need(x, t * h * 2, "ling fmoe prefill epilogue next_xh")?;
        }
    }
    let [gt, gsu, gsv, ut, usu, usv, dt, dsu, dsv] = tb.ptrs();
    let (bias_ptr, n_group, topk_group) = match gr {
        Some(g) => (g.expert_bias.map_or(core::ptr::null(), |b| b.ptr() as *const f32), g.n_group as i32, g.topk_group as i32),
        None => (core::ptr::null(), 0, 0),
    };
    let (hid, nw, no, eps_v) = match &epilogue {
        Some(e) => (e.hidden.ptr(), e.norm_w.ptr() as *const f32, e.norm_out.ptr(), e.eps),
        None => (core::ptr::null_mut(), core::ptr::null(), core::ptr::null_mut(), 0.0),
    };
    let (nsuh, nxh) = match &epilogue {
        Some(e) => {
            (e.next_suh.map_or(core::ptr::null(), |s| s.ptr()), e.next_xh.as_ref().map_or(core::ptr::null_mut(), |x| x.ptr()))
        }
        None => (core::ptr::null(), core::ptr::null_mut()),
    };
    let rc = unsafe {
        ffi::gb_ling_fmoep_forward_with_epilogue(
            x.ptr(),
            logits.ptr(),
            logits_ffi_code(ld),
            shared_gate.ptr(),
            gt,
            gsu,
            gsv,
            ut,
            usu,
            usv,
            dt,
            dsu,
            dsv,
            tb.ktab.ptr() as *const i32,
            op,
            t as i32,
            ne as i32,
            top_k as i32,
            h as i32,
            i as i32,
            max_tokens as i32,
            wp,
            st.raw(),
            bias_ptr,
            n_group,
            topk_group,
            limits.routed,
            limits.shared,
            gr.map_or(0.0, |g| g.scale),
            hid,
            nw,
            no,
            eps_v,
            nsuh,
            nxh,
        )
    };
    CudaError::check(rc)
}
