//! FrostMoE: grouped multi-token MoE decode for packed EXL3 experts (see `csrc/fmoe.cu`).
#![allow(clippy::too_many_arguments)]

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
        pub fn gb_fmoe_workspace_bytes(max_tokens: i32, top_k: i32, h: i32, i: i32, kc: i32) -> usize;
        pub fn gb_fmoe_forward(
            x: *const c_void, logits: *const c_void, shared_gate: *const c_void,
            g_t: *const u64, g_su: *const u64, g_sv: *const u64,
            u_t: *const u64, u_su: *const u64, u_sv: *const u64,
            d_t: *const u64, d_su: *const u64, d_sv: *const u64,
            ktab: *const i32, out: *mut c_void,
            t: i32, ne: i32, top_k: i32, h: i32, i: i32, kc: i32, max_tokens: i32,
            ws: *mut c_void, stream: *mut c_void,
            expert_bias: *const f32, n_group: i32, topk_group: i32,
            routed_limit: f32, shared_limit: f32, routed_scaling: f32,
        ) -> i32;
        pub fn gb_fmoe_unpack(trellis: *const c_void, w: *mut c_void, k: i32, n: i32, k2: i32, stream: *mut c_void) -> i32;
        pub fn gb_fmoe_group_route(
            logits: *const c_void, shared_gate: *const c_void, expert_bias: *const f32, scale: f32,
            sel: *mut i32, wts: *mut f32, ws: *mut c_void,
            t: i32, ne: i32, k: i32, n_group: i32, topk_group: i32, stream: *mut c_void,
        ) -> i32;
        pub fn gb_fmoep_workspace_bytes(max_tokens: i32, top_k: i32, ne: i32, h: i32, i: i32) -> usize;
        pub fn gb_fmoep_forward(
            x: *const c_void, logits: *const c_void, shared_gate: *const c_void,
            g_t: *const u64, g_su: *const u64, g_sv: *const u64,
            u_t: *const u64, u_su: *const u64, u_sv: *const u64,
            d_t: *const u64, d_su: *const u64, d_sv: *const u64,
            ktab: *const i32, out: *mut c_void,
            t: i32, ne: i32, top_k: i32, h: i32, i: i32, max_tokens: i32,
            ws: *mut c_void, stream: *mut c_void,
            expert_bias: *const f32, n_group: i32, topk_group: i32,
            routed_limit: f32, shared_limit: f32, routed_scaling: f32,
        ) -> i32;
    }
}

/// K chunks of the gate/up pass (H / 128 must be a multiple): 10 is Qwen3.8-Flash-Next's (H = 2560, 20 blocks of 128). It is NOT
/// valid at every width: MiniCPM-V-4.7's H = 2048 has 16 blocks, so 10 is refused there; use [`kc_valid`] to check a count and
/// pick one of the divisors of `H / 128` (the sweep that chose MiniCPM's is `bench_decode_moe_kc_sweep` in tests/minicpm_moe.rs).
pub const DEFAULT_KC: usize = 10;

/// K chunks for MiniCPM-V-4.7's H = 2048, from `bench_decode_moe_kc_sweep` (tests/minicpm_moe.rs: 256 experts, top-8 + shared, widths
/// 3..8 mixed, weights streamed from DRAM, four runs on the GB10, microseconds a ROW of one layer):
///
/// | rows | 1 | 4 | 8 | 16 | 32 | 64 |
/// |---|---|---|---|---|---|---|
/// | KC 4 | 104 | 87 | 78 | 67 | 54.5 | 38.7 |
/// | KC 8 | 104 | 88 | 78 | 68.4 | 55.8 | 39.9 |
/// | KC 16 | 107 | 88 | 79 | 69.3 | 56.9 | 41.1 |
/// | CoopMK | 210 | 132 | 112 | 102 | 92 | 86 |
///
/// The counts tie at 1 to 8 rows (run-to-run noise is +-3 us) and fewer, longer chunks win as rows grow (KC 4 is 2% ahead of 8 at 16
/// rows and 3% at 64; 8 beats 16 everywhere), so 4: the fewest partials to write and add per group. FrostMoE is 1.4 to 2.2 times
/// faster than CoopMK at every row count.
pub const MINICPM_KC: usize = 4;
pub const QWEN35_122B_KC: usize = 6;

/// The 128-wide K blocks of a chunk the gate/up kernel can stage: its 4 warps stage 8 tokens each of a chunk (+ 8 halves of padding)
/// in the default 48 KiB of dynamic shared memory.
pub const MAX_BLOCKS_PER_CHUNK: usize = 5;

/// Whether `kc` K chunks is a valid count at hidden width `h`: a divisor of `h / 128` (`h` a multiple of 128) whose chunks are at
/// most [`MAX_BLOCKS_PER_CHUNK`] blocks wide, so at H = 2048 the counts 4, 8 and 16 (1 and 2 stage more than 48 KiB), at H = 2560
/// 4, 5, 10 and 20.
pub const fn kc_valid(h: usize, kc: usize) -> bool {
    h % 128 == 0 && kc >= 1 && (h / 128) % kc == 0 && h / 128 / kc <= MAX_BLOCKS_PER_CHUNK
}

/// Workspace bytes for up to `max_tokens` rows a call (0 for an unsupported shape).
pub fn workspace_bytes(max_tokens: usize, top_k: usize, h: usize, i: usize, kc: usize) -> usize {
    unsafe { ffi::gb_fmoe_workspace_bytes(max_tokens as i32, top_k as i32, h as i32, i as i32, kc as i32) }
}

/// Zeroed workspace for up to `max_tokens` rows a call.
pub fn workspace(max_tokens: usize, top_k: usize, h: usize, i: usize, kc: usize) -> Result<DevBuf, CudaError> {
    let bytes = workspace_bytes(max_tokens, top_k, h, i, kc);
    if bytes == 0 {
        return Err(CudaError(1));
    }
    DevBuf::zeroed(bytes)
}

/// The shape of a FrostMoE layer call.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// Rows this call.
    pub t: usize,
    /// Routed experts (the shared expert is entry `ne` of the tables).
    pub ne: usize,
    pub top_k: usize,
    pub h: usize,
    pub i: usize,
    pub kc: usize,
    /// Rows the workspace was sized for.
    pub max_tokens: usize,
}

/// The group-limited noaux_tc router of `BailingMoeV3Gate`: sigmoid(logits) + expert_bias, the top-2 scores of every
/// group of `n_group` consecutive experts summed, `topk_group` groups selected, top-K over the masked scores; weights
/// are the normalized unbiased sigmoid scores of the selected experts times `scale`. `None`: the legacy top-k softmax
/// (no bias, no scale).
#[derive(Clone, Copy, Debug)]
pub struct GroupRoute<'a> {
    pub expert_bias: Option<&'a DevBuf>,
    pub n_group: usize,
    pub topk_group: usize,
    pub scale: f32,
}

/// SwiGLU clamps of a layer (`BailingMoeV3`'s `expert_swiglu_limit_list` / `share_expert_swiglu_limit_list`): gate
/// clamped to at most the limit and up to +-limit before `silu(gate) * up`; `<= 0.0` disables the clamp.
#[derive(Clone, Copy, Debug, Default)]
pub struct SwigluLimit {
    pub routed: f32,
    pub shared: f32,
}

/// One MoE layer for `shape.t` rows: route (top-k softmax of `logits` bf16 `[t, ne]`, shared expert weighted by
/// sigmoid(`shared_gate` bf16 `[t]`)), every expert run once for all rows that picked it, `out` bf16 `[t, h]`.
/// `gr` (`None`: legacy softmax) and `limits` select the group-limited noaux_tc router and the SwiGLU clamps.
pub fn forward(
    x: &DevBuf,
    logits: &DevBuf,
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
        eprintln!("rexl3 level=error launch contract: fmoe rows {t} outside 1..={max_tokens}");
        return Err(CudaError(1));
    }
    need(x, t * h * 2, "fmoe x")?;
    need(logits, t * ne * 2, "fmoe logits")?;
    need(shared_gate, t * 2, "fmoe shared gate")?;
    tb.check(ne + 1, "fmoe")?;
    need(out, t * h * 2, "fmoe out")?;
    let op = out.ptr();
    need(ws, workspace_bytes(max_tokens, top_k, h, i, kc), "fmoe workspace")?;
    let wp = ws.ptr();
    let [gt, gsu, gsv, ut, usu, usv, dt, dsu, dsv] = tb.ptrs();
    let (bias_ptr, n_group, topk_group) = match gr {
        Some(g) => (g.expert_bias.map_or(core::ptr::null(), |b| b.ptr() as *const f32), g.n_group as i32, g.topk_group as i32),
        None => (core::ptr::null(), 0, 0),
    };
    let rc = unsafe {
        ffi::gb_fmoe_forward(
            x.ptr(),
            logits.ptr(),
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

/// W `[k][n]` fp16 through FrostMoE's own tile decoder (test oracle check against `gb_exl3_reconstruct`).
/// `k2` is twice the bits (2..16, i.e. 1.0, 1.5, 2.5, 3.5 bits).
pub fn unpack(trellis: &DevBuf, w: &mut DevBuf, k: usize, n: usize, k2: usize, st: &Stream) -> Result<(), CudaError> {
    need(trellis, k * n * k2 / 16, "fmoe unpack trellis")?;
    need(w, k * n * 2, "fmoe unpack w")?;
    let wp = w.ptr();
    CudaError::check(unsafe { ffi::gb_fmoe_unpack(trellis.ptr(), wp, k as i32, n as i32, k2 as i32, st.raw()) })
}

/// The group-limited noaux_tc router alone (fork/debug): sel `[t][top_k + 1]` i32 and wts `[t][top_k + 1]` f32 with
/// the shared expert as entry `top_k`, exactly as [`forward`] routes them (see [`GroupRoute`]).
pub fn group_route(
    logits: &DevBuf,
    shared_gate: &DevBuf,
    expert_bias: &DevBuf,
    scale: f32,
    t: usize,
    ne: usize,
    top_k: usize,
    n_group: usize,
    topk_group: usize,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(Vec<u32>, Vec<f32>), CudaError> {
    if t == 0 || t > 64 || ne == 0 || ne > 512 || top_k == 0 || top_k > 16 || n_group < 2 || topk_group == 0 {
        return Err(CudaError(1));
    }
    need(logits, t * ne * 2, "fmoe group route logits")?;
    need(shared_gate, t * 2, "fmoe group route shared gate")?;
    need(expert_bias, ne * 4, "fmoe group route expert bias")?;
    let sel_bytes = t * (top_k + 1) * 4;
    need(ws, sel_bytes * 2 + (ne + 1) * 4, "fmoe group route workspace")?;
    let sel = crate::rt::DevBuf::zeroed(sel_bytes)?;
    let wts = crate::rt::DevBuf::zeroed(sel_bytes)?;
    let rc = unsafe {
        ffi::gb_fmoe_group_route(
            logits.ptr(),
            shared_gate.ptr(),
            expert_bias.ptr() as *const f32,
            scale,
            sel.ptr() as *mut i32,
            wts.ptr() as *mut f32,
            ws.ptr(),
            t as i32,
            ne as i32,
            top_k as i32,
            n_group as i32,
            topk_group as i32,
            st.raw(),
        )
    };
    CudaError::check(rc)?;
    st.sync()?;
    Ok((sel.to_vec()?, wts.to_vec()?))
}

/// Prefill workspace bytes for up to `max_tokens` rows a call (0 for an unsupported shape).
pub fn prefill_workspace_bytes(max_tokens: usize, top_k: usize, ne: usize, h: usize, i: usize) -> usize {
    unsafe { ffi::gb_fmoep_workspace_bytes(max_tokens as i32, top_k as i32, ne as i32, h as i32, i as i32) }
}

/// Zeroed prefill workspace for up to `max_tokens` rows a call.
pub fn prefill_workspace(max_tokens: usize, top_k: usize, ne: usize, h: usize, i: usize) -> Result<DevBuf, CudaError> {
    let bytes = prefill_workspace_bytes(max_tokens, top_k, ne, h, i);
    if bytes == 0 {
        return Err(CudaError(1));
    }
    DevBuf::zeroed(bytes)
}

/// The MoE layer for a prefill chunk of `shape.t` rows (any number up to `shape.max_tokens`): tokens sorted by expert,
/// every expert weight tile decoded once per 64 of its tokens, per-token combine in top-k order. `shape.kc` is unused.
/// `gr` (`None`: legacy softmax) and `limits` select the group-limited noaux_tc router and the SwiGLU clamps.
///
/// A token's output bits are a function of its own row, its router logits and shared gate and the weights alone: its K order is fixed
/// by the layer's shape, its epilogues are per token and the tile (and the company) it is computed in only decides which block does
/// it. So the bits do not depend on `t`, on the row's position, on the other rows or on how a prompt is cut into calls
/// (`tests/minicpm_moe_prefill.rs`, `testdata/gpu/golden_minicpm_moe_prefill.txt`). They are NOT [`forward`]'s bits (that kernel sums
/// `kc` K chunks of partials) and not CoopMK's. The workspace's contents on entry never matter: one workspace serves every layer.
pub fn prefill(
    x: &DevBuf,
    logits: &DevBuf,
    shared_gate: &DevBuf,
    tb: &ExpertTables<'_>,
    out: &mut DevBuf,
    shape: Shape,
    gr: Option<GroupRoute<'_>>,
    limits: SwigluLimit,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    let Shape { t, ne, top_k, h, i, max_tokens, .. } = shape;
    if t == 0 || t > max_tokens {
        eprintln!("rexl3 level=error launch contract: fmoe prefill rows {t} outside 1..={max_tokens}");
        return Err(CudaError(1));
    }
    need(x, t * h * 2, "fmoe prefill x")?;
    need(logits, t * ne * 2, "fmoe prefill logits")?;
    need(shared_gate, t * 2, "fmoe prefill shared gate")?;
    tb.check(ne + 1, "fmoe prefill")?;
    need(out, t * h * 2, "fmoe prefill out")?;
    let op = out.ptr();
    need(ws, prefill_workspace_bytes(max_tokens, top_k, ne, h, i), "fmoe prefill workspace")?;
    let wp = ws.ptr();
    let [gt, gsu, gsv, ut, usu, usv, dt, dsu, dsv] = tb.ptrs();
    let (bias_ptr, n_group, topk_group) = match gr {
        Some(g) => (g.expert_bias.map_or(core::ptr::null(), |b| b.ptr() as *const f32), g.n_group as i32, g.topk_group as i32),
        None => (core::ptr::null(), 0, 0),
    };
    let rc = unsafe {
        ffi::gb_fmoep_forward(
            x.ptr(),
            logits.ptr(),
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
        )
    };
    CudaError::check(rc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_chunk_counts_must_divide_the_128_wide_blocks_of_the_hidden_width() {
        // Qwen3.8-Flash-Next: H = 2560 = 20 blocks; the default 10 is one of its divisors
        assert!(kc_valid(2560, DEFAULT_KC));
        assert!(kc_valid(2560, 20) && kc_valid(2560, 4) && kc_valid(2560, 5) && !kc_valid(2560, 16) && !kc_valid(2560, 8));
        assert!(
            !kc_valid(2560, 1) && !kc_valid(2560, 2),
            "10 and 20 blocks of 128 do not fit 48 KiB of staging in one or two chunks"
        );
        // MiniCPM-V-4.7: H = 2048 = 16 blocks; Flash-Next's default is not one of its divisors, and 1 and 2 chunks stage too much
        assert!(kc_valid(2048, MINICPM_KC));
        assert!(!kc_valid(2048, DEFAULT_KC));
        for kc in [4, 8, 16] {
            assert!(kc_valid(2048, kc), "{kc}");
        }
        for kc in [0, 1, 2, 3, 5, 6, 10, 32] {
            assert!(!kc_valid(2048, kc), "{kc}");
        }
        assert!(!kc_valid(2000, 1), "a width that is no multiple of 128");
    }
}
