//! Fused Runtime-K Cooperative MoE decode kernels for EXL3 weights (CoopMK), and the expert pointer tables every MoE
//! kernel reads.
#![allow(clippy::too_many_arguments)]

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
        pub fn gb_moe_topk_softmax(
            gate_logits: *const c_void,
            shared_gate_scalar: *const c_void,
            expert_indices: *mut u32,
            expert_weights: *mut f32,
            num_experts: u32,
            top_k: u32,
            num_tokens: u32,
            stream: *mut c_void,
        ) -> i32;

        pub fn gb_coopmk_workspace_bytes(num_tokens: u32, top_k: u32, intermediate_dim: u32) -> usize;

        pub fn gb_coopmk_decode(
            x: *const c_void,
            sel: *const u32,
            rw: *const f32,
            g_trellis_ptrs: *const u64,
            g_suh_ptrs: *const u64,
            g_svh_ptrs: *const u64,
            u_trellis_ptrs: *const u64,
            u_suh_ptrs: *const u64,
            u_svh_ptrs: *const u64,
            d_trellis_ptrs: *const u64,
            d_suh_ptrs: *const u64,
            d_svh_ptrs: *const u64,
            k_tab: *const i32,
            act_workspace: *mut c_void,
            out: *mut c_void,
            hidden_dim: u32,
            intermediate_dim: u32,
            top_k: u32,
            num_tokens: u32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// The (hidden, expert intermediate) widths [`coopmk_decode`] is instantiated for: Qwen3.8-Flash-Next's and MiniCPM-V-4.7's
/// (the Qwen3.5/3.6-35B-A3B backbone: 256 experts, top-8, one shared expert). Any other pair is refused with an invalid-value error.
pub const COOPMK_SHAPES: [(usize, usize); 3] = [(2560, 640), (2048, 512), (3072, 1024)];

pub fn coopmk_workspace_bytes(num_tokens: usize, top_k: usize, intermediate_dim: usize) -> usize {
    unsafe { ffi::gb_coopmk_workspace_bytes(num_tokens as u32, top_k as u32, intermediate_dim as u32) }
}

/// The `ktab` entry for an EXL3 width of `bits` bits a value. The fused MoE kernels (CoopMK, FrostMoE) read every entry
/// as K2, twice the bits (2..16), so half widths fit the same table; a table that stored plain bits would be decoded at
/// half its width.
pub const fn ktab_k2(bits: usize) -> i32 {
    (2 * bits) as i32
}

/// The device pointer tables of one MoE layer: per expert (routed experts, then the shared expert) the u64 addresses of
/// its gate / up / down trellis, input sign (`su`) and output sign (`sv`) vectors, and `ktab` i32 `[experts][3]` (the
/// K2 of gate, up, down: twice the bits a value, see [`ktab_k2`]).
#[derive(Clone, Copy)]
pub struct ExpertTables<'a> {
    pub g_t: &'a DevBuf,
    pub g_su: &'a DevBuf,
    pub g_sv: &'a DevBuf,
    pub u_t: &'a DevBuf,
    pub u_su: &'a DevBuf,
    pub u_sv: &'a DevBuf,
    pub d_t: &'a DevBuf,
    pub d_su: &'a DevBuf,
    pub d_sv: &'a DevBuf,
    pub ktab: &'a DevBuf,
}

impl ExpertTables<'_> {
    /// Checks that every table covers `experts` entries.
    pub(crate) fn check(&self, experts: usize, what: &str) -> Result<(), CudaError> {
        for (t, name) in [
            (self.g_t, "g_t"),
            (self.g_su, "g_su"),
            (self.g_sv, "g_sv"),
            (self.u_t, "u_t"),
            (self.u_su, "u_su"),
            (self.u_sv, "u_sv"),
            (self.d_t, "d_t"),
            (self.d_su, "d_su"),
            (self.d_sv, "d_sv"),
        ] {
            need(t, experts * 8, &format!("{what} {name}"))?;
        }
        need(self.ktab, experts * 3 * 4, &format!("{what} ktab"))
    }

    pub(crate) fn ptrs(&self) -> [*const u64; 9] {
        [self.g_t, self.g_su, self.g_sv, self.u_t, self.u_su, self.u_sv, self.d_t, self.d_su, self.d_sv]
            .map(|t| t.ptr() as *const u64)
    }
}

/// Router: per token the top-`top_k` experts of `gate_logits` bf16 `[t, num_experts]` with softmax weights, plus (with
/// `shared_gate_scalar` bf16 `[t]`) the shared expert as entry `top_k` weighted by its sigmoid. `expert_indices` u32 and
/// `expert_weights` f32 hold `top_k` (+1 with the shared gate) entries a token.
pub fn moe_topk_softmax(
    gate_logits: &DevBuf,
    shared_gate_scalar: Option<&DevBuf>,
    expert_indices: &mut DevBuf,
    expert_weights: &mut DevBuf,
    num_experts: usize,
    top_k: usize,
    num_tokens: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    let t = num_tokens;
    let stride = top_k + shared_gate_scalar.is_some() as usize;
    need(gate_logits, t * num_experts * 2, "moe_topk_softmax logits")?;
    let sg = if let Some(b) = shared_gate_scalar {
        need(b, t * 2, "moe_topk_softmax shared gate")?;
        b.ptr()
    } else {
        core::ptr::null()
    };
    need(expert_indices, t * stride * 4, "moe_topk_softmax indices")?;
    let ip = expert_indices.ptr() as *mut u32;
    need(expert_weights, t * stride * 4, "moe_topk_softmax weights")?;
    let wp = expert_weights.ptr() as *mut f32;
    {
        let rc = unsafe {
            ffi::gb_moe_topk_softmax(gate_logits.ptr(), sg, ip, wp, num_experts as u32, top_k as u32, t as u32, stream.raw())
        };
        CudaError::check(rc)
    }
}

/// CoopMK decode: `out` bf16 `[t, h]` = sum over each token's `top_k` slots (`sel` u32, `rw` f32 `[t, top_k]`) of the
/// weighted expert MLPs of `x` bf16 `[t, h]`. `act_workspace` holds [`coopmk_workspace_bytes`]. `(hidden_dim, intermediate_dim)` must
/// be one of [`COOPMK_SHAPES`].
pub fn coopmk_decode(
    x: &DevBuf,
    sel: &DevBuf,
    rw: &DevBuf,
    tables: &ExpertTables<'_>,
    act_workspace: &mut DevBuf,
    out: &mut DevBuf,
    hidden_dim: usize,
    intermediate_dim: usize,
    top_k: usize,
    num_tokens: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    let t = num_tokens;
    need(x, t * hidden_dim * 2, "coopmk x")?;
    need(sel, t * top_k * 4, "coopmk sel")?;
    need(rw, t * top_k * 4, "coopmk rw")?;
    // expert ids come from the router: the tables must cover every expert the layer has (at least one per slot kind)
    tables.check(tables.g_t.bytes() / 8, "coopmk")?;
    need(act_workspace, coopmk_workspace_bytes(t, top_k, intermediate_dim), "coopmk workspace")?;
    let ap = act_workspace.ptr();
    need(out, t * hidden_dim * 2, "coopmk out")?;
    let op = out.ptr();
    let [gt, gsu, gsv, ut, usu, usv, dt, dsu, dsv] = tables.ptrs();
    {
        let rc = unsafe {
            ffi::gb_coopmk_decode(
                x.ptr(),
                sel.ptr() as *const u32,
                rw.ptr() as *const f32,
                gt,
                gsu,
                gsv,
                ut,
                usu,
                usv,
                dt,
                dsu,
                dsv,
                tables.ktab.ptr() as *const i32,
                ap,
                op,
                hidden_dim as u32,
                intermediate_dim as u32,
                top_k as u32,
                t as u32,
                stream.raw(),
            )
        };
        CudaError::check(rc)
    }
}

#[cfg(test)]
mod tests {
    use super::ktab_k2;

    #[test]
    fn a_ktab_entry_is_twice_the_bits() {
        for (bits, k2) in [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10), (6, 12), (8, 16)] {
            assert_eq!(ktab_k2(bits), k2, "{bits} bits");
        }
    }
}
