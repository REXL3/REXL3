//! Nemotron 3.5 Super VL prefill: the dense EXL3 linears on tensor cores at any slice width, under the owner's prefill arithmetic
//! contract "P2" (`csrc/nemotron_dense_mma.cu`, `docs/NEMOTRON_PREFILL_DENSE_MMA.md`).
//!
//! Which layers: the Mamba `in_proj` / `out_proj`, the attention `q` / `k` / `v` / `o` and the MoE block's `fc1`, `fc2` and shared
//! expert (every one a 4-bit trellis in the pack). The router stays where it is (exact fp32, it decides the experts).
//!
//! **P2, in short.** (a) A row's output bits depend on that row and the weights only (not on the slice width, the boundary, padding or
//! row order). (b) Every output's K sum is one chain of m16n8k16 MMAs over the k tiles in order, a function of K alone: the tile shape,
//! M and the TP rank never change a bit, so a 128-column-strip cut across ranks is exact. (c) Activations enter the MMA as fp16 (the
//! codebook values are fp16 and must be decoded exactly; bf16 cannot hold them), fp32 accumulate. (d) Deterministic. The bits differ
//! from the decode path's (accepted, documented); decode and verify windows keep the frozen kernels.
//!
//! **The switch (default OFF).** Nothing changes until BOTH hold: the process opted in (`REXL3_NEMO_DENSE_MMA=1`, or
//! [`set_enabled`]) and the calling host thread is inside a [`PrefillScope`] (the model enters one around a prefill slice and never
//! around a decode, verify or wide decode window). [`active`] answers that, at ANY row count: P2(a) forbids a slice's width from
//! choosing the arithmetic, so a 1-row tail slice of a prompt takes the same path as a 4,096-row slice; the scope, not a row floor, is
//! what keeps decode and verify windows on the frozen kernels. The MoE prefill block (`nemotron_moe_prefill`) consults it for its four
//! dense projections; outside this crate the model must consult it in `linear_rows` (Mamba and attention linears) and in its MoE arm
//! (which sends slices of 64 rows or fewer to the decode block today), lane S wires both.

use crate::exl3_linear::{Dtype, Exl3Weights};
use crate::nemotron_moe::Proj;
use crate::rt::{CudaError, DevBuf, Stream};
use core::ffi::c_void;
use gb_exl3::Codebook;
use std::cell::Cell;
use std::sync::atomic::{AtomicU8, Ordering};

const INVALID: CudaError = CudaError(1);

#[cfg(not(gb_no_cuda))]
mod ffi {
    use core::ffi::c_void;
    extern "C" {
        pub fn gb_ndm_supported(k: i32, n: i32, bits: i32, codebook: i32) -> i32;
        pub fn gb_ndm_rotate(
            x: *const c_void,
            x_dtype: i32,
            ldx: i64,
            suh: *const c_void,
            suh_f32: i32,
            a: *mut c_void,
            m: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_ndm_gemm(
            a: *const c_void,
            trellis: *const c_void,
            svh: *const c_void,
            bias: *const c_void,
            scale_f32: i32,
            y: *mut c_void,
            y_dtype: i32,
            ldy: i64,
            add: *const c_void,
            ld_add: i64,
            next_suh: *const c_void,
            relu2: i32,
            m: i32,
            k: i32,
            n: i32,
            bits: i32,
            codebook: i32,
            tile_force: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_ndm_pick_tile(m: i32, n: i32) -> i32;
        pub fn gb_ndm_norm_rot(
            x: *const c_void,
            w: *const c_void,
            eps: f32,
            suh: *const c_void,
            suh_f32: i32,
            a: *mut c_void,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_ndm_gated_rot(
            y: *const c_void,
            z: *const c_void,
            z_bf16: i32,
            w: *const f32,
            eps: f32,
            suh: *const c_void,
            suh_f32: i32,
            a: *mut c_void,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_ndm_gemm_split(
            a: *const c_void,
            trellis: *const c_void,
            svh: *const c_void,
            bias: *const c_void,
            scale_f32: i32,
            z: *mut c_void,
            xbc: *mut c_void,
            dt: *mut c_void,
            split_z: i32,
            split_x: i32,
            zx_bf16: i32,
            m: i32,
            k: i32,
            n: i32,
            bits: i32,
            codebook: i32,
            tile_force: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

// 0: not read yet, 1: off, 2: on.
static ENABLED: AtomicU8 = AtomicU8::new(0);

thread_local! {
    static IN_PREFILL: Cell<u32> = const { Cell::new(0) };
}

/// Whether this process opted in: `REXL3_NEMO_DENSE_MMA=1` (read once), or the last [`set_enabled`].
pub fn enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        0 => {
            let on = std::env::var("REXL3_NEMO_DENSE_MMA").is_ok_and(|v| v == "1");
            ENABLED.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
        v => v == 2,
    }
}

/// Turns the opt-in on or off for the process (tests, tools).
pub fn set_enabled(on: bool) {
    ENABLED.store(if on { 2 } else { 1 }, Ordering::Relaxed);
}

/// A prefill slice on this host thread: while one is alive (they nest), [`active`] may answer yes. Drop ends it.
pub struct PrefillScope(());

impl PrefillScope {
    pub fn enter() -> Self {
        IN_PREFILL.with(|c| c.set(c.get() + 1));
        PrefillScope(())
    }
}

impl Drop for PrefillScope {
    fn drop(&mut self) {
        IN_PREFILL.with(|c| c.set(c.get().saturating_sub(1)));
    }
}

/// True when a call of `rows` rows on this thread takes the tensor-core prefill path: opted in and inside a [`PrefillScope`]. Any
/// `rows >= 1`: the width of a prefill slice never chooses the arithmetic (P2(a)).
pub fn active(rows: usize) -> bool {
    rows >= 1 && IN_PREFILL.with(|c| c.get() > 0) && enabled()
}

fn codebook_code(cb: Codebook) -> i32 {
    match cb {
        Codebook::Inst3 => 0,
        Codebook::Mcg => 1,
        Codebook::Mul1 => 2,
    }
}

fn align256(v: usize) -> usize {
    (v + 255) & !255
}

/// Bytes of the rotated fp16 input of `m` rows of a layer with input width `k` (the workspace [`linear`] needs).
pub fn workspace_bytes(m: usize, k: usize) -> usize {
    align256(m * k * 2)
}

/// The tile the kernel picks for `m` rows of an `n`-wide layer, `BM * 10 + SN` (64 or 128 rows by 1 or 2 strips of 128 columns).
pub fn pick_tile(m: usize, n: usize) -> i32 {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        ffi::gb_ndm_pick_tile(m as i32, n as i32)
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (m, n);
        0
    }
}

/// One EXL3 layer as the kernel reads it: device pointers and the format. `scale_f32`: `suh`, `svh` and the bias are f32 (the MoE
/// block's tables) rather than fp16 ([`Exl3Weights`]).
#[derive(Clone, Copy, Debug)]
pub struct Layer {
    pub trellis: *const c_void,
    pub suh: *const c_void,
    pub svh: *const c_void,
    pub bias: *const c_void,
    pub scale_f32: bool,
    pub k: usize,
    pub n: usize,
    pub bits: usize,
    pub codebook: Codebook,
}

// SAFETY: device pointers only, never dereferenced on the host; the owner of the buffers outlives every use (see the call sites).
unsafe impl Send for Layer {}
unsafe impl Sync for Layer {}

impl Layer {
    /// A decode-path layer (fp16 scales).
    pub fn of_weights(w: &Exl3Weights) -> Self {
        Layer {
            trellis: w.trellis.ptr() as *const c_void,
            suh: w.suh.ptr() as *const c_void,
            svh: w.svh.ptr() as *const c_void,
            bias: w.bias.as_ref().map_or(core::ptr::null(), |b| b.ptr() as *const c_void),
            scale_f32: false,
            k: w.k,
            n: w.n,
            bits: w.bits,
            codebook: w.codebook,
        }
    }

    /// A MoE block projection (f32 scales); `None` for a bf16 or absent one.
    pub fn of_proj(p: &Proj) -> Option<Self> {
        match p {
            Proj::Exl3 { k, n, bits, codebook, trellis, suh, svh, bias } if bits.twice() % 2 == 0 => Some(Layer {
                trellis: trellis.ptr() as *const c_void,
                suh: suh.ptr() as *const c_void,
                svh: svh.ptr() as *const c_void,
                bias: bias.as_ref().map_or(core::ptr::null(), |b| b.ptr() as *const c_void),
                scale_f32: true,
                k: *k,
                n: *n,
                bits: bits.twice() as usize / 2,
                codebook: *codebook,
            }),
            _ => None,
        }
    }

    /// Whether the kernel takes this layer (K and N multiples of 128; 4 bits, any codebook, or 2, 3 and 5 to 8 bits with Mul1).
    pub fn supported(&self) -> bool {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            ffi::gb_ndm_supported(self.k as i32, self.n as i32, self.bits as i32, codebook_code(self.codebook)) != 0
        }
        #[cfg(gb_no_cuda)]
        false
    }
}

/// `a [m, k]` fp16 = `(x * suh) H / sqrt 128` of `m` rows of `x` (`[m, ldx]` of `x_dtype`). Raw pointers: `a` holds `m * k * 2` bytes.
///
/// # Safety
/// `x`, `a` and the layer's pointers are device memory of the sizes stated; the call is asynchronous on `st`.
pub unsafe fn rotate_raw(
    l: &Layer,
    x: *const c_void,
    x_dtype: Dtype,
    ldx: usize,
    a: *mut c_void,
    m: usize,
    st: *mut c_void,
) -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    {
        CudaError::check(ffi::gb_ndm_rotate(
            x,
            x_dtype as i32,
            ldx as i64,
            l.suh,
            l.scale_f32 as i32,
            a,
            m as i32,
            l.k as i32,
            st,
        ))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (l, x, x_dtype, ldx, a, m, st);
        Err(CudaError(-1))
    }
}

/// What the GEMM's epilogue does after `* svh + bias`.
#[derive(Clone, Copy, Debug)]
pub struct Epilogue {
    /// `relu(v)^2`.
    pub relu2: bool,
    /// f32 `[m, ld_add]` added last (null: none).
    pub add: *const c_void,
    pub ld_add: usize,
    /// The next projection's `suh` (same scale dtype as this layer's): the output becomes that projection's rotated fp16 input
    /// `[m, n]` instead of `y` of `y_dtype` (null: plain output).
    pub next_suh: *const c_void,
    /// The tile shape (0: the kernel's pick; else `BM * 10 + SN`). Never changes a bit.
    pub tile: i32,
}

impl Default for Epilogue {
    fn default() -> Self {
        Epilogue { relu2: false, add: core::ptr::null(), ld_add: 0, next_suh: core::ptr::null(), tile: 0 }
    }
}

/// `y` = the layer applied to the rotated rows `a` (`[m, k]` fp16). `y` is `[m, ldy]` of `y_dtype` (or fp16 `[m, n]` with `next_suh`).
///
/// # Safety
/// Device pointers of the stated sizes; asynchronous on `st`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_raw(
    l: &Layer,
    a: *const c_void,
    y: *mut c_void,
    y_dtype: Dtype,
    ldy: usize,
    m: usize,
    e: &Epilogue,
    st: *mut c_void,
) -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    {
        CudaError::check(ffi::gb_ndm_gemm(
            a,
            l.trellis,
            l.svh,
            l.bias,
            l.scale_f32 as i32,
            y,
            y_dtype as i32,
            ldy as i64,
            e.add,
            e.ld_add as i64,
            e.next_suh,
            e.relu2 as i32,
            m as i32,
            l.k as i32,
            l.n as i32,
            l.bits as i32,
            codebook_code(l.codebook),
            e.tile,
            st,
        ))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (l, a, y, y_dtype, ldy, m, e, st);
        Err(CudaError(-1))
    }
}

/// `y [rows, n] = w(x [rows, k])` on tensor cores under P2 (any `rows >= 1`): the drop-in for `linear_rows` on a prefill slice. `ws` needs
/// [`workspace_bytes`]`(rows, k)`; its contents on entry do not matter. Asynchronous on `st`.
#[allow(clippy::too_many_arguments)]
pub fn linear(
    w: &Exl3Weights,
    x: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    x_dtype: Dtype,
    y_dtype: Dtype,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    linear_tiled(w, x, y, rows, x_dtype, y_dtype, ws, 0, st)
}

/// [`linear`] (f32 out) with the output cut as the Mamba `in_proj` split cuts it (lane S5): columns `[0, split_z)` into `z`
/// (`[rows, split_z]` f32 rounded to the bf16 grid), `[split_z, split_z + split_x)` into `xbc` (`[rows, split_x]` f32), the rest into
/// `dt` (`[rows, n - split_z - split_x]` f32): byte for byte what [`linear`] then `nemotron_glue::split_inproj` write, without the f32
/// `[rows, n]` intermediate and the split's pass over it. Both cuts on 128-column strips. `zx_bf16`: z and xbc stored as bf16 instead
/// (round to nearest even: the values `nemotron_ssm::bf16_in` reads, which are what every consumer of the f32 rows reads them as).
/// `prerotated`: `ws` already holds the rotated input ([`norm_rot`]) and `x` is not read.
#[allow(clippy::too_many_arguments)]
pub fn linear_split(
    w: &Exl3Weights,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    z: &mut DevBuf,
    xbc: &mut DevBuf,
    dt: &mut DevBuf,
    split_z: usize,
    split_x: usize,
    zx_bf16: bool,
    prerotated: bool,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    let l = Layer::of_weights(w);
    let zx = if zx_bf16 { 2 } else { 4 };
    if rows == 0
        || !l.supported()
        || split_z % 128 != 0
        || split_x % 128 != 0
        || split_z + split_x >= l.n
        || x.bytes() < rows * l.k * x_dtype.size()
        || z.bytes() < rows * split_z * zx
        || xbc.bytes() < rows * split_x * zx
        || dt.bytes() < rows * (l.n - split_z - split_x) * 4
        || ws.bytes() < workspace_bytes(rows, l.k)
    {
        return Err(INVALID);
    }
    #[cfg(not(gb_no_cuda))]
    // SAFETY: the sizes were checked above; the buffers are device allocations
    unsafe {
        if !prerotated {
            rotate_raw(&l, x.ptr() as *const c_void, x_dtype, l.k, ws.ptr(), rows, st.raw())?;
        }
        CudaError::check(ffi::gb_ndm_gemm_split(
            ws.ptr() as *const c_void,
            l.trellis,
            l.svh,
            l.bias,
            l.scale_f32 as i32,
            z.ptr(),
            xbc.ptr(),
            dt.ptr(),
            split_z as i32,
            split_x as i32,
            zx_bf16 as i32,
            rows as i32,
            l.k as i32,
            l.n as i32,
            l.bits as i32,
            codebook_code(l.codebook),
            0,
            st.raw(),
        ))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (st, prerotated);
        Err(CudaError(-1))
    }
}

/// `resid [rows, n] += w(x [rows, k])` (f32) on tensor cores under P2, in one pass (lane S5): the GEMM epilogue adds the residual row to
/// the layer's output and writes the sum in place, so the f32 output and the separate residual add (`out + resid`, the same single
/// rounding: IEEE addition commutes) are not made. Asynchronous on `st`.
#[allow(clippy::too_many_arguments)]
pub fn linear_residual(
    w: &Exl3Weights,
    x: &DevBuf,
    x_dtype: Dtype,
    resid: &mut DevBuf,
    rows: usize,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    let l = Layer::of_weights(w);
    if rows == 0
        || !l.supported()
        || x.bytes() < rows * l.k * x_dtype.size()
        || resid.bytes() < rows * l.n * 4
        || ws.bytes() < workspace_bytes(rows, l.k)
    {
        return Err(INVALID);
    }
    let r = resid.ptr();
    // SAFETY: the sizes were checked above; the buffers are device allocations. The epilogue reads add[row, col] and writes
    // y[row, col] of the same element in the same thread, so the in-place call is race free.
    unsafe {
        rotate_raw(&l, x.ptr() as *const c_void, x_dtype, l.k, ws.ptr(), rows, st.raw())?;
        gemm_raw(
            &l,
            ws.ptr() as *const c_void,
            r,
            Dtype::F32,
            l.n,
            rows,
            &Epilogue { add: r as *const c_void, ld_add: l.n, ..Default::default() },
            st.raw(),
        )
    }
}

/// The Mamba gated RMSNorm (`nemotron_ssm::gated_norm`'s bits: y bf16 `[rows, 8192]`, z f32 or bf16 `[rows, 8192]`, `norm_w` f32
/// `[8192]`, eps) written straight as `w`'s rotated fp16 input into `ws` (what [`linear`] would make of the bf16 `gated` rows), lane S5;
/// follow with [`linear_prerotated`]. Asynchronous on `st`.
#[allow(clippy::too_many_arguments)]
pub fn gated_rot(
    y: &DevBuf,
    z: &DevBuf,
    z_bf16: bool,
    norm_w: &DevBuf,
    eps: f32,
    w: &Exl3Weights,
    rows: usize,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    let l = Layer::of_weights(w);
    if rows == 0
        || l.k != 8192
        || y.bytes() < rows * 8192 * 2
        || z.bytes() < rows * 8192 * if z_bf16 { 2 } else { 4 }
        || norm_w.bytes() < 8192 * 4
        || ws.bytes() < workspace_bytes(rows, l.k)
    {
        return Err(INVALID);
    }
    #[cfg(not(gb_no_cuda))]
    // SAFETY: sizes checked above; device allocations
    unsafe {
        CudaError::check(ffi::gb_ndm_gated_rot(
            y.ptr() as *const c_void,
            z.ptr() as *const c_void,
            z_bf16 as i32,
            norm_w.ptr() as *const f32,
            eps,
            l.suh,
            l.scale_f32 as i32,
            ws.ptr(),
            rows as i32,
            st.raw(),
        ))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (st, eps);
        Err(CudaError(-1))
    }
}

/// The layer RMSNorm (`nemotron_glue::rmsnorm_bf16w_f32`'s bits; x f32 `[rows, 4096]`, `norm_w` bf16 `[4096]`) written straight as
/// `w`'s rotated fp16 input into `ws` (lane S5); the f32 normed rows are not written. Follow with a prerotated GEMM.
pub fn norm_rot(
    x: &DevBuf,
    norm_w: &DevBuf,
    eps: f32,
    w: &Exl3Weights,
    rows: usize,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    let l = Layer::of_weights(w);
    if rows == 0
        || l.k != 4096
        || x.bytes() < rows * 4096 * 4
        || norm_w.bytes() < 4096 * 2
        || ws.bytes() < workspace_bytes(rows, l.k)
    {
        return Err(INVALID);
    }
    #[cfg(not(gb_no_cuda))]
    // SAFETY: sizes checked above; device allocations
    unsafe {
        CudaError::check(ffi::gb_ndm_norm_rot(
            x.ptr() as *const c_void,
            norm_w.ptr() as *const c_void,
            eps,
            l.suh,
            l.scale_f32 as i32,
            ws.ptr(),
            rows as i32,
            st.raw(),
        ))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (st, eps);
        Err(CudaError(-1))
    }
}

/// [`linear`]'s GEMM on input already rotated into `ws` ([`gated_rot`]): `y [rows, n]` f32, or with `resid` the output added into
/// `y` in place as [`linear_residual`] does. Asynchronous on `st`.
pub fn linear_prerotated(
    w: &Exl3Weights,
    ws: &DevBuf,
    y: &mut DevBuf,
    resid: bool,
    rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    let l = Layer::of_weights(w);
    if rows == 0 || !l.supported() || y.bytes() < rows * l.n * 4 || ws.bytes() < workspace_bytes(rows, l.k) {
        return Err(INVALID);
    }
    let yp = y.ptr();
    let e = if resid { Epilogue { add: yp as *const c_void, ld_add: l.n, ..Default::default() } } else { Epilogue::default() };
    // SAFETY: sizes checked above; with resid the epilogue reads and writes each element in one thread
    unsafe { gemm_raw(&l, ws.ptr() as *const c_void, yp, Dtype::F32, l.n, rows, &e, st.raw()) }
}

/// [`linear`] with the tile forced (`BM * 10 + SN`, or 0): the gates prove a tile never changes a bit.
#[allow(clippy::too_many_arguments)]
pub fn linear_tiled(
    w: &Exl3Weights,
    x: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    x_dtype: Dtype,
    y_dtype: Dtype,
    ws: &mut DevBuf,
    tile: i32,
    st: &Stream,
) -> Result<(), CudaError> {
    let l = Layer::of_weights(w);
    if rows == 0
        || !l.supported()
        || x.bytes() < rows * l.k * x_dtype.size()
        || y.bytes() < rows * l.n * y_dtype.size()
        || ws.bytes() < workspace_bytes(rows, l.k)
    {
        return Err(INVALID);
    }
    // SAFETY: the sizes were checked above; the buffers are device allocations
    unsafe {
        rotate_raw(&l, x.ptr() as *const c_void, x_dtype, l.k, ws.ptr(), rows, st.raw())?;
        gemm_raw(&l, ws.ptr() as *const c_void, y.ptr(), y_dtype, l.n, rows, &Epilogue { tile, ..Default::default() }, st.raw())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_needs_the_opt_in_and_the_scope_at_any_width() {
        set_enabled(true);
        assert!(!active(256), "no scope");
        {
            let _s = PrefillScope::enter();
            assert!(active(256));
            assert!(active(64) && active(1), "a short prefill slice takes it too (P2(a))");
            assert!(!active(0));
            {
                let _t = PrefillScope::enter();
                assert!(active(512));
            }
            assert!(active(512), "nested scopes");
            set_enabled(false);
            assert!(!active(256), "opted out");
            set_enabled(true);
        }
        assert!(!active(256), "scope over");
        set_enabled(false);
    }

    #[test]
    fn the_workspace_is_the_fp16_rows() {
        assert_eq!(workspace_bytes(256, 4096), 256 * 4096 * 2);
        assert_eq!(workspace_bytes(1, 128), 256);
    }
}
