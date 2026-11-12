//! Ling-owned fused KDA decode projections: `csrc/ling_kda_proj_fused.cu` runs two or three EXL3 layers that take the
//! SAME input `x` (the KDA decode triple `qkv`, `f`, `g` on the layer's input norm; the MLA `q_a` + `kv_a` pair; the
//! dense FFN `gate` + `up` pair) as ONE launch with one paired `rot_in` (one rotation buffer per projection's `suh`),
//! instead of one `gb_exl3_linear_k2` launch per projection each rotating the identical input. Every output element
//! computes the same expressions in the same order as the separate launches (same xh bytes per projection, same k order,
//! same warp-order sum, same finish; a split-K pair finishes from the split partials in split order), so the outputs are
//! bit-identical; only the launch count, the duplicate rotation and the partial wave tails change. Used by
//! `bailing_moe_v3` for decode/verify windows. The pair must share k, k2, codebook and the decode plan (a narrow MLA
//! pair splits K; when the two projections' plans differ the C entry falls back to the two separate launches).

use crate::exl3_linear::Dtype;
use crate::rt::{CudaError, DevBuf, Stream};

mod ffi {
    crate::launchers! {
        pub fn gb_ling_kda_proj_fused_workspace_bytes(m: i32, k: i32) -> usize;
        pub fn gb_ling_kda_proj_fused_split_workspace_bytes(m: i32, k: i32, n0: i32, n1: i32) -> usize;
        pub fn gb_ling_kda_proj_fused(
            x: *const core::ffi::c_void,
            x_dtype: i32,
            trellis0: *const core::ffi::c_void,
            suh0: *const core::ffi::c_void,
            svh0: *const core::ffi::c_void,
            y0: *mut core::ffi::c_void,
            y_dtype0: i32,
            n0: i32,
            trellis1: *const core::ffi::c_void,
            suh1: *const core::ffi::c_void,
            svh1: *const core::ffi::c_void,
            y1: *mut core::ffi::c_void,
            y_dtype1: i32,
            n1: i32,
            m: i32,
            k: i32,
            k2: i32,
            codebook: i32,
            workspace: *mut core::ffi::c_void,
            workspace_bytes: usize,
            stream: *mut core::ffi::c_void,
            xh_pre0: *const core::ffi::c_void,
            xh_pre1: *const core::ffi::c_void,
            down_suh: *const core::ffi::c_void,
            down_xh: *mut core::ffi::c_void,
            col_cnt: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_linear_add_rmsnorm(
            x: *const core::ffi::c_void,
            x_dtype: i32,
            trellis: *const core::ffi::c_void,
            suh: *const core::ffi::c_void,
            svh: *const core::ffi::c_void,
            hidden: *const core::ffi::c_void,
            norm_w: *const core::ffi::c_void,
            hidden_out: *mut core::ffi::c_void,
            y: *mut core::ffi::c_void,
            m: i32,
            k: i32,
            n: i32,
            k2: i32,
            codebook: i32,
            dim: i32,
            eps: f32,
            workspace: *mut core::ffi::c_void,
            workspace_bytes: usize,
            stream: *mut core::ffi::c_void,
            skip_rot_in: i32,
            suh_p0: *const core::ffi::c_void,
            suh_p1: *const core::ffi::c_void,
            suh_p2: *const core::ffi::c_void,
            xh_p0: *mut core::ffi::c_void,
            xh_p1: *mut core::ffi::c_void,
            xh_p2: *mut core::ffi::c_void,
            lin_cnt: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_rmsnorm_linear_bf16(
            x: *mut core::ffi::c_void,
            norm_w: *const core::ffi::c_void,
            trellis: *const core::ffi::c_void,
            suh: *const core::ffi::c_void,
            svh: *const core::ffi::c_void,
            y: *mut core::ffi::c_void,
            m: i32,
            k: i32,
            n: i32,
            k2: i32,
            codebook: i32,
            eps: f32,
            workspace: *mut core::ffi::c_void,
            workspace_bytes: usize,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_mla_qkv_qb_fused(
            x: *const core::ffi::c_void,
            x_dtype: i32,
            res: *const core::ffi::c_void,
            norm_w: *const core::ffi::c_void,
            h_out: *mut core::ffi::c_void,
            y_norm: *mut core::ffi::c_void,
            trellis0: *const core::ffi::c_void,
            suh0: *const core::ffi::c_void,
            svh0: *const core::ffi::c_void,
            y0: *mut core::ffi::c_void,
            y_dtype0: i32,
            n0: i32,
            trellis1: *const core::ffi::c_void,
            suh1: *const core::ffi::c_void,
            svh1: *const core::ffi::c_void,
            y1: *mut core::ffi::c_void,
            y_dtype1: i32,
            n1: i32,
            qa_norm_w: *const core::ffi::c_void,
            qb_trellis: *const core::ffi::c_void,
            qb_suh: *const core::ffi::c_void,
            qb_svh: *const core::ffi::c_void,
            y2: *mut core::ffi::c_void,
            qb_k2: i32,
            n_qb: i32,
            m: i32,
            k: i32,
            k2: i32,
            codebook: i32,
            eps: f32,
            workspace: *mut core::ffi::c_void,
            workspace_bytes: usize,
            xh_pre0: *const core::ffi::c_void,
            xh_pre1: *const core::ffi::c_void,
            qb_xh: *mut core::ffi::c_void,
            qb_xh_bytes: usize,
            stream: *mut core::ffi::c_void,
            pair_cnt: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_bf16_linear_f32(
            x: *const core::ffi::c_void,
            w: *const core::ffi::c_void,
            y: *mut core::ffi::c_void,
            rows: i32,
            k: i32,
            n: i32,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_kda_proj_fused_triple(
            x: *const core::ffi::c_void,
            x_dtype: i32,
            trellis0: *const core::ffi::c_void,
            suh0: *const core::ffi::c_void,
            svh0: *const core::ffi::c_void,
            y0: *mut core::ffi::c_void,
            y_dtype0: i32,
            n0: i32,
            trellis1: *const core::ffi::c_void,
            suh1: *const core::ffi::c_void,
            svh1: *const core::ffi::c_void,
            y1: *mut core::ffi::c_void,
            y_dtype1: i32,
            n1: i32,
            trellis2: *const core::ffi::c_void,
            suh2: *const core::ffi::c_void,
            svh2: *const core::ffi::c_void,
            y2: *mut core::ffi::c_void,
            y_dtype2: i32,
            n2: i32,
            m: i32,
            k: i32,
            k2_0: i32,
            k2_12: i32,
            codebook: i32,
            workspace: *mut core::ffi::c_void,
            workspace_bytes: usize,
            stream: *mut core::ffi::c_void,
            xh_pre0: *const core::ffi::c_void,
            xh_pre1: *const core::ffi::c_void,
            xh_pre2: *const core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_norm_rot_in_triple(
            x: *const core::ffi::c_void,
            res: *const core::ffi::c_void,
            w: *const core::ffi::c_void,
            h_out: *mut core::ffi::c_void,
            y: *mut core::ffi::c_void,
            suh0: *const core::ffi::c_void,
            suh1: *const core::ffi::c_void,
            suh2: *const core::ffi::c_void,
            trellis0: *const core::ffi::c_void,
            svh0: *const core::ffi::c_void,
            y0: *mut core::ffi::c_void,
            y_dtype0: i32,
            n0: i32,
            trellis1: *const core::ffi::c_void,
            svh1: *const core::ffi::c_void,
            y1: *mut core::ffi::c_void,
            y_dtype1: i32,
            n1: i32,
            trellis2: *const core::ffi::c_void,
            svh2: *const core::ffi::c_void,
            y2: *mut core::ffi::c_void,
            y_dtype2: i32,
            n2: i32,
            m: i32,
            k: i32,
            k2_0: i32,
            k2_12: i32,
            codebook: i32,
            dim: i32,
            eps: f32,
            workspace: *mut core::ffi::c_void,
            workspace_bytes: usize,
            stream: *mut core::ffi::c_void,
            token_ids: *const core::ffi::c_void,
            embed_table: *const core::ffi::c_void,
            vocab: i32,
            ) -> i32;
        pub fn gb_ling_norm_rot_in_pair(
            x: *const core::ffi::c_void,
            res: *const core::ffi::c_void,
            w: *const core::ffi::c_void,
            h_out: *mut core::ffi::c_void,
            y: *mut core::ffi::c_void,
            suh0: *const core::ffi::c_void,
            suh1: *const core::ffi::c_void,
            trellis0: *const core::ffi::c_void,
            svh0: *const core::ffi::c_void,
            y0: *mut core::ffi::c_void,
            y_dtype0: i32,
            n0: i32,
            trellis1: *const core::ffi::c_void,
            svh1: *const core::ffi::c_void,
            y1: *mut core::ffi::c_void,
            y_dtype1: i32,
            n1: i32,
            m: i32,
            k: i32,
            k2: i32,
            codebook: i32,
            dim: i32,
            eps: f32,
            workspace: *mut core::ffi::c_void,
            workspace_bytes: usize,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_silu_mul_linear_add_rmsnorm(
            g: *const core::ffi::c_void,
            u: *const core::ffi::c_void,
            trellis: *const core::ffi::c_void,
            suh: *const core::ffi::c_void,
            svh: *const core::ffi::c_void,
            hidden: *const core::ffi::c_void,
            norm_w: *const core::ffi::c_void,
            hidden_out: *mut core::ffi::c_void,
            y: *mut core::ffi::c_void,
            m: i32,
            k: i32,
            n: i32,
            k2: i32,
            codebook: i32,
            dim: i32,
            eps: f32,
            workspace: *mut core::ffi::c_void,
            workspace_bytes: usize,
            stream: *mut core::ffi::c_void,
            suh_p0: *const core::ffi::c_void,
            suh_p1: *const core::ffi::c_void,
            suh_p2: *const core::ffi::c_void,
            xh_p0: *mut core::ffi::c_void,
            xh_p1: *mut core::ffi::c_void,
            xh_p2: *mut core::ffi::c_void,
            skip_rot_in: i32,
            lin_cnt: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_norm_rot_in_single(
            x: *const core::ffi::c_void,
            res: *const core::ffi::c_void,
            w: *const core::ffi::c_void,
            h_out: *mut core::ffi::c_void,
            y: *mut core::ffi::c_void,
            suh: *const core::ffi::c_void,
            xh: *mut core::ffi::c_void,
            m: i32,
            k: i32,
            dim: i32,
            eps: f32,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_p2_gemm(
            a: *const core::ffi::c_void,
            trellis: *const core::ffi::c_void,
            svh: *const core::ffi::c_void,
            bias: *const core::ffi::c_void,
            scale_f32: i32,
            y: *mut core::ffi::c_void,
            y_dtype: i32,
            ldy: i64,
            m: i32,
            k: i32,
            n: i32,
            k2: i32,
            codebook: i32,
            fold: i32,
            tile_force: i32,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_rmsnorm_rot_in(
            x: *const core::ffi::c_void,
            norm_w: *const core::ffi::c_void,
            y: *mut core::ffi::c_void,
            suh: *const core::ffi::c_void,
            xh: *mut core::ffi::c_void,
            m: i32,
            dim: i32,
            eps: f32,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_silu_mul_rot_in(
            g: *const core::ffi::c_void,
            u: *const core::ffi::c_void,
            suh: *const core::ffi::c_void,
            xh: *mut core::ffi::c_void,
            m: i32,
            k: i32,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_exl3_linear_prerot(
            trellis: *const core::ffi::c_void,
            svh: *const core::ffi::c_void,
            bias: *const core::ffi::c_void,
            y: *mut core::ffi::c_void,
            y_dtype: i32,
            m: i32,
            k: i32,
            n: i32,
            k2: i32,
            codebook: i32,
            xh: *const core::ffi::c_void,
            workspace: *mut core::ffi::c_void,
            workspace_bytes: usize,
            stream: *mut core::ffi::c_void,
            ) -> i32;
        pub fn gb_ling_topk_f16_workspace_bytes(rows: i32, valid: i32, count: i32) -> usize;
        pub fn gb_ling_topk_f16(
            logits: *const core::ffi::c_void,
            rows: i32,
            stride: i32,
            valid: i32,
            count: i32,
            ids: *mut core::ffi::c_void,
            vals: *mut core::ffi::c_void,
            workspace: *mut core::ffi::c_void,
            ws_bytes: usize,
            stream: *mut core::ffi::c_void,
            ) -> i32;
    }
}

const INVALID: CudaError = CudaError(1);

/// One side of a fused pair/triple: an EXL3 layer's device tensors and shape (no bias on these projections).
pub struct Proj<'a> {
    pub trellis: &'a DevBuf,
    pub suh: &'a DevBuf,
    pub svh: &'a DevBuf,
    pub y: &'a mut DevBuf,
    pub k: usize,
    pub n: usize,
    pub k2: usize,
    pub codebook: i32,
}

/// The next sub-layer's input rotation, fused into a fused linear epilogue (`ling_rot_out_add_rmsnorm_k`):
/// the epilogue rotates its own rmsnorm output with each projection's `suh` into the projection's rotated input
/// (`xh`), replacing the separate `rot_in_{pair,triple}` launch. Every field is optional; `pair`/`triple` fill
/// the common configurations, `Prerot::default()` disables the fusion.
#[derive(Default)]
pub struct Prerot<'a> {
    pub suh0: Option<&'a DevBuf>,
    pub suh1: Option<&'a DevBuf>,
    pub suh2: Option<&'a DevBuf>,
    pub xh0: Option<&'a mut DevBuf>,
    pub xh1: Option<&'a mut DevBuf>,
    pub xh2: Option<&'a mut DevBuf>,
}

impl<'a> Prerot<'a> {
    pub fn pair(suh0: &'a DevBuf, suh1: &'a DevBuf, xh0: &'a mut DevBuf, xh1: &'a mut DevBuf) -> Self {
        Self { suh0: Some(suh0), suh1: Some(suh1), xh0: Some(xh0), xh1: Some(xh1), ..Default::default() }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn triple(
        suh0: &'a DevBuf,
        suh1: &'a DevBuf,
        suh2: &'a DevBuf,
        xh0: &'a mut DevBuf,
        xh1: &'a mut DevBuf,
        xh2: &'a mut DevBuf,
    ) -> Self {
        Self { suh0: Some(suh0), suh1: Some(suh1), suh2: Some(suh2), xh0: Some(xh0), xh1: Some(xh1), xh2: Some(xh2) }
    }
    pub fn single(suh0: &'a DevBuf, xh0: &'a mut DevBuf) -> Self {
        Self { suh0: Some(suh0), xh0: Some(xh0), ..Default::default() }
    }
    pub fn is_empty(&self) -> bool {
        self.suh0.is_none() && self.suh1.is_none() && self.suh2.is_none()
    }
}

/// Bytes the fused triple reads/writes in `ws` for `m` rows of a `k`-wide input (THREE rotated inputs, `xh0/xh1/xh2`;
/// the triple runs SK = 1, so no split partials). This is also the largest a fused pair needs (its two xf buffers are a
/// subset), so it is what `workspace_bytes` returns.
pub fn workspace_bytes(m: usize, k: usize) -> usize {
    unsafe { ffi::gb_ling_kda_proj_fused_workspace_bytes(m as i32, k as i32) }
}

/// Bytes a fused pair needs in `ws` for `m` rows of a `k`-wide input with projections `[m, n0]` / `[m, n1]`: both
/// rotated inputs plus, when the pair splits K (narrow layers), the two fp32 split-partial buffers.
pub fn pair_workspace_bytes(m: usize, k: usize, n0: usize, n1: usize) -> usize {
    unsafe { ffi::gb_ling_kda_proj_fused_split_workspace_bytes(m as i32, k as i32, n0 as i32, n1 as i32) }
}

/// `y0 [m, n0] = f(x), y1 [m, n1] = g(x)` in one launch sharing the input rotation. Bit-identical per element to two
/// `Exl3Weights::k2_forward` calls with the same arguments. `ws` needs [`pair_workspace_bytes`](`m`, `k`, `n0`, `n1`).
/// `xh_pre0`/`xh_pre1`, when both given, are pre-rotated inputs (the previous fused epilogue wrote them): the input
/// rotation launch is skipped and they are used as-is (the non-fragment layout only).
#[allow(clippy::too_many_arguments)]
pub fn fused_decode_pair_ext(
    x: &DevBuf,
    f: Proj<'_>,
    g: Proj<'_>,
    m: usize,
    x_dtype: Dtype,
    y_dtype: Dtype,
    ws: &mut DevBuf,
    xh_pre0: Option<&DevBuf>,
    xh_pre1: Option<&DevBuf>,
    down_suh: Option<&DevBuf>,
    down_xh: Option<&mut DevBuf>,
    col_cnt: Option<&DevBuf>,
    st: &Stream,
) -> Result<(), CudaError> {
    debug_assert!(f.k == g.k && f.k2 == g.k2 && f.codebook == g.codebook);
    if m == 0
        || x.bytes() < m * f.k * x_dtype.size()
        || f.y.bytes() < m * f.n * y_dtype.size()
        || g.y.bytes() < m * g.n * y_dtype.size()
        || ws.bytes() < pair_workspace_bytes(m, f.k, f.n, g.n)
    {
        return Err(INVALID);
    }
    let rc = unsafe {
        ffi::gb_ling_kda_proj_fused(
            x.ptr(),
            x_dtype as i32,
            f.trellis.ptr(),
            f.suh.ptr(),
            f.svh.ptr(),
            f.y.ptr(),
            y_dtype as i32,
            f.n as i32,
            g.trellis.ptr(),
            g.suh.ptr(),
            g.svh.ptr(),
            g.y.ptr(),
            y_dtype as i32,
            g.n as i32,
            m as i32,
            f.k as i32,
            f.k2 as i32,
            f.codebook,
            ws.ptr(),
            ws.bytes(),
            st.raw(),
            xh_pre0.map_or(core::ptr::null(), |d| d.ptr()),
            xh_pre1.map_or(core::ptr::null(), |d| d.ptr()),
            down_suh.map_or(core::ptr::null(), |d| d.ptr()),
            down_xh.map_or(core::ptr::null_mut(), |d| d.ptr()),
            col_cnt.map_or(core::ptr::null_mut(), |d| d.ptr()),
        )
    };
    CudaError::check(rc)
}

/// [`fused_decode_pair_ext`] without the fused SwiGLU-rotation epilogue (the plain pair: `down_suh`/`down_xh`/
/// `col_cnt` all `None`). Bit-identical per element to two `Exl3Weights::k2_forward` calls with the same arguments.
#[allow(clippy::too_many_arguments)]
pub fn fused_decode_pair(
    x: &DevBuf,
    f: Proj<'_>,
    g: Proj<'_>,
    m: usize,
    x_dtype: Dtype,
    y_dtype: Dtype,
    ws: &mut DevBuf,
    xh_pre0: Option<&DevBuf>,
    xh_pre1: Option<&DevBuf>,
    st: &Stream,
) -> Result<(), CudaError> {
    fused_decode_pair_ext(x, f, g, m, x_dtype, y_dtype, ws, xh_pre0, xh_pre1, None, None, None, st)
}

/// `y [m, dim] = rmsnorm(h)` and `h = bf16(hidden + res)` with `res [m, n] =
/// layer(x [m, k])` (EXL3 k2 decode path) in ONE launch per stage instead of the
/// `gb_exl3_linear_k2` + `glue::add_rmsnorm` pair: the linear runs with the layer's
/// own decode plan into its fp32 split partials inside `ws`, and the fused epilogue
/// (ling_kda_fused) does the rot_out finish into shared memory and then glue.cu's
/// exact `rmsnorm_k<true>` math. Bit-identical per element to the separate launches
/// (the KDA `k.o`, MLA `dense` and dense FFN `down` sites of `bailing_moe_v3`:
/// n = dim = 4096). `hidden_out` may alias `hidden`. `ws` needs `linear_workspace_bytes`
/// (`m`, `k`, `n`, `k2 / 2`) bytes, the same size the separate `forward` call uses.
/// `skip_rot_in` skips the input rotation (the KDA window+norm epilogue already wrote
/// the rotated input into `ws`); `prerot` fuses the next sub-layer's input rotation
/// into this epilogue (see [`Prerot`]).
#[allow(clippy::too_many_arguments)]
pub fn linear_add_rmsnorm(
    x: &DevBuf,
    x_dtype: Dtype,
    trellis: &DevBuf,
    suh: &DevBuf,
    svh: &DevBuf,
    hidden: &DevBuf,
    norm_w: &DevBuf,
    hidden_out: &mut DevBuf,
    y: &mut DevBuf,
    m: usize,
    k: usize,
    n: usize,
    k2: usize,
    codebook: i32,
    dim: usize,
    eps: f32,
    ws: &mut DevBuf,
    skip_rot_in: bool,
    prerot: &Prerot<'_>,
    lin_cnt: Option<&DevBuf>,
    st: &Stream,
) -> Result<(), CudaError> {
    if m == 0
        || x.bytes() < m * k * x_dtype.size()
        || hidden.bytes() < m * dim * 2
        || norm_w.bytes() < dim * 4
        || hidden_out.bytes() < m * dim * 2
        || y.bytes() < m * dim * 2
        || ws.bytes() < crate::exl3_linear::linear_workspace_bytes(m, k, n, k2 / 2)
    {
        return Err(INVALID);
    }
    let rc = unsafe {
        ffi::gb_ling_linear_add_rmsnorm(
            x.ptr(),
            x_dtype as i32,
            trellis.ptr(),
            suh.ptr(),
            svh.ptr(),
            hidden.ptr(),
            norm_w.ptr(),
            hidden_out.ptr(),
            y.ptr(),
            m as i32,
            k as i32,
            n as i32,
            k2 as i32,
            codebook,
            dim as i32,
            eps,
            ws.ptr(),
            ws.bytes(),
            st.raw(),
            skip_rot_in as i32,
            prerot.suh0.map_or(core::ptr::null(), |d| d.ptr()),
            prerot.suh1.map_or(core::ptr::null(), |d| d.ptr()),
            prerot.suh2.map_or(core::ptr::null(), |d| d.ptr()),
            prerot.xh0.as_ref().map_or(core::ptr::null_mut(), |d| d.ptr()),
            prerot.xh1.as_ref().map_or(core::ptr::null_mut(), |d| d.ptr()),
            prerot.xh2.as_ref().map_or(core::ptr::null_mut(), |d| d.ptr()),
            lin_cnt.map_or(core::ptr::null_mut(), |d| d.ptr()),
        )
    };
    CudaError::check(rc)
}

/// `y [m, n] = layer(x [m, k])` with the in-place rmsnorm of `x` (the MLA `q_a`, bf16 `[m, k]`,
/// `k == dim == q_lora_rank`) folded into the layer's input rotation: ONE launch runs the exact
/// `glue::rmsnorm` math in place on `x` and writes the rotated input, and the linear kernel finishes inline
/// (the layer's own decode plan, SK == 1). Bit-identical per element to `glue::rmsnorm` +
/// `Exl3Weights::k2_forward` with the same arguments (same norm bits, same xh bytes, same k order, same
/// warp-order sum, same finish). The MLA `q_b` site of `bailing_moe_v3` (k2 = 7, codebook 2, k = 1536,
/// n = 12288, rows <= 8). Returns `Err` for shapes the fusion does not cover (rows > 8, k > 1536, a split-K
/// plan, an uninstantiated k2/codebook pair): the caller falls back to the separate launches. `ws` needs
/// [`workspace_bytes`] (`m`, `k`) bytes (the rotated input only). `x` is modified in place.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_linear_bf16(
    x: &mut DevBuf,
    norm_w: &DevBuf,
    trellis: &DevBuf,
    suh: &DevBuf,
    svh: &DevBuf,
    y: &mut DevBuf,
    m: usize,
    k: usize,
    n: usize,
    k2: usize,
    codebook: i32,
    eps: f32,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    if m == 0
        || m > 8
        || k == 0
        || k % 128 != 0
        || k > 1536
        || x.bytes() < m * k * 2
        || norm_w.bytes() < k * 4
        || y.bytes() < m * n * 2
        || ws.bytes() < workspace_bytes(m, k)
    {
        return Err(INVALID);
    }
    let rc = unsafe {
        ffi::gb_ling_rmsnorm_linear_bf16(
            x.ptr(),
            norm_w.ptr(),
            trellis.ptr(),
            suh.ptr(),
            svh.ptr(),
            y.ptr(),
            m as i32,
            k as i32,
            n as i32,
            k2 as i32,
            codebook,
            eps,
            ws.ptr(),
            ws.bytes(),
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// The MLA `q_a`/`kv_a` pair AND the `q_b` step in one fused call (the decode/verify `MixerDevice::Mla`
/// site of `bailing_moe_v3`, rows <= 8): the pair runs exactly as [`fused_decode_pair`] (or
/// [`norm_rot_in_pair`] when `norm_w` is given — the input norm is then computed inside), and the pair's two
/// split-K finishes (`rot_out_kernel`), the q_b input norm and the q_b input rotation collapse into one launch
/// (`ling_mla_rot_out_qnorm_qb_rot_in_kernel`) whose output the q_b linear (its own decode plan, SK = 1)
/// reads. Bit-identical per element to `fused_decode_pair`/`norm_rot_in_pair` + [`rmsnorm_linear_bf16`]
/// with the same arguments. `f` is the q_a projection (`k` = hidden, `n` = q_lora_rank, `k2` = 10), `g` the
/// kv_a one (same `k`/`k2`, `n` = kv_a group width); `qb` is the q_b projection (`k` = q_lora_rank =
/// `f.n`, `k2` = 7, `n` = heads * qk_head_dim). `qb_xh` is the caller's buffer (>= m * f.n * 2 bytes, e.g. a
/// `prerot_ws` slice), separate from `ws` (`pair_workspace_bytes`). Returns `Err` for shapes the fusion does
/// not cover: the caller falls back to the separate sequence.
#[allow(clippy::too_many_arguments)]
pub fn mla_qkv_qb_fused(
    x: &DevBuf,
    x_dtype: Dtype,
    res: Option<&DevBuf>,
    norm_w: Option<&DevBuf>,
    h_out: Option<&mut DevBuf>,
    y_norm: Option<&mut DevBuf>,
    f: Proj<'_>,
    g: Proj<'_>,
    qa_norm_w: &DevBuf,
    qb: Proj<'_>,
    m: usize,
    eps: f32,
    ws: &mut DevBuf,
    xh_pre0: Option<&DevBuf>,
    xh_pre1: Option<&DevBuf>,
    qb_xh: &mut DevBuf,
    pair_cnt: Option<&DevBuf>,
    st: &Stream,
) -> Result<(), CudaError> {
    debug_assert!(f.k == g.k && f.k2 == g.k2 && f.codebook == g.codebook);
    debug_assert!(qb.k == f.n);
    if m == 0
        || m > 8
        || x.bytes() < m * f.k * x_dtype.size()
        || f.y.bytes() < m * f.n * 2
        || g.y.bytes() < m * g.n * 2
        || qa_norm_w.bytes() < f.n * 4
        || qb.y.bytes() < m * qb.n * 2
        || ws.bytes() < pair_workspace_bytes(m, f.k, f.n, g.n)
        || qb_xh.bytes() < m * f.n * 2
        || (norm_w.is_some() && y_norm.is_none())
    {
        return Err(INVALID);
    }
    let rc = unsafe {
        ffi::gb_ling_mla_qkv_qb_fused(
            x.ptr(),
            x_dtype as i32,
            res.map_or(core::ptr::null(), |d| d.ptr()),
            norm_w.map_or(core::ptr::null(), |d| d.ptr()),
            h_out.as_ref().map_or(core::ptr::null_mut(), |d| d.ptr()),
            y_norm.as_ref().map_or(core::ptr::null_mut(), |d| d.ptr()),
            f.trellis.ptr(),
            f.suh.ptr(),
            f.svh.ptr(),
            f.y.ptr(),
            1, // BF16
            f.n as i32,
            g.trellis.ptr(),
            g.suh.ptr(),
            g.svh.ptr(),
            g.y.ptr(),
            1, // BF16
            g.n as i32,
            qa_norm_w.ptr(),
            qb.trellis.ptr(),
            qb.suh.ptr(),
            qb.svh.ptr(),
            qb.y.ptr(),
            qb.k2 as i32,
            qb.n as i32,
            m as i32,
            f.k as i32,
            f.k2 as i32,
            f.codebook,
            eps,
            ws.ptr(),
            ws.bytes(),
            xh_pre0.map_or(core::ptr::null(), |d| d.ptr()),
            xh_pre1.map_or(core::ptr::null(), |d| d.ptr()),
            qb_xh.ptr(),
            qb_xh.bytes(),
            st.raw(),
            pair_cnt.map_or(core::ptr::null_mut(), |d| d.ptr()),
        )
    };
    CudaError::check(rc)
}

/// `y [rows, n]` fp32 = `x [rows, k]` bf16 @ `w [n, k]^T` bf16 for the Ling-3.1
/// MoE router (n = 512, k = 4096, any row count): one warp per output column, 16
/// warps per CTA, 32 CTAs (a single wave on GB10), rows in chunks of 8 with the
/// weight row loaded once per column per chunk. Bit-identical per (row, column) to
/// [`mla_gated::bf16_linear_f32`] (same k order, same lane stride, same FMA chain,
/// same warp reduction, same fp32 store).
pub fn bf16_linear_f32(
    x: &DevBuf,
    w: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    k: usize,
    n: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0 || x.bytes() < rows * k * 2 || w.bytes() < n * k * 2 || y.bytes() < rows * n * 4 {
        return Err(INVALID);
    }
    let rc = unsafe { ffi::gb_ling_bf16_linear_f32(x.ptr(), w.ptr(), y.ptr(), rows as i32, k as i32, n as i32, st.raw()) };
    CudaError::check(rc)
}
#[allow(clippy::too_many_arguments)]
pub fn fused_decode_triple(
    x: &DevBuf,
    p0: Proj<'_>,
    p1: Proj<'_>,
    p2: Proj<'_>,
    m: usize,
    x_dtype: Dtype,
    y_dtype0: Dtype,
    y_dtype12: Dtype,
    ws: &mut DevBuf,
    xh_pre0: Option<&DevBuf>,
    xh_pre1: Option<&DevBuf>,
    xh_pre2: Option<&DevBuf>,
    st: &Stream,
) -> Result<(), CudaError> {
    debug_assert!(p0.k == p1.k && p1.k == p2.k && p1.k2 == p2.k2 && p0.codebook == p1.codebook && p1.codebook == p2.codebook);
    if m == 0
        || x.bytes() < m * p0.k * x_dtype.size()
        || p0.y.bytes() < m * p0.n * y_dtype0.size()
        || p1.y.bytes() < m * p1.n * y_dtype12.size()
        || p2.y.bytes() < m * p2.n * y_dtype12.size()
        || ws.bytes() < workspace_bytes(m, p0.k)
    {
        return Err(INVALID);
    }
    let rc = unsafe {
        ffi::gb_ling_kda_proj_fused_triple(
            x.ptr(),
            x_dtype as i32,
            p0.trellis.ptr(),
            p0.suh.ptr(),
            p0.svh.ptr(),
            p0.y.ptr(),
            y_dtype0 as i32,
            p0.n as i32,
            p1.trellis.ptr(),
            p1.suh.ptr(),
            p1.svh.ptr(),
            p1.y.ptr(),
            y_dtype12 as i32,
            p1.n as i32,
            p2.trellis.ptr(),
            p2.suh.ptr(),
            p2.svh.ptr(),
            p2.y.ptr(),
            y_dtype12 as i32,
            p2.n as i32,
            m as i32,
            p0.k as i32,
            p0.k2 as i32,
            p1.k2 as i32,
            p0.codebook,
            ws.ptr(),
            ws.bytes(),
            st.raw(),
            xh_pre0.map_or(core::ptr::null(), |d| d.ptr()),
            xh_pre1.map_or(core::ptr::null(), |d| d.ptr()),
            xh_pre2.map_or(core::ptr::null(), |d| d.ptr()),
        )
    };
    CudaError::check(rc)
}

/// `y [m, dim] = rmsnorm(h)` and `h = bf16(hidden + res)` with `res [m, n] =
/// layer(x [m, k])` (EXL3 k2 decode path) fused with the layer's input rotation:
/// one launch runs glue.cu's exact `rmsnorm_k<true>` math (h_out/y written as by
/// `glue::add_rmsnorm`) and rotates the normalised row into `xh0/xh1/xh2` with
/// the exact `ling_rot_in_triple_kernel` math, then the triple's linear kernels
/// finish inline (SK == 1, `dispatch_triple`). Bit-identical per element to
/// `glue::add_rmsnorm` + `gb_ling_kda_proj_fused_triple` (and, with `res = None`
/// / `hidden_out = None`, to `glue::rmsnorm` + the triple). The layer-0 /
/// post-MoE norm sites of `bailing_moe_v3` (k = dim = 4096, rows <= 8).
/// Returns `Err` for shapes the fusion does not cover (m > 64, a split-K plan, an
/// uninstantiated k2/codebook triple): the caller falls back to the separate
/// launches. `ws` needs [`workspace_bytes`] (`m`, `k`) bytes. `hidden_out` may
/// alias `x` (the model's residual stream is updated in place).
#[allow(clippy::too_many_arguments)]
pub fn norm_rot_in_triple_ext(
    x: &DevBuf,
    res: Option<&DevBuf>,
    norm_w: &DevBuf,
    hidden_out: Option<&mut DevBuf>,
    y: &mut DevBuf,
    p0: Proj<'_>,
    p1: Proj<'_>,
    p2: Proj<'_>,
    m: usize,
    dim: usize,
    y_dtype0: Dtype,
    y_dtype12: Dtype,
    k2_0: usize,
    k2_12: usize,
    codebook: i32,
    eps: f32,
    ws: &mut DevBuf,
    token_ids: Option<&DevBuf>,
    embed_table: Option<&DevBuf>,
    vocab: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    debug_assert!(p0.k == p1.k && p1.k == p2.k && p1.k2 == p2.k2 && p0.codebook == p1.codebook && p1.codebook == p2.codebook);
    if m == 0
        || m > 64
        || dim == 0
        || dim > 4096
        || dim % 8 != 0
        || p0.k != dim
        || x.bytes() < m * dim * 2
        || norm_w.bytes() < dim * 4
        || y.bytes() < m * dim * 2
        || p0.y.bytes() < m * p0.n * y_dtype0.size()
        || p1.y.bytes() < m * p1.n * y_dtype12.size()
        || p2.y.bytes() < m * p2.n * y_dtype12.size()
        || ws.bytes() < workspace_bytes(m, p0.k)
    {
        return Err(INVALID);
    }
    if let Some(r) = res {
        if r.bytes() < m * dim * 2 {
            return Err(INVALID);
        }
    }
    if (token_ids.is_some()) != (embed_table.is_some()) {
        return Err(INVALID);
    }
    if let Some(t) = token_ids {
        if let Some(h) = hidden_out.as_deref() {
            if h.bytes() < m * dim * 2 || t.bytes() < m * 4 {
                return Err(INVALID);
            }
        } else {
            return Err(INVALID); // embed mode writes the hidden stream (h_out)
        }
        if let Some(e) = embed_table {
            if e.bytes() < vocab * dim * 2 {
                return Err(INVALID);
            }
        }
    }
    if let Some(h) = hidden_out.as_deref() {
        if h.bytes() < m * dim * 2 {
            return Err(INVALID);
        }
    }
    let rc = unsafe {
        ffi::gb_ling_norm_rot_in_triple(
            x.ptr(),
            res.map_or(core::ptr::null(), |r| r.ptr()),
            norm_w.ptr(),
            hidden_out.map_or(core::ptr::null_mut(), |h| h.ptr()),
            y.ptr(),
            p0.suh.ptr(),
            p1.suh.ptr(),
            p2.suh.ptr(),
            p0.trellis.ptr(),
            p0.svh.ptr(),
            p0.y.ptr(),
            y_dtype0 as i32,
            p0.n as i32,
            p1.trellis.ptr(),
            p1.svh.ptr(),
            p1.y.ptr(),
            y_dtype12 as i32,
            p1.n as i32,
            p2.trellis.ptr(),
            p2.svh.ptr(),
            p2.y.ptr(),
            y_dtype12 as i32,
            p2.n as i32,
            m as i32,
            p0.k as i32,
            k2_0 as i32,
            k2_12 as i32,
            codebook,
            dim as i32,
            eps,
            ws.ptr(),
            ws.bytes(),
            st.raw(),
            token_ids.map_or(core::ptr::null(), |t| t.ptr()),
            embed_table.map_or(core::ptr::null(), |e| e.ptr()),
            vocab as i32,
        )
    };
    CudaError::check(rc)
}

/// [`norm_rot_in_triple_ext`] without the embedding fusion (`token_ids`/`embed_table`/`vocab` all `None`).
#[allow(clippy::too_many_arguments)]
pub fn norm_rot_in_triple(
    x: &DevBuf,
    res: Option<&DevBuf>,
    norm_w: &DevBuf,
    hidden_out: Option<&mut DevBuf>,
    y: &mut DevBuf,
    p0: Proj<'_>,
    p1: Proj<'_>,
    p2: Proj<'_>,
    m: usize,
    dim: usize,
    y_dtype0: Dtype,
    y_dtype12: Dtype,
    k2_0: usize,
    k2_12: usize,
    codebook: i32,
    eps: f32,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    norm_rot_in_triple_ext(
        x, res, norm_w, hidden_out, y, p0, p1, p2, m, dim, y_dtype0, y_dtype12, k2_0, k2_12, codebook, eps, ws, None, None, 0, st,
    )
}

/// The pair analogue of [`norm_rot_in_triple`]: exact `rmsnorm_k<true>/<false>` +
/// the `ling_rot_in_pair_kernel` rotation in one launch, then the pair's linear
/// kernels (the projections must share the decode plan). Bit-identical per element
/// to `glue::rmsnorm`/`glue::add_rmsnorm` + `gb_ling_kda_proj_fused`. The MLA
/// `q_a + kv_a` and layer-0 sites of `bailing_moe_v3`. `ws` needs
/// [`pair_workspace_bytes`] (`m`, `k`, `n0`, `n1`) bytes.
#[allow(clippy::too_many_arguments)]
pub fn norm_rot_in_pair(
    x: &DevBuf,
    res: Option<&DevBuf>,
    norm_w: &DevBuf,
    hidden_out: Option<&mut DevBuf>,
    y: &mut DevBuf,
    f: Proj<'_>,
    g: Proj<'_>,
    m: usize,
    dim: usize,
    y_dtype: Dtype,
    k2: usize,
    codebook: i32,
    eps: f32,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    debug_assert!(f.k == g.k && f.k2 == g.k2 && f.codebook == g.codebook);
    if m == 0
        || m > 64
        || dim == 0
        || dim > 4096
        || dim % 8 != 0
        || f.k != dim
        || x.bytes() < m * dim * 2
        || norm_w.bytes() < dim * 4
        || y.bytes() < m * dim * 2
        || f.y.bytes() < m * f.n * y_dtype.size()
        || g.y.bytes() < m * g.n * y_dtype.size()
        || ws.bytes() < pair_workspace_bytes(m, f.k, f.n, g.n)
    {
        return Err(INVALID);
    }
    if let Some(r) = res {
        if r.bytes() < m * dim * 2 {
            return Err(INVALID);
        }
    }
    if let Some(h) = hidden_out.as_deref() {
        if h.bytes() < m * dim * 2 {
            return Err(INVALID);
        }
    }
    let rc = unsafe {
        ffi::gb_ling_norm_rot_in_pair(
            x.ptr(),
            res.map_or(core::ptr::null(), |r| r.ptr()),
            norm_w.ptr(),
            hidden_out.map_or(core::ptr::null_mut(), |h| h.ptr()),
            y.ptr(),
            f.suh.ptr(),
            g.suh.ptr(),
            f.trellis.ptr(),
            f.svh.ptr(),
            f.y.ptr(),
            y_dtype as i32,
            f.n as i32,
            g.trellis.ptr(),
            g.svh.ptr(),
            g.y.ptr(),
            y_dtype as i32,
            g.n as i32,
            m as i32,
            f.k as i32,
            k2 as i32,
            codebook,
            dim as i32,
            eps,
            ws.ptr(),
            ws.bytes(),
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// `y [m, dim] = rmsnorm(h)` and `h = bf16(hidden + res)` with `res [m, n] =
/// layer(swiglu(g, u) [m, k])` (EXL3 k2 decode path) in ONE launch per stage: the
/// exact `glue::swiglu` element (silu_f(g) * u, bf16) is fused into the layer's
/// input rotation (cf. [`norm_rot_in_triple`]), then the layer's own decode plan
/// and the rot_out + add_rmsnorm epilogue run as in [`linear_add_rmsnorm`].
/// Bit-identical per element to `glue::swiglu` + [`linear_add_rmsnorm`] with the
/// same arguments. The dense FFN down_proj site of `bailing_moe_v3` (k2 = 7,
/// codebook 2, k = 10240, n = dim = 4096, rows <= 8). Returns `Err` for shapes the
/// fusion does not cover: the caller falls back to the separate launches. `ws`
/// needs [`linear_workspace_bytes`] (`m`, `k`, `n`, `k2 / 2`) bytes. `hidden_out`
/// may alias `hidden`.
#[allow(clippy::too_many_arguments)]
pub fn silu_mul_linear_add_rmsnorm_ext(
    g: &DevBuf,
    u: &DevBuf,
    trellis: &DevBuf,
    suh: &DevBuf,
    svh: &DevBuf,
    hidden: &DevBuf,
    norm_w: &DevBuf,
    hidden_out: &mut DevBuf,
    y: &mut DevBuf,
    m: usize,
    k: usize,
    n: usize,
    k2: usize,
    codebook: i32,
    dim: usize,
    eps: f32,
    ws: &mut DevBuf,
    prerot: &Prerot<'_>,
    skip_rot_in: bool,
    lin_cnt: Option<&DevBuf>,
    st: &Stream,
) -> Result<(), CudaError> {
    if m == 0
        || m > 64
        || g.bytes() < m * k * 2
        || u.bytes() < m * k * 2
        || hidden.bytes() < m * dim * 2
        || norm_w.bytes() < dim * 4
        || hidden_out.bytes() < m * dim * 2
        || y.bytes() < m * dim * 2
        || ws.bytes() < crate::exl3_linear::linear_workspace_bytes(m, k, n, k2 / 2)
    {
        return Err(INVALID);
    }
    let rc = unsafe {
        ffi::gb_ling_silu_mul_linear_add_rmsnorm(
            g.ptr(),
            u.ptr(),
            trellis.ptr(),
            suh.ptr(),
            svh.ptr(),
            hidden.ptr(),
            norm_w.ptr(),
            hidden_out.ptr(),
            y.ptr(),
            m as i32,
            k as i32,
            n as i32,
            k2 as i32,
            codebook,
            dim as i32,
            eps,
            ws.ptr(),
            ws.bytes(),
            st.raw(),
            prerot.suh0.map_or(core::ptr::null(), |d| d.ptr()),
            prerot.suh1.map_or(core::ptr::null(), |d| d.ptr()),
            prerot.suh2.map_or(core::ptr::null(), |d| d.ptr()),
            prerot.xh0.as_ref().map_or(core::ptr::null_mut(), |d| d.ptr()),
            prerot.xh1.as_ref().map_or(core::ptr::null_mut(), |d| d.ptr()),
            prerot.xh2.as_ref().map_or(core::ptr::null_mut(), |d| d.ptr()),
            skip_rot_in as i32,
            lin_cnt.map_or(core::ptr::null_mut(), |d| d.ptr()),
        )
    };
    CudaError::check(rc)
}

/// [`silu_mul_linear_add_rmsnorm_ext`] with `skip_rot_in = false` (the standalone swiglu input rotation runs).
#[allow(clippy::too_many_arguments)]
pub fn silu_mul_linear_add_rmsnorm(
    g: &DevBuf,
    u: &DevBuf,
    trellis: &DevBuf,
    suh: &DevBuf,
    svh: &DevBuf,
    hidden: &DevBuf,
    norm_w: &DevBuf,
    hidden_out: &mut DevBuf,
    y: &mut DevBuf,
    m: usize,
    k: usize,
    n: usize,
    k2: usize,
    codebook: i32,
    dim: usize,
    eps: f32,
    ws: &mut DevBuf,
    prerot: &Prerot<'_>,
    lin_cnt: Option<&DevBuf>,
    st: &Stream,
) -> Result<(), CudaError> {
    silu_mul_linear_add_rmsnorm_ext(
        g, u, trellis, suh, svh, hidden, norm_w, hidden_out, y, m, k, n, k2, codebook, dim, eps, ws, prerot, false, lin_cnt, st,
    )
}

/// The trunk `lm_head`'s input norm fused with its input rotation: `h = bf16(hidden + res)` and
/// `y = rmsnorm(h; norm_w)` with the exact `glue::add_rmsnorm` math (writing both), then the exact
/// `ling_rot_in_kernel` math of `suh` into `xh` (fp16 `[m, k]`). Bit-identical per element to
/// `glue::add_rmsnorm` + the rot_in launch of `Exl3Weights::k2_forward`; [`linear_prerot`] then runs the
/// head's linear on `xh` (no rot_in launch). The last-layer post-MoE site of `bailing_moe_v3` and the P2
/// prefill norm sites of the trunk (k = dim = 4096, rows <= 4096). `res == None` runs plain rmsnorm.
#[allow(clippy::too_many_arguments)]
pub fn norm_rot_in_single(
    x: &DevBuf,
    res: Option<&DevBuf>,
    norm_w: &DevBuf,
    hidden_out: Option<&mut DevBuf>,
    y: &mut DevBuf,
    suh: &DevBuf,
    xh: &mut DevBuf,
    m: usize,
    dim: usize,
    eps: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    if m == 0
        || m > 4096
        || dim == 0
        || dim > 4096
        || dim % 8 != 0
        || x.bytes() < m * dim * 2
        || norm_w.bytes() < dim * 4
        || y.bytes() < m * dim * 2
        || suh.bytes() < dim * 2
        || xh.bytes() < m * dim * 2
    {
        return Err(INVALID);
    }
    if let Some(r) = res {
        if r.bytes() < m * dim * 2 {
            return Err(INVALID);
        }
    }
    if let Some(h) = hidden_out.as_deref() {
        if h.bytes() < m * dim * 2 {
            return Err(INVALID);
        }
    }
    let rc = unsafe {
        ffi::gb_ling_norm_rot_in_single(
            x.ptr(),
            res.map_or(core::ptr::null(), |r| r.ptr()),
            norm_w.ptr(),
            hidden_out.map_or(core::ptr::null_mut(), |h| h.ptr()),
            y.ptr(),
            suh.ptr(),
            xh.ptr(),
            m as i32,
            dim as i32,
            dim as i32,
            eps,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// The P2 (LP3 direct-to-MMA tensor-core GEMM) of a `Lin` on a PRE-ROTATED fp16 input `xh` (`[rows, k]`,
/// written by a fused epilogue: [`norm_rot_in_single`], the KDA prefill norm-commit kernel, the MoE prefill
/// combine kernel, `ling_rmsnorm_rot_in_kernel` or `ling_silu_mul_rot_in_kernel`): the
/// `gb_glm_p2_rotate` launch is skipped and `gb_ling_p2_gemm` runs directly on `xh`. Same bits as
/// `glm_p2::ling_linear_parts` when `xh` holds its rotate output (same k order, `Chain`, same tile pick, same
/// finish). `suh` is not read (the rotation already applied it). Asynchronous on `st`.
#[allow(clippy::too_many_arguments)]
pub fn ling_p2_gemm_prerot(
    xh: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    y_dtype: Dtype,
    k: usize,
    n: usize,
    k2: usize,
    codebook: gb_exl3::Codebook,
    trellis: &DevBuf,
    svh: &DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0
        || rows > i32::MAX as usize / k.max(n).max(1)
        || !crate::glm_p2::ling_supported(k, n, k2, codebook)
        || xh.bytes() < rows * k * 2
        || y.bytes() < rows * n * y_dtype.size()
        || trellis.bytes() < k * n * k2 / 16
        || svh.bytes() < n * 2
    {
        return Err(INVALID);
    }
    let rc = unsafe {
        ffi::gb_ling_p2_gemm(
            xh.ptr(),
            trellis.ptr(),
            svh.ptr(),
            core::ptr::null(),
            0,
            y.ptr(),
            y_dtype as i32,
            n as i64,
            rows as i32,
            k as i32,
            n as i32,
            k2 as i32,
            codebook as i32,
            0, // fold = Chain (LING_DEFAULT_VARIANT)
            0, // tile_force = 0 (the kernel's pick)
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// The rmsnorm of `x` (bf16 `[m, dim]`, in place via `y`) fused with the fp16 input rotation of a P2
/// projection: one launch runs the exact `glue::rmsnorm_k<false>` math (writing `y`) and rotates the normalised
/// row by `suh` into `xh` (fp16 `[m, k]`, `k == dim`) with the exact `ling_rot_in_kernel` math — bit-identical
/// per element to `glue::rmsnorm` + the `gb_glm_p2_rotate` launch `glm_p2::ling_linear_parts` runs. The P2
/// q_a -> q_b step of `bailing_moe_v3` (dim = q_lora_rank <= 1536, rows <= 4096). `xh` needs `rows * k * 2`
/// bytes. Returns `Err` for a shape the fusion does not cover: the caller falls back to the separate launches.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_rot_in(
    x: &DevBuf,
    norm_w: &DevBuf,
    y: &mut DevBuf,
    suh: &DevBuf,
    xh: &mut DevBuf,
    m: usize,
    dim: usize,
    eps: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    if m == 0
        || m > 4096
        || dim == 0
        || dim > 1536
        || dim % 8 != 0
        || dim % 128 != 0
        || eps <= 0.0
        || x.bytes() < m * dim * 2
        || norm_w.bytes() < dim * 4
        || y.bytes() < m * dim * 2
        || suh.bytes() < dim * 2
        || xh.bytes() < m * dim * 2
    {
        return Err(INVALID);
    }
    let rc = unsafe {
        ffi::gb_ling_rmsnorm_rot_in(x.ptr(), norm_w.ptr(), y.ptr(), suh.ptr(), xh.ptr(), m as i32, dim as i32, eps, st.raw())
    };
    CudaError::check(rc)
}

/// The SwiGLU of `g`/`u` (bf16 `[m, k]`) fused with the fp16 input rotation of a P2 projection: one launch
/// computes the exact `glue::ew_vec_k<OpSwiglu>` element (silu_f(g) * u rounded to bf16) and rotates it by
/// `suh` into `xh` (fp16 `[m, k]`) with the exact `ling_rot_in_kernel` math — bit-identical per element to
/// `glue::swiglu` + the `gb_glm_p2_rotate` launch `glm_p2::ling_linear_parts` runs. The P2 dense FFN down_proj
/// step of `bailing_moe_v3` (k = dense_intermediate <= 10240, rows <= 4096). `xh` needs `rows * k * 2` bytes.
#[allow(clippy::too_many_arguments)]
pub fn silu_mul_rot_in(
    g: &DevBuf,
    u: &DevBuf,
    suh: &DevBuf,
    xh: &mut DevBuf,
    m: usize,
    k: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if m == 0
        || m > 4096
        || k == 0
        || k % 128 != 0
        || g.bytes() < m * k * 2
        || u.bytes() < m * k * 2
        || suh.bytes() < k * 2
        || xh.bytes() < m * k * 2
    {
        return Err(INVALID);
    }
    let rc = unsafe { ffi::gb_ling_silu_mul_rot_in(g.ptr(), u.ptr(), suh.ptr(), xh.ptr(), m as i32, k as i32, st.raw()) };
    CudaError::check(rc)
}

/// The linear half of `Exl3Weights::k2_forward` on a caller-provided rotated input `xh` (fp16 `[m, k]`, e.g.
/// written by [`norm_rot_in_single`]): the same decode plan, the same k order, the same warp-order sum and the
/// same finish, so `y` is bit-identical to `k2_forward` on `x` after its own rot_in. Covers the non-fragment
/// layout (m <= 16) and SK == 1 plans (the trunk `lm_head`: k2 = 12, codebook 2, n = vocab). `ws` is the
/// caller's decode workspace (>= the rotated input's bytes); `xh` may be `ws` itself. Returns `Err` for a shape
/// the fusion does not cover: the caller falls back to the separate launches.
#[allow(clippy::too_many_arguments)]
pub fn linear_prerot(
    trellis: &DevBuf,
    svh: &DevBuf,
    bias: Option<&DevBuf>,
    y: &mut DevBuf,
    y_dtype: Dtype,
    m: usize,
    k: usize,
    n: usize,
    k2: usize,
    codebook: i32,
    xh: &DevBuf,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    if m == 0
        || m > 16
        || k == 0
        || n == 0
        || n % 128 != 0
        || k2 < 2
        || k2 > 16
        || trellis.bytes() == 0
        || svh.bytes() < n * 2
        || y.bytes() < m * n * y_dtype.size()
        || xh.bytes() < m * k * 2
        || ws.bytes() < m * k * 2
    {
        return Err(INVALID);
    }
    let rc = unsafe {
        ffi::gb_ling_exl3_linear_prerot(
            trellis.ptr(),
            svh.ptr(),
            bias.map_or(core::ptr::null(), |b| b.ptr()),
            y.ptr(),
            y_dtype as i32,
            m as i32,
            k as i32,
            n as i32,
            k2 as i32,
            codebook,
            xh.ptr(),
            ws.ptr(),
            ws.bytes(),
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// Workspace bytes [`topk_f16`] needs (0 for an unsupported shape): the stage-1 pools of
/// `glm_glue::topk_workspace_bytes`, plus the `rows` int32 fused-combine counters.
pub fn topk_workspace_bytes(rows: usize, valid: usize, count: usize) -> usize {
    if rows > i32::MAX as usize || valid > i32::MAX as usize || count > i32::MAX as usize {
        return 0;
    }
    unsafe { ffi::gb_ling_topk_f16_workspace_bytes(rows as i32, valid as i32, count as i32) }
}

/// Exact per-row top-`count` (value descending, ties to the lower id) of fp16 logits `[rows, stride]` over columns
/// `0..valid` (the columns past `valid` are never read), in ONE launch (the stage-1 CTAs write their pool candidates
/// and the row's LAST CTA folds them). Bit-identical ids/vals to `glm_glue::topk_f16` on the same inputs.
#[allow(clippy::too_many_arguments)]
pub fn topk_f16(
    logits: &DevBuf,
    rows: usize,
    stride: usize,
    valid: usize,
    count: usize,
    ids: &mut DevBuf,
    vals: &mut DevBuf,
    workspace: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    let need = topk_workspace_bytes(rows, valid, count);
    if rows == 0
        || need == 0
        || valid > stride
        || logits.bytes() < rows * stride * 2
        || ids.bytes() < rows * count * 4
        || vals.bytes() < rows * count * 4
        || workspace.bytes() < need
    {
        return Err(INVALID);
    }
    let rc = unsafe {
        ffi::gb_ling_topk_f16(
            logits.ptr(),
            rows as i32,
            stride as i32,
            valid as i32,
            count as i32,
            ids.ptr(),
            vals.ptr(),
            workspace.ptr(),
            workspace.bytes(),
            st.raw(),
        )
    };
    CudaError::check(rc)
}
