//! Exact tensor-parallel cuts of an EXL3 layer (lane E, GLM Phase 2, plan items 4-6: `csrc/exl3_linear.cu` "The tensor-parallel splits").
//!
//! A K x N layer's decode kernel reduces each output column over K in a fixed tree that depends only on the layer's plan `(sk, wk)` =
//! `plan_for(K, N)`: `sk` K splits of `K / 16 / sk` tile rows, each split the warp-order sum of `wk` warps over their k ranges, the splits
//! added in split order by `rot_out` before the output Hadamard (which works per 128-column block). Two cuts keep that tree, so they give
//! the parent's bits:
//!
//! * **S1, a column strip** (whole 128-column blocks): run with the PARENT's plan, every kept column is reduced exactly as in the parent,
//!   at any strip width (the plain `select_strips` keeps the bits only when the strip's own shape gives the same plan, `N >= 64` strips).
//!   [`Exl3Weights::forward_strip`] / [`Exl3Weights::forward_big_strip`] read the parent's trellis in place (the row stride is the parent's
//!   N); [`Exl3Weights::select_strips_planned`] copies the strips and keeps the parent's plan ([`PlannedExl3`]).
//! * **S2, a K range of whole splits** `[s0, s1)` of the parent's plan: the trellis rows of those splits are contiguous, so the kernel runs
//!   on them with the plan `(s1 - s0, wk)` and the input columns of the range; its partials are the parent's partials `s0 .. s1 - 1`
//!   ([`Exl3Weights::forward_split_partials`], also when the range is one split). Partials of every range gathered into `[sk][m][N]` and
//!   finished by [`Exl3Weights::finish_partials`] (`rot_out` over `sk`, in split order) are the parent's output. A rank that owns the heads
//!   of one K range (GLM's KDA / MLA `o_proj`: 32 heads a rank = 4 of the 8 splits) computes only its own partials from its own heads.
//!
//! The big (prefill) path ([`Exl3Weights::forward_big`]) adds the splits inside one block in the same order; its strip view takes the
//! same plan. Tests: `crates/cuda/tests/exl3_split.rs` (every GLM shape: strips == the full layer's columns, split partials + finish ==
//! the full forward, on the decode path at 1-128 rows and on the big path up to 2,048 rows).
//!
//! The strip ranges are plain `[lo, hi)` ranges of 128-column strips, the same unit a column-split GEMM of another arithmetic contract
//! takes (lane P's P2 prefill GEMMs can take the same ranges later).

use crate::exl3_linear::{decode_plan, strip_trellis_plan, strip_vector_plan, Dtype, Exl3Weights};
use crate::rt::{CudaError, DevBuf, Stream};

/// `(sk, wk)`: K splits and warps a block of the decode kernel.
pub type Plan = (usize, usize);

#[cfg(not(gb_no_cuda))]
mod ffi {
    use core::ffi::c_void;
    extern "C" {
        pub fn gb_exl3_linear_ex_workspace_bytes(m: i32, k: i32, n: i32, sk: i32, partials: i32) -> usize;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_exl3_linear_ex(
            x: *const c_void,
            x_dtype: i32,
            trellis: *const c_void,
            stride_n: i32,
            suh: *const c_void,
            svh: *const c_void,
            bias: *const c_void,
            y: *mut c_void,
            y_dtype: i32,
            m: i32,
            k: i32,
            n: i32,
            bits: i32,
            codebook: i32,
            sk: i32,
            wk: i32,
            z: *mut c_void,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_exl3_rotate_out_sk(
            c: *const c_void,
            svh: *const c_void,
            bias: *const c_void,
            y: *mut c_void,
            y_dtype: i32,
            m: i32,
            n: i32,
            sk: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_exl3_linear_big_ex(
            x: *const c_void,
            x_dtype: i32,
            trellis: *const c_void,
            stride_n: i32,
            suh: *const c_void,
            svh: *const c_void,
            bias: *const c_void,
            y: *mut c_void,
            y_dtype: i32,
            m: i32,
            k: i32,
            n: i32,
            bits: i32,
            codebook: i32,
            sk: i32,
            wk: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
    }
}

const INVALID: CudaError = CudaError(1);
const MAX_DECODE_ROWS: usize = 128;

#[cfg(not(gb_no_cuda))]
fn codebook_code(cb: gb_exl3::Codebook) -> i32 {
    match cb {
        gb_exl3::Codebook::Inst3 => 0,
        gb_exl3::Codebook::Mcg => 1,
        gb_exl3::Codebook::Mul1 => 2,
    }
}

/// Bytes of workspace of one launch of the cut kernel: the rotated input (`m` rows of `k`), plus the split partials when `sk > 1` and
/// they stay in the workspace (`partials == false`: the launch finishes into `y`).
pub fn ex_workspace_bytes(m: usize, k: usize, n: usize, sk: usize, partials: bool) -> usize {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        ffi::gb_exl3_linear_ex_workspace_bytes(m as i32, k as i32, n as i32, sk as i32, partials as i32)
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (m, k, n, sk, partials);
        0
    }
}

fn check_strips(n: usize, s0: usize, s1: usize) -> Result<(), CudaError> {
    if s1 <= s0 || s1 > n / 128 {
        return Err(INVALID);
    }
    Ok(())
}

impl Exl3Weights {
    /// The plan this layer's decode (and big) kernel reduces with: `plan_for(K, N)`.
    pub fn plan(&self) -> Result<Plan, CudaError> {
        decode_plan(self.k, self.n)
    }

    /// Workspace of [`forward_strip`](Self::forward_strip) over `strips` strips at `m` rows.
    pub fn strip_workspace_bytes(&self, strips: usize, m: usize) -> usize {
        let sk = self.plan().map_or(1, |p| p.0);
        ex_workspace_bytes(m, self.k, strips * 128, sk, false)
    }

    /// S1, decode path: y `[m, (s1 - s0) * 128]` = the columns of 128-column strips `[s0, s1)` of this layer, read in place, reduced with
    /// this layer's plan: bit for bit the full layer's columns, at any strip count and row count (1 ..= 128).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_strip(
        &self,
        s0: usize,
        s1: usize,
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        x_dtype: Dtype,
        y_dtype: Dtype,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        check_strips(self.n, s0, s1)?;
        let n = (s1 - s0) * 128;
        if m == 0
            || m > MAX_DECODE_ROWS
            || x.bytes() < m * self.k * x_dtype.size()
            || y.bytes() < m * n * y_dtype.size()
            || ws.bytes() < self.strip_workspace_bytes(s1 - s0, m)
        {
            return Err(INVALID);
        }
        let (sk, wk) = self.plan()?;
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_exl3_linear_ex(
                x.ptr(),
                x_dtype as i32,
                self.trellis.ptr_at(s0 * 8 * 8 * self.bits * 4),
                self.n as i32,
                self.suh.ptr(),
                self.svh.ptr_at(s0 * 128 * 2),
                self.bias.as_ref().map_or(core::ptr::null(), |b| b.ptr_at(s0 * 128 * 2) as *const _),
                y.ptr(),
                y_dtype as i32,
                m as i32,
                self.k as i32,
                n as i32,
                self.bits as i32,
                codebook_code(self.codebook),
                sk as i32,
                wk as i32,
                core::ptr::null_mut(),
                ws.ptr(),
                ws.bytes(),
                st.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (sk, wk, st);
            Err(CudaError(-1))
        }
    }

    /// S1, big path: [`forward_strip`](Self::forward_strip) for any row count (a prefill chunk), the strips read in place with this
    /// layer's plan; `ws` needs `big_workspace_bytes(m)` (the rotated input only).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_big_strip(
        &self,
        s0: usize,
        s1: usize,
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        x_dtype: Dtype,
        y_dtype: Dtype,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        check_strips(self.n, s0, s1)?;
        let n = (s1 - s0) * 128;
        if m == 0
            || m > i32::MAX as usize / 128
            || x.bytes() < m * self.k * x_dtype.size()
            || y.bytes() < m * n * y_dtype.size()
            || ws.bytes() < self.big_workspace_bytes(m)
        {
            return Err(INVALID);
        }
        let (sk, wk) = self.plan()?;
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_exl3_linear_big_ex(
                x.ptr(),
                x_dtype as i32,
                self.trellis.ptr_at(s0 * 8 * 8 * self.bits * 4),
                self.n as i32,
                self.suh.ptr(),
                self.svh.ptr_at(s0 * 128 * 2),
                self.bias.as_ref().map_or(core::ptr::null(), |b| b.ptr_at(s0 * 128 * 2) as *const _),
                y.ptr(),
                y_dtype as i32,
                m as i32,
                self.k as i32,
                n as i32,
                self.bits as i32,
                codebook_code(self.codebook),
                sk as i32,
                wk as i32,
                ws.ptr(),
                ws.bytes(),
                st.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (sk, wk, st);
            Err(CudaError(-1))
        }
    }

    /// The input columns `[k0, k1)` of the K splits `[s0, s1)` of this layer's plan (the columns a rank must hold to compute them).
    pub fn split_k_range(&self, s0: usize, s1: usize) -> Result<(usize, usize), CudaError> {
        let (sk, _) = self.plan()?;
        if s1 <= s0 || s1 > sk {
            return Err(INVALID);
        }
        let per = self.k / sk;
        Ok((s0 * per, s1 * per))
    }

    /// Workspace of [`forward_split_partials`](Self::forward_split_partials) over splits `[s0, s1)` at `m` rows.
    pub fn split_partials_workspace_bytes(&self, s0: usize, s1: usize, m: usize) -> usize {
        match self.split_k_range(s0, s1) {
            Ok((k0, k1)) => ex_workspace_bytes(m, k1 - k0, self.n, s1 - s0, true),
            Err(_) => 0,
        }
    }

    /// S2: the fp32 partials of the K splits `[s0, s1)` of this layer's plan into `z` from byte `z_off`: `[s1 - s0][m][N]`, split `s0` first
    /// (in a buffer laid out `[sk][m][N]`, `z_off = s0 * m * N * 4`). `x` `[m, k1 - k0]` holds the input columns of the range
    /// ([`split_k_range`](Self::split_k_range)), nothing else. Bit for bit the partials the full layer computes for those splits; nothing is
    /// finished (also for a one-split range). 1 ..= 128 rows.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_split_partials(
        &self,
        s0: usize,
        s1: usize,
        x: &DevBuf,
        x_dtype: Dtype,
        z: &mut DevBuf,
        z_off: usize,
        m: usize,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let (k0, k1) = self.split_k_range(s0, s1)?;
        let (_, wk) = self.plan()?;
        let kr = k1 - k0;
        if m == 0
            || m > MAX_DECODE_ROWS
            || x.bytes() < m * kr * x_dtype.size()
            || z.bytes() < z_off + (s1 - s0) * m * self.n * 4
            || z_off % 16 != 0
            || ws.bytes() < self.split_partials_workspace_bytes(s0, s1, m)
        {
            return Err(INVALID);
        }
        // the trellis rows of the range: tile rows are (N / 16) tiles of 16 * bits int16 apart
        let row_bytes = (self.n / 16) * 16 * self.bits * 2;
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_exl3_linear_ex(
                x.ptr(),
                x_dtype as i32,
                self.trellis.ptr_at((k0 / 16) * row_bytes),
                self.n as i32,
                self.suh.ptr_at(k0 * 2),
                self.svh.ptr(),
                core::ptr::null(),
                core::ptr::null_mut(),
                Dtype::F32 as i32,
                m as i32,
                kr as i32,
                self.n as i32,
                self.bits as i32,
                codebook_code(self.codebook),
                (s1 - s0) as i32,
                wk as i32,
                z.ptr_at(z_off),
                ws.ptr(),
                ws.bytes(),
                st.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (row_bytes, wk, st);
            Err(CudaError(-1))
        }
    }

    /// y `[m, N]` from the partials of ALL `sk` splits of this layer's plan, `z` `[sk][m][N]` fp32 (gathered from the ranges): the decode
    /// path's second kernel (`rot_out`: the splits added in split order, then H / sqrt(128), * svh, + bias).
    pub fn finish_partials(&self, z: &DevBuf, m: usize, y: &mut DevBuf, y_dtype: Dtype, st: &Stream) -> Result<(), CudaError> {
        let (sk, _) = self.plan()?;
        if m == 0 || z.bytes() < sk * m * self.n * 4 || y.bytes() < m * self.n * y_dtype.size() {
            return Err(INVALID);
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_exl3_rotate_out_sk(
                z.ptr(),
                self.svh.ptr(),
                self.bias.as_ref().map_or(core::ptr::null(), |b| b.ptr() as *const _),
                y.ptr(),
                y_dtype as i32,
                m as i32,
                self.n as i32,
                sk as i32,
                st.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (sk, st);
            Err(CudaError(-1))
        }
    }

    /// S1, stored: a copy of some 128-column strips (`ranges`: ascending, disjoint, half-open, in strips) that keeps THIS layer's plan, so
    /// its columns are this layer's bit for bit at any strip count (no 64-strip floor, unlike [`select_strips`](Self::select_strips)).
    /// A rank that serves only its heads stores this and frees the parent.
    pub fn select_strips_planned(&self, ranges: &[(usize, usize)]) -> Result<PlannedExl3, CudaError> {
        let kept = cut_strips(self.n, ranges).ok_or(INVALID)?;
        let plan = self.plan()?;
        let st = Stream::new()?;
        let mut trellis = DevBuf::alloc((self.k / 16) * kept * 64 * self.bits * 4)?;
        let mut suh = DevBuf::alloc(self.suh.bytes())?;
        let mut svh = DevBuf::alloc(kept * 128 * 2)?;
        let mut bias = match &self.bias {
            Some(_) => Some(DevBuf::alloc(kept * 128 * 2)?),
            None => None,
        };
        for c in strip_trellis_plan(self.k, self.n, self.bits, ranges) {
            trellis.copy_2d_from(c.dst_off, c.dst_pitch, &self.trellis, c.src_off, c.src_pitch, c.width, c.height, &st)?;
        }
        suh.copy_from(&self.suh, self.suh.bytes(), &st)?;
        for c in strip_vector_plan(ranges) {
            svh.copy_range_from(c.dst_off, &self.svh, c.src_off, c.width, &st)?;
            if let (Some(dst), Some(src)) = (bias.as_mut(), self.bias.as_ref()) {
                dst.copy_range_from(c.dst_off, src, c.src_off, c.width, &st)?;
            }
        }
        st.sync()?;
        let w = Exl3Weights { trellis, suh, svh, bias, k: self.k, n: kept * 128, bits: self.bits, codebook: self.codebook };
        Ok(PlannedExl3 { w, plan })
    }
}

/// The strips a cut of an `n`-column layer keeps: `ranges` of 128-column strips, ascending, disjoint, half-open and non-empty, inside the
/// layer. `None` for anything else.
pub fn cut_strips(n: usize, ranges: &[(usize, usize)]) -> Option<usize> {
    let strips = n / 128;
    let mut at = 0usize;
    for &(lo, hi) in ranges {
        if lo < at || hi <= lo || hi > strips {
            return None;
        }
        at = hi;
    }
    let kept: usize = ranges.iter().map(|&(lo, hi)| hi - lo).sum();
    (n % 128 == 0 && kept > 0).then_some(kept)
}

/// Where a stored cut's trellis comes from in its parent's (`k x n`, `bits` wide): `(parent_offset, cut_offset, bytes)` runs, one per
/// k-tile row (`k / 16` of them) and range, in the parent's byte order, laid out as [`Exl3Weights::select_strips_planned`] lays out its copy
/// (each tile row the ranges' strips one after another). Pure: a loader places a rank's strips straight from the pack with it
/// (lane E, the GLM head split's rank strips), and the copy and the load give the same bytes at the same offsets.
pub fn cut_trellis_runs(k: usize, n: usize, bits: usize, ranges: &[(usize, usize)]) -> Option<Vec<(usize, usize, usize)>> {
    cut_strips(n, ranges)?;
    if k % 16 != 0 {
        return None;
    }
    let mut runs = Vec::with_capacity(k / 16 * ranges.len());
    let plan = strip_trellis_plan(k, n, bits, ranges);
    for row in 0..k / 16 {
        for c in &plan {
            runs.push((c.src_off + row * c.src_pitch, c.dst_off + row * c.dst_pitch, c.width));
        }
    }
    Some(runs)
}

/// [`cut_trellis_runs`] for a per-column fp16 vector of the layer (`svh`, a bias): one run a range.
pub fn cut_vector_runs(n: usize, ranges: &[(usize, usize)]) -> Option<Vec<(usize, usize, usize)>> {
    cut_strips(n, ranges)?;
    Some(strip_vector_plan(ranges).iter().map(|c| (c.src_off, c.dst_off, c.width)).collect())
}

/// A stored cut of a layer with its PARENT's plan (see the module docs): its columns are the parent's bit for bit.
pub struct PlannedExl3 {
    pub w: Exl3Weights,
    /// The parent's `(sk, wk)`.
    pub plan: Plan,
}

impl PlannedExl3 {
    /// A stored cut that is already on the device: `w` holds the kept strips of a `w.k x parent_n` layer laid out by [`cut_trellis_runs`] /
    /// [`cut_vector_runs`] (a rank that loaded only its strips, lane E's GLM rank strips), run with the parent's plan
    /// `plan_for(w.k, parent_n)`, so its columns are the parent's bit for bit, exactly as [`Exl3Weights::select_strips_planned`]'s copy.
    /// `w`'s buffers must be views of a block (the load regions): they are shared, not copied.
    pub fn of_loaded_cut(w: &Exl3Weights, parent_n: usize) -> Result<Self, CudaError> {
        if parent_n % 128 != 0 || w.n % 128 != 0 || w.n > parent_n || w.n == 0 {
            return Err(INVALID);
        }
        let share = |b: &DevBuf| -> Result<DevBuf, CudaError> {
            b.block().ok_or(INVALID)?;
            Ok(b.subview(0, b.bytes()))
        };
        let shared = Exl3Weights {
            trellis: share(&w.trellis)?,
            suh: share(&w.suh)?,
            svh: share(&w.svh)?,
            bias: match &w.bias {
                Some(b) => Some(share(b)?),
                None => None,
            },
            k: w.k,
            n: w.n,
            bits: w.bits,
            codebook: w.codebook,
        };
        Ok(Self { w: shared, plan: decode_plan(w.k, parent_n)? })
    }
}

impl PlannedExl3 {
    /// Decode workspace for `m` rows.
    pub fn workspace_bytes(&self, m: usize) -> usize {
        ex_workspace_bytes(m, self.w.k, self.w.n, self.plan.0, false)
    }

    /// Big-path workspace for `m` rows (the rotated input).
    pub fn big_workspace_bytes(&self, m: usize) -> usize {
        self.w.big_workspace_bytes(m)
    }

    /// Decode path (1 ..= 128 rows) with the parent's plan: y `[m, n]` = the parent's columns of the kept strips.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        x_dtype: Dtype,
        y_dtype: Dtype,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let w = &self.w;
        if m == 0
            || m > MAX_DECODE_ROWS
            || x.bytes() < m * w.k * x_dtype.size()
            || y.bytes() < m * w.n * y_dtype.size()
            || ws.bytes() < self.workspace_bytes(m)
        {
            return Err(INVALID);
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_exl3_linear_ex(
                x.ptr(),
                x_dtype as i32,
                w.trellis.ptr(),
                w.n as i32,
                w.suh.ptr(),
                w.svh.ptr(),
                w.bias.as_ref().map_or(core::ptr::null(), |b| b.ptr() as *const _),
                y.ptr(),
                y_dtype as i32,
                m as i32,
                w.k as i32,
                w.n as i32,
                w.bits as i32,
                codebook_code(w.codebook),
                self.plan.0 as i32,
                self.plan.1 as i32,
                core::ptr::null_mut(),
                ws.ptr(),
                ws.bytes(),
                st.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = st;
            Err(CudaError(-1))
        }
    }

    /// Big path (any row count) with the parent's plan.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_big(
        &self,
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        x_dtype: Dtype,
        y_dtype: Dtype,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let w = &self.w;
        if m == 0
            || m > i32::MAX as usize / 128
            || x.bytes() < m * w.k * x_dtype.size()
            || y.bytes() < m * w.n * y_dtype.size()
            || ws.bytes() < self.big_workspace_bytes(m)
        {
            return Err(INVALID);
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_exl3_linear_big_ex(
                x.ptr(),
                x_dtype as i32,
                w.trellis.ptr(),
                w.n as i32,
                w.suh.ptr(),
                w.svh.ptr(),
                w.bias.as_ref().map_or(core::ptr::null(), |b| b.ptr() as *const _),
                y.ptr(),
                y_dtype as i32,
                m as i32,
                w.k as i32,
                w.n as i32,
                w.bits as i32,
                codebook_code(w.codebook),
                self.plan.0 as i32,
                self.plan.1 as i32,
                ws.ptr(),
                ws.bytes(),
                st.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = st;
            Err(CudaError(-1))
        }
    }
}
