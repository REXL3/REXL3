//! Tensor-core prefill path of the Nemotron 3.5 Super VL latent-MoE block under contract P2 (`csrc/nemotron_moe_mma.cu`,
//! `csrc/exl3_mma.cuh`; lane S2, `docs/NEMOTRON_PREFILL_MMA.md`). **Opt-in, default off**: nothing calls it unless the caller asks
//! ([`enabled`] reads `REXL3_NEMO_PREFILL_MMA=1` or [`set_enabled`]); the default prefill path (`forward_prefill_in`, fp32 FMA chains
//! with the decode path's bits) and the decode path are unchanged.
//!
//! Contract P2 (prefill only): a row's output depends on that row alone (not on the other rows, the slice width or boundary, the
//! padding, the order of rows); the k reduction order is a fixed function of K (so a split of the output columns, a TP rank's, is
//! exact); activations enter the MMA as fp16 (bf16 only for the quality comparison, [`set_bf16`]), weights are the EXL3 trellis values
//! exactly as the decode path decodes them, accumulation is fp32; deterministic run to run. The bits differ from the decode path's.
//!
//! [`MmaScratch`] holds the ScaledRows buffers (fp16 values with a power-of-two factor per (row, 128-k block)); the
//! [`PrefillScratch`] of the default path is used as well (router, lists, the slow experts' fp32 activations, the routed output `y`).
//! This module is a child of `nemotron_moe_prefill` so that it reads the [`PrefillMoe`] tables without widening their visibility.

use super::*;
use crate::nemotron_moe::{EXPERTS, EXPERT_INTER, HIDDEN, LATENT, SHARED_INTER, TOPK};

mod ffi_mma {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_nmm_smem_bytes(mt: i32, k: i32) -> i32;
        pub fn gb_nmm_dense_h16(
            a16: *const c_void,
            ainv: *const c_void,
            rows: i32,
            mt: i32,
            tab: *const c_void,
            k: i32,
            n: i32,
            out: *mut c_void,
            relu2: i32,
            add: *const c_void,
            add2: *const c_void,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_nmm_dense_h16_smem(mt: i32, k: i32) -> i32;
        pub fn gb_nmm_plain_rows(src: *const c_void, ksrc: i32, rows: i32, k: i32, out16: *mut c_void, oinv: *mut c_void, s: *mut c_void) -> i32;
        pub fn gb_nmm_router(h: *const c_void, gate: *const c_void, logits: *mut c_void, rows: i32, k: i32, s: *mut c_void) -> i32;
        pub fn gb_nmm_gemm_add2(
            add2: *const c_void,
            a16: *const c_void,
            ainv: *const c_void,
            lists: *const c_void,
            lcap: i32,
            work: *const c_void,
            nwork: *const c_void,
            dense_rows: i32,
            zitems: i32,
            mt: i32,
            tab: *const c_void,
            ntab: *const c_void,
            k: i32,
            n: i32,
            nb0: i32,
            ncb: i32,
            out: *mut c_void,
            out16: *mut c_void,
            oinv: *mut c_void,
            relu2: i32,
            add: *const c_void,
            bf: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_nmm_norm_rot(
            x: *const c_void,
            w: *const c_void,
            eps: f32,
            xn: *mut c_void,
            tab1: *const c_void,
            tab2: *const c_void,
            o1: *mut c_void,
            i1: *mut c_void,
            o2: *mut c_void,
            i2: *mut c_void,
            rows: i32,
            bf: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_nmm_router_f32(h: *const c_void, gate: *const c_void, logits: *mut c_void, rows: i32, s: *mut c_void) -> i32;

        pub fn gb_nmm_router_f32_g32(h: *const c_void, gate: *const c_void, logits: *mut c_void, rows: i32, s: *mut c_void) -> i32;

        pub fn gb_nmm_router_g32(h: *const c_void, gate: *const c_void, logits: *mut c_void, rows: i32, k: i32, s: *mut c_void) -> i32;

        pub fn gb_nmm_router_f32_g16(h: *const c_void, gate: *const c_void, logits: *mut c_void, rows: i32, s: *mut c_void) -> i32;

        pub fn gb_nmm_router_g16(h: *const c_void, gate: *const c_void, logits: *mut c_void, rows: i32, k: i32, s: *mut c_void) -> i32;
        pub fn gb_nmm_items(cnt: *const c_void, tab_up: *const c_void, tab_down: *const c_void, cap: i32, work: *mut c_void, nwork: *mut c_void, s: *mut c_void) -> i32;
        pub fn gb_nmm_items_k(cnt: *const c_void, tab_up: *const c_void, tab_down: *const c_void, cap: i32, work: *mut c_void, nwork: *mut c_void, s: *mut c_void) -> i32;
        pub fn gb_nmm_rotate_k(src: *const c_void, ksrc: i32, div: i32, pick: *const c_void, tab: *const c_void, rows: i32, k: i32, out16: *mut c_void, oinv: *mut c_void, s: *mut c_void) -> i32;
        pub fn gb_nmm_gemm_k(
            a16: *const c_void,
            ainv: *const c_void,
            lists: *const c_void,
            lcap: i32,
            work: *const c_void,
            nwork: *const c_void,
            dense_rows: i32,
            zitems: i32,
            mt: i32,
            tab: *const c_void,
            ntab: *const c_void,
            k: i32,
            n: i32,
            nb0: i32,
            ncb: i32,
            out: *mut c_void,
            out16: *mut c_void,
            oinv: *mut c_void,
            relu2: i32,
            add: *const c_void,
            k2: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_nmm_rotate(src: *const c_void, ksrc: i32, div: i32, pick: *const c_void, tab: *const c_void, rows: i32, k: i32, out16: *mut c_void, oinv: *mut c_void, bf: i32, s: *mut c_void) -> i32;
        pub fn gb_nmm_gemm(
            a16: *const c_void,
            ainv: *const c_void,
            lists: *const c_void,
            lcap: i32,
            work: *const c_void,
            nwork: *const c_void,
            dense_rows: i32,
            zitems: i32,
            mt: i32,
            tab: *const c_void,
            ntab: *const c_void,
            k: i32,
            n: i32,
            nb0: i32,
            ncb: i32,
            out: *mut c_void,
            out16: *mut c_void,
            oinv: *mut c_void,
            relu2: i32,
            add: *const c_void,
            bf: i32,
            s: *mut c_void,
        ) -> i32;
    }
}

static ENABLED: AtomicU8 = AtomicU8::new(2); // 2: not yet read from the environment
static TILE: AtomicU8 = AtomicU8::new(DEFAULT_TILE);
static BF16: AtomicBool = AtomicBool::new(false);
static DENSE: AtomicBool = AtomicBool::new(true);
/// An fp16 dense projection (the MixedK pack's fc1 / fc2) on tensor cores under P2 (`nmm_dense_h16`; lane K, CP2), or on the fp32 chain
/// (`pf_slow4`). ON by default; fp16 operands only (with `set_bf16(true)` such a projection keeps the fp32 chain).
static FP16_DENSE: AtomicBool = AtomicBool::new(true);

/// Whether an fp16 dense projection of a block runs on tensor cores under P2 (`nmm_dense_h16`, the default) or on the fp32 chain. The
/// bits of a layer with fp16 dense projections depend on it; our pack has none. Process-wide, not for use while a block is running.
pub fn set_fp16_dense(on: bool) {
    FP16_DENSE.store(on, Ordering::Relaxed);
}

/// [`set_fp16_dense`]'s state.
pub fn fp16_dense() -> bool {
    FP16_DENSE.load(Ordering::Relaxed)
}
static ROUTER: AtomicBool = AtomicBool::new(false);

/// m16 tiles an item carries (16 MT rows): 4 (64 rows, 126 registers, 2 blocks an SM) is the measured best
/// (docs/NEMOTRON_PREFILL_MMA.md, the tile sweep).
pub const DEFAULT_TILE: u8 = 4;

/// Whether the caller should run the tensor-core path: [`set_enabled`], else `REXL3_NEMO_PREFILL_MMA=1` (read once). Default off.
pub fn enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = std::env::var("REXL3_NEMO_PREFILL_MMA").map(|v| v == "1").unwrap_or(false);
            ENABLED.store(on as u8, Ordering::Relaxed);
            on
        }
    }
}

/// Turn the tensor-core path on or off for callers that consult [`enabled`]. Process-wide.
pub fn set_enabled(on: bool) {
    ENABLED.store(on as u8, Ordering::Relaxed);
}

/// The item tile: 1, 2, 4 or 8 m16 tiles (16 to 128 rows of one expert a block). Bits do not depend on it. Process-wide.
pub fn set_tile(mt: u8) {
    assert!(matches!(mt, 1 | 2 | 4 | 8), "tile {mt}");
    TILE.store(mt, Ordering::Relaxed);
}

/// bf16 operands (the weights' fp16 values rounded to bf16) instead of fp16: built for the quality comparison of contract P2 (c) only.
pub fn set_bf16(on: bool) {
    BF16.store(on, Ordering::Relaxed);
}

/// Run the four dense projections of the block (fc1, fc2, shared up and down) on tensor cores too (the default of this path), or on
/// the default path's fp32 kernels (for measurement). A projection that is not 4-bit Mul1 always takes the default path's kernel.
pub fn set_dense(on: bool) {
    DENSE.store(on, Ordering::Relaxed);
}

/// Router logits on tensor cores (bf16 gate, h split in two bf16 terms, fp32 sums; 1.9 -> 0.8 ms a layer at 2,048 rows) or on the
/// decode path's fp32 kernel (the default: the picks are then the default path's exactly; on tensor cores 1 to 5 rows in 1,024 of
/// real layer inputs pick a different top-22 at near ties). Process-wide.
pub fn set_router_mma(on: bool) {
    ROUTER.store(on, Ordering::Relaxed);
}

/// The fp32 router logits of a prefill block from the shared-memory tiled kernel (`gb_nmm_router_f32`, the bits of
/// `gb_nm2_router_logits`): OFF unless `REXL3_NEMO_P2_ROUTER_TILED=1` (read once; lane S5). MEASURED bit-equal but slower than the
/// decode path's kernel (4.65 vs 3.33 ms at 4,096 rows), so not the default.
fn router_tiled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REXL3_NEMO_P2_ROUTER_TILED").is_ok_and(|v| v.trim() == "1"))
}

fn tile() -> usize {
    TILE.load(Ordering::Relaxed) as usize
}

fn bf() -> i32 {
    BF16.load(Ordering::Relaxed) as i32
}

/// The item stride of [`MmaScratch`]: experts with rows plus `ceil(P / 16)` (the smallest item cap) items.
fn mma_item_stride(rows: usize) -> usize {
    EXPERTS + (rows * TOPK).div_ceil(16)
}

/// Dynamic shared memory of one GEMM block for tile `mt` and `k` inputs (bytes).
pub fn smem_bytes(mt: usize, k: usize) -> usize {
    unsafe { ffi_mma::gb_nmm_smem_bytes(mt as i32, k as i32) as usize }
}

/// The tensor-core path's scratch: ScaledRows buffers (16-bit values and fp32 factors per 128 values) and the work items.
pub struct MmaScratch {
    max_rows: usize,
    items: DevBuf,
    nitems: DevBuf,
    /// the routed up projection's input, a row a pair `[P, 1024]`
    a16: DevBuf,
    a_inv: DevBuf,
    /// the routed down projection's input `[P, 2688]`
    act16: DevBuf,
    act_inv: DevBuf,
    /// fc1's input (`h` rotated), then fc2's (`routed` rotated)
    x16: DevBuf,
    x_inv: DevBuf,
    /// the shared up projection's input (side stream)
    hs16: DevBuf,
    hs_inv: DevBuf,
    /// the shared down projection's input (side stream)
    sh16: DevBuf,
    sh_inv: DevBuf,
    /// Rows whose fc1 and shared-up inputs ([`PrefillMoe::norm_rotate_prefill`]) are already in `x16` / `hs16` for the next block call
    /// (0: none); the block takes and clears it.
    pre_rows: usize,
}

impl MmaScratch {
    /// Bytes of each buffer, in field order (after max_rows), for `max_rows` rows.
    fn lens(max_rows: usize) -> [usize; 12] {
        let r = max_rows;
        let p = r * TOPK;
        [
            4 * 3 * mma_item_stride(r),
            16,
            2 * p * LATENT,
            4 * p * (LATENT / 128),
            2 * p * EXPERT_INTER,
            4 * p * (EXPERT_INTER / 128),
            2 * r * HIDDEN,
            4 * r * (HIDDEN / 128),
            2 * r * HIDDEN,
            4 * r * (HIDDEN / 128),
            2 * r * SHARED_INTER,
            4 * r * (SHARED_INTER / 128),
        ]
    }

    /// The device bytes `new(max_rows)` allocates (on top of the [`PrefillScratch`] the path also uses).
    pub fn bytes(max_rows: usize) -> u64 {
        Self::lens(max_rows).iter().map(|n| *n as u64).sum()
    }

    pub fn new(max_rows: usize) -> Result<Self, CudaError> {
        assert!(max_rows >= 1, "max_rows must be at least 1");
        let l = Self::lens(max_rows);
        let f = |n: usize| DevBuf::zeroed(n);
        Ok(Self {
            max_rows,
            items: f(l[0])?,
            nitems: f(l[1])?,
            a16: f(l[2])?,
            a_inv: f(l[3])?,
            act16: f(l[4])?,
            act_inv: f(l[5])?,
            x16: f(l[6])?,
            x_inv: f(l[7])?,
            hs16: f(l[8])?,
            hs_inv: f(l[9])?,
            sh16: f(l[10])?,
            sh_inv: f(l[11])?,
            pre_rows: 0,
        })
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// Forgets inputs a norm made for the next block call (`PrefillMoe::norm_rotate_prefill`): a caller that will not run the block on them.
    pub fn clear_pre(&mut self) {
        self.pre_rows = 0;
    }

    /// Fills every buffer with a NaN pattern, queued on `st` (tests).
    pub fn poison(&mut self, st: &Stream) -> Result<(), CudaError> {
        for b in [
            &mut self.items,
            &mut self.nitems,
            &mut self.a16,
            &mut self.a_inv,
            &mut self.act16,
            &mut self.act_inv,
            &mut self.x16,
            &mut self.x_inv,
            &mut self.hs16,
            &mut self.hs_inv,
            &mut self.sh16,
            &mut self.sh_inv,
        ] {
            b.memset_async(0xFF, st)?;
        }
        Ok(())
    }

    /// The work items of the last routed call (expert, first list position, rows). Synchronises.
    pub fn download_items(&self, st: &Stream) -> Result<Vec<[i32; 3]>, CudaError> {
        st.sync()?;
        let mut n = [0i32; 4];
        self.nitems.download_range(0, &mut n)?;
        let mut all = vec![0i32; 3 * n[0] as usize];
        self.items.download_range(0, &mut all)?;
        Ok(all.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
    }
}

/// A device table of [`Proj`] entries in the kernels' layout, for the per-projection tests and benchmarks.
pub struct ProjTable(DevBuf);

impl ProjTable {
    pub fn new(projs: &[&Proj]) -> Result<Self, CudaError> {
        let t: Vec<PfProj> = projs.iter().map(|p| pf_proj(p)).collect();
        Ok(Self(DevBuf::from_slice(&t[..])?))
    }
    /// Entry `i`'s device address.
    pub fn entry(&self, i: usize) -> *const c_void {
        (self.0.ptr() as usize + i * PROJ_BYTES) as *const c_void
    }
}

/// The kernels one at a time, for tests and benchmarks: raw device pointers, launched on the stream given, no synchronisation.
pub mod raw {
    use super::*;

    /// The input side of `rows` rows to ScaledRows (`out16` `[rows, k]` 16-bit, `oinv` `[rows, k / 128]` f32).
    ///
    /// # Safety
    /// Device pointers sized for `rows` rows (source row `row / div`), table entries for every pick.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rotate(
        src: *const c_void,
        ksrc: usize,
        div: usize,
        pick: *const c_void,
        tab: *const c_void,
        rows: usize,
        k: usize,
        out16: *mut c_void,
        oinv: *mut c_void,
        bf: bool,
        st: &Stream,
    ) -> Result<(), CudaError> {
        CudaError::check(ffi_mma::gb_nmm_rotate(
            src,
            ksrc as i32,
            div as i32,
            pick,
            tab,
            rows as i32,
            k as i32,
            out16,
            oinv,
            bf as i32,
            st.raw(),
        ))
    }

    /// A dense projection (table entry `tab`) of `rows` ScaledRows rows on tensor cores, column blocks `nb0 .. nb0 + ncb`: fp32 `out`
    /// `[rows, n]` (ntab null), or the next projection's ScaledRows input.
    ///
    /// # Safety
    /// Device pointers sized for `rows` rows.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_dense(
        a16: *const c_void,
        ainv: *const c_void,
        rows: usize,
        mt: usize,
        tab: *const c_void,
        ntab: *const c_void,
        k: usize,
        n: usize,
        nb0: usize,
        ncb: usize,
        out: *mut c_void,
        out16: *mut c_void,
        oinv: *mut c_void,
        relu2: bool,
        add: *const c_void,
        bf: bool,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let z = rows.div_ceil(16 * mt) as i32;
        CudaError::check(ffi_mma::gb_nmm_gemm(
            a16,
            ainv,
            core::ptr::null(),
            0,
            core::ptr::null(),
            core::ptr::null(),
            rows as i32,
            z,
            mt as i32,
            tab,
            ntab,
            k as i32,
            n as i32,
            nb0 as i32,
            ncb as i32,
            out,
            out16,
            oinv,
            relu2 as i32,
            add,
            bf as i32,
            st.raw(),
        ))
    }

    /// Router logits `[rows, 512]` of `h` `[rows, k]` f32 against `gate` `[512, k]` bf16 on tensor cores.
    ///
    /// # Safety
    /// Device pointers sized for `rows` rows.
    pub unsafe fn router(
        h: *const c_void,
        gate: *const c_void,
        logits: *mut c_void,
        rows: usize,
        k: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        CudaError::check(ffi_mma::gb_nmm_router(h, gate, logits, rows as i32, k as i32, st.raw()))
    }

    /// [`Self::router`] for an f32 gate `[512, k]` (an F16 gate widened): h and the gate both split in two bf16 terms.
    ///
    /// # Safety
    /// Device pointers sized for `rows` rows.
    pub unsafe fn router_g32(
        h: *const c_void,
        gate: *const c_void,
        logits: *mut c_void,
        rows: usize,
        k: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        CudaError::check(ffi_mma::gb_nmm_router_g32(h, gate, logits, rows as i32, k as i32, st.raw()))
    }

    /// [`Self::router_g32`] for an F16 gate `[512, k]` as stored (fp16 bits, widened exactly before the split): the same bits.
    ///
    /// # Safety
    /// Device pointers sized for `rows` rows.
    pub unsafe fn router_g16(
        h: *const c_void,
        gate: *const c_void,
        logits: *mut c_void,
        rows: usize,
        k: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        CudaError::check(ffi_mma::gb_nmm_router_g16(h, gate, logits, rows as i32, k as i32, st.raw()))
    }

    /// The shared-memory tiled fp32 router (the decode path's bits) for a gate of form `dt`.
    ///
    /// # Safety
    /// Device pointers sized for `rows` rows (k = 4096).
    pub unsafe fn router_tiled(
        h: *const c_void,
        gate: *const c_void,
        dt: crate::nemotron_moe::GateDtype,
        logits: *mut c_void,
        rows: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        CudaError::check(match dt {
            crate::nemotron_moe::GateDtype::Bf16 => ffi_mma::gb_nmm_router_f32(h, gate, logits, rows as i32, st.raw()),
            crate::nemotron_moe::GateDtype::F16 => ffi_mma::gb_nmm_router_f32_g16(h, gate, logits, rows as i32, st.raw()),
            crate::nemotron_moe::GateDtype::F32 => ffi_mma::gb_nmm_router_f32_g32(h, gate, logits, rows as i32, st.raw()),
        })
    }

    /// Plain fp32 rows to ScaledRows without rotation (`out16` `[rows, k]` fp16, `oinv` `[rows, k / 128]`): an fp16 dense projection's
    /// input side.
    ///
    /// # Safety
    /// Device pointers sized for `rows` rows (`ksrc` floats apart).
    pub unsafe fn plain_rows(
        src: *const c_void,
        ksrc: usize,
        rows: usize,
        k: usize,
        out16: *mut c_void,
        oinv: *mut c_void,
        st: &Stream,
    ) -> Result<(), CudaError> {
        CudaError::check(ffi_mma::gb_nmm_plain_rows(src, ksrc as i32, rows as i32, k as i32, out16, oinv, st.raw()))
    }

    /// An fp16 dense projection (table entry `tab`, `Proj::F16`) of `rows` ScaledRows rows on tensor cores (`nmm_dense_h16`): fp32
    /// `out` `[rows, n]` = relu2, then + `add`, then + `add2` (either may be null; `add2` may be `out`).
    ///
    /// # Safety
    /// Device pointers sized for `rows` rows.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dense_h16(
        a16: *const c_void,
        ainv: *const c_void,
        rows: usize,
        mt: usize,
        tab: *const c_void,
        k: usize,
        n: usize,
        out: *mut c_void,
        relu2: bool,
        add: *const c_void,
        add2: *const c_void,
        st: &Stream,
    ) -> Result<(), CudaError> {
        CudaError::check(ffi_mma::gb_nmm_dense_h16(
            a16,
            ainv,
            rows as i32,
            mt as i32,
            tab,
            k as i32,
            n as i32,
            out,
            relu2 as i32,
            add,
            add2,
            st.raw(),
        ))
    }

    /// `nmm_dense_h16`'s dynamic shared memory at tile `mt` and `k` inputs (bytes; above 101,376 the launch is refused).
    pub fn dense_h16_smem(mt: usize, k: usize) -> usize {
        unsafe { ffi_mma::gb_nmm_dense_h16_smem(mt as i32, k as i32) as usize }
    }

    /// A plain (bf16 or fp16) dense projection on the fp32 chain (`pf_slow`, variant 0, or `pf_slow4`, 1): the decode path's bits.
    ///
    /// # Safety
    /// Device pointers sized for `rows` rows.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn slow_dense(
        x: *const c_void,
        rows: usize,
        tab: *const c_void,
        k: usize,
        n: usize,
        out: *mut c_void,
        relu2: bool,
        variant: u8,
        st: &Stream,
    ) -> Result<(), CudaError> {
        CudaError::check(ffi::gb_nmp_slow(
            x,
            k as i32,
            1,
            core::ptr::null(),
            0,
            core::ptr::null(),
            core::ptr::null(),
            rows as i32,
            rows.div_ceil(BLOCK_ROWS) as i32,
            tab,
            k as i32,
            n as i32,
            out,
            relu2 as i32,
            core::ptr::null(),
            variant as i32,
            st.raw(),
        ))
    }

    /// The default path's fp32 kernel for one dense 4-bit projection (pf_rotate, then pf_gemm4 over chunks of 32 rows): plain
    /// `x` `[rows, k]` (rotated into `rot`, `[rows, k]` f32) to `out` `[rows, n]`.
    ///
    /// # Safety
    /// Device pointers sized for `rows` rows.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fma_dense(
        x: *const c_void,
        rot: *mut c_void,
        rows: usize,
        tab: *const c_void,
        k: usize,
        n: usize,
        out: *mut c_void,
        relu2: bool,
        st: &Stream,
    ) -> Result<(), CudaError> {
        CudaError::check(ffi::gb_nmp_rotate(x, k as i32, 1, core::ptr::null(), tab, rows as i32, k as i32, rot, st.raw()))?;
        CudaError::check(ffi::gb_nmp_gemm(
            rot,
            k as i32,
            core::ptr::null(),
            0,
            core::ptr::null(),
            core::ptr::null(),
            rows as i32,
            rows.div_ceil(BLOCK_ROWS) as i32,
            BLOCK_ROWS as i32,
            tab,
            core::ptr::null(),
            k as i32,
            n as i32,
            slices(k) as i32,
            out,
            relu2 as i32,
            core::ptr::null(),
            st.raw(),
        ))
    }
}

impl PrefillMoe {
    /// The block on the tensor-core path (contract P2): `out = routed + shared` for `rows` rows of `h` (`[rows, 4096]`, f32). Not
    /// bitwise the decode path (see the module docs); deterministic, and a row's bits do not depend on the other rows. `rows` up to
    /// both scratches' `max_rows`; the layer must `supports_prefill()`.
    pub fn forward_prefill_mma(
        &self,
        s: &mut PrefillScratch,
        m: &mut MmaScratch,
        h: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.forward_mma_reduce(s, m, h, rows, out, st, None, false)
    }

    /// The layer's RMSNorm (x f32 `[rows, 4096]`, bf16 weight `norm_w`) into `xn` and, in the same pass, this block's fc1 and shared-up
    /// ScaledRows inputs (lane S5: the bits of `rmsnorm_bf16w_f32` then the block's two `nmm_rotate` of xn), recorded in `m` so the next
    /// [`Self::forward_prefill_mma`] of `rows` rows on xn skips those rotations. `Ok(false)`: not applicable (a dense projection is not on
    /// tensor cores), nothing written.
    #[allow(clippy::too_many_arguments)]
    pub fn norm_rotate_prefill(
        &self,
        m: &mut MmaScratch,
        x: &DevBuf,
        norm_w: &DevBuf,
        eps: f32,
        rows: usize,
        xn: &mut DevBuf,
        st: &Stream,
    ) -> Result<bool, CudaError> {
        let fast = |i: usize| DENSE.load(Ordering::Relaxed) && self.dense[i] == DenseKind::Fast;
        if !(fast(0) && fast(2)) || rows == 0 || rows > m.max_rows {
            return Ok(false);
        }
        assert!(x.bytes() >= rows * HIDDEN * 4 && xn.bytes() >= rows * HIDDEN * 4 && norm_w.bytes() >= HIDDEN * 2);
        let tabd = |i: usize| (self.tab_dense.ptr() as usize + i * PROJ_BYTES) as *const c_void;
        CudaError::check(unsafe {
            ffi_mma::gb_nmm_norm_rot(
                x.ptr() as *const c_void,
                norm_w.ptr() as *const c_void,
                eps,
                xn.ptr(),
                tabd(0),
                tabd(2),
                m.x16.ptr(),
                m.x_inv.ptr(),
                m.hs16.ptr(),
                m.hs_inv.ptr(),
                rows as i32,
                bf(),
                st.raw(),
            )
        })?;
        m.pre_rows = rows;
        Ok(true)
    }

    /// Whether [`Self::forward_prefill_mma_resid`] can run this layer (its fc2 is a tensor-core projection).
    pub fn supports_prefill_mma_resid(&self) -> bool {
        DENSE.load(Ordering::Relaxed) && (self.dense[1] == DenseKind::Fast || self.h16_on(1))
    }

    /// Whether dense projection `which` is an fp16 one the tensor-core block runs with `nmm_dense_h16` (fp16 operands, the switch on).
    fn h16_on(&self, which: usize) -> bool {
        DENSE.load(Ordering::Relaxed) && bf() == 0 && FP16_DENSE.load(Ordering::Relaxed) && self.dense[which] == DenseKind::F16
    }

    /// [`Self::forward_prefill_mma`] with the residual add fused into fc2's epilogue (lane S5): `x += routed + shared` in place, bit for
    /// bit `forward_prefill_mma` into a buffer `mix` followed by `x = x + mix` (fc2's last two operations are `+ shared` then `+ x`; IEEE
    /// addition commutes). `x` is `[rows, 4096]` f32 and must not be `h`. Needs [`Self::supports_prefill_mma_resid`].
    pub fn forward_prefill_mma_resid(
        &self,
        s: &mut PrefillScratch,
        m: &mut MmaScratch,
        h: &DevBuf,
        rows: usize,
        x: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        assert!(self.supports_prefill_mma_resid(), "forward_prefill_mma_resid needs a tensor-core fc2");
        assert!(h.ptr() != x.ptr(), "the residual stream cannot be the block's input");
        self.forward_mma_reduce(s, m, h, rows, x, st, None, true)
    }

    /// `forward_prefill_mma` for one rank of an expert-parallel group: one call of `reduce` on `y` between the routed down projection
    /// and the combine, as `forward_prefill_in_tp`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_mma_tp(
        &self,
        s: &mut PrefillScratch,
        m: &mut MmaScratch,
        h: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        st: &Stream,
        reduce: &mut ReduceHook,
    ) -> Result<(), CudaError> {
        self.forward_mma_reduce(s, m, h, rows, out, st, Some(reduce), false)
    }

    /// [`Self::forward_prefill_mma_tp`] with the residual add fused into fc2's epilogue (`docs/NEMOTRON_PREFILL_SPEED.md` 6.11): the
    /// expert-parallel twin of [`Self::forward_prefill_mma_resid`]. `reduce` runs on `y` BEFORE the combine, so the combine, fc2 and
    /// `+ sh_out` see the single node's operands on every rank (fc2 is replicated, never split; the shared expert is replicated or
    /// split by output columns with its halves copied in before fc2); the `+ x` is then fc2's last operation on a complete value,
    /// never on a rank's partial sum: `fl(mix + x) == fl(x + mix)`. `x` is `[rows, 4096]` f32 and must not be `h`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_mma_resid_tp(
        &self,
        s: &mut PrefillScratch,
        m: &mut MmaScratch,
        h: &DevBuf,
        rows: usize,
        x: &mut DevBuf,
        st: &Stream,
        reduce: &mut ReduceHook,
    ) -> Result<(), CudaError> {
        assert!(self.supports_prefill_mma_resid(), "forward_prefill_mma_resid_tp needs a tensor-core fc2");
        assert!(h.ptr() != x.ptr(), "the residual stream cannot be the block's input");
        self.forward_mma_reduce(s, m, h, rows, x, st, Some(reduce), true)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_mma_reduce(
        &self,
        s: &mut PrefillScratch,
        m: &mut MmaScratch,
        h: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        st: &Stream,
        reduce: Option<&mut ReduceHook>,
        resid: bool,
    ) -> Result<(), CudaError> {
        assert!(self.supported, "this layer has a projection the prefill path does not run (see supports_prefill)");
        s.check_rows(rows);
        assert!(rows <= m.max_rows, "rows {rows} is outside 1..={}", m.max_rows);
        assert!(h.bytes() >= rows * HIDDEN * 4 && out.bytes() >= rows * HIDDEN * 4, "h or out is short for {rows} rows");
        let p = rows * TOPK;
        let stride = item_stride(rows);
        let mt = tile();
        let bf = bf();
        let dense_mma = DENSE.load(Ordering::Relaxed);
        // fc1's and the shared expert's inputs already made by the layer's norm (`norm_rotate_prefill`) for exactly these rows
        let pre = std::mem::take(&mut m.pre_rows) == rows;
        let (hp, sc) = (h.ptr() as *const c_void, st.raw());
        let p_ = |b: &DevBuf| b.ptr() as *const c_void;
        #[allow(clippy::unnecessary_cast)] // the same type in the stub build only
        let pm = |b: &DevBuf| b.ptr() as *mut c_void;
        let null = core::ptr::null::<c_void>();
        let nullm = core::ptr::null_mut::<c_void>();
        let ok = CudaError::check;
        let tabd = |i: usize| (self.tab_dense.ptr() as usize + i * PROJ_BYTES) as *const c_void;
        let fast = |i: usize| dense_mma && self.dense[i] == DenseKind::Fast;
        // a dense projection on tensor cores from ScaledRows input
        let dense_gemm = |a16: *const c_void,
                          ainv: *const c_void,
                          which: usize,
                          ntab: *const c_void,
                          k: usize,
                          n: usize,
                          out: *mut c_void,
                          out16: *mut c_void,
                          oinv: *mut c_void,
                          relu2: bool,
                          add: *const c_void,
                          raw: *mut c_void|
         -> Result<(), CudaError> {
            ok(unsafe {
                ffi_mma::gb_nmm_gemm(
                    a16,
                    ainv,
                    null,
                    0,
                    null,
                    null,
                    rows as i32,
                    rows.div_ceil(16 * mt) as i32,
                    mt as i32,
                    tabd(which),
                    ntab,
                    k as i32,
                    n as i32,
                    0,
                    (n / 128) as i32,
                    out,
                    out16,
                    oinv,
                    relu2 as i32,
                    add,
                    bf,
                    raw,
                )
            })
        };

        // The shared expert (side stream, or in line).
        let side = SIDE_STREAM.load(Ordering::Relaxed);
        let (hs16, hs_inv, sh16, sh_inv) = (pm(&m.hs16), pm(&m.hs_inv), pm(&m.sh16), pm(&m.sh_inv));
        let shared = |sx: &Stream| -> Result<(), CudaError> {
            let raw = sx.raw();
            if fast(2) && fast(3) {
                if !pre {
                    ok(unsafe {
                        ffi_mma::gb_nmm_rotate(
                            hp,
                            HIDDEN as i32,
                            1,
                            null,
                            tabd(2),
                            rows as i32,
                            HIDDEN as i32,
                            hs16,
                            hs_inv,
                            bf,
                            raw,
                        )
                    })?;
                }
                // lane S, P2 TP: this rank's column blocks of both projections, the rest exchanged (6.9 / 6.13)
                if let Some(rank) = super::shared_split_rank(rows) {
                    let b = SharedBufs {
                        hs: (&m.hs16, &m.hs_inv),
                        sh: (&m.sh16, &m.sh_inv),
                        out: &s.sh_out,
                        send: &s.xh_h2,
                        recv: &s.sh_a,
                    };
                    return self.shared_split_tp(rank, b, rows, sx);
                }
                dense_gemm(hs16, hs_inv, 2, tabd(3), HIDDEN, SHARED_INTER, nullm, sh16, sh_inv, true, null, raw)?;
                dense_gemm(sh16, sh_inv, 3, null, SHARED_INTER, HIDDEN, pm(&s.sh_out), nullm, nullm, false, null, raw)
            } else {
                self.dense_proj(2, hp, false, pm(&s.xh_h2), rows, pm(&s.sh_a), true, null, raw)?;
                // an any-width down not rotated by the up's epilogue rotates in place
                self.dense_proj(3, p_(&s.sh_a), self.shared_down_rotated(), pm(&s.sh_a), rows, pm(&s.sh_out), false, null, raw)
            }
        };
        if side {
            fork(st, &s.aux, &s.ev_fork)?;
            shared(&s.aux)?;
        }

        // Router, selection, lists (and the slow class's items), then this path's items of the fast experts.
        // lane S, P2 TP router row split (6.12): this rank's rows only (the same per-row chain whatever the call's rows, RB or EW),
        // then the byte exchange of the two row ranges; the tensor-core router is never split
        let rsplit = if ROUTER.load(Ordering::Relaxed) { None } else { super::router_split_for(rows) };
        use crate::nemotron_moe::GateDtype;
        if ROUTER.load(Ordering::Relaxed) {
            let tc = match self.gate_dt {
                GateDtype::Bf16 => ffi_mma::gb_nmm_router,
                GateDtype::F16 => ffi_mma::gb_nmm_router_g16,
                GateDtype::F32 => ffi_mma::gb_nmm_router_g32,
            };
            ok(unsafe { tc(hp, self.gate as *const c_void, pm(&s.logits), rows as i32, HIDDEN as i32, sc) })?;
        } else if rows > 12 && router_tiled() {
            // the decode path's fp32 logits, bit for bit, from a shared-memory tiled kernel (lane S5)
            let tiled = match self.gate_dt {
                GateDtype::Bf16 => ffi_mma::gb_nmm_router_f32,
                GateDtype::F16 => ffi_mma::gb_nmm_router_f32_g16,
                GateDtype::F32 => ffi_mma::gb_nmm_router_f32_g32,
            };
            ok(unsafe { tiled(hp, self.gate as *const c_void, pm(&s.logits), rows as i32, sc) })?;
        } else if let Some((own, peer)) = rsplit {
            let n = own.len();
            let ho = (h.ptr() as usize + own.start * HIDDEN * 4) as *const c_void;
            let lo = (s.logits.ptr() as usize + own.start * EXPERTS * 4) as *mut c_void;
            unsafe { crate::nemotron_moe::fast_router_logits(ho, self.gate as *const c_void, self.gate_dt, lo, n, sc) }?;
            super::router_exchange(&mut s.logits, own, peer, st)?;
        } else {
            unsafe {
                crate::nemotron_moe::fast_router_logits(hp, self.gate as *const c_void, self.gate_dt, pm(&s.logits), rows, sc)
            }?;
        }
        ok(unsafe {
            ffi::gb_nmp_select(p_(&s.logits), self.bias as *const c_void, self.scale, pm(&s.pick), pm(&s.wts), rows as i32, sc)
        })?;
        ok(unsafe {
            (if self.kmode { ffi::gb_nmp_bucket_k } else { ffi::gb_nmp_bucket })(
                p_(&s.pick),
                rows as i32,
                rows as i32,
                p_(&self.tab_up),
                p_(&self.tab_down),
                pm(&s.cnt),
                pm(&s.lists),
                pm(&s.work),
                stride as i32,
                pm(&s.nwork),
                1,
                sc,
            )
        })?;
        // a MixedK table's fast experts are the any-width ones (and its launches one a width below)
        let items = if self.kmode { ffi_mma::gb_nmm_items_k } else { ffi_mma::gb_nmm_items };
        ok(unsafe {
            items(p_(&s.cnt), p_(&self.tab_up), p_(&self.tab_down), (16 * mt) as i32, pm(&m.items), pm(&m.nitems), sc)
        })?;
        // lane S, P2 TP: the counts are final here; the pair gather's hook waits for this event, not for the whole stream
        if let Some(ev) = super::after_bucket_event() {
            ev.record(st)?;
        }

        // fc1: the latent x = fc1(h).
        if fast(0) {
            if !pre {
                ok(unsafe {
                    ffi_mma::gb_nmm_rotate(
                        hp,
                        HIDDEN as i32,
                        1,
                        null,
                        tabd(0),
                        rows as i32,
                        HIDDEN as i32,
                        pm(&m.x16),
                        pm(&m.x_inv),
                        bf,
                        sc,
                    )
                })?;
            }
            dense_gemm(p_(&m.x16), p_(&m.x_inv), 0, null, HIDDEN, LATENT, pm(&s.xlat), nullm, nullm, false, null, sc)?;
        } else if self.h16_on(0) {
            // an fp16 fc1 (the MixedK pack): its plain rows to ScaledRows, then the fp16 weight on tensor cores
            ok(unsafe {
                ffi_mma::gb_nmm_plain_rows(hp, HIDDEN as i32, rows as i32, HIDDEN as i32, pm(&m.x16), pm(&m.x_inv), sc)
            })?;
            ok(unsafe {
                ffi_mma::gb_nmm_dense_h16(
                    p_(&m.x16),
                    p_(&m.x_inv),
                    rows as i32,
                    mt.min(4) as i32,
                    tabd(0),
                    HIDDEN as i32,
                    LATENT as i32,
                    pm(&s.xlat),
                    0,
                    null,
                    null,
                    sc,
                )
            })?;
        } else {
            self.dense_proj(0, hp, false, pm(&s.xh_h1), rows, pm(&s.xlat), false, null, sc)?;
        }

        // Routed experts: the fast ones on tensor cores, the slow ones (bf16, absent) on the default path's kernels.
        let z = mma_item_stride(rows).min(EXPERTS.min(p) + p.div_ceil(16 * mt)) as i32;
        let slow_items = (s.work.ptr() as usize + 2 * stride * 3 * 4) as *const c_void;
        let slow_n = (s.nwork.ptr() as usize + 2 * 4) as *const c_void;
        // the slow class's items: an expert with c rows is ceil(c / BLOCK_ROWS) of them (pf_bucket_work), so at most
        // ceil(P / BLOCK_ROWS) + n_slow in all; blocks past the device's count exit (lane S5: a TP rank's many absent or bf16 experts
        // no longer launch n_slow x ceil(rows / BLOCK_ROWS) blocks; the grid's size changes no bit)
        let z_slow = (self.n_slow * rows.div_ceil(BLOCK_ROWS)).min(p.div_ceil(BLOCK_ROWS) + self.n_slow).max(1) as i32;
        let bf16_kernel = BF16_KERNEL.load(Ordering::Relaxed) as i32;
        // a TP rank's absent experts (lane S): with every slow expert absent and the fast path on (the default), no `pf_slow` launch (its
        // grid is n_slow x ceil(rows / 32) items, mostly empty) and one zero kernel after the down projection; the same +0.0 bits
        let zero_only = self.n_slow > 0 && self.n_absent == self.n_slow && super::absent_fast();
        let run_slow = self.n_slow > 0 && !zero_only;
        if self.kmode {
            ok(unsafe {
                ffi_mma::gb_nmm_rotate_k(
                    p_(&s.xlat),
                    LATENT as i32,
                    TOPK as i32,
                    p_(&s.pick),
                    p_(&self.tab_up),
                    p as i32,
                    LATENT as i32,
                    pm(&m.a16),
                    pm(&m.a_inv),
                    sc,
                )
            })?;
        } else {
            ok(unsafe {
                ffi_mma::gb_nmm_rotate(
                    p_(&s.xlat),
                    LATENT as i32,
                    TOPK as i32,
                    p_(&s.pick),
                    p_(&self.tab_up),
                    p as i32,
                    LATENT as i32,
                    pm(&m.a16),
                    pm(&m.a_inv),
                    bf,
                    sc,
                )
            })?;
        }
        // up: one launch (4-bit tables), or one a width the table's up projections hold (each skips the other widths' items)
        for k2 in self.routed_launch_widths(true) {
            ok(unsafe {
                let args = (p_(&m.a16), p_(&m.a_inv), p_(&s.lists), rows as i32, p_(&m.items), p_(&m.nitems), 0, z, mt as i32);
                match k2 {
                    None => ffi_mma::gb_nmm_gemm(
                        args.0,
                        args.1,
                        args.2,
                        args.3,
                        args.4,
                        args.5,
                        args.6,
                        args.7,
                        args.8,
                        p_(&self.tab_up),
                        p_(&self.tab_down),
                        LATENT as i32,
                        EXPERT_INTER as i32,
                        0,
                        (EXPERT_INTER / 128) as i32,
                        nullm,
                        pm(&m.act16),
                        pm(&m.act_inv),
                        1,
                        null,
                        bf,
                        sc,
                    ),
                    Some(k2) => ffi_mma::gb_nmm_gemm_k(
                        args.0,
                        args.1,
                        args.2,
                        args.3,
                        args.4,
                        args.5,
                        args.6,
                        args.7,
                        args.8,
                        p_(&self.tab_up),
                        p_(&self.tab_down),
                        LATENT as i32,
                        EXPERT_INTER as i32,
                        0,
                        (EXPERT_INTER / 128) as i32,
                        nullm,
                        pm(&m.act16),
                        pm(&m.act_inv),
                        1,
                        null,
                        k2,
                        sc,
                    ),
                }
            })?;
        }
        if run_slow {
            ok(unsafe {
                ffi::gb_nmp_slow(
                    p_(&s.xlat),
                    LATENT as i32,
                    TOPK as i32,
                    p_(&s.lists),
                    rows as i32,
                    slow_items,
                    slow_n,
                    0,
                    z_slow,
                    p_(&self.tab_up),
                    LATENT as i32,
                    EXPERT_INTER as i32,
                    pm(&s.act),
                    1,
                    null,
                    bf16_kernel,
                    sc,
                )
            })?;
        }
        for k2 in self.routed_launch_widths(false) {
            ok(unsafe {
                let args =
                    (p_(&m.act16), p_(&m.act_inv), p_(&s.lists), rows as i32, p_(&m.items), p_(&m.nitems), 0, z, mt as i32);
                match k2 {
                    None => ffi_mma::gb_nmm_gemm(
                        args.0,
                        args.1,
                        args.2,
                        args.3,
                        args.4,
                        args.5,
                        args.6,
                        args.7,
                        args.8,
                        p_(&self.tab_down),
                        null,
                        EXPERT_INTER as i32,
                        LATENT as i32,
                        0,
                        (LATENT / 128) as i32,
                        pm(&s.y),
                        nullm,
                        nullm,
                        0,
                        null,
                        bf,
                        sc,
                    ),
                    Some(k2) => ffi_mma::gb_nmm_gemm_k(
                        args.0,
                        args.1,
                        args.2,
                        args.3,
                        args.4,
                        args.5,
                        args.6,
                        args.7,
                        args.8,
                        p_(&self.tab_down),
                        null,
                        EXPERT_INTER as i32,
                        LATENT as i32,
                        0,
                        (LATENT / 128) as i32,
                        pm(&s.y),
                        nullm,
                        nullm,
                        0,
                        null,
                        k2,
                        sc,
                    ),
                }
            })?;
        }
        if zero_only && !super::gather_fills_absent() {
            ok(unsafe {
                ffi::gb_nmp_zero_absent(
                    p_(&s.lists),
                    rows as i32,
                    slow_items,
                    slow_n,
                    p_(&self.tab_down),
                    LATENT as i32,
                    pm(&s.y),
                    super::ZERO_BLOCKS,
                    sc,
                )
            })?;
        }
        if run_slow {
            ok(unsafe {
                ffi::gb_nmp_slow(
                    p_(&s.act),
                    EXPERT_INTER as i32,
                    1,
                    p_(&s.lists),
                    rows as i32,
                    slow_items,
                    slow_n,
                    0,
                    z_slow,
                    p_(&self.tab_down),
                    EXPERT_INTER as i32,
                    LATENT as i32,
                    pm(&s.y),
                    0,
                    null,
                    bf16_kernel,
                    sc,
                )
            })?;
        }
        if let Some(reduce) = reduce {
            reduce(&mut s.y, p * LATENT, st)?;
        }
        // The weighted sum of each token's 22 slots in slot order (the default path's combine).
        match super::combine_from() {
            // lane S, P2 TP: the peer's pair rows straight from the receive buffer (the same arithmetic, no unpack copy)
            Some((recv, map)) => unsafe {
                crate::nemotron_tp::combine_gather(
                    s.y.ptr() as usize,
                    recv,
                    map,
                    s.wts.ptr() as usize,
                    rows,
                    LATENT,
                    s.routed.ptr() as usize,
                    st,
                )?
            },
            None => ok(unsafe { ffi::gb_nm_combine(p_(&s.y), p_(&s.wts), rows as i32, LATENT as i32, pm(&s.routed), sc) })?,
        }

        if side {
            join(st, &s.aux, &s.ev_join)?;
        } else {
            shared(st)?;
        }
        // fc2 back to the hidden width, plus the shared expert's output.
        if fast(1) && resid {
            // out += (fc2 + shared): the residual read and written by the epilogue's thread of each element
            ok(unsafe {
                ffi_mma::gb_nmm_rotate(
                    p_(&s.routed),
                    LATENT as i32,
                    1,
                    null,
                    tabd(1),
                    rows as i32,
                    LATENT as i32,
                    pm(&m.x16),
                    pm(&m.x_inv),
                    bf,
                    sc,
                )
            })?;
            ok(unsafe {
                ffi_mma::gb_nmm_gemm_add2(
                    out.ptr() as *const c_void,
                    p_(&m.x16),
                    p_(&m.x_inv),
                    null,
                    0,
                    null,
                    null,
                    rows as i32,
                    rows.div_ceil(16 * mt) as i32,
                    mt as i32,
                    tabd(1),
                    null,
                    LATENT as i32,
                    HIDDEN as i32,
                    0,
                    (HIDDEN / 128) as i32,
                    out.ptr(),
                    nullm,
                    nullm,
                    0,
                    p_(&s.sh_out),
                    bf,
                    sc,
                )
            })
        } else if fast(1) {
            ok(unsafe {
                ffi_mma::gb_nmm_rotate(
                    p_(&s.routed),
                    LATENT as i32,
                    1,
                    null,
                    tabd(1),
                    rows as i32,
                    LATENT as i32,
                    pm(&m.x16),
                    pm(&m.x_inv),
                    bf,
                    sc,
                )
            })?;
            dense_gemm(p_(&m.x16), p_(&m.x_inv), 1, null, LATENT, HIDDEN, out.ptr(), nullm, nullm, false, p_(&s.sh_out), sc)
        } else if self.h16_on(1) {
            // an fp16 fc2: out = (fc2 + shared), or with the residual fused, (fc2 + shared) + x in place (the fast fc2's order)
            ok(unsafe {
                ffi_mma::gb_nmm_plain_rows(p_(&s.routed), LATENT as i32, rows as i32, LATENT as i32, pm(&m.x16), pm(&m.x_inv), sc)
            })?;
            let add2 = if resid { out.ptr() as *const c_void } else { null };
            ok(unsafe {
                ffi_mma::gb_nmm_dense_h16(
                    p_(&m.x16),
                    p_(&m.x_inv),
                    rows as i32,
                    mt.min(4) as i32,
                    tabd(1),
                    LATENT as i32,
                    HIDDEN as i32,
                    out.ptr(),
                    0,
                    p_(&s.sh_out),
                    add2,
                    sc,
                )
            })
        } else {
            assert!(!resid, "the residual is fused only into a tensor-core fc2 (supports_prefill_mma_resid)");
            self.dense_proj(1, p_(&s.routed), false, pm(&s.xh_h1), rows, out.ptr(), false, p_(&s.sh_out), sc)
        }
    }

    /// Lane S, P2 at two ranks (`docs/NEMOTRON_PREFILL_SPEED.md` 6.9 / 6.13): the shared expert split by OUTPUT column blocks. The
    /// up projection's 128-column blocks of this rank's half (21 of 42) into `sh16`/`sh_inv` (ScaledRows: a factor per (row, own
    /// 128-block), written by the epilogue of that block alone), their bytes exchanged with the peer's; then the down projection's
    /// blocks of this rank's half (16 of 32) into `sh_out` (f32), the columns exchanged. Every byte is the one the single node writes
    /// (S2's GEMM fixes a column block's K order by K alone; the cut is at a multiple of 128): no partial sum anywhere, and `sh_out` is
    /// complete when the block joins the side stream, before fc2 (S5's residual constraint). The exchanges go through
    /// [`super::shared_exchange`] (the second communicator) on `sx`, packed through `send` and `recv` (the block's `xh_h2` and `sh_a`,
    /// which the tensor-core shared expert does not use otherwise).
    #[allow(clippy::unnecessary_cast)] // the pointer casts are the same type only in the stub build
    fn shared_split_tp(&self, rank: usize, b: SharedBufs<'_>, rows: usize, sx: &Stream) -> Result<(), CudaError> {
        super::count_split(false, rows);
        let (mt, bf) = (tile(), bf());
        let tabd = |i: usize| (self.tab_dense.ptr() as usize + i * PROJ_BYTES) as *const c_void;
        let gemm = |a: (&DevBuf, &DevBuf),
                    which: usize,
                    ntab: *const c_void,
                    k: usize,
                    n: usize,
                    cols: &core::ops::Range<usize>,
                    out: *mut c_void,
                    o: Option<(&DevBuf, &DevBuf)>,
                    relu2: bool| {
            let (o16, oinv) = o.map_or((core::ptr::null_mut(), core::ptr::null_mut()), |(x, y)| {
                (x.ptr() as *mut c_void, y.ptr() as *mut c_void)
            });
            CudaError::check(unsafe {
                ffi_mma::gb_nmm_gemm(
                    a.0.ptr() as *const c_void,
                    a.1.ptr() as *const c_void,
                    core::ptr::null(),
                    0,
                    core::ptr::null(),
                    core::ptr::null(),
                    rows as i32,
                    rows.div_ceil(16 * mt) as i32,
                    mt as i32,
                    tabd(which),
                    ntab,
                    k as i32,
                    n as i32,
                    cols.start as i32,
                    cols.len() as i32,
                    out,
                    o16,
                    oinv,
                    relu2 as i32,
                    core::ptr::null(),
                    bf,
                    sx.raw(),
                )
            })
        };
        let (mut send, mut recv) = (b.send.alias(), b.recv.alias());
        // the up projection: values (2 bytes) and factors (4 bytes a 128-block), packed [rows, own values] then [rows, own factors]
        let nu = SHARED_INTER / 128;
        let (own, peer) = super::shared_split_blocks(rank, nu);
        gemm(b.hs, 2, tabd(3), HIDDEN, SHARED_INTER, &own, core::ptr::null_mut(), Some(b.sh), true)?;
        let (vp, fp) = (SHARED_INTER * 2, nu * 4);
        let (ov, of) = (own.len() * 256, own.len() * 4);
        let (pv, pf) = (peer.len() * 256, peer.len() * 4);
        send.copy_2d_from(0, ov, b.sh.0, own.start * 256, vp, ov, rows, sx)?;
        send.copy_2d_from(rows * ov, of, b.sh.1, own.start * 4, fp, of, rows, sx)?;
        super::shared_exchange(&send, rows * (ov + of), &mut recv, rows * (pv + pf), sx)?;
        b.sh.0.alias().copy_2d_from(peer.start * 256, vp, &recv, 0, pv, pv, rows, sx)?;
        b.sh.1.alias().copy_2d_from(peer.start * 4, fp, &recv, rows * pv, pf, pf, rows, sx)?;
        // the down projection: f32 columns
        let (own, peer) = super::shared_split_blocks(rank, HIDDEN / 128);
        gemm(b.sh, 3, core::ptr::null(), SHARED_INTER, HIDDEN, &own, b.out.ptr() as *mut c_void, None, false)?;
        let op = HIDDEN * 4;
        let (ow, pw) = (own.len() * 512, peer.len() * 512);
        send.copy_2d_from(0, ow, b.out, own.start * 512, op, ow, rows, sx)?;
        super::shared_exchange(&send, rows * ow, &mut recv, rows * pw, sx)?;
        b.out.alias().copy_2d_from(peer.start * 512, op, &recv, 0, pw, pw, rows, sx)
    }
}

/// The buffers of `shared_split_tp`: the up projection's ScaledRows input and output, the f32 output, the staging.
struct SharedBufs<'a> {
    hs: (&'a DevBuf, &'a DevBuf),
    sh: (&'a DevBuf, &'a DevBuf),
    out: &'a DevBuf,
    send: &'a DevBuf,
    recv: &'a DevBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_item_stride_bounds_the_items_of_every_tile() {
        for rows in [1usize, 64, 256, 2048, 4096] {
            let p = rows * TOPK;
            for mt in [1usize, 2, 4, 8] {
                assert!(mma_item_stride(rows) >= EXPERTS.min(p) + p.div_ceil(16 * mt));
            }
        }
    }

    #[test]
    fn the_scratch_is_linear_in_rows() {
        let a = MmaScratch::bytes(256) as f64;
        let b = MmaScratch::bytes(2048) as f64;
        assert!((b / a - 8.0).abs() < 0.05, "{a} {b}");
    }

    #[test]
    fn the_switch_is_off_by_default() {
        if std::env::var_os("REXL3_NEMO_PREFILL_MMA").is_none() {
            assert!(!enabled());
        }
    }
}
