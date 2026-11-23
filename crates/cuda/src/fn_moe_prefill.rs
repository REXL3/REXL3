//! Qwen3.8-Flash-Next prefill: the routed experts of [`crate::moe_prefill`] with the two grouped GEMMs replaced by ring-fed,
//! column-owning kernels (`csrc/fn_moe_prefill.cu`): one launch each over slices of up to 16 tiles of one expert, a cp.async ring
//! streaming the A rows and the raw trellis tiles, every warp decoding only its own column tiles. The sums keep the per-item kernels'
//! K order (four chains of K / 64 k tiles, added in order), so the output equals [`MoeExperts::run_prefill_f32_wts`] bit for bit
//! (`tests/fn_moe_prefill.rs` on synthetic layers; on the pack, the family's
//! `real_pack_ring_fed_expert_gemms_equal_the_per_item_launches_on_every_layer`). The plan and the combine are the grouped path's
//! launches; the rotation is `csrc/glm_p2_moe.cu`'s (the same values, stored row-major).
//!
//! One node only: a rank of a group keeps the per-item launches (`crate::moe_owned`), as does any bank this does not take
//! ([`MoeExperts::ring_prefill`] is false: mixed widths, a codebook or width outside Mul1 at 2 to 4 bits, a decode plan with more
//! than one K split, the single-block plan).

use crate::exl3_linear::Dtype;
use crate::moe::MoeExperts;
use crate::moe_prefill::{ffi as pf, padded_rows, PrefillScratch};
use crate::rt::{CudaError, DevBuf, Stream};
use core::ffi::c_void;

#[allow(non_snake_case)]
mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_fn_pf_supported(
            hidden: i32,
            inter: i32,
            bits_gu: i32,
            bits_d: i32,
            codebook: i32,
            sk_gu: i32,
            wk_gu: i32,
            sk_d: i32,
            wk_d: i32,
        ) -> i32;
        // csrc/glm_p2_moe.cu's rotation: rot_in's values, stored row-major
        pub fn gb_glm_p2m_rotate(
            x: *const c_void,
            x_dtype: i32,
            pick: *const c_void,
            spair: *const c_void,
            nitems: *const c_void,
            suh0: *const c_void,
            suh1: *const c_void,
            a0: *mut c_void,
            a1: *mut c_void,
            rows: i32,
            topk: i32,
            k: i32,
            e: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_fn_pf_slice_entries(sorted_rows: i64, e: i32) -> i64;
        pub fn gb_fn_pf_slices(toff: *const c_void, cnt: *const c_void, slices: *mut c_void, e: i32, stream: *mut c_void) -> i32;
        pub fn gb_fn_pf_gateup(
            a0: *const c_void,
            a1: *const c_void,
            tg: *const c_void,
            tu: *const c_void,
            slices: *const c_void,
            svh_g: *const c_void,
            svh_u: *const c_void,
            suh_d: *const c_void,
            ad: *mut c_void,
            k: i32,
            n: i32,
            bits: i32,
            codebook: i32,
            sk: i32,
            wk: i32,
            e: i32,
            sorted_rows: i64,
            limit: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_fn_pf_down(
            ad: *const c_void,
            td: *const c_void,
            slices: *const c_void,
            svh_d: *const c_void,
            spair: *const c_void,
            y: *mut c_void,
            k: i32,
            n: i32,
            bits: i32,
            codebook: i32,
            sk: i32,
            wk: i32,
            e: i32,
            sorted_rows: i64,
            stream: *mut c_void,
        ) -> i32;
    }
}

const INVALID: CudaError = CudaError(1);

fn check(rc: i32) -> Result<(), CudaError> {
    CudaError::check(rc)
}

fn codebook_code(cb: gb_exl3::Codebook) -> i32 {
    match cb {
        gb_exl3::Codebook::Inst3 => 0,
        gb_exl3::Codebook::Mcg => 1,
        gb_exl3::Codebook::Mul1 => 2,
    }
}

impl MoeExperts {
    /// Whether [`MoeExperts::run_prefill_ring_f32_wts`] takes this bank with scratch `s`: every expert held, one width band,
    /// gate and up fused, a width, codebook and decode plan (one K split of four warps) the ring-fed kernels reproduce, and the
    /// parallel plan (whose per-expert offsets and counts the slice list is built from).
    pub fn ring_prefill(&self, s: &PrefillScratch) -> bool {
        let d = self.dims;
        self.holds_every_expert()
            && self.fused_gate_up()
            && self.bands.len() == 1
            && s.parallel_plan()
            // the slice list lives in the plan's item buffer
            && unsafe { ffi::gb_fn_pf_slice_entries(padded_rows(s.max_rows * d.topk, d.experts) as i64, d.experts as i32) } as usize * 16
                <= s.items.bytes()
            && unsafe {
                ffi::gb_fn_pf_supported(
                    d.hidden as i32,
                    d.inter as i32,
                    self.bits_gu as i32,
                    self.bits_d as i32,
                    codebook_code(self.codebook),
                    self.plan_gu.0 as i32,
                    self.plan_gu.1 as i32,
                    self.plan_d.0 as i32,
                    self.plan_d.1 as i32,
                ) == 1
            }
    }

    /// [`MoeExperts::run_prefill_f32_wts`] with the two grouped GEMMs ring-fed: the same plan and combine launches around them,
    /// the same bits in `out` and `s.y_pairs`. Refused (`INVALID`) unless [`MoeExperts::ring_prefill`]. Uses the scratch's own
    /// buffers at their sizes (`xs_g`, `xs_u`, `xd` hold row-major rows; `items` the slice list), so it adds no memory.
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill_ring_f32_wts(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut PrefillScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.run_prefill_ring_f32_wts_marked(x, x_dtype, rows, shared, out, s, st, &mut |_| Ok(()))
    }

    /// [`MoeExperts::run_prefill_ring_f32_wts`] calling `mark(stage)` after each launch (profiling).
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill_ring_f32_wts_marked(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut PrefillScratch,
        st: &Stream,
        mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        let d = self.dims;
        if rows == 0
            || !self.ring_prefill(s)
            || rows > s.max_rows
            || !s.fits(&d)
            || x.bytes() < rows * d.hidden * x_dtype.size()
            || out.bytes() < rows * d.hidden * 4
            || shared.is_some_and(|b| b.bytes() < rows * d.hidden * 4)
            || s.wts_f32.bytes() < rows * d.topk * 4
        {
            return Err(INVALID);
        }
        let (sk_gu, wk_gu) = self.plan_gu;
        let (sk_d, wk_d) = self.plan_d;
        let cb = codebook_code(self.codebook);
        let (hidden, inter, experts, topk) = (d.hidden as i32, d.inter as i32, d.experts as i32, d.topk as i32);
        let sorted = padded_rows(rows * d.topk, d.experts) as i64;
        unsafe {
            s.run_plan(s.pick.ptr(), rows, experts, st)?;
            mark("plan")?;
            check(ffi::gb_glm_p2m_rotate(
                x.ptr(),
                x_dtype as i32,
                s.pick.ptr(),
                s.spair.ptr(),
                s.nitems.ptr(),
                self.suh_g.ptr(),
                self.suh_u.ptr(),
                s.xs_g.ptr(),
                s.xs_u.ptr(),
                rows as i32,
                topk,
                hidden,
                experts,
                st.raw(),
            ))?;
            mark("rot_in")?;
            // the plan's work items are not used here: their buffer takes the slice list
            check(ffi::gb_fn_pf_slices(s.plan_toff.ptr(), s.plan_cnt.ptr(), s.items.ptr(), experts, st.raw()))?;
            check(ffi::gb_fn_pf_gateup(
                s.xs_g.ptr(),
                s.xs_u.ptr(),
                self.gate_ptrs.ptr(),
                self.up_ptrs.ptr(),
                s.items.ptr(),
                self.svh_g.ptr(),
                self.svh_u.ptr(),
                self.suh_d.ptr(),
                s.xd.ptr(),
                hidden,
                inter,
                self.bits_gu as i32,
                cb,
                sk_gu as i32,
                wk_gu as i32,
                experts,
                sorted,
                d.limit,
                st.raw(),
            ))?;
            mark("gate_up")?;
            check(ffi::gb_fn_pf_down(
                s.xd.ptr(),
                self.down_ptrs.ptr(),
                s.items.ptr(),
                self.svh_d.ptr(),
                s.spair.ptr(),
                s.y_pairs.ptr(),
                inter,
                hidden,
                self.bits_d as i32,
                cb,
                sk_d as i32,
                wk_d as i32,
                experts,
                sorted,
                st.raw(),
            ))?;
            mark("down")?;
            check(pf::gb_moe_pf_combine_f32(
                s.y_pairs.ptr(),
                s.pick.ptr(),
                s.wts_f32.ptr() as *const c_void,
                shared.map_or(core::ptr::null(), |b| b.ptr() as *const c_void),
                out.ptr(),
                rows as i32,
                topk,
                hidden,
                experts,
                st.raw(),
            ))?;
            mark("combine")
        }
    }
}
