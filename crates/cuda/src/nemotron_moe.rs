//! Latent-MoE block of the Nemotron 3.5 Super VL engine (`csrc/nemotron_moe.cu`): router (fp32, top-22), routed experts (EXL3 4-bit
//! trellis or bf16, per expert), fc1 and fc2 (EXL3 latent projections) and the shared expert (EXL3). The CPU oracle is
//! `gb_model::families::nemotron_h_omni::moe_ref`, and EXL3 decodes exactly as `gb_exl3::unpack` does.
//!
//! Row invariance: a row's bits do not depend on how many rows share the call (the reduction order depends only on the shape), so the
//! decode path (1 row) and the verify path (up to `max_rows`) agree bit for bit. Every function launches on `stream` and returns without
//! synchronising. Buffer sizes are checked; a short buffer panics rather than faulting on the device.
//!
//! Two sets of launches give those bits ([`Variant`]): the reference one (the original kernels, one launch per step) and the fast one that
//! `forward_in` runs (4-bit Mul1 projections with the rotation and the output side fused into the projection kernel, the router on a
//! warp-per-experts layout with selection and buckets in one launch, programmatic dependent launch, and the shared expert on a side
//! stream beside the routed chain). `tests/nemotron_moe_perf.rs` gates the fast one bit for bit against the reference; see
//! `docs/NEMOTRON_MOE_PERF.md`.
//!
//! Layouts: `h`, `out` are f32 `[rows, 4096]`; the gate is bf16 `[512, 4096]` (HF layout), or f32 for an F16 gate (widened exactly,
//! [`GateDtype`]); the EXL3 trellis is the checkpoint's int16
//! `[K/16, N/16, 8 * twice]`; `suh` is f32 `[K]`, `svh` and the bias f32 `[N]`; a bf16 expert weight is `[N, K]` (HF layout).

use crate::graph::{fork, join};
use crate::rt::Event;
use crate::{CudaError, DevBuf, Stream};
use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, Ordering};
use gb_exl3::checkpoint::LoadedGroup;
use gb_exl3::{codebook, Bits, Codebook};

pub const TOPK: usize = 22;
pub const EXPERTS: usize = 512;
pub const HIDDEN: usize = 4096;
pub const LATENT: usize = 1024;
pub const EXPERT_INTER: usize = 2688;
pub const SHARED_INTER: usize = 5376;

/// Bytes of one `NmProj`, the kernel's table entry (56 bytes).
const PROJ_BYTES: usize = 56;

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_nm_router_logits(
            h: *const c_void,
            gate: *const c_void,
            logits: *mut c_void,
            rows: i32,
            k: i32,
            e: i32,
            rb: i32,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nm_router_logits_g32(
            h: *const c_void,
            gate: *const c_void,
            logits: *mut c_void,
            rows: i32,
            k: i32,
            e: i32,
            rb: i32,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nm_router_logits_g16(
            h: *const c_void,
            gate: *const c_void,
            logits: *mut c_void,
            rows: i32,
            k: i32,
            e: i32,
            rb: i32,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nm_router_select(
            logits: *const c_void,
            bias: *const c_void,
            scale: f32,
            pick: *mut c_void,
            wts: *mut c_void,
            rows: i32,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nm_bucket(pick: *const c_void, p: i32, cap: i32, cnt: *mut c_void, lists: *mut c_void, s: *mut c_void) -> i32;

        pub fn gb_nm_t1(
            src: *const c_void,
            div: i32,
            eor: *const c_void,
            tab: *const c_void,
            rows: i32,
            k: i32,
            dst: *mut c_void,
            skip_fast: i32,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nm_gemm_exl3(
            xin: *const c_void,
            lists: *const c_void,
            cnt: *const c_void,
            fixed: i32,
            lcap: i32,
            tab: *const c_void,
            groups: i32,
            k: i32,
            n: i32,
            sl: i32,
            partial: *mut c_void,
            cap: i32,
            skip_fast: i32,
            active: *const c_void,
            nact: *const c_void,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nm_gemv_bf16(
            xin: *const c_void,
            lists: *const c_void,
            cnt: *const c_void,
            fixed: i32,
            lcap: i32,
            tab: *const c_void,
            groups: i32,
            k: i32,
            n: i32,
            partial: *mut c_void,
            cap: i32,
            active: *const c_void,
            nact: *const c_void,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nm_t2(
            partial: *const c_void,
            sl: i32,
            cap: i32,
            eor: *const c_void,
            tab: *const c_void,
            rows: i32,
            n: i32,
            relu2: i32,
            add: *const c_void,
            out: *mut c_void,
            skip_fast: i32,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nm_combine(y: *const c_void, wts: *const c_void, rows: i32, n: i32, routed: *mut c_void, s: *mut c_void) -> i32;

        pub fn gb_nm_decode(tab: *const c_void, idx: i32, out: *mut c_void, s: *mut c_void) -> i32;

        pub fn gb_nm2_gemm4(
            src: *const c_void,
            div: i32,
            ksrc: i32,
            lists: *const c_void,
            cnt: *const c_void,
            active: *const c_void,
            nact: *const c_void,
            fixed: i32,
            lcap: i32,
            tab: *const c_void,
            zgroups: i32,
            k: i32,
            n: i32,
            sl: i32,
            partial: *mut c_void,
            cap: i32,
            out: *mut c_void,
            relu2: i32,
            add: *const c_void,
            tickets: *mut c_void,
            npass: i32,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nm2_gemmk(
            src: *const c_void,
            div: i32,
            ksrc: i32,
            lists: *const c_void,
            cnt: *const c_void,
            active: *const c_void,
            nact: *const c_void,
            fixed: i32,
            lcap: i32,
            tab: *const c_void,
            zgroups: i32,
            k: i32,
            n: i32,
            sl: i32,
            partial: *mut c_void,
            cap: i32,
            out: *mut c_void,
            relu2: i32,
            add: *const c_void,
            tickets: *mut c_void,
            npass: i32,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nm2_dense16(
            x: *const c_void,
            w: *const c_void,
            rows: i32,
            k: i32,
            n: i32,
            out: *mut c_void,
            relu2: i32,
            add: *const c_void,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nm_set_pdl(on: i32) -> i32;

        pub fn gb_nm2_router_logits(h: *const c_void, gate: *const c_void, logits: *mut c_void, rows: i32, rb: i32, ew: i32, s: *mut c_void) -> i32;

        pub fn gb_nm2_router_logits_g32(h: *const c_void, gate: *const c_void, logits: *mut c_void, rows: i32, rb: i32, ew: i32, s: *mut c_void) -> i32;

        pub fn gb_nm2_router_logits_g16(h: *const c_void, gate: *const c_void, logits: *mut c_void, rows: i32, rb: i32, ew: i32, s: *mut c_void) -> i32;

        pub fn gb_nm2_select(
            logits: *const c_void,
            bias: *const c_void,
            scale: f32,
            pick: *mut c_void,
            wts: *mut c_void,
            rows: i32,
            cap: i32,
            cnt: *mut c_void,
            lists: *mut c_void,
            active: *mut c_void,
            nact: *mut c_void,
            ctr: *mut c_void,
            s: *mut c_void,
        ) -> i32;
    }
}

/// The kernel's `struct NmProj`: one projection as the device reads it (field order and sizes match `nemotron_moe.cu`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NmProj {
    w: u64,
    suh: u64,
    svh: u64,
    bias: u64,
    cb: u64,
    fmt: i32,
    twice: i32,
    k: i32,
    n: i32,
}

const FMT_EXL3: i32 = 0;
const FMT_BF16: i32 = 1;
const FMT_ABSENT: i32 = 2;
const FMT_F16: i32 = 3;

/// One projection's device data.
pub enum Proj {
    /// An EXL3 linear layer: the trellis, the input scale `suh` [K], the output scale `svh` [N] and an optional bias [N].
    Exl3 { k: usize, n: usize, bits: Bits, codebook: Codebook, trellis: DevBuf, suh: DevBuf, svh: DevBuf, bias: Option<DevBuf> },
    /// A bf16 weight `[N, K]` in the HF layout (`y = x . W^T`, no transform).
    Bf16 { k: usize, n: usize, w: DevBuf },
    /// An fp16 weight `[N, K]` in the HF layout (a MixedK pack's latent projections fc1 and fc2): the bf16 projection's chain on its
    /// values widened exactly in the kernel (never rounded to bf16, no copy). A dense projection only: the routed experts' prefill kernels
    /// do not take it (`pair_kind`).
    F16 { k: usize, n: usize, w: DevBuf },
    /// An expert with no weights loaded. The kernels write exact `+0.0` for every (row, slot) pair routed to it (its `a` and `y` rows), on
    /// the fast, reference and prefill paths alike: that is the disjoint support expert-parallel TP sums (`forward_in_tp`). On a single
    /// node, routing to one is a caller error (the pair then contributes nothing to the block).
    Absent,
}

fn codebook_index(cb: Codebook) -> usize {
    match cb {
        Codebook::Inst3 => 0,
        Codebook::Mcg => 1,
        Codebook::Mul1 => 2,
    }
}

impl Proj {
    /// True for a projection the fast kernels run: EXL3 at 4 bits with the Mul1 codebook (every projection of the published pack's MoE
    /// layers but the three bf16 expert pairs). Absent experts are neither fast nor slow.
    fn is_fast(&self) -> bool {
        matches!(self, Proj::Exl3 { bits, codebook, .. } if bits.twice() == 8 && *codebook == Codebook::Mul1)
    }

    /// True for a projection the any-width fast kernel (`nm2_gemmk`) runs: EXL3 Mul1 at an integer width of 2 to 8 bits (4 included).
    pub fn is_fastk(&self) -> bool {
        matches!(self, Proj::Exl3 { bits, codebook, .. }
            if *codebook == Codebook::Mul1 && bits.twice() % 2 == 0 && (4..=16).contains(&bits.twice()))
    }

    /// An fp16 projection from its raw bits, HF layout `[n, k]`.
    pub fn f16(k: usize, n: usize, bits: &[u16]) -> Result<Proj, CudaError> {
        assert_eq!(bits.len(), k * n, "f16 weight is not [{n}, {k}]");
        Ok(Proj::F16 { k, n, w: DevBuf::from_slice(bits)? })
    }

    /// An EXL3 projection from a loaded checkpoint group. The trellis is copied as stored; the scales and bias are copied as f32.
    pub fn exl3_from_group(g: &LoadedGroup) -> Result<Proj, CudaError> {
        let f32s = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        let bias = match &g.bias {
            Some(b) => Some(DevBuf::from_slice(&f32s(b))?),
            None => None,
        };
        Ok(Proj::Exl3 {
            k: g.group.k,
            n: g.group.n,
            bits: g.group.bits,
            codebook: g.group.codebook,
            trellis: DevBuf::from_slice(&g.trellis)?,
            suh: DevBuf::from_slice(&f32s(&g.suh))?,
            svh: DevBuf::from_slice(&f32s(&g.svh))?,
            bias,
        })
    }

    /// A bf16 projection from its raw bits, HF layout `[n, k]`.
    pub fn bf16(k: usize, n: usize, bits: &[u16]) -> Result<Proj, CudaError> {
        assert_eq!(bits.len(), k * n, "bf16 weight is not [{n}, {k}]");
        Ok(Proj::Bf16 { k, n, w: DevBuf::from_slice(bits)? })
    }

    fn dims(&self) -> (usize, usize) {
        match self {
            Proj::Exl3 { k, n, .. } | Proj::Bf16 { k, n, .. } | Proj::F16 { k, n, .. } => (*k, *n),
            Proj::Absent => (0, 0),
        }
    }
}

/// The weights of one latent-MoE layer. `up` and `down` have 512 entries each; an expert with no weights loaded is `Proj::Absent`.
pub struct Weights {
    /// The router gate `[512, 4096]` in the form `gate_dtype` names: bf16 bits (the published pack's), fp16 bits (a MixedK pack's,
    /// as stored), or f32. Its byte count must be that form's (`tables` refuses anything else by name).
    pub gate: DevBuf,
    /// The gate's element type. bf16 and fp16 gates have the same bytes, so the type is this tag and never inferred from the size:
    /// an F16 gate is never read as bf16 bits and never rounded to bf16 (v0.1.2 decision D1).
    pub gate_dtype: GateDtype,
    /// `e_score_correction_bias`, f32 `[512]`.
    pub router_bias: DevBuf,
    /// `routed_scaling_factor` (5.0 for the Nemotron 3.5 Super VL checkpoint).
    pub scale: f32,
    pub fc1: Proj,
    pub fc2: Proj,
    pub shared_up: Proj,
    pub shared_down: Proj,
    pub up: Vec<Proj>,
    pub down: Vec<Proj>,
}

/// The element type of a router gate on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateDtype {
    /// bf16 bits: the published pack's gate, run by the original router kernels.
    Bf16,
    /// fp16 bits as stored (a MixedK pack's gate): the router kernels' fp16 instantiation, which widens each value exactly in the
    /// chunk's multiply-adds, so its bits are the f32 kernels' on the widened gate (decision D1: exact, never rounded to bf16), with
    /// the bf16 gate's bytes and no copy.
    F16,
    /// f32: the f32-gate variants of the same kernels (the same chains, the gate's values read as f32). The oracle of the fp16 ones.
    F32,
}

impl GateDtype {
    /// The bytes of a `[512, 4096]` gate of this type.
    pub fn bytes(self) -> usize {
        EXPERTS * HIDDEN * if self == GateDtype::F32 { 4 } else { 2 }
    }
}

/// An F16 gate's values as f32 (exact: every fp16 value, subnormals included, is an f32): the f32 gate the fp16 kernels' bits equal.
pub fn gate_f32_from_f16(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect()
}

/// The fast router's layout for a call of `rows` rows: (rows a warp carries, experts a warp carries). A warp carries one expert and one
/// row up to 12 rows (more warps in flight), then 4 rows of 4 experts (each chunk of x loaded once for four experts), or of 2 for an f32
/// gate (4 x 4 spills there). Every layout gives the same bits: a (row, expert) chain is the same wherever it runs.
pub(crate) fn fast_router_layout(rows: usize, gate: GateDtype) -> (i32, i32) {
    match (rows <= 12, gate) {
        (true, _) => (1, 1),
        (false, GateDtype::Bf16 | GateDtype::F16) => (4, 4),
        (false, GateDtype::F32) => (4, 2),
    }
}

/// The fast router's logits `[rows, 512]` of `h` (`[rows, 4096]` f32) against `gate` of form `dt`, on the raw stream `st`, with the
/// layout of [`fast_router_layout`]. The prefill and tensor-core paths call this too, so every path takes one dispatch.
///
/// # Safety
/// Device pointers sized for `rows` rows and a gate of form `dt`.
pub(crate) unsafe fn fast_router_logits(
    h: *const c_void,
    gate: *const c_void,
    dt: GateDtype,
    logits: *mut c_void,
    rows: usize,
    st: *mut c_void,
) -> Result<(), CudaError> {
    let (rb, ew) = fast_router_layout(rows, dt);
    ok(match dt {
        GateDtype::Bf16 => ffi::gb_nm2_router_logits(h, gate, logits, rows as i32, rb, ew, st),
        GateDtype::F16 => ffi::gb_nm2_router_logits_g16(h, gate, logits, rows as i32, rb, ew, st),
        GateDtype::F32 => ffi::gb_nm2_router_logits_g32(h, gate, logits, rows as i32, rb, ew, st),
    })
}

/// The stages of one block, in launch order, as `forward_staged` reports them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Router logits, top-22 selection and weights.
    Router,
    /// The per-expert pair lists.
    Bucket,
    /// fc1: hidden to latent.
    LatentIn,
    /// The routed experts' up projections with relu2.
    Up,
    /// The routed experts' down projections.
    Down,
    /// The weighted sum of the 22 slots.
    Combine,
    /// fc2: latent back to hidden.
    Fc2,
    /// The shared expert (up, relu2, down, plus the routed sum).
    Shared,
}

/// The reduce hook of an expert-parallel block (`forward_in_tp`, `PrefillMoe::forward_prefill_in_tp`): called once per block on the main
/// stream with the routed experts' output buffer (f32, `[rows, 22, 1024]`, a pair's row at `(row * 22 + slot) * 1024`), the number of f32
/// elements that are valid (`rows * 22 * 1024`; the buffer may be larger) and the stream. It must sum the buffer over the ranks in place,
/// queued on that stream (no host synchronisation if the block is to be captured into a graph).
pub type ReduceHook<'a> = dyn FnMut(&mut DevBuf, usize, &Stream) -> Result<(), CudaError> + 'a;

static SIDE_STREAM: AtomicBool = AtomicBool::new(true);
static ANY_WIDTH: AtomicBool = AtomicBool::new(false);

/// Run every fast EXL3 entry through `nm2_gemmk` (the any-width kernel), 4-bit-only tables included (whose default is `nm2_gemm4`).
/// The bits are the same; this exists for the bit gate of `nm2_gemmk` at 4 bits and to time the two kernels on the same weights.
/// Process-wide, not for use while a block is running.
pub fn set_any_width_kernel(on: bool) {
    ANY_WIDTH.store(on, Ordering::Relaxed);
}

/// The rows up to which the shared expert runs on the side stream (its partial sums are 16 slices of rows x 5376 floats; the scratch
/// holds them for this many rows only, and above it the expert runs in line, where it has the compute units to itself anyway).
pub const SIDE_ROWS: usize = 16;

/// The rows one block of a dense projection's grid carries (`kDenseRb` in the kernel file).
const DENSE_CHUNK_ROWS: usize = 16;

/// Run the shared expert on the scratch's side stream beside the routed chain (the default), or in line on the caller's stream. The bits
/// are the same; this exists to time the difference. Process-wide, not for use while a block is running.
pub fn set_side_stream(on: bool) {
    SIDE_STREAM.store(on, Ordering::Relaxed);
}

/// Whether the fast launches carry the programmatic-dependent-launch attribute (the default), or are ordinary stream launches. The bits
/// do not depend on it; it exists to time the difference. Process-wide, not for use while a block is running.
pub fn set_programmatic_launch(on: bool) {
    unsafe {
        ffi::gb_nm_set_pdl(on as i32);
    }
}

/// Which launches a block runs. Both give the same bits (the per-row arithmetic is one and the same); `Reference` is the original
/// kernels, one launch for each step, and `Fast` the scheduled ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Reference,
    Fast,
}

/// One projection stage: where its input rows are, which rows each group has, and where its partial sums go.
#[derive(Clone, Copy)]
struct ProjRun {
    /// Input: row `id / div` of `src` (f32, row stride `k`) for row `id` of a group.
    src: usize,
    div: usize,
    /// Table entry of each input row (the reference input side), or 0 for entry 0.
    eor: usize,
    /// Rows of the reference input side.
    rows: usize,
    lists: usize,
    cnt: usize,
    active: usize,
    nact: usize,
    fixed: usize,
    lcap: usize,
    tab: usize,
    groups: usize,
    zgroups: usize,
    k: usize,
    n: usize,
    partial: usize,
    cap: usize,
    /// The reference input side's output.
    xh: usize,
    /// The output side: `out` [rows, n], relu2, and `add` [rows, n] (or 0).
    out: usize,
    relu2: bool,
    add: usize,
    /// The ticket array of the fast projection (zero between calls) and the passes of 8 rows a routed group can take.
    tickets: usize,
    npass: usize,
    /// Some entry of `tab` is not fast (needs the reference input side and GEMM beside the fast one), or is absent (its output side
    /// writes zeros): the fast variant then also runs those reference launches, which skip the fast entries.
    slow: bool,
    absent: bool,
    /// The fast launch is `nm2_gemmk` (every Mul1 entry of an integer width, 4 included: a table with another width among its
    /// fast entries), not `nm2_gemm4`; the reference launches then skip what `nm2_gemmk` ran (skip level 2).
    kfast: bool,
    /// A dense projection whose weight is fp16: its address. The fast variant runs it as one `nm2_dense16` launch (the reference
    /// launches' bits); 0 for every other projection.
    f16w: usize,
}

/// Which projection a debug decode reads.
#[derive(Clone, Copy, Debug)]
pub enum Table {
    Up(usize),
    Down(usize),
    /// 0 = fc1, 1 = fc2, 2 = shared up, 3 = shared down.
    Dense(usize),
}

/// The number of EXL3 k slices of a projection with `k` inputs: the largest of 16, 8, 4, 2, 1 that divides the k tile count and leaves at
/// least 16 k tiles a slice (so a slice is a long enough chain, and the partial sums stay small). K 1024 gives 4, 2688 gives 8, 4096 and
/// 5376 give 16.
pub fn slices(k: usize) -> usize {
    let kt = k / 16;
    [16usize, 8, 4, 2, 1].into_iter().find(|&s| kt % s == 0 && kt / s >= 16).unwrap_or(1)
}

fn codebook_table(i: usize) -> Codebook {
    [Codebook::Inst3, Codebook::Mcg, Codebook::Mul1][i]
}

fn nm_proj(p: &Proj, cb: &[DevBuf; 3]) -> NmProj {
    match p {
        Proj::Exl3 { k, n, bits, codebook, trellis, suh, svh, bias } => {
            assert!(k % 16 == 0 && n % 128 == 0, "EXL3 projection [{k}, {n}]: K must be a multiple of 16 and N of 128");
            NmProj {
                w: trellis.ptr() as u64,
                suh: suh.ptr() as u64,
                svh: svh.ptr() as u64,
                bias: bias.as_ref().map_or(0, |b| b.ptr() as u64),
                cb: cb[codebook_index(*codebook)].ptr() as u64,
                // bits 4..5: the codebook id (the fast kernels compute the codebook and take only 4-bit Mul1)
                fmt: FMT_EXL3 | ((codebook_index(*codebook) as i32) << 4),
                twice: bits.twice() as i32,
                k: *k as i32,
                n: *n as i32,
            }
        }
        Proj::Bf16 { k, n, w } => {
            assert!(k % 8 == 0, "bf16 projection: K = {k} is not a multiple of 8");
            NmProj { w: w.ptr() as u64, fmt: FMT_BF16, k: *k as i32, n: *n as i32, ..Default::default() }
        }
        Proj::F16 { k, n, w } => {
            assert!(k % 8 == 0, "f16 projection: K = {k} is not a multiple of 8");
            NmProj { w: w.ptr() as u64, fmt: FMT_F16, k: *k as i32, n: *n as i32, ..Default::default() }
        }
        Proj::Absent => NmProj { fmt: FMT_ABSENT, ..Default::default() },
    }
}

/// A device address as the launchers take it.
fn addr(b: &DevBuf) -> usize {
    b.ptr() as usize
}

fn ok(rc: i32) -> Result<(), CudaError> {
    CudaError::check(rc)
}

/// Scratch for the block's stages, sized for up to `max_rows` tokens (`max_rows * 22` pairs). The layers of a model run one after
/// another on a stream, so one instance can serve all of them.
pub struct MoeScratch {
    max_rows: usize,
    logits: DevBuf,
    pick: DevBuf,
    wts: DevBuf,
    cnt: DevBuf,
    lists: DevBuf,
    /// The experts that have rows (ascending ids) and how many (the fast kernels' grid is over this list).
    active: DevBuf,
    nact: DevBuf,
    /// The selection kernel's ticket counter (zero between calls).
    ctr: DevBuf,
    /// The projection kernel's tickets, one per (group, column block, pass): zero between calls.
    tickets: DevBuf,
    xh: DevBuf,
    partial: DevBuf,
    a: DevBuf,
    y: DevBuf,
    xlat: DevBuf,
    routed: DevBuf,
    routed_out: DevBuf,
    sh_a: DevBuf,
    /// The shared expert's output and the buffers it needs to run beside the routed chain (a side stream and its events).
    sh_out: DevBuf,
    sh_partial: DevBuf,
    sh_tickets: DevBuf,
    aux: Stream,
    ev_fork: Event,
    ev_join: Event,
}

impl MoeScratch {
    /// The number of 4-byte elements of each buffer, in field order, for `max_rows` rows.
    fn lens(max_rows: usize) -> [usize; 20] {
        let r = max_rows;
        let p = r * TOPK;
        // The partial sums of each projection are S slices of `cap` rows of N floats; the largest is the expert up projection.
        let partial_len = [
            slices(LATENT) * p * EXPERT_INTER,
            slices(EXPERT_INTER) * p * LATENT,
            slices(HIDDEN) * r * LATENT,
            slices(LATENT) * r * HIDDEN,
            slices(HIDDEN) * r * SHARED_INTER,
            slices(SHARED_INTER) * r * HIDDEN,
        ]
        .into_iter()
        .max()
        .unwrap();
        [
            r * EXPERTS, // logits
            p,           // pick (i32)
            p,           // wts
            EXPERTS,     // cnt (i32)
            EXPERTS * r, // lists (i32)
            EXPERTS,     // active (i32)
            1,           // nact (i32)
            1,           // ctr (u32)
            // tickets (u32): the routed projections are 512 groups of 21 column blocks and 8-row passes, a dense one 8-row chunks of 42
            (EXPERTS * (EXPERT_INTER / 128) * r.div_ceil(8)).max((SHARED_INTER / 128) * r.div_ceil(8)),
            (p * EXPERT_INTER).max(r * SHARED_INTER), // xh
            partial_len,                              // partial
            p * EXPERT_INTER,                         // a
            p * LATENT,                               // y
            r * LATENT,                               // xlat
            r * LATENT,                               // routed
            r * HIDDEN,                               // routed_out
            r * SHARED_INTER,                         // sh_a
            r * HIDDEN,                               // sh_out
            // sh_partial: the shared expert's partial slices (up: S x rows x 5376, down: S x rows x 4096), beside the routed chain's,
            // for up to SIDE_ROWS rows (a bigger block runs the shared expert in line, on the routed chain's own partials)
            (slices(HIDDEN) * r.min(SIDE_ROWS) * SHARED_INTER).max(slices(SHARED_INTER) * r.min(SIDE_ROWS) * HIDDEN),
            (SHARED_INTER / 128) * r.min(SIDE_ROWS).div_ceil(DENSE_CHUNK_ROWS), // sh_tickets (u32)
        ]
    }

    /// The device bytes `new(max_rows)` allocates.
    pub fn bytes(max_rows: usize) -> u64 {
        Self::lens(max_rows).iter().map(|n| *n as u64 * 4).sum()
    }

    pub fn new(max_rows: usize) -> Result<Self, CudaError> {
        assert!(max_rows >= 1, "max_rows must be at least 1");
        let l = Self::lens(max_rows);
        let f = |n: usize| DevBuf::zeroed(n * 4);
        Ok(Self {
            max_rows,
            logits: f(l[0])?,
            pick: f(l[1])?,
            wts: f(l[2])?,
            cnt: f(l[3])?,
            lists: f(l[4])?,
            active: f(l[5])?,
            nact: f(l[6])?,
            ctr: f(l[7])?,
            tickets: f(l[8])?,
            xh: f(l[9])?,
            partial: f(l[10])?,
            a: f(l[11])?,
            y: f(l[12])?,
            xlat: f(l[13])?,
            routed: f(l[14])?,
            routed_out: f(l[15])?,
            sh_a: f(l[16])?,
            sh_out: f(l[17])?,
            sh_partial: f(l[18])?,
            sh_tickets: f(l[19])?,
            aux: Stream::new_nonblocking()?,
            ev_fork: Event::new()?,
            ev_join: Event::new()?,
        })
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    fn check_rows(&self, rows: usize) {
        assert!((1..=self.max_rows).contains(&rows), "rows {rows} is outside 1..={}", self.max_rows);
    }

    /// Fills every intermediate with a NaN bit pattern, queued on `st`. The ticket counters are left alone: they hold zero between
    /// calls by contract. For tests: a stage that does not write its output cannot then pass by finding the previous call's values.
    pub fn poison(&mut self, st: &Stream) -> Result<(), CudaError> {
        for b in [
            &mut self.logits,
            &mut self.pick,
            &mut self.wts,
            &mut self.cnt,
            &mut self.lists,
            &mut self.active,
            &mut self.nact,
            &mut self.xh,
            &mut self.partial,
            &mut self.a,
            &mut self.y,
            &mut self.xlat,
            &mut self.routed,
            &mut self.routed_out,
            &mut self.sh_a,
            &mut self.sh_out,
            &mut self.sh_partial,
        ] {
            b.memset_async(0xFF, st)?;
        }
        Ok(())
    }

    /// The router's logits of the last route (`[rows, 512]`) on the host. Synchronises.
    pub fn download_logits(&self, rows: usize, st: &Stream) -> Result<Vec<f32>, CudaError> {
        st.sync()?;
        let mut out = vec![0f32; rows * EXPERTS];
        self.logits.download_range(0, &mut out)?;
        Ok(out)
    }

    /// The picks `[rows, 22]` (in rank order) and weights of the last route on the host. Synchronises.
    pub fn download_route(&self, rows: usize, st: &Stream) -> Result<(Vec<i32>, Vec<f32>), CudaError> {
        st.sync()?;
        let mut pick = vec![0i32; rows * TOPK];
        let mut wts = vec![0f32; rows * TOPK];
        self.pick.download_range(0, &mut pick)?;
        self.wts.download_range(0, &mut wts)?;
        Ok((pick, wts))
    }

    /// The routed experts' down-projection outputs of the last block, `[rows, 22, 1024]` f32 (slot `s` of row `r` is the output of the
    /// `s`-th selected expert, before the weighted combine), on the host. A slot whose expert is absent holds zeros. Synchronises.
    /// This is the buffer a TP group sums across ranks (each slot is non-zero on exactly one rank).
    pub fn download_y(&self, rows: usize, st: &Stream) -> Result<Vec<f32>, CudaError> {
        st.sync()?;
        let mut out = vec![0f32; rows * TOPK * LATENT];
        self.y.download_range(0, &mut out)?;
        Ok(out)
    }
}

/// The block's device tables (its weights). Run it with a shared [`MoeScratch`] through [`NemotronMoe::forward_in`], or build it with
/// [`NemotronMoe::new`], which also owns a scratch of its own and keeps the older methods working.
pub struct NemotronMoe {
    w: Weights,
    // Kept alive: the codebooks the tables point at.
    _cb: [DevBuf; 3],
    // Device tables of NmProj: 512 up entries, 512 down entries, and the four dense projections (fc1, fc2, shared up, shared down).
    tab_up: DevBuf,
    tab_down: DevBuf,
    tab_dense: DevBuf,
    /// Some routed projection is not run by the fast kernel of the table (`routed_kfast`: nm2_gemmk, any integer width of Mul1; else
    /// nm2_gemm4, 4-bit Mul1): a bf16 expert, a half-bit width, another codebook. The fast variant then also runs the reference input
    /// side and GEMM for those.
    has_slow: bool,
    /// Some routed projection is Mul1 at an integer width other than 4 (a MixedK pack): the routed fast launch is nm2_gemmk for the
    /// whole table. An all-4-bit table (the published pack) keeps nm2_gemm4, exactly today's launches.
    routed_kfast: bool,
    /// Some expert is absent: the fast variant then runs the reference output side too, which writes the zeros of its pairs.
    has_absent: bool,
    /// Dense projection `which` (fc1, fc2, shared up, shared down) is run by no fast EXL3 kernel (bf16, fp16, another codebook).
    dense_slow: [bool; 4],
    /// Dense projection `which` is Mul1 at an integer width other than 4: nm2_gemmk runs it.
    dense_kfast: [bool; 4],
    /// Dense projection `which` is fp16: its weight's address (nm2_dense16), else 0.
    dense_f16: [usize; 4],
    /// The shared expert's two projections are both run by a fast EXL3 kernel: it then runs on the scratch's side stream beside the
    /// routed chain (they touch only the side stream's partials and tickets; fc1 and fc2 may be anything).
    dense_fast: bool,
    /// The router gate's form (bf16 bits or an f32 widening of an F16 gate): which router kernels run.
    gate_dt: GateDtype,
    own: Option<MoeScratch>,
}

impl NemotronMoe {
    /// Uploads the tables only. Shapes are checked against the block's dimensions. No scratch is allocated: use `forward_in`.
    pub fn tables(w: Weights) -> Result<Self, CudaError> {
        let gate_dt = w.gate_dtype;
        assert_eq!(
            w.gate.bytes(),
            gate_dt.bytes(),
            "gate is {} bytes: a {gate_dt:?} gate [512, 4096] is {} bytes",
            w.gate.bytes(),
            gate_dt.bytes()
        );
        assert_eq!(w.router_bias.bytes(), EXPERTS * 4, "router bias is not f32 [512]");
        assert_eq!(w.up.len(), EXPERTS, "up needs 512 entries");
        assert_eq!(w.down.len(), EXPERTS, "down needs 512 entries");
        assert_eq!(w.fc1.dims(), (HIDDEN, LATENT), "fc1 is not [4096, 1024]");
        assert_eq!(w.fc2.dims(), (LATENT, HIDDEN), "fc2 is not [1024, 4096]");
        assert_eq!(w.shared_up.dims(), (HIDDEN, SHARED_INTER), "shared up is not [4096, 5376]");
        assert_eq!(w.shared_down.dims(), (SHARED_INTER, HIDDEN), "shared down is not [5376, 4096]");
        for (i, p) in w.up.iter().enumerate() {
            if !matches!(p, Proj::Absent) {
                assert_eq!(p.dims(), (LATENT, EXPERT_INTER), "expert {i} up is not [1024, 2688]");
            }
        }
        for (i, p) in w.down.iter().enumerate() {
            if !matches!(p, Proj::Absent) {
                assert_eq!(p.dims(), (EXPERT_INTER, LATENT), "expert {i} down is not [2688, 1024]");
            }
        }
        let cb: [DevBuf; 3] = [0, 1, 2].map(|i| DevBuf::from_slice(&codebook(codebook_table(i))).expect("codebook upload"));
        let up_t: Vec<NmProj> = w.up.iter().map(|p| nm_proj(p, &cb)).collect();
        let down_t: Vec<NmProj> = w.down.iter().map(|p| nm_proj(p, &cb)).collect();
        let dense_t: [NmProj; 4] =
            [nm_proj(&w.fc1, &cb), nm_proj(&w.fc2, &cb), nm_proj(&w.shared_up, &cb), nm_proj(&w.shared_down, &cb)];
        let tab_up = DevBuf::from_slice(&up_t[..])?;
        let tab_down = DevBuf::from_slice(&down_t[..])?;
        let tab_dense = DevBuf::from_slice(&dense_t[..])?;
        let present = || w.up.iter().chain(w.down.iter()).filter(|p| !matches!(p, Proj::Absent));
        // a table with a fast entry of another width than 4 runs nm2_gemmk (it takes the 4-bit entries too); an all-4-bit one keeps
        // nm2_gemm4 and today's launches exactly
        let routed_kfast = present().any(|p| p.is_fastk() && !p.is_fast());
        let has_slow = if routed_kfast { present().any(|p| !p.is_fastk()) } else { present().any(|p| !p.is_fast()) };
        let has_absent = w.up.iter().chain(w.down.iter()).any(|p| matches!(p, Proj::Absent));
        let dense4 = [&w.fc1, &w.fc2, &w.shared_up, &w.shared_down];
        let dense_slow = dense4.map(|p| !p.is_fastk());
        let dense_kfast = dense4.map(|p| p.is_fastk() && !p.is_fast());
        let dense_f16 = dense4.map(|p| match p {
            Proj::F16 { w, .. } => w.ptr() as usize,
            _ => 0,
        });
        let dense_fast = !dense_slow[2] && !dense_slow[3];
        Ok(Self {
            w,
            _cb: cb,
            tab_up,
            tab_down,
            tab_dense,
            has_slow,
            routed_kfast,
            has_absent,
            dense_slow,
            dense_kfast,
            dense_f16,
            dense_fast,
            gate_dt,
            own: None,
        })
    }

    /// The tables and a scratch of its own, sized for up to `max_rows` tokens.
    pub fn new(w: Weights, max_rows: usize) -> Result<Self, CudaError> {
        assert!(max_rows >= 1, "max_rows must be at least 1");
        let mut m = Self::tables(w)?;
        m.own = Some(MoeScratch::new(max_rows)?);
        Ok(m)
    }

    /// The router gate's form on the device (which router kernels run).
    pub fn gate_dtype(&self) -> GateDtype {
        self.gate_dt
    }

    /// Which kernels the fast variant launches for this block's projections (for logs and tests): the routed experts' fast kernel and
    /// whether the reference launches run beside it, then fc1, fc2, shared up and shared down, and whether the shared expert may take
    /// the side stream. For example `routed nm2_gemmk + reference; fc1 nm2_dense16, fc2 nm2_dense16, shared up nm2_gemmk, shared
    /// down nm2_gemm4; side stream`.
    pub fn launch_summary(&self) -> String {
        let routed = if self.routed_kfast { "nm2_gemmk" } else { "nm2_gemm4" };
        let slow = if self.has_slow { " + reference" } else { "" };
        let dense: Vec<String> = ["fc1", "fc2", "shared up", "shared down"]
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let k = if self.dense_f16[i] != 0 {
                    "nm2_dense16"
                } else if self.dense_slow[i] {
                    "reference"
                } else if self.dense_kfast[i] {
                    "nm2_gemmk"
                } else {
                    "nm2_gemm4"
                };
                format!("{name} {k}")
            })
            .collect();
        let side = if self.dense_fast { "side stream" } else { "in line" };
        format!("routed {routed}{slow}; {}; {side}", dense.join(", "))
    }

    /// Routed experts with weights on the device (an expert with both projections loaded): all 512 on one node, a rank's half under expert
    /// parallelism (the others are [`Proj::Absent`]). What a memory account of the block's weights counts.
    pub fn present_experts(&self) -> usize {
        self.w.up.iter().zip(&self.w.down).filter(|(u, d)| !matches!(u, Proj::Absent) && !matches!(d, Proj::Absent)).count()
    }

    /// Device bytes of the block's weights as held: the router (the gate as held, bf16 or f32; the f32 bias) and every projection (a bf16 weight, or an EXL3
    /// trellis with its scales and bias as stored here). Absent experts hold nothing.
    pub fn weight_bytes(&self) -> u64 {
        let proj = |p: &Proj| -> u64 {
            match p {
                Proj::Exl3 { trellis, suh, svh, bias, .. } => {
                    (trellis.bytes() + suh.bytes() + svh.bytes() + bias.as_ref().map_or(0, |b| b.bytes())) as u64
                }
                Proj::Bf16 { w, .. } | Proj::F16 { w, .. } => w.bytes() as u64,
                Proj::Absent => 0,
            }
        };
        let w = &self.w;
        let dense = [&w.fc1, &w.fc2, &w.shared_up, &w.shared_down].into_iter().map(proj).sum::<u64>();
        (w.gate.bytes() + w.router_bias.bytes()) as u64 + dense + w.up.iter().chain(&w.down).map(proj).sum::<u64>()
    }

    /// Whether any projection of the block (a dense one or a routed expert's) is EXL3.
    pub fn has_exl3(&self) -> bool {
        let w = &self.w;
        [&w.fc1, &w.fc2, &w.shared_up, &w.shared_down]
            .into_iter()
            .chain(&w.up)
            .chain(&w.down)
            .any(|p| matches!(p, Proj::Exl3 { .. }))
    }

    /// The row limit of the owned scratch, or 0 for a tables-only block.
    pub fn max_rows(&self) -> usize {
        self.own.as_ref().map_or(0, MoeScratch::max_rows)
    }

    /// Runs `f` with the owned scratch taken out of `self`, so `f` can borrow both.
    fn with_own<T>(&mut self, f: impl FnOnce(&Self, &mut MoeScratch) -> T) -> T {
        let mut s = self.own.take().expect("this block has no scratch of its own: build it with `new`, or use the `_in` methods");
        let r = f(self, &mut s);
        self.own = Some(s);
        r
    }

    /// Router: fp32 logits of `h` (`[rows, 4096]`), then the top-22 picks and weights of each row, and the expert buckets, into `s`.
    pub fn route_in(&self, s: &mut MoeScratch, h: &DevBuf, rows: usize, st: &Stream) -> Result<(), CudaError> {
        self.route_variant(Variant::Fast, s, h, rows, st)
    }

    /// `route_in` on the original kernels (`gb_nm_router_logits`, `gb_nm_router_select`; the buckets are `gb_nm_bucket`'s, separately).
    fn route_variant(&self, v: Variant, s: &MoeScratch, h: &DevBuf, rows: usize, st: &Stream) -> Result<(), CudaError> {
        s.check_rows(rows);
        assert!(h.bytes() >= rows * HIDDEN * 4, "h is short for {rows} rows");
        match v {
            Variant::Reference => {
                // Decode windows (one row) keep one row in flight; the chain of each row is the same whatever the block carries.
                let rb = if rows == 1 {
                    1
                } else if rows <= 4 {
                    4
                } else {
                    16
                };
                let logits = match self.gate_dt {
                    GateDtype::Bf16 => ffi::gb_nm_router_logits,
                    GateDtype::F16 => ffi::gb_nm_router_logits_g16,
                    GateDtype::F32 => ffi::gb_nm_router_logits_g32,
                };
                ok(unsafe {
                    logits(
                        addr(h) as *const c_void,
                        addr(&self.w.gate) as *const c_void,
                        addr(&s.logits) as *mut c_void,
                        rows as i32,
                        HIDDEN as i32,
                        EXPERTS as i32,
                        rb,
                        st.raw(),
                    )
                })?;
                ok(unsafe {
                    ffi::gb_nm_router_select(
                        addr(&s.logits) as *const c_void,
                        addr(&self.w.router_bias) as *const c_void,
                        self.w.scale,
                        addr(&s.pick) as *mut c_void,
                        addr(&s.wts) as *mut c_void,
                        rows as i32,
                        st.raw(),
                    )
                })
            }
            Variant::Fast => {
                // the layout of fast_router_layout: every layout gives the same bits (a (row, expert) chain is the same wherever it runs)
                unsafe {
                    fast_router_logits(
                        addr(h) as *const c_void,
                        addr(&self.w.gate) as *const c_void,
                        self.gate_dt,
                        addr(&s.logits) as *mut c_void,
                        rows,
                        st.raw(),
                    )
                }?;
                self.select(s, rows, st)
            }
        }
    }

    /// `route_in` on the owned scratch.
    pub fn route(&mut self, h: &DevBuf, rows: usize, st: &Stream) -> Result<(), CudaError> {
        self.with_own(|m, s| m.route_in(s, h, rows, st))
    }

    /// Selection, weights and the expert buckets from the logits in the scratch, in one launch.
    fn select(&self, s: &MoeScratch, rows: usize, st: &Stream) -> Result<(), CudaError> {
        ok(unsafe {
            ffi::gb_nm2_select(
                addr(&s.logits) as *const c_void,
                addr(&self.w.router_bias) as *const c_void,
                self.w.scale,
                addr(&s.pick) as *mut c_void,
                addr(&s.wts) as *mut c_void,
                rows as i32,
                rows as i32,
                addr(&s.cnt) as *mut c_void,
                addr(&s.lists) as *mut c_void,
                addr(&s.active) as *mut c_void,
                addr(&s.nact) as *mut c_void,
                addr(&s.ctr) as *mut c_void,
                st.raw(),
            )
        })
    }

    /// Selection from caller-supplied fp32 logits `[rows, 512]`, with the same selection and weights as `route` (for tests).
    pub fn route_from_logits(&mut self, logits: &DevBuf, rows: usize, st: &Stream) -> Result<(), CudaError> {
        self.with_own(|m, s| {
            s.check_rows(rows);
            assert!(logits.bytes() >= rows * EXPERTS * 4, "logits are short for {rows} rows");
            s.logits.copy_from(logits, rows * EXPERTS * 4, st)?;
            m.select(s, rows, st)
        })
    }

    /// The router's logits of the last `route` (`[rows, 512]`) on the host. Synchronises.
    pub fn download_logits(&self, rows: usize, st: &Stream) -> Result<Vec<f32>, CudaError> {
        self.own.as_ref().expect("no owned scratch").download_logits(rows, st)
    }

    /// The picks `[rows, 22]` (in rank order) and weights of the last `route` on the host. Synchronises.
    pub fn download_route(&self, rows: usize, st: &Stream) -> Result<(Vec<i32>, Vec<f32>), CudaError> {
        self.own.as_ref().expect("no owned scratch").download_route(rows, st)
    }

    /// The full block: `out = routed + shared` for `rows` tokens of `h`, using the scratch `s`. Nothing is synchronised.
    pub fn forward_in(
        &self,
        s: &mut MoeScratch,
        h: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.forward_staged(s, h, rows, out, st, &mut |_| Ok(()))
    }

    /// `forward_in` on the reference launches (the original kernels, one launch per step of the arithmetic). The bits are the same as
    /// `forward_in`'s; this is the oracle the scheduled kernels are gated against, and the baseline of the timing tests.
    pub fn forward_reference(
        &self,
        s: &mut MoeScratch,
        h: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.forward_variant(Variant::Reference, s, h, rows, out, st, &mut |_| Ok(()))
    }

    /// `forward_in` for one rank of an expert-parallel (TP-exact) group: exactly `forward_in`'s launches, plus one call of `reduce` on `st`
    /// between the routed down projection and the combine (see [`ReduceHook`]). With the rank's non-owned experts absent, `y` holds this
    /// rank's pairs and exact zeros for the others; `reduce` sums it over the ranks (disjoint support: `x + 0.0 == x`), and the
    /// combine, fc2 and the shared expert then run on the full `y` and give the bits of the whole block (`docs/NEMOTRON_TP.md`). With a
    /// no-op `reduce` this is `forward_in`, bit for bit. Nothing is synchronised, so it captures into a CUDA graph if `reduce` does.
    pub fn forward_in_tp(
        &self,
        s: &mut MoeScratch,
        h: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        st: &Stream,
        reduce: &mut ReduceHook,
    ) -> Result<(), CudaError> {
        self.forward_variant_reduce(Variant::Fast, s, h, rows, out, st, &mut |_| Ok(()), Some(reduce))
    }

    /// `forward_in_tp` on the reference launches (the oracle of the fast ones, as `forward_reference` is of `forward_in`).
    pub fn forward_reference_in_tp(
        &self,
        s: &mut MoeScratch,
        h: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        st: &Stream,
        reduce: &mut ReduceHook,
    ) -> Result<(), CudaError> {
        self.forward_variant_reduce(Variant::Reference, s, h, rows, out, st, &mut |_| Ok(()), Some(reduce))
    }

    /// `forward_in`, calling `mark(stage)` after the launches of each stage are queued on `st` (for timing: the caller records an event
    /// in `mark`). The marks do not change a launch, so the bits are those of `forward_in`.
    pub fn forward_staged(
        &self,
        s: &mut MoeScratch,
        h: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        st: &Stream,
        mark: &mut dyn FnMut(Stage) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        self.forward_variant(Variant::Fast, s, h, rows, out, st, mark)
    }

    /// `forward_staged` on the reference launches.
    pub fn forward_reference_staged(
        &self,
        s: &mut MoeScratch,
        h: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        st: &Stream,
        mark: &mut dyn FnMut(Stage) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        self.forward_variant(Variant::Reference, s, h, rows, out, st, mark)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_variant(
        &self,
        v: Variant,
        s: &mut MoeScratch,
        h: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        st: &Stream,
        mark: &mut dyn FnMut(Stage) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        self.forward_variant_reduce(v, s, h, rows, out, st, mark, None)
    }

    /// `forward_variant` with an optional reduce hook between the routed down projection and the combine (`None`: not a launch, not a
    /// branch of the device work: the launches are exactly the single-node block's).
    #[allow(clippy::too_many_arguments)]
    fn forward_variant_reduce(
        &self,
        v: Variant,
        s: &mut MoeScratch,
        h: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        st: &Stream,
        mark: &mut dyn FnMut(Stage) -> Result<(), CudaError>,
        reduce: Option<&mut ReduceHook>,
    ) -> Result<(), CudaError> {
        s.check_rows(rows);
        assert!(h.bytes() >= rows * HIDDEN * 4 && out.bytes() >= rows * HIDDEN * 4, "h or out is short for {rows} rows");
        let p = rows * TOPK;
        self.route_variant(v, s, h, rows, st)?;
        mark(Stage::Router)?;
        // Each expert's pairs, ascending (the lists have a stride of `rows`). The fast route has built them (and the list of the experts
        // that have pairs) already; the reference one is followed by the original bucket kernel.
        if v == Variant::Reference {
            ok(unsafe {
                ffi::gb_nm_bucket(
                    addr(&s.pick) as *const c_void,
                    p as i32,
                    rows as i32,
                    addr(&s.cnt) as *mut c_void,
                    addr(&s.lists) as *mut c_void,
                    st.raw(),
                )
            })?;
        }
        mark(Stage::Bucket)?;
        let (pick, cnt, lists) = (addr(&s.pick), addr(&s.cnt), addr(&s.lists));
        let (active, nact) = if v == Variant::Fast { (addr(&s.active), addr(&s.nact)) } else { (0, 0) };
        let (xh, partial) = (addr(&s.xh), addr(&s.partial));
        let (a, y, xlat, routed, routed_out, sh_a) =
            (addr(&s.a), addr(&s.y), addr(&s.xlat), addr(&s.routed), addr(&s.routed_out), addr(&s.sh_a));
        let (tab_up, tab_down) = (addr(&self.tab_up), addr(&self.tab_down));
        let h_addr = addr(h);
        let out_addr = addr(out);
        // The groups the grid of an expert projection spans: every expert (reference), or as many as there can be active ones (fast).
        let zgroups = if v == Variant::Fast { EXPERTS.min(p) } else { EXPERTS };

        // The shared expert, relu2(up(h)) then down, depends on h only. The fast variant runs it first, on the scratch's side stream when
        // it can (beside the router and the routed experts, which are latency-bound in places the shared weights' streaming can fill),
        // and the output side of fc2 adds it: shared + routed is the sum the reference variant forms as routed + shared, the same bits.
        let side = v == Variant::Fast && self.dense_fast && rows <= SIDE_ROWS && SIDE_STREAM.load(Ordering::Relaxed);
        if v == Variant::Fast {
            let sh_stream = if side { &s.aux } else { st };
            if side {
                fork(st, &s.aux, &s.ev_fork)?;
            }
            let (part, tk) = if side { (addr(&s.sh_partial), addr(&s.sh_tickets)) } else { (partial, addr(&s.tickets)) };
            self.dense_in(v, 2, h_addr, rows, sh_a, true, None, (part, tk, xh), sh_stream)?;
            self.dense_in(v, 3, sh_a, rows, addr(&s.sh_out), false, None, (part, tk, xh), sh_stream)?;
        }

        // fc1: the latent x = fc1(h).
        self.dense(v, s, 0, h_addr, rows, xlat, false, None, st)?;
        mark(Stage::LatentIn)?;
        // Routed experts, up: relu2(up(x)) on each pair's own expert.
        let up = ProjRun {
            src: xlat,
            div: TOPK,
            eor: pick,
            rows: p,
            lists,
            cnt,
            active,
            nact,
            fixed: 0,
            lcap: rows,
            tab: tab_up,
            groups: EXPERTS,
            zgroups,
            k: LATENT,
            n: EXPERT_INTER,
            partial,
            cap: p,
            xh,
            out: a,
            relu2: true,
            add: 0,
            tickets: addr(&s.tickets),
            npass: rows.div_ceil(8),
            slow: self.has_slow,
            absent: self.has_absent,
            kfast: self.routed_kfast || ANY_WIDTH.load(Ordering::Relaxed),
            f16w: 0,
        };
        self.run_proj(v, &up, st)?;
        mark(Stage::Up)?;
        // Routed experts, down, then the weighted sum of each token's 22 slots in slot order.
        let down = ProjRun { src: a, div: 1, eor: pick, tab: tab_down, k: EXPERT_INTER, n: LATENT, out: y, relu2: false, ..up };
        self.run_proj(v, &down, st)?;
        mark(Stage::Down)?;
        // Expert parallel: the caller sums the y buffer over the ranks here, on the main stream (the shared expert, if it runs on the side
        // stream, depends on `h` only and is joined below, so the hook needs nothing from it and it nothing from the hook).
        if let Some(reduce) = reduce {
            reduce(&mut s.y, p * LATENT, st)?;
        }
        ok(unsafe {
            ffi::gb_nm_combine(
                y as *const c_void,
                addr(&s.wts) as *const c_void,
                rows as i32,
                LATENT as i32,
                routed as *mut c_void,
                st.raw(),
            )
        })?;
        mark(Stage::Combine)?;
        if v == Variant::Fast {
            // fc2 back to the hidden width, plus the shared expert's output: the block's output.
            if side {
                join(st, &s.aux, &s.ev_join)?;
            }
            self.dense(v, s, 1, routed, rows, out_addr, false, Some(addr(&s.sh_out)), st)?;
            mark(Stage::Fc2)?;
        } else {
            // fc2 back to the hidden width, then the shared expert on h with the routed sum added by its output side.
            self.dense(v, s, 1, routed, rows, routed_out, false, None, st)?;
            mark(Stage::Fc2)?;
            self.dense(v, s, 2, h_addr, rows, sh_a, true, None, st)?;
            self.dense(v, s, 3, sh_a, rows, out_addr, false, Some(routed_out), st)?;
        }
        mark(Stage::Shared)?;
        Ok(())
    }

    /// `forward_in` on the owned scratch.
    pub fn forward(&mut self, h: &DevBuf, rows: usize, out: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
        self.with_own(|m, s| m.forward_in(s, h, rows, out, st))
    }

    /// fc1 alone: `out = fc1(h)` for `rows` tokens (`out` is `[rows, 1024]`). For tests of the latent projection.
    pub fn latent_in(&mut self, h: &DevBuf, rows: usize, out: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
        self.with_own(|m, s| {
            s.check_rows(rows);
            m.dense(Variant::Fast, s, 0, addr(h), rows, addr(out), false, None, st)
        })
    }

    /// One expert alone on latent rows `x` (`[rows, 1024]`): `out = down(relu2(up(x)))` (`[rows, 1024]`). For tests of one expert.
    pub fn expert(&mut self, e: usize, x: &DevBuf, rows: usize, out: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
        self.with_own(|m, s| {
            s.check_rows(rows);
            assert!(e < EXPERTS, "expert {e} is out of range");
            let up = addr(&m.tab_up) + e * PROJ_BYTES;
            let down = addr(&m.tab_down) + e * PROJ_BYTES;
            let (xh, a, partial) = (addr(&s.xh), addr(&s.a), addr(&s.partial));
            m.t1(addr(x), 1, 0, up, rows, LATENT, xh, 0, st)?;
            m.gemm(xh, 0, 0, rows, 0, up, 1, LATENT, EXPERT_INTER, partial, rows, 0, (0, 0), st)?;
            m.t2(partial, slices(LATENT), rows, 0, up, rows, EXPERT_INTER, true, 0, a, 0, st)?;
            m.t1(a, 1, 0, down, rows, EXPERT_INTER, xh, 0, st)?;
            m.gemm(xh, 0, 0, rows, 0, down, 1, EXPERT_INTER, LATENT, partial, rows, 0, (0, 0), st)?;
            m.t2(partial, slices(EXPERT_INTER), rows, 0, down, rows, LATENT, false, 0, addr(out), 0, st)
        })
    }

    /// Decodes one projection's EXL3 weights to fp16 bits `[K, N]` (row-major) on the device, then to the host. Synchronises.
    pub fn decode(&self, t: Table, st: &Stream) -> Result<Vec<u16>, CudaError> {
        let (tab, k, n) = match t {
            Table::Up(e) => (addr(&self.tab_up) + e * PROJ_BYTES, LATENT, EXPERT_INTER),
            Table::Down(e) => (addr(&self.tab_down) + e * PROJ_BYTES, EXPERT_INTER, LATENT),
            Table::Dense(0) => (addr(&self.tab_dense), HIDDEN, LATENT),
            Table::Dense(1) => (addr(&self.tab_dense) + PROJ_BYTES, LATENT, HIDDEN),
            Table::Dense(2) => (addr(&self.tab_dense) + 2 * PROJ_BYTES, HIDDEN, SHARED_INTER),
            Table::Dense(_) => (addr(&self.tab_dense) + 3 * PROJ_BYTES, SHARED_INTER, HIDDEN),
        };
        let out = DevBuf::zeroed(k * n * 2)?;
        ok(unsafe { ffi::gb_nm_decode(tab as *const c_void, 0, out.ptr(), st.raw()) })?;
        st.sync()?;
        out.to_vec::<u16>()
    }

    // ---- stage helpers: one kernel launch each, on device addresses ----

    /// A dense projection `which` (0 fc1, 1 fc2, 2 shared up, 3 shared down) on `rows` rows at `x`, into `out` (plus `add` when given),
    /// with relu2 when asked. Partial sums go to the scratch `s`.
    #[allow(clippy::too_many_arguments)]
    fn dense(
        &self,
        v: Variant,
        s: &MoeScratch,
        which: usize,
        x: usize,
        rows: usize,
        out: usize,
        relu2: bool,
        add: Option<usize>,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.dense_in(v, which, x, rows, out, relu2, add, (addr(&s.partial), addr(&s.tickets), addr(&s.xh)), st)
    }

    /// `dense` with the partial-sum, ticket and reference-input buffers named (the shared expert's own, on the side stream).
    #[allow(clippy::too_many_arguments)]
    fn dense_in(
        &self,
        v: Variant,
        which: usize,
        x: usize,
        rows: usize,
        out: usize,
        relu2: bool,
        add: Option<usize>,
        bufs: (usize, usize, usize),
        st: &Stream,
    ) -> Result<(), CudaError> {
        let (k, n) = match which {
            0 => (HIDDEN, LATENT),
            1 => (LATENT, HIDDEN),
            2 => (HIDDEN, SHARED_INTER),
            _ => (SHARED_INTER, HIDDEN),
        };
        let tab = addr(&self.tab_dense) + which * PROJ_BYTES;
        let (partial, tickets, xh) = bufs;
        let run = ProjRun {
            src: x,
            div: 1,
            eor: 0,
            rows,
            lists: 0,
            cnt: 0,
            active: 0,
            nact: 0,
            fixed: rows,
            lcap: 0,
            tab,
            groups: 1,
            zgroups: 1,
            k,
            n,
            partial,
            cap: rows,
            xh,
            out,
            relu2,
            add: add.unwrap_or(0),
            tickets,
            npass: 1,
            slow: self.dense_slow[which],
            absent: false,
            kfast: self.dense_kfast[which] || (ANY_WIDTH.load(Ordering::Relaxed) && !self.dense_slow[which]),
            f16w: self.dense_f16[which],
        };
        self.run_proj(v, &run, st)
    }

    /// One projection with its output side: the sum of the partial slices, H, the output scale and bias, relu2 and the addend, into
    /// `r.out`. Reference: rotate every row into `xh`, the reference GEMM, then `nm_t2`. Fast: the 4-bit Mul1 projections rotate in the
    /// kernel's prologue and the last block of each column block does the output side; any other EXL3 projection (and every bf16 one,
    /// and an absent expert) still takes the reference input side, GEMM and output side, which skip the fast ones.
    fn run_proj(&self, v: Variant, r: &ProjRun, st: &Stream) -> Result<(), CudaError> {
        if v == Variant::Fast && r.f16w != 0 {
            // a dense fp16 projection: one launch with the reference launches' bits
            return ok(unsafe {
                ffi::gb_nm2_dense16(
                    r.src as *const c_void,
                    r.f16w as *const c_void,
                    r.fixed as i32,
                    r.k as i32,
                    r.n as i32,
                    r.out as *mut c_void,
                    r.relu2 as i32,
                    r.add as *const c_void,
                    st.raw(),
                )
            });
        }
        if v == Variant::Fast {
            let launch = if r.kfast { ffi::gb_nm2_gemmk } else { ffi::gb_nm2_gemm4 };
            ok(unsafe {
                launch(
                    r.src as *const c_void,
                    r.div as i32,
                    r.k as i32,
                    r.lists as *const c_void,
                    r.cnt as *const c_void,
                    r.active as *const c_void,
                    r.nact as *const c_void,
                    r.fixed as i32,
                    r.lcap as i32,
                    r.tab as *const c_void,
                    // z: the groups with rows (routed), or the chunks of rows (one dense projection)
                    if r.lists != 0 { r.zgroups } else { r.fixed.div_ceil(DENSE_CHUNK_ROWS) } as i32,
                    r.k as i32,
                    r.n as i32,
                    slices(r.k) as i32,
                    r.partial as *mut c_void,
                    r.cap as i32,
                    r.out as *mut c_void,
                    r.relu2 as i32,
                    r.add as *const c_void,
                    r.tickets as *mut c_void,
                    r.npass as i32,
                    st.raw(),
                )
            })?;
            if !r.slow && !r.absent {
                return Ok(());
            }
        }
        let fast = v == Variant::Fast;
        // what the reference launches skip: nothing (reference variant), the entries nm2_gemm4 ran (1) or nm2_gemmk ran (2)
        let skip = if !fast {
            0
        } else if r.kfast {
            2
        } else {
            1
        };
        if !fast || r.slow {
            self.t1(r.src, r.div, r.eor, r.tab, r.rows, r.k, r.xh, skip, st)?;
            // The reference GEMM over the compacted list of groups with rows (the fast variant's), or over every group.
            let groups = if fast && r.lists != 0 { r.zgroups } else { r.groups };
            self.gemm(
                r.xh,
                r.lists,
                r.cnt,
                r.fixed,
                r.lcap,
                r.tab,
                groups,
                r.k,
                r.n,
                r.partial,
                r.cap,
                skip,
                (r.active, r.nact),
                st,
            )?;
        }
        self.t2(r.partial, slices(r.k), r.cap, r.eor, r.tab, r.rows, r.n, r.relu2, r.add, r.out, skip, st)
    }

    /// Input side: `dst[r] = H(src[r / div] * suh[e(r)])` for EXL3, `src[r / div]` for bf16. e(r) is `eor[r]`, or entry 0 when `eor` is 0.
    #[allow(clippy::too_many_arguments)]
    fn t1(
        &self,
        src: usize,
        div: usize,
        eor: usize,
        tab: usize,
        rows: usize,
        k: usize,
        dst: usize,
        skip_fast: i32,
        st: &Stream,
    ) -> Result<(), CudaError> {
        ok(unsafe {
            ffi::gb_nm_t1(
                src as *const c_void,
                div as i32,
                eor as *const c_void,
                tab as *const c_void,
                rows as i32,
                k as i32,
                dst as *mut c_void,
                skip_fast,
                st.raw(),
            )
        })
    }

    /// The projection of `groups` table entries: group g's rows are `lists[g * lcap ..]` with `cnt[g]` of them, or rows `0..fixed` when
    /// `lists` is 0. Both kernels are launched; each skips the groups of the other format. `cap` is the row count of a partial slice.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        xin: usize,
        lists: usize,
        cnt: usize,
        fixed: usize,
        lcap: usize,
        tab: usize,
        groups: usize,
        k: usize,
        n: usize,
        partial: usize,
        cap: usize,
        skip_fast: i32,
        act: (usize, usize),
        st: &Stream,
    ) -> Result<(), CudaError> {
        let (xin, lists, cnt, tab) = (xin as *const c_void, lists as *const c_void, cnt as *const c_void, tab as *const c_void);
        ok(unsafe {
            ffi::gb_nm_gemm_exl3(
                xin,
                lists,
                cnt,
                fixed as i32,
                lcap as i32,
                tab,
                groups as i32,
                k as i32,
                n as i32,
                slices(k) as i32,
                partial as *mut c_void,
                cap as i32,
                skip_fast,
                act.0 as *const c_void,
                act.1 as *const c_void,
                st.raw(),
            )
        })?;
        ok(unsafe {
            ffi::gb_nm_gemv_bf16(
                xin,
                lists,
                cnt,
                fixed as i32,
                lcap as i32,
                tab,
                groups as i32,
                k as i32,
                n as i32,
                partial as *mut c_void,
                cap as i32,
                act.0 as *const c_void,
                act.1 as *const c_void,
                st.raw(),
            )
        })
    }

    /// Output side: the sum of the `s` partial slices (EXL3) or slot 0 (bf16), then H, the output scale and bias for EXL3; then relu2
    /// when `relu2`, and + `add[r, n]` when `add` is nonzero. `eor` and `tab` as in `t1`.
    #[allow(clippy::too_many_arguments)]
    fn t2(
        &self,
        partial: usize,
        s: usize,
        cap: usize,
        eor: usize,
        tab: usize,
        rows: usize,
        n: usize,
        relu2: bool,
        add: usize,
        out: usize,
        skip_fast: i32,
        st: &Stream,
    ) -> Result<(), CudaError> {
        ok(unsafe {
            ffi::gb_nm_t2(
                partial as *const c_void,
                s as i32,
                cap as i32,
                eor as *const c_void,
                tab as *const c_void,
                rows as i32,
                n as i32,
                relu2 as i32,
                add as *const c_void,
                out as *mut c_void,
                skip_fast,
                st.raw(),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_divide_the_k_tiles() {
        for k in [1024usize, 2688, 4096, 5376] {
            let s = slices(k);
            assert_eq!((k / 16) % s, 0);
        }
        assert_eq!(slices(4096), 16);
        assert_eq!(slices(1024), 4);
        assert_eq!(slices(2688), 8);
        assert_eq!(slices(5376), 16);
    }

    #[test]
    fn the_scratch_at_64_rows_stays_about_100_mb() {
        // gb-model's memory model prices one MoeScratch at 64 rows and asserts it is about 100 MB (90 to 110): the side stream's buffers
        // are sized for SIDE_ROWS rows, not for the whole block, to stay inside that.
        let mb = MoeScratch::bytes(64) as f64 / 1e6;
        assert!((90.0..=110.0).contains(&mb), "{mb} MB");
        assert!(MoeScratch::bytes(1) < MoeScratch::bytes(64));
    }

    #[test]
    fn the_device_struct_is_56_bytes() {
        assert_eq!(core::mem::size_of::<NmProj>(), PROJ_BYTES);
    }
}
