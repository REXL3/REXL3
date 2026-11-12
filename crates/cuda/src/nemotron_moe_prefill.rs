//! Large-row prefill path of the Nemotron 3.5 Super VL latent-MoE block (`csrc/nemotron_moe_prefill.cu`, package P1).
//!
//! The decode path (`nemotron_moe::NemotronMoe::forward_in`, up to 64 rows) reads each expert's weights once per call, so a 64-row
//! prefill slice pays about 94% of a layer's expert weights (1.35 GB at 4 bits) for 64 rows of work. This path takes a slice of 256 rows
//! or more: an expert's weights are decoded once for all of its rows (up to 32 in one block), and the rows are rotated once per projection,
//! so the weights are read once per slice. A row's bits are the decode path's, whatever the row count and position: the per-row
//! arithmetic (every k chain, slice sum, rotation and output step) is the same operation sequence, gated bit for bit against
//! `forward_in` in `tests/nemotron_moe_prefill.rs`. `docs/NEMOTRON_MOE_PREFILL.md` has the design and the numbers.
//!
//! [`PrefillMoe`] wraps a [`NemotronMoe`] (it dereferences to it, so every decode-path method is there) and adds the device tables the
//! prefill kernels read; build it from the same [`Weights`]. [`PrefillScratch`] is the prefill path's scratch (`max_rows` up to a few
//! hundred; bytes are reported by [`PrefillScratch::bytes`]). Everything is launched on the stream passed in and nothing synchronises.

use crate::graph::{fork, join};
use crate::nemotron_moe::{
    slices, NemotronMoe, Proj, ReduceHook, Weights, EXPERTS, EXPERT_INTER, HIDDEN, LATENT, SHARED_INTER, TOPK,
};
use crate::rt::Event;
use crate::{CudaError, DevBuf, Stream};
use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use gb_exl3::Codebook;

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_nm_combine(y: *const c_void, wts: *const c_void, rows: i32, n: i32, routed: *mut c_void, s: *mut c_void) -> i32;

        pub fn gb_nmp_zero_absent(lists: *const c_void, lcap: i32, work: *const c_void, nwork: *const c_void, tab: *const c_void, n: i32, out: *mut c_void, blocks: i32, s: *mut c_void) -> i32;

        pub fn gb_nmp_select(logits: *const c_void, bias: *const c_void, scale: f32, pick: *mut c_void, wts: *mut c_void, rows: i32, s: *mut c_void) -> i32;

        pub fn gb_nmp_bucket(
            pick: *const c_void,
            rows: i32,
            cap: i32,
            tab_up: *const c_void,
            tab_down: *const c_void,
            cnt: *mut c_void,
            lists: *mut c_void,
            work: *mut c_void,
            stride: i32,
            nwork: *mut c_void,
            mode: i32,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nmp_bucket_k(
            pick: *const c_void,
            rows: i32,
            cap: i32,
            tab_up: *const c_void,
            tab_down: *const c_void,
            cnt: *mut c_void,
            lists: *mut c_void,
            work: *mut c_void,
            stride: i32,
            nwork: *mut c_void,
            mode: i32,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nmp_rotate(src: *const c_void, ksrc: i32, div: i32, pick: *const c_void, tab: *const c_void, rows: i32, k: i32, out: *mut c_void, s: *mut c_void) -> i32;

        pub fn gb_nmp_rotate_k(src: *const c_void, ksrc: i32, div: i32, pick: *const c_void, tab: *const c_void, rows: i32, k: i32, out: *mut c_void, s: *mut c_void) -> i32;

        pub fn gb_nmp_gemmk(
            src: *const c_void,
            ksrc: i32,
            lists: *const c_void,
            lcap: i32,
            work: *const c_void,
            nwork: *const c_void,
            dense_rows: i32,
            zitems: i32,
            tab: *const c_void,
            ntab: *const c_void,
            k: i32,
            n: i32,
            sl: i32,
            out: *mut c_void,
            relu2: i32,
            add: *const c_void,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nmp_gemm(
            src: *const c_void,
            ksrc: i32,
            lists: *const c_void,
            lcap: i32,
            work: *const c_void,
            nwork: *const c_void,
            dense_rows: i32,
            zitems: i32,
            rbmax: i32,
            tab: *const c_void,
            ntab: *const c_void,
            k: i32,
            n: i32,
            sl: i32,
            out: *mut c_void,
            relu2: i32,
            add: *const c_void,
            s: *mut c_void,
        ) -> i32;

        pub fn gb_nmp_slow(
            src: *const c_void,
            ksrc: i32,
            div: i32,
            lists: *const c_void,
            lcap: i32,
            work: *const c_void,
            nwork: *const c_void,
            dense_rows: i32,
            zitems: i32,
            tab: *const c_void,
            k: i32,
            n: i32,
            out: *mut c_void,
            relu2: i32,
            add: *const c_void,
            variant: i32,
            s: *mut c_void,
        ) -> i32;
    }
}

/// `struct PfProj` of the kernel file (the same 56 bytes as `nemotron_moe.cu`'s `NmProj`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PfProj {
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

const PROJ_BYTES: usize = 56;
const FMT_EXL3: i32 = 0;
const FMT_BF16: i32 = 1;
const FMT_ABSENT: i32 = 2;
const FMT_F16: i32 = 3;

/// Rows a small-class block carries (an expert with at most this many rows is one block's item) and the most a block carries.
pub const CLASS_ROWS: usize = 16;
pub const BLOCK_ROWS: usize = 32;

static SIDE_STREAM: AtomicBool = AtomicBool::new(true);
/// The routed projections' schedule: 0 two launches by class, 1 one launch of items of up to 32 rows, 2 one launch of items of up to 16.
static SCHEDULE: AtomicU8 = AtomicU8::new(DEFAULT_SCHEDULE);
/// Schedule 1 is the one the bit gates and the model's gates have run on; schedule 2 (items of up to 16 rows) and the four-column bf16
/// kernel below are built, and wait for their GPU runs (docs/NEMOTRON_MOE_PREFILL.md, "Pending GPU runs").
const DEFAULT_SCHEDULE: u8 = 1;
/// The bf16 projections' kernel: 0 a warp a column (`pf_slow`), 1 four columns a warp (`pf_slow4`).
static BF16_KERNEL: AtomicU8 = AtomicU8::new(0);

/// Run the shared expert on the scratch's side stream beside the routed chain (the default), or in line. The bits are the same; this
/// exists to time the difference. Process-wide, not for use while a block is running.
pub fn set_side_stream(on: bool) {
    SIDE_STREAM.store(on, Ordering::Relaxed);
}

/// The routed projections' blocks in one launch, the experts with many rows and the ones with few alternating in the grid (the default:
/// the compute-bound blocks of the first and the memory-bound blocks of the second then share the SMs), or in two launches by class, the
/// large class first. The bits are the same; this exists to time the difference. Process-wide, not for use while a block is running.
pub fn set_interleave(on: bool) {
    SCHEDULE.store(if on { DEFAULT_SCHEDULE } else { 0 }, Ordering::Relaxed);
}

/// The kernel of the bf16 projections (routed bf16 experts and bf16 dense projections): 0 `pf_slow`, one warp a column, rows 4 at a
/// time; 1 `pf_slow4`, four columns and up to 8 rows a warp, the 32 xor trees of a pass done as a reduce-scatter. The bits are the same.
/// Process-wide, not for use while a block is running.
pub fn set_bf16_kernel(variant: u8) {
    assert!(variant <= 1, "bf16 kernel {variant}");
    BF16_KERNEL.store(variant, Ordering::Relaxed);
}

/// The schedule by number: 0 two launches by class, 1 one launch with items of up to 32 rows (the 32-row kernel, 254 registers), 2 one
/// launch with items of up to 16 rows (the 16-row kernel, which holds more blocks an SM; an expert with more rows is two items).
pub fn set_schedule(mode: u8) {
    assert!(mode <= 2, "schedule {mode}");
    SCHEDULE.store(mode, Ordering::Relaxed);
}

/// The absent-expert fast path (lane S, `docs/NEMOTRON_PREFILL_SPEED.md`; ON by default since 2026-10-15, `0` is the opt-out): a layer whose slow experts
/// are all ABSENT (a TP rank's non-owned experts, no bf16 pair) skip the `pf_slow` launches, whose grid is `n_slow * ceil(rows / 32)`
/// items of `N / 8` blocks (about 950k blocks a layer at 256 rows with 256 absent experts, nearly all exiting at once), and write the
/// absent pairs' down output (+0.0, the bits `pf_slow` writes) with one small grid-stride kernel. The absent pairs' up activations
/// are not written: nothing reads them (the down projection of an absent expert reads no input). `0` (or off/false/no): the path as
/// before (`pf_slow` for every slow expert). The bits are the same either way.
pub const ABSENT_FAST_VAR: &str = "REXL3_NEMO_PREFILL_ABSENT_FAST";
/// 0 unset (read the environment on first use), 1 off, 2 on.
static ABSENT_FAST: AtomicU8 = AtomicU8::new(0);
/// Blocks of the absent-zero kernel: a few waves on the 48 SMs of a GB10.
const ZERO_BLOCKS: i32 = 192;

std::thread_local! {
    /// Lane S (P2 TP pair gather): the caller's reduce hook writes every absent expert's pair rows of `y` (the peer's rows, gathered),
    /// so the zero kernel for them is redundant. Set only around such a call on this thread.
    static GATHER_FILLS_ABSENT: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

std::thread_local! {
    /// Lane S (P2 TP pair gather): when set, the block's combine reads the peer's pair rows from the receive buffer through the map
    /// (`nemotron_tp::combine_gather`) instead of an unpacked `y`: (recv address, map address).
    static COMBINE_FROM: core::cell::Cell<Option<(usize, usize)>> = const { core::cell::Cell::new(None) };
}

std::thread_local! {
    /// Lane S (P2 TP pair gather): an event (address of a `rt::Event` the caller owns) the block records on its stream right after the
    /// router's buckets, so the gather hook can wait for the counts alone instead of the whole stream.
    static AFTER_BUCKET: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// See `AFTER_BUCKET` (0: none). The event must outlive the block call; the caller clears it after.
pub fn set_after_bucket_event(ev: Option<&crate::rt::Event>) {
    AFTER_BUCKET.with(|c| c.set(ev.map_or(0, |e| e as *const crate::rt::Event as usize)));
}

pub(crate) fn after_bucket_event() -> Option<&'static crate::rt::Event> {
    let p = AFTER_BUCKET.with(|c| c.get());
    // SAFETY: set_after_bucket_event's contract: the event outlives the block call that reads it
    (p != 0).then(|| unsafe { &*(p as *const crate::rt::Event) })
}

/// See `COMBINE_FROM`: set by the TP pair gather's hook, cleared by its caller after the block call.
pub fn set_combine_from(src: Option<(usize, usize)>) {
    COMBINE_FROM.with(|c| c.set(src));
}

pub(crate) fn combine_from() -> Option<(usize, usize)> {
    COMBINE_FROM.with(|c| c.get())
}

/// See `GATHER_FILLS_ABSENT`: set by the TP pair gather around its block call, cleared after.
pub fn set_gather_fills_absent(on: bool) {
    GATHER_FILLS_ABSENT.with(|c| c.set(on));
}

fn gather_fills_absent() -> bool {
    GATHER_FILLS_ABSENT.with(|c| c.get())
}

/// Lane S (P2 TP router row split, `docs/NEMOTRON_PREFILL_SPEED.md` 6.12): the byte exchange of the router logits between the two
/// ranks. Called by the tensor-core block right after it computed the logits of this rank's rows only: `(send, logits, own, peer, st)`
/// sends `own`'s rows of `send` (a handle to the same bytes as `logits`) and receives the peer's rows into `peer`'s rows of `logits`
/// (`[rows, 512]` f32, rows of 2,048 bytes), queued on `st`. A byte copy: the logits of a row do not depend on the other rows of the call
/// (6.12), so both ranks then hold the single node's whole `[rows, 512]` logits and select, bucket and combine on identical inputs.
pub type RouterExchange<'a> =
    dyn FnMut(&DevBuf, &mut DevBuf, core::ops::Range<usize>, core::ops::Range<usize>, &Stream) -> Result<(), CudaError> + 'a;

/// The smallest call that splits the router's rows (a smaller one computes all its rows on both ranks, as before: the exchange's
/// latency would cost more than the half of the router it saves). Both ranks see the same row count, so they agree.
pub const ROUTER_SPLIT_MIN_ROWS: usize = 32;

type RouterSlot = Option<(usize, *mut RouterExchange<'static>)>;

std::thread_local! {
    /// See [`with_router_split`]: (this rank, the exchange) while a block call inside it runs on this thread.
    static ROUTER_SPLIT: core::cell::Cell<RouterSlot> = const { core::cell::Cell::new(None) };
}

/// Runs `f` with the router row split on (`hook` given) for the tensor-core block calls `f` makes on this thread: `rank` (0 or 1 of
/// two) computes the logits of rows `[0, ceil(rows / 2))` (rank 0) or `[ceil(rows / 2), rows)` (rank 1) and `hook` exchanges them.
/// Calls of fewer than [`ROUTER_SPLIT_MIN_ROWS`] rows, and the tensor-core router (`set_router_mma`), are not split. `None`: just `f`.
/// Cleared when `f` returns or unwinds (the previous setting is restored).
pub fn with_router_split<R>(rank: usize, hook: Option<&mut RouterExchange<'_>>, f: impl FnOnce() -> R) -> R {
    let Some(hook) = hook else { return f() };
    assert!(rank < 2, "the router row split is for two ranks (rank {rank})");
    struct Restore(RouterSlot);
    impl Drop for Restore {
        fn drop(&mut self) {
            ROUTER_SPLIT.with(|c| c.set(self.0));
        }
    }
    // SAFETY: only the lifetime is erased; the pointer is dereferenced only by `router_exchange` inside `f`, on this thread, while
    // `hook` is mutably borrowed by this frame, and the slot is restored before this function returns (or on unwind)
    let p: *mut RouterExchange<'static> =
        unsafe { core::mem::transmute::<*mut RouterExchange<'_>, *mut RouterExchange<'static>>(hook) };
    let _restore = Restore(ROUTER_SPLIT.with(|c| c.replace(Some((rank, p)))));
    f()
}

/// Lane S (P2 TP shared-expert split, `docs/NEMOTRON_PREFILL_SPEED.md` 6.9 / 6.13): a byte exchange with the peer on the SECOND
/// communicator, `(send, send_bytes, recv, recv_bytes, st)` from offset 0 of both, queued on `st` (the block's side stream). No arithmetic.
pub type SharedExchange<'a> = dyn FnMut(&DevBuf, usize, &mut DevBuf, usize, &Stream) -> Result<(), CudaError> + 'a;

/// The smallest call that splits the shared expert (smaller ones run it whole on both ranks, as before).
pub const SHARED_SPLIT_MIN_ROWS: usize = 64;

type SharedSlot = Option<(usize, *mut SharedExchange<'static>)>;

std::thread_local! {
    /// See [`with_shared_split`].
    static SHARED_SPLIT: core::cell::Cell<SharedSlot> = const { core::cell::Cell::new(None) };
}

/// Runs `f` with the shared-expert split on (`hook` given) for the tensor-core block calls `f` makes on this thread: `rank` computes
/// the shared up projection's 128-column blocks of its half ([`shared_split_blocks`] of 42) and the down projection's (of 32), and
/// `hook` exchanges the bytes (the up output's ScaledRows values and factors, then the down output's f32 columns), so `sh_out` is the
/// single node's, complete, before the block joins the side stream. Calls under [`SHARED_SPLIT_MIN_ROWS`] rows and blocks whose shared
/// projections are not on tensor cores run whole. `None`: just `f`. Restored when `f` returns or unwinds.
pub fn with_shared_split<R>(rank: usize, hook: Option<&mut SharedExchange<'_>>, f: impl FnOnce() -> R) -> R {
    let Some(hook) = hook else { return f() };
    assert!(rank < 2, "the shared-expert split is for two ranks (rank {rank})");
    struct Restore(SharedSlot);
    impl Drop for Restore {
        fn drop(&mut self) {
            SHARED_SPLIT.with(|c| c.set(self.0));
        }
    }
    // SAFETY: as `with_router_split`: lifetime erased, used only inside `f` on this thread while `hook` is borrowed here
    let p: *mut SharedExchange<'static> =
        unsafe { core::mem::transmute::<*mut SharedExchange<'_>, *mut SharedExchange<'static>>(hook) };
    let _restore = Restore(SHARED_SPLIT.with(|c| c.replace(Some((rank, p)))));
    f()
}

/// The 128-column blocks of a projection of `nblocks` blocks for `rank`: (own, peer), rank 0 the first half. Pure (for the tests).
pub fn shared_split_blocks(rank: usize, nblocks: usize) -> (core::ops::Range<usize>, core::ops::Range<usize>) {
    router_split_rows(rank, nblocks)
}

/// This thread's rank when its block call of `rows` rows splits the shared expert, else `None`.
pub(crate) fn shared_split_rank(rows: usize) -> Option<usize> {
    let (rank, _) = SHARED_SPLIT.with(|c| c.get())?;
    (rows >= SHARED_SPLIT_MIN_ROWS).then_some(rank)
}

/// Runs this thread's shared-expert exchange (see [`with_shared_split`]).
pub(crate) fn shared_exchange(
    send: &DevBuf,
    send_bytes: usize,
    recv: &mut DevBuf,
    recv_bytes: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    let (_, p) = SHARED_SPLIT.with(|c| c.get()).expect("shared_exchange outside with_shared_split");
    // SAFETY: `with_shared_split`'s contract
    unsafe { (*p)(send, send_bytes, recv, recv_bytes, st) }
}

/// The row halves of a split router call of `rows` rows for `rank`: (own, peer). Pure (for the tests).
pub fn router_split_rows(rank: usize, rows: usize) -> (core::ops::Range<usize>, core::ops::Range<usize>) {
    let h = rows.div_ceil(2);
    if rank == 0 {
        (0..h, h..rows)
    } else {
        (h..rows, 0..h)
    }
}

/// (own rows, peer rows) when this thread's block call of `rows` rows splits the router, else `None`.
pub(crate) fn router_split_for(rows: usize) -> Option<(core::ops::Range<usize>, core::ops::Range<usize>)> {
    let (rank, _) = ROUTER_SPLIT.with(|c| c.get())?;
    (rows >= ROUTER_SPLIT_MIN_ROWS).then(|| router_split_rows(rank, rows))
}

/// Lane S: how many block calls ran the router split and the shared-expert split in this process (the gates count a leg only when its
/// counter is above zero: a split that silently fell back to the whole path would match too). The first of each also logs a line.
static ROUTER_SPLIT_CALLS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static SHARED_SPLIT_CALLS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// (router-split calls, shared-split calls) so far in this process (see `ROUTER_SPLIT_CALLS`).
pub fn split_counters() -> (u64, u64) {
    (ROUTER_SPLIT_CALLS.load(Ordering::Relaxed), SHARED_SPLIT_CALLS.load(Ordering::Relaxed))
}

pub(crate) fn count_split(router: bool, rows: usize) {
    let (c, what) = if router { (&ROUTER_SPLIT_CALLS, "router") } else { (&SHARED_SPLIT_CALLS, "shared") };
    if c.fetch_add(1, Ordering::Relaxed) == 0 {
        eprintln!("rexl3 event=p2tp_split_first kind={what} rows={rows}");
    }
}

/// Runs this thread's router exchange (see [`with_router_split`]) on `logits`.
pub(crate) fn router_exchange(
    logits: &mut DevBuf,
    own: core::ops::Range<usize>,
    peer: core::ops::Range<usize>,
    st: &Stream,
) -> Result<(), CudaError> {
    let (_, p) = ROUTER_SPLIT.with(|c| c.get()).expect("router_exchange outside with_router_split");
    count_split(true, own.len() + peer.len());
    let send = logits.alias();
    // SAFETY: `with_router_split`'s contract: `p` is the caller's live, exclusively borrowed hook for the duration of `f`
    unsafe { (*p)(&send, logits, own, peer, st) }
}

/// Whether the absent-expert fast path is on ([`ABSENT_FAST_VAR`], or [`set_absent_fast`]).
pub fn absent_fast() -> bool {
    match ABSENT_FAST.load(Ordering::Relaxed) {
        0 => {
            let on = parse_absent_fast(std::env::var(ABSENT_FAST_VAR).ok().as_deref());
            ABSENT_FAST.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
        v => v == 2,
    }
}

/// Whether a value of [`ABSENT_FAST_VAR`] leaves the fast path on: unset or anything but `0`/`off`/`false`/`no` is on.
pub fn parse_absent_fast(v: Option<&str>) -> bool {
    !matches!(v.map(|s| s.trim().to_ascii_lowercase()).as_deref(), Some("0" | "off" | "false" | "no"))
}

/// Turns the absent-expert fast path on or off for this process (tests and benches; the bits are the same either way).
pub fn set_absent_fast(on: bool) {
    ABSENT_FAST.store(if on { 2 } else { 1 }, Ordering::Relaxed);
}

/// What a routed expert's pair of projections is, for the prefill kernels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// 4-bit Mul1 EXL3 up and down.
    Fast,
    /// Mul1 EXL3 up and down at integer widths of 2 to 8 bits, not both 4 (a MixedK pack's experts): `pf_gemmk`, the whole layer's
    /// fast experts in its any-width launches. With [`any_width_prefill`] (on since P1-7); off, such a pair is Unsupported and the layer
    /// stays on the decode path (the zero-output hazard of `pf_slow` / `pf_zero_absent` is why it is never Slow).
    FastK,
    /// bf16 up and down, or an expert with no weights loaded (zeros).
    Slow,
    /// Anything else (another width or codebook, or a mix): the prefill path does not run it.
    Unsupported,
}

fn is_fast(p: &Proj) -> bool {
    matches!(p, Proj::Exl3 { bits, codebook, .. } if bits.twice() == 8 && *codebook == Codebook::Mul1)
}

static ANY_WIDTH_PREFILL: AtomicBool = AtomicBool::new(true);

/// Whether the prefill path takes MixedK layers (routed experts and dense projections at integer widths other than 4, fp16 dense
/// projections): read by `PrefillMoe::tables`. ON since P1-7 (v0.1.2 lane K: the any-width prefill kernels, `pf_gemmk` and the
/// tensor-core width launches, equal the decode path / the P2 contract's reference at every width and in mixed tables); off, such a
/// layer is not `supports_prefill` and runs on the decode path in slices (correct, slow). A pair or projection no any-width kernel
/// runs (a half-bit width, another codebook, a width mixed with bf16 or absent) stays Unsupported either way. Process-wide: set it
/// before building the tables.
pub fn set_any_width_prefill(on: bool) {
    ANY_WIDTH_PREFILL.store(on, Ordering::Relaxed);
}

fn any_width_prefill() -> bool {
    ANY_WIDTH_PREFILL.load(Ordering::Relaxed)
}

static FORCE_ANY_WIDTH: AtomicBool = AtomicBool::new(false);

/// Build the prefill tables of a 4-bit layer as a MixedK one (`kmode`: its fast experts on the any-width kernels, `pf_gemmk` and the
/// tensor-core path's width launches, instead of `pf_gemm4` and the 4-bit launch). The bits are the same; this exists for the gate of
/// the any-width prefill kernels at 4 bits and to time them on the same weights. Process-wide, read by `PrefillMoe::tables`.
pub fn set_force_any_width_prefill(on: bool) {
    FORCE_ANY_WIDTH.store(on, Ordering::Relaxed);
}

fn is_slow(p: &Proj) -> bool {
    matches!(p, Proj::Bf16 { .. } | Proj::Absent)
}

/// What one of the four dense projections (fc1, fc2, shared up, shared down) is, for the prefill kernels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DenseKind {
    /// 4-bit Mul1 EXL3: `pf_gemm4` over row chunks, its input rotated first.
    Fast,
    /// Mul1 EXL3 at another integer width (2 to 8 bits): `pf_gemmk` over chunks of 16 rows, its input rotated first ([`any_width_prefill`]).
    FastK,
    /// bf16 `[N, K]`: `pf_slow` over row chunks, plain input.
    Bf16,
    /// fp16 `[N, K]` (a MixedK pack's latent projections): `pf_slow` on the fp16 values, plain input ([`any_width_prefill`]).
    F16,
    /// Another width or codebook: the prefill path does not run it.
    Unsupported,
}

fn dense_kind(p: &Proj) -> DenseKind {
    if is_fast(p) {
        DenseKind::Fast
    } else if matches!(p, Proj::Bf16 { .. }) {
        DenseKind::Bf16
    } else if any_width_prefill() && p.is_fastk() {
        DenseKind::FastK
    } else if any_width_prefill() && matches!(p, Proj::F16 { .. }) {
        DenseKind::F16
    } else {
        DenseKind::Unsupported
    }
}

fn pair_kind(up: &Proj, down: &Proj) -> Kind {
    if is_fast(up) && is_fast(down) {
        Kind::Fast
    } else if any_width_prefill() && up.is_fastk() && down.is_fastk() {
        Kind::FastK
    } else if is_slow(up) && is_slow(down) {
        Kind::Slow
    } else {
        Kind::Unsupported
    }
}

fn codebook_index(cb: Codebook) -> i32 {
    match cb {
        Codebook::Inst3 => 0,
        Codebook::Mcg => 1,
        Codebook::Mul1 => 2,
    }
}

fn pf_proj(p: &Proj) -> PfProj {
    match p {
        Proj::Exl3 { k, n, bits, codebook, trellis, suh, svh, bias } => PfProj {
            w: trellis.ptr() as u64,
            suh: suh.ptr() as u64,
            svh: svh.ptr() as u64,
            bias: bias.as_ref().map_or(0, |b| b.ptr() as u64),
            cb: 0,
            fmt: FMT_EXL3 | (codebook_index(*codebook) << 4),
            twice: bits.twice() as i32,
            k: *k as i32,
            n: *n as i32,
        },
        Proj::Bf16 { k, n, w } => PfProj { w: w.ptr() as u64, fmt: FMT_BF16, k: *k as i32, n: *n as i32, ..Default::default() },
        Proj::F16 { k, n, w } => PfProj { w: w.ptr() as u64, fmt: FMT_F16, k: *k as i32, n: *n as i32, ..Default::default() },
        Proj::Absent => PfProj { fmt: FMT_ABSENT, ..Default::default() },
    }
}

/// The bits of the pairs of items of work a block of `rows` rows holds at most: experts with rows, split in blocks of `BLOCK_ROWS`.
fn item_stride(rows: usize) -> usize {
    EXPERTS + (rows * TOPK).div_ceil(CLASS_ROWS)
}

/// The prefill scratch for up to `max_rows` rows: the router's logits, picks, weights and expert lists, the work items, the rotated
/// inputs of the projections and the activations. Layers run one after another, so one instance serves all of them.
pub struct PrefillScratch {
    max_rows: usize,
    logits: DevBuf,
    pick: DevBuf,
    wts: DevBuf,
    cnt: DevBuf,
    lists: DevBuf,
    work: DevBuf,
    nwork: DevBuf,
    xlat: DevBuf,
    act: DevBuf,
    /// The up projection's rotated input rows (one a pair), then, once the up projection has read them, the down projection's output.
    y: DevBuf,
    routed: DevBuf,
    /// `h` rotated for fc1, then (fc1 having read it) `routed` rotated for fc2.
    xh_h1: DevBuf,
    xh_h2: DevBuf,
    sh_a: DevBuf,
    sh_out: DevBuf,
    aux: Stream,
    ev_fork: Event,
    ev_join: Event,
}

impl PrefillScratch {
    /// The number of 4-byte elements of each buffer, in field order, for `max_rows` rows.
    fn lens(max_rows: usize) -> [usize; 15] {
        let r = max_rows;
        let p = r * TOPK;
        [
            r * EXPERTS,            // logits
            p,                      // pick (i32)
            p,                      // wts
            EXPERTS,                // cnt (i32)
            EXPERTS * r,            // lists (i32)
            3 * 3 * item_stride(r), // work: three classes of (expert, first, rows) (i32)
            4,                      // nwork (i32 x 3)
            r * LATENT,             // xlat
            p * EXPERT_INTER,       // act: the activations, rotated for the down projection (plain for a slow expert)
            p * LATENT,             // y: also the up projection's rotated input rows
            r * LATENT,             // routed
            r * HIDDEN,             // xh_h1: h rotated for fc1 (and routed rotated for fc2)
            r * HIDDEN,             // xh_h2: h rotated for the shared up projection
            r * SHARED_INTER,       // sh_a: the shared activations, rotated for the shared down projection
            r * HIDDEN,             // sh_out
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
            work: f(l[5])?,
            nwork: f(l[6])?,
            xlat: f(l[7])?,
            act: f(l[8])?,
            y: f(l[9])?,
            routed: f(l[10])?,
            xh_h1: f(l[11])?,
            xh_h2: f(l[12])?,
            sh_a: f(l[13])?,
            sh_out: f(l[14])?,
            aux: Stream::new_nonblocking()?,
            ev_fork: Event::new()?,
            ev_join: Event::new()?,
        })
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// Device addresses of the last call's bucket outputs (lane S, the TP pair gather): the per-expert pair counts (i32 [512]) and the
    /// per-expert ascending pair lists (i32 [512 * rows of the call]: list stride = the call's row count).
    pub fn route_ptrs(&self) -> (usize, usize) {
        (self.cnt.ptr() as usize, self.lists.ptr() as usize)
    }

    fn check_rows(&self, rows: usize) {
        assert!((1..=self.max_rows).contains(&rows), "rows {rows} is outside 1..={}", self.max_rows);
    }

    /// Fills every intermediate with a NaN bit pattern, queued on `st` (for tests: a stage that does not write its output cannot then
    /// pass by finding the previous call's values).
    pub fn poison(&mut self, st: &Stream) -> Result<(), CudaError> {
        for b in [
            &mut self.logits,
            &mut self.pick,
            &mut self.wts,
            &mut self.cnt,
            &mut self.lists,
            &mut self.work,
            &mut self.nwork,
            &mut self.xlat,
            &mut self.act,
            &mut self.y,
            &mut self.routed,
            &mut self.xh_h1,
            &mut self.xh_h2,
            &mut self.sh_a,
            &mut self.sh_out,
        ] {
            b.memset_async(0xFF, st)?;
        }
        Ok(())
    }

    /// The router's logits of the last call (`[rows, 512]`) on the host. Synchronises.
    pub fn download_logits(&self, rows: usize, st: &Stream) -> Result<Vec<f32>, CudaError> {
        st.sync()?;
        let mut out = vec![0f32; rows * EXPERTS];
        self.logits.download_range(0, &mut out)?;
        Ok(out)
    }

    /// The picks `[rows, 22]` (in rank order) and weights of the last call on the host. Synchronises.
    pub fn download_route(&self, rows: usize, st: &Stream) -> Result<(Vec<i32>, Vec<f32>), CudaError> {
        st.sync()?;
        let mut pick = vec![0i32; rows * TOPK];
        let mut wts = vec![0f32; rows * TOPK];
        self.pick.download_range(0, &mut pick)?;
        self.wts.download_range(0, &mut wts)?;
        Ok((pick, wts))
    }

    /// The work items of the last call: for each class (small, large, slow) the list of (expert, first row, rows). Synchronises.
    pub fn download_items(&self, rows: usize, st: &Stream) -> Result<[Vec<[i32; 3]>; 3], CudaError> {
        st.sync()?;
        let stride = item_stride(rows);
        let mut n = [0i32; 4];
        self.nwork.download_range(0, &mut n)?;
        let mut all = vec![0i32; 3 * 3 * stride];
        self.work.download_range(0, &mut all)?;
        let class = |c: usize| -> Vec<[i32; 3]> {
            (0..n[c] as usize)
                .map(|i| [all[(c * stride + i) * 3], all[(c * stride + i) * 3 + 1], all[(c * stride + i) * 3 + 2]])
                .collect()
        };
        Ok([class(0), class(1), class(2)])
    }
}

/// A [`NemotronMoe`] and the device tables of the large-row prefill path. Dereferences to the [`NemotronMoe`] (the decode path).
pub struct PrefillMoe {
    moe: NemotronMoe,
    tab_up: DevBuf,
    tab_down: DevBuf,
    tab_dense: DevBuf,
    gate: usize,
    /// The gate's form (bf16 bits, or f32 for an F16 gate): which router kernel runs.
    gate_dt: crate::nemotron_moe::GateDtype,
    bias: usize,
    scale: f32,
    /// Routed experts the fast kernels do not run (bf16 or absent), and whether the prefill path runs this layer at all.
    n_slow: usize,
    /// Of those, the experts with no weights at all (both projections `Proj::Absent`).
    n_absent: usize,
    supported: bool,
    /// Some routed pair is any-width (Kind::FastK): the routed fast experts all run on `pf_gemmk` (the bucket's any-width classes, items
    /// of at most 16 rows), 4-bit ones included. False for the published pack: today's launches.
    kmode: bool,
    /// The widths (bits x 2) the routed up and down projections of the fast experts hold, ascending (kmode): the tensor-core path takes
    /// one launch a width.
    up_widths: Vec<i32>,
    down_widths: Vec<i32>,
    /// fc1, fc2, shared up, shared down.
    dense: [DenseKind; 4],
    /// The same four as the tensor-core prefill GEMM reads them (lane S3, `nemotron_dense_mma`), when it takes all four; it runs them
    /// only inside a prefill scope with the opt-in on (`nemotron_dense_mma::active`), never for a wide decode window.
    mma: Option<[crate::nemotron_dense_mma::Layer; 4]>,
}

impl core::ops::Deref for PrefillMoe {
    type Target = NemotronMoe;
    fn deref(&self) -> &NemotronMoe {
        &self.moe
    }
}

impl PrefillMoe {
    /// The decode path's tables (`NemotronMoe::tables`) and the prefill path's, from the same weights.
    pub fn tables(w: Weights) -> Result<Self, CudaError> {
        let up_t: Vec<PfProj> = w.up.iter().map(pf_proj).collect();
        let down_t: Vec<PfProj> = w.down.iter().map(pf_proj).collect();
        let dense_t = [pf_proj(&w.fc1), pf_proj(&w.fc2), pf_proj(&w.shared_up), pf_proj(&w.shared_down)];
        let kinds: Vec<Kind> = w.up.iter().zip(&w.down).map(|(u, d)| pair_kind(u, d)).collect();
        let dense = [&w.fc1, &w.fc2, &w.shared_up, &w.shared_down].map(dense_kind);
        let supported = dense.iter().all(|k| *k != DenseKind::Unsupported)
            && up_t.len() == EXPERTS
            && down_t.len() == EXPERTS
            && kinds.iter().all(|k| *k != Kind::Unsupported);
        let n_slow = kinds.iter().filter(|k| **k == Kind::Slow).count();
        let kmode = kinds.contains(&Kind::FastK) || FORCE_ANY_WIDTH.load(Ordering::Relaxed);
        let widths = |t: &[Proj]| -> Vec<i32> {
            let mut v: Vec<i32> = t
                .iter()
                .filter_map(|p| match p {
                    Proj::Exl3 { bits, .. } if p.is_fastk() => Some(bits.twice() as i32),
                    _ => None,
                })
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let (up_widths, down_widths) = (widths(&w.up), widths(&w.down));
        let n_absent = w.up.iter().zip(&w.down).filter(|(u, d)| matches!(u, Proj::Absent) && matches!(d, Proj::Absent)).count();
        let tab_up = DevBuf::from_slice(&up_t[..])?;
        let tab_down = DevBuf::from_slice(&down_t[..])?;
        let tab_dense = DevBuf::from_slice(&dense_t[..])?;
        let (gate, bias, scale) = (w.gate.ptr() as usize, w.router_bias.ptr() as usize, w.scale);
        // the layers point into `w`'s buffers, which `moe` owns from here on (device memory: the pointers stay valid)
        let mma = {
            use crate::nemotron_dense_mma::Layer;
            let l = [&w.fc1, &w.fc2, &w.shared_up, &w.shared_down].map(Layer::of_proj);
            if dense.iter().all(|k| *k == DenseKind::Fast) && l.iter().all(|x| x.is_some_and(|x| x.supported())) {
                Some(l.map(Option::unwrap))
            } else {
                None
            }
        };
        let moe = NemotronMoe::tables(w)?;
        let gate_dt = moe.gate_dtype();
        Ok(Self {
            moe,
            tab_up,
            tab_down,
            tab_dense,
            gate,
            gate_dt,
            bias,
            scale,
            n_slow,
            n_absent,
            supported,
            kmode,
            up_widths,
            down_widths,
            dense,
            mma,
        })
    }

    /// The decode path's block.
    pub fn moe(&self) -> &NemotronMoe {
        &self.moe
    }

    /// Whether the prefill path runs this layer: every dense projection (fc1, fc2, the shared expert's) is 4-bit Mul1 EXL3 or bf16 and
    /// every routed expert's pair is 4-bit Mul1 EXL3, bf16 or absent. A layer that is not (another width or codebook) is run by the
    /// decode path in slices of 64 rows.
    pub fn supports_prefill(&self) -> bool {
        self.supported
    }

    /// The routed experts the fast kernels do not run (bf16 or absent).
    pub fn slow_experts(&self) -> usize {
        self.n_slow
    }

    /// One of the four dense projections (`which`: 0 fc1, 1 fc2, 2 shared up, 3 shared down) over `rows` rows, on `raw`.
    ///
    /// `x` is its input `[rows, k]`: plain, or (`rotated`, a 4-bit projection only) already rotated by the previous projection. A 4-bit
    /// projection that is not `rotated` is rotated first into `rot`. The output goes to `out` (`[rows, n]`), plus `add`, after relu2 when
    /// asked. The shared up projection hands the shared down projection a rotated output when that one is 4-bit: its epilogue does it,
    /// or, after a bf16 up projection, a rotation in place.
    #[allow(clippy::too_many_arguments)]
    fn dense_proj(
        &self,
        which: usize,
        x: *const c_void,
        rotated: bool,
        rot: *mut c_void,
        rows: usize,
        out: *mut c_void,
        relu2: bool,
        add: *const c_void,
        raw: *mut c_void,
    ) -> Result<(), CudaError> {
        let (k, n) = match which {
            0 => (HIDDEN, LATENT),
            1 => (LATENT, HIDDEN),
            2 => (HIDDEN, SHARED_INTER),
            _ => (SHARED_INTER, HIDDEN),
        };
        let tab = |i: usize| (self.tab_dense.ptr() as usize + i * PROJ_BYTES) as *const c_void;
        let items = rows.div_ceil(BLOCK_ROWS) as i32;
        let ok = CudaError::check;
        // P2 (lane S3): the tensor-core GEMM for a prefill slice, when opted in. Its rotated buffers hold fp16 rows (the shared up's
        // epilogue writes the shared down's input that way), so all four projections of a block take it or none does.
        if let (Some(l), true) = (self.mma.as_ref(), crate::nemotron_dense_mma::active(rows)) {
            use crate::nemotron_dense_mma::{gemm_raw, rotate_raw, Epilogue};
            let lw = &l[which];
            // SAFETY: the scratch buffers hold `rows` rows of each projection's widths (f32, so fp16 rows fit), the layers' buffers are
            // owned by `self.moe`
            unsafe {
                let src = if rotated {
                    x
                } else {
                    rotate_raw(lw, x, crate::exl3_linear::Dtype::F32, k, rot, rows, raw)?;
                    rot as *const c_void
                };
                let e =
                    Epilogue { relu2, add, ld_add: n, next_suh: if which == 2 { l[3].suh } else { core::ptr::null() }, tile: 0 };
                return gemm_raw(lw, src, out, crate::exl3_linear::Dtype::F32, n, rows, &e, raw);
            }
        }
        match self.dense[which] {
            DenseKind::Bf16 | DenseKind::F16 => {
                assert!(!rotated, "a bf16 or fp16 projection reads plain rows");
                // an fp16 dense projection (the MixedK pack's fc1 / fc2, every row of the slice) on pf_slow4: the same chain per (row,
                // column) as pf_slow's, so the same bits, with the weights of four columns read once for eight rows (lane K, CP2: pf_slow
                // over the 80 fp16 latents was 40 % of his pack's prefill kernel time). A bf16 one keeps the switch's kernel.
                let variant = if self.dense[which] == DenseKind::F16 && n % 32 == 0 {
                    1
                } else {
                    BF16_KERNEL.load(Ordering::Relaxed) as i32
                };
                ok(unsafe {
                    ffi::gb_nmp_slow(
                        x,
                        k as i32,
                        1,
                        core::ptr::null(),
                        0,
                        core::ptr::null(),
                        core::ptr::null(),
                        rows as i32,
                        items,
                        tab(which),
                        k as i32,
                        n as i32,
                        out,
                        relu2 as i32,
                        add,
                        variant,
                        raw,
                    )
                })?;
                if which == 2 && matches!(self.dense[3], DenseKind::Fast | DenseKind::FastK) {
                    let rotate = if self.dense[3] == DenseKind::FastK { ffi::gb_nmp_rotate_k } else { ffi::gb_nmp_rotate };
                    ok(unsafe {
                        rotate(out as *const c_void, n as i32, 1, core::ptr::null(), tab(3), rows as i32, n as i32, out, raw)
                    })?;
                }
                Ok(())
            }
            DenseKind::Fast => {
                let src = if rotated {
                    x
                } else {
                    ok(unsafe {
                        ffi::gb_nmp_rotate(x, k as i32, 1, core::ptr::null(), tab(which), rows as i32, k as i32, rot, raw)
                    })?;
                    rot as *const c_void
                };
                // the epilogue of the shared up projection rotates its output for the shared down one (the kernel checks that it is fast)
                let ntab = if which == 2 { tab(3) } else { core::ptr::null() };
                ok(unsafe {
                    ffi::gb_nmp_gemm(
                        src,
                        k as i32,
                        core::ptr::null(),
                        0,
                        core::ptr::null(),
                        core::ptr::null(),
                        rows as i32,
                        items,
                        BLOCK_ROWS as i32,
                        tab(which),
                        ntab,
                        k as i32,
                        n as i32,
                        slices(k) as i32,
                        out,
                        relu2 as i32,
                        add,
                        raw,
                    )
                })
            }
            DenseKind::FastK => {
                let src = if rotated {
                    x
                } else {
                    ok(unsafe {
                        ffi::gb_nmp_rotate_k(x, k as i32, 1, core::ptr::null(), tab(which), rows as i32, k as i32, rot, raw)
                    })?;
                    rot as *const c_void
                };
                // as the 4-bit arm, the any-width kernel over chunks of 16 rows; its epilogue rotates the shared up's output for an
                // any-width (4-bit included) shared down
                let ntab = if which == 2 { tab(3) } else { core::ptr::null() };
                ok(unsafe {
                    ffi::gb_nmp_gemmk(
                        src,
                        k as i32,
                        core::ptr::null(),
                        0,
                        core::ptr::null(),
                        core::ptr::null(),
                        rows as i32,
                        rows.div_ceil(CLASS_ROWS) as i32,
                        tab(which),
                        ntab,
                        k as i32,
                        n as i32,
                        slices(k) as i32,
                        out,
                        relu2 as i32,
                        add,
                        raw,
                    )
                })
            }
            DenseKind::Unsupported => unreachable!("a layer with an unsupported dense projection is not run by the prefill path"),
        }
    }

    /// The routed GEMM launches of the tensor-core path for the up (`up`) or down projection: one of the 4-bit kernel (`None`) for a
    /// 4-bit table, else one of the width kernel at each width the projections hold.
    pub(crate) fn routed_launch_widths(&self, up: bool) -> Vec<Option<i32>> {
        if !self.kmode {
            return vec![None];
        }
        (if up { &self.up_widths } else { &self.down_widths }).iter().map(|&k2| Some(k2)).collect()
    }

    /// Whether the shared down projection's input arrives rotated: the shared up's epilogue rotates it (nm2 pf_gemm4 for a 4-bit down,
    /// pf_gemmk for any fast down), or a bf16 / fp16 up is followed by an in-place rotation. The published pack (4-bit, 4-bit): true.
    fn shared_down_rotated(&self) -> bool {
        let fast_down = matches!(self.dense[3], DenseKind::Fast | DenseKind::FastK);
        match self.dense[2] {
            DenseKind::Fast => self.dense[3] == DenseKind::Fast,
            DenseKind::FastK | DenseKind::Bf16 | DenseKind::F16 => fast_down,
            DenseKind::Unsupported => false,
        }
    }

    /// A MixedK layer's routed projection on the any-width kernel: the up (relu2, its output rotated for any-width downs) or the down
    /// projection of the items at `work` (16 rows at most each, `z` of them at most).
    #[allow(clippy::too_many_arguments)]
    fn routed_k(
        &self,
        src: *const c_void,
        k: usize,
        lists: *const c_void,
        lcap: usize,
        work: *const c_void,
        nwork: *const c_void,
        z: usize,
        up: bool,
        out: *mut c_void,
        sc: *mut c_void,
    ) -> Result<(), CudaError> {
        let (tab, ntab, n) = if up {
            (self.tab_up.ptr() as *const c_void, self.tab_down.ptr() as *const c_void, EXPERT_INTER)
        } else {
            (self.tab_down.ptr() as *const c_void, core::ptr::null(), LATENT)
        };
        CudaError::check(unsafe {
            ffi::gb_nmp_gemmk(
                src,
                k as i32,
                lists,
                lcap as i32,
                work,
                nwork,
                0,
                z as i32,
                tab,
                ntab,
                k as i32,
                n as i32,
                slices(k) as i32,
                out,
                up as i32,
                core::ptr::null(),
                sc,
            )
        })
    }

    /// The full block for `rows` tokens of `h` (`[rows, 4096]`, f32): `out = routed + shared`, bit for bit what
    /// [`NemotronMoe::forward_in`] gives for each row. `rows` up to the scratch's `max_rows`; the layer must `supports_prefill()`.
    pub fn forward_prefill_in(
        &self,
        s: &mut PrefillScratch,
        h: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.forward_prefill_reduce(s, h, rows, out, st, None)
    }

    /// `forward_prefill_in` for one rank of an expert-parallel (TP-exact) group: exactly its launches, plus one call of `reduce` on `st`
    /// between the routed down projection and the combine, with the routed output buffer `y` (f32, `[rows, 22, 1024]`; see
    /// [`ReduceHook`]). A rank's non-owned experts are `Proj::Absent`: the slow class writes exact zeros for their pairs, so with the
    /// reduce summing `y` over the ranks the combine, fc2 and the shared expert see the whole block's `y` bit for bit. With a no-op
    /// `reduce` this is `forward_prefill_in`.
    pub fn forward_prefill_in_tp(
        &self,
        s: &mut PrefillScratch,
        h: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        st: &Stream,
        reduce: &mut ReduceHook,
    ) -> Result<(), CudaError> {
        self.forward_prefill_reduce(s, h, rows, out, st, Some(reduce))
    }

    fn forward_prefill_reduce(
        &self,
        s: &mut PrefillScratch,
        h: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        st: &Stream,
        reduce: Option<&mut ReduceHook>,
    ) -> Result<(), CudaError> {
        assert!(self.supported, "this layer has a projection the prefill path does not run (see supports_prefill)");
        s.check_rows(rows);
        assert!(h.bytes() >= rows * HIDDEN * 4 && out.bytes() >= rows * HIDDEN * 4, "h or out is short for {rows} rows");
        let p = rows * TOPK;
        let stride = item_stride(rows);
        let (hp, sc) = (h.ptr() as *const c_void, st.raw());
        let p_ = |b: &DevBuf| b.ptr() as *const c_void;
        #[allow(clippy::unnecessary_cast)] // the same type in the stub build only
        let pm = |b: &DevBuf| b.ptr() as *mut c_void;
        let ok = CudaError::check;

        // The shared expert depends on h only: on the side stream beside the router and the routed chain, or in line before it.
        let side = SIDE_STREAM.load(Ordering::Relaxed);
        let shared = |sx: &Stream| -> Result<(), CudaError> {
            let raw = sx.raw();
            // up: relu2(up(h)), rotated for the down projection when that is fast (by the epilogue, or in place after a bf16 up)
            self.dense_proj(2, hp, false, pm(&s.xh_h2), rows, pm(&s.sh_a), true, core::ptr::null(), raw)?;
            // down: its input is rotated already when it is fast (see shared_down_rotated); else it is rotated in place (an any-width
            // down after a 4-bit up, whose epilogue rotates only for a 4-bit down)
            self.dense_proj(
                3,
                p_(&s.sh_a),
                self.shared_down_rotated(),
                pm(&s.sh_a),
                rows,
                pm(&s.sh_out),
                false,
                core::ptr::null(),
                raw,
            )
        };
        if side {
            fork(st, &s.aux, &s.ev_fork)?;
            shared(&s.aux)?;
        }

        // Router: logits, then selection, then the lists and work items. A MixedK layer's fast experts are the any-width kernel's, in items
        // of at most 16 rows (schedule 2).
        let schedule = if self.kmode { 2 } else { SCHEDULE.load(Ordering::Relaxed) };
        unsafe {
            crate::nemotron_moe::fast_router_logits(hp, self.gate as *const c_void, self.gate_dt, pm(&s.logits), rows, sc)
        }?;
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
                schedule as i32,
                sc,
            )
        })?;

        // fc1: the latent x = fc1(h).
        self.dense_proj(0, hp, false, pm(&s.xh_h1), rows, pm(&s.xlat), false, core::ptr::null(), sc)?;

        // Routed experts. The work items of each class: the large class first (its blocks are the long ones), then the small one.
        let item_ptr = |c: usize| (s.work.ptr() as usize + c * stride * 3 * 4) as *const c_void;
        let nwork_ptr = |c: usize| (s.nwork.ptr() as usize + c * 4) as *const c_void;
        let z_small = EXPERTS.min(p);
        let z_large = stride.min(p / (CLASS_ROWS + 1) + p / BLOCK_ROWS + 2);
        // the launches of the fast experts: one over every item (interleaved), or the two classes
        let launches: Vec<(usize, usize, usize)> = match schedule {
            0 => vec![(1, BLOCK_ROWS, z_large), (0, CLASS_ROWS, z_small)],
            1 => vec![(0, BLOCK_ROWS, stride.min(EXPERTS.min(p) + p.div_ceil(BLOCK_ROWS)))],
            _ => vec![(0, CLASS_ROWS, stride.min(EXPERTS.min(p) + p.div_ceil(CLASS_ROWS)))],
        };
        let z_slow = (self.n_slow * rows.div_ceil(BLOCK_ROWS)).max(1);
        let bf16_kernel = BF16_KERNEL.load(Ordering::Relaxed) as i32;
        // every slow expert is absent and the switch is on: no pf_slow launch, one zero kernel after the down projection
        let zero_only = self.n_slow > 0 && self.n_absent == self.n_slow && absent_fast();
        let run_slow = self.n_slow > 0 && !zero_only;
        // Up: relu2(up(x)) for each pair on its expert; the fast experts' output is rotated for the down projection.
        ok(unsafe {
            (if self.kmode { ffi::gb_nmp_rotate_k } else { ffi::gb_nmp_rotate })(
                p_(&s.xlat),
                LATENT as i32,
                TOPK as i32,
                p_(&s.pick),
                p_(&self.tab_up),
                p as i32,
                LATENT as i32,
                pm(&s.y),
                sc,
            )
        })?;
        for &(class, rbmax, z) in &launches {
            if self.kmode {
                self.routed_k(p_(&s.y), LATENT, p_(&s.lists), rows, item_ptr(class), nwork_ptr(class), z, true, pm(&s.act), sc)?;
                continue;
            }
            ok(unsafe {
                ffi::gb_nmp_gemm(
                    p_(&s.y),
                    LATENT as i32,
                    p_(&s.lists),
                    rows as i32,
                    item_ptr(class),
                    nwork_ptr(class),
                    0,
                    z as i32,
                    rbmax as i32,
                    p_(&self.tab_up),
                    p_(&self.tab_down),
                    LATENT as i32,
                    EXPERT_INTER as i32,
                    slices(LATENT) as i32,
                    pm(&s.act),
                    1,
                    core::ptr::null(),
                    sc,
                )
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
                    item_ptr(2),
                    nwork_ptr(2),
                    0,
                    z_slow as i32,
                    p_(&self.tab_up),
                    LATENT as i32,
                    EXPERT_INTER as i32,
                    pm(&s.act),
                    1,
                    core::ptr::null(),
                    bf16_kernel,
                    sc,
                )
            })?;
        }
        // Down: the activations, per pair, on the same expert.
        for &(class, rbmax, z) in &launches {
            if self.kmode {
                self.routed_k(
                    p_(&s.act),
                    EXPERT_INTER,
                    p_(&s.lists),
                    rows,
                    item_ptr(class),
                    nwork_ptr(class),
                    z,
                    false,
                    pm(&s.y),
                    sc,
                )?;
                continue;
            }
            ok(unsafe {
                ffi::gb_nmp_gemm(
                    p_(&s.act),
                    EXPERT_INTER as i32,
                    p_(&s.lists),
                    rows as i32,
                    item_ptr(class),
                    nwork_ptr(class),
                    0,
                    z as i32,
                    rbmax as i32,
                    p_(&self.tab_down),
                    core::ptr::null(),
                    EXPERT_INTER as i32,
                    LATENT as i32,
                    slices(EXPERT_INTER) as i32,
                    pm(&s.y),
                    0,
                    core::ptr::null(),
                    sc,
                )
            })?;
        }
        if zero_only {
            ok(unsafe {
                ffi::gb_nmp_zero_absent(
                    p_(&s.lists),
                    rows as i32,
                    item_ptr(2),
                    nwork_ptr(2),
                    p_(&self.tab_down),
                    LATENT as i32,
                    pm(&s.y),
                    ZERO_BLOCKS,
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
                    item_ptr(2),
                    nwork_ptr(2),
                    0,
                    z_slow as i32,
                    p_(&self.tab_down),
                    EXPERT_INTER as i32,
                    LATENT as i32,
                    pm(&s.y),
                    0,
                    core::ptr::null(),
                    bf16_kernel,
                    sc,
                )
            })?;
        }
        // Expert parallel: the caller sums y over the ranks here, on the main stream (the side stream's shared expert needs nothing from it).
        if let Some(reduce) = reduce {
            reduce(&mut s.y, p * LATENT, st)?;
        }
        // The weighted sum of each token's 22 slots in slot order.
        ok(unsafe { ffi::gb_nm_combine(p_(&s.y), p_(&s.wts), rows as i32, LATENT as i32, pm(&s.routed), sc) })?;

        if side {
            join(st, &s.aux, &s.ev_join)?;
        } else {
            shared(st)?;
        }
        // fc2 back to the hidden width, plus the shared expert's output: the block's output.
        self.dense_proj(1, p_(&s.routed), false, pm(&s.xh_h1), rows, out.ptr(), false, p_(&s.sh_out), sc)
    }
}

/// The tensor-core path of contract P2 (lane S2), opt-in: a child module so that it reads the tables above as they are.
#[path = "nemotron_moe_mma.rs"]
pub mod mma;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_router_row_split_covers_every_row_once_and_only_inside_its_scope() {
        for rows in [1usize, 2, 31, 32, 33, 255, 512, 1024, 4096] {
            let (o0, p0) = router_split_rows(0, rows);
            let (o1, p1) = router_split_rows(1, rows);
            // the ranks agree on who owns what: each one's peer range is the other's own range, and together they tile 0..rows
            assert_eq!((o0.clone(), p0.clone()), (p1.clone(), o1.clone()), "{rows}");
            assert_eq!((o0.start, o0.end, o1.end), (0, rows.div_ceil(2), rows), "{rows}");
        }
        assert!(router_split_for(4096).is_none(), "no split outside with_router_split");
        let mut hook = |_: &DevBuf,
                        _: &mut DevBuf,
                        _: core::ops::Range<usize>,
                        _: core::ops::Range<usize>,
                        _: &Stream|
         -> Result<(), CudaError> { Ok(()) };
        let inside =
            with_router_split(1, Some(&mut hook), || (router_split_for(ROUTER_SPLIT_MIN_ROWS - 1), router_split_for(100)));
        assert_eq!(inside, (None, Some((50..100, 0..50))));
        assert!(router_split_for(100).is_none(), "cleared after the scope");
        assert!(with_router_split(0, None, || router_split_for(100)).is_none(), "None is no split");
    }

    #[test]
    fn the_shared_split_halves_are_whole_128_column_blocks() {
        // up 5,376 = 42 blocks, down 4,096 = 32 blocks: 21 | 21 and 16 | 16, each rank's peer range the other's own
        assert_eq!(SHARED_INTER % 256, 0);
        assert_eq!(HIDDEN % 256, 0);
        for n in [SHARED_INTER / 128, HIDDEN / 128] {
            let (o0, p0) = shared_split_blocks(0, n);
            let (o1, p1) = shared_split_blocks(1, n);
            assert_eq!((o0.clone(), p0), (p1, o1.clone()));
            assert_eq!((o0.len(), o1.len(), o1.end), (n / 2, n / 2, n));
        }
        let mut hook = |_: &DevBuf, _: usize, _: &mut DevBuf, _: usize, _: &Stream| -> Result<(), CudaError> { Ok(()) };
        assert_eq!(
            with_shared_split(1, Some(&mut hook), || (shared_split_rank(SHARED_SPLIT_MIN_ROWS - 1), shared_split_rank(512))),
            (None, Some(1))
        );
        assert!(shared_split_rank(512).is_none(), "cleared after the scope");
    }

    #[test]
    fn the_absent_fast_path_is_on_unless_opted_out() {
        assert!(parse_absent_fast(None));
        assert!(parse_absent_fast(Some("1")));
        assert!(parse_absent_fast(Some("")));
        for off in ["0", "off", "FALSE", " no "] {
            assert!(!parse_absent_fast(Some(off)), "{off}");
        }
    }

    #[test]
    fn the_device_struct_is_56_bytes() {
        assert_eq!(core::mem::size_of::<PfProj>(), PROJ_BYTES);
    }

    #[test]
    fn the_scratch_is_linear_in_rows() {
        let b64 = PrefillScratch::bytes(64) as f64;
        let b256 = PrefillScratch::bytes(256) as f64;
        assert!((b256 / b64 - 4.0).abs() < 0.05, "{b64} {b256}");
        assert!(PrefillScratch::bytes(256) < 200_000_000);
    }

    #[test]
    fn item_stride_bounds_the_items() {
        // an expert with c rows is ceil(c / 32) items: at most 512 + ceil(P / 32) of them in all
        for rows in [1usize, 64, 256, 512] {
            let p = rows * TOPK;
            assert!(item_stride(rows) >= EXPERTS + p.div_ceil(BLOCK_ROWS));
        }
    }
}

/// GUARD (v0.1.2 MixedK, R-MK-rails (d)): the prefill path's kernels run 4-bit Mul1 EXL3 (`pf_gemm4`) and bf16 (`pf_slow`) projections
/// only. `pf_slow` and `pf_zero_absent` write exact zeros for any other "slow" item, so a K != 4 EXL3 projection that reached them would
/// silently zero its expert (or its dense projection). These tests pin the classification that keeps such a layer off the prefill path
/// (`supports_prefill` false, no tensor-core block): every EXL3 width but 4 bits, alone or paired with a 4-bit or bf16 projection, is
/// `Unsupported`, as is a 4-bit projection of another codebook. Enabling another width here without its fast kernels (P1-5) is exactly
/// the zero-output hazard: change these tests only together with the kernels that run that width.
#[cfg(test)]
mod mixedk_guard {
    use super::*;
    use crate::arena::{Arena, BlockKind};
    use gb_exl3::Bits;

    /// A projection of `k x n` with `twice` (bits doubled) and `codebook`, on host buffers (only its form is read here).
    fn exl3(arena: &Arena, k: usize, n: usize, twice: u8, codebook: Codebook) -> Proj {
        let buf = || arena.alloc(16).unwrap();
        Proj::Exl3 { k, n, bits: Bits::from_twice(twice).unwrap(), codebook, trellis: buf(), suh: buf(), svh: buf(), bias: None }
    }

    fn bf16(arena: &Arena, k: usize, n: usize) -> Proj {
        Proj::Bf16 { k, n, w: arena.alloc(16).unwrap() }
    }

    #[test]
    fn a_whole_width_pair_is_any_width_fast_and_every_other_form_unsupported_so_pf_slow_and_pf_zero_absent_never_zero_it() {
        let arena = Arena::with_kind(1 << 20, BlockKind::PageableHost);
        let up = |t: u8| exl3(&arena, LATENT, EXPERT_INTER, t, Codebook::Mul1);
        let down = |t: u8| exl3(&arena, EXPERT_INTER, LATENT, t, Codebook::Mul1);
        // today's 4-bit Mul1 pair is fast, a bf16 pair and an absent pair are the slow kernel's
        assert_eq!(pair_kind(&up(8), &down(8)), Kind::Fast);
        assert_eq!(pair_kind(&bf16(&arena, LATENT, EXPERT_INTER), &bf16(&arena, EXPERT_INTER, LATENT)), Kind::Slow);
        assert_eq!(pair_kind(&Proj::Absent, &Proj::Absent), Kind::Slow);
        assert_eq!(dense_kind(&exl3(&arena, HIDDEN, LATENT, 8, Codebook::Mul1)), DenseKind::Fast);
        assert_eq!(dense_kind(&bf16(&arena, HIDDEN, LATENT)), DenseKind::Bf16);
        // P1-7 (lane K): with the any-width prefill kernels (the switch's default), a whole width of 2 to 8 bits (Mul1) is FastK for a
        // pair (with a 4-bit partner too: the MixedK pack mixes K within an expert) and for a dense projection; 1 bit, a half-bit width, or a
        // width paired with a bf16 or an absent projection, stays Unsupported: never Slow, so pf_slow / pf_zero_absent never see it.
        // The kernels' gates: tests/nemotron_moe_prefill.rs (prefill == decode path on MixedK layers, pf_gemmk at 4 bits ==
        // pf_gemm4), tests/nemotron_moe_mma.rs (the width launches), tests/nemotron_moe_perf.rs (the decode path == reference).
        assert!(super::any_width_prefill(), "the any-width prefill path is on by default since P1-7");
        for &t in Bits::ALL_TWICE.iter().filter(|&&t| t != 8) {
            // whole widths of 2 to 8 bits (twice 4 .. 16); 1 bit and the half-bit widths stay Unsupported
            let whole = t % 2 == 0 && t >= 4;
            let want = if whole { Kind::FastK } else { Kind::Unsupported };
            assert_eq!(pair_kind(&up(t), &down(t)), want, "{} bits", t as f64 / 2.0);
            assert_eq!(pair_kind(&up(8), &down(t)), want, "4 | {} bits", t as f64 / 2.0);
            assert_eq!(pair_kind(&up(t), &down(8)), want, "{} | 4 bits", t as f64 / 2.0);
            assert_eq!(pair_kind(&bf16(&arena, LATENT, EXPERT_INTER), &down(t)), Kind::Unsupported, "bf16 | {}", t as f64 / 2.0);
            assert_eq!(pair_kind(&up(t), &Proj::Absent), Kind::Unsupported, "{} | absent", t as f64 / 2.0);
            // the four dense projections (fc1, fc2, the shared expert) at the width
            for (k, n) in [(HIDDEN, LATENT), (LATENT, HIDDEN), (HIDDEN, SHARED_INTER), (SHARED_INTER, HIDDEN)] {
                let want = if whole { DenseKind::FastK } else { DenseKind::Unsupported };
                assert_eq!(dense_kind(&exl3(&arena, k, n, t, Codebook::Mul1)), want, "[{k}, {n}] {t}");
            }
        }
        // an fp16 dense projection (the MixedK pack's fc1 / fc2): pf_slow on its fp16 values
        assert_eq!(dense_kind(&Proj::F16 { k: HIDDEN, n: LATENT, w: arena.alloc(16).unwrap() }), DenseKind::F16);
        for cb in [Codebook::Mcg, Codebook::Inst3] {
            let (u, d) = (exl3(&arena, LATENT, EXPERT_INTER, 8, cb), exl3(&arena, EXPERT_INTER, LATENT, 8, cb));
            assert_eq!(pair_kind(&u, &d), Kind::Unsupported, "{cb:?}");
            assert_eq!(dense_kind(&exl3(&arena, HIDDEN, LATENT, 8, cb)), DenseKind::Unsupported, "{cb:?}");
        }
    }
}
