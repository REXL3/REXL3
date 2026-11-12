//! The composed model: the layer loop's shared state and the types the three arms and the oracle harness agree on. Wave 1 task CMP
//! fills the bodies. The names and shapes below are frozen at wave 0.
//!
//! One step: embed (`nemotron_glue::embed_gather_f32`), then per layer `bufs.xn = norm(x)`, an arm writes `bufs.mix`, `x += mix`
//! (fp32 residual), then `norm_f`, `lm_head` and `topk_f32`. Mixers read f32 and write f32. The EXL3 linears take f32 and round to
//! fp16 inside the kernel, as the fork does.

use super::arm_attn;
use super::arm_mamba;
use super::arm_moe;
use super::backend::{ModelOptions, NemotronRuntime};
use super::bridge;
use super::config::OmniConfig;
use super::device::{LayerKind, LayerMap, Mixer, NemotronWeights};
use super::inventory::Inventory;
use super::tp::{SingleRank, TpComm};
use super::tp_arm;
use super::vision::SpliceOp;
use super::weights::{self, LayerPlan, UnitId};
use crate::error::ModelError;
use crate::forward::Candidates;
use crate::kvtables::{KvTables, BLOCK_TOKENS};
use crate::rows::RowBatch;
use gb_cuda::attention::BlockTable;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights, MAX_DECODE_ROWS as LINEAR_ROWS};
use gb_cuda::gdn::Batch;
use gb_cuda::glm_mtp::add_f32;
use gb_cuda::graph::{CaptureError, Graph};
use gb_cuda::loader::Options;
use gb_cuda::nemotron_attn::{kv8, KvDtype, NemoKv, RowIndex, Scratch as AttnScratch, CHUNK_KEYS, KV_TOKEN_ELEMS, Q_ROW_ELEMS};
use gb_cuda::nemotron_glue::{
    embed_gather_f32, rmsnorm_bf16w_f32, splice_rows, topk_f32, topk_workspace_bytes, DT_COLS, HIDDEN, INPROJ_COLS, XBC_COLS,
    Z_COLS,
};
use gb_cuda::nemotron_moe::MoeScratch;
use gb_cuda::nemotron_moe_prefill::PrefillScratch;
use gb_cuda::nemotron_ssm::{self, LagRing, LagTable, Record, CONV_DIM, CONV_SLOT_ELEMS, HEADS, INNER, S_SLOT_ELEMS};
use gb_cuda::{CudaError, DevBuf, Stream};
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Reproduces the upload the trunk used before the hazard fix (`DevBuf::upload`: a pageable `cudaMemcpy` on the legacy default stream,
/// which the model's non-blocking stream does not wait for) so that the test of the fix can show the race. Never set outside that test.
static LEGACY_UPLOADS: AtomicBool = AtomicBool::new(false);

/// Test hook: makes the trunk's per-step uploads the unordered copies of the old code (see [`LEGACY_UPLOADS`]).
#[doc(hidden)]
pub fn set_legacy_uploads(on: bool) {
    LEGACY_UPLOADS.store(on, Ordering::Relaxed);
}

/// Writes `v` at the start of `dst` in the order of `st`: a kernel launched on `st` after this call reads the new bytes, and a kernel
/// launched before it has read the old ones. `DevBuf::upload` is a pageable `cudaMemcpy` on the legacy default stream: it may return before
/// its DMA is done and the model's stream is non-blocking, so it does not wait for it either (a kernel launched right after such an upload
/// read the previous step's ids in 2 of 120 chains). This is a `cudaMemcpyAsync` on `st` instead. `v` is pageable host memory, which the
/// driver stages before the call returns, so it needs no more than the caller's borrow; no device synchronisation is made.
fn upload_ordered<T: Copy>(dst: &mut DevBuf, v: &[T], st: &Stream) -> Result<(), CudaError> {
    if LEGACY_UPLOADS.load(Ordering::Relaxed) {
        return dst.upload(v);
    }
    let n = std::mem::size_of_val(v);
    assert!(n <= dst.bytes(), "upload of {n} bytes into a {}-byte buffer", dst.bytes());
    // SAFETY: `dst` is a device allocation of at least `n` bytes and `v` is `n` readable bytes
    unsafe { st.memcpy_h2d_async(dst.ptr(), v.as_ptr() as *const u8, n) }
}

/// `y[..rows] = layer(x[..rows])` through the decode path of an EXL3 linear, which takes at most [`LINEAR_ROWS`] rows a call: a wider step
/// (a prefill slice of up to [`MAX_PREFILL_ROWS`]) is cut in calls of that width. The linear is row invariant (a row's bits do not depend
/// on the rows around it or on its index in the call), so the cut changes no bit. `x` and `y` are `[rows][k]` and `[rows][n]` of the
/// dtypes given.
#[allow(clippy::too_many_arguments)]
pub fn linear_rows(
    l: &Exl3Weights,
    x: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    x_dtype: Dtype,
    y_dtype: Dtype,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    // contract P2 (`contract.rs`): inside a prefill slice's scope the tensor-core prefill linear takes the whole slice in one call (any
    // row count: the width never picks the arithmetic, P2 (a); its K order is a function of K alone, so a TP column slice is exact)
    if gb_cuda::nemotron_dense_mma::active(rows) && gb_cuda::nemotron_dense_mma::Layer::of_weights(l).supported() {
        return gb_cuda::nemotron_dense_mma::linear(l, x, y, rows, x_dtype, y_dtype, ws, st);
    }
    if rows <= LINEAR_ROWS {
        return l.forward(x, y, rows, x_dtype, y_dtype, ws, st);
    }
    let (xb, yb) = (l.k * x_dtype.size(), l.n * y_dtype.size());
    let mut start = 0;
    while start < rows {
        let m = (rows - start).min(LINEAR_ROWS);
        let xs = x.subview(start * xb, m * xb);
        let mut ys = y.subview(start * yb, m * yb);
        l.forward(&xs, &mut ys, m, x_dtype, y_dtype, ws, st)?;
        start += m;
    }
    Ok(())
}

/// What a step is for. A prefill chunk commits its Mamba state at `keep = rows` through one window record. A window (verify or
/// decode) only reads state; `commit(slot, keep)` moves it later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Prefill,
    Window,
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Host bookkeeping. Plain structs with no device calls: the slot table, the KV block table, the window record log, the pool index math
// and the step validation. The model owns one of each and the tests at the bottom of this file run them without a GPU.
// ---------------------------------------------------------------------------------------------------------------------------------

/// Rows of the widest verify or decode window, and the most rows `commit` can keep.
pub const WINDOW_ROWS: usize = 64;
/// The opt-out prefill slice cap (the default since 2026-10-15 is [`WIDE_PREFILL_ROWS`], 512) and the widest WIDE decode window. A slice above [`WINDOW_ROWS`] takes the large-row MoE prefill path
/// (`gb_cuda::nemotron_moe_prefill`), which reads each expert's weights once for the whole slice: the weights of a layer cost the same
/// at 256 rows as at 64, so the MoE block of a 256-row slice runs at about 2.5 times the rows a second of a 64-row slice and a whole
/// prompt prefills about 1.5 times as fast (`docs/NEMOTRON_MOE_PREFILL.md`). 256 is where the
/// prefill scratch (about 105 MB with the per-pair activations, linear in rows) stays in the budget the plan prices (the attention
/// partials do not grow with the slice: `attn_long` runs it in passes of a window's rows), and rows per expert (about 11 of 256) are enough
/// that the weights' DRAM traffic stops being the limit (the block is then bound by the FMA and decode issue, so 512 rows is only 1.2
/// times the rows a second of 256); the scratch and the kernels take any row count to 512.
pub const MAX_PREFILL_ROWS: usize = 256;
/// The default widest prefill slice (lane S, `docs/NEMOTRON_PREFILL_SPEED.md`; default since 2026-10-15 by the owner's decision): the
/// large-row MoE path, the Mamba record and the attention passes take any row count; the prefill scratch is linear in rows (about 210 MB
/// at 512) and the attention partials do not grow with the slice. A row's bits do not depend on the slice width (`slice_sizes_agree`
/// up to 512, the replies golden, the cache-ON captures), so the width changes speed, never a result: +12 to +14 % single node.
pub const WIDE_PREFILL_ROWS: usize = 512;
/// The prefill slice cap: unset (or `512`) is [`WIDE_PREFILL_ROWS`], the default; `256` is the opt-out to [`MAX_PREFILL_ROWS`] (A/B and
/// fallback). Any other value is ignored with a log line (the default stays). Read once per process. Under TP both ranks must run the
/// same cap: `prefill_rows` and the family's `prefill` fact are in the boot gate, so a mismatch is refused by name.
pub const PREFILL_ROWS_VAR: &str = "REXL3_NEMO_PREFILL_ROWS_MAX";

/// The cap a value of [`PREFILL_ROWS_VAR`] asks for: `Ok(512)` when unset, `Ok(256 | 512)` for those values, `Err` (with the reason)
/// for anything else. Pure, for the tests.
pub fn parse_prefill_rows_cap(v: Option<&str>) -> Result<usize, String> {
    match v.map(str::trim) {
        None | Some("") => Ok(WIDE_PREFILL_ROWS),
        Some(s) => match s.parse::<usize>() {
            Ok(n) if n == MAX_PREFILL_ROWS || n == WIDE_PREFILL_ROWS => Ok(n),
            _ => Err(format!("{PREFILL_ROWS_VAR}={s:?} is not {MAX_PREFILL_ROWS} or {WIDE_PREFILL_ROWS}")),
        },
    }
}

/// The most rows of one prefill slice in this process: [`WIDE_PREFILL_ROWS`] unless [`PREFILL_ROWS_VAR`] opts out to
/// [`MAX_PREFILL_ROWS`].
pub fn prefill_rows_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| match parse_prefill_rows_cap(std::env::var(PREFILL_ROWS_VAR).ok().as_deref()) {
        Ok(n) => {
            if n != WIDE_PREFILL_ROWS {
                eprintln!("rexl3 level=info event=prefill_rows_cap var={PREFILL_ROWS_VAR} cap={n} default={WIDE_PREFILL_ROWS}");
            }
            n
        }
        Err(e) => {
            eprintln!("rexl3 level=warn event=prefill_rows_cap_ignored reason=\"{e}\" cap={WIDE_PREFILL_ROWS}");
            WIDE_PREFILL_ROWS
        }
    })
}
/// The widest step of a model built for prefill slices of `prefill_rows` rows and a decode window of `window_rows`: what its step buffers
/// and its large-row and tensor-core MoE scratches are sized by. At [`WINDOW_ROWS`] it builds neither MoE scratch, so a P2 prefill would
/// run P1's MoE kernels: `contract::check_rows_cap` refuses that boot by name.
pub fn step_rows_cap(prefill_rows: usize, window_rows: usize) -> usize {
    prefill_rows.max(WINDOW_ROWS).max(window_rows)
}
/// The most rows a WIDE decode window may carry (`NemotronModel::new_wide`): many single-row sequences in one step, which reads each
/// expert's weights once for all of them (the large-row MoE path above [`WINDOW_ROWS`], bit-equal to the decode path) instead of once
/// per 64. A window of at most [`WINDOW_ROWS`] rows is the window it always was. The cap is the prefill slice's: the same scratch.
pub const MAX_WINDOW_ROWS: usize = MAX_PREFILL_ROWS;
/// The most candidates per row a step returns.
pub const MAX_CANDIDATES: usize = 64;

/// Bytes of one slot of one Mamba layer's S pool (f32 `[128][128][64]`).
pub const S_SLOT_BYTES: usize = S_SLOT_ELEMS * 4;
/// Bytes of one slot of one Mamba layer's conv pool (bf16 `[3][10240]`).
pub const CONV_SLOT_BYTES: usize = CONV_SLOT_ELEMS * 2;
/// Bytes of one row of a window `Record`: bf16 xBC `[10240]`, bf16 dt `[128]`, f32 g `[128]` and f32 decay `[128]`.
pub const RECORD_ROW_BYTES: usize = CONV_DIM * 2 + HEADS * 2 + HEADS * 4 + HEADS * 4;
/// Bytes of one retained f32 xBC row (the pre-conv rows `commit_conv` replays).
pub const XBC_ROW_BYTES: usize = CONV_DIM * 4;
/// What one window row costs one Mamba layer: its record row plus its retained xBC row (62,720 B).
pub const WINDOW_ROW_BYTES: usize = RECORD_ROW_BYTES + XBC_ROW_BYTES;

/// The Mamba state pools: per layer, f32 S `[slots][128][128][64]` and bf16 conv `[slots][3][10240]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatePools {
    pub slots: usize,
    /// Mamba layers on the device.
    pub layers: usize,
}

impl StatePools {
    /// Byte offset of `slot` in one layer's S pool.
    pub fn s_offset(slot: usize) -> usize {
        slot * S_SLOT_BYTES
    }

    /// Byte offset of `slot` in one layer's conv pool.
    pub fn conv_offset(slot: usize) -> usize {
        slot * CONV_SLOT_BYTES
    }

    /// Bytes one sequence holds across every Mamba layer (170,229,760 over 40 layers).
    pub fn slot_bytes(&self) -> usize {
        self.layers * (S_SLOT_BYTES + CONV_SLOT_BYTES)
    }

    /// Bytes of both pools of every layer.
    pub fn pool_bytes(&self) -> usize {
        self.slots * self.slot_bytes()
    }

    /// Bytes the per-layer window records and retained xBC rows of `rows` rows hold across every Mamba layer.
    pub fn window_bytes(&self, rows: usize) -> usize {
        self.layers * rows * WINDOW_ROW_BYTES
    }
}

/// The paged KV pool: one buffer each for K and V that every attention layer owns a stripe of. Layer `a` (its ordinal among the
/// attention layers) owns blocks `a * pages..(a + 1) * pages`; the block table holds layer-local block ids, shared by every layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvGeom {
    /// Blocks of [`BLOCK_TOKENS`] tokens per layer.
    pub pages: usize,
    /// Attention layers on the device.
    pub layers: usize,
    /// The element type of every stripe (`--kv-dtype`): a page's bytes ([`KvDtype::page_bytes`]).
    pub dtype: KvDtype,
}

impl KvGeom {
    /// Bytes of one layer's K (or V) page: `[256 tokens][2 kv heads][128]` fp16, or the int8 page (values and exponents).
    pub fn page_bytes(&self) -> usize {
        self.dtype.page_bytes()
    }

    /// Bytes of one layer's K (or V) stripe: `pages` pages.
    pub fn layer_bytes(&self) -> usize {
        self.pages * self.page_bytes()
    }

    /// Byte offset of layer `a`'s stripe inside each of K and V.
    pub fn layer_offset(&self, a: usize) -> usize {
        a * self.layer_bytes()
    }

    /// Bytes of each of K and V.
    pub fn half_bytes(&self) -> usize {
        self.layers * self.layer_bytes()
    }

    /// Bytes of K and V together.
    pub fn pool_bytes(&self) -> usize {
        2 * self.half_bytes()
    }

    /// What one token costs in every attention layer, K and V (8 KiB over 8 layers in fp16, 4,128 B in int8).
    pub fn token_bytes(&self) -> usize {
        self.layers * self.dtype.token_bytes()
    }

    /// Blocks `tokens` positions need.
    pub fn pages_for(tokens: usize) -> usize {
        tokens.div_ceil(BLOCK_TOKENS)
    }

    /// The block (index into a slot's block list) and the offset inside it of position `pos`.
    pub fn locate(pos: usize) -> (usize, usize) {
        (pos / BLOCK_TOKENS, pos % BLOCK_TOKENS)
    }
}

/// What a slot is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    Free,
    /// Begun, with this many tokens committed (the Mamba state is at this position and the KV holds its first `len` rows).
    Live(usize),
    /// A call failed after it had begun to change the slot's state, or `begin_slot` failed: nothing may run on it until `end_slot`.
    Poisoned,
}

/// Why the slot table refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SlotError {
    OutOfRange { slot: usize, slots: usize },
    AlreadyBegun { slot: usize },
    NotBegun { slot: usize },
    Unusable { slot: usize },
}

impl fmt::Display for SlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlotError::OutOfRange { slot, slots } => write!(f, "slot {slot} of {slots}"),
            SlotError::AlreadyBegun { slot } => write!(f, "slot {slot} has begun already: end it first"),
            SlotError::NotBegun { slot } => write!(f, "slot {slot} was never begun"),
            SlotError::Unusable { slot } => write!(f, "slot {slot} is unusable after an error: end it and begin it again"),
        }
    }
}

impl From<SlotError> for ModelError {
    fn from(e: SlotError) -> Self {
        ModelError::Request(e.to_string())
    }
}

/// The slot table: which slots are live and how many tokens each has committed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotTable {
    state: Vec<SlotState>,
}

impl SlotTable {
    pub fn new(slots: usize) -> Self {
        Self { state: vec![SlotState::Free; slots] }
    }

    pub fn slots(&self) -> usize {
        self.state.len()
    }

    /// Starts a sequence in `slot` with nothing committed. A slot that has begun is refused until it is ended.
    pub fn begin(&mut self, slot: usize) -> Result<(), SlotError> {
        match self.state.get(slot) {
            None => Err(SlotError::OutOfRange { slot, slots: self.state.len() }),
            Some(SlotState::Free) => {
                self.state[slot] = SlotState::Live(0);
                Ok(())
            }
            Some(_) => Err(SlotError::AlreadyBegun { slot }),
        }
    }

    /// Ends the sequence in `slot`. Idempotent: a free or out-of-range slot is left alone. Returns whether the slot held a sequence.
    pub fn end(&mut self, slot: usize) -> bool {
        match self.state.get_mut(slot) {
            Some(s) if *s != SlotState::Free => {
                *s = SlotState::Free;
                true
            }
            _ => false,
        }
    }

    /// Marks `slot` unusable until it is ended (a call failed after it began to change the slot).
    pub fn poison(&mut self, slot: usize) {
        if let Some(s) = self.state.get_mut(slot) {
            if *s != SlotState::Free {
                *s = SlotState::Poisoned;
            }
        }
    }

    /// Tokens committed in a live slot.
    pub fn len(&self, slot: usize) -> Result<usize, SlotError> {
        match self.state.get(slot) {
            None => Err(SlotError::OutOfRange { slot, slots: self.state.len() }),
            Some(SlotState::Free) => Err(SlotError::NotBegun { slot }),
            Some(SlotState::Poisoned) => Err(SlotError::Unusable { slot }),
            Some(SlotState::Live(n)) => Ok(*n),
        }
    }

    pub fn is_live(&self, slot: usize) -> bool {
        matches!(self.state.get(slot), Some(SlotState::Live(_)))
    }

    /// Adds `n` committed tokens to a live slot and returns the new length.
    pub fn advance(&mut self, slot: usize, n: usize) -> Result<usize, SlotError> {
        let len = self.len(slot)?;
        self.state[slot] = SlotState::Live(len + n);
        Ok(len + n)
    }
}

/// One sequence of a step: its slot, its rows `start..start + rows` of the batch and the position of its first row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqPlan {
    pub slot: usize,
    pub start: usize,
    pub rows: usize,
    pub first_pos: usize,
}

/// What a step is checked against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepLimits {
    /// The most rows the step takes (`prefill_rows` for a prefill slice, [`WINDOW_ROWS`] for a window).
    pub max_rows: usize,
    /// Positions must be below this.
    pub max_context: usize,
    /// Token ids must be below this.
    pub vocab: usize,
}

/// Checks a batch against the slot table before anything touches the device (a wrapper's size assertion must never be the check) and
/// returns its sequences. A batch is refused when:
///
/// - it has no rows, more rows than `lim.max_rows`, parents (a tree) or M-RoPE streams (this is a text model);
/// - its arrays disagree, a sequence is empty, or two sequences share a slot;
/// - a slot is out of range, never begun or unusable;
/// - a sequence does not start at its slot's committed length, or its positions are not consecutive, or one is at or past
///   `lim.max_context`;
/// - a token id is outside the vocabulary.
pub fn plan_step(b: &RowBatch, lim: &StepLimits, slots: &SlotTable) -> Result<Vec<SeqPlan>, ModelError> {
    let req = |m: String| Err(ModelError::Request(m));
    if b.parents.is_some() {
        return req("a Nemotron window of trees is not built".into());
    }
    if b.rope3.is_some() {
        return req("a Nemotron step is a plain text chain (no M-RoPE streams)".into());
    }
    if b.rows == 0 {
        return req("a step needs at least one row".into());
    }
    if b.rows > lim.max_rows {
        return req(format!("a step takes 1..={} rows, got {}", lim.max_rows, b.rows));
    }
    if b.ids.len() != b.rows || b.positions.len() != b.rows || b.row_slot.len() != b.rows {
        return req(format!(
            "the batch says {} rows but holds {} ids, {} positions and {} row slots",
            b.rows,
            b.ids.len(),
            b.positions.len(),
            b.row_slot.len()
        ));
    }
    let n_seq = b.seq_slot.len();
    if n_seq == 0 || b.seq_start.len() != n_seq + 1 || b.seq_start[0] != 0 || b.seq_start[n_seq] as usize != b.rows {
        return req(format!("the batch's sequence arrays do not cover its {} rows", b.rows));
    }
    if let Some(&id) = b.ids.iter().find(|&&id| id < 0 || id as usize >= lim.vocab) {
        return req(format!("token id {id} is outside the vocabulary of {}", lim.vocab));
    }
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(n_seq);
    for i in 0..n_seq {
        let (lo, hi) = (b.seq_start[i], b.seq_start[i + 1]);
        if lo < 0 || hi <= lo {
            return req(format!("sequence {i} has no rows"));
        }
        let (lo, hi) = (lo as usize, hi as usize);
        if b.seq_slot[i] < 0 {
            return req(format!("sequence {i} has slot {}", b.seq_slot[i]));
        }
        let slot = b.seq_slot[i] as usize;
        if !seen.insert(slot) {
            return req(format!("slot {slot} is in the batch twice"));
        }
        let len = slots.len(slot)?;
        if b.row_slot[lo..hi].iter().any(|&s| s != b.seq_slot[i]) {
            return req(format!("a row of sequence {i} names a slot other than {slot}"));
        }
        let first = b.positions[lo];
        if first < 0 || first as usize != len {
            return req(format!("slot {slot} has {len} tokens committed, so its next row is position {len}, not {first}"));
        }
        if b.positions[lo..hi].windows(2).any(|p| p[1] != p[0] + 1) {
            return req(format!("the positions of slot {slot} are not consecutive"));
        }
        let last = len + (hi - lo) - 1;
        if last >= lim.max_context {
            return req(format!("position {last} of slot {slot} is past the context of {}", lim.max_context));
        }
        out.push(SeqPlan { slot, start: lo, rows: hi - lo, first_pos: len });
    }
    Ok(out)
}

/// Grows the host block table so every sequence holds the pages of its last position. Returns whether any slot's list changed (the device
/// table must then be uploaded). A sequence that does not fit is a [`ModelError::Request`]; sequences before it stay grown, which only
/// costs them pages their next step takes anyway.
pub fn grow_pages(tab: &mut KvTables, seqs: &[SeqPlan]) -> Result<bool, ModelError> {
    let mut changed = false;
    for s in seqs {
        let need = s.first_pos + s.rows;
        if KvGeom::pages_for(need) > tab.blocks_of(s.slot).len() {
            tab.ensure(s.slot, need).map_err(|e| ModelError::Request(format!("KV pages of slot {}: {e:?}", s.slot)))?;
            changed = true;
        }
    }
    Ok(changed)
}

/// The per-slot block lists the device table holds, in slot order.
pub fn block_lists(tab: &KvTables, slots: usize) -> Vec<Vec<u32>> {
    (0..slots).map(|s| tab.blocks_of(s).to_vec()).collect()
}

/// Why a commit was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommitError {
    /// The slot was not in the last window, or its rows have been committed (or dropped) already.
    NoWindow {
        slot: usize,
    },
    /// `keep` is zero: a window keeps its first row at least.
    KeepZero {
        slot: usize,
    },
    KeepPastWindow {
        slot: usize,
        keep: usize,
        rows: usize,
    },
}

impl fmt::Display for CommitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CommitError::NoWindow { slot } => {
                write!(
                    f,
                    "commit of slot {slot}: it has no window to commit (none ran, or its rows are committed or were dropped)"
                )
            }
            CommitError::KeepZero { slot } => {
                write!(f, "commit of slot {slot}: keep is 0, a window keeps at least its first row")
            }
            CommitError::KeepPastWindow { slot, keep, rows } => {
                write!(f, "commit of slot {slot}: keep {keep} is past the {rows} rows of its window")
            }
        }
    }
}

impl From<CommitError> for ModelError {
    fn from(e: CommitError) -> Self {
        ModelError::Request(e.to_string())
    }
}

/// The rows a commit replays: rows `start..end` of the window are the sequence's, and the first `keep` of them are kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitSpan {
    pub slot: usize,
    pub start: usize,
    pub end: usize,
    pub keep: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WindowSeq {
    slot: usize,
    start: usize,
    rows: usize,
    open: bool,
    /// The window already moved this sequence's Mamba state (a single row applied in the pass that read it): its commit only moves the
    /// head's taps and the slot's length.
    applied: bool,
}

/// The last window's rows per slot. The per-layer records hold exactly one window, so a new window replaces the log, and a slot's rows
/// can be committed once.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WindowLog {
    seqs: Vec<WindowSeq>,
}

impl WindowLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// The records are about to be overwritten: nothing of the previous window can be committed any more.
    pub fn clear(&mut self) {
        self.seqs.clear();
    }

    /// Records the window that has just filled the records. Replaces the log.
    pub fn record(&mut self, seqs: &[SeqPlan]) {
        self.seqs =
            seqs.iter().map(|s| WindowSeq { slot: s.slot, start: s.start, rows: s.rows, open: true, applied: false }).collect();
    }

    /// `slot`'s rows of the last window were applied to its state by the window itself (`WindowSeq::applied`).
    pub fn mark_applied(&mut self, slot: usize) {
        for s in self.seqs.iter_mut().filter(|s| s.slot == slot && s.open) {
            s.applied = true;
        }
    }

    /// Whether `slot`'s open rows were applied by the window itself.
    pub fn is_applied(&self, slot: usize) -> bool {
        self.seqs.iter().any(|s| s.slot == slot && s.open && s.applied)
    }

    /// Rows of `slot` in the last window that nothing has committed yet.
    pub fn pending(&self, slot: usize) -> Option<usize> {
        self.seqs.iter().find(|s| s.slot == slot && s.open).map(|s| s.rows)
    }

    /// Takes the first `keep` rows of `slot`'s part of the last window. A second commit for the same window is refused.
    pub fn commit(&mut self, slot: usize, keep: usize) -> Result<CommitSpan, CommitError> {
        let s = self.seqs.iter_mut().find(|s| s.slot == slot && s.open).ok_or(CommitError::NoWindow { slot })?;
        if keep == 0 {
            return Err(CommitError::KeepZero { slot });
        }
        if keep > s.rows {
            return Err(CommitError::KeepPastWindow { slot, keep, rows: s.rows });
        }
        s.open = false;
        Ok(CommitSpan { slot, start: s.start, end: s.start + s.rows, keep })
    }

    /// Drops `slot`'s rows (its state changed some other way, or it ended).
    pub fn forget(&mut self, slot: usize) {
        for s in self.seqs.iter_mut().filter(|s| s.slot == slot) {
            s.open = false;
        }
    }
}

/// Where the oracle harness can listen. Each tap is the f32 buffer `[rows, width]` after the point, on the model's stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Point {
    /// The embedding rows, `[rows, 4096]`.
    Embed,
    /// The mixer output of trunk layer `i` before the residual add, `[rows, 4096]`.
    Mixer(usize),
    /// The residual after trunk layer `i`, `[rows, 4096]`.
    Layer(usize),
    /// `norm_f` of the last layer's residual, `[rows, 4096]`.
    FinalNorm,
    /// The logits, `[rows, 131072]`.
    Logits,
}

/// An observer the harness installs on a model. A model without one pays nothing.
pub trait Tap {
    fn tap(&mut self, point: Point, buf: &DevBuf, rows: usize, width: usize, st: &Stream) -> Result<(), ModelError>;
}

/// The working buffers of a step (f32 unless a field says bf16), sized for `rows_cap` rows. Row `r` of a buffer is row `r` of the batch, so a row's bits do not
/// depend on its neighbours.
pub struct Bufs {
    pub rows_cap: usize,
    /// The residual stream, `[rows, 4096]`.
    pub x: DevBuf,
    /// The normed input of the mixer, `[rows, 4096]`.
    pub xn: DevBuf,
    /// The mixer's output, `[rows, 4096]`.
    pub mix: DevBuf,
    /// The `in_proj` output, `[rows, 18560]`, and its split: z `[rows, 8192]`, xBC `[rows, 10240]`, dt `[rows, 128]`.
    pub proj: DevBuf,
    pub z: DevBuf,
    pub xbc: DevBuf,
    pub dt: DevBuf,
    /// The scan's core output and the gated-norm output, both bf16 `[rows, 8192]` (what `nemotron_ssm::window` and `gated_norm`
    /// write); `out_proj` reads the second.
    pub ssm_y: DevBuf,
    pub gated: DevBuf,
    /// Attention: q `[rows, 4096]`, k and v `[rows, 256]`, the merged output `[rows, 4096]`.
    pub q: DevBuf,
    pub k: DevBuf,
    pub v: DevBuf,
    pub attn_o: DevBuf,
    /// The logits `[rows, 131072]` and the top-k outputs.
    pub logits: DevBuf,
    pub topk_ws: DevBuf,
    pub topk_ids: DevBuf,
    pub topk_vals: DevBuf,
    /// The workspace of `Exl3Weights::forward`: the largest `workspace_bytes(rows)` of any layer's linear.
    pub ws: DevBuf,
    // ---- tensor parallelism (`tp_arm.rs`): `None` on a single rank, which allocates and runs none of it ----
    /// A sharded linear's local output, `[rows, n_local]` f32 (room for the widest slice: the Mamba `in_proj`'s 9,344 columns).
    pub tp_tmp: Option<DevBuf>,
    /// One attention layer's q, k and v, block by block (`[rows, 4096]`, `[rows, 256]`, `[rows, 256]`), under one collective.
    pub tp_qkv: Option<DevBuf>,
    /// The rank's rows of the logits, `[rows, vocab / 2]` f32, for a window's rows at most ([`WINDOW_ROWS`]).
    pub tp_logits: Option<DevBuf>,
    /// Contract P2 under two-rank TP only (lane S): the packed send and receive rows of the `in_proj` exchange (`tp_arm::linear_tp_p2`).
    pub tp_pack: Option<(DevBuf, DevBuf)>,
    /// Contract P2 under two-rank TP only (lane S): the packed send and receive `y` rows of the MoE pair gather (`tp_arm::moe_prefill_with`).
    pub tp_ypack: Option<(DevBuf, DevBuf, DevBuf)>,
    /// Contract P2 under two-rank TP only (lane S): the exchange stream and its events (the in_proj exchange of the first row half runs
    /// on it while the second half computes).
    pub tp_aux: Option<(gb_cuda::Stream, [gb_cuda::rt::Event; 3])>,
    /// Contract P2 at two TP ranks with the head-group Mamba split (lane S): the send and receive bytes of the per-layer state
    /// re-replication (each slot's S of the rank's 64 heads, 2 MiB, and its conv history of the rank's channels, 30 KiB).
    pub tp_state: Option<(DevBuf, DevBuf)>,
}

/// See [`Core::moe_half`].
pub struct MoeHalf {
    pub prefill: PrefillScratch,
    pub mma: gb_cuda::nemotron_moe_prefill::mma::MmaScratch,
    pub ypack: (DevBuf, DevBuf, DevBuf),
    pub stream: gb_cuda::Stream,
    pub ev: [gb_cuda::rt::Event; 4],
}

impl MoeHalf {
    pub fn new(rows: usize) -> Result<Self, CudaError> {
        let b = rows * gb_cuda::nemotron_moe::TOPK * gb_cuda::nemotron_moe::LATENT * 4;
        let ev = || gb_cuda::rt::Event::new();
        Ok(Self {
            prefill: PrefillScratch::new(rows)?,
            mma: gb_cuda::nemotron_moe_prefill::mma::MmaScratch::new(rows)?,
            ypack: (DevBuf::zeroed(b)?, DevBuf::zeroed(b)?, DevBuf::zeroed(rows * gb_cuda::nemotron_moe::TOPK * 4)?),
            stream: gb_cuda::Stream::new_nonblocking()?,
            ev: [ev()?, ev()?, ev()?, ev()?],
        })
    }

    /// The device bytes [`MoeHalf::new`] allocates.
    pub fn bytes(rows: usize) -> u64 {
        let b = (rows * gb_cuda::nemotron_moe::TOPK * gb_cuda::nemotron_moe::LATENT * 4) as u64;
        PrefillScratch::bytes(rows)
            + gb_cuda::nemotron_moe_prefill::mma::MmaScratch::bytes(rows)
            + 2 * b
            + (rows * gb_cuda::nemotron_moe::TOPK * 4) as u64
    }
}

/// One Mamba layer's per-slot state: the SSD state f32 `[slots][128][128][64]` and the conv rows bf16 `[slots][3][10240]`.
pub struct MambaState {
    pub s: DevBuf,
    pub conv: DevBuf,
}

/// Everything a layer arm touches. The arms take `&mut Core` and a [`StepCtx`], read `bufs.xn` and write `bufs.mix`.
pub struct Core {
    pub w: NemotronWeights,
    pub opts: ModelOptions,
    pub bufs: Bufs,
    /// The paged KV pool shared by the attention layers. K and V are each one buffer of `layers` stripes (see [`KvGeom`]): attention
    /// layer `a` owns blocks `a * kv_geom.pages..(a + 1) * kv_geom.pages` and reads them through [`Core::kv_layer`], so `n_blocks` here
    /// is the pool's total and one block table (layer-local block ids) serves every layer. The MTP head's attention layer, when the model
    /// carries the head, is the stripe after the trunk's (`kv_geom.layers - 1`).
    pub kv: NemoKv,
    /// Per Mamba layer, indexed by the layer's ordinal among the Mamba layers.
    pub mamba: Vec<MambaState>,
    /// One MoE scratch for all 40 MoE layers (about 100 MB), for the windows and the slices of up to [`WINDOW_ROWS`] rows.
    pub moe: MoeScratch,
    /// The large-row prefill path's scratch, one for all 40 layers, when the model takes slices wider than a window.
    pub moe_prefill: Option<PrefillScratch>,
    pub tap: Option<Box<dyn Tap>>,
    /// Set by the layer loop when the layer's norm already wrote the Mamba in_proj's rotated input into `bufs.ws` (lane S5,
    /// `arm_mamba::prerot_ok`); the Mamba arm takes and clears it.
    pub prerot: bool,
    pub stream: Stream,

    // ---- added by the composer (additive: the fields above are the frozen ones) ----
    /// Which loaded layer is which kind, and its ordinal among that kind.
    pub map: LayerMap,
    pub kv_geom: KvGeom,
    /// The device block table `[slots][max_blocks]`, shared by every attention layer.
    pub table: BlockTable,
    /// The attention partials, sized for a window's rows (`min(rows_cap, WINDOW_ROWS)`) and `max_context` keys: a wider call is run in
    /// passes of that many rows.
    pub attn: AttnScratch,
    /// The record a prefill slice's Mamba layers share: a slice commits inside the layer, so one record serves them all.
    pub pre_rec: Record,
    /// Per Mamba layer, the record of the last window (`WINDOW_ROWS` rows). `commit` replays it.
    pub win_rec: Vec<Record>,
    /// Per Mamba layer, the f32 xBC rows of the last window (`WINDOW_ROWS` rows): the conv history of a commit is rebuilt from them.
    pub win_xbc: Vec<DevBuf>,
    /// The keep of each sequence of the current prefill slice (i32 `[slots]`): all of its rows.
    pub keep: DevBuf,
    /// The MTP head's device state (its tap store, scratch, book per slot), when the model carries the head (`families::nemotron_h_omni::mtp`).
    pub mtp: Option<super::mtp::MtpState>,
    /// The image rows of the prefill slice being run: written over the embedding rows right after the gather, and read again by the head's
    /// hook for its own embedding half (`prefill_chunk_mm`). Empty outside that call and for every window: a window's launches are what a
    /// graph captures, and features live outside the graph.
    pub splice: Vec<SpliceOp>,
    /// The batch rows whose candidates a prefill slice of several prompts wants (`NemotronModel::prefill_chunk_rows`), in the order of
    /// the result's rows. Empty outside that call.
    pub gather: Vec<usize>,
    /// Per Mamba layer, the ring of a LAGGED decode (empty unless the model was built with a fold period, [`LAG_ENV`]): the records of the
    /// rows a single-row sequence has decoded since its state was last folded into the S pool.
    pub lag_rings: Vec<LagRing>,
    /// The tensor-parallel group this rank belongs to (`SingleRank` for the single-node engine: no scatter, no collective, no temp).
    pub tp: Arc<dyn TpComm>,
    /// Contract P2 only (lane S): the tensor-core MoE block's scratch (`gb_cuda::nemotron_moe_prefill::mma::MmaScratch`, lane S2), as
    /// wide as the prefill scratch. `None` under P1.
    pub moe_mma: Option<gb_cuda::nemotron_moe_prefill::mma::MmaScratch>,
    /// Contract P2 under two-rank TP only (lane S): the second half of a wide prefill slice's MoE block runs on its own stream with its
    /// own scratches while the first half's `y` exchange is on the wire (`arm_moe`): prefill scratch, tensor-core scratch, the gather's
    /// send / receive / map buffers, the stream, and events (fork, join, the first half's and the second half's after-bucket).
    pub moe_half: Option<MoeHalf>,
}

/// One step's rows. `batch` and `row_index` are built by the model once per step and shared by every layer.
pub struct StepCtx<'a> {
    pub rows: usize,
    pub kind: Kind,
    pub batch: Batch<'a>,
    pub row_index: &'a RowIndex,
    /// A window whose single-row sequences are applied in the pass that reads their state ([`NemotronModel::window_candidates_fused`]):
    /// i32 `[n_seq]` on the device, 1 for a sequence whose one row is kept at once (its S is stored by `window_commit_sel` and its conv
    /// history by `commit_conv` inside the layer), 0 for a sequence left to a later `commit`. `None` for every other step.
    pub apply: Option<&'a DevBuf>,
    /// With `apply`, a LAGGED window: i32 `[n_seq]` on the device, the rows each applied sequence has pending in its layer's ring. `None`:
    /// an applied sequence stores its state at once (`window_commit_sel`).
    pub lag: Option<&'a DevBuf>,
}

impl Core {
    /// Attention layer `a`'s stripe of the KV pool as the kernels take it: views into the pool, nothing is copied.
    pub fn kv_layer(&self, a: usize) -> NemoKv {
        let g = self.kv_geom;
        NemoKv {
            k: self.kv.k.subview(g.layer_offset(a), g.layer_bytes()),
            v: self.kv.v.subview(g.layer_offset(a), g.layer_bytes()),
            n_blocks: g.pages,
            dtype: g.dtype,
        }
    }

    /// Runs layer `li`'s arm: `bufs.mix = mixer(bufs.xn)`.
    /// Runs layer `li`'s mixer. `Ok(true)`: the arm added its output to the residual stream `bufs.x` itself (a P2 prefill slice's fused
    /// residual, [`Core::residual_fused`]) and `bufs.mix` was not written; `Ok(false)`: the output is in `bufs.mix`.
    pub fn run_arm(&mut self, li: usize, x: &StepCtx<'_>) -> Result<bool, ModelError> {
        let kind = match self.w.layer(li) {
            Some(l) => match &l.mixer {
                Mixer::Mamba(_) => 0,
                Mixer::Attn(_) => 1,
                Mixer::Moe(_) => 2,
            },
            None => return Err(ModelError::Request(format!("layer {li} is not on the device"))),
        };
        match kind {
            0 => arm_mamba::run(self, li, x),
            1 => arm_attn::run(self, li, x),
            _ => arm_moe::run(self, li, x),
        }
    }

    /// Whether a mixer may add its output projection into `bufs.x` in its GEMM epilogue (lane S5): a P2 prefill slice of a single rank
    /// with no tap installed (a tap reads `bufs.mix`), unless `REXL3_NEMO_P2_FUSE_RESID=0`. Bit for bit the separate `add_f32`: the
    /// epilogue's last operation is `out + x`, the add's is `x + out`.
    pub fn residual_fused(&self, x: &StepCtx<'_>) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        x.kind == Kind::Prefill
            && self.tap.is_none()
            && self.tp.topology().is_single()
            && gb_cuda::nemotron_dense_mma::active(x.rows)
            && *ON.get_or_init(|| std::env::var("REXL3_NEMO_P2_FUSE_RESID").map_or(true, |v| v.trim() != "0"))
    }

    /// Whether the MoE block of a P2 prefill slice on a rank of TWO may add its output into `bufs.x` in fc2's epilogue
    /// (`docs/NEMOTRON_PREFILL_SPEED.md` 6.11; `tp_arm::moe_prefill_with(.., resid)`): MoE only, because there the cross-rank
    /// exchange of `y` happens before the combine and fc2 is replicated, so every rank adds `x` to the complete output. NOT for the
    /// Mamba or attention epilogues, whose TP outputs are partial sums before their allreduce ([`Core::residual_fused`] stays
    /// single-rank). No tap installed (a tap reads `bufs.mix`); opt-outs `REXL3_NEMO_P2TP_RESID=0` (this one) and
    /// `REXL3_NEMO_P2_FUSE_RESID=0` (every fused residual). Bit-neutral, so not a boot-gate fact.
    pub fn residual_fused_moe_tp(&self, x: &StepCtx<'_>) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static ON_ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        x.kind == Kind::Prefill
            && self.tap.is_none()
            && self.tp.topology().ranks == 2
            && gb_cuda::nemotron_dense_mma::active(x.rows)
            && *ON_ALL.get_or_init(|| std::env::var("REXL3_NEMO_P2_FUSE_RESID").map_or(true, |v| v.trim() != "0"))
            && *ON.get_or_init(|| std::env::var(RESID_TP_VAR).map_or(true, |v| v.trim() != "0"))
    }
}

/// The opt-out of the fused MoE residual add at two ranks ([`Core::residual_fused_moe_tp`]): on unless `0`.
pub const RESID_TP_VAR: &str = "REXL3_NEMO_P2TP_RESID";

/// Checks `opts` against the pack's config before a weight is read, and returns the number of trunk layers to load. A size the model
/// cannot be built for is a [`ModelError::Config`].
pub fn check_options(opts: &ModelOptions, cfg: &OmniConfig) -> Result<usize, ModelError> {
    let bad = |m: String| Err(ModelError::Config(format!("{opts:?}: {m}")));
    let trunk = cfg.pattern.chars().count();
    let n = opts.layers.unwrap_or(trunk);
    if n == 0 || n > trunk {
        return bad(format!("{n} trunk layers of {trunk}"));
    }
    if opts.slots == 0 || opts.kv_pages == 0 || opts.max_context == 0 || opts.prefill_rows == 0 {
        return bad("every size must be positive".into());
    }
    let contract = match super::contract::check_selected() {
        Ok(c) => c,
        Err(e) => return bad(e),
    };
    if let Err(e) = super::contract::check_built(contract) {
        return bad(e);
    }
    let cap = contract.max_prefill_rows(prefill_rows_cap());
    if opts.prefill_rows > cap {
        return bad(format!("a prefill slice of {} rows is past the {cap} the model takes", opts.prefill_rows));
    }
    if opts.max_context > cfg.max_context {
        return bad(format!("a context of {} tokens is past the model's {}", opts.max_context, cfg.max_context));
    }
    if opts.max_context > opts.kv_pages * BLOCK_TOKENS {
        return bad(format!(
            "a context of {} tokens needs more than the {} pages of {BLOCK_TOKENS} the pool holds",
            opts.max_context, opts.kv_pages
        ));
    }
    if opts.max_context.div_ceil(gb_cuda::nemotron_attn::CHUNK_KEYS) > gb_cuda::nemotron_attn::MAX_CHUNKS {
        return bad("the context needs more attention chunks than the partials hold".into());
    }
    Ok(n)
}

/// Tensor parallelism (a group of more than one rank) is built, and verified byte-identical, for the 64-row decode window and the plain
/// decode step only: a wide window or the lagged decode on a rank of a group is refused by name rather than run unverified.
fn check_tp_plain(single: bool, window_rows: usize, lag_k: usize) -> Result<(), ModelError> {
    if single {
        return Ok(());
    }
    if window_rows > WINDOW_ROWS {
        return Err(ModelError::Config(format!(
            "a decode window of {window_rows} rows (REXL3_NEMO_WINDOW_ROWS / --window-rows) is not built for tensor parallelism \
             (--tp-ranks 2 takes the {WINDOW_ROWS}-row window); boot a single rank for the wide window"
        )));
    }
    if (2..=16).contains(&lag_k) {
        return Err(ModelError::Config(format!(
            "the lagged decode ({LAG_ENV}={lag_k}) is not built for tensor parallelism (--tp-ranks 2); unset {LAG_ENV} or set it to 0"
        )));
    }
    Ok(())
}

/// The decode window a model can be built for: [`WINDOW_ROWS`]..=[`MAX_WINDOW_ROWS`] rows. A narrower window is the same model (a
/// caller that wants fewer rows asks the engine for fewer), so the floor is the 64 the buffers and records are always sized for.
pub fn check_window_rows(window_rows: usize) -> Result<(), ModelError> {
    if !(WINDOW_ROWS..=MAX_WINDOW_ROWS).contains(&window_rows) {
        return Err(ModelError::Config(format!(
            "a decode window of {window_rows} rows: the model takes {WINDOW_ROWS}..={MAX_WINDOW_ROWS}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------------------------
// CUDA graphs of the decode windows.
//
// A verify or decode window (Kind::Window, every row's logits) is the same list of ~700 launches every time; what changes between two windows
// of one shape is what the kernels READ from device buffers (token ids, slots, positions, the sequence layout, the block table, the pools and
// the caches), all written in the stream's order into fixed buffers before the launch (`upload_ordered`). A graph bakes in every other
// argument. The audit of what the launch sequence hands a launcher that is not read from a device buffer:
//
// | host value | what it decides | in the key |
// |---|---|---|
// | `rows` | every launch's row count and grid; the MoE router's row block; `Exl3Weights::forward`'s m; the GEMV's row tile; top-k's plan | yes |
// | `n_seq` | the Mamba kernels' sequence count and grid (their `seq_start` and `slot` arrays are device buffers) | yes |
// | `count` | candidates per row: top-k's count and the downloads | yes |
// | attention chunks | the partial launch's grid z. Rounded up to a power of two (a row with fewer chunks exits early, same bits), at most the scratch's | yes |
// | flash splits, row blocks | above 16 chunks every attention call also launches the flash path (`nemotron_attn_prefill::attn_long`): its grid z is `ceil(chunks / 16)` splits and its grid x a bound of `rows` and `n_seq`, and the tree's grid z is 16; which rows each takes is read from the device's position array. A function of the three fields above, so no new field: a graph of a bucket serves every context of it, and a launch for more splits or blocks than a replay needs gives the same bits (`nemotron_graph_probe`) | no (derived) |
// | kind, head | prefill slices and `Head::Row` take other launches | only windows over every row are cached |
// | tap, MTP mutants | extra launches and host work in the loop | such a window runs eagerly |
//
// Policy (as the GLM model's): the first window of a shape runs eagerly, the second captures it and replays it, every later one replays. A
// capture that fails for a non-sticky reason (the runtime refusing a call, a launcher's argument error) marks the shape eager for the
// rest of the process and runs the window eagerly: a capture never fails a request. A sticky device error does, as for a launch. The
// cache is bounded (`GRAPH_LIMIT` graphs, least recently used out) and a graph is destroyed only after the stream has drained.
// ---------------------------------------------------------------------------------------------------------------------------------

/// Instantiated graphs kept at once.
pub const GRAPH_LIMIT: usize = 64;

/// The environment variable that switches graphs: `0`, `off`, `false` or `no` is off, anything else (and unset) is on.
pub const GRAPHS_ENV: &str = "REXL3_NEMO_GRAPHS";

/// The fold period `K` of a LAGGED decode (`2..=16`; unset, `0` or `1`: off): a single-row sequence's Mamba state is stored to the S pool
/// once every `K` steps instead of every step, the rows between wait in a per-slot ring and are re-applied (the same sequential advance,
/// so the same bits) by the next step's kernel. The state of a sequence is read once and written once a `K`th of the steps, which is
/// about 0.6 of the bytes at `K = 4`. Every other consumer of a slot's state (a window of several rows, a prefill slice, a snapshot, a
/// test's state download) folds the slot first.
pub const LAG_ENV: &str = "REXL3_NEMO_LAG";

/// The fold period asked for by [`LAG_ENV`] (0: off).
pub fn lag_period() -> usize {
    std::env::var(LAG_ENV).ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|k| (2..=16).contains(k)).unwrap_or(0)
}

/// What identifies a captured window: every host value that reaches a launcher and is not read from a device buffer (see the table above).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphKey {
    pub rows: usize,
    pub n_seq: usize,
    pub count: usize,
    /// Attention chunks the partial launch was built for (0 when the model has no attention layer).
    pub chunks: usize,
    /// The window applies its single-row sequences inside the layers (`StepCtx::apply`): other kernels run in the Mamba layers
    /// (`window_commit_sel` and `commit_conv` for `window`), and the apply flags are read from a device buffer, so which sequences
    /// are flagged is not in the key.
    pub fused: bool,
}

enum GraphSlot {
    /// Run once eagerly; the next window of the shape captures.
    Seen,
    /// The window's parts, launched in order (`GraphCache::parts` of them), and when it was last used.
    Ready(Vec<Graph>, u64),
    /// A capture of this shape failed: eager from now on.
    Eager,
}

/// Graphs a window is captured in (`REXL3_NEMO_GRAPH_PARTS`, 1 to 16): the layers in that many runs of consecutive layers. A replay
/// launches them in order and the device starts on the first while the host launches the rest.
pub const GRAPH_PARTS_ENV: &str = "REXL3_NEMO_GRAPH_PARTS";
const GRAPH_PARTS_DEFAULT: usize = 4;
/// The default of a rank of a tensor-parallel group: ONE graph a window. Measured on the two-node group (2026-10-11 05:00 CDT, the split
/// head, same binary, two interleaved pairs): 4 parts 56.6 / 56.5 tok/s with the verify phase 45.67 / 45.48 ms, 2 parts 56.8 / 58.0 with
/// 45.10 / 44.35, 1 part **58.8 / 58.7 with 43.65 / 43.68**, the same token ids in all six. The window's collectives open a part with a
/// 0.5 to 0.8 ms bubble on the device (`docs/NEMOTRON_TP.md` 9.2) and the host launch the parts hide on one node is cheaper than that.
const GRAPH_PARTS_DEFAULT_GROUP: usize = 1;

/// What the cache did, for tests and the server's log.
#[derive(Clone, Debug, Default)]
pub struct GraphStats {
    pub replays: u64,
    pub captures: u64,
    pub eager_windows: u64,
    pub capture_failures: u64,
    pub evictions: u64,
    pub capture_ms: f64,
    pub last_failure: Option<String>,
}

struct GraphCache {
    on: bool,
    slots: HashMap<GraphKey, GraphSlot>,
    tick: u64,
    stats: GraphStats,
    parts: usize,
}

impl GraphCache {
    /// `group`: a rank of a tensor-parallel group ([`GRAPH_PARTS_DEFAULT_GROUP`] parts unless the environment says otherwise).
    fn new(group: bool) -> Self {
        let on = !matches!(std::env::var(GRAPHS_ENV).as_deref(), Ok("0" | "off" | "false" | "no"));
        let default = if group { GRAPH_PARTS_DEFAULT_GROUP } else { GRAPH_PARTS_DEFAULT };
        let parts =
            std::env::var(GRAPH_PARTS_ENV).ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(default).clamp(1, 16);
        Self { on, slots: HashMap::new(), tick: 0, stats: GraphStats::default(), parts }
    }
}

/// The chunk count a window of the attention layers is launched with when graphs are on: the deepest row's, up to a power of two and at
/// most the scratch's capacity `cap`. A graph serves every context of its bucket.
pub fn chunk_bucket(real: usize, cap: usize) -> usize {
    real.max(1).next_power_of_two().min(cap.max(1)).max(real)
}

/// Whether an error a launch returned while capturing is the device's (fatal, as for any launch) and not the capture's own refusal: the
/// 900-series codes ("this call cannot be captured") spoil the capture only, whatever `CudaError::is_fatal` says of a code it does not know.
fn closure_is_fatal(e: &ModelError) -> bool {
    match e {
        ModelError::Cuda(c) if gb_cuda::graph::is_capture_code(c.0) => false,
        other => other.is_fatal(),
    }
}

/// The launches of one step once its inputs are in the device buffers: embed, every layer (norm, arm, residual add), the final norm, the
/// MTP head's tap copy, and for `head` the lm_head and top-k. Returns the rows of candidates (0 for [`Head::None`]). Asynchronous: no
/// sync, no allocation and no read from the host, so the same function is what a graph captures.
fn launch_steps(
    c: &mut Core,
    ids: &DevBuf,
    eps: f32,
    vocab: usize,
    ctx: &StepCtx<'_>,
    head: Head,
    count: usize,
) -> Result<usize, ModelError> {
    launch_steps_part(c, ids, eps, vocab, ctx, head, count, 0, 1)
}

/// Part `part` of `parts` of [`launch_steps`]: the layers cut in `parts` runs of consecutive layers, the embedding with the first run, the
/// final norm, the head's tap copy and the lm_head and top-k with the last. One call per part in order is the launch sequence of
/// `launch_steps`, launch for launch. A replayed window is its parts' graphs launched in order: the GPU starts on the first part while the
/// host is still launching the others (a graph of a window's 1,086 nodes takes 0.55 ms to launch, with the device idle).
#[allow(clippy::too_many_arguments)]
fn launch_steps_part(
    c: &mut Core,
    ids: &DevBuf,
    eps: f32,
    vocab: usize,
    ctx: &StepCtx<'_>,
    head: Head,
    count: usize,
    part: usize,
    parts: usize,
) -> Result<usize, ModelError> {
    let rows = ctx.rows;
    let (first, n_layers) = (c.w.first, c.w.layers.len());
    if part == 0 {
        embed_gather_f32(&c.w.embed, ids, rows, &mut c.bufs.x, &c.stream)?;
        // the rows of an image replace the placeholder token's embedding (bf16 features widened exactly): before the tap, so the harness
        // sees what the layers see. Only a prefill slice has any (`prefill_chunk_mm` is the one setter); a window never does.
        if !c.splice.is_empty() {
            if ctx.kind != Kind::Prefill {
                return Err(ModelError::Request(
                    "image rows are only supplied by a prefill slice, never by a verify or decode window".into(),
                ));
            }
            for op in &c.splice {
                splice_rows(&op.feats.buf, op.src_row, &mut c.bufs.x, op.dst_row, op.n, &c.stream)?;
            }
        }
        fire(&mut c.tap, Point::Embed, &c.bufs.x, rows, HIDDEN, &c.stream)?;
    }
    for li in first + part * n_layers / parts..first + (part + 1) * n_layers / parts {
        {
            let norm = &c.w.layers[li - first].norm;
            // a P2 prefill slice's Mamba layer on one rank: the norm writes in_proj's rotated input itself (lane S5: the bits of the
            // norm then the linear's input rotation; xn is read by nothing else in that layer)
            let mamba_in = if super::arm_mamba::prerot_ok(c, li, ctx) {
                match c.w.layer(li).map(|l| &l.mixer) {
                    Some(Mixer::Mamba(m)) => Some(&m.in_proj),
                    _ => None,
                }
            } else {
                None
            };
            // a P2 prefill slice's MoE layer on one rank: the norm also makes the block's fc1 / shared-up inputs (lane S5)
            let moe_pre = mamba_in.is_none() && super::arm_moe::mma_block_single(c, li, ctx);
            match mamba_in {
                Some(in_proj) => {
                    gb_cuda::nemotron_dense_mma::norm_rot(&c.bufs.x, norm, eps, in_proj, rows, &mut c.bufs.ws, &c.stream)?;
                    c.prerot = true;
                }
                None => {
                    let done = match (moe_pre, c.w.layer(li).map(|l| &l.mixer), c.moe_mma.as_mut()) {
                        (true, Some(Mixer::Moe(moe)), Some(m)) => {
                            moe.norm_rotate_prefill(m, &c.bufs.x, norm, eps, rows, &mut c.bufs.xn, &c.stream)?
                        }
                        _ => false,
                    };
                    if !done {
                        rmsnorm_bf16w_f32(&c.bufs.x, norm, rows, HIDDEN, HIDDEN, eps, &mut c.bufs.xn, HIDDEN, &c.stream)?;
                    }
                }
            }
        }
        // an arm that fused the residual add (P2 prefill, no tap: `Core::residual_fused`, or the MoE block at two ranks:
        // `Core::residual_fused_moe_tp`) has already written x + its output into x
        if !c.run_arm(li, ctx)? {
            fire(&mut c.tap, Point::Mixer(li), &c.bufs.mix, rows, HIDDEN, &c.stream)?;
            add_f32(&mut c.bufs.x, &c.bufs.mix, rows * HIDDEN, &c.stream)?;
        }
        fire(&mut c.tap, Point::Layer(li), &c.bufs.x, rows, HIDDEN, &c.stream)?;
    }
    if part + 1 < parts {
        return Ok(0);
    }

    rmsnorm_bf16w_f32(&c.bufs.x, &c.w.norm_f, rows, HIDDEN, HIDDEN, eps, &mut c.bufs.xn, HIDDEN, &c.stream)?;
    fire(&mut c.tap, Point::FinalNorm, &c.bufs.xn, rows, HIDDEN, &c.stream)?;
    // the MTP head's taps: a window's post-`norm_f` rows are retained until its commit moves the kept ones into the slots' stores
    // (a device-to-device copy; a prefill slice's rows are read in place by the head's hook once the slice has run)
    if ctx.kind == Kind::Window {
        if let Some(m) = c.mtp.as_mut() {
            m.retain_window_taps(&c.bufs.xn, &c.bufs.x, rows, &c.stream)?;
        }
    }
    let m = match head {
        Head::None => return Ok(0),
        Head::Row(r) => {
            // the lm_head of one row reads one row of the normed stream
            let xr = c.bufs.xn.subview(r * HIDDEN * 4, HIDDEN * 4);
            // a single rank: the GEMV it always ran; a TP rank: its rows' GEMV, the scatter and the allreduce (`tp_arm::lm_head_logits`)
            let b = &mut c.bufs;
            let head = (&c.w.lm_head).into();
            tp_arm::lm_head_logits(&*c.tp, head, &xr, &mut b.logits, b.tp_logits.as_mut(), 1, vocab, true, &mut b.ws, &c.stream)?;
            1
        }
        Head::Rows(n) => {
            // the wanted rows' normed hidden states into the mixer buffer (idle after the last layer), then one GEMV and one top-k:
            // a row's logits do not depend on the rows beside it, so each is what `Head::Row` gives it
            assert_eq!(c.gather.len(), n, "Head::Rows without its gather list");
            for (i, &r) in c.gather.iter().enumerate() {
                c.bufs.mix.copy_range_from(i * HIDDEN * 4, &c.bufs.xn, r * HIDDEN * 4, HIDDEN * 4, &c.stream)?;
            }
            // (a single rank: the GEMV; the engine never batches a TP rank's prefill, `prefill_rows_supported`)
            let b = &mut c.bufs;
            let head = (&c.w.lm_head).into();
            tp_arm::lm_head_logits(
                &*c.tp,
                head,
                &b.mix,
                &mut b.logits,
                b.tp_logits.as_mut(),
                n,
                vocab,
                true,
                &mut b.ws,
                &c.stream,
            )?;
            n
        }
        Head::All => {
            let b = &mut c.bufs;
            tp_arm::lm_head_logits(
                &*c.tp,
                (&c.w.lm_head).into(),
                &b.xn,
                &mut b.logits,
                b.tp_logits.as_mut(),
                rows,
                vocab,
                true,
                &mut b.ws,
                &c.stream,
            )?;
            rows
        }
    };
    fire(&mut c.tap, Point::Logits, &c.bufs.logits, m, vocab, &c.stream)?;
    let bf = &mut c.bufs;
    topk_f32(&bf.logits, m, vocab, count, &mut bf.topk_ws, &mut bf.topk_ids, &mut bf.topk_vals, &c.stream)?;
    Ok(m)
}

/// A window's launches through the graph cache: replay the graph of its shape, capture it on the second sighting, or run eagerly.
#[allow(clippy::too_many_arguments)]
fn run_cached(
    g: &mut GraphCache,
    c: &mut Core,
    ids: &DevBuf,
    eps: f32,
    vocab: usize,
    ctx: &StepCtx<'_>,
    head: Head,
    count: usize,
    key: GraphKey,
) -> Result<usize, ModelError> {
    g.tick += 1;
    let tick = g.tick;
    match g.slots.get_mut(&key) {
        Some(GraphSlot::Ready(graphs, used)) => {
            *used = tick;
            for graph in graphs.iter() {
                graph.launch(&c.stream)?;
            }
            g.stats.replays += 1;
            return Ok(ctx.rows);
        }
        Some(GraphSlot::Eager) => {
            g.stats.eager_windows += 1;
            return launch_steps(c, ids, eps, vocab, ctx, head, count);
        }
        Some(GraphSlot::Seen) => {}
        None => {
            g.slots.insert(key, GraphSlot::Seen);
            g.stats.eager_windows += 1;
            return launch_steps(c, ids, eps, vocab, ctx, head, count);
        }
    }
    // second sighting: capture. Keep the number of instantiated graphs under the limit first (a graph still executing must not be
    // destroyed: the stream is drained before the least recently used one goes).
    let ready = g.slots.values().filter(|s| matches!(s, GraphSlot::Ready(..))).count();
    if ready >= GRAPH_LIMIT {
        c.stream.sync()?;
        let oldest = g
            .slots
            .iter()
            .filter_map(|(k, s)| match s {
                GraphSlot::Ready(_, used) => Some((*used, *k)),
                _ => None,
            })
            .min_by_key(|x| x.0);
        if let Some((_, k)) = oldest {
            g.slots.insert(k, GraphSlot::Seen);
            g.stats.evictions += 1;
        }
    }
    let t = std::time::Instant::now();
    // the model's stream is non-blocking, so the capture runs on it directly; one capture per part, in launch order (a part that fails
    // spoils the whole: the parts already captured are dropped and the window runs eagerly)
    let parts = g.parts;
    let mut graphs: Vec<Graph> = Vec::with_capacity(parts);
    let mut captured: Result<(), CaptureError<ModelError>> = Ok(());
    for p in 0..parts {
        let one: Result<Graph, CaptureError<ModelError>> = match Graph::begin(&c.stream) {
            Ok(cap) => {
                let ran = launch_steps_part(c, ids, eps, vocab, ctx, head, count, p, parts).map(|_| ());
                cap.finish(ran)
            }
            Err(e) => Err(CaptureError::Refused(e)),
        };
        match one {
            Ok(graph) => graphs.push(graph),
            Err(e) => {
                captured = Err(e);
                break;
            }
        }
    }
    match captured {
        Ok(()) => {
            g.stats.captures += 1;
            g.stats.capture_ms += t.elapsed().as_secs_f64() * 1e3;
            let mut r = Ok(());
            for graph in &graphs {
                r = graph.launch(&c.stream);
                if r.is_err() {
                    break;
                }
            }
            g.slots.insert(key, GraphSlot::Ready(graphs, tick));
            r?;
            Ok(ctx.rows)
        }
        Err(e) => {
            if let Some(fatal) = e.fatal() {
                return Err(ModelError::Cuda(fatal.clone()));
            }
            let why = match e {
                CaptureError::Refused(code) => format!("the runtime refused the capture: {code}"),
                CaptureError::Closure(m) => {
                    if closure_is_fatal(&m) {
                        return Err(m);
                    }
                    format!("a launch failed while capturing: {m}")
                }
            };
            eprintln!(
                "rexl3 level=warn event=graph_capture_failed rows={} n_seq={} count={} chunks={} {why} (this window shape runs eagerly)",
                key.rows, key.n_seq, key.count, key.chunks
            );
            g.stats.capture_failures += 1;
            g.stats.last_failure = Some(why);
            g.slots.insert(key, GraphSlot::Eager);
            g.stats.eager_windows += 1;
            launch_steps(c, ids, eps, vocab, ctx, head, count)
        }
    }
}

/// What the lm_head runs on after the last layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Head {
    /// Nothing: a prefill slice nobody wants logits of.
    None,
    /// One row of the batch (a prefill slice's `want`).
    Row(usize),
    /// The rows `Core::gather` names, `n` of them (a prefill slice of several prompts, each ending in it): gathered into one
    /// contiguous block, one lm_head and one top-k.
    Rows(usize),
    /// Every row (a window).
    All,
}

/// A failure that outlives its call: the model refuses everything after it.
#[derive(Clone, Debug)]
enum Fatal {
    /// The stream reported an error a retry cannot fix.
    Cuda(CudaError),
    /// A commit's replay failed part way, so the recurrent state of some layers moved and others did not.
    StateLost(String),
}

impl Fatal {
    fn error(&self) -> ModelError {
        match self {
            Fatal::Cuda(e) => ModelError::Cuda(e.clone()),
            Fatal::StateLost(m) => ModelError::StateLost(m.clone()),
        }
    }
}

/// Device bytes of what the model allocated besides weights, by category.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolBytes {
    /// The Mamba S and conv pools of every slot.
    pub state: u64,
    /// The paged KV pool (K and V of every attention layer).
    pub kv: u64,
    /// The per-layer window records and retained xBC rows, and the prefill record.
    pub records: u64,
    /// The step buffers (residual, projections, logits, workspaces, top-k).
    pub bufs: u64,
    /// The MoE scratch and the attention partials.
    pub scratch: u64,
    /// The prefix cache's snapshot arena (0 until `enable_snapshots`).
    pub snapshots: u64,
}

impl PoolBytes {
    pub fn total(&self) -> u64 {
        self.state + self.kv + self.records + self.bufs + self.scratch + self.snapshots
    }
}

/// The device model: weights, pools, buffers and the slot table. Built by [`NemotronModel::load`]; the engine reaches it through
/// [`NemotronRuntime`].
///
/// What a call may rely on:
///
/// - A step is checked on the host before any launch ([`plan_step`]); every GPU error is a [`ModelError::Cuda`], and one that a
///   retry cannot fix latches, so every later call and [`NemotronModel::flush`] return it.
/// - `commit` launches the replay at once, on the model's stream, so the next step sees the committed state without any host wait.
///   `commit` has no return value: a refusal (no window, keep out of range) is returned by the next `flush`, and a replay that fails part
///   way is a latched [`ModelError::StateLost`].
/// - A window's records belong to the last window only: starting another window, or any step that moves a slot's state, drops what a
///   later `commit` could have replayed for it.
pub struct NemotronModel {
    pub core: Core,
    /// The fold period of a lagged decode (0: off, [`LAG_ENV`]) and, per slot, the rows its ring holds (host bookkeeping: the kernels
    /// read it from `lag_dev`, written per step). The S pool of a slot with rows pending is the state as of its last fold.
    lag_k: usize,
    lag_host: Vec<usize>,
    lag_dev: DevBuf,
    /// The slots (and their pending rows) a fold launch handles, i32 `[slots]` each.
    fold_slots: DevBuf,
    fold_lags: DevBuf,
    lag_table: Option<LagTable>,
    pub(super) vocab: usize,
    pub(super) eps: f32,
    /// The most rows of one verify or decode window (all sequences): [`WINDOW_ROWS`] unless the model was built wide
    /// ([`NemotronModel::new_wide`]). The per-layer window records, the retained xBC rows and the head's window taps hold this many.
    pub(super) window_rows: usize,
    pub(super) slots: SlotTable,
    pub(super) kv_tab: KvTables,
    /// The host block lists changed (or an upload failed) since the device table was written.
    pub(super) table_dirty: bool,
    pub(super) window: WindowLog,
    /// The step's token ids, and its sequence arrays, on the device. They live here and not in [`Core`] so a [`StepCtx`] can borrow
    /// them while an arm takes the core mutably. The previous step has always synchronised before they are written.
    ids: DevBuf,
    seq_start: DevBuf,
    seq_slot: DevBuf,
    /// i32 `[slots]`: per sequence of a fused window, whether its one row is applied inside the layers (`StepCtx::apply`).
    apply: DevBuf,
    /// The step's `(slot, position)` per row, rewritten in place by each step (room for `rows_cap` rows).
    row_index: RowIndex,
    /// Per slot, the device arguments of a `commit` of that slot (see [`CommitArgs`]).
    commit_args: Vec<CommitArgs>,
    /// The layer table of the one-launch replay (`None` without a Mamba layer).
    commit_table: Option<nemotron_ssm::CommitTable>,
    /// The captured graphs of the decode windows.
    graphs: GraphCache,
    fatal: Option<Fatal>,
    /// A `commit` the host refused: returned once by the next `flush`.
    refused: Option<String>,
    /// Bytes of the attention partials (`Scratch` keeps its buffers private).
    attn_bytes: u64,
    /// The prefix cache's snapshot arena (`snapshot.rs`), once `enable_snapshots` allocated it.
    pub(super) snaps: Option<super::snapshot::SnapArena>,
}

/// This rank's device weights of a tensor-parallel group (`topo.ranks > 1`), from `tp_shard` (`nemotron/tp-weights`): a sharded linear
/// is an `Exl3Weights` whose `n` is the rank's column count (`in_proj`; the matrices whose kernel plan would move when sliced, and the
/// attention matrices unless `REXL3_TP_ATTN=slice`, stay whole: `tp_arm::serving_shard_plan`), a MoE layer holds only the rank's
/// experts, `lm_head` the rank's rows; everything else is whole.
fn rank_weights(
    dir: &Path,
    inv: &Inventory,
    cfg: &OmniConfig,
    plan: &LayerPlan,
    layers: usize,
    head: bool,
    topo: super::tp::TpTopology,
) -> Result<NemotronWeights, ModelError> {
    // a configured expert map (`REXL3_TP_EXPERT_MAP`) that does not fit this model or group is refused by name before any read
    let shard = tp_arm::serving_shard_plan_checked(topo, cfg).map_err(ModelError::Config)?;
    if let Some(m) = shard.expert_map.as_deref() {
        eprintln!(
            "rexl3 event=tp_expert_map rank={} ranks={} map={:016x} head_row={} (ownership is a lookup: the bits do not depend on it)",
            topo.rank,
            topo.ranks,
            m.hash(),
            m.has_head()
        );
    }
    super::tp_shard::load_weights(dir, cfg, inv, plan, layers, head, &shard, &Options::from_env())
}

/// The arguments of one slot's commit, in a region of a device buffer of the model's own: `[start, end, slot, keep]` as i32. The
/// replay kernels are asynchronous and read these arrays after `commit` has returned, so they cannot be temporaries of the call. A slot
/// commits at most once per window and every window ends with the stream synchronised, so the previous commit of a slot has finished
/// reading its region by the time the next one rewrites it; commits of different slots use different regions.
struct CommitArgs {
    /// The whole region (16 bytes), the target of the one upload.
    all: DevBuf,
    /// `[start, end]`: the sequence's `seq_start` array.
    seq_start: DevBuf,
    slot: DevBuf,
    keep: DevBuf,
}

impl CommitArgs {
    /// One region for each of `slots` slots, carved from one allocation.
    fn pool(slots: usize) -> Result<Vec<Self>, CudaError> {
        let block = DevBuf::zeroed_view(slots.max(1) * 16)?;
        Ok((0..slots.max(1))
            .map(|s| {
                let all = block.subview(s * 16, 16);
                let (seq_start, slot, keep) = (all.subview(0, 8), all.subview(8, 4), all.subview(12, 4));
                Self { all, seq_start, slot, keep }
            })
            .collect())
    }
}

impl NemotronModel {
    /// Loads the pack at `dir` (trunk layers `0..opts.layers` or all of them) and allocates the pools and buffers for `opts`. Uses
    /// `weights::upload`, then `bridge::into_device`, on the device of the calling thread.
    pub fn load(dir: &Path, cfg: &OmniConfig, plan: &LayerPlan, opts: ModelOptions) -> Result<Self, ModelError> {
        Self::load_tp(dir, cfg, plan, opts, Arc::new(SingleRank))
    }

    /// Whether `load_tp` can build a rank of a group in this tree: `rank_weights` is wired to the per-rank weights of
    /// `nemotron/tp-weights`. (A single rank always loads.)
    pub const TP_WEIGHTS_BUILT: bool = true;

    /// [`NemotronModel::load`] as one rank of the tensor-parallel group `comm` (`docs/NEMOTRON_TP.md`): this rank's slices of the
    /// weights (a sharded linear's columns, its experts, its `lm_head` rows) and the arms of [`NemotronModel::new_tp`]. With the single
    /// rank's group it IS `load`. Both ranks load the same pack with the same `cfg`, `plan` and `opts`. A group of more than one rank
    /// takes the 64-row window and the plain decode only ([`NemotronModel::load_tp_lag`] refuses the rest by name).
    pub fn load_tp(
        dir: &Path,
        cfg: &OmniConfig,
        plan: &LayerPlan,
        opts: ModelOptions,
        comm: Arc<dyn TpComm>,
    ) -> Result<Self, ModelError> {
        Self::load_tp_lag(dir, cfg, plan, opts, comm, WINDOW_ROWS, lag_period())
    }

    /// [`NemotronModel::load`] for a model whose decode window takes up to `window_rows` rows (at least [`WINDOW_ROWS`], at most
    /// [`MAX_WINDOW_ROWS`]): the buffers, the large-row MoE scratch, the per-layer window records and the head's window taps are sized
    /// for it. At [`WINDOW_ROWS`] this is `load`.
    pub fn load_wide(
        dir: &Path,
        cfg: &OmniConfig,
        plan: &LayerPlan,
        opts: ModelOptions,
        window_rows: usize,
    ) -> Result<Self, ModelError> {
        Self::load_lag(dir, cfg, plan, opts, window_rows, lag_period())
    }

    /// [`NemotronModel::load_wide`] with the fold period of the lagged decode given (`0`: off; [`LAG_ENV`] is what `load_wide` reads).
    pub fn load_lag(
        dir: &Path,
        cfg: &OmniConfig,
        plan: &LayerPlan,
        opts: ModelOptions,
        window_rows: usize,
        lag_k: usize,
    ) -> Result<Self, ModelError> {
        Self::load_tp_lag(dir, cfg, plan, opts, Arc::new(SingleRank), window_rows, lag_k)
    }

    /// The general loader: one rank of the group `comm` with a decode window of `window_rows` rows and the lagged decode of fold period
    /// `lag_k`. A group of more than one rank with a window of more than [`WINDOW_ROWS`] rows or a lagged decode is refused by name
    /// before anything is read (tensor parallelism is verified for the 64-row window and the plain decode only).
    pub fn load_tp_lag(
        dir: &Path,
        cfg: &OmniConfig,
        plan: &LayerPlan,
        opts: ModelOptions,
        comm: Arc<dyn TpComm>,
        window_rows: usize,
        lag_k: usize,
    ) -> Result<Self, ModelError> {
        let n = check_options(&opts, cfg)?;
        // the P2 kernels follow the contract of this process, never a stray opt-in variable (`contract::apply`)
        super::contract::apply();
        check_window_rows(window_rows)?;
        let topo = comm.topology();
        check_tp_plain(topo.is_single(), window_rows, lag_k)?;
        let inv = Inventory::read(dir).map_err(|e| ModelError::Config(format!("nemotron pack {}: {e}", dir.display())))?;
        // The MTP head (5.9 GB, all bf16) loads with the whole trunk when `opts.mtp` asks for it and the pack has one, unless
        // `REXL3_MTP=off`: a layer-limited model is the oracle's and the tests', which never draft.
        let head = opts.mtp && super::mtp::head_wanted(opts.layers) && !cfg.mtp_pattern.is_empty();
        let pick = |u: UnitId| match u {
            UnitId::Trunk(i) => i < n,
            UnitId::Embed | UnitId::NormF | UnitId::LmHead => true,
            UnitId::Mtp(_) => head,
        };
        let w = if topo.is_single() {
            let mut up = weights::upload(dir, &inv, plan, pick, &Options::from_env())
                .map_err(|e| ModelError::Config(format!("nemotron upload: {e}")))?;
            let st = Stream::new_nonblocking()?;
            let mut w = bridge::into_device(&mut up, cfg, plan, 0..n, &st)?;
            if head {
                w.mtp = bridge::take_mtp(&mut up, cfg, plan, &st)?;
            }
            st.sync()?;
            drop(up);
            w
        } else {
            rank_weights(dir, &inv, cfg, plan, n, head, topo)?
        };
        let mut m = Self::new_tp_lag(w, cfg, opts, comm, window_rows, lag_k)?;
        // The restricted draft head is the default: the head runs over the 49,152-id list of `mtp_vocab_48k.txt` (a quarter of the
        // `lm_head`'s read per draft step: docs/NEMOTRON_MTP_PLAN.md). `REXL3_MTP_VOCAB=off` keeps the full head, `=<file>` replaces the
        // list. The plan prices these same ids (`engine::restricted_head_rows`).
        if m.core.mtp.is_some() {
            if let Some(ids) = super::mtp::restricted_vocab_ids()? {
                m.mtp_restrict_vocab(&ids)?;
            }
        }
        Ok(m)
    }

    /// The model on weights already on the device (what `load` ends with, and the path the device tests take). `w` must be trunk layers
    /// `0..n` of the pack `cfg` describes, with `n = opts.layers` or all of them.
    pub fn new(w: NemotronWeights, cfg: &OmniConfig, opts: ModelOptions) -> Result<Self, ModelError> {
        Self::new_tp(w, cfg, opts, Arc::new(SingleRank))
    }

    /// [`NemotronModel::new`] as one rank of the tensor-parallel group `tp` (`docs/NEMOTRON_TP.md`): `w` holds this rank's slices (a
    /// sharded linear's `n` is its local column count, a MoE layer's experts are the rank's, the `lm_head` its rows). With the single
    /// rank's group this is `new`: nothing is allocated or launched for tensor parallelism.
    pub fn new_tp(w: NemotronWeights, cfg: &OmniConfig, opts: ModelOptions, tp: Arc<dyn TpComm>) -> Result<Self, ModelError> {
        Self::new_tp_lag(w, cfg, opts, tp, WINDOW_ROWS, lag_period())
    }

    /// [`NemotronModel::new`] with a decode window of up to `window_rows` rows ([`WINDOW_ROWS`]..=[`MAX_WINDOW_ROWS`]). Above
    /// [`WINDOW_ROWS`] the step buffers, the large-row MoE scratch, every Mamba layer's window record and retained xBC rows and the head's
    /// window taps are sized for it, and a step of more than 64 rows runs the large-row MoE path (bit-equal to the decode path).
    pub fn new_wide(w: NemotronWeights, cfg: &OmniConfig, opts: ModelOptions, window_rows: usize) -> Result<Self, ModelError> {
        Self::new_lag(w, cfg, opts, window_rows, lag_period())
    }

    /// [`NemotronModel::new_wide`] with the fold period of the lagged decode given (`0` or `1`: off; [`LAG_ENV`] is what `new_wide` reads).
    pub fn new_lag(
        w: NemotronWeights,
        cfg: &OmniConfig,
        opts: ModelOptions,
        window_rows: usize,
        lag_k: usize,
    ) -> Result<Self, ModelError> {
        Self::new_tp_lag(w, cfg, opts, Arc::new(SingleRank), window_rows, lag_k)
    }

    /// The general constructor: one rank of the group `tp` with a decode window of up to `window_rows` rows and the lagged decode of fold
    /// period `lag_k` (`0` or `1`: off). A group of more than one rank takes the 64-row window and the plain decode only: anything else is
    /// refused by name (`docs/NEMOTRON_AGG.md`).
    pub fn new_tp_lag(
        w: NemotronWeights,
        cfg: &OmniConfig,
        opts: ModelOptions,
        tp: Arc<dyn TpComm>,
        window_rows: usize,
        lag_k: usize,
    ) -> Result<Self, ModelError> {
        let lag_k = if (2..=16).contains(&lag_k) { lag_k } else { 0 };
        let n = check_options(&opts, cfg)?;
        // the P2 kernels follow the contract of this process, never a stray opt-in variable (`contract::apply`)
        super::contract::apply();
        check_window_rows(window_rows)?;
        let topo = tp.topology();
        if topo.ranks == 0 || topo.rank >= topo.ranks {
            return Err(ModelError::Config(format!(
                "a tensor-parallel rank must be below the group size, got rank {} of {}",
                topo.rank, topo.ranks
            )));
        }
        check_tp_plain(topo.is_single(), window_rows, lag_k)?;
        tp_arm::check_weights(&w, cfg.vocab, topo).map_err(ModelError::Config)?;
        if w.first != 0 || w.layers.len() != n {
            return Err(ModelError::Config(format!(
                "the weights are layers {}..{} of the trunk, the options ask for 0..{n}",
                w.first,
                w.first + w.layers.len()
            )));
        }
        let map = LayerMap::from_pattern(&cfg.pattern, 0, n).map_err(ModelError::Config)?;
        if w.layer_map() != map {
            return Err(ModelError::Config("the device layers do not follow the pack's layer pattern".into()));
        }
        // the head drafts for the whole trunk: its taps are the last layer's, so a layer-limited model cannot carry it
        let head = w.mtp.is_some();
        if head && n != cfg.pattern.chars().count() && !super::mtp::head_on_prefix() {
            return Err(ModelError::Config(format!(
                "the MTP head follows the whole trunk, but the weights are its first {n} layers"
            )));
        }
        let (slots, vocab) = (opts.slots, cfg.vocab);
        let rows_cap = step_rows_cap(opts.prefill_rows, window_rows);
        let (n_mamba, n_attn) = (map.count(LayerKind::Mamba), map.count(LayerKind::Attn));
        let max_blocks = opts.max_context.div_ceil(BLOCK_TOKENS);

        let r = rows_cap;
        let bufs = Bufs {
            rows_cap: r,
            x: DevBuf::zeroed_view(r * HIDDEN * 4)?,
            xn: DevBuf::zeroed_view(r * HIDDEN * 4)?,
            // views of their own block, so that `linear_rows` and the MoE arm can cut a wide step into calls of fewer rows (`subview`)
            mix: DevBuf::zeroed_view(r * HIDDEN * 4)?,
            proj: DevBuf::zeroed_view(r * INPROJ_COLS * 4)?,
            z: DevBuf::zeroed(r * Z_COLS * 4)?,
            xbc: DevBuf::zeroed(r * XBC_COLS * 4)?,
            dt: DevBuf::zeroed(r * DT_COLS * 4)?,
            ssm_y: DevBuf::zeroed(r * INNER * 2)?,
            gated: DevBuf::zeroed_view(r * INNER * 2)?,
            q: DevBuf::zeroed_view(r * Q_ROW_ELEMS * 4)?,
            k: DevBuf::zeroed_view(r * KV_TOKEN_ELEMS * 4)?,
            v: DevBuf::zeroed_view(r * KV_TOKEN_ELEMS * 4)?,
            attn_o: DevBuf::zeroed_view(r * Q_ROW_ELEMS * 4)?,
            // an EXL3 lm_head (the MixedK pack's) writes a window of more than 128 rows in calls of 128 (`linear_rows`, `subview`): a view
            logits: if w.lm_head.exl3().is_some() { DevBuf::zeroed_view(r * vocab * 4)? } else { DevBuf::zeroed(r * vocab * 4)? },
            topk_ws: DevBuf::zeroed(topk_workspace_bytes(r, vocab, MAX_CANDIDATES))?,
            topk_ids: DevBuf::zeroed(r * MAX_CANDIDATES * 4)?,
            topk_vals: DevBuf::zeroed(r * MAX_CANDIDATES * 4)?,
            // the EXL3 linears take at most LINEAR_ROWS rows a call (`linear_rows` cuts a wider step): the workspace is that wide
            // (under P2 the prefill linear takes the whole slice: its fp16 rotated input, `rows x K x 2` at the widest K of 8,192)
            ws: DevBuf::zeroed(super::contract::workspace_bytes(w.max_workspace_bytes(r.min(LINEAR_ROWS)), r))?,
            // tensor parallelism only: the local output of a sharded linear, the merged q, k, v and the rank's logits
            // a view: `linear_rows` cuts a wide step (a prefill slice) with `subview`
            tp_tmp: if topo.is_single() { None } else { Some(DevBuf::zeroed_view(r * tp_arm::max_local_cols(&w) * 4)?) },
            tp_qkv: if topo.is_single() || n_attn == 0 { None } else { Some(DevBuf::zeroed_view(r * tp_arm::QKV_COLS * 4)?) },
            tp_logits: match tp_arm::lm_head_rows_local(&w, vocab, topo) {
                Some(rows_local) => Some(DevBuf::zeroed_view(r.min(WINDOW_ROWS) * rows_local * 4)?),
                None => None,
            },
            tp_pack: if topo.ranks == 2 && super::contract::current() == super::contract::PrefillContract::P2 {
                let b = r * tp_arm::max_local_cols(&w) * 4;
                // views (the overlapped exchange cuts them in row halves with `subview`)
                Some((DevBuf::zeroed_view(b)?, DevBuf::zeroed_view(b)?))
            } else {
                None
            },
            tp_aux: if topo.ranks == 2 && super::contract::current() == super::contract::PrefillContract::P2 {
                Some((
                    gb_cuda::Stream::new_nonblocking()?,
                    [gb_cuda::rt::Event::new()?, gb_cuda::rt::Event::new()?, gb_cuda::rt::Event::new()?],
                ))
            } else {
                None
            },
            tp_state: if topo.ranks == 2 && super::contract::current() == super::contract::PrefillContract::P2 {
                let b = slots.max(1) * super::arm_mamba::HG_STATE_BYTES_PER_SLOT;
                Some((DevBuf::zeroed(b)?, DevBuf::zeroed(b)?))
            } else {
                None
            },
            tp_ypack: if topo.ranks == 2 && super::contract::current() == super::contract::PrefillContract::P2 && r > WINDOW_ROWS
            {
                let b = r * gb_cuda::nemotron_moe::TOPK * gb_cuda::nemotron_moe::LATENT * 4;
                // send rows, receive rows, and the pair -> received-row map of the combine (i32 a pair)
                Some((DevBuf::zeroed(b)?, DevBuf::zeroed(b)?, DevBuf::zeroed(r * gb_cuda::nemotron_moe::TOPK * 4)?))
            } else {
                None
            },
        };
        // the collective paths decide from the plan alone (`tp_arm`): hold at boot that the buffers they need are there and fit it, so a
        // shape problem is this rank's boot error and never a mid-step asymmetry with the peers waiting in a collective
        tp_arm::check_buffers(
            &w,
            topo,
            vocab,
            r,
            bufs.tp_tmp.as_ref().map(|b| b.bytes()),
            bufs.tp_qkv.as_ref().map(|b| b.bytes()),
            bufs.tp_logits.as_ref().map(|b| b.bytes()),
        )
        .map_err(|e| ModelError::Config(format!("tensor parallelism: {e}")))?;

        // the MTP head's attention layer is one more stripe, after the trunk's (1 KiB per token), on the same block table
        let kv_layers = n_attn + usize::from(head);
        let kv_dtype = gb_cuda::nemotron_attn::configured_kv_dtype().map_err(ModelError::Config)?;
        let kv_geom = KvGeom { pages: opts.kv_pages, layers: kv_layers, dtype: kv_dtype };
        // zeroed views: one buffer each for K and V, cut into a stripe per attention layer, and no stale NaN behind a masked key
        let kv = NemoKv {
            k: DevBuf::zeroed_view(kv_geom.half_bytes())?,
            v: DevBuf::zeroed_view(kv_geom.half_bytes())?,
            n_blocks: kv_geom.pages * kv_layers,
            dtype: kv_geom.dtype,
        };
        let mamba = (0..n_mamba)
            .map(|_| Ok(MambaState { s: DevBuf::zeroed(slots * S_SLOT_BYTES)?, conv: DevBuf::zeroed(slots * CONV_SLOT_BYTES)? }))
            .collect::<Result<Vec<_>, ModelError>>()?;
        let table = BlockTable::from_host(&vec![Vec::new(); slots], max_blocks)?;
        // partials for every row, chunk and query head: (m, l) and the unnormalised o[128], all f32
        // The scratch of both arithmetics (`Scratch::new_serving`): the tree's partials for the 16 chunks a row below the flash path's
        // threshold can have, whatever the context, and the flash path's, one group of 32 heads of `(m, l, o)` per 8,192 keys of the
        // context. A call of more rows than a window (a prefill slice of up to 256) is run by `attn_long` in passes of the scratch's
        // rows, a row's bits not depending on the pass, so the scratch is a window wide: 153 MB at a million tokens (the tree sized for
        // the context and the slice would be 8.7 GB at 256 rows).
        let (attn_rows, attn_chunks) =
            if n_attn > 0 { (r.min(WINDOW_ROWS), opts.max_context.div_ceil(CHUNK_KEYS)) } else { (1, 1) };
        let attn = AttnScratch::new_serving(attn_rows, attn_chunks)?;
        // what the buffers hold, not the formula: the plan's term is pinned to `Scratch::serving_bytes` by a test and to this by the
        // long-context gate (`decode_cost_at_depth` compares them at a million tokens)
        let attn_bytes = attn.allocated_bytes() as u64;
        let pre_rec = Record::alloc(r)?;
        let win_rec = (0..n_mamba).map(|_| Record::alloc(window_rows)).collect::<Result<Vec<_>, _>>()?;
        let win_xbc = (0..n_mamba).map(|_| DevBuf::zeroed(window_rows * XBC_ROW_BYTES)).collect::<Result<Vec<_>, _>>()?;
        let mtp = match &w.mtp {
            Some(h) => Some(super::mtp::MtpState::with_window(slots, r, window_rows, super::mtp::weight_bytes(h))?),
            None => None,
        };
        // the rings of a lagged decode (`LAG_ENV`): one per Mamba layer, by slot
        let lag_rings: Vec<LagRing> = if lag_k >= 2 {
            (0..n_mamba).map(|_| LagRing::alloc(slots, lag_k)).collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        let core = Core {
            w,
            opts: opts.clone(),
            bufs,
            kv,
            mamba,
            moe: MoeScratch::new(WINDOW_ROWS)?,
            // the large-row path serves the prefill slices and the wide windows: one scratch for the widest of them
            moe_prefill: if rows_cap > WINDOW_ROWS { Some(PrefillScratch::new(rows_cap)?) } else { None },
            tap: None,
            prerot: false,
            stream: Stream::new_nonblocking()?,
            map,
            kv_geom,
            table,
            attn,
            pre_rec,
            win_rec,
            win_xbc,
            keep: DevBuf::zeroed(slots.max(1) * 4)?,
            mtp,
            splice: Vec::new(),
            gather: Vec::new(),
            lag_rings,
            tp,
            moe_half: if rows_cap >= 1024 && topo.ranks == 2 && super::contract::current() == super::contract::PrefillContract::P2
            {
                Some(MoeHalf::new(rows_cap / 2)?)
            } else {
                None
            },
            moe_mma: if rows_cap > WINDOW_ROWS && super::contract::current() == super::contract::PrefillContract::P2 {
                Some(gb_cuda::nemotron_moe_prefill::mma::MmaScratch::new(rows_cap)?)
            } else {
                None
            },
        };
        // the addresses of every Mamba layer's record, retained rows and pools, for the one-launch replay (the buffers never move)
        let commit_table = if n_mamba == 0 {
            None
        } else {
            let layers: Vec<nemotron_ssm::CommitLayer<'_>> = (0..n_mamba)
                .map(|o| nemotron_ssm::CommitLayer {
                    rec: &core.win_rec[o],
                    xin: &core.win_xbc[o],
                    s: &core.mamba[o].s,
                    conv: &core.mamba[o].conv,
                })
                .collect();
            Some(nemotron_ssm::CommitTable::new(&layers)?)
        };
        let lag_table = if lag_k >= 2 && n_mamba > 0 {
            let layers: Vec<(&DevBuf, &LagRing)> = (0..n_mamba).map(|o| (&core.mamba[o].s, &core.lag_rings[o])).collect();
            Some(LagTable::new(&layers)?)
        } else {
            None
        };
        let m = Self {
            core,
            lag_k: if lag_table.is_some() { lag_k } else { 0 },
            lag_host: vec![0; slots.max(1)],
            lag_dev: DevBuf::zeroed(slots.max(1) * 4)?,
            fold_slots: DevBuf::zeroed(slots.max(1) * 4)?,
            fold_lags: DevBuf::zeroed(slots.max(1) * 4)?,
            lag_table,
            vocab,
            eps: cfg.eps as f32,
            window_rows,
            slots: SlotTable::new(slots),
            kv_tab: KvTables::new(opts.kv_pages, slots, max_blocks),
            table_dirty: false,
            window: WindowLog::new(),
            ids: DevBuf::zeroed(r * 4)?,
            seq_start: DevBuf::zeroed((slots + 1) * 4)?,
            seq_slot: DevBuf::zeroed(slots.max(1) * 4)?,
            apply: DevBuf::zeroed(slots.max(1) * 4)?,
            row_index: RowIndex::with_capacity(r)?,
            commit_args: CommitArgs::pool(slots)?,
            commit_table,
            graphs: GraphCache::new(!topo.is_single()),
            fatal: None,
            refused: None,
            attn_bytes,
            snaps: None,
        };
        // the zeroing above is asynchronous work on the default stream: the model's stream must start after it
        gb_cuda::rt::device_sync()?;
        Ok(m)
    }

    /// Switches the graphs of the decode windows on or off for the windows that follow (they start on unless `REXL3_NEMO_GRAPHS=off`).
    /// Eager and graph windows run the same launches: the digest test runs both. A change to what the launches do (the MoE block's side
    /// stream or programmatic launch switches) needs [`NemotronModel::clear_graphs`] too.
    pub fn set_graphs(&mut self, on: bool) {
        self.graphs.on = on;
    }

    pub fn graphs_on(&self) -> bool {
        self.graphs.on
    }

    /// What the graph cache has done so far.
    pub fn graph_stats(&self) -> GraphStats {
        self.graphs.stats.clone()
    }

    /// Forgets every captured graph (the stream is drained first), so the next windows run eagerly once and capture again.
    pub fn clear_graphs(&mut self) -> Result<(), ModelError> {
        self.core.stream.sync()?;
        self.graphs.slots.clear();
        Ok(())
    }

    /// Installs (or removes) the oracle harness's observer.
    pub fn set_tap(&mut self, tap: Option<Box<dyn Tap>>) {
        self.core.tap = tap;
    }

    /// Tokens committed in `slot` (`None` when it has not begun, or is unusable).
    pub fn slot_len(&self, slot: usize) -> Option<usize> {
        self.slots.len(slot).ok()
    }

    /// Device bytes of everything but the weights, by category.
    pub fn pool_bytes(&self) -> PoolBytes {
        let c = &self.core;
        let sum = |bs: &[&DevBuf]| bs.iter().map(|b| b.bytes() as u64).sum::<u64>();
        let rec = |r: &Record| (r.xbc.bytes() + r.dt.bytes() + r.g.bytes() + r.decay.bytes()) as u64;
        let b = &c.bufs;
        PoolBytes {
            state: c.mamba.iter().map(|m| (m.s.bytes() + m.conv.bytes()) as u64).sum(),
            kv: (c.kv.k.bytes() + c.kv.v.bytes()) as u64,
            records: rec(&c.pre_rec)
                + c.win_rec.iter().map(rec).sum::<u64>()
                + c.win_xbc.iter().map(|x| x.bytes() as u64).sum::<u64>()
                + c.lag_rings.iter().map(|r| (r.xbc.bytes() + r.dt.bytes() + r.decay.bytes()) as u64).sum::<u64>(),
            bufs: sum(&[
                &b.x,
                &b.xn,
                &b.mix,
                &b.proj,
                &b.z,
                &b.xbc,
                &b.dt,
                &b.ssm_y,
                &b.gated,
                &b.q,
                &b.k,
                &b.v,
                &b.attn_o,
                &b.logits,
                &b.topk_ws,
                &b.topk_ids,
                &b.topk_vals,
                &b.ws,
                &c.keep,
                &self.ids,
                &self.seq_start,
                &self.seq_slot,
                &self.apply,
            ]) + [&b.tp_tmp, &b.tp_qkv, &b.tp_logits]
                .iter()
                .filter_map(|o| o.as_ref())
                .map(|d| d.bytes() as u64)
                .sum::<u64>()
                + b.tp_pack.as_ref().map_or(0, |(a, c)| (a.bytes() + c.bytes()) as u64)
                + b.tp_ypack.as_ref().map_or(0, |(a, c, m)| (a.bytes() + c.bytes() + m.bytes()) as u64)
                + b.tp_state.as_ref().map_or(0, |(a, c)| (a.bytes() + c.bytes()) as u64),
            scratch: MoeScratch::bytes(c.moe.max_rows())
                + c.moe_prefill.as_ref().map_or(0, |p| PrefillScratch::bytes(p.max_rows()))
                + c.moe_mma.as_ref().map_or(0, |m| gb_cuda::nemotron_moe_prefill::mma::MmaScratch::bytes(m.max_rows()))
                + c.moe_half.as_ref().map_or(0, |h| MoeHalf::bytes(h.prefill.max_rows()))
                + self.attn_bytes,
            snapshots: self.snapshot_arena_bytes(),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    /// Where this model sits in its tensor-parallel group (`TpTopology::SINGLE` for the single-node engine).
    pub fn tp_topology(&self) -> super::tp::TpTopology {
        self.core.tp.topology()
    }

    pub fn has_mtp(&self) -> bool {
        self.core.w.mtp.is_some()
    }

    /// The form of the loaded head's projections and of the trunk's `lm_head`, which keys the cost tables (`spec::HeadForm::of`): EXL3
    /// when any head projection is (a 4-bit head pack), bf16 otherwise (or with no head), and the MixedK form when the `lm_head` or the
    /// head's attention is EXL3.
    pub fn mtp_head_form(&self) -> super::spec::HeadForm {
        let head = self.core.w.mtp.as_ref();
        // the real forms (lane L's P0-3 / P0-4 made the trunk's `lm_head` and the head's q/k/v/o `HeadLinear`s, bf16 or EXL3; lane T's
        // tripwire stood here until then): a MixedK pack (an EXL3 `lm_head`, an EXL3 head attention) is priced with its own tables
        let lm_head_exl3 = self.core.w.lm_head_exl3();
        let attention_exl3 = head.is_some_and(|h| h.attn_exl3());
        super::spec::HeadForm::of(head.is_some_and(|h| h.has_exl3()), attention_exl3, lm_head_exl3)
    }

    pub(super) fn check_alive(&self) -> Result<(), ModelError> {
        match &self.fatal {
            Some(f) => Err(f.error()),
            None => Ok(()),
        }
    }

    /// A CUDA error that a retry cannot fix spoils the model: every later call returns it.
    pub(super) fn note(&mut self, e: &ModelError) {
        if let ModelError::Cuda(c) = e {
            if c.is_fatal() && self.fatal.is_none() {
                self.fatal = Some(Fatal::Cuda(c.clone()));
            }
        }
    }

    /// Starts a sequence in `slot`: zeroes its Mamba state in every layer and its KV length. A slot that has begun is refused until
    /// it is ended. An error after the slot is taken leaves it unusable until `end_slot`.
    pub fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.check_alive()?;
        self.slots.begin(slot)?;
        self.window.forget(slot);
        self.kv_tab.release(slot);
        self.table_dirty = true;
        self.mtp_reset_slot(slot);
        if let Some(l) = self.lag_host.get_mut(slot) {
            // a new sequence: its state is zeroed below, whatever its ring held
            *l = 0;
        }
        let r = (|| -> Result<(), CudaError> {
            let c = &mut self.core;
            for m in c.mamba.iter_mut() {
                m.s.memset_range_async(StatePools::s_offset(slot), S_SLOT_BYTES, 0, &c.stream)?;
                m.conv.memset_range_async(StatePools::conv_offset(slot), CONV_SLOT_BYTES, 0, &c.stream)?;
            }
            c.stream.sync()
        })();
        match r {
            Ok(()) => Ok(()),
            Err(e) => {
                self.slots.poison(slot);
                let e = ModelError::Cuda(e);
                self.note(&e);
                Err(e)
            }
        }
    }

    /// Debug (timing at a context nobody has prefilled): puts the begun, empty `slot` at committed length `len` without running anything.
    /// Pages for `len + WINDOW_ROWS` positions are allocated and the slot's position is `len`; the KV rows and the recurrent state are
    /// whatever the pools hold (zeros after load), so a step at that depth reads `len` keys of real size and computes on zeros. The
    /// slot drafts nothing (its head bookkeeping is off) unless `head`, which gives it the state of a prompt of `len` tokens that went
    /// through the head's prefill hook (a draft call has a carry to absorb). Needs a slot just begun.
    #[doc(hidden)]
    pub fn debug_seed_depth(&mut self, slot: usize, len: usize, head: bool) -> Result<(), ModelError> {
        self.check_alive()?;
        let have = self.slots.len(slot)?;
        if have != 0 {
            return Err(ModelError::Request(format!(
                "slot {slot} holds {have} tokens: debug_seed_depth needs a slot just begun"
            )));
        }
        if len + self.window_rows > self.core.opts.max_context {
            return Err(ModelError::Request(format!(
                "depth {len} + a window is past the model's context of {}",
                self.core.opts.max_context
            )));
        }
        self.kv_tab.ensure(slot, len + self.window_rows).map_err(|e| ModelError::Request(format!("{e:?}")))?;
        self.table_dirty = true;
        self.slots.advance(slot, len)?;
        if !(head && self.mtp_debug_seed_slot(slot, len)) {
            self.mtp_disable_slot(slot);
        }
        Ok(())
    }

    /// Releases the slot and its KV pages. Idempotent.
    pub fn end_slot(&mut self, slot: usize) {
        if self.slots.end(slot) {
            self.window.forget(slot);
            self.kv_tab.release(slot);
            self.table_dirty = true;
            self.mtp_reset_slot(slot);
            if let Some(l) = self.lag_host.get_mut(slot) {
                *l = 0;
            }
        }
    }

    /// One prefill slice (`b`: rows of one or more sequences, each continuing its slot from its committed length). The Mamba state is
    /// committed at `keep = rows`. With `want = Some(r)`, the candidates of batch row `r` (`MAX_CANDIDATES` of them, descending; the
    /// caller reads the first `n` it needs) and the lm_head runs on that row alone; with `None`, no logits are computed.
    pub fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        if let Some(r) = want {
            if r >= b.rows {
                return Err(ModelError::Request(format!("want row {r} of a slice of {} rows", b.rows)));
            }
        }
        self.prefill_with_head(b, want.map_or(Head::None, Head::Row))
    }

    /// [`NemotronModel::prefill_chunk`] that wants the candidates of SEVERAL rows of the slice (the last row of each of the sequences
    /// that end their prompt in it): the result's row `i` is the candidates of batch row `wants[i]`. The wanted rows' normed hidden states
    /// are gathered and go through the lm_head and the top-k in one call; every kernel of those is row invariant, so each row has the bits
    /// `prefill_chunk(.., Some(wants[i]))` gives it. This is what lets the server put the short prompts of a burst through one forward.
    /// An empty `wants` is `prefill_chunk(b, None)`.
    pub fn prefill_chunk_rows(&mut self, b: &RowBatch, wants: &[usize]) -> Result<Option<Candidates>, ModelError> {
        if let Some(r) = wants.iter().find(|&&r| r >= b.rows) {
            return Err(ModelError::Request(format!("want row {r} of a slice of {} rows", b.rows)));
        }
        if wants.is_empty() {
            return self.prefill_with_head(b, Head::None);
        }
        self.core.gather = wants.to_vec();
        let r = self.prefill_with_head(b, Head::Rows(wants.len()));
        self.core.gather.clear();
        r
    }

    fn prefill_with_head(&mut self, b: &RowBatch, head: Head) -> Result<Option<Candidates>, ModelError> {
        self.check_alive()?;
        let lim =
            StepLimits { max_rows: self.core.opts.prefill_rows, max_context: self.core.opts.max_context, vocab: self.vocab };
        let seqs = plan_step(b, &lim, &self.slots)?;
        // the slice moves these slots' state in place: a window of theirs that is still uncommitted can no longer be replayed
        for s in &seqs {
            self.window.forget(s.slot);
        }
        // and a lagged slot's state must be the true one before the slice continues it
        let slots_of: Vec<usize> = seqs.iter().map(|s| s.slot).collect();
        self.fold_lagged(&slots_of)?;
        match self.forward(b, Kind::Prefill, &seqs, head, MAX_CANDIDATES, false) {
            Ok(c) => {
                for s in &seqs {
                    self.slots.advance(s.slot, s.rows)?;
                }
                // the MTP head's cache is built chunk by chunk with the prompt, from this slice's post-`norm_f` rows (still in `bufs.xn`:
                // the forward ended synchronised). It only proposes: a failure here switches the slots' drafting off and costs the trunk
                // nothing, unless the device itself is lost.
                if self.core.mtp.is_some() {
                    if let Err(e) = self.mtp_prefill_hook(b, &seqs) {
                        for s in &seqs {
                            self.mtp_disable_slot(s.slot);
                        }
                        self.note(&e);
                        if matches!(&e, ModelError::Cuda(c) if c.is_fatal()) {
                            return Err(e);
                        }
                        eprintln!("rexl3 level=warn event=mtp_prefill_failed why={e} (no MTP drafts for these sequences)");
                    }
                }
                Ok(c)
            }
            Err(e) => {
                // layers before the failure have committed their state and layers after it have not
                for s in &seqs {
                    self.slots.poison(s.slot);
                }
                self.note(&e);
                Err(e)
            }
        }
    }

    /// [`NemotronModel::prefill_chunk`] for a slice that holds image rows: `splice` says which rows of which image's features replace
    /// which embedding rows of the slice (batch rows). Everything else, and everything after the embedding, is the text path unchanged
    /// (a slice with no splice IS `prefill_chunk`). The MTP head's hook reads the same splice for its own embedding half. The splice is
    /// cleared when the call ends, an error included, so a later window can never see it.
    pub fn prefill_chunk_mm(
        &mut self,
        b: &RowBatch,
        want: Option<usize>,
        splice: &[SpliceOp],
    ) -> Result<Option<Candidates>, ModelError> {
        if splice.is_empty() {
            return self.prefill_chunk(b, want);
        }
        if b.seq_slot.len() != 1 {
            return Err(ModelError::Request(format!(
                "an image splice is for a slice of one sequence, this one has {}",
                b.seq_slot.len()
            )));
        }
        for op in splice {
            if op.n == 0 || op.dst_row + op.n > b.rows || op.src_row + op.n > op.feats.tokens {
                return Err(ModelError::Request(format!(
                    "an image splice of {} rows (from row {} of {}) at slice row {} does not fit a slice of {} rows",
                    op.n, op.src_row, op.feats.tokens, op.dst_row, b.rows
                )));
            }
        }
        let mut sorted: Vec<(usize, usize)> = splice.iter().map(|o| (o.dst_row, o.dst_row + o.n)).collect();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0].1 > w[1].0) {
            return Err(ModelError::Request("two image splices of one slice overlap".into()));
        }
        self.core.splice = splice.to_vec();
        let r = self.prefill_chunk(b, want);
        self.core.splice.clear();
        r
    }

    /// A verify or decode window over one or more sequences: candidates for every row (`count` per row), the state untouched until
    /// [`NemotronModel::commit`].
    pub fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        self.window_inner(b, count, false)
    }

    /// [`NemotronModel::window_candidates`] for a decode round whose single-row sequences are certain to be kept (the engine commits
    /// `keep = 1` for a sequence of one row, whatever it samples): each such sequence's Mamba state is advanced in the pass that reads it
    /// (`window_commit_sel` and `commit_conv` inside the layer) instead of by a second pass at `commit`, which would read and write the
    /// state again (1.5 ms a sequence). A sequence of several rows (drafts that may be rejected) is left to its `commit`, as in a plain
    /// window. The candidates, the pools after the commit, and the slot's length are those of `window_candidates` followed by a
    /// `commit(slot, 1)` of every one-row sequence (`tests/nemotron_omni_agg.rs`); the commit of an applied sequence only moves the head's
    /// taps and the slot's length, and between the window and the commit the slot's state is one row ahead of its length (the sequence's
    /// next window is refused until it is committed, as ever). The window must be committed or its sequences released.
    pub fn window_candidates_fused(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        if !self.core.tp.topology().is_single() {
            return Err(ModelError::Config(
                "the fused decode step is not built for tensor parallelism (--tp-ranks 2): set REXL3_NEMO_FUSED_DECODE=0".into(),
            ));
        }
        self.window_inner(b, count, true)
    }

    fn window_inner(&mut self, b: &RowBatch, count: usize, fuse: bool) -> Result<Candidates, ModelError> {
        self.check_alive()?;
        if count == 0 || count > MAX_CANDIDATES {
            return Err(ModelError::Request(format!("candidate count must be 1..={MAX_CANDIDATES}, got {count}")));
        }
        let lim = StepLimits { max_rows: self.window_rows, max_context: self.core.opts.max_context, vocab: self.vocab };
        let seqs = plan_step(b, &lim, &self.slots)?;
        // the per-layer records are about to be overwritten: whatever the previous window left is gone
        self.window.clear();
        // a window with no single-row sequence has nothing to apply: it is the plain window (the same launches, the same graph)
        let fuse = fuse && seqs.iter().any(|s| s.rows == 1);
        // a lagged slot's S is the state as of its last fold: everything but a lagged single row reads the true state, so fold first
        if self.lag_k > 0 {
            let others: Vec<usize> = seqs.iter().filter(|s| !(fuse && s.rows == 1)).map(|s| s.slot).collect();
            self.fold_lagged(&others)?;
        }
        match self.forward(b, Kind::Window, &seqs, Head::All, count, fuse) {
            Ok(Some(c)) => {
                self.window.record(&seqs);
                if fuse {
                    for s in seqs.iter().filter(|s| s.rows == 1) {
                        self.window.mark_applied(s.slot);
                        if self.lag_k > 0 {
                            // the row was folded (the ring is empty again) or pushed to ring row `lag`
                            let p = self.lag_host[s.slot];
                            self.lag_host[s.slot] = if p + 1 >= self.lag_k { 0 } else { p + 1 };
                        }
                    }
                }
                Ok(c)
            }
            Ok(None) => Err(ModelError::Request("a window produced no candidates".into())),
            Err(e) => {
                if fuse {
                    // layers before the failure have stored their state for the applied sequences and layers after it have not
                    for s in seqs.iter().filter(|s| s.rows == 1) {
                        self.slots.poison(s.slot);
                    }
                }
                self.note(&e);
                Err(e)
            }
        }
    }

    /// Keeps the first `keep` rows (1..=64) of `slot`'s part of the last window: replays them into the slot's Mamba state and
    /// conv history in every layer, and moves the slot's committed length. A refusal or a failed replay is returned by `flush`.
    pub fn commit(&mut self, slot: usize, keep: usize) {
        if self.fatal.is_some() {
            return;
        }
        // a fused window already moved the state of its single-row sequences: only the head's taps and the length are left
        let applied = self.window.is_applied(slot);
        let span = match self.window.commit(slot, keep) {
            Ok(s) => s,
            Err(e) => {
                self.refused.get_or_insert_with(|| e.to_string());
                return;
            }
        };
        // the head's store: the kept rows' taps (before the slot's length moves: the window's first row is at that position)
        let row0 = self.slots.len(slot).ok();
        self.mtp_commit_taps(&span, row0);
        if !applied {
            if let Err(e) = self.replay(&span) {
                self.slots.poison(slot);
                self.fatal.get_or_insert(Fatal::StateLost(format!(
                    "the replay of {} kept rows of slot {slot} failed in a Mamba layer: {e}",
                    span.keep
                )));
                return;
            }
        }
        if let Err(e) = self.slots.advance(slot, span.keep) {
            self.refused.get_or_insert_with(|| e.to_string());
        }
    }

    /// Folds the pending rows of `slots` (those with any) into the S pool of every Mamba layer: afterwards the pool holds the slots'
    /// true state. One launch for all layers and slots, on the model's stream. A model without a lagged decode has nothing to do.
    pub(super) fn fold_lagged(&mut self, slots: &[usize]) -> Result<(), ModelError> {
        if self.lag_k == 0 {
            return Ok(());
        }
        let mut todo: Vec<(usize, usize)> = Vec::new();
        for &s in slots {
            if s < self.lag_host.len() && self.lag_host[s] > 0 && !todo.iter().any(|t| t.0 == s) {
                todo.push((s, self.lag_host[s]));
            }
        }
        if todo.is_empty() {
            return Ok(());
        }
        let Some(table) = &self.lag_table else { return Ok(()) };
        let ids: Vec<i32> = todo.iter().map(|t| t.0 as i32).collect();
        let lags: Vec<i32> = todo.iter().map(|t| t.1 as i32).collect();
        upload_ordered(&mut self.fold_slots, &ids, &self.core.stream)?;
        upload_ordered(&mut self.fold_lags, &lags, &self.core.stream)?;
        nemotron_ssm::fold_lag_all(&self.core.stream, table, &self.fold_slots, &self.fold_lags, todo.len())?;
        for (s, _) in todo {
            self.lag_host[s] = 0;
        }
        Ok(())
    }

    /// Folds every lagged slot (a test's or a snapshot's view of the state). Synchronises.
    #[doc(hidden)]
    pub fn debug_fold_all(&mut self) -> Result<(), ModelError> {
        let all: Vec<usize> = (0..self.lag_host.len()).collect();
        self.fold_lagged(&all)?;
        self.core.stream.sync()?;
        Ok(())
    }

    /// The fold period of the lagged decode (0: off).
    pub fn lag_period(&self) -> usize {
        self.lag_k
    }

    /// `commit_state` and `commit_conv` of `span` in every Mamba layer, on the model's stream: two launches over all the layers.
    fn replay(&mut self, span: &CommitSpan) -> Result<(), CudaError> {
        // in the stream's order: the replay below must read this slot's arguments, and the previous commit of the slot has read the old ones
        let vals = [span.start as i32, span.end as i32, span.slot as i32, span.keep as i32];
        upload_ordered(&mut self.commit_args[span.slot].all, &vals, &self.core.stream)?;
        let args = &self.commit_args[span.slot];
        // one sequence whose rows are `start..end` of the window the records hold
        let batch =
            Batch { seq_start: &args.seq_start, slot: &args.slot, n_seq: 1, rows: span.end, n_slots: self.core.opts.slots };
        match &self.commit_table {
            Some(table) => nemotron_ssm::commit_all(&self.core.stream, table, &batch, &args.keep),
            None => Ok(()),
        }
    }

    /// Waits for the device. Returns a latched error, the error the stream recorded, or the refusal of a `commit` since the last flush.
    pub fn flush(&mut self) -> Result<(), ModelError> {
        self.check_alive()?;
        if let Err(e) = self.core.stream.sync() {
            let e = ModelError::Cuda(e);
            self.note(&e);
            return Err(e);
        }
        match self.refused.take() {
            Some(m) => Err(ModelError::Request(m)),
            None => Ok(()),
        }
    }

    /// One step through the whole trunk. `seqs` has been checked by [`plan_step`]. Returns the candidates of the rows `head` names.
    /// Always ends with the stream synchronised, an error included: the next step writes the index arrays the failed one may still
    /// have been reading.
    fn forward(
        &mut self,
        b: &RowBatch,
        kind: Kind,
        seqs: &[SeqPlan],
        head: Head,
        count: usize,
        fuse: bool,
    ) -> Result<Option<Candidates>, ModelError> {
        // contract P2: a prefill slice (and only a prefill slice: never a window, wide or not) runs inside the tensor-core scope
        let _p2 = super::contract::prefill_scope(kind == Kind::Prefill);
        // the MTP tap dump (calibration, off by default): a window's token ids, for the records of its committed rows
        if kind == Kind::Window && super::tapdump::enabled() {
            if let Some(m) = self.core.mtp.as_mut() {
                m.dump_ids.clone_from(&b.ids);
                m.dump_slots.clone_from(&b.row_slot);
                m.dump_pos.clone_from(&b.positions);
            }
        }
        let r = self.forward_inner(b, kind, seqs, head, count, fuse);
        if r.is_err() {
            // best effort: the error being returned is the one that matters
            let _ = self.core.stream.sync();
        }
        r
    }

    fn forward_inner(
        &mut self,
        b: &RowBatch,
        kind: Kind,
        seqs: &[SeqPlan],
        head: Head,
        count: usize,
        fuse: bool,
    ) -> Result<Option<Candidates>, ModelError> {
        let rows = b.rows;
        crate::hosttrace::mark("fwd_begin", rows as u64);
        let grown = grow_pages(&mut self.kv_tab, seqs);
        if grown.is_err() {
            // an earlier sequence of the batch may have taken pages before the one that did not fit
            self.table_dirty = true;
        }
        // Every array below is written in the stream's order (`upload_ordered`), so the first kernel that reads one sees the new bytes and
        // no kernel of an earlier step still reading the old ones is overtaken. No device synchronisation is made.
        if grown? || self.table_dirty {
            let lists = block_lists(&self.kv_tab, self.core.opts.slots);
            let t = &mut self.core.table;
            // the layout of `BlockTable::upload`: `[slots][max_blocks]`, unused entries 0
            let mut flat = vec![0i32; t.slots.max(1) * t.max_blocks];
            for (s, blocks) in lists.iter().enumerate().take(t.slots.max(1)) {
                assert!(blocks.len() <= t.max_blocks, "slot {s}: {} blocks > max_blocks {}", blocks.len(), t.max_blocks);
                for (i, &blk) in blocks.iter().enumerate() {
                    flat[s * t.max_blocks + i] = blk as i32;
                }
            }
            upload_ordered(&mut t.buf, &flat, &self.core.stream)?;
            self.table_dirty = false;
        }
        upload_ordered(&mut self.ids, &b.ids, &self.core.stream)?;
        upload_ordered(&mut self.seq_start, &b.seq_start, &self.core.stream)?;
        upload_ordered(&mut self.seq_slot, &b.seq_slot, &self.core.stream)?;
        if kind == Kind::Prefill {
            let keep: Vec<i32> = seqs.iter().map(|s| s.rows as i32).collect();
            upload_ordered(&mut self.core.keep, &keep, &self.core.stream)?;
        }
        if LEGACY_UPLOADS.load(Ordering::Relaxed) {
            self.row_index.upload(&b.row_slot, &b.positions)?;
        } else {
            self.row_index.upload_on(&b.row_slot, &b.positions, &self.core.stream)?;
        }
        if fuse {
            // which sequences' one row is applied inside the layers: a device array the kernels read, so the graph does not hold it
            let flags: Vec<i32> = seqs.iter().map(|s| i32::from(s.rows == 1)).collect();
            upload_ordered(&mut self.apply, &flags, &self.core.stream)?;
            if self.lag_k > 0 {
                // the rows each applied sequence has pending in its ring (device arrays the kernels read: graph-safe)
                let lags: Vec<i32> = seqs.iter().map(|s| if s.rows == 1 { self.lag_host[s.slot] as i32 } else { 0 }).collect();
                upload_ordered(&mut self.lag_dev, &lags, &self.core.stream)?;
            }
        }
        // A window over every row (decode and verify) goes through the graph cache unless something adds launches or host work to the loop.
        // The attention launch is made for the deepest row's chunks rounded up to a power of two then, so one graph serves a range of
        // contexts (a row with fewer chunks of its own exits early: the same bits).
        let cached = self.graphs.on
            && kind == Kind::Window
            && head == Head::All
            && self.core.tap.is_none()
            && self.core.splice.is_empty()
            && self.core.mtp.as_ref().is_none_or(|m| m.mutant.is_none());
        let chunks = if self.core.map.count(LayerKind::Attn) == 0 {
            0
        } else if cached {
            let cap = self.core.opts.max_context.div_ceil(CHUNK_KEYS);
            let bucket = chunk_bucket(self.row_index.real_chunks(), cap);
            self.row_index.set_min_chunks(bucket);
            bucket
        } else {
            self.row_index.real_chunks()
        };
        let batch =
            Batch { seq_start: &self.seq_start, slot: &self.seq_slot, n_seq: seqs.len(), rows, n_slots: self.core.opts.slots };
        let lag = (fuse && self.lag_k > 0).then_some(&self.lag_dev);
        let ctx = StepCtx { rows, kind, batch, row_index: &self.row_index, apply: fuse.then_some(&self.apply), lag };

        crate::hosttrace::mark("fwd_launch_begin", u64::from(cached));
        let m = if cached {
            let key = GraphKey { rows, n_seq: seqs.len(), count, chunks, fused: fuse };
            run_cached(&mut self.graphs, &mut self.core, &self.ids, self.eps, self.vocab, &ctx, head, count, key)?
        } else {
            launch_steps(&mut self.core, &self.ids, self.eps, self.vocab, &ctx, head, count)?
        };
        crate::hosttrace::mark("fwd_launch_end", 0);
        let c = &mut self.core;
        if head == Head::None {
            c.stream.sync()?;
            return Ok(None);
        }
        c.stream.sync()?;
        crate::hosttrace::mark("fwd_synced", 0);
        let mut ids = vec![0i32; m * count];
        let mut vals = vec![0f32; m * count];
        c.bufs.topk_ids.download_range(0, &mut ids)?;
        c.bufs.topk_vals.download_range(0, &mut vals)?;
        Ok(Some(Candidates { rows: m, count, ids, vals }))
    }

    /// The Mamba state of `slot` in the Mamba layer of ordinal `ordinal`: the S pool slot (f32 `[128][128][64]`) and the conv history
    /// (bf16 bits `[3][10240]`). Waits for the device. For the equivalence tests that compare the pools bit for bit.
    #[doc(hidden)]
    pub fn debug_mamba_state(&mut self, ordinal: usize, slot: usize) -> Result<(Vec<f32>, Vec<u16>), ModelError> {
        // a lagged slot's pool is the state as of its last fold: the test wants the true one
        self.fold_lagged(&[slot])?;
        self.flush()?;
        let Some(m) = self.core.mamba.get(ordinal) else {
            return Err(ModelError::Request(format!("Mamba layer {ordinal} of {}", self.core.mamba.len())));
        };
        if slot >= self.core.opts.slots {
            return Err(ModelError::Request(format!("slot {slot} of {}", self.core.opts.slots)));
        }
        let mut s = vec![0f32; S_SLOT_ELEMS];
        let mut conv = vec![0u16; CONV_SLOT_ELEMS];
        m.s.download_range(StatePools::s_offset(slot), &mut s)?;
        m.conv.download_range(StatePools::conv_offset(slot), &mut conv)?;
        Ok((s, conv))
    }

    /// The fp16 bits of the K and V rows `first..first + n` of `slot` in the attention layer of ordinal `ordinal`, `[n][2][128]` each
    /// (an int8 cache's values dequantised: exact fp16 values). Waits for the device.
    #[doc(hidden)]
    pub fn debug_kv_rows(
        &mut self,
        ordinal: usize,
        slot: usize,
        first: usize,
        n: usize,
    ) -> Result<(Vec<u16>, Vec<u16>), ModelError> {
        self.flush()?;
        if ordinal >= self.core.kv_geom.layers {
            return Err(ModelError::Request(format!("attention layer {ordinal} of {}", self.core.kv_geom.layers)));
        }
        let have = self.kv_tab.blocks_of(slot).len() * BLOCK_TOKENS;
        if slot >= self.core.opts.slots || first + n > have {
            return Err(ModelError::Request(format!("rows {first}..{} of slot {slot}, which holds pages for {have}", first + n)));
        }
        let g = self.core.kv_geom;
        let (mut k, mut v) = (vec![0u16; n * KV_TOKEN_ELEMS], vec![0u16; n * KV_TOKEN_ELEMS]);
        for i in 0..n {
            let (blk, off) = KvGeom::locate(first + i);
            let block = self.kv_tab.blocks_of(slot)[blk] as usize;
            if g.dtype.is_8bit() {
                let page = g.layer_offset(ordinal) + block * g.page_bytes();
                for (pool, out) in [(&self.core.kv.k, &mut k), (&self.core.kv.v, &mut v)] {
                    let (mut vals, mut exps) = (vec![0u8; KV_TOKEN_ELEMS], [0u8; 2]);
                    pool.download_range(page + off * KV_TOKEN_ELEMS, &mut vals)?;
                    pool.download_range(page + kv8::EXP_OFF + off * 2, &mut exps)?;
                    for (j, &b) in vals.iter().enumerate() {
                        out[i * KV_TOKEN_ELEMS + j] = kv8::dequant_i8_f16_bits(b, exps[j / 128] as i8);
                    }
                }
                continue;
            }
            let at = g.layer_offset(ordinal) + (block * BLOCK_TOKENS + off) * KV_TOKEN_ELEMS * 2;
            self.core.kv.k.download_range(at, &mut k[i * KV_TOKEN_ELEMS..(i + 1) * KV_TOKEN_ELEMS])?;
            self.core.kv.v.download_range(at, &mut v[i * KV_TOKEN_ELEMS..(i + 1) * KV_TOKEN_ELEMS])?;
        }
        Ok((k, v))
    }
}

/// Fires the tap on `buf` if one is installed.
fn fire(
    tap: &mut Option<Box<dyn Tap>>,
    point: Point,
    buf: &DevBuf,
    rows: usize,
    width: usize,
    st: &Stream,
) -> Result<(), ModelError> {
    match tap.as_mut() {
        Some(t) => t.tap(point, buf, rows, width, st),
        None => Ok(()),
    }
}

impl NemotronRuntime for NemotronModel {
    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
    fn has_mtp(&self) -> bool {
        self.has_mtp()
    }
    fn head_form(&self) -> super::spec::HeadForm {
        self.mtp_head_form()
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.begin_slot(slot)
    }
    fn end_slot(&mut self, slot: usize) {
        self.end_slot(slot)
    }
    fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        self.prefill_chunk(b, want)
    }
    fn prefill_chunk_mm(
        &mut self,
        b: &RowBatch,
        want: Option<usize>,
        splice: &[SpliceOp],
    ) -> Result<Option<Candidates>, ModelError> {
        NemotronModel::prefill_chunk_mm(self, b, want, splice)
    }
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        self.window_candidates(b, count)
    }
    fn window_candidates_fused(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        NemotronModel::window_candidates_fused(self, b, count)
    }
    fn prefill_rows_supported(&self) -> bool {
        // the batched prefill is the single rank's (a TP rank's prefill is the serial slices it was verified with)
        self.core.tp.topology().is_single()
    }
    fn prefill_chunk_rows(&mut self, b: &RowBatch, wants: &[usize]) -> Result<Option<Candidates>, ModelError> {
        NemotronModel::prefill_chunk_rows(self, b, wants)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.commit(slot, keep)
    }
    fn flush(&mut self) -> Result<(), ModelError> {
        self.flush()
    }
    fn mtp_draft(
        &mut self,
        asks: &[super::mtp::DraftAsk<'_>],
        keyed: &[Option<gb_sampling::Sampling>],
    ) -> Result<Vec<Option<super::mtp::Drafts>>, ModelError> {
        // a model without the head (REXL3_MTP=off, a layer-limited oracle) drafts nothing: every ask is answered `None`
        if self.core.mtp.is_none() {
            return Ok(vec![None; asks.len()]);
        }
        NemotronModel::mtp_draft(self, asks, keyed)
    }
    fn mtp_disable_slot(&mut self, slot: usize) {
        NemotronModel::mtp_disable_slot(self, slot)
    }
    fn mtp_pending_rows(&self, slot: usize) -> Option<usize> {
        NemotronModel::mtp_pending_rows(self, slot)
    }
    fn graph_counters(&self) -> Option<gb_engine::GraphCounters> {
        let g = &self.graphs.stats;
        let [draft_captures, draft_replays, draft_eager] = self.mtp_draft_graph_stats();
        Some(gb_engine::GraphCounters {
            replays: g.replays,
            captures: g.captures,
            eager_windows: g.eager_windows,
            capture_failures: g.capture_failures,
            evictions: g.evictions,
            capture_ms: g.capture_ms,
            draft_replays,
            draft_captures,
            draft_eager,
        })
    }
    // the prefix cache (`snapshot.rs`; gates: `tests/nemotron_omni_prefix_gpu.rs`). The family's `prefix_cache()` answers this (the tripwire
    // test `the_family_says_it_can_cache_exactly_when_its_model_can_snapshot`).
    fn snapshots_supported() -> bool {
        true
    }
    fn enable_snapshots(&mut self, slots: usize) -> Result<(), ModelError> {
        NemotronModel::enable_snapshots(self, slots)
    }
    fn snapshot_arena_bytes(&self) -> u64 {
        NemotronModel::snapshot_arena_bytes(self)
    }
    fn free_snapshots(&self) -> usize {
        NemotronModel::free_snapshots(self)
    }
    fn snapshot_slot(&mut self, slot: usize, len: usize) -> Result<Option<gb_engine::PrefixHandle>, ModelError> {
        NemotronModel::snapshot_slot(self, slot, len)
    }
    fn restore_snapshot(&mut self, slot: usize, handle: &gb_engine::PrefixHandle, len: usize) -> Result<(), ModelError> {
        NemotronModel::restore_snapshot(self, slot, handle, len)
    }
    fn release_snapshot(&mut self, handle: gb_engine::PrefixHandle) -> Result<(), ModelError> {
        NemotronModel::release_snapshot(self, handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gb_engine::RowRequest;

    /// `(slot, first position, rows)` per sequence; the ids are 1, 2, 3, ...
    fn batch(reqs: &[(usize, usize, usize)]) -> RowBatch {
        let mut n = 0u32;
        let r: Vec<RowRequest> = reqs
            .iter()
            .map(|&(slot, first, rows)| RowRequest {
                slot,
                first_position: first as u64,
                tokens: (0..rows)
                    .map(|_| {
                        n += 1;
                        n
                    })
                    .collect(),
            })
            .collect();
        RowBatch::from_requests(&r)
    }

    fn limits() -> StepLimits {
        StepLimits { max_rows: 16, max_context: 2048, vocab: 131072 }
    }

    fn live(slots: usize, lens: &[(usize, usize)]) -> SlotTable {
        let mut t = SlotTable::new(slots);
        for &(slot, len) in lens {
            t.begin(slot).unwrap();
            t.advance(slot, len).unwrap();
        }
        t
    }

    fn refused(r: Result<Vec<SeqPlan>, ModelError>, needle: &str) {
        match r {
            Err(ModelError::Request(m)) => assert!(m.contains(needle), "`{m}` should mention `{needle}`"),
            other => panic!("expected a Request error mentioning `{needle}`, got {other:?}"),
        }
    }

    // ---- pool index math

    #[test]
    fn a_slot_of_mamba_state_is_170_229_760_bytes_over_40_layers() {
        assert_eq!(S_SLOT_BYTES, 4_194_304);
        assert_eq!(CONV_SLOT_BYTES, 61_440);
        assert_eq!(StatePools { slots: 1, layers: 40 }.slot_bytes(), 170_229_760);
        assert_eq!(StatePools { slots: 6, layers: 40 }.pool_bytes(), 6 * 170_229_760);
    }

    #[test]
    fn slots_sit_one_slot_stride_apart_in_each_pool() {
        assert_eq!(StatePools::s_offset(0), 0);
        assert_eq!(StatePools::s_offset(3), 3 * 128 * 128 * 64 * 4);
        assert_eq!(StatePools::conv_offset(3), 3 * 3 * 10240 * 2);
        // every slot stays 16-byte aligned (the kernels read float4)
        for slot in 0..7 {
            assert_eq!(StatePools::s_offset(slot) % 16, 0);
            assert_eq!(StatePools::conv_offset(slot) % 16, 0);
        }
    }

    #[test]
    fn a_window_row_costs_a_mamba_layer_62_720_bytes() {
        assert_eq!(RECORD_ROW_BYTES, 21_760);
        assert_eq!(XBC_ROW_BYTES, 40_960);
        assert_eq!(WINDOW_ROW_BYTES, 62_720);
        let p = StatePools { slots: 1, layers: 40 };
        assert_eq!(p.window_bytes(1), 2_508_800, "2.51 MB a row over 40 layers");
        assert_eq!(p.window_bytes(WINDOW_ROWS), 64 * 2_508_800);
    }

    #[test]
    fn kv_costs_8_kib_a_token_over_8_attention_layers_and_the_stripes_tile_the_pool() {
        let g = KvGeom { pages: 4096, layers: 8, dtype: KvDtype::F16 };
        assert_eq!(g.token_bytes(), 8192);
        assert_eq!(g.pool_bytes(), 8 << 30, "1M tokens are 8 GiB");
        assert_eq!(g.layer_bytes(), 4096 * 256 * 256 * 2);
        assert_eq!(g.half_bytes() * 2, g.pool_bytes());
        for a in 0..8 {
            assert_eq!(g.layer_offset(a) % 16, 0);
            let end = if a == 7 { g.half_bytes() } else { g.layer_offset(a + 1) };
            assert_eq!(g.layer_offset(a) + g.layer_bytes(), end, "layer {a}");
        }
        assert_eq!(KvGeom::locate(0), (0, 0));
        assert_eq!(KvGeom::locate(255), (0, 255));
        assert_eq!(KvGeom::locate(256), (1, 0));
        assert_eq!((KvGeom::pages_for(0), KvGeom::pages_for(1), KvGeom::pages_for(256), KvGeom::pages_for(257)), (0, 1, 1, 2));
    }

    #[test]
    fn an_int8_kv_costs_516_bytes_a_token_and_layer_and_its_stripes_tile_the_pool() {
        // 9 stripes (the head's included) of 4,096 pages: 1M tokens
        let g = KvGeom { pages: 4096, layers: 9, dtype: KvDtype::Int8 };
        assert_eq!(g.page_bytes(), 66_048);
        assert_eq!(g.token_bytes(), 9 * 516);
        assert_eq!(g.pool_bytes(), 2 * 9 * 4096 * 66_048, "1M tokens with the head: 4.54 GiB against 9 GiB");
        let f16 = KvGeom { dtype: KvDtype::F16, ..g };
        assert!(g.pool_bytes() * 2 < f16.pool_bytes() + 9 * 4096 * 2 * 512 * 2 + 1);
        for a in 0..9 {
            assert_eq!(g.layer_offset(a) % 16, 0, "cp.async needs 16-byte aligned pages");
            let end = if a == 8 { g.half_bytes() } else { g.layer_offset(a + 1) };
            assert_eq!(g.layer_offset(a) + g.layer_bytes(), end, "layer {a}");
        }
    }

    // ---- slot lifecycle

    #[test]
    fn a_slot_begins_once_ends_idempotently_and_can_begin_again() {
        let mut t = SlotTable::new(2);
        assert_eq!(t.len(0), Err(SlotError::NotBegun { slot: 0 }));
        t.begin(0).unwrap();
        assert_eq!(t.len(0), Ok(0));
        assert_eq!(t.advance(0, 5), Ok(5));
        assert_eq!(t.begin(0), Err(SlotError::AlreadyBegun { slot: 0 }), "a double begin is refused");
        assert_eq!(t.len(0), Ok(5), "and leaves the slot as it was");
        assert!(t.end(0));
        assert!(!t.end(0), "ending twice is a no-op");
        assert!(!t.end(7), "ending a slot out of range is a no-op");
        assert_eq!(t.len(0), Err(SlotError::NotBegun { slot: 0 }));
        t.begin(0).unwrap();
        assert_eq!(t.len(0), Ok(0), "a fresh sequence starts at zero");
        assert_eq!(t.begin(5), Err(SlotError::OutOfRange { slot: 5, slots: 2 }));
    }

    #[test]
    fn a_poisoned_slot_runs_nothing_until_it_is_ended() {
        let mut t = SlotTable::new(1);
        t.begin(0).unwrap();
        t.poison(0);
        assert_eq!(t.len(0), Err(SlotError::Unusable { slot: 0 }));
        assert_eq!(t.advance(0, 1), Err(SlotError::Unusable { slot: 0 }));
        assert_eq!(t.begin(0), Err(SlotError::AlreadyBegun { slot: 0 }));
        assert!(t.end(0), "ending a poisoned slot releases it");
        t.begin(0).unwrap();
        let mut f = SlotTable::new(1);
        f.poison(0);
        assert_eq!(f.len(0), Err(SlotError::NotBegun { slot: 0 }), "a free slot cannot be poisoned");
    }

    // ---- KV block table

    #[test]
    fn kv_pages_grow_on_demand_and_come_back_when_the_slot_ends() {
        let mut tab = KvTables::new(8, 3, 4);
        let seq = |slot, first_pos, rows| SeqPlan { slot, start: 0, rows, first_pos };
        assert!(grow_pages(&mut tab, &[seq(0, 0, 100)]).unwrap());
        assert_eq!(tab.blocks_of(0).len(), 1);
        // positions 100..356 reach the second page
        assert!(grow_pages(&mut tab, &[seq(0, 100, 256)]).unwrap());
        assert_eq!(tab.blocks_of(0).len(), 2);
        // a step that stays inside the pages changes nothing
        assert!(!grow_pages(&mut tab, &[seq(0, 356, 100)]).unwrap());
        assert_eq!(tab.blocks_of(0).len(), 2);
        // a second slot takes its own pages
        assert!(grow_pages(&mut tab, &[seq(1, 0, 600)]).unwrap());
        assert_eq!(tab.blocks_of(1).len(), 3);
        let lists = block_lists(&tab, 3);
        assert_eq!(lists.len(), 3);
        assert!(lists[2].is_empty());
        let mut all: Vec<u32> = lists.iter().flatten().copied().collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 5, "no page is shared between slots");
        assert_eq!(tab.free_blocks(), 3);
        tab.release(0);
        assert_eq!(tab.free_blocks(), 5);
        assert!(block_lists(&tab, 3)[0].is_empty());
        // the released pages serve the next sequence
        assert!(grow_pages(&mut tab, &[seq(2, 0, 1000)]).unwrap());
        assert_eq!(tab.blocks_of(2).len(), 4);
    }

    #[test]
    fn a_context_or_a_pool_that_is_too_small_is_a_request_error_and_changes_nothing() {
        let mut tab = KvTables::new(3, 2, 2);
        let seq = |slot, first_pos, rows| SeqPlan { slot, start: 0, rows, first_pos };
        // two pages is the most a slot's table holds (max_blocks = 2)
        match grow_pages(&mut tab, &[seq(0, 0, 513)]) {
            Err(ModelError::Request(m)) => assert!(m.contains("ContextTooLong"), "{m}"),
            other => panic!("{other:?}"),
        }
        assert!(grow_pages(&mut tab, &[seq(0, 0, 512)]).unwrap());
        // one page is left in the pool and slot 1 asks for two
        match grow_pages(&mut tab, &[seq(1, 0, 512)]) {
            Err(ModelError::Request(m)) => assert!(m.contains("OutOfPages"), "{m}"),
            other => panic!("{other:?}"),
        }
        assert!(tab.blocks_of(1).is_empty());
        assert_eq!(tab.free_blocks(), 1);
    }

    // ---- window records and commit arithmetic

    fn seqs(windows: &[(usize, usize)]) -> Vec<SeqPlan> {
        let mut start = 0;
        windows
            .iter()
            .map(|&(slot, rows)| {
                let s = SeqPlan { slot, start, rows, first_pos: 0 };
                start += rows;
                s
            })
            .collect()
    }

    #[test]
    fn a_commit_keeps_between_one_row_and_all_of_the_slots_rows() {
        let mut w = WindowLog::new();
        w.record(&seqs(&[(3, 5), (1, 2)]));
        // rows 0..5 are slot 3's, rows 5..7 are slot 1's
        assert_eq!(w.commit(1, 2), Ok(CommitSpan { slot: 1, start: 5, end: 7, keep: 2 }));
        assert_eq!(w.commit(3, 1), Ok(CommitSpan { slot: 3, start: 0, end: 5, keep: 1 }));
        // every row can be kept, and the first alone
        for keep in 1..=5 {
            let mut w = WindowLog::new();
            w.record(&seqs(&[(0, 5)]));
            assert_eq!(w.commit(0, keep), Ok(CommitSpan { slot: 0, start: 0, end: 5, keep }));
        }
    }

    #[test]
    fn a_keep_of_zero_or_past_the_window_is_refused_and_the_window_stays_committable() {
        let mut w = WindowLog::new();
        w.record(&seqs(&[(0, 3)]));
        assert_eq!(w.commit(0, 0), Err(CommitError::KeepZero { slot: 0 }));
        assert_eq!(w.commit(0, 4), Err(CommitError::KeepPastWindow { slot: 0, keep: 4, rows: 3 }));
        assert_eq!(w.pending(0), Some(3));
        assert!(w.commit(0, 3).is_ok());
    }

    #[test]
    fn a_second_commit_without_a_new_window_is_refused() {
        let mut w = WindowLog::new();
        assert_eq!(w.commit(0, 1), Err(CommitError::NoWindow { slot: 0 }), "no window has run");
        w.record(&seqs(&[(0, 4), (1, 4)]));
        assert!(w.commit(0, 2).is_ok());
        assert_eq!(w.commit(0, 2), Err(CommitError::NoWindow { slot: 0 }));
        assert_eq!(w.pending(0), None);
        assert_eq!(w.pending(1), Some(4), "the other slot's rows are still there");
        // a new window makes the slot committable again
        w.record(&seqs(&[(0, 1)]));
        assert!(w.commit(0, 1).is_ok());
        assert_eq!(w.commit(1, 1), Err(CommitError::NoWindow { slot: 1 }), "slot 1 was not in the new window");
    }

    #[test]
    fn a_slot_that_was_dropped_or_whose_window_was_overwritten_cannot_commit() {
        let mut w = WindowLog::new();
        w.record(&seqs(&[(0, 4), (1, 4)]));
        w.forget(1);
        assert_eq!(w.commit(1, 1), Err(CommitError::NoWindow { slot: 1 }));
        w.clear();
        assert_eq!(w.commit(0, 1), Err(CommitError::NoWindow { slot: 0 }));
    }

    // ---- step validation and the slice clamp

    #[test]
    fn a_slice_larger_than_the_limit_is_a_request_error() {
        let t = live(2, &[(0, 0)]);
        let lim = limits();
        assert!(plan_step(&batch(&[(0, 0, 16)]), &lim, &t).is_ok());
        refused(plan_step(&batch(&[(0, 0, 17)]), &lim, &t), "1..=16 rows, got 17");
        refused(plan_step(&RowBatch::default(), &lim, &t), "at least one row");
    }

    #[test]
    fn a_batch_must_continue_each_slot_from_its_committed_length() {
        let t = live(3, &[(0, 10), (2, 0)]);
        let lim = limits();
        let ok = plan_step(&batch(&[(0, 10, 3), (2, 0, 2)]), &lim, &t).unwrap();
        assert_eq!(
            ok,
            vec![SeqPlan { slot: 0, start: 0, rows: 3, first_pos: 10 }, SeqPlan { slot: 2, start: 3, rows: 2, first_pos: 0 }]
        );
        refused(plan_step(&batch(&[(0, 9, 3)]), &lim, &t), "next row is position 10, not 9");
        refused(plan_step(&batch(&[(0, 11, 3)]), &lim, &t), "next row is position 10, not 11");
        refused(plan_step(&batch(&[(1, 0, 1)]), &lim, &t), "never begun");
        refused(plan_step(&batch(&[(9, 0, 1)]), &lim, &t), "slot 9 of 3");
    }

    #[test]
    fn a_slot_twice_in_a_batch_and_a_position_past_the_context_are_refused() {
        let t = live(2, &[(0, 0), (1, 2040)]);
        let lim = limits();
        refused(plan_step(&batch(&[(0, 0, 2), (0, 2, 1)]), &lim, &t), "in the batch twice");
        assert!(plan_step(&batch(&[(1, 2040, 8)]), &lim, &t).is_ok(), "the last position is 2047");
        refused(plan_step(&batch(&[(1, 2040, 9)]), &lim, &t), "past the context of 2048");
    }

    #[test]
    fn a_tree_an_mrope_batch_inconsistent_arrays_and_a_bad_token_are_refused() {
        let t = live(1, &[(0, 0)]);
        let lim = limits();
        let mut b = batch(&[(0, 0, 3)]);
        b.parents = Some(vec![-1, 0, 1]);
        refused(plan_step(&b, &lim, &t), "trees");
        let mut b = batch(&[(0, 0, 3)]);
        b.rope3 = Some(vec![0; 9]);
        refused(plan_step(&b, &lim, &t), "M-RoPE");
        let mut b = batch(&[(0, 0, 3)]);
        b.positions[2] = 7;
        refused(plan_step(&b, &lim, &t), "not consecutive");
        let mut b = batch(&[(0, 0, 3)]);
        b.ids.pop();
        refused(plan_step(&b, &lim, &t), "ids");
        let mut b = batch(&[(0, 0, 3)]);
        b.ids[1] = 131072;
        refused(plan_step(&b, &lim, &t), "outside the vocabulary");
        let mut b = batch(&[(0, 0, 3)]);
        b.ids[1] = -1;
        refused(plan_step(&b, &lim, &t), "outside the vocabulary");
        let mut b = batch(&[(0, 0, 3)]);
        b.row_slot[1] = 5;
        refused(plan_step(&b, &lim, &t), "names a slot other than 0");
    }

    #[test]
    fn a_poisoned_slot_is_refused_by_name() {
        let mut t = live(1, &[(0, 4)]);
        t.poison(0);
        refused(plan_step(&batch(&[(0, 4, 1)]), &limits(), &t), "unusable");
    }

    // ---- options

    fn cfg() -> OmniConfig {
        const PACK: &str = include_str!("../../../../../testdata/nemotron_omni/config.json");
        OmniConfig::from_json(&serde_json::from_str(PACK).unwrap()).unwrap()
    }

    fn opts() -> ModelOptions {
        ModelOptions { slots: 2, kv_pages: 32, max_context: 8192, prefill_rows: 64, layers: None, mtp: false }
    }

    #[test]
    fn the_prefill_rows_switch_takes_256_or_512_and_nothing_else() {
        assert_eq!(parse_prefill_rows_cap(None), Ok(WIDE_PREFILL_ROWS));
        assert_eq!(parse_prefill_rows_cap(Some("")), Ok(WIDE_PREFILL_ROWS));
        assert_eq!(parse_prefill_rows_cap(Some("256")), Ok(256));
        assert_eq!(parse_prefill_rows_cap(Some(" 512 ")), Ok(512));
        for bad in ["1024", "300", "64", "0", "-1", "wide", "512x"] {
            assert!(parse_prefill_rows_cap(Some(bad)).is_err(), "{bad} must be refused");
        }
        // the wide cap stays inside what the window path, the memory model and the kernels were sized for
        const { assert!(WIDE_PREFILL_ROWS > MAX_PREFILL_ROWS && WIDE_PREFILL_ROWS <= 512) };
        assert_eq!(MAX_WINDOW_ROWS, MAX_PREFILL_ROWS, "the wide decode window is not widened by the prefill switch");
    }

    #[test]
    fn options_name_the_layers_to_load_and_refuse_sizes_the_model_cannot_take() {
        let c = cfg();
        assert_eq!(check_options(&opts(), &c).unwrap(), 88);
        assert_eq!(check_options(&ModelOptions { layers: Some(8), ..opts() }, &c).unwrap(), 8);
        for bad in [
            ModelOptions { layers: Some(0), ..opts() },
            ModelOptions { layers: Some(89), ..opts() },
            ModelOptions { slots: 0, ..opts() },
            ModelOptions { kv_pages: 0, ..opts() },
            ModelOptions { max_context: 0, ..opts() },
            ModelOptions { prefill_rows: 0, ..opts() },
            // the contract's cap (P2, the default since rc2d: 4,096; P1: 512 or 256)
            ModelOptions { prefill_rows: super::super::contract::current().max_prefill_rows(prefill_rows_cap()) + 1, ..opts() },
            // 32 pages of 256 hold 8192 tokens
            ModelOptions { max_context: 8193, ..opts() },
            ModelOptions { max_context: 1 << 21, kv_pages: 1 << 14, ..opts() },
        ] {
            assert!(matches!(check_options(&bad, &c), Err(ModelError::Config(_))), "{bad:?} should be refused");
        }
        // P1's default cap is 512 rows: a P1 process without the opt-out takes 512 and refuses 513; under P2 (the default since rc2d) the
        // cap is 4,096 whatever the P1 field says
        if std::env::var(PREFILL_ROWS_VAR).is_err() {
            assert_eq!(prefill_rows_cap(), WIDE_PREFILL_ROWS);
            let cap = match super::super::contract::current() {
                super::super::contract::PrefillContract::P1 => WIDE_PREFILL_ROWS,
                super::super::contract::PrefillContract::P2 => super::super::contract::P2_MAX_PREFILL_ROWS,
            };
            assert_eq!(check_options(&ModelOptions { prefill_rows: cap, ..opts() }, &c).unwrap(), 88);
            assert!(check_options(&ModelOptions { prefill_rows: cap + 1, ..opts() }, &c).is_err());
        }
        // the whole 1M context fits when the pool holds it
        let million = ModelOptions { max_context: 1 << 20, kv_pages: 4096, ..opts() };
        assert_eq!(check_options(&million, &c).unwrap(), 88);
    }

    #[test]
    fn the_chunk_bucket_is_a_power_of_two_capped_by_the_scratch_and_never_below_the_rows_chunks() {
        for (real, cap, want) in
            [(0usize, 4usize, 1usize), (1, 4, 1), (2, 4, 2), (3, 4, 4), (4, 4, 4), (5, 16, 8), (9, 16, 16), (3, 3, 3), (1, 1, 1)]
        {
            assert_eq!(chunk_bucket(real, cap), want, "real {real}, cap {cap}");
        }
        // every real count up to the capacity is covered, and a graph serves a whole bucket
        for cap in [1usize, 2, 3, 4, 7, 16, 2048] {
            for real in 1..=cap {
                let b = chunk_bucket(real, cap);
                assert!(b >= real && b <= cap, "real {real}, cap {cap}: bucket {b}");
            }
        }
        // 16 buckets span the whole 1M context (2048 chunks): 1, 2, 4, ... 2048
        let buckets: std::collections::BTreeSet<usize> = (1..=2048).map(|r| chunk_bucket(r, 2048)).collect();
        assert_eq!(buckets.len(), 12);
    }

    #[test]
    fn a_graph_key_is_every_host_value_that_reaches_a_launcher() {
        let k = GraphKey { rows: 3, n_seq: 1, count: 28, chunks: 2, fused: false };
        for other in [
            GraphKey { rows: 4, ..k },
            GraphKey { n_seq: 2, ..k },
            GraphKey { count: 8, ..k },
            GraphKey { chunks: 4, ..k },
            GraphKey { fused: true, ..k },
        ] {
            assert_ne!(k, other);
        }
    }
}
