//! The MiniCPM-V-4.7 text model on the device: the serial forward (prefill chunks, decode and verify windows) over the weights and the
//! state the boot allocated (E3 of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`).
//!
//! The model is a copy-adapt of `qwen_flash_next::model` (design decision 5.2a of the plan: no shared layer runner yet) cut down to
//! what this backbone is: a plain residual stream (no hyper-connections, no PLE, no sparse-attention indexer, no MTP head) of 40 layers
//! that are each a mixer (30 gated delta net layers and 10 full-attention layers at 3, 7, ..., 39) followed by a mixture of experts,
//! in the dataflow Qwen3.8's `forward.rs` runs for its plain-residual layers:
//!
//! ```text
//! x = embed(ids) [+ spliced feature rows];  h = rmsnorm(x; layer 0 input norm)
//! per layer:  a = mixer(h);   r = x + a;  h = rmsnorm(r; post-attention norm)          (one fused add_rmsnorm)
//!             a = moe(h);     x = r + a;  h = rmsnorm(x; next layer's input norm)      (one fused add_rmsnorm; the final norm after 39)
//! logits = lm_head(h)
//! ```
//!
//! **Numerics.** The residual stream and every activation is bf16, the norms' statistics are fp32, every weight norm is `(1 + w)` except
//! the gated delta net's own `linear_attn.norm` (a plain gated RMSNorm whose gate is SiLU: `RMSNormGated` of `Qwen3_5Moe`; Flash-Next's
//! is a sigmoid gate, which is why `fn_glue::gated_rmsnorm_sigmoid` is NOT used here). The router and the shared expert's gate are
//! per-row fp16-weight dot products, the routed experts run through FrostMoE (`fmoe::forward`, `boot::KC` K chunks: `fmoe::DEFAULT_KC`
//! is invalid at hidden 2,048) in windows and through FrostMoE's sorted-token prefill kernels (`fmoe::prefill`, one arithmetic for every
//! prefill chunk of either path: E6a) in prefill; `REXL3_MINICPM_PREFILL_MOE=coopmk` runs E3's `moe_topk_softmax` + CoopMK instead (the
//! oracle arm: `docs/MINICPMV4_7_INTEGRATION_PLAN.md` section 13).
//!
//! **Bits rules** (docs/PREFIX_CACHE.md, the chunk-invariance audit, as Flash-Next): a prefill chunk's arithmetic depends only on the
//! prompt's TOTAL length through [`PrefillPath`] (the decode GEMV up to 128 tokens, the pinned GEMM for every chunk of a longer
//! prompt, a 1-row tail included), never on the chunk's own row count: every EXL3 linear takes the path's arm; the router and the shared
//! gate are the per-row `small_linear_f16w` (a prefill chunk runs its many-row twin, the same bits); the routed experts are
//! `fmoe::prefill` (a token's bits are a function of its own row, its
//! routing and the weights alone: its K order is fixed, its epilogues are per token and its combine is in top-k order, so no chunk grid
//! and no company of other rows moves a bit; it is the same arithmetic on both prefill paths); the gated delta net is
//! `gdn::prefill`, attention the native `attn_prefill_rows`. Windows have their own arithmetic (FrostMoE, the GDN record + window kernels,
//! the split-K decode attention), are row invariant, and are never cached.
//!
//! **Per-slot state a sequence leaves behind** (what a prefix-cache snapshot of E6 must hold, nothing else may differ): the KV pages of
//! the 10 attention layers through one page table ([`MiniCpmModel::kv_tab`]), per GDN layer the delta-rule state and the convolution
//! history ([`GdnPools`]), and the slot's rope delta ([`MiniCpmModel::rope_delta`]).
//!
//! **The head and the pad.** The head writes fp16 logits (11 bits of mantissa: bf16's 1/8 spacing at |logit| in [16, 32) is the
//! larger error against a reference; `REXL3_MINICPM_LOGITS=bf16` takes the family-wide bf16 output) and is 248,192 wide for a
//! vocabulary of 248,144: every logits consumer here reads the first [`MiniCpmModel::vocab_size`] columns only (the top-k's
//! `(stride, valid)`, [`MiniCpmModel::last_rows_logits`]); a poison seam ([`MiniCpmModel::set_pad_poison`]) overwrites the 48 pad columns
//! with the largest finite value of the head's dtype so that a test can prove it.
//!
//! **Seams left for later slices** (nothing of them is wired): positions are `[rows, 3]` M-RoPE triples (text rows replicate the
//! position plus the slot's rope delta, which is the same bits as the 1-D rotation, and a batch may carry explicit triples in
//! `RowBatch::rope3`); the embedding step takes [`SpliceOp`]s (E5b's image feature rows); a layer's output passes through
//! [`MiniCpmModel::after_layer`] (E4's tap copies); `Backend::has_mtp` is false and no draft path exists.

use super::config::MiniCpmConfig;
use super::device::{MiniCpmLayer, MiniCpmWeights, MixerWeights};
use super::spec::{SpecAsk, SpecProposal, SpecRig};
use super::state::{GdnPools, ModelState};
use crate::error::ModelError;
use crate::forward::{Candidates, Mode, PrefillPath, SpliceOp};
use crate::kvtables::{KvTables, BLOCK_TOKENS};
use crate::rows::RowBatch;
use gb_cuda::attention::{self, BlockTable, PagedKv, RowIndex};
use gb_cuda::exl3_linear::{Dtype, Exl3Weights, MAX_DECODE_ROWS};
use gb_cuda::gdn::{self, Batch, PreInputs, PreWeights, Record};
use gb_cuda::glue::AttnPrepCfg;
use gb_cuda::graph::{CaptureError, Graph};
use gb_cuda::{fmoe, fn_moe, glue, DevBuf, Stream};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Rows of the widest decode / verify window (four streams of 16 rows).
pub const WINDOW_ROWS: usize = super::boot::WINDOW_ROWS;
/// Candidates per row the LM head's top-k can return.
pub const MAX_CANDIDATES: usize = 64;
/// The epsilon every norm of this model runs with; a pack that says otherwise is refused at load.
const EPS: f32 = 1e-6;
/// Bits of the largest finite bf16, the value the poison seam writes into the pad columns.
const POISON_BF16: u16 = 0x7F7F;
/// Bits of the largest finite fp16.
const POISON_F16: u16 = 0x7BFF;

const _: () = assert!(WINDOW_ROWS <= MAX_DECODE_ROWS);

/// Sizes the model is built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelOptions {
    pub slots: usize,
    /// KV pages (256 tokens each) shared by every slot.
    pub kv_pages: usize,
    pub max_context: usize,
    /// Largest prefill chunk (rows).
    pub prefill_rows: usize,
}

/// The attention grid a captured graph was sized for: a power-of-two bucket of 512-token chunks covering the live context (the kernel
/// skips chunks past each row's position, so one graph stays exact while the context grows inside its bucket).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AttnGrid(usize);

impl AttnGrid {
    pub fn cover(live: usize, max_context: usize) -> Self {
        Self((live.div_ceil(512).next_power_of_two() * 512).min(max_context).max(live))
    }
    pub fn ctx(self) -> usize {
        self.0
    }
    pub fn covers(self, live: usize) -> bool {
        live <= self.0
    }
}

/// The shape a captured window graph was recorded for; replaying it on any other shape would be wrong. Sequence boundaries matter
/// even at equal row and sequence counts (the GDN kernels receive per-sequence grids), and so does the pad poison seam.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WindowKey {
    rows: usize,
    seqs: usize,
    seq_ends: u64,
    candidates: usize,
    poison: bool,
    grid: AttnGrid,
}

impl WindowKey {
    pub fn for_batch(b: &RowBatch, candidates: usize, grid: AttnGrid, poison: bool) -> Self {
        assert!(b.rows <= WINDOW_ROWS && WINDOW_ROWS <= u64::BITS as usize);
        let seq_ends = b.seq_start.iter().skip(1).fold(0, |bits, &end| bits | (1u64 << (end as usize - 1)));
        Self { rows: b.rows, seqs: b.n_seq(), seq_ends, candidates, poison, grid }
    }
}

/// Where the device bytes of a model go, by category (the plan == allocations accounting of the engine's GPU test).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelBytes {
    /// The activation buffers one step's rows use, the index arrays, the logits and the candidate arrays.
    pub scratch: u64,
    /// The EXL3 decode and pinned-prefill scratch, the GDN prefill workspace, the attention partials, FrostMoE's and CoopMK's
    /// workspaces and the top-k's.
    pub workspaces: u64,
    /// The window's record set and raw qkv rows.
    pub window_records: u64,
}

impl ModelBytes {
    /// What the engine reserve must hold beyond the slots' state and the KV pages.
    pub fn engine_reserve(&self) -> u64 {
        self.scratch + self.workspaces + self.window_records
    }
}

/// A stage-by-stage GPU timer over CUDA events (`set_profile`): the stream is never synchronised between stages, so what it measures is
/// the GPU's time from one stage's end to the next's, launch gaps of the eager path included (a replayed graph has none).
struct Prof {
    /// Events created so far (reused by every forward; a profile is a debugging tool and never frees them).
    pool: Vec<*mut core::ffi::c_void>,
    /// The stage each recorded event closes, in order; event 0 is the start.
    marks: Vec<&'static str>,
    totals: BTreeMap<&'static str, (f64, u64)>,
}

impl Prof {
    fn event(&mut self, i: usize) -> Result<*mut core::ffi::c_void, ModelError> {
        while self.pool.len() <= i {
            let mut e = std::ptr::null_mut();
            // SAFETY: `e` is valid for a pointer write.
            let rc = unsafe { gb_cuda::rt::timed_events::create(&mut e) };
            gb_cuda::CudaError::check(rc)?;
            self.pool.push(e);
        }
        Ok(self.pool[i])
    }

    fn record(&mut self, name: &'static str, st: &Stream) -> Result<(), ModelError> {
        let e = self.event(self.marks.len())?;
        // SAFETY: a live event and the model's stream.
        let rc = unsafe { gb_cuda::rt::timed_events::record(e, st.raw()) };
        gb_cuda::CudaError::check(rc)?;
        self.marks.push(name);
        Ok(())
    }

    /// Wait for the last event and add each stage's time to its name's total.
    fn finish(&mut self) -> Result<(), ModelError> {
        if self.marks.len() < 2 {
            self.marks.clear();
            return Ok(());
        }
        let last = self.pool[self.marks.len() - 1];
        // SAFETY: a recorded event.
        gb_cuda::CudaError::check(unsafe { gb_cuda::rt::timed_events::synchronize(last) })?;
        for i in 1..self.marks.len() {
            let mut ms = 0f32;
            // SAFETY: both events were recorded and are complete.
            let rc = unsafe { gb_cuda::rt::timed_events::elapsed_ms(&mut ms, self.pool[i - 1], self.pool[i]) };
            gb_cuda::CudaError::check(rc)?;
            let t = self.totals.entry(self.marks[i]).or_insert((0.0, 0));
            t.0 += ms as f64;
            t.1 += 1;
        }
        self.marks.clear();
        Ok(())
    }
}

/// Record a stage boundary when the profile is on (a free function so that it borrows only the profiler and the stream).
fn mark(prof: &mut Option<Prof>, st: &Stream, name: &'static str) -> Result<(), ModelError> {
    match prof {
        Some(p) => p.record(name, st),
        None => Ok(()),
    }
}

/// Sizes the model can be built for, checked before anything is allocated.
pub fn check_opts(opts: &ModelOptions) -> Result<(), ModelError> {
    let ModelOptions { slots, kv_pages, max_context, prefill_rows } = *opts;
    if slots == 0 || kv_pages == 0 || max_context == 0 || prefill_rows == 0 {
        return Err(ModelError::Config(format!("{opts:?}: every size must be positive")));
    }
    let cap = gdn::PREFILL_MAX_ROWS.min(attention::MAX_PREFILL_ROWS);
    if prefill_rows > cap {
        return Err(ModelError::Config(format!("a prefill chunk of {prefill_rows} rows is past what the kernels take ({cap})")));
    }
    if max_context > kv_pages * BLOCK_TOKENS {
        return Err(ModelError::Config(format!(
            "a context of {max_context} tokens needs more than the {kv_pages} pages of {BLOCK_TOKENS} the pool holds"
        )));
    }
    Ok(())
}

/// One EXL3 linear of a batch: the decode GEMV for a window and for the prefill of a short prompt, the pinned fixed-algorithm GEMM
/// for every chunk of a longer one. The choice is the mode's, never `m`'s (see [`PrefillPath`]).
#[allow(clippy::too_many_arguments)]
fn lin(
    wt: &Exl3Weights,
    x: &DevBuf,
    y: &mut DevBuf,
    m: usize,
    mode: Mode,
    ws: &mut DevBuf,
    pws: &mut DevBuf,
    st: &Stream,
) -> Result<(), ModelError> {
    match mode {
        Mode::Prefill(PrefillPath::Tiled) => wt.forward_prefill_pinned(x, y, m, Dtype::BF16, pws, st)?,
        Mode::Window | Mode::Prefill(PrefillPath::Gemv) => wt.forward(x, y, m, Dtype::BF16, Dtype::BF16, ws, st)?,
    }
    Ok(())
}

/// The linears a batch pushes through `Exl3Weights` (the routed experts go through the MoE kernels' pointer tables instead).
fn trunk_linears(w: &MiniCpmWeights) -> Vec<&Exl3Weights> {
    let mut v = Vec::new();
    for layer in &w.layers {
        match &layer.mixer {
            MixerWeights::Linear(m) => v.extend([&m.in_proj_qkv, &m.in_proj_z, &m.out_proj]),
            MixerWeights::Full(m) => v.extend([&m.q_proj, &m.k_proj, &m.v_proj, &m.o_proj]),
        }
    }
    v
}

/// The (k, n) shapes a Tiled prefill chunk can dispatch, once each.
fn pinned_shapes(w: &MiniCpmWeights) -> Vec<(usize, usize)> {
    let mut s: Vec<(usize, usize)> = trunk_linears(w).into_iter().map(|x| (x.k, x.n)).collect();
    s.sort_unstable();
    s.dedup();
    s
}

/// Every row count a Tiled prefill can dispatch must run on this cuBLAS: a chunk of any length 1..=`rows` (a one-row tail included)
/// takes the pinned GEMM, and a cuBLAS without the algorithm must stop the boot, not change a prompt's bits.
fn init_pinned_prefill(w: &MiniCpmWeights, rows: usize) -> Result<(), ModelError> {
    gb_cuda::exl3_linear::prefill_init()?;
    for (k, n) in pinned_shapes(w) {
        for m in 1..=rows {
            gb_cuda::exl3_linear::prefill_pin_check(m, k, n).map_err(|e| {
                ModelError::Config(format!(
                    "the pinned prefill GEMM ({}) does not run a {m} x {k} x {n} problem here ({e}); row-invariant prefill cannot be \
                     guaranteed on this cuBLAS",
                    gb_cuda::exl3_linear::prefill_pin_name()
                ))
            })?;
        }
    }
    Ok(())
}

/// The pinned-prefill scratch a chunk of `rows` rows needs: the widest of every trunk linear.
fn prefill_scratch_need(w: &MiniCpmWeights, rows: usize) -> usize {
    trunk_linears(w).into_iter().map(|x| x.prefill_scratch_bytes(rows)).max().unwrap_or(0).max(1 << 20)
}

/// The decode workspace: the widest of every linear at the decode kernel's row limit, and of the LM head at a window.
fn decode_scratch_need(w: &MiniCpmWeights, window_rows: usize) -> usize {
    let mut need = 1 << 20;
    for x in trunk_linears(w) {
        need = need.max(x.workspace_bytes(MAX_DECODE_ROWS));
    }
    need.max(w.lm_head.workspace_bytes(window_rows)).max(w.lm_head.workspace_bytes(1))
}

// --- thin safe wrappers over the raw-pointer glue launchers (every argument is a whole device buffer) ---

fn g_embed(
    table: &DevBuf,
    ids: &DevBuf,
    out: &DevBuf,
    rows: usize,
    dim: usize,
    vocab: usize,
    st: &Stream,
) -> Result<(), ModelError> {
    // SAFETY: bf16 table `[vocab, dim]`, i32 ids `[rows]`, bf16 out `[rows, dim]`, all checked by the caller's sizing.
    unsafe { glue::embed(table.ptr().cast_const(), ids.ptr().cast_const(), out.ptr(), rows, dim, vocab, st) }?;
    Ok(())
}

/// The per-row fp16-weight linear. `many` takes the many-row kernel of a prefill chunk (`small_linear_f16w_rows`: the same bits as the
/// per-row kernel, tests/glue.rs, weights held in registers across a slice of rows); windows keep the per-row kernel (graph-captured,
/// at most 64 rows).
#[allow(clippy::too_many_arguments)]
fn g_small_linear(
    many: bool,
    x: &DevBuf,
    w: &DevBuf,
    y: &DevBuf,
    rows: usize,
    k: usize,
    n: usize,
    st: &Stream,
) -> Result<(), ModelError> {
    // SAFETY: bf16 x `[rows, k]`, f16 w `[n, k]`, bf16 y `[rows, n]`.
    unsafe {
        if many {
            glue::small_linear_f16w_rows(x.ptr().cast_const(), w.ptr().cast_const(), y.ptr(), rows, k, n, st)
        } else {
            glue::small_linear_f16w(x.ptr().cast_const(), w.ptr().cast_const(), y.ptr(), rows, k, n, st)
        }
    }?;
    Ok(())
}

/// `out = out * sigmoid(gate)` in place over `n` elements.
fn g_gate_mul(out: &DevBuf, gate: &DevBuf, n: usize, st: &Stream) -> Result<(), ModelError> {
    let p = out.ptr();
    // SAFETY: the kernel reads every element before it writes it, by the same thread, so in place is allowed.
    unsafe { glue::gate_mul(p.cast_const(), gate.ptr().cast_const(), p, n, st) }?;
    Ok(())
}

/// `y = rmsnorm(x) * w` per row (bf16 in and out, fp32 weight with the `+1` folded in).
fn g_rmsnorm(x: &DevBuf, w: &DevBuf, y: &DevBuf, rows: usize, dim: usize, st: &Stream) -> Result<(), ModelError> {
    // SAFETY: bf16 `[rows, dim]` x and y, fp32 `[dim]` w.
    unsafe { glue::rmsnorm(x.ptr().cast_const(), w.ptr().cast_const(), y.ptr(), rows, dim, EPS, st) }?;
    Ok(())
}

/// `r = bf16(x + a)` and `h = rmsnorm(r) * w`; `r` and `h` are distinct buffers, `x` and `a` are read.
#[allow(clippy::too_many_arguments)]
fn g_add_rmsnorm(
    x: &DevBuf,
    a: &DevBuf,
    w: &DevBuf,
    r: &DevBuf,
    h: &DevBuf,
    rows: usize,
    dim: usize,
    st: &Stream,
) -> Result<(), ModelError> {
    // SAFETY: bf16 `[rows, dim]` x, a, r and h; fp32 `[dim]` w; r and h distinct (the kernel allows y to alias x or res, not h_out).
    unsafe {
        glue::add_rmsnorm(x.ptr().cast_const(), a.ptr().cast_const(), w.ptr().cast_const(), r.ptr(), h.ptr(), rows, dim, EPS, st)
    }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn g_topk_valid(
    logits: &DevBuf,
    rows: usize,
    stride: usize,
    valid: usize,
    count: usize,
    ids: &DevBuf,
    vals: &DevBuf,
    ws: &DevBuf,
    st: &Stream,
) -> Result<(), ModelError> {
    // SAFETY: bf16 logits `[rows, stride]`, i32 ids and f32 vals `[rows, count]`, workspace of `topk_valid_workspace_bytes`.
    unsafe {
        glue::topk_valid(logits.ptr().cast_const(), rows, stride, valid, count, ids.ptr(), vals.ptr(), ws.ptr(), ws.bytes(), st)
    }?;
    Ok(())
}

pub struct MiniCpmModel {
    pub cfg: MiniCpmConfig,
    pub opts: ModelOptions,
    pub stream: Stream,
    /// Paged KV of the 10 attention layers, in layer order (`AttnShape::Q16Kv2`).
    pub kv: Vec<PagedKv>,
    /// The delta-rule state and conv history of the 30 GDN layers, per slot.
    pub pools: GdnPools,
    pub kv_tab: KvTables,
    pub table: BlockTable,
    /// Per slot: rope position minus cache index of every later text token (M-RoPE after an image; zero for text).
    pub rope_delta: Vec<i32>,

    // ---- private: the weights, the scratch and the bookkeeping the forward runs on ----
    /// Shared so a layer's weights can be borrowed while the model runs it (`run_layer` takes `&mut self`).
    w: Arc<MiniCpmWeights>,
    /// Rows the scratch was sized for: `max(WINDOW_ROWS, opts.prefill_rows)`.
    rows: usize,
    rowidx: RowIndex,
    /// Per-row and per-sequence index arrays of the batch being run.
    ids: DevBuf,
    pos3: DevBuf,
    seq_start: DevBuf,
    seq_slot: DevBuf,
    keep: DevBuf,
    /// The residual stream, the post-attention residual, the normed input of the next block and the block output (bf16 `[rows, h]`).
    x: DevBuf,
    r: DevBuf,
    h: DevBuf,
    a: DevBuf,
    moe_out: DevBuf,
    mixer_out: DevBuf,
    router_logits: DevBuf,
    topk_indices: DevBuf,
    topk_weights: DevBuf,
    act_workspace: DevBuf,
    sh_gate_scalar: DevBuf,
    qkv: DevBuf,
    z: DevBuf,
    ga: DevBuf,
    gb: DevBuf,
    fk: DevBuf,
    fv: DevBuf,
    fq: DevBuf,
    fk2: DevBuf,
    fgate: DevBuf,
    /// EXL3 decode workspace, EXL3 pinned-prefill scratch, the GDN prefill workspace, the dense attention partials and FrostMoE's.
    ws: DevBuf,
    prefill_ws: DevBuf,
    gdn_prefill_ws: DevBuf,
    attn_ws: DevBuf,
    fmoe_ws: DevBuf,
    /// FrostMoE's prefill workspace for `rows` rows (a 256-byte placeholder when `grouped_prefill_moe` is off).
    fmoe_pws: DevBuf,
    /// Whether prefill chunks run the sorted-token grouped kernels (`fmoe::prefill`, the default) or E3's CoopMK (the oracle arm,
    /// `REXL3_MINICPM_PREFILL_MOE=coopmk`).
    grouped_prefill_moe: bool,
    /// Per GDN layer, the window's record and the raw qkv rows its commit replays.
    win_qkv: Vec<DevBuf>,
    recs: Vec<Record>,
    logits: DevBuf,
    /// Whether the head writes fp16 logits (else bf16): the top-k kernel and every consumer follow.
    logits_f16: bool,
    topk_ws: DevBuf,
    topk_ids: DevBuf,
    topk_vals: DevBuf,
    /// 48 bf16 of the poison value, read with a row stride of 0 by the poison seam.
    poison_row: DevBuf,
    pad_poison: bool,
    /// Rows of the last prefill chunk or window whose normed final rows are still in `h`.
    last_rows: usize,
    /// Stage timer (`set_profile`), `None` in serving.
    prof: Option<Prof>,
    /// Debug capture of the residual stream after every layer (`set_capture`), as bf16 bits `[rows, hidden]` a layer.
    capture: bool,
    captured: Vec<Vec<u16>>,
    /// The last window, and per slot the rows of it a `commit` keeps until the next forward replays them.
    last_window: Option<RowBatch>,
    pending: BTreeMap<usize, usize>,
    /// Replayable windows, and the context the grid of the one being run covers.
    graphs: HashMap<WindowKey, Graph>,
    attn_ctx: Option<usize>,
    use_graphs: bool,
    /// A page table change made outside a forward that the device table has not seen yet.
    table_dirty: bool,
    bytes: ModelBytes,
    /// The DSpark draft and the tap export it needs (E4b, `spec.rs`); `None` is the text engine of E3, whose bits it leaves alone.
    spec: Option<SpecRig>,
}

/// An on-by-default switch: only `0` turns it off.
fn env_on(key: &str) -> bool {
    std::env::var(key).map(|v| v != "0").unwrap_or(true)
}

impl MiniCpmModel {
    /// The model on the weights and the state a boot produced.
    pub fn new(cfg: MiniCpmConfig, w: MiniCpmWeights, state: ModelState, opts: ModelOptions) -> Result<Self, ModelError> {
        check_opts(&opts)?;
        let ModelOptions { slots, kv_pages, max_context, prefill_rows } = opts;
        if state.snapshots.is_some() {
            return Err(ModelError::Config(
                "the prefix snapshot arena belongs to the backend: take `ModelState::snapshots` before the model is built".into(),
            ));
        }
        if state.slots != slots || state.kv_pages != kv_pages || state.window.rows != WINDOW_ROWS {
            return Err(ModelError::Config(format!(
                "the state was allocated for {} slots, {} pages and {} window rows; the model is asked for {slots}, {kv_pages} and \
                 {WINDOW_ROWS}",
                state.slots, state.kv_pages, state.window.rows
            )));
        }
        if cfg.rms_norm_eps as f32 != EPS {
            return Err(ModelError::Config(format!(
                "rms_norm_eps {} (the MiniCPM-V-4.7 kernels are built for {EPS})",
                cfg.rms_norm_eps
            )));
        }
        let pcfg = Self::prep_cfg(&cfg);
        if pcfg != AttnPrepCfg::MINICPM_TARGET
            && !(cfg.is_qwen3_5_moe_shape()
                && pcfg.heads == 32
                && pcfg.kv_heads == 2
                && pcfg.head_dim == 256
                && pcfg.rotary_dim == 64
                && pcfg.has_gate)
        {
            return Err(ModelError::Config(format!(
                "attention prep {pcfg:?}: the kernels are built for {:?}",
                AttnPrepCfg::MINICPM_TARGET
            )));
        }
        if w.layers.len() != cfg.layers || state.kv.len() != cfg.full_layers() || state.gdn.s.len() != cfg.linear_layers() {
            return Err(ModelError::Config(format!(
                "{} layers of weights, {} KV caches and {} GDN pools for a config of {} layers ({} attention, {} gated delta net)",
                w.layers.len(),
                state.kv.len(),
                state.gdn.s.len(),
                cfg.layers,
                cfg.full_layers(),
                cfg.linear_layers()
            )));
        }
        let (h, top_k, num_experts, interm) = (cfg.hidden, cfg.experts_per_tok, cfg.experts, cfg.moe_intermediate);
        let (q_dim, kv_dim) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
        let rows = WINDOW_ROWS.max(prefill_rows);
        init_pinned_prefill(&w, prefill_rows)?;
        let max_blocks = max_context.div_ceil(BLOCK_TOKENS);
        let (vocab, stride) = (w.vocab, w.lm_head_width);
        if vocab != cfg.vocab || stride != cfg.lm_head_width() {
            return Err(ModelError::Config(format!(
                "the head is {stride} wide with {vocab} ids; the config says {} and {}",
                cfg.lm_head_width(),
                cfg.vocab
            )));
        }
        let pws_bytes = prefill_scratch_need(&w, prefill_rows);
        let ws_bytes = decode_scratch_need(&w, WINDOW_ROWS);
        let one_act = |n: usize| DevBuf::zeroed(rows * n * 2);

        // the state moves in as the boot allocated it
        let ModelState { kv, gdn: pools, window, .. } = state;
        let (recs, win_qkv) = (window.records, window.raw_qkv);

        // the head's output dtype: fp16 logits (11 bits of mantissa: 8 times finer than bf16 at |logit| in [16, 32), where the bf16
        // spacing is 1/8) by default; REXL3_MINICPM_LOGITS=bf16 takes the family-wide bf16 head output instead
        let logits_f16 = std::env::var("REXL3_MINICPM_LOGITS").map_or(true, |v| v != "bf16");
        let grouped_prefill_moe = match std::env::var("REXL3_MINICPM_PREFILL_MOE").as_deref() {
            Err(_) | Ok("grouped") => true,
            Ok("coopmk") => false,
            Ok(other) => {
                return Err(ModelError::Config(format!("REXL3_MINICPM_PREFILL_MOE={other}: expected `grouped` or `coopmk`")));
            }
        };
        let mut poison = DevBuf::zeroed(48 * 2)?;
        poison.upload(&[if logits_f16 { POISON_F16 } else { POISON_BF16 }; 48])?;
        let topk_ws_bytes = glue::topk_valid_workspace_bytes(WINDOW_ROWS, vocab, MAX_CANDIDATES)
            .max(gb_cuda::glm_glue::topk_workspace_bytes(WINDOW_ROWS, vocab, MAX_CANDIDATES));

        let gdn_shape = cfg.gdn_shape();
        let attn_shape = cfg.attn_shape();
        let kc = cfg.kc();
        let mixer_w = q_dim.max(gdn_shape.v_dim());
        let qkv_w = (2 * q_dim).max(gdn_shape.conv_dim());
        // the weights hold device buffers, which are Send but not Sync: this Arc never crosses a thread, it only lets a layer be
        // borrowed while `run_layer` takes the model mutably
        #[allow(clippy::arc_with_non_send_sync)]
        let w = Arc::new(w);
        let mut m = Self {
            stream: Stream::new_nonblocking()?,
            kv,
            pools,
            kv_tab: KvTables::new(kv_pages, slots, max_blocks),
            table: BlockTable::from_host(&vec![Vec::new(); slots], max_blocks)?,
            rope_delta: vec![0; slots],

            rows,
            rowidx: RowIndex { slot: DevBuf::zeroed_view(rows * 4)?, pos: DevBuf::zeroed_view(rows * 4)?, rows },
            ids: DevBuf::zeroed(rows * 4)?,
            pos3: DevBuf::zeroed(rows * 3 * 4)?,
            seq_start: DevBuf::zeroed((slots + 1) * 4)?,
            seq_slot: DevBuf::zeroed(slots * 4)?,
            keep: DevBuf::zeroed((slots + 1) * 4)?,
            x: one_act(h)?,
            r: one_act(h)?,
            h: DevBuf::zeroed_view(rows * h * 2)?,
            a: one_act(h)?,
            moe_out: one_act(h)?,
            mixer_out: DevBuf::zeroed_view(rows * mixer_w * 2)?,
            router_logits: one_act(num_experts)?,
            topk_indices: DevBuf::zeroed(rows * (top_k + 1) * 4)?,
            topk_weights: DevBuf::zeroed(rows * (top_k + 1) * 4)?,
            act_workspace: DevBuf::zeroed(fn_moe::coopmk_workspace_bytes(rows, top_k + 1, interm))?,
            sh_gate_scalar: DevBuf::zeroed(rows * 2)?,
            qkv: one_act(qkv_w)?,
            z: one_act(gdn_shape.v_dim())?,
            ga: one_act(gdn_shape.value_heads())?,
            gb: one_act(gdn_shape.value_heads())?,
            fk: one_act(kv_dim)?,
            fv: one_act(kv_dim)?,
            fq: DevBuf::zeroed_view(rows * q_dim * 2)?,
            fk2: one_act(kv_dim)?,
            fgate: one_act(q_dim)?,
            ws: DevBuf::zeroed(ws_bytes)?,
            prefill_ws: DevBuf::zeroed(pws_bytes)?,
            gdn_prefill_ws: DevBuf::zeroed(gdn::prefill_workspace_bytes_shaped(gdn_shape, rows))?,
            attn_ws: DevBuf::zeroed(attention::decode_workspace_bytes_for(attn_shape, WINDOW_ROWS, max_context))?,
            fmoe_ws: fmoe::workspace(WINDOW_ROWS, top_k, h, interm, kc)?,
            fmoe_pws: if grouped_prefill_moe {
                fmoe::prefill_workspace(rows, top_k, num_experts, h, interm)?
            } else {
                DevBuf::zeroed(256)?
            },
            grouped_prefill_moe,
            win_qkv,
            recs,
            logits: DevBuf::zeroed(WINDOW_ROWS * stride * 2)?,
            topk_ws: DevBuf::zeroed(topk_ws_bytes)?,
            logits_f16,
            topk_ids: DevBuf::zeroed(WINDOW_ROWS * MAX_CANDIDATES * 4)?,
            topk_vals: DevBuf::zeroed(WINDOW_ROWS * MAX_CANDIDATES * 4)?,
            poison_row: poison,
            pad_poison: false,
            prof: None,
            capture: false,
            captured: Vec::new(),
            last_rows: 0,
            last_window: None,
            pending: BTreeMap::new(),
            graphs: HashMap::new(),
            attn_ctx: None,
            use_graphs: env_on("REXL3_GRAPHS"),
            table_dirty: false,
            bytes: ModelBytes { scratch: 0, workspaces: 0, window_records: 0 },
            spec: None,
            w,
            cfg,
            opts,
        };
        m.bytes = m.count_bytes();
        Ok(m)
    }

    fn prep_cfg(cfg: &MiniCpmConfig) -> AttnPrepCfg {
        AttnPrepCfg {
            heads: cfg.heads,
            kv_heads: cfg.kv_heads,
            head_dim: cfg.head_dim,
            rotary_dim: cfg.rotary_dim,
            has_gate: true,
            eps: EPS,
            theta: cfg.rope_theta as f32,
        }
    }

    /// Whether prefill chunks run the grouped prefill MoE (`fmoe::prefill`, the default) or E3's CoopMK (`REXL3_MINICPM_PREFILL_MOE=coopmk`,
    /// the oracle arm). The two arms' bits differ, so a prefix snapshot is only ever restored under the arm that made it.
    pub fn prefill_moe_grouped(&self) -> bool {
        self.grouped_prefill_moe
    }

    /// Real ids of the vocabulary (the candidates' ids are all below it).
    pub fn vocab_size(&self) -> usize {
        self.w.vocab
    }

    /// Whether the head's logits (and so [`MiniCpmModel::last_rows_logits`]) are fp16; bf16 otherwise.
    pub fn logits_are_f16(&self) -> bool {
        self.logits_f16
    }

    /// Width of the head's logits rows (the padded vocabulary).
    pub fn logits_stride(&self) -> usize {
        self.w.lm_head_width
    }

    pub fn has_mtp(&self) -> bool {
        false
    }

    /// Attach a DSpark draft (E4b): from the next forward on, the residual stream after each of the draft's tap layers is exported for
    /// every row the model processes, the rows of the prompt reach the draft's context and so do the accepted rows of the windows.
    /// Slots that hold a sequence now have no draft context for it and do not draft until their next sequence. Window graphs recorded
    /// without the tap copies are dropped.
    pub fn attach_spec(&mut self, rig: SpecRig) -> Result<(), ModelError> {
        if rig.tap_rows() < self.rows {
            return Err(ModelError::Config(format!(
                "the draft's tap buffer holds {} rows, the model runs up to {}",
                rig.tap_rows(),
                self.rows
            )));
        }
        self.flush()?;
        self.stream.sync()?;
        self.graphs.clear();
        self.spec = Some(rig);
        Ok(())
    }

    /// Detach the draft (the model is the text engine again); the window graphs recorded with the tap copies are dropped.
    pub fn detach_spec(&mut self) -> Result<Option<SpecRig>, ModelError> {
        self.flush()?;
        self.stream.sync()?;
        self.graphs.clear();
        Ok(self.spec.take())
    }

    /// The sizes a draft rig is built for on this model (slots, pages, context, tap rows, the head's width and bits).
    pub fn spec_sizes(&self) -> super::spec::SpecSizes {
        super::spec::SpecSizes {
            slots: self.opts.slots,
            kv_pages: self.opts.kv_pages,
            max_context: self.opts.max_context,
            max_blocks: self.opts.max_context.div_ceil(BLOCK_TOKENS),
            tap_rows: self.rows,
            vocab: self.w.vocab,
            head_stride: self.w.lm_head_width,
            head_bits: self.w.lm_head.bits,
            target_layers: self.cfg.layers,
        }
    }

    /// The prefix cache's seam (E6b): after a restore has begun `slot` and installed its pages, declare that the draft's pages `0..len` are
    /// valid because they ARE the snapshot's (shared by reference like the target's, page ids being the same), so the next prefill slice
    /// resumes the draft's context at `len` and the slot drafts. Without this call a restored slot never drafts (`spec.rs`). A model
    /// with no draft ignores it.
    pub fn adopt_draft_context(&mut self, slot: usize, len: usize) -> Result<(), ModelError> {
        match self.spec.as_mut() {
            Some(sp) => sp.adopt_context(slot, len, &self.kv_tab),
            None => Ok(()),
        }
    }

    /// The identity of the attached draft (pack fingerprint, tap layers, page layout: `SpecRig::identity`), `None` without one. The prefix
    /// cache keys its namespace and its draft contexts by it.
    pub fn draft_identity(&self) -> Option<&str> {
        self.spec.as_ref().map(|s| s.identity())
    }

    /// Whether the draft's context of `slot` is exactly the target's committed `len` rows (a snapshot taken now holds a valid one).
    pub fn draft_context_is(&self, slot: usize, len: usize) -> bool {
        self.spec.as_ref().is_some_and(|s| slot < s.book.len() && s.book.committed(slot) == Some(len))
    }

    /// The prefix cache's copy-on-write of a partial page (E6b's `copy_page`): the draft's page of the same id goes with the target's.
    /// A model with no draft ignores it.
    pub fn copy_draft_page(&mut self, src: usize, dst: usize) -> Result<(), ModelError> {
        match self.spec.as_mut() {
            Some(sp) => sp.draft.copy_page(src, dst, &self.stream),
            None => Ok(()),
        }
    }

    pub fn spec(&self) -> Option<&SpecRig> {
        self.spec.as_ref()
    }

    pub fn spec_mut(&mut self) -> Option<&mut SpecRig> {
        self.spec.as_mut()
    }

    /// Whether a draft is attached.
    pub fn has_draft(&self) -> bool {
        self.spec.is_some()
    }

    /// Drafts for the streams of a round (E4b): every pending commit is applied first, so that the draft's rings hold every committed
    /// row; a stream that cannot draft now gets `None`. Only a fatal device error is an `Err`.
    pub fn propose_round(&mut self, asks: &[SpecAsk<'_>]) -> Result<Vec<Option<SpecProposal>>, ModelError> {
        if self.spec.is_none() {
            return Ok(vec![None; asks.len()]);
        }
        self.flush()?;
        let w = Arc::clone(&self.w);
        let head = crate::families::minicpmv4_7::dspark::Head::Exl3 { w: &w.lm_head, valid: w.vocab };
        match self.spec.as_mut() {
            Some(sp) => sp.propose(&w.embed, &head, &self.kv_tab, asks),
            None => Ok(vec![None; asks.len()]),
        }
    }

    /// Whether windows replay captured graphs.
    pub fn graphs_on(&self) -> bool {
        self.use_graphs
    }

    /// Switch graph replay for the windows that follow (a test of the capture path runs both and compares the bits).
    pub fn set_graphs(&mut self, on: bool) {
        self.use_graphs = on;
    }

    /// Captured window graphs held.
    pub fn graph_count(&self) -> usize {
        self.graphs.len()
    }

    /// Forget every captured window graph (the stream is drained first).
    pub fn clear_graphs(&mut self) -> Result<(), ModelError> {
        self.stream.sync()?;
        self.graphs.clear();
        Ok(())
    }

    /// Test seam: overwrite the 48 pad columns of every logits row with the largest finite bf16 before the top-k reads them (and
    /// before [`MiniCpmModel::last_rows_logits`] strips them). With the poison on, a consumer that read the pad would emit an id at or
    /// past [`MiniCpmModel::vocab_size`]. Graphs captured with the other setting are not replayed.
    pub fn set_pad_poison(&mut self, on: bool) {
        self.pad_poison = on;
    }

    /// Set the slot's rope delta (rope position minus cache index of every later text token); E5b's canvas positions after an image.
    pub fn set_rope_delta(&mut self, slot: usize, delta: i32) -> Result<(), ModelError> {
        match self.rope_delta.get_mut(slot) {
            Some(d) => {
                *d = delta;
                Ok(())
            }
            None => Err(ModelError::Request(format!("slot {slot} of {}", self.opts.slots))),
        }
    }

    /// Start a sequence in `slot`: zero its recurrent state, drop its pages and rope delta.
    pub fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        if slot >= self.opts.slots {
            return Err(ModelError::Request(format!("slot {slot} of {}", self.opts.slots)));
        }
        self.flush()?;
        self.pending.remove(&slot);
        self.last_rows = 0;
        self.rope_delta[slot] = 0;
        self.pools.reset_slot(slot, &self.stream)?;
        self.kv_tab.release(slot);
        self.upload_table()?;
        self.stream.sync()?;
        if let Some(sp) = self.spec.as_mut() {
            sp.begin_slot(slot);
        }
        Ok(())
    }

    /// End the sequence in `slot` and give back its pages.
    pub fn end_slot(&mut self, slot: usize) {
        self.pending.remove(&slot);
        if slot < self.opts.slots {
            if let Some(sp) = self.spec.as_mut() {
                sp.end_slot(slot);
            }
            self.kv_tab.release(slot);
            self.rope_delta[slot] = 0;
            self.table_dirty = true;
        }
    }

    /// Whether a request of `n` rows of `slot` starting at position `first` can run: the slot exists, the rows fit the context, and the
    /// pages are there (they are taken now, so a later batch finds them in place; a slot keeps its pages until it ends). An error leaves
    /// every other slot as it was: the host-side half of failure isolation.
    pub fn reserve(&mut self, slot: usize, first: usize, n: usize) -> Result<(), ModelError> {
        if slot >= self.opts.slots {
            return Err(ModelError::Request(format!("slot {slot} of {}", self.opts.slots)));
        }
        if n == 0 {
            return Err(ModelError::Request("a request has no rows".into()));
        }
        let last =
            first.checked_add(n - 1).ok_or_else(|| ModelError::Request(format!("position {first} + {n} rows overflows")))?;
        if last >= self.opts.max_context {
            return Err(ModelError::Request(format!("position {last} is past the context of {}", self.opts.max_context)));
        }
        if (last + 1).div_ceil(BLOCK_TOKENS) > self.kv_tab.blocks_of(slot).len() {
            self.kv_tab.ensure(slot, last + 1).map_err(|e| ModelError::Request(format!("slot {slot}: {e:?}")))?;
            self.table_dirty = true;
        }
        Ok(())
    }

    /// One prefill chunk of ONE sequence (`b`: rows of one slot at consecutive positions). `path` is `PrefillPath::for_prompt(total)` of
    /// the whole prompt, the same for every chunk of it. Candidates of the chunk's last row when `want` is set.
    pub fn prefill_chunk(
        &mut self,
        b: &RowBatch,
        path: PrefillPath,
        want: Option<usize>,
    ) -> Result<Option<Candidates>, ModelError> {
        self.prefill_chunk_spliced(b, path, want, &[])
    }

    /// [`MiniCpmModel::prefill_chunk`] with feature rows spliced over the chunk's embedded rows (E5b's image rows): rows
    /// `src_row..src_row + n` of each op's features replace chunk rows `dst_row..dst_row + n`.
    pub fn prefill_chunk_spliced(
        &mut self,
        b: &RowBatch,
        path: PrefillPath,
        want: Option<usize>,
        splice: &[SpliceOp<'_>],
    ) -> Result<Option<Candidates>, ModelError> {
        if b.n_seq() != 1 {
            return Err(ModelError::Request(format!("a prefill chunk is one sequence, got {}", b.n_seq())));
        }
        if b.rows == 0 || b.rows > self.opts.prefill_rows {
            return Err(ModelError::Request(format!(
                "a prefill chunk takes 1..={} rows, got {}",
                self.opts.prefill_rows, b.rows
            )));
        }
        if path == PrefillPath::Gemv && b.rows > PrefillPath::GEMV_MAX_PROMPT {
            return Err(ModelError::Request(format!(
                "a prefill on the GEMV path takes at most {} rows, got {}",
                PrefillPath::GEMV_MAX_PROMPT,
                b.rows
            )));
        }
        if b.parents.is_some() {
            return Err(ModelError::Request("a MiniCPM-V-4.7 prefill chunk is a plain chain".into()));
        }
        self.forward(b, Mode::Prefill(path), splice)?;
        self.last_rows = b.rows;
        let Some(count) = want else { return Ok(None) };
        self.candidates_of_last_row(b.rows, count)
    }

    /// A decode / verify window over one or more sequences: candidates for every row. The window's recurrent state is committed by
    /// [`MiniCpmModel::commit`] + the next forward (or [`MiniCpmModel::flush`]).
    pub fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        if count == 0 || count > MAX_CANDIDATES.min(self.vocab_size()) {
            return Err(ModelError::Request(format!("candidate count must be 1..={MAX_CANDIDATES}, got {count}")));
        }
        if b.rows == 0 || b.rows > WINDOW_ROWS {
            return Err(ModelError::Request(format!("a window takes 1..={WINDOW_ROWS} rows, got {}", b.rows)));
        }
        if b.parents.is_some() {
            return Err(ModelError::Request("a MiniCPM-V-4.7 window of trees is not built".into()));
        }
        let rows = b.rows;
        self.flush()?;
        self.last_window = None;
        self.last_rows = 0;
        self.prepare_batch(b)?;
        let live = Self::live_of(b);
        let key = WindowKey::for_batch(b, count, AttnGrid::cover(live, self.opts.max_context), self.pad_poison);
        self.run_graph(key, live, |m| m.window_body(b, count))?;
        self.stream.sync()?;
        // the window ran: its record is what a commit replays
        self.last_window = Some(b.clone());
        self.last_rows = rows;
        let mut ids = vec![0i32; rows * count];
        let mut vals = vec![0.0f32; rows * count];
        self.topk_ids.download_range(0, &mut ids)?;
        self.topk_vals.download_range(0, &mut vals)?;
        Ok(Candidates { rows, count, ids, vals })
    }

    /// The raw LM-head logits `[n, vocab]` as the head wrote them (fp16 bits by default, bf16 under `REXL3_MINICPM_LOGITS=bf16`:
    /// [`MiniCpmModel::logits_are_f16`]; the pad columns stripped) of rows `first..first + n` of the last prefill chunk or
    /// window, whose normed trunk rows are still on the device, into `out[..n * vocab]`. At most [`WINDOW_ROWS`] rows a call, so an
    /// all-position parity test reads a long chunk in slices with bounded host memory. Reads no per-slot state, writes none.
    pub fn last_rows_logits(&mut self, first: usize, n: usize, out: &mut [u16]) -> Result<(), ModelError> {
        let (h, vocab, stride) = (self.cfg.hidden, self.vocab_size(), self.logits_stride());
        if n == 0 || n > WINDOW_ROWS || first + n > self.last_rows {
            return Err(ModelError::Request(format!(
                "rows {first}..{} of the last chunk: it left {} rows, and a call reads 1..={WINDOW_ROWS}",
                first + n,
                self.last_rows
            )));
        }
        if out.len() < n * vocab {
            return Err(ModelError::Request(format!("{} logits for {n} rows of {vocab}", out.len())));
        }
        let x = self.h.subview(first * h * 2, n * h * 2);
        let yd = self.head_dtype();
        self.w.lm_head.forward(&x, &mut self.logits, n, Dtype::BF16, yd, &mut self.ws, &self.stream)?;
        self.poison_pad(n)?;
        self.stream.sync()?;
        for r in 0..n {
            self.logits.download_range(r * stride * 2, &mut out[r * vocab..(r + 1) * vocab])?;
        }
        Ok(())
    }

    /// Keep the first `keep` rows of `slot`'s last window.
    pub fn commit(&mut self, slot: usize, keep: usize) {
        self.pending.insert(slot, keep);
    }

    /// Apply pending commits now (before anything reads a slot's state).
    pub fn flush(&mut self) -> Result<(), ModelError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let Some(win) = self.last_window.take() else {
            self.pending.clear();
            return Ok(());
        };
        let keep: Vec<i32> = win
            .seq_slot
            .iter()
            .enumerate()
            .map(|(i, &sl)| self.pending.get(&(sl as usize)).copied().unwrap_or(0).min(win.seq_rows(i).len()) as i32)
            .collect();
        self.pending.clear();
        // the accepted rows' taps go to the draft's rings before the next forward overwrites the tap buffer
        if let Some(sp) = self.spec.as_mut() {
            sp.commit_window(&win, &keep, &self.kv_tab, &self.stream)?;
        }
        // from here the accepted rows are owed to the recurrent state: a replay that fails leaves every later token of every
        // sequence wrong, so it is the engine's fatal error and not the request's
        let mut replay = || -> Result<(), ModelError> {
            self.keep.upload(&keep)?;
            // the window's own index arrays are still on the device: the commit replays exactly its rows
            self.seq_start.upload(&win.seq_start)?;
            self.seq_slot.upload(&win.seq_slot)?;
            let batch = Batch {
                seq_start: &self.seq_start,
                slot: &self.seq_slot,
                n_seq: win.n_seq(),
                rows: win.rows,
                n_slots: self.pools.slots,
            };
            for li in 0..self.recs.len() {
                gdn::commit(
                    &self.stream,
                    &self.recs[li],
                    &self.win_qkv[li],
                    &mut self.pools.s[li],
                    &mut self.pools.conv[li],
                    &batch,
                    &self.keep,
                )?;
            }
            self.stream.sync()?;
            Ok(())
        };
        replay().map_err(|e| {
            ModelError::StateLost(format!("accepted rows of a window were not replayed into the recurrent state: {e}"))
        })
    }

    /// Upload the host page table (`kv_tab`) to the device table after a change made outside a forward.
    pub fn upload_table(&mut self) -> Result<(), ModelError> {
        let tables: Vec<Vec<u32>> = (0..self.opts.slots).map(|s| self.kv_tab.blocks_of(s).to_vec()).collect();
        self.table.upload(&tables)?;
        self.table_dirty = false;
        Ok(())
    }

    /// Device bytes of the engine's working set beyond the weights, the slots' state and the KV pages: for the plan == allocations
    /// test (the boot's reserve holds exactly these).
    pub fn bytes(&self) -> ModelBytes {
        self.bytes
    }

    // ---------------------------------------------------------------- private

    /// Every buffer that is neither a weight, a KV page nor per-slot state, by category.
    fn count_bytes(&self) -> ModelBytes {
        let sum = |bufs: &[&DevBuf]| bufs.iter().map(|b| b.bytes() as u64).sum::<u64>();
        let scratch = sum(&[
            &self.rowidx.slot,
            &self.rowidx.pos,
            &self.ids,
            &self.pos3,
            &self.seq_start,
            &self.seq_slot,
            &self.keep,
            &self.x,
            &self.r,
            &self.h,
            &self.a,
            &self.moe_out,
            &self.mixer_out,
            &self.router_logits,
            &self.topk_indices,
            &self.topk_weights,
            &self.sh_gate_scalar,
            &self.qkv,
            &self.z,
            &self.ga,
            &self.gb,
            &self.fk,
            &self.fv,
            &self.fq,
            &self.fk2,
            &self.fgate,
            &self.logits,
            &self.topk_ids,
            &self.topk_vals,
            &self.poison_row,
            &self.table.buf,
        ]);
        let workspaces = sum(&[
            &self.ws,
            &self.prefill_ws,
            &self.gdn_prefill_ws,
            &self.attn_ws,
            &self.fmoe_ws,
            &self.fmoe_pws,
            &self.act_workspace,
            &self.topk_ws,
        ]);
        let window_records = self.win_qkv.iter().map(|b| b.bytes() as u64).sum::<u64>()
            + self
                .recs
                .iter()
                .map(|r| (r.q.bytes() + r.k.bytes() + r.v.bytes() + r.g_decay.bytes() + r.beta.bytes()) as u64)
                .sum::<u64>();
        ModelBytes { scratch, workspaces, window_records }
    }

    /// The live context a batch needs: one past its highest position.
    fn live_of(b: &RowBatch) -> usize {
        b.positions.iter().map(|&p| p as usize + 1).max().unwrap_or(1)
    }

    fn ensure_pages(&mut self, b: &RowBatch) -> Result<(), ModelError> {
        for i in 0..b.n_seq() {
            let r = b.seq_rows(i);
            if r.is_empty() {
                return Err(ModelError::Request("a sequence of the batch has no rows".into()));
            }
            let (first, n) = (b.positions[r.start], r.len());
            if first < 0 {
                return Err(ModelError::Request(format!("position {first} is negative")));
            }
            self.reserve(b.seq_slot[i] as usize, first as usize, n)?;
        }
        if self.table_dirty {
            self.upload_table()?;
        }
        Ok(())
    }

    /// The rope position triples of a batch: explicit ones when the batch carries them, else the cache index plus the slot's rope
    /// delta on all three streams (which rotates exactly as a 1-D position does).
    fn rope_triples(&self, b: &RowBatch) -> Result<Vec<i32>, ModelError> {
        match &b.rope3 {
            Some(r3) => {
                if r3.len() != 3 * b.rows {
                    return Err(ModelError::Request(format!(
                        "rope3 holds {} positions for {} rows (three a row)",
                        r3.len(),
                        b.rows
                    )));
                }
                Ok(r3.clone())
            }
            None => {
                let mut v = Vec::with_capacity(3 * b.rows);
                for (&p, &sl) in b.positions.iter().zip(&b.row_slot) {
                    let rp = p + self.rope_delta[sl as usize];
                    v.extend([rp, rp, rp]);
                }
                Ok(v)
            }
        }
    }

    fn prepare_batch(&mut self, b: &RowBatch) -> Result<(), ModelError> {
        if b.rows == 0 || b.rows > self.rows {
            return Err(ModelError::Request(format!("a step takes 1..={} rows, got {}", self.rows, b.rows)));
        }
        if b.positions.len() != b.rows || b.ids.len() != b.rows || b.row_slot.len() != b.rows {
            return Err(ModelError::Request("a batch's index arrays do not match its rows".into()));
        }
        let vocab = self.vocab_size() as i32;
        if let Some(&bad) = b.ids.iter().find(|&&t| t < 0 || t >= vocab) {
            return Err(ModelError::Request(format!("token id {bad} is outside the vocabulary of {vocab}")));
        }
        // slots of one launch are distinct (the GDN kernels' contract), and every slot exists
        let mut seen = std::collections::BTreeSet::new();
        for &sl in &b.seq_slot {
            if sl < 0 || sl as usize >= self.opts.slots {
                return Err(ModelError::Request(format!("slot {sl} of {}", self.opts.slots)));
            }
            if !seen.insert(sl) {
                return Err(ModelError::Request(format!("slot {sl} appears twice in one batch")));
            }
        }
        let pos3 = self.rope_triples(b)?;
        self.ensure_pages(b)?;
        self.ids.upload(&b.ids)?;
        self.pos3.upload(&pos3)?;
        self.seq_start.upload(&b.seq_start)?;
        self.seq_slot.upload(&b.seq_slot)?;
        self.rowidx.upload(&b.row_slot, &b.positions)?;
        Ok(())
    }

    fn forward(&mut self, b: &RowBatch, mode: Mode, splice: &[SpliceOp<'_>]) -> Result<(), ModelError> {
        self.flush()?;
        // a failed chunk leaves no window to commit: the next commit is for a window that ran
        self.last_window = None;
        self.last_rows = 0;
        self.prepare_batch(b)?;
        self.compute_layers(b, mode, splice)?;
        self.stream.sync()?;
        if let Some(p) = self.prof.as_mut() {
            p.finish()?;
        }
        // the chunk's tap rows are in the tap buffer: the draft's context takes every row of the prompt (a chunk is one sequence)
        if let Some(sp) = self.spec.as_mut() {
            if b.n_seq() == 1 && b.rows > 0 {
                sp.after_prefill(b.seq_slot[0] as usize, b.positions[0].max(0) as usize, b.rows, &self.kv_tab)?;
            }
        }
        Ok(())
    }

    /// E4's seam: the residual stream after layer `i` is in `x` (bf16 `[rows, hidden]`), and nothing has read it since the layer ran.
    /// The DSpark taps (a runtime list from the draft's config, layers 3, 11, 19, 27, 35) copy their rows out here. The debug capture
    /// ([`MiniCpmModel::set_capture`]) reads the same point.
    fn after_layer(&mut self, i: usize, rows: usize) -> Result<(), ModelError> {
        if let Some(sp) = &self.spec {
            sp.export(i, &self.x, rows, &self.stream)?;
        }
        if self.capture {
            self.stream.sync()?;
            let mut v = vec![0u16; rows * self.cfg.hidden];
            self.x.download_range(0, &mut v)?;
            self.captured.push(v);
        }
        Ok(())
    }

    /// Debug: keep the residual stream after every layer of the forwards that follow (one `Vec` of bf16 bits `[rows, hidden]` a layer
    /// a forward, in layer order), for per-layer comparisons against an oracle. Each layer synchronises the stream: never on in
    /// serving, and window graphs are not captured while it is on.
    pub fn set_capture(&mut self, on: bool) {
        if on {
            self.captured.clear();
        }
        self.capture = on;
    }

    /// Debug: time the stages of the forwards that follow with CUDA events (no host synchronisation between stages). Windows run
    /// eagerly while it is on (a captured graph has no stage boundaries). Read the totals with [`MiniCpmModel::take_profile`].
    pub fn set_profile(&mut self, on: bool) {
        self.prof = on.then(|| Prof { pool: Vec::new(), marks: Vec::new(), totals: BTreeMap::new() });
    }

    /// The stage totals since [`MiniCpmModel::set_profile`] (or the last call): `(stage, milliseconds, calls)`, largest first.
    pub fn take_profile(&mut self) -> Vec<(&'static str, f64, u64)> {
        let Some(p) = self.prof.as_mut() else { return Vec::new() };
        let mut v: Vec<_> = std::mem::take(&mut p.totals).into_iter().map(|(k, (ms, n))| (k, ms, n)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        v
    }

    /// The residual streams captured since the last call (see [`MiniCpmModel::set_capture`]).
    pub fn take_captured(&mut self) -> Vec<Vec<u16>> {
        std::mem::take(&mut self.captured)
    }

    /// One decoder layer on `self.x`: `self.h` holds the layer's normed input on entry, and on exit `x` is the layer's output and `h`
    /// is it normed by `next_norm` (the next layer's input norm, or the final norm after the last layer). `li` indexes the GDN pools and
    /// records of a linear layer, `fi` the KV cache of a full-attention layer.
    #[allow(clippy::too_many_arguments)]
    fn run_layer(
        &mut self,
        layer: &MiniCpmLayer,
        li: usize,
        fi: usize,
        next_norm: &DevBuf,
        b: &RowBatch,
        mode: Mode,
    ) -> Result<(), ModelError> {
        let rows = b.rows;
        let h = self.cfg.hidden;
        let top_k = self.cfg.experts_per_tok;
        let num_experts = self.cfg.experts;
        let interm = self.cfg.moe_intermediate;
        let window = mode == Mode::Window;
        let st = &self.stream;

        // 1. the mixer, from the normed input in `h` into `a`
        match &layer.mixer {
            MixerWeights::Linear(m) => {
                let qkv_buf = if window { &mut self.win_qkv[li] } else { &mut self.qkv };
                lin(&m.in_proj_qkv, &self.h, qkv_buf, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
                lin(&m.in_proj_z, &self.h, &mut self.z, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
                let gdn_shape = self.cfg.gdn_shape();
                g_small_linear(!window, &self.h, &m.in_proj_a, &self.ga, rows, h, gdn_shape.value_heads(), st)?;
                g_small_linear(!window, &self.h, &m.in_proj_b, &self.gb, rows, h, gdn_shape.value_heads(), st)?;
                mark(&mut self.prof, st, "gdn.proj")?;

                let batch =
                    Batch { seq_start: &self.seq_start, slot: &self.seq_slot, n_seq: b.n_seq(), rows, n_slots: self.pools.slots };
                let wts = PreWeights { conv_w: &m.conv_w, a_log: &m.a_log, dt_bias: &m.dt_bias };
                if window {
                    let inp = PreInputs { qkv: &self.win_qkv[li], a: &self.ga, b: &self.gb };
                    gdn::pre(st, &inp, &wts, &self.pools.conv[li], &batch, &mut self.recs[li])?;
                    gdn::window(st, &self.recs[li], &self.pools.s[li], &batch, &mut self.mixer_out)?;
                } else {
                    let inp = PreInputs { qkv: &self.qkv, a: &self.ga, b: &self.gb };
                    gdn::prefill_shaped(
                        gdn_shape,
                        st,
                        &inp,
                        &wts,
                        &mut self.pools.s[li],
                        &mut self.pools.conv[li],
                        &batch,
                        &mut self.gdn_prefill_ws,
                        &mut self.mixer_out,
                    )?;
                }
                mark(&mut self.prof, st, "gdn.core")?;
                // the gated RMSNorm of the delta net: plain weight, SiLU gate (not Flash-Next's sigmoid kernel)
                // SAFETY: bf16 `[rows, HV, 128]` o, z and out (in place), fp32 `[128]` w.
                unsafe {
                    glue::gated_rmsnorm(
                        self.mixer_out.ptr().cast_const(),
                        self.z.ptr().cast_const(),
                        m.norm.ptr().cast_const(),
                        self.mixer_out.ptr(),
                        rows,
                        gdn_shape.value_heads(),
                        gdn_shape.head_dim(),
                        EPS,
                        st,
                    )?;
                }
                lin(&m.out_proj, &self.mixer_out, &mut self.a, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
                mark(&mut self.prof, st, "gdn.norm+out")?;
            }
            MixerWeights::Full(m) => {
                lin(&m.q_proj, &self.h, &mut self.qkv, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
                lin(&m.k_proj, &self.h, &mut self.fk, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
                lin(&m.v_proj, &self.h, &mut self.fv, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
                mark(&mut self.prof, st, "attn.proj")?;
                let pcfg = Self::prep_cfg(&self.cfg);
                // SAFETY: the shapes `AttnPrepCfg` documents, which this model's buffers are sized for (q|gate `[rows, 16, 512]`,
                // k `[rows, 2, 256]`, pos3 i32 `[rows, 3]`, fp32 `[256]` norm weights with the `+1` folded in).
                unsafe {
                    glue::attn_prep_mrope(
                        &pcfg,
                        self.qkv.ptr().cast_const(),
                        self.fk.ptr().cast_const(),
                        self.pos3.ptr().cast_const(),
                        m.q_norm.ptr().cast_const(),
                        m.k_norm.ptr().cast_const(),
                        self.fq.ptr(),
                        self.fgate.ptr(),
                        self.fk2.ptr(),
                        rows,
                        st,
                    )?;
                }
                let kv = &mut self.kv[fi];
                attention::kv_write(&self.fk2, &self.fv, &self.rowidx, &self.table, kv, rows, st)?;
                mark(&mut self.prof, st, "attn.prep+kv")?;
                if window {
                    let live = self.attn_ctx.unwrap_or_else(|| Self::live_of(b));
                    attention::attn_decode(
                        &self.fq,
                        kv,
                        &self.table,
                        &self.rowidx,
                        rows,
                        &mut self.mixer_out,
                        live,
                        &mut self.attn_ws,
                        st,
                    )?;
                } else {
                    attention::attn_prefill_rows(
                        &self.fq,
                        kv,
                        &self.table,
                        b.seq_slot[0] as usize,
                        b.positions[0] as usize,
                        0,
                        rows,
                        &mut self.mixer_out,
                        0,
                        st,
                    )?;
                }
                mark(&mut self.prof, st, "attn.core")?;
                g_gate_mul(&self.mixer_out, &self.fgate, rows * self.cfg.heads * self.cfg.head_dim, st)?;
                lin(&m.o_proj, &self.mixer_out, &mut self.a, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
                mark(&mut self.prof, st, "attn.gate+o")?;
            }
        }

        // 2. r = x + a; h = norm_post(r)
        g_add_rmsnorm(&self.x, &self.a, &layer.post_norm, &self.r, &self.h, rows, h, st)?;
        mark(&mut self.prof, st, "norm.post")?;

        // 3. router and shared-expert gate: per row on every path, so a chunk's routing never depends on its row count
        g_small_linear(!window, &self.h, &layer.moe.router, &self.router_logits, rows, h, num_experts, st)?;
        g_small_linear(!window, &self.h, &layer.moe.shared_gate, &self.sh_gate_scalar, rows, h, 1, st)?;
        mark(&mut self.prof, st, "moe.router")?;

        // 4. the experts. A window groups them across its rows (FrostMoE's decode kernels, `KC` chunks of K); a prefill chunk of either
        // path sorts its (row, expert) entries by expert and runs every expert's tiles once per 64 entries (`fmoe::prefill`, the
        // shared expert entry `num_experts`): a row's bits are its own whatever the chunk holds, so the prompt's chunk grid moves
        // nothing. `REXL3_MINICPM_PREFILL_MOE=coopmk` runs E3's per-row CoopMK instead (the oracle arm).
        let tables = layer.moe.tables();
        if !window && self.grouped_prefill_moe {
            let shape = fmoe::Shape { t: rows, ne: num_experts, top_k, h, i: interm, kc: self.cfg.kc(), max_tokens: self.rows };
            fmoe::prefill(
                &self.h,
                &self.router_logits,
                &self.sh_gate_scalar,
                &tables,
                &mut self.moe_out,
                shape,
                None,
                fmoe::SwigluLimit::default(),
                &mut self.fmoe_pws,
                st,
            )?;
        } else if window {
            let shape = fmoe::Shape { t: rows, ne: num_experts, top_k, h, i: interm, kc: self.cfg.kc(), max_tokens: WINDOW_ROWS };
            fmoe::forward(
                &self.h,
                &self.router_logits,
                &self.sh_gate_scalar,
                &tables,
                &mut self.moe_out,
                shape,
                None,
                fmoe::SwigluLimit::default(),
                &mut self.fmoe_ws,
                st,
            )?;
        } else {
            fn_moe::moe_topk_softmax(
                &self.router_logits,
                Some(&self.sh_gate_scalar),
                &mut self.topk_indices,
                &mut self.topk_weights,
                num_experts,
                top_k,
                rows,
                st,
            )?;
            fn_moe::coopmk_decode(
                &self.h,
                &self.topk_indices,
                &self.topk_weights,
                &tables,
                &mut self.act_workspace,
                &mut self.moe_out,
                h,
                interm,
                top_k + 1,
                rows,
                st,
            )?;
        }

        mark(&mut self.prof, st, "moe.experts")?;

        // 5. x = r + moe; h = norm_next(x)
        g_add_rmsnorm(&self.r, &self.moe_out, next_norm, &self.x, &self.h, rows, h, st)?;
        mark(&mut self.prof, st, "norm.next")?;
        Ok(())
    }

    fn compute_layers(&mut self, b: &RowBatch, mode: Mode, splice: &[SpliceOp<'_>]) -> Result<(), ModelError> {
        let rows = b.rows;
        let h = self.cfg.hidden;
        let w = Arc::clone(&self.w);
        if let Some(p) = self.prof.as_mut() {
            p.marks.clear();
            p.record("start", &self.stream)?;
        }
        g_embed(&w.embed, &self.ids, &self.x, rows, h, self.vocab_size(), &self.stream)?;
        for op in splice {
            if op.dst_row + op.n > rows || op.feats.bytes() < (op.src_row + op.n) * h * 2 {
                return Err(ModelError::Request("a feature splice outside the chunk or the encoder output".into()));
            }
            // SAFETY: bf16 rows of `h` elements, both ranges checked above against their buffers.
            unsafe {
                glue::copy_rows(self.feature_ptr(op), h, self.x.ptr_at(op.dst_row * h * 2), h, op.n, h, &self.stream)?;
            }
        }
        g_rmsnorm(&self.x, &w.layers[0].input_norm, &self.h, rows, h, &self.stream)?;
        mark(&mut self.prof, &self.stream, "embed")?;
        let (mut li, mut fi) = (0usize, 0usize);
        let n = w.layers.len();
        for (i, layer) in w.layers.iter().enumerate() {
            let next = if i + 1 < n { &w.layers[i + 1].input_norm } else { &w.final_norm };
            match layer.mixer {
                MixerWeights::Linear(_) => {
                    self.run_layer(layer, li, 0, next, b, mode)?;
                    li += 1;
                }
                MixerWeights::Full(_) => {
                    self.run_layer(layer, 0, fi, next, b, mode)?;
                    fi += 1;
                }
            }
            self.after_layer(i, rows)?;
        }
        self.last_rows = rows;
        Ok(())
    }

    fn feature_ptr(&self, op: &SpliceOp<'_>) -> *const core::ffi::c_void {
        op.feats.ptr_at(op.src_row * self.cfg.hidden * 2).cast_const()
    }

    /// The dtype the head writes its logits in.
    fn head_dtype(&self) -> Dtype {
        if self.logits_f16 {
            Dtype::F16
        } else {
            Dtype::BF16
        }
    }

    /// Top-`count` of the first `rows` logits rows over the real vocabulary (the pad columns are never read), ids and values in
    /// `topk_ids` / `topk_vals`: value descending, ties to the lower id, in the head's dtype.
    fn topk_of(&mut self, rows: usize, count: usize) -> Result<(), ModelError> {
        let (stride, valid) = (self.logits_stride(), self.vocab_size());
        if self.logits_f16 {
            gb_cuda::glm_glue::topk_f16(
                &self.logits,
                rows,
                stride,
                valid,
                count,
                &mut self.topk_ids,
                &mut self.topk_vals,
                &mut self.topk_ws,
                &self.stream,
            )?;
            Ok(())
        } else {
            g_topk_valid(&self.logits, rows, stride, valid, count, &self.topk_ids, &self.topk_vals, &self.topk_ws, &self.stream)
        }
    }

    /// With the poison seam on, overwrite the pad columns of the first `rows` logits rows.
    fn poison_pad(&self, rows: usize) -> Result<(), ModelError> {
        if !self.pad_poison {
            return Ok(());
        }
        let (valid, stride) = (self.vocab_size(), self.logits_stride());
        // SAFETY: 48 bf16 at row stride 0 into columns `valid..stride` of `rows` rows of the logits buffer.
        unsafe {
            glue::copy_rows(
                self.poison_row.ptr().cast_const(),
                0,
                self.logits.ptr_at(valid * 2),
                stride,
                rows,
                stride - valid,
                &self.stream,
            )?;
        }
        Ok(())
    }

    /// The device work of one window: every layer, the full LM head, top-`count` per row over the real vocabulary. No host sync, no
    /// upload.
    fn window_body(&mut self, b: &RowBatch, count: usize) -> Result<(), ModelError> {
        let rows = b.rows;
        self.compute_layers(b, Mode::Window, &[])?;
        let yd = self.head_dtype();
        self.w.lm_head.forward(&self.h, &mut self.logits, rows, Dtype::BF16, yd, &mut self.ws, &self.stream)?;
        self.poison_pad(rows)?;
        mark(&mut self.prof, &self.stream, "lm_head")?;
        self.topk_of(rows, count)?;
        mark(&mut self.prof, &self.stream, "topk")?;
        if let Some(p) = self.prof.as_mut() {
            p.finish()?;
        }
        Ok(())
    }

    /// The LM head and top-`count` of the last row of the chunk now in `h`, always on exactly one row.
    fn candidates_of_last_row(&mut self, rows: usize, count: usize) -> Result<Option<Candidates>, ModelError> {
        if count == 0 || count > MAX_CANDIDATES.min(self.vocab_size()) {
            return Err(ModelError::Request(format!("candidate count must be 1..={MAX_CANDIDATES}, got {count}")));
        }
        let h = self.cfg.hidden;
        let last = self.h.subview((rows - 1) * h * 2, h * 2);
        let yd = self.head_dtype();
        self.w.lm_head.forward(&last, &mut self.logits, 1, Dtype::BF16, yd, &mut self.ws, &self.stream)?;
        self.poison_pad(1)?;
        self.topk_of(1, count)?;
        self.stream.sync()?;
        let mut ids = vec![0i32; count];
        let mut vals = vec![0.0f32; count];
        self.topk_ids.download_range(0, &mut ids)?;
        self.topk_vals.download_range(0, &mut vals)?;
        Ok(Some(Candidates { rows: 1, count, ids, vals }))
    }

    /// Run `body` for a window of live context `live` under the graph recorded for `key`: captured on first use, then replayed. The
    /// key's attention grid must cover the window, and the grid is what attention launches with while `body` runs, graphs on or off.
    fn run_graph(
        &mut self,
        key: WindowKey,
        live: usize,
        body: impl Fn(&mut Self) -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        if !key.grid.covers(live) {
            return Err(ModelError::Request(format!(
                "graph {key:?} covers {} tokens of context, the window needs {live}",
                key.grid.ctx()
            )));
        }
        self.attn_ctx = Some(key.grid.ctx());
        let r = if self.use_graphs && !self.capture && self.prof.is_none() {
            let mut map = std::mem::take(&mut self.graphs);
            let r = self.capture_and_launch(&mut map, key, body);
            self.graphs = map;
            r
        } else {
            body(self)
        };
        self.attn_ctx = None;
        r
    }

    // not `entry()`: a capture the runtime refuses must leave the map untouched and run the work eagerly instead
    #[allow(clippy::map_entry)]
    fn capture_and_launch(
        &mut self,
        map: &mut HashMap<WindowKey, Graph>,
        key: WindowKey,
        body: impl Fn(&mut Self) -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        if !map.contains_key(&key) {
            let cap = match Graph::begin(&self.stream) {
                Ok(c) => c,
                Err(e) if e.is_fatal() => return Err(ModelError::Cuda(e)),
                Err(e) => {
                    eprintln!("rexl3 level=warn graph capture refused ({e}): running MiniCPM-V-4.7 windows eagerly");
                    self.use_graphs = false;
                    return body(self);
                }
            };
            let ran = body(self);
            match cap.finish(ran) {
                Ok(g) => {
                    map.insert(key, g);
                }
                Err(CaptureError::Closure(e)) => return Err(e),
                Err(e) => {
                    if let Some(f) = e.fatal() {
                        return Err(ModelError::Cuda(gb_cuda::CudaError(f.0)));
                    }
                    eprintln!("rexl3 level=warn {e}: running MiniCPM-V-4.7 windows eagerly");
                    self.use_graphs = false;
                    return body(self);
                }
            }
        }
        map[&key].launch(&self.stream)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gb_engine::RowRequest;

    #[test]
    fn attention_grid_buckets_cover_their_window_and_nothing_larger() {
        // power-of-two buckets of 512-token chunks, capped at the context and never below the live window
        assert_eq!(AttnGrid::cover(1, 8192).ctx(), 512);
        assert_eq!(AttnGrid::cover(512, 8192).ctx(), 512);
        assert_eq!(AttnGrid::cover(513, 8192).ctx(), 1024);
        assert_eq!(AttnGrid::cover(1537, 8192).ctx(), 2048);
        assert_eq!(AttnGrid::cover(5000, 6000).ctx(), 6000);
        assert_eq!(AttnGrid::cover(7000, 6000).ctx(), 7000);
        let g = AttnGrid::cover(700, 8192);
        assert!(g.covers(700) && g.covers(1024) && !g.covers(1025));
        // the grid is part of the key: the same window shape in another bucket is another graph
        let k = |live| {
            let b = RowBatch::from_requests(&[RowRequest { slot: 0, first_position: 100, tokens: vec![1; 4] }]);
            WindowKey::for_batch(&b, 1, AttnGrid::cover(live, 8192), false)
        };
        assert_eq!(k(600), k(1000));
        assert_ne!(k(1000), k(1100));
    }

    #[test]
    fn graph_keys_cover_every_host_dispatch_decision() {
        let b = |first, sizes: &[usize]| {
            RowBatch::from_requests(
                &sizes
                    .iter()
                    .enumerate()
                    .map(|(slot, &n)| RowRequest { slot, first_position: first, tokens: vec![1; n] })
                    .collect::<Vec<_>>(),
            )
        };
        let g = AttnGrid::cover(2054, 8192);
        let base = WindowKey::for_batch(&b(2046, &[2, 2]), 2, g, false);
        let cases = [
            ("row count", WindowKey::for_batch(&b(2046, &[2, 2, 1]), 2, g, false)),
            ("sequence count", WindowKey::for_batch(&b(2046, &[4]), 2, g, false)),
            ("rows per sequence / boundaries", WindowKey::for_batch(&b(2046, &[1, 3]), 2, g, false)),
            ("candidate count", WindowKey::for_batch(&b(2046, &[2, 2]), 3, g, false)),
            ("attention grid", WindowKey::for_batch(&b(2046, &[2, 2]), 2, AttnGrid::cover(4100, 8192), false)),
            ("pad poison", WindowKey::for_batch(&b(2046, &[2, 2]), 2, g, true)),
        ];
        for (branch, key) in cases {
            assert_ne!(base, key, "{branch}");
        }
        // positions inside one grid bucket reuse the graph: they are device arrays, not part of the key
        assert_eq!(base, WindowKey::for_batch(&b(2047, &[2, 2]), 2, g, false));
    }

    #[test]
    fn options_the_kernels_cannot_take_are_refused_by_name() {
        let ok = ModelOptions { slots: 2, kv_pages: 64, max_context: 4096, prefill_rows: 1024 };
        assert!(check_opts(&ok).is_ok());
        for (bad, needle) in [
            (ModelOptions { slots: 0, ..ok }, "every size must be positive"),
            (ModelOptions { prefill_rows: 4097, ..ok }, "4096"),
            (ModelOptions { max_context: 64 * 256 + 1, ..ok }, "the 64 pages of 256"),
        ] {
            let e = check_opts(&bad).unwrap_err().to_string();
            assert!(e.contains(needle), "{e}");
        }
    }

    #[test]
    fn the_prefill_path_follows_the_prompt_total_and_not_the_chunk() {
        assert_eq!(PrefillPath::GEMV_MAX_PROMPT, MAX_DECODE_ROWS);
        for total in [1, 2, 64, 127, 128] {
            assert_eq!(PrefillPath::for_prompt(total), PrefillPath::Gemv, "total {total}");
        }
        for total in [129, 256, 1000, 4096] {
            assert_eq!(PrefillPath::for_prompt(total), PrefillPath::Tiled, "total {total}");
        }
        assert_eq!(PrefillPath::for_prompt(4096), PrefillPath::for_prompt(129));
    }
}
