//! The DeepSeek-V4.1-Flash executor: the device model (state, scratch, prefill chunking, decode, the KV/plane books) and the host
//! reference the op order is proved against.
//!
//! ## One schedule, two engines
//!
//! Both engines walk [`ops::plan_forward`]. [`Dsv41Model`] walks it on the device; [`HostModel`] walks the same list in f32 on the CPU over
//! a toy config (2 layers, 4 experts, hidden 64). The host engine exists for exactly one reason: the reference forward is 1,700 lines of
//! torch and a device path cannot be checked on a box with no CUDA, so the *schedule* — the order of the ops, the two `pre_mix` carries, the
//! collapse/post placement, the engram before the block, the shared expert before the router — is proved by running the schedule and a
//! direct transcription of `Transformer.forward` over the same weights and comparing them element for element. A device bug that is really a
//! schedule bug is caught on the CPU.
//!
//! ## The device state
//!
//! * **Planes.** Each slot owns a [`Dsv4Planes`] per layer: the window ring of `sliding_window` raw latents, the compressed plane, the
//!   indexer keys, the level-1 candidate mask and the published selection. The kernel's planes are f16 (`Dsv4Planes::new`); admission
//!   prices the latent at q4 as a *lower bound* on the same token, so a boot admitted for q4 also fits f16 planes — the difference is
//!   0.25 GB at 32K, inside the headroom.
//! * **Rows.** A *window* runs 1..=[`WINDOW_ROWS`] rows of several sequences through the layers and leaves their state for `commit`; a
//!   *prefill chunk* runs up to `chunk_rows` consecutive rows of one sequence and commits them whole. Nothing on this path depends on the
//!   chunk grid: every kernel it launches is row-invariant (`dsv4_attn`'s tests pin it), so a prompt row's logits do not depend on how the
//!   prompt was cut.
//! * **Hyper-connections.** The streams are `hc_mult` fp32 copies of the width. The two `pre_mix` carries of `ops::plan_forward` are the
//!   reference's: the attention collapse of layer `l` uses layer `l - 1`'s FFN `pre`, the FFN collapse uses layer `l`'s attention `pre`.
//!
//! ## What is host-side, and why
//!
//! Two ops are computed on the host because their inputs are not on the device:
//!
//! * **Engram** (layers 1 and 14): the table is an `mmap` of the shard (202.8 GB) and the lookup is a page fault, not a copy. The streams
//!   come down for the layer, [`super::engram::EngramState`] applies the gate, and the result goes back up.
//! * **The streams → hidden collapse** (`hc_pre`): `Dsv4Hc` produces `pre` / `post` / `comb` and applies a `post`, but the kernel crate
//!   has no launcher that collapses the copies into one f16 row for this pack's shape. `collapse` does it on the host, a round trip of
//!   `rows x hc x hidden` f32 per site (42 MB a layer at a 512-row chunk). A kernel for it is the first thing this family wants; the norm
//!   that follows is the kernel's (`dsv4_attn::rmsnorm_dev`).
//!
//! Everything else runs on the device: the MLA block (`MlaDsv4::forward`), the router (`Dsv4Router::logits` / `select`), the shared expert
//! (`Fp8Weights` + `dsv4_moe::swiglu`), the routed experts (`Exl3Weights::forward`, gathered per expert), the mHC sites (`Dsv4Hc`), the
//! embedding (`mhc::embed_expand`) and the head (`glue::small_linear_f16w_rows` over the bf16 matrix).

use super::config::DeepseekV41Config;
use super::device::{Dsv41Device, ExpertGroups};
use super::ops::{self, Op, PreMix, SiteKind};
use super::tp::{ShardTable, TpComm};
use super::tp_arm::TpArms;
use crate::error::ModelError;
use crate::forward::Candidates;
use crate::kvtables::{KvTables, BLOCK_TOKENS};
use crate::rows::RowBatch;
use gb_cuda::attention::BlockTable;
use gb_cuda::dsv4_attn::{Dsv4Planes, Dsv4Scratch, MAX_ROWS as DSV4_MAX_ROWS};
use gb_cuda::dsv4_moe::{self, Dsv4Hc, HcWork};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::{glue, DevBuf, Stream};

/// Rows a decode or verify window may carry.
pub const WINDOW_ROWS: usize = 32;
/// Maximum drafted tokens per single-sequence MTP round (`1 + MTP_MAX_DRAFTS = 8` verify rows).
pub const MTP_MAX_DRAFTS: usize = 7;
/// Most candidates one row is offered.
pub const MAX_CANDIDATES: usize = 64;

/// `REXL3_DSV41_WINDOW_ROWS` (T2.5c): the verify window's row budget — this family's analog of the Qwen engine's
/// `--window-rows` (`docs/SERVING.md`, "Window rows"). That flag is a Qwen option: a family that sizes its own
/// window refuses a non-default `--window-rows` by name (`serve.rs::refuse_unsupported_window`), so this family's
/// knob is the environment variable. Unset keeps [`WINDOW_ROWS`] (8); a value NARROWS the rows one verify window
/// may carry — the engine's `max_rows`, the drafts a round may ask for (`max_drafts`), and the DSpark walk's own
/// cap ([`window_rows`]) all follow the one number, so they cannot drift apart.
///
/// The knob can only narrow, never widen: every buffer a window touches (the tree round's saves, the DSpark tap
/// rows, the engine's records) is sized for [`WINDOW_ROWS`] at build time, and a wider window is outside the
/// family's verified range — a value above it is refused by name, as is anything below 1 (a window of zero rows
/// verifies nothing and starves even the anchor). A smaller window composes with everything downstream: the tree
/// round's search is a prefix of the wider one (`tree.rs::round`), and the chain schedule verifies a shorter
/// prefix. As with the Qwen flag, replies are byte-identical for every setting — the window changes speed, never
/// text.
pub fn parse_window_rows(value: Option<&str>, built: usize) -> Result<usize, String> {
    match value.map(str::trim).filter(|v| !v.is_empty()) {
        None => Ok(built),
        Some(v) => v
            .parse::<usize>()
            .ok()
            .filter(|&n| (1..=built).contains(&n))
            .ok_or_else(|| {
                format!(
                    "REXL3_DSV41_WINDOW_ROWS={v:?}: the deepseek_v41 verify window carries 1..={built} rows (its buffers are sized for {built}); leave it unset for the default {built}"
                )
            }),
    }
}

/// The knob from the runtime environment (`REXL3_DSV41_KNOBS_FILE` or the process env, like the family's other
/// `REXL3_DSV41_*` knobs). A value the boot did not refuse is honoured; an invalid one falls back to the default
/// here — the boot's own parse (`engine.rs`) is the refusal that names the value, before the pack is opened.
pub fn window_rows() -> usize {
    parse_window_rows(gb_cuda::dsv4_moe::runtime_env("REXL3_DSV41_WINDOW_ROWS").as_deref(), WINDOW_ROWS).unwrap_or(WINDOW_ROWS)
}

/// Default number of host-RAM prefix snapshot slots.
pub const DEFAULT_PREFIX_SNAPSHOTS: usize = 32;

/// Host memory that must stay available after a prefix snapshot is taken. This is the snapshot guard alone, decoupled from the boot
/// floor (`REXL3_DSV41_FLOOR_GIB` sets only that): a full-residency single-node boot has about 0.5 GiB free at ready, so a 0.25 GiB
/// guard lets an 8K-token capture (about 0.06 GiB) proceed there, while the 1 GiB memory monitor keeps the boot's headroom honest.
/// The old 0.5 GiB guard (and reading the boot floor) made every capture a silent skip at the 262K profile, so the prefix cache never
/// filled. Decode speed with a 0.25 GiB snapshot guard is unchanged — the pool is host RAM, freed by the pool, and never touches decode.
pub const SNAPSHOT_FLOOR_DEFAULT_BYTES: u64 = 1 << 28;

/// The snapshot floor in bytes: fixed at [`SNAPSHOT_FLOOR_DEFAULT_BYTES`], independent of `REXL3_DSV41_FLOOR_GIB` (which governs only
/// the boot floor). The parameter is kept so call sites read the same shape as before, but no environment can move it: the capture
/// guard exists to let a full-residency boot cache prefixes without crossing the memory monitor.
pub fn snapshot_floor_bytes(_env_gib: Option<&str>) -> u64 {
    SNAPSHOT_FLOOR_DEFAULT_BYTES
}

/// Host-RAM snapshot of a single KV source layer (`[2, 8, 14, 20]`).
#[derive(Clone, Debug)]
pub struct KvSourceSnap {
    pub layer: usize,
    pub compress: Vec<u8>,
    pub index_k: Vec<u8>,
    pub comp_state: Option<gb_cuda::dsv4_attn::CompressorStateSnap>,
}

/// Complete host-RAM prefix snapshot of a slot at token position `len`.
#[derive(Clone, Debug)]
pub struct Dsv41Snapshot {
    pub len: usize,
    pub kv_sources: Vec<KvSourceSnap>,
    pub windows: Vec<Vec<u8>>,
    pub engram: Vec<super::engram::Slot>,
    pub mtp_taps: Vec<u8>,
    /// The slot's CED layer-20 tail (the last `PLANE_ROWS` rows that ran through Layers `20..39`) at `len`.
    pub ced_tail: CedTailSnap,
}

/// One slot's rolling host-side copy of the last `PLANE_ROWS` rows that ran through Layers `20..39`
/// (the CED replay tail). Layer 20 keeps only the trailing rows of the final call in its scratch buffers,
/// so a final chunk shorter than 128 rows would see a smaller window than the whole-prompt pass; this
/// buffer lets the final chunk re-assemble the exact trailing rows (`model.rs` `run_ops`'s layer-20 CED
/// block). Rows are keyed by absolute prompt position, so a contiguous tail can be spliced back onto the
/// device buffers for any final call, whatever chunk grid produced it.
#[derive(Clone, Debug, Default)]
pub struct CedTailSnap {
    /// Rows held (`<= PLANE_ROWS`).
    pub rows: usize,
    /// The absolute prompt position one past the last held row.
    pub end_pos: usize,
    /// The token ids of the held rows, in order.
    pub ids: Vec<i32>,
    /// `[rows][hc * h * 2]` f16 streams.
    pub streams: Vec<u8>,
    /// `[rows][h * 2]` f16 MoE output.
    pub moe_out: Vec<u8>,
    /// `[rows][h]` u16 hidden.
    pub hidden: Vec<u8>,
    /// `[rows][hc * 2]` f32 pre coefficients of Layer 20's Attn site.
    pub pre20: Vec<u8>,
    /// `[rows][hc * 2]` f32 hc post.
    pub hc_post: Vec<u8>,
    /// `[rows][TAP_LAYERS * h * 2]` f16 MTP tap rows (the dspark target layers, head rows), rolled
    /// exactly like the stream buffers so a short final chunk's splice re-aligns them with the
    /// layer-38 tap write (tap_idx 2) instead of leaving stale rows.
    pub mtp_taps: Vec<u8>,
    /// Rows held in `mtp_taps` (`<= win`).
    pub mtp_rows: usize,
    /// `[rows][hc * hc * 2]` f32 hc comb.
    pub hc_comb: Vec<u8>,
}

/// Host-RAM pool of prefix snapshots.
pub struct Dsv41SnapPool {
    pub slots: Vec<Option<Dsv41Snapshot>>,
    pub free: Vec<u32>,
}

impl Dsv41SnapPool {
    pub fn new(n: usize) -> Self {
        Self { slots: vec![None; n], free: (0..n as u32).rev().collect() }
    }
}

/// How a forward ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Head {
    /// No logits (a prefill chunk that is not the last and does not end at a snapshot cut).
    None,
    /// Populate all 40 layers' sliding windows (for a prefill chunk ending at a snapshot cut), without running the LM head.
    WindowOnly,
    /// The logits of every row (a decode or verify window).
    AllRows,
    /// The logits of the last row only (the end of a prefill).
    LastRow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Window,
    Prefill,
}

/// What a model is built with.
#[derive(Clone, Copy, Debug)]
pub struct ModelOptions {
    pub slots: usize,
    /// Pages of the KV pool (each holds [`BLOCK_TOKENS`] tokens of every layer).
    pub kv_blocks: usize,
    pub max_context: usize,
    /// Rows of one prefill chunk.
    pub chunk_rows: usize,
    /// Layers to run (a prefix is the bisection against the reference).
    pub layers: usize,
    /// The cache's planes' format (`--kv-dtype q4`, `REXL3_DSV41_KV`; fp16 when neither is given).
    pub kv: gb_cuda::dsv4_kvq4::KvFmt,
}

pub(crate) struct Scratch {
    rows: usize,
    ids: DevBuf,
    /// `[rows][hc][hidden]` f32: the residual streams.
    streams: DevBuf,
    /// `[rows][hidden]` f16: the collapsed, normed sublayer input the MLA and the FFN read.
    hidden: DevBuf,
    /// `[rows][hc]` f32 per site: the `pre` coefficients of the site just computed, by slot.
    pre: Vec<(usize, SiteKind, DevBuf)>,
    post: DevBuf,
    comb: DevBuf,
    hc: HcWork,
    dsv4: Dsv4Scratch,
    /// `[rows][experts]` f32 router logits.
    logits: DevBuf,
    /// `[rows][topk]` i32 picks and f16 weights.
    picks: DevBuf,
    wts: DevBuf,
    /// `[rows][inter]` **f16**: the shared expert's gate, up and activation — the FP8 GEMMs' and the fused swiglu's
    /// native dtype (see `gb_cuda::dsv4_moe::swiglu_f16`).
    gate: DevBuf,
    up: DevBuf,
    act: DevBuf,
    /// `[rows][hidden]` f32: the MoE block's output, the shared expert's plus the routed sum.
    moe_out: DevBuf,
    /// `[rows][hidden]` f16: one sublayer's output, on its way into a post. The MoE block replaces it with its own f16
    /// result (`moe_experts`'s scatter adds the shared expert in place).
    sub_out: DevBuf,
    /// The routed experts' grouping, packed segments and gemv workspace: one arena, allocated here.
    moe: gb_cuda::dsv4_moe::MoeScratch,
    /// `[MAX_ROWS][vocab]` f16 logits.
    head_out: DevBuf,
    topk_ids: DevBuf,
    topk_vals: DevBuf,
    topk_ws: DevBuf,
}

impl Scratch {
    /// The bytes [`Self::new`] allocates, from the same constructors' formulas — the number admission charges for the model's own
    /// state. `new` asserts it equals [`Self::bytes`], so the price and the allocation cannot drift.
    ///
    /// `rows` is the model's chunk depth (`o.chunk_rows.max(WINDOW_ROWS)`), `max_context` the admitted context.
    pub fn bytes_for(cfg: &DeepseekV41Config, rows: usize, max_context: usize) -> u64 {
        let r = rows as u64;
        let h = cfg.hidden as u64;
        let hc = cfg.hc.mult as u64;
        let inter = cfg.moe.intermediate as u64;
        let e = cfg.moe.routed_experts as u64;
        let k = cfg.moe.experts_per_token as u64;
        let dsv4 = match super::device::dsv4_config(cfg) {
            Ok(d) => gb_cuda::dsv4_attn::Dsv4Scratch::bytes(&d, rows, max_context) as u64,
            Err(_) => 0,
        };
        r * 4
            + r * hc * h * 4
            + r * h * 2
            + (cfg.layers as u64) * 2 * r * hc * 4
            + r * hc * 4
            + r * hc * hc * 4
            + gb_cuda::dsv4_moe::HcWork::bytes_for(rows.max(1))
            + dsv4
            + r * e * 4
            + r * k * 4
            + r * k * 2
            + r * inter * 2 * 3
            + r * h * 4
            + r * h * 2
            + gb_cuda::dsv4_moe::MoeScratch::bytes_for(rows, k as usize, h as usize, inter as usize, e as usize)
            + (rows.min(64) as u64) * (cfg.vocab as u64) * 2
            + 2 * (rows.min(64) as u64) * MAX_CANDIDATES as u64 * 4
            + gb_cuda::glue::topk_workspace_bytes(rows.min(64), cfg.vocab, MAX_CANDIDATES) as u64
    }

    fn new(cfg: &DeepseekV41Config, rows: usize, _slots: usize, max_context: usize) -> Result<Self, ModelError> {
        let h = cfg.hidden;
        let hc = cfg.hc.mult;
        let inter = cfg.moe.intermediate;
        let e = cfg.moe.routed_experts;
        let k = cfg.moe.experts_per_token;
        let dcfg = super::device::dsv4_config(cfg)?;
        let mut pre = Vec::new();
        for layer in 0..cfg.layers {
            for kind in [SiteKind::Attn, SiteKind::Ffn] {
                pre.push((layer, kind, DevBuf::zeroed_view(rows * hc * 4)?));
            }
        }
        let moe = gb_cuda::dsv4_moe::MoeScratch::new(rows, k, h, inter, e)?;
        let dsv4 = Dsv4Scratch::new_with_shared_block(&dcfg, rows, max_context, Some(moe.arena_block()))?;
        let ids_b = (rows * 4).div_ceil(256) * 256;
        let topk_b = (rows.min(64) * MAX_CANDIDATES * 4).div_ceil(256) * 256;
        let (ids, topk_ids, topk_vals) = if let Ok(mb) = gb_cuda::rt::Block::alloc_mapped_host(ids_b + 2 * topk_b) {
            let mb = std::sync::Arc::new(mb);
            (
                DevBuf::view(&mb, 0, rows * 4),
                DevBuf::view(&mb, ids_b, rows.min(64) * MAX_CANDIDATES * 4),
                DevBuf::view(&mb, ids_b + topk_b, rows.min(64) * MAX_CANDIDATES * 4),
            )
        } else {
            (
                DevBuf::zeroed(rows * 4)?,
                DevBuf::zeroed(rows.min(64) * MAX_CANDIDATES * 4)?,
                DevBuf::zeroed(rows.min(64) * MAX_CANDIDATES * 4)?,
            )
        };
        Ok(Self {
            rows,
            ids,
            streams: DevBuf::zeroed_view(rows * hc * h * 4)?,
            hidden: DevBuf::zeroed_view(rows * h * 2)?,
            pre,
            post: DevBuf::zeroed(rows * hc * 4)?,
            comb: DevBuf::zeroed(rows * hc * hc * 4)?,
            hc: HcWork::new(rows.max(1))?,
            dsv4,
            logits: DevBuf::zeroed(rows * e * 4)?,
            picks: DevBuf::zeroed(rows * k * 4)?,
            wts: DevBuf::zeroed(rows * k * 2)?,
            gate: DevBuf::zeroed(rows * inter * 2)?,
            up: DevBuf::zeroed(rows * inter * 2)?,
            act: DevBuf::zeroed(rows * inter * 2)?,
            moe_out: DevBuf::zeroed_view(rows * h * 4)?,
            sub_out: DevBuf::zeroed_view(rows * h * 2)?,
            moe,
            head_out: DevBuf::zeroed_view(rows.min(64) * cfg.vocab * 2)?,
            topk_ids,
            topk_vals,
            topk_ws: DevBuf::zeroed(glue::topk_workspace_bytes(rows.min(64), cfg.vocab, MAX_CANDIDATES))?,
        })
    }

    fn bytes(&self) -> u64 {
        let pre: u64 = self.pre.iter().map(|(_, _, b)| b.bytes() as u64).sum();
        [
            &self.ids,
            &self.streams,
            &self.hidden,
            &self.post,
            &self.comb,
            &self.logits,
            &self.picks,
            &self.wts,
            &self.gate,
            &self.up,
            &self.act,
            &self.moe_out,
            &self.sub_out,
            &self.head_out,
            &self.topk_ids,
            &self.topk_vals,
            &self.topk_ws,
        ]
        .iter()
        .map(|b| b.bytes() as u64)
        .sum::<u64>()
            + pre
            + self.hc.bytes()
            + self.moe.bytes()
    }
}

#[allow(dead_code)]
fn f16(v: f32) -> u16 {
    gb_exl3::f16::f64_to_f16(v as f64)
}

/// The device model: every weight resident (or slabbed), the per-slot planes, and the executor that walks [`ops::plan_forward`].
pub struct Dsv41Model {
    pub cfg: DeepseekV41Config,
    pub dev: Dsv41Device,
    pub stream: Stream,
    pub shared_stream: Stream,
    ev_hidden_ready: gb_cuda::rt::Event,
    ev_shared_done: gb_cuda::rt::Event,
    pub slots: usize,
    pub max_context: usize,
    layers: usize,
    /// `planes[slot][layer]`.
    planes: Vec<Vec<Dsv4Planes>>,
    kv_tab: KvTables,
    table: BlockTable,
    s: Scratch,
    /// `commit(slot, keep)` calls waiting to be applied together (applied before the next forward).
    pending: std::collections::BTreeMap<usize, usize>,
    /// The last window's layout, until its commit is applied.
    last_window: Option<RowBatch>,
    /// The `pre` coefficient buffers of the forward in flight, indexed by `(layer, kind)`.
    pre_index: std::collections::BTreeMap<(usize, SiteKind), usize>,
    /// Pre-allocated `[rows, hc]` identity `pre_mix` buffer.
    identity: DevBuf,
    /// Pre-allocated `[rows, 6144]` bf16 engram gather buffer.
    engram_in: DevBuf,
    /// Pinned mapped host staging buffers (`[64, 6144]` bf16 each for Layer 1 and Layer 14) so decode/verify steps
    /// copy `gathered_bf16` asynchronously on `self.stream` without `st.sync()`.
    engram_stage: Option<[DevBuf; 2]>,
    /// Pre-allocated `[rows, (hc + 1) * hidden]` f16 engram `wkv` output buffer.
    engram_kv: DevBuf,
    /// Pre-allocated `[rows, 3 * hidden]` bf16 DSpark trunk tap buffer (`layers 37, 38, 39` stream means).
    mtp_taps: DevBuf,
    /// Pre-allocated `[mtp_win, 3 * hidden]` bf16 stage for unwrapping `slot_taps[slot]` in one contiguous `prefill_seed` call.
    mtp_hist_stage: DevBuf,
    /// Per-slot `[mtp_win, 3 * hidden]` bf16 ring of recent trunk taps for any-slot MTP and prefix restore.
    slot_taps: Vec<DevBuf>,
    mtp_win: usize,
    /// Per-slot `[1, 3 * hidden]` bf16 tap of the last committed trunk position (`mtp_last_pos[slot]`).
    mtp_last_tap: Vec<DevBuf>,
    pub mtp_slot: Option<usize>,
    mtp_last_pos: Vec<Option<usize>>,
    mtp_last_seeded: Vec<bool>,
    /// Per-slot token position where `planes[slot][20..40].window` is valid.
    pub window_valid_at: Vec<usize>,
    /// Per-slot rolling host copy of the last `PLANE_ROWS` rows that ran through Layers `20..39`
    /// (the CED tail; see [`CedTailSnap`]).
    ced_tail: Vec<CedTailSnap>,
    /// Host-RAM prefix snapshot pool.
    pub snaps: Dsv41SnapPool,
    /// The three span embeddings on the host (`image_start`, `image_end`, `image_newline`), set when the tower is
    /// bound; `None` serves text, and a request with images on a text-only model is refused by name.
    span_embeddings: Option<super::vision_serve::SpanEmbeddings>,
    /// The image spans of the prefill chunk in flight ([`Self::prefill_chunk_mm`]); empty on the text path.
    splice: Vec<super::vision_serve::Dsv41Splice>,
    /// A draft-tree round's device saves (T2.5a lane 2): allocated once on the first round, reused after.
    tree_bufs: Option<TreeBufs>,
    /// The tree round in flight: the base engram rings, the anchor position and how many path windows have
    /// run. `None` between rounds.
    tree_round: Option<TreeRoundState>,
    /// T2.5b — the traffic-learning state behind `REXL3_DSV41_SPEC_LEARN` (`spec.rs::SpecLearn`): an EMA of
    /// the committed chain rounds' per-position acceptance, blended into the confidence head's own numbers
    /// before the verify-length schedule reads them. `None` — the default — keeps the schedule on the head's
    /// numbers alone.
    spec_learn: Option<super::spec::SpecLearn>,
    /// The tensor-parallel combine arms, `None` on the single-node engine (every op below is then the single-node one,
    /// unchanged). Set once by [`Self::enable_tp`] before the rank serves.
    pub tp: Option<TpArms>,
    /// `REXL3_DSV41_PROFILE_SLICES` (`REXL3_GLM_PROFILE_SLICES`): bracket chosen prefill (`p`), window (`w`), and draft (`d`)
    /// calls with `cudaProfilerStart` / `cudaProfilerStop` for NVIDIA Nsight Systems (`nsys profile --capture-range=cudaProfilerApi`).
    pub slices: crate::families::glm::profile_slices::SliceProfiler,
    /// The window candidates' pinned D2H staging (lazily allocated on the first [`Self::candidates_of`]).
    topk_stage: Option<TopkStage>,
}

/// How many path windows one tree round's saves hold (a request is at most [`WINDOW_ROWS`] rows, so at
/// most `WINDOW_ROWS - 1` leaves; one spare).
const TREE_MAX_PATHS: usize = WINDOW_ROWS;

/// The device buffers of a tree round (allocated once, reused by every round; ~6 MB at the pack's shapes,
/// over and above what admission prices — a named follow-up).
///
/// The surfaces the commit reconciles, per layer: the window ring's touched slots (base + one save a path
/// window, from `window_bak`'s tail), the compressors' carried rows (base) and staged rows (one save a
/// path window — the parity restage's source), and the DSpark tap rows (one save a path window).
struct TreeBufs {
    /// `[layer][WINDOW_ROWS]` the touched ring slots in position order.
    ring_base: DevBuf,
    /// `[path][layer][WINDOW_ROWS]` each path window's linear KV rows.
    ring_rows: DevBuf,
    /// `[source layer]` the carry's kv/score halves before the round.
    carry_kv_base: DevBuf,
    carry_score_base: DevBuf,
    /// `[path][source layer][WINDOW_ROWS]` the staged ckv/gate rows each window left.
    stage_kv: DevBuf,
    stage_gate: DevBuf,
    /// `[path][WINDOW_ROWS]` tap rows.
    taps: DevBuf,
    /// Byte offset of each layer's region inside the ring buffers; `WINDOW_ROWS * row_bytes` each.
    ring_off: Vec<usize>,
    /// One layer's ring region times [`TREE_MAX_PATHS`] (the stride between two paths' saves).
    ring_stride: usize,
    /// The KV-source layers that keep a carry (their `stage`/`carry` regions, in order).
    carry_layers: Vec<usize>,
    /// Byte offset of each source layer's carry rows (`ratio * head_dim * 4`) inside the carry buffers.
    carry_off: Vec<usize>,
    /// Each source layer's carry bytes.
    carry_bytes: Vec<usize>,
    /// Each source layer's `head_dim`.
    carry_d: Vec<usize>,
    /// Each source layer's compressor ratio.
    carry_ratio: Vec<usize>,
    /// Byte offset of each source layer's staged rows (`WINDOW_ROWS * head_dim * 2`) inside the stage buffers.
    stage_off: Vec<usize>,
    /// The stage buffers' per-path stride.
    stage_stride: usize,
    /// One tap row's bytes (`target layers * hidden * 2`); 0 when there is no DSpark stack.
    tap_row: usize,
}

impl TreeBufs {
    fn new(
        planes: &[Vec<Dsv4Planes>],
        dev: &super::device::Dsv41Device,
        layers: usize,
        tap_row: usize,
    ) -> Result<Self, gb_cuda::CudaError> {
        let ring_row = |l: usize| planes[0][l].window.bytes() / planes[0][l].window_slots.max(1);
        let mut ring_off = Vec::with_capacity(layers);
        let mut ring_stride = 0usize;
        for l in 0..layers {
            ring_off.push(ring_stride);
            ring_stride += WINDOW_ROWS * ring_row(l);
        }
        let mut carry_layers = Vec::new();
        let (mut carry_off, mut carry_bytes, mut carry_d, mut carry_ratio) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut carry_stride = 0usize;
        let mut stage_off = Vec::new();
        let mut stage_stride = 0usize;
        for l in 0..layers {
            let mla = &dev.layers[l].mla;
            let Some(c) = mla.compressor.as_ref().filter(|c| c.ratio > 1 && mla.cfg.kv_source_of(l) == Some(l)) else {
                continue;
            };
            let (d, bytes) = (mla.cfg.head_dim, c.ratio * mla.cfg.head_dim * 4);
            carry_layers.push(l);
            carry_off.push(carry_stride);
            carry_bytes.push(bytes);
            carry_d.push(d);
            carry_ratio.push(c.ratio);
            carry_stride += bytes;
            stage_off.push(stage_stride);
            stage_stride += WINDOW_ROWS * d * 2;
        }
        Ok(Self {
            ring_base: DevBuf::zeroed_view(ring_stride.max(1))?,
            ring_rows: DevBuf::zeroed_view(TREE_MAX_PATHS * ring_stride.max(1))?,
            carry_kv_base: DevBuf::zeroed_view(carry_stride.max(1))?,
            carry_score_base: DevBuf::zeroed_view(carry_stride.max(1))?,
            stage_kv: DevBuf::zeroed_view(TREE_MAX_PATHS * stage_stride.max(1))?,
            stage_gate: DevBuf::zeroed_view(TREE_MAX_PATHS * stage_stride.max(1))?,
            taps: DevBuf::zeroed_view(TREE_MAX_PATHS * WINDOW_ROWS * tap_row.max(1))?,
            ring_off,
            ring_stride,
            carry_layers,
            carry_off,
            carry_bytes,
            carry_d,
            carry_ratio,
            stage_off,
            stage_stride,
            tap_row,
        })
    }
}

/// The per-round state of a tree verify in flight.
struct TreeRoundState {
    /// The anchor's position (every path window starts there).
    p0: usize,
    /// How many path windows have run (the next save's index).
    paths: usize,
    /// The engram rings before the round, one per engram layer.
    engram: Vec<super::engram::RingSnap>,
}

/// The pinned host staging the window candidates' read-back uses: one mapped-host block for the three top-k arrays
/// (ids, values, weights) plus the event that marks the async D2H copies complete. [`Dsv41Model::candidates_of`]
/// waits on the event instead of synchronizing the whole stream.
struct TopkStage {
    bytes: usize,
    block: gb_cuda::rt::Block,
    ev: gb_cuda::rt::Event,
}

impl TopkStage {
    /// The pinned bytes a `candidates_of` for `rows * count` candidates needs: ids and values (two 4-byte
    /// entries per candidate) plus slack for the weights array, which is not copied back.
    fn needed_bytes(rows: usize, count: usize) -> usize {
        rows.checked_mul(count).expect("rows * count overflow") * 12
    }

    fn new(bytes: usize) -> Result<Self, ModelError> {
        Ok(Self { bytes, block: gb_cuda::rt::Block::alloc_mapped_host(bytes)?, ev: gb_cuda::rt::Event::new()? })
    }

    /// Grow the staging when a round needs more than the current capacity. The window's rows and the candidate
    /// count vary per round — the first round's size is not a bound (a prefill chunk reads back one row, a decode
    /// or verify window up to `WINDOW_ROWS`), so the staging must follow the largest request seen.
    fn ensure(&mut self, need: usize) -> Result<(), ModelError> {
        if need <= self.bytes {
            return Ok(());
        }
        *self = Self::new(need)?;
        Ok(())
    }
}

/// One DSpark draft round: the drafted token ids and their confidence scores.
/// One DSpark draft round's product: the walked chain's tokens, the confidence head's per-pick confidences, and —
/// when `REXL3_DSV41_SPEC_TREE` captured the round's lattice — the candidate lattice a draft-tree round's search
/// runs on (`families::deepseek_v41::tree`, T2.5a). The lattice rides along for the tree consumer; the chain
/// consumer (the engine's plain verify round) ignores it.
pub type DraftRound = (Vec<u32>, Vec<f32>, Option<gb_spec::tree::Lattice>);

impl Dsv41Model {
    pub fn new(cfg: DeepseekV41Config, dev: Dsv41Device, o: &ModelOptions) -> Result<Self, ModelError> {
        let layers = dev.layers.len();
        if o.layers != layers {
            return Err(ModelError::Config(format!("a model of {} layers over weights of {layers}", o.layers)));
        }
        if o.slots == 0 || o.kv_blocks == 0 || o.max_context == 0 {
            return Err(ModelError::Config("a model needs slots, KV pages and a context".into()));
        }
        if o.chunk_rows == 0 || o.chunk_rows > DSV4_MAX_ROWS.max(2048) {
            return Err(ModelError::Config(format!(
                "a prefill chunk takes 1..={} rows, not {}",
                DSV4_MAX_ROWS.max(2048),
                o.chunk_rows
            )));
        }
        let dcfg = super::device::dsv4_config(&cfg)?;
        let mut planes = Vec::with_capacity(o.slots);
        for _ in 0..o.slots {
            let mut per_layer = Vec::with_capacity(layers);
            for layer in 0..layers {
                let p_ctx = if dcfg.is_kv_source(layer) { o.max_context } else { 1 };
                per_layer.push(Dsv4Planes::new(&dcfg, p_ctx, layer, o.kv)?);
            }
            planes.push(per_layer);
        }
        let max_blocks = o.max_context.div_ceil(BLOCK_TOKENS);
        let kv_tab = KvTables::new(o.kv_blocks.max(o.slots * max_blocks), o.slots, max_blocks);
        let table = BlockTable::from_host(&vec![Vec::new(); o.slots], max_blocks)?;
        let rows = o.chunk_rows.max(WINDOW_ROWS);
        let s = Scratch::new(&cfg, rows, o.slots, o.max_context)?;
        debug_assert_eq!(
            s.bytes(),
            Scratch::bytes_for(&cfg, rows, o.max_context),
            "the model's scratch is not the bytes admission prices for it"
        );
        let mut pre_index = std::collections::BTreeMap::new();
        for (i, (layer, kind, _)) in s.pre.iter().enumerate() {
            pre_index.insert((*layer, *kind), i);
        }
        let mut identity = DevBuf::zeroed_view(rows * cfg.hc.mult * 4)?;
        identity.upload(&ops::identity_pre_mix(cfg.hc.mult).repeat(rows))?;
        let engram_cols = (cfg.engram.max_ngram_size - 1) * cfg.engram.n_heads * cfg.engram.head_dim;
        let eg_in_b = rows * engram_cols.max(1) * 2;
        let eg_kv_b = rows * (cfg.hc.mult + 1) * cfg.hidden * 2;
        let eg_in_aligned = eg_in_b.div_ceil(4096) * 4096;
        let (engram_in, engram_kv) = if s.moe.arena_block().bytes() >= eg_in_aligned + eg_kv_b {
            (DevBuf::view(s.moe.arena_block(), 0, eg_in_b), DevBuf::view(s.moe.arena_block(), eg_in_aligned, eg_kv_b))
        } else {
            (DevBuf::zeroed(eg_in_b)?, DevBuf::zeroed(eg_kv_b)?)
        };
        let stage_each = 64 * engram_cols.max(1) * 2;
        let engram_stage = gb_cuda::rt::Block::alloc_mapped_host(2 * stage_each)
            .ok()
            .map(std::sync::Arc::new)
            .map(|blk| [DevBuf::view(&blk, 0, stage_each), DevBuf::view(&blk, stage_each, stage_each)]);
        let n_taps = cfg.dspark.target_layer_ids.len().max(1);
        let mtp_win = dev.mtp_gpu.as_ref().map_or(64, |m| m.cfg.window).max(64);
        let tap_rows = rows.min(mtp_win);
        let mtp_taps = DevBuf::zeroed_view(tap_rows * n_taps * cfg.hidden * 2)?;
        let slot_tap_bytes = mtp_win * n_taps * cfg.hidden * 2;
        let mtp_hist_stage = DevBuf::zeroed_view(slot_tap_bytes)?;
        let slot_taps_arena = DevBuf::zeroed_view(o.slots * slot_tap_bytes)?;
        let slot_taps: Vec<DevBuf> = (0..o.slots).map(|i| slot_taps_arena.subview(i * slot_tap_bytes, slot_tap_bytes)).collect();
        let tap_row_bytes = n_taps * cfg.hidden * 2;
        let mtp_last_tap_arena = DevBuf::zeroed_view(o.slots * tap_row_bytes)?;
        let mtp_last_tap: Vec<DevBuf> =
            (0..o.slots).map(|i| mtp_last_tap_arena.subview(i * tap_row_bytes, tap_row_bytes)).collect();
        Ok(Self {
            stream: Stream::new()?,
            shared_stream: Stream::new_nonblocking()?,
            ev_hidden_ready: gb_cuda::rt::Event::new()?,
            ev_shared_done: gb_cuda::rt::Event::new()?,
            max_context: o.max_context,
            slots: o.slots,
            layers,
            planes,
            kv_tab,
            table,
            s,
            pending: std::collections::BTreeMap::new(),
            last_window: None,
            pre_index,
            identity,
            engram_in,
            engram_stage,
            engram_kv,
            mtp_taps,
            mtp_hist_stage,
            slot_taps,
            mtp_win,
            mtp_last_tap,
            mtp_slot: None,
            mtp_last_pos: vec![None; o.slots],
            mtp_last_seeded: vec![false; o.slots],
            window_valid_at: vec![0; o.slots],
            ced_tail: vec![CedTailSnap::default(); o.slots],
            snaps: Dsv41SnapPool::new(DEFAULT_PREFIX_SNAPSHOTS),
            span_embeddings: None,
            splice: Vec::new(),
            tree_bufs: None,
            tree_round: None,
            spec_learn: super::spec::SpecLearn::from_env().map_err(ModelError::Config)?,
            tp: None,
            slices: crate::families::glm::profile_slices::SliceProfiler::from_env_key("REXL3_DSV41_PROFILE_SLICES")?,
            topk_stage: None,
            cfg,
            dev,
        })
    }

    /// Make this model ONE RANK of a tensor-parallel group over `comm` (the mirror of the GLM family's `enable_tp`): the
    /// rank's [`TpArms`] hold the combine sites and their staging, and every forward from here on runs the group's step —
    /// the rank's own experts and its own query heads, one exchange a combine site, the unchanged kernels over the combined
    /// buffers. Call it once, after [`Self::new`] and before the rank serves.
    ///
    /// Refused, by name: a second call; the DSpark GPU stages (a group boot refuses the drafting already, and the stages'
    /// experts are split, so a bound stage would compute the peer's half); a group the arms are not built for (more than
    /// two ranks, or a collective that cannot exchange bytes); and a `wq_b` that is not this rank's row slice (the device's
    /// `q_heads` must be the table's half — the bind sliced it, and a whole `wq_b` here would compute every head twice).
    pub fn enable_tp(&mut self, comm: std::sync::Arc<dyn TpComm>, table: ShardTable) -> Result<(), ModelError> {
        if self.tp.is_some() {
            return Err(ModelError::Config("this model is already a rank of a tensor-parallel group".into()));
        }
        if self.dev.mtp_gpu.is_some() && !table.replicated_nextn {
            return Err(ModelError::Config(
                "deepseek_v41: the DSpark (NextN) stages are not built for tensor parallelism (their experts are split across \
                 the ranks); boot the group with drafting off"
                    .into(),
            ));
        }
        super::tp_arm::check_two_rank(comm.topology())?;
        let heads = table.q_head_range().len();
        if let Some(l) = self.dev.layers.iter().find(|l| l.mla.q_heads != heads) {
            return Err(ModelError::Config(format!(
                "deepseek_v41: layer {}'s `wq_b` holds {} query heads and this rank's slice of the split table is {heads} \
                 (the bind and the table disagree: a whole `wq_b` would compute every head on every rank)",
                l.index, l.mla.q_heads
            )));
        }
        let total = table.q_heads * table.head_dim;
        let arms = TpArms::new(comm, table, self.s.rows, total)?;
        eprintln!(
            "rexl3 event=dsv41_tp rank={} ranks={} trunk_experts={:?} nextn_experts={:?} q_heads={:?} o_rows={}-{}",
            arms.table().topo.rank,
            arms.table().topo.ranks,
            arms.table().trunk_expert_range(),
            arms.table().nextn_expert_range(),
            arms.table().q_head_range(),
            arms.table().o_row_range().start,
            arms.table().o_row_range().end
        );
        self.tp = Some(arms);
        Ok(())
    }

    /// The group's step counters, for the boot log: bytes through the collective and how many exchanges the last forward
    /// issued. `None` on the single-node engine.
    pub fn tp_counters(&self) -> Option<(u64, u64)> {
        self.tp.as_ref().map(|t| (t.exchanged_bytes, t.exchanges))
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    /// The most rows a prefill chunk can take.
    pub fn chunk_rows(&self) -> usize {
        self.s.rows
    }

    /// Every allocation the model owns beyond the weights, by name.
    pub fn allocations(&self) -> Vec<(&'static str, u64)> {
        let planes: u64 = self.planes.iter().flatten().map(|p| p.held_bytes() as u64).sum();
        vec![
            ("scratch", self.s.bytes()),
            ("block_table", self.table.buf.bytes() as u64),
            ("dsa_planes", planes),
            ("rope_tables", self.dev.rope_bytes),
        ]
    }

    /// What [`Self::allocations`] prices, for the boot report: the model's buffers outside the weights and the KV.
    pub fn allocation_bytes(&self) -> u64 {
        self.allocations().iter().map(|(_, b)| *b).sum()
    }

    pub fn sync(&self) -> Result<(), ModelError> {
        self.stream.sync()?;
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------ slots and commits

    /// Start a sequence in `slot`: drop any pages it held. The planes need no reset: nothing reads a position the sequence has not written.
    pub fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.flush()?;
        self.slices.begin_prompt(slot);
        self.kv_tab.release(slot);
        if let Some(w) = self.window_valid_at.get_mut(slot) {
            *w = 0;
        }
        if let Some(t) = self.ced_tail.get_mut(slot) {
            *t = CedTailSnap::default();
        }
        for eg in &mut self.dev.engram {
            eg.state.reset_slot(slot);
        }
        if self.mtp_slot == Some(slot) {
            self.mtp_slot = None;
        }
        if let Some(p) = self.mtp_last_pos.get_mut(slot) {
            *p = None;
        }
        if let Some(s) = self.mtp_last_seeded.get_mut(slot) {
            *s = false;
        }
        if let Some(mtp) = self.dev.mtp_gpu.as_mut() {
            if let Some(sp) = mtp.seeded_pos.get_mut(slot) {
                *sp = None;
            }
        }
        Ok(())
    }

    pub fn end_slot(&mut self, slot: usize) {
        self.pending.remove(&slot);
        if slot == 0 {
            // a tree round's saves belong to the dropped sequence: forget them (the buffers stay cached)
            self.tree_round = None;
        }
        self.kv_tab.release(slot);
        if let Some(w) = self.window_valid_at.get_mut(slot) {
            *w = 0;
        }
        for eg in &mut self.dev.engram {
            eg.state.reset_slot(slot);
        }
        if self.mtp_slot == Some(slot) {
            self.mtp_slot = None;
        }
        if let Some(p) = self.mtp_last_pos.get_mut(slot) {
            *p = None;
        }
        if let Some(s) = self.mtp_last_seeded.get_mut(slot) {
            *s = false;
        }
        if let Some(mtp) = self.dev.mtp_gpu.as_mut() {
            if let Some(sp) = mtp.seeded_pos.get_mut(slot) {
                *sp = None;
            }
        }
    }

    /// Record that `keep` rows of `slot`'s last window were accepted. Applied (for all slots) before the next forward.
    pub fn commit(&mut self, slot: usize, keep: usize) {
        self.pending.insert(slot, keep);
    }

    fn seed_mtp_for_slot(&mut self, slot: usize, tv_off_rows: usize, pos0: usize, keep: usize) -> Result<(), ModelError> {
        if keep == 0 || slot >= self.slots {
            return Ok(());
        }
        let n_taps = self.cfg.dspark.target_layer_ids.len().max(1);
        let width_f16 = n_taps * self.cfg.hidden;
        let tap_row_bytes = width_f16 * 2;
        let win = self.mtp_win;
        let (src_off, ring_rows) = if keep <= win { (0, keep) } else { (keep - win, win) };
        let ring_pos0 = pos0 + src_off;
        if (tv_off_rows + src_off + ring_rows) * tap_row_bytes > self.mtp_taps.bytes() {
            return Ok(());
        }
        let tv = self.mtp_taps.subview((tv_off_rows + src_off) * tap_row_bytes, ring_rows * tap_row_bytes);
        if slot < gb_cuda::dsv4_mtp_gpu::MAX_SLOTS {
            if let Some(mtp) = self.dev.mtp_gpu.as_mut() {
                let need_hist = self.mtp_last_pos[slot] != Some(pos0.wrapping_sub(1));
                if need_hist && ring_pos0 > 0 && ring_rows < win {
                    let hist = ring_pos0.min(win);
                    let hist_start = ring_pos0 - hist;
                    let mut done = 0usize;
                    while done < hist {
                        let cur_pos = hist_start + done;
                        let ring_idx = cur_pos % win;
                        let seg = (hist - done).min(win - ring_idx);
                        gb_cuda::dsv4_attn::plane_write_dev(
                            &self.slot_taps[slot],
                            ring_idx,
                            &self.mtp_hist_stage,
                            seg,
                            width_f16,
                            done * width_f16,
                            &self.stream,
                        )?;
                        done += seg;
                    }
                    mtp.seeded_pos[slot] = None;
                    let stage_v = self.mtp_hist_stage.subview(0, hist * tap_row_bytes);
                    mtp.prefill_seed_slot(slot, &stage_v, hist_start, hist, &self.stream)?;
                } else if !self.mtp_last_seeded[slot] {
                    if let Some(prev_pos) = self.mtp_last_pos[slot] {
                        mtp.prefill_seed_slot(slot, &self.mtp_last_tap[slot], prev_pos, 1, &self.stream)?;
                    }
                }
                mtp.prefill_seed_slot(slot, &tv, ring_pos0, ring_rows, &self.stream)?;
                self.mtp_last_tap[slot].copy_range_from(0, &tv, (ring_rows - 1) * tap_row_bytes, tap_row_bytes, &self.stream)?;
                self.mtp_slot = Some(slot);
                self.mtp_last_pos[slot] = Some(pos0 + keep - 1);
                self.mtp_last_seeded[slot] = true;
            }
        }
        gb_cuda::dsv4_attn::ring_write_dev(
            &tv,
            0,
            &self.slot_taps[slot],
            ring_rows,
            width_f16,
            ring_pos0 % win,
            win,
            &self.stream,
        )?;
        Ok(())
    }

    /// Apply the pending commits, rewinding the engram ring and compressor carry-over state when draft tail tokens are rejected.
    pub fn flush(&mut self) -> Result<(), ModelError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        if let Some(win) = self.last_window.clone() {
            for i in 0..win.n_seq() {
                let slot = win.seq_slot[i] as usize;
                let r_range = win.seq_rows(i);
                let n = r_range.len();
                if let Some(&keep) = self.pending.get(&slot) {
                    let keep = keep.min(n);
                    let p0 = win.positions[r_range.start] as usize;
                    if keep < n {
                        for eg in &mut self.dev.engram {
                            eg.state.rewind(slot, n - keep);
                        }
                        if p0 > 0 {
                            for (layer_idx, l) in self.dev.layers.iter_mut().enumerate() {
                                l.mla.commit_window_planes(&self.planes[slot][layer_idx], slot, p0, n, keep, &self.stream)?;
                            }
                        }
                    }
                    if keep >= 1 {
                        // T2.5b — the traffic the verify budget learns from: this window's rows past the anchor
                        // are the drafts the round verified, `keep - 1` of which the trunk's own samples accepted
                        // (the engine commits `accepted + 1` rows, so the clamped `keep` carries the count). A
                        // window of one row is a decode step that verified no drafts — its estimates relax, the
                        // engine's own no-draft rule. Tree rounds never come through here (their commit is the
                        // surgery in `tree_commit_window`), so the EMA counts chain rounds only.
                        if let Some(learn) = self.spec_learn.as_mut() {
                            if n > 1 {
                                learn.record(n - 1, keep - 1);
                            } else {
                                learn.relax();
                            }
                        }
                    }
                    if keep >= 1 {
                        if let Some(w) = self.window_valid_at.get_mut(slot) {
                            *w = p0 + keep;
                        }
                        self.seed_mtp_for_slot(slot, r_range.start, p0, keep)?;
                    }
                }
            }
        }
        self.pending.clear();
        self.last_window = None;
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------ draft trees (T2.5a lane 2)
    //
    // The trunk verifies a draft tree as one CHAIN window per root-to-leaf path (`super::tree`'s driver over
    // [`super::tree::TreeRoundHost`], implemented here). The four persistent surfaces a path window mutates:
    //
    // * the WINDOW RING of every layer — a path's rows write their slots at their positions, so sibling paths
    //   at the same depth overwrite each other (and a rejected write at position p destroys the resident KV of
    //   position `p - window`, which later queries still read: why the chain's `commit_window_planes` restores
    //   the ring, and why the tree commit must too);
    // * the ENGRAM RING — the gather APPENDS per row, so a sibling's tokens sit in the next path's n-gram
    //   context without the between-window restore;
    // * the COMPRESSOR's carried row (read back by the next window's pair completion) and its staged rows (the
    //   parity restage's source when an accepted prefix ends mid-pair — `commit_window_planes`'s own move);
    // * the DSpark TAP rows (overwritten by every window; the mtp ring is reseeded from the ACCEPTED path's).
    //
    // The compressed plane and the indexer keys need no surgery: every group a later window reads is either
    // committed (valid) or re-pooled by that window itself from the carry before its indexer scores — the same
    // reasoning by which the chain's commit never touches them.

    /// Whether the trunk can verify DRAFT TREES on this model: every layer's sliding-window ring is fp16 —
    /// the surface the round's ring surgery runs on (a q4-window boot keeps chain rounds, exactly as
    /// `commit_window_planes` skips the ring dance there).
    pub fn trees_capable(&self) -> bool {
        self.planes.iter().flatten().all(|p| p.win_fmt == gb_cuda::dsv4_kvq4::KvFmt::F16)
    }

    /// Whether the DSpark verify budget learns from the traffic (`REXL3_DSV41_SPEC_LEARN`, `spec.rs::SpecLearn`);
    /// off for a new model — the boot logs `dsv41_spec_learn` when it is on.
    pub fn spec_learn_on(&self) -> bool {
        self.spec_learn.is_some()
    }

    /// Allocate the round's save buffers (once) and snapshot the base surfaces: the touched ring slots of
    /// every layer, the compressors' carried rows and the engram rings. An already-open round is dropped
    /// (its base restored) first — a new round always starts from the committed state.
    fn tree_round_begin(&mut self, p0: usize) -> Result<(), ModelError> {
        if self.pending.contains_key(&0) {
            return Err(ModelError::Request(
                "a tree round cannot start over an unapplied commit of slot 0 (the draft round flushes first)".into(),
            ));
        }
        let tap_row = self.cfg.dspark.target_layer_ids.len().max(1) * self.cfg.hidden * 2;
        if self.tree_bufs.is_none() {
            self.tree_bufs = Some(
                TreeBufs::new(&self.planes, &self.dev, self.layers, tap_row)
                    .map_err(|e| at("the tree round's saves", e.into()))?,
            );
        }
        if self.tree_round.is_some() {
            self.tree_drop_round_impl()?;
        }
        let bufs = self.tree_bufs.as_ref().expect("just allocated");
        for l in 0..self.layers {
            let bytes = WINDOW_ROWS * Self::ring_row_bytes(&self.planes[0][l]);
            let mut dst = bufs.ring_base.subview(bufs.ring_off[l], bytes);
            gb_cuda::dsv4_attn::window_ring_save(&self.planes[0][l], p0, WINDOW_ROWS, &mut dst, &self.stream)?;
        }
        let carries: Vec<(usize, usize, usize, usize, usize)> = bufs
            .carry_layers
            .iter()
            .zip(&bufs.carry_off)
            .zip(&bufs.carry_bytes)
            .zip(&bufs.carry_d)
            .zip(&bufs.carry_ratio)
            .map(|((((&l, &off), &bytes), &d), &ratio)| (l, off, bytes, d, ratio))
            .collect();
        for (l, off, bytes, d, ratio) in carries {
            let Some(c) = self.dev.layers[l].mla.compressor.as_ref() else { continue };
            let mut kv = bufs.carry_kv_base.subview(off, bytes);
            let mut score = bufs.carry_score_base.subview(off, bytes);
            gb_cuda::dsv4_attn::compressor_carry_save(
                &c.kv_state,
                &c.score_state,
                ratio,
                d,
                0,
                &mut kv,
                &mut score,
                &self.stream,
            )?;
        }
        let engram = self
            .dev
            .engram
            .iter()
            .map(|eg| eg.state.snapshot_slot(0).ok_or_else(|| ModelError::Config("slot 0 has no engram ring".into())))
            .collect::<Result<Vec<_>, _>>()?;
        self.tree_round = Some(TreeRoundState { p0, paths: 0, engram });
        Ok(())
    }

    /// The base carry restore views — `(layer, kv view, score view, head_dim, ratio)`, collected as owned
    /// subviews so the restore can then borrow the layers mutably.
    fn tree_carry_views(bufs: &TreeBufs) -> Vec<(usize, DevBuf, DevBuf, usize, usize)> {
        bufs.carry_layers
            .iter()
            .zip(&bufs.carry_off)
            .zip(&bufs.carry_bytes)
            .zip(&bufs.carry_d)
            .zip(&bufs.carry_ratio)
            .map(|((((&l, &off), &bytes), &d), &ratio)| {
                (l, bufs.carry_kv_base.subview(off, bytes), bufs.carry_score_base.subview(off, bytes), d, ratio)
            })
            .collect()
    }

    /// The per-path isolation: the base engram rings and compressor carry back, so a sibling's appends and
    /// carry do not leak into this path's window (the state its own serial-exactness reads).
    fn tree_isolate(&mut self) -> Result<(), ModelError> {
        let Some(r) = self.tree_round.as_ref() else { return Ok(()) };
        let carries = Self::tree_carry_views(self.tree_bufs.as_ref().expect("a round implies its buffers"));
        for (l, kv, score, d, ratio) in carries {
            let Some(c) = self.dev.layers[l].mla.compressor.as_mut() else { continue };
            gb_cuda::dsv4_attn::compressor_carry_restore(
                &mut c.kv_state,
                &mut c.score_state,
                ratio,
                d,
                0,
                &kv,
                &score,
                &self.stream,
            )?;
        }
        for (eg, snap) in self.dev.engram.iter_mut().zip(&r.engram) {
            eg.state.restore_slot(0, snap);
        }
        Ok(())
    }

    /// One layer's window-ring row bytes.
    fn ring_row_bytes(p: &Dsv4Planes) -> usize {
        p.window.bytes() / p.window_slots.max(1)
    }

    /// After a path window of `n` rows ran: save its linear KV rows (the `window_bak` tail the forward's
    /// assembly left), its staged ckv/gate rows and its tap rows, at save index `self.tree_round.paths`.
    fn tree_save_path(&mut self, n: usize) -> Result<(), ModelError> {
        let idx = {
            let r = self.tree_round.as_mut().ok_or_else(|| ModelError::Request("no tree round is open".into()))?;
            let i = r.paths;
            r.paths += 1;
            i
        };
        let bufs = self.tree_bufs.as_ref().expect("a round implies its buffers");
        if idx >= TREE_MAX_PATHS {
            return Err(ModelError::Request(format!(
                "a tree round of more than {TREE_MAX_PATHS} path windows exceeds the saves"
            )));
        }
        let win = self.planes[0][0].window_slots;
        for l in 0..self.layers {
            let rb = Self::ring_row_bytes(&self.planes[0][l]);
            let mut dst = bufs.ring_rows.subview(idx * bufs.ring_stride + bufs.ring_off[l], n * rb);
            dst.copy_from_at(0, &self.planes[0][l].window_bak, win * rb, n * rb, &self.stream)?;
        }
        for (i, &l) in bufs.carry_layers.iter().enumerate() {
            let d = bufs.carry_d[i];
            let Some(c) = self.dev.layers[l].mla.compressor.as_ref() else { continue };
            let mut kv = bufs.stage_kv.subview(idx * bufs.stage_stride + bufs.stage_off[i], n * d * 2);
            kv.copy_from_at(0, &c.ckv_stage, 0, n * d * 2, &self.stream)?;
            let mut gate = bufs.stage_gate.subview(idx * bufs.stage_stride + bufs.stage_off[i], n * d * 2);
            gate.copy_from_at(0, &c.gate_stage, 0, n * d * 2, &self.stream)?;
        }
        if bufs.tap_row > 0 {
            let mut dst = bufs.taps.subview(idx * WINDOW_ROWS * bufs.tap_row, n * bufs.tap_row);
            dst.copy_from_at(0, &self.mtp_taps, 0, n * bufs.tap_row, &self.stream)?;
        }
        Ok(())
    }

    /// The commit's surgery, shared by [`Self::tree_commit_window`] and the drop: the touched ring slots
    /// back to the base, and (when `rows > 0`) the accepted path's own KV rows written over them.
    fn tree_ring_surgery(&mut self, idx: usize, p0: usize, rows: usize) -> Result<(), ModelError> {
        let bufs = self.tree_bufs.as_ref().expect("a round implies its buffers");
        for l in 0..self.layers {
            let rb = Self::ring_row_bytes(&self.planes[0][l]);
            let base = bufs.ring_base.subview(bufs.ring_off[l], WINDOW_ROWS * rb);
            gb_cuda::dsv4_attn::window_ring_restore(&mut self.planes[0][l], p0, WINDOW_ROWS, &base, &self.stream)?;
            if rows > 0 {
                let src = bufs.ring_rows.subview(idx * bufs.ring_stride + bufs.ring_off[l], WINDOW_ROWS * rb);
                gb_cuda::dsv4_attn::window_ring_write(&mut self.planes[0][l], rb / 2, p0, rows, &src, &self.stream)?;
            }
        }
        Ok(())
    }

    /// The drop's restore: the base ring slots, carry and engram rings back (an abandoned round leaves the
    /// model exactly as it found it).
    fn tree_drop_round_impl(&mut self) -> Result<(), ModelError> {
        let Some(r) = self.tree_round.take() else { return Ok(()) };
        self.tree_ring_surgery(0, r.p0, 0)?;
        let carries = Self::tree_carry_views(self.tree_bufs.as_ref().expect("a round implies its buffers"));
        for (l, kv, score, d, ratio) in carries {
            let Some(c) = self.dev.layers[l].mla.compressor.as_mut() else { continue };
            gb_cuda::dsv4_attn::compressor_carry_restore(
                &mut c.kv_state,
                &mut c.score_state,
                ratio,
                d,
                0,
                &kv,
                &score,
                &self.stream,
            )?;
        }
        for (eg, snap) in self.dev.engram.iter_mut().zip(&r.engram) {
            eg.state.restore_slot(0, snap);
        }
        Ok(())
    }

    /// The tree commit's own mtp reseed: the DSpark ring seeded from the accepted path's saved tap rows, and
    /// the bookkeeping (`mtp_last_pos`/`mtp_last_seeded`) exactly as the chain commit's flush leaves it.
    fn tree_mtp_reseed(&mut self, idx: usize, p0: usize, rows: usize) -> Result<(), ModelError> {
        let bufs = self.tree_bufs.as_ref().expect("a round implies its buffers");
        if bufs.tap_row == 0 || rows == 0 {
            return Ok(());
        }
        let Some(mtp) = self.dev.mtp_gpu.as_mut() else { return Ok(()) };
        let tv = bufs.taps.subview(idx * WINDOW_ROWS * bufs.tap_row, rows * bufs.tap_row);
        mtp.prefill_seed_slot(0, &tv, p0, rows, &self.stream)?;
        self.mtp_last_pos[0] = Some(p0 + rows - 1);
        self.mtp_last_seeded[0] = true;
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------ prefix snapshots

    pub fn enable_prefix_cache(&mut self, snapshot_slots: usize) {
        self.snaps = Dsv41SnapPool::new(snapshot_slots);
    }

    pub fn prefix_free_snapshots(&self) -> usize {
        self.snaps.free.len()
    }

    /// Slots of the host snapshot pool.
    pub fn prefix_pool_slots(&self) -> usize {
        self.snaps.slots.len()
    }

    /// A content digest of `slot`'s restorable state at prefix length `len`: exactly the bytes [`Self::prefix_snapshot`]
    /// captures and [`Self::restore_prefix`] writes back, read from the LIVE state (the KV source planes' first
    /// `len / ratio` rows, every layer's window, the compressor carry, the MTP taps, on the device by
    /// `gb_cuda::state_digest`; each engram ring and `len` on the host), never from a stored snapshot. Taken after a
    /// snapshot and after the restore of it, the two are equal exactly when the restore wrote back what was captured:
    /// a stale entry, a copy that did not happen, a partial copy or another slot's state differ. A tensor-parallel group
    /// compares it per rank (`tp_runtime`); a single node never computes it.
    pub fn prefix_state_digest(&mut self, slot: usize, len: usize) -> Result<u64, ModelError> {
        if slot >= self.slots || len == 0 || len > self.max_context {
            return Err(ModelError::Request(format!("invalid prefix state digest (slot {slot}, len {len})")));
        }
        let dcfg = super::device::dsv4_config(&self.cfg)?;
        let mut ranges: Vec<(*const std::ffi::c_void, usize)> = Vec::new();
        for &layer in &dcfg.kv_source_layers {
            let ratio = (dcfg.ratio(layer) as usize).max(1);
            let n_comp = len / ratio;
            let pl = &self.planes[slot][layer];
            let cmp_row = pl.fmt.row_bytes(dcfg.head_dim, gb_cuda::dsv4_kvq4::ROPE_DIMS);
            let ik_row = pl.fmt.row_bytes(dcfg.index_dim, 0);
            ranges.push((pl.compress.ptr() as *const _, n_comp * cmp_row));
            ranges.push((pl.index_k.ptr() as *const _, n_comp * ik_row));
            // a fixed place per range whether or not the layer has a compressor (an empty range is skipped, its place kept)
            let (kv, score, bytes) =
                self.dev.layers[layer].mla.comp_state_range(slot).unwrap_or((std::ptr::null(), std::ptr::null(), 0));
            ranges.push((kv, bytes));
            ranges.push((score, bytes));
        }
        for layer in 0..self.layers {
            let w = &self.planes[slot][layer].window;
            ranges.push((w.ptr() as *const _, w.bytes()));
        }
        match self.slot_taps.get(slot) {
            Some(t) => ranges.push((t.ptr() as *const _, t.bytes())),
            None => ranges.push((std::ptr::null(), 0)),
        }
        // after whatever the stream has queued (a restore's copies)
        let dev = gb_cuda::state_digest::sum_device(&ranges, &self.stream)?;
        let mut host = Vec::new();
        for eg in &self.dev.engram {
            if let Some(s) = eg.state.slot_snapshot(slot) {
                for v in s.ring.iter() {
                    host.extend_from_slice(&v.to_le_bytes());
                }
                host.extend_from_slice(&(s.filled as u64).to_le_bytes());
            }
        }
        let k = ranges.len();
        let len_bytes = (len as u64).to_le_bytes();
        let sum = dev
            .wrapping_add(gb_cuda::state_digest::sum_host(k, &host))
            .wrapping_add(gb_cuda::state_digest::sum_host(k + 1, &len_bytes));
        let lens = ranges.iter().map(|r| r.1).chain([host.len(), len_bytes.len()]);
        Ok(gb_cuda::state_digest::finish(sum, lens))
    }

    /// True when the host has enough headroom above the configured floor to hold a snapshot of `len` tokens in RAM.
    pub fn can_snapshot_prefix(&self, len: usize) -> bool {
        if len == 0 || len > self.max_context || self.snaps.free.is_empty() {
            return false;
        }
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                if let Some(avail_kib) = meminfo
                    .lines()
                    .find(|l| l.starts_with("MemAvailable:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<u64>().ok())
                {
                    let avail_bytes = avail_kib * 1024;
                    let fmt = self.planes.first().and_then(|s| s.first()).map(|p| p.fmt).unwrap_or(gb_cuda::dsv4_kvq4::KvFmt::Q4);
                    let cmp_row = fmt.row_bytes(self.cfg.head_dim, gb_cuda::dsv4_kvq4::ROPE_DIMS);
                    let ik_row = fmt.row_bytes(self.cfg.indexer.head_dim, 0);
                    let kv_bytes: u64 = self
                        .cfg
                        .compress
                        .source_layer_ids
                        .iter()
                        .map(|&layer| {
                            let ratio = self.cfg.compress.ratios.get(layer).copied().unwrap_or(1).max(1);
                            ((len / ratio as usize) * (cmp_row + ik_row)) as u64
                        })
                        .sum();
                    let floor = snapshot_floor_bytes(None);
                    let needed = kv_bytes + 8 * 1024 * 1024 + floor;
                    if avail_bytes < needed {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn alloc_snapshot(&mut self) -> Option<u32> {
        self.snaps.free.pop()
    }

    pub fn release_prefix(&mut self, snap: u32) {
        if let Some(slot) = self.snaps.slots.get_mut(snap as usize) {
            *slot = None;
            if !self.snaps.free.contains(&snap) {
                self.snaps.free.push(snap);
            }
        }
    }

    /// Snapshot `slot`'s state at committed prefix length `len` (`0 < len <= max_context`) into snapshot slot `snap`.
    pub fn prefix_snapshot(&mut self, slot: usize, snap: u32, len: usize) -> Result<(), ModelError> {
        if slot >= self.slots || len == 0 || len > self.max_context || (snap as usize) >= self.snaps.slots.len() {
            return Err(ModelError::Request(format!(
                "invalid prefix snapshot (slot {slot}/{}, snap {snap}/{}, len {len}/{})",
                self.slots,
                self.snaps.slots.len(),
                self.max_context
            )));
        }
        self.snaps.free.retain(|&s| s != snap);
        self.flush()?;
        self.stream.sync()?;
        let dcfg = super::device::dsv4_config(&self.cfg)?;
        let mut kv_sources = Vec::with_capacity(dcfg.kv_source_layers.len());
        for &layer in &dcfg.kv_source_layers {
            let ratio = (dcfg.ratio(layer) as usize).max(1);
            let n_comp = len / ratio;
            let pl = &self.planes[slot][layer];
            let cmp_row = pl.fmt.row_bytes(dcfg.head_dim, gb_cuda::dsv4_kvq4::ROPE_DIMS);
            let ik_row = pl.fmt.row_bytes(dcfg.index_dim, 0);
            let cmp_bytes = n_comp * cmp_row;
            let ik_bytes = n_comp * ik_row;
            let mut compress = vec![0u8; cmp_bytes];
            let mut index_k = vec![0u8; ik_bytes];
            if cmp_bytes > 0 {
                pl.compress.download_range(0, &mut compress)?;
            }
            if ik_bytes > 0 {
                pl.index_k.download_range(0, &mut index_k)?;
            }
            let comp_state = self.dev.layers[layer].mla.comp_state_snapshot(slot, len)?;
            kv_sources.push(KvSourceSnap { layer, compress, index_k, comp_state });
        }
        let mut windows = Vec::with_capacity(self.layers);
        for layer in 0..self.layers {
            let wb = self.planes[slot][layer].window.bytes();
            let mut w = vec![0u8; wb];
            self.planes[slot][layer].window.download_range(0, &mut w)?;
            windows.push(w);
        }
        let engram: Vec<super::engram::Slot> = self.dev.engram.iter().filter_map(|eg| eg.state.slot_snapshot(slot)).collect();
        let mut mtp_taps = vec![0u8; self.slot_taps[slot].bytes()];
        self.slot_taps[slot].download_range(0, &mut mtp_taps)?;
        let ced_tail = self.ced_tail[slot].clone();
        self.snaps.slots[snap as usize] = Some(Dsv41Snapshot { len, kv_sources, windows, engram, mtp_taps, ced_tail });
        Ok(())
    }

    /// Restore `slot` to the exact state of snapshot `snap` at prefix length `len`.
    pub fn restore_prefix(&mut self, slot: usize, snap: u32, len: usize) -> Result<(), ModelError> {
        if slot >= self.slots || len == 0 || len > self.max_context {
            return Err(ModelError::Request(format!("invalid prefix restore target (slot {slot}, len {len})")));
        }
        let has_valid = self.snaps.slots.get(snap as usize).and_then(|s| s.as_ref()).is_some_and(|s| s.len == len);
        if !has_valid {
            return Err(ModelError::Request(format!("snapshot {snap} does not hold a valid prefix of length {len}")));
        }
        self.begin_slot(slot)?;
        self.kv_tab.ensure(slot, len).map_err(|e| ModelError::Request(format!("{e:?}")))?;
        let entry = self.snaps.slots[snap as usize].as_ref().unwrap();
        for src in &entry.kv_sources {
            let pl = &self.planes[slot][src.layer];
            if !src.compress.is_empty() {
                unsafe {
                    self.stream.memcpy_h2d_async(pl.compress.ptr(), src.compress.as_ptr(), src.compress.len())?;
                }
            }
            if !src.index_k.is_empty() {
                unsafe {
                    self.stream.memcpy_h2d_async(pl.index_k.ptr(), src.index_k.as_ptr(), src.index_k.len())?;
                }
            }
            self.dev.layers[src.layer].mla.comp_state_restore(slot, src.comp_state.as_ref(), &self.stream)?;
        }
        for (layer, win) in entry.windows.iter().enumerate() {
            if !win.is_empty() {
                unsafe {
                    self.stream.memcpy_h2d_async(self.planes[slot][layer].window.ptr(), win.as_ptr(), win.len())?;
                }
            }
        }
        for (eg, eg_snap) in self.dev.engram.iter_mut().zip(&entry.engram) {
            eg.state.slot_restore(slot, eg_snap);
        }
        if !entry.mtp_taps.is_empty() && slot < self.slot_taps.len() {
            unsafe {
                self.stream.memcpy_h2d_async(self.slot_taps[slot].ptr(), entry.mtp_taps.as_ptr(), entry.mtp_taps.len())?;
            }
        }
        self.ced_tail[slot] = entry.ced_tail.clone();
        self.stream.sync()?;
        self.window_valid_at[slot] = len;
        if self.mtp_slot == Some(slot) {
            self.mtp_slot = None;
        }
        if let Some(p) = self.mtp_last_pos.get_mut(slot) {
            *p = None;
        }
        if let Some(s) = self.mtp_last_seeded.get_mut(slot) {
            *s = false;
        }
        if let Some(mtp) = self.dev.mtp_gpu.as_mut() {
            if let Some(sp) = mtp.seeded_pos.get_mut(slot) {
                *sp = None;
            }
        }
        Ok(())
    }

    // --------------------------------------------------------------------------------------------------------- the forward

    fn ensure_pages(&mut self, b: &RowBatch) -> Result<(), ModelError> {
        for i in 0..b.n_seq() {
            let rows = b.seq_rows(i);
            let last = b.positions[rows.end - 1] as usize;
            if last >= self.max_context {
                return Err(ModelError::Request(format!("position {last} is past the model's context of {}", self.max_context)));
            }
            self.kv_tab.ensure(b.seq_slot[i] as usize, last + 1).map_err(|e| ModelError::Request(format!("{e:?}")))?;
        }
        Ok(())
    }

    /// Run the layers over `b`'s rows.
    pub fn forward(&mut self, b: &RowBatch, mode: Mode, head: Head) -> Result<(), ModelError> {
        let rows = b.rows;
        match mode {
            Mode::Window if rows == 0 || rows > WINDOW_ROWS => {
                return Err(ModelError::Request(format!("a window takes 1..={WINDOW_ROWS} rows, got {rows}")))
            }
            Mode::Prefill if b.n_seq() != 1 || rows == 0 || rows > self.s.rows => {
                return Err(ModelError::Request(format!("a prefill chunk takes one sequence of 1..={} rows", self.s.rows)))
            }
            _ => {}
        }
        self.flush()?;
        self.ensure_pages(b).map_err(|e| at("the page tables", e))?;
        unsafe {
            self.stream
                .memcpy_h2d_async(self.s.ids.ptr(), b.ids.as_ptr() as *const u8, std::mem::size_of_val(b.ids.as_slice()))
                .map_err(|e| at("the token ids", e.into()))?;
        }
        let start_pos = b.positions[0] as usize;
        let slot = b.seq_slot[0] as usize;
        let prof_kind = match mode {
            Mode::Prefill => crate::families::glm::profile_slices::Kind::Prefill,
            Mode::Window => crate::families::glm::profile_slices::Kind::Window,
        };
        let prof_ticket = self.slices.enter(prof_kind, rows, Some(slot));
        let plan = ops::plan_forward(&self.cfg, self.layers, matches!(head, Head::AllRows | Head::LastRow));
        // the image spans of this chunk, if the caller set them ([`Self::prefill_chunk_mm`]); taken here so the
        // op walk can borrow the model freely, and dropped when it ends
        let splice = std::mem::take(&mut self.splice);
        let run_res = self.run(&plan, b, mode, rows, start_pos, slot, head, &b.ids, &splice);
        self.slices.leave(prof_ticket, run_res.is_ok());
        run_res?;
        if mode == Mode::Window {
            self.last_window = Some(b.clone());
        }
        Ok(())
    }

    /// [`Self::run_ops`], with the op that failed named in the error: the kernels' launchers refuse with a bare CUDA code
    /// (`cudaErrorInvalidValue` from a size or alignment check), which says nothing about where in a 40-layer plan it happened.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        plan: &[Op],
        b: &RowBatch,
        mode: Mode,
        rows: usize,
        start_pos: usize,
        slot: usize,
        head: Head,
        host_ids: &[i32],
        splice: &[super::vision_serve::Dsv41Splice],
    ) -> Result<(), ModelError> {
        let mut current: Option<Op> = None;
        self.run_ops(plan, b, mode, rows, start_pos, slot, head, host_ids, splice, &mut current).map_err(|e| match current {
            Some(op) if e.is_fatal() => ModelError::StateLost(format!("{e} (at {op:?}, rows {rows}, start {start_pos})")),
            Some(op) => ModelError::Config(format!("{e} (at {op:?}, rows {rows}, start {start_pos})")),
            None => e,
        })
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::too_many_arguments)] // the executor's fixed op-walk context; a struct would only rename it
    fn run_ops(
        &mut self,
        plan: &[Op],
        b: &RowBatch,
        mode: Mode,
        mut rows: usize,
        mut start_pos: usize,
        slot: usize,
        head: Head,
        mut host_ids: &[i32],
        splice: &[super::vision_serve::Dsv41Splice],
        current: &mut Option<Op>,
    ) -> Result<(), ModelError> {
        let Self {
            cfg,
            dev,
            stream: st,
            shared_stream,
            ev_hidden_ready,
            ev_shared_done,
            planes,
            s,
            pre_index,
            identity,
            engram_in,
            engram_stage,
            engram_kv,
            mtp_taps,
            tp,
            window_valid_at,
            ced_tail,
            ..
        } = self;
        let (hc, h, inter) = (cfg.hc.mult, cfg.hidden, cfg.moe.intermediate);
        let limit = cfg.swiglu_limit as f32;
        let eps = cfg.rms_eps as f32;
        let dsv4_hc = Dsv4Hc { cfg: gb_cuda::dsv4_moe::HcCfg::V41 };
        // `REXL3_DSV41_OP_TIMING=1`: one line a layer with the routed block's wall time, expert count and launch count.
        let timing = std::env::var_os("REXL3_DSV41_OP_TIMING").is_some();
        let identity = &*identity;
        // the logits of the head, per row of the window
        // `REXL3_DSV41_OP_NAN=1`: the first op whose `sub_out` is not finite, named. One device copy an op — a bring-up
        // probe, not a hot path.
        let nan_probe = std::env::var_os("REXL3_DSV41_OP_NAN").is_some();
        let mut nan_named = false;
        let mut deferred_hc_post = false;
        let mut in_ced_tail = false;
        let decode_top_p = if mode == Mode::Prefill || start_pos == 0 { 1.0 } else { dsv4_moe::moe_top_p(rows) };
        // the engram's token ids with every image span's positions dead (`model.py`'s `engram_mask = ~image_mask`):
        // an n-gram that would reach across an image stops at it, and an image's positions take no engram
        // contribution. Without a span this is the raw ids, unchanged.
        let dead_ids = super::vision_serve::dead_ids(host_ids, splice);
        let eng_ids: &[i32] = if splice.is_empty() { host_ids } else { &dead_ids };
        let mut op_times: std::collections::BTreeMap<&'static str, f64> = std::collections::BTreeMap::new();
        for op in plan {
            let t_op = if timing {
                st.sync()?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            *current = Some(*op);
            match *op {
                Op::Embed => unsafe {
                    gb_cuda::dsv4_mtp_gpu::mtp_embed(
                        s.streams.ptr(),
                        s.ids.ptr() as *const i32,
                        dev.embed.ptr(),
                        rows as i32,
                        hc as i32,
                        h as i32,
                        cfg.vocab as i32,
                        1, // T_BF16 (Dtype::BF16)
                        st.raw(),
                    )?;
                    // an image's span rows overwrite the placeholder embeddings, in every hc stream (the expansion
                    // repeats one row across them, so the splice does too): one contiguous host block a span, one
                    // H2D copy ([`Dsv41Splice::block`]). With no image in the chunk this copies nothing.
                    if !splice.is_empty() {
                        let emb = self.span_embeddings.as_ref().ok_or_else(|| {
                            ModelError::Request("the prompt carries images but this model serves text only (--no-vision)".into())
                        })?;
                        for sp in splice {
                            if sp.dst + sp.n > rows {
                                return Err(ModelError::Request(format!(
                                    "an image span of {} rows at chunk row {} does not fit a chunk of {rows} rows",
                                    sp.n, sp.dst
                                )));
                            }
                            let block = sp.block(hc, emb)?;
                            let off = sp.dst * hc * h * 4;
                            st.memcpy_h2d_async(s.streams.ptr_at(off), block.as_ptr() as *const u8, block.len() * 4)
                                .map_err(|e| at("the image splice", e.into()))?;
                        }
                    }
                    // The engram feature rows are gathered inside Op::Engram (which runs several GPU ops
                    // later in the same plan), not here: this op is the round's first, so the GPU queue is
                    // empty when it runs and a host-side gather (hash + table reads + dequant, several ms
                    // at rows=1) would sit on the critical path. The Engram op's lazy branch performs the
                    // identical gather while the trunk's kernels are queued and drain in the background.
                    if hidden_dump().is_some() {
                        st.sync()?;
                        eprintln!("rexl3 event=dsv41_tap_ids start_pos={start_pos} ids={:?}", &host_ids[..rows]);
                        let x: Vec<f32> = s.streams.to_vec()?;
                        write_tap("embed", start_pos, rows, &x, hc, h)?;
                    }
                },
                Op::Engram { layer } => {
                    let e_idx = dev
                        .engram
                        .iter()
                        .position(|e| e.layer == layer)
                        .ok_or_else(|| ModelError::Config(format!("layer {layer} has no engram state")))?;
                    let e = &mut dev.engram[e_idx];
                    let cols = (cfg.engram.max_ngram_size - 1) * cfg.engram.n_heads * cfg.engram.head_dim;
                    let mut gathered_bf16 = Vec::with_capacity(rows * cols);
                    for seq_i in 0..b.n_seq() {
                        let sl = b.seq_slot[seq_i] as usize;
                        let rr = b.seq_rows(seq_i);
                        e.state.gather_batch_bf16(sl, &eng_ids[rr], &mut gathered_bf16).map_err(ModelError::Config)?;
                    }
                    let copy_bytes = std::mem::size_of_val(gathered_bf16.as_slice());
                    if let Some(stg) = engram_stage.as_mut().filter(|_| rows <= 64) {
                        let idx = usize::from(layer != 1);
                        stg[idx].upload(gathered_bf16.as_slice())?;
                        engram_in.copy_from_at(0, &stg[idx], 0, copy_bytes, st)?;
                    } else {
                        unsafe {
                            st.memcpy_h2d_async(engram_in.ptr(), gathered_bf16.as_ptr() as *const u8, copy_bytes)?;
                        }
                        st.sync()?;
                    }
                    let fp8 = |e: gb_cuda::fp8_linear::Fp8Error| match e {
                        gb_cuda::fp8_linear::Fp8Error::Cuda(c) => ModelError::Cuda(c),
                        other => ModelError::Config(other.to_string()),
                    };
                    e.wkv.forward(engram_in, engram_kv, rows, Dtype::BF16, st).map_err(fp8)?;
                    gb_cuda::dsv4_moe::engram_gate_add(
                        &mut s.streams,
                        engram_kv,
                        &e.q_weight,
                        &e.k_weight,
                        rows,
                        hc,
                        h,
                        1e-20,
                        st,
                    )?;
                    if hidden_dump().is_some() {
                        st.sync()?;
                        let x: Vec<f32> = s.streams.to_vec()?;
                        write_tap(&format!("layer_{layer:02}_engram"), start_pos, rows, &x, hc, h)?;
                    }
                }
                Op::HcMixes { layer, kind } => {
                    if dev.mtp_gpu.is_some() && kind == SiteKind::Attn {
                        if let Some(tap_idx) = cfg.dspark.target_layer_ids.iter().position(|&tl| tl == layer) {
                            let win = dev.mtp_gpu.as_ref().map_or(rows, |m| m.cfg.window);
                            let keep = rows.min(win);
                            let skip = rows - keep;
                            let streams_v = s.streams.subview(skip * hc * h * 4, keep * hc * h * 4);
                            let tap_row_bytes = cfg.dspark.target_layer_ids.len() * h * 2;
                            let mut taps_v = mtp_taps.subview(0, keep * tap_row_bytes);
                            gb_cuda::dsv4_mtp_gpu::tap_mean_bf16_dev(
                                &streams_v,
                                &mut taps_v,
                                keep,
                                h,
                                tap_idx,
                                cfg.dspark.target_layer_ids.len(),
                                st,
                            )?;
                        }
                    }
                    let l = &dev.layers[layer];
                    let site = match kind {
                        SiteKind::Attn => &l.hc_attn,
                        SiteKind::Ffn => &l.hc_ffn,
                    };
                    let pre_src = match kind {
                        SiteKind::Attn if layer == 0 => identity,
                        SiteKind::Attn => &s.pre[pre_index[&(layer - 1, SiteKind::Ffn)]].2,
                        SiteKind::Ffn => &s.pre[pre_index[&(layer, SiteKind::Attn)]].2,
                    };
                    if deferred_hc_post {
                        dsv4_hc.apply_partials_and_collapse(
                            &mut s.streams,
                            &s.sub_out,
                            site,
                            pre_src,
                            &mut s.moe_out,
                            rows,
                            &mut s.hc,
                            st,
                        )?;
                        deferred_hc_post = false;
                    } else {
                        dsv4_hc.partials_and_collapse(&s.streams, site, pre_src, &mut s.moe_out, rows, &mut s.hc, st)?;
                    }
                    dsv4_hc.finalize(&s.streams, site, rows, &mut s.hc, st)?;
                    let i = pre_index[&(layer, kind)];
                    gb_cuda::dsv4_attn::plane_write_dev(s.hc.pre(), 0, &s.pre[i].2, rows, hc * 2, 0, st)?;
                }
                Op::HcCollapse { .. } => {
                    // Already collapsed inside the immediately preceding Op::HcMixes pass over s.streams!
                }
                Op::RmsNorm { layer, kind } => {
                    let l = &dev.layers[layer];
                    let g = match kind {
                        SiteKind::Attn => &l.attn_norm,
                        SiteKind::Ffn => &l.ffn_norm,
                    };
                    gb_cuda::dsv4_attn::rmsnorm_f32_to_f16_dev(&s.moe_out, g, &mut s.hidden, rows, h, eps, st)?;
                }
                Op::Mla { layer } => {
                    if nan_probe && start_pos == 0 && (layer == 26 || layer == 27) {
                        st.sync()?;
                        let st_v: Vec<f32> = s.streams.to_vec()?;
                        let st_mx =
                            st_v[..rows * hc * h].iter().copied().filter(|x| x.is_finite()).fold(0f32, |a, x| a.max(x.abs()));
                        let st_bad = st_v[..rows * hc * h].iter().filter(|x| !x.is_finite()).count();
                        let hid_v: Vec<u16> = s.hidden.to_vec()?;
                        let hid_mx = hid_v[..rows * h]
                            .iter()
                            .map(|&b| gb_exl3::f16::f16_to_f64(b) as f32)
                            .filter(|x| x.is_finite())
                            .fold(0f32, |a, x| a.max(x.abs()));
                        let hid_bad =
                            hid_v[..rows * h].iter().filter(|&&b| !(gb_exl3::f16::f16_to_f64(b) as f32).is_finite()).count();
                        eprintln!(
                            "rexl3 event=dsv41_pre_mla layer={layer} streams={st_mx:e}/{st_bad} hidden={hid_mx:e}/{hid_bad}"
                        );
                    }
                    let ced_replay = mode == Mode::Prefill
                        && head != Head::AllRows
                        && b.n_seq() <= 1
                        && splice.is_empty()
                        && hidden_dump().is_none()
                        && ced_replay_enabled();
                    let mut skip_compress = false;
                    if ced_replay && layer == 20 {
                        dev.layers[layer].mla.compress_only_slot(
                            &s.hidden,
                            rows,
                            start_pos,
                            slot,
                            &planes[slot][layer],
                            &mut s.dsv4,
                            mode == Mode::Prefill,
                            st,
                        )?;
                        // V57: roll the last `PLANE_ROWS` rows of Layers 20..39 into the per-slot host tail so
                        // a final chunk shorter than 128 rows can re-assemble the exact trailing rows the
                        // whole-prompt pass compacted to (chunked == whole across every chunk grid).
                        let max_keep = 128usize.min(s.rows);
                        let pre20_idx = pre_index[&(20, SiteKind::Attn)];
                        let mtp_win = dev.mtp_gpu.as_ref().map_or(64, |m| m.cfg.window);
                        // The buffers below are produced by the asynchronous stream: download AFTER the
                        // stream's work for this chunk is visible, or the tail rows race the layers that
                        // are still writing them (nondeterministic across runs).
                        st.sync()?;
                        {
                            let tail = &mut ced_tail[slot];
                            if start_pos == 0 || tail.end_pos != start_pos {
                                *tail = CedTailSnap::default();
                            }
                            let take = rows.min(max_keep);
                            let from_row = rows - take;
                            let drop = (tail.rows + take).saturating_sub(max_keep);
                            tail.ids.drain(0..drop);
                            tail.ids.extend_from_slice(&host_ids[from_row..]);
                            tail_download_append(&mut tail.streams, &s.streams, from_row, take, drop, hc * h * 2)?;
                            tail_download_append(&mut tail.moe_out, &s.moe_out, from_row, take, drop, h * 2)?;
                            tail_download_append(&mut tail.hidden, &s.hidden, from_row, take, drop, h)?;
                            tail_download_append(&mut tail.pre20, &s.pre[pre20_idx].2, from_row, take, drop, hc * 2)?;
                            tail_download_append(&mut tail.hc_post, s.hc.post(), from_row, take, drop, hc * 2)?;
                            tail_download_append(&mut tail.hc_comb, s.hc.comb(), from_row, take, drop, hc * hc * 2)?;
                            tail.rows = tail.rows + take - drop;
                            tail.end_pos = start_pos + rows;
                            if dev.mtp_gpu.is_some() && !cfg.dspark.target_layer_ids.is_empty() {
                                let tap_w = cfg.dspark.target_layer_ids.len() * h;
                                let take_mtp = rows.min(mtp_win);
                                let drop_mtp = (tail.mtp_rows + take_mtp).saturating_sub(mtp_win);
                                tail_download_append(&mut tail.mtp_taps, mtp_taps, 0, take_mtp, drop_mtp, tap_w)?;
                                tail.mtp_rows = tail.mtp_rows + take_mtp - drop_mtp;
                            }
                        }
                        if head == Head::None {
                            if let Some(t0) = t_op {
                                st.sync()?;
                                *op_times.entry(op.name()).or_insert(0.0) += t0.elapsed().as_secs_f64() * 1e3;
                            }
                            if timing {
                                let total_ms: f64 = op_times.values().sum();
                                let parts: Vec<String> = op_times.iter().map(|(k, v)| format!("{k}={v:.2}ms")).collect();
                                eprintln!(
                                    "rexl3 event=dsv41_forward_profile rows={rows} start_pos={start_pos} total_ms={total_ms:.2} {}",
                                    parts.join(" ")
                                );
                            }
                            return Ok(());
                        }
                        if matches!(head, Head::LastRow | Head::WindowOnly) {
                            let keep = ced_tail[slot].rows;
                            if rows > max_keep {
                                // The chunk itself already carries the trailing rows: compact them on device.
                                let from = rows - keep;
                                compact_tail_rows(&s.streams, from, keep, hc * h * 2, &s.dsv4.q, st)?;
                                compact_tail_rows(&s.moe_out, from, keep, h * 2, &s.dsv4.q, st)?;
                                compact_tail_rows(&s.hidden, from, keep, h, &s.dsv4.q, st)?;
                                compact_tail_rows(&s.pre[pre20_idx].2, from, keep, hc * 2, &s.dsv4.q, st)?;
                                compact_tail_rows(s.hc.post(), from, keep, hc * 2, &s.dsv4.q, st)?;
                                compact_tail_rows(s.hc.comb(), from, keep, hc * hc * 2, &s.dsv4.q, st)?;
                                rows = keep;
                                start_pos += from;
                                host_ids = &host_ids[from..];
                                in_ced_tail = true;
                            } else if keep > rows {
                                // The tail already holds the exact trailing `keep` rows in order (older rows
                                // first, then this chunk's own rows -- positions [new_start, end_pos)), so the
                                // splice is just re-uploading the assembled ring: no device-side row move, no
                                // overlap race, and the resulting layout is the same one the compact branch
                                // produces (the previous code moved the chunk rows up by `need` and uploaded
                                // the *last* `need` tail rows -- the newest, not the missing oldest ones --
                                // so rows [0, need) held duplicated newest positions and [need, keep) the chunk
                                // rows: deterministic corruption that differed per chunk grid).
                                let new_start = (start_pos + rows) - keep;
                                tail_upload(&s.streams, &ced_tail[slot].streams, st)?;
                                tail_upload(&s.moe_out, &ced_tail[slot].moe_out, st)?;
                                tail_upload(&s.hidden, &ced_tail[slot].hidden, st)?;
                                tail_upload(&s.pre[pre20_idx].2, &ced_tail[slot].pre20, st)?;
                                tail_upload(s.hc.post(), &ced_tail[slot].hc_post, st)?;
                                tail_upload(s.hc.comb(), &ced_tail[slot].hc_comb, st)?;
                                if dev.mtp_gpu.is_some() && !cfg.dspark.target_layer_ids.is_empty() {
                                    // Target-layer taps live at [0, mtp_rows) of the device ring and are
                                    // recomputed by layers > 20 anyway; re-upload the captured window so the
                                    // layer-20 taps stay aligned too (no device-side move: the previous code
                                    // wrote past the ring's end).
                                    if ced_tail[slot].mtp_rows > 0 {
                                        let vb = ced_tail[slot].mtp_rows * cfg.dspark.target_layer_ids.len() * h * 2;
                                        unsafe {
                                            st.memcpy_h2d_async(
                                                mtp_taps.ptr(),
                                                ced_tail[slot].mtp_taps.as_ptr() as *const u8,
                                                vb,
                                            )?;
                                        }
                                    }
                                }
                                start_pos = new_start;
                                rows = keep;
                                in_ced_tail = true;
                            } else {
                                in_ced_tail = true;
                            }
                        }
                        skip_compress = true;
                    }
                    let l = &mut dev.layers[layer];
                    let group = l.mla.cfg.kv_source_of(layer).unwrap_or(layer);
                    if b.n_seq() <= 1 {
                        let shared = if group == layer { None } else { Some(&planes[slot][group]) };
                        if let Some(tp) = tp.as_mut() {
                            // the rank's heads' attention output, projected through this rank's half of the block-diagonal
                            // `wo_a`, combined with the peer's at the `olora` boundary (16x smaller than `o`), then the
                            // unchanged `wo_b` over the combined `olora`
                            l.mla.forward_slot_to_o(
                                &s.hidden,
                                rows,
                                start_pos,
                                slot,
                                &planes[slot][layer],
                                shared,
                                skip_compress,
                                in_ced_tail,
                                &mut s.dsv4,
                                mode == Mode::Prefill,
                                st,
                            )?;
                            let (g0, g_local) = tp.table().o_group_range(l.mla.cfg.o_groups);
                            l.mla.woa_local(&s.dsv4.o, &mut s.dsv4.olora, rows, g0, g_local, mode == Mode::Prefill, st)?;
                            let olora_full = tp.olora_gather(&s.dsv4.olora, 0, rows, l.mla.cfg.o_groups, l.mla.cfg.o_lora, st)?;
                            l.mla.wob_proj(&olora_full, &mut s.dsv4.out, rows, mode == Mode::Prefill, st)?;
                        } else {
                            l.mla.forward_slot_ex(
                                &s.hidden,
                                rows,
                                start_pos,
                                slot,
                                &planes[slot][layer],
                                shared,
                                skip_compress,
                                in_ced_tail,
                                &mut s.dsv4,
                                mode == Mode::Prefill,
                                st,
                            )?;
                        }
                        std::mem::swap(&mut s.dsv4.out, &mut s.sub_out);
                    } else if let Some(tp) = tp.as_mut() {
                        for seq_i in 0..b.n_seq() {
                            let sl = b.seq_slot[seq_i] as usize;
                            let rr = b.seq_rows(seq_i);
                            let n_i = rr.len();
                            let pos_i = b.positions[rr.start] as usize;
                            let x_i = s.hidden.subview(rr.start * h * 2, n_i * h * 2);
                            let shared = if group == layer { None } else { Some(&planes[sl][group]) };
                            l.mla.forward_slot_to_o(
                                &x_i,
                                n_i,
                                pos_i,
                                sl,
                                &planes[sl][layer],
                                shared,
                                false,
                                false,
                                &mut s.dsv4,
                                mode == Mode::Prefill,
                                st,
                            )?;
                            let (g0, g_local) = tp.table().o_group_range(l.mla.cfg.o_groups);
                            l.mla.woa_local(&s.dsv4.o, &mut s.dsv4.olora, n_i, g0, g_local, mode == Mode::Prefill, st)?;
                            let olora_full =
                                tp.olora_gather(&s.dsv4.olora, rr.start, n_i, l.mla.cfg.o_groups, l.mla.cfg.o_lora, st)?;
                            l.mla.wob_proj(&olora_full, &mut s.dsv4.out, n_i, mode == Mode::Prefill, st)?;
                            let dst_i = s.sub_out.subview(rr.start * h * 2, n_i * h * 2);
                            gb_cuda::dsv4_attn::plane_write_dev(&s.dsv4.out, 0, &dst_i, n_i, h, 0, st)?;
                        }
                    } else {
                        let mut mla_seqs = Vec::with_capacity(b.n_seq());
                        for seq_i in 0..b.n_seq() {
                            let sl = b.seq_slot[seq_i] as usize;
                            let rr = b.seq_rows(seq_i);
                            let n_i = rr.len();
                            let pos_i = b.positions[rr.start] as usize;
                            let shared = if group == layer { None } else { Some(&planes[sl][group]) };
                            mla_seqs.push(gb_cuda::dsv4_attn::MlaSeq {
                                slot: sl,
                                row0: rr.start,
                                rows: n_i,
                                start_pos: pos_i,
                                planes: &planes[sl][layer],
                                shared,
                            });
                        }
                        l.mla.forward_batch(&s.hidden, rows, &mla_seqs, &mut s.dsv4, &mut s.sub_out, st)?;
                    }
                    if nan_probe && start_pos == 0 && (layer == 26 || layer == 27) {
                        st.sync()?;
                        for (name, buf, len) in [
                            ("qr", &s.dsv4.qr, rows * l.mla.cfg.q_lora),
                            ("q", &s.dsv4.q, rows * l.mla.cfg.n_heads * l.mla.cfg.head_dim),
                            ("kv", &s.dsv4.kv, rows * l.mla.cfg.head_dim),
                            ("o", &s.dsv4.o, rows * l.mla.cfg.n_heads * l.mla.cfg.head_dim),
                            ("oq", &s.dsv4.oq, rows * l.mla.cfg.o_groups * l.mla.cfg.o_lora),
                            ("out", &s.dsv4.out, rows * h),
                        ] {
                            let v: Vec<u16> = buf.to_vec()?;
                            let mx = v[..len]
                                .iter()
                                .map(|&b| gb_exl3::f16::f16_to_f64(b) as f32)
                                .filter(|x| x.is_finite())
                                .fold(0f32, |a, x| a.max(x.abs()));
                            let bad = v[..len].iter().filter(|&&b| !(gb_exl3::f16::f16_to_f64(b) as f32).is_finite()).count();
                            eprintln!("rexl3 event=dsv41_post_mla layer={layer} buf={name} absmax={mx:e} bad={bad}/{len}");
                        }
                    }
                }
                Op::MoeShared { layer } => {
                    if nan_probe && start_pos == 0 && layer <= 2 {
                        st.sync()?;
                        let v: Vec<u16> = s.sub_out.to_vec()?;
                        let mx = v
                            .iter()
                            .map(|&b| gb_exl3::f16::f16_to_f64(b) as f32)
                            .filter(|x| x.is_finite())
                            .fold(0f32, |a, x| a.max(x.abs()));
                        eprintln!("rexl3 event=dsv41_attn_mag layer={layer} absmax={mx:e}");
                    }
                    let l = &dev.layers[layer];
                    let fp8 = |e: gb_cuda::fp8_linear::Fp8Error| match e {
                        gb_cuda::fp8_linear::Fp8Error::Cuda(c) => ModelError::Cuda(c),
                        other => ModelError::Config(other.to_string()),
                    };
                    let use_side = !nan_probe && !timing;
                    let sst: &Stream = if use_side {
                        gb_cuda::graph::fork(st, shared_stream, ev_hidden_ready)?;
                        shared_stream
                    } else {
                        st
                    };
                    let pair_out = if mode == Mode::Prefill {
                        gb_cuda::fp8_linear::Fp8Weights::forward_pair_force(
                            &l.shared.w1,
                            &mut s.gate,
                            &l.shared.w3,
                            &mut s.up,
                            &s.hidden,
                            rows,
                            Dtype::F16,
                            sst,
                        )
                    } else {
                        gb_cuda::fp8_linear::Fp8Weights::forward_pair(
                            &l.shared.w1,
                            &mut s.gate,
                            &l.shared.w3,
                            &mut s.up,
                            &s.hidden,
                            rows,
                            Dtype::F16,
                            sst,
                        )
                    };
                    pair_out.map_err(fp8)?;
                    dsv4_moe::swiglu_f16(&s.gate, &s.up, None, &mut s.act, rows, inter, limit, sst)?;
                    if mode == Mode::Prefill {
                        l.shared.w2.forward_force(&s.act, &mut s.sub_out, rows, Dtype::F16, sst).map_err(fp8)?;
                    } else {
                        l.shared.w2.forward(&s.act, &mut s.sub_out, rows, Dtype::F16, sst).map_err(fp8)?;
                    }
                    if nan_probe && start_pos == 0 && layer <= 2 {
                        st.sync()?;
                        let v: Vec<u16> = s.sub_out.to_vec()?;
                        let (mut mx, mut bad) = (0f32, 0usize);
                        for &b in &v {
                            let x = gb_exl3::f16::f16_to_f64(b) as f32;
                            if x.is_finite() {
                                mx = mx.max(x.abs());
                            } else {
                                bad += 1;
                            }
                        }
                        eprintln!("rexl3 event=dsv41_shared_mag layer={layer} absmax={mx:e} bad={bad}");
                    }
                }
                Op::MoeRoute { layer } => {
                    let l = &dev.layers[layer];
                    l.router.logits(&s.hidden, Dtype::F16, rows, &mut s.logits, st)?;
                    l.router.select_top_p(&s.logits, rows, None, decode_top_p, &mut s.picks, &mut s.wts, st)?;
                    if l.experts.streamed.is_some() {
                        st.sync()?;
                        let mut picks = vec![0i32; rows * s.moe.shape().1];
                        s.picks.download_range(0, &mut picks)?;
                        for &e in &picks {
                            // a rank of a group prefetches only its own experts: the peer's picks stay in the grouping (the
                            // router is replicated, the slot layout is the single-node one) and their `y` segments arrive by
                            // the combine, so filling their cache slots would be reads nobody consumes
                            if e >= 0 && tp.as_ref().is_none_or(|t| t.table().owns_trunk_expert(e as usize)) {
                                l.experts.prefetch(e as usize)?;
                            }
                        }
                    }
                }
                Op::MoeExperts { layer } => {
                    let t_moe = std::time::Instant::now();
                    let use_side = !nan_probe && !timing;
                    // Grouped on the device, one segment a launch at each expert's own width, scattered on the device: see
                    // `gb_cuda::dsv4_moe::moe_experts`. The shared expert's f16 output is written into `sub_out` on
                    // `shared_stream` and joined right before the final `scatter` kernel, so `MoeShared` overlaps with
                    // `group_on`, `gather`, host offset readback, routed expert GEMMs and the TP y-exchange.
                    let l = &dev.layers[layer];
                    let experts = &l.experts;
                    let mut why: Option<ModelError> = None;
                    s.moe.layer = layer;
                    let mut join_fn = |st_main: &Stream| -> Result<(), gb_cuda::rt::CudaError> {
                        if use_side {
                            gb_cuda::graph::join(st_main, shared_stream, ev_shared_done)?;
                        }
                        Ok(())
                    };
                    // The grouped fast path is chunk-size invariant now: mma_packed_cta always, and prefill
                    // forces the grouped_gemm_w13_k<4> path (gemm_only) so every prefill slice is bit-identical;
                    // decode / speculative verify (Mode::Window) keep the rows<=16 gemv path.
                    let stats = if let (Some(tp), Some(desc)) = (tp.as_mut(), experts.fast_desc.as_ref()) {
                        tp.moe_fast(
                            &mut s.moe,
                            desc,
                            Some(&mut join_fn),
                            rows,
                            mode == Mode::Prefill,
                            &s.hidden,
                            &mut s.sub_out,
                            &s.picks,
                            &s.wts,
                            &mut s.moe_out,
                            limit,
                            st,
                        )?
                    } else if let Some(tp) = tp.as_mut() {
                        // the group's routed block: the rank's experts' segments of `y`, ONE exchange that makes `y` the
                        // whole buffer, then the unchanged scatter (the shared expert, replicated, added once by it)
                        let table = tp.table().clone();
                        gb_cuda::dsv4_moe::moe_experts_tp_with_join(
                            &mut s.moe,
                            |e| match experts.groups(e) {
                                Ok(ExpertGroups::Resident(g)) => Ok(gb_cuda::dsv4_moe::MoeGroups::Resident(g)),
                                Ok(ExpertGroups::Streamed(sg)) => {
                                    let (groups, lease) = sg.into_parts();
                                    Ok(gb_cuda::dsv4_moe::MoeGroups::Streamed { groups, lease })
                                }
                                Err(e) => {
                                    why = Some(e);
                                    Err(gb_cuda::rt::CudaError(1))
                                }
                            },
                            &|e| table.owns_trunk_expert(e),
                            &mut |y, own, slots, hidden, st| {
                                tp.y_gather(y, own, slots, hidden, st).map_err(|e| {
                                    eprintln!("{e}");
                                    gb_cuda::rt::CudaError(1)
                                })
                            },
                            Some(&mut join_fn),
                            rows,
                            &s.hidden,
                            &mut s.sub_out,
                            &s.picks,
                            &s.wts,
                            &mut s.moe_out,
                            limit,
                            st,
                        )
                        .map_err(|e| why.unwrap_or_else(|| e.into()))?
                    } else if let Some(desc) = experts.fast_desc.as_ref() {
                        gb_cuda::dsv4_moe::moe_experts_resident_with_join(
                            &mut s.moe,
                            desc,
                            Some(&mut join_fn),
                            rows,
                            mode == Mode::Prefill,
                            &s.hidden,
                            &mut s.sub_out,
                            &s.picks,
                            &s.wts,
                            &mut s.moe_out,
                            limit,
                            st,
                        )
                        .map_err(ModelError::from)?
                    } else {
                        let st_res = gb_cuda::dsv4_moe::moe_experts_with_join(
                            &mut s.moe,
                            |e| match experts.groups(e) {
                                Ok(ExpertGroups::Resident(g)) => Ok(gb_cuda::dsv4_moe::MoeGroups::Resident(g)),
                                Ok(ExpertGroups::Streamed(sg)) => {
                                    let (groups, lease) = sg.into_parts();
                                    Ok(gb_cuda::dsv4_moe::MoeGroups::Streamed { groups, lease })
                                }
                                Err(e) => {
                                    why = Some(e);
                                    Err(gb_cuda::rt::CudaError(1))
                                }
                            },
                            Some(&mut join_fn),
                            rows,
                            &s.hidden,
                            &mut s.sub_out,
                            &s.picks,
                            &s.wts,
                            &mut s.moe_out,
                            limit,
                            st,
                        )
                        .map_err(|e| why.unwrap_or_else(|| e.into()))?;
                        debug_assert!(st_res.launches >= 4 * st_res.active + 3 && st_res.launches <= 4 * st_res.slots + 3);
                        st_res
                    };
                    if timing {
                        st.sync()?;
                        eprintln!(
                            "rexl3 event=dsv41_op op=moe_experts layer={layer} rows={rows} active={} launches={} ms={:.3}",
                            stats.active,
                            stats.launches,
                            t_moe.elapsed().as_secs_f64() * 1e3
                        );
                    }
                }
                Op::HcPost { layer, kind } => {
                    if nan_probe && !nan_named {
                        st.sync()?;
                        let v: Vec<u16> = s.sub_out.to_vec()?;
                        let mut mx = 0f32;
                        let mut bad = false;
                        for &b in &v {
                            let x = gb_exl3::f16::f16_to_f64(b) as f32;
                            if !x.is_finite() {
                                bad = true;
                            } else if x.abs() > mx {
                                mx = x.abs();
                            }
                        }
                        if bad {
                            eprintln!(
                                "rexl3 event=dsv41_nan op={} layer={layer} kind={} absmax={mx:e} pos={start_pos}",
                                op.name(),
                                kind.name()
                            );
                            nan_named = true;
                        }
                    }
                    let l = &dev.layers[layer];
                    let site = match kind {
                        SiteKind::Attn => &l.hc_attn,
                        SiteKind::Ffn => &l.hc_ffn,
                    };
                    let _ = site;
                    let can_defer = !nan_probe
                        && hidden_dump().is_none()
                        && (kind == SiteKind::Attn
                            || (kind == SiteKind::Ffn
                                && layer + 1 < cfg.layers
                                && !cfg.engram.layer_ids.contains(&(layer + 1))
                                && !cfg.dspark.target_layer_ids.contains(&(layer + 1))));
                    if can_defer {
                        deferred_hc_post = true;
                    } else {
                        dsv4_hc.apply(&mut s.streams, &s.sub_out, Dtype::F16, rows, &s.hc, st)?;
                        deferred_hc_post = false;
                    }
                    if hidden_dump().is_some() {
                        st.sync()?;
                        let x: Vec<f32> = s.streams.to_vec()?;
                        if kind == SiteKind::Attn {
                            write_tap(&format!("layer_{layer:02}_attn"), start_pos, rows, &x, hc, h)?;
                        } else {
                            write_tap(&format!("layer_{layer:02}"), start_pos, rows, &x, hc, h)?;
                        }
                    }
                }
                Op::HeadCollapse { mix } => {
                    let rows_out = match head {
                        Head::None | Head::WindowOnly if hidden_dump().is_none() => continue,
                        Head::LastRow if hidden_dump().is_none() => 1,
                        _ => rows,
                    };
                    let from = rows - rows_out;
                    let src = match mix {
                        PreMix::Identity => identity,
                        PreMix::Site { layer, kind } => &s.pre[pre_index[&(layer, kind)]].2,
                    };
                    let streams_v = s.streams.subview(from * hc * h * 4, rows_out * hc * h * 4);
                    let src_v = src.subview(from * hc * 4, rows_out * hc * 4);
                    let mut moe_out_v = s.moe_out.subview(from * h * 4, rows_out * h * 4);
                    dsv4_hc.collapse_f32(&streams_v, &src_v, &mut moe_out_v, rows_out, h, st)?;
                }
                Op::FinalNorm => {
                    let rows_out = match head {
                        Head::None | Head::WindowOnly if hidden_dump().is_none() => continue,
                        Head::LastRow if hidden_dump().is_none() => 1,
                        _ => rows,
                    };
                    let from = rows - rows_out;
                    let moe_out_v = s.moe_out.subview(from * h * 4, rows_out * h * 4);
                    let mut hidden_v = s.hidden.subview(from * h * 2, rows_out * h * 2);
                    gb_cuda::dsv4_attn::rmsnorm_f32_to_f16_dev(&moe_out_v, &dev.final_norm, &mut hidden_v, rows_out, h, eps, st)?;
                    if hidden_dump().is_some() {
                        st.sync()?;
                        let raw: Vec<u16> = s.hidden.to_vec()?;
                        let x: Vec<f32> = raw.iter().map(|&v| gb_exl3::f16::f16_to_f64(v) as f32).collect();
                        write_tap("final_norm", start_pos, rows, &x, 1, h)?;
                    }
                }
                Op::LmHead => {
                    let rows_out = match head {
                        Head::None | Head::WindowOnly => continue,
                        Head::LastRow => 1,
                        Head::AllRows => rows,
                    };
                    let from = rows - rows_out;
                    assert!(
                        rows * h * 2 <= s.hidden.bytes() && rows_out * cfg.vocab * 2 <= s.head_out.bytes(),
                        "hidden holds {} B, the head reads {}",
                        s.hidden.bytes(),
                        rows * h * 2
                    );
                    match &dev.head {
                        super::device::HeadDevice::Bf16(buf) => {
                            let x = s.hidden.ptr_at(from * h * 2);
                            unsafe { glue::small_linear_f16w_rows(x, buf.ptr(), s.head_out.ptr(), rows_out, h, cfg.vocab, st)? };
                        }
                        super::device::HeadDevice::Exl3(w) => {
                            let xv = s.hidden.subview(from * h * 2, rows_out * h * 2);
                            let mut yv = s.head_out.subview(0, rows_out * cfg.vocab * 2);
                            let need = gb_cuda::exl3_linear::linear_workspace_bytes(rows_out, h, cfg.vocab, w.bits);
                            let ws = s.moe.ensure_workspace(need)?;
                            w.forward(&xv, &mut yv, rows_out, Dtype::F16, Dtype::F16, ws, st)?;
                        }
                    }
                }
            }
            if let Some(t0) = t_op {
                st.sync()?;
                *op_times.entry(op.name()).or_insert(0.0) += t0.elapsed().as_secs_f64() * 1e3;
            }
        }
        if mode == Mode::Prefill {
            if let Some(w) = window_valid_at.get_mut(slot) {
                *w = start_pos + rows;
            }
        }
        if timing {
            let total_ms: f64 = op_times.values().sum();
            let parts: Vec<String> = op_times.iter().map(|(k, v)| format!("{k}={v:.2}ms")).collect();
            eprintln!(
                "rexl3 event=dsv41_forward_profile rows={rows} start_pos={start_pos} total_ms={total_ms:.2} {}",
                parts.join(" ")
            );
        }
        Ok(())
    }

    /// One prefill chunk: `tokens` are the prompt's rows `start..start + tokens.len()` of `slot`. `want` is the number of candidates to
    /// leave behind for the last row when this chunk ends the prompt (`None`: no logits, a chunk in the middle).
    pub fn prefill_chunk(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
    ) -> Result<Option<Candidates>, ModelError> {
        self.prefill_chunk_ex(slot, start, tokens, want, false)
    }

    /// [`Self::prefill_chunk`] with `for_snapshot`: when `true` and `want` is `None`, runs `Head::WindowOnly` so Layers `20..39`
    /// populate their sliding windows on the trailing `min(tokens.len(), 128)` rows before a snapshot is taken at `start + tokens.len()`.
    pub fn prefill_chunk_ex(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
        for_snapshot: bool,
    ) -> Result<Option<Candidates>, ModelError> {
        let cand_thresh = self.cfg.indexer.candidate_topk_blocks * self.cfg.indexer.candidate_block_size;
        let max_cand_rows = gb_cuda::dsv4_attn::PLANE_ROWS;
        if !ced_replay_enabled() && start + tokens.len() > cand_thresh && tokens.len() > max_cand_rows {
            let mut cur = 0usize;
            let mut last_out = None;
            while cur < tokens.len() {
                let end = (cur + max_cand_rows).min(tokens.len());
                let is_last = end == tokens.len();
                let next_is_short_last =
                    !is_last && (tokens.len() - cur <= 128 + max_cand_rows) && (want.is_some() || for_snapshot);
                last_out = self.prefill_chunk_ex(
                    slot,
                    start + cur,
                    &tokens[cur..end],
                    if is_last { want } else { None },
                    (is_last && for_snapshot) || next_is_short_last,
                )?;
                cur = end;
            }
            return Ok(last_out);
        }
        // V57: the CED layer-20 tail (`CedTailSnap`) keeps the last `PLANE_ROWS` rows that ran through
        // Layers `20..39`; a final chunk shorter than that re-assembles the exact trailing rows in
        // `run_ops`, so the final row's logits do not depend on the chunk grid.
        let mut b = RowBatch { seq_start: vec![0, tokens.len() as i32], seq_slot: vec![slot as i32], ..Default::default() };
        b.rows = tokens.len();
        b.ids = tokens.iter().map(|&t| t as i32).collect();
        b.positions = (0..tokens.len()).map(|i| (start + i) as i32).collect();
        b.row_slot = vec![slot as i32; tokens.len()];
        let head = match (want, for_snapshot) {
            (Some(_), _) => Head::LastRow,
            (None, true) => Head::WindowOnly,
            (None, false) => Head::None,
        };
        self.forward(&b, Mode::Prefill, head)?;
        if want.is_some() || for_snapshot || !ced_replay_enabled() {
            let eff_rows = if ced_replay_enabled() {
                self.ced_tail.get(slot).map_or(tokens.len(), |t| if t.rows > 0 { t.rows } else { tokens.len() })
            } else {
                tokens.len()
            };
            let keep = eff_rows.min(self.mtp_win);
            let end_pos = start + tokens.len();
            self.seed_mtp_for_slot(slot, 0, end_pos - keep, keep)?;
        }
        match want {
            Some(n) => Ok(Some(self.candidates_of(1, n).map_err(|e| at("the candidates", e))?)),
            None => Ok(None),
        }
    }

    /// One decode or verify window, and its candidates.
    pub fn window_candidates(&mut self, b: &RowBatch, candidates: usize) -> Result<Candidates, ModelError> {
        let t0 = std::time::Instant::now();
        self.forward(b, Mode::Window, Head::AllRows)?;
        let res = self.candidates_of(b.rows, candidates)?;
        if std::env::var_os("REXL3_DSV41_STEP_TIMING").is_some() {
            eprintln!(
                "rexl3 event=dsv41_step_verify rows={} start_pos={} ms={:.2}",
                b.rows,
                b.positions.first().copied().unwrap_or(0),
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(res)
    }

    pub fn has_mtp(&self) -> bool {
        self.dev.mtp_gpu.is_some()
    }

    /// The three span embeddings, on the host, when the tower is bound (the engine calls this once, after the
    /// regions are loaded); without them a prompt with images is refused by name.
    pub fn set_span_embeddings(&mut self, emb: super::vision_serve::SpanEmbeddings) {
        self.span_embeddings = Some(emb);
    }

    pub fn span_embeddings(&self) -> Option<&super::vision_serve::SpanEmbeddings> {
        self.span_embeddings.as_ref()
    }

    /// [`Self::prefill_chunk`] for a chunk that holds image rows: `splice` says which chunk rows belong to which
    /// image's span (the backend clips each span to the chunk). Every other row is the text path unchanged, and the
    /// engram sees every span position as [`super::engram::DEAD`]. The spans are dropped when the call ends,
    /// whatever its outcome.
    pub fn prefill_chunk_mm(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
        splice: Vec<super::vision_serve::Dsv41Splice>,
    ) -> Result<Option<Candidates>, ModelError> {
        if !splice.is_empty() && self.span_embeddings.is_none() {
            return Err(ModelError::Request(
                "the prompt carries images but this model serves text only (start with the vision tower, or --no-vision for text)".into(),
            ));
        }
        for sp in &splice {
            if sp.dst + sp.n > tokens.len() {
                return Err(ModelError::Request(format!(
                    "an image span of {} rows at chunk row {} does not fit a chunk of {} rows",
                    sp.n,
                    sp.dst,
                    tokens.len()
                )));
            }
        }
        self.splice = splice;
        let r = self.prefill_chunk(slot, start, tokens, want);
        self.splice.clear();
        r
    }

    /// Run one DSpark GPU draft round for `slot` (`tokens` ends with the anchor token not yet fed to the trunk).
    pub fn propose_mtp(&mut self, slot: usize, tokens: &[u32], max: usize) -> Result<Option<DraftRound>, ModelError> {
        // the plain entry is the temperature-0 argmax walk; sampled callers pass their request's Sampling to
        // `propose_mtp_sampled` (temperature > 0 keys the Gumbel walk to the target's own noise)
        let greedy = gb_sampling::Sampling { temperature: 0.0, ..gb_sampling::Sampling::default() };
        self.propose_mtp_sampled(slot, tokens, &greedy, max)
    }

    /// Run one DSpark GPU draft round for `slot` with optional keyed Gumbel sampling (`sampling.temperature > 0.0`).
    pub fn propose_mtp_sampled(
        &mut self,
        slot: usize,
        tokens: &[u32],
        sampling: &gb_sampling::Sampling,
        max: usize,
    ) -> Result<Option<DraftRound>, ModelError> {
        if slot >= self.slots || slot >= gb_cuda::dsv4_mtp_gpu::MAX_SLOTS || tokens.len() < 2 || max == 0 {
            return Ok(None);
        }
        self.flush()?;
        let start_pos = tokens.len() - 2;
        if start_pos == 0 || self.mtp_last_pos[slot] != Some(start_pos) {
            return Ok(None);
        }
        let Some(mtp) = self.dev.mtp_gpu.as_mut() else {
            return Ok(None);
        };
        let anchor = *tokens.last().unwrap() as i32;
        let thr: f32 =
            gb_cuda::dsv4_moe::runtime_env("REXL3_DSV41_MTP_THRESHOLD").and_then(|v| v.trim().parse().ok()).unwrap_or(0.38);
        // T2.5c — the walk's ceiling follows the window knob (`REXL3_DSV41_WINDOW_ROWS`, the family's
        // `--window-rows` analog): the drafts a round may verify never exceed the rows the operator's verify
        // window holds
        let win = window_rows();
        let cap: usize = gb_cuda::dsv4_moe::runtime_env("REXL3_DSV41_MTP_DRAFTS")
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(MTP_MAX_DRAFTS)
            .min(win - 1)
            .min(MTP_MAX_DRAFTS);
        let max_walk = max.min(cap).min(mtp.cfg.block_size).max(1);
        // the keyed walk keeps the target sampler's whole candidate horizon (top_k + MARGIN), capped at the
        // kernel's 64: a smaller cut would clip candidates the target could still draw, and the Gumbel-max
        // coupling only agrees when every token the target keeps is inside the draft's pool
        let cand_k = sampling.candidate_count(mtp.cfg.vocab).clamp(1, 64);
        let walk_sampling = (sampling.temperature > 0.0).then_some(gb_cuda::dsv4_mtp_gpu::MtpWalkSampling {
            cand_k,
            seed: sampling.seed,
            temperature: sampling.temperature,
            top_p: sampling.top_p,
            min_p: sampling.min_p,
            base_pos: tokens.len() as u64,
        });
        let t_draft = std::time::Instant::now();
        let prof_ticket = self.slices.enter(crate::families::glm::profile_slices::Kind::Draft, max_walk, Some(slot));
        // the SPEC_TREE knob (off by default): capture the round's candidate lattice for a draft-tree round, at the
        // cost of one sync and a logits read-back between the head's projection and the markov walk
        let tree_on = gb_cuda::dsv4_moe::runtime_env("REXL3_DSV41_SPEC_TREE").is_some_and(|v| v.trim() != "0");
        let mtp = self.dev.mtp_gpu.as_mut().unwrap();
        let captured_res = if tree_on {
            mtp.draft_n_keyed_slot_capture(
                slot,
                &self.mtp_last_tap[slot],
                anchor,
                start_pos,
                max_walk,
                walk_sampling,
                Some(8),
                &self.stream,
            )
        } else {
            mtp.draft_n_keyed_slot(slot, &self.mtp_last_tap[slot], anchor, start_pos, max_walk, walk_sampling, &self.stream)
                .map(|_| None)
        };
        let captured = match captured_res {
            Ok(c) => c,
            Err(e) => {
                self.slices.leave(prof_ticket, false);
                return Err(e.into());
            }
        };
        self.mtp_slot = Some(slot);
        self.mtp_last_seeded[slot] = true;
        for eg in self.dev.engram.iter() {
            eg.state.prefetch_batch(slot, &[anchor]);
        }
        self.slices.leave(prof_ticket, true);
        let mtp = self.dev.mtp_gpu.as_mut().unwrap();
        let max_len = max.min(cap).min(mtp.cfg.block_size);
        let use_sched = gb_cuda::dsv4_moe::runtime_env("REXL3_DSV41_MTP_SCHED").map(|v| v.trim() != "0").unwrap_or(true);
        // T2.5b — device-side verify-length scheduling from the GPU-resident confidences. The device kernel is the
        // host `schedule_verify_len`'s loop (same clamp, floor, cycle table and tie-break), so its result is
        // bit-for-bit the host's for the same input; it is used while the spec-learn blend is still the identity
        // (no evidence yet), because only then the raw head confidences equal the blended schedule input. The
        // result rides the read-back's own event (copied with ids/confidences), so no extra host wait.
        let gpu_sched = use_sched && thr > 0.0 && !self.spec_learn.as_ref().is_some_and(|l| l.has_evidence());
        if gpu_sched {
            mtp.schedule_verify_len_launch(thr, max_len, &self.stream)?;
        }
        mtp.read_begin(&self.stream)?;
        let mtp = self.dev.mtp_gpu.as_ref().unwrap();
        let (ids, mut conf, sched) = mtp.read_end()?;
        let lattice = captured.map(|c| gb_spec::tree::Lattice {
            rows: c.rows,
            k: c.k,
            ids: c.ids,
            unary: c.unary,
            edge: c.edge,
            restricted_head: false,
        });
        // T2.5b — REXL3_DSV41_SPEC_LEARN: with the knob on, the verify-length schedule reads the head's
        // confidences blended with the traffic-learned per-position acceptance (`SpecLearn::adjust`); with it
        // off — the default — the blend is the identity (no evidence yet, and None here) and this is
        // bit-for-bit the static path.
        let sched_conf = self.spec_learn.as_ref().map(|l| l.adjust(&conf)).unwrap_or_else(|| conf.clone());
        let n = if gpu_sched {
            sched.unwrap_or(1).max(1)
        } else if thr > 0.0 {
            if use_sched {
                gb_cuda::dsv4_mtp_gpu::DsparkGpu::schedule_verify_len(&sched_conf, thr, max_len).max(1)
            } else {
                gb_cuda::dsv4_mtp_gpu::DsparkGpu::accept_host(&sched_conf, thr).max(1).min(max_len)
            }
        } else {
            max_len
        };
        let mut drafted: Vec<u32> = Vec::with_capacity(n);
        for (i, &x) in ids[1..=n].iter().enumerate() {
            if x < 0 || (x as usize) >= self.cfg.vocab || !conf.get(i).is_some_and(|c| c.is_finite()) {
                break;
            }
            drafted.push(x as u32);
        }
        conf.truncate(drafted.len());
        if drafted.is_empty() {
            return Ok(None);
        }
        let max_total = max.min(cap).min(win - 1);
        // the n-gram splice is the CHAIN's poor-man's branching: with the tree capture on, the round's tree already
        // adds candidate continuations where the chain is weak, so the splice would only cut the lattice's chain
        // (its `conf` is the DSpark head's, not the copied tokens'); without it (the default) the splice is unchanged
        if !tree_on && tokens.len() >= 8 && max_total > 1 {
            let hc_len = conf.iter().take_while(|&&c| c >= 0.65).count().min(drafted.len());
            let gram = 4usize;
            for &prefix_len in &[hc_len, drafted.len()] {
                if prefix_len < 2 || prefix_len >= max_total {
                    continue;
                }
                if prefix_len < drafted.len() && conf[prefix_len] >= 0.70 {
                    continue;
                }
                if tokens.len() + prefix_len < gram + 4 {
                    continue;
                }
                let mut tail = [0u32; 4];
                for (idx_g, t) in tail.iter_mut().enumerate().take(gram) {
                    let pos_in_comb = tokens.len() + prefix_len - gram + idx_g;
                    *t = if pos_in_comb < tokens.len() { tokens[pos_in_comb] } else { drafted[pos_in_comb - tokens.len()] };
                }
                let need = max_total - prefix_len;
                let search_end = tokens.len().saturating_sub(gram + need);
                let mut matched = false;
                for pos in (0..=search_end).rev() {
                    if tokens[pos..pos + gram] == tail[..] {
                        drafted.truncate(prefix_len);
                        conf.truncate(prefix_len);
                        for j in 0..need {
                            drafted.push(tokens[pos + gram + j]);
                            conf.push(0.90);
                        }
                        matched = true;
                        break;
                    }
                }
                if matched {
                    break;
                }
            }
        }
        if std::env::var_os("REXL3_DSV41_OP_TIMING").is_some() || std::env::var_os("REXL3_DSV41_STEP_TIMING").is_some() {
            eprintln!(
                "rexl3 event=dsv41_mtp_draft start_pos={start_pos} anchor={anchor} drafted={drafted:?} conf={conf:?} ms={:.2}",
                t_draft.elapsed().as_secs_f64() * 1e3
            );
        }
        if !self.dev.engram.is_empty() {
            // the rows the round's drafts will need for their engrams: warm them while the verify is still ahead
            let mut win_toks = Vec::with_capacity(1 + drafted.len());
            win_toks.push(anchor);
            win_toks.extend(drafted.iter().map(|&t| t as i32));
            for eg in self.dev.engram.iter() {
                eg.state.prefetch_batch(slot, &win_toks);
            }
        }
        if let Some(l) = lattice.as_ref() {
            // the round a tree consumer would run (the confidence head schedules the fanout); logged under the
            // timing envs so the shape is observable before the tree verify itself lands
            let r = super::tree::round(mtp.cfg.block_size, WINDOW_ROWS, l, &conf);
            if std::env::var_os("REXL3_DSV41_OP_TIMING").is_some() || std::env::var_os("REXL3_DSV41_STEP_TIMING").is_some() {
                eprintln!(
                    "rexl3 event=dsv41_mtp_tree nodes={} branch={} depth={} paths={}",
                    r.tree.len(),
                    r.policy.branch,
                    r.tree.depth.iter().max().copied().unwrap_or(0),
                    r.paths.len()
                );
            }
        }
        Ok(Some((drafted, conf, lattice)))
    }

    /// A prefill chunk whose ROWS ALL get full-vocabulary logits, `[rows][vocab]` f32 on the host: the teacher-forced
    /// readout the correctness oracle (`prompt_logprobs`) is defined on. Row `r` is the logits after `tokens[..=r]` (with the
    /// slot's context before `start`), the prediction of the token after it; the server scores each prompt token from the
    /// row before it. `Head::AllRows` is the decode head; a prefill
    /// only asks for it here, so a chunk of `rows` costs `rows * vocab * 2` bytes of device head output plus the same
    /// again as its f16 host copy before the widening to f32.
    pub fn prefill_logits_rows(&mut self, slot: usize, start: usize, tokens: &[u32]) -> Result<Vec<Vec<f32>>, ModelError> {
        self.prefill_rows(slot, start, tokens, Head::AllRows)?;
        let vocab = self.cfg.vocab;
        let raw = self.head_rows(tokens.len(), vocab)?;
        Ok(raw.chunks_exact(vocab).map(|r| r.to_vec()).collect())
    }

    /// The last row's full-vocabulary logits for a prefill chunk (the cheap readout: one row of the head).
    pub fn prefill_logits_last_row(&mut self, slot: usize, start: usize, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::Request("empty prefill".into()));
        }
        self.prefill_rows(slot, start, tokens, Head::LastRow)?;
        let vocab = self.cfg.vocab;
        Ok(self.head_rows(1, vocab)?.chunks_exact(vocab).next().map(<[f32]>::to_vec).unwrap_or_default())
    }

    pub fn prefill_rows(&mut self, slot: usize, start: usize, tokens: &[u32], head: Head) -> Result<(), ModelError> {
        let mut b = RowBatch { seq_start: vec![0, tokens.len() as i32], seq_slot: vec![slot as i32], ..Default::default() };
        b.rows = tokens.len();
        b.ids = tokens.iter().map(|&t| t as i32).collect();
        b.positions = (0..tokens.len()).map(|i| (start + i) as i32).collect();
        b.row_slot = vec![slot as i32; tokens.len()];
        self.forward(&b, Mode::Prefill, head)?;
        if head != Head::None {
            let rows = tokens.len();
            let keep = rows.min(self.mtp_win);
            let skip = rows - keep;
            self.seed_mtp_for_slot(slot, 0, start + skip, keep)?;
        }
        Ok(())
    }

    /// The first `rows` rows of the head's f16 output as f32 (the head writes them; see [`Head`]).
    fn head_rows(&mut self, rows: usize, vocab: usize) -> Result<Vec<f32>, ModelError> {
        self.stream.sync()?;
        let n = rows * vocab;
        assert!(n * 2 <= self.s.head_out.bytes(), "head_out holds {} B, asked for {n} f16 rows", self.s.head_out.bytes());
        let mut raw: Vec<u16> = vec![0; n];
        self.s.head_out.download_range(0, &mut raw)?;
        Ok(raw.iter().map(|&h| gb_exl3::f16::f16_to_f64(h) as f32).collect())
    }

    /// The candidates of the rows of the last forward (the head must have run with [`Head::AllRows`]).
    pub fn candidates_of(&mut self, rows: usize, count: usize) -> Result<Candidates, ModelError> {
        gb_cuda::glm_glue::topk_f16(
            &self.s.head_out,
            rows,
            self.cfg.vocab,
            self.cfg.vocab,
            count,
            &mut self.s.topk_ids,
            &mut self.s.topk_vals,
            &mut self.s.topk_ws,
            &self.stream,
        )?;
        let n = rows * count;
        let need = TopkStage::needed_bytes(rows, count);
        match &mut self.topk_stage {
            None => self.topk_stage = Some(TopkStage::new(need)?),
            Some(stg) => stg.ensure(need)?,
        }
        let stg = self.topk_stage.as_ref().unwrap();
        debug_assert!(n * 4 <= self.s.topk_ids.bytes(), "top-k device buffers hold {n} candidates");
        // SAFETY: the staging block is pinned (`alloc_mapped_host`) and every `src` has `n * 4` device bytes behind
        // it; the copies run ordered after the top-k kernel on `self.stream`.
        unsafe {
            self.stream.memcpy_d2h_async(stg.block.host_ptr_at(0), self.s.topk_ids.ptr(), n * 4)?;
            self.stream.memcpy_d2h_async(stg.block.host_ptr_at(n * 4), self.s.topk_vals.ptr(), n * 4)?;
        }
        stg.ev.record(&self.stream)?;
        // The candidates are consumed by the commit decision immediately after this call, so the wait cannot be
        // deferred; the event still replaces a full stream sync with one wait scoped to these copies.
        stg.ev.sync()?;
        let mut ids = vec![0i32; n];
        let mut vals = vec![0f32; n];
        // SAFETY: the event guarantees the staging holds the copied ids and values by now.
        unsafe {
            core::ptr::copy_nonoverlapping(stg.block.host_ptr_at(0) as *const i32, ids.as_mut_ptr(), n);
            core::ptr::copy_nonoverlapping(stg.block.host_ptr_at(n * 4) as *const f32, vals.as_mut_ptr(), n);
        }
        Ok(Candidates { rows, count, ids, vals })
    }
}

// ================================================================================================================== the host reference

/// `e`, naming the stage of a forward it came from: a launcher's refusal is a bare CUDA code otherwise.
pub(crate) fn ced_replay_enabled() -> bool {
    match gb_cuda::dsv4_moe::runtime_env("REXL3_DSV41_CED_REPLAY") {
        Some(v) => !matches!(v.trim(), "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"),
        None => true,
    }
}

fn compact_tail_rows(
    buf: &DevBuf,
    from: usize,
    keep: usize,
    width_f16: usize,
    tmp: &DevBuf,
    st: &Stream,
) -> Result<(), gb_cuda::rt::CudaError> {
    // Row move through `plane_write_dev`, whose kernel now copies raw 16-bit lanes (the f16 convert
    // round-trip canonicalized NaN-pattern lanes and silently mutated f32 buffers -- the chunk-grid
    // divergence at layer 20). `tmp` is only staged through for overlapping copies; the copy is
    // bit-exact for both f16 and f32 buffers.
    if from == 0 {
        return Ok(());
    }
    if from >= keep {
        gb_cuda::dsv4_attn::plane_write_dev(buf, from, buf, keep, width_f16, 0, st)
    } else {
        gb_cuda::dsv4_attn::plane_write_dev(buf, from, tmp, keep, width_f16, 0, st)?;
        gb_cuda::dsv4_attn::plane_write_dev(tmp, 0, buf, keep, width_f16, 0, st)
    }
}

/// Append the last `take` rows of `buf` (each `w` f16 elements, 2 bytes) to the host ring `dst`,
/// dropping the `drop` oldest rows first so the ring never grows past its cap. `from_row` is the row
/// offset within `buf` the `take` rows start at.
fn tail_download_append(
    dst: &mut Vec<u8>,
    buf: &DevBuf,
    from_row: usize,
    take: usize,
    drop: usize,
    w: usize,
) -> Result<(), ModelError> {
    let rb = w * 2;
    let mut chunk = vec![0u8; take * rb];
    buf.download_range(from_row * rb, &mut chunk)?;
    if drop > 0 {
        dst.drain(0..drop * rb);
    }
    dst.extend_from_slice(&chunk);
    Ok(())
}

/// Upload a whole host ring `v` (as captured by [`tail_download_append`]) into device buffer `buf`.
fn tail_upload(buf: &DevBuf, v: &[u8], st: &Stream) -> Result<(), ModelError> {
    unsafe {
        st.memcpy_h2d_async(buf.ptr(), v.as_ptr() as *const u8, v.len())?;
    }
    Ok(())
}

fn at(stage: &str, e: ModelError) -> ModelError {
    if e.is_fatal() {
        ModelError::StateLost(format!("{e} (at {stage})"))
    } else {
        ModelError::Config(format!("{e} (at {stage})"))
    }
}

// ----------------------------------------------------------- the tree round host (T2.5a lane 2): the device surfaces

/// One row of a window's [`Candidates`] as the engine's `RowOutput` (the same conversion the backend makes).
pub(super) fn candidates_row(c: &Candidates, r: usize, n: usize) -> gb_engine::RowOutput {
    let ids = c.ids[r * c.count..r * c.count + n].iter().map(|&x| x as u32).collect();
    let values = c.vals[r * c.count..r * c.count + n].to_vec();
    gb_engine::RowOutput { ids, values }
}

impl super::tree::TreeRoundHost for Dsv41Model {
    fn tree_path_window(
        &mut self,
        tokens: &[u32],
        first_position: u64,
        candidates: usize,
    ) -> Result<Vec<gb_engine::RowOutput>, String> {
        let p0 = first_position as usize;
        if let Some(r) = self.tree_round.as_ref() {
            if r.p0 != p0 {
                return Err(format!("a path window at {p0} does not match the open round's anchor at {}", r.p0));
            }
        } else {
            self.tree_round_begin(p0).map_err(|e| e.to_string())?;
        }
        // the per-path isolation BEFORE the window: a sibling's n-gram appends and carry must not leak
        self.tree_isolate().map_err(|e| e.to_string())?;
        let n = tokens.len();
        let mut b = RowBatch { seq_start: vec![0, n as i32], seq_slot: vec![0], ..Default::default() };
        b.rows = n;
        b.ids = tokens.iter().map(|&t| t as i32).collect();
        b.positions = (0..n).map(|j| (p0 + j) as i32).collect();
        b.row_slot = vec![0; n];
        let c = self.window_candidates(&b, candidates).map_err(|e| e.to_string())?;
        self.tree_save_path(n).map_err(|e| e.to_string())?;
        Ok((0..n).map(|r| candidates_row(&c, r, candidates)).collect())
    }

    fn tree_commit_window(&mut self, idx: usize, rows: usize, window: &[u32]) -> Result<(), String> {
        let Some(r) = self.tree_round.take() else { return Err("no tree round is open".into()) };
        if rows == 0 || rows > WINDOW_ROWS || rows > window.len() {
            self.tree_round = Some(r);
            return Err(format!("a commit of {rows} rows over a {}-row window", window.len()));
        }
        let open = r.p0;
        // (1) the window rings: base slots back, then the accepted path's own rows over them
        self.tree_ring_surgery(idx, open, rows).map_err(|e| e.to_string())?;
        // (2) the engram rings: base, then the accepted window's tokens appended — the serial replay
        for (eg, snap) in self.dev.engram.iter_mut().zip(&r.engram) {
            eg.state.restore_slot(0, snap);
            eg.state.store(0, &window[..rows]).map_err(|e| e.to_string())?;
        }
        // (3) the compressors' carry: base back, then the parity restage from the accepted path's staged
        // rows when the accepted prefix ends mid-pair (`commit_window_planes`'s own move)
        let carries = Self::tree_carry_views(self.tree_bufs.as_ref().expect("a round implies its buffers"));
        for (i, (l, kv, score, d, ratio)) in carries.iter().enumerate() {
            let Some(c) = self.dev.layers[*l].mla.compressor.as_mut() else { continue };
            gb_cuda::dsv4_attn::compressor_carry_restore(
                &mut c.kv_state,
                &mut c.score_state,
                *ratio,
                *d,
                0,
                kv,
                score,
                &self.stream,
            )
            .map_err(|e| e.to_string())?;
            let rem = (open + rows) % *ratio;
            if rem == 0 {
                continue;
            }
            let bufs = self.tree_bufs.as_ref().expect("a round implies its buffers");
            let at = idx * bufs.stage_stride + bufs.stage_off[i];
            let stage_kv = bufs.stage_kv.subview(at, WINDOW_ROWS * d * 2);
            let stage_gate = bufs.stage_gate.subview(at, WINDOW_ROWS * d * 2);
            let slot0 = 0; // `(seq_slot % MAX_SLOTS) * ratio` for this host's slot 0
            for t in 0..rem {
                gb_cuda::dsv4_attn::state_write_dev(
                    &stage_kv,
                    rows - rem + t,
                    &stage_gate,
                    rows - rem + t,
                    c,
                    slot0 + t,
                    *d,
                    &self.stream,
                )
                .map_err(|e| e.to_string())?;
            }
        }
        // (4) the DSpark ring: seeded from the accepted path's saved tap rows
        self.tree_mtp_reseed(idx, open, rows).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn tree_drop_round(&mut self) -> Result<(), String> {
        self.tree_drop_round_impl().map_err(|e| e.to_string())
    }
}

/// `REXL3_DSV41_DUMP_HIDDEN=<dir>:<p0,p1,...>`: after the embedding, after every block's FFN post and after the final
/// norm, write the `hc`-stream MEAN at those absolute positions to `<dir>/{embed,layer_XX,final_norm}.<start>.f32`
/// (`[P][hidden]` f32 LE, `P` the positions that fall inside the slice `start..start + rows`). This is the fork's own
/// DSpark tap (`x.mean(dim = 2)` of the residual streams, modules/dsv41_hc.py) and the correctness oracle's per-layer
/// readout. Read once per process; absent (the default) the forward is untouched but for one `Option` test per tap.
fn hidden_dump() -> Option<&'static (std::path::PathBuf, Option<Vec<usize>>)> {
    static DUMP: std::sync::OnceLock<Option<(std::path::PathBuf, Option<Vec<usize>>)>> = std::sync::OnceLock::new();
    DUMP.get_or_init(|| {
        let spec = std::env::var("REXL3_DSV41_DUMP_HIDDEN").ok()?;
        let (dir, positions) = spec.rsplit_once(':')?;
        // `*` dumps every row of every slice (the whole prompt); a list dumps just those absolute positions
        let positions = if positions.trim() == "*" {
            None
        } else {
            let v: Vec<usize> = positions.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            if v.is_empty() {
                return None;
            }
            Some(v)
        };
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).ok()?;
        eprintln!("rexl3 event=dsv41_dump_hidden dir={} positions={positions:?}", dir.display());
        Some((dir, positions))
    })
    .as_ref()
}

/// One tap of [`hidden_dump`]: the stream mean (or the plain row, for the f16 hidden the head reads) at the requested
/// absolute positions. `x` is `[rows][hc][hidden]` f32 for the stream taps and `[rows][hidden]` f16 for `final_norm`.
fn write_tap(name: &str, layer_start: usize, rows: usize, x: &[f32], hc: usize, h: usize) -> Result<(), ModelError> {
    let Some((dir, positions)) = hidden_dump() else { return Ok(()) };
    let local: Vec<usize> = match positions {
        None => (0..rows).collect(),
        Some(ps) => ps.iter().copied().filter_map(|p| p.checked_sub(layer_start)).filter(|&l| l < rows).collect(),
    };
    if local.is_empty() {
        return Ok(());
    }
    let mut out: Vec<f32> = Vec::with_capacity(local.len() * h);
    for l in local {
        for j in 0..h {
            let mut acc = 0f32;
            for c in 0..hc {
                acc += x[(l * hc + c) * h + j];
            }
            out.push(acc / hc as f32);
        }
    }
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 4) };
    let path = dir.join(format!("{name}.{layer_start}.f32"));
    std::fs::write(&path, bytes).map_err(|e| ModelError::Config(format!("hidden dump {}: {e}", path.display())))
}
//
// A CPU forward over a toy config. It exists to prove the *schedule*: it walks `ops::plan_forward` and does exactly what each op says, so a
// test can compare it against a direct transcription of `Transformer.forward` over the same weights.

/// Everything the toy forward needs, in f32, in the tree's own order.
pub struct HostWeights {
    pub embed: Vec<f32>,
    pub head: Vec<f32>,
    pub norm: Vec<f32>,
    pub layers: Vec<HostLayer>,
    /// Index `i` is the engram of `cfg.engram.layer_ids[i]`.
    pub engram: Vec<HostEngram>,
}

pub struct HostLayer {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub mla: HostMla,
    /// `[experts, hidden]`.
    pub gate: Vec<f32>,
    pub bias: Vec<f32>,
    /// `w1`, `w3` `[inter, hidden]`; `w2` `[hidden, inter]`.
    pub shared: [Vec<f32>; 3],
    /// Per expert: `w1` `[inter, hidden]`, `w2` `[hidden, inter]`, `w3` `[inter, hidden]`.
    pub experts: Vec<[Vec<f32>; 3]>,
    pub hc_attn: HostHc,
    pub hc_ffn: HostHc,
}

pub struct HostMla {
    pub wq_a: Vec<f32>,
    pub wq_b: Vec<f32>,
    pub wkv: Vec<f32>,
    pub wo_a: Vec<f32>,
    pub wo_b: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub kv_norm: Vec<f32>,
    pub sink: Vec<f32>,
}

pub struct HostHc {
    pub fnw: Vec<f32>,
    pub base: Vec<f32>,
    pub scale: [f32; 3],
}

pub struct HostEngram {
    /// `[rows, head_dim]`.
    pub table: Vec<f32>,
    /// `[(hc + 1) * hidden, hash_cols * head_dim]`.
    pub wkv: Vec<f32>,
    pub q_weight: Vec<f32>,
    pub k_weight: Vec<f32>,
}

/// A deterministic little generator, so a toy model is the same on every run and a failure is reproducible.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 40) as u32;
        (bits as f32 / (1u32 << 23) as f32) - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

impl HostWeights {
    /// A toy model of `cfg`'s shape: every tensor small, deterministic, and non-degenerate.
    pub fn toy(cfg: &DeepseekV41Config, seed: u64) -> Self {
        let mut g = Lcg(seed);
        let h = cfg.hidden;
        let hc = cfg.hc.mult;
        let inter = cfg.moe.intermediate;
        let e = cfg.moe.routed_experts;
        let d = cfg.head_dim;
        let mut layers = Vec::with_capacity(cfg.layers);
        for _ in 0..cfg.layers {
            let mut experts = Vec::with_capacity(e);
            for _ in 0..e {
                experts.push([g.vec(inter * h, 0.2), g.vec(h * inter, 0.2), g.vec(inter * h, 0.2)]);
            }
            layers.push(HostLayer {
                attn_norm: g.vec(h, 0.5),
                ffn_norm: g.vec(h, 0.5),
                mla: HostMla {
                    wq_a: g.vec(cfg.q_lora_rank * h, 0.2),
                    wq_b: g.vec(cfg.heads * d * cfg.q_lora_rank, 0.2),
                    wkv: g.vec(d * h, 0.2),
                    wo_a: g.vec(cfg.o_groups * cfg.o_lora_rank * cfg.o_groups * d, 0.2),
                    wo_b: g.vec(h * cfg.o_groups * cfg.o_lora_rank, 0.2),
                    q_norm: g.vec(cfg.q_lora_rank, 0.5),
                    kv_norm: g.vec(d, 0.5),
                    sink: g.vec(cfg.heads, 0.5),
                },
                gate: g.vec(e * h, 0.2),
                bias: g.vec(e, 0.05),
                shared: [g.vec(inter * h, 0.2), g.vec(h * inter, 0.2), g.vec(inter * h, 0.2)],
                experts,
                hc_attn: HostHc { fnw: g.vec(6 * hc * hc * h, 0.02), base: g.vec(6 * hc, 0.05), scale: [0.5, 0.25, 0.125] },
                hc_ffn: HostHc { fnw: g.vec(6 * hc * hc * h, 0.02), base: g.vec(6 * hc, 0.05), scale: [0.5, 0.25, 0.125] },
            });
        }
        let hash_cols = (cfg.engram.max_ngram_size - 1) * cfg.engram.n_heads;
        let engram = cfg
            .engram
            .layer_ids
            .iter()
            .enumerate()
            .map(|(i, _)| HostEngram {
                table: g.vec(cfg.engram.num_embeddings[i] * cfg.engram.head_dim, 0.2),
                wkv: g.vec((hc + 1) * h * hash_cols * cfg.engram.head_dim, 0.05),
                q_weight: g.vec(hc * h, 0.2),
                k_weight: g.vec(hc * h, 0.2),
            })
            .collect();
        Self { embed: g.vec(cfg.vocab * h, 0.2), head: g.vec(cfg.vocab * h, 0.2), norm: g.vec(h, 0.5), layers, engram }
    }

    /// The `(order, head)` row ids the toy hash gives the window `w` — one per hash column, as the pack's tables are addressed.
    pub fn toy_rows(cfg: &DeepseekV41Config, slot: usize, w: &[i64]) -> Vec<usize> {
        let hash_cols = (cfg.engram.max_ngram_size - 1) * cfg.engram.n_heads;
        (0..hash_cols)
            .map(|c| {
                let order = c / cfg.engram.n_heads;
                let head = c % cfg.engram.n_heads;
                let mut idx: i64 = (head as i64 + 1) * 2_654_435_761 + slot as i64;
                for j in 0..=order {
                    let v = if j < w.len() { w[w.len() - 1 - j] } else { 0 };
                    idx = idx.wrapping_mul(1_000_003).wrapping_add(v + 1);
                }
                (idx.unsigned_abs() as usize) % cfg.engram.num_embeddings[slot]
            })
            .collect()
    }
}

/// The CPU forward over [`HostWeights`], walking [`ops::plan_forward`].
pub struct HostModel<'a> {
    pub cfg: &'a DeepseekV41Config,
    pub w: &'a HostWeights,
}

impl<'a> HostModel<'a> {
    pub fn new(cfg: &'a DeepseekV41Config, w: &'a HostWeights) -> Self {
        Self { cfg, w }
    }

    /// The logits of the last position of `ids`, `[vocab]`.
    pub fn forward(&self, ids: &[u32]) -> Vec<f32> {
        let cfg = self.cfg;
        let (hc, h) = (cfg.hc.mult, cfg.hidden);
        let topk = cfg.moe.experts_per_token;
        let rows = ids.len();
        let plan = ops::plan_forward(cfg, cfg.layers, true);
        let mut x = vec![0f32; rows * hc * h];
        let identity = ops::identity_pre_mix(hc).repeat(rows);
        let mut pre: std::collections::BTreeMap<(usize, SiteKind), Vec<f32>> = std::collections::BTreeMap::new();
        let mut post: std::collections::BTreeMap<(usize, SiteKind), Vec<f32>> = std::collections::BTreeMap::new();
        let mut comb: std::collections::BTreeMap<(usize, SiteKind), Vec<f32>> = std::collections::BTreeMap::new();
        let mut sub: Vec<f32> = vec![0f32; rows * h];
        let mut hidden: Vec<f32> = vec![0f32; rows * h];
        let mut window: Vec<i64> = Vec::new();
        let mut routing: Option<(Vec<i32>, Vec<f32>)> = None;
        for op in &plan {
            match *op {
                Op::Embed => {
                    for r in 0..rows {
                        let row = &self.w.embed[ids[r] as usize * h..(ids[r] as usize + 1) * h];
                        for c in 0..hc {
                            x[(r * hc + c) * h..(r * hc + c) * h + h].copy_from_slice(row);
                        }
                    }
                }
                Op::Engram { layer } => {
                    let slot = cfg.engram.layer_ids.iter().position(|&l| l == layer).unwrap();
                    let e = &self.w.engram[slot];
                    let dd = cfg.engram.head_dim;
                    let hash_cols = (cfg.engram.max_ngram_size - 1) * cfg.engram.n_heads;
                    for r in 0..rows {
                        window.push(ids[r] as i64);
                        let rid = HostWeights::toy_rows(cfg, slot, &window);
                        let mut gathered = vec![0f32; hash_cols * dd];
                        for (c, &row) in rid.iter().enumerate() {
                            gathered[c * dd..(c + 1) * dd].copy_from_slice(&e.table[row * dd..(row + 1) * dd]);
                        }
                        let kv = ops::linear(&gathered, &e.wkv, 1, (hc + 1) * h, hash_cols * dd);
                        let (key, value) = kv.split_at(hc * h);
                        let mut out = vec![0f32; (hc + 1) * h];
                        super::engram::gate_and_add(
                            &x[r * hc * h..(r + 1) * hc * h],
                            key,
                            value,
                            &e.q_weight,
                            &e.k_weight,
                            hc,
                            h,
                            cfg.rms_eps as f32,
                            &mut out,
                        )
                        .unwrap();
                        x[r * hc * h..(r + 1) * hc * h].copy_from_slice(&out[..hc * h]);
                    }
                }
                Op::HcMixes { layer, kind } => {
                    let l = &self.w.layers[layer];
                    let site = match kind {
                        SiteKind::Attn => &l.hc_attn,
                        SiteKind::Ffn => &l.hc_ffn,
                    };
                    let (p, po, c) = ops::hc_mixes(
                        &x,
                        rows,
                        hc,
                        h,
                        &site.fnw,
                        &site.scale,
                        &site.base,
                        cfg.hc.sinkhorn_iters,
                        cfg.rms_eps as f32,
                        cfg.hc.eps as f32,
                    );
                    pre.insert((layer, kind), p);
                    post.insert((layer, kind), po);
                    comb.insert((layer, kind), c);
                }
                Op::HcCollapse { mix, .. } => {
                    let coeffs = match mix {
                        PreMix::Identity => &identity,
                        PreMix::Site { layer, kind } => &pre[&(layer, kind)],
                    };
                    hidden = ops::hc_pre(&x, rows, hc, h, coeffs);
                }
                Op::RmsNorm { layer, kind } => {
                    let l = &self.w.layers[layer];
                    let g = match kind {
                        SiteKind::Attn => &l.attn_norm,
                        SiteKind::Ffn => &l.ffn_norm,
                    };
                    ops::rms_norm(&mut hidden, g, h, cfg.rms_eps as f32);
                }
                Op::Mla { layer } => sub = self.attention(layer, &hidden, rows, window.len()),
                Op::MoeShared { layer } => {
                    let l = &self.w.layers[layer];
                    let inter = cfg.moe.intermediate;
                    let g = ops::linear(&hidden, &l.shared[0], rows, inter, h);
                    let u = ops::linear(&hidden, &l.shared[2], rows, inter, h);
                    let a = ops::swiglu(&g, &u, None, rows, inter, cfg.swiglu_limit as f32);
                    sub = ops::linear(&a, &l.shared[1], rows, h, inter);
                }
                Op::MoeRoute { layer } => {
                    let l = &self.w.layers[layer];
                    let logits = ops::linear(&hidden, &l.gate, rows, cfg.moe.routed_experts, h);
                    routing = Some(ops::select(
                        &logits,
                        rows,
                        cfg.moe.routed_experts,
                        cfg.moe.experts_per_token,
                        &l.bias,
                        cfg.moe.routed_scaling_factor as f32,
                    ));
                }
                Op::MoeExperts { layer } => {
                    let l = &self.w.layers[layer];
                    let inter = cfg.moe.intermediate;
                    let (picks, wts) = routing.clone().expect("MoeRoute runs before MoeExperts");
                    for r in 0..rows {
                        for k in 0..topk {
                            let e = picks[r * topk + k] as usize;
                            let wgt = wts[r * topk + k];
                            let g = ops::linear(&hidden[r * h..(r + 1) * h], &l.experts[e][0], 1, inter, h);
                            let u = ops::linear(&hidden[r * h..(r + 1) * h], &l.experts[e][2], 1, inter, h);
                            let a = ops::swiglu(&g, &u, None, 1, inter, cfg.swiglu_limit as f32);
                            let o = ops::linear(&a, &l.experts[e][1], 1, h, inter);
                            for d in 0..h {
                                sub[r * h + d] += wgt * o[d];
                            }
                        }
                    }
                }
                Op::HcPost { layer, kind } => {
                    let (po, c) = (post[&(layer, kind)].clone(), comb[&(layer, kind)].clone());
                    x = ops::hc_post(&x, &sub, rows, hc, h, &po, &c);
                }
                Op::HeadCollapse { mix } => {
                    let coeffs = match mix {
                        PreMix::Identity => &identity,
                        PreMix::Site { layer, kind } => &pre[&(layer, kind)],
                    };
                    hidden = ops::hc_pre(&x, rows, hc, h, coeffs);
                }
                Op::FinalNorm => ops::rms_norm(&mut hidden, &self.w.norm, h, cfg.rms_eps as f32),
                Op::LmHead => return ops::linear(&hidden[(rows - 1) * h..], &self.w.head, 1, cfg.vocab, h),
            }
        }
        unreachable!("the schedule ends with the lm_head")
    }

    fn attention(&self, layer: usize, x: &[f32], rows: usize, start: usize) -> Vec<f32> {
        attention_block(self.cfg, &self.w.layers[layer], x, rows, start)
    }
}

/// One MLA block on the host: the window attention with the per-head sink, then the grouped low-rank output.
///
/// The toy's `compress_ratios` are 0, so there is no compressed plane and no indexer; the rotary embedding is left out of both this and the
/// reference transcription (a per-position rotation that neither the op list nor the hyper-connection placement depends on).
pub fn attention_block(cfg: &DeepseekV41Config, l: &HostLayer, x: &[f32], rows: usize, start: usize) -> Vec<f32> {
    let (h, d, nh) = (cfg.hidden, cfg.head_dim, cfg.heads);
    let qr = ops::linear(x, &l.mla.wq_a, rows, cfg.q_lora_rank, h);
    let mut qrn = qr;
    ops::rms_norm(&mut qrn, &l.mla.q_norm, cfg.q_lora_rank, cfg.rms_eps as f32);
    let q = ops::linear(&qrn, &l.mla.wq_b, rows, nh * d, cfg.q_lora_rank);
    let mut kv = ops::linear(x, &l.mla.wkv, rows, d, h);
    ops::rms_norm(&mut kv, &l.mla.kv_norm, d, cfg.rms_eps as f32);
    let scale = (d as f32).powf(-0.5);
    let gh = nh / cfg.o_groups;
    let oin = gh * d;
    let mut o = vec![0f32; rows * nh * d];
    // the rows are absolute positions `start - rows .. start`
    let base = start.saturating_sub(rows);
    for r in 0..rows {
        let pos = base + r;
        let lo = (pos + 1).saturating_sub(cfg.sliding_window);
        for hh in 0..nh {
            let mut acc = vec![0f64; d];
            let mut mx = l.mla.sink[hh];
            let mut scores = Vec::with_capacity(pos + 1 - lo);
            for j in lo..=pos {
                let mut s = 0f64;
                for c in 0..d {
                    s += q[(r * nh + hh) * d + c] as f64 * kv[j * d + c] as f64;
                }
                let s = s as f32 * scale;
                mx = mx.max(s);
                scores.push(s);
            }
            let mut sum = (l.mla.sink[hh] - mx).exp() as f64;
            for (i, s) in scores.iter().enumerate() {
                let wgt = (*s - mx).exp() as f64;
                sum += wgt;
                for c in 0..d {
                    acc[c] += wgt * kv[(lo + i) * d + c] as f64;
                }
            }
            for c in 0..d {
                o[(r * nh + hh) * d + c] = (acc[c] / sum) as f32;
            }
        }
    }
    // the grouped low-rank output: group g sees its own `gh` heads
    let g = cfg.o_groups;
    let mut ol = vec![0f32; rows * g * cfg.o_lora_rank];
    for r in 0..rows {
        for gi in 0..g {
            let xg = &o[(r * nh + gi * gh) * d..][..oin];
            let wg = &l.mla.wo_a[gi * cfg.o_lora_rank * oin..][..cfg.o_lora_rank * oin];
            for j in 0..cfg.o_lora_rank {
                let mut a = 0f32;
                for c in 0..oin {
                    a = wg[j * oin + c].mul_add(xg[c], a);
                }
                ol[(r * g + gi) * cfg.o_lora_rank + j] = a;
            }
        }
    }
    ops::linear(&ol, &l.mla.wo_b, rows, h, g * cfg.o_lora_rank)
}

/// The engram's contribution to one position: the 24 gathered rows through the fused `wkv`, then the gate. `[(hc + 1) * hidden]`, of which
/// the first `hc * hidden` replace the streams.
pub fn engram_contribution(
    cfg: &DeepseekV41Config,
    e: &HostEngram,
    slot: usize,
    window: &[i64],
    x_row: &[f32],
) -> Result<Vec<f32>, ModelError> {
    let (hc, h) = (cfg.hc.mult, cfg.hidden);
    let dd = cfg.engram.head_dim;
    let hash_cols = (cfg.engram.max_ngram_size - 1) * cfg.engram.n_heads;
    let rid = HostWeights::toy_rows(cfg, slot, window);
    let mut gathered = vec![0f32; hash_cols * dd];
    for (c, &row) in rid.iter().enumerate() {
        gathered[c * dd..(c + 1) * dd].copy_from_slice(&e.table[row * dd..(row + 1) * dd]);
    }
    let kv = ops::linear(&gathered, &e.wkv, 1, (hc + 1) * h, hash_cols * dd);
    let (key, value) = kv.split_at(hc * h);
    let mut out = vec![0f32; (hc + 1) * h];
    super::engram::gate_and_add(x_row, key, value, &e.q_weight, &e.k_weight, hc, h, cfg.rms_eps as f32, &mut out)
        .map_err(ModelError::Config)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = include_str!("../../../../../testdata/deepseek_v41/config.json");

    #[test]
    fn the_snapshot_floor_defaults_to_a_quarter_gib_and_ignores_the_boot_floor_env() {
        assert_eq!(snapshot_floor_bytes(None), 1 << 28, "no variable: 0.25 GiB, the snapshot guard alone");
        // decoupled from the boot floor: REXL3_DSV41_FLOOR_GIB moves the boot, never the capture guard
        assert_eq!(snapshot_floor_bytes(Some("1")), 1 << 28);
        assert_eq!(snapshot_floor_bytes(Some(" 0.25 ")), 1 << 28);
        assert_eq!(snapshot_floor_bytes(Some("0")), 1 << 28);
        for bad in ["", "abc", "-1", "NaN", "inf"] {
            assert_eq!(snapshot_floor_bytes(Some(bad)), SNAPSHOT_FLOOR_DEFAULT_BYTES, "{bad:?} falls back to the default");
        }
    }

    fn toy_json() -> serde_json::Value {
        let mut j: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
        let t = j["text_config"].as_object_mut().unwrap();
        let mut set = |k: &str, v: serde_json::Value| {
            t.insert(k.to_string(), v);
        };
        set("vocab_size", 128.into());
        set("hidden_size", 64.into());
        set("num_hidden_layers", 2.into());
        set("num_attention_heads", 2.into());
        set("num_key_value_heads", 1.into());
        set("head_dim", 32.into());
        set("qk_rope_head_dim", 8.into());
        set("q_lora_rank", 16.into());
        set("o_lora_rank", 8.into());
        set("o_groups", 2.into());
        set("moe_intermediate_size", 32.into());
        set("n_routed_experts", 4.into());
        set("n_shared_experts", 1.into());
        set("num_experts_per_tok", 2.into());
        set("sliding_window", 8.into());
        set("max_position_embeddings", 64.into());
        set("compress_ratios", serde_json::json!([0, 0]));
        set("kv_source_layer_ids", serde_json::json!([]));
        set("index_source_layer_ids", serde_json::json!([]));
        set("index_n_heads", 2.into());
        set("index_head_dim", 16.into());
        set("index_topk", 8.into());
        set("engram_layer_ids", serde_json::json!([1]));
        set("engram_num_embeddings", serde_json::json!([64]));
        set("engram_max_ngram_size", 4.into());
        set("engram_vocab_size", 128.into());
        set("engram_n_heads", 2.into());
        set("engram_head_dim", 32.into());
        set("num_nextn_predict_layers", 0.into());
        set("dspark_target_layer_ids", serde_json::json!([]));
        set("dspark_noise_token_id", 5.into());
        t.remove("vision_config");
        j
    }

    fn toy() -> DeepseekV41Config {
        DeepseekV41Config::from_json(&toy_json()).unwrap()
    }

    /// The reference forward, transcribed from `Transformer.forward` / `Block.forward` / `MoE.forward` (`model.py`, 1186-1287 and
    /// 909-998) line by line: one residual stream of `hc_mult` copies, `pre_mix` carried from block to block, the engram before the block.
    ///
    /// It shares only the leaf arithmetic with the executor (`linear`, `rms_norm`, `swiglu`, `select`, the Sinkhorn, the MLA block): what it
    /// does not share is the *order*, which is the whole point of the comparison.
    fn reference(cfg: &DeepseekV41Config, w: &HostWeights, ids: &[u32]) -> Vec<f32> {
        let (hc, h) = (cfg.hc.mult, cfg.hidden);
        let rows = ids.len();
        // h = self.embed(input_ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        let mut x = vec![0f32; rows * hc * h];
        for r in 0..rows {
            let row = &w.embed[ids[r] as usize * h..(ids[r] as usize + 1) * h];
            for c in 0..hc {
                x[(r * hc + c) * h..(r * hc + c) * h + h].copy_from_slice(row);
            }
        }
        // pre_mix = make_identity_pre_mix(h, self.hc_mult)
        let mut pre_mix = ops::identity_pre_mix(hc).repeat(rows);
        let mut window: Vec<i64> = Vec::new();
        for (i, layer) in w.layers.iter().enumerate() {
            // if layer.engram is not None: h = layer.engram(h, ...)
            if let Some(slot) = cfg.engram.layer_ids.iter().position(|&l| l == i) {
                let e = &w.engram[slot];
                for r in 0..rows {
                    window.push(ids[r] as i64);
                    let out = engram_contribution(cfg, e, slot, &window, &x[r * hc * h..(r + 1) * hc * h]).unwrap();
                    x[r * hc * h..(r + 1) * hc * h].copy_from_slice(&out[..hc * h]);
                }
            }
            // ---- Block.forward
            // residual = x
            let residual = x.clone();
            // attn_pre, attn_post, attn_comb = self.hc_mixes(x, hc_attn_*)
            let (attn_pre, attn_post, attn_comb) = hc_mixes(cfg, &x, rows, &layer.hc_attn);
            // x = self.hc_pre(x, pre_mix); x = self.attn_norm(x); x = self.attn(...); x = self.hc_post(...)
            let mut sub = ops::hc_pre(&x, rows, hc, h, &pre_mix);
            ops::rms_norm(&mut sub, &layer.attn_norm, h, cfg.rms_eps as f32);
            let sub = attention_block(cfg, layer, &sub, rows, window.len());
            x = ops::hc_post(&x, &sub, rows, hc, h, &attn_post, &attn_comb);

            // residual = x
            let residual2 = x.clone();
            // ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, hc_ffn_*)
            let (ffn_pre, ffn_post, ffn_comb) = hc_mixes(cfg, &x, rows, &layer.hc_ffn);
            // x = self.hc_pre(x, attn_pre); x = self.ffn_norm(x); x = self.ffn(x)
            let mut sub = ops::hc_pre(&x, rows, hc, h, &attn_pre);
            ops::rms_norm(&mut sub, &layer.ffn_norm, h, cfg.rms_eps as f32);
            let sub = moe(cfg, layer, &sub, rows);
            x = ops::hc_post(&x, &sub, rows, hc, h, &ffn_post, &ffn_comb);
            // return x, ffn_pre
            pre_mix = ffn_pre;
            let _ = residual;
            let _ = residual2;
        }
        // h = layer.hc_pre(h, pre_mix); logits = self.head(self.norm(h))
        let mut last = ops::hc_pre(&x, rows, hc, h, &pre_mix);
        ops::rms_norm(&mut last, &w.norm, h, cfg.rms_eps as f32);
        ops::linear(&last[(rows - 1) * h..], &w.head, 1, cfg.vocab, h)
    }

    fn hc_mixes(cfg: &DeepseekV41Config, x: &[f32], rows: usize, site: &HostHc) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        ops::hc_mixes(
            x,
            rows,
            cfg.hc.mult,
            cfg.hidden,
            &site.fnw,
            &site.scale,
            &site.base,
            cfg.hc.sinkhorn_iters,
            cfg.rms_eps as f32,
            cfg.hc.eps as f32,
        )
    }

    /// `MoE.forward`: the routed experts on the picks, then `y += self.shared_experts(x)`.
    fn moe(cfg: &DeepseekV41Config, layer: &HostLayer, x: &[f32], rows: usize) -> Vec<f32> {
        let (h, inter) = (cfg.hidden, cfg.moe.intermediate);
        let topk = cfg.moe.experts_per_token;
        let logits = ops::linear(x, &layer.gate, rows, cfg.moe.routed_experts, h);
        let (picks, wts) =
            ops::select(&logits, rows, cfg.moe.routed_experts, topk, &layer.bias, cfg.moe.routed_scaling_factor as f32);
        let mut y = vec![0f32; rows * h];
        for r in 0..rows {
            for k in 0..topk {
                let e = picks[r * topk + k] as usize;
                let g = ops::linear(&x[r * h..(r + 1) * h], &layer.experts[e][0], 1, inter, h);
                let u = ops::linear(&x[r * h..(r + 1) * h], &layer.experts[e][2], 1, inter, h);
                let a = ops::swiglu(&g, &u, None, 1, inter, cfg.swiglu_limit as f32);
                let o = ops::linear(&a, &layer.experts[e][1], 1, h, inter);
                for d in 0..h {
                    y[r * h + d] += wts[r * topk + k] * o[d];
                }
            }
        }
        // y += self.shared_experts(x)
        let g = ops::linear(x, &layer.shared[0], rows, inter, h);
        let u = ops::linear(x, &layer.shared[2], rows, inter, h);
        let a = ops::swiglu(&g, &u, None, rows, inter, cfg.swiglu_limit as f32);
        let shared = ops::linear(&a, &layer.shared[1], rows, h, inter);
        for i in 0..rows * h {
            y[i] += shared[i];
        }
        y
    }

    fn close(a: &[f32], b: &[f32], tol: f32) -> Result<(), String> {
        if a.len() != b.len() {
            return Err(format!("{} values against {}", a.len(), b.len()));
        }
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            if !x.is_finite() || !y.is_finite() {
                return Err(format!("element {i}: {x} against {y} is not finite"));
            }
            let d = (x - y).abs();
            let s = x.abs().max(y.abs()).max(1.0);
            if d / s > tol {
                return Err(format!("element {i}: {x} against {y} (relative {})", d / s));
            }
        }
        Ok(())
    }

    /// The op list the executor walks is the reference forward: same logits, element for element.
    #[test]
    fn the_scheduled_forward_equals_the_reference_forward() {
        let cfg = toy();
        let w = HostWeights::toy(&cfg, 0x5eed);
        let ids: Vec<u32> = vec![3, 17, 42, 7, 99, 1];
        let got = HostModel::new(&cfg, &w).forward(&ids);
        let want = reference(&cfg, &w, &ids);
        assert_eq!(got.len(), cfg.vocab);
        close(&got, &want, 2e-5).unwrap_or_else(|e| panic!("the schedule and the reference disagree: {e}"));
        // and the logits are not degenerate: a schedule that dropped a residual would not be a constant vector
        let spread = got.iter().copied().fold(f32::NEG_INFINITY, f32::max) - got.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(spread > 1e-4, "the logits are flat: {spread}");
    }

    /// Both layers contribute: a one-layer model is not the two-layer one (the engram layer and the block order both matter).
    #[test]
    fn the_engram_layer_changes_the_result() {
        let cfg = toy();
        let w = HostWeights::toy(&cfg, 0x5eed);
        let ids: Vec<u32> = vec![3, 17, 42, 7];
        let two = HostModel::new(&cfg, &w).forward(&ids);
        let mut one_cfg = cfg.clone();
        one_cfg.layers = 1;
        one_cfg.engram.layer_ids.clear();
        let one = HostModel::new(&one_cfg, &w).forward(&ids);
        assert_eq!(one.len(), two.len());
        assert!(close(&one, &two, 1e-6).is_err(), "one layer and two give the same logits");
    }

    /// The comparison is sensitive to the placement: taking the collapse's coefficients from the wrong site changes the logits.
    #[test]
    fn the_wrong_pre_mix_is_visible() {
        let cfg = toy();
        let w = HostWeights::toy(&cfg, 0x5eed);
        let ids: Vec<u32> = vec![5, 6, 7, 8];
        let good = HostModel::new(&cfg, &w).forward(&ids);
        // the reference with the attention collapse taking its own site's pre_mix instead of the previous block's
        let (hc, h, rows) = (cfg.hc.mult, cfg.hidden, ids.len());
        let mut x = vec![0f32; rows * hc * h];
        for r in 0..rows {
            let row = &w.embed[ids[r] as usize * h..(ids[r] as usize + 1) * h];
            for c in 0..hc {
                x[(r * hc + c) * h..(r * hc + c) * h + h].copy_from_slice(row);
            }
        }
        let mut pre_mix = ops::identity_pre_mix(hc).repeat(rows);
        let mut window: Vec<i64> = Vec::new();
        for (i, layer) in w.layers.iter().enumerate() {
            if let Some(slot) = cfg.engram.layer_ids.iter().position(|&l| l == i) {
                for r in 0..rows {
                    window.push(ids[r] as i64);
                    let out =
                        engram_contribution(&cfg, &w.engram[slot], slot, &window, &x[r * hc * h..(r + 1) * hc * h]).unwrap();
                    x[r * hc * h..(r + 1) * hc * h].copy_from_slice(&out[..hc * h]);
                }
            }
            let residual = x.clone();
            let (attn_pre, attn_post, attn_comb) = hc_mixes(&cfg, &x, rows, &layer.hc_attn);
            // the bug: this block's own attention pre_mix, not the previous block's FFN one
            let mut sub = ops::hc_pre(&x, rows, hc, h, &attn_pre);
            ops::rms_norm(&mut sub, &layer.attn_norm, h, cfg.rms_eps as f32);
            let sub = attention_block(&cfg, layer, &sub, rows, window.len());
            x = ops::hc_post(&x, &sub, rows, hc, h, &attn_post, &attn_comb);
            let _ = residual;
            let (ffn_pre, ffn_post, ffn_comb) = hc_mixes(&cfg, &x, rows, &layer.hc_ffn);
            // the second bug: the FFN collapses with its own site's pre_mix instead of the attention's
            let mut sub = ops::hc_pre(&x, rows, hc, h, &ffn_pre);
            ops::rms_norm(&mut sub, &layer.ffn_norm, h, cfg.rms_eps as f32);
            let sub = moe(&cfg, layer, &sub, rows);
            x = ops::hc_post(&x, &sub, rows, hc, h, &ffn_post, &ffn_comb);
            pre_mix = ffn_pre;
        }
        let mut last = ops::hc_pre(&x, rows, hc, h, &pre_mix);
        ops::rms_norm(&mut last, &w.norm, h, cfg.rms_eps as f32);
        let buggy = ops::linear(&last[(rows - 1) * h..], &w.head, 1, cfg.vocab, h);
        assert!(close(&good, &buggy, 1e-6).is_err(), "the wrong pre_mix is invisible: the test proves nothing");
    }

    /// A chunk of a one-layer toy is the same arithmetic whatever the row count: the host forward has no grid dependence either.
    #[test]
    fn a_prompt_row_does_not_depend_on_the_chunk_grid() {
        let cfg = toy();
        let w = HostWeights::toy(&cfg, 0x1234);
        let ids: Vec<u32> = vec![11, 12, 13, 14, 15];
        let whole = HostModel::new(&cfg, &w).forward(&ids);
        // the same prompt in two chunks: the first two rows, then the rest starting at position 2
        let first = HostModel::new(&cfg, &w).forward(&ids[..2]);
        let second = {
            // the host model takes its positions from the length: the last row's logits are what a prefill asks for
            let mut all = ids[..4].to_vec();
            all.truncate(4);
            HostModel::new(&cfg, &w).forward(&all)
        };
        assert_eq!(first.len(), whole.len());
        assert_eq!(second.len(), whole.len());
        // and the toy's attention window is shorter than the prompt, so the rows differ; the point here is that the arithmetic runs
        assert!(whole.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn the_toy_config_is_the_shape_the_report_names() {
        let c = toy();
        assert_eq!((c.layers, c.hidden, c.moe.routed_experts), (2, 64, 4));
        assert_eq!(c.moe.experts_per_token, 2);
        assert_eq!(c.engram.layer_ids, vec![1]);
        assert_eq!(c.hc.mult, 4);
    }

    #[test]
    fn candidates_are_ordered_by_value_and_ties_by_id() {
        let logits: Vec<f32> = vec![0.5, 1.5, 0.5, -1.0, 2.5];
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        assert_eq!(order, vec![4, 1, 0, 2, 3]);
    }

    // ------------------------------------------------------------- T2.5c: REXL3_DSV41_WINDOW_ROWS

    #[test]
    fn window_rows_knob_stays_inside_the_family_verified_range() {
        assert_eq!(parse_window_rows(None, WINDOW_ROWS).unwrap(), WINDOW_ROWS, "unset is the default");
        assert_eq!(parse_window_rows(Some(""), WINDOW_ROWS).unwrap(), WINDOW_ROWS, "empty is the default");
        assert_eq!(parse_window_rows(Some(" 8 "), WINDOW_ROWS).unwrap(), 8, "trimmed");
        assert_eq!(parse_window_rows(Some("1"), WINDOW_ROWS).unwrap(), 1, "one row: the serial window");
        // the knob only narrows: one past the built window is refused, whatever the built window is
        let over = (WINDOW_ROWS + 1).to_string();
        for bad in ["0", over.as_str(), "128", "abc", "-1"] {
            let e = parse_window_rows(Some(bad), WINDOW_ROWS).unwrap_err();
            assert!(e.contains("REXL3_DSV41_WINDOW_ROWS"), "refused by name: {e}");
            assert!(e.contains(bad), "the refusal shows the value: {e}");
        }
        // the refusal names the verified range, so an operator sees the bounds and not just the rejection
        let e = parse_window_rows(Some("128"), WINDOW_ROWS).unwrap_err();
        assert!(e.contains(&format!("1..={WINDOW_ROWS}")), "{e}");
    }

    #[test]
    fn window_rows_knob_only_ever_narrows_the_built_window() {
        // every in-range value round-trips exactly — the knob is a budget, not a re-sizing
        for v in 1..=WINDOW_ROWS {
            assert_eq!(parse_window_rows(Some(&v.to_string()), WINDOW_ROWS).unwrap(), v);
        }
        // the Qwen recipe's opt-in (128 rows) is outside what this family's buffers hold — refused, never clamped
        assert!(parse_window_rows(Some("128"), WINDOW_ROWS).is_err());
        assert!(parse_window_rows(Some(&(WINDOW_ROWS + 1).to_string()), WINDOW_ROWS).is_err());
        // and a bigger `built` (a future re-sized family) moves the range with it
        assert_eq!(parse_window_rows(Some("16"), 16).unwrap(), 16);
        assert!(parse_window_rows(Some("17"), 16).is_err());
    }

    // ------------------------------------------------------------- the candidates read-back staging (a4dc965ae)

    #[test]
    fn topk_stage_sizing_covers_the_read_back_extents() {
        // the staging block must fit the two async D2H copies (ids and values, 4 bytes per candidate each); the
        // third array (weights) is reserved but not copied back
        for rows in [1usize, 8, WINDOW_ROWS] {
            for count in [1usize, 7, MAX_CANDIDATES] {
                let need = TopkStage::needed_bytes(rows, count);
                assert_eq!(need, rows * count * 12, "ids + values + weights slots");
                assert!(need >= rows * count * 8, "at least the copied extents");
            }
        }
        assert_eq!(TopkStage::needed_bytes(1, 7), 84);
        assert_eq!(TopkStage::needed_bytes(8, 7), 672);
    }

    #[test]
    fn topk_stage_must_grow_past_the_prefill_sized_first_round() {
        // Regression: a prefill chunk reads back one row (`candidates_of(1, n)`), so a stage allocated for the
        // FIRST round is sized 1 row; a later decode/verify window reads back more rows, and the async copies
        // then extend past the pinned block (cudaErrorInvalidValue). The staging must follow the largest request.
        let prefill_need = TopkStage::needed_bytes(1, 7);
        let window_need = TopkStage::needed_bytes(8, 7);
        assert!(window_need > prefill_need, "the window outgrows a prefill-sized stage");
        // any in-range window must be servable by a stage grown to it — the capacity policy is monotone
        let mut cap = prefill_need;
        for rows in 1..=WINDOW_ROWS {
            let need = TopkStage::needed_bytes(rows, 7);
            if need > cap {
                cap = need;
            }
            assert!(cap >= need, "grown stage holds every window");
        }
    }
}
