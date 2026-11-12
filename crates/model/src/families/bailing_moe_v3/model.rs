//! The Bailing Moe V3 GPU forward: the serial 72-layer trunk (63 KDA + 9 Gated MLA, dense SwiGLU then 512-expert MoE) and
//! the `L72` NextN MTP drafter. Hidden states are bf16 end to end; every kernel on the path is row-invariant, so a row's
//! logits do not depend on the other rows of the call, their number, or their position in it.
//!
//! Two modes, one arithmetic. A *window* (decode, or a verify window of drafts) runs the rows of one or several sequences
//! through the trunk and leaves each KDA layer's record behind; `commit(slot, keep)` later keeps the first `keep` rows of a
//! sequence's window (the KDA state flips to the buffer the window already wrote when every row is kept, and replays the
//! accepted prefix from the pre-window state when only some are; the MLA latents need nothing: a rejected row is overwritten by
//! the next window before anyone reads it). A *prefill chunk* is the same forward on up to `PREFILL_ROWS` consecutive rows of
//! one sequence followed, layer by layer, by a full-accept commit (so every KDA layer of a chunk shares one record, `prefill_rec`).
//!
//! Every per-step input (token ids, positions, sequence starts and slots, keep counts, the page table) is written in the order of
//! the model's non-blocking stream ([`HostStage`]), never with a pageable `cudaMemcpy` on the legacy stream.
//!
//! The MTP drafter keeps the last `CAP_TAPS` trunk hidden rows of each slot; a draft call runs `L72` over the newly committed
//! rows (absorb), then chains draft tokens through the layer, proposing `argmax` tokens.

use super::alloc::{self, WINDOW_ROWS};
use super::config::BailingConfig;
use super::device::{BailingDevice, FfnDevice, MixerDevice};
use super::load::EmbeddingSidecar;
use crate::error::ModelError;
use crate::forward::Candidates;
use gb_cuda::attention::{BlockTable, RowIndex};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::glm_glue;
use gb_cuda::graph::{CaptureError, Graph};
use gb_cuda::kda::{self, Params};
use gb_cuda::mla_gated::{self, MlaGatedCache};
use gb_cuda::rt::{CudaError, DevBuf, HostStage, Stream};
use gb_cuda::{fmoe, glue};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Rows of the widest decode/verify window.
pub const ROWS: usize = WINDOW_ROWS;
/// Rows of one prefill chunk (the widest trunk call; decode/verify windows stay at `ROWS`). 512 rows amortize the MoE
/// weight reads (512 routed tokens activate nearly every expert, so each expert tile runs many tokens per pass) and the
/// EXL3 prefill GEMMs (the pinned prefill path is a real batching GEMM, unlike the 8-row decode GEMV slicing), which is
/// why prefill at 512 is ~4x faster than at 128.
/// `fmoe::prefill`, `kda::Record`, `kda::Workspace`, `exl3_linear` and `mla_gated` all support 512 rows, and both
/// prefill and verify paths are row-invariant, so any chunking gives identical bits.
pub const PREFILL_ROWS: usize = 512;
const _: () = assert!(PREFILL_ROWS == alloc::MAX_PREFILL_CHUNK && PREFILL_ROWS % 16 == 0);
/// The MTP layer's ordinal among the MLA layers (9 trunk MLA layers, then L72).
pub const MTP_MLA_ORD: usize = 9;
/// Committed trunk hidden rows kept per slot for MTP drafting (ring).
pub use super::alloc::CAP_TAPS;

/// One trunk decode/verify window's launch shape: every kernel of `forward_trunk(b, false)` + `queue_topk` reads only
/// fixed device buffers (`self.s.*`, weights, pools, caches) — the batch arrays ride in `self.s` — so this triple
/// uniquely determines the whole launch graph (the KDA/MLA row positions and the page table live on the device).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowGraphKey {
    pub rows: usize,
    pub n_seq: usize,
    pub count: usize,
}

const RMS_EPS: f32 = 1e-6;
const KDA_LOWER: f32 = -5.0;
/// bf16 bits of 30.0: `sigmoid(30) == 1.0` exactly in fp32, the shared expert's unweighted add.
const SHARED_GATE_ONE: u16 = 0x41F0;
/// FrostMoE K chunks at hidden 4096 (32 x 128 blocks; 8 and 16 are the valid counts, 8 stages 4 blocks/chunk).
const BAILING_KC: usize = alloc::MOE_KC;

fn env_on(key: &str) -> bool {
    std::env::var(key).map(|v| v != "0").unwrap_or(true)
}

pub struct MtpSlot {
    /// How many committed tokens have been fed through `L72` (positions `< absorbed` are in the L72 cache).
    pub absorbed: usize,
    /// The last absorbed row's L72 output (`[hidden]` bf16), the chain's start.
    pub last_out: Option<Vec<u16>>,
    /// The GPU `head_in` register is stale (a slot begin/end or a prefix restore moved the state without re-absorbing).
    pub head_dirty: bool,
}

/// The drafter's own copy of a subset of the trunk `lm_head`'s columns (whole 128-column strips, bit-identical to the full
/// head's): the draft head scores only those columns, so a draft step reads that share of the 483 MiB head. The trunk's full
/// head still verifies every token, so restricting the draft head cannot change a single output bit — it only bounds which
/// tokens can be *proposed* (a draft outside the set is simply rejected at verify).
pub struct MtpSubHead {
    /// The restricted head (`select_strips` of the full `lm_head`).
    w: gb_cuda::exl3_linear::Exl3Weights,
    /// The decode k2 of `w` (2 * bits): the same k2 path the trunk's full head runs.
    k2: usize,
    /// `[cols]`: the full-vocab token id of each kept column (host side; the top-k ids land on the host anyway).
    map: Vec<i32>,
    /// fp16 `[cols]` (one head row per call).
    logits: DevBuf,
    /// The decode workspace for `w`.
    ws: DevBuf,
    /// Kept columns (a strip count of 128).
    cols: usize,
}

/// Confidence-gating thresholds of the MTP drafter (env-overridable for tuning; see `MtpTune::read`).
#[derive(Clone, Copy, Debug)]
pub struct MtpTune {
    /// Minimum top-1 vs top-2 logit margin of `d0` to attempt an N-gram prompt-lookup continuation (0 = always).
    pub pld_margin: f32,
    /// Minimum margin to spend another `L72` chain step: `[0]` after draft 1, `[1]` for later drafts.
    pub chain_gate: [f32; 2],
    /// Minimum margin to *accept* a chained draft (push it into the draft window). A low-margin head output
    /// is near-uniform; verifying it costs a full trunk round, so it is dropped instead.
    pub accept_gate: f32,
}

impl MtpTune {
    pub fn read() -> Self {
        // Tuned on the GB10 (Oct 2026) across story/code/math/reasoning + a 3,607-token prompt: chain gate
        // 1.5/1.5 (logit margin), accept gate 1.0, PLD off. Verified byte-identical greedy parity.
        MtpTune { pld_margin: 9999.0, chain_gate: [1.5, 1.5], accept_gate: 1.0 }
    }
}

pub struct MtpState {
    pub slots: Vec<MtpSlot>,
    /// The restricted draft head, when built (`--spec mtp`).
    pub sub: Option<MtpSubHead>,
    /// Confidence-gating thresholds (tunable via env).
    pub tune: MtpTune,
}

/// One KDA layer's per-slot recurrent state at a committed prefix (both state buffers, parity, conv history).
pub struct KdaSlotSnap {
    pub s: Vec<f32>,
    pub parity: i32,
    pub conv: Vec<u16>,
}

/// One MLA cache's pages of a slot at a committed prefix: the page contents in logical page order.
pub struct MlaCacheSnap {
    pub data: Vec<u8>,
}

/// The MTP drafter's per-slot host state at a committed prefix.
pub struct MtpSnap {
    pub taps: Vec<u16>,
    pub absorbed: usize,
    pub last_out: Option<Vec<u16>>,
}

/// A full prefix snapshot of one slot at a committed length.
pub struct BailingSnapshot {
    pub len: usize,
    pub kda: Vec<KdaSlotSnap>,
    pub mla: Vec<MlaCacheSnap>,
    pub pages: Vec<u32>,
    pub mtp: Option<MtpSnap>,
}

/// Host-RAM snapshot slots (one per `--prefix-cache` entry).
pub struct SnapPool {
    pub slots: Vec<Option<BailingSnapshot>>,
    pub free: Vec<u32>,
}

/// Everything a forward scratches, sized for `ROWS` rows.
pub struct Scratch {
    pub ids: DevBuf,
    pub row_slot: DevBuf,
    pub row_pos: DevBuf,
    pub seq_start: DevBuf,
    pub seq_slot: DevBuf,
    pub keep: DevBuf,
    /// hidden (bf16 `[rows, hidden]`).
    pub hidden: DevBuf,
    /// normed layer input (bf16 `[rows, hidden]`).
    pub norm: DevBuf,
    /// KDA qkv: a window's f32 `[rows, 3 * channels]` (the decode GEMV's output, rounded to bf16 inside `kda::prep`), a prefill
    /// chunk's bf16 `[rows, 3 * channels]` (the pinned EXL3 GEMM's fp32 sums stored with the same round-to-nearest-even, so
    /// `kda::prep` sees the same bf16 values either way).
    pub qkv: DevBuf,
    /// KDA f/g projections (f16 `[rows, channels]` each).
    pub f_out: DevBuf,
    pub g_out: DevBuf,
    /// MLA q_a (bf16 `[rows, q_lora_rank]`, normed in place).
    pub q_a: DevBuf,
    /// MLA q (f16 `[rows, heads * qk_head_dim]`) and kv (f16 `[rows, kv_a_group_n]`).
    pub q: DevBuf,
    pub kv: DevBuf,
    /// MLA intermediates (f16).
    pub q_lat: DevBuf,
    pub q_rot: DevBuf,
    pub o_lat: DevBuf,
    pub mla_out: DevBuf,
    pub gated: DevBuf,
    /// the dense MLP's gate, up and SwiGLU outputs (bf16 `[rows, intermediate]` each); `a` also holds the mixers' outputs and `g`
    /// the down projection's (bf16 `[rows, hidden]`).
    pub g: DevBuf,
    pub u: DevBuf,
    pub a: DevBuf,
    /// MoE routing (logits bf16 `[rows, experts]`, shared gate bf16 `[rows]`, window/prefill fmoe workspaces, out bf16 `[rows, hidden]`).
    pub moe_logits: DevBuf,
    pub moe_shared_gate: DevBuf,
    pub moe_ws: DevBuf,
    pub moe_pws: DevBuf,
    pub moe_out: DevBuf,
    /// Head: final-normed (bf16), logits (f16), top-k.
    pub final_normed: DevBuf,
    pub logits: DevBuf,
    pub topk_ids: DevBuf,
    pub topk_vals: DevBuf,
    pub topk_ws: DevBuf,
    /// The EXL3 decode workspace.
    pub ws: DevBuf,
    /// The EXL3 pinned prefill workspace a prefill chunk needs (the largest of every trunk linear the prefill forward
    /// runs, mixed-dtype included).
    pub prefill_ws: DevBuf,
    /// MTP: the token embedding rows (bf16 `[rows, h]`).
    pub e_next: DevBuf,
    /// The MTP head's input row (bf16 `[1, hidden]`): the L72 output row the head runs on, kept on the GPU across proposes.
    pub head_in: DevBuf,
    /// The MTP chain's working register (bf16 `[1, hidden]`): the chain's previous L72 output, kept on the GPU.
    pub chain_in: DevBuf,
}

impl Scratch {
    /// Every buffer `alloc::AllocPlan::scratch_items` prices, under the names of [`BailingModel::allocations`].
    fn new(cfg: &BailingConfig, r: usize) -> Result<Self, ModelError> {
        // The scratch is sized for the widest call, a prefill chunk of `r` rows; the FrostMoE window workspace is sized for the
        // widest decode/verify window (`fmoe::forward` caps `T` at 64 rows, `fmoe::prefill` does not).
        let rw = ROWS;
        let h = cfg.hidden;
        let ch = cfg.kda.channels();
        let m = &cfg.mla;
        let ffn = &cfg.ffn;
        let e = ffn.routed_experts;
        let topk = ffn.experts_per_token;
        let zeroed = |bytes: usize| DevBuf::zeroed_view(bytes).map_err(|e| ModelError::Config(format!("scratch alloc: {e}")));
        Ok(Self {
            ids: zeroed(r * 4)?,
            row_slot: zeroed(r * 4)?,
            row_pos: zeroed(r * 4)?,
            seq_start: zeroed((r + 1) * 4)?,
            seq_slot: zeroed(r * 4)?,
            keep: zeroed(r * 4)?,
            hidden: zeroed(r * h * 2)?,
            norm: zeroed(r * h * 2)?,
            qkv: zeroed((r * 3 * ch * 2).max(rw * 3 * ch * 4))?,
            f_out: zeroed(r * ch * 2)?,
            g_out: zeroed(r * ch * 2)?,
            q_a: zeroed(r * m.q_lora_rank * 2)?,
            q: zeroed(r * m.heads * m.qk_head_dim() * 2)?,
            kv: zeroed(r * m.kv_a_group_n() * 2)?,
            q_lat: zeroed(r * m.heads * m.kv_lora_rank * 2)?,
            q_rot: zeroed(r * m.heads * m.qk_rope_head_dim * 2)?,
            o_lat: zeroed(r * m.heads * m.kv_lora_rank * 2)?,
            mla_out: zeroed(r * m.heads * m.v_head_dim * 2)?,
            gated: zeroed(r * (m.heads * m.v_head_dim).max(ch) * 2)?,
            g: zeroed(r * ffn.dense_intermediate.max(h) * 2)?,
            u: zeroed(r * ffn.dense_intermediate.max(h) * 2)?,
            a: zeroed(r * ffn.dense_intermediate.max(h) * 2)?,
            moe_logits: zeroed(r * e * 2)?,
            // the shared gate is a per-row constant; fill it once
            moe_shared_gate: DevBuf::from_slice(&vec![SHARED_GATE_ONE; r])
                .map_err(|e| ModelError::Config(format!("shared gate: {e}")))?,
            moe_ws: zeroed(fmoe::workspace_bytes(rw, topk, h, ffn.moe_intermediate, BAILING_KC))?,
            moe_pws: fmoe::prefill_workspace(r, topk, e, h, ffn.moe_intermediate)
                .map_err(|e| ModelError::Config(format!("moe prefill workspace: {e}")))?,
            moe_out: zeroed(r * h * 2)?,
            final_normed: zeroed(r * h * 2)?,
            // the head (lm_head + top-k) only runs on decode/verify windows (<= ROWS rows) and on single rows
            // (prefill chunks' last row, the MTP head), so `ROWS` rows suffice for all of them
            logits: zeroed(ROWS * cfg.vocab * 2)?,
            topk_ids: zeroed(ROWS * alloc::MAX_CANDIDATES * 4)?,
            topk_vals: zeroed(ROWS * alloc::MAX_CANDIDATES * 4)?,
            topk_ws: zeroed(glm_glue::topk_workspace_bytes(ROWS, cfg.vocab, alloc::MAX_CANDIDATES))?,
            ws: zeroed(alloc::exl3_workspace_bytes(cfg, ROWS))?,
            prefill_ws: zeroed(alloc::exl3_prefill_workspace_bytes(cfg, r))?,
            e_next: zeroed(r * h * 2)?,
            head_in: zeroed(h * 2)?,
            chain_in: zeroed(h * 2)?,
        })
    }
}

/// `dst[..v.len()] = v` in the order of `st` (every per-step input of the model goes through here): a kernel launched on `st`
/// after this call reads the new bytes, and one launched before it (still queued, or still running) has read the old ones. The
/// bytes are copied into `stage`, which keeps them until the model's next [`BailingModel::sync`], so `v` may be a temporary.
/// `DevBuf::upload` would not do: it is a pageable `cudaMemcpy` on the legacy default stream, which the model's non-blocking
/// stream does not wait for, and it may return before its DMA lands.
fn put<T: Copy>(stage: &mut HostStage, dst: &mut DevBuf, v: &[T], st: &Stream, what: &str) -> Result<(), ModelError> {
    stage.upload(dst, v, st).map_err(|e: CudaError| ModelError::Config(format!("{what} upload: {e}")))
}

impl Scratch {
    /// Router logits `[rows, E]` bf16 = `x @ router^T` (router bf16 `[E, H]`): one direct bf16 linear.
    fn moe_logits_bf16(&mut self, stream: &Stream, router: &DevBuf, h: usize, e: usize, rows: usize) -> Result<(), ModelError> {
        let mut y = self.moe_logits.subview(0, rows * e * 2);
        mla_gated::bf16_linear(&self.norm, router, &mut y, rows, h, e, stream)
            .map_err(|e| ModelError::Config(format!("router linear: {e}")))?;
        Ok(())
    }
}

/// One window's layout, until its commit is applied.
pub struct WindowRec {
    pub seq_slot: Vec<i32>,
    pub seq_len: Vec<usize>,
    pub rows: usize,
}

pub struct BailingModel {
    pub cfg: BailingConfig,
    pub dev: BailingDevice,
    pub embedding: EmbeddingSidecar,
    pub stream: Stream,
    pub slots: usize,
    pub max_context: usize,
    /// KDA layer `l` -> ordinal among KDA layers.
    pub kda_ord: Vec<usize>,
    pub pools: Vec<kda::Pool>,
    /// Each KDA layer's window record (`ROWS` rows): what a decode/verify window leaves for its lazy commit.
    pub recs: Vec<kda::Record>,
    /// The one record every KDA layer of a prefill chunk writes (`PREFILL_ROWS` rows): a chunk commits each layer right after
    /// its recurrence, before the next layer runs, so the layers never need theirs at the same time.
    pub prefill_rec: kda::Record,
    pub kda_ws: kda::Workspace,
    pub kda_params: Params,
    /// MLA layer `l` (trunk only) -> ordinal among MLA layers.
    pub mla_ord: Vec<usize>,
    pub mla_caches: Vec<MlaGatedCache>,
    pub cos: DevBuf,
    pub sin: DevBuf,
    pub table: BlockTable,
    pub slot_pages: Vec<Vec<u32>>,
    pub free_blocks: Vec<u32>,
    pub s: Scratch,
    pub pending: BTreeMap<usize, usize>,
    pub last_window: Option<WindowRec>,
    pub mtp: Option<MtpState>,
    /// The host bytes of the per-step uploads queued on `stream`, kept until the stream has run them ([`HostStage`]).
    stage: HostStage,
    /// The bf16 embedding table `[vocab, hidden]` the device gathers rows from: the shard read in place where the GPU reads pageable
    /// memory (`load::embed_in_place`), a device copy otherwise.
    pub embed_dev: DevBuf,
    /// `embed_dev` is the shard's bytes in place (host memory), not a device copy.
    embed_in_place: bool,
    /// Device-resident MTP tap rings (`bf16 [CAP_TAPS, hidden]` per slot): D2D copies only, host copies at snapshots.
    pub dev_taps: Vec<DevBuf>,
    /// CUDA graphs of the trunk decode/verify kernels, keyed by window shape.
    graphs: HashMap<WindowGraphKey, Graph>,
    /// Shapes already run eagerly once (the second visit captures, later ones replay).
    seen_shapes: HashSet<WindowGraphKey>,
    /// Graph replay is on; a refused capture falls back to eager runs.
    use_graphs: bool,
    /// The prefix-cache snapshot slots (`None`: the cache is off).
    pub snaps: Option<SnapPool>,
}

impl BailingModel {
    pub fn new(
        cfg: BailingConfig,
        dev: BailingDevice,
        embedding: EmbeddingSidecar,
        slots: usize,
        kv_tokens: usize,
        with_mtp: bool,
    ) -> Result<Self, ModelError> {
        let max_context = kv_tokens.max(256);
        let n_pages = max_context.div_ceil(256).max(1);
        // non-blocking: CUDA graph capture (`Graph::begin`) requires a stream that does not order with the legacy default
        // stream, so every per-step upload goes through `stage` (`put`), in the stream's order
        let stream = Stream::new_nonblocking().map_err(|e| ModelError::Config(format!("stream: {e}")))?;
        gb_cuda::exl3_linear::prefill_init().map_err(|e| ModelError::Config(format!("exl3 prefill init: {e}")))?;
        // Blackwell programmatic dependent launch (GB10/P1000): a launch may start while its predecessor still runs, only
        // reading the predecessor's writes after `pdl_wait` (exl3_linear.cu / pdl.cuh). Default all families (7); the
        // `REXL3_GLM_PDL` env (0..7) overrides.
        let pdl = std::env::var("REXL3_GLM_PDL").ok().and_then(|v| v.trim().parse::<i32>().ok()).unwrap_or(7) & 7;
        gb_cuda::exl3_linear::set_pdl_mask(pdl).map_err(|e| ModelError::Config(format!("pdl mask: {e}")))?;
        let kda_layers: Vec<usize> =
            (0..cfg.layers.len()).filter(|&l| cfg.layers[l].mixer == crate::family::Mixer::Kda).collect();
        let mla_layers: Vec<usize> =
            (0..cfg.layers.len()).filter(|&l| cfg.layers[l].mixer != crate::family::Mixer::Kda).collect();
        let n_mla = mla_layers.len() + usize::from(with_mtp);
        let dims = kda::Dims { heads: cfg.kda.heads, hidden: cfg.hidden };
        let params = Params { lower: KDA_LOWER, eps: RMS_EPS, round: kda::Round::NearestEven, r_step: 8 };
        let mut pools = Vec::with_capacity(kda_layers.len());
        let mut recs = Vec::with_capacity(kda_layers.len());
        for _ in &kda_layers {
            pools.push(kda::Pool::new(dims, slots).map_err(|e| ModelError::Config(format!("kda pool: {e}")))?);
            recs.push(kda::Record::alloc(dims, ROWS).map_err(|e| ModelError::Config(format!("kda record: {e}")))?);
        }
        let prefill_rec =
            kda::Record::alloc(dims, PREFILL_ROWS).map_err(|e| ModelError::Config(format!("kda prefill record: {e}")))?;
        let kda_ws = kda::Workspace::alloc(dims, PREFILL_ROWS).map_err(|e| ModelError::Config(format!("kda workspace: {e}")))?;
        let mut mla_caches = Vec::with_capacity(n_mla);
        for _ in 0..n_mla {
            mla_caches.push(MlaGatedCache::new(n_pages, Dtype::BF16).map_err(|e| ModelError::Config(format!("mla cache: {e}")))?);
        }
        let (cos, sin) = mla_gated::rope_tables(max_context, mla_gated::ROPE_THETA);
        let cos = DevBuf::from_slice(&cos).map_err(|e| ModelError::Config(format!("rope cos: {e}")))?;
        let sin = DevBuf::from_slice(&sin).map_err(|e| ModelError::Config(format!("rope sin: {e}")))?;
        let table = BlockTable::from_host(&vec![Vec::new(); slots], n_pages)
            .map_err(|e| ModelError::Config(format!("block table: {e}")))?;
        let free_blocks: Vec<u32> = (0..n_pages as u32).rev().collect();
        let s = Scratch::new(&cfg, PREFILL_ROWS)?;
        // the bf16 embedding table (1.20 GiB for Ling-3.1) as `glue::embed` gathers its rows on the device: where the GPU reads
        // pageable memory (the GB10), the shard's own bytes in place, so it takes no device memory and its pages stay reclaimable
        // page cache; elsewhere a device copy of the same bytes. Either way a row is the same bytes, so every output is the same.
        let embed_in_place = super::load::embed_in_place();
        let embed_dev = if embed_in_place {
            embedding.map_in_place()?
        } else {
            let all = embedding.read_all_bf16()?;
            DevBuf::from_slice(&all).map_err(|e| ModelError::Config(format!("embed table upload: {e}")))?
        };
        // device-resident tap rings: one `CAP_TAPS * hidden * 2`-byte ring per slot (64 MiB at 1 slot)
        let dev_taps = if with_mtp {
            let mut v = Vec::with_capacity(slots);
            for _ in 0..slots {
                v.push(
                    DevBuf::zeroed(CAP_TAPS * cfg.hidden * 2).map_err(|e| ModelError::Config(format!("dev taps alloc: {e}")))?,
                );
            }
            v
        } else {
            Vec::new()
        };
        let use_graphs = env_on("REXL3_GRAPHS");
        let mtp = if with_mtp {
            // The restricted draft head (`alloc::draft_head_strips`): ~45% of the head's columns, bit-identical to the full
            // head's, so a draft is the full head's whenever the full head's choice lies inside the set (the trunk's full head
            // still verifies every token).
            let sub = {
                let ranges = alloc::draft_head_strips(&cfg);
                // assemble the full head's `Exl3Weights` around the device `Lin`'s buffers (shared views), strip-copy the
                // kept columns (D2D), and keep the decomposed head as an `Exl3Weights` over its own buffers
                let lm = &dev.lm_head;
                let full = gb_cuda::exl3_linear::Exl3Weights::from_dims(
                    gb_cuda::exl3_linear::Exl3Dims { k: lm.k, n: lm.n, bits: lm.k2 / 2, codebook: lm.codebook },
                    lm.trellis.subview(0, lm.trellis.bytes()),
                    lm.suh.subview(0, lm.suh.bytes()),
                    lm.svh.subview(0, lm.svh.bytes()),
                    None,
                )
                .map_err(|e| ModelError::Config(format!("draft head assemble: {e}")))?;
                let w = full.select_strips(&ranges).map_err(|e| ModelError::Config(format!("draft head strips: {e}")))?;
                let k2 = w.bits * 2;
                let cols = w.n;
                let mut map = Vec::with_capacity(cols);
                for &(lo, hi) in &ranges {
                    map.extend((lo * 128..hi * 128).map(|id| id as i32));
                }
                let logits = DevBuf::zeroed(cols * 2).map_err(|e| ModelError::Config(format!("draft head logits: {e}")))?;
                let ws = DevBuf::zeroed(gb_cuda::exl3_linear::linear_workspace_bytes(1, cfg.hidden, cols, w.bits))
                    .map_err(|e| ModelError::Config(format!("draft head ws: {e}")))?;
                Some(MtpSubHead { w, k2, map, logits, ws, cols })
            };
            Some(MtpState {
                slots: (0..slots).map(|_| MtpSlot { absorbed: 0, last_out: None, head_dirty: true }).collect(),
                sub,
                tune: MtpTune::read(),
            })
        } else {
            None
        };
        let mut kda_ord = vec![usize::MAX; cfg.layers.len()];
        for (i, &l) in kda_layers.iter().enumerate() {
            kda_ord[l] = i;
        }
        let mut mla_ord = vec![usize::MAX; cfg.layers.len()];
        for (i, &l) in mla_layers.iter().enumerate() {
            mla_ord[l] = i;
        }
        // the zero fills above (`DevBuf::zeroed`) are asynchronous on the legacy stream, which `stream` does not wait for
        gb_cuda::rt::device_sync().map_err(|e| ModelError::Config(format!("device sync: {e}")))?;
        Ok(Self {
            cfg,
            dev,
            embedding,
            stream,
            slots,
            max_context,
            kda_ord,
            pools,
            recs,
            prefill_rec,
            kda_ws,
            kda_params: params,
            mla_ord,
            mla_caches,
            cos,
            sin,
            table,
            slot_pages: vec![Vec::new(); slots],
            free_blocks,
            s,
            pending: BTreeMap::new(),
            last_window: None,
            mtp,
            stage: HostStage::new(),
            embed_dev,
            embed_in_place,
            dev_taps,
            graphs: HashMap::new(),
            seen_shapes: HashSet::new(),
            use_graphs,
            snaps: None,
        })
    }

    // ------------------------------------------------------------------------------------------------ memory

    /// The device allocations of the backend after the plan (the pack's regions aside), under the names of `alloc::AllocPlan`: each
    /// is the sum of the real buffers' bytes.
    pub fn allocations(&self) -> Vec<(&'static str, u64)> {
        let b = |x: &DevBuf| x.bytes() as u64;
        let sum = |v: &[&DevBuf]| v.iter().map(|x| x.bytes() as u64).sum::<u64>();
        let rec = |r: &kda::Record| sum(&[&r.qkv, &r.g, &r.beta]);
        let s = &self.s;
        let dims = kda::Dims { heads: self.cfg.kda.heads, hidden: self.cfg.hidden };
        let moe_tables = |f: &FfnDevice| match f {
            FfnDevice::Moe(m) => sum(&[&m.g_t, &m.g_su, &m.g_sv, &m.u_t, &m.u_su, &m.u_sv, &m.d_t, &m.d_su, &m.d_sv, &m.ktab]),
            FfnDevice::Dense(_) => 0,
        };
        let d = &self.dev.kda_dummies;
        let mut v = vec![
            ("kda_pools", self.pools.iter().map(|p| sum(&[&p.s, &p.parity, &p.conv])).sum()),
            ("mla_cache", self.mla_caches.iter().map(|c| b(&c.buf)).sum()),
            ("kda_window_records", self.recs.iter().map(rec).sum()),
            ("kda_prefill_record", rec(&self.prefill_rec)),
            ("kda_workspace", kda::Workspace::bytes_per_row(dims) as u64 * self.kda_ws.rows as u64),
            ("row_arrays", sum(&[&s.ids, &s.row_slot, &s.row_pos, &s.seq_start, &s.seq_slot, &s.keep])),
            ("hidden", sum(&[&s.hidden, &s.norm, &s.final_normed])),
            ("kda_qkv", b(&s.qkv)),
            ("kda_fg", sum(&[&s.f_out, &s.g_out])),
            ("mla_acts", sum(&[&s.q_a, &s.q, &s.kv, &s.q_lat, &s.o_lat, &s.q_rot, &s.mla_out, &s.gated])),
            ("dense_mlp", sum(&[&s.g, &s.u, &s.a])),
            ("moe_acts", sum(&[&s.moe_logits, &s.moe_shared_gate, &s.moe_out])),
            ("moe_window_workspace", b(&s.moe_ws)),
            ("moe_prefill_workspace", b(&s.moe_pws)),
            ("logits", b(&s.logits)),
            ("topk", sum(&[&s.topk_ids, &s.topk_vals, &s.topk_ws])),
            ("exl3_workspace", b(&s.ws)),
            ("exl3_prefill_workspace", b(&s.prefill_ws)),
            ("mtp_acts", sum(&[&s.e_next, &s.head_in, &s.chain_in])),
            ("embedding_table", if self.embed_in_place { 0 } else { b(&self.embed_dev) }),
            ("rope_tables", sum(&[&self.cos, &self.sin])),
            ("block_table", b(&self.table.buf)),
            ("kda_dummies", sum(&[&d.f_a, &d.g_a, &d.f_b, &d.g_b])),
            (
                "expert_tables",
                self.dev.layers.iter().map(|l| moe_tables(&l.ffn)).sum::<u64>()
                    + self.dev.mtp.as_ref().map_or(0, |m| moe_tables(&m.layer.ffn)),
            ),
        ];
        if let Some(m) = &self.mtp {
            v.push(("mtp_tap_rings", self.dev_taps.iter().map(b).sum()));
            let head = m
                .sub
                .as_ref()
                .map_or(0, |h| h.w.buffers().iter().map(|(_, x)| x.bytes() as u64).sum::<u64>() + b(&h.logits) + b(&h.ws));
            v.push(("mtp_draft_head", head));
        }
        v
    }

    /// The model's allocations equal `plan`'s totals for `params` (both ways: every allocation is planned, every planned item is
    /// allocated), or the list of the items that differ.
    pub fn check_allocations(&self, plan: &alloc::AllocPlan, params: &alloc::AllocParams) -> Result<(), String> {
        let got = self.allocations();
        let want = plan.totals(params);
        let mut bad = Vec::new();
        for (name, g) in &got {
            match want.iter().find(|(n, _)| n == name) {
                Some((_, w)) if w == g => {}
                Some((_, w)) => bad.push(format!("{name}: allocated {g} B, planned {w} B")),
                None => bad.push(format!("{name}: allocated {g} B, not planned")),
            }
        }
        for (name, w) in &want {
            if !got.iter().any(|(n, _)| n == name) {
                bad.push(format!("{name}: planned {w} B, not allocated"));
            }
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(bad.join("; "))
        }
    }

    /// The rows of the widest prefill chunk (the scratch's).
    pub fn prefill_rows(&self) -> usize {
        PREFILL_ROWS
    }

    // ------------------------------------------------------------------------------------------------ page management

    fn ensure_pages(&mut self, slot: usize, max_pos: usize) -> Result<(), ModelError> {
        let need = (max_pos + 1).div_ceil(256);
        let pages = &mut self.slot_pages[slot];
        while pages.len() < need {
            let b = self.free_blocks.pop().ok_or_else(|| ModelError::Request("no free KV page".into()))?;
            pages.push(b);
        }
        self.table
            .upload_on(&self.slot_pages, &mut self.stage, &self.stream)
            .map_err(|e| ModelError::Config(format!("table upload: {e}")))?;
        Ok(())
    }

    fn release_pages(&mut self, slot: usize) {
        let pages = std::mem::take(&mut self.slot_pages[slot]);
        for b in pages {
            self.free_blocks.push(b);
        }
    }

    // ------------------------------------------------------------------------------------------------ prefix snapshots

    /// Enable the prefix cache with `snapshot_slots` host-RAM snapshot entries.
    pub fn enable_prefix_cache(&mut self, snapshot_slots: usize) {
        self.snaps = Some(SnapPool {
            slots: (0..snapshot_slots).map(|_| None).collect(),
            free: (0..snapshot_slots as u32).rev().collect(),
        });
    }

    pub fn prefix_free_snapshots(&self) -> usize {
        self.snaps.as_ref().map_or(0, |s| s.free.len())
    }

    /// True when the host has enough headroom above a fixed floor to hold a snapshot of `len` tokens in RAM
    /// (the 63 KDA slot states dominate: ~8 MiB per layer, constant).
    pub fn can_snapshot_prefix(&self, len: usize) -> bool {
        if len == 0 || len > self.max_context || self.snaps.as_ref().is_none_or(|s| s.free.is_empty()) {
            return false;
        }
        let kda_bytes: u64 = self.pools.len() as u64 * self.pools.first().map_or(0, |p| p.dims.slot_bytes()) as u64;
        let mla_bytes: u64 = self.mla_caches.len() as u64 * (len as u64) * (mla_gated::CACHE_TOKEN_ELEMS as u64) * 2;
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
                    let needed = kda_bytes + mla_bytes + (1u64 << 30);
                    if avail_bytes < needed {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn alloc_snapshot(&mut self) -> Option<u32> {
        self.snaps.as_mut()?.free.pop()
    }

    pub fn release_prefix(&mut self, snap: u32) {
        if let Some(p) = self.snaps.as_mut() {
            if let Some(sl) = p.slots.get_mut(snap as usize) {
                *sl = None;
                if !p.free.contains(&snap) {
                    p.free.push(snap);
                }
            }
        }
    }

    /// Snapshot `slot`'s state at committed prefix length `len` (`0 < len <= max_context`) into snapshot slot `snap`.
    pub fn prefix_snapshot(&mut self, slot: usize, snap: u32, len: usize) -> Result<(), ModelError> {
        let n_snaps = self.snaps.as_ref().map_or(0, |p| p.slots.len());
        if self.snaps.is_none() || slot >= self.slots || len == 0 || len > self.max_context || (snap as usize) >= n_snaps {
            return Err(ModelError::Request(format!(
                "invalid prefix snapshot (slot {slot}/{}, snap {snap}/{}, len {len}/{})",
                self.slots, n_snaps, self.max_context
            )));
        }
        self.flush()?;
        self.sync()?;
        let mut kda = Vec::with_capacity(self.pools.len());
        for p in &self.pools {
            let (s, parity, conv) = p.slot_snapshot(slot).map_err(|e| ModelError::Config(format!("kda snapshot: {e}")))?;
            kda.push(KdaSlotSnap { s, parity, conv });
        }
        let pages = self.slot_pages[slot].clone();
        let mut mla = Vec::with_capacity(self.mla_caches.len());
        for c in &self.mla_caches {
            let bb = MlaGatedCache::block_bytes(c.dtype);
            let mut data = vec![0u8; pages.len() * bb];
            for (i, &p) in pages.iter().enumerate() {
                c.buf
                    .download_range((p as usize) * bb, &mut data[i * bb..(i + 1) * bb])
                    .map_err(|e| ModelError::Config(format!("mla snapshot: {e}")))?;
            }
            mla.push(MlaCacheSnap { data });
        }
        let mtp = if let Some(m) = &self.mtp {
            // the live tap ring is on the device; the snapshot keeps a host copy (the stream is idle: synced above)
            let mut taps = vec![0u16; CAP_TAPS * self.cfg.hidden];
            self.dev_taps[slot].download(&mut taps).map_err(|e| ModelError::Config(format!("taps download: {e}")))?;
            Some(MtpSnap { taps, absorbed: m.slots[slot].absorbed, last_out: m.slots[slot].last_out.clone() })
        } else {
            None
        };
        if let Some(pool) = self.snaps.as_mut() {
            pool.slots[snap as usize] = Some(BailingSnapshot { len, kda, mla, pages, mtp });
        }
        Ok(())
    }

    /// Restore `slot` to the exact state of snapshot `snap` at prefix length `len`.
    pub fn restore_prefix(&mut self, slot: usize, snap: u32, _pages: &[u32], len: usize) -> Result<(), ModelError> {
        let n_snaps = self.snaps.as_ref().map_or(0, |p| p.slots.len());
        if self.snaps.is_none() || slot >= self.slots || len == 0 || len > self.max_context || (snap as usize) >= n_snaps {
            return Err(ModelError::Request(format!("invalid prefix restore target (slot {slot}, len {len})")));
        }
        // validate the entry before touching the slot, so a mismatch cannot disturb it
        {
            let entry = self.snaps.as_ref().unwrap().slots[snap as usize].as_ref();
            match entry {
                Some(e) if e.len == len => {}
                _ => {
                    let got = entry.map_or(0, |e| e.len);
                    return Err(ModelError::Request(format!("snapshot {snap} holds length {got} not {len}")));
                }
            }
        }
        self.begin_slot(slot)?;
        // take ownership of the entry for the restore, so field methods can borrow `self` freely; put it back
        // either way (a failed restore leaves the snapshot intact for another try)
        let entry = self.snaps.as_mut().unwrap().slots[snap as usize].take().unwrap();
        let result = (|| -> Result<(), ModelError> {
            for (ord, k) in entry.kda.iter().enumerate() {
                self.pools[ord]
                    .restore_slot_state(slot, &k.s, k.parity, &k.conv, &self.stream)
                    .map_err(|e| ModelError::Config(format!("kda restore: {e}")))?;
            }
            // allocate fresh pages for `len` and copy the snapshot's page contents in logical order (the snapshot's
            // physical pages may be owned by other live slots by now, so the contents move, never the pages)
            self.ensure_pages(slot, len.saturating_sub(1))?;
            let np = self.slot_pages[slot].len();
            for (ord, cs) in entry.mla.iter().enumerate() {
                let c = &self.mla_caches[ord];
                let bb = MlaGatedCache::block_bytes(c.dtype);
                for i in 0..np {
                    let data = &cs.data[i * bb..(i + 1) * bb];
                    let p = self.slot_pages[slot][i] as usize;
                    unsafe {
                        self.stream
                            .memcpy_h2d_async(c.buf.ptr_at(p * bb), data.as_ptr(), bb)
                            .map_err(|e| ModelError::Config(format!("mla restore: {e}")))?;
                    }
                }
            }
            if let Some(m) = self.mtp.as_mut() {
                if let Some(mt) = &entry.mtp {
                    // in the stream's order (after `begin_slot`'s zero fill of the ring), straight from the entry, which outlives
                    // the copy: it goes back into its slot only after the sync below (or, on an error, unchanged)
                    let bytes = mt.taps.len() * 2;
                    if bytes > self.dev_taps[slot].bytes() {
                        return Err(ModelError::Config(format!("snapshot taps of {bytes} B exceed the ring")));
                    }
                    unsafe {
                        self.stream
                            .memcpy_h2d_async(self.dev_taps[slot].ptr(), mt.taps.as_ptr() as *const u8, bytes)
                            .map_err(|e| ModelError::Config(format!("taps upload: {e}")))?;
                    }
                    m.slots[slot].absorbed = mt.absorbed;
                    m.slots[slot].last_out = mt.last_out.clone();
                    m.slots[slot].head_dirty = true;
                }
            }
            self.sync()?;
            Ok(())
        })();
        if let Some(pool) = self.snaps.as_mut() {
            pool.slots[snap as usize] = Some(entry);
        }
        result
    }

    // ------------------------------------------------------------------------------------------------------ the trunk

    /// Upload the batch's row arrays.
    fn upload_batch(&mut self, b: &crate::rows::RowBatch) -> Result<(), ModelError> {
        if b.rows == 0 || b.rows > PREFILL_ROWS {
            return Err(ModelError::Request(format!("a forward takes 1..={PREFILL_ROWS} rows, got {}", b.rows)));
        }
        let (stage, st) = (&mut self.stage, &self.stream);
        put(stage, &mut self.s.ids, &b.ids, st, "ids")?;
        put(stage, &mut self.s.row_slot, &b.row_slot, st, "row_slot")?;
        put(stage, &mut self.s.row_pos, &b.positions, st, "positions")?;
        let n_seq = b.n_seq();
        let mut start = vec![0i32; n_seq + 1];
        for (i, s) in b.seq_start.iter().enumerate() {
            start[i] = *s;
        }
        start[n_seq] = b.rows as i32;
        put(stage, &mut self.s.seq_start, &start, st, "seq_start")?;
        put(stage, &mut self.s.seq_slot, &b.seq_slot, st, "seq_slot")?;
        Ok(())
    }

    /// Wait for the stream; the host bytes of the uploads queued on it are free for reuse from here on.
    fn sync(&mut self) -> Result<(), ModelError> {
        self.stage.sync(&self.stream).map_err(|e| ModelError::Config(format!("sync: {e}")))
    }

    fn row_index(&self, rows: usize) -> RowIndex {
        RowIndex { slot: self.s.row_slot.subview(0, rows * 4), pos: self.s.row_pos.subview(0, rows * 4), rows }
    }

    /// Run the trunk kernels over `rows` rows whose batch arrays already sit in `self.s` (see [`Self::upload_batch`]).
    /// `prefill` commits each KDA layer's piece all-rows; a window leaves the records. Every kernel reads only fixed device
    /// buffers plus the scalar `(rows, n_seq)` — a window's entire launch graph is determined by [`WindowGraphKey`].
    fn trunk_kernels(&mut self, rows: usize, n_seq: usize, first_seq_slot: i32, prefill: bool) -> Result<(), ModelError> {
        let st = &self.stream;
        let cfg = self.cfg.clone();
        let h = cfg.hidden;
        let eps = cfg.rms_eps as f32;
        let ch = cfg.kda.channels();
        let heads = cfg.kda.heads;
        let dims = kda::Dims { heads: cfg.kda.heads, hidden: h };

        unsafe {
            glue::embed(
                self.embed_dev.ptr().cast_const(),
                self.s.ids.ptr().cast_const(),
                self.s.hidden.ptr(),
                rows,
                h,
                cfg.vocab,
                st,
            )?;
        }

        // layer 0's input norm; every later layer's input norm is fused into its predecessor's FFN residual
        unsafe {
            glue::rmsnorm(self.s.hidden.ptr(), self.dev.layers[0].input_norm.ptr(), self.s.norm.ptr(), rows, h, eps, st)?;
        }
        for l in 0..cfg.layers.len() {
            let layer = &self.dev.layers[l];
            match &layer.mixer {
                MixerDevice::Kda(k) => {
                    let ord = self.kda_ord[l];
                    // a prefill chunk commits this layer below, before the next layer runs: every layer shares one record
                    let rec = if prefill { &mut self.prefill_rec } else { &mut self.recs[ord] };
                    // the qkv projection: a prefill chunk's in bf16 (the pinned GEMM's fp32 sums rounded to nearest even, the
                    // rounding `kda::prep` applies to a window's f32 rows: the same values reach the conv either way)
                    let qkv_dtype = if prefill { Dtype::BF16 } else { Dtype::F32 };
                    if prefill {
                        k.qkv
                            .forward_prefill_pinned(
                                &self.s.norm,
                                &mut self.s.qkv,
                                rows,
                                Dtype::BF16,
                                qkv_dtype,
                                &mut self.s.prefill_ws,
                                st,
                            )
                            .map_err(|e| ModelError::Config(format!("kda qkv prefill L{l} m={rows}: {e}")))?;
                        k.f.forward_prefill_pinned(
                            &self.s.norm,
                            &mut self.s.f_out,
                            rows,
                            Dtype::BF16,
                            Dtype::F16,
                            &mut self.s.prefill_ws,
                            st,
                        )
                        .map_err(|e| ModelError::Config(format!("kda f prefill L{l} m={rows}: {e}")))?;
                        k.g.forward_prefill_pinned(
                            &self.s.norm,
                            &mut self.s.g_out,
                            rows,
                            Dtype::BF16,
                            Dtype::F16,
                            &mut self.s.prefill_ws,
                            st,
                        )
                        .map_err(|e| ModelError::Config(format!("kda g prefill L{l} m={rows}: {e}")))?;
                    } else {
                        k.qkv.forward(&self.s.norm, &mut self.s.qkv, rows, Dtype::BF16, qkv_dtype, &mut self.s.ws, st)?;
                        k.f.forward(&self.s.norm, &mut self.s.f_out, rows, Dtype::BF16, Dtype::F16, &mut self.s.ws, st)?;
                        k.g.forward(&self.s.norm, &mut self.s.g_out, rows, Dtype::BF16, Dtype::F16, &mut self.s.ws, st)?;
                    }
                    kda::gate_from_full_fg(
                        st,
                        &self.s.f_out,
                        &self.s.g_out,
                        &self.s.norm,
                        &k.b,
                        &k.a_log,
                        &k.dt_bias,
                        KDA_LOWER,
                        rows,
                        h,
                        heads,
                        &mut rec.g,
                        self.kda_ws.z_mut(),
                        &mut rec.beta,
                    )
                    .map_err(|e| ModelError::Config(format!("kda gate L{l} m={rows}: {e}")))?;
                    let w = k.view(&self.dev.kda_dummies);
                    // one batch per forward: `upload_batch` already filled `seq_start`/`seq_slot` (no per-layer
                    // `BatchBuf` alloc/free, no duplicate `gate_from_full_fg`)
                    let batch = kda::Batch { seq_start: &self.s.seq_start, slot: &self.s.seq_slot, n_seq, rows };
                    kda::prep(st, dims, &w, &self.s.qkv, qkv_dtype, &mut self.pools[ord], &batch, rec, &mut self.kda_ws)
                        .map_err(|e| ModelError::Config(format!("kda prep L{l} m={rows}: {e}")))?;
                    if prefill {
                        // the 16-row parallel chunked recurrence over the whole chunk (`window_chunked`), single
                        // sequence of `rows` rows in `first_seq_slot`
                        kda::window_chunked(
                            st,
                            dims,
                            &self.kda_params,
                            &mut self.pools[ord],
                            first_seq_slot as usize,
                            rows,
                            rec,
                            &mut self.kda_ws,
                        )
                        .map_err(|e| ModelError::Config(format!("kda chunk L{l} m={rows}: {e}")))?;
                    } else {
                        kda::window(st, dims, &self.kda_params, &mut self.pools[ord], &batch, rec, &mut self.kda_ws, true)
                            .map_err(|e| ModelError::Config(format!("kda window L{l} m={rows}: {e}")))?;
                    }
                    kda::norm(st, dims, &w, &self.kda_params, rows, &self.kda_ws, &mut self.s.gated)
                        .map_err(|e| ModelError::Config(format!("kda norm L{l} m={rows}: {e}")))?;
                    if prefill {
                        kda::commit(
                            st,
                            dims,
                            &w,
                            &self.kda_params,
                            &mut self.pools[ord],
                            &batch,
                            rec,
                            &mut self.kda_ws,
                            None,
                            true,
                        )
                        .map_err(|e| ModelError::Config(format!("kda commit L{l} m={rows}: {e}")))?;
                    }
                    if prefill {
                        k.o.forward_prefill_pinned(
                            &self.s.gated,
                            &mut self.s.a,
                            rows,
                            Dtype::F16,
                            Dtype::BF16,
                            &mut self.s.prefill_ws,
                            st,
                        )
                        .map_err(|e| ModelError::Config(format!("kda o prefill L{l} m={rows}: {e}")))?;
                    } else {
                        k.o.forward(&self.s.gated, &mut self.s.a, rows, Dtype::F16, Dtype::BF16, &mut self.s.ws, st)?;
                    }
                    // hidden = bf16(hidden + a) and norm = rmsnorm(hidden; post_norm) in one kernel
                    unsafe {
                        glue::add_rmsnorm(
                            self.s.hidden.ptr().cast_const(),
                            self.s.a.ptr().cast_const(),
                            layer.post_norm.ptr().cast_const(),
                            self.s.hidden.ptr(),
                            self.s.norm.ptr(),
                            rows,
                            h,
                            eps,
                            st,
                        )?;
                    }
                }
                MixerDevice::Mla(ml) => {
                    let ord = self.mla_ord[l];
                    if prefill {
                        ml.q_a
                            .forward_prefill_pinned(
                                &self.s.norm,
                                &mut self.s.q_a,
                                rows,
                                Dtype::BF16,
                                Dtype::BF16,
                                &mut self.s.prefill_ws,
                                st,
                            )
                            .map_err(|e| ModelError::Config(format!("mla q_a prefill L{l} m={rows}: {e}")))?;
                    } else {
                        ml.q_a.forward(&self.s.norm, &mut self.s.q_a, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                    }
                    unsafe {
                        glue::rmsnorm(self.s.q_a.ptr(), ml.q_a_norm.ptr(), self.s.q_a.ptr(), rows, cfg.mla.q_lora_rank, eps, st)?
                    };
                    if prefill {
                        ml.q_b
                            .forward_prefill_pinned(
                                &self.s.q_a,
                                &mut self.s.q,
                                rows,
                                Dtype::BF16,
                                Dtype::BF16,
                                &mut self.s.prefill_ws,
                                st,
                            )
                            .map_err(|e| ModelError::Config(format!("mla q_b prefill L{l} m={rows}: {e}")))?;
                    } else {
                        ml.q_b.forward(&self.s.q_a, &mut self.s.q, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                    }
                    if prefill {
                        ml.kv_a
                            .forward_prefill_pinned(
                                &self.s.norm,
                                &mut self.s.kv,
                                rows,
                                Dtype::BF16,
                                Dtype::BF16,
                                &mut self.s.prefill_ws,
                                st,
                            )
                            .map_err(|e| ModelError::Config(format!("mla kv_a prefill L{l} m={rows}: {e}")))?;
                    } else {
                        ml.kv_a.forward(&self.s.norm, &mut self.s.kv, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                    }
                    let ri = self.row_index(rows);
                    mla_gated::mla_gated_rows(
                        &self.s.norm,
                        &self.s.q,
                        &self.s.kv,
                        &ml.kv_a_norm,
                        &self.cos,
                        &self.sin,
                        &ml.w,
                        &mut self.mla_caches[ord],
                        &self.table,
                        &ri,
                        rows,
                        &mut self.s.q_lat,
                        &mut self.s.q_rot,
                        &mut self.s.o_lat,
                        &mut self.s.mla_out,
                        &mut self.s.gated,
                        st,
                    )
                    .map_err(|e| ModelError::Config(format!("mla_gated_rows L{l} m={rows}: {e}")))?;
                    if prefill {
                        ml.dense
                            .forward_prefill_pinned(
                                &self.s.gated,
                                &mut self.s.a,
                                rows,
                                Dtype::BF16,
                                Dtype::BF16,
                                &mut self.s.prefill_ws,
                                st,
                            )
                            .map_err(|e| ModelError::Config(format!("mla dense prefill L{l} m={rows}: {e}")))?;
                    } else {
                        ml.dense.forward(&self.s.gated, &mut self.s.a, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                    }
                    unsafe {
                        glue::add_rmsnorm(
                            self.s.hidden.ptr().cast_const(),
                            self.s.a.ptr().cast_const(),
                            layer.post_norm.ptr().cast_const(),
                            self.s.hidden.ptr(),
                            self.s.norm.ptr(),
                            rows,
                            h,
                            eps,
                            st,
                        )?;
                    }
                }
            }

            match &layer.ffn {
                FfnDevice::Dense(d) => {
                    if prefill {
                        d.gate
                            .forward_prefill_pinned(
                                &self.s.norm,
                                &mut self.s.g,
                                rows,
                                Dtype::BF16,
                                Dtype::BF16,
                                &mut self.s.prefill_ws,
                                st,
                            )
                            .map_err(|e| ModelError::Config(format!("dense gate prefill L{l} m={rows}: {e}")))?;
                        d.up.forward_prefill_pinned(
                            &self.s.norm,
                            &mut self.s.u,
                            rows,
                            Dtype::BF16,
                            Dtype::BF16,
                            &mut self.s.prefill_ws,
                            st,
                        )
                        .map_err(|e| ModelError::Config(format!("dense up prefill L{l} m={rows}: {e}")))?;
                    } else {
                        d.gate.forward(&self.s.norm, &mut self.s.g, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                        d.up.forward(&self.s.norm, &mut self.s.u, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                    }
                    unsafe {
                        glue::swiglu(self.s.g.ptr(), self.s.u.ptr(), self.s.a.ptr(), rows * cfg.ffn.dense_intermediate, st)?
                    };
                    if prefill {
                        d.down
                            .forward_prefill_pinned(
                                &self.s.a,
                                &mut self.s.g,
                                rows,
                                Dtype::BF16,
                                Dtype::BF16,
                                &mut self.s.prefill_ws,
                                st,
                            )
                            .map_err(|e| ModelError::Config(format!("dense down prefill L{l} m={rows}: {e}")))?;
                    } else {
                        d.down.forward(&self.s.a, &mut self.s.g, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                    }
                    // hidden = bf16(hidden + g) fused with the next layer's input norm (or the final norm on the last layer)
                    let (next_w, next_dst) = if l + 1 < cfg.layers.len() {
                        (&self.dev.layers[l + 1].input_norm, &mut self.s.norm)
                    } else {
                        (&self.dev.final_norm, &mut self.s.final_normed)
                    };
                    unsafe {
                        glue::add_rmsnorm(
                            self.s.hidden.ptr().cast_const(),
                            self.s.g.ptr().cast_const(),
                            next_w.ptr().cast_const(),
                            self.s.hidden.ptr(),
                            next_dst.ptr(),
                            rows,
                            h,
                            eps,
                            st,
                        )?;
                    }
                }
                FfnDevice::Moe(moe) => {
                    let router = &moe.router;
                    self.s.moe_logits_bf16(&self.stream, router, h, cfg.ffn.routed_experts, rows)?;
                    let limit_r = cfg.routed_swiglu_limit(l);
                    let limit_s = cfg.shared_swiglu_limit(l);
                    let shape = fmoe::Shape {
                        t: rows,
                        ne: cfg.ffn.routed_experts,
                        top_k: cfg.ffn.experts_per_token,
                        h,
                        i: cfg.ffn.moe_intermediate,
                        kc: BAILING_KC,
                        max_tokens: if prefill { PREFILL_ROWS } else { ROWS },
                    };
                    let gr = Some(fmoe::GroupRoute {
                        expert_bias: Some(&moe.expert_bias),
                        n_group: cfg.ffn.n_group,
                        topk_group: cfg.ffn.topk_group,
                        scale: cfg.ffn.routed_scaling_factor as f32,
                    });
                    let limits = fmoe::SwigluLimit { routed: limit_r, shared: limit_s };
                    if prefill {
                        // sorted-token grouped MoE: every expert's tile decodes once per 64 routed tokens (no split-K
                        // decode groups), row-invariant across chunkings
                        fmoe::prefill(
                            &self.s.norm,
                            &self.s.moe_logits,
                            &self.s.moe_shared_gate,
                            &moe.tables(),
                            &mut self.s.moe_out,
                            shape,
                            gr,
                            limits,
                            &mut self.s.moe_pws,
                            st,
                        )?;
                    } else {
                        fmoe::forward(
                            &self.s.norm,
                            &self.s.moe_logits,
                            &self.s.moe_shared_gate,
                            &moe.tables(),
                            &mut self.s.moe_out,
                            shape,
                            gr,
                            limits,
                            &mut self.s.moe_ws,
                            st,
                        )?;
                    }
                    let (next_w, next_dst) = if l + 1 < cfg.layers.len() {
                        (&self.dev.layers[l + 1].input_norm, &mut self.s.norm)
                    } else {
                        (&self.dev.final_norm, &mut self.s.final_normed)
                    };
                    unsafe {
                        glue::add_rmsnorm(
                            self.s.hidden.ptr().cast_const(),
                            self.s.moe_out.ptr().cast_const(),
                            next_w.ptr().cast_const(),
                            self.s.hidden.ptr(),
                            next_dst.ptr(),
                            rows,
                            h,
                            eps,
                            st,
                        )?;
                    }
                }
            }
        }

        // head: the last layer's FFN residual fused its norm into `final_normed`, so only the logits remain.
        // Prefill chunks need no logits (the requested chunk's last row runs the head once in `prefill_chunk`).
        if !prefill {
            self.dev.lm_head.forward(
                &self.s.final_normed,
                &mut self.s.logits,
                rows,
                Dtype::BF16,
                Dtype::F16,
                &mut self.s.ws,
                st,
            )?;
        }
        Ok(())
    }

    /// The decode/verify kernel body: the trunk plus the LM head and the top-k queue (the part [`WindowGraphKey`] captures).
    fn trunk_decode_kernels(&mut self, rows: usize, n_seq: usize, count: usize) -> Result<(), ModelError> {
        self.trunk_kernels(rows, n_seq, 0, false)?;
        self.queue_topk(rows, count)
    }

    /// Upload the batch and run the trunk. `count` is the top-k depth the decode/verify path queues (unused when `prefill`).
    fn forward_trunk(&mut self, b: &crate::rows::RowBatch, prefill: bool, count: usize) -> Result<(), ModelError> {
        let rows = b.rows;
        if !prefill && rows > ROWS {
            return Err(ModelError::Request(format!("a decode/verify window takes 1..={ROWS} rows, got {rows}")));
        }
        self.upload_batch(b)?;
        if prefill {
            self.trunk_kernels(rows, b.n_seq(), b.seq_slot[0], true)?;
            if self.mtp.is_some() {
                // prefill keeps the tap ring synchronously (the sync is amortized over a chunk); decode/verify
                // defer the device copies to `window_candidates`, which shares the top-k's sync
                self.store_taps(b, rows)?;
            }
        } else {
            let n_seq = b.n_seq();
            let key = WindowGraphKey { rows, n_seq, count };
            self.run_window_graph(key, |m| m.trunk_decode_kernels(rows, n_seq, count))?;
        }
        Ok(())
    }

    /// Run `body` (the trunk decode/verify kernel launches) for a window shape: eagerly on first sight (so lazy kernel
    /// initialization happens outside capture), captured into a [`Graph`] on the second sight and replayed from then on.
    /// A refused capture silently falls back to eager runs.
    fn run_window_graph(
        &mut self,
        key: WindowGraphKey,
        body: impl Fn(&mut Self) -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        if !self.use_graphs {
            return body(self);
        }
        if self.graphs.contains_key(&key) {
            self.graphs[&key].launch(&self.stream)?;
            return Ok(());
        }
        if !self.seen_shapes.contains(&key) {
            body(self)?;
            self.seen_shapes.insert(key);
            return Ok(());
        }
        let mut map = std::mem::take(&mut self.graphs);
        let r = self.capture_and_launch(&mut map, key, body);
        self.graphs = map;
        r
    }

    // not `entry()`: a capture the runtime refuses must leave the map untouched and run the work eagerly instead
    #[allow(clippy::map_entry)]
    fn capture_and_launch(
        &mut self,
        map: &mut HashMap<WindowGraphKey, Graph>,
        key: WindowGraphKey,
        body: impl Fn(&mut Self) -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        let cap = match Graph::begin(&self.stream) {
            Ok(c) => c,
            Err(e) if e.is_fatal() => return Err(ModelError::Cuda(e)),
            Err(e) => {
                eprintln!("rexl3 level=warn ling31 graph capture refused ({e}): running decode windows eagerly");
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
                eprintln!("rexl3 level=warn ling31 {e}: running decode windows eagerly");
                self.use_graphs = false;
                return body(self);
            }
        }
        // always launch the instantiated graph on the capture round too
        map[&key].launch(&self.stream)?;
        Ok(())
    }

    /// Keep the trunk hidden rows of the batch (after the final norm, `final_normed` — the reference applies
    /// `norm(hidden_states)` before the MTP layers, so the taps are the NORMED trunk hidden) per (slot, position), in the
    /// device rings. Prefill keeps them synchronously (the sync is amortized over a chunk); decode/verify defer the device
    /// copies to `window_candidates`, which shares the top-k's sync.
    fn store_taps(&mut self, b: &crate::rows::RowBatch, rows: usize) -> Result<(), ModelError> {
        self.sync()?;
        self.record_taps(b, rows)
    }

    /// Copy the last `rows` normed trunk hidden rows into the device tap rings (D2D, on the stream). No host round-trip.
    fn record_taps(&mut self, b: &crate::rows::RowBatch, rows: usize) -> Result<(), ModelError> {
        let h = self.cfg.hidden;
        let row_bytes = h * 2;
        for r in 0..rows {
            let slot = b.row_slot[r] as usize;
            let pos = b.positions[r] as usize;
            let idx = pos % CAP_TAPS;
            self.dev_taps[slot]
                .copy_from_at(idx * row_bytes, &self.s.final_normed, r * row_bytes, row_bytes, &self.stream)
                .map_err(|e| ModelError::Config(format!("taps d2d: {e}")))?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------------ commits

    pub fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.flush()?;
        let st = &self.stream;
        for p in &mut self.pools {
            p.reset_slot(slot, st).map_err(|e| ModelError::Config(format!("kda reset: {e}")))?;
        }
        self.release_pages(slot);
        if let Some(m) = self.mtp.as_mut() {
            m.slots[slot].absorbed = 0;
            m.slots[slot].last_out = None;
            m.slots[slot].head_dirty = true;
            self.dev_taps[slot].memset_async(0, &self.stream).map_err(|e| ModelError::Config(format!("taps zero: {e}")))?;
        }
        Ok(())
    }

    pub fn end_slot(&mut self, slot: usize) {
        self.pending.remove(&slot);
        self.release_pages(slot);
        if let Some(m) = self.mtp.as_mut() {
            m.slots[slot].absorbed = 0;
            m.slots[slot].last_out = None;
            m.slots[slot].head_dirty = true;
        }
    }

    pub fn commit(&mut self, slot: usize, keep: usize) {
        self.pending.insert(slot, keep);
    }

    /// Apply the pending commits: per KDA layer, flip or replay the state of every sequence of the last window by its keep count.
    pub fn flush(&mut self) -> Result<(), ModelError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let Some(w) = self.last_window.take() else {
            self.pending.clear();
            return Ok(());
        };
        let n_seq = w.seq_slot.len();
        let mut keep = vec![0i32; n_seq];
        for (i, &sl) in w.seq_slot.iter().enumerate() {
            if let Some(&k) = self.pending.get(&(sl as usize)) {
                keep[i] = k.min(w.seq_len[i]) as i32;
            }
        }
        self.pending.clear();
        if keep.iter().all(|&k| k == 0) {
            return Ok(());
        }
        let mut start = vec![0i32; n_seq + 1];
        for i in 0..n_seq {
            start[i + 1] = start[i] + w.seq_len[i] as i32;
        }
        let (stage, st) = (&mut self.stage, &self.stream);
        put(stage, &mut self.s.keep, &keep, st, "keep")?;
        put(stage, &mut self.s.seq_start, &start, st, "seq_start")?;
        put(stage, &mut self.s.seq_slot, &w.seq_slot, st, "seq_slot")?;
        let dims = kda::Dims { heads: self.cfg.kda.heads, hidden: self.cfg.hidden };
        let batch = kda::Batch { seq_start: &self.s.seq_start, slot: &self.s.seq_slot, n_seq, rows: w.rows };
        for l in 0..self.cfg.layers.len() {
            let ord = self.kda_ord[l];
            if ord == usize::MAX {
                continue;
            }
            let MixerDevice::Kda(k) = &self.dev.layers[l].mixer else { continue };
            let wv = k.view(&self.dev.kda_dummies);
            kda::commit(
                &self.stream,
                dims,
                &wv,
                &self.kda_params,
                &mut self.pools[ord],
                &batch,
                &self.recs[ord],
                &mut self.kda_ws,
                Some(&self.s.keep),
                true,
            )?;
        }
        // The commit kernels (partial replay + conv shift + parity flip) publish the recurrent state the next window
        // consumes. Force completion here: without the barrier, multi-draft windows (rows >= 3, a partial accept with
        // `keep < len`) could observe a not-yet-settled state, which showed up as run-to-run nondeterministic divergence
        // at the first post-prefill draft round.
        self.sync()?;
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------------ candidates

    pub fn candidates_of(&mut self, rows: usize, count: usize) -> Result<Candidates, ModelError> {
        if count == 0 || count > 64 {
            return Err(ModelError::Request("candidate count must be 1..=64".into()));
        }
        self.queue_topk(rows, count)?;
        self.finish_topk(rows, count)
    }

    /// Queue the top-k over `logits` for `rows` rows (the kernels run on `self.stream`; nothing is downloaded yet).
    fn queue_topk(&mut self, rows: usize, count: usize) -> Result<(), ModelError> {
        glm_glue::topk_f16(
            &self.s.logits,
            rows,
            self.cfg.vocab,
            self.cfg.vocab,
            count,
            &mut self.s.topk_ids,
            &mut self.s.topk_vals,
            &mut self.s.topk_ws,
            &self.stream,
        )
        .map_err(|e| ModelError::Config(format!("topk: {e}")))
    }

    /// Sync the stream and download the queued top-k ids/values. The sync also makes the already-queued forward's
    /// outputs readable, so callers that need them (e.g. `window_candidates`' MTP taps) must download after this.
    fn finish_topk(&mut self, rows: usize, count: usize) -> Result<Candidates, ModelError> {
        self.sync()?;
        let mut ids = vec![0i32; rows * count];
        let mut vals = vec![0f32; rows * count];
        self.s.topk_ids.download(&mut ids).map_err(|e| ModelError::Config(format!("topk ids: {e}")))?;
        self.s.topk_vals.download(&mut vals).map_err(|e| ModelError::Config(format!("topk vals: {e}")))?;
        Ok(Candidates { rows, count, ids, vals })
    }

    /// Candidates for every row of a window step.
    pub fn window_candidates(&mut self, b: &crate::rows::RowBatch, count: usize) -> Result<Candidates, ModelError> {
        self.flush()?;
        let max_pos = b.positions.iter().copied().max().unwrap_or(0) as usize;
        let mut seen = std::collections::BTreeSet::new();
        for &sl in &b.seq_slot {
            if seen.insert(sl) {
                self.ensure_pages(sl as usize, max_pos)?;
            }
        }
        self.forward_trunk(b, false, count)?;
        let mut seq_len = Vec::with_capacity(b.n_seq());
        for i in 0..b.n_seq() {
            let next = if i + 1 < b.n_seq() { b.seq_start[i + 1] } else { b.rows as i32 };
            seq_len.push((next - b.seq_start[i]) as usize);
        }
        self.last_window = Some(WindowRec { seq_slot: b.seq_slot.clone(), seq_len, rows: b.rows });
        // the top-k was queued inside the trunk body (graph or eager), before any sync: one `cudaStreamSynchronize`
        // covers both downloads below (the MTP taps are copied from the same finished forward's `final_normed`)
        let c = self.finish_topk(b.rows, count)?;
        if self.mtp.is_some() {
            self.record_taps(b, b.rows)?;
        }
        Ok(c)
    }

    /// Prefill `tokens` (positions `start..start + tokens.len()`) of one sequence; with `want`, the candidates of the last row.
    pub fn prefill_chunk(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
    ) -> Result<Option<Candidates>, ModelError> {
        self.flush()?;
        let n = tokens.len();
        let h = self.cfg.hidden;
        let mut at = 0;
        let mut last_rows = 0;
        while at < n {
            let end = (at + PREFILL_ROWS).min(n);
            last_rows = end - at;
            self.ensure_pages(slot, start + end - 1)?;
            let b = crate::rows::RowBatch::from_requests(&[gb_engine::RowRequest {
                slot,
                tokens: tokens[at..end].to_vec(),
                first_position: (start + at) as u64,
            }]);
            self.forward_trunk(&b, true, 0)?;
            at = end;
        }
        // every KDA chunk is committed inside `forward_trunk` (prefill), so there is no window to flush later
        self.last_window = None;
        match want {
            None => Ok(None),
            Some(count) => {
                if n == 0 {
                    return Ok(None);
                }
                // only the single last row's logits (intermediate chunks need none; `forward_trunk` skips the head then)
                let last = self.s.final_normed.subview((last_rows - 1) * h * 2, h * 2);
                self.dev.lm_head.forward(&last, &mut self.s.logits, 1, Dtype::BF16, Dtype::F16, &mut self.s.ws, &self.stream)?;
                Ok(Some(self.candidates_of(1, count)?))
            }
        }
    }

    // ------------------------------------------------------------------------------------------------------ MTP drafter

    /// Draft up to `max` tokens of the sequence `tokens` (all committed, ending with the last fed token) via `L72`.
    pub fn mtp_propose(&mut self, slot: usize, tokens: &[u32], max: usize) -> Result<Option<Vec<u32>>, ModelError> {
        if self.mtp.is_none() || tokens.is_empty() || max == 0 {
            return Ok(None);
        }
        let n = tokens.len();
        let h = self.cfg.hidden;
        // The L72 layer at row `p` takes the trunk hidden of token `p` and the embedding of token `p + 1`
        // to predict token `p + 2` (modeling_bailing_moe_v3.py: roll(input_ids, -1) feeds token p+1 at row p).
        // The last prompt row has no next-token embedding yet, so it joins the draft chain instead of the absorb.
        let limit = n.saturating_sub(1);
        if limit == 0 {
            return Ok(None);
        }
        let mut absorbed = self.mtp.as_ref().unwrap().slots[slot].absorbed.min(limit);
        // the tap ring keeps the last `CAP_TAPS` trunk rows (positions `n - CAP_TAPS..n`); once a prompt outgrew it,
        // skip the rows whose taps are gone (the tail is re-absorbed so the head's GPU state stays consistent)
        absorbed = absorbed.max(limit.saturating_sub(CAP_TAPS - 1));
        while absorbed < limit {
            let hi = (absorbed + PREFILL_ROWS).min(limit);
            self.mtp_absorb_range(slot, tokens, absorbed, hi)?;
            absorbed = hi;
            self.mtp.as_mut().unwrap().slots[slot].absorbed = absorbed;
        }
        if self.mtp.as_ref().unwrap().slots[slot].absorbed < limit {
            return Ok(None);
        }
        // when the head's GPU register is stale (a slot was begun/restored and nothing was absorbed since), regenerate it
        // from the last row so `d0` never runs on stale hidden
        if self.mtp.as_ref().unwrap().slots[slot].head_dirty {
            let p = limit - 1;
            let idx = p % CAP_TAPS;
            let row_bytes = h * 2;
            self.s
                .hidden
                .copy_from_at(0, &self.dev_taps[slot], idx * row_bytes, row_bytes, &self.stream)
                .map_err(|e| ModelError::Config(format!("taps d2d: {e}")))?;
            let ids = vec![tokens[limit] as i32];
            self.ensure_pages(slot, p)?;
            self.mtp_absorb(&ids, &[slot as i32], &[p as i32], 1)?;
        }
        // `head_in` is the L72 output of the last absorbed row; it is now fresh
        self.mtp.as_mut().unwrap().slots[slot].head_dirty = false;
        // draft 0 from the last absorbed row (row n-2 predicts the token after the prompt)
        let mut drafts = Vec::with_capacity(max);
        let (d0, margin0) = self.mtp_head()?;
        drafts.push(d0);
        let tune = self.mtp.as_ref().unwrap().tune;
        // zero-GPU-cost prompt lookup continuation: when the gram ending in `d0` occurred earlier in the context, reuse
        // the tokens that followed it while the head is confident (no `mtp_chain` calls at all). A high-confidence head
        // tries the 4-gram repeat before spending chain compute; a low-margin head falls back to the 4/3-gram lookup
        // entirely. The former 2-gram fallback is gone: nearly every frequent pair (`", the"`, `"\n    "`, `" = "`)
        // matches somewhere in context, and its stuffed continuations cost full trunk verify rounds for little acceptance.
        if margin0 >= tune.chain_gate[0] && margin0 >= tune.pld_margin {
            if let Some(cont) = self.pld_continuation(tokens, d0, max - drafts.len(), 4) {
                drafts.extend(cont);
                return Ok(Some(drafts));
            }
        }
        // confidence-gated chain: only spend `mtp_chain` on drafts the head is sure about (margin >= gate means
        // p(d0) >= ~90%+); low-margin steps stop, and a chained draft is only pushed when its own margin clears
        // `accept_gate` (verifying a near-uniform head output wastes a full trunk round).
        let mut pos = n - 1;
        // the chain runs from the absorbed head's row, whose L72 output is `head_in`; the chain's own previous-output
        // register is `chain_in`, so `head_in` stays the absorb output for the next propose
        self.s
            .chain_in
            .copy_from(&self.s.head_in, h * 2, &self.stream)
            .map_err(|e| ModelError::Config(format!("chain start copy: {e}")))?;
        let mut margin = margin0;
        while drafts.len() < max && pos < self.max_context {
            let gate = if drafts.len() == 1 { tune.chain_gate[0] } else { tune.chain_gate[1] };
            if margin < gate {
                break;
            }
            let prev_tok = drafts[drafts.len() - 1] as i32;
            self.ensure_pages(slot, pos)?;
            self.mtp_chain(slot, prev_tok, pos)?;
            // `mtp_chain` leaves the new L72 row in `chain_in`; the head scores `head_in`, so move the chain's
            // output over before reading the next draft (a D2D copy of one row).
            self.s
                .head_in
                .copy_from(&self.s.chain_in, h * 2, &self.stream)
                .map_err(|e| ModelError::Config(format!("chain head copy: {e}")))?;
            let (d, m) = self.mtp_head()?;
            if m < tune.accept_gate {
                break;
            }
            drafts.push(d);
            margin = m;
            pos += 1;
        }
        // low-margin fallback: 4/3-gram prompt-lookup continuation
        if margin0 < tune.chain_gate[0] && margin0 >= tune.pld_margin {
            if let Some(cont) = self.pld_continuation(tokens, d0, max - drafts.len(), 3) {
                drafts.extend(cont);
                return Ok(Some(drafts));
            }
        }
        Ok(Some(drafts))
    }

    /// Backward search over the last 4096 committed tokens for the most recent occurrence of the 4-gram (then 3-gram)
    /// `(tokens[n-3], tokens[n-2], tokens[n-1], d0)` / `(tokens[n-2], tokens[n-1], d0)` and the tokens that followed it
    /// (up to `max_add`, capped at 2, of them). A match means the model's `d0` repeats an earlier phrase/identifier/code
    /// line, so its continuation drafts are accepted at very high rate with zero GPU work. `min_gram` selects the shortest
    /// gram to try (4 for the high-confidence head path, 3 for the low-margin fallback). There is intentionally no
    /// 2-gram fallback: nearly every frequent pair occurs somewhere in the last 4096 tokens, so a 2-gram match carries
    /// almost no predictive signal and its stuffed continuations force the trunk to verify junk.
    fn pld_continuation(&self, tokens: &[u32], d0: u32, max_add: usize, min_gram: usize) -> Option<Vec<u32>> {
        if max_add == 0 || tokens.len() < 4 {
            return None;
        }
        let n = tokens.len();
        let lo = n.saturating_sub(4096 + 4);
        // 4-gram first (a longer match predicts the continuation better)
        if min_gram <= 4 && n >= 5 {
            let q = [tokens[n - 3], tokens[n - 2], tokens[n - 1], d0];
            let hi = n - 6;
            if lo <= hi {
                let mut m = hi;
                loop {
                    if tokens[m] == q[0] && tokens[m + 1] == q[1] && tokens[m + 2] == q[2] && tokens[m + 3] == q[3] {
                        let take = max_add.min(n - (m + 4)).min(2);
                        return Some(tokens[m + 4..m + 4 + take].to_vec());
                    }
                    if m == lo {
                        break;
                    }
                    m -= 1;
                }
            }
        }
        // 3-gram
        if min_gram <= 3 {
            let q = [tokens[n - 2], tokens[n - 1], d0];
            let hi = n - 4;
            if lo <= hi {
                let mut m = hi;
                loop {
                    if tokens[m] == q[0] && tokens[m + 1] == q[1] && tokens[m + 2] == q[2] {
                        let take = max_add.min(n - (m + 3)).min(2);
                        return Some(tokens[m + 3..m + 3 + take].to_vec());
                    }
                    if m == lo {
                        break;
                    }
                    m -= 1;
                }
            }
        }
        None
    }

    /// `argmax(lm_head(h))` over the single row `head_in` (already on the GPU), plus the top-1 vs top-2 logit margin:
    /// the row is ALREADY `final_layernorm`-ed (the reference runs `lm_head(mtp_hidden_states)` on the layer outputs,
    /// which end in `final_layernorm`), so no norm is applied here. When the restricted draft head is built, it scores only
    /// its kept columns (bit-identical to the full head's) and the chosen column is mapped back to the full-vocab token id.
    /// The top-2 runs on the GPU; only 8 + 8 bytes are downloaded.
    fn mtp_head(&mut self) -> Result<(u32, f32), ModelError> {
        let restricted = self.mtp.as_ref().is_some_and(|m| m.sub.is_some());
        if restricted {
            let Scratch { head_in, topk_ids, topk_vals, topk_ws, .. } = &mut self.s;
            let (st, stage) = (&self.stream, &mut self.stage);
            let m = self.mtp.as_mut().unwrap();
            let sh = m.sub.as_mut().unwrap();
            sh.w.forward_k2(head_in, &mut sh.logits, 1, Dtype::BF16, Dtype::F16, sh.k2, &mut sh.ws, st)?;
            glm_glue::topk_f16(&sh.logits, 1, sh.cols, sh.cols, 2, &mut *topk_ids, &mut *topk_vals, &mut *topk_ws, st)?;
            stage.sync(st).map_err(|e| ModelError::Config(format!("sync: {e}")))?;
            let mut ids = [0i32; 2];
            let mut vals = [0f32; 2];
            topk_ids.subview(0, 8).download(&mut ids).map_err(|e| ModelError::Config(format!("topk download: {e}")))?;
            topk_vals.subview(0, 8).download(&mut vals).map_err(|e| ModelError::Config(format!("topk download: {e}")))?;
            let col = ids[0] as usize;
            if col >= sh.cols || sh.map[col] as u32 >= self.cfg.vocab as u32 {
                return Err(ModelError::Request("MTP head produced a padding id".into()));
            }
            return Ok((sh.map[col] as u32, vals[0] - vals[1]));
        }
        self.dev.lm_head.forward(
            &self.s.head_in,
            &mut self.s.logits,
            1,
            Dtype::BF16,
            Dtype::F16,
            &mut self.s.ws,
            &self.stream,
        )?;
        glm_glue::topk_f16(
            &self.s.logits,
            1,
            self.cfg.vocab,
            self.cfg.vocab,
            2,
            &mut self.s.topk_ids,
            &mut self.s.topk_vals,
            &mut self.s.topk_ws,
            &self.stream,
        )?;
        self.sync()?;
        let mut ids = [0i32; 2];
        let mut vals = [0f32; 2];
        self.s.topk_ids.subview(0, 8).download(&mut ids).map_err(|e| ModelError::Config(format!("topk download: {e}")))?;
        self.s.topk_vals.subview(0, 8).download(&mut vals).map_err(|e| ModelError::Config(format!("topk download: {e}")))?;
        let best = ids[0] as u32;
        if best >= self.cfg.vocab as u32 {
            return Err(ModelError::Request("MTP head produced a padding id".into()));
        }
        Ok((best, vals[0] - vals[1]))
    }

    /// Run `L72` over `rows` absorbed rows whose trunk hidden rows are already in `self.s.hidden` (the caller copied them
    /// from the tap ring): `ids` (tokens), positions `pos`. The last row's L72 output is left on the GPU in `head_in`
    /// (no download, no sync).
    fn mtp_absorb(&mut self, ids: &[i32], slots: &[i32], pos: &[i32], rows: usize) -> Result<(), ModelError> {
        let h = self.cfg.hidden;
        put(&mut self.stage, &mut self.s.ids, ids, &self.stream, "ids")?;
        unsafe {
            glue::embed(
                self.embed_dev.ptr().cast_const(),
                self.s.ids.ptr().cast_const(),
                self.s.e_next.ptr(),
                rows,
                h,
                self.cfg.vocab,
                &self.stream,
            )?;
        }
        put(&mut self.stage, &mut self.s.row_slot, slots, &self.stream, "slots")?;
        put(&mut self.stage, &mut self.s.row_pos, pos, &self.stream, "pos")?;
        self.mtp_layer_forward(rows)?;
        let last = self.s.hidden.subview((rows - 1) * h * 2, h * 2);
        self.s.head_in.copy_from(&last, h * 2, &self.stream).map_err(|e| ModelError::Config(format!("head copy: {e}")))?;
        Ok(())
    }

    /// Absorb the L72 rows `lo..hi` of `tokens` (row `p` takes token `p + 1`'s embedding and position `p`'s trunk tap),
    /// copying the taps straight from the device ring into `hidden` (D2D; no host round-trip), leaving the last row's
    /// output in `head_in`. `PREFILL_ROWS` rows run per chunk; the absorb is row-invariant (every kernel on the L72 path
    /// is), so the chunk size never changes the resulting bits.
    fn mtp_absorb_range(&mut self, slot: usize, tokens: &[u32], lo: usize, hi: usize) -> Result<(), ModelError> {
        let rows = hi - lo;
        if rows == 0 {
            return Ok(());
        }
        let mut ids = Vec::with_capacity(rows);
        let pos_v: Vec<i32> = (lo..hi).map(|p| p as i32).collect();
        for p in lo..hi {
            ids.push(tokens[p + 1] as i32);
        }
        // the tap ring wraps at `CAP_TAPS`: one contiguous D2D copy, or two across the wrap
        let row_bytes = self.cfg.hidden * 2;
        let start = lo % CAP_TAPS;
        if start + rows <= CAP_TAPS {
            self.s
                .hidden
                .copy_from_at(0, &self.dev_taps[slot], start * row_bytes, rows * row_bytes, &self.stream)
                .map_err(|e| ModelError::Config(format!("taps d2d: {e}")))?;
        } else {
            let first = CAP_TAPS - start;
            self.s
                .hidden
                .copy_from_at(0, &self.dev_taps[slot], start * row_bytes, first * row_bytes, &self.stream)
                .map_err(|e| ModelError::Config(format!("taps d2d: {e}")))?;
            self.s
                .hidden
                .copy_from_at(first * row_bytes, &self.dev_taps[slot], 0, (rows - first) * row_bytes, &self.stream)
                .map_err(|e| ModelError::Config(format!("taps d2d: {e}")))?;
        }
        self.ensure_pages(slot, hi - 1)?;
        let slot_v = vec![slot as i32; rows];
        self.mtp_absorb(&ids, &slot_v, &pos_v, rows)?;
        Ok(())
    }

    /// Pre-absorb the MTP drafter's L72 rows of `tokens` up to `tokens.len().saturating_sub(1)` (the last row joins the
    /// draft chain, never the absorb). Unlike `mtp_propose`, no draft chain runs: this only leaves the drafter ready to
    /// propose with every L72 MLA KV page `0..limit` populated, `slots[slot].absorbed = limit` and `head_dirty = false`
    /// (`head_in` holds the last absorbed row's output). The prefix cache calls this right before a snapshot cut, so the
    /// saved MTP state already contains the absorbed L72 pages and a warm hit only re-absorbs the tiny tail (or just
    /// re-generates `head_in`), instead of re-absorbing the whole prompt.
    pub fn mtp_absorb_up_to(&mut self, slot: usize, tokens: &[u32]) -> Result<(), ModelError> {
        if self.mtp.is_none() || tokens.len() < 2 {
            return Ok(());
        }
        let limit = tokens.len() - 1;
        let mut absorbed = self.mtp.as_ref().unwrap().slots[slot].absorbed.min(limit);
        // see `mtp_propose`: the tap ring only keeps the last `CAP_TAPS` trunk rows
        absorbed = absorbed.max(limit.saturating_sub(CAP_TAPS - 1));
        while absorbed < limit {
            let hi = (absorbed + PREFILL_ROWS).min(limit);
            self.mtp_absorb_range(slot, tokens, absorbed, hi)?;
            absorbed = hi;
            self.mtp.as_mut().unwrap().slots[slot].absorbed = absorbed;
        }
        self.mtp.as_mut().unwrap().slots[slot].head_dirty = false;
        Ok(())
    }

    /// One `L72` chain row: the token `tok`'s embedding (gathered on the device from `embed_dev`) and the previous chain
    /// row's L72 output (`chain_in`, still on the GPU) at `pos`.
    fn mtp_chain(&mut self, slot: usize, tok: i32, pos: usize) -> Result<(), ModelError> {
        let h = self.cfg.hidden;
        put(&mut self.stage, &mut self.s.ids, &[tok], &self.stream, "ids")?;
        unsafe {
            glue::embed(
                self.embed_dev.ptr().cast_const(),
                self.s.ids.ptr().cast_const(),
                self.s.e_next.ptr(),
                1,
                h,
                self.cfg.vocab,
                &self.stream,
            )?;
        }
        let mut row0 = self.s.hidden.subview(0, h * 2);
        row0.copy_from(&self.s.chain_in, h * 2, &self.stream).map_err(|e| ModelError::Config(format!("prev copy: {e}")))?;
        put(&mut self.stage, &mut self.s.row_slot, &[slot as i32], &self.stream, "slots")?;
        put(&mut self.stage, &mut self.s.row_pos, &[pos as i32], &self.stream, "pos")?;
        self.mtp_layer_forward(1)?;
        let out = self.s.hidden.subview(0, h * 2);
        self.s.chain_in.copy_from(&out, h * 2, &self.stream).map_err(|e| ModelError::Config(format!("chain copy: {e}")))?;
        Ok(())
    }

    /// The `L72` block over `rows` rows whose ids/positions are already in the scratch (`hidden` holds the trunk hidden rows,
    /// `e_next` the token embeddings): `h_mtp = eh_proj(cat(enorm(e), hnorm(h)))`, then the decoder block.
    fn mtp_layer_forward(&mut self, rows: usize) -> Result<(), ModelError> {
        let mtp = self.dev.mtp.as_ref().ok_or_else(|| ModelError::Config("no MTP layer".into()))?;
        let h = self.cfg.hidden;
        let st = &self.stream;
        unsafe {
            glue::rmsnorm(self.s.e_next.ptr(), mtp.enorm.ptr(), self.s.norm.ptr(), rows, h, RMS_EPS, st)?;
            glue::rmsnorm(self.s.hidden.ptr(), mtp.hnorm.ptr(), self.s.final_normed.ptr(), rows, h, RMS_EPS, st)?;
        }
        // y = cat(enorm(e_next), hnorm(h_trunk)) @ eh_proj^T in one fused bf16 kernel (no f16 staging, no round-trips)
        let mut y = self.s.hidden.subview(0, rows * h * 2);
        mla_gated::eh_proj_cat_bf16(&self.s.norm, &self.s.final_normed, &mtp.eh_proj, &mut y, rows, h, st)
            .map_err(|e| ModelError::Config(format!("eh_proj: {e}")))?;
        // the multi-row absorb (prefill-class) runs the MoE pass `fmoe::prefill`; single-row chain steps stay on `forward`
        self.layer_block_forward(true, rows > 1, rows)?;
        // BailingMoeV3MTPLayer.forward ends with `hidden_states = self.final_layernorm(hidden_states)`
        // (modeling_bailing_moe_v3.py: the absorb/chain outputs are final-normed, the head runs lm_head on them
        // directly, and the next chain step feeds the normed output through hnorm again). Apply it in place.
        let mtp = self.dev.mtp.as_ref().ok_or_else(|| ModelError::Config("no MTP layer".into()))?;
        unsafe {
            glue::rmsnorm(self.s.hidden.ptr(), mtp.final_layernorm.ptr(), self.s.hidden.ptr(), rows, h, RMS_EPS, &self.stream)?;
        }
        Ok(())
    }

    /// One decoder block (mixer + ffn) over the rows already in `hidden` (bf16). `mtp` selects the L72 layer; `prefill`
    /// runs the MoE pass `fmoe::prefill` (up to `PREFILL_ROWS` rows, `moe_pws`) instead of `fmoe::forward` (`moe_ws`).
    fn layer_block_forward(&mut self, mtp: bool, prefill: bool, rows: usize) -> Result<(), ModelError> {
        let layer = if mtp {
            &self.dev.mtp.as_ref().ok_or_else(|| ModelError::Config("no MTP layer".into()))?.layer
        } else {
            &self.dev.layers[0]
        };
        let st = &self.stream;
        let cfg = self.cfg.clone();
        let h = cfg.hidden;
        let eps = cfg.rms_eps as f32;
        unsafe {
            glue::rmsnorm(self.s.hidden.ptr(), layer.input_norm.ptr(), self.s.norm.ptr(), rows, h, eps, st)?;
        }
        match &layer.mixer {
            MixerDevice::Mla(ml) => {
                let ord = MTP_MLA_ORD;
                if prefill {
                    ml.q_a
                        .forward_prefill_pinned(
                            &self.s.norm,
                            &mut self.s.q_a,
                            rows,
                            Dtype::BF16,
                            Dtype::BF16,
                            &mut self.s.prefill_ws,
                            st,
                        )
                        .map_err(|e| ModelError::Config(format!("L72 q_a prefill m={rows}: {e}")))?;
                } else {
                    ml.q_a.forward(&self.s.norm, &mut self.s.q_a, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                }
                unsafe {
                    glue::rmsnorm(self.s.q_a.ptr(), ml.q_a_norm.ptr(), self.s.q_a.ptr(), rows, cfg.mla.q_lora_rank, eps, st)?
                };
                if prefill {
                    ml.q_b
                        .forward_prefill_pinned(
                            &self.s.q_a,
                            &mut self.s.q,
                            rows,
                            Dtype::BF16,
                            Dtype::BF16,
                            &mut self.s.prefill_ws,
                            st,
                        )
                        .map_err(|e| ModelError::Config(format!("L72 q_b prefill m={rows}: {e}")))?;
                } else {
                    ml.q_b.forward(&self.s.q_a, &mut self.s.q, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                }
                if prefill {
                    ml.kv_a
                        .forward_prefill_pinned(
                            &self.s.norm,
                            &mut self.s.kv,
                            rows,
                            Dtype::BF16,
                            Dtype::BF16,
                            &mut self.s.prefill_ws,
                            st,
                        )
                        .map_err(|e| ModelError::Config(format!("L72 kv_a prefill m={rows}: {e}")))?;
                } else {
                    ml.kv_a.forward(&self.s.norm, &mut self.s.kv, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                }
                let ri = self.row_index(rows);
                mla_gated::mla_gated_rows(
                    &self.s.norm,
                    &self.s.q,
                    &self.s.kv,
                    &ml.kv_a_norm,
                    &self.cos,
                    &self.sin,
                    &ml.w,
                    &mut self.mla_caches[ord],
                    &self.table,
                    &ri,
                    rows,
                    &mut self.s.q_lat,
                    &mut self.s.q_rot,
                    &mut self.s.o_lat,
                    &mut self.s.mla_out,
                    &mut self.s.gated,
                    st,
                )?;
                if prefill {
                    ml.dense.forward_prefill_pinned(
                        &self.s.gated,
                        &mut self.s.a,
                        rows,
                        Dtype::BF16,
                        Dtype::BF16,
                        &mut self.s.prefill_ws,
                        st,
                    )?;
                } else {
                    ml.dense.forward(&self.s.gated, &mut self.s.a, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                }
                unsafe {
                    glue::add_rmsnorm(
                        self.s.hidden.ptr().cast_const(),
                        self.s.a.ptr().cast_const(),
                        layer.post_norm.ptr().cast_const(),
                        self.s.hidden.ptr(),
                        self.s.norm.ptr(),
                        rows,
                        h,
                        eps,
                        st,
                    )?;
                }
            }
            MixerDevice::Kda(_) => return Err(ModelError::Config("L72 cannot be KDA".into())),
        }
        match &layer.ffn {
            FfnDevice::Moe(moe) => {
                let router = &moe.router;
                self.s
                    .moe_logits_bf16(&self.stream, router, h, cfg.ffn.routed_experts, rows)
                    .map_err(|e| ModelError::Config(format!("moe router L72 m={rows}: {e}")))?;
                let shape = fmoe::Shape {
                    t: rows,
                    ne: cfg.ffn.routed_experts,
                    top_k: cfg.ffn.experts_per_token,
                    h,
                    i: cfg.ffn.moe_intermediate,
                    kc: BAILING_KC,
                    max_tokens: if prefill { PREFILL_ROWS } else { ROWS },
                };
                let gr = Some(fmoe::GroupRoute {
                    expert_bias: Some(&moe.expert_bias),
                    n_group: cfg.ffn.n_group,
                    topk_group: cfg.ffn.topk_group,
                    scale: cfg.ffn.routed_scaling_factor as f32,
                });
                let limits = fmoe::SwigluLimit { routed: cfg.routed_swiglu_limit(71), shared: cfg.shared_swiglu_limit(71) };
                if prefill {
                    fmoe::prefill(
                        &self.s.norm,
                        &self.s.moe_logits,
                        &self.s.moe_shared_gate,
                        &moe.tables(),
                        &mut self.s.moe_out,
                        shape,
                        gr,
                        limits,
                        &mut self.s.moe_pws,
                        st,
                    )?;
                } else {
                    fmoe::forward(
                        &self.s.norm,
                        &self.s.moe_logits,
                        &self.s.moe_shared_gate,
                        &moe.tables(),
                        &mut self.s.moe_out,
                        shape,
                        gr,
                        limits,
                        &mut self.s.moe_ws,
                        st,
                    )?;
                }
                // the FFN residual is a plain add: the next op is `final_layernorm` in place on `hidden` (`mtp_layer_forward`),
                // which cannot fuse with an add (its input and output are the same buffer)
                unsafe { glue::add(self.s.hidden.ptr(), self.s.moe_out.ptr(), self.s.hidden.ptr(), rows * h, st)? };
            }
            FfnDevice::Dense(_) => return Err(ModelError::Config("L72 must be MoE".into())),
        }
        Ok(())
    }
}
