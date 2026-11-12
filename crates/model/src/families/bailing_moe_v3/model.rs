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
//! rows (absorb), then chains draft tokens through the layer, proposing `argmax` tokens. A prompt longer than the ring leaves the
//! `L72` cache without entries for its first rows (their taps are gone): the drafter's attention is windowed to the entries that exist
//! ([`l72_window`], `MtpSlot::valid_from`) instead of reading what the pages held before.

use super::alloc::{self, WINDOW_ROWS};
use super::config::BailingConfig;
use super::contract::{self, PrefillContract};
use super::device::{BailingDevice, FfnDevice, MixerDevice};
use super::load::EmbeddingSidecar;
use super::spill::{SpillMeta, SpillStore};
use crate::error::ModelError;
use crate::forward::Candidates;
use gb_cuda::attention::{BlockTable, RowIndex};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::glm_glue;
use gb_cuda::graph::{CaptureError, Graph};
use gb_cuda::kda::{self, Params};
use gb_cuda::ling_mla_attn;
use gb_cuda::mla_gated::{self, MlaGatedCache};
use gb_cuda::rt::{CudaError, DevBuf, HostStage, Stream};
use gb_cuda::{fmoe, glue, ling_fmoe};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

/// Rows of the widest decode/verify window.
pub const ROWS: usize = WINDOW_ROWS;
/// The prefill chunk the engine was built for is carried at run time: `BailingModel::prefill_rows` (from the engine's
/// memory options, `alloc::prefill_chunk_for`), with `alloc::DEFAULT_PREFILL_CHUNK` (2,048) the default. 2,048 rows
/// amortize the MoE weight reads (every routed expert tile runs many tokens per pass) and the EXL3 prefill GEMMs (the
/// pinned prefill path is a real batching GEMM, unlike the 8-row decode GEMV slicing). The ONE shared prefill KDA record
/// (instead of one per KDA layer) is what makes the wider slices fit; `fmoe::prefill`, `kda::Record`, `kda::Workspace`,
/// `exl3_linear` and `mla_gated` all take the width, and both prefill and verify paths are row-invariant, so any chunking
/// gives identical bits.
pub const DEFAULT_PREFILL_ROWS: usize = alloc::DEFAULT_PREFILL_CHUNK;
/// Compatibility name for the family's default chunk; a built model may use a smaller aligned `prefill_rows` value.
pub const PREFILL_ROWS: usize = DEFAULT_PREFILL_ROWS;
const _: () = assert!(PREFILL_ROWS == alloc::MAX_PREFILL_CHUNK && PREFILL_ROWS % alloc::PREFILL_ALIGN == 0);
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

/// Which rows of an `L72` absorb go on past the cache write. An absorbed row's KV entry depends only on its own input
/// (`eh_proj`, `input_layernorm`, `kv_a`, the latent norm and RoPE); its Q side, attention, `o_proj`, gate, router and
/// MoE matter only where something reads that row's L72 output, and of an absorb's rows only the last is read (`head_in`,
/// the draft chain's start). Every kernel of those stages is row-invariant (a row's bits do not depend on the other rows
/// of the call, as the whole family's forward is), so running them over the read rows alone changes no bit of the cache
/// or of that row — and skips the full Gated-MLA + 512-expert MoE block over the whole prompt prefix.
/// (The analogous `nemotron_h_omni::mtp::Tail::KvOnly`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tail {
    /// Every row: the chain steps (one row each) and any whole-block forward.
    All,
    /// Only the last row runs the Q/attention/MoE half; every row writes its KV cache entry. The absorbed prompt prefix
    /// of `mtp_propose` / `mtp_absorb_up_to`: up to 8,191 rows of KV-only work instead of a full `L72` per row.
    Last,
}

/// Where an `L72` absorb up to `limit` starts, from the slot's `absorbed` count: `absorbed` itself, or the oldest row the tap ring still holds
/// (`limit - (CAP_TAPS - 1)`) when `absorbed` is older. In that second case the rows between have no entry and never will, and the second
/// value is the slot's new [`MtpSlot::valid_from`]: where the written entries start.
pub fn absorb_start(absorbed: usize, limit: usize) -> (usize, Option<usize>) {
    let absorbed = absorbed.min(limit);
    let first_tapped = limit.saturating_sub(CAP_TAPS - 1);
    if absorbed < first_tapped {
        (first_tapped, Some(first_tapped))
    } else {
        (absorbed, None)
    }
}

/// The window of `L72` cache entries a draft query at `pos` attends, for a slot whose entries start at `valid_from`
/// ([`MtpSlot::valid_from`]): `(first_block, window_pos)`, the first page of the window and the query's position counted from that page's
/// first position. `None` while every position has its entry (`valid_from == 0`): the whole cache, as ever. The window starts at the first
/// page boundary at or after `valid_from` (a page is the unit of the block table), so it holds only written entries and drops at most 255
/// of them; it never starts past the query's own page.
pub fn l72_window(valid_from: usize, pos: usize) -> Option<(usize, i32)> {
    let page = mla_gated::BLOCK_TOKENS;
    let first_block = valid_from.div_ceil(page).min(pos / page);
    (first_block > 0).then(|| (first_block, (pos - first_block * page) as i32))
}

fn env_on(key: &str) -> bool {
    std::env::var(key).map(|v| v != "0").unwrap_or(true)
}

/// `MemAvailable` against the bytes a prefix snapshot needs ([`BailingModel::snapshot_headroom`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotHeadroom {
    pub avail: u64,
    pub needed: u64,
}

pub struct MtpSlot {
    /// How many committed tokens have been fed through `L72` (positions `< absorbed` are in the L72 cache).
    pub absorbed: usize,
    /// The last absorbed row's L72 output (`[hidden]` bf16), the chain's start.
    pub last_out: Option<Vec<u16>>,
    /// The GPU `head_in` register is stale (a slot begin/end or a prefix restore moved the state without re-absorbing).
    pub head_dirty: bool,
    /// The first position whose L72 cache entry was written, with every entry from there to `absorbed` contiguous. The drafter absorbs a
    /// prompt from its tap ring, which keeps the last `CAP_TAPS` rows, so a prompt past that has no entries below `valid_from` (the cache is
    /// not zeroed: those positions hold whatever the page held). Draft attention is windowed to start here ([`l72_window`]). 0 while every
    /// position has its entry.
    pub valid_from: usize,
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
    /// [`MtpSlot::valid_from`] at the cut: the restored pages carry the same unwritten head.
    pub valid_from: usize,
}

/// A full prefix snapshot of one slot at a committed length.
pub struct BailingSnapshot {
    pub len: usize,
    pub kda: Vec<KdaSlotSnap>,
    pub mla: Vec<MlaCacheSnap>,
    pub pages: Vec<u32>,
    pub mtp: Option<MtpSnap>,
}

/// A snapshot stored as one file under the spill store's namespace directory.
pub struct SpillEntry {
    pub len: usize,
    pub n_pages: usize,
    pub file: PathBuf,
}

/// A stored prefix snapshot: host RAM (`BailingSnapshot`) or a spill file.
pub enum StoredSnapshot {
    Ram(BailingSnapshot),
    Spill(SpillEntry),
}

/// Host-RAM snapshot slots (one per `--prefix-cache` entry).
pub struct SnapPool {
    pub slots: Vec<Option<StoredSnapshot>>,
    pub free: Vec<u32>,
}

/// Everything a forward scratches, sized for `prefill_rows` rows (a prefill chunk, the widest call; the head, the top-k and
/// the EXL3 decode workspace stay window-sized because only decode/verify windows and single rows run them).
pub struct Scratch {
    /// Rows the buffer set was sized for (a prefill chunk).
    pub rows: usize,
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
    /// KDA qkv (bf16 `[rows, 3 * channels]`): decode writes it f32, prefill writes bf16 directly (the pinned EXL3 prefill GEMM
    /// emits bf16 and `kda::prep_prefill_bf16` reads it in place, skipping the all-rows raw copy into the record). The buffer is
    /// sized for the wider f32 decode layout so both fit.
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
    /// MoE routing (logits **fp32** `[rows, experts]` — Ling's `router_dtype = "fp32"`; shared gate bf16 `[rows]`,
    /// window/prefill fmoe workspaces, out bf16 `[rows, hidden]`).
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
    /// The next sub-layer's rotated inputs (f16 `[rows, hidden]` each, up to three) fused into the linear
    /// epilogues (`ling_rot_out_add_rmsnorm_k`): the fused `rot_in_{pair,triple}` of the dense FFN gate/up pair
    /// and of the next layer's qkv/f/g (or q_a/kv_a) triple/pair. Separate from `ws` so the epilogue's own
    /// split-partial reads never race the writes.
    pub prerot_ws: DevBuf,
    /// The direct-to-MMA prefill workspace a prefill chunk needs (LP3): the rotated fp16 input of the widest
    /// linear (`glm_p2::workspace_bytes(rows, k)`), no `W_q` reconstruction buffer.
    pub prefill_ws: DevBuf,
    /// The split-key decode attention's fp32 partials (`ling_mla_attn::SPLIT_WS_BYTES`, fixed size): every decode / verify / draft
    /// Gated-MLA call of a step reuses it in turn.
    pub attn_ws: DevBuf,
    /// MTP: the token embedding rows (bf16 `[rows, h]`).
    pub e_next: DevBuf,
    /// The MTP head's input row (bf16 `[1, hidden]`): the L72 output row the head runs on, kept on the GPU across proposes.
    pub head_in: DevBuf,
    /// The MTP chain's working register (bf16 `[1, hidden]`): the chain's previous L72 output, kept on the GPU.
    pub chain_in: DevBuf,
}

impl Scratch {
    pub(crate) fn new(cfg: &BailingConfig, prefill_rows: usize) -> Result<Self, ModelError> {
        // The scratch is sized for the widest call, a prefill chunk; the FrostMoE window workspace is sized for the
        // widest decode/verify window (`fmoe::forward` caps `T` at 64 rows, `fmoe::prefill` does not).
        let r = prefill_rows.max(1);
        let rw = ROWS;
        let h = cfg.hidden;
        let ch = cfg.kda.channels();
        let m = &cfg.mla;
        let ffn = &cfg.ffn;
        let e = ffn.routed_experts;
        let topk = ffn.experts_per_token;
        let zeroed = |bytes: usize| DevBuf::zeroed_view(bytes).map_err(|e| ModelError::Config(format!("scratch alloc: {e}")));
        Ok(Self {
            rows: r,
            ids: zeroed(r * 4)?,
            row_slot: zeroed(r * 4)?,
            row_pos: zeroed(r * 4)?,
            seq_start: zeroed((r + 1) * 4)?,
            seq_slot: zeroed(r * 4)?,
            keep: zeroed(r * 4)?,
            hidden: zeroed(r * h * 2)?,
            norm: zeroed(r * h * 2)?,
            qkv: zeroed((r * 3 * ch * 2).max(r * 3 * ch * 4))?,
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
            moe_logits: zeroed(r * e * 4)?,
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
            topk_ids: zeroed(ROWS * 64 * 4)?,
            topk_vals: zeroed(ROWS * 64 * 4)?,
            topk_ws: zeroed(gb_cuda::ling_kda_proj_fused::topk_workspace_bytes(ROWS, cfg.vocab, 64))?,
            ws: zeroed(exl3_ws_bytes(cfg, ROWS))?,
            prerot_ws: zeroed(3 * ROWS * h * 2 + 1024)?,
            prefill_ws: zeroed(prefill_ws_bytes(cfg, r))?,
            attn_ws: zeroed(ling_mla_attn::SPLIT_WS_BYTES)?,
            e_next: zeroed(r * h * 2)?,
            head_in: zeroed(h * 2)?,
            chain_in: zeroed(h * 2)?,
        })
    }
    /// The device bytes this buffer set actually holds (the `DevBuf` extents; every row-extent buffer is
    /// `self.rows` rows, the head/top-k/decode-workspace ones are window-sized). `alloc::AllocPlan` prices the same items
    /// for the memory plan; the unit test `scratch_bytes_equal_the_plan` checks the two agree.
    pub fn allocated_bytes(&self) -> u64 {
        let b = [
            &self.ids,
            &self.row_slot,
            &self.row_pos,
            &self.seq_start,
            &self.seq_slot,
            &self.keep,
            &self.hidden,
            &self.norm,
            &self.qkv,
            &self.f_out,
            &self.g_out,
            &self.q_a,
            &self.q,
            &self.kv,
            &self.q_lat,
            &self.q_rot,
            &self.o_lat,
            &self.mla_out,
            &self.gated,
            &self.g,
            &self.u,
            &self.a,
            &self.moe_logits,
            &self.moe_shared_gate,
            &self.moe_ws,
            &self.moe_pws,
            &self.moe_out,
            &self.final_normed,
            &self.logits,
            &self.topk_ids,
            &self.topk_vals,
            &self.topk_ws,
            &self.ws,
            &self.prerot_ws,
            &self.prefill_ws,
            &self.attn_ws,
            &self.e_next,
            &self.head_in,
            &self.chain_in,
        ];
        b.iter().map(|d| d.bytes() as u64).sum()
    }
}

/// The EXL3 decode workspace a forward needs: the largest of every linear the forward runs. The shapes live in
/// `alloc::AllocPlan::exl3_ws_bytes` so the plan and the runtime agree by construction.
pub fn exl3_ws_bytes(cfg: &BailingConfig, rows: usize) -> usize {
    alloc::AllocPlan::exl3_ws_bytes(cfg, rows)
}

/// The direct-to-MMA prefill workspace a prefill chunk needs (LP3): the rotated fp16 input of the widest linear
/// the prefill forward runs (`glm_p2::workspace_bytes` = `rows * k * 2`), NOT the old pinned path's
/// `align(m*k*2) + align(k*n*2) + align(m*n*4)`: the trellis is decoded straight into the MMA's B fragments and no
/// fp16 `W_q` or fp32 `c` is materialized. The head stays on the decode kernels (`exl3_ws_bytes`).
pub fn prefill_ws_bytes(cfg: &BailingConfig, rows: usize) -> usize {
    let h = cfg.hidden;
    let ch = cfg.kda.channels();
    let m = &cfg.mla;
    let ffn = &cfg.ffn;
    let ks: Vec<usize> = vec![h, ch, m.q_lora_rank, m.heads * m.v_head_dim, ffn.dense_intermediate];
    ks.into_iter().map(|k| gb_cuda::glm_p2::workspace_bytes(rows, k)).max().unwrap_or(0)
}

/// Device bytes of the KDA records and workspace the runtime allocates: one small record per KDA layer for decode/verify
/// windows, ONE shared prefill record and ONE shared workspace, both `prefill_rows` rows. Same items as
/// `alloc::AllocPlan::kda_items`.
pub fn kda_alloc_bytes(cfg: &BailingConfig, prefill_rows: usize) -> u64 {
    alloc::AllocPlan::kda_items(cfg, prefill_rows).iter().map(|i| i.bytes).sum()
}

/// `dst[..v.len()] = v` in the order of `st` (every per-step input of the model goes through here): a kernel launched on `st`
/// after this call reads the new bytes, and one launched before it (still queued, or still running) has read the old ones. The
/// bytes are copied into `stage`, which keeps them until the model's next [`BailingModel::sync`], so `v` may be a temporary.
/// `DevBuf::upload` would not do: it is a pageable `cudaMemcpy` on the legacy default stream, which the model's non-blocking
/// stream does not wait for, and it may return before its DMA lands.
fn put<T: Copy>(stage: &mut HostStage, dst: &mut DevBuf, v: &[T], st: &Stream, what: &str) -> Result<(), ModelError> {
    stage.upload(dst, v, st).map_err(|e: CudaError| ModelError::Config(format!("{what} upload: {e}")))
}

/// Rows `ids` of the embedding table into the start of `dst` (bf16 `[ids.len(), hidden]`), in the order of `st`, for a table the GPU
/// reads in place from its shard (`load::embed_in_place`): the rows are gathered on the host, from the shard's page cache, into the
/// scratch `host_rows`, and `stage` queues one `cudaMemcpyAsync` of them. The GPU never touches the mapping: a first read of a
/// file-backed page by a kernel is a GPU page fault, whatever the kernel: 0.1 to 0.15 ms per row measured on a warm page cache (a row
/// is 8 KiB, 2 or 3 pages), 10 us once the page is mapped, more for a page that must come off the disk, so 170 ms for a 2048-row
/// prefill chunk and 2 to 4 ms for the one new token of a decode round. The CPU maps the same page in under 1 us (2048 rows: 1 to 3 ms).
/// Every output byte is the table's byte (an id outside `0..vocab` gives a zero row, as the `gb_embed` kernel does).
fn embed_rows_in_place(
    stage: &mut HostStage,
    host_rows: &mut Vec<u8>,
    table: &DevBuf,
    vocab: usize,
    hidden: usize,
    ids: &[i32],
    dst: &mut DevBuf,
    st: &Stream,
) -> Result<(), ModelError> {
    let block = table.block().ok_or_else(|| ModelError::Config("the embedding table is not a view of a mapping".into()))?;
    let base = block.host_ptr_at(0).cast_const();
    host_rows.resize(ids.len() * hidden * 2, 0);
    // SAFETY: `base` is the start of the read-only mapping of `vocab * hidden` bf16 values that `table` views, alive as long as `table`
    unsafe { super::load::gather_rows(base, vocab, hidden * 2, ids, host_rows) };
    stage.upload(dst, host_rows, st).map_err(|e: CudaError| ModelError::Config(format!("embedding rows upload: {e}")))
}

impl Scratch {
    /// Router logits `[rows, E]` **fp32** = `x @ router^T` (router bf16 `[E, H]`; Ling-3.1's `router_dtype = "fp32"`: the logit is NOT
    /// rounded through bf16). `tensor_core` (P2 prompt rows) runs `mla_gated::router_rows_f32`; every other call (P2 decode / verify
    /// windows, and every row under P1) runs the scalar `bf16_linear_f32` kernel. Both are fixed-order reductions per (row, column) at
    /// every row count, so a row's logits do not depend on the call's row count, `cached == cold`, and the logits are chunk-size
    /// invariant. Never cuBLAS: its algorithm depends on M (rows > 4 used to go there, and a 256-row chunk, a 73-row tail slice and a
    /// 3,896-row call gave different low bits for the same row).
    fn moe_logits_f32(
        &mut self,
        stream: &Stream,
        router: &DevBuf,
        h: usize,
        e: usize,
        rows: usize,
        tensor_core: bool,
    ) -> Result<(), ModelError> {
        let mut y = self.moe_logits.subview(0, rows * e * 4);
        if tensor_core {
            mla_gated::router_rows_f32(&self.norm, router, &mut y, rows, h, e, stream)
        } else {
            // the single-wave multi-row GEMV (bit-identical per (row, column) to the scalar kernel; ling_kda_proj_fused)
            gb_cuda::ling_kda_proj_fused::bf16_linear_f32(&self.norm, router, &mut y, rows, h, e, stream)
        }
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
    /// Stream-ordered host->device staging (the batch arrays, the block table): no legacy-stream pageable `memcpy` races,
    /// no sync between an upload and the kernel that reads it (`HostStage`; see #192).
    pub stage: HostStage,
    pub slots: usize,
    pub max_context: usize,
    /// Rows of one prefill chunk: the widest trunk call. All prefill scratch, the shared prefill KDA record and the KDA
    /// workspace are sized for it (`alloc::prefill_chunk_for`, default `DEFAULT_PREFILL_ROWS`).
    pub prefill_rows: usize,
    /// KDA layer `l` -> ordinal among KDA layers.
    pub kda_ord: Vec<usize>,
    pub pools: Vec<kda::Pool>,
    /// One small record per KDA layer (`alloc::KDA_WINDOW_RECORD_ROWS` rows): decode/verify window replay/commit.
    pub recs: Vec<kda::Record>,
    /// ONE shared prefill record (`prefill_rows` rows): a prefill chunk commits each KDA layer before the next runs, so
    /// the layers do not need concurrent prefill-sized records (63 x 512 rows -> 63 window records + 1 prefill record).
    pub prefill_kda_rec: kda::Record,
    pub kda_ws: kda::Workspace,
    /// Per-`(seq, head)` window+norm last-block counters (`n_seq * heads` i32s, zeroed once; the fused kernel
    /// self-resets them on every call).
    pub kda_head_cnt: DevBuf,
    /// The batched commit-tail layer table for `kda::commit_tail_all` (`n_kda * 24` bytes: the raw record rows,
    /// conv pool and parity pointers of every KDA layer, fixed for the model's lifetime).
    pub kda_commit_layers: DevBuf,
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
    /// The bf16 embedding table `[vocab, hidden]`: the shard's own bytes in place where the GPU reads pageable memory
    /// (`load::embed_in_place`), whose rows the host gathers (`embed_rows_in_place`), a device copy otherwise, whose rows `glue::embed` gathers.
    pub embed_dev: DevBuf,
    /// `embed_dev` is the shard's bytes in place (host memory), not a device copy.
    embed_in_place: bool,
    /// Host scratch the rows of a step are gathered into before their copy (up to `prefill_rows * hidden * 2` bytes, 16 MiB at the default
    /// chunk; empty until the first step).
    embed_host: Vec<u8>,
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
    /// The NVMe spill store (`None`: the host-RAM cache), when a writable cache directory exists at boot.
    pub spill: Option<SpillStore>,
    /// The Gated-MLA arithmetic contract of this model (`contract.rs`): the process's at construction. P1
    /// runs the scalar `mla_gated` kernels; P2 the tensor-core `ling_mla_attn` kernels for prefill, verify,
    /// decode and L72 alike (the arithmetic is a property of the model, not of the call type).
    pub mla_contract: PrefillContract,
}

impl BailingModel {
    pub fn new(
        cfg: BailingConfig,
        dev: BailingDevice,
        embedding: EmbeddingSidecar,
        slots: usize,
        kv_tokens: usize,
        with_mtp: bool,
        prefill_rows: usize,
    ) -> Result<Self, ModelError> {
        let prefill_rows = alloc::prefill_chunk_for(prefill_rows);
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
            recs.push(
                kda::Record::alloc(dims, alloc::KDA_WINDOW_RECORD_ROWS)
                    .map_err(|e| ModelError::Config(format!("kda record: {e}")))?,
            );
        }
        // one shared prefill record + one shared workspace, both prefill-chunk sized
        let prefill_kda_rec =
            kda::Record::alloc(dims, prefill_rows).map_err(|e| ModelError::Config(format!("kda prefill record: {e}")))?;
        let kda_ws = kda::Workspace::alloc(dims, prefill_rows).map_err(|e| ModelError::Config(format!("kda workspace: {e}")))?;
        let kda_head_cnt =
            DevBuf::zeroed(slots.max(1) * dims.heads * 4).map_err(|e| ModelError::Config(format!("kda head cnt: {e}")))?;
        // the batched commit-tail layer table (raw record rows / conv pool / parity pointers of every KDA
        // layer): built once here; the device addresses are stable for the model's lifetime
        let mut commit_bytes = Vec::with_capacity(kda_layers.len() * 24);
        for ord in 0..kda_layers.len() {
            commit_bytes.extend_from_slice(&(recs[ord].qkv.ptr() as usize as u64).to_le_bytes());
            commit_bytes.extend_from_slice(&(pools[ord].conv.ptr() as usize as u64).to_le_bytes());
            commit_bytes.extend_from_slice(&(pools[ord].parity.ptr() as usize as u64).to_le_bytes());
        }
        let kda_commit_layers =
            DevBuf::from_slice(&commit_bytes).map_err(|e| ModelError::Config(format!("kda commit layers: {e}")))?;
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
        let s = Scratch::new(&cfg, prefill_rows)?;
        // the bf16 embedding table (1.20 GiB for Ling-3.1): where the GPU reads pageable memory (the GB10), the shard's own bytes in
        // place, so it takes no device memory and its pages stay reclaimable page cache, and the host gathers the rows a step needs
        // (`embed_rows_in_place`); elsewhere a device copy of the same bytes, gathered by `glue::embed`. Either way a row is the same
        // bytes, so every output is the same.
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
                slots: (0..slots).map(|_| MtpSlot { absorbed: 0, last_out: None, head_dirty: true, valid_from: 0 }).collect(),
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
            stage: HostStage::new(),
            slots,
            max_context,
            prefill_rows,
            kda_ord,
            pools,
            recs,
            prefill_kda_rec,
            kda_ws,
            kda_head_cnt,
            kda_commit_layers,
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
            embed_dev,
            embed_in_place,
            embed_host: Vec::new(),
            dev_taps,
            graphs: HashMap::new(),
            seen_shapes: HashSet::new(),
            use_graphs,
            snaps: None,
            spill: None,
            mla_contract: contract::current(),
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
            ("kda_prefill_record", rec(&self.prefill_kda_rec)),
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
            ("prerot_workspace", b(&s.prerot_ws)),
            ("exl3_prefill_workspace", b(&s.prefill_ws)),
            ("mla_split_workspace", b(&s.attn_ws)),
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
        self.prefill_rows
    }

    // ------------------------------------------------------------------------------------------------ page management

    fn ensure_pages(&mut self, slot: usize, max_pos: usize) -> Result<(), ModelError> {
        let need = (max_pos + 1).div_ceil(256);
        let pages = &mut self.slot_pages[slot];
        let before = pages.len();
        while pages.len() < need {
            let b = self.free_blocks.pop().ok_or_else(|| ModelError::Request("no free KV page".into()))?;
            pages.push(b);
        }
        // the page table is uploaded only when this slot's page count grew: a decode step that needs no new page does not
        // re-upload the table (one less device write per step; the table is already current from the last growth)
        if pages.len() > before {
            self.table
                .upload_on(&self.slot_pages, &mut self.stage, &self.stream)
                .map_err(|e| ModelError::Config(format!("table upload: {e}")))?;
        }
        Ok(())
    }

    fn release_pages(&mut self, slot: usize) {
        let pages = std::mem::take(&mut self.slot_pages[slot]);
        for b in pages {
            self.free_blocks.push(b);
        }
    }

    /// Drop the scratch's batch arrays after a slot begin / prefix restore, so the next window (a full prefix hit
    /// resumes with a one-row window step) cannot replay a graph whose KDA batch is the previous call's: a graph key is
    /// only `(rows, n_seq, count)`, and the replay reads these arrays, not the key. One 0-byte write each on the stream,
    /// which also orders the invalidation before the next staged uploads.
    fn invalidate_batch(&mut self) {
        for b in [&mut self.s.ids, &mut self.s.row_slot, &mut self.s.row_pos, &mut self.s.seq_start, &mut self.s.seq_slot] {
            let _ = b.memset_async(0, &self.stream);
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

    /// Switch the snapshot store to the NVMe spill store (`Some`): captures write files, restores read them back with
    /// checksum verification; `None` keeps the host-RAM cache.
    pub fn set_spill_store(&mut self, spill: Option<SpillStore>) {
        self.spill = spill;
    }

    /// True when the host has enough headroom above a fixed floor to hold a snapshot of `len` tokens in RAM
    /// (the 63 KDA slot states dominate: ~8 MiB per layer, constant). A decline is not an error (a skipped capture is a cold
    /// prefill) but it is counted by the caller (`counters::note_skipped_low_memory`, `rexl3_ling31_snapshot_skipped_total`):
    /// see [`Self::snapshot_headroom`].
    pub fn can_snapshot_prefix(&self, len: usize) -> bool {
        if len == 0 || len > self.max_context || self.snaps.as_ref().is_none_or(|s| s.free.is_empty()) {
            return false;
        }
        if self.spill.is_some() {
            // the spill store costs only its staging buffer: a free slot is the whole guard
            return true;
        }
        self.snapshot_headroom(len).is_none_or(|h| h.avail >= h.needed)
    }

    /// What the guard of [`Self::can_snapshot_prefix`] compares for a snapshot of `len` tokens: `MemAvailable` now and the bytes the
    /// snapshot needs (its KDA state, its MLA copies and 1 GiB). `None` where the host reports no `MemAvailable` (the guard then
    /// passes).
    pub fn snapshot_headroom(&self, len: usize) -> Option<SnapshotHeadroom> {
        let kda_bytes: u64 = self.pools.len() as u64 * self.pools.first().map_or(0, |p| p.dims.slot_bytes()) as u64;
        let mla_bytes: u64 = self.mla_caches.len() as u64 * (len as u64) * (mla_gated::CACHE_TOKEN_ELEMS as u64) * 2;
        let needed = kda_bytes + mla_bytes + (1u64 << 30);
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                if let Some(avail_kib) = meminfo
                    .lines()
                    .find(|l| l.starts_with("MemAvailable:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<u64>().ok())
                {
                    return Some(SnapshotHeadroom { avail: avail_kib * 1024, needed });
                }
            }
        }
        let _ = needed;
        None
    }

    pub fn alloc_snapshot(&mut self) -> Option<u32> {
        self.snaps.as_mut()?.free.pop()
    }

    pub fn release_prefix(&mut self, snap: u32) {
        if let Some(p) = self.snaps.as_mut() {
            if let Some(sl) = p.slots.get_mut(snap as usize) {
                if let Some(StoredSnapshot::Spill(e)) = sl.take() {
                    if let Some(st) = self.spill.as_mut() {
                        let _ = st.delete(&e.file);
                    }
                }
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
        if self.spill.is_some() {
            let mut store = self.spill.take().unwrap();
            let r = self.prefix_snapshot_spill(&mut store, slot, snap, len);
            if r.is_err() {
                // Downloads and device snapshots can fail between the store's write calls; do not leave a partial
                // writer behind to reject every later capture.
                store.abort_write();
            }
            self.spill = Some(store);
            return r;
        }
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
            Some(MtpSnap {
                taps,
                absorbed: m.slots[slot].absorbed,
                last_out: m.slots[slot].last_out.clone(),
                valid_from: m.slots[slot].valid_from,
            })
        } else {
            None
        };
        if let Some(pool) = self.snaps.as_mut() {
            pool.slots[snap as usize] = Some(StoredSnapshot::Ram(BailingSnapshot { len, kda, mla, pages, mtp }));
        }
        Ok(())
    }

    /// The spill-write half of [`Self::prefix_snapshot`]: stream the snapshot to one file under the store's namespace
    /// directory (MLA pages, then each KDA layer's state, then the MTP tap ring and metadata), verifying nothing but the
    /// store's own block checksums on the way. The snapshot slot holds a [`SpillEntry`] handle.
    fn prefix_snapshot_spill(&mut self, store: &mut SpillStore, slot: usize, snap: u32, len: usize) -> Result<(), ModelError> {
        let pages = self.slot_pages[slot].clone();
        let n_pages = pages.len() as u64;
        let n_kda = self.pools.len() as u64;
        let mtp = self.mtp.is_some();
        let n_mla = self.mla_caches.len() as u64;
        let bb = if self.mla_caches.is_empty() { 0 } else { MlaGatedCache::block_bytes(Dtype::BF16) as u64 };
        // an upper bound of the payload (the header is rewritten with the exact byte count and checksums at finish): pages,
        // per KDA layer s/parity/conv, and the MTP tap ring + metadata
        let mut payload_len = n_pages * n_mla * bb;
        for p in &self.pools {
            let e = p.dims.s_buf_elems() as u64;
            let conv = p.dims.conv_slot_elems() as u64;
            payload_len += 8 + 2 * e * 4 + 4 + 8 + conv * 2;
        }
        if mtp {
            payload_len += 8 + (CAP_TAPS as u64) * (self.cfg.hidden as u64) * 2 + 8 + 8 + 1 + 8 + (self.cfg.hidden as u64) * 2;
        }
        let meta = SpillMeta {
            namespace: 0,
            len_tokens: len as u64,
            n_pages,
            n_kda_layers: n_kda,
            mtp,
            n_mla_caches: n_mla,
            page_block_bytes: bb,
        };
        store
            .start_write(snap as u64, meta, payload_len)
            .map_err(|e| ModelError::Config(format!("spill snapshot begin: {e}")))?;
        // MLA pages, per cache, page by page
        let mut page = Vec::new();
        for c in &self.mla_caches {
            let cbb = MlaGatedCache::block_bytes(c.dtype) as usize;
            if page.len() < cbb {
                page.resize(cbb, 0);
            }
            for &p in &pages {
                c.buf
                    .download_range((p as usize) * cbb, &mut page[..cbb])
                    .map_err(|e| ModelError::Config(format!("mla spill download: {e}")))?;
                store.push_bytes(&page[..cbb]).map_err(|e| ModelError::Config(format!("mla spill write: {e}")))?;
            }
        }
        // KDA states, one layer at a time (peak host state = one layer, not the whole 63)
        for p in &self.pools {
            let (s, parity, conv) = p.slot_snapshot(slot).map_err(|e| ModelError::Config(format!("kda snapshot: {e}")))?;
            let mut blob = Vec::with_capacity(8 + s.len() * 4 + 4 + 8 + conv.len() * 2);
            blob.extend_from_slice(&(s.len() as u64).to_le_bytes());
            for v in &s {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            blob.extend_from_slice(&parity.to_le_bytes());
            blob.extend_from_slice(&(conv.len() as u64).to_le_bytes());
            for v in &conv {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            store.push_bytes(&blob).map_err(|e| ModelError::Config(format!("kda spill write: {e}")))?;
        }
        // MTP tap ring and metadata
        if let Some(m) = &self.mtp {
            let mut taps = vec![0u16; CAP_TAPS * self.cfg.hidden];
            self.dev_taps[slot].download(&mut taps).map_err(|e| ModelError::Config(format!("taps download: {e}")))?;
            let mut blob = Vec::with_capacity(8 + taps.len() * 2 + 25);
            blob.extend_from_slice(&(taps.len() as u64).to_le_bytes());
            for v in &taps {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            let slot_m = &m.slots[slot];
            blob.extend_from_slice(&(slot_m.absorbed as u64).to_le_bytes());
            blob.extend_from_slice(&(slot_m.valid_from as u64).to_le_bytes());
            match &slot_m.last_out {
                Some(last) => {
                    blob.push(1);
                    blob.extend_from_slice(&(last.len() as u64).to_le_bytes());
                    for v in last {
                        blob.extend_from_slice(&v.to_le_bytes());
                    }
                }
                None => {
                    blob.push(0);
                    blob.extend_from_slice(&0u64.to_le_bytes());
                }
            }
            store.push_bytes(&blob).map_err(|e| ModelError::Config(format!("taps spill write: {e}")))?;
        }
        store.finish_write().map_err(|e| ModelError::Config(format!("spill snapshot finish: {e}")))?;
        let file = store.last_path().to_path_buf();
        if let Some(pool) = self.snaps.as_mut() {
            pool.slots[snap as usize] = Some(StoredSnapshot::Spill(SpillEntry { len, n_pages: n_pages as usize, file }));
        }
        Ok(())
    }

    /// The spill-read half of [`Self::restore_prefix`]: stream the file back, verifying every block checksum; any missing or
    /// corrupted file is an error the engine turns into a cache miss (cold prefill), never silent corruption.
    fn restore_prefix_spill(&mut self, e: &SpillEntry, slot: usize, len: usize) -> Result<(), ModelError> {
        let Some(mut store) = self.spill.take() else {
            return Err(ModelError::Request("spill store gone".into()));
        };
        let result = (|| -> Result<(), ModelError> {
            let mtp = self.mtp.is_some();
            let n_mla = self.mla_caches.len() as u64;
            let bb = if self.mla_caches.is_empty() { 0 } else { MlaGatedCache::block_bytes(Dtype::BF16) as u64 };
            let expect = SpillMeta {
                namespace: 0,
                len_tokens: len as u64,
                n_pages: e.n_pages as u64,
                n_kda_layers: self.pools.len() as u64,
                mtp,
                n_mla_caches: n_mla,
                page_block_bytes: bb,
            };
            store.begin_read(&e.file, expect).map_err(|err| ModelError::Request(format!("spill restore open: {err}")))?;
            // fresh pages for `len`, then copy the stored contents in logical order (the same contents the RAM restore copies)
            self.ensure_pages(slot, len.saturating_sub(1))?;
            let np = self.slot_pages[slot].len();
            for c in &self.mla_caches {
                let cbb = MlaGatedCache::block_bytes(c.dtype) as usize;
                let mut page = vec![0u8; cbb];
                for i in 0..np {
                    store.read_exact_into(&mut page).map_err(|err| ModelError::Request(format!("spill restore pages: {err}")))?;
                    let p = self.slot_pages[slot][i] as usize;
                    unsafe {
                        self.stream
                            .memcpy_h2d_async(c.buf.ptr_at(p * cbb), page.as_ptr(), cbb)
                            .map_err(|e| ModelError::Config(format!("mla spill restore: {e}")))?;
                    }
                }
            }
            // KDA states, in the pools' order
            for ord in 0..self.pools.len() {
                let mut len_buf = [0u8; 8];
                store
                    .read_exact_into(&mut len_buf)
                    .map_err(|err| ModelError::Request(format!("spill restore kda len: {err}")))?;
                let slen = u64::from_le_bytes(len_buf) as usize;
                let mut raw = vec![0u8; slen * 4];
                store.read_exact_into(&mut raw).map_err(|err| ModelError::Request(format!("spill restore kda s: {err}")))?;
                let s: Vec<f32> = raw.chunks_exact(4).map(|ch| f32::from_le_bytes(ch.try_into().unwrap())).collect();
                let mut parity_buf = [0u8; 4];
                store
                    .read_exact_into(&mut parity_buf)
                    .map_err(|err| ModelError::Request(format!("spill restore kda parity: {err}")))?;
                let parity = i32::from_le_bytes(parity_buf);
                let mut clen_buf = [0u8; 8];
                store
                    .read_exact_into(&mut clen_buf)
                    .map_err(|err| ModelError::Request(format!("spill restore kda conv len: {err}")))?;
                let clen = u64::from_le_bytes(clen_buf) as usize;
                let mut craw = vec![0u8; clen * 2];
                store.read_exact_into(&mut craw).map_err(|err| ModelError::Request(format!("spill restore kda conv: {err}")))?;
                let conv: Vec<u16> = craw.chunks_exact(2).map(|ch| u16::from_le_bytes(ch.try_into().unwrap())).collect();
                self.pools[ord]
                    .restore_slot_state(slot, &s, parity, &conv, &self.stream)
                    .map_err(|e| ModelError::Config(format!("kda spill restore: {e}")))?;
            }
            // MTP tap ring and metadata
            if let Some(m) = self.mtp.as_mut() {
                let mut tlen_buf = [0u8; 8];
                store
                    .read_exact_into(&mut tlen_buf)
                    .map_err(|err| ModelError::Request(format!("spill restore taps len: {err}")))?;
                let tlen = u64::from_le_bytes(tlen_buf) as usize;
                let bytes = tlen * 2;
                if bytes > self.dev_taps[slot].bytes() {
                    return Err(ModelError::Request(format!("spill restore taps of {bytes} B exceed the ring")));
                }
                let mut traw = vec![0u8; bytes];
                store.read_exact_into(&mut traw).map_err(|err| ModelError::Request(format!("spill restore taps: {err}")))?;
                let taps: Vec<u16> = traw.chunks_exact(2).map(|ch| u16::from_le_bytes(ch.try_into().unwrap())).collect();
                unsafe {
                    self.stream
                        .memcpy_h2d_async(self.dev_taps[slot].ptr(), taps.as_ptr() as *const u8, bytes)
                        .map_err(|e| ModelError::Config(format!("taps spill upload: {e}")))?;
                }
                let mut abuf = [0u8; 8];
                store.read_exact_into(&mut abuf).map_err(|err| ModelError::Request(format!("spill restore absorbed: {err}")))?;
                let absorbed = u64::from_le_bytes(abuf) as usize;
                let mut vbuf = [0u8; 8];
                store
                    .read_exact_into(&mut vbuf)
                    .map_err(|err| ModelError::Request(format!("spill restore valid_from: {err}")))?;
                let valid_from = u64::from_le_bytes(vbuf) as usize;
                let mut pbuf = [0u8; 1];
                store
                    .read_exact_into(&mut pbuf)
                    .map_err(|err| ModelError::Request(format!("spill restore last_out flag: {err}")))?;
                let mut lbuf = [0u8; 8];
                store
                    .read_exact_into(&mut lbuf)
                    .map_err(|err| ModelError::Request(format!("spill restore last_out len: {err}")))?;
                let llen = u64::from_le_bytes(lbuf) as usize;
                let last_out = if pbuf[0] == 1 {
                    let mut lraw = vec![0u8; llen * 2];
                    store
                        .read_exact_into(&mut lraw)
                        .map_err(|err| ModelError::Request(format!("spill restore last_out: {err}")))?;
                    Some(lraw.chunks_exact(2).map(|ch| u16::from_le_bytes(ch.try_into().unwrap())).collect())
                } else {
                    None
                };
                m.slots[slot].absorbed = absorbed;
                m.slots[slot].valid_from = valid_from;
                m.slots[slot].last_out = last_out;
                m.slots[slot].head_dirty = true;
            }
            store.finish_read().map_err(|err| ModelError::Request(format!("spill restore finish: {err}")))?;
            self.sync()?;
            Ok(())
        })();
        if result.is_err() {
            // Covers errors outside read_exact_into/finish_read too (page admission, device restore and upload).
            store.abort_read();
        }
        self.spill = Some(store);
        result
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
                Some(StoredSnapshot::Ram(e)) if e.len == len => {}
                Some(StoredSnapshot::Spill(e)) if e.len == len => {}
                _ => {
                    let got = entry.map_or(0, |e| match e {
                        StoredSnapshot::Ram(r) => r.len,
                        StoredSnapshot::Spill(s) => s.len,
                    });
                    return Err(ModelError::Request(format!("snapshot {snap} holds length {got} not {len}")));
                }
            }
        }
        self.begin_slot(slot)?;
        self.invalidate_batch();
        // take ownership of the entry for the restore, so field methods can borrow `self` freely; put it back
        // either way (a failed restore leaves the snapshot intact for another try)
        let entry = self.snaps.as_mut().unwrap().slots[snap as usize].take().unwrap();
        let (result, back) = match entry {
            StoredSnapshot::Ram(snapshot) => {
                let result = (|| -> Result<(), ModelError> {
                    for (ord, k) in snapshot.kda.iter().enumerate() {
                        self.pools[ord]
                            .restore_slot_state(slot, &k.s, k.parity, &k.conv, &self.stream)
                            .map_err(|e| ModelError::Config(format!("kda restore: {e}")))?;
                    }
                    // allocate fresh pages for `len` and copy the snapshot's page contents in logical order (the snapshot's
                    // physical pages may be owned by other live slots by now, so the contents move, never the pages)
                    self.ensure_pages(slot, len.saturating_sub(1))?;
                    let np = self.slot_pages[slot].len();
                    for (ord, cs) in snapshot.mla.iter().enumerate() {
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
                        if let Some(mt) = &snapshot.mtp {
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
                            m.slots[slot].valid_from = mt.valid_from;
                            m.slots[slot].last_out = mt.last_out.clone();
                            m.slots[slot].head_dirty = true;
                        }
                    }
                    self.sync()?;
                    Ok(())
                })();
                (result, StoredSnapshot::Ram(snapshot))
            }
            StoredSnapshot::Spill(e) => {
                let result = self.restore_prefix_spill(&e, slot, len);
                (result, StoredSnapshot::Spill(e))
            }
        };
        if let Some(pool) = self.snaps.as_mut() {
            pool.slots[snap as usize] = Some(back);
        }
        result
    }

    // ------------------------------------------------------------------------------------------------------ the trunk

    /// Upload the batch's row arrays.
    fn upload_batch(&mut self, b: &crate::rows::RowBatch) -> Result<(), ModelError> {
        if b.rows == 0 || b.rows > self.prefill_rows {
            return Err(ModelError::Request(format!("a forward takes 1..={} rows, got {}", self.prefill_rows, b.rows)));
        }
        let (stage, st) = (&mut self.stage, &self.stream);
        put(stage, &mut self.s.ids, &b.ids, st, "ids")?;
        if self.embed_in_place {
            embed_rows_in_place(
                stage,
                &mut self.embed_host,
                &self.embed_dev,
                self.cfg.vocab,
                self.cfg.hidden,
                &b.ids,
                &mut self.s.hidden,
                st,
            )?;
        }
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

    /// The last `t` of a call's `rows` rows: the sub-index the Q/attention side of a `Tail::Last` absorb runs on (only
    /// those rows' L72 outputs are read; the KV write used the whole index).
    fn tail_row_index(&self, rows: usize, t: usize) -> RowIndex {
        let skip = rows - t;
        RowIndex { slot: self.s.row_slot.subview(skip * 4, t * 4), pos: self.s.row_pos.subview(skip * 4, t * 4), rows: t }
    }

    /// One Gated-MLA layer's attention math over `rows` rows under this model's contract: the scalar
    /// `mla_gated::mla_gated_rows` under P1, the tensor-core `ling_mla_attn::mla_gated_rows_tc` under P2.
    /// Both consume the same `q`, `kv`, cache, packed weights and scratch; the selection is by contract,
    /// not by call type (prefill / verify / decode / L72 all dispatch here). The `x` layer input feeds the
    /// sigmoid `g_proj` output gate (fused into the expand under P2).
    ///
    /// Free function (not `&mut self`) so the caller can hold `&self.s.norm` / `&self.s.q` / `&self.s.kv`
    /// while passing `&mut self.mla_caches[ord]` and the `&mut self.s.*` scratch — disjoint fields.
    #[allow(clippy::too_many_arguments)]
    fn mla_attention_rows(
        contract: PrefillContract,
        x: &DevBuf,
        q: &DevBuf,
        kv: &DevBuf,
        norm_w: &DevBuf,
        cos: &DevBuf,
        sin: &DevBuf,
        w: &mla_gated::MlaGatedWeights,
        cache: &mut MlaGatedCache,
        table: &BlockTable,
        ri: &RowIndex,
        rows: usize,
        q_lat: &mut DevBuf,
        q_rot: &mut DevBuf,
        o_lat: &mut DevBuf,
        out: &mut DevBuf,
        gated: &mut DevBuf,
        pair: bool,
        split_ws: Option<&mut DevBuf>,
        dense_suh: Option<&DevBuf>,
        dense_xh: Option<&mut DevBuf>,
        st: &Stream,
    ) -> Result<(), ModelError> {
        match contract {
            PrefillContract::P1 => mla_gated::mla_gated_rows(
                x, q, kv, norm_w, cos, sin, w, cache, table, ri, rows, q_lat, q_rot, o_lat, out, gated, st,
            ),
            // a decode / verify call (`split_ws` is its partials buffer) runs the split-key attention; prefill keeps the walk kernels
            PrefillContract::P2 => match split_ws {
                Some(ws) => ling_mla_attn::mla_gated_rows_tc_split(
                    x, q, kv, norm_w, cos, sin, w, cache, table, ri, rows, q_lat, q_rot, o_lat, gated, ws, dense_suh, dense_xh,
                    st,
                ),
                None => ling_mla_attn::mla_gated_rows_tc_prerot(
                    x, q, kv, norm_w, cos, sin, w, cache, table, ri, rows, q_lat, q_rot, o_lat, gated, pair, dense_suh, dense_xh,
                    st,
                ),
            },
        }
        .map_err(|e| ModelError::Config(format!("mla rows m={rows} contract={}: {e}", contract.id())))
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
        // P1: dense projections use the row-invariant decode path (Lin::forward / k2_forward) and fp32 KDA qkv.
        // P2: the fast LP3 direct-MMA prefill GEMM (forward_prefill_auto) and bf16 KDA qkv.
        let p1_dense = self.mla_contract == PrefillContract::P1;

        // the rows of a table read in place from its shard were gathered on the host by `upload_batch` (see `embed_rows_in_place`);
        // a device copy of the table is gathered here, by the kernel. On decode/verify windows whose layer 0 is a
        // KDA triple, the embed rows are fused into layer 0's fused norm+triple kernel instead (it writes `hidden`
        // and normalises straight off the table), so the standalone embed launch is skipped.
        let fuse_layer0_embed =
            !prefill && rows <= 8 && !self.embed_in_place && matches!(self.dev.layers[0].mixer, MixerDevice::Kda(_)) && {
                if let MixerDevice::Kda(k0) = &self.dev.layers[0].mixer {
                    k0.qkv.k == k0.f.k
                        && k0.f.k == k0.g.k
                        && k0.qkv.codebook == k0.f.codebook
                        && k0.f.codebook == k0.g.codebook
                        && k0.qkv.n % 128 == 0
                        && k0.f.n % 128 == 0
                        && k0.g.n % 128 == 0
                } else {
                    false
                }
            };
        if !self.embed_in_place && !fuse_layer0_embed {
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
        }

        // layer 0's input norm; every later layer's input norm is fused into its predecessor's FFN residual.
        // Decode/verify windows defer it into layer 0's first fused projection launch (ling_kda_proj_fused)
        // and run the exact same rmsnorm math inside it (one fewer launch, same bits). P2 prefill instead
        // fuses the layer-0 input norm with the first mixer projection's input rotation into `prefill_ws`
        // (norm_rot_in_single runs the exact glue::rmsnorm math and the exact gb_glm_p2_rotate math), so that
        // projection's P2 GEMM below skips its rotate launch.
        let mut norm_pending: Option<bool> = None;
        let mut l0_prerot = false;
        if !prefill && rows <= 8 {
            norm_pending = Some(false);
        } else if prefill && !p1_dense {
            let first_suh = match &self.dev.layers[0].mixer {
                MixerDevice::Kda(k) => Some(&k.qkv.suh),
                MixerDevice::Mla(m) => Some(&m.q_a.suh),
            };
            let ok = if let Some(suh) = first_suh {
                gb_cuda::ling_kda_proj_fused::norm_rot_in_single(
                    &self.s.hidden.subview(0, rows * h * 2),
                    None,
                    &self.dev.layers[0].input_norm,
                    None,
                    &mut self.s.norm.subview(0, rows * h * 2),
                    suh,
                    &mut self.s.prefill_ws.subview(0, rows * h * 2),
                    rows,
                    h,
                    eps,
                    st,
                )
                .is_ok()
            } else {
                false
            };
            if ok {
                l0_prerot = true;
            } else {
                unsafe {
                    glue::rmsnorm(self.s.hidden.ptr(), self.dev.layers[0].input_norm.ptr(), self.s.norm.ptr(), rows, h, eps, st)?;
                }
            }
        } else {
            unsafe {
                glue::rmsnorm(self.s.hidden.ptr(), self.dev.layers[0].input_norm.ptr(), self.s.norm.ptr(), rows, h, eps, st)?;
            }
        }
        // Prerot state: the previous dense FFN's fused down epilogue (and this layer's mixer epilogue) rotate
        // their own rmsnorm output into `prerot_ws` for the next consumer (the dense gate/up pair, or the next
        // layer's first fused projection), letting those consumers skip their `rot_in_{pair,triple}` launch.
        let mut prerot_next = false;
        let mut prerot_gate_up = false;
        // last-layer head prerot: the final residual add + final norm fused with the lm_head's input rotation
        // (norm_rot_in_single writes xh into `ws`), so the head linear below skips its rot_in launch
        let mut head_prerot = false;
        // P2 prefill: the previous layer's FFN epilogue (or the layer-0 input norm) rotated this layer's first
        // mixer projection input into `prefill_ws`, so that projection's P2 GEMM below skips its rotate launch
        let mut next_proj_prerot = false;
        for l in 0..cfg.layers.len() {
            let layer = &self.dev.layers[l];
            // consume the previous layer's prerot (this layer's first mixer projection); layer 0's comes from
            // the fused input norm above
            let first_proj_prerot = if l == 0 { l0_prerot } else { next_proj_prerot };
            if l > 0 {
                next_proj_prerot = false;
            }
            // P2 prefill dense FFN prerots: the mixer's post-projection norm (gate) and the swiglu fusion (down)
            // write the rotated inputs into prefill_ws; the dense FFN below consumes them
            let mut gate_prerot = false;
            let mut down_prerot = false;
            match &layer.mixer {
                MixerDevice::Kda(k) => {
                    let ord = self.kda_ord[l];
                    let norm_fused = norm_pending.take();
                    // prefill: the ONE shared chunk-sized record (this layer is committed below, before the next runs);
                    // a decode/verify window: the layer's own small record, kept for the later commit
                    let rec = if prefill { &mut self.prefill_kda_rec } else { &mut self.recs[ord] };
                    if prefill && !p1_dense {
                        // P2: the LP3 direct-MMA prefill GEMM, bf16 qkv; when the input norm (layer 0) or the
                        // previous layer's FFN epilogue already rotated this norm into prefill_ws (first_proj_prerot),
                        // run the GEMM directly on it (no rotate launch)
                        if first_proj_prerot {
                            gb_cuda::ling_kda_proj_fused::ling_p2_gemm_prerot(
                                &self.s.prefill_ws.subview(0, rows * k.qkv.k * 2),
                                &mut self.s.qkv,
                                rows,
                                Dtype::BF16,
                                k.qkv.k,
                                k.qkv.n,
                                k.qkv.k2,
                                k.qkv.codebook,
                                &k.qkv.trellis,
                                &k.qkv.svh,
                                st,
                            )
                            .map_err(|e| ModelError::Config(format!("kda qkv prefill L{l} m={rows}: {e}")))?;
                        } else {
                            k.qkv
                                .forward_prefill_auto(
                                    &self.s.norm,
                                    &mut self.s.qkv,
                                    rows,
                                    Dtype::BF16,
                                    Dtype::BF16,
                                    &mut self.s.prefill_ws,
                                    st,
                                )
                                .map_err(|e| ModelError::Config(format!("kda qkv prefill L{l} m={rows}: {e}")))?;
                        }
                        k.f.forward_prefill_auto(
                            &self.s.norm,
                            &mut self.s.f_out,
                            rows,
                            Dtype::BF16,
                            Dtype::F16,
                            &mut self.s.prefill_ws,
                            st,
                        )
                        .map_err(|e| ModelError::Config(format!("kda f prefill L{l} m={rows}: {e}")))?;
                        k.g.forward_prefill_auto(
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
                        // P1 (and decode/verify): the row-invariant decode path, fp32 qkv
                        // qkv/f/g take the same x = norm: one fused triple launch with one rot_in per suh (bit-identical per
                        // element, ling_kda_proj_fused), when the three match (same k, codebook, 128-column n, K2 7/6);
                        // otherwise the separate launches (qkv, then the f/g pair or the two singles)
                        let triple = k.qkv.k == k.f.k
                            && k.f.k == k.g.k
                            && k.qkv.codebook == k.f.codebook
                            && k.f.codebook == k.g.codebook
                            && k.qkv.n % 128 == 0
                            && k.f.n % 128 == 0
                            && k.g.n % 128 == 0;
                        if triple {
                            if let Some(has_res) = norm_fused {
                                // the deferred input norm (layer 0 or the previous MoE layer) runs inside the
                                // triple's rotation launch: exact rmsnorm/add_rmsnorm math + rot_in in one launch
                                let xh = self.s.hidden.subview(0, rows * h * 2);
                                let mut nh = self.s.norm.subview(0, rows * h * 2);
                                let (res_h, mut ho) = if has_res {
                                    (Some(self.s.moe_out.subview(0, rows * h * 2)), Some(self.s.hidden.subview(0, rows * h * 2)))
                                } else if l == 0 && fuse_layer0_embed {
                                    // the embed rows are fused into this norm launch: it writes them to `hidden`
                                    // (Layer 0's residual add reads them later) and normalises straight off the table
                                    (None, Some(self.s.hidden.subview(0, rows * h * 2)))
                                } else {
                                    (None, None)
                                };
                                let (emb_ids, emb_tab, emb_vocab) = if l == 0 && fuse_layer0_embed {
                                    (
                                        Some(self.s.ids.subview(0, rows * 4)),
                                        Some(self.embed_dev.subview(0, cfg.vocab * h * 2)),
                                        cfg.vocab,
                                    )
                                } else {
                                    (None, None, 0)
                                };
                                gb_cuda::ling_kda_proj_fused::norm_rot_in_triple_ext(
                                    &xh,
                                    res_h.as_ref(),
                                    &layer.input_norm,
                                    ho.as_mut(),
                                    &mut nh,
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &k.qkv.trellis,
                                        suh: &k.qkv.suh,
                                        svh: &k.qkv.svh,
                                        y: &mut self.s.qkv,
                                        k: k.qkv.k,
                                        n: k.qkv.n,
                                        k2: k.qkv.k2,
                                        codebook: k.qkv.codebook as i32,
                                    },
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &k.f.trellis,
                                        suh: &k.f.suh,
                                        svh: &k.f.svh,
                                        y: &mut self.s.f_out,
                                        k: k.f.k,
                                        n: k.f.n,
                                        k2: k.f.k2,
                                        codebook: k.f.codebook as i32,
                                    },
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &k.g.trellis,
                                        suh: &k.g.suh,
                                        svh: &k.g.svh,
                                        y: &mut self.s.g_out,
                                        k: k.g.k,
                                        n: k.g.n,
                                        k2: k.g.k2,
                                        codebook: k.g.codebook as i32,
                                    },
                                    rows,
                                    h,
                                    Dtype::F32,
                                    Dtype::F16,
                                    k.qkv.k2,
                                    k.f.k2,
                                    k.qkv.codebook as i32,
                                    eps,
                                    &mut self.s.ws,
                                    emb_ids.as_ref(),
                                    emb_tab.as_ref(),
                                    emb_vocab,
                                    st,
                                )
                                .map_err(|e| ModelError::Cuda(e))?;
                            } else if prerot_next && !prefill && rows <= 8 {
                                // the previous dense FFN's fused down epilogue already rotated this norm into
                                // prerot_ws: feed those directly (no rot_in_triple launch)
                                let rh = rows * h * 2;
                                let xh0 = self.s.prerot_ws.subview(0, rh);
                                let xh1 = self.s.prerot_ws.subview(rh, rh);
                                let xh2 = self.s.prerot_ws.subview(2 * rh, rh);
                                gb_cuda::ling_kda_proj_fused::fused_decode_triple(
                                    &self.s.norm,
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &k.qkv.trellis,
                                        suh: &k.qkv.suh,
                                        svh: &k.qkv.svh,
                                        y: &mut self.s.qkv,
                                        k: k.qkv.k,
                                        n: k.qkv.n,
                                        k2: k.qkv.k2,
                                        codebook: k.qkv.codebook as i32,
                                    },
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &k.f.trellis,
                                        suh: &k.f.suh,
                                        svh: &k.f.svh,
                                        y: &mut self.s.f_out,
                                        k: k.f.k,
                                        n: k.f.n,
                                        k2: k.f.k2,
                                        codebook: k.f.codebook as i32,
                                    },
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &k.g.trellis,
                                        suh: &k.g.suh,
                                        svh: &k.g.svh,
                                        y: &mut self.s.g_out,
                                        k: k.g.k,
                                        n: k.g.n,
                                        k2: k.g.k2,
                                        codebook: k.g.codebook as i32,
                                    },
                                    rows,
                                    Dtype::BF16,
                                    Dtype::F32,
                                    Dtype::F16,
                                    &mut self.s.ws,
                                    Some(&xh0),
                                    Some(&xh1),
                                    Some(&xh2),
                                    st,
                                )
                                .map_err(|e| ModelError::Cuda(e))?;
                            } else {
                                gb_cuda::ling_kda_proj_fused::fused_decode_triple(
                                    &self.s.norm,
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &k.qkv.trellis,
                                        suh: &k.qkv.suh,
                                        svh: &k.qkv.svh,
                                        y: &mut self.s.qkv,
                                        k: k.qkv.k,
                                        n: k.qkv.n,
                                        k2: k.qkv.k2,
                                        codebook: k.qkv.codebook as i32,
                                    },
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &k.f.trellis,
                                        suh: &k.f.suh,
                                        svh: &k.f.svh,
                                        y: &mut self.s.f_out,
                                        k: k.f.k,
                                        n: k.f.n,
                                        k2: k.f.k2,
                                        codebook: k.f.codebook as i32,
                                    },
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &k.g.trellis,
                                        suh: &k.g.suh,
                                        svh: &k.g.svh,
                                        y: &mut self.s.g_out,
                                        k: k.g.k,
                                        n: k.g.n,
                                        k2: k.g.k2,
                                        codebook: k.g.codebook as i32,
                                    },
                                    rows,
                                    Dtype::BF16,
                                    Dtype::F32,
                                    Dtype::F16,
                                    &mut self.s.ws,
                                    None,
                                    None,
                                    None,
                                    st,
                                )
                                .map_err(|e| ModelError::Cuda(e))?;
                            }
                        } else {
                            if let Some(has_res) = norm_fused {
                                // no fused launcher absorbs the deferred norm: emit it now, identically to its producer site
                                if has_res {
                                    unsafe {
                                        glue::add_rmsnorm(
                                            self.s.hidden.ptr().cast_const(),
                                            self.s.moe_out.ptr().cast_const(),
                                            layer.input_norm.ptr().cast_const(),
                                            self.s.hidden.ptr(),
                                            self.s.norm.ptr(),
                                            rows,
                                            h,
                                            eps,
                                            st,
                                        )?;
                                    }
                                } else {
                                    unsafe {
                                        glue::rmsnorm(
                                            self.s.hidden.ptr(),
                                            layer.input_norm.ptr(),
                                            self.s.norm.ptr(),
                                            rows,
                                            h,
                                            eps,
                                            st,
                                        )?;
                                    }
                                }
                            }
                            k.qkv.forward(&self.s.norm, &mut self.s.qkv, rows, Dtype::BF16, Dtype::F32, &mut self.s.ws, st)?;
                            // f/g take the same x = norm: one fused launch with one paired rot_in, a buffer per suh (bit-identical per
                            // element, ling_kda_proj_fused), when the pair matches; otherwise the two separate launches
                            let pair = k.f.k == k.g.k && k.f.n == k.g.n && k.f.k2 == k.g.k2 && k.f.codebook == k.g.codebook;
                            if pair {
                                gb_cuda::ling_kda_proj_fused::fused_decode_pair(
                                    &self.s.norm,
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &k.f.trellis,
                                        suh: &k.f.suh,
                                        svh: &k.f.svh,
                                        y: &mut self.s.f_out,
                                        k: k.f.k,
                                        n: k.f.n,
                                        k2: k.f.k2,
                                        codebook: k.f.codebook as i32,
                                    },
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &k.g.trellis,
                                        suh: &k.g.suh,
                                        svh: &k.g.svh,
                                        y: &mut self.s.g_out,
                                        k: k.g.k,
                                        n: k.g.n,
                                        k2: k.g.k2,
                                        codebook: k.g.codebook as i32,
                                    },
                                    rows,
                                    Dtype::BF16,
                                    Dtype::F16,
                                    &mut self.s.ws,
                                    None,
                                    None,
                                    st,
                                )
                                .map_err(|e| ModelError::Cuda(e))?;
                            } else {
                                k.f.forward(&self.s.norm, &mut self.s.f_out, rows, Dtype::BF16, Dtype::F16, &mut self.s.ws, st)?;
                                k.g.forward(&self.s.norm, &mut self.s.g_out, rows, Dtype::BF16, Dtype::F16, &mut self.s.ws, st)?;
                            }
                        }
                    }
                    let w = k.view(&self.dev.kda_dummies);
                    // one batch per forward: `upload_batch` already filled `seq_start`/`seq_slot` (no per-layer
                    // `BatchBuf` alloc/free, no duplicate `gate_from_full_fg`)
                    let batch = kda::Batch { seq_start: &self.s.seq_start, slot: &self.s.seq_slot, n_seq, rows };
                    let fused_gate_prep = (!prefill && rows <= 8) || (prefill && !p1_dense);
                    // decode/verify windows of up to 8 rows with n_seq within the counter buffer run the FULL fused
                    // prep + beta + fgg + window + norm launch (`prep_beta_fgg_window_norm_fused`); the separate
                    // gate+prep + window+norm pair below is skipped (its work is inside that one launch).
                    let kda_fused_full = !prefill && rows <= 8 && n_seq * heads <= self.kda_head_cnt.bytes() / 4;
                    if !fused_gate_prep {
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
                    }
                    if prefill && !p1_dense {
                        // P2: gate+prep in ONE fused launch (bit-identical per element to `gate_from_full_fg` +
                        // `prep_prefill_bf16`; see `gb_cuda::ling_kda_fused::prep_gate_fused_prefill`). bf16 qkv
                        // already holds the rounded raw rows: skip the all-rows copy into the record.
                        gb_cuda::ling_kda_fused::prep_gate_fused_prefill(
                            st,
                            dims,
                            &w,
                            &self.s.qkv,
                            &self.pools[ord],
                            &batch,
                            rec,
                            &mut self.kda_ws,
                            &self.s.f_out,
                            &self.s.g_out,
                            &self.s.norm,
                            &k.b,
                            &k.a_log,
                            &k.dt_bias,
                            KDA_LOWER,
                        )
                        .map_err(|e| ModelError::Config(format!("kda fused gate+prep L{l} m={rows}: {e}")))?;
                    } else if fused_gate_prep && !kda_fused_full {
                        // decode/verify: the fused gate+prep launch (bit-identical per element)
                        gb_cuda::ling_kda_fused::prep_gate_fused(
                            st,
                            dims,
                            &w,
                            &self.s.qkv,
                            &self.pools[ord],
                            &batch,
                            rec,
                            &mut self.kda_ws,
                            &self.s.f_out,
                            &self.s.g_out,
                            &self.s.norm,
                            &k.b,
                            &k.a_log,
                            &k.dt_bias,
                            KDA_LOWER,
                        )
                        .map_err(|e| ModelError::Config(format!("kda fused gate+prep L{l} m={rows}: {e}")))?;
                    } else if !kda_fused_full {
                        // P1: fp32 qkv through the standard prep path
                        kda::prep(st, dims, &w, &self.s.qkv, Dtype::F32, &mut self.pools[ord], &batch, rec, &mut self.kda_ws)
                            .map_err(|e| ModelError::Config(format!("kda prep L{l} m={rows}: {e}")))?;
                    }
                    if prefill && !p1_dense {
                        // P2: chunk + gated norm + commit tail in ONE fused launch (bit-identical per element to
                        // `window_chunked` + `norm` + `commit_from`; see
                        // `gb_cuda::ling_kda_fused::chunk_norm_commit_fused`).
                        gb_cuda::ling_kda_fused::chunk_norm_commit_fused(
                            st,
                            dims,
                            &self.kda_params,
                            &mut self.pools[ord],
                            first_seq_slot as usize,
                            rows,
                            rec,
                            &mut self.kda_ws,
                            &self.kda_head_cnt,
                            &w,
                            self.s.qkv.ptr(),
                            &mut self.s.gated,
                            Some(&k.o.suh),
                            Some(&mut self.s.prefill_ws.subview(0, rows * k.o.k * 2)),
                        )
                        .map_err(|e| ModelError::Config(format!("kda fused chunk+norm+commit L{l} m={rows}: {e}")))?;
                    } else if prefill {
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
                        kda::norm(st, dims, &w, &self.kda_params, rows, &self.kda_ws, &mut self.s.gated)
                            .map_err(|e| ModelError::Config(format!("kda norm L{l} m={rows}: {e}")))?;
                    } else if kda_fused_full {
                        // decode/verify: prep + beta + fgg + window + gated norm in ONE fused Ling launch
                        // (bit-identical per element to `prep_gate_fused` + `window_and_norm`; see
                        // `gb_cuda::ling_kda_fused::prep_beta_fgg_window_norm_fused`), on the rows <= 8 profile
                        // with n_seq within the counter buffer. The KDA o_proj input rotation (rot_in with k.o.suh)
                        // is fused into the norm epilogue: the epilogue writes k.o rotated input into `ws` (offset
                        // 0, the linear xh slot), so k.o linear below skips its rot_in launch.
                        let mut o_xh = self.s.ws.subview(0, rows * k.o.k * 2);
                        gb_cuda::ling_kda_fused::prep_beta_fgg_window_norm_fused(
                            st,
                            dims,
                            &self.kda_params,
                            &w,
                            &self.s.qkv,
                            &self.pools[ord],
                            &batch,
                            rec,
                            &mut self.kda_ws,
                            &self.kda_head_cnt,
                            &self.s.f_out,
                            &self.s.g_out,
                            &self.s.norm,
                            &k.b,
                            &k.a_log,
                            &k.dt_bias,
                            KDA_LOWER,
                            Some(&k.o.suh),
                            Some(&mut o_xh),
                            &mut self.s.gated,
                        )
                        .map_err(|e| ModelError::Config(format!("kda fused prep+beta+fgg+window+norm L{l} m={rows}: {e}")))?;
                    } else {
                        kda::window(st, dims, &self.kda_params, &mut self.pools[ord], &batch, rec, &mut self.kda_ws, true)
                            .map_err(|e| ModelError::Config(format!("kda window L{l} m={rows}: {e}")))?;
                        kda::norm(st, dims, &w, &self.kda_params, rows, &self.kda_ws, &mut self.s.gated)
                            .map_err(|e| ModelError::Config(format!("kda norm L{l} m={rows}: {e}")))?;
                    }
                    if prefill && p1_dense {
                        // P1: kda::prep already copied the fp32 rows into the record; commit from it.
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
                    if prefill && !p1_dense {
                        // P2: the gated norm commit rotated the k.o input into prefill_ws (k_o_prerot); run the
                        // P2 GEMM directly on it. Falls back to the auto path (which re-rotates from gated).
                        if gb_cuda::ling_kda_proj_fused::ling_p2_gemm_prerot(
                            &self.s.prefill_ws.subview(0, rows * k.o.k * 2),
                            &mut self.s.a,
                            rows,
                            Dtype::BF16,
                            k.o.k,
                            k.o.n,
                            k.o.k2,
                            k.o.codebook,
                            &k.o.trellis,
                            &k.o.svh,
                            st,
                        )
                        .is_err()
                        {
                            k.o.forward_prefill_auto(
                                &self.s.gated,
                                &mut self.s.a,
                                rows,
                                Dtype::F16,
                                Dtype::BF16,
                                &mut self.s.prefill_ws,
                                st,
                            )
                            .map_err(|e| ModelError::Config(format!("kda o prefill L{l} m={rows}: {e}")))?;
                        }
                        // hidden = bf16(hidden + a) and norm = rmsnorm(hidden; post_norm); on a dense FFN the norm
                        // is fused with the gate projection's input rotation into prefill_ws (norm_rot_in_single
                        // runs the exact glue::add_rmsnorm math), so d.gate below skips its rotate launch
                        let gate_suh = match &layer.ffn {
                            FfnDevice::Dense(d) => Some(&d.gate.suh),
                            _ => None,
                        };
                        if let Some(guh) = gate_suh {
                            let hid_in = self.s.hidden.subview(0, rows * h * 2);
                            let res_a = self.s.a.subview(0, rows * h * 2);
                            let mut hid_out = self.s.hidden.subview(0, rows * h * 2);
                            let mut nrm_out = self.s.norm.subview(0, rows * h * 2);
                            if gb_cuda::ling_kda_proj_fused::norm_rot_in_single(
                                &hid_in,
                                Some(&res_a),
                                &layer.post_norm,
                                Some(&mut hid_out),
                                &mut nrm_out,
                                guh,
                                &mut self.s.prefill_ws.subview(0, rows * h * 2),
                                rows,
                                h,
                                eps,
                                st,
                            )
                            .is_ok()
                            {
                                gate_prerot = true;
                            } else {
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
                        } else {
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
                    } else if prefill {
                        // P1 dense prefill: rows may exceed the decode ROWS, so keep the original projector
                        // + residual add + post-norm (the fused kernel is for decode/verify windows only)
                        k.o.forward(&self.s.gated, &mut self.s.a, rows, Dtype::F16, Dtype::BF16, &mut self.s.ws, st)?;
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
                    } else {
                        // fused k.o decode projection + residual add + post-norm in one launch per stage
                        // (bit-identical per element to k.o.forward + glue::add_rmsnorm; ling_kda_proj_fused)
                        let o_hidden = self.s.hidden.subview(0, rows * h * 2);
                        let mut o_out = self.s.hidden.subview(0, rows * h * 2);
                        let mut o_norm = self.s.norm.subview(0, rows * h * 2);
                        // The window+norm epilogue above already rotated `gated` with k.o.suh into `ws` when it
                        // ran (decode/verify, rows <= 8, n_seq within the counter buffer): skip k.o's rot_in then.
                        // When the dense gate/up pair follows, fuse its rot_in_pair into this epilogue (the pair
                        // consumes `prerot_ws`, so `prerot_gate_up` tells the FFN branch).
                        let skip_rot_in = kda_fused_full;
                        let rh = rows * h * 2;
                        let mut pr_xh0 = self.s.prerot_ws.subview(0, rh);
                        let mut pr_xh1 = self.s.prerot_ws.subview(rh, rh);
                        let prerot = if let FfnDevice::Dense(d) = &layer.ffn {
                            let pr = gb_cuda::ling_kda_proj_fused::Prerot::pair(&d.gate.suh, &d.up.suh, &mut pr_xh0, &mut pr_xh1);
                            prerot_gate_up = true;
                            pr
                        } else {
                            gb_cuda::ling_kda_proj_fused::Prerot::default()
                        };
                        let lin_cnt = self.s.prerot_ws.subview(3 * ROWS * h * 2 + 512, 132);
                        gb_cuda::ling_kda_proj_fused::linear_add_rmsnorm(
                            &self.s.gated,
                            Dtype::F16,
                            &k.o.trellis,
                            &k.o.suh,
                            &k.o.svh,
                            &o_hidden,
                            &layer.post_norm,
                            &mut o_out,
                            &mut o_norm,
                            rows,
                            k.o.k,
                            k.o.n,
                            k.o.k2,
                            k.o.codebook as i32,
                            h,
                            eps,
                            &mut self.s.ws,
                            skip_rot_in,
                            &prerot,
                            Some(&lin_cnt),
                            st,
                        )?;
                    }
                }
                MixerDevice::Mla(ml) => {
                    let ord = self.mla_ord[l];
                    let norm_fused = norm_pending.take();
                    // The fused MLA epilogue (q_a/kv_a split finishes + q_a norm + q_b rotation) rides the pair
                    // kernel's last CTA when this self-resetting grid counter is supplied (prerot_ws, the same slot
                    // the dense epilogues use for `lin_cnt`; zeroed at allocation, self-resetting).
                    let pair_cnt = self.s.prerot_ws.subview(3 * ROWS * h * 2 + 512, 132);
                    let mla_qkv_pair = ml.q_a.k2 == ml.kv_a.k2
                        && ml.q_a.codebook == ml.kv_a.codebook
                        && ml.q_a.n % 128 == 0
                        && ml.kv_a.n % 128 == 0;
                    // q_a and kv_a take the same x = norm: one fused pair launch (bit-identical per element,
                    // ling_kda_proj_fused) with kv_a running on the SMs q_a's narrow grid leaves idle. The fused
                    // launch is the decode/P1 path (a window's ws), never the P2 prefill GEMMs.
                    let mla_qkv_fused = mla_qkv_pair && (!prefill || p1_dense);
                    // The full MLA qkv + q_b fusion (decode/verify windows): one call replaces the pair's two
                    // rot_out finishes + the q_b rmsnorm/rot_in launches. When it runs, the q_b block below is
                    // skipped.
                    let mut mla_qkv_qb_ok = false;
                    if mla_qkv_fused {
                        if !prefill && rows <= 8 {
                            let qb_bytes = rows * ml.q_b.k * 2;
                            let mut qb_xh = self.s.prerot_ws.subview(0, qb_bytes);
                            let res = if let Some(has_res) = norm_fused {
                                let xh = self.s.hidden.subview(0, rows * h * 2);
                                let mut nh = self.s.norm.subview(0, rows * h * 2);
                                let (res_h, mut ho) = if has_res {
                                    (Some(self.s.moe_out.subview(0, rows * h * 2)), Some(self.s.hidden.subview(0, rows * h * 2)))
                                } else {
                                    (None, None)
                                };
                                gb_cuda::ling_kda_proj_fused::mla_qkv_qb_fused(
                                    &xh,
                                    Dtype::BF16,
                                    res_h.as_ref(),
                                    Some(&layer.input_norm),
                                    ho.as_mut(),
                                    Some(&mut nh),
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.q_a.trellis,
                                        suh: &ml.q_a.suh,
                                        svh: &ml.q_a.svh,
                                        y: &mut self.s.q_a,
                                        k: ml.q_a.k,
                                        n: ml.q_a.n,
                                        k2: ml.q_a.k2,
                                        codebook: ml.q_a.codebook as i32,
                                    },
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.kv_a.trellis,
                                        suh: &ml.kv_a.suh,
                                        svh: &ml.kv_a.svh,
                                        y: &mut self.s.kv,
                                        k: ml.kv_a.k,
                                        n: ml.kv_a.n,
                                        k2: ml.kv_a.k2,
                                        codebook: ml.kv_a.codebook as i32,
                                    },
                                    &ml.q_a_norm,
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.q_b.trellis,
                                        suh: &ml.q_b.suh,
                                        svh: &ml.q_b.svh,
                                        y: &mut self.s.q,
                                        k: ml.q_b.k,
                                        n: ml.q_b.n,
                                        k2: ml.q_b.k2,
                                        codebook: ml.q_b.codebook as i32,
                                    },
                                    rows,
                                    eps,
                                    &mut self.s.ws,
                                    None,
                                    None,
                                    &mut qb_xh,
                                    Some(&pair_cnt),
                                    st,
                                )
                            } else if prerot_next {
                                let rh = rows * h * 2;
                                let xh0 = self.s.prerot_ws.subview(0, rh);
                                let xh1 = self.s.prerot_ws.subview(rh, rh);
                                gb_cuda::ling_kda_proj_fused::mla_qkv_qb_fused(
                                    &self.s.norm,
                                    Dtype::BF16,
                                    None,
                                    None,
                                    None,
                                    None,
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.q_a.trellis,
                                        suh: &ml.q_a.suh,
                                        svh: &ml.q_a.svh,
                                        y: &mut self.s.q_a,
                                        k: ml.q_a.k,
                                        n: ml.q_a.n,
                                        k2: ml.q_a.k2,
                                        codebook: ml.q_a.codebook as i32,
                                    },
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.kv_a.trellis,
                                        suh: &ml.kv_a.suh,
                                        svh: &ml.kv_a.svh,
                                        y: &mut self.s.kv,
                                        k: ml.kv_a.k,
                                        n: ml.kv_a.n,
                                        k2: ml.kv_a.k2,
                                        codebook: ml.kv_a.codebook as i32,
                                    },
                                    &ml.q_a_norm,
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.q_b.trellis,
                                        suh: &ml.q_b.suh,
                                        svh: &ml.q_b.svh,
                                        y: &mut self.s.q,
                                        k: ml.q_b.k,
                                        n: ml.q_b.n,
                                        k2: ml.q_b.k2,
                                        codebook: ml.q_b.codebook as i32,
                                    },
                                    rows,
                                    eps,
                                    &mut self.s.ws,
                                    Some(&xh0),
                                    Some(&xh1),
                                    &mut qb_xh,
                                    Some(&pair_cnt),
                                    st,
                                )
                            } else {
                                gb_cuda::ling_kda_proj_fused::mla_qkv_qb_fused(
                                    &self.s.norm,
                                    Dtype::BF16,
                                    None,
                                    None,
                                    None,
                                    None,
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.q_a.trellis,
                                        suh: &ml.q_a.suh,
                                        svh: &ml.q_a.svh,
                                        y: &mut self.s.q_a,
                                        k: ml.q_a.k,
                                        n: ml.q_a.n,
                                        k2: ml.q_a.k2,
                                        codebook: ml.q_a.codebook as i32,
                                    },
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.kv_a.trellis,
                                        suh: &ml.kv_a.suh,
                                        svh: &ml.kv_a.svh,
                                        y: &mut self.s.kv,
                                        k: ml.kv_a.k,
                                        n: ml.kv_a.n,
                                        k2: ml.kv_a.k2,
                                        codebook: ml.kv_a.codebook as i32,
                                    },
                                    &ml.q_a_norm,
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.q_b.trellis,
                                        suh: &ml.q_b.suh,
                                        svh: &ml.q_b.svh,
                                        y: &mut self.s.q,
                                        k: ml.q_b.k,
                                        n: ml.q_b.n,
                                        k2: ml.q_b.k2,
                                        codebook: ml.q_b.codebook as i32,
                                    },
                                    rows,
                                    eps,
                                    &mut self.s.ws,
                                    None,
                                    None,
                                    &mut qb_xh,
                                    Some(&pair_cnt),
                                    st,
                                )
                            };
                            mla_qkv_qb_ok = res.is_ok();
                        }
                        if !mla_qkv_qb_ok {
                            if let Some(has_res) = norm_fused {
                                let xh = self.s.hidden.subview(0, rows * h * 2);
                                let mut nh = self.s.norm.subview(0, rows * h * 2);
                                let (res_h, mut ho) = if has_res {
                                    (Some(self.s.moe_out.subview(0, rows * h * 2)), Some(self.s.hidden.subview(0, rows * h * 2)))
                                } else {
                                    (None, None)
                                };
                                gb_cuda::ling_kda_proj_fused::norm_rot_in_pair(
                                    &xh,
                                    res_h.as_ref(),
                                    &layer.input_norm,
                                    ho.as_mut(),
                                    &mut nh,
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.q_a.trellis,
                                        suh: &ml.q_a.suh,
                                        svh: &ml.q_a.svh,
                                        y: &mut self.s.q_a,
                                        k: ml.q_a.k,
                                        n: ml.q_a.n,
                                        k2: ml.q_a.k2,
                                        codebook: ml.q_a.codebook as i32,
                                    },
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.kv_a.trellis,
                                        suh: &ml.kv_a.suh,
                                        svh: &ml.kv_a.svh,
                                        y: &mut self.s.kv,
                                        k: ml.kv_a.k,
                                        n: ml.kv_a.n,
                                        k2: ml.kv_a.k2,
                                        codebook: ml.kv_a.codebook as i32,
                                    },
                                    rows,
                                    h,
                                    Dtype::BF16,
                                    ml.q_a.k2,
                                    ml.q_a.codebook as i32,
                                    eps,
                                    &mut self.s.ws,
                                    st,
                                )
                                .map_err(|e| ModelError::Cuda(e))?;
                            } else if prerot_next && !prefill && rows <= 8 {
                                // the previous dense FFN's fused down epilogue already rotated this norm into
                                // prerot_ws: feed those directly (no rot_in_pair launch)
                                let rh = rows * h * 2;
                                let xh0 = self.s.prerot_ws.subview(0, rh);
                                let xh1 = self.s.prerot_ws.subview(rh, rh);
                                gb_cuda::ling_kda_proj_fused::fused_decode_pair(
                                    &self.s.norm,
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.q_a.trellis,
                                        suh: &ml.q_a.suh,
                                        svh: &ml.q_a.svh,
                                        y: &mut self.s.q_a,
                                        k: ml.q_a.k,
                                        n: ml.q_a.n,
                                        k2: ml.q_a.k2,
                                        codebook: ml.q_a.codebook as i32,
                                    },
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.kv_a.trellis,
                                        suh: &ml.kv_a.suh,
                                        svh: &ml.kv_a.svh,
                                        y: &mut self.s.kv,
                                        k: ml.kv_a.k,
                                        n: ml.kv_a.n,
                                        k2: ml.kv_a.k2,
                                        codebook: ml.kv_a.codebook as i32,
                                    },
                                    rows,
                                    Dtype::BF16,
                                    Dtype::BF16,
                                    &mut self.s.ws,
                                    Some(&xh0),
                                    Some(&xh1),
                                    st,
                                )
                                .map_err(|e| ModelError::Cuda(e))?;
                            } else {
                                gb_cuda::ling_kda_proj_fused::fused_decode_pair(
                                    &self.s.norm,
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.q_a.trellis,
                                        suh: &ml.q_a.suh,
                                        svh: &ml.q_a.svh,
                                        y: &mut self.s.q_a,
                                        k: ml.q_a.k,
                                        n: ml.q_a.n,
                                        k2: ml.q_a.k2,
                                        codebook: ml.q_a.codebook as i32,
                                    },
                                    gb_cuda::ling_kda_proj_fused::Proj {
                                        trellis: &ml.kv_a.trellis,
                                        suh: &ml.kv_a.suh,
                                        svh: &ml.kv_a.svh,
                                        y: &mut self.s.kv,
                                        k: ml.kv_a.k,
                                        n: ml.kv_a.n,
                                        k2: ml.kv_a.k2,
                                        codebook: ml.kv_a.codebook as i32,
                                    },
                                    rows,
                                    Dtype::BF16,
                                    Dtype::BF16,
                                    &mut self.s.ws,
                                    None,
                                    None,
                                    st,
                                )
                                .map_err(|e| ModelError::Cuda(e))?;
                            }
                        }
                    } else if prefill && !p1_dense {
                        if let Some(has_res) = norm_fused {
                            if has_res {
                                unsafe {
                                    glue::add_rmsnorm(
                                        self.s.hidden.ptr().cast_const(),
                                        self.s.moe_out.ptr().cast_const(),
                                        layer.input_norm.ptr().cast_const(),
                                        self.s.hidden.ptr(),
                                        self.s.norm.ptr(),
                                        rows,
                                        h,
                                        eps,
                                        st,
                                    )?;
                                }
                            } else {
                                unsafe {
                                    glue::rmsnorm(
                                        self.s.hidden.ptr(),
                                        layer.input_norm.ptr(),
                                        self.s.norm.ptr(),
                                        rows,
                                        h,
                                        eps,
                                        st,
                                    )?;
                                }
                            }
                        }
                        if first_proj_prerot {
                            gb_cuda::ling_kda_proj_fused::ling_p2_gemm_prerot(
                                &self.s.prefill_ws.subview(0, rows * ml.q_a.k * 2),
                                &mut self.s.q_a,
                                rows,
                                Dtype::BF16,
                                ml.q_a.k,
                                ml.q_a.n,
                                ml.q_a.k2,
                                ml.q_a.codebook,
                                &ml.q_a.trellis,
                                &ml.q_a.svh,
                                st,
                            )
                            .map_err(|e| ModelError::Config(format!("mla q_a prefill L{l} m={rows}: {e}")))?;
                        } else {
                            ml.q_a
                                .forward_prefill_auto(
                                    &self.s.norm,
                                    &mut self.s.q_a,
                                    rows,
                                    Dtype::BF16,
                                    Dtype::BF16,
                                    &mut self.s.prefill_ws,
                                    st,
                                )
                                .map_err(|e| ModelError::Config(format!("mla q_a prefill L{l} m={rows}: {e}")))?;
                        }
                    } else {
                        if let Some(has_res) = norm_fused {
                            if has_res {
                                unsafe {
                                    glue::add_rmsnorm(
                                        self.s.hidden.ptr().cast_const(),
                                        self.s.moe_out.ptr().cast_const(),
                                        layer.input_norm.ptr().cast_const(),
                                        self.s.hidden.ptr(),
                                        self.s.norm.ptr(),
                                        rows,
                                        h,
                                        eps,
                                        st,
                                    )?;
                                }
                            } else {
                                unsafe {
                                    glue::rmsnorm(
                                        self.s.hidden.ptr(),
                                        layer.input_norm.ptr(),
                                        self.s.norm.ptr(),
                                        rows,
                                        h,
                                        eps,
                                        st,
                                    )?;
                                }
                            }
                        }
                        ml.q_a.forward(&self.s.norm, &mut self.s.q_a, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                    }
                    if mla_qkv_qb_ok {
                        // q_b already ran inside the fused MLA call (with the pair's finishes and the norm)
                    } else if prefill && !p1_dense {
                        // P2: rmsnorm(q_a in place) + rotate by q_b.suh into prefill_ws in ONE launch
                        // (rmsnorm_rot_in runs the exact glue::rmsnorm and gb_glm_p2_rotate math), then the P2
                        // GEMM on the rotated input (bit-identical per element to the separate launches)
                        let qa_in = self.s.q_a.subview(0, rows * cfg.mla.q_lora_rank * 2);
                        let mut qa_out = self.s.q_a.subview(0, rows * cfg.mla.q_lora_rank * 2);
                        let mut qb_xh = self.s.prefill_ws.subview(0, rows * ml.q_b.k * 2);
                        if gb_cuda::ling_kda_proj_fused::rmsnorm_rot_in(
                            &qa_in,
                            &ml.q_a_norm,
                            &mut qa_out,
                            &ml.q_b.suh,
                            &mut qb_xh,
                            rows,
                            ml.q_b.k,
                            eps,
                            st,
                        )
                        .is_ok()
                        {
                            gb_cuda::ling_kda_proj_fused::ling_p2_gemm_prerot(
                                &qb_xh,
                                &mut self.s.q,
                                rows,
                                Dtype::BF16,
                                ml.q_b.k,
                                ml.q_b.n,
                                ml.q_b.k2,
                                ml.q_b.codebook,
                                &ml.q_b.trellis,
                                &ml.q_b.svh,
                                st,
                            )
                            .map_err(|e| ModelError::Config(format!("mla q_b prefill L{l} m={rows}: {e}")))?;
                        } else {
                            unsafe {
                                glue::rmsnorm(
                                    self.s.q_a.ptr(),
                                    ml.q_a_norm.ptr(),
                                    self.s.q_a.ptr(),
                                    rows,
                                    cfg.mla.q_lora_rank,
                                    eps,
                                    st,
                                )?
                            };
                            ml.q_b
                                .forward_prefill_auto(
                                    &self.s.q_a,
                                    &mut self.s.q,
                                    rows,
                                    Dtype::BF16,
                                    Dtype::BF16,
                                    &mut self.s.prefill_ws,
                                    st,
                                )
                                .map_err(|e| ModelError::Config(format!("mla q_b prefill L{l} m={rows}: {e}")))?;
                        }
                    } else if gb_cuda::ling_kda_proj_fused::rmsnorm_linear_bf16(
                        &mut self.s.q_a,
                        &ml.q_a_norm,
                        &ml.q_b.trellis,
                        &ml.q_b.suh,
                        &ml.q_b.svh,
                        &mut self.s.q,
                        rows,
                        ml.q_b.k,
                        ml.q_b.n,
                        ml.q_b.k2,
                        ml.q_b.codebook as i32,
                        eps,
                        &mut self.s.ws,
                        st,
                    )
                    .is_err()
                    {
                        // the fused rmsnorm + q_b projection does not cover this shape: separate launches
                        unsafe {
                            glue::rmsnorm(
                                self.s.q_a.ptr(),
                                ml.q_a_norm.ptr(),
                                self.s.q_a.ptr(),
                                rows,
                                cfg.mla.q_lora_rank,
                                eps,
                                st,
                            )?
                        };
                        ml.q_b.forward(&self.s.q_a, &mut self.s.q, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                    }
                    if mla_qkv_fused {
                        // kv_a already ran inside the fused launch
                    } else if prefill && !p1_dense {
                        ml.kv_a
                            .forward_prefill_auto(
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
                    let mla_c = self.mla_contract;
                    // pair the two query rows of a CTA (sharing staged key tiles) only when the whole call is
                    // one sequence: prefill chunks are; verify windows may span slots (each row then needs its
                    // own page table, so the paired kernel is not applicable).
                    let pair = prefill && n_seq <= 1;
                    // On decode/verify windows the expand+gate epilogue also rotates the dense projection's input
                    // into `ws` (the exact ling_rot_in_kernel math), so the dense linear below skips its rot_in.
                    // the expand+gate epilogue rotates the dense projection's input into prefill_ws (P2 prefill)
                    // or ws (decode/verify) with the exact ling_rot_in_kernel math, so the dense linear below skips
                    // its rot_in launch
                    let dense_rot = !prefill && rows <= 8 || (prefill && !p1_dense);
                    let mut dense_xh = if dense_rot {
                        let ws = if prefill && !p1_dense { &self.s.prefill_ws } else { &self.s.ws };
                        Some(ws.subview(0, rows * ml.dense.k * 2))
                    } else {
                        None
                    };
                    Self::mla_attention_rows(
                        mla_c,
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
                        pair,
                        if prefill { None } else { Some(&mut self.s.attn_ws) },
                        if dense_rot { Some(&ml.dense.suh) } else { None },
                        dense_xh.as_mut(),
                        st,
                    )
                    .map_err(|e| ModelError::Config(format!("mla_gated_rows L{l} m={rows}: {e}")))?;
                    if prefill && !p1_dense {
                        // P2: the expand+gate epilogue rotated the gated row into prefill_ws (dense_rot); run the
                        // P2 GEMM directly on it. Falls back to the auto path (which re-rotates from gated).
                        if !dense_rot
                            || gb_cuda::ling_kda_proj_fused::ling_p2_gemm_prerot(
                                &self.s.prefill_ws.subview(0, rows * ml.dense.k * 2),
                                &mut self.s.a,
                                rows,
                                Dtype::BF16,
                                ml.dense.k,
                                ml.dense.n,
                                ml.dense.k2,
                                ml.dense.codebook,
                                &ml.dense.trellis,
                                &ml.dense.svh,
                                st,
                            )
                            .is_err()
                        {
                            ml.dense
                                .forward_prefill_auto(
                                    &self.s.gated,
                                    &mut self.s.a,
                                    rows,
                                    Dtype::BF16,
                                    Dtype::BF16,
                                    &mut self.s.prefill_ws,
                                    st,
                                )
                                .map_err(|e| ModelError::Config(format!("mla dense prefill L{l} m={rows}: {e}")))?;
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
                    } else if prefill {
                        // P1 dense prefill: rows may exceed the decode ROWS, so keep the original projector
                        // + residual add + post-norm (the fused kernel is for decode/verify windows only)
                        ml.dense.forward(&self.s.gated, &mut self.s.a, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
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
                    } else {
                        // fused MLA dense decode projection + residual add + post-norm in one launch per stage
                        // (bit-identical per element to ml.dense.forward + glue::add_rmsnorm; ling_kda_proj_fused)
                        let d_hidden = self.s.hidden.subview(0, rows * h * 2);
                        let mut d_out = self.s.hidden.subview(0, rows * h * 2);
                        let mut d_norm = self.s.norm.subview(0, rows * h * 2);
                        // When the dense gate/up pair follows, fuse its rot_in_pair into this epilogue (the pair
                        // consumes `prerot_ws`, so `prerot_gate_up` tells the FFN branch).
                        let rh = rows * h * 2;
                        let mut pr_xh0 = self.s.prerot_ws.subview(0, rh);
                        let mut pr_xh1 = self.s.prerot_ws.subview(rh, rh);
                        let prerot = if let FfnDevice::Dense(d) = &layer.ffn {
                            let pr = gb_cuda::ling_kda_proj_fused::Prerot::pair(&d.gate.suh, &d.up.suh, &mut pr_xh0, &mut pr_xh1);
                            prerot_gate_up = true;
                            pr
                        } else {
                            gb_cuda::ling_kda_proj_fused::Prerot::default()
                        };
                        let lin_cnt = self.s.prerot_ws.subview(3 * ROWS * h * 2 + 512, 132);
                        gb_cuda::ling_kda_proj_fused::linear_add_rmsnorm(
                            &self.s.gated,
                            Dtype::BF16,
                            &ml.dense.trellis,
                            &ml.dense.suh,
                            &ml.dense.svh,
                            &d_hidden,
                            &layer.post_norm,
                            &mut d_out,
                            &mut d_norm,
                            rows,
                            ml.dense.k,
                            ml.dense.n,
                            ml.dense.k2,
                            ml.dense.codebook as i32,
                            h,
                            eps,
                            &mut self.s.ws,
                            // the expand+gate epilogue already rotated the dense input into `ws`
                            // (`dense_rot`, decode/verify rows <= 8): skip the rot_in launch
                            dense_rot,
                            &prerot,
                            Some(&lin_cnt),
                            st,
                        )?;
                    }
                }
            }

            match &layer.ffn {
                FfnDevice::Dense(d) => {
                    // the dense FFN down_proj's input rotation (swiglu(g, u) with d.down.suh) is fused into the
                    // gate/up pair's epilogue on decode/verify windows: the second CTA of each 128-column block
                    // writes the rotated down input into `ws` (offset 0, the down linear's xh slot), so the
                    // standalone swiglu-rotation launch is skipped (the counter lives in a fixed 1024-byte
                    // region at the end of prerot_ws, zeroed at allocation). Prerot must have supplied the
                    // pair's rotated inputs (so the pair itself leaves `ws` untouched).
                    let fused_down_rot = !prefill
                        && rows <= 8
                        && prerot_gate_up
                        && d.gate.k2 == d.up.k2
                        && d.gate.codebook == d.up.codebook
                        && d.gate.n % 128 == 0
                        && d.up.n % 128 == 0;
                    if prefill && !p1_dense {
                        // P2: when the mixer's post-projection norm fused the gate rotation into prefill_ws
                        // (gate_prerot), run the gate P2 GEMM directly on it; otherwise the auto path (rotates)
                        if gate_prerot {
                            gb_cuda::ling_kda_proj_fused::ling_p2_gemm_prerot(
                                &self.s.prefill_ws.subview(0, rows * d.gate.k * 2),
                                &mut self.s.g,
                                rows,
                                Dtype::BF16,
                                d.gate.k,
                                d.gate.n,
                                d.gate.k2,
                                d.gate.codebook,
                                &d.gate.trellis,
                                &d.gate.svh,
                                st,
                            )
                            .map_err(|e| ModelError::Config(format!("dense gate prefill L{l} m={rows}: {e}")))?;
                        } else {
                            d.gate
                                .forward_prefill_auto(
                                    &self.s.norm,
                                    &mut self.s.g,
                                    rows,
                                    Dtype::BF16,
                                    Dtype::BF16,
                                    &mut self.s.prefill_ws,
                                    st,
                                )
                                .map_err(|e| ModelError::Config(format!("dense gate prefill L{l} m={rows}: {e}")))?;
                        }
                        d.up.forward_prefill_auto(
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
                        // gate and up take the same x = norm: one fused pair launch (bit-identical per element,
                        // ling_kda_proj_fused) when they share k2/codebook and both n are 128-column strips
                        let pair =
                            d.gate.k2 == d.up.k2 && d.gate.codebook == d.up.codebook && d.gate.n % 128 == 0 && d.up.n % 128 == 0;
                        if pair {
                            // the mixer's fused epilogue rotated this norm into prerot_ws (gate/up pair) when it
                            // ran: feed those directly (no rot_in_pair launch)
                            let (xh_pre0, xh_pre1) = if prerot_gate_up && !prefill && rows <= 8 {
                                let rh = rows * h * 2;
                                (Some(self.s.prerot_ws.subview(0, rh)), Some(self.s.prerot_ws.subview(rh, rh)))
                            } else {
                                (None, None)
                            };
                            let mut down_xh = self.s.ws.subview(0, rows * d.down.k * 2);
                            let col_cnt = self.s.prerot_ws.subview(3 * ROWS * h * 2, 1024);
                            gb_cuda::ling_kda_proj_fused::fused_decode_pair_ext(
                                &self.s.norm,
                                gb_cuda::ling_kda_proj_fused::Proj {
                                    trellis: &d.gate.trellis,
                                    suh: &d.gate.suh,
                                    svh: &d.gate.svh,
                                    y: &mut self.s.g,
                                    k: d.gate.k,
                                    n: d.gate.n,
                                    k2: d.gate.k2,
                                    codebook: d.gate.codebook as i32,
                                },
                                gb_cuda::ling_kda_proj_fused::Proj {
                                    trellis: &d.up.trellis,
                                    suh: &d.up.suh,
                                    svh: &d.up.svh,
                                    y: &mut self.s.u,
                                    k: d.up.k,
                                    n: d.up.n,
                                    k2: d.up.k2,
                                    codebook: d.up.codebook as i32,
                                },
                                rows,
                                Dtype::BF16,
                                Dtype::BF16,
                                &mut self.s.ws,
                                xh_pre0.as_ref(),
                                xh_pre1.as_ref(),
                                if fused_down_rot { Some(&d.down.suh) } else { None },
                                if fused_down_rot { Some(&mut down_xh) } else { None },
                                if fused_down_rot { Some(&col_cnt) } else { None },
                                st,
                            )
                            .map_err(|e| ModelError::Cuda(e))?;
                        } else {
                            d.gate.forward(&self.s.norm, &mut self.s.g, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                            d.up.forward(&self.s.norm, &mut self.s.u, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                        }
                    }
                    if prefill && !p1_dense {
                        // P2: SwiGLU(g, u) fused with the down projection's input rotation into prefill_ws in ONE
                        // launch (silu_mul_rot_in runs the exact glue::swiglu and gb_glm_p2_rotate math), then the
                        // P2 GEMM on the rotated input. self.s.a is never read after d.down, so the separate
                        // swiglu launch and its output buffer are skipped entirely.
                        let g_in = self.s.g.subview(0, rows * cfg.ffn.dense_intermediate * 2);
                        let u_in = self.s.u.subview(0, rows * cfg.ffn.dense_intermediate * 2);
                        let mut down_xh = self.s.prefill_ws.subview(0, rows * d.down.k * 2);
                        if gb_cuda::ling_kda_proj_fused::silu_mul_rot_in(
                            &g_in,
                            &u_in,
                            &d.down.suh,
                            &mut down_xh,
                            rows,
                            d.down.k,
                            st,
                        )
                        .is_ok()
                        {
                            down_prerot = true;
                        } else {
                            unsafe {
                                glue::swiglu(
                                    self.s.g.ptr(),
                                    self.s.u.ptr(),
                                    self.s.a.ptr(),
                                    rows * cfg.ffn.dense_intermediate,
                                    st,
                                )?;
                            };
                        }
                        if down_prerot {
                            gb_cuda::ling_kda_proj_fused::ling_p2_gemm_prerot(
                                &down_xh,
                                &mut self.s.g,
                                rows,
                                Dtype::BF16,
                                d.down.k,
                                d.down.n,
                                d.down.k2,
                                d.down.codebook,
                                &d.down.trellis,
                                &d.down.svh,
                                st,
                            )
                            .map_err(|e| ModelError::Config(format!("dense down prefill L{l} m={rows}: {e}")))?;
                        } else {
                            d.down
                                .forward_prefill_auto(
                                    &self.s.a,
                                    &mut self.s.g,
                                    rows,
                                    Dtype::BF16,
                                    Dtype::BF16,
                                    &mut self.s.prefill_ws,
                                    st,
                                )
                                .map_err(|e| ModelError::Config(format!("dense down prefill L{l} m={rows}: {e}")))?;
                        }
                        // hidden = bf16(hidden + g) fused with the next layer's input norm (or the final norm on the last layer);
                        // on a non-last layer the norm is fused with the next layer's first projection input rotation into
                        // prefill_ws (norm_rot_in_single runs the exact glue::add_rmsnorm math), so the next layer's first P2
                        // GEMM skips its rotate launch
                        let (next_w, mut next_dst) = if l + 1 < cfg.layers.len() {
                            (&self.dev.layers[l + 1].input_norm, self.s.norm.subview(0, rows * h * 2))
                        } else {
                            (&self.dev.final_norm, self.s.final_normed.subview(0, rows * h * 2))
                        };
                        let mut fused_next_norm = false;
                        if l + 1 < cfg.layers.len() {
                            if let Some(nuh) = match &self.dev.layers[l + 1].mixer {
                                MixerDevice::Kda(k) => Some(&k.qkv.suh),
                                MixerDevice::Mla(m) => Some(&m.q_a.suh),
                            } {
                                let hid_in = self.s.hidden.subview(0, rows * h * 2);
                                let res_g = self.s.g.subview(0, rows * h * 2);
                                let mut hid_out = self.s.hidden.subview(0, rows * h * 2);
                                if gb_cuda::ling_kda_proj_fused::norm_rot_in_single(
                                    &hid_in,
                                    Some(&res_g),
                                    next_w,
                                    Some(&mut hid_out),
                                    &mut next_dst,
                                    nuh,
                                    &mut self.s.prefill_ws.subview(0, rows * h * 2),
                                    rows,
                                    h,
                                    eps,
                                    st,
                                )
                                .is_ok()
                                {
                                    fused_next_norm = true;
                                }
                            }
                        }
                        if fused_next_norm {
                            next_proj_prerot = true;
                        } else {
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
                    } else if prefill {
                        unsafe {
                            glue::swiglu(self.s.g.ptr(), self.s.u.ptr(), self.s.a.ptr(), rows * cfg.ffn.dense_intermediate, st)?
                        };
                        // P1 dense prefill: rows may exceed the decode ROWS, so keep the original projector
                        // + residual add + next input norm (the fused kernel is for decode/verify windows only)
                        d.down.forward(&self.s.a, &mut self.s.g, rows, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
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
                    } else {
                        // fused dense swiglu + down_proj decode projection + residual add + next input norm
                        // (bit-identical per element to glue::swiglu + d.down.forward + glue::add_rmsnorm; ling_kda_proj_fused)
                        let (next_w, next_dst) = if l + 1 < cfg.layers.len() {
                            (&self.dev.layers[l + 1].input_norm, &mut self.s.norm)
                        } else {
                            (&self.dev.final_norm, &mut self.s.final_normed)
                        };
                        let dn_hidden = self.s.hidden.subview(0, rows * h * 2);
                        let mut dn_out = self.s.hidden.subview(0, rows * h * 2);
                        let mut dn_norm = next_dst.subview(0, rows * h * 2);
                        // The next layer's first fused projection consumes this norm (qkv/f/g on a KDA layer, q_a/
                        // kv_a on an MLA one): fuse its rot_in_{triple,pair} into this epilogue (prerot_ws), so the
                        // next mixer skips its rotation launch.
                        let rh = rows * h * 2;
                        let mut pr_xh0 = self.s.prerot_ws.subview(0, rh);
                        let mut pr_xh1 = self.s.prerot_ws.subview(rh, rh);
                        let mut pr_xh2 = self.s.prerot_ws.subview(2 * rh, rh);
                        let prerot = if l + 1 < cfg.layers.len() {
                            let pr = match &self.dev.layers[l + 1].mixer {
                                MixerDevice::Kda(k) => {
                                    let pr = gb_cuda::ling_kda_proj_fused::Prerot::triple(
                                        &k.qkv.suh,
                                        &k.f.suh,
                                        &k.g.suh,
                                        &mut pr_xh0,
                                        &mut pr_xh1,
                                        &mut pr_xh2,
                                    );
                                    prerot_next = true;
                                    pr
                                }
                                MixerDevice::Mla(ml) => {
                                    let pr = gb_cuda::ling_kda_proj_fused::Prerot::pair(
                                        &ml.q_a.suh,
                                        &ml.kv_a.suh,
                                        &mut pr_xh0,
                                        &mut pr_xh1,
                                    );
                                    prerot_next = true;
                                    pr
                                }
                            };
                            pr
                        } else {
                            gb_cuda::ling_kda_proj_fused::Prerot::default()
                        };
                        let lin_cnt = self.s.prerot_ws.subview(3 * ROWS * h * 2 + 512, 132);
                        gb_cuda::ling_kda_proj_fused::silu_mul_linear_add_rmsnorm_ext(
                            &self.s.g,
                            &self.s.u,
                            &d.down.trellis,
                            &d.down.suh,
                            &d.down.svh,
                            &dn_hidden,
                            next_w,
                            &mut dn_out,
                            &mut dn_norm,
                            rows,
                            d.down.k,
                            d.down.n,
                            d.down.k2,
                            d.down.codebook as i32,
                            h,
                            eps,
                            &mut self.s.ws,
                            &prerot,
                            fused_down_rot,
                            Some(&lin_cnt),
                            st,
                        )?;
                    }
                }
                FfnDevice::Moe(moe) => {
                    let router = &moe.router;
                    let limit_r = cfg.routed_swiglu_limit(l);
                    let limit_s = cfg.shared_swiglu_limit(l);
                    let shape = fmoe::Shape {
                        t: rows,
                        ne: cfg.ffn.routed_experts,
                        top_k: cfg.ffn.experts_per_token,
                        h,
                        i: cfg.ffn.moe_intermediate,
                        kc: BAILING_KC,
                        max_tokens: if prefill { self.prefill_rows } else { ROWS },
                    };
                    let gr = Some(fmoe::GroupRoute {
                        expert_bias: Some(&moe.expert_bias),
                        n_group: cfg.ffn.n_group,
                        topk_group: cfg.ffn.topk_group,
                        scale: cfg.ffn.routed_scaling_factor as f32,
                    });
                    let limits = fmoe::SwigluLimit { routed: limit_r, shared: limit_s };
                    // The fused-router decode shape (see ling_fmoe::forward_with_router): the router GEMV AND the
                    // post-MoE residual add + next input norm + next consumer's rotation ride the same fused launch.
                    let moe_fused_router = !prefill
                        && rows <= 8
                        && h == 4096
                        && cfg.ffn.routed_experts == 512
                        && cfg.ffn.n_group >= 2
                        && cfg.ffn.topk_group >= 1;
                    let mut prefill_epi_ok = false;
                    if prefill {
                        // P1 keeps the scalar router at every row count (its documented arithmetic); P2 gives prompt rows the tensor-core router
                        self.s.moe_logits_f32(&self.stream, router, h, cfg.ffn.routed_experts, rows, !p1_dense)?;
                        // sorted-token grouped MoE: every expert's tile decodes once per 64 routed tokens (no split-K
                        // decode groups), row-invariant across chunkings. The post-MoE residual add + next input norm
                        // ride the combine launch (ling_fmoe::prefill_with_epilogue, exact glue::add_rmsnorm math);
                        // fall back to the separate launches when the fused path is unavailable.
                        let hid = self.s.hidden.subview(0, rows * h * 2);
                        let mut next_dst = if l + 1 < cfg.layers.len() {
                            self.s.norm.subview(0, rows * h * 2)
                        } else {
                            self.s.final_normed.subview(0, rows * h * 2)
                        };
                        let next_w =
                            if l + 1 < cfg.layers.len() { &self.dev.layers[l + 1].input_norm } else { &self.dev.final_norm };
                        // on a non-last layer the combine epilogue also rotates the next layer's first projection
                        // input into prefill_ws (the exact gb_glm_p2_rotate math on the normed row), so the next
                        // layer's first P2 GEMM skips its rotate launch
                        let next_suh = if l + 1 < cfg.layers.len() {
                            match &self.dev.layers[l + 1].mixer {
                                MixerDevice::Kda(k) => Some(&k.qkv.suh),
                                MixerDevice::Mla(m) => Some(&m.q_a.suh),
                            }
                        } else {
                            None
                        };
                        let mut next_xh = self.s.prefill_ws.subview(0, rows * h * 2);
                        let has_next = next_suh.is_some();
                        match ling_fmoe::prefill_with_epilogue(
                            &self.s.norm,
                            &self.s.moe_logits,
                            fmoe::LogitsDtype::F32,
                            &self.s.moe_shared_gate,
                            &moe.tables(),
                            &mut self.s.moe_out,
                            shape,
                            gr,
                            limits,
                            &mut self.s.moe_pws,
                            st,
                            Some(ling_fmoe::PrefillEpilogue {
                                hidden: &hid,
                                norm_w: next_w,
                                norm_out: &mut next_dst,
                                eps,
                                next_suh,
                                next_xh: if has_next { Some(&mut next_xh) } else { None },
                            }),
                        ) {
                            Ok(()) => {
                                prefill_epi_ok = true;
                                if has_next {
                                    next_proj_prerot = true;
                                }
                            }
                            Err(_) => {
                                fmoe::prefill(
                                    &self.s.norm,
                                    &self.s.moe_logits,
                                    fmoe::LogitsDtype::F32,
                                    &self.s.moe_shared_gate,
                                    &moe.tables(),
                                    &mut self.s.moe_out,
                                    shape,
                                    gr,
                                    limits,
                                    &mut self.s.moe_pws,
                                    st,
                                )?;
                            }
                        }
                    } else {
                        // decode: the router GEMV is fused into the route launch on the Ling-3.1 shape (fused, or the
                        // two-launch fallback; both bit-identical to the separate logits + forward calls). When the
                        // fused-router shape holds, the post-MoE residual add + next input norm + the next consumer's
                        // rotation ride the down kernel's last finishing warp instead of a separate
                        // `norm_rot_in_{triple,pair,single}` launch.
                        let rh = rows * h * 2;
                        let hid = self.s.hidden.subview(0, rh);
                        let mut pr_xh0 = self.s.prerot_ws.subview(0, rh);
                        let mut pr_xh1 = self.s.prerot_ws.subview(rh, rh);
                        let mut pr_xh2 = self.s.prerot_ws.subview(2 * rh, rh);
                        let mut no_next = self.s.norm.subview(0, rh);
                        let mut no_final = self.s.final_normed.subview(0, rh);
                        let mut ws_xh = self.s.ws.subview(0, rh);
                        let nw_next = if l + 1 < cfg.layers.len() {
                            Some(self.dev.layers[l + 1].input_norm.subview(0, h * 4))
                        } else {
                            None
                        };
                        let nw_final = self.dev.final_norm.subview(0, h * 4);
                        let epilogue = if moe_fused_router {
                            if l + 1 < cfg.layers.len() {
                                let nw = nw_next.as_ref().unwrap();
                                let prerot = match &self.dev.layers[l + 1].mixer {
                                    MixerDevice::Kda(k) => {
                                        let pr = gb_cuda::ling_kda_proj_fused::Prerot::triple(
                                            &k.qkv.suh,
                                            &k.f.suh,
                                            &k.g.suh,
                                            &mut pr_xh0,
                                            &mut pr_xh1,
                                            &mut pr_xh2,
                                        );
                                        prerot_next = true;
                                        pr
                                    }
                                    MixerDevice::Mla(m2) => {
                                        let pr = gb_cuda::ling_kda_proj_fused::Prerot::pair(
                                            &m2.q_a.suh,
                                            &m2.kv_a.suh,
                                            &mut pr_xh0,
                                            &mut pr_xh1,
                                        );
                                        prerot_next = true;
                                        pr
                                    }
                                };
                                Some(ling_fmoe::FmoeEpilogue { hidden: &hid, norm_w: nw, norm_out: &mut no_next, eps, prerot })
                            } else {
                                // last layer: fuse the residual add + final norm with the lm_head's input rotation
                                // into the down epilogue (writes `hidden`/`final_normed` exactly like
                                // glue::add_rmsnorm / norm_rot_in_single and rotates the normed row into `ws + 0`
                                // = xh, the head linear's pre-rotated input)
                                let prerot = gb_cuda::ling_kda_proj_fused::Prerot::single(&self.dev.lm_head.suh, &mut ws_xh);
                                head_prerot = true;
                                Some(ling_fmoe::FmoeEpilogue {
                                    hidden: &hid,
                                    norm_w: &nw_final,
                                    norm_out: &mut no_final,
                                    eps,
                                    prerot,
                                })
                            }
                        } else {
                            None
                        };
                        ling_fmoe::forward_with_router(
                            &self.s.norm,
                            router,
                            &mut self.s.moe_logits,
                            fmoe::LogitsDtype::F32,
                            &self.s.moe_shared_gate,
                            &moe.tables(),
                            &mut self.s.moe_out,
                            shape,
                            gr,
                            limits,
                            &mut self.s.moe_ws,
                            st,
                            epilogue,
                        )?;
                    }
                    if moe_fused_router {
                        // the down kernel's fused epilogue already ran the residual add + next input norm + the next
                        // consumer's rotation (norm_pending stays None; prerot_next / head_prerot were set above)
                    } else if !prefill && rows <= 8 && l + 1 < cfg.layers.len() {
                        // defer the residual add + next input norm into layer l+1's first fused projection launch
                        // (exact add_rmsnorm math inside ling_kda_proj_fused): one fewer launch, same bits
                        norm_pending = Some(true);
                    } else if !prefill && rows <= 8 && l + 1 == cfg.layers.len() {
                        // last layer (fallback when the fused-router shape did not hold): fuse the residual add +
                        // final norm with the lm_head's input rotation into ONE launch (norm_rot_in_single writes
                        // `hidden`/`final_normed` exactly like glue::add_rmsnorm and rotates the normed row into
                        // `ws + 0` = xh); the head linear below then runs on the pre-rotated input (no rot_in
                        // launch). Falls back to the separate launches when the fused path is unavailable.
                        let mut xh = self.s.ws.subview(0, rows * h * 2);
                        let mut fnrm = self.s.final_normed.subview(0, rows * h * 2);
                        let res_h = self.s.moe_out.subview(0, rows * h * 2);
                        let mut hid_out = self.s.hidden.subview(0, rows * h * 2);
                        match gb_cuda::ling_kda_proj_fused::norm_rot_in_single(
                            &self.s.hidden.subview(0, rows * h * 2),
                            Some(&res_h),
                            &self.dev.final_norm,
                            Some(&mut hid_out),
                            &mut fnrm,
                            &self.dev.lm_head.suh,
                            &mut xh,
                            rows,
                            h,
                            eps,
                            st,
                        ) {
                            Ok(()) => head_prerot = true,
                            Err(_) => unsafe {
                                glue::add_rmsnorm(
                                    self.s.hidden.ptr().cast_const(),
                                    self.s.moe_out.ptr().cast_const(),
                                    self.dev.final_norm.ptr().cast_const(),
                                    self.s.hidden.ptr(),
                                    self.s.final_normed.ptr(),
                                    rows,
                                    h,
                                    eps,
                                    st,
                                )?;
                            },
                        }
                    } else if prefill && prefill_epi_ok {
                        // the fused prefill epilogue already ran the residual add + next input norm (exact
                        // glue::add_rmsnorm math inside ling_fmoe::prefill_with_epilogue)
                    } else {
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
        }

        // head: the last layer's FFN residual fused its norm into `final_normed`, so only the logits remain.
        // Prefill chunks need no logits (the requested chunk's last row runs the head once in `prefill_chunk`).
        // Decode/verify windows of <= 8 rows with a prerot (the final norm fused the lm_head's rotation into `ws`)
        // run the head linear directly on the pre-rotated input (no rot_in launch).
        if !prefill {
            if head_prerot {
                let lm = &self.dev.lm_head;
                let xh = self.s.ws.subview(0, rows * h * 2);
                if gb_cuda::ling_kda_proj_fused::linear_prerot(
                    &lm.trellis,
                    &lm.svh,
                    None,
                    &mut self.s.logits,
                    Dtype::F16,
                    rows,
                    lm.k,
                    lm.n,
                    lm.k2,
                    lm.codebook as i32,
                    &xh,
                    &mut self.s.ws,
                    st,
                )
                .is_err()
                {
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
            } else {
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
    /// device rings. No pre-copy sync: the copies are queued on the same stream that produced `final_normed`, so they are
    /// ordered after the forward and after the staged batch uploads. Prefill's cost is now the copies alone (amortized
    /// over a chunk); decode/verify calls this from `window_candidates`, whose top-k sync then covers them.
    fn store_taps(&mut self, b: &crate::rows::RowBatch, rows: usize) -> Result<(), ModelError> {
        self.record_taps(b, rows)
    }

    /// Copy the last `rows` normed trunk hidden rows into the device tap rings (D2D, on the stream). No host round-trip.
    /// Rows of one sequence are contiguous in `final_normed` and land on consecutive ring positions, so each sequence is
    /// ONE contiguous range copy (two across the ring wrap), not one tiny copy per row.
    fn record_taps(&mut self, b: &crate::rows::RowBatch, rows: usize) -> Result<(), ModelError> {
        let h = self.cfg.hidden;
        let row_bytes = h * 2;
        let n_seq = b.n_seq();
        for i in 0..n_seq {
            let lo = b.seq_start[i] as usize;
            let hi = if i + 1 < n_seq { b.seq_start[i + 1] as usize } else { rows };
            let slot = b.seq_slot[i] as usize;
            let mut src = lo;
            let mut p = b.positions[lo] as usize;
            while src < hi {
                let len = hi - src;
                let idx = p % CAP_TAPS;
                let take = len.min(CAP_TAPS - idx);
                self.dev_taps[slot]
                    .copy_from_at(idx * row_bytes, &self.s.final_normed, src * row_bytes, take * row_bytes, &self.stream)
                    .map_err(|e| ModelError::Config(format!("taps d2d: {e}")))?;
                src += take;
                p += take;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------------ commits

    pub fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.flush()?;
        self.invalidate_batch();
        let st = &self.stream;
        for p in &mut self.pools {
            p.reset_slot(slot, st).map_err(|e| ModelError::Config(format!("kda reset: {e}")))?;
        }
        self.release_pages(slot);
        if let Some(m) = self.mtp.as_mut() {
            m.slots[slot].absorbed = 0;
            m.slots[slot].valid_from = 0;
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
            m.slots[slot].valid_from = 0;
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
        // a full accept of every sequence needs no per-row keep list: `keep = None` skips the prep_raw/replay pair (126
        // launches a step: 189 -> 63) and commits the window's already-written final state directly, with the same bits
        // (`kda::commit` documents the null-keep fast path)
        let keep_all = keep.iter().zip(&w.seq_len).all(|(&k, &l)| k == l as i32);
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
        let n_kda = self.kda_ord.iter().filter(|&&o| o != usize::MAX).count();
        if keep_all {
            // a full accept of every sequence: one batched tail launch for ALL KDA layers, replacing the per-layer
            // `gb_kda_commit_tail` launches of `kda::commit` (same bits per layer)
            gb_cuda::ling_kda_fused::commit_tail_all(&self.stream, dims, &self.kda_commit_layers, n_kda, &batch, None, true)?;
        } else {
            // partial accept: replay each layer's accepted prefix (prep_raw + gb_kda_replay), then ONE batched
            // tail launch for all layers
            for l in 0..self.cfg.layers.len() {
                let ord = self.kda_ord[l];
                if ord == usize::MAX {
                    continue;
                }
                let MixerDevice::Kda(k) = &self.dev.layers[l].mixer else { continue };
                let wv = k.view(&self.dev.kda_dummies);
                gb_cuda::ling_kda_fused::replay(
                    &self.stream,
                    dims,
                    &wv,
                    &self.kda_params,
                    &self.pools[ord],
                    &batch,
                    &self.recs[ord],
                    &mut self.kda_ws,
                    Some(&self.s.keep),
                    true,
                )?;
            }
            gb_cuda::ling_kda_fused::commit_tail_all(
                &self.stream,
                dims,
                &self.kda_commit_layers,
                n_kda,
                &batch,
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
        // The fused single-launch top-k covers every shape the engine queues (rows <= ROWS, count <= 64,
        // vocab-wide columns) and returns `cudaErrorInvalidValue` for a shape it does not cover; fall back
        // to the two-launch glm top-k (bit-identical ids/vals) then. Both need the same workspace bytes.
        match gb_cuda::ling_kda_proj_fused::topk_f16(
            &self.s.logits,
            rows,
            self.cfg.vocab,
            self.cfg.vocab,
            count,
            &mut self.s.topk_ids,
            &mut self.s.topk_vals,
            &mut self.s.topk_ws,
            &self.stream,
        ) {
            Ok(()) => Ok(()),
            Err(gb_cuda::rt::CudaError(1)) => glm_glue::topk_f16(
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
            .map_err(|e| ModelError::Config(format!("topk: {e}"))),
            Err(e) => Err(ModelError::Config(format!("topk: {e}"))),
        }
    }

    /// Sync the stream (which also releases the staged host uploads, `HostStage::sync`) and download the queued top-k
    /// ids/values. The sync also makes the already-queued forward's outputs readable, so callers that need them (e.g.
    /// `window_candidates`' MTP taps) must download after this.
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
            let end = (at + self.prefill_rows).min(n);
            last_rows = end - at;
            self.ensure_pages(slot, start + end - 1)?;
            let b = crate::rows::RowBatch::from_requests(&[gb_engine::RowRequest {
                slot,
                tokens: tokens[at..end].to_vec(),
                first_position: (start + at) as u64,
            }]);
            self.forward_trunk(&b, true, 0)?;
            // Recycle the six staged uploads (including the multi-MiB gathered embedding rows) before the
            // next chunk instead of retaining one set for every chunk in a long prompt.
            self.sync()?;
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

    /// The candidates of the committed row at `pos` of `slot` from its normed trunk hidden in the MTP tap ring: the trunk head
    /// over those bytes, the kernel `prefill_chunk` runs on a chunk's last row, so the same bits. For a slot restored to exactly
    /// its prompt (a full prefix-cache hit), whose last row must not run again; `pos` must be among the ring's last `CAP_TAPS`
    /// committed rows. Refused without the MTP head (no tap ring).
    pub fn tap_candidates(&mut self, slot: usize, pos: usize, count: usize) -> Result<Candidates, ModelError> {
        if self.mtp.is_none() || slot >= self.slots {
            return Err(ModelError::Request("a full prefix hit needs the MTP tap ring".into()));
        }
        self.flush()?;
        let h = self.cfg.hidden;
        let row_bytes = h * 2;
        let mut row = self.s.final_normed.subview(0, row_bytes);
        row.copy_from_at(0, &self.dev_taps[slot], (pos % CAP_TAPS) * row_bytes, row_bytes, &self.stream)
            .map_err(|e| ModelError::Config(format!("taps d2d: {e}")))?;
        self.dev.lm_head.forward(&row, &mut self.s.logits, 1, Dtype::BF16, Dtype::F16, &mut self.s.ws, &self.stream)?;
        self.candidates_of(1, count)
    }

    // ------------------------------------------------------------------------------------------------------ MTP drafter

    /// Where an `L72` absorb up to `limit` starts: the slot's `absorbed` count, raised to the oldest row whose tap the ring still holds
    /// (positions `limit - (CAP_TAPS - 1)..`). Rows below it cannot be absorbed (their taps are gone), so when the absorb has to skip any,
    /// the slot's [`MtpSlot::valid_from`] moves up to where the written entries start and draft attention is windowed to them
    /// ([`l72_window`]); an absorb that skips nothing leaves it alone.
    fn clamp_to_taps(&mut self, slot: usize, limit: usize) -> usize {
        let m = &mut self.mtp.as_mut().unwrap().slots[slot];
        let (start, valid_from) = absorb_start(m.absorbed, limit);
        if let Some(v) = valid_from {
            if m.valid_from != v {
                // the first draft call of a prompt past the ring (or of a hit whose cut the ring has left): say where the drafter's window starts
                eprintln!(
                    "rexl3 level=info event=ling31_draft_window slot={slot} valid_from={v} rows_without_entry={v} note=\"past {CAP_TAPS} tokens the L72 draft cache holds entries only from valid_from; draft attention is windowed to them; output tokens are verified by the trunk and do not change\""
                );
            }
            m.valid_from = v;
        }
        start
    }

    /// The `L72` attention window of a query at `pos` of `slot`: [`l72_window`] of the slot's [`MtpSlot::valid_from`].
    fn l72_window_of(&self, slot: usize, pos: usize) -> Option<(usize, i32)> {
        l72_window(self.mtp.as_ref()?.slots[slot].valid_from, pos)
    }

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
        let mut absorbed = self.clamp_to_taps(slot, limit);
        while absorbed < limit {
            let hi = (absorbed + self.prefill_rows).min(limit);
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
            gb_cuda::ling_kda_proj_fused::topk_f16(
                &sh.logits,
                1,
                sh.cols,
                sh.cols,
                2,
                &mut *topk_ids,
                &mut *topk_vals,
                &mut *topk_ws,
                st,
            )?;
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
        gb_cuda::ling_kda_proj_fused::topk_f16(
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

    /// The embeddings of `ids` into `e_next` (the L72 layer's `e` input): gathered on the host from a table read in place, by the kernel
    /// from a device copy of it (which needs the ids on the device first).
    fn embed_next(&mut self, ids: &[i32]) -> Result<(), ModelError> {
        if self.embed_in_place {
            return embed_rows_in_place(
                &mut self.stage,
                &mut self.embed_host,
                &self.embed_dev,
                self.cfg.vocab,
                self.cfg.hidden,
                ids,
                &mut self.s.e_next,
                &self.stream,
            );
        }
        put(&mut self.stage, &mut self.s.ids, ids, &self.stream, "ids")?;
        unsafe {
            glue::embed(
                self.embed_dev.ptr().cast_const(),
                self.s.ids.ptr().cast_const(),
                self.s.e_next.ptr(),
                ids.len(),
                self.cfg.hidden,
                self.cfg.vocab,
                &self.stream,
            )?;
        }
        Ok(())
    }

    /// Run `L72` over `rows` absorbed rows whose trunk hidden rows are already in `self.s.hidden` (the caller copied them
    /// from the tap ring): `ids` (tokens), positions `pos`. The last row's L72 output is left on the GPU in `head_in`
    /// (no download, no sync).
    fn mtp_absorb(&mut self, ids: &[i32], slots: &[i32], pos: &[i32], rows: usize) -> Result<(), ModelError> {
        let h = self.cfg.hidden;
        self.embed_next(ids)?;
        self.stage
            .upload(&mut self.s.row_slot, slots, &self.stream)
            .map_err(|e| ModelError::Config(format!("slots upload: {e}")))?;
        self.stage.upload(&mut self.s.row_pos, pos, &self.stream).map_err(|e| ModelError::Config(format!("pos upload: {e}")))?;
        let window = self.l72_window_of(slots[0] as usize, pos[rows - 1] as usize);
        self.mtp_layer_forward(rows, Tail::Last, window)?;
        let last = self.s.hidden.subview((rows - 1) * h * 2, h * 2);
        self.s.head_in.copy_from(&last, h * 2, &self.stream).map_err(|e| ModelError::Config(format!("head copy: {e}")))?;
        Ok(())
    }

    /// Absorb the L72 rows `lo..hi` of `tokens` (row `p` takes token `p + 1`'s embedding and position `p`'s trunk tap),
    /// copying the taps straight from the device ring into `hidden` (D2D; no host round-trip), leaving the last row's
    /// output in `head_in`. Up to `self.prefill_rows` rows run per chunk; the absorb is row-invariant (every kernel on
    /// the L72 path is), so the chunk size never changes the resulting bits. Prefix rows past the chunk's last one run
    /// only the KV half of the block (`Tail::KvOnly`): their L72 outputs are never read.
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
        let mut absorbed = self.clamp_to_taps(slot, limit);
        while absorbed < limit {
            let hi = (absorbed + self.prefill_rows).min(limit);
            self.mtp_absorb_range(slot, tokens, absorbed, hi)?;
            // Absorb chunks use the same HostStage as prefill; release their uploads before the next chunk.
            self.sync()?;
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
        self.embed_next(&[tok])?;
        let mut row0 = self.s.hidden.subview(0, h * 2);
        row0.copy_from(&self.s.chain_in, h * 2, &self.stream).map_err(|e| ModelError::Config(format!("prev copy: {e}")))?;
        self.stage
            .upload(&mut self.s.row_slot, &[slot as i32], &self.stream)
            .map_err(|e| ModelError::Config(format!("slots upload: {e}")))?;
        self.stage
            .upload(&mut self.s.row_pos, &[pos as i32], &self.stream)
            .map_err(|e| ModelError::Config(format!("pos upload: {e}")))?;
        let window = self.l72_window_of(slot, pos);
        self.mtp_layer_forward(1, Tail::All, window)?;
        let out = self.s.hidden.subview(0, h * 2);
        self.s.chain_in.copy_from(&out, h * 2, &self.stream).map_err(|e| ModelError::Config(format!("chain copy: {e}")))?;
        Ok(())
    }

    /// The `L72` block over `rows` rows whose ids/positions are already in the scratch (`hidden` holds the trunk hidden rows,
    /// `e_next` the token embeddings): `h_mtp = eh_proj(cat(enorm(e), hnorm(h)))`, then the decoder block. `tail` says how
    /// many rows run past the KV cache write (`Tail::Last` is the absorb; `Tail::All` a single chain row).
    fn mtp_layer_forward(&mut self, rows: usize, tail: Tail, window: Option<(usize, i32)>) -> Result<(), ModelError> {
        let mtp = self.dev.mtp.as_ref().ok_or_else(|| ModelError::Config("no MTP layer".into()))?;
        let h = self.cfg.hidden;
        let st = &self.stream;
        unsafe {
            glue::rmsnorm(self.s.e_next.ptr(), mtp.enorm.ptr(), self.s.norm.ptr(), rows, h, RMS_EPS, st)?;
            glue::rmsnorm(self.s.hidden.ptr(), mtp.hnorm.ptr(), self.s.final_normed.ptr(), rows, h, RMS_EPS, st)?;
        }
        // y = cat(enorm(e_next), hnorm(h_trunk)) @ eh_proj^T in one fused bf16 kernel (no f16 staging, no round-trips)
        let mut y = self.s.hidden.subview(0, rows * h * 2);
        ling_mla_attn::eh_proj_cat_bf16(&self.s.norm, &self.s.final_normed, &mtp.eh_proj, &mut y, rows, h, st)
            .map_err(|e| ModelError::Config(format!("eh_proj: {e}")))?;
        // the multi-row absorb (prefill-class) runs the MoE pass `fmoe::prefill`; single-row chain steps stay on `forward`.
        // Only the final row's MoE ever runs (a single decode-path call), whatever the chunk's width
        // Multi-row absorbs are prefill-class even though only their last row runs the Q/attention/MoE tail. Sending
        // Tail::Last down the decode path overflows the 64-row decode workspace as soon as a prompt needs an absorb chunk.
        self.layer_block_forward(true, rows > 1, rows, tail, window)?;
        // BailingMoeV3MTPLayer.forward ends with `hidden_states = self.final_layernorm(hidden_states)`
        // (modeling_bailing_moe_v3.py: the absorb/chain outputs are final-normed, the head runs lm_head on them
        // directly, and the next chain step feeds the normed output through hnorm again). Apply it on the rows that ran
        // the block's tail (one row for an absorb: its prefix rows never produced an output).
        let mtp = self.dev.mtp.as_ref().ok_or_else(|| ModelError::Config("no MTP layer".into()))?;
        let normed = match tail {
            Tail::All => rows,
            Tail::Last => 1,
        };
        let off = (rows - normed) * h * 2;
        unsafe {
            glue::rmsnorm(
                self.s.hidden.ptr_at(off),
                mtp.final_layernorm.ptr(),
                self.s.hidden.ptr_at(off),
                normed,
                h,
                RMS_EPS,
                &self.stream,
            )?;
        }
        Ok(())
    }

    /// One decoder block (mixer + ffn) over the rows already in `hidden` (bf16). `mtp` selects the L72 layer; `prefill`
    /// runs the MoE pass `fmoe::prefill` (up to `self.prefill_rows` rows, `moe_pws`) instead of `fmoe::forward`
    /// (`moe_ws`). `tail` restricts the Q/attention/MoE half to the last row (`Tail::Last`): every row still runs
    /// `input_layernorm`, `kv_a`, the latent norm + RoPE and the KV cache write, and only the last row's L72 output is
    /// produced (row-invariance makes that the same bits as the full block on that row).
    fn layer_block_forward(
        &mut self,
        mtp: bool,
        prefill: bool,
        rows: usize,
        tail: Tail,
        window: Option<(usize, i32)>,
    ) -> Result<(), ModelError> {
        let layer = if mtp {
            &self.dev.mtp.as_ref().ok_or_else(|| ModelError::Config("no MTP layer".into()))?.layer
        } else {
            &self.dev.layers[0]
        };
        let st = &self.stream;
        let cfg = self.cfg.clone();
        let h = cfg.hidden;
        let eps = cfg.rms_eps as f32;
        let p1_dense = self.mla_contract == PrefillContract::P1;
        unsafe {
            glue::rmsnorm(self.s.hidden.ptr(), layer.input_norm.ptr(), self.s.norm.ptr(), rows, h, eps, st)?;
        }
        match &layer.mixer {
            MixerDevice::Mla(ml) => {
                let ord = MTP_MLA_ORD;
                // the KV half: every absorbed row writes its cache entry, and that is all a prompt prefix row needs.
                // `Tail::Last` runs the pinned prefill GEMM at the chunk's width (the row's bits do not depend on the
                // call width, so this is the same value the full block computed).
                if prefill && !p1_dense {
                    ml.kv_a
                        .forward_prefill_auto(
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
                mla_gated::kv_prep(
                    &self.s.kv,
                    &ml.kv_a_norm,
                    &self.cos,
                    &self.sin,
                    &ri,
                    &self.table,
                    &mut self.mla_caches[ord],
                    rows,
                    st,
                )
                .map_err(|e| ModelError::Config(format!("L72 kv_prep m={rows}: {e}")))?;
                let t = match tail {
                    Tail::All => rows,
                    Tail::Last => 1,
                };
                // the Q/attention/residual half: only the rows whose L72 output is read (one for an absorb), addressed
                // by sub-views of the row-strided buffers at the last `t` rows — the row-invariant kernels give these
                // rows the same bits as a whole-block call would
                let off = |row_elems: usize| (rows - t) * row_elems;
                if prefill && !p1_dense {
                    ml.q_a
                        .forward_prefill_auto(
                            &self.s.norm.subview(off(h * 2), t * h * 2),
                            &mut self.s.q_a.subview(off(cfg.mla.q_lora_rank * 2), t * cfg.mla.q_lora_rank * 2),
                            t,
                            Dtype::BF16,
                            Dtype::BF16,
                            &mut self.s.prefill_ws,
                            st,
                        )
                        .map_err(|e| ModelError::Config(format!("L72 q_a prefill m={t}: {e}")))?;
                } else {
                    ml.q_a.forward(
                        &self.s.norm.subview(off(h * 2), t * h * 2),
                        &mut self.s.q_a.subview(off(cfg.mla.q_lora_rank * 2), t * cfg.mla.q_lora_rank * 2),
                        t,
                        Dtype::BF16,
                        Dtype::BF16,
                        &mut self.s.ws,
                        st,
                    )?;
                }
                let mut qa = self.s.q_a.subview(off(cfg.mla.q_lora_rank * 2), t * cfg.mla.q_lora_rank * 2);
                let q_elems = cfg.mla.heads * cfg.mla.qk_head_dim();
                if prefill && !p1_dense {
                    // P2: rmsnorm(q_a in place) + rotate by q_b.suh into prefill_ws in ONE launch
                    // (rmsnorm_rot_in runs the exact glue::rmsnorm and gb_glm_p2_rotate math), then the P2
                    // GEMM on the rotated input (bit-identical per element to the separate launches)
                    let qa_in = qa.subview(0, t * cfg.mla.q_lora_rank * 2);
                    let mut qa_out = qa.subview(0, t * cfg.mla.q_lora_rank * 2);
                    let mut qb_xh = self.s.prefill_ws.subview(0, t * ml.q_b.k * 2);
                    if gb_cuda::ling_kda_proj_fused::rmsnorm_rot_in(
                        &qa_in,
                        &ml.q_a_norm,
                        &mut qa_out,
                        &ml.q_b.suh,
                        &mut qb_xh,
                        t,
                        ml.q_b.k,
                        eps,
                        st,
                    )
                    .is_ok()
                    {
                        gb_cuda::ling_kda_proj_fused::ling_p2_gemm_prerot(
                            &qb_xh,
                            &mut self.s.q.subview(off(q_elems * 2), t * q_elems * 2),
                            t,
                            Dtype::BF16,
                            ml.q_b.k,
                            ml.q_b.n,
                            ml.q_b.k2,
                            ml.q_b.codebook,
                            &ml.q_b.trellis,
                            &ml.q_b.svh,
                            st,
                        )
                        .map_err(|e| ModelError::Config(format!("L72 q_b prefill m={t}: {e}")))?;
                    } else {
                        unsafe { glue::rmsnorm(qa.ptr(), ml.q_a_norm.ptr(), qa.ptr(), t, cfg.mla.q_lora_rank, eps, st)? };
                        ml.q_b
                            .forward_prefill_auto(
                                &qa,
                                &mut self.s.q.subview(off(q_elems * 2), t * q_elems * 2),
                                t,
                                Dtype::BF16,
                                Dtype::BF16,
                                &mut self.s.prefill_ws,
                                st,
                            )
                            .map_err(|e| ModelError::Config(format!("L72 q_b prefill m={t}: {e}")))?;
                    }
                } else if gb_cuda::ling_kda_proj_fused::rmsnorm_linear_bf16(
                    &mut qa,
                    &ml.q_a_norm,
                    &ml.q_b.trellis,
                    &ml.q_b.suh,
                    &ml.q_b.svh,
                    &mut self.s.q.subview(off(q_elems * 2), t * q_elems * 2),
                    t,
                    ml.q_b.k,
                    ml.q_b.n,
                    ml.q_b.k2,
                    ml.q_b.codebook as i32,
                    eps,
                    &mut self.s.ws,
                    st,
                )
                .is_err()
                {
                    // the fused rmsnorm + q_b projection does not cover this shape: separate launches
                    unsafe { glue::rmsnorm(qa.ptr(), ml.q_a_norm.ptr(), qa.ptr(), t, cfg.mla.q_lora_rank, eps, st)? };
                    ml.q_b.forward(
                        &qa,
                        &mut self.s.q.subview(off(q_elems * 2), t * q_elems * 2),
                        t,
                        Dtype::BF16,
                        Dtype::BF16,
                        &mut self.s.ws,
                        st,
                    )?;
                }
                let q = self.s.q.subview(off(q_elems * 2), t * q_elems * 2);
                let x = self.s.norm.subview(off(h * 2), t * h * 2);
                let ri = self.tail_row_index(rows, t);
                // The attention's key range. The query's rope was applied with its absolute position (`ri`); the keys carry theirs from
                // their write, so a window that starts at page `first_block` is the same attention over fewer keys: logical page 0 is
                // that page, and the row's position counts from there (kept in the first word of `seq_start`, which the next trunk batch
                // overwrites). Only a window of one query row is supported: that is every L72 attention (the chain row, an absorb's last).
                let (first_block, win_ri) = match window {
                    Some((first_block, vpos)) => {
                        assert_eq!(t, 1, "an L72 window is for a single query row");
                        let mut p = self.s.seq_start.subview(0, 4);
                        put(&mut self.stage, &mut p, &[vpos], st, "L72 window position")?;
                        (first_block, Some(RowIndex { slot: ri.slot.subview(0, 4), pos: p, rows: 1 }))
                    }
                    None => (0, None),
                };
                let attn_ri = win_ri.as_ref().unwrap_or(&ri);
                match self.mla_contract {
                    PrefillContract::P1 => {
                        mla_gated::q_prep(&q, &ml.w, &self.cos, &self.sin, &ri, &mut self.s.q_lat, &mut self.s.q_rot, t, st)
                            .map_err(|e| ModelError::Config(format!("L72 q_prep m={t}: {e}")))?;
                        mla_gated::attend_from(
                            &self.s.q_lat,
                            &self.s.q_rot,
                            &self.mla_caches[ord],
                            &self.table,
                            first_block,
                            attn_ri,
                            &mut self.s.o_lat,
                            t,
                            st,
                        )
                        .map_err(|e| ModelError::Config(format!("L72 attend m={t}: {e}")))?;
                        mla_gated::expand(&self.s.o_lat, &ml.w, &mut self.s.mla_out, t, st)
                            .map_err(|e| ModelError::Config(format!("L72 expand m={t}: {e}")))?;
                        mla_gated::gate_apply(&x, &ml.w, &self.s.mla_out, &mut self.s.gated, t, st)
                            .map_err(|e| ModelError::Config(format!("L72 gate m={t}: {e}")))?;
                    }
                    PrefillContract::P2 => {
                        // tensor-core L72: absorb + rope as separate launches, then the paired prefill
                        // attention (row-invariant, so blocks/verify/decode all use it), then the fused
                        // expand + output gate — the same P2 kernels the trunk uses under `mla_attention_rows`
                        ling_mla_attn::absorb(&q, &ml.w, &mut self.s.q_lat, t, st)
                            .map_err(|e| ModelError::Config(format!("L72 absorb tc m={t}: {e}")))?;
                        mla_gated::q_rope_only(&q, &self.cos, &self.sin, &ri, &mut self.s.q_rot, t, st)
                            .map_err(|e| ModelError::Config(format!("L72 q_rope_only m={t}: {e}")))?;
                        if prefill && t >= 2 {
                            // a prompt-row call: never windowed (an L72 window is for a single query row), so `first_block` is 0 here
                            ling_mla_attn::attend_from(
                                &self.s.q_lat,
                                &self.s.q_rot,
                                &self.mla_caches[ord],
                                &self.table,
                                first_block,
                                attn_ri,
                                &mut self.s.o_lat,
                                t,
                                true,
                                st,
                            )
                        } else {
                            // a one-row call (a chain step, or the last row of an absorb): the decode kernel, split over the keys of the window
                            ling_mla_attn::attend_split_from(
                                &self.s.q_lat,
                                &self.s.q_rot,
                                &self.mla_caches[ord],
                                &self.table,
                                first_block,
                                attn_ri,
                                &mut self.s.o_lat,
                                &mut self.s.attn_ws,
                                t,
                                st,
                            )
                        }
                        .map_err(|e| ModelError::Config(format!("L72 attend tc m={t}: {e}")))?;
                        ling_mla_attn::expand_gated(&self.s.o_lat, &x, &ml.w, &mut self.s.gated, t, None, None, st)
                            .map_err(|e| ModelError::Config(format!("L72 expand_gated tc m={t}: {e}")))?;
                    }
                }
                if prefill && !p1_dense {
                    ml.dense.forward_prefill_auto(
                        &self.s.gated,
                        &mut self.s.a,
                        t,
                        Dtype::BF16,
                        Dtype::BF16,
                        &mut self.s.prefill_ws,
                        st,
                    )?;
                    let hid = self.s.hidden.subview(off(h * 2), t * h * 2);
                    let nrm = self.s.norm.subview(off(h * 2), t * h * 2);
                    unsafe {
                        glue::add_rmsnorm(
                            hid.ptr().cast_const(),
                            self.s.a.ptr().cast_const(),
                            layer.post_norm.ptr().cast_const(),
                            hid.ptr(),
                            nrm.ptr(),
                            t,
                            h,
                            eps,
                            st,
                        )?;
                    }
                } else if prefill {
                    // P1 dense prefill: t may exceed the decode ROWS, so keep the original projector
                    // + residual add + post-norm (the fused kernel is for decode/verify windows only)
                    ml.dense.forward(&self.s.gated, &mut self.s.a, t, Dtype::BF16, Dtype::BF16, &mut self.s.ws, st)?;
                    let hid = self.s.hidden.subview(off(h * 2), t * h * 2);
                    let nrm = self.s.norm.subview(off(h * 2), t * h * 2);
                    unsafe {
                        glue::add_rmsnorm(
                            hid.ptr().cast_const(),
                            self.s.a.ptr().cast_const(),
                            layer.post_norm.ptr().cast_const(),
                            hid.ptr(),
                            nrm.ptr(),
                            t,
                            h,
                            eps,
                            st,
                        )?;
                    }
                } else {
                    // fused L72 dense decode projection + residual add + post-norm in one launch per stage
                    // (bit-identical per element to ml.dense.forward + glue::add_rmsnorm; ling_kda_proj_fused)
                    let hid = self.s.hidden.subview(off(h * 2), t * h * 2);
                    let nrm = self.s.norm.subview(off(h * 2), t * h * 2);
                    let mut hid_out = self.s.hidden.subview(off(h * 2), t * h * 2);
                    let mut nrm_out = self.s.norm.subview(off(h * 2), t * h * 2);
                    let lin_cnt = self.s.prerot_ws.subview(3 * ROWS * h * 2 + 512, 132);
                    gb_cuda::ling_kda_proj_fused::linear_add_rmsnorm(
                        &self.s.gated,
                        Dtype::BF16,
                        &ml.dense.trellis,
                        &ml.dense.suh,
                        &ml.dense.svh,
                        &hid,
                        &layer.post_norm,
                        &mut hid_out,
                        &mut nrm_out,
                        t,
                        ml.dense.k,
                        ml.dense.n,
                        ml.dense.k2,
                        ml.dense.codebook as i32,
                        h,
                        eps,
                        &mut self.s.ws,
                        false,
                        &gb_cuda::ling_kda_proj_fused::Prerot::default(),
                        Some(&lin_cnt),
                        st,
                    )?;
                }
            }
            MixerDevice::Kda(_) => return Err(ModelError::Config("L72 cannot be KDA".into())),
        }
        // the MoE runs on the same rows the mixer's tail produced (the whole call, or the last row for an absorb)
        let t = match tail {
            Tail::All => rows,
            Tail::Last => 1,
        };
        let off = (rows - t) * h * 2;
        match &layer.ffn {
            FfnDevice::Moe(moe) => {
                let router = &moe.router;
                let x = self.s.norm.subview(off, t * h * 2);
                let mut y = self.s.moe_logits.subview(0, t * cfg.ffn.routed_experts * 4);
                let shape = fmoe::Shape {
                    t,
                    ne: cfg.ffn.routed_experts,
                    top_k: cfg.ffn.experts_per_token,
                    h,
                    i: cfg.ffn.moe_intermediate,
                    kc: BAILING_KC,
                    max_tokens: if prefill { self.prefill_rows } else { ROWS },
                };
                let gr = Some(fmoe::GroupRoute {
                    expert_bias: Some(&moe.expert_bias),
                    n_group: cfg.ffn.n_group,
                    topk_group: cfg.ffn.topk_group,
                    scale: cfg.ffn.routed_scaling_factor as f32,
                });
                let limits = fmoe::SwigluLimit { routed: cfg.routed_swiglu_limit(71), shared: cfg.shared_swiglu_limit(71) };
                if prefill {
                    // the MTP layer keeps the scalar router at every row count, so its logits do not depend on how many rows share the call
                    gb_cuda::ling_kda_proj_fused::bf16_linear_f32(&x, router, &mut y, t, h, cfg.ffn.routed_experts, &self.stream)
                        .map_err(|e| ModelError::Config(format!("moe router L72 m={t}: {e}")))?;
                    // the fused prefill MoE (scan+scatter fusion; no norm epilogue — the FFN residual is a plain add
                    // followed by `final_layernorm` in place, which cannot fuse with an add). Fall back to the
                    // separate launches when the fused path is unavailable.
                    match ling_fmoe::prefill_with_epilogue(
                        &x,
                        &self.s.moe_logits,
                        fmoe::LogitsDtype::F32,
                        &self.s.moe_shared_gate,
                        &moe.tables(),
                        &mut self.s.moe_out,
                        shape,
                        gr,
                        limits,
                        &mut self.s.moe_pws,
                        st,
                        None,
                    ) {
                        Ok(()) => {}
                        Err(_) => {
                            fmoe::prefill(
                                &x,
                                &self.s.moe_logits,
                                fmoe::LogitsDtype::F32,
                                &self.s.moe_shared_gate,
                                &moe.tables(),
                                &mut self.s.moe_out,
                                shape,
                                gr,
                                limits,
                                &mut self.s.moe_pws,
                                st,
                            )?;
                        }
                    }
                } else {
                    ling_fmoe::forward_with_router(
                        &x,
                        router,
                        &mut y,
                        fmoe::LogitsDtype::F32,
                        &self.s.moe_shared_gate,
                        &moe.tables(),
                        &mut self.s.moe_out,
                        shape,
                        gr,
                        limits,
                        &mut self.s.moe_ws,
                        st,
                        None,
                    )?;
                }
                // the FFN residual is a plain add: the next op is `final_layernorm` in place on `hidden` (`mtp_layer_forward`),
                // which cannot fuse with an add (its input and output are the same buffer)
                let hid = self.s.hidden.subview(off, t * h * 2);
                unsafe { glue::add(hid.ptr(), self.s.moe_out.ptr(), hid.ptr(), t * h, st)? };
            }
            FfnDevice::Dense(_) => return Err(ModelError::Config("L72 must be MoE".into())),
        }
        Ok(())
    }
}

#[cfg(test)]
mod l72_window_tests {
    use super::*;

    #[test]
    fn an_absorb_that_skips_no_row_leaves_the_window_alone() {
        // a short prompt: every row of it is in the ring
        assert_eq!(absorb_start(0, 5_000), (0, None));
        assert_eq!(absorb_start(0, CAP_TAPS - 1), (0, None), "8,191 rows are exactly the ring");
        // a draft call after decode: the backlog is newer than the ring's oldest row
        assert_eq!(absorb_start(20_000, 20_050), (20_000, None));
        // `absorbed` past the limit (a stale count after a rollback) is clamped, not trusted
        assert_eq!(absorb_start(9_000, 8_000), (8_000, None));
    }

    #[test]
    fn a_prompt_past_the_ring_starts_its_entries_at_the_oldest_tapped_row() {
        // 9,000 rows, nothing absorbed: the first 809 have no tap, so no entry; the entries are 809..9,000
        assert_eq!(absorb_start(0, 9_000), (9_000 - (CAP_TAPS - 1), Some(9_000 - (CAP_TAPS - 1))));
        assert_eq!(absorb_start(0, 9_000).0, 809);
        // 30K prompt
        assert_eq!(absorb_start(0, 30_000), (21_809, Some(21_809)));
        // a prefix hit at 4,096 (absorbed 4,095) and a 30K prompt: the run 4,095.. is broken, the window moves up to the ring's start
        assert_eq!(absorb_start(4_095, 30_000), (21_809, Some(21_809)));
        // a hit close enough that the ring still holds the cut: contiguous, nothing skipped
        assert_eq!(absorb_start(25_000, 30_000), (25_000, None));
        // the ring boundary itself: absorbed == first tapped row skips nothing
        assert_eq!(absorb_start(21_809, 30_000), (21_809, None));
    }

    #[test]
    fn the_draft_window_starts_on_a_page_of_written_entries() {
        let page = mla_gated::BLOCK_TOKENS;
        assert_eq!(page, 256);
        // every position has an entry: the whole cache, no window
        assert_eq!(l72_window(0, 5_000), None);
        // entries from 809: the window starts at page 4 (position 1,024), the first boundary at or after 809, and a query at 9,000 sees
        // 9,000 - 1,024 = 7,976 as its position from there
        assert_eq!(l72_window(809, 9_000), Some((4, 7_976)));
        // a valid_from that is itself a boundary starts there
        assert_eq!(l72_window(1_024, 9_000), Some((4, 7_976)));
        assert_eq!(l72_window(1_025, 9_000), Some((5, 9_000 - 1_280)));
        // the window never holds an unwritten entry: its first position is >= valid_from, and less than a page past it
        for vf in [1usize, 255, 256, 257, 809, 4_095, 21_809, 100_000] {
            let pos = vf + CAP_TAPS;
            let (b, p) = l72_window(vf, pos).unwrap_or((0, pos as i32));
            let first = b * page;
            assert!(first >= vf || b == 0, "vf {vf}: window starts at {first}");
            assert!(first < vf + page, "vf {vf}: the window dropped {} written entries", first - vf);
            assert_eq!(first + p as usize, pos, "the window position is the query's counted from the window's first page");
        }
        // a query within a page of valid_from: the window starts at the first boundary, or at the query's own page when that is earlier
        // (never past the query; the drafter's queries are CAP_TAPS - 1 rows past valid_from, so this is a bound, not a case)
        assert_eq!(l72_window(1_000, 1_100), Some((4, 1_100 - 1_024)));
        assert_eq!(l72_window(1_200, 1_250), Some((4, 1_250 - 1_024)));
    }
}
