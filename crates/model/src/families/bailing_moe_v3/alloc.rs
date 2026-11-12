//! The arithmetic behind the Bailing-3.1 memory model: what a slot holds, what a speculative window keeps per row, what a forward
//! scratches, what a KV page costs and what the backend allocates after the plan. Each item is a device allocation of
//! `BailingModel::new` (or `BailingDevice::build`) under the same name: the planner (`memory.rs`) prices the per-slot, per-row and
//! scratch items, `engine::post_alloc_reserve_bytes` the after-plan ones, and `BailingModel::allocations` lists the real buffers so a
//! GPU test (`tests/ling31_alloc.rs`) can check that the two agree byte for byte.

use super::config::BailingConfig;
use gb_cuda::exl3_linear::{linear_workspace_bytes, prefill_scratch_bytes};
use gb_cuda::{fmoe, glm_glue, kda, mla_gated};

/// Rows of the widest decode/verify window (all sequences together).
pub const WINDOW_ROWS: usize = 64;
/// Default prefill chunk rows (the server default is 4096; this family's plan prices its chunked scratch).
pub const DEFAULT_PREFILL_CHUNK: usize = 512;
pub const MAX_PREFILL_CHUNK: usize = 512;
/// FrostMoE K chunks of the decode MoE at hidden 4096 (32 x 128 blocks; 8 and 16 are the valid counts, 8 stages 4 blocks/chunk).
pub const MOE_KC: usize = 8;
/// Top-k depth of the decode/verify head's candidates (the most a request may ask for).
pub const MAX_CANDIDATES: usize = 64;
/// The trunk `lm_head`'s EXL3 width (`mod.rs` groups it at 6.0 bits).
pub const LM_HEAD_BITS: usize = 6;
/// Committed trunk hidden rows kept per slot for MTP drafting (a device ring per slot).
pub const CAP_TAPS: usize = 8192;

pub fn prefill_chunk_for(rows: usize) -> usize {
    rows.clamp(128, MAX_PREFILL_CHUNK)
}

pub fn max_prefill_chunk() -> usize {
    MAX_PREFILL_CHUNK
}

/// The KDA geometry of the pack (as `BailingModel::new` builds its pools, records and workspace).
pub fn kda_dims(cfg: &BailingConfig) -> kda::Dims {
    kda::Dims { heads: cfg.kda.heads, hidden: cfg.hidden }
}

/// The 128-column strips of the trunk `lm_head` the restricted MTP draft head keeps: the first 512 strips (the tokenizer's merge order
/// puts the frequent tokens first), the numeric block (ids 98,304..103,423: the numbers 0..999, which the first strips do not hold and
/// the answers of math, code and tables are full of) and the last strip (the special tokens).
pub fn draft_head_strips(cfg: &BailingConfig) -> [(usize, usize); 3] {
    let strips_full = cfg.vocab / 128;
    [(0, 512), (768, 810), (strips_full - 1, strips_full)]
}

/// The EXL3 decode workspace a window forward needs: the largest of every linear the forward runs, at `rows` rows.
pub fn exl3_workspace_bytes(cfg: &BailingConfig, rows: usize) -> usize {
    let h = cfg.hidden;
    let ch = cfg.kda.channels();
    let m = &cfg.mla;
    let ffn = &cfg.ffn;
    let shapes: [(usize, usize, usize); 10] = [
        (h, 3 * ch, 4),
        (h, ch, 4),
        (ch, h, 5),
        (h, m.q_lora_rank, 5),
        (m.q_lora_rank, m.heads * m.qk_head_dim(), 4),
        (h, m.kv_a_group_n(), 5),
        (m.heads * m.v_head_dim, h, 5),
        (h, ffn.dense_intermediate, 4),
        (ffn.dense_intermediate, h, 4),
        (h, cfg.vocab, LM_HEAD_BITS),
    ];
    shapes.into_iter().map(|(k, n, bits)| linear_workspace_bytes(rows, k, n, bits)).max().unwrap_or(0)
}

/// The EXL3 pinned prefill workspace a prefill chunk of `rows` needs: the largest of every trunk linear the prefill forward runs (the
/// KDA qkv/f/g/o, the MLA q_a/q_b/kv_a/dense, the dense MLP gate/up/down, and the L72 MTP layer's; the head stays on the decode
/// kernels, whose workspace is [`exl3_workspace_bytes`]).
pub fn exl3_prefill_workspace_bytes(cfg: &BailingConfig, rows: usize) -> usize {
    let h = cfg.hidden;
    let ch = cfg.kda.channels();
    let m = &cfg.mla;
    let ffn = &cfg.ffn;
    let shapes: [(usize, usize); 9] = [
        (h, 3 * ch),
        (h, ch),
        (ch, h),
        (h, m.q_lora_rank),
        (m.q_lora_rank, m.heads * m.qk_head_dim()),
        (h, m.kv_a_group_n()),
        (m.heads * m.v_head_dim, h),
        (h, ffn.dense_intermediate),
        (ffn.dense_intermediate, h),
    ];
    shapes.into_iter().map(|(k, n)| prefill_scratch_bytes(rows, k, n)).max().unwrap_or(0)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    pub name: &'static str,
    /// Bytes, per slot / per page / per row / one-time by the item's class.
    pub bytes: u64,
}

/// What the backend is built for: the inputs every item's size depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocParams {
    pub slots: usize,
    /// Pages of the KV pool (`MemoryPlan::kv_blocks`).
    pub kv_blocks: usize,
    /// Rows of a prefill chunk.
    pub prefill_rows: usize,
    /// The MTP head drafts (its latent cache, tap rings and draft head are allocated).
    pub mtp: bool,
    /// The embedding table is read in place from its shard (no device copy: `load::embed_in_place`).
    pub embed_in_place: bool,
}

/// Every device allocation of the backend, by class (each item's bytes are for one unit of its class).
#[derive(Clone, Debug, Default)]
pub struct AllocPlan {
    pub per_slot: Vec<Item>,
    pub per_page: Vec<Item>,
    pub per_window_row: Vec<Item>,
    pub scratch: Vec<Item>,
    pub post_alloc: Vec<Item>,
}

impl AllocPlan {
    pub fn new(cfg: &BailingConfig, p: &AllocParams) -> Self {
        Self {
            per_slot: Self::slot_items(cfg),
            per_page: Self::page_items(cfg, p.mtp),
            per_window_row: Self::window_row_items(cfg),
            scratch: Self::scratch_items(cfg, p.prefill_rows),
            post_alloc: Self::post_alloc_items(cfg, p.slots, p.kv_blocks, p.mtp, p.embed_in_place),
        }
    }

    /// Every item multiplied out for `p`: per slot times the slots, per page times the pages, per window row times
    /// [`WINDOW_ROWS`], the rest once. The names are unique.
    pub fn totals(&self, p: &AllocParams) -> Vec<(&'static str, u64)> {
        let times = |v: &[Item], n: usize| v.iter().map(move |i| (i.name, i.bytes * n as u64)).collect::<Vec<_>>();
        let mut out = times(&self.per_slot, p.slots);
        out.extend(times(&self.per_page, p.kv_blocks));
        out.extend(times(&self.per_window_row, WINDOW_ROWS));
        out.extend(times(&self.scratch, 1));
        out.extend(times(&self.post_alloc, 1));
        out
    }

    pub fn total(&self, p: &AllocParams) -> u64 {
        self.totals(p).iter().map(|(_, b)| b).sum()
    }

    /// One slot's state: the 63 KDA pools (two fp32 state buffers, the conv history of three raw qkv rows and the parity word each,
    /// `kda::Dims::slot_bytes`).
    pub fn slot_items(cfg: &BailingConfig) -> Vec<Item> {
        let kda = cfg.kda_layers().len() as u64;
        vec![Item { name: "kda_pools", bytes: kda * kda_dims(cfg).slot_bytes() as u64 }]
    }

    /// One KV page of the MLA latent caches: the 9 trunk layers', plus the MTP layer's when it drafts (256 tokens of 576 bf16 each).
    pub fn page_items(cfg: &BailingConfig, mtp: bool) -> Vec<Item> {
        let caches = cfg.mla_layers().len() as u64 + u64::from(mtp);
        let block = mla_gated::MlaGatedCache::block_bytes(gb_cuda::exl3_linear::Dtype::BF16) as u64;
        vec![Item { name: "mla_cache", bytes: caches * block }]
    }

    /// One row of a decode/verify window: every KDA layer's record (the raw bf16 qkv row, the fp32 decay and beta).
    pub fn window_row_items(cfg: &BailingConfig) -> Vec<Item> {
        let kda = cfg.kda_layers().len() as u64;
        vec![Item { name: "kda_window_records", bytes: kda * kda_dims(cfg).record_bytes_per_row() as u64 }]
    }

    /// Every activation and workspace buffer of a forward, sized for a prefill chunk of `chunk` rows (decode/verify windows take at
    /// most [`WINDOW_ROWS`] and reuse them; the few window-only buffers are sized for a window).
    pub fn scratch_items(cfg: &BailingConfig, chunk: usize) -> Vec<Item> {
        let r = chunk as u64;
        let rw = WINDOW_ROWS as u64;
        let h = cfg.hidden as u64;
        let ch = cfg.kda.channels() as u64;
        let m = &cfg.mla;
        let ffn = &cfg.ffn;
        let heads = m.heads as u64;
        let e = ffn.routed_experts;
        let topk = ffn.experts_per_token;
        let d = kda_dims(cfg);
        let sc = |name, bytes: u64| Item { name, bytes };
        vec![
            // the shared prefill record (a prefill chunk commits each KDA layer as soon as it ran, so one record serves them all) and
            // the KDA workspace
            sc("kda_prefill_record", r * d.record_bytes_per_row() as u64),
            sc("kda_workspace", r * kda::Workspace::bytes_per_row(d) as u64),
            // ids, row slots and positions, sequence starts [rows + 1] and slots, the commit's keep counts: i32 each
            sc("row_arrays", r * 4 * 3 + (r + 1) * 4 + r * 4 * 2),
            // hidden, the normed layer input and the final-normed rows, bf16 [rows, hidden] each
            sc("hidden", 3 * r * h * 2),
            // the KDA qkv projection: bf16 [rows, 3 * channels] from a prefill chunk, f32 from a window
            sc("kda_qkv", (r * 3 * ch * 2).max(rw * 3 * ch * 4)),
            // the KDA f/g projections, f16 [rows, channels] each
            sc("kda_fg", 2 * r * ch * 2),
            // the MLA q_a [rows, q_lora_rank], q [rows, heads * qk_head_dim], kv [rows, kv_a_group_n], absorbed q and o latents
            // [rows, heads, kv_lora_rank], rotated q rope [rows, heads, rope], the attention and gated outputs [rows, heads * v_head_dim]
            // (the gated buffer also holds the KDA norm output [rows, channels]), 16-bit each
            sc(
                "mla_acts",
                r * 2
                    * (m.q_lora_rank as u64
                        + heads * m.qk_head_dim() as u64
                        + m.kv_a_group_n() as u64
                        + 2 * heads * m.kv_lora_rank as u64
                        + heads * m.qk_rope_head_dim as u64
                        + heads * m.v_head_dim as u64
                        + (heads * m.v_head_dim as u64).max(ch)),
            ),
            // the dense MLP's gate, up and SwiGLU outputs, bf16 [rows, intermediate] each (the mixers' and the down projection's
            // outputs [rows, hidden] reuse them)
            sc("dense_mlp", 3 * r * (ffn.dense_intermediate as u64).max(h) * 2),
            // router logits bf16 [rows, experts], the shared expert's constant gate bf16 [rows], the MoE output bf16 [rows, hidden]
            sc("moe_acts", r * e as u64 * 2 + r * 2 + r * h * 2),
            sc("moe_window_workspace", fmoe::workspace_bytes(WINDOW_ROWS, topk, cfg.hidden, ffn.moe_intermediate, MOE_KC) as u64),
            sc("moe_prefill_workspace", fmoe::prefill_workspace_bytes(chunk, topk, e, cfg.hidden, ffn.moe_intermediate) as u64),
            // the head runs on windows and single rows only: f16 logits [window, vocab] and the top-k ids, values and workspace
            sc("logits", rw * cfg.vocab as u64 * 2),
            sc(
                "topk",
                rw * MAX_CANDIDATES as u64 * 4 * 2
                    + glm_glue::topk_workspace_bytes(WINDOW_ROWS, cfg.vocab, MAX_CANDIDATES) as u64,
            ),
            sc("exl3_workspace", exl3_workspace_bytes(cfg, WINDOW_ROWS) as u64),
            sc("exl3_prefill_workspace", exl3_prefill_workspace_bytes(cfg, chunk) as u64),
            // the MTP layer's token embeddings bf16 [rows, hidden] and its two single-row registers (head input, chain input)
            sc("mtp_acts", r * h * 2 + 2 * h * 2),
        ]
    }

    /// What the backend allocates after the plan that the memory profile does not count: the bf16 embedding table when it is copied to
    /// the device (nothing when the kernels read it in place from its shard), the MTP tap rings and the restricted draft head (its strips
    /// of the trunk head, its logits and decode workspace), the RoPE tables and the block table (sized by the KV pool's `kv_blocks`
    /// pages), the KDA low-rank stand-ins the gate kernel never reads, and the experts' pointer tables (every MoE layer's, the MTP
    /// layer's included: `BailingDevice::build` builds them whenever the pack has the layer).
    pub fn post_alloc_items(cfg: &BailingConfig, slots: usize, kv_blocks: usize, mtp: bool, embed_in_place: bool) -> Vec<Item> {
        let h = cfg.hidden as u64;
        let ch = cfg.kda.channels() as u64;
        let sc = |name, bytes: u64| Item { name, bytes };
        let tokens = (kv_blocks.max(1) * mla_gated::BLOCK_TOKENS) as u64;
        let nex = cfg.ffn.routed_experts as u64 + 1;
        let moe_layers = cfg.moe_layers().len() as u64 + u64::from(cfg.has_mtp());
        let mut v = vec![
            sc("embedding_table", if embed_in_place { 0 } else { cfg.vocab as u64 * h * 2 }),
            sc("rope_tables", tokens * (mla_gated::ROPE_DIM / 2) as u64 * 4 * 2),
            sc("block_table", ((slots.max(1) * kv_blocks.max(1)) as u64 * 4).max(4)),
            sc("kda_dummies", 2 * 128 * h * 2 + 2 * ch * 128 * 2),
            // 9 tables of a u64 pointer per expert (the shared one last) and the K2 table [experts + 1][3] i32, per MoE layer
            sc("expert_tables", moe_layers * (9 * nex * 8 + nex * 3 * 4)),
        ];
        if mtp {
            let strips: u64 = draft_head_strips(cfg).iter().map(|&(lo, hi)| (hi - lo) as u64).sum();
            let cols = strips * 128;
            // `Exl3Weights::select_strips`: the kept strips' trellis, the whole input rotation and the kept output rotation, fp16
            let head = (h / 16) * strips * 64 * LM_HEAD_BITS as u64 * 4 + h * 2 + cols * 2;
            let ws = linear_workspace_bytes(1, cfg.hidden, cols as usize, LM_HEAD_BITS) as u64;
            v.extend([sc("mtp_tap_rings", slots as u64 * CAP_TAPS as u64 * h * 2), sc("mtp_draft_head", head + cols * 2 + ws)]);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL: &str = include_str!("../../../../../testdata/ling31/config.json");

    fn real() -> BailingConfig {
        BailingConfig::from_json(&serde_json::from_str(REAL).unwrap()).unwrap()
    }

    #[test]
    fn slot_bytes_match_the_kda_pools() {
        let cfg = real();
        let items = AllocPlan::slot_items(&cfg);
        let sum: u64 = items.iter().map(|i| i.bytes).sum();
        // per KDA layer: two f32 state buffers (64 heads x 128 x 128), the conv history (3 raw rows of q|k|v, bf16) and the parity word
        let layer = 2 * 64 * 128 * 128 * 4 + 3 * 3 * 64 * 128 * 2 + 4;
        assert_eq!(layer, 8_536_068);
        assert_eq!(sum, 63 * layer);
        assert_eq!(sum, 537_772_284);
        // ~513 MiB of KDA state per slot
        assert_eq!(sum >> 20, 512);
    }

    #[test]
    fn a_window_row_is_every_kda_layers_record() {
        let cfg = real();
        let row: u64 = AllocPlan::window_row_items(&cfg).iter().map(|i| i.bytes).sum();
        // the raw bf16 q|k|v row, the fp32 decay per key channel and the fp32 beta per head
        assert_eq!(row, 63 * (3 * 8192 * 2 + 8192 * 4 + 64 * 4));
        assert_eq!(row * WINDOW_ROWS as u64 >> 20, 315, "the 64-row window records: 316 MiB");
    }

    #[test]
    fn the_prefill_record_is_one_record_of_the_chunk() {
        let cfg = real();
        let get = |v: &[Item], n: &str| v.iter().find(|i| i.name == n).unwrap().bytes;
        let s = AllocPlan::scratch_items(&cfg, 512);
        let per_layer_row = AllocPlan::window_row_items(&cfg)[0].bytes / 63;
        assert_eq!(get(&s, "kda_prefill_record"), 512 * per_layer_row, "one record of 512 rows, not one per layer");
        // the qkv buffer holds a chunk's bf16 projection or a window's f32 one, whichever is larger
        assert_eq!(get(&s, "kda_qkv"), 512 * 3 * 8192 * 2);
        assert_eq!(get(&AllocPlan::scratch_items(&cfg, 128), "kda_qkv"), 64 * 3 * 8192 * 4);
        // the dense MLP's three buffers hold bf16 rows of the 10,240-wide intermediate
        assert_eq!(get(&s, "dense_mlp"), 3 * 512 * 10_240 * 2);
    }

    #[test]
    fn the_after_plan_items_by_hand() {
        let cfg = real();
        let get = |v: &[Item], n: &str| v.iter().find(|i| i.name == n).map(|i| i.bytes);
        let v = AllocPlan::post_alloc_items(&cfg, 1, 129, true, true);
        assert_eq!(get(&v, "embedding_table"), Some(0));
        assert_eq!(get(&AllocPlan::post_alloc_items(&cfg, 1, 129, true, false), "embedding_table"), Some(157_184 * 4096 * 2));
        // 129 pages of 256 positions, 32 cos and 32 sin f32 each
        assert_eq!(get(&v, "rope_tables"), Some(129 * 256 * 32 * 4 * 2));
        assert_eq!(get(&v, "block_table"), Some(129 * 4));
        assert_eq!(get(&v, "mtp_tap_rings"), Some(8192 * 4096 * 2));
        // 68 MoE layers and the MTP layer, 513 entries each
        assert_eq!(get(&v, "expert_tables"), Some(69 * (9 * 513 * 8 + 513 * 12)));
        // the draft head keeps 555 of the 1,228 strips at 6 bits: 256 k-tiles x 555 strips x 64 x 6 x 4 bytes of trellis
        assert_eq!(draft_head_strips(&cfg).iter().map(|&(lo, hi)| hi - lo).sum::<usize>(), 555);
        let head = get(&v, "mtp_draft_head").unwrap();
        assert!(head >= 256 * 555 * 64 * 6 * 4, "{head}");
        assert!(get(&AllocPlan::post_alloc_items(&cfg, 1, 129, false, true), "mtp_draft_head").is_none());
    }

    #[test]
    fn the_totals_have_unique_names() {
        let cfg = real();
        let p = AllocParams { slots: 2, kv_blocks: 129, prefill_rows: 512, mtp: true, embed_in_place: true };
        let t = AllocPlan::new(&cfg, &p).totals(&p);
        let mut names: Vec<_> = t.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "{names:?}");
    }
}
