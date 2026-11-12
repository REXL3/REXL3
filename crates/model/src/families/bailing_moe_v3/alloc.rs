//! The arithmetic behind the Bailing-3.1 memory model: what a slot holds, what a speculative window keeps per row, what a forward
//! scratches, what a KV page costs and what the backend allocates after the plan. Each item is a device allocation of
//! `BailingModel::new` (or `BailingDevice::build`) under the same name: the planner (`memory.rs`) prices the per-slot, per-row and
//! scratch items, `engine::post_alloc_reserve_bytes` the after-plan ones, and `BailingModel::allocations` lists the real buffers so a
//! GPU test (`tests/ling31_alloc.rs`) can check that the two agree byte for byte.

use super::config::BailingConfig;
use gb_cuda::exl3_linear::linear_workspace_bytes;
use gb_cuda::{fmoe, kda, mla_gated};
/// Rows of the widest decode/verify window (all sequences together).
pub const WINDOW_ROWS: usize = 64;
/// The decode kernels' window plus its drafts, and therefore the rows kept in each KDA window record.
pub const MAX_ROWS: usize = 128;
/// Default prefill chunk rows: wide chunks amortize MoE weight reads and the tensor-core EXL3 prefill GEMMs. The ONE shared
/// prefill KDA record (instead of one per KDA layer) is what makes them fit.
pub const DEFAULT_PREFILL_CHUNK: usize = 4096;
pub const MAX_PREFILL_CHUNK: usize = 4096;
/// Smallest prefill chunk (a narrower one only multiplies host round-trips).
pub const MIN_PREFILL_CHUNK: usize = 128;
/// `kda::window_chunked`'s chunk grid (16-row parallel recurrence): prefill chunks snap to it so a prefix cut never splits
/// a chunk, which is also the prefix cache's `cut_align`.
pub const PREFILL_ALIGN: usize = 16;
/// FrostMoE K chunks of the decode MoE at hidden 4096 (32 x 128 blocks; 8 and 16 are the valid counts, 8 stages 4 blocks/chunk).
pub const MOE_KC: usize = 8;
/// Top-k depth of the decode/verify head's candidates (the most a request may ask for).
pub const MAX_CANDIDATES: usize = 64;
/// The trunk `lm_head`'s EXL3 width (`mod.rs` groups it at 6.0 bits).
pub const LM_HEAD_BITS: usize = 6;
/// Committed trunk hidden rows kept per slot for MTP drafting (a device ring per slot).
pub const CAP_TAPS: usize = 8192;
/// Rows of one `kda::Record` kept per KDA layer for decode/verify windows. A window runs at most `MAX_ROWS` rows (the
/// verify window of `WINDOW_ROWS` plus its drafts, one per row): the commit replays accepted prefixes from these records,
/// so each KDA layer must keep its own. Prefill uses the ONE shared prefill record instead.
pub const KDA_WINDOW_RECORD_ROWS: usize = MAX_ROWS;

/// The prefill chunk the family runs at, from the requested rows: clamped into `MIN..=MAX` and snapped down to the KDA
/// chunk grid (`0` means the default).
pub fn prefill_chunk_for(rows: usize) -> usize {
    let rows = if rows == 0 { DEFAULT_PREFILL_CHUNK } else { rows };
    let c = rows.clamp(MIN_PREFILL_CHUNK, MAX_PREFILL_CHUNK);
    (c / PREFILL_ALIGN * PREFILL_ALIGN).max(MIN_PREFILL_CHUNK)
}

pub fn max_prefill_chunk() -> usize {
    MAX_PREFILL_CHUNK
}

/// The KDA geometry of the pack (as `BailingModel::new` builds its pools, records and workspace).
pub fn kda_dims(cfg: &BailingConfig) -> kda::Dims {
    kda::Dims { heads: cfg.kda.heads, hidden: cfg.hidden }
}

/// The 128-column strips of the trunk `lm_head` the restricted MTP draft head keeps: frequent tokens, the numeric block and
/// the final special-token strip.
pub fn draft_head_strips(cfg: &BailingConfig) -> [(usize, usize); 3] {
    let strips_full = cfg.vocab / 128;
    [(0, 512), (768, 810), (strips_full - 1, strips_full)]
}

/// The EXL3 decode workspace a window forward needs: the largest of every linear the forward runs, at `rows` rows,
/// including the fused KDA f/g decode launch (which rotates the input twice, one buffer per projection's `suh`).
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
    let lin = shapes.into_iter().map(|(k, n, bits)| linear_workspace_bytes(rows, k, n, bits)).max().unwrap_or(0);
    // the fused KDA f/g pair and qkv+f+g triple rotate the input once per projection into their own buffers (three xf
    // at most), and the fused MLA q_a/kv_a pair adds the two split-partial buffers of its split-K plan
    let fused = gb_cuda::ling_kda_proj_fused::workspace_bytes(rows, h).max(gb_cuda::ling_kda_proj_fused::pair_workspace_bytes(
        rows,
        h,
        m.q_lora_rank,
        m.kv_a_group_n(),
    ));
    lin.max(fused)
}

/// The direct-to-MMA prefill workspace: the rotated fp16 input of the widest trunk linear. The head stays on the decode
/// kernels, whose workspace is [`exl3_workspace_bytes`]. The LP4 reconstruct-once path's two additional buffers are priced
/// separately in `AllocPlan::scratch_items`.
pub fn exl3_prefill_workspace_bytes(cfg: &BailingConfig, rows: usize) -> usize {
    let h = cfg.hidden;
    let ch = cfg.kda.channels();
    let m = &cfg.mla;
    let ffn = &cfg.ffn;
    let ks = [h, ch, m.q_lora_rank, m.heads * m.v_head_dim, ffn.dense_intermediate];
    ks.into_iter().map(|k| gb_cuda::glm_p2::workspace_bytes(rows, k)).max().unwrap_or(0)
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

    /// Every item multiplied out for `p`: per slot times the slots, per page times the pages, per KDA record row times
    /// [`KDA_WINDOW_RECORD_ROWS`], the rest once. The names are unique.
    pub fn totals(&self, p: &AllocParams) -> Vec<(&'static str, u64)> {
        let times = |v: &[Item], n: usize| v.iter().map(move |i| (i.name, i.bytes * n as u64)).collect::<Vec<_>>();
        let mut out = times(&self.per_slot, p.slots);
        out.extend(times(&self.per_page, p.kv_blocks));
        out.extend(times(&self.per_window_row, KDA_WINDOW_RECORD_ROWS));
        out.extend(times(&self.scratch, 1));
        out.extend(times(&self.post_alloc, 1));
        out
    }

    pub fn total(&self, p: &AllocParams) -> u64 {
        self.totals(p).iter().map(|(_, b)| b).sum()
    }

    /// One slot's state: every KDA layer's two fp32 state buffers, three raw qkv rows of conv history and parity word.
    pub fn slot_items(cfg: &BailingConfig) -> Vec<Item> {
        let kda = cfg.kda_layers().len() as u64;
        vec![Item { name: "kda_pools", bytes: kda * kda_dims(cfg).slot_bytes() as u64 }]
    }

    /// One KV page of the MLA latent caches: the trunk layers', plus the MTP layer's when it drafts.
    pub fn page_items(cfg: &BailingConfig, mtp: bool) -> Vec<Item> {
        let caches = cfg.mla_layers().len() as u64 + u64::from(mtp);
        let block = mla_gated::MlaGatedCache::block_bytes(gb_cuda::exl3_linear::Dtype::BF16) as u64;
        vec![Item { name: "mla_cache", bytes: caches * block }]
    }

    /// One physical row of every KDA layer's decode/verify record. The records have `KDA_WINDOW_RECORD_ROWS` rows each.
    pub fn window_row_items(cfg: &BailingConfig) -> Vec<Item> {
        let kda = cfg.kda_layers().len() as u64;
        vec![Item { name: "kda_window_records", bytes: kda * kda_dims(cfg).record_bytes_per_row() as u64 }]
    }

    /// The EXL3 decode workspace, exposed here for the runtime and its sizing test to share exactly the same shapes.
    pub fn exl3_ws_bytes(cfg: &BailingConfig, rows: usize) -> usize {
        exl3_workspace_bytes(cfg, rows)
    }

    /// The direct-to-MMA prefill workspace, likewise shared with the runtime allocation.
    pub fn prefill_ws_bytes(cfg: &BailingConfig, rows: usize) -> usize {
        exl3_prefill_workspace_bytes(cfg, rows)
    }

    /// The KDA allocation total used by the runtime sizing unit test: all window records, one shared prefill record and one workspace.
    pub fn kda_items(cfg: &BailingConfig, chunk: usize) -> Vec<Item> {
        let layers = cfg.kda_layers().len() as u64;
        let per_row = kda_dims(cfg).record_bytes_per_row() as u64;
        vec![
            Item { name: "kda_window_recs", bytes: layers * KDA_WINDOW_RECORD_ROWS as u64 * per_row },
            Item { name: "kda_prefill_rec", bytes: chunk as u64 * per_row },
            Item { name: "kda_ws", bytes: chunk as u64 * kda_workspace_bytes_per_row(cfg) as u64 },
        ]
    }

    /// Every activation and workspace buffer of a forward, sized for a prefill chunk of `chunk` rows, plus the shared
    /// prefill KDA record and workspace. Decode/verify records are separate `per_window_row` items, not double-counted here.
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
            sc("kda_prefill_record", r * d.record_bytes_per_row() as u64),
            sc("kda_workspace", r * kda::Workspace::bytes_per_row(d) as u64),
            sc("row_arrays", r * 4 * 3 + (r + 1) * 4 + r * 4 * 2),
            sc("hidden", 3 * r * h * 2),
            // the KDA qkv projection: bf16 [rows, 3 * channels] from a P2 prefill chunk, f32 from a window and from a
            // P1 prefill chunk (the row-invariant decode path), so the f32 branch is chunk-sized, not window-sized
            sc("kda_qkv", (r * 3 * ch * 2).max(r * 3 * ch * 4)),
            sc("kda_fg", 2 * r * ch * 2),
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
            // The dense MLP's gate, up and SwiGLU outputs, bf16 [rows, intermediate] each.
            sc("dense_mlp", 3 * r * (ffn.dense_intermediate as u64).max(h) * 2),
            // Ling's router is fp32; the shared gate and MoE output are bf16.
            sc("moe_acts", r * e as u64 * 4 + r * 2 + r * h * 2),
            sc("moe_window_workspace", fmoe::workspace_bytes(WINDOW_ROWS, topk, cfg.hidden, ffn.moe_intermediate, MOE_KC) as u64),
            sc("moe_prefill_workspace", fmoe::prefill_workspace_bytes(chunk, topk, e, cfg.hidden, ffn.moe_intermediate) as u64),
            sc("logits", rw * cfg.vocab as u64 * 2),
            sc(
                "topk",
                rw * MAX_CANDIDATES as u64 * 4 * 2
                    + gb_cuda::ling_kda_proj_fused::topk_workspace_bytes(WINDOW_ROWS, cfg.vocab, MAX_CANDIDATES) as u64,
            ),
            sc("exl3_workspace", exl3_workspace_bytes(cfg, WINDOW_ROWS) as u64),
            // the next sub-layer's fused input rotations (three f16 [rows, hidden] buffers at the decode window size)
            // plus a fixed 1024-byte tail of self-resetting per-block counters (the fused dense gate/up epilogue's
            // per-128-column-col silu-rotation counters and the P2 prefill fused chunk kernel's per-head counters)
            sc("prerot_workspace", 3 * rw * h * 2 + 1024),
            sc("exl3_prefill_workspace", exl3_prefill_workspace_bytes(cfg, chunk) as u64),
            // the split-key decode attention's fp32 partials: fixed size, whatever the rows, the context or the chunk
            sc("mla_split_workspace", gb_cuda::ling_mla_attn::SPLIT_WS_BYTES as u64),
            // e_next and the two single-row registers.
            sc("mtp_acts", r * h * 2 + 2 * h * 2),
        ]
    }

    /// What the backend allocates after the plan that the memory profile does not count.
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
            sc("expert_tables", moe_layers * (9 * nex * 8 + nex * 3 * 4)),
        ];
        if mtp {
            let strips: u64 = draft_head_strips(cfg).iter().map(|&(lo, hi)| (hi - lo) as u64).sum();
            let cols = strips * 128;
            let head = (h / 16) * strips * 64 * LM_HEAD_BITS as u64 * 4 + h * 2 + cols * 2;
            let ws = linear_workspace_bytes(1, cfg.hidden, cols as usize, LM_HEAD_BITS) as u64;
            v.extend([sc("mtp_tap_rings", slots as u64 * CAP_TAPS as u64 * h * 2), sc("mtp_draft_head", head + cols * 2 + ws)]);
        }
        v
    }
}

/// Bytes of one `kda::Workspace` row (`Workspace::bytes_per_row`: q/k/z f32, v/y f16, fa/ga f32 at the low-rank width).
pub fn kda_workspace_bytes_per_row(cfg: &BailingConfig) -> usize {
    kda::Workspace::bytes_per_row(kda_dims(cfg))
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
        let sum: u64 = AllocPlan::slot_items(&cfg).iter().map(|i| i.bytes).sum();
        let layer = 2 * 64 * 128 * 128 * 4 + 3 * 3 * 64 * 128 * 2 + 4;
        assert_eq!(layer, 8_536_068);
        assert_eq!(sum, 63 * layer);
        assert_eq!(sum, 537_772_284);
    }

    #[test]
    fn the_prefill_chunk_clamps_and_snaps_to_the_kda_grid() {
        assert_eq!(prefill_chunk_for(4096), 4096);
        assert_eq!(prefill_chunk_for(2048), 2048);
        assert_eq!(prefill_chunk_for(0), DEFAULT_PREFILL_CHUNK);
        assert_eq!(prefill_chunk_for(1), MIN_PREFILL_CHUNK);
        assert_eq!(prefill_chunk_for(2063), 2048);
        assert_eq!(prefill_chunk_for(1040), 1040);
        assert_eq!(prefill_chunk_for(1033), 1024);
        assert_eq!(prefill_chunk_for(usize::MAX), MAX_PREFILL_CHUNK);
        for r in [128, 129, 511, 512, 2047, 2048, 4096, 8192] {
            assert_eq!(prefill_chunk_for(r) % PREFILL_ALIGN, 0, "{r}");
        }
    }

    #[test]
    fn a_window_row_is_every_kda_layers_record() {
        let cfg = real();
        let row: u64 = AllocPlan::window_row_items(&cfg).iter().map(|i| i.bytes).sum();
        assert_eq!(row, 63 * (3 * 8192 * 2 + 8192 * 4 + 64 * 4));
        assert_eq!(row * KDA_WINDOW_RECORD_ROWS as u64 >> 20, 631);
    }

    #[test]
    fn the_shared_prefill_record_replaces_63_of_them() {
        let cfg = real();
        let per_row = cfg.kda_record_bytes_per_row_per_layer();
        let old = 63 * 512 * per_row;
        let items = AllocPlan::kda_items(&cfg, 2048);
        assert_eq!(items.iter().find(|i| i.name == "kda_window_recs").unwrap().bytes, 63 * 128 * per_row);
        assert_eq!(items.iter().find(|i| i.name == "kda_prefill_rec").unwrap().bytes, 2048 * per_row);
        let saved = old - (63 * 128 + 2048) * per_row;
        assert!(saved > 1 << 30, "the record change reclaims {saved} bytes");
        assert_eq!(items.iter().find(|i| i.name == "kda_ws").unwrap().bytes, 2048 * kda_workspace_bytes_per_row(&cfg) as u64);
    }

    #[test]
    fn the_prefill_scratch_matches_the_optimized_runtime_layout() {
        let cfg = real();
        let get = |v: &[Item], n: &str| v.iter().find(|i| i.name == n).unwrap().bytes;
        let s = AllocPlan::scratch_items(&cfg, 512);
        let per_layer_row = AllocPlan::window_row_items(&cfg)[0].bytes / 63;
        assert_eq!(get(&s, "kda_prefill_record"), 512 * per_layer_row);
        assert_eq!(get(&s, "kda_qkv"), 512 * 3 * 8192 * 4);
        assert_eq!(get(&AllocPlan::scratch_items(&cfg, 128), "kda_qkv"), 128 * 3 * 8192 * 4);
        assert_eq!(get(&s, "dense_mlp"), 3 * 512 * 10_240 * 2);
        assert!(s.iter().find(|i| i.name == "recon_scratch").is_none());
    }

    #[test]
    fn scratch_is_bounded() {
        let cfg = real();
        let sum: u64 = AllocPlan::scratch_items(&cfg, 2048).iter().map(|i| i.bytes).sum();
        assert!(sum < 4 << 30, "scratch {} MiB", sum >> 20);
    }

    #[test]
    fn the_workspace_sizing_fns_match_the_forward_sites() {
        let cfg = real();
        assert_eq!(AllocPlan::exl3_ws_bytes(&cfg, 64), crate::families::bailing_moe_v3::model::exl3_ws_bytes(&cfg, 64));
        assert_eq!(AllocPlan::prefill_ws_bytes(&cfg, 2048), crate::families::bailing_moe_v3::model::prefill_ws_bytes(&cfg, 2048));
    }

    #[test]
    fn the_after_plan_items_by_hand() {
        let cfg = real();
        let get = |v: &[Item], n: &str| v.iter().find(|i| i.name == n).map(|i| i.bytes);
        let v = AllocPlan::post_alloc_items(&cfg, 1, 129, true, true);
        assert_eq!(get(&v, "embedding_table"), Some(0));
        assert_eq!(get(&AllocPlan::post_alloc_items(&cfg, 1, 129, true, false), "embedding_table"), Some(157_184 * 4096 * 2));
        assert_eq!(get(&v, "rope_tables"), Some(129 * 256 * 32 * 4 * 2));
        assert_eq!(get(&v, "block_table"), Some(129 * 4));
        assert_eq!(get(&v, "mtp_tap_rings"), Some(8192 * 4096 * 2));
        assert_eq!(get(&v, "expert_tables"), Some(69 * (9 * 513 * 8 + 513 * 12)));
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
