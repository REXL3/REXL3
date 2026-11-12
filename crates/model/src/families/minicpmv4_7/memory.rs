//! The MiniCPM-V-4.7 memory model: what a token, a sequence slot, a window row and a step's rows cost, evaluated at the serving
//! settings. The arithmetic (all integers, so the planner's page math and admission can never disagree):
//!
//! * **KV per token** only on the 10 full-attention layers: 10 x 2 (K, V) x 2 KV heads x 256 x 2 B = 20,480 B. A 262,144-token
//!   sequence is 5 GiB. With the DSpark draft loaded its 5 layers cache their own K and V in the same pages (the context rows the
//!   taps produce): 5 x 2 x 2 x 256 x 2 B = 10,240 B more, 30,720 B a token.
//! * **State per sequence** (the 30 gated delta net layers): the fp32 delta-rule state 32 value heads x 128 x 128 x 4 B =
//!   2,097,152 B and the bf16 convolution history 3 x 8,192 x 2 B = 49,152 B a layer, 64,389,120 B in all (61.4 MiB).
//! * **Window records per row** (the 30 layers): a window leaves the committed state read-only and writes one record per row
//!   (q and k at 16 key heads, v at 32 value heads, bf16; decay and beta fp32 per value head: 16,640 B), ONE set (E3: the forward
//!   commits a window's accepted rows with `gdn::commit` before the next window runs, as Flash-Next does; the Qwen3.8
//!   `window_owed` design folds that replay into the next window kernel and needs a second set, a performance lead for E6), and the
//!   raw bf16 qkv row the convolution history is rebuilt from (16,384 B): 33,024 B a layer, 990,720 B a row, 63.4 MB for a 64-row
//!   window.
//! * **Scratch**: an allowance per row (the named buffers below, one set reused by every layer) and fixed buffers (the window's
//!   logits are 248,192 wide: the head is padded), until the engine slice's allocation test replaces it with the real list.
//!
//! The draft is a property of the serving run (a `--draft` pack), not of the target pack, so it enters the model as an argument
//! ([`memory_profile`], `MiniCpmMemorySource::with_draft`), not through `MemoryOptions` (shared by every family, whose `mtp`
//! flag means a pack's own MTP head: this checkpoint has none, and a profile asked for one is refused).

use super::config::{padded_vocab, DSparkConfig, MiniCpmConfig};
use crate::memory::MemoryOptions;
use gb_sched::{KvLayerClass, KvLayout, KvPlane, MemoryError, MemoryModel, MemoryProfile, StepShape, VisionMemory};

/// Tokens of a KV page (the paged cache every family shares).
pub const PAGE_TOKENS: u64 = 256;
/// Rows of the widest verify window the engine allocates for (4 streams of 16 rows).
pub const WINDOW_ROWS: u64 = 64;
/// Candidates per row the engine keeps from the LM head (the selector walk's `k`, at most 64).
pub const MAX_CANDIDATES: u64 = 64;
/// What the engine holds back for workspaces allocated after the plan: the EXL3 prefill and decode scratch, the attention
/// partials (16 query heads at 262,144 tokens), the GDN prefill workspace, the grouped-expert prefill workspaces at H 2,048 / I 512
/// and the draft's block workspace. The same blanket every Qwen-family engine holds; the tests check the modelled part stays inside.
pub const ACTIVATION_RESERVE: u64 = 5 << 30;

// ------------------------------------------------------------------------------------------------------- the arithmetic

/// Bytes of one GDN layer's committed state: the fp32 delta-rule state and the bf16 convolution history (`kernel - 1` rows).
pub fn gdn_state_bytes_per_layer(c: &MiniCpmConfig) -> u64 {
    let s = c.lin_value_heads * c.lin_value_dim * c.lin_key_dim * 4;
    let conv = (c.lin_conv_kernel.saturating_sub(1)) * c.conv_channels() * 2;
    (s + conv) as u64
}

/// Bytes of one window row's record set in one GDN layer: q and k (key heads), v (value heads), bf16, and decay and beta (fp32
/// per value head).
pub fn gdn_record_bytes(c: &MiniCpmConfig) -> u64 {
    (2 * (c.lin_key_heads * c.lin_key_dim * 2) + c.lin_value_heads * c.lin_value_dim * 2 + 2 * c.lin_value_heads * 4) as u64
}

/// Bytes one window row keeps in one GDN layer: the record set and the raw qkv row.
pub fn gdn_row_bytes_per_layer(c: &MiniCpmConfig) -> u64 {
    gdn_record_bytes(c) + (c.conv_channels() * 2) as u64
}

/// Bytes of one prefix-cache snapshot slot: the 30 gated delta net layers' state and convolution history of one sequence (the same
/// 64,389,120 B a sequence slot holds; the cached prefix's KV is pages of the pool that the entry references).
pub fn snapshot_bytes(c: &MiniCpmConfig) -> u64 {
    c.linear_layers() as u64 * gdn_state_bytes_per_layer(c)
}

/// The target's K and V bytes per token over the full-attention layers (bf16).
pub fn target_kv_bytes_per_token(c: &MiniCpmConfig) -> u64 {
    (c.full_layers() * 2 * c.kv_heads * c.head_dim * 2) as u64
}

// ------------------------------------------------------------------------------------------------------- the allocations

/// How an allocation scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocKind {
    /// Per sequence slot, whatever its length.
    PerSlot,
    /// Per KV page of [`PAGE_TOKENS`] tokens.
    PerPage,
    /// Per row of the verify window.
    PerWindowRow,
    /// Per prefix-cache snapshot slot (E6b): the GDN state and conv history of one sequence, kept outside the slots.
    PerSnapshot,
}

/// One named allocation (or group of equal ones): bytes per unit of its kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocItem {
    pub name: &'static str,
    pub kind: AllocKind,
    pub bytes: u64,
}

/// What the engine allocates, as arithmetic from the shapes (not from the memory model's methods): the list the engine slice's
/// GPU test compares item by item with the backend's real buffers, and that the CPU test compares with the plan at 1 to 8 slots.
pub fn allocations(c: &MiniCpmConfig, draft: Option<&DSparkConfig>) -> Vec<AllocItem> {
    let gdn = c.linear_layers() as u64;
    let kv_page = |layers: usize, kv_heads: usize, head_dim: usize| PAGE_TOKENS * (layers * kv_heads * head_dim * 2 * 2) as u64;
    let mut v = vec![
        AllocItem {
            name: "gdn_state",
            kind: AllocKind::PerSlot,
            bytes: gdn * (c.lin_value_heads as u64 * c.lin_value_dim as u64 * c.lin_key_dim as u64 * 4),
        },
        AllocItem {
            name: "gdn_conv_history",
            kind: AllocKind::PerSlot,
            bytes: gdn * ((c.lin_conv_kernel as u64).saturating_sub(1) * c.conv_channels() as u64 * 2),
        },
        AllocItem { name: "kv_target", kind: AllocKind::PerPage, bytes: kv_page(c.full_layers(), c.kv_heads, c.head_dim) },
        AllocItem { name: "gdn_records", kind: AllocKind::PerWindowRow, bytes: gdn * gdn_record_bytes(c) },
        AllocItem { name: "gdn_raw_qkv", kind: AllocKind::PerWindowRow, bytes: gdn * (c.conv_channels() as u64 * 2) },
        // E6b: a prefix-cache snapshot is one sequence's recurrent state (the KV of a cached prefix is shared pages of the pool,
        // priced by `kv_target`); the count is the plan's `cache.snapshot_slots`
        AllocItem { name: "prefix_snapshot", kind: AllocKind::PerSnapshot, bytes: gdn * gdn_state_bytes_per_layer(c) },
    ];
    if let Some(d) = draft {
        v.push(AllocItem { name: "kv_draft", kind: AllocKind::PerPage, bytes: kv_page(d.layers, d.kv_heads, d.head_dim) });
        // one private page a slot, after the pool's pages (draft page id `kv_pages + slot`): the draft's block writes K/V at the 15
        // positions past a stream's committed length whether or not the target's pool holds a page for them, and a draft that took
        // pages from that pool could starve another request (E4b, `spec.rs`)
        v.push(AllocItem {
            name: "kv_draft_overflow",
            kind: AllocKind::PerSlot,
            bytes: kv_page(d.layers, d.kv_heads, d.head_dim),
        });
    }
    v
}

/// Bytes the draft adds to one page of the pool (its K and V planes of 256 tokens): what the prefix cache's budget page holds beyond the
/// target's.
pub fn draft_page_bytes(d: &DSparkConfig) -> u64 {
    PAGE_TOKENS * d.kv_bytes_per_token()
}

/// Bytes of the draft's K/V pool for `pages` target pages and `slots` slots: the page of each plus each slot's overflow page.
pub fn draft_pool_bytes(c: &MiniCpmConfig, d: &DSparkConfig, pages: usize, slots: usize) -> u64 {
    let items = allocations(c, Some(d));
    let page = items.iter().find(|i| i.name == "kv_draft").map_or(0, |i| i.bytes);
    let overflow = items.iter().find(|i| i.name == "kv_draft_overflow").map_or(0, |i| i.bytes);
    page * pages as u64 + overflow * slots as u64
}

/// Sum of the items of one kind.
pub fn sum_of(items: &[AllocItem], kind: AllocKind) -> u64 {
    items.iter().filter(|i| i.kind == kind).map(|i| i.bytes).sum()
}

// ----------------------------------------------------------------------------------------------------------- the model

/// The MiniCPM-V-4.7 memory model: the target, and the DSpark draft when one is loaded.
pub struct MiniCpmMemory<'a> {
    pub cfg: &'a MiniCpmConfig,
    pub draft: Option<&'a DSparkConfig>,
    pub opts: MemoryOptions,
}

/// Bytes a row of a prefill chunk holds in FrostMoE's prefill workspace (`fmoe::prefill_workspace`): per (row, expert) entry the expert
/// id, weight, sorted token, sorted expert and sorted position (5 words), the down pass's input (fp16 `I`), the two rotated inputs
/// (fp16 `2 H`) and the fp32 output (`H`). The tile tables (a few KB, 256-byte aligned parts) are inside the engine reserve's rounding.
pub fn grouped_prefill_bytes_per_row(c: &MiniCpmConfig) -> u64 {
    let (h, k1) = (c.hidden as u64, c.experts_per_tok as u64 + 1);
    k1 * (5 * 4 + c.moe_intermediate as u64 * 2 + 2 * h * 2 + h * 4)
}

impl MiniCpmMemory<'_> {
    /// Per-row bytes of the activation buffers one step's rows use (one set, reused by every layer), named in the order the
    /// layer runner allocates them. An allowance: the engine slice replaces it with the real list.
    pub fn scratch_bytes_per_row(&self) -> u64 {
        let c = self.cfg;
        let (h, k1) = (c.hidden as u64, c.experts_per_tok as u64 + 1);
        let (q_dim, kv_dim) = ((c.heads * c.head_dim) as u64, (c.kv_heads * c.head_dim) as u64);
        let (z, hv) = ((c.lin_value_heads * c.lin_value_dim) as u64, c.lin_value_heads as u64);
        let qkv = (2 * q_dim).max(c.conv_channels() as u64);
        let residual = h * 4; // fp32 residual stream
        let normed = 2 * h * 2; // the mixer's input and the post-attention norm
        let mixer_out = q_dim.max(z).max(h) * 2; // attention heads or GDN values
        let proj = qkv * 2; // q with its gate, or the GDN qkv
        let gdn_z = z * 2;
        let gdn_ab = 2 * hv * 2; // a and b, fp16
        let gdn_gb = 2 * hv * 4; // decay and beta, fp32
        let kv = 3 * kv_dim * 2; // k, v and the normed, roped k
        let q = 2 * q_dim * 2; // the normed, roped q and the gate
        let router = c.experts as u64 * 2; // router logits
        let picks = 2 * k1 * 4; // ids and weights, routed plus shared
        let experts = k1 * c.moe_intermediate as u64 * 2; // expert activations (CoopMK's)
        let grouped = grouped_prefill_bytes_per_row(c);
        let shared = 3 * c.shared_intermediate as u64 * 2 + h * 2 + 2; // gate, up, product, output and the gate scalar
        let index = 4 * 4; // ids, positions, row slot and row position (i32)
        let taps = if self.draft.is_some() { self.draft_taps() * h * 2 } else { 0 };
        residual
            + normed
            + mixer_out
            + proj
            + gdn_z
            + gdn_ab
            + gdn_gb
            + kv
            + q
            + router
            + picks
            + experts
            + grouped
            + shared
            + index
            + taps
    }

    fn draft_taps(&self) -> u64 {
        self.draft.map_or(0, |d| d.target_layer_ids.len() as u64)
    }

    /// Buffers that do not scale with the step's rows: the window's logits (bf16, at the padded head width) and candidates, the
    /// per-slot sequence arrays, and with a draft its block logits and the selector walk's candidate arrays.
    pub fn fixed_bytes(&self) -> u64 {
        let w = self.opts.window_rows as u64;
        let vocab = padded_vocab(self.cfg.vocab) as u64;
        let slots_arrays = 3 * 4 * 64;
        let target = w * vocab * 2 + 2 * w * MAX_CANDIDATES * 4 + slots_arrays;
        let draft = if self.draft.is_some() { w * vocab * 2 + w * MAX_CANDIDATES * 12 } else { 0 };
        target + draft
    }
}

impl MemoryModel for MiniCpmMemory<'_> {
    fn family(&self) -> &str {
        "minicpmv4_7"
    }

    fn kv_layout(&self) -> KvLayout {
        let c = self.cfg;
        let mut layout = KvLayout::new().with(KvLayerClass::gqa(c.full_layers() as u64, c.kv_heads as u64, c.head_dim as u64, 2));
        if let Some(d) = self.draft {
            let kv = (d.kv_heads * d.head_dim) as u64;
            layout = layout.with(KvLayerClass {
                name: "dspark_gqa",
                layers: d.layers as u64,
                planes: vec![KvPlane::per_token("k", kv, 2), KvPlane::per_token("v", kv, 2)],
            });
        }
        layout
    }

    fn recurrent_layers(&self) -> u64 {
        self.cfg.linear_layers() as u64
    }

    fn state_bytes_per_seq(&self) -> u64 {
        self.cfg.linear_layers() as u64 * gdn_state_bytes_per_layer(self.cfg)
    }

    fn window_record_bytes_per_row(&self) -> u64 {
        self.cfg.linear_layers() as u64 * gdn_row_bytes_per_layer(self.cfg)
    }

    fn scratch_bytes(&self, rows: u64) -> u64 {
        rows.saturating_mul(self.scratch_bytes_per_row()).saturating_add(self.fixed_bytes())
    }

    fn activation_reserve(&self) -> u64 {
        ACTIVATION_RESERVE
    }

    /// The draft keeps its context K and V in planes of the shared pages, and one private overflow page a slot for the block's rows
    /// that fall past the target's last page (see `allocations`).
    fn drafter_state_bytes_per_slot(&self) -> u64 {
        self.draft.map_or(0, |d| PAGE_TOKENS * d.kv_bytes_per_token())
    }

    /// The tower (E5b), when the server loads it (`opts.vision`): its weights as `mcv_vision::Weights` holds them (fp16 matrices with the
    /// MLP and the patch matrix zero padded, fp32 vectors and position table: `vision::weight_bytes`), one workspace for the launch
    /// size (`REXL3_MINICPM_VISION_MAX_PATCHES`, `vision::workspace_bytes`) and the bf16 rows of one image at the server's token limit.
    /// The weights are loaded after the plan is made, so the boot subtracts them from what the planner is given; the scratch and the
    /// feature buffers of the images in flight are the plan's `vision_reserve_bytes`.
    fn vision(&self) -> Option<VisionMemory> {
        let v = self.opts.vision?;
        Some(VisionMemory {
            weight_bytes: Some(super::vision::weight_bytes()),
            encoder_scratch_bytes: super::vision::workspace_bytes(super::vision::max_patches_setting()),
            feature_bytes_per_image: super::vision::feature_bytes(v.max_image_tokens),
        })
    }
}

/// The validated profile of `cfg` at `opts`, with the DSpark draft's cache when `draft` is given. A config that prices a token at
/// zero bytes is an error here, before the planner or admission exist; `opts.mtp` is refused (there is no MTP head to draft with).
pub fn memory_profile(
    cfg: &MiniCpmConfig,
    draft: Option<&DSparkConfig>,
    opts: &MemoryOptions,
) -> Result<MemoryProfile, MemoryError> {
    if opts.mtp {
        return Err(MemoryError::Inconsistent(format!(
            "minicpmv4_7: MTP speculation was asked for but this checkpoint ships no MTP head (the config declares {} layer(s) and \
             holds none of its tensors); the drafters are dspark and copy",
            cfg.mtp_declared
        )));
    }
    if let Some(d) = draft {
        d.check_pairing(cfg).map_err(MemoryError::Inconsistent)?;
    }
    let shape = StepShape {
        page_tokens: PAGE_TOKENS,
        window_rows: opts.window_rows as u64,
        scratch_rows: (opts.prefill_rows as u64).max(opts.window_rows as u64),
    };
    MemoryProfile::new(&MiniCpmMemory { cfg, draft, opts: *opts }, shape)
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::{draft, real};
    use super::*;
    use crate::plan::plan_memory;

    const GIB: u64 = 1 << 30;

    fn opts() -> MemoryOptions {
        MemoryOptions::serving_defaults()
    }

    #[test]
    fn the_cache_is_the_ten_attention_layers_and_the_draft_adds_its_five() {
        let c = real();
        let p = memory_profile(&c, None, &opts()).unwrap();
        // 10 layers x 2 (K, V) x 2 kv heads x 256 x 2 B
        assert_eq!(p.kv_bytes_per_token(), 10 * 2 * 2 * 256 * 2);
        assert_eq!(p.kv_bytes_per_token(), 20_480);
        assert_eq!(p.kv_page_bytes(), 256 * 20_480);
        assert_eq!(target_kv_bytes_per_token(&c), 20_480);
        // a 262,144-token sequence is 5 GiB
        assert_eq!(p.kv_bytes_per_token() * 262_144, 5 * GIB);
        // with the draft: 5 layers x 2 x 2 x 256 x 2 B more, in the same pages
        let d = draft();
        let pd = memory_profile(&c, Some(&d), &opts()).unwrap();
        assert_eq!(pd.kv_bytes_per_token(), 20_480 + 5 * 2 * 2 * 256 * 2);
        assert_eq!(pd.kv_bytes_per_token(), 30_720);
        assert_eq!(pd.kv_bytes_per_token() - p.kv_bytes_per_token(), d.kv_bytes_per_token());
        assert_eq!(pd.kv_page_bytes(), 256 * 30_720);
        // the draft's overflow page is the only thing it holds per slot outside the pool's pages
        assert_eq!(p.drafter_state_bytes_per_slot(), 0);
        assert_eq!(pd.drafter_state_bytes_per_slot(), 256 * 10_240);
        assert_eq!(pd.state_bytes_per_slot(), p.state_bytes_per_slot() + 2_621_440);
    }

    #[test]
    fn state_and_window_records_by_hand() {
        let c = real();
        let p = memory_profile(&c, None, &opts()).unwrap();
        // per GDN layer: S 32 x 128 x 128 x 4 B = 2,097,152 and the conv history 3 x 8,192 x 2 B = 49,152; x 30 layers
        assert_eq!(gdn_state_bytes_per_layer(&c), 2_097_152 + 49_152);
        assert_eq!(p.state_bytes_per_seq(), 30 * (2_097_152 + 49_152));
        assert_eq!(p.state_bytes_per_seq(), 64_389_120);
        assert_eq!(p.recurrent_layers(), 30);
        // per layer and row: q, k 2 x (16 x 128 x 2) + v 32 x 128 x 2 + decay, beta 2 x 32 x 4 = 16,640, one set, plus the raw
        // qkv row 8,192 x 2 = 16,384
        assert_eq!(gdn_record_bytes(&c), 8_192 + 8_192 + 256);
        assert_eq!(gdn_row_bytes_per_layer(&c), 16_640 + 16_384);
        assert_eq!(gdn_row_bytes_per_layer(&c), 33_024);
        assert_eq!(p.window_record_bytes_per_row(), 30 * 33_024);
        assert_eq!(p.window_record_bytes_per_row(), 990_720);
        // a 64-row window
        assert_eq!(p.window_record_bytes(), 64 * 990_720);
        // the slot state in MiB: the prefix-cache snapshot's order of size
        assert_eq!(p.state_bytes_per_seq() * 10 / (1 << 20), 614);
    }

    #[test]
    fn scratch_by_hand_and_the_padded_head() {
        let c = real();
        let m = MiniCpmMemory { cfg: &c, draft: None, opts: opts() };
        // residual 8,192; normed 2 x 4,096; mixer out 8,192; q|gate or GDN qkv 16,384; z 8,192; a, b 128; decay, beta 256;
        // k, v, k' 3,072; q', gate 16,384; router 512; picks 72; expert activations 9 x 512 x 2 = 9,216;
        // shared 3 x 512 x 2 + 4,096 + 2 = 7,170; index 16; the grouped prefill workspace 9 x (20 + 1,024 + 8,192 + 8,192) = 156,852
        let per_row =
            8_192 + 8_192 + 8_192 + 16_384 + 8_192 + 128 + 256 + 3_072 + 16_384 + 512 + 72 + 9_216 + 156_852 + 7_170 + 16;
        assert_eq!(per_row, 242_830);
        assert_eq!(m.scratch_bytes_per_row(), per_row);
        // the window's logits are 248,192 wide (the head is padded), not 248,144
        let w = 64u64;
        let fixed = w * 248_192 * 2 + 2 * w * 64 * 4 + 768;
        assert_eq!(m.fixed_bytes(), fixed);
        let p = memory_profile(&c, None, &MemoryOptions { prefill_rows: 4096, ..opts() }).unwrap();
        assert_eq!(p.scratch_bytes(), 4096 * per_row + fixed);
        // with the draft: five taps of 2,048 bf16 a row, and the block's logits and candidate arrays
        let d = draft();
        let md = MiniCpmMemory { cfg: &c, draft: Some(&d), opts: opts() };
        assert_eq!(md.scratch_bytes_per_row(), per_row + 5 * 2048 * 2);
        assert_eq!(md.fixed_bytes(), fixed + w * 248_192 * 2 + w * 64 * 12);
    }

    #[test]
    fn the_modelled_workspaces_fit_the_reserve_at_the_widest_settings() {
        let c = real();
        let d = draft();
        // an 8,192-row prefill chunk at the full context, with the draft: the scratch and a full window of records
        let o = MemoryOptions { prefill_rows: 8192, window_rows: 64, max_context: 262_144, ..opts() };
        let p = memory_profile(&c, Some(&d), &o).unwrap();
        let modelled = p.scratch_bytes() + p.window_record_bytes();
        assert!(
            modelled < p.engine_reserve_bytes(),
            "{} MiB modelled against {} MiB",
            modelled >> 20,
            p.engine_reserve_bytes() >> 20
        );
        assert_eq!(p.engine_reserve_bytes(), ACTIVATION_RESERVE);
    }

    #[test]
    fn mtp_is_refused_because_the_checkpoint_has_none() {
        let e = memory_profile(&real(), None, &MemoryOptions { mtp: true, ..opts() }).unwrap_err();
        assert!(matches!(e, MemoryError::Inconsistent(_)));
        let s = e.to_string();
        assert!(s.contains("ships no MTP head") && s.contains("declares 1 layer"), "{s}");
    }

    #[test]
    fn a_zero_kv_shape_cannot_reach_admission() {
        let mut v: serde_json::Value = serde_json::from_str(super::super::config::tests::MINICPM_CONFIG).unwrap();
        v["text_config"]["head_dim"] = serde_json::json!(0);
        let c = MiniCpmConfig::from_json(&v).unwrap();
        assert!(c.check_shapes().is_err(), "the shape check refuses it first");
        let e = memory_profile(&c, None, &opts()).unwrap_err();
        assert!(matches!(e, MemoryError::Zero(_)), "{e}");
    }

    #[test]
    fn a_draft_that_does_not_belong_to_the_target_is_refused() {
        let mut v: serde_json::Value = serde_json::from_str(super::super::config::tests::DSPARK_CONFIG).unwrap();
        v["num_target_layers"] = serde_json::json!(48);
        let d = DSparkConfig::from_json(&v).unwrap();
        let e = memory_profile(&real(), Some(&d), &opts()).unwrap_err();
        assert!(e.to_string().contains("num_target_layers 48 against the target's 40 layers"), "{e}");
    }

    /// The plan the planner hands out equals the allocation list at 1 to 8 slots, with and without the draft: the per-slot state, the
    /// page, the window row and the bytes the plan commits are the sums of the named items, and the plan stays inside what it was given.
    /// (The GPU test of the engine slice compares the same list with the backend's real buffers.)
    #[test]
    fn the_plan_equals_the_allocations_at_one_to_eight_slots() {
        let c = real();
        let d = draft();
        for with_draft in [false, true] {
            let draft_cfg = with_draft.then_some(&d);
            let profile = memory_profile(&c, draft_cfg, &opts()).unwrap();
            let items = allocations(&c, draft_cfg);
            let (per_slot, per_page, per_row) =
                (sum_of(&items, AllocKind::PerSlot), sum_of(&items, AllocKind::PerPage), sum_of(&items, AllocKind::PerWindowRow));
            assert_eq!(per_slot, 64_389_120 + if with_draft { 2_621_440 } else { 0 }, "{with_draft}");
            assert_eq!(per_page, 256 * if with_draft { 30_720 } else { 20_480 }, "{with_draft}");
            assert_eq!(per_row, 990_720, "{with_draft}");
            assert_eq!(per_slot, profile.state_bytes_per_slot());
            assert_eq!(per_page, profile.kv_page_bytes());
            assert_eq!(per_row, profile.window_record_bytes_per_row());
            assert_eq!(per_row * 64, profile.window_record_bytes());
            for slots in 1..=8usize {
                let free = 90 * GIB;
                let floor = 6 * GIB;
                let plan = plan_memory(&profile, free, floor, slots, 262_144, 0).unwrap();
                assert_eq!(plan.block_bytes, per_page);
                assert_eq!(plan.state_bytes_per_slot, per_slot);
                // what the engine allocates beyond the weights: the slots' state and the KV pool's pages, plus its reserve
                let allocated = slots as u64 * per_slot + plan.kv_blocks as u64 * per_page;
                assert_eq!(plan.planned_bytes(slots), allocated + plan.engine_reserve_bytes);
                assert!(
                    plan.planned_bytes(slots) <= free - floor,
                    "{slots} slots: {} of {}",
                    plan.planned_bytes(slots),
                    free - floor
                );
                // the pool holds the whole context of one stream and a page of slack per slot
                assert_eq!(plan.kv_blocks, 262_144 / 256 + slots, "{slots} slots");
                // admission sees exactly what the pool holds: KV for its tokens and the slots' state
                assert_eq!(plan.kv_tokens, 262_144);
                assert_eq!(plan.admit_bytes, plan.kv_tokens * profile.kv_bytes_per_token() + slots as u64 * per_slot);
                let mm = gb_sched::ModelMemory::from_profile(&profile, 0, 0);
                let adm = plan.admission(mm, slots, 262_144).unwrap();
                assert_eq!(adm.request_limit(), Some(262_144));
            }
        }
    }

    #[test]
    fn the_pool_shrinks_by_the_draft_pages_at_equal_memory() {
        // a small box: the same free bytes buy fewer tokens when the draft's planes ride in the pages
        let c = real();
        let d = draft();
        let plain = memory_profile(&c, None, &opts()).unwrap();
        let with = memory_profile(&c, Some(&d), &opts()).unwrap();
        let free = 12 * GIB;
        let a = plan_memory(&plain, free, 2 * GIB, 4, u64::MAX / 2, 0).unwrap();
        let b = plan_memory(&with, free, 2 * GIB, 4, u64::MAX / 2, 0).unwrap();
        assert!(b.kv_tokens < a.kv_tokens, "{} vs {}", b.kv_tokens, a.kv_tokens);
        // within one page of 2/3 of the plain pool
        let ratio = b.kv_tokens as f64 / a.kv_tokens as f64;
        assert!((ratio - 20_480.0 / 30_720.0).abs() < 0.01, "{ratio}");
    }
}
