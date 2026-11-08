//! The Nemotron memory model: prices a slot (recurrent state), a token (KV), a window row (Mamba records) and the scratch for the
//! planner and admission. It is arithmetic over [`OmniConfig`] and [`MemoryOptions`] and a weight-bytes parameter: no device call, so
//! it runs in the probe, in the planner and in tests alike. The coordinator wires [`NemotronMemory`] into the family's `probe()`.
//!
//! What it charges, for the published geometry (40 Mamba2 layers, 8 attention layers, 40 latent-MoE layers, MTP head `*E`):
//!
//! | quantity | formula | bytes |
//! |---|---|---|
//! | state per sequence | 40 x (S f32 `128 x 128 x 64` + conv bf16 `3 x 10,240`) = 40 x (4,194,304 + 61,440) | 170,229,760 |
//! | KV per token | 8 attention layers x (K and V) x 2 heads x 128 x fp16 | 8,192 |
//! | KV per token of the MTP head, when counted | 1 attention layer, by spec | 1,024 |
//! | window record per row | 40 x (record 21,760 + retained f32 xBC 40,960) | 2,508,800 |
//! | window records | 64 rows x the above | 160,563,200 |
//! | logits per row | f32 `[131072]` | 524,288 |
//! | MoE scratch, one instance for all 40 layers | `MoeScratch::bytes(min(rows, 64))` | about 107 MB |
//! | large-row MoE prefill scratch, when the step is wider than a window | `PrefillScratch::bytes(rows)` | about 105 MB at 256 rows |
//! | attention partials | `min(rows, 64)` rows x (16 tree chunks + `ceil(max_context / 8192)` flash splits) x 32 heads x (8 + 128 x 4), plus the row plan (a wider slice runs in passes of 64 rows) | 153 MB at 1M tokens |
//!
//! **The MTP head** (`MemoryOptions::mtp`, from `engine::head_priced`), priced by [`NemotronMemory::mtp_head`] and counted by the profile:
//!
//! | quantity | formula | bytes |
//! |---|---|---|
//! | weights, bf16 | eh_proj + attention + router + latent projections + shared expert + 512 experts + norms + the router's f32 bias | 5,884,651,520 (5,612 MiB) |
//! | KV per token | the `mtp_gqa_nope` class, by spec | 1,024 |
//! | tap store, per slot | `MTP_TAP_ROWS` x hidden f32 = 32 x 4,096 x 4 | 524,288 |
//! | window tap buffer, once | 64 rows x hidden f32 | 1,048,576 |
//! | head activations, once | `max(step rows, 64)` x (2 hidden + 4 hidden) f32 = 64 x (8,192 + 4 x 4,096) x 4 at 64 rows | 6,291,456 |
//! | draft outputs, once | rows x (4 + 4 depths x (4 + 4) + 48) | 5,376 at 64 rows |
//! | restricted sub-head table, once, when configured | `n` x (hidden x 2 + 4) | `n` x 8,196 |
//!
//! The weights are inside the caller's `weight_bytes` (the pack's plan lists the head's tensors): the profile does not add them again.
//! The head's MoE runs in the SHARED trunk [`MoeScratch`] (in calls of at most 64 rows), which is not counted twice. The tap store is a per-slot cost (admission
//! charges it with the slot's state), the rest is scratch (inside the engine's reserve).
//!
//! **The prefix cache** (`prefix.rs`, `docs/NEMOTRON_LONG_CONTEXT.md`) is priced outside the profile, by the plan (`engine::plan_serving_for`),
//! from these terms: one snapshot slot is [`NemotronMemory::snapshot_bytes`] (170,229,760 B of state, 170,246,144 with the head's 16,384-byte
//! carry row); a page is [`NemotronMemory::page_bytes`] (2 MiB, 2.25 MiB with the head's stripe) and a restore copies exactly one; pages a
//! snapshot references are shared, counted once in the cache's page budget however many entries and sequences hold them.
//!
//! The retained f32 xBC is the in_proj's pre-conv output (`nemotron_ssm::commit_conv` replays the accepted prefix from it); the older
//! 1,587,200-byte figure for a row's records predates it and is wrong. A prefill slice commits at `keep = rows` through one record, so
//! it needs no more than a window of the same rows.

use super::backend::{MAX_CANDIDATES, WINDOW_ROWS};
use super::config::OmniConfig;
use super::model::MAX_WINDOW_ROWS;
use super::mtp::MTP_MAX_DEPTH;
use super::spec::MTP_TAP_ROWS;
use super::tp_shard::{head_temp_rows_in, ShardPlan, EXL3_HEAD_ROW_UNIT, HEAD_ROW_UNIT};
use crate::family::{MemorySource, Mixer};
use crate::memory::MemoryOptions;
use gb_cuda::glm_mtp::KeyedRow;
use gb_cuda::nemotron_attn::{KvDtype, BLOCK_TOKENS, CHUNK_KEYS, GROUP_BYTES, MAX_CHUNKS, TREE_CHUNKS_BELOW};
use gb_cuda::nemotron_glue::topk_workspace_bytes;
use gb_cuda::nemotron_moe::MoeScratch;
use gb_cuda::nemotron_moe_prefill::PrefillScratch;
use gb_sched::{KvLayerClass, KvLayout, KvPlane, MemoryError, MemoryModel, MemoryProfile, StepShape, VisionMemory};

/// Tokens of a KV page: the attention kernels' block (`nemotron_attn::BLOCK_TOKENS`), which is the pool's page.
pub const PAGE_TOKENS: u64 = BLOCK_TOKENS as u64;
/// The family's name in errors and logs.
const FAMILY: &str = "nemotron_h_omni";
/// Bytes of one KV element: the cache is fp16 with one rounding point (`tools/attn_gpu_design.md`).
const KV_ELEM_BYTES: u64 = 2;

/// Bytes of decode workspace the EXL3 linears of the trunk's mixers share (`Bufs::ws`: one buffer, the largest any layer asks for).
///
/// `Exl3Weights::workspace_bytes(m)` is `align256(ceil(m/16) x 16 x K x 2)` plus, for a layer that splits K (`sk > 1`),
/// `align256(sk x m x N x 4)` (`csrc/exl3_linear.cu`, `linear_ws_bytes`). The mixer shapes are in_proj K=4096 N=18560 (`sk` 1),
/// out_proj K=8192 N=4096 (`sk` 8), q and o K=4096 N=4096 (`sk` 8) and k and v K=4096 N=256 (`sk` 8). The largest is out_proj:
/// 9,437,184 B at the 64 rows of a window or prefill slice, and 18,874,368 B at 128, the most rows the kernel takes. 32 MiB is
/// above that with room for a shape change; a unit test recomputes the five shapes from the kernel's plan and holds the constant to
/// them. The MoE blocks do not use it: their partial sums live in the shared [`MoeScratch`] and [`PrefillScratch`].
pub const EXL3_WORKSPACE_BYTES: u64 = 32 << 20;

/// What the engine holds back for the workspaces it allocates after the plan and this model does not price row by row: cuBLAS and
/// cuBLASLt handles and workspaces, the decode graph, whatever of the MTP head's scratch [`NemotronMemory::mtp_head`] does not name, and
/// the RADIO tower's activations (counted in [`MemoryModel::vision`] when images are served). The head's tap buffer, activations
/// and sub-head table ARE priced (inside the scratch, so inside this reserve). It is the blanket every family has held back (5 GiB)
/// and it is never lowered; the
/// tests hold the modelled scratch and window records below it up to a 1M-token context (2.5 GB), and a context that outgrows it
/// raises the planner's reserve instead of overcommitting. The first real allocation on node 1 (task I1) should replace the blanket
/// with a measured figure.
pub const ACTIVATION_RESERVE: u64 = 5 << 30;

/// The widest step the model will price. The engine's own cap is far below it (64 rows, `engine::check_options`); this only keeps a
/// garbage option from reaching the arithmetic of `MoeScratch::bytes`, which multiplies the rows by the experts and their widths.
const MAX_PRICED_ROWS: usize = 1 << 20;

/// An allowance, per row, for the top-k workspace of `nemotron_glue::topk_f32`: partial candidates per slice of the vocabulary, up to
/// 128 slices x 64 candidates x (id + value). The kernel's own `topk_workspace_bytes` replaces it when it is larger (the GLM kernel
/// this one follows needs 4 KiB a row at this vocabulary, so the allowance is 16 times that).
const TOPK_ALLOWANCE_PER_ROW: u64 = 128 * MAX_CANDIDATES as u64 * 8;

/// What a step's scratch costs, by buffer group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchBytes {
    /// The named buffers of `model::Bufs` (the logits included), `rows` rows of them.
    pub activations: u64,
    /// The top-k workspace of the LM head's candidate selection.
    pub topk_workspace: u64,
    /// The shared MoE scratches: `MoeScratch::bytes(min(rows, 64))` for the windows and the slices of up to 64 rows, and, when the step is
    /// wider than that, `PrefillScratch::bytes(rows)` for the large-row prefill path; one instance of each for every MoE layer.
    pub moe: u64,
    /// The attention partials (`nemotron_attn::Scratch`): `(m, l)` and the unnormalized `o` per row, chunk and query head.
    pub attn_partials: u64,
    /// The EXL3 linears' shared workspace ([`EXL3_WORKSPACE_BYTES`]).
    pub exl3_workspace: u64,
    /// The MTP head's buffers (window tap buffer, activations, restricted sub-head table): [`MtpHeadBytes::scratch`] when the head is
    /// priced, zero otherwise.
    pub mtp_head: u64,
}

impl ScratchBytes {
    pub fn total(&self) -> u64 {
        [self.activations, self.topk_workspace, self.moe, self.attn_partials, self.exl3_workspace, self.mtp_head]
            .into_iter()
            .fold(0u64, u64::saturating_add)
    }
}

/// What the MTP head holds on the device, by line (see the table in the module documentation). Every line is computed from the config
/// and `MTP_TAP_ROWS`, and the profile counts exactly these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpHeadBytes {
    /// The head's weights in bf16 (the published pack). INSIDE the caller's `weight_bytes`: reported here, never added to it. A 4-bit head
    /// pack's are what its index lists (`LayerPlan::bytes`, which the caller's `weight_bytes` is), about 4 GiB less.
    pub weights: u64,
    /// KV bytes per token of the head's attention layers (the profile's `mtp_gqa_nope` class).
    pub kv_per_token: u64,
    /// One slot's store of pending tapped rows, f32 `[MTP_TAP_ROWS][hidden]`: a per-slot cost.
    pub tap_store_per_slot: u64,
    /// One window's tapped rows, f32 `[64][hidden]` (the trunk's final-normed rows before `commit` moves the kept ones to the stores).
    pub tap_window: u64,
    /// The head's activations over `rows` rows (the model's widest step, `max(prefill_rows, 64)`): `rows x (2 hidden + 4 hidden)` f32 (the
    /// concatenated `[enorm | hnorm]` input and four hidden-wide buffers). The head's MoE uses the shared trunk scratch.
    pub activations: u64,
    /// The draft outputs: per row an id (i32), `MTP_MAX_DEPTH` ids and values (i32, f32) and a keyed-pick record (`KeyedRow`, 48 bytes).
    pub drafts: u64,
    /// The restricted sub-head's table, bf16 `[rows][hidden]`, and its i32 id map; zero for the full head.
    pub sub_head: u64,
}

impl MtpHeadBytes {
    /// What the head adds to the step scratch.
    pub fn scratch(&self) -> u64 {
        self.tap_window.saturating_add(self.activations).saturating_add(self.drafts).saturating_add(self.sub_head)
    }
}

/// Everything one serving shape holds on the device, by line: what the planner sizes and what the first real allocation is compared to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// The resident weights (a parameter: `LayerPlan::bytes(&inventory)`).
    pub weights: u64,
    pub state_per_slot: u64,
    /// `state_per_slot x slots`.
    pub state: u64,
    pub kv_per_token: u64,
    /// `kv_per_token x kv_tokens`, rounded up to whole pages.
    pub kv: u64,
    /// A full window's records.
    pub window_records: u64,
    pub scratch: ScratchBytes,
    /// What the planner holds back: the larger of the blanket and `scratch + window_records`.
    pub engine_reserve: u64,
    /// The MTP head's lines when it is priced. Its weights are inside `weights`, its KV per token inside `kv_per_token`, its tap store
    /// inside `state_per_slot` and its buffers inside `scratch`: this is the breakdown, not an addition to [`Budget::total`].
    pub mtp_head: Option<MtpHeadBytes>,
}

impl Budget {
    /// Weights, slot state, KV and the engine reserve: what the box must hold.
    pub fn total(&self) -> u64 {
        [self.weights, self.state, self.kv, self.engine_reserve].into_iter().fold(0u64, u64::saturating_add)
    }
}

/// The memory model of a Nemotron pack: a [`MemorySource`] for the family's plan, built from the parsed config and the resident weight
/// bytes. The profile does not depend on the weights (the planner takes them out of the free memory before it prices anything); they
/// are carried so that [`NemotronMemory::budget`] can state the whole footprint.
#[derive(Clone, Debug)]
pub struct NemotronMemory {
    cfg: OmniConfig,
    weight_bytes: u64,
    mamba_layers: u64,
    attn_layers: u64,
    mtp_attn_layers: u64,
    /// Rows of the restricted draft head's sub-head table (0: the head runs over the full vocabulary).
    restricted_rows: u64,
    /// One rank of a TP group of this many ranks (0: not a TP rank): the step buffers carry the TP temps
    /// ([`NemotronMemory::tp_scratch_bytes`]).
    tp_ranks: u64,
    /// The KV cache's element type (`--kv-dtype`, `REXL3_KV_DTYPE`): fp16 2 bytes a value, or int8 1 byte a value and 1 exponent byte a
    /// (token, kv head).
    kv_dtype: KvDtype,
    /// The MTP head's device bytes as the loader holds them, when the plan is known ([`NemotronMemory::with_head_weight_bytes`]): the
    /// [`MtpHeadBytes::weights`] line. `None`: the published bf16 head's formula.
    head_weights: Option<u64>,
    /// The trunk's `lm_head` is EXL3 (the MixedK pack's): a TP rank's logits temp is a 128-column-block share of the vocabulary.
    lm_head_exl3: bool,
}

impl NemotronMemory {
    /// `weight_bytes`: what the pack holds on the device (`LayerPlan::bytes(&inventory)`; zero when unknown, as in the probe).
    pub fn new(cfg: &OmniConfig, weight_bytes: u64) -> Self {
        // layers are counted by what they are (their spec), not by the length of a pattern: the MTP head has two layers and one cache
        let count = |specs: Vec<crate::family::LayerSpec>, m: Mixer| specs.iter().filter(|l| l.mixer == m).count() as u64;
        Self {
            mamba_layers: count(cfg.layer_specs(), Mixer::Mamba2),
            attn_layers: count(cfg.layer_specs(), Mixer::GqaNope),
            mtp_attn_layers: count(cfg.mtp_specs(), Mixer::GqaNope),
            cfg: cfg.clone(),
            weight_bytes,
            restricted_rows: 0,
            tp_ranks: 0,
            kv_dtype: gb_cuda::nemotron_attn::configured_kv_dtype().unwrap_or_default(),
            head_weights: None,
            lm_head_exl3: false,
        }
    }

    /// The MTP head's weights are `bytes` on the device, as the loader holds them in the pack's forms (`tp_shard::ShardPlan::rank_bytes`
    /// of the plan's head units: bf16 tensors in place, EXL3 projections at their widths with their scales as the bridge keeps them).
    /// [`MtpHeadBytes::weights`] reports it; without it the line is the published bf16 head's formula.
    pub fn with_head_weight_bytes(mut self, bytes: u64) -> Self {
        self.head_weights = Some(bytes);
        self
    }

    /// The trunk's `lm_head` is EXL3 (`LayerPlan::lm_head_exl3`): a TP rank's logits temp holds its share of the 128-column-block cut
    /// (`tp_shard::head_temp_rows_in`; two ranks: the bf16 head's 65,536 rows).
    pub fn with_lm_head_exl3(mut self, on: bool) -> Self {
        self.lm_head_exl3 = on;
        self
    }

    /// The KV cache is of `dtype` (the process's configured one by default).
    pub fn with_kv_dtype(mut self, dtype: KvDtype) -> Self {
        self.kv_dtype = dtype;
        self
    }

    /// The KV element type this model is priced for.
    pub fn kv_dtype(&self) -> KvDtype {
        self.kv_dtype
    }

    /// Bytes one token costs in ONE attention layer, K and V: 1,024 in fp16 (2 x 256 values x 2 B), 516 in int8 (2 x (256 values + 2
    /// exponents) x 1 B).
    pub fn kv_bytes_per_token_layer(&self) -> u64 {
        match self.kv_dtype {
            KvDtype::F16 => 2 * self.kv_dim() * KV_ELEM_BYTES,
            KvDtype::Int8 => 2 * (self.kv_dim() + self.cfg.kv_heads as u64),
        }
    }

    /// This process is one rank of a two-rank tensor-parallel group (`model::Bufs::tp_tmp`, `tp_qkv`, `tp_logits`): the scratch is
    /// priced with the TP temps. The weights are the rank's share (`engine::resident_weight_bytes_tp`). A group of another size is
    /// [`NemotronMemory::with_tp_ranks`].
    pub fn with_tp_rank(self) -> Self {
        self.with_tp_ranks(2)
    }

    /// This process is one rank of a tensor-parallel group of `ranks` ranks: the scratch is priced with the TP temps at that rank
    /// count ([`NemotronMemory::tp_scratch_bytes`]); fewer than two ranks is not a group and prices nothing.
    pub fn with_tp_ranks(mut self, ranks: usize) -> Self {
        self.tp_ranks = if ranks >= 2 { ranks as u64 } else { 0 };
        self
    }

    /// The draft head runs over a restricted vocabulary of `rows` token ids: its sub-head table (`rows x hidden` bf16, a copy of those
    /// rows of the `lm_head`) is priced with the head's scratch. A count past the vocabulary is refused by `profile`.
    pub fn with_restricted_head(mut self, rows: usize) -> Self {
        self.restricted_rows = rows as u64;
        self
    }

    /// The MTP head's bytes by line at the model's narrowest step (64 rows), for this geometry and tap-store capacity.
    pub fn mtp_head(&self) -> MtpHeadBytes {
        self.mtp_head_for(WINDOW_ROWS as u64)
    }

    /// [`NemotronMemory::mtp_head`] for a model whose widest step is `step_rows` rows: the head's buffers are `max(step_rows, 64)` rows
    /// (`MtpState::new(slots, max(prefill_rows, WINDOW_ROWS), ..)`), what the profile counts when `MemoryOptions::mtp` is set.
    pub fn mtp_head_for(&self, step_rows: u64) -> MtpHeadBytes {
        self.mtp_head_for_window(step_rows, WINDOW_ROWS as u64)
    }

    /// [`NemotronMemory::mtp_head_for`] for a model whose decode window takes `window_rows` rows (a wide model): the window tap buffer is
    /// `max(window_rows, 64)` rows of the trunk's post-norm rows (`MtpState::with_window`), the rest as before.
    pub fn mtp_head_for_window(&self, step_rows: u64, window_rows: u64) -> MtpHeadBytes {
        let c = &self.cfg;
        let (h, f32b, bf16b) = (c.hidden as u64, 4u64, 2u64);
        let rows = step_rows.max(WINDOW_ROWS as u64);
        // eh_proj [hidden, 2 hidden]; attention q and o [q_dim, hidden], k and v [kv_dim, hidden]; the latent MoE: router [experts, hidden],
        // fc1 and fc2 latent projections [latent, hidden], the shared expert up and down [shared_inter, hidden], and per routed expert
        // an up [expert_inter, latent] and a down [latent, expert_inter]
        let elems = h * 2 * h
            + 2 * self.q_dim() * h
            + 2 * self.kv_dim() * h
            + c.experts as u64 * h
            + 2 * c.latent as u64 * h
            + 2 * c.shared_inter as u64 * h
            + c.experts as u64 * 2 * c.expert_inter as u64 * c.latent as u64;
        // the head's norms: enorm, hnorm, the attention and MoE block norms, the final norm; the router's correction bias is f32
        let norms = 5 * h;
        let bias_bytes = c.experts as u64 * f32b;
        // per row of the head's buffers: the draft ids (i32), the chain's ids and values of every depth, and the keyed-pick record
        let draft_row = 4 + (MTP_MAX_DEPTH as u64) * (4 + 4) + std::mem::size_of::<KeyedRow>() as u64;
        MtpHeadBytes {
            // the loader's figure when the plan gave it (a 4-bit or MixedK head), else the published bf16 head's
            weights: self.head_weights.unwrap_or((elems + norms).saturating_mul(bf16b).saturating_add(bias_bytes)),
            kv_per_token: self.mtp_attn_layers.saturating_mul(self.kv_bytes_per_token_layer()),
            tap_store_per_slot: (MTP_TAP_ROWS as u64).saturating_mul(h * f32b),
            // `win_tap`: one window of the trunk's rows (64, or a wide model's window), whatever the step's width
            tap_window: window_rows.max(WINDOW_ROWS as u64) * h * f32b,
            activations: rows * (2 * h + 4 * h) * f32b,
            drafts: rows * draft_row,
            // the table of the restricted rows and their i32 token ids
            sub_head: self.restricted_rows.saturating_mul(h * bf16b + 4),
        }
    }

    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    // ---- the prefix cache's terms (`docs/NEMOTRON_LONG_CONTEXT.md`, section 1.1)

    /// Bytes of the recurrent state a snapshot copies: every Mamba layer's S and conv history. Equals `MemoryProfile::state_bytes_per_seq`.
    pub fn snapshot_state_bytes(&self) -> u64 {
        self.mamba_layers.saturating_mul(self.state_per_layer())
    }

    /// Bytes of one snapshot slot: the recurrent state, and with `mtp` the head's carry (one f32 tap row, [`super::prefix::CARRY_BYTES`]).
    pub fn snapshot_bytes(&self, mtp: bool) -> u64 {
        super::prefix::snapshot_bytes(self.snapshot_state_bytes(), mtp)
    }

    /// Bytes of one KV page across every layer that caches context: 8,192 B a token for the trunk (2 MiB a page), 9,216 with the head's stripe
    /// (2.25 MiB). A page a snapshot references is counted once however many entries and slots share it, and a restore copies exactly one.
    pub fn page_bytes(&self, mtp: bool) -> u64 {
        let mut per_token = self.attn_layers.saturating_mul(self.kv_bytes_per_token_layer());
        if mtp {
            per_token = per_token.saturating_add(self.mtp_attn_layers.saturating_mul(self.kv_bytes_per_token_layer()));
        }
        per_token.saturating_mul(PAGE_TOKENS)
    }

    /// What an entry of a prefix of `tokens` tokens costs alone: its snapshot slot and every page it references.
    pub fn entry_bytes(&self, tokens: u64, mtp: bool) -> u64 {
        self.snapshot_bytes(mtp).saturating_add(tokens.div_ceil(PAGE_TOKENS).saturating_mul(self.page_bytes(mtp)))
    }

    /// The width of the Mamba2 mixer's x and z (`heads x head_dim`).
    fn inner(&self) -> u64 {
        self.cfg.mamba_heads as u64 * self.cfg.mamba_dim as u64
    }

    /// The conv's channels: x | B | C.
    fn conv_dim(&self) -> u64 {
        self.inner() + 2 * self.cfg.groups as u64 * self.cfg.state as u64
    }

    /// The `in_proj` output: z | xBC | dt.
    fn in_proj_cols(&self) -> u64 {
        self.inner() + self.conv_dim() + self.cfg.mamba_heads as u64
    }

    fn q_dim(&self) -> u64 {
        self.cfg.heads as u64 * self.cfg.head_dim as u64
    }

    fn kv_dim(&self) -> u64 {
        self.cfg.kv_heads as u64 * self.cfg.head_dim as u64
    }

    /// Bytes per row of each buffer of `model::Bufs` that scales with the step's rows, in the order the model names them. f32 unless
    /// the buffer is documented bf16 there (`ssm_y`, `gated`); the top-k outputs are `MAX_CANDIDATES` ids (i32) and values (f32).
    pub fn activations_per_row(&self) -> Vec<(&'static str, u64)> {
        let hidden = self.cfg.hidden as u64;
        let (f32b, bf16b) = (4u64, 2u64);
        let cand = MAX_CANDIDATES as u64;
        vec![
            ("x", hidden * f32b),
            ("xn", hidden * f32b),
            ("mix", hidden * f32b),
            ("proj", self.in_proj_cols() * f32b),
            ("z", self.inner() * f32b),
            ("xbc", self.conv_dim() * f32b),
            ("dt", self.cfg.mamba_heads as u64 * f32b),
            ("ssm_y", self.inner() * bf16b),
            ("gated", self.inner() * bf16b),
            ("q", self.q_dim() * f32b),
            ("k", self.kv_dim() * f32b),
            ("v", self.kv_dim() * f32b),
            ("attn_o", self.q_dim() * f32b),
            ("logits", self.cfg.vocab as u64 * f32b),
            ("topk_ids", cand * 4),
            ("topk_vals", cand * f32b),
        ]
    }

    /// Device bytes one rank of a tensor-parallel group adds to the step buffers (`model::Bufs::tp_tmp`, `tp_qkv`, `tp_logits`; none
    /// on a single rank): a sharded linear's local output (`rows` x the widest slice any rank holds of a matrix the plan slices:
    /// `ShardPlan::widest_slice`, at two ranks the `in_proj`'s second half, 9,344 columns, and at three and more a q or o share, the
    /// `in_proj` being replicated there), the merged q, k, v (`rows` x (q + 2 kv) columns, when the model has attention layers) and the
    /// rank's logits (the window's rows at most x `head_temp_rows`, a rank's widest share of the vocabulary: 65,536 at two ranks).
    /// `rows` is the widest step, as for [`NemotronMemory::scratch`]. The group size is [`NemotronMemory::with_tp_ranks`] (two ranks
    /// when the model was not given one: the figure the pinned tests and the two-node runs carry). The temp is priced from what the
    /// default plan slices; a plan that replicates more (`REXL3_TP_ATTN`, `REXL3_TP_INPROJ`) allocates less, never more.
    pub fn tp_scratch_bytes(&self, rows: u64, attention: bool) -> u64 {
        let f32b = 4u64;
        let ranks = if self.tp_ranks >= 2 { self.tp_ranks as usize } else { 2 };
        let tmp = rows * ShardPlan::widest_slice(&self.cfg, ranks) as u64 * f32b;
        let qkv = if attention { rows * (self.q_dim() + 2 * self.kv_dim()) * f32b } else { 0 };
        let unit = if self.lm_head_exl3 { EXL3_HEAD_ROW_UNIT } else { HEAD_ROW_UNIT };
        let logits = rows.min(WINDOW_ROWS as u64) * head_temp_rows_in(self.cfg.vocab, ranks, unit) as u64 * f32b;
        // contract P2 at two ranks (lane S): the packed send and receive rows of the in_proj exchange, each as large as the temp
        let pack = if ranks == 2 && super::contract::current() == super::contract::PrefillContract::P2 {
            // and the MoE pair gather's send and receive `y` rows (22 x 1,024 f32 a row each), for a step wider than a window
            2 * tmp + if rows > WINDOW_ROWS as u64 { 2 * rows * 22 * 1024 * f32b + rows * 22 * 4 } else { 0 }
                // and the second half's MoE scratches for the row-half overlap (`model::MoeHalf`), from 1,024 rows
                + if rows >= 1024 { super::model::MoeHalf::bytes((rows / 2) as usize) } else { 0 }
        } else {
            0
        };
        tmp + qkv + logits + pack
    }

    /// Bytes of the tree's attention partials for `rows` rows: `(m, l)` and the unnormalized `o` per row, chunk and query head, for the
    /// [`TREE_CHUNKS_BELOW`] chunks a row below the flash path's threshold can have **whatever the context** (the tree never serves a
    /// row at position 8,192 or more): 17 MB at 64 rows. `Scratch::tree_bytes` is the allocation.
    pub fn attn_tree_partial_bytes(rows: u64, max_context: u64) -> u64 {
        let chunks = max_context.div_ceil(CHUNK_KEYS as u64).min(TREE_CHUNKS_BELOW as u64);
        rows.saturating_mul(chunks).saturating_mul(GROUP_BYTES as u64)
    }

    /// Bytes of the flash path's attention partials for `rows` rows over a context of `max_context` tokens: one `(m, l, o)` group of 32
    /// query heads per row and 8,192-key split, plus the row plan (a count and a pair a row, 8 bytes each). This is what scales with the
    /// context: 128 splits a row at 1M tokens, 136 MB at 64 rows. `Scratch::flash_bytes` is the allocation.
    pub fn attn_flash_partial_bytes(rows: u64, max_context: u64) -> u64 {
        let splits = (max_context.div_ceil(CHUNK_KEYS as u64).div_ceil(TREE_CHUNKS_BELOW as u64)).max(1);
        rows.saturating_mul(splits).saturating_mul(GROUP_BYTES as u64).saturating_add(rows.saturating_add(1).saturating_mul(8))
    }

    /// Bytes of the attention scratch (`nemotron_attn::Scratch::new_serving`) for `rows` rows over a context of `max_context` tokens:
    /// the tree's and the flash path's. `rows` is the widest pass (`min(prefill rows, 64)` in the model: a wider call is run in passes).
    pub fn attn_partial_bytes(rows: u64, max_context: u64) -> u64 {
        Self::attn_tree_partial_bytes(rows, max_context).saturating_add(Self::attn_flash_partial_bytes(rows, max_context))
    }

    /// The scratch of a step of `rows` rows at a context of `max_context` tokens, by buffer group, without the MTP head's.
    pub fn scratch(&self, rows: u64, max_context: u64) -> ScratchBytes {
        self.scratch_with(rows, max_context, false)
    }

    /// [`NemotronMemory::scratch`], with the MTP head's buffers when `mtp`: sized, as the model sizes them, by the widest step
    /// (`max(rows, 64)`), with the window tap buffer fixed at 64 rows.
    pub fn scratch_with(&self, rows: u64, max_context: u64, mtp: bool) -> ScratchBytes {
        self.scratch_for(rows, max_context, mtp, WINDOW_ROWS as u64)
    }

    /// [`NemotronMemory::scratch_with`] for a model whose decode window takes `window_rows` rows: only the head's window tap buffer
    /// depends on it (`rows` is already the widest step, `max(prefill_rows, window_rows)`, which sizes the activations and the
    /// large-row MoE scratch).
    pub fn scratch_for(&self, rows: u64, max_context: u64, mtp: bool, window_rows: u64) -> ScratchBytes {
        let per_row: u64 = self.activations_per_row().iter().map(|(_, b)| b).sum();
        let vocab = self.cfg.vocab;
        let r = usize::try_from(rows).unwrap_or(usize::MAX);
        let allowance = rows.saturating_mul(TOPK_ALLOWANCE_PER_ROW);
        let tp = if self.tp_ranks >= 2 { self.tp_scratch_bytes(rows, self.attn_layers > 0) } else { 0 };
        ScratchBytes {
            activations: rows.saturating_mul(per_row).saturating_add(tp),
            topk_workspace: allowance.max(topk_workspace_bytes(r, vocab, MAX_CANDIDATES) as u64),
            moe: MoeScratch::bytes(r.min(WINDOW_ROWS))
                + if r > WINDOW_ROWS { PrefillScratch::bytes(r) } else { 0 }
                // contract P2: the tensor-core MoE block's scratch beside it
                + if r > WINDOW_ROWS && super::contract::current() == super::contract::PrefillContract::P2 {
                    gb_cuda::nemotron_moe_prefill::mma::MmaScratch::bytes(r)
                } else {
                    0
                },
            // a call of more rows than a window is run by `attn_long` in passes of the scratch's rows: the partials are a window wide
            attn_partials: Self::attn_partial_bytes(rows.min(WINDOW_ROWS as u64), max_context),
            // P1: the fixed workspace; P2: also the tensor-core prefill linear's fp16 input of the whole slice (`contract::workspace_bytes`)
            exl3_workspace: super::contract::workspace_bytes(EXL3_WORKSPACE_BYTES as usize, rows as usize) as u64,
            mtp_head: if mtp { self.mtp_head_for_window(rows, window_rows).scratch() } else { 0 },
        }
    }

    /// Per-sequence state: the SSD state (f32 `[heads][state][head_dim]`) and the conv's last `conv - 1` pre-conv rows (bf16) in every
    /// Mamba layer.
    fn state_per_layer(&self) -> u64 {
        let c = &self.cfg;
        let ssd = c.mamba_heads as u64 * c.state as u64 * c.mamba_dim as u64 * 4;
        let conv = (c.conv as u64).saturating_sub(1) * self.conv_dim() * 2;
        ssd.saturating_add(conv)
    }

    /// One window row's history in one Mamba layer: the `Record` (post-conv xBC bf16, dt bf16, g and decay f32) and the in_proj's
    /// pre-conv xBC retained in f32 for the conv replay.
    fn record_per_layer_row(&self) -> u64 {
        let heads = self.cfg.mamba_heads as u64;
        let record = self.conv_dim() * 2 + heads * 2 + heads * 4 + heads * 4;
        let retained = self.conv_dim() * 4;
        record + retained
    }

    /// What `opts` asks the model to price, refused by name where the engine could not run it or the arithmetic would be wrong.
    fn check(&self, opts: &MemoryOptions) -> Result<(), MemoryError> {
        // dimensions the shared validation cannot see: a zero here would price a buffer or a state at nothing. (The KV planes' own
        // extents are refused by `MemoryProfile::new`, which names the plane.)
        let c = &self.cfg;
        let mut dims = vec![("hidden size", c.hidden), ("vocabulary", c.vocab)];
        if self.mamba_layers > 0 {
            dims.extend([
                ("Mamba heads", c.mamba_heads),
                ("Mamba head dim", c.mamba_dim),
                ("Mamba groups", c.groups),
                ("SSM state size", c.state),
                ("conv kernel", c.conv),
            ]);
        }
        if let Some((name, _)) = dims.into_iter().find(|(_, v)| *v == 0) {
            return Err(MemoryError::Zero(format!("{FAMILY}: {name} is zero, so a layer or buffer would be priced at nothing")));
        }
        if opts.window_rows == 0 {
            return Err(MemoryError::Zero(format!("{FAMILY}: window rows")));
        }
        if opts.window_rows > MAX_WINDOW_ROWS {
            return Err(MemoryError::Inconsistent(format!(
                "{FAMILY}: a verify window of {} rows; the model's decode window takes at most {MAX_WINDOW_ROWS} (the step buffers and \
                 the large-row MoE scratch are sized for it)",
                opts.window_rows
            )));
        }
        if opts.prefill_rows == 0 {
            return Err(MemoryError::Zero(format!("{FAMILY}: prefill rows")));
        }
        // no engine runs a slice this wide; past it the MoE scratch's own arithmetic (rows x experts x intermediate) would overflow
        if opts.prefill_rows > MAX_PRICED_ROWS {
            return Err(MemoryError::Inconsistent(format!(
                "{FAMILY}: a prefill chunk of {} rows is not a size this model prices (at most {MAX_PRICED_ROWS})",
                opts.prefill_rows
            )));
        }
        if opts.max_context == 0 {
            return Err(MemoryError::Zero(format!("{FAMILY}: max context")));
        }
        let chunks = (opts.max_context as u64).div_ceil(CHUNK_KEYS as u64);
        if chunks > MAX_CHUNKS as u64 {
            return Err(MemoryError::Inconsistent(format!(
                "{FAMILY}: a context of {} tokens needs {chunks} attention chunks; the kernel's scratch holds {MAX_CHUNKS}",
                opts.max_context
            )));
        }
        if opts.mtp && self.mtp_attn_layers == 0 {
            return Err(MemoryError::Inconsistent(format!(
                "{FAMILY}: MTP drafting asked for, but the pack has no MTP attention layer"
            )));
        }
        if opts.mtp {
            // the head's weights and buffers are priced from these: a zero would price a part of the head at nothing
            let head_dims = [
                ("attention heads", c.heads),
                ("routed experts", c.experts),
                ("expert width", c.expert_inter),
                ("latent width", c.latent),
                ("shared expert width", c.shared_inter),
            ];
            if let Some((name, _)) = head_dims.into_iter().find(|(_, v)| *v == 0) {
                return Err(MemoryError::Zero(format!(
                    "{FAMILY}: {name} is zero, so the MTP head's weights or buffers would be priced at nothing"
                )));
            }
            if self.restricted_rows > c.vocab as u64 {
                return Err(MemoryError::Inconsistent(format!(
                    "{FAMILY}: a restricted draft head of {} token ids, but the vocabulary has {}",
                    self.restricted_rows, c.vocab
                )));
            }
        }
        Ok(())
    }

    /// The whole footprint of a serving shape: `slots` sequences, a KV pool of `kv_tokens` tokens, and the step shapes of `opts`.
    pub fn budget(&self, opts: &MemoryOptions, slots: usize, kv_tokens: u64) -> Result<Budget, MemoryError> {
        let profile = self.profile(opts)?;
        let rows = profile.shape().scratch_rows;
        let pages = kv_tokens.div_ceil(PAGE_TOKENS);
        let kv = pages
            .checked_mul(profile.kv_page_bytes())
            .ok_or_else(|| MemoryError::Overflow(format!("{FAMILY}: KV of {kv_tokens} tokens")))?;
        let state = profile
            .state_bytes_per_slot()
            .checked_mul(slots as u64)
            .ok_or_else(|| MemoryError::Overflow(format!("{FAMILY}: state of {slots} slots")))?;
        Ok(Budget {
            weights: self.weight_bytes,
            state_per_slot: profile.state_bytes_per_slot(),
            state,
            kv_per_token: profile.kv_bytes_per_token(),
            kv,
            window_records: profile.window_record_bytes(),
            scratch: self.scratch_for(rows, opts.max_context as u64, opts.mtp, opts.window_rows as u64),
            engine_reserve: profile.engine_reserve_bytes(),
            mtp_head: opts.mtp.then(|| self.mtp_head_for_window(WINDOW_ROWS as u64, opts.window_rows as u64)),
        })
    }
}

/// The model at one serving shape: [`MemoryModel`] needs the context (the attention partials) and the MTP flag (the head's cache).
struct AtShape<'a> {
    mem: &'a NemotronMemory,
    opts: MemoryOptions,
}

impl MemoryModel for AtShape<'_> {
    fn family(&self) -> &str {
        FAMILY
    }

    /// fp16 K and V of 2 heads x 128 on each of the trunk's attention layers; the MTP head's attention layer keeps its own cache in the
    /// same pages when it drafts (a class of its own: its layers are counted by spec, not by the head's two layers).
    fn kv_layout(&self) -> KvLayout {
        let m = self.mem;
        // int8: a plane of values (1 byte) and a plane of exponents (one byte a token and kv head) for each of K and V
        let heads = m.cfg.kv_heads as u64;
        let planes = || match m.kv_dtype {
            KvDtype::F16 => {
                vec![KvPlane::per_token("k", m.kv_dim(), KV_ELEM_BYTES), KvPlane::per_token("v", m.kv_dim(), KV_ELEM_BYTES)]
            }
            KvDtype::Int8 => vec![
                KvPlane::per_token("k", m.kv_dim(), 1),
                KvPlane::per_token("k_exp", heads, 1),
                KvPlane::per_token("v", m.kv_dim(), 1),
                KvPlane::per_token("v_exp", heads, 1),
            ],
        };
        let mut layout = KvLayout::new().with(KvLayerClass { name: "gqa_nope", layers: m.attn_layers, planes: planes() });
        if self.opts.mtp {
            layout = layout.with(KvLayerClass { name: "mtp_gqa_nope", layers: m.mtp_attn_layers, planes: planes() });
        }
        layout
    }

    fn recurrent_layers(&self) -> u64 {
        self.mem.mamba_layers
    }

    fn state_bytes_per_seq(&self) -> u64 {
        self.mem.mamba_layers.saturating_mul(self.mem.state_per_layer())
    }

    fn window_record_bytes_per_row(&self) -> u64 {
        self.mem.mamba_layers.saturating_mul(self.mem.record_per_layer_row())
    }

    fn scratch_bytes(&self, rows: u64) -> u64 {
        self.mem.scratch_for(rows, self.opts.max_context as u64, self.opts.mtp, self.opts.window_rows as u64).total()
    }

    fn activation_reserve(&self) -> u64 {
        ACTIVATION_RESERVE
    }

    /// The MTP head keeps no recurrent state (an attention layer and a MoE layer), and its cache is in the paged pool; what it holds
    /// per slot is the store of pending tapped rows (`[MTP_TAP_ROWS][hidden]` f32), charged to admission with the slot's state.
    fn drafter_state_bytes_per_slot(&self) -> u64 {
        if self.opts.mtp {
            self.mem.mtp_head().tap_store_per_slot
        } else {
            0
        }
    }

    /// The RADIO tower and its projector, when images are served (`MemoryOptions::vision`): its weights and its workspace are resident
    /// for the life of the process (`vision::weight_bytes`, `vision::scratch_bytes`: the numbers the tower checks its allocations
    /// against at load), and each image's bf16 features are held until the sequence that owns them ends (`vision::feature_bytes`).
    /// The tower is loaded before the plan is measured, so the planner takes only the features out of the pool
    /// (`MemoryProfile::vision_reserve_bytes(0, concurrent_images)`); the weights and the scratch are reported for the boot's fail-closed
    /// check and the allocation-equals-plan test. The cap is the tower's (`vision::image_token_cap`).
    fn vision(&self) -> Option<VisionMemory> {
        let v = self.opts.vision?;
        let tokens = super::vision::image_token_cap(v.max_image_tokens);
        Some(VisionMemory {
            weight_bytes: Some(super::vision::weight_bytes()),
            encoder_scratch_bytes: super::vision::scratch_bytes(tokens),
            feature_bytes_per_image: super::vision::feature_bytes(tokens),
        })
    }
}

impl MemorySource for NemotronMemory {
    /// The validated profile at `opts`. A step is priced at `max(prefill_rows, window_rows)` rows, as asked: the engine's cap on a
    /// prefill slice (`engine::check_options`) is its own refusal, so a profile at a wider chunk is the honest price of that chunk.
    fn profile(&self, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
        self.check(opts)?;
        let shape = StepShape {
            page_tokens: PAGE_TOKENS,
            window_rows: opts.window_rows as u64,
            scratch_rows: (opts.prefill_rows as u64).max(opts.window_rows as u64),
        };
        MemoryProfile::new(&AtShape { mem: self, opts: *opts }, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gb_sched::Admission;

    const PACK: &str = include_str!("../../../../../testdata/nemotron_omni/config.json");

    fn real() -> OmniConfig {
        OmniConfig::from_json(&serde_json::from_str(PACK).unwrap()).unwrap()
    }

    /// The serving shape of the first milestone: 64-row windows and slices, a 262,144-token context, text only, no MTP drafting.
    fn opts() -> MemoryOptions {
        MemoryOptions { window_rows: 64, prefill_rows: 64, max_context: 262_144, mtp: false, vision: None, trees: false }
    }

    fn mem() -> NemotronMemory {
        NemotronMemory::new(&real(), 0).with_kv_dtype(KvDtype::F16)
    }

    const GIB: u64 = 1 << 30;

    #[test]
    fn the_layer_counts_come_from_the_specs() {
        let m = mem();
        assert_eq!((m.mamba_layers, m.attn_layers, m.mtp_attn_layers), (40, 8, 1));
        // the MTP head `*E` has two layers and one attention cache
        assert_eq!(real().mtp_pattern.len(), 2);
    }

    #[test]
    fn the_profile_reproduces_the_modelled_numbers_exactly() {
        let p = mem().profile(&opts()).unwrap();
        // 40 Mamba layers x (S f32 128 x 128 x 64 = 4,194,304 + conv bf16 3 x 10,240 = 61,440)
        assert_eq!(p.state_bytes_per_seq(), 40 * (4_194_304 + 61_440));
        assert_eq!(p.state_bytes_per_seq(), 170_229_760);
        assert_eq!(p.state_bytes_per_slot(), 170_229_760, "the MTP head keeps no state of its own");
        assert_eq!(p.recurrent_layers(), 40);
        // 8 attention layers x K and V x 2 heads x 128 x fp16
        assert_eq!(p.kv_bytes_per_token(), 8 * 2 * 2 * 128 * 2);
        assert_eq!(p.kv_bytes_per_token(), 8192);
        assert_eq!(p.kv_page_bytes(), 8192 * 256);
        // per Mamba layer and row: the record 20,480 + 256 + 512 + 512 = 21,760 and the retained f32 xBC 10,240 x 4 = 40,960
        assert_eq!(p.window_record_bytes_per_row(), 40 * (21_760 + 40_960));
        assert_eq!(p.window_record_bytes_per_row(), 2_508_800);
        assert_eq!(p.window_record_bytes(), 64 * 2_508_800);
        assert_eq!(p.window_record_bytes(), 160_563_200);
        assert_eq!(p.drafter_state_bytes_per_slot(), 0);
        assert!(p.vision().is_none());
        assert_eq!(p.vision_reserve_bytes(1, 4), Ok(0));
    }

    #[test]
    fn the_towers_bytes_are_priced_only_when_images_are_served_and_at_the_towers_cap() {
        use crate::memory::VisionOptions;
        let m = mem();
        assert!(m.profile(&opts()).unwrap().vision().is_none(), "text only prices nothing for the tower");
        let v = |tokens| VisionOptions { max_image_tokens: tokens, max_images_per_request: 4 };
        let p = m.profile(&MemoryOptions { vision: Some(v(3328)), ..opts() }).unwrap();
        let vm = p.vision().unwrap();
        // the numbers the tower checks its real allocations against at load (`vision::weight_bytes`, `vision::scratch_bytes`)
        assert_eq!(vm.weight_bytes, Some(1_727_774_720));
        assert_eq!(vm.encoder_scratch_bytes, 688_671_744 + 20_447_232 + 4);
        assert_eq!(vm.feature_bytes_per_image, 3328 * 4096 * 2);
        // the planner holds back the features of every image that can be alive; the tower's own bytes were allocated before it measured
        assert_eq!(p.vision_reserve_bytes(0, 8 * 4), Ok(32 * 27_262_976));
        assert_eq!(p.vision_reserve_bytes(1, 4), Ok(vm.encoder_scratch_bytes + 4 * 27_262_976));
        // the server's own default of 4,096 tokens is above the tower: priced at what it can produce
        let over = m.profile(&MemoryOptions { vision: Some(v(4096)), ..opts() }).unwrap().vision().unwrap();
        assert_eq!(over, vm);
        // and a lower limit buys memory
        let small = m.profile(&MemoryOptions { vision: Some(v(256)), ..opts() }).unwrap().vision().unwrap();
        assert!(small.encoder_scratch_bytes < vm.encoder_scratch_bytes && small.feature_bytes_per_image == 256 * 8192);
        // images change nothing else the profile prices
        let plain = m.profile(&opts()).unwrap();
        assert_eq!(
            (p.kv_bytes_per_token(), p.state_bytes_per_seq(), p.scratch_bytes(), p.window_record_bytes_per_row()),
            (plain.kv_bytes_per_token(), plain.state_bytes_per_seq(), plain.scratch_bytes(), plain.window_record_bytes_per_row())
        );
    }

    #[test]
    fn an_int8_kv_is_516_bytes_a_token_and_layer_and_halves_the_pool() {
        let m = mem().with_kv_dtype(KvDtype::Int8);
        assert_eq!(m.kv_bytes_per_token_layer(), 516);
        let off = m.profile(&opts()).unwrap();
        let on = m.profile(&MemoryOptions { mtp: true, ..opts() }).unwrap();
        assert_eq!(off.kv_bytes_per_token(), 8 * 516);
        assert_eq!(on.kv_bytes_per_token(), 9 * 516, "the head's stripe is int8 too");
        assert_eq!(off.kv_page_bytes(), 8 * 516 * 256);
        assert_eq!(m.page_bytes(true), 9 * 516 * 256, "a cache page (one restore copy): 1.13 MiB against 2.25 MiB");
        assert_eq!(m.mtp_head().kv_per_token, 516);
        // the KV planes as the device lays them out: per token and layer, 256 values and 2 exponents of K and of V
        assert_eq!(gb_cuda::nemotron_attn::KvDtype::Int8.token_bytes() as u64, m.kv_bytes_per_token_layer());
        // 1M tokens with the head: 4.54 GiB of KV against 9 GiB
        let f16 = mem().with_kv_dtype(KvDtype::F16).profile(&MemoryOptions { mtp: true, ..opts() }).unwrap();
        assert_eq!(f16.kv_bytes_per_token(), 9216);
        assert_eq!((1u64 << 20) * on.kv_bytes_per_token(), 4_869_586_944);
    }

    #[test]
    fn the_mtp_attention_layer_is_counted_by_spec_and_only_when_it_drafts() {
        let m = mem();
        let off = m.profile(&opts()).unwrap();
        let on = m.profile(&MemoryOptions { mtp: true, ..opts() }).unwrap();
        assert_eq!(
            on.kv_bytes_per_token() - off.kv_bytes_per_token(),
            1024,
            "one attention layer x K and V x 2 heads x 128 x fp16"
        );
        assert_eq!(on.kv_bytes_per_token(), 9216);
        // by layer count the head would be two layers and 2 KiB
        assert_ne!(on.kv_bytes_per_token() - off.kv_bytes_per_token(), 2 * 1024);
        assert_eq!(on.state_bytes_per_seq(), off.state_bytes_per_seq());
        // a pack without the head cannot be asked to count it
        let mut c = real();
        c.mtp_pattern = String::new();
        let e =
            NemotronMemory::new(&c, 0).with_kv_dtype(KvDtype::F16).profile(&MemoryOptions { mtp: true, ..opts() }).unwrap_err();
        assert!(matches!(&e, MemoryError::Inconsistent(m) if m.contains("MTP")), "{e}");
        assert!(NemotronMemory::new(&c, 0).with_kv_dtype(KvDtype::F16).profile(&opts()).is_ok());
    }

    #[test]
    fn one_million_tokens_of_kv_is_eight_gib_a_sequence_and_prices_at_one_and_two_slots() {
        let o = MemoryOptions { max_context: 1 << 20, ..opts() };
        let p = mem().profile(&o).unwrap();
        assert_eq!(p.kv_bytes_per_token() * (1 << 20), 8 * GIB);
        assert_eq!(p.kv_page_bytes() * 4096, 8 * GIB, "4,096 pages of 256 tokens");
        for slots in [1usize, 2] {
            let model = gb_sched::ModelMemory::from_profile(&p, 0, 0);
            // free memory for the weights' complement: plenty, so the KV cap decides
            let cap = slots as u64 * (1 << 20);
            let plan = crate::plan::plan_memory(&p, 100 * GIB, 4 * GIB, slots, cap, 0).unwrap();
            assert_eq!(plan.state_bytes_per_slot, 170_229_760);
            assert_eq!(plan.block_bytes, 2 << 20);
            // a page per slot of slack beyond the token budget, and the admission pool is the budget itself
            assert_eq!(plan.kv_blocks as u64, slots as u64 * 4096 + slots as u64);
            assert_eq!(plan.kv_tokens, slots as u64 * (1 << 20));
            assert_eq!(plan.admit_bytes, slots as u64 * (8 * GIB + 170_229_760));
            assert_eq!(
                plan.planned_bytes(slots),
                p.engine_reserve_bytes() + slots as u64 * 170_229_760 + plan.kv_blocks as u64 * (2 << 20)
            );
            let mut adm: Admission = plan.admission(model, slots, 1 << 20).unwrap();
            assert_eq!(adm.request_limit(), Some(1 << 20));
            // every slot can hold a full 1M-token sequence at once, and one more is refused for want of a slot
            for _ in 0..slots {
                assert!(adm.admit(1 << 20).is_ok());
            }
            assert!(adm.admit(1).is_err());
            // and the cost admission charges a 1M-token sequence is its KV plus its slot's state
            assert_eq!(adm.cost(1 << 20), 8 * GIB + 170_229_760);
        }
        // the whole footprint at two slots, from the table
        let weights = 60 * GIB;
        let b = NemotronMemory::new(&real(), weights).with_kv_dtype(KvDtype::F16).budget(&o, 2, 2 << 20).unwrap();
        assert_eq!((b.state, b.kv), (2 * 170_229_760, 16 * GIB));
        assert_eq!(b.total(), weights + b.state + b.kv + b.engine_reserve);
    }

    #[test]
    fn a_tp_rank_adds_the_local_linear_output_the_merged_qkv_and_half_the_logits() {
        let m = mem();
        // under contract P2 (the default since rc2d) a rank of two also holds the packed in_proj exchange (send and receive, each a
        // temp) and, above a window, the MoE pair gather's send and receive `y` rows and its map; P1 holds neither
        let p2 = super::super::contract::current() == super::super::contract::PrefillContract::P2;
        let pack = |rows: u64| {
            if p2 {
                2 * rows * 9344 * 4 + if rows > 64 { 2 * rows * 22 * 1024 * 4 + rows * 22 * 4 } else { 0 }
            } else {
                0
            }
        };
        // 64 rows: the widest slice is the second half of the in_proj, 9,344 columns (73 tiles of 128)
        let want = 64 * 9344 * 4 + 64 * (4096 + 2 * 256) * 4 + 64 * 65_536 * 4 + pack(64);
        assert_eq!(m.tp_scratch_bytes(64, true), want);
        // a prefill slice of 256 rows has wider temps but the logits stay a window wide
        assert_eq!(m.tp_scratch_bytes(256, true), 256 * 9344 * 4 + 256 * 4608 * 4 + 64 * 65_536 * 4 + pack(256));
        // no attention layer on the device: no merged buffer
        assert_eq!(m.tp_scratch_bytes(64, false), 64 * 9344 * 4 + 64 * 65_536 * 4 + pack(64));
    }

    #[test]
    fn a_tp_rank_of_a_larger_group_prices_the_temps_of_its_own_cuts() {
        // two ranks (the pinned figures): 9,344 columns of in_proj, 65,536 logits rows
        let two = mem().with_tp_ranks(2);
        assert_eq!(two.tp_scratch_bytes(64, true), mem().tp_scratch_bytes(64, true));
        assert_eq!(mem().with_tp_rank().tp_scratch_bytes(64, true), two.tp_scratch_bytes(64, true));
        // three and more: the in_proj is replicated (no slice, no temp for it); the widest slice is a q or o share, ceil(32 / R) tiles;
        // the logits temp is the widest rank's share of the vocabulary in units of four rows
        for (ranks, slice_cols, logits_rows) in
            [(3usize, 11 * 128, 43_692u64), (4, 8 * 128, 32_768), (5, 7 * 128, 26_216), (8, 4 * 128, 16_384)]
        {
            let m = mem().with_tp_ranks(ranks);
            let want = 64 * slice_cols as u64 * 4 + 64 * (4096 + 2 * 256) * 4 + 64 * logits_rows * 4;
            assert_eq!(m.tp_scratch_bytes(64, true), want, "{ranks} ranks");
            // the pricing never grows with the group: more ranks cut the logits temp and the slices
            assert!(m.tp_scratch_bytes(64, true) < two.tp_scratch_bytes(64, true));
        }
        // a group of one is not a group: nothing is priced
        let one = mem().with_tp_ranks(1).scratch(64, 8192);
        assert_eq!(one.activations, mem().scratch(64, 8192).activations);
    }

    #[test]
    fn a_tp_rank_prices_its_temps_in_the_activations_and_a_single_rank_does_not() {
        let single = mem().scratch(64, 8192);
        let rank = mem().with_tp_rank().scratch(64, 8192);
        assert_eq!(rank.activations, single.activations + mem().tp_scratch_bytes(64, true));
        assert_eq!(
            (rank.topk_workspace, rank.moe, rank.attn_partials),
            (single.topk_workspace, single.moe, single.attn_partials)
        );
    }

    #[test]
    fn the_scratch_is_its_five_named_parts_and_a_row_of_logits_is_512_kib() {
        let m = mem();
        let per_row = m.activations_per_row();
        let get = |n: &str| per_row.iter().find(|(k, _)| *k == n).unwrap().1;
        assert_eq!(get("logits"), 131_072 * 4);
        assert_eq!(get("logits"), 512 << 10);
        assert_eq!((get("x"), get("proj"), get("z"), get("xbc"), get("dt")), (16_384, 74_240, 32_768, 40_960, 512));
        assert_eq!((get("ssm_y"), get("gated"), get("q"), get("k"), get("attn_o")), (16_384, 16_384, 16_384, 1024, 16_384));
        let total: u64 = per_row.iter().map(|(_, b)| b).sum();
        assert_eq!(total, 790_016);
        let s = m.scratch(64, 262_144);
        assert_eq!(s.activations, 64 * 790_016);
        assert_eq!(s.moe, MoeScratch::bytes(64));
        assert!((90_000_000..=110_000_000).contains(&s.moe), "about 100 MB at 64 rows: {}", s.moe);
        assert_eq!(
            s.attn_partials,
            64 * (16 + 32) * 32 * 520 + 65 * 8,
            "16 tree chunks and 32 flash splits a row (256k tokens), and the row plan"
        );
        assert_eq!(s.exl3_workspace, EXL3_WORKSPACE_BYTES);
        assert_eq!(s.topk_workspace, 64 * 65_536);
        assert_eq!(s.total(), s.activations + s.topk_workspace + s.moe + s.attn_partials + s.exl3_workspace);
        // and the profile's scratch is exactly that, at the profile's step rows
        assert_eq!(mem().profile(&opts()).unwrap().scratch_bytes(), s.total());
    }

    #[test]
    fn the_attention_partials_are_the_serving_scratchs_allocation_a_tree_of_16_chunks_and_a_split_per_8192_keys() {
        use gb_cuda::nemotron_attn::Scratch;
        // the plan's formula is `Scratch::new_serving`'s allocation (`Scratch::serving_bytes`), for the shapes the model builds
        for (rows, ctx) in
            [(1u64, 512u64), (1, 513), (64, 8192), (64, 8193), (64, 262_144), (64, 1 << 20), (64, 8 << 20), (256, 1 << 20)]
        {
            let chunks = (ctx as usize).div_ceil(CHUNK_KEYS);
            assert_eq!(
                NemotronMemory::attn_partial_bytes(rows, ctx),
                Scratch::serving_bytes(rows as usize, chunks) as u64,
                "{rows} rows x {ctx}"
            );
            assert_eq!(
                NemotronMemory::attn_tree_partial_bytes(rows, ctx),
                Scratch::tree_bytes(rows as usize, chunks.min(16)) as u64,
                "tree, {rows} rows x {ctx}"
            );
            assert_eq!(
                NemotronMemory::attn_flash_partial_bytes(rows, ctx),
                Scratch::flash_bytes(rows as usize, gb_cuda::nemotron_attn::flash_splits(chunks)) as u64,
                "flash, {rows} rows x {ctx}"
            );
        }
        // 64 rows (a window's) at 1M tokens: 16 chunks of tree and 128 splits of flash, 153 MB; the tree alone sized for the context
        // (what a scratch of `Scratch::new` holds, and what the plan priced before the flash path) was 2.18 GB
        let group = 32 * 520u64;
        assert_eq!(NemotronMemory::attn_tree_partial_bytes(64, 1 << 20), 64 * 16 * group);
        assert_eq!(NemotronMemory::attn_flash_partial_bytes(64, 1 << 20), 64 * 128 * group + 65 * 8);
        assert_eq!(NemotronMemory::attn_partial_bytes(64, 1 << 20), 153_354_760);
        assert_eq!(Scratch::tree_bytes(64, 2048), 2_181_038_080);
        // the tree part is flat in the context, the flash part is one group a row per 8,192 keys
        assert_eq!(NemotronMemory::attn_tree_partial_bytes(64, 8192), NemotronMemory::attn_tree_partial_bytes(64, 8 << 20));
        assert_eq!(
            NemotronMemory::attn_partial_bytes(64, 2 << 20) - NemotronMemory::attn_partial_bytes(64, 1 << 20),
            64 * 128 * group
        );
        // a context that is not a multiple of 512 or 8,192 takes the partial chunk and split too
        assert_eq!(NemotronMemory::attn_tree_partial_bytes(1, 513), 2 * group);
        assert_eq!(NemotronMemory::attn_tree_partial_bytes(1, 512), group);
        assert_eq!(NemotronMemory::attn_flash_partial_bytes(1, 8193), 2 * group + 2 * 8);
        // the scratch the profile prices at 1M tokens is no longer dominated by it
        let o = MemoryOptions { max_context: 1 << 20, ..opts() };
        let s = mem().scratch(64, o.max_context as u64);
        assert!(s.attn_partials < s.activations + s.moe + s.topk_workspace + s.exl3_workspace);
    }

    #[test]
    fn the_activation_reserve_covers_the_modelled_scratch_and_a_longer_context_raises_it() {
        // 64 rows at the default context and at the model's 1M tokens: scratch plus a full window's records stay under the blanket
        for max_context in [262_144usize, 1 << 20] {
            let o = MemoryOptions { max_context, ..opts() };
            let p = mem().profile(&o).unwrap();
            let modelled = p.scratch_bytes() + p.window_record_bytes();
            assert!(
                modelled < ACTIVATION_RESERVE,
                "{max_context}: {} MiB modelled against {} MiB",
                modelled >> 20,
                ACTIVATION_RESERVE >> 20
            );
            assert_eq!(p.engine_reserve_bytes(), ACTIVATION_RESERVE);
        }
        // the 1M-token case in the pieces a reader can check: 153 MB partials + 100 MB MoE + 50 MB buffers + 4 MB top-k + 32 MiB + 160 MB
        let p = mem().profile(&MemoryOptions { max_context: 1 << 20, ..opts() }).unwrap();
        let modelled = p.scratch_bytes() + p.window_record_bytes();
        assert!((450_000_000..600_000_000).contains(&modelled), "{modelled}");
        // the partials are one flash group a row per 8,192 keys now (1.1 GB at 64 rows even at the model's 8M-token limit), so the
        // blanket covers every context the kernels take; a step wider than the engine runs raises the reserve to the modelled
        // buffers instead of overcommitting (4,096 rows: 4.6 GB of buffers)
        let p = mem().profile(&MemoryOptions { max_context: 8 << 20, ..opts() }).unwrap();
        assert_eq!(p.engine_reserve_bytes(), ACTIVATION_RESERVE);
        let p = mem().profile(&MemoryOptions { prefill_rows: 4096, max_context: 1 << 20, ..opts() }).unwrap();
        assert_eq!(p.engine_reserve_bytes(), p.scratch_bytes() + p.window_record_bytes());
        assert!(p.engine_reserve_bytes() > ACTIVATION_RESERVE);
    }

    #[test]
    fn the_large_row_prefill_scratch_is_priced_beside_the_decode_scratch_and_only_above_a_window() {
        let m = mem();
        // at a window's 64 rows there is no prefill scratch (the model allocates one only for a slice wider than a window)
        assert_eq!(m.scratch(64, 262_144).moe, MoeScratch::bytes(64));
        assert_eq!(m.scratch(16, 262_144).moe, MoeScratch::bytes(16));
        // above it: the decode scratch at 64 rows (windows, narrow slices, the head's calls) and the prefill scratch at the step's rows
        // (under contract P2, the default since rc2d, also the tensor-core block's scratch)
        let p2 = super::super::contract::current() == super::super::contract::PrefillContract::P2;
        for rows in [65u64, 128, 256, 512] {
            let s = m.scratch(rows, 262_144);
            let mma = if p2 { gb_cuda::nemotron_moe_prefill::mma::MmaScratch::bytes(rows as usize) } else { 0 };
            assert_eq!(s.moe, MoeScratch::bytes(64) + PrefillScratch::bytes(rows as usize) + mma, "{rows} rows");
        }
        let mb = PrefillScratch::bytes(256) as f64 / 1e6;
        assert!((100.0..160.0).contains(&mb), "the prefill scratch at 256 rows: {mb} MB");
        // the profile at the engine's cap prices exactly that, and the 256k-token case stays under the blanket reserve
        let o = MemoryOptions { prefill_rows: 256, ..opts() };
        let p = m.profile(&o).unwrap();
        assert_eq!(p.shape().scratch_rows, 256);
        assert_eq!(p.scratch_bytes(), m.scratch(256, 262_144).total());
        assert!(
            p.scratch_bytes() + p.window_record_bytes() < ACTIVATION_RESERVE,
            "{}",
            p.scratch_bytes() + p.window_record_bytes()
        );
        assert_eq!(p.engine_reserve_bytes(), ACTIVATION_RESERVE);
        // the attention partials do NOT grow with the rows: `attn_long` runs a slice wider than a window in passes of the scratch's 64
        // rows (a row's bits do not depend on the pass), and what scales with the context is one flash group a row per 8,192 keys: 153 MB
        // at a million tokens whatever the slice (8.7 GB at 256 rows if the tree were sized for the context and the slice)
        let big = m.profile(&MemoryOptions { max_context: 1 << 20, ..o }).unwrap();
        assert_eq!(m.scratch(256, 1 << 20).attn_partials, NemotronMemory::attn_partial_bytes(64, 1 << 20));
        assert_eq!(m.scratch(256, 1 << 20).attn_partials, 153_354_760);
        assert!(
            big.scratch_bytes() + big.window_record_bytes() < ACTIVATION_RESERVE,
            "{}",
            big.scratch_bytes() + big.window_record_bytes()
        );
        assert_eq!(big.engine_reserve_bytes(), ACTIVATION_RESERVE);
    }

    #[test]
    fn a_wider_prefill_chunk_is_priced_as_asked() {
        // the engine clamps a chunk past 256 rows to 256 with a log line; the profile at the shipped defaults (4,096) is still a price, not an error
        let p = mem().profile(&MemoryOptions { prefill_rows: 4096, ..opts() }).unwrap();
        assert_eq!(p.shape().scratch_rows, 4096);
        // priced as asked: the row-linear buffers (activations, top-k, the prefill MoE scratch) grow with the 4,096 rows; the attention
        // partials do not (a window's rows, passes beyond), so the price is a multiple of the 64-row one, not 64 times it
        assert!(p.scratch_bytes() > mem().profile(&opts()).unwrap().scratch_bytes() * 4);
        assert!(mem().profile(&MemoryOptions::serving_defaults()).is_ok());
    }

    #[test]
    fn the_weights_are_a_parameter_and_do_not_move_the_profile() {
        let (a, b) = (
            NemotronMemory::new(&real(), 0).with_kv_dtype(KvDtype::F16),
            NemotronMemory::new(&real(), 57 * GIB).with_kv_dtype(KvDtype::F16),
        );
        assert_eq!(a.profile(&opts()).unwrap(), b.profile(&opts()).unwrap());
        assert_eq!(b.weight_bytes(), 57 * GIB);
        let (x, y) = (a.budget(&opts(), 2, 1 << 20).unwrap(), b.budget(&opts(), 2, 1 << 20).unwrap());
        assert_eq!(y.total() - x.total(), 57 * GIB);
    }

    #[test]
    fn the_exl3_workspace_constant_covers_every_mixer_linear_at_the_kernels_widest_call() {
        // csrc/exl3_linear.cu: plan_for (the split of K a shape needs to fill the SMs) and linear_ws_bytes (the rotated input in
        // fragment order, plus the split-K partials). Ported here to hold the constant to the five shapes.
        fn align256(v: u64) -> u64 {
            (v + 255) & !255
        }
        fn ws(m: u64, k: u64, n: u64) -> u64 {
            let (kt, nb) = (k / 16, n / 128);
            let (mut sk, mut wk) = (1u64, if nb >= 64 { 8u64 } else { 4 });
            if nb < 64 {
                while nb * sk < 192 && sk < 64 {
                    sk *= 2;
                }
            }
            while (wk > 4 || sk > 1) && (kt % (sk * wk) != 0 || kt / (sk * wk) < 8) {
                if wk > 4 {
                    wk = 4;
                } else if sk > 1 {
                    sk /= 2;
                } else {
                    break;
                }
            }
            let mut b = align256(m.div_ceil(16) * 16 * k * 2);
            if sk > 1 {
                b += align256(sk * m * n * 4);
            }
            b
        }
        let c = real();
        let m = NemotronMemory::new(&c, 0).with_kv_dtype(KvDtype::F16);
        let (h, inner, q, kv) = (c.hidden as u64, m.inner(), m.q_dim(), m.kv_dim());
        let shapes =
            [("in_proj", h, m.in_proj_cols()), ("out_proj", inner, h), ("q", h, q), ("k", h, kv), ("v", h, kv), ("o", q, h)];
        assert_eq!(m.in_proj_cols(), 18_560);
        let mut widest = 0;
        for rows in [1u64, 64, 128] {
            for (name, k, n) in shapes {
                let b = ws(rows, k, n);
                assert!(b <= EXL3_WORKSPACE_BYTES, "{name} at {rows} rows needs {b} B");
                if rows == 64 {
                    widest = widest.max(b);
                }
                // with a CUDA build the kernel answers for itself: the port must agree with it
                let real = gb_cuda::exl3_linear::linear_workspace_bytes(rows as usize, k as usize, n as usize, 4) as u64;
                if real != 0 {
                    assert_eq!(real, b, "{name} at {rows} rows: the ported plan disagrees with the kernel");
                }
            }
        }
        // the MixedK pack's EXL3 lm_head (whole and a rank's slice at two and three ranks: K = 4096, no K split) and its EXL3 head
        // attention and eh_proj run through the same workspace (`NemotronWeights::max_workspace_bytes`)
        for (name, k, n) in [
            ("lm_head", h, c.vocab as u64),
            ("lm_head half", h, c.vocab as u64 / 2),
            ("lm_head third", h, 43_776),
            ("eh_proj", 2 * h, h),
            ("head q", h, q),
            ("head k", h, kv),
            ("head o", q, h),
        ] {
            for rows in [1u64, 64, 128] {
                let b = ws(rows, k, n);
                assert!(b <= EXL3_WORKSPACE_BYTES, "{name} at {rows} rows needs {b} B");
                for bits in [4usize, 8] {
                    let real = gb_cuda::exl3_linear::linear_workspace_bytes(rows as usize, k as usize, n as usize, bits) as u64;
                    if real != 0 {
                        assert_eq!(real, b, "{name} at {rows} rows, {bits} bits: the ported plan disagrees with the kernel");
                    }
                }
            }
        }
        assert_eq!(ws(128, h, c.vocab as u64), 1 << 20, "the EXL3 lm_head at 128 rows: the rotated input only (no K split)");
        assert_eq!(widest, 9_437_184, "out_proj at 64 rows is the widest");
        assert_eq!(ws(128, inner, h), 18_874_368);
        assert!(2 * widest <= EXL3_WORKSPACE_BYTES, "headroom for a shape change");
    }

    #[test]
    fn the_topk_workspace_of_the_kernel_stays_inside_its_allowance() {
        // 0 until `nemotron_glue::topk_f32` lands; then it must not exceed what the model prices (or the model takes the larger)
        let need = topk_workspace_bytes(64, 131_072, MAX_CANDIDATES) as u64;
        let s = mem().scratch(64, 262_144);
        assert!(s.topk_workspace >= need);
        if need > 64 * TOPK_ALLOWANCE_PER_ROW {
            eprintln!(
                "note: the top-k kernel needs {need} B, over the {} B allowance: the model already takes the larger",
                64 * TOPK_ALLOWANCE_PER_ROW
            );
        }
    }

    // ---- zero-priced shapes are refused by name

    fn refused(c: &OmniConfig, o: &MemoryOptions) -> MemoryError {
        NemotronMemory::new(c, 0).with_kv_dtype(KvDtype::F16).profile(o).unwrap_err()
    }

    #[test]
    fn a_shape_that_prices_a_token_at_zero_is_refused_naming_the_family_and_the_plane() {
        for edit in [|c: &mut OmniConfig| c.head_dim = 0, |c: &mut OmniConfig| c.kv_heads = 0] {
            let mut c = real();
            edit(&mut c);
            let e = refused(&c, &opts());
            assert!(matches!(&e, MemoryError::Zero(m) if m.contains("nemotron_h_omni") && m.contains("plane `k`")), "{e}");
        }
        // a hybrid with no attention layer prices no KV at all
        let mut c = real();
        c.pattern = c.pattern.replace('*', "M");
        let e = refused(&c, &opts());
        assert!(
            matches!(&e, MemoryError::Zero(m) if m.contains("nemotron_h_omni") && m.contains("`gqa_nope`") && m.contains("no layers")),
            "{e}"
        );
    }

    #[test]
    fn a_mamba_layer_with_a_zero_dimension_is_refused_by_name_and_a_pack_without_them_is_consistent() {
        type Edit = fn(&mut OmniConfig);
        let edits: [(Edit, &str); 7] = [
            (|c| c.mamba_heads = 0, "Mamba heads"),
            (|c| c.mamba_dim = 0, "Mamba head dim"),
            (|c| c.groups = 0, "Mamba groups"),
            (|c| c.state = 0, "SSM state size"),
            (|c| c.conv = 0, "conv kernel"),
            (|c| c.hidden = 0, "hidden size"),
            (|c| c.vocab = 0, "vocabulary"),
        ];
        for (edit, needle) in edits {
            let mut c = real();
            edit(&mut c);
            let e = refused(&c, &opts());
            assert!(matches!(&e, MemoryError::Zero(m) if m.contains("nemotron_h_omni") && m.contains(needle)), "{needle}: {e}");
        }
        // no Mamba layer at all: no recurrent state and no window records, which the shared check accepts for a pure-attention shape
        let mut c = real();
        c.pattern = c.pattern.replace('M', "E");
        let p = NemotronMemory::new(&c, 0).with_kv_dtype(KvDtype::F16).profile(&opts()).unwrap();
        assert_eq!((p.recurrent_layers(), p.state_bytes_per_seq(), p.window_record_bytes_per_row()), (0, 0, 0));
    }

    #[test]
    fn a_step_shape_the_engine_cannot_run_is_refused_by_name() {
        let c = real();
        for (o, needle) in [
            (MemoryOptions { window_rows: 0, ..opts() }, "window rows"),
            (MemoryOptions { window_rows: MAX_WINDOW_ROWS + 1, ..opts() }, "257 rows"),
            (MemoryOptions { prefill_rows: 0, ..opts() }, "prefill rows"),
            (MemoryOptions { prefill_rows: usize::MAX, ..opts() }, "not a size this model prices"),
            (MemoryOptions { max_context: 0, ..opts() }, "max context"),
            // 2^14 chunks of 512 keys is the kernel scratch's limit
            (MemoryOptions { max_context: 8_388_609, ..opts() }, "attention chunks"),
        ] {
            let e = refused(&c, &o);
            assert!(e.to_string().contains("nemotron_h_omni") && e.to_string().contains(needle), "{needle}: {e}");
        }
        assert!(NemotronMemory::new(&c, 0)
            .with_kv_dtype(KvDtype::F16)
            .profile(&MemoryOptions { max_context: 8_388_608, ..opts() })
            .is_ok());
    }

    #[test]
    fn the_budget_table_adds_up_and_prints_for_the_milestone_shapes() {
        // `--nocapture` prints what the first real allocation on node 1 (task I1) is compared to
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        for (max_context, slots) in [(262_144usize, 2usize), (1 << 20, 1), (1 << 20, 2)] {
            let o = MemoryOptions { max_context, ..opts() };
            let kv_tokens = (max_context * slots) as u64;
            let b = NemotronMemory::new(&real(), 0).with_kv_dtype(KvDtype::F16).budget(&o, slots, kv_tokens).unwrap();
            eprintln!(
                "ctx {:>8} slots {slots}: state {:>8.1} MiB ({:.1} per slot), kv {:>8.1} MiB, records {:>6.1} MiB, scratch {:>7.1} MiB \
                 [buffers {:.1}, topk {:.1}, moe {:.1}, attn {:.1}, exl3 {:.1}], reserve {:>7.1} MiB",
                max_context,
                mib(b.state),
                mib(b.state_per_slot),
                mib(b.kv),
                mib(b.window_records),
                mib(b.scratch.total()),
                mib(b.scratch.activations),
                mib(b.scratch.topk_workspace),
                mib(b.scratch.moe),
                mib(b.scratch.attn_partials),
                mib(b.scratch.exl3_workspace),
                mib(b.engine_reserve),
            );
            assert_eq!(b.kv, kv_tokens * b.kv_per_token, "whole pages at these sizes");
            assert_eq!(b.engine_reserve, ACTIVATION_RESERVE.max(b.scratch.total() + b.window_records));
            assert_eq!(b.total(), b.state + b.kv + b.engine_reserve);
        }
    }

    // ---- the MTP head

    fn mtp_opts() -> MemoryOptions {
        MemoryOptions { mtp: true, ..opts() }
    }

    #[test]
    fn the_head_is_priced_line_by_line_and_the_numbers_are_exact() {
        let m = mem();
        let h = m.mtp_head();
        // the weights: eh_proj 4096 x 8192, attention (q, o 4096 x 4096; k, v 256 x 4096), router 512 x 4096, two latent projections
        // 1024 x 4096, the shared expert 2 x 5376 x 4096, 512 experts x 2 x 2688 x 1024, all bf16, and five norms
        let elems: u64 = 4096 * 8192
            + 2 * 4096 * 4096
            + 2 * 256 * 4096
            + 512 * 4096
            + 2 * 1024 * 4096
            + 2 * 5376 * 4096
            + 512 * 2 * 2688 * 1024;
        // and the router's f32 correction bias, 512 x 4
        assert_eq!(h.weights, (elems + 5 * 4096) * 2 + 512 * 4);
        assert_eq!(h.weights, 5_884_651_520, "5,612 MiB: the head's device bytes MT-A measured (mtp_weights)");
        assert_eq!(h.weights >> 20, 5_612);
        assert!((h.weights as f64 / 1e9 - 5.88).abs() < 0.01);
        // one attention layer (by spec, not by the head's two layers), K and V, 2 heads x 128, fp16
        assert_eq!(h.kv_per_token, 2 * 2 * 128 * 2);
        assert_eq!(h.kv_per_token, 1024);
        // 32 rows of 4,096 f32 a slot; 64 rows once; 64 x (8,192 + 4 x 4,096) f32 once
        assert_eq!(h.tap_store_per_slot, 32 * 4096 * 4);
        assert_eq!(h.tap_store_per_slot, 524_288);
        assert_eq!(h.tap_window, 64 * 4096 * 4);
        assert_eq!(h.tap_window, 1_048_576);
        assert_eq!(h.activations, 64 * (8192 + 4096 * 4) * 4);
        assert_eq!(h.activations, 6_291_456);
        // the draft outputs: per row an id, 7 depths of (id, value) (the deepest chain of one call) and a 48-byte keyed record
        assert_eq!(MTP_MAX_DEPTH, 7);
        assert_eq!(h.drafts, 64 * (4 + 7 * 8 + 48));
        assert_eq!(h.drafts, 6_912);
        assert_eq!(h.sub_head, 0, "the full head has no sub-head table");
        assert_eq!(h.scratch(), 1_048_576 + 6_291_456 + 6_912);
        assert_eq!(h.scratch(), 7_346_944, "about 7 MiB, MT-A's 'window taps 1 MiB, scratch 6 MiB'");
        // the restricted head's table: n rows of 4,096 bf16 and an i32 id each (MT-A: 256 MiB at 32k, 384 at 48k, 512 at 64k)
        let r = NemotronMemory::new(&real(), 0).with_kv_dtype(KvDtype::F16).with_restricted_head(32_768).mtp_head();
        assert_eq!(r.sub_head, 32_768 * (4096 * 2 + 4));
        assert_eq!(r.sub_head, 268_566_528);
        assert_eq!((r.sub_head >> 20, (r.sub_head - (32_768 * 4)) >> 20), (256, 256));
        assert_eq!(r.scratch(), h.scratch() + 268_566_528);
        for (rows, mib) in [(49_152usize, 384u64), (65_536, 512)] {
            let b = NemotronMemory::new(&real(), 0).with_kv_dtype(KvDtype::F16).with_restricted_head(rows).mtp_head().sub_head;
            assert_eq!((b - rows as u64 * 4) >> 20, mib, "{rows} ids");
        }
        // the head's buffers are what the model sizes by its widest step: `max(prefill_rows, 64)` rows of activations and drafts, and
        // the window tap buffer stays at 64 rows
        let wide = m.mtp_head_for(4096);
        assert_eq!((wide.tap_window, wide.tap_store_per_slot), (h.tap_window, h.tap_store_per_slot));
        assert_eq!(wide.activations, 64 * h.activations);
        assert_eq!(wide.drafts, 64 * h.drafts);
        assert_eq!(m.mtp_head_for(16), h, "a step narrower than the window is the window");
        assert_eq!((r.weights, r.kv_per_token, r.tap_store_per_slot), (h.weights, h.kv_per_token, h.tap_store_per_slot));
    }

    /// P2-10: the head's weights line is what the loader holds in the pack's forms when the plan gives it: for the published bf16 head
    /// the plan's figure IS the formula above, and the MixedK pack's head (EXL3 throughout, 4 bits) is its trellises and scales as the
    /// bridge keeps them (fp16 in block 0, f32 in the MoE block's tables, no marker).
    #[test]
    fn the_heads_weights_line_follows_the_packs_forms() {
        use super::super::weights::{head_projections, units, units_with, HeadExl3, LayerPlan, Storages, UnitId, Widths};
        let c = real();
        let head = |u: UnitId| matches!(u, UnitId::Mtp(_));
        let bf16 = LayerPlan { units: units(&c, &Storages::new()), outside_plan: 0 };
        let held = ShardPlan::single().rank_bytes(&bf16, c.experts, head);
        let formula = NemotronMemory::new(&c, 0).mtp_head().weights;
        assert_eq!(held, formula, "the bf16 head: the loader's figure is the formula");
        assert_eq!(NemotronMemory::new(&c, 0).with_head_weight_bytes(held).mtp_head().weights, 5_884_651_520);
        let all: HeadExl3 = head_projections(&c).into_iter().map(|(p, _, _)| p).collect();
        let mixedk = LayerPlan { units: units_with(&c, &Storages::new(), &all, &Widths::new()), outside_plan: 0 };
        let held = ShardPlan::single().rank_bytes(&mixedk, c.experts, head);
        let (h, lat, inter, sh) = (c.hidden as u64, c.latent as u64, c.expert_inter as u64, c.shared_inter as u64);
        let (q, kv) = ((c.heads * c.head_dim) as u64, (c.kv_heads * c.head_dim) as u64);
        let exl3 = |k: u64, n: u64, scale: u64| k * n / 2 + (k + n) * scale;
        let want = 5 * h * 2
            + c.experts as u64 * h * 2
            + c.experts as u64 * 4
            + exl3(2 * h, h, 2)
            + exl3(h, q, 2)
            + exl3(q, h, 2)
            + 2 * exl3(h, kv, 2)
            + exl3(h, lat, 4)
            + exl3(lat, h, 4)
            + exl3(h, sh, 4)
            + exl3(sh, h, 4)
            + c.experts as u64 * (exl3(lat, inter, 4) + exl3(inter, lat, 4));
        assert_eq!(held, want);
        let m = NemotronMemory::new(&c, 0).with_head_weight_bytes(held);
        assert_eq!(m.mtp_head().weights, want);
        assert!(want < formula / 3, "a 4-bit head is under a third of the bf16 one: {want} of {formula}");
        // only the weights line moves
        let (a, b) = (NemotronMemory::new(&c, 0).mtp_head(), m.mtp_head());
        assert_eq!((a.scratch(), a.kv_per_token, a.tap_store_per_slot), (b.scratch(), b.kv_per_token, b.tap_store_per_slot));
    }

    /// An EXL3 `lm_head`'s logits temp at a TP rank is its share of the 128-column-block cut: at two ranks the bf16 head's, at three
    /// 43,776 rows a rank (the bf16 cut's 43,692 would be short).
    #[test]
    fn an_exl3_lm_heads_tp_logits_temp_is_its_block_cut_share() {
        let c = real();
        for ranks in [2usize, 3] {
            let bf16 = NemotronMemory::new(&c, 0).with_tp_ranks(ranks).tp_scratch_bytes(64, true);
            let exl3 = NemotronMemory::new(&c, 0).with_tp_ranks(ranks).with_lm_head_exl3(true).tp_scratch_bytes(64, true);
            let extra = if ranks == 2 { 0 } else { 64 * (43_776 - 43_692) * 4 };
            assert_eq!(exl3, bf16 + extra, "{ranks} ranks");
        }
    }

    #[test]
    fn the_profile_counts_exactly_the_head_lines_when_it_drafts_and_nothing_when_it_does_not() {
        let m = mem();
        let (off, on) = (m.profile(&opts()).unwrap(), m.profile(&mtp_opts()).unwrap());
        let h = m.mtp_head();
        assert_eq!(on.kv_bytes_per_token() - off.kv_bytes_per_token(), h.kv_per_token);
        assert_eq!(off.drafter_state_bytes_per_slot(), 0);
        assert_eq!(on.drafter_state_bytes_per_slot(), h.tap_store_per_slot);
        assert_eq!(on.state_bytes_per_slot() - off.state_bytes_per_slot(), 524_288);
        assert_eq!(on.scratch_bytes() - off.scratch_bytes(), h.scratch());
        assert_eq!(on.state_bytes_per_seq(), off.state_bytes_per_seq(), "the head has no recurrent state");
        assert_eq!(on.window_record_bytes(), off.window_record_bytes());
        // the head's weights are the caller's `weight_bytes`: the profile does not move with them, and the budget does not add them twice
        let w = NemotronMemory::new(&real(), 60 * GIB).with_kv_dtype(KvDtype::F16);
        assert_eq!(w.profile(&mtp_opts()).unwrap(), on);
        let b = w.budget(&mtp_opts(), 2, 2 << 20).unwrap();
        assert_eq!(b.mtp_head, Some(h));
        assert_eq!(b.weights, 60 * GIB);
        assert_eq!(b.state_per_slot, 170_229_760 + 524_288);
        assert_eq!(b.state, 2 * (170_229_760 + 524_288));
        assert_eq!(b.kv_per_token, 9216);
        assert_eq!(b.scratch.mtp_head, h.scratch());
        assert_eq!(b.scratch.total(), on.scratch_bytes());
        assert_eq!(b.total(), 60 * GIB + b.state + b.kv + b.engine_reserve);
        assert!(w.budget(&opts(), 2, 2 << 20).unwrap().mtp_head.is_none());
        // the shared MoE scratch is NOT counted again for the head
        assert_eq!(m.scratch_with(64, 262_144, true).moe, m.scratch(64, 262_144).moe);
        // the head's buffers follow the model's widest step (at least its 64-row window), as `MtpState::new` sizes them
        for rows in [1u64, 8, 64] {
            assert_eq!(m.scratch_with(rows, 262_144, true).mtp_head, h.scratch(), "{rows} rows");
        }
        assert_eq!(m.scratch_with(4096, 262_144, true).mtp_head, m.mtp_head_for(4096).scratch());
        assert!(m.scratch_with(4096, 262_144, true).mtp_head > 50 * h.scratch());
        // a restricted sub-head is priced only with the head
        let r = NemotronMemory::new(&real(), 0).with_kv_dtype(KvDtype::F16).with_restricted_head(32_768);
        assert_eq!(r.profile(&mtp_opts()).unwrap().scratch_bytes() - on.scratch_bytes(), 268_566_528);
        assert_eq!(r.profile(&opts()).unwrap(), off);
    }

    #[test]
    fn the_head_adds_a_tap_store_to_every_slot_admission_charges() {
        // admission's cost of a sequence is its KV plus its slot's state: both grow with the head
        let (off, on) = (mem().profile(&opts()).unwrap(), mem().profile(&mtp_opts()).unwrap());
        let tokens = 1000u64;
        let cost = |p: &MemoryProfile| tokens.div_ceil(PAGE_TOKENS) * p.kv_page_bytes() + p.state_bytes_per_slot();
        assert_eq!(cost(&on) - cost(&off), 4 * PAGE_TOKENS * 1024 + 524_288);
    }

    #[test]
    fn a_head_shape_that_prices_a_part_of_the_head_at_zero_is_refused_by_name() {
        type Edit = fn(&mut OmniConfig);
        let edits: [(Edit, &str); 5] = [
            (|c| c.heads = 0, "attention heads"),
            (|c| c.experts = 0, "routed experts"),
            (|c| c.expert_inter = 0, "expert width"),
            (|c| c.latent = 0, "latent width"),
            (|c| c.shared_inter = 0, "shared expert width"),
        ];
        for (edit, needle) in edits {
            let mut c = real();
            edit(&mut c);
            // not drafting: the trunk's own dimensions the profile checks are untouched, so it prices
            let e = NemotronMemory::new(&c, 0).with_kv_dtype(KvDtype::F16).profile(&mtp_opts()).unwrap_err();
            assert!(
                matches!(&e, MemoryError::Zero(m) if m.contains("nemotron_h_omni") && m.contains(needle) && m.contains("MTP head")),
                "{needle}: {e}"
            );
        }
        // a restricted head wider than the vocabulary is inconsistent
        let e = NemotronMemory::new(&real(), 0)
            .with_kv_dtype(KvDtype::F16)
            .with_restricted_head(131_073)
            .profile(&mtp_opts())
            .unwrap_err();
        assert!(matches!(&e, MemoryError::Inconsistent(m) if m.contains("restricted draft head") && m.contains("131073")), "{e}");
        assert!(NemotronMemory::new(&real(), 0)
            .with_kv_dtype(KvDtype::F16)
            .with_restricted_head(131_072)
            .profile(&mtp_opts())
            .is_ok());
        // a pack with no head cannot be asked to price one (it did before; still does)
        let mut c = real();
        c.mtp_pattern = String::new();
        assert!(NemotronMemory::new(&c, 0).with_kv_dtype(KvDtype::F16).profile(&mtp_opts()).is_err());
    }

    #[test]
    fn the_tap_store_rows_are_the_number_the_backend_assumes() {
        assert_eq!(MTP_TAP_ROWS, 32);
        assert_eq!(mem().mtp_head().tap_store_per_slot, MTP_TAP_ROWS as u64 * 4096 * 4);
    }

    // ---- the prefix cache

    #[test]
    fn the_snapshot_and_the_page_are_priced_line_by_line() {
        let m = mem();
        // 40 Mamba layers x (S f32 128 x 128 x 64 + conv bf16 3 x 10,240)
        assert_eq!(m.snapshot_state_bytes(), 40 * (4_194_304 + 61_440));
        assert_eq!(m.snapshot_state_bytes(), 170_229_760);
        let seq = m.profile(&opts()).unwrap().state_bytes_per_seq();
        assert_eq!(m.snapshot_state_bytes(), seq, "the profile's recurrent state is the snapshot's");
        // the head adds its carry: one f32 tap row of the hidden width
        assert_eq!(m.snapshot_bytes(false), 170_229_760);
        assert_eq!(m.snapshot_bytes(true), 170_229_760 + 4096 * 4);
        assert_eq!(m.snapshot_bytes(true), 170_246_144);
        // a page: 8 attention layers x K and V x 2 heads x 128 x fp16 x 256 tokens, and the head's stripe
        assert_eq!(m.page_bytes(false), 8 * 2 * 2 * 128 * 2 * 256);
        assert_eq!(m.page_bytes(false), 2 << 20);
        assert_eq!(m.page_bytes(true), 9 * 2 * 2 * 128 * 2 * 256);
        assert_eq!(m.page_bytes(true), 2_359_296, "2.25 MiB");
        // and it is the profile's page, which is what the pool is made of
        assert_eq!(m.page_bytes(false), m.profile(&opts()).unwrap().kv_page_bytes());
        assert_eq!(m.page_bytes(true), m.profile(&mtp_opts()).unwrap().kv_page_bytes());
        // an entry: its slot and every page it references, whole pages
        assert_eq!(m.entry_bytes(1, true), 170_246_144 + 2_359_296);
        assert_eq!(m.entry_bytes(256, true), 170_246_144 + 2_359_296);
        assert_eq!(m.entry_bytes(257, true), 170_246_144 + 2 * 2_359_296);
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        let table = [
            (512u64, 166.9),
            (2_048, 180.4),
            (8_192, 234.4),
            (32_768, 450.4),
            (131_072, 1_314.4),
            (524_288, 4_770.4),
            (1_048_560, 9_378.4),
        ];
        for (tokens, want) in table {
            let got = mib(m.entry_bytes(tokens, true));
            assert!((got - want).abs() < 0.06, "{tokens}: {got:.1} MiB");
        }
        // a snapshot costs about as much as the KV of this many tokens: below it the state is most of an entry
        assert_eq!(m.snapshot_bytes(true) / (m.page_bytes(true) / PAGE_TOKENS), 18_472);
    }

    #[test]
    fn the_source_is_object_safe_and_send_sync_for_the_family_plan() {
        fn needs(_: Box<dyn MemorySource>) {}
        needs(Box::new(mem()));
    }
}
