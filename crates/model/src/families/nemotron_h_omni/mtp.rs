//! The MTP head on the device: its weights, its KV layer, the per-slot tap store and the draft call (milestone M2, task MT-A fills this
//! file). The types at the top are frozen: the backend (task MT-B) is written against them and the fake runtime of its tests.
//!
//! Contract: `docs/NEMOTRON_MTP_CONTRACT.md`. Design and numbers: `docs/NEMOTRON_MTP_PLAN.md`, `docs/NEMOTRON_MTP_RESEARCH.md`. The head
//! only proposes: the verify window decides every committed token, so drafted == serial whatever this file does; what it can break is
//! acceptance and the drafter's own state, never the trunk's.
//!
//! **The recipe** (`mtp_ref`, vLLM's `NemotronHMTP`): entry `i` of the head pairs the trunk's POST-`norm_f` row at position `i` with the
//! token at `i + 1`: `x0 = eh_proj([enorm(embed(t_{i+1})) | hnorm(h_i)])`, `x1 = x0 + attn(norm(x0))` on the head's own KV layer (NoPE, the
//! target's positions), `x2 = x1 + moe(norm(x1))`, `out = final_layernorm(x2)`, `logits = lm_head(out)` with the TRUNK's `lm_head`. A chain
//! feeds the draft back as the token and `out` back as the hidden. Everything is f32 between kernels, as the trunk's activations are;
//! the head's weights are the pack's bf16 bytes in place.
//!
//! **State.** The head's attention cache is one more attention layer in the same KV pool (`kv_geom.layers - 1`: same block table, same
//! positions, 1 KiB per token). Per slot, the trunk's committed windows leave their taps in a small f32 store ([`TapBook`] is the
//! bookkeeping, pure and tested on the CPU): the rows whose entry is not built yet. A draft call absorbs them (one head window over all
//! pending rows of all asking sequences, only the last row of each needs `lm_head`) and chains `depth - 1` more single rows, all
//! asynchronous on the stream with one sync at the end. The draft state therefore advances only with committed rows: a window's rejected
//! rows never enter the store, and the speculative entries a chain writes are overwritten by the next absorb from the true taps.
//!
//! **Indexing.** [`TapBook`] is entry-indexed as the GLM drafter's is: entry `m` pairs the tap `h_{m-1}` with the token `t_m`, so
//! `first + n == tokens.len()` at a draft call. This head has no entry 0 (nothing predicted `t_0`; GLM builds one from a zero row) and
//! stores entry `m` at the cache position `m - 1`, the position of its tap, which is where `mtp_ref` puts it.

use super::device::{HeadLinear, MtpW};
use super::model::{chunk_bucket, linear_rows, CommitSpan, Core, NemotronModel, SeqPlan, WINDOW_ROWS};
use super::tapbook::{BookError, TapBook};
use super::vision::SpliceOp;
use crate::error::ModelError;
use crate::rows::RowBatch;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::glm_mtp::{add_f32, keyed_pick, KeyedRow};
use gb_cuda::glm_perf::map_ids;
use gb_cuda::graph::{CaptureError, Graph};
use gb_cuda::moe::EventTimer;
use gb_cuda::nemotron_attn::{kv_write, NemoKv, RowIndex, CHUNK_KEYS, KV_TOKEN_ELEMS};
use gb_cuda::nemotron_attn_prefill::attn_long;
use gb_cuda::nemotron_dense::gemv_bf16;
use gb_cuda::nemotron_glue::{embed_gather_f32, rmsnorm_bf16w_f32, splice_rows, topk_f32, HIDDEN};
use gb_cuda::{DevBuf, Stream};
use gb_sampling::Sampling;
use std::collections::{BTreeMap, HashMap};

/// The deepest chain one draft call runs (positions drafted after the sequence's last token). The engine's own cap is 7 drafts a round
/// (`engine::MAX_DRAFTS`); a deeper chain costs one head step more (about 4 ms) and a verify row, so the arbiter holds the depth where the
/// stream's measured acceptance pays for it (`spec::arbiter_config`).
pub const MTP_MAX_DEPTH: usize = 7;

/// One sequence's ask of a draft call.
#[derive(Clone, Copy, Debug)]
pub struct DraftAsk<'a> {
    pub slot: usize,
    /// The sequence so far, ending with the one token not yet fed to the trunk.
    pub tokens: &'a [u32],
    /// Drafts wanted, `0..=MTP_MAX_DEPTH`. Zero only absorbs the slot's pending rows (to make room in its tap store) and drafts nothing.
    pub depth: usize,
}

/// One sequence's drafts and the drafter's confidence in each (the head's logit value of the pick: the argmax, or the keyed draw's token).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Drafts {
    pub ids: Vec<u32>,
    pub vals: Vec<f32>,
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Additive items (everything above is frozen).

/// Pending committed rows one slot's tap store holds (`[32][4096]` f32 per slot). More than that switches the slot's drafting off: a
/// sequence that never asks for drafts must be absorbed (a depth-0 ask) before its store fills.
pub const MTP_TAP_CAP: usize = 32;

/// The most candidates a keyed draft call fetches per row (the trunk's candidate buffers hold this many).
pub const MTP_MAX_CANDIDATES: usize = super::model::MAX_CANDIDATES;

/// A deliberate bug in the drafter, so the tests can show they notice it. Production sets none.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpMutant {
    /// The tap is the residual stream BEFORE `norm_f` (the trunk's last-layer output) instead of `norm_f`'s output.
    TapBeforeNorm,
    /// A committed window's tapped rows all enter the store, kept or not (the draft state advances on rejected rows).
    AdvanceAllRows,
    /// The hidden input of an entry is the tap of the token's OWN position instead of the position before it (the shift is dropped), in
    /// the prompt chunks the head builds its cache from.
    NoShift,
    /// A draft call absorbs only the LAST pending row of each sequence: the entries a previous call wrote speculatively (from the head's
    /// own hidden state, for drafts that were then accepted) are never rewritten from the target's true rows (no rollback of the state).
    AbsorbLastOnly,
    /// Every draft is the id after the one the head chose: numerically clean and always wrong (acceptance about zero).
    WrongDrafts,
    /// The restricted head's argmax is not mapped from its column to the token id.
    RestrictedNoMap,
    /// An image's rows enter the head's cache with the placeholder token's embedding, not the image's features (the prompt chunks' splice is
    /// not applied to the head's embedding half).
    PlaceholderImages,
}

/// The restricted draft vocabulary the model loads with: `off` (also `0`, `full`, `none`, `false`) keeps the head over the trunk's whole
/// 131,072-token `lm_head`, a path names a file of token ids (`tools/nemotron_omni/mtp_vocab.py vocab`) that replaces the default list,
/// and unset (or `default`, `on`) is [`DEFAULT_VOCAB`].
pub const VOCAB_ENV: &str = "REXL3_MTP_VOCAB";

/// The default restricted draft vocabulary: 49,152 of the tokenizer's 131,072 ids (`mtp_vocab_48k.txt`: every special token and the
/// most frequent pure-ASCII tokens by the recipe of `tools/nemotron_omni/mtp_vocab.py`, its header recording the corpus and the counts).
/// It is derived from the tokenizer's vocabulary and this repository's own text, never from a weight. A draft only proposes: a restricted
/// head can lose acceptance (text in a script the list leaves out) and never a byte of output.
pub const DEFAULT_VOCAB: &str = include_str!("mtp_vocab_48k.txt");

/// What `REXL3_MTP_VOCAB` asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VocabSetting {
    /// [`DEFAULT_VOCAB`].
    Default,
    /// The full head.
    Off,
    /// A file of token ids.
    File(std::path::PathBuf),
}

impl VocabSetting {
    /// The setting of the variable's text (unset and empty are the default).
    pub fn parse(v: Option<&str>) -> Self {
        match v.map(str::trim) {
            None | Some("") | Some("default") | Some("on") => VocabSetting::Default,
            Some("off") | Some("0") | Some("full") | Some("none") | Some("false") => VocabSetting::Off,
            Some(path) => VocabSetting::File(path.into()),
        }
    }

    /// The process environment's setting.
    pub fn from_env() -> Self {
        Self::parse(std::env::var(VOCAB_ENV).ok().as_deref())
    }
}

/// The ids of the list a setting names: sorted and de-duplicated, never empty; `None` for the full head. The model builds its sub-head
/// from exactly these (`NemotronModel::load`) and the plan prices them (`engine::restricted_head_rows`): one function, so what is priced
/// is what is allocated.
pub fn restricted_vocab_ids_for(setting: &VocabSetting) -> Result<Option<Vec<u32>>, ModelError> {
    let (name, text) = match setting {
        VocabSetting::Off => return Ok(None),
        VocabSetting::Default => (format!("{VOCAB_ENV} (the default list)"), std::borrow::Cow::Borrowed(DEFAULT_VOCAB)),
        VocabSetting::File(p) => {
            let name = format!("{VOCAB_ENV}={}", p.display());
            let text = std::fs::read_to_string(p).map_err(|e| ModelError::Config(format!("{name}: {e}")))?;
            (name, std::borrow::Cow::Owned(text))
        }
    };
    let mut ids = parse_vocab_file(&text).map_err(|e| ModelError::Config(format!("{name}: {e}")))?;
    ids.sort_unstable();
    ids.dedup();
    if ids.is_empty() {
        return Err(ModelError::Config(format!("{name}: the restricted draft vocabulary is empty")));
    }
    Ok(Some(ids))
}

/// [`restricted_vocab_ids_for`] the process environment's setting.
pub fn restricted_vocab_ids() -> Result<Option<Vec<u32>>, ModelError> {
    restricted_vocab_ids_for(&VocabSetting::from_env())
}

/// The head's candidates at one chain depth (token ids, logits), by value descending and id ascending (`NemotronModel::mtp_debug_chain`).
pub type HeadList = (Vec<u32>, Vec<f32>);

/// What a debug chain feeds to its next step: the token chosen from `(depth, ids, values)` of the depth just run.
pub type ChainPick<'a> = &'a mut dyn FnMut(usize, &[u32], &[f32]) -> u32;

/// The head's own copy of some rows of the trunk's `lm_head`: the head over them reads that share of its 1.07 GB. Every kept row is a
/// bit-identical copy, so a restricted draft is the full head's whenever the full head's argmax lies inside the set.
pub struct SubHead {
    /// bf16 `[cols, 4096]`: the chosen `lm_head` rows, in ascending token id.
    w: DevBuf,
    /// i32 `[cols]`: the token id of each column.
    map: DevBuf,
    cols: usize,
    /// Per token id of the trunk's vocabulary: in the set.
    member: Vec<bool>,
}

impl SubHead {
    /// The share of `ids` that are tokens of the set.
    pub fn coverage(&self, ids: &[u32]) -> f32 {
        if ids.is_empty() {
            return 1.0;
        }
        ids.iter().filter(|&&t| self.member.get(t as usize).copied().unwrap_or(false)).count() as f32 / ids.len() as f32
    }

    /// Tokens in the set.
    pub fn len(&self) -> usize {
        self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.cols == 0
    }

    /// Device bytes of the copy and its id map.
    pub fn bytes(&self) -> u64 {
        (self.w.bytes() + self.map.bytes()) as u64
    }
}

/// The head's device state and scratch.
pub struct MtpState {
    /// f32 `[window_rows][4096]` (`WINDOW_ROWS`, or a wide model's decode window): the last window's post-`norm_f` rows, copied at the end of the window's forward. Valid exactly as long as
    /// the window can be committed (the next window overwrites both).
    win_tap: DevBuf,
    /// f32 `[slots][MTP_TAP_CAP][4096]`: per slot, the pending tapped rows (row 0 is the carry across prefill chunks).
    store: DevBuf,
    /// f32 `[rows][4096]` each: the head's residual, its normed input (also the embedding gather's landing), and its final-normed rows.
    x: DevBuf,
    xn: DevBuf,
    out: DevBuf,
    /// f32 `[rows][8192]`: `[enorm(embed) | hnorm(hidden)]`.
    cat: DevBuf,
    /// f32 `[rows][4096]`: the hidden rows an absorb reads (gathered from the stores and the prompt chunk's taps).
    h_in: DevBuf,
    /// i32 `[rows]`: the token ids of an absorb.
    ids: DevBuf,
    /// Per chain depth: the picked ids (i32) and their logit values (f32) of every asking sequence, `[rows]`.
    draft_ids: Vec<DevBuf>,
    draft_vals: Vec<DevBuf>,
    /// `KeyedRow` `[rows]`: the sampling record of each sequence of the call being run.
    keyed: DevBuf,
    /// The index arrays of a draft call (`MTP_MAX_DEPTH + 1` of them: the absorb's rows, its last rows, one a chain step), allocated once
    /// and rewritten in the stream's order by every call (a call used to allocate and free them: host time with the device idle).
    idx_pool: Vec<RowIndex>,
    /// Captured draft calls of ONE sequence (`REXL3_NEMO_DRAFT_GRAPH`), by shape: see [`DraftKey`].
    draft_graphs: HashMap<DraftKey, DraftSlot>,
    draft_tick: u64,
    /// Draft calls that captured a graph (`[0]`), replayed one (`[1]`), and ran eagerly (`[2]`: a first sighting, a failed capture, or more than
    /// one sequence).
    pub draft_graph_stats: [u64; 3],
    sub: Option<SubHead>,
    /// Whether draft calls use the restricted head (when one was built).
    use_sub: bool,
    book: Vec<TapBook>,
    /// The image rows of the absorb being run, in the head's own rows (`dst_row` is a row of the absorb): their embedding half is the
    /// image's feature row, written over the gather before `enorm`. Set by the prompt hook (and the teacher-forcing debug forward) and
    /// cleared when the call ends: a draft call's absorb never has any (the head's pending rows are text, `spec_allowed_for_window`).
    img: Vec<SpliceOp>,
    pub mutant: Option<MtpMutant>,
    /// Debug: a draft call leaves the slots' pending rows pending (so the same call can be repeated: a determinism check).
    keep_pending: bool,
    /// Debug: milliseconds by stage over the calls since profiling was switched on.
    profile: Option<BTreeMap<&'static str, f64>>,
    /// Device bytes of the head's weights (the pack's tensors in place).
    weight_bytes: u64,
    rows_cap: usize,
    /// Tensor parallelism: the head's lm_head is scored by half of its rows on each rank and combined by one allreduce
    /// (`tp_arm::HeadSplit::lm_head`, read once at construction: a model's collectives must not change under it).
    split_lm_head: bool,
    /// The tap dump only (`tapdump`, off by default): the token ids, slots and positions of the last window's rows, for its committed
    /// rows' records (a commit whose rows they do not describe writes nothing and says so).
    pub(super) dump_ids: Vec<i32>,
    pub(super) dump_slots: Vec<i32>,
    pub(super) dump_pos: Vec<i32>,
}

/// What a debug forward of head rows captured (the gate against `mtp_ref`): the last `tail` rows of each stage, f32.
#[derive(Default, Debug)]
pub struct MtpCapture {
    /// `eh_proj`'s output.
    pub fused: Vec<f32>,
    /// `fused + attention(norm(fused))`.
    pub after_attn: Vec<f32>,
    /// `after_attn + moe(norm(after_attn))`: the block's output on the residual.
    pub block: Vec<f32>,
    /// `final_layernorm(block)`.
    pub out: Vec<f32>,
    /// `lm_head(out)`, `[tail][vocab]`.
    pub logits: Vec<f32>,
}

impl MtpState {
    /// `rows_cap`: the rows of the trunk's widest forward (`max(prefill_rows, WINDOW_ROWS)`).
    pub fn new(slots: usize, rows_cap: usize, weight_bytes: u64) -> Result<Self, ModelError> {
        Self::with_window(slots, rows_cap, WINDOW_ROWS, weight_bytes)
    }

    /// [`MtpState::new`] for a trunk whose decode window takes `window_rows` rows (at least `WINDOW_ROWS`): the window tap buffer holds
    /// that many. `rows_cap` must be at least `window_rows`.
    pub fn with_window(slots: usize, rows_cap: usize, window_rows: usize, weight_bytes: u64) -> Result<Self, ModelError> {
        let window_rows = window_rows.max(WINDOW_ROWS);
        let r = rows_cap.max(WINDOW_ROWS).max(window_rows);
        let mut draft_ids = Vec::new();
        let mut draft_vals = Vec::new();
        for _ in 0..MTP_MAX_DEPTH {
            draft_ids.push(DevBuf::zeroed(r * 4)?);
            draft_vals.push(DevBuf::zeroed(r * 4)?);
        }
        Ok(Self {
            // one window of the trunk's rows (64, or a wide model's decode window)
            win_tap: DevBuf::zeroed(window_rows * HIDDEN * 4)?,
            store: DevBuf::zeroed(slots.max(1) * MTP_TAP_CAP * HIDDEN * 4)?,
            x: DevBuf::zeroed_view(r * HIDDEN * 4)?,
            xn: DevBuf::zeroed_view(r * HIDDEN * 4)?,
            out: DevBuf::zeroed_view(r * HIDDEN * 4)?,
            cat: DevBuf::zeroed_view(r * 2 * HIDDEN * 4)?,
            h_in: DevBuf::zeroed(r * HIDDEN * 4)?,
            ids: DevBuf::zeroed(r * 4)?,
            draft_ids,
            draft_vals,
            keyed: DevBuf::zeroed(r * std::mem::size_of::<KeyedRow>())?,
            idx_pool: (0..MTP_MAX_DEPTH + 1).map(|_| RowIndex::with_capacity(r)).collect::<Result<Vec<_>, _>>()?,
            draft_graphs: HashMap::new(),
            draft_tick: 0,
            draft_graph_stats: [0; 3],
            sub: None,
            use_sub: false,
            book: vec![TapBook::new(); slots.max(1)],
            img: Vec::new(),
            mutant: None,
            keep_pending: false,
            profile: None,
            weight_bytes,
            rows_cap: r,
            split_lm_head: super::tp_arm::head_split().lm_head,
            dump_ids: Vec::new(),
            dump_slots: Vec::new(),
            dump_pos: Vec::new(),
        })
    }

    /// The bytes this state holds, by name: the head's weights (the pack's tensors, counted once here), its KV layer per token, the tap
    /// store and retained window taps, the head's scratch, the draft outputs, and the restricted head when built. `(name, bytes)`.
    pub fn allocations(&self, kv_bytes_per_token: u64) -> Vec<(&'static str, u64)> {
        let b = |d: &DevBuf| d.bytes() as u64;
        let sum = |v: &[DevBuf]| v.iter().map(|d| d.bytes() as u64).sum::<u64>();
        vec![
            ("mtp_weights", self.weight_bytes),
            ("mtp_kv_per_token", kv_bytes_per_token),
            ("mtp_taps_store", b(&self.store)),
            ("mtp_taps_window", b(&self.win_tap)),
            ("mtp_scratch", b(&self.x) + b(&self.xn) + b(&self.out) + b(&self.cat) + b(&self.h_in) + b(&self.ids)),
            ("mtp_drafts", sum(&self.draft_ids) + sum(&self.draft_vals) + b(&self.keyed)),
            ("mtp_restricted_head", self.sub.as_ref().map_or(0, SubHead::bytes)),
        ]
    }

    /// Slot `slot`'s bookkeeping (a test's and the backend's view).
    pub fn book(&self, slot: usize) -> &TapBook {
        &self.book[slot]
    }

    /// Where `slot`'s carry sits in the tap store (`store_buf`): row 0 of its `MTP_TAP_CAP` rows, f32 `[4096]`. After a prompt chunk it is the
    /// post-`norm_f` tap of the slot's last position, which the next chunk's first entry pairs with its first token.
    pub(super) fn carry_offset(slot: usize) -> usize {
        slot * MTP_TAP_CAP * HIDDEN * 4
    }

    /// The tap store (`[slots][MTP_TAP_CAP][4096]` f32), for the prefix cache's snapshot and restore of a carry.
    pub(super) fn store_buf(&self) -> &DevBuf {
        &self.store
    }

    pub(super) fn store_buf_mut(&mut self) -> &mut DevBuf {
        &mut self.store
    }

    /// Puts `slot`'s bookkeeping where a prompt of `len` tokens leaves it, for a slot restored from a snapshot: every entry below `len` is
    /// built (the pages hold them) and the store's row 0 is the carry, so the next chunk continues (`first: len, n: 1`). Without it the
    /// slot's next chunk would find a gap and the slot would silently stop drafting. `ok` is the snapshot's own flag.
    pub(super) fn restore_book(&mut self, slot: usize, len: usize, ok: bool) {
        self.book[slot] = TapBook { ok, first: len, n: 1 };
    }

    /// Copies the window's post-`norm_f` rows (or, for the mutant, the residual before it) into the retained buffer.
    pub(super) fn retain_window_taps(&mut self, xn: &DevBuf, x: &DevBuf, rows: usize, st: &Stream) -> Result<(), ModelError> {
        let src = if self.mutant == Some(MtpMutant::TapBeforeNorm) { x } else { xn };
        if rows * HIDDEN * 4 > self.win_tap.bytes() {
            return Err(ModelError::Request(format!("{rows} tap rows of a window of {}", self.win_tap.bytes() / (HIDDEN * 4))));
        }
        self.win_tap.copy_range_from(0, src, 0, rows * HIDDEN * 4, st)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Pure plans (host only, tested without a device).

/// Where one head entry's hidden row comes from in a prompt chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HidSrc {
    /// The slot's carry: the previous chunk's last tap (store row 0).
    Carry,
    /// The chunk's own tap at this chunk-relative row.
    Tap(usize),
}

/// One head entry built from a prompt chunk: the chunk-relative row of its token, where its hidden row comes from, its cache position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkEntry {
    pub token_row: usize,
    pub hidden: HidSrc,
    pub pos: usize,
}

/// The entries a prompt chunk of `rows` rows at position `first_pos` builds (the entries `first_pos .. first_pos + rows` of the book, minus
/// entry 0, which no tap predicted). The token of row `j` (position `first_pos + j`) pairs with the tap of the position before it: the
/// chunk's own row `j - 1`, or the carry for `j = 0`. `no_shift` is the mutant that pairs it with its own position's tap.
pub fn chunk_entries(first_pos: usize, rows: usize, no_shift: bool) -> Vec<ChunkEntry> {
    (0..rows)
        .filter(|&j| first_pos + j > 0)
        .map(|j| {
            let hidden = match (j, no_shift) {
                (_, true) => HidSrc::Tap(j),
                (0, false) => HidSrc::Carry,
                (_, false) => HidSrc::Tap(j - 1),
            };
            // the entry sits at the position of its tap: one before its token
            ChunkEntry { token_row: j, hidden, pos: first_pos + j - 1 }
        })
        .collect()
}

/// The image rows among the head's absorb rows of one prompt slice. `ops` are the slice's splice as `(dst_row, src_row, n)` over BATCH rows,
/// `token_rows[k]` is the batch row of the token of absorb row `k`. Returns runs `(op index, absorb row, feature row, n)`: absorb rows
/// `absorb_row .. absorb_row + n` take rows `feature_row ..` of op `op index`'s image as their embedding half. Consecutive absorb rows over
/// consecutive feature rows of one op are one run; a row that is no op's is text (nothing returned for it).
pub fn head_image_runs(ops: &[(usize, usize, usize)], token_rows: &[usize]) -> Vec<(usize, usize, usize, usize)> {
    let mut runs: Vec<(usize, usize, usize, usize)> = Vec::new();
    for (k, &row) in token_rows.iter().enumerate() {
        let Some((oi, &(dst, src, _))) = ops.iter().enumerate().find(|(_, &(d, _, n))| (d..d + n).contains(&row)) else {
            continue;
        };
        let feat = src + (row - dst);
        match runs.last_mut() {
            Some(l) if l.0 == oi && l.1 + l.3 == k && l.2 + l.3 == feat => l.3 += 1,
            _ => runs.push((oi, k, feat, 1)),
        }
    }
    runs
}

/// The entries a draft call absorbs for one sequence: the book's pending entries `first .. first + n`, each `(token index, store row,
/// cache position)`. `last_only` is the mutant that absorbs only the last of them.
pub fn absorb_entries(first: usize, n: usize, last_only: bool) -> Vec<(usize, usize, usize)> {
    let from = if last_only { n.saturating_sub(1) } else { 0 };
    (from..n).filter(|&i| first + i > 0).map(|i| (first + i, i, first + i - 1)).collect()
}

/// The keyed pick's record for one sequence of `len` tokens (its depth-0 draft is compared at position `len`), and the number of candidates
/// the call must fetch for it (0: the row is not keyed and takes the head's argmax). A request is keyed when it is sampled (temperature
/// above 0) and its `top_k` is one the call can fetch (`1..=MTP_MAX_CANDIDATES`); a greedy request has no noise.
pub fn keyed_record(s: Option<&Sampling>, len: usize) -> (KeyedRow, usize) {
    let off = KeyedRow { first_position: len as u64, ..Default::default() };
    let Some(s) = s else { return (off, 0) };
    if s.temperature <= 0.0 || !(1..=MTP_MAX_CANDIDATES).contains(&s.top_k) {
        return (off, 0);
    }
    let rec = KeyedRow {
        seed: s.seed,
        first_position: len as u64,
        temperature: s.temperature,
        top_p: s.top_p,
        min_p: s.min_p,
        top_k: s.top_k as i32,
        keyed: 1,
    };
    (rec, s.top_k)
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The model's side.

/// Where a step's token ids come from.
#[derive(Clone, Copy)]
enum IdsFrom {
    /// The host-uploaded ids (an absorb or a debug forward).
    Host,
    /// The picked ids of chain depth `k` (device resident: the chain never touches the host).
    Draft(usize),
}

#[derive(Clone, Copy)]
enum HidFrom {
    HIn,
    Out,
}

/// What identifies a captured draft call of one sequence: every host value that reaches a launcher and is not read from a device buffer.
/// The absorb's rows and the chain's depth fix every launch's rows and grid; `count` is the keyed pick's candidates (0: the argmax),
/// `restricted` the head's table, `chunks` the attention launch's chunks (a power-of-two bucket, as the trunk's windows: a row with fewer
/// chunks of its own exits early with the same bits). The index arrays, token ids, hidden rows, sampling records, block table and every
/// scratch buffer are fixed device buffers the call rewrites before the launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DraftKey {
    rows: usize,
    depth: usize,
    count: usize,
    restricted: bool,
    chunks: usize,
}

enum DraftSlot {
    /// Run once eagerly; the next call of the shape captures.
    Seen,
    Ready(Graph, u64),
    /// A capture of this shape failed: eager from now on.
    Eager,
}

/// Captured draft calls kept at once.
const DRAFT_GRAPH_LIMIT: usize = 48;

/// `REXL3_NEMO_DRAFT_GRAPH=1` replays a draft call of one sequence as one CUDA graph (default: off).
pub const DRAFT_GRAPH_ENV: &str = "REXL3_NEMO_DRAFT_GRAPH";

fn draft_graph_enabled() -> bool {
    matches!(std::env::var(DRAFT_GRAPH_ENV).as_deref(), Ok("1" | "on" | "true" | "yes"))
}

/// Which rows of a head window go on past the keys and values. An entry's K and V depend only on its own input (`eh_proj`, the attention
/// norm, `k` and `v`): the query side, the attention, `o` and the MoE block of a row matter only when something reads that row's output,
/// and of an absorb's rows only each sequence's last one is read (its hidden row starts the chain). Every kernel of those stages is
/// row-invariant (a row's bits do not depend on the other rows of the call), so running them over the rows that are read changes no
/// bit of any of them or of the cache, and the bf16 MoE of the dropped rows (the union of their experts: most of the head's bytes at a
/// few rows) is not read.
#[derive(Clone, Copy)]
enum Tail<'a> {
    /// Every row (the chain steps, one row per sequence, and the debug forwards that capture every stage).
    All,
    /// None: the window only builds cache entries (the prompt hook, an absorb that drafts nothing).
    KvOnly,
    /// `rows` (ascending indices of the window's rows) with their own index `idx`; their residual is compacted to window rows `0..rows.len()`.
    Rows { rows: &'a [usize], idx: &'a RowIndex },
}

/// Waits until every host-to-device upload made so far has landed. `DevBuf::upload` is a pageable `cudaMemcpy` on the default stream: it may
/// return before its DMA is done, and the model's stream is non-blocking, so it does not wait for it either. A kernel launched right after an
/// upload could read the old contents (measured: a chain whose next token was uploaded just before its first kernel read the previous token
/// in 2 of 120 chains). Called after the last upload of a head call and before its first launch.
fn uploads_visible() -> Result<(), ModelError> {
    gb_cuda::rt::device_sync()?;
    Ok(())
}

/// `dst[..v.len()] = v` in the order of `st`: a kernel launched on `st` afterwards reads the new bytes and one launched before has read the
/// old ones, with no device-wide synchronisation (`model::upload_ordered`'s rule: the pageable source is staged before the call returns).
fn upload_in_order<T: Copy>(dst: &mut DevBuf, v: &[T], st: &Stream) -> Result<(), ModelError> {
    let n = std::mem::size_of_val(v);
    assert!(n <= dst.bytes(), "upload of {n} bytes into a {}-byte buffer", dst.bytes());
    // SAFETY: `dst` is a device allocation of at least `n` bytes and `v` is `n` readable bytes
    unsafe { st.memcpy_h2d_async(dst.ptr(), v.as_ptr() as *const u8, n) }?;
    Ok(())
}

/// Device bytes of the head's weights as held: the pack's bf16 tensors in place (a bf16 element is 2 bytes; the router's bias is f32), and
/// for a 4-bit head each EXL3 projection's trellis in place with its scales (fp16 for `eh_proj` and the attention's q, k, v, o, widened to
/// f32 in the MoE block's tables, as the trunk's are). The routed experts count only the ones this rank holds (all 512 on one node, its
/// half under expert parallelism).
pub fn weight_bytes(h: &MtpW) -> u64 {
    let norms = [&h.enorm, &h.hnorm, &h.attn_norm, &h.moe_norm, &h.final_norm];
    norms.iter().map(|b| b.bytes() as u64).sum::<u64>() + h.dense().iter().map(|l| l.bytes()).sum::<u64>() + h.moe.weight_bytes()
}

/// `y[..rows] = l(x[..rows])` for a head projection `[n, k]`: the bf16 GEMV, or an EXL3 linear through the trunk's dense path
/// (`linear_rows`: row invariant, a row's bits do not depend on the call's other rows or its width). `x` and `y` are f32.
#[allow(clippy::too_many_arguments)]
fn head_linear(
    l: &HeadLinear,
    x: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    n: usize,
    k: usize,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), ModelError> {
    match l {
        HeadLinear::Bf16(wb) => gemv_bf16(wb, x, y, rows, n, k, st)?,
        HeadLinear::Exl3(e) => linear_rows(e, x, y, rows, Dtype::F32, Dtype::F32, ws, st)?,
    }
    Ok(())
}

/// Test hook, never set in production: `REXL3_NEMO_HEAD_ON_PREFIX=1` lets a layer-limited model (a prefix of the trunk) carry the head.
/// The head then drafts from the prefix's taps, which says nothing about acceptance and everything about bits: a single-node run and
/// the two shards of a tensor-parallel loopback on the same prefix must agree byte for byte, in 10 GiB and not in 95.
pub const HEAD_ON_PREFIX_ENV: &str = "REXL3_NEMO_HEAD_ON_PREFIX";

/// Whether [`HEAD_ON_PREFIX_ENV`] is set to `1`.
pub fn head_on_prefix() -> bool {
    matches!(std::env::var(HEAD_ON_PREFIX_ENV).as_deref(), Ok("1"))
}

/// Whether the model is built to load the head: the whole trunk (or a prefix, for the test hook [`HEAD_ON_PREFIX_ENV`]), and
/// `REXL3_MTP` is not `off`.
pub fn head_wanted(layers: Option<usize>) -> bool {
    (layers.is_none() || head_on_prefix()) && !matches!(std::env::var("REXL3_MTP").as_deref(), Ok("off" | "0" | "no" | "false"))
}

/// The ids of a `mtp_vocab.py` file: whitespace separated decimal ids, `#` starts a comment line.
pub fn parse_vocab_file(text: &str) -> Result<Vec<u32>, String> {
    let mut ids = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("");
        for w in line.split_whitespace() {
            ids.push(w.parse::<u32>().map_err(|_| format!("line {}: `{w}` is not a token id", n + 1))?);
        }
    }
    Ok(ids)
}

impl NemotronModel {
    // -------------------------------------------------------------------------------------------------------- slots, taps, commits

    /// Whether this model drafts with the pack's head.
    pub fn mtp_loaded(&self) -> bool {
        self.core.mtp.is_some()
    }

    /// The slot's bookkeeping, when the model drafts.
    pub fn mtp_book(&self, slot: usize) -> Option<&TapBook> {
        self.core.mtp.as_ref().and_then(|m| m.book.get(slot))
    }

    /// Switches the slot's drafting off for the rest of its sequence. Never an error: the slot just drafts nothing.
    pub fn mtp_disable_slot(&mut self, slot: usize) {
        if let Some(b) = self.core.mtp.as_mut().and_then(|m| m.book.get_mut(slot)) {
            b.disable();
        }
    }

    /// Debug (timing at a depth nobody has prefilled, [`NemotronModel::debug_seed_depth`]): marks `slot`'s head state as that of a prompt
    /// of `len` tokens that went through the prefill hook in one chunk, so a draft call has a carry to absorb. The head's cache holds
    /// whatever the pool holds (zeros after load). Returns whether the slot drafts.
    #[doc(hidden)]
    pub fn mtp_debug_seed_slot(&mut self, slot: usize, len: usize) -> bool {
        match self.core.mtp.as_mut().and_then(|m| m.book.get_mut(slot)) {
            Some(b) => b.prefill_chunk(0, len).is_ok(),
            None => false,
        }
    }

    /// Committed rows of `slot` the head has not absorbed yet, or `None` when the slot does not draft.
    pub fn mtp_pending_rows(&self, slot: usize) -> Option<usize> {
        let b = self.mtp_book(slot)?;
        b.ok.then_some(b.n)
    }

    /// Debug: set a deliberate drafter bug (tests only).
    #[doc(hidden)]
    pub fn set_mtp_mutant(&mut self, mutant: Option<MtpMutant>) {
        if let Some(m) = self.core.mtp.as_mut() {
            m.mutant = mutant;
        }
    }

    /// Debug: with `true` a draft call leaves the slots' pending rows pending, so the same call can be repeated (a determinism check).
    #[doc(hidden)]
    pub fn set_mtp_keep_pending(&mut self, on: bool) {
        if let Some(m) = self.core.mtp.as_mut() {
            m.keep_pending = on;
        }
    }

    /// Debug: start (or, with `false`, stop and discard) timing the head's stages (it adds a host sync per stage).
    pub fn set_mtp_profile(&mut self, on: bool) {
        if let Some(m) = self.core.mtp.as_mut() {
            m.profile = on.then(Default::default);
        }
    }

    /// Debug: the milliseconds by stage since profiling was switched on (and clear them).
    pub fn take_mtp_profile(&mut self) -> Vec<(&'static str, f64)> {
        match self.core.mtp.as_mut().and_then(|m| m.profile.as_mut()) {
            Some(p) => std::mem::take(p).into_iter().collect(),
            None => Vec::new(),
        }
    }

    /// The head's device bytes by name (see [`MtpState::allocations`]); empty without a head.
    pub fn mtp_allocations(&self) -> Vec<(&'static str, u64)> {
        let per_token = self.core.kv_geom.dtype.token_bytes() as u64;
        self.core.mtp.as_ref().map(|m| m.allocations(per_token)).unwrap_or_default()
    }

    /// A new sequence in `slot` (called by `begin_slot`) or a finished one (`end_slot`).
    pub(super) fn mtp_reset_slot(&mut self, slot: usize) {
        if let Some(b) = self.core.mtp.as_mut().and_then(|m| m.book.get_mut(slot)) {
            b.reset();
        }
    }

    /// A window's rows of `slot` were committed (`span.keep` of them): their taps join the slot's store. `row0` is the position of the
    /// window's first row (the slot's committed length before the commit). Never an error: a slot whose bookkeeping cannot take the rows
    /// is switched off and drafts nothing.
    pub(super) fn mtp_commit_taps(&mut self, span: &CommitSpan, row0: Option<usize>) {
        let Core { mtp, stream, .. } = &mut self.core;
        let Some(m) = mtp.as_mut() else { return };
        let Some(book) = m.book.get_mut(span.slot) else { return };
        let Some(row0) = row0 else {
            book.disable();
            return;
        };
        let kept = if m.mutant == Some(MtpMutant::AdvanceAllRows) { span.end - span.start } else { span.keep };
        if super::tapdump::enabled() {
            // calibration only: the committed rows' taps and tokens, whatever the book does with them (a sync and a download)
            let row = HIDDEN * 4;
            let mut taps = vec![0f32; kept * HIDDEN];
            let got = stream
                .sync()
                .map_err(ModelError::from)
                .and_then(|_| m.win_tap.download_range(span.start * row, &mut taps).map_err(ModelError::from));
            // the stashed ids must be this window's rows of this slot, from position row0 on
            let rows = span.start..span.start + kept;
            let ours = m.dump_slots.get(rows.clone()).is_some_and(|s| s.iter().all(|&x| x == span.slot as i32))
                && m.dump_pos.get(rows.clone()).is_some_and(|p| p.iter().enumerate().all(|(i, &x)| x as usize == row0 + i));
            match (got, m.dump_ids.get(rows)) {
                (Ok(()), Some(ids)) if ours => {
                    super::tapdump::write(super::tapdump::DumpKind::Window, span.slot, row0, ids, &taps, HIDDEN)
                }
                (r, _) => eprintln!(
                    "rexl3 level=warn event=nemo_tap_dump_skipped slot={} ok={} rows_match={ours}",
                    span.slot,
                    r.is_ok()
                ),
            }
        }
        let book = &mut m.book[span.slot];
        match book.window_commit(row0, kept, MTP_TAP_CAP) {
            Ok(dst) => {
                let row = HIDDEN * 4;
                let r = m.store.copy_range_from(
                    (span.slot * MTP_TAP_CAP + dst) * row,
                    &m.win_tap,
                    span.start * row,
                    kept * row,
                    stream,
                );
                if r.is_err() {
                    m.book[span.slot].disable();
                }
            }
            // a sequence that never asks for drafts just fills its store: not worth a line
            Err(BookError::Off | BookError::Overflow) => {}
            Err(e) => {
                eprintln!("rexl3 level=warn event=mtp_slot_off slot={} why={e:?} (no MTP drafts for this sequence)", span.slot)
            }
        }
    }

    /// After a prefill slice ran through the trunk (its post-`norm_f` rows are in `bufs.xn`, the stream is synchronised): build the head's
    /// entries of the slice's positions from each sequence's carry and the slice's taps, and keep each sequence's last tap as its new
    /// carry. No `lm_head`: only the cache entries are wanted. Ends synchronised.
    pub(super) fn mtp_prefill_hook(&mut self, b: &RowBatch, seqs: &[SeqPlan]) -> Result<(), ModelError> {
        let NemotronModel { core, .. } = self;
        let Core { mtp, bufs, stream, splice, .. } = core;
        let Some(m) = mtp.as_mut() else { return Ok(()) };
        if super::tapdump::enabled() {
            // calibration only: each sequence's post-norm_f rows of this slice and their tokens (the forward ended synchronised)
            let row = HIDDEN * 4;
            for s in seqs {
                let mut taps = vec![0f32; s.rows * HIDDEN];
                bufs.xn.download_range(s.start * row, &mut taps)?;
                let ids = &b.ids[s.start..s.start + s.rows];
                super::tapdump::write(super::tapdump::DumpKind::Prefill, s.slot, s.first_pos, ids, &taps, HIDDEN);
            }
        }
        let no_shift = m.mutant == Some(MtpMutant::NoShift);
        // the image rows of this slice (the trunk's splice, still set: this runs inside the slice's call): the embedding half of an entry
        // whose token is one of them is the image's feature row, as the target's own embedding is at that position. The mutant leaves the
        // placeholder's.
        let use_img = m.mutant != Some(MtpMutant::PlaceholderImages);
        let mut token_rows: Vec<usize> = Vec::new();
        let row = HIDDEN * 4;
        // the rows the entries take their taps from: the post-norm_f rows, or the residual before it for the mutant
        let taps: &DevBuf = if m.mutant == Some(MtpMutant::TapBeforeNorm) { &bufs.x } else { &bufs.xn };
        let (mut ids, mut slots, mut pos) = (Vec::new(), Vec::new(), Vec::new());
        // plan every sequence first: a sequence the book refuses is skipped, never an error
        let mut live: Vec<(&SeqPlan, bool)> = Vec::new();
        for s in seqs {
            match m.book[s.slot].prefill_chunk(s.first_pos, s.rows) {
                Ok(carry) => live.push((s, carry)),
                Err(BookError::Off) => {}
                Err(e) => {
                    eprintln!("rexl3 level=warn event=mtp_slot_off slot={} why={e:?} (no MTP drafts for this sequence)", s.slot)
                }
            }
        }
        let mut row_out = 0usize;
        for (s, _carry) in &live {
            for e in chunk_entries(s.first_pos, s.rows, no_shift) {
                ids.push(b.ids[s.start + e.token_row]);
                slots.push(s.slot as i32);
                pos.push(e.pos as i32);
                token_rows.push(s.start + e.token_row);
                match e.hidden {
                    HidSrc::Carry => m.h_in.copy_range_from(row_out * row, &m.store, s.slot * MTP_TAP_CAP * row, row, stream)?,
                    HidSrc::Tap(t) => m.h_in.copy_range_from(row_out * row, taps, (s.start + t) * row, row, stream)?,
                }
                row_out += 1;
            }
        }
        // the new carries: each sequence's last tap (read AFTER the old carries were copied above: the stream orders them)
        for (s, _) in &live {
            m.store.copy_range_from(s.slot * MTP_TAP_CAP * row, taps, (s.start + s.rows - 1) * row, row, stream)?;
        }
        if row_out == 0 {
            stream.sync()?;
            return Ok(());
        }
        m.ids.upload(&ids)?;
        // runs of consecutive entries over consecutive feature rows are one launch
        if use_img {
            let ops: Vec<(usize, usize, usize)> = splice.iter().map(|o| (o.dst_row, o.src_row, o.n)).collect();
            m.img = head_image_runs(&ops, &token_rows)
                .into_iter()
                .map(|(oi, dst_row, src_row, n)| SpliceOp { feats: splice[oi].feats.clone(), dst_row, src_row, n })
                .collect();
        }
        let idx = RowIndex::from_host(&slots, &pos)?;
        uploads_visible()?;
        // only the cache entries are wanted: keys and values, no attention output and no MoE (`Tail`)
        let r = self.head_block(row_out, IdsFrom::Host, HidFrom::HIn, &idx, None, Tail::KvOnly);
        // whatever happened, the next absorb (a draft call's) starts with no image rows
        if let Some(m) = self.core.mtp.as_mut() {
            m.img.clear();
        }
        r?;
        self.core.stream.sync()?;
        Ok(())
    }

    // ----------------------------------------------------------------------------------------------------------------- the block

    /// One forward of the head's two blocks over `rows` rows (`idx` carries each row's slot and cache position): leaves the residual after
    /// the MoE block in `x`; the caller applies the final norm and the head. `cap` copies the last `tail` rows of each stage to the host (a
    /// debug path: it synchronises). `go_on` says which rows continue past the cache write ([`Tail`]): with `Tail::Rows` the residual of
    /// those rows is in `x` rows `0..n`, with `Tail::KvOnly` nothing is left in `x`.
    fn head_block(
        &mut self,
        rows: usize,
        ids_from: IdsFrom,
        hid_from: HidFrom,
        idx: &RowIndex,
        mut cap: Option<(&mut MtpCapture, usize)>,
        go_on: Tail<'_>,
    ) -> Result<(), ModelError> {
        if rows == 0 || rows > idx.rows {
            return Err(ModelError::Request(format!("a head window of {rows} rows (the index holds {})", idx.rows)));
        }
        let NemotronModel { core, eps, .. } = self;
        let eps = *eps;
        let Core { w, bufs, moe, moe_prefill, kv, kv_geom, table, attn, stream, mtp, tp, .. } = core;
        let st = &*stream;
        let m = mtp.as_mut().ok_or_else(|| ModelError::Config("the model has no MTP head".into()))?;
        let hw: &MtpW = w.mtp.as_ref().ok_or_else(|| ModelError::Config("the weights have no MTP head".into()))?;
        let timer = if m.profile.is_some() { Some(EventTimer::new(8)?) } else { None };
        let mark = |i: usize| -> Result<(), ModelError> {
            if let Some(t) = &timer {
                t.record(i, st)?;
            }
            Ok(())
        };
        let tail_of = |c: &Option<(&mut MtpCapture, usize)>| c.as_ref().map(|(_, t)| (*t).min(rows));
        let MtpState { ids, h_in, out, x, xn, cat, draft_ids, profile, img, .. } = m;
        mark(0)?;

        // the input: [enorm(embed(token)) | hnorm(hidden)], the embedding FIRST, into one [rows, 8192] buffer
        {
            let idb: &DevBuf = match ids_from {
                IdsFrom::Host => ids,
                IdsFrom::Draft(k) => &draft_ids[k],
            };
            let hid: &DevBuf = match hid_from {
                HidFrom::HIn => h_in,
                HidFrom::Out => out,
            };
            embed_gather_f32(&w.embed, idb, rows, xn, st)?;
            // an image's rows: its feature row (bf16, widened) is the embedding half, normed by `enorm` as a token's embedding is
            for op in img.iter() {
                splice_rows(&op.feats.buf, op.src_row, xn, op.dst_row, op.n, st)?;
            }
            rmsnorm_bf16w_f32(xn, &hw.enorm, rows, HIDDEN, HIDDEN, eps, cat, 2 * HIDDEN, st)?;
            let mut second = cat.subview(HIDDEN * 4, ((rows - 1) * 2 * HIDDEN + HIDDEN) * 4);
            rmsnorm_bf16w_f32(hid, &hw.hnorm, rows, HIDDEN, HIDDEN, eps, &mut second, 2 * HIDDEN, st)?;
        }
        mark(1)?;
        // a 4-bit head: the trunk's dense EXL3 path (row invariant: a row's bits do not depend on the call's other rows or its width)
        head_linear(&hw.eh_proj, cat, x, rows, HIDDEN, 2 * HIDDEN, &mut bufs.ws, st)?;
        mark(2)?;
        let fetch = |buf: &DevBuf, t: usize| -> Result<Vec<f32>, ModelError> {
            st.sync()?;
            let mut v = vec![0f32; t * HIDDEN];
            buf.download_range((rows - t) * HIDDEN * 4, &mut v)?;
            Ok(v)
        };
        if let (Some(t), Some((c, _))) = (tail_of(&cap), cap.as_mut()) {
            c.fused = fetch(&*x, t)?;
        }

        // attention on the head's own KV layer (the stripe after the trunk's). Keys and values of every row first: they are what the cache
        // keeps, and an entry's K and V need nothing past the attention norm of its own input.
        let g = *kv_geom;
        let ord = g.layers - 1;
        let mut layer = NemoKv {
            k: kv.k.subview(g.layer_offset(ord), g.layer_bytes()),
            v: kv.v.subview(g.layer_offset(ord), g.layer_bytes()),
            n_blocks: g.pages,
            dtype: g.dtype,
        };
        {
            rmsnorm_bf16w_f32(x, &hw.attn_norm, rows, HIDDEN, HIDDEN, eps, xn, HIDDEN, st)?;
            head_linear(&hw.k, xn, &mut bufs.k, rows, KV_TOKEN_ELEMS, HIDDEN, &mut bufs.ws, st)?;
            head_linear(&hw.v, xn, &mut bufs.v, rows, KV_TOKEN_ELEMS, HIDDEN, &mut bufs.ws, st)?;
            kv_write(&mut layer, table, idx, &bufs.k, &bufs.v, st)?;
        }
        // the rows that go on, and their index: all of them, or the ones something reads (compacted to the front of `x`, in place: the
        // i-th kept row index is at least i, so a copy never overwrites a row a later copy still has to read)
        let (n, aidx): (usize, &RowIndex) = match go_on {
            Tail::All => (rows, idx),
            Tail::KvOnly => (0, idx),
            Tail::Rows { rows: keep, idx: ti } => {
                let row = HIDDEN * 4;
                debug_assert!(
                    keep.windows(2).all(|p| p[0] < p[1]) && keep.last().is_none_or(|&r| r < rows) && ti.rows == keep.len()
                );
                for (j, &r) in keep.iter().enumerate() {
                    if r != j {
                        let src = x.subview(r * row, row);
                        x.copy_range_from(j * row, &src, 0, row, st)?;
                    }
                }
                if !keep.is_empty() && keep.len() != rows {
                    // the attention norm of the kept rows again (a row-wise kernel: the same bits as the first pass gave that row)
                    rmsnorm_bf16w_f32(x, &hw.attn_norm, keep.len(), HIDDEN, HIDDEN, eps, xn, HIDDEN, st)?;
                }
                (keep.len(), ti)
            }
        };
        if n > 0 {
            head_linear(&hw.q, xn, &mut bufs.q, n, HIDDEN, HIDDEN, &mut bufs.ws, st)?;
            // the trunk's rule: a row at position >= FLASH_FROM_POS takes the flash kernel, whichever call carries it (the prompt hook,
            // an absorb, a chain step), so the head's cache is a function of the prompt's tokens alone at every position
            attn_long(&bufs.q, &layer, table, aidx, 0, attn, &mut bufs.attn_o, st)?;
            head_linear(&hw.o, &bufs.attn_o, &mut bufs.mix, n, HIDDEN, HIDDEN, &mut bufs.ws, st)?;
            add_f32(x, &bufs.mix, n * HIDDEN, st)?;
        }
        mark(3)?;
        if let (Some(t), Some((c, _))) = (tail_of(&cap), cap.as_mut()) {
            c.after_attn = fetch(&*x, t)?;
        }

        // the latent-MoE block on the model's shared scratch
        if n > 0 {
            rmsnorm_bf16w_f32(x, &hw.moe_norm, n, HIDDEN, HIDDEN, eps, xn, HIDDEN, st)?;
            // The head's block is bf16. Up to a window's rows it runs the decode path; a wider absorb (a prefill slice's rows) takes the
            // large-row prefill path on the model's scratch (each expert's weights read once for the slice) and, without one, goes
            // through the decode path in calls of a window's width. The block is row invariant, so no bit depends on the path or the cut.
            //
            // Tensor parallelism: with the head's experts split (`hw.experts_split`) this rank holds half of them and the block's `y`
            // is allreduced between the routed down projection and the combine (`tp_arm::moe_decode`, `moe_prefill`: one collective
            // per call, on the stream, nothing synchronised). A head whole on every rank is the plain call, whatever the group.
            let cap = moe.max_rows();
            let comm = &**tp;
            if let (true, true, Some(p)) = (n > cap, hw.moe.supports_prefill(), moe_prefill.as_mut()) {
                if hw.experts_split {
                    super::tp_arm::moe_prefill(comm, &hw.moe, p, xn, n, &mut bufs.mix, st)?;
                } else {
                    hw.moe.forward_prefill_in(p, xn, n, &mut bufs.mix, st)?;
                }
            } else {
                let mut at = 0;
                while at < n {
                    let c = (n - at).min(cap);
                    let h = xn.subview(at * HIDDEN * 4, c * HIDDEN * 4);
                    let mut o = bufs.mix.subview(at * HIDDEN * 4, c * HIDDEN * 4);
                    if hw.experts_split {
                        super::tp_arm::moe_decode(comm, &hw.moe, moe, &h, c, &mut o, st)?;
                    } else {
                        hw.moe.forward_in(moe, &h, c, &mut o, st)?;
                    }
                    at += c;
                }
            }
            add_f32(x, &bufs.mix, n * HIDDEN, st)?;
        }
        mark(4)?;
        if let (Some(t), Some((c, _))) = (tail_of(&cap), cap.as_mut()) {
            c.block = fetch(&*x, t)?;
        }
        if let (Some(t), Some(p)) = (&timer, profile.as_mut()) {
            for (name, a, b) in [
                ("embed + enorm + hnorm", 0, 1),
                ("eh_proj", 1, 2),
                ("attention block (norm, q k v, kv_write, attn, o)", 2, 3),
                ("moe block (norm, router, experts, shared)", 3, 4),
            ] {
                *p.entry(name).or_insert(0.0) += t.ms(a, b)? as f64;
            }
        }
        Ok(())
    }

    /// `out[dst_row..]` = `final_layernorm` of residual rows `src_rows` (one launch per run of consecutive rows).
    fn head_final_norm(&mut self, src_rows: &[usize], dst_row0: usize) -> Result<(), ModelError> {
        let NemotronModel { core, eps, .. } = self;
        let eps = *eps;
        let Core { w, stream, mtp, .. } = core;
        let st = &*stream;
        let m = mtp.as_mut().ok_or_else(|| ModelError::Config("the model has no MTP head".into()))?;
        let hw: &MtpW = w.mtp.as_ref().ok_or_else(|| ModelError::Config("the weights have no MTP head".into()))?;
        let timer = if m.profile.is_some() { Some(EventTimer::new(2)?) } else { None };
        if let Some(t) = &timer {
            t.record(0, st)?;
        }
        let MtpState { x, out, profile, .. } = m;
        let row = HIDDEN * 4;
        let mut at = 0;
        while at < src_rows.len() {
            let mut end = at + 1;
            while end < src_rows.len() && src_rows[end] == src_rows[end - 1] + 1 {
                end += 1;
            }
            let n = end - at;
            let src = x.subview(src_rows[at] * row, n * row);
            let mut dst = out.subview((dst_row0 + at) * row, n * row);
            rmsnorm_bf16w_f32(&src, &hw.final_norm, n, HIDDEN, HIDDEN, eps, &mut dst, HIDDEN, st)?;
            at = end;
        }
        if let (Some(t), Some(p)) = (&timer, profile.as_mut()) {
            t.record(1, st)?;
            *p.entry("final_layernorm").or_insert(0.0) += t.ms(0, 1)? as f64;
        }
        Ok(())
    }

    /// The head over `n` final-normed rows (`out` rows `row0..row0 + n`) into chain depth `k`'s `draft_ids` / `draft_vals`: the argmax (`keyed`
    /// None), or the TARGET's keyed draw over the head's top `count` candidates (`keyed` Some(count), noise of position `first_position + k`).
    /// A restricted head's columns are mapped to token ids BEFORE the pick: the noise is keyed by id, never by column.
    fn head_pick(&mut self, n: usize, row0: usize, k: usize, restricted: bool, keyed: Option<usize>) -> Result<(), ModelError> {
        let NemotronModel { core, vocab, .. } = self;
        let vocab = *vocab;
        let Core { w, bufs, stream, mtp, tp, .. } = core;
        let st = &*stream;
        let m = mtp.as_mut().ok_or_else(|| ModelError::Config("the model has no MTP head".into()))?;
        let mutant = m.mutant;
        let timer = if m.profile.is_some() { Some(EventTimer::new(3)?) } else { None };
        if let Some(t) = &timer {
            t.record(0, st)?;
        }
        let MtpState { sub, out, draft_ids, draft_vals, keyed: records, profile, split_lm_head, .. } = m;
        let sub = sub.as_ref().filter(|_| restricted);
        let x = out.subview(row0 * HIDDEN * 4, n * HIDDEN * 4);
        let (cols, table): (usize, super::tp_arm::HeadTable<'_>) = match sub {
            Some(s) => (s.cols, (&s.w).into()),
            None => (vocab, (&w.lm_head).into()),
        };
        // a single rank: the plain GEMV. A rank of a group (`REXL3_TP_HEAD`): its half of the rows (of the restricted sub-table, which
        // every rank holds whole, or of the trunk's head, which it holds as its half), the `-0.0` scatter and one allreduce of
        // `[n, cols]`, so the top-k, the id map and the keyed pick below run on the identical full row on both ranks
        // (`tp_arm::lm_head_logits`). `split_lm_head` off: the replicated call of the first bring-up (no collective).
        let ws = &mut bufs.ws;
        super::tp_arm::lm_head_logits(
            &**tp,
            table,
            &x,
            &mut bufs.logits,
            bufs.tp_logits.as_mut(),
            n,
            cols,
            *split_lm_head,
            ws,
            st,
        )?;
        if let Some(t) = &timer {
            t.record(1, st)?;
        }
        match keyed {
            None => {
                topk_f32(&bufs.logits, n, cols, 1, &mut bufs.topk_ws, &mut draft_ids[k], &mut draft_vals[k], st)?;
                if let Some(s) = sub {
                    if mutant != Some(MtpMutant::RestrictedNoMap) {
                        map_ids(&mut draft_ids[k], &s.map, n, cols, st)?;
                    }
                }
            }
            Some(count) => {
                topk_f32(&bufs.logits, n, cols, count, &mut bufs.topk_ws, &mut bufs.topk_ids, &mut bufs.topk_vals, st)?;
                if let Some(s) = sub {
                    if mutant != Some(MtpMutant::RestrictedNoMap) {
                        map_ids(&mut bufs.topk_ids, &s.map, n * count, cols, st)?;
                    }
                }
                keyed_pick(&bufs.topk_ids, &bufs.topk_vals, count, n, records, k, &mut draft_ids[k], &mut draft_vals[k], st)?;
            }
        }
        if let (Some(t), Some(p)) = (&timer, profile.as_mut()) {
            t.record(2, st)?;
            let key = if sub.is_some() { "lm_head restricted (gemv)" } else { "lm_head full (gemv)" };
            *p.entry(key).or_insert(0.0) += t.ms(0, 1)? as f64;
            *p.entry("top-k + pick").or_insert(0.0) += t.ms(1, 2)? as f64;
        }
        Ok(())
    }

    // ----------------------------------------------------------------------------------------------------------- the restricted head

    /// Restricts the head's draft vocabulary to `ids` (token ids; sorted and de-duplicated here): their rows of the trunk's `lm_head` are
    /// gathered, bit for bit, into a contiguous `[n, 4096]` bf16 table on the device, and the draft top-k runs over that table's logits and
    /// maps back to ids. The kept rows are bit-identical to the full head's, so a restricted draft is the full head's whenever the full
    /// head's choice lies inside the set; a draft only proposes, so nothing about the output can change. Returns the device bytes added.
    /// Turns the restricted head on ([`NemotronModel::set_mtp_restricted`] switches it).
    pub fn mtp_restrict_vocab(&mut self, ids: &[u32]) -> Result<usize, ModelError> {
        let vocab = self.vocab;
        let mut ids: Vec<u32> = ids.to_vec();
        ids.sort_unstable();
        ids.dedup();
        if ids.is_empty() {
            return Err(ModelError::Config("the restricted draft vocabulary is empty".into()));
        }
        if let Some(&bad) = ids.iter().find(|&&t| t as usize >= vocab) {
            return Err(ModelError::Config(format!(
                "the restricted draft vocabulary names token {bad}, past the vocabulary of {vocab}"
            )));
        }
        let Core { w, stream, mtp, tp, .. } = &mut self.core;
        let m = mtp.as_mut().ok_or_else(|| ModelError::Config("the model has no MTP head".into()))?;
        // the rows of the trunk's head, whole on every rank (a rank's half is gathered through the group: `tp_arm::gather_head_rows`)
        let table = super::tp_arm::gather_head_rows(&**tp, &w.lm_head, vocab, &ids, stream)?;
        // a table the plan cuts across the ranks (`tp_arm::head_path`) is read as a sub-range per rank, which needs a view of a block:
        // a boot error here, never a quiet whole GEMV on one rank while its peers wait in the collective
        if m.split_lm_head && super::tp_shard::head_splits(ids.len(), tp.topology().ranks) && table.block().is_none() {
            return Err(ModelError::Config(format!(
                "tensor parallelism: the restricted draft table of {} rows is cut across {} ranks but is not a view of a block",
                ids.len(),
                tp.topology().ranks
            )));
        }
        let map_host: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
        let map = DevBuf::from_slice(&map_host)?;
        stream.sync()?;
        let mut member = vec![false; vocab];
        for &t in &ids {
            member[t as usize] = true;
        }
        let bytes = table.bytes() + map.bytes();
        m.sub = Some(SubHead { w: table, map, cols: ids.len(), member });
        m.use_sub = true;
        // a captured draft call baked the previous table's address
        m.draft_graphs.clear();
        Ok(bytes)
    }

    /// Draft calls that captured a graph (`[0]`), replayed a captured one (`[1]`), and ran eagerly (`[2]`) (`REXL3_NEMO_DRAFT_GRAPH`).
    pub fn mtp_draft_graph_stats(&self) -> [u64; 3] {
        self.core.mtp.as_ref().map_or([0; 3], |m| m.draft_graph_stats)
    }

    /// Uses the restricted head in draft calls (`true`, when one was built) or the full `lm_head` (`false`).
    pub fn set_mtp_restricted(&mut self, on: bool) {
        if let Some(m) = self.core.mtp.as_mut() {
            m.use_sub = on && m.sub.is_some();
        }
    }

    /// The restricted head, when built.
    pub fn mtp_sub_head(&self) -> Option<&SubHead> {
        self.core.mtp.as_ref().and_then(|m| m.sub.as_ref())
    }

    /// Whether draft calls take the restricted head now.
    pub fn mtp_restricted(&self) -> bool {
        self.core.mtp.as_ref().is_some_and(|m| m.use_sub && m.sub.is_some())
    }

    // ----------------------------------------------------------------------------------------------------------------- drafting

    /// Makes sure every `(slot, last position)` has pages in the head's layer (the shared block table) and the device table is current:
    /// a draft writes head entries ahead of the window that will verify them. An error here is a request error (no room), never fatal.
    fn mtp_ensure_positions(&mut self, last: &[(usize, usize)]) -> Result<(), ModelError> {
        let mut grew = false;
        for &(slot, p) in last {
            if p >= self.core.opts.max_context {
                return Err(ModelError::Request(format!(
                    "position {p} is past the model's context of {}",
                    self.core.opts.max_context
                )));
            }
            if super::model::KvGeom::pages_for(p + 1) > self.kv_tab.blocks_of(slot).len() {
                self.kv_tab.ensure(slot, p + 1).map_err(|e| ModelError::Request(format!("{e:?}")))?;
                grew = true;
            }
        }
        if grew || self.table_dirty {
            let lists = super::model::block_lists(&self.kv_tab, self.core.opts.slots);
            // in the stream's order, as the trunk's forward does (`BlockTable::upload` is a pageable copy on the default stream)
            let t = &mut self.core.table;
            let mut flat = vec![0i32; t.slots.max(1) * t.max_blocks];
            for (s, blocks) in lists.iter().enumerate().take(t.slots.max(1)) {
                assert!(blocks.len() <= t.max_blocks, "slot {s}: {} blocks > max_blocks {}", blocks.len(), t.max_blocks);
                for (i, &blk) in blocks.iter().enumerate() {
                    flat[s * t.max_blocks + i] = blk as i32;
                }
            }
            upload_in_order(&mut t.buf, &flat, &self.core.stream)?;
            self.table_dirty = false;
        }
        Ok(())
    }

    /// Draft for several sequences at once: absorb every pending tapped row (one head window over all asking sequences), then chain up to
    /// the deepest ask. `out[i]` is `None` for an ask whose slot cannot draft (switched off, nothing pending, no room, or a state the
    /// tokens do not match); the others get `depth` ids. One synchronisation, at the end. Never touches the trunk's state.
    ///
    /// `keyed[i]` is ask `i`'s request sampling: the drafts of an ask that carries a sampler (temperature above 0 and a `top_k` of 1 to 64)
    /// are the TARGET's keyed draw over the head's candidates (draft `k` of a sequence of `len` tokens draws the noise of `(seed, len + k,
    /// token id)`, what the engine's sampler draws at the position the window verifies it at); an empty slice, a `None`, a greedy request or
    /// a `top_k` out of range keeps the head's argmax for that ask. A draft is a proposal either way.
    pub fn mtp_draft(&mut self, asks: &[DraftAsk], keyed: &[Option<Sampling>]) -> Result<Vec<Option<Drafts>>, ModelError> {
        let mut result: Vec<Option<Drafts>> = vec![None; asks.len()];
        if self.core.mtp.is_none() || asks.is_empty() {
            return Ok(result);
        }
        // a latched trunk error is the caller's to see; a refused commit stays for the next `flush`. The stream is waited for in each
        // group's call, after its host planning (the commit's kernels run meanwhile) and before its uploads: an error the stream holds
        // is returned there as it was here.
        self.check_alive()?;
        let max_context = self.core.opts.max_context;
        let mut live: Vec<usize> = Vec::new();
        {
            let m = self.core.mtp.as_mut().unwrap();
            for (i, a) in asks.iter().enumerate() {
                // the chain writes the positions `len - 2 ..= len + depth - 3`: all of them inside the model's context
                if a.slot >= m.book.len()
                    || a.depth > MTP_MAX_DEPTH
                    || a.tokens.len() < 2
                    || a.tokens.len() + a.depth.max(1) - 2 > max_context
                {
                    continue;
                }
                match m.book[a.slot].absorb_check(a.tokens.len()) {
                    Ok(()) => live.push(i),
                    Err(BookError::Off | BookError::Nothing) => {}
                    Err(e) => eprintln!(
                        "rexl3 level=warn event=mtp_slot_off slot={} why={e:?} (no MTP drafts for this sequence)",
                        a.slot
                    ),
                }
            }
        }
        // one head window holds at most `rows_cap` rows: split the sequences into calls
        let cap = self.core.mtp.as_ref().unwrap().rows_cap;
        let mut at = 0;
        while at < live.len() {
            let mut rows = 0usize;
            let mut end = at;
            while end < live.len() {
                let n = self.core.mtp.as_ref().unwrap().book[asks[live[end]].slot].n;
                if end > at && rows + n > cap {
                    break;
                }
                rows += n;
                end += 1;
            }
            let group = live[at..end].to_vec();
            if let Err(e) = self.mtp_draft_group(asks, keyed, &group, &mut result) {
                // a head failure costs the call's drafts, not the trunk: switch the slots off; only a latched device error is the caller's
                for &i in &group {
                    self.mtp_disable_slot(asks[i].slot);
                    result[i] = None;
                }
                self.note(&e);
                if matches!(&e, ModelError::Cuda(c) if c.is_fatal()) {
                    return Err(e);
                }
                eprintln!("rexl3 level=warn event=mtp_draft_failed why={e} (no MTP drafts for these sequences)");
            }
            at = end;
        }
        Ok(result)
    }

    /// One head call over `group` (asks that fit one window of `rows_cap` rows): the index arrays live in the head's pool between calls.
    fn mtp_draft_group(
        &mut self,
        asks: &[DraftAsk],
        keyed: &[Option<Sampling>],
        group: &[usize],
        result: &mut [Option<Drafts>],
    ) -> Result<(), ModelError> {
        let mut pool = match self.core.mtp.as_mut() {
            Some(m) => std::mem::take(&mut m.idx_pool),
            None => return Ok(()),
        };
        let r = self.mtp_draft_group_run(asks, keyed, group, result, &mut pool);
        if let Some(m) = self.core.mtp.as_mut() {
            m.idx_pool = pool;
        }
        r
    }

    /// The launches of one draft call, eagerly or as one graph. A call of ONE sequence (the single-stream decode) replays the graph of its
    /// shape and captures it on the second sighting (as the trunk's windows); anything else, a model with a mutant or a profile, and graphs
    /// off (`REXL3_NEMO_GRAPHS`, or `REXL3_NEMO_DRAFT_GRAPH` not set) run eagerly. Eager and graph runs launch the same kernels.
    #[allow(clippy::too_many_arguments)]
    fn draft_run(
        &mut self,
        pool: &mut [RowIndex],
        rows: usize,
        s_n: usize,
        depth: usize,
        ends: &[usize],
        restricted: bool,
        pick: Option<usize>,
    ) -> Result<(), ModelError> {
        // a rank of a tensor-parallel group runs its draft calls eagerly: the call holds collectives of the group (the head's MoE `y` and
        // its logits, `docs/NEMOTRON_TP.md` section 9.2), and a graph of them has not been proved on the two-node group (the trunk's
        // windows are; the loopback cannot capture at all). Default off anyway: it saves 0.16 ms on one node.
        let usable = s_n == 1
            && depth > 0
            && self.core.tp.topology().is_single()
            && self.core.mtp.as_ref().is_some_and(|m| m.mutant.is_none() && m.profile.is_none() && m.img.is_empty())
            && self.graphs_on()
            && draft_graph_enabled();
        if !usable {
            if let Some(m) = self.core.mtp.as_mut() {
                m.draft_graph_stats[2] += 1;
            }
            return self.draft_launch(pool, rows, s_n, depth, ends, restricted, pick);
        }
        // the attention launch is built for the deepest row's chunks rounded up to a bucket, on every index the call reads
        let cap = self.core.opts.max_context.div_ceil(CHUNK_KEYS);
        let used = depth + 1;
        let real = pool[..used].iter().map(|i| i.real_chunks()).max().unwrap_or(1);
        let bucket = chunk_bucket(real, cap);
        for i in pool[..used].iter_mut() {
            i.set_min_chunks(bucket);
        }
        let key = DraftKey { rows, depth, count: pick.unwrap_or(0), restricted, chunks: bucket };
        let m = self.core.mtp.as_mut().unwrap();
        m.draft_tick += 1;
        let tick = m.draft_tick;
        match m.draft_graphs.remove(&key) {
            Some(DraftSlot::Ready(g, _)) => {
                let r = g.launch(&self.core.stream);
                let m = self.core.mtp.as_mut().unwrap();
                m.draft_graphs.insert(key, DraftSlot::Ready(g, tick));
                m.draft_graph_stats[1] += 1;
                r?;
                Ok(())
            }
            Some(DraftSlot::Eager) => {
                m.draft_graphs.insert(key, DraftSlot::Eager);
                m.draft_graph_stats[2] += 1;
                self.draft_launch(pool, rows, s_n, depth, ends, restricted, pick)
            }
            None => {
                // a first sighting: mark the shape and run eagerly (the cache stays bounded: the least recently used graph goes)
                if m.draft_graphs.len() >= DRAFT_GRAPH_LIMIT {
                    self.core.stream.sync()?;
                    let m = self.core.mtp.as_mut().unwrap();
                    let oldest = m
                        .draft_graphs
                        .iter()
                        .filter_map(|(k, s)| match s {
                            DraftSlot::Ready(_, used) => Some((*used, *k)),
                            _ => None,
                        })
                        .min_by_key(|x| x.0)
                        .map(|x| x.1);
                    if let Some(k) = oldest {
                        m.draft_graphs.remove(&k);
                    }
                }
                let m = self.core.mtp.as_mut().unwrap();
                m.draft_graphs.insert(key, DraftSlot::Seen);
                m.draft_graph_stats[2] += 1;
                self.draft_launch(pool, rows, s_n, depth, ends, restricted, pick)
            }
            Some(DraftSlot::Seen) => {
                // the second sighting: capture the call's launches (the model's stream is non-blocking, so the capture runs on it) and
                // replay them; a capture that fails for its own reasons marks the shape eager and runs the call eagerly
                let captured: Result<Graph, CaptureError<ModelError>> = match Graph::begin(&self.core.stream) {
                    Ok(cap) => {
                        let ran = self.draft_launch(pool, rows, s_n, depth, ends, restricted, pick);
                        cap.finish(ran)
                    }
                    Err(e) => Err(CaptureError::Refused(e)),
                };
                match captured {
                    Ok(g) => {
                        let r = g.launch(&self.core.stream);
                        let m = self.core.mtp.as_mut().unwrap();
                        m.draft_graphs.insert(key, DraftSlot::Ready(g, tick));
                        m.draft_graph_stats[0] += 1;
                        r?;
                        Ok(())
                    }
                    Err(e) => {
                        if let Some(fatal) = e.fatal() {
                            return Err(ModelError::Cuda(fatal.clone()));
                        }
                        let why = match e {
                            CaptureError::Refused(code) => format!("the runtime refused the capture: {code}"),
                            CaptureError::Closure(m) => {
                                // a launch's own argument error spoils the capture only; a device fault is the call's
                                let fatal = match &m {
                                    ModelError::Cuda(c) if gb_cuda::graph::is_capture_code(c.0) => false,
                                    other => other.is_fatal(),
                                };
                                if fatal {
                                    return Err(m);
                                }
                                format!("a launch failed while capturing: {m}")
                            }
                        };
                        eprintln!("rexl3 level=warn event=draft_graph_capture_failed rows={rows} depth={depth} {why} (this draft shape runs eagerly)");
                        let m = self.core.mtp.as_mut().unwrap();
                        m.draft_graphs.insert(key, DraftSlot::Eager);
                        m.draft_graph_stats[2] += 1;
                        self.draft_launch(pool, rows, s_n, depth, ends, restricted, pick)
                    }
                }
            }
        }
    }

    /// The kernels of a draft call, once its inputs are in the head's device buffers: the absorb's block over `rows` rows, the final norm and
    /// the pick of the first draft, then each chained draft. The head needs only each sequence's last row: every row's entry is built (keys
    /// and values), and the attention, the MoE and the final norm run over the last rows alone (`Tail::Rows` leaves their residuals in
    /// rows `0..s_n`); an absorb that drafts nothing builds the entries only. Asynchronous and with no host read, so a graph captures it.
    #[allow(clippy::too_many_arguments)]
    fn draft_launch(
        &mut self,
        pool: &[RowIndex],
        rows: usize,
        s_n: usize,
        depth: usize,
        ends: &[usize],
        restricted: bool,
        pick: Option<usize>,
    ) -> Result<(), ModelError> {
        let all: Vec<usize> = (0..s_n).collect();
        if depth > 0 {
            self.head_block(rows, IdsFrom::Host, HidFrom::HIn, &pool[0], None, Tail::Rows { rows: ends, idx: &pool[1] })?;
            self.head_final_norm(&all, 0)?;
            self.head_pick(s_n, 0, 0, restricted, pick)?;
        } else {
            self.head_block(rows, IdsFrom::Host, HidFrom::HIn, &pool[0], None, Tail::KvOnly)?;
        }
        for k in 1..depth {
            self.head_block(s_n, IdsFrom::Draft(k - 1), HidFrom::Out, &pool[1 + k], None, Tail::All)?;
            self.head_final_norm(&all, 0)?;
            self.head_pick(s_n, 0, k, restricted, pick)?;
        }
        Ok(())
    }

    fn mtp_draft_group_run(
        &mut self,
        asks: &[DraftAsk],
        keyed: &[Option<Sampling>],
        group: &[usize],
        result: &mut [Option<Drafts>],
        pool: &mut [RowIndex],
    ) -> Result<(), ModelError> {
        crate::hosttrace::mark("draft_begin", group.len() as u64);
        let s_n = group.len();
        let max_context = self.core.opts.max_context;
        // no row may sit past the model's context: near its end every stream in the call drafts a little less
        let room = group.iter().map(|&i| (max_context + 2).saturating_sub(asks[i].tokens.len())).min().unwrap_or(0);
        let depth = group.iter().map(|&i| asks[i].depth).max().unwrap_or(0).min(room);
        let last_only = self.core.mtp.as_ref().unwrap().mutant == Some(MtpMutant::AbsorbLastOnly);
        // host arrays: the absorb's rows, and where each sequence's last absorbed row is
        let (mut ids, mut slots, mut pos) = (Vec::new(), Vec::new(), Vec::new());
        let mut ends = Vec::with_capacity(s_n);
        let mut copies: Vec<(usize, usize, usize)> = Vec::new(); // (store row, h_in row, rows)
        let mut last_pos = Vec::with_capacity(s_n);
        {
            let m = self.core.mtp.as_ref().unwrap();
            for &i in group {
                let a = &asks[i];
                let b = &m.book[a.slot];
                let entries = absorb_entries(b.first, b.n, last_only);
                let Some(&(_, store_row0, _)) = entries.first() else {
                    // an ask with no entry to build cannot happen (`absorb_check` needs n > 0 and the book's `first` is at least 1)
                    return Ok(());
                };
                // one run of consecutive store rows into consecutive rows of the absorb
                copies.push((a.slot * MTP_TAP_CAP + store_row0, ids.len(), entries.len()));
                for (tok, _, p) in entries {
                    ids.push(a.tokens[tok] as i32);
                    slots.push(a.slot as i32);
                    pos.push(p as i32);
                }
                ends.push(ids.len() - 1);
                // the absorb's last row sits at position len - 2; chain row `k` (1-based) at len - 2 + k
                last_pos.push((a.slot, a.tokens.len() - 2 + depth.saturating_sub(1)));
            }
        }
        let rows = ids.len();
        if rows == 0 || rows > self.core.mtp.as_ref().unwrap().rows_cap {
            return Ok(());
        }
        // pages for every position the call writes, and the device table: BEFORE the first launch
        self.mtp_ensure_positions(&last_pos)?;
        // the keyed asks' sampling records, and how many candidates the call fetches (None: no row is keyed)
        let (records, pick) = {
            let (mut recs, mut count) = (Vec::with_capacity(s_n), 0usize);
            for &i in group {
                let (r, k) = keyed_record(keyed.get(i).and_then(|s| s.as_ref()), asks[i].tokens.len());
                recs.push(r);
                count = count.max(k);
            }
            (recs, (count > 0).then_some(count))
        };
        // the host's planning is done: wait for what the stream still holds (the commit's replay, whose kernels ran while the arrays above
        // were built) and return its error here, then queue everything below in the stream's order: no allocation, no device-wide
        // synchronisation (the index arrays are the head's own, rewritten in place by `upload_on`)
        self.core.stream.sync()?;
        crate::hosttrace::mark("draft_synced", 0);
        let restricted = self.mtp_restricted();
        let ends_slot: Vec<i32> = ends.iter().map(|&e| slots[e]).collect();
        let ends_pos: Vec<i32> = ends.iter().map(|&e| pos[e]).collect();
        {
            let Core { mtp, stream, .. } = &mut self.core;
            let m = mtp.as_mut().unwrap();
            let st = &*stream;
            if pick.is_some() {
                upload_in_order(&mut m.keyed, &records, st)?;
            }
            upload_in_order(&mut m.ids, &ids, st)?;
            // index 0: the absorb's rows; 1: its last rows (one a sequence), the only ones the chain reads (their own index for the
            // attention and the MoE); 1 + k: chain step k
            pool[0].upload_on(&slots, &pos, st)?;
            pool[1].upload_on(&ends_slot, &ends_pos, st)?;
            let s: Vec<i32> = group.iter().map(|&i| asks[i].slot as i32).collect();
            for k in 1..depth {
                let p: Vec<i32> = group.iter().map(|&i| (asks[i].tokens.len() - 2 + k) as i32).collect();
                pool[1 + k].upload_on(&s, &p, st)?;
            }
            let row = HIDDEN * 4;
            for &(src, dst, n) in &copies {
                m.h_in.copy_range_from(dst * row, &m.store, src * row, n * row, stream)?;
            }
        }
        crate::hosttrace::mark("draft_launch_begin", 0);
        self.draft_run(pool, rows, s_n, depth, &ends, restricted, pick)?;
        crate::hosttrace::mark("draft_launch_end", 0);
        let timer = self.core.mtp.as_ref().is_some_and(|m| m.profile.is_some());
        let t0 = timer.then(std::time::Instant::now);
        self.core.stream.sync()?;
        // the drafts, from the device
        let m = self.core.mtp.as_mut().unwrap();
        let mut ids_k: Vec<Vec<i32>> = Vec::with_capacity(depth);
        let mut vals_k: Vec<Vec<f32>> = Vec::with_capacity(depth);
        for k in 0..depth {
            let mut a = vec![0i32; s_n];
            let mut v = vec![0f32; s_n];
            m.draft_ids[k].download_range(0, &mut a)?;
            m.draft_vals[k].download_range(0, &mut v)?;
            ids_k.push(a);
            vals_k.push(v);
        }
        if let (Some(t0), Some(p)) = (t0, m.profile.as_mut()) {
            *p.entry("sync + download").or_insert(0.0) += t0.elapsed().as_secs_f64() * 1e3;
        }
        let wrong = m.mutant == Some(MtpMutant::WrongDrafts);
        let vocab = self.vocab as u32;
        for (j, &i) in group.iter().enumerate() {
            let want = asks[i].depth.min(depth);
            let id_of = |k: usize| {
                let id = ids_k[k][j] as u32;
                if wrong {
                    (id + 1) % vocab
                } else {
                    id
                }
            };
            result[i] = Some(Drafts { ids: (0..want).map(id_of).collect(), vals: (0..want).map(|k| vals_k[k][j]).collect() });
            let m = self.core.mtp.as_mut().unwrap();
            if !m.keep_pending {
                m.book[asks[i].slot].absorbed(asks[i].tokens.len());
            }
        }
        Ok(())
    }

    // ----------------------------------------------------------------------------------------------------------------- debug

    /// Debug (the offline study and the tests of keyed drafting): the head's top-`count` candidates (ids as token ids, logits, by value
    /// descending and id ascending) at each of `depth` chain depths for ONE sequence, with the chain fed whatever `next(depth, ids, vals)`
    /// returns (the host's pick, the true continuation, ...). The absorb and the chain write the head's cache exactly as `mtp_draft` does,
    /// but the slot's pending rows are NOT marked absorbed unless `absorb` is set, so this can be called again or followed by a real
    /// `mtp_draft`, which rewrites the very same entries. `None` when the slot cannot draft. Synchronises at every depth.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_debug_chain(
        &mut self,
        slot: usize,
        tokens: &[u32],
        depth: usize,
        count: usize,
        restricted: bool,
        absorb: bool,
        next: ChainPick,
    ) -> Result<Option<Vec<HeadList>>, ModelError> {
        let max_context = self.core.opts.max_context;
        if self.core.mtp.is_none()
            || depth == 0
            || depth > MTP_MAX_DEPTH
            || count == 0
            || count > MTP_MAX_CANDIDATES
            || tokens.len() < 2
            || tokens.len() + depth - 2 > max_context
        {
            return Ok(None);
        }
        self.check_alive()?;
        self.core.stream.sync()?;
        let (first, n) = {
            let m = self.core.mtp.as_mut().unwrap();
            if slot >= m.book.len() {
                return Ok(None);
            }
            match m.book[slot].absorb_check(tokens.len()) {
                Ok(()) => (m.book[slot].first, m.book[slot].n),
                Err(BookError::Off | BookError::Nothing) => return Ok(None),
                Err(e) => return Err(ModelError::Request(format!("the slot's draft state: {e:?}"))),
            }
        };
        let entries = absorb_entries(first, n, false);
        let Some(&(_, store_row0, _)) = entries.first() else { return Ok(None) };
        self.mtp_ensure_positions(&[(slot, tokens.len() - 2 + depth - 1)])?;
        let rows = entries.len();
        let ids: Vec<i32> = entries.iter().map(|e| tokens[e.0] as i32).collect();
        let pos: Vec<i32> = entries.iter().map(|e| e.2 as i32).collect();
        let idx0 = RowIndex::from_host(&vec![slot as i32; rows], &pos)?;
        let mut chain = Vec::new();
        for k in 1..depth {
            chain.push(RowIndex::from_host(&[slot as i32], &[(tokens.len() - 2 + k) as i32])?);
        }
        {
            let Core { mtp, stream, .. } = &mut self.core;
            let m = mtp.as_mut().unwrap();
            m.ids.upload(&ids)?;
            // a record that is not keyed: the pick is candidate 0, and the candidate lists are what is read
            m.keyed.upload(&[KeyedRow { first_position: tokens.len() as u64, ..Default::default() }])?;
            let row = HIDDEN * 4;
            m.h_in.copy_range_from(0, &m.store, (slot * MTP_TAP_CAP + store_row0) * row, rows * row, stream)?;
        }
        uploads_visible()?;
        // every row through the whole block: the full-block oracle the production draft call (`Tail::Rows`) is compared with
        self.head_block(rows, IdsFrom::Host, HidFrom::HIn, &idx0, None, Tail::All)?;
        self.head_final_norm(&[rows - 1], 0)?;
        let restricted = restricted && self.core.mtp.as_ref().is_some_and(|m| m.sub.is_some());
        let mut out = Vec::with_capacity(depth);
        for k in 0..depth {
            if k > 0 {
                self.head_block(1, IdsFrom::Host, HidFrom::Out, &chain[k - 1], None, Tail::All)?;
                self.head_final_norm(&[0], 0)?;
            }
            self.head_pick(1, 0, k, restricted, Some(count))?;
            self.core.stream.sync()?;
            let Core { bufs, mtp, .. } = &mut self.core;
            let m = mtp.as_mut().unwrap();
            let (mut ids_h, mut vals_h) = (vec![0i32; count], vec![0f32; count]);
            bufs.topk_ids.download_range(0, &mut ids_h)?;
            bufs.topk_vals.download_range(0, &mut vals_h)?;
            let ids_u: Vec<u32> = ids_h.into_iter().map(|i| i as u32).collect();
            let t = next(k, &ids_u, &vals_h);
            if k + 1 < depth {
                m.ids.upload(&[t as i32])?;
                uploads_visible()?;
            }
            out.push((ids_u, vals_h));
        }
        if absorb {
            self.core.mtp.as_mut().unwrap().book[slot].absorbed(tokens.len());
        }
        drop((idx0, chain));
        Ok(Some(out))
    }

    /// The retained window taps' first `rows` rows (f32 `[rows][4096]`): the last window's post-`norm_f` rows, as the head will read them.
    /// Synchronises.
    #[doc(hidden)]
    pub fn mtp_tap_rows(&mut self, rows: usize) -> Result<Vec<f32>, ModelError> {
        self.core.stream.sync()?;
        let m = self.core.mtp.as_ref().ok_or_else(|| ModelError::Config("the model has no MTP head".into()))?;
        let window = m.win_tap.bytes() / (HIDDEN * 4);
        if rows > window {
            return Err(ModelError::Request(format!("{rows} tap rows of a window of {window}")));
        }
        let mut v = vec![0f32; rows * HIDDEN];
        m.win_tap.download_range(0, &mut v)?;
        Ok(v)
    }

    /// Debug (the cost measurements): makes `slot`'s store hold `taps` (f32 `[n][4096]`, `n <= MTP_TAP_CAP`) as the pending rows of entries
    /// `first .. first + n`, as if the trunk had committed them, and the slot's book say so. The trunk is not touched: the head's cache below
    /// `first - 1` is whatever it was (zeros on a fresh slot), which costs the same to attend over. Synchronises.
    #[doc(hidden)]
    pub fn mtp_debug_fill_store(&mut self, slot: usize, first: usize, taps: &[f32]) -> Result<(), ModelError> {
        let n = taps.len() / HIDDEN;
        let Core { mtp, stream, .. } = &mut self.core;
        let m = mtp.as_mut().ok_or_else(|| ModelError::Config("the model has no MTP head".into()))?;
        if slot >= m.book.len() || n == 0 || n > MTP_TAP_CAP || taps.len() != n * HIDDEN || first == 0 {
            return Err(ModelError::Request(format!("mtp_debug_fill_store: {n} rows at entry {first} of slot {slot}")));
        }
        let tmp = DevBuf::from_slice(taps)?;
        uploads_visible()?;
        m.store.copy_range_from(slot * MTP_TAP_CAP * HIDDEN * 4, &tmp, 0, n * HIDDEN * 4, stream)?;
        stream.sync()?;
        m.book[slot] = TapBook { ok: true, first, n };
        Ok(())
    }

    /// The slot's pending tapped rows (f32 `[n][4096]`, store rows `0..n`). Synchronises.
    #[doc(hidden)]
    pub fn mtp_store_rows(&mut self, slot: usize, n: usize) -> Result<Vec<f32>, ModelError> {
        self.core.stream.sync()?;
        let m = self.core.mtp.as_ref().ok_or_else(|| ModelError::Config("the model has no MTP head".into()))?;
        if slot >= m.book.len() || n > MTP_TAP_CAP {
            return Err(ModelError::Request(format!("store rows 0..{n} of slot {slot}")));
        }
        let mut v = vec![0f32; n * HIDDEN];
        m.store.download_range(slot * MTP_TAP_CAP * HIDDEN * 4, &mut v)?;
        Ok(v)
    }

    /// The head's KV rows `first..first + n` of `slot` (fp16 bits K, then V: `[n][256]` each), through the slot's page table. Synchronises.
    #[doc(hidden)]
    pub fn mtp_kv_rows(&mut self, slot: usize, first: usize, n: usize) -> Result<(Vec<u16>, Vec<u16>), ModelError> {
        if self.core.mtp.is_none() {
            return Err(ModelError::Config("the model has no MTP head".into()));
        }
        let ord = self.core.kv_geom.layers - 1;
        self.debug_kv_rows(ord, slot, first, n)
    }

    /// Debug (the gate against `mtp_ref`): run head rows at positions `start .. start + tokens.len()` of `slot` with the HOST-supplied hidden
    /// rows (`hidden`: f32 `[rows][4096]`, row `i` the tap that pairs with `tokens[i]`), in calls of at most 64 rows, and capture the last
    /// `tail` rows of every stage with the head's logits over them. Writes the head's cache of `slot` exactly as a draft does and leaves the
    /// slot's book alone. Synchronises. The embedding rows are the TRUNK's (`tokens[i]` is the token that follows the hidden row).
    #[doc(hidden)]
    pub fn mtp_debug_forward(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        hidden: &[f32],
        tail: usize,
    ) -> Result<MtpCapture, ModelError> {
        self.mtp_debug_forward_mm(slot, start, tokens, hidden, tail, &[])
    }

    /// [`NemotronModel::mtp_debug_forward`] with image rows: `splice` is in `tokens` rows (`dst_row` indexes `tokens`, `src_row` the
    /// image's features), and the embedding half of those rows is the feature row, as the prompt hook builds them. The teacher-forced
    /// rebuild of the head's cache over an image prompt: from the same taps and the same features, bit for bit.
    #[doc(hidden)]
    pub fn mtp_debug_forward_mm(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        hidden: &[f32],
        tail: usize,
        splice: &[SpliceOp],
    ) -> Result<MtpCapture, ModelError> {
        let n = tokens.len();
        if self.core.mtp.is_none() || n == 0 || hidden.len() != n * HIDDEN || tail == 0 || tail > n.min(WINDOW_ROWS) {
            return Err(ModelError::Request(
                "mtp_debug_forward: a head, tokens, one hidden row per token and a tail within them".into(),
            ));
        }
        self.core.stream.sync()?;
        let mut cap = MtpCapture::default();
        let cap_rows = self.core.mtp.as_ref().unwrap().rows_cap;
        let mut at = 0;
        while at < n {
            // the captured tail comes from the last call: keep at least `tail` rows for it (the head is row-invariant, so the cut changes no bit)
            let left = n - at;
            let end = if left > cap_rows && left - cap_rows < tail { n - tail } else { (at + cap_rows).min(n) };
            let rows = end - at;
            self.mtp_ensure_positions(&[(slot, start + end - 1)])?;
            let pos: Vec<i32> = (start + at..start + end).map(|p| p as i32).collect();
            let idx = RowIndex::from_host(&vec![slot as i32; rows], &pos)?;
            {
                let m = self.core.mtp.as_mut().unwrap();
                let ids: Vec<i32> = tokens[at..end].iter().map(|&t| t as i32).collect();
                m.ids.upload(&ids)?;
                m.h_in.upload(&hidden[at * HIDDEN..end * HIDDEN])?;
                // this call's share of the splice, in its own rows
                m.img = splice
                    .iter()
                    .filter_map(|op| {
                        let (lo, hi) = (op.dst_row.max(at), (op.dst_row + op.n).min(end));
                        (lo < hi).then(|| SpliceOp {
                            feats: op.feats.clone(),
                            dst_row: lo - at,
                            src_row: op.src_row + (lo - op.dst_row),
                            n: hi - lo,
                        })
                    })
                    .collect();
            }
            let last = end == n;
            let t = if last { tail.min(rows) } else { 0 };
            uploads_visible()?;
            let ran = self.head_block(rows, IdsFrom::Host, HidFrom::HIn, &idx, (t > 0).then_some((&mut cap, t)), Tail::All);
            if let Some(m) = self.core.mtp.as_mut() {
                m.img.clear();
            }
            ran?;
            if last {
                let all: Vec<usize> = (0..rows).collect();
                self.head_final_norm(&all, 0)?;
                self.core.stream.sync()?;
                let v = self.vocab;
                {
                    let m = self.core.mtp.as_ref().unwrap();
                    let mut o = vec![0f32; t * HIDDEN];
                    m.out.download_range((rows - t) * HIDDEN * 4, &mut o)?;
                    cap.out = o;
                }
                // the logits of the tail rows through the trunk's lm_head
                let Core { w, bufs, stream, mtp, tp, .. } = &mut self.core;
                let m = mtp.as_ref().unwrap();
                let x = m.out.subview((rows - t) * HIDDEN * 4, t * HIDDEN * 4);
                super::tp_arm::lm_head_logits(
                    &**tp,
                    (&w.lm_head).into(),
                    &x,
                    &mut bufs.logits,
                    bufs.tp_logits.as_mut(),
                    t,
                    v,
                    false,
                    &mut bufs.ws,
                    stream,
                )?;
                stream.sync()?;
                let mut l = vec![0f32; t * v];
                bufs.logits.download_range(0, &mut l)?;
                cap.logits = l;
            }
            self.core.stream.sync()?;
            drop(idx);
            at = end;
        }
        Ok(cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_keyed_record_says_where_the_draft_is_verified_and_what_the_sampler_keeps() {
        let s = Sampling { seed: 9, temperature: 0.7, top_k: 20, top_p: 0.95, min_p: 0.05 };
        // the first draft of a sequence of 300 tokens is compared with the sample of position 300 (the window's row 0 samples at `len`)
        let (r, fetch) = keyed_record(Some(&s), 300);
        assert_eq!((r.keyed, r.seed, r.first_position, r.top_k, fetch), (1, 9, 300, 20, 20));
        assert_eq!((r.temperature, r.top_p, r.min_p), (0.7, 0.95, 0.05));
        // not keyed: no sampler, a greedy request (no noise), a top_k the call cannot fetch (0 is "every candidate", past the buffers is out of reach)
        for other in [
            None,
            Some(Sampling { temperature: 0.0, ..s }),
            Some(Sampling { top_k: 0, ..s }),
            Some(Sampling { top_k: MTP_MAX_CANDIDATES + 1, ..s }),
        ] {
            let (r, fetch) = keyed_record(other.as_ref(), 300);
            assert_eq!((r.keyed, fetch, r.first_position), (0, 0, 300), "{other:?}");
        }
        // the largest top_k the server allows (56) and the largest the call can fetch (64) are keyed
        assert_eq!(keyed_record(Some(&Sampling { top_k: 56, ..s }), 10).1, 56);
        assert_eq!(keyed_record(Some(&Sampling { top_k: MTP_MAX_CANDIDATES, ..s }), 10).1, MTP_MAX_CANDIDATES);
    }

    #[test]
    fn a_prompt_chunk_builds_the_entries_whose_next_token_it_holds_and_never_entry_zero() {
        // the first chunk of a prompt: row j's token pairs with the tap of row j - 1, row 0 has no entry (nothing predicted it)
        let e = chunk_entries(0, 4, false);
        assert_eq!(
            e,
            vec![
                ChunkEntry { token_row: 1, hidden: HidSrc::Tap(0), pos: 0 },
                ChunkEntry { token_row: 2, hidden: HidSrc::Tap(1), pos: 1 },
                ChunkEntry { token_row: 3, hidden: HidSrc::Tap(2), pos: 2 },
            ]
        );
        // a one-token prompt builds nothing (its tap is the carry)
        assert!(chunk_entries(0, 1, false).is_empty());
        // a later chunk starts from the carry, which sits at the position of the previous chunk's last row
        let e = chunk_entries(128, 3, false);
        assert_eq!(
            e,
            vec![
                ChunkEntry { token_row: 0, hidden: HidSrc::Carry, pos: 127 },
                ChunkEntry { token_row: 1, hidden: HidSrc::Tap(0), pos: 128 },
                ChunkEntry { token_row: 2, hidden: HidSrc::Tap(1), pos: 129 },
            ]
        );
        // the chunk's last tap (row 2) is entered by nobody: it is the next chunk's carry
        assert!(e.iter().all(|x| x.hidden != HidSrc::Tap(2)));
        // chunks of a prompt cover every position 0 .. len - 2 exactly once
        let mut seen = Vec::new();
        for (start, rows) in [(0usize, 64usize), (64, 64), (128, 44)] {
            seen.extend(chunk_entries(start, rows, false).into_iter().map(|x| x.pos));
        }
        assert_eq!(seen, (0..171).collect::<Vec<_>>());
        // the dropped shift pairs every token with its own position's tap
        let e = chunk_entries(128, 3, true);
        assert!(e.iter().all(|x| x.hidden == HidSrc::Tap(x.token_row)));
        assert_eq!(e[0].pos, 127);
    }

    #[test]
    fn the_heads_image_rows_are_the_entries_whose_token_is_an_image_row_and_take_that_rows_features() {
        // a slice of 64 rows at position 128 whose rows 10..40 are rows 25..55 of an image (the image began 25 rows before the slice's row
        // 10): one op (dst 10, src 25, n 30)
        let ops = [(10usize, 25usize, 30usize)];
        // the chunk's entries (position 128 > 0: every row has one, the first from the carry): the absorb row k is the token at batch row k
        let rows: Vec<usize> = chunk_entries(128, 64, false).iter().map(|e| e.token_row).collect();
        assert_eq!(rows.len(), 64);
        let runs = head_image_runs(&ops, &rows);
        assert_eq!(runs, vec![(0, 10, 25, 30)], "absorb rows 10..40 take feature rows 25..55");
        // the very first slice of a prompt has no entry for row 0 (nothing predicted it): the absorb rows are shifted by one against the
        // batch rows, and the feature row of absorb row k is the one of batch row k + 1
        let rows: Vec<usize> = chunk_entries(0, 64, false).iter().map(|e| e.token_row).collect();
        assert_eq!(rows.len(), 63);
        let runs = head_image_runs(&ops, &rows);
        assert_eq!(runs, vec![(0, 9, 25, 30)], "absorb row 9 is batch row 10");
        // two images in one slice are two runs, an image cut by the slice's edge keeps its own feature offsets
        let ops = [(0usize, 40usize, 5usize), (20, 0, 8)];
        let rows: Vec<usize> = (0..32).collect();
        assert_eq!(head_image_runs(&ops, &rows), vec![(0, 0, 40, 5), (1, 20, 0, 8)]);
        // no image: nothing, and an empty plan is no work
        assert!(head_image_runs(&[], &rows).is_empty());
        assert!(head_image_runs(&ops, &[]).is_empty());
        // rows that are image rows of no op of this slice (they are in the neighbouring slice's share) are text here
        assert!(head_image_runs(&[(40, 0, 5)], &(0..32).collect::<Vec<_>>()).is_empty());
    }

    #[test]
    fn an_absorb_takes_the_pending_entries_at_the_positions_of_their_taps() {
        // after a 300-token prompt and the sampled token: entry 300 (token 300, the carry tap of position 299) at position 299
        assert_eq!(absorb_entries(300, 1, false), vec![(300, 0, 299)]);
        // three pending rows: entries 21, 22, 23 at positions 20, 21, 22, taking store rows 0, 1, 2
        assert_eq!(absorb_entries(21, 3, false), vec![(21, 0, 20), (22, 1, 21), (23, 2, 22)]);
        // the mutant absorbs only the last
        assert_eq!(absorb_entries(21, 3, true), vec![(23, 2, 22)]);
        // entry 0 does not exist
        assert_eq!(absorb_entries(0, 2, false), vec![(1, 1, 0)]);
    }

    #[test]
    fn a_vocabulary_file_is_ids_and_comments() {
        assert_eq!(parse_vocab_file("# 3 tokens\n1 2\n\n  7 # seven\n").unwrap(), vec![1, 2, 7]);
        assert!(parse_vocab_file("1 x").unwrap_err().contains("line 1"));
        assert!(parse_vocab_file("-3").is_err());
    }

    #[test]
    fn the_setting_is_the_default_list_unless_it_says_off_or_names_a_file() {
        for unset in [None, Some(""), Some("  "), Some("default"), Some("on")] {
            assert_eq!(VocabSetting::parse(unset), VocabSetting::Default, "{unset:?}");
        }
        for off in ["off", "0", "full", "none", "false", " off "] {
            assert_eq!(VocabSetting::parse(Some(off)), VocabSetting::Off, "{off}");
        }
        assert_eq!(VocabSetting::parse(Some("/tmp/v.txt")), VocabSetting::File("/tmp/v.txt".into()));
        assert_eq!(restricted_vocab_ids_for(&VocabSetting::Off).unwrap(), None);
        let e = restricted_vocab_ids_for(&VocabSetting::File("/nonexistent/vocab".into())).unwrap_err();
        assert!(e.to_string().contains("REXL3_MTP_VOCAB=/nonexistent/vocab"), "{e}");
    }

    #[test]
    fn the_default_list_is_49152_distinct_ids_of_the_vocabulary_and_its_header_says_what_it_holds() {
        let ids = restricted_vocab_ids_for(&VocabSetting::Default).unwrap().unwrap();
        assert_eq!(ids.len(), 49_152);
        assert!(ids.windows(2).all(|w| w[0] < w[1]), "sorted and distinct");
        assert!(*ids.last().unwrap() < 131_072, "inside the tokenizer's vocabulary");
        // the file is the committed recipe's output: its header names the counts the list has, and the ids are written as it sorted them
        let header: Vec<&str> = DEFAULT_VOCAB.lines().take_while(|l| l.starts_with('#')).collect();
        assert!(header[0].contains("49152 of 131072 ids"), "{header:?}");
        assert!(header.iter().any(|l| l.contains("specials 996, pure-ASCII 48156, others 0")), "{header:?}");
        assert!(
            header.iter().any(|l| l.contains("tokenizer.json sha256")),
            "the tokenizer it was derived from is named: {header:?}"
        );
        assert!(header.iter().any(|l| l.contains("corpus:") && l.contains("commit")), "the corpus is described: {header:?}");
        let listed = parse_vocab_file(DEFAULT_VOCAB).unwrap();
        assert_eq!(listed, ids, "written sorted and without repeats");
        // the head's byte count at this size is what the plan prices: 8,196 bytes a row
        assert_eq!(ids.len() as u64 * (4096 * 2 + 4), 402_849_792);
        assert_eq!(402_849_792 >> 20, 384, "MT-A: 384 MiB at 48k");
    }

    #[test]
    fn the_multilingual_union_extends_the_default_and_is_a_file_that_parses() {
        // not built in: `REXL3_MTP_VOCAB=<this file>` (docs/NEMOTRON_MTP_PLAN.md); here it is kept honest
        let union = parse_vocab_file(include_str!("mtp_vocab_62k_multilingual.txt")).unwrap();
        let base = parse_vocab_file(DEFAULT_VOCAB).unwrap();
        assert_eq!(union.len(), 62_014);
        assert!(union.windows(2).all(|w| w[0] < w[1]) && *union.last().unwrap() < 131_072);
        let set: std::collections::HashSet<u32> = union.iter().copied().collect();
        assert!(base.iter().all(|i| set.contains(i)), "the union holds every id of the default list");
        assert!(include_str!("mtp_vocab_62k_multilingual.txt").contains("62014 of 131072 ids"));
    }

    #[test]
    fn the_head_is_wanted_for_the_whole_trunk_unless_switched_off() {
        // the environment is the process's: only the layer rule is tested here
        assert!(!head_wanted(Some(8)));
    }
}
