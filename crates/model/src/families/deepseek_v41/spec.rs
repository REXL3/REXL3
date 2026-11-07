//! The DSpark draft stack in family-neutral terms: three chained NextN stages, the tap that feeds them, and the rule that
//! decides how much of a block the trunk keeps.
//!
//! This is the model-side half of `crates/cuda/src/dsv4_mtp.rs` (which holds the arithmetic). Nothing here touches a weight;
//! it is the declaration a load plan and a serving loop need, and the place the pack's three-way structure is validated
//! before anything is allocated.
//!
//! # Why three stages and not depth-`K` on one
//!
//! `num_nextn_predict_layers` is 3 and `dspark_target_layer_ids` has 3 entries, one per NextN layer
//! (`config.rs` refuses a pack where those disagree). Each `mtp.i` is a full MLA layer with its own KV cache plus a
//! 128-expert MoE, and `Transformer.forward_spec` runs them **in sequence**, feeding each stage's residual stream to the
//! next:
//!
//! ```text
//!   h, main_x = mtp[0].forward_embed(main_hidden, input_ids)   # main_x = main_norm(main_proj(cat taps))
//!   pre_mix   = make_identity_pre_mix(h, hc_mult)
//!   for layer in mtp:                                          # 0 -> 1 -> 2
//!       h, pre_mix = layer(h, start_pos, pre_mix, main_x)      # the SAME main_x at every stage
//!   out = mtp[-1].forward_head(h, pre_mix, input_ids)          # block_size + 1 ids, logits, confidence
//! ```
//!
//! So a depth-`K` loop over one set of weights is *not* this model: stage 1 does not share stage 0's `wq_a`, and stage 0 is
//! the only stage with `main_proj`. Running the same stage three times would produce a different residual stream, which is
//! what `tests/dsv4_mtp.rs::the_chain_is_order_sensitive` pins.
//!
//! # The tap
//!
//! `Transformer.forward` appends `h.mean(dim=2)` **before** `layer(h, ...)` runs for `i in target_layer_ids`, i.e. the
//! *input* of layer 37, 38 and 39 — not their output. With `hc_mult` residual copies, the mean over the copy axis is the
//! collapsed stream; the three are concatenated in `target_layer_ids` order (`[5120 * 3] = 15360`) and projected once by
//! `mtp.0.main_proj`.
//!
//! # The block, and the acceptance rule
//!
//! A step proposes `block_size + 1` ids: the trunk's own sample in slot 0, then `block_size` drafts. The drafts are not
//! independent — slot `k`'s logits are biased by `markov_head(ids[k - 1])`, so the block is a Markov walk through a
//! rank-256 token embedding. Acceptance is therefore a **prefix** ([`accepts`]): the longest run of draft confidences at or
//! above the threshold, capped at `block_size`. Best case is `block_size` accepted drafts on top of the anchor.

use crate::families::deepseek_v41::config::DeepseekV41Config;

/// One stage's role in the chain. The reference branches on `stage_id == 0` / `stage_id == n_mtp_layers - 1`, so this is
/// the whole of the stage-to-stage difference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageRole {
    /// Stage 0: owns `main_proj`/`main_norm`, reads the trunk taps.
    Tap,
    /// A middle stage: a full MLA + MoE layer on the previous stage's stream.
    Chain,
    /// The last stage: also owns `norm`, `markov_head` and `confidence_head`.
    Head,
}

/// What a stage carries between steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cache {
    /// `[window][head_dim]` ring of raw KV latents, one per stage — `DSparkAttention.window_kv_cache`.
    WindowKv,
}

/// One `mtp.i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageSpec {
    /// `i`, and the `n_layers + i` this stage answers to as a layer id.
    pub index: usize,
    pub layer_id: usize,
    pub role: StageRole,
    pub cache: Cache,
    /// The stage's own tensors, under the `mtp.i.` prefix; empty for the borrowed trunk tables.
    pub tensors: &'static [&'static str],
}

/// The three stages plus the arbiter's constants.
#[derive(Clone, Debug, PartialEq)]
pub struct DsparkSpec {
    pub stages: Vec<StageSpec>,
    /// `dspark_block_size`.
    pub block_size: usize,
    /// `dspark_noise_token_id`: the embedding that fills slots `1..block_size` of the draft block.
    pub noise_token_id: u32,
    /// The trunk layers whose attention input is tapped.
    pub target_layer_ids: Vec<usize>,
    /// `dspark_markov_rank`.
    pub markov_rank: usize,
    /// `dspark_n_routed_experts` / `dspark_num_experts_per_tok`.
    pub n_routed_experts: usize,
    pub topk: usize,
    /// `main_proj`'s input width: `hidden * target_layer_ids.len()`.
    pub tap_width: usize,
    /// The confidence head's input width: `hidden + markov_rank`.
    pub conf_in: usize,
    /// The level-1 candidate walk the draft attention inherits from `candidate_source_layer_id`.
    pub candidate_source_layer: usize,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
}

/// The tensors a stage owns, besides the shared `embed`/`head` and its `attn`/`ffn` weights (which `inventory.rs` already
/// enumerates under the same prefix).
const STAGE_TENSORS: &[&str] =
    &["attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight", "attn.wo_a.weight", "attn.wo_b.weight"];
const TAP_TENSORS: &[&str] = &["main_proj.weight", "main_norm.weight"];
const HEAD_TENSORS: &[&str] =
    &["norm.weight", "markov_head.embed.weight", "markov_head.head.weight", "confidence_head.proj.weight"];

impl DsparkSpec {
    /// Reads the stack out of a parsed config, refusing a pack whose NextN layers cannot be this chain.
    ///
    /// Every check here is one the reference asserts or implies: `dspark_target_layer_ids` has one entry per NextN layer
    /// (`assert len(args.dspark_target_layer_ids) > 0`), the noise token is a real vocabulary row (`new_full` would wrap
    /// silently otherwise), and the block is non-empty (`range(self.block_size)` would propose nothing).
    pub fn from_config(c: &DeepseekV41Config) -> Result<Self, String> {
        if c.nextn == 0 {
            return Err("deepseek_v41: the pack has no NextN layers, so it has no DSpark draft stack".into());
        }
        if c.dspark.target_layer_ids.len() != c.nextn {
            return Err(format!(
                "deepseek_v41: {} target layers for {} NextN layers; each stage taps exactly one trunk layer",
                c.dspark.target_layer_ids.len(),
                c.nextn
            ));
        }
        for &l in &c.dspark.target_layer_ids {
            if l >= c.layers {
                return Err(format!("deepseek_v41: target layer {l} is not a trunk layer ({} layers)", c.layers));
            }
        }
        if c.dspark.block_size == 0 {
            return Err("deepseek_v41: dspark_block_size is 0, so a step would propose no drafts".into());
        }
        if c.dspark.noise_token_id as usize >= c.vocab {
            return Err(format!(
                "deepseek_v41: dspark_noise_token_id {} is outside the vocabulary ({} rows)",
                c.dspark.noise_token_id, c.vocab
            ));
        }
        if c.dspark.markov_rank == 0 || c.dspark.n_routed_experts == 0 || c.dspark.num_experts_per_tok == 0 {
            return Err("deepseek_v41: the markov rank and the expert counts must be positive".into());
        }
        if c.dspark.num_experts_per_tok > c.dspark.n_routed_experts {
            return Err("deepseek_v41: dspark_num_experts_per_tok exceeds dspark_n_routed_experts".into());
        }

        let stages = (0..c.nextn)
            .map(|i| {
                let role = if i == 0 {
                    StageRole::Tap
                } else if i + 1 == c.nextn {
                    StageRole::Head
                } else {
                    StageRole::Chain
                };
                StageSpec { index: i, layer_id: c.layers + i, role, cache: Cache::WindowKv, tensors: STAGE_TENSORS }
            })
            .collect();

        Ok(Self {
            stages,
            block_size: c.dspark.block_size,
            noise_token_id: c.dspark.noise_token_id,
            target_layer_ids: c.dspark.target_layer_ids.clone(),
            markov_rank: c.dspark.markov_rank,
            n_routed_experts: c.dspark.n_routed_experts,
            topk: c.dspark.num_experts_per_tok,
            tap_width: c.hidden * c.dspark.target_layer_ids.len(),
            conf_in: c.hidden + c.dspark.markov_rank,
            candidate_source_layer: c.indexer.candidate_source_layer_id,
            candidate_topk_blocks: c.indexer.candidate_topk_blocks,
            candidate_block_size: c.indexer.candidate_block_size,
        })
    }

    /// The `mtp.i.` tensor names the stages that own them imply: `main_proj`/`main_norm` on stage 0, the markov and
    /// confidence heads on the last stage. Names only — shapes are `inventory.rs`'s job.
    pub fn extra_tensors(&self) -> Vec<String> {
        let mut out = Vec::new();
        for s in &self.stages {
            let p = format!("mtp.{}.", s.index);
            out.extend(s.tensors.iter().map(|t| format!("{p}{t}")));
            match s.role {
                StageRole::Tap => out.extend(TAP_TENSORS.iter().map(|t| format!("{p}{t}"))),
                StageRole::Head => out.extend(HEAD_TENSORS.iter().map(|t| format!("{p}{t}"))),
                StageRole::Chain => {}
            }
        }
        out
    }

    /// The block's input ids: the anchor, then `block_size` noise tokens (`forward_embed`'s `new_full`).
    pub fn draft_input_ids(&self, anchor: i32) -> Vec<i32> {
        let mut ids = vec![self.noise_token_id as i32; self.block_size];
        ids[0] = anchor;
        ids
    }

    /// The most tokens one step can emit: the anchor plus every drafted token. `0` would mean the stack proposes nothing.
    pub fn best_case_tokens(&self) -> usize {
        self.block_size + 1
    }

    /// The number of drafts an arbiter keeps, given one block's confidences.
    ///
    /// A prefix, because `d_k` was sampled from `logits_k + markov_head(d_{k-1})`: rejecting `d_1` and keeping `d_3` keeps a
    /// token conditioned on a context the trunk discarded. `model.py` has no rejection loop of its own — it returns
    /// `confidence` and stops — so this rule is the serving contract, and it lives here so it is one function and not a
    /// comparison scattered through the loop.
    pub fn accepts(&self, confidence: &[f32], threshold: f32) -> usize {
        confidence.iter().take_while(|&&c| c >= threshold).count().min(self.block_size)
    }
}

// ------------------------------------------------------------------------------------------------------------------------------
// Traffic learning (T2.5b)
// ------------------------------------------------------------------------------------------------------------------------------

/// `REXL3_DSV41_SPEC_LEARN`: the DSpark verify budget's traffic-learning knob — this family's analog of the Qwen
/// engine's `REXL3_SPEC_LEARN` (`docs/SERVING.md`, "Configuration"). Off by default (unset, `0`, `false`, `off`):
/// the verify-length schedule (`model.rs::propose_mtp` -> `DsparkGpu::schedule_verify_len`) then reads the
/// confidence head's own numbers only, so a round's `usage.spec` counters are a function of the request alone.
/// On (`1`, `true`, `on`) the schedule reads confidences blended with what the traffic actually accepted: an EMA
/// of the per-position accepted/reached counts of the chain rounds the model has committed — the same
/// decayed-count shape as `gb_spec::stats::AcceptCurve`, the engine's own per-stream curve — so the
/// `drafted`/`accepted` counters then depend on earlier traffic, never on what text a request gets. That is
/// exactly the Qwen knob's documented trade ("the budget learns its priors from the traffic").
///
/// The priors this learns are the per-position acceptance rates; the head's own confidence for a round acts as
/// the prior of the blend ([`Self::PRIOR_WEIGHT`] pseudo-trials), so a position the traffic has never tested keeps
/// the head's number exactly, and a position with a history needs that history to outvote the head.
///
/// Nothing here can move a byte of a reply. The schedule only decides HOW MANY of the walked chain's tokens ride
/// the verify window; the trunk still verifies every one against the serial computation and keeps the longest
/// matching prefix (`accepts` above, the engine's `accept`), so the knob changes speed, never text. Tree rounds
/// (`REXL3_DSV41_SPEC_TREE`) do not feed the EMA: their acceptance is a path, not a chain prefix, and their fanout
/// already has its own confidence schedule (`tree.rs::schedule_branch`).
#[derive(Clone, Debug, PartialEq)]
pub struct SpecLearn {
    /// Decayed count of committed rounds that tested draft position `j` (position `j + 1`, 1-based).
    reached: [f64; super::model::WINDOW_ROWS - 1],
    /// Decayed count of committed rounds whose draft at position `j` was accepted.
    accepted: [f64; super::model::WINDOW_ROWS - 1],
}

impl SpecLearn {
    /// The EMA's per-round decay: every count is multiplied by this before a round's evidence is added, so `0.9`
    /// forgets a round's evidence after about ten rounds — `gb_spec::stats::AcceptCurve`'s own constant (the
    /// Qwen engine's per-stream curves), kept the same so the two EMAs read alike.
    pub const DECAY: f64 = 0.9;

    /// How many pseudo-trials the confidence head's own number counts as in the blend (`PositionStats::
    /// conditional`'s `weight` shape): the head is a trained classifier, so its word is worth eight rounds of
    /// traffic before any evidence arrives, and the traffic needs a history to outvote it.
    pub const PRIOR_WEIGHT: f64 = 8.0;

    pub fn new() -> Self {
        Self { reached: [0.0; super::model::WINDOW_ROWS - 1], accepted: [0.0; super::model::WINDOW_ROWS - 1] }
    }

    /// Parse the knob's value the way the Qwen engine parses its switch (`costs::parse_switch`): unset, empty,
    /// `0`, `false` and `off` are off; `1`, `true` and `on` are on; anything else is refused by name.
    pub fn parse(value: Option<&str>) -> Result<bool, String> {
        match value.map(str::trim).filter(|v| !v.is_empty()) {
            None | Some("0") | Some("false") | Some("off") => Ok(false),
            Some("1") | Some("true") | Some("on") => Ok(true),
            Some(other) => Err(format!("REXL3_DSV41_SPEC_LEARN={other:?}: expected 1/true/on or 0/false/off")),
        }
    }

    /// The knob from the runtime environment (`REXL3_DSV41_KNOBS_FILE` or the process env, like the family's
    /// other `REXL3_DSV41_*` knobs). A value that does not parse is an error the boot refuses on.
    pub fn from_env() -> Result<Option<Self>, String> {
        match Self::parse(gb_cuda::dsv4_moe::runtime_env("REXL3_DSV41_SPEC_LEARN").as_deref())? {
            true => Ok(Some(Self::new())),
            false => Ok(None),
        }
    }

    /// Record one committed chain round: `verified` drafts rode the verify window, the first `accepted` matched
    /// the trunk's own samples. Positions `1..=min(verified, accepted + 1)` were tested — a rejected draft ends
    /// what the window proves, so nothing past `accepted + 1` was — which is `AcceptCurve::update`'s own
    /// counting, with the decay applied on every update. A round with no drafts records nothing ([`Self::relax`]
    /// is what fades the estimates then).
    pub fn record(&mut self, verified: usize, accepted: usize) {
        if verified == 0 {
            return;
        }
        debug_assert!(accepted <= verified, "a round cannot accept more drafts than it verified");
        let tested = verified.min(accepted + 1);
        for j in 0..self.reached.len() {
            self.reached[j] *= Self::DECAY;
            self.accepted[j] *= Self::DECAY;
            if j < tested {
                self.reached[j] += 1.0;
            }
            if j < accepted {
                self.accepted[j] += 1.0;
            }
        }
    }

    /// A round that verified no drafts: the counts fade one step toward nothing, so an estimate goes stale
    /// instead of freezing — the mirror of the engine's `AcceptCurve::relax` for a stream budgeted no drafts.
    pub fn relax(&mut self) {
        for j in 0..self.reached.len() {
            self.reached[j] *= Self::DECAY;
            self.accepted[j] *= Self::DECAY;
        }
    }

    /// Decayed number of rounds that tested position `j` (0-based).
    pub fn trials(&self, j: usize) -> f64 {
        self.reached.get(j).copied().unwrap_or(0.0)
    }

    /// The traffic-learned chance draft `j + 1` is accepted given `1..j` were: the decayed evidence as a
    /// posterior mean with `weight` pseudo-trials at `prior` (`PositionStats::conditional`'s blend). With no
    /// evidence it is the prior.
    pub fn conditional(&self, j: usize, prior: f64, weight: f64) -> f64 {
        match (self.reached.get(j), self.accepted.get(j)) {
            (Some(&r), Some(&a)) if r + weight > 0.0 => ((a + weight * prior) / (r + weight)).clamp(0.0, 1.0),
            _ => prior.clamp(0.0, 1.0),
        }
    }

    /// Blend one round's head confidences with the learned curve: the confidence the verify-length schedule
    /// reads becomes the posterior mean of the traffic's per-position acceptance with the head's own number as
    /// prior. A position the traffic never tested keeps the head's number exactly; a position that has been
    /// accepting keeps a pessimistic head from vetoing depth the traffic has earned, and a position that has
    /// been rejecting drags a confident head down.
    pub fn adjust(&self, conf: &[f32]) -> Vec<f32> {
        conf.iter().enumerate().map(|(j, &c)| self.conditional(j, f64::from(c), Self::PRIOR_WEIGHT) as f32).collect()
    }

    /// Whether any round has tested a draft position: while it is false, [`Self::adjust`] is the identity (the
    /// blend's prior weight leaves the head's numbers unchanged), so a device-side schedule over the raw head
    /// confidences is byte-identical to the host schedule over the blended ones. Once evidence exists, the host
    /// blend is no longer the identity and the host path is used again.
    pub fn has_evidence(&self) -> bool {
        self.reached.iter().any(|&r| r > 0.0)
    }
}

impl Default for SpecLearn {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = include_str!("../../../../../testdata/deepseek_v41/config.json");

    fn cfg() -> DeepseekV41Config {
        let v: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
        DeepseekV41Config::from_json(&v).expect("the pack's config parses")
    }

    fn spec() -> DsparkSpec {
        DsparkSpec::from_config(&cfg()).expect("the pack is a three-stage DSpark stack")
    }

    #[test]
    fn the_pack_is_three_chained_stages_with_one_tap_each() {
        let s = spec();
        assert_eq!(s.stages.len(), 3);
        assert_eq!(s.stages.iter().map(|x| x.role).collect::<Vec<_>>(), vec![StageRole::Tap, StageRole::Chain, StageRole::Head]);
        assert_eq!(s.stages.iter().map(|x| x.layer_id).collect::<Vec<_>>(), vec![40, 41, 42]);
        assert_eq!(s.target_layer_ids, vec![37, 38, 39]);
        // the tap is the concatenation of three 5120-wide streams, and the head reads hidden + rank
        assert_eq!(s.tap_width, 5120 * 3);
        assert_eq!((s.conf_in, s.markov_rank), (5376, 256));
        assert_eq!((s.block_size, s.noise_token_id), (5, 128_799));
        assert_eq!((s.n_routed_experts, s.topk), (128, 3));
        // the candidate walk the draft attention inherits
        assert_eq!((s.candidate_source_layer, s.candidate_topk_blocks, s.candidate_block_size), (20, 2048, 8));
    }

    #[test]
    fn only_the_first_stage_projects_the_taps_and_only_the_last_has_a_head() {
        let s = spec();
        let names = s.extra_tensors();
        assert!(names.iter().any(|t| t == "mtp.0.main_proj.weight"), "{names:?}");
        assert!(names.iter().any(|t| t == "mtp.2.markov_head.embed.weight"), "{names:?}");
        assert!(names.iter().any(|t| t == "mtp.2.confidence_head.proj.weight"), "{names:?}");
        assert!(!names.iter().any(|t| t == "mtp.1.main_proj.weight"), "stage 1 is not a tap");
        assert!(!names.iter().any(|t| t == "mtp.1.markov_head.embed.weight"), "stage 1 is not the head");
        // every stage owns its own attention and its own cache
        for i in 0..3 {
            assert!(names.iter().any(|t| t == &format!("mtp.{i}.attn.wkv.weight")));
        }
    }

    #[test]
    fn the_draft_block_is_the_anchor_then_noise() {
        let s = spec();
        assert_eq!(s.draft_input_ids(42), vec![42, 128_799, 128_799, 128_799, 128_799]);
        assert_eq!(s.best_case_tokens(), 6, "the anchor plus five drafts");
    }

    #[test]
    fn acceptance_is_a_prefix_capped_at_the_block() {
        let s = spec();
        assert_eq!(s.accepts(&[0.9, 0.9, 0.9, 0.9, 0.9], 0.5), 5);
        assert_eq!(s.accepts(&[0.9, 0.9, 0.1, 0.9, 0.9], 0.5), 2, "a rejected draft ends the block");
        assert_eq!(s.accepts(&[0.1, 0.9, 0.9, 0.9, 0.9], 0.5), 0);
        assert_eq!(s.accepts(&[0.5; 5], 0.5), 5, "the threshold is inclusive");
        // a longer confidence vector cannot buy more than block_size tokens
        assert_eq!(s.accepts(&[0.9; 9], 0.5), 5);
    }

    #[test]
    fn a_pack_whose_taps_do_not_match_its_stages_is_refused_by_name() {
        let mut c = cfg();
        c.dspark.target_layer_ids.truncate(2);
        let e = DsparkSpec::from_config(&c).unwrap_err();
        assert!(e.contains("target layers for 3 NextN layers"), "{e}");

        let mut c = cfg();
        c.dspark.noise_token_id = 999_999;
        let e = DsparkSpec::from_config(&c).unwrap_err();
        assert!(e.contains("outside the vocabulary"), "{e}");

        let mut c = cfg();
        c.nextn = 0;
        assert!(DsparkSpec::from_config(&c).unwrap_err().contains("no NextN layers"));
    }

    // ------------------------------------------------------------------ T2.5b: REXL3_DSV41_SPEC_LEARN

    #[test]
    fn the_spec_learn_knob_parses_like_the_qwen_engine_switch() {
        assert!(!SpecLearn::parse(None).unwrap(), "unset is off (the default)");
        for off in ["", "0", "false", "off"] {
            assert!(!SpecLearn::parse(Some(off)).unwrap(), "{off:?} is off");
        }
        for on in ["1", "true", "on"] {
            assert!(SpecLearn::parse(Some(on)).unwrap(), "{on:?} is on");
        }
        let e = SpecLearn::parse(Some("yes")).unwrap_err();
        assert!(e.contains("REXL3_DSV41_SPEC_LEARN"), "refused by name: {e}");
        assert!(e.contains("yes"), "the refusal shows the value: {e}");
    }

    #[test]
    fn record_is_an_ema_of_the_committed_rounds() {
        let mut l = SpecLearn::new();
        // ten rounds of 3 drafts, all accepted: every count saturates at sum(0.9^0..=9) = 6.5132.. trials
        for _ in 0..10 {
            l.record(3, 3);
        }
        assert!((l.trials(0) - 6.5132156).abs() < 1e-6, "trials(0) = {}", l.trials(0));
        assert!((l.conditional(0, 0.0, 0.0) - 1.0).abs() < 1e-9);
        // then a round of 3 drafts where only the first accepted: position 1 is tested once and fails
        l.record(3, 1);
        assert!((l.conditional(0, 0.0, 0.0) - 1.0).abs() < 1e-9, "position 0 still always accepted");
        assert!((l.conditional(1, 0.0, 0.0) - 5.8618940 / 6.8618940).abs() < 1e-6, "position 1 failed once");
        // the evidence outweighs a mediocre prior: 6.86 accepted of 6.86 reached at weight 8 gives (6.86+4)/14.86
        assert!((l.conditional(0, 0.5, 8.0) - 0.7308553).abs() < 1e-4, "{}", l.conditional(0, 0.5, 8.0));
        assert!(l.conditional(0, 0.5, 8.0) > 0.5, "pulled toward the traffic, away from the prior");
    }

    #[test]
    fn adjust_blends_the_head_confidence_with_the_traffic() {
        let mut l = SpecLearn::new();
        // a cold engine: the head's numbers are the schedule's input unchanged (the knob-on path with no
        // evidence is the knob-off path)
        assert_eq!(l.adjust(&[0.30, 0.80]), vec![0.30, 0.80]);
        // twenty rounds where the first two drafts always accepted
        for _ in 0..20 {
            l.record(2, 2);
        }
        let a = l.adjust(&[0.10, 0.95]);
        assert!(a[0] > 0.5, "a pessimistic head is pulled toward the traffic's 1.0: {a:?}");
        assert!(a[0] < 1.0, "the blend never certifies above the evidence: {a:?}");
        assert!(a[1] > 0.95 && a[1] < 1.0, "a confident head is nudged up, not past 1: {a:?}");
        // then twenty rounds where the first draft always failed
        for _ in 0..20 {
            l.record(2, 0);
        }
        let b = l.adjust(&[0.95, 0.50]);
        assert!(b[0] < 0.7, "rejecting traffic drags a confident head below the splice's \"sure\" cut: {b:?}");
        assert!(b[1] > 0.5 && b[1] < 0.8, "an untested position fades toward the prior, it does not certify: {b:?}");
    }

    #[test]
    fn a_round_without_drafts_relaxes_instead_of_recording() {
        let mut l = SpecLearn::new();
        l.record(2, 2);
        assert!((l.trials(0) - 1.0).abs() < 1e-12);
        l.relax();
        assert!((l.trials(0) - SpecLearn::DECAY).abs() < 1e-12, "relax fades one decay step");
        l.record(0, 1);
        assert!((l.trials(0) - SpecLearn::DECAY).abs() < 1e-12, "a no-draft round records nothing");
    }

    #[test]
    fn learning_deepens_the_verify_length_the_traffic_has_earned() {
        use gb_cuda::dsv4_mtp_gpu::DsparkGpu;
        // a head that reports one sure draft then a mediocre chain: the static schedule stops early (its
        // expected-token walk multiplies 0.5s against a cycle table that rises faster)
        let conf = vec![0.90f32, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50];
        let static_n = DsparkGpu::schedule_verify_len(&conf, 0.38, 7);
        // traffic that kept accepting four drafts deep earns a longer window
        let mut l = SpecLearn::new();
        for _ in 0..60 {
            l.record(4, 4);
        }
        let learned = DsparkGpu::schedule_verify_len(&l.adjust(&conf), 0.38, 7);
        assert!(learned > static_n, "static {static_n}, learned {learned}");
        // and with no evidence the knob-on path is exactly the static path
        let cold = DsparkGpu::schedule_verify_len(&SpecLearn::new().adjust(&conf), 0.38, 7);
        assert_eq!(cold, static_n);
    }
}
