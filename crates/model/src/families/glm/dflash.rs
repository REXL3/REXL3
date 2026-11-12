//! The DFlash2 draft model of GLM-5.3-Flash (`--draft DIR`): the checks a draft pack must pass before it is loaded, the bookkeeping of each
//! slot's draft context, and the draft call's cost model. The draft itself is [`crate::draft::Draft`] at [`DraftGeom::GLM53_FLASH`]; the
//! GLM model taps the target layers and feeds it (`model/dflash.rs`); the backend's speculation runtime asks it for drafts (`spec.rs`).
//!
//! **What the draft reads.** Each position `p` of the draft's context is `hidden_norm(fc(taps(p)))`, where `taps(p)` are the trunk's
//! hidden rows of position `p` after layers 5, 14, 24, 33 and 42 (the `target_layer_ids` of the draft's config; HF `hidden_states[i + 1]`):
//! the unweighted mean of the four mHC streams at the layer's output, before any norm, in bf16. A draft round for a sequence whose last
//! token (not yet fed to the trunk) is at position `P` needs the context of positions `P - 2048 .. P` (the draft's window).
//!
//! **Exactness.** The draft only proposes: the verify window decides every committed token, so nothing here can change an output byte.
//! Its context is built from committed rows only (a window's rejected rows never enter it), which is what makes its proposals the same
//! whatever the round grid, but that is a property of its acceptance, not of the output.
//!
//! **Licence.** The engine reads a draft pack the operator provides: the published `incoai/GLM-5.3-Flash-DFlash2` checkpoint as it is
//! (bf16, converted to the kernels' formats in memory at load), or the operator's own EXL3 quantization of it. The repository ships no draft
//! weights of any form.

use crate::draft::{DraftGeom, TAPS};
use crate::error::ModelError;
use gb_cuda::dflash2::DraftCtxLayout;
use serde_json::Value;
use std::path::Path;

use super::spec::MtpCost;

/// The trunk layers the draft taps (its config's `dflash_config.target_layer_ids`), in the order the fc reads them.
pub const TAP_LAYERS: [usize; TAPS] = DraftGeom::GLM53_FLASH.tap_layers;

/// Context positions a draft round can look back over (its window).
pub const WINDOW: usize = DraftCtxLayout::WINDOW;

/// Check a draft pack's `config.json` against the GLM-5.3-Flash DFlash2 shapes this engine runs and the target it drafts for (`hidden`,
/// `vocab`, `layers`). Returns the draft's geometry. Every field the kernels depend on is compared, and a mismatch is refused by name: a
/// draft of another shape would load half its tensors and then fail, or worse, propose nonsense.
pub fn check_config(cfg: &Value, hidden: usize, vocab: usize, layers: usize) -> Result<DraftGeom, String> {
    let g = DraftGeom::GLM53_FLASH;
    let mut bad = crate::draft::config_mismatches(cfg, &g);
    if hidden != g.hidden {
        bad.push(format!("the target's hidden size is {hidden}, the draft reads {} wide taps", g.hidden));
    }
    if vocab != g.vocab {
        bad.push(format!("the target's vocabulary is {vocab}, the draft's {}", g.vocab));
    }
    if g.tap_layers.iter().any(|&l| l >= layers) {
        bad.push(format!("the target has {layers} layers; the draft taps {:?}", g.tap_layers));
    }
    if bad.is_empty() {
        Ok(g)
    } else {
        Err(format!("the draft is not a GLM-5.3-Flash DFlash2 draft this engine runs: {}", bad.join("; ")))
    }
}

/// [`check_config`] of the `config.json` in `dir`.
pub fn check_dir(dir: &Path, hidden: usize, vocab: usize, layers: usize) -> Result<DraftGeom, ModelError> {
    let p = dir.join("config.json");
    let text = std::fs::read_to_string(&p).map_err(|e| ModelError::Config(format!("{}: {e}", p.display())))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| ModelError::Config(format!("{}: {e}", p.display())))?;
    check_config(&v, hidden, vocab, layers).map_err(|e| ModelError::Config(format!("{}: {e}", dir.display())))
}

// ---------------------------------------------------------------------------------------------------------------- bookkeeping

/// What one slot's draft context holds: the positions `lo .. end` are in the draft's ring (every position below `end` was seen by the trunk:
/// those below `lo` were skipped, being too far back for any round to read). `ok == false` switches the slot's drafting off until its next
/// sequence (a gap: a chunk or a window that does not continue where the context ends). Pure host state, tested on the CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CtxBook {
    pub ok: bool,
    pub lo: usize,
    pub end: usize,
}

impl Default for CtxBook {
    fn default() -> Self {
        Self::new()
    }
}

impl CtxBook {
    pub const fn new() -> Self {
        Self { ok: true, lo: 0, end: 0 }
    }

    /// A prompt chunk of `rows` rows from `start` was run, and the rows from `from` on are wanted in the ring (the earlier ones no round will
    /// read). Returns the positions to absorb (possibly empty), or `None` when the chunk does not continue the context (drafting is off).
    pub fn prefill(&mut self, start: usize, rows: usize, from: usize) -> Option<std::ops::Range<usize>> {
        if !self.ok || start != self.end {
            self.ok = false;
            return None;
        }
        let end = start + rows;
        let a = from.clamp(start, end);
        if a > start {
            // the rows skipped are below `from`, and so is everything the ring held before them: no round and no snapshot reads them
            // again ([`ctx_from`]), so the ring now starts at the first row absorbed
            self.lo = a;
        }
        self.end = end;
        Some(a..end)
    }

    /// A window whose first row sat at `pos0` committed `keep` rows. Returns whether they continue the context (absorb them), else drafting is off.
    pub fn window(&mut self, pos0: usize, keep: usize) -> bool {
        if !self.ok || pos0 != self.end {
            self.ok = false;
            return false;
        }
        self.end += keep;
        true
    }

    /// May a round draft for a sequence whose last token (not yet fed) is at position `p`? The ring must end at `p` and cover its window.
    pub fn ready(&self, p: usize) -> bool {
        self.ok && self.end == p && self.lo <= p.saturating_sub(WINDOW)
    }

    /// The positions a snapshot of the first `len` tokens keeps: the last [`WINDOW`] of them that the ring holds.
    pub fn window_of(&self, len: usize) -> std::ops::Range<usize> {
        len.saturating_sub(WINDOW).max(self.lo).min(len)..len
    }
}

/// The first prompt position whose taps the draft must absorb for a prompt of `n` tokens whose snapshots are taken at `cuts`: the last
/// window before the prompt's end (the first round's), or earlier, the window before the first cut still to come (a snapshot keeps its
/// window). Rows before it are never read by a round.
pub fn ctx_from(n: usize, cuts: &[usize], done: usize) -> usize {
    let first_cut = cuts.iter().filter(|&&c| c > done).map(|&c| c.saturating_sub(WINDOW)).min();
    n.saturating_sub(WINDOW).min(first_cut.unwrap_or(usize::MAX))
}

// --------------------------------------------------------------------------------------------------------------------- costs

/// The draft call's cost model for the arbiter (the [`MtpCost`] shape): the whole block runs whatever the depth (5 layers over 8 rows
/// a sequence, then the target's `lm_head` over the rows asked for, and the selector's walk), so the call is a base that grows with the
/// draft's weight bytes, a little per sequence, and almost nothing per extra draft. The context absorb happens at the commit, not here.
/// MEASURED on a GB10 node (`docs/GLM53_DFLASH2.md`): see [`dflash_cost`].
pub fn dflash_cost(weight_bytes: u64) -> MtpCost {
    // the block reads every draft weight once a round (about 175 GB/s achieved on these shapes) plus the launch chain; the head is the
    // trunk's 6-bit lm_head over the asked rows (one read); each extra draft is one more head row and walk step
    let gib = weight_bytes as f64 / (1u64 << 30) as f64;
    MtpCost {
        base: 1.5 + gib * 1e3 / 175.0,
        per_stream: 0.35,
        rows: &[(1, 0.0), (64, 0.0)],
        head: 2.1,
        chain: 0.05,
        chain_per_stream: 0.02,
    }
}

/// The most drafts one round proposes (the block's rows past the anchor).
pub const MAX_DEPTH: usize = crate::draft::DRAFTS;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The published draft's config as far as this engine reads it (the facts the checks compare; not a copy of the file).
    fn cfg() -> Value {
        json!({
            "architectures": ["DFlash2DraftModel"],
            "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 5, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128, "rms_norm_eps": 1e-5, "sliding_window": 2048, "vocab_size": 154880,
            "is_causal": false, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 154856,
                "selector_rank": 256, "selector_top_k": 16, "target_layer_ids": [5, 14, 24, 33, 42]}
        })
    }

    #[test]
    fn the_published_config_is_accepted_and_every_field_is_checked() {
        assert_eq!(check_config(&cfg(), 4096, 154880, 45), Ok(DraftGeom::GLM53_FLASH));
        for (path, v) in [
            (vec!["hidden_size"], json!(5120)),
            (vec!["rms_norm_eps"], json!(1e-6)),
            (vec!["is_causal"], json!(true)),
            (vec!["vocab_size"], json!(248320)),
            (vec!["rope_parameters", "rope_theta"], json!(1e7)),
            (vec!["dflash_config", "mask_token_id"], json!(248070)),
            (vec!["dflash_config", "target_layer_ids"], json!([5, 19, 33, 47, 61])),
            (vec!["dflash_config", "block_size"], json!(16)),
            (vec!["architectures"], json!(["Qwen3ForCausalLM"])),
        ] {
            let mut c = cfg();
            let mut at = &mut c;
            for k in &path[..path.len() - 1] {
                at = at.get_mut(*k).unwrap();
            }
            at[path[path.len() - 1]] = v;
            let e = check_config(&c, 4096, 154880, 45).unwrap_err();
            assert!(e.contains(path[path.len() - 1]) || e.contains("architecture"), "{path:?}: {e}");
        }
        // a missing field is refused, and so is a target the taps do not fit
        let mut c = cfg();
        c.as_object_mut().unwrap().remove("head_dim");
        assert!(check_config(&c, 4096, 154880, 45).unwrap_err().contains("head_dim missing"));
        assert!(check_config(&cfg(), 4096, 154880, 40).unwrap_err().contains("40 layers"));
        assert!(check_config(&cfg(), 5120, 154880, 45).unwrap_err().contains("hidden size is 5120"));
    }

    #[test]
    fn a_prompt_fills_the_ring_from_its_last_window_and_windows_continue_it() {
        // a prompt of 5000 tokens in chunks of 2048, the ring wanted from 5000 - 2048 = 2952
        let mut b = CtxBook::new();
        let from = ctx_from(5000, &[], 0);
        assert_eq!(from, 2952);
        assert_eq!(b.prefill(0, 2048, from), Some(2048..2048), "nothing of the first chunk is read by any round");
        assert_eq!(b.prefill(2048, 2048, from), Some(2952..4096));
        assert_eq!(b.prefill(4096, 904, from), Some(4096..5000));
        assert_eq!((b.lo, b.end), (2952, 5000));
        assert!(b.ready(5000) && !b.ready(4999) && !b.ready(5001));
        // a window at 5000 kept 3 rows: the next round's anchor is at 5003
        assert!(b.window(5000, 3));
        assert!(b.ready(5003));
        // a window that does not continue the context switches the slot off
        assert!(!b.window(5007, 1));
        assert!(!b.ok && !b.ready(5008));
        assert_eq!(b.prefill(5008, 4, 0), None);
    }

    #[test]
    fn a_short_prompt_is_absorbed_whole_and_skipped_rows_move_the_ring_start() {
        let mut b = CtxBook::new();
        assert_eq!(b.prefill(0, 100, ctx_from(100, &[], 0)), Some(0..100));
        assert!(b.ready(100));
        // a restored prefix (ring 2000..4048) followed by a long suffix: the rows below the suffix's last window are skipped and the ring
        // starts again where the absorbed rows start (what it held before is never read)
        let mut c = CtxBook { ok: true, lo: 2000, end: 4048 };
        let from = ctx_from(10_000, &[], 4048);
        assert_eq!(c.prefill(4048, 4096, from), Some(7952..8144));
        assert_eq!(c.lo, 7952);
        assert_eq!(c.prefill(8144, 1856, from), Some(8144..10_000));
        assert!(c.ready(10_000));
        // a short suffix after a restore continues the restored window
        let mut r = CtxBook { ok: true, lo: 2000, end: 4048 };
        assert_eq!(r.prefill(4048, 10, ctx_from(4058, &[], 4048)), Some(4048..4058));
        assert!(r.ready(4058) && r.lo == 2000);
        // a chunk that does not start where the context ends
        let mut d = CtxBook::new();
        assert_eq!(d.prefill(5, 10, 0), None);
        assert!(!d.ok);
    }

    #[test]
    fn a_cut_still_to_come_pulls_the_ring_start_back_and_its_snapshot_keeps_its_window() {
        // a prompt of 10000 tokens with snapshots at 4096 and 8192: the ring must hold 4096 - 2048 onward
        assert_eq!(ctx_from(10_000, &[4096, 8192], 0), 2048);
        // once the first cut is behind, the second one decides
        assert_eq!(ctx_from(10_000, &[4096, 8192], 4096), 6144);
        assert_eq!(ctx_from(10_000, &[4096, 8192], 9000), 10_000 - WINDOW);
        let mut b = CtxBook::new();
        let from = ctx_from(10_000, &[4096, 8192], 0);
        b.prefill(0, 4096, from).unwrap();
        assert_eq!(b.window_of(4096), 2048..4096);
        // a snapshot of fewer tokens than the window keeps them all; one below the ring's start keeps what the ring has
        assert_eq!(CtxBook::new().window_of(100), 0..100);
        assert_eq!(CtxBook { ok: true, lo: 3000, end: 4000 }.window_of(4000), 3000..4000);
    }

    #[test]
    fn the_cost_is_flat_in_depth_and_grows_with_the_weights() {
        let bf16 = dflash_cost(2_342_160_896);
        let third = dflash_cost(2_342_160_896 / 3);
        assert!(bf16.ms(1, 0, 7) > third.ms(1, 0, 7) + 5.0, "the published bf16 draft reads three times the bytes");
        assert!(bf16.ms(1, 0, 7) - bf16.ms(1, 0, 1) < 1.0, "a deeper draft costs almost nothing more");
        assert!(bf16.ms(4, 0, 7) > bf16.ms(1, 0, 7));
    }
}
