//! The DFlash2 draft model of Qwen3.8-Flash-Next (`--draft DIR`): the checks a draft pack must pass before it is loaded and the trunk
//! layers it taps. The draft itself is [`crate::draft::Draft`] at [`DraftGeom::FLASH_NEXT`]; the Flash-Next model taps the trunk and
//! feeds it (`model_dflash.rs`); the backend asks it for drafts (`backend.rs`).
//!
//! **What the draft reads.** Each position `p` of the draft's context is `hidden_norm(fc(taps(p)))`, where `taps(p)` are the trunk's
//! rows of position `p` at the output of layers 5, 15, 25, 35 and 45 (zero-based, completed layers: the `target_layer_ids` of the
//! draft's config): the unweighted mean of the four hyper-connection streams, before any norm, in bf16, 2560 wide each. That is the
//! capture convention the draft was trained with. A draft round for a sequence whose last token (not yet fed to the trunk) is at position
//! `P` needs the context of positions `P - 2048 .. P` (the draft's window).
//!
//! **Exactness.** The draft only proposes: the verify window decides every committed token, so nothing here can change an output byte.
//! Its context is built from committed rows only (a window's rejected rows never enter it).
//!
//! **Acceptance.** A `--draft` pack is accepted on its structure ([`crate::draft::check_draft`] at [`DraftGeom::FLASH_NEXT`]: its config,
//! every tensor's name, shape, dtype and EXL3 width, and the target's facts), with no flag; the pair this release was measured with is
//! named in `fingerprint::VALIDATED_FLASH_NEXT_DRAFT`, and the boot reports it as listed.

use super::config::FlashNextConfig;
use crate::draft::{DraftCheck, DraftExpect, DraftGeom, TAPS};
use crate::error::ModelError;
use std::path::Path;

/// The slot bookkeeping of a draft context and the first prompt row it absorbs: the same host logic as GLM's (no device state).
pub use crate::families::glm::dflash::{ctx_from, CtxBook, WINDOW};

/// The trunk layers the draft taps (its config's `dflash_config.target_layer_ids`), in the order the fc reads them.
pub const TAP_LAYERS: [usize; TAPS] = DraftGeom::FLASH_NEXT.tap_layers;

/// Which tap column the output of trunk layer `layer` goes to, if any.
pub fn tap_of(layer: usize) -> Option<usize> {
    TAP_LAYERS.iter().position(|&l| l == layer)
}

/// The structural check of the draft in `dir` for a Flash-Next target of `cfg` ([`crate::draft::check_draft_pack`]): its fingerprint and
/// geometry, or every mismatch by name.
pub fn check_dir(dir: &Path, cfg: &FlashNextConfig) -> Result<DraftCheck, ModelError> {
    let expect = DraftExpect { hidden: cfg.hidden, vocab: cfg.vocab, tap_layers: TAP_LAYERS, target_layers: cfg.layers };
    crate::draft::check_draft_pack(dir, &DraftGeom::FLASH_NEXT, &expect)
        .map_err(|e| ModelError::Config(format!("{}: {e}", dir.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_taps_are_the_geometrys_layers_in_fc_order() {
        assert_eq!(TAP_LAYERS, [5, 15, 25, 35, 45]);
        assert_eq!((0..48).filter_map(tap_of).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
        assert_eq!((tap_of(5), tap_of(45), tap_of(47)), (Some(0), Some(4), None));
    }

    /// Our 4.00 bpw quantization (`REXL3_FN_DRAFT_DIR`, CPU, headers only) for the shipped target's config: accepted on its structure
    /// with no flag, its fingerprint the one `fingerprint::VALIDATED_FLASH_NEXT_DRAFT` lists, and the boot gate's verdict "listed".
    #[test]
    fn the_local_4bpw_draft_passes_the_structural_check_and_is_the_listed_pair() {
        let Ok(dir) = std::env::var("REXL3_FN_DRAFT_DIR") else {
            eprintln!("not run: REXL3_FN_DRAFT_DIR is unset");
            return;
        };
        let cfg: FlashNextConfig = FlashNextConfig::from_json(
            &serde_json::from_str(crate::families::qwen_flash_next::config::tests::FLASH_NEXT_CONFIG).unwrap(),
        )
        .unwrap();
        let c = check_dir(std::path::Path::new(&dir), &cfg).unwrap();
        assert_eq!(c.fingerprint, crate::fingerprint::FLASH_NEXT_DFLASH2_Q4_FP);
        assert_eq!(c.geometry, "qwen38-flash-next");
        assert_eq!(c.pair(crate::fingerprint::QFN305_FP), Some(crate::fingerprint::VALIDATED_FLASH_NEXT_DRAFT[0].0));
        // the same verdict the boot's pack gate gives (target listed alone, draft listed with it)
        let v = crate::serve::pack_verdict(crate::fingerprint::QFN305_FP, Some((&c.fingerprint, &Ok(&DraftGeom::FLASH_NEXT))))
            .unwrap();
        assert_eq!(
            v.draft.as_deref(),
            Some(format!("listed ({})", crate::fingerprint::VALIDATED_FLASH_NEXT_DRAFT[0].0).as_str())
        );
    }
}
