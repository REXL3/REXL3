//! Which drafters a serving run may use: the value of `--spec`.

use std::fmt;
use std::str::FromStr;

/// `--spec {auto|mtp|copy|none}`. The drafters only ever PROPOSE tokens: the verify window decides every committed token, so the mode changes
/// speed, never an output byte (`drafted == serial`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpecMode {
    /// Every drafter the family offers, chosen per round by the arbiter (committed tokens per model-millisecond, with probing).
    #[default]
    Auto,
    /// The family's own multi-token-prediction head, at a fixed depth.
    Mtp,
    /// The copy drafter alone (continue the longest earlier occurrence of the context's suffix): no model compute.
    Copy,
    /// No speculation: every round is one serial row per sequence.
    None,
}

impl SpecMode {
    /// May the multi-token-prediction head draft (so it must be loaded and its cache allocated)?
    pub fn uses_mtp(self) -> bool {
        matches!(self, SpecMode::Auto | SpecMode::Mtp)
    }
    /// May the copy drafter draft?
    pub fn uses_copy(self) -> bool {
        matches!(self, SpecMode::Auto | SpecMode::Copy)
    }
    /// Does any drafter run?
    pub fn drafts(self) -> bool {
        self != SpecMode::None
    }
}

impl FromStr for SpecMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(SpecMode::Auto),
            "mtp" => Ok(SpecMode::Mtp),
            "copy" => Ok(SpecMode::Copy),
            "none" => Ok(SpecMode::None),
            other => Err(format!("--spec takes auto, mtp, copy or none, not `{other}`")),
        }
    }
}

impl fmt::Display for SpecMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SpecMode::Auto => "auto",
            SpecMode::Mtp => "mtp",
            SpecMode::Copy => "copy",
            SpecMode::None => "none",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_four_values_parse_and_print_back() {
        for (s, m) in [("auto", SpecMode::Auto), ("mtp", SpecMode::Mtp), ("copy", SpecMode::Copy), ("none", SpecMode::None)] {
            assert_eq!(s.parse::<SpecMode>(), Ok(m));
            assert_eq!(m.to_string(), s);
        }
        let e = "dflash2".parse::<SpecMode>().unwrap_err();
        assert!(e.contains("auto, mtp, copy or none") && e.contains("dflash2"), "{e}");
        assert_eq!(SpecMode::default(), SpecMode::Auto, "auto is the default");
    }

    #[test]
    fn what_each_mode_may_use() {
        assert!(SpecMode::Auto.uses_mtp() && SpecMode::Auto.uses_copy() && SpecMode::Auto.drafts());
        assert!(SpecMode::Mtp.uses_mtp() && !SpecMode::Mtp.uses_copy());
        assert!(!SpecMode::Copy.uses_mtp() && SpecMode::Copy.uses_copy());
        assert!(!SpecMode::None.uses_mtp() && !SpecMode::None.uses_copy() && !SpecMode::None.drafts());
    }
}
