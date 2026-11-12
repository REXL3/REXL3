//! Qwen3.8-27B prompt rows on the fused tensor-core dense GEMM (`docs/QWEN_PREFILL_FUSED.md`).
//!
//! **What changes.** A prompt of more than [`crate::forward::PrefillPath::GEMV_MAX_PROMPT`] tokens ran every EXL3 linear of every chunk as
//! reconstruct `W_q` to fp16 (`unpack_kernel`), rotate the input, one pinned cuBLAS fp16 GEMM, rotate the output (`rot_in_kernel`,
//! `rot_out_kernel`): about a quarter of a 32K prefill's GPU time was the two rotations and the unpack. [`PrefillGemm::Fused`] runs the
//! linear as `gb_cuda::glm_p2::linear_variant` instead: the input rotation as one light kernel, then one tensor-core GEMM that decodes the
//! trellis tiles straight into the MMA's B fragments (no `W_q` in memory) and applies the output rotation, the scale and the bf16 rounding
//! in its epilogue.
//!
//! **The K order is the single chain** (`Variant::Chain`): one chain of m16n8k16 MMAs over all of K from +0.0, fp32 accumulate, which is
//! what the pinned cuBLAS algorithm does too. MEASURED (`gb-cuda` test
//! `the_chain_order_equals_the_pinned_cublas_path_bit_for_bit_on_the_qwen_shapes`, all seven shapes of the pack at 1 to 4,096 rows;
//! `golden_layers.txt` and the greedy replies on the real model): the fused GEMM's bf16 output is the pinned path's **bit for bit**. So
//! this switch changes speed and nothing else: a prompt row has the bits it always had (chunked == whole, cached == uncached, the prefix
//! cache's snapshots and every committed golden hold unchanged), which is why the prefix namespace and the `system_fingerprint` carry no
//! tag for it. The per-128 fold (`Variant::Fold128`, GLM's default; 7 % slower here and with other bits) would remove the chain's small
//! signed shrink on the 17,408-wide `down_proj` (-1.6e-5 of the output rms, the pinned path's own); it is not the default because it
//! would move every prompt's bits.
//!
//! Prompts of up to 128 tokens stay on the decode GEMV, windows and drafts never come here.
//!
//! **The switch** is [`GEMM_VAR`] (`REXL3_QWEN_PREFILL_GEMM`): unset or empty = `fused` (the default), `pinned` = the previous reconstruct +
//! cuBLAS path (also what the W_q residency, `REXL3_WQ_RESIDENT_GIB`, needs), anything else is refused at load by name.

use std::sync::OnceLock;

/// The environment variable that selects the prompt rows' linear GEMM.
pub const GEMM_VAR: &str = "REXL3_QWEN_PREFILL_GEMM";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrefillGemm {
    /// Reconstruct `W_q` + the pinned cuBLAS GEMM + the output rotation (the path before this switch existed).
    Pinned,
    /// The fused tensor-core GEMM (`gb_cuda::glm_p2`, single-chain K order): the same bits, faster.
    Fused,
}

/// What a process that does not set [`GEMM_VAR`] runs.
pub const DEFAULT: PrefillGemm = PrefillGemm::Fused;

impl PrefillGemm {
    pub const fn id(self) -> &'static str {
        match self {
            PrefillGemm::Pinned => "pinned",
            PrefillGemm::Fused => "fused",
        }
    }
}

/// The path a value of [`GEMM_VAR`] asks for; unset, empty or blank is [`DEFAULT`].
pub fn parse(v: Option<&str>) -> Result<PrefillGemm, String> {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") => Ok(DEFAULT),
        Some("fused") => Ok(PrefillGemm::Fused),
        Some("pinned") => Ok(PrefillGemm::Pinned),
        Some(other) => Err(format!("{GEMM_VAR}={other:?} is not fused or pinned (unset is the default, {})", DEFAULT.id())),
    }
}

/// What [`GEMM_VAR`] selects in this process (read once), or why the value is refused. `serve` and `Model::new` call this before any weight
/// is read, so a bad value stops the boot by name.
pub fn current() -> Result<PrefillGemm, String> {
    static S: OnceLock<Result<PrefillGemm, String>> = OnceLock::new();
    S.get_or_init(|| parse(std::env::var(GEMM_VAR).ok().as_deref())).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_takes_fused_or_pinned_and_fused_is_the_default() {
        assert_eq!(DEFAULT, PrefillGemm::Fused);
        assert_eq!(parse(None), Ok(PrefillGemm::Fused));
        assert_eq!(parse(Some("")), Ok(PrefillGemm::Fused));
        assert_eq!(parse(Some("  ")), Ok(PrefillGemm::Fused));
        assert_eq!(parse(Some("fused")), Ok(PrefillGemm::Fused));
        assert_eq!(parse(Some(" Pinned ")), Ok(PrefillGemm::Pinned), "the opt-out");
        for bad in ["0", "off", "p1", "p2", "true", "fused,pinned", "cublas"] {
            let e = parse(Some(bad)).unwrap_err();
            assert!(e.contains(GEMM_VAR) && e.contains(bad), "{e}");
        }
    }
}
