//! Ling-3.1-flash-1019's prefill arithmetic contract (Lane LP2 of the Ling-3.1 `>1,000 tok/s` prefill plan).
//! Modeled on GLM's (`families/glm/contract.rs`), with one scope difference: **P2 selects the tensor-core
//! Gated-MLA kernels (`gb_cuda::ling_mla_attn`) for prefill, verify, decode and the L72 MTP layer alike** —
//! the arithmetic is selected by contract, not by call type, because the absorbed tensor-core attention is
//! also the fast decode path (the scalar `attend_kernel` is quadratic in every regime).
//!
//! * [`PrefillContract::P1`] (the default until P2 passes qualification): every row of the Gated-MLA runs
//!   `gb_cuda::mla_gated`'s scalar kernels (one fixed-order fp32 FMA chain per absorb/expand output, the
//!   per-`(row, head)` online-softmax `attend_kernel`). This is the arithmetic Ling-3.1 shipped with; the
//!   prefix namespace and fingerprint are unchanged.
//! * [`PrefillContract::P2`] (`REXL3_LING_MLA_CONTRACT=p2` selects it explicitly): the Gated-MLA absorb,
//!   the dense causal absorbed attention and the expand (fused with the sigmoid output gate) run on
//!   BF16 `mma.sync.m16n8k16` tensor cores with FP32 accumulation (`gb_cuda::ling_mla_attn`). The
//!   contraction orders differ from P1 by design (fold128 K order for absorb/expand, 32-key tiles with the
//!   latent d-quarter sum order for the attention), so P2 is **not** bit-equal to P1: a row prefilled under
//!   P1 and the same row decoded under P2 differ within the f64-oracle bound. What holds under P2 (the
//!   contract's (a)-(d)): row and chunk independence (a row's bits are a function of that row's operands,
//!   its slot's keys and its position alone), fixed-order K reductions, bf16 operands / fp32 accumulation,
//!   determinism run to run. The attention of a prefill chunk is one online-softmax walk over the row's keys; the attention of a
//!   decode / verify / draft call (`prefill == false`) cuts the keys into absolute 1,024-key chunks on separate CTAs and merges them
//!   in ascending order (`gb_cuda::ling_mla_attn::attend_split`; [`P2_MLA_ATTN`] names both). Within each of the two a row's bits
//!   are independent of the call's rows; between them a row differs by the bf16 rounding of the re-association, which no path needs
//!   to be equal (a prefix snapshot is cut inside a prefill, and a drafted window is a decode call like the serial step).
//!
//! What stays P1 under both contracts, by design: the MoE experts, the norms, the lm_head, and the MLA
//! `kv_prep` / `q_rot` (RMSNorm and the rope rotation are not contractions, so both contracts share the scalar kernels
//! and their bits). The fp32 MoE router is a fixed-order reduction per (row, column) under both, never a library GEMM
//! (whose algorithm, so whose low bits, depends on the call's row count): P1 runs the scalar kernel at every row count; P2 runs
//! the scalar kernel for decode / verify windows and the tensor-core `gb_cuda::mla_gated::router_rows_f32` for prompt rows (an
//! ascending-k `mma.sync` chain with no split-K, the same instruction sequence for every row at any row count), which is why the
//! P2 namespace names `router=p2(..)`.
//!
//! Under **P1**, the dense EXL3 linears and the KDA recurrence also run the scalar decode path (`Lin::forward`,
//! `kda::prep` with fp32 qkv, `kda::commit`): the row-invariant kernels, not the LP3 direct-MMA prefill GEMM.
//! Under **P2**, the dense projections use the LP3 direct-MMA tensor-core GEMM (`forward_prefill_auto`) and the KDA
//! qkv is bf16 (`prep_prefill_bf16` + `commit_from`).
//!
//! The contract is one process-wide setting read once from [`CONTRACT_VAR`]; a model copies it at
//! construction. The default is **P1** until the LP2 qualification gates (f64-oracle bounds, row/chunk/
//! determinism tests, goldens, P1-vs-P2 KL and Q200) pass; flipping it is this one constant plus the
//! fingerprint tests. P2 has its own prefix-cache namespace ([`PrefillContract::arith_tag`]) so a snapshot
//! written under one contract is never restored under the other.

use std::sync::OnceLock;

/// The environment variable that selects Ling-3.1's Gated-MLA arithmetic contract: unset or empty for the
/// default ([`DEFAULT_CONTRACT`], P2), `p1`, or `p2`; anything else is refused at load by name.
/// [`PREFILL_CONTRACT_VAR`] is accepted as an alias (the Q200 lane's arm scripts set it).
pub const CONTRACT_VAR: &str = "REXL3_LING_MLA_CONTRACT";

/// Alias for [`CONTRACT_VAR`]: the Q200 lane (#210) sets `REXL3_LING_PREFILL_CONTRACT=p1` for its P1 arm.
/// When both are set they must agree; a conflict is refused at load.
pub const PREFILL_CONTRACT_VAR: &str = "REXL3_LING_PREFILL_CONTRACT";

/// The contract of a process that does not set [`CONTRACT_VAR`]. **P2** (the tensor-core kernels) for this
/// combined performance branch.
pub const DEFAULT_CONTRACT: PrefillContract = PrefillContract::P2;

/// Whether this build links the LP2 tensor-core kernels (`gb_cuda::ling_mla_attn`). Always true: the crate
/// compiles the `.cu` into the same static library as the scalar path (a stub build has neither, and the
/// launchers fail at run time under both contracts).
pub const P2_BUILT: bool = true;

/// The version of P2's absorb / expand arithmetic (fold128 K order, the gate reduction tree of
/// `gb_cuda::ling_mla_attn`): part of the prefix namespace and the fingerprint.
pub const P2_MLA_AE: &str = gb_cuda::ling_mla_attn::ARITH_AE;

/// The version of P2's absorbed-attention arithmetic (32-key tiles, d-quarter score order, log2-domain
/// online softmax of `gb_cuda::ling_mla_attn`): part of the prefix namespace and the fingerprint.
pub const P2_MLA_ATTN: &str = gb_cuda::ling_mla_attn::ARITH_ATTN;

/// The arithmetic id of P2's prompt-row MoE router (`gb_cuda::mla_gated::router_rows_f32`: fixed ascending-k `mma.sync` chain, fp32
/// accumulate, no split-K): part of the prefix namespace and the fingerprint.
pub const P2_ROUTER: &str = gb_cuda::mla_gated::ROUTER_ARITH;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrefillContract {
    P1,
    P2,
}

impl PrefillContract {
    /// The short id that goes into logs and the fingerprint.
    pub const fn id(self) -> &'static str {
        match self {
            PrefillContract::P1 => "p1",
            PrefillContract::P2 => "p2",
        }
    }

    /// The tag appended to the prefix cache's arithmetic field: empty for P1 (its namespace is the text it
    /// always was); P2's names the tensor-core kernels' arithmetic ids and what stays P1.
    pub fn arith_tag(self) -> String {
        match self {
            PrefillContract::P1 => String::new(),
            PrefillContract::P2 => {
                format!("+mla=p2(ae={P2_MLA_AE},attn={P2_MLA_ATTN},kda=p2,moe=p1,dense=p2,router=p2({P2_ROUTER}),lm_head=p1)")
            }
        }
    }

    /// Whether a Gated-MLA forward of this model runs the tensor-core kernels. Unlike GLM, this is **not**
    /// gated on `is_prefill`: the absorbed tensor-core attention is the fast path for prefill, verify,
    /// decode and L72 alike, so the arithmetic is a property of the model, not of the call type.
    pub const fn tc_mla(self) -> bool {
        matches!(self, PrefillContract::P2)
    }
}

/// The contract a value of [`CONTRACT_VAR`] asks for: unset or empty is [`DEFAULT_CONTRACT`].
pub fn parse(v: Option<&str>) -> Result<PrefillContract, String> {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") => Ok(DEFAULT_CONTRACT),
        Some("p1") => Ok(PrefillContract::P1),
        Some("p2") => Ok(PrefillContract::P2),
        Some(other) => Err(format!("{CONTRACT_VAR}={other:?} is not p1 or p2 (unset is the default, {})", DEFAULT_CONTRACT.id())),
    }
}

/// What [`CONTRACT_VAR`] (or its alias [`PREFILL_CONTRACT_VAR`]) selects in this process (read once, logged once):
/// the contract, or why it is refused. When both are set they must agree.
fn selection() -> &'static Result<PrefillContract, String> {
    static C: OnceLock<Result<PrefillContract, String>> = OnceLock::new();
    C.get_or_init(|| {
        let a = std::env::var(CONTRACT_VAR).ok();
        let b = std::env::var(PREFILL_CONTRACT_VAR).ok();
        let r = match (a.as_deref(), b.as_deref()) {
            (None, None) => parse(None),
            (Some(v), None) => parse(Some(v)),
            (None, Some(v)) => parse(Some(v)),
            (Some(va), Some(vb)) => match (parse(Some(va)), parse(Some(vb))) {
                (Ok(ca), Ok(cb)) if ca == cb => Ok(ca),
                (Ok(ca), Ok(cb)) => Err(format!(
                    "{CONTRACT_VAR}={va:?} (={}) conflicts with {PREFILL_CONTRACT_VAR}={vb:?} (={})",
                    ca.id(),
                    cb.id()
                )),
                (Err(e), _) | (_, Err(e)) => Err(e),
            },
        };
        match &r {
            Ok(c) => eprintln!(
                "rexl3 level=info event=ling_mla_contract contract={} default={} var={CONTRACT_VAR}{}",
                c.id(),
                (*c == DEFAULT_CONTRACT) as u8,
                if *c == PrefillContract::P2 {
                    format!(" mla-ae={P2_MLA_AE} mla-attn={P2_MLA_ATTN} router={P2_ROUTER}")
                } else {
                    String::new()
                }
            ),
            Err(e) => eprintln!("rexl3 level=error event=ling_mla_contract_refused reason=\"{e}\""),
        }
        r
    })
}

/// The contract of this process. A refused value ([`check_selected`] stops the load first) reads as the
/// default here, so nothing that asks before the load (a log line, the fingerprint) panics.
pub fn current() -> PrefillContract {
    selection().clone().unwrap_or(DEFAULT_CONTRACT)
}

/// `Ok(contract)` when [`CONTRACT_VAR`] names one (or is unset), else the refusal by name.
pub fn check_selected() -> Result<PrefillContract, String> {
    selection().clone()
}

/// `Ok` when this build can run `c`.
pub fn check_built(c: PrefillContract) -> Result<(), String> {
    match c {
        PrefillContract::P1 => Ok(()),
        PrefillContract::P2 if P2_BUILT => Ok(()),
        PrefillContract::P2 => {
            Err(format!("{CONTRACT_VAR}=p2 asks for Ling-3.1's tensor-core MLA, which this build does not link"))
        }
    }
}

/// The fingerprint field of this process's Ling-3.1 MLA contract: `contract=p1`, or under P2 also the
/// tensor-core kernels' arithmetic ids. Part of the prefix namespace and of a TP Hello fact (two ranks on
/// different contracts would leave different bits in the replicated layers).
pub fn mla_fingerprint() -> String {
    mla_fingerprint_for(current())
}

/// [`mla_fingerprint`] for a given contract (pure, for the tests).
pub fn mla_fingerprint_for(c: PrefillContract) -> String {
    match c {
        PrefillContract::P1 => "contract=p1".into(),
        PrefillContract::P2 => {
            format!("contract=p2,mla-ae={P2_MLA_AE},mla-attn={P2_MLA_ATTN},router={P2_ROUTER}")
        }
    }
}

/// The field the server's `system_fingerprint` carries when this process's Ling-3.1 MLA contract is not the
/// default: `lingpfp1` under `REXL3_LING_MLA_CONTRACT=p1` or `REXL3_LING_PREFILL_CONTRACT=p1` (the scalar
/// kernels), `None` for the default P2.
pub fn fingerprint_suffix() -> Option<String> {
    suffix_for(current())
}

/// [`fingerprint_suffix`] for a given contract (pure, for the tests).
pub fn suffix_for(c: PrefillContract) -> Option<String> {
    (c != DEFAULT_CONTRACT).then(|| format!("lingpf{}", c.id()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_contract_switch_takes_p1_or_p2_and_p2_is_the_default() {
        assert_eq!(DEFAULT_CONTRACT, PrefillContract::P2);
        assert_eq!(parse(None), Ok(PrefillContract::P2));
        assert_eq!(parse(Some("")), Ok(PrefillContract::P2));
        assert_eq!(parse(Some("  ")), Ok(PrefillContract::P2));
        assert_eq!(parse(Some("p1")), Ok(PrefillContract::P1));
        assert_eq!(parse(Some(" P1 ")), Ok(PrefillContract::P1));
        assert_eq!(parse(Some("p2")), Ok(PrefillContract::P2), "the explicit tensor-core selection");
        assert_eq!(parse(Some(" P2")), Ok(PrefillContract::P2));
        for bad in ["p3", "mma", "1", "bf16", "p1,p2", "tc"] {
            let e = parse(Some(bad)).unwrap_err();
            assert!(e.contains(CONTRACT_VAR) && e.contains(bad), "{bad}: {e}");
        }
        assert!(check_built(PrefillContract::P1).is_ok());
        assert_eq!(check_built(PrefillContract::P2).is_ok(), P2_BUILT);
        if std::env::var(CONTRACT_VAR).is_err() {
            assert_eq!(current(), PrefillContract::P2);
            assert_eq!(check_selected(), Ok(PrefillContract::P2));
            assert_eq!(fingerprint_suffix(), None);
        }
    }

    #[test]
    fn p1_keeps_its_namespace_and_p2_names_what_it_changes() {
        assert_eq!(PrefillContract::P1.arith_tag(), "");
        let tag = PrefillContract::P2.arith_tag();
        assert!(tag.contains("mla=p2"), "{tag}");
        assert!(tag.contains(P2_MLA_AE), "{tag}");
        assert!(tag.contains(P2_MLA_ATTN), "{tag}");
        // the prompt-row router's arithmetic is named (a snapshot written with the cuBLAS router, whose bits depended on the call's row
        // count, can never be restored under the row-invariant one), and so is what stays P1: a later change of one of those is a
        // namespace change too
        assert!(tag.contains(&format!("router=p2({P2_ROUTER})")), "{tag}");
        assert!(!tag.contains("router=p1"), "{tag}");
        for stay in ["kda=p2", "moe=p1", "dense=p2", "lm_head=p1"] {
            assert!(tag.contains(stay), "{tag} missing {stay}");
        }
        assert_eq!(mla_fingerprint_for(PrefillContract::P1), "contract=p1");
        let fp = mla_fingerprint_for(PrefillContract::P2);
        assert!(
            fp.contains("contract=p2") && fp.contains(P2_MLA_AE) && fp.contains(P2_MLA_ATTN) && fp.contains(P2_ROUTER),
            "{fp}"
        );
        assert_eq!(suffix_for(PrefillContract::P1).as_deref(), Some("lingpfp1"));
        assert_eq!(suffix_for(PrefillContract::P2), None);
    }

    #[test]
    fn p2_selects_tensor_cores_for_every_call_type() {
        // The whole point of the Ling contract: the absorbed tensor-core attention is the fast decode path
        // too, so the selection must not depend on is_prefill.
        assert!(!PrefillContract::P1.tc_mla());
        assert!(PrefillContract::P2.tc_mla());
    }
}
