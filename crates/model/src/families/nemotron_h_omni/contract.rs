//! The prefill arithmetic contract (lane S, `docs/NEMOTRON_PREFILL_SPEED.md`).
//!
//! * [`PrefillContract::P2`] (THE DEFAULT since rc2d, owner 2026-10-15: "we will continue with p2 as the default for now"; kernels of
//!   lanes S2 `nemotron/moe-mma` and S3 `nemotron/dense-mma`): prefill rows only; decode and verify windows keep P1. (a) a row's output
//!   does not depend on the other rows, the slice
//!   width, a slice boundary, padding or row order; (b) every K reduction is in an order that is a function of K alone, so column cuts (TP)
//!   are exact and single node == TP=2; (c) activations enter `mma.sync` as fp16 or bf16, the EXL3 weights decoded exactly as the decode
//!   path decodes them, fp32 accumulation; (d) deterministic run to run. The router's logits and selection stay P1's bits. A prompt row
//!   has other bits under P2 than under P1 (prefill bits differ from decode bits), so P2 is another arithmetic for the prefix cache
//!   (another namespace) and has its own goldens (replies sha `65b59756...`, `testdata/gpu/nemotron_step_digest_p2.txt`). What holds
//!   under P2: row and slice independence, chunked == whole, cached == uncached, single node == TP, determinism, drafted == serial.
//! * [`PrefillContract::P1`] (the default until rc2c; selectable with `REXL3_NEMO_PREFILL_CONTRACT=p1`): every prefill row is computed
//!   operation for operation as the 64-row decode path computes it (fp32 FMA chains of the MoE in the decode kernels' k order, the EXL3
//!   linears' row-invariant kernels), so a prefill row, a verify-window row and a decoded row have the same bits, and P1's goldens
//!   (replies sha `27fc7fe0...`, the 121-line `nemotron_step_digest_repaired.txt`) hold.
//!
//! The contract is one process-wide setting, read once from [`CONTRACT_VAR`]; under TP it is one of the boot gate's facts (both ranks must
//! run the same). A value that is not `p1` or `p2` is refused at load by name ([`check_selected`]), never silently run as another
//! contract; asking for P2 in a build without its kernels is refused the same way ([`check_built`]).

use std::sync::OnceLock;

/// The environment variable that selects the prefill contract: unset or empty for the default ([`DEFAULT_CONTRACT`], P2), `p1` for P1,
/// `p2` for P2; anything else is refused at load by name.
pub const CONTRACT_VAR: &str = "REXL3_NEMO_PREFILL_CONTRACT";

/// The contract of a process that does not set [`CONTRACT_VAR`] (owner decision 2026-10-15, built in rc2d; P1 before). Flipping it back
/// is this one constant: the fingerprint suffix, the defaults of the gates and the tests follow it.
pub const DEFAULT_CONTRACT: PrefillContract = PrefillContract::P2;

/// The widest prefill slice under P2 (the scratch, the engine clamp and the memory plan are priced for it when P2 is selected): the
/// tensor-core MoE reads each expert's weights once a slice, 1.455 GB a layer, so a slice of 256 rows would be bound at about 1,050 tok/s
/// by that read alone (40 layers x 6.06 ms / 256 rows); 1,024 rows lifts the bound to about 4,200 and 4,096 to about 17,000.
pub const P2_MAX_PREFILL_ROWS: usize = 4096;

/// Whether this build links the P2 kernels: S3's tensor-core prefill linears (`gb_cuda::nemotron_dense_mma`: Mamba in/out_proj,
/// attention q/k/v/o, the MoE's fc1/fc2/shared expert), its 4-threads-a-column scan (bit-identical) and its flash routing of prefill
/// rows below 8,192. The routed experts stay on P1's `pf_gemm4` until S2's grouped MMA block is merged.
pub const P2_BUILT: bool = true;

/// The widest input of a Nemotron EXL3 linear (`out_proj`, K = 8,192): what the P2 linear's fp16 workspace is sized by.
pub const P2_MAX_K: usize = 8192;

/// Puts the P2 kernels' process switch (`gb_cuda::nemotron_dense_mma::set_enabled`) in line with the contract: on under P2, off under
/// P1, whatever `REXL3_NEMO_DENSE_MMA` says (the contract variable is the one switch; the boot gate and the fingerprint carry it).
pub fn apply() {
    let p2 = current() == PrefillContract::P2;
    gb_cuda::nemotron_dense_mma::set_enabled(p2);
    // the tensor-core MoE block (lane S2) is chosen per call by the model (`arm_moe`); its process switch follows the contract too, and
    // its router stays on P1's fp32 kernel (tensor-core router logits flip near-tie picks: off until the end-to-end KL clears it)
    gb_cuda::nemotron_moe_prefill::mma::set_enabled(p2);
    gb_cuda::nemotron_moe_prefill::mma::set_router_mma(p2 && router_mma());
}

/// `REXL3_NEMO_PREFILL_ROUTER_MMA=1`: under P2, the router logits on tensor cores too (S2's `set_router_mma`; near-tie picks move).
/// Off by default until the end-to-end KL and the pick-flip counts clear it; part of the prefill fingerprint and the P2 namespace.
pub const ROUTER_MMA_VAR: &str = "REXL3_NEMO_PREFILL_ROUTER_MMA";

/// Whether [`ROUTER_MMA_VAR`] asks for the tensor-core router (read once).
pub fn router_mma() -> bool {
    static R: OnceLock<bool> = OnceLock::new();
    *R.get_or_init(|| std::env::var(ROUTER_MMA_VAR).is_ok_and(|v| v.trim() == "1"))
}

/// The scope a forward pass of a prefill slice holds under P2 (`None` under P1 or for a window): inside it the P2 kernels run
/// (`gb_cuda::nemotron_dense_mma::active`).
pub fn prefill_scope(is_prefill: bool) -> Option<gb_cuda::nemotron_dense_mma::PrefillScope> {
    (is_prefill && current() == PrefillContract::P2).then(gb_cuda::nemotron_dense_mma::PrefillScope::enter)
}

/// Bytes of the EXL3 workspace for a model whose widest step is `rows` rows: P1's (`p1_bytes`), or under P2 also the P2 linear's
/// fp16 input of the whole slice at the widest K.
pub fn workspace_bytes(p1_bytes: usize, rows: usize) -> usize {
    match current() {
        PrefillContract::P1 => p1_bytes,
        PrefillContract::P2 => p1_bytes.max(gb_cuda::nemotron_dense_mma::workspace_bytes(rows, P2_MAX_K)),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrefillContract {
    P1,
    P2,
}

impl PrefillContract {
    /// The short id that goes into logs, the TP boot gate and the system fingerprint.
    pub const fn id(self) -> &'static str {
        match self {
            PrefillContract::P1 => "p1",
            PrefillContract::P2 => "p2",
        }
    }

    /// The tag appended to the prefix cache's arithmetic tag (`prefix::PREFILL_ARITH`): empty for P1, so P1's namespace is the one it
    /// always was; P2's names its version (bump it with any rounding point, tile order or K order of a P2 kernel).
    pub const fn arith_tag(self) -> &'static str {
        match self {
            PrefillContract::P1 => "",
            PrefillContract::P2 => "+prefill=p2(dense-mma-s3-0bb1b152,moe-mma-s2-90596ff8-fp16-tile4,router=p1,flash<8192,v1)",
        }
    }

    /// The widest prefill slice this contract runs: P1's cap (`model::prefill_rows_cap`'s 256, or 512 under its switch) or P2's.
    pub fn max_prefill_rows(self, p1_cap: usize) -> usize {
        match self {
            PrefillContract::P1 => p1_cap,
            PrefillContract::P2 => P2_MAX_PREFILL_ROWS,
        }
    }

    /// Whether a step of this kind runs the contract's prefill arithmetic. Only prefill slices do: a verify or decode window, wide or not,
    /// is P1 under every contract (the large-row MoE path a WIDE decode window takes stays P1's bits).
    pub const fn applies_to_prefill_only(self) -> bool {
        true
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

/// What [`CONTRACT_VAR`] selects in this process (read once): the contract, or why the value is refused.
fn selection() -> &'static Result<PrefillContract, String> {
    static C: OnceLock<Result<PrefillContract, String>> = OnceLock::new();
    C.get_or_init(|| {
        let r = parse(std::env::var(CONTRACT_VAR).ok().as_deref());
        match &r {
            Ok(c) => eprintln!(
                "rexl3 level=info event=prefill_contract contract={} default={} var={CONTRACT_VAR}",
                c.id(),
                (*c == DEFAULT_CONTRACT) as u8
            ),
            Err(e) => eprintln!("rexl3 level=error event=prefill_contract_refused reason=\"{e}\""),
        }
        r
    })
}

/// The contract of this process. A refused value ([`check_selected`] stops the load first) reads as the default here, so nothing that
/// asks before the load (a log line, the fingerprint) panics.
pub fn current() -> PrefillContract {
    selection().clone().unwrap_or(DEFAULT_CONTRACT)
}

/// `Ok(contract)` when [`CONTRACT_VAR`] names one (or is unset), else the refusal by name; the model's option check calls it before any
/// weight is read, so a misspelt value never runs as some other contract.
pub fn check_selected() -> Result<PrefillContract, String> {
    selection().clone()
}

/// `Ok` when this build can run `c`; a P2 request before the P2 kernels are linked is refused by name.
pub fn check_built(c: PrefillContract) -> Result<(), String> {
    match c {
        PrefillContract::P1 => Ok(()),
        PrefillContract::P2 if P2_BUILT => Ok(()),
        PrefillContract::P2 => Err(format!(
            "{CONTRACT_VAR}=p2 asks for the tensor-core prefill contract, which this build does not link (lanes S2/S3)"
        )),
    }
}

/// `Ok` when contract `c` runs as named in a model built for prefill slices of `prefill_rows` rows and a decode window of `window_rows`
/// (v0.1.1, `docs/NEMOTRON_PREFILL_SPEED.md` 6.17). The model's prefill rows cap is `model::step_rows_cap` of the two; at
/// [`super::model::WINDOW_ROWS`] (64) or less the model builds neither the large-row MoE scratch nor the tensor-core MoE scratch, so under
/// P2 its MoE layers would run P1's fp32 kernels inside a P2 prefill: a hybrid no fingerprint names. Under P2 that is refused by name
/// (`--prefill-chunk` 64 or less with the 64-row window); P1 takes any rows (its prefill is the decode path's arithmetic at every width).
pub fn check_rows_cap(c: PrefillContract, prefill_rows: usize, window_rows: usize) -> Result<(), String> {
    let w = super::model::WINDOW_ROWS;
    let cap = super::model::step_rows_cap(prefill_rows, window_rows);
    if c != PrefillContract::P2 || cap > w {
        return Ok(());
    }
    Err(format!(
        "--prefill-chunk {prefill_rows}: under the P2 prefill contract (the default) the prefill rows cap must be more than {w} rows, and \
         this boot's is {cap} (the larger of the {prefill_rows}-row prefill slice and the {window_rows}-row decode window); at {w} or less \
         the model builds no tensor-core MoE scratch and its MoE layers would run P1's fp32 kernels inside a P2 prefill. Use \
         --prefill-chunk {} or more (the default is 4096), or {CONTRACT_VAR}=p1, which takes any chunk",
        w + 1
    ))
}

/// The prefill rows this process's boot runs (`engine::prefill_rows`: `--prefill-chunk`, at most the contract's cap), registered by the
/// server before it logs or serves a fingerprint ([`set_boot_prefill_rows`]).
static BOOT_PREFILL_ROWS: OnceLock<usize> = OnceLock::new();

/// Registers the prefill rows of this process's boot for [`fingerprint_suffix`] (first call wins). The server calls it from its parsed
/// options before the start line, the way `--kv-dtype` is set before anything is built.
pub fn set_boot_prefill_rows(rows: usize) {
    let _ = BOOT_PREFILL_ROWS.set(rows);
}

/// The prefill rows [`fingerprint_suffix`] names: the registered boot's ([`set_boot_prefill_rows`]), or, with none registered
/// (`--version`, tools, tests), a boot at the shipped `--prefill-chunk` of 4,096, which is at or above every cap: the contract's cap.
pub fn boot_prefill_rows() -> usize {
    BOOT_PREFILL_ROWS.get().copied().unwrap_or_else(|| current().max_prefill_rows(super::model::prefill_rows_cap()))
}

/// The prefill rows of a boot at the defaults (the shipped `--prefill-chunk` of 4,096 under [`DEFAULT_CONTRACT`] and the default slice
/// cap): the rows at which [`suffix_for`] adds no field. 4,096 under P2.
pub fn default_prefill_rows() -> usize {
    DEFAULT_CONTRACT.max_prefill_rows(super::model::WIDE_PREFILL_ROWS)
}

/// The prefill settings a rank runs with, for its boot's `prefill_rows` (`ModelOptions::prefill_rows`), as one string for the TP boot gate
/// (`NemoFacts::prefill`) and the logs: `contract=p1,rows=512,absent=1`, and under P2 also the router variant and the TP levers. Two
/// ranks that differ in any of them are refused by name at the Hello.
pub fn prefill_fingerprint(prefill_rows: usize) -> String {
    prefill_fingerprint_for(
        current(),
        prefill_rows,
        gb_cuda::nemotron_moe_prefill::absent_fast(),
        router_mma(),
        &p2tp_lever_label(),
    )
}

/// [`prefill_fingerprint`] for given settings (pure, for the tests).
pub fn prefill_fingerprint_for(c: PrefillContract, rows_cap: usize, absent_fast: bool, router_mma: bool, levers: &str) -> String {
    let base = settings_label(c, rows_cap, absent_fast);
    if c != PrefillContract::P2 {
        return base;
    }
    let base = if router_mma { format!("{base},router=mma") } else { base };
    // the TP exchange levers (lane S): two ranks that disagree on one would issue different exchanges (a hang, not an error), so they are
    // facts of the boot gate
    format!("{base},tp={levers}")
}

/// The router logits split by rows at two ranks (lane S, `docs/NEMOTRON_PREFILL_SPEED.md` 6.12): OFF by default since v0.1.1 (rule
/// R-ROUTER: interleaved over 3 rounds, TP=2 with the split off was faster on all four D1 prompts, median 1.9 to 2.2 % of the TTFT; on by
/// default in rc2d and v0.1.0). Bit-neutral either way. `REXL3_NEMO_P2TP_ROUTER=1` opts in, `=0` forces it off.
pub const ROUTER_SPLIT_DEFAULT: bool = false;
/// The shared-expert N-split on a second communicator (6.13): ON by default since rc2d; `REXL3_NEMO_P2TP_SHARED=0` / `=1`.
pub const SHARED_SPLIT_DEFAULT: bool = true;
/// The Mamba head-group split (6.10): ON by default since rc2d (R3's lever decision, opt-in before; the gates decide);
/// `REXL3_NEMO_P2TP_MAMBA_HG=0` / `=1`. Its path counter is `tp_arm::head_split_calls` (in `tp_arm::p2tp_counters_line`).
pub const MAMBA_HG_DEFAULT: bool = true;

/// One P2 TP exchange lever: its switch, its letter in the lever label and its default.
#[derive(Clone, Copy, Debug)]
pub struct Lever {
    pub var: &'static str,
    pub tag: char,
    pub default_on: bool,
}

/// Every P2 TP exchange lever (contract P2 at two ranks only; bit-neutral by construction), in the order of the lever label. The
/// runtime switches and the boot gate's label read them through [`p2tp_on`], so the label always says what runs.
pub const P2TP_LEVERS: [Lever; 9] = [
    Lever { var: "REXL3_NEMO_P2TP_INPROJ", tag: 'I', default_on: true },
    Lever { var: "REXL3_NEMO_P2TP_YGATHER", tag: 'G', default_on: true },
    Lever { var: "REXL3_NEMO_P2TP_FUSESPLIT", tag: 'F', default_on: true },
    Lever { var: "REXL3_NEMO_P2TP_OVERLAP", tag: 'O', default_on: true },
    Lever { var: "REXL3_NEMO_P2TP_COMBINE", tag: 'C', default_on: true },
    Lever { var: "REXL3_NEMO_P2TP_YOVERLAP", tag: 'Y', default_on: true },
    Lever { var: MAMBA_HG_VAR, tag: 'M', default_on: MAMBA_HG_DEFAULT },
    Lever { var: ROUTER_SPLIT_VAR, tag: 'R', default_on: ROUTER_SPLIT_DEFAULT },
    Lever { var: SHARED_SPLIT_VAR, tag: 'S', default_on: SHARED_SPLIT_DEFAULT },
];
/// See [`MAMBA_HG_DEFAULT`].
pub const MAMBA_HG_VAR: &str = "REXL3_NEMO_P2TP_MAMBA_HG";
/// See [`ROUTER_SPLIT_DEFAULT`].
pub const ROUTER_SPLIT_VAR: &str = "REXL3_NEMO_P2TP_ROUTER";
/// See [`SHARED_SPLIT_DEFAULT`].
pub const SHARED_SPLIT_VAR: &str = "REXL3_NEMO_P2TP_SHARED";

/// A lever's setting for a value of its variable: `0` off, `1` on, anything else (unset included) its default.
pub fn lever_value(v: Option<&str>, default_on: bool) -> bool {
    match v.map(str::trim) {
        Some("0") => false,
        Some("1") => true,
        _ => default_on,
    }
}

/// Whether the P2 TP lever `var` (one of [`P2TP_LEVERS`]) is on in this process. Read per call (an env lookup per layer of a prefill
/// slice); panics on a name that is not a lever (a programming error: the label would not carry it).
pub fn p2tp_on(var: &str) -> bool {
    let l = P2TP_LEVERS.iter().find(|l| l.var == var).unwrap_or_else(|| panic!("{var} is not a P2 TP lever"));
    lever_value(std::env::var(var).ok().as_deref(), l.default_on)
}

/// The P2 TP exchange levers as a compact label, e.g. `I1G1F1O1C1Y1M1R0S1` (the defaults since v0.1.1; rc2d's had `R1`).
pub fn p2tp_lever_label() -> String {
    lever_label_for(|l| p2tp_on(l.var))
}

/// [`p2tp_lever_label`] for given settings (pure, for the tests).
pub fn lever_label_for(on: impl Fn(&Lever) -> bool) -> String {
    P2TP_LEVERS.iter().map(|l| format!("{}{}", l.tag, on(l) as u8)).collect()
}

/// [`prefill_fingerprint`]'s format for given settings (pure, for the tests).
pub fn settings_label(c: PrefillContract, rows_cap: usize, absent_fast: bool) -> String {
    format!("contract={},rows={rows_cap},absent={}", c.id(), absent_fast as u8)
}

/// The suffix the server's `system_fingerprint` carries when this process's prefill settings are not the defaults
/// ([`DEFAULT_CONTRACT`] P2, the boot's prefill rows at [`default_prefill_rows`] (4,096), the absent fast path on, the fp32 router):
/// `pf<contract>r<rows>a<0|1>`, plus `rm` for P2's tensor-core router; e.g. `pfp1r512a1` for `REXL3_NEMO_PREFILL_CONTRACT=p1`,
/// `pfp2r4096a1rm` for the router opt-in, `pfp2r1024a1` for `--prefill-chunk 1024`. `None` at the defaults ('non-default only'). Since
/// rc2d a default process is P2 and shows no `_pf` field (rc2c's P2 arm showed `_pfp2r512a1`, P1 showed none): the default fingerprint
/// changed ONCE, with this flip. `r<rows>` is the boot's EFFECTIVE prefill rows ([`boot_prefill_rows`], v0.1.1; before, the environment's
/// P1 cap, which a P2 process does not run), so a fingerprint never names a slice the process does not cut. Rows and the absent path
/// change no bit; the contract and the router do (and a P2 rows cap of 64 or less, which [`check_rows_cap`] refuses at boot).
pub fn fingerprint_suffix() -> Option<String> {
    suffix_for(current(), boot_prefill_rows(), gb_cuda::nemotron_moe_prefill::absent_fast(), router_mma())
}

/// [`fingerprint_suffix`] for given settings (pure, for the tests). `router_mma` counts under P2 only (P1 never runs it).
pub fn suffix_for(c: PrefillContract, rows: usize, absent_fast: bool, router_mma: bool) -> Option<String> {
    let rm = c == PrefillContract::P2 && router_mma;
    let default = c == DEFAULT_CONTRACT && rows == default_prefill_rows() && absent_fast && !rm;
    (!default).then(|| format!("pf{}r{rows}a{}{}", c.id(), absent_fast as u8, if rm { "rm" } else { "" }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_contract_switch_takes_p1_or_p2_and_p2_is_the_default() {
        assert_eq!(DEFAULT_CONTRACT, PrefillContract::P2);
        assert_eq!(parse(None), Ok(PrefillContract::P2));
        assert_eq!(parse(Some("")), Ok(PrefillContract::P2));
        assert_eq!(parse(Some("P1")), Ok(PrefillContract::P1));
        assert_eq!(parse(Some(" p1 ")), Ok(PrefillContract::P1));
        assert_eq!(parse(Some(" p2 ")), Ok(PrefillContract::P2));
        for bad in ["p3", "mma", "1", "fp16", "p1,p2"] {
            let e = parse(Some(bad)).unwrap_err();
            // refused by name: the variable and the value are in the message
            assert!(e.contains(CONTRACT_VAR) && e.contains(bad), "{bad}: {e}");
        }
    }

    #[test]
    fn the_settings_label_and_the_fingerprint_suffix_name_every_setting() {
        assert_eq!(settings_label(PrefillContract::P1, 512, true), "contract=p1,rows=512,absent=1");
        assert_eq!(settings_label(PrefillContract::P1, 256, false), "contract=p1,rows=256,absent=0");
        // 'non-default only': the default (P2, the boot's 4,096 prefill rows, absent path on, fp32 router) adds nothing
        assert_eq!(default_prefill_rows(), P2_MAX_PREFILL_ROWS);
        // the shipped `--prefill-chunk` is at or above the default cap, so a default boot cuts the default rows and shows no field
        assert!(crate::serve::ServeOptions::new(Default::default()).prefill_chunk >= default_prefill_rows());
        assert_eq!(suffix_for(PrefillContract::P2, 4096, true, false), None);
        assert_eq!(suffix_for(PrefillContract::P2, 4096, true, true).as_deref(), Some("pfp2r4096a1rm"));
        assert_eq!(suffix_for(PrefillContract::P2, 4096, false, false).as_deref(), Some("pfp2r4096a0"));
        // under P2 the rows field is the boot's effective rows: any other slice is named (P1's 512 cap field is not P2's slice)
        assert_eq!(suffix_for(PrefillContract::P2, 512, true, false).as_deref(), Some("pfp2r512a1"));
        assert_eq!(suffix_for(PrefillContract::P2, 256, true, false).as_deref(), Some("pfp2r256a1"));
        // P1 is not the default any more: it always shows its own field (and never `rm`: P1 does not run the tensor-core router)
        assert_eq!(suffix_for(PrefillContract::P1, 512, true, false).as_deref(), Some("pfp1r512a1"));
        assert_eq!(suffix_for(PrefillContract::P1, 512, true, true).as_deref(), Some("pfp1r512a1"));
        assert_eq!(suffix_for(PrefillContract::P1, 256, true, false).as_deref(), Some("pfp1r256a1"));
        assert_eq!(suffix_for(PrefillContract::P1, 512, false, false).as_deref(), Some("pfp1r512a0"));
    }

    #[test]
    fn under_p2_a_rows_cap_of_64_or_less_is_refused_by_name_and_the_fingerprint_names_the_effective_rows() {
        // v0.1.1 (`docs/NEMOTRON_PREFILL_SPEED.md` 6.17): `--prefill-chunk` 16, 64, 65 and 4,096 at the 64-row window, under each
        // contract. The boot's rows are `engine::prefill_rows` for that contract: the chunk, at most the contract's cap (P1's default 512,
        // P2's 4,096).
        let rows = |c: PrefillContract, chunk: usize| chunk.min(c.max_prefill_rows(super::super::model::WIDE_PREFILL_ROWS));
        let levers = lever_label_for(|l| l.default_on);
        for (c, chunk, refused, suffix) in [
            (PrefillContract::P2, 16, true, Some("pfp2r16a1")),
            (PrefillContract::P2, 64, true, Some("pfp2r64a1")),
            (PrefillContract::P2, 65, false, Some("pfp2r65a1")),
            (PrefillContract::P2, 4096, false, None),
            (PrefillContract::P1, 16, false, Some("pfp1r16a1")),
            (PrefillContract::P1, 64, false, Some("pfp1r64a1")),
            (PrefillContract::P1, 65, false, Some("pfp1r65a1")),
            (PrefillContract::P1, 4096, false, Some("pfp1r512a1")),
        ] {
            let r = rows(c, chunk);
            match check_rows_cap(c, r, 64) {
                Ok(()) => assert!(!refused, "{c:?} chunk {chunk}: accepted"),
                Err(e) => {
                    assert!(refused, "{c:?} chunk {chunk}: {e}");
                    // by name: the flag, its value, the limit, the way out
                    let flag = format!("--prefill-chunk {chunk}");
                    for want in [flag.as_str(), "P2", "more than 64 rows", "65 or more", CONTRACT_VAR] {
                        assert!(e.contains(want), "{c:?} chunk {chunk}: {want:?} not in {e}");
                    }
                }
            }
            // the fingerprint names the slice the boot cuts (the defaults, P2 at 4,096, name nothing: the shipped string is unchanged)
            assert_eq!(suffix_for(c, r, true, false).as_deref(), suffix, "{c:?} chunk {chunk}");
            assert!(prefill_fingerprint_for(c, r, true, false, &levers).starts_with(&format!("contract={},rows={r},", c.id())));
        }
        // the cap is the model's (`model::step_rows_cap`): a decode window wider than 64 rows sizes the large-row and tensor-core MoE
        // scratches itself, so a narrow slice then runs P2's MoE block and is accepted
        assert!(check_rows_cap(PrefillContract::P2, 16, 128).is_ok());
        assert!(check_rows_cap(PrefillContract::P2, 64, 65).is_ok());
        assert!(check_rows_cap(PrefillContract::P2, 64, 64).is_err());
        assert!(check_rows_cap(PrefillContract::P1, 1, 64).is_ok());
    }

    #[test]
    fn every_lever_has_one_default_and_the_label_follows_the_settings() {
        // the lever defaults: shared split and head split ON (rc2d), the router split OFF (v0.1.1, rule R-ROUTER), each one constant;
        // the older levers on
        assert_eq!(lever_label_for(|l| l.default_on), "I1G1F1O1C1Y1M1R0S1");
        // the router opt-in (`REXL3_NEMO_P2TP_ROUTER=1`) is the label rc2d and v0.1.0 ran by default
        assert_eq!(lever_label_for(|l| l.default_on || l.tag == 'R'), "I1G1F1O1C1Y1M1R1S1");
        assert_eq!(lever_label_for(|l| l.default_on && l.tag != 'M'), "I1G1F1O1C1Y1M0R0S1");
        let mut vars: Vec<&str> = P2TP_LEVERS.iter().map(|l| l.var).collect();
        vars.sort_unstable();
        vars.dedup();
        assert_eq!(vars.len(), P2TP_LEVERS.len(), "a lever listed twice");
        for d in [true, false] {
            assert!(!lever_value(Some("0"), d));
            assert!(lever_value(Some(" 1 "), d));
            assert_eq!(lever_value(None, d), d);
            assert_eq!(lever_value(Some(""), d), d);
            assert_eq!(lever_value(Some("yes"), d), d);
        }
    }

    #[test]
    fn the_prefill_fact_names_the_contract_and_under_p2_the_router_and_the_levers() {
        let levers = lever_label_for(|l| l.default_on);
        // `rows` is the boot's prefill rows (`ModelOptions::prefill_rows`): 512 for a default P1 rank, 4,096 for a default P2 rank
        assert_eq!(prefill_fingerprint_for(PrefillContract::P1, 512, true, false, &levers), "contract=p1,rows=512,absent=1");
        assert_eq!(
            prefill_fingerprint_for(PrefillContract::P2, 4096, true, false, &levers),
            "contract=p2,rows=4096,absent=1,tp=I1G1F1O1C1Y1M1R0S1"
        );
        assert_eq!(
            prefill_fingerprint_for(PrefillContract::P2, 4096, true, true, &levers),
            "contract=p2,rows=4096,absent=1,router=mma,tp=I1G1F1O1C1Y1M1R0S1"
        );
        // the router split opted in (`REXL3_NEMO_P2TP_ROUTER=1`) is another fact: such a rank and a default one are refused at the Hello
        assert_eq!(
            prefill_fingerprint_for(PrefillContract::P2, 4096, true, false, &lever_label_for(|l| l.default_on || l.tag == 'R')),
            "contract=p2,rows=4096,absent=1,tp=I1G1F1O1C1Y1M1R1S1"
        );
    }

    #[test]
    fn p1_keeps_its_namespace_and_cap_and_the_default_is_p2() {
        assert_eq!(PrefillContract::P1.arith_tag(), "");
        assert_ne!(PrefillContract::P2.arith_tag(), "");
        assert_eq!(PrefillContract::P1.max_prefill_rows(256), 256);
        assert_eq!(PrefillContract::P1.max_prefill_rows(512), 512);
        assert_eq!(PrefillContract::P2.max_prefill_rows(256), P2_MAX_PREFILL_ROWS);
        assert!(check_built(PrefillContract::P1).is_ok());
        assert_eq!(check_built(PrefillContract::P2).is_ok(), P2_BUILT);
        assert!(PrefillContract::P2.applies_to_prefill_only());
        if std::env::var(CONTRACT_VAR).is_err() {
            assert_eq!(current(), PrefillContract::P2);
            assert_eq!(check_selected(), Ok(PrefillContract::P2));
        }
    }
}
