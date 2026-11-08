//! The Ling-3.1 prefill on the whole model (the real pack, ~106 GiB resident, MTP on): what a prompt leaves in a slot, compared at the bit
//! level. `#[ignore]`, env-gated (`REXL3_LING_DIR`; synthetic token ids: row invariance does not depend on them), one model per process
//! (`--test-threads=1`), with a 1 GiB `MemAvailable` guard that aborts the process.
//!
//! * `a_prompt_leaves_the_same_bits_every_time`: the same prompt prefilled twice (another prompt in between, so the second run starts
//!   from dirty scratch, records and pages) leaves the same KDA states and conv histories, MLA rows, last-row logits, MTP absorb and
//!   drafts, and the same greedy continuation; the digests it prints are the record a build is compared on (`DIGEST <name> <hex>` lines:
//!   a change that claims to keep every bit prints the same). One slot: the profile's, and what fits beside the weights on a GB10.

mod ling31_support;

use ling31_support::*;

const SLOTS: usize = 1;
const KV_TOKENS: usize = 32 * 256;
/// Five chunks of 512 rows and a tail: the trunk, the MTP absorb and the tap ring all run more than one chunk.
const N: usize = 2600;
const STEPS: usize = 32;

fn model() -> Option<std::sync::MutexGuard<'static, gb_model::families::bailing_moe_v3::model::BailingModel>> {
    shared(SLOTS, KV_TOKENS, true)
}

/// Everything the prompt `toks` leaves in `slot`, prefilled cold in calls of `piece` tokens, as named digests.
fn run(
    m: &mut gb_model::families::bailing_moe_v3::model::BailingModel,
    slot: usize,
    toks: &[u32],
    piece: usize,
) -> Vec<(String, String)> {
    let n = toks.len();
    m.begin_slot(slot).unwrap();
    let first = prefill(m, slot, toks, 0, n, piece);
    let mut out = vec![("logits".to_string(), logits_digest(m))];
    let d = slot_digest(m, slot, n, 0);
    out.extend([("kda_state".into(), d.kda_state), ("kda_conv".into(), d.kda_conv), ("mla".into(), d.mla)]);
    out.push(("first".into(), format!("{:?} {:?}", first.ids, first.vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>())));
    // the MTP layer absorbs the prompt (rows 0..n-1, in chunks), then drafts from the first sampled token
    m.mtp_absorb_up_to(slot, toks).unwrap();
    out.push(("mtp_absorb".into(), slot_digest(m, slot, n, n - 1).mtp));
    let mut seq = toks.to_vec();
    seq.push(first.ids[0] as u32);
    let drafts = m.mtp_propose(slot, &seq, 3).unwrap();
    out.push(("drafts".into(), format!("{drafts:?}")));
    let (tokens, trace) = greedy(m, slot, toks, &first, STEPS);
    out.push(("greedy_tokens".into(), format!("{tokens:?}")));
    out.push(("greedy_trace".into(), trace));
    m.end_slot(slot);
    out
}

#[test]
#[ignore]
fn a_prompt_leaves_the_same_bits_every_time() {
    let Some(mut m) = model() else { return };
    let m = &mut *m;
    let toks = ids(1, N);
    let a = run(m, 0, &toks, N);
    let _ = run(m, 0, &ids(2, 1100), 1100);
    let b = run(m, 0, &toks, N);
    for (name, value) in &a {
        println!("DIGEST {name} {value}");
    }
    assert_eq!(a, b, "the same prompt left other bits the second time");
    eprintln!("MemAvailable low {:.2} GiB", gib(guard().low()));
}
