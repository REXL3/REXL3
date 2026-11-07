//! The GLM backend as the engine drives it (`gb_engine::Backend`), on the whole model: a request that fails leaves the others alone.
//!
//! The trait's methods cannot fail, so the backend answers a failed call with NaN candidate rows and records the error for `take_error`, which
//! the engine reads after every call (a recoverable one fails the sequences the call served, a fatal one stops the process; see
//! `crates/engine/tests/failures.rs` for the engine's side, with a scripted backend). Here the real backend, the real model:
//!
//! * a prefill that fails (a position past the model's context; a prompt with an image the build cannot encode) comes back as NaN rows and a
//!   non-fatal `take_error`, frees its slot, and does not move a bit of a decode that is running in another slot;
//! * the freed slot serves the next request as if it had never failed (the same prompt gives the same bits in a fresh slot);
//! * a prompt fed in slices of any size and driven through `prefill_slice` gives the bits of one `prefill`.
//!
//! `#[ignore]` and env-gated (REXL3_GLM_DIR, REXL3_GLM_ORACLE_DIR for real token ids); under the GPU lock; one model per process.

mod glm_support;

use gb_engine::{Backend, ImageSpan, RowOutput, RowRequest};
use gb_model::families::glm::backend::GlmBackend;
use glm_support::*;
use std::path::PathBuf;

const SLOTS: usize = 4;
const KV_BLOCKS: usize = 40; // 10,240 tokens
const MAX_CONTEXT: usize = 1024;

fn tokens(n: usize) -> Option<Vec<u32>> {
    let dir = match std::env::var("REXL3_GLM_ORACLE_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            eprintln!("REXL3_GLM_ORACLE_DIR is not set: not run");
            return None;
        }
    };
    let ids = gb_model::families::glm::npy::read(&dir.join("code_700").join("ids.npy")).unwrap().to_i32().unwrap();
    assert!(ids.len() >= n);
    Some(ids[..n].iter().map(|&x| x as u32).collect())
}

/// Prefill `prompt` into `slot` slice by slice (the engine's way) and return the candidates after it.
fn prefill(be: &mut GlmBackend, slot: usize, prompt: &[u32]) -> RowOutput {
    let mut done = 0;
    loop {
        let step = be.prefill_slice(slot, prompt, &[], done, 8);
        assert!(step.done > done, "a slice must advance");
        done = step.done;
        if let Some(row) = step.row {
            assert_eq!(done, prompt.len());
            return row;
        }
    }
}

/// `n` greedy decode steps in `slot` from `row` (the candidates after the prompt), each committed; every step's candidates.
/// `between(i)` runs after step `i` (the failing requests of the other slot go there).
fn decode(
    be: &mut GlmBackend,
    slot: usize,
    prompt_len: usize,
    first: RowOutput,
    n: usize,
    mut between: impl FnMut(&mut GlmBackend, usize),
) -> Vec<RowOutput> {
    let mut rows = vec![first];
    for i in 0..n {
        let tok = rows.last().unwrap().ids[0];
        let out = be.step(&[RowRequest { slot, tokens: vec![tok], first_position: (prompt_len + i) as u64 }], 8);
        be.commit(slot, 1);
        rows.push(out[0][0].clone());
        between(be, i);
    }
    rows
}

fn same_bits(a: &[RowOutput], b: &[RowOutput], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: steps");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.ids, y.ids, "{what}: step {i} ids");
        assert!(x.values.iter().zip(&y.values).all(|(p, q)| p.to_bits() == q.to_bits()), "{what}: step {i} values");
    }
}

/// One model, loaded once, runs both scenarios (a second load in a process would not fit beside the first).
#[test]
#[ignore]
fn the_backend_isolates_failures_and_does_not_depend_on_slices_or_slots() {
    let Some(all) = tokens(300) else { return };
    let Some(model) = load_model(SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    let mut be = GlmBackend::new(model);
    failing_requests(&mut be, &all);
    slices_and_slots(&mut be, &all);
}

fn failing_requests(be: &mut GlmBackend, all: &[u32]) {
    assert_eq!(be.vocab_size(), 154_856, "the sampler is offered the tokenizer's vocabulary, not the model's padded one");
    let prompt = &all[..90];

    // the reference: slot 1 alone
    let first = prefill(be, 1, prompt);
    let reference = decode(be, 1, prompt.len(), first, 8, |_, _| {});
    assert!(be.take_error().is_none());
    be.release(1);

    // the same run with failing requests in slot 0 between the decode steps
    let first = prefill(be, 1, prompt);
    let mut failures = 0;
    let disturbed = decode(be, 1, prompt.len(), first, 8, |be, i| {
        // (a) a prompt that resumes past the model's context: a Request error on the first slice
        // (b) a prompt with an image: the build has no vision tower, so the backend refuses it by name
        let long: Vec<u32> = (0..MAX_CONTEXT as u32 + 100).map(|j| 100 + j % 50).collect();
        let step = if i % 2 == 0 {
            be.prefill_slice(0, &long, &[], MAX_CONTEXT + 4, 8)
        } else {
            let span = ImageSpan { start: 1, tokens: 4, merged: (2, 2), features: std::sync::Arc::new(()), key: 0 };
            be.prefill_slice(0, &long[..64], &[span], 0, 8)
        };
        let row = step.row.expect("a failed prefill answers with a row (of NaN) so that the engine's loop ends");
        assert!(row.values.iter().all(|v| v.is_nan()), "the failed prefill's candidates are NaN, which the sampler refuses");
        let e = be.take_error().expect("the failure is recorded");
        assert!(!e.fatal, "a refused request is not a fatal error: {e}");
        eprintln!("failure {i}: {e}");
        assert!(be.take_error().is_none(), "take_error hands each error out once");
        failures += 1;
    });
    assert_eq!(failures, 8);
    same_bits(&reference, &disturbed, "slot 1 decoding while slot 0's requests fail");
    be.release(1);

    // slot 0, freed by its failures, serves the same prompt with the same bits as slot 1 did
    let first = prefill(be, 0, prompt);
    let again = decode(be, 0, prompt.len(), first, 8, |_, _| {});
    same_bits(&reference, &again, "the freed slot 0 serving the same prompt");
    assert!(be.take_error().is_none());
    eprintln!("8 failed requests between 8 decode steps: the other slot's candidates are bit-identical to an undisturbed run; the freed slot serves the same bits");
}

fn slices_and_slots(be: &mut GlmBackend, all: &[u32]) {
    let prompt = &all[..260]; // 128 + 128 + 4: three slices
    let whole = be.prefill(2, 0, prompt, 8);
    assert!(be.take_error().is_none());
    let sliced = prefill(be, 3, prompt);
    assert_eq!(whole.ids, sliced.ids);
    assert!(whole.values.iter().zip(&sliced.values).all(|(a, b)| a.to_bits() == b.to_bits()), "slices and slots change no bit");
    let a = decode(be, 2, prompt.len(), whole, 6, |_, _| {});
    let b = decode(be, 3, prompt.len(), sliced, 6, |_, _| {});
    same_bits(&a, &b, "decode in two slots");
    // two sequences in one window: each gets the rows it would get alone (a window that is not committed is re-run from the committed state)
    let at = prompt.len() as u64 + 6;
    let req = |slot: usize, tok: u32| RowRequest { slot, tokens: vec![tok], first_position: at };
    let both = be.step(&[req(2, a[6].ids[0]), req(3, b[6].ids[0])], 8);
    let alone2 = be.step(&[req(2, a[6].ids[0])], 8);
    let alone3 = be.step(&[req(3, b[6].ids[0])], 8);
    same_bits(&[both[0][0].clone()], &[alone2[0][0].clone()], "slot 2 in a two-sequence window");
    same_bits(&[both[1][0].clone()], &[alone3[0][0].clone()], "slot 3 in a two-sequence window");
    assert!(be.take_error().is_none());
    eprintln!("one prefill, three slices and two slots give the same bits; a sequence in a two-sequence window gets the bits it gets alone");
}
