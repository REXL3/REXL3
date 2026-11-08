//! The exactness harness. Every check is generic over the backend, so the same code gates the CPU mock
//! today and the CUDA backend on the GB10. A kernel change is not done until all of these pass.
//!
//! 1. `row_invariance`: a row's bits in a multi-sequence, multi-row step equal a one-row step's.
//! 2. `serial_batched_equals_solo`: N streams decoding together equal each stream alone.
//! 3. `drafted_equals_serial`: drafting on, N streams together, equals drafting off, each alone.
//! 4. `chunked_prefill_equals_whole`: a prompt fed in pieces (a resumed conversation) equals one piece.
//! 5. `rollback_equals_replay`: after a partly rejected window, the next rows equal those of a backend that
//!    only ever decoded the accepted tokens one at a time (recurrent state and KV rolled back exactly).

use crate::{Backend, Engine, Output, Request, RowOutput, RowRequest};
use gb_sampling::choose;

#[derive(Debug, Default)]
pub struct Report {
    pub sequences: usize,
    pub tokens: u64,
    pub rounds: u64,
    pub drafted: u64,
    pub accepted: u64,
    /// Sequences that finished on a stop token.
    pub stopped: usize,
    /// Of those, how many stopped on a token inside a draft run (an accepted draft or the sample past one).
    pub stopped_in_draft: usize,
}

impl Report {
    pub fn tokens_per_round(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.tokens as f64 / self.rounds as f64
        }
    }
}

fn describe(a: &RowOutput, b: &RowOutput) -> String {
    let n = a.ids.len().min(b.ids.len()).min(4);
    format!("ids {:?} vs {:?}, values {:?} vs {:?}", &a.ids[..n], &b.ids[..n], &a.values[..n], &b.values[..n])
}

/// `cases`: (context, window) pairs; context has >= 2 tokens, window[0] is the token after it.
pub fn row_invariance<B: Backend>(make: &dyn Fn() -> B, cases: &[(Vec<u32>, Vec<u32>)], candidates: usize) -> Result<(), String> {
    // References: one row per step, built up one committed token at a time.
    let mut refs: Vec<Vec<RowOutput>> = Vec::new();
    for (ctx, win) in cases {
        let mut per_row = Vec::new();
        for j in 0..win.len() {
            let mut b = make();
            b.prefill(0, 0, ctx, candidates);
            let mut pos = ctx.len() as u64;
            for &t in &win[..j] {
                b.step(&[RowRequest { slot: 0, tokens: vec![t], first_position: pos }], candidates);
                b.commit(0, 1);
                pos += 1;
            }
            let out = b.step(&[RowRequest { slot: 0, tokens: vec![win[j]], first_position: pos }], candidates);
            per_row.push(out.into_iter().next().unwrap().into_iter().next().unwrap());
        }
        refs.push(per_row);
    }
    // All cases in one step, in two different orders.
    for reversed in [false, true] {
        let mut b = make();
        let mut order: Vec<usize> = (0..cases.len()).collect();
        if reversed {
            order.reverse();
        }
        let mut reqs = Vec::new();
        for (slot, &i) in order.iter().enumerate() {
            let (ctx, win) = &cases[i];
            b.prefill(slot, 0, ctx, candidates);
            reqs.push(RowRequest { slot, tokens: win.clone(), first_position: ctx.len() as u64 });
        }
        let outs = b.step(&reqs, candidates);
        for (k, &i) in order.iter().enumerate() {
            for (j, row) in outs[k].iter().enumerate() {
                if !row.bits_eq(&refs[i][j]) {
                    return Err(format!(
                        "case {i} row {j} (batch order reversed={reversed}) differs from its one-row step: {}",
                        describe(row, &refs[i][j])
                    ));
                }
            }
        }
    }
    Ok(())
}

fn run_all<B: Backend>(engine: &mut Engine<B>, reqs: &[Request]) -> Result<Vec<Output>, String> {
    for (i, r) in reqs.iter().enumerate() {
        engine.submit(r.clone()).map_err(|e| format!("request {i} refused: {e}"))?;
    }
    engine.run().map_err(|e| format!("engine step failed: {e}"))?;
    let outs = engine.take_outputs();
    if outs.len() != reqs.len() {
        return Err(format!("{} requests produced {} outputs: a sequence was lost", reqs.len(), outs.len()));
    }
    Ok(outs)
}

fn solo<B: Backend>(make_engine: &dyn Fn(bool) -> Engine<B>, reqs: &[Request]) -> Result<Vec<Vec<u32>>, String> {
    reqs.iter()
        .map(|r| {
            let mut e = make_engine(false);
            Ok(run_all(&mut e, std::slice::from_ref(r))?.remove(0).tokens)
        })
        .collect()
}

pub(crate) fn compare(label: &str, want: &[Vec<u32>], got: &[Output]) -> Result<Report, String> {
    if want.len() != got.len() {
        return Err(format!("{label}: {} sequences expected, {} returned", want.len(), got.len()));
    }
    let mut rep = Report { sequences: got.len(), ..Default::default() };
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        if *w != g.tokens {
            let at = w.iter().zip(&g.tokens).position(|(a, b)| a != b).unwrap_or(w.len().min(g.tokens.len()));
            return Err(format!(
                "{label}: sequence {i} diverges at generated token {at} (len {} vs {})",
                w.len(),
                g.tokens.len()
            ));
        }
        rep.tokens += g.tokens.len().saturating_sub(1) as u64;
        rep.rounds += g.rounds;
        rep.drafted += g.drafted;
        rep.accepted += g.accepted;
        rep.stopped += usize::from(g.stopped);
        rep.stopped_in_draft += usize::from(g.stopped_in_draft);
    }
    Ok(rep)
}

/// `make_engine(drafting)` builds a fresh engine on a fresh backend.
pub fn serial_batched_equals_solo<B: Backend>(
    make_engine: &dyn Fn(bool) -> Engine<B>,
    reqs: &[Request],
) -> Result<Report, String> {
    let want = solo(make_engine, reqs)?;
    let mut e = make_engine(false);
    compare("serial batched vs solo", &want, &run_all(&mut e, reqs)?)
}

pub fn drafted_equals_serial<B: Backend>(make_engine: &dyn Fn(bool) -> Engine<B>, reqs: &[Request]) -> Result<Report, String> {
    let want = solo(make_engine, reqs)?;
    let mut e = make_engine(true);
    compare("drafted batched vs serial solo", &want, &run_all(&mut e, reqs)?)
}

/// Feed `prompt` whole and in the given pieces, then decode `decode` tokens serially from each:
/// the last prefill row and every decoded token must match.
pub fn chunked_prefill_equals_whole<B: Backend>(
    make: &dyn Fn() -> B,
    prompt: &[u32],
    cuts: &[usize],
    sampling: &gb_sampling::Sampling,
    decode: usize,
) -> Result<(), String> {
    let cand = sampling.candidate_count(make().vocab_size());
    let run = |pieces: Vec<&[u32]>| -> (RowOutput, Vec<u32>) {
        let mut b = make();
        let mut pos = 0u64;
        let mut last = None;
        for p in pieces {
            last = Some(b.prefill_piece(0, pos, p, prompt.len(), cand));
            pos += p.len() as u64;
        }
        let last = last.unwrap();
        let mut toks = vec![choose(&last.values, &last.ids, pos, sampling)];
        for _ in 0..decode {
            let o = b.step(&[RowRequest { slot: 0, tokens: vec![*toks.last().unwrap()], first_position: pos }], cand);
            b.commit(0, 1);
            pos += 1;
            let r = &o[0][0];
            toks.push(choose(&r.values, &r.ids, pos, sampling));
        }
        (last, toks)
    };
    let (whole_row, whole) = run(vec![prompt]);
    let mut pieces = Vec::new();
    let mut at = 0;
    for &c in cuts {
        pieces.push(&prompt[at..c]);
        at = c;
    }
    pieces.push(&prompt[at..]);
    let (chunk_row, chunked) = run(pieces);
    if !whole_row.bits_eq(&chunk_row) {
        return Err(format!("last prefill row differs with cuts {cuts:?}: {}", describe(&whole_row, &chunk_row)));
    }
    if whole != chunked {
        return Err(format!("decoded tokens differ with cuts {cuts:?}"));
    }
    Ok(())
}

/// After a step whose window was only partly kept, the rows of the next step must equal those of a backend that
/// decoded the kept tokens one at a time. `window` is the step's tokens (`window[0]` is the unfed last token),
/// `keep` how many rows to commit, `next` the following step's tokens (first token = the new unfed one).
pub fn rollback_equals_replay<B: Backend>(
    make: &dyn Fn() -> B,
    ctx: &[u32],
    window: &[u32],
    keep: usize,
    next: &[u32],
    candidates: usize,
) -> Result<(), String> {
    if keep == 0 || keep > window.len() {
        return Err(format!("keep {keep} must be in 1..={}", window.len()));
    }
    let after = (ctx.len() + keep) as u64;
    let mut a = make();
    a.prefill(0, 0, ctx, candidates);
    let first = a.step(&[RowRequest { slot: 0, tokens: window.to_vec(), first_position: ctx.len() as u64 }], candidates);
    if first.len() != 1 || first[0].len() != window.len() {
        return Err("the windowed step returned the wrong number of rows".into());
    }
    a.commit(0, keep);
    let got = a.step(&[RowRequest { slot: 0, tokens: next.to_vec(), first_position: after }], candidates);

    let mut b = make();
    b.prefill(0, 0, ctx, candidates);
    let mut pos = ctx.len() as u64;
    for &t in &window[..keep] {
        b.step(&[RowRequest { slot: 0, tokens: vec![t], first_position: pos }], candidates);
        b.commit(0, 1);
        pos += 1;
    }
    let want = b.step(&[RowRequest { slot: 0, tokens: next.to_vec(), first_position: after }], candidates);
    if got.len() != 1 || want.len() != 1 || got[0].len() != want[0].len() {
        return Err("the follow-up steps returned different numbers of rows".into());
    }
    for (j, (g, w)) in got[0].iter().zip(&want[0]).enumerate() {
        if !g.bits_eq(w) {
            return Err(format!(
                "after committing {keep} of {} rows, follow-up row {j} differs from a clean replay: {}",
                window.len(),
                describe(g, w)
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out(id: u64, tokens: &[u32]) -> Output {
        Output { id, tokens: tokens.to_vec(), ..Default::default() }
    }

    #[test]
    fn compare_catches_a_lost_or_extra_sequence() {
        let want = vec![vec![1, 2], vec![3]];
        assert!(compare("t", &want, &[out(0, &[1, 2]), out(1, &[3])]).is_ok());
        let lost = compare("t", &want, &[out(0, &[1, 2])]).unwrap_err();
        assert!(lost.contains("2 sequences expected, 1 returned"), "{lost}");
        let extra = compare("t", &want, &[out(0, &[1, 2]), out(1, &[3]), out(2, &[9])]).unwrap_err();
        assert!(extra.contains("expected, 3 returned"), "{extra}");
        let diverged = compare("t", &want, &[out(0, &[1, 9]), out(1, &[3])]).unwrap_err();
        assert!(diverges(&diverged), "{diverged}");
    }

    fn diverges(e: &str) -> bool {
        e.contains("diverges at generated token 1")
    }
}
