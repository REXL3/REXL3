//! A drafting script over a GLM backend (`GlmBackend` on one node, `GlmTpBackend` at rank 0 of a group): what the engine does with
//! `--spec mtp`, written out so that every number it sees can be digested. Shared by `glm_tp_mtp.rs` (the two-rank group against one
//! model) and `glm_mtp_probe.rs` (the single-node probe that must be identical before and after lane L12). Uses only APIs that exist at
//! the lane's merge base.
//!
//! Three sequences: a 2,300-token prompt (512-row slices: the batched expert path, past the DSA sparse threshold), a 200-token prompt
//! sliced at 128 rows (the deadline row limit), a 37-token prompt; then `ROUNDS` rounds of `propose_batch`
//! for every slot (slot 1 sampled, so its drafts are keyed; slots 0 and 2 greedy), a verify window of the last token and the drafts, and
//! commits whose keep counts follow a fixed pattern (the committed tokens are the window's own tokens, then the row's argmax), so that
//! accepted drafts, partial keeps and full rejections all occur whatever the drafts' quality on a layer prefix. A digest line per
//! answering call (candidates, every fp16 logit), per round (the drafts) and the keeps.
#![allow(dead_code)]

use gb_engine::{Backend, ProposeRequest, RowOutput, RowRequest};
use gb_sampling::Sampling;

pub const ROUNDS: usize = 10;
pub const COUNT: usize = 8;
pub const SLOTS: usize = 3;

pub fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h = (*h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
}

fn digest_rows(rows: &[RowOutput], logits: &[u16]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for r in rows {
        for &i in &r.ids {
            fnv(&mut h, &i.to_le_bytes());
        }
        for &v in &r.values {
            fnv(&mut h, &v.to_bits().to_le_bytes());
        }
    }
    for x in logits {
        fnv(&mut h, &x.to_le_bytes());
    }
    h
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % 150_000) as u32
    }
    fn tokens(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// What a run produced: one line per answering call and per round, and the drafted tokens of every round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    pub lines: Vec<String>,
    pub drafts: Vec<Vec<Option<Vec<u32>>>>,
    pub keeps: Vec<Vec<usize>>,
}

/// The keep count of slot `s` in round `r` with `d` drafts in its window: all of them, none, or some, by a fixed pattern.
fn keep_of(r: usize, s: usize, d: usize) -> usize {
    match (r + s) % 4 {
        0 => d + 1,
        1 => 1,
        2 => 1 + d.min(1),
        _ => 1 + d / 2,
    }
}

/// Runs the script on `be`; `logits(rows)` reads the fp16 logits of the last answering forward's `rows` rows (`GlmModel::logits_host`).
pub fn run<B: Backend>(be: &mut B, logits: impl Fn(&B, usize) -> Vec<u16>) -> Run {
    let mut lines = Vec::new();
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let prompts = [r.tokens(2300), r.tokens(200), r.tokens(37)];
    let mut seq: Vec<Vec<u32>> = Vec::new();
    for (slot, p) in prompts.iter().enumerate() {
        be.set_prefill_row_limit(if slot == 1 { Some(128) } else { None });
        let mut done = 0;
        loop {
            let step = be.prefill_slice(slot, p, &[], done, COUNT);
            done = step.done;
            if let Some(row) = step.row {
                let l = logits(be, 1);
                lines.push(format!("prefill {slot} {:016x}", digest_rows(std::slice::from_ref(&row), &l)));
                let mut s = p.clone();
                s.push(row.ids[0]);
                seq.push(s);
                break;
            }
        }
    }
    be.set_prefill_row_limit(None);
    assert!(be.take_error().is_none(), "the prompts failed");
    let samplers = [
        Sampling { temperature: 0.0, ..Default::default() },
        Sampling { seed: 0x5eed, temperature: 0.7, top_k: 20, top_p: 0.95, min_p: 0.0 },
        Sampling { temperature: 0.0, ..Default::default() },
    ];
    let mut drafts_all = Vec::new();
    let mut keeps_all = Vec::new();
    for round in 0..ROUNDS {
        // the drafts (none before the first window: a stream that has not committed a window has nothing to absorb but its prompt's carry)
        let reqs: Vec<ProposeRequest> =
            (0..SLOTS).map(|s| ProposeRequest { slot: s, tokens: &seq[s], sampling: &samplers[s], max: 7 }).collect();
        let d = be.propose_batch(&reqs);
        assert!(be.take_error().is_none(), "drafting failed in round {round}");
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for x in &d {
            match x {
                None => fnv(&mut h, &[0]),
                Some(v) => {
                    fnv(&mut h, &(v.len() as u64).to_le_bytes());
                    for t in v {
                        fnv(&mut h, &t.to_le_bytes());
                    }
                }
            }
        }
        lines.push(format!(
            "drafts {round} {:016x} {:?}",
            h,
            d.iter().map(|x| x.as_ref().map_or(0, |v| v.len())).collect::<Vec<_>>()
        ));
        // the verify window
        let w: Vec<RowRequest> = (0..SLOTS)
            .map(|s| {
                let mut t = vec![*seq[s].last().unwrap()];
                t.extend(d[s].clone().unwrap_or_default());
                RowRequest { slot: s, tokens: t, first_position: (seq[s].len() - 1) as u64 }
            })
            .collect();
        let rows: usize = w.iter().map(|q| q.tokens.len()).sum();
        let out = be.step(&w, COUNT);
        assert!(be.take_error().is_none(), "the window of round {round} failed");
        let l = logits(be, rows);
        let flat: Vec<RowOutput> = out.iter().flatten().cloned().collect();
        lines.push(format!("window {round} rows={rows} {:016x}", digest_rows(&flat, &l)));
        let mut keeps = Vec::new();
        for s in 0..SLOTS {
            let nd = w[s].tokens.len() - 1;
            let keep = keep_of(round, s, nd);
            be.commit(s, keep);
            // the committed tokens: the window's drafts kept, then the argmax of the last kept row
            seq[s].extend_from_slice(&w[s].tokens[1..keep]);
            seq[s].push(out[s][keep - 1].ids[0]);
            keeps.push(keep);
        }
        drafts_all.push(d);
        keeps_all.push(keeps);
    }
    for s in 0..SLOTS {
        be.release(s);
    }
    be.quiesce().unwrap();
    Run { lines, drafts: drafts_all, keeps: keeps_all }
}
