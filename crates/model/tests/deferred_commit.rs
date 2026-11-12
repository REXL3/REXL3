//! The deferred commit on the real model: the replay of a window's accepted rows is folded into the sequence's next window
//! kernel instead of running as its own pass (`Model::flush`, `gdn::window_owed`). It must leave every bit where an eager
//! commit leaves it: the candidates of every round, and the whole recurrent state (S and conv of all 48 layers) afterwards.
//!
//! Two runs of the same random rounds (sequences joining, leaving and reordering, windows of 1 to 8 rows, keeps from 1 to the
//! whole window) on one model: one lets the replay stay owed until a later window folds it in, the other applies it at once
//! after every commit (`Model::settle`), so its windows never see an owed replay.
//!
//! `owed_replay_precedes_every_prefill_and_slot_reuse` runs the same kind of comparison through the serving backend
//! (`QwenBackend`: `step`, `commit`, `release`, `prefill_piece`) with commits interleaved with chunked prefills: see its comment.
//!
//!     REXL3_QWEN_DIR=... cargo test --release -p gb-model --test deferred_commit -- --ignored --nocapture

use gb_engine::{Backend, RowOutput, RowRequest};
use gb_model::backend::QwenBackend;
use gb_model::config::QwenConfig;
use gb_model::forward::{Candidates, Model};
use gb_model::pack::Pack;
use gb_model::rows::RowBatch;
use gb_model::weights::Weights;

const COUNT: usize = 28;
const SLOTS: usize = 3;
const ROUNDS: usize = 40;

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

fn hash(c: &Candidates) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for (i, v) in c.ids.iter().zip(&c.vals) {
        fnv(&mut h, &i.to_le_bytes());
        fnv(&mut h, &v.to_bits().to_le_bytes());
    }
    h
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn tokens(rng: &mut Lcg, n: usize) -> Vec<u32> {
    (0..n).map(|_| 1000 + rng.below(99_000) as u32).collect()
}

/// Hash of the recurrent state of the first `slots` slots in every linear layer, one hash per (layer, S or conv).
fn state_hashes(m: &Model, slots: usize) -> Vec<u64> {
    use gb_cuda::gdn::{CONV_SLOT_ELEMS, S_SLOT_ELEMS};
    let mut out = Vec::new();
    for li in 0..m.pools.s.len() {
        let s = m.pools.s[li].to_vec::<f32>().unwrap();
        let mut h = 0xcbf29ce484222325u64;
        for x in &s[..slots * S_SLOT_ELEMS] {
            fnv(&mut h, &x.to_bits().to_le_bytes());
        }
        out.push(h);
        let c = m.pools.conv[li].to_vec::<u16>().unwrap();
        let mut h = 0xcbf29ce484222325u64;
        for x in &c[..slots * CONV_SLOT_ELEMS] {
            fnv(&mut h, &x.to_le_bytes());
        }
        out.push(h);
    }
    out
}

/// The same random rounds every call; `eager` applies each commit at once.
fn run(m: &mut Model, eager: bool) -> (Vec<u64>, Vec<u64>, usize) {
    let mut rng = Lcg(0x9E3779B97F4A7C15);
    let mut pos = [0usize; SLOTS];
    for (s, p) in pos.iter_mut().enumerate() {
        let prompt = tokens(&mut rng, 140 + 33 * s);
        m.begin_slot(s).unwrap();
        m.prefill_chunk(s, 0, &prompt, prompt.len(), None, 0).unwrap();
        *p = prompt.len();
    }
    let mut hashes = Vec::new();
    let mut folded = 0;
    let mut owed_last = [false; SLOTS];
    for _ in 0..ROUNDS {
        // a random non-empty subset of the slots in a random order, each with a window of 1..=8 rows
        let mut order: Vec<usize> = (0..SLOTS).collect();
        for i in (1..SLOTS).rev() {
            order.swap(i, rng.below(i + 1));
        }
        order.truncate(1 + rng.below(SLOTS));
        let reqs: Vec<RowRequest> = order
            .iter()
            .map(|&slot| {
                let n = 1 + rng.below(8);
                RowRequest { slot, tokens: tokens(&mut rng, n), first_position: pos[slot] as u64 }
            })
            .collect();
        let keeps: Vec<usize> = reqs.iter().map(|r| 1 + rng.below(r.tokens.len())).collect();
        folded += order.iter().filter(|&&s| owed_last[s]).count();
        let c = m.window_candidates(&RowBatch::from_requests(&reqs), COUNT).unwrap();
        hashes.push(hash(&c));
        owed_last = [false; SLOTS];
        for (r, &k) in reqs.iter().zip(&keeps) {
            m.commit(r.slot, k);
            pos[r.slot] += k;
            owed_last[r.slot] = true;
        }
        if eager {
            m.settle().unwrap();
            owed_last = [false; SLOTS];
        }
    }
    m.settle().unwrap();
    (hashes, state_hashes(m, SLOTS), folded)
}

#[test]
#[ignore]
fn deferred_commit_equals_eager_commit_on_the_real_model() {
    let Ok(d) = std::env::var("REXL3_QWEN_DIR") else { return eprintln!("REXL3_QWEN_DIR not set: skipping") };
    if !gb_cuda::device_available() {
        return eprintln!("no CUDA device: skipping");
    }
    let d = std::path::PathBuf::from(d);
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
    let mut m = Model::new(cfg, w, 6, 96, 16384, 4096).unwrap();
    let (h_def, s_def, folded) = run(&mut m, false);
    let (h_eag, s_eag, none) = run(&mut m, true);
    assert_eq!(none, 0, "the eager run must never fold");
    assert!(folded >= ROUNDS / 2, "the deferred run folded only {folded} sequences' commits into a window");
    let first = h_def.iter().zip(&h_eag).position(|(a, b)| a != b);
    assert_eq!(first, None, "candidates differ at round {first:?}");
    let bad = s_def.iter().zip(&s_eag).filter(|(a, b)| a != b).count();
    assert_eq!(bad, 0, "{bad} of {} state hashes differ after {ROUNDS} rounds", s_def.len());
    eprintln!("{ROUNDS} rounds, {folded} folded sequence-commits: candidates and all {} state hashes identical", s_def.len());
}

// ---------------------------------------------------------------------------------------------------------------------------
// Owed rows against prefills and slot reuse, through the serving backend.
//
// A window's accepted rows are replayed into a sequence's recurrent state by the NEXT window kernel that sequence runs, so after a
// commit the state of S is behind its tokens until then. Anything else that touches the slot first must settle it: a prefill that
// continues the sequence (`settle_slot`, inside `forward_mm`) and a new sequence in the slot (`begin_slot` / `end_slot` forget
// the rows: they belong to a sequence that is gone and must never reach the zeroed state). Rows of sequences that sit a window out
// are replayed on their own (`fold_owed`) before their records are overwritten, and a prefill of one slot leaves the other slots'
// rows owed.
//
// One scripted session on four slots, run twice on one model: deferred (the serving behaviour) and eager (`Model::settle` after
// every commit, so nothing is ever owed). Every candidate row of every call and all the recurrent state afterwards must be bit for
// bit equal. The script covers, with partial accepts throughout:
//   S1  a verify round, then a chunked prompt (another slot) while the rows stay owed, then a new prompt on a slot whose sequence
//       was cancelled (`release` + `prefill_piece` from 0), then a round over the reused slot and the slots that still owe rows;
//   S2  a verify round, then a prompt that CONTINUES a decoding slot (`prefill_piece` from its position, no `begin_slot`): its owed
//       rows must be in S before the prefill reads it, the other slot's stay owed;
//   S3  a verify round, then a new prompt on a slot with rows owed and no `release` (`begin_slot` alone drops them);
//   S4  rounds that leave sequences out (their rows go through the standalone replay) and rounds with every slot;
//   S5  the engine's round-robin: a prompt fed slice by slice (`prefill_slice`) with verify rounds of the other slots, partial
//       accepts, between the slices, then a round over the new slot and the others.
// Prompts above 128 tokens take Phase 4's pinned GEMM (several chunks of 160), one prompt of 100 the GEMV.
//
// Mutation-checked in `forward.rs` (each fails this test, the tree as committed passes): `settle_slot` moved to after the layer
// loop of a prefill (the continuation's candidates differ: the replay arrives after the prefill read S); `settle_slot` deleted (the
// owed set after the continuation is wrong: slot 2 still owes); `begin_slot` without `forget_owed` (the new prompt without a release
// gets the old sequence's rows replayed into its zeroed state: candidates differ); `end_slot` without `forget_owed` (the owed set
// after a release is wrong; the bits alone cannot show it because `begin_slot` forgets the rows again before any reuse); `fold_owed`
// with the standalone replay of the sequences that sit a round out dropped (the next round's candidates differ).
//
//     REXL3_QWEN_DIR=... cargo test --release -p gb-model --test deferred_commit owed_replay -- --ignored --nocapture

const ORD_SLOTS: usize = 4;
const CHUNK: usize = 160;

fn row_hash(r: &RowOutput) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for (i, v) in r.ids.iter().zip(&r.values) {
        fnv(&mut h, &i.to_le_bytes());
        fnv(&mut h, &v.to_bits().to_le_bytes());
    }
    h
}

struct Session<'a> {
    b: &'a mut QwenBackend,
    /// Settle every commit at once: the reference.
    eager: bool,
    rng: Lcg,
    /// Next position of each slot's sequence.
    pos: [usize; ORD_SLOTS],
    /// What each call returned (a hash of all its candidate rows), by name.
    log: Vec<(String, u64)>,
}

impl Session<'_> {
    fn checked(&mut self, what: &str) {
        let e = self.b.take_error();
        assert!(e.is_none(), "{what}: the backend reported an error: {:?}", e.map(|e| e.message));
    }

    /// The owed slots are what the script says (`deferred`), or none at all in the eager run: the test would pass vacuously if
    /// the deferred run never had anything owed at these points.
    fn expect_owed(&self, at: &str, deferred: &[usize]) {
        let want: Vec<usize> = if self.eager { Vec::new() } else { deferred.to_vec() };
        assert_eq!(self.b.m.owed_slots(), want, "owed slots {at} (eager: {})", self.eager);
    }

    /// A new prompt (`start` 0: the backend begins the slot) of `len` tokens, prefilled in chunks.
    fn prompt(&mut self, slot: usize, len: usize) {
        let toks = tokens(&mut self.rng, len);
        let row = self.b.prefill_piece(slot, 0, &toks, len, COUNT);
        self.checked("prompt");
        self.pos[slot] = len;
        self.log.push((format!("prompt of {len} tokens on slot {slot}"), row_hash(&row)));
    }

    /// `len` more tokens of the sequence in `slot`, prefilled from its position: no `begin_slot`.
    fn extend(&mut self, slot: usize, len: usize) {
        let start = self.pos[slot];
        let toks = tokens(&mut self.rng, len);
        let row = self.b.prefill_piece(slot, start as u64, &toks, start + len, COUNT);
        self.checked("continuation");
        self.pos[slot] = start + len;
        self.log.push((format!("{len} more tokens at {start} on slot {slot}"), row_hash(&row)));
    }

    /// One verify round: `(slot, rows, kept)` for each sequence, in this order.
    fn round(&mut self, name: &str, plan: &[(usize, usize, usize)]) {
        let reqs: Vec<RowRequest> = plan
            .iter()
            .map(|&(slot, rows, _)| RowRequest {
                slot,
                tokens: tokens(&mut self.rng, rows),
                first_position: self.pos[slot] as u64,
            })
            .collect();
        let outs = self.b.step(&reqs, COUNT);
        self.checked(name);
        let mut h = 0xcbf29ce484222325u64;
        for r in outs.iter().flatten() {
            fnv(&mut h, &row_hash(r).to_le_bytes());
        }
        self.log.push((format!("round {name}"), h));
        for &(slot, _, keep) in plan {
            self.b.commit(slot, keep);
            self.pos[slot] += keep;
        }
        if self.eager {
            self.b.m.settle().unwrap();
        }
    }

    /// A new prompt fed one chunk at a time through `prefill_slice`, as the engine's round-robin does, with a verify round of the
    /// other sequences (`between[i]`, as in `round`) after chunk `i`; `owed` is what the deferred run owes after the first chunk.
    fn sliced_prompt(&mut self, slot: usize, len: usize, between: &[&[(usize, usize, usize)]], owed: &[usize]) {
        let toks = tokens(&mut self.rng, len);
        let (mut done, mut i) = (0, 0);
        loop {
            let step = self.b.prefill_slice(slot, &toks, &[], done, COUNT);
            self.checked("slice");
            done = step.done;
            if i == 0 {
                self.expect_owed("after the first slice of a prompt", owed);
            }
            if let Some(row) = step.row {
                self.log.push((format!("sliced prompt of {len} tokens on slot {slot}"), row_hash(&row)));
                break;
            }
            if let Some(plan) = between.get(i) {
                self.round(&format!("between slices {i} on slot {slot}"), plan);
            }
            i += 1;
        }
        self.pos[slot] = len;
    }

    /// A finished or cancelled sequence gives its slot back.
    fn release(&mut self, slot: usize) {
        self.b.release(slot);
        self.checked("release");
    }
}

/// The session; returns every call's hash and the hashes of the whole recurrent state at the end.
fn session(b: &mut QwenBackend, eager: bool) -> (Vec<(String, u64)>, Vec<u64>) {
    let mut s = Session { b, eager, rng: Lcg(0xD1B54A32D192ED03), pos: [0; ORD_SLOTS], log: Vec::new() };
    s.prompt(0, 300); // pinned GEMM, two chunks
    s.prompt(1, 100); // the decode GEMV, one chunk
    s.prompt(2, 420); // pinned GEMM, three chunks
    s.round("R1", &[(0, 6, 2), (1, 5, 3), (2, 4, 1)]);
    s.expect_owed("after R1 (not flushed yet)", &[]);
    // S1: another slot's chunked prompt flushes R1's commits and leaves them owed
    s.prompt(3, 200);
    s.expect_owed("after a prompt on slot 3", &[0, 1, 2]);
    // the sequence on slot 0 is cancelled with its rows owed; a new prompt takes the slot
    s.release(0);
    s.expect_owed("after releasing slot 0", &[1, 2]);
    s.prompt(0, 260);
    s.expect_owed("after the new prompt on slot 0", &[1, 2]);
    // S4: slot 1 sits this round out (its rows are replayed on their own), slot 2 folds its rows in, slot 0 is new
    s.round("R2", &[(2, 4, 3), (0, 3, 2)]);
    s.expect_owed("after R2", &[]);
    // S2: slot 2 continues its sequence by a prompt; its owed rows go in first, slot 0's stay owed
    s.extend(2, 200);
    s.expect_owed("after the continuation of slot 2", &[0]);
    // S3: a new prompt on slot 0 with its R2 rows still owed and no release: `begin_slot` drops them
    s.prompt(0, 150);
    s.expect_owed("after the new prompt on slot 0 without a release", &[]);
    s.round("R3", &[(0, 8, 1), (3, 5, 5), (2, 2, 2), (1, 4, 3)]);
    // S4: only slot 3 runs: the other three slots' rows from R3 are replayed on their own
    s.round("R4", &[(3, 3, 2)]);
    s.round("R5", &[(0, 4, 4), (1, 4, 1), (2, 4, 2), (3, 4, 3)]);
    // the engine's round-robin: slot 1's sequence ends with its R5 rows pending, and its next prompt is fed slice by slice while
    // the other sequences keep verifying with partial accepts between the slices
    s.release(1);
    s.sliced_prompt(1, 400, &[&[(0, 4, 2), (2, 3, 1)], &[(3, 5, 3)]], &[0, 2, 3]);
    s.round("R6", &[(1, 5, 2), (0, 4, 3), (2, 4, 4), (3, 2, 1)]);
    s.b.m.settle().unwrap();
    s.expect_owed("after the final settle", &[]);
    let log = std::mem::take(&mut s.log);
    (log, state_hashes(&s.b.m, ORD_SLOTS))
}

#[test]
#[ignore]
fn owed_replay_precedes_every_prefill_and_slot_reuse() {
    let Ok(d) = std::env::var("REXL3_QWEN_DIR") else { return eprintln!("REXL3_QWEN_DIR not set: skipping") };
    if !gb_cuda::device_available() {
        return eprintln!("no CUDA device: skipping");
    }
    let d = std::path::PathBuf::from(d);
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
    let m = Model::new(cfg, w, ORD_SLOTS, 96, 16384, 256).unwrap();
    let mut b = QwenBackend::new(m, CHUNK);
    let (log_def, state_def) = session(&mut b, false);
    let (log_eag, state_eag) = session(&mut b, true);
    assert_eq!(log_def.len(), log_eag.len());
    let first = log_def.iter().zip(&log_eag).find(|(a, b)| a != b);
    assert!(first.is_none(), "the deferred and the eager session differ at: {first:?}");
    let bad = state_def.iter().zip(&state_eag).filter(|(a, b)| a != b).count();
    assert_eq!(bad, 0, "{bad} of {} state hashes differ at the end", state_def.len());
    eprintln!("{} calls and all {} state hashes identical", log_def.len(), state_def.len());
}
