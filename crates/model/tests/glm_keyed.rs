//! Keyed-noise drafting on the whole GLM-5.3-Flash model with the NextN layer (90.5 GiB resident): the head's draft is the TARGET's keyed draw over the
//! head's own candidates at the draft's position (`GlmModel::mtp_draft_keyed`, `gb_spec::DraftPick`; docs/GLM53.md "Keyed-noise drafting"). All `#[ignore]`,
//! env-gated (REXL3_GLM_DIR: the pack; REXL3_GLM_ORACLE_DIR: the oracle's dumps, for prompt token ids); without them a test prints "not run". One model per
//! process: run with `--test-threads=1` under the GPU lock.
//!
//! * `keyed_drafts_are_the_hosts_pick_over_the_heads_candidates`: four streams of different samplers (and one greedy) in ONE draft call: every draft at every
//!   depth equals `gb_spec::try_draft_pick` on the very candidates the head produced (fetched by `mtp_debug_chain`, which follows the host's picks);
//! * `the_restricted_keyed_draft_equals_the_full_keyed_draft_where_the_kept_candidates_are_inside_the_set`: the restricted head's guarantee with noise (the noise
//!   of a kept column is the full head's noise for that token id);
//! * `a_greedy_draft_is_the_argmax_draft_bit_for_bit_keyed_or_not`: no noise on a greedy request, and a call with no keyed row runs the launches it always ran;
//! * `keyed_drafting_in_the_loop_equals_serial`: the drafter proposes keyed, the verify window of 1 + K rows decides, the committed text equals one-row decoding
//!   for K = 1..4 and several seeds, and the acceptance is printed against the argmax drafter's.
//!
//! `REXL3_MTP_MUTANT=keyed_noise_by_column|keyed_position_off|keyed_no_top_k|keyed_greedy_noise|restricted_no_map` runs the drafter with that deliberate bug: each
//! of the first three tests must then FAIL (`tools/glm_keyed_mutants.sh`).

mod glm_support;

use gb_engine::RowRequest;
use gb_model::families::glm::model::{vocab_ranges, DraftAsk, GlmModel, HeadList, MtpMutant, VocabMode};
use gb_model::families::glm::npy;
use gb_model::rows::RowBatch;
use gb_sampling::{try_choose, Sampling};
use gb_spec::{draft_position, try_draft_pick, DraftPick};
use glm_support::*;
use std::path::{Path, PathBuf};

const SLOTS: usize = 8;
const KV_BLOCKS: usize = 80;
const MAX_CONTEXT: usize = 8192;
/// Candidates of a target row (`top_k` 20 + the sampler's margin 8 is 28; the largest `top_k` of these tests is 40).
const CAND: usize = 48;

fn model() -> Option<std::sync::MutexGuard<'static, GlmModel>> {
    let mut g = shared_model_mtp(SLOTS, KV_BLOCKS, MAX_CONTEXT)?;
    if let Ok(v) = std::env::var("REXL3_MTP_MUTANT") {
        let mutant = match v.as_str() {
            "keyed_noise_by_column" => MtpMutant::KeyedNoiseByColumn,
            "keyed_position_off" => MtpMutant::KeyedPositionOff,
            "keyed_no_top_k" => MtpMutant::KeyedNoTopK,
            "keyed_greedy_noise" => MtpMutant::KeyedGreedyNoise,
            "restricted_no_map" => MtpMutant::RestrictedNoMap,
            other => panic!("REXL3_MTP_MUTANT={other} is not a keyed-drafting mutant"),
        };
        g.set_mtp_mutant(Some(mutant));
    }
    Some(g)
}

fn oracle_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_GLM_ORACLE_DIR") {
        Ok(d) if Path::new(&d).is_dir() => Some(PathBuf::from(d)),
        Ok(d) => panic!("REXL3_GLM_ORACLE_DIR={d} is not a directory"),
        Err(_) => {
            eprintln!("REXL3_GLM_ORACLE_DIR is not set: not run");
            None
        }
    }
}

fn ids_of(oracle: &Path, prompt: &str) -> Vec<u32> {
    npy::read(&oracle.join(prompt).join("ids.npy")).unwrap().to_i32().unwrap().iter().map(|&x| x as u32).collect()
}

/// One sequence of a test: its slot, its tokens (the last one not yet fed) and its sampler.
struct Stream {
    slot: usize,
    tokens: Vec<u32>,
    s: Sampling,
}

fn sample(c: &[i32], v: &[f32], position: usize, s: &Sampling) -> u32 {
    let n = s.candidate_count(154_880).min(c.len());
    let ids: Vec<u32> = c[..n].iter().map(|&x| x as u32).collect();
    try_choose(&v[..n], &ids, position as u64, s).unwrap()
}

/// Prefill `prompt` into `slot` (128-row chunks) and start the stream: its first token is the keyed sample after the prompt.
fn start(m: &mut GlmModel, slot: usize, prompt: &[u32], s: Sampling) -> Stream {
    m.begin_slot(slot).unwrap();
    let mut at = 0;
    let mut first = 0;
    while at < prompt.len() {
        let end = (at + m.chunk_rows()).min(prompt.len());
        let want = (end == prompt.len()).then_some(CAND);
        if let Some(c) = m.prefill_chunk(slot, at, &prompt[at..end], want).unwrap() {
            first = sample(&c.ids[..CAND], &c.vals[..CAND], prompt.len(), &s);
        }
        at = end;
    }
    let mut tokens = prompt.to_vec();
    tokens.push(first);
    Stream { slot, tokens, s }
}

/// One serial row for every stream together (one window, one row each), each stream's next token by its own sampler; commits them.
fn advance(m: &mut GlmModel, streams: &mut [&mut Stream]) {
    let reqs: Vec<RowRequest> = streams
        .iter()
        .map(|st| RowRequest {
            slot: st.slot,
            tokens: vec![*st.tokens.last().unwrap()],
            first_position: (st.tokens.len() - 1) as u64,
        })
        .collect();
    let c = m.window_candidates(&RowBatch::from_requests(&reqs), CAND).unwrap();
    for (i, st) in streams.iter_mut().enumerate() {
        let t = sample(&c.ids[i * CAND..(i + 1) * CAND], &c.vals[i * CAND..(i + 1) * CAND], st.tokens.len(), &st.s);
        m.commit(st.slot, 1);
        st.tokens.push(t);
    }
}

/// A host chain: the host's picks, and the head's candidates at each depth.
type HostChain = (Vec<u32>, Vec<HeadList>);

/// The host's draft chain for `st`: the candidates the head produced at each of `depth` depths, the chain following the host's own keyed pick over them.
fn host_chain(m: &mut GlmModel, st: &Stream, depth: usize, restricted: bool) -> HostChain {
    let len = st.tokens.len();
    let mut picks: Vec<u32> = Vec::new();
    let mut f = |k: usize, ids: &[u32], vals: &[f32]| {
        let t = try_draft_pick(vals, ids, draft_position(len, k), &st.s, DraftPick::Keyed).unwrap();
        picks.push(t);
        t
    };
    let lists = m.mtp_debug_chain(st.slot, &st.tokens, depth, 64, restricted, false, &mut f).unwrap().expect("the slot drafts");
    (picks, lists)
}

fn samplers() -> Vec<Sampling> {
    vec![
        Sampling { seed: 11, temperature: 0.7, top_k: 20, top_p: 0.95, min_p: 0.0 },
        Sampling { seed: 12, temperature: 1.0, top_k: 8, top_p: 1.0, min_p: 0.0 },
        Sampling { seed: 13, temperature: 1.3, top_k: 20, top_p: 0.9, min_p: 0.05 },
        Sampling { seed: 0, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 },
    ]
}

fn prompts(oracle: &Path) -> Vec<Vec<u32>> {
    let chat = ids_of(oracle, "chat_300");
    let code = ids_of(oracle, "code_700");
    vec![chat[..150].to_vec(), code[..300].to_vec(), ids_of(oracle, "short_20"), chat]
}

#[test]
#[ignore]
fn keyed_drafts_are_the_hosts_pick_over_the_heads_candidates() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_vocab(VocabMode::Full);
    let (mut compared, mut differ_from_argmax, mut keyed_compared) = (0usize, 0usize, 0usize);
    let ps = prompts(&oracle);
    let ss = samplers();
    let mut streams: Vec<Stream> = ps.iter().zip(&ss).enumerate().map(|(i, (p, s))| start(&mut m, i, p, *s)).collect();
    for round in 0..24usize {
        let depth = 1 + round % 4;
        // the host's chains first (they leave the slots' pending rows alone), then the one device call for all four streams
        let host: Vec<HostChain> = streams.iter().map(|st| host_chain(&mut m, st, depth, false)).collect();
        let asks: Vec<DraftAsk> = streams.iter().map(|st| DraftAsk { slot: st.slot, tokens: &st.tokens, depth }).collect();
        let keyed: Vec<Option<Sampling>> = streams.iter().map(|st| Some(st.s)).collect();
        let got = m.mtp_draft_keyed(&asks, &keyed).unwrap();
        for (i, st) in streams.iter().enumerate() {
            let d = got[i].as_ref().expect("the stream drafts");
            assert_eq!(d.ids, host[i].0, "round {round} stream {i} ({:?}) depth {depth}: the device's drafts differ from the host's keyed pick over the head's candidates", st.s);
            for k in 0..depth {
                // the value returned is the pick's own logit
                let at = host[i].1[k].0.iter().position(|&x| x == d.ids[k]).expect("the pick is one of the candidates");
                assert_eq!(
                    d.vals[k].to_bits(),
                    host[i].1[k].1[at].to_bits(),
                    "round {round} stream {i} depth {k}: the returned logit"
                );
                compared += 1;
                if st.s.temperature > 0.0 {
                    keyed_compared += 1;
                    differ_from_argmax += usize::from(d.ids[k] != host[i].1[k].0[0]);
                } else {
                    assert_eq!(d.ids[k], host[i].1[k].0[0], "a greedy stream's draft is the argmax");
                }
            }
        }
        let mut refs: Vec<&mut Stream> = streams.iter_mut().collect();
        advance(&mut m, &mut refs);
    }
    eprintln!("{compared} drafts equal the host's pick; {differ_from_argmax} of {keyed_compared} sampled drafts are not the head's argmax");
    assert!(
        differ_from_argmax >= 12,
        "the noise changed only {differ_from_argmax} drafts: the comparison would not see a missing noise"
    );
}

#[test]
#[ignore]
#[allow(clippy::needless_range_loop)]
fn the_restricted_keyed_draft_equals_the_full_keyed_draft_where_the_kept_candidates_are_inside_the_set() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    if m.mtp_sub_head().is_none() {
        m.mtp_restrict_vocab(&vocab_ranges(512, 154_880)).unwrap();
    }
    let chat = ids_of(&oracle, "chat_300");
    let code = ids_of(&oracle, "code_700");
    let ss = [
        Sampling { seed: 21, temperature: 0.7, top_k: 20, top_p: 0.95, min_p: 0.0 },
        Sampling { seed: 22, temperature: 1.0, top_k: 40, top_p: 1.0, min_p: 0.0 },
    ];
    let (mut inside, mut outside, mut differ_from_argmax) = (0usize, 0usize, 0usize);
    for (pi, prompt) in [chat[..200].to_vec(), code[..400].to_vec()].iter().enumerate() {
        let s = ss[pi];
        // two slots fed the same tokens hold the same state: slot 0 drafts with the full head, slot 1 with the restricted one
        let (mut a, mut b) = (start(&mut m, 0, prompt, s), start(&mut m, 1, prompt, s));
        assert_eq!(a.tokens, b.tokens);
        for round in 0..30usize {
            let depth = 1 + round % 4;
            // the full head's chain and candidates, by its own keyed picks
            m.set_mtp_vocab(VocabMode::Full);
            let (full_picks, lists) = host_chain(&mut m, &a, depth, false);
            let full = m
                .mtp_draft_keyed(&[DraftAsk { slot: 0, tokens: &a.tokens, depth }], &[Some(s)])
                .unwrap()
                .remove(0)
                .expect("slot 0 drafts");
            assert_eq!(full.ids, full_picks, "the full head's device drafts are the host's");
            m.set_mtp_vocab(VocabMode::Restricted);
            let restricted = m
                .mtp_draft_keyed(&[DraftAsk { slot: 1, tokens: &b.tokens, depth }], &[Some(s)])
                .unwrap()
                .remove(0)
                .expect("slot 1 drafts");
            let sub = m.mtp_sub_head().unwrap();
            // a chain feeds its draft back: past the first difference the two chains are legitimately different, so compare while the drafts agree
            for k in 0..depth {
                assert_eq!(
                    sub.coverage(&[restricted.ids[k]]),
                    1.0,
                    "round {round}: restricted draft {k} ({}) is outside the set",
                    restricted.ids[k]
                );
                let kept_inside = lists[k].0.iter().take(s.top_k).all(|&id| sub.coverage(&[id]) == 1.0);
                if !kept_inside {
                    outside += 1;
                    break;
                }
                inside += 1;
                differ_from_argmax += usize::from(full.ids[k] != lists[k].0[0]);
                assert_eq!(
                    restricted.ids[k], full.ids[k],
                    "round {round} depth {k}: every candidate the sampler keeps is inside the set, yet the restricted head drafted {} and the full head {}",
                    restricted.ids[k], full.ids[k]
                );
                if restricted.ids[k] != full.ids[k] {
                    break;
                }
            }
            // advance both slots by the same one-row step
            let mut refs: Vec<&mut Stream> = vec![&mut a];
            advance(&mut m, &mut refs);
            let t = a.tokens[a.tokens.len() - 1];
            // slot 1 follows slot 0's text: the same window (slot 1's rows), committed
            let w = RowBatch::from_requests(&[RowRequest {
                slot: 1,
                tokens: vec![*b.tokens.last().unwrap()],
                first_position: (b.tokens.len() - 1) as u64,
            }]);
            m.window_candidates(&w, 1).unwrap();
            m.commit(1, 1);
            b.tokens.push(t);
            assert_eq!(a.tokens, b.tokens);
        }
    }
    eprintln!("restricted keyed head: {inside} drafts compared inside the set ({differ_from_argmax} not the argmax), {outside} chains ended at a draft whose kept candidates leave it");
    assert!(inside >= 40, "the comparison saw too few drafts inside the set ({inside})");
    assert!(differ_from_argmax >= 6, "the noise changed only {differ_from_argmax} of the compared drafts");
}

#[test]
#[ignore]
fn a_greedy_draft_is_the_argmax_draft_bit_for_bit_keyed_or_not() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_vocab(VocabMode::Full);
    let prompt = ids_of(&oracle, "code_700")[..320].to_vec();
    let greedy = Sampling { seed: 5, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 };
    // a request whose top_k the call cannot fetch is not keyed either
    let too_wide = Sampling { seed: 5, temperature: 1.0, top_k: 0, top_p: 0.95, min_p: 0.0 };
    for (case, s) in [("greedy", greedy), ("top_k 0", too_wide)] {
        let (mut a, mut b) = (start(&mut m, 0, &prompt, greedy), start(&mut m, 1, &prompt, greedy));
        for round in 0..16usize {
            let depth = 1 + round % 4;
            let plain = m.mtp_draft(&[DraftAsk { slot: 0, tokens: &a.tokens, depth }]).unwrap().remove(0).expect("slot 0 drafts");
            let keyed = m
                .mtp_draft_keyed(&[DraftAsk { slot: 1, tokens: &b.tokens, depth }], &[Some(s)])
                .unwrap()
                .remove(0)
                .expect("slot 1 drafts");
            assert_eq!(plain.ids, keyed.ids, "{case}, round {round}: the drafts differ");
            assert_eq!(
                plain.vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                keyed.vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{case}: the logits"
            );
            let mut refs: Vec<&mut Stream> = vec![&mut a];
            advance(&mut m, &mut refs);
            let t = a.tokens[a.tokens.len() - 1];
            let w = RowBatch::from_requests(&[RowRequest {
                slot: 1,
                tokens: vec![*b.tokens.last().unwrap()],
                first_position: (b.tokens.len() - 1) as u64,
            }]);
            m.window_candidates(&w, 1).unwrap();
            m.commit(1, 1);
            b.tokens.push(t);
        }
    }
}

/// The configuration the server runs: the restricted head, and a call in which a greedy stream shares the launches with a sampled one (half the streams of a
/// server are greedy). The greedy row's ids and logit bits must be those of the plain call (the old top-1 launches, one row), and the sampled row's drafts
/// the host's keyed pick over the restricted head's own candidates.
#[test]
#[ignore]
fn a_greedy_row_in_a_mixed_call_is_the_argmax_draft_through_the_restricted_head_and_the_sampled_row_the_hosts_pick() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    if m.mtp_sub_head().is_none() {
        m.mtp_restrict_vocab(&vocab_ranges(512, 154_880)).unwrap();
    }
    m.set_mtp_vocab(VocabMode::Restricted);
    let prompt = ids_of(&oracle, "chat_300")[..180].to_vec();
    let greedy = Sampling { seed: 5, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 };
    let sampled = Sampling { seed: 31, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 };
    // slot 0: the plain call; slot 1: the same greedy text in the mixed call; slot 2: the sampled stream beside it
    let (mut a, mut b, mut c) =
        (start(&mut m, 0, &prompt, greedy), start(&mut m, 1, &prompt, greedy), start(&mut m, 2, &prompt, sampled));
    let (mut compared, mut differ) = (0usize, 0usize);
    for round in 0..24usize {
        let depth = 1 + round % 4;
        let (host_picks, lists) = host_chain(&mut m, &c, depth, true);
        let plain = m.mtp_draft(&[DraftAsk { slot: 0, tokens: &a.tokens, depth }]).unwrap().remove(0).expect("slot 0 drafts");
        let mixed = m
            .mtp_draft_keyed(
                &[DraftAsk { slot: 1, tokens: &b.tokens, depth }, DraftAsk { slot: 2, tokens: &c.tokens, depth }],
                &[Some(greedy), Some(sampled)],
            )
            .unwrap();
        let (g, s) = (mixed[0].as_ref().expect("slot 1 drafts"), mixed[1].as_ref().expect("slot 2 drafts"));
        assert_eq!(plain.ids, g.ids, "round {round}: the greedy row's drafts differ from the plain call's");
        assert_eq!(
            plain.vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            g.vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "round {round}: the greedy row's logits"
        );
        assert_eq!(s.ids, host_picks, "round {round} depth {depth}: the sampled row's drafts differ from the host's pick over the restricted head's candidates");
        compared += depth;
        differ += s.ids.iter().zip(&lists).filter(|(d, l)| **d != l.0[0]).count();
        let mut refs: Vec<&mut Stream> = vec![&mut a, &mut b, &mut c];
        advance(&mut m, &mut refs);
        assert_eq!(a.tokens, b.tokens, "the two greedy streams are one text");
    }
    eprintln!(
        "mixed call through the restricted head: {compared} drafts compared, {differ} sampled drafts are not the head's argmax"
    );
    assert!(differ >= 5, "the noise changed only {differ} of the sampled drafts");
}

/// The drafter proposes, a window of `1 + drafts` rows decides, the accepted prefix is committed. Returns the generated tokens and (rounds, drafted, accepted).
fn spec_run(
    m: &mut GlmModel,
    slot: usize,
    prompt: &[u32],
    s: Sampling,
    n_new: usize,
    depth: usize,
    pick: DraftPick,
) -> (Vec<u32>, (usize, usize, usize)) {
    let mut st = start(m, slot, prompt, s);
    let (mut rounds, mut drafted, mut accepted) = (0, 0, 0);
    while st.tokens.len() - prompt.len() < n_new {
        let left = n_new - (st.tokens.len() - prompt.len());
        let max = depth.min(left - 1);
        let drafts: Vec<u32> = if max > 0 {
            let keyed = [pick.is_keyed().then_some(st.s)];
            match m.mtp_draft_keyed(&[DraftAsk { slot, tokens: &st.tokens, depth: max }], &keyed).unwrap().remove(0) {
                Some(d) => d.ids,
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };
        let mut window = vec![*st.tokens.last().unwrap()];
        window.extend_from_slice(&drafts);
        let b =
            RowBatch::from_requests(&[RowRequest { slot, tokens: window.clone(), first_position: (st.tokens.len() - 1) as u64 }]);
        let c = m.window_candidates(&b, CAND).unwrap();
        // row j is sampled at position len + j, as the engine does
        let samples: Vec<u32> = (0..window.len())
            .map(|j| sample(&c.ids[j * CAND..(j + 1) * CAND], &c.vals[j * CAND..(j + 1) * CAND], st.tokens.len() + j, &st.s))
            .collect();
        let acc = drafts.iter().zip(&samples).take_while(|(d, s)| d == s).count();
        m.commit(slot, acc + 1);
        st.tokens.extend_from_slice(&samples[..=acc]);
        rounds += 1;
        drafted += drafts.len();
        accepted += acc;
    }
    st.tokens.truncate(prompt.len() + n_new);
    (st.tokens[prompt.len()..].to_vec(), (rounds, drafted, accepted))
}

fn serial_run(m: &mut GlmModel, slot: usize, prompt: &[u32], s: Sampling, n_new: usize) -> Vec<u32> {
    let mut st = start(m, slot, prompt, s);
    while st.tokens.len() - prompt.len() < n_new {
        let mut refs: Vec<&mut Stream> = vec![&mut st];
        advance(m, &mut refs);
    }
    st.tokens[prompt.len()..].to_vec()
}

#[test]
#[ignore]
fn keyed_drafting_in_the_loop_equals_serial() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_vocab(VocabMode::Full);
    let prompts = [("chat_300", ids_of(&oracle, "chat_300")), ("code_700[..400]", ids_of(&oracle, "code_700")[..400].to_vec())];
    let n_new = 96;
    let mut tally = [[0usize; 3]; 2];
    for (name, prompt) in &prompts {
        for (t, seed) in [(0.7f64, 1u64), (1.0, 2), (1.0, 3)] {
            let s = Sampling { seed, temperature: t, top_k: 20, top_p: 0.95, min_p: 0.0 };
            let serial = serial_run(&mut m, 0, prompt, s, n_new);
            for depth in 1..=4usize {
                for (pi, pick) in [DraftPick::Argmax, DraftPick::Keyed].into_iter().enumerate() {
                    let (tokens, (rounds, drafted, accepted)) = spec_run(&mut m, 1, prompt, s, n_new, depth, pick);
                    assert_eq!(
                        tokens, serial,
                        "{name} T {t} seed {seed} K={depth} {pick:?}: the committed text differs from one-row decoding"
                    );
                    if depth == 3 {
                        tally[pi][0] += rounds;
                        tally[pi][1] += drafted;
                        tally[pi][2] += accepted;
                    }
                }
            }
        }
    }
    let rate = |t: [usize; 3]| (t[2] as f64 / t[1] as f64, (n_new * 6) as f64 / t[0] as f64);
    let (a, k) = (rate(tally[0]), rate(tally[1]));
    eprintln!("K = 3, T 0.7 and 1.0, {} tokens: argmax accepted {:.3} ({:.2} tokens a round), keyed accepted {:.3} ({:.2} tokens a round)", n_new * 6, a.0, a.1, k.0, k.1);
}
