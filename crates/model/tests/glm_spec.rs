//! GPU tests of speculation with the pack's NextN layer on the whole GLM-5.3-Flash model (90.5 GiB resident). All `#[ignore]` and env-gated
//! (REXL3_GLM_DIR: the pack; REXL3_GLM_ORACLE_DIR: the oracle's dumps, for prompt token ids); without them a test prints "not run". One model per
//! process: run with `--test-threads=1` under the GPU lock (`nrun NAME --gpu -- ...`).
//!
//! * `drafter_in_the_loop_equals_serial`: the drafter proposes, a verify window of 1 + K rows decides, the accepted prefix is committed (partial
//!   accepts, rollback of the trunk's KDA state by replay, the draft state advancing only with accepted rows): the greedy token stream, the KDA
//!   state of all 34 layers, the MLA latents and the DSA pools equal those of plain one-row decoding, bit for bit, for K = 1..4, on prompts that end
//!   short of the DSA boundary (2,047/2,048) and cross it.

mod glm_support;

use gb_engine::RowRequest;
use gb_model::families::glm::alloc::{self, AllocParams, AllocPlan};
use gb_model::families::glm::model::{Capture, DraftAsk, GlmModel, MtpMutant};
use gb_model::families::glm::npy;
use gb_model::rows::RowBatch;
use glm_support::*;
use std::path::{Path, PathBuf};

const SLOTS: usize = 8;
const KV_BLOCKS: usize = 80; // 20,480 tokens
const MAX_CONTEXT: usize = 8192;

/// The shared model; with `REXL3_MTP_MUTANT=tap_before_norm|no_shift|advance_all_rows|absorb_last_only` the drafter runs with that deliberate bug (the
/// mutation check: `drafter_in_the_loop_equals_serial` must then FAIL on its acceptance floors, while the committed tokens stay the serial ones).
fn model() -> Option<std::sync::MutexGuard<'static, GlmModel>> {
    let mut g = shared_model_mtp(SLOTS, KV_BLOCKS, MAX_CONTEXT)?;
    if let Ok(v) = std::env::var("REXL3_MTP_MUTANT") {
        let mutant = match v.as_str() {
            "tap_before_norm" => MtpMutant::TapBeforeNorm,
            "no_shift" => MtpMutant::NoShift,
            "advance_all_rows" => MtpMutant::AdvanceAllRows,
            "absorb_last_only" => MtpMutant::AbsorbLastOnly,
            "wrong_drafts" => MtpMutant::WrongDrafts,
            "placeholder_images" => MtpMutant::PlaceholderImages,
            "restricted_no_map" => MtpMutant::RestrictedNoMap,
            other => panic!("REXL3_MTP_MUTANT={other} is not a mutant"),
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

/// Prefill `prompt` into `slot` (128-row chunks, as the backend does) and return the greedy token after it.
fn prefill(m: &mut GlmModel, slot: usize, prompt: &[u32]) -> u32 {
    m.begin_slot(slot).unwrap();
    let mut at = 0;
    let mut first = None;
    while at < prompt.len() {
        let end = (at + m.chunk_rows()).min(prompt.len());
        let want = (end == prompt.len()).then_some(1);
        if let Some(c) = m.prefill_chunk(slot, at, &prompt[at..end], want).unwrap() {
            first = Some(c.ids[0] as u32);
        }
        at = end;
    }
    first.unwrap()
}

/// One verify window of `slot`: `tokens` (the pending token and the drafts) at positions starting at `first_position`; the greedy token of every row.
fn window_argmax(m: &mut GlmModel, slot: usize, tokens: &[u32], first_position: usize) -> Vec<u32> {
    let b = RowBatch::from_requests(&[RowRequest { slot, tokens: tokens.to_vec(), first_position: first_position as u64 }]);
    let c = m.window_candidates(&b, 1).unwrap();
    (0..tokens.len()).map(|r| c.ids[r] as u32).collect()
}

/// Plain decoding: one row per step.
fn serial_run(m: &mut GlmModel, slot: usize, prompt: &[u32], n_new: usize) -> Vec<u32> {
    let mut tokens = prompt.to_vec();
    tokens.push(prefill(m, slot, prompt));
    while tokens.len() - prompt.len() < n_new {
        let last = *tokens.last().unwrap();
        let next = window_argmax(m, slot, &[last], tokens.len() - 1)[0];
        m.commit(slot, 1);
        tokens.push(next);
    }
    tokens
}

#[derive(Default, Debug)]
struct Run {
    rounds: usize,
    drafted: usize,
    accepted: usize,
    /// rounds in which the drafter had nothing to offer
    undrafted: usize,
}

/// Speculative decoding with the NextN drafter at `depth`: draft, verify a window of `1 + drafts` rows, commit the accepted prefix.
fn spec_run(m: &mut GlmModel, slot: usize, prompt: &[u32], n_new: usize, depth: usize) -> (Vec<u32>, Run) {
    let mut run = Run::default();
    let mut tokens = prompt.to_vec();
    tokens.push(prefill(m, slot, prompt));
    while tokens.len() - prompt.len() < n_new {
        let left = n_new - (tokens.len() - prompt.len());
        let max = depth.min(left - 1);
        let drafts: Vec<u32> = if max > 0 {
            match m.mtp_draft(&[DraftAsk { slot, tokens: &tokens, depth: max }]).unwrap().remove(0) {
                Some(d) => d.ids,
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };
        if drafts.is_empty() {
            run.undrafted += 1;
        }
        let mut window = vec![*tokens.last().unwrap()];
        window.extend_from_slice(&drafts);
        let samples = window_argmax(m, slot, &window, tokens.len() - 1);
        let acc = drafts.iter().zip(&samples).take_while(|(d, s)| d == s).count();
        m.commit(slot, acc + 1);
        tokens.extend_from_slice(&samples[..=acc]);
        run.rounds += 1;
        run.drafted += drafts.len();
        run.accepted += acc;
    }
    assert_eq!(tokens.len() - prompt.len(), n_new, "the engine's cap leaves no overshoot");
    (tokens, run)
}

/// The trunk state of two slots that have fed the same tokens, bit for bit: KDA state (the committed buffer and the conv history) of all 34 layers,
/// the MLA latent rows and DSA pool rows of all 11 layers.
fn assert_same_state(m: &mut GlmModel, a: usize, b: usize, fed: usize, what: &str) {
    for l in 0..45 {
        let is_mla = l % 4 == 3;
        if !is_mla {
            // the committed state (the buffer the parity word names) and the conv history: the double buffer's other half and the parity
            // bit legitimately differ with the accept pattern
            let (sa, ca) = m.kda_state(l, a).unwrap();
            let (sb, cb) = m.kda_state(l, b).unwrap();
            assert!(
                ca == cb && sa.iter().zip(&sb).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{what}: KDA layer {l} state differs"
            );
        } else {
            let (la, lb) = (m.latent_rows(l, a, fed).unwrap(), m.latent_rows(l, b, fed).unwrap());
            assert!(la == lb, "{what}: MLA layer {l} latent rows differ");
            let (pa, pb) = (m.pool_rows(l, a, fed / 4).unwrap(), m.pool_rows(l, b, fed / 4).unwrap());
            assert!(pa == pb, "{what}: MLA layer {l} pool rows differ");
        }
    }
}

/// The tap point: the drafter's hidden input is the trunk's POST-final-norm row (the mean of the four streams through `model.norm`, fp16), exactly the
/// rows the head would take its logits from. Run a window and a prompt chunk with the capture on and compare the tap buffer with the captured final norm,
/// bit for bit. (An MTP fed the pre-norm mean still drafts, a little worse: acceptance alone does not notice, this does.)
fn assert_the_tap_is_the_final_norm_output(m: &mut GlmModel, prompt: &[u32]) {
    let h = m.cfg.hidden;
    let first = prefill(m, 3, &prompt[..40]);
    m.capture = Some(Capture::default());
    window_argmax(m, 3, &[first, 7, 11], 40);
    let cap = m.capture.take().unwrap();
    let tap = m.mtp_tap_rows(3).unwrap();
    assert_eq!(cap.final_norm.len(), 3 * h);
    assert!(tap == cap.final_norm, "the drafter's tap is not the final norm's output (the head's input) in a window");
    // a prompt chunk that wants no logits taps the same way (the tap plan runs the head's mean and norm, no lm_head)
    m.begin_slot(3).unwrap();
    m.capture = Some(Capture::default());
    m.prefill_chunk(3, 0, &prompt[..40], None).unwrap();
    let cap = m.capture.take().unwrap();
    let tap = m.mtp_tap_rows(40).unwrap();
    assert!(tap == cap.final_norm, "the drafter's tap is not the final norm's output in a prompt chunk");
    m.end_slot(3);
}

#[test]
#[ignore]
fn drafter_in_the_loop_equals_serial() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    assert!(m.has_mtp(), "the model must carry the NextN layer");
    assert_the_tap_is_the_final_norm_output(&mut m, &ids_of(&oracle, "chat_300"));
    // (prompt, ids range, new tokens): a short chat prompt, a code prompt, and the long prompt cut so that its decoding crosses 2,048 tokens
    let long = ids_of(&oracle, "long_2300");
    let cases: Vec<(&str, Vec<u32>, usize)> = vec![
        ("chat_300", ids_of(&oracle, "chat_300"), 64),
        ("code_700", ids_of(&oracle, "code_700"), 64),
        ("long_2300 cut at 2,040 (crosses the DSA boundary)", long[..2040].to_vec(), 40),
        ("long_2300 cut at 2,044", long[..2044].to_vec(), 24),
        ("long_2300 cut at 2,047", long[..2047].to_vec(), 24),
    ];
    // REXL3_SPEC_QUICK: the two short prompts at K = 1..3 only (the mutation runs: each deliberate bug must make this test fail, and fast)
    let quick = std::env::var("REXL3_SPEC_QUICK").is_ok();
    let cases: Vec<_> = if quick { cases.into_iter().take(2).collect() } else { cases };
    for (name, prompt, n_new) in cases {
        let serial = serial_run(&mut m, 0, &prompt, n_new);
        for depth in 1..=if quick { 3 } else { 4 } {
            let (spec, run) = spec_run(&mut m, 1, &prompt, n_new, depth);
            assert_eq!(spec, serial, "{name}: depth {depth}: the drafted token stream differs from the serial one");
            let fed = prompt.len() + n_new - 1;
            assert_same_state(&mut m, 0, 1, fed, &format!("{name}, K={depth}"));
            let tpr = (n_new - 1) as f64 / run.rounds as f64;
            eprintln!(
                "{name}: K={depth}: {} rounds, {} drafted, {} accepted ({tpr:.2} tokens per round), {} rounds without drafts; state equals serial",
                run.rounds, run.drafted, run.accepted, run.undrafted
            );
            // and the drafter is GOOD, not merely harmless: a drafter with a broken tap or state still gives the serial stream (it only proposes)
            // but its drafts stop matching; these floors are about 80% of the lowest tokens per round seen on these cases (1.77 / 2.17 / 2.30 / 2.30)
            let floor = [0.0, 1.5, 1.8, 1.9, 1.9][depth];
            assert!(
                tpr >= floor,
                "{name}: K={depth}: {tpr:.2} tokens per round, below {floor}: the drafter is not drafting well"
            );
        }
    }
}

/// The drafter's head restricted to a set of vocabulary strips (`mtp_restrict_vocab`): wherever the full head's draft lies inside the set the restricted
/// head's draft is the very same id with the very same logit bits (the kept columns are bit-identical copies and the argmax is over a subset), and a
/// restricted draft is always a token of the set. Two slots fed the same tokens hold the same state; one drafts with the full head, the other with the
/// restricted one, and both are advanced with the full head's verified tokens.
#[test]
#[ignore]
fn the_restricted_draft_head_equals_the_full_head_inside_the_set() {
    use gb_model::families::glm::model::{vocab_ranges, VocabMode};
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    m.mtp_restrict_vocab(&vocab_ranges(512, 154_880)).unwrap();
    let prompt = ids_of(&oracle, "chat_300");
    let mut a = prompt.clone();
    a.push(prefill(&mut m, 0, &prompt));
    let mut b = prompt.clone();
    b.push(prefill(&mut m, 1, &prompt));
    assert_eq!(a, b);
    let (mut inside, mut outside) = (0usize, 0usize);
    for round in 0..40usize {
        let depth = 1 + round % 3;
        m.set_mtp_vocab(VocabMode::Full);
        let full = m.mtp_draft(&[DraftAsk { slot: 0, tokens: &a, depth }]).unwrap().remove(0).expect("slot 0 drafts");
        m.set_mtp_vocab(VocabMode::Restricted);
        let restricted = m.mtp_draft(&[DraftAsk { slot: 1, tokens: &b, depth }]).unwrap().remove(0).expect("slot 1 drafts");
        let sub = m.mtp_sub_head().unwrap();
        // a chain feeds a draft back: past the first difference the two chains are legitimately different, so compare while the drafts agree
        for k in 0..depth {
            assert_eq!(
                sub.coverage(&[restricted.ids[k]]),
                1.0,
                "round {round}: restricted draft {k} ({}) is outside the set",
                restricted.ids[k]
            );
            if sub.coverage(&[full.ids[k]]) == 1.0 {
                inside += 1;
                assert_eq!(
                    restricted.ids[k], full.ids[k],
                    "round {round}: draft {k}: the full head chose {} inside the set, the restricted one {}",
                    full.ids[k], restricted.ids[k]
                );
                assert_eq!(
                    restricted.vals[k].to_bits(),
                    full.vals[k].to_bits(),
                    "round {round}: draft {k}: the logit of the chosen token differs"
                );
            } else {
                outside += 1;
                break;
            }
        }
        // verify with the full head's drafts on both slots and commit the same prefix
        let mut window = vec![*a.last().unwrap()];
        window.extend_from_slice(&full.ids);
        let samples = window_argmax(&mut m, 0, &window, a.len() - 1);
        let acc = full.ids.iter().zip(&samples).take_while(|(d, s)| d == s).count();
        // (a commit is applied by the next forward, with the layout of the LAST window: each slot's is recorded before the other's window runs)
        m.commit(0, acc + 1);
        let samples_b = window_argmax(&mut m, 1, &window, b.len() - 1);
        assert_eq!(samples, samples_b, "round {round}: two slots fed the same tokens answer differently");
        m.commit(1, acc + 1);
        a.extend_from_slice(&samples[..=acc]);
        b.extend_from_slice(&samples[..=acc]);
    }
    eprintln!("restricted head: {inside} drafts compared inside the set, {outside} chains ended at a draft outside it");
    assert!(inside >= 40, "the comparison saw too few drafts inside the set ({inside})");
}

/// The memory model counts the NextN layer: every `DevBuf` the model owns (its cache, rings, tap store and scratch included) equals the plan,
/// item by item, and the plan with the drafter exceeds the one without by exactly the drafter's items.
#[test]
#[ignore]
fn allocations_equal_the_plan_with_the_nextn_layer() {
    let Some(m) = model() else { return };
    let params = AllocParams {
        slots: SLOTS,
        kv_blocks: KV_BLOCKS,
        max_context: MAX_CONTEXT,
        max_rows: alloc::MAX_ROWS,
        window_rows: alloc::WINDOW_ROWS,
        mtp: true,
    };
    let plan = AllocPlan::new_chunked(&m.cfg, &params, m.chunk_rows()).unwrap();
    m.check_allocations(&plan).unwrap();
    let total: u64 = m.allocations().iter().map(|a| a.1).sum();
    assert_eq!(total, plan.total());
    let without = AllocPlan::new_chunked(&m.cfg, &AllocParams { mtp: false, ..params }, m.chunk_rows()).unwrap();
    eprintln!(
        "the model's allocations ({:.3} GiB beyond the weights) equal the plan item by item; the NextN drafter accounts for {:.1} MiB of them",
        gib(total),
        (total - without.total()) as f64 / (1 << 20) as f64
    );
    for (n, b) in m.allocations() {
        if n.starts_with("mtp_") || matches!(n, "mla_latent" | "dsa_rings" | "dsa_pools") {
            eprintln!("  {n:14} {b:>14} B");
        }
    }
}

/// The cost of a draft call by (streams, pending rows per stream, depth): wall milliseconds of `mtp_draft` (absorb, chain, one sync) on a model
/// with `S` sequences of about 300 tokens each. The numbers the arbiter's `MtpCost` is fitted from; printed as `COST streams rows depth ms`
/// (median of 7, after a warm-up). Also the trunk window of the same rows for reference (`WINDOW streams rows ms`). `REXL3_COST_KEYED=1` prices the keyed pick.
#[test]
#[ignore]
fn mtp_draft_costs() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    let code = ids_of(&oracle, "code_700");
    let prompts: Vec<Vec<u32>> = (0..SLOTS).map(|i| code[i * 40..i * 40 + 300].to_vec()).collect();
    let filler: Vec<u32> = ids_of(&oracle, "chat_300");
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    // REXL3_COST_VOCAB=512 prices the drafter with the restricted head of that many prefix strips (`MTP_COST_RESTRICTED`), unset with the full head
    if let Ok(v) = std::env::var("REXL3_COST_VOCAB") {
        use gb_model::families::glm::model::{vocab_ranges, VocabMode};
        m.mtp_restrict_vocab(&vocab_ranges(v.parse().unwrap(), 154_880)).unwrap();
        m.set_mtp_vocab(VocabMode::Restricted);
    }
    // SLOTS is 8 in this binary: streams 1, 2, 4, 8
    for streams in [1usize, 2, 4, 8] {
        let mut seqs: Vec<Vec<u32>> = Vec::new();
        for (slot, p) in prompts.iter().enumerate().take(streams) {
            let mut t = p.clone();
            t.push(prefill(&mut m, slot, p));
            seqs.push(t);
        }
        for rows in [1usize, 2, 3, 5, 8] {
            for depth in [0usize, 1, 2, 3, 4] {
                let mut times = Vec::new();
                let mut wtimes = Vec::new();
                for trial in 0..8 {
                    // a window of `rows` rows per stream, all kept: the pending rows of the next draft call
                    let reqs: Vec<RowRequest> = (0..streams)
                        .map(|i| {
                            let last = *seqs[i].last().unwrap();
                            let mut toks = vec![last];
                            toks.extend((1..rows).map(|j| filler[(trial * 7 + i * 13 + j) % filler.len()]));
                            RowRequest { slot: i, tokens: toks, first_position: (seqs[i].len() - 1) as u64 }
                        })
                        .collect();
                    let b = RowBatch::from_requests(&reqs);
                    let t = std::time::Instant::now();
                    m.window_candidates(&b, 1).unwrap();
                    let w = t.elapsed().as_secs_f64() * 1e3;
                    for (i, r) in reqs.iter().enumerate() {
                        m.commit(i, rows);
                        // the rows' next tokens: the window's own tokens after the first, then one more
                        seqs[i].extend_from_slice(&r.tokens[1..]);
                        seqs[i].push(filler[(trial * 3 + i) % filler.len()]);
                    }
                    let asks: Vec<DraftAsk> = (0..streams).map(|i| DraftAsk { slot: i, tokens: &seqs[i], depth }).collect();
                    // REXL3_COST_KEYED=1 prices the call with the keyed pick (every stream sampled at T 1.0, top_k 20): the top-20 and the pick of every depth
                    let keyed: Vec<Option<gb_sampling::Sampling>> = if std::env::var("REXL3_COST_KEYED").is_ok() {
                        (0..streams)
                            .map(|i| {
                                Some(gb_sampling::Sampling {
                                    seed: 70 + i as u64,
                                    temperature: 1.0,
                                    top_k: 20,
                                    top_p: 0.95,
                                    min_p: 0.0,
                                })
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let t = std::time::Instant::now();
                    let out = m.mtp_draft_keyed(&asks, &keyed).unwrap();
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    assert!(out.iter().all(|o| o.is_some()), "every stream drafts");
                    if trial > 0 {
                        times.push(ms);
                        wtimes.push(w);
                    }
                }
                println!("COST {streams} {rows} {depth} {:.3}", median(&mut times));
                if depth == 0 {
                    println!("WINDOW {streams} {rows} {:.3}", median(&mut wtimes));
                }
            }
        }
    }
}

/// The drafter's invariants, each broken on purpose (`MtpMutant`): the committed tokens must STILL be the serial ones (a drafter only proposes: nothing
/// it does can change an output byte), but its drafts must stop paying, and the test must notice. Baseline: code_700, K = 3.
#[test]
#[ignore]
fn drafter_mutants_are_noticed_by_their_acceptance() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    let prompt = ids_of(&oracle, "code_700");
    let n_new = 64;
    let serial = serial_run(&mut m, 0, &prompt, n_new);
    let rate = |m: &mut GlmModel, label: &str| -> f64 {
        let (toks, run) = spec_run(m, 1, &prompt, n_new, 3);
        assert_eq!(toks, serial, "{label}: the committed tokens must be the serial ones whatever the drafter does");
        let tpr = (n_new - 1) as f64 / run.rounds as f64;
        eprintln!(
            "{label}: {} rounds, {} drafted, {} accepted, {tpr:.2} tokens per round",
            run.rounds, run.drafted, run.accepted
        );
        tpr
    };
    let base = rate(&mut m, "unmutated");
    assert!(base >= 2.5, "the baseline must be a working drafter: {base:.2}");
    // (mutant, the tokens per round it must stay under, what the bug is). The other mutants (NoShift, AbsorbLastOnly) leave acceptance where it is (the
    // drafts hardly depend on a few cache entries): `the_drafters_cache_equals_the_teacher_forced_one` pins them exactly
    let cases = [
        (
            MtpMutant::TapBeforeNorm,
            base * 0.97,
            "the tap is the mean of the streams without the final norm (hnorm hides most of it: a small drop)",
        ),
        (MtpMutant::AdvanceAllRows, base * 0.7, "the draft state advances on rejected rows too"),
        (MtpMutant::WrongDrafts, 1.2, "every draft is the wrong id (the arbiter's fall-back test uses this one)"),
    ];
    let mut missed = Vec::new();
    for (mutant, under, what) in cases {
        m.set_mtp_mutant(Some(mutant));
        let r = rate(&mut m, &format!("{mutant:?} ({what})"));
        m.set_mtp_mutant(None);
        if r >= under {
            missed.push(format!("{mutant:?} ({what}): {r:.2} tokens per round against {base:.2}"));
        }
    }
    assert!(missed.is_empty(), "not noticed by their acceptance: {missed:?}");
    // and the drafter is whole again once the bug is gone
    let again = rate(&mut m, "unmutated again");
    assert!((again - base).abs() < 1e-9, "the baseline is deterministic: {again} vs {base}");
}

/// Where a speculative round goes at R = 3 rows (the pending token and two drafts) and K = 2, one stream at a context of 2,300 tokens: the verify
/// window's time by layer class (CUDA events around every op), and the draft call's by phase. Printed as `PROFILE ...` lines.
#[test]
#[ignore]
fn round_profile_at_three_rows() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    let prompt = ids_of(&oracle, "long_2300");
    let filler = ids_of(&oracle, "chat_300");
    let mut tokens = prompt.clone();
    tokens.push(prefill(&mut m, 0, &prompt));
    let rounds = 16;
    // warm up: two rounds, then the measured ones
    let round = |m: &mut GlmModel, tokens: &mut Vec<u32>, i: usize| -> (f64, f64) {
        let last = *tokens.last().unwrap();
        let toks = vec![last, filler[(i * 5) % filler.len()], filler[(i * 5 + 1) % filler.len()]];
        let b =
            RowBatch::from_requests(&[RowRequest { slot: 0, tokens: toks.clone(), first_position: (tokens.len() - 1) as u64 }]);
        let t = std::time::Instant::now();
        m.window_candidates(&b, 1).unwrap();
        let w = t.elapsed().as_secs_f64() * 1e3;
        m.commit(0, 3);
        tokens.extend_from_slice(&toks[1..]);
        tokens.push(filler[(i * 3) % filler.len()]);
        let t = std::time::Instant::now();
        m.mtp_draft(&[DraftAsk { slot: 0, tokens: tokens.as_slice(), depth: 2 }]).unwrap();
        (w, t.elapsed().as_secs_f64() * 1e3)
    };
    for i in 0..2 {
        round(&mut m, &mut tokens, i);
    }
    m.profile = Some(Default::default());
    m.set_mtp_profile(true);
    let (mut win, mut dr) = (0.0, 0.0);
    for i in 0..rounds {
        let (w, d) = round(&mut m, &mut tokens, 2 + i);
        win += w;
        dr += d;
    }
    let p = m.profile.take().unwrap();
    let mp = m.take_mtp_profile();
    m.set_mtp_profile(false);
    println!(
        "PROFILE context {} tokens, window of 3 rows: {:.2} ms wall; draft call (absorb 3 rows, K=2): {:.2} ms wall",
        tokens.len(),
        win / rounds as f64,
        dr / rounds as f64
    );
    let total: f64 = p.ms.values().sum::<f64>() / p.forwards as f64;
    for (class, ms) in &p.ms {
        let per = ms / p.forwards as f64;
        println!("PROFILE window {class:34} {per:7.2} ms {:5.1}%", 100.0 * per / total);
    }
    println!("PROFILE window total (events) {total:.2} ms");
    let mtotal: f64 = mp.iter().map(|x| x.1).sum::<f64>() / rounds as f64;
    for (phase, ms) in &mp {
        let per = ms / rounds as f64;
        println!("PROFILE draft {phase:46} {per:7.2} ms {:5.1}%", 100.0 * per / mtotal);
    }
    println!("PROFILE draft total (events) {mtotal:.2} ms");
}

/// Acceptance past the dense window: the DSA indexer selects 2,048 positions per row once the context passes 2,048 tokens, and the NextN layer scores
/// with its OWN indexer (the oracle's choice; `index_share_for_mtp_iteration`, reusing the trunk's last selection, is not built). If that cost drafts
/// their quality, tokens per round would fall with context: this measures them at 2,278 tokens and at 3,463, greedy, 64 new tokens, K = 1..4, and the
/// committed tokens must stay the serial ones.
#[test]
#[ignore]
fn acceptance_past_the_dense_window() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    let mut long = ids_of(&oracle, "long_2300");
    let mut longer = long.clone();
    for name in ["code_700", "chat_300", "think_prefix", "short_20"] {
        longer.extend(ids_of(&oracle, name));
    }
    long.truncate(2278);
    for (name, prompt) in [("long_2300", long), ("five prompts joined", longer)] {
        let serial = serial_run(&mut m, 0, &prompt, 64);
        for depth in 1..=4 {
            let (spec, run) = spec_run(&mut m, 1, &prompt, 64, depth);
            assert_eq!(spec, serial, "{name}: K={depth}");
            println!(
                "PAST {name} ({} tokens): K={depth}: {:.2} tokens per round ({} drafted, {} accepted in {} rounds)",
                prompt.len(),
                63.0 / run.rounds as f64,
                run.drafted,
                run.accepted,
                run.rounds
            );
        }
    }
}

/// The drafter's state, exactly. After a speculative run (partial accepts, K = 3) the MTP cache entries below the last absorbed position must be, bit for
/// bit, those of TEACHER FORCING: the committed tokens fed through the layer with the target's own hidden rows (the tap of the position before each
/// token, zeros before the first), one window, no speculation. That one equality pins together: only accepted rows advance the draft state, the speculative
/// entries a chain writes are rewritten from the target's true rows (the rollback), the tap point, the shift by one position, the carry across prompt
/// chunks, and the absorb of several pending rows in one window. Each of those, broken (`MtpMutant`), makes it fail while the committed tokens stay serial.
#[test]
#[ignore]
fn the_drafters_cache_equals_the_teacher_forced_one() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    let prompt = ids_of(&oracle, "chat_300"); // 288 tokens: three prefill chunks, so the carry between chunks is used
    let n_new = 48;
    // the serial run, collecting the target's final-normed rows (the taps) of every position it feeds
    m.begin_slot(0).unwrap();
    let mut taps: Vec<u16> = Vec::new();
    let mut at = 0;
    let mut tokens = prompt.clone();
    while at < prompt.len() {
        let end = (at + m.chunk_rows()).min(prompt.len());
        let want = (end == prompt.len()).then_some(1);
        if let Some(c) = m.prefill_chunk(0, at, &prompt[at..end], want).unwrap() {
            tokens.push(c.ids[0] as u32);
        }
        taps.extend(m.mtp_tap_rows(end - at).unwrap());
        at = end;
    }
    while tokens.len() - prompt.len() < n_new {
        let last = *tokens.last().unwrap();
        let next = window_argmax(&mut m, 0, &[last], tokens.len() - 1)[0];
        taps.extend(m.mtp_tap_rows(1).unwrap());
        m.commit(0, 1);
        tokens.push(next);
    }
    let h = m.cfg.hidden;
    assert_eq!(taps.len(), (tokens.len() - 1) * h, "a tap row for every position the serial run fed");
    // the speculative run in slot 1, then the entries it has built
    let (spec_tokens, run) = spec_run(&mut m, 1, &prompt, n_new, 3);
    assert_eq!(spec_tokens, tokens, "the committed tokens are the serial ones");
    let built = m.mtp_book(1).expect("the model drafts").first;
    assert!(built > prompt.len() + n_new / 2, "the draft state must have advanced with the committed rows: built {built}");
    // teacher forcing in slot 2: entry m takes token t_m and the tap of position m - 1 (zeros for m = 0)
    let mut hidden = vec![0u16; built * h];
    for mm in 1..built {
        hidden[mm * h..(mm + 1) * h].copy_from_slice(&taps[(mm - 1) * h..mm * h]);
    }
    m.begin_slot(2).unwrap();
    m.mtp_debug_forward(2, 0, &tokens[..built], &hidden, 1).unwrap();
    let (lat_spec, lat_ref) = (m.mtp_latent_rows(1, built).unwrap(), m.mtp_latent_rows(2, built).unwrap());
    let first_diff = (0..built).find(|&r| lat_spec[r * 512..(r + 1) * 512] != lat_ref[r * 512..(r + 1) * 512]);
    assert!(
        first_diff.is_none(),
        "the drafter's latent cache differs from the teacher-forced one from position {first_diff:?} of {built}"
    );
    let (pool_spec, pool_ref) = (m.mtp_pool_rows(1, built / 4).unwrap(), m.mtp_pool_rows(2, built / 4).unwrap());
    assert!(pool_spec == pool_ref, "the drafter's DSA pooled keys differ from the teacher-forced ones");
    eprintln!(
        "the drafter's cache after {} rounds of K=3 ({} drafted, {} accepted) equals teacher forcing over {built} positions bit for bit (latent rows and pooled keys)",
        run.rounds, run.drafted, run.accepted
    );
}
