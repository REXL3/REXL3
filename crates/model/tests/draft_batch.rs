#![allow(clippy::needless_range_loop)]
//! A sequence's drafts are the same bits whether its draft round runs alone or batched with other sequences, and
//! whether its committed context rows were absorbed alone or together with other sequences' rows.
//!
//!     REXL3_QWEN_DIR=... REXL3_DRAFT_DIR=... cargo test --release -p gb-model --test draft_batch -- --ignored --nocapture

mod refs;
use gb_engine::RowRequest;
use gb_model::config::QwenConfig;
use gb_model::draft::Draft;
use gb_model::forward::Model;
use gb_model::pack::Pack;
use gb_model::rows::RowBatch;
use gb_model::weights::Weights;
use gb_sampling::Sampling;
use refs::{have_refs, ref_npy};

const SLOTS: usize = 6;

fn load() -> Option<Model> {
    let d = std::path::PathBuf::from(std::env::var("REXL3_QWEN_DIR").ok()?);
    let dd = std::path::PathBuf::from(std::env::var("REXL3_DRAFT_DIR").ok()?);
    if !gb_cuda::device_available() || !have_refs() {
        return None;
    }
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
    let vocab = cfg.vocab;
    let mut m = Model::new(cfg, w, SLOTS, 128, 4096, 256).unwrap();
    m.draft = Some(Draft::load(&Pack::open(&dd).unwrap(), vocab, SLOTS, 256).unwrap().0);
    Some(m)
}

fn req(slot: usize, tokens: &[u32], first: usize) -> RowRequest {
    RowRequest { slot, tokens: tokens.to_vec(), first_position: first as u64 }
}

/// Slot `slot` holds `prompt` as its prefilled context.
fn start(m: &mut Model, slot: usize, prompt: &[u32]) {
    m.begin_slot(slot).unwrap();
    m.prefill_chunk(slot, 0, prompt, prompt.len(), Some(28), 0).unwrap().unwrap();
}

#[test]
#[ignore]
fn batched_drafts_equal_solo_drafts() {
    let Some(mut m) = load() else { return eprintln!("needs REXL3_QWEN_DIR, REXL3_DRAFT_DIR, a GPU and testdata/ref: skipping") };
    let long: Vec<u32> = ref_npy("long/ids.npy").to_i64().into_iter().map(|x| x as u32).collect();
    // six distinct prompts of different lengths (so different positions per slot)
    let prompts: Vec<Vec<u32>> = (0..SLOTS).map(|s| long[s * 9..long.len() - 3 * s].to_vec()).collect();
    for (s, p) in prompts.iter().enumerate() {
        start(&mut m, s, p);
    }
    let samplings = [
        Sampling { seed: 11, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 },
        Sampling { seed: 12, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 },
        Sampling { seed: 13, temperature: 0.7, top_k: 20, top_p: 0.9, min_p: 0.05 },
        Sampling { seed: 14, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 },
        Sampling { seed: 15, temperature: 1.3, top_k: 20, top_p: 0.5, min_p: 0.0 },
        Sampling { seed: 16, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 },
    ];
    // tokens so far: the prompt plus one more (the anchor, not yet fed)
    let seqs: Vec<Vec<u32>> = prompts
        .iter()
        .enumerate()
        .map(|(s, p)| {
            let mut t = p.clone();
            t.push(1000 + 17 * s as u32);
            t
        })
        .collect();

    let solo: Vec<Vec<u32>> =
        (0..SLOTS).map(|s| m.propose_drafts(s, &seqs[s], &samplings[s]).unwrap().expect("a draft")).collect();
    for d in &solo {
        assert_eq!(d.len(), 7);
    }
    eprintln!("solo drafts: {:?}", &solo[0]);

    let asks: Vec<(usize, &[u32], &Sampling)> = (0..SLOTS).map(|s| (s, &seqs[s][..], &samplings[s])).collect();
    let all = m.propose_drafts_batch(&asks).unwrap();
    for s in 0..SLOTS {
        assert_eq!(all[s].as_ref().unwrap(), &solo[s], "slot {s}: batch of {SLOTS} differs from solo");
    }
    // a subset, in another order
    let order = [4usize, 1, 5];
    let asks: Vec<(usize, &[u32], &Sampling)> = order.iter().map(|&s| (s, &seqs[s][..], &samplings[s])).collect();
    let part = m.propose_drafts_batch(&asks).unwrap();
    for (i, &s) in order.iter().enumerate() {
        assert_eq!(part[i].as_ref().unwrap(), &solo[s], "slot {s}: batch of 3 differs from solo");
    }
}

#[test]
#[ignore]
fn batched_absorb_equals_solo_absorb() {
    let Some(mut m) = load() else { return eprintln!("needs REXL3_QWEN_DIR, REXL3_DRAFT_DIR, a GPU and testdata/ref: skipping") };
    let long: Vec<u32> = ref_npy("long/ids.npy").to_i64().into_iter().map(|x| x as u32).collect();
    let prompt = &long[..];
    let other = &long[40..];
    let p = prompt.len();
    let po = other.len();
    let toks: Vec<u32> = (0..8).map(|i| 2000 + 31 * i).collect();
    let toks_o: Vec<u32> = (0..8).map(|i| 5000 + 13 * i).collect();
    let sampling = Sampling { seed: 3, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 };

    // slots 0 and 1 hold the same prompt, slots 2 and 3 another
    for (s, pr) in [(0, prompt), (1, prompt), (2, other), (3, other)] {
        start(&mut m, s, pr);
    }
    // A (slot 0) commits 5 rows of its window alone: its context rows are absorbed by a one-sequence absorb
    m.window_candidates(&RowBatch::from_requests(&[req(0, &toks, p)]), 28).unwrap();
    m.commit(0, 5);
    // B (slot 1, same tokens) and C (slot 2) commit in one window: their rows are absorbed in one batched pass
    m.window_candidates(&RowBatch::from_requests(&[req(1, &toks, p), req(2, &toks_o, po)]), 28).unwrap();
    m.commit(1, 5);
    m.commit(2, 3);
    // A was flushed alone when the B/C window started; B and C are flushed together by the first propose below
    let seq = |pr: &[u32], t: &[u32], keep: usize| {
        let mut v = pr.to_vec();
        v.extend_from_slice(&t[..=keep]);
        v
    };
    let (sa, sc) = (seq(prompt, &toks, 5), seq(other, &toks_o, 3));
    let da = m.propose_drafts(0, &sa, &sampling).unwrap().unwrap();
    let db = m.propose_drafts(1, &sa, &sampling).unwrap().unwrap();
    let dc = m.propose_drafts(2, &sc, &sampling).unwrap().unwrap();
    assert_eq!(da, db, "slot 1 (absorbed in a batch of 2) differs from slot 0 (absorbed alone)");
    // D (slot 3, same as C) commits alone
    m.window_candidates(&RowBatch::from_requests(&[req(3, &toks_o, po)]), 28).unwrap();
    m.commit(3, 3);
    let dd = m.propose_drafts(3, &sc, &sampling).unwrap().unwrap();
    assert_eq!(dc, dd, "slot 2 (absorbed in a batch of 2) differs from slot 3 (absorbed alone)");
    eprintln!("drafts: {da:?} / {dc:?}");
}

/// Six slots holding distinct prompts, an anchor each, and a mix of keyed and greedy samplings.
fn six(m: &mut Model) -> (Vec<Vec<u32>>, Vec<Sampling>) {
    let long: Vec<u32> = ref_npy("long/ids.npy").to_i64().into_iter().map(|x| x as u32).collect();
    let prompts: Vec<Vec<u32>> = (0..SLOTS).map(|s| long[s * 9..long.len() - 3 * s].to_vec()).collect();
    for (s, p) in prompts.iter().enumerate() {
        start(m, s, p);
    }
    let samplings: Vec<Sampling> = (0..SLOTS)
        .map(|s| Sampling {
            seed: 70 + s as u64,
            temperature: if s % 3 == 1 { 0.0 } else { 1.0 },
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
        })
        .collect();
    let seqs = prompts
        .iter()
        .enumerate()
        .map(|(s, p)| {
            let mut t = p.clone();
            t.push(2000 + 19 * s as u32);
            t
        })
        .collect();
    (seqs, samplings)
}

type Asks<'a> = Vec<(usize, &'a [u32], &'a Sampling, usize)>;

fn asks<'a>(seqs: &'a [Vec<u32>], samplings: &'a [Sampling], wants: &[usize]) -> Asks<'a> {
    (0..SLOTS).map(|s| (s, &seqs[s][..], &samplings[s], wants[s])).collect()
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// A draft round that is asked for fewer picks computes only those (the head, top-k and walk run for fewer rows), and the picks
/// and their scores are the first ones of the full round, whatever else is in the batch.
#[test]
#[ignore]
fn fewer_drafts_are_the_first_drafts_of_the_full_round() {
    let Some(mut m) = load() else { return eprintln!("needs REXL3_QWEN_DIR, REXL3_DRAFT_DIR, a GPU and testdata/ref: skipping") };
    let (seqs, samplings) = six(&mut m);
    let full = m.propose_drafts_batch_want(&asks(&seqs, &samplings, &[7; SLOTS])).unwrap();
    for wants in [[7usize, 1, 3, 2, 7, 4], [1, 1, 1, 1, 1, 1], [3, 0, 7, 0, 5, 6], [2, 7, 7, 7, 7, 7]] {
        let part = m.propose_drafts_batch_want(&asks(&seqs, &samplings, &wants)).unwrap();
        for s in 0..SLOTS {
            let f = full[s].as_ref().unwrap();
            if wants[s] == 0 {
                assert!(part[s].is_none(), "a sequence that wants no drafts is left out of the round");
                continue;
            }
            let p = part[s].as_ref().unwrap();
            let w = wants[s];
            assert_eq!((p.tokens.len(), p.score.len(), p.margin.len()), (w, w, w));
            assert_eq!(p.tokens[..], f.tokens[..w], "slot {s} wants {w} of {wants:?}: picks differ from the full round's");
            assert_eq!(bits(&p.score), bits(&f.score[..w]), "slot {s}: scores differ");
            assert_eq!(bits(&p.margin), bits(&f.margin[..w]), "slot {s}: margins differ");
        }
    }
}

/// A draft head restricted to every column is the full head: same picks, same scores. A restriction that leaves tokens out only
/// ever proposes tokens from the set.
#[test]
#[ignore]
fn a_restricted_draft_vocabulary_is_the_full_one_when_it_keeps_everything_and_proposes_only_its_own_tokens() {
    let Some(mut m) = load() else { return eprintln!("needs REXL3_QWEN_DIR, REXL3_DRAFT_DIR, a GPU and testdata/ref: skipping") };
    let (seqs, samplings) = six(&mut m);
    let full = m.propose_drafts_batch_want(&asks(&seqs, &samplings, &[7; SLOTS])).unwrap();
    let vocab = m.cfg.vocab;
    {
        let Model { draft, w, .. } = &mut m;
        draft.as_mut().unwrap().restrict_vocab(&w.lm_head, &[(0, vocab)]).unwrap();
    }
    let all = m.propose_drafts_batch_want(&asks(&seqs, &samplings, &[7; SLOTS])).unwrap();
    for s in 0..SLOTS {
        let (a, b) = (full[s].as_ref().unwrap(), all[s].as_ref().unwrap());
        assert_eq!(a.tokens, b.tokens, "slot {s}: restricting to every column changed the picks");
        assert_eq!(bits(&a.score), bits(&b.score), "slot {s}: ... or the scores");
        assert_eq!(bits(&a.margin), bits(&b.margin), "slot {s}: ... or the margins");
    }
    // a set that excludes everything the full draft picked
    let picked: Vec<u32> = full.iter().flat_map(|f| f.as_ref().unwrap().tokens.clone()).collect();
    let (lo, hi) = (*picked.iter().min().unwrap() as usize, *picked.iter().max().unwrap() as usize);
    let sets: Vec<(usize, usize)> = if lo >= 128 * 64 {
        vec![(0, 128 * 64)]
    } else if hi + 1 + 128 * 64 <= vocab {
        vec![(hi + 1, vocab)]
    } else {
        return eprintln!("the picks span the vocabulary; cannot build an excluding set: skipping that half");
    };
    {
        let Model { draft, w, .. } = &mut m;
        let d = draft.as_mut().unwrap();
        d.restrict_vocab(&w.lm_head, &sets).unwrap();
        assert_eq!(d.vocab_ranges().unwrap().len(), 1);
    }
    let cut = m.propose_drafts_batch_want(&asks(&seqs, &samplings, &[7; SLOTS])).unwrap();
    let ranges = m.draft.as_ref().unwrap().vocab_ranges().unwrap().to_vec();
    for s in 0..SLOTS {
        for &t in &cut[s].as_ref().unwrap().tokens {
            assert!(
                ranges.iter().any(|&(first, len)| (first..first + len).contains(&(t as usize))),
                "slot {s}: draft {t} is outside {ranges:?}"
            );
        }
    }
}

/// An adaptive draft head runs the restricted head in a round only when every sequence is writing text the restricted set can
/// reach, and the full head otherwise (so the whole round's drafts are exactly the unrestricted ones).
#[test]
#[ignore]
fn an_adaptive_draft_head_uses_the_restricted_head_only_for_text_it_can_reach() {
    let Some(mut m) = load() else { return eprintln!("needs REXL3_QWEN_DIR, REXL3_DRAFT_DIR, a GPU and testdata/ref: skipping") };
    let (seqs, samplings) = six(&mut m);
    let full = m.propose_drafts_batch_want(&asks(&seqs, &samplings, &[7; SLOTS])).unwrap();
    {
        let Model { draft, w, .. } = &mut m;
        draft.as_mut().unwrap().restrict_vocab(&w.lm_head, &gb_model::draft::TF_DRAFT_VOCAB).unwrap();
    }
    let always = m.propose_drafts_batch_want(&asks(&seqs, &samplings, &[7; SLOTS])).unwrap();
    m.draft.as_mut().unwrap().set_adaptive(Some(0.98));
    let before = m.draft.as_ref().unwrap().head_rounds;
    // the prompts are ordinary text: inside the set, so the adaptive head is the restricted head
    let inside = m.propose_drafts_batch_want(&asks(&seqs, &samplings, &[7; SLOTS])).unwrap();
    let after_inside = m.draft.as_ref().unwrap().head_rounds;
    assert_eq!(after_inside, [before[0], before[1] + 1], "an in-set round runs the restricted head");
    for s in 0..SLOTS {
        assert_eq!(inside[s].as_ref().unwrap().tokens, always[s].as_ref().unwrap().tokens, "slot {s}");
    }
    // one sequence's recent tokens move outside the set (the anchor, the last token, stays): the whole round runs the full head
    let mut moved = seqs.clone();
    let n = moved[2].len();
    for t in moved[2][n - 40..n - 1].iter_mut() {
        *t = 150_000;
    }
    assert!(m.draft.as_ref().unwrap().coverage(&moved[2][n - 64..]) < 0.98);
    let outside = m.propose_drafts_batch_want(&asks(&moved, &samplings, &[7; SLOTS])).unwrap();
    let after_outside = m.draft.as_ref().unwrap().head_rounds;
    assert_eq!(after_outside, [after_inside[0] + 1, after_inside[1]], "a round with text outside the set runs the full head");
    for s in 0..SLOTS {
        assert_eq!(outside[s].as_ref().unwrap().tokens, full[s].as_ref().unwrap().tokens, "slot {s}: the full head's drafts");
        assert_eq!(bits(&outside[s].as_ref().unwrap().score), bits(&full[s].as_ref().unwrap().score), "slot {s}: and scores");
    }
}

/// A 16-row draft block (`REXL3_DRAFT_BLOCK`, the depth study) gives levels 1..7 the same BITS as the trained 8-row block:
/// the layers' in-block attention and dynconv are causal, so the first 7 drafts cannot move. Only the depth study reads the
/// deeper levels; this is the exactness claim the study's replay rests on (a dump at 16 replays the 8-trained levels as they
/// were). The deeper picks themselves are not compared to anything: the pack was trained at 8 and Stage A measures how good
/// they are.
#[test]
#[ignore]
fn a_16_row_block_keeps_levels_1_to_7_of_the_8_row_block() {
    let (Some(d), Some(dd)) = (std::env::var("REXL3_QWEN_DIR").ok(), std::env::var("REXL3_DRAFT_DIR").ok()) else {
        return eprintln!("needs REXL3_QWEN_DIR, REXL3_DRAFT_DIR, a GPU and testdata/ref: not run");
    };
    if !gb_cuda::device_available() || !have_refs() {
        return eprintln!("needs a GPU and testdata/ref: not run");
    }
    let (d, dd) = (std::path::PathBuf::from(d), std::path::PathBuf::from(dd));
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let vocab = cfg.vocab;
    let long: Vec<u32> = ref_npy("long/ids.npy").to_i64().into_iter().map(|x| x as u32).collect();
    let samplings: Vec<Sampling> = (0..SLOTS)
        .map(|s| Sampling {
            seed: 90 + s as u64,
            temperature: if s % 2 == 0 { 0.0 } else { 1.0 },
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
        })
        .collect();
    let seqs: Vec<Vec<u32>> = (0..SLOTS)
        .map(|s| {
            let mut t = long[s * 9..long.len() - 3 * s].to_vec();
            t.push(3000 + 23 * s as u32);
            t
        })
        .collect();
    let run = |block: usize| {
        let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
        let mut m = Model::new(cfg.clone(), w, SLOTS, 128, 4096, 256).unwrap();
        m.draft = Some(Draft::load_block(&Pack::open(&dd).unwrap(), vocab, SLOTS, 256, block).unwrap().0);
        for s in 0..SLOTS {
            start(&mut m, s, &seqs[s][..seqs[s].len() - 1]);
        }
        let want = block - 1;
        let picked = m.propose_drafts_batch_want(&asks(&seqs, &samplings, &[want; SLOTS])).unwrap();
        for s in 0..SLOTS {
            assert_eq!(picked[s].as_ref().unwrap().tokens.len(), want, "slot {s}: the block's drafts");
        }
        picked
    };
    let b8 = run(8);
    let b16 = run(16);
    for s in 0..SLOTS {
        let (a, b) = (b8[s].as_ref().unwrap(), b16[s].as_ref().unwrap());
        assert_eq!(a.tokens, b.tokens[..7], "slot {s}: levels 1..7 of the 16-block differ from the 8-block");
        assert_eq!(bits(&a.score), bits(&b.score[..7]), "slot {s}: winning scores");
        assert_eq!(bits(&a.margin), bits(&b.margin[..7]), "slot {s}: margins");
    }
    eprintln!("16-row block: levels 1..7 equal the 8-row block's bits at {SLOTS} slots");
}
