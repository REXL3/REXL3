//! The engine's speculation counters agree with what each request reports, split the way the budget will read them, and keep the
//! drafter's confidence aligned with the draft it scores.

use gb_engine::mock::{mock_logits, ChainDrafter, Fault, MockBackend};
use gb_engine::{
    Backend, Drafter, Engine, EngineConfig, ImageSpan, PrefillStep, Proposal, ProposeRequest, Request, RowOutput, RowRequest,
};
use gb_sampling::{top_candidates, Sampling};
use gb_sched::{Admission, CostModel, ModelMemory};
use gb_spec::stats::{conf_bin, Mode, MAX_POSITIONS};

const VOCAB: usize = 512;
const SLOTS: usize = 8;

fn admission() -> Admission {
    let mm = ModelMemory {
        weights_bytes: 1 << 30,
        kv_bytes_per_token: 65536,
        state_bytes_per_slot: 150 << 20,
        activation_reserve: 1 << 30,
    };
    Admission::new(64 << 30, 6 << 30, mm, SLOTS).unwrap()
}

fn cost() -> CostModel {
    CostModel::from_table(vec![(1, 70.0), (16, 74.0), (32, 110.0), (64, 190.0)])
}

fn requests(n: usize) -> Vec<Request> {
    (0..n)
        .map(|i| Request {
            prompt: (0..(5 + i * 3)).map(|j| ((i * 131 + j * 17) % VOCAB) as u32).collect(),
            // a mix: greedy for even i, sampled for odd i
            sampling: Sampling {
                seed: 99 + i as u64,
                temperature: if i % 2 == 0 { 0.0 } else { 0.8 },
                top_k: 20,
                top_p: 0.95,
                min_p: 0.0,
            },
            max_new: 60 + 7 * i,
            stop: vec![],
            draft: true,
            images: vec![],
            cache: true,
        })
        .collect()
}

fn engine() -> Engine<MockBackend> {
    Engine::new(
        MockBackend::new(VOCAB, SLOTS, Fault::None),
        admission(),
        EngineConfig { max_rows: 64, max_drafts: 7, cost: cost(), drafting: true },
        Box::new(|| Box::new(ChainDrafter { vocab: VOCAB, corrupt_every: 4 }) as Box<dyn Drafter>),
    )
}

#[test]
fn the_position_counters_add_up_to_what_every_request_reports() {
    let mut e = engine();
    let reqs = requests(6);
    for r in reqs.iter().cloned() {
        e.submit(r).unwrap();
    }
    e.run().unwrap();
    let outs = e.take_outputs();
    let (rounds, drafted, accepted): (u64, u64, u64) =
        outs.iter().fold((0, 0, 0), |a, o| (a.0 + o.rounds, a.1 + o.drafted, a.2 + o.accepted));
    assert!(accepted > 0 && drafted > accepted, "the run must see both accepted and rejected drafts: {accepted}/{drafted}");
    let s = e.spec_stats();
    let t = s.total();
    assert_eq!(t.rounds, rounds, "one record per sequence-round");
    assert_eq!(t.verified.iter().sum::<u64>(), drafted, "verified rows are the drafts that rode in a window");
    assert_eq!(t.accepted.iter().sum::<u64>(), accepted);
    for j in 0..MAX_POSITIONS {
        assert!(
            t.accepted[j] <= t.reached[j] && t.reached[j] <= t.verified[j] && t.verified[j] <= t.offered[j],
            "position {}",
            j + 1
        );
    }
    // acceptance can only continue a prefix: each position's accepted count never exceeds the one before it
    assert!(t.accepted.windows(2).all(|w| w[1] <= w[0]));
    // the split by mode: greedy requests are the even ones
    let greedy_rounds: u64 = outs.iter().filter(|o| o.id % 2 == 0).map(|o| o.rounds).sum();
    assert_eq!(s.mode_total(Mode::Greedy).rounds, greedy_rounds);
    assert_eq!(s.mode_total(Mode::Sampled).rounds, rounds - greedy_rounds);
    assert!(t.tokens_per_round() > 1.0);
}

#[test]
fn rounds_with_several_streams_are_counted_apart_from_a_lone_stream() {
    let mut e = engine();
    let mut reqs = requests(3);
    // the first request is much longer than the others, so it ends up running alone
    reqs[0].max_new = 300;
    for r in reqs {
        e.submit(r).unwrap();
    }
    e.run().unwrap();
    let longest = e.take_outputs().iter().map(|o| o.rounds).max().unwrap();
    let s = e.spec_stats();
    let multi: u64 = Mode::ALL.iter().map(|&m| s.by[m as usize][1].rounds).sum();
    let alone: u64 = Mode::ALL.iter().map(|&m| s.by[m as usize][0].rounds).sum();
    assert!(multi > 0 && alone > 0, "multi {multi} alone {alone}");
    // phase timers: rounds by stream count add up to the decode rounds, and a 3-stream round carried 3 streams' rows
    let total_rounds: u64 = s.phases.rounds.iter().sum();
    assert_eq!(total_rounds, longest, "the longest request ran every round");
    assert!(s.phases.rounds[2] > 0, "a round of three streams ran");
    assert!(s.phases.rows[2] >= 3 * s.phases.rounds[2], "every stream has at least its base row");
    assert!(s.phases.draft_ns.iter().sum::<u64>() > 0);
}

/// The mock with a drafter that scores its picks: a pick that matches the model's own chain scores 10, a corrupted one -5.
struct Scored {
    inner: MockBackend,
}

impl Backend for Scored {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.inner.prefill(slot, start, tokens, candidates)
    }
    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> PrefillStep {
        self.inner.prefill_slice(slot, prompt, images, done, candidates)
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        self.inner.step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.inner.commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        self.inner.release(slot)
    }
    fn propose_batch_scored(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Proposal>> {
        reqs.iter()
            .map(|r| {
                let mut ctx = r.tokens.to_vec();
                let (mut tokens, mut conf, mut margin) = (Vec::new(), Vec::new(), Vec::new());
                for i in 0..r.max {
                    let (ids, _) = top_candidates(&mock_logits(&ctx, VOCAB), 1);
                    let truth = ids[0];
                    // every third pick of a round is wrong: a draft the target will reject
                    let wrong = (ctx.len() + i) % 3 == 0;
                    let t = if wrong { (truth + 1) % VOCAB as u32 } else { truth };
                    tokens.push(t);
                    conf.push(if wrong { -5.0 } else { 10.0 });
                    margin.push(if wrong { 0.2 } else { 6.0 });
                    ctx.push(t);
                }
                Some(Proposal { tokens, conf, margin, ..Default::default() })
            })
            .collect()
    }
}

#[test]
fn the_drafters_confidence_stays_aligned_with_the_draft_it_scores() {
    let mut e = Engine::new(
        Scored { inner: MockBackend::new(VOCAB, SLOTS, Fault::None) },
        admission(),
        EngineConfig { max_rows: 64, max_drafts: 7, cost: cost(), drafting: true },
        Box::new(|| Box::new(gb_engine::NoDrafts) as Box<dyn Drafter>),
    );
    for mut r in requests(4) {
        r.sampling.temperature = 0.0; // greedy: a pick equal to the model's argmax is accepted for certain
        e.submit(r).unwrap();
    }
    e.run().unwrap();
    let c = &e.spec_stats().conf[Mode::Greedy as usize];
    let (good, bad) = (conf_bin(10.0), conf_bin(-5.0));
    let (mut good_reached, mut good_accepted, mut bad_reached, mut bad_accepted) = (0, 0, 0, 0);
    for j in 0..MAX_POSITIONS {
        good_reached += c.reached[j][good];
        good_accepted += c.accepted[j][good];
        bad_reached += c.reached[j][bad];
        bad_accepted += c.accepted[j][bad];
    }
    assert!(good_reached > 20 && bad_reached > 20, "both populations must occur: {good_reached} {bad_reached}");
    // a confident pick that was tested was always right; a corrupted one never was
    assert_eq!(good_accepted, good_reached, "high-confidence picks are the true chain");
    assert_eq!(bad_accepted, 0, "a score of -5 marks a corrupted draft, which is never accepted");
    // the margin is binned on its own scale and stays aligned the same way
    let m = &e.spec_stats().margin[Mode::Greedy as usize];
    let (mg_good, mg_bad) = (m.bin(6.0), m.bin(0.2));
    assert_ne!(mg_good, mg_bad);
    let sum = |t: &Vec<[u64; gb_spec::stats::CONF_BINS]>, b: usize| -> u64 { t.iter().map(|row| row[b]).sum() };
    assert_eq!(sum(&m.reached, mg_good), good_reached);
    assert_eq!(sum(&m.accepted, mg_good), good_accepted);
    assert_eq!((sum(&m.reached, mg_bad), sum(&m.accepted, mg_bad)), (bad_reached, 0));
}
