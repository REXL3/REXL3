//! Keyed-noise drafting (`gb_spec::DraftPick`, docs/GLM53.md "Keyed-noise drafting"): a drafter with logits of its own proposes the TARGET's keyed
//! draw over them, at the draft's own position, instead of its argmax. On the mock the properties are exact:
//!
//! * a drafter whose logits ARE the target's is accepted 100% at every depth, for every sampled request (temperature, top_k, top_p, min_p, seed): the
//!   test that fails for a wrong position, noise keyed by column instead of token id, a cut that is not the target's, or noise on a greedy request;
//! * a drafter that is close but not equal gains acceptance from the noise at temperature above 0, and is identical to the argmax drafter at 0;
//! * drafted == serial for every pick, depth, seed and rejection pattern (the verify window decides, the draft only proposes).

use gb_engine::harness::drafted_equals_serial;
use gb_engine::mock::{HeadDrafter, MockBackend};
use gb_engine::{Drafter, Engine, EngineConfig, Output, Request};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};
use gb_spec::DraftPick;

const VOCAB: usize = 512;
const SLOTS: usize = 16;

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

#[derive(Clone, Copy)]
struct Head {
    eps: f32,
    pick: DraftPick,
    wrong_at: Option<usize>,
    always_wrong: bool,
    shift: i64,
}

fn engine_with(head: Head, depth: usize) -> impl Fn(bool) -> Engine<MockBackend> {
    move |drafting| {
        let make: Box<dyn Fn() -> Box<dyn Drafter>> = Box::new(move || {
            let mut d = HeadDrafter::new(VOCAB, head.eps, head.pick);
            d.wrong_at = head.wrong_at;
            d.always_wrong = head.always_wrong;
            d.shift_position = head.shift;
            Box::new(d) as Box<dyn Drafter>
        });
        Engine::new(
            MockBackend::new(VOCAB, SLOTS, gb_engine::mock::Fault::None),
            admission(),
            EngineConfig { max_rows: 64, max_drafts: depth, cost: cost(), drafting },
            make,
        )
    }
}

fn head(eps: f32, pick: DraftPick) -> Head {
    Head { eps, pick, wrong_at: None, always_wrong: false, shift: 0 }
}

fn request(i: usize, s: Sampling, max_new: usize) -> Request {
    let prompt: Vec<u32> = (0..(6 + (i % 7) * 4)).map(|j| ((i * 131 + j * 17) % VOCAB) as u32).collect();
    Request { prompt, sampling: s, max_new, stop: vec![], draft: true, images: vec![], cache: true }
}

/// The requests in groups that fit the slots, each group on a fresh engine; the outputs in request order.
fn run(make: &dyn Fn(bool) -> Engine<MockBackend>, reqs: &[Request]) -> Vec<Output> {
    reqs.chunks(SLOTS)
        .flat_map(|group| {
            let mut e = make(true);
            for r in group {
                e.submit(r.clone()).unwrap();
            }
            e.run().unwrap();
            e.take_outputs()
        })
        .collect()
}

fn totals(outs: &[Output]) -> (u64, u64, u64, usize) {
    let drafted = outs.iter().map(|o| o.drafted).sum();
    let accepted = outs.iter().map(|o| o.accepted).sum();
    let rounds = outs.iter().map(|o| o.rounds).sum();
    let tokens = outs.iter().map(|o| o.tokens.len().saturating_sub(1)).sum();
    (drafted, accepted, rounds, tokens)
}

/// Sampling settings that exercise every cut of the sampler.
fn grid() -> Vec<Sampling> {
    let mut v = Vec::new();
    for (t, k, p, m) in [
        (0.7, 20, 0.95, 0.0),
        (1.0, 20, 0.95, 0.0),
        (1.0, 56, 1.0, 0.0),
        (0.7, 5, 1.0, 0.0),
        (1.0, 40, 0.6, 0.0),
        (1.3, 20, 0.9, 0.1),
        (0.4, 8, 1.0, 0.05),
        (1.0, 1, 1.0, 0.0),
    ] {
        v.push(Sampling { seed: 0, temperature: t, top_k: k, top_p: p, min_p: m });
    }
    v
}

#[test]
fn a_drafter_with_the_targets_logits_is_accepted_at_every_depth_for_every_sampled_request() {
    for depth in 1..=4usize {
        let make = engine_with(head(0.0, DraftPick::Keyed), depth);
        let mut reqs = Vec::new();
        for (g, s) in grid().into_iter().enumerate() {
            for seed in 0..3u64 {
                reqs.push(request(g * 3 + seed as usize, Sampling { seed: 17 + seed * 1_000_003 + g as u64, ..s }, 60));
            }
        }
        let outs = run(&make, &reqs);
        let (drafted, accepted, rounds, tokens) = totals(&outs);
        assert!(drafted > 100 * depth as u64 / 2, "depth {depth}: only {drafted} drafts were verified");
        assert_eq!(accepted, drafted, "depth {depth}: a drafter with the target's own logits must be accepted at every position");
        assert!(tokens as f64 / rounds as f64 > depth as f64 * 0.8, "depth {depth}: {tokens} tokens in {rounds} rounds");
    }
}

#[test]
fn the_control_a_shifted_position_is_not_accepted() {
    // the same drafter with the noise of the position one past: the test above has the power to see a wrong position
    let reqs: Vec<Request> = (0..12)
        .map(|i| request(i, Sampling { seed: 5 + i as u64, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 }, 60))
        .collect();
    for shift in [-1i64, 1] {
        let mut h = head(0.0, DraftPick::Keyed);
        h.shift = shift;
        let (drafted, accepted, ..) = totals(&run(&engine_with(h, 3), &reqs));
        assert!(drafted > 100 && (accepted as f64) < 0.85 * drafted as f64, "shift {shift}: {accepted} of {drafted} accepted");
    }
}

#[test]
fn the_argmax_drafter_loses_what_the_noise_wins_and_greedy_requests_are_identical() {
    let hot: Vec<Request> = (0..16)
        .map(|i| request(i, Sampling { seed: 90 + i as u64, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 }, 80))
        .collect();
    let (d0, a0, ..) = totals(&run(&engine_with(head(0.0, DraftPick::Argmax), 3), &hot));
    let (d1, a1, ..) = totals(&run(&engine_with(head(0.0, DraftPick::Keyed), 3), &hot));
    assert!(d0 > 100 && d1 > 100);
    let (r0, r1) = (a0 as f64 / d0 as f64, a1 as f64 / d1 as f64);
    assert!(r1 > 0.999, "keyed {r1}");
    assert!(r0 < 0.8, "the argmax drafter's acceptance at T=1.0 should be well below 1, got {r0}: the mock's distribution is too sharp for the test");
    // greedy: no noise, both picks propose the same tokens, so every counter is equal, even for a perturbed head
    let greedy: Vec<Request> = (0..16)
        .map(|i| request(i, Sampling { seed: 90 + i as u64, temperature: 0.0, top_k: 20, top_p: 0.95, min_p: 0.0 }, 80))
        .collect();
    for eps in [0.0f32, 0.6] {
        let a = run(&engine_with(head(eps, DraftPick::Argmax), 3), &greedy);
        let k = run(&engine_with(head(eps, DraftPick::Keyed), 3), &greedy);
        assert_eq!(totals(&a), totals(&k), "eps {eps}: greedy counters differ between the keyed and the argmax drafter");
        for (x, y) in a.iter().zip(&k) {
            assert_eq!((x.tokens.clone(), x.drafted, x.accepted, x.rounds), (y.tokens.clone(), y.drafted, y.accepted, y.rounds));
        }
    }
}

#[test]
fn a_close_drafter_gains_acceptance_from_the_noise_at_every_sampled_temperature() {
    for eps in [0.3f32, 0.5] {
        for (t, top_p) in [(0.7, 0.95), (1.0, 0.95), (1.0, 1.0)] {
            let reqs: Vec<Request> = (0..24)
                .map(|i| request(i, Sampling { seed: 300 + i as u64, temperature: t, top_k: 20, top_p, min_p: 0.0 }, 80))
                .collect();
            let (da, aa, ra, ta) = totals(&run(&engine_with(head(eps, DraftPick::Argmax), 3), &reqs));
            let (dk, ak, rk, tk) = totals(&run(&engine_with(head(eps, DraftPick::Keyed), 3), &reqs));
            assert!(da > 200 && dk > 200);
            let (ra_, rk_) = (aa as f64 / da as f64, ak as f64 / dk as f64);
            let (tpr_a, tpr_k) = (ta as f64 / ra as f64, tk as f64 / rk as f64);
            eprintln!("eps {eps} T {t} top_p {top_p}: accepted {ra_:.3} -> {rk_:.3}, tokens/round {tpr_a:.2} -> {tpr_k:.2}");
            assert!(rk_ > ra_ + 0.04, "eps {eps} T {t}: keyed acceptance {rk_:.3} against argmax {ra_:.3}");
            assert!(tpr_k > tpr_a * 1.03, "eps {eps} T {t}: tokens per round {tpr_k:.2} against {tpr_a:.2}");
        }
    }
}

#[test]
fn drafted_equals_serial_for_every_pick_depth_seed_and_rejection_pattern() {
    let mut cases = 0;
    for pick in [DraftPick::Keyed, DraftPick::Argmax] {
        for depth in 1..=4usize {
            let mut reqs = Vec::new();
            for (g, s) in grid().into_iter().enumerate() {
                reqs.push(request(g, Sampling { seed: 31 + g as u64, ..s }, 48));
            }
            // a greedy request, and a request whose seed repeats: the same seed on the same prompt twice, and on another prompt
            reqs.push(request(2, Sampling { seed: 31, temperature: 0.0, ..Sampling::default() }, 48));
            reqs.push(request(9, Sampling { seed: 777, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 }, 48));
            reqs.push(request(9, Sampling { seed: 777, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 }, 48));
            reqs.push(request(4, Sampling { seed: 777, temperature: 0.7, top_k: 40, top_p: 0.9, min_p: 0.0 }, 48));
            let mut variants: Vec<Head> = vec![head(0.0, pick), head(0.5, pick), Head { always_wrong: true, ..head(0.5, pick) }];
            // a draft rejected at one depth only, every depth in turn
            for d in 0..depth {
                variants.push(Head { wrong_at: Some(d), ..head(0.0, pick) });
            }
            for v in variants {
                let r = drafted_equals_serial(&engine_with(v, depth), &reqs).unwrap_or_else(|e| {
                    panic!("{pick:?} depth {depth} eps {} wrong_at {:?} always_wrong {}: {e}", v.eps, v.wrong_at, v.always_wrong)
                });
                assert!(r.drafted > 0);
                if v.always_wrong {
                    assert!(
                        r.accepted <= r.drafted / 8 + 1,
                        "an always-wrong drafter is accepted by luck only: {} of {}",
                        r.accepted,
                        r.drafted
                    );
                }
                cases += 1;
            }
        }
    }
    assert!(cases >= 40, "{cases} cases");
}

#[test]
fn a_repeated_seed_gives_the_same_reply_whatever_the_drafter() {
    // two requests with one seed and one prompt: the same text, with the drafter and without, keyed or not
    let s = Sampling { seed: 4242, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 };
    let reqs = vec![request(3, s, 60), request(3, s, 60)];
    let serial = {
        let mut e = engine_with(head(0.5, DraftPick::Keyed), 3)(false);
        for r in &reqs {
            e.submit(r.clone()).unwrap();
        }
        e.run().unwrap();
        e.take_outputs()
    };
    assert_eq!(serial[0].tokens, serial[1].tokens);
    for pick in [DraftPick::Keyed, DraftPick::Argmax] {
        let outs = run(&engine_with(head(0.5, pick), 4), &reqs);
        assert_eq!(outs[0].tokens, serial[0].tokens, "{pick:?}");
        assert_eq!(outs[1].tokens, serial[0].tokens, "{pick:?}");
    }
}
