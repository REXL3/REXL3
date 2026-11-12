//! Prefill under a burst of arrivals: `PrefillPolicy::Burst` (several slices a step while many prompts wait, the prompt with the fewest
//! rows left first, a pass-over guard against starving a long one) against the serial policy of the first release. The policy decides
//! when a prompt's first token arrives, never a token of any reply.

use gb_engine::mock::{ChainDrafter, Fault, MockBackend};
use gb_engine::{Drafter, Engine, EngineConfig, PrefillPolicy, Request, PREFILL_PASS_GUARD};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};
use std::collections::BTreeMap;

const VOCAB: usize = 512;

fn admission(slots: usize) -> Admission {
    let mm = ModelMemory {
        weights_bytes: 1 << 30,
        kv_bytes_per_token: 65536,
        state_bytes_per_slot: 150 << 20,
        activation_reserve: 1 << 30,
    };
    Admission::new(100 << 30, 6 << 30, mm, slots).unwrap()
}

fn engine(policy: PrefillPolicy, slots: usize, slice: usize, drafting: bool) -> Engine<MockBackend> {
    let mut b = MockBackend::new(VOCAB, slots, Fault::None);
    if slice > 0 {
        b = b.with_prefill_slice(slice);
    }
    Engine::new(
        b,
        admission(slots),
        EngineConfig { max_rows: 64, max_drafts: 7, cost: CostModel::from_table(vec![(1, 70.0), (64, 80.0)]), drafting },
        Box::new(|| Box::new(ChainDrafter { vocab: VOCAB, corrupt_every: 5 }) as Box<dyn Drafter>),
    )
    .with_prefill_policy(policy)
}

fn request(i: usize, prompt_len: usize, max_new: usize) -> Request {
    Request {
        prompt: (0..prompt_len).map(|j| ((i * 97 + j * 13 + 5) % VOCAB) as u32).collect(),
        sampling: Sampling {
            seed: 700 + i as u64,
            temperature: if i % 2 == 0 { 0.0 } else { 0.9 },
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
        },
        max_new,
        stop: vec![],
        draft: true,
        images: vec![],
        cache: true,
    }
}

/// Prompt lengths 6..=37 in a scrambled order, so that the shortest prompts are not the first submitted.
fn scrambled_len(i: usize) -> usize {
    6 + (i * 11) % 32
}

/// Step number (1-based) at which each request first has a token, running until all have one.
fn first_token_steps(e: &mut Engine<MockBackend>, ids: &[u64]) -> BTreeMap<u64, usize> {
    let mut at = BTreeMap::new();
    let mut step = 0;
    while at.len() < ids.len() {
        step += 1;
        assert!(step < 10_000, "prompts never got their first token: {at:?}");
        e.step().unwrap();
        for ev in e.take_events() {
            if !ev.tokens.is_empty() {
                at.entry(ev.id).or_insert(step);
            }
        }
    }
    at
}

fn submit_all(e: &mut Engine<MockBackend>, n: usize) -> Vec<u64> {
    (0..n).map(|i| e.submit_deferred(request(i, scrambled_len(i), 40)).unwrap()).collect()
}

#[test]
fn a_burst_of_32_short_prompts_gets_its_first_tokens_in_a_third_of_the_steps() {
    let mut serial = engine(PrefillPolicy::Serial, 40, 0, true);
    let ids = submit_all(&mut serial, 32);
    let s = first_token_steps(&mut serial, &ids);
    let mut burst = engine(PrefillPolicy::Burst { div: 4, cap: 8 }, 40, 0, true);
    let ids = submit_all(&mut burst, 32);
    let b = first_token_steps(&mut burst, &ids);
    let mean = |m: &BTreeMap<u64, usize>| m.values().sum::<usize>() as f64 / m.len() as f64;
    // serial: the i-th prompt waits i steps (a slice a step); burst: ceil(waiting / 4) slices a step, at most 8
    assert_eq!(*s.values().max().unwrap(), 32);
    assert!(*b.values().max().unwrap() <= 12, "the last prompt waited {} steps", b.values().max().unwrap());
    assert!(mean(&b) * 3.0 < mean(&s), "mean steps to the first token: burst {} serial {}", mean(&b), mean(&s));
    eprintln!(
        "steps to the first token, 32 prompts: serial mean {:.1} max 32; burst mean {:.1} max {}",
        mean(&s),
        mean(&b),
        b.values().max().unwrap()
    );
}

#[test]
fn slices_a_step_follow_the_number_waiting_up_to_the_cap() {
    // whole short prompts: one slice finishes one prompt, so the prompts with a first token after step 1 are the slices that step ran
    for (waiting, slices) in [(1usize, 1usize), (2, 1), (3, 1), (4, 1), (5, 2), (8, 2), (9, 3), (16, 4), (32, 8), (64, 8)] {
        let mut e = engine(PrefillPolicy::Burst { div: 4, cap: 8 }, 70, 0, false);
        for i in 0..waiting {
            e.submit_deferred(request(i, scrambled_len(i), 5)).unwrap();
        }
        e.step().unwrap();
        let got = e.take_events().iter().filter(|ev| !ev.tokens.is_empty()).count();
        assert_eq!(got, slices, "{waiting} waiting prompts: {got} slices in the first step");
    }
    // the serial policy and a lone prompt run one slice
    let mut e = engine(PrefillPolicy::Serial, 40, 0, false);
    let ids: Vec<u64> = (0..12).map(|i| e.submit_deferred(request(i, scrambled_len(i + 1), 5)).unwrap()).collect();
    e.step().unwrap();
    let served: Vec<u64> = e.take_events().iter().filter(|ev| !ev.tokens.is_empty()).map(|ev| ev.id).collect();
    assert_eq!(served, vec![ids[0]], "serial: one slice, to the prompt that arrived first, whatever its length");
}

#[test]
fn the_shortest_prompts_go_first_and_replies_are_the_same_under_every_policy() {
    // the first step of a burst serves the prompts with the fewest rows
    let mut e = engine(PrefillPolicy::BATCHED_SHORT, 40, 0, true);
    let ids = submit_all(&mut e, 32);
    e.step().unwrap();
    let served: Vec<u64> = e.take_events().iter().filter(|ev| !ev.tokens.is_empty()).map(|ev| ev.id).collect();
    let mut lens: Vec<(usize, u64)> = ids.iter().enumerate().map(|(i, &id)| (scrambled_len(i), id)).collect();
    lens.sort();
    let mut want: Vec<u64> = lens[..served.len()].iter().map(|x| x.1).collect();
    want.sort();
    let mut got = served.clone();
    got.sort();
    assert_eq!(got, want, "the first slices went to the {} shortest prompts", served.len());
    // every policy gives every request the same tokens (greedy and seeded, drafted)
    let mut outs: Vec<BTreeMap<u64, Vec<u32>>> = Vec::new();
    for policy in [
        PrefillPolicy::Serial,
        PrefillPolicy::BATCHED_SHORT,
        PrefillPolicy::Burst { div: 1, cap: 32 },
        PrefillPolicy::Burst { div: 2, cap: 3 },
    ] {
        let mut e = engine(policy, 40, 5, true);
        submit_all(&mut e, 32);
        e.run().unwrap();
        outs.push(e.take_outputs().into_iter().map(|o| (o.id, o.tokens)).collect());
    }
    for (k, o) in outs.iter().enumerate().skip(1) {
        assert_eq!(*o, outs[0], "policy {k} changed a reply");
    }
    assert_eq!(outs[0].len(), 32);
}

#[test]
fn a_long_prompt_among_short_arrivals_is_slowed_but_never_starved() {
    // a 200-row prompt in slices of 4 (50 slices) while a new 5-row prompt arrives every step: fewest-rows-first alone would serve the
    // short ones for as long as they keep coming. The guard serves the long one at least once in every PREFILL_PASS_GUARD + 1 slices.
    let mut e = engine(PrefillPolicy::BATCHED_SHORT, 700, 4, false);
    let long = e.submit_deferred(request(0, 200, 5)).unwrap();
    let mut long_first = None;
    let mut next = 1usize;
    for step in 1..=700 {
        if step <= 600 {
            e.submit_deferred(request(next, 5, 2)).unwrap();
            next += 1;
        }
        e.step().unwrap();
        for ev in e.take_events() {
            if ev.id == long && !ev.tokens.is_empty() {
                long_first.get_or_insert(step);
            }
        }
        if long_first.is_some() {
            break;
        }
    }
    let at = long_first.expect("the long prompt never got its first token");
    assert!(at < 600, "the long prompt waited until the arrivals stopped (step {at}): it was starved");
    // 50 slices, each served within 9 slices of the last, at 2 or more slices a step
    assert!(at <= 50 * (PREFILL_PASS_GUARD as usize + 1), "step {at}");
}

#[test]
fn cancelled_prompts_in_a_burst_give_everything_back_and_the_rest_finish() {
    let mut e = engine(PrefillPolicy::BATCHED_SHORT, 40, 0, true);
    let ids = submit_all(&mut e, 24);
    let free = e.admission().capacity();
    for id in [ids[1], ids[7], ids[20]] {
        assert!(e.cancel(id), "a waiting prompt can be cancelled");
    }
    assert_eq!(e.prefilling(), 21);
    e.run().unwrap();
    let outs = e.take_outputs();
    assert_eq!(outs.len(), 24);
    let cancelled = outs.iter().filter(|o| o.finish == Some(gb_engine::FinishReason::Cancelled)).count();
    assert_eq!(cancelled, 3);
    assert!(outs.iter().filter(|o| o.finish != Some(gb_engine::FinishReason::Cancelled)).all(|o| o.tokens.len() == 40));
    assert_eq!(e.admission().available(), free, "every slot and byte is back");
    assert!(!e.busy());
}
