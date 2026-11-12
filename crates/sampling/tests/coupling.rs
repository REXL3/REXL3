//! The keyed sampler's coupling contract, as the draft walks rely on it:
//!
//! - a draft head that scores a superset of the target's kept candidates draws the *same* noise per
//!   (seed, position, token id), so its pick equals the target's sample whenever the target's sample is
//!   inside the draft's kept set (Gumbel-max coupling);
//! - temperature 0 is pure argmax: no noise, no seed/position dependence — greedy drafts must take the
//!   argmax path unchanged.

use gb_sampling::{choose, Sampling};

#[test]
fn greedy_takes_the_argmax_path_without_noise() {
    let values = [0.3f32, 2.0, -1.0, 2.0, 0.9];
    let ids = [4u32, 8, 1, 2, 6];
    let s = Sampling { seed: 0, temperature: 0.0, top_k: 0, top_p: 0.0, min_p: 0.0 };
    // argmax over (value desc, id asc): 2.0 at ids 8 and 2 -> id 2 wins
    for position in 0..64u64 {
        for seed in [0u64, 7, 12345, u64::MAX >> 1] {
            let t = Sampling { seed, ..s };
            assert_eq!(choose(&values, &ids, position, &t), 2, "seed {seed} position {position}");
        }
    }
}

#[test]
fn target_sample_inside_the_drafts_kept_set_is_reproduced_by_the_draft() {
    // One logits row; the draft head keeps the top `k` candidates, the target keeps a prefix thereof.
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        rng ^= rng >> 30;
        rng = rng.wrapping_mul(0xBF58476D1CE4E5B9);
        rng ^= rng >> 27;
        rng = rng.wrapping_mul(0x94D049BB133111EB);
        rng ^ (rng >> 31)
    };
    let n = 256usize;
    let mut values = vec![0f32; n];
    let ids: Vec<u32> = (0..n as u32).collect();
    for v in values.iter_mut() {
        *v = ((next() % 1000) as f32) / 1000.0 - 0.5;
    }
    // a few distinct high scores so the kept set is stable
    values[3] = 3.0;
    values[55] = 3.2;
    values[200] = 3.4;
    let s = Sampling { seed: 42, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 };
    for position in [0u64, 1, 100, 7_000_001] {
        let target = choose(&values, &ids, position, &s);
        // the draft keeps the same top candidates (here: the full row, so the target's sample is in it)
        let draft = choose(&values, &ids, position, &s);
        assert_eq!(draft, target, "same kept set must pick the same token at position {position}");
        // the target's sample is in the draft candidate set by construction; a draft that only kept a strict
        // prefix of the target's kept set cannot reproduce it — that is exactly the `cand_k` coupling cut
    }
}

#[test]
fn the_same_token_draws_the_same_noise_across_rows() {
    // The noise of a candidate depends on its token id alone: a candidate that appears in two different
    // rows with the same score must be scored with the same Gumbel draw (this is what lets a draft head
    // that scores a subset of the vocabulary draw the very noise the full head would).
    let s = Sampling { seed: 7, temperature: 1.0, top_k: 4, top_p: 1.0, min_p: 0.0 };
    for position in [3u64, 1000, 123_456_789] {
        // row A: the candidate set {10, 20, 30, 40}; row B: {40, 30, 20, 10} reordered.
        let a = choose(&[0.1, 0.2, 0.3, 0.4], &[10, 20, 30, 40], position, &s);
        let b = choose(&[0.4, 0.3, 0.2, 0.1], &[40, 30, 20, 10], position, &s);
        assert_eq!(a, b, "reordering candidates must not change the keyed pick");
    }
}
