//! How a drafter that has logits of its own turns them into a proposal: its argmax, or the TARGET's own keyed draw over them.
//!
//! The engine's sampler is keyed (`gb_sampling`): the token at absolute position `p` is the argmax over the kept candidates of `logit / T +
//! g(seed, p, id)`, `g` Gumbel noise that is a pure function of (seed, position, token id). A draft is committed only where it equals that
//! sample, so the best proposal a drafter can make at position `p` is the sample the target would draw if the target's distribution were the
//! drafter's: the same candidates (the drafter's top `top_k`), the same temperature, `top_p` and `min_p`, and the SAME noise, because the noise
//! is known before the target has run. Two distributions that agree closely then give the same token whenever the noise does not separate them,
//! instead of only when their argmaxes agree (the draft's argmax is the sample only where the noise happens to favour the target's argmax: at
//! temperature 0.7 to 1.0 that is 73-75% of GLM's drafts).
//!
//! Nothing here can change an output byte. The verify window decides every committed token, with the target's logits, with `gb_sampling` untouched;
//! a draft is a proposal and `drafted == serial` holds for any pick, right or wrong. A greedy request (temperature 0) has no noise: both modes
//! return the argmax, ties to the lowest id, exactly as a drafter that never heard of noise does.
//!
//! Family-agnostic by construction: a drafter that has a list of candidates (`values`, `ids`) and knows where the draft will be verified calls
//! [`try_draft_pick`]. The Qwen DFlash selector's walk does the same on the device (`Walk::Keyed`, `keyed_choose` in `csrc/keyed.cuh`); the GLM
//! NextN head does it on the device over its top-`top_k` (`gb_glm_keyed_pick`, same header). This is the CPU reference they are tested against.

use gb_sampling::{try_choose, Sampling, SamplingError};

/// How a drafter picks its proposal from its own logits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DraftPick {
    /// The highest logit, ties to the lowest id (what a drafter does without noise).
    Argmax,
    /// The target's draw over the drafter's logits at the draft's own position (identical to `Argmax` at temperature 0).
    #[default]
    Keyed,
}

impl DraftPick {
    /// The setting of `REXL3_GLM_DRAFT_NOISE`-style knobs: `0`, `off`, `false`, `argmax` or `1`, `on`, `true`, `keyed`. Unset keeps `default`;
    /// anything else is an error (a measurement must not silently run on the default).
    pub fn from_setting(value: Option<&str>, name: &str, default: DraftPick) -> Result<DraftPick, String> {
        match value.map(str::trim) {
            None => Ok(default),
            Some("0" | "off" | "false" | "argmax") => Ok(DraftPick::Argmax),
            Some("1" | "on" | "true" | "keyed") => Ok(DraftPick::Keyed),
            Some(other) => Err(format!("{name}={other} is not 0 or 1 (off or on)")),
        }
    }

    pub fn is_keyed(self) -> bool {
        self == DraftPick::Keyed
    }
}

/// The absolute position at which draft `k` (0-based) of a sequence of `len` tokens is compared with the target's sample. The sequence ends with
/// one token not yet fed, so a window feeds `[last, d_0, d_1, ..]` at positions `len - 1 ..` and row `j` predicts position `len + j`, which the
/// engine samples with the keyed sampler at `len + j` (`Engine::decode_round`: `base + j`). Draft `k` is what row `k` is compared with.
pub fn draft_position(len: usize, k: usize) -> u64 {
    (len + k) as u64
}

/// One proposal from a drafter's candidates: `values` are its logits for token `ids` (any order; at least the top `s.top_k`, which is all a
/// pick reads), `position` where the draft will be verified ([`draft_position`]). Errors as `try_choose` does (NaN logits, no candidates, invalid
/// sampling parameters).
pub fn try_draft_pick(values: &[f32], ids: &[u32], position: u64, s: &Sampling, mode: DraftPick) -> Result<u32, SamplingError> {
    match mode {
        DraftPick::Keyed => try_choose(values, ids, position, s),
        // temperature 0 is true greedy in the sampler: no noise, no cut but the highest logit, ties to the lowest id
        DraftPick::Argmax => try_choose(values, ids, position, &Sampling { temperature: 0.0, ..*s }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gb_sampling::{top_candidates, uniform};

    fn lcg(r: &mut u64) -> u64 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *r >> 33
    }

    /// A random row of `n` logits with a few strong tokens (so top_p and min_p cut something) and some exact ties.
    fn row(r: &mut u64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let base = ((lcg(r) % 4000) as f32 / 1000.0) - 2.0;
                let bump = if lcg(r) % 9 == 0 { 3.0 + (lcg(r) % 3) as f32 } else { 0.0 };
                if i % 17 == 5 {
                    1.0 // exact ties
                } else {
                    base + bump
                }
            })
            .collect()
    }

    fn settings(r: &mut u64) -> Sampling {
        Sampling {
            seed: lcg(r) & ((1 << 63) - 1),
            temperature: [0.3, 0.7, 1.0, 1.3][(lcg(r) % 4) as usize],
            top_k: [1usize, 2, 8, 20, 40, 56][(lcg(r) % 6) as usize],
            top_p: [1.0, 0.95, 0.8, 0.5][(lcg(r) % 4) as usize],
            min_p: [0.0, 0.05, 0.2][(lcg(r) % 3) as usize],
        }
    }

    #[test]
    fn a_keyed_pick_over_the_targets_own_logits_is_the_targets_sample_at_every_position() {
        // the contract the engine and the mock tests lean on: a drafter whose logits ARE the target's draws what the target draws
        let mut r = 7u64;
        for case in 0..300 {
            let logits = row(&mut r, 300);
            let s = settings(&mut r);
            let (ids, vals) = top_candidates(&logits, s.candidate_count(logits.len()));
            let pos = lcg(&mut r) % 5000;
            let want = try_choose(&vals, &ids, pos, &s).unwrap();
            assert_eq!(try_draft_pick(&vals, &ids, pos, &s, DraftPick::Keyed).unwrap(), want, "case {case}");
        }
    }

    #[test]
    fn the_noise_is_the_targets_at_the_draft_position_and_nowhere_else() {
        // a flat distribution: the pick is decided by the noise alone, so it must follow (seed, position) and move with them
        let ids: Vec<u32> = (0..32).collect();
        let vals = vec![0.0f32; 32];
        let s = Sampling { seed: 41, temperature: 1.0, top_k: 32, top_p: 1.0, min_p: 0.0 };
        let mut moved = 0;
        for pos in 0..200u64 {
            let t = try_draft_pick(&vals, &ids, pos, &s, DraftPick::Keyed).unwrap();
            // by hand: the largest Gumbel draw of the 32 ids at this position
            let best = (0..32u32)
                .max_by(|&a, &b| {
                    let g = |id: u32| -(-uniform(41, pos, id as u64).ln()).ln();
                    g(a).partial_cmp(&g(b)).unwrap()
                })
                .unwrap();
            assert_eq!(t, best, "position {pos}");
            if t != try_draft_pick(&vals, &ids, pos + 1, &s, DraftPick::Keyed).unwrap() {
                moved += 1;
            }
        }
        assert!(moved > 100, "a pick one position over drew the same noise {} times in 200", 200 - moved);
    }

    #[test]
    fn greedy_has_no_noise_in_either_mode_and_argmax_has_none_at_any_temperature() {
        let mut r = 99u64;
        for _ in 0..200 {
            let logits = row(&mut r, 120);
            let (ids, vals) = top_candidates(&logits, 28);
            let g = Sampling { temperature: 0.0, ..settings(&mut r) };
            let want = ids[0];
            for pos in [0u64, 3, 977] {
                assert_eq!(try_draft_pick(&vals, &ids, pos, &g, DraftPick::Keyed).unwrap(), want);
                assert_eq!(try_draft_pick(&vals, &ids, pos, &g, DraftPick::Argmax).unwrap(), want);
            }
            let hot = Sampling { temperature: 1.0, ..settings(&mut r) };
            for pos in [0u64, 3, 977] {
                assert_eq!(
                    try_draft_pick(&vals, &ids, pos, &hot, DraftPick::Argmax).unwrap(),
                    want,
                    "argmax ignores T, seed, position"
                );
            }
        }
    }

    #[test]
    fn a_restricted_candidate_list_draws_the_noise_of_the_token_id_not_of_its_column() {
        // the restricted-vocabulary head: its columns are a subset of the ids, in id order, and a column's id comes from a map. When every
        // candidate the sampler keeps is inside the subset, the pick over the subset is the pick over the whole vocabulary: the noise of a
        // kept token is keyed by its id. (A drafter that keyed by column would draw other noise, which the control below shows.)
        let mut r = 5u64;
        let (mut checked, mut column_keyed_differs) = (0, 0);
        for _ in 0..400 {
            let logits = row(&mut r, 600);
            let s = settings(&mut r);
            let pos = lcg(&mut r) % 4000;
            let (fids, fvals) = top_candidates(&logits, 64);
            let full = try_draft_pick(&fvals, &fids, pos, &s, DraftPick::Keyed).unwrap();
            // the sampler's kept candidates are inside the first `top_k` of the list; the subset drops a third of the other ids
            let kept: Vec<u32> = fids.iter().take(s.top_k.min(64)).copied().collect();
            let keep: Vec<usize> = (0..600).filter(|&i| kept.contains(&(i as u32)) || lcg(&mut r) % 3 != 0).collect();
            let sub: Vec<f32> = keep.iter().map(|&i| logits[i]).collect();
            let (cols, cvals) = top_candidates(&sub, 64);
            let ids: Vec<u32> = cols.iter().map(|&c| keep[c as usize] as u32).collect();
            let got = try_draft_pick(&cvals, &ids, pos, &s, DraftPick::Keyed).unwrap();
            assert_eq!(got, full, "the restricted pick must be the full pick when the kept candidates are inside the set");
            checked += 1;
            // control: the same list with COLUMN numbers as ids draws the noise of the columns
            let by_col = try_draft_pick(&cvals, &cols, pos, &s, DraftPick::Keyed).unwrap();
            if keep[by_col as usize] as u32 != full {
                column_keyed_differs += 1;
            }
        }
        assert_eq!(checked, 400);
        assert!(
            column_keyed_differs > 20,
            "the control must be able to fail: a column-keyed pick differed {column_keyed_differs} times"
        );
    }

    #[test]
    fn the_position_of_a_draft_is_where_the_window_samples_its_row() {
        // a sequence of 10 tokens: the window feeds [t9, d0, d1, ..] at 9, 10, 11 and row j predicts (and samples at) 10 + j
        assert_eq!(draft_position(10, 0), 10);
        assert_eq!(draft_position(10, 3), 13);
    }

    #[test]
    fn the_setting_is_zero_or_one_and_an_unknown_value_is_an_error() {
        let n = "REXL3_GLM_DRAFT_NOISE";
        assert_eq!(DraftPick::from_setting(None, n, DraftPick::Keyed), Ok(DraftPick::Keyed));
        assert_eq!(DraftPick::from_setting(None, n, DraftPick::Argmax), Ok(DraftPick::Argmax));
        for v in ["0", "off", "false", "argmax", " 0 "] {
            assert_eq!(DraftPick::from_setting(Some(v), n, DraftPick::Keyed), Ok(DraftPick::Argmax), "{v}");
        }
        for v in ["1", "on", "true", "keyed"] {
            assert_eq!(DraftPick::from_setting(Some(v), n, DraftPick::Argmax), Ok(DraftPick::Keyed), "{v}");
        }
        for v in ["", "2", "yes", "maybe"] {
            let e = DraftPick::from_setting(Some(v), n, DraftPick::Keyed).unwrap_err();
            assert!(e.contains(n) && e.contains("0 or 1"), "{e}");
        }
    }
}
