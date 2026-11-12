//! Keyed sampling, ported from TensorFold `engine/exact_sampling.py` (MIT, see THIRD_PARTY_NOTICES.md).
//!
//! The token at absolute position `p` is the argmax over the kept candidates of
//! `logit / T + g(seed, p, id)`, where `g` is Gumbel noise from a splitmix64 hash of
//! (seed, position, token id). Candidates are the top-k logits ordered by (value desc, id asc),
//! cut to the smallest prefix holding `top_p` of the probability, then by `min_p`.
//!
//! Because a token depends only on its own row's logits and its position, a drafted token is
//! accepted exactly when it equals what one-token-at-a-time decoding samples there. Drafting then
//! changes speed only, never output. That is the engine's core invariant.
//!
//! Numerics follow numpy's float64 operation order (pairwise summation for `probs.sum()`,
//! sequential `cumsum`) so fixtures generated from the Python reference can be compared.
//! `exp`/`ln` come from the platform libm; `tests/xcheck.rs` records measured agreement.
//!
//! Temperature 0 is true greedy: the candidate with the highest logit, ties to the lowest id, with no noise.
//! (TensorFold's Python clamps the temperature to 1e-6 and still adds noise, so exact ties there are
//! resolved by noise. For every positive temperature the two agree bit for bit.)
#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fmt;

/// Candidates read beyond `top_k`, so tied values at the cut resolve by id.
pub const MARGIN: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub seed: u64,
    pub temperature: f64,
    /// 0 means "every supplied candidate". The engine requires top_k > 0 (see `candidate_count`).
    pub top_k: usize,
    pub top_p: f64,
    pub min_p: f64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self { seed: 0, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 }
    }
}

impl Sampling {
    /// Candidates a backend must return per row: top_k + MARGIN, capped by the vocabulary.
    pub fn candidate_count(&self, vocab: usize) -> usize {
        if self.top_k == 0 {
            vocab
        } else {
            self.top_k.saturating_add(MARGIN).min(vocab)
        }
    }

    /// Reject parameters the sampler cannot honour: a non-finite or negative temperature, or `top_p` /
    /// `min_p` outside [0, 1]. (`top_p` 0 or 1 and `min_p` 0 mean "off".)
    pub fn validate(&self) -> Result<(), SamplingError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(SamplingError::InvalidParams("temperature must be finite and >= 0"));
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(SamplingError::InvalidParams("top_p must be in [0, 1]"));
        }
        if !self.min_p.is_finite() || !(0.0..=1.0).contains(&self.min_p) {
            return Err(SamplingError::InvalidParams("min_p must be in [0, 1]"));
        }
        Ok(())
    }
}

/// Why a row could not be sampled. A bad request or a bad model output must fail that request, not the process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SamplingError {
    LengthMismatch { values: usize, ids: usize },
    NoCandidates,
    NanLogit,
    InvalidParams(&'static str),
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplingError::LengthMismatch { values, ids } => write!(f, "{values} logits but {ids} candidate ids"),
            SamplingError::NoCandidates => write!(f, "no candidates"),
            SamplingError::NanLogit => write!(f, "NaN logit"),
            SamplingError::InvalidParams(m) => write!(f, "invalid sampling parameters: {m}"),
        }
    }
}

impl std::error::Error for SamplingError {}

/// A reproducible seed from the prompt: sha256 of "t0,t1,...|salt", first 8 bytes little-endian,
/// top bit cleared. Same conversation, same reply.
pub fn seed_for(tokens: &[u32], salt: u64) -> u64 {
    // Streams the same bytes as "t0,t1,...|salt" into the hash without building the string.
    fn decimal(h: &mut Sha256, mut n: u64) {
        let mut buf = [0u8; 20];
        let mut i = buf.len();
        loop {
            i -= 1;
            buf[i] = b'0' + (n % 10) as u8;
            n /= 10;
            if n == 0 {
                break;
            }
        }
        h.update(&buf[i..]);
    }
    let mut h = Sha256::new();
    for (i, &t) in tokens.iter().enumerate() {
        if i > 0 {
            h.update(b",");
        }
        decimal(&mut h, t as u64);
    }
    h.update(b"|");
    decimal(&mut h, salt);
    let d = h.finalize();
    u64::from_le_bytes(d[..8].try_into().unwrap()) & ((1u64 << 63) - 1)
}

#[inline]
fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

const TWO_M53: f64 = 1.0 / 9_007_199_254_740_992.0;
const TWO_M54: f64 = TWO_M53 / 2.0;
/// The largest double below 1.
const BELOW_ONE: f64 = 1.0 - TWO_M53;

/// Map 64 hash bits to the open interval (0, 1). The top 53 bits plus half a step would round to exactly 1.0
/// for the all-ones pattern (a tie that rounds to even), and ln(1) = 0 would give an infinite Gumbel value.
#[inline]
pub fn unit_open(x: u64) -> f64 {
    ((x >> 11) as f64 * TWO_M53 + TWO_M54).min(BELOW_ONE)
}

/// Uniform (0, 1) double from a splitmix64 hash of (seed, position, token id).
#[inline]
pub fn uniform(seed: u64, position: u64, id: u64) -> f64 {
    let x = mix(seed.wrapping_add(0x9E3779B97F4A7C15));
    let x = mix(x ^ position.wrapping_mul(0xD1B54A32D192ED03));
    let x = mix(x ^ id);
    unit_open(x)
}

/// numpy's float64 pairwise summation (the `sum()` of a contiguous array), starting from 0.0.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0;
        for &v in a {
            r += v;
        }
        r
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// Candidate indices ordered by (value desc, id asc): numpy `lexsort((ids, -values))`.
fn order(values: &[f32], ids: &[u32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        let (va, vb) = (-values[a], -values[b]);
        va.partial_cmp(&vb).unwrap_or(Ordering::Equal).then(ids[a].cmp(&ids[b]))
    });
    idx
}

fn check_row(values: &[f32], ids: &[u32]) -> Result<(), SamplingError> {
    if values.len() != ids.len() {
        return Err(SamplingError::LengthMismatch { values: values.len(), ids: ids.len() });
    }
    if values.is_empty() {
        return Err(SamplingError::NoCandidates);
    }
    if values.iter().any(|v| v.is_nan()) {
        return Err(SamplingError::NanLogit);
    }
    Ok(())
}

/// One row: candidate logits `values` for token `ids` at absolute `position` -> the sampled token id.
///
/// Fails on NaN logits (a NaN is a model bug, and refusing is safer than a silent choice), on mismatched or
/// empty candidates, and on invalid parameters. Temperature 0 returns the highest logit, ties to the lowest id.
pub fn try_choose(values: &[f32], ids: &[u32], position: u64, s: &Sampling) -> Result<u32, SamplingError> {
    s.validate()?;
    check_row(values, ids)?;
    if s.temperature == 0.0 {
        let mut best = 0usize;
        for i in 1..values.len() {
            if values[i] > values[best] || (values[i] == values[best] && ids[i] < ids[best]) {
                best = i;
            }
        }
        return Ok(ids[best]);
    }
    let ord = order(values, ids);
    let n = ids.len();
    let k = if s.top_k > 0 { s.top_k.min(n) } else { n }.max(1);
    let t = s.temperature.max(1e-6); // parity with TensorFold, and keeps value / t finite
    let mut kid: Vec<u32> = ord[..k].iter().map(|&i| ids[i]).collect();
    let mut scaled: Vec<f64> = ord[..k].iter().map(|&i| values[i] as f64 / t).collect();
    if s.top_p > 0.0 && s.top_p < 1.0 {
        let mx = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = scaled.iter().map(|v| (v - mx).exp()).collect();
        let total = pairwise_sum(&probs);
        for p in probs.iter_mut() {
            *p /= total;
        }
        // searchsorted(cumsum(probs), top_p, side="left") + 1
        let mut c = 0.0;
        let mut below = 0usize;
        for &p in &probs {
            c += p;
            if c < s.top_p {
                below += 1;
            }
        }
        let keep = (below + 1).min(k);
        kid.truncate(keep);
        scaled.truncate(keep);
    }
    if s.min_p > 0.0 {
        let floor = scaled[0] + s.min_p.ln();
        // scaled[0] is the maximum and ln(min_p) <= 0, so the first candidate always survives
        let keep = scaled.iter().filter(|&&v| v >= floor).count().max(1);
        kid.truncate(keep);
        scaled.truncate(keep);
    }
    let mut best = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (i, (&id, &v)) in kid.iter().zip(&scaled).enumerate() {
        let u = uniform(s.seed, position, id as u64);
        let g = -((-(u.ln())).ln());
        let score = v + g;
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    Ok(kid[best])
}

/// `try_choose` for callers that treat a bad row as a bug: panics with the error.
pub fn choose(values: &[f32], ids: &[u32], position: u64, s: &Sampling) -> u32 {
    try_choose(values, ids, position, s).unwrap_or_else(|e| panic!("sampling failed: {e}"))
}

/// Many rows, each exactly `choose` on its own row; row r uses `positions[r]`.
///
/// Panics if the three slices differ in length (a silent truncation would drop rows).
pub fn choose_rows(values: &[&[f32]], ids: &[&[u32]], positions: &[u64], s: &Sampling) -> Vec<u32> {
    assert!(values.len() == ids.len() && ids.len() == positions.len(), "choose_rows: values, ids and positions differ in length");
    values.iter().zip(ids).zip(positions).map(|((v, i), &p)| choose(v, i, p, s)).collect()
}

/// Exact top-`count` of a full logits row by (value desc, id asc). CPU reference for backends.
/// Fails on a NaN logit, because the ordering is not total with one.
pub fn try_top_candidates(logits: &[f32], count: usize) -> Result<(Vec<u32>, Vec<f32>), SamplingError> {
    if logits.iter().any(|v| v.is_nan()) {
        return Err(SamplingError::NanLogit);
    }
    let take = count.min(logits.len());
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let cmp = |&a: &u32, &b: &u32| {
        // ids are unique, so this is a strict total order and partial selection equals a full sort's prefix
        (-logits[a as usize]).partial_cmp(&-logits[b as usize]).unwrap_or(Ordering::Equal).then(a.cmp(&b))
    };
    if take > 0 && take < idx.len() {
        idx.select_nth_unstable_by(take - 1, cmp);
    }
    idx.truncate(take);
    idx.sort_unstable_by(cmp);
    let values = idx.iter().map(|&i| logits[i as usize]).collect();
    Ok((idx, values))
}

pub fn top_candidates(logits: &[f32], count: usize) -> (Vec<u32>, Vec<f32>) {
    try_top_candidates(logits, count).unwrap_or_else(|e| panic!("top_candidates failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_in_open_interval() {
        for p in 0..1000u64 {
            let u = uniform(7, p, p * 31);
            assert!(u > 0.0 && u < 1.0);
        }
    }

    #[test]
    fn ties_resolve_by_id_before_noise() {
        // Equal logits and top_k 3: the kept set must be the three lowest ids, never 7 or 9.
        let vals = [1.0f32; 5];
        let ids = [9u32, 3, 1, 2, 7];
        let s = Sampling { top_k: 3, top_p: 1.0, ..Default::default() };
        for p in 0..200 {
            let t = choose(&vals, &ids, p, &s);
            assert!(t == 1 || t == 2 || t == 3, "got {t}");
        }
    }

    #[test]
    fn candidate_order_is_irrelevant() {
        let vals = [0.3f32, 2.0, -1.0, 2.0, 0.9];
        let ids = [4u32, 8, 1, 2, 6];
        let s = Sampling { seed: 99, ..Default::default() };
        let mut rv = vals;
        rv.reverse();
        let mut ri = ids;
        ri.reverse();
        for p in 0..100 {
            assert_eq!(choose(&vals, &ids, p, &s), choose(&rv, &ri, p, &s));
        }
    }

    #[test]
    fn near_zero_temperature_is_argmax() {
        let vals = [0.1f32, 5.0, 4.9, -2.0];
        let ids = [0u32, 1, 2, 3];
        let s = Sampling { temperature: 0.0, ..Default::default() };
        for p in 0..100 {
            assert_eq!(choose(&vals, &ids, p, &s), 1);
        }
    }

    #[test]
    fn greedy_is_argmax_with_lowest_id_on_ties() {
        let s = Sampling { temperature: 0.0, top_k: 0, ..Default::default() };
        // three tied maxima: the lowest id wins at every position and seed, regardless of order
        let vals = [1.0f32, 3.0, 3.0, 3.0, 0.0];
        let ids = [10u32, 7, 4, 9, 1];
        for p in 0..200 {
            assert_eq!(choose(&vals, &ids, p, &Sampling { seed: p * 3, ..s }), 4);
        }
        // -0.0 and 0.0 tie, as in numpy
        assert_eq!(choose(&[-0.0f32, 0.0], &[9, 2], 0, &s), 2);
    }

    #[test]
    fn invalid_inputs_are_errors_not_panics() {
        let ok = Sampling::default();
        assert_eq!(try_choose(&[], &[], 0, &ok), Err(SamplingError::NoCandidates));
        assert_eq!(try_choose(&[1.0], &[1, 2], 0, &ok), Err(SamplingError::LengthMismatch { values: 1, ids: 2 }));
        assert_eq!(try_choose(&[1.0, f32::NAN], &[1, 2], 0, &ok), Err(SamplingError::NanLogit));
        for bad in [
            Sampling { min_p: 1.5, ..ok },
            Sampling { min_p: f64::INFINITY, ..ok },
            Sampling { min_p: -0.1, ..ok },
            Sampling { top_p: 1.5, ..ok },
            Sampling { top_p: f64::NAN, ..ok },
            Sampling { temperature: -1.0, ..ok },
            Sampling { temperature: f64::NAN, ..ok },
            Sampling { temperature: f64::INFINITY, ..ok },
        ] {
            assert!(matches!(try_choose(&[1.0, 2.0], &[1, 2], 0, &bad), Err(SamplingError::InvalidParams(_))), "{bad:?}");
        }
        assert_eq!(try_top_candidates(&[0.0, f32::NAN], 1), Err(SamplingError::NanLogit));
        // min_p of exactly 1 keeps only the top candidates and never panics
        let t = try_choose(&[1.0, 2.0, 0.5], &[1, 2, 3], 0, &Sampling { min_p: 1.0, ..ok });
        assert_eq!(t, Ok(2));
    }

    #[test]
    fn unit_open_never_reaches_the_ends() {
        assert!(unit_open(0) > 0.0);
        assert!(unit_open(u64::MAX) < 1.0, "the all-ones pattern must not round to 1.0");
        assert!(unit_open(u64::MAX - 0x7FF) < 1.0);
        assert!((-(unit_open(u64::MAX).ln())).ln().is_finite());
    }

    #[test]
    fn top_candidates_matches_a_full_sort() {
        let mut r = 12345u64;
        for n in [1usize, 2, 7, 50, 300] {
            let logits: Vec<f32> = (0..n)
                .map(|_| {
                    r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((r >> 60) as f32) - 8.0 // many exact ties
                })
                .collect();
            let ids: Vec<u32> = (0..n as u32).collect();
            let full = order(&logits, &ids);
            for count in [0usize, 1, 5, n, n + 3] {
                let (got_ids, got_vals) = top_candidates(&logits, count);
                let take = count.min(n);
                assert_eq!(got_ids, full[..take].iter().map(|&i| i as u32).collect::<Vec<_>>());
                assert_eq!(got_vals, full[..take].iter().map(|&i| logits[i]).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn seed_for_streams_the_same_bytes_as_the_string_form() {
        let tokens = [0u32, 7, 4294967295, 12, 100];
        for salt in [0u64, 9, u64::MAX] {
            let s = format!("{}|{}", tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","), salt);
            let d = Sha256::digest(s.as_bytes());
            let want = u64::from_le_bytes(d[..8].try_into().unwrap()) & ((1u64 << 63) - 1);
            assert_eq!(seed_for(&tokens, salt), want);
        }
        let d = Sha256::digest(b"|5");
        assert_eq!(seed_for(&[], 5), u64::from_le_bytes(d[..8].try_into().unwrap()) & ((1u64 << 63) - 1));
    }

    #[test]
    #[should_panic(expected = "differ in length")]
    fn choose_rows_refuses_ragged_input() {
        choose_rows(&[&[1.0]], &[&[1]], &[0, 1], &Sampling::default());
    }

    #[test]
    fn pairwise_matches_naive_on_exact_values() {
        let a: Vec<f64> = (0..300).map(|i| i as f64).collect();
        assert_eq!(pairwise_sum(&a), (0..300).sum::<i64>() as f64);
    }

    /// A sampler setting that reads the tokens generated so far would make a draft-tree node's sample depend on its own root path (docs/
    /// TREE_DRAFTING.md): the keyed sampler reads only these five fields, the row's own candidates and its absolute position. Adding a field
    /// breaks this destructuring on purpose: decide how every tree node gets its own path's history, or how such a request avoids trees,
    /// before extending it.
    #[test]
    fn the_sampler_reads_no_generated_history() {
        let Sampling { seed: _, temperature: _, top_k: _, top_p: _, min_p: _ } = Sampling::default();
        // and the draw of a row depends on nothing but (seed, position, candidates): the same row at the same position gives the same
        // token whatever else was sampled before it, which is what lets siblings share the draw of their depth
        let s = Sampling { seed: 9, temperature: 1.0, top_k: 8, top_p: 1.0, min_p: 0.0 };
        let ids = [3u32, 5, 7, 9];
        let vals = [1.0f32, 0.9, 0.8, 0.7];
        let a = choose(&vals, &ids, 123, &s);
        let _other = choose(&[5.0, 1.0, 1.0, 1.0], &ids, 77, &s);
        assert_eq!(choose(&vals, &ids, 123, &s), a);
    }
}
