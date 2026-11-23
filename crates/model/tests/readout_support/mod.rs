//! Checks shared by the real-pack tests of the teacher-forced readout (`prompt_logprobs`).
#![allow(dead_code)]
use gb_engine::score::RowRecords;

/// Everything in the records, as bits (so a NaN or a -0 cannot hide a difference).
pub fn bits(r: &RowRecords) -> (usize, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let b = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    (r.k, b(&r.lse), b(&r.chosen), r.top_ids.clone(), b(&r.top_logits))
}

/// The largest |a - b| over the chosen tokens' log-probabilities (`chosen - lse`) of two readouts of one prompt, and the first
/// row where any bit differs.
pub fn compare(a: &RowRecords, b: &RowRecords) -> (f64, Option<usize>) {
    assert_eq!((a.rows(), a.k), (b.rows(), b.k));
    let lp = |r: &RowRecords, i: usize| r.chosen[i] as f64 - r.lse[i] as f64;
    let max = (0..a.rows().saturating_sub(1)).map(|i| (lp(a, i) - lp(b, i)).abs()).fold(0f64, f64::max);
    let k = a.k;
    let first = (0..a.rows()).find(|&i| {
        a.lse[i].to_bits() != b.lse[i].to_bits()
            || a.chosen[i].to_bits() != b.chosen[i].to_bits()
            || a.top_ids[i * k..(i + 1) * k] != b.top_ids[i * k..(i + 1) * k]
            || a.top_logits[i * k..(i + 1) * k]
                .iter()
                .zip(&b.top_logits[i * k..(i + 1) * k])
                .any(|(x, y)| x.to_bits() != y.to_bits())
    });
    (max, first)
}

/// What every row of a readout of `prompt` must satisfy by itself: finite values, the top entries sorted by (logit descending,
/// id ascending) with distinct ids, a normaliser no smaller than the top logit, and the next token among the top entries exactly
/// when its rank is below `k` (it ranks ahead of the last entry, or is that entry).
pub fn check_rows(rec: &RowRecords, prompt: &[u32], vocab: usize) {
    let (n, k) = (prompt.len(), rec.k);
    assert_eq!((rec.rows(), rec.chosen.len(), rec.top_ids.len(), rec.top_logits.len()), (n, n, n * k, n * k));
    assert_eq!(k, 20.min(vocab));
    for i in 0..n {
        let (ids, top) = (&rec.top_ids[i * k..(i + 1) * k], &rec.top_logits[i * k..(i + 1) * k]);
        assert!(rec.lse[i].is_finite() && top.iter().all(|v| v.is_finite()), "row {i}: non-finite");
        assert!(ids.iter().all(|&t| (t as usize) < vocab), "row {i}: an id outside the vocabulary");
        for j in 1..k {
            assert!(
                top[j - 1] > top[j] || (top[j - 1] == top[j] && ids[j - 1] < ids[j]),
                "row {i}: top entries {j} out of order"
            );
        }
        assert!(rec.lse[i] >= top[0], "row {i}: the normaliser is below the top logit");
        let mass: f64 = top.iter().map(|&v| (v as f64 - rec.lse[i] as f64).exp()).sum();
        assert!(mass <= 1.0 + 1e-4, "row {i}: the top {k} hold probability {mass}");
        if i + 1 == n {
            assert_eq!(rec.chosen[i], f32::NEG_INFINITY, "the last row has no next token");
            continue;
        }
        let (tok, own) = (prompt[i + 1], rec.chosen[i]);
        assert!(own.is_finite(), "row {i}: the next token's logit");
        let ranked = own > top[k - 1] || (own == top[k - 1] && tok <= ids[k - 1]);
        match ids.iter().position(|&t| t == tok) {
            Some(j) => {
                assert!(ranked && top[j].to_bits() == own.to_bits(), "row {i}: token {tok} is top entry {j} with another value")
            }
            None => assert!(!ranked, "row {i}: token {tok} ranks within the top {k} and is not among them"),
        }
    }
}

/// Row `i` of `rec` against the full logits row `row` of the same position: the next token's logit and the top entries bit for
/// bit, and the rank rule against a full sort. Returns |device log-probability - host f64 log-softmax| of the next token (0 for
/// the last row) and |device lse - host f64 lse|.
pub fn check_against_row(rec: &RowRecords, i: usize, row: &[f32], next: Option<u32>) -> (f64, f64) {
    let k = rec.k;
    let peak = row.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v as f64));
    let lse = peak + row.iter().map(|&v| (v as f64 - peak).exp()).sum::<f64>().ln();
    let mut order: Vec<u32> = (0..row.len() as u32).collect();
    order.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]).then(a.cmp(&b)));
    assert_eq!(&rec.top_ids[i * k..(i + 1) * k], &order[..k], "row {i}: top ids against a full sort");
    for j in 0..k {
        assert_eq!(rec.top_logits[i * k + j].to_bits(), row[order[j] as usize].to_bits(), "row {i}: top logit {j}");
    }
    let d_lse = (rec.lse[i] as f64 - lse).abs();
    let Some(tok) = next else { return (0.0, d_lse) };
    assert_eq!(rec.chosen[i].to_bits(), row[tok as usize].to_bits(), "row {i}: the next token's logit");
    let rank = order.iter().position(|&t| t == tok).unwrap();
    assert_eq!(rec.top_ids[i * k..(i + 1) * k].contains(&tok), rank < k, "row {i}: token {tok} has rank {rank}");
    (((rec.chosen[i] as f64 - rec.lse[i] as f64) - (row[tok as usize] as f64 - lse)).abs(), d_lse)
}
