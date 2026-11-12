//! The numbers the arbiter prices a round with, measured on the real 88-layer model with the MTP head loaded, and printed as the Rust
//! source of `spec.rs`'s constants (`VERIFY_MS`, `MTP_COST` / `MTP_COST_RESTRICTED`), so the tables are regenerated, not retyped, when the
//! step gets faster (the perf workers' MoE and step work).
//!
//! GPU, `#[ignore]` and env-gated (`REXL3_NEMO_OMNI_DIR`, `REXL3_NEMO_ORACLE_DIR`: the oracle's text is the real prefix; without them the
//! test prints "not run" and passes). One model per process, under the GPU lock:
//!
//! ```text
//! REXL3_NEMO_OMNI_DIR=$HOME/projects/rexl3-nemotron3.5-super-vl-exl3-mtp REXL3_NEMO_ORACLE_DIR=$HOME/projects/nemo-oracle/ref-text-1k \
//!   ~/rexl3-ab/agents-tools/nrun1 costs --gpu -- cargo test --release -p gb-model --test nemotron_omni_costs -- --ignored --nocapture --test-threads=1
//! ```
//!
//! * `REXL3_COSTS_WIDE=1` also measures the verify windows of 16, 32 and 64 rows (the table `VERIFY_MS` keeps up to 64; about 6 s each);
//! * `REXL3_MTP_VOCAB` (the load-time switch): the shared model of these tests runs the FULL head when it is unset (the table printed is
//!   `MTP_COST`'s); `default` builds the restricted head of the default 49,152-id list and `<file>` that file's, and the table printed is
//!   then `MTP_COST_RESTRICTED`'s (the one in `spec.rs` is the default list's).
//!
//! The fit and the formatting are plain functions with CPU tests (`cargo test -p gb-model --test nemotron_omni_costs`): they reproduce
//! the committed constants from the measurements they were fitted to.

mod nemotron_omni_mtp_support;

use gb_model::families::nemotron_h_omni::mtp::DraftAsk;
use gb_model::families::nemotron_h_omni::spec::{MtpCost, MTP_COST};
use nemotron_omni_mtp_support::*;

/// One measured draft call: `asks` streams each absorbing `rows_per_ask` pending rows and drafting `depth` deep, in milliseconds.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub asks: usize,
    pub rows_per_ask: usize,
    pub depth: usize,
    pub ms: f64,
}

/// `MtpCost`'s numbers fitted to a set of cells.
#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    pub base: f64,
    pub per_stream: f64,
    /// (total pending rows, milliseconds), ascending.
    pub rows: Vec<(usize, f64)>,
    pub lm_head: f64,
    pub step: f64,
    pub step_per_stream: f64,
    /// Largest and mean relative error of the model against the cells.
    pub worst: f64,
    pub mean: f64,
}

impl Fit {
    /// Milliseconds the fitted model gives a call, with the table read the way `MtpCost::ms` reads it.
    pub fn ms(&self, asks: usize, rows: usize, depth: usize) -> f64 {
        let m = MtpCost {
            base: self.base,
            per_stream: self.per_stream,
            rows: std::borrow::Cow::Owned(self.rows.clone()),
            lm_head: self.lm_head,
            step: self.step,
            step_per_stream: self.step_per_stream,
        };
        m.ms(asks, rows, depth)
    }
}

/// The least-squares form of `MtpCost`, in closed form:
/// * a chained step costs `lm_head + step + step_per_stream x streams`: the increments from one depth to the next of every (asks, rows)
///   cell give it, a line through the asks (`step_per_stream` is its slope, `step` its intercept less the stage-measured `lm_head`);
/// * what is left of a call, `base + rows(R)`, is read off every cell at its total pending rows `R = asks x rows_per_ask` (the mean over
///   the depths): `base` is taken as given (a launch and a sync: it cannot be told from the table's first point), and `per_stream` is 0
///   (the per-stream costs sit in the chained steps and the rows; a measurement that separates them can set it).
///
/// `lm_head` is the measured stage time of the `lm_head` GEMV (the profile of a 1-ask, 1-row, depth-1 call).
pub fn fit(cells: &[Cell], lm_head: f64, base: f64) -> Result<Fit, String> {
    if cells.is_empty() {
        return Err("no cells".into());
    }
    // the chained step's cost by streams: the mean increment over every (rows, depth step) at that many asks
    let mut asks: Vec<usize> = cells.iter().map(|c| c.asks).collect();
    asks.sort_unstable();
    asks.dedup();
    let mut points: Vec<(f64, f64)> = Vec::new();
    for &s in &asks {
        let mut incs = Vec::new();
        for c in cells.iter().filter(|c| c.asks == s && c.depth >= 2) {
            if let Some(prev) = cells.iter().find(|p| p.asks == s && p.rows_per_ask == c.rows_per_ask && p.depth + 1 == c.depth) {
                incs.push(c.ms - prev.ms);
            }
        }
        if !incs.is_empty() {
            points.push((s as f64, incs.iter().sum::<f64>() / incs.len() as f64));
        }
    }
    if points.is_empty() {
        return Err("no two depths of one (asks, rows) cell: the chained step cannot be told".into());
    }
    // a line through (asks, increment): intercept is lm_head + step, slope step_per_stream
    let (chain_fixed, slope) = if points.len() == 1 {
        (points[0].1, 0.0)
    } else {
        let n = points.len() as f64;
        let (sx, sy) = (points.iter().map(|p| p.0).sum::<f64>(), points.iter().map(|p| p.1).sum::<f64>());
        let (sxx, sxy) = (points.iter().map(|p| p.0 * p.0).sum::<f64>(), points.iter().map(|p| p.0 * p.1).sum::<f64>());
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        ((sy - slope * sx) / n, slope.max(0.0))
    };
    let step = (chain_fixed - lm_head).max(0.0);
    // the rest of every cell: what the absorb's layers cost at its total pending rows
    let mut by_rows: std::collections::BTreeMap<usize, Vec<f64>> = Default::default();
    for c in cells {
        let chained = (c.depth.saturating_sub(1)) as f64 * (lm_head + step + slope * c.asks as f64);
        let r = c.ms - lm_head - chained - base;
        by_rows.entry(c.asks * c.rows_per_ask).or_default().push(r.max(0.0));
    }
    let rows: Vec<(usize, f64)> = by_rows.into_iter().map(|(r, v)| (r, v.iter().sum::<f64>() / v.len() as f64)).collect();
    let mut f = Fit { base, per_stream: 0.0, rows, lm_head, step, step_per_stream: slope, worst: 0.0, mean: 0.0 };
    let errs: Vec<f64> = cells.iter().map(|c| (f.ms(c.asks, c.asks * c.rows_per_ask, c.depth) - c.ms).abs() / c.ms).collect();
    f.worst = errs.iter().cloned().fold(0.0, f64::max);
    f.mean = errs.iter().sum::<f64>() / errs.len() as f64;
    Ok(f)
}

/// The fit of the draft call since the absorb runs its attention and MoE over each sequence's LAST row only (`mtp.rs`, `Tail::Rows`): the
/// absorb no longer costs by the pending rows it takes (eh_proj and the keys and values read their weights once whatever the rows) but by
/// the sequences, whose last rows go through the MoE. So the call is `base + per_stream * s + r0 + k * sqrt(R) + lm_head + (d - 1) (lm_head +
/// step + step_per_stream * s)`: the chained step as in [`fit`] (the increments between depths), and the absorb's three terms by least
/// squares over the cells' residuals (`k` multiplies the square root of the pending rows `R`: within noise of zero, clipped to a small
/// positive value so the table stays ascending and concave in rows as the arbiter's shape tests require).
pub fn fit_flat(cells: &[Cell], lm_head: f64, base: f64) -> Result<Fit, String> {
    let chain = fit(cells, lm_head, base)?;
    let obs: Vec<(f64, f64, f64)> = cells
        .iter()
        .map(|c| {
            let chained = (c.depth.saturating_sub(1)) as f64 * (lm_head + chain.step + chain.step_per_stream * c.asks as f64);
            (c.asks as f64, ((c.asks * c.rows_per_ask) as f64).sqrt(), c.ms - lm_head - base - chained)
        })
        .collect();
    // normal equations of r = p s + k sqrt(R) + r0 (`obs.1` is sqrt(R))
    let n = obs.len() as f64;
    let (ss, sr, s1) = (
        obs.iter().map(|o| o.0 * o.0).sum::<f64>(),
        obs.iter().map(|o| o.0 * o.1).sum::<f64>(),
        obs.iter().map(|o| o.0).sum::<f64>(),
    );
    let (rr, r1) = (obs.iter().map(|o| o.1 * o.1).sum::<f64>(), obs.iter().map(|o| o.1).sum::<f64>());
    let (ys, yr, y1) = (
        obs.iter().map(|o| o.0 * o.2).sum::<f64>(),
        obs.iter().map(|o| o.1 * o.2).sum::<f64>(),
        obs.iter().map(|o| o.2).sum::<f64>(),
    );
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let a = [[ss, sr, s1], [sr, rr, r1], [s1, r1, n]];
    let d = det(a);
    if d.abs() < 1e-9 {
        return Err("the cells do not vary in both streams and rows: the absorb's terms cannot be told apart".into());
    }
    let col = |i: usize, y: [f64; 3]| {
        let mut m = a;
        for (row, v) in m.iter_mut().zip(y) {
            row[i] = v;
        }
        det(m) / d
    };
    let y = [ys, yr, y1];
    let (p, k, r0) = (col(0, y).max(0.0), col(1, y).max(0.01), col(2, y));
    // with `k` pinned the others are refitted by the one-dimensional normal equations on what is left
    let rest: Vec<(f64, f64)> = obs.iter().map(|o| (o.0, o.2 - k * o.1)).collect();
    let (ms, my) = (rest.iter().map(|o| o.0).sum::<f64>() / n, rest.iter().map(|o| o.1).sum::<f64>() / n);
    let var = rest.iter().map(|o| (o.0 - ms) * (o.0 - ms)).sum::<f64>();
    let p = if var > 1e-12 { (rest.iter().map(|o| (o.0 - ms) * (o.1 - my)).sum::<f64>() / var).max(0.0) } else { p };
    let r0 = (my - p * ms).max(0.0).max(if r0.is_finite() { 0.0 } else { my });
    let rows: Vec<(usize, f64)> = [1usize, 2, 3, 4, 8, 16, 24, 32].into_iter().map(|r| (r, r0 + k * (r as f64).sqrt())).collect();
    let mut f = Fit {
        base,
        per_stream: p,
        rows,
        lm_head,
        step: chain.step,
        step_per_stream: chain.step_per_stream,
        worst: 0.0,
        mean: 0.0,
    };
    let errs: Vec<f64> = cells.iter().map(|c| (f.ms(c.asks, c.asks * c.rows_per_ask, c.depth) - c.ms).abs() / c.ms).collect();
    f.worst = errs.iter().cloned().fold(0.0, f64::max);
    f.mean = errs.iter().sum::<f64>() / errs.len() as f64;
    Ok(f)
}

/// The fit as the Rust source of a `MtpCost` constant.
pub fn cost_source(name: &str, f: &Fit) -> String {
    let rows: Vec<String> = f.rows.iter().map(|(r, t)| format!("({r}, {t:.2})")).collect();
    format!(
        "pub const {name}: MtpCost = MtpCost {{\n    base: {:.2},\n    per_stream: {:.2},\n    rows: Cow::Borrowed(&[{}]),\n    lm_head: {:.2},\n    step: {:.2},\n    step_per_stream: {:.2},\n}};",
        f.base,
        f.per_stream,
        rows.join(", "),
        f.lm_head,
        f.step,
        f.step_per_stream
    )
}

/// A verify table as the Rust source of `VERIFY_MS`.
pub fn verify_source(table: &[(usize, f64)]) -> String {
    verify_source_named("VERIFY_MS", table)
}

/// A verify table as the Rust source of the constant `name` (the loaded form's: `spec::HeadForm::table_names`).
pub fn verify_source_named(name: &str, table: &[(usize, f64)]) -> String {
    let pts: Vec<String> = table.iter().map(|(r, t)| format!("({r}, {t:.0}.0)")).collect();
    format!("pub const {name}: &[(usize, f64)] = &[{}];", pts.join(", "))
}

// ---------------------------------------------------------------------------------------------------------------------------------
// CPU tests of the fit and the format.

const PUBLISHED_FULL: [((usize, usize), [f64; 2]); 20] = [
    ((1, 1), [7.2, 7.1]),
    ((1, 2), [14.1, 14.3]),
    ((1, 3), [21.8, 20.9]),
    ((1, 4), [27.7, 29.1]),
    ((1, 5), [34.6, 34.7]),
    ((2, 1), [7.7, 8.0]),
    ((2, 2), [15.6, 15.5]),
    ((2, 3), [23.5, 23.9]),
    ((2, 4), [30.7, 31.0]),
    ((2, 5), [38.1, 39.2]),
    ((3, 1), [9.0, 9.2]),
    ((3, 2), [16.8, 16.5]),
    ((3, 3), [25.1, 24.7]),
    ((3, 4), [33.1, 35.1]),
    ((3, 5), [40.7, 41.4]),
    ((4, 1), [8.8, 9.0]),
    ((4, 2), [18.1, 17.5]),
    ((4, 3), [26.9, 25.4]),
    ((4, 4), [34.9, 34.0]),
    ((4, 5), [43.5, 42.5]),
];

const PUBLISHED_RESTRICTED: [((usize, usize), [f64; 2]); 20] = [
    ((1, 1), [4.2, 4.1]),
    ((1, 2), [8.1, 8.2]),
    ((1, 3), [12.2, 12.3]),
    ((1, 4), [16.2, 16.3]),
    ((1, 5), [20.3, 20.3]),
    ((2, 1), [4.7, 5.0]),
    ((2, 2), [9.4, 9.3]),
    ((2, 3), [14.4, 13.7]),
    ((2, 4), [19.0, 17.8]),
    ((2, 5), [24.0, 22.3]),
    ((3, 1), [5.7, 5.7]),
    ((3, 2), [11.0, 10.3]),
    ((3, 3), [16.4, 14.8]),
    ((3, 4), [21.6, 19.3]),
    ((3, 5), [26.5, 24.4]),
    ((4, 1), [5.8, 6.0]),
    ((4, 2), [12.0, 11.0]),
    ((4, 3), [18.1, 16.3]),
    ((4, 4), [23.3, 21.8]),
    ((4, 5), [29.2, 28.2]),
];

fn cells_of(data: &[((usize, usize), [f64; 2])]) -> Vec<Cell> {
    let rows = [1usize, 8];
    let mut v = Vec::new();
    for &((asks, depth), t) in data {
        for (r, ms) in rows.into_iter().zip(t) {
            v.push(Cell { asks, rows_per_ask: r, depth, ms });
        }
    }
    v
}

/// The measurements of the committed `MTP_COST` (this test's own output on node 1, 2026-10-11, the full head, 256-token context, the final
/// kernels): asks x depth at 1 / 8 pending rows per ask.
fn published() -> Vec<Cell> {
    cells_of(&PUBLISHED_FULL)
}

/// The same for the shipped `MTP_COST_RESTRICTED` (the default 49,152-id head).
fn published_restricted() -> Vec<Cell> {
    cells_of(&PUBLISHED_RESTRICTED)
}

#[test]
fn the_flat_fit_reproduces_the_restricted_constants_from_the_default_lists_measurements() {
    use gb_model::families::nemotron_h_omni::spec::MTP_COST_RESTRICTED as M;
    let f = fit_flat(&published_restricted(), M.lm_head, 0.10).unwrap();
    assert!((f.step - M.step).abs() < 0.05, "step {} against {}", f.step, M.step);
    assert!((f.step_per_stream - M.step_per_stream).abs() < 0.05, "{}", f.step_per_stream);
    assert!((f.per_stream - M.per_stream).abs() < 0.05, "per stream {} against {}", f.per_stream, M.per_stream);
    assert_eq!(f.rows.len(), M.rows.len(), "{:?}", f.rows);
    for (&(r, t), &(r2, t2)) in f.rows.iter().zip(M.rows.iter()) {
        assert!(r == r2 && (t - t2).abs() < 0.08, "R {r}: fitted {t:.3}, shipped R {r2} {t2:.2}");
    }
    assert!(f.worst < 0.075 && f.mean < 0.035, "worst {:.3} mean {:.3}", f.worst, f.mean);
    let s = cost_source("MTP_COST_RESTRICTED", &f);
    assert!(s.starts_with("pub const MTP_COST_RESTRICTED: MtpCost = MtpCost {") && s.contains("lm_head: 1.74,"), "{s}");
}

#[test]
fn the_flat_fit_reproduces_the_committed_constants_from_the_measurements_they_were_fitted_to() {
    let f = fit_flat(&published(), MTP_COST.lm_head, 0.10).unwrap();
    // the committed constants are this fit of the measurements before their printing to two decimals (the cells above are rounded)
    assert!((f.step - MTP_COST.step).abs() < 0.05, "step {} against {}", f.step, MTP_COST.step);
    assert!((f.step_per_stream - MTP_COST.step_per_stream).abs() < 0.05, "{}", f.step_per_stream);
    assert!((f.per_stream - MTP_COST.per_stream).abs() < 0.05, "per stream {} against {}", f.per_stream, MTP_COST.per_stream);
    assert_eq!(f.rows.len(), MTP_COST.rows.len(), "{:?}", f.rows);
    for (&(r, t), &(r2, t2)) in f.rows.iter().zip(MTP_COST.rows.iter()) {
        assert!(r == r2 && (t - t2).abs() < 0.08, "R {r}: fitted {t:.3}, committed R {r2} {t2:.2}");
    }
    assert!(f.worst < 0.075 && f.mean < 0.03, "worst {:.3} mean {:.3}", f.worst, f.mean);
    // a fit of one stream count has no per-stream slope to give
    let one: Vec<Cell> = published().into_iter().filter(|c| c.asks == 1).collect();
    assert_eq!(fit(&one, MTP_COST.lm_head, 0.1).unwrap().step_per_stream, 0.0);
    assert!(fit_flat(&one, MTP_COST.lm_head, 0.1).is_err(), "one stream count cannot tell the per-stream absorb from the base");
    // depth 1 alone cannot say what a chained step costs
    let shallow: Vec<Cell> = published().into_iter().filter(|c| c.depth == 1).collect();
    assert!(fit(&shallow, MTP_COST.lm_head, 0.1).is_err());
    assert!(fit(&[], MTP_COST.lm_head, 0.1).is_err());
}

#[test]
fn the_printed_source_is_the_format_of_the_constants() {
    let f = fit_flat(&published(), MTP_COST.lm_head, 0.10).unwrap();
    let s = cost_source("MTP_COST", &f);
    assert!(s.starts_with("pub const MTP_COST: MtpCost = MtpCost {\n    base: 0.10,\n    per_stream: 0.6"), "{s}");
    assert!(
        s.contains("rows: Cow::Borrowed(&[(1, ")
            && s.contains("lm_head: 4.63,")
            && s.contains("step_per_stream: ")
            && s.ends_with("};"),
        "{s}"
    );
    assert_eq!(
        verify_source(&[(1, 269.4), (2, 374.0), (8, 829.2)]),
        "pub const VERIFY_MS: &[(usize, f64)] = &[(1, 269.0), (2, 374.0), (8, 829.0)];"
    );
    // the names a measurement is printed under are the loaded form's constants
    use gb_model::families::nemotron_h_omni::spec::HeadForm;
    let n = HeadForm::MixedK.table_names();
    assert_eq!(verify_source_named(n.verify, &[(1, 40.2)]), "pub const VERIFY_MS_MIXEDK: &[(usize, f64)] = &[(1, 40.0)];");
    let f = fit_flat(&published(), MTP_COST.lm_head, 0.10).unwrap();
    assert!(cost_source(n.mtp, &f).starts_with("pub const MTP_COST_MIXEDK: MtpCost = MtpCost {"));
    assert_eq!(HeadForm::Exl3.table_names().mtp_restricted, "MTP_COST_RESTRICTED_H4");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The measurement.

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore]
fn print_the_cost_tables_in_the_format_of_the_constants() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    let restricted = m.mtp_restricted();
    m.set_mtp_restricted(restricted);
    let wide = std::env::var_os("REXL3_COSTS_WIDE").is_some();
    // the loaded pack's form names the constants the measurement is printed as (v0.1.2: the MixedK form has its own tables)
    let form = m.mtp_head_form();
    let names = form.table_names();
    eprintln!(
        "== the model: {} head, {} slots, form {} (tables {} / {} / {})",
        if restricted { "restricted" } else { "full" },
        SLOTS,
        form.label(),
        names.verify,
        names.mtp,
        names.mtp_restricted
    );

    // ---- the verify window: one sequence at a context of 256 tokens, `rows` rows of real text, committed in full, flushed (the commit
    // applied inside the next forward is in the time), the median of 7 after one warm-up
    const CTX: usize = 256;
    let mut rows_list = vec![1usize, 2, 3, 4, 5, 6, 7, 8];
    if wide {
        rows_list.extend([16, 32, 64]);
    }
    let mut verify = Vec::new();
    for &rows in &rows_list {
        let first = prefill(&mut m, 0, &prefix[..CTX], 64);
        let _ = first;
        let mut times = Vec::new();
        for it in 0..8usize {
            let from = CTX + it * rows;
            let tokens = prefix[from..from + rows].to_vec();
            let b = gb_model::rows::RowBatch::from_requests(&[gb_engine::RowRequest {
                slot: 0,
                first_position: from as u64,
                tokens,
            }]);
            let t = std::time::Instant::now();
            m.window_candidates(&b, 28).unwrap();
            m.commit(0, rows);
            m.flush().unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if it >= 1 {
                times.push(ms);
            }
        }
        let ms = median(&mut times);
        eprintln!("verify window, {rows:2} rows: {ms:8.1} ms");
        verify.push((rows, ms));
        m.end_slot(0);
    }
    eprintln!("\n// verify window, ms: one sequence at ~{CTX} tokens, median of 7");
    eprintln!("{}", verify_source_named(names.verify, &verify));
    if !wide {
        eprintln!("// (rows above 8: REXL3_COSTS_WIDE=1; keep the previous points until they are measured)");
    }

    // ---- the draft call: asks 1..4 x pending rows per ask 1 and 8 x depth 1..5: the wall time of `mtp_draft` (absorb + chain + one sync),
    // the median of 7 after 2 warm-ups. The pending rows are real taps of the oracle's text; every ask's tokens are its own window of it.
    m.end_slot(0);
    m.begin_slot(0).unwrap();
    window_argmax(&mut m, 0, &prefix[400..408], 0);
    let taps = m.mtp_tap_rows(8).unwrap();
    m.end_slot(0);
    const FIRST: usize = 256;
    let slots = 4usize;
    let mut cells: Vec<Cell> = Vec::new();
    eprintln!("\ndraft call wall time, ms (median of 7 after 2 warm-ups); pending rows per ask 1 / 8");
    for asks in 1..=slots {
        for depth in 1..=5usize {
            let mut seqs: Vec<Vec<u32>> = (0..slots)
                .map(|s| {
                    m.end_slot(s);
                    m.begin_slot(s).unwrap();
                    prefix[s * 37..s * 37 + FIRST + 1].to_vec()
                })
                .collect();
            let mut out = Vec::new();
            for rows in [1usize, 8] {
                let mut times = Vec::new();
                for it in 0..9usize {
                    for (s, seq) in seqs.iter_mut().enumerate().take(asks) {
                        let from = s * 37 + (it % 7) * 5;
                        *seq = prefix[from..from + FIRST + rows].to_vec();
                        m.mtp_debug_fill_store(s, FIRST, &taps[..rows * 4096]).unwrap();
                    }
                    let ask: Vec<DraftAsk> = (0..asks).map(|s| DraftAsk { slot: s, tokens: &seqs[s], depth }).collect();
                    let t = std::time::Instant::now();
                    let r = m.mtp_draft(&ask, &[]).unwrap();
                    let dt = t.elapsed().as_secs_f64() * 1e3;
                    assert!(r.iter().all(|d| d.as_ref().is_some_and(|d| d.ids.len() == depth)), "every ask drafts");
                    if it >= 2 {
                        times.push(dt);
                    }
                }
                let ms = median(&mut times);
                cells.push(Cell { asks, rows_per_ask: rows, depth, ms });
                out.push(ms);
            }
            eprintln!("{asks} asks depth {depth}: rows 1 {:6.1}   rows 8 {:6.1}", out[0], out[1]);
        }
        for s in 0..slots {
            m.end_slot(s);
        }
    }

    // ---- the lm_head's stage time: one ask, one row, depth 1 (the profile times the stages with events and syncs between them)
    m.set_mtp_profile(true);
    let calls = 5usize;
    for it in 0..calls + 2 {
        m.end_slot(0);
        m.begin_slot(0).unwrap();
        let seq = prefix[..FIRST + 1].to_vec();
        m.mtp_debug_fill_store(0, FIRST, &taps[..4096]).unwrap();
        if it == 2 {
            m.take_mtp_profile();
        }
        m.mtp_draft(&[DraftAsk { slot: 0, tokens: &seq, depth: 1 }], &[]).unwrap();
    }
    let stages = m.take_mtp_profile();
    m.set_mtp_profile(false);
    m.end_slot(0);
    for (name, ms) in &stages {
        eprintln!("stage {name:40} {:6.2} ms a call", ms / calls as f64);
    }
    let lm_head = stages.iter().filter(|(n, _)| n.starts_with("lm_head")).map(|(_, ms)| ms / calls as f64).sum::<f64>();
    assert!(lm_head > 0.0, "the profile named no lm_head stage: {stages:?}");

    // ---- the fit, as source
    let f = fit_flat(&cells, lm_head, 0.10).unwrap();
    let name = if restricted { names.mtp_restricted } else { names.mtp };
    eprintln!(
        "\n// draft call (MtpCost), fitted to the {} cells above: worst {:.1}%, mean {:.1}% off",
        cells.len(),
        100.0 * f.worst,
        100.0 * f.mean
    );
    eprintln!("{}", cost_source(name, &f));
    eprintln!("// the rows table beyond 32 total pending rows is the last slope: measure with more asks to extend it");
    if form == gb_model::families::nemotron_h_omni::spec::HeadForm::MixedK {
        eprintln!(
            "// MixedK form: paste these over the UNMEASURED placeholders in spec.rs and set MIXEDK_MEASURED = true (both heads: run once \
             with REXL3_MTP_VOCAB=default and once with it unset)"
        );
    }
}
