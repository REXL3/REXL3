//! Interleaved CPU-only real HTTP histogram recording gate; same process/router/client.
//!
//! The claim: switching histogram recording on adds no material latency to a real HTTP completion, at the median and in the tail.
//!
//! The statistics are paired, so the gate holds on a loaded machine (the workspace's parallel CPU test leg), not only on an idle one.
//! Each round sends two identical requests back to back, one with recording off and one with it on, and keeps d = on - off for the
//! pair. The two requests are about a millisecond apart and see the same load, so the load cancels inside d. The order alternates every
//! round (off first, then on first: a crossover design), so whatever the first request of a pair does to the second (a warm cache, a
//! burst that starts mid-pair) enters d with opposite signs in the two orders and cancels too: with no overhead, d is distributed
//! symmetrically about 0 however busy the machine is.
//!
//! * Median overhead: the mean of the medians of d in the two orders. Off first, d = delta + X; on first, d = delta - X', where delta
//!   is the recording overhead and X, X' are second-minus-first effects of the same distribution; the two medians are delta + m and
//!   delta - m, and their mean is delta. Each median is taken within one order, where d is unimodal, and its standard error is set by
//!   the spread of d near its centre (most pairs see no preemption: a few microseconds), not by the spread of the latencies. Until
//!   v0.1.1 the gate compared the median of all `on` latencies with the median of all `off` latencies: two independent estimates, each
//!   carrying the whole load-dependent spread (about 1.25 sigma / sqrt(n), sigma in the hundreds of microseconds under load), whose
//!   difference reached 61 us (10.5 % of the median) in the gate's full-workspace leg while the paired median of the same run was 1.8 us.
//! * Tail overhead: a sign test on the large excursions. E+ counts the pairs whose recorded request was more than T = 500 us slower
//!   than its unrecorded partner, E- the pairs where it was more than T faster. With no overhead the two counts have the same
//!   expectation at any load (a burst of the scheduler hits either arm, by the symmetry above), whatever the size of the bursts; an
//!   overhead of more than T on a fraction f of the recorded requests adds about f n to E+. The gate: E+ - E- <= 5 % of the pairs (the
//!   old p95 claim, "at most 5 % of recorded requests pay more than 500 us extra") plus three standard deviations of the sign noise,
//!   6 sqrt(E-): under no overhead each count has a variance of up to twice its mean (the alternating order puts two same-arm requests
//!   back to back at every pair boundary, so one burst can make two excursions of one sign), the difference up to four times it, and
//!   E- estimates that mean without being inflated by an overhead (which only adds to E+). On an idle machine E- is 0 and the bound is
//!   the old 5 %; under load the bound widens with the noise this run measured, so the test loses power there but never fails on the
//!   load alone. It replaces on_p95 - off_p95, the difference of two independent tail quantiles. Measured on GB10 (2026-10-16), 3 idle
//!   runs and 10 under a scheduler-saturating load (20 busy loops and a parallel test binary beside the test): every run passed;
//!   under the load on_p95 - off_p95 ranged from -485 to +642 us (the old gate would have failed) and q95(d) + q05(d) from -681 to
//!   +503 us, while E+ - E- stayed within +-12 against bounds of 62 to 105. Injected into the recorded runs offline, a 25 us overhead
//!   on every recorded request fails the median gate in all 13 runs, and 600 us on one recorded request in twelve fails the tail gate
//!   in all 3 idle runs (and in 2 of the 10 saturated ones).
//!
//! Bounds, the same sizes as before on the paired estimates, relative to the baseline measured in the same run: median overhead at most
//! 100 us and at most 5 % of the median `off` latency; net tail excursions at most 5 % of the pairs plus the noise term. The unpaired
//! figures and the paired quantiles are still printed.
mod common;
use common::{completion_body, http_timeout, start, Opts};
use std::time::{Duration, Instant};

/// Rounds measured (one pair each, half in each order), after [`WARMUP`] rounds that are dropped.
const PAIRS: usize = 1000;
/// The first rounds, dropped: the router's, the runtime's and the allocator's first requests.
const WARMUP: usize = 50;
/// The tail bound: a recorded request more than this slower than its unrecorded partner is a tail excursion (microseconds).
const TAIL_US: f64 = 500.0;

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q) as usize]
}

#[test]
fn interleaved_histogram_recording_overhead_gate() {
    let s = start(Opts::default());
    let body = completion_body(&[5; 10], 3, false);
    let mut off = Vec::new();
    let mut on = Vec::new();
    // d = on - off of each pair: all of them, and split by the order the pair ran in
    let mut pairs = Vec::new();
    let mut off_first = Vec::new();
    let mut on_first = Vec::new();
    let mut raw_pairs = Vec::new();
    for round in 0..WARMUP + PAIRS {
        let mut times = [0.0; 2];
        for enabled in if round % 2 == 0 { [false, true] } else { [true, false] } {
            s.handle().metrics.observability.set_recording(enabled);
            let at = Instant::now();
            let r = http_timeout(s.port, "POST", "/v1/completions", Some(&body), Duration::from_secs(5));
            times[usize::from(enabled)] = at.elapsed().as_secs_f64() * 1e6;
            assert_eq!(r.status, 200);
        }
        if round >= WARMUP {
            let d = times[1] - times[0];
            raw_pairs.push(times);
            off.push(times[0]);
            on.push(times[1]);
            pairs.push(d);
            let order = if round % 2 == 0 { &mut off_first } else { &mut on_first };
            order.push(d);
        }
    }
    s.handle().metrics.observability.set_recording(true);
    assert_eq!((off_first.len(), on_first.len()), (PAIRS / 2, PAIRS / 2));
    // the gated estimates (module comment)
    let (off_first50, on_first50) = (pct(&mut off_first, 0.50), pct(&mut on_first, 0.50));
    let overhead50 = (off_first50 + on_first50) / 2.0;
    let slower = pairs.iter().filter(|&&d| d > TAIL_US).count();
    let faster = pairs.iter().filter(|&&d| d < -TAIL_US).count();
    let net = slower as f64 - faster as f64;
    let net_bound = 0.05 * PAIRS as f64 + 6.0 * (faster as f64).sqrt();
    // the baseline measured in the same run, and the figures printed but not gated
    let (pair05, pair50, pair95) = (pct(&mut pairs, 0.05), pct(&mut pairs, 0.50), pct(&mut pairs, 0.95));
    let (off50, on50) = (pct(&mut off, 0.50), pct(&mut on, 0.50));
    let (off95, on95) = (pct(&mut off, 0.95), pct(&mut on, 0.95));
    println!("raw_cpu_http_pairs_off_on_us={}", serde_json::to_string(&raw_pairs).unwrap());
    println!(
        "CPU histogram recording gate pairs={PAIRS} warmup={WARMUP} overhead_p50_us={overhead50:.3} (off_first_p50_us={off_first50:.3} \
         on_first_p50_us={on_first50:.3}) overhead_pct_of_off_p50={:.3} tail_slower_{TAIL_US}us={slower} tail_faster_{TAIL_US}us={faster} \
         tail_net={net} tail_net_bound={net_bound:.1} paired_p05_us={pair05:.3} paired_p50_us={pair50:.3} paired_p95_us={pair95:.3} \
         off_p50_us={off50:.3} on_p50_us={on50:.3} unpaired_p50_delta_us={:.3} off_p95_us={off95:.3} on_p95_us={on95:.3} \
         unpaired_p95_delta_us={:.3}; gated: overhead_p50 <= 100us AND <= 5% of off_p50; tail_net <= 5% of pairs + 6 sqrt(tail_faster)",
        overhead50 / off50 * 100.0,
        on50 - off50,
        on95 - off95
    );
    assert!(
        overhead50 <= 100.0 && overhead50 <= off50 * 0.05,
        "histogram recording median overhead exceeded the CPU gate: {overhead50:.1} us against an off median of {off50:.1} us"
    );
    assert!(
        net <= net_bound,
        "histogram recording tail overhead exceeded the CPU gate: {slower} recorded requests more than {TAIL_US} us slower than their \
         pair, {faster} faster: net {net} against {net_bound:.1}"
    );
}
