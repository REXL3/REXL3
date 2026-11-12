//! The speculation arbiter: per round and per stream, which drafter's proposal to verify (the family's multi-token-prediction head, the copy
//! drafter, or none), chosen by committed tokens per model-millisecond, with periodic probing so that no estimate goes stale.
//!
//! It owns no model and no clock: the caller hands in what a round would cost (`ArbiterCost`: the verify window's time by its sequences and its
//! total rows, which on an MoE model grows with the experts the rows touch, and the head's draft call) and reports what happened (`observe`).
//! Everything here is a pure function of those inputs, so it is tested on the CPU, and mutated: each rule below has a test that fails without it.
//!
//! **What it estimates.** For each stream and source a [`DepthCurve`]: the conditional acceptance of the j-th draft given the j-1 before it were
//! accepted (the engine accepts a draft while it equals the keyed sample, so a round commits `1 + a` tokens when the first `a` drafts match), as an
//! exponentially weighted ratio with a prior. Expected tokens of a `d`-deep offer are `1 + a1 + a1 a2 + ...`.
//!
//! **What it chooses.** Per stream one of `None`, `Copy(l)` (the copy drafter's `l` tokens) or `Mtp(d)`; the objective is the whole round's expected
//! tokens over its time, `verify_ms_of(streams, total rows) + mtp_ms(...)`, maximised by coordinate ascent from "nobody drafts" (every step
//! changes one stream's pick to the one that raises the rate most, until none does; deterministic, ties to fewer rows). A stream that must make
//! room in its store of pending rows absorbs them whatever it picks.
//!
//! **Why it probes.** An estimate only moves when its source is verified. A source judged poor would never be tried again, and a stream whose text
//! changes character (prose, then a code block that repeats) would stay with the wrong drafter. So after `probe_every` rounds without the source a
//! stream is offered a cheap one-deep probe of it, and the estimate learns. The probe is the cost of never being stuck: at most one cheap round in
//! `probe_every`, and the gap doubles after every probe that fails (up to `probe_max_gap`), so a source that is useless on this text costs almost
//! nothing to keep checking.

/// The most draft depths a curve tracks.
pub const MAX_DEPTH: usize = 8;

/// A drafter's proposal for one stream, as a choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    /// No drafts: one serial row.
    None,
    /// The copy drafter's first `n` tokens.
    Copy(usize),
    /// The multi-token-prediction head, `d` deep.
    Mtp(usize),
}

impl Choice {
    /// Rows this stream puts in the window beyond its base row.
    pub fn drafts(self) -> usize {
        match self {
            Choice::None => 0,
            Choice::Copy(n) | Choice::Mtp(n) => n,
        }
    }
}

/// Conditional acceptance of the j-th draft (1-based) given the first j - 1 were accepted: `(hits + prior * w) / (trials + w)`, the counts decayed
/// each time the depth is observed so the estimate follows the text.
#[derive(Clone, Debug)]
pub struct DepthCurve {
    hits: [f64; MAX_DEPTH],
    trials: [f64; MAX_DEPTH],
    prior: [f64; MAX_DEPTH],
    /// With a chain ratio `r` the prior of depth `j >= 2` is `r` times the CURRENT estimate of depth `j - 1` instead of a fixed number: a depth that has
    /// never been offered inherits what the depths before it have shown. A fixed prior makes a pessimistic estimate self-confirming (depth 4 is never
    /// offered, so it never learns it would have been right) and an optimistic one costly (deep offers on a drafter that is poor on this text).
    chain: Option<f64>,
    /// What a round that did not offer depth `j` does to the evidence on it (counts times `fade`): a depth's estimate only moves when it is offered, so
    /// a few unlucky early observations would keep it below the break-even for good (measured on the math prompt: depth 2 was offered 3 times out of
    /// 196 rounds, where it is worth 12%). Forgetting hands the estimate back to the chain prior, which follows the depths that ARE being observed, and
    /// the depth is tried again; a head that really is poor at that depth is found out again at the price of one deeper row.
    fade: f64,
    weight: f64,
    decay: f64,
}

impl DepthCurve {
    /// `prior[j - 1]` for the first depths (the last value is carried, shrinking by 5% a depth, to the rest), `weight` pseudo-trials of it, `decay` per
    /// observation of a depth.
    pub fn new(prior: &[f64], weight: f64, decay: f64) -> Self {
        assert!(!prior.is_empty() && weight > 0.0 && (0.0..=1.0).contains(&decay));
        let mut p = [0.0; MAX_DEPTH];
        for (j, v) in p.iter_mut().enumerate() {
            *v = prior.get(j).copied().unwrap_or_else(|| prior[prior.len() - 1] * 0.95f64.powi((j + 1 - prior.len()) as i32));
        }
        Self { hits: [0.0; MAX_DEPTH], trials: [0.0; MAX_DEPTH], prior: p, chain: None, fade: 1.0, weight, decay }
    }

    /// The prior of every depth beyond the first is `ratio` times the estimate of the depth before it (see [`DepthCurve`]).
    pub fn with_chain(mut self, ratio: f64) -> Self {
        assert!(ratio > 0.0 && ratio <= 1.0);
        self.chain = Some(ratio);
        self
    }

    /// A round that did not offer a depth multiplies the evidence on it by `fade` (see [`DepthCurve`]).
    pub fn with_fade(mut self, fade: f64) -> Self {
        assert!(fade > 0.0 && fade <= 1.0);
        self.fade = fade;
        self
    }

    /// The estimated acceptance of the j-th draft (`j >= 1`).
    pub fn alpha(&self, j: usize) -> f64 {
        let i = j.clamp(1, MAX_DEPTH) - 1;
        let prior = match self.chain {
            Some(r) if i > 0 => self.alpha(i) * r,
            _ => self.prior[i],
        };
        ((self.hits[i] + prior * self.weight) / (self.trials[i] + self.weight)).clamp(0.0, 0.999)
    }

    /// Expected tokens a round commits with `d` drafts offered: `1 + a1 + a1 a2 + ...`.
    pub fn expected_tokens(&self, d: usize) -> f64 {
        let (mut sum, mut p) = (1.0, 1.0);
        for j in 1..=d {
            p *= self.alpha(j);
            sum += p;
        }
        sum
    }

    /// A round verified `verified` drafts and `accepted` of them matched: depth j was reached when the j - 1 before it all matched, and hit when it did.
    pub fn observe(&mut self, verified: usize, accepted: usize) {
        for j in 1..=verified.min(MAX_DEPTH) {
            if accepted + 1 >= j {
                let i = j - 1;
                self.hits[i] = self.hits[i] * self.decay + f64::from(u8::from(accepted >= j));
                self.trials[i] = self.trials[i] * self.decay + 1.0;
            }
        }
        for i in verified.min(MAX_DEPTH)..MAX_DEPTH {
            self.hits[i] *= self.fade;
            self.trials[i] *= self.fade;
        }
    }
}

/// What a round costs, measured by the caller (milliseconds).
pub trait ArbiterCost {
    /// The verify window over `rows` rows in all (every stream's base row and drafts). On an MoE model this grows with the experts the rows pick,
    /// not with the number of streams.
    fn verify_ms(&self, rows: usize) -> f64;
    /// The verify window of `streams` sequences over `rows` rows in all: what [`Arbiter::choose`] prices a round with. The default is
    /// `verify_ms(rows)`, a table by total rows whatever the sequences (every cost model before this method was one), so a model that does
    /// not override it prices exactly as it did. A model whose step costs something per SEQUENCE (a recurrent state read and written, an
    /// attention over each sequence's own context) overrides it: in a step of many sequences the base rows then cost more than a table of
    /// one sequence's rows says, and a draft row less in proportion.
    fn verify_ms_of(&self, streams: usize, rows: usize) -> f64 {
        let _ = streams;
        self.verify_ms(rows)
    }
    /// The head's draft call for `streams` streams that absorb `absorb_rows` pending rows in all and chain `depth` drafts (`depth == 0`: absorb only).
    fn mtp_ms(&self, streams: usize, absorb_rows: usize, depth: usize) -> f64;
}

/// What a stream offers this round.
#[derive(Clone, Copy, Debug)]
pub struct Ask {
    /// The stream (the slot): its estimates are kept under this index.
    pub stream: usize,
    /// Drafts the engine would take at most (it also caps by what the sequence has left).
    pub max: usize,
    /// Tokens the copy drafter proposes (0: no match).
    pub copy_len: usize,
    /// Whether the head can draft for this stream (its drafter state is intact and a hidden row is pending).
    pub mtp_ok: bool,
    /// Pending rows the head would absorb if it drafted now.
    pub absorb_rows: usize,
    /// The store of pending rows would overflow after another window: absorb now, whatever else is chosen.
    pub must_absorb: bool,
}

/// The arbiter's settings.
#[derive(Clone, Debug)]
pub struct ArbiterConfig {
    pub allow_mtp: bool,
    pub allow_copy: bool,
    /// The deepest MTP chain offered, and the depth a stream is held to until it has `warmup_rounds` verified head rounds behind it: a new stream's
    /// estimate is the optimistic prior, and a deep offer that turns out wrong costs several rows of verify and a chain of draft calls (a drafter that is
    /// broken on this text would cost 40% more per wasted round at depth 4 than at depth 2).
    pub max_mtp_depth: usize,
    pub warmup_depth: usize,
    pub warmup_rounds: u32,
    /// Rows one window holds at most (all streams).
    pub max_rows: usize,
    /// A draft pick must beat plain serial rows by this fraction of the round's rate before the arbiter takes it (the estimates are noisy and the
    /// largest estimate is the likeliest to be an over-estimate: near break-even the safe choice is not to draft).
    pub min_gain: f64,
    /// Rounds without a source before a stream is offered a probe of it, and the longest the gap grows to when probes keep failing.
    pub probe_every: u32,
    pub probe_max_gap: u32,
    /// Acceptance priors by depth and their weight in pseudo-trials, and the decay per observation.
    pub mtp_prior: Vec<f64>,
    /// How much of the previous depth's acceptance the head keeps one depth further (the prior of an unobserved depth, `DepthCurve::with_chain`).
    pub mtp_chain: f64,
    /// Per round that did not offer a depth, the factor on the evidence about it (`DepthCurve::with_fade`).
    pub mtp_fade: f64,
    pub copy_prior: Vec<f64>,
    pub prior_weight: f64,
    pub decay: f64,
}

impl Default for ArbiterConfig {
    /// The NextN head's conditional acceptance on the GLM pack: 0.86 for the first draft, and each depth keeps 0.87 of the one before until it has been
    /// observed (measured: the increments of tokens per round over K = 1..4 on the six classes of the oracle's prompts fall by 0.79-0.9 a depth in prose
    /// and code and not at all on a repeat; the first version of this prior was the oracle's flat 0.82-0.88, which bought depth 3-4 where K = 2 was the
    /// better choice, and a fixed 0.86 / 0.74 / 0.64 / 0.56 never offered depth 4 on a repeat, where it is worth 49 tok/s against 41), and a copy match's
    /// (a hit of 8 or more tokens usually continues).
    fn default() -> Self {
        Self {
            allow_mtp: true,
            allow_copy: true,
            max_mtp_depth: 4,
            warmup_depth: 2,
            warmup_rounds: 4,
            max_rows: 64,
            min_gain: 0.02,
            probe_every: 16,
            probe_max_gap: 256,
            mtp_prior: vec![0.86],
            mtp_chain: 0.87,
            mtp_fade: 0.96,
            copy_prior: vec![0.75, 0.8, 0.8, 0.8],
            prior_weight: 3.0,
            decay: 0.93,
        }
    }
}

#[derive(Clone, Debug)]
struct StreamState {
    mtp: DepthCurve,
    copy: DepthCurve,
    /// Rounds since the stream last verified drafts of the source.
    since_mtp: u32,
    since_copy: u32,
    /// Rounds of the head this stream has verified (to end the warm-up).
    mtp_rounds: u32,
    /// The gap before the next probe of the source (doubles after a probe that did not make the source worth picking, back to `probe_every` after
    /// one that did), whether the source's last pick was a probe, and whether a probe's result waits to be judged at the next `choose`.
    gap_mtp: u32,
    gap_copy: u32,
    probing_mtp: bool,
    probing_copy: bool,
    judge_mtp: bool,
    judge_copy: bool,
}

/// Counters of what the arbiter did (for the logs and the tests).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArbiterStats {
    /// Picks by kind: none, copy, mtp.
    pub picks: [u64; 3],
    /// Probes forced, and absorbs forced to make room.
    pub probes: u64,
    pub forced_absorbs: u64,
}

/// Whether the head may draft for `a` under `ar`'s settings.
fn opts_has_mtp(ar: &Arbiter, a: &Ask) -> bool {
    ar.cfg.allow_mtp && a.mtp_ok && a.max >= 1
}

pub struct Arbiter {
    pub cfg: ArbiterConfig,
    streams: Vec<StreamState>,
    pub stats: ArbiterStats,
}

impl Arbiter {
    pub fn new(cfg: ArbiterConfig) -> Self {
        Self { cfg, streams: Vec::new(), stats: ArbiterStats::default() }
    }

    fn fresh(&self) -> StreamState {
        StreamState {
            mtp: DepthCurve::new(&self.cfg.mtp_prior, self.cfg.prior_weight, self.cfg.decay)
                .with_chain(self.cfg.mtp_chain)
                .with_fade(self.cfg.mtp_fade),
            copy: DepthCurve::new(&self.cfg.copy_prior, self.cfg.prior_weight, self.cfg.decay),
            since_mtp: 0,
            since_copy: 0,
            mtp_rounds: 0,
            gap_mtp: self.cfg.probe_every,
            gap_copy: self.cfg.probe_every,
            probing_mtp: false,
            probing_copy: false,
            judge_mtp: false,
            judge_copy: false,
        }
    }

    fn state(&mut self, stream: usize) -> &mut StreamState {
        while self.streams.len() <= stream {
            let f = self.fresh();
            self.streams.push(f);
        }
        &mut self.streams[stream]
    }

    /// A new sequence in `stream`: its estimates start from the priors.
    pub fn begin(&mut self, stream: usize) {
        let f = self.fresh();
        *self.state(stream) = f;
    }

    /// The estimated expected tokens of a pick for a stream.
    pub fn expected_tokens(&mut self, stream: usize, c: Choice) -> f64 {
        let s = self.state(stream);
        match c {
            Choice::None => 1.0,
            Choice::Copy(n) => s.copy.expected_tokens(n),
            Choice::Mtp(d) => s.mtp.expected_tokens(d),
        }
    }

    /// The estimated acceptance of the j-th MTP draft of a stream (a test's and the logs' view).
    pub fn mtp_alpha(&mut self, stream: usize, j: usize) -> f64 {
        self.state(stream).mtp.alpha(j)
    }

    /// A round of `stream` verified `verified` drafts of `choice`'s source and `accepted` matched.
    pub fn observe(&mut self, stream: usize, choice: Choice, verified: usize, accepted: usize) {
        let s = self.state(stream);
        s.since_mtp = s.since_mtp.saturating_add(1);
        s.since_copy = s.since_copy.saturating_add(1);
        if verified == 0 {
            // the engine kept none of the offer: nothing was learned, and a probe that was cut is simply asked again
            s.probing_mtp = false;
            s.probing_copy = false;
            return;
        }
        // what a probe was worth is judged at the next `choose`: does the updated estimate make the source worth picking on its own?
        match choice {
            Choice::None => {}
            Choice::Copy(_) => {
                s.copy.observe(verified, accepted);
                s.since_copy = 0;
                s.judge_copy |= std::mem::take(&mut s.probing_copy);
            }
            Choice::Mtp(_) => {
                s.mtp.observe(verified, accepted);
                s.since_mtp = 0;
                s.mtp_rounds = s.mtp_rounds.saturating_add(1);
                s.judge_mtp |= std::mem::take(&mut s.probing_mtp);
            }
        }
    }

    /// The deepest chain this stream may be offered now (the warm-up holds a new stream to `warmup_depth`).
    fn depth_cap(&self, stream: usize) -> usize {
        let seen = self.streams.get(stream).map_or(0, |s| s.mtp_rounds);
        if seen < self.cfg.warmup_rounds {
            self.cfg.max_mtp_depth.min(self.cfg.warmup_depth)
        } else {
            self.cfg.max_mtp_depth
        }
    }

    /// Every feasible pick of one ask, fewest rows first.
    fn options(&self, a: &Ask) -> Vec<Choice> {
        let mut v = vec![Choice::None];
        if self.cfg.allow_copy {
            for n in 1..=a.copy_len.min(a.max) {
                v.push(Choice::Copy(n));
            }
        }
        if self.cfg.allow_mtp && a.mtp_ok {
            for d in 1..=self.depth_cap(a.stream).min(a.max) {
                v.push(Choice::Mtp(d));
            }
        }
        v
    }

    /// Tokens and milliseconds of the round if `asks` take `picks`.
    fn rate(&mut self, costs: &dyn ArbiterCost, asks: &[Ask], picks: &[Choice]) -> (f64, f64) {
        let mut tokens = 0.0;
        let mut rows = 0usize;
        let (mut s_mtp, mut absorb, mut depth) = (0usize, 0usize, 0usize);
        for (a, &c) in asks.iter().zip(picks) {
            tokens += self.expected_tokens(a.stream, c);
            rows += 1 + c.drafts();
            let absorbs = matches!(c, Choice::Mtp(_)) || a.must_absorb;
            if absorbs {
                s_mtp += 1;
                absorb += a.absorb_rows;
                if let Choice::Mtp(d) = c {
                    depth = depth.max(d);
                }
            }
        }
        // every ask is one sequence of the window (the engine asks for the streams that may draft; a stream with nothing left to draft rides
        // its base row unasked, which this count leaves out)
        let mut ms = costs.verify_ms_of(asks.len(), rows);
        if s_mtp > 0 {
            ms += costs.mtp_ms(s_mtp, absorb, depth);
        }
        (tokens, ms)
    }

    /// The pick for every ask this round. `asks` hold at most `cfg.max_rows` base rows.
    pub fn choose(&mut self, costs: &dyn ArbiterCost, asks: &[Ask]) -> Vec<Choice> {
        let opts: Vec<Vec<Choice>> = asks.iter().map(|a| self.options(a)).collect();
        let mut picks = vec![Choice::None; asks.len()];
        let (t0, m0) = self.rate(costs, asks, &picks);
        let mut best = t0 / m0 * (1.0 + self.cfg.min_gain);
        // starting points beyond "nobody drafts": the head's call has fixed costs (launch, the layer's weights) that the streams share, so one stream's
        // draft can look like a loss where all of them drafting is a gain, and a search that moves one stream at a time would never leave serial.
        // Every uniform depth of the head, and the copy drafter's whole offer, are tried as starts (each must beat serial by the margin)
        let mut starts: Vec<Vec<Choice>> = (1..=self.cfg.max_mtp_depth)
            .map(|d| {
                asks.iter()
                    .map(|a| {
                        if opts_has_mtp(self, a) {
                            Choice::Mtp(d.min(a.max).min(self.depth_cap(a.stream)))
                        } else {
                            Choice::None
                        }
                    })
                    .collect()
            })
            .collect();
        starts.push(
            asks.iter()
                .map(|a| {
                    if self.cfg.allow_copy && a.copy_len.min(a.max) > 0 {
                        Choice::Copy(a.copy_len.min(a.max))
                    } else {
                        Choice::None
                    }
                })
                .collect(),
        );
        for start in starts {
            if start.iter().map(|c| 1 + c.drafts()).sum::<usize>() > self.cfg.max_rows {
                continue;
            }
            let (t, m) = self.rate(costs, asks, &start);
            if t / m > best * (1.0 + 1e-9) {
                best = t / m;
                picks = start;
            }
        }
        let mut rows: usize = picks.iter().map(|c| 1 + c.drafts()).sum();
        loop {
            // the single change that raises the round's rate most
            let mut top: Option<(usize, Choice, f64)> = None;
            for (i, o) in opts.iter().enumerate() {
                for &c in o {
                    if c == picks[i] {
                        continue;
                    }
                    let new_rows = rows - picks[i].drafts() + c.drafts();
                    if new_rows > self.cfg.max_rows {
                        continue;
                    }
                    let old = picks[i];
                    picks[i] = c;
                    let (t, m) = self.rate(costs, asks, &picks);
                    picks[i] = old;
                    let r = t / m;
                    // strictly better, and on a tie the change with fewer rows
                    let better = match top {
                        None => r > best * (1.0 + 1e-9),
                        Some((_, tc, tr)) => r > tr * (1.0 + 1e-9) || ((r - tr).abs() <= tr * 1e-9 && c.drafts() < tc.drafts()),
                    };
                    if better {
                        top = Some((i, c, r));
                    }
                }
            }
            let Some((i, c, r)) = top else { break };
            rows = rows - picks[i].drafts() + c.drafts();
            picks[i] = c;
            best = r;
        }
        // the probes of earlier rounds are judged by what the search just chose without them: a source that is picked on its own now earns the normal
        // gap again, one that is not has its gap doubled (a useless source is checked ever more rarely)
        for (a, &c) in asks.iter().zip(&picks) {
            let (base, max_gap) = (self.cfg.probe_every, self.cfg.probe_max_gap);
            let s = self.state(a.stream);
            if std::mem::take(&mut s.judge_mtp) {
                s.gap_mtp = if matches!(c, Choice::Mtp(_)) { base } else { s.gap_mtp.saturating_mul(2).min(max_gap) };
            }
            if std::mem::take(&mut s.judge_copy) {
                s.gap_copy = if matches!(c, Choice::Copy(_)) { base } else { s.gap_copy.saturating_mul(2).min(max_gap) };
            }
        }
        // probes: a stream that has not verified a source for `probe_every` rounds is offered a cheap one-deep try of it
        for (i, a) in asks.iter().enumerate() {
            let (mtp_due, copy_due) = {
                let s = self.state(a.stream);
                (s.since_mtp >= s.gap_mtp, s.since_copy >= s.gap_copy)
            };
            let room = |rows: usize, add: usize| rows + add <= self.cfg.max_rows;
            // a probe only stands in for "no drafts": a stream that already drafts is learning the source it drafts with
            if self.cfg.allow_mtp && a.mtp_ok && a.max >= 1 && mtp_due && matches!(picks[i], Choice::None) && room(rows, 1) {
                picks[i] = Choice::Mtp(1);
                rows += 1;
                self.stats.probes += 1;
                self.state(a.stream).probing_mtp = true;
                continue;
            }
            if self.cfg.allow_copy
                && a.copy_len >= 1
                && a.max >= 1
                && copy_due
                && matches!(picks[i], Choice::None)
                && room(rows, 1)
            {
                picks[i] = Choice::Copy(a.copy_len.min(a.max).min(2));
                rows += picks[i].drafts();
                self.stats.probes += 1;
                self.state(a.stream).probing_copy = true;
            }
        }
        for (a, &c) in asks.iter().zip(&picks) {
            self.stats.picks[match c {
                Choice::None => 0,
                Choice::Copy(_) => 1,
                Choice::Mtp(_) => 2,
            }] += 1;
            if a.must_absorb && !matches!(c, Choice::Mtp(_)) {
                self.stats.forced_absorbs += 1;
            }
        }
        picks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cost model with the GLM numbers of `families/glm/spec.rs`: the measured verify table and the measured draft-call fit.
    struct Glm;
    impl ArbiterCost for Glm {
        fn verify_ms(&self, rows: usize) -> f64 {
            const T: [(usize, f64); 15] = [
                (1, 44.9),
                (2, 54.4),
                (3, 65.1),
                (4, 72.4),
                (5, 81.0),
                (6, 91.1),
                (7, 96.9),
                (8, 103.8),
                (9, 110.4),
                (12, 127.5),
                (16, 150.6),
                (24, 192.5),
                (32, 223.4),
                (48, 274.9),
                (64, 320.8),
            ];
            if rows <= 1 {
                return T[0].1;
            }
            for w in T.windows(2) {
                if rows <= w[1].0 {
                    return w[0].1 + (w[1].1 - w[0].1) * (rows - w[0].0) as f64 / (w[1].0 - w[0].0) as f64;
                }
            }
            T[14].1 + (rows - 64) as f64 * 3.0
        }
        fn mtp_ms(&self, streams: usize, absorb_rows: usize, depth: usize) -> f64 {
            const R: [(usize, f64); 8] = [(1, 0.4), (2, 0.8), (3, 1.15), (5, 2.0), (8, 3.2), (16, 5.6), (32, 9.0), (64, 13.1)];
            let rows = {
                let (mut r0, mut t0) = (0usize, 0.0f64);
                let mut out = None;
                for &(r1, t1) in &R {
                    if absorb_rows <= r1 {
                        out = Some(t0 + (t1 - t0) * (absorb_rows - r0) as f64 / (r1 - r0) as f64);
                        break;
                    }
                    (r0, t0) = (r1, t1);
                }
                out.unwrap_or_else(|| 13.1 + (absorb_rows - 64) as f64 * 0.13)
            };
            let s = streams as f64;
            1.3 + 0.07 * s + rows + if depth >= 1 { 2.1 } else { 0.0 } + depth.saturating_sub(1) as f64 * (3.0 + 0.4 * s)
        }
    }

    fn ask(stream: usize) -> Ask {
        Ask { stream, max: 7, copy_len: 0, mtp_ok: true, absorb_rows: 2, must_absorb: false }
    }

    fn arb() -> Arbiter {
        Arbiter::new(ArbiterConfig::default())
    }

    /// The engine thread chooses for every stream of a round before the verify step: a round of 128 streams is a wide round (a wide
    /// decode window) and the search must not cost a visible part of a step of half a second.
    #[test]
    fn a_round_of_128_streams_is_chosen_in_milliseconds() {
        let mut a = Arbiter::new(ArbiterConfig { max_rows: 256, ..ArbiterConfig::default() });
        let asks: Vec<Ask> = (0..128).map(ask).collect();
        let t = std::time::Instant::now();
        let picks = a.choose(&Glm, &asks);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("arbiter: 128 streams chosen in {ms:.2} ms ({} picks)", picks.len());
        assert_eq!(picks.len(), 128);
        assert!(picks.iter().map(|c| 1 + c.drafts()).sum::<usize>() <= 256);
        // generous: this is a debug build too
        assert!(ms < 2000.0, "{ms} ms");
    }

    #[test]
    fn expected_tokens_are_the_geometric_sums_of_the_conditional_acceptances() {
        let c = DepthCurve::new(&[0.5], 1.0, 0.9);
        // alpha 0.5 at depth 1, 0.475 at depth 2 (the prior carried with a 5% shrink per depth)
        assert!((c.alpha(1) - 0.5).abs() < 1e-12);
        assert!((c.alpha(2) - 0.475).abs() < 1e-12);
        assert!((c.expected_tokens(0) - 1.0).abs() < 1e-12);
        assert!((c.expected_tokens(1) - 1.5).abs() < 1e-12);
        assert!((c.expected_tokens(2) - (1.0 + 0.5 + 0.5 * 0.475)).abs() < 1e-12);
    }

    #[test]
    fn an_unobserved_depth_inherits_what_the_depths_before_it_have_shown() {
        let mk = || DepthCurve::new(&[0.8], 2.0, 1.0).with_chain(0.9);
        // nothing observed: the chain from the first prior
        let c = mk();
        assert!((c.alpha(1) - 0.8).abs() < 1e-12);
        assert!((c.alpha(2) - 0.72).abs() < 1e-12 && (c.alpha(3) - 0.648).abs() < 1e-12);
        // a head that is right every time at depth 1: depth 2 and 3 are expected to be good too, so they get offered and can be tested
        let mut good = mk();
        for _ in 0..20 {
            good.observe(1, 1);
        }
        let mut bad = mk();
        for _ in 0..20 {
            bad.observe(1, 0);
        }
        assert!(good.alpha(1) > 0.95 && bad.alpha(1) < 0.1);
        assert!(good.alpha(3) > 0.75, "{}", good.alpha(3));
        assert!(bad.alpha(3) < 0.1, "{}", bad.alpha(3));
        // a depth that HAS been observed answers for itself, whatever the one before it says
        let mut mixed = mk();
        for _ in 0..30 {
            mixed.observe(2, 1); // depth 1 hit, depth 2 reached and missed
        }
        assert!(mixed.alpha(1) > 0.9 && mixed.alpha(2) < 0.1, "{} {}", mixed.alpha(1), mixed.alpha(2));
        // the plain curve (no chain) still carries a fixed prior
        assert!((DepthCurve::new(&[0.5], 1.0, 0.9).alpha(2) - 0.475).abs() < 1e-12);
    }

    #[test]
    fn a_depth_that_is_not_offered_forgets_its_bad_luck_and_gets_tried_again() {
        let mut c = DepthCurve::new(&[0.85], 3.0, 0.93).with_chain(0.87).with_fade(0.96);
        // three unlucky early rounds at depth 2 (depth 1 hit, depth 2 missed)
        for _ in 0..3 {
            c.observe(2, 1);
        }
        let low = c.alpha(2);
        assert!(low < 0.55, "{low}");
        // from then on only depth 1 is offered and it keeps hitting: depth 2's estimate must climb back to what the chain says, past any break-even
        for _ in 0..60 {
            c.observe(1, 1);
        }
        assert!(c.alpha(1) > 0.9, "{}", c.alpha(1));
        assert!(c.alpha(2) > 0.7, "{} (was {low})", c.alpha(2));
        // while a depth that is offered and misses keeps its evidence
        let mut d = DepthCurve::new(&[0.85], 3.0, 0.93).with_chain(0.87).with_fade(0.96);
        for _ in 0..60 {
            d.observe(2, 1);
        }
        assert!(d.alpha(2) < 0.2, "{}", d.alpha(2));
        // and a curve with no fade (the copy source's) never forgets
        let mut e = DepthCurve::new(&[0.85], 3.0, 0.93);
        for _ in 0..3 {
            e.observe(2, 1);
        }
        let before = e.alpha(2);
        for _ in 0..60 {
            e.observe(1, 1);
        }
        assert!((e.alpha(2) - before).abs() < 1e-12);
    }

    #[test]
    fn a_curve_learns_per_depth_and_only_counts_a_depth_that_was_reached() {
        let mut c = DepthCurve::new(&[0.5, 0.5, 0.5], 2.0, 1.0);
        // a round of three drafts where the first matched and the second did not: depth 1 hit, depth 2 reached and missed, depth 3 never reached
        c.observe(3, 1);
        assert!((c.alpha(1) - (1.0 + 1.0) / (1.0 + 2.0)).abs() < 1e-12, "{}", c.alpha(1));
        assert!((c.alpha(2) - (0.0 + 1.0) / (1.0 + 2.0)).abs() < 1e-12);
        assert!((c.alpha(3) - 0.5).abs() < 1e-12, "depth 3 was not reached, so it learned nothing");
        // plenty of misses at depth 1 pull the estimate far down; the prior is only a start
        for _ in 0..60 {
            c.observe(1, 0);
        }
        assert!(c.alpha(1) < 0.1);
        // the decay lets a long-gone run be forgotten
        let mut d = DepthCurve::new(&[0.5], 1.0, 0.8);
        for _ in 0..50 {
            d.observe(1, 0);
        }
        for _ in 0..20 {
            d.observe(1, 1);
        }
        assert!(d.alpha(1) > 0.7, "{}", d.alpha(1));
    }

    #[test]
    fn a_single_stream_with_the_measured_acceptance_drafts_two_or_three_deep() {
        let mut a = arb();
        let c = a.choose(&Glm, &[ask(0)]);
        // acceptance 0.85, 2 rows cost 54 ms against 45: drafting pays, and more than one deep
        assert!(matches!(c[0], Choice::Mtp(d) if d >= 2), "{:?}", c);
        // with its own acceptance low it does not draft at all: the round is cheaper without
        let mut a = arb();
        for _ in 0..40 {
            a.observe(0, Choice::Mtp(2), 2, 0);
        }
        let c = a.choose(&Glm, &[ask(0)]);
        assert_eq!(c[0], Choice::None, "an acceptance near zero must fall back to serial rows: {c:?}");
    }

    #[test]
    fn never_worse_than_serial_by_its_own_estimate() {
        // whatever the estimates, the pick's rate is at least that of "nobody drafts": the search starts there and only moves up
        for acc in [0.0, 0.2, 0.5, 0.9] {
            for streams in [1usize, 2, 4, 8] {
                let mut a = arb();
                for s in 0..streams {
                    for _ in 0..30 {
                        a.observe(s, Choice::Mtp(2), 2, (acc * 2.0) as usize);
                    }
                }
                let asks: Vec<Ask> = (0..streams).map(ask).collect();
                let picks = a.choose(&Glm, &asks);
                let serial = vec![Choice::None; streams];
                let (t, m) = a.rate(&Glm, &asks, &picks);
                let (t0, m0) = a.rate(&Glm, &asks, &serial);
                assert!(t / m >= t0 / m0 - 1e-12, "acc {acc}, {streams} streams: {picks:?}");
            }
        }
    }

    #[test]
    fn near_break_even_the_margin_keeps_it_serial() {
        // an acceptance whose estimated gain is under 2% is not worth acting on; the same stream with no margin does draft. The estimate is pinned (a
        // prior of a billion pseudo-trials) so that the numbers are exact: 0.36 for the first draft, 0.2 beyond, on the measured costs one deep gives
        // 1.36 tokens for 54.4 + 4.3 ms against 44.9 ms: +4%
        let single = |margin: f64| {
            let mut z = Arbiter::new(ArbiterConfig {
                min_gain: margin,
                mtp_prior: vec![0.36, 0.2],
                prior_weight: 1e9,
                ..ArbiterConfig::default()
            });
            z.choose(&Glm, &[ask(0)])[0]
        };
        assert_eq!(single(0.0), Choice::Mtp(1), "with no margin the estimate is acted on");
        assert_eq!(single(0.05), Choice::None, "with a margin it is not");
    }

    #[test]
    fn a_long_copy_match_beats_the_head_and_a_short_one_does_not() {
        let mut a = arb();
        // seven tokens of an exact repeat: learned to be accepted nearly always
        for _ in 0..30 {
            a.observe(0, Choice::Copy(7), 7, 7);
        }
        let mut k = ask(0);
        k.copy_len = 7;
        let c = a.choose(&Glm, &[k]);
        assert!(matches!(c[0], Choice::Copy(n) if n >= 5), "{c:?}");
        // a copy source that has been wrong every time is not taken even when it offers seven tokens
        let mut b = arb();
        for _ in 0..30 {
            b.observe(0, Choice::Copy(7), 7, 0);
        }
        let c = b.choose(&Glm, &[k]);
        assert!(!matches!(c[0], Choice::Copy(_)), "{c:?}");
    }

    /// A window whose rows each cost `per_row` ms on top of 45, a draft call that is free: the dearer the rows, the fewer drafts.
    struct Linear(f64);
    impl ArbiterCost for Linear {
        fn verify_ms(&self, rows: usize) -> f64 {
            45.0 + self.0 * (rows - 1) as f64
        }
        fn mtp_ms(&self, _: usize, _: usize, _: usize) -> f64 {
            0.0
        }
    }

    #[test]
    fn the_dearer_a_row_the_fewer_drafts_and_the_window_stays_inside_its_budget() {
        // past the warm-up (a new stream is held to a shallow chain: `a_new_stream_is_held_to_a_shallow_chain...`)
        let warm = || Arbiter::new(ArbiterConfig { warmup_rounds: 0, ..ArbiterConfig::default() });
        let drafts = |cost: &dyn ArbiterCost, streams: usize| -> usize {
            let mut a = warm();
            let asks: Vec<Ask> = (0..streams).map(ask).collect();
            a.choose(cost, &asks).iter().map(|c| c.drafts()).sum()
        };
        let (cheap, mid, dear, absurd) =
            (drafts(&Linear(1.0), 1), drafts(&Linear(12.0), 1), drafts(&Linear(30.0), 1), drafts(&Linear(300.0), 1));
        assert!(cheap >= mid && mid >= dear && dear >= absurd, "{cheap} {mid} {dear} {absurd}");
        assert!(cheap >= 3, "rows nearly free: draft deep, {cheap}");
        assert_eq!(absurd, 0, "a row costing seven serial steps is never worth a draft");
        // eight streams share one window: the same economics per row, and the row budget binds
        assert!(drafts(&Linear(12.0), 8) <= 8 * 7);
        let mut a = Arbiter::new(ArbiterConfig { max_rows: 12, warmup_rounds: 0, ..ArbiterConfig::default() });
        let asks: Vec<Ask> = (0..8).map(ask).collect();
        let picks = a.choose(&Linear(0.5), &asks);
        assert!(picks.iter().map(|c| 1 + c.drafts()).sum::<usize>() <= 12, "{picks:?}");
        assert!(picks.iter().map(|c| 1 + c.drafts()).sum::<usize>() >= 11, "nearly free rows fill the budget: {picks:?}");
    }

    /// A window of 45 ms and `per_row` ms a row (as `Linear`), and `per_stream` ms more for every sequence past the first when the model
    /// prices by sequences: the state each sequence reads and writes, which a table of one sequence's rows cannot see.
    struct PerStream {
        per_row: f64,
        per_stream: f64,
        asked: std::cell::Cell<usize>,
    }
    impl ArbiterCost for PerStream {
        fn verify_ms(&self, rows: usize) -> f64 {
            45.0 + self.per_row * (rows - 1) as f64
        }
        fn verify_ms_of(&self, streams: usize, rows: usize) -> f64 {
            self.asked.set(self.asked.get().max(streams));
            self.verify_ms(rows) + self.per_stream * streams.saturating_sub(1) as f64
        }
        fn mtp_ms(&self, _: usize, _: usize, _: usize) -> f64 {
            0.0
        }
    }

    #[test]
    fn a_round_is_priced_by_its_sequences_when_the_model_says_what_a_sequence_costs() {
        // the trait's default is the table by rows: every model written before the method prices as it did
        assert_eq!(Glm.verify_ms_of(16, 32).to_bits(), Glm.verify_ms(32).to_bits());
        assert_eq!(Linear(12.0).verify_ms_of(4, 9).to_bits(), Linear(12.0).verify_ms(9).to_bits());
        let warm = || Arbiter::new(ArbiterConfig { warmup_rounds: 0, ..ArbiterConfig::default() });
        let drafts = |per_stream: f64| -> (usize, usize) {
            let c = PerStream { per_row: 12.0, per_stream, asked: std::cell::Cell::new(0) };
            let asks: Vec<Ask> = (0..8).map(ask).collect();
            let d = warm().choose(&c, &asks).iter().map(|c| c.drafts()).sum();
            (d, c.asked.get())
        };
        let ((flat, asked), (dear, _)) = (drafts(0.0), drafts(40.0));
        // the round is priced with every ask as one sequence of the window
        assert_eq!(asked, 8);
        // a dearer base at the same cost a row: a draft row is a smaller share of the round, so it is worth more of them
        assert!(dear > flat, "{flat} drafts with no cost a sequence, {dear} with 40 ms a sequence");
        assert!(flat >= 1, "{flat}");
    }

    #[test]
    fn picks_respect_the_ask_and_the_mode() {
        let mut a = arb();
        let mut k = ask(0);
        k.max = 1;
        let c = a.choose(&Glm, &[k]);
        assert!(c[0].drafts() <= 1, "{c:?}");
        let mut k = ask(0);
        k.mtp_ok = false;
        assert!(
            !matches!(a.choose(&Glm, &[k])[0], Choice::Mtp(_)),
            "a stream whose drafter state is gone is not offered the head"
        );
        let mut only_copy = Arbiter::new(ArbiterConfig { allow_mtp: false, ..ArbiterConfig::default() });
        assert!(!matches!(only_copy.choose(&Glm, &[ask(0)])[0], Choice::Mtp(_)));
        let mut only_mtp = Arbiter::new(ArbiterConfig { allow_copy: false, ..ArbiterConfig::default() });
        let mut k = ask(0);
        k.copy_len = 7;
        assert!(!matches!(only_mtp.choose(&Glm, &[k])[0], Choice::Copy(_)));
    }

    #[test]
    fn a_source_judged_poor_is_probed_again_after_probe_every_rounds() {
        let mut a = Arbiter::new(ArbiterConfig { probe_every: 8, ..ArbiterConfig::default() });
        for _ in 0..40 {
            a.observe(0, Choice::Mtp(2), 2, 0);
        }
        // serial rounds: the head's acceptance estimate is near zero so nothing drafts ... until the probe is due
        let mut probed_at = None;
        for round in 0..20 {
            let c = a.choose(&Glm, &[ask(0)]);
            if matches!(c[0], Choice::Mtp(_)) {
                probed_at = Some(round);
                break;
            }
            a.observe(0, c[0], 0, 0);
        }
        let at = probed_at.expect("the head was never probed again");
        assert!((7..=9).contains(&at), "probed at round {at}, expected after 8 serial rounds");
        assert_eq!(a.stats.probes, 1);
        // a probe that is accepted pulls the estimate back up: the stream drafts on its own again after a few
        for _ in 0..12 {
            a.observe(0, Choice::Mtp(1), 1, 1);
        }
        assert!(matches!(a.choose(&Glm, &[ask(0)])[0], Choice::Mtp(_)));
    }

    #[test]
    fn a_failing_probe_doubles_the_wait_and_a_good_one_resets_it() {
        let cfg = ArbiterConfig { probe_every: 4, probe_max_gap: 32, ..ArbiterConfig::default() };
        let mut a = Arbiter::new(cfg);
        for _ in 0..40 {
            a.observe(0, Choice::Mtp(2), 2, 0);
        }
        // drive serial rounds, answering every probe with a total miss: the probes come at 4, then 8, 16, 32, 32 rounds apart
        let mut gaps = Vec::new();
        let mut since = 0;
        for _ in 0..200 {
            let c = a.choose(&Glm, &[ask(0)])[0];
            if let Choice::Mtp(d) = c {
                gaps.push(since);
                since = 0;
                a.observe(0, c, d, 0);
            } else {
                since += 1;
                a.observe(0, c, 0, 0);
            }
        }
        assert_eq!(gaps[..5], [4, 8, 16, 32, 32], "{gaps:?}");
        // when the estimate recovers (the text changed: the head is right again) and the search picks the head on its own, the base gap is back: after
        // the next collapse the first probe comes after 4 rounds, not after the doubled gap
        let probe = loop {
            let c = a.choose(&Glm, &[ask(0)])[0];
            if matches!(c, Choice::Mtp(_)) {
                break c;
            }
            a.observe(0, c, 0, 0);
        };
        a.observe(0, probe, 1, 1);
        for _ in 0..16 {
            a.observe(0, Choice::Mtp(2), 2, 2);
        }
        assert!(matches!(a.choose(&Glm, &[ask(0)])[0], Choice::Mtp(_)), "the recovered estimate is picked on its own");
        for _ in 0..60 {
            a.observe(0, Choice::Mtp(2), 2, 0);
        }
        let mut waited = 0;
        loop {
            let c = a.choose(&Glm, &[ask(0)])[0];
            if matches!(c, Choice::Mtp(_)) {
                break;
            }
            waited += 1;
            a.observe(0, c, 0, 0);
            assert!(waited < 100, "no probe came");
        }
        assert_eq!(waited, 4, "a source that was worth picking again has the base gap, not the doubled one");
    }

    /// A round of the simulated service: `streams` streams, each with a true per-draft acceptance `acc` (all depths), the arbiter choosing; the time of
    /// the round by the cost model. Deterministic pseudo-random matches.
    fn simulate(acc: f64, streams: usize, rounds: usize, cfg: ArbiterConfig, serial: bool) -> f64 {
        let mut a = Arbiter::new(cfg);
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut tokens, mut ms) = (0.0, 0.0);
        for _ in 0..rounds {
            let asks: Vec<Ask> = (0..streams).map(ask).collect();
            let picks = if serial { vec![Choice::None; streams] } else { a.choose(&Glm, &asks) };
            let rows: usize = picks.iter().map(|c| 1 + c.drafts()).sum();
            let s_mtp = picks.iter().filter(|c| matches!(c, Choice::Mtp(_))).count();
            let depth = picks.iter().map(|c| if let Choice::Mtp(d) = c { *d } else { 0 }).max().unwrap_or(0);
            ms += Glm.verify_ms(rows) + if s_mtp > 0 { Glm.mtp_ms(s_mtp, 2 * s_mtp, depth) } else { 0.0 };
            for (i, &c) in picks.iter().enumerate() {
                let d = c.drafts();
                let mut accepted = 0;
                while accepted < d && rnd() < acc {
                    accepted += 1;
                }
                tokens += 1.0 + accepted as f64;
                a.observe(i, c, d, accepted);
            }
        }
        tokens / ms
    }

    #[test]
    fn simulated_service_is_never_below_serial_and_gains_where_the_head_is_good() {
        // true acceptance from useless to perfect, 1 to 8 streams: the arbiter's throughput against plain serial rows, learning online from nothing
        // but what each round showed it, probes and all
        for streams in [1usize, 2, 4, 8] {
            let serial = simulate(0.0, streams, 3000, ArbiterConfig::default(), true);
            for acc in [0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0] {
                let auto = simulate(acc, streams, 3000, ArbiterConfig::default(), false);
                eprintln!("SIM {streams} streams, true acceptance {acc}: {:.3}x serial", auto / serial);
                assert!(
                    auto >= serial * 0.995,
                    "{streams} streams, acceptance {acc}: {:.4} against serial {:.4} tokens/ms",
                    auto,
                    serial
                );
                if acc >= 0.8 {
                    let want = if streams == 1 { 1.4 } else { 1.15 };
                    assert!(auto > serial * want, "{streams} streams, acceptance {acc}: only {:.2}x", auto / serial);
                }
            }
        }
    }

    #[test]
    fn a_new_stream_is_held_to_a_shallow_chain_until_the_head_has_shown_what_it_does_on_it() {
        let mut a = arb();
        let c = a.choose(&Glm, &[ask(0)])[0];
        assert!(
            matches!(c, Choice::Mtp(d) if d <= 2),
            "the optimistic prior must not buy a deep chain on the first round: {c:?}"
        );
        // four verified rounds of a head that is right every time: now the whole chain is on offer
        for _ in 0..4 {
            a.observe(0, Choice::Mtp(2), 2, 2);
        }
        let c = a.choose(&Glm, &[ask(0)])[0];
        assert!(matches!(c, Choice::Mtp(d) if d >= 3), "after the warm-up a good head is drafted deep: {c:?}");
        // another stream is still new
        assert!(matches!(a.choose(&Glm, &[ask(1)])[0], Choice::Mtp(d) if d <= 2));
        // and a new sequence in the slot starts the warm-up again
        a.begin(0);
        assert!(matches!(a.choose(&Glm, &[ask(0)])[0], Choice::Mtp(d) if d <= 2));
    }

    #[test]
    fn a_stream_that_must_make_room_absorbs_even_when_it_drafts_nothing() {
        let mut a = arb();
        let mut k = ask(0);
        k.must_absorb = true;
        // the pick itself is unchanged by it (the caller absorbs without drafting), and the round is priced with the absorb
        let picks = a.choose(&Glm, &[k]);
        assert_eq!(a.stats.forced_absorbs, u64::from(!matches!(picks[0], Choice::Mtp(_))));
        let (_, with) = a.rate(&Glm, &[k], &[Choice::None]);
        let (_, without) = a.rate(&Glm, &[ask(0)], &[Choice::None]);
        assert!(with > without, "the forced absorb costs time and must be counted");
    }

    #[test]
    fn a_new_sequence_starts_from_the_priors() {
        let mut a = arb();
        for _ in 0..40 {
            a.observe(3, Choice::Mtp(2), 2, 0);
        }
        assert!(a.mtp_alpha(3, 1) < 0.2, "{}", a.mtp_alpha(3, 1));
        a.begin(3);
        assert!((a.mtp_alpha(3, 1) - 0.86).abs() < 1e-12);
        // and another stream's estimates were never touched
        assert!((a.mtp_alpha(1, 1) - 0.86).abs() < 1e-12);
    }
}
