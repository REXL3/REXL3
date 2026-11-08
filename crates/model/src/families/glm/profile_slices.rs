//! `REXL3_GLM_PROFILE_SLICES`: wrap chosen forward calls of a GLM rank in `cudaProfilerStart` / `cudaProfilerStop`, so that
//! `nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat:N` traces exactly those calls on every rank of a group.
//!
//! **What is counted.** The forward calls of a rank, as the lockstep runtime hands them to it (`tp::GlmRank`, on rank 0 inside the
//! `TpLeader` and on a follower in `run_follower`): prefill chunks (text and image), decode / verify windows, NextN draft calls. A
//! follower sees the same calls in the same order as rank 0, so the k-th prefill chunk is the same call on every rank and the ranks'
//! exports line up (`tools/nemotron_tp_nsys_skew.py`).
//!
//! **The setting.** A comma list of ranges, each `[KIND:]A..B` (half-open over the kind's 0-based ordinal), `[KIND:]A..=B` or
//! `[KIND:]A` (one call). KIND is `p` (prefill chunks), `w` (windows), `d` (draft calls) or `a` (every forward call; the default), and
//! `p@K` (the prefill chunks of the K-th prompt the rank begins, 0-based: `begin_slot` calls, which every rank sees in one order; the
//! ordinal is then the chunk's within that prompt, so a range needs no prompt's token count). The
//! capture opens at the start of the first call a range selects and stays open while some range has started and has calls left, so the
//! calls of another kind that fall between two selected ones are traced too (a prefill slice beside a decoding stream: `p:8..10` also
//! traces the windows between chunks 8 and 9). Each closed window is one nsys range. A malformed setting refuses the model by name.
//!
//! **Bits.** Nothing here reaches the arithmetic or the stream order: the profiler calls are host calls, and closing a window drains the
//! device (a host synchronisation after the call's launches, which every prefill chunk and window already ends with) so that the
//! window's kernels are inside it. Proved by the digest legs with the setting on and off (`glmperf/E/PHASE0.md`).
//!
//! While a window is open, every call logs `event=glm_profile_call` with its kind, ordinals, rows and host wall time (from its start to
//! its end, the host having waited for the device at the end of every call of these kinds), which is the step and chunk time the profile
//! decomposes.

use crate::error::ModelError;
use std::time::Instant;

/// The environment variable (unset: nothing is profiled and nothing is logged).
pub const ENV: &str = "REXL3_GLM_PROFILE_SLICES";

/// The kinds of forward call a range can select.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A prefill chunk (text or with image rows).
    Prefill,
    /// A decode or verify window.
    Window,
    /// A NextN draft call.
    Draft,
}

impl Kind {
    fn index(self) -> usize {
        match self {
            Kind::Prefill => 0,
            Kind::Window => 1,
            Kind::Draft => 2,
        }
    }
    fn tag(self) -> &'static str {
        match self {
            Kind::Prefill => "p",
            Kind::Window => "w",
            Kind::Draft => "d",
        }
    }
}

/// `[lo, hi)` over the ordinal of `kind`'s calls (`None`: of every forward call); with `prompt`, over the prefill chunks of that prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub kind: Option<Kind>,
    pub prompt: Option<u64>,
    pub lo: u64,
    pub hi: u64,
}

/// Parses the setting (see the module doc). An empty setting is no range.
pub fn parse(s: &str) -> Result<Vec<Range>, String> {
    let mut out = Vec::new();
    for item in s.split(',').map(str::trim).filter(|x| !x.is_empty()) {
        let (kind, prompt, span) = match item.split_once(':') {
            Some((k, span)) => {
                let (k, prompt) = match k.trim().split_once('@') {
                    Some((k, n)) => {
                        let n =
                            n.trim().parse::<u64>().map_err(|_| format!("{ENV}: {n:?} in {item:?} is not a prompt ordinal"))?;
                        (k.trim(), Some(n))
                    }
                    None => (k.trim(), None),
                };
                let kind = match k {
                    "p" | "prefill" => Some(Kind::Prefill),
                    "w" | "window" => Some(Kind::Window),
                    "d" | "draft" => Some(Kind::Draft),
                    "a" | "all" => None,
                    other => return Err(format!("{ENV}: unknown call kind {other:?} in {item:?} (p, w, d, a or p@K)")),
                };
                if prompt.is_some() && kind != Some(Kind::Prefill) {
                    return Err(format!("{ENV}: {item:?}: only prefill chunks are counted per prompt (p@K)"));
                }
                (kind, prompt, span.trim())
            }
            None => (None, None, item),
        };
        let num = |x: &str| x.trim().parse::<u64>().map_err(|_| format!("{ENV}: {x:?} in {item:?} is not a call ordinal"));
        let (lo, hi) = if let Some((a, b)) = span.split_once("..=") {
            (num(a)?, num(b)?.checked_add(1).ok_or_else(|| format!("{ENV}: {item:?} overflows"))?)
        } else if let Some((a, b)) = span.split_once("..") {
            (num(a)?, num(b)?)
        } else {
            let a = num(span)?;
            (a, a + 1)
        };
        if hi <= lo {
            return Err(format!("{ENV}: {item:?} selects no call (the range is half-open: A..B with B > A, or A..=B)"));
        }
        out.push(Range { kind, prompt, lo, hi });
    }
    Ok(out)
}

/// The ordinals of one call, from [`SliceProfiler::enter`] to [`SliceProfiler::leave`].
#[derive(Clone, Copy, Debug)]
pub struct Ticket {
    kind: Kind,
    ord: u64,
    all: u64,
    /// A prefill chunk's prompt and its ordinal within it.
    chunk: Option<(u64, u64)>,
    rows: usize,
    at: Option<Instant>,
}

/// The rank's call counters and its capture window. Inert (no counting cost beyond two increments, no log) without a setting.
#[derive(Debug, Default)]
pub struct SliceProfiler {
    ranges: Vec<Range>,
    counts: [u64; 3],
    all: u64,
    /// Prompts begun (`begin_slot` calls), the prompt each slot is prefilling and its chunks so far.
    prompts: u64,
    slot_prompt: std::collections::BTreeMap<usize, (u64, u64)>,
    open: Option<(Instant, u64)>,
    /// Windows opened and closed so far (the nsys ranges written).
    pub windows: u64,
}

impl SliceProfiler {
    /// From the environment; a malformed setting is a configuration error (the rank refuses to start).
    pub fn from_env() -> Result<Self, ModelError> {
        Self::from_env_key(ENV)
    }

    /// From `env_key` (or falling back to [`ENV`] when `env_key` is unset).
    pub fn from_env_key(env_key: &str) -> Result<Self, ModelError> {
        match std::env::var(env_key).or_else(|_| std::env::var(ENV)) {
            Ok(v) => Self::new(parse(&v).map_err(ModelError::Config)?),
            Err(_) => Ok(Self::default()),
        }
    }

    pub fn new(ranges: Vec<Range>) -> Result<Self, ModelError> {
        Ok(Self { ranges, ..Self::default() })
    }

    pub fn active(&self) -> bool {
        !self.ranges.is_empty()
    }

    pub fn is_open(&self) -> bool {
        self.open.is_some()
    }

    fn next_of(&self, kind: Option<Kind>) -> u64 {
        match kind {
            Some(k) => self.counts[k.index()],
            None => self.all,
        }
    }

    fn selects(&self, kind: Kind, ord: u64, all: u64, chunk: Option<(u64, u64)>) -> bool {
        self.ranges.iter().any(|r| match r.prompt {
            Some(k) => chunk.is_some_and(|(p, c)| p == k && r.lo <= c && c < r.hi),
            None => {
                let o = if r.kind.is_some() { ord } else { all };
                (r.kind.is_none() || r.kind == Some(kind)) && r.lo <= o && o < r.hi
            }
        })
    }

    /// Chunks prefilled so far of prompt `k` (0 before it began, `u64::MAX` once its slot has begun another prompt: it is over).
    fn chunks_of(&self, k: u64) -> u64 {
        if k >= self.prompts {
            return 0;
        }
        self.slot_prompt.values().find(|(p, _)| *p == k).map_or(u64::MAX, |&(_, c)| c)
    }

    /// Some range has run a call and has calls left (counted after the current call).
    fn in_progress(&self) -> bool {
        self.ranges.iter().any(|r| {
            let next = match r.prompt {
                Some(k) => self.chunks_of(k),
                None => self.next_of(r.kind),
            };
            r.lo < next && next < r.hi
        })
    }

    /// A slot begins a prompt (`begin_slot`): its prefill chunks count from 0 under the next prompt ordinal.
    pub fn begin_prompt(&mut self, slot: usize) {
        self.slot_prompt.insert(slot, (self.prompts, 0));
        self.prompts += 1;
    }

    /// Before a forward call of `kind` over `rows` rows (a prefill chunk: of `slot`): counts it and opens the capture when a range
    /// selects it.
    pub fn enter(&mut self, kind: Kind, rows: usize, slot: Option<usize>) -> Ticket {
        let (ord, all) = (self.counts[kind.index()], self.all);
        self.counts[kind.index()] += 1;
        self.all += 1;
        let chunk = match (kind, slot.and_then(|s| self.slot_prompt.get_mut(&s))) {
            (Kind::Prefill, Some(e)) => {
                e.1 += 1;
                Some((e.0, e.1 - 1))
            }
            _ => None,
        };
        if !self.active() {
            return Ticket { kind, ord, all, chunk, rows, at: None };
        }
        if self.open.is_none() && self.selects(kind, ord, all, chunk) {
            gb_cuda::rt::profiler_start();
            self.open = Some((Instant::now(), all));
            eprintln!(
                "rexl3 event=glm_profile_open window={} kind={} ord={ord} all={all} prompt={} rows={rows}",
                self.windows,
                kind.tag(),
                chunk.map_or("-".into(), |(p, c)| format!("{p}:{c}"))
            );
        }
        let at = self.open.is_some().then(Instant::now);
        Ticket { kind, ord, all, chunk, rows, at }
    }

    /// After the call (`ok`: it returned `Ok`): logs it when the window is open, and closes the window when no range has calls left.
    pub fn leave(&mut self, t: Ticket, ok: bool) {
        let Some((opened, first)) = self.open else { return };
        if let Some(at) = t.at {
            eprintln!(
                "rexl3 event=glm_profile_call window={} kind={} ord={} all={} prompt={} rows={} ms={:.3} ok={ok}",
                self.windows,
                t.kind.tag(),
                t.ord,
                t.all,
                t.chunk.map_or("-".into(), |(p, c)| format!("{p}:{c}")),
                t.rows,
                at.elapsed().as_secs_f64() * 1e3
            );
        }
        if self.in_progress() {
            return;
        }
        // the window's kernels must be inside it: drain the device before the range ends (a host wait only)
        let drained = gb_cuda::rt::device_sync().is_ok();
        gb_cuda::rt::profiler_stop();
        eprintln!(
            "rexl3 event=glm_profile_close window={} calls={} ms={:.3} drained={drained}",
            self.windows,
            t.all + 1 - first,
            opened.elapsed().as_secs_f64() * 1e3
        );
        self.open = None;
        self.windows += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_setting_parses_kinds_and_ranges_and_refuses_what_selects_nothing() {
        assert_eq!(parse("").unwrap(), vec![]);
        assert_eq!(
            parse("p:8..10, w:3..=4,d:7,12..14").unwrap(),
            vec![
                Range { kind: Some(Kind::Prefill), prompt: None, lo: 8, hi: 10 },
                Range { kind: Some(Kind::Window), prompt: None, lo: 3, hi: 5 },
                Range { kind: Some(Kind::Draft), prompt: None, lo: 7, hi: 8 },
                Range { kind: None, prompt: None, lo: 12, hi: 14 },
            ]
        );
        assert_eq!(parse("a:0..1").unwrap(), vec![Range { kind: None, prompt: None, lo: 0, hi: 1 }]);
        assert_eq!(parse("p@2:7..9").unwrap(), vec![Range { kind: Some(Kind::Prefill), prompt: Some(2), lo: 7, hi: 9 }]);
        for bad in ["x:1..2", "p:3..3", "p:4..2", "p:a..b", "1..", "..3", "w@1:2..3", "p@x:1..2"] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    /// The window of `p:2..4` beside windows: opens at prefill 2, stays open over the windows between, closes after prefill 3.
    #[test]
    fn a_window_opens_at_the_first_selected_call_and_closes_after_the_last() {
        let mut p = SliceProfiler::new(parse("p:2..4").unwrap()).unwrap();
        let mut trace = Vec::new();
        for (kind, n) in [(Kind::Prefill, 2), (Kind::Window, 1), (Kind::Prefill, 1), (Kind::Window, 2), (Kind::Prefill, 2)] {
            for _ in 0..n {
                let t = p.enter(kind, 1, None);
                let was_open = p.is_open();
                p.leave(t, true);
                trace.push((kind, was_open, p.is_open()));
            }
        }
        let open_during: Vec<bool> = trace.iter().map(|x| x.1).collect();
        // p0 p1 w0 | p2 w1 w2 p3 | p4
        assert_eq!(open_during, vec![false, false, false, true, true, true, true, false]);
        assert_eq!(p.windows, 1);
        assert!(!p.is_open());
    }

    #[test]
    fn two_disjoint_ranges_make_two_windows_and_no_setting_counts_without_opening() {
        let mut p = SliceProfiler::new(parse("w:1,w:3..5").unwrap()).unwrap();
        let mut opened = Vec::new();
        for _ in 0..6 {
            let t = p.enter(Kind::Window, 1, None);
            opened.push(p.is_open());
            p.leave(t, true);
        }
        assert_eq!(opened, vec![false, true, false, true, true, false]);
        assert_eq!(p.windows, 2);
        let mut q = SliceProfiler::default();
        let t = q.enter(Kind::Prefill, 512, None);
        q.leave(t, true);
        assert!(!q.active() && !q.is_open() && q.windows == 0 && q.counts == [1, 0, 0]);
    }

    /// `p@1:1..3`: chunks 1 and 2 of the second prompt, whatever the first prompt's length, with the other slot's windows and chunks
    /// between them traced too; the first prompt's chunks never open it.
    #[test]
    fn a_prompt_range_counts_the_chunks_of_that_prompt_only() {
        let mut p = SliceProfiler::new(parse("p@1:1..3").unwrap()).unwrap();
        let mut seen = Vec::new();
        let mut call = |p: &mut SliceProfiler, kind: Kind, slot: Option<usize>, tag: &'static str| {
            let t = p.enter(kind, 1, slot);
            seen.push((tag, p.is_open()));
            p.leave(t, true);
        };
        p.begin_prompt(0);
        for _ in 0..5 {
            call(&mut p, Kind::Prefill, Some(0), "A");
        }
        p.begin_prompt(1);
        call(&mut p, Kind::Prefill, Some(1), "B0");
        call(&mut p, Kind::Window, None, "w");
        call(&mut p, Kind::Prefill, Some(1), "B1");
        call(&mut p, Kind::Window, None, "w");
        call(&mut p, Kind::Prefill, Some(1), "B2");
        call(&mut p, Kind::Prefill, Some(1), "B3");
        let open: Vec<&str> = seen.iter().filter(|x| x.1).map(|x| x.0).collect();
        assert_eq!(open, vec!["B1", "w", "B2"]);
        assert_eq!(p.windows, 1);
    }
}
