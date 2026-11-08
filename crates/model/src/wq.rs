//! The `W_q` residency: the tail-floor lever of `docs/TAIL_FLOOR_CEILING.md`, built and behind a knob, default OFF.
//!
//! The pinned prefill reconstructs `W_q` from the trellis on every layer of every chunk, independent of the chunk's
//! row count: 269.8 ms of every chunk, 51% of the 531.4 ms cached-re-ask floor. The reconstruct is a pure function of
//! the trellis, so a `W_q` kept in device memory holds exactly the bits the per-chunk one would have held, and
//! [`gb_cuda::exl3_linear::Exl3Weights::forward_prefill_pinned_with_w`] runs the SAME pinned cublasLt GEMM on it: a
//! row's bits are what they always were, whatever the chunk grid and whether the matrix is resident (the exactness
//! tests pin this on the real pack). The pinned GEMM must still read the `W_q` either way (~187 ms at roofline), so
//! the recoverable part is the reconstruct alone.
//!
//! The knob is `REXL3_WQ_RESIDENT_GIB` (GiB of resident `W_q`, 0 or unset = off; anything that does not parse is an
//! error, not a silent default). The budget is counted in the fail-closed memory plan ([`crate::plan`]): it comes out
//! of the allocatable bytes before the KV pool is sized and sits in [`crate::plan::MemoryPlan::planned_bytes`], so a
//! boot the box cannot hold is refused, never quietly oversubscribed.
//!
//! Eviction is MRU by matrix (the whole matrix is resident or it is not; a matrix is never split). Under the prefill's
//! cyclic walk over all 400 matrices a most-recently-used victim keeps the head of the order resident — a partial
//! budget recovers proportionally (5.96 ms per GiB per chunk) — where an LRU would thrash to a zero hit rate. A
//! matrix larger than the whole budget is never resident and keeps today's per-chunk path.
//!
//! Filling is lazy: each resident matrix is reconstructed once, into its own slot, on its first use (the first chunk
//! of the first prompt pays what it always paid; every later chunk skips it). Nothing is allocated at boot beyond the
//! plan's booking of the budget.

use crate::error::ModelError;
use gb_cuda::exl3_linear::Exl3Weights;
use gb_cuda::{DevBuf, Stream};
use std::collections::HashMap;

/// The environment knob: GiB of resident `W_q` (`REXL3_WQ_RESIDENT_GIB`, default 0 = off).
pub const WQ_RESIDENT_ENV: &str = "REXL3_WQ_RESIDENT_GIB";

/// The residency budget in bytes: the option's own value, else the environment's, else off (None). Zero, a negative
/// or a non-finite value is an error when asked for by name and "off" when it comes from the environment unset; a
/// value that does not parse is an error either way, never a silent default.
pub fn budget_bytes(opt: Option<f64>) -> Result<Option<u64>, ModelError> {
    let gib = match opt {
        Some(g) => Some(g),
        None => budget_from_env()?,
    };
    let Some(g) = gib else { return Ok(None) };
    if !g.is_finite() || g < 0.0 {
        return Err(ModelError::Config(format!("the W_q residency budget must be a non-negative number of GiB, got `{g}`")));
    }
    Ok((g > 0.0).then_some((g * (1u64 << 30) as f64) as u64))
}

/// `REXL3_WQ_RESIDENT_GIB` as a number of GiB, or `None` when unset or empty.
pub fn budget_from_env() -> Result<Option<f64>, ModelError> {
    match std::env::var(WQ_RESIDENT_ENV) {
        Ok(v) if v.trim().is_empty() => Ok(None),
        Ok(v) => Ok(Some(
            v.trim().parse().map_err(|_| ModelError::Config(format!("{WQ_RESIDENT_ENV} must be a number of GiB, got `{v}`")))?,
        )),
        Err(_) => Ok(None),
    }
}

/// The admission and eviction arithmetic of the residency, with no device in it (the CPU tests pin its dynamics).
/// Matrix `i`'s byte size is `sizes[i]`; `budget` bounds what may be resident at once.
pub struct WqPolicy {
    sizes: Vec<usize>,
    budget: usize,
    resident: Vec<bool>,
    last_use: Vec<u64>,
    clock: u64,
    bytes: usize,
    evictions: u64,
    /// Scratch `admit` hands its victims out through (cleared on entry each call).
    victims: Vec<usize>,
}

impl WqPolicy {
    pub fn new(sizes: Vec<usize>, budget: usize) -> Self {
        let n = sizes.len();
        Self {
            sizes,
            budget,
            resident: vec![false; n],
            last_use: vec![0; n],
            clock: 0,
            bytes: 0,
            evictions: 0,
            victims: Vec::new(),
        }
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    /// The bytes every matrix together would need (45.31 GiB's worth on the target pack).
    pub fn total(&self) -> usize {
        self.sizes.iter().sum()
    }

    pub fn size(&self, i: usize) -> usize {
        self.sizes[i]
    }

    pub fn resident_bytes(&self) -> usize {
        self.bytes
    }

    pub fn resident_count(&self) -> usize {
        self.resident.iter().filter(|&&r| r).count()
    }

    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// The resident matrices, in index order (the prefill's own order).
    pub fn resident_ids(&self) -> Vec<usize> {
        self.resident.iter().enumerate().filter(|(_, &r)| r).map(|(i, _)| i).collect()
    }

    /// Whether `i` is resident; records the use either way (the MRU clock).
    pub fn touch(&mut self, i: usize) -> bool {
        self.clock += 1;
        self.last_use[i] = self.clock;
        self.resident[i]
    }

    /// Whether `i` can ever be resident under this budget.
    pub fn fits(&self, i: usize) -> bool {
        self.sizes[i] <= self.budget
    }

    /// Make `i` resident, evicting most-recently-used matrices until it fits: `Some(victims)` (possibly empty) so the
    /// caller can drop what they held, or `None` when the matrix alone is larger than the budget (it keeps the
    /// per-chunk path); the caller has already missed, so this is the fill, not a hit.
    pub fn admit(&mut self, i: usize) -> Option<&[usize]> {
        if self.sizes[i] > self.budget {
            return None;
        }
        self.victims.clear();
        while self.bytes + self.sizes[i] > self.budget {
            let victim = self
                .resident
                .iter()
                .enumerate()
                .filter(|(_, &r)| r)
                .max_by_key(|&(j, _)| (self.last_use[j], std::cmp::Reverse(j)))
                .expect("bytes exceed the budget, so something is resident")
                .0;
            self.resident[victim] = false;
            self.bytes -= self.sizes[victim];
            self.evictions += 1;
            self.victims.push(victim);
        }
        self.resident[i] = true;
        self.bytes += self.sizes[i];
        Some(&self.victims)
    }
}

/// The residency on the device: one slot per matrix of the prefill's linear list, filled by
/// [`Exl3Weights::reconstruct`] (bit-exact by construction) and read by the pinned prefill. No allocation happens
/// until a matrix's first use.
pub struct WqResidency {
    policy: WqPolicy,
    slots: Vec<Option<DevBuf>>,
    /// The linear's own address -> its index (the weights live as long as the model, so the keys are stable).
    index: HashMap<usize, usize>,
    hits: u64,
    misses: u64,
}

/// The identity of a linear within one model: the weight struct's address (the weights are immovable once loaded).
fn id_of(wt: &Exl3Weights) -> usize {
    std::ptr::from_ref(wt) as usize
}

impl WqResidency {
    /// `linears` is the model's [`crate::weights::Weights::prefill_linears`] (the pinned path's own list; the head,
    /// which only sees a few rows, is not part of it).
    pub fn new(linears: &[&Exl3Weights], budget_bytes: u64) -> Self {
        let budget = usize::try_from(budget_bytes).unwrap_or(usize::MAX);
        let sizes = linears.iter().map(|l| l.k * l.n * 2).collect();
        let index = linears.iter().map(|l| id_of(l)).enumerate().map(|(i, id)| (id, i)).collect();
        let n = linears.len();
        Self { policy: WqPolicy::new(sizes, budget), slots: (0..n).map(|_| None).collect(), index, hits: 0, misses: 0 }
    }

    /// Record a use of `wt`'s matrix and say whether its `W_q` is resident (the reconstruct is skipped). Total: a
    /// matrix whose fill failed midway (the policy marked it, the allocation did not land) reads as not resident and
    /// is filled again, never panicked on.
    pub fn hit(&mut self, wt: &Exl3Weights) -> bool {
        let Some(&i) = self.index.get(&id_of(wt)) else { return false };
        if self.policy.touch(i) && self.slots[i].is_some() {
            self.hits += 1;
            true
        } else {
            false
        }
    }

    /// Fill `wt`'s slot: reconstruct its `W_q` into the residency, dropping what the evicted victims held so the
    /// device bytes follow the policy's books. `Ok(false)` when the matrix can never fit the budget (today's
    /// per-chunk path answers for it).
    pub fn fill(&mut self, wt: &Exl3Weights, st: &Stream) -> Result<bool, ModelError> {
        let Some(&i) = self.index.get(&id_of(wt)) else { return Ok(false) };
        let Some(victims) = self.policy.admit(i).map(|v| v.to_vec()) else { return Ok(false) };
        for v in victims {
            self.slots[v] = None;
        }
        let bytes = self.policy.size(i);
        let mut buf = DevBuf::alloc(bytes)?;
        wt.reconstruct(&mut buf, st)?;
        self.slots[i] = Some(buf);
        self.misses += 1;
        Ok(true)
    }

    /// The resident `W_q` of `wt`, after a `hit` or a `fill` said there is one.
    pub fn w(&self, wt: &Exl3Weights) -> Option<&DevBuf> {
        let &i = self.index.get(&id_of(wt))?;
        self.slots[i].as_ref()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// The reconstructs the residency paid to fill itself (one per resident matrix, on its first use).
    pub fn misses(&self) -> u64 {
        self.misses
    }

    pub fn policy(&self) -> &WqPolicy {
        &self.policy
    }
}

/// What a running model can say about its residency (tests and the boot log; nothing on the serving path reads it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WqStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub resident: usize,
    pub resident_bytes: usize,
    pub budget_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_env(v: Option<&str>) {
        match v {
            Some(v) => std::env::set_var(WQ_RESIDENT_ENV, v),
            None => std::env::remove_var(WQ_RESIDENT_ENV),
        }
    }

    #[test]
    fn the_knob_is_off_by_default_and_parses_gib() {
        set_env(None);
        assert_eq!(budget_bytes(None).unwrap(), None, "unset is off");
        assert_eq!(budget_bytes(Some(0.0)).unwrap(), None, "zero is off");
        assert_eq!(
            budget_bytes(Some(45.31)).unwrap(),
            Some(48_651_242_045),
            "45.31 GiB covers the pack's 400 matrices (45.3125 GiB exactly)"
        );
        set_env(Some("2"));
        assert_eq!(budget_bytes(None).unwrap(), Some(2 << 30), "the environment sets it when the option does not");
        assert_eq!(budget_bytes(Some(1.0)).unwrap(), Some(1 << 30), "the option beats the environment");
        set_env(Some("junk"));
        assert!(budget_bytes(None).is_err(), "an unparsable knob is an error, never a silent default");
        set_env(None);
        assert!(budget_bytes(Some(-1.0)).is_err() && budget_bytes(Some(f64::NAN)).is_err());
    }

    #[test]
    fn a_budget_covering_everything_never_evicts() {
        let mut p = WqPolicy::new(vec![3, 3, 3], 10);
        for i in 0..3 {
            assert!(!p.touch(i), "matrix {i} starts cold");
            assert!(p.admit(i).is_some(), "everything fits");
        }
        for pass in 0..3 {
            for i in 0..3 {
                assert!(p.touch(i), "pass {pass} matrix {i} must hit: everything stays resident");
            }
        }
        assert_eq!((p.resident_count(), p.evictions(), p.resident_bytes()), (3, 0, 9));
    }

    /// The prefill walks all its matrices in a fixed cyclic order; MRU eviction must keep the head of the order
    /// resident (a partial budget recovers its share) where an LRU would thrash to nothing: after the first pass
    /// every pass must hit, and the resident bytes never exceed the budget.
    #[test]
    fn mru_eviction_keeps_the_head_of_a_cyclic_walk() {
        let mut p = WqPolicy::new(vec![1, 1, 1, 1, 1], 3);
        let mut hits_per_pass = Vec::new();
        for _pass in 0..5 {
            let mut hits = 0;
            for i in 0..5 {
                if p.touch(i) {
                    hits += 1;
                } else {
                    assert!(p.admit(i).is_some());
                }
            }
            hits_per_pass.push(hits);
            assert!(p.resident_bytes() <= p.budget());
        }
        assert_eq!(hits_per_pass[0], 0, "the first pass fills");
        assert!(
            hits_per_pass[1..].iter().all(|&h| h > 0),
            "an LRU would thrash to zero hits here; MRU recovers a share: {hits_per_pass:?}"
        );
        assert!(p.evictions() > 0);
        assert_eq!(p.resident_ids(), vec![0, 1, 4], "the head of the order stays resident, the tail cycles");
    }

    #[test]
    fn a_matrix_larger_than_the_budget_is_never_resident() {
        let mut p = WqPolicy::new(vec![5, 1, 1], 3);
        assert!(!p.touch(0) && p.admit(0).is_none(), "the oversized matrix keeps the per-chunk path");
        assert!(p.touch(1) || p.admit(1).is_some());
        assert!(p.admit(2).is_some());
        assert_eq!(p.resident_ids(), vec![1, 2]);
        assert_eq!(p.evictions(), 0);
    }

    #[test]
    fn eviction_takes_the_most_recently_used_not_the_least() {
        // use 0, then 1, then ask for a matrix that needs 1's space: the victim must be 1 (most recent), not 0
        let mut p = WqPolicy::new(vec![2, 2, 2], 4);
        for i in 0..2 {
            assert!(!p.touch(i) && p.admit(i).is_some());
        }
        assert!(!p.touch(2) && p.admit(2).is_some());
        assert_eq!(p.resident_ids(), vec![0, 2], "the least recently used matrix survives");
        assert_eq!(p.evictions(), 1);
    }
}
