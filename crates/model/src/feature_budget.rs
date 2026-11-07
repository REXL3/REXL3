//! The bound on how many bytes of encoded image features may be alive at once ([`FeatureBudget`]) and a claim on part of it
//! ([`FeaturePermit`]). Moved verbatim from the GLM family's `vision_serve` module when that family left the tree; `serve` re-exports
//! both under their old path (`crate::serve::{FeatureBudget, FeaturePermit}`).

/// The bytes of image features that may be alive at once: what the memory plan held back for them (`MemoryProfile::vision_reserve_bytes`: features of
/// `slots x images-per-request` images at the largest size). HTTP encodes an image BEFORE the request is queued for a slot, so without this bound a
/// queue of image requests could hold features admission never priced. An encode takes a permit for its image's bytes; the permit goes back when the
/// features are dropped (when the sequence that owns them ends, or the request fails before it is submitted).
#[derive(Debug)]
pub struct FeatureBudget {
    free: std::sync::atomic::AtomicU64,
    total: u64,
}

impl FeatureBudget {
    pub fn new(total: u64) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self { free: std::sync::atomic::AtomicU64::new(total), total })
    }
    /// Bytes not held by any permit.
    pub fn free(&self) -> u64 {
        self.free.load(std::sync::atomic::Ordering::Acquire)
    }
    pub fn total(&self) -> u64 {
        self.total
    }
    /// Take `bytes`, or `None` when fewer are free.
    pub fn acquire(self: &std::sync::Arc<Self>, bytes: u64) -> Option<FeaturePermit> {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        self.free.fetch_update(AcqRel, Acquire, |f| f.checked_sub(bytes)).ok()?;
        Some(FeaturePermit { budget: self.clone(), bytes })
    }
}

/// A claim on `bytes` of a [`FeatureBudget`], returned on drop.
#[derive(Debug)]
pub struct FeaturePermit {
    budget: std::sync::Arc<FeatureBudget>,
    bytes: u64,
}

impl Drop for FeaturePermit {
    fn drop(&mut self) {
        self.budget.free.fetch_add(self.bytes, std::sync::atomic::Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn the_budget_hands_out_what_it_has_and_takes_it_back_on_drop() {
        let b = FeatureBudget::new(100);
        let p1 = b.acquire(60).unwrap();
        assert_eq!(b.free(), 40);
        assert!(b.acquire(41).is_none(), "more than is free is refused, whole");
        assert_eq!(b.free(), 40, "a refused acquire holds nothing");
        let p2 = b.acquire(40).unwrap();
        assert_eq!(b.free(), 0);
        assert!(b.acquire(1).is_none());
        drop(p1);
        assert_eq!(b.free(), 60);
        drop(p2);
        assert_eq!((b.free(), b.total()), (100, 100));
        assert!(b.acquire(0).is_some(), "an empty permit is allowed and free");
    }

    #[test]
    fn the_budget_is_never_oversubscribed_under_contention() {
        let b = FeatureBudget::new(1000);
        let held = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let max = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let threads: Vec<_> = (0..8u64)
            .map(|t| {
                let (b, held, max) = (b.clone(), held.clone(), max.clone());
                std::thread::spawn(move || {
                    for i in 0..2000u64 {
                        let want = 50 + (t * 37 + i * 11) % 200;
                        if let Some(p) = b.acquire(want) {
                            let now = held.fetch_add(want, std::sync::atomic::Ordering::SeqCst) + want;
                            max.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
                            std::thread::yield_now();
                            held.fetch_sub(want, std::sync::atomic::Ordering::SeqCst);
                            drop(p);
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert!(max.load(std::sync::atomic::Ordering::SeqCst) <= 1000, "the permits held at once exceeded the budget");
        assert_eq!(b.free(), 1000, "every permit came back");
    }
}
