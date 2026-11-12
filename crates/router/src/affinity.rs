//! Sticky affinity map: affinity key -> backend uid, bounded (LRU) with an idle TTL.
//!
//! Keys are the 64-bit hashes from `keying`, so an entry is a few dozen bytes whatever the client sent. Recency is a
//! monotonic tick; `order` maps tick -> key, so the least recently used entry is the first element and expiry (which is
//! "idle for `ttl`") walks the same end. All operations are O(log n) and allocate nothing beyond the two maps.
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

struct Entry {
    backend: u64,
    tick: u64,
    last_used: Instant,
}

pub struct AffinityMap {
    cap: usize,
    ttl: Duration,
    map: HashMap<u64, Entry>,
    order: BTreeMap<u64, u64>,
    tick: u64,
    pub evicted_capacity: u64,
    pub evicted_ttl: u64,
}

impl AffinityMap {
    pub fn new(cap: usize, ttl: Duration) -> Self {
        Self { cap: cap.max(1), ttl, map: HashMap::new(), order: BTreeMap::new(), tick: 0, evicted_capacity: 0, evicted_ttl: 0 }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// The backend `key` is bound to, unless the binding went idle past the TTL (then it is dropped). Does not refresh
    /// recency: the caller `bind`s again once it knows the backend took the request.
    pub fn lookup(&mut self, key: u64, now: Instant) -> Option<u64> {
        let e = self.map.get(&key)?;
        if now.saturating_duration_since(e.last_used) > self.ttl {
            let tick = e.tick;
            self.order.remove(&tick);
            self.map.remove(&key);
            self.evicted_ttl += 1;
            return None;
        }
        Some(e.backend)
    }

    /// Bind (or re-bind) `key` to `backend` and mark it most recently used; then evict expired entries and, if still over
    /// capacity, the least recently used ones.
    pub fn bind(&mut self, key: u64, backend: u64, now: Instant) {
        self.tick += 1;
        if let Some(old) = self.map.insert(key, Entry { backend, tick: self.tick, last_used: now }) {
            self.order.remove(&old.tick);
        }
        self.order.insert(self.tick, key);
        while let Some((&tick, &k)) = self.order.iter().next() {
            let expired = self.map.get(&k).is_some_and(|e| now.saturating_duration_since(e.last_used) > self.ttl);
            if expired {
                self.order.remove(&tick);
                self.map.remove(&k);
                self.evicted_ttl += 1;
            } else if self.map.len() > self.cap {
                self.order.remove(&tick);
                self.map.remove(&k);
                self.evicted_capacity += 1;
            } else {
                break;
            }
        }
    }

    /// Forget every binding to `backend` (it was removed from the router).
    pub fn forget_backend(&mut self, backend: u64) {
        let doomed: Vec<(u64, u64)> = self.map.iter().filter(|(_, e)| e.backend == backend).map(|(k, e)| (*k, e.tick)).collect();
        for (k, tick) in doomed {
            self.map.remove(&k);
            self.order.remove(&tick);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_evicts_least_recently_used() {
        let mut m = AffinityMap::new(3, Duration::from_secs(3600));
        let t = Instant::now();
        for k in 1..=3 {
            m.bind(k, 10, t);
        }
        // touch 1: 2 is now the oldest
        assert_eq!(m.lookup(1, t), Some(10));
        m.bind(1, 10, t);
        m.bind(4, 11, t);
        assert_eq!(m.len(), 3);
        assert_eq!(m.lookup(2, t), None, "least recently used key was evicted");
        assert_eq!(m.lookup(1, t), Some(10));
        assert_eq!(m.lookup(3, t), Some(10));
        assert_eq!(m.lookup(4, t), Some(11));
        assert_eq!(m.evicted_capacity, 1);
        for k in 100..200 {
            m.bind(k, 12, t);
        }
        assert_eq!(m.len(), 3, "never above the bound");
    }

    #[test]
    fn idle_entries_expire() {
        let mut m = AffinityMap::new(100, Duration::from_secs(60));
        let t0 = Instant::now();
        m.bind(1, 10, t0);
        m.bind(2, 10, t0 + Duration::from_secs(30));
        assert_eq!(m.lookup(1, t0 + Duration::from_secs(59)), Some(10));
        // key 1 idle 61 s -> expired on lookup; key 2 (idle 31 s) lives
        assert_eq!(m.lookup(1, t0 + Duration::from_secs(61)), None);
        assert_eq!(m.lookup(2, t0 + Duration::from_secs(61)), Some(10));
        // a bind sweeps expired entries from the old end
        m.bind(3, 11, t0 + Duration::from_secs(200));
        assert_eq!(m.len(), 1);
        assert_eq!(m.evicted_ttl, 2, "key 1 on lookup, key 2 swept by the bind");
    }

    #[test]
    fn forget_backend_drops_its_keys_only() {
        let mut m = AffinityMap::new(10, Duration::from_secs(60));
        let t = Instant::now();
        m.bind(1, 10, t);
        m.bind(2, 11, t);
        m.bind(3, 10, t);
        m.forget_backend(10);
        assert_eq!(m.len(), 1);
        assert_eq!(m.lookup(2, t), Some(11));
    }
}
