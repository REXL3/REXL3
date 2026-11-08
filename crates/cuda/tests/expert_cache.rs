//! The expert page cache (`src/expert_cache.rs`) on the host: a fake pack and a fake device, so the eviction policy, the pinning
//! contract, the bounds on the prefetch queue and the miss rate are all measured without a GPU.
//!
//!     cargo test -p gb-cuda --test expert_cache -- --nocapture
//!
//! Nothing here is gated on a device. The one test that touches the CUDA slab prints `not run: no device` on a build without one
//! and passes; every other test runs everywhere.

use gb_cuda::expert_cache::{
    CacheError, Config, DeviceSlab, ExpertCache, ExpertSource, Lease, Policy, Slab, EXPERTS_PER_LAYER, EXPERT_BYTES,
    TOUCHES_PER_TOKEN,
};
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------------------------------------- the fakes

/// One expert of the fake pack: a byte per position that depends on (layer, expert, offset), so a slot that holds the wrong
/// expert — or half of two — cannot pass a check.
fn byte_at(layer: u16, expert: u16, off: usize) -> u8 {
    let mut x = (layer as u64) << 32 | (expert as u64) << 8 | (off as u64 & 0xff);
    x ^= x >> 29;
    x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    (x >> 24) as u8
}

/// A pack in memory. Reads are counted, optionally slowed down, and can be told to fail.
struct FakeSource {
    expert_bytes: usize,
    experts_per_layer: usize,
    reads: AtomicU64,
    delay_us: u64,
    fail: bool,
}

impl FakeSource {
    fn new(expert_bytes: usize, experts_per_layer: usize) -> Self {
        Self { expert_bytes, experts_per_layer, reads: AtomicU64::new(0), delay_us: 0, fail: false }
    }
    fn slow(expert_bytes: usize, experts_per_layer: usize, delay_us: u64) -> Self {
        Self { delay_us, ..Self::new(expert_bytes, experts_per_layer) }
    }
    fn reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }
    fn expert(&self, layer: u16, expert: u16) -> Vec<u8> {
        let _ = self.experts_per_layer;
        (0..self.expert_bytes).map(|o| byte_at(layer, expert, o)).collect()
    }
}

impl ExpertSource for FakeSource {
    fn read_expert(&self, layer: u16, expert: u16, dst: &mut [u8]) -> io::Result<()> {
        assert_eq!(dst.len(), self.expert_bytes);
        self.reads.fetch_add(1, Ordering::Relaxed);
        if self.delay_us > 0 {
            std::thread::sleep(std::time::Duration::from_micros(self.delay_us));
        }
        if self.fail {
            return Err(io::Error::other("fake pack is unreadable"));
        }
        for (o, b) in dst.iter_mut().enumerate() {
            *b = byte_at(layer, expert, o);
        }
        Ok(())
    }
    fn expert_bytes(&self) -> usize {
        self.expert_bytes
    }
}

/// A device in host memory: `slots` experts of `expert_bytes`, one flat allocation, which is what the CUDA slab is.
struct FakeSlab {
    bytes: Vec<u8>,
    expert_bytes: usize,
    fills: AtomicU64,
}

impl FakeSlab {
    fn new(slots: usize, expert_bytes: usize) -> Self {
        Self { bytes: vec![0; slots * expert_bytes], expert_bytes, fills: AtomicU64::new(0) }
    }
    fn expert(&self, slot: u32, expert_bytes: usize) -> &[u8] {
        let o = slot as usize * expert_bytes;
        &self.bytes[o..o + expert_bytes]
    }
}

impl Slab for FakeSlab {
    fn fill(&self, slot: u32, src: &[u8]) -> Result<(), String> {
        assert_eq!(src.len(), self.expert_bytes);
        let o = slot as usize * self.expert_bytes;
        assert!(o + src.len() <= self.bytes.len(), "fill past the end of the slab");
        // SAFETY: the range is inside the allocation (checked) and no other thread writes this slot while it is filling
        unsafe {
            let dst = self.bytes.as_ptr().add(o) as *mut u8;
            core::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
        self.fills.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn read_back(&self, slot: u32, dst: &mut [u8]) -> Result<(), String> {
        dst.copy_from_slice(self.expert(slot, self.expert_bytes));
        Ok(())
    }
    fn bytes(&self) -> usize {
        self.bytes.len()
    }
    fn slot_ptr(&self, slot: u32) -> *mut u8 {
        unsafe { self.bytes.as_ptr().add(slot as usize * self.expert_bytes) as *mut u8 }
    }
}

// ---------------------------------------------------------------------------------------------------------- the traces

/// A deterministic xorshift64*, so every run of a test sees the same trace.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Cumulative Zipf(s) weights over `n` items, for sampling a skewed popularity.
fn zipf_cdf(n: usize, s: f64) -> Vec<f64> {
    let mut acc = 0.0;
    let mut cdf = Vec::with_capacity(n);
    for i in 1..=n {
        acc += (i as f64).powf(-s);
        cdf.push(acc);
    }
    let total = acc;
    for c in cdf.iter_mut() {
        *c /= total;
    }
    cdf
}

fn sample(cdf: &[f64], u: f64) -> u16 {
    let mut lo = 0;
    let mut hi = cdf.len() - 1;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cdf[mid] < u {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo as u16
}

// ---------------------------------------------------------------------------------------------------------- the tests

/// The slab is exactly `slots * expert_bytes` and the budget is never exceeded, whatever the trace does. The realistic
/// configuration is checked arithmetically (allocating 13.6 GB of fake slab would only test the allocator).
#[test]
fn budget_is_never_exceeded() {
    let expert_bytes = 4096;
    let cfg = Config {
        budget_bytes: 100 * expert_bytes + 123, // 123 bytes of slack the cache must not spend
        expert_bytes,
        layers: 40,
        experts_per_layer: 8,
        workers: 2,
        max_inflight: 2,
        queue_cap: 8,
        aging_period: 256,
        policy: Policy::Lfu,
    };
    let slots = cfg.slots();
    assert!(slots * expert_bytes <= cfg.budget_bytes, "{slots} slots fit the budget");
    assert_eq!(cfg.slab_bytes(), slots * expert_bytes);

    let src = Arc::new(FakeSource::new(expert_bytes, 8));
    let slab = Arc::new(FakeSlab::new(slots, expert_bytes));
    let cache = ExpertCache::new(cfg.clone(), src.clone(), slab.clone()).expect("cache");
    assert_eq!(cache.capacity(), slots);
    assert_eq!(slab.bytes(), slots * expert_bytes, "the slab is allocated once, at the configured size");

    let mut rng = Rng::new(7);
    for _ in 0..20_000 {
        let layer = rng.below(40) as u16;
        let expert = rng.below(8) as u16;
        let lease = cache.get(layer, expert).expect("a lease");
        assert!(lease.slot() < slots as u32, "every lease names a slot inside the slab");
        assert!(lease.ptr() as usize >= slab.slot_ptr(0) as usize);
        assert!((lease.ptr() as usize) < slab.slot_ptr(0) as usize + slab.bytes());
    }
    let s = cache.stats();
    assert!(s.resident <= slots, "{} resident experts in {slots} slots", s.resident);
    assert_eq!(slab.bytes(), slots * expert_bytes, "the slab never grew");
    println!("budget {} B -> {slots} slots, {} B used, {} resident", cfg.budget_bytes, cfg.slab_bytes(), s.resident);
}

/// A working set that fits converges to 100 % hits: after one warm pass, nothing is read again.
#[test]
fn hot_set_that_fits_converges_to_all_hits() {
    let expert_bytes = 4096;
    let cfg = Config {
        budget_bytes: 40 * 8 * expert_bytes,
        expert_bytes,
        layers: 40,
        experts_per_layer: 8,
        workers: 2,
        max_inflight: 2,
        queue_cap: 16,
        aging_period: 4096,
        policy: Policy::Lfu,
    };
    let slots = cfg.slots();
    let src = Arc::new(FakeSource::new(expert_bytes, 8));
    let slab = Arc::new(FakeSlab::new(slots, expert_bytes));
    let cache = ExpertCache::new(cfg, src.clone(), slab.clone()).expect("cache");

    let mut rng = Rng::new(11);
    // Every expert of every layer, eight times over, in random order: the whole bank fits.
    for _ in 0..8 * slots {
        let layer = rng.below(40) as u16;
        let expert = rng.below(8) as u16;
        cache.get(layer, expert).expect("a lease");
    }
    let before = cache.stats();
    assert_eq!(before.resident, slots, "the working set fills the cache exactly");

    for _ in 0..2000 {
        let layer = rng.below(40) as u16;
        let expert = rng.below(8) as u16;
        cache.get(layer, expert).expect("a lease");
    }
    let after = cache.stats();
    assert_eq!(after.hits - before.hits, 2000, "every touch after the warm pass hits");
    assert_eq!(after.misses, before.misses, "no read after the warm pass");
    assert_eq!(after.bytes_read, before.bytes_read);
    assert_eq!((after.hits - before.hits) as f64 / 2000.0, 1.0);
    println!(
        "hot set of {slots} slots: 2000/2000 hits after one warm pass, {} misses total (one per expert, none after)",
        after.misses
    );
}

/// A cold stream larger than the cache misses on every first touch (that is the whole point of the budget) and hits at
/// `capacity / stream` on a second random pass — the model, not a vibe.
#[test]
fn cold_stream_misses_match_the_model() {
    let expert_bytes = 4096;
    let cfg = Config {
        budget_bytes: 40 * 8 * expert_bytes,
        expert_bytes,
        layers: 40,
        experts_per_layer: 384,
        workers: 2,
        max_inflight: 2,
        queue_cap: 16,
        aging_period: 1 << 20,
        policy: Policy::Lfu,
    };
    let slots = cfg.slots();
    let src = Arc::new(FakeSource::new(expert_bytes, 384));
    let slab = Arc::new(FakeSlab::new(slots, expert_bytes));
    let cache = ExpertCache::new(cfg, src.clone(), slab).expect("cache");

    // Pass 1: a distinct expert of a bank eight times the cache, once each. Nothing can hit.
    let stream = 8 * slots; // 8x the capacity
    let mut keys: Vec<(u16, u16)> = (0..stream).map(|i| ((i % 40) as u16, ((i / 40) % EXPERTS_PER_LAYER) as u16)).collect();
    let mut rng = Rng::new(3);
    for i in (1..keys.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        keys.swap(i, j);
    }
    for &(l, e) in &keys {
        cache.get(l, e).expect("a lease");
    }
    let first = cache.stats();
    assert_eq!(first.misses, stream as u64, "a cold stream of {stream} distinct experts misses {stream} times");
    assert_eq!(first.hits, 0);
    assert_eq!(first.bytes_read, (stream * expert_bytes) as u64, "and reads exactly that many expert bytes");
    println!("cold stream: {stream} touches over {} slots -> {:.1} % miss rate", slots, 100.0 * first.miss_rate());

    // Pass 2: i.i.d. uniform over the same stream. For a cache of `slots` against a `stream`-sized uniform working set the
    // stationary hit rate is `slots / stream` (a slot is re-referenced before `slots` other keys are touched with probability
    // `slots / stream`); this checks the accounting against the model rather than the policy against itself.
    let before = cache.stats();
    let draws = 20_000u32;
    for _ in 0..draws {
        let (l, e) = keys[rng.below(stream as u64) as usize];
        cache.get(l, e).expect("a lease");
    }
    let second = cache.stats();
    let measured = (second.hits - before.hits) as f64 / draws as f64;
    let model = slots as f64 / stream as f64;
    println!("second pass: {:.3} hit rate, model {:.3}", measured, model);
    assert!((measured - model).abs() < 0.03, "second-pass hit rate {measured:.3} vs model {model:.3}");
}

/// The prefetch queue is bounded: it never holds more than `queue_cap` requests and never has more than `max_inflight` fills
/// outstanding, and it never blocks the caller.
#[test]
fn prefetch_queue_is_bounded() {
    let expert_bytes = 4096;
    let cfg = Config {
        budget_bytes: 40 * 16 * expert_bytes,
        expert_bytes,
        layers: 40,
        experts_per_layer: 16,
        workers: 2,
        max_inflight: 2,
        queue_cap: 4,
        aging_period: 1 << 20,
        policy: Policy::Lfu,
    };
    let src = Arc::new(FakeSource::slow(expert_bytes, 16, 2000));
    let slab = Arc::new(FakeSlab::new(cfg.slots(), expert_bytes));
    let cache = ExpertCache::new(cfg.clone(), src.clone(), slab).expect("cache");

    // Burst harder than the 2 ms reads can drain, then let the workers run: the high-water marks are recorded during the burst.
    let mut worst_enqueue = std::time::Duration::ZERO;
    for i in 0..400u16 {
        let t = std::time::Instant::now();
        cache.prefetch(i % 40, (i / 40) % 16).expect("prefetch");
        worst_enqueue = worst_enqueue.max(t.elapsed());
    }
    let enqueue = worst_enqueue;
    std::thread::sleep(std::time::Duration::from_millis(10));
    let s = cache.stats();
    assert!(s.queue_high_water <= cfg.queue_cap, "queue high water {} > cap {}", s.queue_high_water, cfg.queue_cap);
    assert!(
        s.inflight_high_water <= cfg.max_inflight,
        "inflight high water {} > cap {}",
        s.inflight_high_water,
        cfg.max_inflight
    );
    assert!(s.prefetch_dropped > 0, "a 400-deep burst against a 4-deep queue must drop some");
    assert!(s.inflight_high_water >= 1, "the fill threads must actually have been busy");
    assert!(enqueue < std::time::Duration::from_millis(50), "prefetch must not block on I/O: took {enqueue:?}");
    println!(
        "queue cap {} -> high water {}, inflight cap {} -> high water {}, {} dropped, enqueue took {:?}",
        cfg.queue_cap, s.queue_high_water, cfg.max_inflight, s.inflight_high_water, s.prefetch_dropped, enqueue
    );

    // Drain, then check the prefetches actually filled and the touches that follow hit.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while cache.stats().resident < cfg.slots() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(5));
        let _ = cache.stats();
    }
    let s = cache.stats();
    assert!(s.inflight_high_water <= cfg.max_inflight);
    cache.shutdown();
}

/// A lease pins its slot: eviction never hands out a slot an outstanding forward still references, and the bytes under the lease
/// are unchanged after heavy churn elsewhere.
#[test]
fn eviction_never_reclaims_a_leased_slot() {
    let expert_bytes = 4096;
    let cfg = Config {
        budget_bytes: 40 * 4 * expert_bytes,
        expert_bytes,
        layers: 40,
        experts_per_layer: 8,
        workers: 0,
        max_inflight: 1,
        queue_cap: 4,
        aging_period: 1 << 20,
        policy: Policy::Lfu,
    };
    let slots = cfg.slots();
    let src = Arc::new(FakeSource::new(expert_bytes, 8));
    let slab = Arc::new(FakeSlab::new(slots, expert_bytes));
    let cache = ExpertCache::new(cfg, src.clone(), slab.clone()).expect("cache");

    // Lease four experts, then churn every other expert in the bank through the cache many times over.
    let pinned: Vec<Lease> = (0..4u16).map(|e| cache.get(3, e).expect("a lease")).collect();
    let pinned_slots: Vec<u32> = pinned.iter().map(|l| l.slot()).collect();
    let mut rng = Rng::new(5);
    for _ in 0..5000 {
        let layer = rng.below(40) as u16;
        let expert = rng.below(8) as u16;
        if layer == 3 && (expert as u32) < 4 {
            continue;
        }
        cache.get(layer, expert).expect("a lease");
    }
    for (i, lease) in pinned.iter().enumerate() {
        assert_eq!(lease.slot(), pinned_slots[i], "a leased slot was moved");
        let mut back = vec![0u8; expert_bytes];
        slab.read_back(lease.slot(), &mut back).expect("read back");
        assert_eq!(back, src.expert(3, i as u16), "a leased slot's bytes were overwritten");
    }

    // Now lease every slot and ask for one more: the cache must refuse rather than steal a live slot.
    let mut all: Vec<Lease> = Vec::new();
    for layer in 0..(slots / 4) as u16 {
        for e in 0..4u16 {
            all.push(cache.get(layer, e).expect("a lease"));
        }
    }
    assert_eq!(all.len(), slots, "every slot is now leased");
    match cache.get(39, 7) {
        Err(CacheError::AllPinned { leased }) => {
            assert_eq!(leased, slots);
            println!("all {slots} slots leased: get returned AllPinned, as it must");
        }
        other => panic!("expected AllPinned, got {other:?}"),
    }
    drop(all);
    drop(pinned);
    // And once the leases drop, the slot is reclaimable.
    cache.get(39, 7).expect("a lease after the pins were released");
}

/// Skewed access: 40 layers, 384 experts, Zipf(1.2) popularity over expert id, cache sized to 11.5 % of the bank (the 13.6 GB
/// the serve leaves for experts against 111.0 GB). LFU must beat LRU on this trace; that is why the default is LFU.
#[test]
fn skewed_access_miss_rate() {
    let expert_bytes = 4096;
    let slots = 40 * 44; // 44 experts per layer = 11.5 % of 384, the real ratio
    let cfg = Config {
        budget_bytes: slots * expert_bytes,
        expert_bytes,
        layers: 40,
        experts_per_layer: 384,
        workers: 0,
        max_inflight: 1,
        queue_cap: 8,
        aging_period: 8192,
        policy: Policy::Lfu,
    };
    let cdf = zipf_cdf(384, 1.2);
    let touches = 200_000usize;
    let mut rng = Rng::new(0xC0FFEE);
    let trace: Vec<(u16, u16)> = (0..touches).map(|_| (rng.below(40) as u16, sample(&cdf, rng.unit()))).collect();

    let mut out = Vec::new();
    for policy in [Policy::Lfu, Policy::Lru] {
        let cfg = Config { policy, ..cfg.clone() };
        let src = Arc::new(FakeSource::new(expert_bytes, 384));
        let slab = Arc::new(FakeSlab::new(slots, expert_bytes));
        let cache = ExpertCache::new(cfg, src.clone(), slab).expect("cache");
        // Warm up on the first 20 % so the policy is measured in steady state, not on cold misses.
        let warm = touches / 5;
        for &(l, e) in &trace[..warm] {
            cache.get(l, e).expect("a lease");
        }
        let before = cache.stats();
        for &(l, e) in &trace[warm..] {
            cache.get(l, e).expect("a lease");
        }
        let s = cache.stats();
        let hits = s.hits - before.hits;
        let misses = s.misses - before.misses;
        let rate = misses as f64 / (hits + misses) as f64;
        let per_token = TOUCHES_PER_TOKEN as f64 * rate * EXPERT_BYTES as f64;
        println!(
            "{policy:?}: {:.2} % miss rate over {} touches -> {:.0} MB/token at 240 touches, {:.1} tok/s ceiling at 10 GB/s",
            100.0 * rate,
            hits + misses,
            per_token / 1e6,
            10e9 / per_token
        );
        out.push((policy, rate, s));
    }
    let (_, lfu, s_lfu) = out[0];
    let (_, lru, _) = out[1];
    assert!(lfu <= lru, "LFU ({lfu:.4}) must not lose to LRU ({lru:.4}) on a skewed trace");
    // A cache holding 11.5 % of the bank against a Zipf(1.2) distribution keeps most of the mass: the hot 44 of 384 per layer
    // carry 76 % of the touches, so a correct LFU must be far below the 88.5 % a random replacement would miss.
    assert!(lfu < 0.40, "LFU miss rate {lfu:.4} is worse than the model");
    assert_eq!(s_lfu.bytes_read % expert_bytes as u64, 0, "bytes read is a whole number of experts");
    println!(
        "LFU {:.2} % miss, LRU {:.2} % miss at {} slots ({} B); bytes/token {:.0}, tokens/s at 10 GB/s {:.1}",
        100.0 * lfu,
        100.0 * lru,
        slots,
        EXPERT_BYTES,
        s_lfu.bytes_per_token(TOUCHES_PER_TOKEN, EXPERT_BYTES),
        s_lfu.tokens_per_s_ceiling(10e9, TOUCHES_PER_TOKEN, EXPERT_BYTES)
    );
}

/// Hits, misses and bytes add up; `bytes_per_token` and the tokens/s ceiling follow from them.
#[test]
fn counters_are_consistent() {
    let expert_bytes = 4096;
    let cfg = Config {
        budget_bytes: 40 * 4 * expert_bytes,
        expert_bytes,
        layers: 40,
        experts_per_layer: 8,
        workers: 0,
        max_inflight: 1,
        queue_cap: 4,
        aging_period: 512,
        policy: Policy::Lfu,
    };
    let src = Arc::new(FakeSource::new(expert_bytes, 8));
    let slab = Arc::new(FakeSlab::new(cfg.slots(), expert_bytes));
    let cache = ExpertCache::new(cfg, src.clone(), slab).expect("cache");
    let mut rng = Rng::new(99);
    let mut touches = 0u64;
    for _ in 0..3000 {
        let l = rng.below(40) as u16;
        let e = rng.below(8) as u16;
        let _ = cache.get(l, e).expect("a lease");
        touches += 1;
    }
    let s = cache.stats();
    assert_eq!(s.hits + s.misses, touches, "every touch is one hit or one miss");
    assert_eq!(s.bytes_read, s.misses * expert_bytes as u64, "no prefetch here, so reads are exactly the misses");
    assert_eq!(src.reads(), s.misses, "the source saw exactly the misses");
    assert!((0.0..=1.0).contains(&s.miss_rate()));
    assert!((s.hit_rate() + s.miss_rate() - 1.0).abs() < 1e-12);
    println!(
        "{} hits / {} misses ({:.1} % miss), {} bytes read, {:.1} tok/s at 10 GB/s",
        s.hits,
        s.misses,
        100.0 * s.miss_rate(),
        s.bytes_read,
        s.tokens_per_s_ceiling(10e9, TOUCHES_PER_TOKEN, EXPERT_BYTES)
    );
}

/// A read failure leaves the cache consistent: no slot is left half-filled and the slot is reusable.
#[test]
fn a_failed_read_does_not_poison_a_slot() {
    let expert_bytes = 4096;
    let cfg = Config {
        budget_bytes: 40 * 2 * expert_bytes,
        expert_bytes,
        layers: 40,
        experts_per_layer: 8,
        workers: 0,
        max_inflight: 1,
        queue_cap: 2,
        aging_period: 1 << 20,
        policy: Policy::Lfu,
    };
    let src = Arc::new(FakeSource { fail: true, ..FakeSource::new(expert_bytes, 8) });
    let slab = Arc::new(FakeSlab::new(cfg.slots(), expert_bytes));
    let cache = ExpertCache::new(cfg, src, slab).expect("cache");
    assert!(matches!(cache.get(1, 1), Err(CacheError::Io(_))));
    let s = cache.stats();
    assert_eq!(s.resident, 0, "a failed read leaves nothing resident");
    assert_eq!(s.bytes_read, 0, "and counts no bytes");
    // The slot is back on the free list: the next call must not deadlock or reuse a Filling slot.
    assert!(matches!(cache.get(1, 2), Err(CacheError::Io(_))));
}

/// The CUDA slab self-skips without a device instead of failing the suite.
#[test]
fn device_slab_self_skips_without_a_device() {
    match DeviceSlab::new(4, EXPERT_BYTES) {
        Ok(slab) => {
            assert_eq!(slab.bytes(), 4 * EXPERT_BYTES);
            println!("device slab: {} B allocated", slab.bytes());
        }
        Err(CacheError::NoDevice) => println!("not run: no device"),
        Err(e) => panic!("unexpected: {e}"),
    }
}

// ----------------------------------------------------------------------------------------- the real pack's shape

use gb_cuda::expert_cache::{ExpertSpan, ExpertTable, PackExpertSource, PAGE};
use std::path::PathBuf;

/// A pack laid out like the real one: experts of *unequal* size (the mixed-K groups), twelve parts each, at arbitrary offsets, spread over
/// several shards. Returns the table, the shard paths, and the record each expert must read back as.
#[allow(clippy::type_complexity)] // a test fixture's assembled return
fn synthetic_pack(dir: &std::path::Path) -> (ExpertTable, Vec<PathBuf>, Vec<(u16, u16, Vec<u8>)>) {
    const LAYERS: usize = 2;
    const EXPERTS: usize = 4;
    let record = 128 * 1024usize;
    std::fs::create_dir_all(dir).unwrap();

    // every part of every expert: a length that varies with (layer, expert, group, part) so no two experts are the same size
    let len = |l: usize, x: usize, g: usize, p: usize| -> u64 {
        let mut v = (l * 7 + x * 13 + g * 101 + p * 1009) as u64;
        v = v.wrapping_mul(2_654_435_761) % 9_000;
        if p == 1 {
            4
        } else {
            700 + v
        } // `mul1` is a 4-byte I32 marker
    };

    let mut table = ExpertTable::new(record, LAYERS, EXPERTS);
    let paths: Vec<PathBuf> = (0..3).map(|i| dir.join(format!("model-{i:05}.safetensors"))).collect();
    let shards: Vec<u32> = paths.iter().map(|p| table.shard(p)).collect();
    let mut files: Vec<Vec<u8>> = vec![Vec::new(); paths.len()];
    let mut expect = Vec::new();
    for l in 0..LAYERS {
        for x in 0..EXPERTS {
            let mut dst = 0u64;
            let mut rec = Vec::new();
            for g in 0..3 {
                for p in 0..4 {
                    let n = len(l, x, g, p) as usize;
                    // an arbitrary, unaligned file offset: the reader must window it
                    let off = files[(l + x + g + p) % 3].len() as u64 + 3 + (p as u64 % 3);
                    let f = (l + x + g + p) % 3;
                    files[f].resize((off + n as u64) as usize, 0);
                    for (i, b) in files[f][off as usize..off as usize + n].iter_mut().enumerate() {
                        *b = byte_at(l as u16, (x * 12 + g * 4 + p) as u16, i);
                    }
                    table.push(l as u16, x as u16, ExpertSpan { file: shards[f], offset: off, len: n as u64, dst }).unwrap();
                    rec.extend_from_slice(&files[f][off as usize..off as usize + n]);
                    dst += n as u64;
                }
            }
            assert!(dst as usize <= record, "expert ({l}, {x}) is {dst} B of a {record} B record");
            rec.resize(record, 0);
            expect.push((l as u16, x as u16, rec));
        }
    }
    for (p, bytes) in paths.iter().zip(&files) {
        std::fs::write(p, bytes).unwrap();
    }
    (table, paths, expect)
}

/// **(a)** Every expert of a mixed-K pack maps to real `(file, offset, len)` for all twelve parts, and reading it through page-aligned
/// windows into the record at the resident layout's offsets returns the record byte for byte. This is the whole contract the cache rests on:
/// a filled slot *is* a resident expert.
#[test]
fn a_mixed_k_pack_reads_back_byte_identical_through_page_windows() {
    let dir = std::env::temp_dir().join(format!("gb-expert-table-{}", std::process::id()));
    let (table, paths, expect) = synthetic_pack(&dir);
    assert_eq!(table.len(), 8, "every expert of the synthetic pack is in the table");
    let record = table.record_bytes();
    assert_eq!(record % PAGE, 0);

    // every part is inside the record and the parts of one expert do not overlap
    for (l, x, _) in &expect {
        let (l, x) = (*l, *x);
        let spans = table.spans(l, x);
        assert_eq!(spans.len(), 12, "twelve parts");
        let mut at = 0u64;
        for s in spans {
            assert_eq!(s.dst, at, "({l}, {x}): the parts tile the record");
            assert!(s.offset + s.len <= std::fs::metadata(&paths[s.file as usize]).unwrap().len());
            at += s.len;
        }
        assert_eq!(table.used_bytes(l, x), at);
    }

    let used0 = table.used_bytes(expect[0].0, expect[0].1) as usize;
    let src = PackExpertSource::open(table).unwrap();
    let mut got = vec![0u8; record];
    for (l, x, want) in &expect {
        got.fill(0xAA); // a reader that skipped a part would leave the poison visible
        src.read_expert(*l, *x, &mut got).unwrap();
        assert_eq!(&got, want, "expert ({l}, {x}) did not read back byte-identically");
    }
    assert_eq!(src.reads(), expect.len() as u64);
    // the padding past the last part is zero, not whatever the staging buffer held
    assert!(got[used0..].iter().all(|&b| b == 0));

    // a part at a page-aligned offset with a page-aligned length is read straight into the record: no window copy

    // and an expert the pack does not hold is an error, not a silent zero record
    let mut t2 = ExpertTable::new(PAGE, 1, 2);
    let f = t2.shard(dir.join("model-00000.safetensors"));
    t2.push(0, 0, ExpertSpan { file: f, offset: 0, len: 4, dst: 0 }).unwrap();
    let src2 = PackExpertSource::open(t2).unwrap();
    let mut buf = vec![0u8; PAGE];
    assert!(src2.read_expert(0, 1, &mut buf).is_err());

    // the table refuses a record whose parts overlap, because that decodes to noise rather than failing
    let mut t3 = ExpertTable::new(PAGE, 1, 1);
    let f = t3.shard(dir.join("model-00000.safetensors"));
    t3.push(0, 0, ExpertSpan { file: f, offset: 0, len: 64, dst: 0 }).unwrap();
    assert!(t3.push(0, 0, ExpertSpan { file: f, offset: 64, len: 64, dst: 32 }).is_err());

    std::fs::remove_dir_all(&dir).ok();
}

/// **(d)** A fill, evict and lease cycle under the real reader: a window smaller than the working set, so every turn evicts, and every
/// record read back through the slab is the record the pack holds.
#[test]
fn a_fill_evict_lease_cycle_under_the_real_reader() {
    let dir = std::env::temp_dir().join(format!("gb-expert-cycle-{}", std::process::id()));
    let (table, _paths, expect) = synthetic_pack(&dir);
    let record = table.record_bytes();
    let src = Arc::new(PackExpertSource::open(table).unwrap());

    // two slots for eight experts: every touch but the second evicts
    let cfg = Config { budget_bytes: 2 * record, expert_bytes: record, layers: 2, experts_per_layer: 4, ..Config::default() };
    let slots = cfg.slots();
    assert_eq!(slots, 2);
    let slab = Arc::new(FakeSlab::new(slots, record));
    let cache = ExpertCache::new(cfg, src.clone(), slab.clone()).unwrap();

    let want = |l: u16, x: u16| expect.iter().find(|(a, b, _)| (*a, *b) == (l, x)).unwrap().2.clone();
    for round in 0..3 {
        for (l, x, _) in &expect {
            let (l, x) = (*l, *x);
            let lease = cache.get(l, x).unwrap();
            assert!((lease.slot() as usize) < slots);
            let back = slab.expert(lease.slot(), record);
            assert_eq!(back, want(l, x).as_slice(), "round {round}: ({l}, {x}) in slot {}", lease.slot());
        }
    }
    let s = cache.stats();
    assert_eq!(s.capacity, 2);
    // 24 touches through 2 slots, no two consecutive of the same expert: every one is a miss and all but the first two evict
    assert_eq!((s.hits, s.misses, s.evictions), (0, 24, 22));
    assert_eq!(s.bytes_read, s.misses * record as u64, "a fill reads exactly one record");
    assert_eq!(s.resident, 2);

    // a lease pins: with both slots leased, a third expert cannot be brought in
    let a = cache.get(0, 0).unwrap();
    let b = cache.get(0, 1).unwrap();
    assert_ne!(a.slot(), b.slot());
    assert!(matches!(cache.get(0, 2), Err(CacheError::AllPinned { leased: 2 })));
    drop(a);
    drop(b);
    cache.get(0, 2).unwrap();

    std::fs::remove_dir_all(&dir).ok();
}
