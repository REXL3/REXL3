//! Rank 0's vision tower borrows its encoder scratch from the language model's prefill scratch (`REXL3_GLM_TOWER_LEND`, opt-in: rank 0
//! of a tensor-parallel group only; every shipped profile keeps the tower's own scratch).
//!
//! **Why it is sound.** The tower's working buffers (`vision::scratch_items`) and the routed experts' prefill scratch
//! (`gb_cuda::moe_prefill::PrefillScratch`) are both per-call temporaries: an encode writes every byte it reads before reading it, and a
//! prompt chunk does the same within its forward (a GLM row has the decode kernels' bits whatever ran before it: chunked == whole, row
//! independence). Neither keeps anything in them from one call to the next, so the two may use the same bytes as long as their work
//! never overlaps on the device. The encoder runs on its own thread and stream, unordered against the engine thread's forwards, so the
//! bytes are held under a [`ScratchLease`]: whoever enqueues work on them holds the lease for the whole enqueue, and the device work of
//! the other party is ordered with an event (`cudaStreamWaitEvent`), never a device-wide sync.
//!
//! **What is lent.** Six of the tower's buffers ([`LENT_ITEMS`]: the residual stream, the normed input, q|k|v, the attention output,
//! the sublayer output and gate|up), placed by [`place`] into the four large buffers of the routed experts' prefill scratch
//! ([`LENDABLE`]) at [`LEND_ALIGN`] boundaries. What does not fit stays the tower's own. The tower keeps the rest: the padded patch rows
//! (their pad columns must stay zero), the rope table and the counter (written once at boot), the GEMM workspace (cuBLAS's) and the
//! EXL3 staging. At `--max-image-tokens 4096` and a 2,048-row chunk all six fit: 512 MiB of rank 0's plan.
//!
//! **What each side leaves behind.** The tower zeroes every lent byte before it gives the lease back, so a forward sees the prefill
//! scratch as the boot left it wherever it reads bytes it does not write (the padding the boot zeroed); and a test fills the lent bytes
//! with 0xFF before an encode to show the features do not depend on what was there.

use crate::error::ModelError;
use gb_cuda::rt::Event;
use gb_cuda::{DevBuf, Stream};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

/// `REXL3_GLM_TOWER_LEND`: `1` / `on` lends rank 0's tower its encoder scratch out of the language model's prefill scratch (a
/// tensor-parallel group's rank 0 with images on; a follower holds no tower and ignores it); unset, `0` or `off` (the default, every
/// shipped profile) keeps the tower's own scratch. A single node is refused by name (its tower is loaded before its model). Anything else
/// is refused by name.
pub const TOWER_LEND_ENV: &str = "REXL3_GLM_TOWER_LEND";

/// [`TOWER_LEND_ENV`] from a lookup.
pub fn tower_lend_from(get: impl Fn(&str) -> Option<String>) -> Result<bool, String> {
    match get(TOWER_LEND_ENV).as_deref().map(str::trim) {
        None | Some("0" | "off") => Ok(false),
        Some("1" | "on") => Ok(true),
        Some(v) => Err(format!("{TOWER_LEND_ENV}={v:?}: expected 1, on, 0 or off (unset: the tower's own scratch)")),
    }
}

/// [`TOWER_LEND_ENV`] from the process's environment.
pub fn tower_lend() -> Result<bool, ModelError> {
    #[cfg(test)]
    if let Some(on) = OVERRIDE.with(|c| c.get()) {
        return Ok(on);
    }
    tower_lend_from(|k| std::env::var(k).ok()).map_err(ModelError::Config)
}

#[cfg(test)]
thread_local! {
    static OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// The unit tests' [`TOWER_LEND_ENV`]: `on` for this thread while `f` runs (the process's environment is shared by every test thread).
#[cfg(test)]
pub fn with_lend<T>(on: bool, f: impl FnOnce() -> T) -> T {
    OVERRIDE.with(|c| c.set(Some(on)));
    let r = f();
    OVERRIDE.with(|c| c.set(None));
    r
}

/// The tower's scratch items that may live in lent bytes, in placement order among equals (`vision::scratch_items`' names).
pub const LENT_ITEMS: [&str; 6] = ["gate_up", "qkv", "residual", "normed", "attention", "sublayer"];

/// The routed experts' prefill scratch buffers that are lent, in the order `PrefillScratch::lendable` returns them.
pub const LENDABLE: [&str; 4] = ["y_pairs", "xs_g", "xs_u", "xd"];

/// Every lent item starts at a multiple of this inside its region: the base alignment a large `cudaMalloc` gets, so a GEMM over a lent
/// buffer sees the same alignment class as over one of its own.
pub const LEND_ALIGN: u64 = 2 << 20;

/// One tower item placed in a lent region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lent {
    pub item: &'static str,
    pub region: usize,
    pub offset: u64,
    pub bytes: u64,
}

/// Where the tower's lendable items go ([`place`]); the items not listed stay the tower's own.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Placement {
    pub lent: Vec<Lent>,
}

impl Placement {
    pub fn get(&self, item: &str) -> Option<&Lent> {
        self.lent.iter().find(|l| l.item == item)
    }
    /// Bytes of the tower's scratch that are lent (not allocated by the tower).
    pub fn lent_bytes(&self) -> u64 {
        self.lent.iter().map(|l| l.bytes).sum()
    }
}

fn align(v: u64) -> u64 {
    v.div_ceil(LEND_ALIGN) * LEND_ALIGN
}

/// First fit decreasing: the [`LENT_ITEMS`] of `items` (the tower's `(name, bytes)`), largest first (ties in [`LENT_ITEMS`] order), each
/// into the first region of `regions` (bytes) with room after its last item at the next [`LEND_ALIGN`] boundary. A pure function of the
/// sizes: the plan and the boot both call it, and the boot checks that the two agree.
pub fn place(items: &[(&'static str, u64)], regions: &[u64]) -> Placement {
    let mut cand: Vec<(usize, &'static str, u64)> = items
        .iter()
        .filter_map(|&(n, b)| LENT_ITEMS.iter().position(|&x| x == n).map(|k| (k, n, b)))
        .filter(|&(_, _, b)| b > 0)
        .collect();
    cand.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
    let mut used = vec![0u64; regions.len()];
    let mut lent = Vec::new();
    for (_, item, bytes) in cand {
        if let Some(r) = (0..regions.len()).find(|&r| align(used[r]) + bytes <= regions[r]) {
            let offset = align(used[r]);
            used[r] = offset + bytes;
            lent.push(Lent { item, region: r, offset, bytes });
        }
    }
    Placement { lent }
}

/// The sizes of the regions a GLM model of `cfg` lends at a `chunk`-row prefill chunk (`GlmModel::new_chunked`): the [`LENDABLE`]
/// buffers of its routed experts' prefill scratch, none at a chunk of `alloc::MAX_ROWS` rows or fewer (no prefill scratch).
pub fn lendable_regions(cfg: &super::GlmConfig, chunk: usize) -> Result<Vec<u64>, ModelError> {
    if chunk <= super::alloc::MAX_ROWS {
        return Ok(Vec::new());
    }
    let items = gb_cuda::moe_prefill::PrefillScratch::items_for(&super::alloc::moe_dims(cfg), chunk)
        .map_err(|e| ModelError::Config(format!("the prefill scratch at {chunk} rows: {e}")))?;
    LENDABLE
        .iter()
        .map(|n| {
            items
                .iter()
                .find(|i| i.0 == *n)
                .map(|i| i.1 as u64)
                .ok_or_else(|| ModelError::Config(format!("the prefill scratch has no `{n}` to lend")))
        })
        .collect()
}

/// The two parties of a lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Party {
    /// The engine thread's prompt-chunk work on the prefill scratch (`GlmModel::forward` in `Mode::Prefill`, the NextN layer's prompt
    /// block): only work that reaches the routed experts' prefill scratch takes it.
    Model = 0,
    /// The tower's encodes (the encoder's worker thread).
    Tower = 1,
}

struct State {
    /// The party holding the lease (enqueueing work on the lent bytes), if any.
    holder: Option<Party>,
    /// The tickets of the acquires waiting, in arrival order: the lease goes to the front one only (first come, first served), so a
    /// party that gives the lease back and asks again at once (the model between two prompt chunks) queues behind a waiting encode.
    queue: std::collections::VecDeque<u64>,
    next_ticket: u64,
    /// The party whose event marks the last work queued on the lent bytes.
    last: Option<Party>,
    /// A holder was dropped without releasing (a panic mid-enqueue): the device ordering is unknown, so every later acquire fails.
    poisoned: bool,
    /// Leases taken by each party, and how many of them had to wait on the device for the other's work.
    taken: [u64; 2],
    waited: [u64; 2],
}

/// The lent bytes and the lease over them (shared by the model and the tower, `Arc`).
///
/// **The protocol.** A party takes the lease before it enqueues work that touches the lent bytes and gives it back once that work is
/// queued (not done): taking it orders the party's stream after the other party's last work with `cudaStreamWaitEvent` (an event per
/// party, recorded at each release), so the device runs the two parties' work on the lent bytes one after the other in the order the
/// lease was taken, and the host is held only while the other party enqueues. The lease is first come, first served (a ticket queue):
/// the model takes it per prompt chunk (each forward that reaches the prefill scratch, and the NextN layer's block of that chunk), so an
/// encode that arrives during a long prompt waits for at most the chunk being queued, never for the whole prompt. No wait is a device-wide sync. Neither party waits for
/// the other on the host while it holds the lease (the tower's critical section only launches; the engine thread never waits for an
/// encode), so no cycle can form: every device wait is on an event recorded strictly earlier in the lease's order.
pub struct ScratchLease {
    /// Views of the lender's buffers (they keep its blocks alive, whichever of the model and the tower is dropped first).
    regions: Vec<DevBuf>,
    state: Mutex<State>,
    freed: Condvar,
    events: [Event; 2],
}

impl std::fmt::Debug for ScratchLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScratchLease {{ regions: {:?} }}", self.region_bytes())
    }
}

// SAFETY: the views are device addresses (no host data behind them) and the events are CUDA handles; the lent bytes are only written by
// work enqueued while the lease is held, and the events are recorded and waited on with the state mutex held.
unsafe impl Sync for ScratchLease {}
unsafe impl Send for ScratchLease {}

/// How long the tower waits for the model to give the lease back before its encode fails by name (a chunk's forward holds it for a few
/// seconds at most; a model that never gives it back has failed).
pub const TOWER_WAIT: Duration = Duration::from_secs(300);

impl ScratchLease {
    /// A lease over `regions` (views of block-backed buffers: the tower takes sub-views of them).
    pub fn new(regions: Vec<DevBuf>) -> Result<Arc<Self>, ModelError> {
        if regions.iter().any(|r| r.block().is_none()) {
            return Err(ModelError::Config("a lent region must be a view of a block".into()));
        }
        Ok(Arc::new(ScratchLease {
            regions,
            state: Mutex::new(State {
                holder: None,
                queue: Default::default(),
                next_ticket: 0,
                last: None,
                poisoned: false,
                taken: [0; 2],
                waited: [0; 2],
            }),
            freed: Condvar::new(),
            events: [Event::new()?, Event::new()?],
        }))
    }

    /// The regions' sizes ([`place`]'s `regions`).
    pub fn region_bytes(&self) -> Vec<u64> {
        self.regions.iter().map(|r| r.bytes() as u64).collect()
    }

    /// A view of the bytes `l` was placed at.
    pub fn view(&self, l: &Lent) -> Result<DevBuf, ModelError> {
        let r = self.regions.get(l.region).ok_or_else(|| ModelError::Config(format!("no lent region {}", l.region)))?;
        if l.offset + l.bytes > r.bytes() as u64 {
            return Err(ModelError::Config(format!("`{}` placed past the end of lent region {}", l.item, l.region)));
        }
        Ok(r.subview(l.offset as usize, l.bytes as usize))
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Takes the lease for `party`, whose work goes on `st`: queues behind the acquires already waiting (first come, first served) and
    /// waits until it is free (the tower at most [`TOWER_WAIT`]), then orders `st` after the other party's last work on the lent bytes.
    /// Give it back with [`Held::release`] once the work is queued.
    pub fn acquire(self: &Arc<Self>, party: Party, st: &Stream) -> Result<Held, ModelError> {
        self.acquire_within(party, st, if party == Party::Tower { TOWER_WAIT } else { Duration::MAX })
    }

    /// [`ScratchLease::acquire`] that gives up after `limit` (refused by name, its place in the queue given up).
    pub fn acquire_within(self: &Arc<Self>, party: Party, st: &Stream, limit: Duration) -> Result<Held, ModelError> {
        let mut g = self.lock();
        if g.holder == Some(party) {
            return Err(ModelError::Config(format!("the lent scratch's lease taken twice by {party:?}")));
        }
        let ticket = g.next_ticket;
        g.next_ticket += 1;
        g.queue.push_back(ticket);
        let deadline = std::time::Instant::now().checked_add(limit);
        loop {
            if g.poisoned {
                g.queue.retain(|&t| t != ticket);
                self.freed.notify_all();
                return Err(ModelError::StateLost("the lent scratch's lease was dropped mid-use".into()));
            }
            if g.holder.is_none() && g.queue.front() == Some(&ticket) {
                g.queue.pop_front();
                break;
            }
            let now = std::time::Instant::now();
            match deadline {
                Some(d) if now >= d => {
                    g.queue.retain(|&t| t != ticket);
                    self.freed.notify_all();
                    return Err(ModelError::Busy(format!(
                        "the {party:?} side waited {:.0} s for the lent prefill scratch; retry shortly",
                        limit.as_secs_f64()
                    )));
                }
                Some(d) => g = self.freed.wait_timeout(g, d - now).unwrap_or_else(|e| e.into_inner()).0,
                None => g = self.freed.wait(g).unwrap_or_else(|e| e.into_inner()),
            }
        }
        g.taken[party as usize] += 1;
        if let Some(other) = g.last.filter(|&p| p != party) {
            g.waited[party as usize] += 1;
            if let Err(e) = gb_cuda::graph::wait(st, &self.events[other as usize]) {
                g.poisoned = true;
                self.freed.notify_all();
                return Err(e.into());
            }
        }
        g.holder = Some(party);
        Ok(Held { lease: self.clone(), party, released: false })
    }

    /// Leases taken and leases that waited on the device, per party (`[model, tower]`).
    pub fn counts(&self) -> ([u64; 2], [u64; 2]) {
        let g = self.lock();
        (g.taken, g.waited)
    }

    /// Whether nobody holds the lease and it is not poisoned (the tests' check that every holder gave it back).
    pub fn is_free(&self) -> bool {
        let g = self.lock();
        g.holder.is_none() && !g.poisoned
    }

    /// Debug (the tests): fill every lent byte with `byte` on `st`, as the model's party (what a forward could leave behind).
    pub fn fill(self: &Arc<Self>, byte: u8, st: &Stream) -> Result<(), ModelError> {
        let h = self.acquire(Party::Model, st)?;
        for r in &self.regions {
            r.subview(0, r.bytes()).memset_async(byte, st)?;
        }
        h.release(st)
    }
}

/// A held lease ([`ScratchLease::acquire`]). Dropped without [`Held::release`] (a panic or an early return mid-enqueue) it poisons the
/// lease, so the other party fails by name instead of running on bytes whose last writer is unknown.
pub struct Held {
    lease: Arc<ScratchLease>,
    party: Party,
    released: bool,
}

impl Held {
    /// Marks the end of this party's queued work on `st` (the stream it was acquired for) and gives the lease back.
    pub fn release(mut self, st: &Stream) -> Result<(), ModelError> {
        self.released = true;
        let mut g = self.lease.lock();
        g.holder = None;
        let r = self.lease.events[self.party as usize].record(st);
        match &r {
            Ok(()) => g.last = Some(self.party),
            Err(_) => g.poisoned = true,
        }
        self.lease.freed.notify_all();
        r.map_err(Into::into)
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        if !self.released {
            let mut g = self.lease.lock();
            g.holder = None;
            g.poisoned = true;
            self.lease.freed.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn the_lend_setting_parses_or_is_refused_by_name() {
        let of = |v: Option<&str>| tower_lend_from(|k| (k == TOWER_LEND_ENV).then(|| v.map(String::from)).flatten());
        assert_eq!(of(None), Ok(false));
        for (v, want) in [("0", false), ("off", false), ("1", true), ("on", true), (" on ", true)] {
            assert_eq!(of(Some(v)), Ok(want), "{v:?}");
        }
        for bad in ["", "yes", "true", "2", "ON"] {
            assert!(of(Some(bad)).unwrap_err().contains(TOWER_LEND_ENV), "{bad:?}");
        }
    }

    fn tower(patches: usize) -> Vec<(&'static str, u64)> {
        super::super::vision::scratch_items(patches, crate::families::glm::config::VisionTowerFormat::Bf16)
    }

    /// RED-SNOW / GLM-5.3-Flash at the profile's `--max-image-tokens 4096` and `--prefill-chunk 2048`: all six lendable items fit, 512 MiB.
    #[test]
    fn at_the_profiles_cap_and_chunk_the_six_buffers_fit_the_experts_scratch() {
        let cfg =
            super::super::GlmConfig::from_dir(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/glm53"))
                .unwrap();
        let regions = lendable_regions(&cfg, 2048).unwrap();
        eprintln!("lendable at 2048 rows (MiB): {:?}", regions.iter().map(|r| *r as f64 / MIB as f64).collect::<Vec<_>>());
        let p = place(&tower(4096 * 4), &regions);
        eprintln!("{p:?}");
        assert_eq!(p.lent.len(), 6, "{p:?}");
        assert_eq!(p.lent_bytes(), 512 * MIB);
        assert_eq!(p.get("gate_up").map(|l| (l.region, l.offset)), Some((0, 0)), "gate|up fills y_pairs");
        for l in &p.lent {
            assert_eq!(l.offset % LEND_ALIGN, 0);
            assert!(l.offset + l.bytes <= regions[l.region]);
        }
        // no two items of a region overlap
        for a in &p.lent {
            for b in &p.lent {
                if a != b && a.region == b.region {
                    assert!(a.offset + a.bytes <= b.offset || b.offset + b.bytes <= a.offset, "{a:?} {b:?}");
                }
            }
        }
        // the 8,000-token default cap needs more than the experts' scratch holds at 2,048 rows: some items stay the tower's own
        let big = place(&tower(8000 * 4), &regions);
        assert!(big.lent.len() < 6 && big.lent_bytes() > 0, "{big:?}");
        // no prefill scratch at a chunk of 128 rows: nothing is lent
        assert!(lendable_regions(&cfg, 128).unwrap().is_empty());
        assert_eq!(place(&tower(4096 * 4), &[]), Placement::default());
    }

    #[test]
    fn placement_is_first_fit_decreasing_and_partial_when_room_is_short() {
        let items = [("gate_up", 10 * MIB), ("qkv", 6 * MIB), ("residual", 4 * MIB), ("patches", 3 * MIB), ("normed", 2 * MIB)];
        // one region of 12 MiB: gate|up at 0; qkv and the residual do not fit after it (the 10 MiB boundary), normed does (10 + 2 = 12)
        let p = place(&items, &[12 * MIB]);
        assert_eq!(
            p.lent.iter().map(|l| (l.item, l.region, l.offset)).collect::<Vec<_>>(),
            vec![("gate_up", 0, 0), ("normed", 0, 10 * MIB)]
        );
        // a second region of 7 MiB takes qkv; the residual fits neither (10 + 4 > 12, 6 + 4 > 7); normed the first region's tail
        let p = place(&items, &[12 * MIB, 7 * MIB]);
        assert_eq!(
            p.lent.iter().map(|l| (l.item, l.region, l.offset)).collect::<Vec<_>>(),
            vec![("gate_up", 0, 0), ("qkv", 1, 0), ("normed", 0, 10 * MIB)]
        );
        assert!(p.get("residual").is_none());
        // `patches` is never lent whatever the room
        let p = place(&items, &[100 * MIB]);
        assert!(p.get("patches").is_none() && p.lent.len() == 4);
        assert_eq!(p.get("normed").unwrap().offset, 20 * MIB, "after 10 + 6 + 4 MiB, each at a 2 MiB boundary");
    }
}
