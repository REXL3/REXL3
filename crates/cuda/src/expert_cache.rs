//! An expert-granular page cache: the routed experts that do not fit in device memory live in the pack on NVMe and are paged
//! into a fixed slab, one 7.72 MB slot per expert, as the router asks for them.
//!
//! ## Why this exists
//!
//! The DeepSeek-V4.1 pack is 332.5 GB. Engram (203 GB) is mapped and streamed; the rest (129.4 GB) is resident against a
//! 121 GiB unified device, so 111.0 GB of expert bank minus what is left over leaves roughly 13.6 GB of the 15,360 routed
//! experts (40 layers x 384) that must be read from NVMe on demand. One expert is 7.72 MB of trellis plus `suh`/`svh`; a
//! decode token touches top-6 of 384 on every layer, 240 expert touches; a 512-token prefill chunk touches essentially all
//! 384 experts of every layer, so prefill is a sequential sweep of the whole bank and no cache size makes it free.
//!
//! ## The shape
//!
//! * [`ExpertCache`] owns a **fixed** number of slots, `budget_bytes / expert_bytes`, rounded down to a multiple of the layer
//!   count. Nothing here grows with demand: the slab is allocated once, at construction, and the slot count never changes.
//!   On a GB10 the page cache and the GPU allocations share one 121 GiB pool, so an unbounded cache is not a cache, it is an
//!   OOM waiting for the allocator to lose a race.
//! * [`ExpertCache::get`] returns a [`Lease`] naming a resident slot, or evicts one and fills it. A lease pins its slot: no
//!   eviction may return a slot while a forward still holds it (a lease is what the GEMM reads from).
//! * [`ExpertCache::prefetch`] takes the router's selection and queues the misses on a **bounded** queue drained by fill
//!   threads, so a decode step can issue its 240 fills and then run the shared-expert and attention work while the reads
//!   land, instead of blocking on 1.85 GB of I/O.
//! * [`Policy::Lfu`] is the default eviction policy: least-frequently-used with periodic ageing, not LRU. See [`Policy`].
//! * [`CacheStats`] carries hits, misses, bytes read and the observed miss rate — the number that decides whether a serve is
//!   viable at all, and the one the report quotes.
//!
//! ## Testability
//!
//! Reads go through [`ExpertSource`] and writes through [`Slab`], so the whole policy — eviction, pinning, the queue, the
//! counters — runs on the host with a fake file and a fake device. [`DeviceSlab`] is the only CUDA-dependent type and it
//! self-skips with `not run: no device` on a build without a GPU.
//!
//! ## Byte accounting
//!
//! Everything that is allocated is accounted for in one place, [`Config::slab_bytes`] (device) and
//! [`Config::staging_bytes`] (host). Neither is a function of the trace.

use std::alloc::Layout;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use gb_exl3::checkpoint::Checkpoint;

/// Page size assumed for `O_DIRECT` and for rounding an expert up to a whole number of pages.
pub const PAGE: usize = 4096;

/// Where every part of an expert record starts, relative to the record (and every resident record relative to its region): a
/// multiple of this. The `mul1` codebook marker is 4 bytes, so a record of parts packed back to back puts `suh`, `svh` and every
/// later projection 4, 8 or 12 bytes off a 16-byte boundary, and the vectorised loads of the EXL3 and scale kernels fault with
/// `cudaErrorMisalignedAddress`. 256 bytes is what a `cudaMalloc` base guarantees, so a part at an aligned offset of an aligned
/// region is as aligned as its own allocation would be. The resident layout (`deepseek_v41::weights::Expert::record_layout`)
/// walks the parts with the same rule; a streamed slot is byte-identical to a resident record only because both do.
pub const EXPERT_PART_ALIGN: u64 = 256;

/// The bytes of one expert: `w1` + `w2` + `w3` trellis with their `suh`/`svh`, measured from the pack.
pub const EXPERT_RAW_BYTES: usize = 7_720_000;

/// [`EXPERT_RAW_BYTES`] rounded up to a page, so every expert starts on a page boundary and an `O_DIRECT` read of one needs no
/// bounce buffer.
pub const EXPERT_BYTES: usize = EXPERT_RAW_BYTES.div_ceil(PAGE) * PAGE;

/// Trunk layers of the model.
pub const LAYERS: usize = 40;

/// Routed experts per layer.
pub const EXPERTS_PER_LAYER: usize = 384;

/// Experts the router selects per layer per token (top-6 of 384), times [`LAYERS`]: the expert touches in one decode step.
pub const TOUCHES_PER_TOKEN: usize = 6 * LAYERS;

/// Why an operation could not be served.
#[derive(Debug)]
pub enum CacheError {
    /// The build has no CUDA runtime and no device was found (`not run: no device`).
    NoDevice,
    /// The pack could not be read.
    Io(String),
    /// Every slot is leased by an outstanding forward: the caller holds more experts live than the budget has room for. Drop a
    /// lease and retry; the cache will not steal a slot that a GEMM may still be reading.
    AllPinned { leased: usize },
    /// The cache was shut down.
    Shutdown,
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::NoDevice => write!(f, "not run: no device"),
            CacheError::Io(m) => write!(f, "{m}"),
            CacheError::AllPinned { leased } => write!(f, "all {leased} slots are leased by outstanding forwards"),
            CacheError::Shutdown => write!(f, "cache is shut down"),
        }
    }
}

impl std::error::Error for CacheError {}

impl From<io::Error> for CacheError {
    fn from(e: io::Error) -> Self {
        CacheError::Io(e.to_string())
    }
}

/// Which resident expert a miss evicts.
///
/// * [`Policy::Lfu`] (default) — least-frequently-used, aged. Expert popularity is heavily skewed (a handful of experts serve
///   most tokens in every layer), and the trace interleaves layers: a token's 240 touches arrive layer by layer, so a hot
///   expert of layer 7 is re-referenced only 39 layers later, behind 234 other slots' worth of pressure. Plain LRU ranks by
///   *recency alone*, so any burst of one-shot experts (a prefill chunk, a domain shift) walks the hot set out of the cache
///   and the next token re-reads it: classic LRU thrash on a skewed distribution. LFU ranks by *how often*, which is the
///   quantity the skew is in, so the hot set stays put under a stream of cold misses.
/// * Ageing bounds that: every `aging_period` accesses all counters halve, so an expert that was hot an hour ago and is cold
///   now decays to the same rank as everything else instead of squatting. Without ageing LFU is not adaptive at all.
/// * Ties break on the older last-touch, so among equally-frequent slots the stale one goes — that is the LRU part, and it is
///   what gives a newly filled slot (frequency 1) a full aging period to prove itself before it can be picked first.
/// * [`Policy::Lru`] exists so the choice can be measured rather than asserted: same cache, same trace, recency instead of
///   frequency. `tests/expert_cache.rs` reports both.
///
/// ARC was the other candidate. Its ghost lists buy scan resistance for a *single* sequential stream, but this working set is
/// 40 independent skewed distributions over a fixed 384 items each, which is LFU's home ground; and ARC's two lists and two
/// ghost buffers cost state per slot to save a policy that is already O(1) per access here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Least-frequently-used with periodic ageing (the default).
    Lfu,
    /// Least-recently-used, for comparison.
    Lru,
}

/// Everything the cache is allowed to allocate, and how it behaves. All of it is explicit; none of it is "whatever malloc
/// gives".
#[derive(Clone, Debug)]
pub struct Config {
    /// Device bytes for expert slots. The slot count is `budget_bytes / expert_bytes`, rounded down to a multiple of `layers`.
    pub budget_bytes: usize,
    /// Bytes per expert slot (page-aligned; [`EXPERT_BYTES`] for the real pack).
    pub expert_bytes: usize,
    /// Layers whose experts share the slab ([`LAYERS`]).
    pub layers: usize,
    /// Experts per layer ([`EXPERTS_PER_LAYER`]); the expert id must be below it.
    pub experts_per_layer: usize,
    /// Fill threads. Each one reads a whole expert at a time.
    pub workers: usize,
    /// Most fills that may be reading at once. `get` and the workers both draw from this pool, so it is a hard cap on bytes in
    /// flight (`max_inflight * expert_bytes` of host staging) and on device-side concurrency.
    pub max_inflight: usize,
    /// Most prefetches that may wait for a fill thread. [`ExpertCache::prefetch`] never blocks: past this depth it drops the
    /// request and counts it, because a decode step that stalls on the queue is worse than a miss that arrives one token late.
    pub queue_cap: usize,
    /// Accesses between ageings (all frequency counters halve).
    pub aging_period: u64,
    /// Eviction policy.
    pub policy: Policy,
}

impl Default for Config {
    /// The serve this module exists for: 13.6 GB of slab against the 111.0 GB expert bank, two fill threads.
    fn default() -> Self {
        Self {
            budget_bytes: 13_600_000_000,
            expert_bytes: EXPERT_BYTES,
            layers: LAYERS,
            experts_per_layer: EXPERTS_PER_LAYER,
            workers: 2,
            max_inflight: 2,
            queue_cap: 32,
            aging_period: 4096,
            policy: Policy::Lfu,
        }
    }
}

impl Config {
    /// Slots the budget buys: `budget_bytes / expert_bytes`, rounded down to a multiple of the layer count (so the per-layer
    /// fair share is a whole number of slots), and at least one. The cache is never larger than the budget.
    pub fn slots(&self) -> usize {
        let raw = self.budget_bytes / self.expert_bytes.max(1);
        if raw == 0 {
            return 0;
        }
        if raw >= self.layers.max(1) {
            (raw / self.layers) * self.layers
        } else {
            raw
        }
    }

    /// Device bytes the slab occupies: `slots * expert_bytes <= budget_bytes`.
    pub fn slab_bytes(&self) -> usize {
        self.slots() * self.expert_bytes
    }

    /// Host bytes the staging pool may occupy: one buffer per in-flight fill, plus one spare.
    pub fn staging_bytes(&self) -> usize {
        (self.max_inflight + 1) * (self.expert_bytes + PAGE)
    }

    /// Per-layer fair share of the slots. Eviction above this share stays inside the layer that missed; below it, a layer may
    /// borrow from the whole cache. Equal shares are the right prior: every layer has the same 384-way router.
    fn quota(&self) -> usize {
        let n = self.slots();
        if n >= self.layers.max(1) {
            n / self.layers
        } else {
            n.max(1)
        }
    }
}

/// A page-aligned host buffer, the memory an `O_DIRECT` read needs.
struct AlignedBuf {
    ptr: *mut u8,
    cap: usize,
}

unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    fn new(cap: usize) -> Self {
        let layout = Layout::from_size_align(cap.max(PAGE), PAGE).expect("a valid staging layout");
        // SAFETY: the layout has a non-zero size
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { ptr, cap: layout.size() }
    }
    fn slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` points at `cap` initialised bytes owned by this buffer
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.cap) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with the same layout
        unsafe { std::alloc::dealloc(self.ptr, Layout::from_size_align(self.cap, PAGE).expect("a valid staging layout")) }
    }
}

/// Where an expert's bytes come from.
pub trait ExpertSource: Send + Sync + 'static {
    /// Read expert `expert` of `layer` into `dst`, which is exactly one expert long and page-aligned.
    fn read_expert(&self, layer: u16, expert: u16, dst: &mut [u8]) -> io::Result<()>;
    /// Bytes of one expert.
    fn expert_bytes(&self) -> usize;
}

/// The pack on disk, read with `O_DIRECT` so a page-in never pollutes the page cache that the rest of the serve shares.
pub struct DirectFile {
    file: File,
    path: PathBuf,
    expert_bytes: usize,
    experts_per_layer: usize,
    /// Experts read, for the report.
    reads: std::sync::atomic::AtomicU64,
}

/// Open a file for O_DIRECT reads. macOS has no `O_DIRECT` flag; it sets the
/// F_NOCACHE page-cache bypass instead (best-effort, no alignment requirement),
/// mirroring the loader's macOS branch.
#[cfg(target_os = "macos")]
fn open_direct(path: &Path) -> io::Result<File> {
    use std::os::unix::io::AsRawFd;
    let file = File::open(path)?;
    unsafe {
        libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
    }
    Ok(file)
}

#[cfg(not(target_os = "macos"))]
fn open_direct(path: &Path) -> io::Result<File> {
    std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path)
}

impl DirectFile {
    /// Open the pack. `expert_bytes` must be a multiple of [`PAGE`].
    pub fn open(path: impl AsRef<Path>, expert_bytes: usize, experts_per_layer: usize) -> io::Result<Self> {
        assert!(expert_bytes % PAGE == 0, "an O_DIRECT expert must be page-aligned");
        assert!(experts_per_layer > 0 && experts_per_layer <= u16::MAX as usize + 1);
        let path = path.as_ref().to_path_buf();
        let file = open_direct(&path)?;
        Ok(Self { file, path, expert_bytes, experts_per_layer, reads: AtomicU64::new(0) })
    }

    /// Experts this file has read.
    pub fn reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }
}

impl ExpertSource for DirectFile {
    fn read_expert(&self, layer: u16, expert: u16, dst: &mut [u8]) -> io::Result<()> {
        use std::os::unix::fs::FileExt;
        let idx = layer as u64 * self.experts_per_layer as u64 + expert as u64;
        let off = idx * self.expert_bytes as u64;
        let mut done = 0;
        while done < dst.len() {
            match self.file.read_at(&mut dst[done..], off + done as u64) {
                Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short expert read")),
                Ok(n) => done += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(io::Error::new(e.kind(), format!("{}: {e}", self.path.display()))),
            }
        }
        self.reads.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn expert_bytes(&self) -> usize {
        self.expert_bytes
    }
}

// -------------------------------------------------------------------------------------------------- the real pack

/// One part of one expert: the shard it lives in, its byte range there, and where it goes inside the expert's record.
///
/// The record is the resident layout's: `w1`'s `trellis`,`mul1`,`suh`,`svh`, then `w2`'s, then `w3`'s, laid end to end with no
/// padding ([`super::loader`]'s `expert_part_offset` is the same walk). `dst` is therefore redundant with the order of the spans, and
/// it is kept explicit so a reader cannot silently reorder them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertSpan {
    pub file: u32,
    pub offset: u64,
    pub len: u64,
    pub dst: u64,
}

/// `(layer, expert) -> the parts of its record`, for a pack whose experts are EXL3 groups of *unequal* size spread over many shards at
/// arbitrary safetensors offsets.
///
/// This is the piece [`DirectFile`] assumes away. A real DeepSeek-V4.1 expert is three groups of 1..6-bit trellis plus two fp16 scale
/// vectors and a 4-byte `mul1` marker; neither the group nor the expert is a fixed number of bytes, and the twelve parts of one expert are
/// twelve separate ranges, often in four different shards. The table records them once, at boot, from the pack's own headers.
#[derive(Clone, Debug)]
pub struct ExpertTable {
    record_bytes: usize,
    layers: usize,
    experts_per_layer: usize,
    files: Vec<PathBuf>,
    spans: HashMap<u32, Vec<ExpertSpan>>,
}

impl ExpertTable {
    /// An empty table over records of `record_bytes` (page-aligned: one `O_DIRECT` slot).
    pub fn new(record_bytes: usize, layers: usize, experts_per_layer: usize) -> Self {
        assert!(record_bytes % PAGE == 0, "an O_DIRECT expert record must be page-aligned");
        Self { record_bytes, layers, experts_per_layer, files: Vec::new(), spans: HashMap::new() }
    }

    /// Register a shard, returning its id. The same path registered twice is one id.
    pub fn shard(&mut self, path: impl Into<PathBuf>) -> u32 {
        let path = path.into();
        if let Some(i) = self.files.iter().position(|p| *p == path) {
            return i as u32;
        }
        self.files.push(path);
        (self.files.len() - 1) as u32
    }

    /// Record one part. `dst + len` must stay inside the record, and a record's spans must not overlap: laid out wrong, an expert decodes
    /// to noise rather than failing, so the table refuses the overlap here.
    pub fn push(&mut self, layer: u16, expert: u16, span: ExpertSpan) -> Result<(), String> {
        if layer as usize >= self.layers || expert as usize >= self.experts_per_layer {
            return Err(format!("expert ({layer}, {expert}) is outside a {}-layer bank", self.layers));
        }
        if span.file as usize >= self.files.len() {
            return Err(format!("expert ({layer}, {expert}) names shard {} of {}", span.file, self.files.len()));
        }
        if span.dst + span.len > self.record_bytes as u64 {
            return Err(format!(
                "expert ({layer}, {expert}): a {}-byte part at {} runs past the {}-byte record",
                span.len, span.dst, self.record_bytes
            ));
        }
        let key = layer as u32 * self.experts_per_layer as u32 + expert as u32;
        let v = self.spans.entry(key).or_default();
        for s in v.iter() {
            if span.dst < s.dst + s.len && s.dst < span.dst + span.len {
                return Err(format!(
                    "expert ({layer}, {expert}): the part at {}..{} overlaps the one at {}..{}",
                    span.dst,
                    span.dst + span.len,
                    s.dst,
                    s.dst + s.len
                ));
            }
        }
        v.push(span);
        v.sort_by_key(|s| s.dst);
        Ok(())
    }

    /// Every expert of a trunk, from a scanned pack: the twelve parts of `layers.L.ffn.experts.E.{w1,w2,w3}.*`, laid out in the resident
    /// order. A layer or expert the pack does not hold is simply absent; a part that is present but has no room in the record is an error.
    pub fn from_checkpoint(
        ck: &Checkpoint,
        record_bytes: usize,
        layers: usize,
        experts_per_layer: usize,
    ) -> Result<Self, String> {
        let mut t = Self::new(record_bytes, layers, experts_per_layer);
        let mut shards: HashMap<String, u32> = HashMap::new();
        for path in ck.files() {
            // `files()` names the shard files relative to the pack root; the source opens them, so they are absolute here
            let full = ck.root.join(path);
            shards.insert(path.to_string(), t.shard(full));
        }
        for l in 0..layers {
            for x in 0..experts_per_layer {
                let mut dst = 0u64;
                for proj in ["w1", "w2", "w3"] {
                    let prefix = format!("layers.{l}.ffn.experts.{x}.{proj}");
                    let Ok(parts) = ck.group_parts(&prefix) else { continue };
                    let g = parts.group;
                    let mut order: Vec<(&str, &gb_exl3::checkpoint::Entry)> =
                        vec![("trellis", parts.trellis), (g.in_scales, parts.in_scales), (g.out_scales, parts.out_scales)];
                    // the codebook marker is a 4-byte I32 scalar beside the group (`mul1` here); it is part of the record
                    for marker in ["mul1", "mcg"] {
                        if let Some(e) = ck.part(&format!("{prefix}.{marker}")) {
                            order.push((marker, e));
                            break;
                        }
                    }
                    if let Some(b) = parts.bias {
                        order.push(("bias", b));
                    }
                    // the resident order is trellis, mul1, suh, svh — not the order the parts are convenient to name in
                    order.sort_by_key(|(n, _)| match *n {
                        "trellis" => 0,
                        "mul1" | "mcg" => 1,
                        n if n == g.in_scales => 2,
                        _ => 3,
                    });
                    for (_, e) in order {
                        let len = e.offsets.1 - e.offsets.0;
                        let file =
                            *shards.get(&e.file).ok_or_else(|| format!("{prefix}: shard {} is not in the scan", e.file))?;
                        // every part starts on an EXPERT_PART_ALIGN boundary, the same walk as the resident record
                        dst = dst.div_ceil(EXPERT_PART_ALIGN) * EXPERT_PART_ALIGN;
                        // `Entry::offsets` is the safetensors `data_offsets` (relative to the data section); a read is at
                        // `8 + header_len + data_offsets`, which is what `file_offset` builds
                        let offset = ck.file_offset(e).map_err(|why| format!("{prefix}: {why}"))?;
                        t.push(l as u16, x as u16, ExpertSpan { file, offset, len, dst })?;
                        dst += len;
                    }
                }
            }
        }
        Ok(t)
    }

    /// The record size every expert is padded to.
    pub fn record_bytes(&self) -> usize {
        self.record_bytes
    }

    /// Bytes of the record an expert actually uses (`0` for one the pack does not hold).
    pub fn used_bytes(&self, layer: u16, expert: u16) -> u64 {
        self.spans(layer, expert).last().map_or(0, |s| s.dst + s.len)
    }

    /// The parts of one expert, in record order.
    pub fn spans(&self, layer: u16, expert: u16) -> &[ExpertSpan] {
        self.spans.get(&(layer as u32 * self.experts_per_layer as u32 + expert as u32)).map_or(&[], |v| v.as_slice())
    }

    pub fn shards(&self) -> &[PathBuf] {
        &self.files
    }

    /// Experts the table has a record for.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

thread_local! {
    /// The one page-aligned bounce window a fill thread needs for a part whose destination or source is not page-aligned.
    static BOUNCE: std::cell::RefCell<Option<AlignedBuf>> = const { std::cell::RefCell::new(None) };
}

fn read_exact_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    read_need_at(f, buf, off, buf.len())
}

/// `read_need_at` that requires `need` bytes and tolerates a short read past the end of the file (the last page block).
fn read_need_at(f: &File, buf: &mut [u8], off: u64, need: usize) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        match f.read_at(&mut buf[done..], off + done as u64) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    if done < need {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, format!("read {done} of {need} bytes at {off}")));
    }
    Ok(())
}

/// The real pack, read one expert at a time: each of its parts `O_DIRECT`, through a page-aligned window, into the record at the offset
/// the resident layout uses. A filled slot is byte-for-byte what a resident expert would have been.
pub struct PackExpertSource {
    table: ExpertTable,
    /// `O_DIRECT` handles, opened lazily; `None` where the filesystem refused one and the part fell back to a buffered read.
    direct: Vec<Option<File>>,
    buffered: Vec<File>,
    /// Length of each shard: the last page block of a file is short, and an `O_DIRECT` read there must not ask for the padding.
    lens: Vec<u64>,
    reads: AtomicU64,
    bytes_read: AtomicU64,
}

impl PackExpertSource {
    /// Open every shard of `table`. Opened read-only; `O_DIRECT` is attempted per shard and a shard that refuses it is read buffered.
    pub fn open(table: ExpertTable) -> io::Result<Self> {
        let mut buffered = Vec::with_capacity(table.shards().len());
        let mut direct = Vec::with_capacity(table.shards().len());
        let mut lens = Vec::with_capacity(table.shards().len());
        for p in table.shards() {
            let b = File::open(p)?;
            lens.push(b.metadata()?.len());
            let d = open_direct(p).ok();
            buffered.push(b);
            direct.push(d);
        }
        Ok(Self { table, direct, buffered, lens, reads: AtomicU64::new(0), bytes_read: AtomicU64::new(0) })
    }

    /// Experts read.
    pub fn reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }

    /// Bytes asked of the filesystem, windows included: what a fill actually costs.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    pub fn table(&self) -> &ExpertTable {
        &self.table
    }

    /// One part, through a page-aligned window. `dst` is the record; the part lands at `span.dst`.
    fn read_part(&self, span: &ExpertSpan, dst: &mut [u8]) -> io::Result<()> {
        let (off, len) = (span.offset, span.len as usize);
        let d0 = span.dst as usize;
        self.bytes_read.fetch_add(span.len, Ordering::Relaxed);
        let direct = self.direct.get(span.file as usize).and_then(|d| d.as_ref());
        let Some(f) = direct else {
            read_exact_at(&self.buffered[span.file as usize], &mut dst[d0..d0 + len], off)?;
            return Ok(());
        };
        // the window: whole pages around the part, so both the file offset and the buffer are aligned
        let a0 = off & !(PAGE as u64 - 1);
        let end = off + span.len;
        let a1 = (end.div_ceil(PAGE as u64) * PAGE as u64).min(self.lens[span.file as usize].div_ceil(PAGE as u64) * PAGE as u64);
        let n = (a1 - a0) as usize;
        let rel = (off - a0) as usize;
        let need = (end - a0) as usize;
        let out = if rel == 0 && d0 % PAGE == 0 && len % PAGE == 0 && n == len {
            // the part is its own window: read straight into the record
            self.bytes_read.fetch_add((n - len) as u64, Ordering::Relaxed);
            read_need_at(f, &mut dst[d0..d0 + len], off, need)
        } else {
            BOUNCE.with(|b| {
                let mut b = b.borrow_mut();
                let buf = b.get_or_insert_with(|| AlignedBuf::new(n.max(PAGE)));
                if buf.cap < n {
                    *buf = AlignedBuf::new(n);
                }
                self.bytes_read.fetch_add(n as u64, Ordering::Relaxed);
                let s = buf.slice();
                read_need_at(f, &mut s[..n], a0, need)?;
                dst[d0..d0 + len].copy_from_slice(&s[rel..rel + len]);
                Ok(())
            })
        };
        match out {
            // a filesystem that refuses `O_DIRECT` reads (EINVAL, e.g. some overlay and network filesystems): read this part buffered
            Err(e) if e.kind() == io::ErrorKind::InvalidInput || e.raw_os_error() == Some(libc::EINVAL) => {
                read_exact_at(&self.buffered[span.file as usize], &mut dst[d0..d0 + len], off)
            }
            other => other,
        }
    }
}

impl ExpertSource for PackExpertSource {
    fn read_expert(&self, layer: u16, expert: u16, dst: &mut [u8]) -> io::Result<()> {
        assert_eq!(dst.len(), self.table.record_bytes, "a fill copies exactly one record");
        let spans = self.table.spans(layer, expert);
        if spans.is_empty() {
            return Err(io::Error::new(io::ErrorKind::NotFound, format!("expert ({layer}, {expert}) is not in the pack")));
        }
        for s in spans {
            if let Err(e) = self.read_part(s, dst) {
                return Err(io::Error::new(e.kind(), format!("expert ({layer}, {expert}) part at {}: {e}", s.offset)));
            }
        }
        // the padding past the last part is part of the record and must be deterministic, not whatever the staging buffer held
        let used = self.table.used_bytes(layer, expert) as usize;
        dst[used..].fill(0);
        self.reads.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn expert_bytes(&self) -> usize {
        self.table.record_bytes
    }
}

/// The device side of a slot: where an expert's bytes land.
pub trait Slab: Send + Sync + 'static {
    /// Copy one expert into `slot`. `src` is exactly one expert long.
    fn fill(&self, slot: u32, src: &[u8]) -> Result<(), String>;
    /// Copy one expert back out, for verification.
    fn read_back(&self, slot: u32, dst: &mut [u8]) -> Result<(), String>;
    /// Bytes allocated (constant for the life of the slab).
    fn bytes(&self) -> usize;
    /// The device address of `slot`, for a caller that hands it to a kernel.
    fn slot_ptr(&self, slot: u32) -> *mut u8;
    /// A [`DevBuf`] view of `bytes` at `offset` inside `slot`, so a kernel type can be constructed over a filled slot.
    ///
    /// A slab that cannot address its own memory (the host fakes) leaves the default: only the CUDA slab is ever read by a kernel.
    fn slot_view(&self, _slot: u32, _offset: usize, _bytes: usize) -> Result<crate::rt::DevBuf, String> {
        Err("this slab exposes no addressable views".into())
    }
}

/// A slab in one `cudaMalloc` of `slots * expert_bytes` (unified memory on a GB10), or, in a build without a CUDA runtime,
/// nothing at all: [`DeviceSlab::new`] returns [`CacheError::NoDevice`] and the host-side tests run against their own fake.
///
/// A slab either owns its allocation ([`DeviceSlab::new`]) or is a view of one somebody else allocated
/// ([`DeviceSlab::over`]). The second form is what a boot uses: the load layout's `Streamed` region *is* the window, so the
/// slab is built over those bytes rather than `cudaMalloc`ing a second copy of them. Either way the slab holds exactly one
/// `DevBuf` and never more than `slots * expert_bytes` of it.
pub struct DeviceSlab {
    /// The one allocation, kept so a slot can be handed to a kernel type as a [`crate::rt::DevBuf`] view.
    #[cfg_attr(gb_no_cuda, allow(dead_code))]
    buf: crate::rt::DevBuf,
    /// One non-blocking stream for the host-to-device copies. A copy is a `memcpy` on unified memory, so serialising the copies
    /// (not the reads) costs nothing and keeps the slab free of per-slot state.
    #[cfg(not(gb_no_cuda))]
    stream: Mutex<crate::rt::Stream>,
    expert_bytes: usize,
    slots: usize,
}

// SAFETY: every access to `buf` goes through a raw device address handed to the driver, which is thread-safe; `stream` is behind a
// mutex. Nothing in the slab aliases host memory.
unsafe impl Sync for DeviceSlab {}

impl DeviceSlab {
    /// Allocate the slab: one `cudaMalloc` of `slots * expert_bytes`.
    pub fn new(slots: usize, expert_bytes: usize) -> Result<Self, CacheError> {
        #[cfg(not(gb_no_cuda))]
        {
            // cudaErrorNoDevice (100) is a missing device, not an I/O failure: the `not run: no device`
            // self-skips key off `CacheError::NoDevice`.
            let block = Arc::new(crate::rt::Block::alloc(slots * expert_bytes).map_err(|e| {
                if e.0 == 100 {
                    CacheError::NoDevice
                } else {
                    CacheError::Io(e.to_string())
                }
            })?);
            let buf = crate::rt::DevBuf::view(&block, 0, slots * expert_bytes);
            Self::over(buf, slots, expert_bytes)
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (slots, expert_bytes);
            Err(CacheError::NoDevice)
        }
    }

    /// Build the slab over bytes that are **already allocated** — the load layout's `Streamed` region — so that a boot
    /// allocates the streaming window once instead of twice.
    ///
    /// `buf` must hold at least `slots * expert_bytes`; the slab addresses the first that many. It keeps `buf` alive (a
    /// [`crate::rt::DevBuf`] view holds its block), so the region outlives the cache.
    pub fn over(buf: crate::rt::DevBuf, slots: usize, expert_bytes: usize) -> Result<Self, CacheError> {
        let want = slots.checked_mul(expert_bytes).ok_or_else(|| CacheError::Io("slab size overflows".into()))?;
        if slots == 0 || expert_bytes == 0 || buf.bytes() < want {
            return Err(CacheError::Io(format!(
                "a {slots}-slot slab of {expert_bytes} B records needs {want} B; the buffer holds {} B",
                buf.bytes()
            )));
        }
        #[cfg(not(gb_no_cuda))]
        {
            let stream = crate::rt::Stream::new_nonblocking().map_err(|e| CacheError::Io(e.to_string()))?;
            Ok(Self { buf, stream: Mutex::new(stream), expert_bytes, slots })
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = buf;
            Err(CacheError::NoDevice)
        }
    }
}

impl Slab for DeviceSlab {
    fn fill(&self, slot: u32, src: &[u8]) -> Result<(), String> {
        assert!((slot as usize) < self.slots, "slot past the end of the slab");
        assert_eq!(src.len(), self.expert_bytes, "a fill copies exactly one expert");
        #[cfg(not(gb_no_cuda))]
        {
            let stream = self.stream.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: `dst` is inside the slab allocation (checked above), `src` is one expert long and outlives the sync
            unsafe {
                stream
                    .memcpy_h2d_async(self.buf.ptr_at(slot as usize * self.expert_bytes), src.as_ptr(), src.len())
                    .map_err(|e| e.to_string())?;
            }
            stream.sync().map_err(|e| e.to_string())
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (slot, src);
            Err(CacheError::NoDevice.to_string())
        }
    }
    fn read_back(&self, slot: u32, dst: &mut [u8]) -> Result<(), String> {
        assert!((slot as usize) < self.slots, "slot past the end of the slab");
        #[cfg(not(gb_no_cuda))]
        {
            self.buf.download_range(slot as usize * self.expert_bytes, dst).map_err(|e| e.to_string())
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (slot, dst);
            Err(CacheError::NoDevice.to_string())
        }
    }
    fn bytes(&self) -> usize {
        self.slots * self.expert_bytes
    }
    fn slot_ptr(&self, slot: u32) -> *mut u8 {
        assert!((slot as usize) < self.slots, "slot past the end of the slab");
        #[cfg(not(gb_no_cuda))]
        {
            self.buf.ptr_at(slot as usize * self.expert_bytes).cast()
        }
        #[cfg(gb_no_cuda)]
        {
            core::ptr::null_mut()
        }
    }
    fn slot_view(&self, slot: u32, offset: usize, bytes: usize) -> Result<crate::rt::DevBuf, String> {
        assert!((slot as usize) < self.slots, "slot past the end of the slab");
        assert!(offset + bytes <= self.expert_bytes, "a view of a slot must stay inside it");
        #[cfg(not(gb_no_cuda))]
        {
            Ok(self.buf.subview(slot as usize * self.expert_bytes + offset, bytes))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (offset, bytes);
            Err(CacheError::NoDevice.to_string())
        }
    }
}

/// A lease on a resident slot: the slot holds this expert and will not be evicted until the lease drops. This is the pin the
/// forward reads through, so "eviction never returns a slot an outstanding forward still references" is a type-level fact
/// rather than a convention.
pub struct Lease<'a> {
    cache: &'a ExpertCache,
    slot: u32,
    layer: u16,
    expert: u16,
    released: bool,
}

impl std::fmt::Debug for Lease<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lease").field("slot", &self.slot).field("layer", &self.layer).field("expert", &self.expert).finish()
    }
}

impl Lease<'_> {
    /// The slot holding the expert.
    pub fn slot(&self) -> u32 {
        self.slot
    }
    /// The device address of the expert's first byte.
    pub fn ptr(&self) -> *mut u8 {
        self.cache.slab.slot_ptr(self.slot)
    }
    pub fn layer(&self) -> u16 {
        self.layer
    }
    pub fn expert(&self) -> u16 {
        self.expert
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        {
            let mut inner = self.cache.lock();
            if let Some(s) = inner.slots.get_mut(self.slot as usize) {
                debug_assert!(s.refs > 0, "lease dropped on an unpinned slot");
                s.refs = s.refs.saturating_sub(1);
            }
        }
        self.cache.cv.notify_all();
    }
}

/// What the cache did, which is the number the serve is judged on.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CacheStats {
    /// Touches served from a resident slot.
    pub hits: u64,
    /// Touches that had to read the pack.
    pub misses: u64,
    /// Bytes read from the pack. Without prefetching this is `misses * expert_bytes`; a prefetch that pays off reads bytes no
    /// miss is charged for, so it can be lower.
    pub bytes_read: u64,
    /// Slots reclaimed for a fill.
    pub evictions: u64,
    /// Prefetches queued.
    pub prefetches: u64,
    /// Prefetches that found the expert already resident (a router prediction that paid off).
    pub prefetch_hits: u64,
    /// Prefetches dropped because the queue was full. Not an error: the touch will miss and fill synchronously.
    pub prefetch_dropped: u64,
    /// Highest queue depth seen.
    pub queue_high_water: usize,
    /// Highest number of simultaneous fills seen.
    pub inflight_high_water: usize,
    /// Slots resident now.
    pub resident: usize,
    /// Slots the budget buys.
    pub capacity: usize,
}

impl CacheStats {
    /// Hits over touches; 0.0 with no touches.
    pub fn hit_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 {
            0.0
        } else {
            self.hits as f64 / n as f64
        }
    }
    /// Misses over touches; 0.0 with no touches.
    pub fn miss_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 {
            0.0
        } else {
            self.misses as f64 / n as f64
        }
    }
    /// Bytes a decode token reads at this miss rate: `touches_per_token * miss_rate * expert_bytes`.
    pub fn bytes_per_token(&self, touches_per_token: usize, expert_bytes: usize) -> f64 {
        touches_per_token as f64 * self.miss_rate() * expert_bytes as f64
    }
    /// Tokens per second the I/O path allows at `bytes_per_second`, if the fills overlap attention and the shared expert and
    /// nothing else is a bottleneck. Infinite when nothing misses.
    pub fn tokens_per_s_ceiling(&self, bytes_per_second: f64, touches_per_token: usize, expert_bytes: usize) -> f64 {
        let per_token = self.bytes_per_token(touches_per_token, expert_bytes);
        if per_token <= 0.0 {
            f64::INFINITY
        } else {
            bytes_per_second / per_token
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Free,
    /// A fill thread is reading this slot's expert; the bytes are not valid yet.
    Filling,
    Resident,
}

struct Slot {
    state: State,
    layer: u16,
    expert: u16,
    /// Ageing frequency counter. `u16` caps at 65535; halving every `aging_period` accesses keeps it far below that.
    freq: u16,
    last_touch: u64,
    /// Outstanding leases. A slot with `refs > 0` is never evicted.
    refs: u32,
}

struct Inner {
    slots: Vec<Slot>,
    /// `key -> slot` for resident experts.
    resident: HashMap<u32, u32>,
    /// Keys being read right now, so a second `get` waits instead of issuing a second read.
    inflight: HashMap<u32, ()>,
    free: Vec<u32>,
    queue: VecDeque<u32>,
    layer_resident: Vec<u32>,
    tick: u64,
    permits_free: usize,
    stats: CacheStats,
    shutdown: bool,
}

impl Inner {
    fn key(&self, layer: u16, expert: u16, experts_per_layer: usize) -> u32 {
        layer as u32 * experts_per_layer as u32 + expert as u32
    }
}

/// The cache. Construct with [`ExpertCache::new`], which starts the fill threads; the slab and the staging pool are allocated
/// once and never grow.
pub struct ExpertCache {
    cfg: Config,
    source: Arc<dyn ExpertSource>,
    slab: Arc<dyn Slab>,
    inner: Mutex<Inner>,
    cv: Condvar,
    staging: Mutex<Vec<AlignedBuf>>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl ExpertCache {
    /// Build the cache: `cfg.slots()` slots in one slab allocation, `cfg.max_inflight + 1` staging buffers, `cfg.workers`
    /// fill threads. No other memory grows with the trace.
    pub fn new(cfg: Config, source: Arc<dyn ExpertSource>, slab: Arc<dyn Slab>) -> Result<Arc<Self>, CacheError> {
        let n = cfg.slots();
        if n == 0 {
            return Err(CacheError::Io(format!(
                "budget {} B is smaller than one expert of {} B",
                cfg.budget_bytes, cfg.expert_bytes
            )));
        }
        assert_eq!(slab.bytes(), n * cfg.expert_bytes, "the slab must be exactly the configured slots");
        let cache = Arc::new(Self {
            inner: Mutex::new(Inner {
                slots: (0..n as u32)
                    .map(|_| Slot { state: State::Free, layer: 0, expert: 0, freq: 0, last_touch: 0, refs: 0 })
                    .collect(),
                resident: HashMap::with_capacity(n * 2),
                inflight: HashMap::with_capacity(cfg.max_inflight * 2),
                free: (0..n as u32).rev().collect(),
                queue: VecDeque::with_capacity(cfg.queue_cap),
                layer_resident: vec![0; cfg.layers],
                tick: 0,
                permits_free: cfg.max_inflight,
                stats: CacheStats { capacity: n, ..CacheStats::default() },
                shutdown: false,
            }),
            cv: Condvar::new(),
            staging: Mutex::new(Vec::with_capacity(cfg.max_inflight + 1)),
            workers: Mutex::new(Vec::new()),
            cfg,
            source,
            slab,
        });
        let mut handles = cache.workers.lock().expect("workers lock");
        for i in 0..cache.cfg.workers {
            let me = Arc::clone(&cache);
            handles.push(
                std::thread::Builder::new()
                    .name(format!("expert-fill-{i}"))
                    .spawn(move || me.worker())
                    .map_err(|e| CacheError::Io(e.to_string()))?,
            );
        }
        drop(handles);
        Ok(cache)
    }

    /// The configuration in force.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A snapshot of the counters.
    pub fn stats(&self) -> CacheStats {
        let inner = self.lock();
        let mut s = inner.stats;
        s.resident = inner.resident.len();
        s
    }

    /// Slots the budget buys.
    pub fn capacity(&self) -> usize {
        self.cfg.slots()
    }

    /// A [`DevBuf`] view of `bytes` at `offset` inside `slot`. The caller must hold a [`Lease`] on the slot: the view is only valid while
    /// the slot holds the expert the lease names.
    pub fn slot_view(&self, slot: u32, offset: usize, bytes: usize) -> Result<crate::rt::DevBuf, CacheError> {
        self.slab.slot_view(slot, offset, bytes).map_err(CacheError::Io)
    }

    fn acquire_staging(&self) -> AlignedBuf {
        let mut pool = self.staging.lock().unwrap_or_else(|e| e.into_inner());
        pool.pop().unwrap_or_else(|| AlignedBuf::new(self.cfg.expert_bytes + PAGE))
    }

    fn release_staging(&self, buf: AlignedBuf) {
        let mut pool = self.staging.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() < self.cfg.max_inflight + 1 {
            pool.push(buf);
        }
    }

    /// Pick a slot to reclaim: a free one, or the least valuable resident one.
    ///
    /// Pinned slots (`refs > 0`) are never candidates. Above its fair share a layer evicts from itself first, so one layer's
    /// churn cannot walk another layer's hot set out; below it, a layer may take a slot from anywhere, which lets a layer with
    /// a small working set lend its share to one that is thrashing.
    fn claim_slot(&self, inner: &mut Inner, layer: u16) -> Option<u32> {
        if let Some(s) = inner.free.pop() {
            return Some(s);
        }
        let quota = self.cfg.quota();
        let over_share = inner.layer_resident[layer as usize] >= quota as u32;
        let victim = self.pick_victim(inner, layer, over_share).or_else(|| {
            if over_share {
                self.pick_victim(inner, layer, false)
            } else {
                None
            }
        })?;
        // Unlink the victim.
        let key = inner.key(inner.slots[victim as usize].layer, inner.slots[victim as usize].expert, self.cfg.experts_per_layer);
        inner.resident.remove(&key);
        let l = inner.slots[victim as usize].layer as usize;
        inner.layer_resident[l] = inner.layer_resident[l].saturating_sub(1);
        inner.stats.evictions += 1;
        Some(victim)
    }

    fn pick_victim(&self, inner: &Inner, layer: u16, only_layer: bool) -> Option<u32> {
        let mut best: Option<(u32, u16, u64)> = None;
        for (i, s) in inner.slots.iter().enumerate() {
            if s.state != State::Resident || s.refs > 0 {
                continue;
            }
            if only_layer && s.layer != layer {
                continue;
            }
            let score = match self.cfg.policy {
                Policy::Lfu => (s.freq, s.last_touch),
                Policy::Lru => (0, s.last_touch),
            };
            if best.is_none_or(|(_, f, t)| score < (f, t)) {
                best = Some((i as u32, score.0, score.1));
            }
        }
        best.map(|(i, _, _)| i)
    }

    /// Read one expert into `slot`. The slot must already be claimed and marked [`State::Filling`]; on return it is resident,
    /// or free again if the read failed.
    fn fill_slot(&self, slot: u32, layer: u16, expert: u16) -> Result<(), CacheError> {
        let mut stage = self.acquire_staging();
        let read = {
            let buf = stage.slice();
            self.source.read_expert(layer, expert, &mut buf[..self.cfg.expert_bytes])
        };
        let written = match read {
            Ok(()) => self.slab.fill(slot, &stage.slice()[..self.cfg.expert_bytes]),
            Err(e) => Err(e.to_string()),
        };
        self.release_staging(stage);

        let mut inner = self.lock();
        if let Err(why) = written {
            inner.slots[slot as usize] = Slot { state: State::Free, layer: 0, expert: 0, freq: 0, last_touch: 0, refs: 0 };
            inner.free.push(slot);
            return Err(CacheError::Io(why));
        }
        inner.stats.bytes_read += self.cfg.expert_bytes as u64;
        inner.tick += 1;
        let t = inner.tick;
        let s = &mut inner.slots[slot as usize];
        s.state = State::Resident;
        s.layer = layer;
        s.expert = expert;
        s.freq = 1;
        s.last_touch = t;
        s.refs = 0;
        let key = inner.key(layer, expert, self.cfg.experts_per_layer);
        inner.resident.insert(key, slot);
        inner.layer_resident[layer as usize] += 1;
        Ok(())
    }

    /// Bring `(layer, expert)` in and pin it, waiting for whatever fill is already doing it. Returns a lease.
    fn fill_and_lease(&self, layer: u16, expert: u16) -> Result<Lease<'_>, CacheError> {
        let key = layer as u32 * self.cfg.experts_per_layer as u32 + expert as u32;
        let slot = loop {
            let mut inner = self.lock();
            if let Some(&s) = inner.resident.get(&key) {
                inner.slots[s as usize].refs += 1;
                break s;
            }
            if inner.inflight.contains_key(&key) {
                let inner = self.cv.wait(inner).unwrap_or_else(|e| e.into_inner());
                drop(inner);
                continue;
            }
            // A fill permit, or a shutdown. This is what `max_inflight` bounds: bytes in flight is `permits * expert_bytes`.
            while inner.permits_free == 0 {
                if inner.shutdown {
                    return Err(CacheError::Shutdown);
                }
                inner = self.cv.wait(inner).unwrap_or_else(|e| e.into_inner());
            }
            inner.permits_free -= 1;
            inner.stats.inflight_high_water = inner.stats.inflight_high_water.max(self.cfg.max_inflight - inner.permits_free);
            let Some(slot) = self.claim_slot(&mut inner, layer) else {
                inner.permits_free += 1;
                self.cv.notify_all();
                let leased = inner.slots.iter().filter(|s| s.refs > 0).count();
                return Err(CacheError::AllPinned { leased });
            };
            inner.slots[slot as usize] = Slot { state: State::Filling, layer, expert, freq: 0, last_touch: 0, refs: 0 };
            inner.inflight.insert(key, ());
            drop(inner);

            let out = self.fill_slot(slot, layer, expert);

            let mut inner = self.lock();
            inner.inflight.remove(&key);
            inner.permits_free += 1;
            self.cv.notify_all();
            out?;
            inner.slots[slot as usize].refs += 1;
            let s = &mut inner.slots[slot as usize];
            s.freq = s.freq.saturating_add(1);
            break slot;
        };
        Ok(Lease { cache: self, slot, layer, expert, released: false })
    }

    /// The expert's resident slot, evicting and filling one if it is not there. The returned lease pins the slot.
    pub fn get(&self, layer: u16, expert: u16) -> Result<Lease<'_>, CacheError> {
        self.check(layer, expert);
        let key = layer as u32 * self.cfg.experts_per_layer as u32 + expert as u32;
        {
            let mut inner = self.lock();
            inner.tick += 1;
            let t = inner.tick;
            if let Some(&s) = inner.resident.get(&key) {
                inner.stats.hits += 1;
                let period = self.cfg.aging_period.max(1);
                if t % period == 0 {
                    for s in inner.slots.iter_mut() {
                        if s.state == State::Resident {
                            s.freq >>= 1;
                        }
                    }
                }
                let slot = &mut inner.slots[s as usize];
                slot.freq = slot.freq.saturating_add(1);
                slot.last_touch = t;
                slot.refs += 1;
                return Ok(Lease { cache: self, slot: s, layer, expert, released: false });
            }
            inner.stats.misses += 1;
        }
        self.fill_and_lease(layer, expert)
    }

    /// Like [`Self::get`] but never blocks and never reads: `Some(lease)` if the expert is resident, `None` otherwise (counted as
    /// a miss, since the caller will have to fill it).
    pub fn try_get(&self, layer: u16, expert: u16) -> Result<Option<Lease<'_>>, CacheError> {
        self.check(layer, expert);
        let key = layer as u32 * self.cfg.experts_per_layer as u32 + expert as u32;
        let mut inner = self.lock();
        inner.tick += 1;
        let t = inner.tick;
        match inner.resident.get(&key) {
            Some(&s) => {
                inner.stats.hits += 1;
                let slot = &mut inner.slots[s as usize];
                slot.freq = slot.freq.saturating_add(1);
                slot.last_touch = t;
                slot.refs += 1;
                Ok(Some(Lease { cache: self, slot: s, layer, expert, released: false }))
            }
            None => {
                inner.stats.misses += 1;
                Ok(None)
            }
        }
    }

    /// Queue the fills a router selected. Never blocks and never reads: past `queue_cap` it drops the request and counts it, so a
    /// decode step issues its 240 touches and moves on to the shared expert and attention while the reads land.
    pub fn prefetch(&self, layer: u16, expert: u16) -> Result<(), CacheError> {
        self.check(layer, expert);
        let key = layer as u32 * self.cfg.experts_per_layer as u32 + expert as u32;
        let mut inner = self.lock();
        if inner.resident.contains_key(&key) {
            inner.stats.prefetch_hits += 1;
            return Ok(());
        }
        if inner.inflight.contains_key(&key) || inner.queue.iter().any(|&k| k == key) {
            return Ok(());
        }
        if inner.queue.len() >= self.cfg.queue_cap {
            inner.stats.prefetch_dropped += 1;
            return Ok(());
        }
        inner.queue.push_back(key);
        inner.stats.prefetches += 1;
        inner.stats.queue_high_water = inner.stats.queue_high_water.max(inner.queue.len());
        self.cv.notify_all();
        Ok(())
    }

    /// Queue every expert of `layer` — a prefill chunk touches essentially all of them, and issuing the sweep in expert order
    /// turns 384 random reads into one sequential pass over 2.96 GB.
    pub fn prefetch_layer(&self, layer: u16) -> Result<(), CacheError> {
        for e in 0..self.cfg.experts_per_layer as u16 {
            self.prefetch(layer, e)?;
        }
        Ok(())
    }

    fn check(&self, layer: u16, expert: u16) {
        assert!(
            (layer as usize) < self.cfg.layers && (expert as usize) < self.cfg.experts_per_layer,
            "expert ({layer}, {expert}) is outside a {}-layer, {}-expert bank",
            self.cfg.layers,
            self.cfg.experts_per_layer
        );
    }

    fn worker(&self) {
        loop {
            let key = {
                let mut inner = self.lock();
                loop {
                    if inner.shutdown {
                        return;
                    }
                    if let Some(k) = inner.queue.pop_front() {
                        break Some(k);
                    }
                    inner = self.cv.wait(inner).unwrap_or_else(|e| e.into_inner());
                }
            };
            let Some(key) = key else { return };
            let layer = (key / self.cfg.experts_per_layer as u32) as u16;
            let expert = (key % self.cfg.experts_per_layer as u32) as u16;
            // A prefetch that cannot be served (all slots leased, short read) is not fatal: the touch that needs it will take
            // the synchronous path.
            let _ = self.fill_and_lease(layer, expert).map(drop);
        }
    }

    /// Stop the fill threads and wait for them. Idempotent.
    pub fn shutdown(&self) {
        {
            let mut inner = self.lock();
            inner.shutdown = true;
            inner.queue.clear();
        }
        self.cv.notify_all();
        let mut handles = self.workers.lock().unwrap_or_else(|e| e.into_inner());
        for h in handles.drain(..) {
            let _ = h.join();
        }
    }
}

impl Drop for ExpertCache {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn slot_count_never_exceeds_the_budget() {
        let cfg = Config::default();
        assert_eq!(cfg.expert_bytes, EXPERT_BYTES);
        assert_eq!(EXPERT_BYTES % PAGE, 0);
        assert_eq!(EXPERT_BYTES, 7_720_960, "7.72 MB (7_720_000 B) rounded up to a 4 KiB page");
        // 13.6 GB / 7.72 MB = 1761.5 -> 1761, layer-aligned to 1760.
        assert_eq!(cfg.slots(), 1760);
        assert!(cfg.slab_bytes() <= cfg.budget_bytes);
        assert_eq!(cfg.quota(), 44);
    }

    #[test]
    fn small_budgets_do_not_round_to_zero() {
        let cfg = Config { budget_bytes: 3 * PAGE, expert_bytes: PAGE, layers: 40, experts_per_layer: 8, ..Config::default() };
        assert_eq!(cfg.slots(), 3);
        let cfg = Config { budget_bytes: PAGE / 2, expert_bytes: PAGE, ..Config::default() };
        assert_eq!(cfg.slots(), 0);
    }
}
