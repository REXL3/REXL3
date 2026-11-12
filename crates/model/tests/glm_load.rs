//! Measurements for loading the GLM-5.3-Flash pack onto the GB10. Everything here is `#[ignore]`d: it needs REXL3_GLM_DIR (the pack) and a
//! GPU, and it must run under the GPU lock for its whole duration (`nrun NAME --gpu -- ...`). Each test refuses to start when
//! `MemAvailable` is low and aborts the process when it falls under 5 GiB, because the GB10's CPU and GPU share one pool of memory and
//! this box has no hardware watchdog.
//!
//! * `scan_cost`: the header scan with and without reading the 37,420 codebook markers, cold and warm (CPU only).
//! * `alloc_strategies`: per-tensor allocation, per-group allocation, one arena per layer and one arena for several layers, with the
//!   real expert group sizes and the real bytes (read once into host memory, so the disk is out of it).
//! * `full_load`: the plan executed in full by a stand-in for the generic pipeline (reader threads, pinned staging), with boot time,
//!   memory before and after, and a sampled byte-for-byte check of what landed on the device.
//!
//! `REXL3_GLM_DIR=... cargo test --release -p gb-model --test glm_load -- --ignored --nocapture <test>`

// Raw CUDA FFI measurement tests cannot link in the explicit CPU-only build profile.
#![cfg(not(rexl3_no_cuda))]

use gb_cuda::{DevBuf, Stream};
use gb_exl3::budget::Category;
use gb_model::families::glm::{GlmFamily, LoadPlan, RegionKind};
use std::ffi::c_void;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

extern "C" {
    fn cudaHostAlloc(p: *mut *mut c_void, bytes: usize, flags: u32) -> i32;
    fn cudaFreeHost(p: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32, stream: *mut c_void) -> i32;
}

#[cfg(target_os = "linux")]
extern "C" {
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}
/// macOS has no `posix_fadvise` (and no way to drop one file's pages): a no-op, so a "cold" read here may be warm.
#[cfg(not(target_os = "linux"))]
unsafe fn posix_fadvise(_fd: i32, _offset: i64, _len: i64, _advice: i32) -> i32 {
    0
}

const H2D: i32 = 1;
const D2H: i32 = 2;
const POSIX_FADV_DONTNEED: i32 = 4;
const GIB: u64 = 1 << 30;
const MIB: usize = 1 << 20;

fn pack_dir() -> PathBuf {
    std::env::var("REXL3_GLM_DIR").map(PathBuf::from).expect("set REXL3_GLM_DIR to the GLM pack")
}

// ------------------------------------------------------------------------------------------------------ memory and its guard

#[derive(Clone, Copy, Debug, Default)]
struct Mem {
    available: u64,
    free: u64,
    cached: u64,
}

fn meminfo() -> Mem {
    let t = std::fs::read_to_string("/proc/meminfo").unwrap();
    let get = |k: &str| -> u64 {
        t.lines()
            .find(|l| l.starts_with(k))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
            * 1024
    };
    Mem { available: get("MemAvailable:"), free: get("MemFree:"), cached: get("Cached:") }
}

fn status_kb(key: &str) -> u64 {
    let t = std::fs::read_to_string("/proc/self/status").unwrap();
    t.lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
        * 1024
}

fn gib(b: u64) -> f64 {
    b as f64 / GIB as f64
}

/// Aborts the whole process when `MemAvailable` falls under the floor, and remembers the lowest `MemAvailable` and `MemFree` it saw.
struct Guard {
    stop: Arc<AtomicBool>,
    min_available: Arc<AtomicU64>,
    min_free: Arc<AtomicU64>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Guard {
    fn start(floor: u64) -> Guard {
        let (stop, min_available, min_free) =
            (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(u64::MAX)), Arc::new(AtomicU64::new(u64::MAX)));
        let (s, a, f) = (stop.clone(), min_available.clone(), min_free.clone());
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let m = meminfo();
                a.fetch_min(m.available, Ordering::Relaxed);
                f.fetch_min(m.free, Ordering::Relaxed);
                if m.available < floor {
                    eprintln!(
                        "GUARD: MemAvailable {:.2} GiB is under the {:.1} GiB floor: aborting the process",
                        gib(m.available),
                        gib(floor)
                    );
                    std::process::abort();
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        });
        Guard { stop, min_available, min_free, thread: Some(thread) }
    }
    fn finish(mut self) -> (u64, u64) {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.take().unwrap().join().unwrap();
        (self.min_available.load(Ordering::Relaxed), self.min_free.load(Ordering::Relaxed))
    }
}

/// Refuse to start under `need` bytes of `MemAvailable`.
fn require_available(need: u64) {
    let m = meminfo();
    assert!(
        m.available >= need,
        "refusing to run: MemAvailable is {:.1} GiB, this test needs {:.0} GiB",
        gib(m.available),
        gib(need)
    );
}

fn cuda_free() -> u64 {
    gb_cuda::rt::mem_info().unwrap().0 as u64
}

fn ck(rc: i32, what: &str) {
    assert_eq!(rc, 0, "{what} failed with CUDA error {rc}");
}

fn evict(path: &Path) {
    use std::os::fd::AsRawFd;
    if let Ok(f) = File::open(path) {
        unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED) };
    }
}

fn evict_pack(dir: &Path) {
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        if e.path().extension().is_some_and(|x| x == "safetensors") {
            evict(&e.path());
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------- scan cost

#[test]
#[ignore]
fn scan_cost() {
    require_available(20 * GIB);
    let dir = pack_dir();
    let timed = |label: &str, markers: bool| {
        let t = Instant::now();
        let ck = gb_exl3::checkpoint::scan(&dir, markers).unwrap();
        eprintln!(
            "scan markers={markers:<5} {label:<5}: {:>8.2?}  ({} groups, {} plain)",
            t.elapsed(),
            ck.groups.len(),
            ck.plain.len()
        );
    };
    for _ in 0..2 {
        evict_pack(&dir);
        eprintln!("page cache {:.1} GiB after evicting the pack", gib(meminfo().cached));
        timed("cold", false);
        timed("warm", false);
        evict_pack(&dir);
        timed("cold", true);
        timed("warm", true);
    }
}

// -------------------------------------------------------------------------------------------------------- raw read throughput

/// How fast the pack can be read cold, with no GPU involved: a spread sample of the plan's 64 MiB extents (about a sixth of the
/// pack) read by 1..16 threads, the pack evicted from the page cache before each run, the pages dropped after each read. This is the
/// floor under any boot of the pack; CPU only, so it runs without the GPU lock.
#[test]
#[ignore]
fn read_throughput() {
    require_available(20 * GIB);
    let dir = pack_dir();
    let pack = GlmFamily::map_pack(&dir).unwrap();
    let plan = &pack.plan;
    let extents = plan.extents(16, 64 << 20);
    let sample: Vec<_> = extents.iter().step_by(6).collect();
    let total: u64 = sample.iter().map(|e| e.bytes).sum();
    let files: Vec<File> = plan.files.iter().map(|f| File::open(dir.join(f)).unwrap()).collect();
    let guard = Guard::start(5 * GIB);
    for threads in [1usize, 2, 4, 8, 16] {
        evict_pack(&dir);
        let next = AtomicUsize::new(0);
        let t = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut buf = vec![0u8; 64 * MIB + 64];
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(e) = sample.get(i) else { return };
                        let n = (e.bytes as usize).min(buf.len());
                        files[e.file as usize].read_exact_at(&mut buf[..n], e.offset).unwrap();
                        use std::os::fd::AsRawFd;
                        unsafe {
                            posix_fadvise(
                                files[e.file as usize].as_raw_fd(),
                                e.offset as i64,
                                e.bytes as i64,
                                POSIX_FADV_DONTNEED,
                            )
                        };
                    }
                });
            }
        });
        let s = t.elapsed().as_secs_f64();
        eprintln!(
            "cold read of {:.1} GiB in {} extents with {threads:>2} threads: {s:>6.2} s = {:.2} GB/s",
            gib(total),
            sample.len(),
            total as f64 / 1e9 / s
        );
    }
    let (min_avail, _) = guard.finish();
    eprintln!("lowest MemAvailable {:.1} GiB", gib(min_avail));
}

// ------------------------------------------------------------------------------------------------------ alloc strategies

/// Host copies of the bytes of every item placed in one region of the plan, in item order.
fn read_region(dir: &Path, plan: &LoadPlan, region: u32) -> Vec<(usize, Vec<u8>)> {
    let files: Vec<File> = plan.files.iter().map(|f| File::open(dir.join(f)).unwrap()).collect();
    plan.items
        .iter()
        .enumerate()
        .filter(|(_, i)| i.region == region)
        .map(|(idx, i)| {
            let mut buf = vec![0u8; i.bytes as usize];
            files[i.file as usize].read_exact_at(&mut buf, i.offset).unwrap();
            (idx, buf)
        })
        .collect()
}

struct Probe {
    label: String,
    secs: f64,
    allocs: usize,
    copies: usize,
    bytes: u64,
    free_delta: i64,
    avail_delta: i64,
    free_secs: f64,
}

impl Probe {
    fn show(&self) {
        let per_alloc = if self.allocs > 0 { self.secs * 1e6 / self.allocs as f64 } else { 0.0 };
        eprintln!(
            "  {:<44} {:>7.3} s  {:>6} allocs {:>6} copies  {:>6.2} GiB  {:>7.2} GB/s  cuda-free -{:>6.3} GiB  MemAvail -{:>6.3} GiB  free {:.3} s  ({:.0} us/alloc-and-copy)",
            self.label,
            self.secs,
            self.allocs,
            self.copies,
            gib(self.bytes),
            self.bytes as f64 / 1e9 / self.secs,
            self.free_delta as f64 / GIB as f64,
            self.avail_delta as f64 / GIB as f64,
            self.free_secs,
            per_alloc
        );
    }
}

/// Run `f` (which fills `keep` with device buffers), time it, read the memory counters around it, then free and time the free.
fn probe(label: &str, allocs: usize, copies: usize, bytes: u64, f: impl FnOnce(&mut Vec<DevBuf>)) -> Probe {
    let (free0, avail0) = (cuda_free(), meminfo().available);
    let mut keep: Vec<DevBuf> = Vec::new();
    let t = Instant::now();
    f(&mut keep);
    gb_cuda::rt::device_sync().unwrap();
    let secs = t.elapsed().as_secs_f64();
    let (free1, avail1) = (cuda_free(), meminfo().available);
    let t = Instant::now();
    drop(keep);
    let free_secs = t.elapsed().as_secs_f64();
    Probe {
        label: label.into(),
        secs,
        allocs,
        copies,
        bytes,
        free_delta: free0 as i64 - free1 as i64,
        avail_delta: avail0 as i64 - avail1 as i64,
        free_secs,
    }
}

fn h2d(dst: *mut c_void, src: &[u8]) {
    ck(unsafe { cudaMemcpy(dst, src.as_ptr() as *const c_void, src.len(), H2D) }, "cudaMemcpy H2D");
}

#[test]
#[ignore]
fn alloc_strategies() {
    require_available(60 * GIB);
    let guard = Guard::start(5 * GIB);
    let dir = pack_dir();
    let pack = GlmFamily::map_pack(&dir).unwrap();
    let plan = &pack.plan;
    assert!(gb_cuda::device_available());
    gb_cuda::rt::device_sync().unwrap();

    // ---- granularity and per-call cost of cudaMalloc alone (nothing is copied or touched)
    eprintln!(
        "== cudaMalloc alone: N allocations of S bytes (no copy); 5 repetitions, median time and the SMALLEST memory delta"
    );
    eprintln!("   (cuda-free and MemAvailable are one pool shared with every process on the box, so a delta carries other processes' noise: the minimum is the honest bound)");
    for (n, size) in
        [(2000usize, 4096usize), (2000, 12288), (1000, 2 * MIB), (1000, 2 * MIB + 12288), (1000, 3 * MIB), (50, 34 * MIB)]
    {
        let total = (n * size) as u64;
        let mut runs: Vec<Probe> = (0..5)
            .map(|_| {
                probe(&format!("{n} x {size} B"), n, 0, total, |keep| {
                    for _ in 0..n {
                        keep.push(DevBuf::alloc(size).unwrap());
                    }
                })
            })
            .collect();
        runs.sort_by(|a, b| a.secs.partial_cmp(&b.secs).unwrap());
        let (med, min_free, min_avail) =
            (&runs[2], runs.iter().map(|r| r.free_delta).min().unwrap(), runs.iter().map(|r| r.avail_delta).min().unwrap());
        eprintln!(
            "  {n:>5} x {size:>8} B = {:>6.3} GiB: {:>7.1} us/alloc ({:.1} GB/s), cuda-free -{:>6.3} GiB (min {:.3}x requested), MemAvail -{:>6.3} GiB (min), free {:.3} s",
            gib(total),
            med.secs * 1e6 / n as f64,
            total as f64 / 1e9 / med.secs,
            min_free as f64 / GIB as f64,
            min_free as f64 / total as f64,
            min_avail as f64 / GIB as f64,
            med.free_secs
        );
    }
    let p = probe("1 x 1.69 GiB", 1, 0, 1700 * MIB as u64, |keep| keep.push(DevBuf::alloc(1700 * MIB).unwrap()));
    eprintln!(
        "  one 1.66 GiB allocation: {:.1} ms, cuda-free -{:.3} GiB, MemAvail -{:.3} GiB (committed at cudaMalloc or at first touch?), free {:.1} ms",
        p.secs * 1e3,
        p.free_delta as f64 / GIB as f64,
        p.avail_delta as f64 / GIB as f64,
        p.free_secs * 1e3
    );

    // ---- the real expert groups: one 2-bit layer (6) and one 3-bit layer (3)
    for layer in [6usize, 3] {
        let region = plan.regions.iter().find(|r| r.kind == RegionKind::Experts(layer)).unwrap();
        let t = Instant::now();
        let host = read_region(&dir, plan, region.id);
        let host_bytes: u64 = host.iter().map(|(_, b)| b.len() as u64).sum();
        eprintln!(
            "\n== layer {layer}: {} items, {:.3} GiB read into host memory in {:.1?}",
            host.len(),
            gib(host_bytes),
            t.elapsed()
        );
        let groups = host.len() / 3;
        let reps = 6; // several layers' worth, to see whether cost grows with what is already allocated
                      // the data in item order is [suh, svh, trellis] per group (file order); the device layout of each variant is its own
        let by_group: Vec<[&Vec<u8>; 3]> = host.chunks(3).map(|c| [&c[2].1, &c[0].1, &c[1].1]).collect(); // trellis, suh, svh

        // 1. per tensor: three allocations and three uploads a group (what `Exl3Weights::from_loaded` does today)
        for round in 0..3 {
            eprintln!("  -- round {round}");
            let mut results = vec![];
            results.push(probe(
                &format!("per tensor (3 allocs/group), {groups} groups"),
                groups * 3,
                groups * 3,
                host_bytes,
                |keep| {
                    for g in &by_group {
                        for part in g {
                            let b = DevBuf::alloc(part.len()).unwrap();
                            h2d(b.ptr(), part);
                            keep.push(b);
                        }
                    }
                },
            ));
            // 2. per group: one allocation holding the group's three parts at 256-byte offsets
            results.push(probe(&format!("per group (1 alloc/group), {groups} groups"), groups, groups * 3, host_bytes, |keep| {
                for g in &by_group {
                    let offs: Vec<usize> = g
                        .iter()
                        .scan(0usize, |c, p| {
                            let o = *c;
                            *c = (o + p.len()).div_ceil(256) * 256;
                            Some(o)
                        })
                        .collect();
                    let total = offs[2] + g[2].len();
                    let b = DevBuf::alloc(total).unwrap();
                    for (o, part) in offs.iter().zip(g) {
                        h2d(b.ptr_at(*o), part);
                    }
                    keep.push(b);
                }
            }));
            // 3. the plan: one region for the layer, copies at the plan's offsets
            results.push(probe(&format!("layer arena (1 alloc/layer), {groups} groups"), 1, groups * 3, host_bytes, |keep| {
                let b = DevBuf::alloc(region.bytes as usize).unwrap();
                for (idx, bytes) in &host {
                    h2d(b.ptr_at(plan.items[*idx].dst as usize), bytes);
                }
                keep.push(b);
            }));
            // 4. the same layer `reps` times: `reps` arenas, then one arena `reps` times as large
            let b6 = host_bytes * reps as u64;
            results.push(probe(&format!("{reps} layer arenas ({reps} allocs)"), reps, groups * 3 * reps, b6, |keep| {
                for _ in 0..reps {
                    let b = DevBuf::alloc(region.bytes as usize).unwrap();
                    for (idx, bytes) in &host {
                        h2d(b.ptr_at(plan.items[*idx].dst as usize), bytes);
                    }
                    keep.push(b);
                }
            }));
            results.push(probe(&format!("one arena for {reps} layers (1 alloc)"), 1, groups * 3 * reps, b6, |keep| {
                let b = DevBuf::alloc(region.bytes as usize * reps).unwrap();
                for r in 0..reps {
                    for (idx, bytes) in &host {
                        h2d(b.ptr_at(region.bytes as usize * r + plan.items[*idx].dst as usize), bytes);
                    }
                }
                keep.push(b);
            }));
            for p in &results {
                p.show();
            }
        }
        // 5. the cost of the copies alone, into an arena already allocated: pageable vs pinned staging, small vs large calls
        let arena = DevBuf::alloc(region.bytes as usize).unwrap();
        let t = Instant::now();
        for (idx, bytes) in &host {
            h2d(arena.ptr_at(plan.items[*idx].dst as usize), bytes);
        }
        let s = t.elapsed().as_secs_f64();
        eprintln!(
            "  copies alone, pageable cudaMemcpy, {} calls: {:.3} s ({:.2} GB/s, {:.0} us/call)",
            host.len(),
            s,
            host_bytes as f64 / 1e9 / s,
            s * 1e6 / host.len() as f64
        );
        let mut pinned: *mut c_void = std::ptr::null_mut();
        ck(unsafe { cudaHostAlloc(&mut pinned, 64 * MIB, 0) }, "cudaHostAlloc");
        let st = Stream::new().unwrap();
        for (label, chunk) in [("2 MiB", 2 * MIB), ("16 MiB", 16 * MIB), ("64 MiB", 64 * MIB)] {
            // stage chunk bytes of real data (the layer's trellises are contiguous in the arena order), copy it over and over
            let src = unsafe { std::slice::from_raw_parts_mut(pinned as *mut u8, 64 * MIB) };
            for (i, b) in src.iter_mut().enumerate() {
                *b = (i % 251) as u8;
            }
            let calls = (region.bytes as usize / chunk).max(1);
            let t = Instant::now();
            for c in 0..calls {
                let off = (c * chunk) % (region.bytes as usize - chunk + 1);
                ck(unsafe { cudaMemcpyAsync(arena.ptr_at(off), pinned, chunk, H2D, st.raw()) }, "cudaMemcpyAsync");
                st.sync().unwrap();
            }
            let s = t.elapsed().as_secs_f64();
            eprintln!(
                "  copies alone, pinned staging {label:>6} x {calls:>4} calls: {:.3} s ({:.2} GB/s)",
                s,
                (calls * chunk) as f64 / 1e9 / s
            );
        }
        unsafe { cudaFreeHost(pinned) };
        drop(arena);
        drop(host);
    }
    let (min_avail, min_free) = guard.finish();
    eprintln!("\nlowest MemAvailable {:.1} GiB, lowest MemFree {:.1} GiB", gib(min_avail), gib(min_free));
}

// -------------------------------------------------------------------------------------------------------------- full load

#[derive(Default)]
struct LoadStats {
    alloc: Duration,
    load: Duration,
    read_bytes: u64,
    extents: usize,
    markers: u64,
}

/// Execute `plan` in full: allocate every region (the guard watches memory between allocations), then `threads` workers each take
/// extents in file order, read them into their own pinned staging buffer, copy the items out to their destinations on their own
/// stream, and drop the pages they read from the page cache.
fn execute(plan: &LoadPlan, dir: &Path, threads: usize, stage: usize, max_gap: u64) -> (Vec<DevBuf>, LoadStats) {
    let t = Instant::now();
    let mut regions: Vec<DevBuf> = Vec::with_capacity(plan.regions.len());
    for r in &plan.regions {
        assert_eq!(r.id as usize, regions.len(), "the full plan's region ids are 0..n");
        regions.push(DevBuf::alloc(r.bytes as usize).unwrap_or_else(|e| panic!("allocating region {} ({}): {e}", r.id, r.kind)));
    }
    let alloc = t.elapsed();
    let bases: Arc<Vec<usize>> = Arc::new(regions.iter().map(|b| b.ptr() as usize).collect());
    let files: Arc<Vec<File>> = Arc::new(plan.files.iter().map(|f| File::open(dir.join(f)).unwrap()).collect());
    let extents = plan.extents(max_gap, stage as u64);
    let next = Arc::new(AtomicUsize::new(0));
    let failure: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let read_bytes = Arc::new(AtomicU64::new(0));
    let checked = Arc::new(AtomicU64::new(0));
    let t = Instant::now();
    std::thread::scope(|s| {
        for w in 0..threads {
            let (bases, files, next, failure, read_bytes, extents, checked) =
                (&bases, &files, &next, &failure, &read_bytes, &extents, &checked);
            s.spawn(move || {
                let mut pinned: *mut c_void = std::ptr::null_mut();
                // an extent may exceed the cap by up to `max_gap` (it takes the marker that follows its last item)
                let capacity = stage + max_gap as usize;
                ck(unsafe { cudaHostAlloc(&mut pinned, capacity, 0) }, "cudaHostAlloc");
                let st = Stream::new().unwrap();
                let buf = unsafe { std::slice::from_raw_parts_mut(pinned as *mut u8, capacity) };
                let mut run = || -> Result<(), String> {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(e) = extents.get(i) else { return Ok(()) };
                        if failure.lock().unwrap().is_some() {
                            return Ok(());
                        }
                        let f = &files[e.file as usize];
                        let items = &plan.items[e.items.clone()];
                        if e.bytes as usize <= capacity {
                            f.read_exact_at(&mut buf[..e.bytes as usize], e.offset)
                                .map_err(|x| format!("{}: {x}", plan.files[e.file as usize]))?;
                            // the codebook markers the plan owes a check for, from the bytes just read
                            for m in plan.markers_in(e) {
                                let at = (m.offset - e.offset) as usize;
                                let got = u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]);
                                if got != m.want {
                                    return Err(format!("{} holds {got:#x}, expected {:#x}", m.name, m.want));
                                }
                                checked.fetch_add(1, Ordering::Relaxed);
                            }
                            for it in items {
                                let src = unsafe { pinned.add((it.offset - e.offset) as usize) };
                                let dst = (bases[it.region as usize] + it.dst as usize) as *mut c_void;
                                ck(unsafe { cudaMemcpyAsync(dst, src, it.bytes as usize, H2D, st.raw()) }, "cudaMemcpyAsync");
                            }
                            st.sync().map_err(|x| x.to_string())?;
                        } else {
                            // one item larger than the staging buffer: stream it through in pieces
                            assert_eq!(items.len(), 1);
                            assert!(plan.markers_in(e).is_empty(), "an oversized single item has no marker inside");
                            let it = &items[0];
                            let mut done = 0usize;
                            while done < it.bytes as usize {
                                let n = stage.min(it.bytes as usize - done);
                                f.read_exact_at(&mut buf[..n], it.offset + done as u64).map_err(|x| x.to_string())?;
                                let dst = (bases[it.region as usize] + it.dst as usize + done) as *mut c_void;
                                ck(unsafe { cudaMemcpyAsync(dst, pinned, n, H2D, st.raw()) }, "cudaMemcpyAsync");
                                st.sync().map_err(|x| x.to_string())?;
                                done += n;
                            }
                        }
                        read_bytes.fetch_add(e.bytes, Ordering::Relaxed);
                        {
                            use std::os::fd::AsRawFd;
                            unsafe { posix_fadvise(f.as_raw_fd(), e.offset as i64, e.bytes as i64, POSIX_FADV_DONTNEED) };
                        }
                    }
                };
                if let Err(m) = run() {
                    *failure.lock().unwrap() = Some(format!("worker {w}: {m}"));
                }
                unsafe { cudaFreeHost(pinned) };
            });
        }
    });
    if let Some(m) = failure.lock().unwrap().take() {
        panic!("load failed: {m}");
    }
    // markers no extent covers get a read of their own
    for m in plan.uncovered_markers(&extents) {
        let mut b = [0u8; 4];
        files[m.file as usize].read_exact_at(&mut b, m.offset).unwrap();
        assert_eq!(u32::from_le_bytes(b), m.want, "{}", m.name);
        checked.fetch_add(1, Ordering::Relaxed);
    }
    assert_eq!(checked.load(Ordering::Relaxed), plan.markers.len() as u64, "every marker of the plan was checked");
    gb_cuda::rt::device_sync().unwrap();
    let stats = LoadStats {
        alloc,
        load: t.elapsed(),
        read_bytes: read_bytes.load(Ordering::Relaxed),
        extents: extents.len(),
        markers: checked.load(Ordering::Relaxed),
    };
    (regions, stats)
}

/// Compare `sample` items of the device copy with the file, byte for byte.
fn verify(plan: &LoadPlan, dir: &Path, regions: &[DevBuf], sample: usize) -> usize {
    let files: Vec<File> = plan.files.iter().map(|f| File::open(dir.join(f)).unwrap()).collect();
    // first and last item of every region, then evenly spaced items
    let mut picks: Vec<usize> = Vec::new();
    for r in &plan.regions {
        let idx: Vec<usize> = plan.items.iter().enumerate().filter(|(_, i)| i.region == r.id).map(|(k, _)| k).collect();
        picks.push(*idx.first().unwrap());
        picks.push(*idx.last().unwrap());
    }
    let stride = (plan.items.len() / sample).max(1);
    picks.extend((0..plan.items.len()).step_by(stride));
    picks.sort_unstable();
    picks.dedup();
    for &k in &picks {
        let it = &plan.items[k];
        let mut want = vec![0u8; it.bytes as usize];
        files[it.file as usize].read_exact_at(&mut want, it.offset).unwrap();
        let mut got = vec![0u8; it.bytes as usize];
        ck(
            unsafe {
                cudaMemcpy(got.as_mut_ptr() as *mut c_void, regions[it.region as usize].ptr_at(it.dst as usize), got.len(), D2H)
            },
            "cudaMemcpy D2H",
        );
        assert!(got == want, "{} differs on the device from its bytes in the shard", it.name);
    }
    picks.len()
}

#[test]
#[ignore]
fn full_load() {
    // the pack is 91.6 GiB: with fewer than 100 GiB available there is no room to hold it and keep the 5 GiB floor
    require_available(100 * GIB);
    let dir = pack_dir();
    let threads: usize = std::env::var("REXL3_LOAD_WORKERS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let stage: usize = std::env::var("REXL3_LOAD_STAGE_MIB").ok().and_then(|v| v.parse().ok()).unwrap_or(64) * MIB;
    let text_only = std::env::var("REXL3_LOAD_TEXT_ONLY").is_ok();
    let t_boot = Instant::now();
    evict_pack(&dir);
    let before = meminfo();
    eprintln!(
        "before: MemAvailable {:.2} GiB, MemFree {:.2} GiB, page cache {:.2} GiB (pack evicted), cuda free {:.2} GiB, VmRSS {:.2} GiB",
        gib(before.available),
        gib(before.free),
        gib(before.cached),
        gib(cuda_free()),
        gib(status_kb("VmRSS:"))
    );
    let guard = Guard::start(5 * GIB);
    // boot step 1: scan and map (headers only)
    let t = Instant::now();
    let pack = GlmFamily::map_pack(&dir).unwrap();
    let map = t.elapsed();
    let plan = if text_only { pack.plan.select(|c| c != Category::Vision) } else { pack.plan.clone() };
    // the plan's regions are renumbered by `select`: execute needs ids 0..n, so a text-only run compacts them
    let plan = compact(plan);
    let total: u64 = plan.items.iter().map(|i| i.bytes).sum();
    eprintln!(
        "plan: {} items, {} regions, {:.2} GiB to copy; scan + map {:.2?}",
        plan.items.len(),
        plan.regions.len(),
        gib(total),
        map
    );
    // boot step 2: allocate and load
    let (regions, stats) = execute(&plan, &dir, threads, stage, 16);
    let boot = t_boot.elapsed();
    let after = meminfo();
    let (min_avail, min_free) = guard.finish();
    eprintln!(
        "loaded {:.2} GiB: allocate {} regions {:.2?}, copy {:.2?} ({:.2} GB/s) with {threads} workers x {} MiB staging, {} extents, {} markers checked in-stream; boot (evict + scan + map + alloc + load) {:.2?}",
        gib(stats.read_bytes),
        regions.len(),
        stats.alloc,
        stats.load,
        stats.read_bytes as f64 / 1e9 / stats.load.as_secs_f64(),
        stage / MIB,
        stats.extents,
        stats.markers,
        boot
    );
    eprintln!(
        "after:  MemAvailable {:.2} GiB (-{:.2}), MemFree {:.2} GiB (-{:.2}), page cache {:.2} GiB, cuda free {:.2} GiB, VmRSS {:.2} GiB (peak {:.2}), lowest MemAvailable during the run {:.2} GiB, lowest MemFree {:.2} GiB",
        gib(after.available),
        gib(before.available) - gib(after.available),
        gib(after.free),
        gib(before.free) - gib(after.free),
        gib(after.cached),
        gib(cuda_free()),
        gib(status_kb("VmRSS:")),
        gib(status_kb("VmHWM:")),
        gib(min_avail),
        gib(min_free)
    );
    let n = verify(&plan, &dir, &regions, 400);
    eprintln!("verified {n} items byte for byte against the shards");
    let t = Instant::now();
    drop(regions);
    eprintln!("freed all regions in {:.2?}; MemAvailable {:.2} GiB afterwards", t.elapsed(), gib(meminfo().available));
}

/// Renumber a selected plan's regions 0..n (the stand-in executor indexes regions by id).
fn compact(mut plan: LoadPlan) -> LoadPlan {
    let map: std::collections::BTreeMap<u32, u32> = plan.regions.iter().enumerate().map(|(i, r)| (r.id, i as u32)).collect();
    for (i, r) in plan.regions.iter_mut().enumerate() {
        r.id = i as u32;
    }
    for it in &mut plan.items {
        it.region = map[&it.region];
    }
    plan
}
