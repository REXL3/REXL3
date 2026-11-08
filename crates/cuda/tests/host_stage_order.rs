//! [`HostStage`] against the race it removes: a step's inputs written with `DevBuf::upload` (a pageable `cudaMemcpy` on the
//! legacy default stream, which can return before its DMA lands) and read at once by a kernel on a non-blocking stream,
//! which does not wait for the legacy stream. The kernel is the embedding gather, the first kernel of a model step: it
//! reads the uploaded token ids. Each step's output is compared bit for bit with the gather on the host.
//!
//! Needs a CUDA device (says `not run` without one). `REXL3_STAGE_ORDER_ITERS` (2000 steps an arm),
//! `REXL3_STAGE_ORDER_LOAD_MIB` (0: off; N: a thread makes pageable copies of N MiB both ways on the legacy default stream
//! while the arms run, which also stalls this test's own read-backs, so it is slow). Run it under other GPU load as well
//! (another process making copies) to widen the window.
use core::ffi::c_void;
use gb_cuda::glue;
use gb_cuda::rt::HostStage;
use gb_cuda::{device_available, DevBuf, Stream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const VOCAB: usize = 4096;
const DIM: usize = 64;
const ROWS: usize = 64;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// A bf16 table whose rows differ (finite values only).
fn table() -> Vec<u16> {
    (0..VOCAB * DIM).map(|i| 0x3f80 ^ (((i / DIM) * 31 + (i % DIM)) & 0x3ff) as u16).collect()
}

/// The ids of step `i`: every row differs from the step before.
fn ids(i: usize) -> Vec<i32> {
    (0..ROWS).map(|j| ((i * 977 + j * 131 + 1) % VOCAB) as i32).collect()
}

fn gather(table: &[u16], ids: &[i32]) -> Vec<u16> {
    ids.iter().flat_map(|&id| table[id as usize * DIM..(id as usize + 1) * DIM].iter().copied()).collect()
}

/// Pageable copies both ways on the legacy default stream until dropped.
struct CopyLoad {
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl CopyLoad {
    fn start(mib: usize) -> Option<Self> {
        if mib == 0 {
            return None;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let quit = Arc::clone(&stop);
        let worker = std::thread::spawn(move || {
            let host = vec![0x5au8; mib << 20];
            let mut back = vec![0u8; mib << 20];
            let mut dev = DevBuf::alloc(mib << 20).expect("load buffer");
            while !quit.load(Ordering::Relaxed) {
                dev.upload(&host).expect("load copy to the device");
                dev.download(&mut back).expect("load copy to the host");
            }
        });
        Some(Self { stop, worker: Some(worker) })
    }
}

impl Drop for CopyLoad {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// `iters` steps of: write the step's ids, gather on the non-blocking stream, wait, read back. Returns (steps whose output
/// was wrong, of those the ones that equal the previous step's gather).
fn run(staged: bool, iters: usize, table: &[u16], dt: &DevBuf, st: &Stream) -> (usize, usize) {
    let mut d_ids = DevBuf::zeroed(ROWS * 4).expect("ids");
    let out = DevBuf::zeroed(ROWS * DIM * 2).expect("out");
    gb_cuda::rt::device_sync().expect("settle the buffers");
    let mut stage = HostStage::new();
    let (mut bad, mut stale) = (0usize, 0usize);
    let mut got = vec![0u16; ROWS * DIM];
    let mut prev = gather(table, &ids(usize::MAX / 2));
    for i in 0..iters {
        let v = ids(i);
        if staged {
            stage.upload(&mut d_ids, &v, st).expect("staged upload");
        } else {
            d_ids.upload(&v).expect("legacy upload");
        }
        // SAFETY: bf16 table `[VOCAB, DIM]`, i32 ids `[ROWS]`, bf16 out `[ROWS, DIM]`
        unsafe {
            glue::embed(dt.ptr() as *const c_void, d_ids.ptr() as *const c_void, out.ptr(), ROWS, DIM, VOCAB, st).expect("embed")
        };
        stage.sync(st).expect("sync");
        out.download(&mut got).expect("read back");
        let want = gather(table, &v);
        if got != want {
            bad += 1;
            stale += usize::from(got == prev);
        }
        prev = want;
    }
    (bad, stale)
}

#[test]
fn a_staged_upload_is_what_the_next_kernel_on_a_nonblocking_stream_reads() {
    if !device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    let iters = env_usize("REXL3_STAGE_ORDER_ITERS", 2000);
    let load_mib = env_usize("REXL3_STAGE_ORDER_LOAD_MIB", 0);
    let table = table();
    let dt = DevBuf::from_slice(&table).expect("table");
    let st = Stream::new_nonblocking().expect("stream");
    gb_cuda::rt::device_sync().expect("settle the table");
    let _load = CopyLoad::start(load_mib);
    let (sb, ss) = run(true, iters, &table, &dt, &st);
    let (lb, ls) = run(false, iters, &table, &dt, &st);
    eprintln!("STAGE ORDER staged: {sb}/{iters} steps read wrong ids ({ss} the previous step's); load {load_mib} MiB");
    eprintln!("STAGE ORDER legacy: {lb}/{iters} steps read wrong ids ({ls} the previous step's); load {load_mib} MiB");
    // the legacy arm is the old code, reported only: a race need not fire in a given run
    assert_eq!(sb, 0, "a kernel launched after HostStage::upload read other bytes than the ones uploaded");
}
