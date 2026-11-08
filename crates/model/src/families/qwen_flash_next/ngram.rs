//! Host n-gram sidecar reader and per-request token context.
//!
//! `NgramTable::open(dir, cfg)` mmaps only the n-gram safetensors file, validates its metadata,
//! and keeps packed rows on the host; `gather(history, 2, output)` returns interleaved u16 words
//! for the PLE device decoder. `row_ids` is pure and shared by the oracle test. `warm` streams
//! the whole file on a completion-tracked worker: readiness waits for its successful completion
//! (`finish_warm` / `warm_preflight`) before serving, so cold mapped-table faults never hit the
//! first decode rounds. The file's pages are reclaimable page cache, not an anonymous/device
//! reservation, so the memory plan never deducts the table twice.

use super::{config::FlashNextConfig, inventory};
use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    sync::{Arc, Mutex, OnceLock},
};

const ROW: usize = 160;
const RING: usize = 128;
/// Bytes faulted per read of the warm worker: small enough that progress and cancellation stay
/// responsive, large enough that a 36 GiB table streams with few syscalls.
const WARM_CHUNK: usize = 1 << 20;

/// Completion state of the n-gram warm worker. `done` flips only after every byte of the file is
/// read (or an error ended the read); [`WarmHandle::wait`] never returns before that.
pub struct WarmHandle {
    done: AtomicBool,
    result: Mutex<Option<Result<(), String>>>,
    read: AtomicU64,
}

impl WarmHandle {
    fn new() -> Self {
        Self { done: AtomicBool::new(false), result: Mutex::new(None), read: AtomicU64::new(0) }
    }
    /// A handle that already failed (test-only): the same completion state a real read error
    /// produces, so the gate's propagation can be checked without a GPU or a large file.
    #[cfg(test)]
    pub fn new_failed_for_test(error: String) -> Self {
        let h = Self::new();
        h.finish(Err(error));
        h
    }
    fn finish(&self, out: Result<(), String>) {
        *self.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(out);
        self.done.store(true, Ordering::Release);
    }
    /// The worker's outcome once it has finished; `None` while it still runs.
    pub fn done(&self) -> Option<Result<(), String>> {
        if self.done.load(Ordering::Acquire) {
            Some(self.result.lock().unwrap_or_else(|e| e.into_inner()).clone().expect("result is set with done"))
        } else {
            None
        }
    }
    /// Bytes read (faulted) so far, on the way to the whole file.
    pub fn bytes_read(&self) -> u64 {
        self.read.load(Ordering::Relaxed)
    }
    /// Block until the worker finishes and report its outcome. Parks briefly between polls: the
    /// caller is the boot/readiness path, which may wait for a multi-GiB read.
    pub fn wait(&self) -> Result<(), String> {
        while !self.done.load(Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        self.result.lock().unwrap_or_else(|e| e.into_inner()).clone().expect("result is set with done")
    }
}

/// Sequentially fault/read the whole file on the warm worker and record the outcome.
fn warm_read(path: PathBuf, total: u64, h: &WarmHandle) -> Result<(), String> {
    use std::io::Read;
    let mut file = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buf = vec![0u8; WARM_CHUNK];
    let mut read = 0u64;
    loop {
        match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                read += n as u64;
                h.read.store(read.min(total), Ordering::Relaxed);
            }
            Err(e) => return Err(format!("{}: {e}", path.display())),
        }
    }
    Ok(())
}

struct Mapped {
    ptr: *const u8,
    len: usize,
}
// SAFETY: the map is read-only and is released only when Mapped is dropped.
unsafe impl Send for Mapped {}
// SAFETY: shared read-only bytes are safe to read from multiple threads.
unsafe impl Sync for Mapped {}
impl Mapped {
    fn open(path: &Path) -> Result<Self, String> {
        use std::os::fd::AsRawFd;
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let len = usize::try_from(file.metadata().map_err(|e| e.to_string())?.len()).map_err(|e| e.to_string())?;
        if len < 8 {
            return Err(format!("{}: short safetensors file", path.display()));
        }
        // SAFETY: a read-only shared mapping of an open regular file; MAP_FAILED is checked.
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, file.as_raw_fd(), 0) };
        if ptr == libc::MAP_FAILED {
            return Err(format!("{}: mmap failed", path.display()));
        }
        // SAFETY: ptr is a valid len-byte mapping.
        unsafe {
            libc::madvise(ptr, len, libc::MADV_RANDOM);
        }
        Ok(Self { ptr: ptr.cast(), len })
    }
    fn bytes(&self) -> &[u8] {
        // SAFETY: ptr and len are owned by this live mapping.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: ptr and len were returned by mmap and not otherwise unmapped.
        unsafe {
            libc::munmap(self.ptr.cast_mut().cast(), self.len);
        }
    }
}

/// Wrapped signed hash arithmetic, Euclidean modulo, and an EOS boundary before t2.
pub fn row_ids(
    tokens: [u32; 3],
    eos: u32,
    multipliers: &[i64],
    offsets: &[i64],
    sizes: &[i64],
    heads_per_ngram: usize,
    out: &mut [i64],
) -> Result<(), String> {
    let [t0, t1, t2] = tokens;
    let ngram = multipliers.len();
    if ngram != 3
        || heads_per_ngram == 0
        || offsets.len() != 2 * heads_per_ngram
        || sizes.len() != offsets.len()
        || out.len() < offsets.len()
        || sizes.contains(&0)
        || sizes.iter().any(|&s| s < 0)
    {
        return Err("invalid 3-gram hash parameters or output".into());
    }
    let tokens = [t0 as i64, t1 as i64, if t1 == eos { eos as i64 } else { t2 as i64 }];
    let mut mix = tokens[0].wrapping_mul(multipliers[0]);
    for n in 1..3 {
        mix ^= tokens[n].wrapping_mul(multipliers[n]);
        for j in (n - 1) * heads_per_ngram..n * heads_per_ngram {
            out[j] = mix.rem_euclid(sizes[j]) + offsets[j];
        }
    }
    Ok(())
}

/// Read-only mmap of packed table rows plus small hash parameters.
pub struct NgramTable {
    map: Mapped,
    path: PathBuf,
    base: usize,
    pub rows: usize,
    pub k: usize,
    pub words: usize,
    pub heads: usize,
    pub heads_per_ngram: usize,
    offsets: Vec<i64>,
    sizes: Vec<i64>,
    multipliers: Vec<i64>,
    pub eos: u32,
    pub head_bias: Vec<u16>,
    /// The completion-tracked warm worker, started at most once per table.
    warm: OnceLock<Arc<WarmHandle>>,
}

impl NgramTable {
    pub fn open(dir: &Path, cfg: &FlashNextConfig) -> Result<Self, String> {
        let path = dir.join(inventory::NGRAM_FILE);
        let map = Mapped::open(&path)?;
        let bytes = map.bytes();
        let hlen = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap())).map_err(|e| e.to_string())?;
        let data = 8usize
            .checked_add(hlen)
            .filter(|&n| n <= bytes.len() && hlen <= gb_exl3::checkpoint::MAX_HEADER_BYTES as usize)
            .ok_or("ngram header exceeds file")?;
        let head: Value = serde_json::from_slice(&bytes[8..data]).map_err(|e| e.to_string())?;
        let meta = &head["__metadata__"];
        if meta["format"] != "exl3_ngram_trellis" || meta["codebook"] != "mul1" {
            return Err("unsupported ngram format or codebook".into());
        }
        let number = |n: &str| -> Result<usize, String> {
            meta[n].as_str().ok_or_else(|| format!("metadata {n}"))?.parse().map_err(|_| format!("metadata {n}"))
        };
        let (k, rows, shard_rows, dim) = (number("K")?, number("rows")?, number("shard_rows")?, number("row_dim")?);
        if dim != ROW || !(1..=8).contains(&k) || shard_rows == 0 || rows == 0 || cfg.ngram_size != 3 {
            return Err("unsupported ngram row geometry".into());
        }
        let words = inventory::ngram_row_words(k);
        let layer = cfg.ple_layer_ids.first().and_then(|&id| id.checked_sub(1)).ok_or("no valid PLE layer")?;
        let prefix = format!("model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding");
        let span = |name: &str, dtype: &str, shape: &[usize]| -> Result<std::ops::Range<usize>, String> {
            let full = format!("{prefix}.{name}");
            let t = head.get(&full).ok_or_else(|| format!("{full}: missing"))?;
            if t["dtype"] != dtype
                || t["shape"].as_array().is_none_or(|a| {
                    a.iter().map(Value::as_u64).collect::<Option<Vec<_>>>().as_deref()
                        != Some(&shape.iter().map(|&x| x as u64).collect::<Vec<_>>())
                })
            {
                return Err(format!("{full}: dtype or shape mismatch"));
            }
            let a = t["data_offsets"][0].as_u64().ok_or_else(|| format!("{full}: start"))?;
            let b = t["data_offsets"][1].as_u64().ok_or_else(|| format!("{full}: end"))?;
            let unit = gb_exl3::checkpoint::dtype_size(dtype).ok_or("dtype")?;
            let size = shape.iter().try_fold(unit, |v, &d| v.checked_mul(d as u64)).ok_or("shape overflow")?;
            if b.checked_sub(a) != Some(size) || b > (bytes.len() - data) as u64 {
                return Err(format!("{full}: extent"));
            }
            Ok(data + a as usize..data + b as usize)
        };
        let n_shards = rows.div_ceil(shard_rows);
        let base = span("shard_0.trellis", "I16", &[rows.min(shard_rows), words])?.start;
        for s in 0..n_shards {
            let count = shard_rows.min(rows - s * shard_rows);
            let range = span(&format!("shard_{s}.trellis"), "I16", &[count, words])?;
            if range.start != base + s * shard_rows * words * 2 {
                return Err(format!("shard_{s} is not contiguous"));
            }
        }
        let heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
        let parse_i64 = |n: &str, len: usize| -> Result<Vec<i64>, String> {
            let range = span(n, "I64", &[len])?;
            Ok(bytes[range].chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect())
        };
        let offsets = parse_i64("head_offsets", heads)?;
        let sizes = parse_i64("head_vocab_sizes", heads)?;
        let multipliers = parse_i64("layer_multipliers", cfg.ngram_size)?;
        let bias = span("head_bias", "F16", &[heads, ROW])?;
        let head_bias = bytes[bias].chunks_exact(2).map(|b| u16::from_le_bytes(b.try_into().unwrap())).collect();
        for (j, (&offset, &size)) in offsets.iter().zip(&sizes).enumerate() {
            if offset < 0 || size <= 0 || offset.checked_add(size).is_none_or(|end| end as u64 > rows as u64) {
                return Err(format!("head {j} outside the ngram table"));
            }
        }
        Ok(Self {
            map,
            path,
            base,
            rows,
            k,
            words,
            heads,
            heads_per_ngram: cfg.heads_per_ngram,
            offsets,
            sizes,
            multipliers,
            eos: cfg.eos_token_id,
            head_bias,
            warm: OnceLock::new(),
        })
    }

    pub fn row_ids(&self, t0: u32, t1: u32, t2: u32, out: &mut [i64]) -> Result<(), String> {
        row_ids([t0, t1, t2], self.eos, &self.multipliers, &self.offsets, &self.sizes, self.heads_per_ngram, out)
    }

    /// Gather packed rows (u16 words), with ctx >= 2 context tokens before the positions to decode.
    pub fn gather(&self, history: &[u32], ctx: usize, out: &mut [u16]) -> Result<(), String> {
        if ctx < 2 || ctx > history.len() {
            return Err("history needs two context tokens".into());
        }
        let n = history.len() - ctx;
        if out.len() < n * self.heads * self.words {
            return Err("packed row buffer is too short".into());
        }
        let all = self.map.bytes();
        for (p, dst) in out.chunks_exact_mut(self.heads * self.words).take(n).enumerate() {
            let q = p + ctx;
            let mut ids = vec![0; self.heads];
            self.row_ids(history[q], history[q - 1], history[q - 2], &mut ids)?;
            for (j, &id) in ids.iter().enumerate() {
                let at = self.base + id as usize * self.words * 2;
                for (w, b) in
                    dst[j * self.words..(j + 1) * self.words].iter_mut().zip(all[at..at + self.words * 2].chunks_exact(2))
                {
                    *w = u16::from_le_bytes(b.try_into().unwrap());
                }
            }
        }
        Ok(())
    }

    /// Sequentially fault/read the whole mmap file on a completion-tracked worker, idempotently:
    /// repeated calls share one worker and its handle. The returned handle's [`WarmHandle::wait`]
    /// reports success or the explicit read failure; readiness waits on it before serving.
    pub fn warm(self: &Arc<Self>) -> Arc<WarmHandle> {
        self.warm
            .get_or_init(|| {
                let path = self.path.clone();
                let total = self.map.len as u64;
                let h = Arc::new(WarmHandle::new());
                let hw = Arc::clone(&h);
                std::thread::Builder::new()
                    .name("qwen-ngram-warm".into())
                    .spawn(move || hw.finish(warm_read(path, total, &hw)))
                    .expect("spawn the n-gram warm worker");
                h
            })
            .clone()
    }
}

/// Start the completion-tracked warm worker when `warm` is on. Off bypasses without touching the
/// table: no worker and no file access.
pub fn warm_if(warm: bool, table: &Arc<NgramTable>) -> Option<Arc<WarmHandle>> {
    warm.then(|| table.warm())
}

/// The readiness gate: block until the warm worker (if any) completes. Failures are reported
/// explicitly (naming the file) and abort load before the server can become ready.
pub fn finish_warm(handle: Option<Arc<WarmHandle>>) -> Result<(), String> {
    handle.map_or(Ok(()), |h| h.wait())
}

/// The warm preflight the loader runs: start warm if requested and wait for its successful
/// completion before load proceeds. `None` table (no PLE stage) or off bypasses outright.
pub fn warm_preflight(warm: bool, table: Option<&Arc<NgramTable>>) -> Result<(), String> {
    match (warm, table) {
        (false, _) | (_, None) => Ok(()),
        (true, Some(t)) => {
            let start = std::time::Instant::now();
            let handle = t.warm();
            finish_warm(Some(Arc::clone(&handle)))?;
            let bytes = handle.bytes_read();
            eprintln!("rexl3 event=qwen_ngram_warm done bytes={} secs={:.3}", bytes, start.elapsed().as_secs_f64());
            gb_engine::boot::mark("ngram_warmed", Some(bytes as f64 / (1u64 << 30) as f64));
            Ok(())
        }
    }
}

/// Per-slot token ring. Rejected draft positions are replaced by the next store at the same indices.
pub struct TokContext {
    ring: Vec<[u32; RING]>,
}
impl TokContext {
    pub fn new(slots: usize) -> Self {
        Self { ring: vec![[0; RING]; slots] }
    }
    pub fn history(&self, slot: usize, first: usize, ids: &[i32], eos: u32) -> Vec<u32> {
        let at = |q: isize| if q < 0 { eos } else { self.ring[slot][q as usize % RING] };
        let mut out = Vec::with_capacity(ids.len() + 2);
        out.push(at(first as isize - 2));
        out.push(at(first as isize - 1));
        out.extend(ids.iter().map(|&v| v as u32));
        out
    }
    pub fn store(&mut self, slot: usize, first: usize, ids: &[i32]) {
        let skip = ids.len().saturating_sub(RING);
        for (j, &token) in ids.iter().enumerate().skip(skip) {
            self.ring[slot][(first + j) % RING] = token as u32;
        }
    }
}

#[cfg(test)]
#[path = "ngram_tests.rs"]
mod tests;
