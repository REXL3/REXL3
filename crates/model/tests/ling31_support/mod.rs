//! Shared by the Ling-3.1 model-level GPU tests: the pack's directory (a test prints "not run" and passes where `REXL3_LING_DIR` is
//! unset), a memory guard for the unified pool (the weights, the activations and the host share one 121 GiB pool; an over-commit can
//! hang the box, which has no hardware watchdog), the model built the way `bailing_moe_v3::engine::build_engine` builds it, and a digest
//! of a slot's state that reads only the model's public buffers.
#![allow(dead_code)]

use gb_model::families::bailing_moe_v3::config::BailingConfig;
use gb_model::families::bailing_moe_v3::device::BailingDevice;
use gb_model::families::bailing_moe_v3::inventory::PackInventory;
use gb_model::families::bailing_moe_v3::load::{load_pack, BailingPack};
use gb_model::families::bailing_moe_v3::model::BailingModel;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub const GIB: u64 = 1 << 30;

/// The pack named by `REXL3_LING_DIR`, or `None` after printing that the test did not run.
pub fn pack_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_LING_DIR") {
        Ok(d) if Path::new(&d).join("config.json").exists() => Some(PathBuf::from(d)),
        Ok(d) => panic!("REXL3_LING_DIR={d} has no config.json"),
        Err(_) => {
            eprintln!("REXL3_LING_DIR is not set: not run");
            None
        }
    }
}

pub fn mem_available() -> u64 {
    let t = std::fs::read_to_string("/proc/meminfo").unwrap();
    t.lines()
        .find(|l| l.starts_with("MemAvailable:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
        * 1024
}

pub fn gib(b: u64) -> f64 {
    b as f64 / GIB as f64
}

/// Aborts the whole process when `MemAvailable` falls under the floor and remembers the lowest value it saw.
pub struct Guard {
    stop: Arc<AtomicBool>,
    min: Arc<AtomicU64>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Guard {
    pub fn start(floor: u64) -> Guard {
        let (stop, min) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(u64::MAX)));
        let (s, m) = (stop.clone(), min.clone());
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let a = mem_available();
                m.fetch_min(a, Ordering::Relaxed);
                if a < floor {
                    eprintln!(
                        "GUARD: MemAvailable {:.2} GiB is under the {:.1} GiB floor: aborting the process",
                        gib(a),
                        gib(floor)
                    );
                    std::process::abort();
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        });
        Guard { stop, min, thread: Some(thread) }
    }
    /// The lowest `MemAvailable` seen so far.
    pub fn low(&self) -> u64 {
        self.min.load(Ordering::Relaxed)
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The process's memory guard (1 GiB floor), started before the first load and kept for the process's life.
pub fn guard() -> &'static Guard {
    static G: std::sync::OnceLock<Guard> = std::sync::OnceLock::new();
    G.get_or_init(|| Guard::start(GIB))
}

/// Build the model as `build_engine` does (the pack's regions on the device, the embedding table from its shard, `BailingModel::new`)
/// with `slots` slots, a KV pool of `kv_tokens` tokens and the MTP head when `mtp`. Call it once per process: a second model does not
/// fit in the pool.
pub fn load(slots: usize, kv_tokens: usize, mtp: bool) -> Option<BailingModel> {
    let dir = pack_dir()?;
    let g = guard();
    let t = std::time::Instant::now();
    let cfg = BailingConfig::from_dir(&dir).unwrap();
    let ck = gb_exl3::checkpoint::scan(&dir, false).unwrap();
    let inv = PackInventory::from_checkpoint(&ck).unwrap();
    let pack = BailingPack::from_inventory(&inv, cfg.clone()).unwrap();
    let loaded = load_pack(&dir, &pack, &mut |_, _| {}).unwrap();
    let dev = BailingDevice::build(&cfg, &pack.weights, loaded.regions).unwrap();
    let m = BailingModel::new(cfg, dev, loaded.embedding, slots, kv_tokens, mtp).unwrap();
    eprintln!(
        "model ready in {:.1?}: {slots} slots, {kv_tokens} KV tokens, mtp {mtp}; MemAvailable {:.2} GiB (low {:.2})",
        t.elapsed(),
        gib(mem_available()),
        gib(g.low())
    );
    Some(m)
}

/// The model of this test process, loaded on first use (`--test-threads=1`).
pub fn shared(slots: usize, kv_tokens: usize, mtp: bool) -> Option<std::sync::MutexGuard<'static, BailingModel>> {
    use std::sync::{Mutex, OnceLock};
    static M: OnceLock<Option<Mutex<BailingModel>>> = OnceLock::new();
    let m = M.get_or_init(|| load(slots, kv_tokens, mtp).map(Mutex::new));
    m.as_ref().map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Deterministic token ids that are not degenerate.
pub fn ids(seed: u64, n: usize) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1 + (x % 150_000) as u32
        })
        .collect()
}

/// FNV-1a over a stream of bytes (fed piece by piece, so a digest of the whole state never holds it in host memory at once).
pub struct Fnv(u64);

impl Fnv {
    pub fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    pub fn update(&mut self, bytes: &[u8]) {
        self.0 = bytes.iter().fold(self.0, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
    }
    pub fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

impl Default for Fnv {
    fn default() -> Self {
        Self::new()
    }
}

/// FNV-1a over bytes, hex.
pub fn fnv(bytes: &[u8]) -> String {
    let mut f = Fnv::new();
    f.update(bytes);
    f.hex()
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// The state a slot holds after `len` committed tokens, as digests: every KDA layer's current state buffer (the one the parity word
/// names; the other holds an older state) and conv history, every trunk MLA cache's rows `0..len`, and the MTP layer's rows
/// `0..mtp_rows`. Equal digests are equal bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotDigest {
    pub kda_state: String,
    pub kda_conv: String,
    pub mla: String,
    pub mtp: String,
}

pub fn slot_digest(m: &BailingModel, slot: usize, len: usize, mtp_rows: usize) -> SlotDigest {
    // the reads below are copies on the legacy stream, which the model's non-blocking stream does not order with
    m.stream.sync().unwrap();
    let (mut state, mut conv) = (Fnv::new(), Fnv::new());
    for p in &m.pools {
        let (s, parity, c) = p.slot_snapshot(slot).unwrap();
        let half = s.len() / 2;
        let cur = parity as usize & 1;
        state.update(as_bytes(&s[cur * half..(cur + 1) * half]));
        conv.update(as_bytes(&c));
    }
    let rows_of = |c: usize, n: usize, f: &mut Fnv| {
        let cache = &m.mla_caches[c];
        let bb = gb_cuda::mla_gated::MlaGatedCache::block_bytes(cache.dtype);
        let row = bb / gb_cuda::mla_gated::BLOCK_TOKENS;
        for (i, &page) in m.slot_pages[slot].iter().enumerate() {
            let lo = i * gb_cuda::mla_gated::BLOCK_TOKENS;
            if lo >= n {
                break;
            }
            let rows = (n - lo).min(gb_cuda::mla_gated::BLOCK_TOKENS);
            let mut buf = vec![0u8; rows * row];
            cache.buf.download_range(page as usize * bb, &mut buf).unwrap();
            f.update(&buf);
        }
    };
    let trunk = m.cfg.mla_layers().len();
    let mut mla = Fnv::new();
    for c in 0..trunk {
        rows_of(c, len, &mut mla);
    }
    let mtp = if m.mla_caches.len() > trunk && mtp_rows > 0 {
        let mut f = Fnv::new();
        rows_of(trunk, mtp_rows, &mut f);
        f.hex()
    } else {
        "-".into()
    };
    SlotDigest { kda_state: state.hex(), kda_conv: conv.hex(), mla: mla.hex(), mtp }
}

/// The f16 logits of the last row the head ran on (a prefill chunk's last row, or a window's first), as a digest.
pub fn logits_digest(m: &BailingModel) -> String {
    m.stream.sync().unwrap();
    let mut v = vec![0u16; m.cfg.vocab];
    m.s.logits.download_range(0, &mut v).unwrap();
    fnv(as_bytes(&v))
}

/// Prefill `toks[from..to]` into `slot` in calls of at most `piece` tokens (the model chunks each call by its own rows), the last call
/// asking for 8 candidates; returns them.
pub fn prefill(
    m: &mut BailingModel,
    slot: usize,
    toks: &[u32],
    from: usize,
    to: usize,
    piece: usize,
) -> gb_model::forward::Candidates {
    let mut at = from;
    let mut out = None;
    while at < to {
        let end = (at + piece).min(to);
        let want = (end == to).then_some(8);
        out = m.prefill_chunk(slot, at, &toks[at..end], want).unwrap();
        at = end;
    }
    out.expect("a prefill of at least one token")
}

/// `steps` greedy decode steps after a prefill whose last row's candidates are `first` (one-row windows, each committed), returning
/// the tokens and a digest of every step's top-8 ids and values.
pub fn greedy(
    m: &mut BailingModel,
    slot: usize,
    toks: &[u32],
    first: &gb_model::forward::Candidates,
    steps: usize,
) -> (Vec<u32>, String) {
    let mut seq = toks.to_vec();
    let mut tok = first.ids[0] as u32;
    let mut trace: Vec<u8> = Vec::new();
    trace.extend_from_slice(as_bytes(&first.ids));
    trace.extend_from_slice(as_bytes(&first.vals));
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        out.push(tok);
        seq.push(tok);
        let b = gb_model::rows::RowBatch::from_requests(&[gb_engine::RowRequest {
            slot,
            tokens: vec![tok],
            first_position: (seq.len() - 1) as u64,
        }]);
        let c = m.window_candidates(&b, 8).unwrap();
        m.commit(slot, 1);
        trace.extend_from_slice(as_bytes(&c.ids));
        trace.extend_from_slice(as_bytes(&c.vals));
        tok = c.ids[0] as u32;
    }
    (out, fnv(&trace))
}
