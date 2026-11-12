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
    let m = BailingModel::new(
        cfg,
        dev,
        loaded.embedding,
        slots,
        kv_tokens,
        mtp,
        gb_model::families::bailing_moe_v3::alloc::DEFAULT_PREFILL_CHUNK,
    )
    .unwrap();
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

/// The bit-identity digest battery (`ling_bits_battery`, `ling_bits_localize` in `ling31_prefill.rs`; run it through `tools/ling_bits.sh`,
/// compare two files with `tools/ling_bits.py`; the file format, the position keying and the layer-major hook are described in
/// `docs/LING_BIT_IDENTITY.md`). Everything here reads the model's public buffers after a `prefill_chunk` call: no tap inside the
/// forward pass, no environment switch in the engine. The `REXL3_BITS_*` variables are read by this test harness only.
pub mod bits {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::io::{BufWriter, Write};

    pub const FORMAT: &str = "ling-bits v1";
    /// A prompt longer than this is digested per `GRAN_BIG` rows (the 16-row KDA chunk grid) instead of per row: the file stays small.
    /// `REXL3_BITS_GRAN=1` forces per-row digests (re-run one prompt with `REXL3_BITS_ONLY` to name the row inside a block).
    pub const BIG_PROMPT: usize = 400;
    pub const GRAN_BIG: usize = 16;
    /// The tail lengths (rows after the cut) of the prefix cache's cut policy: the cut is a multiple of 16 and 64..=79 rows remain.
    pub const TAILS_ALL: [usize; 16] = [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79];
    pub const TAILS_FEW: [usize; 3] = [64, 71, 79];

    pub struct Prompt {
        pub name: String,
        pub ids: Vec<u32>,
    }

    /// `REXL3_BITS_PROMPTS`: one prompt per line, `name<TAB>id id id ...` (made by `tools/ling_bits_prompts.py`).
    pub fn read_prompts(path: &str) -> Vec<Prompt> {
        let only: Option<Vec<String>> =
            std::env::var("REXL3_BITS_ONLY").ok().filter(|v| !v.is_empty()).map(|v| v.split(',').map(String::from).collect());
        let mut out = Vec::new();
        for line in std::fs::read_to_string(path).unwrap_or_else(|e| panic!("REXL3_BITS_PROMPTS {path}: {e}")).lines() {
            let Some((name, ids)) = line.split_once('\t') else { continue };
            if only.as_ref().is_some_and(|o| !o.iter().any(|x| x == name)) {
                continue;
            }
            let ids: Vec<u32> = ids.split_whitespace().map(|t| t.parse().unwrap()).collect();
            out.push(Prompt { name: name.to_string(), ids });
        }
        assert!(!out.is_empty(), "no prompt selected from {path}");
        out
    }

    pub fn env_list(key: &str, default: &[usize]) -> Vec<usize> {
        match std::env::var(key) {
            Ok(v) if !v.trim().is_empty() => {
                v.split(',').map(|t| t.trim().parse().unwrap_or_else(|_| panic!("{key}: bad number {t:?}"))).collect()
            }
            _ => default.to_vec(),
        }
    }

    fn env_num(key: &str, default: usize) -> usize {
        std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    /// Rows per digest block for a prompt of `len` tokens.
    pub fn gran_for(len: usize) -> usize {
        match env_num("REXL3_BITS_GRAN", 0) {
            0 if len > BIG_PROMPT => GRAN_BIG,
            0 => 1,
            g => g,
        }
    }

    fn sha256_file(p: &Path) -> String {
        let mut h = Sha256::new();
        h.update(std::fs::read(p).unwrap_or_default());
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// FNV-1a of every block of `gran` rows (the last block may be short), `row_bytes` bytes a row.
    pub fn block_hashes(bytes: &[u8], row_bytes: usize, gran: usize) -> Vec<u64> {
        let rows = bytes.len() / row_bytes;
        let mut v = Vec::with_capacity(rows.div_ceil(gran));
        let mut r = 0;
        while r < rows {
            let e = (r + gran).min(rows);
            let mut f = Fnv::new();
            f.update(&bytes[r * row_bytes..e * row_bytes]);
            v.push(u64::from_str_radix(&f.hex(), 16).unwrap());
            r = e;
        }
        v
    }

    /// The digest file: header lines, then `R run kind layer pos0 gran nblocks hash...` (per-block row hashes) and
    /// `S run kind layer pos value` (one value) records, flushed after every run.
    pub struct Out {
        f: BufWriter<std::fs::File>,
        pub records: usize,
    }

    impl Out {
        pub fn create(path: &str, header: &[(String, String)]) -> Out {
            let mut f = BufWriter::new(std::fs::File::create(path).unwrap_or_else(|e| panic!("REXL3_BITS_OUT {path}: {e}")));
            writeln!(f, "#{FORMAT}").unwrap();
            for (k, v) in header {
                writeln!(f, "#H {k}={v}").unwrap();
            }
            Out { f, records: 0 }
        }
        pub fn rows(&mut self, run: &str, kind: &str, layer: usize, pos0: usize, gran: usize, h: &[u64]) {
            write!(self.f, "R {run} {kind} {layer} {pos0} {gran} {}", h.len()).unwrap();
            for x in h {
                write!(self.f, " {x:016x}").unwrap();
            }
            writeln!(self.f).unwrap();
            self.records += 1;
        }
        pub fn scalar(&mut self, run: &str, kind: &str, layer: usize, pos: usize, value: &str) {
            writeln!(self.f, "S {run} {kind} {layer} {pos} {value}").unwrap();
            self.records += 1;
        }
        pub fn flush(&mut self) {
            self.f.flush().unwrap();
        }
    }

    fn header(model: &BailingModel, prompts_path: &str, extra: Vec<(String, String)>) -> Vec<(String, String)> {
        let dir = pack_dir().unwrap();
        let mut h = vec![
            ("build".into(), std::env::var("REXL3_BITS_BUILD").unwrap_or_else(|_| "unknown".into())),
            ("binary_sha256".into(), std::env::var("REXL3_BITS_BIN_SHA").unwrap_or_else(|_| "unknown".into())),
            ("pack".into(), dir.display().to_string()),
            ("config_sha256".into(), sha256_file(&dir.join("config.json"))),
            ("index_sha256".into(), sha256_file(&dir.join("model.safetensors.index.json"))),
            ("prompts_sha256".into(), sha256_file(Path::new(prompts_path))),
            ("layers".into(), model.cfg.layers.len().to_string()),
            ("hidden".into(), model.cfg.hidden.to_string()),
            (
                "gran".into(),
                format!(
                    "{} rows up to {BIG_PROMPT} tokens, then {GRAN_BIG} (REXL3_BITS_GRAN={})",
                    1,
                    env_num("REXL3_BITS_GRAN", 0)
                ),
            ),
        ];
        h.extend(extra);
        h
    }

    /// How a prompt is prefilled. `LayerMajor(g)` is the hook for the layer-major scheduler (lane W2): it needs a model method that
    /// runs groups of `g` rows through each layer in turn; until the build has one, the arm below skips with a message.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Sched {
        ChunkMajor,
        LayerMajor(usize),
    }

    impl Sched {
        pub fn tag(&self) -> String {
            match self {
                Sched::ChunkMajor => "chunk".into(),
                Sched::LayerMajor(g) => format!("layer{g}"),
            }
        }
    }

    pub fn scheds() -> Vec<Sched> {
        let v = std::env::var("REXL3_BITS_SCHEDS").unwrap_or_default();
        let mut out = vec![Sched::ChunkMajor];
        for t in v.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            match t {
                "chunk" => {}
                t if t.starts_with("layer") => out.push(Sched::LayerMajor(t[5..].parse().expect("REXL3_BITS_SCHEDS: layer<G>"))),
                t => panic!("REXL3_BITS_SCHEDS: unknown scheduler {t:?}"),
            }
        }
        out
    }

    /// Row hashes of the final hidden state (`s.hidden`, the residual stream after the last layer), accumulated piece by piece.
    struct HidAcc {
        gran: usize,
        next: usize,
        hashes: Vec<u64>,
    }

    impl HidAcc {
        fn new(gran: usize, from: usize) -> Self {
            HidAcc { gran, next: from, hashes: Vec::new() }
        }
        /// Called after a prefill call that left the rows `start..start + rows` in `m.s.hidden[0..rows]`.
        fn piece(&mut self, m: &BailingModel, start: usize, rows: usize) {
            assert_eq!(start, self.next, "pieces must follow each other");
            assert!(self.gran == 1 || start % self.gran == 0, "a piece starts off the digest grid");
            let rb = m.cfg.hidden * 2;
            m.stream.sync().unwrap();
            let mut buf = vec![0u8; rows * rb];
            m.s.hidden.download_range(0, &mut buf[..]).unwrap();
            self.hashes.extend(block_hashes(&buf, rb, self.gran));
            self.next = start + rows;
        }
    }

    /// Prefill `toks[from..to]` into slot 0 in calls of at most `piece` rows (chunk-major), the last call asking for 8 candidates, and
    /// feed each call's final hidden rows to `acc`. The layer-major arm is the hook for lane W2: replace the `None` with
    /// `Some(m.prefill_layer_major(...))`, calling `acc.piece(m, start, rows)` after the LAST layer of each group of rows.
    fn prefill_with(
        m: &mut BailingModel,
        toks: &[u32],
        from: usize,
        to: usize,
        piece: usize,
        sched: Sched,
        acc: &mut HidAcc,
    ) -> Option<gb_model::forward::Candidates> {
        assert!(piece <= m.prefill_rows, "piece {piece} > the model's prefill rows {}", m.prefill_rows);
        match sched {
            Sched::ChunkMajor => {
                let mut at = from;
                let mut out = None;
                while at < to {
                    let end = (at + piece).min(to);
                    let want = (end == to).then_some(8);
                    out = m.prefill_chunk(0, at, &toks[at..end], want).unwrap();
                    acc.piece(m, at, end - at);
                    at = end;
                }
                out
            }
            Sched::LayerMajor(_) => None,
        }
    }

    /// The per-layer state digests: every trunk MLA layer's cache rows `0..n` (per block of rows) and every KDA layer's state and conv.
    fn state_records(m: &BailingModel, o: &mut Out, run: &str, n: usize, gran: usize) {
        m.stream.sync().unwrap();
        for (l, &ord) in m.mla_ord.iter().enumerate() {
            if ord == usize::MAX || ord >= m.mla_caches.len() {
                continue;
            }
            let cache = &m.mla_caches[ord];
            let bb = gb_cuda::mla_gated::MlaGatedCache::block_bytes(cache.dtype);
            let row = bb / gb_cuda::mla_gated::BLOCK_TOKENS;
            let mut hashes = Vec::new();
            for (i, &page) in m.slot_pages[0].iter().enumerate() {
                let lo = i * gb_cuda::mla_gated::BLOCK_TOKENS;
                if lo >= n {
                    break;
                }
                let rows = (n - lo).min(gb_cuda::mla_gated::BLOCK_TOKENS);
                let mut buf = vec![0u8; rows * row];
                cache.buf.download_range(page as usize * bb, &mut buf).unwrap();
                hashes.extend(block_hashes(&buf, row, gran));
            }
            o.rows(run, "mla", l, 0, gran, &hashes);
        }
        for (l, &ord) in m.kda_ord.iter().enumerate() {
            if ord == usize::MAX {
                continue;
            }
            let (s, parity, c) = m.pools[ord].slot_snapshot(0).unwrap();
            let half = s.len() / 2;
            let cur = parity as usize & 1;
            let sb = as_bytes(&s[cur * half..(cur + 1) * half]);
            o.scalar(run, "kdas", l, n, &fnv(sb));
            o.scalar(run, "kdac", l, n, &fnv(as_bytes(&c)));
        }
    }

    /// FNV-1a of the first `rows` rows of a device buffer of `row_elems`-element rows (u16 elements).
    fn row_hashes_u16(m: &BailingModel, buf: &gb_cuda::rt::DevBuf, rows: usize, row_elems: usize) -> Vec<u64> {
        m.stream.sync().unwrap();
        let mut v = vec![0u16; rows * row_elems];
        buf.download_range(0, &mut v).unwrap();
        block_hashes(as_bytes(&v), row_elems * 2, 1)
    }

    /// The decode side: `steps` serial greedy single-row steps after the prefill whose last row's candidates are `first`, then one verify window
    /// each of 4, 3 and 2 rows (fixed pseudo-random token ids: a window's bits do not depend on what the tokens are, only on being the same).
    /// Per step / window row it records the residual stream after the last layer (`dhid` / `vhid<w>`), the whole f16 logits row (`dlog` / `vlog<w>`) and
    /// the top-8 candidates, so a decode kernel that moves one low bit anywhere shows up, not only one that changes the argmax.
    fn decode_records(
        m: &mut BailingModel,
        o: &mut Out,
        run: &str,
        toks: &[u32],
        first: &gb_model::forward::Candidates,
        steps: usize,
    ) {
        let (h, vocab) = (m.cfg.hidden, m.cfg.vocab);
        let last = m.cfg.layers.len() - 1;
        let n = toks.len();
        let mut seq = toks.to_vec();
        let mut tok = first.ids[0] as u32;
        let (mut out_toks, mut dhid, mut dlog, mut cand) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..steps {
            out_toks.push(tok);
            seq.push(tok);
            let b = gb_model::rows::RowBatch::from_requests(&[gb_engine::RowRequest {
                slot: 0,
                tokens: vec![tok],
                first_position: (seq.len() - 1) as u64,
            }]);
            let c = m.window_candidates(&b, 8).unwrap();
            m.commit(0, 1);
            dhid.extend(row_hashes_u16(m, &m.s.hidden, 1, h));
            dlog.extend(row_hashes_u16(m, &m.s.logits, 1, vocab));
            cand.extend_from_slice(as_bytes(&c.ids));
            cand.extend(c.vals.iter().flat_map(|v| v.to_bits().to_le_bytes()));
            tok = c.ids[0] as u32;
        }
        o.scalar(run, "greedy", 9999, n, &out_toks.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
        o.scalar(run, "gtrace", 9999, n, &fnv(&cand));
        o.rows(run, "dhid", 9999, n, 1, &dhid);
        o.rows(run, "dlog", 9999, n, 1, &dlog);
        // verify windows: 4, 3 and 2 rows of one sequence, committed whole
        for (i, w) in [4usize, 3, 2].into_iter().enumerate() {
            let wt = ids(0x57 + i as u64, w);
            let pos = seq.len();
            seq.extend_from_slice(&wt);
            let b = gb_model::rows::RowBatch::from_requests(&[gb_engine::RowRequest {
                slot: 0,
                tokens: wt,
                first_position: pos as u64,
            }]);
            let c = m.window_candidates(&b, 8).unwrap();
            m.commit(0, w);
            o.rows(run, &format!("vhid{w}"), last, pos, 1, &row_hashes_u16(m, &m.s.hidden, w, h));
            o.rows(run, &format!("vlog{w}"), 9999, pos, 1, &row_hashes_u16(m, &m.s.logits, w, vocab));
            let mut cb = Vec::new();
            cb.extend_from_slice(as_bytes(&c.ids));
            cb.extend(c.vals.iter().flat_map(|v| v.to_bits().to_le_bytes()));
            o.scalar(run, &format!("vcand{w}"), 9999, pos, &fnv(&cb));
        }
    }

    /// Everything one finished prefill leaves, as records of `run`; the optional MTP absorb and the greedy continuation come last
    /// (they advance the slot).
    #[allow(clippy::too_many_arguments)]
    fn run_records(
        m: &mut BailingModel,
        o: &mut Out,
        run: &str,
        toks: &[u32],
        hid: &[u64],
        gran: usize,
        first: &gb_model::forward::Candidates,
        greedy_steps: usize,
        mtp: bool,
    ) {
        let n = toks.len();
        o.rows(run, "hid", m.cfg.layers.len() - 1, 0, gran, hid);
        o.scalar(run, "logits", 9999, n - 1, &logits_digest(m));
        let mut cb = Vec::new();
        cb.extend_from_slice(as_bytes(&first.ids));
        cb.extend(first.vals.iter().flat_map(|v| v.to_bits().to_le_bytes()));
        o.scalar(run, "cand", 9999, n - 1, &fnv(&cb));
        state_records(m, o, run, n, gran);
        if mtp {
            m.mtp_absorb_up_to(0, toks).unwrap();
            o.scalar(run, "mtp_absorb", 9999, n, &slot_digest(m, 0, n, n - 1).mtp);
            let mut seq = toks.to_vec();
            seq.push(first.ids[0] as u32);
            let drafts = m.mtp_propose(0, &seq, 3).unwrap();
            o.scalar(run, "drafts", 9999, n, &format!("{drafts:?}").replace(' ', ""));
        }
        if greedy_steps > 0 {
            decode_records(m, o, run, toks, first, greedy_steps);
        }
        o.flush();
    }

    /// The battery: for every prompt, scheduler and prefill piece size, the whole prefill of `cut + tail` rows for every tail of the
    /// prefix cache's window (cut a multiple of 16, 64..=79 rows after it) and the same prompt as a donor prefill of `cut` rows, a
    /// snapshot, and the restore + tail prefill the cache performs. Digest files of two builds are compared by `tools/ling_bits.py`.
    pub fn battery() {
        let (Ok(pp), Ok(op)) = (std::env::var("REXL3_BITS_PROMPTS"), std::env::var("REXL3_BITS_OUT")) else {
            eprintln!("REXL3_BITS_PROMPTS / REXL3_BITS_OUT not set: not run");
            return;
        };
        let prompts = read_prompts(&pp);
        let chunks = env_list("REXL3_BITS_CHUNKS", &[128, 256, 1024, 2048, 4096]);
        let tails_env = env_list("REXL3_BITS_TAILS", &[]);
        let greedy_tails = env_list("REXL3_BITS_GREEDY_TAILS", &[64, 79]);
        let greedy_steps = env_num("REXL3_BITS_GREEDY", 32);
        let mtp_tails = env_list("REXL3_BITS_MTP_TAILS", &[64]);
        let mtp_max = env_num("REXL3_BITS_MTP_MAX_LEN", 4000);
        // a short prompt is one piece at every size of `chunks`: these extra sizes make it several
        let small_pieces = env_list("REXL3_BITS_SMALL_PIECES", &[16, 48]);
        let snap_max = env_num("REXL3_BITS_SNAPSHOT_MAX_LEN", 6000);
        let scheds = scheds();
        let kv = prompts.iter().map(|p| p.ids.len()).max().unwrap() + 512;
        let Some(mut mg) = shared(1, kv.div_ceil(256) * 256, env_num("REXL3_BITS_HEAD", 1) == 1) else { return };
        let m = &mut *mg;
        m.enable_prefix_cache(1);
        let extra = vec![
            ("chunks".to_string(), format!("{chunks:?}")),
            ("tails".to_string(), if tails_env.is_empty() { "auto".to_string() } else { format!("{tails_env:?}") }),
            ("greedy".to_string(), format!("{greedy_steps} at {greedy_tails:?}")),
            ("mtp".to_string(), format!("{mtp_tails:?} up to {mtp_max}")),
            ("small_pieces".to_string(), format!("{small_pieces:?} for prompts up to {BIG_PROMPT} tokens")),
            ("snapshot_max_len".to_string(), snap_max.to_string()),
            ("scheds".to_string(), scheds.iter().map(Sched::tag).collect::<Vec<_>>().join(",")),
            ("host".to_string(), std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim().to_string()),
        ];
        let mut o = Out::create(&op, &header(m, &pp, extra));
        let t0 = std::time::Instant::now();
        for p in &prompts {
            let len = p.ids.len();
            assert!(len >= 80 + 16, "prompt {} has {len} tokens: the cut policy needs more than 80", p.name);
            let cut = (len - 79) / 16 * 16;
            let tails: Vec<usize> = if !tails_env.is_empty() {
                tails_env.clone()
            } else if len > 6000 {
                TAILS_FEW.to_vec()
            } else {
                TAILS_ALL.to_vec()
            };
            let gran = gran_for(len);
            for &sched in &scheds {
                if matches!(sched, Sched::LayerMajor(_)) {
                    eprintln!(
                        "SKIP {} {}: this build has no layer-major scheduler (see docs/LING_BIT_IDENTITY.md)",
                        p.name,
                        sched.tag()
                    );
                    continue;
                }
                let mut pieces = chunks.clone();
                if len <= BIG_PROMPT {
                    pieces.extend(small_pieces.iter().filter(|s| !chunks.contains(s)));
                }
                for &piece in &pieces {
                    // the donor: rows 0..cut, then its snapshot (the state the prefix cache keeps). A long prompt keeps no host
                    // snapshot (the unified pool has no room for it beside the weights): its cut run continues from the live state.
                    m.begin_slot(0).unwrap();
                    let mut dacc = HidAcc::new(gran, 0);
                    prefill_with(m, &p.ids, 0, cut, piece, sched, &mut dacc);
                    let snap = if len <= snap_max {
                        let s = m.alloc_snapshot().expect("a free snapshot slot");
                        m.prefix_snapshot(0, s, cut).unwrap();
                        Some(s)
                    } else {
                        None
                    };
                    for &t in &tails {
                        let n = cut + t;
                        let toks = &p.ids[..n];
                        let g = if greedy_tails.contains(&t) { greedy_steps } else { 0 };
                        let mt = mtp_tails.contains(&t) && n <= mtp_max;
                        let tag = sched.tag();
                        // whole
                        let run = format!("{}|n={n}|{tag}|c={piece}|whole", p.name);
                        m.begin_slot(0).unwrap();
                        let mut acc = HidAcc::new(gran, 0);
                        let first = prefill_with(m, toks, 0, n, piece, sched, &mut acc).unwrap();
                        run_records(m, &mut o, &run, toks, &acc.hashes, gran, &first, g, mt);
                        // cut: restore the donor's snapshot and prefill the tail, as the prefix cache does
                        let run = format!("{}|n={n}|{tag}|c={piece}|cut={cut}", p.name);
                        match snap {
                            Some(s) => m.restore_prefix(0, s, &[], cut).unwrap(),
                            None => {
                                m.begin_slot(0).unwrap();
                                let mut sink = HidAcc::new(gran, 0);
                                prefill_with(m, &p.ids, 0, cut, piece, sched, &mut sink);
                            }
                        }
                        let mut tacc = HidAcc::new(gran, cut);
                        let first = prefill_with(m, toks, cut, n, piece, sched, &mut tacc).unwrap();
                        let mut hid = dacc.hashes.clone();
                        hid.extend_from_slice(&tacc.hashes);
                        run_records(m, &mut o, &run, toks, &hid, gran, &first, g, mt);
                        eprintln!(
                            "{} done at {:.0?}; MemAvailable {:.1} GiB (low {:.1})",
                            run,
                            t0.elapsed(),
                            gib(mem_available()),
                            gib(guard().low())
                        );
                    }
                    if let Some(s) = snap {
                        m.release_prefix(s);
                    }
                }
            }
            m.end_slot(0);
        }
        o.flush();
        eprintln!("battery done: {} records in {:.0?}; MemAvailable low {:.2} GiB", o.records, t0.elapsed(), gib(guard().low()));
    }

    /// Layer localization: the residual stream after layer `d - 1` for every row, found by running the first `d` layers only (the
    /// model's config is cut to `d` layers for the run: `trunk_kernels` loops over `cfg.layers`, and the layer that is then the last
    /// leaves its output in `s.hidden`). `REXL3_BITS_DEPTHS` (default every depth 1..=layers), chunks `REXL3_BITS_LOC_CHUNKS`
    /// (default 4096,256), tail `REXL3_BITS_LOC_TAIL` (default 64). Records are `hid` records of the run `...|trunc`; two files are
    /// compared by `tools/ling_bits.py` like the battery's, and the first differing record names the first layer whose output differs.
    pub fn localize() {
        let (Ok(pp), Ok(op)) = (std::env::var("REXL3_BITS_PROMPTS"), std::env::var("REXL3_BITS_OUT")) else {
            eprintln!("REXL3_BITS_PROMPTS / REXL3_BITS_OUT not set: not run");
            return;
        };
        let prompts = read_prompts(&pp);
        let chunks = env_list("REXL3_BITS_LOC_CHUNKS", &[4096, 256]);
        let tail = env_num("REXL3_BITS_LOC_TAIL", 64);
        let kv = prompts.iter().map(|p| p.ids.len()).max().unwrap() + 512;
        let Some(mut mg) = shared(1, kv.div_ceil(256) * 256, env_num("REXL3_BITS_HEAD", 1) == 1) else { return };
        let m = &mut *mg;
        let full = m.cfg.layers.clone();
        let depths = env_list("REXL3_BITS_DEPTHS", &(1..=full.len()).collect::<Vec<_>>());
        let extra = vec![
            ("chunks".to_string(), format!("{chunks:?}")),
            ("depths".to_string(), format!("{depths:?}")),
            ("tail".to_string(), tail.to_string()),
        ];
        let mut o = Out::create(&op, &header(m, &pp, extra));
        for p in &prompts {
            let len = p.ids.len();
            let n = ((len - 79) / 16 * 16 + tail).min(len);
            let gran = gran_for(n);
            for &piece in &chunks {
                for &d in &depths {
                    assert!(d >= 1 && d <= full.len());
                    m.cfg.layers = full[..d].to_vec();
                    m.begin_slot(0).unwrap();
                    let mut acc = HidAcc::new(gran, 0);
                    let mut at = 0;
                    while at < n {
                        let end = (at + piece).min(n);
                        m.prefill_chunk(0, at, &p.ids[at..end], None).unwrap();
                        acc.piece(m, at, end - at);
                        at = end;
                    }
                    m.cfg.layers = full.clone();
                    let run = format!("{}|n={n}|chunk|c={piece}|trunc", p.name);
                    o.rows(&run, "hid", d - 1, 0, gran, &acc.hashes);
                    o.flush();
                }
                m.end_slot(0);
                eprintln!("{} chunk {piece}: {} depths done", p.name, depths.len());
            }
        }
        m.cfg.layers = full;
        o.flush();
        eprintln!("localize done: {} records; MemAvailable low {:.2} GiB", o.records, gib(guard().low()));
    }
}
