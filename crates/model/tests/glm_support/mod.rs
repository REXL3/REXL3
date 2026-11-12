//! Shared by the GLM model-level GPU tests: the pack's directory (a test prints "not run" and passes where `REXL3_GLM_DIR` is unset), a
//! memory guard for the unified pool (one 121 GiB pool shared by the weights, the activations and the host; an over-commit can hang the
//! box, which has no hardware watchdog), and a byte-for-byte check of what the loader put on the device.
#![allow(dead_code)]

use gb_model::families::glm::load::convert;
use gb_model::families::glm::{LoadPlan, Xform};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub const GIB: u64 = 1 << 30;

/// The prefill chunk rows the model under test is built for: `REXL3_GLM_CHUNK_ROWS`, default 128 (the decode kernels' window, the production default
/// before the large-row path). Above 128 a chunk of more rows runs the large-row kernels.
pub fn chunk_rows() -> usize {
    std::env::var("REXL3_GLM_CHUNK_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(128)
}

/// The pack named by `REXL3_GLM_DIR`, or `None` after printing that the test did not run.
pub fn pack_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_GLM_DIR") {
        Ok(d) if Path::new(&d).join("config.json").exists() => Some(PathBuf::from(d)),
        Ok(d) => panic!("REXL3_GLM_DIR={d} has no config.json"),
        Err(_) => {
            eprintln!("REXL3_GLM_DIR is not set: not run");
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

/// Refuse to start under `need` bytes of `MemAvailable`.
pub fn require_available(need: u64) {
    let a = mem_available();
    assert!(a >= need, "refusing to run: MemAvailable is {:.1} GiB, this test needs {:.0} GiB", gib(a), gib(need));
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
    /// The lowest `MemAvailable` seen.
    pub fn finish(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.take().unwrap().join().unwrap();
        self.min.load(Ordering::Relaxed)
    }
}

/// The whole model, loaded once per test process (a second load would not fit in the pool and can hang the box): the tests of a binary share
/// it through this and must run with `--test-threads=1`. `None` after printing "not run" where `REXL3_GLM_DIR` is unset. Refuses to load under
/// 100 GiB of `MemAvailable` and aborts the process under 5 GiB while it is up.
pub fn shared_model(
    slots: usize,
    kv_blocks: usize,
    max_context: usize,
) -> Option<std::sync::MutexGuard<'static, gb_model::families::glm::model::GlmModel>> {
    use gb_model::families::glm::model::GlmModel;
    use std::sync::{Mutex, OnceLock};
    static M: OnceLock<Option<Mutex<GlmModel>>> = OnceLock::new();
    let m = M.get_or_init(|| load_model(slots, kv_blocks, max_context).map(Mutex::new));
    m.as_ref().map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Load the whole model, owned (for a test that hands it to a backend). Call it once per process, never twice.
pub fn load_model(slots: usize, kv_blocks: usize, max_context: usize) -> Option<gb_model::families::glm::model::GlmModel> {
    use gb_model::families::glm::load::{boot, LoadOptions};
    use gb_model::families::glm::model::{GlmModel, ModelOptions};
    let dir = pack_dir()?;
    require_available(100 * GIB);
    evict_pack(&dir);
    let guard = Guard::start(5 * GIB);
    let t = std::time::Instant::now();
    let b = boot(&dir, None, &LoadOptions::from_env(8 * GIB)).unwrap();
    let load = t.elapsed();
    let cfg = b.pack.config.clone();
    let m =
        GlmModel::new_chunked(cfg, b.device, &ModelOptions { slots, kv_blocks, max_context, layers: 45 }, chunk_rows()).unwrap();
    eprintln!("model ready in {:.1?} (load {:.1?}); MemAvailable {:.1} GiB", t.elapsed(), load, gib(mem_available()));
    // the guard lives for the whole process: leak it (its thread aborts the process under 5 GiB)
    std::mem::forget(guard);
    Some(m)
}

/// The first `layers` trunk layers of the pack (the global region with the embedding and the head, no NextN layer, no tower), loaded once
/// per process like [`shared_model`]: the candidate of a prefix dump, judged against the reference engine's `--trunk-layers N` dump (whose
/// head, like this model's, runs after layer N - 1). A pack whose whole model does not fit one node (sc_4.10bpw, 156 GiB) has only prefix
/// oracles. Refuses `layers` outside 1..(the pack's layers) and refuses to load unless `MemAvailable` covers the prefix's regions plus
/// 24 GiB; aborts the process under 20 GiB (the box's rule) while it is up.
pub fn shared_prefix_model(
    layers: usize,
    slots: usize,
    kv_blocks: usize,
    max_context: usize,
) -> Option<std::sync::MutexGuard<'static, gb_model::families::glm::model::GlmModel>> {
    use gb_model::families::glm::load::{boot_mapped, keep_prefix, LoadOptions};
    use gb_model::families::glm::model::{GlmModel, ModelOptions};
    use gb_model::families::glm::GlmFamily;
    use std::sync::{Mutex, OnceLock};
    static M: OnceLock<Option<Mutex<GlmModel>>> = OnceLock::new();
    let m = M.get_or_init(|| {
        let dir = pack_dir()?;
        let t = std::time::Instant::now();
        gb_cuda::rt::init_context().unwrap();
        let pack = GlmFamily::map_pack(&dir).unwrap();
        let total = pack.config.layers.len();
        assert!(
            (1..total).contains(&layers),
            "a prefix holds 1..{total} trunk layers, not {layers} (the whole model takes no prefix setting)"
        );
        let bytes: u64 = pack.plan.select_regions(keep_prefix(layers)).regions.iter().map(|r| r.bytes).sum();
        require_available(bytes + 24 * GIB);
        evict_pack(&dir);
        let guard = Guard::start(20 * GIB);
        let b = boot_mapped(&dir, pack, Some(layers), &LoadOptions::from_env(20 * GIB), t.elapsed().as_secs_f64()).unwrap();
        let cfg = b.pack.config.clone();
        let m =
            GlmModel::new_chunked(cfg, b.device, &ModelOptions { slots, kv_blocks, max_context, layers }, chunk_rows()).unwrap();
        eprintln!(
            "prefix model ({layers} of {total} trunk layers, regions {:.2} GiB) ready in {:.1?}; MemAvailable {:.1} GiB",
            gib(bytes),
            t.elapsed(),
            gib(mem_available())
        );
        std::mem::forget(guard);
        Some(Mutex::new(m))
    });
    m.as_ref().map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
}

/// The whole model WITH the NextN layer (90.5 GiB), loaded once per process like [`shared_model`]; refuses to load under 105 GiB of `MemAvailable`.
pub fn shared_model_mtp(
    slots: usize,
    kv_blocks: usize,
    max_context: usize,
) -> Option<std::sync::MutexGuard<'static, gb_model::families::glm::model::GlmModel>> {
    use gb_model::families::glm::load::{boot_with, LoadOptions};
    use gb_model::families::glm::model::{GlmModel, ModelOptions};
    use std::sync::{Mutex, OnceLock};
    static M: OnceLock<Option<Mutex<GlmModel>>> = OnceLock::new();
    let m = M.get_or_init(|| {
        let dir = pack_dir()?;
        require_available(105 * GIB);
        evict_pack(&dir);
        let guard = Guard::start(5 * GIB);
        let t = std::time::Instant::now();
        let b = boot_with(&dir, None, true, &LoadOptions::from_env(8 * GIB)).unwrap();
        let load = t.elapsed();
        let cfg = b.pack.config.clone();
        let m = GlmModel::new_chunked(cfg, b.device, &ModelOptions { slots, kv_blocks, max_context, layers: 45 }, chunk_rows())
            .unwrap();
        eprintln!(
            "model + NextN layer ready in {:.1?} (load {:.1?}); MemAvailable {:.1} GiB",
            t.elapsed(),
            load,
            gib(mem_available())
        );
        std::mem::forget(guard);
        Some(Mutex::new(m))
    });
    m.as_ref().map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
}

/// The whole model with the NextN layer, owned (for a test that hands it to a backend). Call it once per process, never twice.
pub fn load_model_mtp(slots: usize, kv_blocks: usize, max_context: usize) -> Option<gb_model::families::glm::model::GlmModel> {
    use gb_model::families::glm::load::{boot_with, LoadOptions};
    use gb_model::families::glm::model::{GlmModel, ModelOptions};
    let dir = pack_dir()?;
    require_available(105 * GIB);
    evict_pack(&dir);
    let guard = Guard::start(5 * GIB);
    let t = std::time::Instant::now();
    let b = boot_with(&dir, None, true, &LoadOptions::from_env(8 * GIB)).unwrap();
    let load = t.elapsed();
    let cfg = b.pack.config.clone();
    let m =
        GlmModel::new_chunked(cfg, b.device, &ModelOptions { slots, kv_blocks, max_context, layers: 45 }, chunk_rows()).unwrap();
    eprintln!(
        "model + NextN layer ready in {:.1?} (load {:.1?}); MemAvailable {:.1} GiB",
        t.elapsed(),
        load,
        gib(mem_available())
    );
    std::mem::forget(guard);
    Some(m)
}

/// A SMALL model for the NextN drafter's tests: the first `layers` trunk layers, the embedding and head, and the NextN layer (about 6 GiB),
/// loaded once per process like [`shared_model`]. The MTP layer needs only the tapped hidden rows the test hands it, so the trunk is not run.
pub fn shared_mtp_model(
    layers: usize,
    slots: usize,
    kv_blocks: usize,
    max_context: usize,
) -> Option<std::sync::MutexGuard<'static, gb_model::families::glm::model::GlmModel>> {
    use gb_model::families::glm::load::{boot_with, LoadOptions};
    use gb_model::families::glm::model::{GlmModel, ModelOptions};
    use std::sync::{Mutex, OnceLock};
    static M: OnceLock<Option<Mutex<GlmModel>>> = OnceLock::new();
    let m = M.get_or_init(|| {
        let dir = pack_dir()?;
        require_available(40 * GIB);
        evict_pack(&dir);
        let guard = Guard::start(8 * GIB);
        let t = std::time::Instant::now();
        let b = boot_with(&dir, Some(layers), true, &LoadOptions::from_env(8 * GIB)).unwrap();
        let cfg = b.pack.config.clone();
        let m =
            GlmModel::new_chunked(cfg, b.device, &ModelOptions { slots, kv_blocks, max_context, layers }, chunk_rows()).unwrap();
        eprintln!(
            "small MTP model ({layers} trunk layers + the NextN layer) ready in {:.1?}; MemAvailable {:.1} GiB",
            t.elapsed(),
            gib(mem_available())
        );
        std::mem::forget(guard);
        Some(Mutex::new(m))
    });
    m.as_ref().map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Drop the pack's pages from the page cache (a cold boot; also keeps `cudaMemGetInfo` honest on this unified-memory box).
pub fn evict_pack(dir: &Path) {
    use std::os::fd::AsRawFd;
    #[cfg(target_os = "linux")]
    extern "C" {
        fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    }
    /// macOS has no `posix_fadvise` (and no way to drop one file's pages): a no-op, so a "cold" read here may be warm.
    #[cfg(not(target_os = "linux"))]
    unsafe fn posix_fadvise(_fd: i32, _offset: i64, _len: i64, _advice: i32) -> i32 {
        0
    }
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        if e.path().extension().is_some_and(|x| x == "safetensors") {
            if let Ok(f) = File::open(e.path()) {
                unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, 4) };
            }
        }
    }
}

/// Compare items of a loaded plan with what the plan says they hold: the shard's bytes through the item's conversion. `picks`: the item
/// indices to check. Returns how many were checked.
pub fn verify_items(
    plan: &LoadPlan,
    dir: &Path,
    device_bytes: impl Fn(&gb_model::families::glm::LoadItem) -> Vec<u8>,
    picks: &[usize],
) -> usize {
    let files: Vec<File> = plan.files.iter().map(|f| File::open(dir.join(f)).unwrap()).collect();
    for &k in picks {
        let it = &plan.items[k];
        let mut src = vec![0u8; it.bytes as usize];
        files[it.file as usize].read_exact_at(&mut src, it.offset).unwrap();
        let want = if it.xform == Xform::None { src } else { convert(it.xform, &src).unwrap() };
        assert_eq!(want.len() as u64, it.dst_bytes, "{}", it.name);
        let got = device_bytes(it);
        assert!(got == want, "{} ({:?}) differs on the device from the shard's bytes through its conversion", it.name, it.xform);
    }
    picks.len()
}

/// The first and last item of every region, then `sample` items evenly spaced.
pub fn pick_items(plan: &LoadPlan, sample: usize) -> Vec<usize> {
    let mut picks: Vec<usize> = Vec::new();
    for r in &plan.regions {
        let idx: Vec<usize> = plan.items.iter().enumerate().filter(|(_, i)| i.region == r.id).map(|(k, _)| k).collect();
        if let (Some(&a), Some(&b)) = (idx.first(), idx.last()) {
            picks.push(a);
            picks.push(b);
        }
    }
    let stride = (plan.items.len() / sample.max(1)).max(1);
    picks.extend((0..plan.items.len()).step_by(stride));
    picks.sort_unstable();
    picks.dedup();
    picks
}

/// The P2 GEMM launches a prompt chunk of `rows` rows makes when every dense linear of every layer takes it: KDA qkv and o, MLA q_a, q_b,
/// kv_a and o, the gated MLPs (3 a piece: the dense MLP of a chunk above 128 rows runs in pieces of `PF_DENSE_ROWS`, a shared expert in one
/// piece of the chunk), and the NextN layer (eh_proj, its MLA's 4, its shared expert's 3).
pub fn p2_expected_launches(m: &gb_model::families::glm::model::GlmModel, rows: usize) -> u64 {
    let dense_layers = m.cfg.dense_layers();
    let (mut n, mut dense, mut moe) = (0u64, 0u64, 0u64);
    for l in 0..m.layers() {
        n += if m.cfg.mla_layers().contains(&l) { 4 } else { 2 };
        if dense_layers.contains(&l) {
            dense += 1;
        } else {
            moe += 1;
        }
    }
    let dense_pieces = if rows > gb_model::families::glm::alloc::MAX_ROWS {
        rows.div_ceil(gb_model::families::glm::alloc::PF_DENSE_ROWS.min(m.chunk_rows()))
    } else {
        1
    } as u64;
    n += dense * 3 * dense_pieces + moe * 3;
    if m.has_mtp() {
        n += 1 + 4 + 3;
    }
    n
}

/// The P2 expert runs a prompt chunk makes when every MoE layer's routed experts take the P2 expert GEMMs: one a MoE layer, and the NextN
/// layer's in its prefill.
pub fn p2_expected_moe_runs(m: &gb_model::families::glm::model::GlmModel) -> u64 {
    let dense_layers = m.cfg.dense_layers();
    (0..m.layers()).filter(|l| !dense_layers.contains(l)).count() as u64 + m.has_mtp() as u64
}

/// The P2 MLA absorb / expand launches a P2 prompt chunk of `rows` rows makes (`gb_cuda::glm_p2_mla`, step (d)): one absorb and one expand
/// a slab of at most `MAX_ROWS` rows in every MLA layer, and in the NextN layer's prefill when it is loaded ([`p2_mla_runs_of_nextn`]).
pub fn p2_expected_mla_runs(m: &gb_model::families::glm::model::GlmModel, rows: usize) -> u64 {
    let mla = m.cfg.mla_layers();
    let trunk = (0..m.layers()).filter(|l| mla.contains(l)).count() as u64;
    p2_mla_runs_of_nextn(rows) * (trunk + m.has_mtp() as u64)
}

/// The P2 MLA absorb / expand launches of one MLA layer (the NextN layer's share of [`p2_expected_mla_runs`]) on `rows` prompt rows.
pub fn p2_mla_runs_of_nextn(rows: usize) -> u64 {
    2 * rows.div_ceil(gb_model::families::glm::alloc::MAX_ROWS).max(1) as u64
}

/// The chunked KDA recurrences a P2 prompt chunk runs (`gb_cuda::kda::prefill_chunked`, one a KDA layer; the NextN layer is MLA).
pub fn p2_expected_kda_runs(m: &gb_model::families::glm::model::GlmModel) -> u64 {
    let mla = m.cfg.mla_layers();
    (0..m.layers()).filter(|l| !mla.contains(l)).count() as u64
}
