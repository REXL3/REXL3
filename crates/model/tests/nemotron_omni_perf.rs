//! Task PERF-STEP: where the time of one Nemotron 3.5 Super VL step goes, and the proof that no change to it costs a bit.
//!
//! One `#[ignore]`d test, env-gated, one model per process, under the device lock on a GB10:
//!
//!     REXL3_NEMO_OMNI_DIR=<pack> [REXL3_NEMO_ORACLE_DIR=<ref-text-1k>] [REXL3_NEMO_PERF_PARTS=steps,layers,digest,trace] \
//!       cargo test --release -p gb-model --test nemotron_omni_perf -- --ignored --nocapture --test-threads=1
//!
//! * `REXL3_NEMO_PERF_LAYERS`: trunk layers to load (default all 88; `8` is the `MEMEMEM*` prefix).
//! * `REXL3_NEMO_PERF_PARTS` (comma list, default `steps,layers`):
//!   - `steps`: host wall time of a verify window of 1, 2, 4, 8, 16 and 64 rows of one sequence at a 256-token context, of a serial
//!     decode step (window of one row + `commit(1)`), and of a 64-row prefill slice: median, min and max of `REXL3_NEMO_PERF_REPS`
//!     (default 7) after 2 warmups, no tap. This is the number `tests/nemotron_omni_harness.rs` calls the step time.
//!   - `layers`: the same shapes with an event tap (a timing CUDA event and a host `Instant` at every tap point of the layer loop,
//!     nothing synchronised inside the step): the GPU time and the host launch time of each part of the step, summed per layer kind.
//!     The host column is the CPU time the part took to launch (a part that is host-bound shows the same time in both columns), the
//!     GPU column is the time the stream took between the two points, kernels and gaps together. `REXL3_NEMO_PERF_PER_LAYER=1` also
//!     prints every layer.
//!   - `digest`: a scripted run (prefill of three sequences, serial steps, verify windows of 2 to 64 rows with partial keeps, a window of
//!     several sequences, windows that cross a 512-key attention chunk, a prefill slice of two sequences), hashing every row's logits and
//!     candidates, the window records, and the Mamba S and conv pools and the KV rows of every touched slot, segment by segment. Lines
//!     `DIGEST <segment> <hex>` go to stdout; `REXL3_DIGEST_OUT=<file>` writes them and `REXL3_DIGEST_EXPECT=<file>` compares every line
//!     with a file an earlier build wrote (the test fails on the first difference and prints the segments that differ).
//!   - `digestlong`: the same kind of script across the flash attention path's threshold at 8,192 positions (prompts of 8,188 and 8,186
//!     tokens, serial steps and windows with rows on both sides of it, a window of one deep and one shallow sequence, a prefill slice of
//!     two deep sequences), loaded with a 9,216-position context. Run it as its own part: `REXL3_DIGEST_OUT` and `REXL3_DIGEST_EXPECT`
//!     name its file (`testdata/gpu/nemotron_step_digest_long.txt`). The graph passes capture the flash launches (chunk buckets 32 and 64).
//!   - `uploads`: the ordering of the per-step uploads (see `uploads`): back-to-back windows of different ids, and windows with commits,
//!     against a reference run that waits for the device around every call, with the ordered uploads and with the old ones.
//!   - `trace`: windows of 1, 4 and 64 rows, a 64-row prefill slice and the serial decode step (window of one row plus its commit),
//!     each in its own `cudaProfilerStart/Stop` range, for `nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat:5`.
//!
//! The inputs are the fork's dump text (`REXL3_NEMO_ORACLE_DIR`, `ids_0.u32` and `ids_1.u32`) when it is set and a xorshift stream
//! otherwise, so the routing of the MoE layers is that of real text in the first case.
#![allow(clippy::arc_with_non_send_sync, clippy::ptr_arg, clippy::too_many_arguments)]

use gb_cuda::{DevBuf, Stream};
use gb_engine::RowRequest;
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::ModelOptions;
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::model::{NemotronModel, Point, Tap};
use gb_model::families::nemotron_h_omni::oracle::Dump;
use gb_model::families::nemotron_h_omni::weights::LayerPlan;
use gb_model::forward::Candidates;
use gb_model::rows::RowBatch;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const SLOTS: usize = 8;
const KV_PAGES: usize = 64;
const MAX_CONTEXT: usize = 4096;
/// The model of the `digestlong` part: three sequences of about 8.3k positions (33 pages each) and a short one.
const LONG_KV_PAGES: usize = 160;
const LONG_CONTEXT: usize = 9216;
/// Candidates per row a window asks for (what the harness and the sampler use).
const COUNT: usize = 28;
const CTX: usize = 256;

// timed events through gb-cuda, so the test links in the stub build too (it then reports itself unrunnable)
use gb_cuda::rt::timed_events::{
    create as cudaEventCreate, elapsed_ms as cudaEventElapsedTime, record as cudaEventRecord, synchronize as cudaEventSynchronize,
};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn parts() -> Vec<String> {
    std::env::var("REXL3_NEMO_PERF_PARTS")
        .unwrap_or_else(|_| "steps,layers".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// A deterministic token stream inside the vocabulary and above the special tokens (xorshift64*).
fn synth(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as u32 % 100_000 + 1000
        })
        .collect()
}

/// The two text sources: the fork's dump ids when the oracle directory is set, else a synthetic stream.
fn texts() -> [Vec<u32>; 2] {
    if let Some(dir) = std::env::var_os("REXL3_NEMO_ORACLE_DIR").map(PathBuf::from) {
        let get =
            |s: usize| Dump::open(&dir, s).and_then(|d| d.ids()).unwrap_or_else(|e| panic!("oracle ids of sample {s}: {e}"));
        return [get(0), get(1)];
    }
    eprintln!("REXL3_NEMO_ORACLE_DIR is not set: synthetic token streams");
    [synth(11, 1024), synth(12, 1024)]
}

fn cycle(src: &[u32], p: usize) -> u32 {
    src[p % src.len()]
}

fn config(dir: &Path) -> OmniConfig {
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| panic!("config.json: {e}"));
    OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap_or_else(|e| panic!("{e}"))
}

fn load(dir: &Path, cfg: &OmniConfig) -> NemotronModel {
    let inv = Inventory::read(dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let layers = env_usize("REXL3_NEMO_PERF_LAYERS", 0);
    // the long-context digest (`digestlong`) needs room for rows above the flash path's threshold at 8,192
    let long = parts().iter().any(|p| p == "digestlong");
    let opts = ModelOptions {
        slots: SLOTS,
        kv_pages: if long { LONG_KV_PAGES } else { KV_PAGES },
        max_context: if long { LONG_CONTEXT } else { MAX_CONTEXT },
        // 256: the widest slice the model takes (the large-row MoE prefill path above 64 rows). The digest's own slices are 64 rows and
        // its windows at most 64, so they take the decode path whatever the option; the `wide` part runs the 256-row slices.
        // 512 by default since 2026-10-15 (256 under the opt-out REXL3_NEMO_PREFILL_ROWS_MAX=256; lane S)
        prefill_rows: gb_model::families::nemotron_h_omni::model::prefill_rows_cap(),
        layers: if layers == 0 { None } else { Some(layers) },
        // the head is not part of the trunk step: loaded only when asked (it adds 5.9 GB and the head's taps to the window)
        mtp: std::env::var_os("REXL3_NEMO_PERF_MTP").is_some(),
    };
    let t = Instant::now();
    let m = NemotronModel::load(dir, cfg, &plan, opts).unwrap_or_else(|e| panic!("load: {e}"));
    eprintln!("loaded in {:.1} s", t.elapsed().as_secs_f64());
    m
}

// ------------------------------------------------------------------------------------------------------------------ the event tap

struct Ev(*mut c_void);
unsafe impl Send for Ev {}

/// One tap point of a step: where, which event (index into the pool) and when the host reached it.
#[derive(Clone, Copy)]
struct Stamp {
    point: Point,
    ev: usize,
    host: Instant,
}

#[derive(Default)]
struct Stamps {
    list: Vec<Stamp>,
}

/// A timing event at every tap point. The pool is created once; `list` is cleared before each step. The first event of the pool is
/// the step's start (recorded by the caller) and the last its end.
#[derive(Clone)]
struct EventTap {
    pool: Arc<Vec<Ev>>,
    stamps: Arc<Mutex<Stamps>>,
}

impl EventTap {
    fn new() -> Self {
        let pool = (0..520)
            .map(|_| {
                let mut e = std::ptr::null_mut();
                let rc = unsafe { cudaEventCreate(&mut e) };
                assert_eq!(rc, 0, "cudaEventCreate: {rc}");
                Ev(e)
            })
            .collect();
        Self { pool: Arc::new(pool), stamps: Arc::new(Mutex::new(Stamps::default())) }
    }

    fn start(&self) -> *mut c_void {
        self.pool[0].0
    }

    fn end(&self) -> *mut c_void {
        self.pool[self.pool.len() - 1].0
    }

    fn reset(&self) {
        self.stamps.lock().unwrap().list.clear();
    }

    fn record(&self, ev: *mut c_void, st: &Stream) {
        let rc = unsafe { cudaEventRecord(ev, st.raw()) };
        assert_eq!(rc, 0, "cudaEventRecord: {rc}");
    }

    fn ms(&self, a: *mut c_void, b: *mut c_void) -> f64 {
        let mut ms = 0f32;
        let rc = unsafe { cudaEventElapsedTime(&mut ms, a, b) };
        assert_eq!(rc, 0, "cudaEventElapsedTime: {rc}");
        ms as f64
    }
}

impl Tap for EventTap {
    fn tap(&mut self, point: Point, _buf: &DevBuf, _rows: usize, _width: usize, st: &Stream) -> Result<(), ModelError> {
        let mut s = self.stamps.lock().unwrap();
        let ev = s.list.len() + 1;
        assert!(ev + 1 < self.pool.len(), "more tap points than events");
        self.record(self.pool[ev].0, st);
        s.list.push(Stamp { point, ev, host: Instant::now() });
        Ok(())
    }
}

/// What one step's parts cost: GPU milliseconds (the stream's time between two points) and host milliseconds (the host's time between
/// the two points). With the GPU saturated the host column is the time the launches spent blocked on a full launch queue behind the
/// previous layers, not the cost of the part: read the GPU column.
#[derive(Clone, Copy, Default)]
struct Cost {
    gpu: f64,
    host: f64,
}

const PARTS: [&str; 9] =
    ["setup", "mamba", "attn", "moe", "residual add", "final norm", "lm_head", "top-k + sync + download", "TOTAL"];

/// The categories of one tapped step. `t0` is the host time before the call and `t_end` after it; the GPU times are measured from the
/// start event to the end event, recorded by the caller on either side of the call.
fn breakdown(
    tap: &EventTap,
    pattern: &[char],
    t0: Instant,
    t_end: Instant,
    per_layer: &mut Vec<(usize, char, Cost, Cost)>,
) -> BTreeMap<&'static str, Cost> {
    let stamps = tap.stamps.lock().unwrap().list.clone();
    let mut out: BTreeMap<&'static str, Cost> = BTreeMap::new();
    let mut add = |k: &'static str, g: f64, h: f64| {
        let c = out.entry(k).or_default();
        c.gpu += g;
        c.host += h;
    };
    per_layer.clear();
    let (mut pg, mut ph) = (0.0f64, 0.0f64);
    let mut arm: Option<(f64, f64)> = None;
    for s in &stamps {
        let g = tap.ms(tap.start(), tap.pool[s.ev].0);
        let h = s.host.duration_since(t0).as_secs_f64() * 1e3;
        let (dg, dh) = (g - pg, h - ph);
        match s.point {
            Point::Embed => add("setup", dg, dh),
            Point::Mixer(li) => {
                let k = match pattern.get(li) {
                    Some('M') => "mamba",
                    Some('*') => "attn",
                    _ => "moe",
                };
                add(k, dg, dh);
                arm = Some((dg, dh));
            }
            Point::Layer(li) => {
                add("residual add", dg, dh);
                let (ag, ah) = arm.take().unwrap_or((0.0, 0.0));
                per_layer.push((
                    li,
                    pattern.get(li).copied().unwrap_or('?'),
                    Cost { gpu: ag, host: ah },
                    Cost { gpu: dg, host: dh },
                ));
            }
            Point::FinalNorm => add("final norm", dg, dh),
            Point::Logits => add("lm_head", dg, dh),
        }
        pg = g;
        ph = h;
    }
    let g_end = tap.ms(tap.start(), tap.end());
    let h_end = t_end.duration_since(t0).as_secs_f64() * 1e3;
    add("top-k + sync + download", g_end - pg, h_end - ph);
    add("TOTAL", g_end, h_end);
    out
}

// ------------------------------------------------------------------------------------------------------------------ steps

fn window_batch(slot: usize, first: usize, src: &[u32], rows: usize) -> RowBatch {
    let toks: Vec<u32> = (0..rows).map(|j| cycle(src, first + j)).collect();
    RowBatch::from_requests(&[RowRequest { slot, tokens: toks, first_position: first as u64 }])
}

fn prefill_ctx(m: &mut NemotronModel, slot: usize, src: &[u32], n: usize) {
    m.begin_slot(slot).unwrap_or_else(|e| panic!("begin_slot: {e}"));
    let mut at = 0;
    while at < n {
        let end = (at + 64).min(n);
        let toks: Vec<u32> = (at..end).map(|q| cycle(src, q)).collect();
        let b = RowBatch::from_requests(&[RowRequest { slot, tokens: toks, first_position: at as u64 }]);
        m.prefill_chunk(&b, None).unwrap_or_else(|e| panic!("prefill: {e}"));
        at = end;
    }
    m.flush().unwrap();
}

fn reset(m: &mut NemotronModel) {
    for s in 0..SLOTS {
        m.end_slot(s);
    }
}

fn row_line(name: &str, ms: &mut Vec<f64>) {
    let lo = ms.iter().cloned().fold(f64::MAX, f64::min);
    let hi = ms.iter().cloned().fold(0.0, f64::max);
    let med = median(ms);
    eprintln!("STEP {name:<22} median {med:9.1} ms   min {lo:9.1}   max {hi:9.1}");
}

fn steps(m: &mut NemotronModel, text: &[Vec<u32>; 2]) {
    let reps = env_usize("REXL3_NEMO_PERF_REPS", 7);
    reset(m);
    prefill_ctx(m, 0, &text[0], CTX);
    for rows in [1usize, 2, 4, 8, 16, 64] {
        let mut ms = Vec::new();
        for it in 0..2 + reps {
            let b = window_batch(0, CTX, &text[1], rows);
            let t = Instant::now();
            let c = m.window_candidates(&b, COUNT).unwrap_or_else(|e| panic!("window {rows}: {e}"));
            let dt = t.elapsed().as_secs_f64() * 1e3;
            assert_eq!(c.rows, rows);
            if it >= 2 {
                ms.push(dt);
            }
        }
        row_line(&format!("window {rows:>2} rows"), &mut ms);
    }
    // a serial decode step: window of one row, commit, the next window follows at once
    {
        let mut ms = Vec::new();
        let mut p = CTX;
        for it in 0..2 + reps {
            let b = window_batch(0, p, &text[1], 1);
            let t = Instant::now();
            m.window_candidates(&b, COUNT).unwrap();
            m.commit(0, 1);
            let dt = t.elapsed().as_secs_f64() * 1e3;
            p += 1;
            if it >= 2 {
                ms.push(dt);
            }
        }
        m.flush().unwrap();
        row_line("decode (win+commit)", &mut ms);
    }
    // a 64-row prefill slice, the logits of its last row
    {
        reset(m);
        prefill_ctx(m, 1, &text[0], CTX);
        let mut ms = Vec::new();
        let mut p = CTX;
        for it in 0..2 + reps {
            let b = window_batch(1, p, &text[1], 64);
            let t = Instant::now();
            let c = m.prefill_chunk(&b, Some(63)).unwrap_or_else(|e| panic!("prefill slice: {e}"));
            let dt = t.elapsed().as_secs_f64() * 1e3;
            assert!(c.is_some());
            p += 64;
            if it >= 2 {
                ms.push(dt);
            }
        }
        row_line("prefill slice 64", &mut ms);
    }
    reset(m);
}

fn print_parts(title: &str, runs: &[BTreeMap<&'static str, Cost>]) {
    eprintln!("LAYERS {title}");
    eprintln!("LAYERS {:<26} {:>10} {:>10}", "part", "gpu ms", "host ms");
    for k in PARTS {
        let mut g: Vec<f64> = runs.iter().map(|r| r.get(k).map_or(0.0, |c| c.gpu)).collect();
        let mut h: Vec<f64> = runs.iter().map(|r| r.get(k).map_or(0.0, |c| c.host)).collect();
        eprintln!("LAYERS {k:<26} {:>10.2} {:>10.2}", median(&mut g), median(&mut h));
    }
}

fn layers(m: &mut NemotronModel, cfg: &OmniConfig, text: &[Vec<u32>; 2]) {
    let reps = env_usize("REXL3_NEMO_PERF_REPS", 7).max(3);
    let per_layer_on = std::env::var("REXL3_NEMO_PERF_PER_LAYER").is_ok_and(|v| v == "1");
    let pattern: Vec<char> = cfg.pattern.chars().collect();
    let tap = EventTap::new();
    reset(m);
    prefill_ctx(m, 0, &text[0], CTX);
    m.set_tap(Some(Box::new(tap.clone())));
    for (name, rows, prefill) in [
        ("window 1 row", 1usize, false),
        ("window 4 rows", 4, false),
        ("window 64 rows", 64, false),
        ("prefill slice 64", 64, true),
    ] {
        if prefill {
            m.set_tap(None);
            reset(m);
            prefill_ctx(m, 1, &text[0], CTX);
            m.set_tap(Some(Box::new(tap.clone())));
        }
        let slot = usize::from(prefill);
        let mut runs = Vec::new();
        let mut last_layers = Vec::new();
        let mut p = CTX;
        for it in 0..2 + reps {
            let b = window_batch(slot, p, &text[1], rows);
            tap.reset();
            tap.record(tap.start(), &m.core.stream);
            let t0 = Instant::now();
            if prefill {
                m.prefill_chunk(&b, Some(rows - 1)).unwrap();
                p += rows;
            } else {
                m.window_candidates(&b, COUNT).unwrap();
            }
            let t_end = Instant::now();
            tap.record(tap.end(), &m.core.stream);
            let rc = unsafe { cudaEventSynchronize(tap.end()) };
            assert_eq!(rc, 0);
            let mut pl = Vec::new();
            let r = breakdown(&tap, &pattern, t0, t_end, &mut pl);
            if it >= 2 {
                runs.push(r);
                last_layers = pl;
            }
        }
        print_parts(name, &runs);
        if per_layer_on {
            for (li, kind, arm, add) in &last_layers {
                eprintln!(
                    "LAYER {name:<18} {li:>3} {kind}  arm gpu {:8.3} host {:8.3}   add gpu {:7.3} host {:7.3}",
                    arm.gpu, arm.host, add.gpu, add.host
                );
            }
        }
    }
    m.set_tap(None);
    reset(m);
}

/// The prefill rate: a 4,096-token prompt through `prefill_chunk` in slices of 64, 128 and 256 rows (the last slice asks for candidates),
/// wall time of the whole prompt (best of 3 after a warm-up), and the cost of one 256-row slice by kind of layer from the event tap, at a
/// 2,048-token context. The host does nothing but launch: the GPU is the wall time.
fn rates(m: &mut NemotronModel, cfg: &OmniConfig, text: &[Vec<u32>; 2]) {
    let n = 4096usize;
    eprintln!("prefill rate, {n}-token prompt, one sequence (best of 3 after a warm-up):");
    eprintln!("  rows/slice   slices        ms    tok/s");
    for width in [64usize, 128, 256] {
        let mut best = f64::MAX;
        for rep in 0..4 {
            reset(m);
            m.begin_slot(0).unwrap();
            let t = Instant::now();
            let mut at = 0;
            while at < n {
                let end = (at + width).min(n);
                let toks: Vec<u32> = (at..end).map(|q| cycle(&text[0], q)).collect();
                let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: toks, first_position: at as u64 }]);
                let want = (end == n).then_some(end - at - 1);
                m.prefill_chunk(&b, want).unwrap();
                at = end;
            }
            m.flush().unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if rep > 0 {
                best = best.min(ms);
            }
        }
        eprintln!("  {width:>10} {:>8} {best:>9.0} {:>8.0}", n.div_ceil(width), n as f64 / best * 1e3);
    }
    // one 256-row slice by kind, tapped
    let pattern: Vec<char> = cfg.pattern.chars().collect();
    let tap = EventTap::new();
    reset(m);
    prefill_ctx(m, 0, &text[0], 2048);
    m.set_tap(Some(Box::new(tap.clone())));
    let mut runs = Vec::new();
    let mut p = 2048usize;
    for it in 0..5 {
        let b = window_batch(0, p, &text[1], 256);
        tap.reset();
        tap.record(tap.start(), &m.core.stream);
        let t0 = Instant::now();
        m.prefill_chunk(&b, Some(255)).unwrap();
        let t_end = Instant::now();
        tap.record(tap.end(), &m.core.stream);
        let rc = unsafe { cudaEventSynchronize(tap.end()) };
        assert_eq!(rc, 0);
        let mut pl = Vec::new();
        let r = breakdown(&tap, &pattern, t0, t_end, &mut pl);
        if it >= 2 {
            runs.push(r);
        }
        p += 256;
        if p + 256 > MAX_CONTEXT {
            break;
        }
    }
    print_parts("prefill slice 256 (context 2,048 to 3,000)", &runs);
    m.set_tap(None);
    reset(m);
}

// ------------------------------------------------------------------------------------------------------------------ the digest

/// A 64-bit mix over the bytes fed to it (8 at a time). Not a cryptographic hash: a change detector.
#[derive(Clone, Copy)]
struct Dg(u64);

impl Dg {
    fn new() -> Self {
        Dg(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        let mut h = self.0;
        let mut it = b.chunks_exact(8);
        for c in &mut it {
            let w = u64::from_le_bytes(c.try_into().unwrap());
            h = (h ^ w).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            h ^= h >> 29;
        }
        for &x in it.remainder() {
            h = (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = h ^ (b.len() as u64);
    }
    fn f32s(&mut self, v: &[f32]) {
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
        self.bytes(b);
    }
    fn u16s(&mut self, v: &[u16]) {
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
        self.bytes(b);
    }
    fn i32s(&mut self, v: &[i32]) {
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
        self.bytes(b);
    }
}

struct Digest {
    lines: Vec<(String, u64)>,
    /// Print every line (the run whose lines are the reference).
    print: bool,
}

impl Digest {
    fn emit(&mut self, name: String, d: Dg) {
        if self.print {
            println!("DIGEST {name} {:016x}", d.0);
        }
        self.lines.push((name, d.0));
    }
}

fn hash_logits(m: &NemotronModel, rows: usize) -> Dg {
    let vocab = m.vocab_size();
    let mut v = vec![0f32; rows * vocab];
    m.core.bufs.logits.download_range(0, &mut v).unwrap();
    let mut d = Dg::new();
    d.f32s(&v);
    d
}

fn hash_cands(c: &Candidates) -> Dg {
    let mut d = Dg::new();
    d.i32s(&c.ids);
    d.f32s(&c.vals);
    d
}

/// The S and conv pools of `slot` in every Mamba layer.
fn hash_state(m: &mut NemotronModel, slot: usize) -> Dg {
    let mut d = Dg::new();
    for ord in 0..m.core.mamba.len() {
        let (s, conv) = m.debug_mamba_state(ord, slot).unwrap();
        d.f32s(&s);
        d.u16s(&conv);
    }
    d
}

/// The fp16 K and V rows `first..first + n` of `slot` in every attention layer.
fn hash_kv(m: &mut NemotronModel, slot: usize, first: usize, n: usize) -> Dg {
    let mut d = Dg::new();
    for ord in 0..m.core.kv_geom.layers {
        let (k, v) = m.debug_kv_rows(ord, slot, first, n).unwrap();
        d.u16s(&k);
        d.u16s(&v);
    }
    d
}

/// The first `rows` rows of every Mamba layer's window record and retained xBC rows (what a commit replays).
fn hash_records(m: &NemotronModel, rows: usize) -> Dg {
    use gb_cuda::nemotron_ssm::{CONV_DIM, HEADS};
    let mut d = Dg::new();
    let c = &m.core;
    for ord in 0..c.mamba.len() {
        let r = &c.win_rec[ord];
        let mut xbc = vec![0u16; rows * CONV_DIM];
        let mut dt = vec![0u16; rows * HEADS];
        let mut g = vec![0f32; rows * HEADS];
        let mut decay = vec![0f32; rows * HEADS];
        let mut xin = vec![0f32; rows * CONV_DIM];
        r.xbc.download_range(0, &mut xbc).unwrap();
        r.dt.download_range(0, &mut dt).unwrap();
        r.g.download_range(0, &mut g).unwrap();
        r.decay.download_range(0, &mut decay).unwrap();
        c.win_xbc[ord].download_range(0, &mut xin).unwrap();
        d.u16s(&xbc);
        d.u16s(&dt);
        d.f32s(&g);
        d.f32s(&decay);
        d.f32s(&xin);
    }
    d
}

/// One sequence of a scripted window.
struct Part<'a> {
    slot: usize,
    first: usize,
    src: &'a [u32],
    off: usize,
    rows: usize,
    keep: usize,
}

fn part_tokens(p: &Part) -> Vec<u32> {
    (0..p.rows).map(|j| cycle(p.src, p.off + p.first + j)).collect()
}

/// A verify window of several sequences, then the commits; digests the logits of every row, the candidates, the window records, and
/// after the commits the state and the KV rows of each sequence.
fn window_seg(m: &mut NemotronModel, dg: &mut Digest, name: &str, parts: &[Part]) {
    let reqs: Vec<RowRequest> =
        parts.iter().map(|p| RowRequest { slot: p.slot, tokens: part_tokens(p), first_position: p.first as u64 }).collect();
    let b = RowBatch::from_requests(&reqs);
    let c = m.window_candidates(&b, COUNT).unwrap_or_else(|e| panic!("{name}: {e}"));
    dg.emit(format!("{name}.logits"), hash_logits(m, b.rows));
    dg.emit(format!("{name}.cands"), hash_cands(&c));
    dg.emit(format!("{name}.records"), hash_records(m, b.rows));
    for p in parts {
        m.commit(p.slot, p.keep);
    }
    m.flush().unwrap_or_else(|e| panic!("{name}: {e}"));
    for p in parts {
        dg.emit(format!("{name}.state.s{}", p.slot), hash_state(m, p.slot));
        dg.emit(format!("{name}.kv.s{}", p.slot), hash_kv(m, p.slot, p.first, p.rows));
    }
}

/// A prompt prefilled in slices of 64 rows (the last slice asks for candidates); digests the last row's logits and candidates, and the
/// state and KV rows at the end. `mid` also hashes the state after the slice that ends at that position.
fn prefill_seg(
    m: &mut NemotronModel,
    dg: &mut Digest,
    name: &str,
    slot: usize,
    src: &[u32],
    off: usize,
    n: usize,
    mid: Option<usize>,
) {
    prefill_seg_w(m, dg, name, slot, src, off, n, mid, 64)
}

/// [`prefill_seg`] in slices of `width` rows.
#[allow(clippy::too_many_arguments)]
fn prefill_seg_w(
    m: &mut NemotronModel,
    dg: &mut Digest,
    name: &str,
    slot: usize,
    src: &[u32],
    off: usize,
    n: usize,
    mid: Option<usize>,
    width: usize,
) {
    m.begin_slot(slot).unwrap_or_else(|e| panic!("{name}: begin_slot: {e}"));
    let mut at = 0;
    while at < n {
        let end = (at + width).min(n);
        let toks: Vec<u32> = (at..end).map(|p| cycle(src, off + p)).collect();
        let b = RowBatch::from_requests(&[RowRequest { slot, tokens: toks, first_position: at as u64 }]);
        let want = (end == n).then_some(end - at - 1);
        let c = m.prefill_chunk(&b, want).unwrap_or_else(|e| panic!("{name}: {e}"));
        if let Some(c) = c {
            m.flush().unwrap();
            dg.emit(format!("{name}.cands"), hash_cands(&c));
            dg.emit(format!("{name}.logits"), hash_logits(m, 1));
        }
        if mid == Some(end) {
            m.flush().unwrap();
            dg.emit(format!("{name}.state@{end}"), hash_state(m, slot));
        }
        at = end;
    }
    m.flush().unwrap_or_else(|e| panic!("{name}: {e}"));
    dg.emit(format!("{name}.state"), hash_state(m, slot));
    dg.emit(format!("{name}.kv"), hash_kv(m, slot, 0, n));
}

/// The scripted run, once. See the header of this file for what it covers.
fn digest_script(m: &mut NemotronModel, text: &[Vec<u32>; 2], print: bool) -> Vec<(String, u64)> {
    reset(m);
    let mut dg = Digest { lines: Vec::new(), print };
    let (a, b, c) = (0usize, 1usize, 2usize);
    let sa: &[u32] = &text[0];
    let sb: &[u32] = &text[1];
    // positions committed so far, per slot
    let mut pa = 1020usize;
    let mut pb = 505usize;
    let mut pc = 70usize;

    // 1. prompts: A crosses the 512-key chunk inside its prefill and ends 4 keys short of 1024; B ends 7 keys short of 512; C is short
    prefill_seg(m, &mut dg, "pA", a, sa, 0, pa, Some(512));
    prefill_seg(m, &mut dg, "pB", b, sb, 0, pb, None);
    prefill_seg(m, &mut dg, "pC", c, sb, 400, pc, None);

    // 2. serial decode steps on A, across the 1024-key chunk edge (positions 1020 .. 1025)
    for i in 0..5 {
        window_seg(m, &mut dg, &format!("serial{i}"), &[Part { slot: a, first: pa, src: sa, off: 0, rows: 1, keep: 1 }]);
        pa += 1;
    }

    // 3. windows of 2 to 9 rows with partial keeps
    for (i, (rows, keep)) in [(2usize, 1usize), (3, 3), (5, 2), (8, 8), (9, 4)].into_iter().enumerate() {
        window_seg(m, &mut dg, &format!("win{i}_{rows}"), &[Part { slot: a, first: pa, src: sa, off: 0, rows, keep }]);
        pa += keep;
    }

    // 4. B across its 512-key edge: the rows before the edge read one chunk and the rows after it two
    for (i, (rows, keep)) in [(9usize, 6usize), (3, 2), (4, 4)].into_iter().enumerate() {
        window_seg(m, &mut dg, &format!("edgeB{i}_{rows}"), &[Part { slot: b, first: pb, src: sb, off: 0, rows, keep }]);
        pb += keep;
    }

    // 5. three sequences in one window, in an order that is not the slot order
    window_seg(
        m,
        &mut dg,
        "multi3",
        &[
            Part { slot: c, first: pc, src: sb, off: 400, rows: 3, keep: 1 },
            Part { slot: a, first: pa, src: sa, off: 0, rows: 4, keep: 3 },
            Part { slot: b, first: pb, src: sb, off: 0, rows: 2, keep: 2 },
        ],
    );
    pc += 1;
    pa += 3;
    pb += 2;

    // 6. the widest windows: 64 rows of one sequence, a serial step, then 64 rows of two sequences
    window_seg(m, &mut dg, "w64_a", &[Part { slot: a, first: pa, src: sa, off: 0, rows: 64, keep: 37 }]);
    pa += 37;
    window_seg(m, &mut dg, "serial_after64", &[Part { slot: a, first: pa, src: sa, off: 0, rows: 1, keep: 1 }]);
    pa += 1;
    window_seg(
        m,
        &mut dg,
        "w64_ab",
        &[
            Part { slot: a, first: pa, src: sa, off: 0, rows: 32, keep: 32 },
            Part { slot: b, first: pb, src: sb, off: 0, rows: 32, keep: 5 },
        ],
    );
    pa += 32;

    // 7. a prefill slice of two sequences in one batch (the slice of a batch of fresh prompts)
    {
        m.begin_slot(3).unwrap();
        m.begin_slot(4).unwrap();
        let t3: Vec<u32> = (0..40).map(|p| cycle(sa, 100 + p)).collect();
        let t4: Vec<u32> = (0..24).map(|p| cycle(sb, 700 + p)).collect();
        let bt = RowBatch::from_requests(&[
            RowRequest { slot: 3, tokens: t3, first_position: 0 },
            RowRequest { slot: 4, tokens: t4, first_position: 0 },
        ]);
        let cands = m.prefill_chunk(&bt, Some(63)).unwrap().expect("candidates");
        m.flush().unwrap();
        dg.emit("pM.cands".into(), hash_cands(&cands));
        dg.emit("pM.logits".into(), hash_logits(m, 1));
        dg.emit("pM.state.s3".into(), hash_state(m, 3));
        dg.emit("pM.state.s4".into(), hash_state(m, 4));
        dg.emit("pM.kv.s3".into(), hash_kv(m, 3, 0, 40));
        dg.emit("pM.kv.s4".into(), hash_kv(m, 4, 0, 24));
    }

    // 8. a window over the slots of the slice and a long sequence together
    window_seg(
        m,
        &mut dg,
        "multi4",
        &[
            Part { slot: 3, first: 40, src: sa, off: 100, rows: 2, keep: 2 },
            Part { slot: 4, first: 24, src: sb, off: 700, rows: 1, keep: 1 },
            Part { slot: a, first: pa, src: sa, off: 0, rows: 1, keep: 1 },
            Part { slot: c, first: pc, src: sb, off: 400, rows: 6, keep: 6 },
        ],
    );

    reset(m);
    dg.lines
}

/// Prefill slices of 128 and 256 rows (and an odd 100), which run the MoE layers through the large-row prefill path, against slices of 64
/// (the decode path): the last row's candidates and logits, the S and conv pools and the KV rows of the whole prompt must be the same
/// bits, for prompts that end before, on and past the 512-key chunk edge, for a slice that holds two sequences, and for slices that
/// start mid-prompt. A test, not a golden: the 64-row digest above is the golden and these must equal it.
fn wide_slices(m: &mut NemotronModel, text: &[Vec<u32>; 2]) {
    let t = Instant::now();
    let (sa, sb): (&[u32], &[u32]) = (&text[0], &text[1]);
    let mut cases = 0usize;
    for (name, src, off, n) in
        [("pA", sa, 0usize, 1020usize), ("pB", sb, 0, 505), ("pC", sb, 400, 300), ("pD", sa, 700, 257), ("pE", sb, 0, 65)]
    {
        let mut want = Digest { lines: Vec::new(), print: false };
        reset(m);
        prefill_seg_w(m, &mut want, name, 0, src, off, n, None, 64);
        for width in [128usize, 256, 100] {
            let mut got = Digest { lines: Vec::new(), print: false };
            reset(m);
            prefill_seg_w(m, &mut got, name, 0, src, off, n, None, width);
            assert_eq!(got.lines.len(), want.lines.len(), "{name} at {width}: the segments emit different lines");
            for ((gn, gh), (wn, wh)) in got.lines.iter().zip(&want.lines) {
                assert_eq!(gn, wn);
                assert_eq!(gh, wh, "{name}, slices of {width} rows: {gn} differs from the slices of 64");
            }
            cases += 1;
        }
    }
    // two sequences in one slice (150 + 90 rows) against each sequence alone in slices of 64
    reset(m);
    let mut want = Digest { lines: Vec::new(), print: false };
    prefill_seg_w(m, &mut want, "s0", 0, sa, 100, 150, None, 64);
    prefill_seg_w(m, &mut want, "s1", 1, sb, 700, 90, None, 64);
    reset(m);
    m.begin_slot(0).unwrap();
    m.begin_slot(1).unwrap();
    let t0: Vec<u32> = (0..150).map(|p| cycle(sa, 100 + p)).collect();
    let t1: Vec<u32> = (0..90).map(|p| cycle(sb, 700 + p)).collect();
    let bt = RowBatch::from_requests(&[
        RowRequest { slot: 0, tokens: t0, first_position: 0 },
        RowRequest { slot: 1, tokens: t1, first_position: 0 },
    ]);
    let cands = m.prefill_chunk(&bt, Some(239)).unwrap().expect("candidates");
    m.flush().unwrap();
    // the wanted row is the last row of slot 1: the same as slot 1's last row alone
    let mut got = Digest { lines: Vec::new(), print: false };
    got.emit("s1.cands".into(), hash_cands(&cands));
    got.emit("s1.logits".into(), hash_logits(m, 1));
    let find = |d: &Digest, n: &str| d.lines.iter().find(|(k, _)| k == n).map(|(_, h)| *h).unwrap();
    assert_eq!(
        find(&got, "s1.cands"),
        find(&want, "s1.cands"),
        "two sequences in one 240-row slice: the last row's candidates differ"
    );
    assert_eq!(
        find(&got, "s1.logits"),
        find(&want, "s1.logits"),
        "two sequences in one 240-row slice: the last row's logits differ"
    );
    for (slot, name, n) in [(0usize, "s0", 150usize), (1, "s1", 90)] {
        assert_eq!(
            hash_state(m, slot).0,
            find(&want, &format!("{name}.state")),
            "{name}: the state after the shared slice differs"
        );
        assert_eq!(
            hash_kv(m, slot, 0, n).0,
            find(&want, &format!("{name}.kv")),
            "{name}: the KV rows after the shared slice differ"
        );
    }
    cases += 1;
    reset(m);
    eprintln!("wide slices: {cases} cases (slices of 128, 256 and 100 rows, and two sequences in one slice) equal the slices of 64, {:.1} s", t.elapsed().as_secs_f64());
}

/// The long-context script: the same segments as [`digest_script`] where the position rule matters, across the flash path's threshold at
/// 8,192 (`nemotron_attn_prefill::FLASH_FROM_POS`): prompts of 8,188 and 8,186 tokens, serial steps whose rows are on both sides of it,
/// windows of 2 to 9 rows with partial keeps (rows below and above the threshold in one window), a window of two sequences one of which
/// is deep and one shallow, the 64-row windows, and a prefill slice of two sequences continuing above the threshold. Under graphs the
/// shapes cross the chunk buckets 16 -> 32 (the flash launches appear) and 32 -> 64.
fn digest_long_script(m: &mut NemotronModel, text: &[Vec<u32>; 2], print: bool) -> Vec<(String, u64)> {
    reset(m);
    let mut dg = Digest { lines: Vec::new(), print };
    let (a, b, c) = (0usize, 1usize, 2usize);
    let sa: &[u32] = &text[0];
    let sb: &[u32] = &text[1];
    let mut pa = 8188usize;
    let mut pb = 8186usize;
    let mut pc = 70usize;

    // 1. prompts: A ends 4 keys short of the threshold, B 6 short of it, C is short
    prefill_seg(m, &mut dg, "lpA", a, sa, 0, pa, Some(8128));
    prefill_seg(m, &mut dg, "lpB", b, sb, 0, pb, None);
    prefill_seg(m, &mut dg, "lpC", c, sb, 400, pc, None);

    // 2. serial decode steps on A across the threshold (positions 8188 .. 8195: 4 rows of the tree, 4 of the flash kernel)
    for i in 0..8 {
        window_seg(m, &mut dg, &format!("lserial{i}"), &[Part { slot: a, first: pa, src: sa, off: 0, rows: 1, keep: 1 }]);
        pa += 1;
    }

    // 3. windows of 2 to 9 rows with partial keeps, all above the threshold
    for (i, (rows, keep)) in [(2usize, 1usize), (3, 3), (5, 2), (8, 8), (9, 4)].into_iter().enumerate() {
        window_seg(m, &mut dg, &format!("lwin{i}_{rows}"), &[Part { slot: a, first: pa, src: sa, off: 0, rows, keep }]);
        pa += keep;
    }

    // 4. B across the threshold inside windows: 4 rows of the tree then 5 rows with the threshold in the middle, then a window above
    for (i, (rows, keep)) in [(4usize, 4usize), (9, 6), (3, 3)].into_iter().enumerate() {
        window_seg(m, &mut dg, &format!("ledgeB{i}_{rows}"), &[Part { slot: b, first: pb, src: sb, off: 0, rows, keep }]);
        pb += keep;
    }

    // 5. three sequences in one window, in an order that is not the slot order: one shallow, two deep
    window_seg(
        m,
        &mut dg,
        "lmulti3",
        &[
            Part { slot: c, first: pc, src: sb, off: 400, rows: 3, keep: 1 },
            Part { slot: a, first: pa, src: sa, off: 0, rows: 4, keep: 3 },
            Part { slot: b, first: pb, src: sb, off: 0, rows: 2, keep: 2 },
        ],
    );
    pc += 1;
    pa += 3;
    pb += 2;

    // 6. the widest windows: 64 rows of one deep sequence, a serial step, then 64 rows of two sequences (one deep, one that crosses)
    window_seg(m, &mut dg, "lw64_a", &[Part { slot: a, first: pa, src: sa, off: 0, rows: 64, keep: 37 }]);
    pa += 37;
    window_seg(m, &mut dg, "lserial_after64", &[Part { slot: a, first: pa, src: sa, off: 0, rows: 1, keep: 1 }]);
    pa += 1;
    window_seg(
        m,
        &mut dg,
        "lw64_ab",
        &[
            Part { slot: a, first: pa, src: sa, off: 0, rows: 32, keep: 32 },
            Part { slot: b, first: pb, src: sb, off: 0, rows: 32, keep: 5 },
        ],
    );
    pa += 32;
    pb += 5;

    // 7. a prefill slice of two sequences continuing above the threshold (a slice of a batch of two prompts that were being prefilled)
    {
        let ta: Vec<u32> = (0..40).map(|p| cycle(sa, pa + p)).collect();
        let tb: Vec<u32> = (0..24).map(|p| cycle(sb, pb + p)).collect();
        let bt = RowBatch::from_requests(&[
            RowRequest { slot: a, tokens: ta, first_position: pa as u64 },
            RowRequest { slot: b, tokens: tb, first_position: pb as u64 },
        ]);
        let cands = m.prefill_chunk(&bt, Some(63)).unwrap().expect("candidates");
        m.flush().unwrap();
        dg.emit("lpM.cands".into(), hash_cands(&cands));
        dg.emit("lpM.logits".into(), hash_logits(m, 1));
        dg.emit("lpM.state.a".into(), hash_state(m, a));
        dg.emit("lpM.state.b".into(), hash_state(m, b));
        dg.emit("lpM.kv.a".into(), hash_kv(m, a, pa, 40));
        dg.emit("lpM.kv.b".into(), hash_kv(m, b, pb, 24));
        pa += 40;
        pb += 24;
    }

    // 8. serial steps after the prefill slice (decode reads what the prefill wrote), and a window of the three sequences again
    window_seg(m, &mut dg, "lserial_after_slice", &[Part { slot: a, first: pa, src: sa, off: 0, rows: 1, keep: 1 }]);
    pa += 1;
    window_seg(
        m,
        &mut dg,
        "lmulti3b",
        &[
            Part { slot: b, first: pb, src: sb, off: 0, rows: 1, keep: 1 },
            Part { slot: a, first: pa, src: sa, off: 0, rows: 5, keep: 5 },
            Part { slot: c, first: pc, src: sb, off: 400, rows: 6, keep: 6 },
        ],
    );

    reset(m);
    dg.lines
}

/// A scripted run: the model, the two texts and whether to print the lines; it returns the `(segment, hash)` lines.
type DigestScript = fn(&mut NemotronModel, &[Vec<u32>; 2], bool) -> Vec<(String, u64)>;

/// The digest with graphs off (the reference: written to `REXL3_DIGEST_OUT`, compared with `REXL3_DIGEST_EXPECT`), then the same script
/// three times with graphs on from an empty cache: the first pass runs every shape eagerly, the second captures them and the third
/// replays them. Every line of every pass must equal the reference.
fn digest(m: &mut NemotronModel, text: &[Vec<u32>; 2], script: DigestScript) {
    let t = Instant::now();
    let graphs_before = m.graphs_on();
    m.set_graphs(false);
    let lines = script(m, text, true);
    eprintln!("digest script: {} lines in {:.1} s", lines.len(), t.elapsed().as_secs_f64());

    if let Some(path) = std::env::var_os("REXL3_DIGEST_OUT") {
        let text: String = lines.iter().map(|(n, h)| format!("{n} {h:016x}\n")).collect();
        std::fs::write(&path, text).unwrap_or_else(|e| panic!("REXL3_DIGEST_OUT: {e}"));
    }
    if let Some(path) = std::env::var_os("REXL3_DIGEST_EXPECT") {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("REXL3_DIGEST_EXPECT: {e}"));
        let want: BTreeMap<&str, &str> = text.lines().filter_map(|l| l.split_once(' ')).collect();
        let mut bad = Vec::new();
        for (n, h) in &lines {
            match want.get(n.as_str()) {
                Some(w) if *w == format!("{h:016x}") => {}
                Some(w) => bad.push(format!("{n}: expected {w}, got {h:016x}")),
                None => bad.push(format!("{n}: not in the expected file")),
            }
        }
        if want.len() != lines.len() {
            bad.push(format!("{} lines, the expected file has {}", lines.len(), want.len()));
        }
        for b in &bad {
            eprintln!("DIGEST MISMATCH {b}");
        }
        assert!(bad.is_empty(), "{} digest lines differ from {}", bad.len(), path.to_string_lossy());
        eprintln!("DIGEST OK: {} lines identical to {}", lines.len(), path.to_string_lossy());
    }

    if std::env::var("REXL3_NEMO_PERF_NO_GRAPH_PASSES").is_ok_and(|v| v == "1") {
        m.set_graphs(graphs_before);
        return;
    }
    m.set_graphs(true);
    m.clear_graphs().unwrap();
    let mut bad = Vec::new();
    for pass in 1..=3 {
        let t = Instant::now();
        let got = script(m, text, false);
        let s = m.graph_stats();
        eprintln!(
            "digest graph pass {pass}: {:.1} s; captures {}, replays {}, eager windows {}, capture failures {}",
            t.elapsed().as_secs_f64(),
            s.captures,
            s.replays,
            s.eager_windows,
            s.capture_failures
        );
        if got.len() != lines.len() {
            bad.push(format!("pass {pass}: {} lines, the eager run has {}", got.len(), lines.len()));
        }
        for ((n, h), (n2, h2)) in lines.iter().zip(&got) {
            if n != n2 || h != h2 {
                bad.push(format!("pass {pass}: {n}: eager {h:016x}, graphs {n2} {h2:016x}"));
            }
        }
    }
    let s = m.graph_stats();
    assert!(
        s.captures > 0 && s.replays > 0,
        "the graph passes captured {} and replayed {}: nothing was exercised",
        s.captures,
        s.replays
    );
    assert_eq!(s.capture_failures, 0, "a capture failed: {:?}", s.last_failure);
    for b in &bad {
        eprintln!("DIGEST GRAPH MISMATCH {b}");
    }
    assert!(bad.is_empty(), "{} digest lines differ between eager and graph windows", bad.len());
    eprintln!("DIGEST GRAPHS OK: three passes identical to the eager run; {} captures, {} replays", s.captures, s.replays);
    m.set_graphs(graphs_before);
}

// ------------------------------------------------------------------------------------------------------------------ uploads

fn cands_eq(a: &Candidates, b: &Candidates) -> bool {
    a.rows == b.rows
        && a.count == b.count
        && a.ids == b.ids
        && a.vals.iter().zip(&b.vals).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The ordering of the per-step uploads. `DevBuf::upload` is a pageable `cudaMemcpy` on the legacy default stream and the model's stream is
/// non-blocking, so a kernel launched right after an upload could read the old bytes. Two checks, each against a reference run that
/// waits for the device before and after every call:
///
/// * `windows`: many short windows with different token ids back to back (no commit, so the answer is a function of the ids alone); a
///   window that ran on its predecessor's ids answers differently.
/// * `chain`: windows with commits of different keeps, whose replay arguments are uploaded just before the replay launches; a replay that
///   ran on the previous commit's arguments leaves another state.
///
/// Both run with the fixed uploads (must be identical to the reference) and with `set_legacy_uploads(true)` (the old code: reported, not
/// asserted, since it is a race and may not fire in a given run).
fn uploads(m: &mut NemotronModel, text: &[Vec<u32>; 2]) {
    use gb_model::families::nemotron_h_omni::model::set_legacy_uploads;
    let n = env_usize("REXL3_NEMO_PERF_UPLOAD_WINDOWS", 300);
    reset(m);
    set_legacy_uploads(false);
    prefill_ctx(m, 0, &text[0], CTX);

    // ---- windows: 8 one-row and 8 three-row windows of fresh ids
    let batches: Vec<RowBatch> = (0..16)
        .map(|i| {
            let rows = if i < 8 { 1 } else { 3 };
            let toks: Vec<u32> = (0..rows).map(|j| cycle(&text[1], 40 + 13 * i + 5 * j)).collect();
            RowBatch::from_requests(&[RowRequest { slot: 0, tokens: toks, first_position: CTX as u64 }])
        })
        .collect();
    let reference = |m: &mut NemotronModel| -> Vec<Candidates> {
        batches
            .iter()
            .map(|b| {
                gb_cuda::rt::device_sync().unwrap();
                let c = m.window_candidates(b, COUNT).unwrap();
                gb_cuda::rt::device_sync().unwrap();
                c
            })
            .collect()
    };
    let want = reference(m);
    let again = reference(m);
    assert!(want.iter().zip(&again).all(|(a, b)| cands_eq(a, b)), "the reference run is not repeatable");
    let mut order = Vec::with_capacity(n);
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut last = usize::MAX;
    while order.len() < n {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let k = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize % batches.len();
        if k != last {
            order.push(k);
            last = k;
        }
    }
    let mut fixed_bad = 0usize;
    for legacy in [false, true] {
        set_legacy_uploads(legacy);
        let mut bad = 0usize;
        for &k in &order {
            let c = m.window_candidates(&batches[k], COUNT).unwrap();
            if !cands_eq(&c, &want[k]) {
                bad += 1;
            }
        }
        eprintln!(
            "UPLOADS windows {}: {n} windows back to back, {bad} answers differ from the reference",
            if legacy { "legacy" } else { "fixed " }
        );
        if !legacy {
            fixed_bad += bad;
        }
    }
    set_legacy_uploads(false);

    // ---- chain: windows with commits of different keeps
    let plan: Vec<(usize, usize)> = (0..40).map(|i| [(1, 1), (2, 1), (3, 3), (5, 2), (8, 8), (4, 4)][i % 6]).collect();
    let run_chain = |m: &mut NemotronModel, sync_each: bool| -> (Vec<Candidates>, u64) {
        reset(m);
        prefill_ctx(m, 0, &text[0], CTX);
        let mut out = Vec::new();
        let mut p = CTX;
        for &(rows, keep) in &plan {
            let b = window_batch(0, p, &text[1], rows);
            if sync_each {
                gb_cuda::rt::device_sync().unwrap();
            }
            out.push(m.window_candidates(&b, COUNT).unwrap());
            m.commit(0, keep);
            if sync_each {
                m.flush().unwrap();
                gb_cuda::rt::device_sync().unwrap();
            }
            p += keep;
        }
        let h = hash_state(m, 0).0;
        (out, h)
    };
    let (want_c, want_h) = run_chain(m, true);
    let mut chain_bad = [0usize; 2];
    for (i, legacy) in [false, true].into_iter().enumerate() {
        set_legacy_uploads(legacy);
        let (got_c, got_h) = run_chain(m, false);
        chain_bad[i] = want_c.iter().zip(&got_c).filter(|(a, b)| !cands_eq(a, b)).count() + usize::from(want_h != got_h);
        eprintln!(
            "UPLOADS chain   {}: {} windows with commits, {} differ from the reference (answers and final state)",
            if legacy { "legacy" } else { "fixed " },
            plan.len(),
            chain_bad[i]
        );
    }
    set_legacy_uploads(false);
    reset(m);
    assert_eq!(fixed_bad, 0, "windows with the ordered uploads differ from the reference");
    assert_eq!(chain_bad[0], 0, "a chain with the ordered uploads differs from the reference");
}

// ------------------------------------------------------------------------------------------------------------------ trace

/// Windows of 1, 4 and 64 rows and a 64-row prefill slice, each in its own profiler range (4 ranges), after warmups.
fn trace(m: &mut NemotronModel, text: &[Vec<u32>; 2]) {
    reset(m);
    prefill_ctx(m, 0, &text[0], CTX);
    prefill_ctx(m, 1, &text[0], CTX);
    let reps = env_usize("REXL3_NEMO_PERF_TRACE_REPS", 6);
    for rows in [1usize, 4, 64] {
        let b = window_batch(0, CTX, &text[1], rows);
        for _ in 0..3 {
            m.window_candidates(&b, COUNT).unwrap();
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
        gb_cuda::rt::profiler_start();
        let t = Instant::now();
        let n = if rows == 64 { 3 } else { reps };
        for _ in 0..n {
            m.window_candidates(&b, COUNT).unwrap();
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
        gb_cuda::rt::profiler_stop();
        eprintln!("TRACE window {rows} rows: {n} steps, {ms:.1} ms each under the profiler");
    }
    let mut p = CTX;
    for _ in 0..3 {
        let b = window_batch(1, p, &text[1], 64);
        m.prefill_chunk(&b, Some(63)).unwrap();
        p += 64;
    }
    std::thread::sleep(std::time::Duration::from_secs(2));
    gb_cuda::rt::profiler_start();
    let t = Instant::now();
    for _ in 0..3 {
        let b = window_batch(1, p, &text[1], 64);
        m.prefill_chunk(&b, Some(63)).unwrap();
        p += 64;
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / 3.0;
    gb_cuda::rt::profiler_stop();
    eprintln!("TRACE prefill slice 64: 3 slices, {ms:.1} ms each under the profiler");
    // the serial decode step: a window of one row and its commit (the replay of every Mamba layer), the next window behind it
    let mut p = CTX;
    for _ in 0..3 {
        let b = window_batch(0, p, &text[1], 1);
        m.window_candidates(&b, COUNT).unwrap();
        m.commit(0, 1);
        p += 1;
    }
    m.flush().unwrap();
    std::thread::sleep(std::time::Duration::from_secs(2));
    gb_cuda::rt::profiler_start();
    let t = Instant::now();
    for _ in 0..reps {
        let b = window_batch(0, p, &text[1], 1);
        m.window_candidates(&b, COUNT).unwrap();
        m.commit(0, 1);
        p += 1;
    }
    m.flush().unwrap();
    let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
    gb_cuda::rt::profiler_stop();
    eprintln!("TRACE decode step: {reps} steps, {ms:.1} ms each under the profiler");
    reset(m);
}

// ------------------------------------------------------------------------------------------------------------------ the test

#[test]
#[ignore]
fn nemotron_step_perf() {
    let Some(dir) = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set");
        return;
    };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let cfg = config(&dir);
    let text = texts();
    let want = parts();
    let mut m = load(&dir, &cfg);
    eprintln!("clocks: {}", clocks());
    for p in &want {
        let t = Instant::now();
        match p.as_str() {
            "steps" => steps(&mut m, &text),
            "layers" => layers(&mut m, &cfg, &text),
            "digest" => digest(&mut m, &text, digest_script),
            "digestlong" => digest(&mut m, &text, digest_long_script),
            "wide" => wide_slices(&mut m, &text),
            "rates" => rates(&mut m, &cfg, &text),
            "uploads" => uploads(&mut m, &text),
            "trace" => trace(&mut m, &text),
            other => panic!("REXL3_NEMO_PERF_PARTS: unknown part `{other}`"),
        }
        eprintln!("part {p}: {:.1} s", t.elapsed().as_secs_f64());
    }
    eprintln!("clocks: {}", clocks());
}

fn clocks() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.sm,clocks.mem,utilization.gpu,power.draw,temperature.gpu", "--format=csv,noheader"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|e| format!("nvidia-smi: {e}"))
}
