//! FLAKE: a stress loop for the window, commit and replay path of the real `NemotronModel`, over one loaded model.
//!
//! `rollback_equals_replay` (the I3 harness, `nemotron_omni_harness.rs`) was reported to fail in two of three full runs of the 8-layer
//! suite and then to pass 25 of 25 times. An intermittent failure of an exactness gate is a bug until proven a test artefact, so this
//! file runs the same cases (a window of `rows` rows after a 70-token context, `keep` of them committed, then a 3-row step, against a
//! backend that decoded the kept tokens one at a time) over and over in ONE process, and says what differed when something did:
//!
//! * `window`: a row the window returned for a kept token differs from the one-row step of the same token (row invariance, WINDOW side);
//! * `state`: the Mamba S/conv pools or the KV rows after `commit(keep)` differ from those after `keep` serial commits (REPLAY side);
//! * `next`: the pools agree but the 3-row step after them does not;
//! * `repeat`: a run differs from the same run done before it in the same process (nondeterminism, whatever the reference says).
//!
//! Every case is run twice on each side (`A1 B1 A2 B2`), so a failure that comes from stale device memory or from timing shows as a
//! disagreement between two runs of the same computation, and a deterministic bug as an agreement that differs from the reference.
//!
//! Needs `REXL3_NEMO_OMNI_DIR` (the published pack) and a CUDA device; without either it says so and passes.
//!
//! * `REXL3_NEMO_STRESS_ITERS`: passes over the case list (default 2).
//! * `REXL3_NEMO_STRESS_CASES`: `rollback` (the harness's 20 cases, default), `small` (windows of 2 and 3 rows, every keep), `wide`
//!   (rollback plus windows of 16 and 48 rows with a spread of keeps), or `rows:keep` pairs separated by commas (`33:17,64:1`).
//! * `REXL3_NEMO_STRESS_LAYERS`: trunk layers to load (default 8; `88` or `full` loads everything).
//! * `REXL3_NEMO_STRESS_SNAP`: `0` skips the pool snapshots after the commit (the runs then differ from the harness's by nothing).
//! * `REXL3_NEMO_STRESS_STOP`: stop at the first failing case (default: run all and report the rate).
//! * `REXL3_NEMO_STRESS_WINDOWS`: the number of windows of the `windows` part: windows of ten rotating widths (1 to 64 rows) over one
//!   prefilled context, no commit between them, each compared with the first run of its width (a window that depends on the windows
//!   before it, or on timing, differs).
//! * `REXL3_NEMO_STRESS_PARTS`: comma list; `cases` (default), `selftest` (the detection power of this file: a window run on a pool or
//!   a KV row that was poisoned on purpose must differ from the clean one, and a window run on poisoned scratch must not),
//!   `windows` (the repeat loop above alone) and `evict` (windows of 1 to 8
//!   sequences of 1 to 8 rows each at two candidate counts, 128 graph shapes against a cache of 64, each three times in a row,
//!   three rounds: every graph is captured, replayed, evicted and captured again, and each window must equal its first, eager, run).
//! * `REXL3_NEMO_STRESS_POISON`: `1` fills every scratch a step writes before it reads (the step buffers, the window records and retained
//!   rows, the attention partials, the MoE scratch, the whole KV pool, the pools of the unused slots) with NaN before each run, so a
//!   read of a value nobody wrote in this step shows as a difference at once instead of passing on what an earlier step left there.
//! * `REXL3_NEMO_STRESS_HOG`: MiB of a background bandwidth hog (a thread of this process that adds two buffers of that size over and
//!   over on a stream of its own, with seeded random rests), so the model's kernels, its side stream and its programmatic launches run
//!   against competing traffic and the timing between them changes from step to step. 0 or unset: none.
//! * `REXL3_NEMO_STRESS_LATE_COMMIT=<cycles>`: before the window run's `commit`, queue a spin kernel of that many cycles on the model's
//!   stream, so the commit's argument copy cannot execute until the host has long left `commit` (and the stack that held its source
//!   has been overwritten: `clobber` below). A copy that reads its pageable source only when it executes then replays garbage.
//! * `REXL3_NEMO_STRESS_SIDE=0`, `REXL3_NEMO_STRESS_PDL=0`, `REXL3_NEMO_STRESS_GRAPHS=0`: switch the shared expert's side stream, the
//!   programmatic launches and the graph cache off (the bits must not depend on any of them).
//!
//!     REXL3_NEMO_OMNI_DIR=<pack> cargo test --release -p gb-model --test nemotron_omni_stress -- --nocapture

use gb_cuda::nemotron_moe::{set_programmatic_launch, set_side_stream};
use gb_engine::{Backend, RowOutput, RowRequest};
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::{ModelOptions, NemotronBackend};
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::model::NemotronModel;
use gb_model::families::nemotron_h_omni::weights::LayerPlan;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const SLOTS: usize = 8;
const COUNT: usize = 28;
const MAX_CONTEXT: usize = 2048;
const KV_PAGES: usize = 64;

fn pack_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set (the published pack directory)");
    }
    dir
}

fn config(dir: &Path) -> OmniConfig {
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| panic!("config.json: {e}"));
    OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap_or_else(|e| panic!("{e}"))
}

/// A deterministic token stream inside the vocabulary and above the special tokens (xorshift64*, the harness's).
fn tokens(seed: u64, n: usize) -> Vec<u32> {
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

type Backed = Rc<RefCell<NemotronBackend<NemotronModel>>>;

/// The backend a case starts from: every slot released, as the harness's `Shared::fresh` does.
fn fresh(b: &Backed) {
    let mut g = b.borrow_mut();
    for slot in 0..SLOTS {
        g.release(slot);
    }
}

type Be = NemotronBackend<NemotronModel>;

/// Fills what a step writes before it reads (the step buffers, the window records and retained rows, the attention partials, the MoE
/// scratch) with NaN bit patterns, queued on the model's stream. Touches no pool and no KV row, so it may run between two steps.
fn poison_scratch(g: &mut Be) {
    let bump = |r: Result<(), gb_cuda::CudaError>| r.unwrap_or_else(|e| panic!("poison: {e}"));
    let core = &mut g.m.core;
    let st: &gb_cuda::Stream = &core.stream;
    for buf in [
        &mut core.bufs.x,
        &mut core.bufs.xn,
        &mut core.bufs.mix,
        &mut core.bufs.proj,
        &mut core.bufs.z,
        &mut core.bufs.xbc,
        &mut core.bufs.dt,
        &mut core.bufs.ssm_y,
        &mut core.bufs.gated,
        &mut core.bufs.q,
        &mut core.bufs.k,
        &mut core.bufs.v,
        &mut core.bufs.attn_o,
        &mut core.bufs.logits,
        &mut core.bufs.topk_ws,
        &mut core.bufs.topk_ids,
        &mut core.bufs.topk_vals,
        &mut core.bufs.ws,
    ] {
        bump(buf.memset_async(0xFF, st));
    }
    for rec in std::iter::once(&mut core.pre_rec).chain(core.win_rec.iter_mut()) {
        for buf in [&mut rec.xbc, &mut rec.dt, &mut rec.g, &mut rec.decay] {
            bump(buf.memset_async(0xFF, st));
        }
    }
    for buf in core.win_xbc.iter_mut() {
        bump(buf.memset_async(0xFF, st));
    }
    bump(core.moe.poison(st));
    bump(core.attn.poison(st));
}

/// [`poison_scratch`], and the whole KV pool and the pools of every slot but 0 (slot 0 is zeroed by its `begin`). Called right after
/// `fresh`, so no live row of the KV pool or of a pool is lost.
fn poison(b: &Backed) {
    use gb_model::families::nemotron_h_omni::model::{StatePools, CONV_SLOT_BYTES, S_SLOT_BYTES};
    let mut g = b.borrow_mut();
    poison_scratch(&mut g);
    let bump = |r: Result<(), gb_cuda::CudaError>| r.unwrap_or_else(|e| panic!("poison: {e}"));
    let core = &mut g.m.core;
    let st: &gb_cuda::Stream = &core.stream;
    bump(core.kv.k.memset_async(0xFF, st));
    bump(core.kv.v.memset_async(0xFF, st));
    for m in core.mamba.iter_mut() {
        bump(m.s.memset_range_async(StatePools::s_offset(1), (SLOTS - 1) * S_SLOT_BYTES, 0xFF, st));
        bump(m.conv.memset_range_async(StatePools::conv_offset(1), (SLOTS - 1) * CONV_SLOT_BYTES, 0xFF, st));
    }
}

/// A thread that keeps the device's memory system busy from a stream of its own until dropped.
struct Hog {
    stop: Arc<AtomicBool>,
    rounds: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Hog {
    fn start(mib: usize) -> Hog {
        let stop = Arc::new(AtomicBool::new(false));
        let rounds = Arc::new(AtomicU64::new(0));
        let (s2, r2) = (stop.clone(), rounds.clone());
        let handle = std::thread::spawn(move || {
            let n = mib * 262_144; // floats
            let mut a = gb_cuda::DevBuf::zeroed(n * 4).unwrap_or_else(|e| panic!("hog: {e}"));
            let b = gb_cuda::DevBuf::zeroed(n * 4).unwrap_or_else(|e| panic!("hog: {e}"));
            let st = gb_cuda::Stream::new_nonblocking().unwrap_or_else(|e| panic!("hog: {e}"));
            let mut x = 0x9E37_79B9_7F4A_7C15u64;
            while !s2.load(Ordering::Relaxed) {
                // a burst of 1 to 4 passes over the buffers, then a rest of 0 to 1.5 ms
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                for _ in 0..1 + (x & 3) {
                    gb_cuda::glm_mtp::add_f32(&mut a, &b, n, &st).unwrap_or_else(|e| panic!("hog: {e}"));
                }
                st.sync().unwrap_or_else(|e| panic!("hog: {e}"));
                r2.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(Duration::from_micros((x >> 8) % 1500));
            }
        });
        Hog { stop, rounds, handle: Some(handle) }
    }
}

impl Drop for Hog {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        eprintln!("hog: {} rounds", self.rounds.load(Ordering::Relaxed));
    }
}

fn bits(r: &RowOutput) -> (Vec<u32>, Vec<u32>) {
    (r.ids.clone(), r.values.iter().map(|v| v.to_bits()).collect())
}

/// First row where two lists differ, with a short description.
fn first_diff(a: &[RowOutput], b: &[RowOutput]) -> Option<String> {
    if a.len() != b.len() {
        return Some(format!("{} rows against {}", a.len(), b.len()));
    }
    for (j, (x, y)) in a.iter().zip(b).enumerate() {
        if !x.bits_eq(y) {
            let (xi, xv) = bits(x);
            let (yi, yv) = bits(y);
            let n = xi.len().min(yi.len());
            let ids_differ = (0..n).filter(|&k| xi[k] != yi[k]).count();
            let vals_differ = (0..n).filter(|&k| xv[k] != yv[k]).count();
            let k = (0..n).find(|&k| xi[k] != yi[k] || xv[k] != yv[k]).unwrap_or(0);
            return Some(format!(
                "row {j}: {ids_differ} ids and {vals_differ} values of {n} differ, first at candidate {k}: id {} / {} value {:e} / {:e}",
                xi[k],
                yi[k],
                f32::from_bits(xv[k]),
                f32::from_bits(yv[k])
            ));
        }
    }
    None
}

/// The Mamba pools of every layer and the KV rows of slot 0, as bit patterns.
#[derive(PartialEq)]
struct Snap {
    mamba: Vec<(Vec<u32>, Vec<u16>)>,
    kv: Vec<(Vec<u16>, Vec<u16>)>,
}

fn snapshot(b: &Backed, n_mamba: usize, n_attn: usize, len: usize) -> Result<Snap, ModelError> {
    let mut g = b.borrow_mut();
    let m = &mut g.m;
    let mut mamba = Vec::new();
    for o in 0..n_mamba {
        let (s, conv) = m.debug_mamba_state(o, 0)?;
        mamba.push((s.iter().map(|x| x.to_bits()).collect(), conv));
    }
    let mut kv = Vec::new();
    for a in 0..n_attn {
        kv.push(m.debug_kv_rows(a, 0, 0, len)?);
    }
    Ok(Snap { mamba, kv })
}

/// Where two snapshots first differ: the layer and what, and how many elements.
fn snap_diff(a: &Snap, b: &Snap) -> Option<String> {
    for (o, (x, y)) in a.mamba.iter().zip(&b.mamba).enumerate() {
        let ds = x.0.iter().zip(&y.0).filter(|(p, q)| p != q).count();
        let dc = x.1.iter().zip(&y.1).filter(|(p, q)| p != q).count();
        if ds != 0 || dc != 0 {
            return Some(format!(
                "Mamba layer ordinal {o}: S differs in {ds} of {} elements, conv in {dc} of {}",
                x.0.len(),
                x.1.len()
            ));
        }
    }
    for (o, (x, y)) in a.kv.iter().zip(&b.kv).enumerate() {
        let dk = x.0.iter().zip(&y.0).filter(|(p, q)| p != q).count();
        let dv = x.1.iter().zip(&y.1).filter(|(p, q)| p != q).count();
        if dk != 0 || dv != 0 {
            return Some(format!("attention layer ordinal {o}: K differs in {dk} elements, V in {dv}"));
        }
    }
    None
}

struct Run {
    /// The rows the window (or the kept one-row steps) returned for the kept tokens.
    kept: Vec<RowOutput>,
    /// The pools after the commit (or the serial commits), when snapshots are on.
    snap: Option<Snap>,
    /// The follow-up step.
    next: Vec<RowOutput>,
}

struct Counts {
    n_mamba: usize,
    n_attn: usize,
    snap: bool,
    poison: bool,
    /// Cycles of a spin kernel queued before the window run's commit (0: none).
    late_commit: i64,
}

/// Overwrites the stack below the caller with a pattern, so a stack array a callee passed to an asynchronous copy and has left behind is
/// not what it was.
#[inline(never)]
fn clobber() -> u64 {
    let mut a = [0u64; 8192];
    for (i, w) in a.iter_mut().enumerate() {
        *w = 0xDEAD_BEEF_0000_0000 | i as u64;
    }
    std::hint::black_box(&mut a);
    a.iter().fold(0, |x, y| x ^ y)
}

/// The window side: one window of `window`, `keep` of it committed, then `next`.
fn run_window(b: &Backed, c: &Counts, dirty: bool, ctx: &[u32], window: &[u32], keep: usize, next: &[u32]) -> Run {
    fresh(b);
    if dirty {
        poison(b);
    }
    let mut g = b.borrow_mut();
    g.prefill(0, 0, ctx, COUNT);
    let first = g.step(&[RowRequest { slot: 0, tokens: window.to_vec(), first_position: ctx.len() as u64 }], COUNT);
    if c.late_commit > 0 {
        gb_cuda::graph::spin(&g.m.core.stream, c.late_commit).unwrap_or_else(|e| panic!("spin: {e}"));
    }
    g.commit(0, keep);
    if c.late_commit > 0 {
        std::hint::black_box(clobber());
    }
    drop(g);
    let snap = if c.snap { snapshot(b, c.n_mamba, c.n_attn, ctx.len() + keep).ok() } else { None };
    let mut g = b.borrow_mut();
    let got = g.step(&[RowRequest { slot: 0, tokens: next.to_vec(), first_position: (ctx.len() + keep) as u64 }], COUNT);
    Run { kept: first.into_iter().next().unwrap()[..keep].to_vec(), snap, next: got.into_iter().next().unwrap() }
}

/// The serial side: the kept tokens one step and one commit each, then `next`.
fn run_serial(b: &Backed, c: &Counts, dirty: bool, ctx: &[u32], window: &[u32], keep: usize, next: &[u32]) -> Run {
    fresh(b);
    if dirty {
        poison(b);
    }
    let mut g = b.borrow_mut();
    g.prefill(0, 0, ctx, COUNT);
    let mut pos = ctx.len() as u64;
    let mut kept = Vec::new();
    for &t in &window[..keep] {
        let o = g.step(&[RowRequest { slot: 0, tokens: vec![t], first_position: pos }], COUNT);
        kept.push(o.into_iter().next().unwrap().into_iter().next().unwrap());
        g.commit(0, 1);
        pos += 1;
    }
    drop(g);
    let snap = if c.snap { snapshot(b, c.n_mamba, c.n_attn, ctx.len() + keep).ok() } else { None };
    let mut g = b.borrow_mut();
    let want = g.step(&[RowRequest { slot: 0, tokens: next.to_vec(), first_position: pos }], COUNT);
    Run { kept, snap, next: want.into_iter().next().unwrap() }
}

fn describe_failure(a1: &Run, b1: &Run, a2: &Run, b2: &Run) -> Vec<String> {
    let mut why = Vec::new();
    let snap_eq = |x: &Run, y: &Run| match (&x.snap, &y.snap) {
        (Some(p), Some(q)) => snap_diff(p, q),
        _ => None,
    };
    // runs of the same computation
    if let Some(d) = first_diff(&a1.kept, &a2.kept) {
        why.push(format!("repeat: window rows differ between two window runs: {d}"));
    }
    if let Some(d) = snap_eq(a1, a2) {
        why.push(format!("repeat: pools after the commit differ between two window runs: {d}"));
    }
    if let Some(d) = first_diff(&a1.next, &a2.next) {
        why.push(format!("repeat: the follow-up differs between two window runs: {d}"));
    }
    if let Some(d) = first_diff(&b1.kept, &b2.kept) {
        why.push(format!("repeat: one-row steps differ between two serial runs: {d}"));
    }
    if let Some(d) = snap_eq(b1, b2) {
        why.push(format!("repeat: pools differ between two serial runs: {d}"));
    }
    if let Some(d) = first_diff(&b1.next, &b2.next) {
        why.push(format!("repeat: the follow-up differs between two serial runs: {d}"));
    }
    // against the reference
    for (tag, a, b) in [("A1/B1", a1, b1), ("A2/B2", a2, b2)] {
        if let Some(d) = first_diff(&a.kept, &b.kept) {
            why.push(format!("window ({tag}): {d}"));
        }
        if let Some(d) = snap_eq(a, b) {
            why.push(format!("state ({tag}): {d}"));
        }
        if let Some(d) = first_diff(&a.next, &b.next) {
            why.push(format!("next ({tag}): {d}"));
        }
    }
    why
}

fn cases_from_env() -> Vec<(usize, usize)> {
    let spec = std::env::var("REXL3_NEMO_STRESS_CASES").unwrap_or_else(|_| "rollback".into());
    let rollback = || {
        let mut v = Vec::new();
        for rows in [2usize, 8, 33, 64] {
            let keeps: Vec<usize> = if rows <= 8 { (1..=rows).collect() } else { vec![1, 2, rows / 2, rows - 1, rows] };
            for k in keeps {
                v.push((rows, k));
            }
        }
        v
    };
    match spec.as_str() {
        "rollback" => rollback(),
        "small" => (2usize..=3).flat_map(|r| (1..=r).map(move |k| (r, k))).collect(),
        "wide" => {
            let mut v = rollback();
            for rows in [16usize, 48] {
                for k in [1, 2, rows / 4, rows / 2, rows - 1, rows] {
                    v.push((rows, k));
                }
            }
            v
        }
        other => other
            .split(',')
            .map(|p| {
                let (r, k) = p.split_once(':').unwrap_or_else(|| panic!("REXL3_NEMO_STRESS_CASES: `{p}` is not rows:keep"));
                (r.trim().parse().unwrap(), k.trim().parse().unwrap())
            })
            .collect(),
    }
}

#[test]
fn rollback_equals_replay_holds_on_every_repetition() {
    let Some(dir) = pack_dir() else { return };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let layers = match std::env::var("REXL3_NEMO_STRESS_LAYERS").as_deref() {
        Ok("88") | Ok("full") => None,
        Ok(n) => Some(n.parse::<usize>().unwrap_or_else(|_| panic!("REXL3_NEMO_STRESS_LAYERS: `{n}`"))),
        Err(_) => Some(8),
    };
    let iters: usize = std::env::var("REXL3_NEMO_STRESS_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let snap = std::env::var("REXL3_NEMO_STRESS_SNAP").map(|v| v != "0").unwrap_or(true);
    let stop = std::env::var("REXL3_NEMO_STRESS_STOP").map(|v| v != "0").unwrap_or(false);
    let parts = std::env::var("REXL3_NEMO_STRESS_PARTS").unwrap_or_else(|_| "cases".into());
    let poison_on = std::env::var("REXL3_NEMO_STRESS_POISON").map(|v| v == "1").unwrap_or(false);
    let off = |k: &str| std::env::var(k).map(|v| v == "0").unwrap_or(false);
    if off("REXL3_NEMO_STRESS_SIDE") {
        set_side_stream(false);
    }
    if off("REXL3_NEMO_STRESS_PDL") {
        set_programmatic_launch(false);
    }
    let cases = cases_from_env();
    let hog_mib: usize = std::env::var("REXL3_NEMO_STRESS_HOG").ok().and_then(|v| v.parse().ok()).unwrap_or(0);

    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let opts = ModelOptions { slots: SLOTS, kv_pages: KV_PAGES, max_context: MAX_CONTEXT, prefill_rows: 64, layers, mtp: false };
    let t = Instant::now();
    let mut model = match NemotronModel::load(&dir, &cfg, &plan, opts.clone()) {
        Ok(m) => m,
        Err(ModelError::NotBuilt(what)) => {
            eprintln!("not run: {what} is not built yet");
            return;
        }
        Err(e) => panic!("load: {e}"),
    };
    if off("REXL3_NEMO_STRESS_GRAPHS") {
        model.set_graphs(false);
    }
    let n_layers = layers.unwrap_or(cfg.pattern.chars().count());
    let pat: Vec<char> = cfg.pattern.chars().take(n_layers).collect();
    let counts = Counts {
        n_mamba: pat.iter().filter(|&&c| c == 'M').count(),
        n_attn: pat.iter().filter(|&&c| c == '*').count(),
        snap,
        poison: poison_on,
        late_commit: std::env::var("REXL3_NEMO_STRESS_LATE_COMMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
    };
    eprintln!(
        "loaded {n_layers} layers ({} Mamba, {} attention) in {:.1} s; graphs {}",
        counts.n_mamba,
        counts.n_attn,
        t.elapsed().as_secs_f64(),
        if model.graphs_on() { "on" } else { "off" }
    );
    let backend = NemotronBackend::with_runtime(model, 64, &opts).unwrap_or_else(|e| panic!("backend: {e}"));
    let b: Backed = Rc::new(RefCell::new(backend));

    let _hog = (hog_mib > 0).then(|| Hog::start(hog_mib));
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0usize;
    let ctx = tokens(31, 70);
    let next = tokens(60, 3);

    if parts.contains("cases") {
        'iters: for it in 0..iters {
            let t0 = Instant::now();
            let mut bad = 0;
            for &(rows, keep) in &cases {
                let window = tokens(40 + rows as u64, rows);
                total += 1;
                // the first pair runs on whatever the model holds, the second (when poison is on) on scratch filled with NaN: a read of
                // something this step did not write then shows as a difference between the pairs
                let a1 = run_window(&b, &counts, false, &ctx, &window, keep, &next);
                let b1 = run_serial(&b, &counts, false, &ctx, &window, keep, &next);
                let a2 = run_window(&b, &counts, counts.poison, &ctx, &window, keep, &next);
                let b2 = run_serial(&b, &counts, counts.poison, &ctx, &window, keep, &next);
                let why = describe_failure(&a1, &b1, &a2, &b2);
                if !why.is_empty() {
                    bad += 1;
                    let line = format!("iteration {it} window {rows} keep {keep}: {}", why.join(" | "));
                    eprintln!("FAIL    {line}");
                    failures.push(line);
                    if stop {
                        break 'iters;
                    }
                }
                if let Some(e) = {
                    let mut g = b.borrow_mut();
                    let f = g.quiesce().err();
                    let l = g.take_error().map(|e| e.message);
                    f.or(l)
                } {
                    let line = format!("iteration {it} window {rows} keep {keep}: backend error: {e}");
                    eprintln!("FAIL    {line}");
                    failures.push(line);
                    break 'iters;
                }
            }
            eprintln!(
                "iteration {it}: {} cases, {bad} failing, {:.1} s (graphs: {:?})",
                cases.len(),
                t0.elapsed().as_secs_f64(),
                b.borrow().m.graph_stats()
            );
        }
    }

    if parts.contains("windows") {
        // windows of rotating widths, repeated over and over without a commit: each repeat of a window must give the bits of its first
        // run, whatever the windows between them left in the scratch (the widths cross the side stream's 16 rows and the router's layouts)
        let n: usize = std::env::var("REXL3_NEMO_STRESS_WINDOWS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
        let shapes = [1usize, 2, 3, 5, 8, 12, 16, 17, 33, 64];
        fresh(&b);
        let mut g = b.borrow_mut();
        g.prefill(0, 0, &ctx, COUNT);
        let at = ctx.len() as u64;
        let wins: Vec<Vec<u32>> = shapes.iter().map(|&r| tokens(7000 + r as u64, r)).collect();
        let mut reference: Vec<Option<Vec<RowOutput>>> = vec![None; shapes.len()];
        let mut bad = 0;
        let t0 = Instant::now();
        for i in 0..n {
            let k = i % shapes.len();
            if poison_on && reference[k].is_some() {
                // everything a window writes before it reads, and the pools of the unused slots, but not slot 0's state or its KV rows
                poison_scratch(&mut g);
            }
            let o = g.step(&[RowRequest { slot: 0, tokens: wins[k].clone(), first_position: at }], COUNT);
            let rows = o.into_iter().next().unwrap();
            match &reference[k] {
                None => reference[k] = Some(rows),
                Some(r) => {
                    total += 1;
                    if let Some(d) = first_diff(r, &rows) {
                        bad += 1;
                        let line = format!("repeat window {i} ({} rows): {d}", shapes[k]);
                        if bad <= 10 {
                            eprintln!("FAIL    {line}");
                        }
                        failures.push(line);
                    }
                }
            }
        }
        eprintln!(
            "windows: {n} windows of {} widths, {bad} differing from their first run, {:.1} s",
            shapes.len(),
            t0.elapsed().as_secs_f64()
        );
    }

    if parts.contains("selftest") {
        // the checks above are only worth their zero if they can see a fault: poison state the next window READS and it must differ
        fresh(&b);
        let mut g = b.borrow_mut();
        g.prefill(0, 0, &ctx, COUNT);
        let at = ctx.len() as u64;
        let win = tokens(7777, 5);
        let run = |g: &mut Be| {
            g.step(&[RowRequest { slot: 0, tokens: win.clone(), first_position: at }], COUNT).into_iter().next().unwrap()
        };
        let clean = run(&mut g);
        let clean2 = run(&mut g);
        assert!(first_diff(&clean, &clean2).is_none(), "selftest: two clean runs of one window differ");
        poison_scratch(&mut g);
        let scratch = run(&mut g);
        assert!(
            first_diff(&clean, &scratch).is_none(),
            "selftest: a window must not read its scratch before it writes it: {:?}",
            first_diff(&clean, &scratch)
        );
        let bump = |r: Result<(), gb_cuda::CudaError>| r.unwrap_or_else(|e| panic!("poison: {e}"));
        // the committed Mamba state of slot 0 is read by the window: poisoning it must show
        {
            use gb_model::families::nemotron_h_omni::model::{StatePools, S_SLOT_BYTES};
            let core = &mut g.m.core;
            let st: &gb_cuda::Stream = &core.stream;
            for m in core.mamba.iter_mut() {
                bump(m.s.memset_range_async(StatePools::s_offset(0), S_SLOT_BYTES, 0xFF, st));
            }
        }
        let state = run(&mut g);
        assert!(
            first_diff(&clean, &state).is_some(),
            "selftest: a poisoned S pool did not change the window: the comparisons are blind"
        );
        // and so must the KV rows of the context (positions 0..70 of slot 0 are read by every row of the window)
        drop(g);
        fresh(&b);
        let mut g = b.borrow_mut();
        g.prefill(0, 0, &ctx, COUNT);
        {
            let core = &mut g.m.core;
            let st: &gb_cuda::Stream = &core.stream;
            bump(core.kv.k.memset_async(0xFF, st));
        }
        let kv = run(&mut g);
        assert!(
            first_diff(&clean, &kv).is_some(),
            "selftest: a poisoned K pool did not change the window: the comparisons are blind"
        );
        eprintln!("selftest: clean == clean, poisoned scratch == clean, poisoned S differs, poisoned K differs");
    }

    if parts.contains("evict") {
        // more window shapes than the graph cache holds, over eight live sequences: the eviction path (drain the stream, drop the least
        // recently used graph, capture the new shape) runs again and again, and every window is compared with its first run
        fresh(&b);
        let mut g = b.borrow_mut();
        for slot in 0..SLOTS {
            g.prefill(slot, 0, &tokens(500 + slot as u64, 70 + slot), COUNT);
        }
        let mut shapes: Vec<(usize, usize, usize)> = Vec::new(); // (count, sequences, rows each)
        for count in [28usize, 20] {
            for n_seq in 1..=8usize {
                for each in 1..=8usize {
                    shapes.push((count, n_seq, each));
                }
            }
        }
        let mut reference: Vec<Option<Vec<Vec<RowOutput>>>> = vec![None; shapes.len()];
        let t0 = Instant::now();
        let mut bad = 0;
        for round in 0..3 {
            for (k, &(count, n_seq, each)) in shapes.iter().enumerate() {
                let reqs: Vec<RowRequest> = (0..n_seq)
                    .map(|slot| RowRequest {
                        slot,
                        tokens: tokens(8000 + (k * 8 + slot) as u64, each),
                        first_position: (70 + slot) as u64,
                    })
                    .collect();
                // three in a row: the first of a round is a capture (the shape was evicted since), the next ones replay it
                for rep in 0..3 {
                    let out = g.step(&reqs, count);
                    match &reference[k] {
                        None => reference[k] = Some(out),
                        Some(r) => {
                            total += 1;
                            let flat = |v: &Vec<Vec<RowOutput>>| v.iter().flatten().cloned().collect::<Vec<_>>();
                            if let Some(d) = first_diff(&flat(r), &flat(&out)) {
                                bad += 1;
                                let line = format!(
                                    "evict round {round}.{rep} shape {k} (count {count}, {n_seq} sequences of {each} rows): {d}"
                                );
                                if bad <= 10 {
                                    eprintln!("FAIL    {line}");
                                }
                                failures.push(line);
                            }
                        }
                    }
                }
            }
        }
        eprintln!(
            "evict: {} shapes x 3 rounds x 3, {bad} differing from their first run, {:.1} s (graphs: {:?})",
            shapes.len(),
            t0.elapsed().as_secs_f64(),
            g.m.graph_stats()
        );
    }

    eprintln!("stress: {total} comparisons, {} failing", failures.len());
    assert!(failures.is_empty(), "{} of {total} stress comparisons failed:\n{}", failures.len(), failures.join("\n"));
}
