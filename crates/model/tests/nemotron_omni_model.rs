//! The composed Nemotron 3.5 Super VL model on a GB10, through the first 8 trunk layers (`MEMEMEM*`: four Mamba2, three latent-MoE and
//! one attention layer), the whole head (`norm_f`, lm_head, top-k) and the pack's own embedding.
//!
//! Needs `REXL3_NEMO_OMNI_DIR` (the published pack) and a CUDA device; without either, or while the glue kernels or the weight bridge are
//! still stubs, it reports what it did not run and passes. One model is built once and the checks run in order on its slots:
//!
//! 1. one 64-token prefill slice with a tap on every point: finite activations at the embedding, every mixer and layer output, the final
//!    norm and the logits, finite and descending candidates, no sticky error;
//! 2. the commit equivalences, bit for bit, on the Mamba S and conv pools of every Mamba layer and on the KV rows of the attention layer:
//!    a 3-row window then `commit(3)`, a 5-row window then `commit(3)`, and a 3-row prefill slice all leave a slot in one state, and the
//!    first rows of the 5-row window and the 3-row window give the same candidates; then one more row each way, and a batched window
//!    equal to the solo one. Under the prefill contract P2 (the default) a prefilled row and a decoded row do not have the same bits
//!    (`docs/NEMOTRON_PREFILL_SPEED.md` 6.14): there the window path is held to the window path and the prefill path to a P2 prefill
//!    reference (the same rows cut differently: chunked == whole); under P1 (`REXL3_NEMO_PREFILL_CONTRACT=p1`) the checks are as above;
//! 3. the host contract on a real model: a slice past `prefill_rows`, a double begin, a double commit (returned by `flush`);
//! 4. peak device memory and the time of each kind of step.

use gb_cuda::{CudaError, DevBuf, Stream};
use gb_engine::RowRequest;
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::ModelOptions;
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::contract::{self, PrefillContract};
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::model::{NemotronModel, Point, Tap};
use gb_model::families::nemotron_h_omni::weights::LayerPlan;
use gb_model::forward::Candidates;
use gb_model::rows::RowBatch;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

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

/// A deterministic token stream inside the vocabulary (xorshift64*).
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

fn batch(reqs: &[(usize, usize, &[u32])]) -> RowBatch {
    let r: Vec<RowRequest> =
        reqs.iter().map(|&(slot, first, t)| RowRequest { slot, first_position: first as u64, tokens: t.to_vec() }).collect();
    RowBatch::from_requests(&r)
}

/// What a tap saw at one point.
#[derive(Debug, Clone)]
struct Seen {
    point: Point,
    rows: usize,
    width: usize,
    finite: bool,
    max_abs: f32,
}

struct CheckTap(Arc<Mutex<Vec<Seen>>>);

impl Tap for CheckTap {
    fn tap(&mut self, point: Point, buf: &DevBuf, rows: usize, width: usize, st: &Stream) -> Result<(), ModelError> {
        st.sync()?;
        let mut v = vec![0f32; rows * width];
        buf.download(&mut v)?;
        let finite = v.iter().all(|x| x.is_finite());
        let max_abs = v.iter().fold(0f32, |m, x| m.max(x.abs()));
        self.0.lock().unwrap().push(Seen { point, rows, width, finite, max_abs });
        Ok(())
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The first `n` candidates of row `r`, as (ids, value bits).
fn top(c: &Candidates, r: usize, n: usize) -> (Vec<i32>, Vec<u32>) {
    (c.ids[r * c.count..r * c.count + n].to_vec(), bits(&c.vals[r * c.count..r * c.count + n]))
}

fn free_gib() -> f64 {
    gb_cuda::rt::mem_info().map(|(f, _)| f as f64 / (1u64 << 30) as f64).unwrap_or(f64::NAN)
}

#[test]
fn gpu_smoke_of_the_first_8_layers_prefill_window_commit_and_the_host_contract() {
    let Some(dir) = pack_dir() else { return };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let opts = ModelOptions { slots: 4, kv_pages: 8, max_context: 2048, prefill_rows: 64, layers: Some(8), mtp: false };

    let (free0, total) = gb_cuda::rt::mem_info().unwrap();
    let t = Instant::now();
    let mut m = match NemotronModel::load(&dir, &cfg, &plan, opts.clone()) {
        Ok(m) => m,
        Err(ModelError::NotBuilt(what)) => {
            eprintln!("not run: {what} is not built yet");
            return;
        }
        Err(ModelError::Cuda(CudaError(801))) => {
            eprintln!("not run: a glue kernel is still the wave-0 stub (cudaErrorNotSupported)");
            return;
        }
        Err(e) => panic!("load: {e}"),
    };
    let load_s = t.elapsed().as_secs_f64();
    let free1 = free_gib();
    let pools = m.pool_bytes();
    eprintln!(
        "load: {load_s:.1} s; device {:.2} GiB used of {:.1}; pools: state {:.0} MiB, kv {:.0} MiB, records {:.0} MiB, buffers {:.0} MiB, scratch {:.0} MiB",
        free0 as f64 / (1u64 << 30) as f64 - free1,
        total as f64 / (1u64 << 30) as f64,
        pools.state as f64 / 1048576.0,
        pools.kv as f64 / 1048576.0,
        pools.records as f64 / 1048576.0,
        pools.bufs as f64 / 1048576.0,
        pools.scratch as f64 / 1048576.0,
    );
    assert_eq!(m.vocab_size(), 131072);
    assert!(!m.has_mtp(), "the MTP head is built in milestone M2");
    let mut min_free = free1;

    // ---- 1. one prefill slice, every tap
    let log = Arc::new(Mutex::new(Vec::new()));
    m.set_tap(Some(Box::new(CheckTap(log.clone()))));
    let prefix = tokens(1, 64);
    m.begin_slot(0).unwrap();
    let t = Instant::now();
    let c = m.prefill_chunk(&batch(&[(0, 0, &prefix)]), Some(63)).unwrap().expect("candidates of the wanted row");
    let tapped_ms = t.elapsed().as_secs_f64() * 1e3;
    m.set_tap(None);
    m.flush().unwrap();
    assert_eq!((c.rows, c.count), (1, 64));
    assert!(c.ids.iter().all(|&i| (0..131072).contains(&i)), "candidate ids are in the vocabulary");
    assert!(c.vals.iter().all(|v| v.is_finite()), "candidate logits are finite");
    assert!(c.vals.windows(2).all(|w| w[0] >= w[1]), "candidates are descending");
    let mut seen_ids = c.ids.clone();
    seen_ids.sort();
    seen_ids.dedup();
    assert_eq!(seen_ids.len(), 64, "candidates are distinct");
    let seen = log.lock().unwrap().clone();
    let at = |p: Point| seen.iter().find(|s| s.point == p).unwrap_or_else(|| panic!("no tap at {p:?}"));
    let mut table = String::new();
    for (p, rows, width) in std::iter::once((Point::Embed, 64, 4096))
        .chain((0..8).flat_map(|i| [(Point::Mixer(i), 64, 4096), (Point::Layer(i), 64, 4096)]))
        .chain([(Point::FinalNorm, 64, 4096), (Point::Logits, 1, 131072)])
    {
        let s = at(p);
        assert_eq!((s.rows, s.width), (rows, width), "{p:?}");
        assert!(s.finite, "non-finite values at {p:?}");
        table.push_str(&format!("  {p:?}: max |x| {:.3}\n", s.max_abs));
    }
    eprintln!("taps (all finite):\n{table}");
    assert_eq!(seen.len(), 1 + 16 + 2, "each point fires once");
    assert_eq!(m.slot_len(0), Some(64));
    min_free = min_free.min(free_gib());

    // ---- 2. commit equivalences
    let n_mamba = m.core.mamba.len();
    assert_eq!(n_mamba, 4, "layers 0, 2, 4 and 6");
    let t = Instant::now();
    for s in 1..=3 {
        m.begin_slot(s).unwrap();
        assert!(m.prefill_chunk(&batch(&[(s, 0, &prefix)]), None).unwrap().is_none());
    }
    // begin_slot (zeroing, a synchronise) and a 64-row slice without logits and without a tap, averaged over three slots
    let prefill64_ms = t.elapsed().as_secs_f64() * 1e3 / 3.0;
    // the same prefix leaves the same state whichever slot it ran in and whether or not logits were asked for
    for ord in 0..n_mamba {
        let base = m.debug_mamba_state(ord, 0).unwrap();
        for s in 1..=3 {
            let o = m.debug_mamba_state(ord, s).unwrap();
            assert_eq!(bits(&base.0), bits(&o.0), "S of Mamba layer {ord}: slot 0 vs slot {s}");
            assert_eq!(base.1, o.1, "conv of Mamba layer {ord}: slot 0 vs slot {s}");
        }
    }
    // Under the prefill contract P2 (the default since rc2d, c69fb643) prompt rows take the tensor-core arithmetic and window rows
    // P1's, so a prefilled row and the same row decoded do not have the same bits (`docs/NEMOTRON_PREFILL_SPEED.md` 6.14: `windows ==
    // prefill` no longer holds). Under P1 every check below is the one this test always made. Under P2 the window path is held to the
    // window path (slots 1 and 3) and the prefill path (slot 2) to a P2 prefill reference: slot 0, which holds the same 64-row prefix
    // (checked above), takes the same rows cut differently (1 + 2 rows: chunked == whole, P2's slice independence).
    let p2 = contract::current() == PrefillContract::P2;
    let before: Vec<_> = (0..n_mamba).map(|ord| m.debug_mamba_state(ord, 0).unwrap()).collect();
    let w = tokens(2, 5);
    let t = Instant::now();
    let ca = m.window_candidates(&batch(&[(1, 64, &w[..3])]), 8).unwrap();
    let window3_ms = t.elapsed().as_secs_f64() * 1e3;
    m.commit(1, 3);
    let cc = m.window_candidates(&batch(&[(3, 64, &w[..5])]), 8).unwrap();
    m.commit(3, 3);
    let t = Instant::now();
    let cb = m.prefill_chunk(&batch(&[(2, 64, &w[..3])]), Some(2)).unwrap().unwrap();
    let prefill3_ms = t.elapsed().as_secs_f64() * 1e3;
    m.flush().unwrap();
    let cr = if p2 {
        assert!(m.prefill_chunk(&batch(&[(0, 64, &w[..1])]), None).unwrap().is_none());
        let r = m.prefill_chunk(&batch(&[(0, 65, &w[1..3])]), Some(1)).unwrap().unwrap();
        m.flush().unwrap();
        Some(r)
    } else {
        None
    };
    for s in 1..=3 {
        assert_eq!(m.slot_len(s), Some(67), "slot {s}");
    }
    // the slots that hold slot 1's state (a 3-row window, commit 3), and under P2 the prefill slot and its reference
    let same_as_1: &[usize] = if p2 { &[3] } else { &[2, 3] };
    assert_eq!(before.len(), n_mamba);
    for (ord, before) in before.iter().enumerate() {
        let a = m.debug_mamba_state(ord, 1).unwrap();
        for &s in same_as_1 {
            let o = m.debug_mamba_state(ord, s).unwrap();
            assert_eq!(bits(&a.0), bits(&o.0), "S of Mamba layer {ord} after 3 rows: slot 1 (window, commit 3) vs slot {s}");
            assert_eq!(a.1, o.1, "conv of Mamba layer {ord} after 3 rows: slot 1 vs slot {s}");
        }
        if p2 {
            let (x, y) = (m.debug_mamba_state(ord, 2).unwrap(), m.debug_mamba_state(ord, 0).unwrap());
            assert_eq!(
                bits(&x.0),
                bits(&y.0),
                "S of Mamba layer {ord} after 3 prefilled rows (P2): one 3-row slice vs 1 + 2 rows"
            );
            assert_eq!(x.1, y.1, "conv of Mamba layer {ord} after 3 prefilled rows (P2): one 3-row slice vs 1 + 2 rows");
            assert_ne!(bits(&before.0), bits(&x.0), "the prefill moved the state of layer {ord}");
        }
        assert_ne!(bits(&before.0), bits(&a.0), "the commit moved the state of layer {ord}");
    }
    let kv1 = m.debug_kv_rows(0, 1, 0, 67).unwrap();
    for &s in same_as_1 {
        assert_eq!(kv1, m.debug_kv_rows(0, s, 0, 67).unwrap(), "KV rows 0..67 of slot 1 vs slot {s}");
    }
    if p2 {
        assert_eq!(
            m.debug_kv_rows(0, 2, 0, 67).unwrap(),
            m.debug_kv_rows(0, 0, 0, 67).unwrap(),
            "KV rows 0..67 (P2): one 3-row slice vs 1 + 2 rows"
        );
        // the third prefilled row: one 3-row slice vs the 1 + 2 cut
        assert_eq!(top(&cb, 0, 8), top(cr.as_ref().unwrap(), 0, 8), "prefill row 2 (P2): one 3-row slice vs 1 + 2 rows");
    } else {
        // the window's third row is the prefill's third row
        assert_eq!(top(&ca, 2, 8), top(&cb, 0, 8), "window row 2 vs prefill row 2");
    }
    // the 5-row window starts with the 3-row window
    for r in 0..3 {
        assert_eq!(top(&ca, r, 8), top(&cc, r, 8), "row {r} of the 3-row window vs the 5-row window");
    }
    // one more row each way, from the committed state
    let nx = tokens(3, 2);
    let t = Instant::now();
    let c1 = m.window_candidates(&batch(&[(1, 67, &nx[..1])]), 8).unwrap();
    let window1_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    m.commit(1, 1);
    m.flush().unwrap();
    let commit_ms = t.elapsed().as_secs_f64() * 1e3;
    let c2 = m.prefill_chunk(&batch(&[(2, 67, &nx[..1])]), Some(0)).unwrap().unwrap();
    let c3 = m.window_candidates(&batch(&[(3, 67, &nx[..1])]), 8).unwrap();
    m.commit(3, 1);
    if p2 {
        let c2r = m.prefill_chunk(&batch(&[(0, 67, &nx[..1])]), Some(0)).unwrap().unwrap();
        assert_eq!(top(&c2, 0, 8), top(&c2r, 0, 8), "the next prefilled row (P2): after one 3-row slice vs after 1 + 2 rows");
    } else {
        assert_eq!(top(&c1, 0, 8), top(&c2, 0, 8), "the next row: window vs prefill");
    }
    assert_eq!(top(&c1, 0, 8), top(&c3, 0, 8), "the next row after a 5-row window kept at 3 vs after a 3-row window");
    // a batched window equals the solo one, row by row
    // (a window's records are the last window's only: both slots commit before the next window starts)
    let c4 = m.window_candidates(&batch(&[(1, 68, &nx[1..2]), (2, 68, &nx[1..2])]), 8).unwrap();
    m.commit(1, 1);
    m.commit(2, 1);
    let c5 = m.window_candidates(&batch(&[(3, 68, &nx[1..2])]), 8).unwrap();
    m.commit(3, 1);
    m.flush().unwrap();
    assert_eq!(top(&c4, 0, 8), top(&c5, 0, 8), "batched row 0 vs solo");
    if p2 {
        // slot 2's state is the prefill's: its solo twin is slot 0, which holds the same state (checked above)
        let c6 = m.window_candidates(&batch(&[(0, 68, &nx[1..2])]), 8).unwrap();
        m.commit(0, 1);
        m.flush().unwrap();
        assert_eq!(top(&c4, 1, 8), top(&c6, 0, 8), "batched row 1 vs solo (P2: slot 0, the same prefilled state as slot 2)");
    } else {
        assert_eq!(top(&c4, 1, 8), top(&c5, 0, 8), "batched row 1 vs solo");
    }
    for ord in 0..n_mamba {
        let a = m.debug_mamba_state(ord, 1).unwrap();
        for &s in same_as_1 {
            let b = m.debug_mamba_state(ord, s).unwrap();
            assert_eq!(bits(&a.0), bits(&b.0), "S of Mamba layer {ord} after the batched window: slot 1 vs slot {s}");
            assert_eq!(a.1, b.1);
        }
        if p2 {
            let (x, y) = (m.debug_mamba_state(ord, 2).unwrap(), m.debug_mamba_state(ord, 0).unwrap());
            assert_eq!(bits(&x.0), bits(&y.0), "S of Mamba layer {ord} after the batched window (P2): slot 2 vs slot 0");
            assert_eq!(x.1, y.1);
        }
    }
    if p2 {
        // the host-contract checks below start from slot 0 holding the 64-row prefix alone, as under P1
        m.end_slot(0);
        m.begin_slot(0).unwrap();
        assert!(m.prefill_chunk(&batch(&[(0, 0, &prefix)]), None).unwrap().is_none());
        m.flush().unwrap();
    }
    // a commit after the slot's window was replaced by another is refused: slot 1's rows are gone once slot 3's window ran
    m.window_candidates(&batch(&[(1, 69, &nx[..1])]), 4).unwrap();
    m.window_candidates(&batch(&[(3, 69, &nx[..1])]), 4).unwrap();
    m.commit(1, 1);
    assert!(matches!(m.flush(), Err(ModelError::Request(_))), "slot 1 has no window once another window replaced it");
    assert_eq!(m.slot_len(1), Some(69), "and its state did not move");
    m.commit(3, 1);
    m.flush().unwrap();
    assert_eq!(m.slot_len(3), Some(70));
    min_free = min_free.min(free_gib());

    // ---- 3. the host contract
    match m.prefill_chunk(&batch(&[(0, 64, &tokens(4, 65))]), None) {
        Err(ModelError::Request(msg)) => assert!(msg.contains("1..=64 rows"), "{msg}"),
        other => panic!("a slice past prefill_rows should be a Request error, got {other:?}"),
    }
    assert!(matches!(m.begin_slot(0), Err(ModelError::Request(_))), "a double begin is refused");
    assert_eq!(m.slot_len(0), Some(64), "and leaves the slot alone");
    m.window_candidates(&batch(&[(0, 64, &nx[..1])]), 4).unwrap();
    m.commit(0, 1);
    m.commit(0, 1);
    match m.flush() {
        Err(ModelError::Request(msg)) => assert!(msg.contains("no window"), "{msg}"),
        other => panic!("a second commit should be refused and returned by flush, got {other:?}"),
    }
    m.flush().expect("the refusal is returned once");
    assert_eq!(m.slot_len(0), Some(65));
    m.window_candidates(&batch(&[(0, 65, &nx[..1])]), 4).unwrap();
    m.commit(0, 2);
    assert!(matches!(m.flush(), Err(ModelError::Request(_))), "keep past the window's rows is refused");
    let used = m.debug_mamba_state(0, 0).unwrap();
    assert!(used.0.iter().any(|&x| x != 0.0), "slot 0 holds state before it is ended");
    m.end_slot(0);
    m.end_slot(0);
    m.begin_slot(0).unwrap();
    assert_eq!(m.slot_len(0), Some(0));
    for ord in 0..n_mamba {
        let z = m.debug_mamba_state(ord, 0).unwrap();
        assert!(z.0.iter().all(|&x| x.to_bits() == 0) && z.1.iter().all(|&x| x == 0), "begin_slot zeroes layer {ord}'s state");
    }
    let untouched = m.debug_mamba_state(0, 1).unwrap();
    assert!(untouched.0.iter().any(|&x| x != 0.0), "and only that slot's");
    // chunked == whole: 63 rows and then a one-row window give the 64-row slice's last row, bit for bit (the slice ran the prefill
    // attention entry point and a 64-row MoE and projection call, the window a one-row call of each). Under P2 the last row is a
    // one-row prefill slice instead: a window row has P1's bits, the 64-row slice's last row P2's (6.14)
    m.prefill_chunk(&batch(&[(0, 0, &prefix[..63])]), None).unwrap();
    if p2 {
        let cw = m.prefill_chunk(&batch(&[(0, 63, &prefix[63..])]), Some(0)).unwrap().unwrap();
        assert_eq!(top(&c, 0, 8), top(&cw, 0, 8), "the last row of a 64-row slice vs a 63-row slice and a one-row slice (P2)");
    } else {
        let cw = m.window_candidates(&batch(&[(0, 63, &prefix[63..])]), 8).unwrap();
        assert_eq!(top(&c, 0, 8), top(&cw, 0, 8), "the last row of a 64-row slice vs a 63-row slice and a one-row window");
    }
    for s in 0..4 {
        m.end_slot(s);
    }
    m.flush().expect("no sticky error");

    // ---- 4. numbers
    min_free = min_free.min(free_gib());
    eprintln!(
        "time per step (8 layers, lm_head and top-k included where candidates are returned): 64-row prefill slice {prefill64_ms:.0} ms \
         (with taps and the wanted row's logits {tapped_ms:.0} ms); 1-row window {window1_ms:.1} ms; 3-row window {window3_ms:.1} ms; \
         3-row prefill slice with a wanted row {prefill3_ms:.1} ms; commit(1) + flush {commit_ms:.1} ms"
    );
    eprintln!(
        "device memory: {:.2} GiB used after load, {:.2} GiB at the peak sampled (of {:.1} GiB)",
        free0 as f64 / (1u64 << 30) as f64 - free1,
        free0 as f64 / (1u64 << 30) as f64 - min_free,
        total as f64 / (1u64 << 30) as f64,
    );
}
