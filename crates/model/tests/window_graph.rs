//! The CUDA graphs of the Qwen decode window on the real model: bit preservation and the mutants.
//!
//!     REXL3_QWEN_DIR=... cargo test --release -p gb-model --test window_graph -- --ignored --nocapture
//!
//! `window_graph_bits` runs the same scripted rounds twice on one model — graphs off (every window eager) and
//! graphs on (warmup, capture, replay) — and requires every candidate row of every round and the whole recurrent
//! state afterwards to be bit for bit equal, with the stats proving the graph path actually ran (captures and
//! replays, no failure). `window_graph_mutants` breaks the path on purpose: a replay that does not upload this
//! window's rows reads the previous window's and must change the candidates; a host synchronisation inside the
//! capture must be refused (eager fallback, bits unchanged, no fatal error).
//!
//! The script is `deferred_commit`'s shape (several slots, windows of 1 to 8 rows, partial keeps) plus a fixed
//! 1-row AR decode pattern, so the cache sees several shapes, the record-parity alternation and the owed fold.

use gb_cuda::gdn::{CONV_SLOT_ELEMS, S_SLOT_ELEMS};
use gb_engine::RowRequest;
use gb_model::config::QwenConfig;
use gb_model::forward::{Candidates, Model};
use gb_model::pack::Pack;
use gb_model::rows::RowBatch;
use gb_model::weights::Weights;
use gb_model::window_graph::WindowGraphMutant;

const COUNT: usize = 28;
const SLOTS: usize = 3;
const ROUNDS: usize = 48;

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

fn hash(c: &Candidates) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for (i, v) in c.ids.iter().zip(&c.vals) {
        fnv(&mut h, &i.to_le_bytes());
        fnv(&mut h, &v.to_bits().to_le_bytes());
    }
    h
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn tokens(rng: &mut Lcg, n: usize) -> Vec<u32> {
    (0..n).map(|_| 1000 + rng.below(99_000) as u32).collect()
}

/// Hash of the recurrent state of the first `slots` slots in every linear layer, one hash per (layer, S or conv).
fn state_hashes(m: &Model, slots: usize) -> Vec<u64> {
    let mut out = Vec::new();
    for li in 0..m.pools.s.len() {
        let s = m.pools.s[li].to_vec::<f32>().unwrap();
        let mut h = 0xcbf29ce484222325u64;
        for x in &s[..slots * S_SLOT_ELEMS] {
            fnv(&mut h, &x.to_bits().to_le_bytes());
        }
        out.push(h);
        let c = m.pools.conv[li].to_vec::<u16>().unwrap();
        let mut h = 0xcbf29ce484222325u64;
        for x in &c[..slots * CONV_SLOT_ELEMS] {
            fnv(&mut h, &x.to_le_bytes());
        }
        out.push(h);
    }
    out
}

/// The same random rounds every call: windows over a random subset of the slots (1..=8 rows each, partial keeps,
/// the owed fold of the previous window), interleaved with a fixed 1-row AR window on slot 0 — the fixed shapes
/// the graph cache is for. Returns every window's candidate hash and the state hashes afterwards.
fn run(m: &mut Model) -> (Vec<u64>, Vec<u64>) {
    let mut rng = Lcg(0x9E3779B97F4A7C15);
    let mut pos = [0usize; SLOTS];
    for (s, p) in pos.iter_mut().enumerate() {
        let prompt = tokens(&mut rng, 140 + 33 * s);
        m.begin_slot(s).unwrap();
        m.prefill_chunk(s, 0, &prompt, prompt.len(), None, 0).unwrap();
        *p = prompt.len();
    }
    let mut hashes = Vec::new();
    for r in 0..ROUNDS {
        if r % 3 == 0 {
            // the AR shape: one row of slot 0, the serial decode path
            let reqs = vec![RowRequest { slot: 0, tokens: tokens(&mut rng, 1), first_position: pos[0] as u64 }];
            let c = m.window_candidates(&RowBatch::from_requests(&reqs), COUNT).unwrap();
            hashes.push(hash(&c));
            m.commit(0, 1);
            pos[0] += 1;
        }
        let mut order: Vec<usize> = (0..SLOTS).collect();
        for i in (1..SLOTS).rev() {
            order.swap(i, rng.below(i + 1));
        }
        order.truncate(1 + rng.below(SLOTS));
        let reqs: Vec<RowRequest> = order
            .iter()
            .map(|&slot| {
                let n = 1 + rng.below(8);
                RowRequest { slot, tokens: tokens(&mut rng, n), first_position: pos[slot] as u64 }
            })
            .collect();
        let keeps: Vec<usize> = reqs.iter().map(|r| 1 + rng.below(r.tokens.len())).collect();
        let c = m.window_candidates(&RowBatch::from_requests(&reqs), COUNT).unwrap();
        hashes.push(hash(&c));
        for (r, &k) in reqs.iter().zip(&keeps) {
            m.commit(r.slot, k);
            pos[r.slot] += k;
        }
    }
    m.settle().unwrap();
    (hashes, state_hashes(m, SLOTS))
}

fn model() -> Option<Model> {
    let Ok(d) = std::env::var("REXL3_QWEN_DIR") else {
        eprintln!("REXL3_QWEN_DIR not set: skipping");
        return None;
    };
    if !gb_cuda::device_available() {
        assert!(
            !std::env::var("REXL3_REQUIRE_GPU").is_ok_and(|v| v == "1"),
            "REXL3_REQUIRE_GPU=1 but window_graph found no CUDA device"
        );
        eprintln!("no CUDA device: not run");
        return None;
    }
    let d = std::path::PathBuf::from(d);
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
    Model::new(cfg, w, 6, 96, 16384, 4096).ok()
}

#[test]
#[ignore]
fn window_graph_bits() {
    let Some(mut m) = model() else { return };
    m.set_graphs(false);
    assert!(!m.graphs_on());
    let (off, state_off) = run(&mut m);
    m.set_graphs(true);
    assert!(m.graphs_on(), "the capture stream exists, so graphs can be switched on");
    let before = m.graph_stats();
    let (on, state_on) = run(&mut m);
    let st = m.graph_stats();
    eprintln!(
        "GRAPHS captures {} replays {} eager {} failures {} capture {:.1} ms last failure {:?}",
        st.captures - before.captures,
        st.replays,
        st.eager_windows,
        st.capture_failures,
        st.capture_ms,
        st.last_failure
    );
    assert_eq!(st.capture_failures, 0, "a capture failed: {:?}", st.last_failure);
    assert!(st.captures > before.captures, "nothing was captured: the graph path never ran");
    assert!(st.replays > 0, "graphs are on but nothing was replayed");
    let first = off.iter().zip(&on).position(|(a, b)| a != b);
    assert_eq!(first, None, "candidates differ at window {first:?} of {}", off.len());
    let bad = state_off.iter().zip(&state_on).filter(|(a, b)| a != b).count();
    assert_eq!(bad, 0, "{bad} of {} state hashes differ", state_off.len());
    eprintln!("{} windows: candidates and all {} state hashes identical with graphs on", on.len(), state_on.len());
}

/// The graph path's invariants, each broken on purpose: a replay that leaves the window's rows stale must change
/// the candidates (the test fails if it does not — the replays are real); a capture with a host synchronisation
/// inside must be refused and fall back to eager launches with the bits unchanged, without a fatal error.
#[test]
#[ignore]
fn window_graph_mutants() {
    let Some(mut m) = model() else { return };
    m.set_graphs(false);
    let want = run(&mut m).0;
    m.clear_graphs().unwrap();
    m.set_graphs(true);
    let mut missed = Vec::new();
    for name in ["stale_rows", "sync_in_capture"] {
        m.clear_graphs().unwrap();
        m.set_window_graph_mutant(Some(match name {
            "stale_rows" => WindowGraphMutant::StaleRows,
            _ => WindowGraphMutant::SyncInCapture,
        }));
        let before = m.graph_stats();
        let got = run(&mut m).0;
        m.set_window_graph_mutant(None);
        let st = m.graph_stats();
        let differs = got != want;
        let replays = st.replays - before.replays;
        eprintln!(
            "MUTANT {name}: digest {} (captures {}, replays {}, failures {})",
            if differs { "DIFFERS (detected)" } else { "equal" },
            st.captures - before.captures,
            replays,
            st.capture_failures - before.capture_failures
        );
        match name {
            "stale_rows" => {
                assert!(replays > 0, "the stale-rows mutant saw no replay: nothing to detect");
                if !differs {
                    missed.push(name);
                }
            }
            _ => {
                assert!(!differs, "a refused capture changed the bits");
                assert!(
                    st.capture_failures > before.capture_failures && st.captures == before.captures,
                    "every capture with a host sync inside must be refused"
                );
            }
        }
    }
    assert!(missed.is_empty(), "graph mutants not detected by the candidates: {missed:?}");
}
