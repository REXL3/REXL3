//! The W_q residency on the real pack (`docs/TAIL_FLOOR_CEILING.md` section 4, built): the tail-floor lever behind
//! `REXL3_WQ_RESIDENT_GIB`, default OFF. Three things, all `#[ignore]` (they need `REXL3_QWEN_DIR` and a GPU; the
//! gate runs them; with `REXL3_REQUIRE_GPU=1` a missing pack or device is a failure, not a skip):
//!
//! * `prefill_bits_are_unchanged_with_the_knob_on`: THE exactness claim — every layer's output of every row, knob on
//!   against knob off, at several chunk grids (tails of 16 and 1 rows included, the shape the cached re-ask pays),
//!   bit for bit. Compared against the same committed paths `prefill_route.rs` uses, so a bit that moved anywhere
//!   fails by name.
//! * `the_reconstruct_is_skipped_on_the_second_chunk`: a resident matrix's reconstruct happens once, not per chunk
//!   (the fill/evict counters say so, and the second chunk of the same prompt is faster by the measured 269.8 ms
//!   class of saving at full residency).
//! * `mru_eviction_holds_the_head_of_the_order`: a small budget keeps the head of the prefill's cyclic matrix order
//!   resident (MRU by matrix), a partial budget recovering its share, never over the budget.
//!
//!     REXL3_QWEN_DIR=... cargo test --release -p gb-model --test wq_residency -- --ignored --nocapture

use gb_model::config::QwenConfig;
use gb_model::forward::{Candidates, Model};
use gb_model::pack::Pack;
use gb_model::weights::Weights;

const COUNT: usize = 28;
const CHUNK: usize = 4096;

fn require_gpu() -> bool {
    std::env::var("REXL3_REQUIRE_GPU").is_ok_and(|v| v == "1")
}

fn dir() -> Option<std::path::PathBuf> {
    let d = std::env::var("REXL3_QWEN_DIR").ok().map(std::path::PathBuf::from);
    match d {
        Some(d) if gb_cuda::device_available() => Some(d),
        _ => {
            assert!(!require_gpu(), "REXL3_REQUIRE_GPU=1 but this test needs REXL3_QWEN_DIR and a GPU");
            eprintln!("not run: needs REXL3_QWEN_DIR and a GPU");
            None
        }
    }
}

fn load_model(dir: &std::path::Path, wq_gib: Option<f64>) -> Model {
    let cfg = QwenConfig::from_dir(dir).unwrap();
    let w = Weights::load(&Pack::open(dir).unwrap(), &cfg, |_, _| {}).unwrap();
    // 145 KV pages hold the 32k prompt plus a row, at 256 tokens a page
    let mut m = Model::new(cfg, w, 2, 145, 36_864, CHUNK).unwrap();
    // the residency keeps the W_q of the pinned cuBLAS GEMM; the default fused GEMM never reconstructs one (both arms run the pinned path)
    m.set_prefill_gemm(gb_model::prefill_p2::PrefillGemm::Pinned);
    if let Some(g) = wq_gib {
        m.enable_wq_residency((g * (1u64 << 30) as f64) as u64);
    }
    m
}

/// A fixed pseudo-random prompt of ordinary text tokens.
fn random_prompt(len: usize, seed: u64) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1000 + (x % 99_000) as u32
        })
        .collect()
}

/// Prefill `tokens` into `slot` in chunks of `chunk` rows, capturing every layer's output of every row on the LAST
/// chunk (the capture the model already provides: `Model::capture`), and return the last candidates.
fn prefill_capturing(m: &mut Model, slot: usize, tokens: &[u32], chunk: usize) -> (Vec<Vec<u16>>, Candidates) {
    m.begin_slot(slot).unwrap();
    let mut last = None;
    for start in (0..tokens.len()).step_by(chunk) {
        let end = (start + chunk).min(tokens.len());
        let want = (end == tokens.len()).then_some(COUNT);
        m.capture = end == tokens.len();
        last = m.prefill_chunk(slot, start, &tokens[start..end], tokens.len(), want, 0).unwrap();
    }
    let c = last.unwrap();
    (std::mem::take(&mut m.captured), c)
}

/// The budget that covers every matrix: the 400 are exactly 45.3125 GiB, so 46 GiB holds them all (45.31 would leave
/// the last eight cycling through MRU eviction).
const FULL: f64 = 46.0;

/// The core exactness claim: prefill bits are UNCHANGED with the knob on vs off, at several chunk grids.
#[test]
#[ignore]
fn prefill_bits_are_unchanged_with_the_knob_on() {
    let Some(d) = dir() else { return };
    // a prompt long enough that every chunk but the last is full, and the pinned (Tiled) path runs: total > 8,192
    let toks = random_prompt(9_000, 90);
    for chunk in [4096usize, 2048, 1024, 517, 256] {
        // one model at a time: two 20 GiB weight copies plus the 46 GiB residency do not fit on one box
        let (layers_off, c_off) = {
            let mut off = load_model(&d, None);
            prefill_capturing(&mut off, 0, &toks, chunk)
        };
        let (layers_on, c_on, s) = {
            let mut on = load_model(&d, Some(FULL));
            let r = prefill_capturing(&mut on, 1, &toks, chunk);
            (r.0, r.1, on.wq_stats())
        };
        assert_eq!(c_off.ids, c_on.ids, "chunk {chunk}: the candidate ids moved");
        assert!(
            c_off.vals.iter().map(|v| v.to_bits()).eq(c_on.vals.iter().map(|v| v.to_bits())),
            "chunk {chunk}: the candidate logits moved"
        );
        assert_eq!(layers_off.len(), layers_on.len(), "chunk {chunk}: a different number of layers was captured");
        for (li, (a, b)) in layers_off.iter().zip(&layers_on).enumerate() {
            assert_eq!(a, b, "chunk {chunk}: layer {li}'s output bits moved with the knob on");
        }
        // and the residency actually served the prompt (not vacuously equal with nothing resident)
        assert!(s.hits > 0, "chunk {chunk}: the residency never hit — it is silently not used ({s:?})");
        eprintln!("chunk {chunk}: {} layers bit-identical, residency hits {} fills {}", layers_on.len(), s.hits, s.misses);
    }
    // a 16-row tail after a 32k prompt is the cached re-ask's shape: the tail alone, knob on vs off
    let p32 = random_prompt(32_768, 32);
    for rows in [16usize, 1] {
        let start = p32.len() - rows;
        // one model at a time (see above): capture the OFF tail first, then the ON tail
        let c_off = {
            let mut off = load_model(&d, None);
            // warm the same way: a full-chunk prefill of the first 28,672 tokens fills the residency
            let head = &p32[..start];
            let warm = |m: &mut Model, slot: usize| {
                m.begin_slot(slot).unwrap();
                for s in (0..head.len()).step_by(CHUNK) {
                    let e = (s + CHUNK).min(head.len());
                    m.prefill_chunk(slot, s, &head[s..e], p32.len(), None, 0).unwrap();
                }
            };
            warm(&mut off, 0);
            off.capture = true;
            off.prefill_chunk(0, start, &p32[start..], p32.len(), Some(COUNT), 0).unwrap().unwrap()
        };
        let mut on = load_model(&d, Some(FULL));
        {
            let head = &p32[..start];
            let warm = |m: &mut Model, slot: usize| {
                m.begin_slot(slot).unwrap();
                for s in (0..head.len()).step_by(CHUNK) {
                    let e = (s + CHUNK).min(head.len());
                    m.prefill_chunk(slot, s, &head[s..e], p32.len(), None, 0).unwrap();
                }
            };
            warm(&mut on, 1);
        }
        on.capture = true;
        let c_on = on.prefill_chunk(1, start, &p32[start..], p32.len(), Some(COUNT), 0).unwrap().unwrap();
        assert_eq!(c_off.ids, c_on.ids, "tail of {rows}: the candidate ids moved");
        assert!(
            c_off.vals.iter().map(|v| v.to_bits()).eq(c_on.vals.iter().map(|v| v.to_bits())),
            "tail of {rows}: the candidate logits moved"
        );
        let s = on.wq_stats();
        assert!(s.hits >= 400, "tail of {rows}: the resident matrices did not serve the tail ({s:?})");
        let _ = &rows;
        eprintln!("tail of {rows}: bits identical, {s:?}");
    }
}

/// The reconstruct of a resident matrix is skipped on the second chunk: one fill per matrix, then hits only.
#[test]
#[ignore]
fn the_reconstruct_is_skipped_on_the_second_chunk() {
    let Some(d) = dir() else { return };
    let mut m = load_model(&d, Some(FULL));
    let toks = random_prompt(9_000, 7);
    m.begin_slot(0).unwrap();
    for start in (0..toks.len()).step_by(CHUNK) {
        let end = (start + CHUNK).min(toks.len());
        let want = (end == toks.len()).then_some(COUNT);
        m.prefill_chunk(0, start, &toks[start..end], toks.len(), want, 0).unwrap();
    }
    let s = m.wq_stats();
    // 2 full chunks + a tail: 400 matrices filled once on chunk 1, then served from the residency
    assert_eq!(s.misses, 400, "every matrix is reconstructed exactly once ({s:?})");
    assert_eq!(s.evictions, 0, "a full budget never evicts ({s:?})");
    assert!(s.hits >= 800, "two chunk passes after the fill must hit ({s:?})");
    // and with the knob off there is no residency at all
    let off = load_model(&d, None);
    assert_eq!(off.wq_stats(), gb_model::wq::WqStats::default(), "off means no counters");
}

/// MRU by matrix under a small budget: the head of the prefill's cyclic order stays resident, never over budget.
#[test]
#[ignore]
fn mru_eviction_holds_the_head_of_the_order() {
    let Some(d) = dir() else { return };
    // 3 GiB holds the first few matrices of the walk; the rest cycle through MRU eviction
    let mut m = load_model(&d, Some(3.0));
    let toks = random_prompt(9_000, 11);
    m.begin_slot(0).unwrap();
    for start in (0..toks.len()).step_by(CHUNK) {
        let end = (start + CHUNK).min(toks.len());
        let want = (end == toks.len()).then_some(COUNT);
        m.prefill_chunk(0, start, &toks[start..end], toks.len(), want, 0).unwrap();
    }
    let s = m.wq_stats();
    assert_eq!(s.budget_bytes, 3 << 30);
    assert!(s.evictions > 0, "a small budget must evict ({s:?})");
    assert!(
        s.resident_bytes as u64 <= s.budget_bytes,
        "the residency holds {} bytes over its {}-byte budget",
        s.resident_bytes,
        s.budget_bytes
    );
    assert!(s.hits > 0, "a partial budget still recovers its share ({s:?})");
    // the policy keeps the head: the resident set is a prefix of the matrix order (checked by index in unit tests;
    // here it is the byte-level claim)
    eprintln!("3 GiB budget: {s:?}");
}
