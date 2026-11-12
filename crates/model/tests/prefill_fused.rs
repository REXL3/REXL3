//! The fused tensor-core GEMM for the prompt rows of the 27B model (`gb_model::prefill_p2`, `docs/QWEN_PREFILL_FUSED.md`), on the real pack.
//! All `#[ignore]` (they need `REXL3_QWEN_DIR` and a GPU; `REXL3_REQUIRE_GPU=1` turns a skip into a failure). The chunk-grid invariance and
//! `chunked_prefill_equals_whole` of the default path are `prefill_route.rs` (it builds its model with the process's default, the fused GEMM).
//!
//! * `only_prompt_rows_of_a_long_prompt_take_the_fused_gemm`: `glm_p2::launches()` before and after: a prompt of more than 128 tokens runs
//!   the pack's 400 dense linears once per chunk through the fused GEMM (a tail chunk of 1 row too); a prompt of 128 tokens or fewer, a decode
//!   window and the opt-out launch none.
//! * `fused_and_pinned_prefill_are_bit_identical_on_the_real_model`: one load, the same prompts on both paths: the next-token candidates
//!   (ids and logit bits) are equal. The per-layer, per-row identity is the cuda test's, and `golden_layers.txt` holds unchanged.
//!
//!     REXL3_QWEN_DIR=... cargo test --release -p gb-model --test prefill_fused -- --ignored --nocapture --test-threads=1

use gb_engine::RowRequest;
use gb_model::config::QwenConfig;
use gb_model::forward::{Candidates, Model};
use gb_model::pack::Pack;
use gb_model::prefill_p2::PrefillGemm;
use gb_model::rows::RowBatch;
use gb_model::weights::Weights;

const CHUNK: usize = 4096;
/// Dense EXL3 linears per chunk of the 64-layer model: 48 linear-attention layers x (qkv, z, out) + 16 full-attention layers x (q, k, v, o)
/// + 64 MLPs x (gate, up, down).
const LINEARS_PER_CHUNK: u64 = 48 * 3 + 16 * 4 + 64 * 3;

fn load_model() -> Option<Model> {
    let d = std::path::PathBuf::from(std::env::var("REXL3_QWEN_DIR").ok()?);
    if !gb_cuda::device_available() {
        assert!(std::env::var("REXL3_REQUIRE_GPU").map_or(true, |v| v != "1"), "REXL3_REQUIRE_GPU=1 but there is no device");
        return None;
    }
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
    Some(Model::new(cfg, w, 2, 160, 24576, CHUNK).unwrap())
}

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

/// Prefill `tokens` (the whole prompt) into slot 0 in chunks of the given lengths; the last chunk's candidates.
fn prefill(m: &mut Model, tokens: &[u32], chunks: &[usize]) -> Candidates {
    assert_eq!(chunks.iter().sum::<usize>(), tokens.len());
    m.begin_slot(0).unwrap();
    let (mut start, mut cand) = (0, None);
    for (i, &n) in chunks.iter().enumerate() {
        let last = i + 1 == chunks.len();
        cand = m.prefill_chunk(0, start, &tokens[start..start + n], tokens.len(), last.then_some(64), 0).unwrap();
        start += n;
    }
    m.sync().unwrap();
    cand.unwrap()
}

#[test]
#[ignore]
fn only_prompt_rows_of_a_long_prompt_take_the_fused_gemm() {
    let Some(mut m) = load_model() else { return eprintln!("needs REXL3_QWEN_DIR and a GPU: skipping") };
    assert_eq!(m.prefill_gemm(), PrefillGemm::Fused, "the default of this build (is REXL3_QWEN_PREFILL_GEMM set?)");
    let launches = || gb_cuda::glm_p2::launches();

    // 129 tokens: the shortest prompt on the GEMM path; one chunk, then in pieces of 128 and 1 (a tail of one row)
    let t = random_prompt(129, 1);
    let before = launches();
    prefill(&mut m, &t, &[129]);
    assert_eq!(launches() - before, LINEARS_PER_CHUNK, "one 129-row chunk");
    let before = launches();
    prefill(&mut m, &t, &[128, 1]);
    assert_eq!(launches() - before, 2 * LINEARS_PER_CHUNK, "a chunk of 128 rows and a tail of 1, both the fused GEMM");
    let t = random_prompt(4200, 2);
    let before = launches();
    prefill(&mut m, &t, &[4096, 104]);
    assert_eq!(launches() - before, 2 * LINEARS_PER_CHUNK, "the server's grid of a 4200-token prompt");

    // 128 tokens: the decode GEMV, as before
    let t = random_prompt(128, 3);
    let before = launches();
    prefill(&mut m, &t, &[128]);
    assert_eq!(launches() - before, 0, "a prompt of 128 tokens stays on the decode GEMV");

    // a decode window never comes here
    let before = launches();
    let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: vec![5, 6, 7, 8], first_position: 128 }]);
    m.window_candidates(&b, 8).unwrap();
    m.sync().unwrap();
    assert_eq!(launches() - before, 0, "a window uses the decode kernel");

    // the opt-out launches none
    m.set_prefill_gemm(PrefillGemm::Pinned);
    let t = random_prompt(300, 4);
    let before = launches();
    prefill(&mut m, &t, &[300]);
    assert_eq!(launches() - before, 0, "the pinned path");
}

#[test]
#[ignore]
fn fused_and_pinned_prefill_are_bit_identical_on_the_real_model() {
    let Some(mut m) = load_model() else { return eprintln!("needs REXL3_QWEN_DIR and a GPU: skipping") };
    for (len, seed) in [(300usize, 11u64), (1500, 12), (4200, 13)] {
        let t = random_prompt(len, seed);
        let grid: Vec<usize> = if len > CHUNK { vec![CHUNK, len - CHUNK] } else { vec![len] };
        m.set_prefill_gemm(PrefillGemm::Pinned);
        let p = prefill(&mut m, &t, &grid);
        let before = gb_cuda::glm_p2::launches();
        m.set_prefill_gemm(PrefillGemm::Fused);
        let f = prefill(&mut m, &t, &grid);
        assert!(gb_cuda::glm_p2::launches() > before, "the fused GEMM ran");
        assert_eq!(p.ids, f.ids, "len {len}: candidate ids");
        assert_eq!(
            p.vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            f.vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "len {len}: candidate logits"
        );
        println!(
            "len {len:5}: top-1 {} ({:.3}), {} candidates bit-identical on the pinned and the fused GEMM",
            f.ids[0],
            f.vals[0],
            f.ids.len()
        );
    }
}
