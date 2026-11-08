//! Controller-owned native CUDA gate. CPU lane compiles this test but NEVER executes --ignored.
//! No vacuous pass: explicitly running it requires a real pack and CUDA device.
use gb_engine::score::{ScoreRequest, Scored};
use gb_engine::{Backend, Request};
use gb_model::serve::{build_engine, PrefixCacheOption, QwenEngine, ServeOptions};
use gb_sampling::Sampling;
fn score(e: &mut QwenEngine, prompt: Vec<u32>, cache: bool) -> Scored {
    let id = e.score_submit(ScoreRequest { prompt, cache, all_rows: false }).unwrap();
    e.run().unwrap();
    let rows = e.take_scores();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0, id);
    assert!(e.take_events().is_empty());
    assert!(e.take_outputs().is_empty());
    rows.into_iter().next().unwrap().1
}
fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|v| v.to_bits()).collect()
}
#[test]
#[ignore = "controller GPU admission required"]
fn full_row_current_request_cache_tail_and_chunk_identity() {
    assert!(gb_cuda::device_available(), "controller gate requires CUDA, do not count a stub run as qualification");
    let dir = std::env::var("REXL3_QWEN_DIR").expect("controller must set REXL3_QWEN_DIR");
    let mut o = ServeOptions::new(dir.into());
    o.slots = 2;
    o.max_model_len = 12288;
    o.kv_tokens = Some(32768);
    o.prefill_chunk = 4096;
    o.vision = None;
    o.prefix_cache = PrefixCacheOption::Mib(1024);
    let mut e = build_engine(&o).unwrap();
    assert!(e.score_supported());
    let v = e.backend.vocab_size();
    for n in [1, 127, 128, 129, 4095, 4096, 4097, 8193] {
        let p: Vec<u32> = (0..n).map(|i| 1000 + ((i * 31 + i / 7) % 99000) as u32).collect();
        eprintln!("NATIVE_SCORE_DIAGNOSTIC n={n} phase=score");
        let a = score(&mut e, p.clone(), false);
        assert_eq!(a.logits_row.len(), v);
        assert!(a.logits_row.iter().all(|v| v.is_finite()));
        let (top, values) = gb_sampling::top_candidates(&a.logits_row, 64);
        // Independently execute the existing candidate-prefill head: identity of row zero, not stale
        // generated logits or an index derived from chunk row count.
        eprintln!("NATIVE_SCORE_DIAGNOSTIC n={n} phase=independent_prefill");
        let candidate = e.backend.prefill(0, 0, &p, 64);
        assert!(e.backend.take_error().is_none());
        e.backend.release(0);
        assert!(e.backend.take_error().is_none(), "n={n} independent release set error");
        eprintln!("NATIVE_SCORE_DIAGNOSTIC n={n} phase=generate");
        assert_eq!(candidate.ids, top);
        assert_eq!(bits(&candidate.values), bits(&values));
        let id = e
            .submit(Request {
                prompt: p.clone(),
                max_new: 1,
                // The backend keeps 64 candidates INCLUDING the tie margin.
                // Greedy's argmax is unchanged; top_k=64 would request 72 CUDA candidates.
                sampling: Sampling { temperature: 0.0, top_k: 64 - gb_sampling::MARGIN, ..Default::default() },
                stop: vec![],
                draft: false,
                images: vec![],
                cache: false,
            })
            .unwrap();
        e.run().unwrap();
        let generated = e.take_outputs();
        assert_eq!(generated[0].id, id);
        assert_eq!(generated[0].tokens, vec![top[0]]);
        e.take_events();
        let _ = score(&mut e, p.clone(), true);
        let hit = score(&mut e, p.clone(), true);
        if n == 8193 {
            assert!(hit.cached_tokens > 0, "live cache gate must actually hit");
        }
        assert_eq!(bits(&a.logits_row), bits(&hit.logits_row));
        // Another prompt replaces old row zero; the resumed original must run its OWN head again.
        let mut other = p.clone();
        other[n - 1] = 1234;
        let b = score(&mut e, other.clone(), false);
        assert_eq!(bits(&b.logits_row), bits(&score(&mut e, other, true).logits_row));
        // Every pass of a prompt over 128 tokens (the tiled path) reconstructs every W_q: about 0.5 s on a GB10 whatever its row
        // count (measured, constant from 1 to 129 rows), so chunk=1 costs n * 0.5 s and the sweep [1, 17, 1024, 4096] over
        // n = 1..8193 takes about 3.1 hours. chunk=1 is kept where the prompt is short (n <= 129, which also covers the GEMV path);
        // above that, chunk = n - 1 (up to the engine's 4096-row chunk) puts the same single-row final chunk through the same path
        // in two passes (n = 4097 and 8193 already end in a single row at chunk 4096), and 17 still cuts every prompt into 241 to
        // 482 chunks with a ragged tail.
        let mut chunks = vec![1, 17, 1024, 4096];
        if n > 129 {
            chunks.retain(|&c| c != 1);
            if n - 1 <= 4096 && !chunks.contains(&(n - 1)) {
                chunks.push(n - 1);
            }
        }
        for chunk in chunks {
            e.backend.chunk = chunk;
            eprintln!("NATIVE_SCORE_DIAGNOSTIC n={n} phase=chunk_identity chunk={chunk}");
            assert_eq!(bits(&a.logits_row), bits(&score(&mut e, p.clone(), false).logits_row), "n={n} chunk={chunk}");
        }
        e.backend.chunk = 4096;
    }
}
