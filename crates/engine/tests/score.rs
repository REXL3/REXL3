use gb_engine::mock::{mock_logits, Fault, MockBackend};
use gb_engine::score::ScoreRequest;
use gb_engine::{Engine, EngineConfig, NoDrafts, PrefillPolicy, Request};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};
fn engine(chunk: usize) -> Engine<MockBackend> {
    Engine::new(
        MockBackend::new(256, 2, Fault::None).with_prefill_slice(chunk),
        Admission::new(
            4096,
            0,
            ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 0, activation_reserve: 0 },
            2,
        )
        .unwrap()
        .with_max_context(256),
        EngineConfig { max_rows: 64, max_drafts: 7, cost: CostModel::from_table(vec![(1, 1.0), (64, 2.0)]), drafting: true },
        Box::new(|| Box::new(NoDrafts)),
    )
}
#[test]
fn score_is_full_row_prompt_only_and_no_generation() {
    let mut e = engine(3);
    let before = e.spec_stats().clone();
    let p = vec![1; 256];
    let id = e.score_submit(ScoreRequest { prompt: p.clone(), cache: false, all_rows: false }).unwrap();
    while e.busy() {
        e.step().unwrap();
    }
    let rows = e.take_scores();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0, id);
    assert_eq!(rows[0].1.logits_row, mock_logits(&p, 256));
    assert_eq!(&before, e.spec_stats(), "score must not alter any speculation counters");
    assert!(e.take_events().is_empty());
    assert!(e.take_outputs().is_empty());
}
/// `all_rows`: one logits row per prompt token, row `i` the logits after tokens `0..=i` (the prediction of token `i + 1`), the
/// last one the request's own row; the same rows whether the prompt is fed whole or in slices; and a request without
/// `all_rows` gets none.
#[test]
fn all_rows_returns_the_row_after_each_prompt_token() {
    let p: Vec<u32> = (0..23u32).map(|i| (i * 37 + 11) % 256).collect();
    let run = |chunk: usize, all_rows: bool| {
        let mut e = engine(chunk);
        e.score_submit(ScoreRequest { prompt: p.clone(), cache: false, all_rows }).unwrap();
        while e.busy() {
            e.step().unwrap();
        }
        e.take_scores().remove(0).1
    };
    let whole = run(64, true);
    assert_eq!(whole.logits_rows.len(), p.len(), "one row per prompt token");
    for (i, row) in whole.logits_rows.iter().enumerate() {
        assert_eq!(row, &mock_logits(&p[..=i], 256), "row {i} follows tokens 0..={i}");
    }
    assert_eq!(whole.logits_rows.last(), Some(&whole.logits_row));
    assert_eq!(run(5, true).logits_rows, whole.logits_rows, "slicing the prefill does not change the rows");
    assert!(run(64, false).logits_rows.is_empty());
}
#[test]
fn chunking_bits_and_cancel_reclaim() {
    let p = vec![3; 31];
    let mut whole = engine(64);
    let mut cut = engine(2);
    whole.score_submit(ScoreRequest { prompt: p.clone(), cache: false, all_rows: false }).unwrap();
    let id = cut.score_submit(ScoreRequest { prompt: p.clone(), cache: false, all_rows: false }).unwrap();
    cut.step().unwrap();
    assert!(cut.cancel(id));
    assert!(!cut.busy());
    assert!(cut.take_events().is_empty());
    assert!(cut.take_outputs().is_empty());
    cut.score_submit(ScoreRequest { prompt: p, cache: false, all_rows: false }).unwrap();
    while whole.busy() {
        whole.step().unwrap();
    }
    while cut.busy() {
        cut.step().unwrap();
    }
    assert_eq!(whole.take_scores()[0].1.logits_row, cut.take_scores()[0].1.logits_row);
}
#[test]
fn unsupported_fails_before_reservation_or_backend_calls() {
    use gb_engine::Backend;
    let mut e = engine(3);
    e.backend = e.backend.with_glm_shape();
    let before = e.admission().available();
    assert!(!e.score_supported());
    assert!(e
        .score_submit(ScoreRequest { prompt: vec![1; 200], cache: true, all_rows: false })
        .unwrap_err()
        .to_string()
        .contains("does not support decisions scoring"));
    assert_eq!(e.admission().available(), before);
    assert!(!e.busy());
    assert!(e.backend.score_slice(0, &[1, 2], 0, &[]).is_err());
}
#[test]
fn score_validation_and_admission_are_prompt_only() {
    let mut e = engine(2);
    let before = e.admission().available();
    for p in [vec![], vec![256], vec![1; 257]] {
        assert!(e.score_submit(ScoreRequest { prompt: p, cache: true, all_rows: false }).is_err());
        assert_eq!(e.admission().available(), before);
    }
    let r = Request {
        prompt: vec![1; 256],
        sampling: Sampling::default(),
        max_new: 1,
        stop: vec![],
        draft: false,
        images: vec![],
        cache: false,
    };
    assert!(e.validate(&r).is_err());
    let id = e.score_submit(ScoreRequest { prompt: vec![1; 256], cache: false, all_rows: false }).unwrap();
    assert_eq!(before - e.admission().available(), 256);
    assert!(e.cancel(id));
    assert_eq!(e.admission().available(), before);
}
#[test]
fn cached_row_bits_tail_identity_and_cancel_storm_books() {
    use gb_engine::mock::PrefixFault;
    use gb_engine::{CacheBudget, CutPolicy, PrefixConfig};
    let mut e = engine(17);
    e.backend = e.backend.with_path_dependent_state().with_prefix_cache(
        CacheBudget { snapshot_slots: 4, page_blocks: 32, snapshot_bytes: 128, page_bytes: 16 },
        PrefixFault::None,
    );
    let mut e = e.with_prefix_cache(PrefixConfig {
        policy: CutPolicy { min_cut: 64, tail: 8, ghost_min_gain: 256, min_gap: 0 },
        ghost_blocks: 4096,
        ..Default::default()
    });
    let p: Vec<u32> = (0..240).map(|i| 1 + ((i * 17) % 254) as u32).collect();
    let run = |e: &mut Engine<MockBackend>, p: Vec<u32>, cache: bool| {
        e.score_submit(ScoreRequest { prompt: p, cache, all_rows: false }).unwrap();
        while e.busy() {
            e.step().unwrap();
        }
        e.take_scores().remove(0).1
    };
    let cold = run(&mut e, p.clone(), false);
    let _ = run(&mut e, p.clone(), true);
    let hit = run(&mut e, p.clone(), true);
    assert_eq!(cold.logits_row, hit.logits_row);
    assert!(hit.cached_tokens > 0);
    let mut changed = p.clone();
    changed[239] = 77;
    let changed_cold = run(&mut e, changed.clone(), false);
    assert_eq!(run(&mut e, changed.clone(), true).logits_row, changed_cold.logits_row);
    e.backend = e.backend.with_prefill_slice(1);
    let available = e.admission().available();
    for i in 0..64 {
        let id = e.score_submit(ScoreRequest { prompt: p.clone(), cache: true, all_rows: false }).unwrap();
        if i % 2 == 0 {
            e.step().unwrap();
        }
        assert!(e.cancel(id));
        assert_eq!(e.admission().available(), available);
        assert!(!e.busy());
    }
    assert!(e.take_events().is_empty());
    assert!(e.take_outputs().is_empty());
    assert!(e.take_scores().is_empty());
    assert_eq!(run(&mut e, p.clone(), true).logits_row, cold.logits_row);
}
#[test]
fn score_and_generate_share_slice_turns() {
    let mut e = engine(2).with_prefill_policy(PrefillPolicy::Serial);
    e.score_submit(ScoreRequest { prompt: vec![1; 80], cache: false, all_rows: false }).unwrap();
    let chat = e
        .submit_deferred(Request {
            prompt: vec![2; 2],
            sampling: Sampling::default(),
            max_new: 8,
            stop: vec![],
            draft: false,
            images: vec![],
            cache: false,
        })
        .unwrap();
    e.step().unwrap();
    e.step().unwrap();
    assert!(e.take_events().iter().any(|ev| ev.id == chat && !ev.tokens.is_empty()));
    e.step().unwrap();
    assert!(e.take_events().iter().any(|ev| ev.id == chat && !ev.tokens.is_empty()));
}

fn burst_engine() -> Engine<MockBackend> {
    Engine::new(
        MockBackend::new(256, 6, Fault::None).with_prefill_slice(128),
        Admission::new(
            4096,
            0,
            ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 0, activation_reserve: 0 },
            6,
        )
        .unwrap()
        .with_max_context(256),
        EngineConfig { max_rows: 64, max_drafts: 7, cost: CostModel::from_table(vec![(1, 1.0), (64, 2.0)]), drafting: true },
        Box::new(|| Box::new(NoDrafts)),
    )
    .with_prefill_policy(PrefillPolicy::Burst { div: 2, cap: 16 })
}
#[test]
fn scored_burst_yields_after_one_head_while_chat_decodes() {
    let mut e = burst_engine();
    let chat = e
        .submit_deferred(Request {
            prompt: vec![2; 2],
            sampling: Sampling::default(),
            max_new: 8,
            stop: vec![],
            draft: false,
            images: vec![],
            cache: false,
        })
        .unwrap();
    e.step().unwrap();
    e.take_events();
    for token in [1, 3, 4] {
        e.score_submit(ScoreRequest { prompt: vec![token; 80], cache: false, all_rows: false }).unwrap();
    }
    e.step().unwrap();
    assert!(e.take_events().iter().any(|event| event.id == chat && !event.tokens.is_empty()));
    let rows = e.take_scores();
    assert_eq!(rows.len(), 1, "multiple full-vocabulary heads must not delay the next decode turn");
    let mut all = rows;
    while e.busy() {
        e.step().unwrap();
        all.extend(e.take_scores());
    }
    assert_eq!(all.len(), 3);
    for (_, scored) in all {
        assert!(scored.logits_row.iter().all(|value| value.is_finite()));
    }
    assert_eq!(e.take_outputs().len(), 1);
}
#[test]
fn score_only_burst_retains_original_batch_budget() {
    let mut e = burst_engine();
    for token in [1, 3, 4] {
        e.score_submit(ScoreRequest { prompt: vec![token; 80], cache: false, all_rows: false }).unwrap();
    }
    e.step().unwrap();
    assert_eq!(e.take_scores().len(), 2, "do not serialize score-only batches without a generation contender");
}

#[test]
fn queued_short_generation_is_not_held_behind_scored_burst() {
    let mut e = burst_engine();
    for token in [1, 3, 4] {
        e.score_submit(ScoreRequest { prompt: vec![token; 80], cache: false, all_rows: false }).unwrap();
    }
    let chat = e
        .submit_deferred(Request {
            prompt: vec![2; 2],
            sampling: Sampling::default(),
            max_new: 8,
            stop: vec![],
            draft: false,
            images: vec![],
            cache: false,
        })
        .unwrap();
    e.step().unwrap();
    assert!(e.take_events().iter().any(|event| event.id == chat && !event.tokens.is_empty()));
    assert!(e.take_scores().is_empty(), "short generation gets its first turn before the full heads");
}
#[test]
fn generation_only_burst_retains_original_batch_budget() {
    let mut e = burst_engine();
    for token in [1, 3, 4] {
        e.submit_deferred(Request {
            prompt: vec![token; 80],
            sampling: Sampling::default(),
            max_new: 1,
            stop: vec![],
            draft: false,
            images: vec![],
            cache: false,
        })
        .unwrap();
    }
    e.step().unwrap();
    assert_eq!(e.take_outputs().len(), 2, "ordinary generation keeps its original burst budget");
}

#[test]
fn scored_prefill_rows_are_bounded_when_generation_competes_and_limit_resets() {
    let mut e = burst_engine();
    e.backend = e.backend.with_prefill_slice(4096);
    let chat = e
        .submit_deferred(Request {
            prompt: vec![2; 2],
            sampling: Sampling::default(),
            max_new: 8,
            stop: vec![],
            draft: false,
            images: vec![],
            cache: false,
        })
        .unwrap();
    e.step().unwrap();
    e.take_events();
    let p = vec![1; 240];
    e.score_submit(ScoreRequest { prompt: p.clone(), cache: false, all_rows: false }).unwrap();
    e.step().unwrap();
    assert!(e.take_scores().is_empty(), "a competing score must not run its entire240-row prefill in one decode turn");
    assert!(e.take_events().iter().any(|event| event.id == chat && !event.tokens.is_empty()));
    assert!(e.cancel(chat));
    e.run().unwrap();
    assert_eq!(e.take_scores()[0].1.logits_row, mock_logits(&p, 256));
    e.score_submit(ScoreRequest { prompt: p.clone(), cache: false, all_rows: false }).unwrap();
    e.step().unwrap();
    assert_eq!(e.take_scores().len(), 1, "the temporary row limit must reset for score-only work");
}
