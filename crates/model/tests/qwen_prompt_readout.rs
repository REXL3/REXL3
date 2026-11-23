//! The teacher-forced readout (`prompt_logprobs`) of the generic Qwen family on a real pack and a CUDA device. Run explicitly on
//! an idle GPU:
//! REXL3_QWEN_DIR=~/models/r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw \
//! cargo test --release -p gb-model --test qwen_prompt_readout -- --ignored --nocapture
mod readout_support;
use gb_engine::score::{RowRecords, ScoreRequest, Scored};
use gb_engine::Backend;
use gb_model::forward::HeadMode;
use gb_model::serve::{build_engine, PrefixCacheOption, QwenEngine, ServeOptions};
use readout_support::{bits, check_against_row, check_rows, compare};

fn readout(e: &mut QwenEngine, prompt: &[u32]) -> Scored {
    let id = e.score_submit(ScoreRequest { prompt: prompt.to_vec(), cache: false, all_rows: true }).unwrap();
    e.run().unwrap();
    let mut rows = e.take_scores();
    assert_eq!((rows.len(), rows[0].0), (1, id));
    assert!(e.take_events().is_empty() && e.take_outputs().is_empty(), "a readout generates nothing");
    rows.remove(0).1
}

/// (a) the readout of a prompt is the same bits prefilled whole and in two other chunkings, (b) and on a repeat; (c) each checked
/// row's values agree with the host f64 log-softmax of that row's full logits, read through the existing one-row head, with the
/// largest difference printed; (d) the top entries are sorted and the next token is among them exactly when its rank is below k.
/// One prompt on each prefill path (128 tokens or fewer: the decode GEMV; longer: the tiled GEMM).
#[test]
#[ignore = "needs REXL3_QWEN_DIR and a CUDA device"]
fn the_readout_is_chunk_invariant_repeatable_and_the_log_softmax_of_the_full_row() {
    let Ok(dir) = std::env::var("REXL3_QWEN_DIR") else {
        eprintln!("not run: REXL3_QWEN_DIR is not set");
        return;
    };
    assert!(gb_cuda::device_available(), "a stub run is not a qualification");
    let mut o = ServeOptions::new(dir.into());
    o.slots = 2;
    o.max_model_len = 8192;
    o.kv_tokens = Some(16384);
    o.prefill_chunk = 4096;
    o.vision = None;
    o.prefix_cache = PrefixCacheOption::Mib(1024);
    let mut e = build_engine(&o).unwrap();
    assert!(e.score_rows_supported());
    let v = e.backend.vocab_size();
    for (n, grids) in [(100usize, [1usize, 37]), (1500, [512, 173])] {
        let p: Vec<u32> = (0..n).map(|i| 1000 + ((i * 31 + i / 7) % 99000) as u32).collect();
        e.backend.chunk = 4096;
        let whole = readout(&mut e, &p);
        assert!(whole.logits_rows.is_empty(), "the compact form replaces the full rows");
        let rec: RowRecords = whole.records.clone().expect("records");
        check_rows(&rec, &p, v);
        // the engine's own last row is the last record's row
        check_against_row(&rec, n - 1, &whole.logits_row, None);
        assert_eq!(bits(&readout(&mut e, &p).records.unwrap()), bits(&rec), "n={n}: a repeat differs");
        for grid in grids {
            e.backend.chunk = grid;
            let cut = readout(&mut e, &p).records.unwrap();
            let (max, first) = compare(&rec, &cut);
            eprintln!("QWEN_READOUT n={n} chunk={grid} vs whole: max_abs_logprob_diff={max:e} first_differing_row={first:?}");
            assert_eq!(
                bits(&cut),
                bits(&rec),
                "n={n} chunk={grid}: differs from the whole prefill by up to {max:e} from row {first:?}"
            );
        }
        e.backend.chunk = 4096;
        // full rows through the existing one-row head: a prefill of the prompt's first i + 1 tokens on the prompt's own path
        let (mut worst, mut worst_lse) = (0f64, 0f64);
        for i in [0usize, 1, n / 3, n / 2, n - 2, n - 1] {
            let m = &mut e.backend.m;
            m.begin_slot(0).unwrap();
            m.prefill_chunk_snap_head(0, 0, &p[..=i], n, HeadMode::FullLastRow, 0, None, &[], &[]).unwrap();
            let row = m.score_head_row_zero().unwrap();
            m.end_slot(0);
            let (d, dl) = check_against_row(&rec, i, &row, p.get(i + 1).copied());
            (worst, worst_lse) = (worst.max(d), worst_lse.max(dl));
        }
        eprintln!("QWEN_READOUT n={n} rows=6 max_abs(logprob - host_f64)={worst:e} max_abs(lse - host_f64)={worst_lse:e}");
        assert!(worst <= 4e-6 && worst_lse <= 4e-6, "n={n}: {worst:e} / {worst_lse:e}");
        assert_eq!(bits(&readout(&mut e, &p).records.unwrap()), bits(&rec), "n={n}: the readout after other work differs");
    }
}
