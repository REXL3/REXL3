//! The teacher-forced readout (`prompt_logprobs`) of Qwen3.8-Flash-Next on a real pack and a CUDA device. Run explicitly on an
//! idle GPU:
//! REXL3_FLASH_NEXT_DIR=~/models/flashnext-exl3-3.05bpw \
//! cargo test --release -p gb-model --test flash_next_prompt_readout -- --ignored --nocapture
mod readout_support;
use gb_engine::score::RowRecords;
use gb_engine::{Backend, RowRequest};
use gb_model::families::qwen_flash_next::backend::FlashNextBackend;
use gb_model::families::qwen_flash_next::model::{FlashNextModel, ModelOptions};
use gb_model::forward::PrefillPath;
use gb_model::rows::RowBatch;
use readout_support::{bits, check_against_row, check_rows, compare};

/// One readout of `prompt` in `slot`, driven as the engine drives it; also the full last row the final slice returns.
fn readout(b: &mut FlashNextBackend, slot: usize, prompt: &[u32]) -> (RowRecords, Vec<f32>) {
    b.score_begin(slot, true);
    let mut done = 0;
    let last = loop {
        let step = b.score_slice(slot, prompt, done, &[]).unwrap();
        assert!(step.captured.is_empty() && step.done > done);
        done = step.done;
        if let Some(row) = step.row {
            assert_eq!(done, prompt.len());
            break row;
        }
    };
    let rec = b.take_score_records(slot).expect("records");
    b.release(slot);
    assert!(b.take_error().is_none());
    (rec, last)
}

/// (a) the readout of a prompt is the same bits prefilled whole and in two other chunkings, (b) and on a repeat and in another
/// slot; (c) each checked row's values agree with the host f64 log-softmax of that row's full logits, read through the one-row
/// head, with the largest difference printed; (d) the top entries are sorted and the next token is among them exactly when its
/// rank is below k. One prompt on each prefill path (128 tokens or fewer: the decode GEMV; longer: the tiled GEMM).
#[test]
#[ignore = "needs REXL3_FLASH_NEXT_DIR and a CUDA device"]
fn the_readout_is_chunk_invariant_repeatable_and_the_log_softmax_of_the_full_row() {
    let Ok(dir) = std::env::var("REXL3_FLASH_NEXT_DIR") else {
        eprintln!("not run: REXL3_FLASH_NEXT_DIR is not set");
        return;
    };
    assert!(gb_cuda::device_available(), "a stub run is not a qualification");
    let opts = ModelOptions { slots: 2, kv_pages: 64, max_context: 4096, prefill_rows: 1024, warm: false };
    let m = FlashNextModel::load(std::path::Path::new(&dir), opts).unwrap();
    let v = m.vocab_size();
    let mut b = FlashNextBackend::new(m, 1024, 0);
    assert!(b.score_rows_supported() && !b.score_supported(), "the readout only: last-row scoring is not served");
    // without a readout begun in the slot a score slice is refused before it touches the device
    assert!(b.score_slice(0, &[1, 2, 3], 0, &[]).is_err());
    for (n, grids) in [(100usize, [1usize, 37]), (1000, [256, 173])] {
        let p: Vec<u32> = (0..n).map(|i| ((i as u64 * 2_654_435_761) % (v as u64 - 16) + 8) as u32).collect();
        b.chunk = 1024;
        let (rec, last) = readout(&mut b, 0, &p);
        check_rows(&rec, &p, v);
        check_against_row(&rec, n - 1, &last, None);
        assert_eq!(bits(&readout(&mut b, 0, &p).0), bits(&rec), "n={n}: a repeat differs");
        assert_eq!(bits(&readout(&mut b, 1, &p).0), bits(&rec), "n={n}: another slot differs");
        for grid in grids {
            b.chunk = grid;
            let (cut, _) = readout(&mut b, 0, &p);
            let (max, first) = compare(&rec, &cut);
            eprintln!(
                "FLASH_NEXT_READOUT n={n} chunk={grid} vs whole: max_abs_logprob_diff={max:e} first_differing_row={first:?}"
            );
            assert_eq!(
                bits(&cut),
                bits(&rec),
                "n={n} chunk={grid}: differs from the whole prefill by up to {max:e} from row {first:?}"
            );
        }
        b.chunk = 1024;
        // full rows through the one-row head: a prefill of the prompt's first i + 1 tokens on the prompt's own path
        let (mut worst, mut worst_lse) = (0f64, 0f64);
        for i in [0usize, 1, n / 3, n / 2, n - 2, n - 1] {
            b.m.begin_slot(0).unwrap();
            b.m.flush().unwrap();
            let chunk = RowBatch::from_requests(&[RowRequest { slot: 0, first_position: 0, tokens: p[..=i].to_vec() }]);
            b.m.prefill_chunk(&chunk, PrefillPath::for_prompt(n), None).unwrap();
            let mut raw = vec![0u16; v];
            b.m.last_rows_logits(i, 1, &mut raw).unwrap();
            b.m.end_slot(0);
            let row: Vec<f32> = raw.into_iter().map(|x| f32::from_bits((x as u32) << 16)).collect();
            let (d, dl) = check_against_row(&rec, i, &row, p.get(i + 1).copied());
            (worst, worst_lse) = (worst.max(d), worst_lse.max(dl));
        }
        eprintln!("FLASH_NEXT_READOUT n={n} rows=6 max_abs(logprob - host_f64)={worst:e} max_abs(lse - host_f64)={worst_lse:e}");
        assert!(worst <= 4e-6 && worst_lse <= 4e-6, "n={n}: {worst:e} / {worst_lse:e}");
        assert_eq!(bits(&readout(&mut b, 0, &p).0), bits(&rec), "n={n}: the readout after other work differs");
    }
}
