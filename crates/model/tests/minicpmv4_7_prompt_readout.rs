//! The teacher-forced readout (`prompt_logprobs`) of MiniCPM-V-4.7 on a real pack and a CUDA device, through the engine booted
//! the way `serve` boots it (prefix cache on, the readout submitted without it). Run explicitly on an idle GPU, once with the
//! default fp16 head and once with `REXL3_MINICPM_LOGITS=bf16`:
//!
//! ```text
//! flock -w 1800 /tmp/rexl3-gpu.lock env REXL3_MINICPM_DIR=<the 6.08 bpw pack> \
//!     cargo test --release -p gb-model --test minicpmv4_7_prompt_readout -- --ignored --test-threads=1 --nocapture
//! ```
mod minicpmv4_7_support;
mod readout_support;
use gb_engine::score::{RowRecords, ScoreRequest, Scored};
use gb_engine::{Backend, RowRequest};
use gb_model::forward::PrefillPath;
use gb_model::rows::RowBatch;
use minicpmv4_7_support::{toks, Eng, Rig};
use readout_support::{bits, check_against_row, check_rows, compare};

fn readout(e: &mut Eng, prompt: &[u32]) -> Scored {
    let id = e.score_submit(ScoreRequest { prompt: prompt.to_vec(), cache: false, all_rows: true }).unwrap();
    e.run().unwrap();
    let mut rows = e.take_scores();
    assert_eq!((rows.len(), rows[0].0), (1, id));
    assert!(e.take_events().is_empty() && e.take_outputs().is_empty(), "a readout generates nothing");
    rows.remove(0).1
}

/// (a) the readout of a prompt is the same bits prefilled whole and in two other chunkings, (b) and on a repeat, with the pad
/// columns poisoned, and after other work; (c) each checked row's values agree with the host f64 log-softmax of that row's full
/// logits, read through the one-row head after a prefill of the prompt's first `i + 1` tokens on the prompt's own path, with the
/// largest difference printed; (d) the top entries are sorted and the next token is among them exactly when its rank is below
/// k. One prompt on each prefill path (128 tokens or fewer: the decode GEMV; longer: the tiled GEMM).
#[test]
#[ignore = "needs REXL3_MINICPM_DIR and a CUDA device"]
fn the_readout_is_chunk_invariant_repeatable_and_the_log_softmax_of_the_full_row() {
    let Some(mut rig) = Rig::load_shipped(2, 8192, 2048, 4) else { return };
    let e = &mut rig.eng;
    assert!(e.score_rows_supported() && !e.score_supported(), "the readout only: last-row scoring is not served");
    let (v, f16) = (e.backend.vocab_size(), e.backend.m.logits_are_f16());
    eprintln!("MINICPM_READOUT head={} vocab={v} stride={}", if f16 { "fp16" } else { "bf16" }, e.backend.m.logits_stride());
    let widen = |x: u16| if f16 { gb_exl3::f16::f16_to_f64(x) as f32 } else { gb_cuda::num::bf16_to_f32(x) };
    for (n, grids) in [(100usize, [1usize, 37]), (1500, [512, 173])] {
        let p = toks(n as u64, n);
        e.backend.chunk = 2048;
        let whole = readout(e, &p);
        assert!(whole.logits_rows.is_empty(), "the compact form replaces the full rows");
        let rec: RowRecords = whole.records.clone().expect("records");
        check_rows(&rec, &p, v);
        // the engine's own last row is the last record's row
        check_against_row(&rec, n - 1, &whole.logits_row, None);
        assert_eq!(bits(&readout(e, &p).records.unwrap()), bits(&rec), "n={n}: a repeat differs");
        // the pad columns are never read: overwritten with the largest finite value, nothing moves
        e.backend.m.set_pad_poison(true);
        assert_eq!(bits(&readout(e, &p).records.unwrap()), bits(&rec), "n={n}: the readout read the pad");
        e.backend.m.set_pad_poison(false);
        for grid in grids {
            e.backend.chunk = grid;
            let cut = readout(e, &p).records.unwrap();
            let (max, first) = compare(&rec, &cut);
            eprintln!("MINICPM_READOUT n={n} chunk={grid} vs whole: max_abs_logprob_diff={max:e} first_differing_row={first:?}");
            assert_eq!(
                bits(&cut),
                bits(&rec),
                "n={n} chunk={grid}: differs from the whole prefill by up to {max:e} from row {first:?}"
            );
        }
        e.backend.chunk = 2048;
        // full rows through the one-row head: a prefill of the prompt's first i + 1 tokens on the prompt's own path
        let (mut worst, mut worst_lse) = (0f64, 0f64);
        for i in [0usize, 1, n / 3, n / 2, n - 2, n - 1] {
            let m = &mut e.backend.m;
            m.begin_slot(0).unwrap();
            m.flush().unwrap();
            m.reserve(0, 0, i + 1).unwrap();
            let chunk = RowBatch::from_requests(&[RowRequest { slot: 0, first_position: 0, tokens: p[..=i].to_vec() }]);
            m.prefill_chunk(&chunk, PrefillPath::for_prompt(n), None).unwrap();
            let mut raw = vec![0u16; v];
            m.last_rows_logits(i, 1, &mut raw).unwrap();
            m.end_slot(0);
            let row: Vec<f32> = raw.into_iter().map(widen).collect();
            let (d, dl) = check_against_row(&rec, i, &row, p.get(i + 1).copied());
            (worst, worst_lse) = (worst.max(d), worst_lse.max(dl));
        }
        eprintln!("MINICPM_READOUT n={n} rows=6 max_abs(logprob - host_f64)={worst:e} max_abs(lse - host_f64)={worst_lse:e}");
        assert!(worst <= 4e-6 && worst_lse <= 4e-6, "n={n}: {worst:e} / {worst_lse:e}");
        assert_eq!(bits(&readout(e, &p).records.unwrap()), bits(&rec), "n={n}: the readout after other work differs");
    }
}
