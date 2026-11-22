//! The teacher-forced readout (`prompt_logprobs`) of Ling-3.1 on the real pack (about 106 GiB resident, MTP on) and a CUDA
//! device, behind its backend. `#[ignore]`, env-gated (`REXL3_LING_DIR`; synthetic token ids), one model per process, with the
//! 1 GiB `MemAvailable` guard of `ling31_support` that aborts the process. Run explicitly on an idle GB10 that holds nothing else:
//!
//! ```text
//! flock -w 1800 /tmp/rexl3-gpu.lock env REXL3_LING_DIR=~/models/vcruz305/Ling-3.1-flash-1019-EXL3-SAGE-1.51bpw \
//!     cargo test --release -p gb-model --test ling31_prompt_readout -- --ignored --test-threads=1 --nocapture
//! ```
mod ling31_support;
mod readout_support;
use gb_engine::score::RowRecords;
use gb_engine::Backend;
use gb_model::families::bailing_moe_v3::backend::BailingBackend;
use gb_model::families::bailing_moe_v3::config::BailingConfig;
use ling31_support::{gib, guard, ids, load, pack_dir};
use readout_support::{bits, check_against_row, check_rows, compare};

/// One readout of `prompt` in `slot`, driven as the engine drives it (slices until the prompt is in); also the full last row
/// the final slice returns.
fn readout(b: &mut BailingBackend, slot: usize, prompt: &[u32]) -> (RowRecords, Vec<f32>) {
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

/// (a) the readout of a prompt is the same bits in one slice and in the other chunkings the server can run (`--prefill-chunk`
/// is taken into 128 to 4096 rows on the 16-row KDA grid: the profile's 512 and the smallest, 128; a 100-token prompt is one
/// chunk at every served size), (b) and on a repeat and after other work;
/// (c) each checked row's values agree with the host f64 log-softmax of that row's full logits, read through the one-row head
/// after a prefill of the prompt's first `i + 1` tokens, with the largest difference printed; (d) the top entries are sorted and
/// the next token is among them exactly when its rank is below k.
#[test]
#[ignore = "needs REXL3_LING_DIR and a CUDA device"]
fn the_readout_is_chunk_invariant_repeatable_and_the_log_softmax_of_the_full_row() {
    let Some(dir) = pack_dir() else { return };
    let mtp = BailingConfig::from_dir(&dir).map(|c| c.layers.len() >= 72 && c.has_mtp()).unwrap_or(false);
    let Some(model) = load(1, 32 * 256, mtp) else { return };
    let v = model.cfg.vocab;
    let mut b = BailingBackend::new(model, mtp);
    assert!(b.score_rows_supported() && !b.score_supported(), "the readout only: last-row scoring is not served");
    // without a readout begun in the slot a score slice is refused before it touches the device
    assert!(b.score_slice(0, &[1, 2, 3], 0, &[]).is_err());
    // Not asked: grids of 16 or 48 rows, which the server never runs (`--prefill-chunk 16` is taken as 128). On them the trunk
    // itself, with no readout code run, leaves another slot state now and then: on N4 on 2026-11-17, 4 of 40 cold prefills of
    // 100 tokens in 16-row chunks, against 0 of 40 whole. A small served tail showed it too (1 of 30 prefills of 1,028 tokens in
    // 512-row chunks, a 4-row tail). That is an engine defect, reported with the diagnosis; this test does not hide it.
    for (n, grids) in [(100usize, &[][..]), (2600, &[512usize, 128][..])] {
        let p = ids(n as u64, n);
        b.readout_chunk = None;
        let (rec, last) = readout(&mut b, 0, &p);
        check_rows(&rec, &p, v);
        check_against_row(&rec, n - 1, &last, None);
        assert_eq!(bits(&readout(&mut b, 0, &p).0), bits(&rec), "n={n}: a repeat differs");
        for &grid in grids {
            b.readout_chunk = Some(grid);
            let (cut, _) = readout(&mut b, 0, &p);
            let (max, first) = compare(&rec, &cut);
            eprintln!("LING_READOUT n={n} chunk={grid} vs whole: max_abs_logprob_diff={max:e} first_differing_row={first:?}");
            assert_eq!(
                bits(&cut),
                bits(&rec),
                "n={n} chunk={grid}: differs from the one-slice readout by up to {max:e} from row {first:?}"
            );
        }
        b.readout_chunk = None;
        // full rows from the existing path: a prefill of the first i + 1 tokens that asks for the last row's candidates runs the
        // one-row head into the logits buffer, read here whole
        let (mut worst, mut worst_lse) = (0f64, 0f64);
        for i in [0usize, 1, n / 3, n / 2, n - 2, n - 1] {
            b.model.begin_slot(0).unwrap();
            b.model.prefill_chunk(0, 0, &p[..=i], Some(8)).unwrap();
            b.model.stream.sync().unwrap();
            let mut raw = vec![0u16; v];
            b.model.s.logits.download_range(0, &mut raw).unwrap();
            let row: Vec<f32> = raw.into_iter().map(|x| gb_exl3::f16::f16_to_f64(x) as f32).collect();
            b.model.end_slot(0);
            let (d, dl) = check_against_row(&rec, i, &row, p.get(i + 1).copied());
            (worst, worst_lse) = (worst.max(d), worst_lse.max(dl));
        }
        eprintln!("LING_READOUT n={n} rows=6 max_abs(logprob - host_f64)={worst:e} max_abs(lse - host_f64)={worst_lse:e}");
        assert!(worst <= 4e-6 && worst_lse <= 4e-6, "n={n}: {worst:e} / {worst_lse:e}");
        assert_eq!(bits(&readout(&mut b, 0, &p).0), bits(&rec), "n={n}: the readout after other work differs");
    }
    eprintln!("MemAvailable low {:.2} GiB", gib(guard().low()));
}
