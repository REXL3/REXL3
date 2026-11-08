#![allow(clippy::needless_range_loop)]
//! A verify window wider than the served 64 rows, on the real model: 32 streams of 4 rows (128 rows) and 16 streams of 8 must give
//! every row the bits that stream decoding alone, one row at a time, gives it; a window past the cap is refused and leaves the model
//! as it was. The proof debt of a served cap above 64 (`Model::with_window_rows`).
//!
//!     REXL3_QWEN_DIR=... cargo test --release -p gb-model --test wide_window -- --ignored --nocapture

mod refs;
use gb_engine::RowRequest;
use gb_model::config::QwenConfig;
use gb_model::error::ModelError;
use gb_model::forward::{Candidates, Model, WINDOW_ROWS};
use gb_model::pack::Pack;
use gb_model::rows::RowBatch;
use gb_model::weights::Weights;
use refs::{have_refs, ref_npy};

const COUNT: usize = 28;
const STREAMS: usize = 32;
/// Tokens decoded serially per stream: two windows of 4 rows after a partial commit of up to 4.
const SERIAL: usize = 8;

fn load(slots: usize, window: usize) -> Option<Model> {
    let d = std::path::PathBuf::from(std::env::var("REXL3_QWEN_DIR").ok()?);
    if !gb_cuda::device_available() || !have_refs() {
        return None;
    }
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
    Some(Model::with_window_rows(cfg, w, slots, slots * 4 + 64, 4096, 256, window).unwrap())
}

fn row(c: &Candidates, r: usize) -> (&[i32], &[f32]) {
    (&c.ids[r * c.count..(r + 1) * c.count], &c.vals[r * c.count..(r + 1) * c.count])
}

fn same(a: (&[i32], &[f32]), b: (&[i32], &[f32])) -> bool {
    a.0 == b.0 && a.1.iter().map(|v| v.to_bits()).eq(b.1.iter().map(|v| v.to_bits()))
}

fn req(slot: usize, tokens: &[i32], first: usize) -> RowRequest {
    RowRequest { slot, tokens: tokens.iter().map(|&t| t as u32).collect(), first_position: first as u64 }
}

fn prompts() -> Vec<Vec<u32>> {
    let long: Vec<u32> = ref_npy("long/ids.npy").to_i64().into_iter().map(|x| x as u32).collect();
    // every stream its own prompt (a different start, so a different length too): a mix-up between streams cannot hide
    (0..STREAMS).map(|s| long[s..].to_vec()).collect()
}

#[test]
#[ignore]
fn a_128_row_window_of_32_streams_equals_each_stream_decoded_alone() {
    let Some(mut m) = load(STREAMS, 128) else { return eprintln!("needs REXL3_QWEN_DIR, a GPU and testdata/ref: skipping") };
    assert_eq!(m.window_rows(), 128);
    let ps = prompts();
    // serial reference: each stream alone, one row a step, greedy, SERIAL tokens
    let mut toks: Vec<Vec<i32>> = Vec::new();
    let mut serial: Vec<Vec<Candidates>> = Vec::new();
    let mut first: Vec<Candidates> = Vec::new();
    for s in 0..STREAMS {
        m.begin_slot(s).unwrap();
        let f = m.prefill_chunk(s, 0, &ps[s], ps[s].len(), Some(COUNT), 0).unwrap().unwrap();
        let mut t = vec![f.ids[0]];
        let mut rows = Vec::new();
        for k in 0..SERIAL {
            let b = RowBatch::from_requests(&[req(s, &t[k..k + 1], ps[s].len() + k)]);
            let c = m.window_candidates(&b, COUNT).unwrap();
            m.commit(s, 1);
            t.push(c.ids[0]);
            rows.push(c);
        }
        toks.push(t);
        serial.push(rows);
        first.push(f);
    }
    eprintln!("serial reference: {STREAMS} streams x {SERIAL} tokens");

    // the same streams from fresh slots, then ONE window of 32 x 4 = 128 rows
    for s in 0..STREAMS {
        m.begin_slot(s).unwrap();
        let again = m.prefill_chunk(s, 0, &ps[s], ps[s].len(), Some(COUNT), 0).unwrap().unwrap();
        assert!(same(row(&first[s], 0), row(&again, 0)), "stream {s}: the prompt's last row differs between two prefills");
    }
    let w1 = RowBatch::from_requests(&(0..STREAMS).map(|s| req(s, &toks[s][0..4], ps[s].len())).collect::<Vec<_>>());
    assert_eq!(w1.rows, 128);
    let c1 = m.window_candidates(&w1, COUNT).unwrap();
    for s in 0..STREAMS {
        let base = w1.seq_start[s] as usize;
        for r in 0..4 {
            assert!(
                same(row(&c1, base + r), row(&serial[s][r], 0)),
                "128-row window: stream {s} row {r} differs from serial decoding"
            );
        }
    }
    eprintln!("128-row window (32 streams x 4 rows): every row bit-identical to serial");
    // streams keep 1..=4 of their 4 rows, then a second 128-row window at their new offsets
    let keep: Vec<usize> = (0..STREAMS).map(|s| 1 + s % 4).collect();
    for s in 0..STREAMS {
        m.commit(s, keep[s]);
    }
    let w2 = RowBatch::from_requests(
        &(0..STREAMS).map(|s| req(s, &toks[s][keep[s]..keep[s] + 4], ps[s].len() + keep[s])).collect::<Vec<_>>(),
    );
    let c2 = m.window_candidates(&w2, COUNT).unwrap();
    for s in 0..STREAMS {
        let base = w2.seq_start[s] as usize;
        for r in 0..4 {
            assert!(
                same(row(&c2, base + r), row(&serial[s][keep[s] + r], 0)),
                "second 128-row window: stream {s} (kept {}) row {r} differs from serial row {}",
                keep[s],
                keep[s] + r
            );
        }
    }
    eprintln!("second 128-row window after partial commits: bit-identical to serial");

    // a different shape, 16 streams x 8 rows = 128 rows, from fresh slots
    for s in 0..16 {
        m.begin_slot(s).unwrap();
        m.prefill_chunk(s, 0, &ps[s], ps[s].len(), None, 0).unwrap();
    }
    let w3 = RowBatch::from_requests(&(0..16).map(|s| req(s, &toks[s][0..8], ps[s].len())).collect::<Vec<_>>());
    assert_eq!(w3.rows, 128);
    let c3 = m.window_candidates(&w3, COUNT).unwrap();
    for s in 0..16 {
        let base = w3.seq_start[s] as usize;
        for r in 0..8 {
            assert!(
                same(row(&c3, base + r), row(&serial[s][r], 0)),
                "16 x 8 window: stream {s} row {r} differs from serial decoding"
            );
        }
    }
    eprintln!("16 streams x 8 rows: bit-identical to serial");
}

#[test]
#[ignore]
fn a_window_past_the_cap_is_refused_and_the_model_is_as_it_was() {
    // served cap: 65 rows are refused with a Request error, the model's next window gives the bits it gave before
    let Some(mut m) = load(2, WINDOW_ROWS) else { return eprintln!("needs REXL3_QWEN_DIR, a GPU and testdata/ref: skipping") };
    let ps = prompts();
    let p = ps[0].len();
    m.begin_slot(0).unwrap();
    let f = m.prefill_chunk(0, 0, &ps[0], p, Some(COUNT), 0).unwrap().unwrap();
    let t0 = f.ids[0];
    let one = RowBatch::from_requests(&[req(0, &[t0], p)]);
    let before = m.window_candidates(&one, COUNT).unwrap();
    for rows in [WINDOW_ROWS + 1, 128, 129, 200] {
        let toks: Vec<i32> = (0..rows as i32).map(|i| 1000 + i).collect();
        let b = RowBatch::from_requests(&[req(0, &toks, p)]);
        match m.window_candidates(&b, COUNT) {
            Err(ModelError::Request(msg)) => assert!(msg.contains("1..=64"), "{msg}"),
            other => panic!("{rows} rows on a 64-row window: {:?}", other.map(|c| c.rows)),
        }
    }
    let after = m.window_candidates(&one, COUNT).unwrap();
    assert!(same(row(&before, 0), row(&after, 0)), "a refused window changed what the next window computes");
    // the widest legal window still runs
    let toks: Vec<i32> = (0..WINDOW_ROWS as i32).map(|i| 1000 + i).collect();
    let full = m.window_candidates(&RowBatch::from_requests(&[req(0, &toks, p)]), COUNT).unwrap();
    assert_eq!(full.rows, WINDOW_ROWS);
}

#[test]
#[ignore]
fn a_128_row_model_refuses_row_129_with_the_cap_in_the_message() {
    let Some(mut m) = load(2, 128) else { return eprintln!("needs REXL3_QWEN_DIR, a GPU and testdata/ref: skipping") };
    let ps = prompts();
    let p = ps[0].len();
    m.begin_slot(0).unwrap();
    let f = m.prefill_chunk(0, 0, &ps[0], p, Some(COUNT), 0).unwrap().unwrap();
    let one = RowBatch::from_requests(&[req(0, &[f.ids[0]], p)]);
    let before = m.window_candidates(&one, COUNT).unwrap();
    let toks: Vec<i32> = (0..129i32).map(|i| 1000 + i).collect();
    match m.window_candidates(&RowBatch::from_requests(&[req(0, &toks, p)]), COUNT) {
        Err(ModelError::Request(msg)) => assert!(msg.contains("1..=128") && msg.contains("129"), "{msg}"),
        other => panic!("129 rows: {:?}", other.map(|c| c.rows)),
    }
    let after = m.window_candidates(&one, COUNT).unwrap();
    assert!(same(row(&before, 0), row(&after, 0)));
    let toks: Vec<i32> = (0..128i32).map(|i| 1000 + i).collect();
    assert_eq!(m.window_candidates(&RowBatch::from_requests(&[req(0, &toks, p)]), COUNT).unwrap().rows, 128);
}
