#![allow(clippy::needless_range_loop)]
//! The engine's core contract on the REAL model: a verify window must give the same bits as serial decoding, whatever
//! window, offset, partial acceptance, or neighbours in the batch.
//!
//!     REXL3_QWEN_DIR=... cargo test --release -p gb-model --test gate_exact -- --ignored --nocapture

mod refs;
use gb_engine::RowRequest;
use gb_model::config::QwenConfig;
use gb_model::forward::{Candidates, Model};
use gb_model::pack::Pack;
use gb_model::rows::RowBatch;
use gb_model::weights::Weights;
use refs::{have_refs, ref_npy};

const COUNT: usize = 28;

fn load_model(slots: usize) -> Option<Model> {
    let d = std::path::PathBuf::from(std::env::var("REXL3_QWEN_DIR").ok()?);
    if !gb_cuda::device_available() || !have_refs() {
        return None;
    }
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let pack = Pack::open(&d).unwrap();
    let w = Weights::load(&pack, &cfg, |_, _| {}).unwrap();
    Some(Model::new(cfg, w, slots, 64, 4096, 256).unwrap())
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

#[test]
#[ignore]
fn windows_equal_serial_decoding_bit_for_bit() {
    let Some(mut m) = load_model(4) else { return eprintln!("needs REXL3_QWEN_DIR, a GPU and testdata/ref: skipping") };
    let prompt: Vec<u32> = ref_npy("long/ids.npy").to_i64().into_iter().map(|x| x as u32).collect();
    let p = prompt.len();

    // A: serial decoding in slot 0: tokens t_0.. each fed as a 1-row window followed by commit(1)
    m.begin_slot(0).unwrap();
    let first = m.prefill_chunk(0, 0, &prompt, p, Some(COUNT), 0).unwrap().unwrap();
    let mut toks = vec![first.ids[0]];
    let mut serial: Vec<Candidates> = Vec::new();
    for k in 0..14 {
        let b = RowBatch::from_requests(&[req(0, &toks[k..k + 1], p + k)]);
        let c = m.window_candidates(&b, COUNT).unwrap();
        m.commit(0, 1);
        toks.push(c.ids[0]); // greedy
        serial.push(c);
    }
    eprintln!("serial greedy tokens: {:?}", &toks[..10]);

    // B: slot 1 gets the same prompt; a verify window of 8 rows must reproduce serial rows 0..8 exactly
    m.begin_slot(1).unwrap();
    let again = m.prefill_chunk(1, 0, &prompt, p, Some(COUNT), 0).unwrap().unwrap();
    assert!(same(row(&first, 0), row(&again, 0)), "prefill must be deterministic across slots");
    let w1 = RowBatch::from_requests(&[req(1, &toks[0..8], p)]);
    let c1 = m.window_candidates(&w1, COUNT).unwrap();
    for r in 0..8 {
        assert!(same(row(&c1, r), row(&serial[r], 0)), "window row {r} differs from serial decoding");
    }
    // accept 5 rows (t_0..t_4 consumed), then the next window starts at t_5 and must again match serial rows 5..
    m.commit(1, 5);
    let w2 = RowBatch::from_requests(&[req(1, &toks[5..12], p + 5)]);
    let c2 = m.window_candidates(&w2, COUNT).unwrap();
    for r in 0..7 {
        assert!(
            same(row(&c2, r), row(&serial[5 + r], 0)),
            "after a partial commit, window row {r} differs from serial row {}",
            5 + r
        );
    }
    eprintln!("windows == serial: 8 rows, partial commit of 5, then 7 more rows: all candidate lists bit-identical");

    // C: the same, with the windows of slots 1, 2 and 3 batched together in different shapes and order
    m.begin_slot(2).unwrap();
    m.prefill_chunk(2, 0, &prompt, p, None, 0).unwrap();
    m.begin_slot(3).unwrap();
    m.prefill_chunk(3, 0, &prompt, p, None, 0).unwrap();
    m.begin_slot(1).unwrap();
    m.prefill_chunk(1, 0, &prompt, p, None, 0).unwrap();
    let batch = RowBatch::from_requests(&[req(3, &toks[0..2], p), req(1, &toks[0..8], p), req(2, &toks[0..1], p)]);
    let c = m.window_candidates(&batch, COUNT).unwrap();
    for (seq, n) in [(0usize, 2usize), (1, 8), (2, 1)] {
        let base = batch.seq_start[seq] as usize;
        for r in 0..n {
            assert!(same(row(&c, base + r), row(&serial[r], 0)), "batched sequence {seq} row {r} differs from serial");
        }
    }
    // commit different amounts for the three sequences, then batch them again at different offsets
    m.commit(3, 2);
    m.commit(1, 3);
    m.commit(2, 1);
    let batch2 = RowBatch::from_requests(&[req(2, &toks[1..6], p + 1), req(3, &toks[2..4], p + 2), req(1, &toks[3..11], p + 3)]);
    let c = m.window_candidates(&batch2, COUNT).unwrap();
    for (seq, (n, off)) in [(5usize, 1usize), (2, 2), (8, 3)].into_iter().enumerate() {
        let base = batch2.seq_start[seq] as usize;
        for r in 0..n {
            assert!(
                same(row(&c, base + r), row(&serial[off + r], 0)),
                "second batch: sequence {seq} row {r} differs from serial row {}",
                off + r
            );
        }
    }
    eprintln!("batched windows of 3 sequences with different commits: all bit-identical to serial");
}
