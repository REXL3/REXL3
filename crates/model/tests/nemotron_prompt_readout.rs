//! Compact Nemotron prompt-log-probability readout: CPU backend contracts and the opt-in real-pack qualification.
use gb_engine::score::RowRecords;
use gb_engine::{Backend, RowRequest};
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::{ModelOptions, NemotronBackend, NemotronRuntime};
use gb_model::families::nemotron_h_omni::model::{readout_tile_plan, NemotronModel};
use gb_model::forward::Candidates;
use gb_model::rows::RowBatch;
use std::collections::BTreeSet;

const VOCAB: usize = 131_072;

#[derive(Default)]
struct FakeRuntime {
    live: BTreeSet<usize>,
    calls: Vec<(usize, usize, usize, bool)>,
}

impl NemotronRuntime for FakeRuntime {
    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn has_mtp(&self) -> bool {
        true
    }
    fn row_records_supported(&self) -> bool {
        true
    }
    fn prefill_readout(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        chosen: &[i32],
        need_last_logits: bool,
        out: &mut RowRecords,
    ) -> Result<Vec<f32>, ModelError> {
        assert!(self.live.contains(&slot));
        assert_eq!(tokens.len(), chosen.len());
        self.calls.push((slot, start, tokens.len(), need_last_logits));
        for (i, &id) in chosen.iter().enumerate() {
            out.lse.push(10.0 + slot as f32);
            out.chosen.push(if id < 0 { f32::NEG_INFINITY } else { id as f32 });
            out.top_ids.extend((0..out.k).map(|j| (slot * 100 + start + i + j) as u32));
            out.top_logits.extend((0..out.k).map(|j| (slot * 100 + start + i + j) as f32));
        }
        Ok(if need_last_logits {
            let mut row = vec![0.0; VOCAB];
            row[0] = slot as f32 + 1.0;
            row
        } else {
            Vec::new()
        })
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        assert!(self.live.insert(slot));
        Ok(())
    }
    fn end_slot(&mut self, slot: usize) {
        self.live.remove(&slot);
    }
    fn prefill_chunk(&mut self, _b: &RowBatch, _want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        unreachable!()
    }
    fn window_candidates(&mut self, _b: &RowBatch, _count: usize) -> Result<Candidates, ModelError> {
        unreachable!()
    }
    fn commit(&mut self, _slot: usize, _keep: usize) {}
    fn flush(&mut self) -> Result<(), ModelError> {
        Ok(())
    }
}

fn options() -> ModelOptions {
    ModelOptions { slots: 2, kv_pages: 64, max_context: 4096, prefill_rows: 256, layers: None, mtp: true }
}

#[test]
fn tiles_are_64_rows_and_only_the_last_can_download_a_full_row() {
    assert!(readout_tile_plan(0, true).is_empty());
    assert_eq!(readout_tile_plan(1, true), vec![(0, 1, true)]);
    assert_eq!(readout_tile_plan(129, true), vec![(0, 64, false), (64, 64, false), (128, 1, true)]);
    assert_eq!(readout_tile_plan(129, false), vec![(0, 64, false), (64, 64, false), (128, 1, false)]);
    let cfg: serde_json::Value = serde_json::from_str(include_str!("../../../testdata/nemotron_omni/config.json")).unwrap();
    assert_eq!(cfg["llm_config"]["vocab_size"].as_u64(), Some(VOCAB as u64));
}

#[test]
fn interleaved_slots_keep_distinct_prompt_records_and_full_rows_separate() {
    let mut b = NemotronBackend::with_runtime(FakeRuntime::default(), 64, &options()).unwrap();
    assert!(b.score_rows_supported());
    let prompts = [(1..=70).collect::<Vec<u32>>(), (5001..=5065).collect::<Vec<u32>>()];
    b.score_begin(0, true);
    b.score_begin(1, true);
    for (slot, done) in [(0, 0), (1, 0), (0, 64), (1, 64)] {
        let prompt = &prompts[slot];
        let want_done = (done + 64).min(prompt.len());
        let step = b.score_slice(slot, prompt, done, &[]).unwrap();
        assert_eq!(step.done, want_done);
        assert_eq!(step.row.is_some(), want_done == prompt.len());
        if let Some(row) = step.row {
            assert_eq!(row.len(), VOCAB);
            assert_eq!(row[0], slot as f32 + 1.0);
        }
    }
    for (slot, prompt) in prompts.iter().enumerate() {
        let rec = b.take_score_records(slot).unwrap();
        let n = prompt.len();
        assert_eq!((rec.lse.len(), rec.chosen.len(), rec.top_ids.len(), rec.top_logits.len()), (n, n, n * rec.k, n * rec.k));
        assert!(rec.lse.iter().all(|&x| x == 10.0 + slot as f32));
        assert_eq!(rec.top_ids[0], (slot * 100) as u32);
        assert_eq!(rec.chosen[n - 1], f32::NEG_INFINITY);
        b.release(slot);
    }
    assert_eq!(b.m.calls, vec![(0, 0, 64, false), (1, 0, 64, false), (0, 64, 6, true), (1, 64, 1, true)]);
    assert!(b.score_slice(0, &prompts[0], 0, &[]).is_err(), "a new slice needs score_begin");
}

#[path = "readout_support/mod.rs"]
mod readout_support;
use readout_support::{bits, check_against_row, check_rows, compare};

fn readout(b: &mut NemotronBackend<NemotronModel>, slot: usize, prompt: &[u32], grid: usize) -> (RowRecords, Vec<f32>) {
    assert!(b.set_prefill_row_limit(Some(grid)));
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

fn generate(m: &mut NemotronModel, slot: usize, prompt: &[u32], new: usize) -> Vec<u32> {
    m.begin_slot(slot).unwrap();
    let b = RowBatch::from_requests(&[RowRequest { slot, first_position: 0, tokens: prompt.to_vec() }]);
    let c = m.prefill_chunk(&b, Some(prompt.len() - 1)).unwrap().unwrap();
    let mut out = vec![c.ids[0] as u32];
    for _ in 1..new {
        let pos = prompt.len() + out.len() - 1;
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: pos as u64, tokens: vec![*out.last().unwrap()] }]);
        let c = m.window_candidates(&b, 1).unwrap();
        m.commit(slot, 1);
        m.flush().unwrap();
        out.push(c.ids[0] as u32);
    }
    m.end_slot(slot);
    out
}

/// Whole, chunked (including one-row slices) and repeated readouts are bit-identical; full f32 logits are checked by a
/// host-f64 log-softmax oracle; the last full row matches generation-head candidates bitwise; and distinct greedy generations
/// are unchanged by sequential readouts and by a readout interleaved with a live generation in another slot.
#[test]
#[ignore = "needs REXL3_TEST_TARGET_PACK and a CUDA device"]
fn real_pack_readout_oracle_and_repeatable_generation() {
    let Some(dir) = std::env::var_os("REXL3_TEST_TARGET_PACK") else {
        eprintln!("not run: REXL3_TEST_TARGET_PACK is not set");
        return;
    };
    assert!(gb_cuda::device_available(), "a stub run is not a qualification");
    let dir = std::path::PathBuf::from(dir);
    let root: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let cfg = gb_model::families::nemotron_h_omni::config::OmniConfig::from_json(&root).unwrap();
    let inv = gb_model::families::nemotron_h_omni::inventory::Inventory::read(&dir).unwrap();
    let plan = gb_model::families::nemotron_h_omni::weights::LayerPlan::build(&cfg, &inv).unwrap();
    let opts = options();
    let m = NemotronModel::load(&dir, &cfg, &plan, opts.clone()).unwrap();
    let v = m.vocab_size();
    assert_eq!(v, VOCAB, "tokenizer vocabulary and f32 LM-head stride are both 131072; there is no padded tail");
    let mut b = NemotronBackend::new(m, 256).unwrap();
    assert!(b.score_rows_supported());
    let prompt: Vec<u32> = (0..130).map(|i| 1000 + ((i * 31 + i / 7) % 99000) as u32).collect();
    let generation_prompts: Vec<Vec<u32>> =
        (0..6).map(|case| (0..32 + case).map(|i| 2000 + ((i * 977 + case * 7919 + i / 5) % 97000) as u32).collect()).collect();
    let baseline: Vec<Vec<u32>> = generation_prompts.iter().map(|p| generate(&mut b.m, 0, p, 8)).collect();

    let (whole, last) = readout(&mut b, 0, &prompt, 256);
    check_rows(&whole, &prompt, v);
    check_against_row(&whole, prompt.len() - 1, &last, None);
    assert_eq!(bits(&readout(&mut b, 0, &prompt, 256).0), bits(&whole));
    for grid in [64, 37, 1] {
        let (cut, _) = readout(&mut b, 0, &prompt, grid);
        let (max, first) = compare(&whole, &cut);
        eprintln!("NEMOTRON_READOUT chunk={grid}: max_abs_logprob_diff={max:e} first_differing_row={first:?}");
        assert_eq!(bits(&cut), bits(&whole));
    }

    b.m.begin_slot(0).unwrap();
    let logits = b.m.prefill_logits_rows(0, 0, &prompt).unwrap();
    b.m.end_slot(0);
    assert_eq!(logits.len(), prompt.len() * v);
    let (mut worst, mut worst_lse) = (0f64, 0f64);
    for i in [0, 1, 31, 63, 64, 65, 127, 128, 129] {
        let (d, dl) = check_against_row(&whole, i, &logits[i * v..(i + 1) * v], prompt.get(i + 1).copied());
        worst = worst.max(d);
        worst_lse = worst_lse.max(dl);
    }
    eprintln!("NEMOTRON_READOUT oracle: max_abs(logprob-host_f64)={worst:e} max_abs(lse-host_f64)={worst_lse:e}");
    assert!(worst <= 5e-6 && worst_lse <= 5e-6);

    b.m.begin_slot(0).unwrap();
    let generation_batch = RowBatch::from_requests(&[RowRequest { slot: 0, first_position: 0, tokens: prompt.clone() }]);
    let generation = b.m.prefill_chunk(&generation_batch, Some(prompt.len() - 1)).unwrap().unwrap();
    b.m.end_slot(0);
    assert_eq!(generation.rows, 1);
    for (&id, &value) in generation.ids.iter().zip(&generation.vals) {
        assert_eq!(last[id as usize].to_bits(), value.to_bits(), "generation/readout logit mismatch at token {id}");
    }

    let live_prompt = &generation_prompts[0];
    b.m.begin_slot(1).unwrap();
    let generation_batch = RowBatch::from_requests(&[RowRequest { slot: 1, first_position: 0, tokens: live_prompt.clone() }]);
    let first = b.m.prefill_chunk(&generation_batch, Some(live_prompt.len() - 1)).unwrap().unwrap();
    let mut interleaved_generation = vec![first.ids[0] as u32];
    b.score_begin(0, true);
    assert!(b.set_prefill_row_limit(Some(37)));
    let (mut readout_done, mut interleaved_steps, mut interleaved_last) = (0, 0, Vec::new());
    for _ in 1..8 {
        if readout_done < prompt.len() {
            let step = b.score_slice(0, &prompt, readout_done, &[]).unwrap();
            assert!(step.captured.is_empty() && step.done > readout_done);
            readout_done = step.done;
            interleaved_steps += 1;
            if let Some(row) = step.row {
                interleaved_last = row;
            }
        }
        let pos = live_prompt.len() + interleaved_generation.len() - 1;
        let decode = RowBatch::from_requests(&[RowRequest {
            slot: 1,
            first_position: pos as u64,
            tokens: vec![*interleaved_generation.last().unwrap()],
        }]);
        let next = b.m.window_candidates(&decode, 1).unwrap();
        b.m.commit(1, 1);
        b.m.flush().unwrap();
        interleaved_generation.push(next.ids[0] as u32);
    }
    b.m.end_slot(1);
    assert_eq!(interleaved_generation, baseline[0], "readout changed a live generation in another slot");
    while readout_done < prompt.len() {
        let step = b.score_slice(0, &prompt, readout_done, &[]).unwrap();
        assert!(step.captured.is_empty() && step.done > readout_done);
        readout_done = step.done;
        if let Some(row) = step.row {
            interleaved_last = row;
        }
    }
    assert_eq!(interleaved_steps, 4);
    assert_eq!(
        interleaved_last.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        last.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    let interleaved = b.take_score_records(0).unwrap();
    b.release(0);
    assert_eq!(bits(&interleaved), bits(&whole));

    let after: Vec<Vec<u32>> = generation_prompts.iter().map(|p| generate(&mut b.m, 0, p, 8)).collect();
    assert_eq!(after, baseline, "six distinct post-readout generations differ from their pre-readout baselines");
    assert!(b.take_error().is_none());
}
