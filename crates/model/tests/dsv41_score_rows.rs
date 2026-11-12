//! The DeepSeek-V4.1 teacher-forced readout (`prompt_logprobs`) through the backend, over a runtime that holds the head's
//! `HEAD_ROWS` rows as the model does: a prompt longer than the head is read out in pieces of it (REXL3/REXL3#91).
use gb_engine::{Backend, RowOutput};
use gb_model::error::ModelError;
use gb_model::families::deepseek_v41::backend::Dsv41Backend;
use gb_model::families::deepseek_v41::model::{DraftRound, HEAD_ROWS};
use gb_model::families::deepseek_v41::tp_runtime::Dsv41Runtime;
use gb_model::forward::Candidates;
use gb_model::rows::RowBatch;

/// A runtime whose all-rows readout holds `HEAD_ROWS` rows, as the model's head does, and answers row `r` of a
/// readout with `[position, token]`, so a stitched readout shows where every row came from.
struct HeadRt {
    chunk: usize,
    reads: Vec<(usize, usize)>,
}

impl gb_model::families::deepseek_v41::tree::TreeRoundHost for HeadRt {
    fn tree_path_window(&mut self, _t: &[u32], _p: u64, _c: usize) -> Result<Vec<RowOutput>, String> {
        Err("no trees".into())
    }
    fn tree_commit_window(&mut self, _i: usize, _r: usize, _w: &[u32]) -> Result<(), String> {
        Err("no trees".into())
    }
    fn tree_drop_round(&mut self) -> Result<(), String> {
        Ok(())
    }
}

impl Dsv41Runtime for HeadRt {
    fn vocab_size(&self) -> usize {
        1 << 20
    }
    fn has_mtp(&self) -> bool {
        false
    }
    fn chunk_rows(&self) -> usize {
        self.chunk
    }
    fn begin_slot(&mut self, _slot: usize) -> Result<(), ModelError> {
        Ok(())
    }
    fn end_slot(&mut self, _slot: usize) {}
    fn prefill_chunk(
        &mut self,
        _slot: usize,
        _start: usize,
        _tokens: &[u32],
        _want: Option<usize>,
    ) -> Result<Option<Candidates>, ModelError> {
        Err(ModelError::Request("no prefill on this mock".into()))
    }
    fn window_candidates(&mut self, _b: &RowBatch, _count: usize) -> Result<Candidates, ModelError> {
        Err(ModelError::Request("no windows on this mock".into()))
    }
    fn commit(&mut self, _slot: usize, _keep: usize) {}
    fn flush(&mut self) -> Result<(), ModelError> {
        Ok(())
    }
    fn prefill_logits_last_row(&mut self, _slot: usize, _start: usize, _tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        Err(ModelError::Request("no last row on this mock".into()))
    }
    fn prefill_logits_rows(&mut self, _slot: usize, start: usize, tokens: &[u32]) -> Result<Vec<Vec<f32>>, ModelError> {
        if tokens.len() > HEAD_ROWS {
            return Err(ModelError::Request(format!("{} rows through a head of {}", tokens.len(), HEAD_ROWS)));
        }
        self.reads.push((start, tokens.len()));
        Ok(tokens.iter().enumerate().map(|(i, &t)| vec![(start + i) as f32, t as f32]).collect())
    }
    fn propose_mtp(&mut self, _slot: usize, _tokens: &[u32], _max: usize) -> Result<Option<DraftRound>, ModelError> {
        Ok(None)
    }
}

/// A `prompt_logprobs` readout of a prompt longer than the head (`HEAD_ROWS`) is read out in pieces of the head: every
/// readout call fits it, the pieces tile the prompt in order, and every prompt row comes back exactly once. Reading a
/// whole slice (512 rows on one node, 2048 on a group) through the 64-row head was an assert that ended the server.
#[test]
fn an_all_rows_score_reads_a_long_prompt_out_in_pieces_of_the_head() {
    for (chunk, n) in [(512usize, 1_100usize), (2048, 2_050), (512, HEAD_ROWS), (512, HEAD_ROWS + 1), (512, 3)] {
        let mut b = Dsv41Backend::new(HeadRt { chunk, reads: Vec::new() });
        let prompt: Vec<u32> = (0..n as u32).map(|i| i * 7 + 1).collect();
        b.score_all_rows(true);
        let mut done = 0;
        while done < n {
            let step = b.score_slice(0, &prompt, done, &[]).unwrap_or_else(|e| panic!("chunk {chunk} n {n}: {e:?}"));
            assert!(step.done > done && step.done <= n, "chunk {chunk} n {n}: a slice moves forward");
            assert_eq!(step.row.is_some(), step.done == n, "chunk {chunk} n {n}: the last row ends the prompt");
            done = step.done;
        }
        let rows = b.take_score_rows();
        let want: Vec<Vec<f32>> = prompt.iter().enumerate().map(|(i, &t)| vec![i as f32, t as f32]).collect();
        assert_eq!(rows, want, "chunk {chunk} n {n}: one row per prompt token, in order");
        let reads = &b.m.reads;
        assert!(reads.iter().all(|&(_, len)| (1..=HEAD_ROWS).contains(&len)), "chunk {chunk} n {n}: {reads:?}");
        let mut at = 0;
        for &(start, len) in reads {
            assert_eq!(start, at, "chunk {chunk} n {n}: the pieces tile the prompt");
            at += len;
        }
        assert_eq!(at, n, "chunk {chunk} n {n}");
        assert_eq!(reads.len(), n.div_ceil(HEAD_ROWS), "chunk {chunk} n {n}: no more pieces than needed");
    }
}
