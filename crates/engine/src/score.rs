//! Native prompt-only scoring. No sampled token, decode round, or drafter is involved.
use crate::{Backend, BackendError, Captured, Engine, Prefilling, Request, SubmitError};
use gb_sched::AdmitError;
#[derive(Clone, Debug)]
pub struct ScoreRequest {
    pub prompt: Vec<u32>,
    pub cache: bool,
    /// Teacher-forced readout: the backend keeps the full-vocabulary logits of EVERY prompt row, not just the last one
    /// (`Backend::take_score_rows`). Row `i` is the logits after prompt tokens `0..=i`, so it predicts token `i + 1`; the
    /// server's `prompt_logprobs` scores token `i` from row `i - 1` (vLLM's convention). Used by the DSV4.1 correctness oracle.
    pub all_rows: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Scored {
    pub logits_row: Vec<f32>,
    /// `[prompt_len][vocab]` when the request asked for `all_rows` (row `i` after tokens `0..=i`; the last is `logits_row`);
    /// empty otherwise.
    pub logits_rows: Vec<Vec<f32>>,
    /// The compact form of the same readout, from a backend that reduces each row on the device
    /// (`Backend::take_score_records`); `logits_rows` is then empty.
    pub records: Option<RowRecords>,
    pub cached_tokens: usize,
}
/// Most top entries a [`RowRecords`] row carries: the ceiling of the server's `prompt_logprobs`.
pub const RECORD_TOP: usize = 20;
/// The compact form of an `all_rows` readout: per prompt row, what a log-probability of the raw distribution needs instead
/// of the row itself. Row `i` is the full-vocabulary logits after prompt tokens `0..=i`, which predict token `i + 1`.
/// A log-probability is `logit - lse[i]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RowRecords {
    /// Top entries kept per row: `min(RECORD_TOP, vocab)`.
    pub k: usize,
    /// The row's log-sum-exp over the whole vocabulary.
    pub lse: Vec<f32>,
    /// The logit of the prompt's next token, `prompt[i + 1]`; `-inf` on the last row, which has none.
    pub chosen: Vec<f32>,
    /// `[rows][k]`: the row's largest logits, descending, ties by the lower token id.
    pub top_ids: Vec<u32>,
    pub top_logits: Vec<f32>,
}
impl RowRecords {
    pub fn new(k: usize) -> Self {
        Self { k, ..Self::default() }
    }
    pub fn rows(&self) -> usize {
        self.lse.len()
    }
    /// The records of full rows (row `i` after `prompt[..=i]`), reduced on the host in f64: what a device readout is
    /// compared with, and the mock backend's compact form.
    pub fn from_rows(rows: &[Vec<f32>], prompt: &[u32]) -> Self {
        let mut out = Self::new(rows.first().map_or(0, |r| r.len().min(RECORD_TOP)));
        for (i, row) in rows.iter().enumerate() {
            let peak = row.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v as f64));
            let lse = if peak.is_finite() { peak + row.iter().map(|&v| (v as f64 - peak).exp()).sum::<f64>().ln() } else { peak };
            out.lse.push(lse as f32);
            out.chosen.push(prompt.get(i + 1).and_then(|&t| row.get(t as usize)).copied().unwrap_or(f32::NEG_INFINITY));
            let mut order: Vec<u32> = (0..row.len() as u32).collect();
            order.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]).then(a.cmp(&b)));
            order.truncate(out.k);
            out.top_logits.extend(order.iter().map(|&t| row[t as usize]));
            out.top_ids.extend(order);
        }
        out
    }
}
pub struct ScoreStep {
    pub done: usize,
    pub row: Option<Vec<f32>>,
    pub captured: Vec<Captured>,
}
pub fn unsupported() -> SubmitError {
    SubmitError::Invalid("this model family does not support decisions scoring".into())
}
/// The refusal of a `prompt_logprobs` readout (`ScoreRequest { all_rows: true }`) by a backend that reads out only the
/// prompt's last row, or that does not score at all.
pub fn rows_unsupported() -> SubmitError {
    SubmitError::Invalid(
        "prompt_logprobs (and logprobs/top_logprobs, which request it) is not supported by this model family".into(),
    )
}
pub fn unsupported_backend() -> BackendError {
    BackendError::retryable("this backend does not support full-row scoring")
}
impl<B: Backend> Engine<B> {
    pub fn score_supported(&self) -> bool {
        self.backend.score_supported()
    }
    pub fn score_rows_supported(&self) -> bool {
        self.backend.score_rows_supported()
    }
    pub fn score_validate(&self, req: &ScoreRequest) -> Result<(), SubmitError> {
        // Capability comes first: unsupported families must never reserve or touch backend state. The two readouts are
        // separate capabilities: a backend may serve the teacher-forced readout without serving last-row scoring.
        if req.all_rows {
            if !self.backend.score_rows_supported() {
                return Err(rows_unsupported());
            }
        } else if !self.score_supported() {
            return Err(unsupported());
        }
        if req.prompt.is_empty() {
            return Err(SubmitError::Invalid("empty score prompt".into()));
        }
        let v = self.backend.vocab_size();
        if let Some(t) = req.prompt.iter().find(|&&t| t as usize >= v) {
            return Err(SubmitError::Invalid(format!("token id {t} is outside the vocabulary of {v}")));
        }
        self.admission.check(req.prompt.len() as u64)?;
        Ok(())
    }
    pub fn score_would_admit(&self, req: &ScoreRequest) -> Result<(), AdmitError> {
        if self.active.len() + self.prefilling.len() >= self.cfg.max_rows {
            return Err(AdmitError::NoSlot);
        }
        self.admission.can_admit(req.prompt.len() as u64).map(|_| ())
    }
    pub fn score_submit(&mut self, req: ScoreRequest) -> Result<u64, SubmitError> {
        self.score_validate(&req)?;
        self.score_would_admit(&req)?;
        let slot = self.admission.admit(req.prompt.len() as u64)?;
        let id = self.next_id;
        self.next_id += 1;
        self.backend.score_all_rows(req.all_rows);
        self.backend.score_begin(slot, req.all_rows);
        // Request is only the existing prefix-cache bookkeeping container here. max_new=0 is
        // internal, never submitted to or sampled by the generate path.
        let req = Request {
            prompt: req.prompt,
            cache: req.cache,
            max_new: 0,
            sampling: Default::default(),
            stop: vec![],
            draft: false,
            images: vec![],
        };
        self.prefilling.push_back(Prefilling {
            id,
            slot,
            req,
            done: 0,
            cache: None,
            planned: false,
            hit: 0,
            passed: 0,
            full_row: true,
            budget: None,
        });
        Ok(id)
    }
    pub fn take_scores(&mut self) -> Vec<(u64, Scored)> {
        std::mem::take(&mut self.scores)
    }
}
