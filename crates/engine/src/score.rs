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
    pub cached_tokens: usize,
}
pub struct ScoreStep {
    pub done: usize,
    pub row: Option<Vec<f32>>,
    pub captured: Vec<Captured>,
}
pub fn unsupported() -> SubmitError {
    SubmitError::Invalid("this model family does not support decisions scoring".into())
}
pub fn unsupported_backend() -> BackendError {
    BackendError::retryable("this backend does not support full-row scoring")
}
impl<B: Backend> Engine<B> {
    pub fn score_supported(&self) -> bool {
        self.backend.score_supported()
    }
    pub fn score_validate(&self, req: &ScoreRequest) -> Result<(), SubmitError> {
        // Capability comes first: unsupported families must never reserve or touch backend state.
        if !self.score_supported() {
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
