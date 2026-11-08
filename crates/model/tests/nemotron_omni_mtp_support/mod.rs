//! Shared helpers of the Nemotron MTP head's GPU tests (`nemotron_omni_mtp*.rs`): one real model with the head per process, prompt
//! tokens, serial and drafted greedy runs.
//!
//! Env: `REXL3_NEMO_OMNI_DIR` (the pack), `REXL3_NEMO_ORACLE_DIR` (the fork's dumps: `ids_0.u32` is the oracle's text, used as a real
//! prefix), `REXL3_MTP_PROMPTS_DIR` (token ids of the natural-text prompts, `tools/nemotron_omni/mtp_vocab.py prompts`). A test whose
//! inputs are missing prints "not run" and passes. One model per process: run with `--test-threads=1` under the GPU lock.
#![allow(dead_code)]

pub mod hostref;

use gb_engine::RowRequest;
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::ModelOptions;
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::model::NemotronModel;
use gb_model::families::nemotron_h_omni::mtp::DraftAsk;
use gb_model::families::nemotron_h_omni::weights::LayerPlan;
use gb_model::rows::RowBatch;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

pub const SLOTS: usize = 6;
pub const KV_PAGES: usize = 40;
pub const MAX_CONTEXT: usize = 8192;

pub fn pack_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set (the published pack directory)");
    }
    dir
}

pub fn config(dir: &Path) -> OmniConfig {
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| panic!("config.json: {e}"));
    OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap_or_else(|e| panic!("{e}"))
}

pub fn oracle_dir() -> Option<PathBuf> {
    let d = std::env::var_os("REXL3_NEMO_ORACLE_DIR").map(PathBuf::from);
    if d.is_none() {
        eprintln!("not run: REXL3_NEMO_ORACLE_DIR is not set (the fork's dump directory, ref-text-1k)");
    }
    d
}

/// The oracle's text as token ids (`ids_0.u32`, little endian u32), a real prefix for the head's tests.
pub fn oracle_ids() -> Option<Vec<u32>> {
    let dir = oracle_dir()?;
    let bytes = std::fs::read(dir.join("ids_0.u32")).unwrap_or_else(|e| panic!("ids_0.u32: {e}"));
    Some(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// The natural-text prompts: `(name, token ids)` from `REXL3_MTP_PROMPTS_DIR/*.u32`, sorted by name.
pub fn prompts() -> Option<Vec<(String, Vec<u32>)>> {
    let dir = match std::env::var_os("REXL3_MTP_PROMPTS_DIR") {
        Some(d) => PathBuf::from(d),
        None => {
            eprintln!("not run: REXL3_MTP_PROMPTS_DIR is not set (tools/nemotron_omni/mtp_vocab.py prompts)");
            return None;
        }
    };
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "u32") {
            let bytes = std::fs::read(&p).unwrap();
            let ids = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            out.push((p.file_stem().unwrap().to_string_lossy().into_owned(), ids));
        }
    }
    out.sort();
    if out.is_empty() {
        eprintln!("not run: no .u32 prompt in {}", dir.display());
        return None;
    }
    Some(out)
}

/// The model is driven from one thread at a time under the mutex (its stream and buffers are never touched concurrently); the observer
/// trait object it may hold is not `Send`, so the shared cell says so.
pub struct Shared(NemotronModel);

// SAFETY: a `NemotronModel` is only ever used under the `Mutex` below, one thread at a time.
unsafe impl Send for Shared {}

impl std::ops::Deref for Shared {
    type Target = NemotronModel;
    fn deref(&self) -> &NemotronModel {
        &self.0
    }
}

impl std::ops::DerefMut for Shared {
    fn deref_mut(&mut self) -> &mut NemotronModel {
        &mut self.0
    }
}

/// The shared model: the whole 88-layer trunk and the MTP head, loaded once per process.
pub fn model() -> Option<MutexGuard<'static, Shared>> {
    static M: OnceLock<Mutex<Shared>> = OnceLock::new();
    let dir = pack_dir()?;
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    if M.get().is_none() {
        gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
        let cfg = config(&dir);
        let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
        let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
        // The model's default is the restricted draft head (the 49,152-id list). These tests compare the head with the full-vocabulary
        // reference and with the full head's own argmax, so they run the FULL head unless the run says otherwise (`REXL3_MTP_VOCAB=default`
        // or `=<file>`), as the load-time test does for each setting.
        if std::env::var_os("REXL3_MTP_VOCAB").is_none() {
            std::env::set_var("REXL3_MTP_VOCAB", "off");
        }
        let opts = ModelOptions {
            slots: SLOTS,
            kv_pages: KV_PAGES,
            max_context: MAX_CONTEXT,
            prefill_rows: 64,
            layers: None,
            mtp: true,
        };
        let t = std::time::Instant::now();
        let m = match NemotronModel::load(&dir, &cfg, &plan, opts) {
            Ok(m) => m,
            Err(ModelError::NotBuilt(what)) => {
                eprintln!("not run: {what} is not built yet");
                return None;
            }
            Err(e) => panic!("load: {e}"),
        };
        eprintln!("model loaded in {:.1} s", t.elapsed().as_secs_f64());
        let _ = M.set(Mutex::new(Shared(m)));
    }
    let g = M.get().unwrap().lock().unwrap_or_else(|p| p.into_inner());
    Some(g)
}

/// Prefill `prompt` into `slot` in chunks of `chunk` rows (as the backend does, on the absolute grid) and return the greedy token after it.
pub fn prefill(m: &mut NemotronModel, slot: usize, prompt: &[u32], chunk: usize) -> u32 {
    m.end_slot(slot);
    m.begin_slot(slot).unwrap();
    let mut at = 0;
    let mut first = None;
    while at < prompt.len() {
        let end = (at + chunk - at % chunk).min(prompt.len());
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: at as u64, tokens: prompt[at..end].to_vec() }]);
        let want = (end == prompt.len()).then_some(b.rows - 1);
        if let Some(c) = m.prefill_chunk(&b, want).unwrap() {
            first = Some(c.ids[0] as u32);
        }
        at = end;
    }
    first.unwrap()
}

/// One window of `slot`: `tokens` at positions from `first_position`; the greedy token of every row.
pub fn window_argmax(m: &mut NemotronModel, slot: usize, tokens: &[u32], first_position: usize) -> Vec<u32> {
    let b = RowBatch::from_requests(&[RowRequest { slot, tokens: tokens.to_vec(), first_position: first_position as u64 }]);
    let c = m.window_candidates(&b, 1).unwrap();
    (0..tokens.len()).map(|r| c.ids[r * c.count] as u32).collect()
}

/// Plain decoding: one row per step; every token generated (the prompt's greedy successor first).
pub fn serial_run(m: &mut NemotronModel, slot: usize, prompt: &[u32], n_new: usize) -> Vec<u32> {
    let mut tokens = prompt.to_vec();
    tokens.push(prefill(m, slot, prompt, 64));
    while tokens.len() - prompt.len() < n_new {
        let last = *tokens.last().unwrap();
        let next = window_argmax(m, slot, &[last], tokens.len() - 1)[0];
        m.commit(slot, 1);
        tokens.push(next);
    }
    m.flush().unwrap();
    tokens
}

#[derive(Default, Debug, Clone)]
pub struct Run {
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
    /// Rounds in which the head had nothing to offer.
    pub undrafted: usize,
}

/// Speculative decoding with the head at `depth`: draft, verify a window of `1 + drafts` rows, commit the accepted prefix.
pub fn spec_run(m: &mut NemotronModel, slot: usize, prompt: &[u32], n_new: usize, depth: usize) -> (Vec<u32>, Run) {
    let mut run = Run::default();
    let mut tokens = prompt.to_vec();
    tokens.push(prefill(m, slot, prompt, 64));
    while tokens.len() - prompt.len() < n_new {
        let left = n_new - (tokens.len() - prompt.len());
        let max = depth.min(left - 1);
        let drafts: Vec<u32> = if max > 0 {
            match m.mtp_draft(&[DraftAsk { slot, tokens: &tokens, depth: max }], &[]).unwrap().remove(0) {
                Some(d) => d.ids,
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };
        if drafts.is_empty() {
            run.undrafted += 1;
        }
        let mut window = vec![*tokens.last().unwrap()];
        window.extend_from_slice(&drafts);
        let samples = window_argmax(m, slot, &window, tokens.len() - 1);
        let acc = drafts.iter().zip(&samples).take_while(|(d, s)| d == s).count();
        m.commit(slot, acc + 1);
        tokens.extend_from_slice(&samples[..=acc]);
        run.rounds += 1;
        run.drafted += drafts.len();
        run.accepted += acc;
    }
    m.flush().unwrap();
    assert_eq!(tokens.len() - prompt.len(), n_new, "the engine's cap leaves no overshoot");
    (tokens, run)
}

pub fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The top candidates of one row: ids and the value bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowCands {
    pub ids: Vec<i32>,
    pub val_bits: Vec<u32>,
}

/// How many candidates per row the exactness runs compare.
pub const CMP_CANDS: usize = 8;

/// A window of `slot` with the first `CMP_CANDS` candidates of every row.
pub fn window_cands(m: &mut NemotronModel, slot: usize, tokens: &[u32], first_position: usize) -> Vec<RowCands> {
    let b = RowBatch::from_requests(&[RowRequest { slot, tokens: tokens.to_vec(), first_position: first_position as u64 }]);
    let c = m.window_candidates(&b, CMP_CANDS).unwrap();
    (0..tokens.len())
        .map(|r| RowCands {
            ids: c.ids[r * c.count..(r + 1) * c.count].to_vec(),
            val_bits: c.vals[r * c.count..(r + 1) * c.count].iter().map(|v| v.to_bits()).collect(),
        })
        .collect()
}

/// A prefill of `prompt` into `slot`, `chunk` rows at a time. With `drafting` false the slot's head bookkeeping is switched off right
/// after `begin_slot` (the closest thing to a model without the head: no tap is moved for it, no entry is built). `taps` collects the
/// trunk's post-`norm_f` row of every prompt position (f32). Returns the greedy token after the prompt and its candidates.
pub fn prefill_ex(
    m: &mut NemotronModel,
    slot: usize,
    prompt: &[u32],
    chunk: usize,
    drafting: bool,
    mut taps: Option<&mut Vec<f32>>,
) -> (u32, RowCands) {
    m.end_slot(slot);
    m.begin_slot(slot).unwrap();
    if !drafting {
        m.mtp_disable_slot(slot);
    }
    let mut at = 0;
    let mut first = None;
    while at < prompt.len() {
        let end = (at + chunk - at % chunk).min(prompt.len());
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: at as u64, tokens: prompt[at..end].to_vec() }]);
        let want = (end == prompt.len()).then_some(b.rows - 1);
        if let Some(c) = m.prefill_chunk(&b, want).unwrap() {
            first = Some((
                c.ids[0] as u32,
                RowCands {
                    ids: c.ids[..CMP_CANDS].to_vec(),
                    val_bits: c.vals[..CMP_CANDS].iter().map(|v| v.to_bits()).collect(),
                },
            ));
        }
        if let Some(t) = taps.as_deref_mut() {
            // the slice's post-norm_f rows are still in the trunk's step buffer (the forward ended synchronised)
            let rows = end - at;
            let mut v = vec![0f32; rows * 4096];
            m.core.bufs.xn.download_range(0, &mut v).unwrap();
            t.extend(v);
        }
        at = end;
    }
    let (id, c) = first.unwrap();
    (id, c)
}

/// What a generation run saw: the tokens, the candidates that produced each generated token (index 0 is the token after the prompt), and
/// the trunk's post-`norm_f` row of every position fed.
pub struct Gen {
    pub tokens: Vec<u32>,
    pub cands: Vec<RowCands>,
    pub taps: Vec<f32>,
}

/// Plain decoding in `slot` (head bookkeeping off): one row per step.
pub fn serial_ex(m: &mut NemotronModel, slot: usize, prompt: &[u32], n_new: usize) -> Gen {
    let mut taps = Vec::new();
    let (first, c0) = prefill_ex(m, slot, prompt, 64, false, Some(&mut taps));
    let mut g = Gen { tokens: prompt.to_vec(), cands: vec![c0], taps: Vec::new() };
    g.tokens.push(first);
    while g.tokens.len() - prompt.len() < n_new {
        let last = *g.tokens.last().unwrap();
        let mut rc = window_cands(m, slot, &[last], g.tokens.len() - 1);
        taps.extend(m.mtp_tap_rows(1).unwrap());
        m.commit(slot, 1);
        let c = rc.remove(0);
        g.tokens.push(c.ids[0] as u32);
        g.cands.push(c);
    }
    m.flush().unwrap();
    g.taps = taps;
    g
}

/// Speculative decoding with the head at `depth` in `slot`: the same tokens as [`serial_ex`] by construction, with the candidates of every
/// committed row.
pub fn spec_ex(m: &mut NemotronModel, slot: usize, prompt: &[u32], n_new: usize, depth: usize) -> (Gen, Run) {
    let mut run = Run::default();
    let (first, c0) = prefill_ex(m, slot, prompt, 64, true, None);
    let mut g = Gen { tokens: prompt.to_vec(), cands: vec![c0], taps: Vec::new() };
    g.tokens.push(first);
    while g.tokens.len() - prompt.len() < n_new {
        let left = n_new - (g.tokens.len() - prompt.len());
        let max = depth.min(left - 1);
        let drafts: Vec<u32> = if max > 0 {
            match m.mtp_draft(&[DraftAsk { slot, tokens: &g.tokens, depth: max }], &[]).unwrap().remove(0) {
                Some(d) => d.ids,
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };
        if drafts.is_empty() {
            run.undrafted += 1;
        }
        let mut window = vec![*g.tokens.last().unwrap()];
        window.extend_from_slice(&drafts);
        let rows = window_cands(m, slot, &window, g.tokens.len() - 1);
        let samples: Vec<u32> = rows.iter().map(|c| c.ids[0] as u32).collect();
        let acc = drafts.iter().zip(&samples).take_while(|(d, s)| d == s).count();
        m.commit(slot, acc + 1);
        for (j, c) in rows.into_iter().enumerate().take(acc + 1) {
            g.tokens.push(c.ids[0] as u32);
            g.cands.push(c);
            let _ = j;
        }
        run.rounds += 1;
        run.drafted += drafts.len();
        run.accepted += acc;
    }
    m.flush().unwrap();
    g.tokens.truncate(prompt.len() + n_new);
    g.cands.truncate(n_new);
    (g, run)
}

/// The trunk state of two slots that have fed the same tokens, bit for bit: the Mamba S and conv of all 40 layers and the KV rows
/// `0..fed` of all 8 attention layers.
pub fn same_trunk_state(m: &mut NemotronModel, a: usize, b: usize, fed: usize) -> Result<(), String> {
    for ord in 0..40 {
        let (sa, ca) = m.debug_mamba_state(ord, a).map_err(|e| e.to_string())?;
        let (sb, cb) = m.debug_mamba_state(ord, b).map_err(|e| e.to_string())?;
        if ca != cb || sa.iter().zip(&sb).any(|(x, y)| x.to_bits() != y.to_bits()) {
            return Err(format!("Mamba layer {ord}: the state of slot {a} differs from slot {b}"));
        }
    }
    for ord in 0..8 {
        let (ka, va) = m.debug_kv_rows(ord, a, 0, fed).map_err(|e| e.to_string())?;
        let (kb, vb) = m.debug_kv_rows(ord, b, 0, fed).map_err(|e| e.to_string())?;
        if ka != kb || va != vb {
            return Err(format!("attention layer {ord}: the KV rows of slot {a} differ from slot {b}"));
        }
    }
    Ok(())
}
