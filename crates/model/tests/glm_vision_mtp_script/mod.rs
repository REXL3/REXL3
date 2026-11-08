//! A drafting script over IMAGE prompts on a GLM backend (`GlmBackend` on one node, `GlmTpBackend` at rank 0 of a group): what the
//! engine does with `--spec mtp` after a prompt with images, written out so that every number it sees can be digested. Shared by
//! `glm_tp_vision_mtp.rs` (two ranks against one model, lane L14) and `glm_vision_mtp_probe.rs` (the single-node probe that must be
//! identical before and after the merge of lanes L12 and L13). Uses only APIs that exist at both lanes' merge base (`ab6a0771`).
//!
//! Three sequences: slot 0 an image prompt fed in 512-row slices (text, image A, text, image B, text: both images inside one chunk);
//! slot 1 an image prompt fed in 128-row slices that ENDS with an image (text, image A across two chunk boundaries, text, image B as the
//! prompt's last rows: the drafter's carry after the prompt is an image row's tap); slot 2 a 37-token text prompt. Then `ROUNDS` rounds
//! of `propose_batch` for every slot (slot 1 sampled, so its drafts are keyed; slots 0 and 2 greedy), a verify window of the last token and
//! the drafts, and commits whose keep counts follow a fixed pattern (as `glm_mtp_script`). A digest line per answering call (candidates,
//! every fp16 logit), per round (the drafts) and the keeps.
#![allow(dead_code)]

use gb_engine::{Backend, ImageSpan, ProposeRequest, RowOutput, RowRequest};
use gb_sampling::Sampling;
use std::sync::Arc;

pub const ROUNDS: usize = 10;
pub const COUNT: usize = 8;
pub const SLOTS: usize = 3;

pub fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h = (*h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
}

fn digest_rows(rows: &[RowOutput], logits: &[u16]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for r in rows {
        for &i in &r.ids {
            fnv(&mut h, &i.to_le_bytes());
        }
        for &v in &r.values {
            fnv(&mut h, &v.to_bits().to_le_bytes());
        }
    }
    for x in logits {
        fnv(&mut h, &x.to_le_bytes());
    }
    h
}

/// An encoded image as the script needs it: its rows, its merge grid and the encoder's features handle.
#[derive(Clone)]
pub struct Image {
    pub tokens: usize,
    pub merged: (usize, usize),
    pub features: Arc<dyn std::any::Any + Send + Sync>,
}

/// A prompt and its image spans.
#[derive(Clone)]
pub struct Prompt {
    pub tokens: Vec<u32>,
    pub images: Vec<ImageSpan>,
}

fn push_image(t: &mut Vec<u32>, images: &mut Vec<ImageSpan>, im: &Image, image_token: u32, key: u64) {
    images.push(ImageSpan { start: t.len(), tokens: im.tokens, merged: im.merged, features: im.features.clone(), key });
    t.extend(std::iter::repeat_n(image_token, im.tokens));
}

/// The three prompts: `[slot 0 (text 5, A, text 40, B, text 30), slot 1 (text 20, A, text 10, B: the image ends the prompt), slot 2 (text 37)]`.
pub fn prompts(image_token: u32, a: &Image, b: &Image) -> [Prompt; 3] {
    let mut t0: Vec<u32> = (0..5).map(|i| 1000 + i).collect();
    let mut i0 = Vec::new();
    push_image(&mut t0, &mut i0, a, image_token, 0);
    t0.extend((0..40).map(|i| 2000 + 37 * i));
    push_image(&mut t0, &mut i0, b, image_token, 1);
    t0.extend((0..30).map(|i| 2001 + 37 * i));
    let mut t1: Vec<u32> = (0..20).map(|i| 4000 + 13 * i).collect();
    let mut i1 = Vec::new();
    push_image(&mut t1, &mut i1, a, image_token, 0);
    t1.extend((0..10).map(|i| 5000 + 7 * i));
    push_image(&mut t1, &mut i1, b, image_token, 1);
    let t2: Vec<u32> = (0..37).map(|i| 3000 + 11 * i).collect();
    [Prompt { tokens: t0, images: i0 }, Prompt { tokens: t1, images: i1 }, Prompt { tokens: t2, images: Vec::new() }]
}

/// What a run produced: one line per answering call and per round, and the drafted tokens of every round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    pub lines: Vec<String>,
    pub drafts: Vec<Vec<Option<Vec<u32>>>>,
    pub keeps: Vec<Vec<usize>>,
}

/// The keep count of slot `s` in round `r` with `d` drafts in its window: all of them, none, or some, by a fixed pattern.
fn keep_of(r: usize, s: usize, d: usize) -> usize {
    match (r + s) % 4 {
        0 => d + 1,
        1 => 1,
        2 => 1 + d.min(1),
        _ => 1 + d / 2,
    }
}

/// Runs the script on `be`; `logits(rows)` reads the fp16 logits of the last answering forward's `rows` rows (`GlmModel::logits_host`).
pub fn run<B: Backend>(be: &mut B, prompts: &[Prompt; 3], logits: impl Fn(&B, usize) -> Vec<u16>) -> Run {
    let mut lines = Vec::new();
    let mut seq: Vec<Vec<u32>> = Vec::new();
    for (slot, p) in prompts.iter().enumerate() {
        be.set_prefill_row_limit(if slot == 1 { Some(128) } else { None });
        let mut done = 0;
        loop {
            let step = be.prefill_slice(slot, &p.tokens, &p.images, done, COUNT);
            done = step.done;
            if let Some(row) = step.row {
                let l = logits(be, 1);
                lines.push(format!("prefill {slot} {:016x}", digest_rows(std::slice::from_ref(&row), &l)));
                let mut s = p.tokens.clone();
                s.push(row.ids[0]);
                seq.push(s);
                break;
            }
        }
    }
    be.set_prefill_row_limit(None);
    assert!(be.take_error().is_none(), "the prompts failed");
    let samplers = [
        Sampling { temperature: 0.0, ..Default::default() },
        Sampling { seed: 0x5eed, temperature: 0.7, top_k: 20, top_p: 0.95, min_p: 0.0 },
        Sampling { temperature: 0.0, ..Default::default() },
    ];
    let mut drafts_all = Vec::new();
    let mut keeps_all = Vec::new();
    for round in 0..ROUNDS {
        let reqs: Vec<ProposeRequest> =
            (0..SLOTS).map(|s| ProposeRequest { slot: s, tokens: &seq[s], sampling: &samplers[s], max: 7 }).collect();
        let d = be.propose_batch(&reqs);
        assert!(be.take_error().is_none(), "drafting failed in round {round}");
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for x in &d {
            match x {
                None => fnv(&mut h, &[0]),
                Some(v) => {
                    fnv(&mut h, &(v.len() as u64).to_le_bytes());
                    for t in v {
                        fnv(&mut h, &t.to_le_bytes());
                    }
                }
            }
        }
        lines.push(format!(
            "drafts {round} {:016x} {:?}",
            h,
            d.iter().map(|x| x.as_ref().map_or(0, |v| v.len())).collect::<Vec<_>>()
        ));
        let w: Vec<RowRequest> = (0..SLOTS)
            .map(|s| {
                let mut t = vec![*seq[s].last().unwrap()];
                t.extend(d[s].clone().unwrap_or_default());
                RowRequest { slot: s, tokens: t, first_position: (seq[s].len() - 1) as u64 }
            })
            .collect();
        let rows: usize = w.iter().map(|q| q.tokens.len()).sum();
        let out = be.step(&w, COUNT);
        assert!(be.take_error().is_none(), "the window of round {round} failed");
        let l = logits(be, rows);
        let flat: Vec<RowOutput> = out.iter().flatten().cloned().collect();
        lines.push(format!("window {round} rows={rows} {:016x}", digest_rows(&flat, &l)));
        let mut keeps = Vec::new();
        for s in 0..SLOTS {
            let nd = w[s].tokens.len() - 1;
            let keep = keep_of(round, s, nd);
            be.commit(s, keep);
            seq[s].extend_from_slice(&w[s].tokens[1..keep]);
            seq[s].push(out[s][keep - 1].ids[0]);
            keeps.push(keep);
        }
        drafts_all.push(d);
        keeps_all.push(keeps);
    }
    for s in 0..SLOTS {
        be.release(s);
    }
    be.quiesce().unwrap();
    Run { lines, drafts: drafts_all, keeps: keeps_all }
}

/// A deterministic synthetic RGB image (`glm_tp_vision.rs`'s): gradients, a checkerboard and a disc, so the tower sees structure.
pub fn synth(width: usize, height: usize, seed: u32) -> gb_vision::Rgb {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (x as f64 / width as f64, y as f64 / height as f64);
            let disc = ((fx - 0.5).powi(2) + (fy - 0.45).powi(2)).sqrt() < 0.25;
            let check = ((x / 32) + (y / 32) + seed as usize) % 2 == 0;
            let r = if disc { 230 } else { (fx * 200.0) as u8 };
            let g = if check { 180 } else { (fy * 160.0) as u8 };
            let b = ((fx + fy) * 100.0) as u8 ^ (seed as u8).wrapping_mul(37);
            data.extend_from_slice(&[r, g, b]);
        }
    }
    gb_vision::Rgb { width, height, data }
}

/// Two synthetic images (448x448: 256 rows; 336x224: 96 rows) through the server's preprocessing of `dir` and the encoder of the tower of
/// `pack` (`GlmEncoder` over `GlmVisionTower`), and the pack's image token. The tower is dropped before this returns: the features
/// live on their own.
pub fn encode_two(dir: &std::path::Path, pack: &gb_model::families::glm::GlmPack) -> (Image, Image, u32) {
    use gb_model::families::glm::vision::{GlmVisionTower, TowerOptions};
    use gb_model::families::glm::vision_serve::{FeatureBudget, GlmEncoder};
    use gb_model::serve::{ImageEncoder, ServeOptions};
    const GIB: u64 = 1 << 30;
    let mut o = ServeOptions::new(dir.to_path_buf());
    o.max_image_tokens = 1024;
    o.vision = Some(Default::default());
    let pp = gb_model::families::glm::image_preprocessor(dir, &o).unwrap().expect("the pack has a vision tower");
    let tower = GlmVisionTower::load(pack, dir, &TowerOptions::new(1024 * 4, 8 * GIB)).unwrap();
    let mut enc = GlmEncoder::new(tower, FeatureBudget::new(4 * GIB), &pp.id()).unwrap();
    let image_token = pack.config.vision.as_ref().unwrap().image_token_id;
    let mut one = |rgb: gb_vision::Rgb| {
        let e = enc.encode(&pp.preprocess(&rgb).unwrap()).unwrap();
        Image { tokens: e.tokens, merged: e.merged, features: e.features }
    };
    let a = one(synth(448, 448, 1));
    let b = one(synth(336, 224, 2));
    (a, b, image_token)
}
