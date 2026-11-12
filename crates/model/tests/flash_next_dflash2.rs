//! GPU tests of speculation with a DFlash2 draft on the whole Qwen3.8-Flash-Next model. All `#[ignore]` and env-gated:
//! `REXL3_FLASH_NEXT_DIR` (the target pack), `REXL3_FN_DRAFT_DIR` (a local DFlash2 draft: the published checkpoint or an EXL3
//! quantization of it) and `REXL3_FLASH_NEXT_REF` (a directory with `ids_0.u32`, little-endian u32 prompt ids, at least 2,400 of them).
//! Without them a test prints "not run". One model per process: run with `--test-threads=1` under the GPU lock.
//!
//! ```text
//! REXL3_FLASH_NEXT_DIR=~/models/flashnext-exl3-3.05bpw REXL3_FN_DRAFT_DIR=<draft> REXL3_FLASH_NEXT_REF=<dir with ids_0.u32> \
//!   cargo test --release -p gb-model --test flash_next_dflash2 -- --ignored --nocapture --test-threads=1
//! ```
//!
//! * `dflash2_in_the_loop_equals_serial`: the draft proposes up to seven tokens, a verify window decides, the accepted prefix is committed:
//!   the greedy stream and every byte of the slot's state equal plain one-row decoding, on a short prompt and on one past the QSA dense
//!   threshold and the draft's 2,048-position window, at depths 7 and 3; the draft is good, not merely harmless (an acceptance floor).
//! * `two_sequences_draft_in_one_round_and_equal_their_serial_runs`: both sequences draft in one batched call and verify in one window of
//!   two sequences: each stream and state equals its serial run.
//! * `a_snapshot_keeps_the_drafts_window`: a prefix snapshot restored into another slot gives that slot the same draft context, bit for bit,
//!   and, after the same suffix, the same drafts as the slot it was taken from.
use gb_engine::RowRequest;
use gb_model::families::qwen_flash_next::dflash;
use gb_model::families::qwen_flash_next::engine::load_draft;
use gb_model::families::qwen_flash_next::model::{DflashAsk, FlashNextModel, ModelOptions};
use gb_model::families::qwen_flash_next::prefix::{self, SliceRequest, SnapshotArena};
use gb_model::families::qwen_flash_next::weights::WeightMap;
use gb_model::forward::PrefillPath;
use gb_model::rows::RowBatch;
use gb_sched::prefix::CacheBudget;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const SLOTS: usize = 4;
const PAGES: usize = 96;
const MAX_CONTEXT: usize = 8192;
const CHUNK: usize = 1024;

struct Fixture {
    dir: PathBuf,
    draft: PathBuf,
    ids: Vec<u32>,
}

fn fixture() -> Option<Fixture> {
    let get = |k: &str| std::env::var_os(k).map(PathBuf::from);
    let (Some(dir), Some(draft), Some(refdir)) =
        (get("REXL3_FLASH_NEXT_DIR"), get("REXL3_FN_DRAFT_DIR"), get("REXL3_FLASH_NEXT_REF"))
    else {
        eprintln!("not run: needs REXL3_FLASH_NEXT_DIR, REXL3_FN_DRAFT_DIR and REXL3_FLASH_NEXT_REF");
        return None;
    };
    let raw = std::fs::read(refdir.join("ids_0.u32")).unwrap();
    let ids: Vec<u32> = raw.chunks_exact(4).map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect();
    assert!(ids.len() >= 2400, "{} prompt ids, the long case needs 2,400", ids.len());
    Some(Fixture { dir, draft, ids })
}

/// The target without its MTP head, with the draft enabled, as `--draft` boots it.
fn model(f: &Fixture) -> FlashNextModel {
    let opts = ModelOptions { slots: SLOTS, kv_pages: PAGES, max_context: MAX_CONTEXT, prefill_rows: CHUNK, warm: false };
    let mut m = FlashNextModel::load_with(&f.dir, opts, false).unwrap();
    assert!(!m.has_mtp(), "a --draft boot leaves the MTP head on disk");
    let weights = WeightMap::from_dir(&f.dir, &m.cfg).unwrap();
    let d = load_draft(&f.draft, &m.cfg, &weights, SLOTS, CHUNK, 0).unwrap();
    eprintln!(
        "draft {} ({}, {}): {:.2} GiB",
        d.check.fingerprint,
        d.form,
        d.check.geometry,
        d.total_bytes as f64 / (1u64 << 30) as f64
    );
    m.enable_dflash(d.draft, d.weight_bytes).unwrap();
    m
}

/// Prefill `prompt` into `slot` in chunks and return the greedy token after it.
fn prefill(m: &mut FlashNextModel, slot: usize, prompt: &[u32]) -> u32 {
    m.begin_slot(slot).unwrap();
    m.set_dflash_from(slot, dflash::ctx_from(prompt.len(), &[], 0));
    let path = PrefillPath::for_prompt(prompt.len());
    let mut at = 0;
    let mut first = None;
    while at < prompt.len() {
        let end = (at + CHUNK).min(prompt.len());
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: at as u64, tokens: prompt[at..end].to_vec() }]);
        if let Some(c) = m.prefill_chunk(&b, path, (end == prompt.len()).then_some(1)).unwrap() {
            first = Some(c.ids[0] as u32);
        }
        at = end;
    }
    first.unwrap()
}

/// One window over several sequences: per sequence, the argmax of each of its rows.
fn window_argmax(m: &mut FlashNextModel, reqs: &[(usize, Vec<u32>, usize)]) -> Vec<Vec<u32>> {
    let rr: Vec<RowRequest> =
        reqs.iter().map(|(slot, t, p)| RowRequest { slot: *slot, tokens: t.clone(), first_position: *p as u64 }).collect();
    let b = RowBatch::from_requests(&rr);
    let c = m.window_candidates(&b, 1).unwrap();
    (0..rr.len()).map(|i| b.seq_rows(i).map(|r| c.ids[r] as u32).collect()).collect()
}

fn serial_run(m: &mut FlashNextModel, slot: usize, prompt: &[u32], n_new: usize) -> Vec<u32> {
    let mut tokens = prompt.to_vec();
    tokens.push(prefill(m, slot, prompt));
    while tokens.len() - prompt.len() < n_new {
        let next = window_argmax(m, &[(slot, vec![*tokens.last().unwrap()], tokens.len() - 1)])[0][0];
        m.commit(slot, 1);
        tokens.push(next);
    }
    tokens
}

#[derive(Default, Debug, Clone, Copy)]
struct Run {
    rounds: usize,
    drafted: usize,
    accepted: usize,
}

impl Run {
    fn tokens_per_round(&self) -> f64 {
        (self.rounds + self.accepted) as f64 / self.rounds.max(1) as f64
    }
}

/// Speculative decoding of several sequences together with the DFlash2 draft at `depth`: one batched draft call for every sequence that
/// still needs tokens, one verify window over all of them, each one's accepted prefix committed.
fn spec_runs(m: &mut FlashNextModel, seqs: &[(usize, &[u32])], n_new: usize, depth: usize) -> Vec<(Vec<u32>, Run)> {
    let mut out: Vec<(Vec<u32>, Run)> = seqs
        .iter()
        .map(|&(slot, prompt)| {
            let mut t = prompt.to_vec();
            t.push(prefill(m, slot, prompt));
            (t, Run::default())
        })
        .collect();
    loop {
        let live: Vec<usize> = (0..seqs.len()).filter(|&i| out[i].0.len() - seqs[i].1.len() < n_new).collect();
        if live.is_empty() {
            break;
        }
        let maxes: Vec<usize> = live.iter().map(|&i| depth.min(n_new - (out[i].0.len() - seqs[i].1.len()) - 1)).collect();
        let ask_idx: Vec<usize> = (0..live.len()).filter(|&k| maxes[k] > 0).collect();
        for &k in &ask_idx {
            let i = live[k];
            assert!(
                m.dflash_ready(seqs[i].0, out[i].0.len()),
                "the draft's context must follow the committed tokens of slot {}",
                seqs[i].0
            );
        }
        let asks: Vec<DflashAsk<'_>> = ask_idx
            .iter()
            .map(|&k| {
                let i = live[k];
                DflashAsk { slot: seqs[i].0, tokens: &out[i].0, depth: maxes[k], sampling: None }
            })
            .collect();
        let mut drafts = vec![Vec::new(); live.len()];
        if !asks.is_empty() {
            for (k, d) in ask_idx.iter().zip(m.dflash_draft(&asks).unwrap()) {
                drafts[*k] = d;
            }
        }
        let reqs: Vec<(usize, Vec<u32>, usize)> = live
            .iter()
            .enumerate()
            .map(|(k, &i)| {
                let mut w = vec![*out[i].0.last().unwrap()];
                w.extend_from_slice(&drafts[k]);
                (seqs[i].0, w, out[i].0.len() - 1)
            })
            .collect();
        let samples = window_argmax(m, &reqs);
        for (k, &i) in live.iter().enumerate() {
            let acc = drafts[k].iter().zip(&samples[k]).take_while(|(d, s)| d == s).count();
            m.commit(seqs[i].0, acc + 1);
            out[i].0.extend_from_slice(&samples[k][..=acc]);
            out[i].1.rounds += 1;
            out[i].1.drafted += drafts[k].len();
            out[i].1.accepted += acc;
        }
    }
    out
}

fn hash_range(h: &mut Sha256, buf: &gb_cuda::DevBuf, at: usize, n: usize) {
    let mut data = vec![0u8; n];
    buf.download_range(at, &mut data).unwrap();
    h.update(data);
}

/// Every byte of `slot`'s state after `len` tokens: GDN state and conv, the PLE ring, the QSA raw-key rings, the KV rows `0..len` and the
/// QSA pooled keys of its complete groups (a model without the MTP head: nothing one-ahead).
fn digest(m: &mut FlashNextModel, slot: usize, len: usize) -> Vec<u8> {
    m.flush().unwrap();
    m.stream.sync().unwrap();
    let mut h = Sha256::new();
    let slots = m.opts.slots;
    for (ss, cc) in m.pools.s.iter().zip(&m.pools.conv) {
        hash_range(&mut h, ss, slot * ss.bytes() / slots, ss.bytes() / slots);
        hash_range(&mut h, cc, slot * cc.bytes() / slots, cc.bytes() / slots);
    }
    hash_range(&mut h, &m.ple_ring, slot * m.ple_ring.bytes() / slots, m.ple_ring.bytes() / slots);
    for ring in &m.qsa_ring {
        hash_range(&mut h, ring, slot * ring.bytes() / slots, ring.bytes() / slots);
    }
    let blocks = m.kv_tab.blocks_of(slot).to_vec();
    for cache in &m.kv {
        let bytes = cache.kv_heads.token_elems() * 2;
        for p in 0..len {
            let at = (blocks[p / 256] as usize * 256 + p % 256) * bytes;
            hash_range(&mut h, &cache.k, at, bytes);
            hash_range(&mut h, &cache.v, at, bytes);
        }
    }
    for pool in &m.qsa_pool {
        for g in 0..len / gb_cuda::qsa::CR {
            hash_range(&mut h, pool, (blocks[g / 64] as usize * 64 + g % 64) * 256, 256);
        }
    }
    h.finalize().to_vec()
}

#[test]
#[ignore]
fn dflash2_in_the_loop_equals_serial() {
    let Some(f) = fixture() else { return };
    let mut m = model(&f);
    let n_new = 96;
    for (name, prompt) in [("short", &f.ids[..300]), ("long", &f.ids[..2400])] {
        let serial = serial_run(&mut m, 0, prompt, n_new);
        let serial_digest = digest(&mut m, 0, serial.len() - 1);
        for depth in [7, 3] {
            let (spec, run) = spec_runs(&mut m, &[(1, prompt)], n_new, depth).remove(0);
            eprintln!(
                "{name} prompt ({} tokens), depth {depth}: {run:?}, {:.2} tokens a round",
                prompt.len(),
                run.tokens_per_round()
            );
            assert_eq!(spec, serial, "{name} prompt, depth {depth}: the drafted stream differs from the serial one");
            assert_eq!(
                digest(&mut m, 1, spec.len() - 1),
                serial_digest,
                "{name} prompt, depth {depth}: the slot's state differs"
            );
            assert!(run.drafted > 0, "{name} prompt, depth {depth}: nothing was drafted");
            assert!(run.tokens_per_round() > 1.3, "{name} prompt, depth {depth}: the draft is not useful ({run:?})");
        }
    }
}

#[test]
#[ignore]
fn two_sequences_draft_in_one_round_and_equal_their_serial_runs() {
    let Some(f) = fixture() else { return };
    let mut m = model(&f);
    let n_new = 80;
    let (a, b) = (&f.ids[..300], &f.ids[..2400]);
    let sa = serial_run(&mut m, 0, a, n_new);
    let da = digest(&mut m, 0, sa.len() - 1);
    let sb = serial_run(&mut m, 1, b, n_new);
    let db = digest(&mut m, 1, sb.len() - 1);
    let runs = spec_runs(&mut m, &[(2, a), (3, b)], n_new, 7);
    for ((name, slot, serial, d), (spec, run)) in [("short", 2, &sa, &da), ("long", 3, &sb, &db)].into_iter().zip(&runs) {
        eprintln!("{name} prompt in a batch of two: {run:?}, {:.2} tokens a round", run.tokens_per_round());
        assert_eq!(spec, serial, "{name} prompt in a batch of two: the stream differs from its serial run");
        assert_eq!(&digest(&mut m, slot, spec.len() - 1), d, "{name} prompt in a batch of two: the slot's state differs");
        assert!(run.tokens_per_round() > 1.3, "{name}: {run:?}");
    }
}

#[test]
#[ignore]
fn a_snapshot_keeps_the_drafts_window() {
    let Some(f) = fixture() else { return };
    let mut m = model(&f);
    let snap = prefix::snapshot_bytes_with(&m.cfg, false, true);
    let page = prefix::page_bytes(&m.cfg, false);
    let mut arena =
        SnapshotArena::new(&m, CacheBudget { snapshot_slots: 2, page_blocks: 16, snapshot_bytes: snap, page_bytes: page })
            .unwrap();
    let prompt = &f.ids[..2400];
    let cut = 2304;
    // slot 0: the whole prompt in slices, a snapshot at the cut
    let mut done = 0;
    let mut captured = Vec::new();
    while done < prompt.len() {
        let r = prefix::slice_with_cuts(
            &mut m,
            &mut arena,
            SliceRequest { slot: 0, prompt, done, candidates: 1, cuts: &[cut], chunk: CHUNK },
        )
        .unwrap();
        captured.extend(r.captured);
        done = r.end;
    }
    let handle = captured.into_iter().find(|c| c.len == cut).expect("a snapshot at the cut").handle;
    let lo = m.dflash_book(0).unwrap().window_of(cut).start;
    // the ring is read with a plain download: the model's (non-blocking) stream must have finished its absorbs and copies
    m.stream.sync().unwrap();
    let want = m.dflash().unwrap().draft.ring_window_hash(0, lo, cut - lo).unwrap();
    // slot 1: restore the cut, then the same suffix
    prefix::restore(&mut m, &arena, 1, &handle, cut).unwrap();
    m.stream.sync().unwrap();
    let b = m.dflash_book(1).unwrap();
    assert!(b.ok && b.end == cut && b.lo == lo, "the restored context is the snapshot's: {b:?}");
    assert_eq!(m.dflash().unwrap().draft.ring_window_hash(1, lo, cut - lo).unwrap(), want, "the restored draft window differs");
    let mut done = cut;
    while done < prompt.len() {
        let r = prefix::slice_with_cuts(
            &mut m,
            &mut arena,
            SliceRequest { slot: 1, prompt, done, candidates: 1, cuts: &[], chunk: CHUNK },
        )
        .unwrap();
        done = r.end;
    }
    assert!(m.dflash_ready(0, prompt.len() + 1) && m.dflash_ready(1, prompt.len() + 1));
    let mut toks = prompt.to_vec();
    toks.push(f.ids[2400]);
    let d0 = m.dflash_draft(&[DflashAsk { slot: 0, tokens: &toks, depth: 7, sampling: None }]).unwrap();
    let d1 = m.dflash_draft(&[DflashAsk { slot: 1, tokens: &toks, depth: 7, sampling: None }]).unwrap();
    assert_eq!(d0, d1, "the restored slot drafts differently from the slot the snapshot was taken from");
    prefix::release(&mut m, &mut arena, handle).unwrap();
}
