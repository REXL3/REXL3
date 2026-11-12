//! GPU tests of speculation with a DFlash2 draft on the whole GLM-5.3-Flash model. All `#[ignore]` and env-gated: REXL3_GLM_DIR (the target
//! pack), REXL3_GLM_DRAFT_DIR (a local copy of the draft: the published `incoai/GLM-5.3-Flash-DFlash2` checkpoint as downloaded, or the
//! operator's own EXL3 quantization of it; never a hosted pack) and REXL3_GLM_ORACLE_DIR (the oracle's dumps, for prompt token ids). Without
//! them a test prints "not run". One model per process: run with `--test-threads=1` under the GPU lock.
//!
//! * `the_taps_are_the_mean_of_the_streams_after_each_tapped_layer`: the tap buffer's rows are, bit for bit, the bf16 mean of the four streams
//!   the capture saw right after layers 5, 14, 24, 33 and 42 (HF `hidden_states[i + 1]`), in a window and in a prompt chunk.
//! * `dflash2_in_the_loop_equals_serial`: the draft proposes up to seven tokens, a verify window decides, the accepted prefix is committed: the
//!   greedy stream and the trunk's state equal plain one-row decoding, bit for bit, on short prompts and on one that crosses the DSA boundary;
//!   the draft is good, not merely harmless (its acceptance has a floor).
//! * `a_snapshot_keeps_the_drafts_window`: a prefix snapshot restored into another slot gives that slot the same draft context, bit for bit, and
//!   the same drafts as the slot it was taken from.
//!
//! The pack-gate tests at the end read headers only (no GPU, not ignored): each runs when its folders are given (REXL3_GLM_DIR with
//! REXL3_GLM_DRAFT_DIR, REXL3_GLM_DRAFT_USER_DIR: a draft quantization the operator made for their own use, REXL3_QWEN_DIR with
//! REXL3_DRAFT_DIR: the Qwen3.8 pair) and prints "not run" otherwise.

mod glm_support;

use gb_engine::RowRequest;
use gb_model::draft::{Draft, DraftGeom, BLOCK};
use gb_model::families::glm::dflash;
use gb_model::families::glm::model::{Capture, DflashAsk, GlmModel};
use gb_model::families::glm::npy;
use gb_model::rows::RowBatch;
use glm_support::*;
use std::path::{Path, PathBuf};

const SLOTS: usize = 4;
const KV_BLOCKS: usize = 64;
const MAX_CONTEXT: usize = 8192;

fn draft_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_GLM_DRAFT_DIR") {
        Ok(d) if Path::new(&d).join("config.json").exists() => Some(PathBuf::from(d)),
        Ok(d) => panic!("REXL3_GLM_DRAFT_DIR={d} has no config.json"),
        Err(_) => {
            eprintln!("REXL3_GLM_DRAFT_DIR is not set: not run");
            None
        }
    }
}

fn oracle_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_GLM_ORACLE_DIR") {
        Ok(d) if Path::new(&d).is_dir() => Some(PathBuf::from(d)),
        Ok(d) => panic!("REXL3_GLM_ORACLE_DIR={d} is not a directory"),
        Err(_) => {
            eprintln!("REXL3_GLM_ORACLE_DIR is not set: not run");
            None
        }
    }
}

fn ids_of(oracle: &Path, prompt: &str) -> Vec<u32> {
    npy::read(&oracle.join(prompt).join("ids.npy")).unwrap().to_i32().unwrap().iter().map(|&x| x as u32).collect()
}

/// The shared model with the draft enabled (loaded once per process).
fn model() -> Option<std::sync::MutexGuard<'static, GlmModel>> {
    use std::sync::{Mutex, OnceLock};
    static M: OnceLock<Option<Mutex<GlmModel>>> = OnceLock::new();
    let m = M.get_or_init(|| {
        let ddir = draft_dir()?;
        let mut m = load_model(SLOTS, KV_BLOCKS, MAX_CONTEXT)?;
        let geom = dflash::check_dir(&ddir, m.cfg.hidden, m.cfg.vocab, m.layers()).unwrap();
        assert_eq!(geom, DraftGeom::GLM53_FLASH);
        let pack = gb_model::pack::Pack::open(&ddir).unwrap();
        let mut ctx = gb_model::loadctx::LoadCtx::from_env();
        let head_bits = m.dev.lm_head.bits;
        let (d, bytes) =
            Draft::declare_geom(&mut ctx, &pack, geom, head_bits, m.cfg.vocab, SLOTS, m.chunk_rows().min(2048), BLOCK).unwrap();
        ctx.finish().unwrap();
        eprintln!("draft loaded: {:.2} GiB ({})", gib(bytes as u64), if pack.has_group("fc") { "exl3" } else { "bf16" });
        m.enable_dflash(d, bytes as u64).unwrap();
        Some(Mutex::new(m))
    });
    m.as_ref().map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Prefill `prompt` into `slot` in the model's chunks and return the greedy token after it.
fn prefill(m: &mut GlmModel, slot: usize, prompt: &[u32]) -> u32 {
    m.begin_slot(slot).unwrap();
    m.set_dflash_from(slot, dflash::ctx_from(prompt.len(), &[], 0));
    let mut at = 0;
    let mut first = None;
    while at < prompt.len() {
        let end = (at + m.chunk_rows()).min(prompt.len());
        let want = (end == prompt.len()).then_some(1);
        if let Some(c) = m.prefill_chunk(slot, at, &prompt[at..end], want).unwrap() {
            first = Some(c.ids[0] as u32);
        }
        at = end;
    }
    first.unwrap()
}

fn window_argmax(m: &mut GlmModel, slot: usize, tokens: &[u32], first_position: usize) -> Vec<u32> {
    let b = RowBatch::from_requests(&[RowRequest { slot, tokens: tokens.to_vec(), first_position: first_position as u64 }]);
    let c = m.window_candidates(&b, 1).unwrap();
    (0..tokens.len()).map(|r| c.ids[r] as u32).collect()
}

fn serial_run(m: &mut GlmModel, slot: usize, prompt: &[u32], n_new: usize) -> Vec<u32> {
    let mut tokens = prompt.to_vec();
    tokens.push(prefill(m, slot, prompt));
    while tokens.len() - prompt.len() < n_new {
        let last = *tokens.last().unwrap();
        let next = window_argmax(m, slot, &[last], tokens.len() - 1)[0];
        m.commit(slot, 1);
        tokens.push(next);
    }
    tokens
}

#[derive(Default, Debug)]
struct Run {
    rounds: usize,
    drafted: usize,
    accepted: usize,
}

/// Speculative decoding with the DFlash2 draft at `depth`: draft, verify `1 + drafts` rows, commit the accepted prefix.
fn spec_run(m: &mut GlmModel, slot: usize, prompt: &[u32], n_new: usize, depth: usize) -> (Vec<u32>, Run) {
    let mut run = Run::default();
    let mut tokens = prompt.to_vec();
    tokens.push(prefill(m, slot, prompt));
    while tokens.len() - prompt.len() < n_new {
        let left = n_new - (tokens.len() - prompt.len());
        let max = depth.min(left - 1);
        let drafts: Vec<u32> = if max > 0 {
            // the last commit is applied lazily, at the next forward or flush: the draft's context joins it there
            m.flush().unwrap();
            assert!(m.dflash_ready(slot, tokens.len()), "the draft's context must follow the committed tokens");
            m.dflash_draft(&[DflashAsk { slot, tokens: &tokens, depth: max, sampling: None }]).unwrap().remove(0)
        } else {
            Vec::new()
        };
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
    (tokens, run)
}

fn assert_same_state(m: &mut GlmModel, a: usize, b: usize, fed: usize, what: &str) {
    for l in 0..45 {
        if l % 4 != 3 {
            let (sa, ca) = m.kda_state(l, a).unwrap();
            let (sb, cb) = m.kda_state(l, b).unwrap();
            assert!(
                ca == cb && sa.iter().zip(&sb).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{what}: KDA layer {l} state differs"
            );
        } else {
            assert!(
                m.latent_rows(l, a, fed).unwrap() == m.latent_rows(l, b, fed).unwrap(),
                "{what}: MLA layer {l} latent rows differ"
            );
            assert!(
                m.pool_rows(l, a, fed / 4).unwrap() == m.pool_rows(l, b, fed / 4).unwrap(),
                "{what}: MLA layer {l} pool rows differ"
            );
        }
    }
}

/// bf16 of an f32, round to nearest even (what the kernels' `__float2bfloat16_rn` does).
fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// The tap rows of the last forward's `rows` rows against the capture's streams after each tapped layer.
fn assert_taps_match(m: &GlmModel, cap: &Capture, rows: usize, what: &str) {
    let h = m.cfg.hidden;
    let taps = m.dflash_tap_rows(rows).unwrap();
    for (t, &l) in dflash::TAP_LAYERS.iter().enumerate() {
        let streams = &cap.layer_out[l];
        assert_eq!(streams.len(), rows * 4 * h, "{what}: the capture of layer {l}");
        for r in 0..rows {
            for i in 0..h {
                let x = |s: usize| streams[r * 4 * h + s * h + i];
                let want = bf16((((x(0) + x(1)) + x(2)) + x(3)) * 0.25);
                let got = taps[r * dflash::TAP_LAYERS.len() * h + t * h + i];
                assert_eq!(got, want, "{what}: row {r} tap of layer {l} channel {i}");
            }
        }
    }
}

#[test]
#[ignore]
fn the_taps_are_the_mean_of_the_streams_after_each_tapped_layer() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    let prompt = ids_of(&oracle, "chat_300");
    m.begin_slot(2).unwrap();
    m.capture = Some(Capture::default());
    m.prefill_chunk(2, 0, &prompt[..40], None).unwrap();
    let cap = m.capture.take().unwrap();
    assert_taps_match(&m, &cap, 40, "a prompt chunk");
    let first = prefill(&mut m, 2, &prompt[..40]);
    m.capture = Some(Capture::default());
    window_argmax(&mut m, 2, &[first, 7, 11], 40);
    let cap = m.capture.take().unwrap();
    assert_taps_match(&m, &cap, 3, "a window");
    m.end_slot(2);
}

#[test]
#[ignore]
fn dflash2_in_the_loop_equals_serial() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    let long = ids_of(&oracle, "long_2300");
    let cases: Vec<(&str, Vec<u32>, usize)> = vec![
        ("chat_300", ids_of(&oracle, "chat_300"), 96),
        ("code_700", ids_of(&oracle, "code_700"), 96),
        ("long_2300 cut at 2,030 (crosses the DSA boundary and the draft's window)", long[..2030].to_vec(), 64),
    ];
    for (name, prompt, n_new) in cases {
        let serial = serial_run(&mut m, 0, &prompt, n_new);
        for depth in [7usize, 3] {
            let (spec, run) = spec_run(&mut m, 1, &prompt, n_new, depth);
            assert_eq!(spec, serial, "{name}: depth {depth}: the drafted token stream differs from the serial one");
            assert_same_state(&mut m, 0, 1, spec.len() - 1, &format!("{name} depth {depth}"));
            let tpr = (run.accepted + run.rounds) as f64 / run.rounds as f64;
            eprintln!(
                "{name}: depth {depth}: {} rounds, {} drafted, {} accepted ({tpr:.2} tokens per round); state equals serial",
                run.rounds, run.drafted, run.accepted
            );
            assert!(
                run.accepted > 0 && tpr > 1.3,
                "{name}: depth {depth}: the draft accepts too little ({tpr:.2} tokens per round)"
            );
        }
    }
}

#[test]
#[ignore]
fn a_snapshot_keeps_the_drafts_window() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    if !m.has_prefix_cache() {
        m.enable_prefix_cache(2).unwrap();
    }
    let prompt = ids_of(&oracle, "long_2300");
    // a prompt of 2,200 tokens snapshotted at its end: the window is its last 2,048 positions
    let p = &prompt[..2200];
    let first = prefill(&mut m, 0, p);
    let snap = m.alloc_snapshot().unwrap();
    m.prefix_snapshot(0, snap, p.len()).unwrap();
    let pages = m.retain_prefix_pages(0, p.len());
    m.restore_prefix(1, snap, &pages, p.len()).unwrap();
    let (a, b) = (m.dflash_book(0).unwrap(), m.dflash_book(1).unwrap());
    assert_eq!((b.lo, b.end), (p.len() - dflash::WINDOW, p.len()), "the restored context is the snapshot's window");
    assert!(a.ready(p.len()) && b.ready(p.len()));
    let lo = p.len() - dflash::WINDOW;
    let (ha, hb) = (
        m.dflash().unwrap().draft.ring_window_hash(0, lo, dflash::WINDOW).unwrap(),
        m.dflash().unwrap().draft.ring_window_hash(1, lo, dflash::WINDOW).unwrap(),
    );
    assert_eq!(ha, hb, "the restored slot's draft window is the snapshot's, bit for bit");
    let mut tokens = p.to_vec();
    tokens.push(first);
    let da = m.dflash_draft(&[DflashAsk { slot: 0, tokens: &tokens, depth: 7, sampling: None }]).unwrap();
    let db = m.dflash_draft(&[DflashAsk { slot: 1, tokens: &tokens, depth: 7, sampling: None }]).unwrap();
    assert_eq!(da, db, "the restored slot drafts what the slot it was taken from drafts");
    assert!(!da[0].is_empty());
    m.release_prefix(snap, &pages);
    m.end_slot(0);
    m.end_slot(1);
}

/// One speculative round under the profiler (`nsys profile --capture-range=cudaProfilerApi`): after a few warm rounds, exactly one draft call
/// (the DFlash2 block, the head over the asked rows, the walk), its verify window and the commit's absorb are bracketed, so a trace shows the
/// draft's share of the round. Also prints the host wall time of each part (median of 9 rounds).
#[test]
#[ignore]
fn one_draft_round_under_the_profiler() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    let prompt = ids_of(&oracle, "code_700");
    let mut tokens = prompt.clone();
    tokens.push(prefill(&mut m, 0, &prompt));
    let (mut draft_ms, mut window_ms, mut flush_ms) = (Vec::new(), Vec::new(), Vec::new());
    for round in 0..10 {
        let traced = round == 9;
        m.flush().unwrap();
        m.sync().unwrap();
        if traced {
            gb_cuda::rt::profiler_start();
        }
        let t = std::time::Instant::now();
        let drafts = m.dflash_draft(&[DflashAsk { slot: 0, tokens: &tokens, depth: 7, sampling: None }]).unwrap().remove(0);
        let t1 = std::time::Instant::now();
        let mut window = vec![*tokens.last().unwrap()];
        window.extend_from_slice(&drafts);
        let samples = window_argmax(&mut m, 0, &window, tokens.len() - 1);
        let t2 = std::time::Instant::now();
        let acc = drafts.iter().zip(&samples).take_while(|(d, s)| d == s).count();
        m.commit(0, acc + 1);
        m.flush().unwrap();
        m.sync().unwrap();
        let t3 = std::time::Instant::now();
        if traced {
            gb_cuda::rt::profiler_stop();
        }
        tokens.extend_from_slice(&samples[..=acc]);
        if round > 0 {
            draft_ms.push((t1 - t).as_secs_f64() * 1e3);
            window_ms.push((t2 - t1).as_secs_f64() * 1e3);
            flush_ms.push((t3 - t2).as_secs_f64() * 1e3);
        }
    }
    // the verify window's weight-streaming floor: the experts its rows pick (their union per MoE layer, from a captured window) are what it
    // must read, beside the rest of the trunk; one row's floor is its own eight a layer
    let mut window = vec![*tokens.last().unwrap()];
    window.extend(m.dflash_draft(&[DflashAsk { slot: 0, tokens: &tokens, depth: 7, sampling: None }]).unwrap().remove(0));
    m.capture = Some(Capture::default());
    window_argmax(&mut m, 0, &window, tokens.len() - 1);
    let cap = m.capture.take().unwrap();
    m.commit(0, 1);
    let (mut union, mut per_row) = (0usize, 0usize);
    for ids in cap.route_ids.values() {
        let mut u: Vec<i32> = ids.clone();
        u.sort_unstable();
        u.dedup();
        union += u.len();
        per_row += ids.len() / window.len();
    }
    eprintln!(
        "verify window of {} rows: {} MoE layers, {union} distinct experts read in all ({:.1} a layer), one row reads {per_row} ({:.1} a layer)",
        window.len(),
        cap.route_ids.len(),
        union as f64 / cap.route_ids.len() as f64,
        per_row as f64 / cap.route_ids.len() as f64
    );
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    eprintln!(
        "one round, median of 9: draft call {:.2} ms, verify window of 8 rows {:.2} ms, commit + absorb {:.2} ms",
        med(&mut draft_ms),
        med(&mut window_ms),
        med(&mut flush_ms)
    );
    m.end_slot(0);
}

fn env_dir(var: &str) -> Option<PathBuf> {
    match std::env::var(var) {
        Ok(d) if Path::new(&d).join("config.json").exists() => Some(PathBuf::from(d)),
        Ok(d) => panic!("{var}={d} has no config.json"),
        Err(_) => {
            eprintln!("{var} is not set: not run");
            None
        }
    }
}

fn verdict_of(target: &Path, draft: &Path) -> (Result<gb_model::serve::PackVerdict, String>, String) {
    let (t, d) = (gb_model::pack::Pack::open(target).unwrap(), gb_model::pack::Pack::open(draft).unwrap());
    let (tf, df) = (gb_model::fingerprint::fingerprint(&t.ck), gb_model::fingerprint::fingerprint(&d.ck));
    let s = gb_model::draft::structural_check(&d, &t.ck.config);
    (gb_model::serve::pack_verdict(&tf, Some((&df, &s))), df)
}

/// The published GLM draft is its listed pair; a draft quantization the operator made passes by its structure; both pass without the flag
/// (`pack_check_dirs` is the boot's gate).
#[test]
fn the_published_glm_draft_and_a_user_quant_pass_the_pack_gate_without_the_flag() {
    let (Some(t), Some(d)) = (env_dir("REXL3_GLM_DIR"), env_dir("REXL3_GLM_DRAFT_DIR")) else { return };
    let (v, df) = verdict_of(&t, &d);
    let v = v.unwrap();
    assert_eq!(df, gb_model::fingerprint::GLM53_DFLASH2_FP, "the published checkpoint");
    assert_eq!(v.pack, gb_model::fingerprint::VALIDATED_GLM53_DRAFT[0].0);
    let mut o = gb_model::serve::ServeOptions::new(t.clone());
    o.draft_dir = Some(d);
    gb_model::serve::check_pack_dirs(&o).unwrap();
    if let Some(user) = env_dir("REXL3_GLM_DRAFT_USER_DIR") {
        // a quantization is another fingerprint: the target's own entry, the draft by its structure
        let (v, df) = verdict_of(&t, &user);
        let v = v.unwrap();
        let alone = gb_model::fingerprint::validated_pack(gb_model::fingerprint::VALIDATED_GLM53_DRAFT[0].1, None);
        assert_eq!(Some(v.pack), alone);
        assert_eq!(v.draft.as_deref(), Some("structural (glm53-flash)"), "{df}");
        o.draft_dir = Some(user);
        gb_model::serve::check_pack_dirs(&o).unwrap();
        eprintln!("the operator's draft {df}: {v:?}");
    }
}

/// The Qwen3.8 draft offered to the GLM target is refused by name (its geometry is not the GLM one), and the flag still overrides it.
#[test]
fn a_draft_of_another_geometry_is_refused_by_name() {
    let (Some(t), Some(qd)) = (env_dir("REXL3_GLM_DIR"), env_dir("REXL3_DRAFT_DIR")) else { return };
    let (v, df) = verdict_of(&t, &qd);
    let e = v.unwrap_err();
    assert!(e.contains(&df) && e.contains("glm53-flash") && e.contains("hidden_size 5120 (expected 4096)"), "{e}");
    let mut o = gb_model::serve::ServeOptions::new(t);
    o.draft_dir = Some(qd);
    assert!(gb_model::serve::check_pack_dirs(&o).is_err());
    o.allow_unvalidated_pack = true;
    gb_model::serve::check_pack_dirs(&o).unwrap();
}

/// The Qwen3.8 pair is what it was (its name), and its draft passes the structural check too.
#[test]
fn the_qwen_pair_keeps_its_name_and_its_draft_is_structurally_valid() {
    let (Some(t), Some(d)) = (env_dir("REXL3_QWEN_DIR"), env_dir("REXL3_DRAFT_DIR")) else { return };
    let (tp, dp) = (gb_model::pack::Pack::open(&t).unwrap(), gb_model::pack::Pack::open(&d).unwrap());
    assert_eq!(gb_model::draft::structural_check(&dp, &tp.ck.config).unwrap().name, "qwen38");
    let (v, _) = verdict_of(&t, &d);
    assert_eq!(v.unwrap().pack, gb_model::fingerprint::VALIDATED[0].0);
}
