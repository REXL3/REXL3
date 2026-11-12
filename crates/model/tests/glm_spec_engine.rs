//! Speculation through the real engine and backend on the whole GLM-5.3-Flash model with the NextN layer (90.5 GiB resident): `Engine<GlmBackend>`
//! drives requests at temperature 0 and with seeded sampling, concurrent streams sharing windows, some drafting and some not. `Request::draft =
//! false` is the engine's serial reference path, so ONE engine gives both sides of the comparison: drafted == serial, token for token.
//!
//! * `engine_drafted_equals_serial`: 44 cases (11 prompts x 4 sampling settings), each run serially and with the NextN drafter at K = 2 and K = 4
//!   (the copy drafter and the arbiter), streams in groups of four; the committed tokens must be identical.
//! * Requests with images (the drafter's cache over an image prompt, the predicate, a request with bad features failing alone) are `glm_spec_vision.rs`.
//!
//! * `auto_falls_back_when_the_drafter_is_broken`: the drafter's tap is broken on purpose (its drafts stop matching): a fixed-depth MTP run falls below
//!   serial speed, the arbiter's run does not (it stops drafting and probes ever more rarely), and both produce the serial tokens.
//! * `speedup_by_class`: the oracle's six text classes (`tools/glm_greedy_check.py prep`, REXL3_GLM_PROMPT_IDS) at greedy, 384 tokens, serial against the
//!   NextN drafter at K = 1..4, the copy drafter and the arbiter: tokens per round, tokens per second, then 2, 4 and 8 streams at once.
//!
//! `#[ignore]`, env-gated (REXL3_GLM_DIR, REXL3_GLM_ORACLE_DIR for the prompts' token ids), under the GPU lock, one model per process.

mod glm_support;

use gb_engine::{Engine, Output, Request};
use gb_model::families::glm::backend::GlmBackend;
use gb_model::families::glm::engine::engine_over;
use gb_model::families::glm::model::{GlmModel, MtpMutant};
use gb_model::families::glm::npy;
use gb_model::families::glm::spec::{Source, SpecConfig};
use gb_sampling::Sampling;
use gb_sched::{Admission, ModelMemory};
use gb_spec::{DraftPick, SpecMode};
use glm_support::*;
use std::path::PathBuf;

const SLOTS: usize = 8;
const KV_BLOCKS: usize = 96; // 24,576 tokens
const MAX_CONTEXT: usize = 4096;

fn oracle_ids(name: &str) -> Option<Vec<u32>> {
    let dir = match std::env::var("REXL3_GLM_ORACLE_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            eprintln!("REXL3_GLM_ORACLE_DIR is not set: not run");
            return None;
        }
    };
    Some(npy::read(&dir.join(name).join("ids.npy")).unwrap().to_i32().unwrap().iter().map(|&x| x as u32).collect())
}

type Eng = Engine<GlmBackend>;

fn engine(mut m: GlmModel, mode: SpecMode) -> Eng {
    // admission over tokens: one byte a token and a byte a slot, so the capacity is the KV pool's
    let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 1, activation_reserve: 0 };
    let admission = Admission::new((KV_BLOCKS * 256 + SLOTS) as u64, 0, mm, SLOTS).unwrap().with_max_context(MAX_CONTEXT as u64);
    let mut cfg = SpecConfig::new(mode);
    // REXL3_GLM_DRAFT_NOISE=0|1: the head's pick for the sampled requests of the run (`engine_drafted_equals_serial` sets it per configuration)
    cfg.noise =
        DraftPick::from_setting(std::env::var("REXL3_GLM_DRAFT_NOISE").ok().as_deref(), "REXL3_GLM_DRAFT_NOISE", cfg.noise)
            .unwrap();
    // REXL3_SPEED_VOCAB=512: the serving default for the drafter's head (the restricted vocabulary of that many prefix strips, adaptive at 90%), priced by
    // the arbiter as such; unset: the full head
    if let Ok(v) = std::env::var("REXL3_SPEED_VOCAB") {
        use gb_model::families::glm::model::{vocab_ranges, VocabMode};
        m.mtp_restrict_vocab(&vocab_ranges(v.parse().unwrap(), 154_880)).unwrap();
        m.set_mtp_vocab(VocabMode::Adaptive(0.9));
        cfg.restricted_head = true;
    }
    engine_over(GlmBackend::with_spec(m, cfg), admission, true)
}

fn request(prompt: &[u32], sampling: Sampling, max_new: usize, draft: bool) -> Request {
    Request { prompt: prompt.to_vec(), sampling, max_new, stop: vec![], draft, images: vec![], cache: true }
}

/// Run `reqs` together to the end; the outputs in submission order.
fn run(e: &mut Eng, reqs: Vec<Request>) -> Vec<Output> {
    let n = reqs.len();
    for r in reqs {
        e.submit_deferred(r).unwrap();
    }
    while e.busy() {
        e.step().unwrap();
    }
    assert!(e.take_failed().is_empty(), "no request may fail");
    let outs = e.take_outputs();
    assert_eq!(outs.len(), n);
    outs
}

fn samplings() -> Vec<(&'static str, Sampling)> {
    vec![
        ("greedy", Sampling { temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0, seed: 0 }),
        ("T0.7 seed 1", Sampling { temperature: 0.7, top_k: 40, top_p: 0.95, min_p: 0.0, seed: 1 }),
        ("T1.0 seed 2", Sampling { temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0, seed: 2 }),
        ("T0.3 seed 3 min_p", Sampling { temperature: 0.3, top_k: 8, top_p: 1.0, min_p: 0.05, seed: 3 }),
    ]
}

#[test]
#[ignore]
fn engine_drafted_equals_serial() {
    let (Some(chat), Some(code), Some(short), Some(think), Some(long)) = (
        oracle_ids("chat_300"),
        oracle_ids("code_700"),
        oracle_ids("short_20"),
        oracle_ids("think_prefix"),
        oracle_ids("long_2300"),
    ) else {
        return;
    };
    let Some(model) = load_model_mtp(SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    let mut e = engine(model, SpecMode::Mtp);
    let prompts: Vec<(&str, Vec<u32>)> = vec![
        ("short_20", short),
        ("think_prefix", think),
        ("chat_300[..100]", chat[..100].to_vec()),
        ("chat_300", chat),
        ("code_700[..200]", code[..200].to_vec()),
        ("code_700[..500]", code[..500].to_vec()),
        ("code_700", code),
        ("long_2300[..1500]", long[..1500].to_vec()),
        ("long_2300[..2040]", long[..2040].to_vec()),
        ("long_2300[..2200]", long[..2200].to_vec()),
        ("long_2300 (2,278)", long),
    ];
    let max_new = 40;
    let mut cases: Vec<(String, Vec<u32>, Sampling)> = Vec::new();
    for (pn, p) in &prompts {
        for (sn, s) in samplings() {
            cases.push((format!("{pn} / {sn}"), p.clone(), s));
        }
    }
    assert_eq!(cases.len(), 44);
    let mut compared = 0;
    // (verified, accepted) drafts of the head by pick: [argmax, keyed]
    let mut by_noise = [(0u64, 0u64); 2];
    for group in cases.chunks(SLOTS) {
        let serial: Vec<Output> = run(&mut e, group.iter().map(|(_, p, s)| request(p, *s, max_new, false)).collect());
        assert!(serial.iter().all(|o| o.drafted == 0), "draft=false never drafts");
        // the head proposes by the keyed draw over its candidates (`REXL3_GLM_DRAFT_NOISE=1`) or by its argmax (0): both must give the serial text
        for (label, mode, depth, noise) in [
            ("MTP K=2 keyed", SpecMode::Mtp, 2usize, DraftPick::Keyed),
            ("MTP K=4 keyed", SpecMode::Mtp, 4, DraftPick::Keyed),
            ("MTP K=2 argmax", SpecMode::Mtp, 2, DraftPick::Argmax),
            ("MTP K=4 argmax", SpecMode::Mtp, 4, DraftPick::Argmax),
            ("copy", SpecMode::Copy, 2, DraftPick::Keyed),
            ("auto keyed", SpecMode::Auto, 2, DraftPick::Keyed),
            ("auto argmax", SpecMode::Auto, 2, DraftPick::Argmax),
        ] {
            e.backend.spec.as_mut().unwrap().cfg.mode = mode;
            e.backend.spec.as_mut().unwrap().cfg.mtp_depth = depth;
            e.backend.spec.as_mut().unwrap().cfg.noise = noise;
            let before = {
                let st = &e.backend.spec.as_ref().unwrap().stats;
                (st.verified[Source::Mtp as usize], st.accepted[Source::Mtp as usize])
            };
            let drafted: Vec<Output> = run(&mut e, group.iter().map(|(_, p, s)| request(p, *s, max_new, true)).collect());
            {
                let st = &e.backend.spec.as_ref().unwrap().stats;
                let k = usize::from(noise.is_keyed());
                by_noise[k].0 += st.verified[Source::Mtp as usize] - before.0;
                by_noise[k].1 += st.accepted[Source::Mtp as usize] - before.1;
            }
            for ((name, _, _), (a, b)) in group.iter().zip(serial.iter().zip(&drafted)) {
                assert_eq!(a.tokens, b.tokens, "{name}: {label} differs from the serial stream");
                compared += 1;
                // a fixed-depth head drafts EVERY round (a round drafted nothing only at the very end): a drafter that quietly stops after the first
                // round still gives the serial tokens, so exactness alone would not notice
                if mode == SpecMode::Mtp {
                    assert!(
                        b.drafted + 12 >= b.rounds * depth as u64,
                        "{name}: {label}: {} drafts verified in {} rounds: the head stopped drafting",
                        b.drafted,
                        b.rounds
                    );
                }
            }
        }
    }
    let st = &e.backend.spec.as_ref().unwrap().stats;
    eprintln!(
        "{compared} (case, drafter) comparisons over {} cases: all token streams equal serial. MTP rounds {} ({:.2} tokens a round, {} drafts verified, {} accepted); copy rounds {} ({} verified, {} accepted); no-draft rounds {}",
        cases.len(),
        st.rounds[Source::Mtp as usize],
        st.tokens_per_round(Source::Mtp),
        st.verified[Source::Mtp as usize],
        st.accepted[Source::Mtp as usize],
        st.rounds[Source::Copy as usize],
        st.verified[Source::Copy as usize],
        st.accepted[Source::Copy as usize],
        st.rounds[Source::None as usize]
    );
    assert!(st.accepted[Source::Mtp as usize] > 0, "the NextN drafter was never right: it did not run");
    eprintln!(
        "head drafts verified / accepted: argmax {} / {} ({:.3}), keyed {} / {} ({:.3}) (all four settings pooled: greedy cases are equal by construction)",
        by_noise[0].0,
        by_noise[0].1,
        by_noise[0].1 as f64 / by_noise[0].0.max(1) as f64,
        by_noise[1].0,
        by_noise[1].1,
        by_noise[1].1 as f64 / by_noise[1].0.max(1) as f64
    );
    assert!(by_noise[0].0 > 0 && by_noise[1].0 > 0, "both picks must have drafted");
}

// ------------------------------------------------------------------------------------------------------------------ speed by class

fn class_ids() -> Option<Vec<(String, Vec<u32>)>> {
    let path = match std::env::var("REXL3_GLM_PROMPT_IDS") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("REXL3_GLM_PROMPT_IDS is not set (tools/glm_greedy_check.py prep): not run");
            return None;
        }
    };
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut out: Vec<(String, Vec<u32>)> = v
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, ids)| (k.clone(), ids.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()))
        .collect();
    out.sort_by_key(|(k, _)| ["code", "chat", "structured", "math", "repeat", "long"].iter().position(|n| n == k).unwrap_or(9));
    Some(out)
}

struct Timed {
    tokens: Vec<u32>,
    prefill_s: f64,
    decode_s: f64,
    outs: Vec<Output>,
}

/// Run `reqs` together; the time to the first event (the prefills) and from there to the end (the decoding of all of them).
fn run_timed(e: &mut Eng, reqs: Vec<Request>) -> Timed {
    let t0 = std::time::Instant::now();
    for r in reqs {
        e.submit_deferred(r).unwrap();
    }
    let mut first: Option<f64> = None;
    let mut seen = 0usize;
    let mut all_prefilled = None;
    while e.busy() {
        e.step().unwrap();
        let ev = e.take_events();
        seen += ev.iter().map(|x| x.tokens.len()).sum::<usize>();
        if first.is_none() && !ev.is_empty() {
            first = Some(t0.elapsed().as_secs_f64());
        }
        // decoding starts when no prompt is waiting any more
        if all_prefilled.is_none() && e.prefilling() == 0 {
            all_prefilled = Some(t0.elapsed().as_secs_f64());
        }
    }
    let total = t0.elapsed().as_secs_f64();
    let outs = e.take_outputs();
    assert!(seen > 0);
    let decode_start = all_prefilled.or(first).unwrap_or(0.0);
    Timed {
        tokens: outs.iter().flat_map(|o| o.tokens.clone()).collect(),
        prefill_s: decode_start,
        decode_s: total - decode_start,
        outs,
    }
}

fn set_mode(e: &mut Eng, mode: SpecMode, depth: usize) {
    let s = e.backend.spec.as_mut().unwrap();
    s.cfg.mode = mode;
    s.cfg.mtp_depth = depth;
    // the arbiter's own acceptance estimates and cost scales start fresh for every configuration
    *s = gb_model::families::glm::spec::SpecRuntime::new(s.cfg, SLOTS);
}

#[test]
#[ignore]
fn speedup_by_class() {
    let Some(classes) = class_ids() else { return };
    let Some(model) = load_model_mtp(SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    let eos: Vec<u32> = model.cfg.eos.clone();
    let mut e = engine(model, SpecMode::Auto);
    let greedy = samplings()[0].1;
    let req = |p: &[u32], draft: bool| Request {
        prompt: p.to_vec(),
        sampling: greedy,
        max_new: 384,
        stop: eos.clone(),
        draft,
        images: vec![],
        cache: true,
    };
    // (label, mode, depth, draft flag)
    let configs: Vec<(&str, SpecMode, usize, bool)> = vec![
        ("AR", SpecMode::None, 2, false),
        ("MTP K=1", SpecMode::Mtp, 1, true),
        ("MTP K=2", SpecMode::Mtp, 2, true),
        ("MTP K=3", SpecMode::Mtp, 3, true),
        ("MTP K=4", SpecMode::Mtp, 4, true),
        ("copy", SpecMode::Copy, 2, true),
        ("auto", SpecMode::Auto, 2, true),
    ];
    // REXL3_SPEED_CONFIGS="MTP K=2,auto" keeps only those single-stream configurations beside AR (the reference) and REXL3_SPEED_STREAMS="" or "4,8" picks the
    // multi-stream phases: a re-measurement of one change takes minutes instead of 13
    let only: Option<Vec<String>> =
        std::env::var("REXL3_SPEED_CONFIGS").ok().map(|v| v.split(',').map(|x| x.trim().to_string()).collect());
    let stream_counts: Vec<usize> = match std::env::var("REXL3_SPEED_STREAMS") {
        Ok(v) => v.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        Err(_) => vec![2, 4, 8],
    };
    // warm up (kernels, allocator) with a short run
    set_mode(&mut e, SpecMode::Mtp, 2);
    run_timed(&mut e, vec![Request { max_new: 24, ..req(&classes[0].1, true) }]);
    for (name, prompt) in &classes {
        let mut serial_tokens: Option<Vec<u32>> = None;
        for &(label, mode, depth, draft) in &configs {
            if label != "AR" && only.as_ref().is_some_and(|o| !o.iter().any(|x| x == label)) {
                continue;
            }
            set_mode(&mut e, mode, depth);
            let t = run_timed(&mut e, vec![req(prompt, draft)]);
            let o = &t.outs[0];
            if let Some(s) = &serial_tokens {
                assert_eq!(&t.tokens, s, "{name} / {label}: the committed tokens differ from the serial run");
            } else {
                serial_tokens = Some(t.tokens.clone());
            }
            if mode == SpecMode::Mtp {
                assert!(
                    o.drafted + 12 >= o.rounds * depth as u64,
                    "{name} / {label}: {} drafts verified in {} rounds: the head stopped drafting",
                    o.drafted,
                    o.rounds
                );
            }
            let n = t.tokens.len();
            println!(
                "RUN single {name:10} {label:8} prompt {:4} tokens {n:4} rounds {:4} tokens_per_round {:.3} drafted {:4} accepted {:4} prefill {:.2} s decode {:.2} s: {:.2} tok/s decode, {:.2} tok/s with prefill",
                prompt.len(),
                o.rounds,
                o.tokens_per_round(),
                o.drafted,
                o.accepted,
                t.prefill_s,
                t.decode_s,
                (n as f64 - 1.0) / t.decode_s,
                n as f64 / (t.prefill_s + t.decode_s)
            );
            if mode == SpecMode::Auto {
                let st = &e.backend.spec.as_ref().unwrap().stats;
                println!(
                    "RUN auto-picks {name:10} none {} copy {} mtp {} (tokens per round: mtp {:.2} copy {:.2}); mtp rounds by depth {:?}; arbiter probes {} forced absorbs {}",
                    st.rounds[0],
                    st.rounds[1],
                    st.rounds[2],
                    st.tokens_per_round(Source::Mtp),
                    st.tokens_per_round(Source::Copy),
                    &st.by_depth[Source::Mtp as usize][1..=4],
                    e.backend.spec.as_ref().unwrap().arbiter.stats.probes,
                    e.backend.spec.as_ref().unwrap().arbiter.stats.forced_absorbs
                );
            }
        }
    }
    // several streams at once: the prompts of the classes cycled, aggregate decode tokens per second
    for &streams in &stream_counts {
        let mut base: Option<Vec<Vec<u32>>> = None;
        for &(label, mode, depth, draft) in &configs {
            if !matches!(label, "AR" | "MTP K=2" | "MTP K=3" | "auto") {
                continue;
            }
            set_mode(&mut e, mode, depth);
            let reqs: Vec<Request> = (0..streams).map(|i| req(&classes[i % classes.len()].1, draft)).collect();
            let t = run_timed(&mut e, reqs);
            let per: Vec<Vec<u32>> = t.outs.iter().map(|o| o.tokens.clone()).collect();
            if let Some(b) = &base {
                assert_eq!(&per, b, "{streams} streams / {label}: the committed tokens differ from the serial run");
            } else {
                base = Some(per);
            }
            let n: usize = t.outs.iter().map(|o| o.tokens.len() - 1).sum();
            let rounds: u64 = t.outs.iter().map(|o| o.rounds).sum();
            println!(
                "RUN streams {streams} {label:8} tokens {n:5} in {:.2} s decode: {:.2} tok/s aggregate ({:.2} per stream), tokens per round {:.3}, prefill phase {:.2} s",
                t.decode_s,
                n as f64 / t.decode_s,
                n as f64 / t.decode_s / streams as f64,
                if rounds == 0 { 0.0 } else { n as f64 / rounds as f64 },
                t.prefill_s
            );
        }
    }
}

#[test]
#[ignore]
fn auto_falls_back_when_the_drafter_is_broken() {
    let Some(classes) = class_ids() else { return };
    let Some(model) = load_model_mtp(SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    let eos: Vec<u32> = model.cfg.eos.clone();
    let mut e = engine(model, SpecMode::Auto);
    let greedy = samplings()[0].1;
    let prompt = &classes.iter().find(|(n, _)| n == "code").unwrap().1;
    let req = |draft: bool| Request {
        prompt: prompt.to_vec(),
        sampling: greedy,
        max_new: 256,
        stop: eos.clone(),
        draft,
        images: vec![],
        cache: true,
    };
    set_mode(&mut e, SpecMode::None, 2);
    let serial = run_timed(&mut e, vec![req(false)]);
    let ar = (serial.tokens.len() as f64 - 1.0) / serial.decode_s;
    // the broken drafter: every draft is wrong (numerically clean, acceptance about zero)
    e.backend.m.set_mtp_mutant(Some(MtpMutant::WrongDrafts));
    set_mode(&mut e, SpecMode::Mtp, 2);
    let fixed = run_timed(&mut e, vec![req(true)]);
    assert_eq!(fixed.tokens, serial.tokens, "a broken drafter still gives the serial tokens");
    let fixed_rate = (fixed.tokens.len() as f64 - 1.0) / fixed.decode_s;
    set_mode(&mut e, SpecMode::Auto, 2);
    let auto = run_timed(&mut e, vec![req(true)]);
    assert_eq!(auto.tokens, serial.tokens, "and so does the arbiter");
    let auto_rate = (auto.tokens.len() as f64 - 1.0) / auto.decode_s;
    let st = &e.backend.spec.as_ref().unwrap().stats;
    let arb = e.backend.spec.as_ref().unwrap().arbiter.stats;
    println!(
        "BROKEN serial {ar:.2} tok/s; fixed MTP K=2 on a broken drafter {fixed_rate:.2} tok/s ({:.3} tokens per round); arbiter {auto_rate:.2} tok/s (rounds none {} mtp {}, probes {})",
        fixed.outs[0].tokens_per_round(),
        st.rounds[0],
        st.rounds[2],
        arb.probes
    );
    e.backend.m.set_mtp_mutant(None);
    assert!(
        fixed_rate < ar * 0.92,
        "the broken fixed-depth run should be visibly slower than serial: {fixed_rate:.2} vs {ar:.2}"
    );
    assert!(auto_rate > ar * 0.95, "the arbiter must not fall below serial on a broken drafter: {auto_rate:.2} vs {ar:.2} tok/s");
}

/// The drafter's head restricted to a set of token-id strips (a draft only proposes, so exactness cannot move; the speed and the acceptance can): per
/// class of the oracle's prompts at greedy, 384 tokens, the NextN drafter at K = 2 with the full head and with sets of the first P strips of 128 ids
/// (the tokenizer's merge order puts the frequent tokens first) plus the numeric block and the special tokens. The committed tokens must equal the
/// serial run's whatever the set; printed: tokens per round, decode tokens per second, and the share of the generated tokens the set covers.
#[test]
#[ignore]
fn vocab_by_class() {
    use gb_model::families::glm::model::VocabMode;
    let Some(classes) = class_ids() else { return };
    let Some(model) = load_model_mtp(SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    let eos: Vec<u32> = model.cfg.eos.clone();
    let mut e = engine(model, SpecMode::Mtp);
    let greedy = samplings()[0].1;
    let req = |p: &[u32], draft: bool| Request {
        prompt: p.to_vec(),
        sampling: greedy,
        max_new: 384,
        stop: eos.clone(),
        draft,
        images: vec![],
        cache: true,
    };
    // REXL3_VOCAB_CONFIGS="full,384,512,512a0.9": the full head, a prefix of that many strips (restricted every call), or with `a<share>` the adaptive
    // mode (the full head in a call where an asking stream has less than that share of its last 32 tokens inside the set)
    /// (label, None = the full head | Some((prefix strips, Some(adaptive share) | None = restricted every call)))
    type Config = (String, Option<(usize, Option<f32>)>);
    let configs: Vec<Config> = std::env::var("REXL3_VOCAB_CONFIGS")
        .unwrap_or_else(|_| "full,192,256,384,512".into())
        .split(',')
        .map(|c| {
            let c = c.trim();
            if c == "full" {
                (c.to_string(), None)
            } else if let Some((p, min)) = c.split_once('a') {
                (c.to_string(), Some((p.parse().unwrap(), Some(min.parse().unwrap()))))
            } else {
                (c.to_string(), Some((c.parse().unwrap(), None)))
            }
        })
        .collect();
    set_mode(&mut e, SpecMode::Mtp, 2);
    run_timed(&mut e, vec![Request { max_new: 24, ..req(&classes[0].1, true) }]);
    for (name, prompt) in &classes {
        set_mode(&mut e, SpecMode::None, 2);
        let serial = run_timed(&mut e, vec![req(prompt, false)]).tokens;
        for (label, set) in &configs {
            let strips = match set {
                Some((p, min)) => {
                    let ranges = gb_model::families::glm::model::vocab_ranges(*p, 154_880);
                    let bytes = e.backend.m.mtp_restrict_vocab(&ranges).unwrap();
                    e.backend.m.set_mtp_vocab(match min {
                        Some(m) => VocabMode::Adaptive(*m),
                        None => VocabMode::Restricted,
                    });
                    println!("VOCAB set {label}: {:?}, {:.1} MiB added", ranges, bytes as f64 / 1048576.0);
                    ranges.iter().map(|&(lo, hi)| hi - lo).sum::<usize>()
                }
                None => {
                    e.backend.m.set_mtp_vocab(VocabMode::Full);
                    1210
                }
            };
            set_mode(&mut e, SpecMode::Mtp, 2);
            let t = run_timed(&mut e, vec![req(prompt, true)]);
            assert_eq!(t.tokens, serial, "{name} / {label}: the committed tokens differ from the serial run");
            let o = &t.outs[0];
            let cover = e.backend.m.mtp_sub_head().map_or(1.0, |s| s.coverage(&t.tokens));
            println!(
                "VOCAB {name:10} {label:9} strips {strips:4} rounds {:4} tokens_per_round {:.3} drafted {:4} accepted {:4} coverage {:.3}: {:.2} tok/s decode",
                o.rounds,
                o.tokens_per_round(),
                o.drafted,
                o.accepted,
                cover,
                (t.tokens.len() as f64 - 1.0) / t.decode_s
            );
        }
    }
}
