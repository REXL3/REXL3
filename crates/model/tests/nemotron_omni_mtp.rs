//! GPU tests of the Nemotron 3.5 Super VL MTP head on the whole 88-layer model (about 70 GiB resident with the head). All `#[ignore]` and
//! env-gated (`REXL3_NEMO_OMNI_DIR`, `REXL3_NEMO_ORACLE_DIR`); without them a test prints "not run" and passes. One model per process: run
//! with `--test-threads=1 --ignored` under the GPU lock (`~/rexl3-ab/agents-tools/nrun1 NAME --gpu -- cargo test --release ...`).
//!
//! * `the_head_is_loaded_with_the_trunk_and_its_state_is_where_the_plan_says`: the loader, the head's KV layer, the tap stores, the book.
//! * `the_head_on_the_device_equals_mtp_ref_on_the_real_weights`: the gate against the CPU reference run on the real head weights.
//! * `drafter_in_the_loop_equals_serial` (tokens, candidates and trunk state bit for bit, and the head's cache equal to teacher forcing, for
//!   K = 1..4), `the_tap_buffer_is_the_final_norm_output`, `commit_moves_exactly_keep_rows`, `an_unsound_slot_drafts_nothing_and_nothing_breaks`,
//!   `every_deliberate_head_bug_is_noticed`: the exactness tests and the mutants.
//! * `keyed_drafts_are_the_hosts_pick_over_the_heads_candidates`, `a_drafts_bits_do_not_depend_on_the_other_asks_of_its_call`,
//!   `keyed_and_greedy_drafts_agree_between_calls_and_with_the_hosts_chain`, `the_draft_call_is_deterministic_under_repetition`,
//!   `the_restricted_draft_head_equals_the_full_head_inside_the_set`: keyed drafting, batching, determinism and the restricted head.
//! * `acceptance_by_depth_and_vocabulary`, `draft_call_costs`, `the_heads_cost_in_prefill_and_in_a_window_commit`: the numbers.

mod nemotron_omni_mtp_support;

use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::mtp::{DraftAsk, MTP_TAP_CAP};
use gb_model::families::nemotron_h_omni::weights::{LayerPlan, UnitId};
use nemotron_omni_mtp_support::*;

fn mib(b: u64) -> f64 {
    b as f64 / 1048576.0
}

#[test]
#[ignore]
fn the_head_is_loaded_with_the_trunk_and_its_state_is_where_the_plan_says() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    assert!(m.has_mtp() && m.mtp_loaded(), "the full trunk loads the head unless REXL3_MTP=off");

    // the head's weights are the pack's tensors in place: the bytes the plan lists for its two blocks
    let dir = pack_dir().unwrap();
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap();
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let planned: u64 = plan.tensors(|u| matches!(u, UnitId::Mtp(_))).iter().map(|w| inv.get(&w.name).unwrap().nbytes).sum();
    let alloc = m.mtp_allocations();
    let get = |n: &str| alloc.iter().find(|(k, _)| *k == n).map(|(_, b)| *b).unwrap_or_else(|| panic!("no allocation named {n}"));
    for (n, b) in &alloc {
        eprintln!("mtp allocation {n}: {:.2} MiB", mib(*b));
    }
    // a 4-bit head's EXL3 projections hold their trellis in place and drop the 4-byte codebook marker; the MoE block's take their scales
    // widened to f32 (twice the pack's fp16), `eh_proj` keeps them fp16. A bf16 head has no linears: the pack's bytes exactly.
    let mut held = planned;
    for u in [UnitId::Mtp(0), UnitId::Mtp(1)] {
        for l in &plan.unit(u).unwrap().linears {
            held -= 4;
            if !l.prefix.ends_with(".eh_proj") {
                held += 2 * (l.k + l.n) as u64;
            }
        }
    }
    eprintln!("head projections stored as EXL3: {}", plan.head_exl3().len());
    assert_eq!(get("mtp_weights"), held, "the head's device bytes are the pack's head tensors");
    assert_eq!(get("mtp_kv_per_token"), 1024, "the head's KV layer is 1 KiB per token");
    assert_eq!(get("mtp_taps_store"), (SLOTS * MTP_TAP_CAP * 4096 * 4) as u64);

    // the head's KV layer is one more attention layer ordinal of the same pool: 8 trunk layers + 1
    let g = m.core.kv_geom;
    assert_eq!(g.layers, 9);
    assert_eq!(g.token_bytes(), 9 * 1024);
    assert_eq!(m.core.kv.k.bytes(), g.half_bytes());

    // a prompt through two chunks of the grid (64 rows): the head's cache is built chunk by chunk, the last tap is the carry
    let prompt = &prefix[..100];
    let first = prefill(&mut m, 0, prompt, 64);
    let b = m.mtp_book(0).unwrap().clone();
    assert!(b.ok && (b.first, b.n) == (100, 1), "after a 100-token prompt the book holds the carry: {b:?}");
    assert_eq!(m.mtp_pending_rows(0), Some(1));
    // the head's cache holds entries 0..=98 (positions of taps 0..98); position 99's entry waits for the first sampled token
    let (k, v) = m.mtp_kv_rows(0, 0, 99).unwrap();
    assert!(k.iter().chain(&v).any(|&x| x != 0), "the head's cache was written");
    let nz_rows = (0..99).filter(|&r| k[r * 256..(r + 1) * 256].iter().any(|&x| x != 0)).count();
    assert_eq!(nz_rows, 99, "every position below the carry has an entry");

    // a draft call: absorbs the carry (one head row) and chains three
    let mut tokens = prompt.to_vec();
    tokens.push(first);
    let t = std::time::Instant::now();
    let d = m.mtp_draft(&[DraftAsk { slot: 0, tokens: &tokens, depth: 3 }], &[]).unwrap().remove(0).expect("the slot drafts");
    eprintln!("first draft call: {:.1} ms (cold), ids {:?}, vals {:?}", t.elapsed().as_secs_f64() * 1e3, d.ids, d.vals);
    assert_eq!(d.ids.len(), 3);
    assert!(d.ids.iter().all(|&i| (i as usize) < 131072) && d.vals.iter().all(|v| v.is_finite()));
    let b = m.mtp_book(0).unwrap();
    assert_eq!((b.first, b.n), (101, 0), "absorbed: entries below 101 are built");
    // the same call again: nothing is pending
    assert!(m.mtp_draft(&[DraftAsk { slot: 0, tokens: &tokens, depth: 3 }], &[]).unwrap()[0].is_none());
    // depth 0 only absorbs: after a window and a commit of 2 rows the store holds 2 rows and a depth-0 ask empties it
    let w = window_argmax(&mut m, 0, &[first, d.ids[0]], 100);
    m.commit(0, 2);
    let mut t2 = tokens.clone();
    t2.extend([w[0], w[1]]);
    assert_eq!(m.mtp_pending_rows(0), Some(2));
    let z =
        m.mtp_draft(&[DraftAsk { slot: 0, tokens: &t2, depth: 0 }], &[]).unwrap().remove(0).expect("a depth-0 ask is answered");
    assert!(z.ids.is_empty());
    assert_eq!(m.mtp_pending_rows(0), Some(0));
    m.end_slot(0);
}

/// The load-time switches, through `NemotronModel::load`: with `REXL3_MTP=off` the model has no head (no weights, no KV stripe, every ask
/// answered `None`); otherwise the head is loaded, over the restricted vocabulary (`REXL3_MTP_VOCAB` unset or `default`: the 49,152-id
/// list shipped with the family; `<file>`: that list) or, with `REXL3_MTP_VOCAB=off`, over the full `lm_head`. The shared `model()` of
/// this suite runs the full head when the variable is unset (the other tests compare it with the full-vocabulary reference), so the default
/// is asked for as `default` here. Run it once per setting.
#[test]
#[ignore]
fn the_load_time_switches_do_what_they_say() {
    use gb_model::families::nemotron_h_omni::engine::sub_head_bytes;
    use gb_model::families::nemotron_h_omni::mtp::{restricted_vocab_ids_for, VocabSetting};
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    let off = matches!(std::env::var("REXL3_MTP").as_deref(), Ok("off"));
    let setting = VocabSetting::from_env();
    eprintln!("REXL3_MTP=off: {off}; REXL3_MTP_VOCAB: {setting:?}");
    if off {
        assert!(!m.has_mtp() && !m.mtp_loaded());
        assert!(m.mtp_allocations().is_empty());
        assert_eq!(m.core.kv_geom.layers, 8, "no KV stripe for a head that is not loaded");
        assert_eq!(m.mtp_pending_rows(0), None);
        let first = prefill_ex(&mut m, 0, &prefix[..40], 64, true, None).0;
        let mut t = prefix[..40].to_vec();
        t.push(first);
        use gb_model::families::nemotron_h_omni::backend::NemotronRuntime;
        assert_eq!(NemotronRuntime::mtp_draft(&mut **m, &[DraftAsk { slot: 0, tokens: &t, depth: 3 }], &[]).unwrap(), vec![None]);
        m.end_slot(0);
        return;
    }
    assert!(m.has_mtp(), "the head loads with the whole trunk");
    assert_eq!(m.core.kv_geom.layers, 9);
    match restricted_vocab_ids_for(&setting).unwrap() {
        Some(ids) => {
            assert!(m.mtp_restricted(), "the restricted head is built at load");
            assert_eq!(m.mtp_sub_head().unwrap().len(), ids.len());
            // what the plan priced (`engine::sub_head_bytes`) is what the load allocated
            assert_eq!(m.mtp_sub_head().unwrap().bytes(), sub_head_bytes(ids.len()));
            if matches!(setting, VocabSetting::Default) {
                assert_eq!(ids.len(), 49_152);
                assert_eq!(m.mtp_sub_head().unwrap().bytes(), 402_849_792, "384 MiB and the id map");
            }
            let first = prefill_ex(&mut m, 0, &prefix[..40], 64, true, None).0;
            let mut t = prefix[..40].to_vec();
            t.push(first);
            let d = m.mtp_draft(&[DraftAsk { slot: 0, tokens: &t, depth: 2 }], &[]).unwrap().remove(0).unwrap();
            let sub = m.mtp_sub_head().unwrap();
            assert!(sub.coverage(&d.ids) == 1.0, "a restricted draft is a token of the set");
            m.end_slot(0);
        }
        None => assert!(!m.mtp_restricted() && m.mtp_sub_head().is_none(), "REXL3_MTP_VOCAB=off keeps the full head"),
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Step 3: the head on the device against `mtp_ref` run on the REAL head weights on the CPU.

use gb_model::families::nemotron_h_omni::mtp_ref::{forward_taps, MtpKv};
use nemotron_omni_mtp_support::hostref::*;

/// Rows of the real trunk run the gate feeds the head with (a window from position 0: one more than the head's rows, which pair each hidden
/// row with the token after it).
const GATE_ROWS: usize = 17;

#[test]
#[ignore]
fn the_head_on_the_device_equals_mtp_ref_on_the_real_weights() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    assert!(m.has_mtp());
    let dir = pack_dir().unwrap();
    let cfg = config(&dir);
    let pack = Pack::open(&dir);

    // a real trunk run: a window of GATE_ROWS rows of the oracle's text from position 0; its post-norm_f rows are the taps
    let toks = &prefix[..GATE_ROWS];
    m.end_slot(1);
    m.begin_slot(1).unwrap();
    window_argmax(&mut m, 1, toks, 0);
    let taps = m.mtp_tap_rows(GATE_ROWS).unwrap();
    assert!(taps.iter().all(|x| x.is_finite()));
    let n = GATE_ROWS - 1;
    // head entry i pairs the tap of position i with the token at i + 1
    let hidden = &taps[..n * 4096];
    let next: Vec<u32> = toks[1..].to_vec();

    // the device: the production layer code over those rows, every stage of every row captured, with the head's logits
    m.end_slot(2);
    m.begin_slot(2).unwrap();
    let cap = m.mtp_debug_forward(2, 0, &next, hidden, n).unwrap();
    assert_eq!((cap.fused.len(), cap.block.len(), cap.out.len(), cap.logits.len()), (n * 4096, n * 4096, n * 4096, n * 131072));

    // the reference: the pack's head in f32 on the CPU, all 512 experts, the same rows
    let t = std::time::Instant::now();
    let w = pack.head_weights(&cfg);
    eprintln!("host head weights loaded in {:.1} s", t.elapsed().as_secs_f64());
    let embed = pack.bf16_rows("language_model.backbone.embeddings.weight", &next, 4096);
    let positions: Vec<usize> = (0..n).collect();
    let mut kv = MtpKv::new(&w.geom);
    let t = std::time::Instant::now();
    let r = forward_taps(&w, &embed, hidden, &positions, &mut kv);
    eprintln!("reference head forward over {n} rows: {:.1} s", t.elapsed().as_secs_f64());
    drop(w);
    let t = std::time::Instant::now();
    let ref_logits = pack.lm_head_logits(&r.out, n, 4096, 131072);
    eprintln!("reference lm_head over {n} rows: {:.1} s", t.elapsed().as_secs_f64());

    // the bf16 noise floor of the logits: what rounding the head's output to the bf16 grid (a bf16 model's last activation) does to them
    let out_bf16: Vec<f32> = r.out.iter().map(|&x| round_bf16(x)).collect();
    let noise_logits = pack.lm_head_logits(&out_bf16, n, 4096, 131072);

    let mut worst = [0f64; 4];
    let (mut max_logit_diff, mut max_noise, mut top1_equal, mut noise_top1) = (0f32, 0f32, 0usize, 0usize);
    let mut logit_rms = 0f64;
    for i in 0..n {
        let s = i * 4096..(i + 1) * 4096;
        let errs = [
            rel_l2(&cap.fused[s.clone()], &r.fused[s.clone()]),
            rel_l2(&cap.after_attn[s.clone()], &r.after_attn[s.clone()]),
            rel_l2(&cap.block[s.clone()], &r.after_moe[s.clone()]),
            rel_l2(&cap.out[s.clone()], &r.out[s.clone()]),
        ];
        for (w, e) in worst.iter_mut().zip(errs) {
            *w = w.max(e);
        }
        let l = i * 131072..(i + 1) * 131072;
        let d = max_abs_diff(&cap.logits[l.clone()], &ref_logits[l.clone()]);
        max_logit_diff = max_logit_diff.max(d);
        max_noise = max_noise.max(max_abs_diff(&noise_logits[l.clone()], &ref_logits[l.clone()]));
        let (g, c) = (argmax(&cap.logits[l.clone()]), argmax(&ref_logits[l.clone()]));
        top1_equal += usize::from(g == c);
        noise_top1 += usize::from(argmax(&noise_logits[l.clone()]) == c);
        logit_rms += ref_logits[l.clone()].iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / 131072.0;
        eprintln!(
            "row {i:2}: rel_l2 fused {:.2e} attn {:.2e} block {:.2e} out {:.2e}; logits max-abs {d:.3e}; top-1 device {g} reference {c}",
            errs[0], errs[1], errs[2], errs[3]
        );
    }
    logit_rms = (logit_rms / n as f64).sqrt();
    eprintln!(
        "worst rel_l2: fused {:.2e} after_attn {:.2e} block {:.2e} out {:.2e}; logits max-abs {max_logit_diff:.3e} (logit rms {logit_rms:.2}); \
         bf16 noise floor of the logits {max_noise:.3e} ({noise_top1} of {n} top-1 kept by it); device top-1 equals the reference on {top1_equal} of {n} rows",
        worst[0], worst[1], worst[2], worst[3]
    );
    assert!(worst[2] <= 2e-3, "the block output differs from the reference by {:.3e} (rel_l2), above 2e-3", worst[2]);
    assert_eq!(top1_equal, n, "the head's top-1 differs from the reference on {} rows", n - top1_equal);
    assert!(
        max_logit_diff <= 2.0 * max_noise.max(1e-3),
        "the logits differ by {max_logit_diff:.3e}, above twice the bf16 noise floor {max_noise:.3e}"
    );
    m.end_slot(1);
    m.end_slot(2);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Step 7: exactness. The head only proposes: the trunk's candidates and state must be bit-identical with it in the loop, and the head's own
// state (its cache, its taps, its book) must be what teacher forcing over the trunk's true rows gives.

use gb_cuda::{DevBuf, Stream};
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::model::{NemotronModel, Point, Tap};
use gb_model::families::nemotron_h_omni::mtp::MtpMutant;
use std::sync::{Arc, Mutex};

/// Keeps the f32 rows of the trunk's last final norm (`Point::FinalNorm`): the tensor the trunk feeds to `lm_head`.
struct FinalNormTap(Arc<Mutex<Vec<f32>>>);

impl Tap for FinalNormTap {
    fn tap(&mut self, point: Point, buf: &DevBuf, rows: usize, width: usize, st: &Stream) -> Result<(), ModelError> {
        if point == Point::FinalNorm {
            st.sync()?;
            let mut v = vec![0f32; rows * width];
            buf.download(&mut v)?;
            *self.0.lock().unwrap() = v;
        }
        Ok(())
    }
}

/// Tokens per verify round the head must reach on the checked prompts at depth `k` (about 80% of the lowest seen: 1.59 / 1.93 / 2.08 / 3.00
/// at K = 1..4 on the 150-token case; the drafts of a head with a broken tap or state, or a wrong one, stop matching and give 1.0 to 1.2).
fn floor_tokens_per_round(k: usize) -> f64 {
    [0.0, 1.3, 1.5, 1.65, 2.0][k]
}

/// One full check at `depth` on `prompt`: the committed tokens, candidates (bit for bit) and trunk state equal the serial run's; the head's
/// cache equals teacher forcing over the true rows; the book advanced with the committed rows; the head drafts well (not merely harmlessly).
/// `Err` names the first thing that is wrong.
fn exact_check(
    m: &mut NemotronModel,
    prompt: &[u32],
    serial: &Gen,
    n_new: usize,
    depth: usize,
    floor: bool,
) -> Result<String, String> {
    let t0 = std::time::Instant::now();
    let (spec, run) = spec_ex(m, 1, prompt, n_new, depth);
    let t_spec = t0.elapsed().as_secs_f64();
    if spec.tokens != serial.tokens {
        return Err("the drafted token stream differs from the serial one".into());
    }
    if let Some(j) = (0..n_new).find(|&j| spec.cands[j] != serial.cands[j]) {
        return Err(format!("the candidates of generated token {j} differ from the serial run's"));
    }
    let fed = prompt.len() + n_new - 1;
    let t1 = std::time::Instant::now();
    same_trunk_state(m, 0, 1, fed)?;
    let t_state = t1.elapsed().as_secs_f64();
    // the head's state: entries below the book's `first` are built; entry m sits at position m - 1
    let built = m.mtp_book(1).map(|b| b.first).unwrap_or(0).saturating_sub(1);
    if built + 2 * depth + 8 < fed {
        return Err(format!("the head's state did not advance with the committed rows: {built} of {fed} positions built"));
    }
    // teacher forcing in slot 2: entry i pairs the true tap of position i with the token at i + 1, one pass over all of them
    let t2 = std::time::Instant::now();
    m.end_slot(2);
    m.begin_slot(2).map_err(|e| e.to_string())?;
    if built > 0 {
        let hidden = &serial.taps[..built * 4096];
        m.mtp_debug_forward(2, 0, &serial.tokens[1..=built], hidden, 1).map_err(|e| e.to_string())?;
        let (ks, vs) = m.mtp_kv_rows(1, 0, built).map_err(|e| e.to_string())?;
        let (kt, vt) = m.mtp_kv_rows(2, 0, built).map_err(|e| e.to_string())?;
        if let Some(r) = (0..built).find(|&r| {
            ks[r * 256..(r + 1) * 256] != kt[r * 256..(r + 1) * 256] || vs[r * 256..(r + 1) * 256] != vt[r * 256..(r + 1) * 256]
        }) {
            return Err(format!("the head's cache differs from the teacher-forced one from position {r} of {built}"));
        }
    }
    m.end_slot(2);
    let t_tf = t2.elapsed().as_secs_f64();
    let tpr = (n_new - 1) as f64 / run.rounds as f64;
    let line = format!(
        "K={depth}: {} rounds, {} drafted, {} accepted ({tpr:.2} tokens per round), {} rounds undrafted; tokens, candidates and state equal \
         serial; the head's cache equals teacher forcing over {built} positions bit for bit [{t_spec:.0} s spec run, {t_state:.0} s state \
         compare, {t_tf:.0} s teacher forcing]",
        run.rounds, run.drafted, run.accepted, run.undrafted
    );
    if floor && tpr < floor_tokens_per_round(depth) {
        return Err(format!(
            "{line}: below the floor of {:.2} tokens per round: the head is not drafting well",
            floor_tokens_per_round(depth)
        ));
    }
    Ok(line)
}

/// The cases of the exactness runs: oracle text cut so that prompts end off the 64-row grid, and decoding crosses the 512-key attention
/// chunk edge.
fn exact_cases(prefix: &[u32]) -> Vec<(&'static str, Vec<u32>, usize)> {
    // the trunk's steps cost about 270 ms and its 64-row prefill slices 2.5 s each today, so the cases are short
    vec![
        ("oracle text, 150 tokens", prefix[..150].to_vec(), 28),
        ("oracle text, 505 tokens (decoding crosses the 512-key chunk edge)", prefix[..505].to_vec(), 16),
    ]
}

#[test]
#[ignore]
fn drafter_in_the_loop_equals_serial() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    assert!(m.has_mtp());
    m.set_mtp_mutant(None);
    for (name, prompt, n_new) in exact_cases(&prefix) {
        let t = std::time::Instant::now();
        let serial = serial_ex(&mut m, 0, &prompt, n_new);
        eprintln!("{name}: the serial run took {:.0} s", t.elapsed().as_secs_f64());
        for depth in 1..=4 {
            let line =
                exact_check(&mut m, &prompt, &serial, n_new, depth, true).unwrap_or_else(|e| panic!("{name}: K={depth}: {e}"));
            eprintln!("{name}: {line}");
            m.end_slot(1);
        }
        m.end_slot(0);
    }
}

#[test]
#[ignore]
fn the_tap_buffer_is_the_final_norm_output() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    let seen = Arc::new(Mutex::new(Vec::new()));
    m.set_tap(Some(Box::new(FinalNormTap(seen.clone()))));
    // a window: the retained tap rows are the final norm's rows, bit for bit
    let first = prefill_ex(&mut m, 3, &prefix[..40], 64, true, None).0;
    window_argmax(&mut m, 3, &[first, 7, 11], 40);
    let cap = seen.lock().unwrap().clone();
    assert_eq!(cap.len(), 3 * 4096);
    let tap = m.mtp_tap_rows(3).unwrap();
    assert!(bits(&tap) == bits(&cap), "the head's tap is not the final norm's output (the lm_head's input) in a window");
    // a prompt slice: the slot's carry (store row 0) is the slice's last final-norm row
    let (_, _) = prefill_ex(&mut m, 3, &prefix[..40], 64, true, None);
    let cap = seen.lock().unwrap().clone();
    assert_eq!(cap.len(), 40 * 4096);
    let carry = m.mtp_store_rows(3, 1).unwrap();
    assert!(bits(&carry) == bits(&cap[39 * 4096..]), "the carry is not the prompt slice's last final-norm row");
    m.set_tap(None);
    m.end_slot(3);
}

#[test]
#[ignore]
fn commit_moves_exactly_keep_rows() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    // two sequences in one window: rows [slot 4: 5 rows][slot 5: 3 rows]
    let (a, b) = (4usize, 5usize);
    let fa = prefill_ex(&mut m, a, &prefix[..30], 64, true, None).0;
    let fb = prefill_ex(&mut m, b, &prefix[100..150], 64, true, None).0;
    let wa: Vec<u32> = std::iter::once(fa).chain(prefix[200..204].iter().copied()).collect();
    let wb: Vec<u32> = std::iter::once(fb).chain(prefix[300..302].iter().copied()).collect();
    let req = [
        gb_engine::RowRequest { slot: a, tokens: wa.clone(), first_position: 30 },
        gb_engine::RowRequest { slot: b, tokens: wb.clone(), first_position: 50 },
    ];
    let batch = gb_model::rows::RowBatch::from_requests(&req);
    m.window_candidates(&batch, 1).unwrap();
    let win = m.mtp_tap_rows(8).unwrap();
    let before_a = m.mtp_store_rows(a, MTP_TAP_CAP).unwrap();
    let before_b = m.mtp_store_rows(b, MTP_TAP_CAP).unwrap();
    let (na, nb) = (m.mtp_book(a).unwrap().n, m.mtp_book(b).unwrap().n);
    assert_eq!((na, nb), (1, 1), "each slot holds its carry");
    // keep 3 of slot a's 5 rows and 1 of slot b's 3
    m.commit(a, 3);
    m.commit(b, 1);
    m.flush().unwrap();
    let after_a = m.mtp_store_rows(a, MTP_TAP_CAP).unwrap();
    let after_b = m.mtp_store_rows(b, MTP_TAP_CAP).unwrap();
    assert_eq!(
        (m.mtp_book(a).unwrap().n, m.mtp_book(b).unwrap().n),
        (na + 3, nb + 1),
        "the books advance by exactly the kept rows"
    );
    let row = |v: &[f32], r: usize| bits(&v[r * 4096..(r + 1) * 4096]);
    // the kept rows are the window's first `keep` rows of each slot, in order, and everything else in the stores is untouched
    for j in 0..3 {
        assert_eq!(row(&after_a, na + j), row(&win, j), "slot a store row {}", na + j);
    }
    assert_eq!(row(&after_b, nb), row(&win, 5), "slot b's kept row is the first of its rows in the window (row 5)");
    for r in (0..MTP_TAP_CAP).filter(|&r| !(na..na + 3).contains(&r)) {
        assert_eq!(row(&after_a, r), row(&before_a, r), "slot a store row {r} was touched");
    }
    for r in (0..MTP_TAP_CAP).filter(|&r| r != nb) {
        assert_eq!(row(&after_b, r), row(&before_b, r), "slot b store row {r} was touched");
    }
    m.end_slot(a);
    m.end_slot(b);
}

#[test]
#[ignore]
fn an_unsound_slot_drafts_nothing_and_nothing_breaks() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    let prompt = &prefix[..60];
    let serial = serial_ex(&mut m, 0, prompt, 12);

    // (1) a slot switched off drafts nothing and its book says so; the trunk goes on and equals serial
    let first = prefill_ex(&mut m, 1, prompt, 64, true, None).0;
    m.mtp_disable_slot(1);
    assert_eq!(m.mtp_pending_rows(1), None);
    let mut tokens = prompt.to_vec();
    tokens.push(first);
    assert!(m.mtp_draft(&[DraftAsk { slot: 1, tokens: &tokens, depth: 3 }], &[]).unwrap()[0].is_none());

    // (2) tokens that do not match the rows the store holds (a longer sequence than the rows account for): no draft, and the slot is off
    let first = prefill_ex(&mut m, 2, prompt, 64, true, None).0;
    let mut wrong = prompt.to_vec();
    wrong.extend([first, 5, 6]);
    assert!(m.mtp_draft(&[DraftAsk { slot: 2, tokens: &wrong, depth: 3 }], &[]).unwrap()[0].is_none());
    assert!(!m.mtp_book(2).unwrap().ok, "a mismatch switches the slot off");
    let mut right = prompt.to_vec();
    right.push(first);
    assert!(m.mtp_draft(&[DraftAsk { slot: 2, tokens: &right, depth: 3 }], &[]).unwrap()[0].is_none(), "and it stays off");

    // (3) more pending rows than the store holds: the slot turns itself off at the commit that overflows, silently
    let first = prefill_ex(&mut m, 3, prompt, 64, true, None).0;
    let mut t = prompt.to_vec();
    t.push(first);
    for _ in 0..MTP_TAP_CAP + 2 {
        let last = *t.last().unwrap();
        let next = window_argmax(&mut m, 3, &[last], t.len() - 1)[0];
        m.commit(3, 1);
        t.push(next);
    }
    m.flush().unwrap();
    assert_eq!(m.mtp_pending_rows(3), None, "an overflowed store switches the slot off");
    assert!(m.mtp_draft(&[DraftAsk { slot: 3, tokens: &t, depth: 2 }], &[]).unwrap()[0].is_none());

    // (4) a good slot drafts in the same call as the three bad ones (and a slot that was never begun is just refused)
    let first = prefill_ex(&mut m, 4, prompt, 64, true, None).0;
    let mut good = prompt.to_vec();
    good.push(first);
    let asks = [
        DraftAsk { slot: 1, tokens: &tokens, depth: 3 },
        DraftAsk { slot: 4, tokens: &good, depth: 3 },
        DraftAsk { slot: 3, tokens: &t, depth: 3 },
        DraftAsk { slot: 5, tokens: &good, depth: 3 },
    ];
    let out = m.mtp_draft(&asks, &[]).unwrap();
    assert!(out[0].is_none() && out[2].is_none() && out[3].is_none());
    assert_eq!(out[1].as_ref().map(|d| d.ids.len()), Some(3));
    // and the trunk is untouched by all of it: slot 4 continues as serial does
    let mut g = good.clone();
    while g.len() - prompt.len() < 12 {
        let last = *g.last().unwrap();
        let next = window_argmax(&mut m, 4, &[last], g.len() - 1)[0];
        m.commit(4, 1);
        g.push(next);
    }
    m.flush().unwrap();
    assert_eq!(g, serial.tokens, "the trunk's stream is the serial one after the head was refused, overflowed and mismatched");
    for s in 1..=4 {
        m.end_slot(s);
    }
}

#[test]
#[ignore]
fn every_deliberate_head_bug_is_noticed() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    // 150 tokens (three prefill chunks: the carry is used), K = 3: rounds with partial accepts
    let prompt = prefix[..150].to_vec();
    m.set_mtp_mutant(None);
    let serial = serial_ex(&mut m, 0, &prompt, 28);
    let clean =
        exact_check(&mut m, &prompt, &serial, 28, 3, true).unwrap_or_else(|e| panic!("the clean head fails its own check: {e}"));
    eprintln!("clean: {clean}");
    m.end_slot(1);
    for mutant in [
        MtpMutant::TapBeforeNorm,
        MtpMutant::NoShift,
        MtpMutant::AdvanceAllRows,
        MtpMutant::AbsorbLastOnly,
        MtpMutant::WrongDrafts,
    ] {
        m.set_mtp_mutant(Some(mutant));
        let r = exact_check(&mut m, &prompt, &serial, 28, 3, true);
        m.set_mtp_mutant(None);
        m.end_slot(1);
        m.end_slot(2);
        match r {
            Ok(line) => panic!("{mutant:?} was NOT noticed: {line}"),
            Err(why) => eprintln!("{mutant:?}: noticed: {why}"),
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Keyed drafting and the restricted head.

use gb_sampling::{try_choose, Sampling};
use gb_spec::{draft_position, try_draft_pick, DraftPick};

/// One stream of a keyed test: its slot, its tokens (the last one not yet fed) and its sampler.
struct Seq {
    slot: usize,
    tokens: Vec<u32>,
    s: Sampling,
}

/// The engine's sample of a row: the keyed draw over the first `candidate_count` candidates at `position`.
fn sample(c: &[i32], v: &[f32], position: usize, s: &Sampling) -> u32 {
    let n = s.candidate_count(131072).min(c.len());
    let ids: Vec<u32> = c[..n].iter().map(|&x| x as u32).collect();
    try_choose(&v[..n], &ids, position as u64, s).unwrap()
}

const CAND: usize = 48;

fn start_stream(m: &mut NemotronModel, slot: usize, prompt: &[u32], s: Sampling) -> Seq {
    m.end_slot(slot);
    m.begin_slot(slot).unwrap();
    let mut at = 0;
    let mut first = 0;
    while at < prompt.len() {
        let end = (at + 64 - at % 64).min(prompt.len());
        let b = gb_model::rows::RowBatch::from_requests(&[gb_engine::RowRequest {
            slot,
            first_position: at as u64,
            tokens: prompt[at..end].to_vec(),
        }]);
        let want = (end == prompt.len()).then_some(b.rows - 1);
        if let Some(c) = m.prefill_chunk(&b, want).unwrap() {
            first = sample(&c.ids[..CAND], &c.vals[..CAND], prompt.len(), &s);
        }
        at = end;
    }
    let mut tokens = prompt.to_vec();
    tokens.push(first);
    Seq { slot, tokens, s }
}

/// One serial row for every stream together, each stream's next token by its own sampler; commits them.
fn advance(m: &mut NemotronModel, streams: &mut [Seq]) {
    let reqs: Vec<gb_engine::RowRequest> = streams
        .iter()
        .map(|st| gb_engine::RowRequest {
            slot: st.slot,
            tokens: vec![*st.tokens.last().unwrap()],
            first_position: (st.tokens.len() - 1) as u64,
        })
        .collect();
    let c = m.window_candidates(&gb_model::rows::RowBatch::from_requests(&reqs), CAND).unwrap();
    for (i, st) in streams.iter_mut().enumerate() {
        let t = sample(&c.ids[i * CAND..(i + 1) * CAND], &c.vals[i * CAND..(i + 1) * CAND], st.tokens.len(), &st.s);
        m.commit(st.slot, 1);
        st.tokens.push(t);
    }
}

type HeadList = (Vec<u32>, Vec<f32>);

/// The host's draft chain for `st`: the candidates the head produced at each of `depth` depths, the chain following the host's own keyed pick.
fn host_chain(m: &mut NemotronModel, st: &Seq, depth: usize, restricted: bool) -> (Vec<u32>, Vec<HeadList>) {
    let len = st.tokens.len();
    let mut picks: Vec<u32> = Vec::new();
    let mut f = |k: usize, ids: &[u32], vals: &[f32]| {
        let t = try_draft_pick(vals, ids, draft_position(len, k), &st.s, DraftPick::Keyed).unwrap();
        picks.push(t);
        t
    };
    let lists = m.mtp_debug_chain(st.slot, &st.tokens, depth, 64, restricted, false, &mut f).unwrap().expect("the slot drafts");
    (picks, lists)
}

fn samplers() -> Vec<Sampling> {
    vec![
        Sampling { seed: 11, temperature: 0.7, top_k: 20, top_p: 0.95, min_p: 0.0 },
        Sampling { seed: 12, temperature: 1.0, top_k: 8, top_p: 1.0, min_p: 0.0 },
        Sampling { seed: 13, temperature: 1.3, top_k: 20, top_p: 0.9, min_p: 0.05 },
        Sampling { seed: 0, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 },
    ]
}

#[test]
#[ignore]
fn keyed_drafts_are_the_hosts_pick_over_the_heads_candidates() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    m.set_mtp_restricted(false);
    let ps = [prefix[..60].to_vec(), prefix[100..180].to_vec(), prefix[300..340].to_vec(), prefix[500..590].to_vec()];
    let ss = samplers();
    let mut streams: Vec<Seq> = ps.iter().zip(&ss).enumerate().map(|(i, (p, s))| start_stream(&mut m, i, p, *s)).collect();
    let (mut compared, mut differ_from_argmax, mut keyed_compared) = (0usize, 0usize, 0usize);
    for round in 0..24usize {
        let depth = 1 + round % 4;
        // the host's chains first (they leave the slots' pending rows alone), then the one device call for all four streams
        let host: Vec<(Vec<u32>, Vec<HeadList>)> = streams.iter().map(|st| host_chain(&mut m, st, depth, false)).collect();
        let asks: Vec<DraftAsk> = streams.iter().map(|st| DraftAsk { slot: st.slot, tokens: &st.tokens, depth }).collect();
        let keyed: Vec<Option<Sampling>> = streams.iter().map(|st| Some(st.s)).collect();
        let got = m.mtp_draft(&asks, &keyed).unwrap();
        for (i, st) in streams.iter().enumerate() {
            let d = got[i].as_ref().expect("the stream drafts");
            if d.ids != host[i].0 {
                let k = (0..depth).find(|&k| d.ids[k] != host[i].0[k]).unwrap();
                let top: Vec<(u32, f32)> = host[i].1[k].0.iter().zip(&host[i].1[k].1).take(6).map(|(&a, &b)| (a, b)).collect();
                panic!(
                    "round {round} stream {i} ({:?}) depth {depth}: the device's drafts {:?} differ from the host's keyed pick {:?} over the \
                     head's candidates, first at draft {k} (device logit {}); the host's top candidates there: {top:?}",
                    st.s, d.ids, host[i].0, d.vals[k]
                );
            }
            for k in 0..depth {
                let at = host[i].1[k].0.iter().position(|&x| x == d.ids[k]).expect("the pick is one of the candidates");
                assert_eq!(
                    d.vals[k].to_bits(),
                    host[i].1[k].1[at].to_bits(),
                    "round {round} stream {i} depth {k}: the returned logit"
                );
                compared += 1;
                if st.s.temperature > 0.0 {
                    keyed_compared += 1;
                    differ_from_argmax += usize::from(d.ids[k] != host[i].1[k].0[0]);
                } else {
                    assert_eq!(d.ids[k], host[i].1[k].0[0], "a greedy stream's draft is the argmax");
                }
            }
        }
        advance(&mut m, &mut streams);
    }
    eprintln!("{compared} drafts equal the host's pick; {differ_from_argmax} of {keyed_compared} sampled drafts are not the head's argmax");
    assert!(
        differ_from_argmax >= 8,
        "the noise changed only {differ_from_argmax} drafts: the comparison would not see a missing noise"
    );
    for s in 0..4 {
        m.end_slot(s);
    }
}

/// The head's drafts and their logit bits for one sequence do not depend on which other sequences share the call (rows of the head's
/// kernels are independent): the same state in two slots, one asked alone and one in a call of four.
#[test]
#[ignore]
fn a_drafts_bits_do_not_depend_on_the_other_asks_of_its_call() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    m.set_mtp_restricted(false);
    let g = Sampling { temperature: 0.0, ..Default::default() };
    let prompts = [
        prefix[..60].to_vec(),
        prefix[..60].to_vec(),
        prefix[100..180].to_vec(),
        prefix[300..340].to_vec(),
        prefix[500..590].to_vec(),
    ];
    // slots 0 and 1 are the same sequence; 2, 3, 4 are the others
    let mut streams: Vec<Seq> = prompts.iter().enumerate().map(|(i, p)| start_stream(&mut m, i, p, g)).collect();
    let mut differing = 0usize;
    for round in 0..10usize {
        let depth = 1 + round % 4;
        let alone = m.mtp_draft(&[DraftAsk { slot: 0, tokens: &streams[0].tokens, depth }], &[]).unwrap().remove(0).unwrap();
        let asks: Vec<DraftAsk> =
            [1usize, 2, 3, 4].iter().map(|&i| DraftAsk { slot: i, tokens: &streams[i].tokens, depth }).collect();
        let batched = m.mtp_draft(&asks, &[]).unwrap().remove(0).unwrap();
        let same_ids = alone.ids == batched.ids;
        let same_vals = alone.vals.iter().map(|v| v.to_bits()).eq(batched.vals.iter().map(|v| v.to_bits()));
        if !(same_ids && same_vals) {
            differing += 1;
        }
        eprintln!(
            "round {round} depth {depth}: alone {:?} {:?}; in a call of four {:?} {:?}; ids equal {same_ids}, logit bits equal {same_vals}",
            alone.ids, alone.vals, batched.ids, batched.vals
        );
        advance(&mut m, &mut streams);
    }
    for s in 0..5 {
        m.end_slot(s);
    }
    assert_eq!(differing, 0, "a draft changed with the other asks of its call in {differing} of 10 rounds");
}

/// The draft call is deterministic: the same call repeated on the same state (the slots' pending rows kept pending) gives the same drafts and
/// logit bits, with and without the host's debug chains running in between.
#[test]
#[ignore]
fn the_draft_call_is_deterministic_under_repetition() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    m.set_mtp_restricted(false);
    let ss = samplers();
    let ps = [prefix[..60].to_vec(), prefix[100..180].to_vec(), prefix[300..340].to_vec(), prefix[500..590].to_vec()];
    let mut streams: Vec<Seq> = ps.iter().zip(&ss).enumerate().map(|(i, (p, s))| start_stream(&mut m, i, p, *s)).collect();
    for _ in 0..4 {
        advance(&mut m, &mut streams);
    }
    let depth = 4;
    let call = |m: &mut NemotronModel, streams: &[Seq]| -> Vec<(Vec<u32>, Vec<u32>)> {
        let asks: Vec<DraftAsk> = streams.iter().map(|st| DraftAsk { slot: st.slot, tokens: &st.tokens, depth }).collect();
        let keyed: Vec<Option<Sampling>> = streams.iter().map(|st| Some(st.s)).collect();
        m.mtp_draft(&asks, &keyed)
            .unwrap()
            .into_iter()
            .map(|d| {
                let d = d.unwrap();
                (d.ids, d.vals.iter().map(|v| v.to_bits()).collect())
            })
            .collect()
    };
    m.set_mtp_keep_pending(true);
    let reference = call(&mut m, &streams);
    let (mut diff_plain, mut diff_with_host, mut diff_host_chain) = (0usize, 0usize, 0usize);
    let host_ref: Vec<Vec<u32>> = streams.iter().map(|st| host_chain(&mut m, st, depth, false).0).collect();
    for rep in 0..30usize {
        if call(&mut m, &streams) != reference {
            diff_plain += 1;
        }
        // the host's chains for all four streams, then the device call
        for (i, st) in streams.iter().enumerate() {
            let h = host_chain(&mut m, st, depth, false).0;
            if h != host_ref[i] {
                diff_host_chain += 1;
                eprintln!("repetition {rep}: the host chain of stream {i} changed: {:?} vs {:?}", h, host_ref[i]);
            }
        }
        let after = call(&mut m, &streams);
        if after != reference {
            diff_with_host += 1;
            for (i, (a, r)) in after.iter().zip(&reference).enumerate() {
                if a != r {
                    eprintln!("repetition {rep}: stream {i} ({:?}): {:?} vs reference {:?}", streams[i].s, a.0, r.0);
                }
            }
        }
    }
    m.set_mtp_keep_pending(false);
    eprintln!(
        "30 repetitions: device calls differing from the first: {diff_plain} plain, {diff_with_host} after the host's chains; host chains that changed: {diff_host_chain}"
    );
    for s in 0..4 {
        m.end_slot(s);
    }
    assert_eq!(diff_plain + diff_with_host + diff_host_chain, 0, "the draft call is not deterministic");
}

/// The same with keyed drafting: two slots holding the same sequence and the same sampler, one asked alone and one in a call of four, then
/// the host's greedy chain against the device's argmax chain (no noise at all): which of the three pieces (the head, the keyed pick, the
/// debug chain) a disagreement lives in.
#[test]
#[ignore]
fn keyed_and_greedy_drafts_agree_between_calls_and_with_the_hosts_chain() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    m.set_mtp_restricted(false);
    let ss = samplers();
    let prompts = [
        prefix[..60].to_vec(),
        prefix[..60].to_vec(),
        prefix[100..180].to_vec(),
        prefix[300..340].to_vec(),
        prefix[500..590].to_vec(),
    ];
    // slots 0 and 1: the same sequence AND the same sampler; 2, 3, 4: others
    let samp = [ss[1], ss[1], ss[0], ss[2], ss[3]];
    let mut streams: Vec<Seq> = prompts.iter().enumerate().map(|(i, p)| start_stream(&mut m, i, p, samp[i])).collect();
    let (mut keyed_diff, mut greedy_diff, mut host_diff) = (0usize, 0usize, 0usize);
    for round in 0..16usize {
        let depth = 4;
        // the host's greedy chain for slot 4's sequence (it leaves the pending rows alone), taken as the argmax at every depth
        let greedy = Sampling { temperature: 0.0, ..streams[4].s };
        let hg = host_chain(&mut m, &Seq { slot: 4, tokens: streams[4].tokens.clone(), s: greedy }, depth, false);
        // keyed, alone and in a call of four
        let k_alone = m
            .mtp_draft(&[DraftAsk { slot: 0, tokens: &streams[0].tokens, depth }], &[Some(streams[0].s)])
            .unwrap()
            .remove(0)
            .unwrap();
        let asks: Vec<DraftAsk> =
            [1usize, 2, 3, 4].iter().map(|&i| DraftAsk { slot: i, tokens: &streams[i].tokens, depth }).collect();
        let keyed: Vec<Option<Sampling>> = [1usize, 2, 3, 4].iter().map(|&i| Some(streams[i].s)).collect();
        let mut out = m.mtp_draft(&asks, &keyed).unwrap();
        let d4 = out.pop().unwrap().unwrap(); // slot 4: the greedy stream
        let k_batched = out.remove(0).unwrap();
        let same = k_alone.ids == k_batched.ids
            && k_alone.vals.iter().map(|v| v.to_bits()).eq(k_batched.vals.iter().map(|v| v.to_bits()));
        keyed_diff += usize::from(!same);
        // slot 4 is greedy: its device draft is the argmax chain, and the host's greedy chain must be the same
        host_diff += usize::from(d4.ids != hg.0);
        greedy_diff += usize::from(d4.ids != (0..depth).map(|k| hg.1[k].0[0]).collect::<Vec<_>>() && d4.ids != hg.0);
        eprintln!(
            "round {round}: keyed alone {:?} / in four {:?} (equal: {same}); greedy device {:?} host {:?}",
            k_alone.ids, k_batched.ids, d4.ids, hg.0
        );
        advance(&mut m, &mut streams);
    }
    for s in 0..5 {
        m.end_slot(s);
    }
    eprintln!("keyed alone-vs-batched differing rounds: {keyed_diff}; greedy device-vs-host differing rounds: {host_diff} ({greedy_diff} by more than a pick)");
    assert_eq!(keyed_diff, 0, "a keyed draft changed with the other asks of its call");
    assert_eq!(host_diff, 0, "the greedy device chain differs from the host's greedy chain over the head's own candidates");
}

#[test]
#[ignore]
fn the_restricted_draft_head_equals_the_full_head_inside_the_set() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    // a deliberately ragged set: the low ids from 5 (so no column number equals its token id), a block of mid ids and every 7th id above
    let ids: Vec<u32> = (5..131072u32).filter(|&i| i < 20000 || (60000..64000).contains(&i) || i % 7 == 0).collect();
    let bytes = m.mtp_restrict_vocab(&ids).unwrap();
    eprintln!("restricted set: {} ids, {:.0} MiB on the device", ids.len(), mib(bytes as u64));
    assert!(m.mtp_restricted());
    let member = |t: u32| ids.binary_search(&t).is_ok();
    let prompt = &prefix[..120];
    // two slots in the same state: slot 0's draft is the full head's, slot 1's the restricted head's; both follow the full head's verified tokens
    let mut a = start_stream(&mut m, 0, prompt, Sampling { temperature: 0.0, ..Default::default() });
    let mut b = start_stream(&mut m, 1, prompt, Sampling { temperature: 0.0, ..Default::default() });
    let (mut inside, mut equal_inside, mut outside) = (0usize, 0usize, 0usize);
    for round in 0..30usize {
        let depth = 3;
        m.set_mtp_restricted(false);
        let full = m.mtp_draft(&[DraftAsk { slot: 0, tokens: &a.tokens, depth }], &[]).unwrap().remove(0).unwrap();
        m.set_mtp_restricted(true);
        let rest = m.mtp_draft(&[DraftAsk { slot: 1, tokens: &b.tokens, depth }], &[]).unwrap().remove(0).unwrap();
        assert_eq!(a.tokens, b.tokens);
        for k in 0..depth {
            assert!(member(rest.ids[k]), "round {round}: the restricted draft {} is not a token of the set", rest.ids[k]);
            // the chains agree up to the first difference; from there the heads were fed different tokens
            if (0..k).any(|j| full.ids[j] != rest.ids[j]) {
                break;
            }
            if member(full.ids[k]) {
                inside += 1;
                if full.ids[k] == rest.ids[k] && full.vals[k].to_bits() == rest.vals[k].to_bits() {
                    equal_inside += 1;
                } else {
                    panic!(
                        "round {round} depth {k}: the full head chose {} ({}), inside the set, the restricted head chose {} ({})",
                        full.ids[k], full.vals[k], rest.ids[k], rest.vals[k]
                    );
                }
            } else {
                outside += 1;
                assert_ne!(full.ids[k], rest.ids[k]);
            }
        }
        let mut both = [Seq { slot: 0, tokens: a.tokens.clone(), s: a.s }, Seq { slot: 1, tokens: b.tokens.clone(), s: b.s }];
        advance(&mut m, &mut both);
        let [x, y] = both;
        a = x;
        b = y;
    }
    eprintln!(
        "restricted vs full over 30 rounds: {inside} drafts with the full head's choice inside the set, {equal_inside} equal with the same logit bits; \
         {outside} where the full choice lay outside"
    );
    assert!(inside >= 30, "too few drafts landed inside the set to say anything: {inside}");
    // the deliberate bug: the argmax column is not mapped to its token id; the same comparison must now fail (every draft is a column number)
    m.set_mtp_mutant(Some(MtpMutant::RestrictedNoMap));
    m.set_mtp_restricted(false);
    let full = m.mtp_draft(&[DraftAsk { slot: 0, tokens: &a.tokens, depth: 2 }], &[]).unwrap().remove(0).unwrap();
    m.set_mtp_restricted(true);
    let rest = m.mtp_draft(&[DraftAsk { slot: 1, tokens: &b.tokens, depth: 2 }], &[]).unwrap().remove(0).unwrap();
    m.set_mtp_mutant(None);
    assert!(
        member(full.ids[0]) && full.ids[0] != rest.ids[0],
        "RestrictedNoMap was NOT noticed: full {:?}, restricted {:?}",
        full.ids,
        rest.ids
    );
    eprintln!("RestrictedNoMap: noticed (full head {:?}, unmapped restricted head {:?})", full.ids, rest.ids);
    m.set_mtp_restricted(false);
    m.end_slot(0);
    m.end_slot(1);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Step 8: acceptance and cost numbers (they feed the arbiter).

/// Per-depth statistics of one acceptance pass.
#[derive(Default, Clone, Debug)]
struct Acc {
    /// Draft positions evaluated (steps with all `DEPTH` true continuation tokens known).
    steps: usize,
    /// `tried[k]`: steps where drafts 0..k all matched (so draft k counts); `hit[k]`: of those, draft k matched too.
    tried: [usize; 4],
    hit: [usize; 4],
}

impl Acc {
    fn add(&mut self, o: &Acc) {
        self.steps += o.steps;
        for k in 0..4 {
            self.tried[k] += o.tried[k];
            self.hit[k] += o.hit[k];
        }
    }

    /// P(draft k matches | drafts 0..k matched).
    fn cond(&self, k: usize) -> f64 {
        if self.tried[k] == 0 {
            f64::NAN
        } else {
            self.hit[k] as f64 / self.tried[k] as f64
        }
    }

    /// Expected accepted drafts per round at depth `d`: the sum over k < d of P(drafts 0..=k all match).
    fn accepted_at(&self, d: usize) -> f64 {
        (0..d).map(|k| self.hit[k] as f64 / self.steps.max(1) as f64).sum()
    }
}

/// The head's drafts replayed teacher-forced over the trunk's own greedy continuation (`g`: the prompt of `p` tokens, `n_new` generated, and the
/// trunk's post-`norm_f` row of every position fed): the head's cache is built over the prompt from the true taps, then at every sequence length
/// `L` from `p + 1` on the slot is given the true tap of position `L - 2` as its one pending row and drafts depth 4 (absorb, then chain on its
/// own drafts), exactly the state a serving round starts from. Draft `k` is compared with the token the trunk produced at position `L + k`.
/// The trunk is not run: the draft calls are the same ones a serial run with a head in the loop makes, on the same rows.
fn acceptance_replay(m: &mut NemotronModel, g: &Gen, p: usize, n_new: usize) -> Acc {
    const D: usize = 4;
    let slot = 1;
    m.end_slot(slot);
    m.begin_slot(slot).unwrap();
    // entries 0..p-1 (positions 0..p-2): the prompt's, one pass over the true taps
    m.mtp_debug_forward(slot, 0, &g.tokens[1..p], &g.taps[..(p - 1) * 4096], 1).unwrap();
    let mut acc = Acc::default();
    let last_len = p + n_new;
    for len in p + 1..=last_len {
        if len + D > g.tokens.len() {
            break;
        }
        // one pending row: the tap of position len - 2 pairs with the token at len - 1 (entry len - 1)
        m.mtp_debug_fill_store(slot, len - 1, &g.taps[(len - 2) * 4096..(len - 1) * 4096]).unwrap();
        let d = m
            .mtp_draft(&[DraftAsk { slot, tokens: &g.tokens[..len], depth: D }], &[])
            .unwrap()
            .remove(0)
            .expect("the slot drafts");
        acc.steps += 1;
        for k in 0..D {
            acc.tried[k] += 1;
            if d.ids[k] == g.tokens[len + k] {
                acc.hit[k] += 1;
            } else {
                break;
            }
        }
    }
    m.end_slot(slot);
    acc
}

fn print_acc(label: &str, a: &Acc) {
    let c: Vec<String> = (0..4).map(|k| format!("{:.3}", a.cond(k))).collect();
    eprintln!(
        "{label}: {} steps; P(draft k ok | previous ok) k=0..3: {}; accepted drafts per round at depth 1/2/3/4: {:.2} / {:.2} / {:.2} / {:.2}",
        a.steps,
        c.join(" "),
        a.accepted_at(1),
        a.accepted_at(2),
        a.accepted_at(3),
        a.accepted_at(4),
    );
}

#[test]
#[ignore]
fn acceptance_by_depth_and_vocabulary() {
    let Some(prompts) = prompts() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    let n_new = 128;
    // the vocabularies to sweep: `REXL3_MTP_VOCAB_DIR/vocab.<N>` (mtp_vocab.py vocab); the full head always
    let mut sets: Vec<(String, Option<Vec<u32>>)> = vec![("full 131072".into(), None)];
    if let Some(dir) = std::env::var_os("REXL3_MTP_VOCAB_DIR") {
        for n in [65536usize, 49152, 32768] {
            let p = std::path::Path::new(&dir).join(format!("vocab.{n}"));
            if let Ok(text) = std::fs::read_to_string(&p) {
                let ids = gb_model::families::nemotron_h_omni::mtp::parse_vocab_file(&text)
                    .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
                sets.push((format!("restricted {n}"), Some(ids)));
            }
        }
    }
    // the trunk's greedy continuation of every prompt, once, with its taps (the head's vocabulary does not change it)
    let mut gens = Vec::new();
    for (name, p) in &prompts {
        let t = std::time::Instant::now();
        let g = serial_ex(&mut m, 0, p, n_new);
        m.end_slot(0);
        eprintln!("{name}: {} prompt tokens, {n_new} generated in {:.0} s", p.len(), t.elapsed().as_secs_f64());
        gens.push(g);
    }
    for (label, ids) in &sets {
        match ids {
            None => m.set_mtp_restricted(false),
            Some(ids) => {
                let bytes = m.mtp_restrict_vocab(ids).unwrap();
                eprintln!("{label}: {} ids, {:.0} MiB on the device", ids.len(), mib(bytes as u64));
            }
        }
        let mut total = Acc::default();
        let t = std::time::Instant::now();
        for ((name, p), g) in prompts.iter().zip(&gens) {
            let a = acceptance_replay(&mut m, g, p.len(), n_new);
            print_acc(&format!("  {label} / {name} ({} prompt tokens)", p.len()), &a);
            total.add(&a);
        }
        print_acc(&format!("{label} / ALL"), &total);
        eprintln!("  ({:.0} s of head work)", t.elapsed().as_secs_f64());
    }
}

#[test]
#[ignore]
fn the_heads_cost_in_prefill_and_in_a_window_commit() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    let time_prefill = |m: &mut NemotronModel, drafting: bool| -> Vec<f64> {
        m.end_slot(0);
        m.begin_slot(0).unwrap();
        if !drafting {
            m.mtp_disable_slot(0);
        }
        let mut ms = Vec::new();
        for c in 0..3usize {
            let toks = prefix[c * 64..(c + 1) * 64].to_vec();
            let b = gb_model::rows::RowBatch::from_requests(&[gb_engine::RowRequest {
                slot: 0,
                first_position: (c * 64) as u64,
                tokens: toks,
            }]);
            let t = std::time::Instant::now();
            m.prefill_chunk(&b, None).unwrap();
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms
    };
    // two rounds each (the first of every kind pays any cold start)
    let _ = time_prefill(&mut m, true);
    let _ = time_prefill(&mut m, false);
    let with = time_prefill(&mut m, true);
    let without = time_prefill(&mut m, false);
    eprintln!("prefill chunks of 64 rows, ms: head on {with:.1?}, head off (slot switched off) {without:.1?}");
    // a one-row window and its commit: the retention copy and the tap move
    let time_window = |m: &mut NemotronModel, drafting: bool| -> f64 {
        let first = prefill_ex(m, 1, &prefix[..64], 64, drafting, None).0;
        let mut t = Vec::new();
        let mut last = first;
        for i in 0..12usize {
            let st = std::time::Instant::now();
            window_argmax(m, 1, &[last], 64 + i);
            m.commit(1, 1);
            m.flush().unwrap();
            t.push(st.elapsed().as_secs_f64() * 1e3);
            last = 1000 + i as u32;
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        t[t.len() / 2]
    };
    let w_on = time_window(&mut m, true);
    let w_off = time_window(&mut m, false);
    eprintln!("a one-row window + commit + flush, median ms: head on {w_on:.2}, head off {w_off:.2}");
    m.end_slot(0);
    m.end_slot(1);
}

#[test]
#[ignore]
fn draft_call_costs() {
    let Some(prefix) = oracle_ids() else { return };
    let Some(mut m) = model() else { return };
    m.set_mtp_mutant(None);
    m.set_mtp_restricted(false);
    let slots = 4usize;
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    // 8 REAL taps (a window of 8 rows of the oracle's text): the pending rows of every ask, so the router sees real hidden rows
    m.end_slot(0);
    m.begin_slot(0).unwrap();
    window_argmax(&mut m, 0, &prefix[400..408], 0);
    let taps = m.mtp_tap_rows(8).unwrap();
    m.end_slot(0);
    // the entry the pending rows start at: a context of about 256 tokens
    const FIRST: usize = 256;
    // seqs[s]: the sequence of slot s (its last token not yet fed)
    let setup = |m: &mut NemotronModel| -> Vec<Vec<u32>> {
        (0..slots)
            .map(|s| {
                m.end_slot(s);
                m.begin_slot(s).unwrap();
                prefix[s * 37..s * 37 + FIRST + 1].to_vec()
            })
            .collect()
    };
    // `rows` pending rows in every one of the first `n_slots` slots, as if the trunk had committed them (the call's cost does not depend on
    // how they got there): their tokens are real text, which varies the routed experts between asks and iterations
    let pend = |m: &mut NemotronModel, seqs: &mut Vec<Vec<u32>>, n_slots: usize, rows: usize, it: usize| {
        for (s, seq) in seqs.iter_mut().enumerate().take(n_slots) {
            let from = s * 37 + (it % 7) * 5;
            *seq = prefix[from..from + FIRST + rows].to_vec();
            m.mtp_debug_fill_store(s, FIRST, &taps[..rows * 4096]).unwrap();
        }
    };
    // REXL3_MTP_COST_RESTRICTED=1: the whole table with the 32k restricted head (REXL3_MTP_VOCAB_DIR/vocab.32768)
    if std::env::var_os("REXL3_MTP_COST_RESTRICTED").is_some() {
        let dir = std::env::var_os("REXL3_MTP_VOCAB_DIR").expect("REXL3_MTP_COST_RESTRICTED needs REXL3_MTP_VOCAB_DIR");
        let p = std::path::Path::new(&dir).join("vocab.32768");
        let ids = gb_model::families::nemotron_h_omni::mtp::parse_vocab_file(&std::fs::read_to_string(p).unwrap()).unwrap();
        m.mtp_restrict_vocab(&ids).unwrap();
        m.set_mtp_restricted(true);
        eprintln!("the table below is for the RESTRICTED head ({} ids)", ids.len());
    }
    eprintln!("draft call wall time, ms (median of 7 after 2 warmups): rows = pending rows absorbed per ask; asks x depth");
    eprintln!("asks depth |   rows=1      2      3      4      5      6      7      8");
    let mut table: Vec<(usize, usize, Vec<f64>)> = Vec::new();
    for asks in 1..=4usize {
        let mut seqs = setup(&mut m);
        for depth in 1..=3usize {
            let mut row = Vec::new();
            for rows in 1..=8usize {
                let mut times = Vec::new();
                for it in 0..9usize {
                    pend(&mut m, &mut seqs, asks, rows, it + 9 * rows + 100 * depth);
                    let ask: Vec<DraftAsk> = (0..asks).map(|s| DraftAsk { slot: s, tokens: &seqs[s], depth }).collect();
                    let t = std::time::Instant::now();
                    let out = m.mtp_draft(&ask, &[]).unwrap();
                    let dt = t.elapsed().as_secs_f64() * 1e3;
                    assert!(out.iter().all(|d| d.as_ref().is_some_and(|d| d.ids.len() == depth)));
                    if it >= 2 {
                        times.push(dt);
                    }
                }
                row.push(median(&mut times));
            }
            eprintln!("{asks:4} {depth:5} | {}", row.iter().map(|t| format!("{t:6.1}")).collect::<Vec<_>>().join(" "));
            table.push((asks, depth, row));
        }
        for s in 0..slots {
            m.end_slot(s);
        }
    }
    // the per-stage breakdown at 1 ask, 1 pending row, depth 1 and 3 (the arbiter's common cases) and 4 asks, 4 rows; then the same with the
    // restricted head (`REXL3_MTP_VOCAB_DIR/vocab.32768` when given)
    let mut cases = vec![(1usize, 1usize, 1usize, false), (1, 1, 3, false), (1, 4, 3, false), (4, 4, 3, false)];
    let restricted_file = std::env::var_os("REXL3_MTP_VOCAB_DIR").map(|d| std::path::Path::new(&d).join("vocab.32768"));
    if let Some(p) = restricted_file.as_ref().filter(|p| p.exists()) {
        let ids = gb_model::families::nemotron_h_omni::mtp::parse_vocab_file(&std::fs::read_to_string(p).unwrap()).unwrap();
        let bytes = m.mtp_restrict_vocab(&ids).unwrap();
        eprintln!("restricted head: {} ids, {:.0} MiB on the device", ids.len(), mib(bytes as u64));
        cases.extend([(1, 1, 1, true), (1, 1, 3, true), (4, 4, 3, true)]);
    }
    for (asks, rows, depth, restricted) in cases {
        m.set_mtp_restricted(restricted);
        let mut seqs = setup(&mut m);
        m.set_mtp_profile(true);
        let calls = 5usize;
        for it in 0..calls + 2 {
            pend(&mut m, &mut seqs, asks, rows, it);
            if it == 2 {
                m.take_mtp_profile();
            }
            let ask: Vec<DraftAsk> = (0..asks).map(|s| DraftAsk { slot: s, tokens: &seqs[s], depth }).collect();
            m.mtp_draft(&ask, &[]).unwrap();
        }
        let p = m.take_mtp_profile();
        m.set_mtp_profile(false);
        eprintln!(
            "stage breakdown, {asks} asks x {rows} pending rows x depth {depth}, {} head (ms per call, mean of {calls}):",
            if restricted { "restricted" } else { "full" }
        );
        let mut sum = 0.0;
        for (name, ms) in p {
            eprintln!("    {name:55} {:7.2}", ms / calls as f64);
            sum += ms / calls as f64;
        }
        eprintln!("    {:55} {:7.2}", "sum of stages", sum);
        for s in 0..slots {
            m.end_slot(s);
        }
    }
}
