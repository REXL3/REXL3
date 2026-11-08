//! The wide decode window and the fused decode step of the Nemotron model (AGG): many single-row sequences in one step.
//!
//! Needs `REXL3_NEMO_OMNI_DIR` (a pack) and a CUDA device; otherwise it reports what it did not run and passes. The first 8 trunk layers
//! (`MEMEMEM*`: four Mamba2, three latent-MoE and one attention layer) carry every kernel a window runs, so the model is cheap enough to
//! hold 130 slots; `REXL3_NEMO_AGG_LAYERS` runs more of the trunk. One model is built once, with a decode window of 128 rows, and the checks
//! run in order on its slots:
//!
//! 1. **row invariance**: a row's candidates (ids and the bits of the values) are the same in a window of 1, 2, 63, 64, 65, 96, 127 and
//!    128 sequences (in two orders, positions all different) as in a solo window of its own sequence; a window of mixed row counts
//!    (1, 3 and 5 rows a sequence) the same, row by row;
//! 2. **window == serial** for 128 single-row sequences, over several steps: the tokens and the candidates of every row of every step,
//!    and the Mamba S and conv pools of every layer and slot at the end, equal those of 128 solo sequences decoded one window and one
//!    commit at a time;
//! 3. **fused == plain**: the same steps through `window_candidates_fused` (one-row sequences applied inside the layers), and a mixed
//!    window (one-row sequences beside sequences of drafts, some rolled back) leave the same pools and candidates as the plain window and
//!    `commit`;
//! 4. **graphs**: the wide shapes are captured and replayed (the counters say so) and eager and replayed windows give the same bits, for
//!    the plain and the fused window.
//!
//! Peak device memory is the weights of 8 layers plus 130 slots of 4 Mamba layers (about 2.3 GB): a short session.
#![allow(clippy::type_complexity, clippy::needless_range_loop)] // GPU test tables and per-stream loops, kept as written

use gb_engine::RowRequest;
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::ModelOptions;
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::model::NemotronModel;
use gb_model::families::nemotron_h_omni::weights::LayerPlan;
use gb_model::forward::Candidates;
use gb_model::rows::RowBatch;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

const SLOTS: usize = 130;
const WIDE: usize = 128;
const COUNT: usize = 64;

fn pack_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set (the pack directory)");
    }
    dir
}

fn config(dir: &Path) -> OmniConfig {
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| panic!("config.json: {e}"));
    OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap_or_else(|e| panic!("{e}"))
}

/// A deterministic token stream inside the vocabulary (xorshift64*).
fn tokens(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as u32 % 100_000 + 1000
        })
        .collect()
}

fn batch(reqs: &[(usize, usize, Vec<u32>)]) -> RowBatch {
    let r: Vec<RowRequest> = reqs
        .iter()
        .map(|(slot, first, t)| RowRequest { slot: *slot, first_position: *first as u64, tokens: t.clone() })
        .collect();
    RowBatch::from_requests(&r)
}

/// The prompt of slot `s`: lengths 20..=44 so that the 128 sequences sit at 25 different positions.
fn prompt(s: usize) -> Vec<u32> {
    tokens(1000 + s as u64, 20 + (s * 7) % 25)
}

/// Begins every slot `0..n` and prefills its prompt, several sequences to a slice. Leaves each slot at the length of its prompt.
fn prefill_all(m: &mut NemotronModel, n: usize) -> Vec<usize> {
    let mut lens = Vec::new();
    let (mut pending, mut rows) = (Vec::new(), 0usize);
    for s in 0..n {
        // a slot that is live is ended first (a second sequence in it starts from nothing)
        m.end_slot(s);
        m.begin_slot(s).unwrap();
        let p = prompt(s);
        lens.push(p.len());
        if rows + p.len() > 256 {
            m.prefill_chunk(&batch(&std::mem::take(&mut pending)), None).unwrap();
            rows = 0;
        }
        rows += p.len();
        pending.push((s, 0, p));
    }
    if !pending.is_empty() {
        m.prefill_chunk(&batch(&pending), None).unwrap();
    }
    m.flush().unwrap();
    lens
}

fn bits(c: &Candidates, r: usize) -> (Vec<i32>, Vec<u32>) {
    (c.ids[r * c.count..(r + 1) * c.count].to_vec(), c.vals[r * c.count..(r + 1) * c.count].iter().map(|v| v.to_bits()).collect())
}

/// One hash of every Mamba layer's S and conv state of slots `0..n`, and of the slots' lengths.
fn digest(m: &mut NemotronModel, n: usize) -> u64 {
    let mut h = DefaultHasher::new();
    let layers = m.core.mamba.len();
    for ord in 0..layers {
        for s in 0..n {
            let (st, conv) = m.debug_mamba_state(ord, s).unwrap();
            st.iter().for_each(|x| x.to_bits().hash(&mut h));
            conv.hash(&mut h);
        }
    }
    (0..n).for_each(|s| m.slot_len(s).hash(&mut h));
    h.finish()
}

/// The next token of a row: the best candidate.
fn pick(c: &Candidates, r: usize) -> u32 {
    c.ids[r * c.count] as u32
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// One window per sequence, one row, then its commit: what a serial decode is.
    Solo,
    /// One window of every sequence, `window_candidates`, then a commit each.
    Wide,
    /// One window of every sequence, `window_candidates_fused`, then a commit each.
    Fused,
}

/// `steps` decode steps of slots `0..n` from the state `prefill_all` left. Returns every step's candidates of every sequence (top
/// `COUNT` ids and value bits, in slot order) and the digest of the pools at the end.
fn decode(m: &mut NemotronModel, n: usize, lens: &[usize], steps: usize, mode: Mode) -> (Vec<Vec<(Vec<i32>, Vec<u32>)>>, u64) {
    // the first row of each sequence is the last prompt token's successor: any token, deterministic
    let mut next: Vec<u32> = (0..n).map(|s| tokens(7000 + s as u64, 1)[0]).collect();
    let mut log = Vec::new();
    for step in 0..steps {
        let mut out: Vec<(Vec<i32>, Vec<u32>)> = Vec::with_capacity(n);
        match mode {
            Mode::Solo => {
                for s in 0..n {
                    let c = m.window_candidates(&batch(&[(s, lens[s] + step, vec![next[s]])]), COUNT).unwrap();
                    out.push(bits(&c, 0));
                    next[s] = pick(&c, 0);
                    m.commit(s, 1);
                }
            }
            Mode::Wide | Mode::Fused => {
                let reqs: Vec<(usize, usize, Vec<u32>)> = (0..n).map(|s| (s, lens[s] + step, vec![next[s]])).collect();
                let b = batch(&reqs);
                let c = if mode == Mode::Wide {
                    m.window_candidates(&b, COUNT).unwrap()
                } else {
                    m.window_candidates_fused(&b, COUNT).unwrap()
                };
                assert_eq!(c.rows, n);
                for s in 0..n {
                    out.push(bits(&c, s));
                    next[s] = pick(&c, s);
                    m.commit(s, 1);
                }
            }
        }
        log.push(out);
    }
    m.flush().unwrap();
    let d = digest(m, n);
    (log, d)
}

#[test]
fn wide_windows_and_the_fused_decode_step_are_the_serial_decode_bit_for_bit() {
    let Some(dir) = pack_dir() else { return };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let layers = std::env::var("REXL3_NEMO_AGG_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize);
    let opts = ModelOptions {
        slots: SLOTS,
        kv_pages: SLOTS + 8,
        max_context: 2048,
        prefill_rows: 256,
        layers: Some(layers),
        mtp: false,
    };
    let mut m = match NemotronModel::load_wide(&dir, &cfg, &plan, opts, WIDE) {
        Ok(m) => m,
        Err(ModelError::NotBuilt(what)) => {
            eprintln!("not run: {what} is not built yet");
            return;
        }
        Err(e) => panic!("load: {e}"),
    };
    let t0 = std::time::Instant::now();

    // ---- 0. the plan prices what was allocated: the window records (a record and a retained xBC row a row, in every Mamba layer; the
    // prefill's shared record at the widest step) and the scratch (the decode MoE scratch, the large-row one, under P2 the tensor-core
    // block's, the attention partials)
    {
        use gb_cuda::nemotron_moe::MoeScratch;
        use gb_cuda::nemotron_moe_prefill::PrefillScratch;
        use gb_model::families::nemotron_h_omni::memory::NemotronMemory;
        use gb_model::families::nemotron_h_omni::model::{RECORD_ROW_BYTES, WINDOW_ROW_BYTES};
        let n_mamba = m.core.mamba.len() as u64;
        let widest = 256u64;
        let pools = m.pool_bytes();
        assert_eq!(pools.records, n_mamba * WIDE as u64 * WINDOW_ROW_BYTES as u64 + widest * RECORD_ROW_BYTES as u64);
        let attn = if m.core.map.count(gb_model::families::nemotron_h_omni::device::LayerKind::Attn) > 0 {
            NemotronMemory::attn_partial_bytes(64, 2048)
        } else {
            0
        };
        // under the prefill contract P2 (the default since c69fb643) the model also allocates the tensor-core MoE block's scratch at
        // the widest step, and the memory model prices it there (`memory.rs` `scratch_for`); under =p1 it is not allocated
        let mma = if gb_model::families::nemotron_h_omni::contract::current()
            == gb_model::families::nemotron_h_omni::contract::PrefillContract::P2
        {
            gb_cuda::nemotron_moe_prefill::mma::MmaScratch::bytes(widest as usize)
        } else {
            0
        };
        assert_eq!(pools.scratch, MoeScratch::bytes(64) + PrefillScratch::bytes(widest as usize) + mma + attn);
        assert_eq!(pools.state, n_mamba * SLOTS as u64 * (4_194_304 + 61_440));
    }

    // ---- 1. row invariance
    let lens = prefill_all(&mut m, WIDE + 2);
    let toks: Vec<u32> = (0..WIDE + 2).map(|s| tokens(7000 + s as u64, 1)[0]).collect();
    let solo: Vec<(Vec<i32>, Vec<u32>)> = (0..WIDE)
        .map(|s| {
            let c = m.window_candidates(&batch(&[(s, lens[s], vec![toks[s]])]), COUNT).unwrap();
            bits(&c, 0)
        })
        .collect();
    let pos = |s: usize| lens[s];
    for &n in &[1usize, 2, 63, 64, 65, 96, 127, 128] {
        // in slot order, and reversed (the row of a sequence is not at the same index)
        for reversed in [false, true] {
            let mut order: Vec<usize> = (0..n).collect();
            if reversed {
                order.reverse();
            }
            let reqs: Vec<(usize, usize, Vec<u32>)> = order.iter().map(|&s| (s, pos(s), vec![toks[s]])).collect();
            let c = m.window_candidates(&batch(&reqs), COUNT).unwrap();
            assert_eq!(c.rows, n);
            for (r, &s) in order.iter().enumerate() {
                assert!(
                    bits(&c, r) == solo[s],
                    "row of slot {s} in a window of {n} sequences (reversed: {reversed}) differs from solo"
                );
            }
        }
    }
    // mixed row counts: 1, 3 and 5 rows a sequence; each row equals the solo window of that sequence with the same rows
    {
        let rows_of = |s: usize| [1usize, 3, 5][s % 3];
        let reqs: Vec<(usize, usize, Vec<u32>)> = (0..40).map(|s| (s, pos(s), tokens(9000 + s as u64, rows_of(s)))).collect();
        let c = m.window_candidates(&batch(&reqs), COUNT).unwrap();
        let mut at = 0;
        for (s, p, t) in &reqs {
            let one = m.window_candidates(&batch(&[(*s, *p, t.clone())]), COUNT).unwrap();
            for r in 0..t.len() {
                assert!(
                    bits(&c, at + r) == bits(&one, r),
                    "row {r} of slot {s} in the mixed window differs from its solo window"
                );
            }
            at += t.len();
        }
        assert_eq!(c.rows, at);
    }
    eprintln!("1. row invariance: ok ({:.1} s)", t0.elapsed().as_secs_f64());

    // ---- 2. window == serial, 128 single-row sequences, 5 steps
    let steps = 5;
    let lens = prefill_all(&mut m, WIDE);
    let (log_solo, d_solo) = decode(&mut m, WIDE, &lens, steps, Mode::Solo);
    let lens = prefill_all(&mut m, WIDE);
    let (log_wide, d_wide) = decode(&mut m, WIDE, &lens, steps, Mode::Wide);
    for step in 0..steps {
        for s in 0..WIDE {
            assert!(log_wide[step][s] == log_solo[step][s], "step {step} slot {s}: wide window != serial");
        }
    }
    assert_eq!(d_wide, d_solo, "the pools after {steps} wide steps differ from the serial decode's");
    eprintln!("2. window == serial (128 sequences, {steps} steps): ok ({:.1} s)", t0.elapsed().as_secs_f64());

    // ---- 3. fused == plain
    let lens = prefill_all(&mut m, WIDE);
    let (log_fused, d_fused) = decode(&mut m, WIDE, &lens, steps, Mode::Fused);
    for step in 0..steps {
        for s in 0..WIDE {
            assert!(log_fused[step][s] == log_solo[step][s], "step {step} slot {s}: fused window != serial");
        }
    }
    assert_eq!(d_fused, d_solo, "the pools after {steps} fused steps differ from the serial decode's");
    // a window of fewer sequences, narrow (at most 64 rows) and in graphs' eager first sighting: the same
    for n in [1usize, 7, 64] {
        let lens = prefill_all(&mut m, n);
        let (a, da) = decode(&mut m, n, &lens, 3, Mode::Solo);
        let lens = prefill_all(&mut m, n);
        let (b, db) = decode(&mut m, n, &lens, 3, Mode::Fused);
        assert!(a == b && da == db, "{n} sequences: fused != serial");
    }
    // mixed: 1-row sequences beside sequences of 3 rows (drafts) over three steps; the plain window and the fused one leave one state
    // and every row's candidates agree
    let run_mixed = |m: &mut NemotronModel, fused: bool| -> (Vec<(Vec<i32>, Vec<u32>)>, u64) {
        let lens = prefill_all(m, 48);
        let mut log = Vec::new();
        for step in 0..3usize {
            let rows_of = |s: usize| if s % 3 == 0 { 3usize } else { 1 };
            // every sequence keeps all its rows, so its position after `step` windows is the prompt plus its rows a window
            let reqs: Vec<(usize, usize, Vec<u32>)> =
                (0..48).map(|s| (s, lens[s] + step * rows_of(s), tokens(500 + (step * 100 + s) as u64, rows_of(s)))).collect();
            let b = batch(&reqs);
            let c = if fused { m.window_candidates_fused(&b, COUNT).unwrap() } else { m.window_candidates(&b, COUNT).unwrap() };
            let mut at = 0;
            for (s, _, t) in &reqs {
                for r in 0..t.len() {
                    log.push(bits(&c, at + r));
                }
                at += t.len();
                m.commit(*s, t.len());
            }
        }
        m.flush().unwrap();
        (log, digest(m, 48))
    };
    let plain = run_mixed(&mut m, false);
    let fused = run_mixed(&mut m, true);
    assert!(plain.0 == fused.0, "mixed windows: the candidates of the fused window differ from the plain window's");
    assert_eq!(plain.1, fused.1, "mixed windows: the pools differ");
    // and a partial keep of a 3-row sequence beside applied one-row sequences
    {
        let lens = prefill_all(&mut m, 6);
        let reqs: Vec<(usize, usize, Vec<u32>)> =
            vec![(0, lens[0], tokens(1, 3)), (1, lens[1], tokens(2, 1)), (2, lens[2], tokens(3, 3)), (3, lens[3], tokens(4, 1))];
        let keep = [2usize, 1, 1, 1];
        let run = |m: &mut NemotronModel, fused: bool, lens: &[usize]| -> u64 {
            let b = batch(&reqs);
            if fused {
                m.window_candidates_fused(&b, 8).unwrap();
            } else {
                m.window_candidates(&b, 8).unwrap();
            }
            for ((s, _, _), &k) in reqs.iter().zip(&keep) {
                m.commit(*s, k);
            }
            m.flush().unwrap();
            let c = m
                .window_candidates(&batch(&(0..4).map(|s| (s, lens[s] + keep[s], vec![11u32])).collect::<Vec<_>>()), COUNT)
                .unwrap();
            let mut h = DefaultHasher::new();
            (0..4).for_each(|r| bits(&c, r).hash(&mut h));
            h.finish() ^ digest(m, 4)
        };
        let a = run(&mut m, false, &lens);
        let lens = prefill_all(&mut m, 6);
        let b = run(&mut m, true, &lens);
        assert_eq!(a, b, "a partial keep beside applied one-row sequences");
    }
    eprintln!("3. fused == plain: ok ({:.1} s)", t0.elapsed().as_secs_f64());

    // ---- 4. graphs: the wide shapes are captured and replayed, and a replay is the eager window
    m.clear_graphs().unwrap();
    m.set_graphs(true);
    let before = m.graph_stats();
    let lens = prefill_all(&mut m, WIDE);
    let (log_g, d_g) = decode(&mut m, WIDE, &lens, 6, Mode::Wide);
    let lens = prefill_all(&mut m, WIDE);
    let (log_gf, d_gf) = decode(&mut m, WIDE, &lens, 6, Mode::Fused);
    let after = m.graph_stats();
    assert!(after.captures >= before.captures + 2, "the wide plain and fused shapes were captured: {before:?} -> {after:?}");
    assert!(after.replays >= before.replays + 6, "and replayed: {before:?} -> {after:?}");
    assert_eq!(after.capture_failures, before.capture_failures, "no capture failed: {:?}", after.last_failure);
    m.set_graphs(false);
    let lens = prefill_all(&mut m, WIDE);
    let (log_e, d_e) = decode(&mut m, WIDE, &lens, 6, Mode::Wide);
    let lens = prefill_all(&mut m, WIDE);
    let (log_ef, d_ef) = decode(&mut m, WIDE, &lens, 6, Mode::Fused);
    for step in 0..6 {
        for s in 0..WIDE {
            assert!(log_g[step][s] == log_e[step][s], "step {step} slot {s}: a replayed wide window != the eager window");
            assert!(
                log_gf[step][s] == log_ef[step][s],
                "step {step} slot {s}: a replayed fused window != the eager fused window"
            );
            assert!(log_gf[step][s] == log_e[step][s], "step {step} slot {s}: fused != plain");
        }
    }
    assert_eq!((d_g, d_gf), (d_e, d_ef), "pools after replayed windows");
    assert_eq!(d_e, d_ef, "pools of plain and fused");
    m.set_graphs(true);
    eprintln!("4. graphs: {after:?}: ok ({:.1} s)", t0.elapsed().as_secs_f64());

    // ---- 5. several prompts in one forward: the candidates of each prompt's last row (one gathered lm_head) and every pool equal the
    // prompts prefilled one at a time with `want = Some(last row)`
    let n = 6;
    let prompts: Vec<Vec<u32>> = (0..n).map(prompt).collect();
    assert!(prompts.iter().map(Vec::len).sum::<usize>() <= 256);
    for s in 0..n {
        m.end_slot(s);
        m.begin_slot(s).unwrap();
    }
    let reqs: Vec<(usize, usize, Vec<u32>)> = prompts.iter().enumerate().map(|(s, p)| (s, 0, p.clone())).collect();
    let b = batch(&reqs);
    let wants: Vec<usize> = (0..n).map(|i| b.seq_rows(i).end - 1).collect();
    let together = m.prefill_chunk_rows(&b, &wants).unwrap().expect("candidates of the wanted rows");
    assert_eq!(together.rows, n);
    let d_together = digest(&mut m, n);
    let mut alone = Vec::new();
    for s in 0..n {
        m.end_slot(s);
        m.begin_slot(s).unwrap();
        let c = m.prefill_chunk(&batch(&[(s, 0, prompts[s].clone())]), Some(prompts[s].len() - 1)).unwrap().unwrap();
        alone.push(bits(&c, 0));
    }
    for s in 0..n {
        assert!(bits(&together, s) == alone[s], "prompt {s}: candidates of a shared forward differ from its own prefill's");
    }
    assert_eq!(d_together, digest(&mut m, n), "the pools after a shared forward differ from separate prefills'");
    eprintln!("5. prompts in one forward: ok ({:.1} s)", t0.elapsed().as_secs_f64());
}

/// The LAGGED decode (a fold period of 4 steps: a single-row sequence's state is stored every fourth step and the rows between are
/// re-applied from a ring) is the plain decode: the candidates of every row of every step and the true state of every slot after the
/// steps (`debug_mamba_state` folds first), through 12 steps of single rows (three fold periods, graphs capturing and replaying), and
/// through a script in which sequences change between one row and three rows from step to step (a sequence that drafts folds its
/// slot first), with a commit of every sequence. 128 sequences.
#[test]
fn a_lagged_decode_is_the_plain_decode_bit_for_bit() {
    let Some(dir) = pack_dir() else { return };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let layers = std::env::var("REXL3_NEMO_AGG_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize);
    let opts = ModelOptions {
        slots: SLOTS,
        kv_pages: SLOTS + 8,
        max_context: 2048,
        prefill_rows: 256,
        layers: Some(layers),
        mtp: false,
    };
    let mut m = NemotronModel::load_lag(&dir, &cfg, &plan, opts, WIDE, 4).unwrap_or_else(|e| panic!("load: {e}"));
    assert_eq!(m.lag_period(), 4);
    let n = WIDE;
    let t0 = std::time::Instant::now();

    // 1. single rows, 12 steps: lagged (fused) against plain windows and commits
    let lens = prefill_all(&mut m, n);
    let (log_plain, d_plain) = decode(&mut m, n, &lens, 12, Mode::Solo);
    let lens = prefill_all(&mut m, n);
    let (log_lag, d_lag) = decode(&mut m, n, &lens, 12, Mode::Fused);
    for step in 0..12 {
        for s in 0..n {
            assert!(log_lag[step][s] == log_plain[step][s], "step {step} slot {s}: lagged != plain");
        }
    }
    assert_eq!(d_lag, d_plain, "the true state after 12 lagged steps differs from the plain decode's");
    eprintln!("lag 1. single rows, 12 steps: ok ({:.1} s)", t0.elapsed().as_secs_f64());

    // 2. a script that mixes rows: at step t a sequence has 3 rows when (slot + t) % 5 == 0, else 1; every sequence keeps all its rows
    // 80 sequences: a fifth of them with three rows is 112 rows, inside the 128-row window
    let nm = 80usize;
    let run = |m: &mut NemotronModel, fused: bool| -> (Vec<(Vec<i32>, Vec<u32>)>, u64) {
        let lens = prefill_all(m, nm);
        let mut pos = lens.clone();
        let mut log = Vec::new();
        for step in 0..12usize {
            let reqs: Vec<(usize, usize, Vec<u32>)> = (0..nm)
                .map(|s| {
                    let rows = if (s + step) % 5 == 0 { 3 } else { 1 };
                    (s, pos[s], tokens(4000 + (step * 1000 + s) as u64, rows))
                })
                .collect();
            let b = batch(&reqs);
            let c = if fused { m.window_candidates_fused(&b, COUNT).unwrap() } else { m.window_candidates(&b, COUNT).unwrap() };
            let mut at = 0;
            for (s, _, t) in &reqs {
                for r in 0..t.len() {
                    log.push(bits(&c, at + r));
                }
                at += t.len();
                m.commit(*s, t.len());
                pos[*s] += t.len();
            }
        }
        m.flush().unwrap();
        (log, digest(m, nm))
    };
    let plain = run(&mut m, false);
    let lagged = run(&mut m, true);
    assert!(plain.0 == lagged.0, "the mixed script: the lagged window's candidates differ from the plain window's");
    assert_eq!(plain.1, lagged.1, "the mixed script: the true state differs");
    eprintln!("lag 2. mixed rows, 12 steps: ok ({:.1} s)", t0.elapsed().as_secs_f64());
}

/// The time of a decode step (window of N single-row sequences, then N commits and a flush) by N, plain and fused, graphs on, on the
/// whole trunk by default. `REXL3_NEMO_AGG_ROWS` (default `1,32,64,96,128`), `REXL3_NEMO_AGG_LAYERS` (default 88), and with
/// `REXL3_NEMO_AGG_PROFILE=1` the timed steps of the LAST row count are wrapped in `cudaProfilerStart/Stop` for
/// `nsys profile --capture-range=cudaProfilerApi`. Run with `--ignored --nocapture`.
#[test]
#[ignore = "a timing run: needs the pack and a GPU, prints a table"]
fn wide_step_times() {
    let Some(dir) = pack_dir() else { return };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let layers: usize = std::env::var("REXL3_NEMO_AGG_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(88);
    let rows_list: Vec<usize> = std::env::var("REXL3_NEMO_AGG_ROWS")
        .unwrap_or_else(|_| "1,32,64,96,128".into())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    let widest = *rows_list.iter().max().unwrap();
    let profile = std::env::var("REXL3_NEMO_AGG_PROFILE").is_ok_and(|v| v == "1");
    let opts = ModelOptions {
        slots: widest + 2,
        kv_pages: widest + 8,
        max_context: 2048,
        prefill_rows: 256,
        layers: if layers >= 88 { None } else { Some(layers) },
        mtp: false,
    };
    let mut m = NemotronModel::load_wide(&dir, &cfg, &plan, opts, widest.clamp(64, 256)).unwrap_or_else(|e| panic!("load: {e}"));
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    eprintln!("decode step (window + N commits + flush), graphs on, {layers} layers:");
    eprintln!("   N | plain ms | tok/s | fused ms | tok/s | window ms (fused) | commits ms (fused)");
    for (k, &n) in rows_list.iter().enumerate() {
        let lens = prefill_all(&mut m, n);
        let mut t = [Vec::new(), Vec::new()];
        let (mut win_ms, mut com_ms) = (Vec::new(), Vec::new());
        let mut next: Vec<u32> = (0..n).map(|s| tokens(7000 + s as u64, 1)[0]).collect();
        let steps = 8usize;
        for (mode, fused) in [false, true].into_iter().enumerate() {
            let lens = if mode == 1 { prefill_all(&mut m, n) } else { lens.clone() };
            for step in 0..steps {
                if profile && k + 1 == rows_list.len() && fused && step == 3 {
                    gb_cuda::rt::profiler_start();
                }
                let reqs: Vec<(usize, usize, Vec<u32>)> = (0..n).map(|s| (s, lens[s] + step, vec![next[s]])).collect();
                let b = batch(&reqs);
                let t0 = std::time::Instant::now();
                let c = if fused { m.window_candidates_fused(&b, 8).unwrap() } else { m.window_candidates(&b, 8).unwrap() };
                let t1 = std::time::Instant::now();
                for s in 0..n {
                    next[s] = pick(&c, s);
                    m.commit(s, 1);
                }
                m.flush().unwrap();
                let t2 = std::time::Instant::now();
                if profile && k + 1 == rows_list.len() && fused && step == 5 {
                    gb_cuda::rt::profiler_stop();
                }
                if step >= 2 {
                    t[mode].push((t2 - t0).as_secs_f64() * 1e3);
                    if fused {
                        win_ms.push((t1 - t0).as_secs_f64() * 1e3);
                        com_ms.push((t2 - t1).as_secs_f64() * 1e3);
                    }
                }
            }
        }
        let (p, f) = (median(t[0].clone()), median(t[1].clone()));
        eprintln!(
            "{n:>4} | {p:>8.1} | {:>5.1} | {f:>8.1} | {:>5.1} | {:>17.1} | {:>18.1}",
            n as f64 / p * 1e3,
            n as f64 / f * 1e3,
            median(win_ms),
            median(com_ms)
        );
    }
}
