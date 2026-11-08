//! M5 (LONGCTX): the one position rule on the REAL 88-layer Nemotron 3.5 Super VL, with the MTP head, at 12 thousand tokens, and what
//! decode costs at 128k, 512k and 1M.
//!
//! The rule (`gb_cuda::nemotron_attn_prefill::attn_long`): an attention row at absolute position below `FLASH_FROM_POS` (8,192) is the
//! exact reduction tree's, a row at or above it the flash tensor-core kernel's, **whichever entry point carries it**: decode, verify
//! window or prefill slice, in the trunk and in the MTP head. So decode == verify == prefill at every position.
//!
//! Env: `REXL3_NEMO_OMNI_DIR` (the pack), optional `REXL3_NEMO_ORACLE_DIR` (the fork's dumps: `ids_0.u32` is real text, repeated to
//! make the prompt; random tokens otherwise), `REXL3_LONGCTX_TOKENS` (default 12,333), `REXL3_LONGCTX_WINDOWS` (64-row windows of the
//! windows == prefill gate, default 24), `REXL3_LONGCTX_DEPTHS` (the decode-cost test, default `131072,524288,1048448`). One model per
//! process, `--test-threads=1 --ignored`, under the GPU lock.
//!
//! `the_one_rule_...` (gates, in order):
//! 1. **chunked == whole across the switch**: the prompt through the 64-row grid (A), a 37-row grid (B: the threshold row 8,192 falls
//!    inside a slice), a plan with 1-, 2-, 3- and 7-row slices and cuts at 8,191, 8,192 and 8,193 (C), the 256-row grid (D: 8,192 is a
//!    slice edge) and 256-row slices offset by 100 (E: a slice that holds rows on both sides) leave the same trunk state (Mamba S and
//!    conv of the 40 layers, K and V of the 8 attention layers), the same head cache and the same last-row candidates, bit for bit.
//! 2. **windows == prefill above 8,192** (the property the position-bounded rule of the first version gave up): the same prompt through
//!    prefill to 8,128 and then 24 64-row verify windows with commits has the K and V of every attention layer at every position, the
//!    candidates at every slice end and, after the windows, the Mamba state, bit-equal to a prefill of the same tokens.
//! 3. **windows == serial** at depth 12,333 for K = 1..4 and over the threshold, from position 8,190 (rows on both sides in one window).
//! 4. **drafted == serial with the head**: at depth 12,333 and crossing the threshold from a prompt of 8,190 tokens, speculative
//!    decoding at depth 4 gives the serial tokens and candidates.
//! 5. **the cost of a 5-row window** at 12k against a short context.

mod nemotron_omni_mtp_support;

use gb_cuda::nemotron_attn_prefill::FLASH_FROM_POS;
use gb_engine::RowRequest;
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::ModelOptions;
use gb_model::families::nemotron_h_omni::contract::{self, PrefillContract};
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::model::NemotronModel;
use gb_model::families::nemotron_h_omni::mtp::DraftAsk;
use gb_model::families::nemotron_h_omni::weights::LayerPlan;
use gb_model::rows::RowBatch;
use nemotron_omni_mtp_support::*;
use std::time::Instant;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// A deterministic token stream inside the vocabulary (xorshift64*).
fn random_tokens(seed: u64, n: usize) -> Vec<u32> {
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

/// `n` tokens: the oracle's real text repeated, each repetition with its first 24 tokens replaced by random ones (so it is not one
/// exact repeat), or random tokens when no oracle text is given.
fn long_prompt(n: usize) -> Vec<u32> {
    match oracle_ids() {
        Some(text) if text.len() >= 256 => {
            let mut out = Vec::with_capacity(n);
            let mut rep = 0u64;
            while out.len() < n {
                let mut block = text.clone();
                let noise = random_tokens(900 + rep, 24);
                block[..24].copy_from_slice(&noise);
                out.extend_from_slice(&block);
                rep += 1;
            }
            out.truncate(n);
            out
        }
        _ => random_tokens(4242, n),
    }
}

/// Slice ends on the absolute grid of `chunk` rows, then the prompt's end.
fn grid_ends(n: usize, chunk: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (1..).map(|i| i * chunk).take_while(|&e| e < n).collect();
    v.push(n);
    v
}

/// Slices of `chunk` rows after a first slice of `first` rows: the grid is offset, so a slice straddles the threshold.
fn offset_ends(n: usize, chunk: usize, first: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..).map(|i| first + i * chunk).take_while(|&e| e < n).collect();
    v.push(n);
    v
}

/// A plan with slices of 1, 2, 3 and 7 rows and cuts at 8,191, 8,192 and 8,193 and at other edges of the path, 64-row grid elsewhere.
fn edge_ends(n: usize) -> Vec<usize> {
    let t = FLASH_FROM_POS;
    let mut v: Vec<usize> = Vec::new();
    let mut at = 0;
    let special = [t - 70, t - 7, t - 5, t - 2, t - 1, t, t + 1, t + 2, t + 9, t + 63, t + 64];
    for &s in &special {
        while at + 64 <= s {
            at += 64;
            v.push(at);
        }
        if s > at {
            v.push(s);
            at = s;
        }
    }
    while at < n {
        at = (at + 64).min(n);
        v.push(at);
    }
    v.dedup();
    assert_eq!(*v.last().unwrap(), n);
    v
}

/// A prefill of `prompt` into `slot` through slices that end at `ends`. Returns the greedy token after it, its candidates, and the
/// candidates of the last row of every slice (`(end, candidates)`).
fn prefill_ends(m: &mut NemotronModel, slot: usize, prompt: &[u32], ends: &[usize]) -> (u32, RowCands, Vec<(usize, RowCands)>) {
    m.end_slot(slot);
    m.begin_slot(slot).unwrap();
    let mut at = 0;
    let mut per = Vec::new();
    for &end in ends {
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: at as u64, tokens: prompt[at..end].to_vec() }]);
        let c = m.prefill_chunk(&b, Some(b.rows - 1)).unwrap().expect("the slice's last row's candidates");
        per.push((
            end,
            RowCands { ids: c.ids[..CMP_CANDS].to_vec(), val_bits: c.vals[..CMP_CANDS].iter().map(|v| v.to_bits()).collect() },
        ));
        at = end;
    }
    let (_, last) = per.last().unwrap().clone();
    (last.ids[0] as u32, last, per)
}

/// The head's cache of `slot` for positions `0..n`.
fn head_rows(m: &mut NemotronModel, slot: usize, n: usize) -> (Vec<u16>, Vec<u16>) {
    m.mtp_kv_rows(slot, 0, n).unwrap()
}

/// Serial greedy decode of `n_new` tokens after the state a prefill left in `slot` (`first` is the token after the prompt). Returns the
/// tokens (first included) and the candidates that produced each.
fn serial_after(
    m: &mut NemotronModel,
    slot: usize,
    prompt_len: usize,
    first: u32,
    c0: RowCands,
    n_new: usize,
) -> (Vec<u32>, Vec<RowCands>) {
    let mut toks = vec![first];
    let mut cands = vec![c0];
    while toks.len() < n_new {
        let last = *toks.last().unwrap();
        let mut rc = window_cands(m, slot, &[last], prompt_len + toks.len() - 1);
        m.commit(slot, 1);
        let c = rc.remove(0);
        toks.push(c.ids[0] as u32);
        cands.push(c);
    }
    m.flush().unwrap();
    (toks, cands)
}

fn load(max_context: usize, kv_pages: usize, slots: usize, prefill_rows: usize) -> Option<NemotronModel> {
    let dir = pack_dir()?;
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let opts = ModelOptions { slots, kv_pages, max_context, prefill_rows, layers: None, mtp: true };
    let t = Instant::now();
    match NemotronModel::load(&dir, &cfg, &plan, opts) {
        Ok(m) => {
            eprintln!("model loaded in {:.1} s", t.elapsed().as_secs_f64());
            Some(m)
        }
        Err(ModelError::NotBuilt(what)) => {
            eprintln!("not run: {what} is not built yet");
            None
        }
        Err(e) => panic!("load: {e}"),
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore]
fn the_one_rule_is_chunk_window_and_serial_invariant_across_the_switch_and_exact_against_the_head() {
    let n = env_usize("REXL3_LONGCTX_TOKENS", 12_333);
    assert!(n > FLASH_FROM_POS + 200, "the prompt must cross the threshold with room");
    let slots = 7;
    let pages = (n + 256).div_ceil(256) + 2;
    let Some(mut m) = load((n + 128).next_multiple_of(512), slots * pages, slots, 256) else { return };
    let m = &mut m;
    let prompt = long_prompt(n);
    let t_all = Instant::now();

    // ---- 1. chunked == whole across the switch
    let t = Instant::now();
    let (first_a, ca, per_a) = prefill_ends(m, 0, &prompt, &grid_ends(n, 64));
    eprintln!("A: 64-row grid: {:.1} s, {} slices", t.elapsed().as_secs_f64(), per_a.len());
    let plans: Vec<(&str, Vec<usize>)> = vec![
        ("B: 37-row grid (a slice holds position 8,192)", grid_ends(n, 37)),
        ("C: edge plan (1-, 2-, 3- and 7-row slices, cuts at 8,190 to 8,194)", edge_ends(n)),
        ("D: 256-row grid (8,192 is a slice edge)", grid_ends(n, 256)),
        ("E: 256-row slices after a first slice of 100 (one slice holds rows on both sides)", offset_ends(n, 256, 100)),
    ];
    let (mut first_c, mut cc) = (first_a, ca.clone());
    for (name, ends) in &plans {
        let t = Instant::now();
        let (f, c, _) = prefill_ends(m, 1, &prompt, ends);
        eprintln!("{name}: {} slices, {:.1} s", ends.len(), t.elapsed().as_secs_f64());
        assert_eq!((first_a, &ca), (f, &c), "{name}: last-row candidates against the 64-row grid");
        same_trunk_state(m, 0, 1, n).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(head_rows(m, 0, n - 1) == head_rows(m, 1, n - 1), "{name}: the head's cache");
        eprintln!("A == {}: trunk state, head cache and candidates bit for bit", &name[..1]);
        first_c = f;
        cc = c;
    }

    // ---- 2. windows == prefill above the threshold: slot 2 runs 64-row verify windows (with commits) over [8,128, ...), slot 3 a prefill.
    // Under the prefill contract P2 (the default since c69fb643) a prefilled row and a decoded row do not have the same bits
    // (`docs/NEMOTRON_PREFILL_SPEED.md` 6.14: `windows == prefill` no longer holds), so there the windows run is held to the WINDOW
    // path instead: slot 3 walks the same positions in 32-row windows (with commits), and every row's candidates, K and V of all 8
    // layers and the Mamba state must equal the 64-row windows' (the window path's row and width independence above the threshold).
    // Under P1 (`REXL3_NEMO_PREFILL_CONTRACT=p1`) this section is the one it always was.
    let p2 = contract::current() == PrefillContract::P2;
    let t = Instant::now();
    let w = env_usize("REXL3_LONGCTX_WINDOWS", 24);
    let start = FLASH_FROM_POS - 64;
    let ctrl_end = start + (((start + 64 * w).min(n) - start) / 64) * 64;
    let _ = prefill_ends(m, 2, &prompt[..start], &grid_ends(start, 64));
    let row_cands = |c: &gb_model::forward::Candidates, r: usize| RowCands {
        ids: c.ids[r * c.count..r * c.count + CMP_CANDS].to_vec(),
        val_bits: c.vals[r * c.count..r * c.count + CMP_CANDS].iter().map(|v| v.to_bits()).collect(),
    };
    let mut at = start;
    let mut diverged = Vec::new();
    let mut rows64 = Vec::new();
    while at < ctrl_end {
        let b =
            RowBatch::from_requests(&[RowRequest { slot: 2, first_position: at as u64, tokens: prompt[at..at + 64].to_vec() }]);
        let c = m.window_candidates(&b, CMP_CANDS).unwrap();
        m.commit(2, 64);
        if p2 {
            rows64.extend((0..64).map(|r| row_cands(&c, r)));
        } else {
            let got = row_cands(&c, 63);
            let want = &per_a.iter().find(|(e, _)| *e == at + 64).expect("A has a slice ending here").1;
            if &got != want {
                diverged.push(at + 64);
            }
        }
        at += 64;
    }
    m.flush().unwrap();
    if p2 {
        let _ = prefill_ends(m, 3, &prompt[..start], &grid_ends(start, 64));
        let mut at = start;
        while at < ctrl_end {
            let b = RowBatch::from_requests(&[RowRequest {
                slot: 3,
                first_position: at as u64,
                tokens: prompt[at..at + 32].to_vec(),
            }]);
            let c = m.window_candidates(&b, CMP_CANDS).unwrap();
            m.commit(3, 32);
            for r in 0..32 {
                if row_cands(&c, r) != rows64[at - start + r] {
                    diverged.push(at + r);
                }
            }
            at += 32;
        }
        m.flush().unwrap();
        assert!(
            diverged.is_empty(),
            "64-row vs 32-row windows (P2): the candidates differ at the rows of positions {diverged:?}"
        );
    } else {
        assert!(diverged.is_empty(), "windows vs prefill: the candidates differ at the slice ends {diverged:?}");
    }
    // K and V: the 64-row windows (slot 2) against the prefill (slot 0, plan A) under P1, against the 32-row windows (slot 3) under P2
    let other = if p2 { 3 } else { 0 };
    for ord in 0..8 {
        let (ka, va) = m.debug_kv_rows(ord, other, 0, ctrl_end).unwrap();
        let (kb, vb) = m.debug_kv_rows(ord, 2, 0, ctrl_end).unwrap();
        let row = 256;
        let first = (0..ctrl_end).find(|&p| {
            ka[p * row..(p + 1) * row] != kb[p * row..(p + 1) * row] || va[p * row..(p + 1) * row] != vb[p * row..(p + 1) * row]
        });
        assert_eq!(
            first, None,
            "attention layer {ord}: K and V of the windows run differ from slot {other}'s from position {first:?}"
        );
    }
    if p2 {
        same_trunk_state(m, 2, 3, ctrl_end).unwrap_or_else(|e| panic!("64-row vs 32-row windows (P2): {e}"));
        eprintln!(
            "windows (P2): {} 64-row windows == {} 32-row windows over [{start}, {ctrl_end}) (rows {FLASH_FROM_POS}..{ctrl_end} are \
             flash): K and V of all 8 layers at every position, the candidates of every row and the Mamba state, bit for bit ({:.1} s)",
            (ctrl_end - start) / 64,
            (ctrl_end - start) / 32,
            t.elapsed().as_secs_f64()
        );
    } else {
        let _ = prefill_ends(m, 3, &prompt[..ctrl_end], &grid_ends(ctrl_end, 64));
        same_trunk_state(m, 2, 3, ctrl_end).unwrap_or_else(|e| panic!("windows vs prefill: {e}"));
        eprintln!(
            "windows == prefill: {} 64-row windows over [{start}, {ctrl_end}) (rows {FLASH_FROM_POS}..{ctrl_end} are flash): K and V of all 8 \
             layers at every position, the candidates at every slice end and the Mamba state, bit for bit ({:.1} s)",
            (ctrl_end - start) / 64,
            t.elapsed().as_secs_f64()
        );
    }

    // ---- 3a. serial decode from the last plan's state, then windows K = 1..4 from A (untouched, equal to it), then serial decode from A
    let (toks_c, cands_c) = serial_after(m, 1, n, first_c, cc.clone(), 12);
    eprintln!("serial decode (12 tokens) from E: {:?}", toks_c);
    for k in 1..=4usize {
        let w = window_cands(m, 0, &toks_c[..k], n);
        for (i, row) in w.iter().enumerate() {
            // row i of the window predicts what serial step i predicted next: the candidates of row i are `cands_c[i + 1]`
            assert_eq!(row, &cands_c[i + 1], "window of {k} rows, row {i}, vs the serial step");
        }
    }
    eprintln!("windows of 1..4 rows == serial steps at position {n}: bit for bit");
    let (toks_a, cands_a) = serial_after(m, 0, n, first_a, ca.clone(), 8);
    assert_eq!(toks_a[..], toks_c[..8], "8 decoded tokens: A vs E");
    assert_eq!(cands_a[..], cands_c[..8], "8 decoded tokens' candidates: A vs E");
    eprintln!("8 decoded tokens from A == from E, candidates bit for bit");

    // ---- 3b + 4b. the threshold in decode: serial greedy from a prompt of 8,190 tokens (slot 5) is the reference; the same state
    // (slot 4) is walked with windows of 1..4 rows over positions 8,190 .. 8,203 before each serial step; the head's drafted run (slot 3)
    // crosses the threshold in its windows
    let np = FLASH_FROM_POS - 2;
    let t = Instant::now();
    let (f5, c5, _) = prefill_ends(m, 5, &prompt[..np], &grid_ends(np, 256));
    let walk = 18usize;
    let (toks_g, cands_g) = serial_after(m, 5, np, f5, c5.clone(), walk + 1);
    eprintln!("serial greedy from a prompt of {np} tokens, {} tokens: {:.1} s", toks_g.len(), t.elapsed().as_secs_f64());
    let (f4, c4, _) = prefill_ends(m, 4, &prompt[..np], &grid_ends(np, 256));
    assert_eq!((f4, &c4), (f5, &c5), "two prefills of the same prompt");
    for d in 0..(walk - 4) {
        for k in 1..=4usize {
            let rows = window_cands(m, 4, &toks_g[d..d + k], np + d);
            for (i, row) in rows.iter().enumerate() {
                assert_eq!(
                    row,
                    &cands_g[d + i + 1],
                    "depth {}: window of {k} rows, row {i} (position {}) vs the serial step",
                    np + d,
                    np + d + i
                );
            }
        }
        let step = window_cands(m, 4, &toks_g[d..d + 1], np + d);
        m.commit(4, 1);
        assert_eq!(step[0], cands_g[d + 1], "depth {}: the walk's serial step vs the reference", np + d);
    }
    m.flush().unwrap();
    eprintln!(
        "windows of 1..4 rows == serial steps at every depth from {np} to {} (rows on both sides of {FLASH_FROM_POS} in one window): bit for bit",
        np + walk - 5
    );
    let t = Instant::now();
    let (g, run) = spec_ex(m, 3, &prompt[..np], 12, 4);
    eprintln!(
        "drafted run from {np} tokens (depth 4): {:.1} s, {} rounds, {} drafted, {} accepted, {} undrafted rounds",
        t.elapsed().as_secs_f64(),
        run.rounds,
        run.drafted,
        run.accepted,
        run.undrafted
    );
    assert_eq!(&g.tokens[np..], &toks_g[..12], "drafted == serial across the threshold: tokens");
    assert_eq!(g.cands, cands_g[..12], "drafted == serial across the threshold: candidates");
    let (g, run) = spec_ex(m, 3, &prompt, 12, 4);
    eprintln!(
        "drafted run from {n} tokens (depth 4): {} rounds, {} drafted, {} accepted, {} undrafted rounds",
        run.rounds, run.drafted, run.accepted, run.undrafted
    );
    assert_eq!(&g.tokens[n..], &toks_c[..], "drafted == serial at depth {n}: tokens");
    assert_eq!(g.cands, cands_c, "drafted == serial at depth {n}: candidates");
    eprintln!("drafted == serial with the head at depth 4: tokens and candidates bit for bit, across the threshold and at {n}");

    // ---- 5. what a 5-row window costs at 12k, against a short context (windows only read the state: repeatable)
    let (_, _, _) = prefill_ends(m, 6, &prompt[..256], &[64, 128, 192, 256]);
    let time_window = |m: &mut NemotronModel, slot: usize, depth: usize, rows: usize| -> f64 {
        let toks: Vec<u32> = prompt[..rows].to_vec();
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: depth as u64, tokens: toks }]);
        let mut ms = Vec::new();
        for i in 0..7 {
            let t = Instant::now();
            m.window_candidates(&b, 28).unwrap();
            if i >= 2 {
                ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        median(&mut ms)
    };
    eprintln!("verify window, host wall time (median of 5; graphs on):");
    eprintln!("  rows | depth 256 | depth {n}");
    let deep_len = m.slot_len(0).expect("slot 0 is live");
    for rows in [1usize, 5] {
        let (shallow, deep) = (time_window(m, 6, 256, rows), time_window(m, 0, deep_len, rows));
        eprintln!("  {rows:>4} | {shallow:>7.1} ms | {deep:>7.1} ms");
    }
    eprintln!("total {:.0} s", t_all.elapsed().as_secs_f64());
}

/// What decode costs at depth on the real model: the serial step (window of one row and its commit), the 5-row verify window of a
/// 4-token draft and a whole speculative round with the head (draft call, window, commit), at 128k, 512k and 1M tokens. Nobody
/// prefilled those contexts (a 1M prefill is an hour): the slot is put at the depth by `debug_seed_depth` (pages allocated, position
/// set, K and V zeros), so the attention reads `depth` keys of real size and computes on zeros; every other cost is the model's.
#[test]
#[ignore]
fn decode_cost_at_depth() {
    let depths: Vec<usize> = std::env::var("REXL3_LONGCTX_DEPTHS")
        .ok()
        .map(|v| v.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![131_072, 524_288, 1_048_448]);
    let deepest = *depths.iter().max().unwrap();
    // the model's own limit is 1,048,576 tokens: the deepest default depth leaves room for a window and a few dozen steps
    let max_context = (deepest + 256).next_multiple_of(512).min(1 << 20);
    let pages = max_context / 256 + 4;
    let Some(mut m) = load(max_context, pages, 1, 64) else { return };
    let m = &mut m;
    // plan == allocation at the context: the model's scratch (MoE + the attention scratch's buffers, summed from the device allocations)
    // is what `memory.rs` prices for a window's rows
    {
        use gb_cuda::nemotron_moe::MoeScratch;
        use gb_model::families::nemotron_h_omni::memory::NemotronMemory;
        let pb = m.pool_bytes();
        let planned_attn = NemotronMemory::attn_partial_bytes(64, max_context as u64);
        let moe = pb.scratch - planned_attn;
        eprintln!(
            "allocated at {max_context} tokens: state {:.0} MiB, kv {:.2} GiB, records {:.0} MiB, step buffers {:.0} MiB, scratch {:.1} MiB (attention {:.1} MiB planned, MoE {:.1} MiB)",
            pb.state as f64 / 1048576.0,
            pb.kv as f64 / (1u64 << 30) as f64,
            pb.records as f64 / 1048576.0,
            pb.bufs as f64 / 1048576.0,
            pb.scratch as f64 / 1048576.0,
            planned_attn as f64 / 1048576.0,
            moe as f64 / 1048576.0
        );
        assert_eq!(
            pb.scratch,
            MoeScratch::bytes(64) + planned_attn,
            "the scratch the model allocated is not the MoE's plus the plan's attention term {planned_attn} B"
        );
    }
    let reps = env_usize("REXL3_LONGCTX_REPS", 5);
    eprintln!("depth   | serial step ms (tok/s) | 5-row window ms | draft(4) + window(5) round ms | head draft ms");
    for &depth in &depths {
        m.end_slot(0);
        m.begin_slot(0).unwrap();
        m.debug_seed_depth(0, depth, true).unwrap();
        let tok = |i: usize| 1000 + (i % 7000) as u32;
        let mut pos = depth;
        let timings = |m: &mut NemotronModel, rows: usize, commit: usize, warm: usize, pos: &mut usize| -> f64 {
            let mut ms = Vec::new();
            for i in 0..(warm + reps) {
                let toks: Vec<u32> = (0..rows).map(|j| tok(*pos + j)).collect();
                let b = RowBatch::from_requests(&[RowRequest { slot: 0, first_position: *pos as u64, tokens: toks }]);
                let t = Instant::now();
                m.window_candidates(&b, 28).unwrap();
                m.commit(0, commit);
                m.flush().unwrap();
                if i >= warm {
                    ms.push(t.elapsed().as_secs_f64() * 1e3);
                }
                *pos += commit;
            }
            median(&mut ms)
        };
        let serial = timings(m, 1, 1, 3, &mut pos);
        let window5 = {
            // a window only reads the state: no commit, the same position each time
            let toks: Vec<u32> = (0..5).map(|j| tok(pos + j)).collect();
            let b = RowBatch::from_requests(&[RowRequest { slot: 0, first_position: pos as u64, tokens: toks }]);
            let mut ms = Vec::new();
            for i in 0..(3 + reps) {
                let t = Instant::now();
                m.window_candidates(&b, 28).unwrap();
                if i >= 3 {
                    ms.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            median(&mut ms)
        };
        // a speculative round with the head: the slot was seeded as a prompt of `depth` tokens through the hook, so the first ask has a
        // carry to absorb; its serial steps above disabled nothing (the head's book follows commits)
        m.end_slot(0);
        m.begin_slot(0).unwrap();
        m.debug_seed_depth(0, depth, true).unwrap();
        let mut tokens: Vec<u32> = (0..depth).map(tok).collect();
        tokens.push(tok(depth));
        let (mut round_ms, mut draft_ms) = (Vec::new(), Vec::new());
        for i in 0..(2 + reps) {
            let t = Instant::now();
            let d = m.mtp_draft(&[DraftAsk { slot: 0, tokens: &tokens, depth: 4 }], &[]).unwrap().remove(0);
            let td = t.elapsed().as_secs_f64() * 1e3;
            let drafts = d.map(|d| d.ids).unwrap_or_default();
            let mut window = vec![*tokens.last().unwrap()];
            window.extend_from_slice(&drafts);
            let b = RowBatch::from_requests(&[RowRequest {
                slot: 0,
                first_position: (tokens.len() - 1) as u64,
                tokens: window.clone(),
            }]);
            let c = m.window_candidates(&b, 28).unwrap();
            let acc = drafts.iter().enumerate().take_while(|(j, d)| c.ids[j * c.count] as u32 == **d).count();
            m.commit(0, acc + 1);
            m.flush().unwrap();
            for j in 0..=acc {
                tokens.push(c.ids[j * c.count] as u32);
            }
            if i >= 2 {
                round_ms.push(t.elapsed().as_secs_f64() * 1e3);
                draft_ms.push(td);
            }
        }
        let (round, draft) = (median(&mut round_ms), median(&mut draft_ms));
        eprintln!("{depth:>7} | {serial:>12.1} ({:>5.1}) | {window5:>14.1} | {round:>26.1} | {draft:>8.1}", 1000.0 / serial);
    }
}
