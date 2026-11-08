//! M5 (M5G): the prefix cache's GPU side on the real Nemotron 3.5 Super VL. A snapshot of a slot after N tokens (the Mamba S and conv of every
//! layer, the head's carry, a reference to the KV pages), a restore of it into another slot that another prompt dirtied, and the rest of the
//! prompt prefilled in another slice grid must leave EXACTLY what a cold prefill of the whole prompt leaves: S and conv of every Mamba layer,
//! the KV rows of every attention layer and of the head, the head's carry and book, the last row's candidates bit for bit, the head's drafts
//! and the committed tokens and candidates of a drafted decode with its counters.
//!
//! Env (all optional but the first): `REXL3_NEMO_OMNI_DIR` (the pack), `REXL3_NEMO_ORACLE_DIR` (real text for the prompts), `REXL3_NEMO_PREFIX_LAYERS`
//! (`8`: the layer-limited model, no head; default: all 88 with the head), `REXL3_NEMO_PREFIX_CUTS` / `REXL3_NEMO_PREFIX_FLASH_CUTS` (comma lists
//! that replace the cut sets). `#[ignore]`, one model per process, `--test-threads=1`, under the GPU lock. The model is built for 256-row
//! prefill slices (the shipped default), so a 256-row grid takes the large-row MoE prefill and a 64-row grid the decode kernels.
//!
//! * `resume_equals_cold_at_every_cut`: cuts at every residue mod 4 and 8, the page boundary and its neighbours, 511, 1,023 to 1,025 and 2,047 to
//!   2,051; the donor stops at the cut (a slice ends there), takes the snapshot and goes; a stranger's prompt dirties the slot that is restored
//!   into; the restore is compared at the cut with the donor's own state, the resumed prompt (another grid) at the end with a cold prompt.
//! * `resume_equals_cold_across_the_flash_switch`: cuts at 8,190 to 8,194 of an 8,400-token prompt, one live donor taking five snapshots.
//! * `the_donor_goes_on_and_a_sharer_comes_later`: copy-on-write both ways, partial and aligned cuts, a sharer with another suffix.
//! * `a_restore_of_the_wrong_snapshot_is_refused_and_leaks_nothing`, `a_handle_of_another_model_is_refused`, `the_arena_fills_skips_and_recovers`,
//!   `allocations_equal_the_plan_with_the_cache`.

mod nemotron_omni_mtp_support;

use gb_engine::{PrefixHandle, RowRequest};
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::ModelOptions;
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::memory::NemotronMemory;
use gb_model::families::nemotron_h_omni::model::NemotronModel;
use gb_model::families::nemotron_h_omni::mtp::DraftAsk;
use gb_model::families::nemotron_h_omni::prefix;
use gb_model::families::nemotron_h_omni::snapshot::{PrefixDigest, SnapArena, SnapMutant};
use gb_model::families::nemotron_h_omni::weights::LayerPlan;
use gb_model::forward::Candidates;
use gb_model::rows::RowBatch;
use nemotron_omni_mtp_support::{config, oracle_ids, pack_dir, window_cands, RowCands, CMP_CANDS};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

const SLOTS: usize = 4;
const SNAPSHOTS: usize = 6;
const MAX_CONTEXT: usize = 8_704;
const KV_PAGES: usize = 360;
/// Prompt length of the short cuts and of the flash cuts.
const N_SHORT: usize = 2_600;
const N_FLASH: usize = 8_400;
/// Tokens a finished prompt decodes (drafted, depth 3) before the runs are compared.
const DECODE_TOKENS: usize = 14;
const DEPTH: usize = 3;

/// The cuts: every residue mod 8 (100..107), the page boundary and its neighbours, 511, 1,023 to 1,025, 2,047 to 2,051.
const CUTS: [usize; 22] =
    [100, 101, 102, 103, 104, 105, 106, 107, 255, 256, 257, 258, 259, 511, 1023, 1024, 1025, 2047, 2048, 2049, 2050, 2051];
/// The cuts across the flash switch (position 8,192): the last row below it, the first at it, and the ones around.
const FLASH_CUTS: [usize; 5] = [8190, 8191, 8192, 8193, 8194];

struct Shared(NemotronModel);
// SAFETY: the model is only ever used under the mutex below, one thread at a time.
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

fn layers() -> Option<usize> {
    match std::env::var("REXL3_NEMO_PREFIX_LAYERS").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(n) if n < 88 => Some(n),
        _ => None,
    }
}

fn load_model(layers: Option<usize>) -> Option<NemotronModel> {
    let dir = pack_dir()?;
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let opts = ModelOptions {
        slots: SLOTS,
        kv_pages: KV_PAGES,
        max_context: MAX_CONTEXT,
        // 512 by default since 2026-10-15 (256 under the opt-out REXL3_NEMO_PREFILL_ROWS_MAX=256; lane S)
        prefill_rows: gb_model::families::nemotron_h_omni::model::prefill_rows_cap(),
        layers,
        mtp: layers.is_none(),
    };
    let t = Instant::now();
    match NemotronModel::load(&dir, &cfg, &plan, opts) {
        Ok(m) => {
            eprintln!("model loaded in {:.1} s ({} layers)", t.elapsed().as_secs_f64(), layers.unwrap_or(88));
            Some(m)
        }
        Err(ModelError::NotBuilt(what)) => {
            eprintln!("not run: {what} is not built yet");
            None
        }
        Err(e) => panic!("load: {e}"),
    }
}

/// The shared model, with its snapshot arena: loaded once per process.
fn model() -> Option<MutexGuard<'static, Shared>> {
    static M: OnceLock<Mutex<Shared>> = OnceLock::new();
    if M.get().is_none() {
        let mut m = load_model(layers())?;
        m.enable_snapshots(SNAPSHOTS).unwrap_or_else(|e| panic!("enable_snapshots: {e}"));
        let _ = M.set(Mutex::new(Shared(m)));
    }
    Some(M.get().unwrap().lock().unwrap_or_else(|p| p.into_inner()))
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

/// `n` tokens of the oracle's real text repeated, each repetition with its first 24 tokens replaced by random ones (a drafter finds
/// something to propose in real text); random tokens when no oracle text is given.
fn prompt(seed: u64, n: usize) -> Vec<u32> {
    match oracle_ids() {
        Some(text) if text.len() >= 256 => {
            let mut out = Vec::with_capacity(n);
            let mut rep = 0u64;
            while out.len() < n {
                let mut block = text.clone();
                let noise = random_tokens(seed * 1000 + rep, 24);
                block[..24].copy_from_slice(&noise);
                out.extend_from_slice(&block);
                rep += 1;
            }
            out.truncate(n);
            out
        }
        _ => random_tokens(seed, n),
    }
}

/// Prefill `toks[from..to]` into `slot` in slices that end on the absolute grid of `grid` rows (and at `to`). With `want`, the last slice asks
/// for the candidates of its last row, which are returned.
fn prefill(
    m: &mut NemotronModel,
    slot: usize,
    toks: &[u32],
    from: usize,
    to: usize,
    grid: usize,
    want: bool,
) -> Option<Candidates> {
    let (mut at, mut last) = (from, None);
    while at < to {
        let end = (at + grid - at % grid).min(to);
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: at as u64, tokens: toks[at..end].to_vec() }]);
        let w = (want && end == to).then_some(b.rows - 1);
        if let Some(c) = m.prefill_chunk(&b, w).unwrap_or_else(|e| panic!("prefill of slot {slot} at {at}..{end}: {e}")) {
            last = Some(c);
        }
        at = end;
    }
    last
}

/// What a finished prompt must show.
#[derive(Debug, PartialEq)]
struct Final {
    digest: PrefixDigest,
    top_ids: Vec<i32>,
    top_bits: Vec<u32>,
    /// The head's first draft call after the prompt (ids, value bits), `None` for a model without the head.
    drafts: Option<(Vec<u32>, Vec<u32>)>,
    tokens: Vec<u32>,
    cands: Vec<RowCands>,
    /// `(rounds, drafted, accepted)` of the drafted decode.
    counters: (usize, usize, usize),
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The state, the last row's candidates, the head's drafts and a drafted decode of `DECODE_TOKENS` tokens from the prompt `toks` that `slot`
/// holds (its last row's candidates are `c`). The slot ends up decoded: the caller ends it.
fn finish(m: &mut NemotronModel, slot: usize, toks: &[u32], c: &Candidates) -> Final {
    let n = toks.len();
    let digest = m.prefix_digest(slot, n).unwrap();
    let first = c.ids[0] as u32;
    let mut seq = toks.to_vec();
    seq.push(first);
    let mut tokens = vec![first];
    let mut cands = vec![RowCands { ids: c.ids[..CMP_CANDS].to_vec(), val_bits: bits(&c.vals[..CMP_CANDS]) }];
    let (mut rounds, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
    let mut first_drafts = None;
    while tokens.len() < DECODE_TOKENS {
        let left = DECODE_TOKENS - tokens.len();
        let max = DEPTH.min(left - 1);
        let drafts: Vec<u32> = if max > 0 && m.mtp_loaded() {
            match m.mtp_draft(&[DraftAsk { slot, tokens: &seq, depth: max }], &[]).unwrap().remove(0) {
                Some(d) => {
                    if first_drafts.is_none() {
                        first_drafts = Some((d.ids.clone(), bits(&d.vals)));
                    }
                    d.ids
                }
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };
        let mut window = vec![*seq.last().unwrap()];
        window.extend_from_slice(&drafts);
        let rows = window_cands(m, slot, &window, seq.len() - 1);
        let samples: Vec<u32> = rows.iter().map(|r| r.ids[0] as u32).collect();
        let acc = drafts.iter().zip(&samples).take_while(|(d, s)| d == s).count();
        m.commit(slot, acc + 1);
        for r in rows.into_iter().take(acc + 1) {
            let t = r.ids[0] as u32;
            tokens.push(t);
            seq.push(t);
            cands.push(r);
        }
        rounds += 1;
        drafted += drafts.len();
        accepted += acc;
    }
    m.flush().unwrap();
    tokens.truncate(DECODE_TOKENS);
    cands.truncate(DECODE_TOKENS);
    Final {
        digest,
        top_ids: c.ids.clone(),
        top_bits: bits(&c.vals),
        drafts: first_drafts,
        tokens,
        cands,
        counters: (rounds, drafted, accepted),
    }
}

fn describe(a: &Final, b: &Final) -> String {
    let mut d = Vec::new();
    let st = a.digest.describe_diff(&b.digest);
    if !st.is_empty() {
        d.push(format!("state: {st}"));
    }
    if a.top_ids != b.top_ids || a.top_bits != b.top_bits {
        d.push("the last row's candidates".into());
    }
    if a.drafts != b.drafts {
        d.push(format!("drafts {:?} vs {:?}", a.drafts.as_ref().map(|d| &d.0), b.drafts.as_ref().map(|d| &d.0)));
    }
    if a.tokens != b.tokens {
        d.push(format!("tokens {:?} vs {:?}", a.tokens, b.tokens));
    }
    if a.cands != b.cands {
        d.push("the decoded candidates".into());
    }
    if a.counters != b.counters {
        d.push(format!("counters (rounds, drafted, accepted) {:?} vs {:?}", a.counters, b.counters));
    }
    d.join("; ")
}

/// The cold reference: the whole prompt in `slot` on the grid `grid`, finished, the slot ended.
fn cold(m: &mut NemotronModel, slot: usize, toks: &[u32], grid: usize) -> Final {
    m.end_slot(slot);
    m.begin_slot(slot).unwrap();
    let c = prefill(m, slot, toks, 0, toks.len(), grid, true).unwrap();
    let f = finish(m, slot, toks, &c);
    m.end_slot(slot);
    f
}

/// A restore target that another prompt dirtied: the stranger's state, its last tap in the head's store and its pages were there, then the
/// slot was ended and begun again (what the backend does before a restore).
fn dirty_slot(m: &mut NemotronModel, slot: usize) {
    let stranger = random_tokens(99 + slot as u64, 300);
    m.end_slot(slot);
    m.begin_slot(slot).unwrap();
    prefill(m, slot, &stranger, 0, stranger.len(), 64, true).unwrap();
    m.end_slot(slot);
    m.begin_slot(slot).unwrap();
}

/// The grids a cut's donor and its resumed prompt run in, by cut index: the decode kernels (64), the large-row MoE prefill (256 and the
/// default 512), odd ones. 512 is capped at the model's slice cap (256 under the opt-out `REXL3_NEMO_PREFILL_ROWS_MAX=256`).
fn grids(k: usize) -> (usize, usize) {
    let w = wide();
    [(64, 256), (256, 64), (48, 256), (256, 100), (w, 64), (100, w)][k % 6]
}

/// The widest slice the model was loaded for: 512 by default.
fn wide() -> usize {
    gb_model::families::nemotron_h_omni::model::prefill_rows_cap().min(512)
}

fn cuts_from_env(var: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(var) {
        Ok(v) => {
            v.split(',').map(|x| x.trim().parse().unwrap_or_else(|_| panic!("{var} is a comma list of token counts"))).collect()
        }
        Err(_) => default.to_vec(),
    }
}

/// The head's counters must not be vacuous for the comparison to mean anything: say what a cold run drafted.
fn report_cold(label: &str, f: &Final) {
    eprintln!(
        "{label}: cold run: {} tokens, {} rounds, {} drafted, {} accepted, first drafts {:?}",
        f.tokens.len(),
        f.counters.0,
        f.counters.1,
        f.counters.2,
        f.drafts.as_ref().map(|d| &d.0)
    );
}

/// One cut of `toks` with the donor stopping there: the donor (grid `gd`) takes the snapshot at `c`, a stranger dirties the slot that is
/// restored into, the restore is compared with the donor's state at the cut, and the prompt resumed on grid `gr` with `want`. Everything is
/// cleaned up whatever the verdict (slots ended, the snapshot released). `Err` names the first difference.
fn one_cut(
    m: &mut NemotronModel,
    toks: &[u32],
    want: &Final,
    c: usize,
    (gd, gr): (usize, usize),
    times: Option<(&mut Vec<f64>, &mut Vec<f64>)>,
) -> Result<(), String> {
    let n = toks.len();
    m.end_slot(1);
    m.end_slot(2);
    m.begin_slot(1).unwrap();
    assert!(prefill(m, 1, toks, 0, c, gd, false).is_none());
    let at_cut = m.prefix_digest(1, c).unwrap();
    let t_snap = Instant::now();
    let handle = m.snapshot_slot(1, c).unwrap().expect("a free snapshot slot");
    m.flush().unwrap();
    let snap_ms = t_snap.elapsed().as_secs_f64() * 1e3;
    m.end_slot(1);
    dirty_slot(m, 2);
    let t_rest = Instant::now();
    let restored = m.restore_snapshot(2, &handle, c);
    m.flush().unwrap();
    let rest_ms = t_rest.elapsed().as_secs_f64() * 1e3;
    if let Some((s, r)) = times {
        s.push(snap_ms);
        r.push(rest_ms);
    }
    let verdict = (|| -> Result<(), String> {
        restored.map_err(|e| format!("the restore failed: {e}"))?;
        let r = m.prefix_digest(2, c).unwrap();
        if r != at_cut {
            return Err(format!(
                "cut {c} (residue {} mod 8, donor grid {gd}): the restored slot is not the state the donor had at the cut: {}",
                c % 8,
                r.describe_diff(&at_cut)
            ));
        }
        let cand = prefill(m, 2, toks, c, n, gr, true).unwrap();
        let got = finish(m, 2, toks, &cand);
        if got != *want {
            return Err(format!(
                "cut {c} (residue {} mod 8, grids {gd} then {gr}): the resumed prompt differs from the cold one: {}",
                c % 8,
                describe(&got, want)
            ));
        }
        Ok(())
    })();
    m.end_slot(2);
    m.release_snapshot(handle).unwrap();
    verdict
}

/// A live donor with prompt `a` takes the snapshot at `c` and goes on past it; a sharer with prompt `b` (the same first `c` tokens, another
/// suffix) restores while the donor is alive and prefills its suffix over the page; then the donor finishes, and later sharers with each prompt
/// restore after the donor wrote above the cut in the page the entry still holds. Cleans up whatever the verdict.
fn live_cut(m: &mut NemotronModel, a: &[u32], b: &[u32], want_a: &Final, want_b: &Final, c: usize) -> Result<(), String> {
    let n = a.len();
    let last = (c - 1) / 256;
    for s in 1..=3 {
        m.end_slot(s);
    }
    m.begin_slot(1).unwrap();
    prefill(m, 1, a, 0, c, 64, false);
    let handle = m.snapshot_slot(1, c).unwrap().expect("a free snapshot slot");
    let verdict = (|| -> Result<(), String> {
        let donor_pages: Vec<u32> = m.slot_pages(1).iter().map(|p| p.0).collect();
        if handle.pages != donor_pages {
            return Err("the entry does not reference the donor's pages".into());
        }
        if !m.slot_pages(1).iter().all(|p| p.1 == 2) {
            return Err(format!("each page should be held by the donor and the entry: {:?}", m.slot_pages(1)));
        }
        // the donor goes on past the cut: it writes into the page the entry holds (trunk rows above the cut, and the head's entry for c - 1)
        prefill(m, 1, a, c, c + 100, 64, false);
        m.begin_slot(2).unwrap();
        m.restore_snapshot(2, &handle, c).map_err(|e| format!("the restore failed: {e}"))?;
        let mine = m.slot_pages(2);
        if mine.len() != last + 1 || mine[..last].iter().map(|p| p.0).collect::<Vec<_>>() != donor_pages[..last] {
            return Err(format!("the pages below the last should be shared: {mine:?} vs {donor_pages:?}"));
        }
        if !mine[..last].iter().all(|p| p.1 == 3) {
            return Err(format!("pages below the last should be held by the donor, the entry and the sharer: {mine:?}"));
        }
        if donor_pages.contains(&mine[last].0) || mine[last].1 != 1 {
            return Err(format!("the page of position c - 1 must be copied, always (c = {c}, last page {last}): {mine:?}"));
        }
        let cand = prefill(m, 2, b, c, n, 256, true).unwrap();
        let sharer = finish(m, 2, b, &cand);
        if sharer != *want_b {
            return Err(format!("cut {c}: the live sharer with another suffix: {}", describe(&sharer, want_b)));
        }
        m.end_slot(2);
        // the donor, whose pages the sharer's prefill must not have touched
        let cand = prefill(m, 1, a, c + 100, n, 256, true).unwrap();
        let donor = finish(m, 1, a, &cand);
        if donor != *want_a {
            return Err(format!("cut {c}: the donor after a sharer wrote over the same positions: {}", describe(&donor, want_a)));
        }
        m.end_slot(1);
        for (name, p, want) in [("the donor's prompt", a, want_a), ("the other suffix", b, want_b)] {
            m.begin_slot(3).unwrap();
            m.restore_snapshot(3, &handle, c).map_err(|e| format!("the restore failed: {e}"))?;
            let cand = prefill(m, 3, p, c, n, 64, true).unwrap();
            let got = finish(m, 3, p, &cand);
            m.end_slot(3);
            if got != *want {
                return Err(format!("cut {c}: a later sharer with {name}: {}", describe(&got, want)));
            }
        }
        Ok(())
    })();
    for s in 1..=3 {
        m.end_slot(s);
    }
    m.release_snapshot(handle).unwrap();
    verdict
}

#[test]
#[ignore]
fn resume_equals_cold_at_every_cut() {
    let Some(mut m) = model() else { return };
    let m = &mut **m;
    let toks = prompt(1, N_SHORT);
    let t0 = Instant::now();
    let want = cold(m, 0, &toks, 256);
    report_cold("prompt 1", &want);
    if m.mtp_loaded() {
        assert!(want.counters.1 > 0, "the head drafted nothing: the counters comparison would be vacuous");
    }
    let free_pages = m.free_pages();
    let cuts = cuts_from_env("REXL3_NEMO_PREFIX_CUTS", &CUTS);
    let (mut snap_ms, mut rest_ms): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for (k, &c) in cuts.iter().enumerate() {
        let t = Instant::now();
        let g = grids(k);
        one_cut(m, &toks, &want, c, g, Some((&mut snap_ms, &mut rest_ms))).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(m.free_pages(), free_pages, "cut {c}: every page is back once the entry and both slots are gone");
        assert_eq!(m.free_snapshots(), SNAPSHOTS, "cut {c}: every arena slot is back");
        eprintln!(
            "cut {c:5} (mod 8 = {}) donor grid {:3}, resumed grid {:3}: identical ({:.1} s)",
            c % 8,
            g.0,
            g.1,
            t.elapsed().as_secs_f64()
        );
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[0], v[v.len() / 2], v[v.len() - 1])
    };
    let (s, r) = (med(&mut snap_ms), med(&mut rest_ms));
    eprintln!(
        "{} cuts identical in {:.0} s; snapshot (enqueue + sync) min/median/max {:.2}/{:.2}/{:.2} ms, restore {:.2}/{:.2}/{:.2} ms",
        cuts.len(),
        t0.elapsed().as_secs_f64(),
        s.0,
        s.1,
        s.2,
        r.0,
        r.1,
        r.2
    );
}

#[test]
#[ignore]
fn resume_equals_cold_across_the_flash_switch() {
    let Some(mut m) = model() else { return };
    let m = &mut **m;
    let toks = prompt(2, N_FLASH);
    let cuts = cuts_from_env("REXL3_NEMO_PREFIX_FLASH_CUTS", &FLASH_CUTS);
    assert!(cuts.windows(2).all(|w| w[0] < w[1]) && cuts.len() <= SNAPSHOTS && cuts.iter().all(|&c| c < N_FLASH));
    let t0 = Instant::now();
    let want = cold(m, 0, &toks, 256);
    report_cold("prompt 2 (8,400 tokens)", &want);
    eprintln!("cold reference: {:.0} s", t0.elapsed().as_secs_f64());
    let free_pages = m.free_pages();
    // one live donor on a 64-row grid takes a snapshot at each cut (a slice ends at each) and goes on to the end
    m.begin_slot(1).unwrap();
    let mut at = 0;
    let mut held: Vec<(usize, PrefixDigest, PrefixHandle)> = Vec::new();
    for &c in &cuts {
        assert!(prefill(m, 1, &toks, at, c, 64, false).is_none());
        at = c;
        let d = m.prefix_digest(1, c).unwrap();
        let h = m.snapshot_slot(1, c).unwrap().expect("a free snapshot slot");
        held.push((c, d, h));
    }
    let cand = prefill(m, 1, &toks, at, N_FLASH, 64, true).unwrap();
    let donor = finish(m, 1, &toks, &cand);
    assert!(donor == want, "the live donor after {} snapshots: {}", cuts.len(), describe(&donor, &want));
    m.end_slot(1);
    eprintln!("donor with {} snapshots == cold ({:.0} s)", cuts.len(), t0.elapsed().as_secs_f64());
    // two cuts are also compared with a cold run that stopped exactly there, from scratch, on the other grid
    for (c, d, _) in held.iter().filter(|(c, _, _)| *c == 8191 || *c == 8192) {
        m.begin_slot(3).unwrap();
        prefill(m, 3, &toks, 0, *c, 256, false);
        let r = m.prefix_digest(3, *c).unwrap();
        assert!(r == *d, "cut {c}: the live donor's state at the cut is not a cold run stopped there: {}", r.describe_diff(d));
        m.end_slot(3);
    }
    for (k, (c, at_cut, handle)) in held.iter().enumerate() {
        let gr = [256, 64, 100, wide(), 48][k % 5];
        dirty_slot(m, 2);
        m.restore_snapshot(2, handle, *c).unwrap();
        let restored = m.prefix_digest(2, *c).unwrap();
        assert!(
            restored == *at_cut,
            "cut {c}: the restored slot is not the donor's state at the cut: {}",
            restored.describe_diff(at_cut)
        );
        let cand = prefill(m, 2, &toks, *c, N_FLASH, gr, true).unwrap();
        let got = finish(m, 2, &toks, &cand);
        assert!(
            got == want,
            "cut {c} (resumed grid {gr}): the resumed prompt differs from the cold one: {}",
            describe(&got, &want)
        );
        m.end_slot(2);
        eprintln!("cut {c}: restore == donor at the cut, resumed on grid {gr} == cold ({:.0} s)", t0.elapsed().as_secs_f64());
    }
    for (_, _, h) in held {
        m.release_snapshot(h).unwrap();
    }
    assert_eq!(m.free_pages(), free_pages);
    assert_eq!(m.free_snapshots(), SNAPSHOTS);
}

#[test]
#[ignore]
fn the_donor_goes_on_and_a_sharer_comes_later() {
    let Some(mut m) = model() else { return };
    let m = &mut **m;
    let a = prompt(3, N_SHORT);
    // a sharer's prompt: the same first `c` tokens and another suffix, so that the head's entry at position c - 1 (paired with the token AFTER
    // the cut) differs between the donor and the sharer
    let other = random_tokens(4, N_SHORT);
    let want_a = cold(m, 0, &a, 256);
    let free_pages = m.free_pages();
    let cuts = cuts_from_env("REXL3_NEMO_PREFIX_LIVE_CUTS", &[1001, 256, 512, 1024, 2048]);
    for &c in &cuts {
        let mut b = a[..c].to_vec();
        b.extend_from_slice(&other[c..]);
        assert!(a[c] != b[c]);
        let want_b = cold(m, 0, &b, 256);
        live_cut(m, &a, &b, &want_a, &want_b, c).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(m.free_pages(), free_pages, "cut {c}: the books balance");
        assert_eq!(m.free_snapshots(), SNAPSHOTS);
        eprintln!(
            "cut {c} (last page {}, {}): the donor, a live sharer with another suffix and later sharers are exact",
            (c - 1) / 256,
            if c % 256 == 0 { "aligned" } else { "partial" }
        );
    }
}

/// The gates have teeth: each deliberate fault of the restore (`SnapMutant`) is caught by the comparison that is meant to catch it, and the
/// same comparison passes with no fault. The page faults need a live donor (nobody writes into a shared page again once the donor is gone).
#[test]
#[ignore]
fn mutants_are_caught() {
    let Some(mut m) = model() else { return };
    let m = &mut **m;
    let head = m.mtp_loaded();
    let a = prompt(8, N_SHORT);
    let other = random_tokens(9, N_SHORT);
    let want_a = cold(m, 0, &a, 256);
    let free_pages = m.free_pages();
    let b_for = |c: usize| {
        let mut b = a[..c].to_vec();
        b.extend_from_slice(&other[c..]);
        b
    };
    // (fault, needs the head, cut, a live donor)
    let faults = [
        (SnapMutant::StateNotRestored, false, 1001, false),
        (SnapMutant::ConvNotRestored, false, 1001, false),
        (SnapMutant::CarryNotRestored, true, 1001, false),
        (SnapMutant::BookNotRestored, true, 1001, false),
        (SnapMutant::LastPageShared, false, 1001, true),
        (SnapMutant::FullLastPageShared, false, 512, true),
        (SnapMutant::HeadStripeNotCopied, true, 1001, true),
    ];
    // the controls first: no fault, both comparisons pass
    for (c, live) in [(1001usize, false), (1001, true), (512, true)] {
        let b = b_for(c);
        let want_b = cold(m, 0, &b, 256);
        let r = if live { live_cut(m, &a, &b, &want_a, &want_b, c) } else { one_cut(m, &a, &want_a, c, (64, 256), None) };
        r.unwrap_or_else(|e| panic!("control (cut {c}, live donor {live}) failed: {e}"));
    }
    let mut caught = 0;
    for (fault, needs_head, c, live) in faults {
        if needs_head && !head {
            eprintln!("{fault:?}: not run (the model has no head)");
            continue;
        }
        m.set_snapshot_mutant(Some(fault));
        let b = b_for(c);
        let want_b = cold(m, 0, &b, 256);
        let r = if live { live_cut(m, &a, &b, &want_a, &want_b, c) } else { one_cut(m, &a, &want_a, c, (64, 256), None) };
        m.set_snapshot_mutant(None);
        match r {
            Err(e) => {
                eprintln!("{fault:?} (cut {c}, live donor {live}): caught: {}", e.chars().take(200).collect::<String>());
                caught += 1;
            }
            Ok(()) => panic!("{fault:?} (cut {c}, live donor {live}) was NOT caught: the gate cannot fail for this fault"),
        }
    }
    assert!(caught >= 4, "at least the state, conv and page faults are caught on a model without the head");
    assert_eq!(m.free_pages(), free_pages, "the books balance after every fault");
    assert_eq!(m.free_snapshots(), SNAPSHOTS);
}

#[test]
#[ignore]
fn a_restore_of_the_wrong_snapshot_is_refused_and_leaks_nothing() {
    let Some(mut m) = model() else { return };
    let m = &mut **m;
    let toks = prompt(5, 700);
    m.begin_slot(1).unwrap();
    prefill(m, 1, &toks, 0, 700, 64, false);
    let handle = m.snapshot_slot(1, 700).unwrap().unwrap();
    m.end_slot(1);
    let (free_pages, free_snaps) = (m.free_pages(), m.free_snapshots());
    let refused = |r: Result<(), ModelError>, needle: &str| {
        let e = r.expect_err(&format!("a refusal containing `{needle}` was expected"));
        assert!(e.to_string().contains(needle), "refused, but not by name `{needle}`: {e}");
    };
    // a begun, empty slot takes every attempt: none may change anything
    m.begin_slot(2).unwrap();
    refused(m.restore_snapshot(2, &handle, 699), "length mismatch");
    refused(m.restore_snapshot(2, &handle, 701), "length mismatch");
    let mut foreign = handle.clone();
    foreign.snap ^= 0x1_0000;
    refused(m.restore_snapshot(2, &foreign, 700), "does not belong");
    refused(m.restore_snapshot(2, &PrefixHandle { snap: 0xFFFF, pages: handle.pages.clone() }, 700), "does not belong");
    let free_slot = (0..SNAPSHOTS as u32).find(|&i| i != handle.snap & 0xFFFF).unwrap();
    refused(m.restore_snapshot(2, &PrefixHandle { snap: free_slot, pages: handle.pages.clone() }, 700), "not live");
    refused(
        m.restore_snapshot(2, &PrefixHandle { snap: handle.snap, pages: handle.pages[..2].to_vec() }, 700),
        "not the snapshot",
    );
    let mut swapped = handle.pages.clone();
    swapped.reverse();
    refused(m.restore_snapshot(2, &PrefixHandle { snap: handle.snap, pages: swapped }, 700), "not the snapshot");
    // a slot that holds something, and one that was never begun
    m.begin_slot(3).unwrap();
    prefill(m, 3, &toks, 0, 10, 64, false);
    refused(m.restore_snapshot(3, &handle, 700), "begun, empty slot");
    m.end_slot(3);
    refused(m.restore_snapshot(3, &handle, 700), "never begun");
    // nothing moved, and the real handle still restores into the slot every refusal left alone
    assert_eq!((m.free_pages(), m.free_snapshots()), (free_pages, free_snaps), "refusals leak no page and no arena slot");
    assert_eq!(m.slot_len(2), Some(0));
    assert!(m.slot_pages(2).is_empty());
    m.restore_snapshot(2, &handle, 700).unwrap();
    m.end_slot(2);
    // a second release, and a stale handle after the arena slot was taken by another snapshot
    m.begin_slot(1).unwrap();
    prefill(m, 1, &toks, 0, 700, 64, false);
    let h2 = m.snapshot_slot(1, 700).unwrap().unwrap();
    m.end_slot(1);
    m.release_snapshot(handle.clone()).unwrap();
    refused(m.release_snapshot(handle.clone()), "not live");
    assert!(m.snapshot_slot(1, 0).is_err(), "a snapshot of no tokens is refused");
    m.begin_slot(1).unwrap();
    prefill(m, 1, &toks, 0, 700, 64, false);
    let h4 = m.snapshot_slot(1, 700).unwrap().unwrap();
    m.end_slot(1);
    // the slot of `handle` was freed and (the free list is a stack) is taken again by `h4`: the stale `handle` must not release `h4`'s snapshot
    assert_eq!(h4.snap & 0xFFFF, handle.snap & 0xFFFF, "the arena slot was reused");
    refused(m.release_snapshot(handle.clone()), "does not belong");
    refused(m.restore_snapshot(2, &handle, 700), "does not belong");
    // a snapshot refused for a length the slot does not hold, or taken under an open window
    m.begin_slot(1).unwrap();
    prefill(m, 1, &toks, 0, 300, 64, false);
    assert!(m.snapshot_slot(1, 299).unwrap_err().to_string().contains("length mismatch"));
    assert!(m.snapshot_slot(1, 301).unwrap_err().to_string().contains("length mismatch"));
    let b = RowBatch::from_requests(&[RowRequest { slot: 1, first_position: 300, tokens: vec![toks[300]] }]);
    m.window_candidates(&b, 4).unwrap();
    assert!(m.snapshot_slot(1, 300).unwrap_err().to_string().contains("uncommitted window"));
    m.commit(1, 1);
    m.flush().unwrap();
    m.end_slot(1);
    m.release_snapshot(h2).unwrap();
    m.release_snapshot(h4).unwrap();
    assert_eq!(m.free_snapshots(), SNAPSHOTS);
    assert_eq!(m.free_pages(), free_pages + 3, "the three pages the first entry held are the only difference");
}

#[test]
#[ignore]
fn a_handle_of_another_model_is_refused() {
    let Some(mut m) = model() else { return };
    let m = &mut **m;
    // a second, small model in the same process with its own arena
    let Some(mut other) = load_model(Some(8)) else { return };
    other.enable_snapshots(SNAPSHOTS).unwrap();
    let toks = prompt(6, 700);
    other.begin_slot(1).unwrap();
    prefill(&mut other, 1, &toks, 0, 700, 64, false);
    let foreign = other.snapshot_slot(1, 700).unwrap().unwrap();
    // fill this model's arena so the foreign handle's slot is live here, with another snapshot (and its own tag)
    m.begin_slot(1).unwrap();
    prefill(m, 1, &toks, 0, 700, 64, false);
    let free_pages = m.free_pages();
    let mut mine = Vec::new();
    while let Some(h) = m.snapshot_slot(1, 700).unwrap() {
        mine.push(h);
    }
    assert_eq!(mine.len(), SNAPSHOTS);
    m.end_slot(1);
    m.begin_slot(2).unwrap();
    let e = m.restore_snapshot(2, &foreign, 700).expect_err("another model's handle must be refused");
    assert!(e.to_string().contains("does not belong"), "refused, but not by name: {e}");
    let e = m.release_snapshot(foreign.clone()).expect_err("another model's handle must not release anything here");
    assert!(e.to_string().contains("does not belong"), "refused, but not by name: {e}");
    assert_eq!(m.free_snapshots(), 0);
    m.end_slot(2);
    for h in mine {
        m.release_snapshot(h).unwrap();
    }
    assert_eq!(m.free_snapshots(), SNAPSHOTS);
    assert!(m.free_pages() >= free_pages, "nothing leaked");
    other.release_snapshot(foreign).unwrap();
    other.end_slot(1);
}

#[test]
#[ignore]
fn the_arena_fills_skips_and_recovers() {
    let Some(mut m) = model() else { return };
    let m = &mut **m;
    let toks = prompt(7, 600);
    m.begin_slot(1).unwrap();
    prefill(m, 1, &toks, 0, 600, 64, false);
    let free_pages = m.free_pages();
    let mut hs = Vec::new();
    for i in 0..SNAPSHOTS {
        assert_eq!(m.free_snapshots(), SNAPSHOTS - i);
        hs.push(m.snapshot_slot(1, 600).unwrap().expect("a free arena slot"));
    }
    // full: the cut is skipped, not an error, and no page reference is taken
    let refs: Vec<u32> = m.slot_pages(1).iter().map(|p| p.1).collect();
    assert_eq!(refs, vec![1 + SNAPSHOTS as u32; 3]);
    assert!(m.snapshot_slot(1, 600).unwrap().is_none());
    assert_eq!(m.slot_pages(1).iter().map(|p| p.1).collect::<Vec<_>>(), refs, "a skipped cut holds no reference");
    // distinct arena slots, distinct handles
    let mut ids: Vec<u32> = hs.iter().map(|h| h.snap & 0xFFFF).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), SNAPSHOTS);
    // one comes back and the next cut is taken again
    m.release_snapshot(hs.pop().unwrap()).unwrap();
    assert_eq!(m.free_snapshots(), 1);
    hs.push(m.snapshot_slot(1, 600).unwrap().expect("the released slot"));
    for h in hs {
        m.release_snapshot(h).unwrap();
    }
    m.end_slot(1);
    assert_eq!(m.free_snapshots(), SNAPSHOTS);
    assert_eq!(m.free_pages(), free_pages + 3);
}

#[test]
#[ignore]
fn allocations_equal_the_plan_with_the_cache() {
    let Some(m) = model() else { return };
    let mtp = m.has_mtp();
    let mamba = m.core.mamba.len();
    let per = SnapArena::slot_bytes(mamba, mtp) as u64;
    // the arena's slot is the plan's snapshot slot: every Mamba layer's S and conv, and the head's carry row
    let state = mamba as u64 * (4_194_304 + 61_440);
    assert_eq!(per, prefix::snapshot_bytes(state, mtp));
    if m.core.mamba.len() == 40 {
        let dir = pack_dir().unwrap();
        let mem = NemotronMemory::new(&config(&dir), 0);
        assert_eq!(per, mem.snapshot_bytes(mtp), "the memory model's snapshot slot is the allocation's");
        assert_eq!(per, if mtp { 170_246_144 } else { 170_229_760 });
    }
    assert_eq!(m.snapshot_arena_bytes(), SNAPSHOTS as u64 * per, "the arena is exactly its slots");
    assert_eq!(m.pool_bytes().snapshots, SNAPSHOTS as u64 * per);
    assert_eq!(m.snapshot_slots(), SNAPSHOTS);
    eprintln!(
        "snapshot slot {per} B ({:.2} MiB), {SNAPSHOTS} slots {:.1} MiB, pool total {:.2} GiB",
        per as f64 / 1048576.0,
        (SNAPSHOTS as u64 * per) as f64 / 1048576.0,
        m.pool_bytes().total() as f64 / (1u64 << 30) as f64
    );
}
