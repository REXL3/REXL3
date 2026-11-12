//! Regression gate for the automatic stateless MTP prefill skip: `mtp_after_prefill` always runs
//! `mtp_prefill_body` for supported prefill calls, which writes only the persistent MTP KV and QSA
//! indexer state (skipping the attention output / MoE / collapse / LM head the draft chain re-derives
//! from trunk rows). The full `mtp_body` remains in use for unsupported mixer/mode.
//!
//! These tests compare the stateless prefill against serial greedy decoding on the real pack. They fail
//! if the state-restoration path is incomplete: dropping the MTP KV write, the QSA indexer pool/ring
//! update, or the pre-mix/fc projections makes the draft chains diverge from serial greedy decoding, and
//! makes a restored prefix differ from a cold continuation.
//!
//! Run on an idle GPU with:
//!   REXL3_FN_REAL=1 REXL3_FLASH_NEXT_DIR=~/models/flashnext-exl3-3.05bpw \
//!   REXL3_FLASH_NEXT_REF=~/rexl3-oracle/ref305 \
//!   cargo test --release -p gb-model --test flash_next_stateless_mtp -- --nocapture
use gb_engine::{Backend, RowRequest};
use gb_model::families::qwen_flash_next::backend::FlashNextBackend;
use gb_model::families::qwen_flash_next::model::{FlashNextModel, ModelOptions, MTP_HEAD_N};
use gb_model::families::qwen_flash_next::prefix;
use gb_model::forward::{Candidates, PrefillPath};
use gb_model::rows::RowBatch;
use gb_sched::prefix::CacheBudget;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn real_dir() -> Option<PathBuf> {
    if std::env::var("REXL3_FN_REAL").ok().as_deref() != Some("1") {
        eprintln!("not run: REXL3_FN_REAL=1 is the opt-in for the real-pack device tests");
        return None;
    }
    let dir = std::env::var("REXL3_FLASH_NEXT_DIR").ok().map(PathBuf::from)?;
    if !dir.join("config.json").exists() {
        eprintln!("not run: {} holds no config.json", dir.display());
        return None;
    }
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    Some(dir)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn real_opts() -> ModelOptions {
    ModelOptions {
        slots: env_usize("REXL3_FN_SLOTS", 1),
        kv_pages: env_usize("REXL3_FN_PAGES", 64),
        max_context: env_usize("REXL3_FN_CTX", 16_384),
        prefill_rows: env_usize("REXL3_FN_PREFILL_ROWS", 1024),
        warm: false,
    }
}

fn synthetic_prompt(n: usize, vocab: usize) -> Vec<u32> {
    (0..n).map(|i| ((i as u64 * 2_654_435_761) % (vocab as u64 - 16) + 8) as u32).collect()
}

fn chain(slot: usize, first: usize, tokens: &[u32]) -> RowBatch {
    RowBatch::from_requests(&[RowRequest { slot, first_position: first as u64, tokens: tokens.to_vec() }])
}

fn top1(c: &Candidates, row: usize) -> u32 {
    c.ids[row * c.count] as u32
}

/// Prefill with the MTP head after every chunk, asserting the head's progress chunk by chunk.
fn prefill_grid_mtp(m: &mut FlashNextModel, slot: usize, prompt: &[u32], grid: usize, count: usize) -> Candidates {
    let path = PrefillPath::for_prompt(prompt.len());
    let mut done = 0usize;
    let mut last = None;
    while done < prompt.len() {
        let n = grid.min(prompt.len() - done);
        let chunk = &prompt[done..done + n];
        let want = if done + n == prompt.len() { Some(count) } else { None };
        last = m.prefill_chunk(&chain(slot, done, chunk), path, want).unwrap();
        let next = prompt.get(done + n).copied();
        m.mtp_after_prefill(slot, done, chunk, next, path).unwrap();
        let through = done + n - usize::from(next.is_none());
        if through > 0 {
            assert_eq!(m.mtp_done.get(&slot).copied(), Some(through), "head progress after chunk at {done}");
        }
        done += n;
    }
    last.expect("the last chunk was asked for candidates")
}

fn prefill_no_mtp(m: &mut FlashNextModel, slot: usize, prompt: &[u32], grid: usize, count: usize) -> Candidates {
    let path = PrefillPath::for_prompt(prompt.len());
    let mut done = 0usize;
    let mut last = None;
    while done < prompt.len() {
        let n = grid.min(prompt.len() - done);
        let want = if done + n == prompt.len() { Some(count) } else { None };
        last = m.prefill_chunk(&chain(slot, done, &prompt[done..done + n]), path, want).unwrap();
        done += n;
    }
    last.expect("the last chunk was asked for candidates")
}

struct SpecRun {
    prefill: Candidates,
    tokens: Vec<u32>,
    drafts: Vec<Vec<u32>>,
    proposed: usize,
    accepted: usize,
}

/// Greedy speculative decoding from a fresh slot, exactly as the trunk verifies it.
fn spec_decode(m: &mut FlashNextModel, prompt: &[u32], grid: usize, k: usize, steps: usize) -> SpecRun {
    m.begin_slot(0).unwrap();
    let first = prefill_grid_mtp(m, 0, prompt, grid, 1);
    let first_token = top1(&first, 0);
    let mut history = prompt.to_vec();
    history.push(first_token);
    let mut run = SpecRun { prefill: first, tokens: vec![first_token], drafts: Vec::new(), proposed: 0, accepted: 0 };
    let limit = MTP_HEAD_N.min(m.vocab_size()) as u32;
    while run.tokens.len() < steps {
        let pos = history.len() - 1;
        let d = m.mtp_propose(0, &history, k).unwrap();
        assert!(!d.is_empty() && d.len() <= k, "{} drafts for k = {k} at position {pos}", d.len());
        assert_eq!(m.mtp_done.get(&0).copied(), Some(pos), "the head caught up to the last token");
        assert!(d.iter().all(|&t| t < limit), "a draft outside the head's drafting range: {d:?}");
        let mut rows = vec![history[pos]];
        rows.extend_from_slice(&d);
        let c = m.window_candidates(&chain(0, pos, &rows), 1).unwrap();
        let a = (0..d.len()).take_while(|&j| top1(&c, j) == d[j]).count();
        m.commit(0, a + 1);
        history.extend_from_slice(&d[..a]);
        history.push(top1(&c, a));
        run.tokens.extend_from_slice(&d[..a]);
        run.tokens.push(top1(&c, a));
        run.proposed += d.len();
        run.accepted += a;
        run.drafts.push(d);
    }
    m.flush().unwrap();
    run
}

/// Serial greedy decoding with no MTP head at all.
fn serial_greedy(m: &mut FlashNextModel, prompt: &[u32], grid: usize, len: usize) -> Vec<u32> {
    m.begin_slot(0).unwrap();
    let first = prefill_no_mtp(m, 0, prompt, grid, 1);
    let mut out = vec![top1(&first, 0)];
    while out.len() < len {
        let j = out.len() - 1;
        let c = m.window_candidates(&chain(0, prompt.len() + j, &[out[j]]), 1).unwrap();
        m.commit(0, 1);
        out.push(top1(&c, 0));
    }
    m.flush().unwrap();
    out
}

fn prompt_for(m: &FlashNextModel, total: usize) -> (Vec<u32>, bool) {
    match std::env::var("REXL3_FN_ORACLE_DIR").ok() {
        Some(o) => {
            let raw = std::fs::read(Path::new(&o).join("ids_0.u32")).unwrap();
            let ids: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            (ids[..total.min(ids.len())].to_vec(), true)
        }
        None => (synthetic_prompt(total, m.vocab_size()), false),
    }
}

/// The stateless prefill must agree with serial greedy identity, reproduce from a fresh slot, and
/// produce drafts the trunk verifies (non-empty first-round drafts prove the skip did not disable MTP).
#[test]
fn stateless_prefill_matches_serial_and_is_reproducible_from_fresh_slot() {
    let _guard = ENV_LOCK.lock().unwrap();
    let Some(dir) = real_dir() else { return };
    let total = env_usize("REXL3_FN_PROMPT", 1031);
    let k = env_usize("REXL3_FN_DRAFTS", 4).clamp(1, 16);
    let steps = env_usize("REXL3_FN_MTP_STEPS", 64).max(2);
    let grid = real_opts().prefill_rows.min(total);

    let mut m = FlashNextModel::load(&dir, real_opts()).unwrap();
    if !m.has_mtp() {
        eprintln!("not run: this pack has no MTP head");
        return;
    }
    let (prompt, real_text) = prompt_for(&m, total);
    let total = prompt.len();
    assert!(total + steps + k < m.opts.max_context, "prompt and decode do not fit the context");
    let a = spec_decode(&mut m, &prompt, grid, k, steps);
    let serial = serial_greedy(&mut m, &prompt, grid, a.tokens.len());
    assert_eq!(a.tokens, serial, "speculative decoding left the trunk's greedy path");
    assert!(!a.drafts.is_empty() && !a.drafts[0].is_empty(), "no first-round drafts with the stateless prefill");

    // A second run from a fresh slot must draft the same chains: nothing may depend on state the
    // first run mutated (prefix pages, recycled MTP rows, stale head streams).
    let c = spec_decode(&mut m, &prompt, grid, k, steps);
    assert_eq!(c.drafts, a.drafts, "stateless prefill is not reproducible from a fresh slot");
    assert_eq!(c.tokens, a.tokens);

    let rate = a.accepted as f64 / a.proposed as f64;
    let min =
        std::env::var("REXL3_FN_MTP_MIN_ACCEPT").ok().and_then(|v| v.parse().ok()).unwrap_or(if real_text { 0.2 } else { 0.0 });
    eprintln!(
        "stateless MTP: {} tokens in {} verify windows, {}/{} drafts accepted ({rate:.3}), k = {k}, {} prompt",
        a.tokens.len(),
        a.drafts.len(),
        a.accepted,
        a.proposed,
        if real_text { "real-text" } else { "synthetic" }
    );
    assert!(rate >= min, "draft acceptance {rate:.3} below {min}");
}

// ------------------------------------------------------- prefix cache / continuation

fn prefix_model(dir: &Path) -> FlashNextModel {
    FlashNextModel::load(dir, ModelOptions { slots: 2, kv_pages: 64, max_context: 4096, prefill_rows: 1024, warm: false })
        .unwrap()
}

fn hash_range(h: &mut Sha256, buf: &gb_cuda::DevBuf, at: usize, n: usize) {
    let mut data = vec![0u8; n];
    buf.download_range(at, &mut data).unwrap();
    h.update(data);
}

fn qsa_includes_mtp_one_ahead(mtp_done: bool, pool_index: usize, pool_count: usize) -> bool {
    mtp_done && pool_index + 1 == pool_count
}

fn hash_qsa_entries(
    h: &mut Sha256,
    len: usize,
    include_mtp_one_ahead: bool,
    blocks: &[u32],
    group_size: usize,
    mut hash_entry: impl FnMut(usize, usize, &mut Sha256),
) {
    let groups = len / group_size + usize::from(include_mtp_one_ahead);
    for group in 0..groups {
        let page = blocks[group / 64] as usize;
        hash_entry(page, group, h);
    }
}

/// Hash only rows a prefill actually wrote; unused bytes in a physical page need not match.
fn digest(m: &mut FlashNextModel, slot: usize, len: usize) -> Vec<u8> {
    m.flush().unwrap();
    m.stream.sync().unwrap();
    let mut h = Sha256::new();
    let opts = m.opts;
    for (ss, cc) in m.pools.s.iter().zip(&m.pools.conv) {
        hash_range(&mut h, ss, slot * ss.bytes() / opts.slots, ss.bytes() / opts.slots);
        hash_range(&mut h, cc, slot * cc.bytes() / opts.slots, cc.bytes() / opts.slots);
    }
    hash_range(&mut h, &m.ple_ring, slot * m.ple_ring.bytes() / opts.slots, m.ple_ring.bytes() / opts.slots);
    for ring in &m.qsa_ring {
        hash_range(&mut h, ring, slot * ring.bytes() / opts.slots, ring.bytes() / opts.slots);
    }
    let previous = m.ple_tok.history(slot, len, &[], m.cfg.eos_token_id);
    for t in previous {
        h.update(t.to_le_bytes());
    }
    h.update([m.mtp_done.get(&slot).is_some_and(|&x| x == len) as u8]);
    let blocks = m.kv_tab.blocks_of(slot);
    for cache in m.kv.iter().chain(m.mtp_kv.iter()) {
        let n = len
            + usize::from(m.mtp_kv.as_ref().is_some_and(|mtp| std::ptr::eq(cache, mtp) && m.mtp_done.get(&slot) == Some(&len)));
        for p in 0..n {
            let at = (blocks[p / 256] as usize * 256 + p % 256) * cache.kv_heads.token_elems() * 2;
            let bytes = cache.kv_heads.token_elems() * 2;
            hash_range(&mut h, &cache.k, at, bytes);
            hash_range(&mut h, &cache.v, at, bytes);
        }
    }
    let mtp_done = m.mtp_done.get(&slot).copied() == Some(len);
    for (pool_index, pool) in m.qsa_pool.iter().enumerate() {
        hash_qsa_entries(
            &mut h,
            len,
            qsa_includes_mtp_one_ahead(mtp_done, pool_index, m.qsa_pool.len()),
            m.kv_tab.blocks_of(slot),
            gb_cuda::qsa::CR,
            |page, group, h| {
                hash_range(h, pool, (page * 64 + group % 64) * 256, 256);
            },
        );
    }
    h.finalize().to_vec()
}

fn ref_ids() -> Option<Vec<u32>> {
    let refdir = std::env::var_os("REXL3_FLASH_NEXT_REF").map(PathBuf::from)?;
    let raw = std::fs::read(refdir.join("ids_0.u32")).ok()?;
    let ids: Vec<u32> = raw.chunks_exact(4).map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect();
    (ids.len() > 2300).then_some(ids)
}

/// The stateless prefill is automatic, so a restored prefix plus its continuation must equal a cold
/// prefill bit for bit, including the MTP KV/QSA state; repeating the same prompt from a released slot
/// must also match.
#[test]
fn stateless_prefill_prefix_restore_and_repeat_match_cold() {
    let _guard = ENV_LOCK.lock().unwrap();
    let Some(dir) = real_dir() else { return };
    let Some(all) = ref_ids() else {
        eprintln!("not run: REXL3_FLASH_NEXT_REF with ids_0.u32 (>2300 tokens) is required");
        return;
    };
    gb_cuda::rt::init_context().unwrap();
    let m = prefix_model(&dir);
    if !m.has_mtp() {
        eprintln!("not run: this pack has no MTP head");
        return;
    }
    let budget = CacheBudget {
        snapshot_slots: 2,
        page_blocks: 16,
        snapshot_bytes: prefix::snapshot_bytes(&m.cfg, m.mtp_kv.is_some()),
        page_bytes: prefix::page_bytes(&m.cfg, m.mtp_kv.is_some()),
    };
    let mut b = FlashNextBackend::new(m, 257, 32).with_prefix_cache(budget, "flash-next-stateless-mtp-prefix".into()).unwrap();

    for (cut, total) in [(512usize, 2300usize), (513, 2300)] {
        let prompt = &all[..total];
        let mut done = 0;
        let handle = loop {
            let step = b.prefill_slice_cuts(0, prompt, &[], done, 32, &[cut]);
            assert!(step.step.row.is_none(), "cut prefix must not produce final logits");
            done = step.step.done;
            if let Some(captured) = step.captured.into_iter().next() {
                assert_eq!(captured.len, cut);
                break captured.handle;
            }
            assert!(done < prompt.len(), "expected a snapshot at cut {cut}");
        };
        b.release(0);

        // Cold reference is a real full prefill from position 0. (Previously this reused `done`, which had
        // advanced to the cut, so slot 1 continued from the cut without ever prefilling the prefix — a
        // garbage reference that made the restored warm arm look like it diverged when it did not.)
        done = 0;

        let cold_row = loop {
            let step = b.prefill_slice(1, prompt, &[], done, 32);
            done = step.done;
            if let Some(row) = step.row {
                break row;
            }
        };
        assert!(b.take_error().is_none(), "cold backend prefill failed at cut {cut}");
        let cold_state = digest(&mut b.m, 1, total);

        b.prefix_restore(0, &handle, cut).unwrap();
        let mut warm_done = cut;
        let warm_row = loop {
            let step = b.prefill_slice(0, prompt, &[], warm_done, 32);
            warm_done = step.done;
            if let Some(row) = step.row {
                break row;
            }
        };
        assert!(b.take_error().is_none(), "restored backend prefill failed at cut {cut}");
        assert_eq!(warm_row.ids, cold_row.ids, "candidate ids after cut {cut}");
        assert!(
            warm_row.values.iter().zip(&cold_row.values).all(|(x, y)| x.to_bits() == y.to_bits()),
            "candidate logit bits after cut {cut}"
        );
        assert_eq!(digest(&mut b.m, 0, total), cold_state, "PLE/QSA/KV/MTP state after cut {cut}");
        b.release(0);

        // Repeated prompt from the released cold slot: same bits again.
        let mut again_done = 0;
        let again_row = loop {
            let step = b.prefill_slice(1, prompt, &[], again_done, 32);
            again_done = step.done;
            if let Some(row) = step.row {
                break row;
            }
        };
        assert_eq!(again_row.ids, cold_row.ids, "repeated prompt ids at cut {cut}");
        assert!(
            again_row.values.iter().zip(&cold_row.values).all(|(x, y)| x.to_bits() == y.to_bits()),
            "repeated prompt logit bits at cut {cut}"
        );
        assert_eq!(digest(&mut b.m, 1, total), cold_state, "repeated prompt state at cut {cut}");
        b.release(1);
        b.prefix_release(handle);
        assert_eq!(b.prefix_free_snapshots(), 2);
    }
}

/// Prefill-only workload for the GB10 nsys proof: the trace must not contain the MTP head's CoopMK
/// MoE calls, because the stateless prefill skip is now automatic. No decode window is run, so every
/// CoopMK instance in the trace is attributable to prefill. Run explicitly, e.g.:
///   REXL3_FN_REAL=1 REXL3_FLASH_NEXT_DIR=... REXL3_FN_PROMPT=8192 REXL3_FN_PREFILL_ROWS=2048 \
///   cargo test --release -p gb-model --test flash_next_stateless_mtp prefill_only_for_nsys -- \
///     --ignored --exact --test-threads=1 --nocapture
#[test]
#[ignore]
fn prefill_only_for_nsys() {
    let _guard = ENV_LOCK.lock().unwrap();
    let Some(dir) = real_dir() else { return };
    let total = env_usize("REXL3_FN_PROMPT", 8192);
    let opts = real_opts();
    let mut m = FlashNextModel::load(&dir, opts).unwrap();
    if !m.has_mtp() {
        eprintln!("not run: this pack has no MTP head");
        return;
    }
    let (prompt, _) = prompt_for(&m, total);
    let grid = opts.prefill_rows.min(prompt.len());
    let c = prefill_grid_mtp(&mut m, 0, &prompt, grid, 1);
    m.flush().unwrap();
    eprintln!("prefill_only_for_nsys: {} tokens, grid {grid}, stateless=1, last top-1 = {}", prompt.len(), top1(&c, 0));
}
