//! GPU tests of the whole GLM-5.3-Flash model (45 layers, 87 GiB resident). All `#[ignore]` and env-gated: REXL3_GLM_DIR (the pack), and per
//! test REXL3_GLM_ORACLE_DIR (the oracle's dumps, anonymized host-d `/srv/oracles/glm53/final4-A (example path)`) and REXL3_GLM_DUMP_OUT (where a candidate dump goes:
//! OUTSIDE the nrun-synced tree, which `--delete`s). Without them a test prints "not run".
//!
//! One model is loaded per process (a second load in the same process would not fit and can hang the box): the tests share it through a
//! `OnceLock` and must run with `--test-threads=1`, under the GPU lock (`nrun NAME --gpu -- ...`).
//!
//! * `candidate_dump`: forced or natural routing (REXL3_GLM_MODE, default forced), prompts REXL3_GLM_PROMPTS (default: the five of the
//!   brief), written for `tools/compare_glm_dump.py ORACLE CANDIDATE --skip mtp,vision,kda_conv`. With REXL3_GLM_TRUNK_LAYERS=N it
//!   loads only the first N trunk layers (`glm_support::shared_prefix_model`: a pack whose whole model does not fit one node, as
//!   sc_4.10bpw, has only `tools/dump_glm_reference.py --trunk-layers N` oracles, whose head runs after layer N - 1 as this model's does)
//!   and stamps `"trunk_layers": N` into every prompt's meta.json; `tools/compare_glm_dump.py` judges it only against an oracle of the
//!   same prefix, with that prefix's tolerances (R-G-oracle (c)). The other tests of this file read all 45 layers: they refuse to run
//!   with REXL3_GLM_TRUNK_LAYERS set.
//! * `allocations_equal_the_plan`: every `DevBuf` the model owns against `alloc::AllocPlan`, item by item.
//! * `window_equals_serial_*`: a window of R rows equals R serial steps, bit for bit (logits, KDA state, latents, pool rows), with
//!   partial accepts, across the 2,047/2,048 boundary where the attention changes regime.

mod glm_support;

use gb_model::families::glm::alloc::{self, AllocParams, AllocPlan};
use gb_model::families::glm::dump::dump_prompt;
use gb_model::families::glm::model::{ForcedRouting, GlmModel, Head, Mode};
use gb_model::families::glm::npy;
use gb_model::families::glm::ops::Mutant;
use gb_model::rows::RowBatch;
use glm_support::*;
use std::path::{Path, PathBuf};

/// Slots of the model under test: `REXL3_GLM_SLOTS`, default 4 (a box with little free device memory after the weights runs fewer).
fn slots() -> usize {
    std::env::var("REXL3_GLM_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(4)
}
const KV_BLOCKS: usize = 80; // 20,480 tokens
const MAX_CONTEXT: usize = 8192;

fn model() -> Option<std::sync::MutexGuard<'static, GlmModel>> {
    assert!(
        trunk_layers().is_none(),
        "REXL3_GLM_TRUNK_LAYERS is set: only candidate_dump takes a prefix model (this test reads all 45 layers)"
    );
    shared_model(slots(), KV_BLOCKS, MAX_CONTEXT)
}

/// `REXL3_GLM_TRUNK_LAYERS`: the trunk layers of `candidate_dump`'s prefix model (unset: the whole model).
fn trunk_layers() -> Option<usize> {
    std::env::var("REXL3_GLM_TRUNK_LAYERS").ok().map(|v| parse_trunk_layers(&v))
}

fn parse_trunk_layers(v: &str) -> usize {
    match v.trim().parse::<usize>() {
        Ok(n) if n > 0 => n,
        _ => panic!("REXL3_GLM_TRUNK_LAYERS={v:?}: a prefix is a positive number of trunk layers"),
    }
}

/// Record in a dumped prompt's meta.json how many trunk layers made it (`prefix`: a prefix model, not the whole one), the key
/// `tools/compare_glm_dump.py` matches against the oracle's `trunk_layers`.
fn stamp_trunk_layers(meta: &Path, layers: usize, prefix: bool) {
    let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(meta).unwrap()).unwrap();
    let o = v.as_object_mut().expect("meta.json holds an object");
    o.insert("trunk_layers".into(), serde_json::json!(layers));
    o.insert("trunk_prefix".into(), serde_json::json!(prefix));
    std::fs::write(meta, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
}

#[test]
fn a_prefix_setting_is_a_positive_layer_count_and_the_stamp_names_it() {
    assert_eq!(parse_trunk_layers("8"), 8);
    assert_eq!(parse_trunk_layers(" 24\n"), 24);
    for bad in ["0", "-1", "eight", ""] {
        assert!(std::panic::catch_unwind(|| parse_trunk_layers(bad)).is_err(), "{bad:?} must be refused");
    }
    let d = std::env::temp_dir().join(format!("rexl3-glm-stamp-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let meta = d.join("meta.json");
    std::fs::write(&meta, br#"{"name": "short_20", "T": 21, "routing_mode": "forced"}"#).unwrap();
    stamp_trunk_layers(&meta, 8, true);
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&meta).unwrap()).unwrap();
    assert_eq!((v["trunk_layers"].as_u64(), v["trunk_prefix"].as_bool()), (Some(8), Some(true)));
    assert_eq!((v["name"].as_str(), v["T"].as_u64(), v["routing_mode"].as_str()), (Some("short_20"), Some(21), Some("forced")));
    std::fs::remove_dir_all(&d).unwrap();
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

fn forced_of(oracle: &Path, prompt: &str) -> ForcedRouting {
    let d = oracle.join(prompt);
    let ids = npy::read(&d.join("route_ids.npy")).unwrap();
    let w = npy::read(&d.join("route_w.npy")).unwrap();
    let lay = npy::read(&d.join("route_layers.npy")).unwrap().to_i32().unwrap();
    ForcedRouting {
        layers: lay.iter().map(|&l| l as usize).collect(),
        t: ids.shape[1],
        ids: ids.to_i32().unwrap(),
        w: w.to_f32().unwrap().iter().map(|&x| gb_exl3::f16::f64_to_f16(x as f64)).collect(),
    }
}

#[test]
#[ignore]
fn candidate_dump() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(out) = std::env::var("REXL3_GLM_DUMP_OUT").ok().map(PathBuf::from) else {
        return eprintln!("REXL3_GLM_DUMP_OUT is not set: not run");
    };
    let prefix = trunk_layers();
    let m = match prefix {
        Some(n) => shared_prefix_model(n, slots(), KV_BLOCKS, MAX_CONTEXT),
        None => model(),
    };
    let Some(mut m) = m else { return };
    let forced = std::env::var("REXL3_GLM_MODE").map_or(true, |v| v != "natural");
    let prompts = std::env::var("REXL3_GLM_PROMPTS").unwrap_or_else(|_| "one_token,short_20,chat_300,code_700,long_2300".into());
    for p in prompts.split(',') {
        let ids = ids_of(&oracle, p);
        m.forced = forced.then(|| forced_of(&oracle, p));
        let d = dump_prompt(&mut m, p, "text", &ids, &out).unwrap();
        stamp_trunk_layers(&out.join(p).join("meta.json"), m.layers(), prefix.is_some());
        eprintln!(
            "{p}: T={} greedy next token {} ({:.1} s, {}, {} trunk layers)",
            d.tokens,
            d.greedy_next_token,
            d.seconds,
            if forced { "forced" } else { "natural" },
            m.layers()
        );
    }
    m.forced = None;
}

/// The vision tower beside the whole model, loaded once (1.05 GiB of weights and 1.04 GiB of scratch for images of 8,000 tokens).
fn tower() -> Option<std::sync::MutexGuard<'static, gb_model::families::glm::vision::GlmVisionTower>> {
    use gb_model::families::glm::vision::{GlmVisionTower, TowerOptions};
    use std::sync::{Mutex, OnceLock};
    static T: OnceLock<Option<Mutex<GlmVisionTower>>> = OnceLock::new();
    let t = T.get_or_init(|| {
        let dir = pack_dir()?;
        let pack = gb_model::families::glm::GlmFamily::map_pack(&dir).unwrap();
        let t = GlmVisionTower::load(&pack, &dir, &TowerOptions::new(32_000, 8 * GIB)).unwrap();
        eprintln!(
            "vision tower beside the model: {:.2} GiB weights, {:.2} GiB scratch; MemAvailable {:.1} GiB",
            gib(t.weight_bytes()),
            gib(t.scratch_bytes()),
            gib(mem_available())
        );
        Some(Mutex::new(t))
    });
    t.as_ref().map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
}

/// The oracle's image prompts through the whole model with the image rows spliced: REXL3_GLM_PROMPTS (default `img_small,img_cap`), forced or natural
/// routing (REXL3_GLM_MODE), and the features either from the tower fed the oracle's patches (REXL3_GLM_FEATURES=tower, the default) or the oracle's
/// own features (`oracle`: the language model's splice alone, no tower error). Written for `tools/compare_glm_dump.py ORACLE CANDIDATE --skip mtp,kda_conv`.
#[test]
#[ignore]
fn candidate_dump_images() {
    use gb_cuda::{DevBuf, Stream};
    use gb_model::families::glm::dump::{dump_prompt_mm, DumpImage};
    use std::sync::Arc;
    let Some(oracle) = oracle_dir() else { return };
    let Some(out) = std::env::var("REXL3_GLM_DUMP_OUT").ok().map(PathBuf::from) else {
        return eprintln!("REXL3_GLM_DUMP_OUT is not set: not run");
    };
    let Some(mut m) = model() else { return };
    let Some(mut t) = tower() else { return };
    let forced = std::env::var("REXL3_GLM_MODE").map_or(true, |v| v != "natural");
    let use_tower = std::env::var("REXL3_GLM_FEATURES").map_or(true, |v| v != "oracle");
    let prompts = std::env::var("REXL3_GLM_PROMPTS").unwrap_or_else(|_| "img_small,img_cap".into());
    for p in prompts.split(',') {
        let ids = ids_of(&oracle, p);
        let vd = oracle.join(p).join("vision");
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(vd.join("meta.json")).unwrap()).unwrap();
        let start = meta["image_start_position"].as_u64().unwrap() as usize;
        let rows = meta["image_rows"].as_u64().unwrap() as usize;
        let g = meta["grid_thw"].as_array().unwrap();
        let (gh, gw) = (g[1].as_u64().unwrap() as usize, g[2].as_u64().unwrap() as usize);
        assert_eq!(gh * gw / 4, rows);
        let st = Stream::new().unwrap();
        let mut buf = DevBuf::zeroed(rows * 4096 * 4).unwrap();
        let t0 = std::time::Instant::now();
        if use_tower {
            let patches = npy::read(&vd.join("patches.npy")).unwrap().to_u16().unwrap();
            let pre =
                gb_vision::Preprocessed { size: (gw * 14, gh * 14), grid: (1, gh, gw), patches, patch_dim: 1176, pixel_hash: 0 };
            let still = gb_vision::glm::still_patches(&pre, 2);
            t.encode(&still, (gh, gw), &mut buf, &st).unwrap();
            st.sync().unwrap();
        } else {
            let f = npy::read(&vd.join("features.npy")).unwrap().to_f32().unwrap();
            buf.upload(&f).unwrap();
        }
        let enc_s = t0.elapsed().as_secs_f64();
        let feats = Arc::new(gb_model::families::glm::vision_serve::GlmImageFeatures {
            buf,
            tokens: rows,
            merged: (gh / 2, gw / 2),
            key: 0,
            permit: None,
        });
        m.forced = forced.then(|| forced_of(&oracle, p));
        let d = dump_prompt_mm(&mut m, p, "image", &ids, &[DumpImage { start, feats }], &out).unwrap();
        eprintln!(
            "{p}: T={} greedy next token {} ({:.1} s, {}, features from the {}, encode {:.2} s)",
            d.tokens,
            d.greedy_next_token,
            d.seconds,
            if forced { "forced" } else { "natural" },
            if use_tower { "tower" } else { "oracle" },
            enc_s
        );
    }
    m.forced = None;
}

#[test]
#[ignore]
fn allocations_equal_the_plan() {
    let Some(m) = model() else { return };
    let plan = AllocPlan::new_chunked(
        &m.cfg,
        &AllocParams {
            slots: slots(),
            kv_blocks: KV_BLOCKS,
            max_context: MAX_CONTEXT,
            max_rows: alloc::MAX_ROWS,
            window_rows: alloc::WINDOW_ROWS,
            mtp: m.has_mtp(),
        },
        m.chunk_rows(),
    )
    .unwrap();
    m.check_allocations(&plan).unwrap();
    let total: u64 = m.allocations().iter().map(|a| a.1).sum();
    assert_eq!(total, plan.total());
    eprintln!(
        "the model's {} allocations ({:.2} GiB beyond the weights) equal the plan item by item",
        m.allocations().len(),
        gib(total)
    );
    for (n, b) in m.allocations() {
        eprintln!("  {n:20} {:>14} B", b);
    }
}

/// Greedy generation of N tokens (REXL3_GLM_GREEDY_N, default 384) from prompt token ids read from REXL3_GLM_GREEDY_IN (a JSON object name ->
/// ids, `tools/glm_greedy_check.py prep`), written to REXL3_GLM_GREEDY_OUT (name -> generated ids, the first one included), stopping on the model's
/// end-of-turn ids. `tools/glm_greedy_check.py compare` reports the matching prefix against the oracle's own greedy completions.
#[test]
#[ignore]
fn greedy_generation() {
    let (Ok(inp), Ok(out)) = (std::env::var("REXL3_GLM_GREEDY_IN"), std::env::var("REXL3_GLM_GREEDY_OUT")) else {
        return eprintln!("REXL3_GLM_GREEDY_IN / REXL3_GLM_GREEDY_OUT are not set: not run");
    };
    let n: usize = std::env::var("REXL3_GLM_GREEDY_N").ok().and_then(|v| v.parse().ok()).unwrap_or(384);
    let Some(mut m) = model() else { return };
    let prompts: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(&inp).unwrap()).unwrap();
    let eos: Vec<u32> = m.cfg.eos.clone();
    let mut result = serde_json::Map::new();
    let mut margins = serde_json::Map::new();
    for (name, ids) in &prompts {
        let ids: Vec<u32> = ids.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let t0 = std::time::Instant::now();
        m.begin_slot(0).unwrap();
        let mut at = 0;
        let mut first = None;
        while at < ids.len() {
            let end = (at + m.chunk_rows()).min(ids.len());
            let want = (end == ids.len()).then_some(2usize);
            first = m.prefill_chunk(0, at, &ids[at..end], want).unwrap();
            at = end;
        }
        let t_prefill = t0.elapsed();
        let first = first.unwrap();
        let mut toks = vec![first.ids[0] as u32];
        // the logit gap between the chosen token and the runner-up at every step: a divergence from the oracle at a tiny gap is a near tie
        let mut gaps = vec![first.vals[0] - first.vals[1]];
        let t1 = std::time::Instant::now();
        while toks.len() < n && !eos.contains(toks.last().unwrap()) {
            let pos = ids.len() + toks.len() - 1;
            let b = RowBatch::from_requests(&[gb_engine::RowRequest {
                slot: 0,
                tokens: vec![*toks.last().unwrap()],
                first_position: pos as u64,
            }]);
            let c = m.window_candidates(&b, 2).unwrap();
            m.commit(0, 1);
            toks.push(c.ids[0] as u32);
            gaps.push(c.vals[0] - c.vals[1]);
        }
        let dt = t1.elapsed().as_secs_f64();
        eprintln!(
            "{name}: prompt {} tokens prefilled in {:.2?} ({:.0} tok/s), {} tokens generated at {:.1} tok/s",
            ids.len(),
            t_prefill,
            ids.len() as f64 / t_prefill.as_secs_f64(),
            toks.len(),
            (toks.len() - 1) as f64 / dt
        );
        result.insert(name.clone(), serde_json::json!(toks));
        margins.insert(name.clone(), serde_json::json!(gaps));
    }
    std::fs::write(&out, serde_json::to_vec(&result).unwrap()).unwrap();
    std::fs::write(format!("{out}.gaps"), serde_json::to_vec(&margins).unwrap()).unwrap();
}

// ----------------------------------------------------------------------------------------------------------------- window == serial

/// What a run leaves: the logits bits of every row it stepped and the committed state.
#[derive(PartialEq, Debug)]
struct Trace {
    logits: Vec<Vec<u16>>,
    kda: Vec<(Vec<u32>, Vec<u16>)>,
    latent: Vec<Vec<u16>>,
    pools: Vec<Vec<u16>>,
}

type StateParts = (Vec<(Vec<u32>, Vec<u16>)>, Vec<Vec<u16>>, Vec<Vec<u16>>);

fn state(m: &mut GlmModel, slot: usize, committed: usize) -> StateParts {
    let mut kda = Vec::new();
    let mut latent = Vec::new();
    let mut pools = Vec::new();
    for l in 0..45 {
        if l % 4 == 3 {
            latent.push(m.latent_rows(l, slot, committed).unwrap());
            pools.push(m.pool_rows(l, slot, committed / 4).unwrap());
        } else {
            let (s, c) = m.kda_state(l, slot).unwrap();
            kda.push((s.iter().map(|x| x.to_bits()).collect(), c));
        }
    }
    (kda, latent, pools)
}

fn prefill(m: &mut GlmModel, slot: usize, ids: &[u32]) {
    m.begin_slot(slot).unwrap();
    let mut at = 0;
    while at < ids.len() {
        let end = (at + m.chunk_rows()).min(ids.len());
        m.prefill_chunk(slot, at, &ids[at..end], None).unwrap();
        at = end;
    }
}

fn window(m: &mut GlmModel, slot: usize, start: usize, toks: &[u32]) -> Vec<Vec<u16>> {
    let b = RowBatch::from_requests(&[gb_engine::RowRequest { slot, tokens: toks.to_vec(), first_position: start as u64 }]);
    m.forward(&b, Mode::Window, Head::AllRows).unwrap();
    let all = m.logits_host(0, toks.len()).unwrap();
    all.chunks(m.cfg.vocab).map(|c| c.to_vec()).collect()
}

/// Serial: R one-row windows, each committed; the trace after `keep` of them.
fn serial(m: &mut GlmModel, ctx: &[u32], toks: &[u32], keep: usize) -> Trace {
    prefill(m, 0, ctx);
    let mut logits = Vec::new();
    for (j, &t) in toks[..keep].iter().enumerate() {
        logits.extend(window(m, 0, ctx.len() + j, &[t]));
        m.commit(0, 1);
    }
    let (kda, latent, pools) = state(m, 0, ctx.len() + keep);
    Trace { logits, kda, latent, pools }
}

/// Windowed: one window of all of `toks`, of which `keep` rows are committed; the trace holds the logits of the kept rows.
fn windowed(m: &mut GlmModel, ctx: &[u32], toks: &[u32], keep: usize) -> Trace {
    prefill(m, 0, ctx);
    let rows = window(m, 0, ctx.len(), toks);
    m.commit(0, keep);
    let logits = rows[..keep].to_vec();
    let (kda, latent, pools) = state(m, 0, ctx.len() + keep);
    Trace { logits, kda, latent, pools }
}

fn same(a: &Trace, b: &Trace, what: &str) {
    assert_eq!(a.logits.len(), b.logits.len(), "{what}: rows");
    for (i, (x, y)) in a.logits.iter().zip(&b.logits).enumerate() {
        assert!(
            x == y,
            "{what}: logits of row {i} differ ({} of {} elements)",
            x.iter().zip(y).filter(|(p, q)| p != q).count(),
            x.len()
        );
    }
    for (l, (x, y)) in a.kda.iter().zip(&b.kda).enumerate() {
        assert!(x.0 == y.0, "{what}: KDA state of KDA layer #{l} differs");
        assert!(x.1 == y.1, "{what}: KDA conv history of KDA layer #{l} differs");
    }
    assert!(a.latent == b.latent, "{what}: MLA latent rows differ");
    assert!(a.pools == b.pools, "{what}: DSA pool rows differ");
}

fn prompt_tokens(n: usize) -> Option<Vec<u32>> {
    let oracle = oracle_dir()?;
    let ids = ids_of(&oracle, "long_2300");
    assert!(ids.len() >= n);
    Some(ids[..n].to_vec())
}

#[test]
#[ignore]
fn window_equals_serial_for_every_width_and_every_partial_accept() {
    let Some(all) = prompt_tokens(2200) else { return };
    let Some(mut m) = model() else { return };
    // contexts of different shapes: a short one, one that ends mid-pool (len % 4 != 0), one that ends mid-page
    for ctx_len in [37usize, 130, 259] {
        let ctx = &all[..ctx_len];
        let toks = &all[ctx_len..ctx_len + 9];
        // R = 1..=9 fully accepted
        for r in 1..=9usize {
            let a = serial(&mut m, ctx, &toks[..r], r);
            let b = windowed(&mut m, ctx, &toks[..r], r);
            same(&a, &b, &format!("ctx {ctx_len}, window of {r} rows, all kept"));
        }
        // a window of 9 with 1..=8 rows kept: the state must be as if only the kept rows were ever decoded, and the next window too
        for keep in 1..9usize {
            let a = serial(&mut m, ctx, toks, keep);
            let b = windowed(&mut m, ctx, toks, keep);
            same(&a, &b, &format!("ctx {ctx_len}, window of 9, {keep} kept"));
            // the next rows after the partial accept (the verified rows are the kept ones: take the following prompt tokens as the next window)
            let next = &all[ctx_len + keep..ctx_len + keep + 5];
            let cont = |m: &mut GlmModel, serial_first: bool| -> Vec<Vec<u16>> {
                if serial_first {
                    prefill(m, 0, ctx);
                    for (j, &t) in toks[..keep].iter().enumerate() {
                        window(m, 0, ctx.len() + j, &[t]);
                        m.commit(0, 1);
                    }
                } else {
                    prefill(m, 0, ctx);
                    window(m, 0, ctx.len(), toks);
                    m.commit(0, keep);
                }
                window(m, 0, ctx.len() + keep, next)
            };
            let (x, y) = (cont(&mut m, true), cont(&mut m, false));
            assert!(x == y, "ctx {ctx_len}: the window after {keep} kept rows differs from the one after serial decoding");
        }
        eprintln!("ctx {ctx_len}: R = 1..9 and keep = 1..8 equal serial decoding bit for bit (logits, KDA state and conv, latents, pools)");
    }
}

#[test]
#[ignore]
fn window_equals_serial_across_the_dsa_regime_boundary() {
    // the window straddles position 2,048: its first rows are dense, the later ones sparse, each as if decoded alone
    let Some(all) = prompt_tokens(2200) else { return };
    let Some(mut m) = model() else { return };
    for (ctx_len, r, keep) in [(2044usize, 9usize, 9usize), (2044, 9, 5), (2044, 9, 2), (2047, 4, 4), (2048, 3, 3)] {
        let ctx = &all[..ctx_len];
        let toks = &all[ctx_len..ctx_len + r];
        let a = serial(&mut m, ctx, toks, keep);
        let b = windowed(&mut m, ctx, toks, keep);
        same(&a, &b, &format!("ctx {ctx_len}, window of {r}, {keep} kept"));
        eprintln!("ctx {ctx_len} + window of {r} ({keep} kept) across the 2,048 boundary equals serial decoding");
    }
}

#[test]
#[ignore]
fn prefill_is_chunk_invariant() {
    // a prompt cut at different places (and fed 1 token at a time through the decode path) leaves the same state and the same next-token logits
    let Some(all) = prompt_tokens(300) else { return };
    let Some(mut m) = model() else { return };
    // `serial_head` tokens go through the decode path (a one-row window committed each time), the rest through prefill pieces cut at `cuts`
    let run = |m: &mut GlmModel, serial_head: usize, cuts: &[usize]| -> (Vec<u16>, Trace) {
        m.begin_slot(0).unwrap();
        for j in 0..serial_head {
            window(m, 0, j, &all[j..=j]);
            m.commit(0, 1);
        }
        let mut at = serial_head;
        let bounds: Vec<usize> = cuts.iter().copied().filter(|&c| c > serial_head).chain([all.len()]).collect();
        for c in bounds {
            // the kernels take at most 128 rows a call: a cut piece longer than that is fed in 128-row slices
            while at < c {
                let end = (at + 128).min(c);
                let want = (end == all.len()).then_some(1usize);
                m.prefill_chunk(0, at, &all[at..end], want).unwrap();
                at = end;
            }
        }
        let row = m.logits_host(0, 1).unwrap();
        let (kda, latent, pools) = state(m, 0, all.len());
        (row, Trace { logits: vec![], kda, latent, pools })
    };
    let (r0, t0) = run(&mut m, 0, &[]);
    for (head, cuts) in [(0usize, vec![1usize, 129]), (0, vec![7, 100, 250]), (0, vec![200]), (20, vec![]), (3, vec![50])] {
        let (r1, t1) = run(&mut m, head, &cuts);
        assert!(r0 == r1, "last-row logits differ with {head} serial tokens and cuts {cuts:?}");
        same(&t0, &t1, &format!("{head} serial tokens, cuts {cuts:?}"));
    }
    eprintln!("a 300-token prompt cut five ways (two of them starting with tokens decoded one at a time) leaves identical logits, state and caches");
}

/// Mutation check of the executor on the whole model: each deliberate schedule bug (`ops::Mutant`) is dumped for REXL3_GLM_MUTANT_PROMPT (default
/// `short_20`, forced routing) into REXL3_GLM_MUTANT_OUT/<mutant>, to be judged by `tools/compare_glm_dump.py ORACLE DIR --skip mtp,vision,kda_conv`,
/// which must FAIL for every one of them (the unmutated forward passes the same comparison). A mutant the executor refuses by itself (a layer whose weights
/// disagree with the mixer the schedule names) is reported and leaves no dump.
#[test]
#[ignore]
fn schedule_mutants_are_dumped() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(out) = std::env::var("REXL3_GLM_MUTANT_OUT").ok().map(PathBuf::from) else {
        return eprintln!("REXL3_GLM_MUTANT_OUT is not set: not run");
    };
    let prompt = std::env::var("REXL3_GLM_MUTANT_PROMPT").unwrap_or_else(|_| "short_20".into());
    let Some(mut m) = model() else { return };
    let ids = ids_of(&oracle, &prompt);
    m.forced = Some(forced_of(&oracle, &prompt));
    for (name, mutant) in [
        ("same_site", Mutant::SameSite),
        ("mla_shifted", Mutant::MlaShifted),
        ("four_dense", Mutant::FourDense),
        ("shared_dropped", Mutant::SharedDropped),
        ("shared_after", Mutant::SharedAfter),
        ("no_head_mean", Mutant::NoHeadMean),
    ] {
        m.mutant = Some(mutant);
        match dump_prompt(&mut m, &prompt, "text", &ids, &out.join(name)) {
            Ok(d) => eprintln!("mutant {name}: dumped {} tokens, greedy next token {}", d.tokens, d.greedy_next_token),
            Err(e) => eprintln!("mutant {name}: the executor refused it: {e}"),
        }
        // a refused forward may leave the slot half-written: start the next one clean
        m.end_slot(0);
    }
    m.mutant = None;
    m.forced = None;
}

/// The vocabulary mask: the 24 padding ids past the tokenizer's vocabulary are never offered, even when their logits are the largest in the row.
/// With the mask removed (`Mutant::UnmaskedVocab`) the same check sees them, which is how the check is shown to bite.
#[test]
#[ignore]
fn padding_ids_are_never_offered() {
    let Some(all) = prompt_tokens(40) else { return };
    let Some(mut m) = model() else { return };
    let offered = |m: &mut GlmModel, mutant: Option<Mutant>| -> Vec<i32> {
        m.mutant = mutant;
        let b = RowBatch::from_requests(&[gb_engine::RowRequest { slot: 0, tokens: all[..5].to_vec(), first_position: 0 }]);
        m.begin_slot(0).unwrap();
        m.forward(&b, Mode::Window, Head::AllRows).unwrap();
        m.poison_padding_logits(5).unwrap();
        let c = m.candidates_of(5, 8).unwrap();
        m.mutant = None;
        m.end_slot(0);
        c.ids.iter().copied().filter(|&i| i >= alloc::TOKENIZER_VOCAB as i32).collect()
    };
    let masked = offered(&mut m, None);
    assert!(masked.is_empty(), "the padding ids {masked:?} were offered through the mask");
    let unmasked = offered(&mut m, Some(Mutant::UnmaskedVocab));
    assert_eq!(
        unmasked.len(),
        5 * 8,
        "with the mask removed every candidate is a padding id (their logits are the largest): the check has teeth"
    );
    eprintln!("masked: no padding id among 5 rows x 8 candidates; unmasked (mutant): {} of 40 are padding ids", unmasked.len());
}

/// Teacher-forced greedy agreement: the prompts of REXL3_GLM_GREEDY_IN followed by the oracle's own completion (REXL3_GLM_FORCE_IN, name -> ids,
/// `tools/glm_greedy_check.py force`), one position at a time through the decode path. At every position it records whether this forward's argmax
/// is the oracle's next token and, where it is not, by how much (logit gap to the oracle's token when that is among the top 8, else to the 8th).
/// Written to REXL3_GLM_FORCE_OUT. Unlike free-running generation, one near-tie costs one position, not the rest of the completion.
#[test]
#[ignore]
fn teacher_forced_greedy() {
    let (Ok(inp), Ok(forced_in), Ok(out)) =
        (std::env::var("REXL3_GLM_GREEDY_IN"), std::env::var("REXL3_GLM_FORCE_IN"), std::env::var("REXL3_GLM_FORCE_OUT"))
    else {
        return eprintln!("REXL3_GLM_GREEDY_IN / REXL3_GLM_FORCE_IN / REXL3_GLM_FORCE_OUT are not set: not run");
    };
    let Some(mut m) = model() else { return };
    let ids_of_json =
        |v: &serde_json::Value| -> Vec<u32> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect() };
    let prompts: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(&inp).unwrap()).unwrap();
    let forced: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(&forced_in).unwrap()).unwrap();
    let mut result = serde_json::Map::new();
    for (name, p) in &prompts {
        let Some(f) = forced.get(name) else { continue };
        let (ids, f) = (ids_of_json(p), ids_of_json(f));
        m.begin_slot(0).unwrap();
        let mut at = 0;
        let mut cands = None;
        while at < ids.len() {
            let end = (at + 128).min(ids.len());
            cands = m.prefill_chunk(0, at, &ids[at..end], (end == ids.len()).then_some(8usize)).unwrap();
            at = end;
        }
        let mut c = cands.unwrap();
        let mut misses = Vec::new();
        for (j, &want) in f.iter().enumerate() {
            let top = c.ids[0] as u32;
            if top != want {
                let w = c.ids[..8].iter().position(|&x| x as u32 == want);
                let gap = c.vals[0] - w.map_or(c.vals[7], |k| c.vals[k]);
                misses.push(serde_json::json!([j, top, want, gap, w.is_some()]));
            }
            if j + 1 < f.len() {
                let b = RowBatch::from_requests(&[gb_engine::RowRequest {
                    slot: 0,
                    tokens: vec![want],
                    first_position: (ids.len() + j) as u64,
                }]);
                c = m.window_candidates(&b, 8).unwrap();
                m.commit(0, 1);
            }
        }
        eprintln!("{name}: {} of {} oracle tokens are this forward's argmax", f.len() - misses.len(), f.len());
        result.insert(name.clone(), serde_json::json!({"n": f.len(), "misses": misses}));
    }
    std::fs::write(out, serde_json::to_vec(&result).unwrap()).unwrap();
}
