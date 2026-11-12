//! The rig the MiniCPM-V-4.7 prefix-cache and long-context tests share: the pack, an engine booted the way `serve` boots it with a
//! prefix cache of a given size, and the small helpers the tests are written in.
#![allow(dead_code)]

use gb_engine::{CutPolicy, Engine, Output, PrefixConfig, Request};
use gb_model::families::minicpmv4_7::backend::MiniCpmBackend;
use gb_model::families::minicpmv4_7::boot::{boot_with, BootExtras, BootPlan, BootReport};
use gb_model::families::minicpmv4_7::config::MiniCpmConfig;
use gb_model::families::minicpmv4_7::engine::engine_from;
use gb_model::families::minicpmv4_7::model::MiniCpmModel;
use gb_model::families::minicpmv4_7::prefix;
use gb_model::serve::{PrefixCacheOption, ServeOptions};
use gb_sampling::Sampling;
use std::path::{Path, PathBuf};

pub type Eng = Engine<MiniCpmBackend>;

pub fn pack_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_MINICPM_DIR") {
        Ok(d) if Path::new(&d).join("config.json").exists() => {
            if !gb_cuda::device_available() {
                eprintln!("not run: no CUDA device");
                return None;
            }
            Some(PathBuf::from(d))
        }
        _ => {
            eprintln!("not run: REXL3_MINICPM_DIR is not set to a pack");
            None
        }
    }
}

pub fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The policy the engine tests use: cuts from 128 tokens (the shipped minimum is 512), so that short prompts exercise it.
pub fn test_policy() -> PrefixConfig {
    PrefixConfig {
        policy: CutPolicy { min_cut: 128, tail: 16, ghost_min_gain: 512, min_gap: 64 },
        ghost_blocks: 1 << 12,
        ..Default::default()
    }
}

pub struct Rig {
    pub eng: Eng,
    pub cfg: MiniCpmConfig,
    pub plan: BootPlan,
    pub report: BootReport,
    /// `MemAvailable` (bytes) before the boot, and after it.
    pub mem_before: Option<u64>,
    pub mem_after: Option<u64>,
}

impl Rig {
    /// The engine of the pack with `slots` slots, `max_ctx` tokens of pool, `chunk` rows a prefill chunk and a prefix cache of
    /// `units` entry units (0: none; a unit is one snapshot slot and the planner's 8 pages, so the cache holds `units` snapshots
    /// and `8 * units` pages: 20.9 MiB on the 8-layer smoke pack, 101.4 MiB on the full pack), booted the way `serve` boots it, with
    /// `policy` for the cuts.
    pub fn load_with(slots: usize, max_ctx: usize, chunk: usize, units: u64, policy: PrefixConfig) -> Option<Rig> {
        let dir = pack_dir()?;
        let cfg = MiniCpmConfig::from_dir(&dir).unwrap();
        let unit = prefix::snapshot_bytes(&cfg) + 8 * prefix::page_bytes(&cfg);
        let cache_mib = (units * unit).div_ceil(1 << 20);
        let mut o = ServeOptions::new(dir);
        o.slots = slots;
        o.slots_auto = false;
        o.allow_unvalidated_pack = true;
        o.max_model_len = max_ctx;
        o.prefill_chunk = chunk;
        o.prefix_cache = if cache_mib == 0 { PrefixCacheOption::Off } else { PrefixCacheOption::Mib(cache_mib) };
        let mem_before = gb_model::serve::mem_available();
        let loaded =
            boot_with(&o, BootExtras { allow_truncated_layers: cfg.layers < 40 }).unwrap_or_else(|e| panic!("boot: {e}"));
        let (plan, report) = (loaded.plan.clone(), loaded.report.clone());
        let eng = engine_from(loaded, &o).unwrap_or_else(|e| panic!("engine: {e}"));
        // the production path turns the cache on by itself (a mutant that never calls `Engine::with_prefix_cache` fails here);
        // the test policy then replaces the configuration, not the decision
        assert_eq!(eng.prefix_stats().enabled, units > 0, "the engine from `engine_from` runs the cache the plan carved out");
        assert_eq!(plan.memory.cache.snapshot_slots as u64, units, "the plan holds the units asked for");
        let eng = eng.with_prefix_cache(policy);
        let mem_after = gb_model::serve::mem_available();
        eprintln!(
            "loaded {} layers: {} slots, {} pool pages, {} snapshot slots, {} cache pages, context {}",
            cfg.layers, report.slots, report.kv_pages, report.snapshot_slots, report.cache_pages, plan.max_context
        );
        Some(Rig { eng, cfg, plan, report, mem_before, mem_after })
    }

    pub fn load(slots: usize, max_ctx: usize, chunk: usize, units: u64) -> Option<Rig> {
        Self::load_with(slots, max_ctx, chunk, units, test_policy())
    }

    pub fn m(&mut self) -> &mut MiniCpmModel {
        &mut self.eng.backend.m
    }
}

/// `n` tokens of a stream named by `seed`, inside the vocabulary.
pub fn toks(seed: u64, n: usize) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            8 + (x % 200_000) as u32
        })
        .collect()
}

pub fn cat(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().chain(b).copied().collect()
}

pub fn greedy() -> Sampling {
    Sampling { seed: 0, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 }
}

pub fn sampled(seed: u64) -> Sampling {
    Sampling { seed, temperature: 0.8, top_k: 20, top_p: 0.95, min_p: 0.0 }
}

pub fn request(prompt: &[u32], sampling: Sampling, max_new: usize, draft: bool, cache: bool) -> Request {
    Request { prompt: prompt.to_vec(), sampling, max_new, stop: vec![], draft, images: vec![], cache }
}

/// One request alone through the engine, to the end.
pub fn one(e: &mut Eng, r: Request) -> Output {
    let id = e.submit(r).unwrap();
    e.run().unwrap();
    assert!(e.take_failed().is_empty(), "no request may fail");
    let mut outs = e.take_outputs();
    outs.retain(|o| o.id == id);
    assert_eq!(outs.len(), 1);
    outs.remove(0)
}

/// Several requests together (submitted before any prefill runs), outputs in submission order.
pub fn many(e: &mut Eng, rs: Vec<Request>) -> Vec<Output> {
    let ids: Vec<u64> = rs.into_iter().map(|r| e.submit_deferred(r).unwrap()).collect();
    e.run().unwrap();
    assert!(e.take_failed().is_empty(), "no request may fail");
    let mut outs = e.take_outputs();
    ids.iter()
        .map(|id| {
            let at = outs.iter().position(|o| o.id == *id).expect("every request finished");
            outs.remove(at)
        })
        .collect()
}

pub fn same(what: &str, a: &Output, b: &Output) {
    let at = a.tokens.iter().zip(&b.tokens).position(|(x, y)| x != y);
    assert!(
        a.tokens == b.tokens && (a.finish, a.stopped) == (b.finish, b.stopped),
        "{what}: the replies differ (first difference at {at:?}; {} against {} tokens; finish {:?} against {:?})",
        a.tokens.len(),
        b.tokens.len(),
        a.finish,
        b.finish
    );
}
