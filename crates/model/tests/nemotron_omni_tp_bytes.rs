//! v0.1.2 P2-8 on the real packs (CPU: the index and the shard headers, never a weight; no GPU): the byte-weighted expert balance
//! (`tp_balance::byte_balanced_map`) against today's tensor-parallel shard plan.
//!
//! * `REXL3_NEMO_H4_DIR` (our 4-bit-head pack, `hf-nemotron-main-h4`, index eb0f1ada...: not the gate's `REXL3_NEMO_OMNI_DIR`, the
//!   bf16-head pack, whose bytes differ): R-MK-noreg (c). The plan after the byte balance is today's, exactly: the same `ShardPlan`,
//!   every tensor's piece, every rank's bytes, every MoE layer's experts, at 2, 3, 4 and 8 ranks.
//! * `REXL3_NEMO_MIXEDK_DIR` (the MixedK pack, vcruz305 3.85 bpw @39638152): what each rank holds under the naive equal-count plan (the
//!   contiguous ranges) and under the byte-balanced table, per MoE layer and in all, printed. `LayerPlan::build` does not take this
//!   pack yet (its K != 4 trellises, F16 router and latent projections, EXL3 `lm_head` and head attention are lane L's), so its plan is
//!   synthesized from the inventory ([`plan_from_inventory`]: every text tensor in the unit its name says, every `.trellis` a linear),
//!   which on our pack gives exactly `LayerPlan::build`'s bytes (checked here). An EXL3 `lm_head` is counted as its N-cut at 128 columns
//!   (two ranks: 65,536 columns each), which `ShardPlan::piece` does not cut yet (lane L): see [`lm_head_rank_bytes`].
//! * `REXL3_NEMO_TRACES_DIR` (with the MixedK pack: the routing traces of `tests/nemotron_omni_routing.rs`, `.rxrt`): the trace solver
//!   at equal cost against the byte-weighted one, scored by bytes.
//!
//! Without the variables each test says what it did not run and passes.

use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::tp::TpTopology;
use gb_model::families::nemotron_h_omni::tp_arm::serving_shard_plan_with;
use gb_model::families::nemotron_h_omni::tp_balance::{
    byte_balanced_map, evaluate_map_weighted, moe_layers, solve, solve_weighted, Cost, ExpertMap, Grouping, RoutingTrace,
    SolveOpts,
};
use gb_model::families::nemotron_h_omni::tp_shard::{column_range, expert_bytes, hg_wanted, rank_listed_bytes, ShardPlan};
use gb_model::families::nemotron_h_omni::weights::{LayerPlan, Linear, Unit, UnitId, Want};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn dir(var: &str, what: &str) -> Option<PathBuf> {
    let d = std::env::var_os(var).map(PathBuf::from);
    if d.is_none() {
        eprintln!("not run: {var} is not set ({what})");
    }
    d
}

fn config(dir: &Path) -> OmniConfig {
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap()
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// The serving plan of `rank` of `ranks` with the table `map` (`None`: the contiguous ranges).
fn serving(rank: usize, ranks: usize, cfg: &OmniConfig, map: Option<&ExpertMap>) -> ShardPlan {
    serving_shard_plan_with(TpTopology { rank, ranks }, cfg, map.map(|m| Arc::new(m.clone())))
}

/// The same plan, as R-MK-noreg (c) means it: the value, every unit's pieces and linears, the rank's bytes (with and without the
/// head-group `in_proj`, and the memory plan's pair) and every MoE layer's experts.
fn assert_same_plan(a: &ShardPlan, b: &ShardPlan, p: &LayerPlan, cfg: &OmniConfig, what: &str) {
    assert_eq!(a, b, "{what}: the plan value");
    for u in &p.units {
        assert_eq!(a.local_linears(u, cfg.experts), b.local_linears(u, cfg.experts), "{what}: {:?}", u.id);
        for w in &u.tensors {
            assert_eq!(a.piece(u, w, cfg.experts), b.piece(u, w, cfg.experts), "{what}: {}", w.name);
        }
    }
    for hg in [false, true] {
        assert_eq!(
            a.rank_bytes_with(p, cfg.experts, |_| true, hg),
            b.rank_bytes_with(p, cfg.experts, |_| true, hg),
            "{what}: bytes (hg {hg})"
        );
    }
    assert_eq!(rank_listed_bytes(a, p, cfg), rank_listed_bytes(b, p, cfg), "{what}: the memory plan's pair");
    for l in moe_layers(cfg) {
        assert_eq!(a.layer_experts(l, cfg.experts), b.layer_experts(l, cfg.experts), "{what}: layer {l}");
    }
}

fn static_dtype(d: &str) -> &'static str {
    match d {
        "BF16" => "BF16",
        "F16" => "F16",
        "F32" => "F32",
        "I16" => "I16",
        "I32" => "I32",
        "U8" => "U8",
        "I8" => "I8",
        "U16" => "U16",
        "U32" => "U32",
        "I64" => "I64",
        other => Box::leak(other.to_string().into_boxed_str()),
    }
}

/// A plan of every text tensor of the inventory as stored (by canonical name): `language_model.backbone.layers.{i}.*` in trunk unit
/// `i`, `language_model.mtp.layers.{b}.*` in head unit `b`, the embedding, the final norm and the `lm_head` in theirs; every `.trellis`
/// `[K/16, N/16, 16 bits]` an EXL3 linear `K x N`. No check of dtypes or shapes: an accounting of what the pack holds, for a pack
/// `LayerPlan::build` does not take yet.
fn plan_from_inventory(inv: &Inventory) -> LayerPlan {
    let mut units: Vec<Unit> = Vec::new();
    let mut outside = 0usize;
    let unit_of = |name: &str| -> Option<UnitId> {
        let rest = name.strip_prefix("language_model.")?;
        let num = |s: &str| s.split('.').next().and_then(|n| n.parse::<usize>().ok());
        if let Some(r) = rest.strip_prefix("backbone.layers.") {
            return num(r).map(UnitId::Trunk);
        }
        if let Some(r) = rest.strip_prefix("mtp.layers.") {
            return num(r).map(UnitId::Mtp);
        }
        if rest.starts_with("backbone.embeddings.") {
            return Some(UnitId::Embed);
        }
        if rest.starts_with("backbone.norm_f.") {
            return Some(UnitId::NormF);
        }
        if rest.starts_with("lm_head.") {
            return Some(UnitId::LmHead);
        }
        None
    };
    for (name, t) in &inv.tensors {
        let Some(id) = unit_of(name) else {
            outside += 1;
            continue;
        };
        let at = match units.iter().position(|u| u.id == id) {
            Some(i) => i,
            None => {
                units.push(Unit { id, label: format!("{id:?}"), tensors: Vec::new(), linears: Vec::new() });
                units.len() - 1
            }
        };
        let u = &mut units[at];
        u.tensors.push(Want { name: name.clone(), dtype: static_dtype(&t.dtype), shape: t.shape.clone(), unit: id });
        if let Some(prefix) = name.strip_suffix(".trellis") {
            if t.shape.len() == 3 {
                // the width from the trellis' last dimension (16 * bits; lane L's per-tensor widths, P0-1)
                u.linears.push(Linear {
                    prefix: prefix.to_string(),
                    k: t.shape[0] * 16,
                    n: t.shape[1] * 16,
                    bits: t.shape[2] / 16,
                });
            }
        }
    }
    LayerPlan { units, outside_plan: outside }
}

/// Bytes a rank holds of an EXL3 `lm_head` cut along N at 128 columns (the vocabulary's tiles, `[floor(T r / R), floor(T (r+1) / R))`,
/// `crate::shard::column_range`): its trellis columns and `svh`, the whole `suh` and the codebook marker. `None` for a bf16 `lm_head`
/// (`ShardPlan::piece` cuts its rows already).
fn lm_head_rank_bytes(p: &LayerPlan, rank: usize, ranks: usize) -> Option<u64> {
    let u = p.unit(UnitId::LmHead)?;
    let tr = u.tensors.iter().find(|w| w.name.ends_with(".trellis"))?;
    let n = tr.shape[1] * 16;
    let (lo, hi) = column_range(n, rank, ranks);
    let cols = hi - lo;
    let mut b = 0u64;
    for w in &u.tensors {
        let numel: u64 = w.shape.iter().map(|&d| d as u64).product();
        let size = gb_exl3::checkpoint::dtype_size(w.dtype).unwrap_or(0);
        b += match w.name.rsplit('.').next() {
            Some("trellis") => (w.shape[0] * (cols / 16) * w.shape[2]) as u64 * size,
            Some("svh") => cols as u64 * size,
            _ => numel * size,
        };
    }
    Some(b)
}

/// What every rank of `ranks` holds of the plan under `map`: (everything but the `lm_head`, of which the head, the `lm_head`), with the
/// head-group `in_proj` as the process's contract says (`hg`).
fn rank_bytes(p: &LayerPlan, cfg: &OmniConfig, ranks: usize, map: Option<&ExpertMap>, hg: bool) -> Vec<(u64, u64, u64)> {
    (0..ranks)
        .map(|r| {
            let sp = if ranks == 1 { ShardPlan::single() } else { serving(r, ranks, cfg, map) };
            let body = sp.rank_bytes_with(p, cfg.experts, |u| u != UnitId::LmHead, hg);
            let head = sp.rank_bytes_with(p, cfg.experts, |u| matches!(u, UnitId::Mtp(_)), false);
            let lm = match lm_head_rank_bytes(p, r, ranks) {
                Some(b) => b,
                None => sp.rank_bytes_with(p, cfg.experts, |u| u == UnitId::LmHead, false),
            };
            (body, head, lm)
        })
        .collect()
}

#[test]
fn our_packs_tp_plan_after_the_byte_balance_is_todays_exactly() {
    let Some(d) = dir("REXL3_NEMO_H4_DIR", "our 4-bit-head pack, e.g. ~/projects/hf-nemotron-main-h4") else { return };
    let cfg = config(&d);
    let inv = Inventory::read(&d).unwrap_or_else(|e| panic!("{}: {e}", d.display()));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let moe = moe_layers(&cfg);
    let t = expert_bytes(&plan, cfg.experts);
    assert_eq!(t.bytes.len(), moe.len(), "every MoE layer has its experts' bytes");
    assert!(t.widths.values().all(|w| w == &vec![4]), "one width: {:?}", t.widths);
    assert!(t.mixed_layers().is_empty(), "no layer mixes widths");
    // the table sees the pack's three bf16 pairs (and nothing else differs)
    let heavy: Vec<(usize, usize)> = t
        .bytes
        .iter()
        .flat_map(|(&l, row)| {
            let base = *row.iter().min().unwrap();
            row.iter().enumerate().filter(move |&(_, &b)| b != base).map(move |(e, _)| (l, e))
        })
        .collect();
    assert_eq!(heavy, vec![(26, 111), (39, 304), (81, 461)], "the measured bf16 expert pairs");
    // the synthesized plan of the other test holds exactly this plan's bytes on this pack (single, and each rank of two)
    let synth = plan_from_inventory(&inv);
    assert_eq!(expert_bytes(&synth, cfg.experts), t, "the synthesized plan's expert table");
    for ranks in [1usize, 2] {
        for r in 0..ranks {
            let sp = if ranks == 1 { ShardPlan::single() } else { serving(r, ranks, &cfg, None) };
            for hg in [false, true] {
                assert_eq!(
                    sp.rank_bytes_with(&synth, cfg.experts, |_| true, hg),
                    sp.rank_bytes_with(&plan, cfg.experts, |_| true, hg),
                    "synthesized plan, R={ranks} rank {r} hg {hg}"
                );
            }
        }
    }
    for ranks in [2usize, 3, 4, 8] {
        let map = byte_balanced_map(&t, ranks, &moe).unwrap();
        assert_eq!(map, None, "R={ranks}: no table, today's contiguous ranges");
        for r in 0..ranks {
            let today = serving(r, ranks, &cfg, None);
            let after = serving(r, ranks, &cfg, map.as_ref());
            assert_same_plan(&today, &after, &plan, &cfg, &format!("R={ranks} rank {r}"));
            assert!(after.expert_map.is_none());
        }
    }
    // the numbers (today's code path, unchanged since v0.1.1: no crate file differs between c2c4801b and the lane's base)
    let single = ShardPlan::single().rank_bytes_with(&plan, cfg.experts, |_| true, false);
    let pairs: Vec<(u64, u64)> = (0..2)
        .map(|r| {
            let sp = serving(r, 2, &cfg, None);
            (sp.rank_bytes_with(&plan, cfg.experts, |_| true, false), sp.rank_bytes_with(&plan, cfg.experts, |_| true, true))
        })
        .collect();
    eprintln!("our pack: single {single} B ({:.3} GiB)", gib(single));
    for (r, (a, b)) in pairs.iter().enumerate() {
        eprintln!(
            "our pack: TP=2 rank {r}: {a} B ({:.3} GiB) without the head-group in_proj, {b} B ({:.3} GiB) with it",
            gib(*a),
            gib(*b)
        );
    }
    assert_eq!(single, OURS_SINGLE, "our pack, one node");
    assert_eq!(pairs, OURS_TP2.to_vec(), "our pack, two ranks (without, with the head-group in_proj)");
}

/// Our pack's planned bytes on one node and on each rank of two (`(without, with the head-group in_proj)`), the serving plan's
/// defaults: what today's code computes for `hf-nemotron-main-h4` (index eb0f1ada).
const OURS_SINGLE: u64 = 64_002_983_464;
const OURS_TP2: [(u64, u64); 2] = [(33_633_325_624, 34_399_861_464), (33_652_064_304, 34_418_600_144)];

#[test]
fn his_packs_byte_balanced_plan_against_the_naive_one() {
    let Some(d) = dir("REXL3_NEMO_MIXEDK_DIR", "the MixedK pack, e.g. ~/projects/vcruz-nemotron-mixedk-385-39638152") else {
        return;
    };
    let cfg = config(&d);
    let inv = Inventory::read(&d).unwrap_or_else(|e| panic!("{}: {e}", d.display()));
    let plan = plan_from_inventory(&inv);
    let moe = moe_layers(&cfg);
    let t = expert_bytes(&plan, cfg.experts);
    assert_eq!(t.bytes.len(), moe.len());
    let mixed = t.mixed_layers();
    eprintln!("MixedK pack: {} of {} MoE layers mix widths; one width: {:?}", mixed.len(), moe.len(), {
        moe.iter().filter(|l| !mixed.contains(l)).map(|l| (l, t.widths[l].clone())).collect::<Vec<_>>()
    });
    let ranks = 2;
    let map = byte_balanced_map(&t, ranks, &moe).unwrap().expect("a MixedK pack moves experts");
    map.check_for(&cfg, ranks).unwrap();
    let contiguous = ExpertMap::contiguous(ranks, cfg.experts, moe.iter().copied()).unwrap();
    eprintln!("\nper MoE layer, routed experts' bytes rank 0 | rank 1: naive (contiguous) -> byte-balanced; widths");
    let (mut worst_naive, mut worst_bal) = (0f64, 0f64);
    for &l in &moe {
        let n = t.rank_bytes(l, contiguous.row(l).unwrap(), ranks);
        let b = t.rank_bytes(l, map.row(l).unwrap(), ranks);
        let rel = |v: &[u64]| (v[0] as f64 - v[1] as f64) / ((v[0] + v[1]) as f64 / 2.0) * 100.0;
        worst_naive = worst_naive.max(rel(&n).abs());
        worst_bal = worst_bal.max(rel(&b).abs());
        eprintln!(
            "  L{l:2}: {:>11} | {:>11} ({:+.2}%) -> {:>11} | {:>11} ({:+.3}%)  K {:?}",
            n[0],
            n[1],
            rel(&n),
            b[0],
            b[1],
            rel(&b),
            t.widths[&l]
        );
        assert_eq!(n[0] + n[1], b[0] + b[1]);
        assert!(b[0].abs_diff(b[1]) <= n[0].abs_diff(n[1]), "layer {l}");
    }
    let (tn, tb) = (t.rank_totals(Some(&contiguous), ranks), t.rank_totals(Some(&map), ranks));
    eprintln!(
        "  all layers: naive {} | {} (diff {:+} B), balanced {} | {} (diff {:+} B); worst layer naive {worst_naive:.2}%, balanced \
         {worst_bal:.3}%",
        tn[0],
        tn[1],
        tn[1] as i64 - tn[0] as i64,
        tb[0],
        tb[1],
        tb[1] as i64 - tb[0] as i64
    );
    // the whole pack per rank, the plan code (the lm_head as its N-cut)
    let hg_default = hg_wanted(&serving(0, 2, &cfg, None));
    eprintln!(
        "\nwhat each rank holds (ShardPlan::rank_bytes; head-group in_proj of the P2 contract: {hg_default} by default here)"
    );
    let line = |what: &str, v: &[(u64, u64, u64)]| {
        for (r, &(body, head, lm)) in v.iter().enumerate() {
            eprintln!(
                "  {what:<28} rank {r}: {:>12} B = {:.3} GiB (of which MTP head {:.3} GiB, lm_head {:.3} GiB)",
                body + lm,
                gib(body + lm),
                gib(head),
                gib(lm)
            );
        }
    };
    line("single node", &rank_bytes(&plan, &cfg, 1, None, false));
    for hg in [false, true] {
        line(&format!("TP=2 naive, hg {hg}"), &rank_bytes(&plan, &cfg, 2, None, hg));
        line(&format!("TP=2 byte-balanced, hg {hg}"), &rank_bytes(&plan, &cfg, 2, Some(&map), hg));
    }
    eprintln!("  (index total_size {:?}; outside the text plan: {} tensors)", inv.index_total_size, plan.outside_plan);
    let naive = rank_bytes(&plan, &cfg, 2, None, false);
    let bal = rank_bytes(&plan, &cfg, 2, Some(&map), false);
    let total = |v: &[(u64, u64, u64)]| v.iter().map(|&(b, _, l)| b + l).sum::<u64>();
    assert_eq!(total(&naive), total(&bal), "every expert once");
    let fullest = |v: &[(u64, u64, u64)]| v.iter().map(|&(b, _, l)| b + l).max().unwrap();
    assert!(fullest(&bal) <= fullest(&naive));

    // the trace solver, equal cost against byte-weighted, scored by bytes (optional)
    let Some(td) = dir("REXL3_NEMO_TRACES_DIR", "routing traces, e.g. ~/projects/nemo-routing-traces/traces") else { return };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&td)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "rxrt"))
        .collect();
    files.sort();
    let traces: Vec<RoutingTrace> = files.iter().map(|p| RoutingTrace::read(p).unwrap()).collect();
    let (train, test): (Vec<&RoutingTrace>, Vec<&RoutingTrace>) = {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for (i, t) in traces.iter().enumerate() {
            if i % 2 == 0 {
                a.push(t)
            } else {
                b.push(t)
            }
        }
        (a, b)
    };
    let opts = SolveOpts { budget: std::time::Duration::from_secs(8), ..SolveOpts::default() };
    eprintln!(
        "\ntrace solver on {} train / {} held-out prompts (the routing of our pack's run: the same model's router)",
        train.len(),
        test.len()
    );
    for cost in [Cost::Pairs, Cost::Passes] {
        let eq = solve(&train, ranks, Grouping::Recorded, 1, cost, &opts, false).unwrap();
        let wt = solve_weighted(&train, ranks, Grouping::Recorded, 1, cost, &opts, false, Some(&t)).unwrap();
        for (label, g) in
            [("windows as run", Grouping::Recorded), ("1 row", Grouping::Window(1)), ("64-row prefill", Grouping::Prefill(64))]
        {
            let s = |m: &ExpertMap| evaluate_map_weighted(m, &test, g, 1, cost, Some(&t)).ratio();
            eprintln!(
                "  [{cost:?}] {label:<16} busiest/mean by bytes, held-out: contiguous {:.4}, static byte map {:.4}, equal-cost solve {:.4}, \
                 byte-weighted solve {:.4}",
                s(&contiguous),
                s(&map),
                s(&eq.map),
                s(&wt.map)
            );
        }
        let rb = |m: &ExpertMap| t.rank_totals(Some(m), ranks);
        eprintln!(
            "  [{cost:?}] experts' bytes a rank: equal-cost solve {:?}, byte-weighted solve {:?}",
            rb(&eq.map),
            rb(&wt.map)
        );
    }
}
