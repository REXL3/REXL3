//! B7 CLI: solve and score balanced expert ownership from routing traces (`tp_balance`, `docs/NEMOTRON_TP.md` section 14).
//! CPU only (`REXL3_NO_CUDA=1` builds it). The traces are pack-derived data and stay outside git (`~/projects/nemo-routing-traces/`).
//!
//!     expert_balance solve   --traces DIR --ranks 2,3,4,8 [--cost pairs|passes] [--budget-s 20] [--out DIR] [--seed N]
//!                            [--min-gain N] [--from-contiguous] [--train-limit N] [--pack DIR]
//!     expert_balance eval    --traces DIR --map FILE [--only train|test|all]
//!     expert_balance shuffle --ranks R --seed N --out FILE [--head] [--config config.json]
//!     expert_balance info    --map FILE
//!     expert_balance bytes   --pack DIR --map FILE      (what each rank holds, against the contiguous ranges)
//!     expert_balance bytes-balance --pack DIR --ranks R [--out FILE]
//!                                                       (the byte-balanced table of a mixed-bit-rate pack, v0.1.2)
//!
//! `solve --pack DIR` weighs each pick by its expert's bytes on the layers of DIR's plan that mix trellis widths (`solve_weighted`);
//! every other layer, and every layer without `--pack`, is solved exactly as before. `bytes-balance` writes the trace-free table that
//! evens the ranks' expert bytes of every mixed layer (`byte_balanced_map`), or says that the pack needs none (one width: today's plan).
//!
//! `solve` splits the prompts of DIR in two halves (sorted by name, alternating within the chat prompts and within the texts: the
//! split is by prompt, never by row), solves each rank count on the TRAIN half and prints busiest-over-mean for the contiguous
//! ownership, the greedy LPT start and the local-search result on train and on the held-out TEST half, under several window shapes;
//! it writes `balanced_R<r>_train.map` (the map the held-out numbers are for) and `balanced_R<r>.map` (solved on every prompt: the one to
//! ship) into `--out`.

use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::tp_balance::{
    byte_balanced_map, evaluate_map, moe_layers, rank_pair_shares, shuffled_map_with_head, solve_weighted, Cost, ExpertBytes,
    ExpertMap, Grouping, RoutingTrace, SolveOpts, KIND_GENERATED,
};
use std::path::{Path, PathBuf};
use std::time::Duration;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("expert_balance: {msg}");
    std::process::exit(2)
}

fn load_traces(dir: &Path) -> Vec<RoutingTrace> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| die(format!("{}: {e}", dir.display())))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "rxrt"))
        .collect();
    files.sort();
    let traces: Vec<RoutingTrace> = files.iter().map(|p| RoutingTrace::read(p).unwrap_or_else(|e| die(e))).collect();
    if traces.is_empty() {
        die(format!("no .rxrt traces in {}", dir.display()));
    }
    traces
}

/// Train and test halves by prompt: chat prompts (those with generated rows) and texts are each sorted by name and alternate.
fn split(traces: &[RoutingTrace]) -> (Vec<&RoutingTrace>, Vec<&RoutingTrace>) {
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for chat in [true, false] {
        let mut group: Vec<&RoutingTrace> = traces.iter().filter(|t| t.kind.contains(&KIND_GENERATED) == chat).collect();
        group.sort_by(|a, b| a.name.cmp(&b.name));
        for (i, t) in group.into_iter().enumerate() {
            if i % 2 == 0 {
                train.push(t);
            } else {
                test.push(t);
            }
        }
    }
    (train, test)
}

/// The window shapes a map is scored under: (label, grouping, streams).
fn shapes() -> Vec<(&'static str, Grouping, usize)> {
    vec![
        ("windows as run (1-8 rows)", Grouping::Recorded, 1),
        ("1 row (serial decode)", Grouping::Window(1), 1),
        ("4 rows (depth-3 verify)", Grouping::Window(4), 1),
        ("8 rows", Grouping::Window(8), 1),
        ("16 streams x 1 row", Grouping::Window(1), 16),
        ("64-row prefill slice", Grouping::Prefill(64), 1),
    ]
}

fn ratio(map: &ExpertMap, set: &[&RoutingTrace], g: Grouping, streams: usize, cost: Cost) -> Option<(f64, usize)> {
    let s = evaluate_map(map, set, g, streams, cost);
    (s.steps > 0).then(|| (s.ratio(), s.steps))
}

fn fmt(v: Option<(f64, usize)>) -> String {
    v.map_or("      -".to_string(), |(r, _)| format!("{r:7.4}"))
}

fn cmd_solve(args: &[String]) {
    let dir = PathBuf::from(arg(args, "--traces").unwrap_or_else(|| die("--traces DIR")));
    let ranks: Vec<usize> = arg(args, "--ranks")
        .unwrap_or_else(|| "2,3,4,8".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap_or_else(|_| die("--ranks 2,3,4,8")))
        .collect();
    let cost = Cost::parse(&arg(args, "--cost").unwrap_or_else(|| "pairs".into())).unwrap_or_else(|e| die(e));
    let budget = arg(args, "--budget-s").map_or(20.0, |v| v.parse().unwrap_or_else(|_| die("--budget-s N")));
    let seed = arg(args, "--seed").map_or(0x5eed, |v| v.parse().unwrap_or_else(|_| die("--seed N")));
    let out = arg(args, "--out").map(PathBuf::from);
    let traces = load_traces(&dir);
    let bytes = arg(args, "--pack").map(|p| pack_bytes(Path::new(&p)).1);
    if let Some(b) = &bytes {
        println!("byte weights from the pack: {} of {} MoE layers mix trellis widths", b.mixed_layers().len(), b.bytes.len());
    }
    let bytes = bytes.as_ref();
    let (mut train, test) = split(&traces);
    // a learning curve: solve on only the first N prompts of the train half (the held-out half does not change)
    if let Some(n) = arg(args, "--train-limit") {
        train.truncate(n.parse().unwrap_or_else(|_| die("--train-limit N")));
    }
    let all: Vec<&RoutingTrace> = traces.iter().collect();
    let rows = |s: &[&RoutingTrace]| s.iter().map(|t| t.rows()).sum::<usize>();
    println!(
        "traces: {} prompts, {} rows; train {} prompts / {} rows, held-out {} prompts / {} rows; cost solved for: {cost:?}; layers {}",
        traces.len(),
        rows(&all),
        train.len(),
        rows(&train),
        test.len(),
        rows(&test),
        traces[0].layers.len()
    );
    println!(
        "train: {}\nheld-out: {}",
        train.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(" "),
        test.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(" ")
    );
    let min_gain = arg(args, "--min-gain").map_or(1, |v| v.parse().unwrap_or_else(|_| die("--min-gain N")));
    let from_contiguous = args.iter().any(|a| a == "--from-contiguous");
    let opts = SolveOpts { budget: Duration::from_secs_f64(budget), seed, min_gain, from_contiguous, ..SolveOpts::default() };
    if let Some(o) = &out {
        std::fs::create_dir_all(o).unwrap_or_else(|e| die(format!("{}: {e}", o.display())));
    }
    for &r in &ranks {
        let t0 = std::time::Instant::now();
        let layers = traces[0].layers.clone();
        let contiguous = ExpertMap::contiguous(r, traces[0].experts, layers).unwrap_or_else(|e| die(e));
        // training steps: the windows as run, one prompt a stream
        let lpt = solve_weighted(&train, r, Grouping::Recorded, 1, cost, &opts, true, bytes).unwrap_or_else(|e| die(e));
        let bal = solve_weighted(&train, r, Grouping::Recorded, 1, cost, &opts, false, bytes).unwrap_or_else(|e| die(e));
        let fin = solve_weighted(&all, r, Grouping::Recorded, 1, cost, &opts, false, bytes).unwrap_or_else(|e| die(e));
        println!("\n== R = {r} (solved in {:.0} s) ==", t0.elapsed().as_secs_f64());
        let pct = |v: Vec<f64>| v.iter().map(|x| format!("{:.1}%", x * 100.0)).collect::<Vec<_>>().join(" ");
        println!(
            "pairs a rank computes, all rows of the held-out half: contiguous [{}], balanced [{}]",
            pct(rank_pair_shares(&contiguous, &test)),
            pct(rank_pair_shares(&bal.map, &test))
        );
        println!(
            "{:<28} {:>6} | {:^23} | {:^23} | {:^23}",
            "busiest / mean", "steps", "contiguous", "LPT (marginal)", "balanced (swap search)"
        );
        println!(
            "{:<28} {:>6} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7}",
            "", "(test)", "train", "test", "gain", "train", "test", "gain", "train", "test", "gain"
        );
        for pc in [Cost::Pairs, Cost::Passes] {
            println!("  [{pc:?}]");
            for (label, g, streams) in shapes() {
                let cell = |m: &ExpertMap, set: &[&RoutingTrace]| ratio(m, set, g, streams, pc);
                let (ctr, cte) = (cell(&contiguous, &train), cell(&contiguous, &test));
                let (ltr, lte) = (cell(&lpt.map, &train), cell(&lpt.map, &test));
                let (btr, bte) = (cell(&bal.map, &train), cell(&bal.map, &test));
                let steps = cte.map_or(0, |(_, n)| n);
                // the gain in the busiest rank's total (what the collectives wait for) on the held-out half
                let gain = |m: &ExpertMap| -> String {
                    let c = evaluate_map(&contiguous, &test, g, streams, pc);
                    let b = evaluate_map(m, &test, g, streams, pc);
                    if c.steps == 0 || c.busiest == 0.0 {
                        "      -".to_string()
                    } else {
                        format!("{:+6.1}%", (b.busiest / c.busiest - 1.0) * 100.0)
                    }
                };
                println!(
                    "  {label:<26} {steps:>6} | {} {} {:>7} | {} {} {} | {} {} {}",
                    fmt(ctr),
                    fmt(cte),
                    "",
                    fmt(ltr),
                    fmt(lte),
                    gain(&lpt.map),
                    fmt(btr),
                    fmt(bte),
                    gain(&bal.map)
                );
            }
        }
        if let Some(o) = &out {
            let p = o.join(format!("balanced_R{r}_train.map"));
            std::fs::write(&p, bal.map.to_text()).unwrap_or_else(|e| die(format!("{}: {e}", p.display())));
            let p2 = o.join(format!("balanced_R{r}.map"));
            std::fs::write(&p2, fin.map.to_text()).unwrap_or_else(|e| die(format!("{}: {e}", p2.display())));
            println!(
                "  wrote {} (hash {:016x}) and {} (hash {:016x})",
                p.display(),
                bal.map.hash(),
                p2.display(),
                fin.map.hash()
            );
        }
    }
}

fn cmd_eval(args: &[String]) {
    let dir = PathBuf::from(arg(args, "--traces").unwrap_or_else(|| die("--traces DIR")));
    let map = ExpertMap::read(Path::new(&arg(args, "--map").unwrap_or_else(|| die("--map FILE")))).unwrap_or_else(|e| die(e));
    let traces = load_traces(&dir);
    let (train, test) = split(&traces);
    let all: Vec<&RoutingTrace> = traces.iter().collect();
    let only = arg(args, "--only").unwrap_or_else(|| "all".into());
    let set: Vec<&RoutingTrace> = match only.as_str() {
        "train" => train,
        "test" => test,
        _ => all,
    };
    let contiguous = ExpertMap::contiguous(map.ranks(), map.experts(), map.layers()).unwrap_or_else(|e| die(e));
    println!(
        "map {:016x}: {} ranks, {} layers; on {} prompts ({only})",
        map.hash(),
        map.ranks(),
        map.layers().count(),
        set.len()
    );
    println!("{:<28} {:>6} | {:^15} | {:^15}", "busiest / mean", "steps", "Pairs: ctg  map", "Passes: ctg  map");
    for (label, g, streams) in shapes() {
        let p = (ratio(&contiguous, &set, g, streams, Cost::Pairs), ratio(&map, &set, g, streams, Cost::Pairs));
        let q = (ratio(&contiguous, &set, g, streams, Cost::Passes), ratio(&map, &set, g, streams, Cost::Passes));
        println!("  {label:<26} {:>6} | {} {} | {} {}", p.0.map_or(0, |x| x.1), fmt(p.0), fmt(p.1), fmt(q.0), fmt(q.1));
    }
}

fn cmd_shuffle(args: &[String]) {
    let ranks: usize = arg(args, "--ranks").unwrap_or_else(|| die("--ranks R")).parse().unwrap_or_else(|_| die("--ranks R"));
    let seed: u64 = arg(args, "--seed").map_or(1, |v| v.parse().unwrap_or_else(|_| die("--seed N")));
    let out = arg(args, "--out").unwrap_or_else(|| die("--out FILE"));
    let cfg_path = arg(args, "--config")
        .unwrap_or_else(|| format!("{}/../../testdata/nemotron_omni/config.json", env!("CARGO_MANIFEST_DIR")));
    let text = std::fs::read_to_string(&cfg_path).unwrap_or_else(|e| die(format!("{cfg_path}: {e}")));
    let cfg = OmniConfig::from_json(&serde_json::from_str(&text).unwrap_or_else(|e| die(e))).unwrap_or_else(|e| die(e));
    let head = args.iter().any(|a| a == "--head");
    let m = shuffled_map_with_head(ranks, cfg.experts, &moe_layers(&cfg), seed, head).unwrap_or_else(|e| die(e));
    std::fs::write(&out, m.to_text()).unwrap_or_else(|e| die(format!("{out}: {e}")));
    println!("wrote {out}: shuffled, {ranks} ranks, seed {seed}, hash {:016x}", m.hash());
}

fn cmd_info(args: &[String]) {
    let m = ExpertMap::read(Path::new(&arg(args, "--map").unwrap_or_else(|| die("--map FILE")))).unwrap_or_else(|e| die(e));
    println!(
        "{}: {} ranks, {} experts, {} layers, head row {}, fingerprint {:016x}",
        m.describe(),
        m.ranks(),
        m.experts(),
        m.layers().count(),
        m.has_head(),
        m.hash()
    );
    let counts: Vec<usize> = (0..m.ranks()).map(|r| m.experts_on(r)).collect();
    println!("experts a rank holds per layer: {counts:?}");
}

/// What each rank of the group holds of the pack under the map and under the contiguous ranges (the memory plan's input).
fn cmd_bytes(args: &[String]) {
    use gb_model::families::nemotron_h_omni::inventory::Inventory;
    use gb_model::families::nemotron_h_omni::tp::TpTopology;
    use gb_model::families::nemotron_h_omni::tp_arm::serving_shard_plan_with;
    use gb_model::families::nemotron_h_omni::weights::LayerPlan;
    let pack = PathBuf::from(arg(args, "--pack").unwrap_or_else(|| die("--pack DIR")));
    let map = ExpertMap::read(Path::new(&arg(args, "--map").unwrap_or_else(|| die("--map FILE")))).unwrap_or_else(|e| die(e));
    let text = std::fs::read_to_string(pack.join("config.json")).unwrap_or_else(|e| die(format!("config.json: {e}")));
    let cfg = OmniConfig::from_json(&serde_json::from_str(&text).unwrap_or_else(|e| die(e))).unwrap_or_else(|e| die(e));
    map.check_for(&cfg, map.ranks()).unwrap_or_else(|e| die(e));
    let inv = Inventory::read(&pack).unwrap_or_else(|e| die(e));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| die(e.join("\n")));
    let arc = std::sync::Arc::new(map);
    let ranks = arc.ranks();
    let bytes = |m: Option<std::sync::Arc<ExpertMap>>, r: usize| {
        serving_shard_plan_with(TpTopology { rank: r, ranks }, &cfg, m).rank_bytes(&plan, cfg.experts, |_| true)
    };
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let (mut max_c, mut max_b) = (0u64, 0u64);
    for r in 0..ranks {
        let (c, b) = (bytes(None, r), bytes(Some(arc.clone()), r));
        max_c = max_c.max(c);
        max_b = max_b.max(b);
        println!(
            "rank {r}: contiguous {c} B ({:.3} GiB), map {b} B ({:.3} GiB), delta {:+} B",
            gib(c),
            gib(b),
            b as i64 - c as i64
        );
    }
    println!(
        "fullest rank (what the memory plan charges): contiguous {max_c} B, map {max_b} B, delta {:+} B ({:+.4} GiB)",
        max_b as i64 - max_c as i64,
        gib(max_b) - gib(max_c)
    );
}

/// The pack's config and its plan's per-expert bytes (`tp_shard::expert_bytes`).
fn pack_bytes(pack: &Path) -> (OmniConfig, ExpertBytes) {
    use gb_model::families::nemotron_h_omni::inventory::Inventory;
    use gb_model::families::nemotron_h_omni::weights::LayerPlan;
    let text = std::fs::read_to_string(pack.join("config.json")).unwrap_or_else(|e| die(format!("config.json: {e}")));
    let cfg = OmniConfig::from_json(&serde_json::from_str(&text).unwrap_or_else(|e| die(e))).unwrap_or_else(|e| die(e));
    let inv = Inventory::read(pack).unwrap_or_else(|e| die(e));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| die(e.join("\n")));
    let bytes = gb_model::families::nemotron_h_omni::tp_shard::expert_bytes(&plan, cfg.experts);
    (cfg, bytes)
}

/// The byte-balanced ownership of a mixed-bit-rate pack (`tp_balance::byte_balanced_map`): per layer the ranks' expert bytes under the
/// contiguous ranges and under the table, and the table's file when `--out` is given.
fn cmd_bytes_balance(args: &[String]) {
    let pack = PathBuf::from(arg(args, "--pack").unwrap_or_else(|| die("--pack DIR")));
    let ranks: usize = arg(args, "--ranks").unwrap_or_else(|| die("--ranks R")).parse().unwrap_or_else(|_| die("--ranks R"));
    let (cfg, t) = pack_bytes(&pack);
    let moe = moe_layers(&cfg);
    let Some(map) = byte_balanced_map(&t, ranks, &moe).unwrap_or_else(|e| die(e)) else {
        println!(
            "{}: no MoE layer mixes trellis widths ({} layers): no table, the contiguous ranges (today's plan)",
            pack.display(),
            moe.len()
        );
        return;
    };
    let contiguous = ExpertMap::contiguous(ranks, cfg.experts, moe.iter().copied()).unwrap_or_else(|e| die(e));
    for &l in &moe {
        let (n, b) =
            (t.rank_bytes(l, contiguous.row(l).unwrap_or(&[]), ranks), t.rank_bytes(l, map.row(l).unwrap_or(&[]), ranks));
        println!("layer {l:2} widths {:?}: contiguous {n:?} -> balanced {b:?}", t.widths.get(&l).cloned().unwrap_or_default());
    }
    println!(
        "all layers: contiguous {:?}, balanced {:?}; table {:016x}",
        t.rank_totals(Some(&contiguous), ranks),
        t.rank_totals(Some(&map), ranks),
        map.hash()
    );
    if let Some(out) = arg(args, "--out") {
        std::fs::write(&out, map.to_text()).unwrap_or_else(|e| die(format!("{out}: {e}")));
        println!("wrote {out} (serve it with REXL3_TP_EXPERT_MAP={out} on every rank)");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("solve") => cmd_solve(&args),
        Some("eval") => cmd_eval(&args),
        Some("shuffle") => cmd_shuffle(&args),
        Some("info") => cmd_info(&args),
        Some("bytes") => cmd_bytes(&args),
        Some("bytes-balance") => cmd_bytes_balance(&args),
        _ => die("usage: expert_balance solve|eval|shuffle|info|bytes|bytes-balance (see the file header)"),
    }
}
