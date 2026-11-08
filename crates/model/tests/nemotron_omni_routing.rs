//! B7: the routing traces the balanced expert ownership is solved from (`docs/CLUSTER_N_NODE_PLAN_2026-10-11.md`, B7;
//! `crates/model/src/families/nemotron_h_omni/tp_balance.rs`).
//!
//! Runs the REAL model, single node, all 88 layers, graphs off, over a corpus of prompts and records, per MoE layer, the routed expert
//! ids of every row (`MoeScratch::download_route` through the engine's route tap, `tp_balance::tap_enable`). The windows are the
//! decode shapes: a text is prefilled in one 64-row slice and then fed in windows of 1, 2, ... 8 rows (a verify window of a draft that is
//! accepted), a chat prompt is prefilled whole and then continued greedily one row at a time (what a decode step routes). A row's
//! routing does not depend on which rows share its window, so `tp_balance` can regroup the rows afterwards.
//!
//!     REXL3_NEMO_OMNI_DIR=<pack> REXL3_NEMO_ROUTING_IDS=<dir: manifest.tsv + <name>.u32> REXL3_NEMO_ROUTING_OUT=<dir> \
//!       [REXL3_NEMO_ROUTING_BUDGET_S=720] \
//!       cargo test --release -p gb-model --test nemotron_omni_routing -- --ignored --nocapture --test-threads=1
//!
//! `manifest.tsv`: one prompt a line, `name <TAB> mode <TAB> max_rows`; `mode` is `text` (teacher forced windows over the ids) or `chat`
//! (the ids are the prompt, `max_rows` greedy rows are generated). Built by `tools/nemotron_omni/expert_routing_trace.py ids`.
//! One `<name>.rxrt` per prompt in the output directory (pack-derived data: keep it OUTSIDE git, `~/projects/nemo-routing-traces/`).
//! Written after each prompt, so a session cut by the budget keeps what it finished. Hold `/tmp/rexl3-gpu.lock`; about 70 GiB of
//! device (host) memory, 5 to 15 minutes.

use gb_engine::RowRequest;
use gb_model::families::nemotron_h_omni::backend::ModelOptions;
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::model::NemotronModel;
use gb_model::families::nemotron_h_omni::tp_balance::{self, RoutingTrace, KIND_GENERATED, KIND_PREFILL, KIND_TEXT};
use gb_model::families::nemotron_h_omni::weights::LayerPlan;
use gb_model::rows::RowBatch;
use std::path::{Path, PathBuf};
use std::time::Instant;

const SLOTS: usize = 2;
const KV_PAGES: usize = 64;
const MAX_CONTEXT: usize = 4096;
/// Candidates per row asked of a window (only the first is used, to continue a chat).
const COUNT: usize = 4;
const GIB: f64 = (1u64 << 30) as f64;

fn mem_available() -> u64 {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    text.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map_or(0, |kb| kb * 1024)
}

fn read_ids(path: &Path) -> Vec<u32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// What the route tap took: `(layer, rows, picks)` of every MoE layer a call ran.
type Tap = Vec<(usize, usize, Vec<i32>)>;

/// Runs `tokens` as one window of slot 0 at `first`, with the tap on; returns the candidates and what the tap took.
fn call(
    m: &mut NemotronModel,
    tokens: Vec<u32>,
    first: usize,
    prefill: bool,
    want: Option<usize>,
) -> (Option<gb_model::forward::Candidates>, Tap) {
    let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens, first_position: first as u64 }]);
    tp_balance::tap_enable(true);
    let c = if prefill {
        m.prefill_chunk(&b, want).unwrap_or_else(|e| panic!("prefill_chunk: {e}"))
    } else {
        let c = m.window_candidates(&b, COUNT).unwrap_or_else(|e| panic!("window_candidates: {e}"));
        m.commit(0, b.rows);
        Some(c)
    };
    m.flush().unwrap_or_else(|e| panic!("flush: {e}"));
    let tap = tp_balance::tap_take();
    tp_balance::tap_enable(false);
    (c, tap)
}

fn trace_prompt(
    m: &mut NemotronModel,
    cfg: &OmniConfig,
    name: &str,
    mode: &str,
    ids: &[u32],
    max_rows: usize,
) -> (RoutingTrace, Vec<u32>) {
    let layers = tp_balance::moe_layers(cfg);
    let mut t = RoutingTrace::new(name, cfg.experts, cfg.topk, layers);
    let mut generated: Vec<u32> = Vec::new();
    m.end_slot(0);
    m.begin_slot(0).unwrap_or_else(|e| panic!("begin_slot: {e}"));
    let n = ids.len();
    match mode {
        "text" => {
            // one 64-row prefill slice, then windows of 1..=8 rows over the rest, every window committed whole
            let p = 64.min(n);
            let (_, tap) = call(m, ids[..p].to_vec(), 0, true, None);
            t.push_window(p, KIND_PREFILL, &tap).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (mut at, mut w) = (p, 0usize);
            while at < n && (at - p) < max_rows {
                let rows = (1 + w % 8).min(n - at);
                w += 1;
                let (_, tap) = call(m, ids[at..at + rows].to_vec(), at, false, None);
                t.push_window(rows, KIND_TEXT, &tap).unwrap_or_else(|e| panic!("{name}: {e}"));
                at += rows;
            }
        }
        "chat" => {
            // the prompt in 64-row slices (the last asks for candidates), then greedy single rows
            let mut at = 0usize;
            let mut next: u32 = 0;
            while at < n {
                let end = (at + 64).min(n);
                let want = (end == n).then_some(end - at - 1);
                let (c, tap) = call(m, ids[at..end].to_vec(), at, true, want);
                t.push_window(end - at, KIND_PREFILL, &tap).unwrap_or_else(|e| panic!("{name}: {e}"));
                if let Some(c) = c {
                    next = c.ids[0] as u32;
                }
                at = end;
            }
            let mut pos = n;
            for _ in 0..max_rows {
                generated.push(next);
                let (c, tap) = call(m, vec![next], pos, false, None);
                t.push_window(1, KIND_GENERATED, &tap).unwrap_or_else(|e| panic!("{name}: {e}"));
                next = c.expect("candidates").ids[0] as u32;
                pos += 1;
            }
        }
        other => panic!("manifest: mode `{other}` is not `text` or `chat`"),
    }
    m.end_slot(0);
    (t, generated)
}

#[test]
#[ignore]
fn nemotron_routing_traces_of_the_real_model() {
    let Some(dir) = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set");
        return;
    };
    let (Some(ids_dir), Some(out_dir)) = (
        std::env::var_os("REXL3_NEMO_ROUTING_IDS").map(PathBuf::from),
        std::env::var_os("REXL3_NEMO_ROUTING_OUT").map(PathBuf::from),
    ) else {
        eprintln!("not run: REXL3_NEMO_ROUTING_IDS and REXL3_NEMO_ROUTING_OUT are not set");
        return;
    };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    let budget: f64 = std::env::var("REXL3_NEMO_ROUTING_BUDGET_S").ok().and_then(|v| v.parse().ok()).unwrap_or(720.0);
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| panic!("config.json: {e}"));
    let cfg = OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap_or_else(|e| panic!("{e}"));
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));

    // device memory is host memory: never take all of it
    let need = plan.bytes(&inv) + (4 << 30) + (12 << 30);
    let have = mem_available();
    eprintln!(
        "ROUTING: the model needs about {:.1} GiB with its margin, the host has {:.1} GiB available",
        need as f64 / GIB,
        have as f64 / GIB
    );
    if have < need {
        eprintln!("not run: {:.1} GiB needed, {:.1} GiB available", need as f64 / GIB, have as f64 / GIB);
        return;
    }
    std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| panic!("{}: {e}", out_dir.display()));
    let manifest = std::fs::read_to_string(ids_dir.join("manifest.tsv")).unwrap_or_else(|e| panic!("manifest.tsv: {e}"));

    let t0 = Instant::now();
    let opts =
        ModelOptions { slots: SLOTS, kv_pages: KV_PAGES, max_context: MAX_CONTEXT, prefill_rows: 64, layers: None, mtp: false };
    let mut m = NemotronModel::load(&dir, &cfg, &plan, opts).unwrap_or_else(|e| panic!("load: {e}"));
    m.set_graphs(false);
    eprintln!("ROUTING: loaded in {:.1} s", t0.elapsed().as_secs_f64());

    let mut done = 0usize;
    let mut rows_total = 0usize;
    for line in manifest.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 3, "manifest line `{line}`: name, mode, max_rows");
        let (name, mode, max_rows) = (f[0], f[1], f[2].parse::<usize>().expect("max_rows"));
        if t0.elapsed().as_secs_f64() > budget {
            eprintln!("ROUTING: the {budget:.0} s budget is spent; stopping before `{name}`");
            break;
        }
        let ids = read_ids(&ids_dir.join(format!("{name}.u32")));
        let t = Instant::now();
        let (trace, generated) = trace_prompt(&mut m, &cfg, name, mode, &ids, max_rows);
        trace.write(&out_dir.join(format!("{name}.rxrt"))).unwrap_or_else(|e| panic!("{e}"));
        if !generated.is_empty() {
            let b: Vec<u8> = generated.iter().flat_map(|g| g.to_le_bytes()).collect();
            std::fs::write(out_dir.join(format!("{name}.gen.u32")), b).unwrap();
        }
        let decode = trace.kind.iter().filter(|&&k| k == KIND_TEXT || k == KIND_GENERATED).count();
        let prefill = trace.kind.iter().filter(|&&k| k == KIND_PREFILL).count();
        done += 1;
        rows_total += trace.rows();
        eprintln!(
            "ROUTING: {name} ({mode}): {prefill} prefill rows, {decode} decode rows in {} windows, {:.1} s",
            trace.windows.len(),
            t.elapsed().as_secs_f64()
        );
    }
    eprintln!("ROUTING: {done} prompts, {rows_total} rows, {:.1} s", t0.elapsed().as_secs_f64());
    assert!(done > 0, "no prompt was traced");
}
