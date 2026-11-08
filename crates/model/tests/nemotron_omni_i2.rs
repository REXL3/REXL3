//! Task I2: the composed Nemotron 3.5 Super VL model, every trunk layer, free-running over the fork's two 1,024-token samples, compared
//! with the fork's dumps by the oracle harness (`families::nemotron_h_omni::oracle`) under the committed per-module limits.
//!
//! Needs `REXL3_NEMO_OMNI_DIR` (the published pack), `REXL3_NEMO_ORACLE_DIR` (the fork's `ref-text-1k` dumps) and a CUDA device; without
//! any of them it reports what it did not run and passes. Optional: `REXL3_NEMO_I2_LAYERS` (compare only the first `n` trunk layers: a
//! layer-limited model, no final norm or logits), `REXL3_NEMO_I2_TOKENS` (rows per sample, a multiple of 64; default the dump's).
//! `REXL3_NEMO_I2_LOGITS_OUT=<dir>` also writes the engine's logits (f32 little-endian `[tokens][131072]`, `engine_logits_<s>.f32`) for
//! the KL comparison against the bf16 original; the files are pack-derived and stay on the node.
//!
//! The tokens go through 64-row verify windows, each followed by `commit(slot, 64)`: every row's logits are tapped (a prefill slice only
//! computes the last row's), and windows are bit-identical to prefill slices by the composer's own gate. Sample 0 is compared at every
//! point (91 modules); sample 1 has logits only.

use gb_cuda::{DevBuf, Stream};
use gb_engine::RowRequest;
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::ModelOptions;
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::model::{NemotronModel, Point, Tap};
use gb_model::families::nemotron_h_omni::oracle::{table, Comparison, OracleTap, Tolerances, TRUNK};
use gb_model::families::nemotron_h_omni::weights::LayerPlan;
use gb_model::rows::RowBatch;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// The oracle tap, and optionally a file the logits are appended to (f32 little-endian rows).
struct Tapped {
    oracle: OracleTap,
    logits: Option<std::fs::File>,
}

impl Tap for Tapped {
    fn tap(&mut self, point: Point, buf: &DevBuf, rows: usize, width: usize, st: &Stream) -> Result<(), ModelError> {
        self.oracle.tap(point, buf, rows, width, st)?;
        if let (Point::Logits, Some(f)) = (point, self.logits.as_mut()) {
            st.sync()?;
            let mut v = vec![0f32; rows * width];
            buf.download_range(0, &mut v)?;
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            f.write_all(&bytes).map_err(|e| ModelError::Request(format!("logits file: {e}")))?;
        }
        Ok(())
    }
}

fn env_dir(name: &str) -> Option<PathBuf> {
    let v = std::env::var_os(name).map(PathBuf::from);
    if v.is_none() {
        eprintln!("not run: {name} is not set");
    }
    v
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

#[test]
fn every_layer_free_running_stays_inside_the_committed_limits_and_top1_reaches_95_percent() {
    let (Some(pack), Some(oracle)) = (env_dir("REXL3_NEMO_OMNI_DIR"), env_dir("REXL3_NEMO_ORACLE_DIR")) else { return };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let layers = env_usize("REXL3_NEMO_I2_LAYERS").unwrap_or(TRUNK).min(TRUNK);
    let full = layers == TRUNK;

    let text = std::fs::read_to_string(pack.join("config.json")).unwrap_or_else(|e| panic!("config.json: {e}"));
    let cfg = OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap_or_else(|e| panic!("{e}"));
    let inv = Inventory::read(&pack).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let opts = ModelOptions {
        slots: 1,
        kv_pages: 8,
        max_context: 2048,
        // the production shape (256-row slices: the large-row MoE prefill scratch is allocated); the windows below are 64 rows and take
        // the decode path whatever this is
        prefill_rows: 256,
        layers: if full { None } else { Some(layers) },
        // the head never changes the trunk's bits; `REXL3_NEMO_I2_MTP=1` runs the oracle with it loaded as well
        mtp: std::env::var_os("REXL3_NEMO_I2_MTP").is_some(),
    };
    let t = Instant::now();
    let mut model = NemotronModel::load(&pack, &cfg, &plan, opts).unwrap_or_else(|e| panic!("load: {e}"));
    eprintln!("loaded {layers} layers in {:.1} s", t.elapsed().as_secs_f64());

    // the points a layer-limited run is held to: the embedding and the layers it has
    let only: Vec<Point> = std::iter::once(Point::Embed).chain((0..layers).map(Point::Layer)).collect();
    let mut expect = only.clone();
    if full {
        expect.push(Point::FinalNorm);
        expect.push(Point::Logits);
    }

    let mut results: Vec<Comparison> = Vec::new();
    for sample in 0..2 {
        let mut tap = OracleTap::open(&oracle, sample).unwrap_or_else(|e| panic!("oracle sample {sample}: {e}"));
        if !full {
            tap = tap.only(only.iter().copied());
        }
        let ids = tap_ids(&oracle, sample);
        let n = env_usize("REXL3_NEMO_I2_TOKENS").unwrap_or(ids.len()).min(ids.len()) / 64 * 64;
        assert!(n > 0, "no whole 64-row windows to run");
        model.begin_slot(0).unwrap_or_else(|e| panic!("begin_slot: {e}"));
        let logits = std::env::var_os("REXL3_NEMO_I2_LOGITS_OUT").map(PathBuf::from).map(|d| {
            std::fs::create_dir_all(&d).unwrap_or_else(|e| panic!("{}: {e}", d.display()));
            std::fs::File::create(d.join(format!("engine_logits_{sample}.f32"))).unwrap_or_else(|e| panic!("logits file: {e}"))
        });
        model.set_tap(Some(Box::new(Tapped { oracle: tap.clone(), logits })));
        let t = Instant::now();
        for (w, chunk) in ids[..n].chunks(64).enumerate() {
            let b = RowBatch::from_requests(&[RowRequest { slot: 0, first_position: (w * 64) as u64, tokens: chunk.to_vec() }]);
            model.window_candidates(&b, 8).unwrap_or_else(|e| panic!("sample {sample} window {w}: {e}"));
            model.commit(0, chunk.len());
            model.flush().unwrap_or_else(|e| panic!("sample {sample} window {w}: {e}"));
        }
        eprintln!("sample {sample}: {n} tokens in {:.1} s", t.elapsed().as_secs_f64());
        model.set_tap(None);
        model.end_slot(0);
        results.extend(tap.results());
    }

    let tol = Tolerances::committed();
    eprintln!("{}", table(&results, Some(&tol)));
    let verdict = tol.apply(&results, &expect);
    if let Some((a, r)) = verdict.top1 {
        eprintln!("top-1 agreement with the fork: {a}/{r} = {:.2}%", 100.0 * a as f64 / r as f64);
    }
    for f in &verdict.failures {
        eprintln!("FAIL {f}");
    }
    assert!(verdict.passed(), "{} failures against the committed limits", verdict.failures.len());
}

fn tap_ids(oracle: &std::path::Path, sample: usize) -> Vec<u32> {
    gb_model::families::nemotron_h_omni::oracle::Dump::open(oracle, sample)
        .and_then(|d| d.ids())
        .unwrap_or_else(|e| panic!("ids of sample {sample}: {e}"))
}
