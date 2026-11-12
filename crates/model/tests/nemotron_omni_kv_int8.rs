//! The quality gate of the int8 KV cache (`--kv-dtype int8`, `docs/NEMOTRON_KV_INT8.md`) on the real model: the logits of the SAME
//! teacher-forced tokens at depth with an fp16 and with an int8 cache, for `tools/nemotron_omni/kv8_compare.py` (KL and argmax).
//!
//! One process per run, one model at a time: for each dtype of `REXL3_KV8_DTYPES` (default `fp16,int8`; the process sets
//! `REXL3_KV_DTYPE` before each load) the model prefills the ids in 256-row slices; at each checkpoint `c` of `REXL3_KV8_CHECKPOINTS`
//! (default `2048,32768,131072,262144,524288`) it runs ONE probe window of the next `REXL3_KV8_ROWS` (64) tokens `ids[c..c + 64]`
//! (teacher forced, positions `c..c + 63`), writes the window's logits `[64][131072]` f32 to `logits_<dtype>_<c>.f32`, keeps the
//! window's first row (`commit(slot, 1)`) and prefills on from `c + 1`. Both dtypes see exactly the same calls, so the only difference
//! is the cache. At the last checkpoint it times `REXL3_KV8_DECODE_STEPS` (default 32) one-row steps (teacher forced): the real-text
//! decode cost at depth of each dtype.
//!
//! Also, with `REXL3_KV8_REPEAT=<c>` (one of the checkpoints): the LAST dtype is loaded again and run to `c`, and its logits must equal
//! the first run's bit for bit (run-to-run determinism). With `REXL3_KV8_DUMP_KV=<n>`: after the fp16 run's first checkpoint at or
//! above `n`, the fp16 K and V of every attention layer for positions `0..n` go to `kv_f16_l<ord>.u16` (`[n][2][128]` fp16 bits): the
//! real K and V the scale-granularity analysis reads (`tools/nemotron_omni/kv8_scales.py`).
//!
//! Env: `REXL3_NEMO_OMNI_DIR` (the pack), `REXL3_KV8_IDS` (little-endian u32 token ids, at least the last checkpoint + 64),
//! `REXL3_KV8_OUT` (pack-derived output: it stays on the node). Without them the test says so and passes. On a GB10 with the device
//! lock held and a model window (about 64 GiB resident):
//!
//!     REXL3_NEMO_OMNI_DIR=$PACK REXL3_KV8_IDS=$IDS REXL3_KV8_OUT=$OUT \
//!       cargo test --release -p gb-model --test nemotron_omni_kv_int8 -- --ignored --nocapture --test-threads=1
#![allow(clippy::needless_range_loop)]

use gb_cuda::{DevBuf, Stream};
use gb_engine::RowRequest;
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::ModelOptions;
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::model::{NemotronModel, Point, Tap};
use gb_model::families::nemotron_h_omni::weights::LayerPlan;
use gb_model::rows::RowBatch;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const SLICE: usize = 256;

/// Writes the logits of the probe window to the open file, if any.
struct Probe {
    file: std::sync::Arc<std::sync::Mutex<Option<std::fs::File>>>,
}

impl Tap for Probe {
    fn tap(&mut self, point: Point, buf: &DevBuf, rows: usize, width: usize, st: &Stream) -> Result<(), ModelError> {
        if point != Point::Logits {
            return Ok(());
        }
        let mut g = self.file.lock().unwrap();
        let Some(f) = g.as_mut() else { return Ok(()) };
        st.sync()?;
        let mut v = vec![0f32; rows * width];
        buf.download_range(0, &mut v)?;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        f.write_all(&bytes).map_err(|e| ModelError::Request(format!("logits file: {e}")))?;
        Ok(())
    }
}

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

struct Run<'a> {
    pack: &'a std::path::Path,
    cfg: &'a OmniConfig,
    plan: &'a LayerPlan,
    ids: &'a [u32],
    out: &'a std::path::Path,
    rows: usize,
}

impl Run<'_> {
    /// Loads the model with the KV dtype `dtype`, prefills to each checkpoint up to `last`, writes the probe logits under `tag`.
    /// Returns the decode step times at the last checkpoint.
    fn go(&self, dtype: &str, tag: &str, checkpoints: &[usize], decode_steps: usize, dump_kv: Option<usize>) -> Vec<f64> {
        std::env::set_var("REXL3_KV_DTYPE", dtype);
        let last = *checkpoints.last().unwrap();
        let max_context = (last + self.rows + decode_steps + SLICE).next_multiple_of(SLICE);
        let opts = ModelOptions {
            slots: 1,
            kv_pages: max_context / 256 + 1,
            max_context,
            prefill_rows: SLICE,
            layers: None,
            mtp: false,
        };
        let t = Instant::now();
        let mut m = NemotronModel::load(self.pack, self.cfg, self.plan, opts).unwrap_or_else(|e| panic!("load ({dtype}): {e}"));
        eprintln!("[{tag}] loaded with REXL3_KV_DTYPE={dtype} in {:.1} s", t.elapsed().as_secs_f64());
        let file = std::sync::Arc::new(std::sync::Mutex::new(None));
        m.set_tap(Some(Box::new(Probe { file: file.clone() })));
        m.begin_slot(0).unwrap();
        let mut at = 0usize;
        let t0 = Instant::now();
        let mut dumped = false;
        let mut decode_ms = Vec::new();
        for (ci, &c) in checkpoints.iter().enumerate() {
            let tc = Instant::now();
            while at < c {
                let end = (at + SLICE).min(c);
                let b = RowBatch::from_requests(&[RowRequest {
                    slot: 0,
                    first_position: at as u64,
                    tokens: self.ids[at..end].to_vec(),
                }]);
                m.prefill_chunk(&b, None).unwrap_or_else(|e| panic!("[{tag}] prefill at {at}: {e}"));
                at = end;
            }
            m.flush().unwrap();
            let prefill_s = tc.elapsed().as_secs_f64();
            let path = self.out.join(format!("logits_{tag}_{c}.f32"));
            *file.lock().unwrap() = Some(std::fs::File::create(&path).unwrap());
            let b = RowBatch::from_requests(&[RowRequest {
                slot: 0,
                first_position: c as u64,
                tokens: self.ids[c..c + self.rows].to_vec(),
            }]);
            m.window_candidates(&b, 8).unwrap_or_else(|e| panic!("[{tag}] probe window at {c}: {e}"));
            m.flush().unwrap();
            *file.lock().unwrap() = None;
            let len = std::fs::metadata(&path).unwrap().len();
            assert_eq!(len, (self.rows * 131_072 * 4) as u64, "[{tag}] logits at {c}");
            m.commit(0, 1);
            m.flush().unwrap();
            at = c + 1;
            eprintln!(
                "[{tag}] checkpoint {c}: prefill {prefill_s:.1} s ({:.0} tok/s), total {:.1} s, probe logits {}",
                (c - checkpoints.get(ci.wrapping_sub(1)).copied().unwrap_or(0)) as f64 / prefill_s,
                t0.elapsed().as_secs_f64(),
                path.display()
            );
            if let (Some(n), false) = (dump_kv, dumped) {
                if c >= n {
                    for ord in 0..8 {
                        let (k, v) = m.debug_kv_rows(ord, 0, 0, n).unwrap();
                        let mut bytes = Vec::with_capacity(k.len() * 4);
                        bytes.extend(k.iter().flat_map(|x| x.to_le_bytes()));
                        bytes.extend(v.iter().flat_map(|x| x.to_le_bytes()));
                        std::fs::write(self.out.join(format!("kv_f16_l{ord}.u16")), bytes).unwrap();
                    }
                    eprintln!("[{tag}] dumped K and V of 8 layers, positions 0..{n}");
                    dumped = true;
                }
            }
        }
        // decode at the last depth: one-row steps, teacher forced (the ids after the probe), each committed
        for i in 0..decode_steps {
            let p = at + i;
            let b = RowBatch::from_requests(&[RowRequest { slot: 0, first_position: p as u64, tokens: vec![self.ids[p]] }]);
            let t = Instant::now();
            m.window_candidates(&b, 1).unwrap_or_else(|e| panic!("[{tag}] decode step {i}: {e}"));
            m.commit(0, 1);
            m.flush().unwrap();
            decode_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        m.set_tap(None);
        m.end_slot(0);
        decode_ms
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        f64::NAN
    } else {
        v[v.len() / 2]
    }
}

#[test]
#[ignore]
fn int8_and_fp16_kv_logits_at_depth() {
    let (Some(pack), Some(ids_path), Some(out)) = (
        std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from),
        std::env::var_os("REXL3_KV8_IDS").map(PathBuf::from),
        std::env::var_os("REXL3_KV8_OUT").map(PathBuf::from),
    ) else {
        eprintln!("REXL3_NEMO_OMNI_DIR, REXL3_KV8_IDS and REXL3_KV8_OUT are not all set: not run");
        return;
    };
    std::fs::create_dir_all(&out).unwrap();
    let raw = std::fs::read(&ids_path).unwrap();
    let ids: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut checkpoints: Vec<usize> =
        env_list("REXL3_KV8_CHECKPOINTS", "2048,32768,131072,262144,524288").iter().map(|s| s.parse().unwrap()).collect();
    checkpoints.sort();
    let rows: usize = std::env::var("REXL3_KV8_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let decode_steps: usize = std::env::var("REXL3_KV8_DECODE_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    let need = checkpoints.last().unwrap() + rows + decode_steps + 1;
    assert!(ids.len() >= need, "{} ids, the last checkpoint needs {need}", ids.len());
    let text = std::fs::read_to_string(pack.join("config.json")).unwrap();
    let cfg = OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap_or_else(|e| panic!("{e}"));
    let inv = Inventory::read(&pack).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let run = Run { pack: &pack, cfg: &cfg, plan: &plan, ids: &ids, out: &out, rows };
    let dump_kv = std::env::var("REXL3_KV8_DUMP_KV").ok().and_then(|v| v.parse().ok());
    let dtypes = env_list("REXL3_KV8_DTYPES", "fp16,int8");
    let mut summary = String::from("{");
    for (i, d) in dtypes.iter().enumerate() {
        // `REXL3_KV8_TAG` names the files instead of the dtype (several runs into one directory)
        let tag = std::env::var("REXL3_KV8_TAG").unwrap_or_else(|_| d.clone());
        let mut ms = run.go(d, &tag, &checkpoints, decode_steps, if d == "fp16" { dump_kv } else { None });
        let med = median(&mut ms);
        eprintln!("[{d}] decode at depth {}: median {med:.2} ms over {} one-row steps", checkpoints.last().unwrap(), ms.len());
        summary.push_str(&format!("{}\"decode_ms_{d}\": {med:.3}", if i > 0 { ", " } else { "" }));
    }
    if let Some(c) = std::env::var("REXL3_KV8_REPEAT").ok().and_then(|v| v.parse::<usize>().ok()) {
        let d = dtypes.last().unwrap();
        let cps: Vec<usize> = checkpoints.iter().copied().filter(|&x| x <= c).collect();
        run.go(d, &format!("{d}_repeat"), &cps, 0, None);
        for &x in &cps {
            let a = std::fs::read(out.join(format!("logits_{d}_{x}.f32"))).unwrap();
            let b = std::fs::read(out.join(format!("logits_{d}_repeat_{x}.f32"))).unwrap();
            assert!(a == b, "{d}: the logits at {x} differ between two runs");
            eprintln!("[{d}] run-to-run: the probe logits at {x} are bit-identical");
        }
        summary.push_str(&format!(", \"repeat_bit_identical_to\": {c}"));
    }
    summary.push('}');
    std::fs::write(out.join("run_summary.json"), summary).unwrap();
}
