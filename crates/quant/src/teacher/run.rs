//! `rexl3-quant capture`: the first N decoder layers captured against the quantized prefix (layers < l quantized
//! before layer l is captured), with the H sums written per module and, given an oracle directory, compared bit for
//! bit with the converter oracle's capture (H, row counts, routing counts, the stream entering each layer).

use super::prefix::{quantize_layer, Hessians};
use super::{Capture, Options, Rows, Teacher};
use crate::core::dev::{Gpu, QuantError, Result};
use crate::gpu::cublas::MathMode;
use crate::pack::{write_safetensors, Part};
use crate::source::Source;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct RunConfig {
    pub model: PathBuf,
    pub rows: PathBuf,
    pub cal_rows: Option<usize>,
    pub cal_cols: Option<usize>,
    pub batch_rows: usize,
    pub layers: usize,
    pub bits: f64,
    pub out: PathBuf,
    pub oracle: Option<PathBuf>,
    pub mode: MathMode,
    pub opts: Options,
}

fn err(e: impl std::fmt::Display) -> QuantError {
    QuantError::Shape(e.to_string())
}

fn write_hessians(dir: &Path, h: &Hessians, counts: &crate::capture::Counts, layer: usize) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(err)?;
    let parts: Vec<Part> = h
        .iter()
        .map(|(q, (k, v, _))| Part {
            name: q.clone(),
            dtype: "F32",
            shape: vec![*k, *k],
            bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        })
        .collect();
    write_safetensors(&dir.join("hessians.safetensors"), &parts, None).map_err(err)?;
    let qm: serde_json::Map<String, Value> = counts
        .rows
        .iter()
        .map(|(q, (r, d))| (q.clone(), json!({"count": r, "dropped": d, "k": h.get(q).map(|x| x.0)})))
        .collect();
    let info = json!({"module": layer + 1, "qmaps": qm, "routing_counts": counts.routing.get(&layer)});
    std::fs::write(dir.join("capture.json"), serde_json::to_string_pretty(&info).map_err(err)?).map_err(err)
}

/// Differing elements (by bits) and the largest absolute difference.
fn diff(a: &[f32], b: &[f32]) -> (usize, f64) {
    let mut n = 0;
    let mut m = 0f64;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            n += 1;
            m = m.max((*x as f64 - *y as f64).abs());
        }
    }
    (n + a.len().abs_diff(b.len()), m)
}

fn compare_module(dir: &Path, h: &Hessians, counts: &crate::capture::Counts, layer: usize) -> Result<Value> {
    let src = Source::open(&dir.join("hessians.safetensors")).map_err(err)?;
    let info: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("capture.json")).map_err(err)?).map_err(err)?;
    let mut rows = Vec::new();
    let mut identical = 0;
    for (q, (_, v, count)) in h {
        let theirs = src.f32(q).map_err(err)?;
        let (nd, md) = diff(v, &theirs);
        let oc = info["qmaps"][q]["count"].as_u64();
        identical += (nd == 0 && oc == Some(*count)) as usize;
        rows.push(json!({"qmap": q, "differing": nd, "max_abs": md, "count": count, "oracle_count": oc}));
    }
    let names: Vec<&String> = src.names().collect();
    let missing: Vec<&&String> = names.iter().filter(|n| !h.contains_key(n.as_str())).collect();
    let routing_ok = info["routing_counts"].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect::<Vec<_>>())
        == counts.routing.get(&layer).cloned();
    Ok(json!({
        "module": layer + 1, "qmaps": h.len(), "identical": identical, "oracle_only": missing,
        "routing_identical": routing_ok, "per_qmap": rows,
    }))
}

fn compare_stream(dir: &Path, hq: &[crate::core::dev::Mat], g: &Gpu) -> Result<Value> {
    let src = Source::open(&dir.join("stream_in.safetensors")).map_err(err)?;
    let mut nd = 0usize;
    let mut md = 0f64;
    for (i, m) in hq.iter().enumerate() {
        let theirs = src.f32(&format!("b{i}")).map_err(err)?;
        let (a, b) = diff(&m.to_host(g)?, &theirs);
        nd += a;
        md = md.max(b);
    }
    Ok(json!({"differing": nd, "max_abs": md}))
}

pub fn run(cfg: &RunConfig) -> Result<Value> {
    let mut g = Gpu::new(cfg.mode)?;
    let t = Teacher::open(&cfg.model, cfg.opts.clone())?;
    let rows = Rows::load(&cfg.rows, cfg.cal_rows, cfg.cal_cols)?;
    let max_len = rows.ids.iter().map(Vec::len).max().unwrap_or(1);
    let batches = rows.batches(cfg.batch_rows * max_len);
    let t0 = Instant::now();
    let mut hq = t.embed(&g, &batches)?;
    let mut report = vec![];
    let mut times = BTreeMap::new();
    times.insert("embed".to_string(), t0.elapsed().as_secs_f64());
    for l in 0..cfg.layers.min(t.cfg.layers) {
        let mdir = cfg.out.join(format!("m{}", l + 1));
        let odir = cfg.oracle.as_ref().map(|o| o.join(format!("m{}", l + 1)));
        let stream = match &odir {
            Some(d) if d.join("stream_in.safetensors").exists() => Some(compare_stream(d, &hq, &g)?),
            _ => None,
        };
        let c0 = Instant::now();
        let layer = t.load_layer(&g, l, BTreeMap::new())?;
        let load_s = c0.elapsed().as_secs_f64();
        let c1 = Instant::now();
        let mut cap = Capture::default();
        for (b, h) in batches.iter().zip(&hq) {
            t.capture(&g, &layer, h, b, &mut cap)?;
        }
        let counts = cap.counts.clone();
        let hess = cap.finish(&g)?;
        let capture_s = c1.elapsed().as_secs_f64();
        write_hessians(&mdir, &hess, &counts, l)?;
        let cmp = match &odir {
            Some(d) if d.join("hessians.safetensors").exists() => Some(compare_module(d, &hess, &counts, l)?),
            _ => None,
        };
        let mut rep = json!({"module": l + 1, "load_s": load_s, "capture_s": capture_s, "stream_in": stream, "oracle": cmp});
        if l + 1 < cfg.layers {
            drop(layer);
            let q0 = Instant::now();
            let qs = quantize_layer(&mut g, &t, l, &hess, cfg.bits, (l + 1) as u64)?;
            let quantize_s = q0.elapsed().as_secs_f64();
            let specs = super::config::targets(&t.cfg, l);
            let deq = qs.into_iter().map(|q| (q.key, q.weight)).collect();
            let a0 = Instant::now();
            let ov = t.overrides(&g, l, &specs, deq)?;
            let lq = t.load_layer(&g, l, ov)?;
            for (i, b) in batches.iter().enumerate() {
                hq[i] = t.forward(&g, &lq, &hq[i], b)?;
            }
            rep["quantize_s"] = json!(quantize_s);
            rep["advance_s"] = json!(a0.elapsed().as_secs_f64());
        }
        eprintln!(" -- module {}: {}", l + 1, serde_json::to_string(&rep).unwrap_or_default());
        report.push(rep);
    }
    Ok(json!({"model": cfg.model, "batches": batches.len(), "rows": rows.ids.len(), "bits": cfg.bits,
              "capture_gemm": format!("{:?}", cfg.opts.capture_gemm), "modules": report, "times": times}))
}
