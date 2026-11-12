//! `rexl3-quant bench` and `rexl3-quant parity`: run the core over an oracle dump (core::fixture), compare every
//! stored output with the dump's, and time the stages.

use crate::core::dev::{Gpu, Mat, QuantError, Result};
use crate::core::fixture::{Dump, Entry};
use crate::core::{self, Options, OutScales, Prepared, Tensor, Times};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

fn io(e: std::io::Error) -> QuantError {
    QuantError::Shape(e.to_string())
}

/// Elements that differ, and the largest difference in units in the last place (fp32 bit patterns).
pub fn ulp_diff(a: &[f32], b: &[f32]) -> (usize, u64) {
    let key = |v: f32| {
        let i = v.to_bits() as i32;
        if i < 0 {
            (i32::MIN as i64) - (i as i64)
        } else {
            i as i64
        }
    };
    let mut n = 0;
    let mut m = 0u64;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            n += 1;
            m = m.max(key(*x).abs_diff(key(*y)));
        }
    }
    (n + a.len().abs_diff(b.len()), m)
}

fn diff_count<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

/// A float scalar of the dump: its exact bit pattern (`scalar_bits`) when recorded, else the JSON number.
fn scalar(e: &Entry, name: &str) -> f64 {
    match e.meta.get("scalar_bits").and_then(|b| b.get(name)).and_then(Value::as_u64) {
        Some(b) => f64::from_bits(b),
        None => e.meta["scalars"][name].as_f64().unwrap_or(f64::NAN),
    }
}

fn options_for(meta: &Value, energy_order: bool) -> Options {
    let k = meta["K"].as_f64().unwrap_or(4.0);
    let mut o = if energy_order { Options::new(k) } else { Options::parity(k) };
    o.sigma_reg = meta["sigma_reg"].as_f64().unwrap_or(0.025);
    o.buf_rows = meta["buf_rows"].as_u64().unwrap_or(128) as usize;
    o.out_scales = match meta["out_scales"].as_str() {
        Some("always") => OutScales::Always,
        Some("never") => OutScales::Never,
        _ => OutScales::Auto,
    };
    o
}

pub struct RunConfig {
    pub energy_order: bool,
    pub refit_guard: bool,
    pub limit: Option<usize>,
    pub stages: bool,
    /// tensors quantized concurrently, each worker on its own stream and handles (1 with `stages`)
    pub workers: usize,
    pub mode: crate::gpu::cublas::MathMode,
}

/// Prepared Hessians shared read-only by the workers (device matrices; nothing mutates them after preparation).
struct Shared(BTreeMap<u64, Prepared>);
unsafe impl Sync for Shared {}

struct Done {
    row: Value,
    times: Times,
    pass: bool,
    rust: f64,
    python: f64,
}

fn opts_of(e: &Entry, cfg: &RunConfig) -> Options {
    let mut opts = options_for(&e.meta, cfg.energy_order);
    opts.refit_guard = cfg.refit_guard;
    opts
}

/// Runs every tensor of the dump; returns a JSON report (per tensor: equality of states / trellis / suh / svh, g and
/// proxy, stage times; the Python seconds from the dump).
pub fn run(g: &mut Gpu, root: &Path, cfg: &RunConfig) -> Result<Value> {
    let dump = Dump::open(root).map_err(io)?;
    let tensors: Vec<(u64, String, u64)> = dump.tensors().into_iter().take(cfg.limit.unwrap_or(usize::MAX)).collect();
    let t_all = Instant::now();
    // every Hessian once, in tensor order, on the main context
    let mut preps = BTreeMap::new();
    let mut prep_secs: BTreeMap<u64, f64> = BTreeMap::new();
    for (id, _, hid) in &tensors {
        if preps.contains_key(hid) {
            continue;
        }
        let opts = opts_of(&dump.tensor(*id).map_err(io)?, cfg);
        let h = dump.hessian(*hid).map_err(io)?;
        let hsum = h.f32s("H").map_err(io)?;
        let k = h.shape("H").map_err(io)?[0];
        let su = h.f32s("su").map_err(io)?;
        let count = h.meta["count"].as_u64().unwrap_or(0);
        let tp = Instant::now();
        preps.insert(*hid, core::prepare(g, Some(&hsum), k, count, &su, &opts)?);
        prep_secs.insert(*hid, tp.elapsed().as_secs_f64());
    }
    let shared = Shared(preps);
    let workers = if cfg.stages { 1 } else { cfg.workers.max(1) };
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut done: Vec<(usize, Result<Done>)> = Vec::new();
    if workers == 1 {
        for (i, t) in tensors.iter().enumerate() {
            done.push((i, one(g, &dump, t, &shared, &prep_secs, cfg)));
        }
    } else {
        let results = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| {
                    let mut wg = match Gpu::new(cfg.mode) {
                        Ok(w) => w,
                        Err(e) => {
                            results.lock().expect("results").push((usize::MAX, Err(e)));
                            return;
                        }
                    };
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        if i >= tensors.len() {
                            break;
                        }
                        let r = one(&mut wg, &dump, &tensors[i], &shared, &prep_secs, cfg);
                        results.lock().expect("results").push((i, r));
                    }
                });
            }
        });
        done = results.into_inner().expect("results");
        done.sort_by_key(|(i, _)| *i);
    }
    let wall = t_all.elapsed().as_secs_f64();
    let mut rows = Vec::new();
    let mut total = Times::default();
    let (mut rust, mut py) = (0.0, 0.0);
    let mut pass_all = true;
    for (_, r) in done {
        let d = r?;
        pass_all &= d.pass;
        total.merge(&d.times);
        rust += d.rust;
        py += d.python;
        rows.push(d.row);
    }
    Ok(json!({
        "dump": root.display().to_string(), "energy_order": cfg.energy_order, "refit_guard": cfg.refit_guard,
        "workers": workers, "tensors": rows.len(), "pass": pass_all,
        "seconds": {"rust": rust, "python": py, "rust_wall": wall},
        "stages": total.stages.iter().map(|(n, v)| json!([n, v])).collect::<Vec<_>>(),
        "rows": rows,
    }))
}

fn one(
    g: &mut Gpu,
    dump: &Dump,
    t: &(u64, String, u64),
    shared: &Shared,
    prep_secs: &BTreeMap<u64, f64>,
    cfg: &RunConfig,
) -> Result<Done> {
    let (id, key, hid) = (t.0, &t.1, t.2);
    let e = dump.tensor(id).map_err(io)?;
    let opts = opts_of(&e, cfg);
    let prep = &shared.0[&hid];
    // the preparation is charged to the first tensor of its Hessian, as the Python core charges it
    let first = dump.tensors().iter().find(|x| x.2 == hid).map(|x| x.0) == Some(id);
    let prep_time = if first { prep_secs[&hid] } else { 0.0 };
    let shape = e.shape("weight").map_err(io)?.to_vec();
    let (k, n) = (shape[0], shape[1]);
    let w = e.floats("weight").map_err(io)?;
    let sens = if e.has("out_sensitivity") { Some(e.f32s("out_sensitivity").map_err(io)?) } else { None };
    let sv = e.f32s("sv_signs").map_err(io)?;
    let tq = Instant::now();
    let out = core::quantize(g, prep, &Tensor { weight: &w, k, n, out_sensitivity: sens.as_deref() }, &sv, &opts)?;
    let q_time = tq.elapsed().as_secs_f64();
    let py_s = e.meta["seconds"].as_f64().unwrap_or(0.0);
    let mut times = out.times.clone();
    times.stages.insert(0, ("prepare", prep_time));

    let states = e.i16s("states").map_err(io)?;
    let trellis = e.i16s("trellis").map_err(io)?;
    let suh = e.u16s("suh", "f16").map_err(io)?;
    let svh = e.u16s("svh", "f16").map_err(io)?;
    let sc = &e.meta["scalars"];
    let g_py = scalar(&e, "g");
    let proxy_py = scalar(&e, "proxy_err");
    let rounding_py = scalar(&e, "proxy_err_rounding");
    let before_py = scalar(&e, "refit_err_before");
    let st_d = diff_count(&states, &out.states);
    let tr_d = diff_count(&trellis, &out.trellis);
    let su_d = diff_count(&suh, &out.suh);
    let sv_d = diff_count(&svh, &out.svh);
    let pass = st_d == 0
        && tr_d == 0
        && su_d == 0
        && sv_d == 0
        && g_py.to_bits() == out.g.to_bits()
        && proxy_py.to_bits() == out.proxy_err.to_bits();
    let mut row = json!({
        "id": id, "key": key, "shape": [k, n], "pass": pass,
        "states_diff": st_d, "states_total": states.len(), "trellis_diff": tr_d, "suh_diff": su_d, "svh_diff": sv_d,
        "g": [g_py, out.g], "proxy": [proxy_py, out.proxy_err], "proxy_rounding": [rounding_py, out.proxy_err_rounding],
        "proxy_rounding_equal": rounding_py.to_bits() == out.proxy_err_rounding.to_bits(),
        "refit_err_before": [before_py, out.refit_err_before],
        "refit_err_before_equal": out.refit_err_before.map(|v| v.to_bits()) == Some(before_py.to_bits()),
        "apply_out_scales": [sc["apply_out_scales"].clone(), out.apply_out_scales],
        "seconds": {"python": py_s, "rust": prep_time + q_time},
        "stages": times.stages.iter().map(|(n, v)| json!([n, v])).collect::<Vec<_>>(),
    });
    if cfg.stages {
        row["stage_checks"] = stage_checks(g, &e, prep, &opts)?;
    }
    eprintln!(
        "{id:3} {key:70} states {}/{} suh {} svh {} g {} proxy {} {:.2}s (py {:.2}s)",
        st_d,
        states.len(),
        su_d,
        sv_d,
        if g_py.to_bits() == out.g.to_bits() { "=" } else { "!=" },
        if proxy_py.to_bits() == out.proxy_err.to_bits() { "=" } else { "!=" },
        prep_time + q_time,
        py_s
    );
    Ok(Done { row, times, pass, rust: prep_time + q_time, python: py_s })
}

fn cmp(a: &[f32], b: &[f32]) -> Value {
    let (n, m) = ulp_diff(a, b);
    json!({"diff": n, "total": a.len(), "max_ulp": m})
}

/// Each stage run on the dump's own inputs to that stage (when the dump has the stage arrays).
fn stage_checks(g: &mut Gpu, e: &Entry, prep: &Prepared, opts: &Options) -> Result<Value> {
    let mut out = serde_json::Map::new();
    if !e.has("reg_w_r") {
        return Ok(Value::Object(out));
    }
    let io = |r: std::io::Result<Vec<f32>>| r.map_err(io);
    let shape = e.shape("weight").map_err(super::bench::io)?.to_vec();
    let (k, n) = (shape[0], shape[1]);
    if e.has("prep_H_rot") {
        let h = prep.h_rot.as_ref().expect("calibrated").to_host(g)?;
        out.insert("prep_H_rot".into(), cmp(&io(e.f32s("prep_H_rot"))?, &h));
        if let (Some(l), true) = (&prep.l, e.has("prep_L")) {
            out.insert("prep_L".into(), cmp(&io(e.f32s("prep_L"))?, &l.to_host(g)?));
        }
    }
    // regularize from the weight and the signs
    let w = io(e.floats("weight"))?;
    let mut wm = Mat::from_host(&w, k, n)?;
    let sens = if e.has("out_sensitivity") { Some(io(e.f32s("out_sensitivity"))?) } else { None };
    let reg = core::regularize::regularize(g, &mut wm, &prep.su, &io(e.f32s("sv_signs"))?, opts, sens.as_deref(), false)?;
    out.insert("reg_w_r".into(), cmp(&io(e.f32s("reg_w_r"))?, &wm.to_host(g)?));
    out.insert("reg_su".into(), cmp(&io(e.f32s("reg_su"))?, &reg.su));
    out.insert("reg_sv".into(), cmp(&io(e.f32s("reg_sv"))?, &reg.sv));
    // g on the dump's regularized weight
    let wr = Mat::from_host(&io(e.f32s("reg_w_r"))?, k, n)?;
    let bits = opts.bits()?;
    let gs = core::gscale::search(g, &wr, bits, opts.gscale_width, core::ldlq_drift(opts.k), opts.gscale_max_tiles)?;
    let g_py = scalar(e, "g");
    out.insert("gscale".into(), json!({"python": g_py, "rust": gs, "equal": g_py.to_bits() == gs.to_bits()}));
    if e.has("gs_samples") {
        let (s, count) = core::gscale::sample_tiles(g, &wr, opts.gscale_width, core::ldlq_drift(opts.k))?;
        let mut v = vec![0f32; count * 256];
        g.sync()?;
        s.download(&mut v)?;
        out.insert("gs_samples".into(), cmp(&io(e.f32s("gs_samples"))?, &v));
    }
    // LDLQ on the dump's rounding input and factor
    let x = Mat::from_host(&io(e.f32s("round_X"))?, k, n)?;
    let l = Mat::from_host(&io(e.f32s("round_L"))?, k, k)?;
    let order = if e.has("round_blocks") {
        Some(core::ldlq::BlockOrder {
            blocks: e.i64s("round_blocks").map_err(super::bench::io)?.into_iter().map(|v| v as usize).collect(),
        })
    } else {
        None
    };
    let (q_r, st) = core::ldlq::one_sided_ordered(g, &x, &l, order.as_ref(), bits, opts.buf_rows)?;
    out.insert("round_Q_r".into(), cmp(&io(e.f32s("round_Q_r"))?, &q_r.to_host(g)?));
    let st_host: Vec<i16> = core::dev::download(g, &st, k / 16 * (n / 16) * 256)?;
    let st_py = e.i16s("round_states").map_err(super::bench::io)?;
    out.insert("round_states".into(), json!({"diff": diff_count(&st_py, &st_host), "total": st_py.len()}));
    // reconstruction from the dump's Q_r and scales
    let su_g: Vec<f32> = {
        let inv = (1.0 / g_py) as f32;
        io(e.f32s("reg_su"))?.iter().map(|v| v * inv).collect()
    };
    let qpy = Mat::from_host(&io(e.f32s("round_Q_r"))?, k, n)?;
    let rq = core::reconstruct(g, &qpy, &su_g, &io(e.f32s("reg_sv"))?)?;
    out.insert("recon_Q".into(), cmp(&io(e.f32s("recon_Q"))?, &rq.to_host(g)?));
    // refit on the dump's reconstruction and refit H
    if e.has("refit_H") {
        let h = Mat::from_host(&io(e.f32s("refit_H"))?, k, k)?;
        if let Some(hr) = &prep.refit_h {
            out.insert("refit_H".into(), cmp(&io(e.f32s("refit_H"))?, &hr.to_host(g)?));
        }
        let wo = Mat::from_host(&w, k, n)?;
        let metric = core::refit::Metric::new(g, &h, &wo, opts.damping_share())?;
        let q = Mat::from_host(&io(e.f32s("recon_Q"))?, k, n)?;
        let r = core::refit::refit_scales(g, q, &su_g, &io(e.f32s("reg_sv"))?, &metric, opts.refit_rounds, false)?;
        out.insert("refit_Q".into(), cmp(&io(e.f32s("refit_Q"))?, &r.q.to_host(g)?));
        out.insert("refit_su".into(), cmp(&io(e.f32s("refit_su"))?, &r.su));
        out.insert("refit_sv".into(), cmp(&io(e.f32s("refit_sv"))?, &r.sv));
        let p = scalar(e, "proxy_err");
        out.insert("refit_proxy".into(), json!({"python": p, "rust": r.e_after, "equal": p.to_bits() == r.e_after.to_bits()}));
    }
    Ok(Value::Object(out))
}

#[cfg(test)]
mod tests {
    #[test]
    fn ulp_diff_counts_and_measures() {
        assert_eq!(super::ulp_diff(&[1.0, 2.0], &[1.0, 2.0]), (0, 0));
        let a = 1.0f32;
        let b = f32::from_bits(a.to_bits() + 3);
        assert_eq!(super::ulp_diff(&[a, 0.0], &[b, 0.0]), (1, 3));
        // across zero: +0 and -0 differ in bits, by 0 ulps of value distance plus the sign step
        assert_eq!(super::ulp_diff(&[f32::from_bits(1)], &[-f32::from_bits(1)]).0, 1);
        assert_eq!(super::ulp_diff(&[1.0], &[1.0, 2.0]).0, 1);
    }
}
