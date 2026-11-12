//! The teacher: a layer-at-a-time forward of the source model on the device, streamed from the checkpoint, and the
//! capture pass that accumulates each qmap's H (see `capture`).
//!
//! The forward follows the converter's Qwen3.5 adapter op for op: GEMMs through cuBLAS in the call forms of
//! [`ops`], element-wise fp32 as PyTorch computes it, and every other reduction as an fp64 pairwise tree. The GDN
//! runs token by token in fp64 and attention in fp64, each rounding its output to fp32 (the forward parity contract
//! of the converter oracle's harness).
//!
//! A capture forward runs a layer from its source weights on the quantized-prefix stream: routed experts are replaced
//! by every expert on every token (each expert's down input feeding its own H) and the router only counts its top-k
//! hits. An advance forward runs a layer (with quantized weights in place) with real routing to produce the next
//! layer's input.

pub mod config;
pub mod ops;
pub mod prefix;
pub mod run;

use crate::capture::{Counts, DevPartial, Stream};
use crate::core::dev::{download, upload, Gpu, Mat, QuantError, Result};
use crate::source::Source;
use config::{Config, Kind, Target};
use gb_cuda::rt::DevBuf;
use std::collections::BTreeMap;

/// How the all-experts capture issues its gate/up GEMMs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureGemm {
    /// one strided-batched GEMM per expert chunk with the input shared (batch stride 0), as torch.matmul issues it
    Batched,
    /// one GEMM per expert chunk with the chunk's experts side by side in the output
    Wide,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub expert_chunk: usize,
    pub capture_gemm: CaptureGemm,
}

impl Default for Options {
    fn default() -> Self {
        Options { expert_chunk: 16, capture_gemm: CaptureGemm::Wide }
    }
}

fn shape(msg: String) -> QuantError {
    QuantError::Shape(msg)
}

fn io(e: std::io::Error) -> QuantError {
    QuantError::Shape(e.to_string())
}

/// A source tensor on the device as fp32 (rows = first dimension or 1, cols = the rest).
pub fn load_f32(g: &Gpu, src: &Source, name: &str) -> Result<Mat> {
    let raw = src.raw(name).map_err(io)?;
    let n: usize = raw.shape.iter().product::<usize>().max(1);
    let rows = if raw.shape.len() >= 2 { raw.shape[0] } else { 1 };
    let cols = n / rows.max(1);
    let m = Mat::alloc(rows, cols)?;
    match raw.dtype.as_str() {
        "BF16" => {
            let b = DevBuf::from_slice(&raw.bytes)?;
            ops::bf16_to_f32(g, m.ptr(), b.ptr() as *const u16, n)?;
            g.sync()?;
        }
        "F32" | "F16" => return Mat::from_host(&raw.to_f32().map_err(io)?, rows, cols),
        d => return Err(shape(format!("{name}: dtype {d} is not read by the teacher"))),
    }
    Ok(m)
}

/// Calibration rows: `input_ids` (rows, cols) and optional `lengths` of a safetensors file, cropped as the converter
/// crops them, grouped into batches of equal-length rows.
pub struct Rows {
    pub ids: Vec<Vec<i32>>,
}

/// One batch: `rows` rows of `len` tokens (row-major ids).
pub struct Batch {
    pub rows: usize,
    pub len: usize,
    pub ids: Vec<i32>,
}

impl Rows {
    pub fn load(path: &std::path::Path, rows: Option<usize>, cols: Option<usize>) -> Result<Rows> {
        let src = Source::open(path).map_err(io)?;
        let ids = src.raw("input_ids").map_err(io)?;
        if ids.shape.len() != 2 || ids.dtype != "I64" {
            return Err(shape(format!("input_ids is {} {:?}, expected I64 (rows, cols)", ids.dtype, ids.shape)));
        }
        let (r, c) = (ids.shape[0], ids.shape[1]);
        let vals: Vec<i64> = ids.bytes.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect();
        let lengths: Option<Vec<i64>> = if src.has("lengths") {
            let l = src.raw("lengths").map_err(io)?;
            Some(l.bytes.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect())
        } else {
            None
        };
        let n = rows.map_or(r, |x| x.min(r));
        let width = cols.map_or(c, |x| x.min(c));
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let m = lengths.as_ref().map_or(width, |l| width.min(l[i].max(0) as usize));
            if m < 1 {
                return Err(shape(format!("row {i} is empty")));
            }
            out.push(vals[i * c..i * c + m].iter().map(|v| *v as i32).collect());
        }
        Ok(Rows { ids: out })
    }

    /// Groups of equal-length rows of at most `batch_tokens` tokens (at least one row), in row order within a length,
    /// lengths in order of first appearance.
    pub fn batches(&self, batch_tokens: usize) -> Vec<Batch> {
        let mut order: Vec<usize> = Vec::new();
        let mut by_len: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, r) in self.ids.iter().enumerate() {
            let e = by_len.entry(r.len()).or_default();
            if e.is_empty() {
                order.push(r.len());
            }
            e.push(i);
        }
        let mut out = Vec::new();
        for len in order {
            let idx = &by_len[&len];
            let per = (batch_tokens / len).max(1);
            for chunk in idx.chunks(per) {
                out.push(Batch {
                    rows: chunk.len(),
                    len,
                    ids: chunk.iter().flat_map(|i| self.ids[*i].iter().copied()).collect(),
                });
            }
        }
        out
    }
}

/// Captured H per qmap: (k, H sum (k x k, fp32), row count).
pub type Hessians = BTreeMap<String, (usize, Vec<f32>, u64)>;

/// The H accumulators of one layer's capture.
#[derive(Default)]
pub struct Capture {
    pub streams: BTreeMap<String, Stream<DevPartial>>,
    pub k: BTreeMap<String, usize>,
    pub counts: Counts,
}

impl Capture {
    /// Adds x^T x of x (rows x k) to qmap's tree as one leaf (rows with a non-finite value are left out and counted).
    pub(crate) fn add(&mut self, g: &Gpu, qmap: &str, x: *const f32, rows: usize, k: usize) -> Result<()> {
        let bad = DevBuf::zeroed(4)?;
        ops::nonfinite_rows(g, bad.ptr() as *mut u32, x, rows, k)?;
        let nbad = download::<u32>(g, &bad, 1)?[0] as usize;
        let c = Mat::alloc(k, k)?;
        let kept;
        if nbad == 0 {
            unsafe { ops::gram(g, x, rows, k, c.ptr())? };
            kept = rows;
        } else {
            let tmp = Mat::alloc(rows, k)?;
            crate::core::dev::copy2d(g, tmp.ptr(), k, x, k, rows, k)?;
            let host = tmp.to_host(g)?;
            let keep: Vec<f32> = host.chunks_exact(k).filter(|r| r.iter().all(|v| v.is_finite())).flatten().copied().collect();
            kept = keep.len() / k;
            if kept > 0 {
                let d = DevBuf::from_slice(&keep)?;
                unsafe { ops::gram(g, d.ptr() as *const f32, kept, k, c.ptr())? };
                g.sync()?;
            }
        }
        self.counts.add_rows(qmap, kept as u64, (rows - kept) as u64);
        if kept == 0 {
            return Ok(());
        }
        if let Some(k0) = self.k.insert(qmap.to_string(), k) {
            if k0 != k {
                return Err(shape(format!("{qmap}: inputs of width {k} and {k0}")));
            }
        }
        let leaf = DevPartial::from_f32(c.ptr(), k * k, &g.stream)?;
        self.streams.entry(qmap.to_string()).or_default().push(leaf);
        g.sync()?;
        Ok(())
    }

    /// Each qmap's H (k x k, fp32 host) and its row count.
    pub fn finish(self, g: &Gpu) -> Result<Hessians> {
        let mut out = BTreeMap::new();
        for (q, s) in self.streams {
            let k = self.k[&q];
            let root = s.finish().ok_or_else(|| shape(format!("{q}: no rows")))?;
            let h = Mat::alloc(k, k)?;
            root.round_into(h.ptr());
            let v = h.to_host(g)?;
            out.insert(q.clone(), (k, v, self.counts.rows.get(&q).map_or(0, |r| r.0)));
        }
        Ok(out)
    }
}

/// The teacher's state for one model.
pub struct Teacher {
    pub cfg: Config,
    pub src: Source,
    pub opts: Options,
    inv_freq: DevBuf,
}

/// One layer's weights on the device, by checkpoint name.
pub struct Layer {
    pub i: usize,
    w: BTreeMap<String, Mat>,
}

impl Layer {
    fn get(&self, name: &str) -> Result<&Mat> {
        self.w.get(name).ok_or_else(|| shape(format!("layer {}: no weight {name}", self.i)))
    }
}

impl Teacher {
    pub fn open(model: &std::path::Path, opts: Options) -> Result<Teacher> {
        let mut cfg = Config::load(model).map_err(shape)?;
        let src = Source::open(model).map_err(io)?;
        if !src.has(&format!("{}.embed_tokens.weight", cfg.prefix)) && src.has("model.embed_tokens.weight") {
            cfg.prefix = "model".into();
        }
        let inv_freq = upload(&cfg.inv_freq())?;
        Ok(Teacher { cfg, src, opts, inv_freq })
    }

    /// The embedding of each batch (rows * len x hidden).
    pub fn embed(&self, g: &Gpu, batches: &[Batch]) -> Result<Vec<Mat>> {
        let name = format!("{}.embed_tokens.weight", self.cfg.prefix);
        let raw = self.src.raw(&name).map_err(io)?;
        if raw.dtype != "BF16" {
            return Err(shape(format!("{name}: {} (BF16 expected)", raw.dtype)));
        }
        let h = self.cfg.hidden;
        let table = DevBuf::from_slice(&raw.bytes)?;
        let mut out = Vec::new();
        for b in batches {
            let ids = upload(&b.ids)?;
            let m = Mat::alloc(b.ids.len(), h)?;
            ops::embed_bf16(g, m.ptr(), table.ptr() as *const u16, ids.ptr() as *const i32, b.ids.len(), h)?;
            g.sync()?;
            out.push(m);
        }
        Ok(out)
    }

    /// Layer i's weights from the checkpoint, with `overrides` (name -> fp32 device tensor) in place of the source.
    pub fn load_layer(&self, g: &Gpu, i: usize, mut overrides: BTreeMap<String, Mat>) -> Result<Layer> {
        let p = format!("{}.", self.cfg.lp(i));
        let names: Vec<String> = self.src.names().filter(|n| n.starts_with(&p)).cloned().collect();
        let mut w = BTreeMap::new();
        for n in names {
            let m = match overrides.remove(&n) {
                Some(m) => m,
                None => load_f32(g, &self.src, &n)?,
            };
            w.insert(n, m);
        }
        if let Some(n) = overrides.keys().next() {
            return Err(shape(format!("override {n} is not a tensor of layer {i}")));
        }
        Ok(Layer { i, w })
    }

    /// The source weight of a target, (out x in) fp32 row-major on the host.
    pub fn target_weight(&self, t: &Target) -> Result<Vec<f32>> {
        match t.expert {
            None => self.src.f32(&format!("{}.weight", t.key)).map_err(io),
            Some(e) => {
                let m = format!("{}.mlp.experts", self.cfg.lp(t.layer));
                let i = self.cfg.moe_inter;
                if t.role == "expert.down" {
                    self.src.raw_rows(&format!("{m}.down_proj"), e, 1).and_then(|r| r.to_f32()).map_err(io)
                } else {
                    let gu = self.src.raw_rows(&format!("{m}.gate_up_proj"), e, 1).and_then(|r| r.to_f32()).map_err(io)?;
                    let h = gu.len() / (2 * i);
                    Ok(if t.role == "expert.gate" { gu[..i * h].to_vec() } else { gu[i * h..].to_vec() })
                }
            }
        }
    }

    fn rms(&self, g: &Gpu, x: &Mat, w: &Mat) -> Result<Mat> {
        let out = Mat::alloc(x.rows, x.cols)?;
        ops::rmsnorm(g, out.ptr(), x.cols, x.ptr(), x.cols, w.ptr(), None, x.rows, x.cols, self.cfg.eps as f32)?;
        Ok(out)
    }

    fn linear(&self, g: &Gpu, x: &Mat, w: &Mat) -> Result<Mat> {
        if w.cols != x.cols {
            return Err(shape(format!("linear: input {} wide, weight {} x {}", x.cols, w.rows, w.cols)));
        }
        let y = Mat::alloc(x.rows, w.rows)?;
        unsafe { ops::linear(g, x.ptr(), x.rows, x.cols, w.ptr(), w.rows, y.ptr())? };
        Ok(y)
    }

    fn cap(&self, g: &Gpu, cap: &mut Option<&mut Capture>, qmap: String, x: &Mat) -> Result<()> {
        if let Some(c) = cap.as_deref_mut() {
            c.add(g, &qmap, x.ptr(), x.rows, x.cols)?;
        }
        Ok(())
    }

    fn gdn(&self, g: &Gpu, l: &Layer, x: &Mat, b: &Batch, cap: &mut Option<&mut Capture>) -> Result<Mat> {
        let c = &self.cfg;
        let p = format!("{}.linear_attn", c.lp(l.i));
        let rows = x.rows;
        self.cap(g, cap, format!("{}.attn_in", c.lp(l.i)), x)?;
        let qkv = self.linear(g, x, l.get(&format!("{p}.in_proj_qkv.weight"))?)?;
        let z = self.linear(g, x, l.get(&format!("{p}.in_proj_z.weight"))?)?;
        let bb = self.linear(g, x, l.get(&format!("{p}.in_proj_b.weight"))?)?;
        let aa = self.linear(g, x, l.get(&format!("{p}.in_proj_a.weight"))?)?;
        let cd = c.conv_dim();
        let conv = Mat::alloc(rows, cd)?;
        ops::conv_silu(g, conv.ptr(), qkv.ptr(), l.get(&format!("{p}.conv1d.weight"))?.ptr(), b.rows, b.len, cd, c.conv_k)?;
        drop(qkv);
        let hv = c.lv_heads;
        let gg = Mat::alloc(rows, hv)?;
        let beta = Mat::alloc(rows, hv)?;
        ops::gdn_gates(
            g,
            gg.ptr(),
            beta.ptr(),
            aa.ptr(),
            bb.ptr(),
            l.get(&format!("{p}.A_log"))?.ptr(),
            l.get(&format!("{p}.dt_bias"))?.ptr(),
            rows,
            hv,
        )?;
        let (hk, dk, dv) = (c.lk_heads, c.lk_dim, c.lv_dim);
        let q64 = DevBuf::alloc(rows * hk * dk * 8)?;
        let k64 = DevBuf::alloc(rows * hk * dk * 8)?;
        ops::l2norm_f64(g, q64.ptr() as *mut f64, conv.ptr(), cd, 0, rows, hk, dk, 1e-6, (dk as f64).powf(-0.5))?;
        ops::l2norm_f64(g, k64.ptr() as *mut f64, conv.ptr(), cd, c.key_dim(), rows, hk, dk, 1e-6, 1.0)?;
        let o = Mat::alloc(rows, hv * dv)?;
        ops::gdn_f64(
            g,
            o.ptr(),
            q64.ptr() as *const f64,
            k64.ptr() as *const f64,
            conv.ptr(),
            cd,
            2 * c.key_dim(),
            gg.ptr(),
            beta.ptr(),
            [b.rows, b.len, hk, hv, dk, dv],
        )?;
        drop((q64, k64, conv));
        let on = Mat::alloc(rows, hv * dv)?;
        ops::rmsnorm(
            g,
            on.ptr(),
            dv,
            o.ptr(),
            dv,
            l.get(&format!("{p}.norm.weight"))?.ptr(),
            Some((z.ptr(), dv)),
            rows * hv,
            dv,
            c.eps as f32,
        )?;
        self.cap(g, cap, format!("{}.attn_o", c.lp(l.i)), &on)?;
        self.linear(g, &on, l.get(&format!("{p}.out_proj.weight"))?)
    }

    fn attn(&self, g: &Gpu, l: &Layer, x: &Mat, b: &Batch, cap: &mut Option<&mut Capture>) -> Result<Mat> {
        let c = &self.cfg;
        let p = format!("{}.self_attn", c.lp(l.i));
        let rows = x.rows;
        let (hd, nq, nkv) = (c.head_dim, c.n_q, c.n_kv);
        self.cap(g, cap, format!("{}.attn_in", c.lp(l.i)), x)?;
        let qg = self.linear(g, x, l.get(&format!("{p}.q_proj.weight"))?)?;
        let q = Mat::alloc(rows * nq, hd)?;
        let gate = Mat::alloc(rows * nq, hd)?;
        crate::core::dev::copy2d(g, q.ptr(), hd, qg.ptr(), 2 * hd, rows * nq, hd)?;
        crate::core::dev::copy2d(g, gate.ptr(), hd, unsafe { qg.ptr().add(hd) }, 2 * hd, rows * nq, hd)?;
        drop(qg);
        let qn = self.rms(g, &q, l.get(&format!("{p}.q_norm.weight"))?)?;
        drop(q);
        let k = self.linear(g, x, l.get(&format!("{p}.k_proj.weight"))?)?;
        let k = Mat { buf: k.buf, rows: rows * nkv, cols: hd };
        let kn = self.rms(g, &k, l.get(&format!("{p}.k_norm.weight"))?)?;
        drop(k);
        let v = self.linear(g, x, l.get(&format!("{p}.v_proj.weight"))?)?;
        let rd = c.rot_dim();
        ops::rope(g, qn.ptr(), rows * nq, b.len, nq, hd, rd, self.inv_freq.ptr() as *const f32)?;
        ops::rope(g, kn.ptr(), rows * nkv, b.len, nkv, hd, rd, self.inv_freq.ptr() as *const f32)?;
        let o = Mat::alloc(rows, nq * hd)?;
        ops::attn_f64(g, o.ptr(), qn.ptr(), kn.ptr(), v.ptr(), [b.rows, b.len, nq, nkv, hd], (hd as f64).powf(-0.5))?;
        ops::mul_sigmoid(g, o.ptr(), o.ptr(), gate.ptr(), rows * nq * hd)?;
        self.cap(g, cap, format!("{}.attn_o", c.lp(l.i)), &o)?;
        self.linear(g, &o, l.get(&format!("{p}.o_proj.weight"))?)
    }

    /// Router top-k of xs: (indices, weights) on the host; hits counted into `counts`.
    fn route(&self, g: &Gpu, l: &Layer, xs: &Mat, counts: Option<&mut Vec<u64>>) -> Result<(Vec<i32>, Vec<f32>)> {
        let c = &self.cfg;
        let m = format!("{}.mlp", c.lp(l.i));
        let logits = self.linear(g, xs, l.get(&format!("{m}.gate.weight"))?)?;
        let (rows, e, k) = (xs.rows, c.experts, c.top_k);
        let ti = DevBuf::alloc(rows * k * 4)?;
        let tw = DevBuf::alloc(rows * k * 4)?;
        let cnt = DevBuf::zeroed(e * 4)?;
        ops::route(g, ti.ptr() as *mut i32, tw.ptr() as *mut f32, cnt.ptr() as *mut u32, logits.ptr(), rows, e, k)?;
        if let Some(cs) = counts {
            let h = download::<u32>(g, &cnt, e)?;
            for (a, b) in cs.iter_mut().zip(h) {
                *a += b as u64;
            }
        }
        Ok((download::<i32>(g, &ti, rows * k)?, download::<f32>(g, &tw, rows * k)?))
    }

    fn shared_hidden(&self, g: &Gpu, l: &Layer, xs: &Mat) -> Result<Mat> {
        let s = format!("{}.mlp.shared_expert", self.cfg.lp(l.i));
        let gt = self.linear(g, xs, l.get(&format!("{s}.gate_proj.weight"))?)?;
        let up = self.linear(g, xs, l.get(&format!("{s}.up_proj.weight"))?)?;
        let sh = Mat::alloc(xs.rows, gt.cols)?;
        ops::silu_mul(g, sh.ptr(), sh.cols, gt.ptr(), gt.cols, up.ptr(), up.cols, xs.rows, sh.cols)?;
        Ok(sh)
    }

    /// The all-experts capture of a MoE block (no output).
    fn moe_capture(&self, g: &Gpu, l: &Layer, xs: &Mat, cap: &mut Capture) -> Result<()> {
        let c = &self.cfg;
        let lp = c.lp(l.i);
        let mut hits = vec![0u64; c.experts];
        self.route(g, l, xs, Some(&mut hits))?;
        cap.counts.add_routing(l.i, &hits);
        let sh = self.shared_hidden(g, l, xs)?;
        cap.add(g, &format!("{lp}.shared_mid"), sh.ptr(), sh.rows, sh.cols)?;
        drop(sh);
        cap.add(g, &format!("{lp}.mlp.experts.input"), xs.ptr(), xs.rows, xs.cols)?;
        let w = l.get(&format!("{lp}.mlp.experts.gate_up_proj"))?;
        let (i2, h, rows) = (2 * c.moe_inter, c.hidden, xs.rows);
        let im = c.moe_inter;
        let chunk = self.opts.expert_chunk.max(1);
        let gu = DevBuf::alloc(chunk * rows * i2 * 4)?;
        let a = Mat::alloc(rows, im)?;
        for e0 in (0..c.experts).step_by(chunk) {
            let n = chunk.min(c.experts - e0);
            let wp = unsafe { w.ptr().add(e0 * i2 * h) };
            let gp = gu.ptr() as *mut f32;
            match self.opts.capture_gemm {
                CaptureGemm::Batched => unsafe { ops::linear_batched_shared_x(g, xs.ptr(), rows, h, wp, i2, n, gp)? },
                CaptureGemm::Wide => unsafe { ops::linear(g, xs.ptr(), rows, h, wp, n * i2, gp)? },
            }
            for j in 0..n {
                let (base, ld) = match self.opts.capture_gemm {
                    CaptureGemm::Batched => (unsafe { gp.add(j * rows * i2) }, i2),
                    CaptureGemm::Wide => (unsafe { gp.add(j * i2) }, n * i2),
                };
                ops::silu_mul(g, a.ptr(), im, base, ld, unsafe { base.add(im) }, ld, rows, im)?;
                cap.add(g, &format!("{lp}.mlp.experts.{}.mid", e0 + j), a.ptr(), rows, im)?;
            }
        }
        Ok(())
    }

    /// The routed MoE block: experts in ascending order, each adding its weighted output to its tokens' rows.
    fn moe(&self, g: &Gpu, l: &Layer, xs: &Mat) -> Result<Mat> {
        let c = &self.cfg;
        let lp = c.lp(l.i);
        let (rows, h, im, k) = (xs.rows, c.hidden, c.moe_inter, c.top_k);
        let sh = self.shared_hidden(g, l, xs)?;
        let shared = self.linear(g, &sh, l.get(&format!("{lp}.mlp.shared_expert.down_proj.weight"))?)?;
        drop(sh);
        let (ti, tw) = self.route(g, l, xs, None)?;
        let mut per: Vec<(Vec<i32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); c.experts];
        for r in 0..rows {
            for s in 0..k {
                let e = ti[r * k + s] as usize;
                per[e].0.push(r as i32);
                per[e].1.push(tw[r * k + s]);
            }
        }
        let out = Mat::zeros(rows, h)?;
        let wgu = l.get(&format!("{lp}.mlp.experts.gate_up_proj"))?;
        let wdn = l.get(&format!("{lp}.mlp.experts.down_proj"))?;
        for (e, (tok, w)) in per.iter().enumerate() {
            let n = tok.len();
            if n == 0 {
                continue;
            }
            let td = upload(tok)?;
            let wd = upload(w)?;
            let xe = Mat::alloc(n, h)?;
            ops::gather_rows(g, xe.ptr(), xs.ptr(), td.ptr() as *const i32, n, h)?;
            let gu = Mat::alloc(n, 2 * im)?;
            unsafe { ops::linear(g, xe.ptr(), n, h, wgu.ptr().add(e * 2 * im * h), 2 * im, gu.ptr())? };
            let hid = Mat::alloc(n, im)?;
            ops::silu_mul(g, hid.ptr(), im, gu.ptr(), 2 * im, unsafe { gu.ptr().add(im) }, 2 * im, n, im)?;
            let ye = Mat::alloc(n, h)?;
            unsafe { ops::linear(g, hid.ptr(), n, im, wdn.ptr().add(e * h * im), h, ye.ptr())? };
            ops::scale_add_rows(g, out.ptr(), ye.ptr(), td.ptr() as *const i32, wd.ptr() as *const f32, n, h)?;
            g.sync()?;
        }
        let gl = self.linear(g, xs, l.get(&format!("{lp}.mlp.shared_expert_gate.weight"))?)?;
        let m = Mat::alloc(rows, h)?;
        ops::shared_combine(g, m.ptr(), out.ptr(), shared.ptr(), gl.ptr(), rows, h)?;
        Ok(m)
    }

    fn dense_mlp(&self, g: &Gpu, l: &Layer, x: &Mat, cap: &mut Option<&mut Capture>) -> Result<Mat> {
        let lp = self.cfg.lp(l.i);
        self.cap(g, cap, format!("{lp}.mlp_in"), x)?;
        let gt = self.linear(g, x, l.get(&format!("{lp}.mlp.gate_proj.weight"))?)?;
        let up = self.linear(g, x, l.get(&format!("{lp}.mlp.up_proj.weight"))?)?;
        let hid = Mat::alloc(x.rows, gt.cols)?;
        ops::silu_mul(g, hid.ptr(), hid.cols, gt.ptr(), gt.cols, up.ptr(), up.cols, x.rows, hid.cols)?;
        self.cap(g, cap, format!("{lp}.mlp_mid"), &hid)?;
        self.linear(g, &hid, l.get(&format!("{lp}.mlp.down_proj.weight"))?)
    }

    fn mixer(&self, g: &Gpu, l: &Layer, h: &Mat, b: &Batch, cap: &mut Option<&mut Capture>) -> Result<Mat> {
        let lp = self.cfg.lp(l.i);
        let x = self.rms(g, h, l.get(&format!("{lp}.input_layernorm.weight"))?)?;
        let a = match self.cfg.kinds[l.i] {
            Kind::Linear => self.gdn(g, l, &x, b, cap)?,
            Kind::Full => self.attn(g, l, &x, b, cap)?,
        };
        let h1 = Mat::alloc(h.rows, h.cols)?;
        crate::core::dev::binary(
            g,
            h1.ptr(),
            h.cols,
            h.ptr(),
            h.cols,
            a.ptr(),
            a.cols,
            h.rows,
            h.cols,
            crate::core::dev::Bin::Add,
        )?;
        Ok(h1)
    }

    /// The capture forward of layer l (source weights) on one batch of the stream.
    pub fn capture(&self, g: &Gpu, l: &Layer, h: &Mat, b: &Batch, cap: &mut Capture) -> Result<()> {
        let mut c = Some(&mut *cap);
        let h1 = self.mixer(g, l, h, b, &mut c)?;
        let lp = self.cfg.lp(l.i);
        let x2 = self.rms(g, &h1, l.get(&format!("{lp}.post_attention_layernorm.weight"))?)?;
        if self.cfg.moe() {
            self.moe_capture(g, l, &x2, cap)?;
        } else {
            self.dense_mlp(g, l, &x2, &mut Some(cap))?;
        }
        g.sync()
    }

    /// The forward of layer l on one batch (real routing): the next layer's input.
    pub fn forward(&self, g: &Gpu, l: &Layer, h: &Mat, b: &Batch) -> Result<Mat> {
        let h1 = self.mixer(g, l, h, b, &mut None)?;
        let lp = self.cfg.lp(l.i);
        let x2 = self.rms(g, &h1, l.get(&format!("{lp}.post_attention_layernorm.weight"))?)?;
        let m = if self.cfg.moe() { self.moe(g, l, &x2)? } else { self.dense_mlp(g, l, &x2, &mut None)? };
        let out = Mat::alloc(h.rows, h.cols)?;
        crate::core::dev::binary(
            g,
            out.ptr(),
            h.cols,
            h1.ptr(),
            h.cols,
            m.ptr(),
            m.cols,
            h.rows,
            h.cols,
            crate::core::dev::Bin::Add,
        )?;
        g.sync()?;
        Ok(out)
    }

    /// Replacement tensors that rebuild layer l from dequantized weights {key: (out x in) device}: plain linears by
    /// name, routed experts restacked into gate_up_proj (E, 2I, H) and down_proj (E, H, I).
    pub fn overrides(
        &self,
        g: &Gpu,
        l: usize,
        specs: &[Target],
        mut deq: BTreeMap<String, Mat>,
    ) -> Result<BTreeMap<String, Mat>> {
        let c = &self.cfg;
        let mut o = BTreeMap::new();
        for s in specs.iter().filter(|s| s.expert.is_none()) {
            let m = deq.remove(&s.key).ok_or_else(|| shape(format!("no dequantized {}", s.key)))?;
            o.insert(format!("{}.weight", s.key), m);
        }
        if specs.iter().any(|s| s.expert.is_some()) {
            let (e, im, h) = (c.experts, c.moe_inter, c.hidden);
            let gu = Mat::zeros(e * 2 * im, h)?;
            let dn = Mat::zeros(e * h, im)?;
            for s in specs.iter().filter(|s| s.expert.is_some()) {
                let x = s.expert.expect("filtered");
                let m = deq.remove(&s.key).ok_or_else(|| shape(format!("no dequantized {}", s.key)))?;
                let (dst, n) = match s.role {
                    "expert.gate" => (gu.at(x * 2 * im, 0), m.len()),
                    "expert.up" => (gu.at(x * 2 * im + im, 0), m.len()),
                    _ => (dn.at(x * h, 0), m.len()),
                };
                crate::core::dev::copy2d(g, dst, n, m.ptr(), n, 1, n)?;
                g.sync()?;
            }
            let p = format!("{}.mlp.experts", c.lp(l));
            o.insert(format!("{p}.gate_up_proj"), gu);
            o.insert(format!("{p}.down_proj"), dn);
        }
        Ok(o)
    }
}
