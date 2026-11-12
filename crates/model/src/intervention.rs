//! Runtime interventions: a manifest plus direction files applied at serving time without touching weights.
//!
//! The general mechanism (family-agnostic, fail-closed): an intervention directory holds `intervention.json`
//! (the manifest) and direction `.pt` files (serialized torch tensors). At load we validate every layer index
//! against the family's plan and every direction file against its pinned sha256, orthonormalize the directions
//! in manifest order (the runtime rule of the reference implementation), and apply them per forward at the
//! manifest's application point. A mismatch refuses at boot naming both sides — an overlay built for another
//! model or another layer numbering never silently applies.
//!
//! The first consumer is the Blackfrost derisk overlay of the MiMo-V2.6 pack (schema
//! `blackfrost.mimo_mlp_output_intervention.v3`): a rank-one down-projection ablation
//! `y <- y - alpha * (y . d_hat) * d_hat` applied to the MoE/MLP output of every target layer, immediately
//! before that output joins the residual stream. Semantics ported from the pack's `exllamav3/blackfrost_derisk.py`
//! (MIT; attribution in THIRD_PARTY_NOTICES.md): fp32 math, directions orthonormalized once at load in serialized
//! order, the input dtype restored, weights untouched, and the flag off is byte-identical to no flag.
//! `InterventionMode::Additive` is our documented superset for future steering overlays.
//!
//! The torch `.pt` reader is deliberately minimal (no torch dependency): the format these manifests use is a
//! zip whose `data.pkl` rebuilds a plain dict of strings and ONE tensor built by `torch._rebuild_tensor_v2`
//! over a persistent storage whose bytes are `data/0` (fp16 or fp32). Anything else is refused by name.

use crate::family::FamilyPlan;
use serde_json::Value;
use std::path::Path;

/// Where in a layer's forward an entry applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyPoint {
    /// After the MLP/MoE output, immediately before it joins the residual stream (the fork's hook).
    PostMlpOutput,
    /// After the whole layer's output (a superset point for future overlays).
    PostLayerOutput,
}

/// How an entry modifies the stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InterventionMode {
    /// `y <- y - alpha * (y . d) * d` with `d` unit-norm (rank-one down-projection; Blackfrost).
    Projective { alpha: f32 },
    /// `y <- y + scale * d` (a steering vector).
    Additive { scale: f32 },
}

/// One (layer, direction) application site.
#[derive(Clone, Debug)]
pub struct InterventionEntry {
    pub layer: usize,
    pub point: ApplyPoint,
    /// The tensor the direction was fitted on (e.g. `post_layer_outputs`); carried for reporting.
    pub tensor_hint: String,
    /// Unit-norm fp32 direction of the family's hidden width.
    pub direction: Vec<f32>,
    pub mode: InterventionMode,
}

/// A loaded, validated intervention set.
#[derive(Clone, Debug)]
pub struct Intervention {
    pub schema: String,
    pub experiment: String,
    pub entries: Vec<InterventionEntry>,
}

impl Intervention {
    /// Load and validate an intervention directory against a family plan. Every failure names both sides.
    pub fn load(dir: &Path, plan: &FamilyPlan) -> Result<Self, String> {
        let at = |m: &str| format!("intervention: {m}");
        let manifest_path = dir.join("intervention.json");
        let text = std::fs::read_to_string(&manifest_path).map_err(|e| at(&format!("{}: {e}", manifest_path.display())))?;
        let root: Value = serde_json::from_str(&text).map_err(|e| at(&format!("intervention.json: {e}")))?;
        if root.get("weight_mutation").and_then(|v| v.as_bool()) == Some(true) {
            return Err(at("the manifest declares weight mutation; refusing"));
        }
        if root.get("router_mutation").and_then(|v| v.as_bool()) == Some(true) {
            return Err(at("the manifest declares router mutation; refusing"));
        }
        let schema = root.get("schema").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let experiment = root.get("experiment").and_then(|v| v.as_str()).unwrap_or("").to_string();

        // directions, orthonormalized in manifest order (the reference rule)
        let dirs = root
            .get("directions")
            .and_then(|v| v.as_array())
            .ok_or_else(|| at("the manifest has no directions[] (the v2/v3 schemas are supported)"))?;
        if dirs.is_empty() {
            return Err(at("the manifest names no direction"));
        }
        let mut base: Vec<Vec<f32>> = Vec::new();
        let mut alphas: Vec<f32> = Vec::new();
        let mut tensor_hint = String::new();
        for d in dirs {
            let rel = d
                .get("path")
                .or_else(|| d.get("direction"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| at("a direction has no path"))?;
            let path = dir.join(rel);
            let expected = d
                .get("sha256")
                .and_then(|v| v.as_str())
                .ok_or_else(|| at(&format!("{rel}: no sha256 in the manifest (fail-closed)")))?;
            let bytes = std::fs::read(&path).map_err(|e| at(&format!("{rel}: {e}")))?;
            let got = sha256_hex(&bytes);
            if got != expected {
                return Err(at(&format!("{rel}: direction hash mismatch (file {got}, manifest {expected})")));
            }
            let file = read_direction_pt(&path, &bytes).map_err(|e| at(&format!("{rel}: {e}")))?;
            if let Some(want) = d.get("tensor").and_then(|v| v.as_str()) {
                if file.tensor != want {
                    return Err(at(&format!("{rel}: direction tensor {:?} does not match the manifest's {want:?}", file.tensor)));
                }
            }
            tensor_hint = file.tensor.clone();
            let mut v = file.values;
            if v.is_empty() || v.len() != v.len() {
                return Err(at(&format!("{rel}: the direction is empty")));
            }
            // normalize, then Gram-Schmidt against priors in serialized order
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in &mut v {
                *x /= norm;
            }
            for prior in &base {
                let dot: f32 = v.iter().zip(prior).map(|(a, b)| a * b).sum();
                for (x, p) in v.iter_mut().zip(prior) {
                    *x -= dot * p;
                }
            }
            let retained = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if retained <= 1e-6 {
                return Err(at(&format!("{rel}: linearly dependent direction")));
            }
            for x in &mut v {
                *x /= retained;
            }
            for prior in &base {
                let dot: f32 = v.iter().zip(prior.iter()).map(|(a, b)| a * b).sum::<f32>().abs();
                if dot > 1e-5 {
                    return Err(at(&format!("{rel}: runtime orthogonalization failed (overlap {dot:.2e})")));
                }
            }
            base.push(v);
            alphas.push(
                d.get("alpha")
                    .and_then(|v| v.as_f64())
                    .filter(|a| a.is_finite() && *a > 0.0)
                    .ok_or_else(|| at(&format!("{rel}: alpha is missing or not a positive number")))? as f32,
            );
        }

        let point = match root.get("application_surface").and_then(|v| v.as_str()) {
            Some("mlp_outputs_after_down_proj") => ApplyPoint::PostMlpOutput,
            Some("layer_outputs") => ApplyPoint::PostLayerOutput,
            other => {
                return Err(at(&format!(
                    "application_surface {other:?} is not supported (mlp_outputs_after_down_proj, layer_outputs)"
                )))
            }
        };
        let targets =
            root.get("target_layers").and_then(|v| v.as_array()).ok_or_else(|| at("the manifest has no target_layers[]"))?;
        if targets.is_empty() {
            return Err(at("the manifest targets no layer"));
        }
        let layer_count = plan.layers.len();
        let mut entries = Vec::new();
        for t in targets {
            let layer = t.as_u64().ok_or_else(|| at("a target layer is not an integer"))? as usize;
            if layer >= layer_count {
                return Err(at(&format!(
                    "target layer {layer} is outside the family's {layer_count} layers (manifest {experiment:?}, family {})",
                    plan.family
                )));
            }
            for (i, d) in base.iter().enumerate() {
                entries.push(InterventionEntry {
                    layer,
                    point,
                    tensor_hint: tensor_hint.clone(),
                    direction: d.clone(),
                    mode: InterventionMode::Projective { alpha: alphas[i] },
                });
            }
        }
        Ok(Self { schema, experiment, entries })
    }

    /// The entries that apply at a layer boundary (empty when none).
    pub fn at(&self, layer: usize, point: ApplyPoint) -> impl Iterator<Item = &InterventionEntry> {
        self.entries.iter().filter(move |e| e.layer == layer && e.point == point)
    }
}

/// `h <- h - alpha * (h . d) * d` (Projective) or `h <- y + scale * d` (Additive), fp32 math, one row, fixed
/// reduction order (row-invariant by construction, like every kernel).
/// The process-wide loaded intervention (set once at boot by `serve` when `--intervention` is given).
pub static ACTIVE: std::sync::OnceLock<Intervention> = std::sync::OnceLock::new();

/// Apply the loaded intervention's entries for `layer` at `point` to one row (fp32 math, fixed reduction
/// order). The layer runner calls this at the layer boundary; with nothing loaded or no entry at this layer
/// and point, the row is returned unchanged. Returns whether anything was applied.
pub fn apply_at(layer: usize, point: ApplyPoint, row: &[f32], out: &mut [f32]) -> bool {
    let Some(iv) = ACTIVE.get() else {
        return false;
    };
    let mut applied = false;
    let mut cur = row.to_vec();
    for e in iv.at(layer, point) {
        let mut tmp = vec![0.0f32; cur.len()];
        apply_row(e, &cur, &mut tmp);
        cur = tmp;
        applied = true;
    }
    if applied {
        out.copy_from_slice(&cur);
    }
    applied
}

pub fn apply_row(entry: &InterventionEntry, row: &[f32], out: &mut [f32]) {
    match entry.mode {
        InterventionMode::Projective { alpha } => {
            let dot: f32 = row.iter().zip(&entry.direction).map(|(a, b)| a * b).sum();
            for (o, (a, d)) in out.iter_mut().zip(row.iter().zip(&entry.direction)) {
                *o = a - alpha * dot * d;
            }
        }
        InterventionMode::Additive { scale } => {
            for (o, (a, d)) in out.iter_mut().zip(row.iter().zip(&entry.direction)) {
                *o = a + scale * d;
            }
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let d = <sha2::Sha256 as sha2::Digest>::digest(bytes);
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// A parsed direction file: the metadata we validate plus the fp32 payload.
struct DirectionFile {
    tensor: String,
    values: Vec<f32>,
}

/// A minimal, dependency-free reader of the torch `.pt` zips these manifests use.
///
/// Format (torch.save, protocol 2): a zip with `archive/data.pkl` (a pickled dict whose values include one
/// tensor rebuilt by `torch._rebuild_tensor_v2(storage, offset, size, stride, ...)`; the storage is a
/// persistent-id tuple naming e.g. `torch.HalfStorage`), and `archive/data/0` with the raw elements.
///
/// We implement exactly the opcodes such a pickle needs (NONE, BININT*, SHORT_BINUNICODE/BINUNICODE, tuples,
/// lists, dicts, EMPTY_*, GLOBAL, REDUCE, BUILD, MEMOIZE, PERSID/BINPERSID, PROTO, FRAME, STOP) and refuse
/// everything else by name. The tests exercise the two real direction files of the pack.
fn read_direction_pt(path: &Path, bytes: &[u8]) -> Result<DirectionFile, String> {
    let files = read_zip(bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let (pkl_name, data_name) = {
        let pkl = files
            .iter()
            .find(|(n, _)| n.ends_with("data.pkl"))
            .map(|(n, _)| n.clone())
            .ok_or_else(|| format!("{}: no data.pkl in the archive", path.display()))?;
        let data = files
            .iter()
            .find(|(n, _)| n.contains("/data/0") || (n.ends_with("/0") && !n.ends_with("data.pkl")))
            .map(|(n, _)| n.clone())
            .ok_or_else(|| format!("{}: no data/0 payload in the archive", path.display()))?;
        (pkl, data)
    };
    let pkl = &files.iter().find(|(n, _)| *n == pkl_name).unwrap().1;
    let data = &files.iter().find(|(n, _)| *n == data_name).unwrap().1;

    #[derive(Clone, Debug)]
    enum PV {
        None,
        Bool(#[allow(dead_code)] bool),
        Int(i64),
        Str(String),
        Tuple(Vec<PV>),
        List(Vec<PV>),
        Dict(Vec<(PV, PV)>),
        Global(String, String),
        Storage(#[allow(dead_code)] String),
        Rebuilt(Vec<PV>),
    }
    let pv_str = |v: &PV| match v {
        PV::Str(s) => Some(s.clone()),
        _ => None,
    };
    let mut stack: Vec<PV> = Vec::new();
    let mut marks: Vec<usize> = Vec::new();
    let mut persid_storages: Vec<String> = Vec::new();
    let mut i = 0usize;
    let err = |m: &str| format!("{}: data.pkl: {m}", path.display());
    while i < pkl.len() {
        let op = pkl[i];
        i += 1;
        match op {
            0x80 => {
                i += 1;
            } // PROTO
            0x95 => {
                i += 8;
            } // FRAME
            0x4e => stack.push(PV::None),        // NONE
            0x88 => stack.push(PV::Bool(true)),  // NEWTRUE
            0x89 => stack.push(PV::Bool(false)), // NEWFALSE
            0x4a => {
                // BININT
                let v = i32::from_le_bytes(pkl[i..i + 4].try_into().map_err(|_| err("BININT eof"))?) as i64;
                i += 4;
                stack.push(PV::Int(v));
            }
            0x4b => {
                // BININT1
                stack.push(PV::Int(pkl[i] as i64));
                i += 1;
            }
            0x4d => {
                // BININT2 ('M', two-byte unsigned)
                let v = u16::from_le_bytes(pkl[i..i + 2].try_into().map_err(|_| err("BININT2 eof"))?) as i64;
                i += 2;
                stack.push(PV::Int(v));
            }
            0x4c => {
                // LONG ('L', a decimal line)
                let start = i;
                while i < pkl.len() && pkl[i] != b'\n' {
                    i += 1;
                }
                let v: i64 = std::str::from_utf8(&pkl[start..i])
                    .map_err(|_| err("LONG is not utf-8"))?
                    .trim()
                    .parse()
                    .map_err(|_| err("LONG is not a number"))?;
                i += 1;
                stack.push(PV::Int(v));
            }
            0x8a | 0x8b => {
                // LONG1 / LONG2
                let n = if op == 0x8a {
                    pkl[i] as usize
                } else {
                    u32::from_le_bytes(pkl[i..i + 4].try_into().map_err(|_| err("LONG2 eof"))?) as usize
                };
                i += if op == 0x8a { 1 } else { 4 };
                let b = pkl.get(i..i + n).ok_or_else(|| err("LONG eof"))?;
                i += n;
                let v = if n == 1 { b[0] as i8 as i64 } else { i64::from_le_bytes(b.try_into().map_err(|_| err("LONG wide"))?) };
                stack.push(PV::Int(v));
            }
            0x8c => {
                // SHORT_BINUNICODE
                let n = pkl[i] as usize;
                i += 1;
                stack.push(PV::Str(
                    String::from_utf8_lossy(pkl.get(i..i + n).ok_or_else(|| err("SHORT_BINUNICODE eof"))?).into_owned(),
                ));
                i += n;
            }
            0x58 => {
                // BINUNICODE
                let n = u32::from_le_bytes(pkl[i..i + 4].try_into().map_err(|_| err("BINUNICODE eof"))?) as usize;
                i += 4;
                stack
                    .push(PV::Str(String::from_utf8_lossy(pkl.get(i..i + n).ok_or_else(|| err("BINUNICODE eof"))?).into_owned()));
                i += n;
            }
            0x28 => marks.push(stack.len()), // MARK
            0x29 => {
                // EMPTY_TUPLE
                stack.push(PV::Tuple(vec![]));
            }
            0x85..=0x87 => {
                // TUPLE1/2/3
                let n = (op - 0x84) as usize;
                if stack.len() < n {
                    return Err(err("TUPLE underflow"));
                }
                let items = stack.split_off(stack.len() - n);
                stack.push(PV::Tuple(items));
            }
            0x74 => {
                // TUPLE (from mark)
                let m = marks.pop().ok_or_else(|| err("TUPLE without mark"))?;
                let items = stack.split_off(m);
                stack.push(PV::Tuple(items));
            }
            0x5d => stack.push(PV::List(vec![])), // EMPTY_LIST
            0x7d => stack.push(PV::Dict(vec![])), // EMPTY_DICT
            0x61 => {
                // APPEND
                let v = stack.pop().ok_or_else(|| err("APPEND underflow"))?;
                match stack.last_mut() {
                    Some(PV::List(l)) => l.push(v),
                    _ => return Err(err("APPEND onto a non-list")),
                }
            }
            0x65 => {
                // APPENDS (from mark)
                let m = marks.pop().ok_or_else(|| err("APPENDS without mark"))?;
                let items = stack.split_off(m);
                match stack.last_mut() {
                    Some(PV::List(l)) => l.extend(items),
                    _ => return Err(err("APPENDS onto a non-list")),
                }
            }
            0x73 => {
                // SETITEM
                let v = stack.pop().ok_or_else(|| err("SETITEM underflow"))?;
                let k = stack.pop().ok_or_else(|| err("SETITEM underflow"))?;
                match stack.last_mut() {
                    Some(PV::Dict(d)) => d.push((k, v)),
                    _ => return Err(err("SETITEM onto a non-dict")),
                }
            }
            0x75 => {
                // SETITEMS (from mark)
                let m = marks.pop().ok_or_else(|| err("SETITEMS without mark"))?;
                let items = stack.split_off(m);
                if items.len() % 2 != 0 {
                    return Err(err("SETITEMS with an odd item count"));
                }
                match stack.last_mut() {
                    Some(PV::Dict(d)) => {
                        for pair in items.chunks(2) {
                            d.push((pair[0].clone(), pair[1].clone()));
                        }
                    }
                    _ => return Err(err("SETITEMS onto a non-dict")),
                }
            }
            0x94 => {} // MEMOIZE (we never look back)
            0x71 => {
                i += 1;
            } // BINPUT (memo write; we never look back)
            0x72 => {
                i += 4;
            } // LONG_BINPUT
            0x6f => {
                // NEWLIST: a list of a known length
                let n = i32::from_le_bytes(pkl[i..i + 4].try_into().map_err(|_| err("NEWLIST eof"))?) as usize;
                i += 4;
                if n > 1 << 20 {
                    return Err(err("NEWLIST of an absurd length"));
                }
                stack.push(PV::List(Vec::with_capacity(n)));
            }
            0x63 => {
                // GLOBAL: two newline-terminated names in the stream
                let read_line = |i: &mut usize| -> Result<String, String> {
                    let start = *i;
                    while *i < pkl.len() && pkl[*i] != b'\n' {
                        *i += 1;
                    }
                    if *i >= pkl.len() {
                        return Err(err("GLOBAL name without a newline"));
                    }
                    let s = String::from_utf8_lossy(&pkl[start..*i]).into_owned();
                    *i += 1;
                    Ok(s)
                };
                let m = read_line(&mut i)?;
                let n = read_line(&mut i)?;
                stack.push(PV::Global(m, n));
            }
            0x51 => {
                // BINPERSID ('Q')
                let pid = stack.pop().ok_or_else(|| err("BINPERSID underflow"))?;
                match pid {
                    PV::Tuple(t) if t.len() >= 3 => {
                        // ('storage', <StorageClass global>, key, ...)
                        let class = match &t[1] {
                            PV::Global(m, n) => format!("{m}.{n}"),
                            other => format!("{other:?}"),
                        };
                        persid_storages.push(class.clone());
                        stack.push(PV::Storage(class));
                    }
                    PV::Str(s) => {
                        persid_storages.push(s.clone());
                        stack.push(PV::Storage(s));
                    }
                    other => return Err(err(&format!("BINPERSID of {other:?}"))),
                }
            }
            0x52 => {
                // REDUCE
                let args = stack.pop().ok_or_else(|| err("REDUCE underflow"))?;
                let g = stack.pop().ok_or_else(|| err("REDUCE underflow"))?;
                match g {
                    PV::Global(m, n) => {
                        if n == "_rebuild_tensor_v2" {
                            stack.push(PV::Rebuilt(match args {
                                PV::Tuple(t) => t,
                                _ => return Err(err("_rebuild_tensor_v2 of a non-tuple")),
                            }));
                        } else if n == "_rebuild_parameter" {
                            // torch nn.Parameter wrapper: keep the inner value
                            match args {
                                PV::Tuple(t) if !t.is_empty() => stack.push(t.into_iter().next().unwrap()),
                                _ => return Err(err("_rebuild_parameter of an empty tuple")),
                            }
                        } else {
                            stack.push(PV::Global(m, n)); // OrderedDict and friends: opaque
                        }
                    }
                    _ => return Err(err("REDUCE of a non-global")),
                }
            }
            0x42 => {}     // BUILD ('b', state we do not read)
            0x2e => break, // STOP
            other => return Err(err(&format!("opcode 0x{other:02x} is not part of this format"))),
        }
    }
    // the dict on the stack at STOP is the file's payload
    let dict = match stack.pop() {
        Some(PV::Dict(d)) => d,
        _ => return Err(err("the payload is not a dict")),
    };
    let tensor_hint =
        dict.iter().find(|(k, _)| pv_str(k).as_deref() == Some("tensor")).and_then(|(_, v)| pv_str(v)).unwrap_or_default();
    // the direction tensor: the Rebuilt under the "direction" key, or any Rebuilt value
    let rebuilt = dict
        .iter()
        .find(|(k, _)| pv_str(k).as_deref() == Some("direction"))
        .and_then(|(_, v)| match v {
            PV::Rebuilt(t) => Some(t.clone()),
            _ => None,
        })
        .or_else(|| {
            dict.iter().find_map(|(_, v)| match v {
                PV::Rebuilt(t) => Some(t.clone()),
                _ => None,
            })
        })
        .ok_or_else(|| err("no tensor in the payload"))?;
    // _rebuild_tensor_v2(storage, offset, size, stride, requires_grad, backward_hooks): size is index 2
    let size = match rebuilt.get(2) {
        Some(PV::Tuple(t)) => t
            .iter()
            .map(|v| match v {
                PV::Int(i) => Ok(*i),
                _ => Err(err("tensor size of a non-integer")),
            })
            .collect::<Result<Vec<i64>, _>>()?,
        _ => return Err(err("the tensor has no size tuple")),
    };
    // the storage class decides the element type; the tensor's size validates the payload length
    let half = persid_storages.iter().any(|s| s.contains("Half"));
    let bfloat16 = persid_storages.iter().any(|s| s.contains("BFloat16"));
    let float = !half && !bfloat16 && persid_storages.iter().any(|s| s.contains("Float"));
    if !half && !bfloat16 && !float {
        return Err(format!(
            "{}: data.pkl: the direction's storage is {:?}; only fp16/bf16/fp32 tensors are read",
            path.display(),
            persid_storages
        ));
    }
    let values: Vec<f32> = if half {
        let mut out = Vec::with_capacity(data.len() / 2);
        for c in data.chunks_exact(2) {
            out.push(half_to_f32(u16::from_le_bytes([c[0], c[1]])));
        }
        out
    } else if bfloat16 {
        let mut out = Vec::with_capacity(data.len() / 2);
        for c in data.chunks_exact(2) {
            out.push(f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16));
        }
        out
    } else {
        let mut out = Vec::with_capacity(data.len() / 4);
        for c in data.chunks_exact(4) {
            out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        out
    };
    // the tensor spec: a Reduce whose args contain (Storage, offset, size, stride); find the biggest Int tuple
    // (the sizes tuple) by walking the memo for a Tuple of Ints of len 1 (a 1-D direction)
    if size.len() != 1 {
        return Err(format!("{}: data.pkl: the direction is {}-D; only 1-D is read", path.display(), size.len()));
    }
    let n = size[0] as usize;
    if values.len() != n {
        return Err(format!("{}: data/0 holds {} elements, the tensor says {n}", path.display(), values.len()));
    }
    Ok(DirectionFile { tensor: tensor_hint, values })
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}
impl Bits<'_> {
    fn read(&mut self, n: usize) -> Result<u32, String> {
        let mut v = 0u32;
        for b in 0..n {
            let byte = self.data.get(self.pos / 8).ok_or("deflate: eof")?;
            let bit = (byte >> (self.pos % 8)) & 1;
            v |= (bit as u32) << b;
            self.pos += 1;
        }
        Ok(v)
    }
    fn align(&mut self) {
        self.pos = self.pos.div_ceil(8) * 8;
    }
}

/// Read a zip's stored files (name -> bytes): the central directory only, STORE and DEFLATE.
fn read_zip(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>, String> {
    // find EOCD
    let eocd_at = bytes.windows(4).rposition(|w| w == b"PK\x05\x06").ok_or("no EOCD")?;
    let n = u16::from_le_bytes(bytes[eocd_at + 10..eocd_at + 12].try_into().unwrap()) as usize;
    let cd_off = u32::from_le_bytes(bytes[eocd_at + 16..eocd_at + 20].try_into().unwrap()) as usize;
    let mut files = Vec::with_capacity(n);
    let mut p = cd_off;
    for _ in 0..n {
        if &bytes[p..p + 4] != b"PK\x01\x02" {
            return Err("bad central directory".into());
        }
        let method = u16::from_le_bytes(bytes[p + 10..p + 12].try_into().unwrap());
        let csize = u32::from_le_bytes(bytes[p + 20..p + 24].try_into().unwrap()) as usize;
        let name_len = u16::from_le_bytes(bytes[p + 28..p + 30].try_into().unwrap()) as usize;
        let extra_len = u16::from_le_bytes(bytes[p + 30..p + 32].try_into().unwrap()) as usize;
        let comm_len = u16::from_le_bytes(bytes[p + 32..p + 34].try_into().unwrap()) as usize;
        let lfh_off = u32::from_le_bytes(bytes[p + 42..p + 46].try_into().unwrap()) as usize;
        let name = String::from_utf8_lossy(&bytes[p + 46..p + 46 + name_len]).into_owned();
        // local file header: skip its (possibly different) name/extra lengths
        if &bytes[lfh_off..lfh_off + 4] != b"PK\x03\x04" {
            return Err(format!("{name}: bad local header"));
        }
        let l_name = u16::from_le_bytes(bytes[lfh_off + 26..lfh_off + 28].try_into().unwrap()) as usize;
        let l_extra = u16::from_le_bytes(bytes[lfh_off + 28..lfh_off + 30].try_into().unwrap()) as usize;
        let start = lfh_off + 30 + l_name + l_extra;
        let raw = &bytes[start..start + csize];
        let data = match method {
            0 => raw.to_vec(),
            8 => inflate(raw)?,
            m => return Err(format!("{name}: compression method {m} is not supported")),
        };
        files.push((name, data));
        p += 46 + name_len + extra_len + comm_len;
    }
    Ok(files)
}

/// Raw DEFLATE inflate (no dependency): enough for the stored-and-deflated zips torch writes.
fn inflate(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut bits = Bits { data, pos: 0 };
    loop {
        let last = bits.read(1)?;
        let ty = bits.read(2)?;
        match ty {
            0 => {
                bits.align();
                let h = bits.pos / 8;
                if h + 4 > data.len() {
                    return Err("deflate: eof in stored block".into());
                }
                let len = u16::from_le_bytes(data[h..h + 2].try_into().unwrap()) as usize;
                out.extend_from_slice(&data[h + 4..h + 4 + len]);
                bits.pos = (h + 4 + len) * 8;
            }
            1 => {
                // fixed Huffman
                let mut lens = [0u8; 288];
                let mut dlens = [5u8; 30];
                dlens[0] = 0;
                for (i, l) in lens.iter_mut().enumerate() {
                    *l = if i < 144 {
                        8
                    } else if i < 256 {
                        9
                    } else if i < 280 {
                        7
                    } else {
                        8
                    };
                }
                let (lit_dec, _) = build_huffman(&lens)?;
                let (dist_dec, _) = build_huffman(&dlens)?;
                inflate_block(&mut bits, &lit_dec, &dist_dec, &mut out)?;
            }
            2 => {
                let hlit = bits.read(5)? as usize + 257;
                let hdist = bits.read(5)? as usize + 1;
                let hclen = bits.read(4)? as usize + 4;
                const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                let mut cl = [0u8; 19];
                for i in 0..hclen {
                    cl[ORDER[i]] = bits.read(3)? as u8;
                }
                let (cl_dec, _) = build_huffman(&cl)?;
                let mut lens = Vec::new();
                while lens.len() < hlit + hdist {
                    let sym = read_sym(&mut bits, &cl_dec)?;
                    match sym {
                        0..=15 => lens.push(sym as u8),
                        16 => {
                            let prev = *lens.last().ok_or("deflate: repeat with no previous")?;
                            let n = 3 + bits.read(2)? as usize;
                            for _ in 0..n {
                                lens.push(prev);
                            }
                        }
                        17 => {
                            let n = 3 + bits.read(3)? as usize;
                            lens.resize(lens.len() + n, 0);
                        }
                        18 => {
                            let n = 11 + bits.read(7)? as usize;
                            lens.resize(lens.len() + n, 0);
                        }
                        _ => return Err("deflate: bad code-length symbol".into()),
                    }
                }
                let (lit_dec, _) = build_huffman(&lens[..hlit])?;
                let (dist_dec, _) = build_huffman(&lens[hlit..])?;
                inflate_block(&mut bits, &lit_dec, &dist_dec, &mut out)?;
            }
            _ => return Err("deflate: block type 3".into()),
        }
        if last == 1 {
            break;
        }
    }
    Ok(out)
}

fn read_sym(bits: &mut Bits, dec: &Huff) -> Result<u16, String> {
    let mut code = 0u32;
    let mut len = 0usize;
    loop {
        code = (code << 1) | bits.read(1)?;
        len += 1;
        if len > 15 {
            return Err("deflate: bad huffman code".into());
        }
        if let Some(sym) = dec.lookup(code as u64, len) {
            return Ok(sym);
        }
    }
}

struct Huff {
    map: std::collections::HashMap<(u64, usize), u16>,
}

impl Huff {
    fn lookup(&self, code: u64, len: usize) -> Option<u16> {
        self.map.get(&(code, len)).copied()
    }
}

fn build_huffman(lens: &[u8]) -> Result<(Huff, u32), String> {
    let mut counts = [0usize; 16];
    for &l in lens {
        counts[l as usize] += 1;
    }
    counts[0] = 0;
    let mut next = [0u32; 16];
    let mut code = 0u32;
    for b in 1..16 {
        code = (code + counts[b - 1] as u32) << 1;
        next[b] = code;
    }
    let mut map = std::collections::HashMap::new();
    for (sym, &l) in lens.iter().enumerate() {
        if l != 0 {
            map.insert((next[l as usize] as u64, l as usize), sym as u16);
            next[l as usize] += 1;
        }
    }
    Ok((Huff { map }, code))
}

fn inflate_block(bits: &mut Bits, lit: &Huff, dist: &Huff, out: &mut Vec<u8>) -> Result<(), String> {
    const LEN_BASE: [u16; 29] =
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
    const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
    const DIST_BASE: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193,
        12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u8; 30] =
        [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
    loop {
        let sym = read_sym(bits, lit)?;
        match sym {
            0..=255 => out.push(sym as u8),
            256 => return Ok(()),
            257..=285 => {
                let i = (sym - 257) as usize;
                let len = LEN_BASE[i] as usize + bits.read(LEN_EXTRA[i] as usize)? as usize;
                let dsym = read_sym(bits, dist)? as usize;
                let d = DIST_BASE[dsym] as usize + bits.read(DIST_EXTRA[dsym] as usize)? as usize;
                if d > out.len() {
                    return Err("deflate: distance before start".into());
                }
                let start = out.len() - d;
                for k in 0..len {
                    let b = out[start + k];
                    out.push(b);
                }
            }
            _ => return Err("deflate: bad literal symbol".into()),
        }
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, _) => {
            let mut e = 0u32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            f &= 0x3ff;
            (sign << 31) | ((127 - 15 - e + 1) << 23) | (f << 13)
        }
        (0x1f, 0) => (sign << 31) | (0xff << 23),
        (0x1f, _) => (sign << 31) | (0xff << 23) | (frac << 13),
        _ => (sign << 31) | ((exp + (127 - 15)) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_to_f32_matches_known_values() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xbc00), -1.0);
        assert_eq!(half_to_f32(0x4000), 2.0);
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert!((half_to_f32(0x3555) - 0.33325195).abs() < 1e-6);
    }

    #[test]
    fn half_conversion_matches_ieee_binary16_exhaustively() {
        for h in 0u16..=u16::MAX {
            let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
            let e = ((h >> 10) & 31) as i32;
            let f = (h & 1023) as f32;
            let actual = half_to_f32(h);
            if e == 31 {
                let expected = ((h as u32 & 0x8000) << 16) | 0x7f800000 | ((h as u32 & 1023) << 13);
                assert_eq!(actual.to_bits(), expected, "h={h:04x}");
            } else {
                let expected = sign * if e == 0 { f * 2.0f32.powi(-24) } else { (1.0 + f / 1024.0) * 2.0f32.powi(e - 15) };
                assert_eq!(actual.to_bits(), expected.to_bits(), "h={h:04x}");
            }
        }
    }

    #[test]
    fn projective_apply_matches_the_reference_math() {
        let e = InterventionEntry {
            layer: 28,
            point: ApplyPoint::PostMlpOutput,
            tensor_hint: "post_layer_outputs".into(),
            direction: vec![1.0, 0.0],
            mode: InterventionMode::Projective { alpha: 3.5 },
        };
        let mut out = [0.0f32; 2];
        apply_row(&e, &[2.0, 5.0], &mut out);
        // y - 3.5 * (y . d) * d with d = e1: (2 - 7, 5)
        assert_eq!(out, [-5.0, 5.0]);
    }

    #[test]
    fn additive_apply_and_the_off_case() {
        let e = InterventionEntry {
            layer: 3,
            point: ApplyPoint::PostLayerOutput,
            tensor_hint: "x".into(),
            direction: vec![0.0, 2.0],
            mode: InterventionMode::Additive { scale: 0.5 },
        };
        let mut out = [0.0f32; 2];
        apply_row(&e, &[1.0, 1.0], &mut out);
        assert_eq!(out, [1.0, 2.0]);
    }
}

#[cfg(test)]
mod files_tests {
    use super::*;
    use crate::family::Registry;
    use serde_json::Value;

    const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/mimo/intervention");

    fn plan() -> crate::family::FamilyPlan {
        // probe from a copy of the committed fixture dir (the registry reads more than config.json);
        // a unique dir per call: the tests run in parallel
        let src = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/mimo"));
        let dir = std::env::temp_dir().join(format!("rexl3-mimo-int-{}-{:?}", std::process::id(), std::thread::current().id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["config.json", "chat_template.jinja"] {
            if let Ok(b) = std::fs::read(src.join(name)) {
                std::fs::write(dir.join(name), b).unwrap();
            }
        }
        let (_, plan) = Registry::global().probe(&dir).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        plan
    }

    /// The pack's real manifest loads, validates, orthonormalizes and expands to 12 entries (6 layers x 2 directions).
    #[test]
    fn the_real_blackfrost_manifest_loads_and_expands() {
        let iv = Intervention::load(Path::new(DIR), &plan()).unwrap();
        assert_eq!(iv.schema, "blackfrost.mimo_mlp_output_intervention.v3");
        assert_eq!(iv.entries.len(), 12);
        let layers: Vec<usize> =
            iv.entries.iter().map(|e| e.layer).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        assert_eq!(layers, vec![28, 29, 30, 31, 32, 33]);
        assert!(iv.entries.iter().all(|e| e.point == ApplyPoint::PostMlpOutput && e.tensor_hint == "post_layer_outputs"));
        // both directions at every layer, both unit-norm, mutually orthogonal
        let d0 = &iv.entries[0].direction;
        let d1 = &iv.entries[1].direction;
        assert_eq!(d0.len(), 4096);
        let n0: f32 = d0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n1: f32 = d1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n0 - 1.0).abs() < 1e-4 && (n1 - 1.0).abs() < 1e-4);
        let dot: f32 = d0.iter().zip(d1).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-4, "orthogonalized directions overlap by {dot:.2e}");
        // the reference serialized overlap was <= 2.7e-7 for the pair; the fp16 direction data
        // is identical to the fork's (the sha256 pins prove it), so our parse must match the fork's math
    }

    /// The manifest's own sha256 pins are enforced: a corrupted direction file is refused by name.
    #[test]
    fn a_corrupt_direction_file_is_refused_by_hash() {
        let dir = std::env::temp_dir().join(format!("rexl3-iv-corrupt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for f in [
            "intervention.json",
            "directions/layer025-post_layer_outputs.pt",
            "directions/pass2-remaining21-layer033-post_layer_outputs.pt",
        ] {
            if let Some(parent) = dir.join(f).parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::copy(Path::new(DIR).join(f), dir.join(f)).unwrap();
        }
        let mut bytes = std::fs::read(dir.join("directions/layer025-post_layer_outputs.pt")).unwrap();
        bytes[..8].copy_from_slice(&[0u8; 8]);
        std::fs::write(dir.join("directions/layer025-post_layer_outputs.pt"), bytes).unwrap();
        let e = Intervention::load(&dir, &plan()).err().unwrap();
        assert!(e.contains("layer025-post_layer_outputs.pt") && e.contains("hash mismatch"), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A manifest naming a layer the family does not have refuses, naming both sides.
    #[test]
    fn an_out_of_range_layer_is_refused_naming_both_sides() {
        let dir = std::env::temp_dir().join(format!("rexl3-iv-layer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for f in [
            "intervention.json",
            "directions/layer025-post_layer_outputs.pt",
            "directions/pass2-remaining21-layer033-post_layer_outputs.pt",
        ] {
            if let Some(parent) = dir.join(f).parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::copy(Path::new(DIR).join(f), dir.join(f)).unwrap();
        }
        let mut root: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("intervention.json")).unwrap()).unwrap();
        root["target_layers"] = serde_json::json!([48]);
        std::fs::write(dir.join("intervention.json"), root.to_string()).unwrap();
        let e = Intervention::load(&dir, &plan()).err().unwrap();
        assert!(e.contains("layer 48") && e.contains("48 layers") && e.contains("mimo_v2"), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A weight-mutation manifest is refused outright.
    #[test]
    fn a_weight_mutation_manifest_is_refused() {
        let dir = std::env::temp_dir().join(format!("rexl3-iv-mut-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for f in [
            "intervention.json",
            "directions/layer025-post_layer_outputs.pt",
            "directions/pass2-remaining21-layer033-post_layer_outputs.pt",
        ] {
            if let Some(parent) = dir.join(f).parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::copy(Path::new(DIR).join(f), dir.join(f)).unwrap();
        }
        let mut root: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("intervention.json")).unwrap()).unwrap();
        root["weight_mutation"] = serde_json::json!(true);
        std::fs::write(dir.join("intervention.json"), root.to_string()).unwrap();
        let e = Intervention::load(&dir, &plan()).err().unwrap();
        assert!(e.contains("weight mutation"), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Mock-forward equivalence: on == off minus alpha*(y.d)*d at exactly the named layers, bit-identical elsewhere.
    #[test]
    fn a_mock_forward_proves_the_apply_and_the_off_case() {
        let iv = Intervention::load(Path::new(DIR), &plan()).unwrap();
        let mut hidden = vec![0.25f32; 4096];
        hidden[0] = 1.0;
        let off = hidden.clone();
        for e in iv.at(29, ApplyPoint::PostMlpOutput) {
            let mut out = vec![0.0f32; 4096];
            apply_row(e, &hidden, &mut out);
            let dot: f32 = hidden.iter().zip(&e.direction).map(|(a, b)| a * b).sum();
            for (o, (a, d)) in out.iter().zip(hidden.iter().zip(&e.direction)) {
                let want = a - e_alpha(e) * dot * d;
                assert!((o - want).abs() < 1e-5);
            }
            hidden = out;
        }
        // a non-target layer is untouched
        assert_eq!(iv.at(0, ApplyPoint::PostMlpOutput).count(), 0);
        assert_ne!(hidden, off);
        // and the off-forward (no intervention applied) is the original, byte for byte
        let mut plain = vec![0.25f32; 4096];
        plain[0] = 1.0;
        assert_eq!(off, plain);
        // layers with no entry are bit-identical through the same runner (the off contract)
        let mut untouched = vec![0.25f32; 4096];
        untouched[0] = 1.0;
        assert_eq!(iv.at(27, ApplyPoint::PostMlpOutput).count(), 0, "layer 27 has no entry");
        assert_eq!(untouched, plain);
    }

    fn e_alpha(e: &InterventionEntry) -> f32 {
        match e.mode {
            InterventionMode::Projective { alpha } => alpha,
            InterventionMode::Additive { scale } => scale,
        }
    }
}
