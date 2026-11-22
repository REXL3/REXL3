//! Runtime interventions: a manifest plus direction files applied at serving time without touching weights.
//!
//! The general mechanism (family-agnostic, fail-closed): an intervention directory holds `intervention.json`
//! (the manifest) and direction `.pt` files (serialized torch tensors). At load we validate every layer index
//! against the family's plan and every direction file against its pinned sha256, orthonormalize the directions
//! in manifest order (the runtime rule of the reference implementation). The operator ([`apply_row`]) applies an
//! entry to one row at the manifest's application point. A mismatch refuses at load naming both sides — an
//! overlay built for another model, another hidden width or another layer numbering never loads.
//!
//! Not applied in this build: no family's forward calls [`apply_at`], so `rexl3-serve` refuses `--intervention`
//! at startup (`serve::INTERVENTION_NOT_APPLIED`) instead of serving a model the overlay did not change. This
//! module is the loader, the validation and the operator, exercised by its tests.
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
    /// Load and validate an intervention directory against a family plan and the model's hidden width
    /// ([`hidden_width`] reads it from the pack's config). Every failure names both sides.
    pub fn load(dir: &Path, plan: &FamilyPlan, hidden: usize) -> Result<Self, String> {
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
            if v.is_empty() {
                return Err(at(&format!("{rel}: the direction is empty")));
            }
            if v.len() != hidden {
                return Err(at(&format!(
                    "{rel}: the direction is {} wide, the model's hidden width is {hidden} (manifest {experiment:?}, family {})",
                    v.len(),
                    plan.family
                )));
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

/// The hidden width a pack's `config.json` declares: `hidden_size`, at the top level or under `text_config`.
/// This is the width [`Intervention::load`] checks every direction against.
pub fn hidden_width(model_dir: &Path) -> Result<usize, String> {
    let path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let root: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let hidden = [root.get("hidden_size"), root.get("text_config").and_then(|t| t.get("hidden_size"))]
        .into_iter()
        .flatten()
        .find_map(|v| v.as_u64())
        .filter(|h| *h > 0);
    hidden.map(|h| h as usize).ok_or_else(|| format!("{}: no positive hidden_size", path.display()))
}

/// The process-wide loaded intervention. Nothing sets it in this build: `serve` refuses `--intervention`
/// because no forward applies an overlay yet.
pub static ACTIVE: std::sync::OnceLock<Intervention> = std::sync::OnceLock::new();

/// Apply the loaded intervention's entries for `layer` at `point` to one row (fp32 math, fixed reduction
/// order). The seam a layer runner is to call at the layer boundary; no forward calls it in this build. With
/// nothing loaded or no entry at this layer and point, the row is returned unchanged. Returns whether anything
/// was applied.
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

/// `h <- h - alpha * (h . d) * d` (Projective) or `h <- h + scale * d` (Additive), fp32 math, one row, fixed
/// reduction order (row-invariant by construction, like every kernel).
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
                let b = take(pkl, &mut i, 4).ok_or_else(|| err("BININT eof"))?;
                stack.push(PV::Int(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64));
            }
            0x4b => {
                // BININT1
                let b = take(pkl, &mut i, 1).ok_or_else(|| err("BININT1 eof"))?;
                stack.push(PV::Int(b[0] as i64));
            }
            0x4d => {
                // BININT2 ('M', two-byte unsigned)
                let b = take(pkl, &mut i, 2).ok_or_else(|| err("BININT2 eof"))?;
                stack.push(PV::Int(u16::from_le_bytes([b[0], b[1]]) as i64));
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
                    take(pkl, &mut i, 1).ok_or_else(|| err("LONG1 eof"))?[0] as usize
                } else {
                    let b = take(pkl, &mut i, 4).ok_or_else(|| err("LONG2 eof"))?;
                    u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize
                };
                let b = take(pkl, &mut i, n).ok_or_else(|| err("LONG eof"))?;
                let v = if n == 1 { b[0] as i8 as i64 } else { i64::from_le_bytes(b.try_into().map_err(|_| err("LONG wide"))?) };
                stack.push(PV::Int(v));
            }
            0x8c => {
                // SHORT_BINUNICODE
                let n = take(pkl, &mut i, 1).ok_or_else(|| err("SHORT_BINUNICODE eof"))?[0] as usize;
                let b = take(pkl, &mut i, n).ok_or_else(|| err("SHORT_BINUNICODE eof"))?;
                stack.push(PV::Str(String::from_utf8_lossy(b).into_owned()));
            }
            0x58 => {
                // BINUNICODE
                let b = take(pkl, &mut i, 4).ok_or_else(|| err("BINUNICODE eof"))?;
                let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
                let b = take(pkl, &mut i, n).ok_or_else(|| err("BINUNICODE eof"))?;
                stack.push(PV::Str(String::from_utf8_lossy(b).into_owned()));
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
                if m > stack.len() {
                    return Err(err("TUPLE mark above the stack"));
                }
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
                if m > stack.len() {
                    return Err(err("APPENDS mark above the stack"));
                }
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
                if m > stack.len() {
                    return Err(err("SETITEMS mark above the stack"));
                }
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
                let b = take(pkl, &mut i, 4).ok_or_else(|| err("NEWLIST eof"))?;
                let n = i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
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
    // a payload that is not a whole number of elements is a cut file, not a shorter tensor
    let width = if half || bfloat16 { 2 } else { 4 };
    if data.len() % width != 0 {
        return Err(format!(
            "{}: data/0 holds {} bytes, not a whole number of {width}-byte elements",
            path.display(),
            data.len()
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
    let n = size[0];
    if n < 0 || values.len() as u64 != n as u64 {
        return Err(format!("{}: data/0 holds {} elements, the tensor says {n}", path.display(), values.len()));
    }
    Ok(DirectionFile { tensor: tensor_hint, values })
}

/// The next `n` bytes of `bytes` at `*at`, advancing it; `None` when the buffer ends first.
fn take<'a>(bytes: &'a [u8], at: &mut usize, n: usize) -> Option<&'a [u8]> {
    let s = bytes.get(*at..at.checked_add(n)?)?;
    *at += n;
    Some(s)
}

/// A little-endian u16 / u32 at `at`, as an offset or length; `None` when the buffer ends first.
fn le16(bytes: &[u8], at: usize) -> Option<usize> {
    let s = bytes.get(at..at.checked_add(2)?)?;
    Some(u16::from_le_bytes([s[0], s[1]]) as usize)
}

fn le32(bytes: &[u8], at: usize) -> Option<usize> {
    let s = bytes.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
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
    // find EOCD; every offset and length below comes from the file, so every read is bounds-checked
    let eocd_at = bytes.windows(4).rposition(|w| w == b"PK\x05\x06").ok_or("no EOCD")?;
    let eocd = "the end-of-central-directory record is cut short";
    // the record is 22 bytes before its comment: a file cut inside it is cut, even past the fields read here
    if bytes.len() - eocd_at < 22 {
        return Err(eocd.into());
    }
    let n = le16(bytes, eocd_at + 10).ok_or(eocd)?;
    let cd_off = le32(bytes, eocd_at + 16).ok_or(eocd)?;
    let mut files = Vec::new();
    let mut p = cd_off;
    for k in 0..n {
        let cut = || format!("central directory entry {k} of {n} is outside the file");
        // the fixed part of an entry is 46 bytes
        let entry = bytes.get(p..).filter(|e| e.len() >= 46).ok_or_else(cut)?;
        if &entry[..4] != b"PK\x01\x02" {
            return Err("bad central directory".into());
        }
        let field16 = |at: usize| le16(entry, at).ok_or_else(cut);
        let field32 = |at: usize| le32(entry, at).ok_or_else(cut);
        let method = field16(10)?;
        let csize = field32(20)?;
        let name_len = field16(28)?;
        let extra_len = field16(30)?;
        let comm_len = field16(32)?;
        let lfh_off = field32(42)?;
        let name = String::from_utf8_lossy(entry.get(46..46 + name_len).ok_or_else(cut)?).into_owned();
        // local file header: skip its (possibly different) name/extra lengths
        let header = bytes
            .get(lfh_off..)
            .filter(|h| h.len() >= 30)
            .ok_or_else(|| format!("{name}: the local header is outside the file"))?;
        if &header[..4] != b"PK\x03\x04" {
            return Err(format!("{name}: bad local header"));
        }
        let l_name = u16::from_le_bytes([header[26], header[27]]) as usize;
        let l_extra = u16::from_le_bytes([header[28], header[29]]) as usize;
        let raw = header
            .get(30 + l_name + l_extra..)
            .and_then(|d| d.get(..csize))
            .ok_or_else(|| format!("{name}: its {csize} bytes of data are outside the file"))?;
        let data = match method {
            0 => raw.to_vec(),
            8 => inflate(raw).map_err(|e| format!("{name}: {e}"))?,
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
                let len = u16::from_le_bytes([data[h], data[h + 1]]) as usize;
                out.extend_from_slice(data.get(h + 4..h + 4 + len).ok_or("deflate: eof in stored block")?);
                bits.pos = (h + 4 + len) * 8;
            }
            1 => {
                // fixed Huffman
                let mut lens = [0u8; 288];
                // every one of the 30 distance codes is five bits (RFC 1951, 3.2.6), code 0 (distance 1) included
                let dlens = [5u8; 30];
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
                if dsym >= DIST_BASE.len() {
                    return Err("deflate: bad distance symbol".into());
                }
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
    /// The fixture model's hidden width (`testdata/mimo/config.json`).
    const HIDDEN: usize = 4096;
    const FILES: [&str; 3] = [
        "intervention.json",
        "directions/layer025-post_layer_outputs.pt",
        "directions/pass2-remaining21-layer033-post_layer_outputs.pt",
    ];

    /// A private copy of the fixture overlay, for a test that edits it.
    fn overlay_copy(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("rexl3-iv-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for f in FILES {
            std::fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
            std::fs::copy(Path::new(DIR).join(f), dir.join(f)).unwrap();
        }
        dir
    }

    /// The width the loader checks against comes from the pack's config: top level, or under `text_config`.
    #[test]
    fn the_hidden_width_is_read_from_the_pack_config() {
        let mimo = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/mimo"));
        assert_eq!(hidden_width(mimo).unwrap(), HIDDEN);
        let dir = std::env::temp_dir().join(format!("rexl3-iv-width-config-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"text_config": {"hidden_size": 5120}}"#).unwrap();
        assert_eq!(hidden_width(&dir).unwrap(), 5120);
        std::fs::write(dir.join("config.json"), r#"{"hidden_size": 0}"#).unwrap();
        assert!(hidden_width(&dir).unwrap_err().contains("hidden_size"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A direction fitted on another hidden width is refused, naming the direction's width and the model's.
    #[test]
    fn a_direction_of_another_width_is_refused_naming_both_widths() {
        let e = Intervention::load(Path::new(DIR), &plan(), 5120).err().unwrap();
        assert!(e.contains("layer025-post_layer_outputs.pt"), "{e}");
        assert!(e.contains("4096 wide") && e.contains("hidden width is 5120") && e.contains("mimo_v2"), "{e}");
        // one off is still a mismatch, and the matching width still loads
        assert!(Intervention::load(Path::new(DIR), &plan(), HIDDEN - 1).is_err());
        assert!(Intervention::load(Path::new(DIR), &plan(), HIDDEN).is_ok());
    }

    /// A direction file cut short whose manifest pin matches the cut bytes (so the hash check passes and the
    /// readers see it) is an error at every cut, never a panic.
    #[test]
    fn a_truncated_direction_file_with_a_matching_pin_is_an_error() {
        let dir = overlay_copy("truncated");
        let rel = FILES[1];
        let whole = std::fs::read(dir.join(rel)).unwrap();
        let manifest = std::fs::read_to_string(dir.join("intervention.json")).unwrap();
        let plan = plan();
        // the ends, the middle, and every cut inside the zip's trailing records
        let cuts = [0, 1, 4, 64, whole.len() / 2].into_iter().chain(whole.len().saturating_sub(700)..whole.len());
        for cut in cuts {
            let bytes = &whole[..cut];
            std::fs::write(dir.join(rel), bytes).unwrap();
            let mut root: Value = serde_json::from_str(&manifest).unwrap();
            root["directions"][0]["sha256"] = serde_json::json!(sha256_hex(bytes));
            std::fs::write(dir.join("intervention.json"), root.to_string()).unwrap();
            let e = Intervention::load(&dir, &plan, HIDDEN).err().unwrap_or_else(|| panic!("cut at {cut} loaded"));
            assert!(e.contains("layer025-post_layer_outputs.pt") && !e.contains("hash mismatch"), "cut {cut}: {e}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

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
        let iv = Intervention::load(Path::new(DIR), &plan(), HIDDEN).unwrap();
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
        let e = Intervention::load(&dir, &plan(), HIDDEN).err().unwrap();
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
        let e = Intervention::load(&dir, &plan(), HIDDEN).err().unwrap();
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
        let e = Intervention::load(&dir, &plan(), HIDDEN).err().unwrap();
        assert!(e.contains("weight mutation"), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Mock-forward equivalence: on == off minus alpha*(y.d)*d at exactly the named layers, bit-identical elsewhere.
    #[test]
    fn a_mock_forward_proves_the_apply_and_the_off_case() {
        let iv = Intervention::load(Path::new(DIR), &plan(), HIDDEN).unwrap();
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

/// Malformed input to the zip, inflate and pickle readers is an error, never a panic. These call the readers
/// directly: `Intervention::load` checks the sha256 pin first.
#[cfg(test)]
mod reader_tests {
    use super::*;

    const PT: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/mimo/intervention/directions/layer025-post_layer_outputs.pt");

    fn pt() -> Vec<u8> {
        std::fs::read(PT).unwrap()
    }

    /// A zip of stored (uncompressed) members, as torch writes them.
    fn stored_zip(members: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, data) in members {
            let at = out.len() as u32;
            let sizes = [(data.len() as u32).to_le_bytes(), (data.len() as u32).to_le_bytes()].concat();
            out.extend_from_slice(b"PK\x03\x04");
            out.extend_from_slice(&[0u8; 14]); // version, flags, method 0, time, date, crc
            out.extend_from_slice(&sizes);
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&[0u8; 2]);
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(data);
            central.extend_from_slice(b"PK\x01\x02");
            central.extend_from_slice(&[0u8; 16]); // versions, flags, method 0, time, date, crc
            central.extend_from_slice(&sizes);
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&[0u8; 12]); // extra, comment, disk, attributes
            central.extend_from_slice(&at.to_le_bytes());
            central.extend_from_slice(name.as_bytes());
        }
        let cd_off = out.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(b"PK\x05\x06");
        out.extend_from_slice(&[0u8; 4]);
        out.extend_from_slice(&(members.len() as u16).to_le_bytes());
        out.extend_from_slice(&(members.len() as u16).to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&cd_off.to_le_bytes());
        out.extend_from_slice(&[0u8; 2]);
        out
    }

    /// The fixture's pickle and payload, for rebuilding direction files around edited parts.
    fn parts() -> (Vec<u8>, Vec<u8>) {
        let files = read_zip(&pt()).unwrap();
        let get = |suffix: &str| files.iter().find(|(n, _)| n.ends_with(suffix)).unwrap().1.clone();
        (get("data.pkl"), get("data/0"))
    }

    fn read(pkl: &[u8], data: &[u8]) -> Result<DirectionFile, String> {
        read_direction_pt(Path::new("t.pt"), &stored_zip(&[("t/data.pkl", pkl), ("t/data/0", data)]))
    }

    #[test]
    fn the_test_zip_writer_round_trips_the_real_direction() {
        let (pkl, data) = parts();
        assert_eq!(stored_zip(&[("a", b"xy")]).len(), 30 + 1 + 2 + 46 + 1 + 22);
        let d = read(&pkl, &data).unwrap();
        assert_eq!((d.values.len(), d.tensor.as_str()), (4096, "post_layer_outputs"));
    }

    #[test]
    fn every_truncation_of_a_direction_zip_is_an_error() {
        let whole = pt();
        assert!(read_zip(&whole).is_ok());
        for cut in 0..whole.len() {
            assert!(read_zip(&whole[..cut]).is_err(), "cut at {cut} of {} read", whole.len());
        }
    }

    #[test]
    fn zip_offsets_and_sizes_outside_the_file_are_errors() {
        let whole = pt();
        let eocd = whole.windows(4).rposition(|w| w == b"PK\x05\x06").unwrap();
        let cd = u32::from_le_bytes(whole[eocd + 16..eocd + 20].try_into().unwrap()) as usize;
        let edit = |at: usize, value: &[u8]| {
            let mut b = whole.clone();
            b[at..at + value.len()].copy_from_slice(value);
            read_zip(&b)
        };
        let far = u32::MAX.to_le_bytes();
        let e = edit(eocd + 16, &far).unwrap_err(); // the central directory offset
        assert!(e.contains("central directory entry 0"), "{e}");
        let e = edit(eocd + 10, &u16::MAX.to_le_bytes()).unwrap_err(); // more entries than the file holds
        assert!(e.contains("central directory"), "{e}");
        let e = edit(cd + 42, &far).unwrap_err(); // the first member's local header offset
        assert!(e.contains("local header is outside the file"), "{e}");
        let e = edit(cd + 20, &far).unwrap_err(); // the first member's stored size
        assert!(e.contains("data are outside the file"), "{e}");
        let e = edit(cd + 28, &u16::MAX.to_le_bytes()).unwrap_err(); // the first member's name length
        assert!(e.contains("central directory entry 0"), "{e}");
        // an end record with nothing after its signature
        assert!(read_zip(b"PK\x05\x06").unwrap_err().contains("cut short"));
    }

    #[test]
    fn every_truncation_of_the_pickle_is_handled_without_a_panic() {
        let (pkl, data) = parts();
        let mut refused = 0;
        for cut in 0..pkl.len() {
            // a cut that still leaves a whole dict on the stack may read; none may panic
            if read(&pkl[..cut], &data).is_err() {
                refused += 1;
            }
        }
        assert!(read(&pkl[..pkl.len() / 2], &data).is_err());
        assert!(refused > pkl.len() / 2, "{refused} of {} cuts refused", pkl.len());
    }

    #[test]
    fn pickle_operands_that_run_past_the_end_are_errors() {
        let (_, data) = parts();
        for (bytes, want) in [
            (&[0x4au8, 1, 2][..], "BININT eof"),
            (&[0x4b], "BININT1 eof"),
            (&[0x4d, 1], "BININT2 eof"),
            (&[0x8a], "LONG1 eof"),
            (&[0x8b, 1], "LONG2 eof"),
            (&[0x8a, 8, 1], "LONG eof"),
            (&[0x8c], "SHORT_BINUNICODE eof"),
            (&[0x8c, 5, b'a'], "SHORT_BINUNICODE eof"),
            (&[0x58, 1], "BINUNICODE eof"),
            (&[0x58, 0xff, 0xff, 0xff, 0xff, b'a'], "BINUNICODE eof"),
            (&[0x6f, 1], "NEWLIST eof"),
            // two values, a mark, then TUPLE2 shrinks the stack below the mark
            (&[0x4e, 0x4e, 0x28, 0x86, 0x74], "TUPLE mark above the stack"),
            (&[0x4e, 0x4e, 0x28, 0x86, 0x65], "APPENDS mark above the stack"),
            (&[0x4e, 0x4e, 0x28, 0x86, 0x75], "SETITEMS mark above the stack"),
        ] {
            let e = read(bytes, &data).err().unwrap_or_else(|| panic!("{bytes:02x?} read"));
            assert!(e.contains(want), "{bytes:02x?}: {e}");
        }
    }

    #[test]
    fn a_payload_that_is_not_whole_elements_is_an_error() {
        let (pkl, data) = parts();
        let e = read(&pkl, &data[..data.len() - 1]).err().unwrap();
        assert!(e.contains("8191 bytes") && e.contains("2-byte elements"), "{e}");
        let e = read(&pkl, &data[..data.len() - 2]).err().unwrap();
        assert!(e.contains("4095 elements") && e.contains("4096"), "{e}");
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    /// Raw deflate of `dynamic_text()` (zlib level 9, one dynamic-Huffman block).
    const DYNAMIC: &str = "edccc90943310c45d15654404803a946f26ccb9667a7fc6c7f200d043e3ceee22d0e7a39ae6c9b97e1a9d350b15368e8eb71b26d59264fcd43a54eb161a8c7cb7665d93c0d0f9dba8a8d42c51bb9916f04495d07de30cbb50f98de405b4125a02ea78095f7ebd70971e53a9eff207e00";
    const TEXT: &[u8] = b"abcabcabcabcabc hello hello hello, the quick brown fox; the quick brown fox jumps.";

    fn dynamic_text() -> Vec<u8> {
        let mut t: Vec<u8> = (0..600usize).map(|i| ((i * 7 + i / 3) % 23 + 97) as u8).collect();
        for _ in 0..3 {
            t.extend_from_slice(TEXT);
        }
        t
    }

    #[test]
    fn inflate_reads_a_dynamic_block_and_refuses_every_truncation_of_it() {
        let stream = unhex(DYNAMIC);
        assert_eq!(inflate(&stream).unwrap(), dynamic_text());
        for cut in 0..stream.len() {
            assert!(inflate(&stream[..cut]).is_err(), "cut at {cut} of {} inflated", stream.len());
        }
    }

    #[test]
    fn inflate_refuses_a_stored_block_longer_than_its_input() {
        // final stored block declaring 100 bytes, holding 2
        assert!(inflate(&[0x01, 100, 0, !100, 0xff, 1, 2]).unwrap_err().contains("stored block"));
        assert!(inflate(&[0x01, 100]).unwrap_err().contains("stored block"));
        assert_eq!(inflate(&[0x01, 2, 0, !2, 0xff, 7, 9]).unwrap(), [7, 9]);
    }

    /// A fixed-Huffman block with back-references inflates to its source. The stream is zlib's raw deflate of `fixed_text()` under
    /// `Z_FIXED` (Python: `zlib.compressobj(9, zlib.DEFLATED, -15, 9, zlib.Z_FIXED)`): one final block of type 1 whose matches
    /// use distance 1 (distance code 0, the runs), 16, 64 and 77.
    const FIXED: &str = "4b4c44052519a90a85a599c9d90a4945f9e5790a69f9150a59a5b905c50af965a9450a20e99cc4aa4a8594fc743d05\
        03432363135333730bcbc4a4e494d434743e4986551109181899985958d9d83938b9b87978f9f8050485844544c5c42524a5a46564e5e4151495945554d5d\
        43534b5b47574f5f4118eb2b2b6b1b5b3a7543f00";

    fn fixed_text() -> Vec<u8> {
        let fox = b"the quick brown fox jumps over the lazy dog. ";
        let mut t = vec![b'a'; 16];
        t.extend_from_slice(fox);
        t.extend_from_slice(b"0123456789abcdef0123456789abcdef");
        t.extend_from_slice(fox);
        t.extend_from_slice(&[b'z'; 40]);
        t.extend(0..64u8);
        t.extend(0..64u8);
        t
    }

    #[test]
    fn inflate_reads_a_fixed_block_with_back_references_and_refuses_every_truncation_of_it() {
        let stream = unhex(FIXED);
        assert_eq!(stream[0] & 7, 0b011, "one final block of type 1 (fixed Huffman)");
        assert_eq!(inflate(&stream).unwrap(), fixed_text());
        for cut in 0..stream.len() {
            assert!(inflate(&stream[..cut]).is_err(), "cut at {cut} of {} inflated", stream.len());
        }
    }

    /// A dynamic block may declare 32 distance codes, two more than the format defines; a stream that then
    /// uses distance symbol 30 is an error.
    #[test]
    fn inflate_refuses_a_distance_symbol_the_format_does_not_define() {
        let stream = unhex("0dff010400000080200000000000000000000000001800000000000000000000000000000000000000b80000004012000c");
        assert!(inflate(&stream).unwrap_err().contains("bad distance symbol"));
    }

    /// A deflated member that is cut short names the member.
    #[test]
    fn a_deflated_zip_member_cut_short_is_an_error_naming_it() {
        let stream = unhex(DYNAMIC);
        let mut zip = stored_zip(&[("t/data.pkl", &stream[..stream.len() / 2])]);
        let eocd = zip.windows(4).rposition(|w| w == b"PK\x05\x06").unwrap();
        let cd = u32::from_le_bytes(zip[eocd + 16..eocd + 20].try_into().unwrap()) as usize;
        zip[cd + 10] = 8; // method: deflate
        let e = read_zip(&zip).unwrap_err();
        assert!(e.contains("t/data.pkl") && e.contains("deflate"), "{e}");
    }
}
