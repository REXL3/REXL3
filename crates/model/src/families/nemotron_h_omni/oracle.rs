//! The layer-by-layer oracle harness: a [`super::model::Tap`] that compares the model's per-layer activations with the fork's dumps,
//! plus the committed tolerance numbers. Wave 1 task B1 (O1).
//!
//! # The dumps
//!
//! `<root>/layers<s>/m<iii>.f16` is module `iii` of the fork's forward for sample `s`, f16 little-endian `[tokens][width]`, and
//! `<root>/layers<s>/meta.json` lists the modules with their shapes. Module 0 is the embedding ([`Point::Embed`]), modules `1..=88` are
//! the trunk layers (`Layer(i - 1)`), 89 is `norm_f` ([`Point::FinalNorm`]) and 90 the logits ([`Point::Logits`], `[tokens][131072]`).
//! `<root>/logits_<s>.f16`, `top1_<s>.i32` and `ids_<s>.u32` are the whole-sequence logits, their argmax and the token ids; only
//! sample 0 has layer dumps. The fork's [`Point::Mixer`] outputs are not dumped, so a mixer tap has no reference and is only counted.
//!
//! # What is measured
//!
//! Per point, over the rows the model tapped: `rel_l2 = ||a - b|| / ||b||` (a: the model's f32, b: the fork's f16 widened), the largest
//! absolute difference, and for the logits the top-1 agreement with the fork's recorded argmax (`top1_<s>.i32`) and with the argmax
//! of the dumped f16 rows (ties go to the lower id). Results are plain data ([`Comparison`]); [`table`] prints them and
//! [`Tolerances::apply`] applies the committed limits (`crates/model/testdata/nemotron_omni/oracle_tolerances.json`, made by
//! `tools/nemotron_omni/noise_layers_omni.py`).
//!
//! # The limits
//!
//! Per module the limit is `max(2 x spread, floor)`: the spread is the fork's own whole-versus-chunked rel_l2 at that module (a free
//! running forward, so it grows with depth as the fork's differences are amplified), and the floor sits above the f16 rounding of the
//! dump. A free-running model is held to these directly; a teacher-forced layer (fed the dump's input) differs from the dump by
//! one layer's worth and sits far under them. Top-1 is held to at least 95% of the rows over every sample compared.
//!
//! # Rows
//!
//! A tap carries no position, so each point keeps a cursor: its first call covers dump rows `0..rows`, the next the rows after
//! those, and so on. [`OracleTap::reset`] starts over and [`OracleTap::seek`] starts at another row. A model that taps a chunk of a
//! prefill therefore sees the chunks compared with the matching rows of the whole-sequence dump.

use super::bridge::f16_bits_to_f32;
use super::model::{Point, Tap};
use crate::error::ModelError;
use gb_cuda::{DevBuf, Stream};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// The committed numbers, compiled in (see [`Tolerances`]).
pub const COMMITTED_TOLERANCES: &str = include_str!("../../../testdata/nemotron_omni/oracle_tolerances.json");

/// The gate on top-1 agreement with the fork, over every logits row compared (the owner's decision: at least 95% on 2 x 1024 tokens).
pub const TOP1_MIN: f64 = 0.95;

/// The trunk's layer count: modules `1..=TRUNK` of a dump are `Layer(0..TRUNK)`.
pub const TRUNK: usize = 88;
/// The module index of `norm_f` and of the logits in a dump.
pub const MODULE_FINAL_NORM: usize = TRUNK + 1;
pub const MODULE_LOGITS: usize = TRUNK + 2;
/// Modules in a complete dump.
pub const MODULES: usize = TRUNK + 3;

/// The dump's module index of a point, or `None` for a point the fork does not dump ([`Point::Mixer`], or a layer outside the trunk).
pub fn module_of(point: Point) -> Option<usize> {
    match point {
        Point::Embed => Some(0),
        Point::Layer(i) if i < TRUNK => Some(i + 1),
        Point::FinalNorm => Some(MODULE_FINAL_NORM),
        Point::Logits => Some(MODULE_LOGITS),
        Point::Layer(_) | Point::Mixer(_) => None,
    }
}

/// The point of a dump module index (the inverse of [`module_of`]).
pub fn point_of(module: usize) -> Option<Point> {
    match module {
        0 => Some(Point::Embed),
        m if m <= TRUNK => Some(Point::Layer(m - 1)),
        MODULE_FINAL_NORM => Some(Point::FinalNorm),
        MODULE_LOGITS => Some(Point::Logits),
        _ => None,
    }
}

/// The file name of module `m` in a layer dump: `m000.f16`.
pub fn module_file(m: usize) -> String {
    format!("m{m:03}.f16")
}

/// A short label of a point for tables: `embed`, `L12`, `norm_f`, `logits`, `mixer12`.
pub fn label(point: Point) -> String {
    match point {
        Point::Embed => "embed".into(),
        Point::Layer(i) => format!("L{i}"),
        Point::FinalNorm => "norm_f".into(),
        Point::Logits => "logits".into(),
        Point::Mixer(i) => format!("mixer{i}"),
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Reading the dumps

/// The 65536-entry f16 -> f32 table, built once.
fn f16_table() -> &'static [f32] {
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    TABLE.get_or_init(|| (0..=u16::MAX).map(f16_bits_to_f32).collect())
}

/// f16 little-endian bytes widened to f32 (exact).
pub fn f16_le_to_f32(bytes: &[u8]) -> Vec<f32> {
    let t = f16_table();
    bytes.chunks_exact(2).map(|c| t[u16::from_le_bytes([c[0], c[1]]) as usize]).collect()
}

/// `rows` rows of `width` f16 values starting at row `first`, from an f16 little-endian `[tokens][width]` file, as f32. A missing or
/// short file is an error naming the file.
pub fn read_f16_rows(path: &Path, first: usize, rows: usize, width: usize) -> Result<Vec<f32>, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let len = f.metadata().map_err(|e| format!("{}: {e}", path.display()))?.len();
    if width == 0 || len % (2 * width as u64) != 0 {
        return Err(format!("{}: {len} bytes is not a whole number of rows of {width} f16 values", path.display()));
    }
    let have = (len / (2 * width as u64)) as usize;
    if first + rows > have {
        return Err(format!("{}: rows {first}..{} are past the {have} rows of the dump", path.display(), first + rows));
    }
    f.seek(SeekFrom::Start(first as u64 * width as u64 * 2)).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut bytes = vec![0u8; rows * width * 2];
    f.read_exact(&mut bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(f16_le_to_f32(&bytes))
}

fn read_i32s(path: &Path) -> Result<Vec<i32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() % 4 != 0 {
        return Err(format!("{}: {} bytes is not a whole number of i32", path.display(), b.len()));
    }
    Ok(b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// One module of a dump's `meta.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ModuleMeta {
    key: String,
    /// Elements per row.
    width: usize,
}

/// A fork dump directory, for one sample.
#[derive(Clone, Debug)]
pub struct Dump {
    root: PathBuf,
    sample: usize,
    /// Tokens per sample.
    tokens: usize,
    /// The layer dumps of this sample (`layers<s>/meta.json`): `None` when the sample only has logits.
    layers: Option<Vec<ModuleMeta>>,
    top1: Vec<i32>,
}

impl Dump {
    /// Opens `root` for `sample`. The sample's `top1_<s>.i32` and `logits_<s>.f16` must exist, and so must its `layers<s>/meta.json`
    /// when that directory does. Reads `meta.json`s and the top-1 file only.
    pub fn open(root: &Path, sample: usize) -> Result<Self, String> {
        let top1 = read_i32s(&root.join(format!("top1_{sample}.i32")))?;
        let logits = root.join(format!("logits_{sample}.f16"));
        if !logits.is_file() {
            return Err(format!("{}: no such file (the sample's whole-sequence logits)", logits.display()));
        }
        let dir = root.join(format!("layers{sample}"));
        let layers = if dir.is_dir() {
            let meta = dir.join("meta.json");
            let text = std::fs::read_to_string(&meta).map_err(|e| format!("{}: {e}", meta.display()))?;
            let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", meta.display()))?;
            let mods =
                v.get("modules").and_then(Value::as_array).ok_or_else(|| format!("{}: no `modules` list", meta.display()))?;
            let mut out = Vec::with_capacity(mods.len());
            for (i, m) in mods.iter().enumerate() {
                let width = m
                    .get("shape")
                    .and_then(Value::as_array)
                    .and_then(|s| s.last())
                    .and_then(Value::as_u64)
                    .ok_or_else(|| format!("{}: module {i} has no shape", meta.display()))? as usize;
                let key = m.get("key").and_then(Value::as_str).unwrap_or("?").to_string();
                out.push(ModuleMeta { key, width });
            }
            Some(out)
        } else {
            None
        };
        Ok(Self { root: root.to_path_buf(), sample, tokens: top1.len(), layers, top1 })
    }

    /// The tokens of the sample (rows of every dump).
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn sample(&self) -> usize {
        self.sample
    }

    /// Whether the sample has layer dumps (sample 1 of the committed oracle does not).
    pub fn has_layers(&self) -> bool {
        self.layers.is_some()
    }

    /// The fork's recorded argmax of the logits, one per token.
    pub fn top1(&self) -> &[i32] {
        &self.top1
    }

    /// The token ids of the sample.
    pub fn ids(&self) -> Result<Vec<u32>, String> {
        let p = self.root.join(format!("ids_{}.u32", self.sample));
        let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        if b.len() % 4 != 0 {
            return Err(format!("{}: {} bytes is not a whole number of u32", p.display(), b.len()));
        }
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// The file holding `module`: `layers<s>/m<iii>.f16`, and for the logits the sample's `logits_<s>.f16` when the layer dump has
    /// no `m090.f16` of its own (the two are identical on the committed oracle).
    pub fn module_path(&self, module: usize) -> PathBuf {
        let layer = self.root.join(format!("layers{}", self.sample)).join(module_file(module));
        if module == MODULE_LOGITS && !layer.is_file() {
            return self.root.join(format!("logits_{}.f16", self.sample));
        }
        layer
    }

    /// The dump's key and row width for `module` (`None` when the sample has no layer dumps, or no such module).
    fn module_meta(&self, module: usize) -> Option<&ModuleMeta> {
        self.layers.as_ref().and_then(|l| l.get(module))
    }

    /// `rows` rows of `module` from row `first`, as f32.
    pub fn read(&self, module: usize, first: usize, rows: usize, width: usize) -> Result<Vec<f32>, String> {
        read_f16_rows(&self.module_path(module), first, rows, width)
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Comparison results

/// Top-1 agreement of the model's logits rows with the fork's.
#[derive(Clone, Debug, PartialEq)]
pub struct Top1 {
    pub rows: usize,
    /// Rows where the model's argmax equals the fork's recorded argmax (`top1_<s>.i32`).
    pub agree_recorded: usize,
    /// Rows where it equals the argmax of the fork's dumped f16 row (the recorded argmax was taken before the f16 rounding).
    pub agree_dumped: usize,
    /// Rows where the fork's recorded argmax equals the argmax of its own dumped row (the fork against itself).
    pub fork_self_agree: usize,
    /// The largest gap, over the rows that disagree with the recorded argmax, between the fork's logit of its own top-1 and of the
    /// model's choice (in the fork's f16 logits). A small gap is a near-tie, not a defect.
    pub max_gap: f32,
}

impl Top1 {
    /// The gate's measure: agreement with the recorded argmax.
    pub fn agreement(&self) -> f64 {
        if self.rows == 0 {
            return 0.0;
        }
        self.agree_recorded as f64 / self.rows as f64
    }
}

/// One point's comparison with the fork, over every row the model tapped there. Plain data.
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    pub sample: usize,
    pub point: Point,
    /// The module of the dump.
    pub module: usize,
    /// The fork's module key, e.g. `backbone.layers.12` (`?` when the dump has no `meta.json`).
    pub key: String,
    pub rows: usize,
    pub width: usize,
    /// `||a - b|| / ||b||` over all compared elements; infinite when `b` is all zero and `a` is not, NaN when `a` has a NaN.
    pub rel_l2: f64,
    /// `max |a - b|`.
    pub max_abs: f64,
    /// `||b|| / sqrt(elements)`: the fork's rms, for scale.
    pub ref_rms: f64,
    /// Non-finite values in the model's rows.
    pub nonfinite: usize,
    /// Set for the logits.
    pub top1: Option<Top1>,
}

/// The running sums of one point, over its chunks.
#[derive(Clone, Debug, Default)]
struct Acc {
    rows: usize,
    elems: u64,
    sq_diff: f64,
    sq_ref: f64,
    max_abs: f64,
    nonfinite: usize,
    top1: Option<Top1>,
}

/// `(sum (a - b)^2, sum b^2, max |a - b|, non-finite count in a)` over two equal slices, accumulated in f64.
pub fn sums(got: &[f32], want: &[f32]) -> (f64, f64, f64, usize) {
    assert_eq!(got.len(), want.len(), "compared slices differ in length");
    let (mut d2, mut r2, mut mx, mut nf) = (0f64, 0f64, 0f64, 0usize);
    for (&a, &b) in got.iter().zip(want) {
        if !a.is_finite() {
            nf += 1;
        }
        let d = (a as f64 - b as f64).abs();
        d2 += d * d;
        r2 += b as f64 * b as f64;
        if d > mx || d.is_nan() {
            mx = d;
        }
    }
    (d2, r2, mx, nf)
}

/// The index of the largest value; ties go to the lower index; a NaN never wins (an all-NaN row gives index 0).
pub fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut v = f32::NEG_INFINITY;
    for (i, &x) in row.iter().enumerate() {
        if x > v {
            v = x;
            best = i;
        }
    }
    best
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The tap

/// What a tap at a point leads to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plan {
    /// Filtered out by [`OracleTap::only`]: nothing recorded.
    Ignore,
    /// No reference exists (a mixer output, or a layer point of a sample with logits only): the rows are counted and not compared.
    Skip,
    Compare,
}

struct State {
    dump: Dump,
    /// Points to compare; `None` is every point the dump can answer.
    only: Option<Vec<Point>>,
    /// The next dump row of each module.
    cursor: BTreeMap<usize, usize>,
    /// The row every module starts at after a reset.
    start: usize,
    acc: BTreeMap<usize, Acc>,
    /// Rows tapped at points with no reference, in order of first appearance.
    skipped: Vec<(Point, usize)>,
}

impl State {
    fn plan(&mut self, point: Point, rows: usize) -> Plan {
        if self.only.as_ref().is_some_and(|o| !o.contains(&point)) {
            return Plan::Ignore;
        }
        let referenced = match module_of(point) {
            None => false,
            Some(MODULE_LOGITS) => true,
            Some(_) => self.dump.has_layers(),
        };
        if referenced {
            return Plan::Compare;
        }
        match self.skipped.iter_mut().find(|(p, _)| *p == point) {
            Some((_, n)) => *n += rows,
            None => self.skipped.push((point, rows)),
        }
        Plan::Skip
    }
}

/// The oracle tap. Clone it before installing: the clone given to the model records into the same state, and the one you keep
/// reads the results (`NemotronModel::set_tap` takes the tap by value).
#[derive(Clone)]
pub struct OracleTap {
    state: Arc<Mutex<State>>,
}

impl OracleTap {
    /// A tap comparing against sample `sample` of the dumps in `root`.
    pub fn open(root: &Path, sample: usize) -> Result<Self, String> {
        let dump = Dump::open(root, sample)?;
        Ok(Self {
            state: Arc::new(Mutex::new(State {
                dump,
                only: None,
                cursor: BTreeMap::new(),
                start: 0,
                acc: BTreeMap::new(),
                skipped: Vec::new(),
            })),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Compare only these points; a tap at any other point is ignored (an 8-layer model must not be held to `FinalNorm`).
    pub fn only(self, points: impl IntoIterator<Item = Point>) -> Self {
        self.lock().only = Some(points.into_iter().collect());
        self
    }

    /// The model's first tapped row is dump row `row` from now on (and after every [`Self::reset`]).
    pub fn seek(&self, row: usize) {
        let mut s = self.lock();
        s.start = row;
        s.cursor.clear();
    }

    /// Forgets every result and starts each point at the seek row (0 by default).
    pub fn reset(&self) {
        let mut s = self.lock();
        s.cursor.clear();
        s.acc.clear();
        s.skipped.clear();
    }

    /// The dump's tokens per sample.
    pub fn tokens(&self) -> usize {
        self.lock().dump.tokens()
    }

    /// Compares `rows` rows of `got` (`[rows][width]`, f32 on the host) at `point` with the next rows of the dump. The device tap is
    /// this plus a download.
    pub fn compare_host(&self, point: Point, got: &[f32], rows: usize, width: usize) -> Result<(), String> {
        if got.len() != rows * width {
            return Err(format!("{}: {} values for {rows} rows of {width}", label(point), got.len()));
        }
        let mut s = self.lock();
        match s.plan(point, rows) {
            Plan::Compare => Self::compare(&mut s, point, got, rows, width),
            Plan::Ignore | Plan::Skip => Ok(()),
        }
    }

    /// The comparison itself; the point has a reference.
    fn compare(s: &mut State, point: Point, got: &[f32], rows: usize, width: usize) -> Result<(), String> {
        let module = module_of(point).expect("a compared point is a dumped module");
        let logits = module == MODULE_LOGITS;
        if let Some(m) = s.dump.module_meta(module) {
            if m.width != width {
                return Err(format!(
                    "{}: the model's rows are {width} wide, the dump's module {module} ({}) is {}",
                    label(point),
                    m.key,
                    m.width
                ));
            }
        } else if !logits {
            return Err(format!("{}: the dump's meta.json has no module {module}", label(point)));
        }
        let first = s.cursor.get(&module).copied().unwrap_or(s.start);
        if first + rows > s.dump.tokens() {
            return Err(format!(
                "{}: rows {first}..{} are past the {} tokens of sample {}",
                label(point),
                first + rows,
                s.dump.tokens(),
                s.dump.sample()
            ));
        }
        let want = s.dump.read(module, first, rows, width)?;
        let (d2, r2, mx, nf) = sums(got, &want);
        let mut top1 = None;
        if logits {
            let recorded = &s.dump.top1()[first..first + rows];
            let mut t = Top1 { rows, agree_recorded: 0, agree_dumped: 0, fork_self_agree: 0, max_gap: 0.0 };
            for r in 0..rows {
                let mine = argmax(&got[r * width..(r + 1) * width]);
                let fork_row = &want[r * width..(r + 1) * width];
                let dumped = argmax(fork_row);
                let rec = recorded[r] as i64;
                t.agree_recorded += (mine as i64 == rec) as usize;
                t.agree_dumped += (mine == dumped) as usize;
                t.fork_self_agree += (dumped as i64 == rec) as usize;
                if mine as i64 != rec {
                    t.max_gap = t.max_gap.max(fork_row[dumped] - fork_row[mine]);
                }
            }
            top1 = Some(t);
        }
        let a = s.acc.entry(module).or_default();
        a.rows += rows;
        a.elems += (rows * width) as u64;
        a.sq_diff += d2;
        a.sq_ref += r2;
        a.max_abs = a.max_abs.max(mx);
        a.nonfinite += nf;
        if let Some(t) = top1 {
            match &mut a.top1 {
                None => a.top1 = Some(t),
                Some(x) => {
                    x.rows += t.rows;
                    x.agree_recorded += t.agree_recorded;
                    x.agree_dumped += t.agree_dumped;
                    x.fork_self_agree += t.fork_self_agree;
                    x.max_gap = x.max_gap.max(t.max_gap);
                }
            }
        }
        s.cursor.insert(module, first + rows);
        Ok(())
    }

    /// Everything compared so far, one entry per point, in module order.
    pub fn results(&self) -> Vec<Comparison> {
        let s = self.lock();
        s.acc
            .iter()
            .map(|(&module, a)| {
                let point = point_of(module).expect("only modules of the dump are recorded");
                let width = if a.rows == 0 { 0 } else { (a.elems / a.rows as u64) as usize };
                let rel_l2 = if a.sq_ref > 0.0 {
                    (a.sq_diff / a.sq_ref).sqrt()
                } else if a.sq_diff > 0.0 {
                    f64::INFINITY
                } else {
                    0.0
                };
                Comparison {
                    sample: s.dump.sample(),
                    point,
                    module,
                    key: s.dump.module_meta(module).map_or_else(|| "?".to_string(), |m| m.key.clone()),
                    rows: a.rows,
                    width,
                    rel_l2: if a.nonfinite > 0 { f64::NAN } else { rel_l2 },
                    max_abs: a.max_abs,
                    ref_rms: if a.elems == 0 { 0.0 } else { (a.sq_ref / a.elems as f64).sqrt() },
                    nonfinite: a.nonfinite,
                    top1: a.top1.clone(),
                }
            })
            .collect()
    }

    /// The points that were tapped but have no reference (mixer outputs, or layer points of a logits-only sample), with the rows seen.
    pub fn skipped(&self) -> Vec<(Point, usize)> {
        self.lock().skipped.clone()
    }
}

impl Tap for OracleTap {
    fn tap(&mut self, point: Point, buf: &DevBuf, rows: usize, width: usize, st: &Stream) -> Result<(), ModelError> {
        // A point with no reference, or filtered out, costs no download.
        if self.lock().plan(point, rows) != Plan::Compare {
            return Ok(());
        }
        let n = rows * width;
        if buf.bytes() < n * 4 {
            return Err(ModelError::Request(format!(
                "oracle tap {}: the buffer holds {} bytes, {rows} x {width} f32 needs {}",
                label(point),
                buf.bytes(),
                n * 4
            )));
        }
        // The model's kernels run on `st`; a readback on the legacy stream does not wait for a non-blocking one.
        st.sync()?;
        let mut host = vec![0f32; n];
        buf.download_range(0, &mut host)?;
        Self::compare(&mut self.lock(), point, &host, rows, width).map_err(ModelError::Request)
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Tolerances

/// One module's committed limit.
#[derive(Clone, Debug, PartialEq)]
pub struct Limit {
    pub module: usize,
    pub key: String,
    /// The fork's own whole-versus-chunked rel_l2 at this module (the larger of the two chunkings measured).
    pub spread: f64,
    /// What a comparison may reach: `max(2 * spread, floor)`.
    pub limit: f64,
}

/// The committed tolerances: the top-1 floor and one rel_l2 limit per module.
#[derive(Clone, Debug, PartialEq)]
pub struct Tolerances {
    pub top1_min: f64,
    pub limits: BTreeMap<usize, Limit>,
}

impl Tolerances {
    /// The numbers committed with the engine (`oracle_tolerances.json`).
    pub fn committed() -> Self {
        Self::from_json(COMMITTED_TOLERANCES).expect("the committed oracle_tolerances.json parses")
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("oracle tolerances: {e}"))?;
        let top1_min = v.get("top1_min").and_then(Value::as_f64).ok_or("oracle tolerances: no `top1_min`")?;
        let mods = v.get("modules").and_then(Value::as_array).ok_or("oracle tolerances: no `modules` list")?;
        let mut limits = BTreeMap::new();
        for m in mods {
            let module = m.get("module").and_then(Value::as_u64).ok_or("oracle tolerances: an entry has no `module`")? as usize;
            let num = |k: &str| {
                m.get(k).and_then(Value::as_f64).ok_or_else(|| format!("oracle tolerances: module {module} has no `{k}`"))
            };
            let key = m.get("key").and_then(Value::as_str).unwrap_or("?").to_string();
            let (spread, limit) = (num("spread")?, num("limit")?);
            if !(limit.is_finite() && limit > 0.0 && spread.is_finite() && spread >= 0.0) {
                return Err(format!("oracle tolerances: module {module} has spread {spread}, limit {limit}"));
            }
            if limits.insert(module, Limit { module, key, spread, limit }).is_some() {
                return Err(format!("oracle tolerances: module {module} appears twice"));
            }
        }
        Ok(Self { top1_min, limits })
    }

    /// Applies the limits to `results`: every comparison's rel_l2 must be at most its module's limit (a NaN fails), and the top-1
    /// agreement over all logits comparisons (every sample) must reach `top1_min`. `expect` lists the points that must have been
    /// compared; the ones that were not are reported (an empty `expect` accepts whatever was compared).
    pub fn apply(&self, results: &[Comparison], expect: &[Point]) -> Verdict {
        let mut failures = Vec::new();
        let mut checked = 0usize;
        for c in results {
            let Some(l) = self.limits.get(&c.module) else {
                failures.push(format!("{}: no committed limit for module {}", label(c.point), c.module));
                continue;
            };
            checked += 1;
            // written so that a NaN fails
            let within = c.rel_l2 <= l.limit;
            if !within {
                failures.push(format!(
                    "{} ({}): rel_l2 {:.3e} is over the limit {:.3e} ({:.2}x; fork spread {:.3e}, sample {})",
                    label(c.point),
                    c.key,
                    c.rel_l2,
                    l.limit,
                    c.rel_l2 / l.limit,
                    l.spread,
                    c.sample
                ));
            }
        }
        let (mut rows, mut agree) = (0usize, 0usize);
        for t in results.iter().filter_map(|c| c.top1.as_ref()) {
            rows += t.rows;
            agree += t.agree_recorded;
        }
        let top1 = (rows > 0).then_some((agree, rows));
        if let Some((a, r)) = top1 {
            if (a as f64) < self.top1_min * r as f64 {
                failures.push(format!(
                    "top-1 agreement {a}/{r} = {:.2}% is under {:.0}%",
                    100.0 * a as f64 / r as f64,
                    100.0 * self.top1_min
                ));
            }
        }
        let not_compared: Vec<Point> = expect.iter().copied().filter(|p| !results.iter().any(|c| c.point == *p)).collect();
        for p in &not_compared {
            failures.push(format!("{}: expected a comparison and none was made", label(*p)));
        }
        Verdict { failures, checked, not_compared, top1 }
    }
}

/// The outcome of [`Tolerances::apply`].
#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    /// One line per failure, naming the point.
    pub failures: Vec<String>,
    /// Comparisons held to a limit.
    pub checked: usize,
    /// Expected points with no comparison.
    pub not_compared: Vec<Point>,
    /// `(rows that agree with the recorded top-1, logits rows)` over every sample, when any logits were compared.
    pub top1: Option<(usize, usize)>,
}

impl Verdict {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The table

/// A fixed-width table of `results`, one line per point; with `tol`, the limit, the ratio and `ok` or `FAIL` per line, then the top-1
/// totals. Pack-derived values are numbers only.
pub fn table(results: &[Comparison], tol: Option<&Tolerances>) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{:<8} {:>3} {:<22} {:>6} {:>10} {:>10} {:>10} {:>10} {:>7}  status",
        "point", "smp", "module", "rows", "rel_l2", "max_abs", "ref_rms", "limit", "x lim"
    );
    for c in results {
        let (limit, ratio, status) = match tol.and_then(|t| t.limits.get(&c.module)) {
            Some(l) => (
                format!("{:.3e}", l.limit),
                format!("{:.2}", c.rel_l2 / l.limit),
                if c.rel_l2 <= l.limit { "ok" } else { "FAIL" }.to_string(),
            ),
            None => ("-".into(), "-".into(), "-".into()),
        };
        let _ = writeln!(
            s,
            "{:<8} {:>3} {:<22} {:>6} {:>10.3e} {:>10.3e} {:>10.3e} {:>10} {:>7}  {}{}",
            label(c.point),
            c.sample,
            c.key,
            c.rows,
            c.rel_l2,
            c.max_abs,
            c.ref_rms,
            limit,
            ratio,
            status,
            if c.nonfinite > 0 { format!("  ({} non-finite)", c.nonfinite) } else { String::new() }
        );
    }
    let (mut rows, mut rec, mut dumped, mut selfa) = (0, 0, 0, 0);
    let mut gap = 0f32;
    for t in results.iter().filter_map(|c| c.top1.as_ref()) {
        rows += t.rows;
        rec += t.agree_recorded;
        dumped += t.agree_dumped;
        selfa += t.fork_self_agree;
        gap = gap.max(t.max_gap);
    }
    if rows > 0 {
        let _ = writeln!(
            s,
            "top-1 over {rows} rows: {rec} agree with the recorded argmax ({:.2}%), {dumped} with the argmax of the dumped f16 rows ({:.2}%); \
             the fork against itself: {selfa}; largest fork-logit gap at a disagreement {gap:.3}",
            100.0 * rec as f64 / rows as f64,
            100.0 * dumped as f64 / rows as f64,
        );
        if let Some(t) = tol {
            let _ = writeln!(s, "top-1 floor {:.0}%", 100.0 * t.top1_min);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic dump of `tokens` rows: modules 0..3 of width 8 (embed, two layers, ...) are named by `modules`, logits of width
    /// 16. Values are small integers so every f16 is exact.
    fn write_dump(dir: &Path, sample: usize, tokens: usize, with_layers: bool) {
        std::fs::create_dir_all(dir).unwrap();
        let logits_w = 16usize;
        let hidden_w = 8usize;
        let mut top1 = Vec::new();
        let mut logits = Vec::new();
        for t in 0..tokens {
            let top = (t * 5) % logits_w;
            top1.extend((top as i32).to_le_bytes());
            for j in 0..logits_w {
                let v = if j == top { 10.0f32 } else { (j % 3) as f32 };
                logits.extend(f32_to_f16_bits(v).to_le_bytes());
            }
        }
        std::fs::write(dir.join(format!("top1_{sample}.i32")), &top1).unwrap();
        std::fs::write(dir.join(format!("logits_{sample}.f16")), &logits).unwrap();
        let ids: Vec<u8> = (0..tokens as u32).flat_map(|i| i.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("ids_{sample}.u32")), ids).unwrap();
        if !with_layers {
            return;
        }
        let ld = dir.join(format!("layers{sample}"));
        std::fs::create_dir_all(&ld).unwrap();
        let mut mods = Vec::new();
        for m in 0..=MODULE_LOGITS {
            let w = if m == MODULE_LOGITS { logits_w } else { hidden_w };
            mods.push(serde_json::json!({"i": m, "class": "X", "key": format!("mod{m}"), "shape": [1, tokens, w]}));
            // module files exist for the embedding, layer 0, layer 1 and the logits only
            if m <= 2 || m == MODULE_LOGITS {
                let mut bytes = Vec::new();
                for t in 0..tokens {
                    for j in 0..w {
                        let v = if m == MODULE_LOGITS {
                            if j == (t * 5) % logits_w {
                                10.0
                            } else {
                                (j % 3) as f32
                            }
                        } else {
                            ((t + j + m) % 7) as f32 - 3.0
                        };
                        bytes.extend(f32_to_f16_bits(v).to_le_bytes());
                    }
                }
                std::fs::write(ld.join(module_file(m)), bytes).unwrap();
            }
        }
        std::fs::write(ld.join("meta.json"), serde_json::json!({"n": tokens, "modules": mods}).to_string()).unwrap();
    }

    /// Small integers only: the f16 bits of an integer in [-16, 16].
    fn f32_to_f16_bits(v: f32) -> u16 {
        if v == 0.0 {
            return 0;
        }
        let sign = if v < 0.0 { 0x8000u16 } else { 0 };
        let a = v.abs();
        let e = a.log2().floor() as i32;
        let m = ((a / 2f32.powi(e) - 1.0) * 1024.0).round() as u16;
        sign | (((e + 15) as u16) << 10) | m
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rexl3-oracle-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn dump_rows(dir: &Path, sample: usize, module: usize, rows: usize, width: usize) -> Vec<f32> {
        let d = Dump::open(dir, sample).unwrap();
        d.read(module, 0, rows, width).unwrap()
    }

    #[test]
    fn modules_and_points_map_both_ways() {
        assert_eq!(module_of(Point::Embed), Some(0));
        assert_eq!(module_of(Point::Layer(0)), Some(1));
        assert_eq!(module_of(Point::Layer(87)), Some(88));
        assert_eq!(module_of(Point::FinalNorm), Some(89));
        assert_eq!(module_of(Point::Logits), Some(90));
        assert_eq!(module_of(Point::Layer(88)), None);
        assert_eq!(module_of(Point::Mixer(3)), None);
        for m in 0..MODULES {
            assert_eq!(module_of(point_of(m).unwrap()), Some(m));
        }
        assert_eq!(point_of(MODULES), None);
        assert_eq!(module_file(7), "m007.f16");
    }

    #[test]
    fn the_f16_reader_is_exact_and_reads_row_ranges() {
        let d = tmp("reader");
        write_dump(&d, 0, 6, true);
        let all = dump_rows(&d, 0, 1, 6, 8);
        let part = read_f16_rows(&d.join("layers0").join(module_file(1)), 2, 3, 8).unwrap();
        assert_eq!(part, all[2 * 8..5 * 8]);
        assert!(all.iter().all(|v| v.fract() == 0.0 && v.abs() <= 3.0), "{all:?}");
        let e = read_f16_rows(&d.join("layers0").join(module_file(1)), 4, 3, 8).unwrap_err();
        assert!(e.contains("past the 6 rows"), "{e}");
        let e = read_f16_rows(&d.join("layers0").join(module_file(1)), 0, 1, 7).unwrap_err();
        assert!(e.contains("not a whole number of rows of 7"), "{e}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn the_dump_itself_compares_to_zero_and_full_top1_in_chunks() {
        let d = tmp("self");
        write_dump(&d, 0, 12, true);
        let tap = OracleTap::open(&d, 0).unwrap();
        assert_eq!(tap.tokens(), 12);
        // feed the dump back in uneven chunks: 5 + 4 + 3 rows
        for (m, w) in [(0usize, 8usize), (1, 8), (2, 8), (MODULE_LOGITS, 16)] {
            let all = dump_rows(&d, 0, m, 12, w);
            let mut at = 0;
            for n in [5usize, 4, 3] {
                tap.compare_host(point_of(m).unwrap(), &all[at * w..(at + n) * w], n, w).unwrap();
                at += n;
            }
        }
        let r = tap.results();
        assert_eq!(r.iter().map(|c| c.module).collect::<Vec<_>>(), vec![0, 1, 2, MODULE_LOGITS]);
        for c in &r {
            assert_eq!((c.rel_l2, c.max_abs, c.rows, c.nonfinite), (0.0, 0.0, 12, 0), "{c:?}");
        }
        let t = r.last().unwrap().top1.clone().unwrap();
        assert_eq!((t.rows, t.agree_recorded, t.agree_dumped, t.fork_self_agree, t.max_gap), (12, 12, 12, 12, 0.0));
        assert_eq!(r[1].key, "mod1");
        assert_eq!(r[1].point, Point::Layer(0));
        // a thirteenth row is past the dump
        let e = tap.compare_host(Point::Embed, &[0.0; 8], 1, 8).unwrap_err();
        assert!(e.contains("rows 12..13 are past the 12 tokens"), "{e}");
        // reset starts over
        tap.reset();
        assert!(tap.results().is_empty());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn known_noise_gives_the_known_rel_l2_and_a_known_top1_loss() {
        let d = tmp("noise");
        write_dump(&d, 0, 20, true);
        let tap = OracleTap::open(&d, 0).unwrap();
        let w = 8usize;
        let base = dump_rows(&d, 0, 1, 20, w);
        // scale by (1 + 0.01): rel_l2 = 0.01, max_abs = 0.01 * max|b|
        let scaled: Vec<f32> = base.iter().map(|v| v * 1.01).collect();
        tap.compare_host(Point::Layer(0), &scaled, 20, w).unwrap();
        // a single element off by 1: rel_l2 = 1 / ||b||
        let mut poked = dump_rows(&d, 0, 2, 20, w);
        let norm: f64 = poked.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
        poked[37] += 1.0;
        tap.compare_host(Point::Layer(1), &poked, 20, w).unwrap();
        // logits: swap the top-2 on 4 rows (4 of 20 disagree)
        let lw = 16usize;
        let mut logits = dump_rows(&d, 0, MODULE_LOGITS, 20, lw);
        for r in [0usize, 3, 7, 19] {
            let row = &mut logits[r * lw..(r + 1) * lw];
            let top = argmax(row);
            let other = (top + 1) % lw;
            row[other] = 11.0;
        }
        tap.compare_host(Point::Logits, &logits, 20, lw).unwrap();
        let r = tap.results();
        let l0 = &r[0];
        assert!((l0.rel_l2 - 0.01).abs() < 1e-6, "{}", l0.rel_l2);
        let l1 = &r[1];
        assert!((l1.rel_l2 - 1.0 / norm).abs() < 1e-12, "{} vs {}", l1.rel_l2, 1.0 / norm);
        assert_eq!(l1.max_abs, 1.0);
        let t = r[2].top1.clone().unwrap();
        assert_eq!((t.rows, t.agree_recorded, t.agree_dumped), (20, 16, 16));
        // the fork's own logits at the model's picks are 0 or 1 (the dump's filler), against its top-1 of 10: the largest gap is 10
        assert_eq!(t.max_gap, 10.0);
    }

    #[test]
    fn a_missing_or_short_file_and_a_wrong_width_are_clear_errors() {
        let d = tmp("errors");
        write_dump(&d, 0, 6, true);
        // module 5 has no file
        let tap = OracleTap::open(&d, 0).unwrap();
        let e = tap.compare_host(Point::Layer(4), &[0.0; 8], 1, 8).unwrap_err();
        assert!(e.contains("m005.f16") && e.contains("No such file"), "{e}");
        // the wrong width
        let e = tap.compare_host(Point::Layer(0), &[0.0; 16], 1, 16).unwrap_err();
        assert!(e.contains("16 wide") && e.contains("is 8"), "{e}");
        // an input of the wrong length
        let e = tap.compare_host(Point::Layer(0), &[0.0; 9], 1, 8).unwrap_err();
        assert!(e.contains("9 values for 1 rows of 8"), "{e}");
        // a directory with no top-1 file
        let empty = tmp("errors-empty");
        std::fs::create_dir_all(&empty).unwrap();
        let e = OracleTap::open(&empty, 0).err().unwrap();
        assert!(e.contains("top1_0.i32"), "{e}");
        // logits missing
        std::fs::remove_file(d.join("logits_0.f16")).unwrap();
        let e = OracleTap::open(&d, 0).err().unwrap();
        assert!(e.contains("logits_0.f16"), "{e}");
        std::fs::remove_dir_all(&d).ok();
        std::fs::remove_dir_all(&empty).ok();
    }

    #[test]
    fn mixer_points_a_filtered_point_and_a_logits_only_sample_are_not_errors() {
        let d = tmp("skip");
        write_dump(&d, 0, 6, true);
        write_dump(&d, 1, 6, false);
        // a mixer output has no reference: counted, not compared, not an error
        let tap = OracleTap::open(&d, 0).unwrap();
        tap.compare_host(Point::Mixer(0), &[0.0; 8], 1, 8).unwrap();
        tap.compare_host(Point::Mixer(0), &[0.0; 16], 2, 8).unwrap();
        assert_eq!(tap.skipped(), vec![(Point::Mixer(0), 3)]);
        assert!(tap.results().is_empty());
        // a point outside `only` is ignored altogether
        let tap = OracleTap::open(&d, 0).unwrap().only([Point::Embed]);
        tap.compare_host(Point::Logits, &[0.0; 16], 1, 16).unwrap();
        tap.compare_host(Point::Mixer(0), &[0.0; 8], 1, 8).unwrap();
        tap.compare_host(Point::Embed, &dump_rows(&d, 0, 0, 1, 8), 1, 8).unwrap();
        assert!(tap.skipped().is_empty());
        assert_eq!(tap.results().len(), 1);

        let t1 = OracleTap::open(&d, 1).unwrap();
        assert!(!Dump::open(&d, 1).unwrap().has_layers());
        t1.compare_host(Point::Layer(3), &[0.0; 8], 1, 8).unwrap();
        assert_eq!(t1.skipped(), vec![(Point::Layer(3), 1)]);
        let lg = dump_rows(&d, 1, MODULE_LOGITS, 6, 16);
        t1.compare_host(Point::Logits, &lg, 6, 16).unwrap();
        assert_eq!(t1.results()[0].top1.as_ref().unwrap().agree_recorded, 6);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn seek_places_the_first_tapped_row() {
        let d = tmp("seek");
        write_dump(&d, 0, 10, true);
        let tap = OracleTap::open(&d, 0).unwrap();
        tap.seek(4);
        let all = dump_rows(&d, 0, 1, 10, 8);
        tap.compare_host(Point::Layer(0), &all[4 * 8..7 * 8], 3, 8).unwrap();
        tap.compare_host(Point::Layer(0), &all[7 * 8..10 * 8], 3, 8).unwrap();
        let c = &tap.results()[0];
        assert_eq!((c.rows, c.rel_l2), (6, 0.0));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn the_committed_tolerances_parse_and_apply() {
        let t = Tolerances::committed();
        assert_eq!(t.top1_min, TOP1_MIN);
        assert_eq!(t.limits.len(), MODULES, "one limit per dump module");
        for m in 0..MODULES {
            let l = &t.limits[&m];
            assert!(l.limit >= 2.0 * l.spread && l.limit > 0.0, "module {m}: {l:?}");
        }
        // apply on synthetic comparisons
        let ok = Comparison {
            sample: 0,
            point: Point::Layer(5),
            module: 6,
            key: "k".into(),
            rows: 1,
            width: 8,
            rel_l2: 0.0,
            max_abs: 0.0,
            ref_rms: 1.0,
            nonfinite: 0,
            top1: None,
        };
        let v = t.apply(std::slice::from_ref(&ok), &[Point::Layer(5)]);
        assert!(v.passed() && v.checked == 1 && v.top1.is_none(), "{v:?}");
        let mut over = ok.clone();
        over.rel_l2 = t.limits[&6].limit * 1.01;
        let v = t.apply(&[over], &[]);
        assert!(!v.passed() && v.failures[0].starts_with("L5 (k): rel_l2"), "{v:?}");
        let mut nan = ok.clone();
        nan.rel_l2 = f64::NAN;
        assert!(!t.apply(&[nan], &[]).passed(), "a NaN must fail");
        let v = t.apply(&[ok], &[Point::Layer(5), Point::FinalNorm]);
        assert_eq!(v.not_compared, vec![Point::FinalNorm]);
        assert!(!v.passed());
        // top-1 over two samples
        let lg = |sample, rows, agree| Comparison {
            sample,
            point: Point::Logits,
            module: MODULE_LOGITS,
            key: "lm_head".into(),
            rows,
            width: 16,
            rel_l2: 0.0,
            max_abs: 0.0,
            ref_rms: 1.0,
            nonfinite: 0,
            top1: Some(Top1 { rows, agree_recorded: agree, agree_dumped: agree, fork_self_agree: rows, max_gap: 0.0 }),
        };
        let v = t.apply(&[lg(0, 1024, 990), lg(1, 1024, 940)], &[]);
        assert_eq!(v.top1, Some((1930, 2048)));
        assert!(!v.passed() && v.failures[0].contains("top-1 agreement 1930/2048 = 94.24%"), "{v:?}");
        let v = t.apply(&[lg(0, 1024, 990), lg(1, 1024, 960)], &[]);
        assert!(v.passed(), "1950/2048 = 95.2%: {v:?}");
    }

    #[test]
    fn the_table_prints_a_line_per_point_and_the_top1_totals() {
        let c = Comparison {
            sample: 0,
            point: Point::Layer(0),
            module: 1,
            key: "backbone.layers.0".into(),
            rows: 1024,
            width: 4096,
            rel_l2: 1.5e-3,
            max_abs: 0.25,
            ref_rms: 1.0,
            nonfinite: 0,
            top1: None,
        };
        let s = table(&[c], Some(&Tolerances::committed()));
        assert!(s.contains("L0") && s.contains("backbone.layers.0") && s.contains("1.500e-3"), "{s}");
        assert!(s.lines().count() == 2, "{s}");
    }
}
