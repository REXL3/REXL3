//! Read every `*.safetensors` header in a model folder (no weight data) and sort tensors into EXL3
//! groups, plain tensors, and groups that fail to parse. Port of TensorFold `format.scan` /
//! `format.parse_group`: a group is refused for the same reasons ExLlamaV3's LinearEXL3 would refuse it.
//!
//! Headers are untrusted input. Every length, offset and size is checked with checked arithmetic against the
//! real file size, and anything malformed lands in `Checkpoint::issues` (or in `bad` when it belongs to an EXL3
//! group) instead of being dropped, wrapped or aborting the scan.

use crate::f16::f16_to_f64;
use crate::loadstats::{self, Stage};
use crate::{unpack_signs, Bits, Codebook, HAD, MARKER_MCG, MARKER_MUL1};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const PARTS: [&str; 8] = ["trellis", "suh", "su", "svh", "sv", "mcg", "mul1", "bias"];

/// The safetensors format caps the JSON header at 100 MB.
pub const MAX_HEADER_BYTES: u64 = 100_000_000;

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub file: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offsets: (u64, u64),
}

impl Entry {
    /// Bytes on disk. `scan` rejects entries whose end is before their start, so this never saturates for
    /// an entry it returned.
    pub fn nbytes(&self) -> u64 {
        self.offsets.1.saturating_sub(self.offsets.0)
    }
}

/// Bytes per element of a safetensors dtype, if known.
pub fn dtype_size(dtype: &str) -> Option<u64> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => Some(1),
        "F16" | "BF16" | "I16" | "U16" => Some(2),
        "F32" | "I32" | "U32" => Some(4),
        "F64" | "I64" | "U64" => Some(8),
        _ => None,
    }
}

/// One EXL3 linear layer's metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub prefix: String,
    pub bits: Bits,
    pub codebook: Codebook,
    pub k: usize,
    pub n: usize,
    /// "suh" (fp16 scales) or "su" (packed signs)
    pub in_scales: &'static str,
    /// "svh" or "sv"
    pub out_scales: &'static str,
    pub bias: bool,
    pub files: Vec<String>,
    /// Bytes on disk of every part of the group.
    pub bytes: u64,
}

impl Group {
    pub fn trellis_bytes(&self) -> u64 {
        (self.k as u64) * (self.n as u64) * self.bits.twice() as u64 / 16
    }
}

#[derive(Debug, Default)]
pub struct Checkpoint {
    pub root: PathBuf,
    pub groups: BTreeMap<String, Group>,
    pub plain: BTreeMap<String, Entry>,
    pub bad: BTreeMap<String, String>,
    pub config: Value,
    pub markers: BTreeMap<String, u32>,
    /// Everything that looked wrong but did not belong to a refused group: malformed or duplicate tensor
    /// entries, a missing or unparsable config or index, shards the index lists but the folder lacks.
    /// A checkpoint with issues should not be served without a human looking at them.
    pub issues: Vec<String>,
    header_sizes: BTreeMap<String, u64>,
    /// Every tensor of every accepted group, by full name, so `load_group` can read them.
    parts: BTreeMap<String, Entry>,
}

/// One group's tensors read from disk, in the form `forward` takes.
#[derive(Clone, Debug)]
pub struct LoadedGroup {
    pub group: Group,
    pub trellis: Vec<i16>,
    pub kt: usize,
    pub nt: usize,
    /// Input scales: fp16 `suh` as floats, or +-1 from packed `su` signs.
    pub suh: Vec<f64>,
    pub svh: Vec<f64>,
    pub bias: Option<Vec<f64>>,
}

/// The tensors of one EXL3 group (see [`Checkpoint::group_parts`]).
#[derive(Clone, Copy, Debug)]
pub struct GroupParts<'a> {
    pub group: &'a Group,
    pub trellis: &'a Entry,
    /// `suh` (fp16) or `su` (packed signs), per `group.in_scales`.
    pub in_scales: &'a Entry,
    pub out_scales: &'a Entry,
    pub bias: Option<&'a Entry>,
}

fn invalid(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

/// Read and check a safetensors header: returns (header bytes, data bytes after the header, tensors).
/// The length prefix is capped at `MAX_HEADER_BYTES` and must fit in the file, so a hostile or truncated file
/// is an error and never a huge allocation.
pub fn read_header_checked(path: &Path) -> std::io::Result<(u64, u64, BTreeMap<String, Value>)> {
    let ctx = |e: Error| Error::new(e.kind(), format!("{}: {e}", path.display()));
    let mut f = File::open(path).map_err(ctx)?;
    let file_len = f.metadata().map_err(ctx)?.len();
    if file_len < 8 {
        return Err(ctx(invalid("shorter than the 8-byte header length")));
    }
    let mut len = [0u8; 8];
    f.read_exact(&mut len).map_err(ctx)?;
    let size = u64::from_le_bytes(len);
    if size > MAX_HEADER_BYTES {
        return Err(ctx(invalid(format!("header length {size} exceeds the {MAX_HEADER_BYTES}-byte cap"))));
    }
    let data_len = file_len
        .checked_sub(8)
        .and_then(|r| r.checked_sub(size))
        .ok_or_else(|| ctx(invalid(format!("header length {size} runs past the end of a {file_len}-byte file"))))?;
    let mut buf = vec![0u8; size as usize];
    f.read_exact(&mut buf).map_err(ctx)?;
    let v: Value = serde_json::from_slice(&buf).map_err(|e| ctx(invalid(e.to_string())))?;
    let mut map: BTreeMap<String, Value> = match v {
        Value::Object(m) => m.into_iter().collect(),
        _ => return Err(ctx(invalid("header is not an object"))),
    };
    map.remove("__metadata__");
    Ok((size, data_len, map))
}

pub fn read_header(path: &Path) -> std::io::Result<(u64, BTreeMap<String, Value>)> {
    let (size, _, map) = read_header_checked(path)?;
    Ok((size, map))
}

fn entry_of(file: &str, v: &Value) -> Result<Entry, String> {
    let dtype = v.get("dtype").and_then(|d| d.as_str()).ok_or("missing dtype")?.to_string();
    let shape = v
        .get("shape")
        .and_then(|s| s.as_array())
        .ok_or("missing shape")?
        .iter()
        .map(|d| d.as_u64().and_then(|x| usize::try_from(x).ok()).ok_or("shape holds a non-integer"))
        .collect::<Result<Vec<_>, _>>()?;
    let off = v.get("data_offsets").and_then(|o| o.as_array()).ok_or("missing data_offsets")?;
    if off.len() != 2 {
        return Err(format!("data_offsets has {} values, expected 2", off.len()));
    }
    let (a, b) =
        (off[0].as_u64().ok_or("data_offsets holds a non-integer")?, off[1].as_u64().ok_or("data_offsets holds a non-integer")?);
    Ok(Entry { file: file.to_string(), dtype, shape, offsets: (a, b) })
}

/// Offsets must be ordered and inside the file, and, for a known dtype, span exactly shape x element size.
fn validate_entry(e: &Entry, data_len: u64) -> Result<(), String> {
    let (a, b) = e.offsets;
    if b < a {
        return Err(format!("data_offsets end {b} is before start {a}"));
    }
    if b > data_len {
        return Err(format!("data_offsets end {b} is past the {data_len} data bytes in {}", e.file));
    }
    if let Some(sz) = dtype_size(&e.dtype) {
        let numel = e.shape.iter().try_fold(1u64, |acc, &d| acc.checked_mul(d as u64)).ok_or("shape element count overflows")?;
        let want = numel.checked_mul(sz).ok_or("tensor byte size overflows")?;
        if b - a != want {
            return Err(format!("{} {:?} needs {want} bytes but data_offsets span {}", e.dtype, e.shape, b - a));
        }
    }
    Ok(())
}

/// Parse one group from its parts; Err(reason) for a group LinearEXL3 would not load.
pub fn parse_group(prefix: &str, parts: &BTreeMap<String, Entry>) -> Result<Group, String> {
    let tr = parts.get("trellis").ok_or("no .trellis")?;
    if tr.dtype != "I16" || tr.shape.len() != 3 {
        return Err(format!("trellis must be int16 [K/16, N/16, 16 * bits], got {} {:?}", tr.dtype, tr.shape));
    }
    if tr.shape.contains(&0) {
        return Err(format!("trellis {:?} has an empty dimension", tr.shape));
    }
    let bits = Bits::from_trellis_last(tr.shape[2]).ok_or_else(|| format!("unsupported trellis width {}", tr.shape[2]))?;
    let has: Vec<&str> = ["mcg", "mul1"].into_iter().filter(|m| parts.contains_key(*m)).collect();
    if has.len() > 1 {
        return Err("both mcg and mul1 markers".into());
    }
    for m in &has {
        let e = &parts[*m];
        if e.dtype != "I32" || e.nbytes() != 4 {
            return Err(format!(".{m} marker must be one int32, got {} {:?}", e.dtype, e.shape));
        }
    }
    let cb = match has.first() {
        Some(&"mcg") => Codebook::Mcg,
        Some(&"mul1") => Codebook::Mul1,
        _ => Codebook::Inst3,
    };
    if !bits.is_integer() && cb != Codebook::Mul1 {
        return Err(format!("{}-bit tiles need the mul1 codebook, found {}", bits.as_f64(), cb.name()));
    }
    let (k, n) = (16 * tr.shape[0], 16 * tr.shape[1]);
    if parts.contains_key("suh") && parts.contains_key("su") {
        return Err("both .suh and .su input scales".into());
    }
    if parts.contains_key("svh") && parts.contains_key("sv") {
        return Err("both .svh and .sv output scales".into());
    }
    let (ins, want_in): (&'static str, (&str, Vec<usize>)) = if parts.contains_key("suh") {
        ("suh", ("F16", vec![k]))
    } else if parts.contains_key("su") {
        ("su", ("I16", vec![k / 16]))
    } else {
        return Err("no input scales (.suh or .su)".into());
    };
    let e = &parts[ins];
    if (e.dtype.as_str(), &e.shape) != (want_in.0, &want_in.1) {
        return Err(format!(".{ins} must be {} {:?}, got {} {:?}", want_in.0, want_in.1, e.dtype, e.shape));
    }
    let (outs, want_out): (&'static str, (&str, Vec<usize>)) = if parts.contains_key("svh") {
        ("svh", ("F16", vec![n]))
    } else if parts.contains_key("sv") {
        ("sv", ("I16", vec![n / 16]))
    } else {
        return Err("no output scales (.svh or .sv)".into());
    };
    let e = &parts[outs];
    if (e.dtype.as_str(), &e.shape) != (want_out.0, &want_out.1) {
        return Err(format!(".{outs} must be {} {:?}, got {} {:?}", want_out.0, want_out.1, e.dtype, e.shape));
    }
    if let Some(b) = parts.get("bias") {
        if !matches!(b.dtype.as_str(), "F16" | "BF16" | "F32") || b.shape != [n] {
            return Err(format!(".bias must be a float vector of {n}, got {} {:?}", b.dtype, b.shape));
        }
    }
    if k % HAD != 0 || n % HAD != 0 {
        return Err(format!("K={k} and N={n} must be multiples of {HAD} (the Hadamard blocks)"));
    }
    let files: BTreeSet<String> = parts.values().map(|e| e.file.clone()).collect();
    Ok(Group {
        prefix: prefix.to_string(),
        bits,
        codebook: cb,
        k,
        n,
        in_scales: ins,
        out_scales: outs,
        bias: parts.contains_key("bias"),
        files: files.into_iter().collect(),
        bytes: parts.values().map(|e| e.nbytes()).sum(),
    })
}

/// Byte offset of a tensor's data in its file: the 8-byte length prefix, the header, then the tensor's own offset.
fn tensor_start(header_size: u64, e: &Entry) -> std::io::Result<u64> {
    8u64.checked_add(header_size).and_then(|s| s.checked_add(e.offsets.0)).ok_or_else(|| invalid("tensor offset overflows"))
}

fn read_bytes(path: &Path, header_size: u64, e: &Entry) -> std::io::Result<Vec<u8>> {
    let start = tensor_start(header_size, e)?;
    let len = usize::try_from(e.nbytes()).map_err(|_| invalid("tensor too large for this platform"))?;
    let _t = loadstats::timer(Stage::Read, len as u64);
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(start))?;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

/// Tensors at least this large are read by several threads; smaller ones by one, as before.
const PARALLEL_READ_MIN_BYTES: u64 = 4 << 20;
/// Bytes a reader thread pulls per `pread`: small enough to stay in cache between the read and the widening copy.
const READ_PIECE_BYTES: usize = 1 << 20;

/// Reader threads for one large tensor: `REXL3_LOAD_THREADS` (1 = serial), else the core count capped at 8.
fn read_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("REXL3_LOAD_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()).min(8))
    })
}

/// `count` little-endian 16-bit values starting at byte `start` of `f`, each passed through `conv`. With `threads > 1`
/// the output is split into contiguous parts that scoped threads fill with `pread` through a small scratch buffer, so
/// there is no tensor-sized intermediate buffer and the page faults of the output are taken in parallel. The values are
/// exactly those of reading every byte and widening with `chunks_exact(2)`.
#[cfg(unix)]
fn read_le16_at<T, F>(f: &File, start: u64, count: usize, threads: usize, conv: F) -> std::io::Result<Vec<T>>
where
    T: Copy + Default + Send,
    F: Fn(u16) -> T + Sync,
{
    use std::os::unix::fs::FileExt;
    let mut out = vec![T::default(); count];
    let fill = |dst: &mut [T], first: usize| -> std::io::Result<()> {
        let mut scratch = vec![0u8; READ_PIECE_BYTES.min(dst.len() * 2)];
        let mut done = 0;
        while done < dst.len() {
            let n = (scratch.len() / 2).min(dst.len() - done);
            f.read_exact_at(&mut scratch[..n * 2], start + ((first + done) as u64) * 2)?;
            for (o, b) in dst[done..done + n].iter_mut().zip(scratch.chunks_exact(2)) {
                *o = conv(u16::from_le_bytes([b[0], b[1]]));
            }
            done += n;
        }
        Ok(())
    };
    if threads <= 1 || count == 0 {
        fill(&mut out, 0)?;
    } else {
        let per = count.div_ceil(threads.min(count));
        std::thread::scope(|s| {
            let fill = &fill;
            let handles: Vec<_> = out.chunks_mut(per).enumerate().map(|(i, dst)| s.spawn(move || fill(dst, i * per))).collect();
            handles.into_iter().try_for_each(|h| h.join().unwrap_or_else(|_| Err(Error::other("reader thread panicked"))))
        })?;
    }
    Ok(out)
}

#[cfg(not(unix))]
fn read_le16_at<T, F>(f: &File, start: u64, count: usize, _threads: usize, conv: F) -> std::io::Result<Vec<T>>
where
    T: Copy + Default + Send,
    F: Fn(u16) -> T + Sync,
{
    let mut f = f.try_clone()?;
    f.seek(SeekFrom::Start(start))?;
    let mut b = vec![0u8; count * 2];
    f.read_exact(&mut b)?;
    Ok(b.chunks_exact(2).map(|c| conv(u16::from_le_bytes([c[0], c[1]]))).collect())
}

fn read_marker(path: &Path, header_size: u64, e: &Entry) -> Result<u32, String> {
    if e.dtype != "I32" || e.nbytes() != 4 {
        return Err(format!("marker must be one int32, got {} {:?}", e.dtype, e.shape));
    }
    let b = read_bytes(path, header_size, e).map_err(|err| format!("cannot read marker: {err}"))?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// The `weight_map` of a `model.safetensors.index.json` (tensor name -> shard file), if the folder has one.
fn index_files(root: &Path, issues: &mut Vec<String>) -> Option<BTreeMap<String, String>> {
    let p = root.join("model.safetensors.index.json");
    if !p.exists() {
        return None;
    }
    let parsed =
        std::fs::read(&p).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice::<Value>(&b).map_err(|e| e.to_string()));
    match parsed {
        Ok(v) => match v.get("weight_map").and_then(|m| m.as_object()) {
            Some(m) => Some(m.iter().filter_map(|(k, f)| f.as_str().map(|f| (k.clone(), f.to_string()))).collect()),
            None => {
                issues.push("model.safetensors.index.json has no weight_map".into());
                None
            }
        },
        Err(e) => {
            issues.push(format!("model.safetensors.index.json: {e}"));
            None
        }
    }
}

/// Scan a model folder's safetensors headers into a `Checkpoint`.
///
/// If a `model.safetensors.index.json` is present only the shards it lists are scanned, so a stray file in the
/// folder cannot inject tensors, and a tensor stored in two shards resolves to the copy the index names (the
/// other is recorded in `issues`). Without an index, a duplicate is malformed. An unreadable shard header is an
/// error (the checkpoint is unusable); anything narrower is recorded in `issues` or `bad`.
///
/// The resolution is pairwise: it decides between the copy it holds and the one it meets next, and a name that
/// three or more shards list, with the index's shard none of the first two met, is refused as malformed (which is
/// what a pack with an overlay shard on top of a published duplicate looks like). [`scan_indexed`] resolves any
/// number of copies; this function's behaviour is unchanged.
pub fn scan(root: &Path, read_markers: bool) -> std::io::Result<Checkpoint> {
    scan_impl(root, read_markers, false)
}

/// [`scan`], with the index as the only authority on placement, exactly as the Nemotron engine's `Inventory` has it:
/// when the folder has a `model.safetensors.index.json`, a tensor name is taken from the shard the index maps it to and
/// from no other, however many shards list it. A header entry the index places in another shard, or does not name at
/// all, is a stray: recorded in `issues` ("also stored in ...; using ... as the index says"), never loaded, never
/// checked. A name the index maps to a shard whose header does not list it is an issue as well (and the group that
/// needs it is refused later, as with any missing part). Without an index it is [`scan`] (a duplicate is malformed).
///
/// Everything else (what a group is, the refusal reasons, the markers, `plain`) is `scan`'s. On a pack whose duplicates
/// are all pairs naming one of the two, and whose headers list no unindexed name, the two return the same entries.
pub fn scan_indexed(root: &Path, read_markers: bool) -> std::io::Result<Checkpoint> {
    scan_impl(root, read_markers, true)
}

fn scan_impl(root: &Path, read_markers: bool, index_places: bool) -> std::io::Result<Checkpoint> {
    let mut ck = Checkpoint { root: root.to_path_buf(), ..Default::default() };
    let mut files: Vec<PathBuf> = std::fs::read_dir(root)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    files.sort();
    let weight_map = index_files(root, &mut ck.issues);
    if let Some(wm) = &weight_map {
        let listed: BTreeSet<&String> = wm.values().collect();
        let present: BTreeSet<String> =
            files.iter().filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string())).collect();
        for f in listed.iter().filter(|f| !present.contains(f.as_str())) {
            ck.issues.push(format!("index lists {f}, which is not in the folder"));
        }
        files.retain(|p| p.file_name().map(|n| listed.iter().any(|f| f.as_str() == n.to_string_lossy())).unwrap_or(false));
    }
    let mut entries: BTreeMap<String, Entry> = BTreeMap::new();
    let mut malformed: BTreeMap<String, String> = BTreeMap::new();
    for path in &files {
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let (hs, data_len, header) = read_header_checked(path)?;
        ck.header_sizes.insert(fname.clone(), hs);
        for (name, v) in header {
            if index_places {
                if let Some(wm) = &weight_map {
                    match wm.get(&name) {
                        Some(owner) if *owner == fname => {}
                        Some(owner) => {
                            ck.issues.push(format!("tensor {name} is also stored in {fname}; using {owner} as the index says"));
                            continue;
                        }
                        None => {
                            ck.issues.push(format!("tensor {name} in {fname} is not in the index: not loaded"));
                            continue;
                        }
                    }
                }
            }
            if malformed.contains_key(&name) {
                continue;
            }
            let e = match entry_of(&fname, &v).and_then(|e| validate_entry(&e, data_len).map(|_| e)) {
                Ok(e) => e,
                Err(why) => {
                    malformed.insert(name, why);
                    continue;
                }
            };
            match entries.remove(&name) {
                Some(prev) => match weight_map.as_ref().and_then(|wm| wm.get(&name)) {
                    Some(owner) if *owner == prev.file || *owner == e.file => {
                        let (keep, drop) = if *owner == prev.file { (prev, e) } else { (e, prev) };
                        ck.issues.push(format!(
                            "tensor {name} is also stored in {}; using {} as the index says",
                            drop.file, keep.file
                        ));
                        entries.insert(name, keep);
                    }
                    _ => {
                        malformed.insert(name, format!("appears in both {} and {}", prev.file, e.file));
                    }
                },
                None => {
                    entries.insert(name, e);
                }
            }
        }
    }
    for (name, why) in &malformed {
        ck.issues.push(format!("tensor {name}: {why}"));
    }
    if index_places {
        if let Some(wm) = &weight_map {
            let scanned: BTreeSet<&String> = ck.header_sizes.keys().collect();
            for (name, shard) in wm {
                if scanned.contains(shard) && !entries.contains_key(name) && !malformed.contains_key(name) {
                    ck.issues.push(format!("tensor {name}: the index names {shard}, whose header does not list it"));
                }
            }
        }
    }
    // a malformed part of an EXL3 group refuses the whole group
    let mut bad_prefix: BTreeMap<String, String> = BTreeMap::new();
    for (name, why) in &malformed {
        if let Some((prefix, part)) = name.rsplit_once('.') {
            let trellis = format!("{prefix}.trellis");
            if PARTS.contains(&part) && (entries.contains_key(&trellis) || malformed.contains_key(&trellis)) {
                bad_prefix.entry(prefix.to_string()).or_insert_with(|| format!("tensor .{part} is malformed: {why}"));
            }
        }
    }
    let mut by_prefix: BTreeMap<String, BTreeMap<String, Entry>> = BTreeMap::new();
    for (name, e) in &entries {
        if let Some((prefix, part)) = name.rsplit_once('.') {
            if PARTS.contains(&part) {
                by_prefix.entry(prefix.to_string()).or_default().insert(part.to_string(), e.clone());
            }
        }
    }
    let mut taken: BTreeSet<String> = BTreeSet::new();
    for (prefix, parts) in &by_prefix {
        if !parts.contains_key("trellis") && !bad_prefix.contains_key(prefix) {
            continue; // e.g. a plain layer's .bias
        }
        for p in parts.keys() {
            taken.insert(format!("{prefix}.{p}"));
        }
        if let Some(why) = bad_prefix.get(prefix) {
            ck.bad.insert(prefix.clone(), why.clone());
            continue;
        }
        match parse_group(prefix, parts) {
            Ok(g) => {
                if read_markers && g.codebook != Codebook::Inst3 {
                    let e = &parts[g.codebook.name()];
                    let want = if g.codebook == Codebook::Mcg { MARKER_MCG } else { MARKER_MUL1 };
                    match read_marker(&root.join(&e.file), ck.header_sizes[&e.file], e) {
                        Ok(v) if v == want => {
                            ck.markers.insert(format!("{prefix}.{}", g.codebook.name()), v);
                        }
                        Ok(v) => {
                            // decoding with the wrong codebook would be silently wrong: refuse
                            ck.bad.insert(
                                prefix.clone(),
                                format!(".{} marker holds {v:#x}, expected {want:#x}", g.codebook.name()),
                            );
                            continue;
                        }
                        Err(why) => {
                            ck.bad.insert(prefix.clone(), why);
                            continue;
                        }
                    }
                }
                for (p, e) in parts {
                    ck.parts.insert(format!("{prefix}.{p}"), e.clone());
                }
                ck.groups.insert(prefix.clone(), g);
            }
            Err(why) => {
                ck.bad.insert(prefix.clone(), why);
            }
        }
    }
    // prefixes whose trellis itself was malformed have no entry in by_prefix
    for (prefix, why) in &bad_prefix {
        ck.bad.entry(prefix.clone()).or_insert_with(|| why.clone());
    }
    for (name, e) in entries {
        if !taken.contains(&name) {
            ck.plain.insert(name, e);
        }
    }
    let cfg = root.join("config.json");
    if cfg.exists() {
        match std::fs::read(&cfg).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string())) {
            Ok(v) => ck.config = v,
            Err(e) => ck.issues.push(format!("config.json: {e}")),
        }
    }
    Ok(ck)
}

impl Checkpoint {
    /// The raw bytes of one tensor.
    pub fn read_tensor(&self, e: &Entry) -> std::io::Result<Vec<u8>> {
        let hs = *self.header_sizes.get(&e.file).ok_or_else(|| invalid(format!("{} was not scanned", e.file)))?;
        read_bytes(&self.root.join(&e.file), hs, e)
    }

    /// A 16-bit tensor as little-endian values, each mapped through `conv` (an odd trailing byte is ignored, as
    /// `chunks_exact(2)` would). Tensors of 4 MiB or more are read by several threads (`REXL3_LOAD_THREADS`).
    pub fn read_le16<T, F>(&self, e: &Entry, conv: F) -> std::io::Result<Vec<T>>
    where
        T: Copy + Default + Send,
        F: Fn(u16) -> T + Sync,
    {
        let hs = *self.header_sizes.get(&e.file).ok_or_else(|| invalid(format!("{} was not scanned", e.file)))?;
        let start = tensor_start(hs, e)?;
        let count = usize::try_from(e.nbytes() / 2).map_err(|_| invalid("tensor too large for this platform"))?;
        let threads = if e.nbytes() >= PARALLEL_READ_MIN_BYTES { read_threads() } else { 1 };
        let _t = loadstats::timer(Stage::Read, e.nbytes());
        read_le16_at(&File::open(self.root.join(&e.file))?, start, count, threads, conv)
    }

    /// A 16-bit tensor as raw little-endian bits.
    pub fn read_u16(&self, e: &Entry) -> std::io::Result<Vec<u16>> {
        self.read_le16(e, |v| v)
    }

    fn read_i16(&self, e: &Entry) -> std::io::Result<Vec<i16>> {
        self.read_le16(e, |v| v as i16)
    }

    /// A float tensor (F16, BF16 or F32) widened to `f64`.
    pub fn read_floats(&self, e: &Entry) -> std::io::Result<Vec<f64>> {
        let b = self.read_tensor(e)?;
        let _t = loadstats::timer(Stage::Convert, b.len() as u64);
        match e.dtype.as_str() {
            "F16" => Ok(b.chunks_exact(2).map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect()),
            "BF16" => {
                Ok(b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64).collect())
            }
            "F32" => Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()),
            d => Err(invalid(format!("cannot read {d} as floats"))),
        }
    }

    /// Read one group's trellis, scales and bias from disk, ready for `forward` and `unpack`.
    pub fn load_group(&self, prefix: &str) -> std::io::Result<LoadedGroup> {
        let g = self.groups.get(prefix).ok_or_else(|| invalid(format!("no EXL3 group {prefix}")))?.clone();
        let part = |p: &str| -> std::io::Result<&Entry> {
            let name = format!("{prefix}.{p}");
            self.lookup_part(&name).ok_or_else(|| invalid(format!("{name} is not in the scan")))
        };
        let trellis = self.read_i16(part("trellis")?)?;
        let suh = self.read_scale(part(g.in_scales)?)?;
        let svh = self.read_scale(part(g.out_scales)?)?;
        let bias = if g.bias { Some(self.read_floats(part("bias")?)?) } else { None };
        Ok(LoadedGroup { kt: g.k / 16, nt: g.n / 16, group: g, trellis, suh, svh, bias })
    }

    /// An EXL3 scale vector (`suh`/`svh` as fp16, or `su`/`sv` as packed signs) as `f64` values.
    pub fn read_scale(&self, e: &Entry) -> std::io::Result<Vec<f64>> {
        if e.dtype == "I16" {
            let packed = self.read_i16(e)?;
            let _t = loadstats::timer(Stage::Convert, packed.len() as u64 * 2);
            Ok(unpack_signs(&packed))
        } else {
            self.read_floats(e)
        }
    }

    /// The tensors of one EXL3 group, by role, so a loader can copy them without going through `load_group`.
    pub fn group_parts(&self, prefix: &str) -> std::io::Result<GroupParts<'_>> {
        let g = self.groups.get(prefix).ok_or_else(|| invalid(format!("no EXL3 group {prefix}")))?;
        let part = |p: &str| -> std::io::Result<&Entry> {
            let name = format!("{prefix}.{p}");
            self.lookup_part(&name).ok_or_else(|| invalid(format!("{name} is not in the scan")))
        };
        Ok(GroupParts {
            group: g,
            trellis: part("trellis")?,
            in_scales: part(g.in_scales)?,
            out_scales: part(g.out_scales)?,
            bias: if g.bias { Some(part("bias")?) } else { None },
        })
    }

    /// The file a tensor lives in and the absolute byte offset of its data in that file.
    pub fn tensor_location(&self, e: &Entry) -> std::io::Result<(PathBuf, u64)> {
        let hs = *self.header_sizes.get(&e.file).ok_or_else(|| invalid(format!("{} was not scanned", e.file)))?;
        Ok((self.root.join(&e.file), tensor_start(hs, e)?))
    }

    fn lookup_part(&self, name: &str) -> Option<&Entry> {
        self.parts.get(name)
    }

    /// One tensor of an accepted EXL3 group by its full name (`<prefix>.trellis`, `.suh`, `.svh`, `.mul1`, ...).
    pub fn part(&self, name: &str) -> Option<&Entry> {
        self.lookup_part(name)
    }

    /// Byte offset of a tensor's data in its shard file (the 8-byte length prefix, the header, then the tensor's own offset):
    /// where a loader reads it. `e` must come from this scan.
    pub fn file_offset(&self, e: &Entry) -> std::io::Result<u64> {
        let hs = *self.header_sizes.get(&e.file).ok_or_else(|| invalid(format!("{} was not scanned", e.file)))?;
        tensor_start(hs, e)
    }

    /// The shard files this scan read, in name order.
    pub fn files(&self) -> impl Iterator<Item = &str> {
        self.header_sizes.keys().map(String::as_str)
    }
}

/// The EXL3 block of config.json (top level or text_config; `quantization_config` or `quantization`).
pub fn config_fields(config: &Value) -> Option<&serde_json::Map<String, Value>> {
    let mut blocks: Vec<&Value> = Vec::new();
    for k in ["quantization_config", "quantization"] {
        if let Some(b) = config.get(k) {
            blocks.push(b);
        }
    }
    if let Some(t) = config.get("text_config") {
        for k in ["quantization_config", "quantization"] {
            if let Some(b) = t.get(k) {
                blocks.push(b);
            }
        }
    }
    blocks
        .into_iter()
        .filter_map(|b| b.as_object())
        .find(|b| b.get("quant_method").and_then(|m| m.as_str()).map(|m| m.eq_ignore_ascii_case("exl3")).unwrap_or(false))
}

#[cfg(all(test, unix))]
mod read_le16_tests {
    use super::*;
    use std::io::Write;

    /// A scratch file of `len` pseudo-random bytes, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str, len: usize) -> (Self, Vec<u8>) {
            let path = std::env::temp_dir().join(format!("gb-exl3-read-le16-{}-{tag}", std::process::id()));
            let mut x = 0x9e37_79b9u32;
            let bytes: Vec<u8> = (0..len)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    (x >> 11) as u8
                })
                .collect();
            File::create(&path).unwrap().write_all(&bytes).unwrap();
            (Self(path), bytes)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn reference(bytes: &[u8], start: usize, count: usize) -> Vec<u16> {
        bytes[start..start + count * 2].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }

    #[test]
    fn matches_the_serial_widening_for_every_split() {
        // 2_000_001 values are 4 MB: with 1-3 threads each thread needs more than one 1 MiB piece
        let (file, bytes) = Scratch::new("split", 5 + 2 * 2_000_001 + 7);
        let f = File::open(&file.0).unwrap();
        for start in [0usize, 5] {
            for count in [0usize, 1, 2, 3, 7, 1000, 2_000_001] {
                let want = reference(&bytes, start, count);
                let want_signed: Vec<i16> = want.iter().map(|&v| v as i16).collect();
                for threads in [1usize, 2, 3, 8, 64] {
                    let got = read_le16_at(&f, start as u64, count, threads, |v| v).unwrap();
                    assert_eq!(got, want, "u16: start {start} count {count} threads {threads}");
                    let signed = read_le16_at(&f, start as u64, count, threads, |v| v as i16).unwrap();
                    assert_eq!(signed, want_signed, "i16: start {start} count {count} threads {threads}");
                }
            }
        }
    }

    #[test]
    fn a_read_past_the_end_is_an_error_not_zeros() {
        let (file, _) = Scratch::new("eof", 4096);
        let f = File::open(&file.0).unwrap();
        for threads in [1usize, 4] {
            // 8192 bytes wanted, 4096 present
            let err = read_le16_at(&f, 0, 4096, threads, |v| v).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::UnexpectedEof, "threads {threads}");
        }
    }
}
