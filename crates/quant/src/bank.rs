//! Encode bank: finished per-tensor encodes, keyed by everything that fixes the result bit for bit, so a later pack or
//! allocation run reuses an encode instead of re-running it.
//!
//! On disk a bank is a directory holding `bank.json` (the index below) and the safetensors files of the stored parts,
//! which the index references by path relative to the directory. One writer per bank at a time.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The index's format name.
pub const FORMAT: &str = "rexl3.encode-bank";
/// The index version this build reads and writes.
pub const VERSION: u32 = 1;
/// The index file name inside a bank directory.
pub const INDEX: &str = "bank.json";

/// Lower-case hex sha256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// The Hessian digest of a key: sha256 of the captured H sum bytes followed by the sample count (u64 LE).
pub fn hessian_digest(h_sum: &[u8], count: u64) -> String {
    let mut h = Sha256::new();
    h.update(h_sum);
    h.update(count.to_le_bytes());
    hex(&h.finalize())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f'))
}

// f64 values are stored as IEEE bits (u64): serde_json without float_roundtrip does not parse every float exactly.

/// What determines an encode bit for bit.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntryKey {
    /// The linear's tensor key, e.g. `model.language_model.layers.0.mlp.experts.3.down_proj`.
    pub tensor: String,
    /// The rate K as IEEE bits (`f64::to_bits`); see [`EntryKey::rate`].
    pub k_bits: u64,
    /// The codebook, e.g. `mul1`.
    pub codebook: String,
    /// The caller's digest of the core options.
    pub options: String,
    /// See [`hessian_digest`].
    pub hessian: String,
    /// sha256 hex of the source weight bytes.
    pub weight: String,
    /// The sign seed.
    pub seed: u64,
    /// The encoder / build identity.
    pub encoder: String,
}

impl EntryKey {
    /// The rate K.
    pub fn rate(&self) -> f64 {
        f64::from_bits(self.k_bits)
    }

    /// Stable sha256 hex of the key. Encoding: the tag `rexl3.encode-bank.key/1` and a NUL, then the fields in
    /// declaration order, each string as its u64-LE byte length and bytes, each u64 as 8 LE bytes. The rate goes in as
    /// its bits, so 0.0 and -0.0 differ.
    pub fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"rexl3.encode-bank.key/1\0");
        let s = |h: &mut Sha256, v: &str| {
            h.update((v.len() as u64).to_le_bytes());
            h.update(v.as_bytes());
        };
        s(&mut h, &self.tensor);
        h.update(self.k_bits.to_le_bytes());
        s(&mut h, &self.codebook);
        s(&mut h, &self.options);
        s(&mut h, &self.hessian);
        s(&mut h, &self.weight);
        h.update(self.seed.to_le_bytes());
        s(&mut h, &self.encoder);
        hex(&h.finalize())
    }
}

/// One stored part: a tensor in a safetensors file of the bank.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartRef {
    /// The file, relative to the bank directory.
    pub file: String,
    /// The tensor name inside the file.
    pub tensor: String,
    /// The safetensors dtype, e.g. `F16`.
    pub dtype: String,
    pub shape: Vec<usize>,
    /// sha256 hex of the tensor's bytes.
    pub sha256: String,
}

/// The stored parts of an encode.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parts {
    pub suh: PartRef,
    pub svh: PartRef,
    pub trellis: PartRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mul1: Option<PartRef>,
}

impl Parts {
    fn iter(&self) -> impl Iterator<Item = &PartRef> {
        [&self.suh, &self.svh, &self.trellis].into_iter().chain(self.mul1.as_ref())
    }
}

/// The encode's metrics; the f64 values as IEEE bits, read with the accessors.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    pub proxy_err_bits: u64,
    pub g_bits: u64,
    pub out_scales: bool,
    pub seconds_bits: u64,
}

impl Metrics {
    pub fn new(proxy_err: f64, g: f64, out_scales: bool, seconds: f64) -> Metrics {
        Metrics { proxy_err_bits: proxy_err.to_bits(), g_bits: g.to_bits(), out_scales, seconds_bits: seconds.to_bits() }
    }
    pub fn proxy_err(&self) -> f64 {
        f64::from_bits(self.proxy_err_bits)
    }
    pub fn g(&self) -> f64 {
        f64::from_bits(self.g_bits)
    }
    pub fn seconds(&self) -> f64 {
        f64::from_bits(self.seconds_bits)
    }
}

/// One finished encode.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub key: EntryKey,
    pub parts: Parts,
    pub metrics: Metrics,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    format: String,
    version: u32,
    created: String,
    entries: Vec<Entry>,
}

// read before the strict parse, so a newer index reports its version rather than a field it added
#[derive(Deserialize)]
struct Header {
    format: String,
    version: u64,
}

/// A bank load or save failure.
#[derive(Debug)]
pub enum Error {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    /// The index names another format.
    Format {
        path: PathBuf,
        found: String,
    },
    /// The index has a version other than [`VERSION`].
    Version {
        path: PathBuf,
        found: u64,
    },
    /// An entry fails validation (a malformed digest, an escaping part path, a duplicate key).
    Invalid {
        path: PathBuf,
        msg: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Error::Parse { path, source } => write!(f, "{}: not a valid encode bank index: {source}", path.display()),
            Error::Format { path, found } => write!(f, "{}: format is {found:?}, expected {FORMAT:?}", path.display()),
            Error::Version { path, found } if *found > VERSION as u64 => write!(
                f,
                "{}: encode bank version {found} is newer than this build reads ({VERSION}); use a newer rexl3-quant",
                path.display()
            ),
            Error::Version { path, found } => {
                write!(f, "{}: encode bank version {found} is not supported (this build reads {VERSION})", path.display())
            }
            Error::Invalid { path, msg } => write!(f, "{}: {msg}", path.display()),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            Error::Parse { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// An open bank: the index in memory, written by [`Bank::save`].
#[derive(Debug)]
pub struct Bank {
    dir: PathBuf,
    created: String,
    entries: Vec<Entry>,
    by_key: HashMap<EntryKey, usize>,
}

impl Bank {
    /// Opens the bank in `dir`, or starts an empty one (creating `dir`; nothing is written until [`Bank::save`]).
    pub fn open_or_create(dir: impl AsRef<Path>) -> Result<Bank> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir).map_err(|source| Error::Io { path: dir.clone(), source })?;
        let path = dir.join(INDEX);
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Ok(Bank { dir, created: now_rfc3339(), entries: Vec::new(), by_key: HashMap::new() });
            }
            Err(source) => return Err(Error::Io { path, source }),
        };
        let header: Header = serde_json::from_str(&text).map_err(|source| Error::Parse { path: path.clone(), source })?;
        if header.format != FORMAT {
            return Err(Error::Format { path, found: header.format });
        }
        if header.version != VERSION as u64 {
            return Err(Error::Version { path, found: header.version });
        }
        let index: Index = serde_json::from_str(&text).map_err(|source| Error::Parse { path: path.clone(), source })?;
        let mut by_key = HashMap::with_capacity(index.entries.len());
        for (i, e) in index.entries.iter().enumerate() {
            validate(e).map_err(|msg| Error::Invalid { path: path.clone(), msg })?;
            if by_key.insert(e.key.clone(), i).is_some() {
                let msg = format!("duplicate entry for {} (key {})", e.key.tensor, e.key.digest());
                return Err(Error::Invalid { path, msg });
            }
        }
        Ok(Bank { dir, created: index.created, entries: index.entries, by_key })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// When the bank was first created (RFC 3339, UTC).
    pub fn created(&self) -> &str {
        &self.created
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn lookup(&self, key: &EntryKey) -> Option<&Entry> {
        self.by_key.get(key).map(|&i| &self.entries[i])
    }

    /// Adds `entry`, replacing (and returning) an entry with the same key.
    pub fn insert(&mut self, entry: Entry) -> Option<Entry> {
        match self.by_key.get(&entry.key) {
            Some(&i) => Some(std::mem::replace(&mut self.entries[i], entry)),
            None => {
                self.by_key.insert(entry.key.clone(), self.entries.len());
                self.entries.push(entry);
                None
            }
        }
    }

    /// The path of a stored part's file.
    pub fn part_path(&self, part: &PartRef) -> PathBuf {
        self.dir.join(&part.file)
    }

    /// Writes the index atomically (a temp file in the bank directory, then a rename). Entries are written sorted by
    /// tensor and key digest, so the index does not depend on insertion order.
    pub fn save(&mut self) -> Result<()> {
        let path = self.dir.join(INDEX);
        for e in &self.entries {
            validate(e).map_err(|msg| Error::Invalid { path: path.clone(), msg })?;
        }
        let mut order: Vec<(String, String, usize)> =
            self.entries.iter().enumerate().map(|(i, e)| (e.key.tensor.clone(), e.key.digest(), i)).collect();
        order.sort();
        let mut slots: Vec<Option<Entry>> = std::mem::take(&mut self.entries).into_iter().map(Some).collect();
        self.entries = order.iter().map(|&(_, _, i)| slots[i].take().expect("each index once")).collect();
        self.by_key = self.entries.iter().enumerate().map(|(i, e)| (e.key.clone(), i)).collect();

        let index =
            Index { format: FORMAT.into(), version: VERSION, created: self.created.clone(), entries: self.entries.clone() };
        let mut text = serde_json::to_string_pretty(&index).map_err(|e| Error::Io { path: path.clone(), source: e.into() })?;
        text.push('\n');
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
        let tmp = self.dir.join(format!(".{INDEX}.tmp.{}.{nanos}", std::process::id()));
        let write = || -> io::Result<()> {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(text.as_bytes())?;
            f.sync_all()?;
            fs::rename(&tmp, &path)?;
            // persist the rename
            fs::File::open(&self.dir)?.sync_all()
        };
        write().map_err(|source| {
            let _ = fs::remove_file(&tmp);
            Error::Io { path: path.clone(), source }
        })
    }
}

fn validate(e: &Entry) -> std::result::Result<(), String> {
    let k = &e.key;
    for (what, v) in [("hessian", &k.hessian), ("weight", &k.weight)] {
        if !is_sha256_hex(v) {
            return Err(format!("{}: {what} digest {v:?} is not lower-case sha256 hex", k.tensor));
        }
    }
    for p in e.parts.iter() {
        if !is_sha256_hex(&p.sha256) {
            return Err(format!("{}: part {} sha256 {:?} is not lower-case sha256 hex", k.tensor, p.tensor, p.sha256));
        }
        let rel = Path::new(&p.file);
        if p.file.is_empty() || !rel.components().all(|c| matches!(c, Component::Normal(_))) {
            return Err(format!("{}: part file {:?} is not a plain path inside the bank", k.tensor, p.file));
        }
    }
    Ok(())
}

// RFC 3339 UTC, seconds resolution
fn now_rfc3339() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // civil date from days since 1970-01-01 (Hinnant's algorithm)
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> TempDir {
            static N: AtomicUsize = AtomicUsize::new(0);
            let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
            let n = N.fetch_add(1, Ordering::Relaxed);
            TempDir(std::env::temp_dir().join(format!("gb-quant-bank-{}-{n}-{nanos}", std::process::id())))
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn key() -> EntryKey {
        EntryKey {
            tensor: "model.language_model.layers.0.mlp.experts.3.down_proj".into(),
            k_bits: 2.5f64.to_bits(),
            codebook: "mul1".into(),
            options: "opts-v1".into(),
            hessian: sha256_hex(b"hessian"),
            weight: sha256_hex(b"weight"),
            seed: 0x5eed,
            encoder: "rexl3-quant test".into(),
        }
    }

    fn part(t: &str, dtype: &str, shape: Vec<usize>) -> PartRef {
        PartRef {
            file: "parts/l0.safetensors".into(),
            tensor: t.into(),
            dtype: dtype.into(),
            shape,
            sha256: sha256_hex(t.as_bytes()),
        }
    }

    fn entry(key: EntryKey, m: Metrics) -> Entry {
        let t = key.tensor.clone();
        Entry {
            key,
            parts: Parts {
                suh: part(&format!("{t}.suh"), "F16", vec![256]),
                svh: part(&format!("{t}.svh"), "F16", vec![512]),
                trellis: part(&format!("{t}.trellis"), "I16", vec![16, 32, 40]),
                mul1: Some(part(&format!("{t}.mul1"), "I32", vec![])),
            },
            metrics: m,
        }
    }

    fn write_index(dir: &Path, v: &serde_json::Value) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join(INDEX), serde_json::to_string(v).unwrap()).unwrap();
    }

    fn saved_index(e: Entry) -> (TempDir, serde_json::Value) {
        let d = TempDir::new();
        let mut b = Bank::open_or_create(&d.0).unwrap();
        b.insert(e);
        b.save().unwrap();
        let v = serde_json::from_str(&fs::read_to_string(d.0.join(INDEX)).unwrap()).unwrap();
        (d, v)
    }

    #[test]
    fn bank_round_trip_is_bit_exact() {
        let d = TempDir::new();
        let subnormal = f64::from_bits(1);
        let mut b = Bank::open_or_create(&d.0).unwrap();
        assert!(b.is_empty() && !d.0.join(INDEX).exists());
        let e1 = entry(key(), Metrics::new(0.1, 1e-300, true, subnormal));
        let mut k2 = key();
        k2.tensor = "model.language_model.layers.0.mlp.experts.2.down_proj".into();
        k2.k_bits = 2.0f64.to_bits();
        let mut e2 = entry(k2, Metrics::new(-0.0, f64::MAX, false, 1.0 / 3.0));
        e2.parts.mul1 = None;
        b.insert(e1.clone());
        b.insert(e2.clone());
        b.save().unwrap();
        let created = b.created().to_string();
        let names: Vec<String> =
            fs::read_dir(&d.0).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        assert_eq!(names, vec![INDEX.to_string()], "no temp file left behind");

        let r = Bank::open_or_create(&d.0).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r.created(), created);
        for e in [&e1, &e2] {
            let got = r.lookup(&e.key).expect("entry present after reopen");
            assert_eq!(got, e);
            assert_eq!(got.metrics.proxy_err().to_bits(), e.metrics.proxy_err().to_bits());
            assert_eq!(got.metrics.g().to_bits(), e.metrics.g().to_bits());
            assert_eq!(got.metrics.seconds().to_bits(), e.metrics.seconds().to_bits());
        }
        let m = &r.lookup(&e1.key).unwrap().metrics;
        assert_eq!((m.proxy_err(), m.g(), m.seconds()), (0.1, 1e-300, subnormal));
        assert_eq!(r.lookup(&e2.key).unwrap().metrics.proxy_err().to_bits(), (-0.0f64).to_bits());
        // sorted by tensor on save
        assert_eq!(r.entries()[0].key.tensor, e2.key.tensor);
        assert_eq!(r.part_path(&e1.parts.suh), d.0.join("parts/l0.safetensors"));
    }

    #[test]
    fn bank_rejects_other_versions() {
        let (d, mut v) = saved_index(entry(key(), Metrics::new(0.5, 1.0, false, 2.0)));
        v["version"] = 2.into();
        v["added_in_v2"] = true.into();
        write_index(&d.0, &v);
        let err = Bank::open_or_create(&d.0).unwrap_err();
        assert!(matches!(err, Error::Version { found: 2, .. }), "{err:?}");
        assert!(err.to_string().contains("newer than this build"), "{err}");
        v["version"] = 0.into();
        write_index(&d.0, &v);
        let err = Bank::open_or_create(&d.0).unwrap_err();
        assert!(matches!(err, Error::Version { found: 0, .. }), "{err:?}");
        assert!(err.to_string().contains("not supported"), "{err}");
    }

    #[test]
    fn bank_rejects_other_format() {
        let (d, mut v) = saved_index(entry(key(), Metrics::new(0.5, 1.0, false, 2.0)));
        v["format"] = "other.bank".into();
        write_index(&d.0, &v);
        let err = Bank::open_or_create(&d.0).unwrap_err();
        assert!(matches!(&err, Error::Format { found, .. } if found == "other.bank"), "{err:?}");
    }

    #[test]
    fn bank_rejects_unknown_fields() {
        let (d, v) = saved_index(entry(key(), Metrics::new(0.5, 1.0, false, 2.0)));
        let mut top = v.clone();
        top["extra"] = 1.into();
        write_index(&d.0, &top);
        assert!(matches!(Bank::open_or_create(&d.0), Err(Error::Parse { .. })));
        let mut nested = v;
        nested["entries"][0]["metrics"]["proxy_err"] = 0.5.into();
        write_index(&d.0, &nested);
        assert!(matches!(Bank::open_or_create(&d.0), Err(Error::Parse { .. })));
    }

    #[test]
    fn bank_rejects_invalid_entries() {
        let (d, v) = saved_index(entry(key(), Metrics::new(0.5, 1.0, false, 2.0)));
        let mut dup = v.clone();
        let e0 = dup["entries"][0].clone();
        dup["entries"].as_array_mut().unwrap().push(e0);
        write_index(&d.0, &dup);
        assert!(matches!(Bank::open_or_create(&d.0), Err(Error::Invalid { .. })));
        for file in ["../x.safetensors", "/abs.safetensors", ""] {
            let mut bad = v.clone();
            bad["entries"][0]["parts"]["suh"]["file"] = file.into();
            write_index(&d.0, &bad);
            assert!(matches!(Bank::open_or_create(&d.0), Err(Error::Invalid { .. })), "{file:?}");
        }
        let mut bad = v;
        bad["entries"][0]["key"]["weight"] = "ABC".into();
        write_index(&d.0, &bad);
        assert!(matches!(Bank::open_or_create(&d.0), Err(Error::Invalid { .. })));
    }

    #[test]
    fn bank_insert_replaces_same_key() {
        let d = TempDir::new();
        let mut b = Bank::open_or_create(&d.0).unwrap();
        let first = entry(key(), Metrics::new(0.5, 1.0, false, 2.0));
        let second = entry(key(), Metrics::new(0.25, 1.5, true, 3.0));
        assert!(b.insert(first.clone()).is_none());
        assert_eq!(b.insert(second.clone()), Some(first));
        assert_eq!(b.len(), 1);
        assert_eq!(b.lookup(&key()), Some(&second));
        b.save().unwrap();
        let r = Bank::open_or_create(&d.0).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r.lookup(&key()), Some(&second));
    }

    #[test]
    fn bank_key_digest_is_stable() {
        // computed independently from the encoding documented on `digest`
        assert_eq!(key().digest(), "8364b36ac1d124459a83b3606a4b060ca394d5a1e4311b6ab737400d7e04458a");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let mut hb = b"h".to_vec();
        hb.extend_from_slice(&7u64.to_le_bytes());
        assert_eq!(hessian_digest(b"h", 7), sha256_hex(&hb));
        let base = key().digest();
        let changes: [fn(&mut EntryKey); 9] = [
            |k| k.tensor.push('x'),
            |k| k.k_bits = 2.0f64.to_bits(),
            |k| k.k_bits = (-2.5f64).to_bits(),
            |k| k.codebook = "3inst".into(),
            |k| k.options.push('x'),
            |k| k.hessian = sha256_hex(b"other"),
            |k| k.weight = sha256_hex(b"other"),
            |k| k.seed += 1,
            |k| k.encoder.push('x'),
        ];
        for (i, change) in changes.iter().enumerate() {
            let mut k = key();
            change(&mut k);
            assert_ne!(k.digest(), base, "change {i}");
        }
        // field boundaries are length-prefixed
        let (mut a, mut b) = (key(), key());
        a.codebook = "mul1x".into();
        b.options = format!("x{}", b.options);
        assert_ne!(a.digest(), b.digest());
        let (z, nz) = (EntryKey { k_bits: 0.0f64.to_bits(), ..key() }, EntryKey { k_bits: (-0.0f64).to_bits(), ..key() });
        assert_ne!(z.digest(), nz.digest());
    }
}
