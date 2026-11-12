//! Reader of oracle dumps: `<root>/index.json`, per entry a `meta.json` with scalars and an array table
//! `{name: [dtype, shape, file]}`, arrays as raw little-endian files.

use serde_json::Value;
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

pub struct Arr {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub path: PathBuf,
}

pub struct Entry {
    pub dir: PathBuf,
    pub meta: Value,
    pub arrays: BTreeMap<String, Arr>,
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl Entry {
    pub fn open(dir: &Path) -> io::Result<Entry> {
        let meta: Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?).map_err(|e| bad(e.to_string()))?;
        let mut arrays = BTreeMap::new();
        if let Some(t) = meta.get("arrays").and_then(Value::as_object) {
            for (name, v) in t {
                let dtype = v[0].as_str().unwrap_or("").to_string();
                let shape =
                    v[1].as_array().map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect()).unwrap_or_default();
                let path = dir.join(v[2].as_str().unwrap_or(""));
                arrays.insert(name.clone(), Arr { dtype, shape, path });
            }
        }
        Ok(Entry { dir: dir.to_path_buf(), meta, arrays })
    }

    pub fn has(&self, name: &str) -> bool {
        self.arrays.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> io::Result<&[usize]> {
        Ok(&self.arr(name)?.shape)
    }

    fn arr(&self, name: &str) -> io::Result<&Arr> {
        self.arrays.get(name).ok_or_else(|| bad(format!("{}: no array {name}", self.dir.display())))
    }

    fn bytes(&self, name: &str, dtype: &str, size: usize) -> io::Result<Vec<u8>> {
        let a = self.arr(name)?;
        if a.dtype != dtype {
            return Err(bad(format!("{name}: dtype {} (want {dtype})", a.dtype)));
        }
        let b = std::fs::read(&a.path)?;
        let n: usize = a.shape.iter().product();
        if b.len() != n * size {
            return Err(bad(format!("{name}: {} bytes for {n} values", b.len())));
        }
        Ok(b)
    }

    pub fn f32s(&self, name: &str) -> io::Result<Vec<f32>> {
        Ok(self.bytes(name, "f32", 4)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn u16s(&self, name: &str, dtype: &str) -> io::Result<Vec<u16>> {
        Ok(self.bytes(name, dtype, 2)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }

    pub fn i16s(&self, name: &str) -> io::Result<Vec<i16>> {
        Ok(self.u16s(name, "i16")?.into_iter().map(|v| v as i16).collect())
    }

    pub fn i64s(&self, name: &str) -> io::Result<Vec<i64>> {
        Ok(self
            .bytes(name, "i64", 8)?
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    /// An f16 or f32 array as f32 (exact).
    pub fn floats(&self, name: &str) -> io::Result<Vec<f32>> {
        match self.arr(name)?.dtype.as_str() {
            "f16" => Ok(self.u16s(name, "f16")?.into_iter().map(|h| gb_exl3::f16::f16_to_f64(h) as f32).collect()),
            _ => self.f32s(name),
        }
    }

    pub fn scalar(&self, path: &[&str]) -> Option<&Value> {
        let mut v = &self.meta;
        for p in path {
            v = v.get(*p)?;
        }
        Some(v)
    }
}

pub struct Dump {
    pub root: PathBuf,
    pub index: Value,
}

impl Dump {
    pub fn open(root: &Path) -> io::Result<Dump> {
        let index: Value = serde_json::from_slice(&std::fs::read(root.join("index.json"))?).map_err(|e| bad(e.to_string()))?;
        Ok(Dump { root: root.to_path_buf(), index })
    }

    pub fn tensors(&self) -> Vec<(u64, String, u64)> {
        self.index["tensors"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|t| {
                        (
                            t["id"].as_u64().unwrap_or(0),
                            t["key"].as_str().unwrap_or("").to_string(),
                            t["hessian"].as_u64().unwrap_or(0),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn tensor(&self, id: u64) -> io::Result<Entry> {
        Entry::open(&self.root.join(id.to_string()))
    }

    pub fn hessian(&self, id: u64) -> io::Result<Entry> {
        Entry::open(&self.root.join("hessians").join(id.to_string()))
    }
}
