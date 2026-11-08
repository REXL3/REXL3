//! Reading a model pack's tensors from disk: plain tensors in the forms the kernels want, and EXL3 groups.

use gb_exl3::checkpoint::{scan, Checkpoint, LoadedGroup};
use gb_exl3::f16::f16_to_f64;
use gb_exl3::loadstats::{self, Stage};
use std::path::Path;

pub struct Pack {
    pub ck: Checkpoint,
}

#[derive(Debug)]
pub enum PackError {
    Io(String),
    Missing(String),
    Dtype { name: String, want: &'static str, got: String },
    Shape { name: String, want: Vec<usize>, got: Vec<usize> },
}

impl std::fmt::Display for PackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackError::Io(m) => write!(f, "{m}"),
            PackError::Missing(n) => write!(f, "tensor `{n}` is not in the pack"),
            PackError::Dtype { name, want, got } => write!(f, "tensor `{name}`: expected {want}, found {got}"),
            PackError::Shape { name, want, got } => write!(f, "tensor `{name}`: expected shape {want:?}, found {got:?}"),
        }
    }
}

impl std::error::Error for PackError {}

impl Pack {
    /// Scan a pack. Refuses a pack with refused groups or header issues other than the known benign duplicates.
    pub fn open(dir: &Path) -> Result<Self, PackError> {
        let ck = scan(dir, true).map_err(|e| PackError::Io(e.to_string()))?;
        if !ck.bad.is_empty() {
            return Err(PackError::Io(format!("{} EXL3 group(s) refused, e.g. {:?}", ck.bad.len(), ck.bad.iter().next())));
        }
        Ok(Self { ck })
    }

    /// A plain tensor's entry, checked against the dtype and shape the caller expects (`shape` empty means "any").
    pub fn entry(&self, name: &str, dtype: &'static str, shape: &[usize]) -> Result<&gb_exl3::checkpoint::Entry, PackError> {
        let e = self.ck.plain.get(name).ok_or_else(|| PackError::Missing(name.to_string()))?;
        if e.dtype != dtype {
            return Err(PackError::Dtype { name: name.into(), want: dtype, got: e.dtype.clone() });
        }
        if !shape.is_empty() && e.shape != shape {
            return Err(PackError::Shape { name: name.into(), want: shape.to_vec(), got: e.shape.clone() });
        }
        Ok(e)
    }

    /// A bf16 tensor as raw u16 bits. `shape` empty means "any".
    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>, PackError> {
        let e = self.entry(name, "BF16", shape)?;
        self.ck.read_u16(e).map_err(|err| PackError::Io(err.to_string()))
    }

    /// An fp16 tensor as raw u16 bits.
    pub fn f16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>, PackError> {
        let e = self.entry(name, "F16", shape)?;
        self.ck.read_u16(e).map_err(|err| PackError::Io(err.to_string()))
    }

    /// A bf16 tensor widened to fp32, optionally adding 1 in fp32 (the pack stores RMSNorm gamma minus one for the
    /// target's input/post/q/k/final norms; spec 2.1). The sum is kept in fp32, not rounded back to bf16.
    pub fn f32_from_bf16(&self, name: &str, shape: &[usize], add_one: bool) -> Result<Vec<f32>, PackError> {
        let raw = self.bf16(name, shape)?;
        let _t = loadstats::timer(Stage::Convert, raw.len() as u64 * 2);
        Ok(raw.into_iter().map(|b| f32::from_bits((b as u32) << 16) + if add_one { 1.0 } else { 0.0 }).collect())
    }

    /// A bf16 tensor converted to fp16.
    pub fn f16_from_bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>, PackError> {
        let raw = self.bf16(name, shape)?;
        let _t = loadstats::timer(Stage::Convert, raw.len() as u64 * 2);
        Ok(raw.into_iter().map(|b| gb_exl3::f16::f64_to_f16(f32::from_bits((b as u32) << 16) as f64)).collect())
    }

    /// An fp16 tensor widened to fp32.
    pub fn f32_from_f16(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, PackError> {
        let raw = self.f16(name, shape)?;
        let _t = loadstats::timer(Stage::Convert, raw.len() as u64 * 2);
        Ok(raw.into_iter().map(|b| f16_to_f64(b) as f32).collect())
    }

    /// One EXL3 group (`prefix` without the `.trellis` suffix).
    pub fn group(&self, prefix: &str) -> Result<LoadedGroup, PackError> {
        if !self.ck.groups.contains_key(prefix) {
            return Err(PackError::Missing(format!("{prefix}.trellis")));
        }
        self.ck.load_group(prefix).map_err(|e| PackError::Io(format!("{prefix}: {e}")))
    }

    /// One EXL3 group's metadata (no tensor is read).
    pub fn group_meta(&self, prefix: &str) -> Result<&gb_exl3::checkpoint::Group, PackError> {
        self.ck.groups.get(prefix).ok_or_else(|| PackError::Missing(format!("{prefix}.trellis")))
    }

    /// The names of the shard files the scan found tensors in, sorted.
    pub fn shard_files(&self) -> Vec<String> {
        let mut s: std::collections::BTreeSet<String> = self.ck.plain.values().map(|e| e.file.clone()).collect();
        s.extend(self.ck.groups.values().flat_map(|g| g.files.iter().cloned()));
        s.into_iter().collect()
    }

    pub fn has_group(&self, prefix: &str) -> bool {
        self.ck.groups.contains_key(prefix)
    }
}
