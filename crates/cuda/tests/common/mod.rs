//! Helpers shared by the kernel integration tests: a tiny .npy reader for the reference dumps in `testdata/ref`
//! (written by tools/dump_reference.py), error metrics, and a seeded generator.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// The repository root: `REXL3_REPO_ROOT` when set (a prebuilt test binary run from an installed bin directory that carries the
/// tree's `testdata/`, e.g. on a node without this checkout), else the checkout this binary was built from.
pub fn repo_root() -> PathBuf {
    std::env::var_os("REXL3_REPO_ROOT").map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

/// Where the ExLlamaV3 reference dumps live: `REXL3_REF_DIR`, else this checkout's `testdata/ref` (an agent worktree,
/// which does not carry the git-ignored dumps, sets `REXL3_REF_DIR` to the main checkout's).
pub fn ref_dir() -> PathBuf {
    if let Ok(d) = std::env::var("REXL3_REF_DIR") {
        return PathBuf::from(d);
    }
    repo_root().join("testdata/ref")
}

/// True when the reference dumps exist (they are git-ignored; regenerate with tools/dump_reference.py).
pub fn have_refs() -> bool {
    ref_dir().join("manifest.json").exists()
}

pub struct Npy {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data: Vec<u8>,
}

impl Npy {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    /// Any float dtype as f32 (f2, f4, f8; bf16 is not a numpy dtype and never appears in the dumps).
    pub fn to_f32(&self) -> Vec<f32> {
        match self.dtype.as_str() {
            "<f4" => self.data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "<f8" => self.data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            "<f2" => {
                self.data.chunks_exact(2).map(|c| gb_exl3::f16::f16_to_f64(u16::from_le_bytes([c[0], c[1]])) as f32).collect()
            }
            d => panic!("to_f32 on dtype {d}"),
        }
    }
    pub fn to_i64(&self) -> Vec<i64> {
        match self.dtype.as_str() {
            "<i8" => self.data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
            "<i4" => self.data.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64).collect(),
            d => panic!("to_i64 on dtype {d}"),
        }
    }
}

pub fn read_npy(path: &Path) -> Npy {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&b[..6], b"\x93NUMPY", "{} is not a .npy file", path.display());
    let (hlen, off) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        _ => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
    };
    let header = std::str::from_utf8(&b[off..off + hlen]).unwrap();
    let field = |key: &str| -> &str {
        let i = header.find(key).unwrap_or_else(|| panic!("no {key} in {header}")) + key.len();
        &header[i..]
    };
    let descr = field("'descr':").trim_start().trim_start_matches('\'');
    let dtype = descr[..descr.find('\'').unwrap()].to_string();
    assert!(field("'fortran_order':").trim_start().starts_with("False"), "fortran-order arrays are not supported");
    let sh = field("'shape':").trim_start();
    let sh = &sh[1..sh.find(')').unwrap()];
    let shape: Vec<usize> = sh.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse().unwrap()).collect();
    Npy { shape, dtype, data: b[off + hlen..].to_vec() }
}

pub fn ref_npy(rel: &str) -> Npy {
    read_npy(&ref_dir().join(rel))
}

/// (cosine similarity, relative RMS error) of `got` against `want`.
pub fn cos_rel_rms(got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let (mut dot, mut gg, mut ww, mut ee) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (g as f64, w as f64);
        dot += g * w;
        gg += g * g;
        ww += w * w;
        ee += (g - w) * (g - w);
    }
    (dot / (gg.sqrt() * ww.sqrt()).max(1e-30), (ee / ww.max(1e-30)).sqrt())
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in [lo, hi).
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / 16777216.0)
    }
    pub fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }
}

pub mod exl3_synth;
