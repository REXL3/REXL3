//! Source model reading: safetensors headers, whole tensors and runs of rows along the first dimension (an expert of
//! a stacked tensor), as stored or as fp32.

use gb_exl3::checkpoint::read_header_checked;
use serde_json::Value;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// A tensor read as fp32, with its shape.
pub struct Loaded {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// The `*.safetensors` file of `dir` (or `dir` itself when it is a file) that holds `name`.
pub fn find(dir: &Path, name: &str) -> io::Result<PathBuf> {
    let files: Vec<PathBuf> = if dir.is_file() {
        vec![dir.to_path_buf()]
    } else {
        let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        v.sort();
        v
    };
    for f in files {
        let (_, _, map) = read_header_checked(&f)?;
        if map.contains_key(name) {
            return Ok(f);
        }
    }
    Err(bad(format!("{}: no tensor {name}", dir.display())))
}

/// Reads `name` from a safetensors file as fp32.
pub fn read(path: &Path, name: &str) -> io::Result<Loaded> {
    let (hlen, data_len, map) = read_header_checked(path)?;
    let v = map.get(name).ok_or_else(|| bad(format!("{}: no tensor {name}", path.display())))?;
    let dtype = v.get("dtype").and_then(Value::as_str).unwrap_or("");
    let shape: Vec<usize> = v
        .get("shape")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
        .unwrap_or_default();
    let off = v.get("data_offsets").and_then(Value::as_array).ok_or_else(|| bad(format!("{name}: no data_offsets")))?;
    let (a, b) = (off[0].as_u64().unwrap_or(0), off[1].as_u64().unwrap_or(0));
    let size = match dtype {
        "F32" => 4,
        "F16" | "BF16" => 2,
        _ => return Err(bad(format!("{name}: dtype {dtype} is not read (F32, F16, BF16)"))),
    };
    let n: usize = shape.iter().product();
    if b < a || b > data_len || (b - a) as usize != n * size {
        return Err(bad(format!("{name}: offsets {a}..{b} do not hold {n} x {size} bytes")));
    }
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(8 + hlen + a))?;
    let mut raw = vec![0u8; (b - a) as usize];
    f.read_exact(&mut raw)?;
    let data = match dtype {
        "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "F16" => raw.chunks_exact(2).map(|c| gb_exl3::f16::f16_to_f64(u16::from_le_bytes([c[0], c[1]])) as f32).collect(),
        _ => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
    };
    Ok(Loaded { shape, data })
}

/// A tensor's raw little-endian bytes as stored (dtype name, shape, bytes).
pub struct Raw {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl Raw {
    pub fn elem_bytes(&self) -> usize {
        dtype_bytes(&self.dtype).unwrap_or(0)
    }

    /// The values as fp32 (F32, F16, BF16).
    pub fn to_f32(&self) -> io::Result<Vec<f32>> {
        let r = &self.bytes;
        Ok(match self.dtype.as_str() {
            "F32" => r.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "F16" => r.chunks_exact(2).map(|c| gb_exl3::f16::f16_to_f64(u16::from_le_bytes([c[0], c[1]])) as f32).collect(),
            "BF16" => r.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            d => return Err(bad(format!("dtype {d} is not read as fp32 (F32, F16, BF16)"))),
        })
    }
}

pub fn dtype_bytes(d: &str) -> Option<usize> {
    Some(match d {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "F8_E4M3" | "F8_E5M2" | "I8" | "U8" | "BOOL" => 1,
        _ => return None,
    })
}

struct Entry {
    file: PathBuf,
    dtype: String,
    shape: Vec<usize>,
    /// absolute byte offset of the data in the file
    start: u64,
    len: u64,
}

/// A checkpoint directory (sharded with `model.safetensors.index.json`, or any set of `*.safetensors` files): every
/// tensor's location, read on demand (whole tensors or runs of rows along the first dimension).
pub struct Source {
    pub dir: PathBuf,
    map: std::collections::BTreeMap<String, Entry>,
}

impl Source {
    pub fn open(dir: &Path) -> io::Result<Source> {
        let mut files: Vec<PathBuf> = if dir.is_file() {
            vec![dir.to_path_buf()]
        } else {
            std::fs::read_dir(dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect()
        };
        files.sort();
        let mut map = std::collections::BTreeMap::new();
        for f in files {
            let (hlen, data_len, hdr) = read_header_checked(&f)?;
            for (name, v) in hdr {
                let dtype = v.get("dtype").and_then(Value::as_str).unwrap_or("").to_string();
                let shape: Vec<usize> = v
                    .get("shape")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
                    .unwrap_or_default();
                let off =
                    v.get("data_offsets").and_then(Value::as_array).ok_or_else(|| bad(format!("{name}: no data_offsets")))?;
                let (a, b) = (off[0].as_u64().unwrap_or(0), off[1].as_u64().unwrap_or(0));
                let n: usize = shape.iter().product();
                let eb = dtype_bytes(&dtype).ok_or_else(|| bad(format!("{name}: unknown dtype {dtype}")))?;
                if b < a || b > data_len || (b - a) as usize != n * eb {
                    return Err(bad(format!("{name}: offsets {a}..{b} do not hold {n} x {eb} bytes")));
                }
                if map.insert(name.clone(), Entry { file: f.clone(), dtype, shape, start: 8 + hlen + a, len: b - a }).is_some() {
                    return Err(bad(format!("{name}: in more than one file")));
                }
            }
        }
        Ok(Source { dir: dir.to_path_buf(), map })
    }

    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.map.get(name).map(|e| e.shape.as_slice())
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    fn entry(&self, name: &str) -> io::Result<&Entry> {
        self.map.get(name).ok_or_else(|| bad(format!("{}: no tensor {name}", self.dir.display())))
    }

    /// The whole tensor as stored.
    pub fn raw(&self, name: &str) -> io::Result<Raw> {
        let e = self.entry(name)?;
        let rows = e.shape.first().copied().unwrap_or(1);
        if e.shape.is_empty() {
            return self.read_bytes(e, 0, e.len as usize, vec![]);
        }
        self.raw_rows(name, 0, rows)
    }

    /// Rows [r0, r0 + rows) along the first dimension (a contiguous byte run), e.g. expert e of a stacked
    /// (E, out, in) tensor.
    pub fn raw_rows(&self, name: &str, r0: usize, rows: usize) -> io::Result<Raw> {
        let e = self.entry(name)?;
        let first = *e.shape.first().ok_or_else(|| bad(format!("{name}: a scalar has no rows")))?;
        if r0 + rows > first {
            return Err(bad(format!("{name}: rows {r0}..{} of {first}", r0 + rows)));
        }
        let row_bytes = e.len as usize / first.max(1);
        let mut shape = e.shape.clone();
        shape[0] = rows;
        self.read_bytes(e, r0 * row_bytes, rows * row_bytes, shape)
    }

    fn read_bytes(&self, e: &Entry, off: usize, len: usize, shape: Vec<usize>) -> io::Result<Raw> {
        let mut f = File::open(&e.file)?;
        f.seek(SeekFrom::Start(e.start + off as u64))?;
        let mut bytes = vec![0u8; len];
        f.read_exact(&mut bytes)?;
        Ok(Raw { dtype: e.dtype.clone(), shape, bytes })
    }

    /// The whole tensor as fp32.
    pub fn f32(&self, name: &str) -> io::Result<Vec<f32>> {
        self.raw(name)?.to_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gbq-source-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn row_reads_match_whole_reads() {
        use crate::pack::{write_safetensors, Part};
        let d = tmp("rows");
        let (e, o, i) = (5usize, 3usize, 4usize);
        let vals: Vec<f32> = (0..e * o * i).map(|x| x as f32 * 0.25 - 3.0).collect();
        let bf: Vec<u8> = vals.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
        let parts = [
            Part { name: "a.gate_up".into(), dtype: "BF16", shape: vec![e, o, i], bytes: bf },
            Part::f16("b.norm".into(), vec![4], &[0x3c00, 0x4000, 0xc000, 0x0000]),
        ];
        write_safetensors(&d.join("m-1.safetensors"), &parts[..1], None).unwrap();
        write_safetensors(&d.join("m-2.safetensors"), &parts[1..], None).unwrap();
        let s = Source::open(&d).unwrap();
        let whole = s.f32("a.gate_up").unwrap();
        assert_eq!(whole, vals);
        for (r0, n) in [(0, 1), (2, 2), (4, 1), (0, 5)] {
            let r = s.raw_rows("a.gate_up", r0, n).unwrap();
            assert_eq!(r.shape, vec![n, o, i]);
            assert_eq!(r.to_f32().unwrap(), vals[r0 * o * i..(r0 + n) * o * i].to_vec());
        }
        assert!(s.raw_rows("a.gate_up", 4, 2).is_err());
        assert_eq!(s.f32("b.norm").unwrap(), vec![1.0, 2.0, -2.0, 0.0]);
        assert_eq!(s.shape("b.norm"), Some(&[4usize][..]));
        assert!(s.raw("nope").is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
