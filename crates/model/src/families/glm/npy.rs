//! A minimal NumPy `.npy` (format 1.0) reader and writer for the oracle dumps (`tools/dump_glm_reference.py`): little-endian arrays of f32, f16
//! (as raw bits), i16, i32, i64, C order. Enough to read the oracle's `ids`, `route_*` and layer files and to write a candidate dump that
//! `tools/compare_glm_dump.py` reads.

use std::io::Write;
use std::path::Path;

/// An array read from a `.npy` file: its shape, its dtype string (`<f4`, `<i8`, ...) and its bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct Npy {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data: Vec<u8>,
}

impl Npy {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn to_f32(&self) -> Result<Vec<f32>, String> {
        match self.dtype.as_str() {
            "<f4" => Ok(self.data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
            "<f2" => {
                Ok(self.data.chunks_exact(2).map(|c| gb_exl3::f16::f16_to_f64(u16::from_le_bytes([c[0], c[1]])) as f32).collect())
            }
            d => Err(format!("to_f32 on dtype {d}")),
        }
    }
    pub fn to_i32(&self) -> Result<Vec<i32>, String> {
        match self.dtype.as_str() {
            "<i2" => Ok(self.data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as i32).collect()),
            "<i4" => Ok(self.data.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
            "<i8" => Ok(self.data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as i32).collect()),
            d => Err(format!("to_i32 on dtype {d}")),
        }
    }
    pub fn to_u16(&self) -> Result<Vec<u16>, String> {
        match self.dtype.as_str() {
            "<f2" | "<u2" | "<i2" => Ok(self.data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()),
            d => Err(format!("to_u16 on dtype {d}")),
        }
    }
}

/// Read a `.npy` file (version 1.0 or 2.0, C order).
pub fn read(path: &Path) -> Result<Npy, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&b).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parse the bytes of a `.npy` file.
pub fn parse(b: &[u8]) -> Result<Npy, String> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let (hlen, off) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        _ => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
    };
    let header = std::str::from_utf8(b.get(off..off + hlen).ok_or("truncated header")?).map_err(|e| e.to_string())?;
    let field = |key: &str| -> Result<&str, String> {
        let i = header.find(key).ok_or_else(|| format!("no {key} in {header}"))? + key.len();
        Ok(&header[i..])
    };
    let descr = field("'descr':")?.trim_start().trim_start_matches('\'');
    let dtype = descr[..descr.find('\'').ok_or("bad descr")?].to_string();
    if !field("'fortran_order':")?.trim_start().starts_with("False") {
        return Err("fortran-order arrays are not supported".into());
    }
    let sh = field("'shape':")?.trim_start();
    let sh = &sh[1..sh.find(')').ok_or("bad shape")?];
    let shape: Vec<usize> = sh
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().map_err(|_| format!("bad shape {sh}")))
        .collect::<Result<_, _>>()?;
    let data = b[off + hlen..].to_vec();
    let size = match dtype.as_str() {
        "<f4" | "<i4" | "<u4" => 4,
        "<f2" | "<i2" | "<u2" => 2,
        "<i8" | "<f8" | "<u8" => 8,
        "|u1" | "|i1" | "|b1" => 1,
        d => return Err(format!("dtype {d} is not supported")),
    };
    if data.len() != shape.iter().product::<usize>() * size {
        return Err(format!("{} data bytes for shape {shape:?} of {size}-byte elements", data.len()));
    }
    Ok(Npy { shape, dtype, data })
}

/// The bytes of a `.npy` file (version 1.0) holding `data` (little-endian, C order) of `dtype` and `shape`.
pub fn encode(dtype: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let shape_s = match shape {
        [] => "()".to_string(),
        [n] => format!("({n},)"),
        s => format!("({})", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape_s}, }}");
    // pad so that magic (6) + version (2) + length (2) + header + newline is a multiple of 64
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

/// Write an array to `path`, creating the directory.
pub fn write(path: &Path, dtype: &str, shape: &[usize], data: &[u8]) -> Result<(), String> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let mut f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    f.write_all(&encode(dtype, shape, data)).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn write_f32(path: &Path, shape: &[usize], v: &[f32]) -> Result<(), String> {
    write(path, "<f4", shape, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}
pub fn write_i16(path: &Path, shape: &[usize], v: &[i16]) -> Result<(), String> {
    write(path, "<i2", shape, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}
pub fn write_i32(path: &Path, shape: &[usize], v: &[i32]) -> Result<(), String> {
    write(path, "<i4", shape, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}
pub fn write_i64(path: &Path, shape: &[usize], v: &[i64]) -> Result<(), String> {
    write(path, "<i8", shape, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_dtype_and_shape() {
        let v = vec![1.5f32, -2.0, 3.25, 0.0, 7.0, -0.5];
        let b = encode("<f4", &[2, 3], &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        assert_eq!((b.len() - 24) % 64, 0, "the header (magic, version, length and text) is padded to 64 bytes like numpy's");
        let n = parse(&b).unwrap();
        assert_eq!((n.shape.clone(), n.dtype.as_str()), (vec![2, 3], "<f4"));
        assert_eq!(n.to_f32().unwrap(), v);
        let b = encode("<i8", &[3], &[1i64, -2, 3].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        assert_eq!(parse(&b).unwrap().to_i32().unwrap(), vec![1, -2, 3]);
        let b = encode("<i2", &[2, 1, 2], &[5i16, -6, 7, 8].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let n = parse(&b).unwrap();
        assert_eq!(n.shape, vec![2, 1, 2]);
        assert_eq!(n.to_i32().unwrap(), vec![5, -6, 7, 8]);
        // a scalar-shaped array
        assert_eq!(parse(&encode("<f4", &[], &1.0f32.to_le_bytes())).unwrap().shape, Vec::<usize>::new());
    }

    #[test]
    fn what_numpy_writes_parses() {
        // the header numpy 1.26 writes for np.arange(6, dtype=np.int32).reshape(2, 3)
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        let h = "{'descr': '<i4', 'fortran_order': False, 'shape': (2, 3), }";
        let pad = 64 - (10 + h.len() + 1) % 64;
        let hdr = format!("{h}{}\n", " ".repeat(pad));
        b.extend_from_slice(&(hdr.len() as u16).to_le_bytes());
        b.extend_from_slice(hdr.as_bytes());
        b.extend((0..6i32).flat_map(|x| x.to_le_bytes()));
        let n = parse(&b).unwrap();
        assert_eq!((n.shape.clone(), n.to_i32().unwrap()), (vec![2, 3], (0..6).collect::<Vec<i32>>()));
    }

    #[test]
    fn a_truncated_or_foreign_file_is_an_error() {
        assert!(parse(b"hello world, not numpy").is_err());
        let mut b = encode("<f4", &[4], &[0u8; 16]);
        b.truncate(b.len() - 3);
        assert!(parse(&b).is_err());
        assert!(parse(&encode("<c8", &[1], &[0u8; 8])).is_err(), "unsupported dtype");
    }
}
