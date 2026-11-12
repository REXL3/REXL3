//! The pack writer: tensors to safetensors (the stored parts of a quantized linear: suh, svh, trellis, the codebook
//! marker).

use serde_json::{json, Map, Value};
use std::io::{self, Write};
use std::path::Path;

/// One tensor to write: name, safetensors dtype, shape, little-endian bytes.
pub struct Part {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl Part {
    pub fn f16(name: String, shape: Vec<usize>, v: &[u16]) -> Part {
        Part { name, dtype: "F16", shape, bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }
    pub fn i16(name: String, shape: Vec<usize>, v: &[i16]) -> Part {
        Part { name, dtype: "I16", shape, bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }
    pub fn i32_scalar(name: String, v: i32) -> Part {
        Part { name, dtype: "I32", shape: vec![], bytes: v.to_le_bytes().to_vec() }
    }
}

/// Writes `parts` as one safetensors file (offsets in the given order, header padded to 8 bytes).
pub fn write_safetensors(path: &Path, parts: &[Part], metadata: Option<Map<String, Value>>) -> io::Result<()> {
    let mut header = Map::new();
    if let Some(m) = metadata {
        header.insert("__metadata__".into(), Value::Object(m));
    }
    let mut off = 0usize;
    for p in parts {
        header.insert(p.name.clone(), json!({"dtype": p.dtype, "shape": p.shape, "data_offsets": [off, off + p.bytes.len()]}));
        off += p.bytes.len();
    }
    let mut h = serde_json::to_vec(&Value::Object(header)).map_err(io::Error::other)?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&(h.len() as u64).to_le_bytes())?;
    f.write_all(&h)?;
    for p in parts {
        f.write_all(&p.bytes)?;
    }
    f.flush()
}
