//! What a Bailing pack holds, as plain data: every plain tensor and every EXL3 group with its dtype, shape and where its bytes sit
//! in which shard. The mapper (`weights::BailingWeights::map`) and the load plan work on this and nowhere else, so they are pure
//! functions: no file is opened, no CUDA context exists, and a test can hand them an inventory it has edited.
//!
//! An inventory comes from [`PackInventory::from_checkpoint`] (a scanned pack: what serving uses) or from
//! [`PackInventory::from_headers`] (safetensors headers as a JSON map: the synthetic and stub-pack tests).

use gb_exl3::checkpoint::{dtype_size, parse_group, Checkpoint, Entry};
use gb_exl3::{Bits, Codebook};
use std::collections::{BTreeMap, BTreeSet};

/// Where one tensor's bytes are, and what they are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    /// Index into [`PackInventory::files`].
    pub file: u16,
    /// safetensors dtype name (`F16`, `BF16`, `F32`, `I16`, `I32`, ...).
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute byte offset of the data in its file.
    pub offset: u64,
    pub bytes: u64,
}

/// One EXL3 linear layer: `y = ((x * suh) H_K . W_q . H_N) * svh (+ bias)`.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupInfo {
    pub k: usize,
    pub n: usize,
    /// Read from the trellis' last dimension, never from `quantization_config.json`.
    pub bits: Bits,
    pub codebook: Codebook,
    /// `suh` (fp16 scales) or `su` (packed signs); likewise `svh` / `sv`.
    pub in_scales: &'static str,
    pub out_scales: &'static str,
    pub trellis: TensorInfo,
    pub in_scale: TensorInfo,
    pub out_scale: TensorInfo,
    pub bias: Option<TensorInfo>,
    /// The codebook marker (`mul1` / `mcg`), a single int32 the scan has already checked; none for `3inst`.
    pub marker: Option<TensorInfo>,
}

/// Every tensor of a pack.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackInventory {
    /// Shard file names; a [`TensorInfo::file`] indexes this.
    pub files: Vec<String>,
    /// Tensors that belong to no EXL3 group.
    pub plain: BTreeMap<String, TensorInfo>,
    /// EXL3 groups by prefix (the tensor name without `.trellis`).
    pub groups: BTreeMap<String, GroupInfo>,
    /// Groups the scan refused, with its reason. A pack with any is not served.
    pub refused: BTreeMap<String, String>,
    /// Everything else the scan noticed (see `Checkpoint::issues`).
    pub issues: Vec<String>,
}

impl PackInventory {
    /// Tensors in the pack: plain ones plus the parts of every group.
    pub fn tensor_count(&self) -> usize {
        self.plain.len()
            + self.groups.values().map(|g| 3 + g.bias.is_some() as usize + g.marker.is_some() as usize).sum::<usize>()
    }

    /// The inventory of a scanned pack.
    pub fn from_checkpoint(ck: &Checkpoint) -> Result<Self, String> {
        let files: Vec<String> = ck.files().map(str::to_string).collect();
        let file_index = |name: &str| -> Result<u16, String> {
            let i = files.iter().position(|f| f == name).ok_or_else(|| format!("{name} was not scanned"))?;
            u16::try_from(i).map_err(|_| "more than 65,535 shards".to_string())
        };
        let info = |e: &Entry| -> Result<TensorInfo, String> {
            let offset = ck.file_offset(e).map_err(|err| err.to_string())?;
            Ok(TensorInfo {
                file: file_index(&e.file)?,
                dtype: e.dtype.clone(),
                shape: e.shape.clone(),
                offset,
                bytes: e.nbytes(),
            })
        };
        let mut inv =
            PackInventory { files: files.clone(), refused: ck.bad.clone(), issues: ck.issues.clone(), ..Default::default() };
        for (name, e) in &ck.plain {
            inv.plain.insert(name.clone(), info(e)?);
        }
        for (prefix, g) in &ck.groups {
            let part = |p: &str| -> Result<TensorInfo, String> {
                let name = format!("{prefix}.{p}");
                info(ck.part(&name).ok_or_else(|| format!("{name} is not in the scan"))?)
            };
            inv.groups.insert(
                prefix.clone(),
                GroupInfo {
                    k: g.k,
                    n: g.n,
                    bits: g.bits,
                    codebook: g.codebook,
                    in_scales: g.in_scales,
                    out_scales: g.out_scales,
                    trellis: part("trellis")?,
                    in_scale: part(g.in_scales)?,
                    out_scale: part(g.out_scales)?,
                    bias: if g.bias { Some(part("bias")?) } else { None },
                    marker: if g.codebook == Codebook::Inst3 { None } else { Some(part(g.codebook.name())?) },
                },
            );
        }
        Ok(inv)
    }

    /// Build an inventory from safetensors headers: a JSON map of `name -> {"dtype", "shape", "data_offsets": [lo, hi]}`
    /// (the format `safetensors` writes), with all tensors laid out sequentially in one synthetic shard. Groups are detected by
    /// the `.trellis` part and parsed with the checkpoint scanner's own `parse_group`, so a header the scanner would refuse is
    /// refused here. Used by the tests (stub pack and synthetic layouts).
    pub fn from_headers(headers: &BTreeMap<String, serde_json::Value>) -> Result<Self, String> {
        const PARTS: [&str; 8] = ["trellis", "suh", "su", "svh", "sv", "mcg", "mul1", "bias"];
        let file = "model-00001-of-00001.safetensors".to_string();
        let mut tensors: BTreeMap<String, (String, Vec<usize>)> = BTreeMap::new();
        for (name, v) in headers {
            let dtype = v.get("dtype").and_then(|x| x.as_str()).ok_or_else(|| format!("{name}: missing dtype"))?;
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|x| x.as_array())
                .ok_or_else(|| format!("{name}: missing shape"))?
                .iter()
                .map(|x| x.as_u64().map(|x| x as usize).ok_or_else(|| format!("{name}: bad shape")))
                .collect::<Result<_, _>>()?;
            tensors.insert(name.clone(), (dtype.to_string(), shape));
        }
        // which names are group parts?
        let mut by_prefix: BTreeSet<String> = BTreeSet::new();
        for name in tensors.keys() {
            if let Some((prefix, part)) = name.rsplit_once('.') {
                if PARTS.contains(&part) && tensors.contains_key(&format!("{prefix}.trellis")) {
                    by_prefix.insert(prefix.to_string());
                }
            }
        }
        let mut inv = PackInventory { files: vec![file.clone()], ..Default::default() };
        let mut cursor = 0u64;
        let mut place = |name: &str, tensors: &BTreeMap<String, (String, Vec<usize>)>| -> Result<(Entry, TensorInfo), String> {
            let (dtype, shape) = &tensors[name];
            let size = dtype_size(dtype).ok_or_else(|| format!("{name}: unknown dtype {dtype}"))?;
            let bytes = shape.iter().product::<usize>() as u64 * size;
            let e = Entry { file: file.clone(), dtype: dtype.clone(), shape: shape.clone(), offsets: (cursor, cursor + bytes) };
            let t = TensorInfo { file: 0, dtype: dtype.clone(), shape: shape.clone(), offset: cursor, bytes };
            cursor += bytes;
            Ok((e, t))
        };
        let is_group_part = |n: &str| n.rsplit_once('.').is_some_and(|(p, part)| PARTS.contains(&part) && by_prefix.contains(p));
        let mut units: Vec<(String, bool)> = tensors.keys().filter(|n| !is_group_part(n)).map(|n| (n.clone(), false)).collect();
        units.extend(by_prefix.iter().map(|p| (p.clone(), true)));
        units.sort();
        for (name, is_group) in units {
            if !is_group {
                let (_, t) = place(&name, &tensors)?;
                inv.plain.insert(name, t);
                continue;
            }
            let mut entries: BTreeMap<String, Entry> = BTreeMap::new();
            let mut infos: BTreeMap<String, TensorInfo> = BTreeMap::new();
            for part in PARTS {
                let full = format!("{name}.{part}");
                if tensors.contains_key(&full) {
                    let (e, t) = place(&full, &tensors)?;
                    entries.insert(part.to_string(), e);
                    infos.insert(part.to_string(), t);
                }
            }
            match parse_group(&name, &entries) {
                Ok(g) => {
                    let get = |p: &str| infos.get(p).cloned().ok_or_else(|| format!("{}.{p} is missing", g.prefix));
                    let ginfo = GroupInfo {
                        k: g.k,
                        n: g.n,
                        bits: g.bits,
                        codebook: g.codebook,
                        in_scales: g.in_scales,
                        out_scales: g.out_scales,
                        trellis: get("trellis")?,
                        in_scale: get(g.in_scales)?,
                        out_scale: get(g.out_scales)?,
                        bias: if g.bias { Some(get("bias")?) } else { None },
                        marker: if g.codebook == Codebook::Inst3 { None } else { Some(get(g.codebook.name())?) },
                    };
                    inv.groups.insert(name, ginfo);
                }
                Err(why) => {
                    inv.refused.insert(name, why);
                }
            }
        }
        Ok(inv)
    }
}
