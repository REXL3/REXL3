//! The pack's tensor inventory: every name the index maps, with the shard it is loaded from and the dtype, shape and byte offset that
//! shard's header states. It is built from the index and the shard headers only; no tensor data is read.
//!
//! The index is the only authority on placement. A shard header may list a name that the index maps to another shard (the duplicate
//! copies the published pack carries). Such an entry is a [`Stray`]: recorded here, never loaded.
//!
//! **Names.** Two layouts of the text backbone's names are known, and every name here is CANONICAL, the first one's: a pack is read in
//! exactly one of them ([`Layout::detect`]), and a pack that mixes them, or names a tensor that is neither text nor vision, is refused by
//! name. The plan, the bridge and every lookup see canonical names only; the files are read by the shard and offset each name's header
//! record gives, so a layout never changes which bytes are read.

use super::weights::{read_shard_header, ShardHeader};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// The index file's name in a pack directory.
pub const INDEX_FILE: &str = "model.safetensors.index.json";

/// The canonical prefix of every text-backbone tensor (`language_model.backbone.*`, `language_model.mtp.*`, `language_model.lm_head.*`).
pub const TEXT_PREFIX: &str = "language_model.";

/// The vision tower's and projector's names, the same in both layouts: outside the text plan, counted and not loaded.
pub const VISION_PREFIXES: [&str; 3] = ["vision_model.", "mlp1.", "vision_projector."];

/// The roots of the text backbone's names in the [`Layout::Bare`] layout.
const BARE_ROOTS: [&str; 3] = ["backbone.", "mtp.", "lm_head."];

/// How a pack names its text backbone's tensors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Layout {
    /// `language_model.backbone.*`, `language_model.mtp.*`, `language_model.lm_head.*`: the published pack's (and every pack of ours).
    /// Canonical: names are used as they are.
    #[default]
    Prefixed,
    /// `backbone.*`, `mtp.*`, `lm_head.*`, no `language_model.`: exllamav3's own writer (the MixedK pack
    /// vcruz305/Nemotron-Super-VL-EXL3-3.85bpw). A text name is canonical with [`TEXT_PREFIX`] in front of it.
    Bare,
}

impl Layout {
    /// The layout of an index's names: [`Layout::Bare`] when no name has [`TEXT_PREFIX`] and some name starts with `backbone.`, `mtp.`
    /// or `lm_head.`; else [`Layout::Prefixed`] (also for an index with no text name at all: the plan then names what is missing). Refused
    /// by name: names of both layouts in one index, and, in the bare layout, a name that is neither a text name nor a vision name (the
    /// prefixed layout leaves such a name to the plan's checks, as it always did).
    pub fn detect<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<Layout, String> {
        let (mut prefixed, mut bare, mut other) = (None, None, None);
        for n in names {
            if n.starts_with(TEXT_PREFIX) {
                prefixed.get_or_insert(n);
            } else if BARE_ROOTS.iter().any(|r| n.starts_with(r)) {
                bare.get_or_insert(n);
            } else if !VISION_PREFIXES.iter().any(|v| n.starts_with(v)) {
                other.get_or_insert(n);
            }
        }
        match (prefixed, bare, other) {
            (Some(p), Some(b), _) => Err(format!(
                "{INDEX_FILE}: the text backbone is named both with `{TEXT_PREFIX}` (`{p}`) and without it (`{b}`); a pack uses one layout"
            )),
            (None, Some(_), Some(o)) => Err(format!(
                "{INDEX_FILE}: `{o}` is neither a text tensor (`backbone.`, `mtp.`, `lm_head.`) nor a vision tensor ({})",
                VISION_PREFIXES.map(|v| format!("`{v}`")).join(", ")
            )),
            (None, Some(_), None) => Ok(Layout::Bare),
            _ => Ok(Layout::Prefixed),
        }
    }

    /// The canonical name of a name of this layout (a vision name is the same in both).
    pub fn canonical(self, file_name: &str) -> String {
        match self {
            Layout::Bare if !VISION_PREFIXES.iter().any(|v| file_name.starts_with(v)) => format!("{TEXT_PREFIX}{file_name}"),
            _ => file_name.to_string(),
        }
    }

    /// The name the pack's files use for a canonical name (the inverse of [`Layout::canonical`]).
    pub fn file_name(self, canonical: &str) -> String {
        match self {
            Layout::Bare => canonical.strip_prefix(TEXT_PREFIX).unwrap_or(canonical).to_string(),
            Layout::Prefixed => canonical.to_string(),
        }
    }
}

/// One tensor as the index places it, with its header record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Byte offset of the tensor's data from the start of its shard.
    pub offset: u64,
    pub nbytes: u64,
}

/// A header entry the index does not place in that shard. Never loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stray {
    pub shard: String,
    pub name: String,
    pub nbytes: u64,
    /// The shard the index names for this tensor, or `None` when the index does not name it at all.
    pub indexed_in: Option<String>,
}

/// `model.safetensors.index.json`: the map from tensor name to shard, and the index's own total size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Index {
    pub weight_map: BTreeMap<String, String>,
    pub total_size: Option<u64>,
}

impl Index {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("{INDEX_FILE}: {e}"))?;
        let map = v.get("weight_map").and_then(Value::as_object).ok_or_else(|| format!("{INDEX_FILE}: no `weight_map`"))?;
        let mut weight_map = BTreeMap::new();
        for (name, shard) in map {
            let shard = shard.as_str().ok_or_else(|| format!("{INDEX_FILE}: `{name}` does not name a shard"))?;
            weight_map.insert(name.clone(), shard.to_string());
        }
        let total_size = v.get("metadata").and_then(|m| m.get("total_size")).and_then(Value::as_u64);
        Ok(Self { weight_map, total_size })
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_json(&text)
    }

    /// The distinct shard names the index uses, sorted.
    pub fn shards(&self) -> Vec<String> {
        let mut s: Vec<String> = self.weight_map.values().cloned().collect();
        s.sort();
        s.dedup();
        s
    }
}

/// The inventory of one pack: each indexed tensor with its header record, and the strays.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Inventory {
    /// By CANONICAL name ([`Layout`]); [`Tensor::name`] is the canonical name too.
    pub tensors: BTreeMap<String, Tensor>,
    /// The shard names the index uses, sorted.
    pub shards: Vec<String>,
    /// Header entries the index does not place, by the files' own names.
    pub strays: Vec<Stray>,
    pub index_total_size: Option<u64>,
    /// How the pack names its text backbone (the files' names are [`Layout::file_name`] of the canonical ones).
    pub layout: Layout,
}

impl Inventory {
    /// The inventory of the pack in `dir`: the index, then the header of each shard it names, each read once.
    pub fn read(dir: &Path) -> Result<Self, String> {
        let index = Index::read(&dir.join(INDEX_FILE))?;
        let mut headers = BTreeMap::new();
        for shard in index.shards() {
            let header = read_shard_header(&dir.join(&shard))?;
            headers.insert(shard, header);
        }
        Self::from_index(&index, &headers)
    }

    /// The inventory from an index and the headers of the shards it names. A name the index maps to a shard whose header does not
    /// list it is an error; a header entry the index does not place there is a stray. The index's names are read in their layout
    /// ([`Layout::detect`], refused by name when they mix two) and kept by their canonical names.
    pub fn from_index(index: &Index, headers: &BTreeMap<String, ShardHeader>) -> Result<Self, String> {
        let layout = Layout::detect(index.weight_map.keys().map(String::as_str))?;
        let mut tensors = BTreeMap::new();
        for (name, shard) in &index.weight_map {
            let header = headers.get(shard).ok_or_else(|| format!("`{name}`: shard `{shard}` has no header"))?;
            let e = header
                .entries
                .get(name)
                .ok_or_else(|| format!("`{name}`: the index names `{shard}`, whose header does not list it"))?;
            let canonical = layout.canonical(name);
            let t = Tensor {
                name: canonical.clone(),
                shard: shard.clone(),
                dtype: e.dtype.clone(),
                shape: e.shape.clone(),
                offset: e.offset,
                nbytes: e.nbytes,
            };
            // a bare name and its canonical form cannot both be in one index (`detect` refuses the mix), so no name is overwritten
            tensors.insert(canonical, t);
        }
        let mut strays = Vec::new();
        for (shard, header) in headers {
            for (name, e) in &header.entries {
                if index.weight_map.get(name) != Some(shard) {
                    strays.push(Stray {
                        shard: shard.clone(),
                        name: name.clone(),
                        nbytes: e.nbytes,
                        indexed_in: index.weight_map.get(name).cloned(),
                    });
                }
            }
        }
        Ok(Self { tensors, shards: index.shards(), strays, index_total_size: index.total_size, layout })
    }

    /// The name the pack's files use for the canonical `name` (for messages that name a file's tensor).
    pub fn file_name(&self, name: &str) -> String {
        self.layout.file_name(name)
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Bytes of every indexed tensor (each counted once, from its index shard).
    pub fn indexed_bytes(&self) -> u64 {
        self.tensors.values().map(|t| t.nbytes).sum()
    }

    /// Bytes of every stray entry.
    pub fn stray_bytes(&self) -> u64 {
        self.strays.iter().map(|s| s.nbytes).sum()
    }

    /// `(layer, expert)` pairs whose up and down projections are both bf16 (`.weight`), in ascending order.
    pub fn bf16_expert_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for name in self.tensors.keys() {
            let Some(rest) = name.strip_prefix("language_model.backbone.layers.") else { continue };
            let Some(rest) = rest.strip_suffix(".up_proj.weight") else { continue };
            let parts: Vec<&str> = rest.split('.').collect();
            let [layer, "mixer", "experts", expert] = parts[..] else { continue };
            let (Ok(layer), Ok(expert)) = (layer.parse::<usize>(), expert.parse::<usize>()) else { continue };
            if self
                .tensors
                .contains_key(&format!("language_model.backbone.layers.{layer}.mixer.experts.{expert}.down_proj.weight"))
            {
                pairs.push((layer, expert));
            }
        }
        pairs.sort();
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::super::weights::HeaderEntry;
    use super::*;

    const TRIMMED: &str = include_str!("../../../../../testdata/nemotron_omni/model.safetensors.index.json");
    const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/nemotron_omni");

    fn header(entries: &[(&str, &str, Vec<usize>, u64)]) -> ShardHeader {
        let mut h = ShardHeader::default();
        let mut at = 0u64;
        for (name, dtype, shape, nbytes) in entries {
            h.entries.insert(
                name.to_string(),
                HeaderEntry { dtype: dtype.to_string(), shape: shape.clone(), offset: 1000 + at, nbytes: *nbytes },
            );
            at += nbytes;
        }
        h.file_len = 1000 + at;
        h
    }

    fn index(pairs: &[(&str, &str)]) -> Index {
        Index { weight_map: pairs.iter().map(|(n, s)| (n.to_string(), s.to_string())).collect(), total_size: None }
    }

    #[test]
    fn the_trimmed_fixture_index_names_expert_zero_down_projections_on_the_moe_layers() {
        let idx = Index::from_json(TRIMMED).unwrap();
        assert_eq!(idx.weight_map.len(), 40, "the trimmed index keeps one key per MoE layer");
        assert!(idx.total_size.is_none());
        let cfg_text = std::fs::read_to_string(format!("{FIXTURE_DIR}/config.json")).unwrap();
        let cfg = super::super::config::OmniConfig::from_json(&serde_json::from_str(&cfg_text).unwrap()).unwrap();
        let moe: Vec<usize> = cfg.pattern.chars().enumerate().filter(|(_, c)| *c == 'E').map(|(i, _)| i).collect();
        assert_eq!(moe.len(), 40);
        for (name, shard) in &idx.weight_map {
            let layer: usize = name.split('.').nth(3).unwrap().parse().unwrap();
            assert!(moe.contains(&layer), "{name}: layer {layer} is not a MoE layer");
            assert_eq!(name, &format!("language_model.backbone.layers.{layer}.mixer.experts.0.down_proj.trellis"));
            assert!(shard.starts_with("model-") && shard.ends_with(".safetensors"), "{shard}");
        }
        assert_eq!(idx.shards().len(), 28, "the trimmed index spans 28 shards of the pack");
    }

    #[test]
    fn a_duplicate_copy_is_a_stray_and_the_index_places_the_tensor() {
        let idx = index(&[("a.weight", "s2.safetensors")]);
        let mut headers = BTreeMap::new();
        headers.insert("s1.safetensors".to_string(), header(&[("a.weight", "BF16", vec![4], 8)]));
        headers.insert("s2.safetensors".to_string(), header(&[("a.weight", "BF16", vec![4], 8)]));
        let inv = Inventory::from_index(&idx, &headers).unwrap();
        assert_eq!(inv.tensors.len(), 1);
        assert_eq!(inv.get("a.weight").unwrap().shard, "s2.safetensors");
        assert_eq!(
            inv.strays,
            vec![Stray {
                shard: "s1.safetensors".into(),
                name: "a.weight".into(),
                nbytes: 8,
                indexed_in: Some("s2.safetensors".into())
            }]
        );
        assert_eq!(inv.stray_bytes(), 8);
        assert_eq!(inv.indexed_bytes(), 8);
    }

    #[test]
    fn an_index_name_its_shard_does_not_list_is_refused_by_name() {
        let idx = index(&[("b.weight", "s1.safetensors")]);
        let mut headers = BTreeMap::new();
        headers.insert("s1.safetensors".to_string(), header(&[("a.weight", "BF16", vec![4], 8)]));
        let err = Inventory::from_index(&idx, &headers).unwrap_err();
        assert!(err.contains("`b.weight`") && err.contains("s1.safetensors"), "{err}");
    }

    #[test]
    fn an_unindexed_header_entry_is_a_stray_without_a_shard() {
        let idx = index(&[]);
        let mut headers = BTreeMap::new();
        headers.insert("s1.safetensors".to_string(), header(&[("x", "F32", vec![2], 8)]));
        let inv = Inventory::from_index(&idx, &headers).unwrap();
        assert!(inv.tensors.is_empty());
        assert_eq!(inv.strays[0].indexed_in, None);
    }

    #[test]
    fn the_layout_is_one_of_the_two_known_ones_and_a_mix_or_a_stranger_is_refused_by_name() {
        let ours =
            ["language_model.backbone.layers.0.norm.weight", "language_model.lm_head.weight", "vision_model.radio_model.x"];
        assert_eq!(Layout::detect(ours), Ok(Layout::Prefixed));
        let his = ["backbone.layers.0.norm.weight", "mtp.layers.0.eh_proj.trellis", "lm_head.trellis", "mlp1.0.weight"];
        assert_eq!(Layout::detect(his), Ok(Layout::Bare));
        assert_eq!(Layout::detect(["vision_projector.vision_final_layernorm.weight"]), Ok(Layout::Prefixed), "no text name");
        assert_eq!(Layout::detect([]), Ok(Layout::Prefixed));
        let mixed =
            Layout::detect(["language_model.backbone.layers.0.norm.weight", "backbone.layers.1.norm.weight"]).unwrap_err();
        assert!(
            mixed.contains("`language_model.backbone.layers.0.norm.weight`") && mixed.contains("`backbone.layers.1.norm.weight`")
        );
        assert!(mixed.contains("one layout"), "{mixed}");
        let stranger = Layout::detect(["backbone.layers.0.norm.weight", "model.layers.0.norm.weight"]).unwrap_err();
        assert!(stranger.contains("`model.layers.0.norm.weight` is neither a text tensor"), "{stranger}");
        // the canonical name and back; a vision name is the same in both layouts
        for (file, canonical) in [
            ("backbone.layers.7.mixer.q_proj.trellis", "language_model.backbone.layers.7.mixer.q_proj.trellis"),
            ("mtp.layers.1.mixer.gate.weight", "language_model.mtp.layers.1.mixer.gate.weight"),
            ("lm_head.svh", "language_model.lm_head.svh"),
            ("vision_model.radio_model.input_conditioner.norm_mean", "vision_model.radio_model.input_conditioner.norm_mean"),
        ] {
            assert_eq!(Layout::Bare.canonical(file), canonical);
            assert_eq!(Layout::Bare.file_name(canonical), file);
            assert_eq!(Layout::Prefixed.canonical(canonical), canonical);
            assert_eq!(Layout::Prefixed.file_name(canonical), canonical);
        }
    }

    #[test]
    fn a_bare_index_is_read_by_canonical_names_from_the_same_header_records() {
        let idx = index(&[("backbone.norm_f.weight", "s1.safetensors"), ("mlp1.0.weight", "s1.safetensors")]);
        let mut headers = BTreeMap::new();
        headers.insert(
            "s1.safetensors".to_string(),
            header(&[
                ("backbone.norm_f.weight", "BF16", vec![4], 8),
                ("mlp1.0.weight", "F32", vec![2], 8),
                ("dup", "F32", vec![2], 8),
            ]),
        );
        let inv = Inventory::from_index(&idx, &headers).unwrap();
        assert_eq!(inv.layout, Layout::Bare);
        assert_eq!(inv.tensors.keys().collect::<Vec<_>>(), ["language_model.backbone.norm_f.weight", "mlp1.0.weight"]);
        let t = inv.get("language_model.backbone.norm_f.weight").unwrap();
        let e = &headers["s1.safetensors"].entries["backbone.norm_f.weight"];
        assert_eq!(
            (t.name.as_str(), t.offset, t.nbytes, t.dtype.as_str()),
            ("language_model.backbone.norm_f.weight", e.offset, 8, "BF16")
        );
        assert!(inv.get("backbone.norm_f.weight").is_none(), "only canonical names are looked up");
        assert_eq!(inv.file_name("language_model.backbone.norm_f.weight"), "backbone.norm_f.weight");
        // a stray keeps the file's own name
        assert_eq!(inv.strays.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["dup"]);
        // a mixed index is refused before anything is read
        let idx = index(&[("backbone.norm_f.weight", "s1.safetensors"), ("language_model.lm_head.weight", "s1.safetensors")]);
        assert!(Inventory::from_index(&idx, &headers).unwrap_err().contains("one layout"));
    }

    #[test]
    fn the_bf16_expert_pairs_need_both_projections_as_weight() {
        let base = "language_model.backbone.layers.26.mixer.experts";
        let mut tensors = BTreeMap::new();
        for (name, shard) in [
            (format!("{base}.111.up_proj.weight"), 0u64),
            (format!("{base}.111.down_proj.weight"), 0),
            (format!("{base}.7.up_proj.weight"), 0),
        ] {
            tensors.insert(
                name.clone(),
                Tensor { name, shard: "s".into(), dtype: "BF16".into(), shape: vec![1], offset: shard, nbytes: 2 },
            );
        }
        let inv = Inventory { tensors, ..Default::default() };
        assert_eq!(inv.bf16_expert_pairs(), vec![(26, 111)], "expert 7 has only its up projection in bf16");
    }
}
