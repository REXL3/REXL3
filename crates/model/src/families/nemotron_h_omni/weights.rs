//! Index-driven weight binding and upload for Nemotron 3.5 Super VL.
//!
//! The pack is read through `model.safetensors.index.json` and nothing else. Each name is loaded from the shard the index gives
//! it; a header entry in another shard (the duplicate copies the published pack carries) is never read. [`read_shard_header`]
//! reads the 8-byte length and the JSON header of one shard and checks each record; it never reads tensor data.
//!
//! [`LayerPlan::build`] maps every trunk layer (`M`, `*`, `E` from `config.rs`) and every MTP block (`*E`) to the tensors it needs,
//! with the dtype and shape the plan expects. It refuses, naming the unit and the tensor, when a tensor is missing, has another
//! dtype or shape, or is claimed by no unit. A routed expert projection is EXL3 (`.trellis`, `.suh`, `.svh`, `.mul1`) or bf16
//! (`.weight`), whichever the index names. Vision tensors (`vision_model.`, `mlp1.`, `vision_projector.`) are outside this plan and
//! are counted, not loaded.
//!
//! [`upload`] copies each planned tensor's bytes, unchanged, from its shard into an arena view of the same size, through the
//! pipelined loader, and checks the loader's byte total against the header sums. [`Weights`] looks tensors up by name, typed.

use super::config::OmniConfig;
use super::inventory::{Inventory, TEXT_PREFIX};
use gb_cuda::arena::Arena;
use gb_cuda::exl3_linear::Exl3Weights;
use gb_cuda::loader::{Loader, Options, Report, Xform};
use gb_cuda::DevBuf;
use gb_exl3::checkpoint::{dtype_size, read_header_checked, Group};
use gb_exl3::{Bits, Codebook};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The text backbone's tensor names all start with this (canonical names: `inventory::Layout`).
pub const LM: &str = "language_model";
/// Bits per weight of an EXL3 linear the plan is not given a width for ([`units`], [`units_with_head`]): its trellis is
/// `[K/16, N/16, 16 * bits]`, last dimension 64. Every linear of our packs. [`LayerPlan::build`] reads each linear's width from the index
/// instead ([`widths_of`]: the MixedK pack's are 2 to 8 bits, per projection).
pub const EXL3_BITS: usize = 4;
/// The whole widths an EXL3 linear of the plan may have: the dense decode kernels are instantiated for 1 to 8 bits (`exl3_linear.cu`)
/// and the MoE block's reference kernels decode up to 8. A fractional width (1.5, 2.5, 3.5 bits) is refused by name.
pub const EXL3_WIDTHS: std::ops::RangeInclusive<usize> = 1..=8;
/// Arena block size for the upload (the same as the shared loader context's).
const ARENA_BLOCK_BYTES: usize = 1 << 30;

/// One tensor's record in a shard header. `offset` is absolute, from the start of the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeaderEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub nbytes: u64,
}

/// What one shard's header lists. Nothing here is tensor data.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShardHeader {
    pub file_len: u64,
    pub entries: BTreeMap<String, HeaderEntry>,
}

/// Read the length and JSON header of a safetensors file and check each record: ordered extents inside the data section, and for
/// a known dtype, a size of shape times element size. Tensor data is never read.
pub fn read_shard_header(path: &Path) -> Result<ShardHeader, String> {
    let (header_len, data_len, map) = read_header_checked(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let data_start = 8 + header_len;
    let mut entries = BTreeMap::new();
    for (name, v) in map {
        let (dtype, shape, start, nbytes) =
            parse_entry(&v, data_len).map_err(|e| format!("{}: `{name}`: {e}", path.display()))?;
        entries.insert(name, HeaderEntry { dtype, shape, offset: data_start + start, nbytes });
    }
    Ok(ShardHeader { file_len: data_start + data_len, entries })
}

fn parse_entry(v: &Value, data_len: u64) -> Result<(String, Vec<usize>, u64, u64), String> {
    let dtype = v.get("dtype").and_then(Value::as_str).ok_or("missing dtype")?.to_string();
    let shape = v
        .get("shape")
        .and_then(Value::as_array)
        .ok_or("missing shape")?
        .iter()
        .map(|d| d.as_u64().and_then(|x| usize::try_from(x).ok()).ok_or("shape holds a non-integer"))
        .collect::<Result<Vec<_>, _>>()?;
    let o =
        v.get("data_offsets").and_then(Value::as_array).filter(|o| o.len() == 2).ok_or("data_offsets must be two integers")?;
    let (a, b) =
        (o[0].as_u64().ok_or("data_offsets holds a non-integer")?, o[1].as_u64().ok_or("data_offsets holds a non-integer")?);
    if b < a || b > data_len {
        return Err(format!("data_offsets [{a}, {b}] lie outside the {data_len} data bytes"));
    }
    let size = dtype_size(&dtype).ok_or_else(|| format!("unknown dtype {dtype}"))?;
    let numel = shape.iter().try_fold(1u64, |n, &d| n.checked_mul(d as u64)).ok_or("shape overflows")?;
    let want = numel.checked_mul(size).ok_or("tensor size overflows")?;
    if want != b - a {
        return Err(format!("{dtype} {shape:?} needs {want} bytes, data_offsets span {}", b - a));
    }
    Ok((dtype, shape, a, b - a))
}

/// Which part of the model a tensor belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitId {
    /// Trunk layer `i`, in `layers_block_type` order.
    Trunk(usize),
    /// MTP block `b` (`0` is the attention block, `1` the latent MoE block).
    Mtp(usize),
    Embed,
    NormF,
    LmHead,
}

/// One tensor the plan needs: its name, and the dtype and shape the plan expects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Want {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub unit: UnitId,
}

/// An EXL3 linear: its four parts are `{prefix}.trellis`, `.suh`, `.svh` and `.mul1`. `k` is the input width, `n` the output, `bits` the
/// whole bits per weight (the trellis is `[k/16, n/16, 16 * bits]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Linear {
    pub prefix: String,
    pub k: usize,
    pub n: usize,
    pub bits: usize,
}

/// Each EXL3 linear's whole bits per weight, by prefix (canonical names), as the index's trellis shapes give them ([`widths_of`]). A
/// linear the map does not name has [`EXL3_BITS`].
pub type Widths = BTreeMap<String, usize>;

/// How a routed expert projection is stored in the index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    Exl3,
    Bf16,
}

/// The routed expert projections the index stores as bf16, keyed by (trunk layer, expert, `up_proj` or `down_proj`). The layer's
/// shared expert takes the expert number [`SHARED_EXPERT`]: its `up_proj` and `down_proj` may each be bf16 too (a pack that keeps a
/// badly quantised shared projection at the original precision). The latent projections and the Mamba and attention linears are
/// EXL3 only.
pub type Storages = BTreeMap<(usize, usize, &'static str), Storage>;

/// The expert number under which [`Storages`] records a layer's shared expert (`mixer.shared_experts.{up,down}_proj`).
pub const SHARED_EXPERT: usize = usize::MAX;

/// A layer or MTP block: its label for refusals, the tensors it needs, and its EXL3 linears.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    pub id: UnitId,
    pub label: String,
    pub tensors: Vec<Want>,
    pub linears: Vec<Linear>,
}

impl Unit {
    fn new(id: UnitId, label: String) -> Self {
        Self { id, label, tensors: Vec::new(), linears: Vec::new() }
    }

    fn plain(&mut self, name: String, dtype: &'static str, shape: Vec<usize>) {
        self.tensors.push(Want { name, dtype, shape, unit: self.id });
    }

    fn linear(&mut self, prefix: String, k: usize, n: usize) {
        let id = self.id;
        let part = |s: &str| format!("{prefix}.{s}");
        self.tensors.push(Want { name: part("trellis"), dtype: "I16", shape: vec![k / 16, n / 16, 16 * EXL3_BITS], unit: id });
        self.tensors.push(Want { name: part("suh"), dtype: "F16", shape: vec![k], unit: id });
        self.tensors.push(Want { name: part("svh"), dtype: "F16", shape: vec![n], unit: id });
        self.tensors.push(Want { name: part("mul1"), dtype: "I32", shape: vec![], unit: id });
        self.linears.push(Linear { prefix, k, n, bits: EXL3_BITS });
    }

    /// Each linear `widths` names at its width: the linear's `bits` and its trellis' last dimension (`16 * bits`). The rest keep theirs.
    fn set_widths(&mut self, widths: &Widths) {
        for l in &mut self.linears {
            if let Some(&b) = widths.get(&l.prefix) {
                l.bits = b;
            }
        }
        for t in &mut self.tensors {
            if let Some(&b) = t.name.strip_suffix(".trellis").and_then(|p| widths.get(p)) {
                t.shape[2] = 16 * b;
            }
        }
    }
}

/// The whole width of every EXL3 trellis of the text backbone the index names (`[K/16, N/16, 16 * bits]`: the last dimension over 16),
/// by linear prefix, or a refusal by name: a trellis that is not three-dimensional, whose last dimension is no width ExLlamaV3 writes, or
/// whose width is fractional or past [`EXL3_WIDTHS`] (no kernel of this engine decodes it). Mixed widths within a layer and within one
/// expert (an up and a down projection of different widths) are per projection here, as the tables hold them.
pub fn widths_of(inv: &Inventory) -> (Widths, Vec<String>) {
    let mut widths = Widths::new();
    let mut errors = Vec::new();
    for (name, t) in inv.tensors.iter().filter(|(n, _)| n.starts_with(TEXT_PREFIX)) {
        let Some(prefix) = name.strip_suffix(".trellis") else { continue };
        let file = inv.file_name(name);
        let [_, _, last] = t.shape[..] else {
            errors.push(format!("`{file}` is {} {:?}: an EXL3 trellis is [K/16, N/16, 16 * bits]", t.dtype, t.shape));
            continue;
        };
        match Bits::from_trellis_last(last) {
            Some(b) if b.is_integer() && EXL3_WIDTHS.contains(&(b.twice() as usize / 2)) => {
                widths.insert(prefix.to_string(), b.twice() as usize / 2);
            }
            Some(b) => errors.push(format!(
                "`{file}` is {} bits per weight: this engine serves whole widths of {} to {} bits only",
                b.as_f64(),
                EXL3_WIDTHS.start(),
                EXL3_WIDTHS.end()
            )),
            None => errors.push(format!("`{file}`: a last dimension of {last} is 16 x no width ExLlamaV3 writes")),
        }
    }
    (widths, errors)
}

/// The MTP head's projections the index stores as EXL3 (`{prefix}.trellis`, `.suh`, `.svh`, `.mul1`), by prefix (the tensor name
/// without `.weight`): the 4-bit head. A head projection not in the set is bf16 `{prefix}.weight`, which is every one of them on the
/// published pack (the set is empty). Only the projections [`head_projections`] lists may be EXL3; the norms and the router stay bf16.
/// Our 4-bit head pack keeps the attention bf16; the MixedK pack's head is EXL3 throughout (`eh_proj`, the attention, the MoE block).
pub type HeadExl3 = BTreeSet<String>;

/// The MTP head's projections that may be stored as EXL3, as `(prefix, k, n)` (`k` the input width, `n` the output): `eh_proj` and the
/// attention's q, k, v and o (block 0), and the latent projections, the shared expert and every routed expert's up and down of the MoE
/// block (block 1).
pub fn head_projections(cfg: &OmniConfig) -> Vec<(String, usize, usize)> {
    let h = cfg.hidden;
    let b0 = format!("{LM}.mtp.layers.0");
    let m = format!("{LM}.mtp.layers.1.mixer");
    let (attn_n, kv_n) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
    let mut v = vec![
        (format!("{b0}.eh_proj"), 2 * h, h),
        (format!("{b0}.mixer.q_proj"), h, attn_n),
        (format!("{b0}.mixer.k_proj"), h, kv_n),
        (format!("{b0}.mixer.v_proj"), h, kv_n),
        (format!("{b0}.mixer.o_proj"), attn_n, h),
        (format!("{m}.fc1_latent_proj"), h, cfg.latent),
        (format!("{m}.fc2_latent_proj"), cfg.latent, h),
        (format!("{m}.shared_experts.up_proj"), h, cfg.shared_inter),
        (format!("{m}.shared_experts.down_proj"), cfg.shared_inter, h),
    ];
    for e in 0..cfg.experts {
        v.push((format!("{m}.experts.{e}.up_proj"), cfg.latent, cfg.expert_inter));
        v.push((format!("{m}.experts.{e}.down_proj"), cfg.expert_inter, cfg.latent));
    }
    v
}

/// Which head projections the index stores as EXL3: a projection whose `{prefix}.trellis` the index names. One with both forms is an
/// error naming it; one with neither is left to the plan's missing-tensor check (it then asks for the bf16 `.weight`), and one with a
/// partial EXL3 set (a trellis without its scales or marker) is refused there by the missing part: the head fails closed. A pack with no
/// head (`mtp_pattern` empty) has no head projections.
pub fn head_exl3_of(cfg: &OmniConfig, inv: &Inventory) -> (HeadExl3, Vec<String>) {
    let mut set = HeadExl3::new();
    let mut errors = Vec::new();
    if cfg.mtp_pattern.is_empty() {
        return (set, errors);
    }
    for (prefix, _, _) in head_projections(cfg) {
        match (inv.get(&format!("{prefix}.trellis")).is_some(), inv.get(&format!("{prefix}.weight")).is_some()) {
            (true, true) => errors
                .push(format!("MTP head: `{prefix}` is in both EXL3 (`{prefix}.trellis`) and bf16 (`{prefix}.weight`) form")),
            (true, false) => {
                set.insert(prefix);
            }
            _ => {}
        }
    }
    (set, errors)
}

/// The units of the text backbone, in order: the trunk layers, the MTP blocks, then the embedding, final norm and lm_head.
/// `storages` says how each routed expert projection is stored; an absent projection is EXL3. The head is all bf16 ([`units_with_head`]
/// plans a 4-bit head), every linear 4-bit ([`units_with`] gives each its width).
pub fn units(cfg: &OmniConfig, storages: &Storages) -> Vec<Unit> {
    units_with_head(cfg, storages, &HeadExl3::new())
}

/// [`units_with_head`] with each linear `widths` names at its width (the others 4-bit). An empty `widths` is [`units_with_head`].
///
/// The `lm_head` is planned as the EXL3 linear `{LM}.lm_head` (`k` = hidden, `n` = vocabulary) when `widths` names it (the index stores
/// its trellis: the MixedK pack's 8-bit head), else as the bf16 `{LM}.lm_head.weight` `[vocab, hidden]`.
pub fn units_with(cfg: &OmniConfig, storages: &Storages, head: &HeadExl3, widths: &Widths) -> Vec<Unit> {
    let mut units = units_with_head(cfg, storages, head);
    let lm_head = lm_head_prefix();
    if widths.contains_key(&lm_head) {
        let u = units.iter_mut().find(|u| u.id == UnitId::LmHead).expect("the units hold the lm_head");
        *u = Unit::new(UnitId::LmHead, "lm_head".into());
        u.linear(lm_head, cfg.hidden, cfg.vocab);
    }
    for u in &mut units {
        u.set_widths(widths);
    }
    units
}

/// The `lm_head`'s name without a suffix: the prefix of its EXL3 linear, or of its bf16 `.weight`.
pub fn lm_head_prefix() -> String {
    format!("{LM}.lm_head")
}

/// [`units`] with the head's EXL3 projections `head` ([`head_exl3_of`]): each is planned as an EXL3 linear (its four parts), every other
/// head projection as bf16 `.weight`. An empty `head` is [`units`].
pub fn units_with_head(cfg: &OmniConfig, storages: &Storages, head: &HeadExl3) -> Vec<Unit> {
    let inner = cfg.mamba_heads * cfg.mamba_dim;
    let conv_dim = inner + 2 * cfg.groups * cfg.state;
    let in_proj_n = 2 * inner + 2 * cfg.groups * cfg.state + cfg.mamba_heads;
    let attn_n = cfg.heads * cfg.head_dim;
    let kv_n = cfg.kv_heads * cfg.head_dim;
    let mut out = Vec::new();

    for (i, kind) in cfg.pattern.chars().enumerate() {
        let base = format!("{LM}.backbone.layers.{i}");
        let mut u = Unit::new(UnitId::Trunk(i), format!("layer {i} ({kind})"));
        u.plain(format!("{base}.norm.weight"), "BF16", vec![cfg.hidden]);
        let m = format!("{base}.mixer");
        match kind {
            'M' => {
                for part in ["A_log", "D", "dt_bias"] {
                    u.plain(format!("{m}.{part}"), "BF16", vec![cfg.mamba_heads]);
                }
                u.plain(format!("{m}.conv1d.weight"), "BF16", vec![conv_dim, 1, cfg.conv]);
                u.plain(format!("{m}.conv1d.bias"), "BF16", vec![conv_dim]);
                u.plain(format!("{m}.norm.weight"), "BF16", vec![inner]);
                u.linear(format!("{m}.in_proj"), cfg.hidden, in_proj_n);
                u.linear(format!("{m}.out_proj"), inner, cfg.hidden);
            }
            '*' => {
                u.linear(format!("{m}.q_proj"), cfg.hidden, attn_n);
                u.linear(format!("{m}.k_proj"), cfg.hidden, kv_n);
                u.linear(format!("{m}.v_proj"), cfg.hidden, kv_n);
                u.linear(format!("{m}.o_proj"), attn_n, cfg.hidden);
            }
            _ => {
                u.plain(format!("{m}.gate.weight"), "BF16", vec![cfg.experts, cfg.hidden]);
                u.plain(format!("{m}.gate.e_score_correction_bias"), "F32", vec![cfg.experts]);
                u.linear(format!("{m}.fc1_latent_proj"), cfg.hidden, cfg.latent);
                u.linear(format!("{m}.fc2_latent_proj"), cfg.latent, cfg.hidden);
                for (proj, k, n) in [("up_proj", cfg.hidden, cfg.shared_inter), ("down_proj", cfg.shared_inter, cfg.hidden)] {
                    let prefix = format!("{m}.shared_experts.{proj}");
                    match storages.get(&(i, SHARED_EXPERT, proj)).copied().unwrap_or(Storage::Exl3) {
                        Storage::Exl3 => u.linear(prefix, k, n),
                        Storage::Bf16 => u.plain(format!("{prefix}.weight"), "BF16", vec![n, k]),
                    }
                }
                for e in 0..cfg.experts {
                    for (proj, k, n) in [("up_proj", cfg.latent, cfg.expert_inter), ("down_proj", cfg.expert_inter, cfg.latent)] {
                        let prefix = format!("{m}.experts.{e}.{proj}");
                        match storages.get(&(i, e, proj)).copied().unwrap_or(Storage::Exl3) {
                            Storage::Exl3 => u.linear(prefix, k, n),
                            Storage::Bf16 => u.plain(format!("{prefix}.weight"), "BF16", vec![n, k]),
                        }
                    }
                }
            }
        }
        out.push(u);
    }

    // a head projection `[n, k]`: EXL3 when the index stores it so, else bf16 `.weight`
    let proj = |u: &mut Unit, prefix: String, k: usize, n: usize| {
        if head.contains(&prefix) {
            u.linear(prefix, k, n);
        } else {
            u.plain(format!("{prefix}.weight"), "BF16", vec![n, k]);
        }
    };
    for (b, kind) in cfg.mtp_pattern.chars().enumerate() {
        let base = format!("{LM}.mtp.layers.{b}");
        let mut u = Unit::new(UnitId::Mtp(b), format!("mtp block {b} ({kind})"));
        if kind == '*' {
            proj(&mut u, format!("{base}.eh_proj"), 2 * cfg.hidden, cfg.hidden);
            u.plain(format!("{base}.enorm.weight"), "BF16", vec![cfg.hidden]);
            u.plain(format!("{base}.hnorm.weight"), "BF16", vec![cfg.hidden]);
            u.plain(format!("{base}.norm.weight"), "BF16", vec![cfg.hidden]);
            for (name, n, k) in [
                ("q_proj", attn_n, cfg.hidden),
                ("k_proj", kv_n, cfg.hidden),
                ("v_proj", kv_n, cfg.hidden),
                ("o_proj", cfg.hidden, attn_n),
            ] {
                proj(&mut u, format!("{base}.mixer.{name}"), k, n);
            }
        } else {
            let m = format!("{base}.mixer");
            u.plain(format!("{base}.final_layernorm.weight"), "BF16", vec![cfg.hidden]);
            u.plain(format!("{base}.norm.weight"), "BF16", vec![cfg.hidden]);
            u.plain(format!("{m}.gate.weight"), "BF16", vec![cfg.experts, cfg.hidden]);
            u.plain(format!("{m}.gate.e_score_correction_bias"), "F32", vec![cfg.experts]);
            proj(&mut u, format!("{m}.fc1_latent_proj"), cfg.hidden, cfg.latent);
            proj(&mut u, format!("{m}.fc2_latent_proj"), cfg.latent, cfg.hidden);
            proj(&mut u, format!("{m}.shared_experts.up_proj"), cfg.hidden, cfg.shared_inter);
            proj(&mut u, format!("{m}.shared_experts.down_proj"), cfg.shared_inter, cfg.hidden);
            for e in 0..cfg.experts {
                proj(&mut u, format!("{m}.experts.{e}.up_proj"), cfg.latent, cfg.expert_inter);
                proj(&mut u, format!("{m}.experts.{e}.down_proj"), cfg.expert_inter, cfg.latent);
            }
        }
        out.push(u);
    }

    let global = |id: UnitId, label: &str, name: String, shape: Vec<usize>| {
        let mut u = Unit::new(id, label.to_string());
        u.plain(name, "BF16", shape);
        u
    };
    out.push(global(UnitId::Embed, "embeddings", format!("{LM}.backbone.embeddings.weight"), vec![cfg.vocab, cfg.hidden]));
    out.push(global(UnitId::NormF, "final norm", format!("{LM}.backbone.norm_f.weight"), vec![cfg.hidden]));
    out.push(global(UnitId::LmHead, "lm_head", format!("{LM}.lm_head.weight"), vec![cfg.vocab, cfg.hidden]));
    out
}

/// Which routed expert projections, and which shared-expert projections ([`SHARED_EXPERT`]), the index stores as bf16 (`.weight`). A
/// projection with both forms, or with neither, is not recorded here: the first is returned as an error, the second is left to the
/// plan's missing-tensor check.
pub fn storages_of(cfg: &OmniConfig, inv: &Inventory) -> (Storages, Vec<String>) {
    let mut map = Storages::new();
    let mut errors = Vec::new();
    for (layer, kind) in cfg.pattern.chars().enumerate() {
        if kind != 'E' {
            continue;
        }
        let base = format!("{LM}.backbone.layers.{layer}.mixer.experts");
        for e in 0..cfg.experts {
            for proj in ["up_proj", "down_proj"] {
                let prefix = format!("{base}.{e}.{proj}");
                match (inv.get(&format!("{prefix}.trellis")).is_some(), inv.get(&format!("{prefix}.weight")).is_some()) {
                    (true, true) => errors.push(format!(
                        "layer {layer} (E): expert {e} `{proj}` is in both EXL3 (`{prefix}.trellis`) and bf16 (`{prefix}.weight`) form"
                    )),
                    (false, true) => {
                        map.insert((layer, e, proj), Storage::Bf16);
                    }
                    _ => {}
                }
            }
        }
        for proj in ["up_proj", "down_proj"] {
            let prefix = format!("{LM}.backbone.layers.{layer}.mixer.shared_experts.{proj}");
            match (inv.get(&format!("{prefix}.trellis")).is_some(), inv.get(&format!("{prefix}.weight")).is_some()) {
                (true, true) => errors.push(format!(
                    "layer {layer} (E): the shared expert's `{proj}` is in both EXL3 (`{prefix}.trellis`) and bf16 (`{prefix}.weight`) form"
                )),
                (false, true) => {
                    map.insert((layer, SHARED_EXPERT, proj), Storage::Bf16);
                }
                _ => {}
            }
        }
    }
    (map, errors)
}

/// The router gates (`*.mixer.gate.weight`, `[experts, hidden]`) the index stores as F16 are planned as F16 (v0.1.2 decision D1: the
/// bridge widens an F16 gate to f32 on the device for the f32-gate router kernels, `bridge::take_gate`; it is never rounded to bf16).
/// Every other gate stays BF16, so a gate of any third dtype, or of another shape, is still refused by the presence check.
fn router_gate_dtypes(units: &mut [Unit], inv: &Inventory) {
    for w in units.iter_mut().flat_map(|u| u.tensors.iter_mut()) {
        if w.name.ends_with(".mixer.gate.weight") && w.dtype == "BF16" {
            if let Some(t) = inv.get(&w.name) {
                if t.dtype == "F16" && t.shape == w.shape {
                    w.dtype = "F16";
                }
            }
        }
    }
}

/// The trunk's latent projections (`fc1_latent_proj` `[hidden -> latent]`, `fc2_latent_proj` `[latent -> hidden]`) the index stores as
/// an F16 `.weight` `[n, k]` with no trellis (the MixedK pack's 80) are planned as that plain F16 tensor instead of an EXL3 linear: the
/// MoE kernels read it in place as `Proj::F16` (the bf16 chain on the values widened exactly; never rounded to bf16, no copy). One with
/// both forms is an error naming it; a `.weight` of any other dtype or shape keeps the EXL3 plan, so the presence check refuses it by
/// name (the missing trellis, the unclaimed weight): fail-closed. Our packs (EXL3 latents) plan exactly as before.
fn latent_f16(units: &mut [Unit], inv: &Inventory) -> Vec<String> {
    let mut errors = Vec::new();
    for u in units.iter_mut().filter(|u| matches!(u.id, UnitId::Trunk(_))) {
        let latents: Vec<Linear> = u
            .linears
            .iter()
            .filter(|l| l.prefix.ends_with(".fc1_latent_proj") || l.prefix.ends_with(".fc2_latent_proj"))
            .cloned()
            .collect();
        for l in latents {
            let weight = format!("{}.weight", l.prefix);
            let Some(t) = inv.get(&weight) else { continue };
            if inv.get(&format!("{}.trellis", l.prefix)).is_some() {
                errors.push(format!(
                    "{}: `{}` is in both EXL3 (`{}.trellis`) and dense (`{}`) form",
                    u.label,
                    inv.file_name(&l.prefix),
                    inv.file_name(&l.prefix),
                    inv.file_name(&weight)
                ));
                continue;
            }
            if t.dtype != "F16" || t.shape != [l.n, l.k] {
                continue; // refused by the presence check: the trellis is missing and the weight unclaimed
            }
            let part = format!("{}.", l.prefix);
            u.tensors.retain(|w| !w.name.starts_with(&part));
            u.linears.retain(|x| x.prefix != l.prefix);
            u.plain(weight, "F16", vec![l.n, l.k]);
        }
    }
    errors
}

/// The plan: every unit of the text backbone, each tensor checked against the inventory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerPlan {
    pub units: Vec<Unit>,
    /// Tensors outside the text backbone (the vision tower and projector), counted and not loaded.
    pub outside_plan: usize,
}

impl LayerPlan {
    /// The plan for the pack, or every refusal: a missing tensor, a dtype or shape other than the plan's, a language tensor that no
    /// unit claims, an expert projection in both forms, or an EXL3 width no kernel serves. Each message names the unit or the tensor.
    /// Every EXL3 linear is planned at the width its trellis has in the index ([`widths_of`]).
    pub fn build(cfg: &OmniConfig, inv: &Inventory) -> Result<Self, Vec<String>> {
        let (storages, mut errors) = storages_of(cfg, inv);
        let (head, head_errors) = head_exl3_of(cfg, inv);
        errors.extend(head_errors);
        let (widths, width_errors) = widths_of(inv);
        errors.extend(width_errors);
        let lm = lm_head_prefix();
        if inv.get(&format!("{lm}.trellis")).is_some() && inv.get(&format!("{lm}.weight")).is_some() {
            errors
                .push(format!("lm_head: `{0}` is in both EXL3 (`{0}.trellis`) and bf16 (`{0}.weight`) form", inv.file_name(&lm)));
        }
        let mut units = units_with(cfg, &storages, &head, &widths);
        router_gate_dtypes(&mut units, inv);
        errors.extend(latent_f16(&mut units, inv));
        Self::check(units, inv, errors)
    }

    /// The `lm_head` this plan has as an EXL3 linear (the MixedK pack's), or `None` for a bf16 `lm_head`.
    pub fn lm_head_exl3(&self) -> Option<&Linear> {
        self.unit(UnitId::LmHead).and_then(|u| u.linears.first())
    }

    /// The head projections this plan has as EXL3 linears (the 4-bit head), by prefix: empty for a bf16 head or none.
    pub fn head_exl3(&self) -> HeadExl3 {
        self.units
            .iter()
            .filter(|u| matches!(u.id, UnitId::Mtp(_)))
            .flat_map(|u| u.linears.iter().map(|l| l.prefix.clone()))
            .collect()
    }

    /// Check `units` against the inventory. A tensor claimed by two places is refused: the upload reads each planned tensor once.
    fn check(units: Vec<Unit>, inv: &Inventory, mut errors: Vec<String>) -> Result<Self, Vec<String>> {
        let mut claimed = BTreeSet::new();
        for u in &units {
            for w in &u.tensors {
                if !claimed.insert(w.name.as_str()) {
                    errors.push(format!("{}: `{}` is claimed twice in the plan", u.label, w.name));
                    continue;
                }
                // a refusal names the tensor as the pack's files do (`Inventory::file_name`: the canonical name in the prefixed layout)
                match inv.get(&w.name) {
                    None => errors.push(format!("{}: missing `{}`", u.label, inv.file_name(&w.name))),
                    Some(t) if t.dtype != w.dtype || t.shape != w.shape => errors.push(format!(
                        "{}: `{}` is {} {:?} in the index, the plan expects {} {:?}",
                        u.label,
                        inv.file_name(&w.name),
                        t.dtype,
                        t.shape,
                        w.dtype,
                        w.shape
                    )),
                    Some(_) => {}
                }
            }
        }
        for name in inv.tensors.keys().filter(|n| n.starts_with(TEXT_PREFIX)) {
            if !claimed.contains(name.as_str()) {
                errors.push(format!("`{}`: no unit of the plan claims this tensor", inv.file_name(name)));
            }
        }
        if !errors.is_empty() {
            return Err(errors);
        }
        let outside_plan = inv.tensors.keys().filter(|n| !n.starts_with(TEXT_PREFIX)).count();
        Ok(Self { units, outside_plan })
    }

    /// The tensors of the units `pick` selects, in plan order.
    pub fn tensors(&self, pick: impl Fn(UnitId) -> bool) -> Vec<&Want> {
        self.units.iter().filter(|u| pick(u.id)).flat_map(|u| u.tensors.iter()).collect()
    }

    pub fn unit(&self, id: UnitId) -> Option<&Unit> {
        self.units.iter().find(|u| u.id == id)
    }

    /// Bytes of every planned tensor, from the inventory, plus what the device holds beyond a tensor's bytes in a form our packs do not
    /// have ([`held_extra_bytes`]: an F16 router gate, widened). For every pack of ours this is the inventory's sum, as it always was.
    pub fn bytes(&self, inv: &Inventory) -> u64 {
        self.units
            .iter()
            .flat_map(|u| u.tensors.iter())
            .filter_map(|w| inv.get(&w.name).map(|t| (w, t.nbytes)))
            .map(|(w, n)| n + held_extra_bytes(w, n))
            .sum()
    }
}

/// Device bytes a planned plain tensor of `nbytes` in the index takes beyond them, as the bridge holds it. Every planned tensor is held
/// as stored (0): an F16 router gate (`*.gate.weight`, the MixedK pack's) is its fp16 bits, read by the fp16 router kernels (lane K,
/// decision D1; it was widened to an f32 copy, twice its bytes, until the in-register widening landed), and an F16 latent projection
/// (`fc1_latent_proj` / `fc2_latent_proj` `.weight`) is read in place by the fp16 dense kernels. The memory plan follows the form the
/// bridge holds, from this one function: a form held otherwise gets its arm here.
pub fn held_extra_bytes(_want: &Want, _nbytes: u64) -> u64 {
    0
}

/// A tensor on the device, with the place its bytes came from.
pub struct Uploaded {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub shard: String,
    /// Byte offset of the tensor's data in its shard.
    pub offset: u64,
    pub nbytes: u64,
    pub buf: DevBuf,
}

/// An EXL3 linear's four parts on the device.
pub struct LinearParts<'a> {
    pub linear: &'a Linear,
    pub trellis: &'a Uploaded,
    pub suh: &'a Uploaded,
    pub svh: &'a Uploaded,
    pub mul1: &'a Uploaded,
}

/// An EXL3 linear moved out of an upload as its raw parts ([`Weights::take_parts`]). `suh` and `svh` are fp16 as the pack stores them;
/// the `mul1` marker part is dropped, as [`Weights::take_linear`] drops it (the codebook is `Mul1`).
pub struct TakenLinear {
    pub linear: Linear,
    pub bits: Bits,
    /// int16 `[K/16, N/16, 16 * bits]`.
    pub trellis: Uploaded,
    /// fp16 `[K]`.
    pub suh: Uploaded,
    /// fp16 `[N]`.
    pub svh: Uploaded,
}

/// Every tensor of one upload, by name.
pub struct Weights {
    tensors: BTreeMap<String, Uploaded>,
    linears: BTreeMap<String, Linear>,
    /// What the loader copied.
    pub report: Report,
    /// The header sum of every uploaded tensor; equal to `report.bytes` once the load succeeds.
    pub bytes: u64,
}

impl Weights {
    /// An upload assembled from tensors that are already on the device (or in host memory, for a test without a GPU). `linears` names
    /// the EXL3 linears whose four parts are among `tensors`. [`upload`] is the one constructor that reads a pack.
    pub fn from_parts(tensors: Vec<Uploaded>, linears: Vec<Linear>) -> Self {
        let bytes = tensors.iter().map(|t| t.nbytes).sum();
        Self {
            tensors: tensors.into_iter().map(|t| (t.name.clone(), t)).collect(),
            linears: linears.into_iter().map(|l| (l.prefix.clone(), l)).collect(),
            report: Report { bytes, ..Report::default() },
            bytes,
        }
    }

    /// Move a plain tensor out of the upload. A name that is not on the device (or an EXL3 part already taken) is an error naming it.
    pub fn take(&mut self, name: &str) -> Result<Uploaded, String> {
        self.tensors.remove(name).ok_or_else(|| format!("`{name}` is not on the device"))
    }

    /// Move an EXL3 linear's trellis, `suh` and `svh` out as they are (fp16 scales). For consumers that rebuild the layer in another form
    /// (the MoE expert tables). The `mul1` marker leaves the upload too.
    pub fn take_parts(&mut self, prefix: &str) -> Result<TakenLinear, String> {
        let linear = self.linears.remove(prefix).ok_or_else(|| format!("`{prefix}` is not an EXL3 linear of this upload"))?;
        let mut take =
            |s: &str| self.tensors.remove(&format!("{prefix}.{s}")).ok_or_else(|| format!("`{prefix}.{s}` is not on the device"));
        let (trellis, suh, svh, _mul1) = (take("trellis")?, take("suh")?, take("svh")?, take("mul1")?);
        let bits = trellis
            .shape
            .get(2)
            .and_then(|&last| Bits::from_trellis_last(last))
            .ok_or_else(|| format!("`{prefix}.trellis`: unsupported bit width"))?;
        Ok(TakenLinear { linear, bits, trellis, suh, svh })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&Uploaded> {
        self.tensors.get(name)
    }

    /// The tensor under `name`, which must have `dtype` and `shape`.
    pub fn typed(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<&Uploaded, String> {
        let t = self.tensors.get(name).ok_or_else(|| format!("`{name}` is not on the device"))?;
        if t.dtype != dtype || t.shape != shape {
            return Err(format!("`{name}` is {} {:?}, not {dtype} {shape:?}", t.dtype, t.shape));
        }
        Ok(t)
    }

    /// Whether the upload holds the EXL3 linear `prefix`.
    pub fn has_linear(&self, prefix: &str) -> bool {
        self.linears.contains_key(prefix)
    }

    /// An EXL3 linear's parts, by prefix.
    pub fn linear(&self, prefix: &str) -> Result<LinearParts<'_>, String> {
        let linear = self.linears.get(prefix).ok_or_else(|| format!("`{prefix}` is not an EXL3 linear of this upload"))?;
        let part =
            |s: &str| self.tensors.get(&format!("{prefix}.{s}")).ok_or_else(|| format!("`{prefix}.{s}` is not on the device"));
        Ok(LinearParts { linear, trellis: part("trellis")?, suh: part("suh")?, svh: part("svh")?, mul1: part("mul1")? })
    }

    /// Move an EXL3 linear's four parts out as a device layer ([`Exl3Weights::from_parts`]). The parts leave this upload.
    pub fn take_linear(&mut self, prefix: &str) -> Result<Exl3Weights, String> {
        let linear = self.linears.remove(prefix).ok_or_else(|| format!("`{prefix}` is not an EXL3 linear of this upload"))?;
        let mut take =
            |s: &str| self.tensors.remove(&format!("{prefix}.{s}")).ok_or_else(|| format!("`{prefix}.{s}` is not on the device"));
        let (trellis, suh, svh, mul1) = (take("trellis")?, take("suh")?, take("svh")?, take("mul1")?);
        let bits =
            Bits::from_trellis_last(trellis.shape[2]).ok_or_else(|| format!("`{prefix}.trellis`: unsupported bit width"))?;
        let group = Group {
            prefix: prefix.to_string(),
            bits,
            codebook: Codebook::Mul1,
            k: linear.k,
            n: linear.n,
            in_scales: "suh",
            out_scales: "svh",
            bias: false,
            files: Vec::new(),
            bytes: trellis.nbytes + suh.nbytes + svh.nbytes + mul1.nbytes,
        };
        drop(mul1);
        Exl3Weights::from_parts(&group, trellis.buf, suh.buf, svh.buf, None).map_err(|e| format!("`{prefix}`: {e}"))
    }
}

/// Upload every planned tensor the `pick` selects: each one's bytes, unchanged, from its shard into a device view of its own size.
/// Each tensor is read once, by its index entry. The loader's byte total must equal the header sum.
pub fn upload(
    dir: &Path,
    inv: &Inventory,
    plan: &LayerPlan,
    pick: impl Fn(UnitId) -> bool,
    o: &Options,
) -> Result<Weights, String> {
    let wants = plan.tensors(&pick);
    let arena = Arena::new(ARENA_BLOCK_BYTES);
    let mut loader = Loader::new();
    let mut pending = Vec::with_capacity(wants.len());
    let mut bytes = 0u64;
    let mut linears = BTreeMap::new();
    for w in &wants {
        let t = inv.get(&w.name).ok_or_else(|| format!("`{}` is not in the inventory", w.name))?;
        let len = usize::try_from(t.nbytes).map_err(|_| format!("`{}`: {} bytes", w.name, t.nbytes))?;
        let buf = arena.alloc(len).map_err(|e| format!("`{}`: {e}", w.name))?;
        let file = loader.file(&dir.join(&t.shard));
        loader.copy_file(&buf, file, t.offset, len, Xform::None).map_err(|e| format!("`{}`: {e}", w.name))?;
        bytes += t.nbytes;
        pending.push((w, t, buf));
    }
    for u in plan.units.iter().filter(|u| pick(u.id)) {
        for l in &u.linears {
            linears.insert(l.prefix.clone(), l.clone());
        }
    }
    let report = loader.run(o).map_err(|e| format!("loader: {e}"))?;
    if report.bytes != bytes {
        return Err(format!("the loader copied {} bytes, the headers sum to {bytes}", report.bytes));
    }
    let tensors = pending
        .into_iter()
        .map(|(w, t, buf)| {
            let up = Uploaded {
                name: w.name.clone(),
                dtype: w.dtype,
                shape: w.shape.clone(),
                shard: t.shard.clone(),
                offset: t.offset,
                nbytes: t.nbytes,
                buf,
            };
            (w.name.clone(), up)
        })
        .collect();
    Ok(Weights { tensors, linears, report, bytes })
}

#[cfg(test)]
mod tests {
    use super::super::inventory::{Index, Inventory, Tensor};
    use super::*;

    const CONFIG: &str = include_str!("../../../../../testdata/nemotron_omni/config.json");
    const TRIMMED: &str = include_str!("../../../../../testdata/nemotron_omni/model.safetensors.index.json");
    const SSM_FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/nemotron_omni/ssm/layer0_fixture.safetensors");
    /// The measured bf16 expert pairs: (trunk layer, expert). Both projections of each are `.weight`.
    const BF16_PAIRS: [(usize, usize); 3] = [(26, 111), (39, 304), (81, 461)];

    fn cfg() -> OmniConfig {
        OmniConfig::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    /// The measured storage: EXL3 everywhere except the bf16 pairs.
    fn measured_storages() -> Storages {
        let mut storages = Storages::new();
        for (l, e) in BF16_PAIRS {
            for p in ["up_proj", "down_proj"] {
                storages.insert((l, e, p), Storage::Bf16);
            }
        }
        storages
    }

    /// Every tensor the plan expects, laid out as the pack holds them (EXL3 except the measured bf16 pairs), plus one vision tensor.
    fn expected_inventory(cfg: &OmniConfig) -> Inventory {
        inventory_with(cfg, &measured_storages())
    }

    /// The same for any storage map (a patched pack: more routed projections, or shared-expert projections, kept bf16).
    fn inventory_with(cfg: &OmniConfig, storages: &Storages) -> Inventory {
        inventory_with_head(cfg, storages, &HeadExl3::new())
    }

    /// The same with the head's projections `head` in EXL3 form (a 4-bit head pack).
    fn inventory_with_head(cfg: &OmniConfig, storages: &Storages, head: &HeadExl3) -> Inventory {
        inventory_of(units_with_head(cfg, storages, head))
    }

    /// Every tensor of `units`, laid out one after another, plus one vision tensor.
    fn inventory_of(units: Vec<Unit>) -> Inventory {
        let mut tensors = BTreeMap::new();
        let mut at = 0u64;
        let mut put = |tensors: &mut BTreeMap<String, Tensor>, name: &str, dtype: &str, shape: &[usize]| {
            let numel: u64 = shape.iter().map(|&d| d as u64).product();
            let nbytes = numel * dtype_size(dtype).unwrap();
            tensors.insert(
                name.to_string(),
                Tensor {
                    name: name.into(),
                    shard: "synthetic.safetensors".into(),
                    dtype: dtype.into(),
                    shape: shape.to_vec(),
                    offset: at,
                    nbytes,
                },
            );
            at += nbytes;
        };
        for u in units {
            for w in &u.tensors {
                put(&mut tensors, &w.name, w.dtype, &w.shape);
            }
        }
        put(&mut tensors, "vision_model.radio_model.input_conditioner.norm_mean", "F32", &[3, 1, 1]);
        Inventory {
            tensors,
            shards: vec!["synthetic.safetensors".into()],
            strays: vec![],
            index_total_size: None,
            ..Default::default()
        }
    }

    fn remove(inv: &mut Inventory, name: &str) {
        assert!(inv.tensors.remove(name).is_some(), "{name} was not in the synthetic inventory");
    }

    fn refusal(cfg: &OmniConfig, inv: &Inventory) -> String {
        match LayerPlan::build(cfg, inv) {
            Ok(_) => panic!("the plan should refuse"),
            Err(errors) => errors.join("\n"),
        }
    }

    #[test]
    fn the_expected_inventory_plans_every_language_tensor_once() {
        let cfg = cfg();
        let inv = expected_inventory(&cfg);
        let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
        assert_eq!(plan.units.len(), 88 + 2 + 3, "88 trunk layers, 2 MTP blocks, and the embedding, final norm and lm_head");
        let claimed: usize = plan.units.iter().map(|u| u.tensors.len()).sum();
        assert_eq!(claimed, 166_361, "the text backbone's tensors: 165,318 trunk, 1,040 MTP, 3 global");
        assert_eq!(plan.outside_plan, 1);
        assert_eq!(plan.unit(UnitId::Trunk(26)).unwrap().tensors.len(), 4109, "an E layer with one bf16 expert pair");
        assert_eq!(plan.unit(UnitId::Trunk(0)).unwrap().tensors.len(), 15, "a Mamba2 layer");
        assert_eq!(plan.unit(UnitId::Trunk(7)).unwrap().tensors.len(), 17, "an attention layer");
        assert_eq!(plan.tensors(|u| u == UnitId::LmHead).len(), 1);
        assert_eq!(plan.unit(UnitId::Mtp(1)).unwrap().tensors.len(), 8 + 1024);
    }

    #[test]
    fn a_patched_pack_keeps_routed_and_shared_projections_at_bf16_and_the_plan_follows_the_index() {
        let cfg = cfg();
        let mut storages = measured_storages();
        // a routed projection of one expert only (not the pair), and the shared expert's projections of two layers
        storages.insert((3, 12, "up_proj"), Storage::Bf16);
        storages.insert((10, SHARED_EXPERT, "down_proj"), Storage::Bf16);
        storages.insert((1, SHARED_EXPERT, "up_proj"), Storage::Bf16);
        let inv = inventory_with(&cfg, &storages);
        // storages_of reads exactly the map back from the inventory
        let (found, errors) = storages_of(&cfg, &inv);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(found, storages);
        let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
        let l10 = plan.unit(UnitId::Trunk(10)).unwrap();
        let down = "language_model.backbone.layers.10.mixer.shared_experts.down_proj";
        let w =
            l10.tensors.iter().find(|t| t.name == format!("{down}.weight")).expect("the bf16 shared down projection is planned");
        assert_eq!((w.dtype, w.shape.clone()), ("BF16", vec![cfg.hidden, cfg.shared_inter]));
        assert!(
            l10.tensors.iter().all(|t| !t.name.starts_with(&format!("{down}.")) || t.name.ends_with(".weight")),
            "no EXL3 part of a bf16 shared projection is planned"
        );
        assert!(!l10.linears.iter().any(|l| l.prefix == down));
        // the other shared projection of that layer stays EXL3
        let up = "language_model.backbone.layers.10.mixer.shared_experts.up_proj";
        assert!(l10.linears.iter().any(|l| l.prefix == up));
        let l1 = plan.unit(UnitId::Trunk(1)).unwrap();
        let up1 =
            l1.tensors.iter().find(|t| t.name == "language_model.backbone.layers.1.mixer.shared_experts.up_proj.weight").unwrap();
        assert_eq!(up1.shape, vec![cfg.shared_inter, cfg.hidden]);
        // a routed projection kept at bf16 without its partner
        let l3 = plan.unit(UnitId::Trunk(3)).unwrap();
        assert!(l3.tensors.iter().any(|t| t.name == "language_model.backbone.layers.3.mixer.experts.12.up_proj.weight"));
        assert!(l3.linears.iter().any(|l| l.prefix == "language_model.backbone.layers.3.mixer.experts.12.down_proj"));
        // four EXL3 parts become one tensor
        let base = LayerPlan::build(&cfg, &expected_inventory(&cfg)).unwrap();
        assert_eq!(l10.tensors.len(), base.unit(UnitId::Trunk(10)).unwrap().tensors.len() - 3);
        assert_eq!(l3.tensors.len(), base.unit(UnitId::Trunk(3)).unwrap().tensors.len() - 3);
    }

    /// A 4-bit head: every scoped projection EXL3 except the ones a flagged-matrix policy would keep (here routed expert 3's down and the
    /// shared up stay bf16).
    fn four_bit_head(cfg: &OmniConfig) -> HeadExl3 {
        let keep_bf16 =
            ["language_model.mtp.layers.1.mixer.experts.3.down_proj", "language_model.mtp.layers.1.mixer.shared_experts.up_proj"];
        head_projections(cfg)
            .into_iter()
            .map(|(p, _, _)| p)
            .filter(|p| !keep_bf16.contains(&p.as_str()) && !is_head_attention(p))
            .collect()
    }

    /// Whether `prefix` is one of the head attention's q, k, v and o (bf16 on our 4-bit head pack, EXL3 on the MixedK pack's).
    fn is_head_attention(prefix: &str) -> bool {
        ["q_proj", "k_proj", "v_proj", "o_proj"].iter().any(|p| prefix == format!("language_model.mtp.layers.0.mixer.{p}"))
    }

    #[test]
    fn the_published_head_is_bf16_and_a_four_bit_head_is_planned_as_the_index_stores_it() {
        let cfg = cfg();
        // the published pack: no head projection is EXL3, the plan is the bf16 one
        let inv = expected_inventory(&cfg);
        assert!(head_exl3_of(&cfg, &inv).0.is_empty());
        let bf16 = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
        assert!(bf16.head_exl3().is_empty());
        assert!(bf16.unit(UnitId::Mtp(0)).unwrap().linears.is_empty() && bf16.unit(UnitId::Mtp(1)).unwrap().linears.is_empty());
        // the 4-bit head: the index decides each projection's form
        let head = four_bit_head(&cfg);
        assert_eq!(head.len(), 1 + 4 + 2 * cfg.experts - 2);
        let inv = inventory_with_head(&cfg, &measured_storages(), &head);
        let (found, errors) = head_exl3_of(&cfg, &inv);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(found, head);
        let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
        assert_eq!(plan.head_exl3(), head);
        let (u0, u1) = (plan.unit(UnitId::Mtp(0)).unwrap(), plan.unit(UnitId::Mtp(1)).unwrap());
        let eh = "language_model.mtp.layers.0.eh_proj";
        assert_eq!(u0.linears, vec![Linear { prefix: eh.into(), k: 2 * cfg.hidden, n: cfg.hidden, bits: EXL3_BITS }]);
        let tr = u0.tensors.iter().find(|t| t.name == format!("{eh}.trellis")).expect("eh_proj's trellis is planned");
        assert_eq!((tr.dtype, tr.shape.clone()), ("I16", vec![2 * cfg.hidden / 16, cfg.hidden / 16, 16 * EXL3_BITS]));
        assert!(!u0.tensors.iter().any(|t| t.name == format!("{eh}.weight")));
        // the attention, the norms and the router stay bf16
        for name in ["language_model.mtp.layers.0.mixer.q_proj.weight", "language_model.mtp.layers.0.enorm.weight"] {
            assert!(u0.tensors.iter().any(|t| t.name == name && t.dtype == "BF16"), "{name}");
        }
        assert!(u1.tensors.iter().any(|t| t.name == "language_model.mtp.layers.1.mixer.gate.weight" && t.dtype == "BF16"));
        // the projections kept bf16 are `.weight`, the others EXL3 linears with the trunk's shapes
        let m = "language_model.mtp.layers.1.mixer";
        assert!(u1.tensors.iter().any(|t| t.name == format!("{m}.experts.3.down_proj.weight")));
        assert!(u1.tensors.iter().any(|t| t.name == format!("{m}.shared_experts.up_proj.weight")));
        assert!(u1.linears.contains(&Linear {
            prefix: format!("{m}.experts.3.up_proj"),
            k: cfg.latent,
            n: cfg.expert_inter,
            bits: 4
        }));
        assert!(u1.linears.contains(&Linear {
            prefix: format!("{m}.shared_experts.down_proj"),
            k: cfg.shared_inter,
            n: cfg.hidden,
            bits: 4
        }));
        assert_eq!(u1.linears.len(), 2 * cfg.experts - 1 + 3);
        // a linear is four tensors where a bf16 projection is one; the trunk's plan is unchanged
        assert_eq!(u0.tensors.len(), bf16.unit(UnitId::Mtp(0)).unwrap().tensors.len() + 3);
        assert_eq!(u1.tensors.len(), bf16.unit(UnitId::Mtp(1)).unwrap().tensors.len() + 3 * (head.len() - 1));
        assert_eq!(plan.unit(UnitId::Trunk(26)), bf16.unit(UnitId::Trunk(26)));
        // the 4-bit expert's trellis is the trunk expert's shape
        let t = |name: &str| inv.get(name).map(|t| (t.dtype.clone(), t.shape.clone())).unwrap();
        for part in ["trellis", "suh", "svh", "mul1"] {
            assert_eq!(
                t(&format!("{m}.experts.7.up_proj.{part}")),
                t(&format!("language_model.backbone.layers.1.mixer.experts.7.up_proj.{part}")),
                "{part}"
            );
        }
    }

    /// P0-3: the MixedK pack's head is EXL3 throughout: `eh_proj`, the attention's q, k, v and o and the MoE block's projections are
    /// planned as EXL3 linears with the attention's shapes, the norms and the router stay bf16, and each linear takes its own width.
    #[test]
    fn a_head_with_exl3_attention_is_planned_as_the_index_stores_it() {
        let cfg = cfg();
        let all: HeadExl3 = head_projections(&cfg).into_iter().map(|(p, _, _)| p).collect();
        assert_eq!(all.len(), 1 + 4 + 4 + 2 * cfg.experts);
        let b0 = "language_model.mtp.layers.0.mixer";
        let widths = Widths::from([(format!("{b0}.k_proj"), 6), (format!("{b0}.o_proj"), 5)]);
        let inv = inventory_of(units_with(&cfg, &measured_storages(), &all, &widths));
        assert_eq!(head_exl3_of(&cfg, &inv).0, all);
        let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
        assert_eq!(plan.head_exl3(), all);
        let u0 = plan.unit(UnitId::Mtp(0)).unwrap();
        let (attn_n, kv_n) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
        for (p, k, n, bits) in [
            ("q_proj", cfg.hidden, attn_n, 4),
            ("k_proj", cfg.hidden, kv_n, 6),
            ("v_proj", cfg.hidden, kv_n, 4),
            ("o_proj", attn_n, cfg.hidden, 5),
        ] {
            let prefix = format!("{b0}.{p}");
            assert!(u0.linears.contains(&Linear { prefix: prefix.clone(), k, n, bits }), "{p}");
            let tr = u0.tensors.iter().find(|t| t.name == format!("{prefix}.trellis")).unwrap();
            assert_eq!(tr.shape, vec![k / 16, n / 16, 16 * bits], "{p}");
            assert!(!u0.tensors.iter().any(|t| t.name == format!("{prefix}.weight")), "{p} has no bf16 weight planned");
        }
        assert_eq!(u0.linears.len(), 5, "eh_proj and the attention");
        for name in ["language_model.mtp.layers.0.enorm.weight", "language_model.mtp.layers.0.norm.weight"] {
            assert!(u0.tensors.iter().any(|t| t.name == name && t.dtype == "BF16"), "{name}");
        }
        // a bf16 attention weight next to its trellis is in both forms: refused by name
        let mut inv = inv.clone();
        let w = format!("{b0}.q_proj.weight");
        let t = inv.tensors[&format!("{b0}.q_proj.trellis")].clone();
        inv.tensors.insert(w.clone(), Tensor { name: w, dtype: "BF16".into(), shape: vec![attn_n, cfg.hidden], ..t });
        let msg = refusal(&cfg, &inv);
        assert!(msg.contains(&format!("MTP head: `{b0}.q_proj` is in both EXL3")), "{msg}");
    }

    #[test]
    fn a_head_projection_in_both_forms_or_with_a_partial_exl3_set_is_refused() {
        let cfg = cfg();
        let head = four_bit_head(&cfg);
        let m = "language_model.mtp.layers.1.mixer";
        // both forms: the bf16 weight next to the trellis
        let mut inv = inventory_with_head(&cfg, &measured_storages(), &head);
        let w = format!("{m}.experts.9.up_proj.weight");
        let t = inv.tensors[&format!("{m}.experts.9.up_proj.trellis")].clone();
        inv.tensors.insert(w.clone(), Tensor { name: w, dtype: "BF16".into(), shape: vec![cfg.expert_inter, cfg.latent], ..t });
        let msg = refusal(&cfg, &inv);
        assert!(msg.contains(&format!("MTP head: `{m}.experts.9.up_proj` is in both EXL3")), "{msg}");
        // a partial set: the trellis without its output scales, and eh_proj without its marker
        for gone in [format!("{m}.experts.9.up_proj.svh"), "language_model.mtp.layers.0.eh_proj.mul1".to_string()] {
            let mut inv = inventory_with_head(&cfg, &measured_storages(), &head);
            remove(&mut inv, &gone);
            let msg = refusal(&cfg, &inv);
            assert!(msg.contains(&format!("missing `{gone}`")), "{gone}: {msg}");
        }
        // a projection with neither form asks for its bf16 weight
        let mut inv = inventory_with_head(&cfg, &measured_storages(), &head);
        for part in ["trellis", "suh", "svh", "mul1"] {
            remove(&mut inv, &format!("{m}.fc1_latent_proj.{part}"));
        }
        let msg = refusal(&cfg, &inv);
        assert!(msg.contains(&format!("mtp block 1 (E): missing `{m}.fc1_latent_proj.weight`")), "{msg}");
    }

    #[test]
    fn a_shared_projection_in_both_forms_is_refused() {
        let cfg = cfg();
        let mut inv = expected_inventory(&cfg);
        let name = "language_model.backbone.layers.10.mixer.shared_experts.down_proj.weight";
        let t = inv.tensors["language_model.backbone.layers.10.mixer.shared_experts.down_proj.trellis"].clone();
        inv.tensors.insert(name.into(), Tensor { name: name.into(), dtype: "BF16".into(), ..t });
        let msg = refusal(&cfg, &inv);
        assert!(msg.contains("layer 10 (E): the shared expert's `down_proj` is in both EXL3"), "{msg}");
    }

    #[test]
    fn a_missing_tensor_is_refused_with_its_layer_and_name() {
        let cfg = cfg();
        for (name, layer) in [
            ("language_model.backbone.layers.26.mixer.gate.weight", "layer 26 (E)"),
            ("language_model.backbone.layers.7.mixer.q_proj.mul1", "layer 7 (*)"),
            ("language_model.backbone.layers.0.mixer.conv1d.bias", "layer 0 (M)"),
            ("language_model.backbone.layers.26.mixer.experts.5.up_proj.svh", "layer 26 (E)"),
            ("language_model.mtp.layers.1.mixer.experts.511.down_proj.weight", "mtp block 1 (E)"),
            ("language_model.mtp.layers.0.eh_proj.weight", "mtp block 0 (*)"),
        ] {
            let mut inv = expected_inventory(&cfg);
            remove(&mut inv, name);
            let msg = refusal(&cfg, &inv);
            assert!(msg.contains(&format!("{layer}: missing `{name}`")), "{name}: {msg}");
        }
    }

    #[test]
    fn an_expert_with_neither_form_is_refused_with_the_missing_name() {
        let cfg = cfg();
        let mut inv = expected_inventory(&cfg);
        remove(&mut inv, "language_model.backbone.layers.26.mixer.experts.111.up_proj.weight");
        let msg = refusal(&cfg, &inv);
        assert!(
            msg.contains("layer 26 (E): missing `language_model.backbone.layers.26.mixer.experts.111.up_proj.trellis`"),
            "{msg}"
        );
    }

    #[test]
    fn a_wrong_dtype_or_shape_is_refused_with_the_tensor_and_both_sides() {
        let cfg = cfg();
        let mut inv = expected_inventory(&cfg);
        inv.tensors.get_mut("language_model.backbone.layers.0.mixer.A_log").unwrap().shape = vec![64];
        let msg = refusal(&cfg, &inv);
        assert!(msg.contains("layer 0 (M): `language_model.backbone.layers.0.mixer.A_log` is BF16 [64]"), "{msg}");
        let mut inv = expected_inventory(&cfg);
        inv.tensors.get_mut("language_model.backbone.layers.7.mixer.q_proj.trellis").unwrap().dtype = "F16".into();
        let msg = refusal(&cfg, &inv);
        assert!(msg.contains("layer 7 (*): `language_model.backbone.layers.7.mixer.q_proj.trellis` is F16"), "{msg}");
    }

    /// GUARD (v0.1.2 R-MK-rails (d)): an F16 router gate (the MixedK pack's, trunk and head) is refused at plan time by name. Its bytes
    /// are the bf16 gate's, so nothing downstream that counts bytes would notice it (the bridge's own dtype check is the second guard:
    /// `bridge::tests::an_f16_router_gate_is_refused_by_name_before_any_moe_table_even_when_the_plan_admits_it`).
    #[test]
    fn an_f16_router_gate_is_planned_as_f16_and_a_gate_of_any_other_dtype_is_refused_by_name() {
        // v0.1.2 D1 (lane K): an F16 gate is planned as F16, for the bridge's widening to f32 (`bridge::take_gate`) and the f32-gate
        // router kernels; it was refused here until that path existed (lane L's guard), and the bridge keeps refusing any gate that is
        // neither BF16 nor F16. A bf16 gate is planned as before.
        let cfg = cfg();
        let gates = ["language_model.backbone.layers.1.mixer.gate.weight", "language_model.mtp.layers.1.mixer.gate.weight"];
        let mut inv = expected_inventory(&cfg);
        for g in gates {
            let t = inv.tensors.get_mut(g).unwrap();
            let bf16_bytes = t.nbytes;
            t.dtype = "F16".into();
            assert_eq!(dtype_size("F16").unwrap() * (cfg.experts * cfg.hidden) as u64, bf16_bytes, "{g}: the same bytes as bf16");
        }
        let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("an F16 gate is refused: {e:?}"));
        for g in gates {
            let want = plan.units.iter().flat_map(|u| u.tensors.iter()).find(|w| w.name == g).unwrap();
            assert_eq!(want.dtype, "F16", "{g}");
        }
        let other =
            plan.units.iter().flat_map(|u| u.tensors.iter()).find(|w| w.name.contains("layers.3.mixer.gate.weight")).unwrap();
        assert_eq!(other.dtype, "BF16", "a bf16 gate of the same pack is still planned as BF16");
        // a third dtype (F32 [512, 4096], twice the bytes) is refused by name, as before
        let mut inv = expected_inventory(&cfg);
        let t = inv.tensors.get_mut(gates[0]).unwrap();
        t.dtype = "F32".into();
        t.nbytes *= 2;
        let msg = refusal(&cfg, &inv);
        assert!(
            msg.contains(&format!(
                "layer 1 (E): `{}` is F32 [512, 4096] in the index, the plan expects BF16 [512, 4096]",
                gates[0]
            )),
            "{msg}"
        );
    }

    /// The expected inventory as the files of a pack in `layout` name it: an index and one shard header, read back by
    /// `Inventory::from_index` (the reader the engine uses).
    fn reread(inv: &Inventory, layout: super::super::inventory::Layout) -> Inventory {
        use super::super::inventory::Index;
        let mut weight_map = BTreeMap::new();
        let mut h = ShardHeader::default();
        for t in inv.tensors.values() {
            let file = layout.file_name(&t.name);
            weight_map.insert(file.clone(), t.shard.clone());
            h.entries
                .insert(file, HeaderEntry { dtype: t.dtype.clone(), shape: t.shape.clone(), offset: t.offset, nbytes: t.nbytes });
        }
        let headers = BTreeMap::from([("synthetic.safetensors".to_string(), h)]);
        Inventory::from_index(&Index { weight_map, total_size: None }, &headers).unwrap_or_else(|e| panic!("{e}"))
    }

    /// MixedK widths in the shapes the MixedK pack has them: experts of 2 to 6 bits mixed within a layer and within an expert (up and
    /// down of different widths), the shared expert and the Mamba and attention linears at 4 to 8 bits.
    fn mixedk_widths() -> Widths {
        let l = |s: &str| format!("language_model.backbone.layers.{s}");
        Widths::from([
            (l("1.mixer.experts.0.up_proj"), 2),
            (l("1.mixer.experts.0.down_proj"), 3),
            (l("1.mixer.experts.1.up_proj"), 5),
            (l("1.mixer.experts.1.down_proj"), 6),
            (l("1.mixer.experts.2.up_proj"), 4),
            (l("1.mixer.experts.2.down_proj"), 5),
            (l("3.mixer.shared_experts.up_proj"), 6),
            (l("0.mixer.out_proj"), 5),
            (l("7.mixer.q_proj"), 5),
            (l("7.mixer.k_proj"), 7),
            (l("7.mixer.v_proj"), 8),
            (l("7.mixer.o_proj"), 4),
        ])
    }

    /// P0-1: every EXL3 linear is planned at the width its trellis has in the index, per projection; the rest stay 4-bit, and an index
    /// with every trellis at 4 bits is planned exactly as before.
    #[test]
    fn each_exl3_linear_is_planned_at_the_width_its_trellis_has_in_the_index() {
        let cfg = cfg();
        // our pack: every width 4, the plan the one of `units`
        let inv = expected_inventory(&cfg);
        let (w4, errors) = widths_of(&inv);
        assert!(errors.is_empty(), "{errors:?}");
        assert!(w4.values().all(|&b| b == 4) && w4.len() == 40 * (4 + 2 * 512) - 6 + 40 * 2 + 8 * 4, "{}", w4.len());
        let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
        assert_eq!(plan.units, units(&cfg, &measured_storages()), "a 4-bit index is planned exactly as before");
        // MixedK widths: the index decides each projection's width
        let widths = mixedk_widths();
        let inv = inventory_of(units_with(&cfg, &measured_storages(), &HeadExl3::new(), &widths));
        let (found, errors) = widths_of(&inv);
        assert!(errors.is_empty(), "{errors:?}");
        for (p, b) in &widths {
            assert_eq!(found.get(p), Some(b), "{p}");
        }
        let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
        for (p, &b) in &widths {
            let layer: usize = p.split('.').nth(3).unwrap().parse().unwrap();
            let u = plan.unit(UnitId::Trunk(layer)).unwrap();
            let l = u.linears.iter().find(|l| &l.prefix == p).unwrap();
            assert_eq!(l.bits, b, "{p}");
            let t = u.tensors.iter().find(|t| t.name == format!("{p}.trellis")).unwrap();
            assert_eq!(t.shape, vec![l.k / 16, l.n / 16, 16 * b], "{p}");
            assert_eq!(inv.get(&t.name).unwrap().nbytes, (l.k * l.n * b / 8) as u64, "{p}: {b} bits a weight");
        }
        // a neighbour the map does not name keeps 4 bits
        let u1 = plan.unit(UnitId::Trunk(1)).unwrap();
        assert_eq!(u1.linears.iter().find(|l| l.prefix.ends_with(".experts.3.up_proj")).unwrap().bits, 4);
        // the 4-bit plan of the same pack refuses the other widths by shape: the width is the index's, never assumed
        let msg = LayerPlan::check(units(&cfg, &measured_storages()), &inv, Vec::new()).unwrap_err().join("\n");
        assert!(
            msg.contains("`language_model.backbone.layers.7.mixer.k_proj.trellis` is I16 [256, 16, 112] in the index"),
            "{msg}"
        );
    }

    /// P0-4: an EXL3 `lm_head` (the MixedK pack's 8-bit head) is planned as the EXL3 linear `lm_head` `[hidden -> vocab]` at its width
    /// when the index stores its trellis; one in both forms is refused by name.
    #[test]
    fn an_exl3_lm_head_is_planned_as_an_exl3_linear_and_both_forms_are_refused() {
        let cfg = cfg();
        let widths = Widths::from([(lm_head_prefix(), 8)]);
        let inv = inventory_of(units_with(&cfg, &measured_storages(), &HeadExl3::new(), &widths));
        assert!(inv.get(LMH).is_none() && inv.get("language_model.lm_head.trellis").is_some());
        let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
        let l = plan.lm_head_exl3().expect("the lm_head is EXL3");
        assert_eq!((l.prefix.as_str(), l.k, l.n, l.bits), ("language_model.lm_head", cfg.hidden, cfg.vocab, 8));
        let u = plan.unit(UnitId::LmHead).unwrap();
        assert_eq!(u.label, "lm_head");
        let tr = u.tensors.iter().find(|t| t.name == "language_model.lm_head.trellis").unwrap();
        assert_eq!(tr.shape, vec![256, 8192, 128], "the MixedK pack's (256, 8192, 128)");
        assert_eq!(u.tensors.iter().find(|t| t.name == "language_model.lm_head.svh").unwrap().shape, vec![131_072]);
        assert_eq!(u.tensors.len(), 4);
        // the bf16 pack's lm_head is the one bf16 tensor, as before
        assert!(LayerPlan::build(&cfg, &expected_inventory(&cfg)).unwrap().lm_head_exl3().is_none());
        // both forms
        let mut both = inv.clone();
        let t = inv.tensors["language_model.backbone.embeddings.weight"].clone();
        both.tensors.insert(LMH.into(), Tensor { name: LMH.into(), ..t });
        let msg = refusal(&cfg, &both);
        assert!(msg.contains("lm_head: `language_model.lm_head` is in both EXL3"), "{msg}");
    }

    /// The plan counts an F16 router gate as the bridge holds it (widened to f32 today: twice its bytes), every other tensor as stored:
    /// our packs' sums are the inventory's.
    #[test]
    fn an_f16_gate_is_counted_as_held_and_everything_else_as_stored() {
        let gate = |dtype: &'static str| Want {
            name: "language_model.backbone.layers.1.mixer.gate.weight".into(),
            dtype,
            shape: vec![512, 4096],
            unit: UnitId::Trunk(1),
        };
        let n = 512 * 4096 * 2;
        assert_eq!(held_extra_bytes(&gate("F16"), n), 0, "an F16 gate is held as stored (the fp16 router kernels)");
        assert_eq!(held_extra_bytes(&gate("BF16"), n), 0);
        let suh = Want {
            name: "language_model.backbone.layers.1.mixer.fc1_latent_proj.suh".into(),
            dtype: "F16",
            shape: vec![4096],
            unit: UnitId::Trunk(1),
        };
        assert_eq!(held_extra_bytes(&suh, 8192), 0);
        let cfg = cfg();
        let inv = expected_inventory(&cfg);
        let plan = LayerPlan::build(&cfg, &inv).unwrap();
        assert_eq!(plan.bytes(&inv), plan.tensors(|_| true).iter().map(|w| inv.get(&w.name).unwrap().nbytes).sum::<u64>());
    }

    /// The MixedK pack's trunk `fc1_latent_proj` / `fc2_latent_proj` are F16 `.weight` `[n, k]`, not EXL3 (lane L's guard refused them
    /// until the F16 dense path landed, lane K): such a projection is planned as that plain F16 tensor (`Proj::F16`, read in place); one
    /// in both forms is refused by name; a dense weight of another dtype (BF16 here) still is, by the missing trellis and the unclaimed
    /// weight. Our pack's latents stay EXL3 linears.
    #[test]
    fn an_f16_dense_latent_projection_is_planned_as_f16_and_any_other_dense_form_refused_by_name() {
        let cfg = cfg();
        let p = "language_model.backbone.layers.1.mixer.fc1_latent_proj";
        let q = "language_model.backbone.layers.1.mixer.fc2_latent_proj";
        let dense = |dtype: &str, keep_trellis: bool| -> Inventory {
            let mut inv = expected_inventory(&cfg);
            for (pre, n, k) in [(p, cfg.latent, cfg.hidden), (q, cfg.hidden, cfg.latent)] {
                let t = inv.tensors[&format!("{pre}.trellis")].clone();
                if !keep_trellis {
                    for part in ["trellis", "suh", "svh", "mul1"] {
                        remove(&mut inv, &format!("{pre}.{part}"));
                    }
                }
                let w = format!("{pre}.weight");
                inv.tensors.insert(w.clone(), Tensor { name: w, dtype: dtype.into(), shape: vec![n, k], ..t });
            }
            inv
        };
        let plan = LayerPlan::build(&cfg, &dense("F16", false)).unwrap_or_else(|e| panic!("the F16 latents are refused: {e:?}"));
        let u = plan.unit(UnitId::Trunk(1)).unwrap();
        for (pre, n, k) in [(p, cfg.latent, cfg.hidden), (q, cfg.hidden, cfg.latent)] {
            let w = u.tensors.iter().find(|t| t.name == format!("{pre}.weight")).expect("the F16 weight is planned");
            assert_eq!((w.dtype, w.shape.clone()), ("F16", vec![n, k]));
            assert!(!u.linears.iter().any(|l| l.prefix == pre), "{pre} is no longer an EXL3 linear");
            assert!(!u.tensors.iter().any(|t| t.name.starts_with(&format!("{pre}.")) && !t.name.ends_with(".weight")));
        }
        // another layer keeps its EXL3 latents
        let u3 = plan.unit(UnitId::Trunk(3)).unwrap();
        assert!(u3.linears.iter().any(|l| l.prefix.ends_with("layers.3.mixer.fc1_latent_proj")));
        // both forms: refused by name
        let msg = refusal(&cfg, &dense("F16", true));
        assert!(
            msg.contains(&format!("layer 1 (E): `{p}` is in both EXL3 (`{p}.trellis`) and dense (`{p}.weight`) form")),
            "{msg}"
        );
        // a bf16 dense latent: refused by name (the missing trellis, the unclaimed weight)
        let msg = refusal(&cfg, &dense("BF16", false));
        assert!(msg.contains(&format!("layer 1 (E): missing `{p}.trellis`")), "{msg}");
        assert!(msg.contains(&format!("`{p}.weight`: no unit of the plan claims this tensor")), "{msg}");
        // our pack: unchanged
        let inv = expected_inventory(&cfg);
        let ours = LayerPlan::build(&cfg, &inv).unwrap();
        assert!(ours.unit(UnitId::Trunk(1)).unwrap().linears.iter().any(|l| l.prefix == p));
    }

    /// A width no kernel of the engine decodes is refused at plan time by name: a fractional width (the decode kernels take whole widths),
    /// and a last dimension that is no width at all.
    #[test]
    fn a_fractional_or_unknown_exl3_width_is_refused_by_name() {
        let cfg = cfg();
        for (last, says) in
            [(40, "is 2.5 bits per weight: this engine serves whole widths of 1 to 8 bits only"), (36, "a last dimension of 36")]
        {
            let mut inv = expected_inventory(&cfg);
            let name = "language_model.backbone.layers.1.mixer.experts.9.down_proj.trellis";
            inv.tensors.get_mut(name).unwrap().shape[2] = last;
            let msg = refusal(&cfg, &inv);
            assert!(msg.contains(&format!("`{name}`")) && msg.contains(says), "{last}: {msg}");
        }
        let mut inv = expected_inventory(&cfg);
        inv.tensors.get_mut("language_model.backbone.layers.0.mixer.in_proj.trellis").unwrap().shape = vec![256, 1160];
        let msg = refusal(&cfg, &inv);
        assert!(msg.contains("an EXL3 trellis is [K/16, N/16, 16 * bits]"), "{msg}");
    }

    /// The MixedK pack's layout (no `language_model.` on the text names): the same canonical inventory and the same plan as the
    /// prefixed layout, and a refusal names the tensor as the pack's files do.
    #[test]
    fn a_pack_without_the_language_model_prefix_plans_exactly_as_the_prefixed_one() {
        use super::super::inventory::Layout;
        let cfg = cfg();
        let inv = expected_inventory(&cfg);
        let ours = reread(&inv, Layout::Prefixed);
        let his = reread(&inv, Layout::Bare);
        assert_eq!((ours.layout, his.layout), (Layout::Prefixed, Layout::Bare));
        assert_eq!(ours.tensors, inv.tensors, "the prefixed layout is the canonical one");
        assert_eq!(his.tensors, ours.tensors, "the same canonical names, dtypes, shapes and records");
        let a = LayerPlan::build(&cfg, &ours).unwrap_or_else(|e| panic!("{}", e.join("\n")));
        let b = LayerPlan::build(&cfg, &his).unwrap_or_else(|e| panic!("{}", e.join("\n")));
        assert_eq!(a, b);
        assert_eq!(b.outside_plan, 1, "the vision tensor is outside the plan in both layouts");
        // refusals name the file's tensor: a missing one, another dtype, an unclaimed one
        let mut gone = his.clone();
        remove(&mut gone, "language_model.backbone.layers.0.norm.weight");
        // (F32: an F16 gate is planned as F16 since the f32-gate router path, v0.1.2 D1)
        gone.tensors.get_mut("language_model.backbone.layers.1.mixer.gate.weight").unwrap().dtype = "F32".into();
        let extra = Tensor { name: "language_model.backbone.layers.0.mixer.extra.weight".into(), ..inv.tensors[LMH].clone() };
        gone.tensors.insert(extra.name.clone(), extra);
        let msg = refusal(&cfg, &gone);
        assert!(msg.contains("layer 0 (M): missing `backbone.layers.0.norm.weight`"), "{msg}");
        assert!(msg.contains("layer 1 (E): `backbone.layers.1.mixer.gate.weight` is F32 [512, 4096] in the index"), "{msg}");
        assert!(msg.contains("`backbone.layers.0.mixer.extra.weight`: no unit of the plan claims this tensor"), "{msg}");
    }

    const LMH: &str = "language_model.lm_head.weight";

    /// The MixedK pack's layout and forms, synthesized from the plan (the real index is 16 MB; `tests/nemotron_omni_mixedk.rs` reads it when
    /// `REXL3_NEMO_MIXEDK_DIR` names the pack): bare names, K 2 to 8 per projection, an EXL3 head and lm_head, F16 router gates and F16 dense
    /// latent projections. Since lane K's F16 gate and F16 latent paths (lane L's test refused those two forms, its last causes) the whole
    /// plan builds: 41 F16 gates and 80 F16 latents planned as such, and a form no path serves is still refused by its cause, by name.
    #[test]
    fn the_mixedk_forms_plan_in_full_and_an_unserved_form_is_refused_by_its_cause() {
        use super::super::inventory::Layout;
        let cfg = cfg();
        let all: HeadExl3 = head_projections(&cfg).into_iter().map(|(p, _, _)| p).collect();
        let mut widths = mixedk_widths();
        widths.insert(lm_head_prefix(), 8);
        let mut inv = reread(&inventory_of(units_with(&cfg, &Storages::new(), &all, &widths)), Layout::Bare);
        assert_eq!(inv.layout, Layout::Bare);
        for (i, kind) in cfg.pattern.chars().enumerate() {
            if kind != 'E' {
                continue;
            }
            let m = format!("language_model.backbone.layers.{i}.mixer");
            inv.tensors.get_mut(&format!("{m}.gate.weight")).unwrap().dtype = "F16".into();
            for (p, n, k) in [("fc1_latent_proj", cfg.latent, cfg.hidden), ("fc2_latent_proj", cfg.hidden, cfg.latent)] {
                let t = inv.tensors[&format!("{m}.{p}.trellis")].clone();
                for part in ["trellis", "suh", "svh", "mul1"] {
                    remove(&mut inv, &format!("{m}.{p}.{part}"));
                }
                let w = format!("{m}.{p}.weight");
                inv.tensors.insert(w.clone(), Tensor { name: w, dtype: "F16".into(), shape: vec![n, k], ..t });
            }
        }
        inv.tensors.get_mut("language_model.mtp.layers.1.mixer.gate.weight").unwrap().dtype = "F16".into();
        let plan =
            LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("the MixedK plan is refused: {:?}", &e[..e.len().min(5)]));
        let wants: Vec<&Want> = plan.tensors(|_| true);
        let f16 = |suffix: &str| wants.iter().filter(|w| w.dtype == "F16" && w.name.ends_with(suffix)).count();
        assert_eq!(f16(".gate.weight"), 41, "41 F16 router gates");
        assert_eq!(f16("_latent_proj.weight"), 80, "80 F16 latent projections");
        // a dtype no form uses is refused by that cause, by name
        let mut odd = inv.clone();
        odd.tensors.get_mut("language_model.backbone.layers.0.mixer.A_log").unwrap().dtype = "F64".into();
        let msg = refusal(&cfg, &odd);
        assert!(
            msg.contains("layer 0 (M): `backbone.layers.0.mixer.A_log` is F64 [128] in the index, the plan expects BF16 [128]"),
            "{msg}"
        );
    }

    #[test]
    fn an_unclaimed_language_tensor_is_refused_but_a_vision_tensor_is_not() {
        let cfg = cfg();
        let mut inv = expected_inventory(&cfg);
        let extra = Tensor {
            name: "language_model.backbone.layers.0.mixer.extra.weight".into(),
            ..inv.tensors["language_model.backbone.layers.0.norm.weight"].clone()
        };
        inv.tensors.insert(extra.name.clone(), extra);
        let msg = refusal(&cfg, &inv);
        assert!(
            msg.contains("`language_model.backbone.layers.0.mixer.extra.weight`: no unit of the plan claims this tensor"),
            "{msg}"
        );
        let inv = expected_inventory(&cfg);
        assert_eq!(LayerPlan::build(&cfg, &inv).unwrap().outside_plan, 1);
    }

    #[test]
    fn a_tensor_claimed_twice_is_refused_so_no_tensor_is_read_twice() {
        let cfg = cfg();
        let inv = expected_inventory(&cfg);
        let mut units = units(&cfg, &measured_storages());
        let again = units[0].tensors[0].clone();
        units[1].tensors.push(again);
        let errors = LayerPlan::check(units, &inv, Vec::new()).unwrap_err();
        assert_eq!(
            errors,
            vec!["layer 1 (E): `language_model.backbone.layers.0.norm.weight` is claimed twice in the plan".to_string()]
        );
    }

    #[test]
    fn an_expert_in_both_forms_is_refused() {
        let cfg = cfg();
        let mut inv = expected_inventory(&cfg);
        let t = inv.tensors["language_model.backbone.layers.26.mixer.experts.111.up_proj.weight"].clone();
        inv.tensors.insert(
            "language_model.backbone.layers.26.mixer.experts.111.up_proj.trellis".into(),
            Tensor {
                name: "language_model.backbone.layers.26.mixer.experts.111.up_proj.trellis".into(),
                dtype: "I16".into(),
                ..t
            },
        );
        let msg = refusal(&cfg, &inv);
        assert!(msg.contains("layer 26 (E): expert 111 `up_proj` is in both EXL3"), "{msg}");
    }

    #[test]
    fn the_trimmed_fixture_is_refused_at_layer_zero_by_name() {
        let cfg = cfg();
        let idx = Index::from_json(TRIMMED).unwrap();
        let tensors = idx
            .weight_map
            .iter()
            .map(|(name, shard)| {
                let t = Tensor {
                    name: name.clone(),
                    shard: shard.clone(),
                    dtype: "I16".into(),
                    shape: vec![64, 168, 64],
                    offset: 0,
                    nbytes: 1,
                };
                (name.clone(), t)
            })
            .collect();
        let inv = Inventory { tensors, shards: idx.shards(), strays: vec![], index_total_size: None, ..Default::default() };
        let errors = LayerPlan::build(&cfg, &inv).unwrap_err();
        assert_eq!(errors[0], "layer 0 (M): missing `language_model.backbone.layers.0.norm.weight`");
        assert!(
            errors.iter().any(|e| e == "layer 1 (E): missing `language_model.backbone.layers.1.mixer.experts.0.up_proj.trellis`"),
            "{errors:?}"
        );
    }

    #[test]
    fn the_ssm_fixture_header_reads_and_agrees_with_the_mamba_plan() {
        let cfg = cfg();
        let h = read_shard_header(Path::new(SSM_FIXTURE)).unwrap();
        assert_eq!(h.file_len, 753_352);
        let inner = cfg.mamba_heads * cfg.mamba_dim;
        let conv_dim = inner + 2 * cfg.groups * cfg.state;
        let want: [(&str, &str, Vec<usize>); 6] = [
            ("A_log", "BF16", vec![cfg.mamba_heads]),
            ("D", "BF16", vec![cfg.mamba_heads]),
            ("dt_bias", "BF16", vec![cfg.mamba_heads]),
            ("conv1d.bias", "BF16", vec![conv_dim]),
            ("conv1d.weight", "BF16", vec![conv_dim, 1, cfg.conv]),
            ("norm.weight", "BF16", vec![inner]),
        ];
        for (name, dtype, shape) in want {
            let e = &h.entries[name];
            assert_eq!((e.dtype.as_str(), e.shape.clone()), (dtype, shape), "{name}");
        }
        // data starts after the 8-byte length and the 1,960-byte header; A_log's data is the first record at 631,832
        assert_eq!(h.entries["A_log"].offset, 8 + 1960 + 631_832);
        assert_eq!(h.entries["A_log"].nbytes, 256);
    }

    #[test]
    fn a_header_whose_record_disagrees_with_its_shape_or_extent_is_refused_by_name() {
        let dir = std::env::temp_dir().join(format!("rexl3-omni-header-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |file: &str, header: serde_json::Value, data: &[u8]| {
            let text = header.to_string();
            let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(text.as_bytes());
            bytes.extend_from_slice(data);
            let path = dir.join(file);
            std::fs::write(&path, bytes).unwrap();
            path
        };
        let ok = write(
            "ok.safetensors",
            serde_json::json!({"good": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}),
            &[0u8; 8],
        );
        assert_eq!(read_shard_header(&ok).unwrap().entries["good"].nbytes, 8);
        let short = write(
            "short.safetensors",
            serde_json::json!({"bad_size": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 4]}}),
            &[0u8; 8],
        );
        let err = read_shard_header(&short).unwrap_err();
        assert!(err.contains("`bad_size`") && err.contains("needs 8 bytes"), "{err}");
        let past = write(
            "past.safetensors",
            serde_json::json!({"past_end": {"dtype": "F32", "shape": [1], "data_offsets": [0, 8]}}),
            &[0u8; 4],
        );
        let err = read_shard_header(&past).unwrap_err();
        assert!(err.contains("`past_end`") && err.contains("outside"), "{err}");
        let unknown = write(
            "dtype.safetensors",
            serde_json::json!({"odd": {"dtype": "C64", "shape": [1], "data_offsets": [0, 8]}}),
            &[0u8; 8],
        );
        assert!(read_shard_header(&unknown).unwrap_err().contains("unknown dtype C64"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
