//! The MiniCPM-V-4.7 weights on the device, claimed slot by slot from the pack's [`WeightMap`].
//!
//! [`MiniCpmWeights::load`] binds the recipe (a pack that disagrees with its own config is refused by name before a byte is read),
//! stages every uploaded slot through the pipelined loader (`LoadCtx`: one arena of 1 GiB blocks, one pass over the shards, the
//! conversions of [`Transform`] on the loader's threads) and claims the staged buffers by checkpoint name into the typed structs the
//! forward runs on. Nothing is loaded by convention: a slot the recipe wants and the map does not hold is an error naming it, and a
//! slot the map holds and no struct claims is reported too, so a pack cannot quietly carry a tensor the forward ignores.
//!
//! What the structs hold (all on the device after [`MiniCpmWeights::load`] returns):
//!
//! * per layer: the pre-mixer and post-attention norms as fp32 `(1 + w)`; a gated delta net mixer (`in_proj_qkv`, `in_proj_z`,
//!   `out_proj` EXL3; `in_proj_a`, `in_proj_b` fp16 `[32, 2048]`; the bf16 convolution `[8192, 1, 4]`; `A_log`, `dt_bias` and the
//!   gated norm as fp32) or an attention mixer (`q_proj` one EXL3 group K 2,048 -> N 8,192 whose rows are, per head, 256 query then
//!   256 gate values, split by `attn_prep_mrope`; `k_proj`, `v_proj`, `o_proj`; `q_norm`, `k_norm` as fp32 `(1 + w)`);
//! * the MoE of every layer: the router (fp16 `[256, 2048]`), the shared expert's sigmoid gate (fp16 `[1, 2048]`), the shared
//!   expert and the 256 routed experts, each three EXL3 groups, and the device pointer tables `fmoe::forward` indexes by expert id
//!   (the shared expert is the entry past the routed ones) with the per-matrix bit widths;
//! * the embedding (bf16 `[248144, 2048]`), the final norm (fp32 `(1 + w)`) and the head, one EXL3 group at N 248,192 whose first
//!   [`MiniCpmWeights::vocab`] columns are the vocabulary: every consumer of the logits masks the rest.
//!
//! The tower and the projector are recorded in [`MiniCpmWeights::tower`] (their count and bytes) and not uploaded.

use super::config::MiniCpmConfig;
use super::weights::{Kind, Slot, Transform, WeightMap};
use crate::error::ModelError;
use crate::loadctx::LoadCtx;
use crate::pack::Pack;
use gb_cuda::exl3_linear::Exl3Weights;
use gb_cuda::fn_moe::{ktab_k2, ExpertTables};
use gb_cuda::loader::Report;
use gb_cuda::DevBuf;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Prefix of every trunk tensor of the text model.
pub const LM: &str = "model.language_model";

/// A gated delta net mixer.
pub struct LinearAttnWeights {
    pub in_proj_qkv: Exl3Weights,
    pub in_proj_z: Exl3Weights,
    pub out_proj: Exl3Weights,
    /// fp16 `[32, 2048]`.
    pub in_proj_a: DevBuf,
    /// fp16 `[32, 2048]`.
    pub in_proj_b: DevBuf,
    /// bf16 `[8192, 1, 4]`.
    pub conv_w: DevBuf,
    /// fp32 `[32]`.
    pub a_log: DevBuf,
    /// fp32 `[32]`.
    pub dt_bias: DevBuf,
    /// fp32 `[128]`, the gated RMSNorm's weight as stored (no `+1`).
    pub norm: DevBuf,
}

/// A grouped-query attention mixer with its fused output gate.
pub struct FullAttnWeights {
    /// K 2,048 -> N 8,192: per head 256 query rows then 256 gate rows.
    pub q_proj: Exl3Weights,
    pub k_proj: Exl3Weights,
    pub v_proj: Exl3Weights,
    pub o_proj: Exl3Weights,
    /// fp32 `[256]`, `1 + w`.
    pub q_norm: DevBuf,
    /// fp32 `[256]`, `1 + w`.
    pub k_norm: DevBuf,
}

pub enum MixerWeights {
    Linear(Box<LinearAttnWeights>),
    Full(Box<FullAttnWeights>),
}

pub struct ExpertWeights {
    pub gate: Exl3Weights,
    pub up: Exl3Weights,
    pub down: Exl3Weights,
}

impl ExpertWeights {
    fn bytes(&self) -> usize {
        self.gate.weight_bytes() + self.up.weight_bytes() + self.down.weight_bytes()
    }
}

/// One layer's MoE: the router, the shared expert's sigmoid gate, the shared expert, the routed experts and the device pointer
/// tables the MoE kernels index by expert id (the shared expert is slot `experts`).
pub struct MoeWeights {
    /// fp16 `[experts, hidden]`.
    pub router: DevBuf,
    /// fp16 `[1, hidden]`.
    pub shared_gate: DevBuf,
    pub shared_expert: Box<ExpertWeights>,
    pub experts: Vec<ExpertWeights>,
    pub g_trellis_ptrs: DevBuf,
    pub g_suh_ptrs: DevBuf,
    pub g_svh_ptrs: DevBuf,
    pub u_trellis_ptrs: DevBuf,
    pub u_suh_ptrs: DevBuf,
    pub u_svh_ptrs: DevBuf,
    pub d_trellis_ptrs: DevBuf,
    pub d_suh_ptrs: DevBuf,
    pub d_svh_ptrs: DevBuf,
    /// i32 `[experts + 1][3]`: the bit width of gate, up and down.
    pub k_tab: DevBuf,
}

impl MoeWeights {
    pub fn tables(&self) -> ExpertTables<'_> {
        ExpertTables {
            g_t: &self.g_trellis_ptrs,
            g_su: &self.g_suh_ptrs,
            g_sv: &self.g_svh_ptrs,
            u_t: &self.u_trellis_ptrs,
            u_su: &self.u_suh_ptrs,
            u_sv: &self.u_svh_ptrs,
            d_t: &self.d_trellis_ptrs,
            d_su: &self.d_suh_ptrs,
            d_sv: &self.d_svh_ptrs,
            ktab: &self.k_tab,
        }
    }

    fn table_bytes(&self) -> usize {
        9 * self.g_trellis_ptrs.bytes() + self.k_tab.bytes()
    }
}

pub struct MiniCpmLayer {
    /// fp32 `[2048]`, `1 + w`: the norm in front of the mixer.
    pub input_norm: DevBuf,
    /// fp32 `[2048]`, `1 + w`: the norm in front of the MoE.
    pub post_norm: DevBuf,
    pub mixer: MixerWeights,
    pub moe: MoeWeights,
}

/// The tower and the projector as the pack carries them: present and counted, not on the device (the vision slice loads them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TowerRecord {
    pub tensors: usize,
    pub bytes: u64,
}

pub struct MiniCpmWeights {
    /// bf16 `[vocab, hidden]`.
    pub embed: DevBuf,
    pub layers: Vec<MiniCpmLayer>,
    /// fp32 `[hidden]`, `1 + w`.
    pub final_norm: DevBuf,
    /// One EXL3 group, K 2,048 -> N [`MiniCpmWeights::lm_head_width`].
    pub lm_head: Exl3Weights,
    /// Real vocabulary ids: the head's columns `vocab..lm_head_width` are padding.
    pub vocab: usize,
    /// The head's output width (the padded vocabulary).
    pub lm_head_width: usize,
    pub tower: TowerRecord,
    /// Device bytes of every weight buffer above (the pointer tables included, the arena's alignment padding not).
    pub bytes: usize,
}

/// What loading the weights moved and held.
#[derive(Clone, Debug)]
pub struct LoadStats {
    /// The pipelined loader's report (`None` for the serial reference loader).
    pub report: Option<Report>,
    /// Bytes the loader's arena carved (every item rounded up to the arena's alignment): [`WeightMap::upload_bytes`] exactly.
    pub arena_used: u64,
    /// Bytes `cudaMalloc`ed for the arena's blocks.
    pub arena_reserved: u64,
    pub arena_blocks: usize,
    /// Device bytes of the 10 pointer tables a layer, a tensor each.
    pub table_bytes: u64,
}

/// The device addresses of one EXL3 linear's trellis, input scales and output scales, which the MoE kernels read out of a per-expert
/// pointer table. [`Exl3Weights::buffers`] names them in that fixed order.
fn ptrs_of(w: &Exl3Weights, what: &str) -> Result<[u64; 3], ModelError> {
    let b = w.buffers();
    match (b.first(), b.get(1), b.get(2)) {
        (Some(("trellis", t)), Some(("suh", su)), Some(("svh", sv))) => Ok([t.ptr() as u64, su.ptr() as u64, sv.ptr() as u64]),
        _ => Err(ModelError::Config(format!("{what}: an EXL3 linear without the three parts the MoE kernels read"))),
    }
}

/// A bound map's slots on the device, claimed by name.
struct Staged {
    plain: BTreeMap<String, DevBuf>,
    groups: BTreeMap<String, Exl3Weights>,
}

impl Staged {
    fn take(&mut self, name: &str) -> Result<DevBuf, ModelError> {
        self.plain.remove(name).ok_or_else(|| ModelError::Config(format!("{name}: the pack has no such weight")))
    }
    fn take_group(&mut self, prefix: &str) -> Result<Exl3Weights, ModelError> {
        self.groups.remove(prefix).ok_or_else(|| ModelError::Config(format!("{prefix}: the pack has no such EXL3 linear")))
    }
    /// Slots no struct claimed: a pack the forward would only partly run.
    fn leftovers(&self) -> Vec<String> {
        self.plain.keys().chain(self.groups.keys()).cloned().collect()
    }
}

/// The slots [`stage`] puts on the device, in source order: every `Plain` and `Exl3Part` slot of the load plan. The markers are
/// checked by the scan and the tower is recorded, so neither is among them.
fn upload_plan(map: &WeightMap) -> Vec<Slot> {
    map.load_plan().items
}

/// Stage every `Plain` and `Exl3Part` slot of `map` on the device, in source order.
fn stage<'p>(ctx: &mut LoadCtx<'p>, pack: &'p Pack, map: &WeightMap) -> Result<Staged, ModelError> {
    let mut out = Staged { plain: BTreeMap::new(), groups: BTreeMap::new() };
    for slot in upload_plan(map).iter() {
        match slot.kind {
            Kind::Plain => {
                let dtype = match slot.dtype.as_str() {
                    "BF16" => "BF16",
                    "F16" => "F16",
                    other => return Err(ModelError::Config(format!("{}: dtype {other} is not a weight", slot.name))),
                };
                let buf = match slot.transform {
                    Transform::None => ctx.raw16(pack, &slot.name, dtype, &slot.shape)?,
                    Transform::Bf16ToF32 => ctx.f32_from_bf16(pack, &slot.name, &slot.shape, false)?,
                    Transform::Bf16AddOneToF32 => ctx.f32_from_bf16(pack, &slot.name, &slot.shape, true)?,
                    Transform::Bf16ToF16 => ctx.f16_from_bf16(pack, &slot.name, &slot.shape)?,
                };
                out.plain.insert(slot.name.clone(), buf);
            }
            Kind::Exl3Part => {
                let Some((prefix, _)) = slot.name.rsplit_once('.') else {
                    return Err(ModelError::Config(format!("{}: not an EXL3 part name", slot.name)));
                };
                if !out.groups.contains_key(prefix) {
                    if !map.groups.contains_key(prefix) {
                        return Err(ModelError::Config(format!("{prefix}: the bound map has no EXL3 group")));
                    }
                    let w = ctx.group(pack, prefix)?;
                    out.groups.insert(prefix.to_string(), w);
                }
            }
            Kind::Marker | Kind::Recorded => {
                return Err(ModelError::Config(format!("{}: a {:?} slot in the upload plan", slot.name, slot.kind)))
            }
        }
    }
    Ok(out)
}

fn claim_layer(s: &mut Staged, cfg: &MiniCpmConfig, p: &str, full: bool, bytes: &mut usize) -> Result<MiniCpmLayer, ModelError> {
    let mut b = 0usize;
    let input_norm = s.take(&format!("{p}.input_layernorm.weight"))?;
    let post_norm = s.take(&format!("{p}.post_attention_layernorm.weight"))?;
    b += input_norm.bytes() + post_norm.bytes();

    let mixer = if full {
        let ap = format!("{p}.self_attn");
        let q_proj = s.take_group(&format!("{ap}.q_proj"))?;
        let k_proj = s.take_group(&format!("{ap}.k_proj"))?;
        let v_proj = s.take_group(&format!("{ap}.v_proj"))?;
        let o_proj = s.take_group(&format!("{ap}.o_proj"))?;
        let q_norm = s.take(&format!("{ap}.q_norm.weight"))?;
        let k_norm = s.take(&format!("{ap}.k_norm.weight"))?;
        b += q_proj.weight_bytes() + k_proj.weight_bytes() + v_proj.weight_bytes() + o_proj.weight_bytes();
        b += q_norm.bytes() + k_norm.bytes();
        MixerWeights::Full(Box::new(FullAttnWeights { q_proj, k_proj, v_proj, o_proj, q_norm, k_norm }))
    } else {
        let ap = format!("{p}.linear_attn");
        let in_proj_qkv = s.take_group(&format!("{ap}.in_proj_qkv"))?;
        let in_proj_z = s.take_group(&format!("{ap}.in_proj_z"))?;
        let out_proj = s.take_group(&format!("{ap}.out_proj"))?;
        let in_proj_a = s.take(&format!("{ap}.in_proj_a.weight"))?;
        let in_proj_b = s.take(&format!("{ap}.in_proj_b.weight"))?;
        let conv_w = s.take(&format!("{ap}.conv1d.weight"))?;
        let a_log = s.take(&format!("{ap}.A_log"))?;
        let dt_bias = s.take(&format!("{ap}.dt_bias"))?;
        let norm = s.take(&format!("{ap}.norm.weight"))?;
        b += in_proj_qkv.weight_bytes() + in_proj_z.weight_bytes() + out_proj.weight_bytes();
        b += in_proj_a.bytes() + in_proj_b.bytes() + conv_w.bytes() + a_log.bytes() + dt_bias.bytes() + norm.bytes();
        MixerWeights::Linear(Box::new(LinearAttnWeights {
            in_proj_qkv,
            in_proj_z,
            out_proj,
            in_proj_a,
            in_proj_b,
            conv_w,
            a_log,
            dt_bias,
            norm,
        }))
    };

    let mp = format!("{p}.mlp");
    let router = s.take(&format!("{mp}.gate.weight"))?;
    let shared_gate = s.take(&format!("{mp}.shared_expert_gate.weight"))?;
    b += router.bytes() + shared_gate.bytes();
    let shared_expert = Box::new(ExpertWeights {
        gate: s.take_group(&format!("{mp}.shared_expert.gate_proj"))?,
        up: s.take_group(&format!("{mp}.shared_expert.up_proj"))?,
        down: s.take_group(&format!("{mp}.shared_expert.down_proj"))?,
    });
    b += shared_expert.bytes();
    let mut experts = Vec::with_capacity(cfg.experts);
    for e in 0..cfg.experts {
        let ep = format!("{mp}.experts.{e}");
        let w = ExpertWeights {
            gate: s.take_group(&format!("{ep}.gate_proj"))?,
            up: s.take_group(&format!("{ep}.up_proj"))?,
            down: s.take_group(&format!("{ep}.down_proj"))?,
        };
        b += w.bytes();
        experts.push(w);
    }

    let n = cfg.experts + 1;
    let moe = MoeWeights {
        router,
        shared_gate,
        shared_expert,
        experts,
        g_trellis_ptrs: DevBuf::zeroed(n * 8)?,
        g_suh_ptrs: DevBuf::zeroed(n * 8)?,
        g_svh_ptrs: DevBuf::zeroed(n * 8)?,
        u_trellis_ptrs: DevBuf::zeroed(n * 8)?,
        u_suh_ptrs: DevBuf::zeroed(n * 8)?,
        u_svh_ptrs: DevBuf::zeroed(n * 8)?,
        d_trellis_ptrs: DevBuf::zeroed(n * 8)?,
        d_suh_ptrs: DevBuf::zeroed(n * 8)?,
        d_svh_ptrs: DevBuf::zeroed(n * 8)?,
        k_tab: DevBuf::zeroed(n * 3 * 4)?,
    };
    b += moe.table_bytes();
    *bytes += b;
    Ok(MiniCpmLayer { input_norm, post_norm, mixer, moe })
}

impl MiniCpmWeights {
    /// Scan, bind, upload and claim the text model of the pack in `dir`. `progress(done, total)` counts trunk layers. The
    /// convenience path of the device tests; the boot (`super::boot`) scans once and calls [`MiniCpmWeights::load_from`].
    pub fn load(dir: &Path, cfg: &MiniCpmConfig, progress: impl FnMut(usize, usize)) -> Result<(Self, LoadStats), ModelError> {
        let pack = Pack::open(dir)?;
        let map = WeightMap::from_checkpoint(&pack.ck, cfg).map_err(|errs| {
            let head: Vec<&str> = errs.iter().take(8).map(String::as_str).collect();
            ModelError::Config(format!("MiniCPM-V-4.7 pack: {} problems: {}", errs.len(), head.join("; ")))
        })?;
        Self::load_from(&pack, &map, cfg, progress)
    }

    /// Upload and claim a pack already scanned and bound.
    pub fn load_from(
        pack: &Pack,
        map: &WeightMap,
        cfg: &MiniCpmConfig,
        progress: impl FnMut(usize, usize),
    ) -> Result<(Self, LoadStats), ModelError> {
        let mut ctx = LoadCtx::from_env();
        let mut staged = stage(&mut ctx, pack, map)?;
        let mut w = Self::claim(&mut staged, cfg, map, progress)?;
        let (arena_used, arena_reserved, arena_blocks) = ctx.arena_stats();
        // the loader's copies run here; every buffer claimed above holds its weight only afterwards
        let report = ctx.finish()?;
        w.init_ptr_tables()?;
        let table_bytes = w.layers.iter().map(|l| l.moe.table_bytes() as u64).sum();
        Ok((w, LoadStats { report, arena_used, arena_reserved, arena_blocks, table_bytes }))
    }

    /// Claim every staged slot into the typed structs. Host work only: no copy has run yet.
    fn claim(
        s: &mut Staged,
        cfg: &MiniCpmConfig,
        map: &WeightMap,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Self, ModelError> {
        let mut bytes = 0usize;
        let embed = s.take(&format!("{LM}.embed_tokens.weight"))?;
        bytes += embed.bytes();

        let mut layers = Vec::with_capacity(cfg.layers);
        for (i, t) in cfg.layer_types.iter().enumerate() {
            layers.push(claim_layer(s, cfg, &format!("{LM}.layers.{i}"), t == "full_attention", &mut bytes)?);
            progress(i + 1, cfg.layers);
        }

        let final_norm = s.take(&format!("{LM}.norm.weight"))?;
        bytes += final_norm.bytes();
        let lm_head = s.take_group("lm_head")?;
        bytes += lm_head.weight_bytes();
        if lm_head.n != map.lm_head_width || lm_head.k != cfg.hidden {
            return Err(ModelError::Config(format!(
                "lm_head: {} -> {} (the recipe needs {} -> {})",
                lm_head.k, lm_head.n, cfg.hidden, map.lm_head_width
            )));
        }

        // a weight the map bound and nothing above claimed would be one the forward silently ignores
        let left = s.leftovers();
        if !left.is_empty() {
            let head: Vec<&str> = left.iter().take(8).map(String::as_str).collect();
            return Err(ModelError::Config(format!(
                "{} bound weights no MiniCPM-V-4.7 slot claims: {}",
                left.len(),
                head.join("; ")
            )));
        }
        let plan = map.load_plan();
        let tower = TowerRecord { tensors: plan.recorded.len(), bytes: plan.recorded_bytes() };
        Ok(Self { embed, layers, final_norm, lm_head, vocab: map.vocab, lm_head_width: map.lm_head_width, tower, bytes })
    }

    /// Fill the per-layer expert pointer and bit-width tables the MoE kernels index by expert id (the shared expert is the entry
    /// past the routed ones).
    pub fn init_ptr_tables(&mut self) -> Result<(), ModelError> {
        for layer in self.layers.iter_mut() {
            let m = &mut layer.moe;
            let n = m.experts.len() + 1;
            let mut t: Vec<Vec<u64>> = (0..9).map(|_| Vec::with_capacity(n)).collect();
            let mut kt: Vec<i32> = Vec::with_capacity(n * 3);
            for (slot, e) in m.experts.iter().chain(std::iter::once(&*m.shared_expert)).enumerate() {
                for (j, w) in [&e.gate, &e.up, &e.down].into_iter().enumerate() {
                    let [tr, su, sv] = ptrs_of(w, &format!("expert slot {slot} projection {j}"))?;
                    t[3 * j].push(tr);
                    t[3 * j + 1].push(su);
                    t[3 * j + 2].push(sv);
                    kt.push(ktab_k2(w.bits));
                }
            }
            for (buf, v) in [
                (&mut m.g_trellis_ptrs, 0),
                (&mut m.g_suh_ptrs, 1),
                (&mut m.g_svh_ptrs, 2),
                (&mut m.u_trellis_ptrs, 3),
                (&mut m.u_suh_ptrs, 4),
                (&mut m.u_svh_ptrs, 5),
                (&mut m.d_trellis_ptrs, 6),
                (&mut m.d_suh_ptrs, 7),
                (&mut m.d_svh_ptrs, 8),
            ] {
                buf.upload(&t[v])?;
            }
            m.k_tab.upload(&kt)?;
        }
        Ok(())
    }

    /// Every EXL3 linear the forward runs, as `(name, K, N, bits)`: the shapes the boot's kernel gates are checked over.
    pub fn linears(&self) -> Vec<(String, usize, usize, usize)> {
        let mut v = Vec::new();
        let mut add = |name: String, w: &Exl3Weights| v.push((name, w.k, w.n, w.bits));
        for (i, l) in self.layers.iter().enumerate() {
            match &l.mixer {
                MixerWeights::Linear(m) => {
                    add(format!("layers.{i}.linear_attn.in_proj_qkv"), &m.in_proj_qkv);
                    add(format!("layers.{i}.linear_attn.in_proj_z"), &m.in_proj_z);
                    add(format!("layers.{i}.linear_attn.out_proj"), &m.out_proj);
                }
                MixerWeights::Full(m) => {
                    add(format!("layers.{i}.self_attn.q_proj"), &m.q_proj);
                    add(format!("layers.{i}.self_attn.k_proj"), &m.k_proj);
                    add(format!("layers.{i}.self_attn.v_proj"), &m.v_proj);
                    add(format!("layers.{i}.self_attn.o_proj"), &m.o_proj);
                }
            }
            let sh = &l.moe.shared_expert;
            add(format!("layers.{i}.mlp.shared_expert.gate_proj"), &sh.gate);
            add(format!("layers.{i}.mlp.shared_expert.up_proj"), &sh.up);
            add(format!("layers.{i}.mlp.shared_expert.down_proj"), &sh.down);
            for (e, x) in l.moe.experts.iter().enumerate() {
                add(format!("layers.{i}.mlp.experts.{e}.gate_proj"), &x.gate);
                add(format!("layers.{i}.mlp.experts.{e}.up_proj"), &x.up);
                add(format!("layers.{i}.mlp.experts.{e}.down_proj"), &x.down);
            }
        }
        add("lm_head".into(), &self.lm_head);
        v
    }

    /// Every uploaded buffer under its checkpoint name (`<group>.trellis`, `<group>.suh`, `<group>.svh` for an EXL3 group): exactly
    /// the names of the load plan's items, which is what lets a test compare the device with the shards tensor by tensor and
    /// prove that no slot is missing from the structs.
    pub fn named_buffers(&self) -> Vec<(String, &DevBuf)> {
        let mut v: Vec<(String, &DevBuf)> = Vec::new();
        fn group<'a>(v: &mut Vec<(String, &'a DevBuf)>, prefix: String, w: &'a Exl3Weights) {
            for (part, buf) in w.buffers() {
                v.push((format!("{prefix}.{part}"), buf));
            }
        }
        v.push((format!("{LM}.embed_tokens.weight"), &self.embed));
        v.push((format!("{LM}.norm.weight"), &self.final_norm));
        group(&mut v, "lm_head".into(), &self.lm_head);
        for (i, l) in self.layers.iter().enumerate() {
            let p = format!("{LM}.layers.{i}");
            v.push((format!("{p}.input_layernorm.weight"), &l.input_norm));
            v.push((format!("{p}.post_attention_layernorm.weight"), &l.post_norm));
            match &l.mixer {
                MixerWeights::Linear(m) => {
                    group(&mut v, format!("{p}.linear_attn.in_proj_qkv"), &m.in_proj_qkv);
                    group(&mut v, format!("{p}.linear_attn.in_proj_z"), &m.in_proj_z);
                    group(&mut v, format!("{p}.linear_attn.out_proj"), &m.out_proj);
                    v.push((format!("{p}.linear_attn.in_proj_a.weight"), &m.in_proj_a));
                    v.push((format!("{p}.linear_attn.in_proj_b.weight"), &m.in_proj_b));
                    v.push((format!("{p}.linear_attn.conv1d.weight"), &m.conv_w));
                    v.push((format!("{p}.linear_attn.A_log"), &m.a_log));
                    v.push((format!("{p}.linear_attn.dt_bias"), &m.dt_bias));
                    v.push((format!("{p}.linear_attn.norm.weight"), &m.norm));
                }
                MixerWeights::Full(m) => {
                    group(&mut v, format!("{p}.self_attn.q_proj"), &m.q_proj);
                    group(&mut v, format!("{p}.self_attn.k_proj"), &m.k_proj);
                    group(&mut v, format!("{p}.self_attn.v_proj"), &m.v_proj);
                    group(&mut v, format!("{p}.self_attn.o_proj"), &m.o_proj);
                    v.push((format!("{p}.self_attn.q_norm.weight"), &m.q_norm));
                    v.push((format!("{p}.self_attn.k_norm.weight"), &m.k_norm));
                }
            }
            v.push((format!("{p}.mlp.gate.weight"), &l.moe.router));
            v.push((format!("{p}.mlp.shared_expert_gate.weight"), &l.moe.shared_gate));
            let sh = &l.moe.shared_expert;
            group(&mut v, format!("{p}.mlp.shared_expert.gate_proj"), &sh.gate);
            group(&mut v, format!("{p}.mlp.shared_expert.up_proj"), &sh.up);
            group(&mut v, format!("{p}.mlp.shared_expert.down_proj"), &sh.down);
            for (e, x) in l.moe.experts.iter().enumerate() {
                group(&mut v, format!("{p}.mlp.experts.{e}.gate_proj"), &x.gate);
                group(&mut v, format!("{p}.mlp.experts.{e}.up_proj"), &x.up);
                group(&mut v, format!("{p}.mlp.experts.{e}.down_proj"), &x.down);
            }
        }
        v
    }

    /// The distinct (K, N) shapes of the model's EXL3 linears, with the bit widths each runs at.
    pub fn linear_shapes(&self) -> BTreeMap<(usize, usize), BTreeSet<usize>> {
        let mut m: BTreeMap<(usize, usize), BTreeSet<usize>> = BTreeMap::new();
        for (_, k, n, bits) in self.linears() {
            m.entry((k, n)).or_default().insert(bits);
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::real;
    use super::super::inventory;
    use super::*;

    const TABLE: &str = include_str!("../../../../../testdata/minicpmv4_7/tensor_inventory.tsv");

    fn map() -> WeightMap {
        WeightMap::bind(&real(), &inventory::expand_table(TABLE).unwrap()).unwrap()
    }

    /// The names the claim takes, written out independently of the claim: plain tensors, and EXL3 groups by prefix.
    fn claimed_by_recipe(cfg: &MiniCpmConfig) -> (BTreeSet<String>, BTreeSet<String>) {
        let (mut plain, mut groups) = (BTreeSet::new(), BTreeSet::new());
        plain.insert(format!("{LM}.embed_tokens.weight"));
        plain.insert(format!("{LM}.norm.weight"));
        groups.insert("lm_head".to_string());
        for (i, t) in cfg.layer_types.iter().enumerate() {
            let p = format!("{LM}.layers.{i}");
            plain.insert(format!("{p}.input_layernorm.weight"));
            plain.insert(format!("{p}.post_attention_layernorm.weight"));
            if t == "full_attention" {
                for g in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    groups.insert(format!("{p}.self_attn.{g}"));
                }
                for n in ["q_norm", "k_norm"] {
                    plain.insert(format!("{p}.self_attn.{n}.weight"));
                }
            } else {
                for g in ["in_proj_qkv", "in_proj_z", "out_proj"] {
                    groups.insert(format!("{p}.linear_attn.{g}"));
                }
                for n in ["in_proj_a.weight", "in_proj_b.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight"] {
                    plain.insert(format!("{p}.linear_attn.{n}"));
                }
            }
            plain.insert(format!("{p}.mlp.gate.weight"));
            plain.insert(format!("{p}.mlp.shared_expert_gate.weight"));
            for g in ["gate_proj", "up_proj", "down_proj"] {
                groups.insert(format!("{p}.mlp.shared_expert.{g}"));
                for e in 0..cfg.experts {
                    groups.insert(format!("{p}.mlp.experts.{e}.{g}"));
                }
            }
        }
        (plain, groups)
    }

    /// What the device stages is exactly what the claim takes: every plain tensor (in a dtype the stager converts), every EXL3
    /// part under a bound group, no marker and no tower tensor, so no staged buffer is left unclaimed and none is claimed twice.
    #[test]
    fn the_upload_plan_is_exactly_what_the_claim_takes() {
        let (cfg, map) = (real(), map());
        let items = upload_plan(&map);
        let (want_plain, want_groups) = claimed_by_recipe(&cfg);
        let mut plain = BTreeSet::new();
        let mut groups: BTreeMap<String, usize> = BTreeMap::new();
        for s in &items {
            match s.kind {
                Kind::Plain => {
                    assert!(matches!(s.dtype.as_str(), "BF16" | "F16"), "{}: {}", s.name, s.dtype);
                    assert!(plain.insert(s.name.clone()), "{} staged twice", s.name);
                }
                Kind::Exl3Part => {
                    let (prefix, part) = s.name.rsplit_once('.').unwrap();
                    assert!(map.groups.contains_key(prefix), "{}: no bound group", s.name);
                    assert!(matches!(part, "trellis" | "suh" | "svh"), "{}", s.name);
                    *groups.entry(prefix.to_string()).or_default() += 1;
                }
                Kind::Marker | Kind::Recorded => panic!("{}: a {:?} slot in the upload plan", s.name, s.kind),
            }
        }
        assert_eq!(plain, want_plain);
        assert_eq!(groups.keys().cloned().collect::<BTreeSet<_>>(), want_groups);
        assert!(groups.values().all(|&n| n == 3));
        // the tower never reaches the device
        assert!(items.iter().all(|s| !s.name.starts_with("model.vision_tower.") && !s.name.starts_with("model.merger.")));
        assert_eq!(want_groups.len(), 30_971);
        assert_eq!(want_plain.len(), 362);
    }

    #[test]
    fn the_eight_layer_claim_lists_its_own_layers() {
        let mut cfg = real();
        cfg.layers = 8;
        cfg.layer_types.truncate(8);
        let (plain, groups) = claimed_by_recipe(&cfg);
        assert_eq!(groups.len(), 6 * 774 + 2 * 775 + 1);
        assert_eq!(plain.len(), 6 * 10 + 2 * 6 + 2);
        assert!(plain.contains(&format!("{LM}.layers.7.self_attn.k_norm.weight")));
        assert!(!plain.iter().any(|n| n.contains("layers.8.")));
    }
}
