//! The bridge from the loader's upload ([`super::weights::Weights`]) to the device layers ([`super::device`]). Wave 1 task B1.
//! The signature of [`into_device`] is frozen.
//!
//! Two passes. The first only looks: [`check_presence`] asks the upload for every tensor the plan names for the requested layers and
//! the globals (by name, dtype and shape), and [`check_codebooks`] reads each EXL3 linear's `mul1` marker from the device. A failure
//! there names the unit and the tensor and leaves the upload untouched. The second pass moves: buffers are not copied (the trellis and
//! every bf16 tensor keep their place in the upload's arena), except the vectors the kernels read as f32, which are widened once
//! into a small arena of their own: `A_log`, `D`, `dt_bias` and the gated-norm weight of each Mamba layer, and `suh` and `svh` of
//! every EXL3 expert projection and latent or shared projection of the MoE blocks (the pack holds them as fp16).
//!
//! The widening goes through one private function ([`widen`]). Today it is a host round trip; `nemotron_glue::widen_bf16_f32` and
//! `widen_f16_f32` produce the same bits on the device, so swapping them in changes the body of that function and nothing else.

use super::config::OmniConfig;
use super::device::{AttnW, HeadLinear, Layer, MambaW, Mixer, MtpW, NemotronWeights};
use super::tp_shard::ShardPlan;
use super::weights::{LayerPlan, Unit, UnitId, Weights};
use crate::error::ModelError;
use gb_cuda::arena::Arena;
use gb_cuda::nemotron_glue::{widen_bf16_f32, widen_f16_f32};
use gb_cuda::nemotron_moe::{self, GateDtype, Proj};
use gb_cuda::nemotron_moe_prefill::PrefillMoe;
use gb_cuda::{CudaError, DevBuf, Stream};
use gb_exl3::{Codebook, MARKER_MUL1};
use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

/// Bytes per block of the arena that holds the widened vectors (each one is a few KiB; the blocks are shared by the layers).
const WIDEN_BLOCK_BYTES: usize = 16 << 20;

/// Moves trunk layers `layers` and the globals out of `w` into device layers.
///
/// - EXL3 linears of the Mamba and attention layers move with `Weights::take_linear` (their buffers are not copied).
/// - `A_log`, `D` and `dt_bias`, and the Mamba gated-norm weight, are widened bf16 to f32. The routed experts, the latent projections
///   and the shared expert of an `E` layer keep their trellis in place and take f32 `suh` and `svh` widened from the pack's fp16.
/// - The routed experts in bf16 (`LayerPlan` knows which) become `Proj::Bf16`; the others `Proj::Exl3`, built from the moved buffers.
/// - The MTP head is not taken (`mtp` is `None`): [`take_mtp`] builds it from what this call leaves in `w`.
///
/// A tensor the plan names that is not on the device is an error naming the unit and the tensor, found before anything moves.
/// `layers` must be within the trunk (an empty range yields the globals only). `w` keeps whatever the call did not take (the vision
/// tower is never in this upload, and the MTP blocks stay in it if they were uploaded).
pub fn into_device(
    w: &mut Weights,
    cfg: &OmniConfig,
    plan: &LayerPlan,
    layers: Range<usize>,
    st: &Stream,
) -> Result<NemotronWeights, ModelError> {
    into_device_tp(w, cfg, plan, layers, st, &ShardPlan::single())
}

/// [`into_device`] for one rank of a TP group. `w` is the rank's upload ([`super::tp_shard::upload_tp`]): its dense linears carry
/// their local width (`Exl3Weights::n` is `n_local`, which the arms read), the routed experts of the other ranks are not in it and
/// become `Proj::Absent` (the MoE kernels write zeros for their pairs), and `lm_head` holds the rank's rows. Everything else is as for
/// one node. With `ShardPlan::single()` this is [`into_device`]: the same code path.
pub fn into_device_tp(
    w: &mut Weights,
    cfg: &OmniConfig,
    plan: &LayerPlan,
    layers: Range<usize>,
    st: &Stream,
    shard: &ShardPlan,
) -> Result<NemotronWeights, ModelError> {
    check_range(cfg, &layers)?;
    let trunk = layers.clone().map(|i| trunk_unit(plan, i)).collect::<Result<Vec<_>, _>>()?;
    let embed = global_unit(plan, UnitId::Embed)?;
    let norm_f = global_unit(plan, UnitId::NormF)?;
    let lm_head = global_unit(plan, UnitId::LmHead)?;
    let all: Vec<&Unit> = trunk.iter().copied().chain([embed, norm_f, lm_head]).collect();

    check_presence_tp(w, &all, cfg.experts, shard)?;
    // The loader's copies ran on its own stream; wait for them (and for whatever else is queued on ours) before reading any marker.
    st.sync()?;
    check_codebooks(w, &all, cfg.experts, shard)?;

    let ctx = Ctx { st, arena: Arena::new(WIDEN_BLOCK_BYTES) };
    let mut out = Vec::with_capacity(trunk.len());
    for (i, unit) in layers.clone().zip(&trunk) {
        out.push(match kind_of(cfg, i)? {
            'M' => build_mamba(&ctx, w, unit, i)?,
            '*' => build_attn(w, unit, i)?,
            _ => build_moe(&ctx, w, cfg, unit, i, shard)?,
        });
    }
    let embed = take_buf(w, embed, only(embed)?)?;
    let norm_f = take_buf(w, norm_f, only(norm_f)?)?;
    let lm_head = take_lm_head(w, lm_head)?;
    // A widened vector may still be in flight once `widen` runs on the device; hand the weights over complete.
    st.sync()?;
    Ok(NemotronWeights { first: layers.start, embed, norm_f, lm_head, layers: out, mtp: None })
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Pure host parts: ranges, names, errors, the presence check.

/// The layer range must lie in the trunk and run forward.
fn check_range(cfg: &OmniConfig, layers: &Range<usize>) -> Result<(), ModelError> {
    let n = cfg.pattern.chars().count();
    if layers.start > layers.end {
        return Err(ModelError::Config(format!("layer range {}..{} runs backwards", layers.start, layers.end)));
    }
    if layers.end > n {
        return Err(ModelError::Config(format!(
            "layer range {}..{} is outside the trunk's {n} layers",
            layers.start, layers.end
        )));
    }
    Ok(())
}

/// The block type of trunk layer `i`: `M`, `*` or `E`.
fn kind_of(cfg: &OmniConfig, i: usize) -> Result<char, ModelError> {
    match cfg.pattern.chars().nth(i) {
        Some(c @ ('M' | '*' | 'E')) => Ok(c),
        Some(c) => Err(ModelError::Config(format!("layer {i}: block type `{c}` is not one the bridge builds"))),
        None => Err(ModelError::Config(format!("layer {i} is outside the trunk"))),
    }
}

fn unit_err(unit: &Unit, msg: impl std::fmt::Display) -> ModelError {
    ModelError::Config(format!("{}: {msg}", unit.label))
}

fn trunk_unit(plan: &LayerPlan, i: usize) -> Result<&Unit, ModelError> {
    plan.unit(UnitId::Trunk(i)).ok_or_else(|| ModelError::Config(format!("layer {i}: the plan has no such unit")))
}

fn global_unit(plan: &LayerPlan, id: UnitId) -> Result<&Unit, ModelError> {
    plan.unit(id).ok_or_else(|| ModelError::Config(format!("{id:?}: the plan has no such unit")))
}

/// The `lm_head` in the form its unit plans: the EXL3 linear (the MixedK pack's 8-bit head; on a TP rank its slice, which the presence
/// check held at the rank's width), or the bf16 `[vocab, hidden]` (a rank's rows).
fn take_lm_head(w: &mut Weights, unit: &Unit) -> Result<HeadLinear, ModelError> {
    match unit.linears.as_slice() {
        [l] => Ok(HeadLinear::Exl3(take_exl3(w, unit, &l.prefix)?)),
        [] => Ok(HeadLinear::Bf16(take_buf(w, unit, only(unit)?)?)),
        more => Err(unit_err(unit, format!("expected one linear, the plan has {}", more.len()))),
    }
}

/// The single tensor of a global unit (embeddings, final norm, a bf16 lm_head).
fn only(unit: &Unit) -> Result<&str, ModelError> {
    match unit.tensors.as_slice() {
        [t] => Ok(&t.name),
        other => Err(unit_err(unit, format!("expected one tensor, the plan has {}", other.len()))),
    }
}

/// Every tensor the plan names for `units` is in the upload with the plan's dtype and shape, and every EXL3 linear is one of the
/// upload's. The first miss is returned, naming the unit and the tensor. Nothing is read from the device.
///
/// For a rank of a TP group (`shard`) the expectations are the rank's: a sliced tensor has its local shape, a routed expert of another
/// rank is not expected at all, and a sliced linear has its local width. For `ShardPlan::single()` they are the plan's.
#[cfg(test)]
fn check_presence(w: &Weights, units: &[&Unit]) -> Result<(), ModelError> {
    check_presence_tp(w, units, 0, &ShardPlan::single())
}

/// [`check_presence`] for a rank of a group; `experts` is the routed-expert count of an MoE layer (unused by `ShardPlan::single()`).
fn check_presence_tp(w: &Weights, units: &[&Unit], experts: usize, shard: &ShardPlan) -> Result<(), ModelError> {
    for unit in units {
        for want in &unit.tensors {
            let piece = shard.piece(unit, want, experts);
            let Some(shape) = shard.local_shape(want, &piece) else { continue };
            w.typed(&want.name, want.dtype, &shape).map_err(|e| unit_err(unit, e))?;
        }
        for l in &shard.local_linears(unit, experts) {
            let parts = w.linear(&l.prefix).map_err(|e| unit_err(unit, e))?;
            if (parts.linear.k, parts.linear.n) != (l.k, l.n) {
                return Err(unit_err(
                    unit,
                    format!(
                        "`{}` is [{}, {}] in the upload, the plan expects [{}, {}]",
                        l.prefix, parts.linear.k, parts.linear.n, l.k, l.n
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Every EXL3 linear's `mul1` marker on the device is the mul1 codebook's. The engine decodes each linear as `Mul1`; a pack with another
/// codebook would be decoded as noise, so it is refused here, by name. Reads 4 bytes per linear.
///
/// For a rank of a group, the linears it holds (`ShardPlan::local_linears`); `experts` is the routed-expert count of an MoE layer.
fn check_codebooks(w: &Weights, units: &[&Unit], experts: usize, shard: &ShardPlan) -> Result<(), ModelError> {
    for unit in units {
        for l in &shard.local_linears(unit, experts) {
            let name = format!("{}.mul1", l.prefix);
            let t = w.get(&name).ok_or_else(|| unit_err(unit, format!("`{name}` is not on the device")))?;
            let mut v = [0u8; 4];
            t.buf.download_range(0, &mut v)?;
            let got = u32::from_le_bytes(v);
            if got != MARKER_MUL1 {
                return Err(unit_err(
                    unit,
                    format!("`{name}` holds {got:#010x}, not the mul1 codebook marker {MARKER_MUL1:#010x}"),
                ));
            }
        }
    }
    Ok(())
}

/// The names the bridge claims for one trunk layer, by role. [`unit_names`] lists them against the plan in the tests.
fn layer_base(i: usize) -> String {
    format!("{}.backbone.layers.{i}", super::weights::LM)
}

struct MambaNames {
    norm: String,
    in_proj: String,
    out_proj: String,
    conv_w: String,
    conv_b: String,
    a_log: String,
    d: String,
    dt_bias: String,
    gate_norm: String,
}

impl MambaNames {
    fn of(i: usize) -> Self {
        let base = layer_base(i);
        let m = format!("{base}.mixer");
        Self {
            norm: format!("{base}.norm.weight"),
            in_proj: format!("{m}.in_proj"),
            out_proj: format!("{m}.out_proj"),
            conv_w: format!("{m}.conv1d.weight"),
            conv_b: format!("{m}.conv1d.bias"),
            a_log: format!("{m}.A_log"),
            d: format!("{m}.D"),
            dt_bias: format!("{m}.dt_bias"),
            gate_norm: format!("{m}.norm.weight"),
        }
    }
}

struct AttnNames {
    norm: String,
    q: String,
    k: String,
    v: String,
    o: String,
}

impl AttnNames {
    fn of(i: usize) -> Self {
        let base = layer_base(i);
        let m = format!("{base}.mixer");
        Self {
            norm: format!("{base}.norm.weight"),
            q: format!("{m}.q_proj"),
            k: format!("{m}.k_proj"),
            v: format!("{m}.v_proj"),
            o: format!("{m}.o_proj"),
        }
    }
}

struct MoeNames {
    norm: String,
    gate: String,
    bias: String,
    fc1: String,
    fc2: String,
    shared_up: String,
    shared_down: String,
    experts: String,
}

impl MoeNames {
    fn of(i: usize) -> Self {
        let base = layer_base(i);
        let m = format!("{base}.mixer");
        Self {
            norm: format!("{base}.norm.weight"),
            gate: format!("{m}.gate.weight"),
            bias: format!("{m}.gate.e_score_correction_bias"),
            fc1: format!("{m}.fc1_latent_proj"),
            fc2: format!("{m}.fc2_latent_proj"),
            shared_up: format!("{m}.shared_experts.up_proj"),
            shared_down: format!("{m}.shared_experts.down_proj"),
            experts: format!("{m}.experts"),
        }
    }

    /// The prefix of expert `e`'s `up_proj` or `down_proj`: an EXL3 linear, or `{prefix}.weight` in bf16.
    fn expert(&self, e: usize, proj: &str) -> String {
        format!("{}.{e}.{proj}", self.experts)
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Widening. ONE function touches the device for it, so the glue kernels replace its body and nothing else.

/// The narrow float format a stored vector is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Narrow {
    Bf16,
    F16,
}

/// bf16 bits to f32: the bits shifted into the high half (exact). The widen kernels do this on the device; tests compare them.
#[cfg(test)]
fn bf16_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()
}

/// fp16 bits to f32 (exact, subnormals, infinities and NaN payloads included). Test reference for the widen kernels.
#[cfg(test)]
fn f16_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&h| f16_bits_to_f32(h)).collect()
}

pub(super) fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, _) => {
            // subnormal: shift the top set bit of the 10-bit fraction up to the implicit-one position (bit 10)
            let shift = frac.leading_zeros() - 21;
            let f = (frac << shift) & 0x3ff;
            sign | ((113 - shift) << 23) | (f << 13)
        }
        (0x1f, _) => sign | (0xff << 23) | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// `dst[i] = f32(src[i])` for `n` elements of `from`, `dst` being `n * 4` bytes: the glue widen kernels, launched on `st` (the caller
/// waits on it before it hands the weights over). Both buffers come from the arena and are 16-byte aligned.
fn widen(from: Narrow, src: &DevBuf, n: usize, dst: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
    match from {
        Narrow::Bf16 => widen_bf16_f32(src, n, dst, st),
        Narrow::F16 => widen_f16_f32(src, n, dst, st),
    }
}

/// What the move pass needs besides the upload: the stream, and the arena the widened vectors are carved from.
struct Ctx<'a> {
    st: &'a Stream,
    arena: Arc<Arena>,
}

impl Ctx<'_> {
    /// A new f32 buffer of `n` elements holding `src` widened from `from`.
    fn widened(&self, src: &DevBuf, n: usize, from: Narrow) -> Result<DevBuf, ModelError> {
        let mut dst = self.arena.alloc(n * 4)?;
        widen(from, src, n, &mut dst, self.st)?;
        Ok(dst)
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The move pass. The presence check has passed, so a failure here is a bug in the bridge or the device, and says which tensor.

/// A plain tensor's buffer, moved out of the upload unchanged.
fn take_buf(w: &mut Weights, unit: &Unit, name: &str) -> Result<DevBuf, ModelError> {
    Ok(w.take(name).map_err(|e| unit_err(unit, e))?.buf)
}

/// A plain tensor widened to f32 (`n` elements), moved out of the upload.
fn take_widened(ctx: &Ctx<'_>, w: &mut Weights, unit: &Unit, name: &str, from: Narrow) -> Result<DevBuf, ModelError> {
    let t = w.take(name).map_err(|e| unit_err(unit, e))?;
    let n = t.shape.iter().product::<usize>();
    ctx.widened(&t.buf, n, from)
}

fn take_exl3(w: &mut Weights, unit: &Unit, prefix: &str) -> Result<gb_cuda::exl3_linear::Exl3Weights, ModelError> {
    w.take_linear(prefix).map_err(|e| unit_err(unit, e))
}

/// An EXL3 projection of an MoE table: the trellis moves, `suh` and `svh` are widened to the f32 the table reads.
fn take_proj_exl3(ctx: &Ctx<'_>, w: &mut Weights, unit: &Unit, prefix: &str) -> Result<Proj, ModelError> {
    let t = w.take_parts(prefix).map_err(|e| unit_err(unit, e))?;
    exl3_proj(ctx, t)
}

/// The MoE table entry of an EXL3 linear taken out of the upload: the trellis in place, `suh` and `svh` widened from fp16 to f32.
fn exl3_proj(ctx: &Ctx<'_>, t: super::weights::TakenLinear) -> Result<Proj, ModelError> {
    let (k, n) = (t.linear.k, t.linear.n);
    let suh = ctx.widened(&t.suh.buf, k, Narrow::F16)?;
    let svh = ctx.widened(&t.svh.buf, n, Narrow::F16)?;
    Ok(Proj::Exl3 { k, n, bits: t.bits, codebook: Codebook::Mul1, trellis: t.trellis.buf, suh, svh, bias: None })
}

/// A routed expert's (or the shared expert's) projection `[n, k]`: EXL3 when the plan lists `prefix` as a linear, else the bf16
/// `{prefix}.weight`.
fn take_expert(
    ctx: &Ctx<'_>,
    w: &mut Weights,
    unit: &Unit,
    exl3: &BTreeSet<&str>,
    prefix: &str,
    k: usize,
    n: usize,
) -> Result<Proj, ModelError> {
    if exl3.contains(prefix) {
        return take_proj_exl3(ctx, w, unit, prefix);
    }
    // the dtype too, not the shape alone: `Proj::Bf16` is read as bf16 bits by every MoE kernel, and an F16 `[n, k]` has the same bytes
    take_bf16_proj(w, unit, &format!("{prefix}.weight"), k, n)
}

/// The router gate `[experts, hidden]` as moved out of the upload, before the device pass ([`TakenGate::finish`]).
enum TakenGate {
    /// bf16 bits: moved as stored (today's path; the original router kernels read them).
    Bf16(DevBuf),
    /// An F16 gate's bits: moved as stored and tagged F16, for the router kernels' fp16 instantiation, which widens each value exactly
    /// in its multiply-adds (v0.1.2 decision D1: the f32 kernels' bits on the widened gate, never rounded to bf16, no copy).
    F16(DevBuf),
}

impl TakenGate {
    /// The gate as `nemotron_moe::Weights` holds it, with its tag: the stored bits either way (nothing is copied or widened).
    fn finish(self) -> (DevBuf, GateDtype) {
        match self {
            TakenGate::Bf16(b) => (b, GateDtype::Bf16),
            TakenGate::F16(b) => (b, GateDtype::F16),
        }
    }
}

/// The router gate `[experts, hidden]` moved out of the upload: BF16 (today's path, unchanged) or F16 (for the fp16 router kernels);
/// any other dtype or shape is refused by name. A bf16 kernel decodes the gate's bits as bf16 and an F16 gate has the bf16 gate's
/// bytes, so this dtype check, by name and independent of what the plan admits, is the last one before the tables: the tag it sets
/// (`nemotron_moe::GateDtype`) is what picks the kernels, never the size.
fn take_gate(w: &mut Weights, unit: &Unit, name: &str, experts: usize, hidden: usize) -> Result<TakenGate, ModelError> {
    let t = w.take(name).map_err(|e| unit_err(unit, e))?;
    let shape = [experts, hidden];
    match (t.dtype, t.shape == shape) {
        ("BF16", true) => Ok(TakenGate::Bf16(t.buf)),
        ("F16", true) => Ok(TakenGate::F16(t.buf)),
        _ => Err(unit_err(unit, format!("`{name}` is {} {:?}, expected BF16 or F16 {shape:?}", t.dtype, t.shape))),
    }
}

/// A trunk latent projection `[n, k]` (`fc1_latent_proj`, `fc2_latent_proj`): the EXL3 linear when the plan lists one of that name (our
/// packs), else the F16 `{prefix}.weight` the plan admitted (`weights::latent_f16`: the MixedK pack's), dtype and shape checked by name
/// here too, moved as stored into `Proj::F16` (read in place by the fp16 dense kernels; never rounded to bf16).
fn take_latent(
    ctx: &Ctx<'_>,
    w: &mut Weights,
    unit: &Unit,
    exl3: &BTreeSet<&str>,
    prefix: &str,
    k: usize,
    n: usize,
) -> Result<Proj, ModelError> {
    if exl3.contains(prefix) {
        return take_proj_exl3(ctx, w, unit, prefix);
    }
    Ok(Proj::F16 { k, n, w: take_shaped(w, unit, &format!("{prefix}.weight"), "F16", &[n, k])? })
}

fn build_mamba(ctx: &Ctx<'_>, w: &mut Weights, unit: &Unit, i: usize) -> Result<Layer, ModelError> {
    let n = MambaNames::of(i);
    let mixer = MambaW {
        in_proj: take_exl3(w, unit, &n.in_proj)?,
        out_proj: take_exl3(w, unit, &n.out_proj)?,
        conv_w: take_buf(w, unit, &n.conv_w)?,
        conv_b: take_buf(w, unit, &n.conv_b)?,
        a_log: take_widened(ctx, w, unit, &n.a_log, Narrow::Bf16)?,
        dt_bias: take_widened(ctx, w, unit, &n.dt_bias, Narrow::Bf16)?,
        d: take_widened(ctx, w, unit, &n.d, Narrow::Bf16)?,
        gate_norm: take_widened(ctx, w, unit, &n.gate_norm, Narrow::Bf16)?,
        in_proj_hg: {
            let hg = format!("{}_hg", n.in_proj);
            if w.has_linear(&hg) {
                Some(take_exl3(w, unit, &hg)?)
            } else {
                None
            }
        },
    };
    Ok(Layer { norm: take_buf(w, unit, &n.norm)?, mixer: Mixer::Mamba(mixer) })
}

fn build_attn(w: &mut Weights, unit: &Unit, i: usize) -> Result<Layer, ModelError> {
    let n = AttnNames::of(i);
    let mixer = AttnW {
        q: take_exl3(w, unit, &n.q)?,
        k: take_exl3(w, unit, &n.k)?,
        v: take_exl3(w, unit, &n.v)?,
        o: take_exl3(w, unit, &n.o)?,
    };
    Ok(Layer { norm: take_buf(w, unit, &n.norm)?, mixer: Mixer::Attn(mixer) })
}

/// The device tables of one `E` layer: the gate and its bias, the four dense projections, and every routed expert.
#[cfg(test)]
fn moe_weights(
    ctx: &Ctx<'_>,
    w: &mut Weights,
    cfg: &OmniConfig,
    unit: &Unit,
    i: usize,
) -> Result<nemotron_moe::Weights, ModelError> {
    moe_weights_tp(ctx, w, cfg, unit, i, &ShardPlan::single())
}

/// [`moe_weights`] for a rank of a group: the routed experts the rank does not own are `Proj::Absent`.
fn moe_weights_tp(
    ctx: &Ctx<'_>,
    w: &mut Weights,
    cfg: &OmniConfig,
    unit: &Unit,
    i: usize,
    shard: &ShardPlan,
) -> Result<nemotron_moe::Weights, ModelError> {
    let n = MoeNames::of(i);
    let exl3: BTreeSet<&str> = unit.linears.iter().map(|l| l.prefix.as_str()).collect();
    let (gate, gate_dtype) = take_gate(w, unit, &n.gate, cfg.experts, cfg.hidden)?.finish();
    let router_bias = take_shaped(w, unit, &n.bias, "F32", &[cfg.experts])?;
    let fc1 = take_latent(ctx, w, unit, &exl3, &n.fc1, cfg.hidden, cfg.latent)?;
    let fc2 = take_latent(ctx, w, unit, &exl3, &n.fc2, cfg.latent, cfg.hidden)?;
    // the shared expert's projections are EXL3, or bf16 `{prefix}.weight` when the plan lists no linear of that name
    let shared_up = take_expert(ctx, w, unit, &exl3, &n.shared_up, cfg.hidden, cfg.shared_inter)?;
    let shared_down = take_expert(ctx, w, unit, &exl3, &n.shared_down, cfg.shared_inter, cfg.hidden)?;
    let mut up = Vec::with_capacity(cfg.experts);
    let mut down = Vec::with_capacity(cfg.experts);
    for e in 0..cfg.experts {
        // an expert of another rank is not in the upload: the kernels write zeros for its pairs (the disjoint support the group's
        // sum needs)
        if !shard.owns_layer_expert(i, e, cfg.experts) {
            up.push(Proj::Absent);
            down.push(Proj::Absent);
            continue;
        }
        up.push(take_expert(ctx, w, unit, &exl3, &n.expert(e, "up_proj"), cfg.latent, cfg.expert_inter)?);
        down.push(take_expert(ctx, w, unit, &exl3, &n.expert(e, "down_proj"), cfg.expert_inter, cfg.latent)?);
    }
    Ok(nemotron_moe::Weights {
        gate,
        gate_dtype,
        router_bias,
        scale: cfg.routed_scaling as f32,
        fc1,
        fc2,
        shared_up,
        shared_down,
        up,
        down,
    })
}

fn build_moe(
    ctx: &Ctx<'_>,
    w: &mut Weights,
    cfg: &OmniConfig,
    unit: &Unit,
    i: usize,
    shard: &ShardPlan,
) -> Result<Layer, ModelError> {
    let norm = take_buf(w, unit, &MoeNames::of(i).norm)?;
    let tables = moe_weights_tp(ctx, w, cfg, unit, i, shard)?;
    // The scratch is the model's, shared by every layer (`MoeScratch`, `PrefillScratch`): the block is built from its tables only.
    let moe = PrefillMoe::tables(tables)?;
    Ok(Layer { norm, mixer: Mixer::Moe(Box::new(moe)) })
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The MTP head (`mtp_layers_block_type` `*E`): block 0 is `eh_proj`, `enorm`, `hnorm`, the attention's pre-`norm` and q, k, v and o;
// block 1 is the MoE block's pre-`norm`, `final_layernorm`, the gate and its bias, and the projections (latent, shared, 512 experts).
// Each of `eh_proj`, the attention's projections and the MoE block's projections is EXL3 when the plan has it as a linear (the index
// stores its trellis: our 4-bit head keeps the attention bf16, the MixedK pack's head is EXL3 throughout), else bf16 `.weight` (the
// published pack: every one bf16). One forward path takes either (`Proj`, `HeadLinear`).

/// The names the bridge claims for the head, by role.
struct MtpNames {
    eh_proj: String,
    enorm: String,
    hnorm: String,
    attn_norm: String,
    q: String,
    k: String,
    v: String,
    o: String,
    moe_norm: String,
    final_norm: String,
    gate: String,
    bias: String,
    fc1: String,
    fc2: String,
    shared_up: String,
    shared_down: String,
    experts: String,
}

impl MtpNames {
    fn new() -> Self {
        let lm = super::weights::LM;
        let (b0, b1) = (format!("{lm}.mtp.layers.0"), format!("{lm}.mtp.layers.1"));
        let m = format!("{b1}.mixer");
        Self {
            eh_proj: format!("{b0}.eh_proj.weight"),
            enorm: format!("{b0}.enorm.weight"),
            hnorm: format!("{b0}.hnorm.weight"),
            attn_norm: format!("{b0}.norm.weight"),
            q: format!("{b0}.mixer.q_proj.weight"),
            k: format!("{b0}.mixer.k_proj.weight"),
            v: format!("{b0}.mixer.v_proj.weight"),
            o: format!("{b0}.mixer.o_proj.weight"),
            moe_norm: format!("{b1}.norm.weight"),
            final_norm: format!("{b1}.final_layernorm.weight"),
            gate: format!("{m}.gate.weight"),
            bias: format!("{m}.gate.e_score_correction_bias"),
            fc1: format!("{m}.fc1_latent_proj.weight"),
            fc2: format!("{m}.fc2_latent_proj.weight"),
            shared_up: format!("{m}.shared_experts.up_proj.weight"),
            shared_down: format!("{m}.shared_experts.down_proj.weight"),
            experts: format!("{m}.experts"),
        }
    }

    fn expert(&self, e: usize, proj: &str) -> String {
        format!("{}.{e}.{proj}.weight", self.experts)
    }

    /// Every tensor name the head claims, in the order of the plan's units (the tests compare it with the plan).
    #[cfg(test)]
    fn all(&self, experts: usize) -> Vec<String> {
        let mut v = vec![
            self.eh_proj.clone(),
            self.enorm.clone(),
            self.hnorm.clone(),
            self.attn_norm.clone(),
            self.q.clone(),
            self.k.clone(),
            self.v.clone(),
            self.o.clone(),
            self.final_norm.clone(),
            self.moe_norm.clone(),
            self.gate.clone(),
            self.bias.clone(),
            self.fc1.clone(),
            self.fc2.clone(),
            self.shared_up.clone(),
            self.shared_down.clone(),
        ];
        for e in 0..experts {
            v.push(self.expert(e, "up_proj"));
            v.push(self.expert(e, "down_proj"));
        }
        v
    }
}

/// The prefix of a head projection's `.weight` name: its EXL3 linear's prefix when the plan has it as one.
fn prefix_of(name: &str) -> &str {
    name.strip_suffix(".weight").unwrap_or(name)
}

/// A head projection of the MoE block as the host pass takes it: ready (bf16, or absent for another rank's expert), or an EXL3 linear's
/// parts with their fp16 scales, which [`HeadTables::finish`] widens on the device.
enum HeadProj {
    Ready(Proj),
    Exl3(Box<super::weights::TakenLinear>),
}

impl HeadProj {
    #[cfg(test)]
    fn dims(&self) -> Option<(usize, usize)> {
        match self {
            HeadProj::Ready(Proj::Absent) => None,
            HeadProj::Ready(Proj::Exl3 { k, n, .. } | Proj::Bf16 { k, n, .. } | Proj::F16 { k, n, .. }) => Some((*k, *n)),
            HeadProj::Exl3(t) => Some((t.linear.k, t.linear.n)),
        }
    }

    #[cfg(test)]
    fn is_exl3(&self) -> bool {
        matches!(self, HeadProj::Exl3(_))
    }

    #[cfg(test)]
    fn is_bf16(&self) -> bool {
        matches!(self, HeadProj::Ready(Proj::Bf16 { .. }))
    }

    #[cfg(test)]
    fn is_absent(&self) -> bool {
        matches!(self, HeadProj::Ready(Proj::Absent))
    }

    fn finish(self, ctx: &Ctx<'_>) -> Result<Proj, ModelError> {
        match self {
            HeadProj::Ready(p) => Ok(p),
            HeadProj::Exl3(t) => exl3_proj(ctx, *t),
        }
    }
}

/// The head's MoE tables before the device pass ([`nemotron_moe::Weights`] with [`HeadProj`] projections).
struct HeadTables {
    gate: TakenGate,
    router_bias: DevBuf,
    scale: f32,
    fc1: HeadProj,
    fc2: HeadProj,
    shared_up: HeadProj,
    shared_down: HeadProj,
    up: Vec<HeadProj>,
    down: Vec<HeadProj>,
}

impl HeadTables {
    /// The MoE block's tables: every EXL3 projection's scales widened to f32 on `ctx`'s stream (the caller syncs before handing them over).
    fn finish(self, ctx: &Ctx<'_>) -> Result<nemotron_moe::Weights, ModelError> {
        let HeadTables { gate, router_bias, scale, fc1, fc2, shared_up, shared_down, up, down } = self;
        let (gate, gate_dtype) = gate.finish();
        Ok(nemotron_moe::Weights {
            gate,
            gate_dtype,
            router_bias,
            scale,
            fc1: fc1.finish(ctx)?,
            fc2: fc2.finish(ctx)?,
            shared_up: shared_up.finish(ctx)?,
            shared_down: shared_down.finish(ctx)?,
            up: up.into_iter().map(|p| p.finish(ctx)).collect::<Result<_, _>>()?,
            down: down.into_iter().map(|p| p.finish(ctx)).collect::<Result<_, _>>()?,
        })
    }
}

/// A head projection `[n, k]` of the MoE block, moved out of the upload: the EXL3 linear `prefix` when the plan has it (`exl3`), its
/// shape checked, else the bf16 `{prefix}.weight` of dtype and shape checked by name.
fn take_head_proj(
    w: &mut Weights,
    unit: &Unit,
    exl3: &BTreeSet<&str>,
    prefix: &str,
    k: usize,
    n: usize,
) -> Result<HeadProj, ModelError> {
    if exl3.contains(prefix) {
        let t = w.take_parts(prefix).map_err(|e| unit_err(unit, e))?;
        if (t.linear.k, t.linear.n) != (k, n) {
            return Err(unit_err(unit, format!("`{prefix}` is [{}, {}], expected [{k}, {n}]", t.linear.k, t.linear.n)));
        }
        return Ok(HeadProj::Exl3(Box::new(t)));
    }
    take_bf16_proj(w, unit, &format!("{prefix}.weight"), k, n).map(HeadProj::Ready)
}

/// A bf16 projection `[n, k]` (HF layout) moved out of the upload, its dtype and shape checked by name.
fn take_bf16_proj(w: &mut Weights, unit: &Unit, name: &str, k: usize, n: usize) -> Result<Proj, ModelError> {
    let t = w.take(name).map_err(|e| unit_err(unit, e))?;
    if t.dtype != "BF16" || t.shape != [n, k] {
        return Err(unit_err(unit, format!("`{name}` is {} {:?}, expected BF16 [{n}, {k}]", t.dtype, t.shape)));
    }
    Ok(Proj::Bf16 { k, n, w: t.buf })
}

/// A dense projection of the head's block 0 (`eh_proj`, the attention's q, k, v, o) `[n, k]`, whose bf16 name is `name`: the EXL3 linear
/// `{prefix}` when the plan has it (`exl3`), an `Exl3Weights` as the trunk's dense linears are (fp16 scales, the trellis in place), its
/// `k` and `n` checked; else the bf16 `name` of dtype and shape checked by name.
fn take_head_linear(
    w: &mut Weights,
    unit: &Unit,
    exl3: &BTreeSet<&str>,
    name: &str,
    k: usize,
    n: usize,
) -> Result<HeadLinear, ModelError> {
    let prefix = prefix_of(name);
    if exl3.contains(prefix) {
        let l = take_exl3(w, unit, prefix)?;
        if (l.k, l.n) != (k, n) {
            return Err(unit_err(unit, format!("`{prefix}` is [{}, {}], expected [{k}, {n}]", l.k, l.n)));
        }
        return Ok(HeadLinear::Exl3(l));
    }
    Ok(HeadLinear::Bf16(take_shaped(w, unit, name, "BF16", &[n, k])?))
}

/// A plain tensor of exactly `dtype` and `shape`, moved out of the upload.
fn take_shaped(w: &mut Weights, unit: &Unit, name: &str, dtype: &str, shape: &[usize]) -> Result<DevBuf, ModelError> {
    let t = w.take(name).map_err(|e| unit_err(unit, e))?;
    if t.dtype != dtype || t.shape != shape {
        return Err(unit_err(unit, format!("`{name}` is {} {:?}, expected {dtype} {shape:?}", t.dtype, t.shape)));
    }
    Ok(t.buf)
}

/// Builds the MTP head from the tensors `w` still holds, or `None` when the pack declares no head (`mtp_pattern` empty).
///
/// Refused by name, before anything is taken: a head that is not the `*E` pair this engine builds; a missing tensor, or one with another
/// dtype or shape than the plan's ([`check_presence`]); an EXL3 projection whose `mul1` marker is not the mul1 codebook's
/// ([`check_codebooks`]); `eh_proj` that is not `[hidden, 2 * hidden]`; a head tensor that would be an embedding or an output head of its
/// own (the head scores with the trunk's `embeddings` and `lm_head`, which it never owns: a head that ran without them would accept
/// exactly `1 / (1 + k)`, and one whose tensors were loaded under the wrong prefix would run on random weights). A bf16 projection is
/// `Proj::Bf16` (`HeadLinear::Bf16` for `eh_proj`): the block reads the pack's bytes in place, nothing is copied. An EXL3 projection (the
/// 4-bit head) keeps its trellis in place; the MoE block's take f32 `suh` and `svh` widened from the pack's fp16, as the trunk's do, and
/// `eh_proj` is an `Exl3Weights` as the trunk's dense linears are.
pub fn take_mtp(w: &mut Weights, cfg: &OmniConfig, plan: &LayerPlan, st: &Stream) -> Result<Option<MtpW>, ModelError> {
    take_mtp_tp(w, cfg, plan, st, &ShardPlan::single())
}

/// [`take_mtp`] for one rank of a group: with [`ShardPlan::head_experts`] the routed experts the rank does not own are not in the upload
/// (`tp_shard::upload_tp` never read them) and become `Proj::Absent`, whose pairs the MoE kernels write as exact zeros: the block then
/// runs `forward_in_tp` and the caller's reduce hook sums `y` over the ranks (`mtp::head_block`). `MtpW::experts_split` says so.
pub fn take_mtp_tp(
    w: &mut Weights,
    cfg: &OmniConfig,
    plan: &LayerPlan,
    st: &Stream,
    shard: &ShardPlan,
) -> Result<Option<MtpW>, ModelError> {
    if !cfg.mtp_pattern.is_empty() && !plan.head_exl3().is_empty() {
        // a 4-bit head: every EXL3 projection's codebook marker, read from the device before anything moves (the loader's copies ran on
        // its own stream)
        let units = [global_unit(plan, UnitId::Mtp(0))?, global_unit(plan, UnitId::Mtp(1))?];
        check_presence_tp(w, &units, cfg.experts, shard)?;
        st.sync()?;
        check_codebooks(w, &units, cfg.experts, shard)?;
    }
    let Some(p) = take_mtp_parts_tp(w, cfg, plan, shard)? else { return Ok(None) };
    let ctx = Ctx { st, arena: Arena::new(WIDEN_BLOCK_BYTES) };
    let tables = p.tables.finish(&ctx)?;
    let experts_split = tables.up.iter().chain(tables.down.iter()).any(|e| matches!(e, Proj::Absent));
    // the widened scales may still be in flight; the table upload is synchronous
    st.sync()?;
    let moe = PrefillMoe::tables(tables)?;
    st.sync()?;
    Ok(Some(MtpW {
        experts_split,
        eh_proj: p.eh_proj,
        enorm: p.enorm,
        hnorm: p.hnorm,
        attn_norm: p.attn_norm,
        q: p.q,
        k: p.k,
        v: p.v,
        o: p.o,
        moe_norm: p.moe_norm,
        moe: Box::new(moe),
        final_norm: p.final_norm,
    }))
}

/// The head's tensors, taken and checked, before the MoE tables are built on the device.
struct MtpParts {
    eh_proj: HeadLinear,
    enorm: DevBuf,
    hnorm: DevBuf,
    attn_norm: DevBuf,
    q: HeadLinear,
    k: HeadLinear,
    v: HeadLinear,
    o: HeadLinear,
    moe_norm: DevBuf,
    final_norm: DevBuf,
    tables: HeadTables,
}

/// The host side of [`take_mtp`]: every check but the codebook markers, and the moves out of the upload. No device call.
#[cfg(test)]
fn take_mtp_parts(w: &mut Weights, cfg: &OmniConfig, plan: &LayerPlan) -> Result<Option<MtpParts>, ModelError> {
    take_mtp_parts_tp(w, cfg, plan, &ShardPlan::single())
}

/// [`take_mtp_parts`] for a rank of a group: the expectations of the presence check are the rank's (`ShardPlan::piece`: an expert of
/// the other rank is not in the upload), and those experts are `Proj::Absent`.
fn take_mtp_parts_tp(
    w: &mut Weights,
    cfg: &OmniConfig,
    plan: &LayerPlan,
    shard: &ShardPlan,
) -> Result<Option<MtpParts>, ModelError> {
    if cfg.mtp_pattern.is_empty() {
        return Ok(None);
    }
    if cfg.mtp_pattern != "*E" {
        return Err(ModelError::Config(format!(
            "the MTP head pattern is `{}`; this engine builds the checkpoint's `*E` pair (attention block, latent-MoE block)",
            cfg.mtp_pattern
        )));
    }
    let u0 = global_unit(plan, UnitId::Mtp(0))?;
    let u1 = global_unit(plan, UnitId::Mtp(1))?;
    check_presence_tp(w, &[u0, u1], cfg.experts, shard)?;
    let n = MtpNames::new();
    let h = cfg.hidden;
    for unit in [u0, u1] {
        if let Some(t) = unit.tensors.iter().find(|t| t.name.contains("embed") || t.name.contains("lm_head")) {
            return Err(unit_err(unit, format!("`{}` would be an embedding or output head of the MTP head's own", t.name)));
        }
    }
    // the projections the plan has as EXL3 linears (the index stores their trellis); every other one is bf16 `.weight`
    let exl3: BTreeSet<&str> = [u0, u1].iter().flat_map(|u| u.linears.iter().map(|l| l.prefix.as_str())).collect();
    let attn_n = cfg.heads * cfg.head_dim;
    let kv_n = cfg.kv_heads * cfg.head_dim;
    // `eh_proj` is `[hidden, 2 * hidden]`: the fusion of the embedding-first concat (shape and dtype checked here, with the tensor's name)
    let eh_proj = take_head_linear(w, u0, &exl3, &n.eh_proj, 2 * h, h)?;
    let enorm = take_shaped(w, u0, &n.enorm, "BF16", &[h])?;
    let hnorm = take_shaped(w, u0, &n.hnorm, "BF16", &[h])?;
    let attn_norm = take_shaped(w, u0, &n.attn_norm, "BF16", &[h])?;
    // the attention's q, k, v and o, each bf16 or EXL3 as the pack stores it (the MixedK pack's head is EXL3 throughout)
    let q = take_head_linear(w, u0, &exl3, &n.q, h, attn_n)?;
    let k = take_head_linear(w, u0, &exl3, &n.k, h, kv_n)?;
    let v = take_head_linear(w, u0, &exl3, &n.v, h, kv_n)?;
    let o = take_head_linear(w, u0, &exl3, &n.o, attn_n, h)?;
    let moe_norm = take_shaped(w, u1, &n.moe_norm, "BF16", &[h])?;
    let final_norm = take_shaped(w, u1, &n.final_norm, "BF16", &[h])?;
    let gate = take_gate(w, u1, &n.gate, cfg.experts, h)?;
    // the router's bias is f32 in the pack, as the MoE block reads it
    let router_bias = take_shaped(w, u1, &n.bias, "F32", &[cfg.experts])?;
    let fc1 = take_head_proj(w, u1, &exl3, prefix_of(&n.fc1), h, cfg.latent)?;
    let fc2 = take_head_proj(w, u1, &exl3, prefix_of(&n.fc2), cfg.latent, h)?;
    let shared_up = take_head_proj(w, u1, &exl3, prefix_of(&n.shared_up), h, cfg.shared_inter)?;
    let shared_down = take_head_proj(w, u1, &exl3, prefix_of(&n.shared_down), cfg.shared_inter, h)?;
    let mut up = Vec::with_capacity(cfg.experts);
    let mut down = Vec::with_capacity(cfg.experts);
    for e in 0..cfg.experts {
        // an expert of another rank is not in the upload: its pairs are exact zeros here, the other rank's rows there
        if !shard.is_single() && shard.head_experts && !shard.owns_head_expert(e, cfg.experts) {
            up.push(HeadProj::Ready(Proj::Absent));
            down.push(HeadProj::Ready(Proj::Absent));
            continue;
        }
        up.push(take_head_proj(w, u1, &exl3, prefix_of(&n.expert(e, "up_proj")), cfg.latent, cfg.expert_inter)?);
        down.push(take_head_proj(w, u1, &exl3, prefix_of(&n.expert(e, "down_proj")), cfg.expert_inter, cfg.latent)?);
    }
    let tables = HeadTables { gate, router_bias, scale: cfg.routed_scaling as f32, fc1, fc2, shared_up, shared_down, up, down };
    Ok(Some(MtpParts { eh_proj, enorm, hnorm, attn_norm, q, k, v, o, moe_norm, final_norm, tables }))
}

#[cfg(test)]
mod tests {
    use super::super::weights::{units, Linear, Storage, Storages, Uploaded};
    use super::*;
    use gb_cuda::arena::BlockKind;

    const CONFIG: &str = include_str!("../../../../../testdata/nemotron_omni/config.json");
    /// The measured bf16 expert pairs: (trunk layer, expert). Both projections of each are `.weight`.
    const BF16_PAIRS: [(usize, usize); 3] = [(26, 111), (39, 304), (81, 461)];

    fn cfg() -> OmniConfig {
        OmniConfig::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    fn measured_storages() -> Storages {
        let mut s = Storages::new();
        for (l, e) in BF16_PAIRS {
            for p in ["up_proj", "down_proj"] {
                s.insert((l, e, p), Storage::Bf16);
            }
        }
        s
    }

    /// The plan the pack has: every unit of the config, EXL3 except the measured bf16 pairs.
    fn plan(cfg: &OmniConfig) -> LayerPlan {
        LayerPlan { units: units(cfg, &measured_storages()), outside_plan: 0 }
    }

    /// An upload of the listed units' tensors, each a one-byte buffer in host memory (no GPU needed), and their EXL3 linears.
    /// `skip` names tensors left out, and `retype` tensors given another dtype than the plan's.
    fn synthetic(plan: &LayerPlan, pick: impl Fn(UnitId) -> bool, skip: &[&str], retype: &[(&str, &'static str)]) -> Weights {
        let arena = Arena::with_kind(4 << 20, BlockKind::PageableHost);
        let mut tensors = Vec::new();
        let mut linears: Vec<Linear> = Vec::new();
        for u in plan.units.iter().filter(|u| pick(u.id)) {
            for want in u.tensors.iter().filter(|t| !skip.contains(&t.name.as_str())) {
                // a 4-bit head's `eh_proj` (and an EXL3 attention's q, k, v, o) becomes an `Exl3Weights`, which checks its buffers'
                // sizes: those parts are full size
                let exl3_part = [".trellis", ".suh", ".svh", ".mul1"].iter().any(|s| want.name.ends_with(s));
                let full = want.name.starts_with("language_model.mtp.layers.0.eh_proj.")
                    || (want.name.starts_with("language_model.mtp.layers.0.") && exl3_part);
                let bytes = if full {
                    want.shape.iter().product::<usize>() * gb_exl3::checkpoint::dtype_size(want.dtype).unwrap() as usize
                } else {
                    1
                };
                tensors.push(Uploaded {
                    name: want.name.clone(),
                    dtype: retype.iter().find(|(n, _)| *n == want.name).map_or(want.dtype, |(_, d)| d),
                    shape: want.shape.clone(),
                    shard: "synthetic.safetensors".into(),
                    offset: 0,
                    nbytes: bytes as u64,
                    buf: arena.alloc(bytes).unwrap(),
                });
            }
            linears.extend(u.linears.iter().cloned());
        }
        Weights::from_parts(tensors, linears)
    }

    fn units_of(plan: &LayerPlan, layers: Range<usize>) -> Vec<&Unit> {
        let mut v: Vec<&Unit> = layers.map(|i| plan.unit(UnitId::Trunk(i)).unwrap()).collect();
        v.extend([UnitId::Embed, UnitId::NormF, UnitId::LmHead].map(|id| plan.unit(id).unwrap()));
        v
    }

    fn message(e: ModelError) -> String {
        e.to_string()
    }

    #[test]
    fn the_layer_range_must_lie_in_the_trunk_and_run_forward() {
        let cfg = cfg();
        assert_eq!(cfg.pattern.len(), 88);
        for ok in [0..88, 0..8, 87..88, 5..5, 0..0] {
            assert!(check_range(&cfg, &ok).is_ok(), "{ok:?}");
        }
        let past = message(check_range(&cfg, &(80..89)).unwrap_err());
        assert!(past.contains("80..89") && past.contains("outside the trunk's 88 layers"), "{past}");
        let far = message(check_range(&cfg, &(90..91)).unwrap_err());
        assert!(far.contains("90..91"), "{far}");
        #[allow(clippy::reversed_empty_ranges)]
        let back = message(check_range(&cfg, &(5..3)).unwrap_err());
        assert!(back.contains("5..3") && back.contains("backwards"), "{back}");
    }

    #[test]
    fn the_block_type_of_each_layer_comes_from_the_pattern() {
        let cfg = cfg();
        for (i, c) in cfg.pattern.chars().enumerate() {
            assert_eq!(kind_of(&cfg, i).unwrap(), c);
        }
        assert!(message(kind_of(&cfg, 88).unwrap_err()).contains("layer 88 is outside the trunk"));
    }

    /// Every name the bridge claims for a layer is a tensor of that layer's unit, and together with the parts of the EXL3 linears
    /// (and, for an `E` layer, the experts in the form the plan holds them) they are exactly the unit's tensors.
    #[test]
    fn the_bridge_claims_exactly_the_tensors_the_plan_has_for_each_layer_kind() {
        let cfg = cfg();
        let plan = plan(&cfg);
        let parts = |p: &str| ["trellis", "suh", "svh", "mul1"].map(|s| format!("{p}.{s}"));
        let mut seen = BTreeSet::new();
        for i in [0usize, 1, 7, 26, 39, 81] {
            let kind = kind_of(&cfg, i).unwrap();
            let unit = plan.unit(UnitId::Trunk(i)).unwrap();
            let mut claimed: Vec<String> = Vec::new();
            match kind {
                'M' => {
                    let n = MambaNames::of(i);
                    claimed.extend([n.norm, n.conv_w, n.conv_b, n.a_log, n.d, n.dt_bias, n.gate_norm]);
                    claimed.extend(parts(&n.in_proj));
                    claimed.extend(parts(&n.out_proj));
                }
                '*' => {
                    let n = AttnNames::of(i);
                    claimed.push(n.norm);
                    for p in [&n.q, &n.k, &n.v, &n.o] {
                        claimed.extend(parts(p));
                    }
                }
                _ => {
                    let n = MoeNames::of(i);
                    claimed.extend([n.norm.clone(), n.gate.clone(), n.bias.clone()]);
                    for p in [&n.fc1, &n.fc2, &n.shared_up, &n.shared_down] {
                        claimed.extend(parts(p));
                    }
                    let exl3: BTreeSet<&str> = unit.linears.iter().map(|l| l.prefix.as_str()).collect();
                    for e in 0..cfg.experts {
                        for proj in ["up_proj", "down_proj"] {
                            let p = n.expert(e, proj);
                            if exl3.contains(p.as_str()) {
                                claimed.extend(parts(&p));
                            } else {
                                claimed.push(format!("{p}.weight"));
                            }
                        }
                    }
                }
            }
            let planned: BTreeSet<&str> = unit.tensors.iter().map(|t| t.name.as_str()).collect();
            let got: BTreeSet<&str> = claimed.iter().map(String::as_str).collect();
            assert_eq!(claimed.len(), got.len(), "layer {i}: a name is claimed twice");
            assert_eq!(got, planned, "layer {i} ({kind}): the bridge's names differ from the plan's tensors");
            seen.insert(kind);
        }
        assert_eq!(seen.len(), 3, "all three block types are covered");
    }

    #[test]
    fn the_plan_holds_exactly_the_measured_expert_pairs_in_bf16() {
        let cfg = cfg();
        let plan = plan(&cfg);
        for (l, e) in BF16_PAIRS {
            let unit = plan.unit(UnitId::Trunk(l)).unwrap();
            let n = MoeNames::of(l);
            for proj in ["up_proj", "down_proj"] {
                let p = n.expert(e, proj);
                assert!(!unit.linears.iter().any(|x| x.prefix == p), "{p} is EXL3 in the plan");
                assert!(
                    unit.tensors.iter().any(|t| t.name == format!("{p}.weight") && t.dtype == "BF16"),
                    "{p}.weight is not planned"
                );
            }
            let neighbour = n.expert(e - 1, "up_proj");
            assert!(unit.linears.iter().any(|x| x.prefix == neighbour), "{neighbour} should be EXL3");
        }
    }

    fn head_pick(u: UnitId) -> bool {
        matches!(u, UnitId::Mtp(_))
    }

    /// The names the bridge claims for the head are exactly the tensors of the plan's two MTP units: none missing, none claimed twice,
    /// every one bf16 but the router's f32 bias, and none an embedding or an output head of the head's own.
    #[test]
    fn the_bridge_claims_exactly_the_head_tensors_the_plan_has() {
        let cfg = cfg();
        let plan = plan(&cfg);
        let claimed = MtpNames::new().all(cfg.experts);
        let got: BTreeSet<&str> = claimed.iter().map(String::as_str).collect();
        assert_eq!(claimed.len(), got.len(), "a head name is claimed twice");
        let planned: Vec<&super::super::weights::Want> = plan.tensors(head_pick);
        let want: BTreeSet<&str> = planned.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(got, want, "the bridge's head names differ from the plan's");
        assert_eq!(claimed.len(), 16 + 2 * 512);
        for t in planned {
            let bias = t.name.ends_with("e_score_correction_bias");
            assert_eq!(t.dtype, if bias { "F32" } else { "BF16" }, "{}", t.name);
            assert!(t.name.starts_with("language_model.mtp.layers."), "{}", t.name);
            assert!(!t.name.contains("embed") && !t.name.contains("lm_head"), "{}", t.name);
        }
        // the shapes the head's kernels are built for
        let shape = |name: &str| plan.tensors(head_pick).into_iter().find(|t| t.name == name).unwrap().shape.clone();
        let n = MtpNames::new();
        assert_eq!(shape(&n.eh_proj), vec![4096, 8192]);
        assert_eq!(shape(&n.q), vec![4096, 4096]);
        assert_eq!(shape(&n.k), vec![256, 4096]);
        assert_eq!(shape(&n.o), vec![4096, 4096]);
        assert_eq!(shape(&n.fc1), vec![1024, 4096]);
        assert_eq!(shape(&n.fc2), vec![4096, 1024]);
        assert_eq!(shape(&n.shared_up), vec![5376, 4096]);
        assert_eq!(shape(&n.expert(511, "down_proj")), vec![1024, 2688]);
    }

    #[test]
    fn a_complete_head_is_taken_and_a_missing_wrong_or_foreign_tensor_is_refused_by_name() {
        let cfg = cfg();
        let plan = plan(&cfg);
        let n = MtpNames::new();
        // complete: every tensor leaves the upload, none is left behind
        let mut w = synthetic(&plan, head_pick, &[], &[]);
        let before = w.len();
        let p = take_mtp_parts(&mut w, &cfg, &plan).unwrap_or_else(|e| panic!("{e}")).expect("the pack has a head");
        assert_eq!(before, 16 + 1024);
        assert_eq!(w.len(), 0, "the head took every tensor it was given");
        assert_eq!((p.tables.up.len(), p.tables.down.len()), (512, 512));
        assert!(p.tables.fc1.is_bf16() && p.tables.fc1.dims() == Some((4096, 1024)));
        assert!(p.tables.shared_down.is_bf16() && p.tables.shared_down.dims() == Some((5376, 4096)));
        assert!(p.tables.up[7].is_bf16() && p.tables.up[7].dims() == Some((1024, 2688)));
        assert!(p.tables.down[7].is_bf16() && p.tables.down[7].dims() == Some((2688, 1024)));
        assert!(matches!(p.eh_proj, HeadLinear::Bf16(_)));
        // a missing tensor is named with its unit
        for (skip, label) in [
            (n.eh_proj.as_str(), "mtp block 0 (*)"),
            (n.final_norm.as_str(), "mtp block 1 (E)"),
            (n.bias.as_str(), "mtp block 1 (E)"),
        ] {
            let mut w = synthetic(&plan, head_pick, &[skip], &[]);
            let msg = message(take_mtp_parts(&mut w, &cfg, &plan).err().unwrap());
            assert!(msg.contains(&format!("{label}: `{skip}` is not on the device")), "{skip}: {msg}");
        }
        let missing_expert = n.expert(300, "up_proj");
        let mut w = synthetic(&plan, head_pick, &[&missing_expert], &[]);
        let msg = message(take_mtp_parts(&mut w, &cfg, &plan).err().unwrap());
        assert!(msg.contains(&missing_expert), "{msg}");
        // a wrong dtype is refused with the tensor
        let mut w = synthetic(&plan, head_pick, &[], &[(n.eh_proj.as_str(), "F16")]);
        let msg = message(take_mtp_parts(&mut w, &cfg, &plan).err().unwrap());
        assert!(msg.contains(&n.eh_proj) && msg.contains("F16") && msg.contains("BF16"), "{msg}");
        // a head that is not `*E` is refused by its pattern
        let mut odd = cfg.clone();
        odd.mtp_pattern = "**E".into();
        let mut w = synthetic(&plan, head_pick, &[], &[]);
        let msg = message(take_mtp_parts(&mut w, &odd, &plan).err().unwrap());
        assert!(msg.contains("`**E`"), "{msg}");
        // and a pack with no head builds none
        let mut none = cfg.clone();
        none.mtp_pattern = String::new();
        let mut w = synthetic(&plan, head_pick, &[], &[]);
        assert!(take_mtp_parts(&mut w, &none, &plan).unwrap().is_none());
    }

    /// A rank of a group holds half of the head's routed experts: the upload of rank `r` has only those, the head is built with
    /// `Proj::Absent` for the rest, and a rank asked to replicate the head refuses an upload that lacks them by name.
    #[test]
    fn a_ranks_head_has_its_half_of_the_experts_and_absent_for_the_rest() {
        use super::super::tp::TpTopology;
        let cfg = cfg();
        let plan = plan(&cfg);
        let n = MtpNames::new();
        for rank in 0..2usize {
            let shard = ShardPlan::for_config(TpTopology { rank, ranks: 2 }, &cfg);
            let others: Vec<String> = (0..cfg.experts)
                .filter(|&e| !shard.owns_expert(e, cfg.experts))
                .flat_map(|e| [n.expert(e, "up_proj"), n.expert(e, "down_proj")])
                .collect();
            let skip: Vec<&str> = others.iter().map(String::as_str).collect();
            assert_eq!(skip.len(), 512);
            let mut w = synthetic(&plan, head_pick, &skip, &[]);
            assert_eq!(w.len(), 16 + 1024 - 512, "the rank's upload holds the dense part and its half of the experts");
            let p =
                take_mtp_parts_tp(&mut w, &cfg, &plan, &shard).unwrap_or_else(|e| panic!("{e}")).expect("the pack has a head");
            assert_eq!(w.len(), 0, "the rank's head took every tensor it was given");
            let own = shard.experts(cfg.experts);
            for e in 0..cfg.experts {
                let (up, down) = (&p.tables.up[e], &p.tables.down[e]);
                if own.contains(&e) {
                    assert!(up.is_bf16() && up.dims() == Some((1024, 2688)), "rank {rank} expert {e} up");
                    assert!(down.is_bf16() && down.dims() == Some((2688, 1024)), "rank {rank} expert {e} down");
                } else {
                    assert!(up.is_absent() && down.is_absent(), "rank {rank} expert {e}");
                }
            }
            // the dense part is whole
            assert!(p.tables.fc1.is_bf16() && p.tables.fc1.dims() == Some((4096, 1024)));
            assert!(p.tables.shared_up.is_bf16() && p.tables.shared_up.dims() == Some((4096, 5376)));
            // a rank that replicates the head wants every expert, and says which one is missing
            let mut w = synthetic(&plan, head_pick, &skip, &[]);
            let rep = ShardPlan::for_config(TpTopology { rank, ranks: 2 }, &cfg).replicating_head();
            let msg = message(take_mtp_parts_tp(&mut w, &cfg, &plan, &rep).err().unwrap());
            assert!(msg.contains("mtp.layers.1.mixer.experts."), "{msg}");
        }
    }

    /// A 4-bit head: `eh_proj` and the MoE block's projections EXL3 (two of them kept bf16, as the flagged-matrix policy keeps a matrix),
    /// the attention bf16 (our 4-bit head pack's form).
    fn four_bit_plan(cfg: &OmniConfig) -> LayerPlan {
        let keep =
            ["language_model.mtp.layers.1.mixer.experts.3.down_proj", "language_model.mtp.layers.1.mixer.shared_experts.up_proj"];
        let attn = |p: &str| p.starts_with("language_model.mtp.layers.0.mixer.");
        let head: super::super::weights::HeadExl3 = super::super::weights::head_projections(cfg)
            .into_iter()
            .map(|(p, _, _)| p)
            .filter(|p| !keep.contains(&p.as_str()) && !attn(p))
            .collect();
        LayerPlan { units: super::super::weights::units_with_head(cfg, &measured_storages(), &head), outside_plan: 0 }
    }

    /// The MixedK pack's head: every projection EXL3, the attention's k at 6 bits and o at 5 (the others 4).
    fn exl3_head_plan(cfg: &OmniConfig) -> LayerPlan {
        use super::super::weights::{head_projections, units_with, HeadExl3, Widths};
        let head: HeadExl3 = head_projections(cfg).into_iter().map(|(p, _, _)| p).collect();
        let b0 = "language_model.mtp.layers.0.mixer";
        let widths = Widths::from([(format!("{b0}.k_proj"), 6), (format!("{b0}.o_proj"), 5)]);
        LayerPlan { units: units_with(cfg, &measured_storages(), &head, &widths), outside_plan: 0 }
    }

    /// P0-3: a head whose attention is EXL3 is taken as `HeadLinear::Exl3` for q, k, v and o (each at its width and shape), every tensor
    /// taken; a missing part is refused by name.
    #[test]
    fn a_head_with_exl3_attention_is_taken_as_exl3_linears_at_their_widths() {
        let cfg = cfg();
        let plan = exl3_head_plan(&cfg);
        let mut w = synthetic(&plan, head_pick, &[], &[]);
        let p = take_mtp_parts(&mut w, &cfg, &plan).unwrap_or_else(|e| panic!("{e}")).expect("the pack has a head");
        assert_eq!(w.len(), 0, "the head took every tensor it was given");
        let (attn_n, kv_n) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
        for (what, l, k, n, bits) in [
            ("eh_proj", &p.eh_proj, 2 * cfg.hidden, cfg.hidden, 4),
            ("q", &p.q, cfg.hidden, attn_n, 4),
            ("k", &p.k, cfg.hidden, kv_n, 6),
            ("v", &p.v, cfg.hidden, kv_n, 4),
            ("o", &p.o, attn_n, cfg.hidden, 5),
        ] {
            let l = l.exl3().unwrap_or_else(|| panic!("{what} is EXL3 in this pack"));
            assert_eq!((l.k, l.n, l.bits), (k, n, bits), "{what}");
        }
        assert!(p.tables.fc1.is_exl3() && p.tables.up[0].is_exl3() && p.tables.down[511].is_exl3());
        for gone in ["language_model.mtp.layers.0.mixer.k_proj.svh", "language_model.mtp.layers.0.mixer.o_proj.mul1"] {
            let mut w = synthetic(&plan, head_pick, &[gone], &[]);
            let msg = message(take_mtp_parts(&mut w, &cfg, &plan).err().unwrap());
            assert!(msg.contains(&format!("mtp block 0 (*): `{gone}` is not on the device")), "{gone}: {msg}");
        }
        // our 4-bit head's attention stays bf16
        let mut w = synthetic(&four_bit_plan(&cfg), head_pick, &[], &[]);
        let p = take_mtp_parts(&mut w, &cfg, &four_bit_plan(&cfg)).unwrap_or_else(|e| panic!("{e}")).unwrap();
        assert!([&p.q, &p.k, &p.v, &p.o].iter().all(|l| matches!(l, HeadLinear::Bf16(_))));
    }

    /// The same loader takes a 4-bit head: each projection in the form the plan (the index) has it, the attention, norms and router bf16,
    /// every tensor taken, and a partial EXL3 set refused by the missing part's name.
    #[test]
    fn a_four_bit_head_is_taken_projection_by_projection_in_the_form_the_pack_stores() {
        let cfg = cfg();
        let plan = four_bit_plan(&cfg);
        let mut w = synthetic(&plan, head_pick, &[], &[]);
        let before = w.len();
        assert_eq!(before, 16 + 1024 + 3 * (1 + 3 + 1023), "every EXL3 projection is four tensors");
        let p = take_mtp_parts(&mut w, &cfg, &plan).unwrap_or_else(|e| panic!("{e}")).expect("the pack has a head");
        assert_eq!(w.len(), 0, "the head took every tensor it was given");
        match &p.eh_proj {
            HeadLinear::Exl3(l) => assert_eq!((l.k, l.n, l.bits), (8192, 4096, 4)),
            HeadLinear::Bf16(_) => panic!("eh_proj is EXL3 in this pack"),
        }
        assert!(p.tables.fc1.is_exl3() && p.tables.fc1.dims() == Some((4096, 1024)));
        assert!(p.tables.fc2.is_exl3() && p.tables.fc2.dims() == Some((1024, 4096)));
        assert!(p.tables.shared_up.is_bf16() && p.tables.shared_up.dims() == Some((4096, 5376)), "kept bf16");
        assert!(p.tables.shared_down.is_exl3() && p.tables.shared_down.dims() == Some((5376, 4096)));
        for e in 0..cfg.experts {
            assert!(p.tables.up[e].is_exl3() && p.tables.up[e].dims() == Some((1024, 2688)), "expert {e} up");
            if e == 3 {
                assert!(p.tables.down[e].is_bf16(), "expert 3 down is kept bf16");
            } else {
                assert!(p.tables.down[e].is_exl3() && p.tables.down[e].dims() == Some((2688, 1024)), "expert {e} down");
            }
        }
        // fail closed: a part of an EXL3 projection missing from the upload is named, nothing half-built
        let n = MtpNames::new();
        for gone in [
            "language_model.mtp.layers.1.mixer.experts.9.up_proj.svh".to_string(),
            "language_model.mtp.layers.0.eh_proj.trellis".to_string(),
            "language_model.mtp.layers.1.mixer.fc2_latent_proj.suh".to_string(),
        ] {
            let mut w = synthetic(&plan, head_pick, &[gone.as_str()], &[]);
            let msg = message(take_mtp_parts(&mut w, &cfg, &plan).err().unwrap());
            assert!(msg.contains(&gone), "{gone}: {msg}");
        }
        // a bf16 head's upload is refused by a 4-bit plan (and the reverse): the forms come from one index
        let bf16 = plan_bf16(&cfg);
        let mut w = synthetic(&bf16, head_pick, &[], &[]);
        let msg = message(take_mtp_parts(&mut w, &cfg, &plan).err().unwrap());
        assert!(msg.contains(".trellis") && msg.contains("is not on the device"), "{msg}");
        let mut w = synthetic(&plan, head_pick, &[], &[]);
        let msg = message(take_mtp_parts(&mut w, &cfg, &bf16).err().unwrap());
        assert!(msg.contains(&n.eh_proj) && msg.contains("is not on the device"), "{msg}");
    }

    fn plan_bf16(cfg: &OmniConfig) -> LayerPlan {
        plan(cfg)
    }

    /// A rank of a group with a 4-bit head holds its half of the EXL3 experts (their linears too), the rest absent.
    #[test]
    fn a_ranks_four_bit_head_has_its_half_of_the_experts() {
        use super::super::tp::TpTopology;
        use super::super::tp_shard::head_expert_of;
        let cfg = cfg();
        let plan = four_bit_plan(&cfg);
        for rank in 0..2usize {
            let shard = ShardPlan::for_config(TpTopology { rank, ranks: 2 }, &cfg);
            let u1 = plan.unit(UnitId::Mtp(1)).unwrap();
            let local = shard.local_linears(u1, cfg.experts);
            // the dense EXL3 projections whole (fc1, fc2, shared down), and the owned experts' (expert 3's down is bf16)
            assert!(local.iter().all(|l| head_expert_of(&l.prefix).is_none_or(|e| shard.owns_head_expert(e, cfg.experts))));
            let owned_linears: usize =
                (0..cfg.experts).filter(|&e| shard.owns_head_expert(e, cfg.experts)).map(|e| if e == 3 { 1 } else { 2 }).sum();
            assert_eq!(local.len(), 3 + owned_linears, "rank {rank}");
            let skip: Vec<String> = u1
                .tensors
                .iter()
                .filter(|t| head_expert_of(&t.name).is_some_and(|e| !shard.owns_head_expert(e, cfg.experts)))
                .map(|t| t.name.clone())
                .collect();
            let skip: Vec<&str> = skip.iter().map(String::as_str).collect();
            let mut w = synthetic_linears(&plan, head_pick, &skip, &shard, cfg.experts);
            let p =
                take_mtp_parts_tp(&mut w, &cfg, &plan, &shard).unwrap_or_else(|e| panic!("{e}")).expect("the pack has a head");
            assert_eq!(w.len(), 0, "the rank's head took every tensor it was given");
            for e in 0..cfg.experts {
                let own = shard.owns_head_expert(e, cfg.experts);
                assert_eq!(!p.tables.up[e].is_absent(), own, "rank {rank} expert {e}");
                assert_eq!(!p.tables.down[e].is_absent(), own, "rank {rank} expert {e}");
                if own {
                    assert!(p.tables.up[e].is_exl3());
                }
            }
            assert!(p.tables.fc1.is_exl3() && matches!(p.eh_proj, HeadLinear::Exl3(_)), "the dense part is whole");
        }
    }

    /// [`synthetic`] whose EXL3 linears are the rank's (`ShardPlan::local_linears`), as `tp_shard::upload_tp` registers them.
    fn synthetic_linears(
        plan: &LayerPlan,
        pick: impl Fn(UnitId) -> bool + Copy,
        skip: &[&str],
        shard: &ShardPlan,
        experts: usize,
    ) -> Weights {
        let w = synthetic(plan, pick, skip, &[]);
        let names: Vec<String> = plan
            .units
            .iter()
            .filter(|u| pick(u.id))
            .flat_map(|u| u.tensors.iter().map(|t| t.name.clone()))
            .filter(|n| !skip.contains(&n.as_str()))
            .collect();
        let mut tensors = Vec::new();
        let mut w = w;
        for n in names {
            tensors.push(w.take(&n).unwrap());
        }
        let linears: Vec<Linear> =
            plan.units.iter().filter(|u| pick(u.id)).flat_map(|u| shard.local_linears(u, experts)).collect();
        Weights::from_parts(tensors, linears)
    }

    #[test]
    fn f16_widening_is_exact_for_every_bit_pattern() {
        let all: Vec<u16> = (0..=u16::MAX).collect();
        let got = f16_to_f32(&all);
        for (h, g) in all.iter().zip(&got) {
            let want = gb_exl3::f16::f16_to_f64(*h) as f32;
            if want.is_nan() {
                assert!(g.is_nan(), "{h:#06x}");
                // the payload and the sign are the half's: exponent all ones, fraction shifted into the top of f32's
                assert_eq!(g.to_bits() & 0x7f80_0000, 0x7f80_0000);
                assert_eq!(g.to_bits() >> 31, (*h >> 15) as u32);
            } else {
                assert_eq!(g.to_bits(), want.to_bits(), "{h:#06x}");
            }
        }
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0xc000), -2.0);
        assert_eq!(f16_bits_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_bits_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_bits_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    }

    /// An F16 router gate leaves the bridge as stored, tagged F16 (the fp16 router kernels read it; no widened copy); a bf16 one as
    /// stored, tagged Bf16. Host memory, no GPU.
    #[test]
    fn a_router_gate_leaves_the_bridge_as_stored_with_its_tag() {
        let arena = Arena::with_kind(4 << 20, BlockKind::PageableHost);
        for (taken, dt) in [(TakenGate::F16 as fn(DevBuf) -> TakenGate, GateDtype::F16), (TakenGate::Bf16, GateDtype::Bf16)] {
            let b = arena.alloc(64).unwrap();
            let ptr = b.ptr();
            let (out, got) = taken(b).finish();
            assert_eq!((out.ptr(), out.bytes(), got), (ptr, 64, dt), "the gate is moved as stored, tagged {dt:?}");
        }
    }

    #[test]
    fn bf16_widening_is_the_bits_in_the_high_half() {
        assert_eq!(bf16_to_f32(&[0x3f80, 0xc000, 0x7f80, 0x0000, 0x8000]), vec![1.0, -2.0, f32::INFINITY, 0.0, -0.0]);
        let all: Vec<u16> = (0..=u16::MAX).collect();
        for (b, g) in all.iter().zip(bf16_to_f32(&all)) {
            assert_eq!(g.to_bits(), (*b as u32) << 16);
        }
    }

    #[test]
    fn the_presence_check_passes_on_a_complete_upload_and_names_a_missing_tensor() {
        let cfg = cfg();
        let plan = plan(&cfg);
        let pick =
            |u: UnitId| matches!(u, UnitId::Trunk(i) if i < 3) || matches!(u, UnitId::Embed | UnitId::NormF | UnitId::LmHead);
        let w = synthetic(&plan, pick, &[], &[]);
        let units = units_of(&plan, 0..3);
        check_presence(&w, &units).unwrap_or_else(|e| panic!("{e}"));

        for (skip, label) in [
            ("language_model.backbone.layers.0.mixer.A_log", "layer 0 (M)"),
            ("language_model.backbone.layers.1.mixer.experts.300.up_proj.svh", "layer 1 (E)"),
            ("language_model.backbone.layers.1.mixer.gate.e_score_correction_bias", "layer 1 (E)"),
            ("language_model.backbone.embeddings.weight", "embeddings"),
            ("language_model.lm_head.weight", "lm_head"),
        ] {
            let w = synthetic(&plan, pick, &[skip], &[]);
            let msg = message(check_presence(&w, &units).unwrap_err());
            assert!(msg.contains(&format!("{label}: `{skip}` is not on the device")), "{skip}: {msg}");
        }
    }

    #[test]
    fn a_layer_the_upload_does_not_hold_is_named_by_its_unit() {
        let cfg = cfg();
        let plan = plan(&cfg);
        let w = synthetic(
            &plan,
            |u| matches!(u, UnitId::Trunk(i) if i < 2) || matches!(u, UnitId::Embed | UnitId::NormF | UnitId::LmHead),
            &[],
            &[],
        );
        let units = units_of(&plan, 0..3);
        let msg = message(check_presence(&w, &units).unwrap_err());
        let label = format!("layer 2 ({})", kind_of(&cfg, 2).unwrap());
        assert!(msg.contains(&format!("{label}: `language_model.backbone.layers.2.norm.weight` is not on the device")), "{msg}");
        // and the first two layers alone pass
        check_presence(&w, &units_of(&plan, 0..2)).unwrap();
    }

    #[test]
    fn a_wrong_dtype_or_form_is_refused_with_the_tensor() {
        let cfg = cfg();
        let plan = plan(&cfg);
        // the plan holds A_log as bf16; an upload that has it as f32 is another file
        let a_log = "language_model.backbone.layers.0.mixer.A_log";
        let w = synthetic(&plan, |u| u == UnitId::Trunk(0), &[], &[(a_log, "F32")]);
        let msg = message(check_presence(&w, &[plan.unit(UnitId::Trunk(0)).unwrap()]).unwrap_err());
        assert!(msg.contains(&format!("layer 0 (M): `{a_log}` is F32 [128], not BF16 [128]")), "{msg}");

        // a measured bf16 expert that the upload lacks (the plan says `.weight`) is refused naming the `.weight`
        let p = MoeNames::of(26).expert(111, "up_proj");
        let unit = plan.unit(UnitId::Trunk(26)).unwrap();
        let mut w = synthetic(&plan, |u| u == UnitId::Trunk(26), &[&format!("{p}.weight")], &[]);
        let msg = message(check_presence(&w, &[unit]).unwrap_err());
        assert!(msg.contains(&format!("layer 26 (E): `{p}.weight` is not on the device")), "{msg}");
        // taking what is not there is an error too, not a panic
        assert!(w.take(&format!("{p}.weight")).is_err());
    }

    #[test]
    fn take_and_take_parts_move_tensors_out_of_the_upload() {
        let cfg = cfg();
        let plan = plan(&cfg);
        let mut w = synthetic(&plan, |u| u == UnitId::Trunk(1), &[], &[]);
        let before = w.len();
        let n = MoeNames::of(1);
        let gate = w.take(&n.gate).unwrap();
        assert_eq!((gate.dtype, gate.shape.clone()), ("BF16", vec![512, 4096]));
        assert!(w.get(&n.gate).is_none());
        assert!(w.take(&n.gate).err().unwrap().contains("is not on the device"));
        let parts = w.take_parts(&n.expert(0, "up_proj")).unwrap();
        assert_eq!((parts.linear.k, parts.linear.n, parts.bits.twice()), (1024, 2688, 8));
        assert_eq!(parts.trellis.shape, vec![64, 168, 64]);
        assert_eq!(w.len(), before - 1 - 4, "the gate and the four parts left the upload");
        assert!(w.linear(&n.expert(0, "up_proj")).is_err());
        assert!(w.take_parts(&n.expert(0, "up_proj")).is_err());
    }

    /// `plan` with the dtype of each named tensor changed (a plan relaxed to admit another form, as a later change might).
    fn relaxed(plan: &LayerPlan, retype: &[(&str, &'static str)]) -> LayerPlan {
        let mut p = plan.clone();
        for u in &mut p.units {
            for t in &mut u.tensors {
                if let Some((_, d)) = retype.iter().find(|(n, _)| *n == t.name) {
                    t.dtype = d;
                }
            }
        }
        p
    }

    /// GUARD (v0.1.2 R-MK-rails (d), lane L; reworked by lane K with the fp16 gate path, decision D1): an F16 router gate never reaches
    /// a bf16 router kernel (which would decode its bits as bf16; the F16 gate has the bf16 gate's bytes). The bridge takes a gate by
    /// dtype, by name and independent of what the plan admits: BF16 moves unchanged (today's path), F16 moves as stored with the tag
    /// `GateDtype::F16` (the fp16 kernels), and any other dtype is refused by name before any table is built, trunk and head alike.
    #[test]
    fn a_router_gate_is_taken_by_dtype_bf16_and_f16_as_stored_any_other_refused_by_name() {
        let cfg = cfg();
        let trunk_gate = MoeNames::of(1).gate;
        let head_gate = MtpNames::new().gate;
        // the hole a bytes-only check leaves: the same byte count as the bf16 gate
        let f16 = gb_exl3::checkpoint::dtype_size("F16").unwrap() as usize;
        assert_eq!(cfg.experts * cfg.hidden * f16, gb_cuda::nemotron_moe::EXPERTS * gb_cuda::nemotron_moe::HIDDEN * 2);
        // F16: taken for the widening, never as bits a router reads
        let f16p = relaxed(&plan(&cfg), &[(trunk_gate.as_str(), "F16"), (head_gate.as_str(), "F16")]);
        let unit = f16p.unit(UnitId::Trunk(1)).unwrap();
        let mut w = synthetic(&f16p, |u| u == UnitId::Trunk(1), &[], &[]);
        check_presence(&w, &[unit]).expect("the plan admits the F16 gate");
        assert!(matches!(take_gate(&mut w, unit, &trunk_gate, cfg.experts, cfg.hidden), Ok(TakenGate::F16(_))));
        let mut w = synthetic(&f16p, head_pick, &[], &[]);
        let p = take_mtp_parts(&mut w, &cfg, &f16p).unwrap_or_else(|e| panic!("{e}")).expect("a head");
        assert!(matches!(p.tables.gate, TakenGate::F16(_)), "the head's F16 gate is taken as stored, tagged F16");
        // any other dtype: refused by name
        let other = relaxed(&plan(&cfg), &[(trunk_gate.as_str(), "F32"), (head_gate.as_str(), "I16")]);
        let unit = other.unit(UnitId::Trunk(1)).unwrap();
        let mut w = synthetic(&other, |u| u == UnitId::Trunk(1), &[], &[]);
        let msg = message(take_gate(&mut w, unit, &trunk_gate, cfg.experts, cfg.hidden).err().expect("an F32 gate is refused"));
        assert!(
            msg.contains(&format!("layer 1 (E): `{trunk_gate}` is F32 [512, 4096], expected BF16 or F16 [512, 4096]")),
            "{msg}"
        );
        let mut w = synthetic(&other, head_pick, &[], &[]);
        let msg = message(take_mtp_parts(&mut w, &cfg, &other).err().expect("an I16 head gate is refused"));
        assert!(
            msg.contains(&format!("mtp block 1 (E): `{head_gate}` is I16 [512, 4096], expected BF16 or F16 [512, 4096]")),
            "{msg}"
        );
        // BF16: today's path, the buffer moved as stored
        let base = plan_bf16(&cfg);
        let mut w = synthetic(&base, |u| u == UnitId::Trunk(1), &[], &[]);
        assert!(matches!(
            take_gate(&mut w, base.unit(UnitId::Trunk(1)).unwrap(), &trunk_gate, cfg.experts, cfg.hidden),
            Ok(TakenGate::Bf16(_))
        ));
    }

    /// GUARD: a bf16 expert projection is `Proj::Bf16`, whose bits every MoE kernel reads as bf16; a `.weight` of another dtype (an F16
    /// `[n, k]` has the same bytes) is refused by name by the bridge itself, whatever the plan admitted.
    #[test]
    fn an_expert_weight_that_is_not_bf16_is_refused_by_name_whatever_the_plan_admits() {
        let cfg = cfg();
        let p = format!("{}.weight", MoeNames::of(26).expert(111, "up_proj"));
        let plan = relaxed(&plan(&cfg), &[(p.as_str(), "F16")]);
        let unit = plan.unit(UnitId::Trunk(26)).unwrap();
        let mut w = synthetic(&plan, |u| u == UnitId::Trunk(26), &[], &[]);
        check_presence(&w, &[unit]).expect("the relaxed plan admits the F16 weight");
        let msg = message(take_bf16_proj(&mut w, unit, &p, cfg.latent, cfg.expert_inter).err().expect("refused"));
        assert!(msg.contains(&format!("layer 26 (E): `{p}` is F16 [2688, 1024], expected BF16 [2688, 1024]")), "{msg}");
    }

    // -----------------------------------------------------------------------------------------------------------------------------
    // GPU gate. Needs a CUDA device and REXL3_NEMO_OMNI_DIR (the pack); without either the test says so and passes.
    //   cargo test -p gb-model --release --lib nemotron_h_omni::bridge::tests::gpu -- --nocapture --test-threads=1

    use super::super::inventory::Inventory;
    use super::super::moe_ref::{latent_moe, route, Dense as RefDense, Exl3 as RefExl3, Mlp, Proj as RefProj, Router};
    use super::super::weights::upload;
    use gb_cuda::loader::Options;
    use gb_cuda::nemotron_moe::{MoeScratch, NemotronMoe};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::fs::File;
    use std::os::unix::fs::FileExt;
    use std::path::{Path, PathBuf};

    const CHUNK: usize = 64 << 20;
    /// The ORIGINAL pack's layer-1 fixture (index sha256 4bd072f9...).
    const LAYER1_FIXTURE: &[u8] = include_bytes!("../../../../../testdata/nemotron_omni/moe/layer1_fixture.safetensors");
    /// The repaired pack's own (the repair re-encoded layer 1's shared up projection, so the block differs from the original fixture
    /// by 3.7e-2 by construction): the fixture of every pack that holds the repaired tensors, chosen by the pack's index as
    /// gb-cuda's tests/nemotron_moe.rs (0d18d29b) and gb-model's tests/nemotron_omni_moe.rs choose theirs.
    const LAYER1_FIXTURE_REPAIRED: &[u8] =
        include_bytes!("../../../../../testdata/nemotron_omni/moe/layer1_fixture_repaired.safetensors");
    /// FNV-1a 64 of `model.safetensors.index.json` of the packs that hold the repaired tensors: the repaired pack (5c4938c5...) and
    /// the published Hub pack (d2c82b14..., the repaired tensors re-sharded), and the H4 pack (eb0f1ada...: the repaired trunk, layer 1
    /// byte-identical, with a 4-bit head). Any other index: the original fixture.
    const REPAIRED_INDEX_FNV1A: [u64; 3] = [0xfae2a114dd50e0e2, 0xc7f15360ccd846e4, 0x502dfb7c3ea85676];
    /// The block tolerance of the existing GPU gate (`crates/cuda/tests/nemotron_moe.rs`): max |got - want| / max |want|.
    const BLOCK_TOL: f64 = 1e-3;
    /// The same gate's tolerance for one bf16 expert against torch.
    const EXPERT_TOL: f64 = 1e-4;

    fn pack_dir() -> Option<PathBuf> {
        let dir = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from);
        if dir.is_none() {
            eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set (the published pack directory)");
            return None;
        }
        if !gb_cuda::device_available() {
            eprintln!("not run: no CUDA device");
            return None;
        }
        gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
        dir
    }

    fn pack_config(dir: &Path) -> OmniConfig {
        let text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| panic!("config.json: {e}"));
        OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap_or_else(|e| panic!("{e}"))
    }

    fn digest_device(buf: &DevBuf, len: usize) -> [u8; 32] {
        let mut h = Sha256::new();
        let mut chunk = vec![0u8; CHUNK.min(len).max(1)];
        let mut at = 0;
        while at < len {
            let n = CHUNK.min(len - at);
            buf.download_range(at, &mut chunk[..n]).unwrap_or_else(|e| panic!("download at {at}: {e}"));
            h.update(&chunk[..n]);
            at += n;
        }
        h.finalize().into()
    }

    fn read_file(path: &Path, off: u64, len: u64) -> Vec<u8> {
        let f = File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut v = vec![0u8; len as usize];
        f.read_exact_at(&mut v, off).unwrap_or_else(|e| panic!("{} at {off}: {e}", path.display()));
        v
    }

    fn digest_file(path: &Path, off: u64, len: u64) -> [u8; 32] {
        let f = File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut h = Sha256::new();
        let mut chunk = vec![0u8; CHUNK.min(len as usize).max(1)];
        let mut at = 0u64;
        while at < len {
            let n = (CHUNK as u64).min(len - at) as usize;
            f.read_exact_at(&mut chunk[..n], off + at).unwrap_or_else(|e| panic!("{} at {}: {e}", path.display(), off + at));
            h.update(&chunk[..n]);
            at += n as u64;
        }
        h.finalize().into()
    }

    fn u16s(bytes: &[u8]) -> Vec<u16> {
        bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }

    /// The device buffer `buf` holds exactly the bytes of tensor `name` in its shard.
    fn assert_bytes_equal_file(dir: &Path, inv: &Inventory, name: &str, buf: &DevBuf) {
        let t = inv.get(name).unwrap_or_else(|| panic!("`{name}` is not in the inventory"));
        assert_eq!(buf.bytes() as u64, t.nbytes, "`{name}`: {} device bytes, {} in the file", buf.bytes(), t.nbytes);
        assert_eq!(
            digest_device(buf, buf.bytes()),
            digest_file(&dir.join(&t.shard), t.offset, t.nbytes),
            "`{name}` differs from its file bytes"
        );
    }

    /// The f32 device buffer `buf` is, bit for bit, the stored `name` (bf16 or fp16) widened.
    fn assert_widened_equals_file(dir: &Path, inv: &Inventory, name: &str, buf: &DevBuf) {
        let t = inv.get(name).unwrap_or_else(|| panic!("`{name}` is not in the inventory"));
        let raw = u16s(&read_file(&dir.join(&t.shard), t.offset, t.nbytes));
        let want = match t.dtype.as_str() {
            "BF16" => bf16_to_f32(&raw),
            "F16" => f16_to_f32(&raw),
            other => panic!("`{name}` is {other}"),
        };
        let got = buf.to_vec::<f32>().unwrap();
        assert_eq!(got.len(), want.len(), "`{name}`: {} f32 on the device, {} in the file", got.len(), want.len());
        let wrong = got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
        assert_eq!(wrong, 0, "`{name}`: {wrong} of {} widened values differ from the file's", want.len());
    }

    /// The layer-1 fixture's tensors by name (safetensors header, then bytes).
    struct Fixture {
        tensors: BTreeMap<String, (String, Range<usize>)>,
        data: &'static [u8],
    }

    impl Fixture {
        /// The fixture that goes with the pack in `dir`, chosen by its index (see [`REPAIRED_INDEX_FNV1A`]).
        fn for_pack(dir: &Path) -> Self {
            let index = dir.join("model.safetensors.index.json");
            let bytes = std::fs::read(&index).unwrap_or_else(|e| panic!("{}: {e}", index.display()));
            let h = gb_exl3::fnv1a(bytes.iter().copied());
            let repaired = REPAIRED_INDEX_FNV1A.contains(&h);
            eprintln!("pack index fnv1a {h:016x}: layer-1 fixture {}", if repaired { "repaired" } else { "original" });
            Self::parse(if repaired { LAYER1_FIXTURE_REPAIRED } else { LAYER1_FIXTURE })
        }

        fn parse(b: &'static [u8]) -> Self {
            let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
            let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&b[8..8 + n]).unwrap();
            let mut tensors = BTreeMap::new();
            for (name, v) in header {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = v["dtype"].as_str().unwrap().to_string();
                let o = v["data_offsets"].as_array().unwrap();
                let (a, e) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
                tensors.insert(name, (dtype, 8 + n + a..8 + n + e));
            }
            Self { tensors, data: b }
        }

        fn bytes(&self, name: &str, dtype: &str) -> &'static [u8] {
            let (d, r) = self.tensors.get(name).unwrap_or_else(|| panic!("fixture has no tensor {name}"));
            assert_eq!(d, dtype, "{name} is {d}, not {dtype}");
            &self.data[r.clone()]
        }

        fn f32(&self, name: &str) -> Vec<f32> {
            self.bytes(name, "F32").chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
        }

        fn i64(&self, name: &str) -> Vec<i64> {
            self.bytes(name, "I64").chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
        }
    }

    /// max |got - want| / max |want|, the existing gate's measure.
    fn rel_err(got: &[f32], want: &[f32]) -> f64 {
        assert_eq!(got.len(), want.len(), "length {} against {}", got.len(), want.len());
        let scale = want.iter().fold(0f64, |m, &v| m.max((v as f64).abs()));
        let diff = got.iter().zip(want).fold(0f64, |m, (&g, &w)| m.max(((g - w) as f64).abs()));
        diff / scale
    }

    /// The CPU router of an `E` layer, from the pack's gate and bias.
    fn cpu_router(ck: &gb_exl3::checkpoint::Checkpoint, layer: usize, cfg: &OmniConfig) -> Router {
        let m = format!("language_model.backbone.layers.{layer}.mixer");
        let floats = |name: &str| -> Vec<f32> {
            let e = ck.plain.get(name).unwrap_or_else(|| panic!("the pack has no plain tensor {name}"));
            ck.read_floats(e).unwrap().into_iter().map(|v| v as f32).collect()
        };
        Router {
            gate: floats(&format!("{m}.gate.weight")),
            bias: floats(&format!("{m}.gate.e_score_correction_bias")),
            hidden: cfg.hidden,
            topk: cfg.topk,
            scale: cfg.routed_scaling as f32,
        }
    }

    /// The experts each of `rows` rows selects (22 per row, in rank order), on the CPU.
    fn cpu_picks(router: &Router, h: &[f32], rows: usize, cfg: &OmniConfig) -> Vec<usize> {
        let logits = router.logits(h, rows);
        let mut picks = Vec::new();
        for t in 0..rows {
            let (idx, _) = route(&logits[t * cfg.experts..(t + 1) * cfg.experts], &router.bias, cfg.topk, router.scale);
            picks.extend(idx);
        }
        picks
    }

    /// The CPU reference path of the layer's block (`moe_ref::latent_moe` on the pack's weights read through `gb_exl3`), on `h`
    /// (`rows` rows). Returns the block's output and the experts each token selected (22 per row).
    fn cpu_block(
        ck: &gb_exl3::checkpoint::Checkpoint,
        layer: usize,
        h: &[f32],
        rows: usize,
        cfg: &OmniConfig,
    ) -> (Vec<f32>, Vec<usize>) {
        let m = format!("language_model.backbone.layers.{layer}.mixer");
        let floats = |name: &str| -> Vec<f32> {
            let e = ck.plain.get(name).unwrap_or_else(|| panic!("the pack has no plain tensor {name}"));
            ck.read_floats(e).unwrap().into_iter().map(|v| v as f32).collect()
        };
        let exl3 = |prefix: &str| RefExl3 { group: ck.load_group(prefix).unwrap_or_else(|e| panic!("{prefix}: {e}")) };
        let router = cpu_router(ck, layer, cfg);
        let picks = cpu_picks(&router, h, rows, cfg);
        let mut experts: Vec<Option<Mlp>> = (0..cfg.experts).map(|_| None).collect();
        for &e in picks.iter().collect::<BTreeSet<_>>() {
            let load = |proj: &str| -> Box<dyn RefProj> {
                let prefix = format!("{m}.experts.{e}.{proj}");
                if ck.plain.contains_key(&format!("{prefix}.weight")) {
                    let (out, inp) =
                        if proj == "up_proj" { (cfg.expert_inter, cfg.latent) } else { (cfg.latent, cfg.expert_inter) };
                    Box::new(RefDense::from_hf(out, inp, &floats(&format!("{prefix}.weight"))))
                } else {
                    Box::new(exl3(&prefix))
                }
            };
            experts[e] = Some(Mlp { up: load("up_proj"), down: load("down_proj") });
        }
        let shared = Mlp {
            up: Box::new(exl3(&format!("{m}.shared_experts.up_proj"))),
            down: Box::new(exl3(&format!("{m}.shared_experts.down_proj"))),
        };
        let out = latent_moe(
            h,
            &router,
            &exl3(&format!("{m}.fc1_latent_proj")),
            &exl3(&format!("{m}.fc2_latent_proj")),
            &experts,
            &shared,
        );
        (out, picks)
    }

    fn gib(bytes: usize) -> f64 {
        bytes as f64 / (1u64 << 30) as f64
    }

    /// Layers 0..8 and the globals: uploaded, moved by `into_device`, and checked. A request past the upload is refused by name and
    /// moves nothing. Mamba layer 0 and the first attention layer equal their file bytes (SHA-256 of a readback), the widened vectors
    /// equal the file's widened bit for bit, and the layer-1 MoE block run through the bridge's tables matches the CPU reference path
    /// and torch's fixture within the existing gate's tolerance.
    #[test]
    fn gpu_the_first_eight_layers_and_the_globals_move_and_match_the_pack() {
        let Some(dir) = pack_dir() else { return };
        let cfg = pack_config(&dir);
        let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
        let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
        assert_eq!(cfg.pattern.as_bytes()[1], b'E', "the gate runs the MoE block of layer 1");
        let picked =
            |u: UnitId| matches!(u, UnitId::Trunk(i) if i < 8) || matches!(u, UnitId::Embed | UnitId::NormF | UnitId::LmHead);
        let st = Stream::new().unwrap();

        let (free0, total) = gb_cuda::rt::mem_info().unwrap();
        let mut w = upload(&dir, &inv, &plan, picked, &Options::from_env()).unwrap_or_else(|e| panic!("{e}"));
        st.sync().unwrap();
        let (free1, _) = gb_cuda::rt::mem_info().unwrap();
        let uploaded = w.len();
        let w_bytes = w.bytes;
        eprintln!(
            "upload: {uploaded} tensors, {} bytes ({:.3} GiB) in {:.2} s",
            w.report.bytes,
            gib(w.report.bytes as usize),
            w.report.seconds
        );

        // A request past what was uploaded is refused by unit and tensor, and nothing has moved.
        let msg = into_device(&mut w, &cfg, &plan, 0..9, &st).err().expect("layer 8 was not uploaded").to_string();
        let kind8 = cfg.pattern.chars().nth(8).unwrap();
        assert!(
            msg.contains(&format!("layer 8 ({kind8}): `language_model.backbone.layers.8.norm.weight` is not on the device")),
            "{msg}"
        );
        assert_eq!(w.len(), uploaded, "a refused call must not move anything");
        let msg = into_device(&mut w, &cfg, &plan, 0..89, &st).err().unwrap().to_string();
        assert!(msg.contains("outside the trunk's 88 layers"), "{msg}");

        let t0 = std::time::Instant::now();
        let nw = into_device(&mut w, &cfg, &plan, 0..8, &st).unwrap_or_else(|e| panic!("{e}"));
        let bridge_s = t0.elapsed().as_secs_f64();
        let (free2, _) = gb_cuda::rt::mem_info().unwrap();
        assert!(w.is_empty(), "the bridge left {} tensors of the upload unclaimed", w.len());
        assert_eq!((nw.first, nw.layers.len(), nw.mtp.is_none()), (0, 8, true));
        for (i, l) in nw.layers.iter().enumerate() {
            let got = match &l.mixer {
                Mixer::Mamba(_) => b'M',
                Mixer::Attn(_) => b'*',
                Mixer::Moe(_) => b'E',
            };
            assert_eq!(got, cfg.pattern.as_bytes()[i], "layer {i}");
            assert_eq!(l.norm.bytes(), cfg.hidden * 2);
        }
        assert_eq!(
            (nw.embed.bytes(), nw.norm_f.bytes(), nw.lm_head.bf16().expect("a bf16 lm_head").bytes()),
            (cfg.vocab * cfg.hidden * 2, cfg.hidden * 2, cfg.vocab * cfg.hidden * 2)
        );
        // Resident weights: the upload (arena views, moved not copied) plus the widened f32 vectors. `cudaMemGetInfo` on the GB10 counts
        // unified memory (page cache included), so its deltas are printed for reference only.
        let widened_elems: usize = (0..8)
            .map(|i| match cfg.pattern.as_bytes()[i] {
                b'M' => 3 * cfg.mamba_heads + cfg.mamba_heads * cfg.mamba_dim,
                b'E' => plan.unit(UnitId::Trunk(i)).unwrap().linears.iter().map(|l| l.k + l.n).sum::<usize>(),
                _ => 0,
            })
            .sum();
        let resident = w_bytes as usize + widened_elems * 4;
        eprintln!(
            "peak device memory, layers 0..8 and the globals: {:.3} GiB = upload {:.3} GiB + widened f32 vectors {:.3} GiB ({} elements); \
             nothing else is held (the MoE tables are 512 + 512 + 4 56-byte entries per E layer, and the shared scratch is the model's); \
             bridge {bridge_s:.2} s. cudaMemGetInfo used-memory deltas (unified memory, for reference): upload {:.3} GiB, bridge {:.3} GiB",
            gib(resident),
            gib(w_bytes as usize),
            gib(widened_elems * 4),
            widened_elems,
            gib(free0.saturating_sub(free1)),
            gib(free1.saturating_sub(free2)),
        );
        let _ = total;

        // Mamba layer 0 against the file bytes.
        let n0 = MambaNames::of(0);
        let Mixer::Mamba(m) = &nw.layers[0].mixer else { panic!("layer 0 is not Mamba") };
        assert_bytes_equal_file(&dir, &inv, &n0.norm, &nw.layers[0].norm);
        assert_bytes_equal_file(&dir, &inv, &n0.conv_w, &m.conv_w);
        assert_bytes_equal_file(&dir, &inv, &n0.conv_b, &m.conv_b);
        for (prefix, lin) in [(&n0.in_proj, &m.in_proj), (&n0.out_proj, &m.out_proj)] {
            let bufs = lin.buffers();
            assert_eq!(bufs.iter().map(|(n, _)| *n).collect::<Vec<_>>(), ["trellis", "suh", "svh"]);
            for (part, buf) in bufs {
                assert_bytes_equal_file(&dir, &inv, &format!("{prefix}.{part}"), buf);
            }
        }
        for (name, buf) in [(&n0.a_log, &m.a_log), (&n0.d, &m.d), (&n0.dt_bias, &m.dt_bias), (&n0.gate_norm, &m.gate_norm)] {
            assert_widened_equals_file(&dir, &inv, name, buf);
        }
        eprintln!("layer 0 (Mamba): norm, conv, in_proj and out_proj parts equal their file bytes (SHA-256); a_log, d, dt_bias and the gate norm are the widened file values");

        // The first attention layer in the range, against the file bytes.
        let attn = cfg.pattern.find('*').filter(|&i| i < 8).expect("an attention layer in layers 0..8");
        let na = AttnNames::of(attn);
        let Mixer::Attn(a) = &nw.layers[attn].mixer else { panic!("layer {attn} is not attention") };
        assert_bytes_equal_file(&dir, &inv, &na.norm, &nw.layers[attn].norm);
        for (prefix, lin) in [(&na.q, &a.q), (&na.k, &a.k), (&na.v, &a.v), (&na.o, &a.o)] {
            for (part, buf) in lin.buffers() {
                assert_bytes_equal_file(&dir, &inv, &format!("{prefix}.{part}"), buf);
            }
        }
        eprintln!("layer {attn} (attention): q, k, v, o parts equal their file bytes");

        assert_bytes_equal_file(&dir, &inv, "language_model.backbone.norm_f.weight", &nw.norm_f);
        assert_bytes_equal_file(&dir, &inv, "language_model.backbone.embeddings.weight", &nw.embed);
        assert_bytes_equal_file(&dir, &inv, "language_model.lm_head.weight", nw.lm_head.bf16().expect("a bf16 lm_head"));

        // The layer-1 MoE block, through the bridge's tables and a shared scratch, on the fixture's rows.
        let fx = Fixture::for_pack(&dir);
        let h = fx.f32("h");
        let rows = h.len() / cfg.hidden;
        assert_eq!(rows, 4);
        let Mixer::Moe(moe) = &nw.layers[1].mixer else { panic!("layer 1 is not MoE") };
        let mut scratch = MoeScratch::new(64).unwrap();
        let hd = DevBuf::from_slice(&h).unwrap();
        moe.route_in(&mut scratch, &hd, rows, &st).unwrap();
        let (pick, _) = scratch.download_route(rows, &st).unwrap();
        let got_idx: Vec<i64> = pick.iter().map(|&v| v as i64).collect();
        assert_eq!(got_idx, fx.i64("route_indices"), "the bridge-built router picks other experts than torch");
        let mut out = DevBuf::zeroed(rows * cfg.hidden * 4).unwrap();
        moe.forward_in(&mut scratch, &hd, rows, &mut out, &st).unwrap();
        st.sync().unwrap();
        let got = out.to_vec::<f32>().unwrap();
        let vs_torch = rel_err(&got, &fx.f32("out"));
        eprintln!("layer 1 block through the bridge vs torch's fixture: max rel {vs_torch:.3e} (gate {BLOCK_TOL:e})");
        assert!(vs_torch <= BLOCK_TOL, "block vs torch: {vs_torch:e}");

        let ck = gb_exl3::checkpoint::scan(&dir, true).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let (want, picks) = cpu_block(&ck, 1, &h, rows, &cfg);
        let gpu_picks: Vec<usize> = pick.iter().map(|&v| v as usize).collect();
        assert_eq!(gpu_picks, picks, "GPU and CPU routes differ");
        let vs_cpu = rel_err(&got, &want);
        eprintln!(
            "layer 1 block through the bridge vs the CPU reference (moe_ref::latent_moe on the pack): max rel {vs_cpu:.3e} (gate {BLOCK_TOL:e})"
        );
        assert!(vs_cpu <= BLOCK_TOL, "block vs CPU reference: {vs_cpu:e}");

        // One row alone (the decode path) has the bits of row 0 of the 4-row call.
        let h1 = DevBuf::from_slice(&h[..cfg.hidden]).unwrap();
        let mut o1 = DevBuf::zeroed(cfg.hidden * 4).unwrap();
        moe.forward_in(&mut scratch, &h1, 1, &mut o1, &st).unwrap();
        st.sync().unwrap();
        assert!(o1.to_vec::<f32>().unwrap() == got[..cfg.hidden], "the 1-row call differs from row 0 of the 4-row call");
    }

    /// An `E` layer with a measured bf16 expert pair (layer 26, expert 111): the pair is `Proj::Bf16` holding the file's bytes, its
    /// neighbours are `Proj::Exl3` with the file's trellis and the fp16 scales widened, and the block built from those tables runs
    /// like the CPU reference. The bf16 expert alone matches torch's spot output.
    #[test]
    fn gpu_a_layer_with_a_bf16_expert_pair_is_built_as_the_plan_says() {
        let Some(dir) = pack_dir() else { return };
        let cfg = pack_config(&dir);
        let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
        let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
        assert_eq!(inv.bf16_expert_pairs(), vec![(26, 111), (39, 304), (81, 461)]);
        let st = Stream::new().unwrap();
        let mut w = upload(&dir, &inv, &plan, |u| u == UnitId::Trunk(26), &Options::from_env()).unwrap_or_else(|e| panic!("{e}"));
        let unit = plan.unit(UnitId::Trunk(26)).unwrap();
        let ctx = Ctx { st: &st, arena: Arena::new(WIDEN_BLOCK_BYTES) };
        let norm = take_buf(&mut w, unit, &MoeNames::of(26).norm).unwrap();
        assert_eq!(norm.bytes(), cfg.hidden * 2);
        let mw = moe_weights(&ctx, &mut w, &cfg, unit, 26).unwrap_or_else(|e| panic!("{e}"));
        st.sync().unwrap();
        assert!(w.is_empty(), "the layer's tables and norm left {} tensors unclaimed", w.len());

        let n = MoeNames::of(26);
        let (up_p, down_p) = (n.expert(111, "up_proj"), n.expert(111, "down_proj"));
        match (&mw.up[111], &mw.down[111]) {
            (Proj::Bf16 { k: uk, n: un, w: uw }, Proj::Bf16 { k: dk, n: dn, w: dw }) => {
                assert_eq!((*uk, *un, *dk, *dn), (1024, 2688, 2688, 1024));
                assert_bytes_equal_file(&dir, &inv, &format!("{up_p}.weight"), uw);
                assert_bytes_equal_file(&dir, &inv, &format!("{down_p}.weight"), dw);
            }
            _ => panic!("expert 111 of layer 26 is not Proj::Bf16"),
        }
        for e in [110usize, 112, 0, 511] {
            let up = n.expert(e, "up_proj");
            match &mw.up[e] {
                Proj::Exl3 { k, n: nn, bits, codebook, trellis, suh, svh, bias } => {
                    assert_eq!(
                        (*k, *nn, bits.twice(), *codebook, bias.is_none()),
                        (1024, 2688, 8, Codebook::Mul1, true),
                        "expert {e}"
                    );
                    assert_bytes_equal_file(&dir, &inv, &format!("{up}.trellis"), trellis);
                    assert_widened_equals_file(&dir, &inv, &format!("{up}.suh"), suh);
                    assert_widened_equals_file(&dir, &inv, &format!("{up}.svh"), svh);
                }
                _ => panic!("expert {e} of layer 26 is not Proj::Exl3"),
            }
        }
        match &mw.fc1 {
            Proj::Exl3 { k, n: nn, suh, svh, .. } => {
                assert_eq!((*k, *nn), (4096, 1024));
                assert_widened_equals_file(&dir, &inv, &format!("{}.suh", n.fc1), suh);
                assert_widened_equals_file(&dir, &inv, &format!("{}.svh", n.fc1), svh);
            }
            _ => panic!("fc1 is not Proj::Exl3"),
        }
        assert_bytes_equal_file(&dir, &inv, &n.gate, &mw.gate);
        assert_bytes_equal_file(&dir, &inv, &n.bias, &mw.router_bias);

        // The bf16 expert alone against torch's spot output (layer 1's latents through layer 26's expert 111).
        let fx = Fixture::for_pack(&dir);
        let mut moe = NemotronMoe::new(mw, 16).unwrap();
        let x = DevBuf::from_slice(&fx.f32("spot_bf16_input")).unwrap();
        let mut y = DevBuf::zeroed(4 * cfg.latent * 4).unwrap();
        moe.expert(111, &x, 4, &mut y, &st).unwrap();
        st.sync().unwrap();
        let e = rel_err(&y.to_vec::<f32>().unwrap(), &fx.f32("spot_bf16_output"));
        eprintln!("bf16 expert (layer 26, expert 111) built by the bridge vs torch: max rel {e:.3e} (gate {EXPERT_TOL:e})");
        assert!(e <= EXPERT_TOL, "{e:e}");

        // The whole block against the CPU reference path on the same rows.
        let h = fx.f32("h");
        let ck = gb_exl3::checkpoint::scan(&dir, true).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let (want, picks) = cpu_block(&ck, 26, &h, 4, &cfg);
        let hd = DevBuf::from_slice(&h).unwrap();
        let mut out = DevBuf::zeroed(4 * cfg.hidden * 4).unwrap();
        moe.forward(&hd, 4, &mut out, &st).unwrap();
        st.sync().unwrap();
        let (pick, _) = moe.download_route(4, &st).unwrap();
        assert_eq!(pick.iter().map(|&v| v as usize).collect::<Vec<_>>(), picks, "GPU and CPU routes differ");
        let vs_cpu = rel_err(&out.to_vec::<f32>().unwrap(), &want);
        eprintln!(
            "layer 26 block through the bridge vs the CPU reference: max rel {vs_cpu:.3e} (gate {BLOCK_TOL:e}); the routes include expert 111: {}",
            picks.contains(&111)
        );
        assert!(vs_cpu <= BLOCK_TOL, "{vs_cpu:e}");

        // A block whose routes include the bf16 expert. Rows are noise plus a multiple of the router's own gate row for expert 111, so
        // that its logit is large and it is selected; two more rows are noise only. The block built from the bridge's tables runs on
        // them against the CPU reference, bf16 expert included on both sides.
        let router = cpu_router(&ck, 26, &cfg);
        let g111 = router.gate[111 * cfg.hidden..112 * cfg.hidden].to_vec();
        let g2: f32 = g111.iter().map(|v| v * v).sum();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut draw = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        };
        let (mut hit, mut other) = (Vec::new(), Vec::new());
        for target in [8.0f32, 16.0, 32.0, 64.0] {
            for _ in 0..2 {
                let row: Vec<f32> = g111.iter().map(|&g| 0.25 * draw() + target / g2 * g).collect();
                if cpu_picks(&router, &row, 1, &cfg).contains(&111) {
                    hit.push(row);
                }
            }
            if hit.len() >= 2 {
                break;
            }
        }
        for _ in 0..2 {
            other.push((0..cfg.hidden).map(|_| 0.25 * draw()).collect::<Vec<f32>>());
        }
        hit.truncate(2);
        assert!(!hit.is_empty(), "no row built from expert 111's gate row selected it");
        let rows: Vec<f32> = hit.iter().chain(&other).flatten().copied().collect();
        let n = rows.len() / cfg.hidden;
        let (want, picks) = cpu_block(&ck, 26, &rows, n, &cfg);
        let hd = DevBuf::from_slice(&rows).unwrap();
        let mut out = DevBuf::zeroed(n * cfg.hidden * 4).unwrap();
        moe.forward(&hd, n, &mut out, &st).unwrap();
        st.sync().unwrap();
        let (pick, _) = moe.download_route(n, &st).unwrap();
        assert_eq!(pick.iter().map(|&v| v as usize).collect::<Vec<_>>(), picks, "GPU and CPU routes differ on the random rows");
        assert!(picks.contains(&111));
        let vs_cpu = rel_err(&out.to_vec::<f32>().unwrap(), &want);
        eprintln!(
            "layer 26 block on {n} rows ({} select the bf16 expert 111) vs the CPU reference: max rel {vs_cpu:.3e} (gate {BLOCK_TOL:e})",
            hit.len()
        );
        assert!(vs_cpu <= BLOCK_TOL, "{vs_cpu:e}");
    }
}
