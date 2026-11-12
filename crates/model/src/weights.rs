//! Target-model weights on the GPU.
//!
//! EXL3 groups become [`Exl3Weights`]; plain tensors become device buffers in the dtype the kernels read: RMSNorm
//! gamma as fp32 (with the pack's stored gamma-1 folded to gamma in fp32, spec 2.1), `A_log` and `dt_bias` as fp32,
//! the depthwise conv and the embedding as bf16, `in_proj_a/b` as fp16.

use crate::config::{LayerKind, QwenConfig};
use crate::error::ModelError;
use crate::loadctx::{push_linear, LoadCtx};
use crate::pack::Pack;
use gb_cuda::exl3_linear::Exl3Weights;
use gb_cuda::DevBuf;

const P: &str = "model.language_model.";

pub struct Mlp {
    pub gate: Exl3Weights,
    pub up: Exl3Weights,
    pub down: Exl3Weights,
}

pub struct LinearAttn {
    pub in_proj_qkv: Exl3Weights,
    pub in_proj_z: Exl3Weights,
    /// fp16 [48, 5120]
    pub in_proj_a: DevBuf,
    /// fp16 [48, 5120]
    pub in_proj_b: DevBuf,
    /// bf16 [10240, 4] (the pack's [10240, 1, 4])
    pub conv_w: DevBuf,
    /// f32 [48]
    pub a_log: DevBuf,
    /// f32 [48]
    pub dt_bias: DevBuf,
    /// f32 [128], plain gamma
    pub norm: DevBuf,
    pub out_proj: Exl3Weights,
}

pub struct FullAttn {
    pub q_proj: Exl3Weights,
    pub k_proj: Exl3Weights,
    pub v_proj: Exl3Weights,
    pub o_proj: Exl3Weights,
    /// f32 [256], gamma
    pub q_norm: DevBuf,
    pub k_norm: DevBuf,
}

pub enum Mixer {
    Linear(Box<LinearAttn>),
    Full(Box<FullAttn>),
}

pub struct Layer {
    /// f32 [5120], gamma
    pub input_norm: DevBuf,
    pub post_norm: DevBuf,
    pub mixer: Mixer,
    pub mlp: Mlp,
}

pub struct Weights {
    /// bf16 [vocab, hidden]
    pub embed: DevBuf,
    pub layers: Vec<Layer>,
    /// f32 [5120], gamma
    pub final_norm: DevBuf,
    pub lm_head: Exl3Weights,
    /// Bytes of device memory the weights occupy.
    pub bytes: usize,
}

fn exl3<'p>(
    ctx: &mut LoadCtx<'p>,
    pack: &'p Pack,
    prefix: &str,
    bytes: &mut usize,
    k: usize,
    n: usize,
) -> Result<Exl3Weights, ModelError> {
    let g = pack.group_meta(prefix)?;
    if (g.k, g.n) != (k, n) {
        return Err(ModelError::Config(format!("{prefix}: expected K={k} N={n}, found K={} N={}", g.k, g.n)));
    }
    let w = ctx.group(pack, prefix)?;
    *bytes += w.weight_bytes();
    Ok(w)
}

impl Weights {
    /// Read every text tensor of the target from `pack` and upload it, in a pipeline of its own. `progress` is called
    /// after each layer is declared (in serial mode, uploaded).
    pub fn load(pack: &Pack, c: &QwenConfig, progress: impl FnMut(usize, usize)) -> Result<Self, ModelError> {
        let mut ctx = LoadCtx::from_env();
        let w = Self::declare(&mut ctx, pack, c, progress)?;
        ctx.finish()?;
        Ok(w)
    }

    /// Declare every text tensor of the target against `ctx`. The buffers hold nothing until `ctx.finish()` has run.
    pub fn declare<'p>(
        ctx: &mut LoadCtx<'p>,
        pack: &'p Pack,
        c: &QwenConfig,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Self, ModelError> {
        let mut bytes = 0usize;
        let d = c.hidden;
        let embed = ctx.raw16(pack, &format!("{P}embed_tokens.weight"), "BF16", &[c.vocab, d])?;
        bytes += embed.bytes();
        let mut layers = Vec::with_capacity(c.layers.len());
        for (i, kind) in c.layers.iter().enumerate() {
            let l = format!("{P}layers.{i}.");
            let input_norm = ctx.f32_from_bf16(pack, &format!("{l}input_layernorm.weight"), &[d], true)?;
            let post_norm = ctx.f32_from_bf16(pack, &format!("{l}post_attention_layernorm.weight"), &[d], true)?;
            bytes += input_norm.bytes() + post_norm.bytes();
            let mlp = Mlp {
                gate: exl3(ctx, pack, &format!("{l}mlp.gate_proj"), &mut bytes, d, c.intermediate)?,
                up: exl3(ctx, pack, &format!("{l}mlp.up_proj"), &mut bytes, d, c.intermediate)?,
                down: exl3(ctx, pack, &format!("{l}mlp.down_proj"), &mut bytes, c.intermediate, d)?,
            };
            let mixer = match kind {
                LayerKind::Linear => {
                    let a = format!("{l}linear_attn.");
                    let hv = c.lin_value_heads;
                    let (qkv, z) = (c.conv_channels(), hv * c.lin_value_dim);
                    let m = LinearAttn {
                        in_proj_qkv: exl3(ctx, pack, &format!("{a}in_proj_qkv"), &mut bytes, d, qkv)?,
                        in_proj_z: exl3(ctx, pack, &format!("{a}in_proj_z"), &mut bytes, d, z)?,
                        in_proj_a: ctx.raw16(pack, &format!("{a}in_proj_a.weight"), "F16", &[hv, d])?,
                        in_proj_b: ctx.raw16(pack, &format!("{a}in_proj_b.weight"), "F16", &[hv, d])?,
                        conv_w: ctx.raw16(pack, &format!("{a}conv1d.weight"), "BF16", &[qkv, 1, c.conv_kernel])?,
                        a_log: ctx.f32_from_bf16(pack, &format!("{a}A_log"), &[hv], false)?,
                        dt_bias: ctx.f32_from_bf16(pack, &format!("{a}dt_bias"), &[hv], false)?,
                        norm: ctx.f32_from_bf16(pack, &format!("{a}norm.weight"), &[c.lin_value_dim], false)?,
                        out_proj: exl3(ctx, pack, &format!("{a}out_proj"), &mut bytes, z, d)?,
                    };
                    bytes += m.in_proj_a.bytes()
                        + m.in_proj_b.bytes()
                        + m.conv_w.bytes()
                        + m.a_log.bytes()
                        + m.dt_bias.bytes()
                        + m.norm.bytes();
                    Mixer::Linear(Box::new(m))
                }
                LayerKind::Full => {
                    let a = format!("{l}self_attn.");
                    let (hq, hk, hd) = (c.heads, c.kv_heads, c.head_dim);
                    let m = FullAttn {
                        // q_proj carries the output gate: per head [q (hd) | gate (hd)]
                        q_proj: exl3(ctx, pack, &format!("{a}q_proj"), &mut bytes, d, hq * hd * 2)?,
                        k_proj: exl3(ctx, pack, &format!("{a}k_proj"), &mut bytes, d, hk * hd)?,
                        v_proj: exl3(ctx, pack, &format!("{a}v_proj"), &mut bytes, d, hk * hd)?,
                        o_proj: exl3(ctx, pack, &format!("{a}o_proj"), &mut bytes, hq * hd, d)?,
                        q_norm: ctx.f32_from_bf16(pack, &format!("{a}q_norm.weight"), &[hd], true)?,
                        k_norm: ctx.f32_from_bf16(pack, &format!("{a}k_norm.weight"), &[hd], true)?,
                    };
                    bytes += m.q_norm.bytes() + m.k_norm.bytes();
                    Mixer::Full(Box::new(m))
                }
            };
            layers.push(Layer { input_norm, post_norm, mixer, mlp });
            progress(i + 1, c.layers.len());
        }
        let final_norm = ctx.f32_from_bf16(pack, &format!("{P}norm.weight"), &[d], true)?;
        bytes += final_norm.bytes();
        let lm_head = exl3(ctx, pack, "lm_head", &mut bytes, d, c.vocab)?;
        Ok(Self { embed, layers, final_norm, lm_head, bytes })
    }

    /// Every device buffer of the target, named and in a fixed order, so two loads of one pack can be compared byte for byte.
    pub fn buffers(&self) -> Vec<(String, &DevBuf)> {
        let mut v: Vec<(String, &DevBuf)> = vec![("embed".into(), &self.embed)];
        for (i, l) in self.layers.iter().enumerate() {
            let p = format!("layers.{i}");
            v.push((format!("{p}.input_norm"), &l.input_norm));
            v.push((format!("{p}.post_norm"), &l.post_norm));
            push_linear(&mut v, format!("{p}.gate"), &l.mlp.gate);
            push_linear(&mut v, format!("{p}.up"), &l.mlp.up);
            push_linear(&mut v, format!("{p}.down"), &l.mlp.down);
            match &l.mixer {
                Mixer::Linear(m) => {
                    push_linear(&mut v, format!("{p}.in_proj_qkv"), &m.in_proj_qkv);
                    push_linear(&mut v, format!("{p}.in_proj_z"), &m.in_proj_z);
                    v.push((format!("{p}.in_proj_a"), &m.in_proj_a));
                    v.push((format!("{p}.in_proj_b"), &m.in_proj_b));
                    v.push((format!("{p}.conv_w"), &m.conv_w));
                    v.push((format!("{p}.a_log"), &m.a_log));
                    v.push((format!("{p}.dt_bias"), &m.dt_bias));
                    v.push((format!("{p}.norm"), &m.norm));
                    push_linear(&mut v, format!("{p}.out_proj"), &m.out_proj);
                }
                Mixer::Full(m) => {
                    push_linear(&mut v, format!("{p}.q_proj"), &m.q_proj);
                    push_linear(&mut v, format!("{p}.k_proj"), &m.k_proj);
                    push_linear(&mut v, format!("{p}.v_proj"), &m.v_proj);
                    push_linear(&mut v, format!("{p}.o_proj"), &m.o_proj);
                    v.push((format!("{p}.q_norm"), &m.q_norm));
                    v.push((format!("{p}.k_norm"), &m.k_norm));
                }
            }
        }
        v.push(("final_norm".into(), &self.final_norm));
        push_linear(&mut v, "lm_head".into(), &self.lm_head);
        v
    }

    /// Every EXL3 linear the prefill runs (all layers; not the head, which only sees one row), in layer order. The prefill's
    /// pinned GEMM must hold for each of these shapes, so start-up checks them and the row-invariance sweep covers them.
    pub fn prefill_linears(&self) -> Vec<&Exl3Weights> {
        let mut v = Vec::new();
        for l in &self.layers {
            match &l.mixer {
                Mixer::Linear(m) => v.extend([&m.in_proj_qkv, &m.in_proj_z, &m.out_proj]),
                Mixer::Full(m) => v.extend([&m.q_proj, &m.k_proj, &m.v_proj, &m.o_proj]),
            }
            v.extend([&l.mlp.gate, &l.mlp.up, &l.mlp.down]);
        }
        v
    }
}
