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

/// Refuse an EXL3 group stored at a half-integer width (an odd `k2`: 1.5, 2.5 or 3.5 bits) before anything is read or allocated.
/// The Qwen dense linears run whole widths only: the decode window, the GEMV prefill and the pinned prefill return a CUDA error on an
/// odd `k2`, and the fused prefill GEMM would decode the trellis at the next whole width down. `name` is the tensor as the error names
/// it, `who` what cannot serve it.
pub(crate) fn refuse_half_width(name: &str, g: &gb_exl3::checkpoint::Group, who: &str) -> Result<(), ModelError> {
    if g.bits.is_integer() {
        return Ok(());
    }
    let k2 = g.bits.twice() as usize;
    Err(ModelError::Config(format!(
        "{name}: an EXL3 tensor at {} bits per weight (a half-integer width, k2 = {k2}); half-width tensors are not servable by {who} \
         yet: quantize it at a whole width ({} or {} bits)",
        g.bits.as_f64(),
        k2 / 2,
        k2 / 2 + 1
    )))
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
    refuse_half_width(prefix, g, "the Qwen dense family")?;
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
                        a_log: ctx.f32_plain(pack, &format!("{a}A_log"), &[hv], false)?,
                        dt_bias: ctx.f32_plain(pack, &format!("{a}dt_bias"), &[hv], false)?,
                        norm: ctx.f32_plain(pack, &format!("{a}norm.weight"), &[c.lin_value_dim], false)?,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loadctx::LoadMode;

    const K: usize = 128;
    const N: usize = 256;

    /// A one-shard pack on disk of Mul1 EXL3 groups `K` x `N`, one per `(prefix, k2)`: what `Pack::open` scans from a real pack (the
    /// trellis `[K/16, N/16, 8 * k2]`, the `.mul1` marker, the fp16 scales), so the width the loader sees is the one the scan parsed.
    fn pack_of(tag: &str, groups: &[(&str, usize)]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("gb-model-half-width-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"quantization_config":{"quant_method":"exl3"}}"#).unwrap();
        let (mut header, mut data) = (serde_json::Map::new(), Vec::<u8>::new());
        for &(prefix, k2) in groups {
            let parts: [(&str, &str, Vec<usize>, Vec<u8>); 4] = [
                ("trellis", "I16", vec![K / 16, N / 16, 8 * k2], vec![0u8; (K / 16) * (N / 16) * 16 * k2]),
                ("mul1", "I32", vec![1], gb_exl3::MARKER_MUL1.to_le_bytes().to_vec()),
                ("suh", "F16", vec![K], [0x00u8, 0x3c].repeat(K)),
                ("svh", "F16", vec![N], [0x00u8, 0x3c].repeat(N)),
            ];
            for (part, dtype, shape, bytes) in parts {
                let at = data.len();
                data.extend_from_slice(&bytes);
                header.insert(
                    format!("{prefix}.{part}"),
                    serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [at, data.len()]}),
                );
            }
        }
        let header = serde_json::Value::Object(header).to_string();
        let mut f = (header.len() as u64).to_le_bytes().to_vec();
        f.extend_from_slice(header.as_bytes());
        f.extend_from_slice(&data);
        std::fs::write(dir.join("model.safetensors"), f).unwrap();
        dir
    }

    /// A half-integer-width tensor (odd k2: 1.5, 2.5, 3.5 bits) is refused when the target is loaded, before anything is read or
    /// allocated, by the tensor's name and width with the whole widths around it; the head goes through the same check. A whole width
    /// passes the check (whatever the upload then does on this machine).
    #[test]
    fn a_half_width_tensor_is_refused_at_load_by_name_and_width() {
        let mlp = "model.language_model.layers.0.mlp.gate_proj";
        let names: Vec<String> = (0..5).map(|i| format!("model.language_model.layers.{i}.mlp.up_proj")).collect();
        let groups: Vec<(&str, usize)> =
            vec![(mlp, 5), ("lm_head", 7), (&names[0], 3), (&names[1], 4), (&names[2], 6), (&names[3], 2), (&names[4], 8)];
        let dir = pack_of("load", &groups);
        let pack = Pack::open(&dir).unwrap();
        for &(prefix, k2) in &groups {
            assert_eq!(pack.group_meta(prefix).unwrap().bits.twice() as usize, k2, "{prefix}: the scan parsed the width");
            let mut ctx = LoadCtx::new(LoadMode::Serial);
            let mut bytes = 0usize;
            let r = exl3(&mut ctx, &pack, prefix, &mut bytes, K, N);
            if k2 % 2 == 0 {
                if let Err(e) = r {
                    assert!(
                        !e.to_string().contains("half-integer"),
                        "{prefix}: a whole width (k2 = {k2}) must pass the check: {e}"
                    );
                }
                continue;
            }
            let e = r.err().unwrap_or_else(|| panic!("{prefix}: k2 = {k2} must be refused"));
            assert!(matches!(e, ModelError::Config(_)), "{prefix}: {e:?}");
            let (bits, lo, hi) = (k2 as f64 / 2.0, k2 / 2, k2 / 2 + 1);
            let msg = e.to_string();
            assert!(msg.contains(&format!("{prefix}: an EXL3 tensor at {bits} bits per weight")), "{msg}");
            assert!(msg.contains(&format!("k2 = {k2}")) && msg.contains("not servable by the Qwen dense family"), "{msg}");
            assert!(msg.contains(&format!("({lo} or {hi} bits)")), "{msg}");
            assert_eq!(bytes, 0, "{prefix}: nothing is counted for a refused tensor");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
