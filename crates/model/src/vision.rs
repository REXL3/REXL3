//! The Qwen3.8 vision tower on the GPU (spec in `docs/VISION.md`, reference ExLlamaV3's `Qwen3VLVisionModel`, MIT).
//!
//! `encode` turns one preprocessed image (`gb_vision::Preprocessed`) into the embeddings the language model consumes: one
//! bf16 row of 5120 per merged 2x2 block of patches. The residual stream is fp32, the GEMMs run through the EXL3 prefill
//! path (reconstruct W_q, cuBLAS fp16 with fp32 accumulate, bias in the output rotation), attention is the bidirectional
//! head_dim-72 flash kernel. One image at a time: the output is deterministic for an image but, like every cuBLAS path, is not
//! row-invariant across batchings.

use crate::error::ModelError;
use crate::loadctx::{push_linear, LoadCtx};
use crate::pack::Pack;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::{vision as vis, DevBuf, Stream};
use gb_vision::{tables, Preprocessed};

pub const DEPTH: usize = 27;
pub const HIDDEN: usize = 1152;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = 72;
/// MLP width as stored (the config says 4304; the pack pads it to a multiple of 128 with inert zeros).
pub const MLP: usize = 4352;
pub const PATCH_IN: usize = 3 * 2 * 16 * 16;
pub const MERGE_IN: usize = HIDDEN * 4;
pub const OUT_HIDDEN: usize = 5120;
pub const POS_TABLE: usize = 2304;
pub const LN_EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 10000.0;

/// An image's encoder output on the device: bf16 `[tokens, 5120]`. Written once by the encoder and only read afterwards (by the
/// engine thread's prefill), so sharing it between the encoding thread and the engine thread is sound.
pub struct ImageFeatures {
    pub buf: DevBuf,
    pub tokens: usize,
}

// SAFETY: the buffer is immutable after encoding and device memory is not tied to a host thread.
unsafe impl Sync for ImageFeatures {}

struct Block {
    norm1_w: DevBuf,
    norm1_b: DevBuf,
    q: Exl3Weights,
    k: Exl3Weights,
    v: Exl3Weights,
    proj: Exl3Weights,
    norm2_w: DevBuf,
    norm2_b: DevBuf,
    fc1: Exl3Weights,
    fc2: Exl3Weights,
}

pub struct VisionTower {
    patch_w: DevBuf,
    patch_b: DevBuf,
    pos_table: DevBuf,
    blocks: Vec<Block>,
    merger_norm_w: DevBuf,
    merger_norm_b: DevBuf,
    merger_fc1: Exl3Weights,
    merger_fc2: Exl3Weights,
    /// Most patches one image may have (scratch is sized for it).
    pub max_patches: usize,
    // scratch
    xin: DevBuf,
    x: DevBuf,
    h: DevBuf,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    att: DevBuf,
    tmp: DevBuf,
    mlp: DevBuf,
    mout: DevBuf,
    idx: DevBuf,
    w: DevBuf,
    cs: DevBuf,
    pws: DevBuf,
    /// Weight bytes on the device.
    pub weight_bytes: usize,
}

fn lin<'p>(
    ctx: &mut LoadCtx<'p>,
    pack: &'p Pack,
    name: &str,
    k: usize,
    n: usize,
    bytes: &mut usize,
) -> Result<Exl3Weights, ModelError> {
    let g = pack.group_meta(name)?;
    if (g.k, g.n) != (k, n) {
        return Err(ModelError::Config(format!("vision {name}: expected K={k} N={n}, found K={} N={}", g.k, g.n)));
    }
    let w = ctx.group(pack, name)?;
    *bytes += w.weight_bytes();
    Ok(w)
}

fn plain16<'p>(
    ctx: &mut LoadCtx<'p>,
    pack: &'p Pack,
    name: &str,
    shape: &[usize],
    bytes: &mut usize,
) -> Result<DevBuf, ModelError> {
    let b = ctx.raw16(pack, name, "F16", shape)?;
    *bytes += b.bytes();
    Ok(b)
}

/// The numbers the encoder is written for, checked against the pack's `config.json` so a different tower fails loudly.
pub fn check_config(config_json: &str) -> Result<(), ModelError> {
    let v: serde_json::Value = serde_json::from_str(config_json).map_err(|e| ModelError::Config(format!("config.json: {e}")))?;
    let vc = v.get("vision_config").ok_or_else(|| ModelError::Config("config.json has no vision_config".into()))?;
    let get = |k: &str| vc.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let checks = [
        ("depth", DEPTH),
        ("hidden_size", HIDDEN),
        ("num_heads", HEADS),
        ("patch_size", 16),
        ("temporal_patch_size", 2),
        ("spatial_merge_size", 2),
        ("num_position_embeddings", POS_TABLE),
        ("out_hidden_size", OUT_HIDDEN),
    ];
    for (k, want) in checks {
        if get(k) != want {
            return Err(ModelError::Config(format!("vision_config.{k} is {}, the encoder is written for {want}", get(k))));
        }
    }
    if vc.get("deepstack_visual_indexes").and_then(|d| d.as_array()).is_some_and(|d| !d.is_empty()) {
        return Err(ModelError::Config("this tower has DeepStack injections, which the encoder does not implement".into()));
    }
    Ok(())
}

impl VisionTower {
    /// Load the tower from the target pack in a pipeline of its own. `max_patches` bounds the image size (patches before the
    /// 2x2 merge).
    pub fn load(pack: &Pack, max_patches: usize) -> Result<Self, ModelError> {
        let mut ctx = LoadCtx::from_env();
        let t = Self::declare(&mut ctx, pack, max_patches)?;
        ctx.finish()?;
        Ok(t)
    }

    /// Every weight buffer of the tower (not its scratch), named and in a fixed order, for comparing two loads of one pack.
    pub fn weight_buffers(&self) -> Vec<(String, &DevBuf)> {
        let mut v: Vec<(String, &DevBuf)> =
            vec![("patch_w".into(), &self.patch_w), ("patch_b".into(), &self.patch_b), ("pos_table".into(), &self.pos_table)];
        for (i, b) in self.blocks.iter().enumerate() {
            let p = format!("blocks.{i}");
            v.push((format!("{p}.norm1_w"), &b.norm1_w));
            v.push((format!("{p}.norm1_b"), &b.norm1_b));
            v.push((format!("{p}.norm2_w"), &b.norm2_w));
            v.push((format!("{p}.norm2_b"), &b.norm2_b));
            for (n, w) in [("q", &b.q), ("k", &b.k), ("v", &b.v), ("proj", &b.proj), ("fc1", &b.fc1), ("fc2", &b.fc2)] {
                push_linear(&mut v, format!("{p}.{n}"), w);
            }
        }
        v.push(("merger_norm_w".into(), &self.merger_norm_w));
        v.push(("merger_norm_b".into(), &self.merger_norm_b));
        push_linear(&mut v, "merger_fc1".into(), &self.merger_fc1);
        push_linear(&mut v, "merger_fc2".into(), &self.merger_fc2);
        v
    }

    /// Declare the tower's weights against `ctx` and allocate its scratch. The weight buffers hold nothing until
    /// `ctx.finish()` has run.
    pub fn declare<'p>(ctx: &mut LoadCtx<'p>, pack: &'p Pack, max_patches: usize) -> Result<Self, ModelError> {
        if max_patches == 0 || max_patches % 4 != 0 {
            return Err(ModelError::Config("max_patches must be a positive multiple of 4".into()));
        }
        let mut bytes = 0usize;
        // [1152, 1536] -> [1536, 1152] so the GEMM reads it as [k, n]
        pack.entry("model.visual.patch_embed.proj.weight", "F16", &[HIDDEN, 3, 2, 16, 16])?;
        let patch_w = ctx.computed(PATCH_IN * HIDDEN * 2, move || {
            let pw = pack.f16("model.visual.patch_embed.proj.weight", &[HIDDEN, 3, 2, 16, 16])?;
            let mut wt = vec![0u16; PATCH_IN * HIDDEN];
            let _t = gb_exl3::loadstats::timer(gb_exl3::loadstats::Stage::Convert, wt.len() as u64 * 2);
            for n in 0..HIDDEN {
                for k in 0..PATCH_IN {
                    wt[k * HIDDEN + n] = pw[n * PATCH_IN + k];
                }
            }
            Ok(wt.iter().flat_map(|x| x.to_le_bytes()).collect())
        })?;
        bytes += patch_w.bytes();
        let patch_b = plain16(ctx, pack, "model.visual.patch_embed.proj.bias", &[HIDDEN], &mut bytes)?;
        let pos_table = ctx.f32_from_bf16(pack, "model.visual.pos_embed.weight", &[POS_TABLE, HIDDEN], false)?;
        bytes += pos_table.bytes();
        let mut blocks = Vec::with_capacity(DEPTH);
        for i in 0..DEPTH {
            let p = format!("model.visual.blocks.{i}.");
            blocks.push(Block {
                norm1_w: plain16(ctx, pack, &format!("{p}norm1.weight"), &[HIDDEN], &mut bytes)?,
                norm1_b: plain16(ctx, pack, &format!("{p}norm1.bias"), &[HIDDEN], &mut bytes)?,
                q: lin(ctx, pack, &format!("{p}attn.q_proj"), HIDDEN, HIDDEN, &mut bytes)?,
                k: lin(ctx, pack, &format!("{p}attn.k_proj"), HIDDEN, HIDDEN, &mut bytes)?,
                v: lin(ctx, pack, &format!("{p}attn.v_proj"), HIDDEN, HIDDEN, &mut bytes)?,
                proj: lin(ctx, pack, &format!("{p}attn.proj"), HIDDEN, HIDDEN, &mut bytes)?,
                norm2_w: plain16(ctx, pack, &format!("{p}norm2.weight"), &[HIDDEN], &mut bytes)?,
                norm2_b: plain16(ctx, pack, &format!("{p}norm2.bias"), &[HIDDEN], &mut bytes)?,
                fc1: lin(ctx, pack, &format!("{p}mlp.linear_fc1"), HIDDEN, MLP, &mut bytes)?,
                fc2: lin(ctx, pack, &format!("{p}mlp.linear_fc2"), MLP, HIDDEN, &mut bytes)?,
            });
        }
        let merger_norm_w = plain16(ctx, pack, "model.visual.merger.norm.weight", &[HIDDEN], &mut bytes)?;
        let merger_norm_b = plain16(ctx, pack, "model.visual.merger.norm.bias", &[HIDDEN], &mut bytes)?;
        let merger_fc1 = lin(ctx, pack, "model.visual.merger.linear_fc1", MERGE_IN, MERGE_IN, &mut bytes)?;
        let merger_fc2 = lin(ctx, pack, "model.visual.merger.linear_fc2", MERGE_IN, OUT_HIDDEN, &mut bytes)?;

        let p = max_patches;
        let b0 = &blocks[0];
        let pws_bytes = [b0.q.prefill_scratch_bytes(p), b0.fc1.prefill_scratch_bytes(p), b0.fc2.prefill_scratch_bytes(p)]
            .into_iter()
            .chain([merger_fc1.prefill_scratch_bytes(p / 4), merger_fc2.prefill_scratch_bytes(p / 4)])
            .max()
            .unwrap_or(0);
        let z = DevBuf::zeroed;
        Ok(Self {
            patch_w,
            patch_b,
            pos_table,
            blocks,
            merger_norm_w,
            merger_norm_b,
            merger_fc1,
            merger_fc2,
            max_patches,
            xin: z(p * PATCH_IN * 2)?,
            x: z(p * HIDDEN * 4)?,
            h: z(p * HIDDEN * 2)?,
            q: z(p * HIDDEN * 2)?,
            k: z(p * HIDDEN * 2)?,
            v: z(p * HIDDEN * 2)?,
            att: z(p * HIDDEN * 2)?,
            tmp: z(p * HIDDEN * 2)?,
            mlp: z(p * MLP * 2)?,
            mout: z(p / 4 * OUT_HIDDEN * 4)?,
            idx: z(p * 16)?,
            w: z(p * 16)?,
            cs: z(p * 72 * 4)?,
            pws: z(pws_bytes)?,
            weight_bytes: bytes,
        })
    }

    /// Device bytes of scratch beyond the weights (what admission must reserve while an image is encoded).
    pub fn scratch_bytes(&self) -> usize {
        [
            &self.xin, &self.x, &self.h, &self.q, &self.k, &self.v, &self.att, &self.tmp, &self.mlp, &self.mout, &self.idx,
            &self.w, &self.cs, &self.pws,
        ]
        .iter()
        .map(|b| b.bytes())
        .sum()
    }

    /// Encode one image into `out`: bf16 `[n_tokens, 5120]`, one row per merged block of 2x2 patches (the rows that replace the
    /// image's `<|image_pad|>` tokens). Runs on `st` without synchronising.
    pub fn encode(&mut self, pre: &Preprocessed, out: &mut DevBuf, st: &Stream) -> Result<(), ModelError> {
        self.encode_observed(pre, out, st, &mut |_, _, _| {})
    }

    /// [`VisionTower::encode`] that shows intermediate buffers to `obs(name, buffer, stream)` as they are produced, for
    /// comparison with the reference dumps (the observer must synchronise `stream` before reading). Names: `pos_embed`
    /// and `block_NN` (the fp32 residual stream `[P, 1152]` after the position embedding and after each block), and for
    /// block 0 `b00_norm1_out`, `b00_attn_out` (the output projection, fp16), `b00_norm2_out`, `b00_mlp_out` (fp16).
    pub fn encode_observed(
        &mut self,
        pre: &Preprocessed,
        out: &mut DevBuf,
        st: &Stream,
        obs: &mut dyn FnMut(&str, &DevBuf, &Stream),
    ) -> Result<(), ModelError> {
        let p = pre.n_patches();
        let (gh, gw) = (pre.grid.1, pre.grid.2);
        if p == 0 || p > self.max_patches || gh % 2 != 0 || gw % 2 != 0 || pre.patch_dim != PATCH_IN {
            return Err(ModelError::Request(format!(
                "image of {p} patches ({gh}x{gw}); the encoder takes up to {}",
                self.max_patches
            )));
        }
        let tokens = p / 4;
        if out.bytes() < tokens * OUT_HIDDEN * 2 {
            return Err(ModelError::Request("output buffer too small for the image's embeddings".into()));
        }
        let (idx, w) = tables::pos_embed_plan(gh, gw);
        self.idx.upload(&idx)?;
        self.w.upload(&w)?;
        self.cs.upload(&tables::rope_cos_sin(gh, gw, ROPE_THETA))?;
        self.xin.upload(&pre.patches)?;

        let Self {
            patch_w,
            patch_b,
            pos_table,
            blocks,
            merger_norm_w,
            merger_norm_b,
            merger_fc1,
            merger_fc2,
            xin,
            x,
            h,
            q,
            k,
            v,
            att,
            tmp,
            mlp,
            mout,
            idx,
            w,
            cs,
            pws,
            ..
        } = self;
        // patch embedding (a GEMM over the 1536-wide patches) and the position embedding, in the fp32 stream
        vis::gemm_f16(xin, patch_w, x, p, PATCH_IN, HIDDEN, st)?;
        vis::pos_add(x, Some(patch_b), pos_table, idx, w, p, HIDDEN, st)?;
        obs("pos_embed", x, st);
        for (bi, b) in blocks.iter().enumerate() {
            vis::layernorm(x, true, &b.norm1_w, &b.norm1_b, h, p, HIDDEN, LN_EPS, st)?;
            if bi == 0 {
                obs("b00_norm1_out", h, st);
            }
            b.q.forward_prefill(h, q, p, Dtype::F16, pws, st)?;
            b.k.forward_prefill(h, k, p, Dtype::F16, pws, st)?;
            b.v.forward_prefill(h, v, p, Dtype::F16, pws, st)?;
            vis::rope2d(q, k, cs, p, HEADS, st)?;
            vis::attention(q, k, v, att, p, HEADS, st)?;
            b.proj.forward_prefill(att, tmp, p, Dtype::F16, pws, st)?;
            if bi == 0 {
                obs("b00_attn_out", tmp, st);
            }
            vis::add_f16(x, tmp, p * HIDDEN, st)?;
            vis::layernorm(x, true, &b.norm2_w, &b.norm2_b, h, p, HIDDEN, LN_EPS, st)?;
            if bi == 0 {
                obs("b00_norm2_out", h, st);
            }
            b.fc1.forward_prefill(h, mlp, p, Dtype::F16, pws, st)?;
            vis::gelu_tanh(mlp, p * MLP, st)?;
            b.fc2.forward_prefill(mlp, tmp, p, Dtype::F16, pws, st)?;
            if bi == 0 {
                obs("b00_mlp_out", tmp, st);
            }
            vis::add_f16(x, tmp, p * HIDDEN, st)?;
            obs(&format!("block_{bi:02}"), x, st);
        }
        // merger: LayerNorm per patch, then 4 consecutive patches (one 2x2 block) are one 4608-wide row
        vis::layernorm(x, true, merger_norm_w, merger_norm_b, h, p, HIDDEN, LN_EPS, st)?;
        merger_fc1.forward_prefill(h, mlp, tokens, Dtype::F16, pws, st)?;
        vis::gelu_tanh(mlp, tokens * MERGE_IN, st)?;
        merger_fc2.forward_prefill_io(mlp, mout, tokens, Dtype::F16, Dtype::F32, pws, st)?;
        vis::f32_to_bf16(mout, out, tokens * OUT_HIDDEN, st)?;
        Ok(())
    }
}
