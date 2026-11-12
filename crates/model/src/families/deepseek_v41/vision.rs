//! The DeepSeek-V4.1-Flash vision tower and its aligner: 32 pre-norm blocks of hidden 1024, a final RMSNorm, and the
//! 3 x 3 aligner from 9216 to the language model's 5120 — plus the two things the rest of the model needs from it, the
//! image-span splice into the text stream and the engram layer's token mask.
//!
//! # The reference
//!
//! `/tmp/ds41ref/vision.py` (119 lines) is the whole tower: `PatchEmbed`, `Attention`, `MLP`, `Block`, `ViT`, `Aligner`.
//! `/tmp/ds41ref/image_processor.py` is the token layout. Both are small enough to read in one sitting, and every ordering
//! decision below is one of theirs:
//!
//! * [`PATCH_K`] = **588** = `3 * 14 * 14` = `vision_patch_size^2 * 3`, the input width `PatchEmbed.proj` is
//!   `nn.Linear(3 * args.vision_patch_size**2, args.vision_dim)` for. A patch row is the reference's `x.flatten(1)` on
//!   `[3, 14, 14]`: **channel-major**, then the patch row, then the patch column. [`crate::dsv4_vision::flatten_index`] is
//!   that index and [`crate::dsv4_vision::patchify`] produces it.
//! * The blocks are **pre-norm**: `x = x + attn(norm1(x))`, then `x = x + mlp(norm2(x))`, with `wqkv` one fused
//!   `[3072, 1024]` projection chunked into q | k | v, and the MLP's `w1` `[5632, 1024]` chunked into gate | up.
//! * The tower is **bidirectional** over one image: `F.scaled_dot_product_attention` with no mask, and no position
//!   embedding — positions enter only through a 2-D RoPE on q and k.
//! * The aligner is `w2(gelu(w1(x)))` over `F.unfold(3, stride=3)` of the patch grid padded on the right and bottom, i.e.
//!   `ceil(n_h/3) * ceil(n_w/3)` tokens, each a 3 x 3 x 1024 window in `(C, kH, kW)` order.
//!
//! # The span
//!
//! An image costs [`num_image_tokens`] = `n_llm_h * (n_llm_w + 1) + 2` positions, laid out
//!
//! ```text
//! [IMAGE_START] ([IMAGE] * n_llm_w [IMAGE_NEW_LINE]) * n_llm_h [IMAGE_END]
//! ```
//!
//! Every one of them carries [`IMAGE_TOKEN_ID`] in `input_ids`; only `token_types` tells them apart. The `IMAGE` slots
//! take the aligner's rows in reading order, the rest take learned embeddings ([`splice_into`]).
//!
//! # The engram mask
//!
//! The engram layers (1 and 14) hash n-grams over the compressed token stream. Inside an image span the token ids are all
//! the same placeholder, so an n-gram that reached across one would hash noise — and worse, would hash *the same* noise for
//! every image. So `model.py` computes `engram_mask = None if image_mask is None else ~image_mask` and passes it down;
//! [`engram_mask`] is that function, and a `None` is the text-only fast path (no `where`, no `masked_fill`). A `false`
//! position becomes [`crate::families::deepseek_v41::engram::DEAD`] in the compressed stream, which blocks every n-gram that
//! would span it (see `NgramHash::window`).
//!
//! # What a GPU would check
//!
//! Nothing here needs a device to be wrong about *shape*: [`inventory`] is the whole 266-tensor list and it sums to
//! 970,536,960 bytes, the pack's 0.97 GB. The numerics it cannot check are the ones in `csrc/dsv4_vision.cu` — see the
//! module's own tests and `tests/dsv4_vision.rs`.

use super::config::VisionConfig;
use crate::error::ModelError;
use gb_cuda::dsv4_vision as dv;
pub use gb_cuda::dsv4_vision::{
    aligner_index, engram_mask, image_mask, num_image_tokens, patchify, splice_order, token_types, SpliceSrc, TokenType,
    ALIGN_IN, HEADS, HEAD_DIM, HIDDEN, INTER, LAYERS, MAX_IMAGE_TOKENS, MERGE, MLP, OUT_HIDDEN, PATCH, PATCH_K, RMS_EPS,
    ROPE_DIM, ROPE_THETA,
};
use gb_cuda::{DevBuf, Stream};

/// The image placeholder's id in this pack's vocabulary. Every position of an image span carries it.
pub const IMAGE_TOKEN_ID: u32 = 129_264;

/// `vision_patch_size`.
pub const PATCH_SIZE: usize = PATCH;

// ------------------------------------------------------------------------------------------------------------------ config

/// The numbers the kernels are written for, checked against the pack's `vision_config`: a different tower fails loudly
/// rather than running with the wrong shapes. `max_image_tokens` is *not* checked — it is the processor's budget, and a
/// pack that states a different one still runs (it only changes how big an image may be).
pub fn check_config(v: &VisionConfig) -> Result<(), ModelError> {
    let checks = [
        ("num_hidden_layers", v.layers, LAYERS),
        ("hidden_size", v.hidden, HIDDEN),
        ("num_attention_heads", v.heads, HEADS),
        ("intermediate_size", v.intermediate, INTER),
        ("patch_size", v.patch_size, PATCH),
        ("downsample_ratio", v.downsample_ratio, MERGE),
    ];
    for (k, got, want) in checks {
        if got != want {
            return Err(ModelError::Config(format!("vision_config.{k} is {got}, the DeepSeek-V4.1 tower is written for {want}")));
        }
    }
    if (v.rope_theta - ROPE_THETA as f64).abs() > 1e-9 {
        return Err(ModelError::Config(format!(
            "vision_config.rope_theta is {}, the tower is written for {ROPE_THETA}",
            v.rope_theta
        )));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------------ inventory

/// One tensor of the tower: its pack name, and its shape `[out, in]` (or `[n]` for a norm/embedding, where `in` is 0).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: [usize; 2],
}

impl TensorSpec {
    fn new(name: impl Into<String>, shape: [usize; 2]) -> Self {
        Self { name: name.into(), shape }
    }
    /// Bytes in bf16.
    pub fn bytes(&self) -> u64 {
        let n = if self.shape[1] == 0 { self.shape[0] } else { self.shape[0] * self.shape[1] };
        n as u64 * 2
    }
    /// `true` for a 1-D tensor (a norm gamma or a span embedding).
    pub fn is_vector(&self) -> bool {
        self.shape[1] == 0
    }
}

/// Every tensor the tower owns, in load order: `vision.patch_embed.*`, `vision.blocks.0..31.*`, `vision.norm`,
/// `aligner.*`, then the three span embeddings. 266 tensors, 970,536,960 bytes.
///
/// The names are the pack's, so this doubles as the checklist a `from_regions`-style bind is driven from.
pub fn inventory(v: &VisionConfig) -> Vec<TensorSpec> {
    let (h, i) = (v.hidden, v.intermediate);
    let mut t = Vec::with_capacity(2 + LAYERS * 8 + 1 + 4 + 3);
    t.push(TensorSpec::new("vision.patch_embed.proj.weight", [h, v.patch_dim()]));
    t.push(TensorSpec::new("vision.patch_embed.proj.bias", [h, 0]));
    for l in 0..v.layers {
        let p = format!("vision.blocks.{l}");
        t.push(TensorSpec::new(format!("{p}.attn.wqkv.weight"), [3 * h, h]));
        t.push(TensorSpec::new(format!("{p}.attn.wqkv.bias"), [3 * h, 0]));
        t.push(TensorSpec::new(format!("{p}.attn.wo.weight"), [h, h]));
        t.push(TensorSpec::new(format!("{p}.attn.wo.bias"), [h, 0]));
        t.push(TensorSpec::new(format!("{p}.mlp.w1.weight"), [2 * i, h]));
        t.push(TensorSpec::new(format!("{p}.mlp.w2.weight"), [h, i]));
        t.push(TensorSpec::new(format!("{p}.norm1.weight"), [h, 0]));
        t.push(TensorSpec::new(format!("{p}.norm2.weight"), [h, 0]));
    }
    t.push(TensorSpec::new("vision.norm.weight", [h, 0]));
    t.push(TensorSpec::new("aligner.w1.weight", [OUT_HIDDEN, ALIGN_IN]));
    t.push(TensorSpec::new("aligner.w1.bias", [OUT_HIDDEN, 0]));
    t.push(TensorSpec::new("aligner.w2.weight", [OUT_HIDDEN, OUT_HIDDEN]));
    t.push(TensorSpec::new("aligner.w2.bias", [OUT_HIDDEN, 0]));
    t.push(TensorSpec::new("image_start", [OUT_HIDDEN, 0]));
    t.push(TensorSpec::new("image_end", [OUT_HIDDEN, 0]));
    t.push(TensorSpec::new("image_newline", [OUT_HIDDEN, 0]));
    t
}

/// Bytes of the tower's weights, from the config alone (no 256-byte placement padding: this is the packed size).
pub fn weight_bytes(v: &VisionConfig) -> u64 {
    inventory(v).iter().map(|t| t.bytes()).sum()
}

// ------------------------------------------------------------------------------------------------------------------ the span

/// One image's place in the token stream: where its span starts and what grid it was resized to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanPlan {
    /// Index of the `IMAGE_START` position in `input_ids`.
    pub start: usize,
    /// Patch grid the ViT sees (`best_height / patch_size`, `best_width / patch_size`).
    pub n_vit_h: usize,
    pub n_vit_w: usize,
}

impl SpanPlan {
    /// A span for a `n_vit_h x n_vit_w` patch grid starting at `start`.
    pub fn new(start: usize, n_vit_h: usize, n_vit_w: usize) -> Result<Self, ModelError> {
        if n_vit_h == 0 || n_vit_w == 0 {
            return Err(ModelError::Request("an image has at least one patch".into()));
        }
        Ok(Self { start, n_vit_h, n_vit_w })
    }

    /// `ceil(n_vit_h / 3)`.
    pub fn n_llm_h(&self) -> usize {
        dv::llm_grid(self.n_vit_h)
    }

    /// `ceil(n_vit_w / 3)`.
    pub fn n_llm_w(&self) -> usize {
        dv::llm_grid(self.n_vit_w)
    }

    /// Positions the span occupies, `n_llm_h * (n_llm_w + 1) + 2`.
    pub fn len(&self) -> usize {
        num_image_tokens(self.n_llm_h(), self.n_llm_w())
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// One past the last position.
    pub fn end(&self) -> usize {
        self.start + self.len()
    }

    /// The span's token types, as `image_processor.image_token_types` builds them.
    pub fn types(&self) -> Vec<i8> {
        token_types(self.n_llm_h(), self.n_llm_w())
    }

    /// Where each aligner row and each delimiter lands, `(span_offset, source)` — the same order the splice kernel uses.
    pub fn order(&self) -> Vec<(usize, SpliceSrc)> {
        splice_order(self.n_llm_h(), self.n_llm_w())
    }
}

/// What [`splice_into`] needs to write one image's span.
pub struct SpanFeatures<'a> {
    /// The aligner's output, `[n_llm_h * n_llm_w, OUT_HIDDEN]`.
    pub rows: &'a [f32],
    pub image_start: &'a [f32],
    pub image_end: &'a [f32],
    pub image_newline: &'a [f32],
}

/// `DeepseekV41Model::merge_image_embeddings` on the host: overwrite the span at `plan.start` in `h` `[L, OUT_HIDDEN]`
/// with the aligner's rows in reading order and the three learned embeddings at their delimiters.
///
/// This is the reference's whole splice, and it is stated as one loop over [`SpanPlan::order`] so that the order is a
/// value a test can read rather than a nesting of loops a reader has to unwind.
pub fn splice_into(h: &mut [f32], plan: &SpanPlan, f: &SpanFeatures) -> Result<(), ModelError> {
    let (nh, nw) = (plan.n_llm_h(), plan.n_llm_w());
    let d = OUT_HIDDEN;
    if h.len() < plan.end() * d {
        return Err(ModelError::Request(format!("the stream holds {} rows, the span ends at {}", h.len() / d, plan.end())));
    }
    if f.rows.len() != nh * nw * d {
        return Err(ModelError::Request(format!("{} aligner values for a {nh} x {nw} grid of {d}", f.rows.len())));
    }
    for e in [f.image_start, f.image_end, f.image_newline] {
        if e.len() != d {
            return Err(ModelError::Request(format!("a span embedding has {} values, the stream is {d}", e.len())));
        }
    }
    for (off, src) in plan.order() {
        let dst = &mut h[(plan.start + off) * d..(plan.start + off + 1) * d];
        match src {
            SpliceSrc::Start => dst.copy_from_slice(f.image_start),
            SpliceSrc::End => dst.copy_from_slice(f.image_end),
            SpliceSrc::NewLine => dst.copy_from_slice(f.image_newline),
            SpliceSrc::Aligner(r) => dst.copy_from_slice(&f.rows[r * d..(r + 1) * d]),
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------------ the tower

/// The tower's weights, bound to device buffers by name.
struct Weights {
    patch_w: DevBuf,
    patch_b: DevBuf,
    blocks: Vec<BlockW>,
    norm: DevBuf,
    aligner_w1: DevBuf,
    aligner_b1: DevBuf,
    aligner_w2: DevBuf,
    aligner_b2: DevBuf,
}

struct BlockW {
    norm1: DevBuf,
    norm2: DevBuf,
    qkv_w: DevBuf,
    qkv_b: DevBuf,
    wo_w: DevBuf,
    wo_b: DevBuf,
    w1: DevBuf,
    w2: DevBuf,
}

/// What a tower is built with.
#[derive(Clone, Debug)]
pub struct TowerOptions {
    /// Most patches (before the 3 x 3 merge) one image may have: the scratch is sized for it.
    pub max_patches: usize,
}

impl TowerOptions {
    /// The scratch for images of up to `max_image_tokens` tokens, plus the 3 x 3 window's slack on each axis.
    pub fn new(max_image_tokens: usize) -> Self {
        // a grid of `ceil(h/3) * ceil(w/3) <= max_image_tokens` is at most `3 * sqrt(n) + 2` on a side
        let side = 3 * (max_image_tokens as f64).sqrt().ceil() as usize + 2;
        Self { max_patches: side * side }
    }
}

struct Scratch {
    /// `[P, PATCH_K]` fp16, the patches as the patch GEMM reads them (the pack's bf16 narrowed by the upload).
    pat: DevBuf,
    /// `[P, HIDDEN]` fp32 residual stream.
    x: DevBuf,
    /// `[P, HIDDEN]` fp16, the normed input of the next GEMM.
    h: DevBuf,
    /// `[P, 3 * HIDDEN]` fp16, q | k | v in place.
    qkv: DevBuf,
    /// `[P, HIDDEN]` fp16, the attention output; also the sublayer's raw GEMM output.
    att: DevBuf,
    /// `[P, 2 * INTER]` fp16, gate | up.
    gu: DevBuf,
    /// `[P, ROPE_DIM]` fp32 angles.
    ang: DevBuf,
    /// `[rows, ALIGN_IN]` fp16.
    align: DevBuf,
    ws: DevBuf,
}

/// The DeepSeek-V4.1 vision tower: owns its weights, its scratch and nothing else. One image per call.
pub struct Dsv4VisionTower {
    w: Weights,
    s: Scratch,
    pub image_token_id: u32,
    pub max_patches: usize,
}

impl Dsv4VisionTower {
    /// Bind the tower to buffers a `resolve` callback hands back by pack name (`inventory`'s names), and allocate the
    /// scratch for `o.max_patches`. Every buffer must be bf16 and at least its inventory size.
    ///
    /// A callback rather than a region type because this family's load plan has no vision region yet: whichever commit
    /// adds one resolves names into it and calls this unchanged.
    pub fn bind(
        v: &VisionConfig,
        o: &TowerOptions,
        resolve: &dyn Fn(&str, usize) -> Result<DevBuf, ModelError>,
    ) -> Result<Self, ModelError> {
        check_config(v)?;
        let get = |name: &str, want: usize| -> Result<DevBuf, ModelError> {
            let b = resolve(name, want)?;
            if b.bytes() < want {
                return Err(ModelError::Config(format!("`{name}` is {} B, the tower needs {want}", b.bytes())));
            }
            Ok(b)
        };
        let (h, i) = (HIDDEN, INTER);
        // the rmsnorm kernels read their weight as fp32 (`gb_dsv4vis_rmsnorm` wants `dim * 4` bytes); the pack stores
        // bf16, so the norm tensors are widened once here — 33 x 4 KiB, noise next to the tower's 0.9 GiB
        let widen = |name: &str| -> Result<DevBuf, ModelError> {
            let b = get(name, h * 2)?;
            let bits: Vec<u16> = b.to_vec().map_err(|e| ModelError::Config(format!("{name}: {e}")))?;
            let f: Vec<f32> = bits.iter().map(|&x| f32::from_bits(u32::from(x) << 16)).collect();
            DevBuf::from_slice(&f).map_err(|e| ModelError::Config(format!("{name}: {e}")))
        };
        let mut blocks = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            let p = format!("vision.blocks.{l}");
            blocks.push(BlockW {
                norm1: widen(&format!("{p}.norm1.weight"))?,
                norm2: widen(&format!("{p}.norm2.weight"))?,
                qkv_w: get(&format!("{p}.attn.wqkv.weight"), 3 * h * h * 2)?,
                qkv_b: get(&format!("{p}.attn.wqkv.bias"), 3 * h * 2)?,
                wo_w: get(&format!("{p}.attn.wo.weight"), h * h * 2)?,
                wo_b: get(&format!("{p}.attn.wo.bias"), h * 2)?,
                w1: get(&format!("{p}.mlp.w1.weight"), 2 * i * h * 2)?,
                w2: get(&format!("{p}.mlp.w2.weight"), h * i * 2)?,
            });
        }
        let mut w = Weights {
            patch_w: get("vision.patch_embed.proj.weight", h * PATCH_K * 2)?,
            patch_b: get("vision.patch_embed.proj.bias", h * 2)?,
            blocks,
            norm: widen("vision.norm.weight")?,
            aligner_w1: get("aligner.w1.weight", OUT_HIDDEN * ALIGN_IN * 2)?,
            aligner_b1: get("aligner.w1.bias", OUT_HIDDEN * 2)?,
            aligner_w2: get("aligner.w2.weight", OUT_HIDDEN * OUT_HIDDEN * 2)?,
            aligner_b2: get("aligner.w2.bias", OUT_HIDDEN * 2)?,
        };
        // the dump probe's bind half: the gammas of block 0 as the device holds them post-widen
        if let Some(dir) = std::env::var_os("REXL3_DSV41_DUMP_FEATS") {
            let n1: Vec<f32> = w.blocks[0].norm1.to_vec().unwrap_or_default();
            let n2: Vec<f32> = w.blocks[0].norm2.to_vec().unwrap_or_default();
            let bs: &[u8] = unsafe { core::slice::from_raw_parts(n1.as_ptr().cast::<u8>(), n1.len() * 4) };
            let _ = std::fs::write(std::path::Path::new(&dir).join("bind-norm1.f32"), bs);
            let bs: &[u8] = unsafe { core::slice::from_raw_parts(n2.as_ptr().cast::<u8>(), n2.len() * 4) };
            let _ = std::fs::write(std::path::Path::new(&dir).join("bind-norm2.f32"), bs);
        }
        Self::prepare(&mut w)?;
        let s = Scratch::alloc(o.max_patches)?;
        Ok(Self { w, s, image_token_id: IMAGE_TOKEN_ID, max_patches: o.max_patches })
    }

    /// bf16 -> fp16 in place for every GEMM operand and bias the fork loads through `Linear`. The reference holds the
    /// tower in bf16 and narrows nothing; this build runs fp16 GEMMs over an fp32 stream, so the conversion is once,
    /// here, rather than per call. A value beyond fp16's range fails by name.
    ///
    /// The NORM gammas are NOT in this list: `bind` already widened them to fp32 (`rmsnorm` reads `dim * 4` bytes), and
    /// converting those f32 bits as if they were bf16 pairs would mangle every gamma — 1.0 would come out 0.0078.
    fn prepare(w: &mut Weights) -> Result<(), ModelError> {
        let st = Stream::new()?;
        let mut bad = DevBuf::zeroed(4)?;
        let mut all: Vec<(&str, &mut DevBuf)> = vec![
            ("vision.patch_embed.proj.weight", &mut w.patch_w),
            ("vision.patch_embed.proj.bias", &mut w.patch_b),
            ("aligner.w1.weight", &mut w.aligner_w1),
            ("aligner.w1.bias", &mut w.aligner_b1),
            ("aligner.w2.weight", &mut w.aligner_w2),
            ("aligner.w2.bias", &mut w.aligner_b2),
        ];
        for b in &mut w.blocks {
            all.extend([
                ("attn.wqkv.weight", &mut b.qkv_w),
                ("attn.wqkv.bias", &mut b.qkv_b),
                ("attn.wo.weight", &mut b.wo_w),
                ("attn.wo.bias", &mut b.wo_b),
                ("mlp.w1.weight", &mut b.w1),
                ("mlp.w2.weight", &mut b.w2),
            ]);
        }
        for (name, b) in all {
            let n = b.bytes() / 2;
            dv::bf16_to_f16(b, n, &mut bad, &st).map_err(|e| ModelError::Config(format!("`{name}`: {e}")))?;
        }
        st.sync()?;
        let n: Vec<i32> = bad.to_vec()?;
        if n.first().copied().unwrap_or(1) != 0 {
            return Err(ModelError::Config(format!("{} vision weights are not finite in fp16", n[0])));
        }
        Ok(())
    }

    /// Bytes of the tower's weights on the device.
    pub fn weight_bytes(&self) -> u64 {
        let mut n = self.w.patch_w.bytes() + self.w.patch_b.bytes() + self.w.norm.bytes();
        n += self.w.aligner_w1.bytes() + self.w.aligner_b1.bytes();
        n += self.w.aligner_w2.bytes() + self.w.aligner_b2.bytes();
        for b in &self.w.blocks {
            n += b.norm1.bytes()
                + b.norm2.bytes()
                + b.qkv_w.bytes()
                + b.qkv_b.bytes()
                + b.wo_w.bytes()
                + b.wo_b.bytes()
                + b.w1.bytes()
                + b.w2.bytes();
        }
        n as u64
    }

    /// Bytes of scratch, sized for `max_patches`.
    pub fn scratch_bytes(&self) -> u64 {
        let s = &self.s;
        [&s.pat, &s.x, &s.h, &s.qkv, &s.att, &s.gu, &s.ang, &s.align, &s.ws].iter().map(|b| b.bytes() as u64).sum()
    }

    /// Encode one image: `patches` are its `[P, 588]` values in [`crate::dsv4_vision::patchify`]'s order (bf16 bits, the
    /// pack's own dtype), `grid` its patch grid. `out` receives f32 `[ceil(n_h/3) * ceil(n_w/3), OUT_HIDDEN]`.
    ///
    /// Runs on `st` without synchronising unless a check fails; the host upload reads `patches` before it returns.
    pub fn encode(&mut self, patches: &[u16], grid: (usize, usize), out: &mut DevBuf, st: &Stream) -> Result<(), ModelError> {
        let (gh, gw) = grid;
        let p = gh.checked_mul(gw).unwrap_or(0);
        if p == 0 || p > self.max_patches {
            return Err(ModelError::Request(format!(
                "image of {gh}x{gw} patches; the tower takes up to {} patches",
                self.max_patches
            )));
        }
        if patches.len() != p * PATCH_K {
            return Err(ModelError::Request(format!("{} patch values for a {gh}x{gw} grid of {PATCH_K}", patches.len())));
        }
        let tokens = dv::llm_grid(gh) * dv::llm_grid(gw);
        if out.bytes() < tokens * OUT_HIDDEN * 4 {
            return Err(ModelError::Request("output buffer too small for the image's features".into()));
        }
        let Self { w, s, .. } = self;
        // the dump probe's tower half: the residual after block 0 and the final norm's output, for a host reference
        let tap = std::env::var_os("REXL3_DSV41_DUMP_FEATS").map(std::path::PathBuf::from);
        let mut tap_n = 0;
        // patch embedding: the pack's bf16 patches narrow to fp16 on the way in, then one fp16 GEMM into fp32 with the
        // fp16 bias added in fp32 (the reference's `Linear` on a float tensor, then `y += self.bias`)
        s.pat.upload(patches)?;
        dv::gemm(dv::Epi::StoreF32, &s.pat, &w.patch_w, Some(&w.patch_b), &mut s.x, 0, p, HIDDEN, PATCH_K, PATCH_K, HIDDEN, st)?;
        if let Some(t) = tap.as_deref() {
            st.sync().ok();
            let v: Vec<f32> = s.x.to_vec().unwrap_or_default();
            let b: &[u8] = unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) };
            let _ = std::fs::write(t.join("dump-xpre.f32"), b);
        }
        dv::rope_table(&mut s.ang, gh, gw, ROPE_DIM, ROPE_THETA, st)?;

        for (bi, b) in w.blocks.iter().enumerate() {
            dv::rmsnorm(&s.x, &b.norm1, &mut s.h, p, HIDDEN, RMS_EPS, st)?;
            if bi == 0 {
                if let Some(t) = tap.as_deref() {
                    st.sync().ok();
                    let v: Vec<u16> = s.h.to_vec().unwrap_or_default();
                    let bs: &[u8] = unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 2) };
                    let _ = std::fs::write(t.join("dump-h1.u16"), bs);
                }
            }
            dv::gemm(
                dv::Epi::BiasF16,
                &s.h,
                &b.qkv_w,
                Some(&b.qkv_b),
                &mut s.qkv,
                0,
                p,
                3 * HIDDEN,
                HIDDEN,
                HIDDEN,
                3 * HIDDEN,
                st,
            )?;
            if bi == 0 {
                if let Some(t) = tap.as_deref() {
                    st.sync().ok();
                    let v: Vec<u16> = s.qkv.to_vec().unwrap_or_default();
                    let bs: &[u8] = unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 2) };
                    let _ = std::fs::write(t.join("dump-qkv.u16"), bs);
                }
            }
            dv::rope(&mut s.qkv, &s.ang, p, gw, HEADS, st)?;
            if bi == 0 {
                if let Some(t) = tap.as_deref() {
                    st.sync().ok();
                    let v: Vec<u16> = s.qkv.to_vec().unwrap_or_default();
                    let bs: &[u8] = unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 2) };
                    let _ = std::fs::write(t.join("dump-qkvrope.u16"), bs);
                }
            }
            dv::attention(&s.qkv, &mut s.att, p, st)?;
            if bi == 0 {
                if let Some(t) = tap.as_deref() {
                    st.sync().ok();
                    let v: Vec<u16> = s.att.to_vec().unwrap_or_default();
                    let bs: &[u8] = unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 2) };
                    let _ = std::fs::write(t.join("dump-att.u16"), bs);
                }
            }
            dv::gemm(dv::Epi::BiasAddF32, &s.att, &b.wo_w, Some(&b.wo_b), &mut s.x, 0, p, HIDDEN, HIDDEN, HIDDEN, HIDDEN, st)?;
            dv::rmsnorm(&s.x, &b.norm2, &mut s.h, p, HIDDEN, RMS_EPS, st)?;
            dv::gemm(dv::Epi::BiasF16, &s.h, &b.w1, None, &mut s.gu, 0, p, MLP, HIDDEN, HIDDEN, MLP, st)?;
            dv::silu_mul(&mut s.gu, INTER, MLP, p, st)?;
            // the activation is the first INTER columns of a 2 * INTER-strided row: lda = MLP, not INTER
            dv::gemm(dv::Epi::BiasAddF32, &s.gu, &b.w2, None, &mut s.x, 0, p, HIDDEN, INTER, MLP, HIDDEN, st)?;
            if tap_n == 0 {
                if let Some(t) = tap.as_deref() {
                    st.sync().ok();
                    let x: Vec<f32> = s.x.to_vec().unwrap_or_default();
                    let bytes: &[u8] = unsafe { core::slice::from_raw_parts(x.as_ptr().cast::<u8>(), x.len() * 4) };
                    let _ = std::fs::write(t.join(format!("dump-{tap_n}.x0.f32")), bytes);
                    tap_n += 1;
                }
            }
        }
        dv::rmsnorm(&s.x, &w.norm, &mut s.h, p, HIDDEN, RMS_EPS, st)?;
        if let Some(t) = tap.as_deref() {
            st.sync().ok();
            let h: Vec<u16> = s.h.to_vec().unwrap_or_default();
            let bytes: &[u8] = unsafe { core::slice::from_raw_parts(h.as_ptr().cast::<u8>(), h.len() * 2) };
            let _ = std::fs::write(t.join(format!("dump-{tap_n}.hnorm.u16")), bytes);
        }

        // aligner: the 3 x 3 gather, then w1 (bias, gelu_erf) and w2 (bias) into fp32
        dv::aligner_gather(&s.h, &mut s.align, gh, gw, HIDDEN, st)?;
        dv::gemm(
            dv::Epi::BiasGeluF16,
            &s.align,
            &w.aligner_w1,
            Some(&w.aligner_b1),
            &mut s.att,
            0,
            tokens,
            OUT_HIDDEN,
            ALIGN_IN,
            ALIGN_IN,
            OUT_HIDDEN,
            st,
        )?;
        dv::gemm(
            dv::Epi::StoreF32,
            &s.att,
            &w.aligner_w2,
            Some(&w.aligner_b2),
            out,
            0,
            tokens,
            OUT_HIDDEN,
            OUT_HIDDEN,
            OUT_HIDDEN,
            OUT_HIDDEN,
            st,
        )?;
        Ok(())
    }
}

impl Scratch {
    fn alloc(max_patches: usize) -> Result<Self, ModelError> {
        let p = max_patches as u64;
        let e = |n: u64| -> Result<DevBuf, ModelError> { DevBuf::alloc(n as usize).map_err(ModelError::from) };
        Ok(Self {
            pat: e(p * PATCH_K as u64 * 2)?,
            x: e(p * HIDDEN as u64 * 4)?,
            h: e(p * HIDDEN as u64 * 2)?,
            qkv: e(p * 3 * HIDDEN as u64 * 2)?,
            att: e(p * HIDDEN as u64 * 2)?,
            gu: e(p * MLP as u64 * 2)?,
            ang: e(p * ROPE_DIM as u64 * 4)?,
            align: e(p * MERGE as u64 * MERGE as u64 * HIDDEN as u64 * 2)?,
            ws: e(dv::GEMM_WORKSPACE as u64)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VisionConfig {
        VisionConfig {
            model_type: "deepseek_v4_1_vision".into(),
            layers: 32,
            hidden: 1024,
            heads: 16,
            intermediate: 2816,
            patch_size: 14,
            rope_theta: 10000.0,
            downsample_ratio: 3,
            max_image_tokens: MAX_IMAGE_TOKENS,
            min_pixels: 295_936,
            max_wh_ratio: None,
        }
    }

    /// The three numbers the tower is written for, from the reference and the pack's `config.json` — not from a guess.
    #[test]
    fn the_588_is_three_channels_of_a_14_square_patch() {
        assert_eq!(PATCH_SIZE, 14);
        assert_eq!(PATCH_K, 3 * PATCH_SIZE * PATCH_SIZE);
        assert_eq!(PATCH_K, 588);
        // and it is what the pack's `Linear(3 * patch_size**2, vision_dim)` takes
        assert_eq!(cfg().patch_dim(), 588);
        assert_eq!(ALIGN_IN, 1024 * 9, "the aligner's window is 3 x 3 patches of 1024");
        assert_eq!(MLP, 5632, "w1's output: gate | up of 2816");
    }

    /// The whole 266-tensor list, and its size: the pack's 0.97 GB. A shape that drifted would show here.
    #[test]
    fn the_inventory_is_the_packs_266_tensors_and_0_97_gb() {
        let v = cfg();
        let inv = inventory(&v);
        assert_eq!(inv.len(), 32 * 8 + 2 + 1 + 4 + 3);
        assert_eq!(weight_bytes(&v), 970_536_960);
        let names: std::collections::HashSet<&str> = inv.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names.len(), inv.len(), "every tensor name is distinct");
        for n in [
            "vision.patch_embed.proj.weight",
            "vision.blocks.0.attn.wqkv.weight",
            "vision.blocks.31.attn.wqkv.bias",
            "vision.blocks.31.mlp.w2.weight",
            "vision.norm.weight",
            "aligner.w1.weight",
            "aligner.w2.bias",
            "image_start",
            "image_end",
            "image_newline",
        ] {
            assert!(names.contains(n), "{n} is missing from the inventory");
        }
        let get = |n: &str| inv.iter().find(|t| t.name == n).unwrap().clone();
        assert_eq!(get("vision.patch_embed.proj.weight").shape, [1024, 588]);
        assert_eq!(get("vision.blocks.0.attn.wqkv.weight").shape, [3072, 1024]);
        assert_eq!(get("vision.blocks.0.attn.wqkv.bias").shape, [3072, 0]);
        assert_eq!(get("vision.blocks.0.attn.wo.weight").shape, [1024, 1024]);
        assert_eq!(get("vision.blocks.0.mlp.w1.weight").shape, [5632, 1024]);
        assert_eq!(get("vision.blocks.0.mlp.w2.weight").shape, [1024, 2816]);
        assert_eq!(get("vision.blocks.0.norm1.weight").shape, [1024, 0]);
        assert_eq!(get("aligner.w1.weight").shape, [5120, 9216]);
        assert_eq!(get("aligner.w2.weight").shape, [5120, 5120]);
        for n in ["image_start", "image_end", "image_newline"] {
            assert_eq!(get(n).shape, [5120, 0]);
            assert!(get(n).is_vector());
        }
        // the patch embedding's 588 is the only non-round GEMM in the tower
        assert_eq!(get("vision.patch_embed.proj.weight").bytes(), 1024 * 588 * 2);
    }

    #[test]
    fn a_config_for_another_tower_is_refused_by_name() {
        let mut v = cfg();
        v.hidden = 1152;
        let e = check_config(&v).unwrap_err().to_string();
        assert!(e.contains("hidden_size") && e.contains("1152") && e.contains("1024"), "{e}");
        let mut v = cfg();
        v.downsample_ratio = 2;
        assert!(check_config(&v).unwrap_err().to_string().contains("downsample_ratio"));
        let mut v = cfg();
        v.rope_theta = 500_000.0;
        assert!(check_config(&v).unwrap_err().to_string().contains("rope_theta"));
        // max_image_tokens is a processor budget, not a shape: a different one still runs
        let mut v = cfg();
        v.max_image_tokens = 4096;
        assert!(check_config(&v).is_ok());
    }

    /// An image span is `[START] ([IMAGE] * w [NEW_LINE]) * h [END]`, and its whole length carries the placeholder id.
    #[test]
    fn an_image_span_is_the_processor_layout() {
        let plan = SpanPlan::new(5, 14, 28).unwrap();
        assert_eq!((plan.n_llm_h(), plan.n_llm_w()), (5, 10));
        assert_eq!(plan.len(), 5 * 11 + 2);
        assert_eq!(plan.end(), 5 + 57);
        let t = plan.types();
        assert_eq!(t[0], TokenType::ImageStart as i8);
        assert_eq!(t[1], TokenType::Image as i8);
        assert_eq!(t[1 + 10], TokenType::ImageNewLine as i8);
        assert_eq!(t[t.len() - 1], TokenType::ImageEnd as i8);
        assert_eq!(t.iter().filter(|&&v| v == TokenType::Image as i8).count(), 50);
        assert_eq!(t.iter().filter(|&&v| v == TokenType::ImageNewLine as i8).count(), 5);
        // a 1 x 1 grid (the smallest image) costs six tokens? no: one IMAGE row, so 1 + 2 + 2 = wait — 1*(1+1)+2 = 4
        let one = SpanPlan::new(0, 3, 3).unwrap();
        assert_eq!(one.len(), 4);
        assert_eq!(one.types(), vec![0, 1, 2, 3]);
    }

    /// The engram contract: `None` on a text-only prompt (the fast path), and `false` across every image span otherwise.
    #[test]
    fn an_image_span_produces_exactly_the_token_mask_engram_expects() {
        // text-only: the caller passes no token_types, so there is no mask at all
        assert_eq!(engram_mask(None), None);

        // "hi" + image + "there": the mask is true outside the span and false inside it
        let plan = SpanPlan::new(2, 6, 6).unwrap();
        let mut types = vec![TokenType::Text as i8; 2];
        types.extend(plan.types());
        types.extend([TokenType::Text as i8; 3]);
        let m = engram_mask(Some(&types)).expect("an image is a mask");
        assert_eq!(m.len(), types.len());
        assert_eq!(m[..2], [true, true]);
        assert_eq!(m[2..plan.end()], vec![false; plan.len()]);
        assert_eq!(m[plan.end()..], [true, true, true]);
        // 2 x 2 grid: 4 aligner rows, 2 newlines, start, end
        assert_eq!(plan.len(), 2 * 3 + 2);

        // the consumed form: a dead position blocks every n-gram that spans it, so the text after an image starts clean
        let dead: Vec<i64> = m.iter().map(|&t| if t { 7 } else { super::super::engram::DEAD }).collect();
        assert_eq!(dead[1], 7, "the token before the span");
        assert_eq!(dead[2], super::super::engram::DEAD, "the span's first position");
        assert_eq!(dead[plan.end()], 7, "the token after the span");
        // and the reference's complement, which the text attention reads
        let im = image_mask(&types);
        assert_eq!(im, m.iter().map(|b| !b).collect::<Vec<_>>());
    }

    /// The splice fills the span with the aligner's rows in reading order, and the delimiters with their embeddings.
    #[test]
    fn the_splice_writes_the_aligner_rows_in_reading_order() {
        let plan = SpanPlan::new(1, 6, 9).unwrap(); // 2 x 3 grid -> 6 aligner rows
        assert_eq!((plan.n_llm_h(), plan.n_llm_w()), (2, 3));
        let d = OUT_HIDDEN;
        let mut h = vec![-1f32; (plan.end() + 1) * d];
        // each aligner row is a constant: row r is r + 1, so a misplaced row is obvious
        let rows: Vec<f32> = (0..6).flat_map(|r| std::iter::repeat_n((r + 1) as f32, d)).collect();
        let start = vec![100f32; d];
        let end = vec![200f32; d];
        let nl = vec![300f32; d];
        splice_into(&mut h, &plan, &SpanFeatures { rows: &rows, image_start: &start, image_end: &end, image_newline: &nl })
            .unwrap();

        let row = |i: usize| h[i * d];
        assert_eq!(row(0), -1.0, "before the span");
        assert_eq!(row(1), 100.0, "IMAGE_START");
        assert_eq!([row(2), row(3), row(4)], [1.0, 2.0, 3.0], "row 0 of the aligner grid");
        assert_eq!(row(5), 300.0, "IMAGE_NEW_LINE after row 0");
        assert_eq!([row(6), row(7), row(8)], [4.0, 5.0, 6.0], "row 1");
        assert_eq!(row(9), 300.0);
        assert_eq!(row(10), 200.0, "IMAGE_END");
        assert_eq!(row(11), -1.0, "after the span");
        // every position of the span was written
        assert!(h[plan.start * d..plan.end() * d].iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn a_splice_that_does_not_fit_is_refused_rather_than_truncated() {
        let plan = SpanPlan::new(0, 3, 3).unwrap();
        let d = OUT_HIDDEN;
        let mut h = vec![0f32; (plan.len() - 1) * d];
        let e = vec![0f32; d];
        let rows = vec![0f32; d];
        assert!(
            splice_into(&mut h, &plan, &SpanFeatures { rows: &rows, image_start: &e, image_end: &e, image_newline: &e }).is_err()
        );
        // a feature block of the wrong width is refused too
        let mut h = vec![0f32; plan.len() * d];
        assert!(splice_into(
            &mut h,
            &plan,
            &SpanFeatures { rows: &vec![0f32; d + 1], image_start: &e, image_end: &e, image_newline: &e }
        )
        .is_err());
    }
}
