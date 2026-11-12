//! Kernels of the Nemotron 3.5 Super VL vision path (`csrc/nemotron_vit.cu`): the RADIO ViT-H tower (32 blocks, hidden 1280,
//! 16 heads of 80, exact GELU MLP 5120, LayerNorm with bias and eps 1e-6, qkv bias, layer scale an exact identity), the
//! `vision_final_layernorm`, the pixel shuffle (ps_version v2, ratio 2) and mlp1 (RMSNorm eps 1e-5, relu^2 MLP, no biases).
//!
//! The reference is the pack's own bf16 code (`tools/dump_radio_reference.py`). [`tower_forward`] starts from the normalized
//! pixels as fp16 `[3, H, W]` and reproduces it: the residual stream is fp32, GEMM operands and activations are fp16 with fp32
//! accumulation, and LayerNorm, softmax and RMSNorm statistics are fp32. The position embedding is the pack's square-then-crop
//! resample (see `pos_embed_kernel`), not a direct resample to `(h, w)`.
//!
//! Every function launches on `stream` and returns without synchronising; buffer sizes are checked (a short buffer panics
//! rather than faulting on the device).

pub use gpu::*;

/// Hidden size of the tower, of its LayerNorms and of the features.
pub const HIDDEN: usize = 1280;
/// Attention heads of the tower.
pub const HEADS: usize = 16;
/// Head dimension (`HIDDEN / HEADS`).
pub const HEAD_DIM: usize = 80;
/// Width of the tower's MLP (exact GELU).
pub const MLP_HIDDEN: usize = 5120;
/// Blocks in the tower.
pub const BLOCKS: usize = 32;
/// Patch side in pixels.
pub const PATCH: usize = 16;
/// cls and register tokens prepended to the patches (dropped from the features).
pub const PREFIX: usize = 10;
/// Side of the pack's position-embedding grid.
pub const POS_GRID: usize = 128;
/// Largest patch count the workspace is sized for (the engine's budget).
pub const MAX_PATCHES: usize = 13312;
/// Epsilon of the tower's LayerNorms and of `vision_final_layernorm`.
pub const LN_EPS: f32 = 1e-6;
/// Epsilon of the mlp1.0 RMSNorm.
pub const RMS_EPS: f32 = 1e-5;
/// Width of mlp1's hidden layer.
pub const MLP1_HIDDEN: usize = 20480;
/// Width of mlp1's output (the language model's hidden size).
pub const LLM_HIDDEN: usize = 4096;

/// Planes of a video tubelet: the three channels of two consecutive frames, stacked (`forward_video`'s `T * C`).
pub const VIDEO_PLANES: usize = 6;
/// Width of a video patch vector, `VIDEO_PLANES * PATCH * PATCH`: the K of the video patch embedder's GEMM.
pub const VIDEO_PATCH_DIM: usize = VIDEO_PLANES * PATCH * PATCH;

/// How many tubelets of `patches` patches (each `[PREFIX + patches]` rows of the residual stream) one workspace of `max_patches`
/// patches runs through the blocks in one batch; 0 when one tubelet does not fit. Two limits: the stream's rows
/// (`B * (PREFIX + patches)` against the workspace's `PREFIX + max_patches`) and the staging of one tubelet's six-plane patch matrix in
/// the workspace's `patches` buffer, which is sized for three planes (`patches <= max_patches / 2`). Patch embedding runs one tubelet
/// at a time, so the second limit is per tubelet and not per batch.
pub fn video_batch_capacity(max_patches: usize, patches: usize) -> usize {
    if patches == 0 || 2 * patches > max_patches {
        return 0;
    }
    ((PREFIX + max_patches) / (PREFIX + patches)).min(max_patches / patches)
}

/// Patch grid `(gh, gw)` of an image of `height` x `width` pixels, or `None` when it cannot be run. Both sides must be multiples
/// of 32 (16-pixel patches, and an even grid for the 2 x 2 pixel shuffle), and the patch count must be at most [`MAX_PATCHES`].
pub fn patch_grid(height: usize, width: usize) -> Option<(usize, usize)> {
    if height == 0 || width == 0 || height % (2 * PATCH) != 0 || width % (2 * PATCH) != 0 {
        return None;
    }
    let (gh, gw) = (height / PATCH, width / PATCH);
    (gh * gw <= MAX_PATCHES).then_some((gh, gw))
}

mod gpu {
    use super::*;
    use crate::{CudaError, DevBuf, Stream};
    use core::ffi::c_void;
    use std::error::Error;
    use std::path::Path;

    mod ffi {
        use core::ffi::c_void;
        crate::launchers! {
            pub fn gb_nvit_layernorm(
                x: *const c_void,
                w: *const c_void,
                b: *const c_void,
                y: *mut c_void,
                rows: i32,
                dim: i32,
                eps: f32,
                s: *mut c_void,
            ) -> i32;
            pub fn gb_nvit_rmsnorm(x: *const c_void, w: *const c_void, y: *mut c_void, rows: i32, dim: i32, eps: f32, s: *mut c_void) -> i32;
            pub fn gb_nvit_patchify(pix: *const c_void, out: *mut c_void, h: i32, w: i32, gh: i32, gw: i32, s: *mut c_void) -> i32;
            pub fn gb_nvit_patchify_c(
                pix: *const c_void,
                out: *mut c_void,
                h: i32,
                w: i32,
                gh: i32,
                gw: i32,
                c: i32,
                s: *mut c_void,
            ) -> i32;
            pub fn gb_nvit_pos_embed(h: *mut c_void, pos: *const c_void, cls: *const c_void, gh: i32, gw: i32, s: *mut c_void) -> i32;
            pub fn gb_nvit_gemm(
                epi: i32,
                a: *const c_void,
                b: *const c_void,
                bias: *const c_void,
                c: *mut c_void,
                m: i32,
                n: i32,
                k: i32,
                ldc: i32,
                s: *mut c_void,
            ) -> i32;
            pub fn gb_nvit_pixel_shuffle(x: *const c_void, y: *mut c_void, gh: i32, gw: i32, s: *mut c_void) -> i32;
            pub fn gb_nvit_attention(qkv: *const c_void, out: *mut c_void, t: i32, s: *mut c_void) -> i32;
            pub fn gb_vis_f32_to_bf16(x: *const c_void, y: *mut c_void, n: usize, s: *mut c_void) -> i32;
        }
    }

    fn i(v: usize) -> Result<i32, CudaError> {
        i32::try_from(v).map_err(|_| CudaError(1))
    }

    fn need(b: &DevBuf, off: usize, bytes: usize, what: &str) {
        assert!(off + bytes <= b.bytes(), "nemotron_vit {what}: buffer of {} bytes, need {} at {off}", b.bytes(), bytes);
    }

    /// Epilogue of [`gemm`]: what happens to each output element.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Epi {
        /// `C = fp16(acc + bias)`.
        BiasF16 = 0,
        /// `C = fp16(gelu(acc + bias))`, exact GELU (the tower MLP's fc1).
        BiasGeluF16 = 1,
        /// `C += acc + bias` in fp32, in place (residual adds: attention proj and fc2).
        BiasAddF32 = 2,
        /// `C = acc (+ bias)` in fp32 (patch embedding, mlp1.3).
        StoreF32 = 3,
        /// `C = fp16(relu(acc)^2)` (mlp1.1, which has no bias).
        Relu2F16 = 4,
    }

    impl Epi {
        fn out_bytes(self) -> usize {
            match self {
                Epi::BiasF16 | Epi::BiasGeluF16 | Epi::Relu2F16 => 2,
                Epi::BiasAddF32 | Epi::StoreF32 => 4,
            }
        }
    }

    /// `C[m, n] = A[m, k] * B[n, k]^T` (plus the epilogue), fp16 `A` and `B`, fp32 accumulation. `C` is written at byte offset
    /// `c_off` with row stride `ldc` elements; `bias` (if any) has `n` fp32 values.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        epi: Epi,
        a: &DevBuf,
        b: &DevBuf,
        bias: Option<&DevBuf>,
        c: &mut DevBuf,
        c_off: usize,
        m: usize,
        n: usize,
        k: usize,
        ldc: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(a, 0, m * k * 2, "gemm a");
        need(b, 0, n * k * 2, "gemm b");
        if let Some(bi) = bias {
            need(bi, 0, n * 4, "gemm bias");
        }
        if m > 0 {
            need(c, c_off, ((m - 1) * ldc + n) * epi.out_bytes(), "gemm c");
        }
        let bp = bias.map_or(core::ptr::null(), |x| x.ptr() as *const c_void);
        let cp = if m > 0 { c.ptr_at(c_off) } else { core::ptr::null_mut() };
        CudaError::check(unsafe {
            ffi::gb_nvit_gemm(epi as i32, a.ptr(), b.ptr(), bp, cp, i(m)?, i(n)?, i(k)?, i(ldc)?, st.raw())
        })
    }

    /// `y[r, :] = LayerNorm(x[r, :]) * w + b` over `dim`, eps `eps`, for `rows` rows starting at byte offset `x_off` of the fp32
    /// `x`. `w` and `b` are fp32 `[dim]`; `y` is fp16 `[rows, dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm(
        x: &DevBuf,
        x_off: usize,
        w: &DevBuf,
        b: &DevBuf,
        y: &mut DevBuf,
        rows: usize,
        dim: usize,
        eps: f32,
        st: &Stream,
    ) -> Result<(), CudaError> {
        layernorm_at(x, x_off, w, b, y, 0, rows, dim, eps, st)
    }

    /// [`layernorm`] writing its rows at byte offset `y_off` of `y` (the video path normalises one tubelet's rows at a time into the
    /// rows of a batch).
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm_at(
        x: &DevBuf,
        x_off: usize,
        w: &DevBuf,
        b: &DevBuf,
        y: &mut DevBuf,
        y_off: usize,
        rows: usize,
        dim: usize,
        eps: f32,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(x, x_off, rows * dim * 4, "layernorm x");
        need(w, 0, dim * 4, "layernorm w");
        need(b, 0, dim * 4, "layernorm b");
        need(y, y_off, rows * dim * 2, "layernorm y");
        // the kernel reads x from its own pointer, so the offsets are applied here
        let xp = x.ptr_at(x_off) as *const c_void;
        let yp = y.ptr_at(y_off);
        CudaError::check(unsafe { ffi::gb_nvit_layernorm(xp, w.ptr(), b.ptr(), yp, i(rows)?, i(dim)?, eps, st.raw()) })
    }

    /// `y[r, :] = w * x[r, :] / sqrt(mean(x[r, :]^2) + eps)` for fp16 `x` and `y` `[rows, dim]`, fp32 `w` `[dim]` (mlp1.0).
    pub fn rmsnorm(
        x: &DevBuf,
        w: &DevBuf,
        y: &mut DevBuf,
        rows: usize,
        dim: usize,
        eps: f32,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(x, 0, rows * dim * 2, "rmsnorm x");
        need(w, 0, dim * 4, "rmsnorm w");
        need(y, 0, rows * dim * 2, "rmsnorm y");
        CudaError::check(unsafe { ffi::gb_nvit_rmsnorm(x.ptr(), w.ptr(), y.ptr(), i(rows)?, i(dim)?, eps, st.raw()) })
    }

    /// Patch embedding input: the normalized pixels fp16 `[3, H, W]` to fp16 patches `[gh * gw, 768]`.
    pub fn patchify(
        pix: &DevBuf,
        out: &mut DevBuf,
        height: usize,
        width: usize,
        gh: usize,
        gw: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(pix, 0, 3 * height * width * 2, "patchify pixels");
        need(out, 0, gh * gw * 768 * 2, "patchify out");
        CudaError::check(unsafe { ffi::gb_nvit_patchify(pix.ptr(), out.ptr(), i(height)?, i(width)?, i(gh)?, i(gw)?, st.raw()) })
    }

    /// [`patchify`] for `channels` stacked planes: fp16 `[channels, H, W]` to fp16 patches `[gh * gw, channels * 256]`, the column
    /// order `(channel, row, column)`. `channels` is 3 for an image and 6 for two stacked frames (frame 0's planes, then frame 1's),
    /// the input of a video patch projection (`[HIDDEN, 6 * 256]` weight) in front of the same position embedding.
    #[allow(clippy::too_many_arguments)]
    pub fn patchify_channels(
        pix: &DevBuf,
        out: &mut DevBuf,
        height: usize,
        width: usize,
        gh: usize,
        gw: usize,
        channels: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(pix, 0, channels * height * width * 2, "patchify pixels");
        need(out, 0, gh * gw * channels * PATCH * PATCH * 2, "patchify out");
        CudaError::check(unsafe {
            ffi::gb_nvit_patchify_c(pix.ptr(), out.ptr(), i(height)?, i(width)?, i(gh)?, i(gw)?, i(channels)?, st.raw())
        })
    }

    /// Adds the position embedding to the patch rows of `h` (fp32 `[PREFIX + gh * gw, HIDDEN]`, patches already written by the
    /// embedding GEMM) and writes the cls and register rows from `cls`. `pos` is fp32 `[POS_GRID^2, HIDDEN]`, `cls` fp32
    /// `[PREFIX, HIDDEN]`.
    pub fn pos_embed(h: &mut DevBuf, pos: &DevBuf, cls: &DevBuf, gh: usize, gw: usize, st: &Stream) -> Result<(), CudaError> {
        pos_embed_at(h, 0, pos, cls, gh, gw, st)
    }

    /// [`pos_embed`] for the rows that start at byte offset `h_off` of `h` (tubelet `b` of a video batch starts at row `b * (PREFIX + n)`).
    pub fn pos_embed_at(
        h: &mut DevBuf,
        h_off: usize,
        pos: &DevBuf,
        cls: &DevBuf,
        gh: usize,
        gw: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(h, h_off, (PREFIX + gh * gw) * HIDDEN * 4, "pos_embed h");
        need(pos, 0, POS_GRID * POS_GRID * HIDDEN * 4, "pos_embed table");
        need(cls, 0, PREFIX * HIDDEN * 4, "pos_embed cls");
        CudaError::check(unsafe { ffi::gb_nvit_pos_embed(h.ptr_at(h_off), pos.ptr(), cls.ptr(), i(gh)?, i(gw)?, st.raw()) })
    }

    /// Pixel shuffle, ps_version v2, ratio 2: fp16 `x` `[gh * gw, HIDDEN]` (token-major) to fp16 `y` `[gh * gw / 4, 4 * HIDDEN]`
    /// with the channel order (row in block, column in block, channel).
    pub fn pixel_shuffle(x: &DevBuf, y: &mut DevBuf, gh: usize, gw: usize, st: &Stream) -> Result<(), CudaError> {
        pixel_shuffle_at(x, 0, y, 0, gh, gw, st)
    }

    /// [`pixel_shuffle`] reading `x` at byte offset `x_off` and writing `y` at `y_off`.
    #[allow(clippy::too_many_arguments)]
    pub fn pixel_shuffle_at(
        x: &DevBuf,
        x_off: usize,
        y: &mut DevBuf,
        y_off: usize,
        gh: usize,
        gw: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(x, x_off, gh * gw * HIDDEN * 2, "pixel_shuffle x");
        need(y, y_off, gh * gw / 4 * 4 * HIDDEN * 2, "pixel_shuffle y");
        CudaError::check(unsafe { ffi::gb_nvit_pixel_shuffle(x.ptr_at(x_off), y.ptr_at(y_off), i(gh)?, i(gw)?, st.raw()) })
    }

    /// Full bidirectional attention among the `t` tokens of one image, every head. `qkv` fp16 `[t, 3 * HIDDEN]` (the fused qkv
    /// GEMM's output); `out` fp16 `[t, HIDDEN]`. Softmax in fp32.
    pub fn attention(qkv: &DevBuf, out: &mut DevBuf, t: usize, st: &Stream) -> Result<(), CudaError> {
        attention_at(qkv, 0, out, 0, t, st)
    }

    /// [`attention`] over the `t` rows that start at byte offset `qkv_off` of `qkv` (the output at `out_off`): the tokens of one tubelet
    /// inside a batch, so that no attention crosses a tubelet. `qkv_off` must keep the 16-byte alignment of the fused qkv rows
    /// (`t_row * 7,680`), and `out_off` 4.
    pub fn attention_at(
        qkv: &DevBuf,
        qkv_off: usize,
        out: &mut DevBuf,
        out_off: usize,
        t: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(qkv, qkv_off, t * 3 * HIDDEN * 2, "attention qkv");
        need(out, out_off, t * HIDDEN * 2, "attention out");
        CudaError::check(unsafe { ffi::gb_nvit_attention(qkv.ptr_at(qkv_off), out.ptr_at(out_off), i(t)?, st.raw()) })
    }

    /// `y[y_off..] = bf16(x[x_off..])` over `n` elements (the round-to-nearest-even export `gb_cuda::vision::f32_to_bf16` makes, from and
    /// to an offset): one tubelet's mlp1 rows of a video batch into that tubelet's own feature buffer.
    pub fn f32_to_bf16_at(
        x: &DevBuf,
        x_off: usize,
        y: &mut DevBuf,
        y_off: usize,
        n: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(x, x_off, n * 4, "f32_to_bf16 x");
        need(y, y_off, n * 2, "f32_to_bf16 y");
        CudaError::check(unsafe { ffi::gb_vis_f32_to_bf16(x.ptr_at(x_off), y.ptr_at(y_off), n, st.raw()) })
    }

    /// One tower block's parameters on the device. Matrices are fp16 in the checkpoint's `[out, in]` layout; vectors are fp32.
    pub struct Block {
        ln1_w: DevBuf,
        ln1_b: DevBuf,
        qkv_w: DevBuf,
        qkv_b: DevBuf,
        proj_w: DevBuf,
        proj_b: DevBuf,
        ln2_w: DevBuf,
        ln2_b: DevBuf,
        fc1_w: DevBuf,
        fc1_b: DevBuf,
        fc2_w: DevBuf,
        fc2_b: DevBuf,
    }

    /// The vision tensors the tower, `vision_final_layernorm` and mlp1 use, on the device.
    pub struct Weights {
        patch: DevBuf,
        /// The video patch embedder `[HIDDEN, VIDEO_PATCH_DIM]` (`video_embedder.weight`, fp16): the same projection over a tubelet's
        /// six planes. Everything after it (position embedding, blocks, projector) is the image path's.
        video_patch: DevBuf,
        pos: DevBuf,
        cls: DevBuf,
        blocks: Vec<Block>,
        final_ln_w: DevBuf,
        final_ln_b: DevBuf,
        mlp1_norm: DevBuf,
        mlp1_fc1: DevBuf,
        mlp1_fc2: DevBuf,
    }

    const PG: &str = "vision_model.radio_model.model.patch_generator.";
    const BLK: &str = "vision_model.radio_model.model.blocks.";

    /// bf16 bytes to fp16 bits (each value rounded once, to nearest even).
    fn bf16_to_f16(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|c| gb_exl3::f16::f64_to_f16(crate::num::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64))
            .collect()
    }

    /// bf16 bytes to fp32 values (exact).
    fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes.chunks_exact(2).map(|c| crate::num::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
    }

    impl Weights {
        /// Reads the vision tensors of a Nemotron 3.5 Super VL pack through its `model.safetensors.index.json` (only the
        /// listed shards' headers are read, and only the needed tensors' bytes). Every tensor is checked for dtype and shape.
        pub fn load(pack: &Path) -> Result<Weights, Box<dyn Error>> {
            let ck = gb_exl3::checkpoint::scan(pack, false)?;
            let get = |name: &str, shape: &[usize]| -> Result<Vec<u8>, Box<dyn Error>> {
                let e = ck.plain.get(name).ok_or_else(|| format!("{name}: not a plain tensor of the pack"))?;
                if e.dtype != "BF16" || e.shape != shape {
                    return Err(format!("{name}: {} {:?}, want BF16 {shape:?}", e.dtype, e.shape).into());
                }
                Ok(ck.read_tensor(e)?)
            };
            let mat16 = |name: &str, shape: &[usize]| -> Result<DevBuf, Box<dyn Error>> {
                Ok(DevBuf::from_slice(&bf16_to_f16(&get(name, shape)?))?)
            };
            let vec32 = |name: &str, n: usize| -> Result<DevBuf, Box<dyn Error>> {
                Ok(DevBuf::from_slice(&bf16_to_f32(&get(name, &[n])?))?)
            };
            let patch = mat16(&format!("{PG}embedder.weight"), &[HIDDEN, 3 * PATCH * PATCH])?;
            let video_patch = mat16(&format!("{PG}video_embedder.weight"), &[HIDDEN, VIDEO_PATCH_DIM])?;
            let pos = DevBuf::from_slice(&bf16_to_f32(&get(&format!("{PG}pos_embed"), &[1, POS_GRID * POS_GRID, HIDDEN])?))?;
            let cls = DevBuf::from_slice(&bf16_to_f32(&get(&format!("{PG}cls_token.token"), &[PREFIX, HIDDEN])?))?;
            let mut blocks = Vec::with_capacity(BLOCKS);
            for i in 0..BLOCKS {
                let p = format!("{BLK}{i}.");
                blocks.push(Block {
                    ln1_w: vec32(&format!("{p}norm1.weight"), HIDDEN)?,
                    ln1_b: vec32(&format!("{p}norm1.bias"), HIDDEN)?,
                    qkv_w: mat16(&format!("{p}attn.qkv.weight"), &[3 * HIDDEN, HIDDEN])?,
                    qkv_b: vec32(&format!("{p}attn.qkv.bias"), 3 * HIDDEN)?,
                    proj_w: mat16(&format!("{p}attn.proj.weight"), &[HIDDEN, HIDDEN])?,
                    proj_b: vec32(&format!("{p}attn.proj.bias"), HIDDEN)?,
                    ln2_w: vec32(&format!("{p}norm2.weight"), HIDDEN)?,
                    ln2_b: vec32(&format!("{p}norm2.bias"), HIDDEN)?,
                    fc1_w: mat16(&format!("{p}mlp.fc1.weight"), &[MLP_HIDDEN, HIDDEN])?,
                    fc1_b: vec32(&format!("{p}mlp.fc1.bias"), MLP_HIDDEN)?,
                    fc2_w: mat16(&format!("{p}mlp.fc2.weight"), &[HIDDEN, MLP_HIDDEN])?,
                    fc2_b: vec32(&format!("{p}mlp.fc2.bias"), HIDDEN)?,
                });
            }
            Ok(Weights {
                patch,
                video_patch,
                pos,
                cls,
                blocks,
                final_ln_w: vec32("vision_projector.vision_final_layernorm.weight", HIDDEN)?,
                final_ln_b: vec32("vision_projector.vision_final_layernorm.bias", HIDDEN)?,
                mlp1_norm: vec32("mlp1.0.weight", 4 * HIDDEN)?,
                mlp1_fc1: mat16("mlp1.1.weight", &[MLP1_HIDDEN, 4 * HIDDEN])?,
                mlp1_fc2: mat16("mlp1.3.weight", &[LLM_HIDDEN, MLP1_HIDDEN])?,
            })
        }
    }

    impl Weights {
        /// Device bytes of every tensor the tower, `vision_final_layernorm` and mlp1 hold (matrices fp16, vectors and the position table
        /// fp32): what the memory model prices, by name `weights_bytes` in `families::nemotron_h_omni::vision`.
        pub fn bytes(&self) -> u64 {
            let mut n = [
                &self.patch,
                &self.video_patch,
                &self.pos,
                &self.cls,
                &self.final_ln_w,
                &self.final_ln_b,
                &self.mlp1_norm,
                &self.mlp1_fc1,
                &self.mlp1_fc2,
            ]
            .iter()
            .map(|b| b.bytes() as u64)
            .sum::<u64>();
            for b in &self.blocks {
                n += [
                    &b.ln1_w, &b.ln1_b, &b.qkv_w, &b.qkv_b, &b.proj_w, &b.proj_b, &b.ln2_w, &b.ln2_b, &b.fc1_w, &b.fc1_b,
                    &b.fc2_w, &b.fc2_b,
                ]
                .iter()
                .map(|x| x.bytes() as u64)
                .sum::<u64>();
            }
            n
        }
    }

    /// Activation buffers, sized for up to `max_patches` patches. Reuse one across images.
    pub struct Workspace {
        max_patches: usize,
        patches: DevBuf,
        /// fp32 `[PREFIX + patches, HIDDEN]`: the residual stream. Rows `PREFIX..` after [`tower_forward`] are the tower features
        /// (before `vision_final_layernorm`), 4 bytes per element.
        h: DevBuf,
        xn: DevBuf,
        qkv: DevBuf,
        att: DevBuf,
        mid: DevBuf,
        lnf: DevBuf,
        ps: DevBuf,
        ps_norm: DevBuf,
        mlp1_mid: DevBuf,
        mlp1_out: DevBuf,
    }

    /// Byte offset of the tower features inside [`Workspace::h`].
    pub const FEATURE_OFFSET: usize = PREFIX * HIDDEN * 4;

    impl Workspace {
        /// Buffers for images of up to `max_patches` patches (a multiple of 4, at most [`MAX_PATCHES`]).
        pub fn new(max_patches: usize) -> Result<Workspace, CudaError> {
            assert!(max_patches > 0 && max_patches % 4 == 0 && max_patches <= MAX_PATCHES, "max_patches {max_patches}");
            let t = PREFIX + max_patches;
            let np = max_patches / 4;
            Ok(Workspace {
                max_patches,
                patches: DevBuf::zeroed(max_patches * 3 * PATCH * PATCH * 2)?,
                h: DevBuf::zeroed(t * HIDDEN * 4)?,
                xn: DevBuf::zeroed(t * HIDDEN * 2)?,
                qkv: DevBuf::zeroed(t * 3 * HIDDEN * 2)?,
                att: DevBuf::zeroed(t * HIDDEN * 2)?,
                mid: DevBuf::zeroed(t * MLP_HIDDEN * 2)?,
                lnf: DevBuf::zeroed(max_patches * HIDDEN * 2)?,
                ps: DevBuf::zeroed(np * 4 * HIDDEN * 2)?,
                ps_norm: DevBuf::zeroed(np * 4 * HIDDEN * 2)?,
                mlp1_mid: DevBuf::zeroed(np * MLP1_HIDDEN * 2)?,
                mlp1_out: DevBuf::zeroed(np * LLM_HIDDEN * 4)?,
            })
        }

        /// The residual stream (fp32). After [`tower_forward`] the features are the rows from [`FEATURE_OFFSET`] on.
        pub fn h(&self) -> &DevBuf {
            &self.h
        }

        /// vision_final_layernorm's output (fp16 `[patches, HIDDEN]`) after [`tower_forward`].
        pub fn final_ln(&self) -> &DevBuf {
            &self.lnf
        }

        /// mlp1's output (fp32 `[patches / 4, LLM_HIDDEN]`) after [`tower_forward`].
        pub fn mlp1(&self) -> &DevBuf {
            &self.mlp1_out
        }

        /// Patch capacity this workspace was sized for.
        pub fn max_patches(&self) -> usize {
            self.max_patches
        }

        /// Device bytes of every buffer of this workspace.
        pub fn bytes(&self) -> u64 {
            [
                &self.patches,
                &self.h,
                &self.xn,
                &self.qkv,
                &self.att,
                &self.mid,
                &self.lnf,
                &self.ps,
                &self.ps_norm,
                &self.mlp1_mid,
                &self.mlp1_out,
            ]
            .iter()
            .map(|b| b.bytes() as u64)
            .sum()
        }
    }

    /// Optional per-stage readout of [`tower_forward`], for comparing against the reference's stages. Every row is read
    /// after a stream sync, so this slows the forward; use it in tests only.
    #[derive(Debug, Default, Clone)]
    pub struct Trace {
        /// `embeddings`, then the 32 block outputs (33 entries): each `[3, HIDDEN]` f32 holding token 0 (cls), token `PREFIX`
        /// (first patch) and the last token, in that order.
        pub blocks: Vec<Vec<f32>>,
        /// `[2, HIDDEN]`: the first and the last token of vision_final_layernorm's output.
        pub final_ln: Vec<f32>,
        /// `[2, 4 * HIDDEN]`: the first and the last token of the pixel shuffle.
        pub pixel_shuffle: Vec<f32>,
        /// `[2, LLM_HIDDEN]`: the first and the last token of mlp1.
        pub mlp1: Vec<f32>,
    }

    fn rows_of_h(h: &DevBuf, rows: &[usize], st: &Stream) -> Result<Vec<f32>, CudaError> {
        st.sync()?;
        let mut out = vec![0f32; rows.len() * HIDDEN];
        for (k, &r) in rows.iter().enumerate() {
            h.download_range(r * HIDDEN * 4, &mut out[k * HIDDEN..(k + 1) * HIDDEN])?;
        }
        Ok(out)
    }

    fn rows_of_f16(x: &DevBuf, row_elems: usize, rows: &[usize], st: &Stream) -> Result<Vec<f32>, CudaError> {
        st.sync()?;
        let mut out = Vec::with_capacity(rows.len() * row_elems);
        let mut tmp = vec![0u16; row_elems];
        for &r in rows {
            x.download_range(r * row_elems * 2, &mut tmp)?;
            out.extend(tmp.iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32));
        }
        Ok(out)
    }

    /// One stage of a [`Profile`]: all the calls of one named stage of the forward, summed.
    #[derive(Debug, Clone, PartialEq)]
    pub struct StageTime {
        /// `patchify`, `patch_gemm`, `pos_embed`, then per block `ln1`, `qkv`, `attn`, `proj`, `ln2`, `fc1`, `fc2` (32 calls each),
        /// then `final_ln`, `pixel_shuffle`, `mlp1_rms`, `mlp1_fc1`, `mlp1_fc2`.
        pub name: &'static str,
        pub calls: u32,
        /// GPU time from the previous stage's end to this one's, summed over the calls.
        pub us: f64,
    }

    /// Per-stage GPU time of one [`tower_forward_profiled`] call, from CUDA events recorded between the stages (the events
    /// add no device work and the stream is not synchronised until the end).
    #[derive(Debug, Default, Clone)]
    pub struct Profile {
        /// In order of first appearance.
        pub stages: Vec<StageTime>,
        /// First event to last event, microseconds.
        pub total_us: f64,
    }

    impl Profile {
        /// The summed time of the stage `name` (0 when it did not run).
        pub fn us(&self, name: &str) -> f64 {
            self.stages.iter().find(|s| s.name == name).map_or(0.0, |s| s.us)
        }
    }

    /// Timing events are never destroyed by the runtime wrappers (`rt::timed_events` has no destroy), so a pool bounds what a
    /// process holds at the most one profile ever used.
    static EVENT_POOL: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

    struct Rec {
        events: Vec<usize>,
        names: Vec<&'static str>,
    }

    impl Rec {
        fn event(&mut self) -> Result<usize, CudaError> {
            if let Some(e) = EVENT_POOL.lock().unwrap().pop() {
                return Ok(e);
            }
            let mut e: *mut c_void = core::ptr::null_mut();
            CudaError::check(unsafe { crate::rt::timed_events::create(&mut e) })?;
            Ok(e as usize)
        }

        fn new(st: &Stream) -> Result<Rec, CudaError> {
            let mut r = Rec { events: Vec::new(), names: Vec::new() };
            let e = r.event()?;
            r.events.push(e);
            CudaError::check(unsafe { crate::rt::timed_events::record(e as *mut c_void, st.raw()) })?;
            Ok(r)
        }

        /// Ends the stage `name` here.
        fn mark(&mut self, name: &'static str, st: &Stream) -> Result<(), CudaError> {
            let e = self.event()?;
            self.events.push(e);
            self.names.push(name);
            CudaError::check(unsafe { crate::rt::timed_events::record(e as *mut c_void, st.raw()) })
        }

        fn finish(self) -> Result<Profile, CudaError> {
            let last = *self.events.last().unwrap();
            let mut res = CudaError::check(unsafe { crate::rt::timed_events::synchronize(last as *mut c_void) });
            let mut prof = Profile::default();
            if res.is_ok() {
                for (k, &name) in self.names.iter().enumerate() {
                    let mut ms = 0f32;
                    let rc = unsafe {
                        crate::rt::timed_events::elapsed_ms(
                            &mut ms,
                            self.events[k] as *mut c_void,
                            self.events[k + 1] as *mut c_void,
                        )
                    };
                    if rc != 0 {
                        res = Err(CudaError(rc));
                        break;
                    }
                    let us = ms as f64 * 1000.0;
                    prof.total_us += us;
                    match prof.stages.iter_mut().find(|s| s.name == name) {
                        Some(s) => {
                            s.calls += 1;
                            s.us += us;
                        }
                        None => prof.stages.push(StageTime { name, calls: 1, us }),
                    }
                }
            }
            EVENT_POOL.lock().unwrap().extend(self.events);
            res.map(|_| prof)
        }
    }

    /// The patch embedding with the position embedding: `patchify` (fp16 `[3, H, W]` pixels to `[gh * gw, 768]`), the embedder
    /// GEMM (fp32 out into rows `PREFIX..` of `ws.h`), then the cls rows and the position embedding. A 6-channel (two stacked
    /// frames) projection is the same three steps with `[6, H, W]` pixels, `[gh * gw, 1536]` patches and a `[HIDDEN, 1536]`
    /// weight in the GEMM; the position embedding is the same.
    #[allow(clippy::too_many_arguments)]
    fn patch_embed(
        w: &Weights,
        ws: &mut Workspace,
        pixels: &DevBuf,
        height: usize,
        width: usize,
        gh: usize,
        gw: usize,
        mut rec: Option<&mut Rec>,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let n = gh * gw;
        patchify(pixels, &mut ws.patches, height, width, gh, gw, st)?;
        if let Some(r) = rec.as_deref_mut() {
            r.mark("patchify", st)?;
        }
        gemm(
            Epi::StoreF32,
            &ws.patches,
            &w.patch,
            None,
            &mut ws.h,
            PREFIX * HIDDEN * 4,
            n,
            HIDDEN,
            3 * PATCH * PATCH,
            HIDDEN,
            st,
        )?;
        if let Some(r) = rec.as_deref_mut() {
            r.mark("patch_gemm", st)?;
        }
        pos_embed(&mut ws.h, &w.pos, &w.cls, gh, gw, st)?;
        if let Some(r) = rec {
            r.mark("pos_embed", st)?;
        }
        Ok(())
    }

    /// The vision tower from the pixels: fp16 normalized `pixels` `[3, height, width]` through the patch embedding (with the
    /// square-then-crop position embedding), 32 blocks, vision_final_layernorm (its output in [`Workspace::final_ln`]), the
    /// pixel shuffle and mlp1. The tower features (fp32, prefix tokens dropped) are the rows of [`Workspace::h`] from
    /// [`FEATURE_OFFSET`] on, and mlp1's output is [`Workspace::mlp1`]. `height` and `width` must pass [`patch_grid`].
    pub fn tower_forward(
        w: &Weights,
        ws: &mut Workspace,
        pixels: &DevBuf,
        height: usize,
        width: usize,
        trace: Option<&mut Trace>,
        st: &Stream,
    ) -> Result<(), CudaError> {
        forward(w, ws, pixels, height, width, trace, None, st)
    }

    /// [`tower_forward`] with the GPU time of every stage read back from events (the stream is synchronised once, at the end).
    /// The launches are the same, so the total is the forward's own.
    pub fn tower_forward_profiled(
        w: &Weights,
        ws: &mut Workspace,
        pixels: &DevBuf,
        height: usize,
        width: usize,
        st: &Stream,
    ) -> Result<Profile, CudaError> {
        let mut rec = Rec::new(st)?;
        forward(w, ws, pixels, height, width, None, Some(&mut rec), st)?;
        rec.finish()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        w: &Weights,
        ws: &mut Workspace,
        pixels: &DevBuf,
        height: usize,
        width: usize,
        mut trace: Option<&mut Trace>,
        mut rec: Option<&mut Rec>,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let (gh, gw) =
            patch_grid(height, width).unwrap_or_else(|| panic!("tower_forward: {height}x{width} is not a runnable size"));
        let n = gh * gw;
        assert!(n <= ws.max_patches, "tower_forward: {n} patches, the workspace holds {}", ws.max_patches);
        let t = PREFIX + n;
        let np = n / 4;
        if let Some(tr) = trace.as_deref_mut() {
            *tr = Trace::default();
        }
        macro_rules! mark {
            ($name:expr) => {
                if let Some(r) = rec.as_deref_mut() {
                    r.mark($name, st)?;
                }
            };
        }
        patch_embed(w, ws, pixels, height, width, gh, gw, rec.as_deref_mut(), st)?;
        let stage_rows = [0, PREFIX, t - 1];
        if let Some(tr) = trace.as_deref_mut() {
            tr.blocks.push(rows_of_h(&ws.h, &stage_rows, st)?);
        }
        for blk in &w.blocks {
            // attention: LN, fused qkv (bias), full attention, output projection (bias, residual)
            layernorm(&ws.h, 0, &blk.ln1_w, &blk.ln1_b, &mut ws.xn, t, HIDDEN, LN_EPS, st)?;
            mark!("ln1");
            gemm(Epi::BiasF16, &ws.xn, &blk.qkv_w, Some(&blk.qkv_b), &mut ws.qkv, 0, t, 3 * HIDDEN, HIDDEN, 3 * HIDDEN, st)?;
            mark!("qkv");
            attention(&ws.qkv, &mut ws.att, t, st)?;
            mark!("attn");
            gemm(Epi::BiasAddF32, &ws.att, &blk.proj_w, Some(&blk.proj_b), &mut ws.h, 0, t, HIDDEN, HIDDEN, HIDDEN, st)?;
            mark!("proj");
            // MLP: LN, fc1 with exact GELU, fc2 (bias, residual)
            layernorm(&ws.h, 0, &blk.ln2_w, &blk.ln2_b, &mut ws.xn, t, HIDDEN, LN_EPS, st)?;
            mark!("ln2");
            gemm(Epi::BiasGeluF16, &ws.xn, &blk.fc1_w, Some(&blk.fc1_b), &mut ws.mid, 0, t, MLP_HIDDEN, HIDDEN, MLP_HIDDEN, st)?;
            mark!("fc1");
            gemm(Epi::BiasAddF32, &ws.mid, &blk.fc2_w, Some(&blk.fc2_b), &mut ws.h, 0, t, HIDDEN, MLP_HIDDEN, HIDDEN, st)?;
            mark!("fc2");
            if let Some(tr) = trace.as_deref_mut() {
                tr.blocks.push(rows_of_h(&ws.h, &stage_rows, st)?);
            }
        }
        // vision_final_layernorm over the features (prefix dropped), pixel shuffle v2, mlp1
        layernorm(&ws.h, FEATURE_OFFSET, &w.final_ln_w, &w.final_ln_b, &mut ws.lnf, n, HIDDEN, LN_EPS, st)?;
        mark!("final_ln");
        if let Some(tr) = trace.as_deref_mut() {
            tr.final_ln = rows_of_f16(&ws.lnf, HIDDEN, &[0, n - 1], st)?;
        }
        pixel_shuffle(&ws.lnf, &mut ws.ps, gh, gw, st)?;
        mark!("pixel_shuffle");
        if let Some(tr) = trace.as_deref_mut() {
            tr.pixel_shuffle = rows_of_f16(&ws.ps, 4 * HIDDEN, &[0, np - 1], st)?;
        }
        rmsnorm(&ws.ps, &w.mlp1_norm, &mut ws.ps_norm, np, 4 * HIDDEN, RMS_EPS, st)?;
        mark!("mlp1_rms");
        gemm(Epi::Relu2F16, &ws.ps_norm, &w.mlp1_fc1, None, &mut ws.mlp1_mid, 0, np, MLP1_HIDDEN, 4 * HIDDEN, MLP1_HIDDEN, st)?;
        mark!("mlp1_fc1");
        gemm(Epi::StoreF32, &ws.mlp1_mid, &w.mlp1_fc2, None, &mut ws.mlp1_out, 0, np, LLM_HIDDEN, MLP1_HIDDEN, LLM_HIDDEN, st)?;
        mark!("mlp1_fc2");
        if let Some(tr) = trace {
            st.sync()?;
            let mut rows = vec![0f32; 2 * LLM_HIDDEN];
            ws.mlp1_out.download_range(0, &mut rows[..LLM_HIDDEN])?;
            ws.mlp1_out.download_range((np - 1) * LLM_HIDDEN * 4, &mut rows[LLM_HIDDEN..])?;
            tr.mlp1 = rows;
        }
        Ok(())
    }

    // ----------------------------------------------------------------------------------------------------------------- video

    /// The patch embedding of tubelet `tubelet` of a video batch: fp16 normalized `pixels` `[VIDEO_PLANES, height, width]` (frame 0's
    /// planes, then frame 1's) through the six-plane patchify, the `[HIDDEN, VIDEO_PATCH_DIM]` embedder GEMM (K = 1,536) and the
    /// position embedding, into the tubelet's block of the residual stream: rows `tubelet * (PREFIX + n)` of `ws.h` are its cls and
    /// register rows, then its `n` patches. The tubelet's patch matrix is staged in the workspace's `patches` buffer, which is sized for
    /// three planes, so `n` must be at most `max_patches / 2` (see [`video_batch_capacity`]). Call it once per tubelet of the batch, then
    /// [`video_forward`].
    pub fn video_embed(
        w: &Weights,
        ws: &mut Workspace,
        pixels: &DevBuf,
        height: usize,
        width: usize,
        tubelet: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let (gh, gw) =
            patch_grid(height, width).unwrap_or_else(|| panic!("video_embed: {height}x{width} is not a runnable size"));
        let n = gh * gw;
        assert!(
            2 * n <= ws.max_patches,
            "video_embed: a {n}-patch tubelet needs a workspace of {} patches, this one holds {}",
            2 * n,
            ws.max_patches
        );
        let t = PREFIX + n;
        assert!((tubelet + 1) * t * HIDDEN * 4 <= ws.h.bytes(), "video_embed: tubelet {tubelet} is past the residual stream");
        patchify_channels(pixels, &mut ws.patches, height, width, gh, gw, VIDEO_PLANES, st)?;
        let at = (tubelet * t + PREFIX) * HIDDEN * 4;
        gemm(Epi::StoreF32, &ws.patches, &w.video_patch, None, &mut ws.h, at, n, HIDDEN, VIDEO_PATCH_DIM, HIDDEN, st)?;
        pos_embed_at(&mut ws.h, tubelet * t * HIDDEN * 4, &w.pos, &w.cls, gh, gw, st)
    }

    /// The 32 blocks, `vision_final_layernorm`, the pixel shuffle and mlp1 over `batch` tubelets whose patch embeddings
    /// ([`video_embed`], tubelets `0..batch`) are in `ws.h`, each of `height x width` pixels.
    ///
    /// The row-wise stages run once over all `batch * (PREFIX + n)` rows (LayerNorm, the qkv, proj, fc1 and fc2 GEMMs) and mlp1 once over
    /// `batch * n / 4` rows; attention runs per tubelet (a launch over each tubelet's `PREFIX + n` rows, so nothing attends across
    /// tubelets), as do the final LayerNorm and the pixel shuffle, whose rows skip the prefix. Every output element is computed by the
    /// same operations in the same order as a batch of one would compute it (the GEMM accumulates each output over `k` alone), so a
    /// tubelet's rows do not depend on the batch it ran in. The features are the rows `tubelet * (PREFIX + n) + PREFIX ..` of `ws.h`,
    /// and mlp1's output is rows `tubelet * n / 4 ..` of [`Workspace::mlp1`] (fp32). `traces`, when given, receives one [`Trace`] per
    /// tubelet (it synchronises; tests only).
    pub fn video_forward(
        w: &Weights,
        ws: &mut Workspace,
        height: usize,
        width: usize,
        batch: usize,
        mut traces: Option<&mut Vec<Trace>>,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let (gh, gw) =
            patch_grid(height, width).unwrap_or_else(|| panic!("video_forward: {height}x{width} is not a runnable size"));
        let n = gh * gw;
        let capacity = video_batch_capacity(ws.max_patches, n);
        assert!(
            (1..=capacity).contains(&batch),
            "video_forward: {batch} tubelets of {n} patches in a workspace of {} (it holds {capacity})",
            ws.max_patches
        );
        let t = PREFIX + n;
        let np = n / 4;
        let rows = batch * t;
        if let Some(tr) = traces.as_deref_mut() {
            tr.clear();
            tr.resize_with(batch, Trace::default);
            for (b, one) in tr.iter_mut().enumerate() {
                one.blocks.push(rows_of_h(&ws.h, &[b * t, b * t + PREFIX, b * t + t - 1], st)?);
            }
        }
        for blk in &w.blocks {
            layernorm(&ws.h, 0, &blk.ln1_w, &blk.ln1_b, &mut ws.xn, rows, HIDDEN, LN_EPS, st)?;
            gemm(Epi::BiasF16, &ws.xn, &blk.qkv_w, Some(&blk.qkv_b), &mut ws.qkv, 0, rows, 3 * HIDDEN, HIDDEN, 3 * HIDDEN, st)?;
            for b in 0..batch {
                attention_at(&ws.qkv, b * t * 3 * HIDDEN * 2, &mut ws.att, b * t * HIDDEN * 2, t, st)?;
            }
            gemm(Epi::BiasAddF32, &ws.att, &blk.proj_w, Some(&blk.proj_b), &mut ws.h, 0, rows, HIDDEN, HIDDEN, HIDDEN, st)?;
            layernorm(&ws.h, 0, &blk.ln2_w, &blk.ln2_b, &mut ws.xn, rows, HIDDEN, LN_EPS, st)?;
            gemm(
                Epi::BiasGeluF16,
                &ws.xn,
                &blk.fc1_w,
                Some(&blk.fc1_b),
                &mut ws.mid,
                0,
                rows,
                MLP_HIDDEN,
                HIDDEN,
                MLP_HIDDEN,
                st,
            )?;
            gemm(Epi::BiasAddF32, &ws.mid, &blk.fc2_w, Some(&blk.fc2_b), &mut ws.h, 0, rows, HIDDEN, MLP_HIDDEN, HIDDEN, st)?;
            if let Some(tr) = traces.as_deref_mut() {
                for (b, one) in tr.iter_mut().enumerate() {
                    one.blocks.push(rows_of_h(&ws.h, &[b * t, b * t + PREFIX, b * t + t - 1], st)?);
                }
            }
        }
        for b in 0..batch {
            let (lnf_off, ps_off) = (b * n * HIDDEN * 2, b * np * 4 * HIDDEN * 2);
            let x_off = (b * t + PREFIX) * HIDDEN * 4;
            layernorm_at(&ws.h, x_off, &w.final_ln_w, &w.final_ln_b, &mut ws.lnf, lnf_off, n, HIDDEN, LN_EPS, st)?;
            pixel_shuffle_at(&ws.lnf, lnf_off, &mut ws.ps, ps_off, gh, gw, st)?;
        }
        if let Some(tr) = traces.as_deref_mut() {
            for (b, one) in tr.iter_mut().enumerate() {
                one.final_ln = rows_of_f16(&ws.lnf, HIDDEN, &[b * n, b * n + n - 1], st)?;
                one.pixel_shuffle = rows_of_f16(&ws.ps, 4 * HIDDEN, &[b * np, b * np + np - 1], st)?;
            }
        }
        rmsnorm(&ws.ps, &w.mlp1_norm, &mut ws.ps_norm, batch * np, 4 * HIDDEN, RMS_EPS, st)?;
        let m = batch * np;
        gemm(Epi::Relu2F16, &ws.ps_norm, &w.mlp1_fc1, None, &mut ws.mlp1_mid, 0, m, MLP1_HIDDEN, 4 * HIDDEN, MLP1_HIDDEN, st)?;
        gemm(Epi::StoreF32, &ws.mlp1_mid, &w.mlp1_fc2, None, &mut ws.mlp1_out, 0, m, LLM_HIDDEN, MLP1_HIDDEN, LLM_HIDDEN, st)?;
        if let Some(tr) = traces {
            st.sync()?;
            for (b, one) in tr.iter_mut().enumerate() {
                let mut rows = vec![0f32; 2 * LLM_HIDDEN];
                ws.mlp1_out.download_range(b * np * LLM_HIDDEN * 4, &mut rows[..LLM_HIDDEN])?;
                ws.mlp1_out.download_range((b * np + np - 1) * LLM_HIDDEN * 4, &mut rows[LLM_HIDDEN..])?;
                one.mlp1 = rows;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_grid_accepts_even_grids_within_the_budget() {
        assert_eq!(patch_grid(512, 512), Some((32, 32)));
        assert_eq!(patch_grid(320, 448), Some((20, 28)));
        assert_eq!(patch_grid(448, 320), Some((28, 20)));
        assert_eq!(patch_grid(16, 512), None, "odd patch grid cannot be pixel-shuffled");
        assert_eq!(patch_grid(0, 512), None);
        assert_eq!(patch_grid(512 + 8, 512), None, "sides must be multiples of 32");
        // 13312 patches is the budget: 104 x 128 patches = 13312 is allowed, one more even row is not
        assert_eq!(patch_grid(104 * 16, 128 * 16), Some((104, 128)));
        assert_eq!(patch_grid(106 * 16, 126 * 16), None);
    }

    #[test]
    fn a_video_batch_is_what_the_stream_and_the_patch_staging_hold() {
        // the default cap: a 1,008-patch tile (16:9 or 4:3) runs 13 tubelets at once, a 1,024-patch square tile 12
        assert_eq!(video_batch_capacity(MAX_PATCHES, 1008), 13);
        assert_eq!(video_batch_capacity(MAX_PATCHES, 1024), 12);
        assert_eq!(video_batch_capacity(MAX_PATCHES, 1040), 12);
        // the six-plane patch matrix of one tubelet must fit the three-plane staging: n <= max_patches / 2
        assert_eq!(video_batch_capacity(MAX_PATCHES, MAX_PATCHES / 2), 1);
        assert_eq!(video_batch_capacity(MAX_PATCHES, MAX_PATCHES / 2 + 4), 0);
        // --max-image-tokens 256 is 1,024 patches: no video tile fits
        assert_eq!(video_batch_capacity(1024, 1008), 0);
        assert_eq!(video_batch_capacity(2048, 1008), 2);
        assert_eq!(video_batch_capacity(MAX_PATCHES, 0), 0);
        assert_eq!(VIDEO_PATCH_DIM, 1536);
    }

    #[test]
    fn constants_match_the_pack() {
        assert_eq!(HIDDEN / HEADS, HEAD_DIM);
        assert_eq!(MLP_HIDDEN, 4 * HIDDEN);
        assert_eq!(4 * HIDDEN, 5120, "pixel shuffle channels feed mlp1.0 at 5120");
        assert_eq!(MAX_PATCHES % 4, 0);
    }
}
