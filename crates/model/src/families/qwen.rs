//! The Qwen3.5/3.8 family: dense feed-forward, grouped-query attention interleaved with gated delta net, a vision tower, DFlash2
//! drafting. This wraps what the engine has always run: the configuration is `config::QwenConfig`, the engine is
//! `serve::build_engine`, and the memory model below reproduces the planner's numbers exactly (`plan::tests` pins them).

use crate::config::QwenConfig;
use crate::error::ModelError;
use crate::family::{BuildCtx, FamilyError, FamilyPlan, MemorySource, ModelFamily, VisionSpec};
use crate::kvtables::BLOCK_TOKENS;
use crate::memory::{MemoryOptions, VisionOptions};
use gb_engine::ServeEngine;
use gb_sched::{KvLayerClass, KvLayout, MemoryError, MemoryModel, MemoryProfile, StepShape, VisionMemory};
use serde_json::Value;
use std::path::Path;

/// What the Qwen engine holds back for everything it allocates after the plan (activation scratch, window records, kernel
/// workspaces). Subtracted once, by the planner, exactly where `build_engine` used to subtract its `scratch_reserve`.
pub const ACTIVATION_RESERVE: u64 = 5 << 30;
/// Per sequence slot, on top of the recurrent state: the DFlash2 draft's context buffers. Applied whether or not a draft pack is
/// loaded, as it always was.
pub const DRAFT_STATE_PER_SLOT: u64 = 16 << 20;
/// What the Qwen plan holds back for the features of the images in flight (`docs/EVAL_RESULTS.md` "Q-ADM on Qwen"; `docs/VISION.md`). The tower
/// and its scratch are loaded before the plan, so they are already out of `free` (no encoder is counted: `encoders` 0, as GLM does); what is not
/// is the feature buffer each image of a request keeps (`tokens x 5,120 x 2 B`: 40 MiB at the 4,096-token default) from its encode until the
/// sequence that owns it ends. Every slot may hold a request of `max_images_per_request` images, so the reserve is `slots x images` buffers, and
/// it moves with the slot count. Zero without a loaded tower (`--no-vision`, or a pack whose vision config the loader refused): nothing is encoded,
/// so nothing is held. (Before this the plan counted none, which is what the pinned table in `plan::tests` still pins for a pack without images.)
pub fn counted_vision_reserve(
    profile: &MemoryProfile,
    vision: Option<VisionOptions>,
    tower_loaded: bool,
    slots: usize,
) -> Result<u64, MemoryError> {
    match vision {
        Some(v) if tower_loaded => profile.vision_reserve_bytes(0, v.concurrent_images(slots)),
        _ => Ok(0),
    }
}

/// The components Qwen needs, all built. (The vision tower is listed in the plan only when the pack has one.)
const REQUIRED: &[&str] = &["exl3_linear", "gqa_attention", "gdn", "dense_mlp", "plain_residual", "mrope"];

pub struct QwenFamily;

impl ModelFamily for QwenFamily {
    fn id(&self) -> &'static str {
        "qwen3_5"
    }
    fn architectures(&self) -> &'static [&'static str] {
        &["Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM", "Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"]
    }
    fn model_types(&self) -> &'static [&'static str] {
        &["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"]
    }
    /// The Qwen model takes its verify window from `--window-rows` (up to the decode kernel's 128 rows).
    fn window_rows_adjustable(&self) -> bool {
        true
    }

    fn probe(&self, dir: &Path) -> Result<FamilyPlan, FamilyError> {
        if is_qwen3_5_moe(dir) {
            let path = dir.join("config.json");
            let text = std::fs::read_to_string(&path).map_err(|e| FamilyError::Config(format!("{}: {e}", path.display())))?;
            let root: Value = serde_json::from_str(&text).map_err(|e| FamilyError::Config(format!("config.json: {e}")))?;
            crate::family::require_exl3(&root)?;
            let cfg = crate::families::minicpmv4_7::config::MiniCpmConfig::from_json(&root).map_err(FamilyError::Config)?;
            cfg.check_shapes().map_err(FamilyError::Config)?;
            let vision = vision_spec(dir)?.map(|mut v| {
                v.tower = "vision_qwen3_5_moe";
                v
            });
            return Ok(FamilyPlan {
                family: self.id(),
                dir: dir.to_path_buf(),
                layers: cfg.layer_specs(),
                mtp_layers: vec![],
                memory: Box::new(crate::families::minicpmv4_7::MiniCpmMemorySource { cfg, draft: None }),
                vision,
                chat_format: "qwen",
                drafters: vec!["copy"],
                required_components: crate::families::minicpmv4_7::REQUIRED.to_vec(),
            });
        }
        let cfg = QwenConfig::from_dir(dir).map_err(FamilyError::Config)?;
        // the kernels are built for one attention and GDN geometry: refuse any other before a weight is read (release plan R11)
        crate::shapes::check_target(&cfg).map_err(FamilyError::Config)?;
        let vision = vision_spec(dir)?;
        let mut required: Vec<&'static str> = REQUIRED.to_vec();
        if vision.is_some() {
            required.push("vision_qwen3");
        }
        Ok(FamilyPlan {
            family: self.id(),
            dir: dir.to_path_buf(),
            layers: cfg.layer_specs(),
            mtp_layers: vec![],
            memory: Box::new(QwenMemorySource(cfg)),
            vision,
            chat_format: "qwen",
            drafters: vec!["dflash2", "copy"],
            required_components: required,
        })
    }

    /// Qwen snapshots the GDN state, the conv history, the draft's context window and the KV pages of a prompt (`serve::build_engine`).
    fn prefix_cache(&self) -> bool {
        true
    }

    fn build(&self, plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<Box<dyn ServeEngine>, ModelError> {
        if !plan.missing_components().is_empty() {
            return Err(plan.not_built("Qwen3.5 engine"));
        }
        if is_qwen3_5_moe(&plan.dir) {
            return Ok(Box::new(crate::families::minicpmv4_7::engine::build_engine(ctx.options)?));
        }
        let engine: crate::serve::QwenEngine = crate::serve::build_engine(ctx.options)?;
        Ok(Box::new(engine))
    }

    /// Qwen3.8's smart-resize preprocessing (`gb_vision::Preprocessing`), the one the server has always used; `--max-image-tokens` is
    /// enforced by the server's limits (the image is refused, not shrunk), as before.
    fn image_preprocessor(
        &self,
        _dir: &Path,
        _o: &crate::serve::ServeOptions,
    ) -> Result<Option<std::sync::Arc<dyn gb_vision::preprocessor::ImagePreprocessor>>, ModelError> {
        Ok(Some(std::sync::Arc::new(gb_vision::Preprocessing::qwen38())))
    }
}

/// The vision component from `vision_config`, when the pack has a readable one. A `vision_config` this probe cannot read is not an
/// error and not a tower: `build_engine` checks the tower itself (`vision::check_config`) and serves text only, with a warning,
/// when it cannot run it, exactly as it always did.
fn is_qwen3_5_moe(dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else { return false };
    let Ok(root) = serde_json::from_str::<Value>(&text) else { return false };
    root.get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|s| s.as_str())
        .is_some_and(|s| s.starts_with("Qwen3_5Moe"))
        || root.get("model_type").and_then(|s| s.as_str()).is_some_and(|s| s.starts_with("qwen3_5_moe"))
}

fn vision_spec(dir: &Path) -> Result<Option<VisionSpec>, FamilyError> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| FamilyError::Config(format!("{}: {e}", path.display())))?;
    let root: Value = serde_json::from_str(&text).map_err(|e| FamilyError::Config(format!("config.json: {e}")))?;
    let Some(v) = root.get("vision_config") else { return Ok(None) };
    let get = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
    let (Some(patch), Some(merge), Some(temporal), Some(out)) =
        (get("patch_size"), get("spatial_merge_size"), get("temporal_patch_size"), get("out_hidden_size"))
    else {
        return Ok(None);
    };
    let Some(image) = root.get("image_token_id").and_then(|x| x.as_u64()) else { return Ok(None) };
    Ok(Some(VisionSpec {
        tower: "vision_qwen3",
        patch_size: patch,
        merge_size: merge,
        temporal_patch_size: temporal,
        out_hidden: out,
        image_token_id: image as u32,
        max_image_tokens: 4096,
    }))
}

// ------------------------------------------------------------------------------------------------------------ memory model

/// The memory model of a Qwen pack, ready to be evaluated at any serving settings.
pub struct QwenMemorySource(pub QwenConfig);

impl MemorySource for QwenMemorySource {
    fn profile(&self, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
        memory_profile(&self.0, opts)
    }
}

/// The validated profile of `cfg` at `opts`. A config that prices a token at zero bytes (`head_dim: 0`, which the parser accepts)
/// is an error here, before the planner or admission exist.
pub fn memory_profile(cfg: &QwenConfig, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
    let shape = StepShape {
        page_tokens: BLOCK_TOKENS as u64,
        window_rows: opts.window_rows as u64,
        // `Scratch` is always allocated for at least the widest decode step
        scratch_rows: (opts.prefill_rows as u64).max(crate::scratch::MAX_DECODE_ROWS as u64),
    };
    MemoryProfile::new(&QwenMemory { cfg, opts: *opts }, shape)
}

/// The Qwen3.5/3.8 memory model.
pub struct QwenMemory<'a> {
    pub cfg: &'a QwenConfig,
    pub opts: MemoryOptions,
}

impl QwenMemory<'_> {
    /// Per-row bytes of one decode or prefill row in the activation buffers `scratch::Scratch` allocates (one set of buffers,
    /// reused by every layer): the residual and normed streams, the MLP's gate, up and product, the GDN and attention
    /// projections and records, the index arrays (11 i32 a row: ids, positions, row slot, sequence slot, keep and sequence start,
    /// and the five of the deferred commit: first owed row, owed rows, and the standalone replay's start, slot and rows; plus the
    /// 3 i32 of M-RoPE) and the DFlash2 tap rows.
    fn scratch_bytes_per_row(&self) -> u64 {
        let c = self.cfg;
        let bf = |n: usize| (n * 2) as u64;
        let (d, conv, z) = (c.hidden, c.conv_channels(), c.lin_value_heads * c.lin_value_dim);
        let (qdim, kvdim, kd) = (c.heads * c.head_dim, c.kv_heads * c.head_dim, c.lin_key_heads * c.lin_key_dim);
        bf(4 * d)
            + bf(3 * c.intermediate)
            + bf(conv.max(2 * qdim))
            + bf(2 * z.max(qdim))
            + bf(2 * kd)
            + bf(z)
            + (2 * c.lin_value_heads * 4) as u64
            + bf(2 * c.lin_value_heads)
            + bf(3 * kvdim)
            + bf(2 * qdim)
            + (11 * 4 + 12) as u64
            + bf(crate::forward::TAP_LAYERS.len() * d)
    }
    /// Buffers that do not scale with the step's rows: the window logits (bf16) and the top-k candidates.
    fn fixed_bytes(&self) -> u64 {
        let w = self.opts.window_rows as u64;
        w * self.cfg.vocab as u64 * 2 + 2 * w * 64 * 4 + 4
    }
}

impl MemoryModel for QwenMemory<'_> {
    fn family(&self) -> &str {
        "qwen3_5"
    }
    fn kv_layout(&self) -> KvLayout {
        // bf16 K and V on every full-attention layer
        KvLayout::new().with(KvLayerClass::gqa(
            self.cfg.full_layers() as u64,
            self.cfg.kv_heads as u64,
            self.cfg.head_dim as u64,
            2,
        ))
    }
    fn recurrent_layers(&self) -> u64 {
        self.cfg.linear_layers() as u64
    }
    fn state_bytes_per_seq(&self) -> u64 {
        self.cfg.state_bytes_per_seq()
    }
    /// Per linear layer and row: q and k (bf16, key heads), v (bf16, value heads), decay and beta (fp32 per value head) and the
    /// raw qkv row (bf16) the convolution history is rebuilt from (`gdn::Record` and `win_qkv`). The records exist twice: the
    /// window that follows a commit reads the previous window's set (to replay its accepted rows) while it fills the other.
    fn window_record_bytes_per_row(&self) -> u64 {
        let c = self.cfg;
        let record =
            2 * (c.lin_key_heads * c.lin_key_dim * 2) + c.lin_value_heads * c.lin_value_dim * 2 + 2 * c.lin_value_heads * 4;
        let per_layer = 2 * record + c.conv_channels() * 2;
        // with trees enabled (`--spec-tree`) a window row also holds its K/V per attention layer and the plan, path and commit tables
        let trees = if self.opts.trees { crate::forward::tree_bytes_per_row(c) } else { 0 };
        (c.linear_layers() * per_layer) as u64 + trees
    }
    fn scratch_bytes(&self, rows: u64) -> u64 {
        rows.saturating_mul(self.scratch_bytes_per_row()).saturating_add(self.fixed_bytes())
    }
    fn activation_reserve(&self) -> u64 {
        ACTIVATION_RESERVE
    }
    fn drafter_state_bytes_per_slot(&self) -> u64 {
        DRAFT_STATE_PER_SLOT
    }
    /// The tower's scratch (without the EXL3 prefill scratch, ~0.3 GiB, which `docs/VISION.md` measured on top) and the feature
    /// buffer of one largest image. The weights are measured by the loader, not predicted (the tower is EXL3 at 6 bits).
    fn vision(&self) -> Option<VisionMemory> {
        use crate::vision::{HIDDEN, MLP, OUT_HIDDEN, PATCH_IN};
        let v = self.opts.vision?;
        let patches = v.max_image_tokens as u64 * 4;
        let per_patch = (PATCH_IN * 2 + HIDDEN * 4 + 6 * HIDDEN * 2 + MLP * 2 + 16 + 16 + 72 * 4) as u64;
        Some(VisionMemory {
            weight_bytes: None,
            encoder_scratch_bytes: patches * per_patch + v.max_image_tokens as u64 * (OUT_HIDDEN * 4) as u64,
            feature_bytes_per_image: v.max_image_tokens as u64 * OUT_HIDDEN as u64 * 2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ENCODERS_IN_FLIGHT;

    const QWEN_CONFIG: &str = include_str!("../../../../testdata/qwen38/config.json");
    const GIB: u64 = 1 << 30;

    fn real() -> QwenConfig {
        QwenConfig::from_json(&serde_json::from_str(QWEN_CONFIG).unwrap()).unwrap()
    }

    fn opts() -> MemoryOptions {
        MemoryOptions::serving_defaults()
    }

    /// Tree windows' buffers are in the memory plan exactly when trees are enabled: the window record of a row grows by the K and V of
    /// every attention layer plus the plan, path and commit tables, and by nothing when they are off (the shipped plan does not move).
    #[test]
    fn tree_buffers_are_counted_when_trees_are_enabled() {
        use gb_sched::MemoryModel;
        let c = real();
        let (off, on) =
            (QwenMemory { cfg: &c, opts: opts() }, QwenMemory { cfg: &c, opts: MemoryOptions { trees: true, ..opts() } });
        // per row: 16 layers x (K + V) x 4 heads x 256 x 2 bytes = 64 KiB, and the i32 tables (parent, base, path 128, plan 384, commit
        // keep, commit path 128, KV scatter 3, owed keep, owed path 128, standalone path 128)
        let per_row = 65536 + 4 * (1 + 1 + 128 + 384 + 1 + 128 + 3 + 1 + 128 + 128);
        assert_eq!(crate::forward::tree_bytes_per_row(&c), per_row);
        assert_eq!(on.window_record_bytes_per_row() - off.window_record_bytes_per_row(), per_row);
        assert_eq!(crate::forward::tree_buffer_bytes(&c), crate::forward::WINDOW_ROWS as u64 * per_row);
        // a wider window (`--window-rows 128`, c32): the model allocates its tree buffers by its own window and the plan counts those bytes
        for w in [8usize, 64, 128] {
            let (off, on) = (
                memory_profile(&c, &MemoryOptions { window_rows: w, ..opts() }).unwrap(),
                memory_profile(&c, &MemoryOptions { window_rows: w, trees: true, ..opts() }).unwrap(),
            );
            assert_eq!(
                on.window_record_bytes() - off.window_record_bytes(),
                crate::forward::tree_buffer_bytes_for(&c, w),
                "the plan counts the tree buffers of a {w}-row window"
            );
        }
        assert_eq!(crate::forward::tree_buffer_bytes_for(&c, 128), 128 * per_row);
        let p_off = memory_profile(&c, &opts()).unwrap();
        let p_on = memory_profile(&c, &MemoryOptions { trees: true, ..opts() }).unwrap();
        assert!(p_on.engine_reserve_bytes() >= p_off.engine_reserve_bytes(), "counting more never lowers the reserve");
    }

    #[test]
    fn the_qwen_profile_is_the_legacy_arithmetic() {
        let c = real();
        let p = memory_profile(&c, &opts()).unwrap();
        // 16 layers x 4 kv heads x 256 x 2 (K, V) x 2 bytes = 64 KiB per token, a 16 MiB page of 256 tokens
        assert_eq!((p.kv_bytes_per_token(), p.kv_page_bytes()), (65536, 16 << 20));
        assert_eq!(p.kv_bytes_per_token(), c.kv_bytes_per_token(), "the generalised layout equals the GQA formula");
        // 48 x (48 x 128 x 128 x 4 + 3 x 10240 x 2) = 153,944,064; plus 16 MiB of draft context per slot
        assert_eq!(p.state_bytes_per_seq(), 153_944_064);
        assert_eq!(p.state_bytes_per_seq(), c.state_bytes_per_seq());
        assert_eq!(p.state_bytes_per_slot(), 170_721_280);
        assert_eq!(p.recurrent_layers(), 48);
    }

    #[test]
    fn qwen_window_records_and_scratch_by_hand() {
        let p = memory_profile(&real(), &opts()).unwrap();
        // per linear layer and row: q,k 2 x (16 x 128 x 2) + v 48 x 128 x 2 + decay,beta 2 x 48 x 4 + raw qkv 10240 x 2
        //   = 8192 + 12288 + 384 + 20480 = 41,344, plus the second record set the next window's replay reads
        //   (8192 + 12288 + 384 = 20,864) = 62,208; x 48 layers
        assert_eq!(p.window_record_bytes_per_row(), 48 * 62_208);
        assert_eq!(p.window_record_bytes(), 64 * 48 * 62_208);
        // the Scratch buffers per row, summed by hand for d 5120, intermediate 17408, conv 10240, z = q = 6144, kv 1024, kd 2048:
        // 40960 + 104448 + 24576 + 24576 + 8192 + 12288 + 384 + 192 + 6144 + 24576 + 56 + 51200
        // (the 56 are the index arrays: 11 i32 = 44 and the M-RoPE triple = 12; the deferred commit's five arrays are 20 of them)
        let per_row = 40_960 + 104_448 + 24_576 + 24_576 + 8_192 + 12_288 + 384 + 192 + 6_144 + 24_576 + 56 + 51_200;
        assert_eq!(per_row, 297_592);
        // widest step: a 4096-row prefill chunk, plus the 64-row logits (bf16 x 248,320) and top-k candidates
        assert_eq!(p.scratch_bytes(), 4096 * 297_592 + 64 * 248_320 * 2 + 2 * 64 * 64 * 4 + 4);
    }

    #[test]
    fn a_wider_window_costs_its_records_and_logits_and_stays_inside_the_blanket() {
        let c = real();
        let at = |w: usize| memory_profile(&c, &MemoryOptions { window_rows: w, ..opts() }).unwrap();
        let (p64, p128) = (at(64), at(128));
        // records scale with the window; the logits and candidate buffers are the only fixed bytes that do
        assert_eq!(p128.window_record_bytes(), 2 * p64.window_record_bytes());
        assert_eq!(p128.scratch_bytes() - p64.scratch_bytes(), 64 * 248_320 * 2 + 2 * 64 * 64 * 4);
        // what Model::with_window_rows allocates that grows with the window, at the served 262,144-token context: the logits, the two
        // record sets with the raw qkv rows, and the attention decode workspace (96 MiB per 8 rows). All of it, with the
        // 4,096-row scratch, must fit the 5 GiB blanket the planner holds back
        let attn = |w: usize| gb_cuda::attention::decode_workspace_bytes(w, 262_144) as u64;
        assert_eq!((attn(64) >> 20, attn(128) >> 20), (774, 1548), "MiB");
        let modelled = p128.scratch_bytes() + p128.window_record_bytes() + attn(128);
        assert!(
            modelled < p128.engine_reserve_bytes(),
            "{} MiB modelled against a {} MiB reserve",
            modelled >> 20,
            p128.engine_reserve_bytes() >> 20
        );
        assert_eq!(p128.engine_reserve_bytes(), 5 * GIB, "the blanket still governs at 128 rows");
    }

    #[test]
    fn the_qwen_reserve_is_the_blanket_for_every_legal_setting() {
        let c = real();
        // prefill chunks 1..=4096 (the kernels' limit) and window rows up to the engine's 128-row decode step
        for prefill_rows in [1usize, 64, 128, 1024, 2048, 4096] {
            for window_rows in [1usize, 8, 64, 128] {
                for max_context in [4096usize, 262_144, 1 << 20] {
                    let o = MemoryOptions { prefill_rows, window_rows, max_context, ..opts() };
                    let p = memory_profile(&c, &o).unwrap();
                    assert_eq!(p.engine_reserve_bytes(), 5 * GIB, "prefill {prefill_rows} window {window_rows}");
                }
            }
        }
    }

    #[test]
    fn qwen_vision_is_priced_and_its_features_are_counted_in_the_plan() {
        let p = memory_profile(&real(), &opts()).unwrap();
        let v = p.vision().unwrap();
        // 4096 tokens x 5120 x 2 bytes = 41,943,040: "up to 42 MB per 4096-token image" (docs/VISION.md)
        assert_eq!(v.feature_bytes_per_image, 41_943_040);
        assert_eq!(v.weight_bytes, None, "the tower's weights are measured by the loader");
        // per patch 3072 + 4608 + 13824 + 8704 + 16 + 16 + 288 = 30,528 x 16,384 patches, plus 4096 x 20,480 merger output:
        // 0.54 GiB, which is the 0.85 GiB measured in docs/VISION.md less the ~0.3 GiB EXL3 prefill scratch it does not model
        assert_eq!(v.encoder_scratch_bytes, 16_384 * 30_528 + 4096 * 20_480);
        assert!(v.encoder_scratch_bytes < 850 * (1 << 20) && v.encoder_scratch_bytes > 500 * (1 << 20));
        // what the family would reserve for 8 slots x 4 images: the scratch of the one encoder plus 32 feature buffers
        let concurrent = VisionOptions::SERVER_DEFAULT.concurrent_images(8);
        assert_eq!(p.vision_reserve_bytes(ENCODERS_IN_FLIGHT, concurrent), Ok(v.encoder_scratch_bytes + 32 * 41_943_040));
        // the plan counts the features only (the tower and its scratch are resident before it, as GLM's are): 32 buffers of 40 MiB at 8 slots,
        // growing with the slots, and nothing when no tower was loaded
        let counted = |slots: usize, tower: bool| counted_vision_reserve(&p, opts().vision, tower, slots).unwrap();
        assert_eq!(counted(8, true), 32 * 41_943_040);
        assert_eq!(counted(8, true), 1_342_177_280, "1.25 GiB");
        assert_eq!((counted(1, true), counted(6, true), counted(32, true)), (4 * 41_943_040, 24 * 41_943_040, 128 * 41_943_040));
        assert_eq!(counted(8, false), 0, "no tower, no features");
        assert_eq!(counted_vision_reserve(&p, None, true, 8), Ok(0), "text-only serving counts nothing");
        // `--max-image-tokens` shrinks a buffer, and with it the reserve
        let small = memory_profile(
            &real(),
            &MemoryOptions { vision: Some(VisionOptions { max_image_tokens: 1024, ..VisionOptions::SERVER_DEFAULT }), ..opts() },
        )
        .unwrap();
        assert_eq!(counted_vision_reserve(&small, opts().vision, true, 8), Ok(32 * 10_485_760));
    }

    #[test]
    fn text_only_serving_reserves_nothing_for_vision() {
        let p = memory_profile(&real(), &MemoryOptions { vision: None, ..opts() }).unwrap();
        assert!(p.vision().is_none());
        assert_eq!(p.vision_reserve_bytes(1, 32), Ok(0));
    }

    #[test]
    fn a_zero_head_dim_cannot_reach_the_planner_or_admission() {
        // `QwenConfig::from_json` accepts head_dim 0 (rotary dim becomes 0, which passes its checks) ...
        let mut v: Value = serde_json::from_str(QWEN_CONFIG).unwrap();
        v["text_config"]["head_dim"] = serde_json::json!(0);
        let c = QwenConfig::from_json(&v).expect("the Qwen parser does not refuse head_dim 0");
        // ... and its KV cost would be zero, which used to divide by zero in the planner and charge nothing in admission
        assert_eq!(c.kv_bytes_per_token(), 0);
        let e = memory_profile(&c, &opts()).unwrap_err();
        assert!(matches!(&e, MemoryError::Zero(m) if m.contains("qwen3_5") && m.contains("plane")), "{e}");
    }

    #[test]
    fn the_qwen_probe_reports_the_recipe_the_vision_component_and_the_drafters() {
        let dir = std::env::temp_dir().join(format!("rexl3-qwen-probe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), QWEN_CONFIG).unwrap();
        let plan = QwenFamily.probe(&dir).unwrap();
        assert_eq!(plan.layers.len(), 64);
        assert_eq!((plan.chat_format, plan.drafters.clone()), ("qwen", vec!["dflash2", "copy"]));
        assert!(plan.required_components.contains(&"vision_qwen3"));
        let v = plan.vision.unwrap();
        assert_eq!((v.tower, v.patch_size, v.merge_size, v.temporal_patch_size, v.out_hidden), ("vision_qwen3", 16, 2, 2, 5120));
        assert_eq!(v.image_token_id, 248056);
        // a pack whose vision_config this probe cannot read is not refused (build_engine decides about the tower, as before): it just
        // has no vision component, and does not require one
        for edit in ["no vision_config", "partial vision_config", "no image_token_id"] {
            let mut cfg: Value = serde_json::from_str(QWEN_CONFIG).unwrap();
            match edit {
                "no vision_config" => drop(cfg.as_object_mut().unwrap().remove("vision_config")),
                "partial vision_config" => drop(cfg["vision_config"].as_object_mut().unwrap().remove("out_hidden_size")),
                _ => drop(cfg.as_object_mut().unwrap().remove("image_token_id")),
            }
            std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
            let plan = QwenFamily.probe(&dir).unwrap();
            assert!(plan.vision.is_none(), "{edit}");
            assert!(!plan.required_components.contains(&"vision_qwen3"), "{edit}");
            assert!(plan.missing_components().is_empty(), "{edit}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
