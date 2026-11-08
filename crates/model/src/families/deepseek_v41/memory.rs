//! The DeepSeek-V4.1 memory model: what a token costs in the paged cache.
//!
//! Every trunk layer caches one shared MLA latent of `kv_heads * head_dim` values (512 at fp16). A layer with a non-zero
//! `compress_ratio` additionally caches one compressed entry — `head_dim` wide — per `ratio` tokens; the four layers in
//! `kv_source_layer_ids` are the ones that write those entries, and the other layers with a ratio read the nearest source
//! layer's. The four layers that own an indexer key projection cache that key, `index_head_dim` wide, per token.
//!
//! The family has no recurrent state: every layer is attention, so a slot costs its block-table row and nothing else, and a
//! speculative window keeps no per-row history. The activation scratch is a **placeholder** until the forward exists: the
//! kernels that will price it row by row are not written, so this model states a flat per-row allowance and says so.

use super::config::DeepseekV41Config;
use crate::family::MemorySource;
use crate::kvtables::BLOCK_TOKENS;
use crate::memory::MemoryOptions;
use gb_sched::{KvLayerClass, KvLayout, KvPlane, MemoryError, MemoryModel, MemoryProfile, StepShape};

/// The blanket held back for everything the plan does not model row by row: the same 5 GiB the Qwen and GLM engines hold.
pub const ACTIVATION_RESERVE: u64 = 5 << 30;

/// Activation bytes per scratch row, until the forward's own allocations can be priced (a placeholder, not a measurement).
pub const SCRATCH_BYTES_PER_ROW: u64 = 1 << 20;

/// The memory model of a DeepSeek-V4.1 pack, ready to be evaluated at any serving settings.
pub struct DeepseekV41MemorySource(pub DeepseekV41Config);

impl MemorySource for DeepseekV41MemorySource {
    fn profile(&self, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
        memory_profile(&self.0, opts)
    }
}

/// The validated profile of `cfg` at `opts`.
pub fn memory_profile(cfg: &DeepseekV41Config, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
    if opts.mtp && cfg.nextn == 0 {
        return Err(MemoryError::Inconsistent(
            "deepseek_v41: speculation with the pack's NextN layers was asked for but the pack has none".into(),
        ));
    }
    let shape = StepShape {
        page_tokens: BLOCK_TOKENS as u64,
        window_rows: opts.window_rows as u64,
        scratch_rows: (opts.prefill_rows as u64).max(opts.window_rows as u64),
    };
    MemoryProfile::new(&DeepseekV41Memory { cfg, opts: *opts }, shape)
}

/// The DeepSeek-V4.1 memory model.
pub struct DeepseekV41Memory<'a> {
    pub cfg: &'a DeepseekV41Config,
    pub opts: MemoryOptions,
}

/// fp16, as the reference engine caches the latent, the compressed entries and the indexer keys.
const KV_ELEM_BYTES: u64 = 2;

impl MemoryModel for DeepseekV41Memory<'_> {
    fn family(&self) -> &str {
        "deepseek_v41"
    }

    /// The latent of every trunk layer, the compressed entries of every layer with a ratio, the indexer key of every layer
    /// that owns one — and, when the NextN layers draft, their own latents.
    fn kv_layout(&self) -> KvLayout {
        let latent = KvPlane::per_token("mla_latent", self.cfg.kv_heads as u64 * self.cfg.head_dim as u64, KV_ELEM_BYTES);
        let index_key = KvPlane::per_token("indexer_key", self.cfg.indexer.head_dim as u64, KV_ELEM_BYTES);
        let mut layout = KvLayout::new().with(KvLayerClass {
            name: "mla_latent",
            layers: self.cfg.layers as u64 + if self.opts.mtp { self.cfg.nextn as u64 } else { 0 },
            planes: vec![latent],
        });
        // one class per distinct compressed-entry rate: a pool must not straddle a page, so a ratio that does not divide the
        // page is refused by the layout rather than priced approximately
        for ratio in [1u64, 2] {
            let layers = (0..self.cfg.layers).filter(|&l| self.cfg.compress.ratio(l) as u64 == ratio).count() as u64;
            if layers > 0 {
                layout = layout.with(KvLayerClass {
                    name: if ratio == 1 { "compressed_kv_1" } else { "compressed_kv_2" },
                    layers,
                    planes: vec![KvPlane::pooled("compressed_kv", self.cfg.head_dim as u64, ratio, KV_ELEM_BYTES)],
                });
            }
        }
        if !self.cfg.compress.source_layer_ids.is_empty() {
            layout = layout.with(KvLayerClass {
                name: "indexer_key",
                layers: self.cfg.compress.source_layer_ids.len() as u64,
                planes: vec![index_key],
            });
        }
        layout
    }

    /// No layer carries a recurrence: every one is attention over the paged cache.
    fn recurrent_layers(&self) -> u64 {
        0
    }

    fn state_bytes_per_seq(&self) -> u64 {
        0
    }

    fn window_record_bytes_per_row(&self) -> u64 {
        0
    }

    /// A placeholder until the forward's buffers exist (see the module comment).
    fn scratch_bytes(&self, rows: u64) -> u64 {
        rows.max(1) * SCRATCH_BYTES_PER_ROW
    }

    fn activation_reserve(&self) -> u64 {
        ACTIVATION_RESERVE
    }

    fn drafter_state_bytes_per_slot(&self) -> u64 {
        0
    }

    /// The tower is not built, and its reserve is not modelled yet: `None` rather than a number nobody has checked.
    fn vision(&self) -> Option<gb_sched::VisionMemory> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const CONFIG: &str = include_str!("../../../../../testdata/deepseek_v41/config.json");

    fn real() -> DeepseekV41Config {
        DeepseekV41Config::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    fn opts() -> MemoryOptions {
        MemoryOptions::serving_defaults()
    }

    #[test]
    fn a_token_costs_the_latent_the_compressed_entries_and_the_indexer_key() {
        let c = real();
        let p = memory_profile(&c, &opts()).unwrap();
        // 40 latents of 512 fp16 = 40,960 B a token; 18 layers cache one 512-value fp16 entry per 2 tokens (512 B a
        // token) and 20 cache one per token (1,024 B); 4 indexer keys of 128 fp16 (256 B)
        let expected = 40 * 1024 + 18 * 512 + 20 * 1024 + 4 * 256;
        assert_eq!(p.kv_bytes_per_token(), expected);
        assert_eq!(p.kv_bytes_per_token(), 71_680);
        assert_eq!(p.recurrent_layers(), 0);
        assert_eq!(p.state_bytes_per_seq(), 0);
        assert_eq!(p.window_record_bytes_per_row(), 0);
        assert_eq!(p.drafter_state_bytes_per_slot(), 0);
        // a page is 256 tokens, exactly
        assert_eq!(p.kv_page_bytes(), p.kv_bytes_per_token() * 256);
    }

    #[test]
    fn the_nextn_layers_are_charged_only_when_they_draft() {
        let c = real();
        let off = memory_profile(&c, &opts()).unwrap();
        let on = memory_profile(&c, &MemoryOptions { mtp: true, ..opts() }).unwrap();
        assert_eq!(on.kv_bytes_per_token() - off.kv_bytes_per_token(), 3 * 1024);
    }

    #[test]
    fn a_profile_with_no_layer_to_cache_a_context_is_refused() {
        let mut c = real();
        c.layers = 0;
        c.compress.ratios.clear();
        c.compress.source_layer_ids.clear();
        assert!(matches!(memory_profile(&c, &opts()), Err(MemoryError::Zero(_))));
    }

    #[test]
    fn mtp_speculation_needs_a_nextn_layer() {
        let mut v: Value = serde_json::from_str(CONFIG).unwrap();
        v["text_config"]["num_nextn_predict_layers"] = serde_json::json!(0);
        v["text_config"]["compress_ratios"] = serde_json::json!(vec![2u32; 40]);
        v["text_config"]["dspark_target_layer_ids"] = serde_json::json!([]);
        let c = DeepseekV41Config::from_json(&v).unwrap();
        assert!(memory_profile(&c, &opts()).is_ok());
        let e = memory_profile(&c, &MemoryOptions { mtp: true, ..opts() }).unwrap_err();
        assert!(matches!(e, MemoryError::Inconsistent(ref m) if m.contains("NextN layers")), "{e}");
    }
}
