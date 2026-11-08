//! The Bailing-3.1 memory model: Gated MLA latent + rope per token on 10 layers (9 trunk + the MTP head), Kimi delta attention
//! state and conv history per slot on the 63 KDA layers, and the activation scratch. Every figure is the arithmetic of `alloc.rs`.

use super::alloc::{self, AllocPlan};
use super::config::BailingConfig;
use crate::family::MemorySource;
use crate::kvtables::BLOCK_TOKENS;
use crate::memory::MemoryOptions;
use gb_sched::{KvLayerClass, KvLayout, KvPlane, MemoryError, MemoryModel, MemoryProfile, StepShape};

/// The blanket reserve (same as GLM/Qwen).
pub const ACTIVATION_RESERVE: u64 = 512 << 20;

pub struct BailingMemorySource(pub BailingConfig);

impl MemorySource for BailingMemorySource {
    fn profile(&self, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
        memory_profile(&self.0, opts)
    }
}

pub fn engine_options(opts: &MemoryOptions) -> MemoryOptions {
    MemoryOptions { window_rows: alloc::WINDOW_ROWS, prefill_rows: alloc::prefill_chunk_for(opts.prefill_rows), ..*opts }
}

pub fn memory_profile(cfg: &BailingConfig, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
    if opts.mtp && !cfg.has_mtp() {
        return Err(MemoryError::Inconsistent(
            "bailing_moe_v3: speculation with the pack's MTP head was asked for but the pack has no NextN layer".into(),
        ));
    }
    let window = opts.window_rows as u64;
    let opts = &MemoryOptions { prefill_rows: alloc::prefill_chunk_for(opts.prefill_rows), ..*opts };
    let shape =
        StepShape { page_tokens: BLOCK_TOKENS as u64, window_rows: window, scratch_rows: (opts.prefill_rows as u64).max(window) };
    MemoryProfile::new(&BailingMemory { cfg, opts: *opts }, shape)
}

pub struct BailingMemory<'a> {
    pub cfg: &'a BailingConfig,
    pub opts: MemoryOptions,
}

impl MemoryModel for BailingMemory<'_> {
    fn family(&self) -> &str {
        "bailing_moe_v3"
    }

    /// Gated MLA: per token, per layer, fp16 latent `kv_lora_rank` + fp16 rope `qk_rope_head_dim`. 9 trunk layers, plus the
    /// MTP layer's own cache when it drafts.
    fn kv_layout(&self) -> KvLayout {
        let trunk = self.cfg.mla_layers().len() as u64;
        let layers = trunk + if self.opts.mtp { 1 } else { 0 };
        let latent = self.cfg.mla.kv_lora_rank as u64;
        let rope = self.cfg.mla.qk_rope_head_dim as u64;
        KvLayout::new().with(KvLayerClass {
            name: "mla_gated",
            layers,
            planes: vec![KvPlane::per_token("latent", latent, 2), KvPlane::per_token("rope", rope, 2)],
        })
    }

    fn recurrent_layers(&self) -> u64 {
        self.cfg.kda_layers().len() as u64
    }

    /// A slot's state: the 63 KDA pools (two fp32 state buffers, the conv history and the parity word each), allocated per slot up
    /// front. The slot's block-table row is sized by the KV pool and priced with it (`engine::post_alloc_reserve_bytes`).
    fn state_bytes_per_seq(&self) -> u64 {
        AllocPlan::slot_items(self.cfg).iter().map(|i| i.bytes).sum()
    }

    /// One window row of every KDA layer's record (the 63 window records hold `WINDOW_ROWS` rows each).
    fn window_record_bytes_per_row(&self) -> u64 {
        AllocPlan::window_row_items(self.cfg).iter().map(|i| i.bytes).sum()
    }

    /// Every buffer of a forward for a prefill chunk of `rows` rows (`AllocPlan::scratch_items`), the shared prefill KDA record and the
    /// KDA workspace included.
    fn scratch_bytes(&self, rows: u64) -> u64 {
        AllocPlan::scratch_items(self.cfg, alloc::prefill_chunk_for(rows as usize)).iter().map(|i| i.bytes).sum()
    }

    fn activation_reserve(&self) -> u64 {
        ACTIVATION_RESERVE
    }

    /// Nothing: the MTP head's latent cache is per token (the 10th layer in every page), its buffers for a call are in the scratch,
    /// and its tap rings and draft head are allocated after the plan (`engine::post_alloc_reserve_bytes`).
    fn drafter_state_bytes_per_slot(&self) -> u64 {
        0
    }

    fn vision(&self) -> Option<gb_sched::VisionMemory> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const REAL: &str = include_str!("../../../../../testdata/ling31/config.json");
    const GIB: u64 = 1 << 30;

    fn real() -> BailingConfig {
        BailingConfig::from_json(&serde_json::from_str(REAL).unwrap()).unwrap()
    }

    fn opts() -> MemoryOptions {
        engine_options(&MemoryOptions { prefill_rows: 128, ..MemoryOptions::serving_defaults() })
    }

    #[test]
    fn kv_per_token_by_hand() {
        let c = real();
        let p = memory_profile(&c, &opts()).unwrap();
        // 9 trunk MLA layers x (latent 512 + rope 64) x 2 B
        assert_eq!(p.kv_bytes_per_token(), 9 * (512 + 64) * 2);
        assert_eq!(p.kv_bytes_per_token(), 10_368);
        let pt = crate::kvtables::BLOCK_TOKENS as u64;
        assert_eq!(p.kv_page_bytes(), 9 * pt * 1152);
        assert_eq!(p.kv_bytes_per_token() * 262_144, 2_717_908_992);
        assert!((p.kv_bytes_per_token() * 262_144) as f64 / GIB as f64 > 2.5);
        // with MTP: a 10th layer
        let with_mtp = memory_profile(&c, &MemoryOptions { mtp: true, ..opts() }).unwrap();
        assert_eq!(with_mtp.kv_bytes_per_token(), 10 * 1152);
        assert_eq!(with_mtp.kv_page_bytes(), 10 * pt * 1152);
    }

    #[test]
    fn state_and_records_by_hand() {
        let c = real();
        assert_eq!(c.kda_state_bytes_one_buffer(), 8_388_608);
        assert_eq!(c.kda_record_bytes_per_row_per_layer(), 82_176);
        let p = memory_profile(&c, &opts()).unwrap();
        assert_eq!(p.recurrent_layers(), 63);
        // what `kda::Pool::new` allocates per slot and layer: both state buffers, the conv history (3 raw rows of the 24,576-wide
        // q|k|v, bf16) and the parity word
        let layer = c.kda_state_bytes_one_buffer() + 3 * 3 * 8192 * 2 + 4;
        assert_eq!(layer, 8_536_068);
        assert_eq!(p.state_bytes_per_seq(), 63 * layer);
        assert_eq!(p.state_bytes_per_seq(), 537_772_284);
        assert_eq!(p.drafter_state_bytes_per_slot(), 0);
        assert_eq!(p.window_record_bytes_per_row(), 63 * 82_176);
        assert_eq!(p.window_record_bytes(), 63 * 82_176 * 64);
    }

    #[test]
    fn the_scratch_is_the_alloc_plans_and_grows_with_the_chunk() {
        let c = real();
        let at = |rows: usize| memory_profile(&c, &engine_options(&MemoryOptions { prefill_rows: rows, ..opts() })).unwrap();
        let p = at(512);
        let want: u64 = AllocPlan::scratch_items(&c, 512).iter().map(|i| i.bytes).sum();
        assert_eq!(p.scratch_bytes(), want);
        // the shared prefill KDA record of 512 rows is in it (one record, not 63)
        assert!(p.scratch_bytes() >= 512 * 82_176);
        assert!(at(128).scratch_bytes() < p.scratch_bytes());
        // the server's default `--prefill-chunk` (4096) clamps to this family's widest chunk
        assert_eq!(at(4096).scratch_bytes(), at(alloc::MAX_PREFILL_CHUNK).scratch_bytes());
    }

    #[test]
    fn mtp_speculation_needs_an_mtp_layer() {
        let mut v: Value = serde_json::from_str(REAL).unwrap();
        v["num_nextn_predict_layers"] = serde_json::json!(0);
        let c = BailingConfig::from_json(&v).unwrap();
        assert!(memory_profile(&c, &opts()).is_ok());
        let e = memory_profile(&c, &MemoryOptions { mtp: true, ..opts() }).unwrap_err();
        assert!(matches!(e, MemoryError::Inconsistent(m) if m.contains("no NextN layer")));
    }
}
