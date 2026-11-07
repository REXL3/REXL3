//! The Qwen backend's prefix-cache memory: a pool of snapshot slots, each holding what a prompt leaves behind that is not paged KV (see
//! `docs/PREFIX_CACHE.md`): the recurrent state `S` and the conv history of every linear layer, laid out like [`GdnPools`] (a snapshot
//! slot is a slot of a pool of its own), and the DFlash2 draft's context window (K and V of its layers for the 2,048 positions before the
//! cut). The pool is allocated once at boot; nothing is allocated while serving.

use crate::config::QwenConfig;
use crate::draft::Draft;
use crate::error::ModelError;
use crate::state::GdnPools;
use gb_cuda::dflash2::DraftCtxLayout;
use gb_cuda::DevBuf;

/// What a snapshot slot holds besides the state: where its draft window lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapMeta {
    /// Tokens of the prompt the snapshot is the state after.
    pub len: usize,
    /// First position of the saved draft window (`len - 2048`, or 0).
    pub lo: usize,
    /// Positions saved (`len - lo`); 0 when the model has no draft.
    pub n: usize,
}

pub struct SnapPool {
    pub slots: usize,
    /// `S` and conv history, one slot per snapshot.
    pub gdn: GdnPools,
    /// Per draft layer, `[slots][WINDOW][kv_heads x head_dim]` bf16 (empty without a draft).
    pub dk: Vec<DevBuf>,
    pub dv: Vec<DevBuf>,
    free: Vec<u32>,
    meta: Vec<Option<SnapMeta>>,
}

impl SnapPool {
    /// Bytes of one snapshot slot for a model with (`with_draft`) or without a draft. The memory plan and the allocation both use this.
    pub fn bytes_per_slot(cfg: &QwenConfig, with_draft: bool) -> u64 {
        cfg.state_bytes_per_seq() + if with_draft { Draft::window_snapshot_bytes() } else { 0 }
    }

    pub fn new(cfg: &QwenConfig, slots: usize, with_draft: bool) -> Result<Self, ModelError> {
        if slots == 0 {
            return Err(ModelError::Config("a prefix cache needs at least one snapshot slot".into()));
        }
        // every slot is written by a capture before a restore reads it (`SnapMeta` says which hold one): no need to zero gigabytes at boot
        let gdn = GdnPools::new_uninit(cfg, slots)?;
        let (mut dk, mut dv) = (Vec::new(), Vec::new());
        if with_draft {
            let per_layer = slots * DraftCtxLayout::WINDOW * Draft::RING_ROW_BYTES;
            for _ in 0..crate::draft::LAYERS {
                dk.push(DevBuf::alloc(per_layer)?);
                dv.push(DevBuf::alloc(per_layer)?);
            }
        }
        Ok(Self { slots, gdn, dk, dv, free: (0..slots as u32).rev().collect(), meta: vec![None; slots] })
    }

    /// Bytes allocated on the device.
    pub fn allocated_bytes(&self) -> u64 {
        self.gdn.bytes() as u64 + self.dk.iter().chain(self.dv.iter()).map(|b| b.bytes() as u64).sum::<u64>()
    }

    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Take a free slot, or `None`.
    pub fn alloc(&mut self) -> Option<u32> {
        self.free.pop()
    }

    /// Give a slot back (a capture that was never finished, or an evicted entry).
    pub fn free_slot(&mut self, s: u32) {
        assert!((s as usize) < self.slots && !self.free.contains(&s), "snapshot slot {s} freed twice or out of range");
        self.meta[s as usize] = None;
        self.free.push(s);
    }

    /// Hashes of snapshot `s`'s recurrent state and conv history (FNV-1a over every linear layer's bytes, in layer order): the same fold as
    /// `forward::StateHashes::{s, conv}`, so a snapshot can be compared with a slot's state.
    pub fn gdn_hashes(&self, s: u32) -> Result<(u64, u64), ModelError> {
        let fnv = |h: u64, bytes: &[u8]| bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
        let (sb, cb) = (gb_cuda::gdn::S_SLOT_ELEMS * 4, gb_cuda::gdn::CONV_SLOT_ELEMS * 2);
        let (mut hs, mut hc) = (0xcbf2_9ce4_8422_2325u64, 0xcbf2_9ce4_8422_2325u64);
        let mut buf = vec![0u8; sb];
        for (si, ci) in self.gdn.s.iter().zip(self.gdn.conv.iter()) {
            si.download_range(s as usize * sb, &mut buf[..sb])?;
            hs = fnv(hs, &buf[..sb]);
            ci.download_range(s as usize * cb, &mut buf[..cb])?;
            hc = fnv(hc, &buf[..cb]);
        }
        Ok((hs, hc))
    }

    /// FNV-1a over snapshot `s`'s saved draft window, `n` positions from its start (the same order [`Draft::ring_window_hash`] reads a ring).
    pub fn window_hash(&self, s: u32, n: usize) -> Result<u64, ModelError> {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let base = s as usize * DraftCtxLayout::WINDOW * Draft::RING_ROW_BYTES;
        for (k, v) in self.dk.iter().zip(self.dv.iter()) {
            for buf in [k, v] {
                let mut bytes = vec![0u8; n * Draft::RING_ROW_BYTES];
                buf.download_range(base, &mut bytes)?;
                h = bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
            }
        }
        Ok(h)
    }

    pub fn set_meta(&mut self, s: u32, m: SnapMeta) {
        self.meta[s as usize] = Some(m);
    }

    pub fn meta(&self, s: u32) -> Option<SnapMeta> {
        self.meta.get(s as usize).copied().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN_CONFIG: &str = include_str!("../../../testdata/qwen38/config.json");

    fn cfg() -> QwenConfig {
        QwenConfig::from_json(&serde_json::from_str(QWEN_CONFIG).unwrap()).unwrap()
    }

    #[test]
    fn a_snapshot_slot_is_the_state_plus_the_draft_window() {
        let c = cfg();
        // 48 x (48 x 128 x 128 x 4 + 3 x 10240 x 2)
        assert_eq!(c.state_bytes_per_seq(), 153_944_064);
        // 5 layers x (K, V) x 2,048 positions x (8 kv heads x 128 x 2 bytes)
        assert_eq!(Draft::window_snapshot_bytes(), 41_943_040);
        assert_eq!(SnapPool::bytes_per_slot(&c, true), 195_887_104);
        assert_eq!(SnapPool::bytes_per_slot(&c, false), 153_944_064);
    }
}
