//! Per-slot recurrent state of the linear-attention layers.
//!
//! One pool per linear layer, indexed by sequence slot: the delta-rule state `S` (fp32) and the conv history (bf16).
//! The kernels treat a pool as read-only during a verify window and write it only in `commit` / `prefill`, and in the next
//! window kernel of a sequence whose last window still has accepted rows owed to S (`forward::Model::flush`); `Model::settle`
//! applies those rows so the pools can be read.

use crate::config::QwenConfig;
use crate::error::ModelError;
use gb_cuda::gdn::{CONV_SLOT_ELEMS, S_SLOT_ELEMS};
use gb_cuda::DevBuf;

pub struct GdnPools {
    pub slots: usize,
    /// f32 `[slots][48][128][128]`, one buffer per linear layer in layer order.
    pub s: Vec<DevBuf>,
    /// bf16 `[slots][3][10240]`, one per linear layer.
    pub conv: Vec<DevBuf>,
}

impl GdnPools {
    pub fn new(c: &QwenConfig, slots: usize) -> Result<Self, ModelError> {
        Self::alloc(c, slots, true)
    }

    /// Pools whose slots hold whatever the allocator gave: for a pool every slot of which is written (a prefix-cache snapshot) before it is read,
    /// where zeroing gigabytes at boot would be wasted time.
    pub fn new_uninit(c: &QwenConfig, slots: usize) -> Result<Self, ModelError> {
        Self::alloc(c, slots, false)
    }

    fn alloc(c: &QwenConfig, slots: usize, zero: bool) -> Result<Self, ModelError> {
        if c.lin_value_heads * c.lin_value_dim * c.lin_key_dim != S_SLOT_ELEMS
            || (c.conv_kernel - 1) * c.conv_channels() != CONV_SLOT_ELEMS
        {
            return Err(ModelError::Config(
                "the GDN kernels are compiled for 48 x 128 x 128 state and 3 x 10240 conv history".into(),
            ));
        }
        let n = c.linear_layers();
        let mut s = Vec::with_capacity(n);
        let mut conv = Vec::with_capacity(n);
        for _ in 0..n {
            if zero {
                s.push(DevBuf::zeroed(slots * S_SLOT_ELEMS * 4)?);
                conv.push(DevBuf::zeroed(slots * CONV_SLOT_ELEMS * 2)?);
            } else {
                s.push(DevBuf::alloc(slots * S_SLOT_ELEMS * 4)?);
                conv.push(DevBuf::alloc(slots * CONV_SLOT_ELEMS * 2)?);
            }
        }
        Ok(Self { slots, s, conv })
    }

    /// Zero one slot's state in every layer (a new sequence starts from a zero state).
    pub fn reset_slot(&mut self, slot: usize, st: &gb_cuda::Stream) -> Result<(), ModelError> {
        assert!(slot < self.slots);
        for (s, conv) in self.s.iter_mut().zip(self.conv.iter_mut()) {
            zero_range(s, slot * S_SLOT_ELEMS * 4, S_SLOT_ELEMS * 4, st)?;
            zero_range(conv, slot * CONV_SLOT_ELEMS * 2, CONV_SLOT_ELEMS * 2, st)?;
        }
        Ok(())
    }

    pub fn bytes(&self) -> usize {
        self.s.iter().chain(self.conv.iter()).map(|b| b.bytes()).sum()
    }
}

fn zero_range(buf: &mut DevBuf, offset: usize, len: usize, st: &gb_cuda::Stream) -> Result<(), ModelError> {
    buf.memset_range_async(offset, len, 0, st)?;
    Ok(())
}
