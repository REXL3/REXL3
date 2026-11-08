//! The persistent device state of a MiniCPM-V-4.7 engine, allocated at the shapes the memory model prices
//! (`super::memory::allocations`): the paged KV of the 10 attention layers, the 30 gated delta net layers' per-slot state and
//! convolution history, and the window's record set and raw qkv rows.
//!
//! The prefix cache's snapshot arena (E6b) is allocated here too, at the planned slot count, so that the boot compares it with the plan
//! item by item; it moves to the backend (`prefix::SnapshotArena`) before the model is built.
//!
//! Nothing here runs a kernel. The buffers are the ones the serial forward (E3) will fill, laid out for the kernels the E1 slice
//! built: the GDN pools at the 16 key x 32 value head shape (`GdnShape::Hv32`: `[slots][32][128][128]` fp32 and
//! `[slots][3][8192]` bf16) and the KV pages at 16 query x 2 KV heads of 256 (`AttnShape::Q16Kv2`). Flash-Next's pools carry the
//! 48-head module constants and are not reused: at this shape they would be 1.5 times too large and read as a different layout.
//!
//! [`ModelState::bytes`] sums what the device really holds, per kind, so the boot can compare it item by item with the plan.

use super::config::MiniCpmConfig;
use super::memory::{AllocItem, AllocKind};
use crate::error::ModelError;
use gb_cuda::attention::{AttnShape, PagedKv};
use gb_cuda::gdn::{GdnShape, Record};
use gb_cuda::{DevBuf, Stream};

/// The gated delta net shape of this model.
pub const GDN: GdnShape = GdnShape::Hv32;
/// The attention shape of this model's full-attention layers.
pub const ATTN: AttnShape = AttnShape::Q16Kv2;

/// Per GDN layer, in layer order: the delta-rule state `S` (fp32 `[slots][32][128][128]`) and the convolution history
/// (bf16 `[slots][3][8192]`).
pub struct GdnPools {
    pub slots: usize,
    pub shape: GdnShape,
    pub s: Vec<DevBuf>,
    pub conv: Vec<DevBuf>,
}

impl GdnPools {
    /// Zeroed pools for `layers` gated delta net layers and `slots` sequence slots (a new sequence starts from a zero state).
    pub fn new(layers: usize, slots: usize) -> Result<Self, ModelError> {
        Self::new_shaped(layers, slots, GDN)
    }

    pub fn new_shaped(layers: usize, slots: usize, shape: GdnShape) -> Result<Self, ModelError> {
        let mut s = Vec::with_capacity(layers);
        let mut conv = Vec::with_capacity(layers);
        for _ in 0..layers {
            s.push(DevBuf::zeroed(slots * shape.s_slot_elems() * 4)?);
            conv.push(DevBuf::zeroed(slots * shape.conv_slot_elems() * 2)?);
        }
        Ok(Self { slots, shape, s, conv })
    }

    /// Zero one slot's state in every layer.
    pub fn reset_slot(&mut self, slot: usize, st: &Stream) -> Result<(), ModelError> {
        if slot >= self.slots {
            return Err(ModelError::Request(format!("slot {slot} of {}", self.slots)));
        }
        let shape = self.shape;
        for (s, conv) in self.s.iter_mut().zip(self.conv.iter_mut()) {
            s.memset_range_async(slot * shape.s_slot_elems() * 4, shape.s_slot_elems() * 4, 0, st)?;
            conv.memset_range_async(slot * shape.conv_slot_elems() * 2, shape.conv_slot_elems() * 2, 0, st)?;
        }
        Ok(())
    }

    pub fn s_bytes(&self) -> u64 {
        self.s.iter().map(|b| b.bytes() as u64).sum()
    }

    pub fn conv_bytes(&self) -> u64 {
        self.conv.iter().map(|b| b.bytes() as u64).sum()
    }
}

/// What a verify window leaves behind for the gated delta net layers: one record set per layer (the window's post-convolution rows;
/// the committed state is only read while the window runs, and `gdn::commit` replays the accepted prefix from this record before the
/// next window or prefill chunk touches the slot) and the raw qkv rows the convolution history is rebuilt from. Rows are the widest
/// window (64: four streams of 16).
pub struct WindowBuffers {
    pub rows: usize,
    pub records: Vec<Record>,
    pub raw_qkv: Vec<DevBuf>,
}

impl WindowBuffers {
    pub fn new(gdn_layers: usize, rows: usize) -> Result<Self, ModelError> {
        Self::new_shaped(gdn_layers, rows, GDN)
    }

    pub fn new_shaped(gdn_layers: usize, rows: usize, shape: GdnShape) -> Result<Self, ModelError> {
        let mut records = Vec::with_capacity(gdn_layers);
        let mut raw = Vec::with_capacity(gdn_layers);
        for _ in 0..gdn_layers {
            records.push(Record::alloc_shaped(rows, shape)?);
            raw.push(DevBuf::zeroed(rows * shape.conv_dim() * 2)?);
        }
        Ok(Self { rows, records, raw_qkv: raw })
    }

    fn record_bytes(r: &Record) -> u64 {
        (r.q.bytes() + r.k.bytes() + r.v.bytes() + r.g_decay.bytes() + r.beta.bytes()) as u64
    }

    pub fn records_bytes(&self) -> u64 {
        self.records.iter().map(Self::record_bytes).sum()
    }

    pub fn raw_qkv_bytes(&self) -> u64 {
        self.raw_qkv.iter().map(|b| b.bytes() as u64).sum()
    }
}

/// What the device holds of the state, by the names of `memory::allocations`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateBytes {
    pub gdn_state: u64,
    pub gdn_conv_history: u64,
    pub kv_target: u64,
    pub gdn_records: u64,
    pub gdn_raw_qkv: u64,
    /// The prefix cache's snapshot arena (E6b); 0 without a cache.
    pub prefix_snapshots: u64,
}

impl StateBytes {
    /// Everything the state holds.
    pub fn total(&self) -> u64 {
        self.gdn_state + self.gdn_conv_history + self.kv_target + self.gdn_records + self.gdn_raw_qkv + self.prefix_snapshots
    }
}

pub struct ModelState {
    pub slots: usize,
    /// KV pages (256 tokens each), shared by every slot.
    pub kv_pages: usize,
    /// Paged K and V of the full-attention layers, in layer order.
    pub kv: Vec<PagedKv>,
    pub gdn: GdnPools,
    pub window: WindowBuffers,
    /// The prefix cache's snapshot arena: `snapshot_slots` snapshots of one sequence's GDN state and convolution history each, one
    /// device buffer. `None` without a cache. Take it before the model is built (`MiniCpmModel::new` refuses a state that still holds
    /// it): it belongs to the backend.
    pub snapshots: Option<DevBuf>,
    pub snapshot_slots: usize,
}

impl ModelState {
    /// Allocate the state of `slots` slots over a pool of `kv_pages` pages and a window of `window_rows` rows.
    pub fn alloc(cfg: &MiniCpmConfig, slots: usize, kv_pages: usize, window_rows: usize) -> Result<Self, ModelError> {
        Self::alloc_cached(cfg, slots, kv_pages, window_rows, 0)
    }

    /// [`ModelState::alloc`] with the prefix cache's `snapshot_slots` snapshot slots (0: none). `kv_pages` is the whole pool,
    /// the cache's extra pages included.
    pub fn alloc_cached(
        cfg: &MiniCpmConfig,
        slots: usize,
        kv_pages: usize,
        window_rows: usize,
        snapshot_slots: usize,
    ) -> Result<Self, ModelError> {
        let want_gdn = cfg.gdn_shape();
        let want_attn = cfg.attn_shape();
        if GdnShape::from_counts(cfg.lin_key_heads, cfg.lin_value_heads, cfg.lin_key_dim, cfg.lin_value_dim) != Some(want_gdn) {
            return Err(ModelError::Config(format!(
                "gated delta net {} key x {} value heads of {} (the state is laid out for 16 x 32 of 128)",
                cfg.lin_key_heads, cfg.lin_value_heads, cfg.lin_key_dim
            )));
        }
        if AttnShape::from_counts(cfg.heads, cfg.kv_heads) != Some(want_attn) || cfg.head_dim != 256 {
            return Err(ModelError::Config(format!(
                "attention {} query x {} KV heads of {} (the cache is laid out for 16 x 2 of 256)",
                cfg.heads, cfg.kv_heads, cfg.head_dim
            )));
        }
        if slots == 0 || kv_pages == 0 || window_rows == 0 {
            return Err(ModelError::Config(format!("state of {slots} slots, {kv_pages} pages, {window_rows} window rows")));
        }
        let kv = (0..cfg.full_layers()).map(|_| PagedKv::with_shape(kv_pages, want_attn)).collect::<Result<Vec<_>, _>>()?;
        let gdn = GdnPools::new_shaped(cfg.linear_layers(), slots, want_gdn)?;
        let window = WindowBuffers::new_shaped(cfg.linear_layers(), window_rows, want_gdn)?;
        let snapshots = if snapshot_slots == 0 {
            None
        } else {
            let bytes = snapshot_slots
                .checked_mul(cfg.linear_layers() * (want_gdn.s_slot_elems() * 4 + want_gdn.conv_slot_elems() * 2))
                .ok_or_else(|| ModelError::Config("prefix arena size overflows".into()))?;
            Some(DevBuf::zeroed(bytes)?)
        };
        Ok(Self { slots, kv_pages, kv, gdn, window, snapshots, snapshot_slots })
    }

    /// The bytes of every buffer, by kind.
    pub fn bytes(&self) -> StateBytes {
        StateBytes {
            gdn_state: self.gdn.s_bytes(),
            gdn_conv_history: self.gdn.conv_bytes(),
            kv_target: self.kv.iter().map(|c| (c.k.bytes() + c.v.bytes()) as u64).sum(),
            gdn_records: self.window.records_bytes(),
            gdn_raw_qkv: self.window.raw_qkv_bytes(),
            prefix_snapshots: self.snapshots.as_ref().map_or(0, |b| b.bytes() as u64),
        }
    }
}

/// Compare what the device holds with the allocation list the memory model prices (`memory::allocations`), item by item: each
/// item's bytes times its count (slots, pages or window rows). Every difference is named; empty when plan and device agree. No
/// prefix cache: [`compare_with_plan_cached`] with 0 snapshot slots.
pub fn compare_with_plan(
    items: &[AllocItem],
    state: &StateBytes,
    slots: usize,
    kv_pages: usize,
    window_rows: usize,
) -> Vec<String> {
    compare_with_plan_cached(items, state, slots, kv_pages, window_rows, 0)
}

/// [`compare_with_plan`] with the prefix cache's `snapshots` snapshot slots: the `prefix_snapshot` item times that count must equal
/// the arena's bytes, and `kv_pages` is the whole pool (the cache's extra pages included).
pub fn compare_with_plan_cached(
    items: &[AllocItem],
    state: &StateBytes,
    slots: usize,
    kv_pages: usize,
    window_rows: usize,
    snapshots: usize,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = Vec::new();
    for it in items {
        let count = match it.kind {
            AllocKind::PerSlot => slots as u64,
            AllocKind::PerPage => kv_pages as u64,
            AllocKind::PerWindowRow => window_rows as u64,
            AllocKind::PerSnapshot => snapshots as u64,
        };
        let held = match it.name {
            "gdn_state" => state.gdn_state,
            "gdn_conv_history" => state.gdn_conv_history,
            "kv_target" => state.kv_target,
            "gdn_records" => state.gdn_records,
            "gdn_raw_qkv" => state.gdn_raw_qkv,
            "prefix_snapshot" => state.prefix_snapshots,
            other => {
                out.push(format!("{other}: the plan prices an allocation the engine state does not make"));
                continue;
            }
        };
        seen.push(it.name);
        if held != it.bytes * count {
            out.push(format!(
                "{}: the device holds {held} B, the plan prices {} x {count} = {} B",
                it.name,
                it.bytes,
                it.bytes * count
            ));
        }
    }
    for (name, held) in [
        ("gdn_state", state.gdn_state),
        ("gdn_conv_history", state.gdn_conv_history),
        ("kv_target", state.kv_target),
        ("gdn_records", state.gdn_records),
        ("gdn_raw_qkv", state.gdn_raw_qkv),
        ("prefix_snapshot", state.prefix_snapshots),
    ] {
        if held != 0 && !seen.contains(&name) {
            out.push(format!("{name}: the device holds {held} B and the plan does not price it"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::real;
    use super::super::memory::{allocations, AllocKind};
    use super::*;

    /// The sizes the state holds, derived from the shapes alone (no device): what `ModelState::alloc` asks for.
    fn expected(cfg: &MiniCpmConfig, slots: usize, pages: usize, rows: usize) -> StateBytes {
        let (gdn, full) = (cfg.linear_layers() as u64, cfg.full_layers() as u64);
        StateBytes {
            gdn_state: gdn * slots as u64 * (32 * 128 * 128 * 4),
            gdn_conv_history: gdn * slots as u64 * (3 * 8192 * 2),
            kv_target: full * pages as u64 * 2 * (256 * 2 * 256 * 2),
            gdn_records: gdn * rows as u64 * (2 * 16 * 128 * 2 + 32 * 128 * 2 + 2 * 32 * 4),
            gdn_raw_qkv: gdn * rows as u64 * 8192 * 2,
            prefix_snapshots: 0,
        }
    }

    /// `expected` with a prefix cache of `snapshots` snapshot slots: one sequence's state each.
    fn expected_cached(cfg: &MiniCpmConfig, slots: usize, pages: usize, rows: usize, snapshots: usize) -> StateBytes {
        let mut s = expected(cfg, slots, pages, rows);
        s.prefix_snapshots = cfg.linear_layers() as u64 * snapshots as u64 * (32 * 128 * 128 * 4 + 3 * 8192 * 2);
        s
    }

    /// The device layout's arithmetic, written from the kernels' shapes, is the memory model's list: for 1 to 8 slots, at the
    /// 40-layer model and at the 8-layer smoke pack, per kind and in total. (The device test of `tests/minicpmv4_7_boot.rs`
    /// compares the same list with the real buffers.)
    #[test]
    fn the_state_layout_equals_the_priced_allocations_at_one_to_eight_slots() {
        for layers in [40usize, 8] {
            let mut cfg = real();
            cfg.layers = layers;
            cfg.layer_types.truncate(layers);
            let items = allocations(&cfg, None);
            for slots in 1..=8usize {
                for (pages, rows) in [(1usize, 1usize), (1_028, 64), (300 + slots, 64)] {
                    let held = expected(&cfg, slots, pages, rows);
                    let diffs = compare_with_plan(&items, &held, slots, pages, rows);
                    assert!(diffs.is_empty(), "{layers} layers, {slots} slots, {pages} pages, {rows} rows: {diffs:?}");
                }
            }
        }
        // the 40-layer constants of the plan: 64,389,120 B a slot, 20,480 B a token, 990,720 B a window row (one record set)
        let c = real();
        let items = allocations(&c, None);
        let sum = |k| items.iter().filter(|i| i.kind == k).map(|i| i.bytes).sum::<u64>();
        assert_eq!(sum(AllocKind::PerSlot), 64_389_120);
        assert_eq!(sum(AllocKind::PerPage), 256 * 20_480);
        assert_eq!(sum(AllocKind::PerWindowRow), 990_720);
        assert_eq!(sum(AllocKind::PerSnapshot), 64_389_120);
    }

    /// E6b: the snapshot arena is a priced item, counted by the plan's snapshot slots; the pool is the whole pool (the cache's extra
    /// pages included). Every way of getting it wrong is named.
    #[test]
    fn the_prefix_arena_is_priced_per_snapshot_slot() {
        let cfg = real();
        let items = allocations(&cfg, None);
        for (slots, snapshots) in [(1usize, 2usize), (4, 7), (8, 130)] {
            let held = expected_cached(&cfg, slots, 1_028 + 640, 64, snapshots);
            assert_eq!(held.prefix_snapshots, snapshots as u64 * 64_389_120);
            assert!(compare_with_plan_cached(&items, &held, slots, 1_028 + 640, 64, snapshots).is_empty());
            assert_eq!(held.total(), expected(&cfg, slots, 1_028 + 640, 64).total() + held.prefix_snapshots);
        }
        let held = expected_cached(&cfg, 4, 100, 64, 7);
        // an arena of the wrong size, a plan that counts no snapshot, and a state that holds an arena the plan does not price
        let d = compare_with_plan_cached(&items, &held, 4, 100, 64, 8);
        assert!(d.len() == 1 && d[0].starts_with("prefix_snapshot: the device holds"), "{d:?}");
        let d = compare_with_plan(&items, &held, 4, 100, 64);
        assert!(d.iter().any(|m| m.starts_with("prefix_snapshot") && m.contains("x 0")), "{d:?}");
        let without: Vec<_> = items.iter().copied().filter(|i| i.name != "prefix_snapshot").collect();
        let d = compare_with_plan_cached(&without, &held, 4, 100, 64, 7);
        assert!(d.iter().any(|m| m.contains("prefix_snapshot") && m.contains("does not price it")), "{d:?}");
        // pages the cache added are pages of the pool: a pool of kv_blocks only is not the pool the cache needs
        let d = compare_with_plan_cached(&items, &expected_cached(&cfg, 4, 100, 64, 7), 4, 100 + 640, 64, 7);
        assert!(d.iter().any(|m| m.starts_with("kv_target")), "{d:?}");
    }

    #[test]
    fn a_state_that_disagrees_with_the_plan_is_named() {
        let cfg = real();
        let items = allocations(&cfg, None);
        let mut held = expected(&cfg, 4, 100, 64);
        assert!(compare_with_plan(&items, &held, 4, 100, 64).is_empty());
        // a Flash-Next-sized pool (48 value heads) and a second record set the plan does not price
        held.gdn_state = held.gdn_state / 32 * 48;
        held.gdn_records *= 2;
        let d = compare_with_plan(&items, &held, 4, 100, 64);
        assert_eq!(d.len(), 2, "{d:?}");
        assert!(d[0].starts_with("gdn_state: the device holds"), "{d:?}");
        assert!(d[1].starts_with("gdn_records: the device holds"), "{d:?}");
        // a slot count the pools were not made for
        let d = compare_with_plan(&items, &expected(&cfg, 4, 100, 64), 5, 100, 64);
        assert!(d.iter().any(|m| m.starts_with("gdn_state") && m.contains("x 5")), "{d:?}");
        // an allocation the plan does not price
        let without: Vec<_> = items.iter().copied().filter(|i| i.name != "gdn_raw_qkv").collect();
        let d = compare_with_plan(&without, &expected(&cfg, 1, 1, 1), 1, 1, 1);
        assert!(d.iter().any(|m| m.contains("does not price it")), "{d:?}");
    }
}
