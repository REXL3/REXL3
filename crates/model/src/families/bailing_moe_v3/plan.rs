//! The load plan: what a generic pipeline executes to put a Bailing pack on the device.
//!
//! A [`LoadPlan`] is plain data: the shard files, the device [`Region`]s to allocate, and an ordered list of [`LoadItem`]s, each
//! one `(tensor, part) -> (shard, offset, bytes) -> (region, offset)`. Nothing in it names a family or a kernel, and building it
//! touches no file and no GPU.
//!
//! Regions: one `Sidecar` for the embedding (which is mmap'd, never allocated), one `Global` for the final norm and `lm_head`,
//! per layer one `Trunk` for everything but the routed experts and one `Experts` for them.
//!
//! The fused KDA conv1d (`CatQkvConv`) is three items into one slot: the three source tensors are read at their own offsets and
//! written back to back at the fused slot.

use super::weights::{BailingWeights, Leaf, RegionId, Xform, ALIGN};
use gb_exl3::budget::{Bytes, Category};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    /// The embedding, addressed through an `mmap` of its shard (never a device allocation).
    Sidecar,
    /// The final norm and `lm_head`.
    Global,
    /// Every tensor of a layer but its routed experts (for the MTP layer, its `eh_proj` and norms too).
    Trunk(usize),
    /// A MoE layer's routed experts.
    Experts(usize),
}

impl std::fmt::Display for RegionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegionKind::Sidecar => write!(f, "sidecar"),
            RegionKind::Global => write!(f, "global"),
            RegionKind::Trunk(l) => write!(f, "layer{l}"),
            RegionKind::Experts(l) => write!(f, "experts{l}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub id: RegionId,
    pub kind: RegionKind,
    /// Bytes to allocate (a sidecar region is never allocated), a multiple of [`ALIGN`].
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemPart {
    Tensor,
    Trellis,
    Suh,
    Svh,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadItem {
    pub name: String,
    pub part: ItemPart,
    pub category: Category,
    pub file: u16,
    pub offset: u64,
    pub bytes: u64,
    pub xform: Xform,
    pub dst_bytes: u64,
    pub region: RegionId,
    pub dst: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarkerCheck {
    pub name: String,
    pub category: Category,
    pub file: u16,
    pub offset: u64,
    pub want: u32,
}

pub const MARKER_BYTES: u64 = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadExtent {
    pub file: u16,
    pub offset: u64,
    pub bytes: u64,
    pub items: std::ops::Range<usize>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlanTotals {
    pub items: usize,
    pub regions: usize,
    pub uploaded: Bytes,
    pub skipped: Bytes,
    pub region_bytes: u64,
}

impl PlanTotals {
    pub fn uploaded_total(&self) -> u64 {
        let b = &self.uploaded;
        b.routed + b.decoder + b.head + b.other + b.mtp + b.embedding + b.vision
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadPlan {
    pub files: Vec<String>,
    pub regions: Vec<Region>,
    /// Sorted by (file, offset).
    pub items: Vec<LoadItem>,
    /// Markers and other on-disk bytes the plan leaves out, by category.
    pub skipped: Bytes,
    pub markers: Vec<MarkerCheck>,
}

impl BailingWeights {
    /// The plan that loads this tree: one item per tensor and per uploaded part of every group, in (shard, offset) order, after
    /// checking it is internally consistent ([`LoadPlan::validate`]).
    pub fn load_plan(&self) -> Result<LoadPlan, String> {
        let mut items = Vec::new();
        let mut markers = Vec::new();
        let mut skipped = Bytes::default();
        let mut bad: Vec<String> = Vec::new();
        self.visit(&mut |leaf| match leaf {
            Leaf::Plain(t) => {
                if t.dst.region == RegionId::MAX {
                    // a sidecar-only source (the embedding): no device item
                    return;
                }
                items.push(LoadItem {
                    name: t.name.clone(),
                    part: ItemPart::Tensor,
                    category: t.category,
                    file: t.src.file,
                    offset: t.src.offset,
                    bytes: t.bytes,
                    xform: t.xform,
                    dst_bytes: t.dst_bytes,
                    region: t.dst.region,
                    dst: t.dst.offset,
                });
            }
            Leaf::Exl3(g) => {
                for (part, suffix, p) in
                    [(ItemPart::Trellis, "trellis", &g.trellis), (ItemPart::Suh, "suh", &g.suh), (ItemPart::Svh, "svh", &g.svh)]
                {
                    items.push(LoadItem {
                        name: format!("{}.{suffix}", g.prefix),
                        part,
                        category: g.category,
                        file: p.src.file,
                        offset: p.src.offset,
                        bytes: p.bytes,
                        xform: Xform::None,
                        dst_bytes: p.bytes,
                        region: p.dst.region,
                        dst: p.dst.offset,
                    });
                }
                if let Some((src, bytes)) = g.marker {
                    skipped.add(g.category, bytes);
                    markers.push(marker_of(g, src));
                }
            }
            Leaf::CatQkvConv(c) => {
                let elem = c.fused.dst_bytes / 3;
                for (i, p) in c.parts.iter().enumerate() {
                    items.push(LoadItem {
                        name: p.name.clone(),
                        part: ItemPart::Tensor,
                        category: p.category,
                        file: p.src.file,
                        offset: p.src.offset,
                        bytes: p.bytes,
                        xform: Xform::None,
                        dst_bytes: p.bytes,
                        region: c.fused.dst.region,
                        dst: c.fused.dst.offset + i as u64 * elem,
                    });
                }
            }
        });
        if !bad.is_empty() {
            return Err(bad.join("; "));
        }
        items.sort_by(|a, b| (a.file, a.offset).cmp(&(b.file, b.offset)));
        markers.sort_by(|a, b| (a.file, a.offset).cmp(&(b.file, b.offset)));
        let plan = LoadPlan { files: self.files.clone(), regions: self.regions.clone(), items, skipped, markers };
        plan.validate()?;
        Ok(plan)
    }
}

fn marker_of(g: &super::weights::Exl3Linear, src: super::weights::Src) -> MarkerCheck {
    MarkerCheck {
        name: format!("{}.{}", g.prefix, g.codebook.name()),
        category: g.category,
        file: src.file,
        offset: src.offset,
        want: match g.codebook {
            gb_exl3::Codebook::Mcg => gb_exl3::MARKER_MCG,
            _ => gb_exl3::MARKER_MUL1,
        },
    }
}

impl LoadPlan {
    pub fn items_in(&self, c: Category) -> impl Iterator<Item = &LoadItem> {
        self.items.iter().filter(move |i| i.category == c)
    }

    pub fn region(&self, id: RegionId) -> Option<&Region> {
        self.regions.iter().find(|r| r.id == id)
    }

    /// The device regions to allocate (everything but the mmap sidecar).
    pub fn device_regions(&self) -> Vec<&Region> {
        self.regions.iter().filter(|r| r.kind != RegionKind::Sidecar).collect()
    }

    /// Bytes the device must allocate: the regions (including alignment), not the sidecar.
    pub fn device_bytes(&self) -> u64 {
        self.device_regions().iter().map(|r| r.bytes).sum()
    }

    pub fn totals(&self) -> PlanTotals {
        let mut uploaded = Bytes::default();
        for i in &self.items {
            uploaded.add(i.category, i.bytes);
        }
        PlanTotals {
            items: self.items.len(),
            regions: self.regions.len(),
            uploaded,
            skipped: self.skipped,
            region_bytes: self.device_bytes(),
        }
    }

    pub fn select(&self, keep: impl Fn(Category) -> bool) -> LoadPlan {
        let items: Vec<LoadItem> = self.items.iter().filter(|i| keep(i.category)).cloned().collect();
        let used: std::collections::BTreeSet<RegionId> = items.iter().map(|i| i.region).collect();
        let mut skipped = Bytes::default();
        for c in [
            Category::RoutedExpert,
            Category::Decoder,
            Category::Head,
            Category::Other,
            Category::Mtp,
            Category::Embedding,
            Category::Vision,
        ] {
            if keep(c) {
                skipped.add(c, bytes_of(&self.skipped, c));
            }
        }
        LoadPlan {
            files: self.files.clone(),
            regions: self.regions.iter().filter(|r| used.contains(&r.id)).cloned().collect(),
            items,
            skipped,
            markers: self.markers.iter().filter(|m| keep(m.category)).cloned().collect(),
        }
    }

    pub fn markers_in(&self, e: &ReadExtent) -> &[MarkerCheck] {
        let key = |m: &MarkerCheck| (m.file, m.offset);
        let lo = self.markers.partition_point(|m| key(m) < (e.file, e.offset));
        let hi = self.markers.partition_point(|m| key(m) < (e.file, e.offset + e.bytes));
        let inside = &self.markers[lo..hi];
        match inside.last() {
            Some(m) if m.offset + MARKER_BYTES > e.offset + e.bytes => &inside[..inside.len() - 1],
            _ => inside,
        }
    }

    pub fn uncovered_markers(&self, extents: &[ReadExtent]) -> Vec<&MarkerCheck> {
        let mut covered = std::collections::BTreeSet::new();
        for e in extents {
            for m in self.markers_in(e) {
                covered.insert((m.file, m.offset));
            }
        }
        self.markers.iter().filter(|m| !covered.contains(&(m.file, m.offset))).collect()
    }

    pub fn extents(&self, max_gap: u64, max_bytes: u64) -> Vec<ReadExtent> {
        let mut out: Vec<ReadExtent> = Vec::new();
        for (idx, it) in self.items.iter().enumerate() {
            if let Some(e) = out.last_mut() {
                let end = e.offset + e.bytes;
                if e.file == it.file
                    && it.offset >= end
                    && it.offset - end <= max_gap
                    && it.offset + it.bytes - e.offset <= max_bytes
                {
                    e.bytes = it.offset + it.bytes - e.offset;
                    e.items.end = idx + 1;
                    continue;
                }
            }
            out.push(ReadExtent { file: it.file, offset: it.offset, bytes: it.bytes, items: idx..idx + 1 });
        }
        for e in &mut out {
            let next_item = self.items.get(e.items.end).filter(|i| i.file == e.file).map_or(u64::MAX, |i| i.offset);
            let mut at = self.markers.partition_point(|m| (m.file, m.offset) < (e.file, e.offset + e.bytes));
            while let Some(m) = self.markers.get(at) {
                if m.file != e.file || m.offset - (e.offset + e.bytes) > max_gap || m.offset + MARKER_BYTES > next_item {
                    break;
                }
                e.bytes = m.offset + MARKER_BYTES - e.offset;
                at += 1;
            }
        }
        out
    }

    pub fn validate(&self) -> Result<(), String> {
        let known: BTreeMap<RegionId, &Region> = self.regions.iter().map(|r| (r.id, r)).collect();
        if known.len() != self.regions.len() {
            return Err("two regions share an id".into());
        }
        for r in &self.regions {
            if r.kind != RegionKind::Sidecar && r.bytes % ALIGN != 0 {
                return Err(format!("region {} ({}) has {} bytes, not a multiple of {ALIGN}", r.id, r.kind, r.bytes));
            }
        }
        for w in self.items.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            if (a.file, a.offset) > (b.file, b.offset) {
                return Err(format!("items are not sorted: {} before {}", a.name, b.name));
            }
            if a.file == b.file && a.offset + a.bytes > b.offset {
                return Err(format!("{} and {} overlap in shard {}", a.name, b.name, a.file));
            }
        }
        for w in self.markers.windows(2) {
            if (w[0].file, w[0].offset) >= (w[1].file, w[1].offset) {
                return Err(format!("markers are not sorted: {} before {}", w[0].name, w[1].name));
            }
        }
        for m in &self.markers {
            let at = self.items.partition_point(|i| (i.file, i.offset + i.bytes) <= (m.file, m.offset));
            if self.items.get(at).is_some_and(|i| i.file == m.file && i.offset < m.offset + MARKER_BYTES) {
                return Err(format!("marker {} overlaps an item", m.name));
            }
        }
        let mut by_region: BTreeMap<RegionId, Vec<(u64, u64, &str)>> = BTreeMap::new();
        for i in &self.items {
            if usize::from(i.file) >= self.files.len() {
                return Err(format!("{} names shard {} of {}", i.name, i.file, self.files.len()));
            }
            let Some(r) = known.get(&i.region) else {
                return Err(format!("{} is placed in region {}, which the plan does not have", i.name, i.region));
            };
            if r.kind == RegionKind::Sidecar {
                return Err(format!("{} is placed in the sidecar region, which is never allocated", i.name));
            }
            if i.dst % ALIGN != 0 {
                return Err(format!("{} is placed at {} in {}, not aligned to {ALIGN}", i.name, i.dst, r.kind));
            }
            if i.xform.dst_bytes(i.bytes) != i.dst_bytes {
                return Err(format!(
                    "{} reads {} bytes through {:?} and says it takes {} on the device",
                    i.name, i.bytes, i.xform, i.dst_bytes
                ));
            }
            if i.dst + i.dst_bytes > r.bytes {
                return Err(format!(
                    "{} ({} bytes at {}) runs past region {} ({} bytes)",
                    i.name, i.dst_bytes, i.dst, r.kind, r.bytes
                ));
            }
            by_region.entry(i.region).or_default().push((i.dst, i.dst_bytes, &i.name));
        }
        for (id, mut v) in by_region {
            v.sort();
            for w in v.windows(2) {
                if w[0].0 + w[0].1 > w[1].0 {
                    return Err(format!("{} and {} overlap in region {}", w[0].2, w[1].2, known[&id].kind));
                }
            }
        }
        Ok(())
    }
}

pub fn bytes_of(b: &Bytes, c: Category) -> u64 {
    match c {
        Category::RoutedExpert => b.routed,
        Category::Decoder => b.decoder,
        Category::Head => b.head,
        Category::Other => b.other,
        Category::Mtp => b.mtp,
        Category::Embedding => b.embedding,
        Category::Vision => b.vision,
    }
}
