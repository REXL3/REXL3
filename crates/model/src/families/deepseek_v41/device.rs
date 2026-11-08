//! The device-resident weight set of DeepSeek-V4.1-Flash: every tensor of the typed tree ([`super::weights`]) as the kernel that reads it
//! wants it.
//!
//! Three kinds of binding, and the layout ([`super::load::Layout`]) has already decided which tensor is which:
//!
//! * **FP8 block matrices** — the MLA's five projections a layer, the shared expert's three, the engram's fused `wkv` — become
//!   [`Fp8Weights`], uploaded from the shard's bytes through `Fp8Weights::new`. The kernel type owns its buffers and exposes no
//!   constructor from a `DevBuf`, so these are read from the shard here rather than being views of a region; [`super::load`]'s header
//!   names the deviation and the layout leaves them out of the regions so that nothing is allocated twice.
//! * **EXL3 experts** — [`Exl3Weights::from_parts`] over four *views* of the slab: the trellis, the two fp16 scale vectors and the
//!   codebook marker. Nothing is copied for these; the slab fill is the device copy.
//! * **Raw device buffers** — the embedding and the head (bf16, read as stored by the embedding and `lm_head` kernels), the engram
//!   branch's `q_weight`/`k_weight` (bf16), and the NextN layers' everything (the DSpark drafter is not built: they are loaded and
//!   accounted for, and nothing runs them yet).
//!
//! **The engram tables are mapped, not copied.** [`EngramDevice`] holds an [`EngramState`], whose two tables are `mmap`s of their shards
//! (202.8 GB); the forward addresses rows through the mapping. No code path here builds a `Vec` of a table's contents.
//!
//! **The geometry is checked, not assumed.** [`dsv4_config`] translates [`DeepseekV41Config`] into the kernel crate's [`Dsv4Config`] and
//! [`bind`] refuses a config the `MlaDsv4` kernels were not compiled for (`hc_mult` other than 4, more than one KV head): a mismatch there
//! would decode noise rather than fail, so it fails at bind time instead.

use super::config::DeepseekV41Config;
use super::engram::EngramState;
use super::load::{Layout, Regions};
use super::weights::{DeepseekV41Weights, MtpExpert, Tensor};
use crate::error::ModelError;
use gb_cuda::dsv4_attn::{CompressorPack, Dsv4Config, IndexerPack, MlaDsv4, MlaDsv4Pack, SharedRope};
use gb_cuda::dsv4_moe::{Dsv4Dims, Dsv4Router, HcSite};
use gb_cuda::exl3_linear::{Exl3Dims, Exl3Weights};
use gb_cuda::expert_cache::{ExpertCache, Lease};
use gb_cuda::fp8_linear::Fp8Weights;
use gb_cuda::DevBuf;
use gb_exl3::checkpoint::{Checkpoint, Group};
use std::sync::Arc;

/// The shared expert's three FP8 matrices: `down(act(gate(x)) * up(x))`.
pub struct Dsv4Mlp {
    pub w1: Fp8Weights,
    pub w2: Fp8Weights,
    pub w3: Fp8Weights,
}

impl Dsv4Mlp {
    pub fn bytes(&self) -> u64 {
        (self.w1.weight_bytes() + self.w2.weight_bytes() + self.w3.weight_bytes()) as u64
    }
}

/// Where one projection of a streamed expert sits inside its cache record, and the geometry needed to build the kernel type over it.
///
/// Offsets, not bytes: a slot is filled from the pack by [`gb_cuda::expert_cache::PackExpertSource`] and read here, so nothing is copied
/// twice and no per-expert allocation exists.
#[derive(Clone, Copy, Debug)]
pub struct Exl3Slot {
    pub dims: Exl3Dims,
    pub trellis: u64,
    pub suh: u64,
    pub svh: u64,
}

impl Exl3Slot {
    /// Where projection `g` of one expert lands in that expert's cache record: its own dims and bit width (the pack is mixed-K, so
    /// two experts of one layer differ), and the `trellis`/`suh`/`svh` offsets of [`super::weights::Expert::record_layout`] for
    /// this projection (`offs` is that projection's row: trellis, mul1, suh, svh).
    fn of(g: &Group, offs: [u64; 4]) -> Result<Self, ModelError> {
        let dims = Exl3Dims {
            k: g.k,
            n: g.n,
            bits: Exl3Weights::bits_of(g).map_err(|e| ModelError::Config(e.to_string()))?,
            codebook: g.codebook,
        };
        if g.bias {
            // a bias would sit after `svh`; the pack's experts have none, and a slot that silently skipped it would decode wrong
            return Err(ModelError::Config(format!("{}: a streamed expert with a bias is not supported", g.prefix)));
        }
        Ok(Self { dims, trellis: offs[0], suh: offs[2], svh: offs[3] })
    }
}

/// The three groups of one streamed expert, and the lease that keeps their slot from being evicted while a GEMM reads it.
pub struct StreamedGroups<'a> {
    lease: Lease<'a>,
    pub groups: Vec<Exl3Weights>,
}

impl<'a> StreamedGroups<'a> {
    pub fn slot(&self) -> u32 {
        self.lease.slot()
    }

    /// The groups and the lease that pins their slot, for a caller that must hold the pin for as long as it reads them.
    pub fn into_parts(self) -> (Vec<Exl3Weights>, Lease<'a>) {
        (self.groups, self.lease)
    }
}

/// A routed expert's three groups, however it is stored.
pub enum ExpertGroups<'a> {
    /// In the slab, bound at load time.
    Resident(&'a [Exl3Weights]),
    /// Past the residency budget: filled into a cache slot on first touch and leased for the duration of the GEMM.
    Streamed(StreamedGroups<'a>),
}

impl ExpertGroups<'_> {
    pub fn get(&self) -> &[Exl3Weights] {
        match self {
            ExpertGroups::Resident(g) => g,
            ExpertGroups::Streamed(s) => &s.groups,
        }
    }
}

/// One layer's routed experts: the ones the slab holds, bound in place, and the cache that streams the rest.
pub struct ExpertsDevice {
    /// The trunk layer this bank belongs to: the cache is keyed by `(layer, expert)`, so a cache shared by 40 layers must be told which.
    pub layer: usize,
    /// `resident[e]` is the three groups of expert `e` (`w1`, `w2`, `w3`), in place in the slab; `None` for an expert that streams.
    pub resident: Vec<Option<Vec<Exl3Weights>>>,
    /// The streamed tail, when the budget does not hold every expert. `None` with every expert resident.
    pub streamed: Option<Arc<ExpertCache>>,
    /// Where each expert's three groups land inside its cache record, by expert id: each expert's own bit widths and offsets
    /// (the pack is mixed-K, so one layer's experts do not share a geometry).
    pub slots: Vec<[Exl3Slot; 3]>,
    /// Device descriptor table for `moe_experts_resident` when all experts in this layer are resident.
    pub fast_desc: Option<DevBuf>,
}

impl ExpertsDevice {
    pub fn resident_count(&self) -> usize {
        self.resident.iter().filter(|e| e.is_some()).count()
    }

    /// The three groups of `expert`, resident or leased from the cache. A routed expert is never `NotBuilt`: this is the path that replaced
    /// that error.
    pub fn groups(&self, expert: usize) -> Result<ExpertGroups<'_>, ModelError> {
        if let Some(Some(g)) = self.resident.get(expert) {
            return Ok(ExpertGroups::Resident(g));
        }
        let cache = self.streamed.as_ref().ok_or_else(|| {
            ModelError::Config(format!(
                "deepseek_v41: expert {expert} is neither resident nor streamed, and the layer has no cache"
            ))
        })?;
        let lease = cache
            .get(self.layer as u16, expert as u16)
            .map_err(|e| ModelError::Config(format!("layer {} expert {expert}: {e}", self.layer)))?;
        let slot = lease.slot();
        let geometry = self.slots.get(expert).ok_or_else(|| {
            ModelError::Config(format!("deepseek_v41: layer {} has no cache geometry for expert {expert}", self.layer))
        })?;
        let mut groups = Vec::with_capacity(3);
        for s in geometry {
            let d = s.dims;
            let view = |off: u64, bytes: usize| {
                cache.slot_view(slot, off as usize, bytes).map_err(|e| ModelError::Config(e.to_string()))
            };
            let trellis = view(s.trellis, (d.k / 16) * (d.n / 16) * 16 * d.bits * 2)?;
            let suh = view(s.suh, d.k * 2)?;
            let svh = view(s.svh, d.n * 2)?;
            groups.push(Exl3Weights::from_dims(d, trellis, suh, svh, None).map_err(|e| ModelError::Config(e.to_string()))?);
        }
        Ok(ExpertGroups::Streamed(StreamedGroups { lease, groups }))
    }

    /// Queue a fill for `expert` if it is not resident. Never blocks: the router's selection goes out on the queue and the fills land while
    /// the shared expert and the hyper-connection post run.
    pub fn prefetch(&self, expert: usize) -> Result<(), ModelError> {
        if self.resident.get(expert).is_some_and(|e| e.is_some()) {
            return Ok(());
        }
        if let Some(c) = &self.streamed {
            c.prefetch(self.layer as u16, expert as u16).map_err(|e| ModelError::Config(e.to_string()))?;
        }
        Ok(())
    }
}

/// One trunk layer's tensors.
pub struct LayerDevice {
    pub index: usize,
    /// fp32 `[hidden]` RMSNorm gammas (the pack stores bf16; every kernel takes fp32).
    pub attn_norm: DevBuf,
    pub ffn_norm: DevBuf,
    pub mla: MlaDsv4,
    pub router: Dsv4Router,
    pub shared: Dsv4Mlp,
    pub experts: ExpertsDevice,
    pub hc_attn: HcSite,
    pub hc_ffn: HcSite,
}

/// One engram layer: the `mmap`ped table, the n-gram hash and the fused `wkv`. The branch runs on the CPU (`super::model`), because the
/// table is mapped rather than resident and a gather from it is a page fault, not a copy to the device.
pub struct EngramDevice {
    pub layer: usize,
    pub state: EngramState,
    pub wkv: Fp8Weights,
    pub q_weight: DevBuf,
    pub k_weight: DevBuf,
}

/// One NextN layer, as loaded: every tensor a raw view. The DSpark drafter (the Markov head, the confidence head, the `main_proj` tap) is
/// **not built** — nothing runs these layers yet, and the plan's 7.9 GB of them are loaded so that a boot either has them or says which is
/// missing.
pub struct MtpDevice {
    pub index: usize,
    /// `(name, buffer)` in tree order, so a later commit that binds them into kernel types starts from the same regions.
    pub tensors: Vec<(String, DevBuf)>,
    /// The layer's routed experts as kernel types, when the pack stores them as EXL3 (`K2` in the re-upload): `experts[e]` is
    /// `w1`, `w2`, `w3` in place in the layer's NextN expert region. `None` for the first upload's int8 experts, which the
    /// drafter reaches through `tensors`. These are views of regions already counted in `tensors` — no bytes are added here.
    pub experts: Option<Vec<Vec<Exl3Weights>>>,
}

impl MtpDevice {
    /// The three groups of expert `e`, when this layer's experts are EXL3.
    pub fn expert(&self, e: usize) -> Option<&[Exl3Weights]> {
        self.experts.as_ref()?.get(e).map(Vec::as_slice)
    }

    /// How many of the layer's experts are bound as EXL3 groups (`0` for the first upload's int8 pack).
    pub fn exl3_experts(&self) -> usize {
        self.experts.as_ref().map_or(0, Vec::len)
    }
}

/// The bound LM head: a bf16 `[vocab, hidden]` view in `Global`, or an EXL3 group view over the same region.
pub enum HeadDevice {
    Bf16(DevBuf),
    Exl3(Exl3Weights),
}

impl HeadDevice {
    pub fn bytes(&self) -> u64 {
        match self {
            HeadDevice::Bf16(b) => b.bytes() as u64,
            HeadDevice::Exl3(w) => (w.weight_bytes() + 4) as u64,
        }
    }
}

/// Every weight of the text model on the device.
pub struct Dsv41Device {
    /// bf16 `[vocab, hidden]`.
    pub embed: DevBuf,
    /// fp32 `[hidden]`.
    pub final_norm: DevBuf,
    /// bf16 `[vocab, hidden]` or EXL3 group `k = hidden, n = vocab`: the untied head.
    pub head: HeadDevice,
    pub layers: Vec<LayerDevice>,
    pub mtp: Vec<MtpDevice>,
    pub mtp_gpu: Option<gb_cuda::dsv4_mtp_gpu::DsparkGpu>,
    pub engram: Vec<EngramDevice>,
    /// What the kernel constructors uploaded rather than viewed, for the boot report.
    pub uploaded_bytes: u64,
    /// The device bytes of the distinct RoPE tables, counted once each however many layers share them. Summing
    /// [`LayerDevice::mla`]'s own `bytes()` would count a shared table once a layer.
    pub rope_bytes: u64,
}

impl Dsv41Device {
    pub fn bytes(&self) -> u64 {
        let l: u64 = self
            .layers
            .iter()
            .map(|l| l.attn_norm.bytes() as u64 + l.ffn_norm.bytes() as u64 + l.mla.bytes() as u64 + l.shared.bytes())
            .sum();
        let m: u64 = self.mtp.iter().flat_map(|m| m.tensors.iter().map(|(_, b)| b.bytes() as u64)).sum();
        let eg: u64 =
            self.engram.iter().map(|e| e.wkv.weight_bytes() as u64 + e.q_weight.bytes() as u64 + e.k_weight.bytes() as u64).sum();
        self.embed.bytes() as u64 + self.final_norm.bytes() as u64 + self.head.bytes() + l + m + eg + self.rope_bytes
    }

    pub fn resident_experts(&self) -> usize {
        self.layers.iter().map(|l| l.experts.resident_count()).sum()
    }
}

/// The kernel crate's view of this pack's MLA geometry, from the family's config.
///
/// The `MlaDsv4` kernels bake the head count, the latent width and the group count into their block shapes; a config that disagrees is
/// refused here rather than decoded into noise.
pub fn dsv4_config(cfg: &DeepseekV41Config) -> Result<Dsv4Config, ModelError> {
    if cfg.kv_heads != 1 {
        return Err(ModelError::Config(format!(
            "deepseek_v41: the MLA kernels take one shared KV head, the config states {}",
            cfg.kv_heads
        )));
    }
    if cfg.heads % cfg.o_groups != 0 {
        return Err(ModelError::Config(format!(
            "deepseek_v41: {} heads do not divide into {} output groups",
            cfg.heads, cfg.o_groups
        )));
    }
    let mut ratios: Vec<u8> = cfg.compress.ratios.iter().map(|&r| r.min(255) as u8).collect();
    ratios.truncate(cfg.layers + cfg.nextn);
    Ok(Dsv4Config {
        dim: cfg.hidden,
        n_heads: cfg.heads,
        head_dim: cfg.head_dim,
        rope_dim: cfg.qk_rope_head_dim,
        q_lora: cfg.q_lora_rank,
        o_lora: cfg.o_lora_rank,
        o_groups: cfg.o_groups,
        window: cfg.sliding_window,
        n_layers: cfg.layers + cfg.nextn,
        index_heads: cfg.indexer.n_heads,
        index_dim: cfg.indexer.head_dim,
        index_topk: cfg.indexer.topk,
        compress_ratios: ratios,
        kv_source_layers: cfg.compress.source_layer_ids.clone(),
        index_source_layers: cfg.indexer.source_layer_ids.clone(),
        candidate_source_layer: cfg.indexer.candidate_source_layer_id,
        candidate_topk_blocks: cfg.indexer.candidate_topk_blocks,
        candidate_block_size: cfg.indexer.candidate_block_size,
        rope_theta: cfg.rope.theta,
        compress_rope_theta: cfg.rope.compress_theta,
        rope_factor: cfg.rope.yarn.factor,
        original_seq_len: cfg.rope.yarn.original_max_positions,
        beta_fast: cfg.rope.yarn.beta_fast,
        beta_slow: cfg.rope.yarn.beta_slow,
        eps: cfg.rms_eps as f32,
    })
}

struct Binder<'a> {
    cfg: &'a DeepseekV41Config,
    w: &'a DeepseekV41Weights,
    ck: &'a Checkpoint,
    layout: &'a Layout,
    regions: &'a Regions,
    /// The rank's split table, `None` on the single node: the only tensors it cuts at bind time are the FP8 `wq_b` (its row
    /// slice, weight and scale rows together — a valid FP8 matrix of its own, the boundary a multiple of the 32-row scale
    /// block) and the NextN experts (never bound on a rank: the DSpark stages are refused under tensor parallelism).
    shard: Option<&'a super::tp::ShardTable>,
    preloaded: std::cell::RefCell<std::collections::BTreeMap<String, Vec<u8>>>,
}

fn err(m: String) -> ModelError {
    ModelError::Config(m)
}

/// The attention sink rows of the heads this device computes: the sink is PER HEAD — one value per query head —
/// and the kernel indexes it by LOCAL head, so a rank of a group binds exactly [`ShardTable::q_head_range`]'s
/// values. Binding the whole `[64]` tensor on every rank (the first live TP boot) fed heads 32..63 the sinks of
/// heads 0..31: every attention slightly off, identically on both ranks, ~2% off the single rank — the exactness
/// oracle caught it at block 0. The single rank (and an unsharded bind) keeps the whole tensor.
fn sink_slice<'a>(sink: &'a [f32], shard: Option<&super::tp::ShardTable>) -> &'a [f32] {
    let Some(sh) = shard else { return sink };
    if sh.is_single() {
        return sink;
    }
    let heads = sh.q_head_range();
    assert_eq!(
        sink.len(),
        sh.q_heads,
        "deepseek_v41: `attn_sink` is one value per head ({}) of the config, found {}",
        sh.q_heads,
        sink.len()
    );
    &sink[heads]
}

impl Binder<'_> {
    /// A device view of one tensor of the plan.
    fn view(&self, name: &str) -> Result<DevBuf, ModelError> {
        self.regions.tensor(self.layout, name)
    }

    /// The shard's bytes of one plain tensor.
    fn read(&self, name: &str) -> Result<Vec<u8>, ModelError> {
        if let Some(v) = self.preloaded.borrow_mut().remove(name) {
            return Ok(v);
        }
        let e = self.ck.plain.get(name).ok_or_else(|| err(format!("deepseek_v41: `{name}` is not in the pack")))?;
        let out = self.ck.read_tensor(e).map_err(|e| err(format!("{name}: {e}")))?;
        #[cfg(target_os = "linux")]
        if out.len() >= (16 << 20) {
            if let Ok(off) = self.ck.file_offset(e) {
                if let Ok(f) = std::fs::File::open(self.ck.root.join(&e.file)) {
                    use std::os::unix::io::AsRawFd;
                    const PAGE: u64 = 4096;
                    let a0 = (off & !(PAGE - 1)) as libc::off_t;
                    let a1 = ((off + out.len() as u64) & !(PAGE - 1)) as libc::off_t;
                    if a1 > a0 {
                        unsafe {
                            libc::posix_fadvise(f.as_raw_fd(), a0, a1 - a0, libc::POSIX_FADV_DONTNEED);
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// A bf16 gamma widened to fp32 (exactly: bf16 is the top half of an f32).
    fn gamma(&self, name: &str) -> Result<DevBuf, ModelError> {
        let raw = self.read(name)?;
        if raw.len() % 2 != 0 {
            return Err(err(format!("{name}: {} bytes is not a bf16 vector", raw.len())));
        }
        let v: Vec<f32> = raw.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect();
        Ok(DevBuf::from_slice(&v)?)
    }

    /// A float vector widened to fp32 by the tensor's own dtype: F32 as stored, BF16 widened exactly. The pack keeps
    /// `attn_sink`, the router biases and the hyper-connection tensors in F32 but `q_norm`/`kv_norm` in BF16; reading a BF16
    /// vector as raw f32 bytes gave half the elements, each two bf16 values glued together, and the q_norm kernel then read
    /// past the end of its gamma. Any other dtype is refused by name.
    fn f32s(&self, name: &str) -> Result<Vec<f32>, ModelError> {
        let e = self.ck.plain.get(name).ok_or_else(|| err(format!("deepseek_v41: `{name}` is not in the pack")))?;
        let numel: usize = e.shape.iter().product::<usize>().max(1);
        let raw = self.read(name)?;
        let v: Vec<f32> = match e.dtype.as_str() {
            "F32" => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            "BF16" => raw.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect(),
            d => return Err(err(format!("{name}: {d} is not an F32 or BF16 vector"))),
        };
        if v.len() != numel {
            return Err(err(format!("{name}: {} values read, the header says {numel}", v.len())));
        }
        Ok(v)
    }

    fn bf16_bits(&self, name: &str) -> Result<Vec<u16>, ModelError> {
        let raw = self.read(name)?;
        Ok(raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
    }

    /// An `[n, k]` FP8 block matrix and its E8M0 scales (or an EXL3 `[k, n]` group), as the kernel takes them.
    fn fp8(&self, prefix: &str) -> Result<Fp8Weights, ModelError> {
        if let Some(group) = self.ck.groups.get(prefix) {
            let exl = self.exl3(prefix, group.k, group.n)?;
            return Fp8Weights::from_exl3(exl).map_err(|e| err(format!("{prefix}: {e}")));
        }
        let weight = format!("{prefix}.weight");
        let e = self.ck.plain.get(&weight).ok_or_else(|| err(format!("deepseek_v41: `{weight}` is not in the pack")))?;
        let (n, k) = match e.shape.as_slice() {
            [n, k] => (*n, *k),
            other => return Err(err(format!("{weight}: expected a matrix, found {other:?}"))),
        };
        let w = self.read(&weight)?;
        let s = self.read(&format!("{prefix}.scale"))?;
        Fp8Weights::new(n, k, &w, &s).map_err(|e| err(format!("{prefix}: {e}")))
    }

    /// One EXL3 group, over four views of the slab.
    fn exl3(&self, prefix: &str, k: usize, n: usize) -> Result<Exl3Weights, ModelError> {
        let group: Group = self
            .ck
            .groups
            .get(prefix)
            .cloned()
            .ok_or_else(|| err(format!("deepseek_v41: `{prefix}` is not an EXL3 group of the pack")))?;
        let part = |p: &str| self.view(&format!("{prefix}.{p}"));
        let w = Exl3Weights::from_parts(&group, part("trellis")?, part("suh")?, part("svh")?, None)
            .map_err(|e| err(format!("{prefix}: {e}")))?;
        if (w.k, w.n) != (k, n) {
            return Err(err(format!("{prefix}: the group is {}x{}, the config says {k}x{n}", w.k, w.n)));
        }
        Ok(w)
    }

    fn mlp(&self, prefix: &str) -> Result<Dsv4Mlp, ModelError> {
        Ok(Dsv4Mlp {
            w1: self.fp8(&format!("{prefix}w1"))?,
            w2: self.fp8(&format!("{prefix}w2"))?,
            w3: self.fp8(&format!("{prefix}w3"))?,
        })
    }

    fn site(&self, prefix: &str) -> Result<HcSite, ModelError> {
        let m = self.cfg.hc.mult;
        let fnw = self.f32s(&format!("{prefix}fn"))?;
        let base = self.f32s(&format!("{prefix}base"))?;
        let scale = self.f32s(&format!("{prefix}scale"))?;
        if fnw.len() != 6 * m * m * self.cfg.hidden || base.len() != 6 * m || scale.len() != 3 {
            return Err(err(format!(
                "{prefix}: an mHC site is [{}] x [{}] + [{}] wide, found {} x {} + {}",
                6 * m,
                m * self.cfg.hidden,
                3,
                fnw.len(),
                base.len(),
                scale.len()
            )));
        }
        let s: [f32; 3] = [scale[0], scale[1], scale[2]];
        HcSite::new(&fnw, &base, &s).map_err(|e| err(format!("{prefix}: {e}")))
    }

    fn layer(&self, index: usize, dcfg: &Dsv4Config, rope: &SharedRope) -> Result<LayerDevice, ModelError> {
        let cfg = self.cfg;
        let p = format!("layers.{index}.");
        let attn = format!("{p}attn.");
        let ffn = format!("{p}ffn.");
        // the MLA's ten byte tensors and three fp32 vectors, borrowed for the constructor
        let (wq_a, wq_a_s) = (self.read(&format!("{attn}wq_a.weight"))?, self.read(&format!("{attn}wq_a.scale"))?);
        let has_exl3_wqb = self.ck.groups.contains_key(&format!("{attn}wq_b"));
        let has_exl3_wob = self.ck.groups.contains_key(&format!("{attn}wo_b"));
        let (wq_b, wq_b_s) = if has_exl3_wqb {
            (Vec::new(), Vec::new())
        } else {
            match self.shard {
                // a rank of a group holds its heads' row slice of `wq_b` — the weight rows `[heads*dim r/R, …(r+1)/R)` and the
                // matching scale rows (the cut is a multiple of `head_dim`, hence of the 32-row scale block, so the slice is a
                // valid FP8 matrix of its own) — and computes only those heads (`MlaDsv4`'s `q_heads`, derived from this shape)
                Some(s) => {
                    let k = cfg.q_lora_rank;
                    let r = s.o_row_range();
                    if r.start % 32 != 0 || r.end % 32 != 0 {
                        return Err(err(format!(
                            "deepseek_v41: layer {index}'s `wq_b` row slice [{}, {}) is not a multiple of the 32-row FP8 scale block",
                            r.start, r.end
                        )));
                    }
                    (
                        self.read(&format!("{attn}wq_b.weight"))?[r.start * k..r.end * k].to_vec(),
                        self.read(&format!("{attn}wq_b.scale"))?[(r.start / 32) * (k / 32)..(r.end / 32) * (k / 32)].to_vec(),
                    )
                }
                None => (self.read(&format!("{attn}wq_b.weight"))?, self.read(&format!("{attn}wq_b.scale"))?),
            }
        };
        let (wkv, wkv_s) = (self.read(&format!("{attn}wkv.weight"))?, self.read(&format!("{attn}wkv.scale"))?);
        let (wo_a, wo_a_s) = (self.read(&format!("{attn}wo_a.weight"))?, self.read(&format!("{attn}wo_a.scale"))?);
        let (wo_b, wo_b_s) = if has_exl3_wob {
            (Vec::new(), Vec::new())
        } else {
            (self.read(&format!("{attn}wo_b.weight"))?, self.read(&format!("{attn}wo_b.scale"))?)
        };
        let q_norm = self.f32s(&format!("{attn}q_norm.weight"))?;
        let kv_norm = self.f32s(&format!("{attn}kv_norm.weight"))?;
        let sink_vec = self.f32s(&format!("{attn}attn_sink"))?;
        // the sink is PER HEAD: a rank of a group holds its heads' values only, so the kernel's local-head indexing
        // reads the right ones (a whole [64] sink on a rank that computes heads [32,64) would feed heads 32..63 the
        // sinks of heads 0..31 — every attention slightly off, identical on both ranks, ~2% off the single rank; the
        // first live boot's exactness oracle found it at block 0)
        let sink = sink_slice(&sink_vec, self.shard);
        // The compressor and the DSA indexer exist only on the layers that own one (`kv_source_layer_ids` for the pool and
        // the index keys, `index_source_layer_ids` for the query side). Their four small matrices are BF16 in the pack —
        // no `.scale` beside any of them — and `MlaDsv4::with_rope` is what quantises them into the FP8 block format the
        // kernels take; `indexer.wq_b` is the one that is stored FP8 already.
        let c_wkv = if cfg.compress.is_source(index) { Some(self.read(&format!("{attn}compressor.wkv.weight"))?) } else { None };
        let c_wgate =
            if cfg.compress.has_gate(index) { Some(self.read(&format!("{attn}compressor.wgate.weight"))?) } else { None };
        let c_norm =
            if cfg.compress.is_source(index) { Some(self.f32s(&format!("{attn}compressor.norm.weight"))?) } else { None };
        let (i_w, i_s) = if cfg.indexer.is_source(index) {
            (Some(self.read(&format!("{attn}indexer.wq_b.weight"))?), Some(self.read(&format!("{attn}indexer.wq_b.scale"))?))
        } else {
            (None, None)
        };
        let i_wproj =
            if cfg.indexer.is_source(index) { Some(self.read(&format!("{attn}indexer.weights_proj.weight"))?) } else { None };
        let i_wk = if cfg.compress.is_source(index) { Some(self.read(&format!("{attn}indexer.wk.weight"))?) } else { None };
        // a query-only index source (24, 28, ...) builds no index keys — the pool owners' `index_k` plane is shared — so its
        // `k_norm` is the empty slice rather than a tensor the pack does not carry for it
        let i_knorm =
            if cfg.compress.is_source(index) { self.f32s(&format!("{attn}indexer.k_norm.weight"))? } else { Vec::new() };
        let compressor = match (&c_wkv, &c_norm) {
            (Some(wkv), Some(norm)) => Some(CompressorPack { wkv, wgate: c_wgate.as_deref(), norm }),
            _ => None,
        };
        let indexer = match (&i_w, &i_s, &i_wproj) {
            (Some(w), Some(s), Some(wp)) => {
                Some(IndexerPack { wq_b: (w, s), weights_proj: wp, wk: i_wk.as_deref(), k_norm: &i_knorm })
            }
            _ => None,
        };
        let pack = MlaDsv4Pack {
            wq_a: (&wq_a, &wq_a_s),
            wq_b: (&wq_b, &wq_b_s),
            wkv: (&wkv, &wkv_s),
            wo_a: (&wo_a, &wo_a_s),
            wo_b: (&wo_b, &wo_b_s),
            q_norm: &q_norm,
            kv_norm: &kv_norm,
            sink,
            compressor,
            indexer,
        };
        let mut mla = MlaDsv4::new_shared(dcfg, index, &pack, rope).map_err(|e| err(format!("layer {index}'s MLA: {e}")))?;
        if has_exl3_wqb {
            let mut exl = self.exl3(&format!("{attn}wq_b"), cfg.q_lora_rank, cfg.heads * cfg.head_dim)?;
            if let Some(s) = self.shard {
                let r = s.o_row_range();
                if r.start % 128 != 0 || r.end % 128 != 0 {
                    return Err(err(format!(
                        "deepseek_v41: layer {index}'s EXL3 `wq_b` row slice [{}, {}) is not a multiple of 128",
                        r.start, r.end
                    )));
                }
                exl = exl
                    .select_strips(&[(r.start / 128, r.end / 128)])
                    .map_err(|e| err(format!("layer {index}'s EXL3 wq_b slice: {e}")))?;
            }
            mla.q_heads = exl.n / cfg.head_dim;
            mla.wq_b = Fp8Weights::from_exl3(exl).map_err(|e| err(format!("layer {index}'s EXL3 wq_b: {e}")))?;
        }
        if has_exl3_wob {
            let exl = self.exl3(&format!("{attn}wo_b"), cfg.o_groups * cfg.o_lora_rank, cfg.hidden)?;
            mla.wo_b = Fp8Weights::from_exl3(exl).map_err(|e| err(format!("layer {index}'s EXL3 wo_b: {e}")))?;
        }
        let dims = Dsv4Dims {
            hidden: cfg.hidden,
            experts: cfg.moe.routed_experts,
            topk: cfg.moe.experts_per_token,
            scale: cfg.moe.routed_scaling_factor as f32,
            gate_temp: 1.0,
        };
        let router = Dsv4Router::new(
            &self.bf16_bits(&format!("{ffn}gate.weight"))?,
            &self.f32s(&format!("{ffn}gate.bias"))?,
            &self.f32s(&format!("{ffn}gate.bias_vl"))?,
            dims,
        )
        .map_err(|e| err(format!("layer {index}'s router: {e}")))?;
        let shared = self.mlp(&format!("{ffn}shared_experts."))?;
        let inter = cfg.moe.intermediate;
        // where each expert's three groups land in its cache record: that expert's own groups (bit width) and its own record walk
        let experts_w = &self
            .w
            .layers
            .get(index)
            .ok_or_else(|| err(format!("deepseek_v41: layer {index} is not in the weights")))?
            .experts
            .experts;
        let mut slots = Vec::with_capacity(experts_w.len());
        for (e, expert) in experts_w.iter().enumerate() {
            let (offsets, _) = expert.record_layout();
            let mut three = Vec::with_capacity(3);
            for (p, (proj, k)) in [("w1", cfg.hidden), ("w2", inter), ("w3", cfg.hidden)].into_iter().enumerate() {
                let prefix = format!("{ffn}experts.{e}.{proj}");
                let g = self
                    .ck
                    .groups
                    .get(&prefix)
                    .ok_or_else(|| err(format!("deepseek_v41: `{prefix}` is not an EXL3 group of the pack")))?;
                let n = if proj == "w2" { cfg.hidden } else { inter };
                if (g.k, g.n) != (k, n) {
                    return Err(err(format!("{prefix}: the group is {}x{}, the config says {k}x{n}", g.k, g.n)));
                }
                three.push(Exl3Slot::of(g, offsets[p])?);
            }
            slots.push(three.try_into().map_err(|_| err("three projections".into()))?);
        }
        let mut resident = Vec::with_capacity(cfg.moe.routed_experts);
        for _ in 0..cfg.moe.routed_experts {
            // filled by `bind_experts` once the slab exists: a two-phase boot sizes the slab from what is left after everything
            // else (this whole layer included) has been built
            resident.push(None);
        }
        Ok(LayerDevice {
            index,
            attn_norm: self.gamma(&format!("{p}attn_norm.weight"))?,
            ffn_norm: self.gamma(&format!("{p}ffn_norm.weight"))?,
            mla,
            router,
            shared,
            experts: ExpertsDevice { layer: index, resident, streamed: None, slots, fast_desc: None },
            hc_attn: self.site(&format!("{p}hc_attn_"))?,
            hc_ffn: self.site(&format!("{p}hc_ffn_"))?,
        })
    }
}

/// Bind every tensor of `w` that is not a routed expert: views of `regions` where the kernel takes a buffer, an upload where its
/// constructor takes the pack's bytes, and the two engram tables as the mappings `engram` already opened.
///
/// `max_seq` sizes the RoPE tables and the compressed planes (the admitted context). The routed experts are **not** bound here: they
/// are the last thing a boot allocates, and [`bind_experts`] fills them in once the slab that holds them has been sized from what the
/// rest of the boot left on the box.
#[allow(clippy::too_many_arguments)]
pub fn bind(
    cfg: &DeepseekV41Config,
    w: &DeepseekV41Weights,
    ck: &Checkpoint,
    layout: &Layout,
    regions: &Regions,
    max_seq: usize,
    engram: Vec<EngramState>,
    shard: Option<&super::tp::ShardTable>,
) -> Result<Dsv41Device, ModelError> {
    if cfg.hc.mult != gb_cuda::dsv4_moe::STREAMS {
        return Err(ModelError::Config(format!(
            "deepseek_v41: the hyper-connection kernels are {} streams wide, the pack's hc_mult is {}",
            gb_cuda::dsv4_moe::STREAMS,
            cfg.hc.mult
        )));
    }
    let dcfg = dsv4_config(cfg)?;
    // **one table a distinct rotation, not one a layer.** 43 layers of this pack rotate with two specs; building the pair a layer cost
    // 0.67 GiB at a 32K context, which admission did not charge for.
    let rope = SharedRope::new(&dcfg, max_seq).map_err(|e| err(format!("deepseek_v41: the RoPE tables: {e}")))?;
    let b = Binder { cfg, w, ck, layout, regions, shard, preloaded: std::cell::RefCell::new(std::collections::BTreeMap::new()) };
    let embed = b.view("embed.weight")?;
    let head = match &w.head {
        super::weights::LmHead::Bf16(_) => HeadDevice::Bf16(b.view("head.weight")?),
        super::weights::LmHead::Exl3(_) => HeadDevice::Exl3(b.exl3("head", cfg.hidden, cfg.vocab)?),
    };
    let final_norm = b.gamma("norm.weight")?;
    let mut layers = Vec::with_capacity(w.layers.len());
    for chunk in w.layers.chunks(8) {
        let mut names = Vec::with_capacity(chunk.len() * 36);
        for l in chunk {
            let p = format!("layers.{}.", l.index);
            for (k, _) in ck.plain.range(p.clone()..) {
                if !k.starts_with(&p) {
                    break;
                }
                if layout.bindings.get(k).is_some_and(|bi| matches!(bi, super::load::Binding::Upload { .. })) {
                    names.push(k.as_str());
                }
            }
        }
        if names.len() >= 8 {
            let n_workers = 8usize.min(names.len());
            let step = names.len().div_ceil(n_workers);
            let mut results = Vec::with_capacity(n_workers);
            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(n_workers);
                for sub in names.chunks(step) {
                    handles.push(s.spawn(move || {
                        sub.iter()
                            .map(|&name| {
                                let e = ck
                                    .plain
                                    .get(name)
                                    .ok_or_else(|| err(format!("deepseek_v41: `{name}` is not in the pack")))?;
                                let bytes = ck.read_tensor(e).map_err(|er| err(format!("{name}: {er}")))?;
                                Ok::<_, ModelError>((name.to_string(), bytes))
                            })
                            .collect::<Vec<_>>()
                    }));
                }
                for h in handles {
                    results.push(h.join().unwrap_or_default());
                }
            });
            let mut map = b.preloaded.borrow_mut();
            for batch in results {
                for item in batch {
                    let (k, v) = item?;
                    map.insert(k, v);
                }
            }
        }
        for l in chunk {
            layers.push(b.layer(l.index, &dcfg, &rope)?);
        }
        b.preloaded.borrow_mut().clear();
        super::engine::fadvise_pack_shards(&ck.root);
    }
    // the NextN layers: loaded, accounted for, not yet bound into a kernel type
    let mut mtp = Vec::with_capacity(w.mtp.len());
    for m in &w.mtp {
        let mut names: Vec<&Tensor> = vec![&m.attn_norm, &m.ffn_norm];
        names.extend(m.mla.tensors());
        names.extend([&m.router.weight, &m.router.bias, &m.router.bias_vl]);
        for g in [&m.shared.w1, &m.shared.w2, &m.shared.w3] {
            names.extend([&g.weight, &g.scale]);
        }
        for (ei, e) in m.experts.iter().enumerate() {
            // a rank of a group binds ONLY its own half of a NextN stage's experts (the peer's are absent from the
            // plan — `Piece::Skip` — never read and never allocated; binding them would ask the regions for views
            // the plan does not have, which is the first real boot's `not in the load plan`)
            if b.shard.is_some_and(|sh| !sh.owns_nextn_expert(ei)) {
                continue;
            }
            match e {
                MtpExpert::Int4(e) => {
                    for g in e.projections() {
                        names.extend([&g.weight, &g.scale]);
                    }
                }
                MtpExpert::Exl3(e) => {
                    for g in e.projections() {
                        names.extend([&g.trellis, &g.mul1, &g.suh, &g.svh]);
                    }
                }
            }
        }
        names.extend(m.hc_attn.tensors());
        names.extend(m.hc_ffn.tensors());
        if let Some(g) = &m.main_proj {
            names.extend([&g.weight, &g.scale]);
        }
        for t in [&m.main_norm, &m.markov_embed, &m.markov_head, &m.confidence, &m.norm].into_iter().flatten() {
            names.push(t);
        }
        let mut tensors = Vec::with_capacity(names.len());
        for t in names {
            tensors.push((t.name.clone(), b.view(&t.name)?));
        }
        // and, for the re-upload, the experts as the kernel type the drafter's grouped GEMM takes: views of the parts above.
        // A rank of a group binds none (its NextN stages are refused under tensor parallelism — their experts are split —
        // so `DsparkGpu` is never built and `MtpDevice::experts` staying `None` is what keeps it that way)
        let experts = if b.shard.is_none_or(|sh| sh.replicated_nextn) && m.experts.iter().any(|e| e.as_exl3().is_some()) {
            let mut out = Vec::with_capacity(m.experts.len());
            for (e, _) in m.experts.iter().enumerate() {
                let p = format!("mtp.{}.ffn.experts.{e}.", m.index);
                out.push(vec![
                    b.exl3(&format!("{p}w1"), cfg.hidden, cfg.moe.intermediate)?,
                    b.exl3(&format!("{p}w2"), cfg.moe.intermediate, cfg.hidden)?,
                    b.exl3(&format!("{p}w3"), cfg.hidden, cfg.moe.intermediate)?,
                ]);
            }
            Some(out)
        } else {
            None
        };
        mtp.push(MtpDevice { index: m.index, tensors, experts });
    }
    let mut engram_dev = Vec::with_capacity(engram.len());
    for (state, t) in engram.into_iter().zip(w.engram_tables()) {
        let l = t.layer;
        let p = format!("layers.{l}.engram.");
        engram_dev.push(EngramDevice {
            layer: l,
            state,
            wkv: b.fp8(&format!("{p}wkv"))?,
            q_weight: b.view(&format!("{p}q_weight"))?,
            k_weight: b.view(&format!("{p}k_weight"))?,
        });
    }
    let engram = engram_dev;
    // the distinct tables, by pointer: a shared table is one allocation however many layers point at it
    let mut seen: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let mut rope_bytes = 0u64;
    for l in &layers {
        for t in [&l.mla.rope, &l.mla.crope] {
            if seen.insert(Arc::as_ptr(t) as usize) {
                rope_bytes += t.bytes();
            }
        }
    }
    debug_assert_eq!(rope_bytes, rope.bytes(), "every shared table is used by at least one layer");
    let mtp_gpu = if gb_cuda::device_available()
        && cfg.nextn > 0
        && mtp.len() == cfg.nextn
        && mtp.iter().all(|m| m.experts.is_some())
        && matches!(w.head, super::weights::LmHead::Exl3(_))
    {
        use gb_cuda::dsv4_mtp::MtpConfig;
        use gb_cuda::dsv4_mtp_gpu::{DsparkGpu, HcPack, MtpDense, MtpStage, RoutedExperts};
        use gb_cuda::exl3_linear::Dtype;
        let mcfg = MtpConfig {
            dim: cfg.hidden,
            n_heads: cfg.heads,
            head_dim: cfg.head_dim,
            rope_dim: cfg.qk_rope_head_dim,
            q_lora: cfg.q_lora_rank,
            o_lora: cfg.o_lora_rank,
            o_groups: cfg.o_groups,
            window: cfg.sliding_window,
            block_size: cfg.dspark.block_size,
            noise_token_id: cfg.dspark.noise_token_id as i32,
            n_stages: cfg.nextn,
            target_layer_ids: cfg.dspark.target_layer_ids.clone(),
            vocab: cfg.vocab,
            markov_rank: cfg.dspark.markov_rank,
            n_routed_experts: cfg.dspark.n_routed_experts,
            topk: cfg.dspark.num_experts_per_tok,
            inter_dim: cfg.moe.intermediate,
            swiglu_limit: cfg.swiglu_limit as f32,
            route_scale: cfg.moe.routed_scaling_factor as f32,
            gate_temp: 1.0,
            hc: gb_cuda::dsv4_moe::HcCfg::V41,
            eps: cfg.rms_eps as f32,
            rope_theta: cfg.rope.theta,
            temperature: 0.0,
        };
        let dense = |prefix: &str, n: usize, k: usize| -> Result<MtpDense, ModelError> {
            MtpDense::from_parts(n, k, b.view(&format!("{prefix}.weight"))?, b.view(&format!("{prefix}.scale"))?)
                .map_err(|e| err(format!("mtp dense {prefix}: {e}")))
        };
        let shared_mtp_rope = rope.get(&dcfg, 0).clone_views_padded();
        let mut stages = Vec::with_capacity(cfg.nextn);
        for idx in 0..cfg.nextn {
            let p = format!("mtp.{idx}.");
            let attn = format!("{p}attn.");
            let ffn = format!("{p}ffn.");
            let last = idx + 1 == cfg.nextn;
            let mut exps = Vec::with_capacity(mcfg.n_routed_experts);
            for e in 0..mcfg.n_routed_experts {
                let ep = format!("{ffn}experts.{e}.");
                exps.push([
                    b.exl3(&format!("{ep}w1"), mcfg.dim, mcfg.inter_dim)?,
                    b.exl3(&format!("{ep}w2"), mcfg.inter_dim, mcfg.dim)?,
                    b.exl3(&format!("{ep}w3"), mcfg.dim, mcfg.inter_dim)?,
                ]);
            }
            let hc_site = |tag: &str| -> Result<HcPack, ModelError> {
                HcPack::from_parts(
                    mcfg.dim,
                    b.view(&format!("{p}{tag}fn"))?,
                    b.view(&format!("{p}{tag}base"))?,
                    b.view(&format!("{p}{tag}scale"))?,
                )
                .map_err(|e| err(format!("mtp hc {p}{tag}: {e}")))
            };
            let stg = MtpStage {
                stage: idx,
                wq_a: dense(&format!("{attn}wq_a"), mcfg.q_lora, mcfg.dim)?,
                wq_b: dense(&format!("{attn}wq_b"), mcfg.n_heads * mcfg.head_dim, mcfg.q_lora)?,
                wkv: dense(&format!("{attn}wkv"), mcfg.head_dim, mcfg.dim)?,
                wo_a: dense(&format!("{attn}wo_a"), mcfg.o_groups * mcfg.o_lora, mcfg.o_in_per_group())?,
                wo_b: dense(&format!("{attn}wo_b"), mcfg.dim, mcfg.o_groups * mcfg.o_lora)?,
                q_norm: b.gamma(&format!("{attn}q_norm.weight"))?,
                kv_norm: b.gamma(&format!("{attn}kv_norm.weight"))?,
                sink: b.view(&format!("{attn}attn_sink"))?,
                rope: shared_mtp_rope.clone_views(),
                window: DevBuf::zeroed(mcfg.window * mcfg.head_dim * 2).map_err(|e| err(format!("mtp window {idx}: {e}")))?,
                gate: b.view(&format!("{ffn}gate.weight"))?,
                gate_bias: b.view(&format!("{ffn}gate.bias"))?,
                routed: RoutedExperts::Exl3(exps),
                shared_w1: dense(&format!("{ffn}shared_experts.w1"), mcfg.inter_dim, mcfg.dim)?,
                shared_w3: dense(&format!("{ffn}shared_experts.w3"), mcfg.inter_dim, mcfg.dim)?,
                shared_w2: dense(&format!("{ffn}shared_experts.w2"), mcfg.dim, mcfg.inter_dim)?,
                hc_attn: hc_site("hc_attn_")?,
                hc_ffn: hc_site("hc_ffn_")?,
                attn_norm: b.gamma(&format!("{p}attn_norm.weight"))?,
                ffn_norm: b.gamma(&format!("{p}ffn_norm.weight"))?,
                main_proj: if idx == 0 { Some(dense(&format!("{p}main_proj"), mcfg.dim, mcfg.tap_width())?) } else { None },
                main_norm: if idx == 0 { Some(b.gamma(&format!("{p}main_norm.weight"))?) } else { None },
                norm: if last { Some(b.gamma(&format!("{p}norm.weight"))?) } else { None },
                markov_embed: if last { Some(b.view(&format!("{p}markov_head.embed.weight"))?) } else { None },
                markov_head: if last { Some(b.view(&format!("{p}markov_head.head.weight"))?) } else { None },
                conf_proj: if last { Some(b.view(&format!("{p}confidence_head.proj.weight"))?) } else { None },
                conf_bias: if last { Some(DevBuf::from_slice(&[0f32]).map_err(|e| err(e.to_string()))?) } else { None },
                lm_head: None,
                lm_head_exl3: if last { Some(b.exl3("head", cfg.hidden, cfg.vocab)?) } else { None },
            };
            stages.push(stg);
        }
        let gpu = DsparkGpu::from_parts(mcfg, stages, b.view("embed.weight")?, Dtype::BF16, Dtype::BF16)
            .map_err(|e| err(format!("mtp DsparkGpu: {e}")))?;
        Some(gpu)
    } else {
        None
    };
    Ok(Dsv41Device { embed, final_norm, head, layers, mtp, mtp_gpu, engram, uploaded_bytes: layout.upload_bytes, rope_bytes })
}

/// **Phase two of the boot's expert binding**: for every layer, the experts the (measured) slab holds become
/// [`Exl3Weights::from_parts`] views of that layer's region, and the rest are reached through `cache` — built over the layout's own
/// streaming-window region, so nothing is allocated here.
///
/// Which experts are resident is the layout's decision (`layout.resident_experts`, layer-major, the same walk
/// [`super::plan::LoadPlan::build`] takes), so this function invents nothing: it reads `Binding::View` for a slab expert and
/// `Binding::Streamed` for a tail one, and a name that is neither is a layout that disagrees with itself.
pub fn bind_experts(
    cfg: &DeepseekV41Config,
    w: &DeepseekV41Weights,
    ck: &Checkpoint,
    layout: &Layout,
    regions: &Regions,
    cache: Option<Arc<ExpertCache>>,
    dev: &mut Dsv41Device,
) -> Result<(), ModelError> {
    let b = Binder {
        cfg,
        w,
        ck,
        layout,
        regions,
        shard: None,
        preloaded: std::cell::RefCell::new(std::collections::BTreeMap::new()),
    };
    let inter = cfg.moe.intermediate;
    let mut bound = 0usize;
    for layer in dev.layers.iter_mut() {
        let index = layer.index;
        let x = format!("layers.{index}.ffn.experts.");
        let mut resident = Vec::with_capacity(cfg.moe.routed_experts);
        let mut layer_has_streamed = false;
        for e in 0..cfg.moe.routed_experts {
            let p = format!("{x}{e}.");
            let bnd = layout.bindings.get(&format!("{p}w1.trellis"));
            let in_slab = bnd.is_some_and(|b| b.is_region());
            if matches!(bnd, Some(super::load::Binding::Streamed { .. })) {
                layer_has_streamed = true;
            }
            resident.push(if in_slab {
                bound += 1;
                Some(vec![
                    b.exl3(&format!("{p}w1"), cfg.hidden, inter)?,
                    b.exl3(&format!("{p}w2"), inter, cfg.hidden)?,
                    b.exl3(&format!("{p}w3"), cfg.hidden, inter)?,
                ])
            } else {
                None
            });
        }
        let layer_all_resident = !layer_has_streamed && resident.iter().any(|r| r.is_some());
        let fast_moe = std::env::var_os("REXL3_DSV41_FAST_MOE").is_some_and(|v| v == "1");
        let fast_desc = if fast_moe && layer_all_resident {
            let opts: Vec<Option<&[Exl3Weights]>> = resident.iter().map(|r| r.as_deref()).collect();
            gb_cuda::dsv4_moe::build_expert_descs_sparse(&opts, cfg.hidden, inter)
                .map_err(|e| ModelError::Config(format!("deepseek_v41: layer {index} fast_desc: {e}")))?
        } else {
            None
        };
        layer.experts.resident = resident;
        layer.experts.streamed = if layer_all_resident { None } else { cache.clone() };
        layer.experts.fast_desc = fast_desc;
    }
    // the slab is whole experts and the layout counted them: a mismatch is a residency decision made twice and differently
    if bound != layout.resident_experts {
        return Err(ModelError::Config(format!(
            "deepseek_v41: the layout holds {} experts resident and {bound} were bound",
            layout.resident_experts
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = include_str!("../../../../../testdata/deepseek_v41/config.json");

    fn real() -> DeepseekV41Config {
        DeepseekV41Config::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    #[test]
    fn the_kernel_geometry_is_the_configs() {
        let c = real();
        let d = dsv4_config(&c).unwrap();
        assert_eq!((d.dim, d.n_heads, d.head_dim, d.rope_dim), (5_120, 64, 512, 64));
        assert_eq!((d.q_lora, d.o_lora, d.o_groups), (1_280, 1_024, 8));
        assert_eq!((d.window, d.n_layers), (128, 43), "the 40 trunk layers and the 3 NextN ones");
        // the ratios are the config's, in order, and 0 where a layer caches no compressed entry
        assert_eq!(d.compress_ratios, c.compress.ratios.iter().map(|&r| r as u8).collect::<Vec<u8>>());
        assert_eq!(d.compress_ratios.len(), 43);
        assert_eq!(d.kv_source_layers, vec![2, 8, 14, 20]);
        assert_eq!(d.index_source_layers, vec![2, 8, 14, 20, 24, 28, 32, 36]);
        // the kernel crate's own constant instance agrees field for field
        let v41 = Dsv4Config::v41();
        assert_eq!((d.dim, d.n_heads, d.head_dim), (v41.dim, v41.n_heads, v41.head_dim));
        assert_eq!((d.window, d.index_topk), (v41.window, v41.index_topk));
        assert_eq!((d.eps, v41.eps), (v41.eps, v41.eps));
    }

    #[test]
    fn a_config_with_more_than_one_kv_head_is_refused_by_name() {
        let mut c = real();
        c.kv_heads = 2;
        let e = dsv4_config(&c).unwrap_err().to_string();
        assert!(e.contains("one shared KV head"), "{e}");
    }

    #[test]
    fn a_config_whose_heads_do_not_divide_into_groups_is_refused() {
        let mut c = real();
        c.o_groups = 7;
        let e = dsv4_config(&c).unwrap_err().to_string();
        assert!(e.contains("output groups"), "{e}");
    }

    #[test]
    fn a_rank_binds_only_its_own_heads_sink_values() {
        // the attention sink is one value per query head and the kernel indexes it by LOCAL head: rank 1 of a
        // two-rank group computes heads [32,64) and must bind sinks [32,64). The first live TP boot bound the
        // whole [64] tensor on every rank — heads 32..63 read the sinks of heads 0..31, every attention slightly
        // off, identically on both ranks (the ~2%-off-single-rank band the exactness oracle caught at block 0).
        use super::super::tp::{ShardTable, TpTopology};
        let whole: Vec<f32> = (0..64).map(|h| h as f32).collect();
        let cfg = real();
        let rank1 = ShardTable::for_config(TpTopology { rank: 1, ranks: 2 }, &cfg).unwrap();
        assert_eq!(rank1.q_head_range(), 32..64, "the fixture: rank 1 owns heads 32..63");
        let sink = sink_slice(&whole, Some(&rank1));
        assert_ne!(sink, &whole[..32], "rank 1 must never read heads 0..31's sinks");
        assert_eq!(sink, &whole[32..], "rank 1's heads 32..63 must read sinks 32..63");
        // the two ranks' sinks concatenate back to exactly the whole tensor
        let rank0 = ShardTable::for_config(TpTopology { rank: 0, ranks: 2 }, &cfg).unwrap();
        let mut pair = sink_slice(&whole, Some(&rank0)).to_vec();
        pair.extend_from_slice(sink_slice(&whole, Some(&rank1)));
        assert_eq!(pair, whole, "the ranks' sinks partition the whole");
        // the single rank — an unsharded bind or a single-table shard — keeps the whole tensor
        assert_eq!(sink_slice(&whole, None), &whole[..]);
        assert_eq!(sink_slice(&whole, Some(&ShardTable::single())), &whole[..]);
    }
}
