//! The Nemotron 3.5 Super VL weights as the layer loop reads them: one device layer per trunk block, plus the MTP head and the three
//! global tensors. [`super::bridge`] fills these from the loader's upload, and [`super::model`] runs them.
//!
//! The types are frozen at wave 0 of the engine plan. Fields are public because the three arms in `arm_*.rs` are separate files.
//! Every norm and scale vector stays bf16 as in the pack, except where the field says f32: the loader's bf16 `A_log`, `D` and `dt_bias`
//! are widened once at load.

use gb_cuda::exl3_linear::Exl3Weights;
use gb_cuda::nemotron_moe_prefill::PrefillMoe;
use gb_cuda::DevBuf;

/// The three kinds of trunk block, as `layers_block_type` spells them (`M`, `*`, `E`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerKind {
    Mamba,
    Attn,
    Moe,
}

impl LayerKind {
    /// The kind a pattern character names.
    pub fn from_pattern_char(c: char) -> Option<Self> {
        match c {
            'M' => Some(LayerKind::Mamba),
            '*' => Some(LayerKind::Attn),
            'E' => Some(LayerKind::Moe),
            _ => None,
        }
    }

    fn slot(self) -> usize {
        match self {
            LayerKind::Mamba => 0,
            LayerKind::Attn => 1,
            LayerKind::Moe => 2,
        }
    }
}

/// Which loaded layer is which kind, and each layer's ordinal among the layers of its kind. Plain host data: the Mamba state pools, the
/// window records and the KV stripes are all indexed by these ordinals, so this is the one place that maps a trunk index to them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerMap {
    first: usize,
    kinds: Vec<LayerKind>,
    ordinals: Vec<usize>,
    counts: [usize; 3],
}

impl LayerMap {
    /// The map of layers `first..first + kinds.len()` of the trunk, whose kinds are `kinds`.
    pub fn new(first: usize, kinds: &[LayerKind]) -> Self {
        let mut counts = [0usize; 3];
        let ordinals = kinds
            .iter()
            .map(|&k| {
                let o = counts[k.slot()];
                counts[k.slot()] += 1;
                o
            })
            .collect();
        Self { first, kinds: kinds.to_vec(), ordinals, counts }
    }

    /// The map of trunk layers `first..first + count` of `pattern` (`M`, `*`, `E` per layer).
    pub fn from_pattern(pattern: &str, first: usize, count: usize) -> Result<Self, String> {
        let chars: Vec<char> = pattern.chars().collect();
        if first + count > chars.len() {
            return Err(format!("layers {first}..{} are past the {} of the trunk", first + count, chars.len()));
        }
        let kinds = chars[first..first + count]
            .iter()
            .enumerate()
            .map(|(i, &c)| LayerKind::from_pattern_char(c).ok_or_else(|| format!("layer {} is `{c}`, not M, * or E", first + i)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(first, &kinds))
    }

    /// The trunk index of the first loaded layer.
    pub fn first(&self) -> usize {
        self.first
    }

    /// Loaded layers.
    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    /// Whether trunk layer `li` is loaded.
    pub fn contains(&self, li: usize) -> bool {
        li >= self.first && li - self.first < self.kinds.len()
    }

    /// The kind of trunk layer `li`, if it is loaded.
    pub fn kind(&self, li: usize) -> Option<LayerKind> {
        li.checked_sub(self.first).and_then(|i| self.kinds.get(i)).copied()
    }

    /// The ordinal of trunk layer `li` among the loaded layers of its kind.
    pub fn ordinal(&self, li: usize) -> Option<usize> {
        li.checked_sub(self.first).and_then(|i| self.ordinals.get(i)).copied()
    }

    /// The ordinal of `li` when it is a layer of `kind`.
    pub fn ordinal_of(&self, li: usize, kind: LayerKind) -> Option<usize> {
        if self.kind(li) == Some(kind) {
            self.ordinal(li)
        } else {
            None
        }
    }

    /// Loaded layers of `kind`.
    pub fn count(&self, kind: LayerKind) -> usize {
        self.counts[kind.slot()]
    }

    /// The trunk indices of the loaded layers of `kind`, in order.
    pub fn indices(&self, kind: LayerKind) -> Vec<usize> {
        self.kinds.iter().enumerate().filter(|(_, &k)| k == kind).map(|(i, _)| self.first + i).collect()
    }
}

/// One Mamba2 layer. `in_proj` is `[18560, 4096]` (z 8192 | xBC 10240 | dt 128) and `out_proj` `[4096, 8192]`.
pub struct MambaW {
    pub in_proj: Exl3Weights,
    pub out_proj: Exl3Weights,
    /// bf16 `[10240][4]` (the pack's `conv1d.weight[:, 0, :]`).
    pub conv_w: DevBuf,
    /// bf16 `[10240]`.
    pub conv_b: DevBuf,
    /// f32 `[128]`.
    pub a_log: DevBuf,
    /// f32 `[128]`.
    pub dt_bias: DevBuf,
    /// f32 `[128]`.
    pub d: DevBuf,
    /// The gated norm's weight, f32 `[8192]` (`nemotron_ssm::gated_norm` reads f32; widened from the pack's bf16 at load).
    pub gate_norm: DevBuf,
    /// Contract P2 at two TP ranks only (lane S): this rank's head-group in_proj (`tp_shard::hg_runs`: z, x of its heads, B, C of its
    /// groups, dt of every head; 9,344 columns), beside the column slice `in_proj` that decode and windows use.
    pub in_proj_hg: Option<Exl3Weights>,
}

/// One attention layer (32 query heads, 2 KV heads, head dim 128, no RoPE, no bias).
pub struct AttnW {
    pub q: Exl3Weights,
    pub k: Exl3Weights,
    pub v: Exl3Weights,
    pub o: Exl3Weights,
}

pub enum Mixer {
    Mamba(MambaW),
    Attn(AttnW),
    /// The latent MoE block: the decode path's tables and the large-row prefill path's (`PrefillMoe` dereferences to the
    /// [`gb_cuda::nemotron_moe::NemotronMoe`]). Its scratch is not inside: the model owns one [`gb_cuda::nemotron_moe::MoeScratch`]
    /// (windows and slices of up to 64 rows) and one [`gb_cuda::nemotron_moe_prefill::PrefillScratch`] (wider slices) that every layer
    /// shares.
    Moe(Box<PrefillMoe>),
}

impl Mixer {
    pub fn kind(&self) -> LayerKind {
        match self {
            Mixer::Mamba(_) => LayerKind::Mamba,
            Mixer::Attn(_) => LayerKind::Attn,
            Mixer::Moe(_) => LayerKind::Moe,
        }
    }
}

/// A trunk block: `x = x + mixer(norm(x))`.
pub struct Layer {
    /// The block's pre-norm weight, bf16 `[4096]`.
    pub norm: DevBuf,
    pub mixer: Mixer,
}

/// A dense projection that may be stored either way: the MTP head's `eh_proj` and the attention's q, k, v and o, and the trunk's
/// `lm_head`. bf16 `[n, k]` in the pack's HF layout, or an EXL3 linear (a 4-bit head's `eh_proj`; the MixedK pack's head attention and
/// its 8-bit `lm_head` too), run as the trunk's dense linears are.
pub enum HeadLinear {
    Bf16(DevBuf),
    Exl3(Exl3Weights),
}

impl HeadLinear {
    /// Device bytes of the weight: the bf16 tensor, or the trellis and the fp16 scales.
    pub fn bytes(&self) -> u64 {
        match self {
            HeadLinear::Bf16(b) => b.bytes() as u64,
            HeadLinear::Exl3(l) => l.weight_bytes() as u64,
        }
    }

    /// The EXL3 linear, for a 4-bit head.
    pub fn exl3(&self) -> Option<&Exl3Weights> {
        match self {
            HeadLinear::Exl3(l) => Some(l),
            HeadLinear::Bf16(_) => None,
        }
    }

    /// The bf16 weight `[n, k]`.
    pub fn bf16(&self) -> Option<&DevBuf> {
        match self {
            HeadLinear::Bf16(b) => Some(b),
            HeadLinear::Exl3(_) => None,
        }
    }

    /// Output rows (`n`) of a projection whose input is `k` wide: the bf16 weight's rows, or the EXL3 linear's columns (`Exl3Weights::n`,
    /// a TP rank's local width).
    pub fn rows(&self, k: usize) -> usize {
        match self {
            HeadLinear::Bf16(b) => b.bytes() / (k * 2),
            HeadLinear::Exl3(l) => l.n,
        }
    }
}

/// The MTP head `*E`: bf16 on the published pack (about 5.9 GB, of which 5.6 GB is the experts); on a 4-bit head pack `eh_proj` and the
/// MoE block's projections are EXL3 where the pack stores them so, and on the MixedK pack the attention too (`bridge::take_mtp`).
pub struct MtpW {
    /// This rank holds only its half of the head's routed experts (the others are `Proj::Absent`, tensor parallelism): the head's MoE
    /// block then runs `forward_in_tp` and allreduces its `y`. False for a single rank and for a group that replicates the head.
    pub experts_split: bool,
    /// `[4096, 8192]` (bf16, or EXL3 K = 8192, N = 4096): the fusion of the embedding-first concat.
    pub eh_proj: HeadLinear,
    pub enorm: DevBuf,
    pub hnorm: DevBuf,
    /// The attention block's pre-norm.
    pub attn_norm: DevBuf,
    /// `[4096, 4096]`, `[256, 4096]`, `[256, 4096]` and `[4096, 4096]` (bf16 `[n, k]`, or EXL3 K = 4096, N = 4096 / 256 / 256 / 4096).
    pub q: HeadLinear,
    pub k: HeadLinear,
    pub v: HeadLinear,
    pub o: HeadLinear,
    /// The MoE block's pre-norm.
    pub moe_norm: DevBuf,
    /// The head's block: its experts and dense projections, bf16 or EXL3 per projection. Its wide absorbs (a prefill slice's rows) take
    /// the large-row prefill path.
    pub moe: Box<PrefillMoe>,
    pub final_norm: DevBuf,
}

impl MtpW {
    /// Whether any of the head's projections is EXL3 (a 4-bit head pack): `eh_proj`, the attention, or a projection of the MoE block.
    pub fn has_exl3(&self) -> bool {
        self.dense().iter().any(|l| l.exl3().is_some()) || self.moe.has_exl3()
    }

    /// The head's dense projections of block 0, in forward order: `eh_proj`, q, k, v, o.
    pub fn dense(&self) -> [&HeadLinear; 5] {
        [&self.eh_proj, &self.q, &self.k, &self.v, &self.o]
    }

    /// Whether the head's attention (q, k, v, o) is EXL3 (the MixedK pack's head).
    pub fn attn_exl3(&self) -> bool {
        [&self.q, &self.k, &self.v, &self.o].iter().any(|l| l.exl3().is_some())
    }
}

/// The device weights of trunk layers `first..first + layers.len()` and the globals.
pub struct NemotronWeights {
    /// The index of `layers[0]` in the trunk (`0` for the whole model).
    pub first: usize,
    /// bf16 `[131072, 4096]`.
    pub embed: DevBuf,
    /// bf16 `[4096]`.
    pub norm_f: DevBuf,
    /// bf16 `[131072, 4096]`, or EXL3 K = 4096, N = 131072 (the MixedK pack's 8-bit head). On a rank of a TP group (`tp_shard`) the
    /// rank's rows only (`[65536, 4096]` bf16, or an EXL3 slice of N = 65536 cut at 128-column blocks); the dense EXL3 linears of the layers
    /// likewise report their local width in `Exl3Weights::n`.
    pub lm_head: HeadLinear,
    pub layers: Vec<Layer>,
    pub mtp: Option<MtpW>,
}

impl NemotronWeights {
    /// Whether the trunk's `lm_head` is EXL3 (the MixedK pack's 8-bit head): the cost tables' head form (`spec::HeadForm`) reads it.
    pub fn lm_head_exl3(&self) -> bool {
        self.lm_head.exl3().is_some()
    }

    /// The layer at trunk index `li`, if it is on the device.
    pub fn layer(&self, li: usize) -> Option<&Layer> {
        li.checked_sub(self.first).and_then(|i| self.layers.get(i))
    }

    /// The kind and ordinal of every layer on the device.
    pub fn layer_map(&self) -> LayerMap {
        let kinds: Vec<LayerKind> = self.layers.iter().map(|l| l.mixer.kind()).collect();
        LayerMap::new(self.first, &kinds)
    }

    /// Every EXL3 linear the layer loop runs through `Exl3Weights::forward`: the Mamba projections and the attention q, k, v and o (the
    /// MoE block's projections run inside [`gb_cuda::nemotron_moe::NemotronMoe`], on its own workspace).
    pub fn linears(&self) -> Vec<&Exl3Weights> {
        let mut v = Vec::new();
        for l in &self.layers {
            match &l.mixer {
                Mixer::Mamba(m) => v.extend([&m.in_proj, &m.out_proj]),
                Mixer::Attn(a) => v.extend([&a.q, &a.k, &a.v, &a.o]),
                Mixer::Moe(_) => {}
            }
        }
        v
    }

    /// The decode workspace `Exl3Weights::forward` needs for `rows` rows on the largest of [`NemotronWeights::linears`]. One buffer of this
    /// size serves every layer: the layers run one after another on one stream and the workspace needs no initialisation.
    pub fn max_workspace_bytes(&self, rows: usize) -> usize {
        // a 4-bit head's `eh_proj` and an EXL3 head attention run through the same workspace (`mtp::head_block`), and so does an EXL3
        // `lm_head` (`tp_arm::lm_head_logits`: `linear_rows`, at most `rows` a call)
        let head =
            self.mtp.as_ref().map(|h| h.dense().into_iter().filter_map(HeadLinear::exl3).collect::<Vec<_>>()).unwrap_or_default();
        let lm_head = self.lm_head.exl3();
        self.linears().into_iter().chain(head).chain(lm_head).map(|w| w.workspace_bytes(rows)).max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PACK: &str = include_str!("../../../../../testdata/nemotron_omni/config.json");

    fn pattern() -> String {
        let root: serde_json::Value = serde_json::from_str(PACK).unwrap();
        super::super::config::OmniConfig::from_json(&root).unwrap().pattern
    }

    #[test]
    fn the_trunk_has_40_mamba_8_attention_and_40_moe_layers_at_the_measured_places() {
        let p = pattern();
        let m = LayerMap::from_pattern(&p, 0, p.len()).unwrap();
        assert_eq!(m.len(), 88);
        assert_eq!((m.count(LayerKind::Mamba), m.count(LayerKind::Attn), m.count(LayerKind::Moe)), (40, 8, 40));
        assert_eq!(m.indices(LayerKind::Attn), vec![7, 16, 25, 36, 47, 58, 69, 78]);
        assert_eq!(m.kind(0), Some(LayerKind::Mamba));
        assert_eq!(m.kind(1), Some(LayerKind::Moe));
        assert_eq!(m.kind(88), None);
    }

    #[test]
    fn ordinals_count_within_each_kind_and_follow_the_first_loaded_layer() {
        let p = pattern();
        let m = LayerMap::from_pattern(&p, 0, 16).unwrap();
        // layers 0..16 of the pattern are `MEMEMEM*EMEMEMEM`: the one attention layer is at 7
        assert_eq!(m.ordinal_of(7, LayerKind::Attn), Some(0));
        assert_eq!(m.count(LayerKind::Attn), 1);
        assert_eq!(m.ordinal_of(6, LayerKind::Mamba), Some(3));
        assert_eq!(m.ordinal_of(8, LayerKind::Moe), Some(3));
        assert_eq!(m.ordinal_of(7, LayerKind::Mamba), None);
        let mamba = m.indices(LayerKind::Mamba);
        for (i, &li) in mamba.iter().enumerate() {
            assert_eq!(m.ordinal(li), Some(i), "mamba layer {li}");
        }
        let moe = m.indices(LayerKind::Moe);
        for (i, &li) in moe.iter().enumerate() {
            assert_eq!(m.ordinal_of(li, LayerKind::Moe), Some(i));
        }
        // a map that starts later numbers from its own first layer
        let later = LayerMap::from_pattern(&p, 5, 6).unwrap();
        assert_eq!(later.first(), 5);
        assert!(!later.contains(4) && later.contains(5) && later.contains(10) && !later.contains(11));
        assert_eq!(later.ordinal(5), Some(0));
        assert_eq!(later.kind(4), None);
    }

    #[test]
    fn a_pattern_past_the_trunk_or_with_an_unknown_block_is_refused() {
        assert!(LayerMap::from_pattern("ME*", 0, 4).is_err());
        assert!(LayerMap::from_pattern("MX*", 0, 3).unwrap_err().contains("layer 1"));
        assert!(LayerMap::from_pattern("ME*", 1, 2).is_ok());
    }
}
