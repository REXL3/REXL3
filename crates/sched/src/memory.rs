//! The memory model a model family hands to the planner and to admission, and the checks that make it fail closed.
//!
//! A model family (dense GQA, GQA plus gated delta net, MLA plus a sparse indexer plus Kimi delta attention, ...) differs in what
//! a token costs in the paged cache, what a sequence costs in recurrent state, what a speculative window costs in per-row
//! history, and how big the activation buffers are. [`MemoryModel`] is the one question every family answers; [`MemoryProfile`]
//! is the validated snapshot of the answers. A profile can only be built through [`MemoryProfile::new`], which refuses (with an
//! error that names the family and the quantity) every shape that would make the arithmetic downstream wrong in the unsafe
//! direction: a zero KV cost per token (admission would hand out unlimited context and the planner would divide by zero), a
//! recurrent family with no recurrent state, a step with no scratch, an arithmetic overflow. Nothing here touches the GPU.

use std::fmt;

/// Why a memory model was refused. The strings name the family and the quantity so a startup log says what to fix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryError {
    /// A quantity that must be positive is zero (a family with `head_dim: 0` read as grouped-query attention gets here).
    Zero(String),
    /// The byte arithmetic does not fit in 64 bits.
    Overflow(String),
    /// The numbers contradict each other (recurrent layers without state, a pool that does not divide the page, ...).
    Inconsistent(String),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryError::Zero(m) => write!(f, "memory model refused, a quantity is zero: {m}"),
            MemoryError::Overflow(m) => write!(f, "memory model refused, byte arithmetic overflows: {m}"),
            MemoryError::Inconsistent(m) => write!(f, "memory model refused, inconsistent: {m}"),
        }
    }
}

impl std::error::Error for MemoryError {}

/// One tensor a layer caches per token: K, V, an MLA latent, an indexer key plane, a pooled plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPlane {
    pub name: &'static str,
    /// Elements stored per group of `group_tokens` tokens.
    pub elems: u64,
    /// Tokens per group: 1 for a per-token plane, the pool size for a pooled plane (one entry per `group_tokens` tokens).
    pub group_tokens: u64,
    pub bytes_per_elem: u64,
}

impl KvPlane {
    /// A plane with `elems` elements for every token.
    pub const fn per_token(name: &'static str, elems: u64, bytes_per_elem: u64) -> Self {
        Self { name, elems, group_tokens: 1, bytes_per_elem }
    }
    /// A plane with `elems` elements for every `group_tokens` consecutive tokens.
    pub const fn pooled(name: &'static str, elems: u64, group_tokens: u64, bytes_per_elem: u64) -> Self {
        Self { name, elems, group_tokens, bytes_per_elem }
    }
}

/// A group of layers that cache the same planes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvLayerClass {
    pub name: &'static str,
    pub layers: u64,
    pub planes: Vec<KvPlane>,
}

impl KvLayerClass {
    /// Grouped-query attention: a K and a V plane of `kv_heads x head_dim` per token. Zero heads or zero width is not
    /// special-cased: [`KvLayout::bytes_per_page`] refuses it.
    pub fn gqa(layers: u64, kv_heads: u64, head_dim: u64, bytes_per_elem: u64) -> Self {
        let elems = kv_heads.saturating_mul(head_dim);
        Self {
            name: "gqa",
            layers,
            planes: vec![KvPlane::per_token("k", elems, bytes_per_elem), KvPlane::per_token("v", elems, bytes_per_elem)],
        }
    }
}

/// What a token costs in the paged cache, as layer classes with their planes. Per-page bytes are exact (integers), so the
/// planner's page arithmetic and admission's per-token arithmetic can never disagree.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KvLayout {
    pub classes: Vec<KvLayerClass>,
}

impl KvLayout {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, class: KvLayerClass) -> Self {
        self.classes.push(class);
        self
    }

    /// Bytes of one page of `page_tokens` tokens across every layer. Errors on an empty layout, a class with no layers, a plane
    /// with a zero extent, a pool that does not divide the page (a pool must not straddle pages), and overflow.
    pub fn bytes_per_page(&self, family: &str, page_tokens: u64) -> Result<u64, MemoryError> {
        if page_tokens == 0 {
            return Err(MemoryError::Zero(format!("{family}: page size in tokens")));
        }
        if self.classes.is_empty() {
            return Err(MemoryError::Zero(format!("{family}: the KV layout has no layer class")));
        }
        let mut total = 0u64;
        for c in &self.classes {
            if c.layers == 0 {
                return Err(MemoryError::Zero(format!("{family}: KV layer class `{}` has no layers", c.name)));
            }
            if c.planes.is_empty() {
                return Err(MemoryError::Zero(format!("{family}: KV layer class `{}` caches no plane", c.name)));
            }
            let mut per_layer = 0u64;
            for p in &c.planes {
                if p.elems == 0 || p.bytes_per_elem == 0 || p.group_tokens == 0 {
                    return Err(MemoryError::Zero(format!(
                        "{family}: KV plane `{}` of class `{}` has {} elements of {} bytes per {} tokens",
                        p.name, c.name, p.elems, p.bytes_per_elem, p.group_tokens
                    )));
                }
                if page_tokens % p.group_tokens != 0 {
                    return Err(MemoryError::Inconsistent(format!(
                        "{family}: KV plane `{}` is pooled per {} tokens, which does not divide the {page_tokens}-token page",
                        p.name, p.group_tokens
                    )));
                }
                let bytes = p
                    .elems
                    .checked_mul(p.bytes_per_elem)
                    .and_then(|b| b.checked_mul(page_tokens / p.group_tokens))
                    .ok_or_else(|| MemoryError::Overflow(format!("{family}: KV plane `{}`", p.name)))?;
                per_layer = per_layer
                    .checked_add(bytes)
                    .ok_or_else(|| MemoryError::Overflow(format!("{family}: KV class `{}`", c.name)))?;
            }
            total = per_layer
                .checked_mul(c.layers)
                .and_then(|b| b.checked_add(total))
                .ok_or_else(|| MemoryError::Overflow(format!("{family}: KV layout")))?;
        }
        Ok(total)
    }

    /// Bytes per token: the page bytes divided by the page size, which must be exact (an inexact per-token cost would make the
    /// planner and admission count different bytes).
    pub fn bytes_per_token(&self, family: &str, page_tokens: u64) -> Result<u64, MemoryError> {
        let page = self.bytes_per_page(family, page_tokens)?;
        if page % page_tokens != 0 {
            return Err(MemoryError::Inconsistent(format!(
                "{family}: a {page_tokens}-token page of {page} bytes is not a whole number of bytes per token"
            )));
        }
        Ok(page / page_tokens)
    }
}

/// The vision tower's memory, as the family states it. All figures are for the largest image the server accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionMemory {
    /// Tower weights as loaded, when the family can state them from its config (`None`: measured at load, not predictable).
    pub weight_bytes: Option<u64>,
    /// Working buffers of one encode at the largest image (transient, but allocated for the life of the process).
    pub encoder_scratch_bytes: u64,
    /// The encoder's output for one image at the largest size (`tokens x LM hidden x 2`), held until the sequence that owns the
    /// image ends.
    pub feature_bytes_per_image: u64,
}

impl VisionMemory {
    /// Bytes to hold back for vision: one scratch set per encoder that can run at once, and the feature buffers of every image
    /// that can be alive at once.
    pub fn reserve_bytes(&self, encoders: u64, concurrent_images: u64) -> Result<u64, MemoryError> {
        let scratch = self.encoder_scratch_bytes.checked_mul(encoders);
        let features = self.feature_bytes_per_image.checked_mul(concurrent_images);
        scratch
            .zip(features)
            .and_then(|(s, f)| s.checked_add(f))
            .ok_or_else(|| MemoryError::Overflow("vision reserve".to_string()))
    }
}

/// The question every model family answers for the planner and for admission.
pub trait MemoryModel {
    /// Name for logs and errors.
    fn family(&self) -> &str;
    /// What a token costs in the paged cache (the attention layers, plus the drafter's own cache when it has one).
    fn kv_layout(&self) -> KvLayout;
    /// Layers that carry recurrent state (gated delta net, Kimi delta attention). Zero for a pure-attention family.
    fn recurrent_layers(&self) -> u64;
    /// Committed recurrent state of one sequence, in every recurrent layer, including any second buffer the design keeps.
    fn state_bytes_per_seq(&self) -> u64;
    /// Per-row history a speculative window keeps in every recurrent layer so an accepted prefix can be replayed.
    fn window_record_bytes_per_row(&self) -> u64;
    /// Activation buffers of the widest step when it carries `rows` rows.
    fn scratch_bytes(&self, rows: u64) -> u64;
    /// The blanket held back for everything allocated after the plan that is not modelled row by row (workspaces, handles).
    /// The planner reserves at least this much and more when the modelled buffers exceed it.
    fn activation_reserve(&self) -> u64;
    /// What a sequence slot needs on top of the target's own state for an external drafter (zero when there is none).
    fn drafter_state_bytes_per_slot(&self) -> u64;
    /// The vision tower, if the family serves images.
    fn vision(&self) -> Option<VisionMemory>;
}

/// The shapes a profile is evaluated at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepShape {
    /// Tokens per KV page.
    pub page_tokens: u64,
    /// Rows of the widest decode or verify window (all sequences together): sizes the per-row history.
    pub window_rows: u64,
    /// Rows of the widest step of any kind (a prefill chunk is wider than a window): sizes the activation buffers.
    pub scratch_rows: u64,
}

/// A family's memory model after validation. Fields are private: the only way to have one is [`MemoryProfile::new`], so
/// everything that takes a `&MemoryProfile` (the planner, `ModelMemory::from_profile`) can rely on `kv_bytes_per_token() > 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryProfile {
    family: String,
    shape: StepShape,
    kv_page_bytes: u64,
    kv_bytes_per_token: u64,
    recurrent_layers: u64,
    state_bytes_per_seq: u64,
    window_record_bytes_per_row: u64,
    scratch_bytes: u64,
    activation_floor: u64,
    drafter_state_bytes_per_slot: u64,
    vision: Option<VisionMemory>,
}

impl MemoryProfile {
    pub fn new(model: &dyn MemoryModel, shape: StepShape) -> Result<Self, MemoryError> {
        let family = model.family().to_string();
        let f = family.as_str();
        if shape.window_rows == 0 || shape.scratch_rows == 0 {
            return Err(MemoryError::Zero(format!("{f}: window rows or scratch rows")));
        }
        let layout = model.kv_layout();
        let kv_page_bytes = layout.bytes_per_page(f, shape.page_tokens)?;
        // exact and, because every extent is checked non-zero above, at least one byte
        let kv_bytes_per_token = layout.bytes_per_token(f, shape.page_tokens)?;
        let (rec_layers, state, records) =
            (model.recurrent_layers(), model.state_bytes_per_seq(), model.window_record_bytes_per_row());
        if rec_layers > 0 && (state == 0 || records == 0) {
            return Err(MemoryError::Zero(format!(
                "{f}: {rec_layers} recurrent layers but {state} state bytes per sequence and {records} window-record bytes per row"
            )));
        }
        if rec_layers == 0 && (state != 0 || records != 0) {
            return Err(MemoryError::Inconsistent(format!(
                "{f}: no recurrent layers but {state} state bytes per sequence and {records} window-record bytes per row"
            )));
        }
        let scratch_bytes = model.scratch_bytes(shape.scratch_rows);
        if scratch_bytes == 0 {
            return Err(MemoryError::Zero(format!("{f}: scratch bytes for {} rows", shape.scratch_rows)));
        }
        let vision = model.vision();
        if let Some(v) = &vision {
            if v.encoder_scratch_bytes == 0 || v.feature_bytes_per_image == 0 {
                return Err(MemoryError::Zero(format!("{f}: vision encoder scratch or feature bytes per image")));
            }
        }
        let p = Self {
            family,
            shape,
            kv_page_bytes,
            kv_bytes_per_token,
            recurrent_layers: rec_layers,
            state_bytes_per_seq: state,
            window_record_bytes_per_row: records,
            scratch_bytes,
            activation_floor: model.activation_reserve(),
            drafter_state_bytes_per_slot: model.drafter_state_bytes_per_slot(),
            vision,
        };
        // every derived total must fit in 64 bits, so the getters below never wrap
        p.state_bytes_per_slot_checked()?;
        p.window_record_bytes_checked()?;
        p.engine_reserve_checked()?;
        Ok(p)
    }

    pub fn family(&self) -> &str {
        &self.family
    }
    pub fn shape(&self) -> StepShape {
        self.shape
    }
    /// Bytes of one KV page across every layer.
    pub fn kv_page_bytes(&self) -> u64 {
        self.kv_page_bytes
    }
    /// KV bytes per token across every layer. Never zero.
    pub fn kv_bytes_per_token(&self) -> u64 {
        self.kv_bytes_per_token
    }
    pub fn recurrent_layers(&self) -> u64 {
        self.recurrent_layers
    }
    /// Committed recurrent state of one sequence (zero only for a family with no recurrent layer).
    pub fn state_bytes_per_seq(&self) -> u64 {
        self.state_bytes_per_seq
    }
    pub fn window_record_bytes_per_row(&self) -> u64 {
        self.window_record_bytes_per_row
    }
    pub fn drafter_state_bytes_per_slot(&self) -> u64 {
        self.drafter_state_bytes_per_slot
    }
    /// Activation buffers of the widest step.
    pub fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }
    /// The family's blanket for unmodelled allocations (the floor of [`MemoryProfile::engine_reserve_bytes`]).
    pub fn activation_floor(&self) -> u64 {
        self.activation_floor
    }
    pub fn vision(&self) -> Option<VisionMemory> {
        self.vision
    }

    fn state_bytes_per_slot_checked(&self) -> Result<u64, MemoryError> {
        self.state_bytes_per_seq
            .checked_add(self.drafter_state_bytes_per_slot)
            .ok_or_else(|| MemoryError::Overflow(format!("{}: state bytes per slot", self.family)))
    }
    fn window_record_bytes_checked(&self) -> Result<u64, MemoryError> {
        self.window_record_bytes_per_row
            .checked_mul(self.shape.window_rows)
            .ok_or_else(|| MemoryError::Overflow(format!("{}: window records", self.family)))
    }
    fn engine_reserve_checked(&self) -> Result<u64, MemoryError> {
        let modelled = self
            .scratch_bytes
            .checked_add(self.window_record_bytes_checked()?)
            .ok_or_else(|| MemoryError::Overflow(format!("{}: scratch plus window records", self.family)))?;
        Ok(modelled.max(self.activation_floor))
    }

    /// Bytes of state one sequence slot holds: the target's recurrent state plus the drafter's context.
    pub fn state_bytes_per_slot(&self) -> u64 {
        self.state_bytes_per_slot_checked().unwrap_or(u64::MAX)
    }
    /// Per-row history of a full window, in every recurrent layer.
    pub fn window_record_bytes(&self) -> u64 {
        self.window_record_bytes_checked().unwrap_or(u64::MAX)
    }
    /// What the planner holds back for allocations made after the plan: the larger of the family's blanket and the modelled
    /// activation buffers plus a full window of per-row history. It saturates at `u64::MAX` (nothing can satisfy that), never
    /// wraps.
    pub fn engine_reserve_bytes(&self) -> u64 {
        self.engine_reserve_checked().unwrap_or(u64::MAX)
    }
    /// Bytes to reserve for vision with `encoders` encoder scratch sets and `concurrent_images` live feature buffers; zero
    /// when the family has no tower.
    pub fn vision_reserve_bytes(&self, encoders: u64, concurrent_images: u64) -> Result<u64, MemoryError> {
        match &self.vision {
            Some(v) => v.reserve_bytes(encoders, concurrent_images),
            None => Ok(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A family whose numbers a test sets directly.
    struct Fake {
        layout: KvLayout,
        rec_layers: u64,
        state: u64,
        records: u64,
        scratch_per_row: u64,
        floor: u64,
        vision: Option<VisionMemory>,
    }

    impl Fake {
        fn gqa() -> Self {
            Self {
                layout: KvLayout::new().with(KvLayerClass::gqa(16, 4, 256, 2)),
                rec_layers: 48,
                state: 100,
                records: 10,
                scratch_per_row: 1000,
                floor: 5000,
                vision: None,
            }
        }
    }

    impl MemoryModel for Fake {
        fn family(&self) -> &str {
            "fake"
        }
        fn kv_layout(&self) -> KvLayout {
            self.layout.clone()
        }
        fn recurrent_layers(&self) -> u64 {
            self.rec_layers
        }
        fn state_bytes_per_seq(&self) -> u64 {
            self.state
        }
        fn window_record_bytes_per_row(&self) -> u64 {
            self.records
        }
        fn scratch_bytes(&self, rows: u64) -> u64 {
            self.scratch_per_row * rows
        }
        fn activation_reserve(&self) -> u64 {
            self.floor
        }
        fn drafter_state_bytes_per_slot(&self) -> u64 {
            7
        }
        fn vision(&self) -> Option<VisionMemory> {
            self.vision
        }
    }

    const SHAPE: StepShape = StepShape { page_tokens: 256, window_rows: 64, scratch_rows: 4096 };

    #[test]
    fn gqa_kv_matches_the_published_qwen_budget() {
        // r0b0tlab/qwen38-exl3-dflash2: 16 full-attention layers x 4 KV heads x 256 dim x 2 (K and V), fp16:
        // 64 KiB per token, 16.0 GiB at 262,144 tokens.
        let l = KvLayout::new().with(KvLayerClass::gqa(16, 4, 256, 2));
        assert_eq!(l.bytes_per_token("qwen", 256), Ok(65536));
        assert_eq!(l.bytes_per_page("qwen", 256), Ok(16 << 20));
        assert_eq!(l.bytes_per_token("qwen", 256).unwrap() * 262144, 16 * (1u64 << 30));
    }

    #[test]
    fn a_pooled_plane_costs_its_entries_per_page() {
        // 128 fp16 values per 4 tokens: 64 pools in a 256-token page, 16 KiB a page, 64 bytes a token
        let l = KvLayout::new().with(KvLayerClass { name: "p", layers: 1, planes: vec![KvPlane::pooled("pool", 128, 4, 2)] });
        assert_eq!(l.bytes_per_page("f", 256), Ok(128 * 2 * 64));
        assert_eq!(l.bytes_per_token("f", 256), Ok(64));
    }

    #[test]
    fn layouts_with_a_zero_extent_are_refused() {
        let zero_dim = KvLayout::new().with(KvLayerClass::gqa(16, 4, 0, 2));
        assert!(matches!(zero_dim.bytes_per_token("f", 256), Err(MemoryError::Zero(m)) if m.contains("plane `k`")));
        let zero_heads = KvLayout::new().with(KvLayerClass::gqa(16, 0, 256, 2));
        assert!(matches!(zero_heads.bytes_per_page("f", 256), Err(MemoryError::Zero(_))));
        let zero_layers = KvLayout::new().with(KvLayerClass::gqa(0, 4, 256, 2));
        assert!(matches!(zero_layers.bytes_per_page("f", 256), Err(MemoryError::Zero(_))));
        assert!(matches!(KvLayout::new().bytes_per_page("f", 256), Err(MemoryError::Zero(_))));
        assert!(matches!(KvLayout::new().with(KvLayerClass::gqa(1, 1, 1, 2)).bytes_per_page("f", 0), Err(MemoryError::Zero(_))));
    }

    #[test]
    fn a_pool_must_divide_the_page_and_the_page_must_divide_into_bytes_per_token() {
        let l = KvLayout::new().with(KvLayerClass { name: "p", layers: 1, planes: vec![KvPlane::pooled("pool", 8, 3, 2)] });
        assert!(matches!(l.bytes_per_page("f", 256), Err(MemoryError::Inconsistent(m)) if m.contains("does not divide")));
        // one byte for a 2-token page is exact per page but not a whole number of bytes per token
        let odd = KvLayout::new().with(KvLayerClass { name: "p", layers: 1, planes: vec![KvPlane::pooled("x", 1, 2, 1)] });
        assert_eq!(odd.bytes_per_page("f", 2), Ok(1));
        assert!(matches!(odd.bytes_per_token("f", 2), Err(MemoryError::Inconsistent(_))));
    }

    #[test]
    fn layout_arithmetic_overflow_is_an_error_not_a_wrap() {
        let l = KvLayout::new().with(KvLayerClass::gqa(u64::MAX / 2, 1 << 20, 1 << 20, 2));
        assert!(matches!(l.bytes_per_page("f", 256), Err(MemoryError::Overflow(_))));
    }

    #[test]
    fn a_sound_family_becomes_a_profile() {
        let p = MemoryProfile::new(&Fake::gqa(), SHAPE).unwrap();
        assert_eq!(p.kv_bytes_per_token(), 65536);
        assert_eq!(p.kv_page_bytes(), 16 << 20);
        assert_eq!(p.state_bytes_per_slot(), 107, "state 100 plus the drafter's 7");
        assert_eq!(p.window_record_bytes(), 640);
        assert_eq!(p.scratch_bytes(), 4_096_000);
        assert_eq!(p.vision_reserve_bytes(1, 5), Ok(0), "no tower, no reserve");
    }

    #[test]
    fn zero_kv_cost_cannot_become_a_profile() {
        // the GLM hazard: a config read as grouped-query attention with head_dim 0 costs no KV bytes at all
        let mut f = Fake::gqa();
        f.layout = KvLayout::new().with(KvLayerClass::gqa(11, 64, 0, 2));
        let e = MemoryProfile::new(&f, SHAPE).unwrap_err();
        assert!(matches!(&e, MemoryError::Zero(m) if m.contains("fake") && m.contains("plane `k`")), "{e}");
    }

    #[test]
    fn recurrent_families_must_state_their_state_and_pure_attention_families_must_not() {
        let mut f = Fake::gqa();
        f.state = 0;
        assert!(matches!(MemoryProfile::new(&f, SHAPE), Err(MemoryError::Zero(m)) if m.contains("recurrent layers")));
        let mut f = Fake::gqa();
        f.records = 0;
        assert!(matches!(MemoryProfile::new(&f, SHAPE), Err(MemoryError::Zero(_))));
        let mut f = Fake::gqa();
        f.rec_layers = 0;
        assert!(matches!(MemoryProfile::new(&f, SHAPE), Err(MemoryError::Inconsistent(_))), "state without a recurrent layer");
        f.state = 0;
        f.records = 0;
        assert!(MemoryProfile::new(&f, SHAPE).is_ok(), "a pure-attention family has no recurrent state");
    }

    #[test]
    fn a_step_with_no_scratch_or_no_rows_is_refused() {
        let mut f = Fake::gqa();
        f.scratch_per_row = 0;
        assert!(matches!(MemoryProfile::new(&f, SHAPE), Err(MemoryError::Zero(m)) if m.contains("scratch")));
        let bad = StepShape { window_rows: 0, ..SHAPE };
        assert!(matches!(MemoryProfile::new(&Fake::gqa(), bad), Err(MemoryError::Zero(_))));
        let bad = StepShape { scratch_rows: 0, ..SHAPE };
        assert!(matches!(MemoryProfile::new(&Fake::gqa(), bad), Err(MemoryError::Zero(_))));
    }

    #[test]
    fn the_engine_reserve_is_the_larger_of_the_blanket_and_the_modelled_buffers() {
        // scratch 4_096_000 + records 640 = 4_096_640 against a 5_000 blanket: the model governs
        let p = MemoryProfile::new(&Fake::gqa(), SHAPE).unwrap();
        assert_eq!(p.engine_reserve_bytes(), 4_096_640);
        // a blanket above the model governs
        let mut f = Fake::gqa();
        f.floor = 10_000_000;
        assert_eq!(MemoryProfile::new(&f, SHAPE).unwrap().engine_reserve_bytes(), 10_000_000);
        // a model that outgrows its blanket raises the reserve instead of overcommitting
        let mut f = Fake::gqa();
        f.floor = 1;
        let narrow = StepShape { scratch_rows: 8, ..SHAPE };
        assert_eq!(MemoryProfile::new(&f, narrow).unwrap().engine_reserve_bytes(), 8_000 + 640);
    }

    #[test]
    fn derived_totals_that_overflow_are_refused_at_construction() {
        let mut f = Fake::gqa();
        f.records = u64::MAX / 2;
        assert!(matches!(MemoryProfile::new(&f, SHAPE), Err(MemoryError::Overflow(m)) if m.contains("window records")));
        let mut f = Fake::gqa();
        f.state = u64::MAX;
        assert!(matches!(MemoryProfile::new(&f, SHAPE), Err(MemoryError::Overflow(m)) if m.contains("state bytes per slot")));
    }

    #[test]
    fn the_vision_reserve_counts_scratch_per_encoder_and_features_per_image() {
        let mut f = Fake::gqa();
        f.vision = Some(VisionMemory { weight_bytes: Some(9), encoder_scratch_bytes: 1000, feature_bytes_per_image: 30 });
        let p = MemoryProfile::new(&f, SHAPE).unwrap();
        assert_eq!(p.vision_reserve_bytes(1, 32), Ok(1000 + 32 * 30));
        assert_eq!(p.vision_reserve_bytes(2, 0), Ok(2000));
        assert!(matches!(p.vision_reserve_bytes(1, u64::MAX), Err(MemoryError::Overflow(_))));
        f.vision = Some(VisionMemory { weight_bytes: None, encoder_scratch_bytes: 1000, feature_bytes_per_image: 0 });
        assert!(matches!(MemoryProfile::new(&f, SHAPE), Err(MemoryError::Zero(m)) if m.contains("vision")));
    }
}
