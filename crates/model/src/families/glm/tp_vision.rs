//! Images in a GLM tensor-parallel group (`docs/GLM_TP.md` section 10): the family call [`PrefillImage`] that carries the f32 feature
//! rows of the images inside ONE prefill chunk, the leader's side (the rows read off rank 0's device through a [`FeatureSource`], the
//! chunk shortened so the call fits one frame) and every rank's side ([`ImageRuntime`]: the rows uploaded to a buffer of the rank's own
//! and spliced by `GlmModel::prefill_chunk_mm`, the single node's call).
//!
//! **Only rank 0 has a vision tower.** The server encodes an image on rank 0 (its `GlmEncoder`); a follower never loads the tower (1.05 GiB
//! of weights and up to 1.04 GiB of scratch it does not hold) and never recomputes it: it splices the bytes it is sent. The tower's own
//! arithmetic is therefore rank 0's alone, and the group's exactness needs only that the bytes the follower splices are the bytes rank 0's
//! tower wrote, which a device-to-host copy, a frame and a host-to-device copy preserve bit for bit.
//!
//! **What travels (checklist 3.1).** Everything the splice reads from outside the call stream: the chunk's `RowBatch` (as a core prefill
//! carries it), `want`, and per image that the chunk touches the destination row, the row count and the rows themselves (`n x 4096` f32,
//! little-endian, the device bytes). The source row is NOT sent: a rank builds a buffer of exactly the sent rows, so its splice reads from
//! row 0. The image spans of a slot (what `spec_allowed_for_window` asks) are host state of rank 0's backend and never travel: the follower
//! runs no drafter arbiter. The chunk grid is rank 0's choice and moves no bit (every GLM row has the decode kernels' bits whatever the grid,
//! `docs/GLM53.md`); a chunk of an image prompt that holds no image row is the core `prefill_chunk`, exactly as `GlmBackend` runs it.
//!
//! **Frames.** One call is one frame of at most `MAX_FRAME` bytes. A chunk of the largest grid (2,048 rows) carries 32 MiB of rows, under
//! the 64 MiB limit; the leader still measures every call before it forwards it and ends the chunk earlier when it would not fit (exact, by
//! the grid argument above), and refuses the request by name if not even one row fits: a frame is never truncated and an oversized frame
//! never reaches the link (which would lose the group).

use super::backend::window_is_text;
use super::tp::{chunk_of, GlmCall, GlmRank};
use super::vision::{self, GlmVisionTower, TowerOptions, OUT_HIDDEN};
use super::vision_serve::{FeatureBudget, GlmEncoder, GlmImageFeatures, SpliceOp};
use super::GlmPack;
use crate::error::ModelError;
use crate::forward::Candidates;
use crate::lockstep::{
    candidates_digest, dec_batch, enc_batch, Dec, Enc, LockstepRuntime, TpLeader, WireError, HEADER, MAX_FRAME,
};
use crate::memory::VisionOptions;
use crate::rows::RowBatch;
use crate::serve::{gib, mem_available, ServeOptions};
use gb_engine::{ImageSpan, RowRequest};
use gb_sched::MemoryProfile;
use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

/// The frame kind of [`PrefillImage`]. Kinds 32 and 33 are planned for the NextN drafter's calls (`docs/GLM_TP.md` 4.1); 40 leaves room.
pub const K_PREFILL_IMAGE: u8 = 40;

/// Bytes of one feature row: `OUT_HIDDEN` f32.
pub const ROW_BYTES: usize = OUT_HIDDEN * 4;

/// The largest call body one frame holds (the frame's header is the rest of `MAX_FRAME`).
pub const MAX_IMAGE_BODY: usize = MAX_FRAME - HEADER;

/// One image's rows inside a chunk: chunk rows `dst_row..dst_row + n` are overwritten by `data` (`n` rows of [`ROW_BYTES`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpliceWire {
    pub dst_row: u64,
    pub n: u64,
    pub data: Vec<u8>,
}

/// A prefill chunk that holds image rows: the core prefill's batch and `want`, and the rows to splice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefillImage {
    pub batch: RowBatch,
    pub want: Option<u32>,
    pub splice: Vec<SpliceWire>,
}

impl PrefillImage {
    pub fn encode(&self, e: &mut Enc) {
        enc_batch(e, &self.batch);
        e.opt_u32(self.want);
        e.u64(self.splice.len() as u64);
        for s in &self.splice {
            e.u64(s.dst_row);
            e.u64(s.n);
            e.bytes(&s.data);
        }
    }

    pub fn decode(d: &mut Dec<'_>) -> Result<PrefillImage, WireError> {
        let batch = dec_batch(d)?;
        let want = d.opt_u32("prefill_image want")?;
        let n = d.len(24, "prefill_image splices")?;
        let mut splice = Vec::with_capacity(n);
        for _ in 0..n {
            splice.push(SpliceWire {
                dst_row: d.u64("splice dst_row")?,
                n: d.u64("splice rows")?,
                data: d.bytes("splice rows' bytes")?,
            });
        }
        Ok(PrefillImage { batch, want, splice })
    }

    /// The exact length of [`PrefillImage::encode`]'s output, computed without encoding (the leader measures a call before it builds it).
    pub fn encoded_len(&self) -> usize {
        let b = &self.batch;
        let vec = |n: usize| 8 + 4 * n;
        let opt = |v: &Option<Vec<i32>>| 1 + v.as_ref().map_or(0, |v| vec(v.len()));
        8 + vec(b.ids.len())
            + vec(b.positions.len())
            + vec(b.row_slot.len())
            + vec(b.seq_start.len())
            + vec(b.seq_slot.len())
            + opt(&b.rope3)
            + opt(&b.parents)
            + if self.want.is_some() { 5 } else { 1 }
            + 8
            + self.splice.iter().map(|s| 24 + s.data.len()).sum::<usize>()
    }

    /// The `(slot, start, tokens)` of the chunk, refused (before anything is launched, so on every rank alike) unless the batch is one
    /// sequence of consecutive rows (`tp::chunk_of`) and every splice is a whole number of rows inside the chunk, in order, not overlapping.
    pub fn check(&self) -> Result<(usize, usize, Vec<u32>), ModelError> {
        let (slot, start, tokens) = chunk_of(&self.batch)?;
        let mut at = 0u64;
        for s in &self.splice {
            let fits = s.n > 0
                && s.dst_row >= at
                && s.dst_row.checked_add(s.n).is_some_and(|e| e <= tokens.len() as u64)
                && (s.n as usize).checked_mul(ROW_BYTES) == Some(s.data.len());
            if !fits {
                return Err(ModelError::Request(format!(
                    "an image splice of {} rows ({} bytes) at chunk row {} does not fit a chunk of {} rows after row {at}",
                    s.n,
                    s.data.len(),
                    s.dst_row,
                    tokens.len()
                )));
            }
            at = s.dst_row + s.n;
        }
        Ok((slot, start, tokens))
    }
}

/// How rank 0 reads the rows of an image's features (`ImageSpan::features`, made by its encoder): the device bridge in serving, a host
/// one in the CPU tests.
pub trait FeatureSource: Send {
    /// The encoder rows and merged grid of features this source produced (`None`: not features of this source).
    fn shape(&self, f: &(dyn Any + Send + Sync)) -> Option<(usize, (usize, usize))>;
    /// The bytes of rows `src_row..src_row + n` (`n x ROW_BYTES`).
    fn export(&self, f: &(dyn Any + Send + Sync), src_row: usize, n: usize) -> Result<Vec<u8>, ModelError>;
}

/// Features made by `GlmEncoder` (`GlmImageFeatures`, f32 rows on the device): read with a device-to-host copy.
pub struct DeviceFeatures;

impl FeatureSource for DeviceFeatures {
    fn shape(&self, f: &(dyn Any + Send + Sync)) -> Option<(usize, (usize, usize))> {
        f.downcast_ref::<GlmImageFeatures>().map(|g| (g.tokens, g.merged))
    }

    fn export(&self, f: &(dyn Any + Send + Sync), src_row: usize, n: usize) -> Result<Vec<u8>, ModelError> {
        let g = f
            .downcast_ref::<GlmImageFeatures>()
            .ok_or_else(|| ModelError::Request("image features were not produced by this backend".into()))?;
        if n == 0 || src_row + n > g.tokens || g.buf.bytes() < g.tokens * ROW_BYTES {
            return Err(ModelError::Request(format!(
                "rows {src_row}..{} of an image of {} rows ({} bytes) cannot be read",
                src_row + n,
                g.tokens,
                g.buf.bytes()
            )));
        }
        let mut out = vec![0u8; n * ROW_BYTES];
        g.buf.download_range(src_row * ROW_BYTES, &mut out)?;
        Ok(out)
    }
}

/// Host features (`HostImageFeatures`): what the CPU tests put behind an `ImageSpan`.
pub struct HostFeatures;

/// Features held on the host: `rows` is `tokens x OUT_HIDDEN` f32.
#[derive(Debug)]
pub struct HostImageFeatures {
    pub rows: Vec<f32>,
    pub tokens: usize,
    pub merged: (usize, usize),
}

impl FeatureSource for HostFeatures {
    fn shape(&self, f: &(dyn Any + Send + Sync)) -> Option<(usize, (usize, usize))> {
        f.downcast_ref::<HostImageFeatures>().map(|h| (h.tokens, h.merged))
    }

    fn export(&self, f: &(dyn Any + Send + Sync), src_row: usize, n: usize) -> Result<Vec<u8>, ModelError> {
        let h = f
            .downcast_ref::<HostImageFeatures>()
            .ok_or_else(|| ModelError::Request("image features were not produced by this backend".into()))?;
        let rows = h.rows.get(src_row * OUT_HIDDEN..(src_row + n) * OUT_HIDDEN).ok_or_else(|| {
            ModelError::Request(format!("rows {src_row}..{} of an image of {} rows cannot be read", src_row + n, h.tokens))
        })?;
        Ok(rows.iter().flat_map(|x| x.to_le_bytes()).collect())
    }
}

/// What rank 0's backend needs to serve images: where the rows come from, the placeholder token, the frame budget of one call.
pub struct TpVision {
    pub source: Box<dyn FeatureSource>,
    /// The pack's image placeholder id (`GlmVisionConfig::image_token_id`): every row of a span must hold it.
    pub image_token: Option<u32>,
    /// The largest call body forwarded ([`MAX_IMAGE_BODY`]; smaller only in tests).
    pub frame_budget: usize,
}

impl TpVision {
    /// The serving configuration: the device bridge and the whole frame.
    pub fn device(image_token: Option<u32>) -> TpVision {
        TpVision { source: Box::new(DeviceFeatures), image_token, frame_budget: MAX_IMAGE_BODY }
    }

    /// Checks the images of a prompt of `prompt.len()` tokens before anything is touched (as `GlmBackend::image_features`): features
    /// this source produced, one encoder row per placeholder, the placeholder tokens at the span, the spans in order and inside the prompt.
    pub fn check_images(&self, prompt: &[u32], images: &[ImageSpan]) -> Result<(), ModelError> {
        let mut at = 0usize;
        for im in images {
            let (tokens, merged) = self
                .source
                .shape(&*im.features)
                .ok_or_else(|| ModelError::Request("image features were not produced by this backend".into()))?;
            if tokens != im.tokens || merged != im.merged {
                return Err(ModelError::Request(format!(
                    "image has {} encoder rows ({}x{}) for {} placeholders ({}x{})",
                    tokens, merged.0, merged.1, im.tokens, im.merged.0, im.merged.1
                )));
            }
            if im.start < at || im.start + im.tokens > prompt.len() {
                return Err(ModelError::Request(format!(
                    "image span at {} of {} tokens does not fit the prompt of {} tokens",
                    im.start,
                    im.tokens,
                    prompt.len()
                )));
            }
            if self.image_token.is_some_and(|t| prompt[im.start..im.start + im.tokens].iter().any(|&x| x != t)) {
                return Err(ModelError::Request(format!(
                    "the {} tokens at {} are not all image placeholders",
                    im.tokens, im.start
                )));
            }
            at = im.start + im.tokens;
        }
        Ok(())
    }

    /// The call of the chunk `at..end` of `prompt` in `slot` (`end == prompt.len()` asks for `candidates`): a core prefill when no image
    /// row is inside it, else a [`PrefillImage`] with the rows read now. When the call would not fit [`TpVision::frame_budget`] the chunk
    /// is ended earlier (halved until it fits); a chunk of one row that does not fit is refused by name. Returns the chunk's end too.
    pub fn chunk_call(
        &self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        at: usize,
        mut end: usize,
        candidates: usize,
    ) -> Result<(ChunkCall, usize), ModelError> {
        loop {
            let want = (end == prompt.len()).then_some(candidates as u32);
            let batch =
                RowBatch::from_requests(&[RowRequest { slot, tokens: prompt[at..end].to_vec(), first_position: at as u64 }]);
            let parts = splice_parts(images, at, end);
            if parts.is_empty() {
                return Ok((ChunkCall::Text(batch, want.map(|w| w as usize)), end));
            }
            // measure the call before reading a byte of it
            let rows: usize = parts.iter().map(|p| p.n).sum();
            let probe = PrefillImage {
                batch,
                want,
                splice: parts.iter().map(|p| SpliceWire { dst_row: p.dst_row as u64, n: p.n as u64, data: Vec::new() }).collect(),
            };
            let len = probe.encoded_len() + rows * ROW_BYTES;
            if len <= self.frame_budget {
                let mut call = probe;
                for (w, p) in call.splice.iter_mut().zip(&parts) {
                    w.data = self.source.export(&*images[p.image].features, p.src_row, p.n)?;
                }
                debug_assert_eq!(call.encoded_len(), len);
                return Ok((ChunkCall::Image(call), end));
            }
            if end - at <= 1 {
                return Err(ModelError::Request(format!(
                    "one image row of a prefill call is {len} bytes, over the {}-byte frame of a tensor-parallel call",
                    self.frame_budget
                )));
            }
            end = at + (end - at) / 2;
        }
    }
}

/// One image's rows inside a chunk (as `GlmBackend::splice_for`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplicePart {
    pub image: usize,
    pub dst_row: usize,
    pub src_row: usize,
    pub n: usize,
}

/// The rows of each image that fall inside the chunk `at..end`.
pub fn splice_parts(images: &[ImageSpan], at: usize, end: usize) -> Vec<SplicePart> {
    images
        .iter()
        .enumerate()
        .filter_map(|(i, im)| {
            let (lo, hi) = (im.start.max(at), (im.start + im.tokens).min(end));
            (lo < hi).then(|| SplicePart { image: i, dst_row: lo - at, src_row: lo - im.start, n: hi - lo })
        })
        .collect()
}

/// The image spans of every slot (rank 0's host state: `spec_allowed_for_window` reads it; it never travels).
#[derive(Clone, Debug, Default)]
pub struct SlotImages(Vec<Vec<Range<usize>>>);

impl SlotImages {
    pub fn new(slots: usize) -> SlotImages {
        SlotImages(vec![Vec::new(); slots])
    }
    pub fn set(&mut self, slot: usize, images: &[ImageSpan]) {
        if let Some(s) = self.0.get_mut(slot) {
            *s = images.iter().map(|im| im.start..im.start + im.tokens).collect();
        }
    }
    pub fn clear(&mut self, slot: usize) {
        if let Some(s) = self.0.get_mut(slot) {
            s.clear();
        }
    }
    /// Every slot's spans (what `SpecRuntime::propose` reads, as one node's `slot_images`).
    pub fn spans(&self) -> &[Vec<Range<usize>>] {
        &self.0
    }
    /// Whether none of the `rows` positions from `first` of `slot` holds an image row (`GlmBackend::spec_allowed_for_window`).
    pub fn allows(&self, slot: usize, first: usize, rows: usize) -> bool {
        window_is_text(self.0.get(slot).map_or(&[], |v| v.as_slice()), first, rows)
    }
}

/// The call one chunk of an image prompt makes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChunkCall {
    /// No image row inside: the core prefill (the batch and `want`).
    Text(RowBatch, Option<usize>),
    Image(PrefillImage),
}

/// A runtime that can apply a [`PrefillImage`] (every rank: rank 0 applies the call it forwards, the follower the call it receives).
pub trait ImageRuntime: LockstepRuntime<Call = GlmCall> {
    fn prefill_image(&mut self, call: &PrefillImage) -> Result<Option<Candidates>, ModelError>;
}

/// A rank's own buffer of the sent rows, as a splice from its row 0.
fn import(s: &SpliceWire) -> Result<SpliceOp, ModelError> {
    let n = s.n as usize;
    let buf = gb_cuda::DevBuf::from_slice(&s.data)?;
    let feats = GlmImageFeatures { buf, tokens: n, merged: (n, 1), key: 0, permit: None };
    Ok(SpliceOp { feats: Arc::new(feats), dst_row: s.dst_row as usize, src_row: 0, n })
}

impl ImageRuntime for GlmRank {
    fn prefill_image(&mut self, call: &PrefillImage) -> Result<Option<Candidates>, ModelError> {
        let (slot, start, tokens) = call.check()?;
        if slot >= self.m.slots {
            return Err(ModelError::Request(format!("slot {slot} of a model of {} slots", self.m.slots)));
        }
        let ops = call.splice.iter().map(import).collect::<Result<Vec<_>, _>>()?;
        // the model's stream is non-blocking: the copies must have landed before the chunk that reads them is launched
        gb_cuda::rt::device_sync()?;
        let t = self.m.slices.enter(super::profile_slices::Kind::Prefill, tokens.len(), Some(slot));
        let r = self.m.prefill_chunk_mm(slot, start, &tokens, call.want.map(|w| w as usize), ops);
        self.m.slices.leave(t, r.is_ok());
        r
    }
}

/// Rank 0: forwards `call` and applies it here (the same decoded bytes on every rank); the ack's digest is the candidates'.
pub fn forward_image<R: ImageRuntime>(leader: &mut TpLeader<R>, call: PrefillImage) -> Result<Option<Candidates>, ModelError> {
    let local = call.clone();
    leader.forward_call(
        GlmCall::PrefillImage(call),
        move |m, _| m.prefill_image(&local),
        |c: &Option<Candidates>| c.as_ref().map_or(0, candidates_digest),
    )
}

/// The follower's answer to a [`PrefillImage`]: the candidates' digest, compared with rank 0's.
pub fn apply_image<R: ImageRuntime>(rt: &mut R, call: &PrefillImage) -> Result<crate::lockstep::CallReply, ModelError> {
    rt.prefill_image(call).map(|c| crate::lockstep::CallReply::digest(c.as_ref().map_or(0, candidates_digest)))
}

// ------------------------------------------------------------------------------------------------------------------- the leader's tower

/// Whether this rank of a group holds the vision tower: rank 0 only, serving images (`ServeOptions::vision`), with a pack that has one.
/// The `vision` fact of the Hello (`GlmFacts::vision`).
pub fn holds_tower(o: &ServeOptions, pack: &GlmPack) -> bool {
    let rank0 = o.tp.as_ref().is_none_or(|t| t.rank == 0);
    rank0 && o.vision.is_some() && pack.config.vision.is_some() && pack.weights.vision.is_some()
}

/// What the tower costs a rank's plan, before anything is loaded: `(the tower's weights and scratch, the features reserve)`. Both zero
/// unless [`holds_tower`]. The single node's arithmetic (`engine::build_engine`): the tower is allocated before the pool is planned, and
/// the features of the images in flight are held back from it (`vision_reserve_bytes(0, slots x images a request)`).
pub fn planned_vision(
    o: &ServeOptions,
    pack: &GlmPack,
    profile: &MemoryProfile,
    vision: Option<VisionOptions>,
) -> Result<(u64, u64), ModelError> {
    let (Some(v), Some(vc), true) = (vision, pack.config.vision.as_ref(), holds_tower(o, pack)) else { return Ok((0, 0)) };
    let tokens = vision::image_token_cap(&pack.config, o.max_image_tokens);
    let tower = vision::weight_region_bytes(vc) + vision::scratch_bytes(tokens * vision::PATCHES_PER_TOKEN, vc.tower);
    let reserve = profile.vision_reserve_bytes(0, v.concurrent_images(o.slots)).map_err(|e| ModelError::Config(e.to_string()))?;
    Ok((tower, reserve))
}

/// Loads rank 0's tower (when [`holds_tower`]) from `pack` (the rank's map: the vision region is mapped whatever the expert share),
/// refused by name when MemAvailable would fall under the floor, and its allocations checked against the memory model.
pub fn load_leader_tower(o: &ServeOptions, pack: &GlmPack) -> Result<Option<GlmVisionTower>, ModelError> {
    if !holds_tower(o, pack) {
        return Ok(None);
    }
    let vc = pack.config.vision.as_ref().ok_or_else(|| ModelError::Config("the pack has no vision tower".into()))?;
    let tokens = vision::image_token_cap(&pack.config, o.max_image_tokens);
    let need = vision::weight_region_bytes(vc) + vision::scratch_bytes(tokens * vision::PATCHES_PER_TOKEN, vc.tower);
    if let Some(avail) = mem_available() {
        if avail < need.saturating_add(o.floor_bytes) {
            return Err(ModelError::Config(format!(
                "rank 0's vision tower needs {:.2} GiB (weights and scratch for images of {tokens} tokens) and MemAvailable is {:.2} GiB with a \
                 {:.2} GiB floor; start rank 0 with --no-vision or a lower --max-image-tokens",
                gib(need),
                gib(avail),
                gib(o.floor_bytes)
            )));
        }
    }
    let t = GlmVisionTower::load(pack, &o.model_dir, &TowerOptions::new(tokens * vision::PATCHES_PER_TOKEN, o.floor_bytes))?;
    if t.scratch_allocations() != vision::scratch_items(t.max_patches, t.format)
        || t.weight_bytes() != vision::weight_region_bytes(vc)
    {
        return Err(ModelError::Config("the vision memory model and the tower's allocations disagree".into()));
    }
    eprintln!(
        "rexl3 event=glm_vision_loaded tokens={tokens} weights_gib={:.3} scratch_gib={:.3} load_s={:.2} tp_rank=0",
        gib(t.weight_bytes()),
        gib(t.scratch_bytes()),
        t.boot.load_s
    );
    Ok(Some(t))
}

/// Publishes rank 0's tower as the server's image encoder (after the engine exists, as the single node does), with the features budget
/// the plan held back.
pub fn publish_encoder(o: &ServeOptions, t: GlmVisionTower, features_reserve: u64) -> Result<(), ModelError> {
    let Some(slot) = o.vision.as_ref() else { return Ok(()) };
    let tokens = t.max_patches / vision::PATCHES_PER_TOKEN;
    let pp_id = super::image_preprocessor(&o.model_dir, o)?.map(|p| p.id()).unwrap_or_default();
    let enc = GlmEncoder::new(t, FeatureBudget::new(features_reserve), &pp_id)?;
    let _ = slot.set(Arc::new(crate::serve::VisionService::new(Box::new(enc), tokens)));
    eprintln!("rexl3 event=tp_vision_ready family=glm tokens={tokens} features_reserve_gib={:.2}", gib(features_reserve));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(tokens: usize, merged: (usize, usize), seed: u32) -> Arc<dyn Any + Send + Sync> {
        let rows = (0..tokens * OUT_HIDDEN)
            .map(|i| f32::from_bits(0x3f80_0000 ^ (i as u32).wrapping_mul(2_654_435_761) ^ seed))
            .collect();
        Arc::new(HostImageFeatures { rows, tokens, merged })
    }

    fn span(start: usize, tokens: usize, f: Arc<dyn Any + Send + Sync>) -> ImageSpan {
        ImageSpan { start, tokens, merged: (1, tokens), features: f, key: 0 }
    }

    #[test]
    fn the_encoded_length_is_the_encoders_and_the_call_round_trips() {
        let b = RowBatch::from_requests(&[RowRequest { slot: 1, tokens: vec![5, 6, 7, 8], first_position: 30 }]);
        for want in [None, Some(3)] {
            let c = PrefillImage {
                batch: b.clone(),
                want,
                splice: vec![
                    SpliceWire { dst_row: 0, n: 1, data: vec![1; ROW_BYTES] },
                    SpliceWire { dst_row: 2, n: 2, data: vec![2; 2 * ROW_BYTES] },
                ],
            };
            let mut e = Enc::new();
            c.encode(&mut e);
            assert_eq!(e.0.len(), c.encoded_len());
            let mut d = Dec::new(&e.0);
            assert_eq!(PrefillImage::decode(&mut d).unwrap(), c);
            d.done("prefill_image").unwrap();
            assert_eq!(c.check().unwrap(), (1, 30, vec![5, 6, 7, 8]));
        }
    }

    #[test]
    fn a_splice_outside_its_chunk_or_of_a_partial_row_is_refused_by_name() {
        let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: vec![1, 2, 3], first_position: 0 }]);
        let bad = [
            vec![SpliceWire { dst_row: 2, n: 2, data: vec![0; 2 * ROW_BYTES] }],
            vec![SpliceWire { dst_row: 0, n: 1, data: vec![0; ROW_BYTES - 4] }],
            vec![SpliceWire { dst_row: 0, n: 0, data: vec![] }],
            vec![
                SpliceWire { dst_row: 1, n: 1, data: vec![0; ROW_BYTES] },
                SpliceWire { dst_row: 1, n: 1, data: vec![0; ROW_BYTES] },
            ],
        ];
        for s in bad {
            let e = PrefillImage { batch: b.clone(), want: None, splice: s }.check().unwrap_err().to_string();
            assert!(e.contains("image splice"), "{e}");
        }
    }

    #[test]
    fn the_rows_of_a_chunk_are_the_images_rows_inside_it() {
        let f = host(10, (2, 5), 1);
        let g = host(3, (1, 3), 2);
        let ims = [span(4, 10, f), span(20, 3, g)];
        assert_eq!(splice_parts(&ims, 0, 4), vec![]);
        assert_eq!(splice_parts(&ims, 0, 8), vec![SplicePart { image: 0, dst_row: 4, src_row: 0, n: 4 }]);
        assert_eq!(
            splice_parts(&ims, 8, 22),
            vec![SplicePart { image: 0, dst_row: 0, src_row: 4, n: 6 }, SplicePart { image: 1, dst_row: 12, src_row: 0, n: 2 }]
        );
        assert_eq!(splice_parts(&ims, 23, 30), vec![]);
    }

    #[test]
    fn a_call_over_the_frame_budget_ends_its_chunk_earlier_and_a_single_row_over_it_is_refused() {
        let prompt: Vec<u32> = (0..40).map(|i| if (2..34).contains(&i) { 7 } else { i }).collect();
        let ims = [span(2, 32, host(32, (1, 32), 3))];
        let mut v = TpVision { source: Box::new(HostFeatures), image_token: Some(7), frame_budget: MAX_IMAGE_BODY };
        v.check_images(&prompt, &ims).unwrap();
        let (c, end) = v.chunk_call(0, &prompt, &ims, 0, 40, 4).unwrap();
        assert_eq!(end, 40);
        let ChunkCall::Image(c) = c else { panic!("an image chunk") };
        assert_eq!((c.splice.len(), c.splice[0].n, c.want), (1, 32, Some(4)));
        // a budget of about 9 rows: 40 -> 20 -> 10 rows, the rows of the image inside it only, no candidates (not the end)
        v.frame_budget = 9 * ROW_BYTES + 4096;
        let (c, end) = v.chunk_call(0, &prompt, &ims, 0, 40, 4).unwrap();
        assert_eq!(end, 10);
        let ChunkCall::Image(c) = c else { panic!("an image chunk") };
        assert!(c.encoded_len() <= v.frame_budget);
        assert_eq!((c.splice[0].dst_row, c.splice[0].n, c.want), (2, 8, None));
        // the rows are the image's, byte for byte
        let HostImageFeatures { rows, .. } = ims[0].features.downcast_ref::<HostImageFeatures>().unwrap();
        let want: Vec<u8> = rows[..8 * OUT_HIDDEN].iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(c.splice[0].data, want);
        // not even one row fits: refused by name, nothing read
        v.frame_budget = ROW_BYTES;
        let e = v.chunk_call(0, &prompt, &ims, 2, 40, 4).unwrap_err().to_string();
        assert!(e.contains("frame of a tensor-parallel call"), "{e}");
        // a chunk with no image row is the core prefill whatever the budget
        let (c, end) = v.chunk_call(0, &prompt, &ims, 34, 40, 4).unwrap();
        assert_eq!((end, matches!(c, ChunkCall::Text(_, Some(4)))), (40, true));
    }

    #[test]
    fn the_largest_chunk_of_image_rows_fits_one_frame() {
        let rows = super::super::alloc::MAX_PREFILL_CHUNK;
        let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: vec![1; rows], first_position: 0 }]);
        let c =
            PrefillImage { batch: b, want: Some(64), splice: vec![SpliceWire { dst_row: 0, n: rows as u64, data: Vec::new() }] };
        let len = c.encoded_len() + rows * ROW_BYTES;
        assert!(len <= MAX_IMAGE_BODY, "{len} bytes for {rows} rows");
        eprintln!("a {rows}-row chunk of image rows: {len} bytes of {MAX_IMAGE_BODY}");
    }

    #[test]
    fn the_images_are_checked_against_the_prompt_by_name() {
        let v = TpVision { source: Box::new(HostFeatures), image_token: Some(7), frame_budget: MAX_IMAGE_BODY };
        let p = [1, 7, 7, 7, 2];
        v.check_images(&p, &[span(1, 3, host(3, (1, 3), 0))]).unwrap();
        let e = v.check_images(&p, &[span(0, 3, host(3, (1, 3), 0))]).unwrap_err().to_string();
        assert!(e.contains("not all image placeholders"), "{e}");
        let e = v.check_images(&p, &[span(1, 3, host(2, (1, 2), 0))]).unwrap_err().to_string();
        assert!(e.contains("encoder rows"), "{e}");
        let e = v.check_images(&p, &[span(3, 3, host(3, (1, 3), 0))]).unwrap_err().to_string();
        assert!(e.contains("does not fit the prompt"), "{e}");
        let other: Arc<dyn Any + Send + Sync> = Arc::new(0u8);
        let e = v.check_images(&p, &[span(1, 3, other)]).unwrap_err().to_string();
        assert!(e.contains("not produced by this backend"), "{e}");
    }
}
