//! The vision tower's side of serving: what an encoded image is ([`Dsv41ImageFeatures`], host f32 rows — the
//! prefill splice reads them on the host and writes them into the chunk's embedding rows), the span a request's
//! image occupies in a prefill chunk ([`Dsv41Splice`]), the engram's DEAD mask over one, and the tower as the
//! server's image encoder ([`Dsv41Encoder`], `serve::ImageEncoder`). The tower itself, which knows nothing of
//! requests, is [`super::vision`].
//!
//! The features are host rows on purpose: the splice happens right after the embedding expansion
//! (`model.py`'s `merge_image_embeddings` before `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`), and the model's
//! residual stream is f32 `[rows, hc, hidden]` — so one contiguous host block a span builds ([`Dsv41Splice::block`])
//! is one `H2D` copy per span, with no strided device writes and no new kernel.

use super::vision::{self, SpanPlan, OUT_HIDDEN, PATCH_K};
use crate::error::ModelError;
use crate::serve::{EncodedImage, FeatureBudget, FeaturePermit, ImageEncoder};
use gb_cuda::dsv4_vision as dv;
use gb_cuda::{DevBuf, Stream};
use std::sync::Arc;

/// The three learned rows an image span's delimiters take (`image_start`, `image_end`, `image_newline`), on the
/// host in f32: the splice is a host-side write, so they are read once at boot rather than per prefill.
pub struct SpanEmbeddings {
    pub start: Vec<f32>,
    pub end: Vec<f32>,
    pub newline: Vec<f32>,
}

impl SpanEmbeddings {
    /// The three embeddings from device buffers holding the pack's bf16 rows (`image_start`, `image_end`,
    /// `image_newline`, each `[OUT_HIDDEN]`).
    pub fn from_device(start: &DevBuf, end: &DevBuf, newline: &DevBuf) -> Result<Self, ModelError> {
        let row = |b: &DevBuf, name: &str| -> Result<Vec<f32>, ModelError> {
            if b.bytes() < OUT_HIDDEN * 2 {
                return Err(ModelError::Config(format!("`{name}` is {} B, the span needs {}", b.bytes(), OUT_HIDDEN * 2)));
            }
            let bits: Vec<u16> = b.to_vec()?;
            Ok(bits.into_iter().take(OUT_HIDDEN).map(gb_cuda::num::bf16_to_f32).collect())
        };
        Ok(Self { start: row(start, "image_start")?, end: row(end, "image_end")?, newline: row(newline, "image_newline")? })
    }

    fn check(&self) -> Result<(), ModelError> {
        for (v, n) in [(&self.start, "image_start"), (&self.end, "image_end"), (&self.newline, "image_newline")] {
            if v.len() != OUT_HIDDEN {
                return Err(ModelError::Config(format!("`{n}` has {} values, the stream is {OUT_HIDDEN}", v.len())));
            }
        }
        Ok(())
    }
}

/// One encoded image: the aligner's output as host f32 `[tokens, OUT_HIDDEN]` rows in reading order of the merged
/// grid, and the budget permit that holds their bytes until the features are dropped.
pub struct Dsv41ImageFeatures {
    pub rows: Vec<f32>,
    pub tokens: usize,
    /// The merged grid (rows, columns); `tokens == merged.0 * merged.1`.
    pub merged: (usize, usize),
    /// The image's cache-key material (`sched::image_content_hash` of the pixel hash, the canvas and the
    /// preprocessing's id): what the engine's `ImageSpan::key` carries.
    pub key: u64,
    /// The share of the feature budget this image holds until it is dropped.
    pub permit: Option<FeaturePermit>,
}

impl std::fmt::Debug for Dsv41ImageFeatures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dsv41ImageFeatures {{ tokens: {}, merged: {:?} }}", self.tokens, self.merged)
    }
}

/// The preprocessing's identity as a key: FNV-1a of `ImagePreprocessor::id()`, only ever compared for equality.
pub fn preprocessing_key(id: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in id.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// The cache-key material of an image: its canvas (`pixel_hash`), the canvas size, and the preprocessing that made
/// it — the same ingredients the server keys the prefix cache with.
pub fn image_key(pixel_hash: u64, canvas: (usize, usize), preprocessing_key: u64) -> u64 {
    gb_sched::image_content_hash(pixel_hash, canvas.0 as u32, canvas.1 as u32, preprocessing_key)
}

/// One image's rows of a prefill chunk: the span positions `src..src + n` of `types` land on the chunk's rows
/// `dst..dst + n`, taking the aligner's rows (in reading order) and the three learned delimiter embeddings.
#[derive(Clone, Debug)]
pub struct Dsv41Splice {
    pub feats: Arc<Dsv41ImageFeatures>,
    /// The whole span's token types ([`SpanPlan::types`]), so a chunk that starts mid-span still knows which of its
    /// positions are delimiters and which are aligner rows.
    pub types: Vec<i8>,
    /// Index into `types` of this chunk's first span position.
    pub src: usize,
    /// Chunk-relative row that position lands on.
    pub dst: usize,
    /// Span positions in this chunk.
    pub n: usize,
}

impl Dsv41Splice {
    /// A splice for a whole span starting at absolute prompt row `start` (chunk row 0 of a full-span chunk).
    pub fn whole(feats: Arc<Dsv41ImageFeatures>, start: usize) -> Result<Self, ModelError> {
        let plan = SpanPlan::new(start, feats.merged.0 * dv::MERGE, feats.merged.1 * dv::MERGE)?;
        let types = plan.types();
        let slots = types.iter().filter(|&&t| t == dv::TokenType::Image as i8).count();
        if feats.tokens != slots {
            return Err(ModelError::Request(format!(
                "{} encoder rows for a {} x {} grid's {slots} IMAGE slots",
                feats.tokens, feats.merged.0, feats.merged.1
            )));
        }
        let n = types.len();
        Ok(Self { feats, types, src: 0, dst: start, n })
    }

    /// The rows of the span this chunk holds: chunk rows `dst..dst + n`, restricted to `at..end`.
    pub fn clip(&self, at: usize, end: usize) -> Option<Self> {
        let (lo, hi) = (self.dst.max(at), (self.dst + self.n).min(end));
        (lo < hi).then(|| Self {
            feats: Arc::clone(&self.feats),
            types: self.types.clone(),
            src: self.src + (lo - self.dst),
            dst: lo - at,
            n: hi - lo,
        })
    }

    /// The f32 block this splice writes over the chunk's embedding rows: `[n, hc, OUT_HIDDEN]`, each span row's
    /// value repeated across every hc stream (the embedding expansion's own repeat — `model.py` merges the image
    /// into `h` *before* `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`).
    pub fn block(&self, hc: usize, emb: &SpanEmbeddings) -> Result<Vec<f32>, ModelError> {
        emb.check()?;
        let d = OUT_HIDDEN;
        if self.feats.rows.len() != self.feats.tokens * d {
            return Err(ModelError::Request(format!(
                "{} feature values for {} rows of {d}",
                self.feats.rows.len(),
                self.feats.tokens
            )));
        }
        // how many aligner rows the span consumed before this chunk's first position
        let mut row = self.types[..self.src].iter().filter(|&&t| t == dv::TokenType::Image as i8).count();
        let mut out = vec![0f32; self.n * hc * d];
        for i in 0..self.n {
            let src: &[f32] = match dv::TokenType::from_i8(self.types[self.src + i]) {
                Some(dv::TokenType::ImageStart) => &emb.start,
                Some(dv::TokenType::ImageEnd) => &emb.end,
                Some(dv::TokenType::ImageNewLine) => &emb.newline,
                Some(dv::TokenType::Image) => {
                    let r = row;
                    row += 1;
                    &self.feats.rows[r * d..(r + 1) * d]
                }
                Some(dv::TokenType::Text) | None => {
                    return Err(ModelError::Request("a span position is not an image type".into()));
                }
            };
            for c in 0..hc {
                out[(i * hc + c) * d..(i * hc + c + 1) * d].copy_from_slice(src);
            }
        }
        Ok(out)
    }
}

/// The engram's view of a chunk that holds image spans: `ids` with every span position replaced by
/// [`super::engram::DEAD`] (as the i32 the gather reads), so no n-gram crosses an image and an image's positions
/// take no engram contribution (`model.py`'s `engram_mask = ~image_mask`).
pub fn dead_ids(ids: &[i32], splice: &[Dsv41Splice]) -> Vec<i32> {
    let mut out = ids.to_vec();
    for sp in splice {
        for r in sp.dst..sp.dst + sp.n {
            if r < out.len() {
                out[r] = super::engram::DEAD as i32;
            }
        }
    }
    out
}

/// The tower as the server's image encoder (`serve::ImageEncoder`): one image at a time, on the caller's thread
/// (the `VisionService` serializes encodes), over the tower's own stream. The rows come back to the host — the
/// prefill splice reads them there ([`Dsv41Splice::block`]).
pub struct Dsv41Encoder {
    tower: vision::Dsv4VisionTower,
    st: Stream,
    budget: Arc<FeatureBudget>,
    /// Hash of the preprocessing's `id()`: part of every image's cache key.
    preprocessing_key: u64,
    max_tokens: usize,
}

impl Dsv41Encoder {
    pub fn new(
        tower: vision::Dsv4VisionTower,
        budget: Arc<FeatureBudget>,
        preprocessing_id: &str,
        max_tokens: usize,
    ) -> Result<Self, ModelError> {
        Ok(Self { tower, st: Stream::new()?, budget, preprocessing_key: preprocessing_key(preprocessing_id), max_tokens })
    }

    /// The most LLM tokens one image may take (the `VisionService`'s own cap).
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

impl ImageEncoder for Dsv41Encoder {
    fn encode(&mut self, pre: &gb_vision::Preprocessed) -> Result<EncodedImage, ModelError> {
        let (gh, gw) = (pre.grid.1, pre.grid.2);
        let p = gh * gw;
        let merged = (dv::llm_grid(gh), dv::llm_grid(gw));
        let tokens = merged.0 * merged.1;
        if pre.grid.0 != 1 || p == 0 || p > self.tower.max_patches || pre.patch_dim != PATCH_K {
            return Err(ModelError::Request(format!(
                "image of {p} patches ({gh}x{gw}, patch width {}); the encoder takes still images of up to {} patches of {PATCH_K}",
                pre.patch_dim, self.tower.max_patches
            )));
        }
        if vision::num_image_tokens(merged.0, merged.1) > self.max_tokens {
            return Err(ModelError::Request(format!(
                "the {} x {} grid costs {} tokens, over the cap of {}",
                merged.0,
                merged.1,
                vision::num_image_tokens(merged.0, merged.1),
                self.max_tokens
            )));
        }
        let bytes = tokens * OUT_HIDDEN * 4;
        let permit = self.budget.acquire(bytes as u64).ok_or_else(|| {
            ModelError::Busy(format!(
                "the memory for image features is in use ({:.0} MiB free of {:.0} MiB, this image needs {:.0} MiB); retry shortly",
                self.budget.free() as f64 / 1048576.0,
                self.budget.total() as f64 / 1048576.0,
                bytes as f64 / 1048576.0
            ))
        })?;
        let mut out = DevBuf::alloc(bytes)?;
        // `REXL3_DSV41_DUMP_FEATS=<dir>`: a bring-up probe, not a serving path — the patches and the feature rows of
        // every encode land in <dir>/dump-<n>-{patches.u16,grid.txt,feats.f32} for a host reference to compare against.
        let dump = std::env::var_os("REXL3_DSV41_DUMP_FEATS").map(|dir| {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            std::fs::create_dir_all(&dir).ok();
            (dir, N.fetch_add(1, Ordering::Relaxed))
        });
        if let Some((dir, n)) = &dump {
            let p = std::path::Path::new(dir).join(format!("dump-{n}"));
            let bits: &[u16] = &pre.patches;
            let _ = std::fs::write(p.with_extension("patches.u16"), unsafe {
                core::slice::from_raw_parts(bits.as_ptr().cast::<u8>(), bits.len() * 2)
            });
            let _ = std::fs::write(p.with_extension("grid.txt"), format!("{gh} {gw}\n"));
        }
        self.tower.encode(&pre.patches, (gh, gw), &mut out, &self.st)?;
        self.st.sync()?;
        let rows: Vec<f32> = out.to_vec()?;
        if let Some((dir, n)) = &dump {
            let p = std::path::Path::new(dir).join(format!("dump-{n}"));
            let bytes: &[u8] = unsafe { core::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), rows.len() * 4) };
            let _ = std::fs::write(p.with_extension("feats.f32"), bytes);
        }
        if rows.len() != bytes / 4 {
            return Err(ModelError::Request(format!("{} feature values for {tokens} rows of {OUT_HIDDEN}", rows.len())));
        }
        Ok(EncodedImage {
            tokens,
            merged,
            features: Arc::new(Dsv41ImageFeatures {
                rows,
                tokens,
                merged,
                key: image_key(pre.pixel_hash, pre.size, self.preprocessing_key),
                permit: Some(permit),
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emb() -> SpanEmbeddings {
        SpanEmbeddings { start: vec![100f32; OUT_HIDDEN], end: vec![200f32; OUT_HIDDEN], newline: vec![300f32; OUT_HIDDEN] }
    }

    fn feats(rows: usize, cols: usize) -> Arc<Dsv41ImageFeatures> {
        let tokens = rows * cols;
        // row r is the constant r + 1, so a misplaced row is obvious
        Arc::new(Dsv41ImageFeatures {
            rows: (0..tokens).flat_map(|r| std::iter::repeat_n((r + 1) as f32, OUT_HIDDEN)).collect(),
            tokens,
            merged: (rows, cols),
            key: 0,
            permit: None,
        })
    }

    /// The splice block is the reference's span layout, repeated across every hc stream: `[START] ([IMAGE] * w
    /// [NEWLINE]) * h [END]`, the IMAGE slots in reading order.
    #[test]
    fn the_block_is_the_span_layout_repeated_over_the_streams() {
        let sp = Dsv41Splice::whole(feats(2, 3), 7).unwrap();
        assert_eq!(sp.types.len(), 2 * 4 + 2);
        let hc = 4;
        let b = sp.block(hc, &emb()).unwrap();
        let d = OUT_HIDDEN;
        assert_eq!(b.len(), sp.n * hc * d);
        let head = |i: usize, c: usize| b[(i * hc + c) * d];
        for c in 0..hc {
            assert_eq!(head(0, c), 100.0, "IMAGE_START, stream {c}");
            assert_eq!([head(1, c), head(2, c), head(3, c)], [1.0, 2.0, 3.0], "row 0, stream {c}");
            assert_eq!(head(4, c), 300.0, "NEWLINE, stream {c}");
            assert_eq!([head(5, c), head(6, c), head(7, c)], [4.0, 5.0, 6.0], "row 1, stream {c}");
            assert_eq!(head(8, c), 300.0);
            assert_eq!(head(9, c), 200.0, "IMAGE_END, stream {c}");
        }
    }

    /// A chunk that starts mid-span still knows which of its positions are delimiters: `clip` keeps the layout and
    /// the aligner rows continue in reading order.
    #[test]
    fn a_chunk_that_starts_mid_span_splices_from_there() {
        let whole = Dsv41Splice::whole(feats(2, 3), 2).unwrap();
        // the chunk holds prompt rows 4..8: span positions 2..5 — [A1 A2 NEWLINE A3], the aligner rows in order
        let cut = whole.clip(4, 8).unwrap();
        assert_eq!((cut.dst, cut.n), (0, 4));
        let b = cut.block(1, &emb()).unwrap();
        let d = OUT_HIDDEN;
        let head = |i: usize| b[i * d];
        assert_eq!([head(0), head(1), head(3)], [2.0, 3.0, 4.0], "aligner rows 1, 2, 3 in order");
        assert_eq!(head(2), 300.0, "the NEWLINE after row 0's third slot");
        // a chunk that misses the span entirely clips to nothing
        assert!(whole.clip(20, 30).is_none());
    }

    /// The engram's DEAD mask over a chunk: every span position dies, everything else keeps its id.
    #[test]
    fn the_engram_ids_of_a_chunk_with_an_image_are_dead_across_the_span() {
        let sp = Dsv41Splice::whole(feats(2, 2), 3).unwrap();
        let ids: Vec<i32> = (0..12i32).collect();
        let span_len = sp.n;
        let dead = dead_ids(&ids, &[sp]);
        assert_eq!(dead[..3], ids[..3], "the text before the span");
        assert_eq!(dead[3..3 + span_len], vec![super::super::engram::DEAD as i32; span_len], "the span");
        assert_eq!(dead[3 + span_len..], ids[3 + span_len..], "the text after");
        // and no ids at all change without a span
        assert_eq!(dead_ids(&ids, &[]), ids);
    }

    /// The features' key changes with the pixels, the canvas and the preprocessing.
    #[test]
    fn an_images_key_changes_with_its_pixels_its_size_and_its_preprocessing() {
        let k = preprocessing_key("dsv4.1-vision-v1:p14m3:px295936:tok1000:rationone");
        let base = image_key(0x1234, (812, 602), k);
        assert_eq!(base, image_key(0x1234, (812, 602), k), "a key is a function of its inputs");
        assert_ne!(base, image_key(0x1235, (812, 602), k), "different pixels");
        assert_ne!(base, image_key(0x1234, (602, 812), k), "another shape");
        assert_ne!(
            base,
            image_key(0x1234, (812, 602), preprocessing_key("dsv4.1-vision-v1:p14m3:px295936:tok512:rationone")),
            "another budget"
        );
        assert_eq!(base, gb_sched::image_content_hash(0x1234, 812, 602, k), "it is the scheduler's image hash");
    }

    /// The features' row count is checked against the grid, and a whole-splice of the wrong shape is refused.
    #[test]
    fn a_whole_splice_of_the_wrong_shape_is_refused() {
        let f = feats(2, 2);
        assert!(Dsv41Splice::whole(f.clone(), 0).is_ok());
        // claim a 3 x 3 merged grid over 2 x 2 rows of features: 9 IMAGE slots, 4 rows
        let mut wrong = feats(2, 2);
        Arc::get_mut(&mut wrong).unwrap().merged = (3, 3);
        let e = Dsv41Splice::whole(wrong, 0).unwrap_err().to_string();
        assert!(e.contains("encoder rows"), "{e}");
    }
}
