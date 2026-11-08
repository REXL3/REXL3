//! The splice of image and video features into a MiniCPM-V-4.7 prefill (E5b of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`), host side only:
//! what a prompt with spans means for the forward, from the ids and the spans alone, with no device and no weights.
//!
//! **What a span is.** One [`gb_engine::ImageSpan`] per frame (a picture, or one frame of a clip): `start` is the first token of the
//! frame (its `<image>` marker; the `<image_id>i</image_id>` prefix is plain text before it), `tokens` runs through the last `</slice>`
//! (or `</image>`) and the chat template's trailing `\n`s, which the reference model absorbs into the frame, and `merged == (1, tokens)`.
//! The span's feature rows are the projector's output of the frame's crops, crop-major (the thumbnail first, then the slices
//! row-major), and fill the PAD tokens of the span in order; the markers inside the span stay ordinary embedding rows. So the number
//! of pad tokens of a span equals its features' rows, and the engine's own geometry checks (`tokens == merged.0 * merged.1`, spans
//! ascending and inside the prompt) see a span of `tokens` tokens, which is what a prefix cut must not split.
//!
//! **What the forward is given** ([`Geometry`]):
//! - rows to splice: for a chunk `[a, b)` the maximal runs of pad tokens as `(dst row in the chunk, src row of the span's features,
//!   n)`; the device copies them over the embedded chunk before the first norm, where `masked_scatter` of the reference sits;
//! - positions: [`crate::mrope_canvas::canvas_positions`] over the WHOLE prompt (the canvas of a frame needs every slice of it, and a
//!   later frame starts where the earlier ones ended), as explicit `(T, H, W)` triples for every chunk that touches a frame token, and
//!   as a plain text offset for a chunk that does not (`index + offset` on all three streams: the slot's rope delta);
//! - the decode offset: `rope_delta` of the canvas, set on the slot when the prompt completes (`index + delta` on every stream);
//! - the chunk grid: the backend's absolute grid, but a chunk never ends strictly inside a visual group while the group fits a chunk.
//!   (A group longer than the chunk, a long clip, is cut on the grid: the triples come from the whole prompt, so the bits do not
//!   depend on where the cut falls, and only a PREFIX-CACHE cut cares, which is [`Geometry::cut_outside_frames`]'s.)

use super::vision::MiniCpmFeatures;
use crate::error::ModelError;
use crate::mrope_canvas::{canvas_positions, CanvasInput, CanvasPositions, Downsample, Markers};
use crate::serve::feature_ref;
use gb_engine::ImageSpan;
use std::any::Any;

/// `<|image_pad|>`: the token an image's feature rows replace.
pub const IMAGE_PAD: u32 = 248_056;
/// `<|video_pad|>`: the token a video frame's feature rows replace.
pub const VIDEO_PAD: u32 = 248_057;

/// What the splice needs to know of one span's features, without the device buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameLayout {
    /// Feature rows = pad tokens of the span.
    pub rows: usize,
    /// The patch grid `(h, w)` of every crop, thumbnail first (the processor's `target_sizes`).
    pub crops: Vec<[u32; 2]>,
    pub downsample: Downsample,
    /// `(rows, columns)` of the slice grid; `(0, 0)` for a thumbnail alone.
    pub slice_grid: (usize, usize),
}

/// Features that exist on the host only: what CPU tests and a fake runtime put in an `ImageSpan` instead of device rows.
#[derive(Clone, Debug)]
pub struct HostFeatures(pub FrameLayout);

/// The layout behind a span's features: the tower's own [`MiniCpmFeatures`] (as the encoder returns them, leased or not) or
/// [`HostFeatures`]. `None` for anything else (another family's features, a unit).
pub fn layout_of(features: &(dyn Any + Send + Sync)) -> Option<FrameLayout> {
    let any: &dyn Any = features;
    if let Some(h) = any.downcast_ref::<HostFeatures>() {
        return Some(h.0.clone());
    }
    let f = feature_ref::<MiniCpmFeatures>(any)?;
    let down = match f.merge_factor()? {
        4 => Downsample::X16,
        2 => Downsample::X4,
        _ => return None,
    };
    Some(FrameLayout {
        rows: f.tokens,
        crops: f.crops.iter().map(|c| [c.patches.0 as u32, c.patches.1 as u32]).collect(),
        downsample: down,
        slice_grid: f.slice_grid,
    })
}

/// One run of pad tokens of a chunk and the feature rows that replace them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowSplice {
    /// Index into the request's spans.
    pub span: usize,
    /// First row of the chunk.
    pub dst_row: usize,
    /// First row of the span's features.
    pub src_row: usize,
    pub n: usize,
}

#[derive(Clone, Debug)]
struct SpanInfo {
    start: usize,
    end: usize,
    pad: u32,
}

/// A prompt with spans, laid out: the canvas positions of every token and where each span's pad tokens are.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub canvas: CanvasPositions,
    spans: Vec<SpanInfo>,
}

fn request(m: impl Into<String>) -> ModelError {
    ModelError::Request(m.into())
}

impl Geometry {
    /// The geometry of `prompt` with `images`. Refuses by name a span whose features are not MiniCPM-V-4.7 rows, whose pad tokens do not
    /// number its feature rows, spans that overlap or leave the prompt, and a prompt whose crop runs the canvas rule cannot place.
    pub fn new(prompt: &[u32], images: &[ImageSpan]) -> Result<Geometry, ModelError> {
        let mut labels = vec![0u8; prompt.len()];
        let (mut image_sizes, mut video_sizes) = (Vec::new(), Vec::new());
        let mut down: Option<Downsample> = None;
        let mut spans = Vec::with_capacity(images.len());
        let mut at = 0usize;
        for (i, im) in images.iter().enumerate() {
            let end = im.start.checked_add(im.tokens).filter(|&e| im.tokens > 0 && e <= prompt.len()).ok_or_else(|| {
                request(format!("image {i}: span {} + {} leaves the prompt of {}", im.start, im.tokens, prompt.len()))
            })?;
            if im.start < at {
                return Err(request(format!("image {i}: span at {} overlaps the one before it (ends at {at})", im.start)));
            }
            at = end;
            let layout = layout_of(&*im.features).ok_or_else(|| {
                request(format!("image {i}: the span's features are not MiniCPM-V-4.7 projector rows (vision_minicpmv)"))
            })?;
            let mut pad = None;
            let mut n = 0usize;
            for &t in &prompt[im.start..end] {
                if t == IMAGE_PAD || t == VIDEO_PAD {
                    if pad.is_some_and(|p| p != t) {
                        return Err(request(format!("image {i}: the span mixes image and video pad tokens")));
                    }
                    pad = Some(t);
                    n += 1;
                }
            }
            let pad = pad.ok_or_else(|| request(format!("image {i}: the span holds no pad token")))?;
            if n != layout.rows {
                return Err(request(format!(
                    "image {i}: the span holds {n} pad tokens but its features have {} rows",
                    layout.rows
                )));
            }
            if down.is_some_and(|d| d != layout.downsample) {
                return Err(request(format!("image {i}: downsample modes differ inside one prompt")));
            }
            down = Some(layout.downsample);
            let label = if pad == IMAGE_PAD { 1 } else { 2 };
            for (p, &t) in prompt[im.start..end].iter().enumerate() {
                if t == pad {
                    labels[im.start + p] = label;
                }
            }
            if pad == IMAGE_PAD {
                image_sizes.extend(layout.crops.iter().copied());
            } else {
                video_sizes.extend(layout.crops.iter().copied());
            }
            spans.push(SpanInfo { start: im.start, end, pad });
        }
        let canvas = canvas_positions(&CanvasInput {
            ids: prompt,
            mm_token_type_ids: &labels,
            image_sizes: &image_sizes,
            video_sizes: &video_sizes,
            downsample: down.unwrap_or(Downsample::X16),
            markers: Markers::MINICPMV4_7,
        })
        .map_err(|e| request(e.to_string()))?;
        if canvas.image_crops != image_sizes.len() || canvas.video_crops != video_sizes.len() {
            return Err(request(format!(
                "the prompt lays out {} image and {} video crops but the features hold {} and {}",
                canvas.image_crops,
                canvas.video_crops,
                image_sizes.len(),
                video_sizes.len()
            )));
        }
        Ok(Geometry { canvas, spans })
    }

    /// Spans of the request.
    pub fn n_spans(&self) -> usize {
        self.spans.len()
    }

    /// `index + rope_delta` is the rope position of every token generated after the prompt, on all three streams.
    pub fn rope_delta(&self) -> i32 {
        self.canvas.rope_delta
    }

    /// Where a chunk that starts at `done` ends: the next boundary of the absolute grid of `chunk` rows, moved to the start of the
    /// visual group it would split when `done` is before it, and carried to the group's end when the group fits `max_rows` and the
    /// chunk starts at (or inside) it. A group longer than `max_rows` is cut on the grid.
    pub fn chunk_end(&self, done: usize, total: usize, chunk: usize, max_rows: usize) -> usize {
        debug_assert!(chunk > 0 && done < total);
        let c = done.saturating_add(chunk - done % chunk).min(total);
        match self.canvas.groups.iter().find(|g| g.start < c && c < g.end) {
            None => c,
            Some(g) if g.start > done => g.start,
            Some(g) => {
                let whole = g.end.min(total);
                if whole - done <= max_rows {
                    whole
                } else {
                    c
                }
            }
        }
    }

    /// Whether chunk `[a, b)` must carry explicit triples (it touches a frame token).
    pub fn is_explicit(&self, a: usize, b: usize) -> bool {
        self.canvas.chunk_is_explicit(a..b)
    }

    /// The `(T, H, W)` triples of chunk `[a, b)`, row-major `[rows, 3]`.
    pub fn rope3(&self, a: usize, b: usize) -> Vec<i32> {
        self.canvas.positions[a..b].iter().flatten().copied().collect()
    }

    /// For a chunk with no frame token: `rope position - index` of every row (constant over a run of text).
    pub fn text_offset(&self, a: usize) -> i32 {
        self.canvas.positions[a][0] - a as i32
    }

    /// The largest cut at or below `c` that no visual group straddles (a prefix of the prompt that ends there has the positions the
    /// whole prompt gives its tokens).
    pub fn cut_outside_frames(&self, c: usize) -> usize {
        self.canvas.cut_outside_frames(c)
    }

    /// The pad runs of chunk `[a, b)` of `prompt` and the feature rows that replace them.
    pub fn splice(&self, prompt: &[u32], a: usize, b: usize) -> Vec<RowSplice> {
        let mut out = Vec::new();
        for (k, s) in self.spans.iter().enumerate() {
            if s.end <= a || s.start >= b {
                continue;
            }
            let (lo, hi) = (a.max(s.start), b.min(s.end));
            let mut ordinal = prompt[s.start..lo].iter().filter(|&&t| t == s.pad).count();
            let mut run: Option<RowSplice> = None;
            for (p, &t) in prompt.iter().enumerate().take(hi).skip(lo) {
                if t == s.pad {
                    match &mut run {
                        Some(r) => r.n += 1,
                        None => run = Some(RowSplice { span: k, dst_row: p - a, src_row: ordinal, n: 1 }),
                    }
                    ordinal += 1;
                } else if let Some(r) = run.take() {
                    out.push(r);
                }
            }
            out.extend(run);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    const IMG_OPEN: u32 = 248_078;
    const IMG_CLOSE: u32 = 248_079;
    const SLICE_OPEN: u32 = 248_088;
    const SLICE_CLOSE: u32 = 248_089;
    const NL: u32 = 198;
    const TEXT: u32 = 1000;

    /// A frame: `<image>` pad x thumb `</image>`, then rows x cols slices of `per` pads, rows joined by a newline.
    fn frame(pad: u32, thumb: usize, grid: (usize, usize), per: usize) -> Vec<u32> {
        let mut v = vec![IMG_OPEN];
        v.extend(std::iter::repeat_n(pad, thumb));
        v.push(IMG_CLOSE);
        for r in 0..grid.0 {
            if r > 0 {
                v.push(NL);
            }
            for _ in 0..grid.1 {
                v.push(SLICE_OPEN);
                v.extend(std::iter::repeat_n(pad, per));
                v.push(SLICE_CLOSE);
            }
        }
        v
    }

    /// The host features of a frame of 448-pixel crops at 16x: a crop is 32 x 32 patches and 64 tokens.
    fn features(grid: (usize, usize)) -> Arc<dyn Any + Send + Sync> {
        let n = 1 + grid.0 * grid.1;
        Arc::new(HostFeatures(FrameLayout {
            rows: n * 64,
            crops: vec![[32, 32]; n],
            downsample: Downsample::X16,
            slice_grid: grid,
        }))
    }

    fn span(start: usize, tokens: usize, grid: (usize, usize)) -> ImageSpan {
        ImageSpan { start, tokens, merged: (1, tokens), features: features(grid), key: 7 }
    }

    /// text, a 2 x 2 image, newline, text, a thumbnail-only image, text
    fn prompt() -> (Vec<u32>, Vec<ImageSpan>) {
        let mut p = vec![TEXT; 5];
        let a = frame(IMAGE_PAD, 64, (2, 2), 64);
        let s0 = p.len();
        p.extend(&a);
        p.push(NL);
        let len0 = a.len() + 1;
        p.extend(vec![TEXT + 1; 7]);
        let b = frame(IMAGE_PAD, 64, (0, 0), 0);
        let s1 = p.len();
        p.extend(&b);
        let len1 = b.len();
        p.extend(vec![TEXT + 2; 4]);
        (p, vec![span(s0, len0, (2, 2)), span(s1, len1, (0, 0))])
    }

    #[test]
    fn pad_runs_are_spliced_in_feature_order_and_markers_stay_text() {
        let (p, images) = prompt();
        let g = Geometry::new(&p, &images).unwrap();
        // the whole prompt as one chunk: thumbnail (64 rows), then four slices (64 each) of image 0, then the thumbnail of image 1
        let ops = g.splice(&p, 0, p.len());
        assert_eq!(ops.len(), 5 + 1);
        let mut src = 0;
        for o in &ops[..5] {
            assert_eq!((o.span, o.src_row, o.n), (0, src, 64));
            assert!(p[o.dst_row..o.dst_row + 64].iter().all(|&t| t == IMAGE_PAD));
            assert!(o.dst_row == 0 || p[o.dst_row - 1] != IMAGE_PAD, "a run is maximal");
            src += 64;
        }
        assert_eq!((ops[5].span, ops[5].src_row, ops[5].n), (1, 0, 64));
        // every pad token is covered exactly once, and nothing else
        let covered: usize = ops.iter().map(|o| o.n).sum();
        assert_eq!(covered, p.iter().filter(|&&t| t == IMAGE_PAD).count());
        // a chunk that starts in the middle of a run takes the rest of that run with the right source row
        let mid = ops[1].dst_row + 10;
        let part = g.splice(&p, mid, mid + 20);
        assert_eq!(part, vec![RowSplice { span: 0, dst_row: 0, src_row: 64 + 10, n: 20 }]);
        // a chunk of text outside every span splices nothing
        assert!(g.splice(&p, 0, 5).is_empty());
    }

    #[test]
    fn positions_cover_the_whole_prompt_and_a_text_chunk_is_a_plain_offset() {
        let (p, images) = prompt();
        let g = Geometry::new(&p, &images).unwrap();
        assert_eq!(g.canvas.len(), p.len());
        // text before the first frame is 1-D from 0
        assert!(!g.is_explicit(0, 5));
        assert_eq!(g.text_offset(0), 0);
        assert_eq!(g.rope3(0, 3), vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);
        // a chunk that touches a frame is explicit
        assert!(g.is_explicit(3, 8) && g.is_explicit(0, p.len()));
        // the text between the images resumes after the first frame's canvas, so it is offset by a frame's worth
        let s1 = images[0].start + images[0].tokens;
        assert!(!g.is_explicit(s1, s1 + 7));
        assert!(g.text_offset(s1) < 0, "the canvas of a frame takes fewer positions than tokens: {}", g.text_offset(s1));
        // decode continues at index + delta from the last token
        assert_eq!(g.rope_delta(), g.canvas.rope_delta);
        assert_eq!(g.canvas.next_position(), p.len() as i32 + g.rope_delta());
    }

    #[test]
    fn a_chunk_never_ends_inside_a_group_that_fits_it_at_any_chunk_size() {
        let (p, images) = prompt();
        let g = Geometry::new(&p, &images).unwrap();
        let longest = g.canvas.groups.iter().map(|r| r.len()).max().unwrap();
        for chunk in 1..=p.len() + 3 {
            // the largest chunk the workspace takes is the grid itself: a group longer than it is cut on the grid
            let max_rows = chunk;
            let mut done = 0;
            let mut ends = Vec::new();
            while done < p.len() {
                let end = g.chunk_end(done, p.len(), chunk, max_rows);
                assert!(end > done && end <= p.len(), "progress at {done} with chunk {chunk}");
                assert!(end - done <= max_rows.max(1) || g.canvas.groups.iter().any(|r| r.start <= done && end <= r.end));
                ends.push(end);
                done = end;
            }
            for &e in &ends {
                for r in &g.canvas.groups {
                    if r.start < e && e < r.end {
                        assert!(r.len() > max_rows, "chunk {chunk}: a cut at {e} splits a group of {} that fits", r.len());
                    }
                }
            }
            if chunk >= longest {
                assert!(ends.iter().all(|&e| g.canvas.is_cut_outside_frames(e)), "chunk {chunk} >= longest group {longest}");
            }
        }
    }

    #[test]
    fn a_group_longer_than_the_chunk_is_cut_on_the_grid_and_the_walk_still_ends() {
        let (p, images) = prompt();
        let g = Geometry::new(&p, &images).unwrap();
        let big = &g.canvas.groups[0];
        let chunk = 100;
        assert!(big.len() > chunk);
        let mut done = 0;
        let mut ends = Vec::new();
        while done < p.len() {
            done = g.chunk_end(done, p.len(), chunk, chunk);
            ends.push(done);
        }
        // the chunk before the group stops where it starts, the group is cut on the absolute grid, and the walk reaches the end
        assert!(ends.contains(&big.start), "{ends:?}");
        let inside: Vec<usize> = ends.iter().copied().filter(|&e| big.start < e && e < big.end).collect();
        assert!(inside.len() >= 2 && inside.iter().all(|e| e % chunk == 0), "{inside:?}");
        assert_eq!(*ends.last().unwrap(), p.len());
    }

    #[test]
    fn prefix_cuts_move_to_the_start_of_the_group_they_would_split() {
        let (p, images) = prompt();
        let g = Geometry::new(&p, &images).unwrap();
        let big = g.canvas.groups[0].clone();
        assert_eq!(g.cut_outside_frames(big.start + 30), big.start);
        assert_eq!(g.cut_outside_frames(big.start), big.start);
        assert_eq!(g.cut_outside_frames(big.end), big.end);
        assert_eq!(g.cut_outside_frames(3), 3);
    }

    #[test]
    fn a_span_that_does_not_match_its_features_is_refused_by_name() {
        let (p, images) = prompt();
        // features with one row too many
        let mut bad = images.clone();
        bad[1].features = Arc::new(HostFeatures(FrameLayout {
            rows: 65,
            crops: vec![[32, 32]],
            downsample: Downsample::X16,
            slice_grid: (0, 0),
        }));
        let e = Geometry::new(&p, &bad).unwrap_err().to_string();
        assert!(e.contains("image 1") && e.contains("65 rows"), "{e}");
        // features of another kind
        let mut alien = images.clone();
        alien[0].features = Arc::new(());
        assert!(Geometry::new(&p, &alien).unwrap_err().to_string().contains("not MiniCPM-V-4.7"));
        // a span past the prompt, and overlapping spans
        let mut far = images.clone();
        far[1].tokens += 100;
        assert!(Geometry::new(&p, &far).unwrap_err().to_string().contains("leaves the prompt"));
        let mut over = images.clone();
        over[1].start = images[0].start + 10;
        assert!(Geometry::new(&p, &over).unwrap_err().to_string().contains("overlaps"));
        // a crop whose patch grid does not make the run of pads in the prompt
        let mut wrong = images.clone();
        wrong[0].features = Arc::new(HostFeatures(FrameLayout {
            rows: 5 * 64,
            crops: vec![[32, 32], [32, 32], [32, 32], [32, 32], [16, 16]],
            downsample: Downsample::X16,
            slice_grid: (2, 2),
        }));
        let e = Geometry::new(&p, &wrong).unwrap_err().to_string();
        assert!(e.contains("canvas rope"), "{e}");
    }

    #[test]
    fn video_frames_are_their_own_spans_and_one_group() {
        // three adjacent frames of a clip (video pad, no slices), text after
        let mut p = vec![TEXT; 4];
        let mut images = Vec::new();
        for _ in 0..3 {
            let f = frame(VIDEO_PAD, 64, (0, 0), 0);
            images.push(span(p.len(), f.len(), (0, 0)));
            p.extend(f);
        }
        p.extend(vec![TEXT; 6]);
        let g = Geometry::new(&p, &images).unwrap();
        assert_eq!(g.canvas.groups.len(), 1, "markers alone separate the frames: one group");
        assert_eq!(g.canvas.video_crops, 3);
        let ops = g.splice(&p, 0, p.len());
        assert_eq!(ops.iter().map(|o| (o.span, o.src_row, o.n)).collect::<Vec<_>>(), vec![(0, 0, 64), (1, 0, 64), (2, 0, 64)]);
        // a prefix cut between two frames of the clip is inside the group
        let between = images[1].start;
        assert_eq!(g.cut_outside_frames(between), images[0].start);
    }
}
