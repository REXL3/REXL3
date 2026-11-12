//! Canvas M-RoPE positions of MiniCPM-V 4.7 (`mrope_mode = "canvas"`): a CPU port of Hugging Face `MiniCPMV4_7Model.get_rope_index`
//! (`_frame_end_idx`, `_group_visual_frames`, `get_vision_position_ids`) for one unpadded sequence.
//!
//! **Attribution.** This is a re-expression in Rust of the position logic of `transformers/models/minicpmv4_7/modular_minicpmv4_7.py` /
//! `modeling_minicpmv4_7.py` (transformers 5.18.0, Apache-2.0, Copyright 2026 OpenBMB and the HuggingFace Inc. team); the rows of
//! `THIRD_PARTY_NOTICES.md` and `docs/UPSTREAMS.md` name it. The fixtures under `testdata/minicpmv4_7/rope` are the output of that code
//! (`tools/gen_minicpmv_rope_fixtures.py`); the structure, the types, the errors and the frame-span API are REXL3's own.
//!
//! # What the canvas is
//!
//! Text is plain 1-D: all three streams (T, H, W) advance together. A *frame* (one picture, one video frame) is a canvas of
//! `rows * slice_h` by `cols * slice_w` tokens (or the thumbnail's own grid when there are no slices): T is frozen at the canvas
//! origin, H and W are `origin + row` and `origin + column`. The thumbnail (the first crop, wrapped in `<image>` ... `</image>`) is
//! stretched over the whole canvas with `linspace(0, canvas - 1, n).round()`; every slice sits at its place in the canvas, `<image>`
//! one step before the origin (the "halo"), `</image>` at the far corner, `<slice>` at its slice's top-left corner, `</slice>` at its
//! bottom-right corner, the `\n` that ends a row of slices just past the right edge, and the next origin is
//! `origin + max(canvas_h, canvas_w) + 1`. Consecutive frames that only markers separate (a video clip) form a *group*: the text
//! between two of them (there is none in practice) is 1-D and the next frame starts one step further.
//!
//! # Why a prefix cut or a chunk must respect frames
//!
//! The position of a token inside a group is a function of the whole group: the canvas size needs every slice (the thumbnail is
//! stretched over it), the slice grid needs the newlines, and the group's end decides where the text after it resumes (`origin +
//! max(canvas) + 1`). K is cached with its rotary position applied, so a cached prefix is reusable only if its tokens' positions are the
//! ones the full prompt gives them. A cut strictly inside a group would leave the tokens before it with positions computed for a
//! truncated canvas that the full prompt never produces. Hence [`CanvasPositions::cut_outside_frames`] (cuts), and
//! [`CanvasPositions::chunk_is_explicit`] (a chunk that touches any frame token, markers and the absorbed trailing `\n` included, must
//! be fed explicit `(T, H, W)` rows; a chunk of text outside every frame is `index + offset` on all three streams).
//!
//! # Decode and scope
//!
//! `rope_delta = max(position) + 1 - len`. Every token the model generates afterwards (cache index `p >= len`) is at `[p + rope_delta; 3]`
//! on all three streams: HF's `compute_3d_position_ids` adds the stored delta to the 1-D arange of the cache positions. The first
//! generated token is therefore at [`CanvasPositions::next_position`]. The port is for one unpadded sequence (the engine has no
//! padding); HF's `attention_mask` path, which lays the canvas over the kept tokens of a left-padded row, is not ported. Audio (label 3)
//! has no canvas and is refused.
//!
//! # Quirks mirrored on purpose
//!
//! - After a frame's last crop every consecutive marker token (`<slice>`, `</slice>`, `<image>`, `</image>`, `\n`) is part of the frame
//!   (`_frame_end_idx`). In a real prompt the `\n` that the chat template puts between an image and the following text is token 198: it
//!   is absorbed, and sits at `(origin, origin, origin)`; the text after it resumes at `origin + max(canvas) + 1`.
//! - The slice grid is read from the gaps: the first gap wider than two tokens (`</slice>` `\n` `<slice>`) fixes the column count; when
//!   `rows * cols != slices` the layout falls back to `(1, slices)`. Any token strictly inside a gap wider than two is pinned like a newline.
//! - A crop preceded by `<slice>` attaches to the most recent frame as one of its slices, wherever it is.
//! - `halo = max(origin - 1, 0)`.
//! - A size mismatch between a crop's run and its grid is an error here (HF would broadcast a 1-token grid or raise inside torch).

use std::fmt;
use std::ops::Range;

/// Token ids of the five markers the canvas layout reads (`config.json`: `image_start_id`, `image_end_id`, `slice_start_id`,
/// `slice_end_id`, `newline_id`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Markers {
    pub image_start: u32,
    pub image_end: u32,
    pub slice_start: u32,
    pub slice_end: u32,
    pub newline: u32,
}

impl Markers {
    /// The ids of the MiniCPM-V-4.7-35B-A3B checkpoint.
    pub const MINICPMV4_7: Markers =
        Markers { image_start: 248_078, image_end: 248_079, slice_start: 248_088, slice_end: 248_089, newline: 198 };

    fn is_marker(&self, t: u32) -> bool {
        t == self.slice_start || t == self.slice_end || t == self.image_start || t == self.image_end || t == self.newline
    }
}

/// `downsample_mode`: how many patches one LM token covers per side (4 for `16x`: the window merger 2x2 then the projector 2x2; 2 for
/// `4x`, which skips the window merger).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Downsample {
    X16,
    X4,
}

impl Downsample {
    /// HF `merge_factor`: 2 for `"4x"`, 4 otherwise.
    pub fn merge_factor(self) -> usize {
        match self {
            Downsample::X16 => 4,
            Downsample::X4 => 2,
        }
    }

    /// `"16x"` or `"4x"` (anything else is `None`; HF treats every value but `"4x"` as 16x, the config validates it).
    pub fn parse(mode: &str) -> Option<Downsample> {
        match mode {
            "16x" => Some(Downsample::X16),
            "4x" => Some(Downsample::X4),
            _ => None,
        }
    }
}

/// The modality of a crop (`mm_token_type_ids`: 0 text, 1 image, 2 video).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Modality {
    Image,
    Video,
}

impl Modality {
    pub fn label(self) -> u8 {
        match self {
            Modality::Image => 1,
            Modality::Video => 2,
        }
    }
}

/// The plain inputs of one sequence. `image_sizes` and `video_sizes` are the patch grids `[h, w]` (processor `target_sizes`) of every
/// crop in the order the crops appear in the prompt: for each frame the thumbnail first, then its slices row-major.
#[derive(Clone, Copy, Debug)]
pub struct CanvasInput<'a> {
    pub ids: &'a [u32],
    pub mm_token_type_ids: &'a [u8],
    pub image_sizes: &'a [[u32; 2]],
    pub video_sizes: &'a [[u32; 2]],
    pub downsample: Downsample,
    pub markers: Markers,
}

/// Why the canvas could not be laid out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanvasError {
    /// `ids` and `mm_token_type_ids` differ in length.
    LengthMismatch { ids: usize, mm_token_type_ids: usize },
    /// No token (HF's `amax` of an empty tensor raises).
    EmptyPrompt,
    /// The sequence is too long for 32-bit positions.
    TooLong { len: usize },
    /// A modality label other than 0, 1 and 2 (3 is audio, which has no canvas).
    UnsupportedModality { index: usize, label: u8 },
    /// A crop at index 0 has no marker in front of it, or two crops touch (the marker position would be inside a crop).
    CropWithoutMarker { start: usize },
    /// The prompt has more crops of a modality than target sizes were given (HF: `StopIteration`).
    TargetSizesExhausted { modality: Modality, crop: usize, have: usize },
    /// A crop's placeholder run is not `(h / merge) * (w / merge)` tokens long.
    RunLengthMismatch { start: usize, len: usize, grid: [u32; 2], expected: usize },
}

impl fmt::Display for CanvasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CanvasError::LengthMismatch { ids, mm_token_type_ids } => {
                write!(f, "canvas rope: {ids} token ids but {mm_token_type_ids} modality labels")
            }
            CanvasError::EmptyPrompt => write!(f, "canvas rope: empty prompt"),
            CanvasError::TooLong { len } => write!(f, "canvas rope: {len} tokens do not fit 32-bit positions"),
            CanvasError::UnsupportedModality { index, label } => {
                write!(
                    f,
                    "canvas rope: token {index} has modality label {label}; only 0 (text), 1 (image) and 2 (video) have a canvas"
                )
            }
            CanvasError::CropWithoutMarker { start } => {
                write!(f, "canvas rope: the crop starting at token {start} has no marker token in front of it")
            }
            CanvasError::TargetSizesExhausted { modality, crop, have } => {
                write!(f, "canvas rope: crop {crop} of the {modality:?} modality has no target size (only {have} were given)")
            }
            CanvasError::RunLengthMismatch { start, len, grid, expected } => write!(
                f,
                "canvas rope: the crop at token {start} is {len} tokens long but its patch grid {}x{} makes {expected}",
                grid[0], grid[1]
            ),
        }
    }
}

impl std::error::Error for CanvasError {}

/// One frame (a picture, or one frame of a clip) as the layout sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanvasFrame {
    pub modality: Modality,
    /// First token (the `<image>` marker in front of the thumbnail) up to and excluding the first token after the markers that close
    /// the frame (the trailing `\n` of the chat template included).
    pub span: Range<usize>,
    /// The thumbnail's tokens.
    pub thumbnail: Range<usize>,
    /// The slices' tokens, row-major.
    pub slices: Vec<Range<usize>>,
    /// `(rows, cols)` of the slice grid; `(0, 0)` without slices.
    pub grid: (usize, usize),
    /// The canvas `(height, width)` in tokens.
    pub canvas: (i64, i64),
    /// The canvas origin: T of every token of the frame.
    pub origin: i64,
}

/// The positions of one prompt and where its frames are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanvasPositions {
    /// `(T, H, W)` of every token.
    pub positions: Vec<[i32; 3]>,
    /// `max(position) + 1 - len`: the offset of the rope position over the cache index for every token generated after the prompt
    /// (HF adds it to the 1-D arange on all three streams: token `p` of the cache is at `[p + delta; 3]`).
    pub rope_delta: i32,
    /// The visual groups: runs of frames that only markers separate, markers included.
    pub groups: Vec<Range<usize>>,
    /// Every frame, in order.
    pub frames: Vec<CanvasFrame>,
    /// Crops consumed from `image_sizes` / `video_sizes` (a caller that wants HF's silence about leftovers ignores them).
    pub image_crops: usize,
    pub video_crops: usize,
    /// `frame_prefix[i]` = number of frame tokens among the first `i` tokens.
    frame_prefix: Vec<u32>,
}

impl CanvasPositions {
    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Whether token `i` is inside a frame: a crop token or one of the markers that open, separate or close a frame (`<image>`,
    /// `</image>`, `<slice>`, `</slice>`, the `\n` between rows of slices, and the markers absorbed after the last crop).
    pub fn is_frame_token(&self, i: usize) -> bool {
        i < self.len() && self.frame_prefix[i + 1] > self.frame_prefix[i]
    }

    /// The per-token flag set.
    pub fn frame_tokens(&self) -> Vec<bool> {
        (0..self.len()).map(|i| self.is_frame_token(i)).collect()
    }

    /// A chunk `[start, end)` of the prompt is **explicit** when it touches a frame token: its rope rows must be the explicit
    /// `(T, H, W)` of [`CanvasPositions::positions`]. Any other chunk is plain text, at `index + offset` on all three streams where the
    /// offset is `positions[i][0] - i` (constant over the chunk), whatever its length.
    pub fn chunk_is_explicit(&self, chunk: Range<usize>) -> bool {
        let (s, e) = (chunk.start.min(self.len()), chunk.end.min(self.len()));
        s < e && self.frame_prefix[e] > self.frame_prefix[s]
    }

    /// Whether a prefix of `c` tokens may be cut: no group has tokens on both sides of the cut. A cut at a group's first or last token
    /// is fine (the group is whole on one side).
    pub fn is_cut_outside_frames(&self, c: usize) -> bool {
        !self.groups.iter().any(|g| g.start < c && c < g.end)
    }

    /// The largest valid cut at or below `c`: `c` itself, or the start of the group `c` falls strictly inside.
    pub fn cut_outside_frames(&self, c: usize) -> usize {
        self.groups.iter().find(|g| g.start < c && c < g.end).map_or(c, |g| g.start)
    }

    /// The smallest valid cut at or above `c`: `c` itself, or the end of the group `c` falls strictly inside.
    pub fn next_cut_outside_frames(&self, c: usize) -> usize {
        self.groups.iter().find(|g| g.start < c && c < g.end).map_or(c, |g| g.end)
    }

    /// The rope position of the first token generated after the prompt (all three streams).
    pub fn next_position(&self) -> i32 {
        self.len() as i32 + self.rope_delta
    }

    /// The rope position of the token at cache index `cache_index >= len()` (a generated token): `[cache_index + delta; 3]`.
    pub fn decode_position(&self, cache_index: usize) -> [i32; 3] {
        [cache_index as i32 + self.rope_delta; 3]
    }
}

// ---------------------------------------------------------------------------------------------------------------- linspace

/// `torch.linspace(0, canvas - 1, n, dtype=float32)` of the CPU kernel, the float32 samples: `step = (end - start) / (n - 1)` in
/// float32; the first `n / 2` samples are `start + step * i` and the others `end - step * (n - 1 - i)`, the latter as one fused
/// multiply-add (the compiled kernel contracts it; the fixture compares the bits with torch 2.14 on aarch64 for every n and canvas in
/// 1..=160, and the generator's float64 emulation matched 12,000 larger pairs); `n == 1` is `[start]`. `canvas >= 1`, `n >= 0`.
pub fn linspace_f32(canvas: usize, n: usize, out: &mut Vec<f32>) {
    out.clear();
    if n == 0 {
        return;
    }
    if n == 1 {
        out.push(0.0);
        return;
    }
    let end = (canvas - 1) as f32;
    let step = end / (n - 1) as f32;
    let halfway = n / 2;
    for i in 0..n {
        out.push(if i < halfway { 0.0f32 + step * i as f32 } else { (-step).mul_add((n - i - 1) as f32, end) });
    }
}

/// `torch.linspace(0, canvas - 1, n).round().long()`: [`linspace_f32`] rounded half to even (`torch.round`).
pub fn linspace_round(canvas: usize, n: usize, out: &mut Vec<i64>) {
    let mut v = Vec::with_capacity(n);
    linspace_f32(canvas, n, &mut v);
    out.clear();
    out.extend(v.iter().map(|x| x.round_ties_even() as i64));
}

// ----------------------------------------------------------------------------------------------------------------- grouping

#[derive(Clone, Debug)]
struct RawFrame {
    modality: Modality,
    thumb: Range<usize>,
    slices: Vec<Range<usize>>,
    /// `_frame_end_idx` of the last crop: past every consecutive marker.
    end: usize,
}

#[derive(Clone, Debug)]
struct RawGroup {
    start: usize,
    end: usize,
    frames: Vec<RawFrame>,
}

fn modality_of(index: usize, label: u8) -> Result<Option<Modality>, CanvasError> {
    match label {
        0 => Ok(None),
        1 => Ok(Some(Modality::Image)),
        2 => Ok(Some(Modality::Video)),
        _ => Err(CanvasError::UnsupportedModality { index, label }),
    }
}

/// HF `_group_visual_frames`.
fn group_visual_frames(ids: &[u32], mm: &[u8], markers: &Markers) -> Result<Vec<RawGroup>, CanvasError> {
    let n = ids.len();
    // every crop is a maximal run of one non-zero label; a run of label 1 touching one of label 2 is two crops
    let mut groups: Vec<(usize, Vec<RawFrame>)> = Vec::new();
    let mut previous_end = 0usize;
    let mut i = 0usize;
    while i < n {
        let Some(modality) = modality_of(i, mm[i])? else {
            i += 1;
            continue;
        };
        let start = i;
        while i < n && mm[i] == mm[start] {
            i += 1;
        }
        let end = i;
        // the marker in front of the crop must be a token outside every crop
        if start == 0 || mm[start - 1] != 0 {
            return Err(CanvasError::CropWithoutMarker { start });
        }
        let has_frame = !groups.is_empty();
        if has_frame && ids[start - 1] == markers.slice_start {
            groups.last_mut().unwrap().1.last_mut().unwrap().slices.push(start..end);
        } else {
            let frame = RawFrame { modality, thumb: start..end, slices: Vec::new(), end: 0 };
            let gap = &ids[previous_end.min(start - 1)..start - 1];
            let adjacent = has_frame && gap.iter().all(|&t| markers.is_marker(t));
            if adjacent {
                groups.last_mut().unwrap().1.push(frame);
            } else {
                groups.push((start - 1, vec![frame]));
            }
        }
        previous_end = end;
    }
    let mut out = Vec::with_capacity(groups.len());
    for (start, mut frames) in groups {
        for f in &mut frames {
            let mut end = f.slices.last().map_or(f.thumb.end, |s| s.end);
            while end < n && markers.is_marker(ids[end]) {
                end += 1;
            }
            f.end = end;
        }
        let end = frames.last().unwrap().end;
        out.push(RawGroup { start, end, frames });
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------------------------------ layout

struct Sizes<'a> {
    image: &'a [[u32; 2]],
    video: &'a [[u32; 2]],
    next_image: usize,
    next_video: usize,
}

impl Sizes<'_> {
    fn next(&mut self, m: Modality) -> Result<[u32; 2], CanvasError> {
        let (list, cursor) = match m {
            Modality::Image => (self.image, &mut self.next_image),
            Modality::Video => (self.video, &mut self.next_video),
        };
        let v = list.get(*cursor).copied().ok_or(CanvasError::TargetSizesExhausted {
            modality: m,
            crop: *cursor,
            have: list.len(),
        })?;
        *cursor += 1;
        Ok(v)
    }
}

/// `get_vision_position_ids`: the `(T, H, W)` of the tokens of one crop (`llm_h * llm_w` of them, row-major) written into `out`.
/// `canvas_h` / `canvas_w` of 0 mean "the crop's own grid".
#[allow(clippy::too_many_arguments)]
fn crop_positions(
    origin: i64,
    grid: [u32; 2],
    canvas_h: i64,
    canvas_w: i64,
    h_off: i64,
    w_off: i64,
    merge: usize,
    out: &mut [[i32; 3]],
    scratch: &mut Vec<i64>,
) {
    let llm_h = grid[0] as usize / merge;
    let llm_w = grid[1] as usize / merge;
    let ch = if canvas_h == 0 { llm_h } else { canvas_h as usize };
    let cw = if canvas_w == 0 { llm_w } else { canvas_w as usize };
    let mut rows = Vec::new();
    linspace_round(ch, llm_h, scratch);
    rows.extend(scratch.iter().map(|&r| r + h_off));
    linspace_round(cw, llm_w, scratch);
    for (r, &hr) in rows.iter().enumerate() {
        for (c, &wc) in scratch.iter().enumerate() {
            out[r * llm_w + c] = [origin as i32, (origin + hr) as i32, (origin + wc + w_off) as i32];
        }
    }
}

/// The canvas M-RoPE positions of a prompt: HF `get_rope_index` for one unpadded sequence, plus the frame spans.
pub fn canvas_positions(input: &CanvasInput<'_>) -> Result<CanvasPositions, CanvasError> {
    let n = input.ids.len();
    if n != input.mm_token_type_ids.len() {
        return Err(CanvasError::LengthMismatch { ids: n, mm_token_type_ids: input.mm_token_type_ids.len() });
    }
    if n == 0 {
        return Err(CanvasError::EmptyPrompt);
    }
    if n > (1 << 28) {
        return Err(CanvasError::TooLong { len: n });
    }
    let merge = input.downsample.merge_factor();
    let groups = group_visual_frames(input.ids, input.mm_token_type_ids, &input.markers)?;
    let mut pos: Vec<[i32; 3]> = (0..n).map(|i| [i as i32; 3]).collect();
    let mut sizes = Sizes { image: input.image_sizes, video: input.video_sizes, next_image: 0, next_video: 0 };
    let mut scratch: Vec<i64> = Vec::new();
    let mut frames_out: Vec<CanvasFrame> = Vec::new();
    let mut groups_out: Vec<Range<usize>> = Vec::new();
    let mut flag = vec![false; n];

    let mut current_pos: i64 = 0;
    let mut current_idx = 0usize;
    let arange = |pos: &mut [[i32; 3]], range: Range<usize>, base: i64| {
        for (k, p) in pos[range].iter_mut().enumerate() {
            *p = [(base + k as i64) as i32; 3];
        }
    };
    for g in &groups {
        // text in front of the group is plain 1-D
        if g.start > current_idx {
            let text_len = g.start - current_idx;
            arange(&mut pos, current_idx..g.start, current_pos);
            current_pos += text_len as i64;
        }
        let mut frame_cursor = g.start;
        for f in &g.frames {
            let thumb_sizes = sizes.next(f.modality)?;
            let frame_start = f.thumb.start - 1;
            // tokens between two frames of a clip are 1-D text; the extra step keeps the next halo off the last of them
            if frame_start > frame_cursor {
                let gap_len = frame_start - frame_cursor;
                arange(&mut pos, frame_cursor..frame_start, current_pos);
                current_pos += gap_len as i64 + 1;
            }
            let frame_end = f.end.min(g.end);
            let origin = current_pos;
            let halo = (origin - 1).max(0);

            let (mut llm_slice_h, mut llm_slice_w) = (0i64, 0i64);
            let (mut num_rows, mut num_cols) = (0usize, 0usize);
            let mut canvas_h = (thumb_sizes[0] as usize / merge) as i64;
            let mut canvas_w = (thumb_sizes[1] as usize / merge) as i64;
            let mut first_slice_sizes = [0u32; 2];
            if !f.slices.is_empty() {
                // a gap wider than the two `</slice><slice>` markers is the newline that ends a row and fixes the column count
                num_cols = f.slices.len();
                for k in 0..f.slices.len() - 1 {
                    if f.slices[k + 1].start - f.slices[k].end > 2 {
                        num_cols = k + 1;
                        break;
                    }
                }
                num_rows = f.slices.len() / num_cols;
                if num_rows * num_cols != f.slices.len() {
                    num_rows = 1;
                    num_cols = f.slices.len();
                }
                first_slice_sizes = sizes.next(f.modality)?;
                llm_slice_h = (first_slice_sizes[0] as usize / merge) as i64;
                llm_slice_w = (first_slice_sizes[1] as usize / merge) as i64;
                canvas_h = num_rows as i64 * llm_slice_h;
                canvas_w = num_cols as i64 * llm_slice_w;
            }

            // base coat for the whole frame, then `<image>` on the halo just outside the canvas
            for p in &mut pos[frame_start..frame_end] {
                *p = [origin as i32; 3];
            }
            pos[frame_start][1] = halo as i32;
            pos[frame_start][2] = halo as i32;
            for fl in &mut flag[frame_start..frame_end] {
                *fl = true;
            }

            // `</image>` closes the thumbnail on the far corner of the canvas
            if f.thumb.end < frame_end {
                pos[f.thumb.end][1] = (origin + canvas_h) as i32;
                pos[f.thumb.end][2] = (origin + canvas_w) as i32;
            }

            // thumbnail tokens: stretched over the whole canvas when slices surround it
            let check_run = |range: &Range<usize>, grid: [u32; 2]| -> Result<(), CanvasError> {
                let expected = (grid[0] as usize / merge) * (grid[1] as usize / merge);
                if range.len() != expected {
                    return Err(CanvasError::RunLengthMismatch { start: range.start, len: range.len(), grid, expected });
                }
                Ok(())
            };
            check_run(&f.thumb, thumb_sizes)?;
            crop_positions(origin, thumb_sizes, canvas_h, canvas_w, 0, 0, merge, &mut pos[f.thumb.clone()], &mut scratch);

            // slices, with the `<slice>` / `</slice>` markers pinning their corners
            for (k, s) in f.slices.iter().enumerate() {
                let h_off = (k / num_cols) as i64 * llm_slice_h;
                let w_off = (k % num_cols) as i64 * llm_slice_w;
                let sl_sizes = if k == 0 { first_slice_sizes } else { sizes.next(f.modality)? };
                let slice_h = (sl_sizes[0] as usize / merge) as i64;
                let slice_w = (sl_sizes[1] as usize / merge) as i64;
                let marker = s.start - 1;
                if marker >= frame_start {
                    pos[marker][1] = (origin + h_off) as i32;
                    pos[marker][2] = (origin + w_off) as i32;
                }
                if s.end < frame_end {
                    pos[s.end][1] = (origin + h_off + slice_h - 1) as i32;
                    pos[s.end][2] = (origin + w_off + slice_w - 1) as i32;
                }
                check_run(s, sl_sizes)?;
                crop_positions(origin, sl_sizes, 0, 0, h_off, w_off, merge, &mut pos[s.clone()], &mut scratch);
            }

            // the "\n" that ends a row of slices sits just past the right edge of that row
            for k in 0..f.slices.len().saturating_sub(1) {
                let (gap_start, gap_end) = (f.slices[k].end, f.slices[k + 1].start);
                if gap_end - gap_start <= 2 {
                    continue;
                }
                let boundary_h = ((k / num_cols) as i64 + 1) * llm_slice_h - 1;
                let right_edge_w = num_cols as i64 * llm_slice_w;
                for p in &mut pos[gap_start + 1..gap_end - 1] {
                    p[1] = (origin + boundary_h) as i32;
                    p[2] = (origin + right_edge_w) as i32;
                }
            }

            current_pos = origin + canvas_h.max(canvas_w) + 1;
            frame_cursor = frame_end;
            frames_out.push(CanvasFrame {
                modality: f.modality,
                span: frame_start..frame_end,
                thumbnail: f.thumb.clone(),
                slices: f.slices.clone(),
                grid: (num_rows, num_cols),
                canvas: (canvas_h, canvas_w),
                origin,
            });
        }
        if frame_cursor < g.end {
            let trail_len = g.end - frame_cursor;
            arange(&mut pos, frame_cursor..g.end, current_pos);
            current_pos += trail_len as i64;
        }
        current_idx = g.end;
        groups_out.push(g.start..g.end);
        for fl in &mut flag[g.start..g.end] {
            *fl = true;
        }
    }
    if current_idx < n {
        arange(&mut pos, current_idx..n, current_pos);
    }
    let max_pos = pos.iter().flat_map(|p| p.iter().copied()).max().unwrap_or(0) as i64;
    let rope_delta = (max_pos + 1 - n as i64) as i32;
    let mut frame_prefix = Vec::with_capacity(n + 1);
    frame_prefix.push(0u32);
    for &f in &flag {
        frame_prefix.push(frame_prefix.last().unwrap() + f as u32);
    }
    Ok(CanvasPositions {
        positions: pos,
        rope_delta,
        groups: groups_out,
        frames: frames_out,
        image_crops: sizes.next_image,
        video_crops: sizes.next_video,
        frame_prefix,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use std::sync::OnceLock;

    const CASES: &str = include_str!("../../../testdata/minicpmv4_7/rope/cases.json");

    pub(crate) fn doc() -> &'static Value {
        static D: OnceLock<Value> = OnceLock::new();
        D.get_or_init(|| serde_json::from_str(CASES).expect("cases.json"))
    }

    pub(crate) fn b64(s: &str) -> Vec<u8> {
        let val = |c: u8| -> u32 {
            match c {
                b'A'..=b'Z' => (c - b'A') as u32,
                b'a'..=b'z' => (c - b'a') as u32 + 26,
                b'0'..=b'9' => (c - b'0') as u32 + 52,
                b'+' => 62,
                b'/' => 63,
                _ => panic!("base64 byte {c}"),
            }
        };
        let s = s.trim_end_matches('=').as_bytes();
        let mut out = Vec::with_capacity(s.len() * 3 / 4);
        for chunk in s.chunks(4) {
            let mut acc = 0u32;
            for &c in chunk {
                acc = (acc << 6) | val(c);
            }
            acc <<= 6 * (4 - chunk.len()) as u32;
            let bytes = acc.to_be_bytes();
            out.extend_from_slice(&bytes[1..1 + (chunk.len() * 6 / 8)]);
        }
        out
    }

    /// The generator's position codec: per token and channel the zig-zag varint of the delta against the previous token's channel.
    pub(crate) fn decode_positions(bytes: &[u8], n: usize) -> Vec<[i32; 3]> {
        let (mut at, mut prev) = (0usize, [0i64; 3]);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut t = [0i32; 3];
            for (c, slot) in t.iter_mut().enumerate() {
                let (mut v, mut shift) = (0u64, 0u32);
                loop {
                    let b = bytes[at];
                    at += 1;
                    v |= ((b & 0x7f) as u64) << shift;
                    if b & 0x80 == 0 {
                        break;
                    }
                    shift += 7;
                }
                prev[c] += ((v >> 1) as i64) ^ -((v & 1) as i64);
                *slot = prev[c] as i32;
            }
            out.push(t);
        }
        assert_eq!(at, bytes.len(), "trailing bytes in a position blob");
        out
    }

    pub(crate) struct Rendered {
        pub(crate) ids: Vec<u32>,
        pub(crate) mm: Vec<u8>,
        pub(crate) img: Vec<[u32; 2]>,
        pub(crate) vid: Vec<[u32; 2]>,
    }

    fn hw(v: &Value) -> [u32; 2] {
        [v[0].as_u64().unwrap() as u32, v[1].as_u64().unwrap() as u32]
    }

    pub(crate) fn markers_of(d: &Value) -> Markers {
        let m = &d["markers"];
        let g = |k: &str| m[k].as_u64().unwrap() as u32;
        Markers {
            image_start: g("image_start"),
            image_end: g("image_end"),
            slice_start: g("slice_start"),
            slice_end: g("slice_end"),
            newline: g("newline"),
        }
    }

    /// processing_minicpmv4_7.py's layout from a recipe: `<image>` pad*n `</image>`, then (`<slice>` pad*m `</slice>`)*cols per row,
    /// rows joined by a newline; the thumbnail's and every slice's patch grid appended to the modality's size list.
    fn render_frame(frame: &Value, pad: u32, label: u8, merge: usize, d: &Value, r: &mut Rendered, sizes_video: bool) {
        let mk = markers_of(d);
        let ntok = |g: [u32; 2]| (g[0] as usize / merge) * (g[1] as usize / merge);
        let thumb = hw(&frame["thumb"]);
        let push_sizes = |g: [u32; 2], r: &mut Rendered| if sizes_video { r.vid.push(g) } else { r.img.push(g) };
        r.ids.push(mk.image_start);
        r.mm.push(0);
        for _ in 0..ntok(thumb) {
            r.ids.push(pad);
            r.mm.push(label);
        }
        r.ids.push(mk.image_end);
        r.mm.push(0);
        push_sizes(thumb, r);
        let grid = frame["grid"]
            .as_array()
            .map(|g| (g[0].as_u64().unwrap() as usize, g[1].as_u64().unwrap() as usize))
            .unwrap_or((0, 0));
        if grid.0 > 0 && grid.1 > 0 {
            let slices: Vec<[u32; 2]> = match frame.get("slices") {
                Some(s) => s.as_array().unwrap().iter().map(hw).collect(),
                None => vec![hw(&frame["slice"]); grid.0 * grid.1],
            };
            let mut k = 0;
            for row in 0..grid.0 {
                if row > 0 {
                    r.ids.push(mk.newline);
                    r.mm.push(0);
                }
                for _ in 0..grid.1 {
                    r.ids.push(mk.slice_start);
                    r.mm.push(0);
                    for _ in 0..ntok(slices[k]) {
                        r.ids.push(pad);
                        r.mm.push(label);
                    }
                    r.ids.push(mk.slice_end);
                    r.mm.push(0);
                    push_sizes(slices[k], r);
                    k += 1;
                }
            }
        }
    }

    pub(crate) fn render(case: &Value, d: &Value) -> Rendered {
        let mode = Downsample::parse(case["mode"].as_str().unwrap()).unwrap();
        let merge = mode.merge_factor();
        let mut r = Rendered { ids: vec![], mm: vec![], img: vec![], vid: vec![] };
        let (image_pad, video_pad) = (d["image_pad"].as_u64().unwrap() as u32, d["video_pad"].as_u64().unwrap() as u32);
        for seg in case["segs"].as_array().unwrap() {
            match seg[0].as_str().unwrap() {
                "text" => {
                    for t in seg[1].as_array().unwrap() {
                        r.ids.push(t.as_u64().unwrap() as u32);
                        r.mm.push(0);
                    }
                }
                "image" => {
                    if let Some(id) = seg[1].get("image_id").and_then(Value::as_u64) {
                        r.ids.extend([
                            d["image_id_start"].as_u64().unwrap() as u32,
                            15 + id as u32,
                            d["image_id_end"].as_u64().unwrap() as u32,
                        ]);
                        r.mm.extend([0, 0, 0]);
                    }
                    render_frame(&seg[1], image_pad, 1, merge, d, &mut r, false);
                }
                "video" => {
                    for fr in seg[1].as_array().unwrap() {
                        render_frame(fr, video_pad, 2, merge, d, &mut r, true);
                    }
                }
                "raw" => {
                    r.ids.extend(seg[1].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32));
                    r.mm.extend(seg[2].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u8));
                    r.img.extend(seg[3].as_array().unwrap().iter().map(hw));
                    r.vid.extend(seg[4].as_array().unwrap().iter().map(hw));
                }
                other => panic!("segment kind {other}"),
            }
        }
        r
    }

    fn ids_sha(ids: &[u32]) -> String {
        let mut h = Sha256::new();
        for t in ids {
            h.update(t.to_le_bytes());
        }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    fn range_pairs(v: &Value) -> Vec<Range<usize>> {
        v.as_array().unwrap().iter().map(|p| p[0].as_u64().unwrap() as usize..p[1].as_u64().unwrap() as usize).collect()
    }

    fn run_case(case: &Value, d: &Value) -> Result<CanvasPositions, String> {
        let name = case["name"].as_str().unwrap();
        let r = render(case, d);
        if ids_sha(&r.ids) != case["ids_sha256"].as_str().unwrap() || r.ids.len() != case["n"].as_u64().unwrap() as usize {
            return Err(format!("{name}: the rendered prompt differs from the generator's"));
        }
        let out = canvas_positions(&CanvasInput {
            ids: &r.ids,
            mm_token_type_ids: &r.mm,
            image_sizes: &r.img,
            video_sizes: &r.vid,
            downsample: Downsample::parse(case["mode"].as_str().unwrap()).unwrap(),
            markers: markers_of(d),
        })
        .map_err(|e| format!("{name}: {e}"))?;
        let want = decode_positions(&b64(case["pos"].as_str().unwrap()), r.ids.len());
        if let Some(i) = (0..want.len()).find(|&i| want[i] != out.positions[i]) {
            return Err(format!(
                "{name}: token {i} (id {}) is at {:?}, HF has {:?} (the first of {} differing tokens)",
                r.ids[i],
                out.positions[i],
                want[i],
                (0..want.len()).filter(|&j| want[j] != out.positions[j]).count()
            ));
        }
        let delta = case["delta"].as_i64().unwrap();
        if out.rope_delta as i64 != delta {
            return Err(format!("{name}: rope_delta {} but HF has {delta}", out.rope_delta));
        }
        if out.groups != range_pairs(&case["groups"]) {
            return Err(format!("{name}: groups {:?} but HF has {:?}", out.groups, range_pairs(&case["groups"])));
        }
        let spans: Vec<Range<usize>> = out.frames.iter().map(|f| f.span.clone()).collect();
        if spans != range_pairs(&case["frames"]) {
            return Err(format!("{name}: frame spans {spans:?} but HF has {:?}", range_pairs(&case["frames"])));
        }
        if out.image_crops != r.img.len() || out.video_crops != r.vid.len() {
            return Err(format!(
                "{name}: consumed {}+{} crop sizes of {}+{}",
                out.image_crops,
                out.video_crops,
                r.img.len(),
                r.vid.len()
            ));
        }
        Ok(out)
    }

    #[test]
    fn the_float32_linspace_matches_torch_for_every_n_and_canvas_to_160() {
        let ls = &doc()["linspace"];
        assert_eq!(ls["n_max"], 160);
        assert_eq!(ls["c_max"], 160);
        let per_c = ls["per_c_sha256"].as_array().unwrap();
        assert_eq!(per_c.len(), 160);
        let mut buf = Vec::new();
        for c in 1..=160usize {
            let mut h = Sha256::new();
            for n in 1..=160usize {
                linspace_round(c, n, &mut buf);
                assert_eq!(buf.len(), n);
                for &v in &buf {
                    h.update((v as i32).to_le_bytes());
                }
            }
            let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(
                got,
                per_c[c - 1].as_str().unwrap(),
                "canvas {c}: some n in 1..=160 differs from torch.linspace(0, {c} - 1, n).round()"
            );
        }
        // the float32 samples themselves, bit for bit (the rounding hides the fused multiply-add in the integers)
        let per_c_f32 = ls["per_c_f32_sha256"].as_array().unwrap();
        assert_eq!(per_c_f32.len(), 160);
        let mut raw = Vec::new();
        for c in 1..=160usize {
            let mut h = Sha256::new();
            for n in 1..=160usize {
                linspace_f32(c, n, &mut raw);
                for v in &raw {
                    h.update(v.to_le_bytes());
                }
            }
            let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(got, per_c_f32[c - 1].as_str().unwrap(), "canvas {c}: float32 samples differ from torch.linspace");
        }
        // and the readable part of the same table
        let table = ls["table_le16"].as_object().unwrap();
        assert_eq!(table.len(), 256);
        for (key, want) in table {
            let (n, c) = key.split_once(',').unwrap();
            linspace_round(c.parse().unwrap(), n.parse().unwrap(), &mut buf);
            let want: Vec<i64> = want.as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
            assert_eq!(buf, want, "n={n} canvas={c}");
        }
    }

    #[test]
    fn linspace_rounds_half_to_even_and_a_single_sample_is_zero() {
        let mut v = Vec::new();
        // 5 samples over [0, 2]: 0, 0.5, 1, 1.5, 2 -> 0, 0, 1, 2, 2 (ties to even)
        linspace_round(3, 5, &mut v);
        assert_eq!(v, [0, 0, 1, 2, 2]);
        linspace_round(9, 1, &mut v);
        assert_eq!(v, [0]);
        linspace_round(4, 4, &mut v);
        assert_eq!(v, [0, 1, 2, 3], "n == canvas is the identity");
    }

    #[test]
    fn at_least_400_prompt_cases_match_hf_to_the_token() {
        let d = doc();
        let cases = d["cases"].as_array().unwrap();
        assert!(cases.len() >= 400, "{} cases", cases.len());
        let mut tokens = 0usize;
        let mut with_slices = 0usize;
        let mut with_video = 0usize;
        let mut failures = Vec::new();
        for c in cases {
            match run_case(c, d) {
                Ok(out) => {
                    tokens += out.len();
                    with_slices += out.frames.iter().any(|f| !f.slices.is_empty()) as usize;
                    with_video += out.frames.iter().any(|f| f.modality == Modality::Video) as usize;
                }
                Err(e) => failures.push(e),
            }
        }
        assert!(failures.is_empty(), "{} of {} cases differ from HF; first: {}", failures.len(), cases.len(), failures[0]);
        // the fixture exercises what it claims to
        assert!(
            tokens > 50_000 && with_slices > 200 && with_video > 40,
            "tokens {tokens}, with slices {with_slices}, with video {with_video}"
        );
    }

    #[test]
    fn a_thumbnail_without_slices_text_around_and_the_trailing_newline_absorbed() {
        // HF output for this prompt (probed once): ids 1 2 <image> pad*4 </image> \n 5; the thumbnail is 8x8 patches = 2x2 tokens at 16x
        let m = Markers::MINICPMV4_7;
        let ids = [1, 2, m.image_start, 248_056, 248_056, 248_056, 248_056, m.image_end, m.newline, 5];
        let mm = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0];
        let out = canvas_positions(&CanvasInput {
            ids: &ids,
            mm_token_type_ids: &mm,
            image_sizes: &[[8, 8]],
            video_sizes: &[],
            downsample: Downsample::X16,
            markers: m,
        })
        .unwrap();
        let t = [0, 1, 2, 2, 2, 2, 2, 2, 2, 5];
        let h = [0, 1, 1, 2, 2, 3, 3, 4, 2, 5];
        let w = [0, 1, 1, 2, 3, 2, 3, 4, 2, 5];
        let want: Vec<[i32; 3]> = (0..10).map(|i| [t[i], h[i], w[i]]).collect();
        assert_eq!(out.positions, want);
        assert_eq!(out.rope_delta, -4);
        assert_eq!(out.groups.len(), 1);
        assert_eq!(out.groups[0], 2..9);
        assert_eq!(out.frames.len(), 1);
        assert_eq!(out.frames[0].origin, 2);
        assert_eq!(out.frames[0].canvas, (2, 2));
        assert_eq!(out.frames[0].grid, (0, 0));
        assert_eq!(out.frame_tokens(), [false, false, true, true, true, true, true, true, true, false]);
        assert_eq!(out.next_position(), 6);
        assert_eq!(out.decode_position(10), [6; 3]);
    }

    /// Two slices side by side under a thumbnail of the same size, derived by hand from the rules in the module doc: 16x, every crop
    /// 8x8 patches = 2x2 tokens, so the canvas is 2 rows x 4 columns, T = origin = 0.
    #[test]
    fn two_slices_side_by_side_by_hand() {
        let m = Markers::MINICPMV4_7;
        let p = 248_056u32;
        let mut ids = vec![m.image_start, p, p, p, p, m.image_end];
        ids.extend([m.slice_start, p, p, p, p, m.slice_end]);
        ids.extend([m.slice_start, p, p, p, p, m.slice_end]);
        let mm: Vec<u8> = ids.iter().map(|&t| (t == p) as u8).collect();
        let out = canvas_positions(&CanvasInput {
            ids: &ids,
            mm_token_type_ids: &mm,
            image_sizes: &[[8, 8], [8, 8], [8, 8]],
            video_sizes: &[],
            downsample: Downsample::X16,
            markers: m,
        })
        .unwrap();
        // canvas 2 x 4, origin 0, halo max(0 - 1, 0) = 0
        let want: [[i32; 3]; 18] = [
            [0, 0, 0], // <image>: halo
            // thumbnail 2x2 stretched over 2 x 4: rows linspace(0,1,2) = 0,1; columns linspace(0,3,2) = 0,3
            [0, 0, 0],
            [0, 0, 3],
            [0, 1, 0],
            [0, 1, 3],
            [0, 2, 4], // </image>: (origin + canvas_h, origin + canvas_w)
            [0, 0, 0], // <slice> of slice 0: its top-left corner
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 1], // </slice>: bottom-right corner (0 + 0 + 2 - 1, 0 + 0 + 2 - 1)
            [0, 0, 2], // <slice> of slice 1: column offset 2
            [0, 0, 2],
            [0, 0, 3],
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 3], // </slice>
        ];
        assert_eq!(out.positions, want);
        assert_eq!(out.frames[0].grid, (1, 2));
        assert_eq!(out.frames[0].canvas, (2, 4));
        // nothing after the group: the last position is 4 (W of </image>), so delta = 4 + 1 - 18
        assert_eq!(out.rope_delta, 5 - 18);
        assert!(out.chunk_is_explicit(0..1) && out.chunk_is_explicit(17..18));
    }

    fn input_of<'a>(
        r: &'a Rendered,
        mode: Downsample,
        img: &'a [[u32; 2]],
        vid: &'a [[u32; 2]],
        ids: &'a [u32],
        mm: &'a [u8],
    ) -> CanvasInput<'a> {
        let _ = r;
        CanvasInput {
            ids,
            mm_token_type_ids: mm,
            image_sizes: img,
            video_sizes: vid,
            downsample: mode,
            markers: Markers::MINICPMV4_7,
        }
    }

    #[test]
    fn malformed_inputs_have_typed_errors() {
        let m = Markers::MINICPMV4_7;
        let p = 248_056u32;
        let dummy = Rendered { ids: vec![], mm: vec![], img: vec![], vid: vec![] };
        let ok_ids = [m.image_start, p, p, p, p, m.image_end];
        let ok_mm = [0u8, 1, 1, 1, 1, 0];
        let go = |ids: &[u32], mm: &[u8], img: &[[u32; 2]], vid: &[[u32; 2]], mode| {
            canvas_positions(&input_of(&dummy, mode, img, vid, ids, mm))
        };
        assert!(go(&ok_ids, &ok_mm, &[[8, 8]], &[], Downsample::X16).is_ok());
        assert_eq!(
            go(&ok_ids, &ok_mm[..5], &[[8, 8]], &[], Downsample::X16),
            Err(CanvasError::LengthMismatch { ids: 6, mm_token_type_ids: 5 })
        );
        assert_eq!(go(&[], &[], &[], &[], Downsample::X16), Err(CanvasError::EmptyPrompt));
        // a modality the canvas does not know (3 is audio)
        assert_eq!(
            go(&[1, p, 2], &[0, 3, 0], &[], &[], Downsample::X16),
            Err(CanvasError::UnsupportedModality { index: 1, label: 3 })
        );
        // a crop with no marker in front of it: at the very start, and two crops of different modalities touching
        assert_eq!(go(&[p, p], &[1, 1], &[[8, 8]], &[], Downsample::X16), Err(CanvasError::CropWithoutMarker { start: 0 }));
        assert_eq!(
            go(
                &[m.image_start, p, p, p, p, 248_057, 248_057, 248_057, 248_057],
                &[0, 1, 1, 1, 1, 2, 2, 2, 2],
                &[[8, 8]],
                &[[8, 8]],
                Downsample::X16
            ),
            Err(CanvasError::CropWithoutMarker { start: 5 })
        );
        // target sizes exhausted: the prompt has an image, the list is empty; a video with only an image size
        assert_eq!(
            go(&ok_ids, &ok_mm, &[], &[], Downsample::X16),
            Err(CanvasError::TargetSizesExhausted { modality: Modality::Image, crop: 0, have: 0 })
        );
        let vids = [m.image_start, 248_057, 248_057, 248_057, 248_057, m.image_end];
        let vmm = [0u8, 2, 2, 2, 2, 0];
        assert_eq!(
            go(&vids, &vmm, &[[8, 8]], &[], Downsample::X16),
            Err(CanvasError::TargetSizesExhausted { modality: Modality::Video, crop: 0, have: 0 })
        );
        // slices: the thumbnail's size is there, the slice's is not
        let mut ids = ok_ids.to_vec();
        ids.extend([m.slice_start, p, p, p, p, m.slice_end]);
        let mm: Vec<u8> = ids.iter().map(|&t| (t == p) as u8).collect();
        assert_eq!(
            go(&ids, &mm, &[[8, 8]], &[], Downsample::X16),
            Err(CanvasError::TargetSizesExhausted { modality: Modality::Image, crop: 1, have: 1 })
        );
        // run length: 8x8 patches at 16x are 4 tokens, at 4x they are 16
        assert_eq!(
            go(&ok_ids, &ok_mm, &[[8, 8]], &[], Downsample::X4),
            Err(CanvasError::RunLengthMismatch { start: 1, len: 4, grid: [8, 8], expected: 16 })
        );
        assert_eq!(
            go(&ok_ids, &ok_mm, &[[16, 8]], &[], Downsample::X16),
            Err(CanvasError::RunLengthMismatch { start: 1, len: 4, grid: [16, 8], expected: 8 })
        );
        // leftover target sizes are not an error (HF does not check), but they are reported
        let out = go(&ok_ids, &ok_mm, &[[8, 8], [8, 8]], &[], Downsample::X16).unwrap();
        assert_eq!((out.image_crops, out.video_crops), (1, 0));
        assert!(!format!("{}", CanvasError::EmptyPrompt).is_empty());
    }

    /// A chat-template prompt: user text, an image with two slices (row of two), the template's `\n`, the question. Pins every flag, the
    /// explicit chunks and the cuts.
    #[test]
    fn frame_tokens_chunks_and_cuts_outside_frames() {
        let m = Markers::MINICPMV4_7;
        let p = 248_056u32;
        let mut ids = vec![100, 101, 102, 198]; // text, the template's "\n" before the image (text: not absorbed)
        let first = ids.len();
        ids.extend([m.image_start, p, p, p, p, m.image_end]);
        ids.extend([m.slice_start, p, p, p, p, m.slice_end, m.slice_start, p, p, p, p, m.slice_end]);
        let frame_last = ids.len();
        ids.extend([m.newline, 7, 8, 9]); // the "\n" after the image is absorbed; 7 8 9 are text
        let mm: Vec<u8> = ids.iter().map(|&t| (t == p) as u8).collect();
        let out = canvas_positions(&CanvasInput {
            ids: &ids,
            mm_token_type_ids: &mm,
            image_sizes: &[[8, 8], [8, 8], [8, 8]],
            video_sizes: &[],
            downsample: Downsample::X16,
            markers: m,
        })
        .unwrap();
        assert_eq!(out.groups.len(), 1);
        assert_eq!(out.groups[0], first..frame_last + 1);
        let flags = out.frame_tokens();
        for (i, &f) in flags.iter().enumerate() {
            assert_eq!(f, (first..=frame_last).contains(&i), "token {i}");
        }
        // the text before: 0,1,2,3; the frame at origin 4; after it text at origin + max(2, 4) + 1 = 9
        assert_eq!(out.positions[3], [3; 3]);
        assert_eq!(out.positions[first], [4, 3, 3], "<image> sits on the halo: one step before the origin on H and W");
        assert_eq!(out.positions[frame_last], [4; 3], "the absorbed newline sits at the origin on all streams");
        assert_eq!(out.positions[frame_last + 1], [9; 3]);
        assert_eq!(out.positions[frame_last + 3], [11; 3]);
        // a chunk of the text before the image is implicit, one touching the opening marker or the absorbed newline is explicit
        assert!(!out.chunk_is_explicit(0..first));
        assert!(out.chunk_is_explicit(0..first + 1));
        assert!(out.chunk_is_explicit(first..first + 1));
        assert!(out.chunk_is_explicit(frame_last..frame_last + 1));
        assert!(!out.chunk_is_explicit(frame_last + 1..ids.len()));
        assert!(!out.chunk_is_explicit(2..2));
        // cuts: anywhere strictly inside the group moves back to its first token (or forward to just past it)
        assert!(out.is_cut_outside_frames(first) && out.is_cut_outside_frames(frame_last + 1) && out.is_cut_outside_frames(0));
        for c in first + 1..=frame_last {
            assert!(!out.is_cut_outside_frames(c), "cut {c}");
            assert_eq!(out.cut_outside_frames(c), first);
            assert_eq!(out.next_cut_outside_frames(c), frame_last + 1);
        }
        assert_eq!(out.cut_outside_frames(2), 2);
        assert_eq!(out.cut_outside_frames(frame_last + 2), frame_last + 2);
    }

    /// The claim the cut helpers rest on, checked on every committed prompt: at a cut outside every group the prefix alone gets the
    /// positions the full prompt gives its tokens (the target sizes are the same list; the crops after the cut are simply not read), and
    /// the decode offset of the prefix is the one the helpers report. Cuts strictly inside a group are not required to agree, and on a
    /// frame with slices they almost never do (counted).
    #[test]
    fn a_cut_outside_frames_leaves_the_prefix_positions_unchanged() {
        let d = doc();
        let mut checked = 0usize;
        let mut inside_changed = 0usize;
        let mut inside_total = 0usize;
        for case in d["cases"].as_array().unwrap() {
            let r = render(case, d);
            let mode = Downsample::parse(case["mode"].as_str().unwrap()).unwrap();
            let run = |len: usize| {
                canvas_positions(&CanvasInput {
                    ids: &r.ids[..len],
                    mm_token_type_ids: &r.mm[..len],
                    image_sizes: &r.img,
                    video_sizes: &r.vid,
                    downsample: mode,
                    markers: markers_of(d),
                })
            };
            let full = run(r.ids.len()).unwrap();
            // every valid cut of the committed prompts that is a group end, a group start, or a stride through the text
            let mut cuts: Vec<usize> = vec![0, 1, r.ids.len()];
            for g in &full.groups {
                cuts.extend([g.start, g.end, g.start.saturating_sub(1), (g.end + 1).min(r.ids.len())]);
            }
            cuts.extend((0..r.ids.len()).step_by(37));
            cuts.sort_unstable();
            cuts.dedup();
            for c in cuts {
                if c == 0 {
                    continue;
                }
                let valid = full.is_cut_outside_frames(c);
                // an invalid cut that lands inside a crop run is a partial crop: HF's own run-length rule rejects it, so only count
                // those that parse
                match run(c) {
                    Ok(prefix) => {
                        if valid {
                            assert_eq!(&prefix.positions[..], &full.positions[..c], "{}: prefix of {c} tokens", case["name"]);
                            checked += 1;
                        } else {
                            inside_total += 1;
                            if prefix.positions[..] != full.positions[..c] {
                                inside_changed += 1;
                            }
                        }
                    }
                    Err(_) => assert!(!valid, "{}: a valid cut at {c} must parse", case["name"]),
                }
            }
        }
        eprintln!(
            "valid cuts checked: {checked}; cuts inside frames: {inside_total}, of which positions changed: {inside_changed}"
        );
        assert!(checked > 1000, "{checked} valid cuts checked");
        assert!(
            inside_total > 200 && inside_changed * 10 >= inside_total * 8,
            "{inside_changed} of {inside_total} cuts inside frames changed positions"
        );
    }

    #[test]
    fn frame_tokens_are_only_ever_inside_groups_and_text_outside_is_one_dimensional() {
        let d = doc();
        for case in d["cases"].as_array().unwrap() {
            let out = run_case(case, d).unwrap();
            let flags = out.frame_tokens();
            let mut off = 0i64;
            for (i, p) in out.positions.iter().enumerate() {
                let in_group = out.groups.iter().any(|g| g.contains(&i));
                assert_eq!(flags[i], in_group, "{} token {i}", case["name"]);
                if !in_group {
                    assert!(p[0] == p[1] && p[1] == p[2], "{} text token {i} is not 1-D: {p:?}", case["name"]);
                    let o = p[0] as i64 - i as i64;
                    // the offset only changes across a group
                    if i > 0 && !flags[i - 1] {
                        assert_eq!(o, off, "{} offset moved inside text at {i}", case["name"]);
                    }
                    off = o;
                }
            }
            // decode: HF adds one delta to every stream
            assert_eq!(out.decode_position(out.len()), [out.next_position(); 3]);
            let max = out.positions.iter().flat_map(|p| p.iter()).copied().max().unwrap();
            assert_eq!(out.next_position(), max + 1, "{}", case["name"]);
        }
    }

    /// The exhaustive sweep (thousands of prompts, `tools/gen_minicpmv_rope_fixtures.py --sweep PATH`; the file stays outside the
    /// repository): `REXL3_MCV_ROPE_SWEEP=PATH cargo test -p gb-model canvas_rope_sweep_against_hf -- --ignored`.
    #[test]
    #[ignore = "needs REXL3_MCV_ROPE_SWEEP=<file written by tools/gen_minicpmv_rope_fixtures.py --sweep>"]
    fn canvas_rope_sweep_against_hf() {
        let Some(path) = std::env::var_os("REXL3_MCV_ROPE_SWEEP") else {
            panic!("set REXL3_MCV_ROPE_SWEEP to the sweep file");
        };
        let text = std::fs::read_to_string(&path).expect("sweep file");
        let d: Value = serde_json::from_str(&text).expect("sweep json");
        let cases = d["cases"].as_array().unwrap();
        assert!(cases.len() >= 1000, "{} cases", cases.len());
        let failures: Vec<String> = cases.iter().filter_map(|c| run_case(c, &d).err()).collect();
        assert!(failures.is_empty(), "{} of {} sweep cases differ from HF; first: {}", failures.len(), cases.len(), failures[0]);
    }
}
