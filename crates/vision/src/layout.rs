//! The block-major patch layout both preprocessors share (Qwen3.8 patch 16, GLM-5.3-Flash patch 14), table-driven and threaded.
//!
//! A tensor element is a function of one `u8` sample alone: `x * (1/255)` then `(x - mean) / std` in float32 (then the fp16 cast) depends on
//! (channel, byte) and nothing else, so the 768 results are computed once per call, with the very same scalar expression the per-sample code
//! used (`Preprocessing::lut_f16`, `GlmPreprocessing::lut_f16`, each checked against that expression for all 3 x 256 inputs), and the layout is
//! gather and scatter: [`patchify`] reads each source pixel once, looks its three samples up and writes them at their place in the tensor.
//!
//! **The layout.** `[grid_h * grid_w, 3 * temporal * patch^2]`, patches block-major over `merge x merge` blocks (block row, block column, row in
//! block, column in block), inner order (channel, temporal, row, column), the single frame of a still image repeated over the temporal axis. One
//! row of blocks (`merge` rows of patches) is one contiguous run of the output and rows are independent, so [`patchify`] hands them to
//! `resample::par_rows` and the bytes cannot depend on the thread count.
//!
//! **The loops.** Inside a patch nothing computes an index: the three channel regions are split off the patch, their first temporal slots are cut
//! into rows of `patch` elements and zipped with the pixels of the source row (`chunks_exact(3)`), so the inner loop is three table loads and
//! three stores per pixel with no bounds checks; the other temporal slots are one contiguous `copy_from_slice` of the first (the same function of
//! the same samples, so the same bits). The output is `vec![0; n]`, which is a zeroed allocation: large ones come from fresh pages the kernel has
//! already zeroed (no memset), and the cost of writing 145 MB for a 12 MP Qwen image is the page faults of those pages, which the threads share
//! (`docs/VISION.md`, "Cost"). No `unsafe`.

use crate::resample::par_rows;

/// Source pixels one layout worker is given for another to pay for its start: a scoped thread costs about 15 to 30 us to start and join once
/// the first is running (130 us for the first, `docs/VISION.md` "Cost"), and the layout does about 3 ns a pixel on one core, so a worker pays
/// from about 10,000 pixels; 2^16 leaves room for the cores that are slower than that (`layout_scaling` in `tests/glm_speed.rs` prints the times
/// this was chosen from).
pub const MIN_PIXELS_PER_WORKER: usize = 1 << 16;

/// Pixels from which the hash of the canvas runs on a thread of its own beside the layout instead of after it: it is a serial chain (about 2 ns a
/// byte), the biggest single cost of a large image, and a thread costs a small fraction of it from here on.
pub const MIN_PIXELS_FOR_HASH_THREAD: usize = 1 << 16;

/// The geometry of a layout: patch side, merge side (patches per block side) and temporal patch size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub patch: usize,
    pub merge: usize,
    pub temporal: usize,
}

impl Shape {
    /// Elements per patch: `3 * temporal * patch^2`.
    pub fn patch_dim(&self) -> usize {
        3 * self.temporal * self.patch * self.patch
    }
    /// Pixels per side of an image this layout takes (`patch * merge`).
    pub fn factor(&self) -> usize {
        self.patch * self.merge
    }
}

/// Layout workers for `pixels` source pixels, of at most `max_workers`: one per [`MIN_PIXELS_PER_WORKER`], at least 1. Small images run inline.
pub fn layout_workers(max_workers: usize, pixels: usize) -> usize {
    (pixels / MIN_PIXELS_PER_WORKER).clamp(1, max_workers.max(1))
}

/// Whether the hash of `pixels` pixels runs beside the layout (needs a second thread to be allowed and enough work to hide its start).
pub fn hash_on_a_thread(max_workers: usize, pixels: usize) -> bool {
    max_workers > 1 && pixels >= MIN_PIXELS_FOR_HASH_THREAD
}

/// The threads an entry gives `pixels` pixels of layout under a cap of `max_workers` threads in all: the layout workers and whether the hash runs beside
/// them. The hash is a busy thread of its own, so beside it the layout gets one worker fewer than the cap (a cap of 2 is a layout on one thread and
/// the hash on the other; `workers + hash_aside as usize <= max_workers` always, which is what `GlmPreprocessing.threads` and
/// `REXL3_VISION_THREADS` promise).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub workers: usize,
    pub hash_aside: bool,
}

#[cfg(test)]
thread_local! {
    /// The plan the last entry on this thread chose (for the tests of the entries, as `resize::last_workers` is for the resize).
    static LAST_PLAN: std::cell::Cell<Option<Plan>> = const { std::cell::Cell::new(None) };
}

/// The plan of the last [`plan`] call on this thread, if any.
#[cfg(test)]
pub(crate) fn last_plan() -> Option<Plan> {
    LAST_PLAN.with(|p| p.get())
}

/// [`layout_workers`] and [`hash_on_a_thread`] for `pixels` pixels under `max_workers` threads: what every entry asks before it lays a tensor out.
/// `with_hash` says whether a hash is taken of the same bytes (the preprocessing entries; not `still_patches` or the float32 tensor): then the hash
/// thread counts against the cap and the layout gets `max_workers - 1` at most.
pub fn plan(max_workers: usize, pixels: usize, with_hash: bool) -> Plan {
    let hash_aside = with_hash && hash_on_a_thread(max_workers, pixels);
    let layout_cap = if hash_aside { max_workers - 1 } else { max_workers };
    let plan = Plan { workers: layout_workers(layout_cap, pixels), hash_aside };
    #[cfg(test)]
    LAST_PLAN.with(|p| p.set(Some(plan)));
    plan
}

/// Normalise and lay out `width x height` RGB bytes with a value table per channel: `[grid_h * grid_w, shape.patch_dim()]`, block-major (see the
/// module documentation), on exactly `workers` threads (1 runs inline; never more than there are rows of blocks). The bytes are the same for every
/// `workers`. Both sides must be multiples of `patch * merge`, which the canvas rules of both models guarantee; a zero side gives an empty tensor.
pub fn patchify<T: Copy + Default + Send + Sync>(
    rgb: &[u8],
    width: usize,
    height: usize,
    lut: &[[T; 256]; 3],
    shape: Shape,
    workers: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); tensor_len(width, height, shape)];
    patchify_into(&mut out, rgb, width, height, lut, shape, workers);
    out
}

/// Elements of the tensor of a `width x height` image: `grid_h * grid_w * patch_dim`.
pub fn tensor_len(width: usize, height: usize, shape: Shape) -> usize {
    (height / shape.patch.max(1)) * (width / shape.patch.max(1)) * shape.patch_dim()
}

/// [`patchify`] into a buffer of [`tensor_len`] elements the caller owns (every element is written, so its old content does not matter: a buffer
/// that has been used before skips the page faults of a fresh one, which cost more than the layout itself at 12 MP).
pub fn patchify_into<T: Copy + Send + Sync>(
    out: &mut [T],
    rgb: &[u8],
    width: usize,
    height: usize,
    lut: &[[T; 256]; 3],
    shape: Shape,
    workers: usize,
) {
    let Shape { patch: ps, merge: m, temporal: t } = shape;
    assert!(ps > 0 && m > 0 && t > 0, "patch, merge and temporal sizes must be positive");
    assert!(
        width % shape.factor() == 0 && height % shape.factor() == 0,
        "{width}x{height} is not a multiple of patch * merge = {}",
        shape.factor()
    );
    assert_eq!(rgb.len(), width * height * 3, "a {width}x{height} RGB image is {} bytes", width * height * 3);
    assert_eq!(
        out.len(),
        tensor_len(width, height, shape),
        "the tensor of a {width}x{height} image has {} elements",
        tensor_len(width, height, shape)
    );
    // one row of merge blocks (m rows of patches) is a contiguous run of the output, and rows are independent
    let block_row = (width / ps) * m * shape.patch_dim();
    par_rows(out, block_row, workers, |br, dst| fill_block_row(dst, br, rgb, width, lut, shape));
}

/// Run `work` and `hash` and return both results: with `hash_aside` the layout (`work`, which is one of its own workers and starts the others) runs on a scoped thread
/// while the hash runs on the calling thread, otherwise one after the other. The two read the same bytes and write nothing shared, so the order and
/// the overlap cannot change a result. The hash is a serial chain (about 2 ns a byte on a Cortex-X925 core, 3 on an A725), the longest single stage
/// of a large image, which is why it does not wait for the layout; it stays on the calling thread, which the scheduler has had on a fast core
/// for as long as it has been running, rather than on a new thread of unknown placement.
pub fn beside<R: Send>(hash_aside: bool, hash: impl FnOnce() -> u64, work: impl FnOnce() -> R + Send) -> (R, u64) {
    if !hash_aside {
        let r = work();
        return (r, hash());
    }
    std::thread::scope(|s| {
        let w = s.spawn(work);
        let h = hash();
        (w.join().unwrap_or_else(|e| std::panic::resume_unwind(e)), h)
    })
}

/// One row of blocks: `dst` holds its `grid_w / merge` blocks, each `merge * merge` patches of `patch_dim` elements, patch order (row in block,
/// column in block); block row `br` reads source pixel rows `br * merge * patch ..`.
fn fill_block_row<T: Copy>(dst: &mut [T], br: usize, rgb: &[u8], width: usize, lut: &[[T; 256]; 3], shape: Shape) {
    let Shape { patch: ps, merge: m, temporal: t } = shape;
    let pd = shape.patch_dim();
    let stride = width * 3;
    for (bc, block) in dst.chunks_exact_mut(m * m * pd).enumerate() {
        for (mr, patch_row) in block.chunks_exact_mut(m * pd).enumerate() {
            let py = (br * m + mr) * ps;
            for (mc, patch) in patch_row.chunks_exact_mut(pd).enumerate() {
                let px = (bc * m + mc) * ps;
                fill_patch(patch, &rgb[(py * width + px) * 3..], stride, lut, ps, t);
            }
        }
    }
}

/// One patch: `src` starts at its first pixel and its rows are `stride` bytes apart. The first temporal slot of each channel is written pixel by
/// pixel, the other slots copy it.
#[inline]
fn fill_patch<T: Copy>(patch: &mut [T], src: &[u8], stride: usize, lut: &[[T; 256]; 3], ps: usize, t: usize) {
    let per = ps * ps;
    let (c0, rest) = patch.split_at_mut(t * per);
    let (c1, c2) = rest.split_at_mut(t * per);
    // each channel region is `t` slots of `ps` rows of `ps` elements; the rows of the first slot come first
    let rows = c0.chunks_exact_mut(ps).zip(c1.chunks_exact_mut(ps)).zip(c2.chunks_exact_mut(ps)).take(ps);
    for (r, ((o0, o1), o2)) in rows.enumerate() {
        let pixels = &src[r * stride..r * stride + ps * 3];
        for (((a, b), c), px) in o0.iter_mut().zip(o1.iter_mut()).zip(o2.iter_mut()).zip(pixels.chunks_exact(3)) {
            *a = lut[0][px[0] as usize];
            *b = lut[1][px[1] as usize];
            *c = lut[2][px[2] as usize];
        }
    }
    for channel in [c0, c1, c2] {
        let (first, copies) = channel.split_at_mut(per);
        for slot in copies.chunks_exact_mut(per) {
            slot.copy_from_slice(first);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resize::reference::{noise, Rng};
    use crate::{f32_to_f16_bits, pixel_hash, preprocess, preprocess_with, reference, Preprocessing, Rgb, VisionError};

    const THREADS: [usize; 7] = [1, 2, 3, 5, 8, 16, 64];

    fn img(w: usize, h: usize, seed: u64) -> Rgb {
        Rgb { width: w, height: h, data: noise(w * h * 3, seed) }
    }

    /// A Qwen-style preprocessing for the layout `(ps, m, t)` with the given normalisation, set so that an image whose sides are multiples of
    /// `ps * m` is not resized by `smart_resize` (no minimum, no maximum): the oracle then lays out exactly the pixels it is given.
    fn config(ps: usize, m: usize, t: usize, mean: [f32; 3], std: [f32; 3]) -> Preprocessing {
        Preprocessing {
            patch_size: ps,
            temporal_patch_size: t,
            merge_size: m,
            min_pixels: 1,
            max_pixels: usize::MAX / 8,
            mean,
            std,
        }
    }

    fn random_norm(r: &mut Rng) -> ([f32; 3], [f32; 3]) {
        let mut v = || (r.below(2000) as f32) / 1000.0 - 0.5;
        let mean = [v(), v(), v()];
        let mut s = || 0.01 + (r.below(3000) as f32) / 1000.0;
        (mean, [s(), s(), s()])
    }

    #[test]
    fn the_qwen_table_is_the_scalar_function_for_every_channel_and_byte() {
        let rescale = (1.0f64 / 255.0) as f32;
        let mut r = Rng::new(7);
        let mut sets = vec![
            ([0.5f32; 3], [0.5f32; 3]),
            // the CLIP statistics (GLM's), where mean and std differ per channel and no value is a short binary fraction
            (
                [0.48145466f64 as f32, 0.4578275f64 as f32, 0.40821073f64 as f32],
                [0.26862954f64 as f32, 0.26130258f64 as f32, 0.27577711f64 as f32],
            ),
            ([0.0; 3], [1.0; 3]),
            ([0.0; 3], [0.0001; 3]), // values far beyond the fp16 range: infinities
            ([1.0; 3], [0.003; 3]),  // negative infinities and the subnormal edge
        ];
        sets.extend((0..40).map(|_| random_norm(&mut r)));
        let mut checked = 0;
        for (mean, std) in sets {
            let p = config(16, 2, 2, mean, std);
            let lut = p.lut_f16();
            for c in 0..3 {
                for v in 0..256usize {
                    // the expression of the per-sample code, written out again here
                    let want = f32_to_f16_bits(((v as u8 as f32) * rescale - mean[c]) / std[c]);
                    assert_eq!(lut[c][v], want, "mean {mean:?} std {std:?} channel {c} byte {v}");
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 45 * 768);
    }

    #[test]
    fn an_image_with_every_byte_in_every_channel_lays_out_like_the_oracle() {
        // 64x64 pixels: pixel i has bytes (i, i + 85, i + 171) mod 256, so each channel sees all 256 values 16 times
        let (w, h) = (64usize, 64usize);
        let data: Vec<u8> = (0..w * h).flat_map(|i| [i as u8, (i + 85) as u8, (i + 171) as u8]).collect();
        let image = Rgb { width: w, height: h, data };
        for (mean, std) in [([0.5f32; 3], [0.5f32; 3]), ([0.1, 0.4, 0.7], [0.3, 0.2, 0.9])] {
            let p = config(16, 2, 2, mean, std);
            let want = reference::preprocess(&image, &p).unwrap();
            for t in THREADS {
                assert_eq!(patchify(&image.data, w, h, &p.lut_f16(), p.shape(), t), want.patches, "{t} threads");
            }
            assert_eq!(preprocess(&image, &p).unwrap(), want);
        }
    }

    #[test]
    fn the_layout_equals_the_verbatim_oracle_on_random_shapes_and_every_thread_count() {
        let mut r = Rng::new(11);
        // (patch, merge, temporal): the Qwen layout, the GLM one, and others that move every index (odd sizes, one, three or five frames)
        let layouts = [
            (16usize, 2usize, 2usize),
            (14, 2, 2),
            (1, 1, 1),
            (1, 2, 1),
            (2, 1, 2),
            (3, 2, 3),
            (5, 3, 1),
            (7, 1, 4),
            (8, 4, 2),
            (4, 2, 5),
        ];
        let mut shapes = 0;
        let mut multi_row = 0;
        for round in 0..320 {
            let (ps, m, t) = layouts[round % layouts.len()];
            let f = ps * m;
            // sides in units of the factor: 1 to 9, and every tenth shape is thin or tall (64 to 73 units: a row of blocks for each of 64 threads)
            let (mut wu, mut hu) = (1 + r.below(9), 1 + r.below(9));
            match round % 10 {
                3 => (wu, hu) = (1, 64 + r.below(10)),
                7 => (wu, hu) = (64 + r.below(10), 1),
                _ => {}
            }
            let (w, h) = (wu * f, hu * f);
            if w.max(h) > 200 * w.min(h) {
                continue;
            }
            let (mean, std) = random_norm(&mut r);
            let p = config(ps, m, t, mean, std);
            let image = img(w, h, round as u64 + 1);
            let want = reference::preprocess(&image, &p).unwrap();
            assert_eq!(
                (want.size, want.grid.1, want.grid.2),
                ((w, h), h / ps, w / ps),
                "the oracle must not resize {w}x{h} at {ps}/{m}/{t}"
            );
            let lut = p.lut_f16();
            for workers in THREADS {
                let got = patchify(&image.data, w, h, &lut, p.shape(), workers);
                assert_eq!(got, want.patches, "{w}x{h} layout {ps}/{m}/{t} on {workers} threads");
                let aside = workers % 2 == 0;
                let (both, hash) =
                    beside(aside, || pixel_hash(w, h, &image.data), || patchify(&image.data, w, h, &lut, p.shape(), workers));
                assert_eq!(
                    (both, hash),
                    (want.patches.clone(), want.pixel_hash),
                    "{w}x{h} {workers} threads, hash beside: {aside}"
                );
            }
            for cap in [0usize, 1, 2, 7] {
                assert_eq!(preprocess_with(&image, &p, cap).unwrap(), want, "{w}x{h} layout {ps}/{m}/{t}, cap {cap}");
            }
            multi_row += (h / f >= 64) as usize;
            shapes += 1;
        }
        eprintln!("{shapes} shapes, {multi_row} with at least 64 rows of blocks");
        assert!(shapes >= 300, "{shapes} shapes");
        assert!(multi_row >= 16, "{multi_row} shapes with at least 64 rows of blocks (a row for each of 64 threads)");
    }

    #[test]
    fn the_f32_table_lays_out_like_the_u16_one() {
        // the same function over f32: the element at each place is the table entry of the same sample, whatever the element type
        let mut r = Rng::new(3);
        for _ in 0..20 {
            let (ps, m, t) = [(14usize, 2usize, 2usize), (3, 2, 3), (2, 1, 1)][r.below(3)];
            let (w, h) = ((1 + r.below(6)) * ps * m, (1 + r.below(6)) * ps * m);
            let image = img(w, h, r.next());
            let shape = Shape { patch: ps, merge: m, temporal: t };
            let l16: [[u16; 256]; 3] = std::array::from_fn(|c| std::array::from_fn(|v| (c * 1000 + v) as u16));
            let l32: [[f32; 256]; 3] = l16.map(|a| a.map(|v| v as f32));
            for workers in [1usize, 3, 64] {
                let a = patchify(&image.data, w, h, &l16, shape, workers);
                let b = patchify(&image.data, w, h, &l32, shape, workers);
                assert_eq!(a.len(), b.len());
                assert!(a.iter().zip(&b).all(|(x, y)| *x as f32 == *y), "{w}x{h} {ps}/{m}/{t}");
            }
        }
    }

    #[test]
    fn the_hash_is_the_oracles_and_runs_beside_the_layout_from_the_threshold() {
        let image = img(96, 64, 5);
        let p = config(16, 2, 2, [0.5; 3], [0.5; 3]);
        assert_eq!(pixel_hash(96, 64, &image.data), reference::preprocess(&image, &p).unwrap().pixel_hash);
        // the size is part of the hash: the same bytes read as another shape hash differently
        assert_ne!(pixel_hash(96, 64, &image.data), pixel_hash(64, 96, &image.data));
        assert_ne!(pixel_hash(96, 64, &image.data), pixel_hash(96, 64, &image.data[1..]));
        assert!(!hash_on_a_thread(1, 1 << 30), "a cap of one thread is inline");
        assert!(!hash_on_a_thread(8, MIN_PIXELS_FOR_HASH_THREAD - 1));
        assert!(hash_on_a_thread(2, MIN_PIXELS_FOR_HASH_THREAD));
    }

    #[test]
    fn the_worker_rule_gives_one_worker_per_block_of_pixels_up_to_the_cap() {
        let u = MIN_PIXELS_PER_WORKER;
        assert_eq!(layout_workers(8, 0), 1);
        assert_eq!(layout_workers(8, 2 * u - 1), 1, "under two workers' worth: inline");
        assert_eq!(layout_workers(8, 2 * u), 2);
        assert_eq!(layout_workers(8, 5 * u + 3), 5);
        assert_eq!(layout_workers(8, 100 * u), 8, "capped");
        assert_eq!(layout_workers(1, 100 * u), 1);
        assert_eq!(layout_workers(0, 100 * u), 1, "a cap of 0 still runs");
    }

    #[test]
    fn the_hash_thread_counts_against_the_cap() {
        let big = 100 * MIN_PIXELS_PER_WORKER;
        for cap in [1usize, 2, 3, 8, 64] {
            let with = plan(cap, big, true);
            let without = plan(cap, big, false);
            assert_eq!(with, Plan { workers: if cap > 1 { cap - 1 } else { 1 }, hash_aside: cap > 1 }, "cap {cap}");
            assert_eq!(without, Plan { workers: cap, hash_aside: false }, "cap {cap}, no hash");
            assert!(with.workers + with.hash_aside as usize <= cap, "cap {cap}");
        }
        // under the thresholds nothing is spawned, hash or not
        assert_eq!(plan(8, MIN_PIXELS_PER_WORKER, true), Plan { workers: 1, hash_aside: true });
        assert_eq!(plan(8, MIN_PIXELS_FOR_HASH_THREAD - 1, true), Plan { workers: 1, hash_aside: false });
    }

    #[test]
    fn the_entries_equal_the_oracle_on_300_odd_images_and_refuse_the_same_ones() {
        let p = Preprocessing::qwen38();
        let mut r = Rng::new(21);
        let (mut ok, mut refused, mut resized) = (0, 0, 0);
        for i in 0..320 {
            // every side from 1 to 300, with thin and tall shapes and sides on the factor and just beside it
            let (w, h) = match i % 8 {
                0 => (1 + r.below(300), 1 + r.below(300)),
                1 => (1, 1 + r.below(400)),
                2 => (1 + r.below(400), 1),
                3 => (32 * (1 + r.below(9)) + r.below(3) - 1, 32 * (1 + r.below(9)) + r.below(3) - 1),
                4 => (32, 32 + r.below(900)),
                5 => (32 + r.below(900), 32),
                6 => (31 + r.below(60), 31 + r.below(60)),
                _ => (64 + r.below(250), 64 + r.below(250)),
            };
            let image = img(w, h, i as u64);
            let want = reference::preprocess(&image, &p);
            for cap in [0usize, 1, 3] {
                assert_eq!(preprocess_with(&image, &p, cap), want, "{w}x{h} cap {cap}");
            }
            match &want {
                Ok(pre) => {
                    ok += 1;
                    resized += (pre.size != (w, h)) as usize;
                }
                Err(VisionError::Unsupported(_)) => refused += 1,
                Err(e) => panic!("{w}x{h}: {e}"),
            }
        }
        eprintln!("{ok} odd images processed ({resized} of them resized), {refused} refused");
        assert!(ok >= 150 && refused >= 40 && resized >= 100, "{ok} images processed ({resized} resized), {refused} refused");
    }

    #[test]
    fn large_images_take_the_threaded_layout_and_the_hash_thread_and_give_the_oracles_bytes() {
        let p = Preprocessing::qwen38();
        // from the first size that gets a second worker and a hash thread up to what a phone gives under a small budget; some resize, some do not
        for (w, h) in [(384usize, 352usize), (512, 512), (333, 517), (640, 480), (1024, 640), (700, 1100)] {
            let image = img(w, h, (w * h) as u64);
            let want = reference::preprocess(&image, &p).unwrap();
            let pixels = want.size.0 * want.size.1;
            assert!(pixels >= 2 * MIN_PIXELS_PER_WORKER && hash_on_a_thread(2, pixels), "{w}x{h} is too small to thread");
            for cap in THREADS {
                assert_eq!(preprocess_with(&image, &p, cap).unwrap(), want, "{w}x{h} cap {cap}");
                // the entry asked for one layout worker per block of pixels, and for the hash beside them whenever a second thread is allowed; the
                // hash thread counts against the cap, so the layout has at most one worker fewer (all of it, when the cap is one thread)
                let workers = (pixels / MIN_PIXELS_PER_WORKER).clamp(1, if cap > 1 { cap - 1 } else { 1 });
                assert_eq!(last_plan(), Some(Plan { workers, hash_aside: cap > 1 }), "{w}x{h} cap {cap}");
                assert!(
                    workers + (cap > 1) as usize <= cap.max(1),
                    "{workers} layout workers and the hash thread under a cap of {cap}"
                );
            }
        }
        // the smallest Qwen canvas is 256x256 (the minimum of 65,536 pixels): under two workers' worth, so the layout is inline, and exactly at the
        // hash threshold, so the hash runs beside it whenever a second thread is allowed
        let small = img(64, 64, 1);
        for cap in [0usize, 1, 8, 64] {
            preprocess_with(&small, &p, cap).unwrap();
            let allowed = if cap == 0 { crate::resize::default_workers() } else { cap };
            assert_eq!(last_plan(), Some(Plan { workers: 1, hash_aside: allowed > 1 }), "a 256x256 canvas with cap {cap}");
        }
    }
}
