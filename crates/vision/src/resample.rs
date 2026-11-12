//! Resampling helpers for the GLM preprocessing: Pillow's bicubic resize as the current releases run it, and the row-parallel loop the patchify uses.
//!
//! The resampler itself is `resize.rs` (a reimplementation of the algorithm of Pillow's `libImaging/Resample.c`, HPND, byte-identical to
//! `PIL.Image.resize(BICUBIC)`); this module adds the one behaviour of Pillow 12.2 and later that `resize::bicubic_rgb` does not have (see
//! [`bicubic_rgb_pil`]). Both GLM profiles resize through it. The resize runs its two passes on threads (`resize.rs`); `par_rows` below is the row
//! splitter both it and the patchify use.

/// Chunks per thread `par_rows` cuts the rows into: more than one so that a thread on a slow core (the GB10 has ten Cortex-A725 beside ten
/// Cortex-X925, and the server runs other threads) is not waited for by the rest.
const CHUNKS_PER_THREAD: usize = 8;

/// How `par_rows` splits `rows` (at least 1) rows for up to `threads` threads: the threads it uses (at least 1, never more than there are rows) and the
/// rows in a chunk (at least 1; the last chunk is the shorter remainder).
fn plan_rows(rows: usize, threads: usize) -> (usize, usize) {
    let threads = threads.clamp(1, rows);
    (threads, rows.div_ceil(threads * CHUNKS_PER_THREAD))
}

/// Run `f(row index, row)` over the rows of `buf` (`row_len` elements each) on up to `threads` scoped threads. The rows are independent, so the
/// result does not depend on the thread count or on which thread gets which rows; `threads <= 1` runs inline (and the caller is always one of the
/// threads, so `threads - 1` are spawned). The rows are cut into contiguous
/// chunks of `ceil(rows / (threads * CHUNKS_PER_THREAD))` rows (the last one shorter) that the threads take one after another, so a thread that
/// finishes early takes more; each row goes to exactly one call of `f`, with its global index.
pub(crate) fn par_rows<T: Send>(buf: &mut [T], row_len: usize, threads: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    if row_len == 0 || buf.len() < row_len {
        return;
    }
    let rows = buf.len() / row_len;
    let (threads, per) = plan_rows(rows, threads);
    if threads == 1 {
        for (i, row) in buf.chunks_exact_mut(row_len).enumerate() {
            f(i, row);
        }
        return;
    }
    let chunks = std::sync::Mutex::new(buf.chunks_mut(per * row_len).enumerate());
    let worker = || loop {
        // the lock is held only to take the next chunk
        let next = chunks.lock().unwrap().next();
        let Some((c, chunk)) = next else { break };
        for (i, row) in chunk.chunks_exact_mut(row_len).enumerate() {
            f(c * per + i, row);
        }
    };
    // the calling thread is one of the workers (a spawn and join costs about 70 us on the GB10), so `threads - 1` are spawned
    std::thread::scope(|s| {
        for _ in 1..threads {
            s.spawn(worker);
        }
        worker();
    });
}

/// Pillow's `Image.resize(size, BICUBIC)` for RGB as the current releases run it: `resize::bicubic_rgb`, except that Pillow 12.2 and later resize an
/// image taller than 100 times its width, whose height shrinks, in two steps (the vertical pass alone first, with its uint8 intermediate, then
/// the horizontal one). Pillow 12.1 and earlier, and `resize::bicubic_rgb` as used for Qwen, do both in one call (horizontal first), which gives
/// different pixels for such images. The fork's `Glm5NextVisionModel.preprocess` resizes with `Image.resize`, and both GLM profiles use this.
pub fn bicubic_rgb_pil(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    bicubic_rgb_pil_with(src, sw, sh, dw, dh, 0)
}

/// [`bicubic_rgb_pil`] on at most `max_workers` threads (0: `resize::default_workers`, which `REXL3_VISION_THREADS` overrides); each step decides
/// by its own work how many it needs (`resize::resize_workers`), so a small image runs inline. The bytes are the same for every value.
pub fn bicubic_rgb_pil_with(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, max_workers: usize) -> Vec<u8> {
    if two_step(sw, sh, dh) {
        let tall = crate::resize::bicubic_rgb_with(src, sw, sh, sw, dh, max_workers);
        crate::resize::bicubic_rgb_with(&tall, sw, dh, dw, dh, max_workers)
    } else {
        crate::resize::bicubic_rgb_with(src, sw, sh, dw, dh, max_workers)
    }
}

/// [`bicubic_rgb_pil`] with an exact thread count for the passes of both steps (`resize::bicubic_rgb_threads`: no inline rule, for the tests and the
/// speed test). The bytes are the same for every `workers`.
pub fn bicubic_rgb_pil_threads(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, workers: usize) -> Vec<u8> {
    if two_step(sw, sh, dh) {
        let tall = crate::resize::bicubic_rgb_threads(src, sw, sh, sw, dh, workers);
        crate::resize::bicubic_rgb_threads(&tall, sw, dh, dw, dh, workers)
    } else {
        crate::resize::bicubic_rgb_threads(src, sw, sh, dw, dh, workers)
    }
}

/// Pillow 12.2's rule: an image taller than 100 times its width whose height shrinks is resized in two steps.
fn two_step(sw: usize, sh: usize, dh: usize) -> bool {
    sh > sw.saturating_mul(100) && dh < sh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_rows_covers_every_row_once_for_any_thread_count() {
        for rows in [1usize, 2, 5, 7, 63, 64, 65, 100, 129, 1000] {
            for threads in [0usize, 1, 2, 3, 5, 8, 16, 100] {
                let mut buf = vec![0u32; rows * 4];
                par_rows(&mut buf, 4, threads, |i, row| {
                    for (j, v) in row.iter_mut().enumerate() {
                        *v = (i * 10 + j) as u32 + 1;
                    }
                });
                let want: Vec<u32> = (0..rows).flat_map(|i| (0..4).map(move |j| (i * 10 + j) as u32 + 1)).collect();
                assert_eq!(buf, want, "{rows} rows on {threads} threads");
            }
        }
        // an empty buffer and a zero row length are no-ops, not panics
        par_rows(&mut Vec::<u8>::new(), 4, 4, |_, _| unreachable!());
        par_rows(&mut [1u8, 2], 0, 4, |_, _| unreachable!());
    }

    #[test]
    fn pillow_two_step_for_very_tall_images() {
        // 3 x 400 reduced to 7 x 50: taller than 100x the width, so two steps; the one-call result differs
        let src: Vec<u8> = (0..3 * 400 * 3).map(|i| ((i * 2654435761usize) >> 7) as u8).collect();
        let two = bicubic_rgb_pil(&src, 3, 400, 7, 50);
        let one = crate::resize::bicubic_rgb(&src, 3, 400, 7, 50);
        assert_ne!(two, one);
        // exactly 100x is not "taller than 100x": one call; a growing height is one call too
        assert_eq!(
            bicubic_rgb_pil(&src[..3 * 300 * 3], 3, 300, 7, 50),
            crate::resize::bicubic_rgb(&src[..3 * 300 * 3], 3, 300, 7, 50)
        );
        assert_eq!(bicubic_rgb_pil(&src, 3, 400, 7, 500), crate::resize::bicubic_rgb(&src, 3, 400, 7, 500));
    }

    #[test]
    fn the_row_plan_never_uses_more_threads_than_rows_and_covers_every_row() {
        for rows in (1..=300usize).chain([1000, 4096, 65_536]) {
            for threads in (0..=70usize).chain([100, 1000, usize::MAX / 2]) {
                let (t, per) = plan_rows(rows, threads);
                let chunks = rows.div_ceil(per);
                assert!(t >= 1 && t <= rows && t <= threads.max(1), "{rows} rows, {threads} threads asked: {t} used");
                assert!(per >= 1 && per <= rows, "{rows} rows, {threads} threads: chunks of {per}");
                // the chunks tile the rows: all full but the last, which is non-empty and no longer than the others
                assert!(per * chunks >= rows && per * (chunks - 1) < rows, "{rows} rows in {chunks} chunks of {per}");
                // every thread can get a chunk, and the chunks are no more than 8 a thread
                assert!(
                    chunks >= t && chunks <= t * CHUNKS_PER_THREAD,
                    "{rows} rows, {threads} threads: {chunks} chunks of {per} for {t}"
                );
            }
        }
    }

    #[test]
    fn par_rows_gives_each_row_to_one_call_and_runs_on_that_many_threads() {
        use std::collections::HashSet;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::sync::Mutex;
        use std::time::{Duration, Instant};
        // (rows, threads): chunks of one row (rows <= 8 * threads), several rows, a short last chunk, fewer rows than threads, one thread
        for (rows, threads) in [(64usize, 8usize), (9, 8), (64, 64), (5, 64), (100, 3), (1000, 7), (7, 1), (1, 16)] {
            let mut buf = vec![0u8; rows * 3];
            let ids = Mutex::new(HashSet::new());
            let calls: Vec<AtomicUsize> = (0..rows).map(|_| AtomicUsize::new(0)).collect();
            let want_threads = threads.min(rows);
            // with one-row chunks the first `want_threads` rows are each held until all of them have started, so they can only be running together on
            // distinct threads; the rows after them take a moment, so a thread that starts late still finds some. A single thread would wait for
            // the others until the deadline (one deadline for the case; after it nobody waits any more) and the test would fail, not hang.
            let held = want_threads > 1 && rows <= 8 * want_threads;
            let arrived = AtomicUsize::new(0);
            let gave_up = AtomicBool::new(false);
            let deadline = Instant::now() + Duration::from_secs(5);
            par_rows(&mut buf, 3, threads, |i, row| {
                ids.lock().unwrap().insert(std::thread::current().id());
                calls[i].fetch_add(1, Ordering::SeqCst);
                row.fill((i as u8).wrapping_add(1));
                if held && i < want_threads {
                    arrived.fetch_add(1, Ordering::SeqCst);
                    while arrived.load(Ordering::SeqCst) < want_threads && !gave_up.load(Ordering::SeqCst) {
                        if Instant::now() > deadline {
                            gave_up.store(true, Ordering::SeqCst);
                        }
                        std::thread::yield_now();
                    }
                } else if held {
                    std::thread::sleep(Duration::from_millis(1));
                }
            });
            assert!(
                calls.iter().all(|c| c.load(Ordering::SeqCst) == 1),
                "{rows} rows on {threads} threads: every row exactly once"
            );
            let want: Vec<u8> = (0..rows).flat_map(|i| [(i as u8).wrapping_add(1); 3]).collect();
            assert_eq!(buf, want, "{rows} rows on {threads} threads");
            let used = ids.lock().unwrap().len();
            assert!(used <= want_threads, "{rows} rows on {threads} threads used {used}");
            if held {
                assert!(
                    !gave_up.load(Ordering::SeqCst),
                    "{rows} rows on {threads} threads never ran {want_threads} rows at once"
                );
                assert_eq!(used, want_threads, "{rows} rows on {threads} threads");
            }
            if threads == 1 {
                assert_eq!(used, 1);
                assert!(ids.lock().unwrap().contains(&std::thread::current().id()), "one thread runs inline");
            }
        }
    }

    #[test]
    fn the_pillow_resize_is_the_same_bytes_on_every_thread_count_including_the_two_step() {
        use crate::resize::reference::{self, noise};
        // (src w, src h, dst w, dst h): the strips take the two steps (taller than 100x the width and shrinking), the rest one call; the 100x boundary
        // is exact (not two-step), a growing strip is one call
        let shapes = [
            (3usize, 5000usize, 3usize, 40usize),
            (3, 5000, 1, 448),
            (3, 5000, 7, 4000),
            (3, 400, 7, 50),
            (3, 300, 7, 50),
            (3, 301, 7, 50),
            (1, 150, 1, 20),
            (1, 101, 1, 100),
            (2, 201, 2, 66),
            (2, 200, 2, 66),
            (10, 1500, 28, 448),
            (20, 2001, 14, 448),
            (40, 4100, 1, 17),
            (1, 4000, 90, 3999),
            (64, 64, 112, 112),
            (300, 120, 1, 1),
        ];
        let mut two_step_shapes = 0;
        for (i, &(sw, sh, dw, dh)) in shapes.iter().enumerate() {
            let src = noise(sw * sh * 3, 900 + i as u64);
            let want = reference::bicubic_rgb_pil(&src, sw, sh, dw, dh);
            two_step_shapes += (sh > sw * 100 && dh < sh) as usize;
            for t in [1usize, 2, 3, 5, 8, 16, 64] {
                assert_eq!(bicubic_rgb_pil_threads(&src, sw, sh, dw, dh, t), want, "{sw}x{sh} -> {dw}x{dh} on {t} threads");
            }
            for t in [0usize, 1, 7] {
                assert_eq!(bicubic_rgb_pil_with(&src, sw, sh, dw, dh, t), want, "{sw}x{sh} -> {dw}x{dh} with up to {t} workers");
            }
            assert_eq!(bicubic_rgb_pil(&src, sw, sh, dw, dh), want, "{sw}x{sh} -> {dw}x{dh}");
        }
        assert!(two_step_shapes >= 8, "{two_step_shapes} two-step shapes");
    }
}
