//! `torch.nn.functional.interpolate(uint8, mode="bicubic", antialias=True)` for 8-bit RGB, which is what `torchvision.transforms.v2.functional.resize`
//! runs on a uint8 tensor and so what the HF torchvision image processors (`Glm5NextImageProcessor` and most of the family) use. `Profile::Hf` of the
//! GLM preprocessing resizes with it.
//!
//! **Attribution.** This is a port (a re-expression in Rust) of the weight and precision logic of PyTorch's `aten/src/ATen/native/cpu/UpSampleKernel.cpp`
//! (`HelperInterpBase::_compute_index_ranges_int16_weights`, `_compute_indices_min_size_weights_aa`, `basic_loop_separable_1d_*`) and of the
//! `cubic_convolution1/2` of `aten/src/ATen/native/UpSample.h`. PyTorch is BSD-3-Clause, Copyright (c) Facebook, Inc. and its affiliates, and the other
//! holders named in its licence; the notice and the full licence text are in `THIRD_PARTY_NOTICES.md` ("PyTorch license text"), as `docs/UPSTREAMS.md`
//! requires of a BSD-3-Clause port. The threading, the loop structure, the policy and the tests are REXL3's own.
//!
//! **The algorithm.** It is Pillow's resampler (see `resize.rs`: the same bicubic filter with a = -0.5, the same support widened by the downscale factor,
//! the same per-pixel window and normalisation, a horizontal pass and then a vertical pass with a uint8 intermediate) with a different fixed-point
//! format: the coefficients are `i16` at a precision chosen per axis, not Pillow's `i32` at 22 bits. For one axis, with `wt_max` the largest normalised
//! weight over all output pixels, the precision is the smallest `p` for which `(int)(0.5 + wt_max * 2^(p+1)) >= 2^15` (capped at 22), the coefficients
//! are `round_half_away(w * 2^p)`, and an output is `clamp((sum + 2^(p-1)) >> p, 0, 255)`. Two things follow that the Pillow code does not do: the result
//! differs from Pillow's in a few percent of pixels by one level (more on strong downscales, where `p` is 17 or more and the 15-bit headroom is spent on
//! a wider window), and an axis whose size does not change is skipped (a pass over it would give its input back).
//!
//! The AVX2 and NEON kernels PyTorch dispatches to for channels-last 3-channel tensors compute the same thing (same weights, same precision, 32-bit
//! accumulation); the generic kernel for contiguous tensors is not used here because it disagrees with them in degenerate cases (a 1-pixel-wide
//! destination). Verified against `tvF.resize` on a channels-last uint8 tensor (the layout the HF processors pass) by `tests/glm_fixtures.rs`
//! (`torch_resize_matches_tvf_resize`: 190 shapes, on every thread count) and through the whole HF processor on 59 named and 253 sweep images.
//!
//! **Threads.** Both passes run on scoped threads over disjoint row ranges through `resample::par_rows`, exactly as the Pillow passes do: the horizontal
//! pass makes an output row from one source row, the vertical pass makes an output row from a fixed window of intermediate rows, so rows are independent
//! and the bytes are the same for every thread count (the tests compare the threaded kernel with a serial reference, shape by shape). How many threads
//! a call uses is policy, kept apart from the kernel and shared with the Pillow resize rather than copied: [`bicubic_rgb_torch_threads`] is the kernel
//! with an exact count, [`bicubic_rgb_torch_with`] adds `resize::resize_workers` (one worker per `resize::MIN_WORK_PER_WORKER` multiply-accumulates,
//! so small images run inline) under a cap that is the caller's (`GlmPreprocessing.threads`) or else `resize::default_workers`
//! (`REXL3_VISION_THREADS`, else `min(8, cores)`).

use crate::resample::par_rows;
use crate::resize::{default_workers, resize_workers};
use std::borrow::Cow;

/// The cubic convolution kernel `aa_filter<double, true>` of ATen (a = -0.5), in ATen's operation order (it differs from Pillow's factored form in the
/// last bits of the double).
fn aa_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// One axis of the resample: per output pixel the first source index and the number of taps, and the `i16` coefficients (`ksize` slots a pixel).
struct Axis {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    kk: Vec<i16>,
    precision: u32,
}

/// The weights of an axis resized from `in_size` to `out_size` (both at least 1): `_compute_indices_min_size_weights_aa`, then the precision and the
/// integer weights of `_compute_index_ranges_int16_weights`.
fn axis(in_size: usize, out_size: usize) -> Axis {
    let scale = in_size as f64 / out_size as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let ksize = support.ceil() as usize * 2 + 1;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut bounds = Vec::with_capacity(out_size);
    let mut w = vec![0f64; out_size * ksize];
    let mut wt_max = 0.0f64;
    for i in 0..out_size {
        let center = scale * (i as f64 + 0.5);
        // C++ `static_cast<int64_t>(x + 0.5)` truncates toward zero; `as i64` does too
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xsize = (((center + support + 0.5) as i64).min(in_size as i64) - xmin).clamp(0, ksize as i64);
        let row = &mut w[i * ksize..(i + 1) * ksize];
        let mut total = 0.0f64;
        for (j, slot) in row.iter_mut().take(xsize as usize).enumerate() {
            let v = aa_filter(((j as i64 + xmin) as f64 - center + 0.5) * invscale);
            *slot = v;
            total += v;
        }
        if total != 0.0 {
            for slot in row.iter_mut().take(xsize as usize) {
                *slot /= total;
                wt_max = wt_max.max(*slot);
            }
        }
        bounds.push((xmin as usize, xsize as usize));
    }
    let mut precision = 0u32;
    while precision < 22 {
        let next = (0.5 + wt_max * (1u64 << (precision + 1)) as f64) as i64;
        if next >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    let kk = w
        .iter()
        .map(|&x| {
            let v = x * (1u64 << precision) as f64;
            // `(v < 0) ? (int)(-0.5 + v) : (int)(0.5 + v)`, stored into an int16
            (if v < 0.0 { (-0.5 + v) as i32 } else { (0.5 + v) as i32 }) as i16
        })
        .collect();
    Axis { ksize, bounds, kk, precision }
}

#[inline]
fn clip8(acc: i32, precision: u32) -> u8 {
    (acc >> precision).clamp(0, 255) as u8
}

/// One row of the horizontal pass: output pixel `xx` is the `n` source pixels from `xmin` weighted by its coefficients, the three channels summed
/// together (three independent i32 sums over the same taps) with the rounding bias `2^(p-1)`, then shifted down and clipped.
fn horizontal_row(row: &[u8], hz: &Axis, dst: &mut [u8]) {
    let init = 1i32 << (hz.precision - 1);
    for (xx, d) in dst.chunks_exact_mut(3).enumerate() {
        let (xmin, n) = hz.bounds[xx];
        let k = &hz.kk[xx * hz.ksize..xx * hz.ksize + n];
        let px = &row[xmin * 3..(xmin + n) * 3];
        let mut ss = [init; 3];
        for (p, &kx) in px.chunks_exact(3).zip(k) {
            let kx = kx as i32;
            ss[0] += p[0] as i32 * kx;
            ss[1] += p[1] as i32 * kx;
            ss[2] += p[2] as i32 * kx;
        }
        d[0] = clip8(ss[0], hz.precision);
        d[1] = clip8(ss[1], hz.precision);
        d[2] = clip8(ss[2], hz.precision);
    }
}

/// Columns of a vertical-pass row accumulated together: 1 KiB of i32 sums stays in registers and L1 across the taps (as in `resize.rs`).
const VERTICAL_BLOCK: usize = 256;

/// One row of the vertical pass: output row `yy` is the window of intermediate rows `ymin - first ..` (`line` bytes each) weighted by its coefficients,
/// summed with the bias in i32 and clipped. The taps are the outer loop and the columns the inner one, so every column's sum is built in a vector of
/// accumulators: the same integer additions as one column at a time (integer addition does not depend on order, and `255 * sum|k| + 2^(p-1) < 2^31`
/// for every coefficient set, so no partial sum overflows; `the_accumulator_cannot_overflow_in_any_order`).
fn vertical_row(mid: &[u8], line: usize, first: usize, vt: &Axis, yy: usize, dst: &mut [u8]) {
    let (ymin, n) = vt.bounds[yy];
    let k = &vt.kk[yy * vt.ksize..yy * vt.ksize + n];
    let base = (ymin - first) * line;
    let init = 1i32 << (vt.precision - 1);
    for (b, d) in dst.chunks_mut(VERTICAL_BLOCK).enumerate() {
        let col = b * VERTICAL_BLOCK;
        let mut sums = [init; VERTICAL_BLOCK];
        let sums = &mut sums[..d.len()];
        for (y, &ky) in k.iter().enumerate() {
            let ky = ky as i32;
            let src = &mid[base + y * line + col..][..d.len()];
            for (s, &p) in sums.iter_mut().zip(src) {
                *s += p as i32 * ky;
            }
        }
        for (o, &s) in d.iter_mut().zip(sums.iter()) {
            *o = clip8(s, vt.precision);
        }
    }
}

/// The horizontal pass over the source rows `first..last` (`sw` pixels each): `dw` pixels a row, rows split over `workers` threads.
fn horizontal_pass(src: &[u8], sw: usize, first: usize, last: usize, hz: &Axis, dw: usize, workers: usize) -> Vec<u8> {
    let mut out = vec![0u8; dw * (last - first) * 3];
    par_rows(&mut out, dw * 3, workers, |ty, dst| {
        let y = first + ty;
        horizontal_row(&src[y * sw * 3..(y + 1) * sw * 3], hz, dst);
    });
    out
}

/// The vertical pass over `mid`, whose first row is source row `first` (`line` bytes a row): `dh` output rows split over `workers` threads.
fn vertical_pass(mid: &[u8], line: usize, first: usize, vt: &Axis, dh: usize, workers: usize) -> Vec<u8> {
    let mut out = vec![0u8; line * dh];
    par_rows(&mut out, line, workers, |yy, dst| vertical_row(mid, line, first, vt, yy, dst));
    out
}

/// The weights of one resize and the source rows its vertical pass reads. An axis that keeps its size has no weights: it is not resampled.
struct Plan {
    hz: Option<Axis>,
    vt: Option<Axis>,
    /// The source rows `first..last` are the ones the vertical pass reads (all of them when the height is kept). For the weights this kernel makes the
    /// first window starts at row 0 and the last one reaches the end, so this is always `0..sh`; the plan keeps the general form of `resize.rs`.
    first: usize,
    last: usize,
}

impl Plan {
    /// For a resize that changes the size (all four sizes at least 1).
    fn new(sw: usize, sh: usize, dw: usize, dh: usize) -> Self {
        let vt = (dh != sh).then(|| axis(sh, dh));
        let (first, last) = vt.as_ref().map_or((0, sh), |vt| (vt.bounds[0].0, vt.bounds[dh - 1].0 + vt.bounds[dh - 1].1));
        Plan { hz: (dw != sw).then(|| axis(sw, dw)), vt, first, last }
    }

    /// Multiply-accumulates of the passes that run (three channels): the unit `resize::resize_workers` divides into threads.
    fn work(&self, dw: usize) -> usize {
        let taps = |a: &Option<Axis>| a.as_ref().map_or(0, |a| a.bounds.iter().map(|b| b.1).sum::<usize>());
        3 * ((self.last - self.first) * taps(&self.hz) + dw * taps(&self.vt))
    }

    /// Both passes with `workers` threads each (each pass splits its rows, so never more threads than rows). The horizontal pass only makes the rows the
    /// vertical one reads; a pass over an axis that keeps its size is not run (when both keep it the caller returns the source).
    fn run(&self, src: &[u8], sw: usize, dw: usize, dh: usize, workers: usize) -> Vec<u8> {
        let (first, last) = (self.first, self.last);
        let mid: Cow<[u8]> = match &self.hz {
            Some(hz) => Cow::Owned(horizontal_pass(src, sw, first, last, hz, dw, workers)),
            None => Cow::Borrowed(&src[first * sw * 3..last * sw * 3]),
        };
        match &self.vt {
            Some(vt) => vertical_pass(&mid, dw * 3, first, vt, dh, workers),
            None => mid.into_owned(),
        }
    }
}

fn check(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) {
    assert!(sw > 0 && sh > 0 && dw > 0 && dh > 0, "a resize of {sw}x{sh} to {dw}x{dh} has an empty side");
    assert_eq!(src.len(), sw * sh * 3);
}

/// Multiply-accumulates [`bicubic_rgb_torch`] does for this resize (the passes that run, three channels): what `resize::resize_workers` divides into
/// threads. It only builds the weights, a negligible part of the resize.
pub fn torch_resize_work(sw: usize, sh: usize, dw: usize, dh: usize) -> usize {
    assert!(sw > 0 && sh > 0 && dw > 0 && dh > 0, "a resize of {sw}x{sh} to {dw}x{dh} has an empty side");
    if (sw, sh) == (dw, dh) {
        return 0;
    }
    Plan::new(sw, sh, dw, dh).work(dw)
}

#[cfg(test)]
thread_local! {
    /// The worker count the last `bicubic_rgb_torch_with` on this thread chose (the policy's answer, for the tests).
    static LAST_WORKERS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The worker count the last torch resize on this thread chose (0 before any), for the tests of the callers.
#[cfg(test)]
pub(crate) fn last_workers() -> usize {
    LAST_WORKERS.with(|w| w.get())
}

/// Resize an RGB image (3 bytes a pixel) from `sw x sh` to `dw x dh` the way torch's uint8 antialiased bicubic does, on the default number of threads
/// (`resize::default_workers`, fewer for little work). Returns the source unchanged when the size is; an axis that keeps its size is not resampled.
pub fn bicubic_rgb_torch(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    bicubic_rgb_torch_with(src, sw, sh, dw, dh, 0)
}

/// [`bicubic_rgb_torch`] on at most `max_workers` threads (0: `resize::default_workers`); a resize with little work still runs inline
/// (`resize::resize_workers`). The bytes are the same for every value.
pub fn bicubic_rgb_torch_with(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, max_workers: usize) -> Vec<u8> {
    check(src, sw, sh, dw, dh);
    if (sw, sh) == (dw, dh) {
        return src.to_vec();
    }
    let plan = Plan::new(sw, sh, dw, dh);
    let max_workers = if max_workers == 0 { default_workers() } else { max_workers };
    let workers = resize_workers(max_workers, plan.work(dw));
    #[cfg(test)]
    LAST_WORKERS.with(|w| w.set(workers));
    plan.run(src, sw, dw, dh, workers)
}

/// The kernel with an exact thread count: both passes split their rows over `min(workers, rows)` scoped threads, however little the work (no policy; the
/// tests and the speed test use it to compare thread counts). The bytes are the same for every `workers`; 1 runs inline.
pub fn bicubic_rgb_torch_threads(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, workers: usize) -> Vec<u8> {
    check(src, sw, sh, dw, dh);
    if (sw, sh) == (dw, dh) {
        return src.to_vec();
    }
    Plan::new(sw, sh, dw, dh).run(src, sw, dw, dh, workers)
}

/// The serial resize: the loops of the first, single-threaded version of the kernel (branch `glm53-vision-hf-torch-resample`, tip 9f48197, from which this module descends), with the
/// row loops written out where that version called `par_rows` on one thread. The oracle every thread count is compared with; it shares only `axis`
/// (with `aa_filter`) and `clip8` with the kernel, which the fixtures check against `tvF.resize`.
#[cfg(test)]
pub(crate) mod reference {
    use super::{axis, clip8};

    pub(crate) fn bicubic_rgb_torch(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
        assert_eq!(src.len(), sw * sh * 3);
        assert!(sw > 0 && sh > 0 && dw > 0 && dh > 0);
        if (sw, sh) == (dw, dh) {
            return src.to_vec();
        }
        // the horizontal pass only needs the rows the vertical pass reads
        let (first, last, vt) = if dh != sh {
            let vt = axis(sh, dh);
            (vt.bounds[0].0, vt.bounds[dh - 1].0 + vt.bounds[dh - 1].1, Some(vt))
        } else {
            (0, sh, None)
        };
        let tmp: Vec<u8>;
        let mid: &[u8] = if dw != sw {
            let hz = axis(sw, dw);
            let init = 1i32 << (hz.precision - 1);
            let mut buf = vec![0u8; dw * (last - first) * 3];
            for ty in 0..last - first {
                let out = &mut buf[ty * dw * 3..(ty + 1) * dw * 3];
                let y = first + ty;
                let row = &src[y * sw * 3..(y + 1) * sw * 3];
                for (xx, px) in out.chunks_exact_mut(3).enumerate() {
                    let (xmin, n) = hz.bounds[xx];
                    let k = &hz.kk[xx * hz.ksize..xx * hz.ksize + n];
                    let (mut r, mut g, mut b) = (init, init, init);
                    for (x, &kx) in k.iter().enumerate() {
                        let p = &row[(xmin + x) * 3..(xmin + x) * 3 + 3];
                        let kx = kx as i32;
                        r += p[0] as i32 * kx;
                        g += p[1] as i32 * kx;
                        b += p[2] as i32 * kx;
                    }
                    px[0] = clip8(r, hz.precision);
                    px[1] = clip8(g, hz.precision);
                    px[2] = clip8(b, hz.precision);
                }
            }
            tmp = buf;
            &tmp
        } else {
            &src[first * sw * 3..last * sw * 3]
        };
        let Some(vt) = vt else { return mid.to_vec() };
        let mut out = vec![0u8; dw * dh * 3];
        let init = 1i32 << (vt.precision - 1);
        let line = dw * 3;
        for yy in 0..dh {
            let dst = &mut out[yy * line..(yy + 1) * line];
            let (ymin, n) = vt.bounds[yy];
            let k = &vt.kk[yy * vt.ksize..yy * vt.ksize + n];
            let mut acc = vec![init; line];
            for (y, &ky) in k.iter().enumerate() {
                let src_row = &mid[(ymin - first + y) * line..(ymin - first + y + 1) * line];
                let ky = ky as i32;
                for (a, &p) in acc.iter_mut().zip(src_row) {
                    *a += p as i32 * ky;
                }
            }
            for (o, a) in dst.iter_mut().zip(acc) {
                *o = clip8(a, vt.precision);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::reference::bicubic_rgb_torch as serial_torch;
    use super::*;
    use crate::resize::reference::{noise, sweep_shapes};

    /// Thread counts of the equality checks: serial, small, not dividing the usual row counts, the default's cap, and far more than most images have rows.
    const THREADS: [usize; 7] = [1, 2, 3, 5, 8, 16, 64];

    #[test]
    fn identity_flat_and_one_axis() {
        let src: Vec<u8> = (0..4 * 3 * 3).map(|i| (i * 7) as u8).collect();
        assert_eq!(bicubic_rgb_torch(&src, 4, 3, 4, 3), src);
        let flat = vec![123u8; 10 * 7 * 3];
        for (w, h) in [(20, 14), (5, 4), (13, 9), (10, 3), (3, 7), (1, 1)] {
            assert!(bicubic_rgb_torch(&flat, 10, 7, w, h).iter().all(|&v| v == 123), "{w}x{h}");
        }
        // an unchanged axis is not resampled: resizing only the height equals resizing each column on its own
        let img: Vec<u8> = (0..9 * 5 * 3).map(|i| ((i * 37 + i / 7 * 11) % 256) as u8).collect(); // 5 wide, 9 high
        let tall = bicubic_rgb_torch(&img, 5, 9, 5, 4);
        for x in 0..5 {
            let col: Vec<u8> = (0..9).flat_map(|y| img[(y * 5 + x) * 3..(y * 5 + x) * 3 + 3].to_vec()).collect();
            let r = bicubic_rgb_torch(&col, 1, 9, 1, 4);
            for y in 0..4 {
                assert_eq!(&tall[(y * 5 + x) * 3..(y * 5 + x) * 3 + 3], &r[y * 3..y * 3 + 3], "column {x} row {y}");
            }
        }
    }

    #[test]
    fn precision_follows_the_largest_weight() {
        // identity-like axis (scale 1): the weights are [0, 1, 0, 0, 0] so wt_max = 1 and p = 14 (1 * 2^15 reaches 2^15)
        assert_eq!(axis(100, 100).precision, 14);
        // a 6.25x downscale spreads the weight over a 26-tap window: a larger p
        assert!(axis(700, 112).precision >= 17);
        // coefficients of a pixel sum to ~1 in fixed point
        for (i, o) in [(100, 37), (37, 100), (1920, 1088), (5, 700)] {
            let a = axis(i, o);
            for xx in 0..o {
                let (_, n) = a.bounds[xx];
                let s: i64 = a.kk[xx * a.ksize..xx * a.ksize + n].iter().map(|&v| v as i64).sum();
                assert!((s - (1 << a.precision)).abs() <= n as i64, "{i}->{o} pixel {xx}: sum {s}");
            }
        }
        // the precision is capped at 22, and the largest weight of every axis fits an i16 at the precision chosen
        assert_eq!(axis(65_536, 1).precision, 22);
        for (i, o) in [(1, 1), (7, 3), (100, 37), (37, 100), (4032, 2874), (6000, 3052), (3000, 112), (20_000, 7), (1, 300)] {
            let a = axis(i, o);
            assert!((14..=22).contains(&a.precision), "{i}->{o}: precision {}", a.precision);
            let worst =
                (0..o).flat_map(|xx| a.kk[xx * a.ksize..xx * a.ksize + a.bounds[xx].1].iter().map(|&v| v as i32)).max().unwrap();
            assert!(worst < 1 << 15, "{i}->{o}: largest coefficient {worst}");
        }
    }

    #[test]
    fn an_axis_that_keeps_its_size_has_one_unit_weight_and_is_not_resampled() {
        // the premise of skipping a pass whose axis keeps its size: at every pixel exactly one coefficient is nonzero, 1 << 14 at precision 14, on the
        // source pixel with the same index, so `(p * 2^14 + 2^13) >> 14 == p` for every byte
        let mut sizes: Vec<usize> = (1..=600).collect();
        sizes.extend([2874, 3008, 3024, 4032, 6000, 20_000]);
        for n in sizes {
            let a = axis(n, n);
            assert_eq!(a.precision, 14, "{n}");
            for xx in 0..n {
                let (xmin, taps) = a.bounds[xx];
                let k = &a.kk[xx * a.ksize..(xx + 1) * a.ksize];
                let hot: Vec<usize> = (0..a.ksize).filter(|&t| k[t] != 0).collect();
                assert!(
                    hot.len() == 1 && k[hot[0]] == 1 << 14 && hot[0] < taps && xmin + hot[0] == xx,
                    "{n} pixel {xx}: {hot:?}"
                );
            }
        }
        for p in 0..=255i32 {
            assert_eq!(clip8((p << 14) + (1 << 13), 14), p as u8);
        }
        // and the plan of a resize that keeps an axis has no weights for it
        let plan = Plan::new(640, 480, 640, 240);
        assert!(plan.hz.is_none() && plan.vt.is_some());
        let plan = Plan::new(640, 480, 320, 480);
        assert!(plan.hz.is_some() && plan.vt.is_none() && (plan.first, plan.last) == (0, 480));
    }

    #[test]
    fn a_pass_that_keeps_the_size_is_skipped_and_costs_no_work() {
        // 4032x3024 -> 4032x3008: only the vertical pass runs
        let full = Plan::new(4032, 3024, 4032, 3008);
        let vertical_only = 3 * 4032 * full.vt.as_ref().unwrap().bounds.iter().map(|b| b.1).sum::<usize>();
        assert_eq!(full.work(4032), vertical_only);
        assert_eq!(torch_resize_work(4032, 3024, 4032, 3008), vertical_only);
        // 640x480 -> 320x480: only the horizontal pass runs, over all 480 rows
        let wide_only = Plan::new(640, 480, 320, 480);
        assert_eq!(wide_only.work(320), 3 * 480 * wide_only.hz.as_ref().unwrap().bounds.iter().map(|b| b.1).sum::<usize>());
        assert_eq!(torch_resize_work(640, 480, 640, 480), 0);
        // both axes change: both passes count, the horizontal one over the rows the vertical one reads
        let both = Plan::new(800, 600, 700, 500);
        let (hz, vt) = (both.hz.as_ref().unwrap(), both.vt.as_ref().unwrap());
        let rows = both.last - both.first;
        assert!(rows <= 600);
        assert_eq!(
            both.work(700),
            3 * (rows * hz.bounds.iter().map(|b| b.1).sum::<usize>() + 700 * vt.bounds.iter().map(|b| b.1).sum::<usize>())
        );
        assert_eq!(torch_resize_work(800, 600, 700, 500), both.work(700));
    }

    #[test]
    fn the_vertical_windows_read_every_source_row() {
        // the first output row's window starts at source row 0 and the last one's reaches the last row, whatever the sizes (up, down, equal):
        // the rows `first..last` the horizontal pass makes are all of them
        for (sh, dh) in sweep_shapes().iter().map(|&(_, sh, _, dh)| (sh, dh)).chain([
            (1, 20_000),
            (20_000, 1),
            (65_536, 3),
            (7, 7),
            (3024, 2156),
        ]) {
            if sh == dh {
                continue;
            }
            let plan = Plan::new(1, sh, 1, dh);
            assert_eq!((plan.first, plan.last), (0, sh), "{sh} -> {dh}");
        }
    }

    #[test]
    fn the_reference_is_the_kernel_at_one_thread() {
        let src = noise(97 * 61 * 3, 3);
        assert_eq!(bicubic_rgb_torch_threads(&src, 97, 61, 64, 80, 1), serial_torch(&src, 97, 61, 64, 80));
    }

    #[test]
    fn every_thread_count_gives_the_serial_bytes_over_the_shape_sweep() {
        let mut shapes = sweep_shapes();
        // beyond the small sweep: large reductions (precision 17 and up, windows of hundreds of taps), a wide one, a tall one, a photo-sized pair
        shapes.extend([
            (2000, 1500, 112, 84),
            (1500, 2000, 84, 112),
            (3000, 400, 29, 31),
            (400, 3000, 31, 29),
            (700, 5, 112, 112),
            (5, 700, 112, 112),
            (1200, 900, 800, 600),
            (900, 1200, 1300, 1700),
            (2874, 200, 2874, 100),
            (200, 2156, 100, 2156),
        ]);
        assert!(shapes.len() >= 500, "{} shapes", shapes.len());
        // coverage the equality is worth something for: chunks under every thread count, short last chunks, fewer rows than threads, each axis
        // identity or not, both passes with at least 64 rows (so 64 threads get one row each), the precisions the weights are kept at
        let (mut wide_rows, mut few_rows, mut hz_identity, mut vt_identity, mut up, mut down) = (0, 0, 0, 0, 0, 0);
        let mut precisions = std::collections::BTreeSet::new();
        let mut runs = 0;
        for (i, &(sw, sh, dw, dh)) in shapes.iter().enumerate() {
            let src = noise(sw * sh * 3, 100 + i as u64);
            let want = serial_torch(&src, sw, sh, dw, dh);
            if (sw, sh) != (dw, dh) {
                let p = Plan::new(sw, sh, dw, dh);
                wide_rows += (p.last - p.first >= 64 && dh >= 64) as usize;
                few_rows += (p.last - p.first < 5 || dh < 5) as usize;
                precisions.extend(p.hz.iter().chain(p.vt.iter()).map(|a| a.precision));
            }
            hz_identity += (sw == dw && sh != dh) as usize;
            vt_identity += (sh == dh && sw != dw) as usize;
            up += (dw > sw || dh > sh) as usize;
            down += (dw < sw || dh < sh) as usize;
            for t in THREADS {
                assert_eq!(bicubic_rgb_torch_threads(&src, sw, sh, dw, dh, t), want, "{sw}x{sh} -> {dw}x{dh} on {t} threads");
                runs += 1;
            }
            // the entry with the policy (whatever it picks, requested 1 to 64) is the same bytes
            for t in [0usize, 1, 7, 64] {
                assert_eq!(
                    bicubic_rgb_torch_with(&src, sw, sh, dw, dh, t),
                    want,
                    "{sw}x{sh} -> {dw}x{dh} with up to {t} workers"
                );
            }
            assert_eq!(bicubic_rgb_torch(&src, sw, sh, dw, dh), want, "{sw}x{sh} -> {dw}x{dh}");
        }
        eprintln!(
            "{} shapes x {} thread counts = {runs} threaded torch resizes equal to the serial reference; {wide_rows} with >= 64 rows in both passes, \
             {few_rows} with under 5 rows, {hz_identity} identity width / {vt_identity} identity height, {up} growing, {down} shrinking; \
             precisions seen {precisions:?}",
            shapes.len(),
            THREADS.len()
        );
        assert!(wide_rows >= 40 && few_rows >= 40 && hz_identity >= 20 && vt_identity >= 20 && up >= 100 && down >= 100);
        assert!(
            precisions.len() >= 6 && precisions.contains(&14) && precisions.iter().any(|&p| p >= 17),
            "precisions {precisions:?}"
        );
    }

    #[test]
    fn the_accumulator_cannot_overflow_in_any_order() {
        // the columns and rows of every pass accumulate in i32 with the 2^(p-1) bias: 255 * sum |k| + 2^(p-1) must stay below 2^31, so the sum is the
        // same in any order of the taps (the arithmetic is not changed by threading or loop order)
        let mut axes: Vec<(usize, usize)> = sweep_shapes().iter().flat_map(|&(sw, sh, dw, dh)| [(sw, dw), (sh, dh)]).collect();
        axes.extend([
            (20_000, 1),
            (65_536, 2),
            (1, 20_000),
            (65_535, 112),
            (112, 65_535),
            (4032, 2874),
            (6000, 3052),
            (65_536, 1),
            (65_536, 65_535),
        ]);
        let mut worst = 0i64;
        for (i, o) in axes {
            let a = axis(i, o);
            for xx in 0..o {
                let (_, n) = a.bounds[xx];
                let abs: i64 = a.kk[xx * a.ksize..xx * a.ksize + n].iter().map(|&v| (v as i64).abs()).sum();
                // the bound in units of this axis' own 2^p
                worst = worst.max((abs << 8) >> a.precision);
                assert!(
                    255 * abs + (1 << (a.precision - 1)) < 1 << 31,
                    "{i}->{o} pixel {xx}: sum of |k| {abs} at precision {}",
                    a.precision
                );
            }
        }
        eprintln!("largest sum of |k| {:.4} x 2^p", worst as f64 / 256.0);
    }

    #[test]
    fn the_torch_entry_applies_the_policy_and_the_cap() {
        let last = last_workers;
        // a thumbnail-sized resize is inline whatever the cap says
        let small = noise(100 * 100 * 3, 5);
        assert!(torch_resize_work(100, 100, 112, 112) < 2 * crate::resize::MIN_WORK_PER_WORKER);
        bicubic_rgb_torch_with(&small, 100, 100, 112, 112, 8);
        assert_eq!(last(), 1, "100x100 -> 112x112 must not spawn threads");
        // a photo-sized one gets the cap, and fewer when the cap is lower
        let big = noise(800 * 600 * 3, 6);
        let per_worker = torch_resize_work(800, 600, 700, 500) / crate::resize::MIN_WORK_PER_WORKER;
        assert!(per_worker >= 8, "{per_worker} workers' worth of work");
        let want = serial_torch(&big, 800, 600, 700, 500);
        for (cap, workers) in [(8, 8), (3, 3), (1, 1), (1000, per_worker)] {
            assert_eq!(bicubic_rgb_torch_with(&big, 800, 600, 700, 500, cap), want, "cap {cap}");
            assert_eq!(last(), workers, "800x600 -> 700x500 with a cap of {cap}");
        }
        assert_eq!(bicubic_rgb_torch(&big, 800, 600, 700, 500), want);
        assert_eq!(last(), default_workers().min(per_worker));
        // a downscale with a tiny output but a huge input (every source row is read) counts its horizontal pass: threaded
        assert!(torch_resize_work(3000, 2000, 112, 75) >= 8 * crate::resize::MIN_WORK_PER_WORKER);
        // where the inline rule sits (shared with the Pillow resize): a 160x120 -> 112x84 thumbnail stays inline, a 256x256 -> 224x224 gets a second worker
        assert_eq!(resize_workers(8, torch_resize_work(160, 120, 112, 84)), 1);
        assert!(resize_workers(8, torch_resize_work(256, 256, 224, 224)) >= 2);
        // the same MACs as the Pillow resize for the same shape: the windows are the same
        for (sw, sh, dw, dh) in [(4032, 3024, 2874, 2156), (800, 600, 700, 500), (4032, 3024, 4032, 3008), (3000, 2000, 112, 75)]
        {
            assert_eq!(torch_resize_work(sw, sh, dw, dh), crate::resize::resize_work(sw, sh, dw, dh), "{sw}x{sh} -> {dw}x{dh}");
        }
    }

    #[test]
    fn the_torch_kernel_is_not_pillows_but_is_close() {
        // the two resizers share the filter and the windows and differ in the fixed-point format: a few percent of the pixels by a level or two
        let (sw, sh, dw, dh) = (97usize, 61usize, 64usize, 80usize);
        let src = noise(sw * sh * 3, 8);
        let (t, p) = (bicubic_rgb_torch(&src, sw, sh, dw, dh), crate::resize::reference::bicubic_rgb(&src, sw, sh, dw, dh));
        let n_diff = t.iter().zip(&p).filter(|(a, b)| a != b).count();
        let worst = t.iter().zip(&p).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        assert!(n_diff > 0 && worst <= 3, "{n_diff} of {} samples differ, by at most {worst}", t.len());
    }
}
