//! Pillow's `Image.resize(size, BICUBIC)` for 8-bit RGB, reproduced operation for operation (Pillow `libImaging/Resample.c`):
//! the same filter (a = -0.5, support widened by the downscale factor), the same coefficient normalization, fixed-point
//! coefficients with 22 fractional bits rounded half away from zero, a horizontal pass over only the rows the vertical pass
//! reads, then a vertical pass, each output rounded with a `1 << 21` bias and clipped to 0..=255. ExLlamaV3 resizes with exactly
//! this call, so matching it bit for bit keeps the encoder's inputs identical to the oracle's.
//!
//! Both passes run on scoped threads over disjoint row ranges (`resample::par_rows`): the horizontal pass computes an output row from one source
//! row, the vertical pass an output row from a fixed window of intermediate rows, so rows are independent and the result is the same bytes for every
//! thread count (the tests compare against a verbatim copy of the serial loops). Which thread count a call uses is policy, kept apart from the
//! kernel: [`bicubic_rgb_threads`] is the kernel with an exact count, [`bicubic_rgb_with`] adds the policy ([`resize_workers`], [`default_workers`]).

use crate::resample::par_rows;
use std::borrow::Cow;
use std::sync::OnceLock;

const PRECISION_BITS: u32 = 32 - 8 - 2;
const SUPPORT: f64 = 2.0;

/// Pillow's bicubic filter with a = -0.5.
fn bicubic_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Per output pixel: first source index, taps used, and the fixed-point coefficients (`ksize` slots a pixel).
struct Coeffs {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    kk: Vec<i32>,
}

fn precompute(in_size: usize, out_size: usize) -> Coeffs {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = SUPPORT * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * ksize];
    let mut k = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // (int)(x + 0.5) truncates toward zero; the values here are positive after the clamp below
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax_abs = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
        let xmax = xmax_abs.saturating_sub(xmin);
        let mut ww = 0.0;
        for (x, kx) in k.iter_mut().enumerate().take(xmax) {
            let w = bicubic_filter(((x + xmin) as f64 - center + 0.5) * ss);
            *kx = w;
            ww += w;
        }
        if ww != 0.0 {
            k.iter_mut().take(xmax).for_each(|kx| *kx /= ww);
        }
        k.iter_mut().skip(xmax).for_each(|kx| *kx = 0.0);
        for (x, &kx) in k.iter().enumerate() {
            let v = kx * (1u32 << PRECISION_BITS) as f64;
            // (int)(0.5 + v) for v >= 0 and (int)(-0.5 + v) for v < 0: round half away from zero via C truncation
            kk[xx * ksize + x] = if kx < 0.0 { (-0.5 + v) as i32 } else { (0.5 + v) as i32 };
        }
        bounds.push((xmin, xmax));
    }
    Coeffs { ksize, bounds, kk }
}

#[inline]
fn clip8(ss: i32) -> u8 {
    (ss >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Most worker threads the default policy uses (`min(this, available cores)`).
pub const DEFAULT_MAX_WORKERS: usize = 8;

/// Largest count `REXL3_VISION_THREADS` may ask for (more would only be clamped to the rows anyway).
const MAX_ENV_WORKERS: usize = 256;

/// Multiply-accumulates one worker must be given for another to pay for its start (`docs/VISION.md` "Cost"). Measured on a GB10 core: the kernel runs
/// at about 0.6 ns per multiply-accumulate (both passes, [`resize_work`]'s unit) and a pass that starts threads pays 130 to 300 us for it, twice a
/// resize, so a worker pays for itself from about 0.3 ms of work, 2^19 of them. A resize with less than twice this much runs inline (up to about
/// 0.6 ms serial; `resize_scaling` in `tests/glm_speed.rs` prints the times it was chosen from).
pub const MIN_WORK_PER_WORKER: usize = 1 << 19;

/// The worker count `REXL3_VISION_THREADS` and the core count give: a positive integer in the variable wins (at most 256); anything else (unset,
/// empty, zero, not a number) falls back to `min(8, cores)`.
fn parse_workers(var: Option<&str>, cores: usize) -> usize {
    match var.and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(n) if n >= 1 => n.min(MAX_ENV_WORKERS),
        _ => cores.clamp(1, DEFAULT_MAX_WORKERS),
    }
}

/// Worker threads a resize or patchify may use when the caller does not say: `REXL3_VISION_THREADS` when it is a positive integer, else
/// `min(8, available cores)`. Read once per process.
pub fn default_workers() -> usize {
    static WORKERS: OnceLock<usize> = OnceLock::new();
    *WORKERS.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        parse_workers(std::env::var("REXL3_VISION_THREADS").ok().as_deref(), cores)
    })
}

/// How many of up to `max_workers` threads a resize of `work` multiply-accumulates ([`resize_work`]) gets: one worker per
/// [`MIN_WORK_PER_WORKER`], at least 1, at most `max_workers`. So small images run inline and never pay a spawn. A pass never uses more threads
/// than it has rows (`par_rows` splits whole rows).
pub fn resize_workers(max_workers: usize, work: usize) -> usize {
    (work / MIN_WORK_PER_WORKER).clamp(1, max_workers.max(1))
}

/// One row of the horizontal pass: output pixel `xx` is the `xmax` source pixels from `xmin` weighted by its coefficients, the three channels
/// summed together (three independent i32 sums over the same taps, in the same order as one channel at a time) with the `1 << 21` bias, then clipped.
fn horizontal_row(row: &[u8], hz: &Coeffs, dst: &mut [u8]) {
    for (xx, d) in dst.chunks_exact_mut(3).enumerate() {
        let (xmin, xmax) = hz.bounds[xx];
        let k = &hz.kk[xx * hz.ksize..xx * hz.ksize + xmax];
        let px = &row[xmin * 3..(xmin + xmax) * 3];
        let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
        for (p, &kx) in px.chunks_exact(3).zip(k) {
            ss[0] += p[0] as i32 * kx;
            ss[1] += p[1] as i32 * kx;
            ss[2] += p[2] as i32 * kx;
        }
        d[0] = clip8(ss[0]);
        d[1] = clip8(ss[1]);
        d[2] = clip8(ss[2]);
    }
}

/// Columns of a vertical-pass row accumulated together: 1 KiB of i32 sums stays in registers and L1 across the taps.
const VERTICAL_BLOCK: usize = 256;

/// One row of the vertical pass: output row `yy` is the window of intermediate rows `ymin - first ..` (`rowlen` bytes each) weighted by its
/// coefficients, summed with the `1 << 21` bias in i32 and clipped. The taps are the outer loop and the columns the inner one, so the sum of every
/// column is built in a vector of accumulators: the same integer additions as one column at a time (integer addition does not depend on order, and
/// `255 * sum|k| + 2^21 < 2^31` for every coefficient set, so no partial sum overflows; `the_accumulator_cannot_overflow_in_any_order`).
fn vertical_row(tmp: &[u8], rowlen: usize, first: usize, vt: &Coeffs, yy: usize, dst: &mut [u8]) {
    let (ymin, ymax) = vt.bounds[yy];
    let k = &vt.kk[yy * vt.ksize..yy * vt.ksize + ymax];
    let base = (ymin - first) * rowlen;
    for (b, d) in dst.chunks_mut(VERTICAL_BLOCK).enumerate() {
        let col = b * VERTICAL_BLOCK;
        let mut sums = [1i32 << (PRECISION_BITS - 1); VERTICAL_BLOCK];
        let sums = &mut sums[..d.len()];
        for (y, &ky) in k.iter().enumerate() {
            let line = &tmp[base + y * rowlen + col..][..d.len()];
            for (s, &p) in sums.iter_mut().zip(line) {
                *s += p as i32 * ky;
            }
        }
        for (o, &s) in d.iter_mut().zip(sums.iter()) {
            *o = clip8(s);
        }
    }
}

/// Both coefficient tables of one resize and the source rows the vertical pass reads.
struct Plan {
    hz: Coeffs,
    vt: Coeffs,
    /// The source rows `first..last` are the ones the vertical pass reads.
    first: usize,
    last: usize,
    /// An axis that keeps its size has the coefficients `[0, 0, 1 << 22, 0, 0]` at every pixel (`precompute(n, n)`, tested for every `n` up to 2000
    /// and the large sizes), so its pass returns its input: `((p << 22) + (1 << 21)) >> 22 == p`. Pillow skips such a pass too. It is skipped here.
    hz_identity: bool,
    vt_identity: bool,
}

impl Plan {
    fn new(sw: usize, sh: usize, dw: usize, dh: usize) -> Self {
        let hz = precompute(sw, dw);
        let vt = precompute(sh, dh);
        let first = vt.bounds[0].0;
        let last = vt.bounds[dh - 1].0 + vt.bounds[dh - 1].1;
        Plan { hz, vt, first, last, hz_identity: sw == dw, vt_identity: sh == dh }
    }

    /// Multiply-accumulates of the passes that run (three channels).
    fn work(&self, dw: usize) -> usize {
        let hz_taps: usize = if self.hz_identity { 0 } else { self.hz.bounds.iter().map(|b| b.1).sum() };
        let vt_taps: usize = if self.vt_identity { 0 } else { self.vt.bounds.iter().map(|b| b.1).sum() };
        3 * ((self.last - self.first) * hz_taps + dw * vt_taps)
    }

    /// Run both passes with `workers` threads each (each pass splits its rows, so never more threads than rows). Not for an identity resize.
    fn run(&self, src: &[u8], sw: usize, dw: usize, dh: usize, workers: usize) -> Vec<u8> {
        let (hz, vt, first, last) = (&self.hz, &self.vt, self.first, self.last);
        // horizontal pass on rows first..last: row `ty` of the intermediate image comes from source row `first + ty` alone; when the width is kept
        // the intermediate image is those source rows
        let tmp: Cow<[u8]> = if self.hz_identity {
            Cow::Borrowed(&src[first * sw * 3..last * sw * 3])
        } else {
            let mut tmp = vec![0u8; dw * (last - first) * 3];
            par_rows(&mut tmp, dw * 3, workers, |ty, dst| {
                let y = first + ty;
                horizontal_row(&src[y * sw * 3..(y + 1) * sw * 3], hz, dst);
            });
            Cow::Owned(tmp)
        };
        // when the height is kept the intermediate image (all `dh` rows) is the result
        if self.vt_identity {
            debug_assert!(first == 0 && last == dh);
            return tmp.into_owned();
        }
        // vertical pass: output row `yy` comes from the fixed window of intermediate rows ymin..ymin + ymax
        let mut out = vec![0u8; dw * dh * 3];
        par_rows(&mut out, dw * 3, workers, |yy, dst| vertical_row(&tmp, dw * 3, first, vt, yy, dst));
        out
    }
}

/// Multiply-accumulates `bicubic_rgb` does for this resize (both passes, three channels): what [`resize_workers`] divides into threads. It only
/// builds the coefficient tables, a negligible part of the resize.
pub fn resize_work(sw: usize, sh: usize, dw: usize, dh: usize) -> usize {
    if (sw, sh) == (dw, dh) {
        return 0;
    }
    Plan::new(sw, sh, dw, dh).work(dw)
}

#[cfg(test)]
thread_local! {
    /// The worker count the last `bicubic_rgb_with` on this thread chose (the policy's answer, for the tests).
    static LAST_WORKERS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The worker count the last resize on this thread chose (0 before any), for the tests of the callers.
#[cfg(test)]
pub(crate) fn last_workers() -> usize {
    LAST_WORKERS.with(|w| w.get())
}

/// Resize an RGB image (3 bytes a pixel) from `sw x sh` to `dw x dh` the way Pillow's BICUBIC does, on the default number of threads
/// ([`default_workers`], fewer for little work). Returns the source unchanged when the size is.
pub fn bicubic_rgb(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    bicubic_rgb_with(src, sw, sh, dw, dh, 0)
}

/// [`bicubic_rgb`] on at most `max_workers` threads (0: [`default_workers`]); a resize with little work still runs inline ([`resize_workers`]).
/// The bytes are the same for every value.
pub fn bicubic_rgb_with(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, max_workers: usize) -> Vec<u8> {
    assert_eq!(src.len(), sw * sh * 3);
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

/// The kernel with an exact thread count: both passes split their rows over `min(workers, rows)` scoped threads, however little the work (no policy;
/// the tests and the speed test use it to compare thread counts). The bytes are the same for every `workers`; 1 runs inline.
pub fn bicubic_rgb_threads(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, workers: usize) -> Vec<u8> {
    assert_eq!(src.len(), sw * sh * 3);
    if (sw, sh) == (dw, dh) {
        return src.to_vec();
    }
    Plan::new(sw, sh, dw, dh).run(src, sw, dw, dh, workers)
}

/// The serial resize as it was before the passes were threaded, copied verbatim (the loops, their indexing and their order) from the commit that
/// preceded the threading: the oracle every thread count is compared with. It shares only `precompute` and `clip8` with the kernel, which are
/// unchanged and checked against Pillow by the fixtures.
#[cfg(test)]
pub(crate) mod reference {
    use super::{clip8, precompute, PRECISION_BITS};

    pub(crate) fn bicubic_rgb(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
        assert_eq!(src.len(), sw * sh * 3);
        if (sw, sh) == (dw, dh) {
            return src.to_vec();
        }
        let hz = precompute(sw, dw);
        let vt = precompute(sh, dh);
        // the rows of the source the vertical pass reads
        let first = vt.bounds[0].0;
        let last = vt.bounds[dh - 1].0 + vt.bounds[dh - 1].1;
        // horizontal pass on rows first..last
        let mut tmp = vec![0u8; dw * (last - first) * 3];
        for (ty, y) in (first..last).enumerate() {
            let row = &src[y * sw * 3..(y + 1) * sw * 3];
            for xx in 0..dw {
                let (xmin, xmax) = hz.bounds[xx];
                let k = &hz.kk[xx * hz.ksize..xx * hz.ksize + xmax];
                for c in 0..3 {
                    let mut ss: i32 = 1 << (PRECISION_BITS - 1);
                    for (x, &kx) in k.iter().enumerate() {
                        ss += row[(xmin + x) * 3 + c] as i32 * kx;
                    }
                    tmp[(ty * dw + xx) * 3 + c] = clip8(ss);
                }
            }
        }
        // vertical pass
        let mut out = vec![0u8; dw * dh * 3];
        for yy in 0..dh {
            let (ymin, ymax) = vt.bounds[yy];
            let k = &vt.kk[yy * vt.ksize..yy * vt.ksize + ymax];
            for x in 0..dw * 3 {
                let mut ss: i32 = 1 << (PRECISION_BITS - 1);
                for (y, &ky) in k.iter().enumerate() {
                    ss += tmp[(ymin - first + y) * dw * 3 + x] as i32 * ky;
                }
                out[yy * dw * 3 + x] = clip8(ss);
            }
        }
        out
    }

    /// The old two-step wrapper of `resample::bicubic_rgb_pil` over the old kernel.
    pub(crate) fn bicubic_rgb_pil(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
        if sh > sw.saturating_mul(100) && dh < sh {
            let tall = bicubic_rgb(src, sw, sh, sw, dh);
            bicubic_rgb(&tall, sw, dh, dw, dh)
        } else {
            bicubic_rgb(src, sw, sh, dw, dh)
        }
    }

    /// (source width, source height, destination width, destination height) of the sweep: hand-picked edge shapes first (1-pixel, thin, tall, wide, each
    /// axis identity alone, up, down, prime and awkward row counts), then random ones up to 560 in all.
    pub(crate) fn sweep_shapes() -> Vec<(usize, usize, usize, usize)> {
        let mut shapes = vec![
            (1, 1, 1, 1),
            (1, 1, 7, 5),
            (1, 1, 300, 200),
            (37, 29, 1, 1),
            (300, 200, 1, 1),
            (1, 500, 1, 17),
            (500, 1, 17, 1),
            (1, 500, 1, 500),
            (1, 500, 40, 500),
            (500, 1, 500, 40),
            (3, 5000, 3, 40),
            (3, 5000, 1, 448),
            (5000, 3, 40, 3),
            (2, 300, 5, 300),
            (300, 2, 300, 5),
            (300, 300, 300, 64),
            (300, 300, 64, 300),
            (300, 300, 300, 301),
            (300, 300, 301, 300),
            (200, 150, 130, 97),
            (97, 200, 150, 130),
            (257, 255, 255, 257),
            (211, 197, 173, 191),
            (1000, 5, 333, 5),
            (5, 1000, 5, 333),
            (64, 64, 112, 112),
            (128, 96, 96, 128),
            (100, 100, 2, 2),
            (2, 2, 100, 100),
            (131, 67, 131, 67),
            (4, 4, 3, 3),
            (4, 4, 5, 5),
            (65, 65, 64, 64),
            (65, 65, 130, 130),
        ];
        let mut r = Rng::new(0x1234_5678);
        let dim = |r: &mut Rng| match r.below(8) {
            0 => 1,
            1 => 2 + r.below(3),
            2 | 3 => 5 + r.below(36),
            4 | 5 => 41 + r.below(90),
            6 => 131 + r.below(130),
            _ => 1 + r.below(300),
        };
        while shapes.len() < 560 {
            let (sw, sh) = (dim(&mut r), dim(&mut r));
            // an axis stays the same size one time in eight, so each identity axis is met with the other one resized
            let dw = if r.below(8) == 0 { sw } else { dim(&mut r) };
            let dh = if r.below(8) == 0 { sh } else { dim(&mut r) };
            if sw * sh <= 60_000 {
                shapes.push((sw, sh, dw, dh));
            }
        }
        shapes
    }

    /// Per-pixel noise (never flat: a wrong row index or a dropped chunk must change bytes) from a seed.
    pub(crate) fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut r = Rng::new(seed);
        (0..len).map(|_| (r.next() >> 24) as u8).collect()
    }

    /// xorshift64, deterministic.
    pub(crate) struct Rng(u64);

    impl Rng {
        pub(crate) fn new(seed: u64) -> Self {
            Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
        }
        pub(crate) fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        pub(crate) fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::reference::{self, noise, sweep_shapes};
    use super::*;

    /// Thread counts of the equality checks: serial, small, not dividing the usual row counts, the default's cap, and far more than most images have rows.
    const THREADS: [usize; 7] = [1, 2, 3, 5, 8, 16, 64];

    #[test]
    fn identity_size_is_a_copy_and_flat_images_stay_flat() {
        let src: Vec<u8> = (0..4 * 3 * 3).map(|i| (i * 7) as u8).collect();
        assert_eq!(bicubic_rgb(&src, 4, 3, 4, 3), src);
        let flat = vec![123u8; 10 * 7 * 3];
        for (w, h) in [(20, 14), (5, 4), (13, 9)] {
            assert!(bicubic_rgb(&flat, 10, 7, w, h).iter().all(|&v| v == 123), "{w}x{h}");
        }
    }

    #[test]
    fn coefficients_sum_to_one_in_fixed_point() {
        for (i, o) in [(100, 37), (37, 100), (512, 512), (1920, 1088)] {
            let c = precompute(i, o);
            for xx in 0..o {
                let (_, n) = c.bounds[xx];
                let s: i64 = c.kk[xx * c.ksize..xx * c.ksize + n].iter().map(|&v| v as i64).sum();
                assert!((s - (1 << PRECISION_BITS)).abs() <= n as i64, "{i}->{o} pixel {xx}: sum {s}");
            }
        }
    }

    #[test]
    fn an_axis_that_keeps_its_size_has_identity_coefficients() {
        // the premise of skipping a pass whose axis keeps its size: at every pixel exactly one coefficient is nonzero, 1 << 22, on the source pixel with
        // the same index, so `(p * 2^22 + 2^21) >> 22 == p` for every byte
        let mut sizes: Vec<usize> = (1..=2000).collect();
        sizes.extend([2874, 3008, 3024, 4032, 6000, 20_000, 65_536]);
        for n in sizes {
            let c = precompute(n, n);
            for xx in 0..n {
                let (xmin, taps) = c.bounds[xx];
                let k = &c.kk[xx * c.ksize..(xx + 1) * c.ksize];
                let hot: Vec<usize> = (0..c.ksize).filter(|&t| k[t] != 0).collect();
                assert!(
                    hot.len() == 1 && k[hot[0]] == 1 << PRECISION_BITS && hot[0] < taps && xmin + hot[0] == xx,
                    "{n} pixel {xx}: {hot:?}"
                );
            }
        }
        for p in 0..=255i32 {
            assert_eq!(clip8((p << PRECISION_BITS) + (1 << (PRECISION_BITS - 1))), p as u8);
        }
    }

    #[test]
    fn a_pass_that_keeps_the_size_is_skipped_and_costs_no_work() {
        // 4032x3024 -> 4032x3008 (a phone photo for Qwen): only the vertical pass runs
        let full = Plan::new(4032, 3024, 4032, 3008);
        let vertical_only = 3 * 4032 * full.vt.bounds.iter().map(|b| b.1).sum::<usize>();
        assert_eq!(full.work(4032), vertical_only);
        assert_eq!(resize_work(4032, 3024, 4032, 3008), vertical_only);
        let wide_only = Plan::new(640, 480, 320, 480);
        assert_eq!(wide_only.work(320), 3 * 480 * wide_only.hz.bounds.iter().map(|b| b.1).sum::<usize>());
        assert_eq!(resize_work(640, 480, 640, 480), 0);
    }

    #[test]
    fn the_reference_is_the_kernel_at_one_thread() {
        // the verbatim old loops and the new kernel at one thread, on a shape of the fixtures' kind
        let src = noise(97 * 61 * 3, 3);
        assert_eq!(bicubic_rgb_threads(&src, 97, 61, 64, 80, 1), reference::bicubic_rgb(&src, 97, 61, 64, 80));
    }

    #[test]
    fn every_thread_count_gives_the_serial_bytes_over_the_shape_sweep() {
        let shapes = sweep_shapes();
        assert!(shapes.len() >= 500, "{} shapes", shapes.len());
        // coverage the equality is worth something for: chunks under every thread count, short last chunks, fewer rows than threads, each axis
        // identity or not, both passes with at least 64 rows (so 64 threads get one row each), the pixels all different
        let (mut wide_rows, mut few_rows, mut hz_identity, mut vt_identity, mut up, mut down) = (0, 0, 0, 0, 0, 0);
        let mut runs = 0;
        for (i, &(sw, sh, dw, dh)) in shapes.iter().enumerate() {
            let src = noise(sw * sh * 3, 100 + i as u64);
            let want = reference::bicubic_rgb(&src, sw, sh, dw, dh);
            let plan = ((sw, sh) != (dw, dh)).then(|| Plan::new(sw, sh, dw, dh));
            if let Some(p) = &plan {
                wide_rows += (p.last - p.first >= 64 && dh >= 64) as usize;
                few_rows += (p.last - p.first < 5 || dh < 5) as usize;
            }
            hz_identity += (sw == dw && sh != dh) as usize;
            vt_identity += (sh == dh && sw != dw) as usize;
            up += (dw > sw || dh > sh) as usize;
            down += (dw < sw || dh < sh) as usize;
            for t in THREADS {
                assert_eq!(bicubic_rgb_threads(&src, sw, sh, dw, dh, t), want, "{sw}x{sh} -> {dw}x{dh} on {t} threads");
                runs += 1;
            }
            // the entry with the policy (whatever it picks, requested 1 to 64) is the same bytes
            for t in [0usize, 1, 7, 64] {
                assert_eq!(bicubic_rgb_with(&src, sw, sh, dw, dh, t), want, "{sw}x{sh} -> {dw}x{dh} with up to {t} workers");
            }
        }
        eprintln!(
            "{} shapes x {} thread counts = {runs} threaded resizes equal to the serial reference; {wide_rows} with >= 64 rows in both passes, {few_rows} with under 5 rows, \
             {hz_identity} identity width / {vt_identity} identity height, {up} growing, {down} shrinking",
            shapes.len(),
            THREADS.len()
        );
        assert!(wide_rows >= 40 && few_rows >= 40 && hz_identity >= 20 && vt_identity >= 20 && up >= 100 && down >= 100);
    }

    #[test]
    fn the_accumulator_cannot_overflow_in_any_order() {
        // the columns and rows of every pass accumulate in i32 with the 2^21 bias: 255 * sum |k| + 2^21 must stay below 2^31, so the sum is
        // the same in any order of the taps (the arithmetic is not changed by threading or loop order)
        let mut axes: Vec<(usize, usize)> = sweep_shapes().iter().flat_map(|&(sw, sh, dw, dh)| [(sw, dw), (sh, dh)]).collect();
        axes.extend([(20_000, 1), (65_536, 2), (1, 20_000), (65_535, 112), (112, 65_535), (4032, 2874), (6000, 3052)]);
        let mut worst = 0i64;
        for (i, o) in axes {
            let c = precompute(i, o);
            for xx in 0..o {
                let (_, n) = c.bounds[xx];
                let abs: i64 = c.kk[xx * c.ksize..xx * c.ksize + n].iter().map(|&v| (v as i64).abs()).sum();
                worst = worst.max(abs);
                assert!(255 * abs + (1 << (PRECISION_BITS - 1)) < 1 << 31, "{i}->{o} pixel {xx}: sum of |k| {abs}");
            }
        }
        eprintln!("largest sum of |k| {worst} = {:.4} x 2^22", worst as f64 / (1u64 << PRECISION_BITS) as f64);
    }

    #[test]
    fn the_worker_environment_value_is_parsed_and_clamped() {
        // REXL3_VISION_THREADS: a positive integer wins; anything else is the default min(8, cores)
        for (var, cores, want) in [
            (None, 20, 8),
            (None, 4, 4),
            (None, 0, 1),
            (Some("3"), 20, 3),
            (Some(" 12 "), 2, 12),
            (Some("1"), 20, 1),
            (Some("0"), 20, 8),
            (Some(""), 20, 8),
            (Some("many"), 5, 5),
            (Some("-2"), 20, 8),
            (Some("2.5"), 3, 3),
            (Some("100000"), 20, MAX_ENV_WORKERS),
        ] {
            assert_eq!(parse_workers(var, cores), want, "{var:?} on {cores} cores");
        }
        assert!((1..=MAX_ENV_WORKERS).contains(&default_workers()));
    }

    #[test]
    fn default_workers_follows_the_environment_variable() {
        // `default_workers` reads the variable once per process, and a test cannot set it for itself (the others run on threads), so the test runs
        // itself in child processes with the variable set; in a child it checks the answer
        if let Ok(want) = std::env::var("REXL3_TEST_EXPECT_WORKERS") {
            assert_eq!(default_workers().to_string(), want);
            return;
        }
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let fallback = parse_workers(None, cores).to_string();
        let exe = std::env::current_exe().unwrap();
        for (value, want) in
            [("3", "3"), ("1", "1"), ("24", "24"), ("0", fallback.as_str()), ("lots", fallback.as_str()), ("", fallback.as_str())]
        {
            let status = std::process::Command::new(&exe)
                .args(["--exact", "resize::tests::default_workers_follows_the_environment_variable", "--test-threads=1"])
                .env("REXL3_VISION_THREADS", value)
                .env("REXL3_TEST_EXPECT_WORKERS", want)
                .output()
                .unwrap();
            let out = String::from_utf8_lossy(&status.stdout);
            assert!(
                status.status.success() && out.contains("1 passed"),
                "REXL3_VISION_THREADS={value:?} must give {want}:\n{out}"
            );
        }
    }

    #[test]
    fn small_work_runs_inline_and_large_work_gets_every_worker() {
        let m = MIN_WORK_PER_WORKER;
        assert_eq!(resize_workers(8, 0), 1);
        assert_eq!(resize_workers(8, m - 1), 1);
        assert_eq!(resize_workers(8, 2 * m - 1), 1, "under two workers' worth of work: inline");
        assert_eq!(resize_workers(8, 2 * m), 2);
        assert_eq!(resize_workers(8, 5 * m + 3), 5);
        assert_eq!(resize_workers(8, 100 * m), 8);
        assert_eq!(resize_workers(8, usize::MAX), 8);
        assert_eq!(resize_workers(1, 100 * m), 1);
        assert_eq!(resize_workers(0, 100 * m), 1, "a zero cap is one worker, not a panic");
        assert_eq!(resize_workers(3, 100 * m), 3);
    }

    #[test]
    fn the_resize_entry_applies_the_policy_and_the_cap() {
        let last = || LAST_WORKERS.with(|w| w.get());
        // a thumbnail-sized resize is inline whatever the cap says
        let small = noise(100 * 100 * 3, 5);
        assert!(resize_work(100, 100, 112, 112) < 2 * MIN_WORK_PER_WORKER);
        bicubic_rgb_with(&small, 100, 100, 112, 112, 8);
        assert_eq!(last(), 1, "100x100 -> 112x112 must not spawn threads");
        // a photo-sized one gets the cap, and fewer when the cap is lower
        let big = noise(800 * 600 * 3, 6);
        let per_worker = resize_work(800, 600, 700, 500) / MIN_WORK_PER_WORKER;
        assert!(per_worker >= 8, "{per_worker} workers' worth of work");
        let want = reference::bicubic_rgb(&big, 800, 600, 700, 500);
        for (cap, workers) in [(8, 8), (3, 3), (1, 1), (1000, per_worker)] {
            assert_eq!(bicubic_rgb_with(&big, 800, 600, 700, 500, cap), want, "cap {cap}");
            assert_eq!(last(), workers, "800x600 -> 700x500 with a cap of {cap}");
        }
        assert_eq!(bicubic_rgb(&big, 800, 600, 700, 500), want);
        assert_eq!(last(), default_workers().min(per_worker));
        // a downscale with a tiny output but a huge input (every source row is read) counts its horizontal pass: threaded
        assert!(resize_work(3000, 2000, 112, 75) >= 8 * MIN_WORK_PER_WORKER);
        assert_eq!(resize_work(7, 9, 7, 9), 0, "identity does no work");
        // where the inline rule sits (`MIN_WORK_PER_WORKER`, measured): a 160x120 -> 112x84 thumbnail (0.4 M MACs, 0.2 ms) stays inline, a
        // 256x256 -> 224x224 (1.5 M MACs, 0.9 ms) is worth a second worker
        assert_eq!(resize_workers(8, resize_work(160, 120, 112, 84)), 1);
        assert!(resize_workers(8, resize_work(256, 256, 224, 224)) >= 2);
    }
}
