//! The Nemotron glue kernels (`csrc/nemotron_glue.cu`) against CPU models, bit for bit: the embedding gather, the strided RMSNorm
//! (the model replays the kernel's fixed reduction tree), the in_proj split with its bf16 rounding of z, both widens, the f32 top-k
//! (a stable sort on (value desc, id asc)) and the feature splice. Every kernel is also checked row-invariant: row r has the same bits
//! in calls of 1, 7 and 64 rows. The last test prints microseconds per call. GPU tests skip when there is no device.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::Rng;
use core::ffi::c_void;
use gb_cuda::nemotron_glue as ng;
use gb_cuda::{device_available, DevBuf, Stream};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

/// CUDA events (the crate's `rt::Event` is created without timing).
mod ev {
    use core::ffi::c_void;
    extern "C" {
        pub fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
        pub fn cudaEventDestroy(e: *mut c_void) -> i32;
    }
}

/// Microseconds per call of `f`: `warm` untimed calls, then `iters` calls queued back to back between one pair of events. The call
/// index is passed so a caller can rotate buffers.
fn batch_us(st: &Stream, warm: usize, iters: usize, mut f: impl FnMut(usize)) -> f64 {
    for i in 0..warm {
        f(i);
    }
    st.sync().unwrap();
    let (mut a, mut b) = (core::ptr::null_mut::<c_void>(), core::ptr::null_mut::<c_void>());
    let mut ms = 0f32;
    unsafe {
        assert_eq!(ev::cudaEventCreateWithFlags(&mut a, 0), 0);
        assert_eq!(ev::cudaEventCreateWithFlags(&mut b, 0), 0);
        assert_eq!(ev::cudaEventRecord(a, st.raw()), 0);
    }
    for i in 0..iters {
        f(warm + i);
    }
    unsafe {
        assert_eq!(ev::cudaEventRecord(b, st.raw()), 0);
    }
    st.sync().unwrap();
    unsafe {
        assert_eq!(ev::cudaEventElapsedTime(&mut ms, a, b), 0);
        ev::cudaEventDestroy(a);
        ev::cudaEventDestroy(b);
    }
    ms as f64 * 1000.0 / iters as f64
}

fn bf16_val(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Bit-pattern equality, except that two NaNs are equal when `nan_any` (a NaN's payload is not the kernel's to preserve).
fn mismatches(got: &[f32], want: &[f32], nan_any: bool) -> usize {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).filter(|(g, w)| !(g.to_bits() == w.to_bits() || (nan_any && g.is_nan() && w.is_nan()))).count()
}

fn upload_f32(v: &[f32]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}

fn download_f32(b: &DevBuf, n: usize) -> Vec<f32> {
    let mut v = vec![0f32; n];
    b.download(&mut v).unwrap();
    v
}

// ---------------------------------------------------------------------------------------------------------------------------------
// embed_gather_f32 and splice_rows

/// `out[r] = table[ids[r]]` widened (a zero row for an id outside `0..vocab`).
fn embed_model(table: &[u16], vocab: usize, ids: &[i32]) -> Vec<f32> {
    let mut out = vec![0f32; ids.len() * ng::HIDDEN];
    for (r, &id) in ids.iter().enumerate() {
        if id >= 0 && (id as usize) < vocab {
            for c in 0..ng::HIDDEN {
                out[r * ng::HIDDEN + c] = bf16_val(table[id as usize * ng::HIDDEN + c]);
            }
        }
    }
    out
}

fn random_u16(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n).map(|_| (rng.next_u64() >> 32) as u16).collect()
}

fn run_embed(st: &Stream, table: &DevBuf, ids: &[i32]) -> Vec<f32> {
    let di = DevBuf::from_slice(ids).unwrap();
    let mut out = DevBuf::zeroed(ids.len() * ng::HIDDEN * 4).unwrap();
    // poison the output so a row the kernel forgot to write cannot pass as zero
    out.upload(&vec![f32::from_bits(0x7fc0_dead); ids.len() * ng::HIDDEN]).unwrap();
    ng::embed_gather_f32(table, &di, ids.len(), &mut out, st).unwrap();
    st.sync().unwrap();
    download_f32(&out, ids.len() * ng::HIDDEN)
}

#[test]
fn embed_gather_matches_the_model_and_gives_zero_rows_for_bad_ids() {
    gpu!();
    let st = Stream::new().unwrap();
    let vocab = 1000usize;
    let mut rng = Rng(11);
    // every 16-bit pattern appears (NaN, inf, subnormal, -0): the widening is a shift and is exact
    let table = random_u16(&mut rng, vocab * ng::HIDDEN);
    let dt = DevBuf::from_slice(&table).unwrap();
    let mut ids: Vec<i32> = (0..64).map(|_| (rng.next_u64() % vocab as u64) as i32).collect();
    ids[0] = 0;
    ids[1] = vocab as i32 - 1;
    ids[2] = -1;
    ids[3] = vocab as i32;
    ids[4] = i32::MIN;
    ids[5] = i32::MAX;
    ids[6] = vocab as i32 + 1;
    let want = embed_model(&table, vocab, &ids);
    let got = run_embed(&st, &dt, &ids);
    assert_eq!(mismatches(&got, &want, false), 0, "embed_gather_f32 differs from the model");
    // the bad rows are zero (positive zero bits), not stale
    for r in [2usize, 3, 4, 5, 6] {
        assert!(
            got[r * ng::HIDDEN..(r + 1) * ng::HIDDEN].iter().all(|v| v.to_bits() == 0),
            "row {r}: an out-of-range id is not a zero row"
        );
    }
    // row invariance: row r is the same bits in calls of 1, 7 and 64 rows
    let g7 = run_embed(&st, &dt, &ids[..7]);
    assert_eq!(mismatches(&g7, &got[..7 * ng::HIDDEN], false), 0, "rows 0..7 change bits between the 7- and 64-row calls");
    for r in 0..64 {
        let g1 = run_embed(&st, &dt, &ids[r..r + 1]);
        assert_eq!(mismatches(&g1, &got[r * ng::HIDDEN..(r + 1) * ng::HIDDEN], false), 0, "row {r} changes bits in a 1-row call");
    }
    eprintln!("embed_gather_f32: 64 rows bitwise equal to the model (7 bad ids give zero rows), rows 1, 7 and 64 identical");
}

#[test]
fn splice_rows_widens_exactly_and_touches_only_its_rows() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(23);
    let feats = random_u16(&mut rng, 20 * ng::HIDDEN);
    let df = DevBuf::from_slice(&feats).unwrap();
    let sentinel = f32::from_bits(0x3f80_1234);
    let mut dst = DevBuf::zeroed(10 * ng::HIDDEN * 4).unwrap();
    dst.upload(&vec![sentinel; 10 * ng::HIDDEN]).unwrap();
    ng::splice_rows(&df, 3, &mut dst, 2, 5, &st).unwrap();
    ng::splice_rows(&df, 19, &mut dst, 9, 1, &st).unwrap();
    ng::splice_rows(&df, 0, &mut dst, 0, 0, &st).unwrap(); // n = 0 is a no-op
    st.sync().unwrap();
    let got = download_f32(&dst, 10 * ng::HIDDEN);
    let mut want = vec![sentinel; 10 * ng::HIDDEN];
    for k in 0..5 {
        for c in 0..ng::HIDDEN {
            want[(2 + k) * ng::HIDDEN + c] = bf16_val(feats[(3 + k) * ng::HIDDEN + c]);
        }
    }
    for c in 0..ng::HIDDEN {
        want[9 * ng::HIDDEN + c] = bf16_val(feats[19 * ng::HIDDEN + c]);
    }
    assert_eq!(mismatches(&got, &want, false), 0, "splice_rows differs from the model (or wrote outside its rows)");
    eprintln!("splice_rows: rows 2..7 and row 9 equal the widened features, the other rows keep their bits");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// rmsnorm_bf16w_f32

/// One row on the CPU, replaying the kernel's tree (the header of `nemotron_glue.cu`): 256 strided partial sums, a 32-lane xor
/// butterfly per warp, then ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)); `inv = 1.0 / sqrt(mean + eps)`; `(x * inv) * w`.
fn norm_row_model(x: &[f32], w: &[u16], eps: f32, out: &mut [f32]) {
    let d = x.len();
    let mut s = [0f32; 8];
    for warp in 0..8 {
        let mut v = [0f32; 32];
        for lane in 0..32 {
            let mut acc = 0f32;
            let mut c = warp * 32 + lane;
            while c < d {
                let sq = x[c] * x[c];
                acc += sq;
                c += 256;
            }
            v[lane] = acc;
        }
        for off in [16usize, 8, 4, 2, 1] {
            let old = v;
            for lane in 0..32 {
                v[lane] = old[lane] + old[lane ^ off];
            }
        }
        // every lane holds the same bits (a + b == b + a), so the kernel may read any lane
        assert!(
            v.iter().all(|t| t.to_bits() == v[0].to_bits() || t.is_nan()),
            "the warp tree must leave one value on every lane"
        );
        s[warp] = v[0];
    }
    let total = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    let mean = total / d as f32;
    let inv = 1.0f32 / (mean + eps).sqrt();
    for c in 0..d {
        out[c] = (x[c] * inv) * bf16_val(w[c]);
    }
}

/// What the strided call writes into an `out` buffer that started as `fill`: row r's `d` elements at `r * out_stride + off`.
fn norm_model(
    x: &[f32],
    w: &[u16],
    rows: usize,
    d: usize,
    x_stride: usize,
    eps: f32,
    out_stride: usize,
    off: usize,
    fill: f32,
) -> Vec<f32> {
    let mut out = vec![fill; rows * out_stride];
    for r in 0..rows {
        let (xr, o) = (&x[r * x_stride..r * x_stride + d], &mut out[r * out_stride + off..r * out_stride + off + d]);
        norm_row_model(xr, w, eps, o);
    }
    out
}

/// Runs the kernel with `out` the whole buffer of `rows * out_stride` floats (poisoned with `fill`), or a view starting `off`
/// elements into it. Returns the whole buffer.
fn run_norm(
    st: &Stream,
    x: &[f32],
    w: &[u16],
    rows: usize,
    d: usize,
    x_stride: usize,
    eps: f32,
    out_stride: usize,
    off: usize,
    fill: f32,
) -> Vec<f32> {
    let dx = upload_f32(x);
    let dw = DevBuf::from_slice(w).unwrap();
    let total = rows * out_stride;
    let mut buf = DevBuf::zeroed_view(total * 4).unwrap();
    buf.upload(&vec![fill; total]).unwrap();
    let mut view = buf.subview(off * 4, (total - off) * 4);
    ng::rmsnorm_bf16w_f32(&dx, &dw, rows, d, x_stride, eps, &mut view, out_stride, st).unwrap();
    st.sync().unwrap();
    download_f32(&buf, total)
}

/// Row data of one class, `d` long.
fn norm_data(class: usize, d: usize, rng: &mut Rng) -> Vec<f32> {
    match class % 6 {
        0 => rng.vec(d, -3.0, 3.0),
        1 => vec![0.0; d],
        // subnormals (and a few zeros) of both signs: the sum of squares underflows to zero
        2 => {
            (0..d).map(|_| f32::from_bits((rng.next_u64() as u32 & 0x007f_ffff) | ((rng.next_u64() as u32 & 1) << 31))).collect()
        }
        // large magnitudes whose squares are still finite, 1e17 .. 1e19
        3 => (0..d).map(|_| rng.uniform(1.0e17, 1.0e19) * if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 }).collect(),
        // squares overflow: the sum is inf, inv is 0
        4 => (0..d).map(|_| rng.uniform(1.0e20, 1.0e22)).collect(),
        // a mix of scales in one row
        _ => (0..d)
            .map(|i| match i % 4 {
                0 => rng.uniform(-1.0e-30, 1.0e-30),
                1 => rng.uniform(-1.0, 1.0),
                2 => rng.uniform(-1.0e9, 1.0e9),
                _ => f32::from_bits(rng.next_u64() as u32 & 0x007f_ffff),
            })
            .collect(),
    }
}

fn random_weights(rng: &mut Rng, d: usize) -> Vec<u16> {
    rng.vec(d, -2.0, 2.0).iter().map(|v| (v.to_bits() >> 16) as u16).collect()
}

#[test]
fn rmsnorm_matches_the_tree_model_bitwise() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(101);
    let eps = 1e-5f32;
    let mut cases = 0usize;
    // d: the hidden, the other widths the model uses, and widths that are not a multiple of 256, of 32 or of 4
    for &d in &[4096usize, 8192, 2688, 5376, 1024, 1, 7, 255, 256, 257, 1000] {
        for &rows in &[1usize, 7, 64] {
            let x: Vec<f32> = (0..rows).flat_map(|r| norm_data(r, d, &mut rng)).collect();
            let w = random_weights(&mut rng, d);
            let got = run_norm(&st, &x, &w, rows, d, d, eps, d, 0, 0.0);
            let want = norm_model(&x, &w, rows, d, d, eps, d, 0, 0.0);
            let bad = mismatches(&got, &want, true);
            assert_eq!(bad, 0, "d={d} rows={rows}: {bad} elements differ from the tree model");
            cases += 1;
        }
    }
    // strided: x rows 8192 apart (the concatenated [embedding | hidden] row), out into the second half of a 8192-wide row; the
    // first half of `out` and everything between rows must keep the poison
    let (rows, d) = (5usize, 4096usize);
    let x: Vec<f32> = (0..rows * 8192).map(|_| rng.uniform(-2.0, 2.0)).collect();
    let w = random_weights(&mut rng, d);
    let poison = f32::from_bits(0x3f80_beef);
    let got = run_norm(&st, &x, &w, rows, d, 8192, eps, 8192, 4096, poison);
    let want = norm_model(&x, &w, rows, d, 8192, eps, 8192, 4096, poison);
    assert_eq!(mismatches(&got, &want, true), 0, "strided norm differs from the model, or wrote outside its half");
    // x stride larger than the out stride and an out stride of d + 3 (rows start at unaligned addresses)
    let (rows, d) = (4usize, 2688usize);
    let x: Vec<f32> = (0..rows * 5000).map(|_| rng.uniform(-2.0, 2.0)).collect();
    let w = random_weights(&mut rng, d);
    let got = run_norm(&st, &x, &w, rows, d, 5000, eps, d + 3, 0, poison);
    let want = norm_model(&x, &w, rows, d, 5000, eps, d + 3, 0, poison);
    assert_eq!(mismatches(&got, &want, true), 0, "odd-stride norm differs from the model");
    // eps matters: the same data with eps = 1.0 (the model and the kernel agree there too)
    let (rows, d) = (3usize, 4096usize);
    let x: Vec<f32> = (0..rows * d).map(|_| rng.uniform(-0.01, 0.01)).collect();
    let w = random_weights(&mut rng, d);
    let got = run_norm(&st, &x, &w, rows, d, d, 1.0, d, 0, 0.0);
    assert_eq!(mismatches(&got, &norm_model(&x, &w, rows, d, d, 1.0, d, 0, 0.0), true), 0);
    eprintln!(
        "rmsnorm_bf16w_f32: {cases} (d, rows) cases of 6 data classes plus 3 strided cases bitwise equal to the tree model"
    );
}

#[test]
fn rmsnorm_rows_are_bitwise_the_same_in_calls_of_1_7_and_64_rows() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(202);
    for &d in &[4096usize, 2688, 257] {
        let rows = 64;
        let x: Vec<f32> = (0..rows).flat_map(|r| norm_data(r, d, &mut rng)).collect();
        let w = random_weights(&mut rng, d);
        let full = run_norm(&st, &x, &w, rows, d, d, 1e-5, d, 0, 0.0);
        let r7 = run_norm(&st, &x[..7 * d], &w, 7, d, d, 1e-5, d, 0, 0.0);
        assert_eq!(mismatches(&r7, &full[..7 * d], true), 0, "d={d}: rows 0..7 change bits between the 7- and 64-row calls");
        for r in 0..rows {
            let r1 = run_norm(&st, &x[r * d..(r + 1) * d], &w, 1, d, d, 1e-5, d, 0, 0.0);
            assert_eq!(mismatches(&r1, &full[r * d..(r + 1) * d], true), 0, "d={d}: row {r} changes bits in a 1-row call");
        }
    }
    eprintln!("rmsnorm_bf16w_f32: every row identical in calls of 1, 7 and 64 rows (d = 4096, 2688, 257)");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// split_inproj

/// f32 -> the nearest bf16 (ties to even) as f32; a NaN becomes the quiet NaN 0x7fc00000.
fn round_bf16_model(v: f32) -> f32 {
    let u = v.to_bits();
    if v.is_nan() {
        return f32::from_bits(0x7fc0_0000);
    }
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
}

/// A y row: half random bit patterns (so NaN, inf, subnormal, -0 and every exponent appear), half uniform values, and the rounding
/// edge cases (exact halfway values to even and to odd, the largest finite values, and the first value that rounds to inf).
fn inproj_row(rng: &mut Rng) -> Vec<f32> {
    let mut y: Vec<f32> = (0..ng::INPROJ_COLS)
        .map(|i| if i % 2 == 0 { f32::from_bits(rng.next_u64() as u32) } else { rng.uniform(-100.0, 100.0) })
        .collect();
    let edges: [u32; 14] = [
        0x3f80_8000, // halfway between bf16 0x3f80 (even) and 0x3f81: rounds to 0x3f80
        0x3f81_8000, // halfway between 0x3f81 and 0x3f82 (even): rounds to 0x3f82
        0xbf80_8000,
        0xbf81_8000,
        0x3f80_7fff, // just below halfway
        0x3f80_8001, // just above halfway
        0x7f7f_ffff, // FLT_MAX rounds to inf
        0x7f7f_8000, // halfway between 0x7f7f (odd) and inf: rounds to inf
        0x7f7f_7fff,
        0x0000_8000, // a subnormal at halfway: rounds to zero (even)
        0x0001_8000,
        0x8000_0000, // -0
        0x7fc0_0001, // a NaN with a payload
        0xff80_0001, // a signalling NaN of negative sign
    ];
    for (k, &e) in edges.iter().enumerate() {
        // spread over z (first 8192), xbc and dt so that every part sees them
        for base in [0usize, ng::Z_COLS, ng::Z_COLS + ng::XBC_COLS] {
            y[base + 7 * k] = f32::from_bits(e);
        }
    }
    y
}

fn inproj_model(y: &[f32], rows: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut z, mut xbc, mut dt) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..rows {
        let row = &y[r * ng::INPROJ_COLS..(r + 1) * ng::INPROJ_COLS];
        z.extend(row[..ng::Z_COLS].iter().map(|&v| round_bf16_model(v)));
        xbc.extend_from_slice(&row[ng::Z_COLS..ng::Z_COLS + ng::XBC_COLS]);
        dt.extend_from_slice(&row[ng::Z_COLS + ng::XBC_COLS..]);
    }
    (z, xbc, dt)
}

fn run_split(st: &Stream, y: &[f32], rows: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dy = upload_f32(y);
    let mut z = DevBuf::zeroed(rows * ng::Z_COLS * 4).unwrap();
    let mut xbc = DevBuf::zeroed(rows * ng::XBC_COLS * 4).unwrap();
    let mut dt = DevBuf::zeroed(rows * ng::DT_COLS * 4).unwrap();
    ng::split_inproj(&dy, rows, &mut z, &mut xbc, &mut dt, st).unwrap();
    st.sync().unwrap();
    (download_f32(&z, rows * ng::Z_COLS), download_f32(&xbc, rows * ng::XBC_COLS), download_f32(&dt, rows * ng::DT_COLS))
}

#[test]
fn split_inproj_matches_the_model_and_is_row_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(303);
    let rows = 64;
    let y: Vec<f32> = (0..rows).flat_map(|_| inproj_row(&mut rng)).collect();
    let (z, xbc, dt) = run_split(&st, &y, rows);
    let (wz, wx, wd) = inproj_model(&y, rows);
    assert_eq!(mismatches(&z, &wz, false), 0, "z (bf16 grid) differs from the model");
    assert_eq!(mismatches(&xbc, &wx, false), 0, "xbc is not an exact copy");
    assert_eq!(mismatches(&dt, &wd, false), 0, "dt is not an exact copy");
    // z really is on the bf16 grid: the low 16 bits are zero
    assert!(z.iter().all(|v| v.to_bits() & 0xffff == 0), "a z value is off the bf16 grid");
    for &n in &[7usize, 1] {
        let (z7, x7, d7) = run_split(&st, &y[..n * ng::INPROJ_COLS], n);
        assert_eq!(mismatches(&z7, &z[..n * ng::Z_COLS], false), 0, "z rows change bits in a {n}-row call");
        assert_eq!(mismatches(&x7, &xbc[..n * ng::XBC_COLS], false), 0);
        assert_eq!(mismatches(&d7, &dt[..n * ng::DT_COLS], false), 0);
    }
    for r in [5usize, 33, 63] {
        let (z1, x1, d1) = run_split(&st, &y[r * ng::INPROJ_COLS..(r + 1) * ng::INPROJ_COLS], 1);
        assert_eq!(mismatches(&z1, &z[r * ng::Z_COLS..(r + 1) * ng::Z_COLS], false), 0, "z row {r} changes bits in a 1-row call");
        assert_eq!(mismatches(&x1, &xbc[r * ng::XBC_COLS..(r + 1) * ng::XBC_COLS], false), 0);
        assert_eq!(mismatches(&d1, &dt[r * ng::DT_COLS..(r + 1) * ng::DT_COLS], false), 0);
    }
    eprintln!(
        "split_inproj: 64 rows bitwise equal to the model (z on the bf16 grid, xbc and dt exact), rows 1, 7 and 64 identical"
    );
}

// ---------------------------------------------------------------------------------------------------------------------------------
// widens

/// An fp16 bit pattern as the f32 bits it widens to (exact); `None` for a NaN, whose payload is not compared.
fn f16_to_f32_model(h: u16) -> Option<f32> {
    let sign = if h >> 15 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    match exp {
        0 => Some(sign * man * 2f32.powi(-24)), // zero and subnormals: man * 2^-24, exact
        31 if h & 0x3ff == 0 => Some(sign * f32::INFINITY),
        31 => None,
        _ => Some(sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15)),
    }
}

#[test]
fn widens_are_exact_for_every_bit_pattern_and_for_every_tail_length() {
    gpu!();
    let st = Stream::new().unwrap();
    // all 65536 patterns, then a ragged length so the scalar tail and the vector body are both exercised
    for &n in &[65536usize, 1, 7, 8, 9, 15, 4097, 100_003] {
        let src: Vec<u16> = (0..n).map(|i| (i as u32).wrapping_mul(2654435761).wrapping_shr(7) as u16 ^ (i as u16)).collect();
        let src = if n == 65536 { (0..=65535u32).map(|v| v as u16).collect::<Vec<u16>>() } else { src };
        let ds = DevBuf::from_slice(&src).unwrap();
        let mut dbf = DevBuf::zeroed(n * 4).unwrap();
        let mut dhf = DevBuf::zeroed(n * 4).unwrap();
        ng::widen_bf16_f32(&ds, n, &mut dbf, &st).unwrap();
        ng::widen_f16_f32(&ds, n, &mut dhf, &st).unwrap();
        st.sync().unwrap();
        let gb = download_f32(&dbf, n);
        let gh = download_f32(&dhf, n);
        for i in 0..n {
            assert_eq!(gb[i].to_bits(), (src[i] as u32) << 16, "bf16 widen of {:#06x} at {i} (n={n})", src[i]);
            match f16_to_f32_model(src[i]) {
                Some(v) => assert_eq!(gh[i].to_bits(), v.to_bits(), "fp16 widen of {:#06x} at {i} (n={n})", src[i]),
                None => assert!(gh[i].is_nan(), "fp16 NaN {:#06x} at {i} (n={n}) did not widen to a NaN", src[i]),
            }
        }
    }
    eprintln!("widen_bf16_f32 and widen_f16_f32: all 65536 patterns and tails of 1, 7, 8, 9, 15, 4097, 100003 exact");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// topk_f32

#[derive(Clone, Copy, Debug)]
enum Kind {
    Random,
    Ties,
    Nans,
    AllNan,
    InfMix,
    RampUp,
    RampDown,
    Constant,
    ZeroSigns,
    Spikes,
}

const KINDS: [Kind; 10] = [
    Kind::Random,
    Kind::Ties,
    Kind::Nans,
    Kind::AllNan,
    Kind::InfMix,
    Kind::RampUp,
    Kind::RampDown,
    Kind::Constant,
    Kind::ZeroSigns,
    Kind::Spikes,
];

fn nan_with_payload(rng: &mut Rng) -> f32 {
    let r = rng.next_u64() as u32;
    f32::from_bits(((r & 1) << 31) | 0x7f80_0000 | ((r >> 1) & 0x007f_ffff) | 1)
}

fn topk_row(kind: Kind, vocab: usize, rng: &mut Rng) -> Vec<f32> {
    match kind {
        Kind::Random => rng.vec(vocab, -8.0, 8.0),
        // a handful of distinct values, both zeros among them: almost everything ties
        Kind::Ties => {
            let set = [-1.0f32, -0.0, 0.0, 0.5, 2.0, 2.0, f32::from_bits(1)];
            (0..vocab).map(|_| set[(rng.next_u64() % set.len() as u64) as usize]).collect()
        }
        Kind::Nans => {
            (0..vocab).map(|_| if rng.next_u64() % 10 == 0 { nan_with_payload(rng) } else { rng.uniform(-5.0, 5.0) }).collect()
        }
        Kind::AllNan => (0..vocab).map(|_| nan_with_payload(rng)).collect(),
        Kind::InfMix => {
            let set = [f32::NEG_INFINITY, f32::INFINITY, f32::NAN, 1.0, -1.0, 0.0, f32::NEG_INFINITY];
            (0..vocab).map(|_| set[(rng.next_u64() % set.len() as u64) as usize]).collect()
        }
        Kind::RampUp => (0..vocab).map(|i| i as f32).collect(),
        Kind::RampDown => (0..vocab).map(|i| -(i as f32)).collect(),
        Kind::Constant => vec![3.5; vocab],
        Kind::ZeroSigns => (0..vocab)
            .map(|_| match rng.next_u64() % 4 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::from_bits(1 + (rng.next_u64() as u32 % 5)),
                _ => -f32::from_bits(1 + (rng.next_u64() as u32 % 5)),
            })
            .collect(),
        Kind::Spikes => {
            let mut v: Vec<f32> = (0..vocab).map(|_| rng.uniform(-1.0, 1.0) * 1e-3).collect();
            for _ in 0..(vocab / 400).max(3) {
                let at = (rng.next_u64() % vocab as u64) as usize;
                v[at] = rng.uniform(10.0, 1.0e30);
            }
            v
        }
    }
}

/// The stable sort the spec names: value descending (NaN below every number, -0.0 equal to +0.0), ties keep id order.
fn topk_model(row: &[f32], count: usize) -> (Vec<i32>, Vec<f32>) {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (row[a], row[b]);
        match (x.is_nan(), y.is_nan()) {
            (true, true) => core::cmp::Ordering::Equal,
            (true, false) => core::cmp::Ordering::Greater,
            (false, true) => core::cmp::Ordering::Less,
            (false, false) => y.partial_cmp(&x).unwrap(),
        }
    });
    idx.truncate(count);
    (idx.iter().map(|&i| i as i32).collect(), idx.iter().map(|&i| row[i]).collect())
}

/// Runs the kernel with a workspace of exactly `topk_workspace_bytes` and a poisoned guard behind it, which must survive.
fn run_topk(st: &Stream, logits: &DevBuf, rows: usize, vocab: usize, count: usize) -> (Vec<i32>, Vec<f32>) {
    let wsb = ng::topk_workspace_bytes(rows, vocab, count);
    let guard = 4096usize;
    let mut block = DevBuf::zeroed_view(wsb + guard).unwrap();
    block.upload(&vec![0xABu8; wsb + guard]).unwrap();
    let mut ws = block.subview(0, wsb);
    let mut ids = DevBuf::zeroed(rows * count * 4).unwrap();
    let mut vals = DevBuf::zeroed(rows * count * 4).unwrap();
    ng::topk_f32(logits, rows, vocab, count, &mut ws, &mut ids, &mut vals, st).unwrap();
    st.sync().unwrap();
    let mut tail = vec![0u8; guard];
    block.download_range(wsb, &mut tail).unwrap();
    assert!(tail.iter().all(|&b| b == 0xAB), "topk wrote past its exact workspace (rows={rows} vocab={vocab} count={count})");
    (ids.to_vec::<i32>().unwrap(), vals.to_vec::<f32>().unwrap())
}

fn check_topk(
    st: &Stream,
    vocab: usize,
    rows: usize,
    count: usize,
    kinds: &[Kind],
    rng: &mut Rng,
) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
    let mut data = Vec::with_capacity(rows * vocab);
    let mut row_kinds = Vec::new();
    for r in 0..rows {
        let kind = kinds[r % kinds.len()];
        row_kinds.push(kind);
        data.extend(topk_row(kind, vocab, rng));
    }
    let dl = upload_f32(&data);
    let (ids, vals) = run_topk(st, &dl, rows, vocab, count);
    for r in 0..rows {
        let (wi, wv) = topk_model(&data[r * vocab..(r + 1) * vocab], count);
        assert_eq!(
            &ids[r * count..(r + 1) * count],
            &wi[..],
            "vocab={vocab} rows={rows} count={count} row {r} ({:?}): ids differ",
            row_kinds[r]
        );
        let gv: Vec<u32> = vals[r * count..(r + 1) * count].iter().map(|v| v.to_bits()).collect();
        let wv: Vec<u32> = wv.iter().map(|v| v.to_bits()).collect();
        assert_eq!(gv, wv, "vocab={vocab} rows={rows} count={count} row {r} ({:?}): values differ in bits", row_kinds[r]);
    }
    (ids, vals, data)
}

#[test]
fn topk_equals_the_stable_sort_bit_for_bit() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(404);
    let mut n = 0usize;
    // the lm_head shape, every data kind at once, several counts
    for &(rows, count) in &[(64usize, 40usize), (64, 64), (7, 1), (1, 64), (1, 1), (3, 17)] {
        check_topk(&st, 131072, rows, count, &KINDS, &mut rng);
        n += 1;
    }
    // vocabularies that are not a multiple of 4 or of anything, thin rows, and count = vocab
    for &vocab in &[1003usize, 5000, 4099, 2049, 70, 64, 33, 13] {
        for &count in &[1usize, 2, 40, 63, 64] {
            if count > vocab {
                continue;
            }
            for &rows in &[1usize, 7, 64] {
                check_topk(&st, vocab, rows, count, &KINDS, &mut rng);
                n += 1;
            }
        }
    }
    eprintln!("topk_f32: {n} (vocab, rows, count) cases over 10 data kinds equal the stable sort in ids and in value bits");
}

#[test]
fn topk_rows_are_bitwise_the_same_in_calls_of_1_7_and_64_rows() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(505);
    for &(vocab, count) in &[(131072usize, 40usize), (1003, 64), (5000, 1)] {
        let (ids, vals, data) = check_topk(&st, vocab, 64, count, &KINDS, &mut rng);
        let d7 = upload_f32(&data[..7 * vocab]);
        let (i7, v7) = run_topk(&st, &d7, 7, vocab, count);
        assert_eq!(i7, ids[..7 * count], "vocab={vocab}: ids of rows 0..7 change between the 7- and 64-row calls");
        assert_eq!(bits(&v7), bits(&vals[..7 * count]));
        for r in 0..64 {
            let d1 = upload_f32(&data[r * vocab..(r + 1) * vocab]);
            let (i1, v1) = run_topk(&st, &d1, 1, vocab, count);
            assert_eq!(&i1[..], &ids[r * count..(r + 1) * count], "vocab={vocab}: row {r} ids change in a 1-row call");
            assert_eq!(
                bits(&v1),
                bits(&vals[r * count..(r + 1) * count]),
                "vocab={vocab}: row {r} values change in a 1-row call"
            );
        }
    }
    eprintln!(
        "topk_f32: every row identical in calls of 1, 7 and 64 rows (the plan uses 16, 6 and 1 blocks per row at vocab 131072)"
    );
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn topk_refuses_what_is_outside_its_contract_and_panics_on_a_short_workspace() {
    gpu!();
    let st = Stream::new().unwrap();
    let logits = upload_f32(&vec![0.0; 100]);
    let mut ws = DevBuf::zeroed(1 << 20).unwrap();
    let mut ids = DevBuf::zeroed(65 * 4).unwrap();
    let mut vals = DevBuf::zeroed(65 * 4).unwrap();
    for (vocab, count) in [(100usize, 0usize), (100, 65), (10, 11)] {
        let r = ng::topk_f32(&logits, 1, vocab, count, &mut ws, &mut ids, &mut vals, &st);
        assert_eq!(r.unwrap_err().0, 1, "vocab={vocab} count={count} must be refused with cudaErrorInvalidValue");
    }
    // rows = 0 is a no-op
    ng::topk_f32(&logits, 0, 100, 5, &mut ws, &mut ids, &mut vals, &st).unwrap();
    // one byte short of the exact workspace panics before launching
    let need = ng::topk_workspace_bytes(1, 100, 5);
    let mut short = DevBuf::zeroed(need - 1).unwrap();
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = ng::topk_f32(&logits, 1, 100, 5, &mut short, &mut ids, &mut vals, &st);
    }));
    assert!(res.is_err(), "a workspace one byte short must panic");
    // and the exact size works
    let mut exact = DevBuf::zeroed(need).unwrap();
    ng::topk_f32(&logits, 1, 100, 5, &mut exact, &mut ids, &mut vals, &st).unwrap();
    st.sync().unwrap();
    let got = ids.to_vec::<i32>().unwrap();
    assert_eq!(&got[..5], &[0, 1, 2, 3, 4], "a constant row ties to the lowest ids");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// timings

#[test]
fn microseconds_per_kernel() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(606);
    let line = |name: &str, rows: usize, us: f64| eprintln!("glue timing: {name:<34} rows {rows:2}: {us:8.2} us/call");
    let d = ng::HIDDEN;

    // D2D copy bandwidth, for the top-k floor
    let big = 32 << 20;
    let src = DevBuf::zeroed(big).unwrap();
    let mut dst = DevBuf::zeroed(big).unwrap();
    let copy_us = batch_us(&st, 3, 20, |_| dst.copy_from(&src, big, &st).unwrap());
    eprintln!(
        "glue timing: D2D copy of 32 MiB: {copy_us:.1} us = {:.1} GB/s of read + write traffic",
        2.0 * big as f64 / (copy_us * 1e-6) / 1e9
    );

    for &rows in &[1usize, 64] {
        // embed
        let table = DevBuf::from_slice(&random_u16(&mut rng, 2048 * d)).unwrap();
        let ids: Vec<i32> = (0..rows).map(|_| (rng.next_u64() % 2048) as i32).collect();
        let dids = DevBuf::from_slice(&ids).unwrap();
        let mut out = DevBuf::zeroed(rows * d * 4).unwrap();
        let us = batch_us(&st, 5, 200, |_| ng::embed_gather_f32(&table, &dids, rows, &mut out, &st).unwrap());
        line("embed_gather_f32", rows, us);

        // norm
        let x = upload_f32(&rng.vec(rows * d, -1.0, 1.0));
        let w = DevBuf::from_slice(&random_weights(&mut rng, d)).unwrap();
        let mut o = DevBuf::zeroed(rows * d * 4).unwrap();
        let us = batch_us(&st, 5, 200, |_| ng::rmsnorm_bf16w_f32(&x, &w, rows, d, d, 1e-5, &mut o, d, &st).unwrap());
        line("rmsnorm_bf16w_f32 (d = 4096)", rows, us);

        // split
        let y = upload_f32(&rng.vec(rows * ng::INPROJ_COLS, -2.0, 2.0));
        let mut z = DevBuf::zeroed(rows * ng::Z_COLS * 4).unwrap();
        let mut xbc = DevBuf::zeroed(rows * ng::XBC_COLS * 4).unwrap();
        let mut dt = DevBuf::zeroed(rows * ng::DT_COLS * 4).unwrap();
        let us = batch_us(&st, 5, 200, |_| ng::split_inproj(&y, rows, &mut z, &mut xbc, &mut dt, &st).unwrap());
        line("split_inproj", rows, us);

        // splice
        let feats = DevBuf::from_slice(&random_u16(&mut rng, rows * d)).unwrap();
        let mut sd = DevBuf::zeroed(rows * d * 4).unwrap();
        let us = batch_us(&st, 5, 200, |_| ng::splice_rows(&feats, 0, &mut sd, 0, rows, &st).unwrap());
        line("splice_rows", rows, us);

        // top-k: warm (the same logits every call) and cold (four buffers that together exceed the L2)
        let (vocab, count) = (131072usize, 40usize);
        let bufs: Vec<DevBuf> = (0..4).map(|_| upload_f32(&rng.vec(rows * vocab, -8.0, 8.0))).collect();
        let mut ws = DevBuf::zeroed(ng::topk_workspace_bytes(rows, vocab, count)).unwrap();
        let mut ids = DevBuf::zeroed(rows * count * 4).unwrap();
        let mut vals = DevBuf::zeroed(rows * count * 4).unwrap();
        let warm =
            batch_us(&st, 5, 100, |_| ng::topk_f32(&bufs[0], rows, vocab, count, &mut ws, &mut ids, &mut vals, &st).unwrap());
        let cold =
            batch_us(&st, 8, 100, |i| ng::topk_f32(&bufs[i % 4], rows, vocab, count, &mut ws, &mut ids, &mut vals, &st).unwrap());
        let bytes = (rows * vocab * 4) as f64;
        line("topk_f32 (131072, count 40), same buf", rows, warm);
        line("topk_f32 (131072, count 40), 4 bufs", rows, cold);
        // a copy moves its bytes twice (read and write): that traffic rate is what one read pass of the logits can get from DRAM
        let traffic = 2.0 * big as f64 / (copy_us * 1e-6);
        eprintln!(
            "glue timing: topk reads {:.1} MB; at the copy's {:.0} GB/s of DRAM traffic one pass takes {:.1} us",
            bytes / 1e6,
            traffic / 1e9,
            bytes / traffic * 1e6
        );
    }
    // load-time widens, 16 M elements
    let n = 16 << 20;
    let src = DevBuf::zeroed(n * 2).unwrap();
    let mut dst = DevBuf::zeroed(n * 4).unwrap();
    let us = batch_us(&st, 3, 20, |_| ng::widen_bf16_f32(&src, n, &mut dst, &st).unwrap());
    eprintln!("glue timing: widen_bf16_f32 of 16M elements: {us:.1} us = {:.1} GB/s moved", (n * 6) as f64 / (us * 1e-6) / 1e9);
    let us = batch_us(&st, 3, 20, |_| ng::widen_f16_f32(&src, n, &mut dst, &st).unwrap());
    eprintln!("glue timing: widen_f16_f32 of 16M elements: {us:.1} us = {:.1} GB/s moved", (n * 6) as f64 / (us * 1e-6) / 1e9);
}
