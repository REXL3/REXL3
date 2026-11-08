//! Bit-exactness of the decode fast hyper-connection kernels (hcx k1b/k2c, the regridded hc_post) against the
//! reference launches, at the family's real shapes and the decode window's 1..8 rows.
#![allow(clippy::needless_range_loop)]
mod common;
use common::Rng;
use gb_cuda::{fn_mhc, hcx, DevBuf, Stream};
use gb_exl3::f16::f64_to_f16;

const H: usize = 2560; // hidden
const HC: usize = 4; // streams
const R: usize = 320; // low rank
const EPS: f32 = 1e-6;

fn gpu() -> Option<Stream> {
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    gb_cuda::rt::init_context().ok()?;
    Stream::new().ok()
}

fn hf(x: f32) -> u16 {
    f64_to_f16(x as f64)
}

fn dl<T: Copy + Default>(b: &DevBuf, s: &Stream) -> Vec<T> {
    s.sync().unwrap();
    b.to_vec().unwrap()
}

fn fill_half(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n).map(|_| hf(rng.uniform(-1.5, 1.5))).collect()
}

fn mixer<'a>(norm: &'a DevBuf, down: &'a DevBuf, up: &'a DevBuf, inj: &'a DevBuf) -> gb_cuda::fn_mhc::Mixer<'a> {
    gb_cuda::fn_mhc::Mixer { norm, down, up, inject: Some(inj) }
}

#[test]
fn hcx_decode_fast_bit_exact_vs_ref_rows_1to8() {
    let Some(st) = gpu() else { return };
    for t in 1..=8usize {
        for seed in 0..2u64 {
            let mut rng = Rng(seed * 131 + t as u64);
            let x: Vec<f32> = rng.vec(t * HC * H, -2.0, 2.0);
            let nw = fill_half(&mut rng, HC * H);
            let down = fill_half(&mut rng, R * HC * H);
            let up = fill_half(&mut rng, HC * H * R);
            let iw = fill_half(&mut rng, HC * HC * H);

            let mut dx = DevBuf::zeroed(t * HC * H * 4).unwrap();
            dx.upload(&x).unwrap();
            let mut dnw = DevBuf::zeroed(HC * H * 2).unwrap();
            dnw.upload(&nw).unwrap();
            let mut ddown = DevBuf::zeroed(R * HC * H * 2).unwrap();
            ddown.upload(&down).unwrap();
            let mut dup = DevBuf::zeroed(HC * H * R * 2).unwrap();
            dup.upload(&up).unwrap();
            let mut diw = DevBuf::zeroed(HC * HC * H * 2).unwrap();
            diw.upload(&iw).unwrap();

            let scr = hcx::scratch_bytes(HC, R);
            let mut y_new = DevBuf::zeroed(t * H * 2).unwrap();
            let mut y_ref = DevBuf::zeroed(t * H * 2).unwrap();
            let mut ij_new = DevBuf::zeroed(t * HC * 4).unwrap();
            let mut ij_ref = DevBuf::zeroed(t * HC * 4).unwrap();
            let mut s_new = DevBuf::zeroed(scr).unwrap();
            let mut s_ref = DevBuf::zeroed(scr).unwrap();
            let m = mixer(&dnw, &ddown, &dup, &diw);
            hcx::pre(&dx, &m, &mut y_new, Some(&mut ij_new), &mut s_new, H, HC, R, EPS, t, &st).unwrap();
            hcx::pre_ref(&dx, &m, &mut y_ref, Some(&mut ij_ref), &mut s_ref, H, HC, R, EPS, t, &st).unwrap();

            let yn: Vec<u16> = dl(&y_new, &st);
            let yr: Vec<u16> = dl(&y_ref, &st);
            let inj_new: Vec<f32> = dl(&ij_new, &st);
            let inj_ref: Vec<f32> = dl(&ij_ref, &st);
            let yd = yn.iter().zip(&yr).filter(|(a, b)| a != b).count();
            let id = inj_new.iter().zip(&inj_ref).filter(|(a, b)| a != b).count();
            if yd > 0 {
                let mut first: Vec<String> = Vec::new();
                for (i, (a, b)) in yn.iter().zip(&yr).enumerate() {
                    if a != b && first.len() < 12 {
                        first.push(format!("d={i} new={a} ref={b}"));
                    }
                }
                eprintln!("DEBUG y diffs at t={t} seed={seed}: {}", first.join(", "));
            }
            assert_eq!(
                (yd, id),
                (0, 0),
                "hcx bits differ at t={t} seed={seed}: y diff {yd}/{}, inj diff {id}/{}",
                yn.len(),
                inj_new.len()
            );
        }
    }
}

#[test]
fn hcx_decode_fast_bit_exact_vs_ref_single_c_iter_rank64() {
    // rank=64: every lane runs exactly one c-iteration, so the double-buffer prefetch path is never taken.
    // If this passes while rank=320 fails, the prefetch/swap is implicated; if it fails too, the core math is.
    let Some(st) = gpu() else { return };
    for t in 1..=2usize {
        for seed in 0..2u64 {
            let mut rng = Rng(seed * 977 + t as u64);
            let x: Vec<f32> = rng.vec(t * HC * H, -2.0, 2.0);
            let nw = fill_half(&mut rng, HC * H);
            let down = fill_half(&mut rng, 64 * HC * H);
            let up = fill_half(&mut rng, HC * H * 64);
            let iw = fill_half(&mut rng, HC * HC * H);
            let mut dx = DevBuf::zeroed(t * HC * H * 4).unwrap();
            dx.upload(&x).unwrap();
            let mut dnw = DevBuf::zeroed(HC * H * 2).unwrap();
            dnw.upload(&nw).unwrap();
            let mut ddown = DevBuf::zeroed(64 * HC * H * 2).unwrap();
            ddown.upload(&down).unwrap();
            let mut dup = DevBuf::zeroed(HC * H * 64 * 2).unwrap();
            dup.upload(&up).unwrap();
            let mut diw = DevBuf::zeroed(HC * HC * H * 2).unwrap();
            diw.upload(&iw).unwrap();
            let scr = hcx::scratch_bytes(HC, 64);
            let mut y_new = DevBuf::zeroed(t * H * 2).unwrap();
            let mut y_ref = DevBuf::zeroed(t * H * 2).unwrap();
            let mut ij_new = DevBuf::zeroed(t * HC * 4).unwrap();
            let mut ij_ref = DevBuf::zeroed(t * HC * 4).unwrap();
            let mut s_new = DevBuf::zeroed(scr).unwrap();
            let mut s_ref = DevBuf::zeroed(scr).unwrap();
            let m = mixer(&dnw, &ddown, &dup, &diw);
            hcx::pre(&dx, &m, &mut y_new, Some(&mut ij_new), &mut s_new, H, HC, 64, EPS, t, &st).unwrap();
            hcx::pre_ref(&dx, &m, &mut y_ref, Some(&mut ij_ref), &mut s_ref, H, HC, 64, EPS, t, &st).unwrap();
            let yn: Vec<u16> = dl(&y_new, &st);
            let yr: Vec<u16> = dl(&y_ref, &st);
            let yd = yn.iter().zip(&yr).filter(|(a, b)| a != b).count();
            assert_eq!(yd, 0, "rank64 hcx y bits differ at t={t} seed={seed}: {yd}/{}", yn.len());
        }
    }
}

#[test]
fn hcx_k2_z_new_vs_ref() {
    // Development-only: compare the pre-sigmoid dot-product sums (z) of the new and reference k2 kernels.
    let Some(st) = gpu() else { return };
    for t in 1..=4usize {
        for seed in 0..2u64 {
            let mut rng = Rng(seed * 331 + t as u64);
            let x: Vec<f32> = rng.vec(t * HC * H, -2.0, 2.0);
            let nw = fill_half(&mut rng, HC * H);
            let down = fill_half(&mut rng, R * HC * H);
            let up = fill_half(&mut rng, HC * H * R);
            let iw = fill_half(&mut rng, HC * HC * H);
            let mut dx = DevBuf::zeroed(t * HC * H * 4).unwrap();
            dx.upload(&x).unwrap();
            let mut dnw = DevBuf::zeroed(HC * H * 2).unwrap();
            dnw.upload(&nw).unwrap();
            let mut ddown = DevBuf::zeroed(R * HC * H * 2).unwrap();
            ddown.upload(&down).unwrap();
            let mut dup = DevBuf::zeroed(HC * H * R * 2).unwrap();
            dup.upload(&up).unwrap();
            let mut diw = DevBuf::zeroed(HC * HC * H * 2).unwrap();
            diw.upload(&iw).unwrap();
            let scr = hcx::scratch_bytes(HC, R);
            let mut y_new = DevBuf::zeroed(t * H * 2).unwrap();
            let mut y_ref = DevBuf::zeroed(t * H * 2).unwrap();
            let mut ij_new = DevBuf::zeroed(t * HC * 4).unwrap();
            let mut ij_ref = DevBuf::zeroed(t * HC * 4).unwrap();
            let mut s_new = DevBuf::zeroed(scr).unwrap();
            let mut s_ref = DevBuf::zeroed(scr).unwrap();
            let mut z_new = DevBuf::zeroed(HC * t * H * 4).unwrap();
            let mut z_ref = DevBuf::zeroed(HC * t * H * 4).unwrap();
            let m = mixer(&dnw, &ddown, &dup, &diw);
            mock_pre_z(&st, &dx, &m, &mut y_new, &mut ij_new, &mut s_new, &mut z_new, t);
            mock_pre_z_ref(&st, &dx, &m, &mut y_ref, &mut ij_ref, &mut s_ref, &mut z_ref, t);
            let zn: Vec<f32> = dl(&z_new, &st);
            let zr: Vec<f32> = dl(&z_ref, &st);
            let zd = zn.iter().zip(&zr).filter(|(a, b)| a != b).count();
            let yn: Vec<u16> = dl(&y_new, &st);
            let yr: Vec<u16> = dl(&y_ref, &st);
            let yd = yn.iter().zip(&yr).filter(|(a, b)| a != b).count();
            if zd > 0 && t == 1 && seed == 0 {
                let mut first: Vec<String> = Vec::new();
                for (i, (a, b)) in zn.iter().zip(&zr).enumerate() {
                    if a != b && first.len() < 16 {
                        let s = i / H;
                        let d = i % H;
                        first.push(format!("s={s} d={d} new={a:e} ref={b:e}"));
                    }
                }
                eprintln!("DEBUG z diffs: {}", first.join(", "));
            }
            assert_eq!((zd, yd), (0, 0), "k2 z or y differ at t={t} seed={seed}: z {zd}/{} y {yd}/{}", zn.len(), yn.len());
        }
    }
}

fn mock_pre_z(
    st: &Stream,
    streams: &DevBuf,
    mix: &gb_cuda::fn_mhc::Mixer<'_>,
    y_out: &mut DevBuf,
    inj_out: &mut DevBuf,
    scratch: &mut DevBuf,
    z_out: &mut DevBuf,
    t: usize,
) {
    let rc = unsafe {
        gb_cuda::hcx::hcx_ffi_z(
            streams.ptr(),
            mix.norm.ptr(),
            mix.down.ptr(),
            mix.up.ptr(),
            mix.inject.unwrap().ptr(),
            y_out.ptr(),
            inj_out.ptr() as *mut f32,
            scratch.ptr(),
            z_out.ptr() as *mut f32,
            H as u32,
            HC as u32,
            R as u32,
            EPS,
            t as u32,
            st.raw(),
        )
    };
    assert_eq!(rc, 0);
}

fn mock_pre_z_ref(
    st: &Stream,
    streams: &DevBuf,
    mix: &gb_cuda::fn_mhc::Mixer<'_>,
    y_out: &mut DevBuf,
    inj_out: &mut DevBuf,
    scratch: &mut DevBuf,
    z_out: &mut DevBuf,
    t: usize,
) {
    let rc = unsafe {
        gb_cuda::hcx::hcx_ffi_z_ref(
            streams.ptr(),
            mix.norm.ptr(),
            mix.down.ptr(),
            mix.up.ptr(),
            mix.inject.unwrap().ptr(),
            y_out.ptr(),
            inj_out.ptr() as *mut f32,
            scratch.ptr(),
            z_out.ptr() as *mut f32,
            H as u32,
            HC as u32,
            R as u32,
            EPS,
            t as u32,
            st.raw(),
        )
    };
    assert_eq!(rc, 0);
}

#[test]
fn hc_post_regrid_bit_exact_vs_host() {
    let Some(st) = gpu() else { return };
    for t in 1..=8usize {
        let mut rng = Rng(7 * t as u64);
        let bo: Vec<u16> = (0..t * H).map(|_| hf(rng.uniform(-2.0, 2.0))).collect();
        let res: Vec<f32> = rng.vec(t * HC * H, -2.0, 2.0);
        let inj: Vec<f32> = rng.vec(t * HC, 0.0, 2.0);
        let mut dbo = DevBuf::zeroed(t * H * 2).unwrap();
        dbo.upload(&bo).unwrap();
        let mut dres = DevBuf::zeroed(t * HC * H * 4).unwrap();
        dres.upload(&res).unwrap();
        let mut dinj = DevBuf::zeroed(t * HC * 4).unwrap();
        dinj.upload(&inj).unwrap();
        let mut dout = DevBuf::zeroed(t * HC * H * 4).unwrap();
        fn_mhc::hc_post(&dbo, Some(&dres), &dinj, &mut dout, H, HC, t, &st).unwrap();
        let got: Vec<f32> = dl(&dout, &st);
        let mut expect = vec![0f32; t * HC * H];
        for e in 0..t * HC * H {
            let tt = e / (HC * H);
            let rem = e % (HC * H);
            let s = rem / H;
            let d = rem % H;
            // block_out is bf16 on the device (the model's hidden state); decode the stored u16 the same way.
            // The device compiles `rv + vx * inj` to a fused multiply-add (FMAD on by default) — replicate the single
            // rounding with f32::mul_add so the comparison is bit-exact.
            let vx = f32::from_bits((bo[tt * H + d] as u32) << 16);
            expect[e] = f32::mul_add(vx, inj[tt * HC + s], res[e]);
        }
        let diffs = got
            .iter()
            .zip(&expect)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .take(8)
            .map(|(i, (a, b))| format!("e={i} got={a:e} exp={b:e}"))
            .collect::<Vec<_>>()
            .join(", ");
        assert_eq!(got, expect, "hc_post bits differ at t={t}: {diffs}");
    }
}

// ── Perf probe (print-only; no pass/fail on speed) ──
// Times the decode hyper-connection kernels at the family's real shapes so a kernel/grid change can be judged on one
// box under identical conditions: hc_post (post-injection) and the full hcx pre stage via the fast k1b/k2c path vs
// the reference k1/k2 path. Every cell is warmed, then N runs are timed end-to-end (launch + stream sync) and the
// median is reported. Run with `--nocapture` to see the numbers.
#[test]
fn hcx_perf_bench() {
    let Some(st) = gpu() else { return };
    const ITERS: usize = 400;
    const WARM: u32 = 60;

    fn median_us<F: FnMut()>(warm: u32, n: usize, mut f: F) -> f64 {
        for _ in 0..warm {
            f();
        }
        let mut us = Vec::with_capacity(n);
        for _ in 0..n {
            let t0 = std::time::Instant::now();
            f();
            us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        us[n / 2]
    }

    // ── hc_post ──
    let mut rng = Rng(0x5EED_131u64);
    let bo: Vec<u16> = (0..8 * H).map(|_| hf(rng.uniform(-2.0, 2.0))).collect();
    let res: Vec<f32> = rng.vec(8 * HC * H, -2.0, 2.0);
    let inj: Vec<f32> = rng.vec(8 * HC, 0.0, 2.0);
    let mut dbo = DevBuf::zeroed(8 * H * 2).unwrap();
    dbo.upload(&bo).unwrap();
    let mut dres = DevBuf::zeroed(8 * HC * H * 4).unwrap();
    dres.upload(&res).unwrap();
    let mut dinj = DevBuf::zeroed(8 * HC * 4).unwrap();
    dinj.upload(&inj).unwrap();
    let mut dout = DevBuf::zeroed(8 * HC * H * 4).unwrap();
    for t in [1usize, 4, 8] {
        let us = median_us(WARM, ITERS, || {
            fn_mhc::hc_post(&dbo, Some(&dres), &dinj, &mut dout, H, HC, t, &st).unwrap();
            st.sync().unwrap();
        });
        println!("[hcx_bench] hc_post t={t}: {us:.1} us (median of {ITERS})");
    }

    // ── hcx pre: fast k1b/k2c vs reference k1/k2 ──
    let x: Vec<f32> = rng.vec(8 * HC * H, -2.0, 2.0);
    let nw = fill_half(&mut rng, HC * H);
    let down = fill_half(&mut rng, R * HC * H);
    let up = fill_half(&mut rng, HC * H * R);
    let iw = fill_half(&mut rng, HC * HC * H);
    let mut dx = DevBuf::zeroed(8 * HC * H * 4).unwrap();
    dx.upload(&x).unwrap();
    let mut dnw = DevBuf::zeroed(HC * H * 2).unwrap();
    dnw.upload(&nw).unwrap();
    let mut ddown = DevBuf::zeroed(R * HC * H * 2).unwrap();
    ddown.upload(&down).unwrap();
    let mut dup = DevBuf::zeroed(HC * H * R * 2).unwrap();
    dup.upload(&up).unwrap();
    let mut diw = DevBuf::zeroed(HC * HC * H * 2).unwrap();
    diw.upload(&iw).unwrap();
    let scr = hcx::scratch_bytes(HC, R);
    let mut y = DevBuf::zeroed(8 * H * 2).unwrap();
    let mut ij = DevBuf::zeroed(8 * HC * 4).unwrap();
    let mut s_new = DevBuf::zeroed(scr).unwrap();
    let mut s_ref = DevBuf::zeroed(scr).unwrap();
    let m = mixer(&dnw, &ddown, &dup, &diw);
    for t in [1usize, 4, 8] {
        let us_fast = median_us(WARM, ITERS, || {
            hcx::pre(&dx, &m, &mut y, Some(&mut ij), &mut s_new, H, HC, R, EPS, t, &st).unwrap();
            st.sync().unwrap();
        });
        println!("[hcx_bench] pre k1b/k2c t={t}: {us_fast:.1} us (median of {ITERS})");
        let us_ref = median_us(WARM, ITERS, || {
            hcx::pre_ref(&dx, &m, &mut y, Some(&mut ij), &mut s_ref, H, HC, R, EPS, t, &st).unwrap();
            st.sync().unwrap();
        });
        println!("[hcx_bench] pre k1/k2 ref t={t}: {us_ref:.1} us (median of {ITERS})");
    }
}
