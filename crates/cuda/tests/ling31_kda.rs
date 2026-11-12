//! The full-rank KDA gate of Ling-3.1-flash-1019 (`no_kda_lora = true`): `gb_kda_gate_from_full_fg` fills beta, the
//! per-channel decay g and the output gate z from the full-rank fp16 `f_proj` / `g_proj` outputs and the plain bf16
//! `b_proj` weight, instead of the low-rank `proj_in` + `proj_out` pair. The kernel is row-invariant (a row's values
//! depend only on its own x / f / g) and matches the CPU reference.
mod common;

const HEADS: usize = 64;
const HIDDEN: usize = 4096;
const C: usize = HEADS * 128;

use common::Rng;
use gb_cuda::rt::{DevBuf, Stream};

fn gpu() -> Option<Stream> {
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    gb_cuda::rt::init_context().ok()?;
    Stream::new().ok()
}
fn bf(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn bff(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}
fn dl<T: Copy + Default>(v: &DevBuf, st: &Stream) -> Vec<T> {
    st.sync().unwrap();
    v.to_vec().unwrap()
}
fn f16(v: f64) -> u16 {
    gb_exl3::f16::f64_to_f16(v)
}

#[test]
#[ignore]
fn full_rank_gate_matches_cpu_and_is_row_invariant() {
    let Some(st) = gpu() else { return };
    let mut r = Rng(0x7a11);
    let rows = 32usize;
    // x bf16 [rows, hidden]; b_weight bf16 [heads, hidden]; a_log f32 [heads] (log of uniform(1, 16));
    // dt_bias f32 [C]; f_out / g_out fp16 [rows, C]
    let x: Vec<u16> = (0..rows * HIDDEN).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
    let b_weight: Vec<u16> = (0..HEADS * HIDDEN).map(|_| bf(r.uniform(-0.05, 0.05))).collect();
    let a_log: Vec<f32> = (0..HEADS).map(|_| (1.0 + 15.0 * r.uniform(0.0, 1.0)).ln()).collect();
    let dt_bias: Vec<f32> = (0..C).map(|_| r.uniform(-1.0, 1.0)).collect();
    let f_out: Vec<u16> = (0..rows * C).map(|_| f16(r.uniform(-1.0, 1.0) as f64)).collect();
    let g_out: Vec<u16> = (0..rows * C).map(|_| f16(r.uniform(-1.0, 1.0) as f64)).collect();
    let lower = -5.0f32;

    let x_d = DevBuf::from_slice(&x).unwrap();
    let b_d = DevBuf::from_slice(&b_weight).unwrap();
    let a_d = DevBuf::from_slice(&a_log).unwrap();
    let dt_d = DevBuf::from_slice(&dt_bias).unwrap();
    let f_d = DevBuf::from_slice(&f_out).unwrap();
    let g_d = DevBuf::from_slice(&g_out).unwrap();
    let mut g_decay = DevBuf::zeroed(rows * C * 4).unwrap();
    let mut z_gate = DevBuf::zeroed(rows * C * 4).unwrap();
    let mut beta = DevBuf::zeroed(rows * HEADS * 4).unwrap();

    gb_cuda::kda::gate_from_full_fg(
        &st,
        &f_d,
        &g_d,
        &x_d,
        &b_d,
        &a_d,
        &dt_d,
        lower,
        rows,
        HIDDEN,
        HEADS,
        &mut g_decay,
        &mut z_gate,
        &mut beta,
    )
    .unwrap();

    let g_decay = dl::<f32>(&g_decay, &st);
    let z_gate = dl::<f32>(&z_gate, &st);
    let beta = dl::<f32>(&beta, &st);
    for row in 0..rows {
        // CPU reference
        let mut beta_cpu = vec![0f32; HEADS];
        let xr: Vec<f32> = (0..HIDDEN).map(|j| bff(x[row * HIDDEN + j])).collect();
        for h in 0..HEADS {
            let mut s = 0f64;
            for j in 0..HIDDEN {
                s += xr[j] as f64 * bff(b_weight[h * HIDDEN + j]) as f64;
            }
            beta_cpu[h] = (1.0 / (1.0 + (-s).exp())) as f32;
        }
        for c in 0..C {
            let h = c / 128;
            let fv = gb_exl3::f16::f16_to_f64(f_out[row * C + c]);
            let decay = (a_log[h] as f64).exp();
            let g = ((lower as f64) * (1.0 / (1.0 + (-(decay * (fv + dt_bias[c] as f64))).exp()))).exp();
            // decay and z are copied/transformed exactly; compare with tight tolerances
            assert!((g_decay[row * C + c] as f64 - g).abs() < 1e-3, "g_decay row {row} c {c}: {} vs {g}", g_decay[row * C + c]);
            assert_eq!(
                z_gate[row * C + c],
                gb_exl3::f16::f16_to_f64(g_out[row * C + c]) as f32,
                "z must be the fp32 copy of g_out"
            );
        }
        // beta: f32(bf16(sigmoid(dot))); the dot order differs from the kernel's fmaf chain and beta is bf16-rounded, allow an ulp or two
        for h in 0..HEADS {
            let got = beta[row * HEADS + h];
            let want = beta_cpu[h];
            assert!((got - want).abs() < 5e-3, "beta row {row} h {h}: {got} vs {want}");
        }
    }

    // row invariance: rows 0..7 of the 32-row call equal the same rows of a 8-row call
    let rows8 = 8usize;
    let x8 = DevBuf::from_slice(&x[..rows8 * HIDDEN]).unwrap();
    let f8 = DevBuf::from_slice(&f_out[..rows8 * C]).unwrap();
    let g8 = DevBuf::from_slice(&g_out[..rows8 * C]).unwrap();
    let mut g_decay8 = DevBuf::zeroed(rows8 * C * 4).unwrap();
    let mut z_gate8 = DevBuf::zeroed(rows8 * C * 4).unwrap();
    let mut beta8 = DevBuf::zeroed(rows8 * HEADS * 4).unwrap();
    gb_cuda::kda::gate_from_full_fg(
        &st,
        &f8,
        &g8,
        &x8,
        &b_d,
        &a_d,
        &dt_d,
        lower,
        rows8,
        HIDDEN,
        HEADS,
        &mut g_decay8,
        &mut z_gate8,
        &mut beta8,
    )
    .unwrap();
    let g_decay8 = dl::<f32>(&g_decay8, &st);
    let beta8 = dl::<f32>(&beta8, &st);
    assert_eq!(&g_decay8[..rows8 * C], &g_decay[..rows8 * C], "g_decay row invariance");
    assert_eq!(&beta8[..rows8 * HEADS], &beta[..rows8 * HEADS], "beta row invariance");
}
