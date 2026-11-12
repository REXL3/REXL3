//! Wave-6b: the in-CTA shared-memory K-reduction kernels (`gate_up_kernel_smem`, `down_kernel_smem`) must be
//! bit-exact with the legacy global-partial kernels: the reduction order is the same ascending kc float32 chain,
//! so every output bit of the fused path must equal the legacy path's, at every row count.
//!
//! Two shapes cover the interesting template counts:
//!   * Qwen3.8-Flash-Next (fn_moe.rs): H 2560, I 640, KC 10 -> gate_up_smem<10>, down_smem<5>
//!   * MiniCPM-V-4.7 (minicpm_moe.rs): H 2048, I 512, KC 4 -> gate_up_smem<4>, down_smem<4>
//! Each file declares its own constants; this one runs both by including cases.rs twice with different names is not
//! possible (include! of a module is per-file), so the two shapes live in two targets below.
mod common;
const H: usize = 2560;
const I: usize = 640;
const NE: usize = 512;
const TK: usize = 10;
const KC: usize = fmoe::DEFAULT_KC; // 10, valid at H = 2560
const SEED_STRIDE: usize = 1000;
fn bits_of(matrix: usize, expert: usize) -> usize {
    3 + (matrix * 5 + expert * 3) % 6
}
include!("fn_moe/cases.rs");

/// Every supported row count: the fused smem path and the legacy global-partial path produce the same bits.
#[test]
#[ignore]
fn smem_path_is_bit_exact_with_legacy() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let inp = inputs();
    // prime the smem path (cudaFuncSetAttribute) and check both directions of the toggle
    let prev = fmoe::set_legacy(false);
    for t in [1usize, 8, 64] {
        let fused = run_frost(&o, &inp, t, KC, &st).unwrap();
        assert_eq!(fmoe::set_legacy(true), false, "toggle was not false before");
        let legacy = run_frost(&o, &inp, t, KC, &st).unwrap();
        assert_eq!(fmoe::set_legacy(false), true, "toggle was not true after enabling legacy");
        // determinism probe: the fused path must be repeatable (a race would add run-to-run noise)
        let fused2 = run_frost(&o, &inp, t, KC, &st).unwrap();
        assert_eq!(fused, fused2, "t={t}: fused smem path is not deterministic (run-to-run)");
        if fused != legacy {
            let n = fused.iter().zip(&legacy).filter(|(a, b)| a != b).count();
            let first = fused.iter().zip(&legacy).position(|(a, b)| a != b).unwrap_or(usize::MAX);
            let total = t * H;
            eprintln!(
                "t={t}: fused vs legacy differ at {n}/{total} u16, first at index {first} (row {}, col {})",
                first / H,
                (first % H) * 2
            );
            let (fa, la) = (fused[first], legacy[first]);
            eprintln!("  fused={fa} legacy={la} (bf16 0x{fa:04x} vs 0x{la:04x})");
        }
        assert_eq!(fused, legacy, "t={t}: in-CTA smem path differs from the legacy global-partial path (bits)");
    }
    // the toggle returns to whatever it was
    let _ = fmoe::set_legacy(prev);
}

/// The fused path alone is still row-invariant and within the packed-weight CPU reference (the same checks the
/// legacy kernels are held to).
#[test]
fn smem_path_row_invariance_and_cpu_reference() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let inp = inputs();
    let full = run_frost(&o, &inp, 64, KC, &st).unwrap();
    for t in [1usize, 8, 37] {
        let g = run_frost(&o, &inp, t, KC, &st).unwrap();
        assert_eq!(g, full[..t * H], "smem path row invariance at t={t}");
    }
    let one = run_frost(&o, &inp, 1, KC, &st).unwrap();
    let cpu = cpu_row(&o, &inp, &one_sel(&o, &inp, &st), &one_rw(&o, &inp, &st));
    let err = rel(&one, &cpu);
    assert!(err < 0.03, "smem path vs the CPU reference: rel_l2 {err}");
}

/// Crude same-machine timing of the smem path vs the legacy global-partial path (T=8, one layer). Not a
/// substitute for nsys on the real decode: it isolates the kernel change on synthetic weights.
#[test]
#[ignore]
fn smem_vs_legacy_wallclock_probe() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let inp = inputs();
    let t = 8usize;
    // warm up (set attributes, caches)
    let _ = run_frost(&o, &inp, t, KC, &st).unwrap();
    assert_eq!(fmoe::set_legacy(true), false);
    let _ = run_frost(&o, &inp, t, KC, &st).unwrap();
    assert_eq!(fmoe::set_legacy(false), true);
    let mut smem = f64::MAX;
    let mut legacy = f64::MAX;
    for _ in 0..5 {
        let s0 = std::time::Instant::now();
        for _ in 0..20 {
            let _ = run_frost(&o, &inp, t, KC, &st).unwrap();
        }
        let d1 = s0.elapsed().as_secs_f64();
        assert_eq!(fmoe::set_legacy(true), false);
        let s1 = std::time::Instant::now();
        for _ in 0..20 {
            let _ = run_frost(&o, &inp, t, KC, &st).unwrap();
        }
        let d2 = s1.elapsed().as_secs_f64();
        assert_eq!(fmoe::set_legacy(false), true);
        smem = smem.min(d1 / 20.0);
        legacy = legacy.min(d2 / 20.0);
    }
    eprintln!("T=8 one layer: smem path {smem:.4} ms/call, legacy {legacy:.4} ms/call ({:.1}%)", smem / legacy * 100.0);
}

fn one_sel(o: &Owned, inp: &Inputs, st: &Stream) -> Vec<u32> {
    let x = DevBuf::from_slice(&inp.x[..H]).unwrap();
    let logits = DevBuf::from_slice(&inp.logits[..NE]).unwrap();
    let sg = DevBuf::from_slice(&inp.sg[..1]).unwrap();
    let mut sel = DevBuf::zeroed(1 * (TK + 1) * 4).unwrap();
    let mut rw = DevBuf::zeroed(1 * (TK + 1) * 4).unwrap();
    fn_moe::moe_topk_softmax(&logits, Some(&sg), &mut sel, &mut rw, NE, TK, 1, st).unwrap();
    dl(&sel, st)
}
fn one_rw(o: &Owned, inp: &Inputs, st: &Stream) -> Vec<f32> {
    let x = DevBuf::from_slice(&inp.x[..H]).unwrap();
    let logits = DevBuf::from_slice(&inp.logits[..NE]).unwrap();
    let sg = DevBuf::from_slice(&inp.sg[..1]).unwrap();
    let mut sel = DevBuf::zeroed(1 * (TK + 1) * 4).unwrap();
    let mut rw = DevBuf::zeroed(1 * (TK + 1) * 4).unwrap();
    fn_moe::moe_topk_softmax(&logits, Some(&sg), &mut sel, &mut rw, NE, TK, 1, st).unwrap();
    dl(&rw, st)
}
