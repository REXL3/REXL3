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

fn run_frost_row_staged(o: &Owned, inp: &Inputs, t: usize, st: &Stream) -> Vec<u16> {
    let x = DevBuf::from_slice(&inp.x[..t * H]).unwrap();
    let logits = DevBuf::from_slice(&inp.logits[..t * NE]).unwrap();
    let sg = DevBuf::from_slice(&inp.sg[..t]).unwrap();
    let shape = fmoe::Shape { t, ne: NE, top_k: TK, h: H, i: I, kc: KC, max_tokens: 64 };
    let mut frost = DevBuf::zeroed(t * H * 2).unwrap();
    let mut ws = fmoe::workspace(64, TK, H, I, KC).unwrap();
    fmoe::forward_row_staged(
        &x,
        &logits,
        fmoe::LogitsDtype::Bf16,
        &sg,
        &o.tables(),
        &mut frost,
        shape,
        None,
        fmoe::SwigluLimit::default(),
        &mut ws,
        st,
    )
    .unwrap();
    dl(&frost, st)
}

/// The row-staged entry (`fmoe::forward_row_staged`: min(T, 8) staged token rows and two CTAs an SM at one to four rows, the stock
/// launch above) writes the bits of `fmoe::forward` at every row count, across the one-to-four edge and the 8-token group split,
/// and repeats.
#[test]
fn row_staged_entry_is_bit_exact_with_forward() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let inp = inputs();
    for t in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 33, 64] {
        let stock = run_frost(&o, &inp, t, KC, &st).unwrap();
        let staged = run_frost_row_staged(&o, &inp, t, &st);
        let again = run_frost_row_staged(&o, &inp, t, &st);
        let n = stock.iter().zip(&staged).filter(|(a, b)| a != b).count();
        assert_eq!(n, 0, "t={t}: row-staged entry differs from fmoe::forward at {n}/{} bf16 outputs", t * H);
        assert_eq!(staged, again, "t={t}: row-staged entry is not repeatable");
    }
}

/// Kernel A/B timing at one to eight rows, one MoE layer (route + gate/up + down), weights at the pack's widths (routed experts
/// 3-bit, the shared expert 5-bit): the stock `fmoe::forward`, the row-staged entry, and the legacy split-K gate/up
/// (`set_legacy(true)`). 16 windows of different rows rotate the expert picks; a cell is the median of 9 timings of 64 calls.
/// Also checks the three arms' bits are equal on every window. Run alone (`--test-threads=1`: it flips the global legacy toggle).
/// `FMOE_PROBE_T` (comma list, default 1..=8) picks the row counts and `FMOE_PROBE_REPS=0` skips the timing (an ncu run of the
/// bit loop: 16 windows x 3 arms of gate/up + down a row count).
#[test]
#[ignore]
fn row_staged_vs_stock_vs_legacy_probe() {
    let Some(st) = gpu() else { return };
    let o = Owned::with_bits(|_, e| if e == NE { 5 } else { 3 });
    let mut r = Rng(0x5eed_f00d);
    const WIN: usize = 16;
    let x: Vec<u16> = (0..WIN * 8 * H).map(|_| bf(r.uniform(-0.2, 0.2))).collect();
    let logits: Vec<u16> = (0..WIN * 8 * NE).map(|_| bf(r.uniform(-2., 2.))).collect();
    let sg: Vec<u16> = (0..WIN * 8).map(|_| bf(r.uniform(-1., 1.))).collect();
    let mut ws = fmoe::workspace(64, TK, H, I, KC).unwrap();
    let prev = fmoe::set_legacy(false);
    let ts: Vec<usize> =
        std::env::var("FMOE_PROBE_T").map_or((1..=8).collect(), |v| v.split(',').map(|x| x.trim().parse().unwrap()).collect());
    let reps: usize = std::env::var("FMOE_PROBE_REPS").map_or(9, |v| v.parse().unwrap());
    for t in ts {
        let bufs: Vec<(DevBuf, DevBuf, DevBuf)> = (0..WIN)
            .map(|w| {
                let o0 = w * 8;
                (
                    DevBuf::from_slice(&x[o0 * H..(o0 + t) * H]).unwrap(),
                    DevBuf::from_slice(&logits[o0 * NE..(o0 + t) * NE]).unwrap(),
                    DevBuf::from_slice(&sg[o0..o0 + t]).unwrap(),
                )
            })
            .collect();
        let shape = fmoe::Shape { t, ne: NE, top_k: TK, h: H, i: I, kc: KC, max_tokens: 64 };
        let mut out = DevBuf::zeroed(t * H * 2).unwrap();
        let tb = o.tables();
        let call = |arm: usize, w: usize, out: &mut DevBuf, ws: &mut DevBuf| {
            let (bx, bl, bs) = &bufs[w];
            if arm == 1 {
                fmoe::forward_row_staged(
                    bx,
                    bl,
                    fmoe::LogitsDtype::Bf16,
                    bs,
                    &tb,
                    out,
                    shape,
                    None,
                    fmoe::SwigluLimit::default(),
                    ws,
                    &st,
                )
                .unwrap();
            } else {
                fmoe::set_legacy(arm == 2);
                fmoe::forward(bx, bl, fmoe::LogitsDtype::Bf16, bs, &tb, out, shape, None, fmoe::SwigluLimit::default(), ws, &st)
                    .unwrap();
            }
        };
        // bits: the three arms agree on every window
        for w in 0..WIN {
            let mut outs = Vec::new();
            for arm in 0..3 {
                call(arm, w, &mut out, &mut ws);
                outs.push(dl::<u16>(&out, &st));
            }
            assert_eq!(outs[0], outs[1], "t={t} window {w}: row-staged differs from stock");
            assert_eq!(outs[0], outs[2], "t={t} window {w}: legacy differs from stock");
        }
        if reps == 0 {
            continue;
        }
        let mut cell = [0f64; 3];
        for (arm, c) in cell.iter_mut().enumerate() {
            let mut v = Vec::new();
            for _ in 0..reps {
                st.sync().unwrap();
                let s0 = std::time::Instant::now();
                for k in 0..64 {
                    call(arm, k % WIN, &mut out, &mut ws);
                }
                st.sync().unwrap();
                v.push(s0.elapsed().as_secs_f64() * 1e6 / 64.0);
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            *c = v[reps / 2];
        }
        eprintln!(
            "probe t={t}: stock {:.1} us, row-staged {:.1} us ({:+.1}%), legacy split-K {:.1} us ({:+.1}%) a layer (route + gate/up + down)",
            cell[0],
            cell[1],
            (cell[1] / cell[0] - 1.0) * 100.0,
            cell[2],
            (cell[2] / cell[0] - 1.0) * 100.0
        );
    }
    fmoe::set_legacy(prev);
}
