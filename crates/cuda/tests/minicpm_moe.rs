//! The MoE cases of tests/fn_moe.rs at MiniCPM-V-4.7's layer shape (the Qwen3.5/3.6-35B-A3B text backbone): hidden 2048, expert
//! intermediate 512, 256 routed experts + the shared expert (id 256, sigmoid gate), top-8, every EXL3 width 3..8 mixed across the
//! experts' gate/up/down matrices. The decode kernels (CoopMK, FrostMoE decode and prefill) against the packed-weight CPU reference
//! and bit-exact row invariance (one row's result is the same in a call of 1, 7, 37 or 64 rows, at any position, whatever the other
//! rows pick), the router at 256 experts with ties, the K-chunk counts FrostMoE accepts at this width, and an (ignored) K-chunk sweep.
//! GPU tests skip when there is no device.
mod common;
const H: usize = 2048;
const I: usize = 512;
const NE: usize = 256;
const TK: usize = 8;
/// FrostMoE's K chunks of the gate/up pass: the shipping count (the sweep below chose it; 16 blocks of 128 must be a multiple of it, and
/// the Flash-Next default of 10 is not).
const KC: usize = fmoe::MINICPM_KC;
const SEED_STRIDE: usize = 1000;
/// Widths 3..=8 spread over the experts and the three matrices (the runtime-K decoder takes each width's own code path).
fn bits_of(matrix: usize, expert: usize) -> usize {
    3 + (matrix * 5 + expert * 3) % 6
}
include!("fn_moe/cases.rs");

use std::time::Instant;

/// The K chunks FrostMoE accepts at H = 2048 are the divisors of 16 whose chunks fit the 48 KiB of staging (4, 8, 16: `fmoe::kc_valid`);
/// every one is row invariant on its own (the chunk partials are added in chunk order, so the bits differ between counts but never
/// between row counts) and within the CPU reference's tolerance; any other count, the Flash-Next default 10 and the 1 and 2 that would
/// stage 64 and 128 KiB among them, is refused before anything is queued.
#[test]
#[ignore]
fn fmoe_chunk_counts_at_h2048_are_row_invariant_and_the_flash_next_default_is_refused() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let inp = inputs();
    let cpu_in = {
        let g = run(&o, &inp, 1, &st);
        cpu_row(&o, &inp, &g.sel, &g.rw)
    };
    let mut first: Option<Vec<u16>> = None;
    for kc in [4usize, 8, 16] {
        assert!(fmoe::kc_valid(H, kc));
        let full = run_frost(&o, &inp, 64, kc, &st).unwrap();
        for t in [1usize, 4, 8, 16, 37] {
            let g = run_frost(&o, &inp, t, kc, &st).unwrap();
            assert_eq!(g, &full[..t * H], "FrostMoE kc={kc}: {t} rows differ from the same rows of a 64-row call");
        }
        let one = run_frost(&o, &inp, 1, kc, &st).unwrap();
        let err = rel(&one, &cpu_in);
        assert!(err < 0.03, "FrostMoE kc={kc} vs the CPU reference: rel_l2 {err}");
        // different chunk counts sum the same partials in another grouping: close, not equal
        match &first {
            None => first = Some(one),
            Some(f) => assert!(
                rel(&one, &f.iter().map(|&b| bff(b)).collect::<Vec<f32>>()) < 5e-3,
                "kc={kc} is far from kc=4 (a grouping change must not move the result by more than rounding)"
            ),
        }
    }
    assert_eq!(fmoe::DEFAULT_KC, 10);
    for kc in [1usize, 2, 3, 5, 6, 10, 32] {
        assert!(!fmoe::kc_valid(H, kc));
        assert!(run_frost(&o, &inp, 4, kc, &st).is_err(), "kc={kc} is no valid chunk count at H = 2048 and must be refused");
    }
}

/// The router at the real expert count: 256 logits a row, top-8 by (value DESC, index ASC) with heavy ties (logits on a quarter grid),
/// softmax over the selected eight, the shared expert (id 256) weighted by the sigmoid of its gate; exact against the CPU.
#[test]
#[ignore]
fn router_at_256_experts_top8_breaks_ties_to_the_lowest_index() {
    let Some(st) = gpu() else { return };
    let mut r = Rng(0x2565);
    let t = 64;
    let q = |v: f32| (v * 4.0).round() / 4.0;
    let logits: Vec<u16> = (0..t * NE).map(|_| bf(q(r.uniform(-4., 4.)))).collect();
    let sg: Vec<u16> = (0..t).map(|_| bf(r.uniform(-3., 3.))).collect();
    let (ld, sd) = (DevBuf::from_slice(&logits).unwrap(), DevBuf::from_slice(&sg).unwrap());
    let mut sel = DevBuf::zeroed(t * (TK + 1) * 4).unwrap();
    let mut rw = DevBuf::zeroed(t * (TK + 1) * 4).unwrap();
    fn_moe::moe_topk_softmax(&ld, Some(&sd), &mut sel, &mut rw, NE, TK, t, &st).unwrap();
    let (sel, rw): (Vec<u32>, Vec<f32>) = (dl(&sel, &st), dl(&rw, &st));
    let mut ties = 0;
    for row in 0..t {
        let l: Vec<f32> = (0..NE).map(|e| bff(logits[row * NE + e])).collect();
        let (want_sel, want_w) = cpu_route(&l, TK);
        // ties at the cut: the eighth value also occurs outside the selection with a higher index
        ties += (0..NE).filter(|e| !want_sel.contains(e) && l[*e] == l[want_sel[TK - 1]]).count().min(1);
        for k in 0..TK {
            assert_eq!(sel[row * (TK + 1) + k] as usize, want_sel[k], "row {row} rank {k}");
            assert!((rw[row * (TK + 1) + k] as f64 - want_w[k]).abs() < 5e-6, "row {row} weight {k}");
        }
        assert_eq!(sel[row * (TK + 1) + TK] as usize, NE);
        assert!((rw[row * (TK + 1) + TK] - 1. / (1. + (-bff(sg[row])).exp())).abs() < 1e-6, "row {row} shared weight");
    }
    assert!(ties >= 8, "the logits must produce ties at the cut ({ties} rows)");
}

/// CoopMK is instantiated for two (H, I) pairs; any other is refused with an invalid-value error instead of reading past its
/// shared-memory tiles (the literals of the old kernel were wrong for every other width).
#[test]
#[ignore]
fn coopmk_refuses_the_widths_it_is_not_instantiated_for() {
    let Some(st) = gpu() else { return };
    assert_eq!(fn_moe::COOPMK_SHAPES, [(2560, 640), (2048, 512), (3072, 1024)]);
    let o = Owned::new();
    let x = DevBuf::zeroed(2 * 2560 * 2).unwrap();
    let sel = DevBuf::zeroed(2 * (TK + 1) * 4).unwrap();
    let rw = DevBuf::zeroed(2 * (TK + 1) * 4).unwrap();
    let mut out = DevBuf::zeroed(2 * 2560 * 2).unwrap();
    for (h, i) in [(2048usize, 640usize), (2560, 512), (1024, 512), (4096, 512), (2048, 256)] {
        let mut act = DevBuf::zeroed(fn_moe::coopmk_workspace_bytes(2, TK + 1, i)).unwrap();
        let r = fn_moe::coopmk_decode(&x, &sel, &rw, &o.tables(), &mut act, &mut out, h, i, TK + 1, 2, &st);
        assert!(r.is_err(), "coopmk accepted H={h} I={i}");
    }
}

fn ms_per_call(st: &Stream, iters: usize, mut f: impl FnMut(usize)) -> f64 {
    for i in 0..8 {
        f(i);
    }
    st.sync().unwrap();
    let t0 = Instant::now();
    for i in 0..iters {
        f(i);
    }
    st.sync().unwrap();
    t0.elapsed().as_secs_f64() * 1e3 / iters as f64
}

/// The K-chunk sweep (alone under the GPU lock): the decode MoE of one layer at 1, 4, 8, 16, 32 and 64 rows, FrostMoE at every valid KC (4, 8,
/// 16) and CoopMK, 256 experts top-8, mixed widths 3..8, a fresh set of rows each call so the weights are read from
/// DRAM like a layer of a real decode step (the pool of 64 row sets cycles through most of the layer's expert weights, which are far
/// beyond the L2).
///
///     flock /tmp/rexl3-gpu.lock cargo test --release -p gb-cuda --test minicpm_moe bench_decode_moe_kc_sweep -- --ignored --nocapture
#[test]
#[ignore]
fn bench_decode_moe_kc_sweep() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let weight_bytes: usize = (0..E).map(|e| (0..3).map(|p| H * I * bits_of(p, e) / 8).sum::<usize>()).sum();
    eprintln!("layer weights {:.0} MB, {NE} experts, top-{TK} + shared, H {H} I {I}", weight_bytes as f64 / 1e6);
    let mut r = Rng(0xbe9c);
    for rows in [1usize, 4, 8, 16, 32, 64] {
        // 64 distinct row sets
        let sets: Vec<(DevBuf, DevBuf, DevBuf)> = (0..64)
            .map(|_| {
                let x: Vec<u16> = (0..rows * H).map(|_| bf(r.uniform(-0.2, 0.2))).collect();
                let l: Vec<u16> = (0..rows * NE).map(|_| bf(r.uniform(-2., 2.))).collect();
                let g: Vec<u16> = (0..rows).map(|_| bf(r.uniform(-1., 1.))).collect();
                (DevBuf::from_slice(&x).unwrap(), DevBuf::from_slice(&l).unwrap(), DevBuf::from_slice(&g).unwrap())
            })
            .collect();
        let mut line = format!("rows {rows:>2}:");
        for kc in [4usize, 8, 16] {
            let mut ws = fmoe::workspace(64, TK, H, I, kc).unwrap();
            let mut out = DevBuf::zeroed(rows * H * 2).unwrap();
            let shape = fmoe::Shape { t: rows, ne: NE, top_k: TK, h: H, i: I, kc, max_tokens: 64 };
            let ms = ms_per_call(&st, 192, |i| {
                let (x, l, g) = &sets[i % 64];
                fmoe::forward(x, l, g, &o.tables(), &mut out, shape, None, fmoe::SwigluLimit::default(), &mut ws, &st).unwrap();
            });
            line += &format!("  frost kc={kc:<2} {:7.1} us ({:5.1} us/row)", ms * 1e3, ms * 1e3 / rows as f64);
        }
        let mut sel = DevBuf::zeroed(rows * (TK + 1) * 4).unwrap();
        let mut rw = DevBuf::zeroed(rows * (TK + 1) * 4).unwrap();
        let mut act = DevBuf::zeroed(fn_moe::coopmk_workspace_bytes(rows, TK + 1, I)).unwrap();
        let mut out = DevBuf::zeroed(rows * H * 2).unwrap();
        let ms = ms_per_call(&st, 192, |i| {
            let (x, l, g) = &sets[i % 64];
            fn_moe::moe_topk_softmax(l, Some(g), &mut sel, &mut rw, NE, TK, rows, &st).unwrap();
            fn_moe::coopmk_decode(x, &sel, &rw, &o.tables(), &mut act, &mut out, H, I, TK + 1, rows, &st).unwrap();
        });
        line += &format!("  coopmk {:7.1} us ({:5.1} us/row)", ms * 1e3, ms * 1e3 / rows as f64);
        eprintln!("{line}");
    }
}
