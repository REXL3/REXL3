//! The MiniCPM-V-4.7 prefill MoE at chunk scale (E6a): `fmoe::prefill` (`gb_fmoep_forward`, the sorted-token grouped GEMM) at hidden 2048,
//! expert intermediate 512, 256 routed experts top-8 + the shared expert, against the contract the engine's prefill rests on:
//!
//! * chunk invariance: a prompt's rows leave the same bits whatever grid they were cut into (1 row to 4,096, odd sizes), because the
//!   prompt cache and `chunked_prefill_equals_whole` compare bits;
//! * row invariance: a row's bits do not depend on the other rows of the call, on its position in the call, or on how many rows
//!   picked the same expert (a tile full of one expert's tokens against a lone token);
//! * the packed-weight CPU reference and CoopMK (the oracle of E3) within the tolerances of the decode kernels.
//!
//! The weights are the synthetic packed EXL3 layer of `tests/fn_moe/cases.rs` (widths 3..8 mixed across the matrices, so every
//! runtime-K decoder runs). GPU tests skip without a device; all are `#[ignore]` (the gate runs them with `--include-ignored`).
mod common;
const H: usize = 2048;
const I: usize = 512;
const NE: usize = 256;
const TK: usize = 8;
const KC: usize = fmoe::MINICPM_KC;
const SEED_STRIDE: usize = 1000;
fn bits_of(matrix: usize, expert: usize) -> usize {
    3 + (matrix * 5 + expert * 3) % 6
}
include!("fn_moe/cases.rs");

use std::time::Instant;

/// Rows of the widest prefill chunk the engine runs (the model's `prefill_rows`).
const CHUNK: usize = 4096;

struct Rows {
    x: Vec<u16>,
    logits: Vec<u16>,
    sg: Vec<u16>,
}

/// `t` random rows. `skew` adds a per-expert bias (std about `skew`) to the router logits: real routers are far from balanced, and a
/// bias makes some experts take many tiles and others none.
fn rows(t: usize, seed: u64, skew: f32) -> Rows {
    let mut r = Rng(seed);
    let bias: Vec<f32> = (0..NE).map(|_| skew * (r.uniform(-1., 1.) + r.uniform(-1., 1.) + r.uniform(-1., 1.))).collect();
    Rows {
        x: (0..t * H).map(|_| bf(r.uniform(-0.2, 0.2))).collect(),
        logits: (0..t * NE).map(|i| bf(r.uniform(-2., 2.) + bias[i % NE])).collect(),
        sg: (0..t).map(|_| bf(r.uniform(-1., 1.))).collect(),
    }
}

impl Rows {
    fn slice(&self, a: usize, n: usize) -> Rows {
        Rows {
            x: self.x[a * H..(a + n) * H].to_vec(),
            logits: self.logits[a * NE..(a + n) * NE].to_vec(),
            sg: self.sg[a..a + n].to_vec(),
        }
    }
    fn t(&self) -> usize {
        self.sg.len()
    }
}

fn prefill_ws(max_tokens: usize) -> DevBuf {
    fmoe::prefill_workspace(max_tokens, TK, NE, H, I).unwrap()
}

/// `fmoe::prefill` of `r` as one call against a workspace sized for `max_tokens` rows.
fn call(o: &Owned, r: &Rows, ws: &mut DevBuf, max_tokens: usize, st: &Stream) -> Vec<u16> {
    let t = r.t();
    let x = DevBuf::from_slice(&r.x).unwrap();
    let logits = DevBuf::from_slice(&r.logits).unwrap();
    let sg = DevBuf::from_slice(&r.sg).unwrap();
    let mut out = DevBuf::zeroed(t * H * 2).unwrap();
    let shape = fmoe::Shape { t, ne: NE, top_k: TK, h: H, i: I, kc: KC, max_tokens };
    fmoe::prefill(&x, &logits, &sg, &o.tables(), &mut out, shape, None, fmoe::SwigluLimit::default(), ws, st).unwrap();
    dl(&out, st)
}

/// The same rows cut into calls of `grid` rows (the last one shorter), concatenated.
fn by_grid(o: &Owned, r: &Rows, grid: usize, ws: &mut DevBuf, st: &Stream) -> Vec<u16> {
    let mut out = Vec::with_capacity(r.t() * H);
    let mut at = 0;
    while at < r.t() {
        let n = grid.min(r.t() - at);
        out.extend(call(o, &r.slice(at, n), ws, CHUNK, st));
        at += n;
    }
    out
}

fn first_difference(a: &[u16], b: &[u16]) -> Option<(usize, usize)> {
    a.iter().zip(b).position(|(x, y)| x != y).map(|i| (i / H, i % H))
}

/// CoopMK on the same rows (the router in its own kernel): the oracle E3's gate was set on.
fn coopmk(o: &Owned, r: &Rows, st: &Stream) -> Vec<u16> {
    let t = r.t();
    let x = DevBuf::from_slice(&r.x).unwrap();
    let logits = DevBuf::from_slice(&r.logits).unwrap();
    let sg = DevBuf::from_slice(&r.sg).unwrap();
    let mut sel = DevBuf::zeroed(t * (TK + 1) * 4).unwrap();
    let mut rw = DevBuf::zeroed(t * (TK + 1) * 4).unwrap();
    let mut out = DevBuf::zeroed(t * H * 2).unwrap();
    fn_moe::moe_topk_softmax(&logits, Some(&sg), &mut sel, &mut rw, NE, TK, t, st).unwrap();
    let mut act = DevBuf::zeroed(fn_moe::coopmk_workspace_bytes(t, TK + 1, I)).unwrap();
    fn_moe::coopmk_decode(&x, &sel, &rw, &o.tables(), &mut act, &mut out, H, I, TK + 1, t, st).unwrap();
    dl(&out, st)
}

fn rel_rows(a: &[u16], b: &[u16]) -> (f64, f64) {
    let t = a.len() / H;
    let (mut worst, mut sum) = (0f64, 0f64);
    for r in 0..t {
        let (mut e, mut n) = (0f64, 0f64);
        for j in 0..H {
            let (x, y) = (f64::from(bff(a[r * H + j])), f64::from(bff(b[r * H + j])));
            e += (x - y) * (x - y);
            n += y * y;
        }
        let v = (e / n.max(1e-30)).sqrt();
        worst = worst.max(v);
        sum += v;
    }
    (worst, sum / t as f64)
}

/// A whole 4,096-row chunk against the same rows cut into grids of 1, 3, 7, 64, 97, 512, 1000, 2047 and 4095 rows, on balanced and on
/// skewed routing: every row's bits equal. The workspace is the one sized for the widest chunk in every call (as the model's is).
#[test]
#[ignore]
fn prefill_moe_leaves_the_same_bits_in_any_chunk_grid() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let mut ws = prefill_ws(CHUNK);
    for (seed, skew) in [(0x6001u64, 0.0f32), (0x6002, 1.2)] {
        let r = rows(CHUNK, seed, skew);
        let whole = call(&o, &r, &mut ws, CHUNK, &st);
        assert!(whole.iter().any(|&v| v != 0), "an all-zero output proves nothing");
        for grid in [1usize, 3, 7, 64, 97, 512, 1000, 2047, 4095] {
            let cut = by_grid(&o, &r, grid, &mut ws, &st);
            assert!(
                cut == whole,
                "skew {skew}: grid {grid} differs from the whole chunk at (row, col) {:?}",
                first_difference(&cut, &whole)
            );
        }
        eprintln!("skew {skew}: 4,096 rows identical in 9 chunk grids");
    }
}

/// A row's bits do not depend on the workspace's size (the model sizes it for the widest chunk), on the rows around it, on its
/// position in the call, or on a call that is a permutation of the rows.
#[test]
#[ignore]
fn prefill_moe_row_invariance_and_workspace_independence() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let r = rows(CHUNK, 0x6003, 0.8);
    let mut ws_wide = prefill_ws(CHUNK);
    let whole = call(&o, &r, &mut ws_wide, CHUNK, &st);
    // 1. the same 37 rows in a workspace sized for 64 rows, and sized for 4,096
    let mut ws_small = prefill_ws(64);
    let head = r.slice(0, 37);
    let a = call(&o, &head, &mut ws_small, 64, &st);
    let b = call(&o, &head, &mut ws_wide, CHUNK, &st);
    assert_eq!(a, b, "the workspace size moved a bit");
    assert_eq!(a, &whole[..37 * H], "37 rows alone differ from the same rows inside 4,096");
    // 2. a permutation of the rows: reversed, and a stride-7 shuffle
    for (name, perm) in [
        ("reversed", (0..CHUNK).rev().collect::<Vec<usize>>()),
        ("stride 7", (0..CHUNK).map(|i| (i * 7) % CHUNK).collect::<Vec<usize>>()),
    ] {
        let p = Rows {
            x: perm.iter().flat_map(|&i| r.x[i * H..(i + 1) * H].iter().copied()).collect(),
            logits: perm.iter().flat_map(|&i| r.logits[i * NE..(i + 1) * NE].iter().copied()).collect(),
            sg: perm.iter().map(|&i| r.sg[i]).collect(),
        };
        let out = call(&o, &p, &mut ws_wide, CHUNK, &st);
        for (pos, &src) in perm.iter().enumerate() {
            assert_eq!(
                &out[pos * H..(pos + 1) * H],
                &whole[src * H..(src + 1) * H],
                "{name}: row {src} moved to {pos} changed bits"
            );
        }
    }
    // 3. a tile full of one expert's tokens: 200 copies of one row (every expert it picks sees 200 tokens: three full tiles and a
    // partial one) against that row alone and against its first copy in a call of unrelated rows
    let one = r.slice(5, 1);
    let alone = call(&o, &one, &mut ws_wide, CHUNK, &st);
    let many = Rows {
        x: (0..200).flat_map(|_| one.x.iter().copied()).collect(),
        logits: (0..200).flat_map(|_| one.logits.iter().copied()).collect(),
        sg: vec![one.sg[0]; 200],
    };
    let out = call(&o, &many, &mut ws_wide, CHUNK, &st);
    for c in 0..200 {
        assert_eq!(&out[c * H..(c + 1) * H], &alone[..], "copy {c} of a row picked by 200 rows differs from the row alone");
    }
    assert_eq!(&whole[5 * H..6 * H], &alone[..], "row 5 inside 4,096 rows differs from row 5 alone");
    // the copies mixed into unrelated rows: rows 0..4096 with 200 copies of row 5 appended (4,096 total: replace the tail)
    let mut mixed = r.slice(0, CHUNK - 200);
    mixed.x.extend(many.x.iter());
    mixed.logits.extend(many.logits.iter());
    mixed.sg.extend(many.sg.iter());
    let out = call(&o, &mixed, &mut ws_wide, CHUNK, &st);
    assert_eq!(&out[..(CHUNK - 200) * H], &whole[..(CHUNK - 200) * H], "changing the last 200 rows moved an earlier row");
    for c in 0..200 {
        let at = (CHUNK - 200 + c) * H;
        assert_eq!(&out[at..at + H], &alone[..], "copy {c} among unrelated rows differs from the row alone");
    }
}

/// The new arithmetic against the packed-weight CPU reference (rows spread over the chunk) and against CoopMK on the same rows, at
/// 4,096 rows: the rel_l2 of the two bf16 outputs is the rounding of two accumulation orders, nothing structural.
#[test]
#[ignore]
fn prefill_moe_matches_the_cpu_reference_and_coopmk_at_chunk_scale() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let r = rows(CHUNK, 0x6004, 0.8);
    let mut ws = prefill_ws(CHUNK);
    let got = call(&o, &r, &mut ws, CHUNK, &st);
    let oracle = coopmk(&o, &r, &st);
    let (worst, mean) = rel_rows(&got, &oracle);
    eprintln!("fmoep vs CoopMK, 4,096 rows: rel_l2 per row worst {worst:.3e} mean {mean:.3e}");
    assert!(worst < 0.03 && mean < 0.01, "fmoep is far from CoopMK: worst {worst}, mean {mean}");
    // the CPU reference: a row's routing from the same router kernel (it is the kernel the chunk runs), its experts on the CPU
    let sel_rw = |row: usize| -> (Vec<u32>, Vec<f32>) {
        let logits = DevBuf::from_slice(&r.logits[row * NE..(row + 1) * NE]).unwrap();
        let sg = DevBuf::from_slice(&r.sg[row..row + 1]).unwrap();
        let mut sel = DevBuf::zeroed((TK + 1) * 4).unwrap();
        let mut rw = DevBuf::zeroed((TK + 1) * 4).unwrap();
        fn_moe::moe_topk_softmax(&logits, Some(&sg), &mut sel, &mut rw, NE, TK, 1, &st).unwrap();
        (dl(&sel, &st), dl(&rw, &st))
    };
    let mut worst_cpu = 0f64;
    for row in [0usize, 1, 63, 64, 777, 2048, 3333, 4095] {
        let (sel, rw) = sel_rw(row);
        let single = Inputs { x: r.x[row * H..(row + 1) * H].to_vec(), logits: vec![], sg: vec![] };
        let cpu = cpu_row(&o, &single, &sel, &rw);
        let err = rel(&got[row * H..(row + 1) * H], &cpu);
        worst_cpu = worst_cpu.max(err);
        assert!(err < 0.03, "row {row} vs the packed EXL3 CPU reference: rel_l2 {err}");
    }
    eprintln!("fmoep vs the CPU reference, 8 rows of 4,096: worst rel_l2 {worst_cpu:.3e}");
}

fn ms_per_call(st: &Stream, iters: usize, mut f: impl FnMut(usize)) -> f64 {
    for i in 0..3 {
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

/// The prefill MoE of one layer at 64 to 4,096 rows with the shipping pack's widths (every matrix 6 bits), balanced and skewed
/// routing: FrostMoE's prefill kernels against CoopMK. A timing lane: run it alone under `flock -x`.
///
///     flock -x /tmp/rexl3-gpu.lock cargo test --release -p gb-cuda --test minicpm_moe_prefill bench_prefill_moe -- --ignored --nocapture
#[test]
#[ignore]
fn bench_prefill_moe_sweep() {
    let Some(st) = gpu() else { return };
    let o = Owned::with_bits(|_, _| 6);
    let weight_bytes: usize = (0..E).map(|_| 3 * H * I * 6 / 8).sum();
    eprintln!(
        "BENCH layer weights {:.0} MB (6 bit), H {H} I {I}, {NE} experts top-{TK} + shared; flop a row {:.1} MFLOP",
        weight_bytes as f64 / 1e6,
        (TK + 1) as f64 * 3.0 * (H * I) as f64 * 2.0 / 1e6
    );
    let mut ws = prefill_ws(CHUNK);
    // REXL3_BENCH_ROWS=4096 (a comma list) narrows the sweep, e.g. for one profiler pass
    let sweep: Vec<usize> = std::env::var("REXL3_BENCH_ROWS")
        .ok()
        .map_or_else(|| vec![64, 256, 512, 1024, 2048, 4096], |v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect());
    let skews: Vec<f32> = std::env::var("REXL3_BENCH_SKEW").ok().map_or_else(|| vec![0.0, 0.8], |v| vec![v.parse().unwrap()]);
    let coopmk_too = std::env::var("REXL3_BENCH_NO_COOPMK").is_err();
    for &skew in &skews {
        for &t in &sweep {
            let r = rows(t, 0x7000 + t as u64, skew);
            let x = DevBuf::from_slice(&r.x).unwrap();
            let logits = DevBuf::from_slice(&r.logits).unwrap();
            let sg = DevBuf::from_slice(&r.sg).unwrap();
            let mut out = DevBuf::zeroed(t * H * 2).unwrap();
            let shape = fmoe::Shape { t, ne: NE, top_k: TK, h: H, i: I, kc: KC, max_tokens: CHUNK };
            let ms = ms_per_call(&st, if t <= 512 { 20 } else { 8 }, |_| {
                fmoe::prefill(&x, &logits, &sg, &o.tables(), &mut out, shape, None, fmoe::SwigluLimit::default(), &mut ws, &st)
                    .unwrap();
            });
            let flop = t as f64 * (TK + 1) as f64 * 3.0 * (H * I) as f64 * 2.0;
            let mut sel = DevBuf::zeroed(t * (TK + 1) * 4).unwrap();
            let mut rw = DevBuf::zeroed(t * (TK + 1) * 4).unwrap();
            let mut act = DevBuf::zeroed(fn_moe::coopmk_workspace_bytes(t, TK + 1, I)).unwrap();
            let mut cout = DevBuf::zeroed(t * H * 2).unwrap();
            let ms_c = if coopmk_too {
                ms_per_call(&st, if t <= 512 { 4 } else { 2 }, |_| {
                    fn_moe::moe_topk_softmax(&logits, Some(&sg), &mut sel, &mut rw, NE, TK, t, &st).unwrap();
                    fn_moe::coopmk_decode(&x, &sel, &rw, &o.tables(), &mut act, &mut cout, H, I, TK + 1, t, &st).unwrap();
                })
            } else {
                f64::NAN
            };
            eprintln!(
                "BENCH skew {skew:.1} rows {t:>4}: fmoep {:8.3} ms ({:6.1} us/row, {:5.1} TFLOP/s)   coopmk {:8.3} ms ({:6.1} us/row)   {:4.1}x",
                ms,
                ms * 1e3 / t as f64,
                flop / ms / 1e9,
                ms_c,
                ms_c * 1e3 / t as f64,
                ms_c / ms
            );
        }
    }
}

/// The short-prompt end: the prefill MoE at 1 to 128 rows (the GEMV path's prompts), FrostMoE's prefill kernels against CoopMK and,
/// up to its 64 rows, FrostMoE's decode window kernel. Weights read from DRAM (a fresh row set a call, as in the decode sweep).
#[test]
#[ignore]
fn bench_prefill_moe_short_prompts() {
    let Some(st) = gpu() else { return };
    let o = Owned::with_bits(|_, _| 6);
    let mut ws = prefill_ws(CHUNK);
    let mut fws = fmoe::workspace(64, TK, H, I, KC).unwrap();
    for t in [1usize, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128] {
        let sets: Vec<Rows> = (0..16).map(|i| rows(t, 0x8000 + (t * 16 + i) as u64, 0.8)).collect();
        let dev: Vec<(DevBuf, DevBuf, DevBuf)> = sets
            .iter()
            .map(|r| {
                (DevBuf::from_slice(&r.x).unwrap(), DevBuf::from_slice(&r.logits).unwrap(), DevBuf::from_slice(&r.sg).unwrap())
            })
            .collect();
        let mut out = DevBuf::zeroed(t * H * 2).unwrap();
        let shape = fmoe::Shape { t, ne: NE, top_k: TK, h: H, i: I, kc: KC, max_tokens: CHUNK };
        let ms = ms_per_call(&st, 64, |i| {
            let (x, l, g) = &dev[i % 16];
            fmoe::prefill(x, l, g, &o.tables(), &mut out, shape, None, fmoe::SwigluLimit::default(), &mut ws, &st).unwrap();
        });
        let frost = if t <= 64 {
            let fshape = fmoe::Shape { t, ne: NE, top_k: TK, h: H, i: I, kc: KC, max_tokens: 64 };
            ms_per_call(&st, 64, |i| {
                let (x, l, g) = &dev[i % 16];
                fmoe::forward(x, l, g, &o.tables(), &mut out, fshape, None, fmoe::SwigluLimit::default(), &mut fws, &st).unwrap();
            })
        } else {
            f64::NAN
        };
        let mut sel = DevBuf::zeroed(t * (TK + 1) * 4).unwrap();
        let mut rw = DevBuf::zeroed(t * (TK + 1) * 4).unwrap();
        let mut act = DevBuf::zeroed(fn_moe::coopmk_workspace_bytes(t, TK + 1, I)).unwrap();
        let ms_c = ms_per_call(&st, 16, |i| {
            let (x, l, g) = &dev[i % 16];
            fn_moe::moe_topk_softmax(l, Some(g), &mut sel, &mut rw, NE, TK, t, &st).unwrap();
            fn_moe::coopmk_decode(x, &sel, &rw, &o.tables(), &mut act, &mut out, H, I, TK + 1, t, &st).unwrap();
        });
        eprintln!(
            "BENCH rows {t:>3}: fmoep {:7.1} us   coopmk {:7.1} us   frost decode {:7.1} us",
            ms * 1e3,
            ms_c * 1e3,
            frost * 1e3
        );
    }
}

/// The bits a chunk of 1,000 rows leaves (committed: `testdata/gpu/golden_minicpm_moe_prefill.txt`). It pins the prefill kernels
/// themselves at chunk scale; the 64-row hash of `golden_minicpm_moe.txt` stays as it was.
#[test]
#[ignore]
fn golden_prefill_hashes() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let mut ws = prefill_ws(CHUNK);
    let r = rows(1000, 0x6005, 0.8);
    let whole = call(&o, &r, &mut ws, CHUNK, &st);
    let cut = by_grid(&o, &r, 333, &mut ws, &st);
    assert_eq!(whole, cut, "a cut grid differs: the hash below would not be one number");
    let lines = [format!("prefill1000 {:016x}", fnv(&whole)), format!("prefill1000_first64 {:016x}", fnv(&whole[..64 * H]))];
    std::fs::write(std::env::var("REXL3_GOLDEN_OUT").unwrap(), format!("{}\n", lines.join("\n"))).unwrap();
}
