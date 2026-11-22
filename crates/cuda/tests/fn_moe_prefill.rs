//! The routed experts of a Qwen3.8-Flash-Next prefill chunk at the family's own layer shape (hidden 2560, expert width 640, 512
//! experts, top-10, 3-bit Mul1). The ring-fed expert GEMMs (`csrc/fn_moe_prefill.cu`, `src/fn_moe_prefill.rs`) equal the grouped
//! path's per-item launches (`MoeExperts::run_prefill_f32_wts`) bit for bit at every row count, routing style and width they
//! take, on synthetic layers (no pack needed); plus the speed table of both, stage by stage, against the roofs.
//!
//!     cargo test --release -p gb-cuda --test fn_moe_prefill -- --test-threads=1
//!     cargo test --release -p gb-cuda --test fn_moe_prefill -- --ignored --nocapture bench_
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::exl3_synth::synth_group;
use common::*;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::moe::*;
use gb_cuda::moe_prefill::PrefillScratch;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::LoadedGroup;
use gb_exl3::Codebook;

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: not run");
            return;
        }
    };
}

fn up<T: Copy>(v: &[T]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}

fn unit(r: &mut Rng) -> f64 {
    (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

fn normal(r: &mut Rng) -> f64 {
    let (a, b) = (unit(r), unit(r));
    (-2.0 * a.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
}

fn below(r: &mut Rng, n: usize) -> usize {
    (r.next_u64() % n as u64) as usize
}

fn flash_next_dims(experts: usize) -> MoeDims {
    MoeDims { hidden: 2560, inter: 640, experts, topk: 10, scale: 1.0, limit: 0.0 }
}

fn synth_layer(dims: MoeDims, bits: usize, seed: u64) -> MoeExperts {
    let mk = |k, n, tag: u64| -> Vec<LoadedGroup> {
        (0..dims.experts).map(|x| synth_group(k, n, bits, Codebook::Mul1, seed * 1000 + tag * 100_000 + x as u64)).collect()
    };
    let gate = mk(dims.hidden, dims.inter, 1);
    let up = mk(dims.hidden, dims.inter, 2);
    let down = mk(dims.inter, dims.hidden, 3);
    MoeExperts::from_groups(&gate, &up, &down, dims).unwrap()
}

#[derive(Clone, Copy, Debug)]
enum Style {
    /// `topk` distinct uniformly random experts a row
    Uniform,
    /// a few hot experts take many rows, the tail is rare
    Skewed,
    /// a quarter of the picks are outside [0, E)
    Holes,
    /// every row picks the same experts
    Hot,
}

fn make_picks(r: &mut Rng, style: Style, rows: usize, e: usize, topk: usize) -> Vec<i32> {
    let mut pick = Vec::with_capacity(rows * topk);
    for _ in 0..rows {
        let mut chosen: Vec<usize> = Vec::new();
        while chosen.len() < topk {
            let c = match style {
                Style::Uniform | Style::Holes => below(r, e),
                Style::Skewed => ((1.0 / (unit(r) + 1e-6)) as usize - 1).min(e - 1),
                Style::Hot => chosen.len(),
            };
            if !chosen.contains(&c) {
                chosen.push(c);
            }
        }
        for id in chosen {
            let hole = matches!(style, Style::Holes) && below(r, 4) == 0;
            pick.push(if hole { -1 + (below(r, 2) as i32) * (e as i32 + 1) } else { id as i32 });
        }
    }
    pick
}

fn rand_wts(r: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| (0.02 + 0.3 * unit(r)) as f32).collect()
}

fn rand_bf16(r: &mut Rng, n: usize, sigma: f64) -> Vec<u16> {
    (0..n).map(|_| gb_cuda::dsv4_attn::bf16_bits((sigma * normal(r)) as f32)).collect()
}

const STAGES: [&str; 5] = ["plan", "rot_in", "gate_up", "down", "combine"];

/// Per-stage milliseconds of one grouped call (median of `iters` after a warm-up, the L2 flushed before each).
fn time_stages(
    ex: &MoeExperts,
    st: &Stream,
    s: &mut PrefillScratch,
    x: &DevBuf,
    shared: &DevBuf,
    out: &mut DevBuf,
    rows: usize,
    iters: usize,
    ring: bool,
) -> ([f64; 5], f64) {
    let flush = DevBuf::zeroed(256 << 20).unwrap();
    let mut flush_out = DevBuf::zeroed(16).unwrap();
    let t = EventTimer::new(8).unwrap();
    let mut per: Vec<[f64; 6]> = Vec::new();
    for it in 0..iters + 1 {
        gb_cuda::moe::read_probe(&flush, 256 << 20, &mut flush_out, st).unwrap();
        t.record(0, st).unwrap();
        let mut n = 0;
        let mut mark = |_: &'static str| {
            n += 1;
            t.record(n, st)
        };
        if ring {
            ex.run_prefill_ring_f32_wts_marked(x, Dtype::BF16, rows, Some(shared), out, s, st, &mut mark).unwrap();
        } else {
            ex.run_prefill_f32_wts_marked(x, Dtype::BF16, rows, Some(shared), out, s, st, &mut mark).unwrap();
        }
        st.sync().unwrap();
        if it > 0 {
            let mut v = [0f64; 6];
            for i in 0..5 {
                v[i] = t.ms(i, i + 1).unwrap() as f64;
            }
            v[5] = t.ms(0, 5).unwrap() as f64;
            per.push(v);
        }
    }
    let med = |i: usize| {
        let mut v: Vec<f64> = per.iter().map(|p| p[i]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    ([med(0), med(1), med(2), med(3), med(4)], med(5))
}

// ---------------------------------------------------------------------------------------------------------------------------------
// bit identity

struct Run {
    out: Vec<f32>,
    y: Vec<f32>,
}

struct Case {
    x: DevBuf,
    shared: DevBuf,
    pick: Vec<i32>,
    wts: Vec<f32>,
    rows: usize,
}

fn case(r: &mut Rng, style: Style, rows: usize, d: &MoeDims) -> Case {
    Case {
        x: up(&rand_bf16(r, rows * d.hidden, 1.0)),
        shared: up(&(0..rows * d.hidden).map(|_| normal(r) as f32).collect::<Vec<f32>>()),
        pick: make_picks(r, style, rows, d.experts, d.topk),
        wts: rand_wts(r, rows * d.topk),
        rows,
    }
}

/// One call of either path on a scratch poisoned first (nothing in it may be read before it is written).
fn run(ex: &MoeExperts, st: &Stream, s: &mut PrefillScratch, c: &Case, ring: bool, poison: u8) -> Run {
    let d = ex.dims;
    s.poison(poison, st).unwrap();
    s.set_picks_f32(&c.pick, &c.wts).unwrap();
    let mut out = DevBuf::zeroed(c.rows * d.hidden * 4).unwrap();
    gb_cuda::rt::device_sync().unwrap();
    if ring {
        ex.run_prefill_ring_f32_wts(&c.x, Dtype::BF16, c.rows, Some(&c.shared), &mut out, s, st).unwrap();
    } else {
        ex.run_prefill_f32_wts(&c.x, Dtype::BF16, c.rows, Some(&c.shared), &mut out, s, st).unwrap();
    }
    st.sync().unwrap();
    let y: Vec<f32> = s.y_pairs.to_vec().unwrap();
    Run { out: out.to_vec().unwrap(), y: y[..c.rows * d.topk * d.hidden].to_vec() }
}

/// `got` equals `want` bit for bit: every element of `out`, and `y` of every pair that was computed (a pick outside [0, E) writes
/// nothing: its `y` is whatever the scratch held).
fn assert_same(what: &str, got: &Run, want: &Run, d: &MoeDims, pick: &[i32]) {
    let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).position(|(x, y)| x.to_bits() != y.to_bits());
    assert_eq!(got.out.len(), want.out.len(), "{what}: out length");
    for (p, &e) in pick.iter().enumerate() {
        if e < 0 || e as usize >= d.experts {
            continue;
        }
        let r = p * d.hidden..(p + 1) * d.hidden;
        if let Some(i) = diff(&got.y[r.clone()], &want.y[r.clone()]) {
            panic!("{what}: pair {p} (row {}, slot {}, expert {e}) column {i} of y differs", p / d.topk, p % d.topk);
        }
    }
    if let Some(i) = diff(&got.out, &want.out) {
        panic!("{what}: row {} column {} of out differs: {} vs {}", i / d.hidden, i % d.hidden, got.out[i], want.out[i]);
    }
}

const STYLES: [Style; 4] = [Style::Uniform, Style::Skewed, Style::Holes, Style::Hot];

/// The headline: on the Flash-Next layer shape the ring-fed GEMMs give the per-item launches' `out` and `y` bit for bit, at
/// row counts on both sides of every tile, group and chunk size and for every routing style (hot experts take many groups a
/// block; holes and unpicked experts take none).
#[test]
fn ring_fed_gemms_equal_per_item_launches_bit_for_bit_on_the_flash_next_shape() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = flash_next_dims(512);
    let ex = synth_layer(d, 3, 11);
    let mut s = PrefillScratch::new(&d, 4096).unwrap();
    s.use_parallel_plan(true);
    assert!(ex.ring_prefill(&s));
    let mut r = Rng(21);
    let mut compared = 0usize;
    for rows in [1usize, 17, 129, 300, 1000, 2478, 4096] {
        for style in STYLES {
            let c = case(&mut r, style, rows, &d);
            let want = run(&ex, &st, &mut s, &c, false, 0x5a);
            let got = run(&ex, &st, &mut s, &c, true, 0xa5);
            assert_same(&format!("{style:?} rows {rows}"), &got, &want, &d, &c.pick);
            assert!(
                got.out.iter().any(|v| *v != 0.0) && got.out.iter().all(|v| v.is_finite()),
                "{style:?} rows {rows}: degenerate"
            );
            compared += got.out.len() + got.y.len();
        }
    }
    println!("FNPF-BITS flash-next shape: {compared} fp32 values compared, 0 differ");
}

/// The golden (`scripts/gpu_gate.sh`): hashes of the ring-fed path's `out` and computed `y` on synthetic Flash-Next layers (3 bits
/// at the pack's shape; 2 and 4 bits at a smaller one), at row counts on both sides of a tile, a group and a chunk, for every
/// routing style. They equal the per-item launches' by the tests above, and are committed so that a change to either kernel family
/// that moves a bit shows up in the gate even if both moved together.
#[test]
#[ignore]
fn golden_fn_moe_prefill_hashes() {
    gpu!();
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    let hash = |v: &[f32]| gb_exl3::fnv1a(v.iter().flat_map(|x| x.to_bits().to_le_bytes()));
    let st = Stream::new().unwrap();
    let mut lines = Vec::new();
    for (experts, bits, seed) in [(512usize, 3usize, 700u64), (64, 2, 710), (64, 4, 720)] {
        let d = flash_next_dims(experts);
        let ex = synth_layer(d, bits, seed);
        let mut s = PrefillScratch::new(&d, 4096).unwrap();
        s.use_parallel_plan(true);
        assert!(ex.ring_prefill(&s));
        let mut r = Rng(seed + 1);
        for rows in [17usize, 300, 2478] {
            for style in STYLES {
                let c = case(&mut r, style, rows, &d);
                let got = run(&ex, &st, &mut s, &c, true, 0xa5);
                // the y of a pair that was never computed is whatever the scratch held: hash the computed ones
                let mut y = got.y.clone();
                for (p, &e) in c.pick.iter().enumerate() {
                    if e < 0 || e as usize >= d.experts {
                        y[p * d.hidden..(p + 1) * d.hidden].fill(0.0);
                    }
                }
                lines.push(format!(
                    "ring prefill experts e={experts} bits={bits} rows={rows} {style:?} out {:016x} y {:016x}",
                    hash(&got.out),
                    hash(&y)
                ));
            }
        }
    }
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    eprintln!("wrote {} golden lines to {path}", lines.len());
}

/// Every width the ring-fed kernels are built for (2, 3 and 4 bits) on two smaller shapes whose chains end at a ring stage's end and
/// inside one, a tight SwiGLU clamp included.
#[test]
fn ring_fed_gemms_equal_per_item_launches_at_every_width_and_chain_length() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut r = Rng(22);
    // (hidden, inter): chains of 8 and 4 k tiles, and of 10 and 6 (a chain that ends inside a ring stage)
    for (hidden, inter, experts, topk) in [(512usize, 256usize, 48usize, 6usize), (640, 384, 40, 5)] {
        for (bits, limit) in [(2usize, 0.0f32), (3, 0.75), (4, 0.0)] {
            let d = MoeDims { hidden, inter, experts, topk, scale: 1.0, limit };
            let ex = synth_layer(d, bits, 30 + bits as u64);
            assert_eq!((ex.plan_gu, ex.plan_d), ((1, 4), (1, 4)));
            let mut s = PrefillScratch::new(&d, 1100).unwrap();
            s.use_parallel_plan(true);
            assert!(ex.ring_prefill(&s));
            for rows in [1usize, 16, 33, 257, 1100] {
                for style in STYLES {
                    let c = case(&mut r, style, rows, &d);
                    let want = run(&ex, &st, &mut s, &c, false, 0x11);
                    let got = run(&ex, &st, &mut s, &c, true, 0xee);
                    assert_same(&format!("{hidden}x{inter} {bits}-bit {style:?} rows {rows}"), &got, &want, &d, &c.pick);
                }
            }
        }
    }
}

/// A bank or scratch the ring-fed GEMMs do not take is refused, never launched: the single-block plan (no per-expert offsets in
/// the scratch), a width outside 2 to 4 bits, a decode plan that splits K.
#[test]
fn unsupported_banks_are_refused() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = MoeDims { hidden: 512, inter: 256, experts: 16, topk: 4, scale: 1.0, limit: 0.0 };
    let mut r = Rng(23);
    let c = case(&mut r, Style::Uniform, 40, &d);
    let mut out = DevBuf::zeroed(40 * d.hidden * 4).unwrap();
    let ex = synth_layer(d, 3, 41);
    let mut s = PrefillScratch::new(&d, 64).unwrap();
    s.set_picks_f32(&c.pick, &c.wts).unwrap();
    assert!(!ex.ring_prefill(&s), "the single-block plan leaves no per-expert offsets");
    assert!(ex.run_prefill_ring_f32_wts(&c.x, Dtype::BF16, 40, None, &mut out, &mut s, &st).is_err());
    s.use_parallel_plan(true);
    assert!(ex.ring_prefill(&s));
    let wide = synth_layer(d, 5, 42);
    assert!(!wide.ring_prefill(&s), "5 bits keeps the per-item launches");
    assert!(wide.run_prefill_ring_f32_wts(&c.x, Dtype::BF16, 40, None, &mut out, &mut s, &st).is_err());
    // a shape whose decode plan splits K (two splits of four warps): another sum order than these kernels reproduce
    let d2 = MoeDims { hidden: 1024, inter: 256, experts: 16, topk: 4, scale: 1.0, limit: 0.0 };
    let split = synth_layer(d2, 3, 43);
    assert_eq!(split.plan_gu.0, 2);
    let mut s2 = PrefillScratch::new(&d2, 64).unwrap();
    s2.use_parallel_plan(true);
    assert!(!split.ring_prefill(&s2), "a split K keeps the per-item launches");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// speed

/// Both launch schemes' stages on a Flash-Next-shaped layer (512 synthetic 3-bit experts), with each stage's bytes and FLOPs so that
/// its fraction of the bandwidth and compute roofs can be read off (GB10: 235 GB/s measured streaming, 124 TFLOP/s fp16 MMA peak).
#[test]
#[ignore]
fn bench_flash_next_prefill_experts() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = flash_next_dims(512);
    let bits = 3usize;
    let ex = synth_layer(d, bits, 7);
    let mut r = Rng(5);
    let (k, n, e, topk) = (d.hidden as f64, d.inter as f64, d.experts as f64, d.topk as f64);
    println!("Flash-Next-shaped layer, {bits}-bit, 512 experts, expert bytes {:.3} GB", ex.expert_bytes as f64 * e / 1e9);
    for style in [Style::Uniform, Style::Skewed, Style::Hot] {
        for rows in [1024usize, 2478, 4096] {
            let mut s = PrefillScratch::new(&d, rows).unwrap();
            s.use_parallel_plan(true);
            let x = up(&rand_bf16(&mut r, rows * d.hidden, 1.0));
            let shared = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
            let mut out = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
            let pick = make_picks(&mut r, style, rows, d.experts, d.topk);
            s.set_picks_f32(&pick, &rand_wts(&mut r, rows * d.topk)).unwrap();
            gb_cuda::rt::device_sync().unwrap();
            let (stage, total) = time_stages(&ex, &st, &mut s, &x, &shared, &mut out, rows, 5, false);
            let (stage_e, total_e) = time_stages(&ex, &st, &mut s, &x, &shared, &mut out, rows, 5, true);
            let ic = s.item_counts().unwrap();
            let items = (ic[0] + ic[1] + ic[2]) as f64;
            let pairs = rows as f64 * topk;
            let wbits = bits as f64 / 8.0;
            // bytes each stage must move (arithmetic): rot_in reads x once and writes two fp16 rows a pair; gate/up reads both matrices
            // once an item and the two rows a pair, writes the down input; down reads its matrix once an item and that input, writes
            // fp32 y; the combine reads y and writes a row.
            let gb = [
                0.0,
                (rows as f64 * k * 2.0 + pairs * k * 4.0) / 1e9,
                (items * 2.0 * k * n * wbits + pairs * k * 4.0 + pairs * n * 2.0) / 1e9,
                (items * k * n * wbits + pairs * n * 2.0 + pairs * k * 4.0) / 1e9,
                (pairs * k * 4.0 + rows as f64 * k * 8.0) / 1e9,
            ];
            let tflop = [0.0, 0.0, pairs * 2.0 * 2.0 * k * n / 1e12, pairs * 2.0 * k * n / 1e12, 0.0];
            println!(
                "FNPF {style:?} R={rows:5} items 1/2/3-4 tiles {}/{}/{} (decodes an expert {:.2}) total {total:6.2} ms",
                ic[0],
                ic[1],
                ic[2],
                items / e
            );
            for i in 0..5 {
                println!(
                    "FNPF   {:8} {:6.2} ms  {:5.2} GB = {:5.2} ms at 235 GB/s ({:3.0}%)  {:5.2} TFLOP = {:5.2} ms at 124 TFLOP/s ({:3.0}%)",
                    STAGES[i],
                    stage[i],
                    gb[i],
                    gb[i] / 235.0 * 1e3,
                    gb[i] / 235.0 * 1e3 / stage[i] * 100.0,
                    tflop[i],
                    tflop[i] / 124.0 * 1e3,
                    tflop[i] / 124.0 * 1e3 / stage[i] * 100.0
                );
            }
            // the ring-fed GEMMs read each picked expert's weights from memory once (a slice after the first comes from the cache, ideally)
            let once = pick.iter().filter(|&&p| p >= 0).collect::<std::collections::BTreeSet<_>>().len() as f64;
            let gb_e = [
                gb[0],
                gb[1],
                gb[2] - (items - once) * 2.0 * k * n * wbits / 1e9,
                gb[3] - (items - once) * k * n * wbits / 1e9,
                gb[4],
            ];
            println!("FNPF  ring-fed: total {total_e:6.2} ms ({:.2}x)", total / total_e);
            for i in 0..5 {
                println!(
                    "FNPF   {:8} {:6.2} ms  {:5.2} GB = {:5.2} ms at 235 GB/s ({:3.0}%)",
                    STAGES[i],
                    stage_e[i],
                    gb_e[i],
                    gb_e[i] / 235.0 * 1e3,
                    gb_e[i] / 235.0 * 1e3 / stage_e[i] * 100.0
                );
            }
        }
    }
}

/// For comparison only: GLM's P2 expert GEMMs on the Flash-Next shape. Their K order (one chain, or 128-k folds) is not the
/// decode path's, so they would move every bit; this measures what the same schedule costs without the four-chain order.
#[test]
#[ignore]
fn bench_p2_gemms_on_the_flash_next_shape() {
    gpu!();
    use gb_cuda::glm_p2_moe::{pairs_with, Order};
    let st = Stream::new().unwrap();
    let d = flash_next_dims(512);
    let ex = synth_layer(d, 3, 7);
    let mut r = Rng(5);
    let flush = DevBuf::zeroed(256 << 20).unwrap();
    let mut flush_out = DevBuf::zeroed(16).unwrap();
    for style in [Style::Uniform, Style::Skewed, Style::Hot] {
        for rows in [1024usize, 2478, 4096] {
            let mut s = PrefillScratch::new(&d, rows).unwrap();
            let x = up(&rand_bf16(&mut r, rows * d.hidden, 1.0));
            let pick = make_picks(&mut r, style, rows, d.experts, d.topk);
            s.set_picks_f32(&pick, &rand_wts(&mut r, rows * d.topk)).unwrap();
            gb_cuda::rt::device_sync().unwrap();
            for order in [Order::Chain, Order::Fold128] {
                let t = EventTimer::new(8).unwrap();
                let mut best = [f64::MAX; 4];
                for _ in 0..6 {
                    gb_cuda::moe::read_probe(&flush, 256 << 20, &mut flush_out, &st).unwrap();
                    t.record(0, &st).unwrap();
                    let mut n = 0;
                    pairs_with(&ex, &x, Dtype::BF16, rows, &mut s, order, &st, &mut |_| {
                        n += 1;
                        t.record(n, &st)
                    })
                    .unwrap();
                    st.sync().unwrap();
                    assert_eq!(n, 4);
                    for i in 0..4 {
                        best[i] = best[i].min(t.ms(i, i + 1).unwrap() as f64);
                    }
                }
                println!(
                    "FNP2 {style:?} R={rows} {order:?}: plan {:.2} rot {:.2} gate_up {:.2} down {:.2}",
                    best[0], best[1], best[2], best[3]
                );
            }
        }
    }
}
