//! Performance and bit-identity gates of the Nemotron 3.5 Super VL latent-MoE block (`csrc/nemotron_moe.cu`).
//!
//! The kernel's per-row arithmetic is frozen (the large-row prefill kernel is gated on it), so every change to the launches is gated
//! here by bits: the output of the block, the router's logits, picks and weights, at 1, 7 and 64 rows, on synthetic layers (14 of 512
//! experts present, and all 512 present with two bf16 experts), digested with FNV-1a and compared with the digests the pre-change kernels
//! produced (`GOLDEN`). With the pack (`REXL3_NEMO_OMNI_DIR`) the real layer 1 (all 512 experts) is also run through the reference
//! launches and the current ones and the bits are compared in process; that test only reads the pack.
//!
//! Timing (ignored, `bench` in the name so the gate skips them): a per-stage table at 1, 8 and 64 rows with the weight bytes each stage
//! reads, from the picks of each call, and the block's floor at the achievable bandwidth. Another layer runs before every timed call (so L2 holds clean lines of other
//! weights), as in a decode step.
//!
//!     cargo test -p gb-cuda --release --test nemotron_moe_perf -- --test-threads=1 --nocapture
//!     cargo test -p gb-cuda --release --test nemotron_moe_perf -- --ignored --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::exl3_synth::synth_group;
use common::Rng;
use core::ffi::c_void;
use gb_cuda::nemotron_moe::{
    MoeScratch, NemotronMoe, Proj, Stage, Weights, EXPERTS, EXPERT_INTER, HIDDEN, LATENT, SHARED_INTER, TOPK,
};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::{scan, Checkpoint, LoadedGroup};
use gb_exl3::{fnv1a, Codebook};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

const SCALE: f32 = 5.0;
/// The achievable DRAM bandwidth of the GB10 the floors are taken at (about 273 GB/s peak).
const FLOOR_GBPS: f64 = 240.0;

mod ev {
    use core::ffi::c_void;
    extern "C" {
        pub fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> i32;
        pub fn cudaEventSynchronize(e: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
        pub fn cudaEventDestroy(e: *mut c_void) -> i32;
    }
}

/// A timing event.
struct Ev(*mut c_void);

impl Ev {
    fn new() -> Self {
        let mut e = core::ptr::null_mut();
        assert_eq!(unsafe { ev::cudaEventCreateWithFlags(&mut e, 0) }, 0);
        Ev(e)
    }
    fn record(&self, st: &Stream) {
        assert_eq!(unsafe { ev::cudaEventRecord(self.0, st.raw()) }, 0);
    }
    /// Microseconds from `self` to `later` (both recorded, `later` reached).
    fn us_to(&self, later: &Ev) -> f64 {
        let mut ms = 0f32;
        assert_eq!(unsafe { ev::cudaEventSynchronize(later.0) }, 0);
        assert_eq!(unsafe { ev::cudaEventElapsedTime(&mut ms, self.0, later.0) }, 0);
        ms as f64 * 1000.0
    }
}

impl Drop for Ev {
    fn drop(&mut self) {
        unsafe {
            ev::cudaEventDestroy(self.0);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Layers

/// The bytes the kernels read for one projection: the trellis (or the bf16 weight), the f32 scales and the bias.
fn proj_bytes(p: &Proj) -> u64 {
    match p {
        Proj::Exl3 { k, n, trellis, bias, .. } => {
            trellis.bytes() as u64 + 4 * (*k as u64 + *n as u64) + bias.as_ref().map_or(0, |_| 4 * *n as u64)
        }
        Proj::Bf16 { w, .. } | Proj::F16 { w, .. } => w.bytes() as u64,
        Proj::Absent => 0,
    }
}

/// A block and the bytes of each of its projections (for the achieved bandwidth).
struct Layer {
    moe: NemotronMoe,
    gate: u64,
    fc1: u64,
    fc2: u64,
    shared_up: u64,
    shared_down: u64,
    up: Vec<u64>,
    down: Vec<u64>,
}

impl Layer {
    fn new(w: Weights) -> Self {
        let gate = w.gate.bytes() as u64;
        let (fc1, fc2, shared_up, shared_down) =
            (proj_bytes(&w.fc1), proj_bytes(&w.fc2), proj_bytes(&w.shared_up), proj_bytes(&w.shared_down));
        let up: Vec<u64> = w.up.iter().map(proj_bytes).collect();
        let down: Vec<u64> = w.down.iter().map(proj_bytes).collect();
        Layer { moe: NemotronMoe::tables(w).unwrap(), gate, fc1, fc2, shared_up, shared_down, up, down }
    }

    /// Bytes each stage reads for a call whose picks are `pick` (`[rows, 22]`): the weights of the distinct experts, once each.
    fn stage_bytes(&self, pick: &[i32], rows: usize) -> BTreeMap<&'static str, u64> {
        let distinct: BTreeSet<usize> = pick.iter().map(|&e| e as usize).collect();
        let mut m = BTreeMap::new();
        m.insert("router", self.gate);
        m.insert("bucket", 0);
        m.insert("latent in", self.fc1);
        m.insert("up", distinct.iter().map(|&e| self.up[e]).sum());
        m.insert("down", distinct.iter().map(|&e| self.down[e]).sum());
        m.insert("combine", (rows * TOPK * LATENT * 4) as u64);
        m.insert("fc2", self.fc2);
        m.insert("shared", self.shared_up + self.shared_down);
        m
    }
}

fn bf16_bits(r: &mut Rng, n: usize) -> Vec<u16> {
    (0..n).map(|_| (r.uniform(-0.05, 0.05).to_bits() >> 16) as u16).collect()
}

/// A synthetic layer: a random gate and bias, EXL3 dense projections, and the experts in `present` (those also in `bf16` are bf16). Every
/// expert has its own random trellis, so no two share bytes.
fn synthetic_layer(seed: u64, present: &[usize], bf16: &[usize]) -> Weights {
    let mut r = Rng(seed);
    let gate_bits = bf16_bits(&mut r, EXPERTS * HIDDEN);
    let bias: Vec<f32> = r.vec(EXPERTS, -0.1, 0.1);
    let g = |k, n, s| Proj::exl3_from_group(&synth_group(k, n, 4, Codebook::Mul1, s)).unwrap();
    let mut up: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    let mut down: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    for &e in present {
        if bf16.contains(&e) {
            up[e] = Proj::bf16(LATENT, EXPERT_INTER, &bf16_bits(&mut r, EXPERT_INTER * LATENT)).unwrap();
            down[e] = Proj::bf16(EXPERT_INTER, LATENT, &bf16_bits(&mut r, LATENT * EXPERT_INTER)).unwrap();
        } else {
            up[e] = g(LATENT, EXPERT_INTER, seed + 1000 + 2 * e as u64);
            down[e] = g(EXPERT_INTER, LATENT, seed + 1001 + 2 * e as u64);
        }
    }
    Weights {
        gate: DevBuf::from_slice(&gate_bits).unwrap(),
        gate_dtype: gb_cuda::nemotron_moe::GateDtype::Bf16,
        router_bias: DevBuf::from_slice(&bias).unwrap(),
        scale: SCALE,
        fc1: g(HIDDEN, LATENT, seed + 1),
        fc2: g(LATENT, HIDDEN, seed + 2),
        shared_up: g(HIDDEN, SHARED_INTER, seed + 3),
        shared_down: g(SHARED_INTER, HIDDEN, seed + 4),
        up,
        down,
    }
}

/// A MixedK synthetic layer (v0.1.2 lane K): the present experts' up and down at their own Mul1 widths (`up_bits(e)`, `down_bits(e)`
/// bits, 2 to 8), the experts in `bf16` bf16, and the dense projections fc1, fc2, shared up, shared down at `dense[i]` bits (0: an fp16
/// weight `[N, K]`, a MixedK pack's latent projections). Each expert has its own random trellis.
fn mixed_layer(
    seed: u64,
    present: &[usize],
    bf16: &[usize],
    up_bits: impl Fn(usize) -> usize,
    down_bits: impl Fn(usize) -> usize,
    dense: [usize; 4],
) -> Weights {
    let mut r = Rng(seed);
    let gate_bits = bf16_bits(&mut r, EXPERTS * HIDDEN);
    let bias: Vec<f32> = r.vec(EXPERTS, -0.1, 0.1);
    let g = |k, n, b, s| Proj::exl3_from_group(&synth_group(k, n, b, Codebook::Mul1, s)).unwrap();
    let mut up: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    let mut down: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    for &e in present {
        if bf16.contains(&e) {
            up[e] = Proj::bf16(LATENT, EXPERT_INTER, &bf16_bits(&mut r, EXPERT_INTER * LATENT)).unwrap();
            down[e] = Proj::bf16(EXPERT_INTER, LATENT, &bf16_bits(&mut r, LATENT * EXPERT_INTER)).unwrap();
        } else {
            up[e] = g(LATENT, EXPERT_INTER, up_bits(e), seed + 1000 + 2 * e as u64);
            down[e] = g(EXPERT_INTER, LATENT, down_bits(e), seed + 1001 + 2 * e as u64);
        }
    }
    let shapes = [(HIDDEN, LATENT), (LATENT, HIDDEN), (HIDDEN, SHARED_INTER), (SHARED_INTER, HIDDEN)];
    let mut d: Vec<Proj> = Vec::new();
    for (i, &(k, n)) in shapes.iter().enumerate() {
        d.push(if dense[i] == 0 {
            let bits: Vec<u16> = (0..k * n).map(|_| gb_exl3::f16::f64_to_f16(r.uniform(-0.05, 0.05) as f64)).collect();
            Proj::f16(k, n, &bits).unwrap()
        } else {
            g(k, n, dense[i], seed + 1 + i as u64)
        });
    }
    let mut d = d.into_iter();
    Weights {
        gate: DevBuf::from_slice(&gate_bits).unwrap(),
        gate_dtype: gb_cuda::nemotron_moe::GateDtype::Bf16,
        router_bias: DevBuf::from_slice(&bias).unwrap(),
        scale: SCALE,
        fc1: d.next().unwrap(),
        fc2: d.next().unwrap(),
        shared_up: d.next().unwrap(),
        shared_down: d.next().unwrap(),
        up,
        down,
    }
}

/// A width of {2, 3, 4, 5, 6} bits for expert `e` and stream `salt`, spread so that every (up, down) pair of widths occurs.
fn width_of(e: usize, salt: u64) -> usize {
    let h = (e as u64 ^ salt).wrapping_mul(0x9E3779B97F4A7C15) >> 59;
    [2, 3, 4, 5, 6][(h % 5) as usize]
}

/// The MixedK synthetic cases of the bit gate: his pack's shapes (all 512 experts at mixed widths, up and down apart; fc1 and fc2
/// fp16; the shared expert at 5 and 4 bits), the wide widths (7 and 8 bits, dense at 6 to 8 bits, two bf16 experts among them) and
/// a TP rank's half (absent experts) at mixed widths with 4-bit dense projections.
fn mixed_cases() -> Vec<(&'static str, Weights)> {
    let all: Vec<usize> = (0..EXPERTS).collect();
    let sparse: Vec<usize> = (0..EXPERTS).step_by(5).collect();
    let even: Vec<usize> = (0..EXPERTS).step_by(2).collect();
    vec![
        ("mixedk512 (his shapes)", mixed_layer(201, &all, &[], |e| width_of(e, 1), |e| width_of(e, 2), [0, 0, 5, 4])),
        (
            "wide (7, 8 bits; dense 8/6/7/6; bf16 among them)",
            mixed_layer(202, &sparse, &[sparse[3], sparse[40]], |e| [7, 8, 4][e % 3], |e| [8, 7, 5][e % 3], [8, 6, 7, 6]),
        ),
        ("tp half (absent) mixed, dense 4", mixed_layer(203, &even, &[], |e| width_of(e, 3), |e| width_of(e, 4), [4, 4, 4, 4])),
    ]
}

/// The layer-1-like synthetic cases of the bit gate: (name, seed, present experts, bf16 experts).
fn synthetic_cases() -> Vec<(&'static str, u64, Vec<usize>, Vec<usize>)> {
    let sparse: Vec<usize> = (0..EXPERTS).step_by(37).collect();
    let bf = sparse[3];
    vec![("sparse14", 61, sparse, vec![bf]), ("all512", 101, (0..EXPERTS).collect(), vec![5, 300])]
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The pack's layer

fn pack_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set; the pack-backed checks need the weights");
    }
    dir
}

fn plain_u16(ck: &Checkpoint, name: &str) -> Vec<u16> {
    let e = ck.plain.get(name).unwrap_or_else(|| panic!("the pack has no plain tensor {name}"));
    assert_eq!(e.dtype, "BF16", "{name} is {}, not BF16", e.dtype);
    ck.read_u16(e).unwrap()
}

fn plain_f32(ck: &Checkpoint, name: &str) -> Vec<f32> {
    let e = ck.plain.get(name).unwrap_or_else(|| panic!("the pack has no plain tensor {name}"));
    ck.read_floats(e).unwrap().into_iter().map(|v| v as f32).collect()
}

/// A pack layer's EXL3 projection, with the (bits, codebook) it was stored with added to `seen`.
fn pack_group(ck: &Checkpoint, prefix: &str, seen: &mut BTreeMap<(usize, usize, u8, String), usize>) -> Proj {
    let g: LoadedGroup = ck.load_group(prefix).unwrap_or_else(|e| panic!("{prefix}: {e}"));
    *seen.entry((g.group.k, g.group.n, g.group.bits.twice(), format!("{:?}", g.group.codebook))).or_insert(0) += 1;
    Proj::exl3_from_group(&g).unwrap()
}

/// Text layer `layer`'s block from the pack: the gate, the bias, the four dense EXL3 projections and every routed expert (EXL3 only).
fn pack_layer(ck: &Checkpoint, layer: usize) -> Weights {
    let mp = format!("language_model.backbone.layers.{layer}.mixer");
    let mut seen = BTreeMap::new();
    let up: Vec<Proj> = (0..EXPERTS).map(|e| pack_group(ck, &format!("{mp}.experts.{e}.up_proj"), &mut seen)).collect();
    let down: Vec<Proj> = (0..EXPERTS).map(|e| pack_group(ck, &format!("{mp}.experts.{e}.down_proj"), &mut seen)).collect();
    let w = Weights {
        gate: DevBuf::from_slice(&plain_u16(ck, &format!("{mp}.gate.weight"))).unwrap(),
        gate_dtype: gb_cuda::nemotron_moe::GateDtype::Bf16,
        router_bias: DevBuf::from_slice(&plain_f32(ck, &format!("{mp}.gate.e_score_correction_bias"))).unwrap(),
        scale: SCALE,
        fc1: pack_group(ck, &format!("{mp}.fc1_latent_proj"), &mut seen),
        fc2: pack_group(ck, &format!("{mp}.fc2_latent_proj"), &mut seen),
        shared_up: pack_group(ck, &format!("{mp}.shared_experts.up_proj"), &mut seen),
        shared_down: pack_group(ck, &format!("{mp}.shared_experts.down_proj"), &mut seen),
        up,
        down,
    };
    for ((k, n, twice, cb), count) in &seen {
        eprintln!("pack layer {layer}: {count:>4} EXL3 projections [{k}, {n}], {} bits, codebook {cb}", *twice as f64 / 2.0);
    }
    w
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Bits

fn digest_f32(v: &[f32]) -> u64 {
    fnv1a(v.iter().flat_map(|x| x.to_bits().to_le_bytes()))
}

fn digest_i32(v: &[i32]) -> u64 {
    fnv1a(v.iter().flat_map(|x| x.to_le_bytes()))
}

/// Everything a call produces: the output rows, the router's logits, picks and weights, digested.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Digest {
    out: u64,
    logits: u64,
    pick: u64,
    wts: u64,
}

fn run_digest(layer: &Layer, scratch: &mut MoeScratch, h: &[f32], rows: usize, st: &Stream) -> Digest {
    let hd = DevBuf::from_slice(h).unwrap();
    let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
    scratch.poison(st).unwrap();
    layer.moe.forward_in(scratch, &hd, rows, &mut out, st).unwrap();
    st.sync().unwrap();
    let o = out.to_vec::<f32>().unwrap();
    let logits = scratch.download_logits(rows, st).unwrap();
    let (pick, wts) = scratch.download_route(rows, st).unwrap();
    Digest { out: digest_f32(&o), logits: digest_f32(&logits), pick: digest_i32(&pick), wts: digest_f32(&wts) }
}

/// The digests of the pre-change kernels (the commit before the first kernel change), by (case, rows): out, logits, pick, wts. The test
/// prints each digest in this form, so a deliberate change of bits (never allowed here) would re-record them. An empty table passes.
const GOLDEN: &[(&str, usize, [u64; 4])] = &[
    ("sparse14", 1, [0x2f0bb6fb471900e7, 0x7a4edcba92caa6c5, 0xb096583bafcf55f0, 0x1ba97dc262eadaac]),
    ("sparse14", 7, [0xa1e0f0c4cdfd8181, 0x4794624fe8c22b91, 0x5ca32c713a5039b6, 0x1dcb5ffe97d92867]),
    ("sparse14", 64, [0x0742613aa1cb7799, 0xda95c9a3f9fb706f, 0xa4aac2e4160f6847, 0xc95aa6581a2b96a8]),
    ("all512", 1, [0xc6ef678ad7a86f3f, 0x9bf4051a0dbde65a, 0xbfdb0859b5d3866e, 0x0fe171eb66ff3e86]),
    ("all512", 7, [0x68c526dec1a9c98d, 0x425d9322658f3dda, 0x305332d90e460c1e, 0xdf4ede46c16df421]),
    ("all512", 64, [0x6824a2ad9689793e, 0xbeb123ddd3d684ee, 0x37589a0142b40be7, 0x7dd127418bf246d3]),
];

fn check_golden(case: &str, rows: usize, d: Digest) {
    let got = [d.out, d.logits, d.pick, d.wts];
    eprintln!("GOLDEN (\"{case}\", {rows}, [{:#018x}, {:#018x}, {:#018x}, {:#018x}]),", got[0], got[1], got[2], got[3]);
    // GOLDEN ships non-empty: emptying the list is the regolden mode (the digests print above),
    // so this arm is unreachable as committed, on purpose.
    #[allow(clippy::const_is_empty)]
    if GOLDEN.is_empty() {
        return;
    }
    let want = GOLDEN
        .iter()
        .find(|g| g.0 == case && g.1 == rows)
        .unwrap_or_else(|| panic!("no golden digest for {case} at {rows} rows"));
    assert_eq!(got, want.2, "{case}, {rows} rows: the block's bits differ from the pre-change kernels (out, logits, pick, wts)");
}

/// The synthetic layers at 1, 7 and 64 rows against the digests of the pre-change kernels.
#[test]
fn block_bits_match_the_pre_change_kernels() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut scratch = MoeScratch::new(64).unwrap();
    for (name, seed, present, bf16) in synthetic_cases() {
        let layer = Layer::new(synthetic_layer(seed, &present, &bf16));
        for &rows in &[1usize, 7, 64] {
            let h = Rng(0xB17 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0);
            let d = run_digest(&layer, &mut scratch, &h, rows, &st);
            check_golden(name, rows, d);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The scheduled kernels against the reference launches, in process

/// Output, router logits, picks and weights of `forward_in` (`fast`) or `forward_reference` on the same input, on the host.
fn run_full(
    layer: &Layer,
    scratch: &mut MoeScratch,
    h: &[f32],
    rows: usize,
    fast: bool,
    st: &Stream,
) -> (Vec<f32>, Vec<f32>, Vec<i32>, Vec<f32>) {
    let hd = DevBuf::from_slice(h).unwrap();
    // Poison the output and every intermediate: a stage that does not write what it should cannot pass by finding the previous call's
    // values (a pair routed to an absent expert must come out as zeros, for one).
    let mut out = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * HIDDEN]).unwrap();
    scratch.poison(st).unwrap();
    if fast {
        layer.moe.forward_in(scratch, &hd, rows, &mut out, st).unwrap();
    } else {
        layer.moe.forward_reference(scratch, &hd, rows, &mut out, st).unwrap();
    }
    st.sync().unwrap();
    let o = out.to_vec::<f32>().unwrap();
    let logits = scratch.download_logits(rows, st).unwrap();
    let (pick, wts) = scratch.download_route(rows, st).unwrap();
    (o, logits, pick, wts)
}

fn assert_same_bits(what: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    let bad: Vec<usize> = (0..a.len()).filter(|&i| a[i].to_bits() != b[i].to_bits()).collect();
    assert!(
        bad.is_empty(),
        "{what}: {} of {} values differ in bits, first at {} ({:e} against {:e})",
        bad.len(),
        a.len(),
        bad[0],
        a[bad[0]],
        b[bad[0]]
    );
}

/// `layer` at each of `rows_list`: the scheduled launches and the reference launches give the same output, logits, picks and weights.
fn check_fast_equals_reference(label: &str, layer: &Layer, rows_list: &[usize]) {
    let st = Stream::new().unwrap();
    let mut scratch = MoeScratch::new(*rows_list.iter().max().unwrap()).unwrap();
    for &rows in rows_list {
        let h = Rng(0xFA57 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0);
        let (fo, fl, fp, fw) = run_full(layer, &mut scratch, &h, rows, true, &st);
        let (ro, rl, rp, rw) = run_full(layer, &mut scratch, &h, rows, false, &st);
        assert!(fo.iter().any(|v| *v != 0.0 && v.is_finite()), "{label}, {rows} rows: the output is all zero or not finite");
        assert_same_bits(&format!("{label}, {rows} rows: output"), &fo, &ro);
        assert_same_bits(&format!("{label}, {rows} rows: router logits"), &fl, &rl);
        assert_same_bits(&format!("{label}, {rows} rows: router weights"), &fw, &rw);
        assert_eq!(fp, rp, "{label}, {rows} rows: router picks");
        eprintln!("{label}, {rows} rows: output, logits, picks and weights are bitwise the reference launches'");
    }
}

#[test]
fn scheduled_launches_equal_the_reference_launches_on_synthetic_layers() {
    gpu!();
    for (name, seed, present, bf16) in synthetic_cases() {
        let layer = Layer::new(synthetic_layer(seed, &present, &bf16));
        assert_eq!(
            layer.moe.launch_summary(),
            "routed nm2_gemm4 + reference; fc1 nm2_gemm4, fc2 nm2_gemm4, shared up nm2_gemm4, shared down nm2_gemm4; side stream",
            "{name}: a 4-bit table keeps nm2_gemm4"
        );
        check_fast_equals_reference(name, &layer, &[1, 2, 3, 7, 8, 9, 16, 33, 64]);
        // The shared expert in line on the caller's stream instead of the side stream: the same bits.
        gb_cuda::nemotron_moe::set_side_stream(false);
        check_fast_equals_reference(&format!("{name}, shared expert in line"), &layer, &[1, 8, 64]);
        gb_cuda::nemotron_moe::set_side_stream(true);
    }
}

/// An F16 router gate (as stored; and the f32 form, decision D1 of v0.1.2) on the full layer: the scheduled launches equal the
/// reference ones.
#[test]
fn scheduled_launches_equal_the_reference_launches_with_an_f32_gate() {
    gpu!();
    let (name, seed, present, bf16) = synthetic_cases().remove(1);
    let mut w = synthetic_layer(seed, &present, &bf16);
    let mut r = Rng(0xF32);
    let bits: Vec<u16> = (0..EXPERTS * HIDDEN).map(|_| gb_exl3::f16::f64_to_f16(r.uniform(-0.05, 0.05) as f64)).collect();
    w.gate = DevBuf::from_slice(&bits).unwrap();
    w.gate_dtype = gb_cuda::nemotron_moe::GateDtype::F16;
    let layer = Layer::new(w);
    assert_eq!(layer.moe.gate_dtype(), gb_cuda::nemotron_moe::GateDtype::F16);
    check_fast_equals_reference(&format!("{name}, F16 gate as stored"), &layer, &[1, 2, 7, 12, 13, 64, 100]);
    let mut w = synthetic_layer(seed, &present, &bf16);
    w.gate = DevBuf::from_slice(&gb_cuda::nemotron_moe::gate_f32_from_f16(&bits)).unwrap();
    w.gate_dtype = gb_cuda::nemotron_moe::GateDtype::F32;
    check_fast_equals_reference(&format!("{name}, f32 gate"), &Layer::new(w), &[1, 13, 64]);
}

/// MixedK layers (v0.1.2 lane K, P1-5): the scheduled launches (`nm2_gemmk` for the routed experts at mixed widths, `nm2_gemmk` for
/// dense projections of another width than 4, `nm2_dense16` for fp16 ones) equal the reference launches bit for bit: output, logits,
/// picks and weights, at the decode and verify windows and past the passes of 8 and the dense chunk of 16, with the shared expert on
/// the side stream and in line.
#[test]
fn scheduled_launches_equal_the_reference_launches_on_mixedk_layers() {
    gpu!();
    let want = [
        "routed nm2_gemmk; fc1 nm2_dense16, fc2 nm2_dense16, shared up nm2_gemmk, shared down nm2_gemm4; side stream",
        "routed nm2_gemmk + reference; fc1 nm2_gemmk, fc2 nm2_gemmk, shared up nm2_gemmk, shared down nm2_gemmk; side stream",
        "routed nm2_gemmk; fc1 nm2_gemm4, fc2 nm2_gemm4, shared up nm2_gemm4, shared down nm2_gemm4; side stream",
    ];
    for ((name, w), want) in mixed_cases().into_iter().zip(want) {
        let layer = Layer::new(w);
        assert_eq!(layer.moe.launch_summary(), want, "{name}: the launches");
        check_fast_equals_reference(name, &layer, &[1, 2, 3, 7, 8, 9, 16, 17, 33, 64]);
        gb_cuda::nemotron_moe::set_side_stream(false);
        check_fast_equals_reference(&format!("{name}, shared expert in line"), &layer, &[1, 8, 64]);
        gb_cuda::nemotron_moe::set_side_stream(true);
    }
}

/// Every integer width alone (all 512 experts and the four dense projections at K bits, K = 2 .. 8): scheduled == reference.
#[test]
fn scheduled_launches_equal_the_reference_launches_at_every_width() {
    gpu!();
    let all: Vec<usize> = (0..EXPERTS).collect();
    for k in 2..=8usize {
        let layer = Layer::new(mixed_layer(300 + k as u64, &all, &[], |_| k, |_| k, [k; 4]));
        let kern = if k == 4 { "nm2_gemm4" } else { "nm2_gemmk" };
        assert_eq!(
            layer.moe.launch_summary(),
            format!("routed {kern}; fc1 {kern}, fc2 {kern}, shared up {kern}, shared down {kern}; side stream"),
            "{k} bits"
        );
        check_fast_equals_reference(&format!("all {k}-bit"), &layer, &[1, 4, 9, 64]);
    }
}

/// `nm2_gemmk` at 4 bits gives `nm2_gemm4`'s bits: the published pack's layer shape (all 512 experts 4-bit, two bf16) run with the
/// any-width kernel forced (`set_any_width_kernel`) equals its default run (nm2_gemm4), output, logits, picks and weights, and the
/// GOLDEN digests hold under the switch too.
#[test]
fn the_any_width_kernel_at_4_bits_gives_the_4_bit_kernels_bits() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut scratch = MoeScratch::new(64).unwrap();
    for (name, seed, present, bf16) in synthetic_cases() {
        let layer = Layer::new(synthetic_layer(seed, &present, &bf16));
        for &rows in &[1usize, 7, 64] {
            let h = Rng(0xB17 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0);
            let a = run_full(&layer, &mut scratch, &h, rows, true, &st);
            gb_cuda::nemotron_moe::set_any_width_kernel(true);
            let b = run_full(&layer, &mut scratch, &h, rows, true, &st);
            let d = run_digest(&layer, &mut scratch, &h, rows, &st);
            gb_cuda::nemotron_moe::set_any_width_kernel(false);
            assert_same_bits(&format!("{name}, {rows} rows: output, nm2_gemmk vs nm2_gemm4"), &b.0, &a.0);
            assert_same_bits(&format!("{name}, {rows} rows: logits"), &b.1, &a.1);
            assert_eq!(b.2, a.2, "{name}, {rows} rows: picks");
            check_golden(name, rows, d);
        }
    }
}

/// More rows than the decode and verify windows (the router's selection scans for its buckets above 64 rows, the projections take more
/// passes): the same bits.
#[test]
fn scheduled_launches_equal_the_reference_launches_above_64_rows() {
    gpu!();
    let (name, seed, present, bf16) = synthetic_cases().remove(1);
    let layer = Layer::new(synthetic_layer(seed, &present, &bf16));
    check_fast_equals_reference(name, &layer, &[65, 100, 128]);
}

/// The pack's layer 1 (all 512 experts, 4-bit Mul1): the scheduled launches against the reference ones. Reads the pack only.
#[test]
fn scheduled_launches_equal_the_reference_launches_on_the_pack_layer() {
    gpu!();
    let Some(dir) = pack_dir() else { return };
    let ck = scan(&dir, true).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let layer = Layer::new(pack_layer(&ck, 1));
    // the published pack takes exactly the launches it took before MixedK (v0.1.2 R-MK-noreg (b))
    assert_eq!(
        layer.moe.launch_summary(),
        "routed nm2_gemm4; fc1 nm2_gemm4, fc2 nm2_gemm4, shared up nm2_gemm4, shared down nm2_gemm4; side stream"
    );
    check_fast_equals_reference("pack layer 1", &layer, &[1, 7, 64]);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Timing

/// The row counts of the timing tests: 1, 8 and 64, or `REXL3_NEMO_PERF_ROWS="1,16"`.
fn bench_rows() -> Vec<usize> {
    match std::env::var("REXL3_NEMO_PERF_ROWS") {
        Ok(v) => v
            .split(',')
            .map(|s| s.trim().parse().expect("REXL3_NEMO_PERF_ROWS is a comma-separated list of row counts"))
            .collect(),
        Err(_) => vec![1, 8, 64],
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// The per-stage table of `layer` at each of `rows_list`, `reps` calls on different inputs, medians. Before each timed call another
/// layer (`other`) runs on the same stream, as the previous layer does in a decode step: its weights are clean lines that fill L2, so the
/// timed call finds the cache cold and holding nothing dirty. (A memset of a big buffer leaves dirty lines that the timed call must
/// write back, which costs a streaming kernel a third of its bandwidth; a device-to-host copy leaves a state that is slower still.)
fn stage_table(label: &str, layer: &Layer, other: &Layer, rows_list: &[usize], reps: usize, fast: bool) {
    let st = Stream::new_nonblocking().unwrap();
    let mut scratch = MoeScratch::new(*rows_list.iter().max().unwrap()).unwrap();
    let stages =
        [Stage::Router, Stage::Bucket, Stage::LatentIn, Stage::Up, Stage::Down, Stage::Combine, Stage::Fc2, Stage::Shared];
    let names = ["router", "bucket", "latent in", "up", "down", "combine", "fc2", "shared"];
    let events: Vec<Ev> = (0..stages.len() + 1).map(|_| Ev::new()).collect();
    let (whole_a, whole_b) = (Ev::new(), Ev::new());
    let which = if fast { "scheduled launches" } else { "reference launches" };
    eprintln!("\n=== {label}, {which}: per-stage times, another layer run before each call, median of {reps}, bandwidth floor at {FLOOR_GBPS} GB/s ===");
    for &rows in rows_list {
        let hs: Vec<DevBuf> = (0..reps)
            .map(|i| DevBuf::from_slice(&Rng(7000 + 31 * i as u64 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0)).unwrap())
            .collect();
        let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); stages.len()];
        let mut bytes: Vec<Vec<f64>> = vec![Vec::new(); stages.len()];
        let mut whole: Vec<f64> = Vec::new();
        let mut distinct: Vec<f64> = Vec::new();
        let run = |scratch: &mut MoeScratch,
                   h: &DevBuf,
                   out: &mut DevBuf,
                   mark: &mut dyn FnMut(Stage) -> Result<(), gb_cuda::CudaError>| {
            if fast {
                layer.moe.forward_staged(scratch, h, rows, out, &st, mark).unwrap();
            } else {
                layer.moe.forward_reference_staged(scratch, h, rows, out, &st, mark).unwrap();
            }
        };
        // One warm-up call (code, tables), then the timed ones.
        run(&mut scratch, &hs[0], &mut out, &mut |_| Ok(()));
        st.sync().unwrap();
        for i in 0..reps {
            other.moe.forward_in(&mut scratch, &hs[(i + 1) % reps], rows, &mut out, &st).unwrap();
            events[0].record(&st);
            let mut k = 0usize;
            run(&mut scratch, &hs[i], &mut out, &mut |stage| {
                assert_eq!(stage, stages[k]);
                k += 1;
                events[k].record(&st);
                Ok(())
            });
            assert_eq!(k, stages.len());
            for s in 0..stages.len() {
                times[s].push(events[s].us_to(&events[s + 1]));
            }
            let (pick, _) = scratch.download_route(rows, &st).unwrap();
            let sb = layer.stage_bytes(&pick, rows);
            for (s, n) in names.iter().enumerate() {
                bytes[s].push(sb[n] as f64);
            }
            distinct.push(pick.iter().collect::<BTreeSet<_>>().len() as f64);
            if rows == 1 && i < 3 && std::env::var_os("REXL3_NEMO_PERF_PICKS").is_some() {
                eprintln!("PICKS {:?}", pick);
            }
            // The same call with no marks between the stages.
            other.moe.forward_in(&mut scratch, &hs[(i + 1) % reps], rows, &mut out, &st).unwrap();
            whole_a.record(&st);
            run(&mut scratch, &hs[i], &mut out, &mut |_| Ok(()));
            whole_b.record(&st);
            whole.push(whole_a.us_to(&whole_b));
        }
        let (mut total_t, mut total_b) = (0f64, 0f64);
        eprintln!("--- {rows} row(s), {:.0} distinct experts per call ---", median(&mut distinct));
        eprintln!("{:<10} {:>10} {:>10} {:>9} {:>10}", "stage", "us", "MB", "GB/s", "floor us");
        for s in 0..stages.len() {
            let t = median(&mut times[s]);
            let b = median(&mut bytes[s]);
            total_t += t;
            total_b += b;
            let gbps = if b > 0.0 { format!("{:.1}", b / t / 1e3) } else { "-".into() };
            eprintln!("{:<10} {:>10.1} {:>10.2} {:>9} {:>10.1}", names[s], t, b / 1e6, gbps, b / (FLOOR_GBPS * 1e3));
        }
        let w = median(&mut whole);
        eprintln!(
            "{:<10} {:>10.1} {:>10.2} {:>9.1} {:>10.1}   (stages summed {:.1} us; whole block unmarked {:.1} us = {:.2}x the floor)",
            "block",
            w,
            total_b / 1e6,
            total_b / w / 1e3,
            total_b / (FLOOR_GBPS * 1e3),
            total_t,
            w,
            w / (total_b / (FLOOR_GBPS * 1e3))
        );
    }
}

/// Whole-block microseconds of `layer` (median of `reps` calls on different inputs, another layer run before each), on the reference
/// launches or the scheduled ones, with the current launch switches.
fn whole_block_us(layer: &Layer, other: &Layer, rows: usize, reps: usize, fast: bool) -> f64 {
    let st = Stream::new_nonblocking().unwrap();
    let mut scratch = MoeScratch::new(rows).unwrap();
    let hs: Vec<DevBuf> = (0..reps)
        .map(|i| DevBuf::from_slice(&Rng(9000 + 17 * i as u64 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0)).unwrap())
        .collect();
    let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
    let (a, b) = (Ev::new(), Ev::new());
    let mut t = Vec::new();
    for i in 0..reps + 1 {
        other.moe.forward_in(&mut scratch, &hs[(i + 1) % reps], rows, &mut out, &st).unwrap();
        a.record(&st);
        if fast {
            layer.moe.forward_in(&mut scratch, &hs[i % reps], rows, &mut out, &st).unwrap();
        } else {
            layer.moe.forward_reference(&mut scratch, &hs[i % reps], rows, &mut out, &st).unwrap();
        }
        b.record(&st);
        let us = a.us_to(&b);
        if i > 0 {
            t.push(us);
        }
    }
    median(&mut t)
}

/// What each launch-level change is worth on the pack's layer: the whole block with programmatic dependent launch and the side stream
/// switched on and off, against the reference launches.
#[test]
#[ignore = "timing"]
fn moe_bench_launch_variants_pack_layer() {
    gpu!();
    use gb_cuda::nemotron_moe::{set_programmatic_launch, set_side_stream};
    let Some(dir) = pack_dir() else { return };
    let ck = scan(&dir, true).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let layer = Layer::new(pack_layer(&ck, 1));
    let other = Layer::new(pack_layer(&ck, 3));
    eprintln!("\n=== pack layer 1, whole block (us), another layer run before each call ===");
    eprintln!("{:>5} {:>12} {:>14} {:>16} {:>16}", "rows", "reference", "no PDL no side", "PDL, no side", "PDL and side");
    for rows in bench_rows() {
        let reference = whole_block_us(&layer, &other, rows, 5, false);
        set_programmatic_launch(false);
        set_side_stream(false);
        let plain = whole_block_us(&layer, &other, rows, 15, true);
        set_programmatic_launch(true);
        let pdl = whole_block_us(&layer, &other, rows, 15, true);
        set_side_stream(true);
        let both = whole_block_us(&layer, &other, rows, 15, true);
        eprintln!("{rows:>5} {reference:>12.1} {plain:>14.1} {pdl:>16.1} {both:>16.1}");
    }
}

/// Which (bits, codebook) the pack's MoE-related EXL3 projections (routed experts, shared expert, latent projections) are stored with,
/// counted over every layer: only the headers are read.
/// The per-format counters of `pack_moe_projection_formats`.
type SeenFormats = BTreeMap<(String, usize, usize, u8, String, bool, &'static str), usize>;

#[test]
fn pack_moe_projection_formats() {
    gpu!();
    let Some(dir) = pack_dir() else { return };
    let ck = scan(&dir, true).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let mut seen: SeenFormats = BTreeMap::new();
    for (name, g) in &ck.groups {
        let kind = if name.contains(".experts.") {
            "expert"
        } else if name.contains("shared_experts") {
            "shared"
        } else if name.contains("latent_proj") {
            "latent"
        } else {
            continue;
        };
        *seen.entry((kind.into(), g.k, g.n, g.bits.twice(), g.codebook.name().into(), g.bias, g.in_scales)).or_insert(0) += 1;
    }
    for ((kind, k, n, twice, cb, bias, su), count) in &seen {
        eprintln!("{kind:<7} {count:>6} x [{k}, {n}] {} bits, {cb}, bias {bias}, input scales {su}", *twice as f64 / 2.0);
    }
    assert!(!seen.is_empty(), "no MoE projections found");
}

/// The per-stage tables on a synthetic layer with all 512 experts present (EXL3 4-bit, two bf16 experts): reference launches, then the
/// scheduled ones.
#[test]
#[ignore = "timing"]
fn moe_bench_stage_table_synthetic_all_experts() {
    gpu!();
    let all: Vec<usize> = (0..EXPERTS).collect();
    let layer = Layer::new(synthetic_layer(101, &all, &[5, 300]));
    let other = Layer::new(synthetic_layer(202, &all, &[]));
    if std::env::var_os("REXL3_NEMO_PERF_SKIP_REFERENCE").is_none() {
        stage_table("synthetic layer, 512 experts present", &layer, &other, &bench_rows(), 9, false);
    }
    stage_table("synthetic layer, 512 experts present", &layer, &other, &bench_rows(), 15, true);
}

/// Per-width throughput (v0.1.2 lane K, P1-5): the per-stage tables of a layer whose 512 experts are all at K bits (K = 2 .. 8; the
/// dense projections 4-bit, so the routed stages are the width's), on the scheduled launches (nm2_gemmk for K != 4, nm2_gemm4 at 4),
/// then the 4-bit layer through nm2_gemmk (`set_any_width_kernel`), then his pack's shape (mixed widths, fp16 fc1 and fc2) on the
/// scheduled and the reference launches. `REXL3_NEMO_PERF_WIDTHS="2,4"` picks the widths.
#[test]
#[ignore = "timing"]
fn moe_bench_widths() {
    gpu!();
    let all: Vec<usize> = (0..EXPERTS).collect();
    let other = Layer::new(synthetic_layer(202, &all, &[]));
    let widths: Vec<usize> = match std::env::var("REXL3_NEMO_PERF_WIDTHS") {
        Ok(v) => v.split(',').map(|s| s.trim().parse().unwrap()).collect(),
        Err(_) => (2..=8).collect(),
    };
    for &k in &widths {
        let layer = Layer::new(mixed_layer(400 + k as u64, &all, &[], |_| k, |_| k, [4; 4]));
        stage_table(&format!("all experts {k}-bit, dense 4-bit"), &layer, &other, &bench_rows(), 15, true);
        if k == 4 {
            gb_cuda::nemotron_moe::set_any_width_kernel(true);
            stage_table("all experts 4-bit through nm2_gemmk", &layer, &other, &bench_rows(), 15, true);
            gb_cuda::nemotron_moe::set_any_width_kernel(false);
        }
    }
    let (name, his) = mixed_cases().remove(0);
    let his = Layer::new(his);
    stage_table(name, &his, &other, &bench_rows(), 15, true);
    if std::env::var_os("REXL3_NEMO_PERF_SKIP_REFERENCE").is_none() {
        stage_table(name, &his, &other, &bench_rows(), 5, false);
    }
}

/// The per-stage tables on the pack's layer (read only).
#[test]
#[ignore = "timing"]
fn moe_bench_stage_table_pack_layer() {
    gpu!();
    let Some(dir) = pack_dir() else { return };
    let ck = scan(&dir, true).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let layer_idx: usize = std::env::var("REXL3_NEMO_PERF_LAYER").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let layer = Layer::new(pack_layer(&ck, layer_idx));
    // The other MoE layer that runs before each timed call (layers 1, 3, 5, ... are MoE layers).
    let other = Layer::new(pack_layer(&ck, if layer_idx == 3 { 5 } else { 3 }));
    if std::env::var_os("REXL3_NEMO_PERF_SKIP_REFERENCE").is_none() {
        stage_table(&format!("pack layer {layer_idx}, 512 experts"), &layer, &other, &bench_rows(), 5, false);
    }
    stage_table(&format!("pack layer {layer_idx}, 512 experts"), &layer, &other, &bench_rows(), 15, true);
}
