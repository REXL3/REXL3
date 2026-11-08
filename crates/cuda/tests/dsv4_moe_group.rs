//! DeepSeek-V4.1's routed-expert block on the device (`src/dsv4_moe.rs`'s `MoeScratch` / `moe_experts`,
//! `csrc/dsv4_moe_group.cu`): grouping, gather, per-expert segments at each expert's own bit width, and the weighted
//! scatter-add.
//!
//! # What is compared, and against what
//!
//! The reference is the *same arithmetic done per token*: for every `(row, k)` slot, run its expert's `w1` and `w3` on
//! one row, apply the clamps and the silu on the host in fp32 (rounding once to fp16, exactly as the kernel does), run
//! `w2` on that one row, and accumulate `w * y` into the row in fp32 in ascending `k`. That is the old host-driven loop,
//! which is the thing this replaces — so the test pins the grouping, the permutation, the packing, the segment views, the
//! dtypes and the scatter against it. The GEMM itself is not re-derived here: it is the same `Exl3Weights::forward` on
//! both sides, and `exl3_big.rs` / the EXL3 suite pin it against the decode path and the pack.
//!
//! The experts are **mixed-K**: each of the four experts has its own `(w1, w2, w3)` bit widths (1..6, and `w1 != w3`),
//! its own codebook, and its own random trellis, so a block that assumed one width for the layer decodes garbage.
//!
//! GPU tests self-skip with "not run: no device" when there is no device.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::Rng;
use gb_cuda::dsv4_moe::{moe_experts, swiglu_scalar, MoeGroups, MoeScratch};
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::rt::Block;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::{Group, LoadedGroup};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::Bits;
use gb_exl3::Codebook;
use std::sync::Arc;

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("not run: no device");
            return;
        }
    };
}

const H: usize = 256; // model width
const I: usize = 128; // expert width
const E: usize = 4; // experts
const TOPK: usize = 3;
const LIMIT: f32 = 10.0;

fn normal(r: &mut Rng) -> f64 {
    let (a, b) = ((r.next_u64() >> 11) as f64 / (1u64 << 53) as f64, (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64);
    (-2.0 * a.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
}

fn scale_vec(r: &mut Rng, n: usize, mag: f64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            f16_to_f64(f64_to_f16(
                (if r.next_u64() & 1 == 1 { -1.0 } else { 1.0 }) * mag * (0.5 + (r.next_u64() >> 40) as f64 / 16777216.0),
            ))
        })
        .collect()
}

/// One projection's raw parts, as the pack states them: random trellis words, fp16 scales.
struct Parts {
    k: usize,
    n: usize,
    bits: usize,
    cb: Codebook,
    trellis: Vec<i16>,
    suh: Vec<f64>,
    svh: Vec<f64>,
}

fn parts(r: &mut Rng, k: usize, n: usize, bits: usize, cb: Codebook) -> Parts {
    let words = (k / 16) * (n / 16) * 16 * bits;
    Parts {
        k,
        n,
        bits,
        cb,
        trellis: (0..words).map(|_| r.next_u64() as i16).collect(),
        suh: scale_vec(r, k, 1.0),
        svh: scale_vec(r, n, 1.0 / (k as f64).sqrt()),
    }
}

fn group_of(p: &Parts) -> Group {
    Group {
        prefix: "synthetic".into(),
        bits: Bits::from_twice(2 * p.bits as u8).unwrap(),
        codebook: p.cb,
        k: p.k,
        n: p.n,
        in_scales: "suh",
        out_scales: "svh",
        bias: false,
        files: vec![],
        bytes: 0,
    }
}

fn triples() -> Vec<[(usize, Codebook); 3]> {
    // (w1, w2, w3) a expert, as (bits, codebook); w1 != w3 on every expert, and both ends of the pack's 1..=6 range
    vec![
        [(1, Codebook::Mul1), (6, Codebook::Mcg), (3, Codebook::Inst3)],
        [(6, Codebook::Inst3), (1, Codebook::Mul1), (6, Codebook::Mcg)],
        [(3, Codebook::Mcg), (3, Codebook::Inst3), (1, Codebook::Mul1)],
        [(2, Codebook::Mul1), (4, Codebook::Mcg), (5, Codebook::Inst3)],
    ]
}

/// The four experts of the test bank: `[expert][w1, w2, w3]`, each projection its own width.
fn bank_parts() -> Vec<[Parts; 3]> {
    let mut r = Rng(0x4453_5634_3121);
    triples()
        .into_iter()
        .enumerate()
        .map(|(e, t)| {
            let _ = e;
            [parts(&mut r, H, I, t[0].0, t[0].1), parts(&mut r, I, H, t[1].0, t[1].1), parts(&mut r, H, I, t[2].0, t[2].1)]
        })
        .collect()
}

fn loaded(p: &Parts) -> LoadedGroup {
    LoadedGroup {
        group: group_of(p),
        trellis: p.trellis.clone(),
        kt: p.k / 16,
        nt: p.n / 16,
        suh: p.suh.clone(),
        svh: p.svh.clone(),
        bias: None,
    }
}

fn owned_bank(parts: &[[Parts; 3]]) -> Vec<[Exl3Weights; 3]> {
    parts
        .iter()
        .map(|t| {
            [
                Exl3Weights::from_loaded(&loaded(&t[0])).unwrap(),
                Exl3Weights::from_loaded(&loaded(&t[1])).unwrap(),
                Exl3Weights::from_loaded(&loaded(&t[2])).unwrap(),
            ]
        })
        .collect()
}

fn align512(n: usize) -> usize {
    n.div_ceil(512) * 512
}

/// The same bank, laid out as a streamed expert would be: one slab, every projection at its own offset inside it, and
/// every `Exl3Weights` a view filled from the pack's own bytes — what the ExpertCache's `slot_view` hands the model.
fn slab_bank(parts: &[[Parts; 3]]) -> (Arc<Block>, Vec<[Exl3Weights; 3]>) {
    let mut offsets = Vec::new();
    let mut total = 0usize;
    for t in parts {
        let mut row = [(0usize, 0usize, 0usize); 3];
        for (i, p) in t.iter().enumerate() {
            let trellis = (p.k / 16) * (p.n / 16) * 16 * p.bits * 2;
            row[i] = (total, trellis, p.k * 2);
            total += align512(trellis + p.k * 2 + p.n * 2);
        }
        offsets.push(row);
    }
    let block = Arc::new(Block::alloc(total).unwrap());
    let bank = parts
        .iter()
        .zip(&offsets)
        .map(|(t, o)| {
            let mut three = Vec::with_capacity(3);
            for (i, p) in t.iter().enumerate() {
                let (off, trellis_bytes, suh_bytes) = o[i];
                let mut trellis = DevBuf::view(&block, off, trellis_bytes);
                let mut suh = DevBuf::view(&block, off + trellis_bytes, suh_bytes);
                let mut svh = DevBuf::view(&block, off + trellis_bytes + suh_bytes, p.n * 2);
                trellis.upload(&p.trellis).unwrap();
                suh.upload(&p.suh.iter().map(|&v| f64_to_f16(v)).collect::<Vec<u16>>()).unwrap();
                svh.upload(&p.svh.iter().map(|&v| f64_to_f16(v)).collect::<Vec<u16>>()).unwrap();
                let dims = gb_cuda::exl3_linear::Exl3Dims { k: p.k, n: p.n, bits: p.bits, codebook: p.cb };
                three.push(Exl3Weights::from_dims(dims, trellis, suh, svh, None).unwrap());
            }
            [three.remove(0), three.remove(0), three.remove(0)]
        })
        .collect();
    (block, bank)
}

fn f16_bytes(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f64_to_f16(x as f64)).collect()
}

fn unf16(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&x| f16_to_f64(x) as f32).collect()
}

fn dl16(b: &DevBuf, st: &Stream) -> Vec<f32> {
    st.sync().unwrap();
    unf16(&b.to_vec::<u16>().unwrap())
}

/// The old per-token loop: one row an expert, host swiglu, host accumulation, in ascending `k`.
fn host_reference(
    bank: &[[Exl3Weights; 3]],
    x: &[f32],
    picks: &[i32],
    wts: &[f32],
    rows: usize,
    shared: &[f32],
    st: &Stream,
) -> Vec<f32> {
    let mut out = shared.to_vec();
    let mut ws = DevBuf::zeroed(bank[0][0].workspace_bytes(1).max(1024)).unwrap();
    for r in 0..rows {
        for k in 0..TOPK {
            let e = picks[r * TOPK + k] as usize;
            let xr = DevBuf::from_slice(&f16_bytes(&x[r * H..(r + 1) * H])).unwrap();
            let mut g = DevBuf::zeroed(I * 2).unwrap();
            let mut u = DevBuf::zeroed(I * 2).unwrap();
            let mut y = DevBuf::zeroed(H * 2).unwrap();
            bank[e][0].forward(&xr, &mut g, 1, Dtype::F16, Dtype::F16, &mut ws, st).unwrap();
            bank[e][2].forward(&xr, &mut u, 1, Dtype::F16, Dtype::F16, &mut ws, st).unwrap();
            let (gv, uv) = (dl16(&g, st), dl16(&u, st));
            let act: Vec<f32> = (0..I).map(|i| swiglu_scalar(gv[i], uv[i], 1.0, LIMIT)).collect();
            let ab = DevBuf::from_slice(&f16_bytes(&act)).unwrap();
            bank[e][1].forward(&ab, &mut y, 1, Dtype::F16, Dtype::F16, &mut ws, st).unwrap();
            let yv = dl16(&y, st);
            let w = wts[r * TOPK + k];
            for d in 0..H {
                out[r * H + d] += w * yv[d];
            }
        }
    }
    out
}

/// A deterministic selection: `rows * topk` distinct-per-row expert ids and fp16 weights in `[-1, 1]`.
fn selection(rows: usize, seed: u64) -> (Vec<i32>, Vec<f32>) {
    let mut r = Rng(seed);
    let mut picks = vec![0i32; rows * TOPK];
    let mut wts = vec![0f32; rows * TOPK];
    for row in 0..rows {
        let mut pool: Vec<i32> = (0..E as i32).collect();
        for k in 0..TOPK {
            let j = (r.next_u64() % pool.len() as u64) as usize;
            picks[row * TOPK + k] = pool.remove(j);
            wts[row * TOPK + k] = f16_to_f64(f64_to_f16(normal(&mut r) * 0.3)) as f32;
        }
    }
    (picks, wts)
}

fn run(
    bank: &[[Exl3Weights; 3]],
    rows: usize,
    seed: u64,
    st: &Stream,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, gb_cuda::dsv4_moe::MoeStats) {
    let (picks, wts) = selection(rows, seed);
    let mut r = Rng(seed ^ 0x9e37);
    let x: Vec<f32> = (0..rows * H).map(|_| normal(&mut r) as f32).collect();
    let shared: Vec<f32> = (0..rows * H).map(|_| normal(&mut r) as f32).collect();

    let xb = DevBuf::from_slice(&f16_bytes(&x)).unwrap();
    let mut sharedb = DevBuf::from_slice(&f16_bytes(&shared)).unwrap();
    let mut outb = DevBuf::zeroed(rows * H * 4).unwrap();
    let picksb = DevBuf::from_slice(&picks).unwrap();
    let wtsb = DevBuf::from_slice(&f16_bytes(&wts)).unwrap();
    let mut moe = MoeScratch::new(rows, TOPK, H, I, E).unwrap();

    let stats = moe_experts(
        &mut moe,
        |e| Ok(MoeGroups::Resident(&bank[e])),
        rows,
        &xb,
        &mut sharedb,
        &picksb,
        &wtsb,
        &mut outb,
        LIMIT,
        st,
    )
    .unwrap();

    let want = host_reference(bank, &x, &picks, &wts, rows, &unf16(&f16_bytes(&shared)), st);
    (dl32(&outb, st), dl16(&sharedb, st), want, stats)
}

fn dl32(b: &DevBuf, st: &Stream) -> Vec<f32> {
    st.sync().unwrap();
    b.to_vec::<f32>().unwrap()
}

/// Both of the block's outputs against the per-token loop.
///
/// The tolerance is fp16 noise, not slack: the reference runs each expert on **one** row and the block runs it on its
/// whole segment, and the EXL3 decode path's reduction order depends on `m` (`rotate_in` vs `rotate_in_frag`, and the
/// tile grouping), so two evaluations of the same arithmetic differ in the last fp16 bit of `y` — 2^-11 relative, which
/// is what 2e-3 covers. A wrong expert, a wrong permutation or a wrong weight is O(1) relative and cannot hide here.
/// The fp16 copy of the same sum is checked at the same relative tolerance (an ulp count is the wrong measure near
/// zero, where fp16's resolution is 6e-8).
fn both(what: &str, got32: &[f32], got16: &[f32], want: &[f32]) {
    close(&format!("{what} (fp32)"), got32, want, 2e-3);
    close(&format!("{what} (fp16)"), got16, want, 2e-3);
}

fn close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    let mut worst = (0f32, 0usize);
    for i in 0..got.len() {
        let d = (got[i] - want[i]).abs() / want[i].abs().max(1.0);
        if d > worst.0 {
            worst = (d, i);
        }
    }
    assert!(
        worst.0 <= tol,
        "{what}: element {} is {} vs {}, relative {:.3e} > {tol:.1e}",
        worst.1,
        got[worst.1],
        want[worst.1],
        worst.0
    );
}

// ================================================================================================================================

/// The whole routed block at decode depth, on four experts of four different `(w1, w2, w3)` widths: the f16 output and
/// the fp32 output both equal the per-token loop's.
#[test]
fn gpu_moe_decode_matches_the_per_token_loop_on_mixed_k_experts() {
    gpu!();
    let st = Stream::new().unwrap();
    let bank = owned_bank(&bank_parts());
    for rows in [1usize, 8] {
        let (got32, got16, want, stats) = run(&bank, rows, 0x1234 + rows as u64, &st);
        both(&format!("decode {rows} rows"), &got32, &got16, &want);
        // one row names exactly top-k experts; eight name all four
        let want_active = if rows == 1 { TOPK } else { E };
        assert_eq!(stats.active, want_active, "rows {rows}: {stats:?}");
        assert_eq!(stats.launches, 4 * stats.active + 3);
        assert_eq!(stats.slots, rows * TOPK);
    }
}

/// The same at a full prefill chunk (128 rows): the segment views, the permutation and the scatter all still hold, and
/// an expert's segment is its own rows only.
#[test]
fn gpu_moe_prefill_128_rows_matches_the_per_token_loop() {
    gpu!();
    let st = Stream::new().unwrap();
    let bank = owned_bank(&bank_parts());
    let (got32, got16, want, stats) = run(&bank, 128, 0xfeed, &st);
    both("prefill 128 rows", &got32, &got16, &want);
    assert_eq!(stats.active, E);
    assert_eq!(stats.launches, 4 * E + 3);
}

/// The same weights as views into one slab (what a streamed expert is) give the same bits as the resident bank: the
/// segment addressing is a pointer offset, not an allocation, and a lease-free view decodes identically.
#[test]
fn gpu_moe_streamed_slab_views_match_the_resident_bank() {
    gpu!();
    let st = Stream::new().unwrap();
    let parts = bank_parts();
    let owned = owned_bank(&parts);
    let (_block, slabbed) = slab_bank(&parts);
    for rows in [1usize, 128] {
        let (a, a16, _, _) = run(&owned, rows, 0xabc + rows as u64, &st);
        let (b, b16, _, _) = run(&slabbed, rows, 0xabc + rows as u64, &st);
        close(&format!("slab vs owned, {rows} rows"), &b, &a, 0.0);
        both(&format!("slab vs owned, {rows} rows"), &b, &b16, &a);
        let _ = a16;
    }
}

/// A selection that names one expert from every slot and one that names a single expert: the degenerate groupings the
/// histogram and the segment loop have to survive.
#[test]
fn gpu_moe_degenerate_selections() {
    gpu!();
    let st = Stream::new().unwrap();
    let bank = owned_bank(&bank_parts());
    let rows = 5usize;
    let mut r = Rng(7);
    let x: Vec<f32> = (0..rows * H).map(|_| normal(&mut r) as f32).collect();
    let shared: Vec<f32> = (0..rows * H).map(|_| normal(&mut r) as f32).collect();

    for (tag, picks) in [
        ("all one expert", vec![2i32; rows * TOPK]),
        ("each slot the next expert", (0..rows * TOPK).map(|i| (i % E) as i32).collect::<Vec<_>>()),
    ] {
        let wts: Vec<f32> = (0..rows * TOPK).map(|i| (0.1 + 0.01 * i as f32)).collect();
        let xb = DevBuf::from_slice(&f16_bytes(&x)).unwrap();
        let mut sharedb = DevBuf::from_slice(&f16_bytes(&shared)).unwrap();
        let mut outb = DevBuf::zeroed(rows * H * 4).unwrap();
        let picksb = DevBuf::from_slice(&picks).unwrap();
        let wtsb = DevBuf::from_slice(&f16_bytes(&wts)).unwrap();
        let mut moe = MoeScratch::new(rows, TOPK, H, I, E).unwrap();
        let stats = moe_experts(
            &mut moe,
            |e| Ok(MoeGroups::Resident(&bank[e])),
            rows,
            &xb,
            &mut sharedb,
            &picksb,
            &wtsb,
            &mut outb,
            LIMIT,
            &st,
        )
        .unwrap();
        let want = host_reference(&bank, &x, &picks, &wts, rows, &unf16(&f16_bytes(&shared)), &st);
        both(tag, &dl32(&outb, &st), &dl16(&sharedb, &st), &want);
        if tag == "all one expert" {
            assert_eq!((stats.active, stats.launches), (1, 7), "{stats:?}");
        }
    }
}
/// An arena sized for a full chunk, driven by a one-row call: the grouping must cover `rows * topk` slots and no more.
/// (A block that grouped the arena's depth would fold 127 rows of stale picks into expert 0 — which is what this pins.)
#[test]
fn gpu_moe_a_short_call_groups_only_its_own_rows() {
    gpu!();
    let st = Stream::new().unwrap();
    let bank = owned_bank(&bank_parts());
    let (cap, rows) = (128usize, 1usize);
    let (picks, wts) = selection(cap, 0x5151);
    let mut r = Rng(0x2222);
    let x: Vec<f32> = (0..cap * H).map(|_| normal(&mut r) as f32).collect();
    let shared: Vec<f32> = (0..cap * H).map(|_| normal(&mut r) as f32).collect();

    let xb = DevBuf::from_slice(&f16_bytes(&x)).unwrap();
    let mut sharedb = DevBuf::from_slice(&f16_bytes(&shared)).unwrap();
    let mut outb = DevBuf::zeroed(cap * H * 4).unwrap();
    let picksb = DevBuf::from_slice(&picks).unwrap();
    let wtsb = DevBuf::from_slice(&f16_bytes(&wts)).unwrap();
    let mut moe = MoeScratch::new(cap, TOPK, H, I, E).unwrap();

    let stats = moe_experts(
        &mut moe,
        |e| Ok(MoeGroups::Resident(&bank[e])),
        rows,
        &xb,
        &mut sharedb,
        &picksb,
        &wtsb,
        &mut outb,
        LIMIT,
        &st,
    )
    .unwrap();
    assert_eq!(stats.slots, rows * TOPK);
    assert_eq!(stats.active, TOPK);
    assert_eq!(stats.launches, 4 * TOPK + 3);
    let want = host_reference(&bank, &x, &picks, &wts, rows, &unf16(&f16_bytes(&shared)), &st);
    let n = rows * H;
    both("short call", &dl32(&outb, &st)[..n], &dl16(&sharedb, &st)[..n], &want[..n]);
}

// ------------------------------------------------------------------ the stable packing (the tensor-parallel y-exchange's contract)

/// The grouped block's packed order must be ASCENDING SLOT within every expert's segment — `group_k`'s documented
/// contract, which its `atomicAdd` placement cannot guarantee (arrival order). The single-rank result is
/// packing-invariant (the scatter sums in `k` order), but the TP y-exchange swaps `y` BY PACKED POSITION: two
/// ranks holding different permutations of the same routing exchange bytes the receiver scatters as the wrong
/// slots — the first live two-rank gate's layer-3 three-way divergence and, one divergence later, a collective
/// hang. Pure host test: no device needed.
#[test]
fn the_stable_pack_is_ascending_slot_within_each_expert() {
    use gb_cuda::dsv4_moe::pack_stable_host;
    // 8 slots over 4 experts, collision-heavy: expert 2 owns slots 1, 4, 6 (descending interleaved), expert 0 owns 0, 5
    let picks: Vec<i32> = vec![0, 2, -1, 3, 2, 0, 2, 3];
    let wts: Vec<u16> = (0..8).map(|i| 0x3000 + i as u16).collect();
    let counts = [2usize, 0, 3, 2]; // expert 1 unrouted, one pruned slot (-1) not counted, 7 placed of 8
    let mut offsets = vec![0i32];
    for (e, &c) in counts.iter().enumerate() {
        offsets.push(offsets[e] + c as i32);
    }
    let (perm, pinv, wpack) = pack_stable_host(&offsets, &picks, &wts);
    // the segments, in expert order, are exactly the expert's slots ascending
    let seg = |e: usize| (offsets[e] as usize..offsets[e + 1] as usize).map(|p| perm[p]).collect::<Vec<i32>>();
    assert_eq!(seg(0), vec![0i32, 5], "expert 0's segment is its slots ascending");
    assert_eq!(seg(1), Vec::<i32>::new(), "an unrouted expert has an empty segment");
    assert_eq!(seg(2), vec![1i32, 4, 6], "expert 2's segment is its slots ascending — the race cannot say this");
    assert_eq!(seg(3), vec![3i32, 7]);
    // unplaced tail beyond the packed total stays the kernel's padding
    assert_eq!(perm[7], -1);
    // pinv is perm's inverse, pruned slots unplaced
    for (i, &e) in picks.iter().enumerate() {
        match e {
            -1 => assert_eq!(pinv[i], -1, "a pruned slot is unplaced"),
            _ => assert_eq!(perm[pinv[i] as usize], i as i32, "pinv inverts perm at slot {i}"),
        }
    }
    // the packed weight rides its slot: wpack[p] == wts[perm[p]]
    for p in 0..8 {
        if perm[p] >= 0 {
            assert_eq!(wpack[p], wts[perm[p] as usize]);
        }
    }
    // a pure function of the picks: same inputs, same bytes (the two ranks' contract)
    assert_eq!(pack_stable_host(&offsets, &picks, &wts), (perm.clone(), pinv.clone(), wpack.clone()));
}

/// `MoeScratch::pack_stable` lands the host's stable permutation in the scratch the kernels read: after it, the
/// device's `perm`/`pinv`/`wpack` are exactly `pack_stable_host`'s, whatever the racy grouping produced first.
/// Device-gated, like the rest of this file.
#[test]
fn pack_stable_lands_the_stable_permutation_in_the_scratch() {
    gpu!();
    use gb_cuda::dsv4_moe::pack_stable_host;
    let (rows, topk) = (8usize, TOPK);
    let slots = rows * topk;
    // collision-heavy: every row picks the same two experts, so segments are long and arrival order is free to differ
    let picks: Vec<i32> = (0..slots).map(|i| if i % 2 == 0 { 1 } else { 2 }).collect();
    let wts: Vec<u16> = (0..slots).map(|i| 0x3000 + (i % 8) as u16).collect();
    let counts = [0usize, slots / 2, slots / 2, 0];
    let mut offsets = vec![0i32];
    for (e, &c) in counts.iter().enumerate() {
        offsets.push(offsets[e] + c as i32);
    }
    let mut moe = MoeScratch::new(rows, topk, H, I, E).unwrap();
    let picksb = DevBuf::from_slice(&picks).unwrap();
    let wtsb = DevBuf::from_slice(&wts).unwrap();
    let st = Stream::new().unwrap();
    moe.group_on(&picksb, &wtsb, rows, &st).unwrap();
    moe.pack_stable(&picksb, &wtsb, &offsets, rows, &st).unwrap();
    st.sync().unwrap();
    let (perm, pinv, wpack) = pack_stable_host(&offsets, &picks, &wts);
    assert_eq!(moe.perm_host().unwrap(), perm, "the scratch's perm is the stable one");
    assert_eq!(moe.pinv_host().unwrap(), pinv);
    assert_eq!(moe.wpack_host().unwrap(), wpack);
}

/// The same contract at the LIVE shape the TP y-exchange first failed on (arena 2048 rows deep, a 17-row call, topk 6
/// over 384 experts, every row pinning the same six experts on one rank's half): the race has the widest room here, so a
/// regrouping that regresses to arrival order is caught by the one test the group's bytes depend on.
#[test]
fn pack_stable_holds_at_the_tp_live_shape() {
    gpu!();
    use gb_cuda::dsv4_moe::pack_stable_host;
    let (arena_rows, rows, topk, h, inter, experts) = (2048usize, 17usize, 6usize, 128usize, 64usize, 384usize);
    let arena_slots = arena_rows * topk;
    let slots = rows * topk;
    // In Dsv41Model, `s.picks` and `s.wts` are preallocated at `MAX_ROWS * topk` with zeroed tail (`0` = expert 0):
    // `pack_stable` must slice to the call's `slots`, never walk the arena's trailing zeros as expert-0 picks.
    let mut picks: Vec<i32> = vec![0; arena_slots];
    let mut wts: Vec<u16> = vec![0; arena_slots];
    for i in 0..slots {
        picks[i] = ((i % 6) + 1) as i32;
        wts[i] = 0x3800 + (i % 4) as u16;
    }
    let picksb = DevBuf::from_slice(&picks).unwrap();
    let wtsb = DevBuf::from_slice(&wts).unwrap();
    let mut moe = MoeScratch::new(arena_rows, topk, h, inter, experts).unwrap();
    let st = Stream::new().unwrap();
    moe.group_on(&picksb, &wtsb, rows, &st).unwrap();
    st.sync().unwrap();
    let offsets = moe.offsets_host().unwrap();
    moe.pack_stable(&picksb, &wtsb, &offsets, rows, &st).unwrap();
    st.sync().unwrap();
    let (perm, pinv, wpack) = pack_stable_host(&offsets[..=experts], &picks[..slots], &wts[..slots]);
    assert_eq!(&moe.perm_host().unwrap()[..slots], &perm[..slots], "the live positions are the stable pack's");
    assert_eq!(&moe.pinv_host().unwrap()[..slots], &pinv[..slots], "the live pinv is the stable pack's");
    assert_eq!(&moe.wpack_host().unwrap()[..slots], &wpack[..slots], "the live weights are the stable pack's");
}

#[test]
fn moe_experts_resident_tp_and_interleave_o_halves_are_bit_identical_to_single_rank() {
    gpu!();
    use gb_cuda::dsv4_moe::{
        build_expert_descs, build_expert_descs_sparse, interleave_o_halves, moe_experts_resident, moe_experts_resident_tp,
    };
    let st = Stream::new().unwrap();
    let (h, inter, experts, topk) = (256usize, 128usize, 4usize, 6usize);
    let mut r = Rng(0x9988_7766);
    let raw: Vec<[Parts; 3]> = (0..experts)
        .map(|e| {
            let b1 = (e % 4) + 1;
            let b2 = ((e + 1) % 4) + 1;
            let b3 = ((e + 2) % 4) + 1;
            [
                parts(&mut r, h, inter, b1, Codebook::Mul1),
                parts(&mut r, inter, h, b2, Codebook::Mul1),
                parts(&mut r, h, inter, b3, Codebook::Mul1),
            ]
        })
        .collect();
    let bank = owned_bank(&raw);
    let refs_all: Vec<&[Exl3Weights]> = bank.iter().map(|g| &g[..]).collect();
    let desc_all = build_expert_descs(&refs_all, h, inter).unwrap().unwrap();
    let opts_r0: Vec<Option<&[Exl3Weights]>> = (0..experts).map(|e| if e < 2 { Some(&bank[e][..]) } else { None }).collect();
    let opts_r1: Vec<Option<&[Exl3Weights]>> = (0..experts).map(|e| if e >= 2 { Some(&bank[e][..]) } else { None }).collect();
    let desc_r0 = build_expert_descs_sparse(&opts_r0, h, inter).unwrap().unwrap();
    let desc_r1 = build_expert_descs_sparse(&opts_r1, h, inter).unwrap().unwrap();

    for rows in [1usize, 4, 8, 17] {
        let slots = rows * topk;
        let x: Vec<f32> = (0..rows * h).map(|_| normal(&mut r) as f32).collect();
        let shared: Vec<f32> = (0..rows * h).map(|_| normal(&mut r) as f32).collect();
        let picks: Vec<i32> = (0..slots).map(|i| if i % 11 == 10 { -1 } else { ((i * 3 + 1) % experts) as i32 }).collect();
        let wts: Vec<f32> = (0..slots).map(|i| 0.1 + 0.02 * (i % 7) as f32).collect();

        let xb = DevBuf::from_slice(&f16_bytes(&x)).unwrap();
        let picksb = DevBuf::from_slice(&picks).unwrap();
        let wtsb = DevBuf::from_slice(&f16_bytes(&wts)).unwrap();

        let mut sh_single = DevBuf::from_slice(&f16_bytes(&shared)).unwrap();
        let mut out_single = DevBuf::zeroed(rows * h * 4).unwrap();
        let mut moe_s = MoeScratch::new(32, topk, h, inter, experts).unwrap();
        moe_experts_resident(&mut moe_s, &desc_all, rows, &xb, &mut sh_single, &picksb, &wtsb, &mut out_single, LIMIT, &st)
            .unwrap();

        let mut sh_r0 = DevBuf::from_slice(&f16_bytes(&shared)).unwrap();
        let mut sh_r1 = DevBuf::from_slice(&f16_bytes(&shared)).unwrap();
        let mut out_r0 = DevBuf::zeroed(rows * h * 4).unwrap();
        let mut out_r1 = DevBuf::zeroed(rows * h * 4).unwrap();
        let mut moe_0 = MoeScratch::new(32, topk, h, inter, experts).unwrap();
        let mut moe_1 = MoeScratch::new(32, topk, h, inter, experts).unwrap();
        let mut y_peer_0 = DevBuf::zeroed(32 * topk * h * 2).unwrap();
        let mut y_peer_1 = DevBuf::zeroed(32 * topk * h * 2).unwrap();

        // Simulate rank 1's pre-pass first so we have rank 1's `y` ready for rank 0's swap, and vice versa.
        let mut r1_y_bytes = vec![0u8; slots * h * 2];
        let mut r0_y_bytes = vec![0u8; slots * h * 2];
        moe_experts_resident_tp(
            &mut moe_1,
            &desc_r1,
            2..4,
            |y1, _yp1, bytes, s| {
                s.sync()?;
                y1.download_range(0, &mut r1_y_bytes[..bytes])
            },
            &mut y_peer_1,
            rows,
            false,
            &xb,
            &mut sh_r1,
            &picksb,
            &wtsb,
            &mut out_r1,
            LIMIT,
            &st,
        )
        .unwrap();
        moe_experts_resident_tp(
            &mut moe_0,
            &desc_r0,
            0..2,
            |y0, yp0, bytes, s| {
                s.sync()?;
                y0.download_range(0, &mut r0_y_bytes[..bytes])?;
                unsafe { s.memcpy_h2d_async(yp0.ptr(), r1_y_bytes.as_ptr(), bytes) }
            },
            &mut y_peer_0,
            rows,
            false,
            &xb,
            &mut sh_r0,
            &picksb,
            &wtsb,
            &mut out_r0,
            LIMIT,
            &st,
        )
        .unwrap();
        // Re-run rank 1 with rank 0's `y` bytes so both ranks are verified.
        let mut sh_r1 = DevBuf::from_slice(&f16_bytes(&shared)).unwrap();
        moe_experts_resident_tp(
            &mut moe_1,
            &desc_r1,
            2..4,
            |_y1, yp1, bytes, s| unsafe { s.memcpy_h2d_async(yp1.ptr(), r0_y_bytes.as_ptr(), bytes) },
            &mut y_peer_1,
            rows,
            false,
            &xb,
            &mut sh_r1,
            &picksb,
            &wtsb,
            &mut out_r1,
            LIMIT,
            &st,
        )
        .unwrap();
        st.sync().unwrap();

        let single_f32: Vec<u32> = out_single.to_vec().unwrap();
        let r0_f32: Vec<u32> = out_r0.to_vec().unwrap();
        let r1_f32: Vec<u32> = out_r1.to_vec().unwrap();
        let single_f16: Vec<u16> = sh_single.to_vec().unwrap();
        let r0_f16: Vec<u16> = sh_r0.to_vec().unwrap();
        let r1_f16: Vec<u16> = sh_r1.to_vec().unwrap();
        assert_eq!(r0_f32, single_f32, "rows={rows}: rank 0 f32 output must match single-rank bit for bit");
        assert_eq!(r1_f32, single_f32, "rows={rows}: rank 1 f32 output must match single-rank bit for bit");
        assert_eq!(r0_f16, single_f16, "rows={rows}: rank 0 f16 output must match single-rank bit for bit");
        assert_eq!(r1_f16, single_f16, "rows={rows}: rank 1 f16 output must match single-rank bit for bit");

        // Also verify `interleave_o_halves` at this row count.
        let half = 128usize;
        let left_h: Vec<u16> = (0..rows * half).map(|i| (i as u16).wrapping_mul(3)).collect();
        let right_h: Vec<u16> = (0..rows * half).map(|i| (i as u16).wrapping_mul(7).wrapping_add(1)).collect();
        let left_b = DevBuf::from_slice(&left_h).unwrap();
        let right_b = DevBuf::from_slice(&right_h).unwrap();
        let mut dst_b = DevBuf::zeroed(rows * 2 * half * 2).unwrap();
        interleave_o_halves(&left_b, &right_b, &mut dst_b, rows, half, &st).unwrap();
        st.sync().unwrap();
        let got: Vec<u16> = dst_b.to_vec().unwrap();
        for rr in 0..rows {
            assert_eq!(&got[rr * 2 * half..rr * 2 * half + half], &left_h[rr * half..(rr + 1) * half]);
            assert_eq!(&got[rr * 2 * half + half..(rr + 1) * 2 * half], &right_h[rr * half..(rr + 1) * half]);
        }
    }
}
