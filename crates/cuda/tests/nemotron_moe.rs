//! The Nemotron 3.5 Super VL latent-MoE block on the GPU (`csrc/nemotron_moe.cu`) against the fp32 oracle.
//!
//! Synthetic checks (any GPU): the decoded trellis is bitwise `gb_exl3::unpack`; the router's top-22 matches a CPU route on random
//! rows with no near-tie; a dense EXL3 projection matches `gb_exl3::forward`; the full block, with one bf16 expert among EXL3 experts,
//! matches a CPU reference; and a row's bits do not change with the number of rows in the call.
//!
//! Pack checks (need `REXL3_NEMO_OMNI_DIR`, the pack): against torch's layer-1 fixture (`testdata/nemotron_omni/moe/layer1_fixture.
//! safetensors`, written by `tools/gen_moe_fixture.py`): the router's indices and weights, fc1, one EXL3 expert, the whole block, the
//! decoded trellis digests, and the bf16 expert (layer 26, expert 111). The pack is read through its index (`scan_indexed`), the
//! engine's placement, and the fixture goes with the pack: the ORIGINAL pack's, or for a pack that holds the repaired tensors (the
//! repaired pack, whose layer-1 shared expert was re-encoded, and the Hub pack, which is the repaired pack re-sharded) its own,
//! `layer1_fixture_repaired.safetensors`, chosen by the pack's index. Tolerances are the same.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::exl3_synth::synth_group;
use common::{repo_root, Rng};
use gb_cuda::nemotron_moe::{MoeScratch, NemotronMoe, Proj, Table, Weights, EXPERTS, EXPERT_INTER, HIDDEN, LATENT, TOPK};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::{read_header, scan_indexed, Checkpoint, LoadedGroup};
use gb_exl3::{codebook, fnv1a, forward, unpack, Codebook};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

const SCALE: f32 = 5.0;
const SHARED_INTER: usize = 5376;

// ---------------------------------------------------------------------------------------------------------------------------------
// The oracle in fp32: moe_ref's route and latent_moe; EXL3 through gb_exl3::forward (f64, then f32).

/// max |got - want| / max |want|.
fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "length {} against {}", got.len(), want.len());
    let scale = want.iter().fold(0f64, |m, &v| m.max((v as f64).abs()));
    let diff = got.iter().zip(want).fold(0f64, |m, (&g, &w)| m.max(((g - w) as f64).abs()));
    diff / scale
}

/// bf16 bits as f32 (exact).
fn bf16_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()
}

/// moe_ref's `Router::logits` for one row: the sequential fp32 dot product with each expert's gate row.
fn logits_cpu(h: &[f32], gate: &[f32]) -> Vec<f32> {
    (0..EXPERTS).map(|j| h.iter().zip(&gate[j * HIDDEN..(j + 1) * HIDDEN]).map(|(a, b)| a * b).sum()).collect()
}

/// moe_ref's `route`: top-k of sigmoid + bias (ties to the lower index), weights = unbiased sigmoid over the selected set, times `scale`.
/// Also returns the gap between the 22nd and 23rd selection values (the boundary the selection depends on).
fn route_cpu(logits: &[f32], bias: &[f32], scale: f32) -> (Vec<usize>, Vec<f32>, f32) {
    let sig: Vec<f32> = logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    let sel: Vec<f32> = sig.iter().zip(bias).map(|(s, b)| s + b).collect();
    let mut order: Vec<usize> = (0..sel.len()).collect();
    order.sort_by(|&a, &b| sel[b].partial_cmp(&sel[a]).unwrap().then(a.cmp(&b)));
    let mut sorted = sel.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let gap = sorted[TOPK - 1] - sorted[TOPK];
    order.truncate(TOPK);
    let denom = order.iter().map(|&i| sig[i]).sum::<f32>() + 1e-20;
    let w = order.iter().map(|&i| sig[i] / denom * scale).collect();
    (order, w, gap)
}

/// An EXL3 projection on the host (the group and the codebook table): gb_exl3::forward, then f32.
fn exl3_cpu(g: &LoadedGroup, table: &[u16], x: &[f32], rows: usize) -> Vec<f32> {
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    forward(&x64, rows, &g.trellis, g.kt, g.nt, g.group.bits, table, &g.suh, &g.svh, g.bias.as_deref())
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

/// A bf16 projection on the host, HF layout `[n, k]`: y = x . W^T, summed in k order.
fn bf16_cpu(w: &[f32], k: usize, n: usize, x: &[f32], rows: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * n];
    for r in 0..rows {
        for j in 0..n {
            y[r * n + j] = x[r * k..(r + 1) * k].iter().zip(&w[j * k..(j + 1) * k]).map(|(a, b)| a * b).sum();
        }
    }
    y
}

fn relu2(v: &mut [f32]) {
    for x in v.iter_mut() {
        let r = x.max(0.0);
        *x = r * r;
    }
}

/// A routed expert on the host.
enum ExpertRef {
    Exl3(Box<LoadedGroup>, Box<LoadedGroup>),
    /// bf16 up `[2688, 1024]` and down `[1024, 2688]`, HF layout, as f32.
    Bf16(Vec<f32>, Vec<f32>),
}

/// The host copy of a synthetic layer, for the oracle.
struct Host {
    gate: Vec<f32>,
    bias: Vec<f32>,
    table: Vec<u16>,
    fc1: LoadedGroup,
    fc2: LoadedGroup,
    shared_up: LoadedGroup,
    shared_down: LoadedGroup,
    experts: BTreeMap<usize, ExpertRef>,
}

/// A synthetic layer: a random gate and bias, EXL3 dense projections, and the listed experts (the expert `bf16` is bf16, the rest EXL3).
/// Returns the block's device weights and the host copy.
fn synthetic(seed: u64, experts: &[usize], bf16: Option<usize>) -> (Weights, Host) {
    let mut r = Rng(seed);
    let gate_bits: Vec<u16> = (0..EXPERTS * HIDDEN).map(|_| (r.uniform(-0.05, 0.05).to_bits() >> 16) as u16).collect();
    let bias: Vec<f32> = r.vec(EXPERTS, -0.1, 0.1);
    let fc1 = synth_group(HIDDEN, LATENT, 4, Codebook::Mul1, seed + 1);
    let fc2 = synth_group(LATENT, HIDDEN, 4, Codebook::Mul1, seed + 2);
    let shared_up = synth_group(HIDDEN, SHARED_INTER, 4, Codebook::Mul1, seed + 3);
    let shared_down = synth_group(SHARED_INTER, HIDDEN, 4, Codebook::Mul1, seed + 4);
    let mut up: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    let mut down: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    let mut map = BTreeMap::new();
    for &e in experts {
        if Some(e) == bf16 {
            let uw: Vec<u16> = (0..EXPERT_INTER * LATENT).map(|_| (r.uniform(-0.05, 0.05).to_bits() >> 16) as u16).collect();
            let dw: Vec<u16> = (0..LATENT * EXPERT_INTER).map(|_| (r.uniform(-0.05, 0.05).to_bits() >> 16) as u16).collect();
            up[e] = Proj::bf16(LATENT, EXPERT_INTER, &uw).unwrap();
            down[e] = Proj::bf16(EXPERT_INTER, LATENT, &dw).unwrap();
            map.insert(e, ExpertRef::Bf16(bf16_f32(&uw), bf16_f32(&dw)));
        } else {
            let ug = synth_group(LATENT, EXPERT_INTER, 4, Codebook::Mul1, seed + 1000 + 2 * e as u64);
            let dg = synth_group(EXPERT_INTER, LATENT, 4, Codebook::Mul1, seed + 1001 + 2 * e as u64);
            up[e] = Proj::exl3_from_group(&ug).unwrap();
            down[e] = Proj::exl3_from_group(&dg).unwrap();
            map.insert(e, ExpertRef::Exl3(Box::new(ug), Box::new(dg)));
        }
    }
    let weights = Weights {
        gate: DevBuf::from_slice(&gate_bits).unwrap(),
        gate_dtype: gb_cuda::nemotron_moe::GateDtype::Bf16,
        router_bias: DevBuf::from_slice(&bias).unwrap(),
        scale: SCALE,
        fc1: Proj::exl3_from_group(&fc1).unwrap(),
        fc2: Proj::exl3_from_group(&fc2).unwrap(),
        shared_up: Proj::exl3_from_group(&shared_up).unwrap(),
        shared_down: Proj::exl3_from_group(&shared_down).unwrap(),
        up,
        down,
    };
    let host = Host {
        gate: bf16_f32(&gate_bits),
        bias,
        table: codebook(Codebook::Mul1),
        fc1,
        fc2,
        shared_up,
        shared_down,
        experts: map,
    };
    (weights, host)
}

/// The block on the host, as moe_ref's `latent_moe` computes it, except that the routed sum is taken in slot order (as the GPU does).
/// `picks` and `wts` are the routes of the `rows` rows.
fn block_cpu(s: &Host, h: &[f32], rows: usize, picks: &[Vec<usize>], wts: &[Vec<f32>]) -> Vec<f32> {
    let x = exl3_cpu(&s.fc1, &s.table, h, rows);
    let mut y_pair = vec![vec![0f32; LATENT]; rows * TOPK];
    let used: BTreeSet<usize> = picks.iter().flatten().copied().collect();
    for e in used {
        let pairs: Vec<(usize, usize)> =
            (0..rows).flat_map(|t| (0..TOPK).map(move |sl| (t, sl))).filter(|&(t, sl)| picks[t][sl] == e).collect();
        let mut batch = Vec::with_capacity(pairs.len() * LATENT);
        for &(t, _) in &pairs {
            batch.extend_from_slice(&x[t * LATENT..(t + 1) * LATENT]);
        }
        let y = match &s.experts[&e] {
            ExpertRef::Exl3(ug, dg) => {
                let mut a = exl3_cpu(ug, &s.table, &batch, pairs.len());
                relu2(&mut a);
                exl3_cpu(dg, &s.table, &a, pairs.len())
            }
            ExpertRef::Bf16(uw, dw) => {
                let mut a = bf16_cpu(uw, LATENT, EXPERT_INTER, &batch, pairs.len());
                relu2(&mut a);
                bf16_cpu(dw, EXPERT_INTER, LATENT, &a, pairs.len())
            }
        };
        for (i, &(t, sl)) in pairs.iter().enumerate() {
            y_pair[t * TOPK + sl].copy_from_slice(&y[i * LATENT..(i + 1) * LATENT]);
        }
    }
    let mut routed = vec![0f32; rows * LATENT];
    for t in 0..rows {
        for sl in 0..TOPK {
            let w = wts[t][sl];
            for (d, &v) in routed[t * LATENT..(t + 1) * LATENT].iter_mut().zip(&y_pair[t * TOPK + sl]) {
                *d += w * v;
            }
        }
    }
    let routed_out = exl3_cpu(&s.fc2, &s.table, &routed, rows);
    let mut sa = exl3_cpu(&s.shared_up, &s.table, h, rows);
    relu2(&mut sa);
    let shared_out = exl3_cpu(&s.shared_down, &s.table, &sa, rows);
    routed_out.iter().zip(&shared_out).map(|(a, b)| a + b).collect()
}

fn run_block(moe: &mut NemotronMoe, h: &[f32], rows: usize, st: &Stream) -> Vec<f32> {
    let hd = DevBuf::from_slice(h).unwrap();
    let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
    moe.forward(&hd, rows, &mut out, st).unwrap();
    st.sync().unwrap();
    out.to_vec::<f32>().unwrap()
}

/// `base` (`base_rows` rows of 4096) tiled to `rows` rows: row r is base row r % base_rows.
fn tile(base: &[f32], base_rows: usize, rows: usize) -> Vec<f32> {
    (0..rows).flat_map(|r| base[(r % base_rows) * HIDDEN..(r % base_rows + 1) * HIDDEN].iter().copied()).collect()
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Synthetic checks (any GPU)

/// The GPU's decode of every projection is bitwise gb_exl3::unpack's.
#[test]
fn decode_is_bitwise_gb_exl3_unpack() {
    gpu!();
    let st = Stream::new().unwrap();
    let (w, s) = synthetic(7, &[0], None);
    let moe = NemotronMoe::new(w, 4).unwrap();
    let check = |name: &str, got: Vec<u16>, g: &LoadedGroup| {
        let want = unpack(&g.trellis, g.kt, g.nt, g.group.bits, &s.table);
        eprintln!("decode {name:<12} fnv {:016x}", fnv1a(got.iter().flat_map(|v| v.to_le_bytes())));
        assert_eq!(got, want, "{name}: the GPU decode differs from gb_exl3::unpack");
    };
    check("fc1", moe.decode(Table::Dense(0), &st).unwrap(), &s.fc1);
    check("fc2", moe.decode(Table::Dense(1), &st).unwrap(), &s.fc2);
    check("shared up", moe.decode(Table::Dense(2), &st).unwrap(), &s.shared_up);
    check("shared down", moe.decode(Table::Dense(3), &st).unwrap(), &s.shared_down);
    let ExpertRef::Exl3(ug, dg) = &s.experts[&0] else { unreachable!() };
    check("expert up", moe.decode(Table::Up(0), &st).unwrap(), ug);
    check("expert down", moe.decode(Table::Down(0), &st).unwrap(), dg);
}

/// The router on random rows: the GPU's top-22 indices and weights match the CPU route on every row with no near-tie, and the logits
/// agree with the CPU's dot products to fp32 summation-order noise.
#[test]
fn router_matches_cpu_on_random_rows() {
    gpu!();
    let st = Stream::new().unwrap();
    let (w, s) = synthetic(11, &[], None);
    let mut moe = NemotronMoe::new(w, 64).unwrap();
    let mut r = Rng(0xA11CE);
    let (mut checked, mut skipped) = (0usize, 0usize);
    let mut max_logit = 0f32;
    let mut max_wrel = 0f64;
    for &rows in &[1usize, 4, 16, 64] {
        let h = r.vec(rows * HIDDEN, -1.0, 1.0);
        let hd = DevBuf::from_slice(&h).unwrap();
        moe.route(&hd, rows, &st).unwrap();
        let logits_gpu = moe.download_logits(rows, &st).unwrap();
        let (pick, wts) = moe.download_route(rows, &st).unwrap();
        for t in 0..rows {
            let lc = logits_cpu(&h[t * HIDDEN..(t + 1) * HIDDEN], &s.gate);
            for e in 0..EXPERTS {
                max_logit = max_logit.max((lc[e] - logits_gpu[t * EXPERTS + e]).abs());
            }
            let (idx, w, gap) = route_cpu(&lc, &s.bias, SCALE);
            if gap <= 1e-5 {
                skipped += 1;
                continue;
            }
            let got: Vec<usize> = pick[t * TOPK..(t + 1) * TOPK].iter().map(|&v| v as usize).collect();
            assert_eq!(got, idx, "rows {rows}, row {t}: the GPU's top-22 differs from the CPU route");
            max_wrel = max_wrel.max(rel_err(&wts[t * TOPK..(t + 1) * TOPK], &w));
            checked += 1;
        }
    }
    eprintln!("router, random rows: {checked} checked, {skipped} skipped (boundary gap <= 1e-5); max |logit gpu - cpu| {max_logit:.3e}; weights max rel {max_wrel:.3e}");
    assert!(checked >= 80, "too few rows with a clear boundary");
    assert!(max_wrel <= 1e-5, "weights differ from the CPU route by {max_wrel:e}");

    // Row invariance of the router: row 0 alone (the decode path, one row per block) and as row 0 of 4- and 64-row calls give the same bits.
    let h = r.vec(64 * HIDDEN, -1.0, 1.0);
    // The rows `start .. start + rows` of `h` as one call.
    let mut route_of = |start: usize, rows: usize| -> (Vec<f32>, Vec<i32>, Vec<f32>) {
        let hd = DevBuf::from_slice(&h[start * HIDDEN..(start + rows) * HIDDEN]).unwrap();
        moe.route(&hd, rows, &st).unwrap();
        let logits = moe.download_logits(rows, &st).unwrap();
        let (pick, wts) = moe.download_route(rows, &st).unwrap();
        (logits, pick, wts)
    };
    let (l64, p64, w64) = route_of(0, 64);
    let (l4, p4, w4) = route_of(0, 4);
    assert!(
        l4[..EXPERTS * 4] == l64[..EXPERTS * 4] && p4[..TOPK * 4] == p64[..TOPK * 4] && w4[..TOPK * 4] == w64[..TOPK * 4],
        "router rows 0..4 change bits between the 4- and 64-row calls"
    );
    // Every row of the 64-row call alone (the decode path) gives the same bits.
    for row in 0..64 {
        let (l1, p1, w1) = route_of(row, 1);
        assert!(l1[..] == l64[row * EXPERTS..(row + 1) * EXPERTS], "router logits of row {row} change bits with the row count");
        assert!(p1[..] == p64[row * TOPK..(row + 1) * TOPK], "router picks of row {row} change with the row count");
        assert!(w1[..] == w64[row * TOPK..(row + 1) * TOPK], "router weights of row {row} change bits with the row count");
    }
    eprintln!("router row invariance: all 64 rows identical at 1 row and in the 4- and 64-row calls (logits, picks, weights)");
}

/// fc1 on the GPU matches gb_exl3::forward on the same projection, and a row has the same bits in a 1-, 4-, 16- and 64-row call.
#[test]
fn latent_projection_matches_forward_and_is_row_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let (w, s) = synthetic(23, &[], None);
    let mut moe = NemotronMoe::new(w, 64).unwrap();
    let mut r = Rng(5);
    let base = r.vec(4 * HIDDEN, -1.0, 1.0);
    let want = exl3_cpu(&s.fc1, &s.table, &base, 4);
    let run = |moe: &mut NemotronMoe, rows: usize| -> Vec<f32> {
        let hd = DevBuf::from_slice(&tile(&base, 4, rows)).unwrap();
        let mut out = DevBuf::zeroed(rows * LATENT * 4).unwrap();
        moe.latent_in(&hd, rows, &mut out, &st).unwrap();
        st.sync().unwrap();
        out.to_vec::<f32>().unwrap()
    };
    let g4 = run(&mut moe, 4);
    let g16 = run(&mut moe, 16);
    let g64 = run(&mut moe, 64);
    let g1 = run(&mut moe, 1);
    let e = rel_err(&g4, &want);
    eprintln!("fc1 (synthetic, 4 rows) vs gb_exl3::forward: max rel {e:.3e}");
    assert!(e <= 1e-4, "fc1 is off by {e:e}");
    assert!(g4 == g16[..4 * LATENT] && g4 == g64[..4 * LATENT], "a row changes bits with the number of rows");
    assert!(g1 == g4[..LATENT], "the 1-row call differs from the 4-row call");
}

/// The full block, with a bf16 expert among EXL3 experts, matches the CPU reference; rows are bitwise the same in 4-, 16- and 64-row calls
/// (routing included).
#[test]
fn block_matches_cpu_with_a_bf16_expert_and_is_row_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut r = Rng(31);
    let base = r.vec(4 * HIDDEN, -1.0, 1.0);
    // The routes on these rows, from the layer's gate (the gate and bias come first in the seeded stream, so a layer with no experts has
    // the same ones), so that the loaded experts are the ones the block uses.
    let (_, probe) = synthetic(41, &[], None);
    let mut picks: Vec<Vec<usize>> = Vec::new();
    let mut wts: Vec<Vec<f32>> = Vec::new();
    for t in 0..4 {
        let (idx, w, gap) = route_cpu(&logits_cpu(&base[t * HIDDEN..(t + 1) * HIDDEN], &probe.gate), &probe.bias, SCALE);
        assert!(gap > 1e-5, "row {t} has a near-tie at the boundary: pick other rows");
        picks.push(idx);
        wts.push(w);
    }
    let used: Vec<usize> = picks.iter().flatten().copied().collect::<BTreeSet<usize>>().into_iter().collect();
    let bf16 = used[used.len() / 2];
    eprintln!("block: {} distinct experts for 4 rows; expert {bf16} is bf16", used.len());
    let (w, s) = synthetic(41, &used, Some(bf16));
    let mut moe = NemotronMoe::new(w, 64).unwrap();

    let got = run_block(&mut moe, &base, 4, &st);
    let want = block_cpu(&s, &base, 4, &picks, &wts);
    let e = rel_err(&got, &want);
    eprintln!("block (synthetic, 4 rows, bf16 expert {bf16}) vs CPU reference: max rel {e:.3e}");
    assert!(e <= 1e-3, "block is off by {e:e}");

    let g16 = run_block(&mut moe, &tile(&base, 4, 16), 16, &st);
    let g64 = run_block(&mut moe, &tile(&base, 4, 64), 64, &st);
    let g1 = run_block(&mut moe, &base[..HIDDEN], 1, &st);
    assert!(g1 == got[..HIDDEN], "the 1-row (decode) call differs from row 0 of the 4-row call");
    // Every row of the 16- and 64-row calls against its base row (rows past 16 sit in the second and later row chunks of a block).
    for (rows, g) in [(16usize, &g16), (64, &g64)] {
        for r in 0..rows {
            assert!(
                g[r * HIDDEN..(r + 1) * HIDDEN] == got[(r % 4) * HIDDEN..(r % 4 + 1) * HIDDEN],
                "{rows}-row call, row {r} changes bits"
            );
        }
    }
    eprintln!("block row invariance: 1 vs 4 vs 16 vs 64 rows identical");
}

/// Two blocks built from their tables only, sharing one scratch and run in alternation, give the same bits as two blocks that each
/// own a scratch. This is how a 40-layer model runs: one scratch for every MoE layer.
#[test]
fn a_shared_scratch_gives_the_same_bits_as_a_scratch_of_its_own() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut r = Rng(77);
    let h = r.vec(8 * HIDDEN, -1.0, 1.0);
    let used: Vec<usize> = (0..EXPERTS).step_by(37).collect();
    let bf16 = used[3];
    let (wa1, _) = synthetic(61, &used, Some(bf16));
    let (wb1, _) = synthetic(62, &used, None);
    let (wa2, _) = synthetic(61, &used, Some(bf16));
    let (wb2, _) = synthetic(62, &used, None);

    let mut own_a = NemotronMoe::new(wa1, 64).unwrap();
    let mut own_b = NemotronMoe::new(wb1, 64).unwrap();
    let ra = run_block(&mut own_a, &h, 8, &st);
    let rb = run_block(&mut own_b, &h, 8, &st);

    let ta = NemotronMoe::tables(wa2).unwrap();
    let tb = NemotronMoe::tables(wb2).unwrap();
    assert_eq!(ta.max_rows(), 0, "a tables-only block owns no scratch");
    let mut scratch = MoeScratch::new(64).unwrap();
    assert_eq!(scratch.max_rows(), 64);
    let hd = DevBuf::from_slice(&h).unwrap();
    let mut oa = DevBuf::zeroed(8 * HIDDEN * 4).unwrap();
    let mut ob = DevBuf::zeroed(8 * HIDDEN * 4).unwrap();
    for _ in 0..2 {
        ta.forward_in(&mut scratch, &hd, 8, &mut oa, &st).unwrap();
        tb.forward_in(&mut scratch, &hd, 8, &mut ob, &st).unwrap();
    }
    st.sync().unwrap();
    assert!(oa.to_vec::<f32>().unwrap() == ra, "block A on the shared scratch differs from its own-scratch run");
    assert!(ob.to_vec::<f32>().unwrap() == rb, "block B on the shared scratch differs from its own-scratch run");
    // The shared scratch is one allocation for every layer: about 100 MB at 64 rows, not 100 MB per layer.
    let mb = MoeScratch::bytes(64) as f64 / 1e6;
    eprintln!("shared scratch at 64 rows: {mb:.1} MB");
    assert!(mb > 50.0 && mb < 200.0, "unexpected scratch size {mb} MB");
}

/// The bf16 expert alone, on 4 rows, against the CPU bf16 math.
#[test]
fn bf16_expert_alone_matches_the_cpu() {
    gpu!();
    let st = Stream::new().unwrap();
    let (w, s) = synthetic(53, &[3], Some(3));
    let mut moe = NemotronMoe::new(w, 16).unwrap();
    let mut r = Rng(9);
    let x = r.vec(4 * LATENT, -1.0, 1.0);
    let ExpertRef::Bf16(uw, dw) = &s.experts[&3] else { unreachable!() };
    let mut a = bf16_cpu(uw, LATENT, EXPERT_INTER, &x, 4);
    relu2(&mut a);
    let want = bf16_cpu(dw, EXPERT_INTER, LATENT, &a, 4);
    let xd = DevBuf::from_slice(&x).unwrap();
    let mut out = DevBuf::zeroed(4 * LATENT * 4).unwrap();
    moe.expert(3, &xd, 4, &mut out, &st).unwrap();
    st.sync().unwrap();
    let e = rel_err(&out.to_vec::<f32>().unwrap(), &want);
    eprintln!("bf16 expert alone (4 rows) vs CPU: max rel {e:.3e}");
    assert!(e <= 1e-4, "bf16 expert is off by {e:e}");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// MixedK (v0.1.2 lane K, P1-5): projections at every integer width, mixed within a layer, and fp16 dense projections

/// A dense projection on the host: EXL3 (any width), or an fp16 weight `[n, k]` (HF layout) as its f32 values.
enum DenseRef {
    Exl3(LoadedGroup),
    F16(Vec<f32>),
}

/// The host copy of a MixedK synthetic layer.
struct MixHost {
    gate: Vec<f32>,
    table: Vec<u16>,
    /// fc1, fc2, shared up, shared down.
    dense: [DenseRef; 4],
    experts: BTreeMap<usize, ExpertRef>,
}

const DENSE_SHAPES: [(usize, usize); 4] = [(HIDDEN, LATENT), (LATENT, HIDDEN), (HIDDEN, SHARED_INTER), (SHARED_INTER, HIDDEN)];

/// A MixedK synthetic layer: the gate and bias of [`synthetic`] (the same seeded stream, so `synthetic(seed, &[], None)` probes its
/// routes), each listed expert's up and down at `up_bits(e)` and `down_bits(e)` (Mul1, 2 to 8 bits), and the dense projections at
/// `dense[i]` bits (0: fp16).
fn synthetic_mix(
    seed: u64,
    experts: &[usize],
    up_bits: impl Fn(usize) -> usize,
    down_bits: impl Fn(usize) -> usize,
    dense: [usize; 4],
) -> (Weights, MixHost) {
    let mut r = Rng(seed);
    let gate_bits: Vec<u16> = (0..EXPERTS * HIDDEN).map(|_| (r.uniform(-0.05, 0.05).to_bits() >> 16) as u16).collect();
    let bias: Vec<f32> = r.vec(EXPERTS, -0.1, 0.1);
    let mut dev = Vec::new();
    let mut host = Vec::new();
    for (i, &(k, n)) in DENSE_SHAPES.iter().enumerate() {
        if dense[i] == 0 {
            let bits: Vec<u16> = (0..k * n).map(|_| gb_exl3::f16::f64_to_f16(r.uniform(-0.05, 0.05) as f64)).collect();
            dev.push(Proj::f16(k, n, &bits).unwrap());
            host.push(DenseRef::F16(bits.iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect()));
        } else {
            let g = synth_group(k, n, dense[i], Codebook::Mul1, seed + 1 + i as u64);
            dev.push(Proj::exl3_from_group(&g).unwrap());
            host.push(DenseRef::Exl3(g));
        }
    }
    let mut up: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    let mut down: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    let mut map = BTreeMap::new();
    for &e in experts {
        let ug = synth_group(LATENT, EXPERT_INTER, up_bits(e), Codebook::Mul1, seed + 1000 + 2 * e as u64);
        let dg = synth_group(EXPERT_INTER, LATENT, down_bits(e), Codebook::Mul1, seed + 1001 + 2 * e as u64);
        up[e] = Proj::exl3_from_group(&ug).unwrap();
        down[e] = Proj::exl3_from_group(&dg).unwrap();
        map.insert(e, ExpertRef::Exl3(Box::new(ug), Box::new(dg)));
    }
    let mut d = dev.into_iter();
    let weights = Weights {
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
    };
    let host = MixHost {
        gate: bf16_f32(&gate_bits),
        table: codebook(Codebook::Mul1),
        dense: host.try_into().ok().unwrap(),
        experts: map,
    };
    (weights, host)
}

fn dense_cpu(d: &DenseRef, table: &[u16], x: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    match d {
        DenseRef::Exl3(g) => exl3_cpu(g, table, x, rows),
        DenseRef::F16(w) => bf16_cpu(w, k, n, x, rows),
    }
}

/// [`block_cpu`] for a MixedK layer.
fn block_cpu_mix(s: &MixHost, h: &[f32], rows: usize, picks: &[Vec<usize>], wts: &[Vec<f32>]) -> Vec<f32> {
    let x = dense_cpu(&s.dense[0], &s.table, h, rows, HIDDEN, LATENT);
    let mut y_pair = vec![vec![0f32; LATENT]; rows * TOPK];
    for e in picks.iter().flatten().copied().collect::<BTreeSet<usize>>() {
        let pairs: Vec<(usize, usize)> =
            (0..rows).flat_map(|t| (0..TOPK).map(move |sl| (t, sl))).filter(|&(t, sl)| picks[t][sl] == e).collect();
        let batch: Vec<f32> = pairs.iter().flat_map(|&(t, _)| x[t * LATENT..(t + 1) * LATENT].iter().copied()).collect();
        let ExpertRef::Exl3(ug, dg) = &s.experts[&e] else { unreachable!() };
        let mut a = exl3_cpu(ug, &s.table, &batch, pairs.len());
        relu2(&mut a);
        let y = exl3_cpu(dg, &s.table, &a, pairs.len());
        for (i, &(t, sl)) in pairs.iter().enumerate() {
            y_pair[t * TOPK + sl].copy_from_slice(&y[i * LATENT..(i + 1) * LATENT]);
        }
    }
    let mut routed = vec![0f32; rows * LATENT];
    for t in 0..rows {
        for sl in 0..TOPK {
            for (d, &v) in routed[t * LATENT..(t + 1) * LATENT].iter_mut().zip(&y_pair[t * TOPK + sl]) {
                *d += wts[t][sl] * v;
            }
        }
    }
    let routed_out = dense_cpu(&s.dense[1], &s.table, &routed, rows, LATENT, HIDDEN);
    let mut sa = dense_cpu(&s.dense[2], &s.table, h, rows, HIDDEN, SHARED_INTER);
    relu2(&mut sa);
    let shared_out = dense_cpu(&s.dense[3], &s.table, &sa, rows, SHARED_INTER, HIDDEN);
    routed_out.iter().zip(&shared_out).map(|(a, b)| a + b).collect()
}

/// The GPU's decode of a projection at every integer width (2 to 8 bits, routed and dense) is bitwise gb_exl3::unpack's.
#[test]
fn decode_is_bitwise_gb_exl3_unpack_at_every_width() {
    gpu!();
    let st = Stream::new().unwrap();
    for k in 2..=8usize {
        let (w, s) = synthetic_mix(70 + k as u64, &[0], |_| k, |_| k, [k; 4]);
        let moe = NemotronMoe::new(w, 4).unwrap();
        for (i, t) in [Table::Dense(0), Table::Dense(1), Table::Dense(2), Table::Dense(3)].into_iter().enumerate() {
            let DenseRef::Exl3(g) = &s.dense[i] else { unreachable!() };
            assert!(moe.decode(t, &st).unwrap() == unpack(&g.trellis, g.kt, g.nt, g.group.bits, &s.table), "{k} bits, dense {i}");
        }
        let ExpertRef::Exl3(ug, dg) = &s.experts[&0] else { unreachable!() };
        assert!(moe.decode(Table::Up(0), &st).unwrap() == unpack(&ug.trellis, ug.kt, ug.nt, ug.group.bits, &s.table), "{k}: up");
        assert!(
            moe.decode(Table::Down(0), &st).unwrap() == unpack(&dg.trellis, dg.kt, dg.nt, dg.group.bits, &s.table),
            "{k}: down"
        );
    }
    eprintln!("decode: every projection at 2 to 8 bits is bitwise gb_exl3::unpack");
}

/// At every integer width: one expert on the reference kernels (`expert`) and fc1 on the fast ones (`latent_in`: nm2_gemmk, or
/// nm2_gemm4 at 4 bits) against gb_exl3::forward, at the tolerance of the 4-bit checks; fc1's rows are bitwise the same at 1, 4 and 64.
#[test]
fn every_width_matches_forward_on_the_reference_and_fast_kernels() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut r = Rng(0x3E3);
    let x = r.vec(4 * LATENT, -1.0, 1.0);
    let h = r.vec(4 * HIDDEN, -1.0, 1.0);
    for k in 2..=8usize {
        let (w, s) = synthetic_mix(90 + k as u64, &[7], |_| k, |_| k, [k; 4]);
        let mut moe = NemotronMoe::new(w, 64).unwrap();
        let ExpertRef::Exl3(ug, dg) = &s.experts[&7] else { unreachable!() };
        let mut a = exl3_cpu(ug, &s.table, &x, 4);
        relu2(&mut a);
        let want = exl3_cpu(dg, &s.table, &a, 4);
        let xd = DevBuf::from_slice(&x).unwrap();
        let mut out = DevBuf::zeroed(4 * LATENT * 4).unwrap();
        moe.expert(7, &xd, 4, &mut out, &st).unwrap();
        st.sync().unwrap();
        let e_ref = rel_err(&out.to_vec::<f32>().unwrap(), &want);
        let fc1_want = dense_cpu(&s.dense[0], &s.table, &h, 4, HIDDEN, LATENT);
        let run = |moe: &mut NemotronMoe, rows: usize| -> Vec<f32> {
            let hd = DevBuf::from_slice(&tile(&h, 4, rows)).unwrap();
            let mut o = DevBuf::zeroed(rows * LATENT * 4).unwrap();
            moe.latent_in(&hd, rows, &mut o, &st).unwrap();
            st.sync().unwrap();
            o.to_vec::<f32>().unwrap()
        };
        let g4 = run(&mut moe, 4);
        let e_fc1 = rel_err(&g4, &fc1_want);
        let g64 = run(&mut moe, 64);
        let g1 = run(&mut moe, 1);
        eprintln!("{k} bits: expert (reference kernels) vs forward {e_ref:.3e}; fc1 (fast) vs forward {e_fc1:.3e}");
        assert!(e_ref <= 1e-4 && e_fc1 <= 1e-4, "{k} bits: expert {e_ref:e}, fc1 {e_fc1:e}");
        assert!(g64[..4 * LATENT] == g4[..] && g1[..] == g4[..LATENT], "{k} bits: fc1 changes bits with the row count");
    }
}

/// A MixedK block (his pack's shapes: experts at 2 to 6 bits, up and down at different widths; fp16 fc1 and fc2; shared up 5-bit,
/// shared down 4-bit) against the CPU reference, and a row's bits are the same in 1-, 4-, 16- and 64-row calls.
#[test]
fn a_mixedk_block_matches_the_cpu_and_is_row_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let base = Rng(37).vec(4 * HIDDEN, -1.0, 1.0);
    let (_, probe) = synthetic(47, &[], None);
    let (mut picks, mut wts) = (Vec::new(), Vec::new());
    for t in 0..4 {
        let (idx, w, gap) = route_cpu(&logits_cpu(&base[t * HIDDEN..(t + 1) * HIDDEN], &probe.gate), &probe.bias, SCALE);
        assert!(gap > 1e-5, "row {t} has a near-tie at the boundary");
        picks.push(idx);
        wts.push(w);
    }
    let used: Vec<usize> = picks.iter().flatten().copied().collect::<BTreeSet<usize>>().into_iter().collect();
    let widths = [2usize, 3, 4, 5, 6];
    let (w, s) = synthetic_mix(47, &used, |e| widths[e % 5], |e| widths[(e / 5) % 5], [0, 0, 5, 4]);
    assert_eq!(s.gate, probe.gate);
    let pairs: BTreeSet<(usize, usize)> = used.iter().map(|&e| (widths[e % 5], widths[(e / 5) % 5])).collect();
    let mut moe = NemotronMoe::new(w, 64).unwrap();
    let got = run_block(&mut moe, &base, 4, &st);
    let want = block_cpu_mix(&s, &base, 4, &picks, &wts);
    let e = rel_err(&got, &want);
    eprintln!(
        "MixedK block (4 rows, {} experts, {} (up, down) width pairs, fp16 fc1/fc2) vs CPU: max rel {e:.3e}",
        used.len(),
        pairs.len()
    );
    assert!(e <= 1e-3, "MixedK block is off by {e:e}");
    assert!(pairs.len() >= 10, "too few width pairs: {pairs:?}");
    let g16 = run_block(&mut moe, &tile(&base, 4, 16), 16, &st);
    let g64 = run_block(&mut moe, &tile(&base, 4, 64), 64, &st);
    let g1 = run_block(&mut moe, &base[..HIDDEN], 1, &st);
    assert!(g1 == got[..HIDDEN], "the 1-row call differs from row 0 of the 4-row call");
    for (rows, g) in [(16usize, &g16), (64, &g64)] {
        for r in 0..rows {
            assert!(g[r * HIDDEN..(r + 1) * HIDDEN] == got[(r % 4) * HIDDEN..(r % 4 + 1) * HIDDEN], "{rows}-row call, row {r}");
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The F16 router gate (decision D1 of v0.1.2): the router kernels' fp16 instantiation (the pack's bits as stored, widened exactly in the
// chunk) and their f32 one (the oracle: an F16 gate widened on the host)

/// The row counts of the router checks: both sides of the fast router's layout switch (12 / 13 rows) and of the selection's bucket
/// scan (64 rows), and the decode (1) and verify (2, 4) windows.
const ROUTER_ROWS: [usize; 9] = [1, 2, 4, 12, 13, 16, 33, 64, 100];

/// Random F16 values in [-0.05, 0.05] as fp16 bits (subnormals included near zero), and their f32 widening.
fn f16_gate(seed: u64) -> (Vec<u16>, Vec<f32>) {
    let mut r = Rng(seed);
    let bits: Vec<u16> = (0..EXPERTS * HIDDEN).map(|_| gb_exl3::f16::f64_to_f16(r.uniform(-0.05, 0.05) as f64)).collect();
    let f = gb_cuda::nemotron_moe::gate_f32_from_f16(&bits);
    (bits, f)
}

/// A router-only block (no experts loaded) with `gate` (of form `dt`) and `bias`.
fn router_block(gate: DevBuf, dt: gb_cuda::nemotron_moe::GateDtype, bias: &[f32], seed: u64) -> NemotronMoe {
    let (mut w, _) = synthetic(seed, &[], None);
    w.gate = gate;
    w.gate_dtype = dt;
    w.router_bias = DevBuf::from_slice(bias).unwrap();
    NemotronMoe::tables(w).unwrap()
}

/// The fast route (`route_in`) and the reference one (the reference block, whose router is `nm_router_logits` + `nm_router_select`) of
/// `rows` rows of `h`: (logits, picks, weights) of each.
fn routes(moe: &NemotronMoe, s: &mut MoeScratch, h: &[f32], rows: usize, st: &Stream) -> [(Vec<f32>, Vec<i32>, Vec<f32>); 2] {
    let hd = DevBuf::from_slice(h).unwrap();
    let take = |s: &MoeScratch| {
        let (pick, wts) = s.download_route(rows, st).unwrap();
        (s.download_logits(rows, st).unwrap(), pick, wts)
    };
    s.poison(st).unwrap();
    moe.route_in(s, &hd, rows, st).unwrap();
    let fast = take(s);
    let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
    s.poison(st).unwrap();
    moe.forward_reference(s, &hd, rows, &mut out, st).unwrap();
    [fast, take(s)]
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The gate's form is its explicit tag (bf16 and fp16 gates have the same bytes): the published pack's bf16 gate takes today's kernels,
/// an F16 gate is held as its 4 MiB of fp16 bits (no widened copy), an f32 one as 8 MiB.
#[test]
fn the_gate_form_is_its_tag() {
    gpu!();
    use gb_cuda::nemotron_moe::GateDtype;
    assert_eq!((GateDtype::Bf16.bytes(), GateDtype::F16.bytes(), GateDtype::F32.bytes()), (4 << 20, 4 << 20, 8 << 20));
    let (w, _) = synthetic(3, &[], None);
    assert_eq!(
        NemotronMoe::tables(w).unwrap().gate_dtype(),
        GateDtype::Bf16,
        "the published pack's bf16 gate takes today's kernels"
    );
    let (bits, f) = f16_gate(4);
    let m = router_block(DevBuf::from_slice(&bits).unwrap(), GateDtype::F16, &vec![0f32; EXPERTS], 3);
    assert_eq!(m.gate_dtype(), GateDtype::F16);
    assert_eq!(m.weight_bytes(), (EXPERTS * HIDDEN * 2 + EXPERTS * 4) as u64 + dense_bytes(), "the F16 gate is held as stored");
    let m = router_block(DevBuf::from_slice(&f).unwrap(), GateDtype::F32, &vec![0f32; EXPERTS], 3);
    assert_eq!(m.weight_bytes(), (EXPERTS * HIDDEN * 4 + EXPERTS * 4) as u64 + dense_bytes(), "the f32 gate is priced as held");
}

/// The bytes of the synthetic layer's four dense EXL3 projections as `weight_bytes` counts them (trellis + f32 scales).
fn dense_bytes() -> u64 {
    [(HIDDEN, LATENT), (LATENT, HIDDEN), (HIDDEN, SHARED_INTER), (SHARED_INTER, HIDDEN)]
        .iter()
        .map(|&(k, n)| ((k / 16) * (n / 16) * 16 * 4 * 2 + 4 * (k + n)) as u64)
        .sum()
}

/// A gate whose bytes are not its tag's form is refused by name, not read as any form.
#[test]
fn a_gate_of_another_size_than_its_tag_is_refused() {
    gpu!();
    use gb_cuda::nemotron_moe::GateDtype;
    for (bytes, dt) in
        [(EXPERTS * HIDDEN * 3, GateDtype::Bf16), (EXPERTS * HIDDEN * 4, GateDtype::F16), (EXPERTS * HIDDEN * 2, GateDtype::F32)]
    {
        let (mut w, _) = synthetic(3, &[], None);
        w.gate = DevBuf::zeroed(bytes).unwrap();
        w.gate_dtype = dt;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || NemotronMoe::tables(w).map(|_| ())));
        let msg = match r {
            Ok(_) => panic!("a {bytes}-byte {dt:?} gate was accepted"),
            Err(e) => e.downcast_ref::<String>().cloned().unwrap_or_default(),
        };
        assert!(msg.contains(&format!("gate is {bytes} bytes: a {dt:?} gate [512, 4096] is")), "unexpected refusal: {msg}");
    }
}

/// The f32-gate kernels keep the bf16 kernels' summation order: a gate of bf16 values given as f32 gives the bf16 router's logits, picks
/// and weights BITWISE, on the fast route and the reference one, at every row count of the layout switches.
#[test]
fn an_f32_gate_of_bf16_values_gives_the_bf16_router_bits() {
    gpu!();
    let st = Stream::new().unwrap();
    let (w, host) = synthetic(11, &[], None);
    let bias = host.bias.clone();
    let bf = NemotronMoe::tables(w).unwrap();
    let f32g = router_block(DevBuf::from_slice(&host.gate).unwrap(), gb_cuda::nemotron_moe::GateDtype::F32, &bias, 11);
    let mut s = MoeScratch::new(*ROUTER_ROWS.iter().max().unwrap()).unwrap();
    let mut r = Rng(0x6A7E);
    for &rows in &ROUTER_ROWS {
        let h = r.vec(rows * HIDDEN, -1.0, 1.0);
        let [bfast, bref] = routes(&bf, &mut s, &h, rows, &st);
        let [ffast, fref] = routes(&f32g, &mut s, &h, rows, &st);
        for (what, a, b) in [("fast", &bfast, &ffast), ("reference", &bref, &fref)] {
            assert!(
                same_bits(&a.0, &b.0),
                "{rows} rows, {what} route: the f32 gate's logits differ in bits from the bf16 gate's"
            );
            assert_eq!(a.1, b.1, "{rows} rows, {what} route: picks differ");
            assert!(same_bits(&a.2, &b.2), "{rows} rows, {what} route: weights differ in bits");
        }
        assert!(
            same_bits(&ffast.0, &fref.0) && ffast.1 == fref.1 && same_bits(&ffast.2, &fref.2),
            "{rows} rows: f32 fast != reference"
        );
    }
    eprintln!(
        "f32 gate of bf16 values: logits, picks and weights bitwise the bf16 gate's (fast and reference) at rows {ROUTER_ROWS:?}"
    );
}

/// An F16 gate as stored (the fp16 instantiation) gives the f32 kernels' bits on the gate widened on the host: logits, picks and
/// weights bitwise, on the fast route and the reference one, at every row count of the layout switches.
#[test]
fn an_f16_gate_as_stored_gives_the_widened_f32_gates_bits() {
    gpu!();
    use gb_cuda::nemotron_moe::GateDtype;
    let st = Stream::new().unwrap();
    let (bits, f) = f16_gate(0xF15);
    let bias = Rng(0xB1A6).vec(EXPERTS, -0.1, 0.1);
    let g16 = router_block(DevBuf::from_slice(&bits).unwrap(), GateDtype::F16, &bias, 5);
    let g32 = router_block(DevBuf::from_slice(&f).unwrap(), GateDtype::F32, &bias, 5);
    let mut s = MoeScratch::new(*ROUTER_ROWS.iter().max().unwrap()).unwrap();
    let mut r = Rng(0x6A7F);
    for &rows in &ROUTER_ROWS {
        let h = r.vec(rows * HIDDEN, -1.0, 1.0);
        let [a_fast, a_ref] = routes(&g16, &mut s, &h, rows, &st);
        let [b_fast, b_ref] = routes(&g32, &mut s, &h, rows, &st);
        for (what, a, b) in [("fast", &a_fast, &b_fast), ("reference", &a_ref, &b_ref)] {
            assert!(
                same_bits(&a.0, &b.0),
                "{rows} rows, {what}: the F16 gate's logits differ in bits from the widened f32 gate's"
            );
            assert_eq!(a.1, b.1, "{rows} rows, {what}: picks");
            assert!(same_bits(&a.2, &b.2), "{rows} rows, {what}: weights");
        }
    }
    eprintln!("F16 gate as stored: logits, picks and weights bitwise the widened f32 gate's (fast and reference) at rows {ROUTER_ROWS:?}");
}

/// An F16 gate (as stored): the route matches the CPU route (moe_ref's, in fp32 on the widened values) on every row with no near-tie,
/// the fast route equals the reference one bitwise, and a row's bits do not depend on the rows in the call.
#[test]
fn an_f16_gate_widened_routes_like_the_cpu_and_its_paths_agree() {
    gpu!();
    let st = Stream::new().unwrap();
    let (bits, gate) = f16_gate(0xF16);
    let bias = Rng(0xB1A5).vec(EXPERTS, -0.1, 0.1);
    let moe = router_block(DevBuf::from_slice(&bits).unwrap(), gb_cuda::nemotron_moe::GateDtype::F16, &bias, 5);
    let mut s = MoeScratch::new(*ROUTER_ROWS.iter().max().unwrap()).unwrap();
    let mut r = Rng(0xA11CF);
    let (mut checked, mut skipped, mut max_logit, mut max_wrel) = (0usize, 0usize, 0f32, 0f64);
    let mut keep = Vec::new();
    for &rows in &ROUTER_ROWS {
        let h = r.vec(rows * HIDDEN, -1.0, 1.0);
        let [fast, reference] = routes(&moe, &mut s, &h, rows, &st);
        assert!(same_bits(&fast.0, &reference.0), "{rows} rows: fast logits differ in bits from the reference kernel's");
        assert_eq!(fast.1, reference.1, "{rows} rows: picks");
        assert!(same_bits(&fast.2, &reference.2), "{rows} rows: weights differ in bits");
        for t in 0..rows {
            let lc = logits_cpu(&h[t * HIDDEN..(t + 1) * HIDDEN], &gate);
            for e in 0..EXPERTS {
                max_logit = max_logit.max((lc[e] - fast.0[t * EXPERTS + e]).abs());
            }
            let (idx, w, gap) = route_cpu(&lc, &bias, SCALE);
            if gap <= 1e-5 {
                skipped += 1;
                continue;
            }
            let got: Vec<usize> = fast.1[t * TOPK..(t + 1) * TOPK].iter().map(|&v| v as usize).collect();
            assert_eq!(got, idx, "rows {rows}, row {t}: the GPU's top-22 differs from the CPU route");
            max_wrel = max_wrel.max(rel_err(&fast.2[t * TOPK..(t + 1) * TOPK], &w));
            checked += 1;
        }
        if rows == 100 {
            keep = vec![(h, fast)];
        }
    }
    eprintln!("f16 gate (widened): {checked} rows checked, {skipped} skipped; max |logit gpu - cpu| {max_logit:.3e}; weights max rel {max_wrel:.3e}");
    assert!(checked >= 200, "too few rows with a clear boundary");
    assert!(max_wrel <= 1e-5, "weights differ from the CPU route by {max_wrel:e}");
    // row invariance: each of the 100-row call's rows alone, and rows 40..44 as a 4-row call
    let (h, full) = &keep[0];
    for row in (0..100).step_by(7) {
        let [one, _] = routes(&moe, &mut s, &h[row * HIDDEN..(row + 1) * HIDDEN], 1, &st);
        assert!(same_bits(&one.0, &full.0[row * EXPERTS..(row + 1) * EXPERTS]), "row {row}: logits change with the row count");
        assert_eq!(one.1[..], full.1[row * TOPK..(row + 1) * TOPK], "row {row}: picks change with the row count");
    }
    let [four, _] = routes(&moe, &mut s, &h[40 * HIDDEN..44 * HIDDEN], 4, &st);
    assert!(same_bits(&four.0, &full.0[40 * EXPERTS..44 * EXPERTS]), "rows 40..44 change bits in a 4-row call");
    eprintln!("f16 gate (widened): fast == reference bitwise at rows {ROUTER_ROWS:?}; rows identical alone, in 4 and in 100");
}

/// The whole block with an F16 gate (as stored) and EXL3 experts: against the CPU reference, and the fast launches equal the reference
/// ones.
#[test]
fn the_block_with_an_f16_gate_matches_the_cpu() {
    gpu!();
    let st = Stream::new().unwrap();
    let base = Rng(31).vec(4 * HIDDEN, -1.0, 1.0);
    let (gbits, gate) = f16_gate(0xF17);
    let (_, probe) = synthetic(43, &[], None);
    let (mut picks, mut wts) = (Vec::new(), Vec::new());
    for t in 0..4 {
        let (idx, w, gap) = route_cpu(&logits_cpu(&base[t * HIDDEN..(t + 1) * HIDDEN], &gate), &probe.bias, SCALE);
        assert!(gap > 1e-5, "row {t} has a near-tie at the boundary");
        picks.push(idx);
        wts.push(w);
    }
    let used: Vec<usize> = picks.iter().flatten().copied().collect::<BTreeSet<usize>>().into_iter().collect();
    let (mut w, s) = synthetic(43, &used, None);
    w.gate = DevBuf::from_slice(&gbits).unwrap();
    w.gate_dtype = gb_cuda::nemotron_moe::GateDtype::F16;
    let mut moe = NemotronMoe::new(w, 64).unwrap();
    let got = run_block(&mut moe, &base, 4, &st);
    let want = block_cpu(&s, &base, 4, &picks, &wts);
    let e = rel_err(&got, &want);
    eprintln!("block (synthetic, 4 rows, F16 gate as stored) vs CPU reference: max rel {e:.3e}");
    assert!(e <= 1e-3, "block is off by {e:e}");
    let mut sc = MoeScratch::new(4).unwrap();
    let hd = DevBuf::from_slice(&base).unwrap();
    let mut out = DevBuf::zeroed(4 * HIDDEN * 4).unwrap();
    moe.forward_reference(&mut sc, &hd, 4, &mut out, &st).unwrap();
    st.sync().unwrap();
    assert!(same_bits(&out.to_vec::<f32>().unwrap(), &got), "the reference launches differ from the fast ones with an f32 gate");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Pack checks (REXL3_NEMO_OMNI_DIR)

/// The fixture's tensors, read through its safetensors header (`read_header`) and the file's bytes.
struct Fixture {
    bytes: Vec<u8>,
    tensors: BTreeMap<String, (String, Vec<usize>, usize, usize)>,
}

/// The ORIGINAL pack's layer-1 fixture (the first upload, before the repair): made by `tools/gen_moe_fixture.py` from that pack (index
/// sha256 4bd072f9...). Not the Hub pack published since 2026-10-14: that one holds the repaired tensors (see [`REPAIRED_INDEX_FNV1A`]).
const ORIGINAL_FIXTURE: &str = "layer1_fixture.safetensors";
/// The repaired pack's own: the same generator, the same seed and input, run on the repaired pack (index sha256 5c4938c5...0776). The
/// repair re-encoded layer 1's shared up projection (and routed experts 11 and 429, which these four tokens do not pick), so the
/// block's output differs from the published pack's by 3.7e-2 (relative to its max) by construction: against the published fixture
/// the 1e-3 gate would measure the repair, not the kernels. Every other tensor of the two fixtures is equal (router, x_latent, the
/// routed latents, the bf16 expert's input and output, all the digests): the router, fc1, fc2 and the 82 picked experts are unchanged.
const REPAIRED_FIXTURE: &str = "layer1_fixture_repaired.safetensors";
/// FNV-1a 64 of the `model.safetensors.index.json` of every pack that holds the repaired tensors: the repaired pack itself (18,166,901
/// bytes, sha256 5c4938c539d2...) and the Hub pack `r0b0tlab/rexl3-nemotron3.5-super-vl-exl3-mtp` (sha256 d2c82b141efa..., FNV-1a
/// measured by lane G2), which is the repaired pack re-sharded: its tensors equal the repaired pack's (166,757 of 166,757), so its
/// layer 1 is the repaired layer 1 and its fixture is the repaired one. Any other index falls back to the original pack's fixture.
/// Since v0.1.1 also the H4 pack `r0b0tlab/rexl3-nemotron3.5-super-vl-exl3-4.22bpw-mtp` (weights e3857098, index sha256 eb0f1ada...):
/// its trunk is the repaired pack's (layer 1: 4,115 of 4,115 tensors byte-identical, lane K 2026-10-17), with a 4-bit head.
const REPAIRED_INDEX_FNV1A: [(u64, &str); 3] = [
    (0xfae2a114dd50e0e2, "the repaired pack's index (5c4938c5)"),
    (0xc7f15360ccd846e4, "the published Hub pack's index (d2c82b14): the repaired tensors, re-sharded"),
    (0x502dfb7c3ea85676, "the H4 pack's index (eb0f1ada): the repaired trunk, with a 4-bit head"),
];

impl Fixture {
    /// The fixture that goes with the pack in `dir`, chosen by its index: the repaired one for an index of [`REPAIRED_INDEX_FNV1A`],
    /// else the original pack's (so any other pack is compared with the original pack's fixture, as it always was, and fails loudly if
    /// its layer 1 differs).
    fn for_pack(dir: &Path) -> Self {
        let index = dir.join("model.safetensors.index.json");
        let bytes = std::fs::read(&index).unwrap_or_else(|e| panic!("{}: {e}", index.display()));
        let h = fnv1a(bytes.iter().copied());
        let (name, why) = match REPAIRED_INDEX_FNV1A.iter().find(|(f, _)| *f == h) {
            Some((_, which)) => (REPAIRED_FIXTURE, *which),
            None => (ORIGINAL_FIXTURE, "not a repaired-tensor index: the original pack's fixture"),
        };
        eprintln!("pack index fnv1a {h:016x}");
        eprintln!("fixture {name} ({why})");
        Self::open(name)
    }

    fn open(name: &str) -> Self {
        let path = repo_root().join("testdata/nemotron_omni/moe").join(name);
        let (n, header) = read_header(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let bytes = std::fs::read(&path).unwrap();
        let data = 8 + n as usize;
        let mut tensors = BTreeMap::new();
        for (name, v) in header {
            let dtype = v["dtype"].as_str().unwrap().to_string();
            let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let off = v["data_offsets"].as_array().unwrap();
            let (s, e) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
            tensors.insert(name, (dtype, shape, data + s, data + e));
        }
        Self { bytes, tensors }
    }

    fn get(&self, name: &str, dtype: &str) -> &[u8] {
        let (d, _, s, e) = self.tensors.get(name).unwrap_or_else(|| panic!("fixture has no tensor {name}"));
        assert_eq!(d, dtype, "{name} is {d}, not {dtype}");
        &self.bytes[*s..*e]
    }

    fn f32(&self, name: &str) -> Vec<f32> {
        self.get(name, "F32").chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn i64(&self, name: &str) -> Vec<i64> {
        self.get(name, "I64").chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn u64(&self, name: &str) -> u64 {
        u64::from_le_bytes(self.get(name, "U64").try_into().unwrap())
    }
}

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

/// The pack's tensors as the engine places them: through the index, so a repaired pack's overlay shard wins over the published copies
/// (`gb_exl3::checkpoint::scan` resolves duplicates pairwise and refuses a tensor three shards list; see `scan_indexed`).
fn scan_pack(dir: &Path) -> Checkpoint {
    scan_indexed(dir, true).unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
}

fn group(ck: &Checkpoint, prefix: &str) -> LoadedGroup {
    let g = ck.load_group(prefix).unwrap_or_else(|e| panic!("{prefix}: {e} (refused: {:?})", ck.bad.get(prefix)));
    assert!(g.bias.is_none(), "{prefix} has a bias, which this test does not expect");
    g
}

/// Layer 1's block from the pack: the gate, the bias, the four dense EXL3 projections, and the routed EXL3 experts in `used`.
fn pack_layer1(ck: &Checkpoint, used: &BTreeSet<usize>) -> Weights {
    let mp = "language_model.backbone.layers.1.mixer";
    let mut up: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    let mut down: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    for &e in used {
        up[e] = Proj::exl3_from_group(&group(ck, &format!("{mp}.experts.{e}.up_proj"))).unwrap();
        down[e] = Proj::exl3_from_group(&group(ck, &format!("{mp}.experts.{e}.down_proj"))).unwrap();
    }
    Weights {
        gate: DevBuf::from_slice(&plain_u16(ck, &format!("{mp}.gate.weight"))).unwrap(),
        gate_dtype: gb_cuda::nemotron_moe::GateDtype::Bf16,
        router_bias: DevBuf::from_slice(&plain_f32(ck, &format!("{mp}.gate.e_score_correction_bias"))).unwrap(),
        scale: SCALE,
        fc1: Proj::exl3_from_group(&group(ck, &format!("{mp}.fc1_latent_proj"))).unwrap(),
        fc2: Proj::exl3_from_group(&group(ck, &format!("{mp}.fc2_latent_proj"))).unwrap(),
        shared_up: Proj::exl3_from_group(&group(ck, &format!("{mp}.shared_experts.up_proj"))).unwrap(),
        shared_down: Proj::exl3_from_group(&group(ck, &format!("{mp}.shared_experts.down_proj"))).unwrap(),
        up,
        down,
    }
}

/// The pack's layer 1 against torch's fixture: the router (indices and weights), fc1, the spot EXL3 expert, the block, the decoded
/// trellis digests, and row invariance of the block on the fixture's tokens tiled to 16 and 64 rows.
#[test]
fn pack_layer1_matches_torch_fixture() {
    gpu!();
    let Some(dir) = pack_dir() else { return };
    let st = Stream::new().unwrap();
    let fx = Fixture::for_pack(&dir);
    let h = fx.f32("h");
    let idx_want = fx.i64("route_indices");
    let used: BTreeSet<usize> = idx_want.iter().map(|&e| e as usize).collect();
    let ck = scan_pack(&dir);
    let mp = "language_model.backbone.layers.1.mixer";
    assert_eq!(
        plain_f32(&ck, &format!("{mp}.gate.e_score_correction_bias")),
        fx.f32("router_bias"),
        "the pack's bias is not the fixture's"
    );
    let mut moe = NemotronMoe::new(pack_layer1(&ck, &used), 64).unwrap();
    eprintln!("loaded {} routed experts for the fixture's 4 tokens", used.len());

    // Router: logits against torch's, and the selection and weights against torch's indices and weights.
    let hd = DevBuf::from_slice(&h).unwrap();
    moe.route(&hd, 4, &st).unwrap();
    let logits = moe.download_logits(4, &st).unwrap();
    let max_logit = logits.iter().zip(&fx.f32("router_logits")).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let (pick, wts) = moe.download_route(4, &st).unwrap();
    let got_idx: Vec<i64> = pick.iter().map(|&v| v as i64).collect();
    assert_eq!(got_idx, idx_want, "the GPU router's top-22 differs from torch's on the fixture");
    let rw = rel_err(&wts, &fx.f32("route_weights"));
    eprintln!("router vs torch: logits max |diff| {max_logit:.3e}; 88/88 indices equal; weights max rel {rw:.3e}");
    assert!(rw <= 1e-4, "router weights differ from torch by {rw:e}");

    // fc1 against torch's x_latent, and the decoded W_q against torch's digest and gb_exl3::unpack.
    let mut x = DevBuf::zeroed(4 * LATENT * 4).unwrap();
    moe.latent_in(&hd, 4, &mut x, &st).unwrap();
    st.sync().unwrap();
    let e_fc1 = rel_err(&x.to_vec::<f32>().unwrap(), &fx.f32("x_latent"));
    eprintln!("fc1 (x_latent) vs torch: max rel {e_fc1:.3e}");
    assert!(e_fc1 <= 1e-4, "fc1 is off by {e_fc1:e}");
    let fc1_bits = moe.decode(Table::Dense(0), &st).unwrap();
    assert_eq!(
        fnv1a(fc1_bits.iter().flat_map(|v| v.to_le_bytes())),
        fx.u64("digest_fc1_latent_proj"),
        "fc1's W_q differs from torch's decode"
    );
    let fc1_group = group(&ck, &format!("{mp}.fc1_latent_proj"));
    assert!(fc1_bits == unpack(&fc1_group.trellis, fc1_group.kt, fc1_group.nt, fc1_group.group.bits, &codebook(Codebook::Mul1)));

    // The spot EXL3 expert on token 0's latent, against torch; its decoded up and down against torch's digests.
    let spot = fx.i64("spot_exl3_expert")[0] as usize;
    let xs = DevBuf::from_slice(&fx.f32("spot_exl3_input")).unwrap();
    let mut ys = DevBuf::zeroed(LATENT * 4).unwrap();
    moe.expert(spot, &xs, 1, &mut ys, &st).unwrap();
    st.sync().unwrap();
    let e_spot = rel_err(&ys.to_vec::<f32>().unwrap(), &fx.f32("spot_exl3_output"));
    eprintln!("EXL3 expert {spot} (spot) vs torch: max rel {e_spot:.3e}");
    assert!(e_spot <= 1e-4, "the spot expert is off by {e_spot:e}");
    for (name, table) in [("up", Table::Up(spot)), ("down", Table::Down(spot))] {
        let g = group(&ck, &format!("{mp}.experts.{spot}.{name}_proj"));
        let got = moe.decode(table, &st).unwrap();
        assert_eq!(
            fnv1a(got.iter().flat_map(|v| v.to_le_bytes())),
            fx.u64(&format!("digest_expert_spot_{name}_proj")),
            "expert {spot} {name}: decode differs from torch"
        );
        assert!(
            got == unpack(&g.trellis, g.kt, g.nt, g.group.bits, &codebook(Codebook::Mul1)),
            "expert {spot} {name}: decode differs from unpack"
        );
    }
    eprintln!("decoded W_q: fc1 and the spot expert's up and down match torch's digests and gb_exl3::unpack bitwise");

    // The block against torch's out; then the tiled rows, which must give the same bits.
    let out = run_block(&mut moe, &h, 4, &st);
    let e_block = rel_err(&out, &fx.f32("out"));
    eprintln!("block (pack layer 1, 4 rows) vs torch: max rel {e_block:.3e}");
    assert!(e_block <= 1e-3, "block is off by {e_block:e}");
    let g16 = run_block(&mut moe, &tile(&h, 4, 16), 16, &st);
    let g64 = run_block(&mut moe, &tile(&h, 4, 64), 64, &st);
    assert!(out == g16[..4 * HIDDEN] && out == g64[..4 * HIDDEN], "a row changes bits with the number of rows");
    let g1 = run_block(&mut moe, &h[..HIDDEN], 1, &st);
    assert!(g1 == out[..HIDDEN], "the 1-row (decode) call differs from row 0 of the 4-row call");
    for (rows, g) in [(16usize, &g16), (64, &g64)] {
        for r in 0..rows {
            assert!(
                g[r * HIDDEN..(r + 1) * HIDDEN] == out[(r % 4) * HIDDEN..(r % 4 + 1) * HIDDEN],
                "{rows}-row call, row {r} changes bits"
            );
        }
    }
    eprintln!("block row invariance on the pack: 1 vs 4 vs 16 vs 64 rows identical");
}

/// The bf16 expert of layer 26 (expert 111) on torch's spot input (layer 1's latents), against torch.
#[test]
fn pack_bf16_expert_matches_torch() {
    gpu!();
    let Some(dir) = pack_dir() else { return };
    let st = Stream::new().unwrap();
    let fx = Fixture::for_pack(&dir);
    let ck = scan_pack(&dir);
    let bp = "language_model.backbone.layers.26.mixer.experts.111";
    // Layer 1's gate and dense projections make a valid layer; the expert under test sits in table entry 0.
    let mut ws = pack_layer1(&ck, &BTreeSet::new());
    ws.up[0] = Proj::bf16(LATENT, EXPERT_INTER, &plain_u16(&ck, &format!("{bp}.up_proj.weight"))).unwrap();
    ws.down[0] = Proj::bf16(EXPERT_INTER, LATENT, &plain_u16(&ck, &format!("{bp}.down_proj.weight"))).unwrap();
    let mut moe = NemotronMoe::new(ws, 16).unwrap();
    let xd = DevBuf::from_slice(&fx.f32("spot_bf16_input")).unwrap();
    let mut out = DevBuf::zeroed(4 * LATENT * 4).unwrap();
    moe.expert(0, &xd, 4, &mut out, &st).unwrap();
    st.sync().unwrap();
    let e = rel_err(&out.to_vec::<f32>().unwrap(), &fx.f32("spot_bf16_output"));
    eprintln!("bf16 expert (layer 26, expert 111) vs torch: max rel {e:.3e}");
    assert!(e <= 1e-4, "the bf16 expert is off by {e:e}");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The MixedK pack's F16 router gates (`REXL3_NEMO_MIXEDK_DIR`: vcruz305/Nemotron-Super-VL-EXL3-3.85bpw). Reads the gate and bias of a
// few MoE layers only (4 MiB each), no model load.

fn mixedk_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("REXL3_NEMO_MIXEDK_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("not run: REXL3_NEMO_MIXEDK_DIR is not set; the MixedK pack's router check needs its gates");
    }
    dir
}

/// The pack's tensor `name`, under its bare name or with the `language_model.` prefix.
fn plain_entry<'a>(ck: &'a Checkpoint, name: &str) -> (&'a gb_exl3::checkpoint::Entry, String) {
    for n in [name.to_string(), format!("language_model.{name}")] {
        if let Some(e) = ck.plain.get(&n) {
            return (e, n);
        }
    }
    panic!("the pack has no plain tensor {name} (bare or under language_model.)")
}

/// Each sampled MoE layer's F16 gate, as stored (what the bridge hands over), routes random rows as the CPU route does on every row with
/// no near-tie, gives the f32 kernels' bits on the gate widened on the host, its fast and reference routes are bitwise equal, and a row
/// alone equals its bits in a 64-row call.
#[test]
fn mixedk_pack_f16_gates_route_like_the_cpu() {
    gpu!();
    let Some(dir) = mixedk_dir() else { return };
    let st = Stream::new().unwrap();
    let ck = scan_indexed(&dir, true).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let mut s = MoeScratch::new(64).unwrap();
    let mut r = Rng(0x3D85);
    for m in ["backbone.layers.1.mixer", "backbone.layers.3.mixer", "backbone.layers.87.mixer", "mtp.layers.1.mixer"] {
        let (ge, gname) = plain_entry(&ck, &format!("{m}.gate.weight"));
        assert_eq!((ge.dtype.as_str(), ge.shape.clone()), ("F16", vec![EXPERTS, HIDDEN]), "{gname}");
        let bits = ck.read_u16(ge).unwrap();
        let gate = gb_cuda::nemotron_moe::gate_f32_from_f16(&bits);
        assert!(gate.iter().all(|v| v.is_finite()), "{gname}: a non-finite gate value");
        let (be, bname) = plain_entry(&ck, &format!("{m}.gate.e_score_correction_bias"));
        assert_eq!(be.dtype, "F32", "{bname}");
        let bias: Vec<f32> = ck.read_floats(be).unwrap().into_iter().map(|v| v as f32).collect();
        let moe = router_block(DevBuf::from_slice(&bits).unwrap(), gb_cuda::nemotron_moe::GateDtype::F16, &bias, 9);
        let wide = router_block(DevBuf::from_slice(&gate).unwrap(), gb_cuda::nemotron_moe::GateDtype::F32, &bias, 9);
        let (mut checked, mut skipped, mut max_wrel) = (0usize, 0usize, 0f64);
        for &rows in &[1usize, 4, 13, 64] {
            // rows of an RMS-normed hidden's scale (the router sees the normed block input)
            let h = r.vec(rows * HIDDEN, -2.0, 2.0);
            let [fast, reference] = routes(&moe, &mut s, &h, rows, &st);
            assert!(
                same_bits(&fast.0, &reference.0) && fast.1 == reference.1 && same_bits(&fast.2, &reference.2),
                "{gname}: {rows} rows: fast != reference"
            );
            let [wfast, _] = routes(&wide, &mut s, &h, rows, &st);
            assert!(
                same_bits(&fast.0, &wfast.0) && fast.1 == wfast.1 && same_bits(&fast.2, &wfast.2),
                "{gname}: {rows} rows: F16 != f32 widened"
            );
            for t in 0..rows {
                let (idx, w, gap) = route_cpu(&logits_cpu(&h[t * HIDDEN..(t + 1) * HIDDEN], &gate), &bias, SCALE);
                if gap <= 1e-5 {
                    skipped += 1;
                    continue;
                }
                let got: Vec<usize> = fast.1[t * TOPK..(t + 1) * TOPK].iter().map(|&v| v as usize).collect();
                assert_eq!(got, idx, "{gname}: {rows} rows, row {t}: the GPU's top-22 differs from the CPU route");
                max_wrel = max_wrel.max(rel_err(&fast.2[t * TOPK..(t + 1) * TOPK], &w));
                checked += 1;
            }
            if rows == 64 {
                let [one, _] = routes(&moe, &mut s, &h[17 * HIDDEN..18 * HIDDEN], 1, &st);
                assert!(
                    same_bits(&one.0, &fast.0[17 * EXPERTS..18 * EXPERTS]),
                    "{gname}: row 17 changes bits with the row count"
                );
            }
        }
        eprintln!("{gname} (F16 as stored): {checked} rows checked, {skipped} skipped; weights max rel {max_wrel:.3e}; fast == reference == f32 widened, bitwise");
        assert!(checked >= 60, "{gname}: too few rows with a clear boundary");
        assert!(max_wrel <= 1e-5, "{gname}: weights differ from the CPU route by {max_wrel:e}");
    }
}
