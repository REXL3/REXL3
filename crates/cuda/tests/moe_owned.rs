//! Whole-expert ownership of the routed experts (`src/moe_owned.rs`, `csrc/moe_owned.cu`): a TWO-RANK EMULATION ON ONE GPU on real
//! layers of the GLM-5.3-Flash pack. Rank 0 holds experts 0..144, rank 1 holds 144..288; each runs its pairs stage, the two pair
//! buffers are summed with the loopback reduction (`nccl::add_f32_into`, one IEEE add, no flush to zero), and the unchanged combine
//! runs over the sum. The result must equal the single-node engine BYTE FOR BYTE: `out` of every row, and `y` of every valid pair, on
//! the decode path (`run`) and on the batched path (`run_prefill`), at 1, 2, 5, 22 and 64 rows. Needs `REXL3_GLM_DIR` (default
//! `~/models/r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121`); skips without a device or a pack.
//!
//!     cargo test --release -p gb-cuda --test moe_owned -- --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments)]
mod common;

use common::exl3_synth::synth_group;
use common::*;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::moe::*;
use gb_cuda::moe_prefill::PrefillScratch;
use gb_cuda::{device_available, nccl, DevBuf, Stream};
use gb_exl3::checkpoint::{scan, Checkpoint, LoadedGroup};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::Codebook;

const ROWS: [usize; 5] = [1, 2, 5, 22, 64];
const NEG_ZERO: u32 = 0x8000_0000;

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

fn pack_dir() -> Option<std::path::PathBuf> {
    let p = std::env::var("REXL3_GLM_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| {
        std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("models/r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121")
    });
    p.join("model.safetensors.index.json").exists().then_some(p)
}

fn checkpoint() -> Option<&'static Checkpoint> {
    static CK: std::sync::OnceLock<Option<Checkpoint>> = std::sync::OnceLock::new();
    CK.get_or_init(|| Some(scan(&pack_dir()?, false).expect("scan the GLM pack"))).as_ref()
}

/// Every expert's three groups of one layer, on the host (global id order).
struct LayerGroups {
    gate: Vec<LoadedGroup>,
    up: Vec<LoadedGroup>,
    down: Vec<LoadedGroup>,
}

fn load_layer(ck: &Checkpoint, l: usize, n: usize) -> LayerGroups {
    let t = std::time::Instant::now();
    let load = |proj: &str| -> Vec<LoadedGroup> {
        let threads = 8usize;
        let per = n.div_ceil(threads);
        let mut out: Vec<Option<LoadedGroup>> = (0..n).map(|_| None).collect();
        std::thread::scope(|sc| {
            for (t, chunk) in out.chunks_mut(per).enumerate() {
                sc.spawn(move || {
                    for (j, slot) in chunk.iter_mut().enumerate() {
                        let prefix = format!("model.language_model.layers.{l}.mlp.experts.{}.{proj}", t * per + j);
                        *slot = Some(ck.load_group(&prefix).unwrap_or_else(|e| panic!("{prefix}: {e}")));
                    }
                });
            }
        });
        out.into_iter().map(|o| o.unwrap()).collect()
    };
    let g = LayerGroups { gate: load("gate_proj"), up: load("up_proj"), down: load("down_proj") };
    eprintln!("layer {l}: groups read in {:.1} s", t.elapsed().as_secs_f64());
    g
}

fn pick_owned(v: &[LoadedGroup], own: &ExpertOwnership) -> Vec<LoadedGroup> {
    own.owned().iter().map(|&g| v[g as usize].clone()).collect()
}

fn bank(lg: &LayerGroups, d: MoeDims, own: ExpertOwnership) -> MoeExperts {
    MoeExperts::from_groups_owned(&pick_owned(&lg.gate, &own), &pick_owned(&lg.up, &own), &pick_owned(&lg.down, &own), d, own)
        .unwrap()
}

fn up<T: Copy>(v: &[T]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}

fn dl<T: Copy + Default>(b: &DevBuf, st: &Stream) -> Vec<T> {
    st.sync().unwrap();
    b.to_vec().unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Picks of `rows` rows: random distinct experts, then the special rows: all 8 on rank 0, all 8 on rank 1, an invalid pick (-1 and
/// 999) in the middle of a row, and an expert repeated across rows.
fn picks(r: &mut Rng, rows: usize, d: &MoeDims) -> (Vec<i32>, Vec<u16>) {
    let (e, k) = (d.experts, d.topk);
    let mut pick = Vec::with_capacity(rows * k);
    for _ in 0..rows {
        let mut ids: Vec<i32> = (0..e as i32).collect();
        for i in (1..ids.len()).rev() {
            ids.swap(i, (r.next_u64() % (i as u64 + 1)) as usize);
        }
        pick.extend_from_slice(&ids[..k]);
    }
    let special: [Vec<i32>; 4] = [
        (0..k as i32).map(|j| 3 + 17 * j).collect(),   // all on rank 0
        (0..k as i32).map(|j| 150 + 16 * j).collect(), // all on rank 1
        (0..k as i32)
            .map(|j| {
                if j == 2 {
                    -1
                } else if j == 5 {
                    999
                } else {
                    140 + j
                }
            })
            .collect(), // invalid picks, straddling 144
        (0..k as i32).map(|j| 143 + j).collect(),      // the boundary
    ];
    for (i, s) in special.iter().enumerate() {
        let row = [rows / 2, rows - 1, 0, rows / 3][i];
        if rows > i {
            pick[row * k..(row + 1) * k].copy_from_slice(s);
        }
    }
    let wts = (0..rows * k).map(|_| f64_to_f16(0.05 + 0.6 * (r.next_u64() >> 40) as f64 / 16777216.0)).collect();
    (pick, wts)
}

struct Case {
    x: DevBuf,
    shared: DevBuf,
    pick: Vec<i32>,
    wts: Vec<u16>,
}

fn case(d: &MoeDims, rows: usize, seed: u64) -> Case {
    let mut r = Rng(seed);
    let xh: Vec<u16> = r.vec(rows * d.hidden, -2.0, 2.0).iter().map(|&v| f64_to_f16(v as f64)).collect();
    let sh = r.vec(rows * d.hidden, -0.5, 0.5);
    let (pick, wts) = picks(&mut r, rows, d);
    Case { x: up(&xh), shared: up(&sh), pick, wts }
}

/// Single-node reference and the two-rank emulation of one case on the decode path. Returns (out bytes equal, y bytes equal).
fn decode_case(whole: &MoeExperts, ranks: &[&MoeExperts], c: &Case, rows: usize, st: &Stream, tag: &str) {
    let d = whole.dims;
    let p = rows * d.topk;
    // single node, today's call
    let mut s = MoeScratch::new(&d, 64).unwrap();
    s.set_picks(&c.pick, &c.wts).unwrap();
    let mut out_ref = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    whole.run(&c.x, Dtype::F16, rows, Some(&c.shared), &mut out_ref, &mut s, true, st).unwrap();
    let out_ref: Vec<f32> = dl(&out_ref, st);
    let y_ref: Vec<f32> = dl::<f32>(&s.y_pairs, st)[..p * d.hidden].to_vec();

    // the ranks, each with its own (poisoned) scratch
    let mut ys: Vec<MoeScratch> = Vec::new();
    for (ri, ex) in ranks.iter().enumerate() {
        let mut sr = MoeScratch::new(&d, 64).unwrap();
        sr.poison(0x7f, st).unwrap();
        sr.set_picks(&c.pick, &c.wts).unwrap();
        let mut os = OwnScratch::new(&d, 64).unwrap();
        ex.run_pairs(&c.x, Dtype::F16, rows, &mut sr, &mut os, st).unwrap();
        let y: Vec<f32> = dl(&sr.y_pairs, st);
        // the filler: every pair this rank does not hold is exactly -0.0; every pair it holds is the single node's y
        let own = ex.ownership();
        for q in 0..p {
            let e = c.pick[q];
            let held = e >= 0 && (e as usize) < d.experts && own.is_none_or(|o| o.owns(e as usize));
            let row = &y[q * d.hidden..(q + 1) * d.hidden];
            if held {
                assert_eq!(bits(row), bits(&y_ref[q * d.hidden..(q + 1) * d.hidden]), "{tag} rank {ri} pair {q}: y");
            } else {
                assert!(row.iter().all(|v| v.to_bits() == NEG_ZERO), "{tag} rank {ri} pair {q}: filler is not -0.0");
            }
        }
        ys.push(sr);
    }
    // the exchange: rank 0's buffer += every other rank's
    let (first, rest) = ys.split_at_mut(1);
    for other in rest.iter() {
        nccl::add_f32_into(&mut first[0].y_pairs, 0, &other.y_pairs, 0, p * d.hidden, st).unwrap();
    }
    let s0 = &first[0];
    let mut out = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    combine_pairs(&s0.y_pairs, &s0.pick, &s0.wts, Some(&c.shared), &mut out, rows, &d, st).unwrap();
    let out: Vec<f32> = dl(&out, st);
    let ysum: Vec<f32> = dl::<f32>(&s0.y_pairs, st)[..p * d.hidden].to_vec();
    for q in 0..p {
        let e = c.pick[q];
        let r = &ysum[q * d.hidden..(q + 1) * d.hidden];
        if e >= 0 && (e as usize) < d.experts {
            assert_eq!(bits(r), bits(&y_ref[q * d.hidden..(q + 1) * d.hidden]), "{tag}: summed y of pair {q}");
        } else {
            assert!(r.iter().all(|v| v.to_bits() == NEG_ZERO), "{tag}: summed y of invalid pair {q}");
        }
    }
    assert_eq!(bits(&out), bits(&out_ref), "{tag}: out");
}

/// The same on the batched path; also pins the batched single node against the decode single node (the existing contract).
fn batched_case(whole: &MoeExperts, ranks: &[&MoeExperts], c: &Case, rows: usize, st: &Stream, tag: &str) {
    let d = whole.dims;
    let p = rows * d.topk;
    let cap = rows.max(64);
    let mut s = PrefillScratch::new(&d, cap).unwrap();
    s.set_picks(&c.pick, &c.wts).unwrap();
    let mut out_ref = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    whole.run_prefill(&c.x, Dtype::F16, rows, Some(&c.shared), &mut out_ref, &mut s, st).unwrap();
    let out_ref: Vec<f32> = dl(&out_ref, st);
    let y_ref: Vec<f32> = dl::<f32>(&s.y_pairs, st)[..p * d.hidden].to_vec();
    // decode single node gives the same out (the batched path's own contract; cheap to re-pin here)
    if rows <= MAX_ROWS {
        let mut sd = MoeScratch::new(&d, 64).unwrap();
        sd.set_picks(&c.pick, &c.wts).unwrap();
        let mut out_dec = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
        whole.run(&c.x, Dtype::F16, rows, Some(&c.shared), &mut out_dec, &mut sd, false, st).unwrap();
        assert_eq!(bits(&dl::<f32>(&out_dec, st)), bits(&out_ref), "{tag}: batched single node != decode single node");
    }

    let mut ys: Vec<PrefillScratch> = Vec::new();
    for (ri, ex) in ranks.iter().enumerate() {
        let mut sr = PrefillScratch::new(&d, cap).unwrap();
        sr.poison(0x7f, st).unwrap();
        sr.set_picks(&c.pick, &c.wts).unwrap();
        let mut os = OwnScratch::new(&d, cap).unwrap();
        ex.run_prefill_pairs(&c.x, Dtype::F16, rows, &mut sr, &mut os, st).unwrap();
        let y: Vec<f32> = dl(&sr.y_pairs, st);
        let own = ex.ownership();
        for q in 0..p {
            let e = c.pick[q];
            let held = e >= 0 && (e as usize) < d.experts && own.is_none_or(|o| o.owns(e as usize));
            let row = &y[q * d.hidden..(q + 1) * d.hidden];
            if held {
                assert_eq!(bits(row), bits(&y_ref[q * d.hidden..(q + 1) * d.hidden]), "{tag} rank {ri} pair {q}: y");
            } else {
                assert!(row.iter().all(|v| v.to_bits() == NEG_ZERO), "{tag} rank {ri} pair {q}: filler is not -0.0");
            }
        }
        ys.push(sr);
    }
    let (first, rest) = ys.split_at_mut(1);
    for other in rest.iter() {
        nccl::add_f32_into(&mut first[0].y_pairs, 0, &other.y_pairs, 0, p * d.hidden, st).unwrap();
    }
    let s0 = &first[0];
    let mut out = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    combine_pairs(&s0.y_pairs, &s0.pick, &s0.wts, Some(&c.shared), &mut out, rows, &d, st).unwrap();
    assert_eq!(bits(&dl::<f32>(&out, st)), bits(&out_ref), "{tag}: out");
}

/// Layer 6 (2-bit), layer 3 (3-bit), layer 45 (the MTP layer, 4-bit): every width of the pack.
const LAYERS: [usize; 3] = [6, 3, 45];

#[test]
fn two_ranks_on_one_gpu_equal_the_single_node_engine_byte_for_byte() {
    gpu!();
    let Some(ck) = checkpoint() else {
        eprintln!("no GLM pack (set REXL3_GLM_DIR): not run");
        return;
    };
    let d = MoeDims::GLM53;
    let st = Stream::new().unwrap();
    for l in LAYERS {
        let lg = load_layer(ck, l, d.experts);
        let whole = MoeExperts::from_groups(&lg.gate, &lg.up, &lg.down, d).unwrap();
        let r0 = bank(&lg, d, ExpertOwnership::range(d.experts, 0, 2).unwrap());
        let r1 = bank(&lg, d, ExpertOwnership::range(d.experts, 1, 2).unwrap());
        assert_eq!((r0.experts_local(), r1.experts_local()), (144, 144));
        assert_eq!((r0.bits_gu, r0.bits_d, r0.plan_gu, r0.plan_d), (whole.bits_gu, whole.bits_d, whole.plan_gu, whole.plan_d));
        // a scattered table (3 ranks, section-16 style): exactness does not depend on contiguous ranges
        let table: Vec<u32> = (0..d.experts as u32).map(|e| (e * 7 + e / 5) % 3).collect();
        let t3: Vec<MoeExperts> = (0..3).map(|r| bank(&lg, d, ExpertOwnership::from_table(&table, r, 3).unwrap())).collect();
        drop(lg);
        // a partial bank refuses the single-node entries
        {
            let mut s = MoeScratch::new(&d, 4).unwrap();
            let c = case(&d, 1, 1);
            s.set_picks(&c.pick, &c.wts).unwrap();
            let mut o = DevBuf::zeroed(d.hidden * 4).unwrap();
            assert!(r0.run(&c.x, Dtype::F16, 1, None, &mut o, &mut s, false, &st).is_err());
            let mut ps = PrefillScratch::new(&d, 4).unwrap();
            ps.set_picks(&c.pick, &c.wts).unwrap();
            assert!(r0.run_prefill(&c.x, Dtype::F16, 1, None, &mut o, &mut ps, &st).is_err());
        }
        for rows in ROWS {
            let c = case(&d, rows, 1000 * l as u64 + rows as u64);
            let t = std::time::Instant::now();
            decode_case(&whole, &[&r0, &r1], &c, rows, &st, &format!("layer {l} decode rows {rows} R=2"));
            decode_case(&whole, &[&whole], &c, rows, &st, &format!("layer {l} decode rows {rows} R=1"));
            batched_case(&whole, &[&r0, &r1], &c, rows, &st, &format!("layer {l} batched rows {rows} R=2"));
            batched_case(&whole, &[&whole], &c, rows, &st, &format!("layer {l} batched rows {rows} R=1"));
            let t3r: Vec<&MoeExperts> = t3.iter().collect();
            decode_case(&whole, &t3r, &c, rows, &st, &format!("layer {l} decode rows {rows} R=3 table"));
            batched_case(&whole, &t3r, &c, rows, &st, &format!("layer {l} batched rows {rows} R=3 table"));
            eprintln!(
                "layer {l} rows {rows}: decode + batched, R = 1, 2, 3: byte-identical ({:.2} s)",
                t.elapsed().as_secs_f64()
            );
        }
        // skewed: every row picks the same 8 experts across the boundary (140..148), so an expert has 22 or 64 pairs: repeated 16-row
        // passes of the decode GEMV and the 2- and 4-tile item classes of the batched path, on rank banks
        for rows in [22usize, 64] {
            let mut c = case(&d, rows, 5000 + rows as u64);
            for (q, p) in c.pick.iter_mut().enumerate() {
                *p = 140 + (q % d.topk) as i32;
            }
            decode_case(&whole, &[&r0, &r1], &c, rows, &st, &format!("layer {l} decode skewed rows {rows} R=2"));
            batched_case(&whole, &[&r0, &r1], &c, rows, &st, &format!("layer {l} batched skewed rows {rows} R=2"));
            eprintln!("layer {l} skewed rows {rows}: decode + batched R = 2: byte-identical");
        }
        // a prefill chunk at serving scale (batched only)
        let rows = 1024;
        let c = case(&d, rows, 9000 + l as u64);
        batched_case(&whole, &[&r0, &r1], &c, rows, &st, &format!("layer {l} batched rows {rows} R=2"));
        eprintln!("layer {l} rows {rows}: batched R = 2: byte-identical");
    }
}

/// Unfilled pairs keep poison in every unowned row. The local combine must read only owned rows and reproduce the
/// same slot-ordered f32 fmaf chain as the single-node pair values.
#[test]
fn unfilled_pairs_and_local_f32_combine_never_read_unowned_rows() {
    gpu!();
    let Some(ck) = checkpoint() else {
        eprintln!("no GLM pack (set REXL3_GLM_DIR): not run");
        return;
    };
    let d = MoeDims::GLM53;
    let st = Stream::new().unwrap();
    let lg = load_layer(ck, 6, d.experts);
    let whole = MoeExperts::from_groups(&lg.gate, &lg.up, &lg.down, d).unwrap();
    let owns: Vec<ExpertOwnership> = (0..3).map(|r| ExpertOwnership::range(d.experts, r, 3).unwrap()).collect();
    let banks: Vec<MoeExperts> = owns.iter().map(|o| bank(&lg, d, o.clone())).collect();
    drop(lg);
    for rows in [1usize, 5, 22, 64] {
        let c = case(&d, rows, 17_000 + rows as u64);
        let wf: Vec<f32> = c.wts.iter().map(|&w| f16_to_f64(w) as f32).collect();
        let cap = rows.max(64);
        let mut reference = PrefillScratch::new(&d, cap).unwrap();
        reference.set_picks_f32(&c.pick, &wf).unwrap();
        let mut out_ref = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
        whole.run_prefill_f32_wts(&c.x, Dtype::F16, rows, None, &mut out_ref, &mut reference, &st).unwrap();
        let y_ref: Vec<f32> = dl(&reference.y_pairs, &st)[..rows * d.topk * d.hidden].to_vec();
        for (ri, ex) in banks.iter().enumerate() {
            let mut sr = PrefillScratch::new(&d, cap).unwrap();
            sr.poison(0x7f, &st).unwrap();
            sr.set_picks_f32(&c.pick, &wf).unwrap();
            let mut os = OwnScratch::new(&d, cap).unwrap();
            ex.run_prefill_pairs_unfilled(&c.x, Dtype::F16, rows, &mut sr, &mut os, &st).unwrap();
            let y: Vec<f32> = dl(&sr.y_pairs, &st)[..rows * d.topk * d.hidden].to_vec();
            let mut expected = vec![0.0f32; rows * d.hidden];
            for r in 0..rows {
                for k in 0..d.topk {
                    let q = r * d.topk + k;
                    let e = c.pick[q];
                    let held = e >= 0 && (e as usize) < d.experts && owns[ri].owns(e as usize);
                    let row = &y[q * d.hidden..(q + 1) * d.hidden];
                    if held {
                        assert_eq!(bits(row), bits(&y_ref[q * d.hidden..(q + 1) * d.hidden]), "rows {rows} rank {ri} pair {q}");
                        for n in 0..d.hidden {
                            let at = r * d.hidden + n;
                            expected[at] = wf[q].mul_add(y_ref[q * d.hidden + n], expected[at]);
                        }
                    } else {
                        assert!(
                            row.iter().all(|v| v.to_bits() == 0x7f7f_7f7f),
                            "unowned pair was filled: rows {rows} rank {ri} pair {q}"
                        );
                    }
                }
            }
            let mut part = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
            combine_pairs_local_f32(&sr.y_pairs, os.pick_local(), &sr.wts_f32, &mut part, rows, &d, &st).unwrap();
            assert_eq!(bits(&dl::<f32>(&part, &st)), bits(&expected), "local combine: rows {rows} rank {ri}");
        }
    }
}

/// The loader's form: ONE region holding only the owned experts (stacked in owned order, the `from_region` layout sized for
/// `E_local`) gives the same bank as `from_groups_owned`, so the same pair bits.
#[test]
fn a_region_of_owned_experts_equals_the_groups_bank() {
    gpu!();
    let Some(ck) = checkpoint() else {
        eprintln!("no GLM pack (set REXL3_GLM_DIR): not run");
        return;
    };
    let d = MoeDims::GLM53;
    let st = Stream::new().unwrap();
    let l = 6;
    let lg = load_layer(ck, l, d.experts);
    let own = ExpertOwnership::range(d.experts, 1, 2).unwrap();
    let (gate, upg, down) = (pick_owned(&lg.gate, &own), pick_owned(&lg.up, &own), pick_owned(&lg.down, &own));
    let el = own.local();
    // build the region on the host: per projection [trellises][suh stack][svh stack], 16-byte aligned
    let mut bytes: Vec<u8> = Vec::new();
    let mut proj = |gs: &[LoadedGroup]| -> ProjOffsets {
        let align = |b: &mut Vec<u8>| b.resize(b.len().div_ceil(256) * 256, 0);
        align(&mut bytes);
        let trellis = bytes.len();
        let stride = gs[0].trellis.len() * 2;
        for g in gs {
            bytes.extend(g.trellis.iter().flat_map(|v| v.to_le_bytes()));
        }
        align(&mut bytes);
        let suh = bytes.len();
        for g in gs {
            bytes.extend(g.suh.iter().flat_map(|&v| f64_to_f16(v).to_le_bytes()));
        }
        align(&mut bytes);
        let svh = bytes.len();
        for g in gs {
            bytes.extend(g.svh.iter().flat_map(|&v| f64_to_f16(v).to_le_bytes()));
        }
        ProjOffsets { trellis, trellis_stride: stride, suh, svh }
    };
    let (pg, pu, pd) = (proj(&gate), proj(&upg), proj(&down));
    let half = |g: &LoadedGroup| (g.group.bits.twice() / 2) as usize;
    let lay = ExpertRegion {
        gate: pg,
        up: pu,
        down: pd,
        bits_gu: half(&gate[0]),
        bits_d: half(&down[0]),
        codebook: gate[0].group.codebook,
    };
    let mut region = DevBuf::zeroed_view(bytes.len()).unwrap();
    region.upload(&bytes).unwrap();
    let from_region = MoeExperts::from_region_owned(region, &lay, d, own.clone()).unwrap();
    let from_groups = MoeExperts::from_groups_owned(&gate, &upg, &down, d, own).unwrap();
    assert_eq!(from_region.experts_local(), el);
    for rows in [1usize, 22] {
        let c = case(&d, rows, 77 + rows as u64);
        let mut ys = Vec::new();
        for ex in [&from_region, &from_groups] {
            let mut s = MoeScratch::new(&d, 64).unwrap();
            s.poison(0x7f, &st).unwrap();
            s.set_picks(&c.pick, &c.wts).unwrap();
            let mut os = OwnScratch::new(&d, 64).unwrap();
            ex.run_pairs(&c.x, Dtype::F16, rows, &mut s, &mut os, &st).unwrap();
            ys.push(bits(&dl::<f32>(&s.y_pairs, &st)[..rows * d.topk * d.hidden]));
        }
        assert_eq!(ys[0], ys[1], "rows {rows}");
    }
}

/// A rank of a group that loaded the WHOLE pack (`OwnScratch::masked`, lane L2b's first two-rank proof of GLM): the full bank with the
/// rank's mask gives, on both paths, exactly the pair buffer of that rank's own bank (owned pairs bit for bit, `-0.0` elsewhere), so
/// the two masked whole banks sum to the single-node pairs and the combine over that sum is the single-node `out`.
#[test]
fn a_whole_bank_with_a_rank_mask_is_that_ranks_share() {
    gpu!();
    let Some(ck) = checkpoint() else {
        eprintln!("no GLM pack (set REXL3_GLM_DIR): not run");
        return;
    };
    let d = MoeDims::GLM53;
    let st = Stream::new().unwrap();
    for l in [6usize, 3] {
        let lg = load_layer(ck, l, d.experts);
        let whole = MoeExperts::from_groups(&lg.gate, &lg.up, &lg.down, d).unwrap();
        let owns: Vec<ExpertOwnership> = (0..2).map(|r| ExpertOwnership::range(d.experts, r, 2).unwrap()).collect();
        let banks: Vec<MoeExperts> = owns.iter().map(|o| bank(&lg, d, o.clone())).collect();
        drop(lg);
        // a mask of another expert count is refused
        assert!(OwnScratch::masked(&d, 8, &ExpertOwnership::range(d.experts + 1, 0, 2).unwrap()).is_err());
        for rows in [1usize, 5, 64] {
            let c = case(&d, rows, 31 * l as u64 + rows as u64);
            let p = rows * d.topk * d.hidden;
            // decode: masked whole bank == rank bank, per rank; sum + combine == single node
            let mut sums: Vec<MoeScratch> = Vec::new();
            for (r, own) in owns.iter().enumerate() {
                let mut a = MoeScratch::new(&d, 64).unwrap();
                a.poison(0x7f, &st).unwrap();
                a.set_picks(&c.pick, &c.wts).unwrap();
                let mut osa = OwnScratch::masked(&d, 64, own).unwrap();
                whole.run_pairs(&c.x, Dtype::F16, rows, &mut a, &mut osa, &st).unwrap();
                let mut b = MoeScratch::new(&d, 64).unwrap();
                b.poison(0x3c, &st).unwrap();
                b.set_picks(&c.pick, &c.wts).unwrap();
                let mut osb = OwnScratch::new(&d, 64).unwrap();
                banks[r].run_pairs(&c.x, Dtype::F16, rows, &mut b, &mut osb, &st).unwrap();
                assert_eq!(
                    bits(&dl::<f32>(&a.y_pairs, &st)[..p]),
                    bits(&dl::<f32>(&b.y_pairs, &st)[..p]),
                    "layer {l} decode rows {rows} rank {r}"
                );
                sums.push(a);
            }
            let (first, rest) = sums.split_at_mut(1);
            nccl::add_f32_into(&mut first[0].y_pairs, 0, &rest[0].y_pairs, 0, p, &st).unwrap();
            let mut out = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
            combine_pairs(&first[0].y_pairs, &first[0].pick, &first[0].wts, Some(&c.shared), &mut out, rows, &d, &st).unwrap();
            let mut s = MoeScratch::new(&d, 64).unwrap();
            s.set_picks(&c.pick, &c.wts).unwrap();
            let mut out_ref = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
            whole.run(&c.x, Dtype::F16, rows, Some(&c.shared), &mut out_ref, &mut s, false, &st).unwrap();
            assert_eq!(bits(&dl::<f32>(&out, &st)), bits(&dl::<f32>(&out_ref, &st)), "layer {l} decode rows {rows}: out");
            // batched: masked whole bank == rank bank, per rank
            for (r, own) in owns.iter().enumerate() {
                let mut a = PrefillScratch::new(&d, 64).unwrap();
                a.poison(0x7f, &st).unwrap();
                a.set_picks(&c.pick, &c.wts).unwrap();
                let mut osa = OwnScratch::masked(&d, 64, own).unwrap();
                whole.run_prefill_pairs(&c.x, Dtype::F16, rows, &mut a, &mut osa, &st).unwrap();
                let mut b = PrefillScratch::new(&d, 64).unwrap();
                b.poison(0x3c, &st).unwrap();
                b.set_picks(&c.pick, &c.wts).unwrap();
                let mut osb = OwnScratch::new(&d, 64).unwrap();
                banks[r].run_prefill_pairs(&c.x, Dtype::F16, rows, &mut b, &mut osb, &st).unwrap();
                assert_eq!(
                    bits(&dl::<f32>(&a.y_pairs, &st)[..p]),
                    bits(&dl::<f32>(&b.y_pairs, &st)[..p]),
                    "layer {l} batched rows {rows} rank {r}"
                );
            }
            eprintln!("layer {l} rows {rows}: masked whole bank == rank bank (decode, batched), sum + combine == single node");
        }
    }
}

/// Cyberfrost's 4096-row TP prefill: the rank-local plan must honor the same selector as the single-GPU path.
/// Also prints the 48-layer serial/parallel timing and the parallel run's stage breakdown on the serving shape.
#[test]
fn cyberfrost_tp_prefill_parallel_plan_bit_identity_and_bench() {
    gpu!();
    let d = MoeDims { hidden: 2560, inter: 640, experts: 512, topk: 10, scale: 1.0, limit: 0.0 };
    let rows = 4096usize;
    let st = Stream::new().unwrap();
    let mk = |k, n, tag: u64| -> Vec<LoadedGroup> {
        (0..d.experts).map(|e| synth_group(k, n, 3, Codebook::Mul1, 20_000 + tag * 1000 + e as u64)).collect()
    };
    let lg = LayerGroups { gate: mk(d.hidden, d.inter, 1), up: mk(d.hidden, d.inter, 2), down: mk(d.inter, d.hidden, 3) };
    let mut route_rng = Rng(20_000);
    let mut pick = Vec::with_capacity(rows * d.topk);
    for _ in 0..rows {
        let mut chosen = Vec::with_capacity(d.topk);
        while chosen.len() < d.topk {
            let u = (route_rng.next_u64() >> 40) as f64 / 16_777_216.0 + 1e-6;
            let e = ((1.0 / u) as usize - 1).min(d.experts - 1);
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        pick.extend(chosen.into_iter().map(|e| e as i32));
    }
    let wts = vec![0.1f32; rows * d.topk];
    let xh: Vec<u16> = (0..rows * d.hidden).map(|i| f64_to_f16(((i % 251) as f64 - 125.0) / 128.0)).collect();
    let x = up(&xh);
    let flush = DevBuf::zeroed(256 << 20).unwrap();
    let mut flush_out = DevBuf::zeroed(16).unwrap();
    for tp in [2usize, 4] {
        let own = ExpertOwnership::range(d.experts, 0, tp).unwrap();
        let el = own.local();
        let ex = bank(&lg, d, own);
        assert_eq!(ex.experts_local(), el);
        let local: Vec<i32> = pick.iter().map(|&e| if (e as usize) < el { e } else { -1 }).collect();
        let mut s = PrefillScratch::new(&d, rows).unwrap();
        let mut os = OwnScratch::new(&d, rows).unwrap();
        s.use_parallel_plan(false);
        let legacy_plan = s.plan(&local, rows, el as i32, &st).unwrap();
        s.use_parallel_plan(true);
        let parallel_plan = s.plan(&local, rows, el as i32, &st).unwrap();
        assert_eq!(legacy_plan, parallel_plan, "TP={tp}: rank-local plan arrays differ");

        let mut part = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
        let mut run = |s: &mut PrefillScratch, os: &mut OwnScratch, parallel: bool| {
            s.use_parallel_plan(parallel);
            s.set_picks_f32(&pick, &wts).unwrap();
            ex.run_prefill_pairs_unfilled(&x, Dtype::F16, rows, s, os, &st).unwrap();
            combine_pairs_local_f32(&s.y_pairs, os.pick_local(), &s.wts_f32, &mut part, rows, &d, &st).unwrap();
            let y = dl::<f32>(&s.y_pairs, &st);
            let out = dl::<f32>(&part, &st);
            (y, out)
        };
        let (legacy_y, legacy_out) = run(&mut s, &mut os, false);
        let (parallel_y, parallel_out) = run(&mut s, &mut os, true);
        assert!(legacy_y.iter().zip(&parallel_y).all(|(a, b)| a.to_bits() == b.to_bits()), "TP={tp}: y_pairs differ");
        assert!(legacy_out.iter().zip(&parallel_out).all(|(a, b)| a.to_bits() == b.to_bits()), "TP={tp}: local output differs");
        drop((legacy_y, legacy_out, parallel_y, parallel_out));

        let names = ["remap", "plan", "rot_in", "gateup", "down", "combine_local_f32"];
        let timer = EventTimer::new(7).unwrap();
        let mut stage = [0f64; 6];
        for it in 0..4 {
            gb_cuda::moe::read_probe(&flush, 256 << 20, &mut flush_out, &st).unwrap();
            timer.record(0, &st).unwrap();
            let mut n = 0;
            ex.run_prefill_pairs_unfilled_marked(&x, Dtype::F16, rows, &mut s, &mut os, &st, &mut |_| {
                n += 1;
                timer.record(n, &st)
            })
            .unwrap();
            combine_pairs_local_f32(&s.y_pairs, os.pick_local(), &s.wts_f32, &mut part, rows, &d, &st).unwrap();
            timer.record(6, &st).unwrap();
            if it > 0 {
                for j in 0..6 {
                    stage[j] += timer.ms(j, j + 1).unwrap() as f64 / 3.0;
                }
            }
        }
        eprintln!(
            "CYBERFROST TP={tp} parallel stages ms: {}",
            names.iter().zip(stage).map(|(n, ms)| format!("{n} {ms:.3}")).collect::<Vec<_>>().join(", ")
        );

        let total_timer = EventTimer::new(2).unwrap();
        let mut totals = [0f64; 2];
        for (arm, parallel) in [false, true].into_iter().enumerate() {
            s.use_parallel_plan(parallel);
            for it in 0..4 {
                total_timer.record(0, &st).unwrap();
                for _ in 0..48 {
                    gb_cuda::moe::read_probe(&flush, 256 << 20, &mut flush_out, &st).unwrap();
                    ex.run_prefill_pairs_unfilled(&x, Dtype::F16, rows, &mut s, &mut os, &st).unwrap();
                    combine_pairs_local_f32(&s.y_pairs, os.pick_local(), &s.wts_f32, &mut part, rows, &d, &st).unwrap();
                }
                total_timer.record(1, &st).unwrap();
                if it > 0 {
                    totals[arm] += total_timer.ms(0, 1).unwrap() as f64 / 3.0;
                }
            }
        }
        eprintln!(
            "CYBERFROST 48-layer TP={tp} rows=4096: serial {:.3} ms, parallel {:.3} ms, speedup {:.2}x",
            totals[0],
            totals[1],
            totals[0] / totals[1]
        );
    }
}

mod fmoe_owned_local {
    use gb_cuda::fmoe::{self, LogitsDtype, Shape, SwigluLimit};
    use gb_cuda::fn_moe::ExpertTables;
    use gb_cuda::rt::{device_available, DevBuf, Stream};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn signed(&mut self) -> f32 {
            (self.next() >> 40) as f32 / ((1u32 << 24) - 1) as f32 * 2.0 - 1.0
        }

        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.next() as u8).collect()
        }
    }

    fn bf16(x: f32) -> u16 {
        let bits = x.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
        (rounded >> 16) as u16
    }

    fn bf16_f32(x: u16) -> f32 {
        f32::from_bits((x as u32) << 16)
    }

    fn upload<T: Copy>(values: &[T]) -> DevBuf {
        let mut out = DevBuf::alloc(std::mem::size_of_val(values)).unwrap();
        out.upload(values).unwrap();
        out
    }

    fn pointers(buffers: &[DevBuf]) -> DevBuf {
        let pointers: Vec<u64> = buffers.iter().map(|b| b.ptr() as usize as u64).collect();
        upload(&pointers)
    }

    struct Fixture {
        _g_t_bufs: Vec<DevBuf>,
        _g_su_bufs: Vec<DevBuf>,
        _g_sv_bufs: Vec<DevBuf>,
        _u_t_bufs: Vec<DevBuf>,
        _u_su_bufs: Vec<DevBuf>,
        _u_sv_bufs: Vec<DevBuf>,
        _d_t_bufs: Vec<DevBuf>,
        _d_su_bufs: Vec<DevBuf>,
        _d_sv_bufs: Vec<DevBuf>,
        g_t: DevBuf,
        g_su: DevBuf,
        g_sv: DevBuf,
        u_t: DevBuf,
        u_su: DevBuf,
        u_sv: DevBuf,
        d_t: DevBuf,
        d_su: DevBuf,
        d_sv: DevBuf,
        ktab: DevBuf,
    }

    impl Fixture {
        fn new(ne: usize, h: usize, i: usize) -> Self {
            let mut rng = Rng(0x4f1a_37c2_90d5_6be8);
            let widths = [2usize, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16];
            let scale_bits = [0x1000u16, 0x1400, 0x1800, 0x1c00, 0x9000, 0x9400, 0x9800, 0x9c00];
            let mut g_t_bufs = Vec::new();
            let mut g_su_bufs = Vec::new();
            let mut g_sv_bufs = Vec::new();
            let mut u_t_bufs = Vec::new();
            let mut u_su_bufs = Vec::new();
            let mut u_sv_bufs = Vec::new();
            let mut d_t_bufs = Vec::new();
            let mut d_su_bufs = Vec::new();
            let mut d_sv_bufs = Vec::new();
            let mut ktab = Vec::new();
            for e in 0..=ne {
                let kg = widths[(3 * e) % widths.len()];
                let ku = widths[(3 * e + 1) % widths.len()];
                let kd = widths[(3 * e + 2) % widths.len()];
                ktab.extend_from_slice(&[kg as i32, ku as i32, kd as i32]);
                g_t_bufs.push(upload(&rng.bytes(h * i * kg / 16)));
                u_t_bufs.push(upload(&rng.bytes(h * i * ku / 16)));
                d_t_bufs.push(upload(&rng.bytes(h * i * kd / 16)));
                let mut scales = |n: usize| {
                    let values: Vec<u16> = (0..n).map(|_| scale_bits[(rng.next() & 7) as usize]).collect();
                    upload(&values)
                };
                g_su_bufs.push(scales(h));
                g_sv_bufs.push(scales(i));
                u_su_bufs.push(scales(h));
                u_sv_bufs.push(scales(i));
                d_su_bufs.push(scales(i));
                d_sv_bufs.push(scales(h));
            }
            let g_t = pointers(&g_t_bufs);
            let g_su = pointers(&g_su_bufs);
            let g_sv = pointers(&g_sv_bufs);
            let u_t = pointers(&u_t_bufs);
            let u_su = pointers(&u_su_bufs);
            let u_sv = pointers(&u_sv_bufs);
            let d_t = pointers(&d_t_bufs);
            let d_su = pointers(&d_su_bufs);
            let d_sv = pointers(&d_sv_bufs);
            let ktab = upload(&ktab);
            Self {
                _g_t_bufs: g_t_bufs,
                _g_su_bufs: g_su_bufs,
                _g_sv_bufs: g_sv_bufs,
                _u_t_bufs: u_t_bufs,
                _u_su_bufs: u_su_bufs,
                _u_sv_bufs: u_sv_bufs,
                _d_t_bufs: d_t_bufs,
                _d_su_bufs: d_su_bufs,
                _d_sv_bufs: d_sv_bufs,
                g_t,
                g_su,
                g_sv,
                u_t,
                u_su,
                u_sv,
                d_t,
                d_su,
                d_sv,
                ktab,
            }
        }

        fn tables(&self) -> ExpertTables<'_> {
            ExpertTables {
                g_t: &self.g_t,
                g_su: &self.g_su,
                g_sv: &self.g_sv,
                u_t: &self.u_t,
                u_su: &self.u_su,
                u_sv: &self.u_sv,
                d_t: &self.d_t,
                d_su: &self.d_su,
                d_sv: &self.d_sv,
                ktab: &self.ktab,
            }
        }
    }

    fn poisoned_workspace(shape: Shape) -> DevBuf {
        let bytes = fmoe::workspace_bytes(shape.max_tokens, shape.top_k, shape.h, shape.i, shape.kc);
        let mut host = vec![0xffu8; bytes];
        let full_shape = Shape { t: shape.max_tokens, ..shape };
        let (y_offset, y_bytes) = fmoe::pair_rows(full_shape);
        // The workspace contract requires its arrival counters to be zero at rest. Everything before them, including all of y, is NaN poison.
        let counters = (y_offset + y_bytes + 255) & !255;
        host[counters..].fill(0);
        upload(&host)
    }

    #[test]
    fn owned_local_matches_full_fmoe() {
        if !device_available() {
            return;
        }
        let st = Stream::new().unwrap();
        let ne = 12;
        let top_k = 3;
        let h = 256;
        let i = 384;
        let max_tokens = 16;
        let fixture = Fixture::new(ne, h, i);
        let tables = fixture.tables();
        let mut rng = Rng(0x9d31_77aa_42c0_1185);
        let x = upload(&(0..max_tokens * h).map(|_| bf16(rng.signed() * 0.25)).collect::<Vec<_>>());
        let logits = upload(&(0..max_tokens * ne).map(|_| bf16(rng.signed() * 2.0)).collect::<Vec<_>>());
        let shared_gate = upload(&(0..max_tokens).map(|_| bf16(rng.signed())).collect::<Vec<_>>());
        let limits = SwigluLimit { routed: 0.7, shared: 0.9 };

        for t in [1, 3, 8, 16] {
            let shape = Shape { t, ne, top_k, h, i, kc: 1, max_tokens };
            let mut reference = DevBuf::alloc(t * h * 2).unwrap();
            let mut reference_ws = fmoe::workspace(max_tokens, top_k, h, i, 1).unwrap();
            fmoe::forward(
                &x,
                &logits,
                LogitsDtype::Bf16,
                &shared_gate,
                &tables,
                &mut reference,
                shape,
                None,
                limits,
                &mut reference_ws,
                &st,
            )
            .unwrap();
            st.sync().unwrap();
            let reference: Vec<u16> = reference.to_vec().unwrap();

            for ranks in [2, 4, 7] {
                let mut sum = vec![0.0f32; t * h];
                for rank in 0..ranks {
                    let ownership: Vec<i32> = (0..=ne).map(|e| (e % ranks == rank) as i32).collect();
                    let own = upload(&ownership);
                    let mut ws = poisoned_workspace(shape);
                    fmoe::forward_pairs_unfilled(
                        &x,
                        &logits,
                        LogitsDtype::Bf16,
                        &shared_gate,
                        &tables,
                        &own,
                        shape,
                        limits,
                        &mut ws,
                        &st,
                    )
                    .unwrap();
                    let mut partial = DevBuf::alloc(t * h * 4).unwrap();
                    fmoe::combine_owned_f32(&ws, &own, &mut partial, shape, &st).unwrap();
                    st.sync().unwrap();
                    let partial: Vec<f32> = partial.to_vec().unwrap();
                    for (dst, src) in sum.iter_mut().zip(partial) {
                        *dst += src;
                    }
                }
                let sum = upload(&sum);
                let mut actual = DevBuf::alloc(t * h * 2).unwrap();
                fmoe::cast_f32_to_bf16(&sum, &mut actual, t * h, &st).unwrap();
                st.sync().unwrap();
                let actual: Vec<u16> = actual.to_vec().unwrap();
                let mut squared_error = 0.0f64;
                let mut squared_reference = 0.0f64;
                for (&expected_bits, &actual_bits) in reference.iter().zip(&actual) {
                    let expected = bf16_f32(expected_bits);
                    let got = bf16_f32(actual_bits);
                    assert!(got.is_finite(), "NaN or infinity at ranks={ranks}, t={t}");
                    let error = (got - expected) as f64;
                    squared_error += error * error;
                    squared_reference += (expected as f64) * (expected as f64);
                }
                let relative_rms = (squared_error / squared_reference.max(1e-30)).sqrt();
                assert!(relative_rms < 1e-5, "relative RMS {relative_rms} at ranks={ranks}, t={t}");
            }
        }
    }

    #[test]
    fn owned_local_is_graph_capturable() {
        if !device_available() {
            return;
        }
        let st = Stream::new().unwrap();
        let (ne, top_k, h, i, max_tokens, t) = (12, 3, 256, 384, 16, 3);
        let shape = Shape { t, ne, top_k, h, i, kc: 1, max_tokens };
        let fixture = Fixture::new(ne, h, i);
        let tables = fixture.tables();
        let mut rng = Rng(0x65d3_118a_f031_90bc);
        let x = upload(&(0..max_tokens * h).map(|_| bf16(rng.signed() * 0.25)).collect::<Vec<_>>());
        let logits = upload(&(0..max_tokens * ne).map(|_| bf16(rng.signed() * 2.0)).collect::<Vec<_>>());
        let shared_gate = upload(&(0..max_tokens).map(|_| bf16(rng.signed())).collect::<Vec<_>>());
        let own = upload(&(0..=ne).map(|e| (e % 2 == 0) as i32).collect::<Vec<_>>());
        let mut ws = poisoned_workspace(shape);
        let mut partial = DevBuf::alloc(t * h * 4).unwrap();
        let mut expected = DevBuf::alloc(t * h * 2).unwrap();
        fmoe::forward_pairs_unfilled(
            &x,
            &logits,
            LogitsDtype::Bf16,
            &shared_gate,
            &tables,
            &own,
            shape,
            SwigluLimit::default(),
            &mut ws,
            &st,
        )
        .unwrap();
        fmoe::combine_owned_f32(&ws, &own, &mut partial, shape, &st).unwrap();
        fmoe::cast_f32_to_bf16(&partial, &mut expected, t * h, &st).unwrap();
        st.sync().unwrap();
        let expected: Vec<u16> = expected.to_vec().unwrap();

        let mut ws = poisoned_workspace(shape);
        let mut partial = DevBuf::alloc(t * h * 4).unwrap();
        let mut actual = DevBuf::alloc(t * h * 2).unwrap();
        let graph = gb_cuda::graph::Graph::capture(&st, || {
            fmoe::forward_pairs_unfilled(
                &x,
                &logits,
                LogitsDtype::Bf16,
                &shared_gate,
                &tables,
                &own,
                shape,
                SwigluLimit::default(),
                &mut ws,
                &st,
            )?;
            fmoe::combine_owned_f32(&ws, &own, &mut partial, shape, &st)?;
            fmoe::cast_f32_to_bf16(&partial, &mut actual, t * h, &st)
        })
        .unwrap();
        for replay in 0..3 {
            graph.launch(&st).unwrap();
            st.sync().unwrap();
            let actual: Vec<u16> = actual.to_vec().unwrap();
            assert_eq!(actual, expected, "CUDA Graph replay {replay}");
        }
    }

    #[test]
    fn owned_local_handles_zero_owned_experts() {
        if !device_available() {
            return;
        }
        let st = Stream::new().unwrap();
        let (ne, top_k, h, i, max_tokens, t) = (12, 3, 256, 384, 16, 1);
        let shape = Shape { t, ne, top_k, h, i, kc: 1, max_tokens };
        let fixture = Fixture::new(ne, h, i);
        let tables = fixture.tables();
        let mut rng = Rng(0x2c17_9180_6e41_bb05);
        let x = upload(&(0..max_tokens * h).map(|_| bf16(rng.signed() * 0.25)).collect::<Vec<_>>());
        let logits = upload(&(0..max_tokens * ne).map(|_| bf16(rng.signed() * 2.0)).collect::<Vec<_>>());
        let shared_gate = upload(&(0..max_tokens).map(|_| bf16(rng.signed())).collect::<Vec<_>>());
        let own = upload(&vec![0i32; ne + 1]);
        let mut ws = poisoned_workspace(shape);
        let mut partial = upload(&vec![f32::NAN; t * h]);
        fmoe::forward_pairs_unfilled(
            &x,
            &logits,
            LogitsDtype::Bf16,
            &shared_gate,
            &tables,
            &own,
            shape,
            SwigluLimit::default(),
            &mut ws,
            &st,
        )
        .unwrap();
        fmoe::combine_owned_f32(&ws, &own, &mut partial, shape, &st).unwrap();
        st.sync().unwrap();
        let partial: Vec<f32> = partial.to_vec().unwrap();
        assert_eq!(partial, vec![0.0; t * h]);
    }
}
