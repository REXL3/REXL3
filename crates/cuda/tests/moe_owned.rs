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

use common::*;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::moe::*;
use gb_cuda::moe_prefill::PrefillScratch;
use gb_cuda::{device_available, nccl, DevBuf, Stream};
use gb_exl3::checkpoint::{scan, Checkpoint, LoadedGroup};
use gb_exl3::f16::f64_to_f16;

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
