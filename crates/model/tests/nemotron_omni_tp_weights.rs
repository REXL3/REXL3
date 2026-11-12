//! TP-W: the per-rank weights of two-rank TP-exact (`docs/NEMOTRON_TP.md`), against the published pack.
//!
//! - `a_slice_of_every_dense_matrix_is_its_columns_of_the_whole_matrix_byte_for_byte`: the proof. For the first and the last Mamba
//!   layer and the first and the last attention layer of the pack, every dense EXL3 linear is uploaded whole and as each rank's slice
//!   (`tp_shard::upload_tp`, the production loader path), run on the same rows through the decode path the arms use
//!   (`Exl3Weights::forward`, f32 in and f32 out, bf16 in for `out_proj`), and the slice's output is compared with the same columns
//!   of the whole matrix's output BIT FOR BIT. It also holds `tp_shard::decode_plan` against the kernel's own plan for every shape.
//!   A shape whose slice runs another K reduction (`out_proj`: 8 K splits whole, 16 on a 2048-wide slice) is expected NOT to match;
//!   that is the negative control and the reason the serving plan replicates it.
//! - `slices_at_three_to_eight_ranks_are_exact_exactly_where_the_kernel_plan_is_kept`: the same proof at the R = 3, 4, 5 and 8 widths
//!   (layers 0 and 7, rows 1, 5, 64, 128): `in_proj` and `out_proj` are not exact (their slices run another K reduction, which is why
//!   the plan replicates them), q and o are exact through eight ranks, k and v (two tiles) are not cut. The plan mirror is held
//!   against `big_plan` at every width of every cut.
//! - `half_the_experts_absent_on_each_rank_sum_to_the_full_layers_expert_outputs`: a MoE layer loaded whole and as each rank's half
//!   (`into_device_tp`: `Proj::Absent` for the other rank's experts), run on the same rows; per slot the owning rank's down-projection
//!   output equals the full layer's, the other rank's is zero, and the two ranks' buffers sum to the full layer's byte for byte.
//!   `experts_of_three_four_and_five_ranks_sum_to_the_full_layers_expert_outputs` is the same for groups of 3, 4 and 5.
//! - `experts_under_an_ownership_table_sum_to_the_full_layers_expert_outputs`: the same proof with the experts owned by a TABLE
//!   (`tp_balance::ExpertMap`, B7) instead of the contiguous ranges, at 2, 3, 4, 5 and 8 ranks: a seeded random table (the worst case)
//!   and, when `REXL3_NEMO_EXPERT_MAPS` names a directory, its `balanced_R<r>.map` files (the solver's), on layers 1, 26, 39 and 81 (the
//!   last three each hold one bf16 expert pair).
//! - `an_lm_head_row_cut_gemv_is_the_whole_gemvs_rows_at_any_rank_count`: the `lm_head` GEMV at every rank count's row cut is the
//!   whole GEMV's rows, and the scatter of the ranks' rows sums to the whole logits.
//! - `the_rank_layers_report_their_local_widths_and_rows`: `into_device_tp` on layers 0..8 and the globals, per rank.
//! - `per_rank_resident_bytes_of_the_published_pack`, `a_tp_rank_is_charged_only_its_weights_and_the_plan_admits_more_slots` and
//!   `resident_weights_and_slots_per_rank_*`: inventory arithmetic, no device (the last two print the rank-count table of the doc).
//!
//! Gated on `REXL3_NEMO_OMNI_DIR` (the pack directory); the device tests also need a CUDA device. Without either the test reports what it
//! did not run and passes. A layer-limited upload: the biggest is about 4 GiB of device memory.

use gb_cuda::exl3_big::big_plan;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::loader::Options;
use gb_cuda::nemotron_moe::MoeScratch;
use gb_cuda::{DevBuf, Stream};
use gb_model::families::nemotron_h_omni::bridge::into_device_tp;
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::device::Mixer;
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::tp::TpTopology;
use gb_model::families::nemotron_h_omni::tp_balance::{moe_layers, shuffled_map, ExpertMap};
use gb_model::families::nemotron_h_omni::tp_shard::{
    column_range, decode_plan, dense_shapes, read_lm_head_rows, slice_keeps_plan, sliceable, upload_tp, Dense, ShardPlan,
};
use gb_model::families::nemotron_h_omni::weights::{self, LayerPlan, UnitId, LM};
use gb_model::shard;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const GIB: f64 = (1u64 << 30) as f64;
const ROWS: [usize; 9] = [1, 2, 5, 16, 17, 33, 64, 100, 128];

fn pack_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set (the published pack directory)");
    }
    dir
}

fn device_pack() -> Option<PathBuf> {
    let dir = pack_dir()?;
    if !gb_cuda::rt::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    Some(dir)
}

struct Pack {
    dir: PathBuf,
    cfg: OmniConfig,
    inv: Inventory,
    plan: LayerPlan,
}

fn open(dir: &Path) -> Pack {
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| panic!("config.json: {e}"));
    let cfg = OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap_or_else(|e| panic!("{e}"));
    let inv = Inventory::read(dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    Pack { dir: dir.to_path_buf(), cfg, inv, plan }
}

fn rank(rank: usize) -> TpTopology {
    TpTopology { rank, ranks: 2 }
}

/// A deterministic stream of floats in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn first_and_last(pattern: &str, kind: char) -> Vec<usize> {
    let all: Vec<usize> = pattern.chars().enumerate().filter(|(_, c)| *c == kind).map(|(i, _)| i).collect();
    vec![*all.first().unwrap(), *all.last().unwrap()]
}

fn prefix(layer: usize, role: Dense) -> String {
    let leaf = match role {
        Dense::MambaIn => "in_proj",
        Dense::MambaOut => "out_proj",
        Dense::AttnQ => "q_proj",
        Dense::AttnK => "k_proj",
        Dense::AttnV => "v_proj",
        Dense::AttnO => "o_proj",
    };
    format!("{LM}.backbone.layers.{layer}.mixer.{leaf}")
}

/// The dtype the arms feed the linear (`arm_mamba`, `arm_attn`).
fn input_dtype(role: Dense) -> Dtype {
    if role == Dense::MambaOut {
        Dtype::BF16
    } else {
        Dtype::F32
    }
}

fn upload_for(p: &Pack, layers: &[usize], shard: &ShardPlan) -> weights::Weights {
    upload_tp(
        &p.dir,
        &p.inv,
        &p.plan,
        p.cfg.experts,
        |u| matches!(u, UnitId::Trunk(i) if layers.contains(&i)),
        &Options::from_env(),
        shard,
    )
    .unwrap_or_else(|e| panic!("upload (rank {}): {e}", shard.topo.rank))
}

/// One cell of the exactness table: a dense matrix of one layer cut into `ranks` slices, run on the decode path against the whole matrix.
struct Cell {
    role: Dense,
    layer: usize,
    ranks: usize,
    /// The matrix was cut (a matrix of fewer tiles than ranks is whole on every rank, in every plan).
    sliced: bool,
    /// The plan mirror says every slice keeps the whole matrix's K reduction.
    keeps: bool,
    mismatched: usize,
    total: usize,
}

/// The slice-vs-whole proof at `ranks` ranks (every dense linear sliced, `ShardPlan::slice_everything`) on `layers` (`mamba` are the
/// Mamba ones, the rest attention), for each row count of `rows_list`: every rank's slice compared with the same columns of the whole
/// matrix, bit for bit, and `decode_plan` held against the kernel's own plan (`big_plan`) at every width the cuts produce.
/// A slice whose plan the mirror says is kept must be bit-exact; one it says moves (the negative control) must differ, or the
/// replicate decision rests on nothing. A matrix with fewer tiles than ranks is held whole by every rank and is not compared.
fn slice_vs_whole(p: &Pack, st: &Stream, ranks: usize, layers: &[usize], mamba: &[usize], rows_list: &[usize]) -> Vec<Cell> {
    // the mirror against the kernel's own plan, for every shape the plan can run: the whole matrices and every rank's slice
    for (role, k, n) in dense_shapes(&p.cfg) {
        assert_eq!(big_plan(k, n).unwrap(), decode_plan(k, n), "{}: the whole {k} x {n}", role.name());
        let mut slice_plans = Vec::new();
        if sliceable(n, ranks) {
            for r in 0..ranks {
                let (lo, hi) = column_range(n, r, ranks);
                assert_eq!(
                    big_plan(k, hi - lo).unwrap(),
                    decode_plan(k, hi - lo),
                    "{}: slice {r} {k} x {}",
                    role.name(),
                    hi - lo
                );
                slice_plans.push((hi - lo, decode_plan(k, hi - lo)));
            }
        }
        eprintln!(
            "plan R={ranks} {:<15} K={k:<5} N={n:<6} whole {:?} slices {:?} -> {}",
            role.name(),
            decode_plan(k, n),
            slice_plans,
            if !sliceable(n, ranks) {
                "fewer tiles than ranks: whole on every rank"
            } else if slice_keeps_plan(k, n, ranks) {
                "same K reduction"
            } else {
                "K reduction differs"
            }
        );
    }

    let mut whole = upload_for(p, layers, &ShardPlan::single());
    let mut slice: Vec<weights::Weights> =
        (0..ranks).map(|r| upload_for(p, layers, &ShardPlan::slice_everything(TpTopology { rank: r, ranks }))).collect();
    let mut cells = Vec::new();
    for &layer in layers {
        let roles: &[Dense] = if mamba.contains(&layer) {
            &[Dense::MambaIn, Dense::MambaOut]
        } else {
            &[Dense::AttnQ, Dense::AttnK, Dense::AttnV, Dense::AttnO]
        };
        for &role in roles {
            let name = prefix(layer, role);
            let w = whole.take_linear(&name).unwrap_or_else(|e| panic!("{e}"));
            let parts: Vec<Exl3Weights> =
                slice.iter_mut().map(|s| s.take_linear(&name).unwrap_or_else(|e| panic!("{e}"))).collect();
            let sliced = sliceable(w.n, ranks);
            let keeps = slice_keeps_plan(w.k, w.n, ranks);
            for (r, part) in parts.iter().enumerate() {
                let want_n = if sliced { column_range(w.n, r, ranks).1 - column_range(w.n, r, ranks).0 } else { w.n };
                assert_eq!((part.k, part.n), (w.k, want_n), "{name}: rank {r}'s slice geometry at {ranks} ranks");
                assert_eq!(part.bits, w.bits);
            }
            // the comparison itself is the family-neutral harness (`shard::compare_slices`): the same inputs (seeded by layer and
            // row count), the same decode path, the same bit-for-bit verdict against the plan mirror
            let report = shard::compare_slices(&w, &parts, input_dtype(role), rows_list, 0x5eed ^ ((layer as u64) << 8), st)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!((report.sliced, report.keeps_plan), (sliced, keeps), "{name}: the harness and the plan mirror disagree");
            if keeps {
                report.check().unwrap_or_else(|e| panic!("{name}: {e}"));
            }
            let (mismatched, total) = (report.mismatched, report.total);
            let widths: Vec<usize> = parts.iter().map(|x| x.n).collect();
            eprintln!(
                "EXACTNESS R={ranks} {:<15} layer {layer:<3} K={:<5} N={:<6} slices {widths:?} rows {rows_list:?}: {}",
                role.name(),
                w.k,
                w.n,
                if !sliced {
                    "whole on every rank (fewer tiles than ranks)".to_string()
                } else if mismatched == 0 {
                    format!("BIT-EXACT ({total} values)")
                } else {
                    format!("NOT exact: {mismatched} of {total} output values differ")
                }
            );
            cells.push(Cell { role, layer, ranks, sliced, keeps, mismatched, total });
        }
    }
    for c in &cells {
        if !c.sliced {
            continue;
        }
        if c.keeps {
            assert_eq!(c.mismatched, 0, "{} layer {} at {} ranks", c.role.name(), c.layer, c.ranks);
        } else {
            assert!(
                c.mismatched > 0,
                "{} layer {} at {} ranks: the kernel plan of the slice differs from the whole matrix's, so the K reduction is regrouped \
                 and the bits must differ (the replicate decision rests on it)",
                c.role.name(),
                c.layer,
                c.ranks
            );
        }
    }
    cells
}

#[test]
fn a_slice_of_every_dense_matrix_is_its_columns_of_the_whole_matrix_byte_for_byte() {
    let Some(dir) = device_pack() else { return };
    let p = open(&dir);
    let st = Stream::new().unwrap();
    let mamba = first_and_last(&p.cfg.pattern, 'M');
    let attn = first_and_last(&p.cfg.pattern, '*');
    let layers: Vec<usize> = mamba.iter().chain(attn.iter()).copied().collect();
    eprintln!("layers: mamba {mamba:?}, attention {attn:?}");
    let cells = slice_vs_whole(&p, &st, 2, &layers, &mamba, &ROWS);
    // the verified two-rank plan: only out_proj moves its K reduction (8 splits whole, 16 on a 2048-wide slice)
    for c in &cells {
        assert!(c.sliced);
        assert_eq!(c.keeps, c.role != Dense::MambaOut, "{}", c.role.name());
        assert_eq!(c.mismatched == 0, c.keeps, "{} layer {}", c.role.name(), c.layer);
    }
}

/// The widths of three, four, five and eight ranks (docs/CLUSTER_N_NODE_PLAN section B3): which dense matrices a slice keeps the whole
/// matrix's bits for on the decode path, on the pack's real weights (layers 0 and 7), rows 1, 5, 64 and 128. Expected, and asserted
/// through the plan mirror and the byte comparison both: `in_proj` is NOT exact at three ranks and more (a slice below 64 tiles runs
/// another K reduction), q and o are exact through eight ranks, `out_proj` is never exact, k and v (2 tiles) are not cut.
#[test]
fn slices_at_three_to_eight_ranks_are_exact_exactly_where_the_kernel_plan_is_kept() {
    let Some(dir) = device_pack() else { return };
    let p = open(&dir);
    let st = Stream::new().unwrap();
    let mamba = vec![0usize];
    assert_eq!(p.cfg.pattern.chars().next(), Some('M'));
    assert_eq!(p.cfg.pattern.chars().nth(7), Some('*'));
    let layers = [0usize, 7];
    let mut table: Vec<(usize, Dense, &'static str)> = Vec::new();
    for ranks in [3usize, 4, 5, 8] {
        let cells = slice_vs_whole(&p, &st, ranks, &layers, &mamba, &[1, 5, 64, 128]);
        assert_eq!(cells.len(), 6);
        for c in &cells {
            let verdict = if !c.sliced {
                "whole (2 tiles)"
            } else if c.mismatched == 0 {
                "exact"
            } else {
                "NOT exact"
            };
            table.push((ranks, c.role, verdict));
            match c.role {
                Dense::MambaIn | Dense::MambaOut => assert!(c.sliced && c.mismatched > 0, "{} at {ranks}", c.role.name()),
                Dense::AttnQ | Dense::AttnO => {
                    assert!(c.sliced && c.mismatched == 0 && c.total > 0, "{} at {ranks}", c.role.name())
                }
                Dense::AttnK | Dense::AttnV => assert!(!c.sliced, "{} at {ranks}", c.role.name()),
            }
        }
    }
    for (ranks, role, verdict) in table {
        eprintln!("TABLE R={ranks} {:<15} {verdict}", role.name());
    }
}

/// The row cut of the `lm_head` GEMV at every rank count (`tp_shard::head_row_cut`: units of four rows, the last rank taking a tail):
/// each rank's rows computed on their own are bitwise the same rows of the whole GEMV, at 1, 5 and 64 rows, for the full vocabulary
/// and for restricted tables of 49,152 rows and an odd 62,017; and the scatter of every rank's rows into `-0.0`-filled buffers sums to
/// the whole logits (the allreduce's identity, signed zeros included). "Any row cut is exact" rests on this.
#[test]
fn an_lm_head_row_cut_gemv_is_the_whole_gemvs_rows_at_any_rank_count() {
    use gb_cuda::nemotron_dense::gemv_bf16;
    use gb_cuda::nemotron_tp::scatter_cols_f32;
    use gb_model::families::nemotron_h_omni::tp_shard::head_row_cut;
    let Some(dir) = device_pack() else { return };
    let p = open(&dir);
    let st = Stream::new().unwrap();
    let name = format!("{LM}.lm_head.weight");
    let only_head = |u: UnitId| u == UnitId::LmHead;
    let mut up =
        upload_tp(&p.dir, &p.inv, &p.plan, p.cfg.experts, only_head, &Options::from_env(), &ShardPlan::single()).unwrap();
    let head = up.take(&name).unwrap().buf;
    let (hidden, row) = (4096usize, 4096 * 2);
    let mut rng = Lcg(0xc0de);
    let x_all: Vec<f32> = (0..64 * hidden).map(|_| rng.next() * 3.0).collect();
    for &m in &[1usize, 5, 64] {
        let x = DevBuf::from_slice(&x_all[..m * hidden]).unwrap();
        for &total in &[p.cfg.vocab, 49_152usize, 62_017] {
            // a table of `total` rows: the first rows of the whole head (a restricted table is a gather of rows, and rows are
            // independent: which rows they are does not matter to the arithmetic)
            let table = head.subview(0, total * row);
            let mut whole = DevBuf::zeroed(m * total * 4).unwrap();
            gemv_bf16(&table, &x, &mut whole, m, total, hidden, &st).unwrap();
            st.sync().unwrap();
            let want: Vec<f32> = whole.to_vec().unwrap();
            for ranks in [2usize, 3, 4, 5, 8] {
                let mut sum: Vec<f32> = vec![f32::from_bits(0x8000_0000); m * total];
                for r in 0..ranks {
                    let (lo, hi) = head_row_cut(total, r, ranks);
                    let n_local = hi - lo;
                    let w = table.subview(lo * row, n_local * row);
                    let mut local = DevBuf::zeroed(m * n_local * 4).unwrap();
                    gemv_bf16(&w, &x, &mut local, m, n_local, hidden, &st).unwrap();
                    // its rows alone are the whole GEMV's rows
                    st.sync().unwrap();
                    let got: Vec<f32> = local.to_vec().unwrap();
                    for mi in 0..m {
                        for j in 0..n_local {
                            assert_eq!(
                                got[mi * n_local + j].to_bits(),
                                want[mi * total + lo + j].to_bits(),
                                "m={m} total={total} ranks={ranks} rank {r}: row {} differs",
                                lo + j
                            );
                        }
                    }
                    // and the scatter of them is the disjoint-support buffer the allreduce sums
                    let mut scattered = DevBuf::zeroed(m * total * 4).unwrap();
                    scatter_cols_f32(&local, &mut scattered, m, n_local, total, lo, &st).unwrap();
                    st.sync().unwrap();
                    let s: Vec<f32> = scattered.to_vec().unwrap();
                    for (acc, v) in sum.iter_mut().zip(&s) {
                        *acc += *v;
                    }
                }
                let same = sum.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "m={m} total={total} ranks={ranks}: the scattered ranks do not sum to the whole GEMV");
            }
        }
        eprintln!("lm_head row cuts at 2, 3, 4, 5 and 8 ranks are bit-exact at {m} rows (131072, 49152 and 62017-row tables)");
    }
}

#[test]
fn half_the_experts_absent_on_each_rank_sum_to_the_full_layers_expert_outputs() {
    let Some(dir) = device_pack() else { return };
    let p = open(&dir);
    let st = Stream::new_nonblocking().unwrap();
    // layer 1 is the first MoE layer; layer 26 holds the measured bf16 expert pair (26, 111), which rank 0 owns
    let layers = [first_and_last(&p.cfg.pattern, 'E')[0], 26usize];
    assert_eq!(p.cfg.pattern.chars().nth(26), Some('E'));
    for &layer in &layers {
        let mut rng = Lcg(0xe7e7 ^ layer as u64);
        let h_all: Vec<f32> = (0..64 * 4096).map(|_| rng.next() * 2.0).collect();
        // (pick, y) of each variant, per row count
        let mut results: Vec<Vec<(Vec<i32>, Vec<f32>)>> = Vec::new();
        for shard in [ShardPlan::single(), ShardPlan::for_config(rank(0), &p.cfg), ShardPlan::for_config(rank(1), &p.cfg)] {
            let pick = |u: UnitId| matches!(u, UnitId::Embed | UnitId::NormF | UnitId::LmHead) || u == UnitId::Trunk(layer);
            let mut up = upload_tp(&p.dir, &p.inv, &p.plan, p.cfg.experts, pick, &Options::from_env(), &shard)
                .unwrap_or_else(|e| panic!("{e}"));
            let w = into_device_tp(&mut up, &p.cfg, &p.plan, layer..layer + 1, &st, &shard).unwrap_or_else(|e| panic!("{e}"));
            drop(up);
            let Mixer::Moe(moe) = &w.layers[0].mixer else { panic!("layer {layer} is not a MoE layer") };
            if !shard.is_single() {
                // the other rank's experts are absent: the prefill tables count them with the bf16 pair as the experts the fast kernels skip
                assert!(moe.slow_experts() >= 256, "rank {}: {} slow experts", shard.topo.rank, moe.slow_experts());
            }
            let mut scratch = MoeScratch::new(64).unwrap();
            let mut per_rows = Vec::new();
            for rows in [1usize, 5, 64] {
                let h = DevBuf::from_slice(&h_all[..rows * 4096]).unwrap();
                let mut out = DevBuf::zeroed(rows * 4096 * 4).unwrap();
                scratch.poison(&st).unwrap();
                moe.forward_in(&mut scratch, &h, rows, &mut out, &st).unwrap_or_else(|e| panic!("moe: {e}"));
                st.sync().unwrap();
                let (pick, _) = scratch.download_route(rows, &st).unwrap();
                let y = scratch.download_y(rows, &st).unwrap();
                per_rows.push((pick, y));
                // informational: what the block costs with half its experts absent (launch to launch, no graph)
                let t0 = std::time::Instant::now();
                for _ in 0..20 {
                    moe.forward_in(&mut scratch, &h, rows, &mut out, &st).unwrap();
                }
                st.sync().unwrap();
                let who = if shard.is_single() { "whole layer".to_string() } else { format!("rank {}", shard.topo.rank) };
                eprintln!(
                    "timing: layer {layer} {who:<11} {rows:>2} rows: {:.3} ms per block",
                    t0.elapsed().as_secs_f64() * 1e3 / 20.0
                );
            }
            results.push(per_rows);
        }
        for (ri, rows) in [1usize, 5, 64].into_iter().enumerate() {
            let (pick_f, y_f) = &results[0][ri];
            let (pick_0, y_0) = &results[1][ri];
            let (pick_1, y_1) = &results[2][ri];
            assert_eq!(pick_f, pick_0, "layer {layer}, {rows} rows: the router is replicated, rank 0 picks the same experts");
            assert_eq!(pick_f, pick_1, "layer {layer}, {rows} rows: rank 1 picks the same experts");
            let width = 1024usize;
            let (mut own0, mut own1) = (0usize, 0usize);
            for (slot, &e) in pick_f.iter().enumerate() {
                let range = slot * width..(slot + 1) * width;
                let (mine, other) = if e < 256 { (y_0, y_1) } else { (y_1, y_0) };
                if e < 256 {
                    own0 += 1;
                } else {
                    own1 += 1;
                }
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
                assert_eq!(
                    bits(&y_f[range.clone()]),
                    bits(&mine[range.clone()]),
                    "layer {layer}, {rows} rows: slot {slot} (expert {e}) differs on its owner"
                );
                assert!(
                    other[range.clone()].iter().all(|v| *v == 0.0),
                    "layer {layer}, {rows} rows: slot {slot} (expert {e}) is not zero on the other rank"
                );
            }
            // the sum of the two ranks' buffers is the full layer's, value for value (x + 0.0 == x; a zero's sign is not a bit anyone reads)
            let canon = |v: f32| if v == 0.0 { 0u32 } else { v.to_bits() };
            for i in 0..y_f.len() {
                assert_eq!(canon(y_0[i] + y_1[i]), canon(y_f[i]), "layer {layer}, {rows} rows: element {i}");
            }
            eprintln!(
                "layer {layer}, {rows:>2} rows: {} slots, {own0} on rank 0 and {own1} on rank 1; y0 + y1 == y_full byte for byte",
                pick_f.len()
            );
            assert!(own0 > 0 || rows > 1, "rank 0 owns no slot of the one-row call");
            if rows == 64 {
                assert!(own0 > 0 && own1 > 0, "the test must exercise both halves");
            }
        }
    }
}

/// The same proof as `half_the_experts_absent_...` for groups of three, four and five ranks: each rank holds experts
/// `[512 r / R, 512 (r + 1) / R)` and writes exact zeros for the pairs of the others; per slot the owner's down-projection output is the
/// whole layer's, every other rank's is zero, and the buffers sum to the whole layer's (`x + 0.0`, the canonical-zero comparison the
/// two-rank test makes). The decode path (`forward_in`) at 1, 5 and 64 rows on layers 1 and 26 (the layer with the bf16 pair).
#[test]
fn experts_of_three_four_and_five_ranks_sum_to_the_full_layers_expert_outputs() {
    let Some(dir) = device_pack() else { return };
    let p = open(&dir);
    let st = Stream::new_nonblocking().unwrap();
    let layers = [first_and_last(&p.cfg.pattern, 'E')[0], 26usize];
    for &layer in &layers {
        let mut rng = Lcg(0xe7e7 ^ layer as u64);
        let h_all: Vec<f32> = (0..64 * 4096).map(|_| rng.next() * 2.0).collect();
        let run_layer = |shard: &ShardPlan| -> Vec<(Vec<i32>, Vec<f32>)> {
            let pick = |u: UnitId| matches!(u, UnitId::Embed | UnitId::NormF | UnitId::LmHead) || u == UnitId::Trunk(layer);
            let mut up = upload_tp(&p.dir, &p.inv, &p.plan, p.cfg.experts, pick, &Options::from_env(), shard)
                .unwrap_or_else(|e| panic!("{e}"));
            let w = into_device_tp(&mut up, &p.cfg, &p.plan, layer..layer + 1, &st, shard).unwrap_or_else(|e| panic!("{e}"));
            drop(up);
            let Mixer::Moe(moe) = &w.layers[0].mixer else { panic!("layer {layer} is not a MoE layer") };
            let mut scratch = MoeScratch::new(64).unwrap();
            let mut per_rows = Vec::new();
            for rows in [1usize, 5, 64] {
                let h = DevBuf::from_slice(&h_all[..rows * 4096]).unwrap();
                let mut out = DevBuf::zeroed(rows * 4096 * 4).unwrap();
                scratch.poison(&st).unwrap();
                moe.forward_in(&mut scratch, &h, rows, &mut out, &st).unwrap_or_else(|e| panic!("moe: {e}"));
                st.sync().unwrap();
                let (pick, _) = scratch.download_route(rows, &st).unwrap();
                per_rows.push((pick, scratch.download_y(rows, &st).unwrap()));
            }
            per_rows
        };
        let full = run_layer(&ShardPlan::single());
        for ranks in [3usize, 4, 5] {
            let parts: Vec<Vec<(Vec<i32>, Vec<f32>)>> =
                (0..ranks).map(|rank| run_layer(&ShardPlan::for_config(TpTopology { rank, ranks }, &p.cfg))).collect();
            let owner_of = |e: usize| {
                (0..ranks)
                    .find(|&r| ShardPlan::for_config(TpTopology { rank: r, ranks }, &p.cfg).owns_expert(e, p.cfg.experts))
                    .expect("every expert has an owner")
            };
            for (ri, rows) in [1usize, 5, 64].into_iter().enumerate() {
                let (pick_f, y_f) = &full[ri];
                for (r, part) in parts.iter().enumerate() {
                    assert_eq!(&part[ri].0, pick_f, "layer {layer}, {ranks} ranks, {rows} rows: rank {r} picks other experts");
                }
                let width = 1024usize;
                let mut owned = vec![0usize; ranks];
                for (slot, &e) in pick_f.iter().enumerate() {
                    let range = slot * width..(slot + 1) * width;
                    let o = owner_of(e as usize);
                    owned[o] += 1;
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
                    assert_eq!(
                        bits(&y_f[range.clone()]),
                        bits(&parts[o][ri].1[range.clone()]),
                        "layer {layer}, {ranks} ranks, {rows} rows: slot {slot} (expert {e}) differs on its owner {o}"
                    );
                    for (r, part) in parts.iter().enumerate() {
                        if r != o {
                            assert!(
                                part[ri].1[range.clone()].iter().all(|v| *v == 0.0),
                                "layer {layer}, {ranks} ranks, {rows} rows: slot {slot} (expert {e}) is not zero on rank {r}"
                            );
                        }
                    }
                }
                let canon = |v: f32| if v == 0.0 { 0u32 } else { v.to_bits() };
                for (i, want) in y_f.iter().enumerate() {
                    let mut acc = f32::from_bits(0x8000_0000);
                    for part in &parts {
                        acc += part[ri].1[i];
                    }
                    assert_eq!(canon(acc), canon(*want), "layer {layer}, {ranks} ranks, {rows} rows: element {i}");
                }
                eprintln!("layer {layer}, {ranks} ranks, {rows:>2} rows: pairs per rank {owned:?}; the {ranks} buffers sum to the whole layer's");
                if rows == 64 {
                    assert!(owned.iter().all(|&n| n > 0), "the test must exercise every rank");
                }
            }
        }
    }
}

/// Ownership by a table: per slot the owner's down-projection output is the whole layer's, every other rank's is zero, and the ranks'
/// buffers sum to the whole layer's. Tables: a seeded shuffle at 2, 3, 4, 5 and 8 ranks, and the solver's `balanced_R<r>.map` files
/// from `REXL3_NEMO_EXPERT_MAPS` when set (a map for another group size is skipped). Decode path, 1, 5 and 64 rows.
#[test]
fn experts_under_an_ownership_table_sum_to_the_full_layers_expert_outputs() {
    let Some(dir) = device_pack() else { return };
    let p = open(&dir);
    let st = Stream::new_nonblocking().unwrap();
    let moe = moe_layers(&p.cfg);
    let layers = [moe[0], 26usize, 39, 81];
    let mut tables: Vec<(String, Arc<ExpertMap>)> = Vec::new();
    for ranks in [2usize, 3, 4, 5, 8] {
        let m = shuffled_map(ranks, p.cfg.experts, &moe, 0xb7 + ranks as u64).unwrap();
        tables.push((format!("shuffled R={ranks}"), Arc::new(m)));
        if let Some(d) = std::env::var_os("REXL3_NEMO_EXPERT_MAPS").map(PathBuf::from) {
            let f = d.join(format!("balanced_R{ranks}.map"));
            if f.is_file() {
                let m = ExpertMap::read(&f).unwrap_or_else(|e| panic!("{e}"));
                m.check_for(&p.cfg, ranks).unwrap_or_else(|e| panic!("{e}"));
                tables.push((format!("balanced R={ranks} ({})", m.describe()), Arc::new(m)));
            }
        }
    }
    for &layer in &layers {
        let mut rng = Lcg(0xe7e7 ^ layer as u64);
        let h_all: Vec<f32> = (0..64 * 4096).map(|_| rng.next() * 2.0).collect();
        let run_layer = |shard: &ShardPlan| -> Vec<(Vec<i32>, Vec<f32>)> {
            let pick = |u: UnitId| matches!(u, UnitId::Embed | UnitId::NormF | UnitId::LmHead) || u == UnitId::Trunk(layer);
            let mut up = upload_tp(&p.dir, &p.inv, &p.plan, p.cfg.experts, pick, &Options::from_env(), shard)
                .unwrap_or_else(|e| panic!("{e}"));
            let w = into_device_tp(&mut up, &p.cfg, &p.plan, layer..layer + 1, &st, shard).unwrap_or_else(|e| panic!("{e}"));
            drop(up);
            let Mixer::Moe(moe) = &w.layers[0].mixer else { panic!("layer {layer} is not a MoE layer") };
            let mut scratch = MoeScratch::new(64).unwrap();
            let mut per_rows = Vec::new();
            for rows in [1usize, 5, 64] {
                let h = DevBuf::from_slice(&h_all[..rows * 4096]).unwrap();
                let mut out = DevBuf::zeroed(rows * 4096 * 4).unwrap();
                scratch.poison(&st).unwrap();
                moe.forward_in(&mut scratch, &h, rows, &mut out, &st).unwrap_or_else(|e| panic!("moe: {e}"));
                st.sync().unwrap();
                let (pick, _) = scratch.download_route(rows, &st).unwrap();
                per_rows.push((pick, scratch.download_y(rows, &st).unwrap()));
            }
            per_rows
        };
        let full = run_layer(&ShardPlan::single());
        for (name, table) in &tables {
            let ranks = table.ranks();
            let plans: Vec<ShardPlan> = (0..ranks)
                .map(|rank| ShardPlan::for_config(TpTopology { rank, ranks }, &p.cfg).with_expert_map(table.clone()))
                .collect();
            // the table's holdings per rank are the contiguous plan's counts (the memory plan does not move)
            for (r, sp) in plans.iter().enumerate() {
                let mine = sp.layer_experts(layer, p.cfg.experts).len();
                let ctg = ShardPlan::for_config(TpTopology { rank: r, ranks }, &p.cfg).experts(p.cfg.experts).len();
                assert_eq!(mine, ctg, "{name}: layer {layer} rank {r}");
            }
            let parts: Vec<Vec<(Vec<i32>, Vec<f32>)>> = plans.iter().map(run_layer).collect();
            for (ri, rows) in [1usize, 5, 64].into_iter().enumerate() {
                let (pick_f, y_f) = &full[ri];
                for (r, part) in parts.iter().enumerate() {
                    assert_eq!(&part[ri].0, pick_f, "{name}, layer {layer}, {rows} rows: rank {r} picks other experts");
                }
                let width = 1024usize;
                let mut owned = vec![0usize; ranks];
                for (slot, &e) in pick_f.iter().enumerate() {
                    let range = slot * width..(slot + 1) * width;
                    let o = table.owner(layer, e as usize).expect("the table covers the layer");
                    owned[o] += 1;
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
                    assert_eq!(
                        bits(&y_f[range.clone()]),
                        bits(&parts[o][ri].1[range.clone()]),
                        "{name}, layer {layer}, {rows} rows: slot {slot} (expert {e}) differs on its owner {o}"
                    );
                    for (r, part) in parts.iter().enumerate() {
                        if r != o {
                            assert!(
                                part[ri].1[range.clone()].iter().all(|v| *v == 0.0),
                                "{name}, layer {layer}, {rows} rows: slot {slot} (expert {e}) is not zero on rank {r}"
                            );
                        }
                    }
                }
                let canon = |v: f32| if v == 0.0 { 0u32 } else { v.to_bits() };
                for (i, want) in y_f.iter().enumerate() {
                    let mut acc = f32::from_bits(0x8000_0000);
                    for part in &parts {
                        acc += part[ri].1[i];
                    }
                    assert_eq!(canon(acc), canon(*want), "{name}, layer {layer}, {rows} rows: element {i}");
                }
                eprintln!("{name}, layer {layer}, {rows:>2} rows: pairs per rank {owned:?}; the {ranks} buffers sum to the whole layer's");
            }
        }
    }
}

#[test]
fn the_rank_layers_report_their_local_widths_and_rows() {
    let Some(dir) = device_pack() else { return };
    let p = open(&dir);
    let st = Stream::new_nonblocking().unwrap();
    let layers = 8usize; // M E M E M E M *
    let pick = |u: UnitId| match u {
        UnitId::Trunk(i) => i < layers,
        UnitId::Embed | UnitId::NormF | UnitId::LmHead => true,
        UnitId::Mtp(_) => false,
    };
    let mut resident = Vec::new();
    for r in 0..2 {
        let shard = ShardPlan::for_config(rank(r), &p.cfg);
        let mut up = upload_tp(&p.dir, &p.inv, &p.plan, p.cfg.experts, pick, &Options::from_env(), &shard)
            .unwrap_or_else(|e| panic!("{e}"));
        let held = up.bytes;
        // the upload holds exactly the rank's plan bytes
        let want = shard.rank_bytes(&p.plan, p.cfg.experts, pick);
        assert_eq!(held, want, "rank {r}: the upload's bytes are the plan's");
        let w = into_device_tp(&mut up, &p.cfg, &p.plan, 0..layers, &st, &shard).unwrap_or_else(|e| panic!("{e}"));
        let (lo, hi) = shard.lm_head_rows(p.cfg.vocab);
        assert_eq!(w.lm_head.bytes(), ((hi - lo) * 4096 * 2) as u64, "rank {r}: lm_head rows [{lo}, {hi})");
        assert_eq!(w.embed.bytes(), p.cfg.vocab * 4096 * 2, "the embedding is replicated");
        for l in &w.layers {
            match &l.mixer {
                Mixer::Mamba(m) => {
                    let (a, b) = shard.columns(Dense::MambaIn, 18560);
                    assert_eq!((m.in_proj.k, m.in_proj.n), (4096, b - a), "rank {r} in_proj");
                    assert_eq!((m.out_proj.k, m.out_proj.n), (8192, 4096), "rank {r} out_proj is replicated");
                }
                Mixer::Attn(a) => {
                    assert_eq!((a.q.n, a.k.n, a.v.n, a.o.n), (2048, 128, 128, 2048), "rank {r} attention");
                    assert_eq!((a.q.k, a.o.k), (4096, 4096));
                }
                Mixer::Moe(m) => assert!(m.slow_experts() >= 256, "rank {r}: the other half is absent"),
            }
        }
        resident.push(held);
        eprintln!("rank {r}: layers 0..{layers} and the globals hold {:.3} GiB (rank_bytes agrees)", held as f64 / GIB);
    }
    let single = ShardPlan::single().rank_bytes(&p.plan, p.cfg.experts, pick);
    assert!(resident[0] < single && resident[1] < single);
    eprintln!("the same layers on one node: {:.3} GiB", single as f64 / GIB);
}

/// SHA-256-free byte comparison of `len` device bytes, in chunks.
fn same_bytes(a: &DevBuf, a_off: usize, b: &DevBuf, b_off: usize, len: usize) -> bool {
    const CHUNK: usize = 32 << 20;
    let (mut x, mut y) = (vec![0u8; CHUNK.min(len)], vec![0u8; CHUNK.min(len)]);
    let mut at = 0;
    while at < len {
        let n = CHUNK.min(len - at);
        a.download_range(a_off + at, &mut x[..n]).unwrap();
        b.download_range(b_off + at, &mut y[..n]).unwrap();
        if x[..n] != y[..n] {
            return false;
        }
        at += n;
    }
    true
}

#[test]
fn a_ranks_lm_head_rows_and_the_rows_read_by_id_are_the_whole_lm_heads_bytes() {
    let Some(dir) = device_pack() else { return };
    let p = open(&dir);
    let name = format!("{LM}.lm_head.weight");
    let row = 4096 * 2;
    let only_head = |u: UnitId| u == UnitId::LmHead;
    let mut whole =
        upload_tp(&p.dir, &p.inv, &p.plan, p.cfg.experts, only_head, &Options::from_env(), &ShardPlan::single()).unwrap();
    let whole = whole.take(&name).unwrap();
    assert_eq!(whole.shape, vec![p.cfg.vocab, 4096]);
    for r in 0..2 {
        let shard = ShardPlan::for_config(rank(r), &p.cfg);
        let mut up = upload_tp(&p.dir, &p.inv, &p.plan, p.cfg.experts, only_head, &Options::from_env(), &shard).unwrap();
        let part = up.take(&name).unwrap();
        let (lo, hi) = shard.lm_head_rows(p.cfg.vocab);
        assert_eq!(part.shape, vec![hi - lo, 4096], "rank {r}");
        assert_eq!(part.buf.bytes(), (hi - lo) * row);
        assert!(
            same_bytes(&part.buf, 0, &whole.buf, lo * row, (hi - lo) * row),
            "rank {r}: rows [{lo}, {hi}) differ from the whole lm_head"
        );
        eprintln!("rank {r}: lm_head rows [{lo}, {hi}) are the whole lm_head's bytes");
    }
    // rows by id, across both halves: runs, singletons, the first and last row
    let mut ids: Vec<u32> = (0..p.cfg.vocab as u32).step_by(1531).collect();
    ids.extend(65_530..65_545);
    ids.extend([0, 1, 2, p.cfg.vocab as u32 - 1]);
    ids.sort_unstable();
    ids.dedup();
    let table = read_lm_head_rows(&p.dir, &p.inv, &p.plan, &ids, &Options::from_env()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(table.bytes(), ids.len() * row);
    for (i, &id) in ids.iter().enumerate() {
        assert!(same_bytes(&table, i * row, &whole.buf, id as usize * row, row), "row of token {id}");
    }
    assert!(read_lm_head_rows(&p.dir, &p.inv, &p.plan, &[5, 5], &Options::from_env()).is_err());
    assert!(read_lm_head_rows(&p.dir, &p.inv, &p.plan, &[p.cfg.vocab as u32], &Options::from_env()).is_err());
    eprintln!("{} rows read by id match the whole lm_head", ids.len());
}

#[test]
fn per_rank_resident_bytes_of_the_published_pack() {
    let Some(dir) = pack_dir() else { return };
    let p = open(&dir);
    let whole = p.plan.bytes(&p.inv);
    let single = ShardPlan::single();
    assert_eq!(single.rank_bytes(&p.plan, p.cfg.experts, |_| true), whole);
    let head = |s: &ShardPlan| s.rank_bytes(&p.plan, p.cfg.experts, |u| matches!(u, UnitId::Mtp(_)));
    eprintln!(
        "whole pack, one node: {:.3} GiB listed, of which the MTP head {:.3} GiB",
        whole as f64 / GIB,
        head(&single) as f64 / GIB
    );
    let mut sums = [0u64; 2];
    for (r, sum) in sums.iter_mut().enumerate() {
        let s = ShardPlan::for_config(rank(r), &p.cfg);
        let all = s.rank_bytes(&p.plan, p.cfg.experts, |_| true);
        let no_head = all - head(&s);
        let experts: u64 = p
            .plan
            .units
            .iter()
            .filter(|u| matches!(u.id, UnitId::Trunk(_)))
            .flat_map(|u| u.tensors.iter().map(move |w| (u, w)))
            .filter(|(u, w)| s.local_bytes(w, &s.piece(u, w, p.cfg.experts)).is_some() && w.name.contains(".mixer.experts."))
            .map(|(u, w)| s.local_bytes(w, &s.piece(u, w, p.cfg.experts)).unwrap())
            .sum();
        eprintln!(
            "rank {r}: {:.3} GiB with the head ({:.3} without); routed experts {:.3} GiB; replicated out_proj included; vs one node {:.3} GiB",
            all as f64 / GIB,
            no_head as f64 / GIB,
            experts as f64 / GIB,
            whole as f64 / GIB
        );
        assert!(all < whole, "a rank holds less than the pack");
        *sum = all;
        // the owned half of the experts is about half of all of them
        assert!(experts > 0);
    }
    // what slicing out_proj as well (the K reduction made identical by the planned-slice entry) would take off each rank
    let mut sliced_out = ShardPlan::slice_everything(rank(0));
    sliced_out.replicate.clear();
    let with_out = sliced_out.rank_bytes(&p.plan, p.cfg.experts, |_| true);
    eprintln!(
        "rank 0 if out_proj were sliced too: {:.3} GiB (saves {:.3} GiB)",
        with_out as f64 / GIB,
        (sums[0] - with_out) as f64 / GIB
    );
    let mut full_head = ShardPlan::for_config(rank(0), &p.cfg);
    full_head.lm_head_full = true;
    eprintln!(
        "rank 0 with the whole lm_head resident: {:.3} GiB (costs {:.3} GiB)",
        full_head.rank_bytes(&p.plan, p.cfg.experts, |_| true) as f64 / GIB,
        (full_head.rank_bytes(&p.plan, p.cfg.experts, |_| true) - sums[0]) as f64 / GIB
    );
}

#[test]
fn a_tp_rank_is_charged_only_its_weights_and_the_plan_admits_more_slots() {
    use gb_model::families::nemotron_h_omni::engine::{plan_serving_with, resident_weight_bytes_tp, PlanInputs};
    use gb_model::families::nemotron_h_omni::memory::NemotronMemory;
    use gb_model::family::FamilyPlan;
    use gb_model::hostmem::MemInfo;
    use gb_model::serve::ServeOptions;

    // no pack needed: the plan is built from the published config's units, the same ones the pack's index is checked against
    let text = include_str!("../../../testdata/nemotron_omni/config.json");
    let cfg = OmniConfig::from_json(&serde_json::from_str(text).unwrap()).unwrap();
    let mut storages = weights::Storages::new();
    for (l, e) in [(26, 111), (39, 304), (81, 461)] {
        for proj in ["up_proj", "down_proj"] {
            storages.insert((l, e, proj), weights::Storage::Bf16);
        }
    }
    let plan = LayerPlan { units: weights::units(&cfg, &storages), outside_plan: 0 };
    let fam = FamilyPlan {
        family: "nemotron_h_omni",
        dir: PathBuf::from("/nonexistent"),
        layers: cfg.layer_specs(),
        mtp_layers: cfg.mtp_specs(),
        memory: Box::new(NemotronMemory::new(&cfg, 0)),
        vision: None,
        chat_format: "nemotron_omni",
        drafters: vec!["copy"],
        required_components: vec![],
    };
    let mut o = ServeOptions::new(PathBuf::from("/nonexistent"));
    o.prefill_chunk = 64;
    o.floor_bytes = 8 << 30;
    o.slots = 1024;
    o.slots_auto = true;
    let avail = 100u64 << 30;
    let host = MemInfo { total: 121 << 30, free: avail, available: avail, cached: 0 };
    let single = resident_weight_bytes_tp(&plan, cfg.experts, &ShardPlan::single(), &fam, &o);
    let rank0 = resident_weight_bytes_tp(&plan, cfg.experts, &ShardPlan::for_config(rank(0), &cfg), &fam, &o);
    let rank1 = resident_weight_bytes_tp(&plan, cfg.experts, &ShardPlan::for_config(rank(1), &cfg), &fam, &o);
    assert!(rank0 < single && rank1 < single);
    let slots = |weights: u64| {
        let inp =
            PlanInputs { cuda_free: host.free, host: Some(&host), weight_bytes: weights, model_max_context: cfg.max_context };
        plan_serving_with(&fam, &o, &inp, 0).unwrap().slots
    };
    let (s1, s0, s1r) = (slots(single), slots(rank0), slots(rank1));
    eprintln!(
        "weights: one node {:.2} GiB, rank 0 {:.2} GiB, rank 1 {:.2} GiB; slots admitted from {:.0} GiB: {s1} on one node, {s0} and {s1r} per rank",
        single as f64 / GIB,
        rank0 as f64 / GIB,
        rank1 as f64 / GIB,
        avail as f64 / GIB
    );
    assert!(s0 > s1 && s1r > s1, "a rank that holds half the weights admits more slots: {s0} and {s1r} against {s1}");
}

/// What the fullest rank of a group of `ranks` holds and what `plan_serving` admits from `avail` bytes of free memory (the 8 GiB floor,
/// the MTP head priced with its default 49,152-row restricted table, the serving plan: attention replicated, the head's experts split,
/// the TP scratch priced at the group's cuts). `ranks == 1` is one node.
struct GroupRow {
    ranks: usize,
    weights: u64,
    slots: usize,
    kv_tokens: u64,
    max_context: usize,
}

fn group_row(cfg: &OmniConfig, plan: &LayerPlan, ranks: usize, avail: u64) -> GroupRow {
    use gb_model::families::nemotron_h_omni::engine::{
        plan_serving_with, resident_weight_bytes_group, sub_head_bytes, PlanInputs,
    };
    use gb_model::families::nemotron_h_omni::memory::NemotronMemory;
    use gb_model::family::FamilyPlan;
    use gb_model::hostmem::MemInfo;
    use gb_model::serve::{ServeOptions, SpecMode};

    let shards: Vec<ShardPlan> = if ranks == 1 {
        vec![ShardPlan::single()]
    } else {
        (0..ranks)
            .map(|rank| {
                ShardPlan::for_config(TpTopology { rank, ranks }, cfg).replicating(&[
                    Dense::AttnQ,
                    Dense::AttnK,
                    Dense::AttnV,
                    Dense::AttnO,
                ])
            })
            .collect()
    };
    let memory = if ranks == 1 { NemotronMemory::new(cfg, 0) } else { NemotronMemory::new(cfg, 0).with_tp_ranks(ranks) };
    let fam = FamilyPlan {
        family: "nemotron_h_omni",
        dir: PathBuf::from("/nonexistent"),
        layers: cfg.layer_specs(),
        mtp_layers: cfg.mtp_specs(),
        memory: Box::new(memory),
        vision: None,
        chat_format: "nemotron_omni",
        drafters: vec!["copy", "mtp"],
        required_components: vec![],
    };
    let mut o = ServeOptions::new(PathBuf::from("/nonexistent"));
    o.prefill_chunk = 64;
    o.floor_bytes = 8 << 30;
    o.slots = 1024;
    o.slots_auto = true;
    o.spec = SpecMode::Auto;
    o.drafting = true;
    let weights = resident_weight_bytes_group(plan, cfg.experts, &shards, &fam, &o);
    let host = MemInfo { total: 121 << 30, free: avail, available: avail, cached: 0 };
    let inp = PlanInputs { cuda_free: host.free, host: Some(&host), weight_bytes: weights, model_max_context: cfg.max_context };
    let sp = plan_serving_with(&fam, &o, &inp, sub_head_bytes(49_152)).unwrap_or_else(|e| panic!("{ranks} ranks: {e}"));
    GroupRow { ranks, weights, slots: sp.slots, kv_tokens: sp.plan.kv_tokens, max_context: sp.max_context }
}

fn print_group_table(what: &str, rows: &[GroupRow]) {
    eprintln!("{what}: resident weights of the fullest rank and what plan_serving admits from 100 GiB free (8 GiB floor)");
    eprintln!("| ranks | weights per rank (GiB) | slots | KV tokens | max context |");
    eprintln!("| ---: | ---: | ---: | ---: | ---: |");
    for r in rows {
        eprintln!("| {} | {:.2} | {} | {} | {} |", r.ranks, r.weights as f64 / GIB, r.slots, r.kv_tokens, r.max_context);
    }
}

/// The rank-count table of the doc, from the committed config (no pack): the weights per rank fall and the plan admits at least as many
/// slots as the group grows, and two ranks is the plan the verified runs carry.
#[test]
fn resident_weights_and_slots_per_rank_at_one_to_eight_ranks() {
    let text = include_str!("../../../testdata/nemotron_omni/config.json");
    let cfg = OmniConfig::from_json(&serde_json::from_str(text).unwrap()).unwrap();
    let mut storages = weights::Storages::new();
    for (l, e) in [(26, 111), (39, 304), (81, 461)] {
        for proj in ["up_proj", "down_proj"] {
            storages.insert((l, e, proj), weights::Storage::Bf16);
        }
    }
    let plan = LayerPlan { units: weights::units(&cfg, &storages), outside_plan: 0 };
    let avail = 100u64 << 30;
    let rows: Vec<GroupRow> = [1usize, 2, 3, 4, 5, 8].into_iter().map(|r| group_row(&cfg, &plan, r, avail)).collect();
    print_group_table("committed config, three bf16 expert pairs", &rows);
    for w in rows.windows(2) {
        assert!(
            w[1].weights < w[0].weights,
            "{} ranks hold {} bytes, {} ranks {}",
            w[0].ranks,
            w[0].weights,
            w[1].ranks,
            w[1].weights
        );
        assert!(w[1].slots >= w[0].slots, "{} ranks admit {} slots, {} ranks {}", w[0].ranks, w[0].slots, w[1].ranks, w[1].slots);
    }
    // two ranks: the fullest rank is rank 1 (the odd in_proj tile and the larger half of the vocabulary's units); the head is priced
    // (unless the environment says REXL3_MTP=off, which the pinned figure does not describe)
    // (the numbers are what the tree before the rank-generic sharding, 921d8d5, planned from the same config: one node 68,294,164,008
    // bytes and 128 slots; two ranks, rank 1 the fuller, 35,836,917,808 bytes and 316 slots at 100 GiB)
    // Since rc2d the default contract is P2 with the Mamba head split on, so a rank of two also holds the head-group in_proj
    // (766,535,840 bytes, `tp_shard` pins) and its exchange scratch: 311 slots instead of 316 at 100 GiB (re-frozen from this run).
    // Under P1 (REXL3_NEMO_PREFILL_CONTRACT=p1) or with the head split off the 921d8d5 numbers hold.
    use gb_model::families::nemotron_h_omni::contract;
    let p2 = contract::current() == contract::PrefillContract::P2;
    let hg = p2 && contract::p2tp_on(contract::MAMBA_HG_VAR);
    if std::env::var_os("REXL3_MTP").is_none() {
        assert_eq!((rows[0].weights, rows[0].slots), (68_294_164_008, 128), "one node");
        if hg {
            assert_eq!(
                (rows[1].weights, rows[1].slots),
                (35_836_917_808 + 766_535_840, 311),
                "two ranks, P2 with the head split"
            );
        } else if !p2 {
            assert_eq!((rows[1].weights, rows[1].slots), (35_836_917_808, 316), "the two-rank plan the verified runs carry");
        } else {
            assert_eq!(rows[1].weights, 35_836_917_808, "two ranks, P2 without the head split");
        }
        assert_eq!((rows[0].kv_tokens, rows[1].kv_tokens), (262_144, 262_144));
    }
}

/// The same table from the published pack's inventory (CPU only; needs `REXL3_NEMO_OMNI_DIR`).
#[test]
fn resident_weights_and_slots_per_rank_of_the_published_pack() {
    let Some(dir) = pack_dir() else { return };
    let p = open(&dir);
    let avail = 100u64 << 30;
    let rows: Vec<GroupRow> = [1usize, 2, 3, 4, 5, 8].into_iter().map(|r| group_row(&p.cfg, &p.plan, r, avail)).collect();
    print_group_table("published pack", &rows);
    for w in rows.windows(2) {
        assert!(w[1].weights < w[0].weights && w[1].slots >= w[0].slots);
    }
}
