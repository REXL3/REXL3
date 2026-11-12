//! The family-neutral half of tensor-parallel sharding: how a family DESCRIBES which dense EXL3 matrices a rank holds in part, and the
//! per-shape PROOF that a cut keeps every output column's bits. A family implements [`ShardPlan`] (the Nemotron family's is
//! `families::nemotron_h_omni::tp_shard::ShardPlan`); the proof is data ([`proof_table`], [`ShapeProof`]) computed from the kernel's own
//! plan mirror, and a device harness ([`compare_slices`]) holds the data against the kernel byte for byte on any [`Exl3Weights`].
//!
//! **Why a proof, not a rule.** TP-exact keeps every output element computed by one rank with the single-node kernel, so a slice of an
//! EXL3 matrix must give the same bytes as the same columns of the whole matrix. An output column accumulates over K alone, and the
//! kernel's K-reduction order is `plan_for(K, N)`: (K splits, warps a block), a function of the SHAPE. A slice keeps the whole matrix's
//! bits exactly when `plan_for(K, n_local) == plan_for(K, N)` ([`slice_keeps_plan`]); otherwise the reduction is regrouped and the bits
//! differ, so the matrix is replicated (every rank holds it whole: it costs the matrix's bytes on every rank and never different bits).
//! The plan depends on the width, so it moves with the rank count: the verdict is per (matrix shape, rank count), not per family.
//!
//! **What a family provides** (`docs/CLUSTER_FAMILY_PORTING.md`): its dense matrices as [`MatrixSpec`]s (name, K, N, bits, a role
//! string), a [`ShardPlan`] that answers [`Cut`] for each, and the tests of this module's harness on its own shapes. Replicating an
//! exact matrix for cost is a plan choice; slicing a matrix whose proof says "replicate" is a bug ([`audit`] finds it on the host).

use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::{DevBuf, Stream};

/// The tile, in output columns, of the trellis and of the output Hadamard.
pub const TILE: usize = 128;

/// The decode kernel's reduction plan `(K splits, warps a block)` of a K x N layer: a mirror of `plan_for` in `csrc/exl3_linear.cu`
/// (and `exl3_big.cu`, which must keep it identical). A family's GPU test holds it against `gb_cuda::exl3_big::big_plan`, the kernel's
/// own answer, for every shape of its pack; this copy exists so a plan can be made without a device.
pub fn decode_plan(k: usize, n: usize) -> (usize, usize) {
    let (blocks, min_tiles) = (192usize, 8usize);
    let (kt, nb) = (k / 16, n / TILE);
    let mut sk = 1usize;
    let mut wk = if nb >= 64 { 8usize } else { 4 };
    if nb < 64 {
        while nb * sk < blocks && sk < 64 {
            sk *= 2;
        }
    }
    while (wk > 4 || sk > 1) && (kt % (sk * wk) != 0 || kt / (sk * wk) < min_tiles) {
        if wk > 4 {
            wk = 4;
        } else if sk > 1 {
            sk /= 2;
        } else {
            break;
        }
    }
    (sk, wk)
}

/// The output columns `[lo, hi)` rank `rank` of `ranks` computes of an `n`-wide matrix: whole 128-column tiles, the cut at
/// `floor(tiles * rank / ranks)` (`gb_exl3::tp::tile_cut`, the engine's rule, [`gb_exl3::tp::SplitMode::Floor`]). The extra tiles of a
/// tile count that does not divide are spread by the floor, so at two ranks the HIGH rank holds the odd tile: 145 tiles are
/// `[0, 9216)` and `[9216, 18560)`, 32 tiles `2048 | 2048`, 2 tiles `128 | 128`; at three ranks 32 tiles are 10 | 11 | 11.
/// **This is the one cut rule of a plan's memory model, its loader and its arms.** `gb_exl3::tp::split_width(.., SplitMode::Uneven)` gives the odd tile to the LOW rank (the mirror image
/// at two ranks); a test here pins this function to `SplitMode::Floor` so the two cannot be mixed up by accident.
///
/// A rank holds NO tile when `n / TILE < ranks` (the 256 columns of k and v at three ranks and more): the range is empty, and
/// [`sliceable`] says so; such a matrix is never sliced.
pub fn column_range(n: usize, rank: usize, ranks: usize) -> (usize, usize) {
    assert!(n % TILE == 0, "{n} is not a whole number of {TILE}-column tiles");
    let (lo, hi) = gb_exl3::tp::tile_cut(n / TILE, rank, ranks);
    (lo * TILE, hi * TILE)
}

/// Whether an `n`-column matrix can be cut into `ranks` slices of at least one whole tile each: a multiple of the tile, and no fewer
/// tiles than ranks. There is no zero-width EXL3 layer, and the kernel-plan mirror cannot see one (`decode_plan(k, 0)` happens to equal
/// the 256-column plan), so [`slice_keeps_plan`] and every plan's cut ask this first.
pub fn sliceable(n: usize, ranks: usize) -> bool {
    ranks >= 2 && n % TILE == 0 && n / TILE >= ranks
}

/// Whether a `k x n` matrix cut into `ranks` tile-aligned column ranges runs each slice on the whole matrix's kernel plan. False when
/// a rank would hold no tile ([`sliceable`]): the plan mirror is blind to a zero-width slice (`decode_plan(k, 0)` equals the plan of a
/// 256-column matrix), and a matrix with no columns on some rank cannot be uploaded as an EXL3 layer. One rank holds whole matrices.
pub fn slice_keeps_plan(k: usize, n: usize, ranks: usize) -> bool {
    if ranks <= 1 {
        return true;
    }
    if !sliceable(n, ranks) {
        return false;
    }
    let whole = decode_plan(k, n);
    (0..ranks).all(|r| {
        let (lo, hi) = column_range(n, r, ranks);
        decode_plan(k, hi - lo) == whole
    })
}

/// Why a `k x n` matrix is replicated across `ranks` ranks instead of sliced (what a plan decides on), `None` when the slices keep
/// every column's bits. Logged at boot so the per-matrix decision is on record.
pub fn replication_reason(k: usize, n: usize, ranks: usize) -> Option<String> {
    if ranks <= 1 || slice_keeps_plan(k, n, ranks) {
        return None;
    }
    if !sliceable(n, ranks) {
        return Some(format!("{} tiles for {ranks} ranks: a rank would hold no column", n / TILE));
    }
    let whole = decode_plan(k, n);
    let (r, plan) = (0..ranks)
        .map(|r| {
            let (lo, hi) = column_range(n, r, ranks);
            (r, decode_plan(k, hi - lo))
        })
        .find(|(_, p)| *p != whole)
        .expect("a sliceable matrix that does not keep its plan has a slice that moves it");
    let (lo, hi) = column_range(n, r, ranks);
    Some(format!(
        "the K reduction moves: rank {r}'s {} tiles run {} K splits x {} warps, the whole {} tiles {} x {}",
        (hi - lo) / TILE,
        plan.0,
        plan.1,
        n / TILE,
        whole.0,
        whole.1
    ))
}

// ------------------------------------------------------------------------------------------------------------------- the description

/// One dense EXL3 matrix a family's plan decides on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixSpec {
    /// The tensor's name (or a layer-independent prefix), for messages.
    pub name: String,
    /// The family's role of the matrix: what its plan keys on (`mamba in_proj`, `attn q_proj`, `mlp down`, ...).
    pub role: String,
    /// Input and output width.
    pub k: usize,
    pub n: usize,
    /// Whole bits per weight; `0` when the spec stands for every width (the plan mirror does not depend on it).
    pub bits: u8,
}

impl MatrixSpec {
    pub fn new(name: impl Into<String>, role: impl Into<String>, k: usize, n: usize, bits: u8) -> MatrixSpec {
        MatrixSpec { name: name.into(), role: role.into(), k, n, bits }
    }
}

/// How one rank holds a matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cut {
    /// Every output column: the rank computes the whole matrix itself (no collective is needed after it).
    Whole,
    /// Output columns `[lo, lo + n_local)`, whole 128-column tiles: the slice is a valid EXL3 layer of `n_local` columns.
    Cols { lo: usize, n_local: usize },
}

impl Cut {
    /// The columns `[lo, hi)` this cut holds of an `n`-wide matrix.
    pub fn range(&self, n: usize) -> (usize, usize) {
        match self {
            Cut::Whole => (0, n),
            Cut::Cols { lo, n_local } => (*lo, lo + n_local),
        }
    }
}

/// A family's decision of which dense matrices a rank of a tensor-parallel group holds in part.
///
/// The contract: a matrix is cut ([`Cut::Cols`]) only where [`slice_keeps_plan`] says the slices run the whole matrix's kernel plan,
/// and exactly at [`column_range`]'s boundaries (the one cut rule of the loader, the arms and the memory plan); every other answer is
/// [`Cut::Whole`]. [`audit`] checks a plan against that on the host; [`compare_slices`] checks the kernel on a device.
///
/// (The Nemotron family's plan is a struct also called `ShardPlan` in `families::nemotron_h_omni::tp_shard`; it implements this trait.)
pub trait ShardPlan {
    /// This rank's place: `(rank, ranks)`.
    fn topology(&self) -> (usize, usize);
    /// How this rank holds `m`.
    fn cut(&self, m: &MatrixSpec) -> Cut;
}

// ------------------------------------------------------------------------------------------------------------------- the proof, as data

/// What the kernel's plan mirror says of cutting a matrix across `ranks` ranks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every rank's slice runs the whole matrix's K reduction: slicing is exact, byte for byte.
    Exact,
    /// A slice would run another K reduction (the bits would differ) or a rank would hold no tile: held whole by every rank. The
    /// text is [`replication_reason`] (what a boot logs).
    Replicate(String),
}

/// The proof for one matrix shape at one rank count: the whole matrix's kernel plan, each rank's slice width and plan, the verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeProof {
    pub matrix: MatrixSpec,
    pub ranks: usize,
    /// `(K splits, warps a block)` of the whole matrix.
    pub whole_plan: (usize, usize),
    /// Per rank, in rank order: `(columns, plan)` of its slice. Empty when the matrix has fewer tiles than ranks (it is never cut).
    pub slices: Vec<(usize, (usize, usize))>,
    pub verdict: Verdict,
}

impl ShapeProof {
    pub fn is_exact(&self) -> bool {
        self.verdict == Verdict::Exact
    }
}

/// The proof of `m` at `ranks` ranks, from the host-side plan mirror.
pub fn prove_shape(m: &MatrixSpec, ranks: usize) -> ShapeProof {
    let whole_plan = decode_plan(m.k, m.n);
    let slices: Vec<(usize, (usize, usize))> = if sliceable(m.n, ranks) {
        (0..ranks)
            .map(|r| {
                let (lo, hi) = column_range(m.n, r, ranks);
                (hi - lo, decode_plan(m.k, hi - lo))
            })
            .collect()
    } else {
        Vec::new()
    };
    let verdict = match replication_reason(m.k, m.n, ranks) {
        None => Verdict::Exact,
        Some(why) => Verdict::Replicate(why),
    };
    ShapeProof { matrix: m.clone(), ranks, whole_plan, slices, verdict }
}

/// The table: one proof for every (matrix, rank count). The rank count 1 is every matrix whole and exact (nothing is cut).
pub fn proof_table(specs: &[MatrixSpec], rank_counts: &[usize]) -> Vec<ShapeProof> {
    rank_counts.iter().flat_map(|&r| specs.iter().map(move |m| prove_shape(m, r))).collect()
}

/// Checks `plan`, for its rank, against the proof of every spec. A violation is a matrix the plan cuts although its proof says the
/// bits would differ (or cuts somewhere other than the engine's cut rule): the message names the matrix. Replicating an exact matrix
/// is the plan's choice (a cost) and is not a violation. Empty: the plan is sound for these shapes at its rank count.
pub fn audit(plan: &dyn ShardPlan, specs: &[MatrixSpec]) -> Vec<String> {
    let (rank, ranks) = plan.topology();
    let mut bad = Vec::new();
    for m in specs {
        match plan.cut(m) {
            Cut::Whole => {}
            Cut::Cols { lo, n_local } => {
                if ranks < 2 {
                    bad.push(format!("{} ({} x {}): a single rank cuts a matrix", m.name, m.k, m.n));
                    continue;
                }
                if !sliceable(m.n, ranks) {
                    bad.push(format!(
                        "{} ({} x {}): cut across {ranks} ranks that have fewer tiles than ranks",
                        m.name, m.k, m.n
                    ));
                    continue;
                }
                let (a, b) = column_range(m.n, rank, ranks);
                if (lo, lo + n_local) != (a, b) {
                    bad.push(format!(
                        "{} ({} x {}): rank {rank} of {ranks} holds columns [{lo}, {}) but the cut rule gives [{a}, {b})",
                        m.name,
                        m.k,
                        m.n,
                        lo + n_local
                    ));
                }
                if let Some(why) = replication_reason(m.k, m.n, ranks) {
                    bad.push(format!(
                        "{} ({} x {}): cut across {ranks} ranks although the proof says replicate: {why}",
                        m.name, m.k, m.n
                    ));
                }
            }
        }
    }
    bad
}

// ------------------------------------------------------------------------------------------------------------------- the proof, on a device

/// A deterministic stream of floats in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// f32 to the bits of the nearest bf16 (ties to even).
fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// `forward` of `w` on `rows` rows of `x` (f32 `[rows, k]`, rounded to bf16 for a bf16 input), f32 out `[rows, n]`: the decode path.
fn run(w: &Exl3Weights, x: &[f32], dtype: Dtype, rows: usize, st: &Stream) -> Result<Vec<f32>, String> {
    let xb = match dtype {
        Dtype::F32 => DevBuf::from_slice(x),
        Dtype::BF16 => DevBuf::from_slice(&x.iter().map(|&v| bf16_bits(v)).collect::<Vec<u16>>()),
        Dtype::F16 => return Err("the arms feed f32 or bf16".into()),
    }
    .map_err(|e| format!("upload: {e}"))?;
    let mut y = DevBuf::zeroed(rows * w.n * 4).map_err(|e| format!("alloc: {e}"))?;
    let mut ws = DevBuf::zeroed(w.workspace_bytes(rows).max(256)).map_err(|e| format!("alloc: {e}"))?;
    w.forward(&xb, &mut y, rows, dtype, Dtype::F32, &mut ws, st).map_err(|e| format!("forward: {e}"))?;
    st.sync().map_err(|e| format!("sync: {e}"))?;
    let out: Vec<f32> = y.to_vec().map_err(|e| format!("download: {e}"))?;
    if out.len() != rows * w.n {
        return Err(format!("{} outputs for {rows} x {}", out.len(), w.n));
    }
    Ok(out)
}

/// What comparing every rank's slice with the whole matrix found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceReport {
    pub k: usize,
    pub n: usize,
    pub ranks: usize,
    /// The matrix could be cut into a tile for every rank (a matrix with fewer tiles than ranks is held whole and is not compared).
    pub sliced: bool,
    /// The plan mirror says every slice keeps the whole matrix's K reduction.
    pub keeps_plan: bool,
    /// Each rank's slice width.
    pub widths: Vec<usize>,
    /// `(rank, rows, mismatched values)` of every comparison.
    pub cells: Vec<(usize, usize, usize)>,
    pub mismatched: usize,
    pub total: usize,
}

impl SliceReport {
    /// The verdict the device must give: a slice whose plan is kept is bit-exact at every row count; one whose plan moves differs
    /// (the negative control: the decision to replicate rests on it, so it must be able to fail).
    pub fn check(&self) -> Result<(), String> {
        if !self.sliced {
            return Ok(());
        }
        if self.keeps_plan {
            if let Some((rank, rows, bad)) = self.cells.iter().find(|c| c.2 != 0) {
                return Err(format!(
                    "rank {rank}'s slice differs from the whole matrix's columns at {rows} rows ({} ranks): {bad} values; the plan \
                     mirror says the K reduction is kept ({} x {})",
                    self.ranks, self.k, self.n
                ));
            }
        } else if self.mismatched == 0 {
            return Err(format!(
                "{} x {} at {} ranks: the kernel plan of the slice differs from the whole matrix's, so the K reduction is regrouped \
                 and the bits must differ (the replicate decision rests on it), but every value matched",
                self.k, self.n, self.ranks
            ));
        }
        Ok(())
    }

    /// One line for a log: what was compared and what was found.
    pub fn line(&self) -> String {
        if !self.sliced {
            "whole on every rank (fewer tiles than ranks)".to_string()
        } else if self.mismatched == 0 {
            format!("BIT-EXACT ({} values)", self.total)
        } else {
            format!("NOT exact: {} of {} output values differ", self.mismatched, self.total)
        }
    }
}

/// Proves slice-vs-whole on a device for one matrix: each of `parts` (rank order, `parts.len()` the rank count) is run on the same
/// rows as `whole` through the decode path (f32 in for `input` F32, bf16 in for BF16; f32 out), and its output is compared with the
/// same columns of the whole matrix's, BIT FOR BIT, at every row count of `rows_list` (the inputs are a deterministic function of
/// `seed` and the row count). `parts[r]` must be the columns [`column_range`] gives rank `r`; a matrix with fewer tiles than ranks
/// is held whole by every rank (each part is then the matrix itself) and is not compared. The report's [`SliceReport::check`] is the
/// verdict against the plan mirror.
pub fn compare_slices(
    whole: &Exl3Weights,
    parts: &[Exl3Weights],
    input: Dtype,
    rows_list: &[usize],
    seed: u64,
    st: &Stream,
) -> Result<SliceReport, String> {
    let ranks = parts.len();
    let (k, n) = (whole.k, whole.n);
    let sliced = sliceable(n, ranks);
    let keeps_plan = slice_keeps_plan(k, n, ranks);
    let mut widths = Vec::with_capacity(ranks);
    for (r, p) in parts.iter().enumerate() {
        let want = if sliced {
            let (lo, hi) = column_range(n, r, ranks);
            hi - lo
        } else {
            n
        };
        if (p.k, p.n) != (k, want) || p.bits != whole.bits {
            return Err(format!(
                "rank {r}'s slice is {} x {} at {} bits, the cut of {k} x {n} at {ranks} ranks wants {k} x {want} at {} bits",
                p.k, p.n, p.bits, whole.bits
            ));
        }
        widths.push(p.n);
    }
    let (mut cells, mut mismatched, mut total) = (Vec::new(), 0usize, 0usize);
    if sliced {
        for &rows in rows_list {
            let mut rng = Lcg(seed ^ rows as u64);
            let x: Vec<f32> = (0..rows * k).map(|_| rng.next()).collect();
            let want = run(whole, &x, input, rows, st)?;
            for (r, part) in parts.iter().enumerate() {
                let got = run(part, &x, input, rows, st)?;
                let (lo, hi) = column_range(n, r, ranks);
                let width = hi - lo;
                let mut bad = 0usize;
                for row in 0..rows {
                    let a = &want[row * n + lo..row * n + hi];
                    let b = &got[row * width..(row + 1) * width];
                    bad += a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                }
                mismatched += bad;
                total += rows * width;
                cells.push((r, rows, bad));
            }
        }
    }
    Ok(SliceReport { k, n, ranks, sliced, keeps_plan, widths, cells, mismatched, total })
}

/// Synthetic EXL3 layers: a whole matrix of random trellis words and scales, and its column slices cut on the host with the stored
/// layout `[K/16, N/16, 16 * bits]` (a 128-column strip is 8 consecutive tile columns of every K row). The harness's own test material:
/// a shape needs no pack, so a family's shapes can be proved before its checkpoint exists.
pub mod synthetic {
    use super::*;
    use gb_exl3::checkpoint::Group;
    use gb_exl3::{Bits, Codebook};

    /// A layer on the host.
    #[derive(Clone)]
    pub struct HostLayer {
        pub k: usize,
        pub n: usize,
        /// Whole bits per weight.
        pub bits: usize,
        pub trellis: Vec<i16>,
        pub suh: Vec<u16>,
        pub svh: Vec<u16>,
    }

    impl HostLayer {
        /// A random layer: uniform trellis words, scales in `[0.5, 1.5)` (fp16).
        pub fn random(k: usize, n: usize, bits: usize, seed: u64) -> HostLayer {
            assert!(k % 128 == 0 && n % 128 == 0 && (1..=8).contains(&bits), "{k} x {n} at {bits} bits");
            let mut s = seed | 1;
            let mut next = move || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            };
            let words = k * n * bits / 16;
            let trellis: Vec<i16> = (0..words).map(|_| (next() >> 24) as i16).collect();
            let mut scale = |len: usize| -> Vec<u16> {
                (0..len).map(|_| gb_exl3::f16::f64_to_f16(0.5 + (next() >> 11) as f64 / (1u64 << 53) as f64)).collect()
            };
            let suh = scale(k);
            let svh = scale(n);
            HostLayer { k, n, bits, trellis, suh, svh }
        }

        /// Output columns `[lo, hi)` as a layer of their own (`lo` and `hi` multiples of 128).
        pub fn columns(&self, lo: usize, hi: usize) -> HostLayer {
            assert!(lo % 128 == 0 && hi % 128 == 0 && lo < hi && hi <= self.n);
            let tile_words = 16 * self.bits; // int16 words a 16 x 16 tile holds
            let (t_lo, t_hi) = (lo / 16, hi / 16);
            let row = self.n / 16 * tile_words;
            let mut trellis = Vec::with_capacity(self.k / 16 * (t_hi - t_lo) * tile_words);
            for kt in 0..self.k / 16 {
                trellis.extend_from_slice(&self.trellis[kt * row + t_lo * tile_words..kt * row + t_hi * tile_words]);
            }
            HostLayer { k: self.k, n: hi - lo, bits: self.bits, trellis, suh: self.suh.clone(), svh: self.svh[lo..hi].to_vec() }
        }

        /// The layer on the device.
        pub fn upload(&self) -> Result<Exl3Weights, String> {
            let g = Group {
                prefix: "synthetic".into(),
                bits: Bits::from_twice((2 * self.bits) as u8).ok_or_else(|| format!("{} bits", self.bits))?,
                codebook: Codebook::Mul1,
                k: self.k,
                n: self.n,
                in_scales: "suh",
                out_scales: "svh",
                bias: false,
                files: Vec::new(),
                bytes: 0,
            };
            let up = |e: gb_cuda::CudaError| format!("upload: {e}");
            Exl3Weights::from_parts(
                &g,
                DevBuf::from_slice(&self.trellis).map_err(up)?,
                DevBuf::from_slice(&self.suh).map_err(up)?,
                DevBuf::from_slice(&self.svh).map_err(up)?,
                None,
            )
            .map_err(|e| format!("from_parts: {e}"))
        }
    }

    /// The whole matrix and each rank's slice at `ranks` ranks (the whole matrix for every rank when it has fewer tiles than ranks).
    pub fn whole_and_parts(h: &HostLayer, ranks: usize) -> Result<(Exl3Weights, Vec<Exl3Weights>), String> {
        let whole = h.upload()?;
        let parts = (0..ranks)
            .map(|r| {
                if sliceable(h.n, ranks) {
                    let (lo, hi) = column_range(h.n, r, ranks);
                    h.columns(lo, hi).upload()
                } else {
                    h.upload()
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((whole, parts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dense shapes of the Nemotron 3.5 Super VL trunk (`tp_shard::dense_shapes` on the published config), as specs: the first
    /// family. And the dense shapes of a Qwen3.8-27B-class layer: a family that has not adopted TP, whose shapes the harness proves
    /// just the same.
    fn nemotron() -> Vec<MatrixSpec> {
        [
            ("mamba in_proj", 4096, 18560),
            ("mamba out_proj", 8192, 4096),
            ("attn q_proj", 4096, 4096),
            ("attn k_proj", 4096, 256),
            ("attn v_proj", 4096, 256),
            ("attn o_proj", 4096, 4096),
        ]
        .into_iter()
        .map(|(role, k, n)| MatrixSpec::new(role, role, k, n, 0))
        .collect()
    }

    fn qwen_dense() -> Vec<MatrixSpec> {
        [
            ("mlp gate_up", 5120, 17408),
            ("mlp down", 17408, 5120),
            ("attn q", 5120, 6144),
            ("attn o", 6144, 5120),
            ("attn kv", 5120, 1024),
        ]
        .into_iter()
        .map(|(role, k, n)| MatrixSpec::new(role, role, k, n, 0))
        .collect()
    }

    fn find<'a>(t: &'a [ShapeProof], role: &str, ranks: usize) -> &'a ShapeProof {
        t.iter().find(|p| p.matrix.role == role && p.ranks == ranks).unwrap_or_else(|| panic!("{role} at {ranks}"))
    }

    #[test]
    fn the_proof_table_of_the_first_family_is_the_documented_one() {
        // docs/NEMOTRON_TP.md section 13.2: which dense matrices are sliced exactly, by rank count, with the kernel's own plans
        let t = proof_table(&nemotron(), &[1, 2, 3, 4, 5, 8]);
        assert_eq!(t.len(), 36);
        // in_proj: whole (1, 8); exact at two ranks (72 | 73 tiles keep it), regrouped from three
        let p = find(&t, "mamba in_proj", 2);
        assert_eq!((p.whole_plan, p.is_exact()), ((1, 8), true));
        assert_eq!(p.slices, vec![(9216, (1, 8)), (9344, (1, 8))]);
        for (r, plan) in [(3, (4, 4)), (4, (8, 4)), (5, (8, 4)), (8, (8, 4))] {
            let p = find(&t, "mamba in_proj", r);
            assert!(!p.is_exact(), "R = {r}");
            assert!(p.slices.iter().any(|s| s.1 == plan), "R = {r}: {:?}", p.slices);
            assert!(matches!(&p.verdict, Verdict::Replicate(w) if w.contains("K reduction moves")), "{:?}", p.verdict);
        }
        // out_proj: (8, 4) whole, a half-width slice runs 16 K splits: replicated at every rank count
        for r in [2, 3, 4, 5, 8] {
            assert!(!find(&t, "mamba out_proj", r).is_exact(), "R = {r}");
        }
        assert_eq!(find(&t, "mamba out_proj", 2).slices, vec![(2048, (16, 4)), (2048, (16, 4))]);
        // q and o: exact through eight ranks; k and v (2 tiles) are exact at two ranks and have no tile for a third rank
        for r in [2, 3, 4, 5, 8] {
            assert!(find(&t, "attn q_proj", r).is_exact() && find(&t, "attn o_proj", r).is_exact(), "R = {r}");
        }
        assert!(find(&t, "attn k_proj", 2).is_exact());
        for r in [3, 4, 5, 8] {
            let p = find(&t, "attn k_proj", r);
            assert!(p.slices.is_empty() && matches!(&p.verdict, Verdict::Replicate(w) if w.contains("no column")), "{p:?}");
        }
        // one rank: nothing is cut, everything exact
        assert!(t.iter().filter(|p| p.ranks == 1).all(|p| p.is_exact() && p.slices.is_empty()));
    }

    #[test]
    fn a_proof_is_a_property_of_the_shape_the_widths_cover_it_once_and_the_verdict_is_the_plans_agreement() {
        let all: Vec<MatrixSpec> = nemotron().into_iter().chain(qwen_dense()).collect();
        for p in proof_table(&all, &[2, 3, 4, 5, 6, 7, 8]) {
            let (k, n, r) = (p.matrix.k, p.matrix.n, p.ranks);
            if sliceable(n, r) {
                assert_eq!(p.slices.iter().map(|s| s.0).sum::<usize>(), n, "{} at {r}", p.matrix.name);
                assert!(p.slices.iter().all(|s| s.0 % TILE == 0 && s.0 > 0));
                // skew: the floor cut gives widths within one tile of each other
                let (mn, mx) = (p.slices.iter().map(|s| s.0).min().unwrap(), p.slices.iter().map(|s| s.0).max().unwrap());
                assert!(mx - mn <= TILE, "{} at {r}: {mn}..{mx}", p.matrix.name);
                let agree = p.slices.iter().all(|s| s.1 == p.whole_plan);
                assert_eq!(p.is_exact(), agree, "{} at {r}", p.matrix.name);
                assert_eq!(p.is_exact(), slice_keeps_plan(k, n, r));
            } else {
                assert!(p.slices.is_empty() && !p.is_exact());
            }
            assert_eq!(p.whole_plan, decode_plan(k, n));
        }
    }

    #[test]
    fn the_cut_rule_is_the_floor_over_tiles() {
        // 145 tiles over two ranks: the high rank holds the odd tile; 32 tiles over three: 10 | 11 | 11
        assert_eq!((column_range(18560, 0, 2), column_range(18560, 1, 2)), ((0, 9216), (9216, 18560)));
        assert_eq!((0..3).map(|r| column_range(4096, r, 3)).map(|(a, b)| (b - a) / TILE).collect::<Vec<_>>(), vec![10, 11, 11]);
        assert!(!sliceable(256, 3) && sliceable(256, 2) && !sliceable(300, 2) && !sliceable(4096, 1));
    }

    /// A plan for the tests: cuts every matrix whose role is not in `whole` (a family's plan reduced to what the audit looks at).
    struct Plan {
        rank: usize,
        ranks: usize,
        whole: Vec<String>,
        cut: fn(usize, usize, usize) -> (usize, usize),
    }

    impl ShardPlan for Plan {
        fn topology(&self) -> (usize, usize) {
            (self.rank, self.ranks)
        }
        fn cut(&self, m: &MatrixSpec) -> Cut {
            if self.whole.contains(&m.role) || !sliceable(m.n, self.ranks) {
                return Cut::Whole;
            }
            let (lo, hi) = (self.cut)(m.n, self.rank, self.ranks);
            Cut::Cols { lo, n_local: hi - lo }
        }
    }

    #[test]
    fn the_audit_accepts_a_plan_that_follows_the_proof_and_names_one_that_does_not() {
        let specs = nemotron();
        for ranks in [2usize, 3, 4, 8] {
            for rank in 0..ranks {
                // the serving plan: replicate exactly what the proof says to
                let whole: Vec<String> =
                    specs.iter().filter(|m| replication_reason(m.k, m.n, ranks).is_some()).map(|m| m.role.clone()).collect();
                let good = Plan { rank, ranks, whole: whole.clone(), cut: column_range };
                assert_eq!(audit(&good, &specs), Vec::<String>::new(), "R = {ranks} rank {rank}");
                // a cost choice: replicating exact matrices too is not a violation
                let mut more = whole;
                more.extend(["attn q_proj".to_string(), "attn o_proj".to_string()]);
                let frugal = Plan { rank, ranks, whole: more, cut: column_range };
                assert_eq!(audit(&frugal, &specs), Vec::<String>::new(), "R = {ranks} rank {rank}");
                // forgetting to replicate what the proof says to is a violation, by name
                let forgot = Plan { rank, ranks, whole: vec!["attn q_proj".to_string()], cut: column_range };
                assert!(audit(&forgot, &specs).iter().any(|s| s.contains("mamba out_proj")), "R = {ranks}");
            }
        }
        // the negative controls: slice everything (out_proj regroups its K reduction at every R, in_proj from R = 3)
        let all = Plan { rank: 1, ranks: 2, whole: vec![], cut: column_range };
        let v = audit(&all, &specs);
        assert_eq!(v.len(), 1, "{v:?}");
        assert!(v[0].contains("mamba out_proj") && v[0].contains("proof says replicate"), "{v:?}");
        let all3 = Plan { rank: 0, ranks: 3, whole: vec![], cut: column_range };
        let v = audit(&all3, &specs);
        assert!(v.iter().any(|s| s.contains("mamba in_proj")) && v.iter().any(|s| s.contains("mamba out_proj")), "{v:?}");
        // a cut that is not the engine's rule is named even where the shape is exact
        let skew = Plan {
            rank: 0,
            ranks: 2,
            whole: vec!["mamba out_proj".to_string(), "mamba in_proj".to_string()],
            cut: |n, _, _| (0, n / 2 + TILE),
        };
        let v = audit(&skew, &specs);
        assert!(v.iter().any(|s| s.contains("cut rule")), "{v:?}");
    }

    /// The dense shapes of three families that have not adopted tensor parallelism (K x N from their `testdata/*/tensor_inventory.tsv`
    /// trellis shapes and configs: `[K/16, N/16, 16 * bits]`), with the verdict by rank count: `E` exact, `M` the K reduction moves
    /// (replicate), `W` fewer tiles than ranks (whole). The doc `docs/CLUSTER_FAMILY_PORTING.md` section 6 is this table.
    fn verdicts(k: usize, n: usize) -> String {
        [2usize, 3, 4, 5, 8]
            .iter()
            .map(|&r| {
                let p = prove_shape(&MatrixSpec::new("m", "m", k, n, 0), r);
                match (&p.verdict, p.slices.is_empty()) {
                    (Verdict::Exact, _) => 'E',
                    (_, true) => 'W',
                    _ => 'M',
                }
            })
            .collect()
    }

    #[test]
    fn the_dense_shapes_of_the_next_families_have_their_proof_tables_on_the_host() {
        // (family, matrix, K, N, whole plan, E/M/W at R = 2, 3, 4, 5, 8)
        type Row = (&'static str, &'static str, usize, usize, (usize, usize), &'static str);
        let rows: [Row; 20] = [
            ("flash-next", "gdn in_proj_qkv", 2560, 10240, (1, 8), "MMMMM"),
            ("flash-next", "gdn in_proj_z", 2560, 6144, (4, 4), "EEEEE"),
            ("flash-next", "gdn out_proj", 6144, 2560, (8, 4), "EEEEE"),
            ("flash-next", "attn q_proj", 2560, 12288, (1, 8), "MMMMM"),
            ("flash-next", "attn k_proj", 2560, 512, (4, 4), "EEEWW"),
            ("flash-next", "mtp fc_hidden", 2560, 2560, (4, 4), "EEEEE"),
            ("glm-5.3-flash", "kda qkv_proj", 4096, 24576, (1, 8), "EEMMM"),
            ("glm-5.3-flash", "kda o_proj", 8192, 4096, (8, 4), "MMMMM"),
            ("glm-5.3-flash", "mla q_a_proj", 4096, 1536, (8, 4), "EEEEE"),
            ("glm-5.3-flash", "mla q_b_proj", 1536, 16384, (1, 8), "EMMMM"),
            ("glm-5.3-flash", "mla kv_a_proj", 4096, 512, (8, 4), "EEEWW"),
            ("glm-5.3-flash", "mla o_proj", 16384, 4096, (8, 4), "MMMMM"),
            ("glm-5.3-flash", "dense mlp gate_up", 4096, 12288, (1, 8), "MMMMM"),
            ("glm-5.3-flash", "dense mlp down", 12288, 4096, (8, 4), "MMMMM"),
            ("glm-5.3-flash", "lm_head", 4096, 154880, (1, 8), "EEEEE"),
            ("qwen3.8-27b", "mlp gate_up", 5120, 17408, (1, 8), "EMMMM"),
            ("qwen3.8-27b", "mlp down", 17408, 5120, (8, 4), "MMMMM"),
            ("qwen3.8-27b", "gdn in_proj_qkv", 5120, 10240, (1, 8), "MMMMM"),
            ("qwen3.8-27b", "gdn out_proj", 6144, 5120, (8, 4), "EEEEE"),
            ("qwen3.8-27b", "attn q_proj", 5120, 12288, (1, 8), "MMMMM"),
        ];
        for (family, name, k, n, whole, want) in rows {
            assert_eq!(decode_plan(k, n), whole, "{family} {name}: the whole matrix's plan");
            assert_eq!(verdicts(k, n), want, "{family} {name} ({k} x {n}) at R = 2, 3, 4, 5, 8");
        }
    }

    // ---- on a device (no pack: synthetic layers), skipped without one

    fn device() -> Option<Stream> {
        if !gb_cuda::rt::device_available() {
            eprintln!("not run: no CUDA device");
            return None;
        }
        Some(Stream::new().expect("stream"))
    }

    #[test]
    fn the_host_strip_slicer_is_the_whole_layers_columns() {
        let h = synthetic::HostLayer::random(256, 1024, 4, 5);
        let s = h.columns(256, 640);
        assert_eq!((s.k, s.n, s.trellis.len(), s.svh.len(), s.suh.len()), (256, 384, 256 / 16 * 384 / 16 * 64, 384, 256));
        // row `kt` of the slice is the whole row's tiles 16..40
        for kt in 0..256 / 16 {
            let a = &h.trellis[(kt * 64 + 16) * 64..(kt * 64 + 40) * 64];
            let b = &s.trellis[kt * 24 * 64..(kt + 1) * 24 * 64];
            assert_eq!(a, b, "kt {kt}");
        }
        assert_eq!(s.svh, h.svh[256..640]);
    }

    #[test]
    fn on_a_device_every_shape_keeps_its_bits_exactly_where_the_proof_says_and_loses_them_where_it_says_so() {
        let Some(st) = device() else { return };
        // two families' shapes, rank counts 2 to 5, a spread of row counts through the decode path
        let shapes: Vec<(usize, usize, Dtype)> = vec![
            (4096, 18560, Dtype::F32), // nemotron in_proj
            (8192, 4096, Dtype::BF16), // nemotron out_proj
            (4096, 4096, Dtype::F32),  // nemotron q, o
            (4096, 256, Dtype::F32),   // nemotron k, v
            (5120, 17408, Dtype::F32), // qwen gate_up
            (17408, 5120, Dtype::F32), // qwen down
        ];
        let rows = [1usize, 5, 33, 128];
        let (mut exact, mut regrouped, mut whole) = (0, 0, 0);
        for (k, n, input) in shapes {
            let h = synthetic::HostLayer::random(k, n, 4, (k + n) as u64);
            for ranks in [2usize, 3, 4, 5] {
                let (w, parts) = synthetic::whole_and_parts(&h, ranks).unwrap();
                let rep = compare_slices(&w, &parts, input, &rows, 0x5eed ^ k as u64, &st).unwrap();
                eprintln!("{k} x {n} R={ranks}: slices {:?}: {}", rep.widths, rep.line());
                rep.check().unwrap_or_else(|e| panic!("{e}"));
                // the harness agrees with the host proof
                let p = prove_shape(&MatrixSpec::new("s", "s", k, n, 4), ranks);
                assert_eq!(rep.keeps_plan, p.is_exact());
                match (rep.sliced, rep.mismatched) {
                    (false, _) => whole += 1,
                    (true, 0) => exact += 1,
                    (true, _) => regrouped += 1,
                }
            }
        }
        // the control fired both ways: some cells exact, some regrouped, some not cut
        assert!(exact > 0 && regrouped > 0 && whole > 0, "exact {exact} regrouped {regrouped} whole {whole}");
    }
}
