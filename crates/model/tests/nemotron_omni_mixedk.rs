//! v0.1.2 MixedK (vcruz305/Nemotron-Super-VL-EXL3-3.85bpw @39638152): the plan of the real pack, and its weights module by module.
//!
//! `REXL3_NEMO_MIXEDK_DIR` names the MixedK pack's directory; without it every test says what it did not run and passes. The plan test
//! reads the index and the shard headers only (no tensor data, no GPU).
//!
//! The pack (exllamav3 1.5.1.post1, SAGE MixedK): text names without `language_model.`; routed experts K 2 to 6, mixed within a layer and
//! within an expert; the other trunk linears K 4 to 8; lm_head EXL3 K8; the MTP head fully EXL3 K4 (attention and eh_proj included);
//! the router gates F16 and the trunk's latent projections F16 dense, which this engine does not serve yet (refused by name).

use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::inventory::{Inventory, Layout};
use gb_model::families::nemotron_h_omni::weights::{widths_of, LayerPlan};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn mixedk_dir() -> Option<PathBuf> {
    let d = std::env::var_os("REXL3_NEMO_MIXEDK_DIR").map(PathBuf::from);
    if d.is_none() {
        eprintln!("not run: REXL3_NEMO_MIXEDK_DIR is not set (the MixedK pack's directory)");
    }
    d
}

fn config(dir: &std::path::Path) -> OmniConfig {
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| panic!("config.json: {e}"));
    OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap_or_else(|e| panic!("{e}"))
}

/// The class of a refusal: the tensor's role with the layer and expert numbers taken out.
fn class(msg: &str) -> String {
    let mut out = String::new();
    let mut digits = false;
    for c in msg.chars() {
        if c.is_ascii_digit() {
            if !digits {
                out.push('N');
            }
            digits = true;
        } else {
            digits = false;
            out.push(c);
        }
    }
    out
}

/// The real pack: its layout is the bare one, every trellis is a whole width the engine serves (the census below is the pack's), and
/// the plan takes the whole pack: its last two unserved forms (lane L's census of refusals), the 41 F16 router gates and the 80 F16 dense
/// latent projections, are planned as F16 since lane K's fp16 gate and dense paths. A refusal would print by class.
#[test]
fn the_mixedk_pack_is_read_in_its_layout_at_its_widths_and_planned_in_full() {
    let Some(dir) = mixedk_dir() else { return };
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(inv.layout, Layout::Bare);
    let (widths, errors) = widths_of(&inv);
    assert!(errors.is_empty(), "{errors:?}");
    let mut census: BTreeMap<usize, usize> = BTreeMap::new();
    for b in widths.values() {
        *census.entry(*b).or_default() += 1;
    }
    eprintln!("EXL3 linears by width: {census:?}");
    assert_eq!(widths.len(), 42_186, "every trellis of the pack");
    let plan = match LayerPlan::build(&cfg, &inv) {
        Ok(p) => p,
        Err(errors) => {
            let mut classes: BTreeMap<String, usize> = BTreeMap::new();
            for e in &errors {
                *classes.entry(class(e)).or_default() += 1;
            }
            for (c, n) in &classes {
                eprintln!("{n:6} x {c}");
            }
            panic!("the MixedK pack's plan is refused ({} refusals, first: {})", errors.len(), errors[0]);
        }
    };
    let wants = plan.tensors(|_| true);
    let f16 = |suffix: &str| wants.iter().filter(|w| w.dtype == "F16" && w.name.ends_with(suffix)).count();
    assert_eq!(f16(".gate.weight"), 41, "41 F16 router gates (trunk and head)");
    assert_eq!(f16("_latent_proj.weight"), 80, "80 F16 trunk latent projections");
    // the family probe takes the pack (its layout is read, not refused as "EXL3 packs only")
    gb_model::families::nemotron_h_omni::expert_storage(&dir, &cfg)
        .unwrap_or_else(|e| panic!("the probe refuses the MixedK pack: {e}"));
    eprintln!("the MixedK pack plans in full: {} tensors, 41 F16 gates, 80 F16 latents", wants.len());
}

// ---------------------------------------------------------------------------------------------------------------------------------
// GPU: the MixedK pack's weights on the device, module by module, against the host decode (R-MK-rails (b)). Needs a CUDA device and
// `REXL3_NEMO_MIXEDK_DIR`; without either each test says so and passes. The pack cannot run end to end on this branch (its F16 router
// gates and F16 dense latent projections are lane K's), so the modules that can are checked one by one: every dense EXL3 width of the
// pack (trunk and head attention), the EXL3 lm_head through the engine's loader whole and as a TP rank's slice, its restricted rows
// (decision D2), and an MoE block of K 2 to 6 experts on a synthetic bf16 gate.
//   cargo test -p gb-model --release --test nemotron_omni_mixedk gpu_ -- --nocapture --test-threads=1

use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::loader::Options;
use gb_cuda::nemotron_moe::{self, MoeScratch, Proj as DevProj};
use gb_cuda::nemotron_moe_prefill::PrefillMoe;
use gb_cuda::{DevBuf, Stream};
use gb_exl3::checkpoint::{scan, Checkpoint, LoadedGroup};
use gb_exl3::f16::f64_to_bf16;
use gb_exl3::{codebook, decode_strip, Codebook};
use gb_model::families::nemotron_h_omni::bridge::{into_device, into_device_tp};
use gb_model::families::nemotron_h_omni::device::HeadLinear;
use gb_model::families::nemotron_h_omni::model::linear_rows;
use gb_model::families::nemotron_h_omni::moe_ref::{latent_moe, route, Dense, Exl3 as RefExl3, Mlp, Proj as RefProj, Router};
use gb_model::families::nemotron_h_omni::mtp::{restricted_vocab_ids_for, VocabSetting};
use gb_model::families::nemotron_h_omni::tp::{SingleRank, TpTopology};
use gb_model::families::nemotron_h_omni::tp_arm::{gather_head_rows, lm_head_logits};
use gb_model::families::nemotron_h_omni::tp_shard::{upload_tp, ShardPlan};
use gb_model::families::nemotron_h_omni::weights::{self, head_exl3_of, storages_of, units_with, UnitId};

fn gpu_dir() -> Option<PathBuf> {
    let dir = mixedk_dir()?;
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    Some(dir)
}

/// A deterministic fill in [-1, 1) on the bf16 grid (the kernels' committed limit is for inputs exact in the activation format, as
/// `exl3_linear.rs`'s gates encode theirs; f32 activations add their own rounding to the rotated input).
fn bf16_noise(seed: u64, n: usize) -> Vec<f32> {
    noise(seed, n).into_iter().map(|v| f32::from_bits((f64_to_bf16(v as f64) as u32) << 16)).collect()
}

/// A deterministic fill in [-1, 1).
fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The device decode path against the float64 host decode, per row: the relative L2 error of the row (at most [`ROW_REL_L2`]) and the
/// worst element's error in units of `2^-9 rms(row)` (at most [`ELEM_UNITS`]). The kernel's own gates hold `2^-23 |want| + 2^-9 rms` on
/// their synthetic layers; the packs' real scales spread the fp16 rotated input's rounding wider on a few elements (our K = 4 linears
/// too: `gpu_the_same_harness_on_our_pack_calibrates_the_limits`), so the gate here is the row's L2 with a loose element bound. A wrong
/// decode (a width, a tile order, a scale) is wrong by O(1). Returns `(worst row rel L2, worst element units)`.
fn dense_check(got: &[f32], want: &[f64], rows: usize, n: usize, what: &str) -> (f64, f64) {
    assert_eq!((got.len(), want.len()), (rows * n, rows * n), "{what}");
    let (mut worst_l2, mut worst_units) = (0f64, 0f64);
    for r in 0..rows {
        let (g, w) = (&got[r * n..(r + 1) * n], &want[r * n..(r + 1) * n]);
        let ss = w.iter().map(|v| v * v).sum::<f64>();
        let rms = (ss / n as f64).sqrt();
        let err2: f64 = g.iter().zip(w).map(|(&a, &b)| (a as f64 - b) * (a as f64 - b)).sum();
        let l2 = (err2 / ss).sqrt();
        let units = g.iter().zip(w).map(|(&a, &b)| (a as f64 - b).abs() / (2f64.powi(-9) * rms)).fold(0f64, f64::max);
        assert!(l2 <= ROW_REL_L2, "{what}: row {r}: relative L2 {l2:.3e} (limit {ROW_REL_L2:e})");
        assert!(units <= ELEM_UNITS, "{what}: row {r}: an element is {units:.2} x 2^-9 rms off (limit {ELEM_UNITS})");
        worst_l2 = worst_l2.max(l2);
        worst_units = worst_units.max(units);
    }
    (worst_l2, worst_units)
}

/// A row's relative L2 error to the float64 host decode: `2^-9`, the `c` of the dense kernel's own gate
/// (`crates/cuda/src/exl3_linear.rs` `compare`: `|err| <= eps_out |want| + 2^-9 rms(row)` per element) taken over the row. Set by lane L for
/// v0.1.2 (2026-10-17) after the calibration on our pack's K4 linears (2.1e-4 to 3.1e-4, a tenfold margin; the MixedK pack's dense linears
/// and its lm_head over whole rows 2.0e-4 to 2.4e-4).
const ROW_REL_L2: f64 = 1.953_125e-3;
/// A row's worst element, in units of `2^-9 rms(row)`: the kernel gate's element bound, which real scales exceed by up to 1.8 units on
/// our own K4 q_proj and 2.6 on the MixedK head's o_proj (a few elements of the fp16 rotated input's rounding), widened to 4.
const ELEM_UNITS: f64 = 4.0;
/// The MoE block's limit: max |got - want| / max |want| <= 1e-3, the committed gate of the block against the CPU reference
/// (`crates/cuda/tests/nemotron_moe.rs`, and `bridge::tests::BLOCK_TOL` of the bridge's GPU gate).
const BLOCK_TOL: f64 = 1e-3;

/// `y = layer(x)` on the device for `rows` rows (f32 in and out), through the trunk's dense path.
fn device_forward(l: &Exl3Weights, x: &[f32], rows: usize, st: &Stream) -> Vec<f32> {
    let xd = DevBuf::from_slice(x).unwrap();
    let mut y = DevBuf::zeroed_view(rows * l.n * 4).unwrap();
    let mut ws = DevBuf::zeroed(l.workspace_bytes(rows.min(128)).max(1)).unwrap();
    linear_rows(l, &xd, &mut y, rows, Dtype::F32, Dtype::F32, &mut ws, st).unwrap();
    st.sync().unwrap();
    y.to_vec::<f32>().unwrap()
}

/// Every dense EXL3 width of the pack (the trunk's Mamba and attention linears at 4 to 8 bits, the head's EXL3 attention and eh_proj),
/// one linear of each (role, width): the device decode path == the float64 host decode within the kernel's committed limit, and a row
/// alone has the bits it has in a call of many rows.
#[test]
fn gpu_mixedk_dense_linears_at_every_width_match_the_host_decode() {
    let Some(dir) = gpu_dir() else { return };
    let ck = scan(&dir, true).unwrap_or_else(|e| panic!("{e}"));
    let roles = ["in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "eh_proj"];
    let mut picked: BTreeMap<(String, String, u8), String> = BTreeMap::new();
    for (prefix, g) in &ck.groups {
        let dense = prefix.starts_with("backbone.layers.") || prefix.starts_with("mtp.layers.0.");
        let role = prefix.rsplit('.').next().unwrap_or("");
        if dense && roles.contains(&role) && !prefix.contains("experts") {
            let part = if prefix.starts_with("mtp.") { "head" } else { "trunk" };
            picked.entry((part.to_string(), role.to_string(), g.bits.twice())).or_insert_with(|| prefix.clone());
        }
    }
    let widths: Vec<u8> = picked.keys().map(|(_, _, t)| t / 2).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    eprintln!("dense linears picked: {} (widths {widths:?})", picked.len());
    assert!(
        widths.contains(&7) && widths.contains(&8) && widths.contains(&5),
        "the pack's 5, 7 and 8-bit dense linears are covered"
    );
    assert!(picked.keys().any(|(p, r, _)| p == "head" && r == "q_proj"), "the head's EXL3 attention is covered");
    let st = Stream::new().unwrap();
    let table = codebook(Codebook::Mul1);
    let rows = 6;
    for ((part, role, twice), prefix) in &picked {
        let g = ck.load_group(prefix).unwrap_or_else(|e| panic!("{prefix}: {e}"));
        assert_eq!(g.group.codebook, Codebook::Mul1, "{prefix}");
        let l = Exl3Weights::from_loaded(&g).unwrap_or_else(|e| panic!("{prefix}: {e}"));
        let (k, n) = (g.group.k, g.group.n);
        let x = bf16_noise(0x5EED ^ (k * n) as u64, rows * k);
        let got = device_forward(&l, &x, rows, &st);
        let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let want = gb_exl3::forward(&xf, rows, &g.trellis, g.kt, g.nt, g.group.bits, &table, &g.suh, &g.svh, g.bias.as_deref());
        let (l2, units) = dense_check(&got, &want, rows, n, prefix);
        // a row alone is the same bits as inside the call
        let one = device_forward(&l, &x[(rows - 1) * k..], 1, &st);
        assert!(one == got[(rows - 1) * n..], "{prefix}: a row alone differs from the same row in a {rows}-row call");
        eprintln!(
            "{part:5} {role:8} {:>3} bits [{k:5} -> {n:6}] {prefix}: device == host decode, row rel L2 <= {l2:.2e}, element <= {units:.2}",
            twice / 2
        );
    }
}

/// Calibration: the same harness on our qualified pack's 4-bit linears (`REXL3_NEMO_OMNI_DIR`): what the device decode path's distance to
/// the float64 host decode is on weights this engine serves today, under the same limits.
#[test]
fn gpu_the_same_harness_on_our_pack_calibrates_the_limits() {
    let Some(ours) = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set (our pack, for the calibration)");
        return;
    };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let ck = scan(&ours, true).unwrap_or_else(|e| panic!("{e}"));
    let st = Stream::new().unwrap();
    let table = codebook(Codebook::Mul1);
    let rows = 6;
    for prefix in [
        "language_model.backbone.layers.7.mixer.q_proj",
        "language_model.backbone.layers.7.mixer.k_proj",
        "language_model.backbone.layers.7.mixer.o_proj",
        "language_model.backbone.layers.0.mixer.out_proj",
        "language_model.backbone.layers.0.mixer.in_proj",
    ] {
        let g = ck.load_group(prefix).unwrap_or_else(|e| panic!("{prefix}: {e}"));
        let l = Exl3Weights::from_loaded(&g).unwrap();
        let (k, n) = (g.group.k, g.group.n);
        let x = bf16_noise(0x5EED ^ (k * n) as u64, rows * k);
        let got = device_forward(&l, &x, rows, &st);
        let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let want = gb_exl3::forward(&xf, rows, &g.trellis, g.kt, g.nt, g.group.bits, &table, &g.suh, &g.svh, g.bias.as_deref());
        let (l2, units) = dense_check(&got, &want, rows, n, prefix);
        eprintln!(
            "ours {:>3} bits [{k:5} -> {n:6}] {prefix}: row rel L2 <= {l2:.2e}, element <= {units:.2} units",
            g.group.bits.as_f64()
        );
    }
}

/// The plan of the MixedK pack's units the engine CAN build here (the globals, with the EXL3 lm_head): its forms read from the index.
fn globals_plan(dir: &std::path::Path) -> (OmniConfig, Inventory, LayerPlan) {
    let cfg = config(dir);
    let inv = Inventory::read(dir).unwrap_or_else(|e| panic!("{e}"));
    let (storages, _) = storages_of(&cfg, &inv);
    let (head, _) = head_exl3_of(&cfg, &inv);
    let (widths, errors) = widths_of(&inv);
    assert!(errors.is_empty(), "{errors:?}");
    let plan = LayerPlan { units: units_with(&cfg, &storages, &head, &widths), outside_plan: 0 };
    (cfg, inv, plan)
}

/// The host decode of the lm_head's strips `strips` (float64 `[K, 128]` each, `gb_exl3::decode_strip`).
fn host_strips(g: &LoadedGroup, strips: &[usize]) -> Vec<Vec<f64>> {
    let table = codebook(Codebook::Mul1);
    strips.iter().map(|&s| decode_strip(&g.trellis, g.kt, g.nt, g.group.bits, &table, &g.suh, &g.svh, s)).collect()
}

/// The host logits of rows `rows` of `x` (`[_, k]`) over every column of the EXL3 layer `g`, every strip decoded (`decode_strip`) on up
/// to 16 threads: float64 `[rows.len(), n]`.
fn host_logits_all(g: &LoadedGroup, x: &[f32], rows: &[usize], k: usize) -> Vec<f64> {
    let n = g.group.n;
    let strips = n / 128;
    let table = codebook(Codebook::Mul1);
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).clamp(1, 16);
    let parts: Vec<Vec<(usize, Vec<f64>)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let table = &table;
                sc.spawn(move || {
                    let mut out = Vec::new();
                    for s in (t..strips).step_by(threads) {
                        let d = decode_strip(&g.trellis, g.kt, g.nt, g.group.bits, table, &g.suh, &g.svh, s);
                        let mut y = vec![0f64; rows.len() * 128];
                        for (ri, &r) in rows.iter().enumerate() {
                            for i in 0..k {
                                let xi = x[r * k + i] as f64;
                                for c in 0..128 {
                                    y[ri * 128 + c] += xi * d[i * 128 + c];
                                }
                            }
                        }
                        out.push((s, y));
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut y = vec![0f64; rows.len() * n];
    for (s, v) in parts.into_iter().flatten() {
        for ri in 0..rows.len() {
            y[ri * n + s * 128..ri * n + (s + 1) * 128].copy_from_slice(&v[ri * 128..(ri + 1) * 128]);
        }
    }
    y
}

/// The EXL3 lm_head (K8, [4096 -> 131072]) loaded by the engine's own path (plan, upload, bridge: `HeadLinear::Exl3`), scored by
/// `tp_arm::lm_head_logits` over a 200-row window (two decode calls of at most 128 rows): its logits on sampled 128-column strips ==
/// the float64 host decode within the kernel's limit; a row alone == the same row in the window, bit for bit; a TP rank's slice (the
/// 128-block cut, rank 1 of 2, `tp_shard::upload_tp` + `into_device_tp`) gives the whole head's columns bit for bit; and the restricted
/// table (D2, `gather_head_rows`) is the decoded columns rounded once to bf16, bit for bit.
#[test]
fn gpu_mixedk_lm_head_matches_the_host_decode_whole_sliced_and_restricted() {
    let Some(dir) = gpu_dir() else { return };
    let (cfg, inv, plan) = globals_plan(&dir);
    let (vocab, h) = (cfg.vocab, cfg.hidden);
    let lm = plan.lm_head_exl3().expect("the MixedK lm_head is EXL3").clone();
    assert_eq!((lm.k, lm.n, lm.bits), (h, vocab, 8));
    let globals = |u: UnitId| matches!(u, UnitId::Embed | UnitId::NormF | UnitId::LmHead);
    let st = Stream::new().unwrap();
    let t = std::time::Instant::now();
    let mut up = weights::upload(&dir, &inv, &plan, globals, &Options::from_env()).unwrap_or_else(|e| panic!("{e}"));
    let w = into_device(&mut up, &cfg, &plan, 0..0, &st).unwrap_or_else(|e| panic!("{e}"));
    assert!(up.is_empty(), "the bridge left {} tensors", up.len());
    let HeadLinear::Exl3(l) = &w.lm_head else { panic!("the lm_head is not EXL3 on the device") };
    assert_eq!((l.k, l.n, l.bits), (h, vocab, 8));
    eprintln!("lm_head loaded through the plan and the bridge in {:.1} s", t.elapsed().as_secs_f64());
    // a 200-row window: two decode calls of at most 128 rows (`linear_rows`)
    let m = 200;
    let x = bf16_noise(0xA11CE, m * h);
    // views: a window of more than 128 rows is scored in calls of 128 (`subview`), as the model's buffers are
    let mut xd = DevBuf::zeroed_view(m * h * 4).unwrap();
    xd.upload(&x).unwrap();
    let mut logits = DevBuf::zeroed_view(m * vocab * 4).unwrap();
    let mut ws = DevBuf::zeroed(w.max_workspace_bytes(128)).unwrap();
    lm_head_logits(&SingleRank, (&w.lm_head).into(), &xd, &mut logits, None, m, vocab, true, &mut ws, &st).unwrap();
    st.sync().unwrap();
    let got = logits.to_vec::<f32>().unwrap();
    // the host decode of sampled strips (first, last, the TP cut's two sides, and one between)
    let ck = scan(&dir, true).unwrap_or_else(|e| panic!("{e}"));
    let g = ck.load_group("lm_head").unwrap_or_else(|e| panic!("lm_head: {e}"));
    let strips = [0usize, 1, 511, 512, 777, 1023];
    let dec = host_strips(&g, &strips);
    let sample_rows = [0usize, 1, 127, 128, 199];
    let (mut l2, mut units) = (0f64, 0f64);
    for (si, &s) in strips.iter().enumerate() {
        let (mut gv, mut wv) = (Vec::new(), Vec::new());
        for &r in &sample_rows {
            for c in 0..128 {
                let acc: f64 = (0..h).map(|i| x[r * h + i] as f64 * dec[si][i * 128 + c]).sum();
                wv.push(acc);
                gv.push(got[r * vocab + s * 128 + c]);
            }
        }
        // per strip the norm is the strip's own 128 columns: a strip of small logits reads larger here (1.1e-3 measured) than the whole
        // row does (2.0e-4, below), the same absolute error over a smaller denominator
        let (a, b) = dense_check(&gv, &wv, sample_rows.len(), 128, &format!("lm_head strip {s}"));
        (l2, units) = (l2.max(a), units.max(b));
    }
    eprintln!(
        "lm_head: {} strips x {} rows == the float64 host decode, row rel L2 <= {l2:.2e}, element <= {units:.2} units",
        strips.len(),
        sample_rows.len()
    );
    // two whole rows against the host decode of every strip (the strip sample's limit holds over the full 131,072 columns too)
    let full_rows = [0usize, 199];
    let host = host_logits_all(&g, &x, &full_rows, h);
    let gotr: Vec<f32> = full_rows.iter().flat_map(|&r| got[r * vocab..(r + 1) * vocab].iter().copied()).collect();
    let (fl2, funits) = dense_check(&gotr, &host, full_rows.len(), vocab, "lm_head full rows");
    eprintln!("lm_head: rows {full_rows:?} over all {vocab} columns == the float64 host decode, row rel L2 <= {fl2:.2e}, element <= {funits:.2} units");
    // the pinned prefill path is a different arithmetic (W_q reconstructed to fp16, a cuBLAS GEMM): the head never takes it, so a
    // window's logits are the decode step's row for row (informational: how far the two are)
    {
        let mut pinned = DevBuf::zeroed_view(m * vocab * 4).unwrap();
        let mut scratch = DevBuf::zeroed(l.prefill_scratch_bytes(m)).unwrap();
        match l.forward_prefill_pinned(&xd, &mut pinned, m, Dtype::F32, &mut scratch, &st) {
            Ok(()) => {
                st.sync().unwrap();
                let p = pinned.to_vec::<f32>().unwrap();
                let differ = p.iter().zip(&got).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                let worst = p.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                eprintln!("lm_head: the pinned prefill path on the window differs from the decode path in {differ} of {} logits (max |diff| {worst:.2e}): not used for the head", m * vocab);
            }
            Err(e) => eprintln!("lm_head: the pinned prefill path does not take this call ({e}): not used for the head"),
        }
    }
    // a row alone (the decode step) is the same bits as in the window, in either chunk of it
    let one = DevBuf::from_slice(&x[..h]).unwrap();
    let mut l0 = DevBuf::zeroed(vocab * 4).unwrap();
    lm_head_logits(&SingleRank, (&w.lm_head).into(), &one, &mut l0, None, 1, vocab, true, &mut ws, &st).unwrap();
    st.sync().unwrap();
    assert!(l0.to_vec::<f32>().unwrap() == got[..vocab], "row 0 alone differs from row 0 of the 200-row window");
    let one = DevBuf::from_slice(&x[199 * h..]).unwrap();
    let mut l1 = DevBuf::zeroed(vocab * 4).unwrap();
    lm_head_logits(&SingleRank, (&w.lm_head).into(), &one, &mut l1, None, 1, vocab, true, &mut ws, &st).unwrap();
    st.sync().unwrap();
    assert!(l1.to_vec::<f32>().unwrap() == got[199 * vocab..], "row 199 alone differs from row 199 of the 200-row window");
    // the restricted table (D2): the decoded columns of the ids, rounded once to bf16, bit for bit
    let ids: Vec<u32> = [0u32, 1, 2, 127, 128, 65_535, 65_536, 65_600, 99_457, 131_071]
        .into_iter()
        .chain((65_408..65_664).step_by(3))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let t = std::time::Instant::now();
    let table = gather_head_rows(&SingleRank, &w.lm_head, vocab, &ids, &st).unwrap_or_else(|e| panic!("{e}"));
    eprintln!("restricted table of {} ids decoded in {:.2} s", ids.len(), t.elapsed().as_secs_f64());
    let mut rows = vec![0u16; ids.len() * h];
    table.download_range(0, &mut rows).unwrap();
    let all_strips: Vec<usize> =
        ids.iter().map(|&i| i as usize / 128).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    let dec_all = host_strips(&g, &all_strips);
    for (r, &id) in ids.iter().enumerate() {
        let si = all_strips.iter().position(|&s| s == id as usize / 128).unwrap();
        let col = id as usize % 128;
        for i in 0..h {
            assert_eq!(rows[r * h + i], f64_to_bf16(dec_all[si][i * 128 + col]), "id {id} [{i}]");
        }
    }
    // the restricted rows against the EXL3 head's own logits (informational: a draft only proposes)
    let mut far = 0f64;
    for (r, &id) in ids.iter().enumerate() {
        let wrow: Vec<f64> = rows[r * h..(r + 1) * h].iter().map(|&b| f32::from_bits((b as u32) << 16) as f64).collect();
        let d: f64 = (0..h).map(|i| x[i] as f64 * wrow[i]).sum();
        far = far.max((d - got[id as usize] as f64).abs());
    }
    eprintln!("restricted rows: bit for bit the decoded columns in bf16; row 0's logits within {far:.2e} of the EXL3 head's");
    drop(table);
    // D2 at production size: the default restricted list (49,152 ids), the table every boot of this pack builds
    let ids48 = restricted_vocab_ids_for(&VocabSetting::Default).unwrap().expect("the default list");
    let t = std::time::Instant::now();
    let table48 = gather_head_rows(&SingleRank, &w.lm_head, vocab, &ids48, &st).unwrap_or_else(|e| panic!("{e}"));
    st.sync().unwrap();
    let secs = t.elapsed().as_secs_f64();
    let strips48 = ids48.iter().map(|&i| i / 128).collect::<std::collections::BTreeSet<_>>().len();
    assert_eq!(table48.bytes(), ids48.len() * h * 2);
    eprintln!(
        "D2 boot cost: the default list's {} ids ({strips48} of 1024 strips) decoded to a {:.0} MiB bf16 table in {secs:.2} s ({} threads)",
        ids48.len(),
        table48.bytes() as f64 / (1 << 20) as f64,
        std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 16)
    );
    drop(table48);
    drop(w);
    // a TP rank's slice: rank 1 of 2 holds columns [65536, 131072) (the 128-block cut), and its logits are the whole head's columns
    let shard = ShardPlan::for_config(TpTopology { rank: 1, ranks: 2 }, &cfg);
    let (lo, hi) = shard.lm_head_cols_exl3(vocab);
    assert_eq!((lo, hi), (65_536, 131_072));
    let mut up =
        upload_tp(&dir, &inv, &plan, cfg.experts, globals, &Options::from_env(), &shard).unwrap_or_else(|e| panic!("{e}"));
    let ws1 = into_device_tp(&mut up, &cfg, &plan, 0..0, &st, &shard).unwrap_or_else(|e| panic!("{e}"));
    let HeadLinear::Exl3(s1) = &ws1.lm_head else { panic!("the rank's lm_head is not EXL3") };
    assert_eq!((s1.k, s1.n, s1.bits), (h, hi - lo, 8));
    let mut ys = DevBuf::zeroed_view(m * (hi - lo) * 4).unwrap();
    let st2 = Stream::new().unwrap();
    linear_rows(s1, &xd, &mut ys, m, Dtype::F32, Dtype::F32, &mut ws, &st2).unwrap();
    st2.sync().unwrap();
    let ys = ys.to_vec::<f32>().unwrap();
    let mut differ = 0usize;
    for r in 0..m {
        differ += (0..hi - lo).filter(|&c| ys[r * (hi - lo) + c].to_bits() != got[r * vocab + lo + c].to_bits()).count();
    }
    assert_eq!(differ, 0, "rank 1's slice differs from the whole head's columns [{lo}, {hi})");
    eprintln!(
        "TP: rank 1's 128-block slice [{lo}, {hi}) of the EXL3 lm_head == the whole head's columns, bit for bit, over {m} rows"
    );
}

/// An MoE block of the pack (trunk layer 1: routed experts of K 2 to 6, mixed within an expert, the shared expert EXL3) on a SYNTHETIC
/// bf16 router gate and bf16 latent projections rounded from the pack's F16 ones (the pack's own gate and latents are F16: lane K).
/// Lane L's GUARD (b) kept such a layer off the prefill path (`supports_prefill` false) until the any-width prefill kernels landed
/// (lane K, P1-7): it is on it now, and its prefill rows are the decode path's bits. The decode-path block == the host reference
/// (`moe_ref::latent_moe` on the same weights) within the block limit, and the routes agree.
#[test]
fn gpu_mixedk_moe_block_of_k2_to_k6_experts_matches_the_host_reference_and_its_prefill_path_the_decode_path() {
    let Some(dir) = gpu_dir() else { return };
    let cfg = config(&dir);
    let ck: Checkpoint = scan(&dir, true).unwrap_or_else(|e| panic!("{e}"));
    // the E layer whose routed experts span the most widths (the pack's K 2 to 6 live in different layers)
    let layer = cfg
        .pattern
        .chars()
        .enumerate()
        .filter(|(_, c)| *c == 'E')
        .map(|(i, _)| {
            let w: std::collections::BTreeSet<u8> = (0..cfg.experts)
                .flat_map(|e| ["up_proj", "down_proj"].map(|p| format!("backbone.layers.{i}.mixer.experts.{e}.{p}")))
                .filter_map(|p| ck.groups.get(&p).map(|g| g.bits.twice() / 2))
                .collect();
            (w.len(), std::cmp::Reverse(i), w)
        })
        .max()
        .map(|(_, std::cmp::Reverse(i), w)| {
            eprintln!("layer {i}: routed experts at widths {w:?}");
            i
        })
        .expect("an E layer");
    run_moe_block(&dir, &cfg, &ck, layer, false);
    // and the first E layer that holds 2-bit experts
    let k2 = cfg
        .pattern
        .chars()
        .enumerate()
        .filter(|(_, c)| *c == 'E')
        .map(|(i, _)| i)
        .find(|&i| {
            (0..cfg.experts).any(|e| {
                ["up_proj", "down_proj"].iter().any(|p| {
                    ck.groups.get(&format!("backbone.layers.{i}.mixer.experts.{e}.{p}")).is_some_and(|g| g.bits.twice() == 4)
                })
            })
        })
        .expect("a layer with 2-bit experts");
    run_moe_block(&dir, &cfg, &ck, k2, false);
}

/// The pack's MoE blocks on its OWN router gate (F16 as stored, the fp16 router kernels) and its OWN F16 latent projections
/// (`Proj::F16`, read in place: lane K, decision D1 and the F16 dense path) against the host reference (`moe_ref::latent_moe` on the same
/// values widened exactly): the routes agree, the block is within the block limit, and the fast launches equal the reference launches
/// bit for bit. Layers 1 (the widest mix), the first with 2-bit experts, and the last E layer.
#[test]
fn gpu_mixedk_moe_block_on_its_own_f16_gate_and_latents_matches_the_host_reference() {
    let Some(dir) = gpu_dir() else { return };
    let cfg = config(&dir);
    let ck: Checkpoint = scan(&dir, true).unwrap_or_else(|e| panic!("{e}"));
    let es: Vec<usize> = cfg.pattern.chars().enumerate().filter(|(_, c)| *c == 'E').map(|(i, _)| i).collect();
    let k2 = *es
        .iter()
        .find(|&&i| {
            (0..cfg.experts).any(|e| {
                ["up_proj", "down_proj"].iter().any(|p| {
                    ck.groups.get(&format!("backbone.layers.{i}.mixer.experts.{e}.{p}")).is_some_and(|g| g.bits.twice() == 4)
                })
            })
        })
        .expect("a layer with 2-bit experts");
    let mut layers = vec![1, k2, *es.last().unwrap()];
    layers.dedup();
    for layer in layers {
        run_moe_block(&dir, &cfg, &ck, layer, true);
    }
}

/// One E layer's block (see the tests above): with `own`, on the pack's own F16 router gate and F16 latents (the fp16 kernels);
/// without, on a synthetic bf16 gate and the latents rounded to bf16. Off the prefill path, device == host, and (own) fast == reference.
fn run_moe_block(_dir: &std::path::Path, cfg: &OmniConfig, ck: &Checkpoint, layer: usize, own: bool) {
    let m = format!("backbone.layers.{layer}.mixer");
    let (h, lat) = (cfg.hidden, cfg.latent);
    let raw16 = |name: &str| -> Vec<u16> {
        let e = ck.plain.get(name).unwrap_or_else(|| panic!("no plain tensor {name}"));
        assert_eq!(e.dtype, "F16", "{name}");
        ck.read_u16(e).unwrap()
    };
    // the router gate and the latent projections, the same values on both sides: the pack's own F16 (`own`), or a synthetic bf16 gate
    // and the latents rounded to bf16
    let (gate_bits, gate_dtype) = if own {
        (raw16(&format!("{m}.gate.weight")), nemotron_moe::GateDtype::F16)
    } else {
        (noise(0x6A7E, cfg.experts * h).iter().map(|&v| f64_to_bf16(v as f64 * 0.05)).collect(), nemotron_moe::GateDtype::Bf16)
    };
    let f16f = |b: u16| gb_exl3::f16::f16_to_f64(b) as f32;
    let bf = |b: u16| if own { f16f(b) } else { f32::from_bits((b as u32) << 16) };
    let f16_weight = |name: &str| -> Vec<u16> {
        let e = ck.plain.get(name).unwrap_or_else(|| panic!("no plain tensor {name}"));
        assert_eq!(e.dtype, "F16", "{name}");
        if own {
            ck.read_u16(e).unwrap()
        } else {
            ck.read_floats(e).unwrap().into_iter().map(f64_to_bf16).collect()
        }
    };
    let fc1 = f16_weight(&format!("{m}.fc1_latent_proj.weight")); // [latent, hidden]
    let fc2 = f16_weight(&format!("{m}.fc2_latent_proj.weight")); // [hidden, latent]
    let bias: Vec<f32> = ck
        .read_floats(ck.plain.get(&format!("{m}.gate.e_score_correction_bias")).expect("the bias"))
        .unwrap()
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let group = |p: &str| ck.load_group(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    let mut census: BTreeMap<(u8, u8), usize> = BTreeMap::new();
    // one expert of each width the layer has (up or down), which rows below are steered to select
    let mut targets: BTreeMap<u8, usize> = BTreeMap::new();
    let (mut up, mut down) = (Vec::new(), Vec::new());
    let (mut census_up, mut census_down) = (vec![0u8; cfg.experts], vec![0u8; cfg.experts]);
    for e in 0..cfg.experts {
        let (u, d) = (group(&format!("{m}.experts.{e}.up_proj")), group(&format!("{m}.experts.{e}.down_proj")));
        *census.entry((u.group.bits.twice() / 2, d.group.bits.twice() / 2)).or_default() += 1;
        (census_up[e], census_down[e]) = (u.group.bits.twice() / 2, d.group.bits.twice() / 2);
        // the first expert of each width; on the pack's own gate the one of each width with the largest selection bias (its own
        // `e_score_correction_bias` can keep a steered expert out of the top 22 whatever its logit)
        for b in [u.group.bits.twice() / 2, d.group.bits.twice() / 2] {
            let t = targets.entry(b).or_insert(e);
            if own && bias[e] > bias[*t] {
                *t = e;
            }
        }
        up.push(DevProj::exl3_from_group(&u).unwrap());
        down.push(DevProj::exl3_from_group(&d).unwrap());
    }
    eprintln!("layer {layer}: experts by (up, down) bits {census:?}");
    // lane L's layers are chosen to mix K within an expert; the own-gate runs also take the last E layer, whose 512 experts are all
    // 2-bit (a uniform narrow layer is a case of its own)
    assert!(own || census.keys().any(|(u, d)| u != d), "the layer mixes K within an expert");
    let su = group(&format!("{m}.shared_experts.up_proj"));
    let sd = group(&format!("{m}.shared_experts.down_proj"));
    let (fc1d, fc2d) = (DevBuf::from_slice(&fc1).unwrap(), DevBuf::from_slice(&fc2).unwrap());
    let weights = nemotron_moe::Weights {
        gate: DevBuf::from_slice(&gate_bits).unwrap(),
        gate_dtype,
        router_bias: DevBuf::from_slice(&bias).unwrap(),
        scale: cfg.routed_scaling as f32,
        fc1: if own { DevProj::F16 { k: h, n: lat, w: fc1d } } else { DevProj::Bf16 { k: h, n: lat, w: fc1d } },
        fc2: if own { DevProj::F16 { k: lat, n: h, w: fc2d } } else { DevProj::Bf16 { k: lat, n: h, w: fc2d } },
        shared_up: DevProj::exl3_from_group(&su).unwrap(),
        shared_down: DevProj::exl3_from_group(&sd).unwrap(),
        up,
        down,
    };
    let moe = PrefillMoe::tables(weights).unwrap();
    // P1-7 (lane K; lane L's GUARD (b) until then): a layer of K != 4 experts takes the prefill path, the any-width kernels; its rows
    // there are the decode path's bits (checked below on these rows)
    assert!(moe.supports_prefill(), "a MixedK layer takes the prefill path since P1-7");
    // the block on the decode path: noise rows, and one row per width steered to that width's expert (the noise plus a multiple of the
    // expert's gate row, so its logit is large)
    let mut x = bf16_noise(0xB10C, 16 * h);
    if own {
        // On the pack's own gate a steered row does not reliably pick its target (the gate's rows correlate, sigmoid saturates and the
        // selection bias decides), so the rows per width are found instead: a pool of noise rows routed on the host, and for each width
        // the first row whose picks include an expert of that width (which becomes the width's target).
        let pool = 512;
        let cand = bf16_noise(0x9001, pool * h);
        let router = Router {
            gate: gate_bits.iter().map(|&b| bf(b)).collect(),
            bias: bias.clone(),
            hidden: h,
            topk: cfg.topk,
            scale: 1.0,
        };
        let logits = router.logits(&cand, pool);
        let widths_of = |e: usize| -> [u8; 2] { [census_up[e], census_down[e]] };
        let wanted: Vec<u8> = targets.keys().copied().collect();
        targets.clear();
        for t in 0..pool {
            let (picks, _) = route(&logits[t * cfg.experts..(t + 1) * cfg.experts], &router.bias, cfg.topk, 1.0);
            let mut took = false;
            for &e in &picks {
                for b in widths_of(e) {
                    if wanted.contains(&b) && !targets.contains_key(&b) {
                        targets.insert(b, e);
                        took = true;
                    }
                }
            }
            if took {
                x.extend_from_slice(&cand[t * h..(t + 1) * h]);
            }
            if targets.len() == wanted.len() {
                break;
            }
        }
        assert_eq!(targets.len(), wanted.len(), "layer {layer}: no row of the pool picks an expert of every width {wanted:?}");
    } else {
        for (t, &e) in targets.values().enumerate() {
            let g: Vec<f32> = gate_bits[e * h..(e + 1) * h].iter().map(|&b| bf(b)).collect();
            let g2: f32 = g.iter().map(|v| v * v).sum();
            let nz = bf16_noise(0x7A46 + t as u64, h);
            x.extend(
                nz.iter().zip(&g).map(|(&n, &gv)| f32::from_bits((f64_to_bf16((0.25 * n + 12.0 / g2 * gv) as f64) as u32) << 16)),
            );
        }
    }
    let rows = x.len() / h;
    let st = Stream::new().unwrap();
    let mut scratch = MoeScratch::new(64).unwrap();
    let xd = DevBuf::from_slice(&x).unwrap();
    let mut out = DevBuf::zeroed(rows * h * 4).unwrap();
    moe.forward_in(&mut scratch, &xd, rows, &mut out, &st).unwrap();
    st.sync().unwrap();
    let got = out.to_vec::<f32>().unwrap();
    let (pick, _) = scratch.download_route(rows, &st).unwrap();
    // P1-7: the prefill path (the any-width kernels) on the same rows gives the decode path's bits
    {
        let mut ps = gb_cuda::nemotron_moe_prefill::PrefillScratch::new(rows).unwrap();
        let mut pout = DevBuf::zeroed(rows * h * 4).unwrap();
        moe.forward_prefill_in(&mut ps, &xd, rows, &mut pout, &st).unwrap();
        st.sync().unwrap();
        let pv = pout.to_vec::<f32>().unwrap();
        let differ = pv.iter().zip(&got).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(differ, 0, "layer {layer}: the prefill path differs from the decode path in {differ} values");
        eprintln!("layer {layer}: prefill path == decode path bitwise ({rows} rows)");
    }
    if own {
        eprintln!("layer {layer}: launches {}", moe.launch_summary());
        let mut rout = DevBuf::zeroed(rows * h * 4).unwrap();
        moe.forward_reference(&mut scratch, &xd, rows, &mut rout, &st).unwrap();
        st.sync().unwrap();
        let r = rout.to_vec::<f32>().unwrap();
        let differ = got.iter().zip(&r).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(differ, 0, "layer {layer}: the fast launches differ from the reference launches in {differ} values");
    }
    // the host reference on the same weights
    let router = Router {
        gate: gate_bits.iter().map(|&b| bf(b)).collect(),
        bias,
        hidden: h,
        topk: cfg.topk,
        scale: cfg.routed_scaling as f32,
    };
    let logits = router.logits(&x, rows);
    let mut want_pick = Vec::new();
    for t in 0..rows {
        want_pick.extend(route(&logits[t * cfg.experts..(t + 1) * cfg.experts], &router.bias, cfg.topk, router.scale).0);
    }
    assert_eq!(pick.iter().map(|&v| v as usize).collect::<Vec<_>>(), want_pick, "the device's routes differ from the host's");
    let picked: std::collections::BTreeSet<usize> = want_pick.iter().copied().collect();
    let mut experts: Vec<Option<Mlp>> = (0..cfg.experts).map(|_| None).collect();
    let mut widths = std::collections::BTreeSet::new();
    for &e in &picked {
        let (u, d) = (group(&format!("{m}.experts.{e}.up_proj")), group(&format!("{m}.experts.{e}.down_proj")));
        widths.insert(u.group.bits.twice() / 2);
        widths.insert(d.group.bits.twice() / 2);
        experts[e] = Some(Mlp { up: Box::new(RefExl3 { group: u }), down: Box::new(RefExl3 { group: d }) });
    }
    let fc1_ref = Dense::from_hf(lat, h, &fc1.iter().map(|&b| bf(b)).collect::<Vec<_>>());
    let fc2_ref = Dense::from_hf(h, lat, &fc2.iter().map(|&b| bf(b)).collect::<Vec<_>>());
    let shared = Mlp { up: Box::new(RefExl3 { group: su }), down: Box::new(RefExl3 { group: sd }) };
    let want = latent_moe(&x, &router, &fc1_ref as &dyn RefProj, &fc2_ref, &experts, &shared);
    let scale = want.iter().fold(0f64, |a, &v| a.max((v as f64).abs()));
    let diff = got.iter().zip(&want).fold(0f64, |a, (&g, &w)| a.max(((g - w) as f64).abs()));
    let rel = diff / scale;
    eprintln!(
        "layer {layer} block ({}), {rows} rows, {} experts picked at widths {widths:?}: device vs host max rel {rel:.3e} (block limit 1e-3)",
        if own { "its own F16 gate and latents" } else { "synthetic bf16 gate, latents rounded to bf16" },
        picked.len()
    );
    for (b, e) in &targets {
        assert!(picked.contains(e), "the {b}-bit expert {e} the rows were steered to is not picked");
    }
    assert_eq!(widths, targets.keys().copied().collect(), "the picks cover every width of the layer");
    assert!(rel <= BLOCK_TOL, "block vs host reference: {rel:e} (limit {BLOCK_TOL:e})");
}

/// The EXL3 lm_head at TWO ranks (a loopback group on one device, each rank holding its 128-block slice through `upload_tp`): the
/// logits of a window (`tp_arm::lm_head_logits`: the rank's slice, the `-0.0` scatter, one allreduce) and the restricted table (D2,
/// `gather_head_rows`: each rank decodes the ids it holds, the group gathers them) are the single node's, bit for bit, on every rank.
#[test]
fn gpu_mixedk_lm_head_at_two_ranks_is_the_single_nodes_logits_and_restricted_rows() {
    use gb_model::families::nemotron_h_omni::tp_comm::{HostBarrier, LoopbackGroup};
    use gb_model::families::nemotron_h_omni::tp_shard::head_temp_rows_in;
    let Some(dir) = gpu_dir() else { return };
    let (cfg, inv, plan) = globals_plan(&dir);
    let (vocab, h) = (cfg.vocab, cfg.hidden);
    let globals = |u: UnitId| matches!(u, UnitId::Embed | UnitId::NormF | UnitId::LmHead);
    // the default restricted list (49,152 ids, the production table) plus both ends and the ranks' cut (65,536): many 2,048-row chunks
    let mut ids: Vec<u32> = restricted_vocab_ids_for(&VocabSetting::Default).unwrap().expect("the default list");
    ids.extend((0..200).chain(65_400..65_700).chain(vocab as u32 - 50..vocab as u32));
    ids.sort_unstable();
    ids.dedup();
    assert!(ids.len() > 49_152, "{} ids", ids.len());
    let m = 64;
    let x = bf16_noise(0x2A2A, m * h);
    let st = Stream::new().unwrap();
    // the single node
    let (want_logits, want_table) = {
        let mut up = weights::upload(&dir, &inv, &plan, globals, &Options::from_env()).unwrap_or_else(|e| panic!("{e}"));
        let w = into_device(&mut up, &cfg, &plan, 0..0, &st).unwrap_or_else(|e| panic!("{e}"));
        let mut xd = DevBuf::zeroed_view(m * h * 4).unwrap();
        xd.upload(&x).unwrap();
        let mut logits = DevBuf::zeroed_view(m * vocab * 4).unwrap();
        let mut ws = DevBuf::zeroed(w.max_workspace_bytes(128)).unwrap();
        lm_head_logits(&SingleRank, (&w.lm_head).into(), &xd, &mut logits, None, m, vocab, true, &mut ws, &st).unwrap();
        let t = gather_head_rows(&SingleRank, &w.lm_head, vocab, &ids, &st).unwrap();
        st.sync().unwrap();
        let mut table = vec![0u16; ids.len() * h];
        t.download_range(0, &mut table).unwrap();
        (logits.to_vec::<f32>().unwrap(), table)
    };
    let ranks = 2;
    let comms = LoopbackGroup::new(ranks);
    let ready = std::sync::Arc::new(HostBarrier::new(ranks));
    let got: Vec<(Vec<f32>, Vec<u16>)> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                let (dir, cfg, plan, inv, ids, x, ready) = (&dir, &cfg, &plan, &inv, &ids, &x, ready.clone());
                s.spawn(move || {
                    gb_cuda::rt::init_context().unwrap();
                    let st = Stream::new().unwrap();
                    let shard = ShardPlan::for_config(TpTopology { rank, ranks }, cfg);
                    let mut up = upload_tp(dir, inv, plan, cfg.experts, globals, &Options::from_env(), &shard)
                        .unwrap_or_else(|e| panic!("rank {rank}: {e}"));
                    let w = into_device_tp(&mut up, cfg, plan, 0..0, &st, &shard).unwrap_or_else(|e| panic!("rank {rank}: {e}"));
                    let HeadLinear::Exl3(l) = &w.lm_head else { panic!("rank {rank}: the lm_head is not EXL3") };
                    assert_eq!(l.n, vocab / 2, "rank {rank} holds half of the head's columns");
                    let mut xd = DevBuf::zeroed_view(m * h * 4).unwrap();
                    xd.upload(x).unwrap();
                    let mut logits = DevBuf::zeroed_view(m * vocab * 4).unwrap();
                    let mut local = DevBuf::zeroed_view(m * head_temp_rows_in(vocab, ranks, 128) * 4).unwrap();
                    let mut ws = DevBuf::zeroed(w.max_workspace_bytes(128)).unwrap();
                    ready.wait(std::time::Duration::from_secs(600)).unwrap();
                    lm_head_logits(&comm, (&w.lm_head).into(), &xd, &mut logits, Some(&mut local), m, vocab, true, &mut ws, &st)
                        .unwrap_or_else(|e| panic!("rank {rank}: {e}"));
                    st.sync().unwrap();
                    let t0 = std::time::Instant::now();
                    let t = gather_head_rows(&comm, &w.lm_head, vocab, ids, &st).unwrap_or_else(|e| panic!("rank {rank}: {e}"));
                    st.sync().unwrap();
                    eprintln!(
                        "rank {rank}: the {}-id restricted table decoded and gathered in {:.2} s",
                        ids.len(),
                        t0.elapsed().as_secs_f64()
                    );
                    let mut table = vec![0u16; ids.len() * h];
                    t.download_range(0, &mut table).unwrap();
                    (logits.to_vec::<f32>().unwrap(), table)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a rank panicked")).collect()
    });
    for (rank, (logits, table)) in got.iter().enumerate() {
        let bad = logits.iter().zip(&want_logits).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(bad, 0, "rank {rank}: {bad} logits differ from the single node's");
        assert!(table == &want_table, "rank {rank}: the restricted table differs from the single node's");
    }
    eprintln!(
        "two ranks: the EXL3 head's {m}-row logits and the {}-id restricted table are the single node's, bit for bit, on both ranks",
        ids.len()
    );
}
