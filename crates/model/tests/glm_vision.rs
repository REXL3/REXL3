//! GPU tests of the GLM-5.3-Flash vision tower against the ExLlamaV3 fork's tower-level dumps (`tools/dump_glm_vision_reference.py`, manifest
//! `testdata/glm53/vision_manifest.json`). Every test is `#[ignore]` and env-gated: it needs REXL3_GLM_DIR (the pack: only the 1.1 GB vision tower is
//! loaded) and REXL3_GLM_VISION_DIR (the dump directory), runs under the GPU lock and prints "not run" where either is unset.
//!
//! What can be demanded of the features. The fork's tower amplifies noise: a few "massive activation" tokens appear at blocks 12 to 16 (residual norms 10 to
//! 30 times the median), every later block attends to them, and a +-1 ulp change of 0.1% of the input values moves the features of an image by 0.4% to 4%
//! (`the_noise_floor_of_the_features`). So an implementation that differs from the fork only in arithmetic order cannot match its features to the plan's
//! 1e-2 / 4e-5 (the V2 figure of docs/PLAN.md, a guess). What is demanded instead, and what a wrong tower cannot pass:
//!
//! * every STAGE alone (`every_stage_alone_is_within_its_bound`): the fork's stream replaced after each block (and after the post norm), so a stage's error
//!   is its own: every block within 6e-4 relative L2 (measured 3.4e-4), the sublayers of blocks 0, 11 and 23, the merger from the fork's post norm;
//! * the whole run (`the_tower_follows_the_fork_stage_by_stage`): the accumulated per-block table and the features within 3x of what the tower itself moves
//!   under 1%-of-the-values ulp noise;
//! * `every_mutation_is_caught`: 19 deliberate bugs, one tower each, judged by the stage bounds and the feature gate; a survivor fails the test.
//!
//! `an_image_is_a_function_of_itself`, `the_scratch_is_what_the_plan_says`, `the_patches_the_tower_reads_are_the_forks` and `encode_time` complete the set.
//!
//! `the_tower_is_within_the_manifests_per_image_bounds` is the sc_4.10bpw leg R-G-oracle (d): it judges with the manifest's per-image bounds
//! (`REXL3_GLM_VISION_MANIFEST=testdata/glm53-sc410/vision_manifest.json`), not with the bounds in this file.
//!
//! `REXL3_GLM_DIR=... REXL3_GLM_VISION_DIR=~/rexl3-ab/glm53-vision/ref cargo test --release -p gb-model --test glm_vision -- --ignored --nocapture --test-threads=1`
//! (`REXL3_GLM_VISION_ONLY=a,b` restricts the image set, `REXL3_GLM_VISION_SAVE=DIR` writes what the tower produced next to the dumps for analysis.)

mod glm_support;

use gb_cuda::{DevBuf, Stream};
use gb_exl3::f16::f16_to_f64;
use gb_model::families::glm::npy;
use gb_model::families::glm::vision::{self, GlmVisionTower, Mutant, TowerOptions};
use gb_model::families::glm::GlmFamily;
use gb_vision::Preprocessed;
use glm_support::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

fn dump_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_GLM_VISION_DIR") {
        Ok(d) if Path::new(&d).is_dir() => Some(PathBuf::from(d)),
        Ok(d) => panic!("REXL3_GLM_VISION_DIR={d} is not a directory"),
        Err(_) => {
            eprintln!("REXL3_GLM_VISION_DIR is not set: not run");
            None
        }
    }
}

/// The pack and the dumps.
fn env() -> Option<(PathBuf, PathBuf)> {
    Some((pack_dir()?, dump_dir()?))
}

/// The tower, loaded once per process with scratch for `MAX_PATCHES` (8,000 tokens).
const MAX_PATCHES: usize = 32_000;

fn tower() -> Option<std::sync::MutexGuard<'static, GlmVisionTower>> {
    static T: OnceLock<Option<Mutex<GlmVisionTower>>> = OnceLock::new();
    let t = T.get_or_init(|| {
        let dir = pack_dir()?;
        require_available(20 * GIB);
        let guard = Guard::start(5 * GIB);
        let pack = GlmFamily::map_pack(&dir).unwrap();
        let t0 = std::time::Instant::now();
        let t = GlmVisionTower::load(&pack, &dir, &TowerOptions::new(MAX_PATCHES, 8 * GIB)).unwrap();
        eprintln!(
            "tower loaded in {:.2?}: alloc {:.2}s read {:.2}s prepare {:.2}s scratch {:.2}s; weights {:.3} GiB, scratch {:.3} GiB; MemAvailable min {:.1} GiB",
            t0.elapsed(),
            t.boot.alloc_s,
            t.boot.load_s,
            t.boot.prepare_s,
            t.boot.scratch_s,
            gib(t.weight_bytes()),
            gib(t.scratch_bytes()),
            gib(guard.finish())
        );
        Some(Mutex::new(t))
    });
    t.as_ref().map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
}

// ------------------------------------------------------------------------------------------------------------------ metrics

#[derive(Clone, Copy, Debug, Default)]
struct M {
    rel_l2: f64,
    one_minus_cos: f64,
    max_abs: f64,
    ref_rms: f64,
}

fn metrics(got: &[f32], want: &[f32]) -> M {
    assert_eq!(got.len(), want.len(), "shapes differ");
    let (mut dd, mut ww, mut gg, mut gw, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in got.iter().zip(want) {
        let (a, b) = (a as f64, b as f64);
        let d = a - b;
        dd += d * d;
        ww += b * b;
        gg += a * a;
        gw += a * b;
        mx = mx.max(d.abs());
    }
    let cos = if gg > 0.0 && ww > 0.0 { gw / (gg.sqrt() * ww.sqrt()) } else { 0.0 };
    M { rel_l2: (dd / ww.max(1e-300)).sqrt(), one_minus_cos: 1.0 - cos, max_abs: mx, ref_rms: (ww / want.len() as f64).sqrt() }
}

fn fmt(m: &M) -> String {
    format!("rel_l2 {:.2e}  1-cos {:.2e}  max_abs {:.2e}  (ref rms {:.2e})", m.rel_l2, m.one_minus_cos, m.max_abs, m.ref_rms)
}

fn f16_vec(b: &DevBuf, n: usize) -> Vec<f32> {
    let mut v = vec![0u16; n];
    b.download(&mut v).unwrap();
    v.iter().map(|&h| f16_to_f64(h) as f32).collect()
}

fn f32_vec(b: &DevBuf, n: usize) -> Vec<f32> {
    let mut v = vec![0f32; n];
    b.download(&mut v).unwrap();
    v
}

fn load(dir: &Path, image: &str, name: &str) -> Option<Vec<f32>> {
    let p = dir.join(image).join(format!("{name}.npy"));
    p.exists().then(|| npy::read(&p).unwrap().to_f32().unwrap())
}

/// `[rows, ld]` columns `c0..c0 + n` of a row-major matrix.
fn cols(v: &[f32], ld: usize, c0: usize, n: usize) -> Vec<f32> {
    v.chunks_exact(ld).flat_map(|r| r[c0..c0 + n].to_vec()).collect()
}

/// fp16 `a + b` as torch does it on half tensors: both exact in fp32, one rounding.
fn f16_add(a: &[f32], bias: &[f32], n: usize) -> Vec<f32> {
    a.iter().enumerate().map(|(i, &x)| f16_to_f64(gb_exl3::f16::f64_to_f16((x + bias[i % n]) as f64)) as f32).collect()
}

// ------------------------------------------------------------------------------------------------------------------ the image set

/// The qkv, projection, gate, up and down biases of a block, as host f32.
type Biases = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

struct Case {
    name: String,
    patches: usize,
    grid: (usize, usize),
    still: Vec<u16>,
    tokens: usize,
}

/// The dumps' manifest: the committed one of the 2.25 bpw pack (the fork's), or `REXL3_GLM_VISION_MANIFEST` (another pack's dumps, e.g. the
/// sc_4.10bpw EXL3 tower against exllamav3 1.6.2).
fn manifest() -> serde_json::Value {
    let p = match std::env::var("REXL3_GLM_VISION_MANIFEST") {
        Ok(p) => PathBuf::from(p),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/glm53/vision_manifest.json"),
    };
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn cases(dir: &Path) -> Vec<Case> {
    let manifest = manifest();
    let mut out = Vec::new();
    let only: Option<Vec<String>> =
        std::env::var("REXL3_GLM_VISION_ONLY").ok().map(|v| v.split(',').map(str::to_string).collect());
    for (name, m) in manifest["images"].as_object().unwrap() {
        if !dir.join(name).join("features.npy").exists() || only.as_ref().is_some_and(|o| !o.contains(name)) {
            continue;
        }
        let g = m["grid_thw"].as_array().unwrap();
        let (gh, gw) = (g[1].as_u64().unwrap() as usize, g[2].as_u64().unwrap() as usize);
        let patches = npy::read(&dir.join(name).join("patches.npy")).unwrap().to_u16().unwrap();
        let pre = Preprocessed { size: (gw * 14, gh * 14), grid: (1, gh, gw), patches, patch_dim: 1176, pixel_hash: 0 };
        let still = gb_vision::glm::still_patches(&pre, 2);
        out.push(Case { name: name.clone(), patches: gh * gw, grid: (gh, gw), still, tokens: gh * gw / 4 });
    }
    out.sort_by_key(|c| c.patches);
    out
}

/// How the stages of a run are fed: the tower's own stream all the way (`Full`), or the fork's stream put back after every block and after the post
/// norm (`Alone`, for images of at most 8,000 patches, whose 24 blocks are all dumped), so that each stage's error is its own.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Feed {
    Full,
    Alone,
}

/// One image through the tower, every stage the dumps hold compared. Returns the stage table (name -> metrics) and the features.
fn run_case(
    t: &mut GlmVisionTower,
    dir: &Path,
    c: &Case,
    detail_blocks: &[usize],
    feed: Feed,
    every_block: bool,
) -> (BTreeMap<String, M>, Vec<f32>) {
    let st = Stream::new().unwrap();
    let mut out = DevBuf::zeroed(c.tokens * 4096 * 4).unwrap();
    let (p, n) = (c.patches, c.tokens);
    let bias_of = |t: &GlmVisionTower, name: &str| -> Vec<f32> {
        let b = t.weight_buffers().into_iter().find(|(k, _)| k == name).unwrap().1;
        f16_vec(b, b.bytes() / 2)
    };
    let detail: Vec<usize> = detail_blocks.to_vec();
    let mut table: BTreeMap<String, M> = BTreeMap::new();
    // the bias vectors the comparisons add on the host (the tower adds them inside later kernels)
    let mut biases: BTreeMap<usize, Biases> = BTreeMap::new();
    for &b in &detail {
        biases.insert(
            b,
            (
                bias_of(t, &format!("blocks.{b}.qkv_b")),
                bias_of(t, &format!("blocks.{b}.proj_b")),
                bias_of(t, &format!("blocks.{b}.gate_b")),
                bias_of(t, &format!("blocks.{b}.up_b")),
                bias_of(t, &format!("blocks.{b}.down_b")),
            ),
        );
    }
    let dumped_blocks: Vec<usize> = (0..24).filter(|b| dir.join(&c.name).join(format!("block_{b:02}.npy")).exists()).collect();
    if feed == Feed::Alone {
        assert!(c.patches <= 8000, "{}: the fork's stream is dumped for every block only up to 8,000 patches", c.name);
        let (d2, n2) = (dir.to_path_buf(), c.name.clone());
        t.teacher = Some(Box::new(move |name| load(&d2, &n2, name)));
    }
    {
        let mut obs = |name: &str, buf: &DevBuf, st: &Stream| {
            let mut put = |k: &str, got: Vec<f32>, want: Option<Vec<f32>>| {
                if let Ok(d) = std::env::var("REXL3_GLM_VISION_SAVE") {
                    let dd = Path::new(&d).join(&c.name);
                    std::fs::create_dir_all(&dd).unwrap();
                    // flat: the analysis script reshapes it like the dump it is compared with
                    npy::write_f32(&dd.join(format!("{k}.npy")), &[got.len()], &got).unwrap();
                }
                if let Some(w) = want {
                    table.insert(k.to_string(), metrics(&got, &w));
                }
            };
            // decide first whether this name is wanted at all: observing synchronises the stream
            let wanted = match name {
                "patch_embed" | "post_layernorm" | "downsample" | "m_proj" | "m_norm" | "m_act" | "features" => true,
                n if n.starts_with("block_") => {
                    let b = n[6..].parse::<usize>().unwrap();
                    dumped_blocks.contains(&b) && (every_block || [0, 11, 23].contains(&b))
                }
                n if n.starts_with('d') && n.len() > 4 => detail.contains(&n[1..3].parse::<usize>().unwrap()),
                _ => false,
            };
            if !wanted {
                return;
            }
            st.sync().unwrap();
            match name {
                "patch_embed" => put(name, f32_vec(buf, p * 1024), load(dir, &c.name, "patch_embed")),
                "post_layernorm" => put(name, f16_vec(buf, p * 1024), load(dir, &c.name, "post_layernorm")),
                "downsample" => put(name, f16_vec(buf, n * 4096), load(dir, &c.name, "downsample")),
                "m_proj" => put(name, f16_vec(buf, n * 4096), load(dir, &c.name, "m_proj")),
                "m_norm" => put(name, f16_vec(buf, n * 4096), load(dir, &c.name, "m_norm")),
                "m_act" => {
                    let g = f16_vec(buf, n * 20480);
                    put(name, cols(&g, 20480, 0, 10240), load(dir, &c.name, "m_act"));
                }
                "features" => put(name, f32_vec(buf, n * 4096), load(dir, &c.name, "features")),
                n if n.starts_with("block_") => put(n, f32_vec(buf, p * 1024), load(dir, &c.name, n)),
                n => {
                    let b: usize = n[1..3].parse().unwrap();
                    let (bq, bp, bg, bu, bd) = &biases[&b];
                    let key = &n[4..];
                    let f = |s: &str| load(dir, &c.name, &format!("d{b:02}_{s}"));
                    match key {
                        "norm1_out" => put(n, f16_vec(buf, p * 1024), f("norm1_out")),
                        "qkv_raw" => {
                            let raw = f16_vec(buf, p * 3072);
                            put(&format!("d{b:02}_q_proj"), f16_add(&cols(&raw, 3072, 0, 1024), &bq[..1024], 1024), f("q_proj"));
                            put(
                                &format!("d{b:02}_k_proj"),
                                f16_add(&cols(&raw, 3072, 1024, 1024), &bq[1024..2048], 1024),
                                f("k_proj"),
                            );
                        }
                        "qkv" => {
                            let v = f16_vec(buf, p * 3072);
                            put(&format!("d{b:02}_q_rope"), cols(&v, 3072, 0, 1024), f("q_rope"));
                            put(&format!("d{b:02}_k_rope"), cols(&v, 3072, 1024, 1024), f("k_rope"));
                            put(&format!("d{b:02}_v"), cols(&v, 3072, 2048, 1024), f("v"));
                        }
                        "attn_core" => put(n, f16_vec(buf, p * 1024), f("attn_core")),
                        "attn_out_raw" => {
                            put(&format!("d{b:02}_attn_out"), f16_add(&f16_vec(buf, p * 1024), bp, 1024), f("attn_out"))
                        }
                        "norm2_out" => put(n, f16_vec(buf, p * 1024), f("norm2_out")),
                        "gu_raw" => {
                            let raw = f16_vec(buf, p * 8192);
                            put(&format!("d{b:02}_mlp_gate"), f16_add(&cols(&raw, 8192, 0, 4096), bg, 4096), f("mlp_gate"));
                            put(&format!("d{b:02}_mlp_up"), f16_add(&cols(&raw, 8192, 4096, 4096), bu, 4096), f("mlp_up"));
                        }
                        "mlp_act" => put(n, cols(&f16_vec(buf, p * 8192), 8192, 0, 4096), f("mlp_act")),
                        "mlp_out_raw" => {
                            put(&format!("d{b:02}_mlp_out"), f16_add(&f16_vec(buf, p * 1024), bd, 1024), f("mlp_out"))
                        }
                        _ => {}
                    }
                }
            }
        };
        t.encode_observed(&c.still, c.grid, &mut out, &st, &mut obs).unwrap();
    }
    st.sync().unwrap();
    t.teacher = None;
    (table, f32_vec(&out, n * 4096))
}

/// A stage name without its block number: `d11_mlp_act` -> `d_mlp_act`, `block_07` -> `block_`.
fn kind(stage: &str) -> String {
    stage.chars().filter(|c| !c.is_ascii_digit()).collect()
}

/// Whether the pack under test stores an EXL3 tower (the sc_4.10bpw pack, judged against exllamav3 1.6.2's dumps) rather than the bf16 one.
fn exl3_tower() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        pack_dir()
            .and_then(|d| gb_model::families::glm::GlmConfig::from_dir(&d).ok())
            .and_then(|c| c.vision)
            .is_some_and(|v| v.tower != gb_model::families::glm::VisionTowerFormat::Bf16)
    })
}

/// The features gate: within this many times what the tower moves under 1%-of-the-values ulp noise (plus a small absolute allowance). 3 for the
/// bf16 tower against the fork. 4 for the EXL3 tower against exllamav3 1.6.2: the two engines' EXL3 GEMMs round different operands (this tower the
/// rotated input of the rotated-basis W_q, exllamav3 at 1,024+ rows a fused original-basis fp16 W), a difference in each of the 172 GEMMs, and the
/// measured features sit at up to 3.4 times the floor (photo_like, 1-cos; 7 images, 2026-10-17).
fn feature_k() -> f64 {
    if exl3_tower() {
        4.0
    } else {
        3.0
    }
}

/// The bound on a stage's own error of the EXL3 tower against exllamav3 1.6.2 (relative L2), measured on the 7-image set of the sc_4.10bpw dumps as
/// the worst case with the unmutated tower, times about 2 (the bf16 tower's method). The dense stages (patch embedding, downsample) and the norms keep
/// the bf16 tower's bounds; the block GEMMs differ by the two engines' EXL3 formulations (see `feature_k`), about 3x the bf16 tower's dense error.
fn stage_bound_exl3(k: &str) -> Option<f64> {
    Some(match k {
        "patch_embed" => 2e-6,                                   // 7.6e-7
        "block_" => 3e-3,                                        // 1.59e-3
        "d_q_proj" | "d_k_proj" => 2e-3,                         // 9.9e-4, 9.1e-4
        "d_q_rope" | "d_k_rope" | "d_v" | "d_attn_core" => 3e-3, // 1.16e-3, 1.08e-3, 1.38e-3, 1.56e-3
        "d_norm_out" => 2.5e-3,                                  // 1.10e-3
        "d_attn_out" => 4.5e-3,                                  // 2.19e-3
        "d_mlp_gate" => 3e-3,                                    // 1.55e-3
        "d_mlp_up" => 4e-3,                                      // 1.95e-3
        "d_mlp_act" | "d_mlp_out" => 5e-3,                       // 2.42e-3, 2.37e-3
        "post_layernorm" => 4e-4,                                // 2.1e-4
        "downsample" | "m_proj" => 9e-4,                         // 2.9e-5, 4.6e-4 (img_cap 5.3e-4)
        "m_norm" => 1.2e-3,                                      // 5.0e-4 (img_cap 5.9e-4)
        "m_act" => 5e-3,                                         // 2.6e-3 (px_1x1)
        "features" => 5.5e-3,                                    // 2.0e-3 (img_cap 2.7e-3)
        _ => return None,
    })
}

/// The bound on a stage's own error (relative L2 against the fork), measured on the 16-image set as the worst case with the unmutated tower, times about 2.
/// `None`: not a bounded stage (the accumulated blocks, judged by `the_tower_follows_the_fork_stage_by_stage`). An EXL3 tower: `stage_bound_exl3`.
fn stage_bound(k: &str) -> Option<f64> {
    if exl3_tower() {
        return stage_bound_exl3(k);
    }
    // measured worst case over the 16 images (stages alone) -> bound; the measurements are in the doc comment of the test that prints them
    Some(match k {
        "patch_embed" => 2e-6, // 7.9e-7
        "block_" => 8e-4,      // 3.9e-4
        // the sublayers of blocks 0, 11 and 23, each with the fork's stream in
        "d_q_proj" | "d_k_proj" => 3e-4,                         // 8.1e-5, 9.5e-5
        "d_q_rope" | "d_k_rope" | "d_v" | "d_attn_core" => 3e-4, // 1.1e-4, 1.3e-4, 1.3e-4, 1.5e-4
        "d_norm_out" => 4e-4,                                    // 2.0e-4
        "d_attn_out" => 6e-4,                                    // 2.8e-4
        "d_mlp_gate" => 7e-4,                                    // 3.2e-4
        "d_mlp_up" => 8e-4,                                      // 3.8e-4
        "d_mlp_act" | "d_mlp_out" => 1.2e-3,                     // 5.7e-4, 6.0e-4
        // the merger from the fork's post norm
        "post_layernorm" => 4e-4,        // 2.1e-4
        "downsample" | "m_proj" => 9e-4, // 4.6e-4, 4.7e-4
        "m_norm" => 1e-3,                // 5.2e-4
        "m_act" => 4e-3,                 // 2.0e-3 (a 25-token image; img_cap: 4e-4)
        "features" => 3e-3,              // 1.5e-3
        _ => return None,
    })
}

// ------------------------------------------------------------------------------------------------------------------ tests

#[test]
#[ignore]
fn every_stage_alone_is_within_its_bound() {
    let Some((_, dir)) = env() else { return };
    let Some(mut t) = tower() else { return };
    let mut worst: BTreeMap<String, (M, String)> = BTreeMap::new();
    for c in cases(&dir).iter().filter(|c| c.patches <= 8000) {
        let (table, _) = run_case(&mut t, &dir, c, &[0, 11, 23], Feed::Alone, true);
        for (k, m) in &table {
            let w = worst.entry(kind(k)).or_insert((M::default(), String::new()));
            if m.rel_l2 >= w.0.rel_l2 {
                *w = (*m, format!("{} {k}", c.name));
            }
        }
    }
    let mut bad = Vec::new();
    eprintln!("{:16} {:>9} {:>9} | worst case", "stage (alone)", "bound", "measured");
    for (k, (m, who)) in &worst {
        let b = stage_bound(k);
        eprintln!(
            "{k:16} {:>9} {:>9.2e} | {who}   1-cos {:.1e}",
            b.map_or("-".into(), |b| format!("{b:.0e}")),
            m.rel_l2,
            m.one_minus_cos
        );
        if let Some(b) = b {
            if m.rel_l2 > b {
                bad.push(format!("{k}: {:.2e} > {b:.0e} ({who})", m.rel_l2));
            }
        }
    }
    // the images above 8,000 patches: img_cap, from the fork's post norm
    for c in cases(&dir).iter().filter(|c| c.patches > 8000) {
        let (d2, n2) = (dir.clone(), c.name.clone());
        t.teacher = Some(Box::new(move |name| (name == "post_layernorm").then(|| load(&d2, &n2, name)).flatten()));
        let st = Stream::new().unwrap();
        let mut out = DevBuf::zeroed(c.tokens * 4096 * 4).unwrap();
        let mut got: BTreeMap<String, M> = BTreeMap::new();
        {
            let mut obs = |name: &str, buf: &DevBuf, st: &Stream| {
                if ["downsample", "m_proj", "m_norm", "features"].contains(&name) {
                    st.sync().unwrap();
                    let v = if name == "features" { f32_vec(buf, c.tokens * 4096) } else { f16_vec(buf, c.tokens * 4096) };
                    got.insert(name.to_string(), metrics(&v, &load(&dir, &c.name, name).unwrap()));
                }
            };
            t.encode_observed(&c.still, c.grid, &mut out, &st, &mut obs).unwrap();
        }
        t.teacher = None;
        for (k, m) in &got {
            eprintln!("{} {k} (merger from the fork's post norm): {}", c.name, fmt(m));
            if let Some(b) = stage_bound(k) {
                if m.rel_l2 > b {
                    bad.push(format!("{} {k}: {:.2e} > {b:.0e}", c.name, m.rel_l2));
                }
            }
        }
    }
    assert!(bad.is_empty(), "stages above their bound: {bad:#?}");
}

/// The features of `c` through a second run whose input is jittered by +-1 ulp on `q` of the values: the tower's own response to noise of the size of fp16
/// rounding (the floor of any comparison of two arithmetically different implementations of this tower).
fn jittered(t: &mut GlmVisionTower, c: &Case, q: f64) -> Vec<f32> {
    let mut s = 0x9E3779B97F4A7C15u64 ^ (c.patches as u64);
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let jit: Vec<u16> = c
        .still
        .iter()
        .map(|&h| {
            if (next() >> 11) as f64 / (1u64 << 53) as f64 >= q || h & 0x7fff == 0 {
                h
            } else if next() & 1 == 0 {
                h + 1
            } else {
                h - 1
            }
        })
        .collect();
    let st = Stream::new().unwrap();
    let mut out = DevBuf::zeroed(c.tokens * 4096 * 4).unwrap();
    t.encode(&jit, c.grid, &mut out, &st).unwrap();
    st.sync().unwrap();
    f32_vec(&out, c.tokens * 4096)
}

#[test]
#[ignore]
fn the_tower_follows_the_fork_stage_by_stage() {
    let Some((_, dir)) = env() else { return };
    let Some(mut t) = tower() else { return };
    let cases = cases(&dir);
    assert!(!cases.is_empty());
    let mut growth: Vec<(String, Vec<f64>)> = Vec::new();
    let mut failures = Vec::new();
    eprintln!("{:>16} | {:>22} | {:>22} | features vs the fork's", "image", "tower vs fork", "tower vs tower+1% noise");
    for c in &cases {
        let t0 = std::time::Instant::now();
        let (table, feats) = run_case(&mut t, &dir, c, &[], Feed::Full, true);
        let want = load(&dir, &c.name, "features").unwrap();
        let f = metrics(&feats, &want);
        // the floor: the same tower against itself with +-1 ulp on 1% of the input values
        let floor = metrics(&jittered(&mut t, c, 0.01), &feats);
        eprintln!(
            "{:>16} | rel {:.2e} 1-cos {:.1e} | rel {:.2e} 1-cos {:.1e} | {}x{} patches = {} tokens ({:.1?})",
            c.name,
            f.rel_l2,
            f.one_minus_cos,
            floor.rel_l2,
            floor.one_minus_cos,
            c.grid.0,
            c.grid.1,
            c.tokens,
            t0.elapsed()
        );
        let mut blocks = vec![f64::NAN; 24];
        for (k, m) in &table {
            if let Some(b) = k.strip_prefix("block_") {
                blocks[b.parse::<usize>().unwrap()] = m.rel_l2;
            }
        }
        if let (Some(b), Some(pe)) = (stage_bound("patch_embed"), table.get("patch_embed")) {
            if pe.rel_l2 > b {
                failures.push(format!("{}: patch_embed {:.2e} > {b:.0e}", c.name, pe.rel_l2));
            }
        }
        growth.push((c.name.clone(), blocks));
        // the features: within 3x of what the tower itself moves under 1%-of-the-values ulp noise, plus a small absolute allowance
        let (b_rel, b_cos) = (feature_k() * floor.rel_l2 + 2e-3, feature_k() * floor.one_minus_cos + 2e-5);
        if f.rel_l2 > b_rel || f.one_minus_cos > b_cos {
            failures.push(format!(
                "{}: features {} against the floor-based bound rel {b_rel:.2e}, 1-cos {b_cos:.1e}",
                c.name,
                fmt(&f)
            ));
        }
    }
    eprintln!("\nper-block error growth (rel L2 of the fp32 residual stream against the fork's, the tower fed its own stream):");
    eprint!("{:>16}", "image");
    for b in 0..24 {
        eprint!(" {b:>8}");
    }
    eprintln!();
    for (name, blocks) in &growth {
        eprint!("{name:>16}");
        for v in blocks {
            if v.is_nan() {
                eprint!(" {:>8}", "-");
            } else {
                eprint!(" {v:>8.1e}");
            }
        }
        eprintln!();
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The images whose bounds are recorded, not gated (PASS_RULES_GLM_ADDENDUM_ORACLE D-O6: px_1x1, tiny_3x5, thin_5x700, frozen 2026-10-17).
const DEGENERATE: [&str; 3] = ["px_1x1", "tiny_3x5", "thin_5x700"];

/// | ||got|| / ||want|| - 1 |, the features' norm-ratio check of the manifest (`tools/glm_derive_vision_tolerances.py`).
fn norm_ratio(got: &[f32], want: &[f32]) -> f64 {
    let n = |v: &[f32]| v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let w = n(want);
    if w == 0.0 {
        return if n(got) == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (n(got) / w - 1.0).abs()
}

/// R-G-oracle (d) as frozen (PASS_RULES_GLM_ADDENDUM_ORACLE D-O5..D-O7): the EXL3 tower, fed its own stream (as the reference's jittered
/// runs were), against the reference's clean dumps, every tap judged by THE MANIFEST's per-image bounds (`tolerances_by_image`: rel L2,
/// 1 - cos, and for the features the norm ratio), never by bounds in code. The non-degenerate images gate; the degenerate ones (`DEGENERATE`)
/// and the all-image envelope (`tolerances`) are printed, not gated. Needs `REXL3_GLM_VISION_MANIFEST` (a manifest with
/// `tolerances_by_image`, e.g. testdata/glm53-sc410/vision_manifest.json) and `REXL3_GLM_VISION_DIR` (its clean dumps); "not run"
/// otherwise. Every image of the manifest must have its dumps (a missing one fails: nothing is skipped silently). The patches are judged by
/// `the_patches_the_tower_reads_are_the_forks` (byte-identical).
#[test]
#[ignore]
fn the_tower_is_within_the_manifests_per_image_bounds() {
    if std::env::var("REXL3_GLM_VISION_MANIFEST").is_err() {
        eprintln!("REXL3_GLM_VISION_MANIFEST is not set: not run (the per-image bounds are a manifest's, never the code's)");
        return;
    }
    let m = manifest();
    let Some(by_image) = m["tolerances_by_image"].as_object() else {
        panic!("the manifest has no `tolerances_by_image`: it cannot judge R-G-oracle (d)");
    };
    let envelope = m["tolerances"].as_object().expect("the manifest has no `tolerances` (the envelope)");
    let Some((_, dir)) = env() else { return };
    let Some(mut t) = tower() else { return };
    let cases = cases(&dir);
    let listed: Vec<&String> = m["images"].as_object().unwrap().keys().collect();
    let missing: Vec<&&String> = listed.iter().filter(|n| !cases.iter().any(|c| &c.name == **n)).collect();
    assert!(missing.is_empty(), "images of the manifest without dumps in {}: {missing:?}", dir.display());
    let bound = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_f64());
    let (mut failures, mut judged, mut informational) = (Vec::new(), 0usize, Vec::new());
    let mut unobserved: BTreeMap<String, usize> = BTreeMap::new();
    let mut unbounded: BTreeMap<String, usize> = BTreeMap::new();
    let mut outside_envelope = 0usize;
    eprintln!(
        "{:>16} {:>5} {:>16} | {:>9} {:>9} | {:>9} {:>9} | verdict",
        "image", "gate", "tap", "rel_l2", "bound", "1-cos", "bound"
    );
    for c in &cases {
        let gated = !DEGENERATE.contains(&c.name.as_str());
        let bounds = by_image.get(&c.name).unwrap_or_else(|| panic!("{}: no per-image bounds in the manifest", c.name));
        let (table, feats) = run_case(&mut t, &dir, c, &[0, 11, 23], Feed::Full, true);
        let want = load(&dir, &c.name, "features").unwrap();
        for tap in bounds.as_object().unwrap().keys() {
            if !table.contains_key(tap) {
                *unobserved.entry(tap.clone()).or_default() += 1;
            }
        }
        for (tap, got) in &table {
            let Some(b) = bounds.get(tap) else {
                *unbounded.entry(tap.clone()).or_default() += 1;
                continue;
            };
            let (b_rel, b_cos) = (bound(b, "rel_l2").unwrap(), bound(b, "one_minus_cos").unwrap());
            let mut why = Vec::new();
            if got.rel_l2 > b_rel {
                why.push(format!("rel_l2 {:.3e} > {b_rel:.3e}", got.rel_l2));
            }
            if got.one_minus_cos > b_cos {
                why.push(format!("1-cos {:.3e} > {b_cos:.3e}", got.one_minus_cos));
            }
            let mut nr_txt = String::new();
            if tap == "features" {
                let nr = norm_ratio(&feats, &want);
                let b_nr = bound(b, "norm_ratio").expect("the features' bounds have no norm_ratio");
                nr_txt = format!(" norm_ratio {nr:.3e} (bound {b_nr:.3e})");
                if nr > b_nr {
                    why.push(format!("norm_ratio {nr:.3e} > {b_nr:.3e}"));
                }
            }
            if let Some(e) = envelope.get(tap) {
                if got.rel_l2 > bound(e, "rel_l2").unwrap_or(f64::INFINITY)
                    || got.one_minus_cos > bound(e, "one_minus_cos").unwrap_or(f64::INFINITY)
                {
                    outside_envelope += 1;
                }
            }
            let verdict = if why.is_empty() { "within".to_string() } else { why.join("; ") };
            if tap == "features"
                || !why.is_empty()
                || ["patch_embed", "block_23", "post_layernorm", "m_act"].contains(&tap.as_str())
            {
                eprintln!(
                    "{:>16} {:>5} {:>16} | {:>9.3e} {b_rel:>9.3e} | {:>9.3e} {b_cos:>9.3e} | {verdict}{nr_txt}",
                    c.name,
                    if gated { "gate" } else { "info" },
                    tap,
                    got.rel_l2,
                    got.one_minus_cos
                );
            }
            if gated {
                judged += 1;
                if !why.is_empty() {
                    failures.push(format!("{} {tap}: {}", c.name, why.join("; ")));
                }
            } else if !why.is_empty() {
                informational.push(format!("{} {tap}: {}", c.name, why.join("; ")));
            }
        }
    }
    let gated_images = cases.iter().filter(|c| !DEGENERATE.contains(&c.name.as_str())).count();
    eprintln!(
        "\n{} images ({gated_images} gated, degenerate informational: {DEGENERATE:?}); {judged} gated tap checks; {} gated failures; {} \
         informational (degenerate) excursions: {informational:?}",
        cases.len(),
        failures.len(),
        informational.len()
    );
    eprintln!("taps of the manifest the instrument does not observe (image count): {unobserved:?}");
    eprintln!("taps observed without a per-image bound (image count): {unbounded:?}");
    eprintln!("tap checks outside the all-image envelope (recorded, not gated): {outside_envelope}");
    assert!(gated_images >= 13 && judged > 0, "only {gated_images} gated images, {judged} tap checks");
    assert!(failures.is_empty(), "taps outside their per-image bound: {failures:#?}");
}

/// DX-1 (PASS_RULES_GLM_ADDENDUM_AMEND1 D-A1, a post-hoc diagnostic of R-G-oracle (d); it judges nothing): the tower's taps exactly as the
/// (d) judge `the_tower_is_within_the_manifests_per_image_bounds` computes them (`run_case`, the tower fed its own stream, detail blocks 0, 11
/// and 23, every dumped block), written as flat f32 `.npy` files by `REXL3_GLM_VISION_SAVE=DIR` (required here: `DIR/<image>/<tap>.npy`; the
/// fp16 taps widened exactly, the bias-added ones after the judge's host fp16 add), for the images of `REXL3_GLM_VISION_ONLY` (required: the
/// judge refuses a partial image set, this test exists to dump a few). Prints the same per-tap metrics against the reference's dumps, so a
/// run can be checked against the (d) leg's table. `tools/glm_dx1_compare.py` reads the files beside exllamav3 1.6.2's dumps and the fp64
/// host forward (`tools/glm_vision_fp64.py`). Needs REXL3_GLM_DIR (the pack), REXL3_GLM_VISION_DIR (the reference dumps) and
/// REXL3_GLM_VISION_MANIFEST (their manifest); "not run" otherwise. One GPU, under the lock.
#[test]
#[ignore]
fn dx1_the_towers_taps_are_dumped() {
    let (Ok(save), Ok(only)) = (std::env::var("REXL3_GLM_VISION_SAVE"), std::env::var("REXL3_GLM_VISION_ONLY")) else {
        eprintln!("REXL3_GLM_VISION_SAVE and REXL3_GLM_VISION_ONLY are not both set: not run");
        return;
    };
    if std::env::var("REXL3_GLM_VISION_MANIFEST").is_err() {
        eprintln!("REXL3_GLM_VISION_MANIFEST is not set: not run");
        return;
    }
    assert!(!Path::new(&save).exists(), "REXL3_GLM_VISION_SAVE={save} exists: a new directory every run");
    let Some((_, dir)) = env() else { return };
    let Some(mut t) = tower() else { return };
    let cases = cases(&dir);
    let want: Vec<&str> = only.split(',').collect();
    let names: Vec<&str> = cases.iter().map(|c| c.name.as_str()).collect();
    assert!(want.iter().all(|w| names.contains(w)), "images of REXL3_GLM_VISION_ONLY without dumps: {want:?} vs {names:?}");
    for c in &cases {
        let t0 = std::time::Instant::now();
        let (table, _) = run_case(&mut t, &dir, c, &[0, 11, 23], Feed::Full, true);
        let files = std::fs::read_dir(Path::new(&save).join(&c.name)).map(|d| d.count()).unwrap_or(0);
        eprintln!(
            "{}: {} patches, {} taps compared, {files} files written in {:.1?}",
            c.name,
            c.patches,
            table.len(),
            t0.elapsed()
        );
        for (tap, m) in &table {
            eprintln!("  {:>16} {:>16} {}", c.name, tap, fmt(m));
        }
    }
}

/// One tower per mutation, on a small scratch, judged by the stages alone on three images and by the features against the floor-based bound.
#[test]
#[ignore]
fn every_mutation_is_caught() {
    let Some((_, dir)) = env() else { return };
    let dir_pack = pack_dir().unwrap();
    let pack = GlmFamily::map_pack(&dir_pack).unwrap();
    let cases: Vec<Case> =
        cases(&dir).into_iter().filter(|c| ["img_small", "photo_like", "rgba_soft"].contains(&c.name.as_str())).collect();
    assert_eq!(cases.len(), 3);
    require_available(20 * GIB);
    // the unmutated tower first: its feature floors and the control that the checks below pass for it
    let floors: BTreeMap<String, M> = {
        let mut o = TowerOptions::new(4_000, 8 * GIB);
        o.mutant = None;
        let mut t = GlmVisionTower::load(&pack, &dir_pack, &o).unwrap();
        cases
            .iter()
            .map(|c| {
                (c.name.clone(), {
                    let (_, f) = run_case(&mut t, &dir, c, &[], Feed::Full, false);
                    metrics(&jittered(&mut t, c, 0.01), &f)
                })
            })
            .collect()
    };
    let mut survivors = Vec::new();
    eprintln!(
        "{:36} | {:>6} stages over | first (stage, image: measured / bound) | features rel / 1-cos (bound)",
        "mutation", ""
    );
    let mut judged: Vec<Option<Mutant>> = vec![None];
    // the fold and merge-order mutants act on the bf16 tower's device preparation; an EXL3 tower's fold and permutation are the plan's host
    // conversions (`load::convert`, tested on the CPU), and its preparation refuses those mutants
    let device_prep = [Mutant::FoldSlot0Only, Mutant::MergeOrderConv, Mutant::MergeOrderTransposed];
    let exl3 = pack.weights.vision.as_ref().is_some_and(|v| v.format != gb_model::families::glm::VisionTowerFormat::Bf16);
    judged.extend(Mutant::ALL.iter().filter(|m| !(exl3 && device_prep.contains(m))).map(|m| Some(*m)));
    for m in judged {
        let mut o = TowerOptions::new(4_000, 8 * GIB);
        o.mutant = m;
        let mut t = GlmVisionTower::load(&pack, &dir_pack, &o).unwrap();
        let (mut over, mut first, mut feat) = (0usize, String::new(), String::new());
        let mut feat_over = false;
        for c in &cases {
            let (table, _) = run_case(&mut t, &dir, c, &[0, 11, 23], Feed::Alone, false);
            for (k, mm) in &table {
                if let Some(b) = stage_bound(&kind(k)) {
                    if mm.rel_l2 > b {
                        over += 1;
                        if first.is_empty() {
                            first = format!("{k}, {}: {:.1e} / {b:.0e}", c.name, mm.rel_l2);
                        }
                    }
                }
            }
            let (_, f) = run_case(&mut t, &dir, c, &[], Feed::Full, false);
            let fm = metrics(&f, &load(&dir, &c.name, "features").unwrap());
            let fl = floors[&c.name];
            let (b_rel, b_cos) = (feature_k() * fl.rel_l2 + 2e-3, feature_k() * fl.one_minus_cos + 2e-5);
            if fm.rel_l2 > b_rel || fm.one_minus_cos > b_cos {
                feat_over = true;
            }
            if feat.is_empty() || fm.rel_l2 > 0.0 {
                feat = format!("{} {:.1e} / {:.1e} ({:.1e} / {:.1e})", c.name, fm.rel_l2, fm.one_minus_cos, b_rel, b_cos);
            }
        }
        let name = m.map_or("(none: the unmutated tower)", |m| m.name());
        eprintln!(
            "{name:36} | {over:>6} stages over | {} | {feat}{}",
            if first.is_empty() { "-".into() } else { first },
            if feat_over { "  FEATURES OVER" } else { "" }
        );
        match m {
            None => assert!(over == 0 && !feat_over, "the control (no mutation) fails its own checks"),
            Some(m) if over == 0 && !feat_over => survivors.push(m),
            Some(_) => {}
        }
        drop(t);
    }
    // What the tower-level checks cannot see on these 16 images (the effect is under fp16 rounding in every stage) is killed one level down, by a kernel
    // test that drives the very same kernel argument the mutation sets, on an input where it matters (`crates/cuda/tests/glm_vision.rs`); one is
    // equivalent. Anything else that survives is a hole.
    let known: &[(Mutant, &str)] = &[
        (Mutant::RmsEps1e6, "kernel test rmsnorm_matches_the_formula_and_its_eps_is_the_forks (a near-constant row, 2.4x apart) and the constants test"),
        (Mutant::LnEps1e5, "kernel test layernorm_matches_the_formula_and_its_eps_is_the_forks (a near-constant row) and the constants test"),
        (Mutant::GeluTanh, "kernel test gelu_is_the_exact_erf_form_and_the_tanh_form_is_not_it (every fp16 input in [-9, 9])"),
        (Mutant::MergerNoClamp, "kernel test the_merger_activation_clamps_before_the_silu_in_fp32 (a sweep to 40)"),
        (Mutant::BlockClampBeforeSilu, "EQUIVALENT: identical fp16 outputs on a sweep to 40 (the_block_activation_clamps_..._are_the_same_in_fp16)"),
    ];
    eprintln!("\nsurvivors of the tower-level checks and what kills them:");
    for m in &survivors {
        match known.iter().find(|k| k.0 == *m) {
            Some(k) => eprintln!("  {:36} {}", m.name(), k.1),
            None => panic!("mutation `{}` survives every check and no lower-level test is named for it", m.name()),
        }
    }
}

#[test]
#[ignore]
fn the_noise_floor_of_the_features() {
    let Some((_, dir)) = env() else { return };
    let Some(mut t) = tower() else { return };
    let cases = cases(&dir);
    eprintln!("{:>16} {:>9} | {:>22} | features vs the fork's", "image", "fraction", "vs the unperturbed run");
    for c in cases.iter().filter(|c| {
        ["img_small", "photo_like", "noise_257x255", "odd_331x257", "tall_40x2500", "exact_280x196"].contains(&c.name.as_str())
    }) {
        let st = Stream::new().unwrap();
        let mut out = DevBuf::zeroed(c.tokens * 4096 * 4).unwrap();
        t.encode(&c.still, c.grid, &mut out, &st).unwrap();
        st.sync().unwrap();
        let base = f32_vec(&out, c.tokens * 4096);
        let f = metrics(&base, &load(&dir, &c.name, "features").unwrap());
        for q in [1.0f64, 0.1, 0.01, 0.001] {
            let m = metrics(&jittered(&mut t, c, q), &base);
            eprintln!(
                "{:>16} {q:>9} | rel_l2 {:.2e} 1-cos {:.2e} | fork: rel_l2 {:.2e} 1-cos {:.2e}",
                c.name, m.rel_l2, m.one_minus_cos, f.rel_l2, f.one_minus_cos
            );
        }
    }
}

#[test]
#[ignore]
fn encode_time() {
    let Some((_, dir)) = env() else { return };
    let Some(mut t) = tower() else { return };
    let cases = cases(&dir);
    let st = Stream::new().unwrap();
    for c in &cases {
        if !["img_small", "photo_like", "img_cap"].contains(&c.name.as_str()) {
            continue;
        }
        let mut out = DevBuf::zeroed(c.tokens * 4096 * 4).unwrap();
        t.encode(&c.still, c.grid, &mut out, &st).unwrap();
        st.sync().unwrap();
        let mut times = Vec::new();
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            t.encode(&c.still, c.grid, &mut out, &st).unwrap();
            st.sync().unwrap();
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // FLOPs: the GEMMs 2 * (patch 592 + 24 blocks * (4 h^2 + 3 h i)) per patch, the merger's per token, and attention 4 * P^2 * h per block
        let (pp, nn) = (c.patches as f64, c.tokens as f64);
        let gemm = pp * 2.0 * (592.0 * 1024.0 + 24.0 * (4.0 * 1024.0 * 1024.0 + 3.0 * 1024.0 * 4096.0))
            + nn * 2.0 * (4096.0 * 4096.0 * 2.0 + 2.0 * 4096.0 * 10240.0 + 10240.0 * 4096.0);
        let attn = 24.0 * 4.0 * pp * pp * 1024.0;
        let med = times[times.len() / 2];
        eprintln!(
            "{}: {} patches / {} tokens: encode median {:.1} ms (min {:.1}, max {:.1}); {:.2} TFLOP GEMM + {:.2} TFLOP attention = {:.1} TFLOP/s overall",
            c.name,
            c.patches,
            c.tokens,
            med,
            times[0],
            times[times.len() - 1],
            gemm / 1e12,
            attn / 1e12,
            (gemm + attn) / 1e12 / (med / 1e3)
        );
    }
}

/// Where the encode time goes: the stream is synchronised after every stage (a few tens of microseconds each, nothing against the stages of a large
/// image) and the time between two observations is charged to the stage that ran, summed over the 24 blocks, with the effective TFLOP/s of the GEMMs
/// and of the attention. The roofline gap in `docs/GLM53.md` is this table against `cublas` at the same shapes and the tensor-core peak.
#[test]
#[ignore]
fn encode_profile() {
    let Some((_, dir)) = env() else { return };
    let Some(mut t) = tower() else { return };
    let cases = cases(&dir);
    let st = Stream::new().unwrap();
    for c in cases.iter().filter(|c| ["img_small", "img_cap"].contains(&c.name.as_str())) {
        let mut out = DevBuf::zeroed(c.tokens * 4096 * 4).unwrap();
        t.encode(&c.still, c.grid, &mut out, &st).unwrap();
        st.sync().unwrap();
        let (p, n) = (c.patches as f64, c.tokens as f64);
        // (stage that ENDS at the observation, class)
        let class = |name: &str| -> Option<&'static str> {
            Some(match name.split_once('_').map_or(name, |x| x.1) {
                "qkv_raw" => "qkv gemm",
                "qkv" => "qkv prep (bias, head norm, rope)",
                "attn_core" => "attention",
                "attn_out_raw" => "proj gemm",
                "norm2_out" => "residual add + norm2",
                "gu_raw" => "gate+up gemms",
                "mlp_act" => "swiglu",
                "mlp_out_raw" => "down gemm",
                _ if name.starts_with("block_") => "residual add + norm1 / post norm",
                _ => return None,
            })
        };
        let mut classes: BTreeMap<&'static str, f64> = BTreeMap::new();
        let mut rest: BTreeMap<String, f64> = BTreeMap::new();
        let mut last = std::time::Instant::now();
        let t_start = last;
        {
            let mut obs = |name: &str, _b: &DevBuf, s: &Stream| {
                s.sync().unwrap();
                let now = std::time::Instant::now();
                let dt = now.duration_since(last).as_secs_f64() * 1e3;
                last = now;
                match class(name) {
                    Some(k) => *classes.entry(k).or_default() += dt,
                    None if name.ends_with("norm1_out") => {} // the start of a block: the time since the previous stage is that stage's, already charged
                    None => *rest.entry(name.to_string()).or_default() += dt,
                }
            };
            t.encode_observed(&c.still, c.grid, &mut out, &st, &mut obs).unwrap();
        }
        let total = t_start.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "\n{}: {} patches, {} tokens, {total:.1} ms with a sync after each of {} stages",
            c.name,
            c.patches,
            c.tokens,
            24 * 9 + 8
        );
        // FLOP of a class over the 24 blocks
        let flops = |k: &str| -> Option<f64> {
            Some(match k {
                "qkv gemm" => 24.0 * p * 2.0 * 1024.0 * 3072.0,
                "proj gemm" => 24.0 * p * 2.0 * 1024.0 * 1024.0,
                "gate+up gemms" => 24.0 * p * 2.0 * 1024.0 * 8192.0,
                "down gemm" => 24.0 * p * 2.0 * 4096.0 * 1024.0,
                "attention" => 24.0 * 4.0 * p * p * 1024.0,
                _ => return None,
            })
        };
        let mut sum = 0.0;
        for (k, ms) in &classes {
            sum += ms;
            match flops(k) {
                Some(f) => eprintln!("  {k:36} {ms:9.2} ms {:5.1}%  {:6.1} TFLOP/s", 100.0 * ms / total, f / 1e12 / (ms / 1e3)),
                None => eprintln!("  {k:36} {ms:9.2} ms {:5.1}%", 100.0 * ms / total),
            }
        }
        let merger = ["downsample", "m_proj", "m_norm", "m_gelu", "m_act", "features"];
        let mut mt = 0.0;
        for k in merger {
            mt += rest.get(k).copied().unwrap_or(0.0);
        }
        let mflop = n * 2.0 * (4096.0 * 4096.0 * 2.0 + 2.0 * 4096.0 * 10240.0 + 10240.0 * 4096.0);
        eprintln!(
            "  {:36} {mt:9.2} ms {:5.1}%  {:6.1} TFLOP/s",
            "downsample + merger",
            100.0 * mt / total,
            mflop / 1e12 / (mt / 1e3)
        );
        eprintln!("  {:36} {:9.2} ms (patch embed, the first norm, the uploads)", "the rest", total - sum - mt);
    }
}

#[test]
#[ignore]
fn an_image_is_a_function_of_itself() {
    let Some((_, dir)) = env() else { return };
    let Some(mut t) = tower() else { return };
    let cases = cases(&dir);
    let find = |n: &str| cases.iter().find(|c| c.name == n).unwrap();
    let (a, b) = (find("img_small"), find("photo_like"));
    let st = Stream::new().unwrap();
    let run = |t: &mut GlmVisionTower, c: &Case| -> Vec<f32> {
        let mut out = DevBuf::zeroed(c.tokens * 4096 * 4).unwrap();
        t.encode(&c.still, c.grid, &mut out, &st).unwrap();
        st.sync().unwrap();
        f32_vec(&out, c.tokens * 4096)
    };
    let a1 = run(&mut t, a);
    let b1 = run(&mut t, b);
    let a2 = run(&mut t, a);
    let b2 = run(&mut t, b);
    assert!(a1.iter().zip(&a2).all(|(x, y)| x.to_bits() == y.to_bits()), "A, B, A: the first image changed");
    assert!(b1.iter().zip(&b2).all(|(x, y)| x.to_bits() == y.to_bits()), "B, A, B: the second image changed");
    // an image does not depend on what the scratch was sized for: a second tower with a scratch for 4,000 patches gives the same bits
    let pack = GlmFamily::map_pack(&pack_dir().unwrap()).unwrap();
    let mut small = GlmVisionTower::load(&pack, &pack_dir().unwrap(), &TowerOptions::new(4_000, 8 * GIB)).unwrap();
    let a3 = run(&mut small, a);
    assert!(a1.iter().zip(&a3).all(|(x, y)| x.to_bits() == y.to_bits()), "the features depend on the tower's scratch size");
    eprintln!(
        "A,B,A,B bit-identical; identical through a tower with a 4,000-patch scratch ({} and {} tokens)",
        a.tokens, b.tokens
    );
}

#[test]
#[ignore]
fn the_scratch_is_what_the_plan_says() {
    let Some(t) = tower() else { return };
    let plan = vision::scratch_items(t.max_patches, t.format);
    let got = t.scratch_allocations();
    assert_eq!(plan.iter().map(|i| (i.0, i.1)).collect::<Vec<_>>(), got, "the tower's allocations are not scratch_items");
    assert_eq!(t.scratch_bytes(), vision::scratch_bytes(t.max_patches, t.format));
    let pack = GlmFamily::map_pack(&pack_dir().unwrap()).unwrap();
    let cfg = pack.config.vision.as_ref().unwrap();
    assert_eq!(t.weight_bytes(), vision::weight_region_bytes(cfg), "the weights are the plan's region");
    assert_eq!(
        cfg.encoder_scratch_bytes(t.max_patches / 4),
        t.scratch_bytes(),
        "the memory model's encoder scratch is what is allocated"
    );
    eprintln!(
        "weights {} B, scratch {} B (max {} patches): equal to the plan",
        t.weight_bytes(),
        t.scratch_bytes(),
        t.max_patches
    );
}

/// The fork's `preprocess` output, from the dump, against this crate's Fork-profile preprocessing of the same PNG: the tower is fed the dumped patches
/// above, so this closes the loop between the two halves.
#[test]
#[ignore]
fn the_patches_the_tower_reads_are_the_forks() {
    let Some(dir) = dump_dir() else { return };
    let pp = gb_vision::glm::GlmPreprocessing::default();
    let m = manifest();
    let mut checked = 0;
    for (name, e) in m["images"].as_object().unwrap() {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(e["source"].as_str().unwrap());
        if !dir.join(name).join("patches.npy").exists() || !src.exists() {
            continue;
        }
        let img = pp.decode(&std::fs::read(&src).unwrap()).unwrap();
        let pre = pp.preprocess(&img).unwrap();
        let want = npy::read(&dir.join(name).join("patches.npy")).unwrap().to_u16().unwrap();
        assert_eq!(pre.patches.len(), want.len(), "{name}");
        assert!(pre.patches == want, "{name}: the Fork-profile patches differ from the fork's");
        checked += 1;
    }
    let listed = m["images"].as_object().unwrap().len();
    assert!(checked >= listed.min(14), "only {checked} images were checked");
    eprintln!("{checked} images: the Rust Fork-profile preprocessing equals the fork's patches bit for bit");
}
