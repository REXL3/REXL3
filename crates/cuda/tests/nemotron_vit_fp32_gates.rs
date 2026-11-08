//! The Nemotron 3.5 Super VL vision tower against the pack's fp32 run: the owner's fp32-based gates (signed off 2026-10-10, set out in
//! `docs/NEMOTRON_VISION_SDPA.md`), executable. The engine tower and projector run on the processor's pixels of 13 images and every
//! stage is compared with the HF reference classes run in fp32 on the same bf16-rounded pixels (the bf16 weights widened exactly, no
//! TF32). The bf16-based gates (the torch dumps, `nemotron_vit.rs`) stay as they are; these are the primary quality check.
//!
//! Env-gated (the test prints `not run: REXL3_NEMO_...` and passes without the variables, as the other Nemotron GPU tests do; the
//! Nemotron gate script runs it with them set and fails if it did not run):
//!
//! * `REXL3_NEMO_OMNI_DIR` (or `REXL3_NEMOTRON_PACK`): the pack directory (the vision tensors are read through its index).
//! * `REXL3_NEMO_VISION_FP32_DIR`: the node-local reference directory written by `tools/nemotron_omni/vision_sdpa_check.py refs`
//!   (`$HOME/projects/nemo-vision-fp32/` on node 4): per image `<name>.pixels.bf16`, `<name>.tower.f32`, `<name>.final_ln.f32`,
//!   `<name>.mlp1.f32` and `<name>.blocks.f32`. They hold activations of the pack's weights and are never committed.
//! * `REXL3_NEMO_VISION_FP32_RECORD=1`: also write `<dir>/engine_metrics.txt` (the engine's scalars) for
//!   `vision_sdpa_check.py manifest`, which composes the committed manifest.
//! * `REXL3_NEMO_VISION_FP32_MANIFEST=<file>`: a manifest other than the committed one (the first run, before `manifest` has made the
//!   committed one: `<dir>/refs.txt`).
//! * `REXL3_NEMO_VISION_FP32_SHA=off`: a reference file whose SHA-256 differs from the manifest's is reported, not a failure (a box
//!   whose cuBLAS makes other fp32 bits; the numeric gates still decide).
//!
//! The committed `testdata/gpu/nemotron_vision_fp32_gates.txt` holds scalars only: the image list, the SHA-256 and size of every
//! reference file (so the gates are run against exactly the references they were signed against), the bf16 eager model's own distance
//! to fp32 per stage (the floor) and the engine's numbers when the manifest was made (printed beside the current ones, not gated).
//!
//! Gates (rel_l2 against fp32 unless said; 1.7x to 2.9x the worst of the 13 images, measured on the tensor-core tower of `66a5ac1`:
//! 5.2e-4, 4.0e-4, 3.7e-3, 4.0e-3, 5.8e-3 rel_max and 1.4e-3 in the order below): tower features 1e-3; vision_final_layernorm 1e-3
//! (and the first and last rows of the pixel shuffle, which is a permutation of it); mlp1 as the tower leaves it (fp32) 8e-3; mlp1 as
//! the encoder serves it (rounded to bf16) rel_l2 1e-2 and rel_max 1e-2; the 33 per-block stages (cls, first patch and last token)
//! worst rel_l2 4e-3. A broken gate is reported per image and stage with its factor, and fails the test: it is never loosened here.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use gb_cuda::nemotron_vit as nv;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const GATE_TOWER: f64 = 1.0e-3;
const GATE_FINAL_LN: f64 = 1.0e-3;
const GATE_MLP1_RAW: f64 = 8.0e-3;
const GATE_MLP1_SERVED_L2: f64 = 1.0e-2;
const GATE_MLP1_SERVED_MAX: f64 = 1.0e-2;
const GATE_BLOCKS: f64 = 4.0e-3;

const MANIFEST: &str = "testdata/gpu/nemotron_vision_fp32_gates.txt";

// ------------------------------------------------------------------------------------------------------------------ SHA-256

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

struct Sha256 {
    h: [u32; 8],
    buf: Vec<u8>,
    len: u64,
}

impl Sha256 {
    fn new() -> Self {
        Sha256 {
            h: [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19],
            buf: Vec::with_capacity(64),
            len: 0,
        }
    }

    fn block(h: &mut [u32; 8], b: &[u8]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = *h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.len += data.len() as u64;
        if !self.buf.is_empty() {
            let take = (64 - self.buf.len()).min(data.len());
            self.buf.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.buf.len() == 64 {
                let b = std::mem::take(&mut self.buf);
                Self::block(&mut self.h, &b);
            }
        }
        while data.len() >= 64 {
            Self::block(&mut self.h, &data[..64]);
            data = &data[64..];
        }
        self.buf.extend_from_slice(data);
    }

    fn hex(mut self) -> String {
        let bits = self.len * 8;
        self.buf.push(0x80);
        while self.buf.len() % 64 != 56 {
            self.buf.push(0);
        }
        self.buf.extend_from_slice(&bits.to_be_bytes());
        let tail = std::mem::take(&mut self.buf);
        for c in tail.chunks_exact(64) {
            Self::block(&mut self.h, c);
        }
        self.h.iter().map(|w| format!("{w:08x}")).collect()
    }
}

fn sha256_file(path: &Path) -> (String, u64) {
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut s = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = f.read(&mut buf).unwrap();
        if k == 0 {
            break;
        }
        s.update(&buf[..k]);
        n += k as u64;
    }
    (s.hex(), n)
}

#[test]
fn sha256_matches_the_standard_vectors() {
    let hex = |b: &[u8]| {
        let mut s = Sha256::new();
        s.update(b);
        s.hex()
    };
    assert_eq!(hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert_eq!(hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    let two = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    assert_eq!(hex(two), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // a split update across the 64-byte boundary equals the whole
    let big: Vec<u8> = (0..1000u32).map(|i| (i * 7 + 3) as u8).collect();
    let mut s = Sha256::new();
    s.update(&big[..13]);
    s.update(&big[13..200]);
    s.update(&big[200..]);
    assert_eq!(s.hex(), hex(&big));
}

// ------------------------------------------------------------------------------------------------------------------ files

fn read_f32(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(b.len() % 4, 0, "{}", path.display());
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn read_u16(path: &Path) -> Vec<u16> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(b.len() % 2, 0, "{}", path.display());
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

#[derive(Clone, Copy, Debug, Default)]
struct Metric {
    rel_max: f64,
    rel_l2: f64,
}

/// `a` against the reference `b`: `rel_max = max|a - b| / max|b|`, `rel_l2 = ||a - b|| / ||b||` (fp64 sums).
fn metric(a: &[f32], b: &[f32]) -> Metric {
    assert_eq!(a.len(), b.len(), "metric: lengths differ");
    assert!(a.iter().all(|x| x.is_finite()), "candidate has non-finite values");
    let (mut dmax, mut bmax, mut d2, mut b2) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dmax = dmax.max((x - y).abs());
        bmax = bmax.max(y.abs());
        d2 += (x - y) * (x - y);
        b2 += y * y;
    }
    Metric { rel_max: dmax / bmax.max(1e-300), rel_l2: d2.sqrt() / b2.sqrt().max(1e-300) }
}

/// fp32 rounded to bf16 (round to nearest even) and widened: the features the splice sees.
fn bf16_round(v: f32) -> f32 {
    let b = v.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
    f32::from_bits(r & 0xffff_0000)
}

// ------------------------------------------------------------------------------------------------------------------ manifest

#[derive(Default)]
struct Manifest {
    /// `(name, height, width)` in file order.
    images: Vec<(String, usize, usize)>,
    /// `(name, kind)` -> `(bytes, sha256)`.
    files: BTreeMap<(String, String), (u64, String)>,
    /// `(name, stage)` -> value.
    floor: BTreeMap<(String, String), f64>,
    engine: BTreeMap<(String, String), f64>,
}

fn read_manifest() -> Manifest {
    let path = env_path(&["REXL3_NEMO_VISION_FP32_MANIFEST"]).unwrap_or_else(|| repo_root().join(MANIFEST));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut m = Manifest::default();
    for (ln, line) in text.lines().enumerate() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.is_empty() || f[0].starts_with('#') {
            continue;
        }
        let bad = || panic!("{}:{}: cannot read {line:?}", path.display(), ln + 1);
        match f[0] {
            "image" if f.len() == 4 => m.images.push((f[1].to_string(), f[2].parse().unwrap(), f[3].parse().unwrap())),
            "file" if f.len() == 5 => {
                m.files.insert((f[1].to_string(), f[2].to_string()), (f[3].parse().unwrap(), f[4].to_string()));
            }
            "floor" if f.len() == 4 => {
                m.floor.insert((f[1].to_string(), f[2].to_string()), f[3].parse().unwrap());
            }
            "engine" if f.len() == 4 => {
                m.engine.insert((f[1].to_string(), f[2].to_string()), f[3].parse().unwrap());
            }
            "manifest" | "reference" | "torch" | "transformers" | "gpu" | "seed" => {}
            _ => bad(),
        }
    }
    assert!(!m.images.is_empty(), "{}: no images", path.display());
    m
}

// ------------------------------------------------------------------------------------------------------------------ the test

/// One image's numbers against fp32.
#[derive(Default, Clone)]
struct Row {
    tokens: usize,
    tower: Metric,
    final_ln: Metric,
    ps_rows: Metric,
    mlp1_raw: Metric,
    mlp1_served: Metric,
    blocks_worst: f64,
    blocks_worst_at: usize,
}

fn env_path(names: &[&str]) -> Option<PathBuf> {
    names.iter().find_map(std::env::var_os).map(PathBuf::from)
}

fn run_image(w: &nv::Weights, ws: &mut nv::Workspace, st: &Stream, dir: &Path, name: &str, h: usize, wd: usize) -> Row {
    let (gh, gw) = nv::patch_grid(h, wd).unwrap_or_else(|| panic!("{name}: {h}x{wd} is not a runnable size"));
    let n = gh * gw;
    let np = n / 4;
    // the processor's pixels (bf16 bits) widened to fp16 as the encoder does (round to nearest even)
    let px16: Vec<u16> = read_u16(&dir.join(format!("{name}.pixels.bf16")))
        .iter()
        .map(|&b| f64_to_f16(f32::from_bits((b as u32) << 16) as f64))
        .collect();
    assert_eq!(px16.len(), 3 * h * wd, "{name}: pixels");
    let pix = DevBuf::from_slice(&px16).unwrap();
    let mut tr = nv::Trace::default();
    nv::tower_forward(w, ws, &pix, h, wd, Some(&mut tr), st).unwrap();
    st.sync().unwrap();

    let mut feat = vec![0f32; n * nv::HIDDEN];
    ws.h().download_range(nv::FEATURE_OFFSET, &mut feat).unwrap();
    let mut lnf16 = vec![0u16; n * nv::HIDDEN];
    ws.final_ln().download_range(0, &mut lnf16).unwrap();
    let lnf: Vec<f32> = lnf16.iter().map(|&b| f16_to_f64(b) as f32).collect();
    let mut mlp1 = vec![0f32; np * nv::LLM_HIDDEN];
    ws.mlp1().download_range(0, &mut mlp1).unwrap();

    let ref_tower = read_f32(&dir.join(format!("{name}.tower.f32")));
    let ref_lnf = read_f32(&dir.join(format!("{name}.final_ln.f32")));
    let ref_mlp1 = read_f32(&dir.join(format!("{name}.mlp1.f32")));
    let ref_blocks = read_f32(&dir.join(format!("{name}.blocks.f32")));
    assert_eq!(ref_tower.len(), n * nv::HIDDEN, "{name}: reference tower features");
    assert_eq!(ref_lnf.len(), n * nv::HIDDEN, "{name}: reference final layernorm");
    assert_eq!(ref_mlp1.len(), np * nv::LLM_HIDDEN, "{name}: reference mlp1");
    assert_eq!(ref_blocks.len(), (nv::BLOCKS + 1) * 3 * nv::HIDDEN, "{name}: reference blocks");

    let mut row = Row { tokens: np, ..Default::default() };
    row.tower = metric(&feat, &ref_tower);
    row.final_ln = metric(&lnf, &ref_lnf);
    // the pixel shuffle (v2) of the reference's final layernorm at its first and last token: token (I, J) holds the patches
    // (2I + a, 2J + b) for (a, b) in (0,0), (0,1), (1,0), (1,1), each 1,280 wide
    let ps_token = |i: usize, j: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(4 * nv::HIDDEN);
        for (a, b) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let p = (2 * i + a) * gw + 2 * j + b;
            v.extend_from_slice(&ref_lnf[p * nv::HIDDEN..(p + 1) * nv::HIDDEN]);
        }
        v
    };
    let mut ps_ref = ps_token(0, 0);
    ps_ref.extend(ps_token(gh / 2 - 1, gw / 2 - 1));
    assert_eq!(tr.pixel_shuffle.len(), ps_ref.len(), "{name}: pixel shuffle rows");
    row.ps_rows = metric(&tr.pixel_shuffle, &ps_ref);
    row.mlp1_raw = metric(&mlp1, &ref_mlp1);
    let served: Vec<f32> = mlp1.iter().map(|&v| bf16_round(v)).collect();
    row.mlp1_served = metric(&served, &ref_mlp1);
    assert_eq!(tr.blocks.len(), nv::BLOCKS + 1, "{name}: trace");
    for (k, got) in tr.blocks.iter().enumerate() {
        let want = &ref_blocks[k * 3 * nv::HIDDEN..(k + 1) * 3 * nv::HIDDEN];
        let m = metric(got, want);
        if m.rel_l2 > row.blocks_worst {
            row.blocks_worst = m.rel_l2;
            row.blocks_worst_at = k;
        }
    }
    row
}

#[test]
fn the_engine_tower_meets_the_fp32_gates() {
    if !device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    let (Some(pack), Some(dir)) =
        (env_path(&["REXL3_NEMO_OMNI_DIR", "REXL3_NEMOTRON_PACK"]), env_path(&["REXL3_NEMO_VISION_FP32_DIR"]))
    else {
        eprintln!(
            "not run: REXL3_NEMO_OMNI_DIR or REXL3_NEMO_VISION_FP32_DIR is not set (the pack; the node-local fp32 references of \
             tools/nemotron_omni/vision_sdpa_check.py refs)"
        );
        return;
    };
    let record = std::env::var("REXL3_NEMO_VISION_FP32_RECORD").is_ok_and(|v| v == "1");
    let sha_off = std::env::var("REXL3_NEMO_VISION_FP32_SHA").is_ok_and(|v| v == "off");
    let m = read_manifest();

    // the references are the ones the gates were signed against
    let t0 = std::time::Instant::now();
    let mut sha_bad = Vec::new();
    for (name, _, _) in &m.images {
        for ((n, kind), (bytes, want)) in m.files.range((name.clone(), String::new())..) {
            if n != name {
                break;
            }
            let path = dir.join(format!("{name}.{kind}"));
            let (got, size) = sha256_file(&path);
            if got != *want || size != *bytes {
                sha_bad.push(format!("{name}.{kind}: {size} B sha256 {got}, the manifest has {bytes} B {want}"));
            }
        }
    }
    println!("reference files: {} of {} images checked in {:.1} s", m.files.len(), m.images.len(), t0.elapsed().as_secs_f64());
    for s in &sha_bad {
        println!("SHA256 MISMATCH {s}");
    }

    let max_patches = m
        .images
        .iter()
        .map(|(n, h, w)| {
            let (gh, gw) = nv::patch_grid(*h, *w).unwrap_or_else(|| panic!("{n}: {h}x{w} is not a runnable size"));
            gh * gw
        })
        .max()
        .unwrap();
    let st = Stream::new().unwrap();
    let w = nv::Weights::load(&pack).unwrap();
    let mut ws = nv::Workspace::new(max_patches).unwrap();

    let mut rows = Vec::new();
    let mut violations = Vec::new();
    for (name, h, wd) in &m.images {
        let r = run_image(&w, &mut ws, &st, &dir, name, *h, *wd);
        let checks: [(&str, f64, f64); 7] = [
            ("tower features rel_l2", r.tower.rel_l2, GATE_TOWER),
            ("final_ln rel_l2", r.final_ln.rel_l2, GATE_FINAL_LN),
            ("pixel shuffle rows rel_l2", r.ps_rows.rel_l2, GATE_FINAL_LN),
            ("mlp1 raw rel_l2", r.mlp1_raw.rel_l2, GATE_MLP1_RAW),
            ("mlp1 served rel_l2", r.mlp1_served.rel_l2, GATE_MLP1_SERVED_L2),
            ("mlp1 served rel_max", r.mlp1_served.rel_max, GATE_MLP1_SERVED_MAX),
            ("worst block stage rel_l2", r.blocks_worst, GATE_BLOCKS),
        ];
        for (what, got, gate) in checks {
            // NaN fails the gate too: only a `Less` verdict passes.
            if !matches!(got.partial_cmp(&gate), Some(std::cmp::Ordering::Less)) {
                violations.push(format!("{name}: {what} {got:.3e} is {:.2}x the gate {gate:.0e}", got / gate));
            }
        }
        rows.push((name.clone(), r));
    }

    println!("\nengine tower against fp32 (rel_l2 unless said; gates: tower {GATE_TOWER:.0e}, final_ln and pixel shuffle {GATE_FINAL_LN:.0e}, mlp1 raw {GATE_MLP1_RAW:.0e}, mlp1 served {GATE_MLP1_SERVED_L2:.0e} and rel_max {GATE_MLP1_SERVED_MAX:.0e}, blocks {GATE_BLOCKS:.0e})");
    println!(
        "{:<20} {:>6} | {:>9} {:>9} {:>9} {:>9} | {:>9} {:>9} | {:>9} {:>5} | {:>9} {:>9}",
        "image",
        "tokens",
        "tower",
        "final_ln",
        "ps rows",
        "mlp1 raw",
        "served",
        "rel_max",
        "blocks",
        "at",
        "bf16 twr",
        "bf16 mlp1"
    );
    for (name, r) in &rows {
        let fl = |s: &str| m.floor.get(&(name.clone(), s.to_string())).map_or("-".to_string(), |v| format!("{v:.2e}"));
        println!(
            "{:<20} {:>6} | {:>9.2e} {:>9.2e} {:>9.2e} {:>9.2e} | {:>9.2e} {:>9.2e} | {:>9.2e} {:>5} | {:>9} {:>9}",
            name,
            r.tokens,
            r.tower.rel_l2,
            r.final_ln.rel_l2,
            r.ps_rows.rel_l2,
            r.mlp1_raw.rel_l2,
            r.mlp1_served.rel_l2,
            r.mlp1_served.rel_max,
            r.blocks_worst,
            r.blocks_worst_at,
            fl("tower"),
            fl("mlp1"),
        );
    }
    let worst = |f: &dyn Fn(&Row) -> f64| rows.iter().map(|(_, r)| f(r)).fold(0f64, f64::max);
    println!(
        "{:<20} {:>6} | {:>9.2e} {:>9.2e} {:>9.2e} {:>9.2e} | {:>9.2e} {:>9.2e} | {:>9.2e}",
        "worst of all",
        "",
        worst(&|r| r.tower.rel_l2),
        worst(&|r| r.final_ln.rel_l2),
        worst(&|r| r.ps_rows.rel_l2),
        worst(&|r| r.mlp1_raw.rel_l2),
        worst(&|r| r.mlp1_served.rel_l2),
        worst(&|r| r.mlp1_served.rel_max),
        worst(&|r| r.blocks_worst),
    );
    println!("(the bf16 columns: the bf16 eager model's own distance to fp32 on that image, from the manifest)");
    // the engine's numbers at the manifest's tower beside the present ones: a drift, printed, not gated
    if !m.engine.is_empty() {
        println!("\nengine numbers now / when the manifest was made:");
        for (name, r) in &rows {
            let g = |s: &str| m.engine.get(&(name.clone(), s.to_string())).copied();
            let rat = |now: f64, s: &str| g(s).map_or("-".to_string(), |v| format!("{:.2}x", now / v));
            println!(
                "  {:<20} tower {:>6} final_ln {:>6} mlp1 raw {:>6} served {:>6} blocks {:>6}",
                name,
                rat(r.tower.rel_l2, "tower"),
                rat(r.final_ln.rel_l2, "final_ln"),
                rat(r.mlp1_raw.rel_l2, "mlp1_raw"),
                rat(r.mlp1_served.rel_l2, "mlp1_served"),
                rat(r.blocks_worst, "blocks"),
            );
        }
    }

    if record {
        let path = dir.join("engine_metrics.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        for (name, r) in &rows {
            for (s, v) in [
                ("tower", r.tower.rel_l2),
                ("final_ln", r.final_ln.rel_l2),
                ("mlp1_raw", r.mlp1_raw.rel_l2),
                ("mlp1_served", r.mlp1_served.rel_l2),
                ("mlp1_served_rel_max", r.mlp1_served.rel_max),
                ("blocks", r.blocks_worst),
            ] {
                writeln!(f, "engine {name} {s} {v:.4e}").unwrap();
            }
        }
        println!("recorded: {}", path.display());
    }

    let sha_fail = !sha_bad.is_empty() && !sha_off;
    if violations.is_empty() && !sha_fail {
        println!(
            "\nVISION FP32 GATES: PASS ({} images, {} references checked{})",
            rows.len(),
            m.files.len(),
            if sha_bad.is_empty() { "" } else { ", SHA-256 drift reported" }
        );
    } else {
        println!("\nVISION FP32 GATES: FAIL");
        for v in &violations {
            println!("  GATE {v}");
        }
        if sha_fail {
            println!(
                "  {} reference file(s) are not the ones the manifest names (REXL3_NEMO_VISION_FP32_SHA=off to go on)",
                sha_bad.len()
            );
        }
    }
    assert!(violations.is_empty(), "fp32 gates broken: {violations:#?}");
    assert!(!sha_fail, "reference files differ from the manifest: {sha_bad:#?}");
}

/// The repository root: `REXL3_REPO_ROOT` when set (a prebuilt test binary run from an installed bin directory that carries the
/// tree's `testdata/`, e.g. on a node without this checkout), else the checkout this binary was built from.
fn repo_root() -> std::path::PathBuf {
    std::env::var_os("REXL3_REPO_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}
