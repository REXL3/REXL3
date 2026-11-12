//! The engine's logits for any set of token samples, for the KL comparison against the bf16 original (`tools/nemotron_omni/kl_compare.py`).
//!
//! The same path as `nemotron_omni_i2`'s logits output: the composed model, 64-row verify windows each followed by `commit(slot, 64)`,
//! every row's logits tapped (a prefill slice only computes the last row's), written as f32 little-endian `[tokens][131072]` in
//! `engine_logits_<k>.f32`, plus the argmax of every row in `engine_top1_<k>.i32`. No oracle dump is needed.
//!
//! Env (without `REXL3_NEMO_OMNI_DIR`, `REXL3_NEMO_IDS_DIR` and `REXL3_NEMO_KL_OUT` the test reports what it did not run and passes):
//! - `REXL3_NEMO_OMNI_DIR`: the pack (or a patched pack: an index naming bf16 `.weight` tensors in place of EXL3 parts);
//! - `REXL3_NEMO_IDS_DIR`: a directory of `ids_<k>.u32` (little-endian u32 token ids);
//! - `REXL3_NEMO_KL_OUT`: where the files go (pack-derived: they stay on the node);
//! - `REXL3_NEMO_KL_SAMPLES`: which `k` to run, e.g. `0-5,12` (default: every `ids_<k>.u32` in the directory);
//! - `REXL3_NEMO_KL_TOKENS`: tokens per sample, a multiple of 64 (default: the file's length);
//! - `REXL3_NEMO_KL_SKIP_DONE=1`: skip a sample whose logits file is already complete (a run that was cut short resumes);
//! - `REXL3_NEMO_KL_DIGEST_OUT=<dir>`: for the first sample run, a 64-bit FNV-1a digest of every window's buffer at the embedding, every
//!   trunk layer's output and the final norm (`digest_s<k>.jsonl`: window, point, digest), and the f32 rows of the first window at
//!   those points (`rows_s<k>_w0.f32`, in the order of `points_s<k>.json`). Two runs, on two packs, show up to which point the
//!   engine's activations are bit-identical (`tools/nemotron_omni/digest_compare.py`);
//! - `REXL3_NEMO_KL_DECODE_STEPS=n`: after the first sample, time `n` one-row steps (the sample's own tokens, teacher forced, each
//!   followed by `commit` and `flush`) and print their median, mean and best in ms: the speed of the pack on text that routes to
//!   the experts it really routes to (a pack with bf16 experts takes the original kernels for them).

use gb_cuda::{DevBuf, Stream};
use gb_engine::RowRequest;
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::ModelOptions;
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::model::{NemotronModel, Point, Tap};
use gb_model::families::nemotron_h_omni::weights::LayerPlan;
use gb_model::rows::RowBatch;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const VOCAB: usize = 131_072;

/// What the digest tap has seen: per window and point the FNV-1a digest, and the first window's rows.
#[derive(Default)]
struct Digests {
    window: usize,
    lines: Vec<(usize, String, u64)>,
    points: Vec<String>,
    rows: Vec<f32>,
}

fn point_name(p: Point) -> Option<String> {
    match p {
        Point::Embed => Some("embed".into()),
        Point::Layer(i) => Some(format!("layer{i}")),
        Point::FinalNorm => Some("final_norm".into()),
        Point::Mixer(_) | Point::Logits => None,
    }
}

fn fnv1a(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Writes the logits rows of each window to a file and keeps the argmax of every row; optionally digests the other points.
struct Logits {
    file: std::fs::File,
    top1: Vec<i32>,
    non_finite: usize,
    digests: Option<std::sync::Arc<std::sync::Mutex<Digests>>>,
}

impl Tap for Logits {
    fn tap(&mut self, point: Point, buf: &DevBuf, rows: usize, width: usize, st: &Stream) -> Result<(), ModelError> {
        if let (Some(d), Some(name)) = (&self.digests, point_name(point)) {
            st.sync()?;
            let mut v = vec![0f32; rows * width];
            buf.download_range(0, &mut v)?;
            let h = fnv1a(v.iter().flat_map(|x| x.to_le_bytes()));
            let mut g = d.lock().unwrap();
            let w = g.window;
            g.lines.push((w, name.clone(), h));
            if w == 0 {
                g.points.push(name);
                g.rows.extend_from_slice(&v);
            }
            return Ok(());
        }
        if point != Point::Logits {
            return Ok(());
        }
        st.sync()?;
        let mut v = vec![0f32; rows * width];
        buf.download_range(0, &mut v)?;
        for r in v.chunks(width) {
            let (mut best, mut at) = (f32::NEG_INFINITY, 0usize);
            for (i, &x) in r.iter().enumerate() {
                if !x.is_finite() {
                    self.non_finite += 1;
                } else if x > best {
                    best = x;
                    at = i;
                }
            }
            self.top1.push(at as i32);
        }
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.file.write_all(&bytes).map_err(|e| ModelError::Request(format!("logits file: {e}")))?;
        Ok(())
    }
}

fn env_dir(name: &str) -> Option<PathBuf> {
    let v = std::env::var_os(name).map(PathBuf::from);
    if v.is_none() {
        eprintln!("not run: {name} is not set");
    }
    v
}

/// `0-5,12` to `[0, 1, 2, 3, 4, 5, 12]`.
fn parse_samples(spec: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (usize, usize) =
                    (a.trim().parse().expect("sample range start"), b.trim().parse().expect("sample range end"));
                out.extend(a..=b);
            }
            None => out.push(part.parse().expect("sample index")),
        }
    }
    out
}

fn read_ids(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(bytes.len() % 4 == 0, "{}: {} bytes is not a whole number of u32", path.display(), bytes.len());
    bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

#[test]
fn parse_samples_reads_lists_and_ranges() {
    assert_eq!(parse_samples("0-3,7, 12"), vec![0, 1, 2, 3, 7, 12]);
    assert_eq!(parse_samples("5"), vec![5]);
    assert!(parse_samples("").is_empty());
}

#[test]
fn engine_logits_for_the_samples_of_a_directory() {
    let (Some(pack), Some(ids_dir), Some(out)) =
        (env_dir("REXL3_NEMO_OMNI_DIR"), env_dir("REXL3_NEMO_IDS_DIR"), env_dir("REXL3_NEMO_KL_OUT"))
    else {
        return;
    };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    std::fs::create_dir_all(&out).unwrap_or_else(|e| panic!("{}: {e}", out.display()));

    let samples = match std::env::var("REXL3_NEMO_KL_SAMPLES") {
        Ok(s) if !s.trim().is_empty() => parse_samples(&s),
        _ => {
            let mut v: Vec<usize> = std::fs::read_dir(&ids_dir)
                .unwrap_or_else(|e| panic!("{}: {e}", ids_dir.display()))
                .filter_map(|e| {
                    let name = e.ok()?.file_name().into_string().ok()?;
                    name.strip_prefix("ids_")?.strip_suffix(".u32")?.parse().ok()
                })
                .collect();
            v.sort_unstable();
            v
        }
    };
    assert!(!samples.is_empty(), "no samples in {}", ids_dir.display());
    let tokens_cap = std::env::var("REXL3_NEMO_KL_TOKENS").ok().and_then(|v| v.parse::<usize>().ok());
    let skip_done = std::env::var("REXL3_NEMO_KL_SKIP_DONE").is_ok_and(|v| v == "1");
    let decode_steps = std::env::var("REXL3_NEMO_KL_DECODE_STEPS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
    let mut decode_done = false;
    let digest_out = std::env::var_os("REXL3_NEMO_KL_DIGEST_OUT").map(PathBuf::from);
    let mut digest_done = false;

    let text = std::fs::read_to_string(pack.join("config.json")).unwrap_or_else(|e| panic!("config.json: {e}"));
    let cfg = OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap_or_else(|e| panic!("{e}"));
    let inv = Inventory::read(&pack).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let opts = ModelOptions { slots: 1, kv_pages: 8, max_context: 2048, prefill_rows: 64, layers: None, mtp: false };
    let t = Instant::now();
    let mut model = NemotronModel::load(&pack, &cfg, &plan, opts).unwrap_or_else(|e| panic!("load: {e}"));
    eprintln!("loaded in {:.1} s ({} bytes planned)", t.elapsed().as_secs_f64(), plan.bytes(&inv));

    for &k in &samples {
        let ids = read_ids(&ids_dir.join(format!("ids_{k}.u32")));
        let n = tokens_cap.unwrap_or(ids.len()).min(ids.len()) / 64 * 64;
        assert!(n > 0, "sample {k}: no whole 64-row windows");
        let logits_path = out.join(format!("engine_logits_{k}.f32"));
        let want = (n * VOCAB * 4) as u64;
        if skip_done && std::fs::metadata(&logits_path).map(|m| m.len() == want).unwrap_or(false) {
            eprintln!("sample {k}: {n} tokens already done");
            continue;
        }
        let tmp = out.join(format!("engine_logits_{k}.f32.tmp"));
        let file = std::fs::File::create(&tmp).unwrap_or_else(|e| panic!("{}: {e}", tmp.display()));
        model.begin_slot(0).unwrap_or_else(|e| panic!("begin_slot: {e}"));
        let digests = match (&digest_out, digest_done) {
            (Some(_), false) => Some(std::sync::Arc::new(std::sync::Mutex::new(Digests::default()))),
            _ => None,
        };
        model.set_tap(Some(Box::new(Logits { file, top1: Vec::new(), non_finite: 0, digests: digests.clone() })));
        let t = Instant::now();
        for (w, chunk) in ids[..n].chunks(64).enumerate() {
            if let Some(d) = &digests {
                d.lock().unwrap().window = w;
            }
            let b = RowBatch::from_requests(&[RowRequest { slot: 0, first_position: (w * 64) as u64, tokens: chunk.to_vec() }]);
            model.window_candidates(&b, 8).unwrap_or_else(|e| panic!("sample {k} window {w}: {e}"));
            model.commit(0, chunk.len());
            model.flush().unwrap_or_else(|e| panic!("sample {k} window {w}: {e}"));
        }
        model.set_tap(None);
        model.end_slot(0);
        eprintln!("sample {k}: {n} tokens in {:.1} s", t.elapsed().as_secs_f64());
        if let (Some(d), Some(dir)) = (&digests, &digest_out) {
            std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
            let g = d.lock().unwrap();
            let mut text = String::new();
            for (w, name, h) in &g.lines {
                text.push_str(&format!("{{\"window\": {w}, \"point\": \"{name}\", \"fnv1a\": \"{h:016x}\"}}\n"));
            }
            std::fs::write(dir.join(format!("digest_s{k}.jsonl")), text).unwrap();
            let rows: Vec<u8> = g.rows.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dir.join(format!("rows_s{k}_w0.f32")), rows).unwrap();
            let pts: Vec<String> = g.points.iter().map(|p| format!("\"{p}\"")).collect();
            std::fs::write(
                dir.join(format!("points_s{k}.json")),
                format!("{{\"rows\": 64, \"width\": 4096, \"points\": [{}]}}", pts.join(", ")),
            )
            .unwrap();
            eprintln!("digests of sample {k}: {} lines, {} points in the first window", g.lines.len(), g.points.len());
            digest_done = true;
        }
        if decode_steps > 0 && !decode_done {
            // single-row steps continuing the sample (its tokens again, positions n..): the model stays in its slot until the end
            model.begin_slot(0).unwrap_or_else(|e| panic!("begin_slot: {e}"));
            for (w, chunk) in ids[..n].chunks(64).enumerate() {
                let b =
                    RowBatch::from_requests(&[RowRequest { slot: 0, first_position: (w * 64) as u64, tokens: chunk.to_vec() }]);
                model.window_candidates(&b, 8).unwrap_or_else(|e| panic!("decode warm-up window {w}: {e}"));
                model.commit(0, chunk.len());
            }
            model.flush().unwrap();
            let mut ms = Vec::with_capacity(decode_steps);
            for i in 0..decode_steps {
                let tok = ids[(i * 7) % n];
                let b = RowBatch::from_requests(&[RowRequest { slot: 0, first_position: (n + i) as u64, tokens: vec![tok] }]);
                let t = Instant::now();
                model.window_candidates(&b, 1).unwrap_or_else(|e| panic!("decode step {i}: {e}"));
                model.commit(0, 1);
                model.flush().unwrap_or_else(|e| panic!("decode step {i}: {e}"));
                ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            model.end_slot(0);
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = ms.iter().sum::<f64>() / ms.len() as f64;
            eprintln!(
                "decode: {} one-row steps after {n} tokens of sample {k}: median {:.2} ms, mean {:.2} ms, best {:.2} ms, worst {:.2} ms",
                ms.len(),
                ms[ms.len() / 2],
                mean,
                ms[0],
                ms[ms.len() - 1]
            );
            decode_done = true;
        }
        let meta = std::fs::metadata(&tmp).unwrap();
        assert_eq!(meta.len(), want, "sample {k}: logits file is {} bytes, expected {want}", meta.len());
        std::fs::rename(&tmp, &logits_path).unwrap_or_else(|e| panic!("rename: {e}"));
    }
}
