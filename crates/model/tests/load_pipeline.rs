//! The pipelined weight loader against the serial reference loader.
//!
//! A synthetic pack (EXL3 groups of several widths with fp16 scales holding NaN payloads and signed zeros, packed-sign
//! scales, fp16 and bf16 biases, plain 16-bit tensors small and larger than a staging buffer, tensors spread over two
//! shards) is loaded once serially and once per pipeline configuration; every device buffer must be byte-identical.
//! The configurations include tiny staging spans, one slot and many workers, so that a reused staging buffer, a span
//! boundary inside a tensor or a wrong offset shows up as a differing byte.
//!
//! The same comparison on the real packs (target, draft, vision tower) runs with `REXL3_QWEN_DIR` / `REXL3_DRAFT_DIR` set.

use gb_cuda::loader::{DirectMode, Options};
use gb_cuda::DevBuf;
use gb_exl3::f16::f64_to_f16;
use gb_exl3::{Bits, MARKER_MUL1};
use gb_model::loadctx::{LoadCtx, LoadMode};
use gb_model::pack::Pack;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn gpu() -> bool {
    if gb_cuda::device_available() {
        true
    } else {
        eprintln!("no CUDA device: skipping");
        false
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
    fn words(&mut self, n: usize) -> Vec<u16> {
        (0..n).map(|_| self.next() as u16).collect()
    }
}

fn tmp(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("load_pipeline").join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct T {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn t(name: &str, dtype: &'static str, shape: &[usize], data: Vec<u8>) -> T {
    T { name: name.into(), dtype, shape: shape.to_vec(), data }
}

fn le16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn safetensors(tensors: &[T]) -> Vec<u8> {
    let mut hdr = serde_json::Map::new();
    let mut off = 0u64;
    for x in tensors {
        let end = off + x.data.len() as u64;
        hdr.insert(x.name.clone(), json!({"dtype": x.dtype, "shape": x.shape, "data_offsets": [off, end]}));
        off = end;
    }
    let h = serde_json::to_vec(&Value::Object(hdr)).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    for x in tensors {
        out.extend_from_slice(&x.data);
    }
    out
}

/// What each scale vector of a group holds: random fp16, fp16 with the awkward patterns, or packed signs.
enum Scales {
    Random,
    Awkward,
    Signs,
}

/// One EXL3 group's tensors (mul1 codebook) for K x N at `twice` half-bits.
fn group(rng: &mut Rng, prefix: &str, k: usize, n: usize, twice: u8, scales: Scales, bias: Option<&'static str>) -> Vec<T> {
    let bits = Bits::from_twice(twice).unwrap();
    let (kt, nt) = (k / 16, n / 16);
    let mut v =
        vec![t(&format!("{prefix}.trellis"), "I16", &[kt, nt, bits.tile_words()], rng.bytes(kt * nt * bits.tile_words() * 2))];
    let fp16 = |rng: &mut Rng, len: usize, awkward: bool| {
        let mut w: Vec<u16> = (0..len).map(|_| f64_to_f16(0.25 + (rng.next() % 4000) as f64 / 1000.0)).collect();
        if awkward {
            // NaN payloads of both signs, +-0, the smallest subnormal, +-inf, the largest finite value
            let odd = [0x7E01u16, 0xFFFF, 0x7C01, 0x8000, 0x0000, 0x0001, 0x7C00, 0xFC00, 0x7BFF, 0xFBFF, 0x8001, 0xFE00];
            for (i, &o) in odd.iter().enumerate() {
                w[(i * 7 + 3) % len] = o;
            }
        }
        w
    };
    match scales {
        Scales::Signs => {
            v.push(t(&format!("{prefix}.su"), "I16", &[kt], le16(&rng.words(kt))));
            v.push(t(&format!("{prefix}.sv"), "I16", &[nt], le16(&rng.words(nt))));
        }
        s => {
            let awkward = matches!(s, Scales::Awkward);
            v.push(t(&format!("{prefix}.suh"), "F16", &[k], le16(&fp16(rng, k, awkward))));
            v.push(t(&format!("{prefix}.svh"), "F16", &[n], le16(&fp16(rng, n, awkward))));
        }
    }
    match bias {
        Some("F16") => v.push(t(&format!("{prefix}.bias"), "F16", &[n], le16(&fp16(rng, n, false)))),
        // a bf16 bias: the serial loader rounds it to fp16
        Some(_) => {
            let bf: Vec<u16> = (0..n).map(|_| (0x3E00 + (rng.next() % 0x300)) as u16).collect();
            v.push(t(&format!("{prefix}.bias"), "BF16", &[n], le16(&bf)));
        }
        None => {}
    }
    v.push(t(&format!("{prefix}.mul1"), "I32", &[], MARKER_MUL1.to_le_bytes().to_vec()));
    v
}

/// Two shards: groups of several shapes and plain tensors, in an order that is not the declaration order.
fn build_pack(dir: &Path) {
    let mut rng = Rng(0xC0FFEE);
    let mut a: Vec<T> = Vec::new();
    let mut b: Vec<T> = Vec::new();
    a.extend(group(&mut rng, "m.big", 1024, 1024, 8, Scales::Random, None)); // 512 KiB trellis: several spans when they are small
    a.push(t("p.bf16big", "BF16", &[400, 400], le16(&rng.words(160_000)))); // 320 KB
    a.extend(group(&mut rng, "m.awk", 128, 256, 4, Scales::Awkward, Some("F16")));
    a.push(t("p.f16", "F16", &[3, 1000], le16(&rng.words(3000))));
    b.extend(group(&mut rng, "m.signs", 128, 128, 10, Scales::Signs, None));
    b.extend(group(&mut rng, "m.bf", 256, 128, 6, Scales::Random, Some("BF16")));
    b.push(t("p.norm", "BF16", &[5120], le16(&rng.words(5120))));
    a.extend(group(&mut rng, "m.six", 384, 512, 12, Scales::Awkward, None));
    b.extend(group(&mut rng, "m.eight", 640, 128, 16, Scales::Random, None));
    a.push(t("p.tiny", "BF16", &[3], le16(&[0x3F80, 0x4000, 0xC040])));
    std::fs::write(dir.join("a.safetensors"), safetensors(&a)).unwrap();
    std::fs::write(dir.join("b.safetensors"), safetensors(&b)).unwrap();
}

const GROUPS: [(&str, usize, usize); 6] = [
    ("m.big", 1024, 1024),
    ("m.awk", 128, 256),
    ("m.signs", 128, 128),
    ("m.bf", 256, 128),
    ("m.six", 384, 512),
    ("m.eight", 640, 128),
];

fn download(b: &DevBuf) -> Vec<u8> {
    b.to_vec::<u8>().unwrap()
}

fn options(workers: usize, slots: usize, span: usize, direct: DirectMode) -> Options {
    Options { workers, slots, span_bytes: span, gap_bytes: 4096, direct }
}

/// Load the group buffers and plain buffers both ways and compare.
fn compare(serial: &mut Vec<(String, Vec<u8>)>, o: Options, tag: &str) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("load_pipeline").join("pack");
    let pack = Pack::open(&dir).unwrap();
    let mut ctx = LoadCtx::with_options(LoadMode::Pipeline, o.clone());
    let mut groups = Vec::new();
    for &(p, _, _) in GROUPS.iter().rev() {
        groups.push((p, ctx.group(&pack, p).unwrap()));
    }
    let f16 = ctx.raw16(&pack, "p.f16", "F16", &[3, 1000]).unwrap();
    let bf16big = ctx.raw16(&pack, "p.bf16big", "BF16", &[400, 400]).unwrap();
    let tiny = ctx.raw16(&pack, "p.tiny", "BF16", &[3]).unwrap();
    let norm = ctx.f32_from_bf16(&pack, "p.norm", &[5120], true).unwrap();
    let bigf32 = ctx.f32_from_bf16(&pack, "p.bf16big", &[400, 400], false).unwrap();
    let computed =
        ctx.computed(100_000, || Ok((0..100_000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect())).unwrap();
    let report = ctx.finish().unwrap().expect("pipeline mode returns a report");
    assert!(report.bytes > 0 && report.spans > 0, "{tag}: nothing was loaded");
    match o.direct {
        // a configuration that asks for direct reads must have made some, or the aligned-read path is untested here
        DirectMode::Always if std::env::var("REXL3_ALLOW_NO_DIRECT").is_err() => {
            assert!(
                report.direct_spans > 0,
                "{tag}: no span was read with O_DIRECT (does this filesystem refuse it? set REXL3_ALLOW_NO_DIRECT=1 to accept that)"
            )
        }
        DirectMode::Never => assert_eq!(report.direct_spans, 0, "{tag}: a direct read was made with DirectMode::Never"),
        _ => {}
    }
    let mut got: Vec<(String, Vec<u8>)> = Vec::new();
    for (p, w) in &groups {
        for (part, b) in w.buffers() {
            got.push((format!("{p}.{part}"), download(b)));
        }
    }
    for (n, b) in [
        ("p.f16", &f16),
        ("p.bf16big", &bf16big),
        ("p.tiny", &tiny),
        ("p.norm.f32+1", &norm),
        ("p.bf16big.f32", &bigf32),
        ("computed", &computed),
    ] {
        got.push((n.into(), download(b)));
    }
    if serial.is_empty() {
        // the reference: the serial loader's buffers
        let mut sctx = LoadCtx::with_options(LoadMode::Serial, o);
        let mut sg = Vec::new();
        for &(p, _, _) in GROUPS.iter().rev() {
            sg.push((p, sctx.group(&pack, p).unwrap()));
        }
        let sf16 = sctx.raw16(&pack, "p.f16", "F16", &[3, 1000]).unwrap();
        let sbig = sctx.raw16(&pack, "p.bf16big", "BF16", &[400, 400]).unwrap();
        let stiny = sctx.raw16(&pack, "p.tiny", "BF16", &[3]).unwrap();
        let snorm = sctx.f32_from_bf16(&pack, "p.norm", &[5120], true).unwrap();
        let sbigf32 = sctx.f32_from_bf16(&pack, "p.bf16big", &[400, 400], false).unwrap();
        let scomputed =
            sctx.computed(100_000, || Ok((0..100_000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect())).unwrap();
        assert!(sctx.finish().unwrap().is_none());
        for (p, w) in &sg {
            for (part, b) in w.buffers() {
                serial.push((format!("{p}.{part}"), download(b)));
            }
        }
        for (n, b) in [
            ("p.f16", &sf16),
            ("p.bf16big", &sbig),
            ("p.tiny", &stiny),
            ("p.norm.f32+1", &snorm),
            ("p.bf16big.f32", &sbigf32),
            ("computed", &scomputed),
        ] {
            serial.push((n.into(), download(b)));
        }
    }
    assert_eq!(got.len(), serial.len(), "{tag}: different buffer sets");
    for ((gn, gb), (sn, sb)) in got.iter().zip(serial.iter()) {
        assert_eq!(gn, sn);
        assert_eq!(gb.len(), sb.len(), "{tag}: {gn} has {} bytes, the serial loader's has {}", gb.len(), sb.len());
        let first = gb.iter().zip(sb).position(|(a, b)| a != b);
        assert!(first.is_none(), "{tag}: {gn} differs from the serial loader's at byte {:?} of {}", first, gb.len());
    }
}

#[test]
fn pipeline_matches_serial_for_every_configuration() {
    if !gpu() {
        return;
    }
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("load_pipeline").join("pack");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    build_pack(&dir);
    let mut serial = Vec::new();
    // (workers, slots, span bytes, direct): tiny spans cut tensors mid-way, one slot forces constant reuse, many workers
    // contend for the stream and the file, direct reads exercise the aligned-read path (or its fallback)
    let configs = [
        (1, 1, 4096, DirectMode::Never),
        (3, 1, 8192, DirectMode::Never),
        (8, 2, 5000, DirectMode::Never),
        (8, 1, 65536, DirectMode::Always),
        (4, 2, 100_000, DirectMode::Always),
        (6, 3, 30_000, DirectMode::Auto),
        (2, 2, 8 << 20, DirectMode::Auto),
        (16, 2, 1 << 20, DirectMode::Never),
    ];
    for (i, &(w, s, span, d)) in configs.iter().enumerate() {
        compare(&mut serial, options(w, s, span, d), &format!("config {i} (workers {w}, slots {s}, span {span}, {d:?})"));
    }
}

#[test]
fn awkward_fp16_scales_come_out_as_the_serial_loader_makes_them() {
    if !gpu() {
        return;
    }
    let dir = tmp("awk");
    let mut rng = Rng(7);
    let ts = group(&mut rng, "m.awk", 128, 128, 4, Scales::Awkward, None);
    std::fs::write(dir.join("a.safetensors"), safetensors(&ts)).unwrap();
    let pack = Pack::open(&dir).unwrap();
    let mut ctx = LoadCtx::with_options(LoadMode::Pipeline, options(2, 2, 4096, DirectMode::Never));
    let w = ctx.group(&pack, "m.awk").unwrap();
    ctx.finish().unwrap();
    let suh: Vec<u16> = w.buffers().into_iter().find(|(n, _)| *n == "suh").unwrap().1.to_vec().unwrap();
    // NaNs of either sign and payload become 0x7E00; everything else is itself (+-0, subnormals and infinities included)
    let raw: Vec<u16> = ts
        .iter()
        .find(|x| x.name == "m.awk.suh")
        .unwrap()
        .data
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let mut nans = 0;
    for (g, r) in suh.iter().zip(&raw) {
        let is_nan = (r & 0x7C00) == 0x7C00 && (r & 0x03FF) != 0;
        nans += is_nan as usize;
        assert_eq!(*g, if is_nan { 0x7E00 } else { *r }, "raw {r:#06x}");
    }
    assert!(nans >= 4, "the pack holds NaN patterns");
}

#[test]
fn a_shard_cut_short_after_the_scan_is_an_error_and_not_zeros() {
    if !gpu() {
        return;
    }
    let dir = tmp("cut");
    let mut rng = Rng(9);
    let ts = group(&mut rng, "m.cut", 256, 256, 8, Scales::Random, None);
    let file = dir.join("a.safetensors");
    std::fs::write(&file, safetensors(&ts)).unwrap();
    let pack = Pack::open(&dir).unwrap();
    let len = std::fs::metadata(&file).unwrap().len();
    std::fs::OpenOptions::new().write(true).open(&file).unwrap().set_len(len - 5000).unwrap();
    let mut ctx = LoadCtx::with_options(LoadMode::Pipeline, options(2, 2, 4096, DirectMode::Never));
    let _w = ctx.group(&pack, "m.cut").unwrap();
    let err = ctx.finish().expect_err("a truncated shard must fail the load");
    let msg = err.to_string();
    assert!(msg.contains("a.safetensors") || msg.contains("bytes"), "{msg}");
}

/// Every buffer of two loads must be the same bytes: names, lengths and contents.
fn assert_same_buffers(what: &str, serial: &[(String, &DevBuf)], pipeline: &[(String, &DevBuf)]) {
    assert_eq!(serial.len(), pipeline.len(), "{what}: different buffer counts");
    let mut bytes = 0usize;
    for ((sn, sb), (pn, pb)) in serial.iter().zip(pipeline) {
        assert_eq!(sn, pn, "{what}: buffer lists differ");
        assert_eq!(sb.bytes(), pb.bytes(), "{what}: {sn} has a different length");
        let (a, b) = (download(sb), download(pb));
        let first = a.iter().zip(&b).position(|(x, y)| x != y);
        assert!(
            first.is_none(),
            "{what}: {sn} differs between the serial and the pipelined load at byte {first:?} of {}",
            a.len()
        );
        bytes += a.len();
    }
    eprintln!("{what}: {} buffers, {:.2} GiB, byte-identical", serial.len(), bytes as f64 / (1u64 << 30) as f64);
}

/// The proof behind "device contents are byte-identical": the real packs, every weight buffer of the target, the draft and the
/// vision tower, loaded by the serial reference loader and by the pipeline, compared byte for byte. About 32 GiB of device
/// memory while the target is compared, then less. Needs `REXL3_QWEN_DIR` and `REXL3_DRAFT_DIR`; ignored by default (the GPU
/// gate runs it).
#[test]
#[ignore]
fn real_packs_load_byte_identically_both_ways() {
    use gb_model::config::QwenConfig;
    use gb_model::draft::Draft;
    use gb_model::vision::VisionTower;
    use gb_model::weights::Weights;
    let dir = |v: &str| std::env::var(v).ok().map(PathBuf::from).filter(|p| p.join("config.json").exists());
    let (Some(t), Some(d)) = (dir("REXL3_QWEN_DIR"), dir("REXL3_DRAFT_DIR")) else {
        return eprintln!("REXL3_QWEN_DIR / REXL3_DRAFT_DIR not set: skipping");
    };
    if !gpu() {
        return;
    }
    let cfg = QwenConfig::from_dir(&t).unwrap();
    let (tp, dp) = (Pack::open(&t).unwrap(), Pack::open(&d).unwrap());
    let opts = Options::from_env();
    let pair =
        || (LoadCtx::with_options(LoadMode::Serial, opts.clone()), LoadCtx::with_options(LoadMode::Pipeline, opts.clone()));
    {
        let (mut sc, mut pc) = pair();
        let s = Weights::declare(&mut sc, &tp, &cfg, |_, _| {}).unwrap();
        sc.finish().unwrap();
        let p = Weights::declare(&mut pc, &tp, &cfg, |_, _| {}).unwrap();
        let report = pc.finish().unwrap().expect("a pipeline report");
        assert!(report.bytes > 14 << 30, "the pipeline moved only {} bytes", report.bytes);
        assert_same_buffers("target", &s.buffers(), &p.buffers());
    }
    {
        let (mut sc, mut pc) = pair();
        let (s, _) = Draft::declare(&mut sc, &dp, cfg.vocab, 2, 256, gb_model::draft::BLOCK).unwrap();
        sc.finish().unwrap();
        let (p, _) = Draft::declare(&mut pc, &dp, cfg.vocab, 2, 256, gb_model::draft::BLOCK).unwrap();
        pc.finish().unwrap();
        assert_same_buffers("draft", &s.weight_buffers(), &p.weight_buffers());
    }
    {
        let (mut sc, mut pc) = pair();
        let s = VisionTower::declare(&mut sc, &tp, 256).unwrap();
        sc.finish().unwrap();
        let p = VisionTower::declare(&mut pc, &tp, 256).unwrap();
        pc.finish().unwrap();
        assert_same_buffers("vision tower", &s.weight_buffers(), &p.weight_buffers());
    }
}
