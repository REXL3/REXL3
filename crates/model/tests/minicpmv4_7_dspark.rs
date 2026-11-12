//! The MiniCPM-V-4.7 DSpark draft's block forward, stand-alone (E4a of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`): stage-by-stage
//! against the torch oracle (`tools/minicpm_v47/dspark_ref.py dump`), row invariance, chunk invariance, determinism, rollback ==
//! replay, the visibility bound, the confidence chain and the Markov walk against host mirrors, plan == allocation, and the
//! timing lane (`--ignored`).
//!
//! Needs a device and, by environment (every test skips with `not run: ...` without them):
//!   * `REXL3_DSPARK_DIR`: the draft the engine runs: an EXL3 draft pack (`.../draft-packs/dspark-6.00bpw`) or the bf16 checkpoint
//!     (`.../dspark-hf`: the dense cuBLAS reference path); default the 6.0 bpw pack;
//!   * `REXL3_DSPARK_TARGET_DIR`: the bf16 target checkpoint (its embedding table and lm_head); default `.../hf`;
//!   * `REXL3_DSPARK_ORACLE`: a directory of oracle dumps (default `~/projects/minicpm-dspark/oracle`; made by
//!     `dspark_ref.py dump --out-dir`). Dumps whose manifest names another draft than `REXL3_DSPARK_DIR` are ignored.
//!
//! ```text
//! flock -s -w 1800 /tmp/rexl3-gpu.lock cargo test --release -p gb-model --test minicpmv4_7_dspark -- --nocapture --test-threads=1
//! flock -x -w 1800 /tmp/rexl3-gpu.lock cargo test --release -p gb-model --test minicpmv4_7_dspark -- --ignored --nocapture timing
//! ```
use gb_cuda::dspark::confidence_host;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_model::families::minicpmv4_7::config::{DSparkConfig, MiniCpmConfig};
use gb_model::families::minicpmv4_7::dspark::{
    compare_with_plan, CtxDump, DenseTarget, DraftModel, DraftOptions, Head, HeadKind, Proposal, ProposeReq, StageDump,
    WeightKind, BLOCK, HIDDEN, RANK,
};
use gb_model::families::minicpmv4_7::memory::allocations;
use gb_sampling::Sampling;
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

const VOCAB: usize = 248_144;
const TAPS: [usize; 5] = [3, 11, 19, 27, 35];

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME"))
}

fn dirs() -> Option<(PathBuf, PathBuf, PathBuf)> {
    if !device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    let m = home().join("models/minicpm-v-4.7-35b-a3b");
    let draft = std::env::var_os("REXL3_DSPARK_DIR").map(PathBuf::from).unwrap_or_else(|| m.join("draft-packs/dspark-6.00bpw"));
    let target = std::env::var_os("REXL3_DSPARK_TARGET_DIR").map(PathBuf::from).unwrap_or_else(|| m.join("hf"));
    let oracle = std::env::var_os("REXL3_DSPARK_ORACLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join("projects/minicpm-dspark/oracle"));
    for (what, p) in [("draft", &draft), ("target", &target)] {
        if !p.join("config.json").exists() {
            eprintln!("not run: no {what} directory at {}", p.display());
            return None;
        }
    }
    Some((draft, target, oracle))
}

// ----------------------------------------------------------------------------------------------------- oracle dumps

struct Oracle {
    dir: PathBuf,
    mode: String,
    /// "fp32" or "bf16": the precision the oracle rounded the base logits to before the top-K and the walk.
    logits: String,
    name: String,
    t: BTreeMap<String, (String, Vec<usize>, String)>,
}

impl Oracle {
    fn load(dir: &Path) -> Option<(Self, String)> {
        let m: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).ok()?).ok()?;
        let meta = &m["meta"];
        let mut t = BTreeMap::new();
        for (k, v) in m["tensors"].as_object()? {
            t.insert(
                k.clone(),
                (
                    v["dtype"].as_str()?.to_string(),
                    v["shape"].as_array()?.iter().map(|x| x.as_u64().unwrap() as usize).collect(),
                    v["file"].as_str()?.to_string(),
                ),
            );
        }
        let draft = meta["draft"].as_str()?.to_string();
        Some((
            Self {
                dir: dir.to_path_buf(),
                mode: meta["mode"].as_str()?.to_string(),
                logits: meta["logits"].as_str().unwrap_or("fp32").to_string(),
                name: dir.file_name()?.to_string_lossy().to_string(),
                t,
            },
            draft,
        ))
    }
    fn f32(&self, k: &str) -> Vec<f32> {
        let (dt, _, f) = &self.t[k];
        assert_eq!(dt, "f32", "{k}");
        std::fs::read(self.dir.join(f)).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
    fn i32(&self, k: &str) -> Vec<i32> {
        let (dt, _, f) = &self.t[k];
        assert_eq!(dt, "i32", "{k}");
        std::fs::read(self.dir.join(f)).unwrap().chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
    fn scalar(&self, k: &str) -> i32 {
        self.i32(k)[0]
    }
}

fn oracles(dir: &Path, draft: &Path) -> Vec<Oracle> {
    let want = std::fs::canonicalize(draft).unwrap();
    let mut v = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if let Some((o, d)) = Oracle::load(&e.path()) {
                if std::fs::canonicalize(&d).ok().as_deref() == Some(want.as_path()) {
                    v.push(o);
                }
            }
        }
    }
    v.sort_by(|a, b| a.name.cmp(&b.name));
    v
}

fn bf16_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut d, mut n) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += (x as f64 - y as f64).powi(2);
        n += (y as f64).powi(2);
    }
    (d / n.max(1e-300)).sqrt()
}

fn widen(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| bf16_f32(b)).collect()
}

// ----------------------------------------------------------------------------------------------------- the rig

struct Rig {
    dm: DraftModel,
    tgt: DenseTarget,
    st: Stream,
    draft_dir: PathBuf,
}

fn options(slots: usize, streams: usize) -> DraftOptions {
    DraftOptions {
        slots,
        pages: 64,
        max_streams: streams,
        max_ctx_rows: 512,
        max_context: 8192,
        max_blocks: 40,
        vocab: VOCAB,
        target_layers: 40,
        head_stride: VOCAB,
        head: HeadKind::Dense,
    }
}

fn rig(draft: &Path, target: &Path, slots: usize, streams: usize) -> Rig {
    let cfg = DSparkConfig::from_dir(draft).expect("draft config");
    let t0 = Instant::now();
    let tgt = DenseTarget::load(target, VOCAB).expect("target tables");
    let (dm, bytes) = DraftModel::new(&cfg, draft, options(slots, streams)).expect("draft");
    println!(
        "draft {:?} loaded in {:.1}s: {:.1} MiB (weights {:.1} MiB)",
        dm.weight_kind(),
        t0.elapsed().as_secs_f64(),
        bytes as f64 / 1048576.0,
        dm.weight_bytes() as f64 / 1048576.0
    );
    Rig { dm, tgt, st: Stream::new().unwrap(), draft_dir: draft.to_path_buf() }
}

/// The oracle's taps for rows `0..rows` as one bf16 `[rows, 5 x 2048]` buffer.
fn taps_buffer(o: &Oracle, rows: usize) -> DevBuf {
    let mut out = vec![0u16; rows * TAPS.len() * HIDDEN];
    for (ti, &n) in TAPS.iter().enumerate() {
        let t = o.f32(&format!("tap_{n}"));
        for r in 0..rows {
            for d in 0..HIDDEN {
                out[r * TAPS.len() * HIDDEN + ti * HIDDEN + d] = bf16_bits(t[r * HIDDEN + d]);
            }
        }
    }
    DevBuf::from_slice(&out).unwrap()
}

fn blocks_for(positions: usize, first: u32) -> Vec<u32> {
    (0..positions.div_ceil(256)).map(|i| first + i as u32).collect()
}

fn greedy() -> Sampling {
    Sampling { seed: 0, temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0 }
}

struct Run {
    props: Vec<Proposal>,
    dump: StageDump,
}

fn round(r: &mut Rig, reqs: &[ProposeReq<'_>], k: usize) -> Run {
    let mut dump = StageDump::default();
    let head = r.tgt.head();
    let props = r.dm.propose_debug(&r.tgt.embed, &head, reqs, k, &r.st, Some(&mut dump)).unwrap();
    Run { props, dump }
}

fn stage_row(d: &StageDump, stream: usize) -> (Vec<Vec<u16>>, Vec<u16>, Vec<u16>) {
    let rows = |v: &Vec<u16>| v[stream * BLOCK * HIDDEN..(stream + 1) * BLOCK * HIDDEN].to_vec();
    (
        d.layer_out.iter().map(rows).collect(),
        rows(&d.final_norm),
        d.logits[stream * BLOCK * d.stride..(stream + 1) * BLOCK * d.stride].to_vec(),
    )
}

// ----------------------------------------------------------------------------------------------------- tests

#[test]
fn stages_match_the_torch_oracle() {
    let Some((draft, target, odir)) = dirs() else { return };
    let os = oracles(&odir, &draft);
    if os.is_empty() {
        eprintln!("not run: no oracle dumps for {} in {}", draft.display(), odir.display());
        return;
    }
    let mut r = rig(&draft, &target, 2, 2);
    let g = greedy();
    println!(
        "{:<26} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7} {:>6}",
        "case", "fc", "hnorm", "ctx_k", "ctx_v", "embed", "layer0", "layer4", "logits", "conf", "tok"
    );
    let mut worst = [0f64; 8];
    for o in os.iter().filter(|o| o.mode == "fp32") {
        let a = o.scalar("anchor_pos") as usize;
        let ids = o.i32("ids");
        let anchor = o.scalar("anchor_token") as u32;
        assert_eq!(ids[a] as u32, anchor);
        let taps = taps_buffer(o, a);
        r.dm.begin_slot(0, &blocks_for(a + BLOCK, 0)).unwrap();
        let mut cd = CtxDump::default();
        r.dm.ingest_taps_debug(0, 0, a, &taps, 0, &r.st, Some(&mut cd)).unwrap();
        let rl = |got: &[u16], want: &str| rel_l2(&widen(got), &o.f32(want)[..got.len()]);
        let (fc, hn) = (rl(&cd.fc, "fc"), rl(&cd.hidden_norm, "hidden_norm"));
        let (mut kk, mut vv) = (0f64, 0f64);
        for l in 0..5 {
            let (k, v) = r.dm.read_kv(0, l, 0, a).unwrap();
            kk = kk.max(rl(&k, &format!("ctx_k{l}")));
            vv = vv.max(rl(&v, &format!("ctx_v{l}")));
        }
        let req = ProposeReq { slot: 0, anchor, p: a, sampling: &g };
        let run = round(&mut r, &[req], 64);
        let d = &run.dump;
        let emb = rl(&d.block_embed, "block_embed");
        let l0 = rl(&d.layer_out[0], "layer0_out");
        let l4 = rl(&d.layer_out[4], "layer4_out");
        let lg = rl(&d.logits, "base_logits");
        let fnorm = rl(&d.final_norm, "final_norm");
        // candidates and tokens of the top-64 walk against the oracle's (the same base logits up to the engine's bf16)
        let (oidx, otok, oconf) = (o.i32("idx_64"), o.i32("tokens_64"), o.f32("conf_64"));
        let overlap: f64 = (0..BLOCK)
            .map(|i| {
                let e: std::collections::HashSet<i32> = d.cand_ids[i * 64..(i + 1) * 64].iter().copied().collect();
                oidx[i * 64..(i + 1) * 64].iter().filter(|x| e.contains(x)).count() as f64 / 64.0
            })
            .sum::<f64>()
            / BLOCK as f64;
        let tok_eq = run.props[0].tokens.iter().zip(&otok).take_while(|(a, b)| **a as i32 == **b).count();
        let cdiff = run.props[0].confidence.iter().zip(&oconf).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!(
            "{:<26} {fc:>9.2e} {hn:>9.2e} {kk:>9.2e} {vv:>9.2e} {emb:>9.2e} {l0:>9.2e} {l4:>9.2e} {lg:>9.2e} {cdiff:>7.4} {tok_eq:>3}/15  (final_norm {fnorm:.2e}, candidate overlap {overlap:.3})",
            o.name
        );
        for (w, v) in worst.iter_mut().zip([fc, hn, kk, vv, l0, l4, lg, fnorm]) {
            *w = w.max(v);
        }
        assert!(emb < 1e-6, "{}: the block's embedding rows are the target table's rows (bf16 exact): {emb:.2e}", o.name);
        assert!(overlap > 0.95, "{}: candidate overlap {overlap}", o.name);
        assert!(tok_eq >= 1, "{}: the first draft token differs from the oracle's", o.name);
    }
    println!(
        "worst: fc {:.2e} hnorm {:.2e} ctx_k {:.2e} ctx_v {:.2e} layer0 {:.2e} layer4 {:.2e} logits {:.2e} final_norm {:.2e}",
        worst[0], worst[1], worst[2], worst[3], worst[4], worst[5], worst[6], worst[7]
    );
    // calibrated gates (about twice the worst engine value of this box's runs; see section 11 of the plan)
    let gates = [
        ("fc", worst[0], 3.5e-3),
        ("hidden_norm", worst[1], 5.0e-3),
        ("ctx_k", worst[2], 7.0e-3),
        ("ctx_v", worst[3], 7.0e-3),
        ("layer0", worst[4], 7.5e-3),
        ("layer4", worst[5], 2.4e-2),
        ("logits", worst[6], 2.4e-2),
        ("final_norm", worst[7], 2.6e-2),
    ];
    for (n, v, g) in gates {
        assert!(v < g, "stage {n}: rel_l2 {v:.3e} against the fp32 oracle exceeds the gate {g:.1e}");
    }
    let _ = (RANK, &r.draft_dir, Instant::now());
}

#[test]
fn plan_equals_allocation() {
    let Some((draft, _target, _)) = dirs() else { return };
    let tcfg = MiniCpmConfig::from_dir(&home().join("models/minicpm-v-4.7-35b-a3b/hf")).expect("target config");
    let dcfg = DSparkConfig::from_dir(&draft).unwrap();
    for pages in [1usize, 17, 64] {
        let kv = gb_model::families::minicpmv4_7::dspark::DraftKv::new(2, pages, 8).unwrap();
        let items = allocations(&tcfg, Some(&dcfg));
        let d = compare_with_plan(&items, &kv);
        assert!(d.is_empty(), "{d:?}");
        assert_eq!(kv.bytes(), pages as u64 * 256 * 10_240);
    }
    let kv = gb_model::families::minicpmv4_7::dspark::DraftKv::new(1, 3, 8).unwrap();
    let items = allocations(&tcfg, Some(&dcfg));
    let mut wrong = items.clone();
    for i in wrong.iter_mut().filter(|i| i.name == "kv_draft") {
        i.bytes /= 2;
    }
    assert!(compare_with_plan(&wrong, &kv)[0].starts_with("kv_draft: the device holds"));
}

// ----------------------------------------------------------------------------------------------------- host mirrors

/// A minimal safetensors reader: rows of a 2-D bf16 tensor of the draft's file.
struct St {
    f: std::fs::File,
    data0: u64,
    hdr: Value,
}

impl St {
    fn open(dir: &Path) -> Self {
        let mut f = std::fs::File::open(dir.join("model.safetensors")).unwrap();
        let mut n = [0u8; 8];
        f.read_exact(&mut n).unwrap();
        let n = u64::from_le_bytes(n);
        let mut h = vec![0u8; n as usize];
        f.read_exact(&mut h).unwrap();
        Self { f, data0: 8 + n, hdr: serde_json::from_slice(&h).unwrap() }
    }
    fn rows(&mut self, name: &str, row: usize, rows: usize, width: usize) -> Vec<u16> {
        let off = self.hdr[name]["data_offsets"][0].as_u64().unwrap();
        assert_eq!(self.hdr[name]["dtype"], "BF16", "{name}");
        self.f.seek(SeekFrom::Start(self.data0 + off + (row * width * 2) as u64)).unwrap();
        let mut b = vec![0u8; rows * width * 2];
        self.f.read_exact(&mut b).unwrap();
        b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }
    fn f32s(&mut self, name: &str, n: usize) -> Vec<f32> {
        widen(&self.rows(name, 0, 1, n))
    }
}

fn dot(a: &[u16], b: &[u16]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| bf16_f32(x) as f64 * bf16_f32(y) as f64).sum()
}

/// A bf16 value to fp16 bits, round to nearest even (the Markov tables' magnitudes; the few below 6.1e-5 are subnormal in fp16).
fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if v == 0.0 || exp <= 0 {
        return if v == 0.0 { sign } else { sign | ((v.abs() * 16777216.0).round() as u16).min(0x3ff) };
    }
    let mut m = (man >> 13) as u16;
    let rest = man & 0x1fff;
    let mut e = exp as u16;
    if rest > 0x1000 || (rest == 0x1000 && (m & 1) == 1) {
        m += 1;
        if m == 0x400 {
            m = 0;
            e += 1;
        }
    }
    sign | (e << 10) | m
}

#[test]
fn the_walk_and_the_confidence_equal_their_host_mirrors() {
    let Some((draft, target, odir)) = dirs() else { return };
    let os = oracles(&odir, &draft);
    let Some(o) = os.iter().find(|o| o.mode == "fp32") else {
        eprintln!("not run: no oracle dumps in {}", odir.display());
        return;
    };
    let mut r = rig(&draft, &target, 2, 2);
    let a = o.scalar("anchor_pos") as usize;
    let ids = o.i32("ids");
    let taps = taps_buffer(o, a);
    r.dm.begin_slot(0, &blocks_for(a + BLOCK, 0)).unwrap();
    r.dm.ingest_taps(0, 0, a, &taps, 0, &r.st).unwrap();
    let mut st = St::open(&draft);
    let (cw, cb) = (st.f32s("confidence_head.proj.weight", HIDDEN + RANK), st.f32s("confidence_head.proj.bias", 1)[0]);
    let g = greedy();
    for k in [16usize, 32, 64] {
        let req = ProposeReq { slot: 0, anchor: ids[a] as u32, p: a, sampling: &g };
        let run = round(&mut r, &[req], k);
        let (d, toks, conf) = (&run.dump, &run.props[0].tokens, &run.props[0].confidence);
        let mut prev = ids[a] as usize;
        for i in 0..BLOCK {
            let w1 = st.rows("markov_head.markov_w1.weight", prev, 1, RANK);
            let mut best = (f64::NEG_INFINITY, 0usize);
            let mut picked = f64::NEG_INFINITY;
            for c in 0..k {
                let id = d.cand_ids[i * k + c] as usize;
                let w2 = st.rows("markov_head.markov_w2.weight", id, 1, RANK);
                let sc = d.cand_vals[i * k + c] as f64 + dot(&w1, &w2);
                if sc > best.0 {
                    best = (sc, id);
                }
                if id == toks[i] as usize {
                    picked = sc;
                }
            }
            assert!(picked > f64::NEG_INFINITY, "k {k} row {i}: the pick {} is not among the candidates", toks[i]);
            assert!(
                best.0 - picked < 1e-3,
                "k {k} row {i}: pick {} scores {picked}, the host's best {} scores {}",
                toks[i],
                best.1,
                best.0
            );
            // the confidence of the row from the engine's own final-norm row and the pick's predecessor
            let h = &d.final_norm[i * HIDDEN..(i + 1) * HIDDEN];
            let m16: Vec<u16> = w1.iter().map(|&b| f32_to_f16(bf16_f32(b))).collect();
            let want = confidence_host(h, &m16, &cw, cb);
            assert!((conf[i] - want).abs() <= want.abs() * 1e-6 + 1e-9, "k {k} row {i}: confidence {} vs {want}", conf[i]);
            prev = toks[i] as usize;
        }
    }
}

// ----------------------------------------------------------------------------------------------------- invariances

/// Two contexts of different lengths from two oracle dumps.
fn two_cases(os: &[Oracle]) -> Option<(&Oracle, &Oracle)> {
    let mut it = os.iter().filter(|o| o.mode == "fp32" && o.logits == "fp32");
    let a = it.next()?;
    let b = it.find(|o| o.scalar("anchor_pos") != a.scalar("anchor_pos"))?;
    Some((a, b))
}

fn setup_two(r: &mut Rig, a: &Oracle, b: &Oracle, slots: &[usize]) -> ((usize, u32), (usize, u32)) {
    let (pa, pb) = (a.scalar("anchor_pos") as usize, b.scalar("anchor_pos") as usize);
    let (ta, tb) = (taps_buffer(a, pa), taps_buffer(b, pb));
    for (i, &s) in slots.iter().enumerate() {
        let (p, taps) = if i % 2 == 0 { (pa, &ta) } else { (pb, &tb) };
        r.dm.begin_slot(s, &blocks_for(pb.max(pa) + BLOCK, (s * 4) as u32)).unwrap();
        r.dm.ingest_taps(s, 0, p, taps, 0, &r.st).unwrap();
    }
    ((pa, a.i32("ids")[pa] as u32), (pb, b.i32("ids")[pb] as u32))
}

fn same_bits(x: &Run, xi: usize, y: &Run, yi: usize, what: &str) {
    let (xl, xf, xg) = stage_row(&x.dump, xi);
    let (yl, yf, yg) = stage_row(&y.dump, yi);
    assert_eq!(xl, yl, "{what}: layer outputs");
    assert_eq!(xf, yf, "{what}: final norm");
    assert!(xg == yg, "{what}: logits");
    assert_eq!(x.props[xi], y.props[yi], "{what}: proposal");
}

#[test]
fn a_stream_is_the_same_bits_alone_batched_reordered_and_repeated() {
    let Some((draft, target, odir)) = dirs() else { return };
    let os = oracles(&odir, &draft);
    let Some((a, b)) = two_cases(&os) else {
        eprintln!("not run: two oracle dumps with different anchors needed in {}", odir.display());
        return;
    };
    let mut r = rig(&draft, &target, 4, 4);
    if r.dm.weight_kind() == WeightKind::Dense {
        eprintln!("not run: the dense reference path is cuBLAS, which is not row invariant (the EXL3 path is, by construction)");
        return;
    }
    let ((pa, ta), (pb, tb)) = setup_two(&mut r, a, b, &[0, 1, 2, 3]);
    let g = greedy();
    let req = |slot: usize, p: usize, t: u32| ProposeReq { slot, anchor: t, p, sampling: &g };
    let solo_a = round(&mut r, &[req(0, pa, ta)], 64);
    let solo_b = round(&mut r, &[req(1, pb, tb)], 64);
    let again = round(&mut r, &[req(0, pa, ta)], 64);
    same_bits(&solo_a, 0, &again, 0, "determinism: a round twice");
    let ab = round(&mut r, &[req(0, pa, ta), req(1, pb, tb)], 64);
    same_bits(&solo_a, 0, &ab, 0, "stream A alone vs with B");
    same_bits(&solo_b, 0, &ab, 1, "stream B alone vs with A");
    let ba = round(&mut r, &[req(1, pb, tb), req(0, pa, ta)], 64);
    same_bits(&solo_a, 0, &ba, 1, "stream A: reordered");
    same_bits(&solo_b, 0, &ba, 0, "stream B: reordered");
    // four streams, two of them copies in other slots (other pages)
    let four = round(&mut r, &[req(0, pa, ta), req(1, pb, tb), req(2, pa, ta), req(3, pb, tb)], 64);
    for (i, solo) in [(0, &solo_a), (1, &solo_b), (2, &solo_a), (3, &solo_b)] {
        same_bits(solo, 0, &four, i, &format!("stream {i} of four"));
    }
    // K changes the candidates the walk sees, not the layers' activations
    let k16 = round(&mut r, &[req(0, pa, ta)], 16);
    let (l16, f16, g16) = stage_row(&k16.dump, 0);
    let (l64, f64_, g64) = stage_row(&solo_a.dump, 0);
    assert!(l16 == l64 && f16 == f64_ && g16 == g64, "the layers do not depend on K");
}

#[test]
fn chunking_the_context_does_not_move_a_bit() {
    let Some((draft, target, odir)) = dirs() else { return };
    let os = oracles(&odir, &draft);
    let Some(o) = os.iter().find(|o| o.mode == "fp32") else {
        eprintln!("not run: no oracle dumps in {}", odir.display());
        return;
    };
    let mut r = rig(&draft, &target, 2, 1);
    if r.dm.weight_kind() == WeightKind::Dense {
        eprintln!("not run: the dense reference path is cuBLAS, which is not row invariant (the EXL3 path is, by construction)");
        return;
    }
    let a = o.scalar("anchor_pos") as usize;
    let taps = taps_buffer(o, a);
    r.dm.begin_slot(0, &blocks_for(a + BLOCK, 0)).unwrap();
    r.dm.ingest_taps(0, 0, a, &taps, 0, &r.st).unwrap();
    r.dm.begin_slot(1, &blocks_for(a + BLOCK, 4)).unwrap();
    let mut pos = 0;
    let chunks = [1usize, 7, 100, 3, 129, a - 240];
    assert!(a > 240);
    for c in chunks {
        r.dm.ingest_taps(1, pos, c, &taps, pos, &r.st).unwrap();
        pos += c;
    }
    assert_eq!(pos, a);
    for l in 0..5 {
        let (k0, v0) = r.dm.read_kv(0, l, 0, a).unwrap();
        let (k1, v1) = r.dm.read_kv(1, l, 0, a).unwrap();
        assert!(k0 == k1 && v0 == v1, "layer {l}: K/V rows differ between a whole ingest and chunks of 1, 7, 100, 3, 129, ...");
    }
}

#[test]
fn rollback_equals_replay_and_nothing_past_the_bound_is_read() {
    let Some((draft, target, odir)) = dirs() else { return };
    let os = oracles(&odir, &draft);
    let Some(o) = os.iter().find(|o| o.mode == "fp32") else {
        eprintln!("not run: no oracle dumps in {}", odir.display());
        return;
    };
    let mut r = rig(&draft, &target, 2, 1);
    if r.dm.weight_kind() == WeightKind::Dense {
        eprintln!("not run: the dense reference path is cuBLAS, which is not row invariant (the EXL3 path is, by construction)");
        return;
    }
    let g = greedy();
    let a = o.scalar("anchor_pos") as usize;
    let ids = o.i32("ids");
    let taps = taps_buffer(o, a);
    let req = |slot: usize, p: usize| ProposeReq { slot, anchor: ids[p] as u32, p, sampling: &g };
    r.dm.begin_slot(0, &blocks_for(a + BLOCK, 0)).unwrap();
    r.dm.ingest_taps(0, 0, a, &taps, 0, &r.st).unwrap();
    let r1 = round(&mut r, &[req(0, a)], 64);
    // a proposal at the wrong committed length is refused by name
    let e = r.dm.propose(&r.tgt.embed, &r.tgt.head(), &[req(0, a - 1)], 16, &r.st).unwrap_err().to_string();
    assert!(e.contains("holds"), "{e}");
    // roll back 50 rows: slot 0 now holds stale K/V for 250..300 and the first round's block rows at 300..314
    r.dm.rollback(0, a - 50).unwrap();
    assert_eq!(r.dm.len(0), a - 50);
    let r2 = round(&mut r, &[req(0, a - 50)], 64);
    // a clean slot that never saw rows past a - 50
    r.dm.begin_slot(1, &blocks_for(a + BLOCK, 4)).unwrap();
    r.dm.ingest_taps(1, 0, a - 50, &taps, 0, &r.st).unwrap();
    let clean = round(&mut r, &[req(1, a - 50)], 64);
    same_bits(&r2, 0, &clean, 0, "rollback then propose == a slot that never held the rows");
    // replay the 50 rows and ask again at the original position: the first round's bits
    r.dm.ingest_taps(0, a - 50, 50, &taps, a - 50, &r.st).unwrap();
    assert_eq!(r.dm.len(0), a);
    let r3 = round(&mut r, &[req(0, a)], 64);
    same_bits(&r1, 0, &r3, 0, "rollback then replay == the first round");
    // a rollback past the committed length and a hole are refused
    assert!(r.dm.rollback(0, a + 1).is_err());
    assert!(r.dm.ingest_taps(1, a - 40, 1, &taps, 0, &r.st).is_err(), "a hole in the context must be refused");
}

/// The sampled walk is wired through the target's keyed sampler: a seed gives one chain (twice the same), other seeds give other
/// chains, every pick is one of the row's candidates, and a greedy request is the argmax chain. That the draws agree with the target's
/// own keyed draws at the same positions is `drafted_equals_serial (sampled)`, E4b's test (it needs the engine's sampler).
#[test]
fn a_sampled_walk_is_keyed_deterministic_and_inside_the_candidates() {
    let Some((draft, target, odir)) = dirs() else { return };
    let os = oracles(&odir, &draft);
    let Some(o) = os.iter().find(|o| o.mode == "fp32") else { return };
    let mut r = rig(&draft, &target, 1, 1);
    let a = o.scalar("anchor_pos") as usize;
    let ids = o.i32("ids");
    let taps = taps_buffer(o, a);
    r.dm.begin_slot(0, &blocks_for(a + BLOCK, 0)).unwrap();
    r.dm.ingest_taps(0, 0, a, &taps, 0, &r.st).unwrap();
    let samp = |seed: u64| Sampling { seed, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 };
    let go = |r: &mut Rig, s: &Sampling| {
        let req = ProposeReq { slot: 0, anchor: ids[a] as u32, p: a, sampling: s };
        round(r, &[req], 32)
    };
    let (s1, s2, s3) = (samp(1), samp(2), samp(3));
    let first = go(&mut r, &s1);
    let again = go(&mut r, &s1);
    assert_eq!(first.props[0], again.props[0], "one seed, one chain");
    let others: Vec<Vec<u32>> = [&s2, &s3].iter().map(|s| go(&mut r, s).props[0].tokens.clone()).collect();
    assert!(
        others.iter().any(|t| *t != first.props[0].tokens),
        "three seeds gave one chain over {BLOCK} positions: the sampled walk is not keyed by the seed"
    );
    for run in [&first, &go(&mut r, &s2)] {
        for i in 0..BLOCK {
            let cands = &run.dump.cand_ids[i * 32..(i + 1) * 32];
            assert!(cands.contains(&(run.props[0].tokens[i] as i32)), "row {i}: a pick outside the candidates");
        }
    }
    // a greedy request is the argmax chain of the same candidates
    let g = greedy();
    let greedy_run = go(&mut r, &g);
    let greedy_again = go(&mut r, &g);
    assert_eq!(greedy_run.props[0], greedy_again.props[0]);
}

#[test]
fn refusals_are_named() {
    let Some((draft, target, odir)) = dirs() else { return };
    let os = oracles(&odir, &draft);
    let Some(o) = os.iter().find(|o| o.mode == "fp32") else { return };
    let mut r = rig(&draft, &target, 2, 2);
    let g = greedy();
    let a = o.scalar("anchor_pos") as usize;
    let taps = taps_buffer(o, a);
    // no pages: the ingest is refused before anything is written
    assert!(r.dm.ingest_taps(0, 0, 4, &taps, 0, &r.st).unwrap_err().to_string().contains("pages for"));
    r.dm.begin_slot(0, &blocks_for(a + BLOCK, 0)).unwrap();
    r.dm.ingest_taps(0, 0, a, &taps, 0, &r.st).unwrap();
    let ids = o.i32("ids");
    let head = r.tgt.head();
    let req = ProposeReq { slot: 0, anchor: ids[a] as u32, p: a, sampling: &g };
    assert!(r.dm.propose(&r.tgt.embed, &head, &[req], 0, &r.st).is_err(), "k 0");
    assert!(r.dm.propose(&r.tgt.embed, &head, &[req], 65, &r.st).is_err(), "k 65");
    assert!(r.dm.propose(&r.tgt.embed, &head, &[req, req], 16, &r.st).is_err(), "one slot twice");
    assert!(r.dm.set_blocks(0, &[9999]).is_err(), "a block past the pool");
    assert!(r.dm.propose(&r.tgt.embed, &head, &[], 16, &r.st).unwrap().is_empty());
}

/// The target pack's EXL3 `lm_head` (N 248,192) and bf16 embedding table, when the pack is there.
fn load_exl3_target() -> Option<(DevBuf, gb_cuda::exl3_linear::Exl3Weights)> {
    let pack_dir = std::env::var_os("REXL3_DSPARK_PACK")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join("models/minicpm-v-4.7-35b-a3b/packs/q3-6.00bpw"));
    if !pack_dir.join("model.safetensors.index.json").exists() {
        return None;
    }
    let pack = gb_model::pack::Pack::open(&pack_dir).expect("target pack");
    let mut ctx = gb_model::loadctx::LoadCtx::from_env();
    let head = ctx.group(&pack, "lm_head").expect("lm_head group");
    let embed = ctx.raw16(&pack, "model.language_model.embed_tokens.weight", "BF16", &[VOCAB, HIDDEN]).expect("embed");
    ctx.finish().expect("load");
    Some((embed, head))
}

/// The serving head: the draft behind the target's EXL3 `lm_head` (stride 248,192, 48 pad columns). The pad columns are filled with
/// the largest finite bf16 before every top-K: a mask that failed would propose an id past the vocabulary. The same rounds through the
/// dense head give the agreement of the two (they differ by the head's 6-bit quantisation).
#[test]
fn the_exl3_head_never_proposes_a_pad_id() {
    let Some((draft, target, odir)) = dirs() else { return };
    let Some((embed, head_w)) = load_exl3_target() else {
        eprintln!("not run: no target pack (REXL3_DSPARK_PACK)");
        return;
    };
    let os = oracles(&odir, &draft);
    let os: Vec<&Oracle> = os.iter().filter(|o| o.mode == "fp32" && o.logits == "fp32").collect();
    if os.is_empty() {
        eprintln!("not run: no oracle dumps in {}", odir.display());
        return;
    }
    let cfg = DSparkConfig::from_dir(&draft).expect("draft config");
    let st = Stream::new().unwrap();
    let g = greedy();
    let valid = VOCAB;
    assert_eq!(head_w.n, 248_192, "the pack pads the head to 248,192");
    let mut opts = options(1, 1);
    opts.head_stride = head_w.n;
    opts.head = HeadKind::Exl3 { bits: head_w.bits };
    let (mut dm, bytes) = DraftModel::new(&cfg, &draft, opts).expect("draft with the EXL3 head");
    dm.poison_pad_columns(true);
    // a dense-head draft for the agreement figure
    let dense = DenseTarget::load(&target, VOCAB).expect("dense target");
    let (mut dm_dense, _) = DraftModel::new(&cfg, &draft, options(1, 1)).expect("dense-head draft");
    println!("EXL3-head draft: {:.1} MiB in all", bytes as f64 / 1048576.0);
    // the head is the construction's: a head of another kind or width is refused by name
    let (mut agree, mut total, mut first_agree) = (0usize, 0usize, 0usize);
    for o in &os {
        let a = o.scalar("anchor_pos") as usize;
        let ids = o.i32("ids");
        let taps = taps_buffer(o, a);
        for dm in [&mut dm, &mut dm_dense] {
            dm.begin_slot(0, &blocks_for(a + BLOCK, 0)).unwrap();
            dm.ingest_taps(0, 0, a, &taps, 0, &st).unwrap();
        }
        let req = ProposeReq { slot: 0, anchor: ids[a] as u32, p: a, sampling: &g };
        let wrong = dm.propose(&dense.embed, &dense.head(), &[req], 16, &st).unwrap_err().to_string();
        assert!(wrong.contains("head"), "{wrong}");
        for k in [16usize, 64] {
            let mut dump = StageDump::default();
            let exl3 =
                dm.propose_debug(&embed, &Head::Exl3 { w: &head_w, valid }, &[req], k, &st, Some(&mut dump)).unwrap().remove(0);
            // the pad columns of the head's output are the poison (the knob ran), yet no id past the vocabulary comes out
            for r in 0..BLOCK {
                let pad = &dump.logits[r * dump.stride + valid..(r + 1) * dump.stride];
                assert!(pad.iter().all(|&b| b == 0x7f7f), "row {r}: the pad columns were not poisoned");
            }
            assert!(dump.cand_ids.iter().all(|&i| (i as usize) < valid), "a candidate id past the vocabulary: k {k}");
            assert!(exl3.tokens.iter().all(|&t| (t as usize) < valid), "a drafted id past the vocabulary: {:?}", exl3.tokens);
            let dn = dm_dense.propose(&dense.embed, &dense.head(), &[req], k, &st).unwrap().remove(0);
            total += BLOCK;
            agree += exl3.tokens.iter().zip(&dn.tokens).filter(|(a, b)| a == b).count();
            first_agree += (exl3.tokens[0] == dn.tokens[0]) as usize;
        }
    }
    println!(
        "EXL3 head vs dense bf16 head: {agree}/{total} drafted tokens equal ({:.1}%), first token {first_agree}/{} equal",
        100.0 * agree as f64 / total as f64,
        total / BLOCK
    );
}

// ----------------------------------------------------------------------------------------------------- timing

/// Round time at 1, 2 and 4 streams (the budget rule's draft cost) and the cost of absorbing a round's tap rows, median of 30 after
/// warm-up, at the context length of the oracle's longest case. Exclusive lock for the numbers.
#[test]
#[ignore]
fn timing_ms_per_round() {
    let Some((draft, target, odir)) = dirs() else { return };
    let os = oracles(&odir, &draft);
    let Some(o) = os.iter().filter(|o| o.mode == "fp32").max_by_key(|o| o.scalar("anchor_pos")) else { return };
    // the serving head: the target pack's EXL3 lm_head (N 248,192) and its bf16 embedding, when the pack is there; else the dense stand-ins
    let exl3 = load_exl3_target();
    let cfg = DSparkConfig::from_dir(&draft).expect("draft config");
    let stride = exl3.as_ref().map_or(VOCAB, |(_, h)| h.n);
    let mut opts = options(4, 4);
    opts.head_stride = stride;
    if let Some((_, h)) = &exl3 {
        opts.head = HeadKind::Exl3 { bits: h.bits };
    }
    let (mut dm, _) = DraftModel::new(&cfg, &draft, opts).expect("draft");
    let dense = if exl3.is_none() { Some(DenseTarget::load(&target, VOCAB).expect("target tables")) } else { None };
    let st = Stream::new().unwrap();
    let g = greedy();
    let a = o.scalar("anchor_pos") as usize;
    let ids = o.i32("ids");
    let taps = taps_buffer(o, a);
    for s in 0..4 {
        dm.begin_slot(s, &blocks_for(a + BLOCK, (s * 4) as u32)).unwrap();
        dm.ingest_taps(s, 0, a, &taps, 0, &st).unwrap();
    }
    let (embed, head) = match (&exl3, &dense) {
        (Some((e, h)), _) => (e, Head::Exl3 { w: h, valid: VOCAB }),
        (None, Some(d)) => (&d.embed, d.head()),
        _ => unreachable!(),
    };
    let med = |mut v: Vec<f64>| {
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        v[v.len() / 2]
    };
    println!(
        "timing at context {a}: {:?} draft weights ({:.0} MiB), {} head",
        dm.weight_kind(),
        dm.weight_bytes() as f64 / 1048576.0,
        if exl3.is_some() { "EXL3 6-bit lm_head of the target pack" } else { "dense fp16 (stand-in)" }
    );
    for n in [1usize, 2, 4] {
        for k in [16usize, 64] {
            let reqs: Vec<ProposeReq> =
                (0..n).map(|s| ProposeReq { slot: s, anchor: ids[a] as u32, p: a, sampling: &g }).collect();
            for _ in 0..5 {
                dm.propose(embed, &head, &reqs, k, &st).unwrap();
            }
            let t: Vec<f64> = (0..30)
                .map(|_| {
                    let t0 = Instant::now();
                    dm.propose(embed, &head, &reqs, k, &st).unwrap();
                    t0.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            println!("  {n} stream(s), k {k:>2}: {:.3} ms a round (median of 30)", med(t));
        }
    }
    for rows in [1usize, 4, 16] {
        let t: Vec<f64> = (0..30)
            .map(|_| {
                let t0 = Instant::now();
                dm.ingest_taps(0, a - rows, rows, &taps, a - rows, &st).unwrap();
                st.sync().unwrap();
                t0.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        println!("  ingest of {rows:>2} tap rows: {:.3} ms (median of 30)", med(t));
    }
    // the same rounds split into their stages once, for the table of where the milliseconds go
    let reqs: Vec<ProposeReq> = (0..1).map(|s| ProposeReq { slot: s, anchor: ids[a] as u32, p: a, sampling: &g }).collect();
    let mut dump = StageDump::default();
    let t0 = Instant::now();
    dm.propose_debug(embed, &head, &reqs, 64, &st, Some(&mut dump)).unwrap();
    println!("  (one debug round with read-backs: {:.3} ms)", t0.elapsed().as_secs_f64() * 1e3);
}
