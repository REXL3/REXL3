//! The layer-by-layer oracle harness (`families::nemotron_h_omni::oracle`) against the fork's real dumps.
//!
//! Env-gated: `REXL3_NEMO_ORACLE_DIR` names the fork's dump directory (`ref-text-1k`: `ids_<s>.u32`, `top1_<s>.i32`,
//! `logits_<s>.f16`, `layers0/m<iii>.f16` and `layers0/meta.json`). Unset, every test says so and passes. The test that pushes buffers
//! through the [`Tap`] also needs a CUDA device; the rest is host-only.
//!
//! - `the_dump_as_the_model_output_gives_zero_error_and_full_top_1`: each of the 91 modules of sample 0 is uploaded and tapped as if
//!   it were the model's activation (layers in one call, the logits in 64-row chunks), and sample 1's logits likewise. Every rel_l2
//!   and max_abs is exactly 0, top-1 is 100% against both the recorded argmax and the dumped rows, and the committed tolerances pass.
//! - `known_noise_gives_the_known_numbers`: sign noise of known relative size gives that rel_l2, and a known set of top-2 swaps in
//!   the logits gives the known agreement, max gap and rel_l2.
//! - `the_committed_limits_are_what_apply_enforces`: the dump scaled to 0.5x the limit passes everywhere, one module at 1.5x fails
//!   by name, a NaN fails.
//! - `a_missing_or_malformed_dump_is_a_clear_error`: a missing file, a short file, a width that differs and a row past the end.

use gb_cuda::{DevBuf, Stream};
use gb_model::families::nemotron_h_omni::model::{Point, Tap};
use gb_model::families::nemotron_h_omni::oracle::{
    argmax, module_file, point_of, table, Comparison, Dump, OracleTap, Tolerances, MODULES, MODULE_LOGITS, TOP1_MIN,
};
use std::path::PathBuf;

const TOKENS: usize = 1024;
const HIDDEN: usize = 4096;
const VOCAB: usize = 131_072;

fn oracle_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("REXL3_NEMO_ORACLE_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("not run: REXL3_NEMO_ORACLE_DIR is not set (the fork's dump directory, ref-text-1k)");
    }
    dir
}

fn width_of(module: usize) -> usize {
    if module == MODULE_LOGITS {
        VOCAB
    } else {
        HIDDEN
    }
}

/// Every module of sample 0 as f32 rows, read through the harness's own reader.
fn load_module(dump: &Dump, module: usize) -> Vec<f32> {
    dump.read(module, 0, TOKENS, width_of(module)).unwrap_or_else(|e| panic!("{e}"))
}

fn assert_all_zero(results: &[Comparison]) {
    for c in results {
        assert_eq!((c.rel_l2, c.max_abs, c.nonfinite), (0.0, 0.0, 0), "{} ({}) is not exact", c.key, c.module);
    }
}

#[test]
fn the_dump_as_the_model_output_gives_zero_error_and_full_top_1() {
    let Some(root) = oracle_dir() else { return };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device (this test taps device buffers)");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let st = Stream::new().unwrap();

    let dump = Dump::open(&root, 0).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(dump.tokens(), TOKENS);
    assert!(dump.has_layers());
    let mut tap = OracleTap::open(&root, 0).unwrap_or_else(|e| panic!("{e}"));
    let installed = tap.clone();
    for module in 0..MODULES {
        let point = point_of(module).unwrap();
        let w = width_of(module);
        let rows = load_module(&dump, module);
        if module == MODULE_LOGITS {
            // 16 chunks of 64 rows: the cursor places each chunk against the matching rows of the dump
            for c in 0..TOKENS / 64 {
                let buf = DevBuf::from_slice(&rows[c * 64 * w..(c + 1) * 64 * w]).unwrap();
                tap.tap(point, &buf, 64, w, &st).unwrap_or_else(|e| panic!("{e}"));
            }
        } else {
            let buf = DevBuf::from_slice(&rows).unwrap();
            tap.tap(point, &buf, TOKENS, w, &st).unwrap_or_else(|e| panic!("{e}"));
        }
    }
    // a mixer output has no dump: no error, no download, counted
    let buf = DevBuf::from_slice(&vec![0f32; 2 * HIDDEN]).unwrap();
    tap.tap(Point::Mixer(0), &buf, 2, HIDDEN, &st).unwrap();

    let results = installed.results();
    assert_eq!(results.len(), MODULES, "one comparison per module");
    assert_all_zero(&results);
    for (m, c) in results.iter().enumerate() {
        assert_eq!((c.module, c.point, c.rows, c.width, c.sample), (m, point_of(m).unwrap(), TOKENS, width_of(m), 0));
        assert!(c.ref_rms > 0.0, "{}: an all-zero reference", c.key);
    }
    assert_eq!(installed.skipped(), vec![(Point::Mixer(0), 2)]);
    let t = results[MODULE_LOGITS].top1.clone().expect("the logits carry top-1");
    assert_eq!((t.rows, t.agree_recorded, t.agree_dumped, t.fork_self_agree), (TOKENS, TOKENS, TOKENS, TOKENS));
    assert_eq!(t.max_gap, 0.0);

    // sample 1 has the logits only
    let mut tap1 = OracleTap::open(&root, 1).unwrap_or_else(|e| panic!("{e}"));
    let probe = tap1.clone();
    let d1 = Dump::open(&root, 1).unwrap();
    assert!(!d1.has_layers());
    let lg1 = d1.read(MODULE_LOGITS, 0, TOKENS, VOCAB).unwrap();
    for c in 0..TOKENS / 128 {
        let buf = DevBuf::from_slice(&lg1[c * 128 * VOCAB..(c + 1) * 128 * VOCAB]).unwrap();
        tap1.tap(Point::Logits, &buf, 128, VOCAB, &st).unwrap();
    }
    // a layer point of that sample is not an error: it has no reference
    let buf = DevBuf::from_slice(&vec![0f32; HIDDEN]).unwrap();
    tap1.tap(Point::Layer(3), &buf, 1, HIDDEN, &st).unwrap();
    let r1 = probe.results();
    assert_eq!(r1.len(), 1);
    assert_all_zero(&r1);
    assert_eq!(r1[0].top1.as_ref().unwrap().agree_recorded, TOKENS);

    // the committed limits hold for the dump itself, over 2 x 1024 tokens
    let mut all = results.clone();
    all.extend(r1);
    let tol = Tolerances::committed();
    let expect: Vec<Point> = (0..MODULES).map(|m| point_of(m).unwrap()).collect();
    let v = tol.apply(&all, &expect);
    assert!(v.passed(), "{:?}", v.failures);
    assert_eq!(v.top1, Some((2 * TOKENS, 2 * TOKENS)));
    eprintln!("{}", table(&all[MODULES - 3..], Some(&tol)));
}

/// A tap fed host rows, for the tests that need no device.
fn feed(tap: &OracleTap, module: usize, rows: &[f32]) -> Result<(), String> {
    tap.compare_host(point_of(module).unwrap(), rows, rows.len() / width_of(module), width_of(module))
}

#[test]
fn known_noise_gives_the_known_numbers() {
    let Some(root) = oracle_dir() else { return };
    let dump = Dump::open(&root, 0).unwrap_or_else(|e| panic!("{e}"));
    let tap = OracleTap::open(&root, 0).unwrap_or_else(|e| panic!("{e}"));

    // sign noise of relative size eps: every element moves by eps * rms, so ||noise|| / ||b|| = eps exactly
    let cases = [(0usize, 1e-3f64), (1, 5e-3), (44, 1e-2), (88, 2e-2), (89, 4e-2)];
    for &(module, eps) in &cases {
        let b = load_module(&dump, module);
        let rms = (b.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / b.len() as f64).sqrt();
        let step = (eps * rms) as f32;
        let mut s = 0x2545_F491_4F6C_DD1Du64 ^ module as u64;
        let noisy: Vec<f32> = b
            .iter()
            .map(|&v| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                if s & 1 == 0 {
                    v + step
                } else {
                    v - step
                }
            })
            .collect();
        feed(&tap, module, &noisy).unwrap();
    }
    let r = tap.results();
    for (c, &(module, eps)) in r.iter().zip(&cases) {
        assert_eq!(c.module, module);
        // f32 rounding of v +- step moves the exact value by about 1e-7 relative
        assert!((c.rel_l2 - eps).abs() <= 2e-5 * eps.max(1e-3), "module {module}: rel_l2 {} for noise {eps}", c.rel_l2);
        assert!(c.max_abs > 0.0 && c.top1.is_none());
    }

    // top-2 swaps on every 37th row of the logits (28 rows): the model's argmax becomes the fork's runner-up
    let lg = load_module(&dump, MODULE_LOGITS);
    let mut swapped = lg.clone();
    let (mut k, mut max_gap, mut sq, mut sq_ref) = (0usize, 0f32, 0f64, 0f64);
    for &v in &lg {
        sq_ref += (v as f64) * (v as f64);
    }
    for r in (0..TOKENS).step_by(37) {
        let row = &lg[r * VOCAB..(r + 1) * VOCAB];
        let t = argmax(row);
        let s = (0..VOCAB).filter(|&j| j != t).fold(None::<usize>, |best, j| match best {
            Some(b) if row[b] >= row[j] => Some(b),
            _ => Some(j),
        });
        let s = s.unwrap();
        let new = row[t] + 1.0;
        swapped[r * VOCAB + s] = new;
        max_gap = max_gap.max(row[t] - row[s]);
        sq += ((new - row[s]) as f64) * ((new - row[s]) as f64);
        k += 1;
    }
    assert_eq!(k, 28);
    // the logits go through in uneven chunks too
    for (a, b) in [(0usize, 100usize), (100, 612), (612, TOKENS)] {
        feed(&tap, MODULE_LOGITS, &swapped[a * VOCAB..b * VOCAB]).unwrap();
    }
    let r = tap.results();
    let c = r.last().unwrap();
    let t = c.top1.clone().unwrap();
    assert_eq!((t.rows, t.agree_recorded, t.agree_dumped, t.fork_self_agree), (TOKENS, TOKENS - k, TOKENS - k, TOKENS));
    assert_eq!(t.max_gap, max_gap);
    let want = (sq / sq_ref).sqrt();
    assert!((c.rel_l2 - want).abs() <= 1e-9 * want, "logits rel_l2 {} against {want}", c.rel_l2);
    assert!((t.agreement() - (TOKENS - k) as f64 / TOKENS as f64).abs() < 1e-12);
    assert!(t.agreement() >= TOP1_MIN, "28 swaps in 1024 rows stay above the floor");
}

#[test]
fn the_committed_limits_are_what_apply_enforces() {
    let Some(root) = oracle_dir() else { return };
    let dump = Dump::open(&root, 0).unwrap_or_else(|e| panic!("{e}"));
    let tol = Tolerances::committed();
    assert_eq!(tol.limits.len(), MODULES);
    let expect: Vec<Point> = (0..MODULES).map(|m| point_of(m).unwrap()).collect();
    let scale = |tap: &OracleTap, module: usize, rel: f64| {
        let rows = load_module(&dump, module);
        // scaling every element by (1 + rel) is a relative error of exactly `rel`
        let k = (1.0 + rel) as f32;
        let scaled: Vec<f32> = rows.iter().map(|&v| v * k).collect();
        feed(tap, module, &scaled).unwrap();
    };
    // half the limit everywhere (the logits too): passes
    let tap = OracleTap::open(&root, 0).unwrap();
    for m in 0..MODULES {
        scale(&tap, m, 0.5 * tol.limits[&m].limit);
    }
    let v = tol.apply(&tap.results(), &expect);
    assert!(v.passed(), "{:?}", v.failures);
    assert_eq!(v.checked, MODULES);
    // layer 43 at 1.5x its limit fails, by name, and nothing else does
    let tap = OracleTap::open(&root, 0).unwrap();
    for m in 0..MODULES {
        scale(&tap, m, if m == 44 { 1.5 } else { 0.5 } * tol.limits[&m].limit);
    }
    let v = tol.apply(&tap.results(), &expect);
    assert_eq!(v.failures.len(), 1, "{:?}", v.failures);
    assert!(v.failures[0].starts_with("L43 (backbone.layers.43): rel_l2"), "{}", v.failures[0]);
    // a NaN fails, and a point that was never compared is reported
    let tap = OracleTap::open(&root, 0).unwrap();
    let mut rows = load_module(&dump, 1);
    rows[17] = f32::NAN;
    feed(&tap, 1, &rows).unwrap();
    let r = tap.results();
    assert!(r[0].rel_l2.is_nan() && r[0].nonfinite == 1);
    let v = tol.apply(&r, &expect);
    assert!(!v.passed() && v.not_compared.len() == MODULES - 1);
}

#[test]
fn a_missing_or_malformed_dump_is_a_clear_error() {
    let Some(root) = oracle_dir() else { return };
    let tmp = std::env::temp_dir().join(format!("rexl3-omni-oracle-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(tmp.join("layers0")).unwrap();

    // nothing there
    let e = OracleTap::open(&tmp, 0).err().expect("an empty directory is refused");
    assert!(e.contains("top1_0.i32"), "{e}");

    // the whole-sequence files and meta.json are linked in; the module files are not
    let link = |name: &str| std::os::unix::fs::symlink(root.join(name), tmp.join(name)).unwrap();
    for f in ["top1_0.i32", "logits_0.f16", "ids_0.u32"] {
        link(f);
    }
    std::fs::copy(root.join("layers0").join("meta.json"), tmp.join("layers0").join("meta.json")).unwrap();
    let tap = OracleTap::open(&tmp, 0).unwrap_or_else(|e| panic!("{e}"));
    let row = vec![0f32; HIDDEN];
    let e = tap.compare_host(Point::Layer(0), &row, 1, HIDDEN).unwrap_err();
    assert!(e.contains(&module_file(1)) && e.contains("No such file"), "a missing module file: {e}");

    // a file that is not a whole number of rows
    std::fs::write(tmp.join("layers0").join(module_file(1)), [0u8; 100]).unwrap();
    let e = tap.compare_host(Point::Layer(0), &row, 1, HIDDEN).unwrap_err();
    assert!(e.contains(&module_file(1)) && e.contains("not a whole number of rows"), "a short file: {e}");

    // a file of the right shape but fewer rows than asked for
    std::fs::write(tmp.join("layers0").join(module_file(2)), vec![0u8; 2 * HIDDEN * 2]).unwrap();
    let e = tap.compare_host(Point::Layer(1), &vec![0f32; 3 * HIDDEN], 3, HIDDEN).unwrap_err();
    assert!(e.contains(&module_file(2)) && e.contains("past the 2 rows"), "too few rows: {e}");

    // the real dump: a width the dump does not have, and a row past the last token
    let real = OracleTap::open(&root, 0).unwrap();
    let e = real.compare_host(Point::Layer(0), &vec![0f32; 2 * 8192], 2, 8192).unwrap_err();
    assert!(e.contains("8192 wide") && e.contains("is 4096"), "{e}");
    let full = load_module(&Dump::open(&root, 0).unwrap(), 1);
    real.compare_host(Point::Layer(0), &full, TOKENS, HIDDEN).unwrap();
    let e = real.compare_host(Point::Layer(0), &row, 1, HIDDEN).unwrap_err();
    assert!(e.contains("rows 1024..1025 are past the 1024 tokens"), "{e}");
    std::fs::remove_dir_all(&tmp).ok();
}
