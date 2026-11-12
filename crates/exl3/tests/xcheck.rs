//! Cross-check against TensorFold's `cuda/exl3/format.py` (fixtures from `tools/gen_fixtures.py`).

use gb_exl3::checkpoint::scan;
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::{codebook, fnv1a, forward, unpack, Bits, Codebook};
use serde_json::Value;
use std::path::Path;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn fixture(name: &str) -> Value {
    let p = root().join("testdata/fixtures").join(name);
    serde_json::from_str(&std::fs::read_to_string(p).expect("run tools/gen_fixtures.py")).unwrap()
}

fn u16_fnv(v: &[u16]) -> u64 {
    fnv1a(v.iter().flat_map(|x| x.to_le_bytes()))
}

fn trellis(seed: u64, kt: usize, nt: usize, bits: Bits) -> Vec<i16> {
    let mut r = Rng(seed);
    (0..kt * nt * bits.tile_words()).map(|_| (r.next() & 0xFFFF) as u16 as i16).collect()
}

fn f64s(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let mut r = Rng(seed);
    (0..n).map(|_| lo + (hi - lo) * ((r.next() >> 11) as f64 * (2.0f64).powi(-53))).collect()
}

fn combos() -> Vec<(Codebook, Bits)> {
    let mut v = Vec::new();
    for cb in Codebook::ALL {
        for t in Bits::ALL_TWICE {
            let b = Bits::from_twice(t).unwrap();
            if b.is_integer() || cb == Codebook::Mul1 {
                v.push((cb, b));
            }
        }
    }
    v
}

fn key(cb: Codebook, b: Bits) -> String {
    if b.is_integer() {
        format!("{}:{}", cb.name(), b.twice() / 2)
    } else {
        format!("{}:{:.1}", cb.name(), b.as_f64())
    }
}

#[test]
fn codebooks_bit_exact() {
    let f = fixture("exl3.json");
    for cb in Codebook::ALL {
        assert_eq!(u16_fnv(&codebook(cb)), f["codebook_fnv"][cb.name()].as_u64().unwrap(), "{}", cb.name());
    }
}

#[test]
fn f64_to_f16_matches_numpy() {
    let f = fixture("exl3.json");
    let mut r = Rng(7);
    let mut out = Vec::with_capacity(200000);
    for _ in 0..200000 {
        let mant = (r.next() >> 11) as f64 * (2.0f64).powi(-53);
        let e = (r.next() % 60) as i32 - 30;
        let sign = if r.next() & 1 == 1 { -1.0 } else { 1.0 };
        out.push(f64_to_f16(sign * mant * (2.0f64).powi(e)));
    }
    assert_eq!(u16_fnv(&out), f["f16_convert_fnv"].as_u64().unwrap());
}

#[test]
fn unpack_bit_exact_all_27_combinations() {
    let f = fixture("exl3.json");
    let exp = &f["unpack_kt4_nt3"];
    let combos = combos();
    assert_eq!(combos.len(), 27);
    for (idx, (cb, b)) in combos.into_iter().enumerate() {
        let t = trellis(5000 + idx as u64, 4, 3, b);
        let w = unpack(&t, 4, 3, b, &codebook(cb));
        let k = key(cb, b);
        assert_eq!(u16_fnv(&w), exp[&k].as_u64().unwrap_or_else(|| panic!("fixture lacks {k}")), "{k}");
    }
}

#[test]
fn forward_matches_reference_in_float64() {
    let f = fixture("exl3.json");
    for (idx, e) in f["forward_k128_n256_rows2"].as_array().unwrap().iter().enumerate() {
        let cb = Codebook::from_name(e["codebook"].as_str().unwrap()).unwrap();
        let b = Bits::from_twice((e["bits"].as_f64().unwrap() * 2.0) as u8).unwrap();
        let (kt, nt, rows) = (8usize, 16usize, 2usize);
        let idx = idx as u64;
        let t = trellis(9000 + idx, kt, nt, b);
        let x = f64s(9100 + idx, rows * kt * 16, -1.0, 1.0);
        let q = |v: Vec<f64>| v.into_iter().map(|v| f16_to_f64(f64_to_f16(v))).collect::<Vec<_>>();
        let suh = q(f64s(9200 + idx, kt * 16, 0.5, 1.5));
        let svh = q(f64s(9300 + idx, nt * 16, 0.001, 0.05));
        let bias = e["bias"].as_bool().unwrap().then(|| q(f64s(9400 + idx, nt * 16, -0.1, 0.1)));
        let y = forward(&x, rows, &t, kt, nt, b, &codebook(cb), &suh, &svh, bias.as_deref());
        let want: Vec<f64> = e["y"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = y.iter().zip(&want).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        // float64 sums in a different order than numpy's BLAS matmul: equal to ~1e-15 relative
        assert!(worst <= 1e-12 * scale, "{} {}: worst abs diff {worst:e} vs scale {scale:e}", cb.name(), b.as_f64());
    }
}

#[test]
fn checkpoint_scan_matches_tensorfold() {
    let f = fixture("exl3_ckpt_scan.json");
    let ck = scan(&root().join("testdata/exl3_ckpt"), true).unwrap();
    let groups = f["groups"].as_object().unwrap();
    assert_eq!(ck.groups.len(), groups.len());
    for (p, g) in groups {
        let mine = &ck.groups[p];
        assert_eq!(mine.bits.as_f64(), g["bits"].as_f64().unwrap(), "{p}");
        assert_eq!(mine.codebook.name(), g["codebook"].as_str().unwrap(), "{p}");
        assert_eq!(mine.k as u64, g["k"].as_u64().unwrap());
        assert_eq!(mine.n as u64, g["n"].as_u64().unwrap());
        assert_eq!(mine.in_scales, g["in_scales"].as_str().unwrap());
        assert_eq!(mine.out_scales, g["out_scales"].as_str().unwrap());
        assert_eq!(mine.bias, g["bias"].as_bool().unwrap());
        let files: Vec<&str> = g["files"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(mine.files.iter().map(|s| s.as_str()).collect::<Vec<_>>(), files);
    }
    let plain = f["plain"].as_object().unwrap();
    assert_eq!(ck.plain.len(), plain.len());
    for (n, e) in plain {
        let mine = &ck.plain[n];
        assert_eq!(mine.dtype, e["dtype"].as_str().unwrap());
        let shape: Vec<usize> = e["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert_eq!(mine.shape, shape);
    }
    let bad: Vec<&str> = f["bad"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(ck.bad.keys().map(|s| s.as_str()).collect::<Vec<_>>(), bad);
    for (m, v) in f["markers"].as_object().unwrap() {
        assert_eq!(ck.markers[m] as u64, v.as_u64().unwrap());
    }
}
