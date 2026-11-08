//! Cross-check against TensorFold's Python reference (`tools/gen_fixtures.py` wrote the fixture).
//! Inputs are regenerated here from the same splitmix64 stream; `inputs_fnv` proves they match.

use gb_sampling::{choose, seed_for, uniform, Sampling};
use serde_json::Value;

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

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn fixture() -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/fixtures/sampling.json");
    serde_json::from_str(&std::fs::read_to_string(p).expect("run tools/gen_fixtures.py")).unwrap()
}

const TEMPS: [f64; 5] = [0.0, 0.3, 0.6, 1.0, 1.5];
const TOPK: [usize; 4] = [0, 1, 5, 20];
const TOPP: [f64; 5] = [1.0, 0.95, 0.8, 0.5, 0.0];
const MINP: [f64; 4] = [0.0, 0.0, 0.05, 0.2];

fn case(i: u64) -> (Vec<f32>, Vec<u32>, u64, Sampling) {
    let mut r = Rng(1000 + i);
    let n = 1 + r.next() % 28;
    let mode = r.next() % 3;
    let mut vals = Vec::new();
    for _ in 0..n {
        let x = r.next();
        let base = ((x >> 40) as f32) / 16777216.0f32 * 16.0f32 - 8.0f32;
        vals.push(match mode {
            0 => base,
            1 => (x >> 61) as f32,
            _ => base * 3.0f32,
        });
    }
    let ids: Vec<u32> = (0..n).map(|j| (j * 5003 + r.next() % 5003) as u32).collect();
    let temperature = TEMPS[(r.next() % 5) as usize];
    let top_k = TOPK[(r.next() % 4) as usize];
    let top_p = TOPP[(r.next() % 5) as usize];
    let min_p = MINP[(r.next() % 4) as usize];
    let seed = r.next() >> 1;
    let position = r.next() % 300000;
    (vals, ids, position, Sampling { seed, temperature, top_k, top_p, min_p })
}

#[test]
fn uniform_bits_match_tensorfold() {
    let f = fixture();
    let mut bytes = Vec::new();
    for seed in [0u64, 1, (1u64 << 63) - 1, 123456789] {
        for pos in 0..32u64 {
            for id in 0..32u64 {
                bytes.extend_from_slice(&uniform(seed, pos, id).to_le_bytes());
            }
        }
    }
    assert_eq!(fnv(&bytes), f["uniform_fnv"].as_u64().unwrap());
}

#[test]
fn seed_for_matches_tensorfold() {
    let f = fixture();
    for e in f["seed_for"].as_array().unwrap() {
        let tokens: Vec<u32> = e["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
        assert_eq!(seed_for(&tokens, e["salt"].as_u64().unwrap()), e["seed"].as_u64().unwrap());
    }
}

#[test]
fn choose_matches_tensorfold() {
    let f = fixture();
    let n = f["cases"].as_u64().unwrap();
    let expected = f["choose"].as_array().unwrap();
    let mut inp = Vec::new();
    let mut mismatches = Vec::new();
    let mut greedy_ties = 0u32;
    for i in 0..n {
        let (vals, ids, pos, s) = case(i);
        for v in &vals {
            inp.extend_from_slice(&v.to_le_bytes());
        }
        for id in &ids {
            inp.extend_from_slice(&id.to_le_bytes());
        }
        inp.extend_from_slice(&pos.to_le_bytes());
        inp.extend_from_slice(&s.seed.to_le_bytes());
        let got = choose(&vals, &ids, pos, &s);
        let top = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let tied: Vec<u32> = ids.iter().zip(&vals).filter(|(_, &v)| v == top).map(|(&id, _)| id).collect();
        if s.temperature == 0.0 && tied.len() > 1 {
            // Deliberate difference from the Python reference: true greedy takes the lowest id among exact
            // ties, where TensorFold clamps T to 1e-6 and lets the noise pick. Everywhere else they agree.
            assert_eq!(got, *tied.iter().min().unwrap(), "case {i}: greedy must take the lowest tied id");
            greedy_ties += 1;
            continue;
        }
        if got as u64 != expected[i as usize].as_u64().unwrap() {
            mismatches.push(i);
        }
    }
    assert_eq!(fnv(&inp), f["inputs_fnv"].as_u64().unwrap(), "case generation differs from the Python side");
    assert!(
        mismatches.is_empty(),
        "{} of {n} cases differ from TensorFold: {:?}",
        mismatches.len(),
        &mismatches[..mismatches.len().min(20)]
    );
    eprintln!("{greedy_ties} temperature-0 cases with exact ties checked against lowest-id greedy instead of the fixture");
    assert!(greedy_ties > 0, "the fixture should contain temperature-0 ties, or this branch is untested");
}
