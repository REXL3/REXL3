//! Our trellis encoder in the quantizer crate (`gb_quant::exl3_trellis`, `csrc/exl3_trellis.cu`), bit for bit against the
//! quantizer's encoder (`tools/rexl3_quant/trellis`):
//!
//! * `testdata/fixtures/exl3_trellis_encode.txt` (from `tools/rexl3_quant/trellis/gen_encode_fixture.py`): 8 fixed tiles
//!   (Gaussian, a wide one, zeros, a ramp) at every width of the format (1, 1.5, 2, 2.5, 3, 3.5, 4..8 bits), in the
//!   default mode (fp32 costs, window (128, 128)), with fp16 costs, and with the (256, 256) window: states, values,
//!   costs and wrap nodes must equal the file. The file is the NumPy reference's output, and `gen_encode_fixture.py
//!   --gpu` checks the quantizer's GPU build reproduces it;
//! * `REXL3_TRELLIS_FIXTURE=FILE[:FILE...]` adds large fixtures the same script writes from the quantizer's GPU build
//!   (`--large`, e.g. real MiniCPM tiles, which stay out of git); prints "not run" when unset;
//! * round trip: the states pack with `gb_exl3::encode::pack_tile`, read back with `gb_exl3::states`, and
//!   `gb_exl3::unpack` (the serving decoder) reconstructs exactly the encoder's values;
//! * the kernel's history sizing equals the host mirror, and the kernels are native SASS for the device;
//! * a tile's result does not depend on its batch, its place in it, the grid or repetition; the stream API reuses and
//!   grows its scratch.
//!
//!     cargo test --release -p gb-quant --test exl3_trellis -- --nocapture
#![cfg(not(gb_quant_no_cuda))]
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::encode::pack_tile;
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::{codebook, fnv1a, states, tile_positions, unpack, Bits, Codebook};
use gb_quant::exl3_trellis::{self as tr, Costs, EncodeError, Encoded, Encoder, HostEncoded, Options, Window, POSITIONS};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The repository root: `REXL3_REPO_ROOT` when set (a prebuilt test binary run elsewhere), else this checkout.
fn repo_root() -> PathBuf {
    std::env::var_os("REXL3_REPO_ROOT").map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

macro_rules! gpu {
    () => {
        if !device_available() {
            assert!(std::env::var("REXL3_REQUIRE_GPU").as_deref() != Ok("1"), "REXL3_REQUIRE_GPU=1 and no CUDA device");
            eprintln!("no CUDA device: not run");
            return;
        }
    };
}

extern "C" {
    fn cudaGetDevice(dev: *mut i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, dev: i32) -> i32;
}

/// Compute capability of the current device as 10 * major + minor.
fn device_arch() -> i32 {
    let (mut d, mut major, mut minor) = (0, 0, 0);
    unsafe {
        assert_eq!(cudaGetDevice(&mut d), 0);
        assert_eq!(cudaDeviceGetAttribute(&mut major, 75, d), 0);
        assert_eq!(cudaDeviceGetAttribute(&mut minor, 76, d), 0);
    }
    10 * major + minor
}

fn all_bits() -> Vec<Bits> {
    Bits::ALL_TWICE.iter().map(|&t| Bits::from_twice(t).unwrap()).collect()
}

fn opts(costs: Costs, pre: u32, post: u32) -> Options {
    Options { costs, window: Window::new(pre, post).unwrap(), ..Options::default() }
}

// ---- the fixture format (see gen_encode_fixture.py) -------------------------------------------------------------

struct Case {
    twice: u8,
    costs: Costs,
    pre: u32,
    post: u32,
    fields: HashMap<String, String>,
}

struct Fixture {
    path: PathBuf,
    x: Vec<f32>,
    cases: Vec<Case>,
}

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd hex length");
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn read_fixture(path: &Path) -> Fixture {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("exl3_trellis_encode v1"), "{}: not a trellis encode fixture", path.display());
    let (mut x, mut tiles, mut cases, mut cur) = (Vec::new(), 0usize, Vec::new(), None::<Case>);
    for ln in lines {
        let (k, v) = ln.split_once(' ').unwrap_or((ln, ""));
        match k {
            "about" => {}
            "tiles" => tiles = v.parse().unwrap(),
            "x" => x = unhex(v).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "x_file" => {
                let b = std::fs::read(path.with_file_name(v)).unwrap_or_else(|e| panic!("{v}: {e}"));
                x = b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            }
            "case" => {
                let f: Vec<&str> = v.split_whitespace().collect();
                let costs = match f[1] {
                    "fp32" => Costs::Fp32,
                    "fp16" => Costs::Fp16,
                    c => panic!("costs {c}"),
                };
                cur = Some(Case {
                    twice: f[0].parse().unwrap(),
                    costs,
                    pre: f[2].parse().unwrap(),
                    post: f[3].parse().unwrap(),
                    fields: HashMap::new(),
                });
            }
            "end" => cases.push(cur.take().expect("end outside a case")),
            _ => {
                cur.as_mut().unwrap_or_else(|| panic!("{k} outside a case")).fields.insert(k.to_string(), v.to_string());
            }
        }
    }
    assert_eq!(x.len(), tiles * POSITIONS, "{}: tiles and x disagree", path.display());
    Fixture { path: path.to_path_buf(), x, cases }
}

fn fnv_hex(bytes: impl IntoIterator<Item = u8>) -> String {
    format!("{:#018x}", fnv1a(bytes))
}

fn le<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|&x| f(x)).collect()
}

/// Compares one encode with a case; returns the mismatches.
fn check_case(label: &str, c: &Case, got: &HostEncoded) -> Vec<String> {
    let mut bad = Vec::new();
    let st = le(&got.states, u16::to_le_bytes);
    let va = le(&got.values, f32::to_le_bytes);
    let co = le(&got.cost, f32::to_le_bytes);
    let wr = le(&got.wrap, i32::to_le_bytes);
    for (key, bytes) in [("states", &st), ("cost", &co), ("wrap", &wr)] {
        if let Some(want) = c.fields.get(key) {
            let want = unhex(want);
            if want != *bytes {
                let i = (0..want.len().min(bytes.len())).find(|&i| want[i] != bytes[i]).unwrap_or(0);
                bad.push(format!("{label}: {key} differ (first at byte {i} of {})", want.len()));
            }
        }
    }
    for (key, bytes) in [("states_fnv", &st), ("values_fnv", &va), ("cost_fnv", &co), ("wrap_fnv", &wr)] {
        if let Some(want) = c.fields.get(key) {
            let have = fnv_hex(bytes.iter().copied());
            if &have != want {
                bad.push(format!("{label}: {key} {have} != {want}"));
            }
        }
    }
    bad
}

fn run_fixture(fx: &Fixture) -> usize {
    let mut bad = Vec::new();
    for c in &fx.cases {
        let bits = Bits::from_twice(c.twice).unwrap();
        let label = format!("{} {} bits {:?} ({}, {})", fx.path.display(), bits.as_f64(), c.costs, c.pre, c.post);
        let got = tr::encode_host(&fx.x, bits, &opts(c.costs, c.pre, c.post)).unwrap();
        // the values are the codebook's decode of the states, whatever the file stores
        let table = codebook(Codebook::Mul1);
        for (i, (&s, &v)) in got.states.iter().zip(&got.values).enumerate() {
            assert_eq!(v, f16_to_f64(table[s as usize]) as f32, "{label}: value {i} is not the decode of its state");
        }
        bad.extend(check_case(&label, c, &got));
    }
    assert!(bad.is_empty(), "{} mismatches:\n{}", bad.len(), bad.join("\n"));
    fx.cases.len()
}

#[test]
fn committed_fixture_equals_the_quantizer_s_encoder_at_every_width_and_mode() {
    gpu!();
    let fx = read_fixture(&repo_root().join("testdata/fixtures/exl3_trellis_encode.txt"));
    // every width of the format in all three modes
    for bits in all_bits() {
        for (costs, pre) in [(Costs::Fp32, 128), (Costs::Fp16, 128), (Costs::Fp32, 256)] {
            assert!(
                fx.cases.iter().any(|c| c.twice == bits.twice() && c.costs == costs && c.pre == pre),
                "fixture lacks {} bits {costs:?} window {pre}",
                bits.as_f64()
            );
        }
    }
    let n = run_fixture(&fx);
    eprintln!("{n} cases of {} tiles bit-identical", fx.x.len() / POSITIONS);
}

#[test]
fn large_fixtures_are_bit_identical() {
    gpu!();
    let Ok(list) = std::env::var("REXL3_TRELLIS_FIXTURE") else {
        eprintln!("REXL3_TRELLIS_FIXTURE unset: not run");
        return;
    };
    for p in list.split(':').filter(|p| !p.is_empty()) {
        let fx = read_fixture(Path::new(p));
        let n = run_fixture(&fx);
        eprintln!("{p}: {n} cases of {} tiles bit-identical", fx.x.len() / POSITIONS);
    }
}

// ---- round trip through the serving decoder ---------------------------------------------------------------------

fn normal(r: &mut Rng) -> f32 {
    let (a, b) = ((r.next_u64() >> 11) as f64 / (1u64 << 53) as f64, (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64);
    ((-2.0 * a.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
}

fn gaussian_tiles(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed);
    (0..n * POSITIONS).map(|_| normal(&mut r)).collect()
}

#[test]
fn states_pack_and_decode_to_the_encoder_s_values_at_every_width() {
    gpu!();
    let (kt, nt) = (2usize, 3usize);
    let x = gaussian_tiles(kt * nt, 0x007E_1115);
    let table = codebook(Codebook::Mul1);
    let (row, col) = tile_positions();
    for bits in all_bits() {
        let got = tr::encode_host(&x, bits, &Options::default()).unwrap();
        let ends = bits.stream_ends();
        let mut trellis = Vec::with_capacity(kt * nt * bits.tile_words());
        for t in 0..kt * nt {
            let idx: [u16; 256] = got.states[t * 256..(t + 1) * 256].try_into().unwrap();
            let words = pack_tile(&idx, bits);
            assert_eq!(states(&words, bits, &ends), idx, "{} bits: tile {t} states do not survive the packing", bits.as_f64());
            trellis.extend_from_slice(&words);
        }
        let wq = unpack(&trellis, kt, nt, bits, &table);
        for t in 0..kt * nt {
            let (a, c) = (t / nt, t % nt);
            for p in 0..POSITIONS {
                let i = (16 * a + row[p] as usize) * 16 * nt + 16 * c + col[p] as usize;
                let want = f64_to_f16(got.values[t * 256 + p] as f64);
                assert_eq!(
                    wq[i],
                    want,
                    "{} bits: tile {t} position {p}: decoder {:#06x}, encoder {want:#06x}",
                    bits.as_f64(),
                    wq[i]
                );
            }
        }
        // the search's cost is the tile's squared error, up to its truncation (at most 8 mantissa bits a step, so
        // under 256 * 2^-16 = 0.4% in all) and fp32 sums
        for t in 0..kt * nt {
            let e: f64 = (0..256).map(|p| ((got.values[t * 256 + p] - x[t * 256 + p]) as f64).powi(2)).sum();
            assert!(
                (got.cost[t] as f64 - e).abs() <= 1e-2 * e + 1e-6,
                "{} bits tile {t}: cost {} vs error {e}",
                bits.as_f64(),
                got.cost[t]
            );
        }
    }
}

// ---- kernels and sizing -----------------------------------------------------------------------------------------

#[test]
fn kernel_sizing_equals_the_host_mirror_and_the_kernels_are_native() {
    gpu!();
    let arch = device_arch();
    for ka in 1..=8u32 {
        let masks: &[u32] = if ka == 8 { &[0] } else { &[0, tr::HALF_MASK, 0x8888, 0x0101] };
        for &mask in masks {
            for weighted in [false, true] {
                for costs in [Costs::Fp32, Costs::Fp16] {
                    let q = tr::kernel_info_raw(ka, mask, weighted, costs).unwrap();
                    assert_eq!(q.hist_bytes, tr::hist_bytes(ka, mask), "ka={ka} mask={mask:#x}");
                    assert_eq!(q.threads, 512);
                    assert!(q.blocks_per_sm >= 1, "ka={ka} mask={mask:#x}: does not fit an SM");
                    assert_eq!(q.binary_version, arch, "ka={ka}: kernel SASS sm_{} on an sm_{arch} device", q.binary_version);
                }
            }
        }
    }
    assert!(tr::kernel_info_raw(3, 1, false, Costs::Fp32).is_err(), "an odd-popcount mask must be refused");
}

// ---- independence and determinism -------------------------------------------------------------------------------

#[test]
fn tiles_are_independent_and_deterministic() {
    gpu!();
    let n = 300;
    let x = gaussian_tiles(n, 5);
    let mut r = Rng(17);
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        perm.swap(i, (r.next_u64() % (i as u64 + 1)) as usize);
    }
    let xp: Vec<f32> = perm.iter().flat_map(|&t| x[t * 256..(t + 1) * 256].iter().copied()).collect();
    for twice in [4u8, 5, 12] {
        let bits = Bits::from_twice(twice).unwrap();
        let o = Options::default();
        let a = tr::encode_host(&x, bits, &o).unwrap();
        assert_eq!(a, tr::encode_host(&x, bits, &o).unwrap(), "{twice}/2 bits: repeat");
        let p = tr::encode_host(&xp, bits, &o).unwrap();
        for (j, &t) in perm.iter().enumerate() {
            assert_eq!(p.states[j * 256..(j + 1) * 256], a.states[t * 256..(t + 1) * 256], "{twice}/2 bits: permuted tile {t}");
            assert_eq!(p.cost[j].to_bits(), a.cost[t].to_bits());
            assert_eq!(p.wrap[j], a.wrap[t]);
        }
        for t in [0usize, 137, 299] {
            let one = tr::encode_host(&x[t * 256..(t + 1) * 256], bits, &o).unwrap();
            assert_eq!(one.states[..], a.states[t * 256..(t + 1) * 256], "{twice}/2 bits: tile {t} alone");
            assert_eq!(one.values[..], a.values[t * 256..(t + 1) * 256]);
        }
        // a one-byte scratch cap runs a single resident block that walks every tile in turn
        let single = tr::encode_host(&x, bits, &Options { max_scratch_bytes: 1, ..o }).unwrap();
        assert_eq!(single, a, "{twice}/2 bits: one block");
    }
}

#[test]
fn stream_api_reuses_and_grows_its_scratch() {
    gpu!();
    let n = 96;
    let x = gaussian_tiles(n, 23);
    let stream = Stream::new().unwrap();
    let dx = DevBuf::from_slice(&x).unwrap();
    let mut enc = Encoder::new().unwrap();
    // 8 bits needs the least scratch a tile, 1 bit the most: the second call must grow it
    for twice in [16u8, 2, 5, 16] {
        let bits = Bits::from_twice(twice).unwrap();
        let mut out = Encoded::alloc(n).unwrap();
        enc.encode(&dx, n, bits, &Options::default(), &mut out, &stream).unwrap();
        stream.sync().unwrap();
        assert_eq!(out.to_host().unwrap(), tr::encode_host(&x, bits, &Options::default()).unwrap(), "{twice}/2 bits");
    }
    let bits = Bits::from_twice(4).unwrap();
    let mut small = Encoded::alloc(n - 1).unwrap();
    assert!(matches!(enc.encode(&dx, n, bits, &Options::default(), &mut small, &stream), Err(EncodeError::Invalid(_))));
    let mut out = Encoded::alloc(n + 1).unwrap();
    assert!(matches!(enc.encode(&dx, n + 1, bits, &Options::default(), &mut out, &stream), Err(EncodeError::Invalid(_))));
}
