//! GPU microbenchmark: classic vs reworked (pipelined) FP8 small-M GEMM/pair kernels and the grouped `wo_a`
//! projection, on the real DeepSeek-V4.1-Flash decode shapes, rows 1..8.
//!
//! Run on a CUDA box with:
//! ```text
//! cargo +1.89.0 test --release -p gb-cuda --test dsv41_fp8_woa_bench -- --nocapture
//! ```
//! Warm + timed iterations via `cudaEventElapsedTime`; prints one line per shape/path.

use gb_cuda::dsv4_attn::{Dsv4Config, MlaDsv4};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::fp8_linear::Fp8Weights;
use gb_cuda::rt::timed_events;
use gb_cuda::{device_available, DevBuf, Stream};

// ------------------------------------------------------------------------------------------------ rng

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * u
    }
    fn e4m3(&mut self) -> u8 {
        let b = (self.next_u64() & 0x7F) as u8;
        if b == 0x7F {
            0x3C
        } else {
            b
        }
    }
    fn e8m0(&mut self) -> u8 {
        (0x79 + (self.next_u64() % 13) as u8) & 0x7F
    }
}

fn build_case(rng: &mut Rng, m: usize, n: usize, k: usize) -> (Vec<u16>, Vec<u8>, Vec<u8>) {
    // x as f16 bits (bf16-style truncation of uniform f32 is fine for traffic shape).
    let x: Vec<u16> = (0..m * k).map(|_| (rng.uniform(-1.0, 1.0).to_bits() >> 16) as u16).collect();
    let w: Vec<u8> = (0..n * k).map(|_| rng.e4m3()).collect();
    let s: Vec<u8> = (0..(n / 32) * (k / 32)).map(|_| rng.e8m0()).collect();
    (x, w, s)
}

// ------------------------------------------------------------------------------------------------ timing

fn time_it(st: &Stream, warmup: usize, iters: usize, mut f: impl FnMut()) -> f32 {
    for _ in 0..warmup {
        f();
    }
    unsafe {
        let mut e0 = std::ptr::null_mut();
        let mut e1 = std::ptr::null_mut();
        assert_eq!(timed_events::create(&mut e0), 0, "event create");
        assert_eq!(timed_events::create(&mut e1), 0, "event create");
        assert_eq!(timed_events::record(e0, st.raw()), 0, "record e0");
        for _ in 0..iters {
            f();
        }
        assert_eq!(timed_events::record(e1, st.raw()), 0, "record e1");
        assert_eq!(timed_events::synchronize(e1), 0, "sync e1");
        let mut ms = 0.0f32;
        assert_eq!(timed_events::elapsed_ms(&mut ms, e0, e1), 0, "elapsed");
        ms / iters as f32
    }
}

fn report(name: &str, m: usize, path: &str, ms: f32, bytes: f64) {
    let gbs = bytes / (ms as f64 * 1e-3) / 1e9;
    println!("{name} m={m} {path}: {ms:.4} ms/call  {gbs:.1} GB/s");
}

// ------------------------------------------------------------------------------------------------ benches

fn bench_single(name: &str, rng: &mut Rng, m: usize, n: usize, k: usize, st: &Stream) {
    let (x16, w, s) = build_case(rng, 8, n, k);
    let wt = Fp8Weights::new(n, k, &w, &s).unwrap();
    let x = DevBuf::from_slice(&x16).unwrap();
    let mut y = DevBuf::zeroed(m * n * 2).unwrap();
    let bytes = (n * k + m * k * 2) as f64;

    let ms_c = time_it(st, 10, 200, || wt.forward_classic(&x, &mut y, m, Dtype::F16, st).unwrap());
    report(name, m, "classic", ms_c, bytes);
    let ms_n = time_it(st, 10, 200, || wt.forward(&x, &mut y, m, Dtype::F16, st).unwrap());
    report(name, m, "new", ms_n, bytes);
}

fn bench_pair(name: &str, rng: &mut Rng, m: usize, n1: usize, n2: usize, k: usize, st: &Stream) {
    let (x16, w1, s1) = build_case(rng, 8, n1, k);
    let (_, w2, s2) = build_case(rng, 8, n2, k);
    let wa = Fp8Weights::new(n1, k, &w1, &s1).unwrap();
    let wb = Fp8Weights::new(n2, k, &w2, &s2).unwrap();
    let x = DevBuf::from_slice(&x16).unwrap();
    let mut y1 = DevBuf::zeroed(m * n1 * 2).unwrap();
    let mut y2 = DevBuf::zeroed(m * n2 * 2).unwrap();
    let bytes = ((n1 + n2) * k + m * k * 2) as f64;

    let ms_c =
        time_it(st, 10, 200, || Fp8Weights::forward_pair_classic(&wa, &mut y1, &wb, &mut y2, &x, m, Dtype::F16, st).unwrap());
    report(name, m, "classic", ms_c, bytes);
    let ms_n = time_it(st, 10, 200, || Fp8Weights::forward_pair(&wa, &mut y1, &wb, &mut y2, &x, m, Dtype::F16, st).unwrap());
    report(name, m, "new", ms_n, bytes);
}

fn bench_woa(name: &str, rng: &mut Rng, rows: usize, st: &Stream) {
    let cfg = Dsv4Config::v41();
    let (ol, oin, groups) = (cfg.o_lora, cfg.o_in_per_group(), cfg.o_groups);
    let hd = groups * oin;
    let x16: Vec<u16> = (0..rows * hd).map(|_| (rng.uniform(-1.0, 1.0).to_bits() >> 16) as u16).collect();
    let w: Vec<u8> = (0..groups * ol * oin).map(|_| rng.e4m3()).collect();
    let s: Vec<u8> = (0..groups * (ol / 32) * (oin / 32)).map(|_| rng.e8m0()).collect();
    let od = DevBuf::from_slice(&x16).unwrap();
    let wd = DevBuf::from_slice(&w).unwrap();
    let sd = DevBuf::from_slice(&s).unwrap();
    let mut out = DevBuf::zeroed(rows * groups * ol * 2).unwrap();
    let bytes = (groups * ol * oin + rows * groups * oin * 2 + rows * groups * ol * 2) as f64;

    let ms_c = time_it(st, 10, 50, || {
        MlaDsv4::woa_grouped_ab(&od, &wd, &sd, &mut out, rows, groups, ol, oin, Dtype::F16, true, st).unwrap()
    });
    report(name, rows, "classic", ms_c, bytes);
    let ms_n = time_it(st, 10, 50, || {
        MlaDsv4::woa_grouped_ab(&od, &wd, &sd, &mut out, rows, groups, ol, oin, Dtype::F16, false, st).unwrap()
    });
    report(name, rows, "new", ms_n, bytes);
}

#[test]
fn fp8_small_m_woa_bench() {
    let Some(st) = (|| {
        if device_available() {
            Some(Stream::new().unwrap())
        } else {
            eprintln!("not run: no device");
            None
        }
    })() else {
        return;
    };
    let mut rng = Rng(0xBEEF_CAFE_5EED_1234);

    println!("== fp8 small-M GEMM: classic vs pipelined new (warm + timed) ==");
    for m in [1usize, 2, 4, 6, 8] {
        bench_single("wq_b", &mut rng, m, 16384, 1280, &st);
        bench_single("wo_b", &mut rng, m, 5120, 8192, &st);
        bench_single("w2", &mut rng, m, 5120, 2304, &st);
        bench_pair("wq_a+wkv", &mut rng, m, 1280, 512, 5120, &st);
        bench_pair("w1+w3", &mut rng, m, 2304, 2304, 5120, &st);
        bench_woa("wo_a", &mut rng, m, &st);
    }
}

#[cfg(test)]
mod perf_probe_ffi_contract {
    //! Check the raw declarations without a CUDA device or kernel launch.

    const PROBE: &str = include_str!("dsv4_perf_probe.rs");
    const CUDA: &str = include_str!("../csrc/dsv4_attn.cu");
    const RUST_WOA: &str = "fn gb_dsv4_woa_grouped(";
    const CUDA_WOA: &str = "GB_EXPORT int gb_dsv4_woa_grouped(";

    fn declarations<'a>(source: &'a str, marker: &str) -> Vec<&'a str> {
        let mut declarations = Vec::new();
        let mut rest = source;
        while let Some(start) = rest.find(marker) {
            let after_open = &rest[start + marker.len()..];
            let mut depth = 1usize;
            let mut close = None;
            for (offset, character) in after_open.char_indices() {
                match character {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            close = Some(offset);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let close = close.expect("unterminated FFI declaration");
            declarations.push(&after_open[..close]);
            rest = &after_open[close + 1..];
        }
        declarations
    }

    fn rust_type_classes(declaration: &str) -> Vec<&str> {
        declaration
            .split(',')
            .filter(|parameter| !parameter.trim().is_empty())
            .map(|parameter| {
                let rust_type = parameter.split_once(':').expect("Rust FFI parameter needs a name and type").1.trim();
                if rust_type.starts_with('*') {
                    "ptr"
                } else {
                    rust_type
                }
            })
            .collect()
    }

    fn cuda_type_classes(declaration: &str) -> Vec<&str> {
        declaration
            .split(',')
            .filter(|parameter| !parameter.trim().is_empty())
            .map(|parameter| if parameter.contains('*') { "ptr" } else { "i32" })
            .collect()
    }

    fn c_woa_parameters() -> &'static str {
        let declarations = declarations(CUDA, CUDA_WOA);
        assert_eq!(declarations.len(), 1, "expected one C wo_a export");
        declarations[0]
    }

    #[test]
    fn probe_woa_declarations_match_the_c_export() {
        let expected = cuda_type_classes(c_woa_parameters());
        assert_eq!(expected, ["ptr", "ptr", "ptr", "ptr", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "ptr"]);

        let rust_declarations = declarations(PROBE, RUST_WOA);
        assert_eq!(rust_declarations.len(), 2, "both raw probe declarations must be checked");
        for declaration in rust_declarations {
            assert_eq!(rust_type_classes(declaration), expected);
        }
    }

    #[test]
    fn abi_check_rejects_the_old_eleven_parameter_declaration() {
        let old_declaration = "o: *const core::ffi::c_void, w: *const u8, s: *const u8, \
                               out: *mut core::ffi::c_void, rows: i32, groups: i32, o_lora: i32, \
                               o_in: i32, dtype: i32, use_classic: i32, stream: *mut core::ffi::c_void";
        let old_types = rust_type_classes(old_declaration);
        assert_eq!(old_types.len(), 11);
        assert_ne!(old_types, cuda_type_classes(c_woa_parameters()));
    }
}
