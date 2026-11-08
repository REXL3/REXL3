//! The Mamba2 CPU reference against torch. The fixture `testdata/nemotron_omni/ssm/layer0_fixture.safetensors` is made on node 4 by
//! `tools/gen_ssm_fixture.py`, which runs the fork's torch math over 16 tokens of a fixed-seed input with layer 0's bf16 parameters
//! from the pack. This test regenerates the same inputs (splitmix64, see the generator), runs the Rust reference, and compares.
//!
//! The gate is a scale-normalized relative error: max |rust - torch| / max |torch| <= 1e-4 per tensor. Element-wise relative error
//! is reported as well, with a floor of 1e-3 * max |torch|, because near-zero outputs make a pure element-wise ratio meaningless.

use gb_model::families::nemotron_h_omni::ssm_ref::{
    conv_step, discretize, gated_norm, prefill_scan, recurrent_step, split_xbc, Dims,
};
use serde_json::Value;

const TOKENS: usize = 16;
const SEED: u64 = 20261010;
const EPS: f32 = 1e-5;
const MAX_FIXTURE_BYTES: usize = 1 << 20;
const REL_TOL: f32 = 1e-4;
const STEP_TOL: f32 = 1e-6;
const D: Dims = Dims::OMNI;

struct Fixture {
    bytes: Vec<u8>,
    header: serde_json::Map<String, Value>,
    data_start: usize,
}

impl Fixture {
    fn load() -> Self {
        let path = repo_root().join("testdata/nemotron_omni/ssm/layer0_fixture.safetensors");
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(bytes.len() <= MAX_FIXTURE_BYTES, "the fixture is {} bytes, over 1 MiB", bytes.len());
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Map<String, Value> = serde_json::from_slice(&bytes[8..8 + n]).expect("safetensors header");
        Fixture { bytes, header, data_start: 8 + n }
    }

    fn raw(&self, name: &str, dtype: &str, shape: &[usize]) -> &[u8] {
        let e = self.header.get(name).unwrap_or_else(|| panic!("fixture has no tensor `{name}`"));
        assert_eq!(e["dtype"], dtype, "{name} dtype");
        let got: Vec<usize> = e["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert_eq!(got, shape, "{name} shape");
        let off = e["data_offsets"].as_array().unwrap();
        let (s, t) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
        &self.bytes[self.data_start + s..self.data_start + t]
    }

    fn f32(&self, name: &str, shape: &[usize]) -> Vec<f32> {
        self.raw(name, "F32", shape).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn bf16(&self, name: &str, shape: &[usize]) -> Vec<f32> {
        self.raw(name, "BF16", shape)
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    }

    fn i64(&self, name: &str, shape: &[usize]) -> Vec<i64> {
        self.raw(name, "I64", shape).chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
    }
}

fn splitmix64(z: u64) -> u64 {
    let z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The generator of `tools/gen_ssm_fixture.py`: element `t * cols + j` of stream `stream` is `k / 2^23 - 1`, where `k` is the top 24
/// bits of `splitmix64(SEED + stream << 48 + index)`. Returns the values and the sum of the `k`s.
fn gen_stream(stream: u64, rows: usize, cols: usize) -> (Vec<f32>, i64) {
    let mut out = Vec::with_capacity(rows * cols);
    let mut sum = 0i64;
    for idx in 0..(rows * cols) as u64 {
        let k = splitmix64(SEED.wrapping_add(stream << 48).wrapping_add(idx)) >> 40;
        sum += k as i64;
        out.push((k as f32 - 8_388_608.0) / 8_388_608.0);
    }
    (out, sum)
}

#[derive(Debug, Clone, Copy)]
struct Errors {
    max_abs: f32,
    scale_rel: f32,
    elem_rel: f32,
}

fn errors(got: &[f32], want: &[f32]) -> Errors {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let floor = 1e-3 * scale;
    let mut e = Errors { max_abs: 0.0, scale_rel: 0.0, elem_rel: 0.0 };
    for (a, b) in got.iter().zip(want) {
        let d = (a - b).abs();
        e.max_abs = e.max_abs.max(d);
        e.elem_rel = e.elem_rel.max(d / b.abs().max(floor));
    }
    e.scale_rel = e.max_abs / scale;
    e
}

fn report(name: &str, e: Errors) {
    eprintln!(
        "{name}: max abs {:.3e}, max rel (over max |torch|) {:.3e}, max elementwise rel (floor 1e-3 * max) {:.3e}",
        e.max_abs, e.scale_rel, e.elem_rel
    );
}

/// The inputs of the SSM after conv and discretization, one row per token, plus the parameters.
struct Case {
    z: Vec<f32>,
    xs: Vec<f32>,
    bs: Vec<f32>,
    cs: Vec<f32>,
    dts: Vec<f32>,
    gs: Vec<f32>,
    d: Vec<f32>,
    norm_w: Vec<f32>,
}

fn build_case(fx: &Fixture) -> Case {
    let (inner, cc, h) = (D.inner(), D.conv_channels(), D.heads);
    let conv_w = fx.bf16("conv1d.weight", &[cc, 1, 4]);
    let conv_b = fx.bf16("conv1d.bias", &[cc]);
    let a_log = fx.bf16("A_log", &[h]);
    let d = fx.bf16("D", &[h]);
    let dt_bias = fx.bf16("dt_bias", &[h]);
    let norm_w = fx.bf16("norm.weight", &[inner]);

    let (z, sum_z) = gen_stream(0, TOKENS, inner);
    let (xbc, sum_xbc) = gen_stream(1, TOKENS, cc);
    let (dt_raw, sum_dt) = gen_stream(2, TOKENS, h);
    let checksum = fx.i64("input_checksum", &[3]);
    assert_eq!(checksum, vec![sum_z, sum_xbc, sum_dt], "the Rust input generator drifted from the torch generator");

    let mut history = vec![0.0f32; 3 * cc];
    let mut conv_all = vec![0.0f32; TOKENS * cc];
    for t in 0..TOKENS {
        conv_step(&mut history, &conv_w, &conv_b, &xbc[t * cc..(t + 1) * cc], &mut conv_all[t * cc..(t + 1) * cc]);
    }
    let last = (TOKENS - 1) * cc;
    let e_conv = errors(&conv_all[last..], &fx.f32("conv_last", &[cc]));
    report("conv_last (silu conv, token 15)", e_conv);
    assert!(e_conv.scale_rel <= REL_TOL, "conv output");

    let (mut xs, mut bs, mut cs) = (Vec::new(), Vec::new(), Vec::new());
    let (mut dts, mut gs) = (vec![0.0f32; TOKENS * h], vec![0.0f32; TOKENS * h]);
    for t in 0..TOKENS {
        let (x, b, c) = split_xbc(D, &conv_all[t * cc..(t + 1) * cc]);
        xs.extend_from_slice(x);
        bs.extend_from_slice(b);
        cs.extend_from_slice(c);
        for hd in 0..h {
            let (dt, g) = discretize(dt_raw[t * h + hd], dt_bias[hd], a_log[hd]);
            dts[t * h + hd] = dt;
            gs[t * h + hd] = g;
        }
    }
    let e_dt = errors(&dts[(TOKENS - 1) * h..], &fx.f32("dt_last", &[h]));
    let e_g = errors(&gs[(TOKENS - 1) * h..], &fx.f32("g_last", &[h]));
    report("dt_last (softplus, clamp)", e_dt);
    report("g_last (-exp(A_log) dt)", e_g);
    assert!(e_dt.scale_rel <= REL_TOL && e_g.scale_rel <= REL_TOL, "discretization");

    Case { z, xs, bs, cs, dts, gs, d, norm_w }
}

#[test]
fn the_fixture_is_committed_at_its_size_and_its_inputs_regenerate() {
    let fx = Fixture::load();
    assert!(fx.header.contains_key("input_checksum"));
    let _ = build_case(&fx);
}

#[test]
fn the_layer0_ssm_matches_torch_fp32_within_1e4() {
    let fx = Fixture::load();
    let case = build_case(&fx);
    let inner = D.inner();

    let mut state = vec![0.0f32; D.state_len()];
    let mut ys = vec![0.0f32; TOKENS * inner];
    prefill_scan(D, &mut state, TOKENS, &case.xs, &case.bs, &case.cs, &case.dts, &case.gs, &case.d, &mut ys);

    let per_head = D.state * D.head_dim;
    let state_ref = fx.f32("state_ref_heads", &[2, D.state, D.head_dim]);
    let e_state = errors(&state[..2 * per_head], &state_ref);
    report("final state, heads 0-1", e_state);
    assert!(e_state.scale_rel <= REL_TOL, "state");

    let mut gated = vec![0.0f32; TOKENS * inner];
    for t in 0..TOKENS {
        let r = t * inner..(t + 1) * inner;
        gated_norm(D, &ys[r.clone()], &case.z[r.clone()], &case.norm_w, EPS, &mut gated[r]);
    }
    let want = fx.f32("gated_ref", &[TOKENS, inner]);
    let e = errors(&gated, &want);
    report("gated output y (16 x 8192, before out_proj)", e);
    assert!(e.scale_rel <= REL_TOL, "output: max relative error {:.3e} > {REL_TOL:e}", e.scale_rel);
}

#[test]
fn prefill_scan_equals_the_step_loop_within_1e6() {
    let fx = Fixture::load();
    let case = build_case(&fx);
    let (inner, h) = (D.inner(), D.heads);

    let mut scan_state = vec![0.0f32; D.state_len()];
    let mut scan_y = vec![0.0f32; TOKENS * inner];
    prefill_scan(D, &mut scan_state, TOKENS, &case.xs, &case.bs, &case.cs, &case.dts, &case.gs, &case.d, &mut scan_y);

    let mut loop_state = vec![0.0f32; D.state_len()];
    let mut loop_y = vec![0.0f32; TOKENS * inner];
    for t in 0..TOKENS {
        recurrent_step(
            D,
            &mut loop_state,
            &case.xs[t * inner..(t + 1) * inner],
            &case.bs[t * D.bc()..(t + 1) * D.bc()],
            &case.cs[t * D.bc()..(t + 1) * D.bc()],
            &case.dts[t * h..(t + 1) * h],
            &case.gs[t * h..(t + 1) * h],
            &case.d,
            &mut loop_y[t * inner..(t + 1) * inner],
        );
    }
    let ey = errors(&scan_y, &loop_y);
    let es = errors(&scan_state, &loop_state);
    report("prefill_scan vs step loop, y", ey);
    report("prefill_scan vs step loop, state", es);
    assert!(ey.max_abs <= STEP_TOL && es.max_abs <= STEP_TOL, "prefill_scan must equal the step loop");
}

/// The repository root: `REXL3_REPO_ROOT` when set (a prebuilt test binary run from an installed bin directory that carries the
/// tree's `testdata/`, e.g. on a node without this checkout), else the checkout this binary was built from.
fn repo_root() -> std::path::PathBuf {
    std::env::var_os("REXL3_REPO_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}
