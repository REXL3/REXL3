//! The latent-MoE block of Nemotron 3.5 Super VL, layer 1, checked against torch in fp32 on 4 tokens.
//!
//! `testdata/nemotron_omni/moe/layer1_fixture.safetensors` is written by `tools/gen_moe_fixture.py` (run on node 4). It holds the
//! activations and torch's results: router logits, the selection and its weights, fc1's latent, the routed sum, fc2's output, the
//! shared expert, the block's output, two expert spot checks (an EXL3 expert that layer 1 selects, and the bf16 expert 26/111), and
//! fnv1a digests of four decoded trellis weights. It does not hold the weights: at fp16 the router, fc1, fc2, the shared expert and the
//! 82 routed experts are hundreds of MB, far over the fixture's 2 MB cap. Those come from the pack, read through its index, when
//! `REXL3_NEMO_OMNI_DIR` names the pack directory.
//!
//! Without the pack, the test still checks the router's selection (`route`) against torch's logits and indices. With it, the test
//! also checks the logits, fc1, each expert, fc2, the shared expert and the whole block against torch within a relative error of 1e-4.
//! The fixture goes with the pack, chosen by its index: the repaired pack's own (`layer1_fixture_repaired.safetensors`) for the repaired
//! pack and the published Hub pack (the repaired tensors, re-sharded), the original one for any other index; the tolerance is the same.

use gb_exl3::checkpoint::{scan, Checkpoint};
use gb_exl3::{codebook, fnv1a, unpack, Codebook};
use gb_model::families::nemotron_h_omni::moe_ref::{expert_forward, latent_moe, route, Dense, Exl3, Mlp, Proj, Router};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// The ORIGINAL pack's layer-1 fixture (index sha256 4bd072f9...): the one the checks without a pack use.
const FIXTURE: &[u8] = include_bytes!("../../../testdata/nemotron_omni/moe/layer1_fixture.safetensors");
/// The repaired pack's own fixture (`tools/gen_moe_fixture.py`, the same seed and input, run on the repaired pack): the repair
/// re-encoded layer 1's shared up projection, so against the original fixture the shared expert and the block differ by construction.
/// Its router, fc1, fc2, picked experts and digests are the original's (`testdata/nemotron_omni/moe/README.md`).
const REPAIRED_FIXTURE: &[u8] = include_bytes!("../../../testdata/nemotron_omni/moe/layer1_fixture_repaired.safetensors");
/// FNV-1a 64 of the `model.safetensors.index.json` of every pack that holds the repaired tensors, as gb-cuda's
/// `tests/nemotron_moe.rs` (0d18d29b) selects its fixture: the repaired pack (sha256 5c4938c5...) and the published Hub pack
/// (sha256 d2c82b14..., the repaired tensors re-sharded; FNV-1a measured from the file by lanes G2 and G3). Any other index: the
/// original fixture. Since v0.1.1 also the H4 pack (index sha256 eb0f1ada...): the repaired trunk (layer 1 byte-identical), a 4-bit head.
const REPAIRED_INDEX_FNV1A: [(u64, &str); 3] = [
    (0xfae2a114dd50e0e2, "the repaired pack's index (5c4938c5)"),
    (0xc7f15360ccd846e4, "the published Hub pack's index (d2c82b14): the repaired tensors, re-sharded"),
    (0x502dfb7c3ea85676, "the H4 pack's index (eb0f1ada): the repaired trunk, with a 4-bit head"),
];

/// The fixture that goes with the pack in `dir`, chosen by its index (see [`REPAIRED_INDEX_FNV1A`]).
fn fixture_for_pack(dir: &std::path::Path) -> &'static [u8] {
    let index = dir.join("model.safetensors.index.json");
    let bytes = std::fs::read(&index).unwrap_or_else(|e| panic!("{}: {e}", index.display()));
    let h = fnv1a(bytes.iter().copied());
    eprintln!("pack index fnv1a {h:016x}");
    match REPAIRED_INDEX_FNV1A.iter().find(|(f, _)| *f == h) {
        Some((_, which)) => {
            eprintln!("fixture layer1_fixture_repaired.safetensors ({which})");
            REPAIRED_FIXTURE
        }
        None => {
            eprintln!("fixture layer1_fixture.safetensors (not a repaired-tensor index: the original pack's fixture)");
            FIXTURE
        }
    }
}
const TOL: f64 = 1e-4;
const LAYER: usize = 1;
const BF16_LAYER: usize = 26;
const BF16_EXPERT: usize = 111;
const TOKENS: usize = 4;
const TOPK: usize = 22;
const EXPERTS: usize = 512;
const HIDDEN: usize = 4096;
const LATENT: usize = 1024;
const EXPERT_INTER: usize = 2688;
const SCALE: f32 = 5.0;

/// The fixture's tensors, read from the safetensors bytes. Only the fields this test uses are decoded.
struct Fixture {
    tensors: BTreeMap<String, (String, Vec<usize>, &'static [u8])>,
}

impl Fixture {
    fn parse(bytes: &'static [u8]) -> Self {
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Map<String, Value> = serde_json::from_slice(&bytes[8..8 + n]).expect("fixture header");
        let data = &bytes[8 + n..];
        let mut tensors = BTreeMap::new();
        for (name, v) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v["dtype"].as_str().unwrap().to_string();
            let shape = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let off = v["data_offsets"].as_array().unwrap();
            let (s, e) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
            tensors.insert(name, (dtype, shape, &data[s..e]));
        }
        Self { tensors }
    }

    fn get(&self, name: &str, dtype: &str) -> (&[usize], &'static [u8]) {
        let (d, shape, bytes) = self.tensors.get(name).unwrap_or_else(|| panic!("fixture has no tensor {name}"));
        assert_eq!(d, dtype, "{name} is {d}, not {dtype}");
        (shape.as_slice(), *bytes)
    }

    fn f32(&self, name: &str) -> Vec<f32> {
        self.get(name, "F32").1.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn i64(&self, name: &str) -> Vec<i64> {
        self.get(name, "I64").1.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn u64(&self, name: &str) -> u64 {
        let b = self.get(name, "U64").1;
        u64::from_le_bytes(b.try_into().unwrap())
    }
}

/// max|got - want| / max|want| and max|got - want|, over one tensor.
fn errors(got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "length {} against {}", got.len(), want.len());
    let scale = want.iter().fold(0f64, |m, &v| m.max((v as f64).abs()));
    let diff = got.iter().zip(want).fold(0f64, |m, (&g, &w)| m.max(((g - w) as f64).abs()));
    (diff / scale, diff)
}

/// Compares one tensor with torch's, prints its error, and fails above the tolerance.
fn check(name: &str, got: &[f32], want: &[f32]) {
    let (rel, abs) = errors(got, want);
    eprintln!("{name:<34} max rel {rel:.3e}   max abs {abs:.3e}");
    assert!(rel <= TOL, "{name}: max relative error {rel:.3e} is over {TOL:e}");
}

/// The pack's root, when the env var names one; otherwise the pack-backed checks are reported as not run.
fn pack_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set; the pack-backed checks need the weights");
    }
    dir
}

/// The router's selection on torch's logits must be exactly torch's indices, on every token.
#[test]
fn route_picks_the_torch_indices_on_every_token() {
    let fx = Fixture::parse(FIXTURE);
    let logits = fx.f32("router_logits");
    let bias = fx.f32("router_bias");
    let want_idx = fx.i64("route_indices");
    let want_w = fx.f32("route_weights");
    assert_eq!(logits.len(), TOKENS * EXPERTS);
    assert_eq!(want_idx.len(), TOKENS * TOPK);
    let mut agree = 0usize;
    let mut weight_err = 0f64;
    for t in 0..TOKENS {
        let (idx, w) = route(&logits[t * EXPERTS..(t + 1) * EXPERTS], &bias, TOPK, SCALE);
        let got: Vec<i64> = idx.iter().map(|&i| i as i64).collect();
        assert_eq!(got, want_idx[t * TOPK..(t + 1) * TOPK], "token {t}: route picks different experts than torch");
        agree += got.len();
        weight_err = weight_err.max(errors(&w, &want_w[t * TOPK..(t + 1) * TOPK]).0);
    }
    eprintln!(
        "route: {agree}/{} indices agree with torch on {TOKENS}/{TOKENS} tokens; weights max rel {weight_err:.3e}",
        TOKENS * TOPK
    );
    assert!(weight_err <= TOL, "route weights: max relative error {weight_err:.3e} is over {TOL:e}");
}

/// The fixture's internal shapes and its digests' types, so a truncated fixture fails here and not deep in a comparison.
#[test]
fn the_fixture_has_the_measured_shapes() {
    let fx = Fixture::parse(FIXTURE);
    assert_eq!(fx.get("h", "F32").0, &[TOKENS, HIDDEN]);
    assert_eq!(fx.get("router_logits", "F32").0, &[TOKENS, EXPERTS]);
    assert_eq!(fx.get("route_indices", "I64").0, &[TOKENS, TOPK]);
    assert_eq!(fx.get("x_latent", "F32").0, &[TOKENS, LATENT]);
    assert_eq!(fx.get("out", "F32").0, &[TOKENS, HIDDEN]);
    assert_eq!(fx.get("spot_exl3_input", "F32").0, &[1, LATENT]);
    assert_eq!(fx.get("spot_bf16_output", "F32").0, &[TOKENS, LATENT]);
    assert!(FIXTURE.len() <= 2 * 1024 * 1024, "the fixture is over the 2 MB cap");
    assert!(REPAIRED_FIXTURE.len() <= 2 * 1024 * 1024, "the repaired fixture is over the 2 MB cap");
}

/// Loads one EXL3 linear layer by its group prefix, through the checkpoint's index.
fn exl3(ck: &Checkpoint, prefix: &str) -> Exl3 {
    Exl3 { group: ck.load_group(prefix).unwrap_or_else(|e| panic!("{prefix}: {e}")) }
}

/// The router's gate and bias, from the pack's plain tensors.
fn read_plain(ck: &Checkpoint, name: &str) -> Vec<f32> {
    let e = ck.plain.get(name).unwrap_or_else(|| panic!("the pack has no plain tensor {name}"));
    ck.read_floats(e).unwrap_or_else(|err| panic!("{name}: {err}")).into_iter().map(|v| v as f32).collect()
}

/// The fixture's EXL3 digest must equal the digest of the same weight decoded by `gb_exl3::unpack` from the pack's trellis.
fn check_digest(name: &str, ck: &Checkpoint, prefix: &str, want: u64) {
    let g = ck.load_group(prefix).unwrap_or_else(|e| panic!("{prefix}: {e}"));
    // the fixture's projections are 4-bit on the measured pack; the decode the digest checks is the same at every whole width
    assert!(g.group.bits.is_integer(), "{prefix}: a {}-bit projection (the engine serves whole widths)", g.group.bits.as_f64());
    assert_eq!(g.group.codebook, Codebook::Mul1, "{prefix}: the measured projections use the mul1 codebook");
    let wq = unpack(&g.trellis, g.kt, g.nt, g.group.bits, &codebook(g.group.codebook));
    let got = fnv1a(wq.iter().flat_map(|v| v.to_le_bytes()));
    eprintln!("digest {name:<32} {got:016x}");
    assert_eq!(got, want, "{name}: gb_exl3::unpack disagrees with the torch-side decode of the same trellis");
}

/// The full block on the pack's weights, against torch: logits, fc1, the routed experts, fc2, the shared expert, and the output.
#[test]
fn the_block_matches_torch_on_the_pack_weights() {
    let Some(dir) = pack_dir() else { return };
    let fx = Fixture::parse(fixture_for_pack(&dir));
    let ck = scan(&dir, true).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let mp = format!("language_model.backbone.layers.{LAYER}.mixer");

    let h = fx.f32("h");
    let gate = read_plain(&ck, &format!("{mp}.gate.weight"));
    let bias = read_plain(&ck, &format!("{mp}.gate.e_score_correction_bias"));
    assert_eq!(bias, fx.f32("router_bias"), "the pack's selection bias is not the one the fixture was made with");
    let router = Router { gate, bias, hidden: HIDDEN, topk: TOPK, scale: SCALE };

    let fc1 = exl3(&ck, &format!("{mp}.fc1_latent_proj"));
    let fc2 = exl3(&ck, &format!("{mp}.fc2_latent_proj"));
    let shared = Mlp {
        up: Box::new(exl3(&ck, &format!("{mp}.shared_experts.up_proj"))),
        down: Box::new(exl3(&ck, &format!("{mp}.shared_experts.down_proj"))),
    };

    let idx = fx.i64("route_indices");
    let used: BTreeSet<usize> = idx.iter().map(|&e| e as usize).collect();
    let mut experts: Vec<Option<Mlp>> = (0..EXPERTS).map(|_| None).collect();
    for &e in &used {
        experts[e] = Some(Mlp {
            up: Box::new(exl3(&ck, &format!("{mp}.experts.{e}.up_proj"))),
            down: Box::new(exl3(&ck, &format!("{mp}.experts.{e}.down_proj"))),
        });
    }
    eprintln!("loaded {} routed experts for {TOKENS} tokens", used.len());

    let rust_logits = router.logits(&h, TOKENS);
    check("router logits", &rust_logits, &fx.f32("router_logits"));
    let want_idx: Vec<usize> = idx.iter().map(|&e| e as usize).collect();
    for t in 0..TOKENS {
        let (got, _) = route(&rust_logits[t * EXPERTS..(t + 1) * EXPERTS], &router.bias, TOPK, SCALE);
        assert_eq!(
            got,
            want_idx[t * TOPK..(t + 1) * TOPK],
            "token {t}: route on the Rust logits picks different experts than torch"
        );
    }

    check("fc1 latent (x_latent)", &fc1.apply(&h, TOKENS), &fx.f32("x_latent"));
    check("fc2 on the torch routed sum", &fc2.apply(&fx.f32("routed_latent"), TOKENS), &fx.f32("routed_out"));
    check("shared expert", &shared.forward(&h), &fx.f32("shared_out"));

    let out = latent_moe(&h, &router, &fc1, &fc2, &experts, &shared);
    check("block output (out)", &out, &fx.f32("out"));

    let spot_e = fx.i64("spot_exl3_expert")[0] as usize;
    let spot = experts[spot_e].as_ref().expect("the spot expert is one of the selected experts");
    check(
        "EXL3 expert spot (layer 1)",
        &expert_forward(&fx.f32("spot_exl3_input"), spot.up.as_ref(), spot.down.as_ref()),
        &fx.f32("spot_exl3_output"),
    );

    // The bf16 expert: HF [out, in] weights, transposed to [in, out] by `Dense::from_hf`.
    let bp = format!("language_model.backbone.layers.{BF16_LAYER}.mixer.experts.{BF16_EXPERT}");
    let bf16 = |name: &str, out: usize, inp: usize| -> Dense {
        let e = ck.plain.get(name).unwrap_or_else(|| panic!("the pack has no plain tensor {name}"));
        let w: Vec<f32> = ck.read_floats(e).unwrap().into_iter().map(|v| v as f32).collect();
        Dense::from_hf(out, inp, &w)
    };
    let up = bf16(&format!("{bp}.up_proj.weight"), EXPERT_INTER, LATENT);
    let down = bf16(&format!("{bp}.down_proj.weight"), LATENT, EXPERT_INTER);
    check(
        "bf16 expert spot (26, 111)",
        &expert_forward(&fx.f32("spot_bf16_input"), &up as &dyn Proj, &down as &dyn Proj),
        &fx.f32("spot_bf16_output"),
    );

    check_digest("fc1_latent_proj", &ck, &format!("{mp}.fc1_latent_proj"), fx.u64("digest_fc1_latent_proj"));
    check_digest("fc2_latent_proj", &ck, &format!("{mp}.fc2_latent_proj"), fx.u64("digest_fc2_latent_proj"));
    check_digest("expert spot up_proj", &ck, &format!("{mp}.experts.{spot_e}.up_proj"), fx.u64("digest_expert_spot_up_proj"));
    check_digest(
        "expert spot down_proj",
        &ck,
        &format!("{mp}.experts.{spot_e}.down_proj"),
        fx.u64("digest_expert_spot_down_proj"),
    );
}
